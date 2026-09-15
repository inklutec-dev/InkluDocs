"""Export-Abnahme (15.09.2026): Jede exportierte PDF wird NACH dem Schreiben unabhaengig gemessen,
mit einem anderen Werkzeug (pikepdf) als dem, das geschrieben hat (PyMuPDF bzw. PDFix).

Anlass: Am 14.09.2026 haette der Ersatzweg (fitz) ein Kundendokument mit 234 Alt-Texten ausgeliefert,
die im Strukturbaum unerreichbar waren und vom Abschluss-Schritt entfernt wurden — die Oberflaeche
haette „234 getaggt“ gemeldet. Die Abnahme prueft, was WIRKLICH in der Datei steht, und gilt fuer
beide Wege (PDFix und Ersatzweg) gleich. Steve: „Der Kunde soll auf beiden Wegen sauberen Service
bekommen.“

Regeln (jede Verletzung = Befund, Abnahme nicht bestanden):
  1. Die Export-Datei laesst sich oeffnen und hat so viele Seiten wie das Original.
  2. Wurden Alt-Texte geschrieben, gibt es einen Strukturbaum (StructTreeRoot).
  3. Kein Figure-Element mit Alt-Text liegt ausserhalb des Strukturbaums (keine Waisen).
  4. Auf keiner Seite sind BDC/BMC- und EMC-Marker unbalanciert.
  5. Mindestens so viele der geschriebenen Texte stehen erreichbar in der Datei, wie der
     Export als „getaggt“ meldet (die Zaehlung im Dialog sagt die Wahrheit).

Die Abnahme aendert die Datei nie. Sie darf den Export nicht scheitern lassen (Aufrufer faengt
Ausnahmen); ihr Ergebnis geht als Warnung in den Export-Dialog und als Logzeile
„EXPORT-ABNAHME ...“ an Nexus.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional

import pikepdf

PREFIX = 60  # Zeichen, ueber die ein geschriebener Text mit dem /Alt in der Datei verglichen wird
MAX_SEITEN_IM_BEFUND = 10


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _figures_erreichbar(pdf: pikepdf.Pdf) -> tuple[int, int, list[str]]:
    """Zaehlt Figure-Elemente, die vom StructTreeRoot aus ueber /K erreichbar sind.
    Rueckgabe: (figures gesamt, figures mit /Alt, Liste der Alt-Texte)."""
    root = pdf.Root.get("/StructTreeRoot")
    if root is None:
        return 0, 0, []
    gesehen: set = set()
    figures = 0
    mit_alt = 0
    alts: list[str] = []
    stapel = [root]
    while stapel:
        el = stapel.pop()
        if isinstance(el, pikepdf.Dictionary):
            og = el.objgen
            if og != (0, 0):
                if og in gesehen:
                    continue
                gesehen.add(og)
            if str(el.get("/S", "")) == "/Figure":
                figures += 1
                if "/Alt" in el:
                    mit_alt += 1
                    alts.append(_norm(str(el.Alt)))
            k = el.get("/K")
            if k is not None:
                stapel.append(k)
        elif isinstance(el, pikepdf.Array):
            for x in el:
                stapel.append(x)
    return figures, mit_alt, alts


def _figures_mit_alt_gesamt(pdf: pikepdf.Pdf) -> int:
    """Alle Figure-Objekte mit /Alt in der Datei — auch unerreichbare (Waisen)."""
    n = 0
    for o in pdf.objects:
        if isinstance(o, pikepdf.Dictionary) and str(o.get("/S", "")) == "/Figure" and "/Alt" in o:
            n += 1
    return n


def _unbalancierte_seiten(pdf: pikepdf.Pdf) -> list:
    """Seiten (1-basiert), auf denen die Marker BDC/BMC und EMC nicht aufgehen."""
    befund = []
    for nr, seite in enumerate(pdf.pages, start=1):
        try:
            ops = pikepdf.parse_content_stream(seite)
        except Exception:
            befund.append(f"{nr} (Inhaltsstrom nicht lesbar)")
            continue
        tiefe = 0
        negativ = False
        for _operanden, op in ops:
            o = str(op)
            if o in ("BDC", "BMC"):
                tiefe += 1
            elif o == "EMC":
                tiefe -= 1
                if tiefe < 0:
                    negativ = True
        if tiefe != 0 or negativ:
            befund.append(nr)
    return befund


def _seitenzahl(pfad: Optional[str]) -> Optional[int]:
    if not pfad or not os.path.isfile(pfad):
        return None
    try:
        with pikepdf.open(pfad) as p:
            return len(p.pages)
    except Exception:
        return None


def abnahme_pdf(export_pfad: str, original_pfad: Optional[str],
                geschriebene_texte: Iterable[str],
                erwartet_getaggt: Optional[int] = None) -> dict:
    """Misst die exportierte Datei. Rueckgabe:
    {"ok": bool, "befunde": [str], "kennzahlen": {...}}.
    geschriebene_texte: die Alt-Texte, die der Export schreiben sollte (ohne "" und ohne "dekorativ").
    erwartet_getaggt: was der Export als geschrieben meldet; None = alle geschriebenen Texte."""
    texte = [_norm(t) for t in geschriebene_texte if t and _norm(t) and _norm(t) != "dekorativ"]
    erwartet = len(texte) if erwartet_getaggt is None else max(0, min(int(erwartet_getaggt), len(texte)))
    befunde: list[str] = []
    kz: dict = {"texte_geschrieben": len(texte), "erwartet_getaggt": erwartet}

    try:
        pdf = pikepdf.open(export_pfad)
    except Exception as e:
        return {"ok": False, "befunde": [f"Export-Datei nicht lesbar: {e}"], "kennzahlen": kz}

    with pdf:
        kz["seiten"] = len(pdf.pages)
        kz["seiten_original"] = _seitenzahl(original_pfad)
        if kz["seiten_original"] is not None and kz["seiten"] != kz["seiten_original"]:
            befunde.append(f"Seitenzahl {kz['seiten']} statt {kz['seiten_original']} wie im Original")

        hat_baum = pdf.Root.get("/StructTreeRoot") is not None
        kz["strukturbaum"] = hat_baum
        if texte and erwartet > 0 and not hat_baum:
            befunde.append("Kein Strukturbaum in der Datei, obwohl Alt-Texte geschrieben wurden")

        figures, mit_alt, alts = _figures_erreichbar(pdf)
        gesamt = _figures_mit_alt_gesamt(pdf)
        kz["figures"] = figures
        kz["figures_mit_alt"] = mit_alt
        kz["figures_mit_alt_objekte"] = gesamt
        waisen = gesamt - mit_alt
        kz["waisen"] = waisen
        if waisen > 0:
            befunde.append(f"{waisen} Bild-Element(e) mit Alt-Text ausserhalb des Strukturbaums (unerreichbar)")

        unbal = _unbalancierte_seiten(pdf)
        kz["seiten_unbalanciert"] = len(unbal)
        if unbal:
            zeige = ", ".join(str(s) for s in unbal[:MAX_SEITEN_IM_BEFUND])
            mehr = "" if len(unbal) <= MAX_SEITEN_IM_BEFUND else f" und {len(unbal) - MAX_SEITEN_IM_BEFUND} weitere"
            befunde.append(f"Marker unbalanciert auf Seite(n) {zeige}{mehr}")

        gefunden = sum(1 for t in texte if any(t[:PREFIX] in a for a in alts))
        kz["texte_gefunden"] = gefunden
        if gefunden < erwartet:
            befunde.append(f"Nur {gefunden} von {erwartet} als getaggt gemeldeten Texten stehen erreichbar in der Datei")

    return {"ok": not befunde, "befunde": befunde, "kennzahlen": kz}


def abnahme_loggen(ergebnis: dict, projekt=None, dokument=None, verfahren=None, datei=None) -> str:
    """Eine Logzeile fuer Nexus (Stundencheck greift „EXPORT-ABNAHME FEHLGESCHLAGEN“)."""
    kz = ergebnis.get("kennzahlen", {})
    status = "ok" if ergebnis.get("ok") else "FEHLGESCHLAGEN"
    zeile = (f"EXPORT-ABNAHME {status} projekt={projekt} dokument={dokument} verfahren={verfahren} "
             f"seiten={kz.get('seiten')}/{kz.get('seiten_original')} figures_alt={kz.get('figures_mit_alt')} "
             f"waisen={kz.get('waisen')} unbalanciert={kz.get('seiten_unbalanciert')} "
             f"texte={kz.get('texte_gefunden')}/{kz.get('erwartet_getaggt')}")
    if ergebnis.get("befunde"):
        zeile += " befunde=" + " | ".join(ergebnis["befunde"])
    if datei:
        zeile += f" datei={os.path.basename(datei)}"
    print(zeile)
    return zeile
