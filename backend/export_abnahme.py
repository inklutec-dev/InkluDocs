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
  6. Lesereihenfolge: Die Seiten der Bild-Elemente steigen in Baumreihenfolge an. Kleine
     Ruecksprünge (Doppelseiten, InDesign-Reihenfolge) sind erlaubt, ein Ruecksprung um mehr als
     LESEREIHENFOLGE_TOLERANZ_SEITEN Seiten ist ein Befund (Rollout d am 14.09.: neue Elemente
     hingen am Baumende, ein Screenreader haette sie nach der letzten Seite vorgelesen).
  7. veraPDF (PDF/UA-1) ueber den Konverter-Dienst: Der Export darf keine Regel verletzen, die das
     Original nicht verletzt, und keine Regel haeufiger verletzen als das Original („nicht schlechter
     als die Quelle“). Ausgenommen sind die Dokument-Metadaten-Regeln (VERAPDF_METADATEN_REGELN:
     PDF/UA-Kennung, Metadata-Strom, dc:title, ViewerPreferences): Sie sind Sache des spaeteren
     PDF/UA-Werkzeugs und tauchen als Folge auf, sobald ein Metadata-Strom ueberhaupt existiert. Fand am 14.09. „tagged content inside Artifact“ und doppelte Figures, die die
     Regeln 1-6 nicht sehen. Ist der Konverter nicht erreichbar, ist das KEIN Befund (Kennzahl
     verapdf = „nicht moeglich“, Logzeile), damit ein Ausfall des Pruefdienstes den Export nicht sperrt.

Die Abnahme aendert die Datei nie. Sie darf den Export nicht scheitern lassen (Aufrufer faengt
Ausnahmen); ihr Ergebnis geht als Warnung in den Export-Dialog und als Logzeile
„EXPORT-ABNAHME ...“ an Nexus.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional

import pikepdf

try:
    import pdfua_export as _pdfua  # Konverter-Anbindung (KONVERTER_URL), liefert veraPDF-Berichte
except Exception:  # noqa: BLE001 — Abnahme muss auch ohne Konverter-Modul laden
    _pdfua = None

PREFIX = 60  # Zeichen, ueber die ein geschriebener Text mit dem /Alt in der Datei verglichen wird
MAX_SEITEN_IM_BEFUND = 10
VERAPDF_MAX_BYTES = 60 * 1024 * 1024  # Grenze des Konverters (MAX_UPLOAD_BYTES)
# Dokument-Metadaten-Regeln (nicht Inhalt/Tagging): 5-1 PDF/UA-Kennung, 7.1-8 Metadata-Strom, 7.1-9 dc:title,
# 7.1-10 ViewerPreferences. Korpus 15.09.: Sobald finalize einen Metadata-Strom anlegt, verschwindet 7.1-8 und
# 5-1/7.1-9 erscheinen — eine Folge, keine Verschlechterung des Inhalts.
VERAPDF_METADATEN_REGELN = {("5", 1), ("7.1", 8), ("7.1", 9), ("7.1", 10)}
LESEREIHENFOLGE_TOLERANZ_SEITEN = 8  # Doppelseiten/InDesign-Reihenfolge: Kundendokument 14.09. hatte 84->80


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _seite_von(el: pikepdf.Dictionary, seiten: dict) -> Optional[int]:
    """Seitenindex (1-basiert) eines Strukturelements: /Pg am Element, sonst /Pg eines MCR-Kindes."""
    pg = el.get("/Pg")
    if isinstance(pg, pikepdf.Dictionary) and pg.objgen in seiten:
        return seiten[pg.objgen]
    k = el.get("/K")
    kinder = list(k) if isinstance(k, pikepdf.Array) else ([k] if k is not None else [])
    for kind in kinder:
        if isinstance(kind, pikepdf.Dictionary) and str(kind.get("/Type", "")) == "/MCR":
            pg = kind.get("/Pg")
            if isinstance(pg, pikepdf.Dictionary) and pg.objgen in seiten:
                return seiten[pg.objgen]
    return None


def _ist_figure(el: pikepdf.Dictionary, rolemap: dict) -> bool:
    """/S ist Figure — direkt oder ueber die RoleMap (InDesign: PlacedGraphic -> Figure; Rollout c 14.09.)."""
    s = str(el.get("/S", ""))
    if s == "/Figure":
        return True
    gesehen = set()
    while s in rolemap and s not in gesehen:
        gesehen.add(s)
        s = rolemap[s]
        if s == "/Figure":
            return True
    return False


def _figures_erreichbar(pdf: pikepdf.Pdf) -> tuple[int, int, list[str], list[int]]:
    """Laeuft den Strukturbaum vom StructTreeRoot in DOKUMENTREIHENFOLGE ab (Tiefensuche, Kinder
    in ihrer Reihenfolge). Rueckgabe: (figures gesamt, figures mit /Alt, Alt-Texte,
    Seitenindex je Figure-mit-Alt in Baumreihenfolge; None-Seiten ausgelassen)."""
    root = pdf.Root.get("/StructTreeRoot")
    if root is None:
        return 0, 0, [], []
    seiten = {p.obj.objgen: i for i, p in enumerate(pdf.pages, start=1)}
    rm = root.get("/RoleMap")
    rolemap = {str(k): str(v) for k, v in rm.items()} if isinstance(rm, pikepdf.Dictionary) else {}
    gesehen: set = set()
    figures = 0
    mit_alt = 0
    alts: list[str] = []
    folge: list[int] = []
    stapel = [root]
    while stapel:
        el = stapel.pop()
        if isinstance(el, pikepdf.Dictionary):
            og = el.objgen
            if og != (0, 0):
                if og in gesehen:
                    continue
                gesehen.add(og)
            if _ist_figure(el, rolemap):
                figures += 1
                if "/Alt" in el:
                    mit_alt += 1
                    alts.append(_norm(str(el.Alt)))
                    seite = _seite_von(el, seiten)
                    if seite is not None:
                        folge.append(seite)
            k = el.get("/K")
            if isinstance(k, pikepdf.Array):
                for x in reversed(list(k)):
                    stapel.append(x)
            elif k is not None:
                stapel.append(k)
        elif isinstance(el, pikepdf.Array):
            for x in reversed(list(el)):
                stapel.append(x)
    return figures, mit_alt, alts, folge


def _figures_mit_alt_gesamt(pdf: pikepdf.Pdf) -> int:
    """Alle Figure-Objekte (auch ueber RoleMap) mit /Alt in der Datei — auch unerreichbare (Waisen)."""
    root = pdf.Root.get("/StructTreeRoot")
    rm = root.get("/RoleMap") if root is not None else None
    rolemap = {str(k): str(v) for k, v in rm.items()} if isinstance(rm, pikepdf.Dictionary) else {}
    n = 0
    for o in pdf.objects:
        if isinstance(o, pikepdf.Dictionary) and "/Alt" in o and "/S" in o and _ist_figure(o, rolemap):
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


def _verapdf_regeln(pfad: str) -> Optional[dict]:
    """veraPDF-Regelverstoesse einer Datei als {(clause, test): failed}; None, wenn nicht pruefbar."""
    if _pdfua is None or not _pdfua.verfuegbar():
        return None
    if os.path.getsize(pfad) > VERAPDF_MAX_BYTES:
        return None
    with open(pfad, "rb") as f:
        bericht = _pdfua.pruefe(f.read())
    return {(r.get("clause"), r.get("test")): int(r.get("failed") or 0) for r in bericht.get("rules") or []}


def verapdf_vergleich(original_pfad: Optional[str], export_pfad: str) -> dict:
    """Regel 7: Export gegen Original. Rueckgabe {"moeglich": bool, "neu": [...], "schlechter": [...],
    "regeln_original": n, "regeln_export": n, "grund": str}."""
    erg = {"moeglich": False, "neu": [], "schlechter": [], "regeln_original": None, "regeln_export": None, "grund": ""}
    if not original_pfad or not os.path.isfile(original_pfad):
        erg["grund"] = "kein Original"
        return erg
    try:
        vor = _verapdf_regeln(original_pfad)
        nach = _verapdf_regeln(export_pfad)
    except Exception as e:  # noqa: BLE001 — Pruefdienst-Ausfall ist kein Befund
        erg["grund"] = f"veraPDF nicht moeglich: {e}"[:200]
        return erg
    if vor is None or nach is None:
        erg["grund"] = "Konverter nicht eingerichtet oder Datei zu gross"
        return erg
    erg["moeglich"] = True
    erg["regeln_original"] = len(vor)
    erg["regeln_export"] = len(nach)
    for schluessel, n in nach.items():
        if (str(schluessel[0]), int(schluessel[1] or 0)) in VERAPDF_METADATEN_REGELN:
            continue
        if schluessel not in vor:
            erg["neu"].append(f"{schluessel[0]}-{schluessel[1]} ({n}x)")
        elif n > vor[schluessel]:
            erg["schlechter"].append(f"{schluessel[0]}-{schluessel[1]} ({vor[schluessel]}->{n})")
    return erg


def abnahme_pdf(export_pfad: str, original_pfad: Optional[str],
                geschriebene_texte: Iterable[str],
                erwartet_getaggt: Optional[int] = None,
                verapdf: bool = True) -> dict:
    """Misst die exportierte Datei. Rueckgabe:
    {"ok": bool, "befunde": [str], "kennzahlen": {...}}.
    geschriebene_texte: die Alt-Texte, die der Export schreiben sollte (ohne "" und ohne "dekorativ").
    erwartet_getaggt: was der Export als geschrieben meldet; None = alle geschriebenen Texte.
    verapdf: Regel 7 ueber den Konverter ausfuehren (Tests ohne Konverter setzen False)."""
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

        figures, mit_alt, alts, folge = _figures_erreichbar(pdf)
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

        gross = [(folge[i - 1], folge[i]) for i in range(1, len(folge))
                 if folge[i] < folge[i - 1] - LESEREIHENFOLGE_TOLERANZ_SEITEN]
        kz["ruecksprünge_klein"] = sum(1 for i in range(1, len(folge)) if folge[i] < folge[i - 1]) - len(gross)
        kz["ruecksprünge_gross"] = len(gross)
        if gross:
            zeige = ", ".join(f"{a}->{b}" for a, b in gross[:MAX_SEITEN_IM_BEFUND])
            befunde.append(f"Lesereihenfolge: {len(gross)} Bild-Element(e) springen um mehr als "
                           f"{LESEREIHENFOLGE_TOLERANZ_SEITEN} Seiten zurueck ({zeige})")

        gefunden = sum(1 for t in texte if any(t[:PREFIX] in a for a in alts))
        kz["texte_gefunden"] = gefunden
        if gefunden < erwartet:
            befunde.append(f"Nur {gefunden} von {erwartet} als getaggt gemeldeten Texten stehen erreichbar in der Datei")

    # Regel 7 ausserhalb des with-Blocks (Datei ist zu, der Konverter liest sie frisch)
    if verapdf:
        v = verapdf_vergleich(original_pfad, export_pfad)
        if v["moeglich"]:
            kz["verapdf"] = f"{v['regeln_export']}/{v['regeln_original']}"
            if v["neu"]:
                befunde.append("veraPDF: neue Regelverletzung(en) gegenueber dem Original: " + ", ".join(v["neu"][:8]))
            if v["schlechter"]:
                befunde.append("veraPDF: Regel(n) haeufiger verletzt als im Original: " + ", ".join(v["schlechter"][:8]))
        else:
            kz["verapdf"] = "nicht moeglich"
            kz["verapdf_grund"] = v["grund"]
    else:
        kz["verapdf"] = "uebersprungen"

    return {"ok": not befunde, "befunde": befunde, "kennzahlen": kz}


def abnahme_loggen(ergebnis: dict, projekt=None, dokument=None, verfahren=None, datei=None) -> str:
    """Eine Logzeile fuer Nexus (Stundencheck greift „EXPORT-ABNAHME FEHLGESCHLAGEN“)."""
    kz = ergebnis.get("kennzahlen", {})
    status = "ok" if ergebnis.get("ok") else "FEHLGESCHLAGEN"
    zeile = (f"EXPORT-ABNAHME {status} projekt={projekt} dokument={dokument} verfahren={verfahren} "
             f"seiten={kz.get('seiten')}/{kz.get('seiten_original')} figures_alt={kz.get('figures_mit_alt')} "
             f"waisen={kz.get('waisen')} unbalanciert={kz.get('seiten_unbalanciert')} "
             f"ruecksprung={kz.get('ruecksprünge_gross')} verapdf={kz.get('verapdf')} "
             f"texte={kz.get('texte_gefunden')}/{kz.get('erwartet_getaggt')}")
    if ergebnis.get("befunde"):
        zeile += " befunde=" + " | ".join(ergebnis["befunde"])
    if datei:
        zeile += f" datei={os.path.basename(datei)}"
    print(zeile)
    return zeile
