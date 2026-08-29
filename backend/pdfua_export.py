"""Barrierefreie PDF aus Word (29.08.2026, Steve + Michael + Fable 5).

Ablauf: Word-Datei mit Alt-Texten (docx_export) -> Umwandler-Dienst (LibreOffice,
PDF/UA-Filter) -> Nachbearbeitung mit pikepdf (fehlende /Alt an Figure-Elementen
aus unseren Alt-Texten nachtragen — LibreOffice verliert sie bei VML-Bildern und
Bildern in Textfeldern) -> veraPDF-Pruefung (PDF/UA-1) -> Klartext fuer Menschen.

Der Umwandler laeuft als eigener Container (konverter/), erreichbar ueber
KONVERTER_URL im Compose-Netz; fehlt die Variable, ist die Funktion aus
(verfuegbar() = False, Endpunkt antwortet 503).

Klartext: veraPDF meldet Regeln nach ISO 14289-1 (Klausel 7.x). Wir fassen sie zu
Bereichen zusammen, die auch Nicht-Techniker verstehen ("Bilder und Grafiken",
"Überschriften", "Tabellen" ...), und uebersetzen die haeufigsten Einzelregeln.
Alle Texte gehen durch `_` (gettext), damit sie in der Sprache des Nutzers
erscheinen (Stufe 2, 29.08.2026).
"""
from __future__ import annotations

import base64
import os
from typing import Callable, Optional

import httpx

KONVERTER_URL = (os.environ.get("KONVERTER_URL") or "").rstrip("/")
KONVERTER_TIMEOUT = float(os.environ.get("KONVERTER_TIMEOUT", "300"))
DOCX_MEDIA = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _identitaet(s: str) -> str:
    return s


def verfuegbar() -> bool:
    return bool(KONVERTER_URL)


class UmwandlungFehlgeschlagen(Exception):
    pass


def _post(pfad: str, feldname: str, dateiname: str, data: bytes, media: str) -> dict:
    if not verfuegbar():
        raise UmwandlungFehlgeschlagen("Umwandler nicht eingerichtet (KONVERTER_URL fehlt)")
    try:
        r = httpx.post(f"{KONVERTER_URL}{pfad}", files={feldname: (dateiname, data, media)}, timeout=KONVERTER_TIMEOUT)
    except httpx.HTTPError as e:
        raise UmwandlungFehlgeschlagen(f"Umwandler nicht erreichbar: {e}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = r.text[:300]
        raise UmwandlungFehlgeschlagen(f"Umwandler meldet {r.status_code}: {detail}")
    return r.json()


def konvertiere(docx_path: str, dateiname: str = "dokument.docx") -> tuple[bytes, dict]:
    """Schickt die Word-Datei zum Umwandler; liefert (PDF-Bytes, veraPDF-Bericht)."""
    with open(docx_path, "rb") as f:
        data = f.read()
    d = _post("/pdfua", "datei", dateiname, data, DOCX_MEDIA)
    return base64.b64decode(d["pdf_b64"]), d.get("verapdf") or {}


def pruefe(pdf_bytes: bytes) -> dict:
    """Nur veraPDF (PDF/UA-1) fuer eine fertige PDF, z. B. nach der Nachbearbeitung."""
    d = _post("/pruefe", "datei", "pruefling.pdf", pdf_bytes, "application/pdf")
    return d.get("verapdf") or {}


# ---------------------------------------------------------------------------
# Nachbearbeitung: fehlende Alt-Texte in der PDF nachtragen (pikepdf)
# ---------------------------------------------------------------------------

def _figures_in_reihenfolge(root) -> list:
    """Alle Struktur-Elemente /S /Figure in Lesereihenfolge (Tiefensuche ueber /K)."""
    import pikepdf
    out = []
    gesehen = set()

    def besuche(el):
        try:
            key = (el.objgen if hasattr(el, "objgen") else None)
        except Exception:  # noqa: BLE001
            key = None
        if key and key in gesehen:
            return
        if key:
            gesehen.add(key)
        if not isinstance(el, pikepdf.Dictionary):
            return
        if el.get("/S") == pikepdf.Name("/Figure"):
            out.append(el)
        kinder = el.get("/K")
        if kinder is None:
            return
        if isinstance(kinder, pikepdf.Array):
            for k in kinder:
                if isinstance(k, pikepdf.Dictionary):
                    besuche(k)
        elif isinstance(kinder, pikepdf.Dictionary):
            besuche(kinder)

    besuche(root)
    return out


def _rahmen_umwandeln(root) -> int:
    """Leere Figure-Elemente, auf die direkt ein /Div mit "/Frame contents" folgt,
    zu /Div machen (LibreOffice-Textfeldrahmen). Liefert die Anzahl."""
    import pikepdf
    n = 0
    gesehen = set()

    def ist_frame_div(el) -> bool:
        if not isinstance(el, pikepdf.Dictionary) or el.get("/S") != pikepdf.Name("/Div"):
            return False
        k = el.get("/K")
        kinder = list(k) if isinstance(k, pikepdf.Array) else ([k] if isinstance(k, pikepdf.Dictionary) else [])
        return any(isinstance(x, pikepdf.Dictionary) and str(x.get("/S") or "") == "/Frame contents" for x in kinder)

    def besuche(el):
        nonlocal n
        if not isinstance(el, pikepdf.Dictionary):
            return
        try:
            key = el.objgen
        except Exception:  # noqa: BLE001
            key = None
        if key and key in gesehen:
            return
        if key:
            gesehen.add(key)
        k = el.get("/K")
        if isinstance(k, pikepdf.Array):
            kinder = [x for x in k if isinstance(x, pikepdf.Dictionary)]
            for i, kind in enumerate(kinder):
                if (kind.get("/S") == pikepdf.Name("/Figure") and not str(kind.get("/Alt") or "").strip()
                        and len(_figures_in_reihenfolge(kind)) == 1
                        and i + 1 < len(kinder) and ist_frame_div(kinder[i + 1])):
                    kind["/S"] = pikepdf.Name("/Div")
                    n += 1
                besuche(kind)
        elif isinstance(k, pikepdf.Dictionary):
            besuche(k)

    besuche(root)
    return n


def alt_nachtragen(pdf_bytes: bytes, alts: list[Optional[str]]) -> tuple[bytes, dict]:
    """Traegt fehlende /Alt an Figure-Elementen nach.

    alts: Alt-Texte der Bilder des Dokumentkoerpers in Dokumentreihenfolge
    (None = kein Text, "dekorativ" = Schmuckbild). Zugeordnet wird nur, wenn
    die Zahl der Figure-Elemente in der PDF genau der Zahl der Bilder entspricht
    — sonst wird nichts angefasst (lieber ehrlich melden als falsch zuordnen).
    Rueckgabe: (PDF-Bytes, {"nachgetragen": n, "figures": n_pdf, "bilder": n_docx,
                            "zugeordnet": bool, "dekorativ_offen": n})"""
    info = {"nachgetragen": 0, "figures": 0, "bilder": len(alts), "zugeordnet": False, "dekorativ_offen": 0,
            "rahmen_umgewandelt": 0}
    try:
        import io
        import pikepdf
    except Exception:  # noqa: BLE001
        return pdf_bytes, info
    try:
        pdf = pikepdf.open(io.BytesIO(pdf_bytes))
        root = pdf.Root.get("/StructTreeRoot")
        if root is None:
            return pdf_bytes, info
        # Rahmen: LibreOffice taggt ein Textfeld als eigenes, leeres Figure-Element,
        # unmittelbar gefolgt von einem /Div, dessen Kind "/Frame contents" das
        # eigentliche Bild als Figure traegt (gemessen 29.08.2026, Projekt 317).
        # Ein Rahmen ist kein Bild — als Gruppe (/Div) braucht er keinen Alt-Text.
        geaendert = False
        info["rahmen_umgewandelt"] += _rahmen_umwandeln(root)
        if info["rahmen_umgewandelt"]:
            geaendert = True
        alle = _figures_in_reihenfolge(root)
        # Zweites Muster (Vorsicht): ein Figure, das selbst Figures enthaelt, ist
        # ebenfalls ein Container, kein Bild.
        figures = []
        for fig in alle:
            if len(_figures_in_reihenfolge(fig)) > 1:     # fig selbst + mindestens ein inneres Figure
                fig["/S"] = pikepdf.Name("/Div")
                info["rahmen_umgewandelt"] += 1
                geaendert = True
            else:
                figures.append(fig)
        info["figures"] = len(figures)
        if not figures or len(figures) != len(alts):
            if geaendert:
                out = io.BytesIO()
                pdf.save(out)
                return out.getvalue(), info
            return pdf_bytes, info
        info["zugeordnet"] = True
        for fig, alt in zip(figures, alts):
            vorhanden = str(fig.get("/Alt") or "").strip()
            if vorhanden:
                continue
            if alt is None or not str(alt).strip():
                continue
            if str(alt).strip().lower() == "dekorativ":
                # Ein Schmuckbild muesste als Artefakt aus der Struktur genommen werden —
                # das ist ein Eingriff in den Inhaltsstrom (spaeter). Vorerst nur zaehlen.
                info["dekorativ_offen"] += 1
                continue
            fig["/Alt"] = pikepdf.String(str(alt).strip())
            info["nachgetragen"] += 1
            geaendert = True
        if not geaendert:
            return pdf_bytes, info
        out = io.BytesIO()
        pdf.save(out)
        return out.getvalue(), info
    except Exception:  # noqa: BLE001
        return pdf_bytes, info


# ---------------------------------------------------------------------------
# Klartext
# ---------------------------------------------------------------------------

# Bereiche nach Klausel-Praefix von ISO 14289-1 (PDF/UA-1). Reihenfolge = Anzeige.
# Texte als Funktionen von _, damit gettext greift.
def _bereiche(_):
    return [
        ("7.1", _("Struktur und Lesereihenfolge"), _("Der Inhalt ist als Struktur ausgezeichnet, ein Screenreader kann ihn in der richtigen Reihenfolge lesen.")),
        ("7.2", _("Text und Sprache"), _("Text ist als Text hinterlegt und die Sprache des Dokuments ist gesetzt, damit die Sprachausgabe richtig ausspricht.")),
        ("7.3", _("Bilder und Grafiken"), _("Jedes Bild hat einen Alternativtext oder ist als Schmuckbild markiert.")),
        ("7.4", _("Überschriften"), _("Überschriften sind als Überschriften ausgezeichnet und in sinnvoller Reihenfolge.")),
        ("7.5", _("Tabellen"), _("Tabellen haben Kopfzellen, damit Zellen einer Spalte zugeordnet werden können.")),
        ("7.6", _("Listen"), _("Listen sind als Listen ausgezeichnet.")),
        ("7.7", _("Formeln"), _("Mathematische Formeln sind mit einem Alternativtext versehen.")),
        ("7.9", _("Fußnoten und Anmerkungen"), _("Fußnoten und Anmerkungen sind zugänglich verknüpft.")),
        ("7.10", _("Optionale Inhalte"), _("Ein- und ausblendbare Inhalte sind benannt.")),
        ("7.11", _("Eingebettete Dateien"), _("Eingebettete Dateien sind beschrieben.")),
        ("7.16", _("Schriften"), _("Schriften sind eingebettet, damit Text überall gleich erscheint und vorgelesen werden kann.")),
        ("7.17", _("Wiedergabe"), _("Multimedia-Inhalte sind zugänglich.")),
        ("7.18", _("Formularfelder und Verknüpfungen"), _("Formularfelder und Links haben eine Beschreibung.")),
        ("7.20", _("Metadaten"), _("Das Dokument nennt sich selbst als PDF/UA.")),
        ("7.21", _("Schriften"), _("Schriften sind eingebettet, damit Text überall gleich erscheint und vorgelesen werden kann.")),
    ]


def _regeln_klartext(_):
    return {
        ("7.1", 1): _("Das Dokument ist nicht als getaggte PDF gekennzeichnet."),
        ("7.1", 2): _("Die Struktur beginnt nicht mit einem Dokument-Element."),
        ("7.1", 3): _("Es gibt Inhalte, die weder als Struktur noch als Schmuck markiert sind — Screenreader können sie überspringen oder unpassend lesen."),
        ("7.1", 8): _("Ein Dokumenttitel fehlt im Dokument."),
        ("7.1", 9): _("Der Dokumenttitel fehlt in den Metadaten."),
        ("7.1", 10): _("Die PDF ist nicht so eingestellt, dass der Titel statt des Dateinamens angezeigt wird."),
        ("7.2", 3): _("Die Sprache des Dokuments ist nicht gesetzt."),
        ("7.3", 1): _("Ein Bild hat keinen Alternativtext."),
        ("7.3", 2): _("Der Alternativtext eines Bildes ist ein Platzhalter."),
        ("7.4", 1): _("Die Überschriften-Ebenen sind nicht durchgehend (zum Beispiel Ebene 1 gefolgt von Ebene 3)."),
        ("7.5", 1): _("Eine Tabelle hat keine Kopfzellen."),
        ("7.5", 2): _("Zellen einer Tabelle sind nicht ihren Kopfzellen zugeordnet."),
        ("7.16", 1): _("Eine Schrift ist nicht eingebettet."),
        ("7.18", 1): _("Ein Formularfeld oder ein Link hat keine Beschreibung."),
        ("7.18", 5): _("Ein Link hat keinen beschreibenden Text."),
        ("7.21", 1): _("Eine Schrift ist nicht eingebettet."),
    }


# Rueckwaertskompatibel (Tests, Doku): deutsche Tabellen ohne gettext
BEREICHE = _bereiche(_identitaet)
REGELN_KLARTEXT = _regeln_klartext(_identitaet)


def _bereich(clause: str, bereiche, _):
    for praefix, name, gut in bereiche:
        if clause == praefix or clause.startswith(praefix + "."):
            return praefix, name, gut
    return "7.x", _("Weitere Prüfpunkte"), _("Weitere technische Anforderungen sind erfüllt.")


def klartext(verapdf: dict, _: Callable[[str], str] = _identitaet) -> dict:
    """Aus dem veraPDF-Bericht eine Liste verstaendlicher Punkte machen.

    Rueckgabe: {"bestanden": bool, "profil": str, "regeln_fehlgeschlagen": int,
                "punkte": [{"bereich", "status": "ok"|"befund", "text", "regeln": [..]}]}
    """
    bereiche = _bereiche(_)
    regeln_kt = _regeln_klartext(_)
    regeln = list((verapdf or {}).get("rules") or [])
    bestanden = bool((verapdf or {}).get("compliant")) and not regeln
    je_bereich: dict[str, list] = {}
    for r in regeln:
        praefix, _n, _g = _bereich(str(r.get("clause") or ""), bereiche, _)
        je_bereich.setdefault(praefix, []).append(r)
    punkte = []
    # Kernbereiche immer nennen (auch wenn in Ordnung), die anderen nur bei Befund.
    immer = {"7.1", "7.2", "7.3", "7.4", "7.5"}
    for praefix, name, gut in bereiche:
        if praefix == "7.21":   # gleicher Bereich wie 7.16, nicht doppelt zeigen
            continue
        betroffen = je_bereich.get(praefix, []) + (je_bereich.get("7.21", []) if praefix == "7.16" else [])
        if not betroffen and praefix not in immer:
            continue
        if not betroffen:
            punkte.append({"bereich": name, "status": "ok", "text": gut, "regeln": []})
            continue
        saetze = []
        for r in betroffen:
            s = regeln_kt.get((str(r.get("clause")), r.get("test")))
            n = int(r.get("failed") or 0)
            if s:
                if n > 1:
                    s = s + " " + _("({n}-mal)").format(n=n)
                saetze.append(s)
            else:
                saetze.append(_("Ein technischer Prüfpunkt ist nicht erfüllt ({beschreibung})").format(
                    beschreibung=(r.get("description") or _("ohne Beschreibung"))[:160])
                    + (" " + _("({n}-mal)").format(n=n) if n > 1 else "") + ".")
        punkte.append({"bereich": name, "status": "befund", "text": " ".join(saetze),
                       "regeln": [f"{r.get('clause')}-{r.get('test')}" for r in betroffen]})
    rest = [r for p, rs in je_bereich.items() for r in rs if p == "7.x"]
    if rest:
        beschr = "; ".join((r.get("description") or f"Regel {r.get('clause')}-{r.get('test')}")[:120] for r in rest)
        punkte.append({"bereich": _("Weitere Prüfpunkte"), "status": "befund",
                       "text": (_("{n} weitere technische Prüfpunkte sind nicht erfüllt: {beschreibung}.").format(n=len(rest), beschreibung=beschr)
                                if len(rest) > 1 else _("Ein weiterer technischer Prüfpunkt ist nicht erfüllt: {beschreibung}.").format(beschreibung=beschr)),
                       "regeln": [f"{r.get('clause')}-{r.get('test')}" for r in rest]})
    return {"bestanden": bestanden, "profil": (verapdf or {}).get("profile") or "PDF/UA-1",
            "regeln_fehlgeschlagen": len(regeln), "punkte": punkte}


def zusammenfassung(pruefung: dict, _: Callable[[str], str] = _identitaet) -> str:
    """Ein Satz fuer die Ansage."""
    if pruefung.get("bestanden"):
        return _("Deine PDF ist fertig und hat die Prüfung auf PDF/UA bestanden.")
    n = sum(1 for p in pruefung.get("punkte", []) if p.get("status") == "befund")
    if n == 1:
        return _("Deine PDF ist fertig. Die Prüfung meldet einen Bereich mit Hinweisen, den du dir ansehen solltest.")
    return _("Deine PDF ist fertig. Die Prüfung meldet {n} Bereiche mit Hinweisen, die du dir ansehen solltest.").format(n=n)


_NS = {"cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
       "dc": "http://purl.org/dc/elements/1.1/"}


def dokumenttitel_setzen(docx_path: str, titel: str, sprache: Optional[str] = None) -> bool:
    """Titel (und Sprache) in docProps/core.xml schreiben, falls leer — LibreOffice
    uebernimmt den Titel in die PDF-Metadaten (PDF/UA 7.1-8/-9) und die Sprache in
    den Dokumentkatalog (7.2-3). Ohne python-docx: Zip + lxml, nur dieser eine
    Teil wird ersetzt, alles andere bleibt byte-gleich. True = etwas geaendert."""
    import shutil
    import tempfile
    import zipfile
    from lxml import etree

    try:
        with zipfile.ZipFile(docx_path) as z:
            if "docProps/core.xml" not in z.namelist():
                return False
            core = etree.fromstring(z.read("docProps/core.xml"))
        geaendert = False
        t = core.find("dc:title", _NS)
        if t is None:
            t = etree.SubElement(core, "{%s}title" % _NS["dc"])
        if not (t.text or "").strip():
            t.text = titel[:250]
            geaendert = True
        if sprache:
            l = core.find("dc:language", _NS)
            if l is None:
                l = etree.SubElement(core, "{%s}language" % _NS["dc"])
            if not (l.text or "").strip():
                l.text = sprache
                geaendert = True
        if not geaendert:
            return False
        neu = etree.tostring(core, xml_declaration=True, encoding="UTF-8", standalone=True)
        fd, tmp = tempfile.mkstemp(suffix=".docx", dir=os.path.dirname(docx_path) or None)
        os.close(fd)
        with zipfile.ZipFile(docx_path) as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = neu if item.filename == "docProps/core.xml" else zin.read(item.filename)
                zout.writestr(item, data)
        shutil.move(tmp, docx_path)
        return True
    except Exception:  # noqa: BLE001
        return False
