"""Barrierefreie PDF aus Word (29.08.2026, Steve + Michael + Fable 5).

Ablauf: Word-Datei mit Alt-Texten (docx_export) -> Umwandler-Dienst (LibreOffice,
PDF/UA-Filter) -> veraPDF-Pruefung (PDF/UA-1) -> Klartext fuer Menschen.
Der Umwandler laeuft als eigener Container (konverter/), erreichbar ueber
KONVERTER_URL im Compose-Netz; fehlt die Variable, ist die Funktion aus
(verfuegbar() = False, Endpunkt antwortet 503).

Klartext: veraPDF meldet Regeln nach ISO 14289-1 (Klausel 7.x). Wir fassen sie zu
Bereichen zusammen, die auch Nicht-Techniker verstehen ("Bilder und Grafiken",
"Überschriften", "Tabellen" ...), und uebersetzen die haeufigsten Einzelregeln.
"""
from __future__ import annotations

import base64
import os
from typing import Optional

import httpx

KONVERTER_URL = (os.environ.get("KONVERTER_URL") or "").rstrip("/")
KONVERTER_TIMEOUT = float(os.environ.get("KONVERTER_TIMEOUT", "300"))


def verfuegbar() -> bool:
    return bool(KONVERTER_URL)


class UmwandlungFehlgeschlagen(Exception):
    pass


def konvertiere(docx_path: str, dateiname: str = "dokument.docx") -> tuple[bytes, dict]:
    """Schickt die Word-Datei zum Umwandler; liefert (PDF-Bytes, veraPDF-Bericht)."""
    if not verfuegbar():
        raise UmwandlungFehlgeschlagen("Umwandler nicht eingerichtet (KONVERTER_URL fehlt)")
    with open(docx_path, "rb") as f:
        data = f.read()
    try:
        r = httpx.post(f"{KONVERTER_URL}/pdfua", files={"datei": (dateiname, data,
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                       timeout=KONVERTER_TIMEOUT)
    except httpx.HTTPError as e:
        raise UmwandlungFehlgeschlagen(f"Umwandler nicht erreichbar: {e}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = r.text[:300]
        raise UmwandlungFehlgeschlagen(f"Umwandler meldet {r.status_code}: {detail}")
    d = r.json()
    return base64.b64decode(d["pdf_b64"]), d.get("verapdf") or {}


# ---------------------------------------------------------------------------
# Klartext
# ---------------------------------------------------------------------------

# Bereiche nach Klausel-Praefix von ISO 14289-1 (PDF/UA-1). Reihenfolge = Anzeige.
BEREICHE = [
    ("7.1", "Struktur und Lesereihenfolge", "Der Inhalt ist als Struktur ausgezeichnet, ein Screenreader kann ihn in der richtigen Reihenfolge lesen."),
    ("7.2", "Text und Sprache", "Text ist als Text hinterlegt und die Sprache des Dokuments ist gesetzt, damit die Sprachausgabe richtig ausspricht."),
    ("7.3", "Bilder und Grafiken", "Jedes Bild hat einen Alternativtext oder ist als Schmuckbild markiert."),
    ("7.4", "Überschriften", "Überschriften sind als Überschriften ausgezeichnet und in sinnvoller Reihenfolge."),
    ("7.5", "Tabellen", "Tabellen haben Kopfzellen, damit Zellen einer Spalte zugeordnet werden können."),
    ("7.6", "Listen", "Listen sind als Listen ausgezeichnet."),
    ("7.7", "Formeln", "Mathematische Formeln sind mit einem Alternativtext versehen."),
    ("7.9", "Fußnoten und Anmerkungen", "Fußnoten und Anmerkungen sind zugänglich verknüpft."),
    ("7.10", "Optionale Inhalte", "Ein- und ausblendbare Inhalte sind benannt."),
    ("7.11", "Eingebettete Dateien", "Eingebettete Dateien sind beschrieben."),
    ("7.16", "Schriften", "Schriften sind eingebettet, damit Text überall gleich erscheint und vorgelesen werden kann."),
    ("7.17", "Wiedergabe", "Multimedia-Inhalte sind zugänglich."),
    ("7.18", "Formularfelder und Verknüpfungen", "Formularfelder und Links haben eine Beschreibung."),
    ("7.20", "Metadaten", "Das Dokument nennt sich selbst als PDF/UA."),
    ("7.21", "Schriften", "Schriften sind eingebettet, damit Text überall gleich erscheint und vorgelesen werden kann."),
]

# Die haeufigsten Einzelregeln in Alltagssprache (Klausel, Testnummer).
REGELN_KLARTEXT = {
    ("7.1", 1): "Das Dokument ist nicht als getaggte PDF gekennzeichnet.",
    ("7.1", 2): "Die Struktur beginnt nicht mit einem Dokument-Element.",
    ("7.1", 3): "Es gibt Inhalte, die weder als Struktur noch als Schmuck markiert sind — Screenreader können sie überspringen oder unpassend lesen.",
    ("7.1", 8): "Ein Dokumenttitel fehlt im Dokument.",
    ("7.1", 9): "Der Dokumenttitel fehlt in den Metadaten.",
    ("7.1", 10): "Die PDF ist nicht so eingestellt, dass der Titel statt des Dateinamens angezeigt wird.",
    ("7.2", 3): "Die Sprache des Dokuments ist nicht gesetzt.",
    ("7.3", 1): "Ein Bild hat keinen Alternativtext.",
    ("7.3", 2): "Der Alternativtext eines Bildes ist ein Platzhalter.",
    ("7.4", 1): "Die Überschriften-Ebenen sind nicht durchgehend (zum Beispiel Ebene 1 gefolgt von Ebene 3).",
    ("7.5", 1): "Eine Tabelle hat keine Kopfzellen.",
    ("7.5", 2): "Zellen einer Tabelle sind nicht ihren Kopfzellen zugeordnet.",
    ("7.16", 1): "Eine Schrift ist nicht eingebettet.",
    ("7.18", 1): "Ein Formularfeld oder ein Link hat keine Beschreibung.",
    ("7.18", 5): "Ein Link hat keinen beschreibenden Text.",
    ("7.21", 1): "Eine Schrift ist nicht eingebettet.",
}


def _bereich(clause: str):
    for praefix, name, gut in BEREICHE:
        if clause == praefix or clause.startswith(praefix + "."):
            return praefix, name, gut
    return "7.x", "Weitere Prüfpunkte", "Weitere technische Anforderungen sind erfüllt."


def klartext(verapdf: dict) -> dict:
    """Aus dem veraPDF-Bericht eine Liste verstaendlicher Punkte machen.

    Rueckgabe: {"bestanden": bool, "profil": str, "regeln_fehlgeschlagen": int,
                "punkte": [{"bereich", "status": "ok"|"befund", "text", "regeln": [..]}]}
    """
    regeln = list((verapdf or {}).get("rules") or [])
    bestanden = bool((verapdf or {}).get("compliant")) and not regeln
    je_bereich: dict[str, list] = {}
    for r in regeln:
        praefix, _n, _g = _bereich(str(r.get("clause") or ""))
        je_bereich.setdefault(praefix, []).append(r)
    punkte = []
    # Kernbereiche immer nennen (auch wenn in Ordnung), die anderen nur bei Befund.
    immer = {"7.1", "7.2", "7.3", "7.4", "7.5"}
    for praefix, name, gut in BEREICHE:
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
            s = REGELN_KLARTEXT.get((str(r.get("clause")), r.get("test")))
            n = int(r.get("failed") or 0)
            if s:
                if n > 1 and s.startswith(("Ein ", "Eine ")):
                    s = s + f" ({n}-mal)"
                saetze.append(s)
            else:
                saetze.append(f"Ein technischer Prüfpunkt ist nicht erfüllt ({r.get('description') or 'ohne Beschreibung'})"
                              + (f", {n}-mal" if n > 1 else "") + ".")
        punkte.append({"bereich": name, "status": "befund", "text": " ".join(saetze),
                       "regeln": [f"{r.get('clause')}-{r.get('test')}" for r in betroffen]})
    rest = [r for p, rs in je_bereich.items() for r in rs if p == "7.x"]
    if rest:
        # Seltene Regeln: Anzahl plus die Original-Beschreibung von veraPDF (englisch,
        # aber ehrlich) — besser als ein nichtssagender Zaehler.
        beschr = "; ".join((r.get("description") or f"Regel {r.get('clause')}-{r.get('test')}")[:120] for r in rest)
        punkte.append({"bereich": "Weitere Prüfpunkte", "status": "befund",
                       "text": (f"{len(rest)} weitere technische Prüfpunkte sind nicht erfüllt: {beschr}."
                                if len(rest) > 1 else f"Ein weiterer technischer Prüfpunkt ist nicht erfüllt: {beschr}."),
                       "regeln": [f"{r.get('clause')}-{r.get('test')}" for r in rest]})
    return {"bestanden": bestanden, "profil": (verapdf or {}).get("profile") or "PDF/UA-1",
            "regeln_fehlgeschlagen": len(regeln), "punkte": punkte}


def zusammenfassung(pruefung: dict) -> str:
    """Ein Satz fuer die Ansage."""
    if pruefung.get("bestanden"):
        return "Deine PDF ist fertig und hat die Prüfung auf PDF/UA bestanden."
    n = sum(1 for p in pruefung.get("punkte", []) if p.get("status") == "befund")
    return (f"Deine PDF ist fertig. Die Prüfung meldet {n} {'Bereich' if n == 1 else 'Bereiche'} mit Hinweisen, "
            "die du dir ansehen solltest.")


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
