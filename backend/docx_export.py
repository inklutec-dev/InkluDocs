"""Alt-Texte in eine Word-Datei (.docx) zurueckschreiben.

Gegenstueck zu pdf_export.write_alt_texts_to_pdf fuer das Werkzeug
"Alt-Texte fuer Word-Dokumente" (26.08.2026).

GRUNDSATZ: Der Kunde bekommt SEINE Datei zurueck. Wir schreiben ein neues Zip,
in dem jeder Bestandteil byteweise aus dem Original kopiert wird — nur die
XML-Teile, in denen ein Bild einen neuen Alt-Text bekommt (word/document.xml,
word/headerN.xml, word/footerN.xml), werden neu serialisiert. Formatierung,
Formatvorlagen, Kommentare, Aenderungsverfolgung, eingebettete Schriften,
Eigenschaften: alles unveraendert. Word oeffnet die Datei wie zuvor.

WAS GESCHRIEBEN WIRD (pro Bildvorkommen, Anker "<part>|<docPr-id>"):
  - Alt-Text  -> wp:docPr@descr   (das Feld "Beschreibung" in Word)
  - Titel     -> wp:docPr@title   (optional; wird nur gesetzt, wenn uebergeben)
  - dekorativ -> wp:docPr/a:extLst/a:ext[uri=C183D7F6-...]/adec:decorative val="1"
                 UND descr wird geleert (so macht es Word selbst).
  - Ein Bild, fuer das KEIN Alt-Text uebergeben wird, bleibt exakt wie es war
    (auch ein vorhandener alter Alt-Text bleibt stehen).

Die Regeln "leerer Alt-Text = nicht anfassen" und "'dekorativ' = Kennzeichen
setzen" entsprechen _exportable_alt_text() im PDF-Weg.

SICHERHEIT: gleicher XML-Parser wie docx_processor (keine Entities, kein
Netzwerk), Zip-Grenzen werden vor dem Lesen geprueft, Ausgabe erst als
Temp-Datei und dann atomar umbenannt. Der Ausgabepfad muss der Aufrufer
in ein sicheres Verzeichnis legen (RESULTS_DIR/.../_export).
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass, field

from lxml import etree

from docx_processor import NS, DECORATIVE_EXT_URI, DocxFehler, _pruefe_zip, _lese_xml, _safe_parser

DEKORATIV = "dekorativ"


@dataclass
class DocxExportErgebnis:
    path: str
    geschrieben: int = 0            # Bilder mit gesetztem Alt-Text
    dekorativ: int = 0              # davon als dekorativ markiert
    uebersprungen: int = 0          # Anker ohne (verwertbaren) Alt-Text
    nicht_gefunden: list[str] = field(default_factory=list)   # Anker nicht im Dokument
    warnungen: list[str] = field(default_factory=list)


def _setze_dekorativ(docpr: etree._Element, an: bool) -> None:
    """adec:decorative in docPr/a:extLst setzen oder entfernen."""
    extlst = docpr.find("a:extLst", NS)
    if an:
        if extlst is None:
            extlst = etree.SubElement(docpr, f"{{{NS['a']}}}extLst")
        ext = None
        for e in extlst.findall("a:ext", NS):
            if e.get("uri") == DECORATIVE_EXT_URI:
                ext = e
                break
        if ext is None:
            ext = etree.SubElement(extlst, f"{{{NS['a']}}}ext")
            ext.set("uri", DECORATIVE_EXT_URI)
        deco = ext.find("adec:decorative", NS)
        if deco is None:
            deco = etree.SubElement(ext, f"{{{NS['adec']}}}decorative")
        deco.set("val", "1")
    else:
        if extlst is None:
            return
        for e in list(extlst.findall("a:ext", NS)):
            if e.get("uri") == DECORATIVE_EXT_URI:
                extlst.remove(e)
        if len(extlst) == 0:
            docpr.remove(extlst)


def write_alt_texts_to_docx(input_path: str, output_path: str,
                            alt_texts: dict[str, str],
                            titles: dict[str, str] | None = None) -> DocxExportErgebnis:
    """alt_texts: Anker -> Alt-Text ("dekorativ" = Kennzeichen setzen; leer/None = auslassen).
    titles: Anker -> Titel (optional). Schreibt output_path atomar."""
    titles = titles or {}
    erg = DocxExportErgebnis(path=output_path)

    # Anker nach Part gruppieren: {"word/document.xml": {7: "text", ...}}
    je_part: dict[str, dict[int, str]] = {}
    for anker, text in alt_texts.items():
        text = (text or "").strip()
        if not text:
            erg.uebersprungen += 1
            continue
        try:
            part, sid = anker.rsplit("|", 1)
            docpr_id = int(sid)
        except (ValueError, AttributeError):
            erg.nicht_gefunden.append(str(anker))
            continue
        je_part.setdefault(part, {})[docpr_id] = text

    try:
        zin = zipfile.ZipFile(input_path)
    except zipfile.BadZipFile:
        raise DocxFehler("Die Originaldatei ist keine gültige Word-Datei mehr.")
    with zin:
        _pruefe_zip(zin)
        namen = set(zin.namelist())
        for part in je_part:
            if part not in namen:
                erg.nicht_gefunden.extend(f"{part}|{i}" for i in je_part[part])
        neu_bytes: dict[str, bytes] = {}
        for part, ziele in je_part.items():
            if part not in namen:
                continue
            root = _lese_xml(zin, part)
            gefunden: set[int] = set()
            for docpr in root.iter(f"{{{NS['wp']}}}docPr"):
                try:
                    did = int(docpr.get("id"))
                except (TypeError, ValueError):
                    continue
                if did not in ziele or did in gefunden:
                    continue
                gefunden.add(did)
                text = ziele[did]
                anker = f"{part}|{did}"
                if text.lower() == DEKORATIV:
                    docpr.set("descr", "")
                    _setze_dekorativ(docpr, True)
                    erg.dekorativ += 1
                else:
                    docpr.set("descr", text)
                    _setze_dekorativ(docpr, False)
                    if anker in titles and (titles[anker] or "").strip():
                        docpr.set("title", titles[anker].strip())
                erg.geschrieben += 1
            for did in ziele:
                if did not in gefunden:
                    erg.nicht_gefunden.append(f"{part}|{did}")
            # Word schreibt seine XML-Teile mit XML-Deklaration und standalone="yes".
            neu_bytes[part] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

        out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".docx-export-", suffix=".tmp", dir=out_dir)
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp, "w") as zout:
                for zi in zin.infolist():
                    # Metadaten (Name, Datum, Kompression, Attribute) uebernehmen,
                    # damit die Datei fuer Word "dieselbe" bleibt.
                    neu = zipfile.ZipInfo(zi.filename, date_time=zi.date_time)
                    neu.compress_type = zi.compress_type
                    neu.external_attr = zi.external_attr
                    neu.create_system = zi.create_system
                    daten = neu_bytes.get(zi.filename)
                    if daten is None:
                        daten = zin.read(zi.filename)
                    zout.writestr(neu, daten)
            os.replace(tmp, output_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    if erg.nicht_gefunden:
        erg.warnungen.append(
            f"{len(erg.nicht_gefunden)} Bild(er) wurden im Dokument nicht mehr gefunden "
            "(Datei zwischenzeitlich geändert?) und blieben ohne Alt-Text.")
    return erg


def pruefe_unveraendert(original: str, export: str, geaenderte_parts: set[str]) -> list[str]:
    """Testhilfe: Liefert die Namen aller Zip-Mitglieder, die sich unterscheiden,
    obwohl sie NICHT geaendert werden durften. Leere Liste = alles gut."""
    unterschiede = []
    with zipfile.ZipFile(original) as a, zipfile.ZipFile(export) as b:
        if a.namelist() != b.namelist():
            unterschiede.append("<Mitgliederliste oder Reihenfolge weicht ab>")
        for n in a.namelist():
            if n in geaenderte_parts:
                continue
            if n not in b.namelist() or a.read(n) != b.read(n):
                unterschiede.append(n)
    return unterschiede
