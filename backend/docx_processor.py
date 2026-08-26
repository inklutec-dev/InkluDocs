"""Word-Dokumente (.docx) lesen: Bilder, vorhandene Alt-Texte, Dekorativ-Kennzeichen
und der Textkontext jedes Bildes.

Gegenstueck zu pdf_processor.extract_images_from_pdf fuer das Werkzeug
"Alt-Texte fuer Word-Dokumente" (26.08.2026). Bewusst OHNE python-docx als
Laufzeit-Abhaengigkeit: Wir lesen das OOXML-Zip direkt mit zipfile + lxml, weil
wir (a) exakt kontrollieren wollen, was geparst wird (Sicherheit), (b) auch
Kopf-/Fusszeilen und frei positionierte Bilder (wp:anchor) sehen muessen, die
python-docx nur ueber Umwege liefert, und (c) das Zurueckschreiben
(docx_export.py) byte-identisch fuer alle unberuehrten Teile sein soll.

WO DER ALT-TEXT IN WORD STEHT
  Jedes Bild ist ein <w:drawing> mit <wp:inline> (im Textfluss) oder <wp:anchor>
  (frei positioniert). Darin traegt <wp:docPr id=".." name=".." descr=".." title="..">
  den Alternativtext (descr) und den Titel. "Dekorativ" (Word 2019+) ist eine
  Erweiterung <adec:decorative val="1"/> in docPr/a:extLst. Das eigentliche Bild
  referenziert <a:blip r:embed="rIdN"/> -> Relationship -> word/media/... .

ANKER
  Jedes gefundene Bild bekommt einen stabilen Anker "<partname>|<docPr-id>"
  (z. B. "word/document.xml|7"). Damit findet docx_export.py genau dieses
  Bildelement wieder, ohne auf Reihenfolge oder Dateinamen angewiesen zu sein.
  Voraussetzung: Die Originaldatei bleibt zwischen Lesen und Schreiben
  unveraendert (sie liegt unter /app/data/... und wird nie angefasst).

SICHERHEIT
  - XML nur mit lxml-Parser ohne Entity-Aufloesung und ohne Netzwerk (XXE).
  - Zip-Bomben: Grenzen fuer Mitgliederzahl, Gesamtgroesse und Kompressionsrate.
  - Es wird NICHTS aus dem Zip auf die Platte entpackt; Bilddaten werden im
    Speicher gelesen, mit Pillow geprueft und unter EIGENEM Namen gespeichert.
  - Zip-Eintragsnamen mit ".." oder absoluten Pfaden werden abgewiesen.
  - Makro-Dateien (.docm) und das Altformat (.doc) werden vor dem Lesen abgewiesen
    (Aufrufer prueft die Endung, hier zusaetzlich der Inhaltstyp).

KONTEXT (Vorteil gegenueber PDF: alles steht explizit im Dokument)
  Teil, Ueberschrift(en) ueber dem Bild, Absatz davor/danach, Bildunterschrift
  (Formatvorlage Caption/Beschriftung oder Muster "Abbildung N:"), Tabellenzeile,
  Querverweise im Text auf die Bildunterschrift ("siehe Abbildung 1"),
  Dokumenttitel. Alles wird zu EINEM Kontext-String zusammengesetzt, der wie bei
  PDF in images.context_text landet und im Editor lesbar ist.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field

from lxml import etree
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- Grenzen
MAX_ZIP_MEMBERS = 5000          # ein normales Word-Dokument hat < 100
MAX_TOTAL_UNCOMPRESSED = 400 * 1024 * 1024
MAX_XML_PART = 60 * 1024 * 1024  # document.xml riesiger Dokumente bleibt darunter
MAX_COMPRESSION_RATIO = 300      # XML komprimiert gut, aber nicht 1:300 (Bombe)
MAX_IMAGES = 2000                # Sicherheitsdeckel; Tageslimit greift separat
MAX_CONTEXT_CHARS = 1500

# ---------------------------------------------------------------- Namensraeume
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "adec": "http://schemas.microsoft.com/office/drawing/2017/decorative",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "v": "urn:schemas-microsoft-com:vml",
}
DECORATIVE_EXT_URI = "{C183D7F6-B498-43B3-948B-1728B52AA6E4}"
CT_DOCM = "application/vnd.ms-word.document.macroEnabled.main+xml"
CT_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"

# Bildformate, die wir direkt verarbeiten koennen (Pillow). EMF/WMF/SVG sind
# Vektorformate ohne Pillow-Unterstuetzung -> werden als "nicht unterstuetzt"
# gemeldet (Stufe 2: Rendern ueber LibreOffice).
RASTER_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
VEKTOR_EXT = {".emf", ".wmf", ".svg"}

# Ueberschriften-Formatvorlagen: Word-intern heissen sie "Heading1".."Heading9"
# (auch in deutschem Word; der Anzeigename "Ueberschrift 1" ist nur Etikett),
# aeltere/andere Vorlagen koennen "berschrift1" heissen. Title = Dokumenttitel.
_HEADING_RE = re.compile(r"^(heading|berschrift|titre|t[ií]tulo|rubrik|overskrift)\s*(\d)?$", re.I)
_CAPTION_STYLE_RE = re.compile(r"^(caption|beschriftung|l[ée]gende|descripci[oó]n|billedtekst|bildtext)$", re.I)
_CAPTION_TEXT_RE = re.compile(
    r"^\s*(abbildung|abb\.|bild|grafik|figure|fig\.|figura|illustration|diagramm|tabelle|table)\s*\d+", re.I)

_safe_parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False,
                               remove_blank_text=False)


class DocxFehler(ValueError):
    """Verstaendliche Fehlermeldung fuer den Nutzer (wird als 400 ausgegeben)."""


@dataclass
class DocxBild:
    """Ein Bildvorkommen im Dokument (ein docPr). Dasselbe Medium kann mehrfach
    vorkommen -> mehrere DocxBild mit gleichem hash."""
    anker: str                  # "<partname>|<docPr-id>"
    part: str                   # z. B. word/document.xml
    docpr_id: int
    name: str                   # docPr name (Word: "Grafik 3")
    original_alt: str           # descr (vorhandener Alt-Text) oder ""
    original_title: str         # title oder ""
    decorative: bool            # adec:decorative val=1
    media_part: str             # word/media/image1.png
    media_ext: str
    hash: str                   # sha256 der Bildbytes
    order: int                  # Reihenfolge im Dokument (Kopfzeilen zuerst)
    abschnitt: int              # 1-basiert: Index der Ueberschrift-1-Gruppe
    ort: str                    # "Text", "Tabelle", "Kopfzeile", "Fusszeile", "Textfeld"
    anchored: bool              # wp:anchor (frei positioniert) statt inline
    context: str = ""
    caption: str = ""
    width: int = 0
    height: int = 0
    image_path: str = ""
    unsupported: str = ""       # Grund, falls nicht verarbeitbar (z. B. EMF)


@dataclass
class DocxErgebnis:
    bilder: list[DocxBild] = field(default_factory=list)
    titel: str = ""
    ueberschriften: list[str] = field(default_factory=list)
    volltext_zeichen: int = 0
    uebersprungen: list[dict] = field(default_factory=list)   # {anker, grund}
    warnungen: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- Zip-Pruefung
def _pruefe_zip(zf: zipfile.ZipFile) -> None:
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise DocxFehler("Die Datei enthält ungewöhnlich viele Bestandteile und wird nicht verarbeitet.")
    total = 0
    for zi in infos:
        n = zi.filename
        if n.startswith(("/", "\\")) or ".." in n.split("/") or ":" in n[:3]:
            raise DocxFehler("Die Datei enthält ungültige Pfade und wird nicht verarbeitet.")
        total += zi.file_size
        if zi.compress_size and zi.file_size / max(zi.compress_size, 1) > MAX_COMPRESSION_RATIO \
                and zi.file_size > 1024 * 1024:
            raise DocxFehler("Die Datei ist ungewöhnlich stark komprimiert und wird nicht verarbeitet.")
    if total > MAX_TOTAL_UNCOMPRESSED:
        raise DocxFehler("Die Datei ist entpackt zu groß (über 400 MB).")
    namen = set(zf.namelist())
    if "[Content_Types].xml" not in namen or "word/document.xml" not in namen:
        raise DocxFehler("Das ist keine Word-Datei im DOCX-Format (word/document.xml fehlt).")


def _lese_xml(zf: zipfile.ZipFile, name: str) -> etree._Element:
    zi = zf.getinfo(name)
    if zi.file_size > MAX_XML_PART:
        raise DocxFehler("Ein Bestandteil der Datei ist zu groß, um ihn sicher zu lesen.")
    return etree.fromstring(zf.read(name), _safe_parser)


def _inhaltstyp_pruefen(zf: zipfile.ZipFile) -> None:
    ct = _lese_xml(zf, "[Content_Types].xml")
    for ov in ct.findall("ct:Override", NS):
        if ov.get("PartName") == "/word/document.xml":
            typ = ov.get("ContentType", "")
            if typ == CT_DOCM:
                raise DocxFehler("Word-Dateien mit Makros (.docm) werden nicht verarbeitet. "
                                 "Bitte in Word als .docx ohne Makros speichern.")
            if typ != CT_DOCX:
                raise DocxFehler("Unbekannter Dokumenttyp – bitte eine normale .docx-Datei hochladen.")
            return
    raise DocxFehler("Das ist keine Word-Datei im DOCX-Format (Inhaltstyp fehlt).")


def _rels(zf: zipfile.ZipFile, part: str) -> dict[str, str]:
    """Relationship-Id -> Zielpart (absolut, z. B. word/media/image1.png)."""
    d, f = os.path.split(part)
    relname = f"{d}/_rels/{f}.rels" if d else f"_rels/{f}.rels"
    if relname not in zf.namelist():
        return {}
    out = {}
    for r in _lese_xml(zf, relname).findall("rel:Relationship", NS):
        if r.get("TargetMode") == "External":
            continue      # externe Bilder (Links) lassen wir aus
        target = r.get("Target", "")
        if target.startswith("/"):
            ziel = target.lstrip("/")
        else:
            ziel = os.path.normpath(os.path.join(d, target)).replace("\\", "/")
        if ziel.startswith("..") or ziel.startswith("/"):
            continue
        out[r.get("Id")] = ziel
    return out


def _text(el: etree._Element) -> str:
    """Sichtbarer Text eines Absatz-/Zellen-Elements (w:t, Tabs, Umbrueche)."""
    teile = []
    for t in el.iter():
        if t.tag == f"{{{NS['w']}}}t":
            teile.append(t.text or "")
        elif t.tag in (f"{{{NS['w']}}}tab",):
            teile.append(" ")
        elif t.tag in (f"{{{NS['w']}}}br", f"{{{NS['w']}}}cr"):
            teile.append(" ")
    return re.sub(r"\s+", " ", "".join(teile)).strip()


def _pstyle(p: etree._Element) -> str:
    ps = p.find("w:pPr/w:pStyle", NS)
    return (ps.get(f"{{{NS['w']}}}val") or "") if ps is not None else ""


def _heading_level(style_id: str, styles: dict[str, str]) -> int | None:
    """0 = Titel, 1..9 = Ueberschrift n, None = keine Ueberschrift."""
    for kandidat in (style_id, styles.get(style_id, "")):
        m = _HEADING_RE.match(kandidat.replace(" ", ""))
        if m:
            return int(m.group(2)) if m.group(2) else (0 if m.group(1).lower() in ("title", "titel") else 1)
        if kandidat.lower() in ("title", "titel"):
            return 0
    return None


def _ist_caption(p: etree._Element, styles: dict[str, str]) -> bool:
    sid = _pstyle(p)
    if _CAPTION_STYLE_RE.match(sid) or _CAPTION_STYLE_RE.match(styles.get(sid, "")):
        return True
    return bool(_CAPTION_TEXT_RE.match(_text(p)))


def _styles(zf: zipfile.ZipFile) -> dict[str, str]:
    """styleId -> Anzeigename (w:name), fuer Ueberschriften-/Beschriftungs-Erkennung."""
    if "word/styles.xml" not in zf.namelist():
        return {}
    out = {}
    for s in _lese_xml(zf, "word/styles.xml").findall("w:style", NS):
        n = s.find("w:name", NS)
        if n is not None:
            out[s.get(f"{{{NS['w']}}}styleId", "")] = n.get(f"{{{NS['w']}}}val", "")
    return out


def _dokumenttitel(zf: zipfile.ZipFile) -> str:
    if "docProps/core.xml" not in zf.namelist():
        return ""
    try:
        t = _lese_xml(zf, "docProps/core.xml").find("dc:title", NS)
        return (t.text or "").strip() if t is not None else ""
    except Exception:
        return ""


# ---------------------------------------------------------------- Kontext
def _absaetze_des_teils(root: etree._Element) -> list[etree._Element]:
    """Alle w:p in Dokumentreihenfolge (auch in Tabellen/Textfeldern)."""
    return list(root.iter(f"{{{NS['w']}}}p"))


def _kontext_fuer(drawing: etree._Element, absaetze: list[etree._Element], index_von: dict,
                  styles: dict[str, str], teil_label: str, titel: str) -> tuple[str, str, str, int]:
    """Liefert (kontext, caption, ort, abschnitt)."""
    p = drawing
    while p is not None and p.tag != f"{{{NS['w']}}}p":
        p = p.getparent()
    if p is None:
        return "", "", teil_label, 1
    i = index_von.get(id(p), 0)

    # Ort: Tabelle? Textfeld?
    ort = teil_label
    anc = p.getparent()
    zelle = None
    while anc is not None:
        if anc.tag == f"{{{NS['w']}}}tc":
            zelle = anc; ort = "Tabelle" if ort == "Text" else ort
        if anc.tag == f"{{{NS['w']}}}txbxContent":
            ort = "Textfeld" if ort == "Text" else ort
        anc = anc.getparent()

    # Ueberschriften ueber dem Bild (Pfad: H1 > H2 > ...), und Abschnittsnummer
    pfad: dict[int, str] = {}
    abschnitt = 0
    for q in absaetze[:i + 1]:
        lvl = _heading_level(_pstyle(q), styles)
        if lvl is not None:
            txt = _text(q)
            if not txt:
                continue
            pfad[lvl] = txt
            for tiefer in [k for k in pfad if k > lvl]:
                del pfad[tiefer]
    # Abschnitt = wievielte Ueberschrift 1 (oder Titel) vor dem Bild
    for q in absaetze[:i + 1]:
        lvl = _heading_level(_pstyle(q), styles)
        if lvl is not None and lvl <= 1 and _text(q):
            abschnitt += 1
    abschnitt = max(abschnitt, 1)

    eigener = _text(p)
    davor = ""
    for q in reversed(absaetze[max(0, i - 3):i]):
        t = _text(q)
        if t and _heading_level(_pstyle(q), styles) is None:
            davor = t; break
    danach = ""; caption = ""
    for q in absaetze[i + 1:i + 4]:
        t = _text(q)
        if not t:
            continue
        if not caption and _ist_caption(q, styles):
            caption = t; continue
        if not danach and _heading_level(_pstyle(q), styles) is None:
            danach = t
        if caption and danach:
            break
    # Bildunterschrift kann auch UEBER dem Bild stehen (Tabellen-Stil)
    if not caption:
        for q in reversed(absaetze[max(0, i - 2):i]):
            if _ist_caption(q, styles):
                caption = _text(q); break

    # Querverweise: Saetze im Dokument, die die Abbildungsnummer nennen
    verweise = []
    m = re.match(r"^\s*((?:abbildung|abb\.|bild|grafik|figure|fig\.|figura|illustration|diagramm)\s*\d+)",
                 caption, re.I)
    if m:
        nummer = re.sub(r"\s+", r"\\s*", re.escape(m.group(1)))
        muster = re.compile(r"[^.!?]*\b" + nummer + r"\b[^.!?]*[.!?]?", re.I)
        for q in absaetze:
            if q is p or _ist_caption(q, styles):
                continue
            for s in muster.findall(_text(q)):
                s = s.strip()
                if s and s not in verweise:
                    verweise.append(s)
            if len(verweise) >= 3:
                break

    zeilen = []
    if titel:
        zeilen.append(f"Dokument: {titel}")
    if teil_label != "Text":
        zeilen.append(f"Position: {teil_label}")
    if pfad:
        zeilen.append("Abschnitt: " + " > ".join(pfad[k] for k in sorted(pfad)))
    if zelle is not None:
        zeile_el = zelle.getparent()
        zellen = [_text(tc) for tc in zeile_el.findall("w:tc", NS)] if zeile_el is not None else []
        zellen = [z for z in zellen if z]
        if zellen:
            zeilen.append("Tabellenzeile: " + " | ".join(zellen))
        # Kopfzeile der Tabelle
        tbl = zeile_el.getparent() if zeile_el is not None else None
        if tbl is not None:
            erste = tbl.find("w:tr", NS)
            if erste is not None and erste is not zeile_el:
                kopf = [z for z in (_text(tc) for tc in erste.findall("w:tc", NS)) if z]
                if kopf:
                    zeilen.append("Tabellenkopf: " + " | ".join(kopf))
    if caption:
        zeilen.append(f"Bildunterschrift: {caption}")
    if eigener:
        zeilen.append(f"Text im Absatz des Bildes: {eigener}")
    if davor:
        zeilen.append(f"Absatz davor: {davor}")
    if danach:
        zeilen.append(f"Absatz danach: {danach}")
    for v in verweise[:3]:
        zeilen.append(f"Verweis im Text: {v}")
    kontext = "\n".join(zeilen)
    if len(kontext) > MAX_CONTEXT_CHARS:
        kontext = kontext[:MAX_CONTEXT_CHARS - 1].rstrip() + "…"
    return kontext, caption, ort, abschnitt


def validiere_docx(docx_path: str) -> None:
    """Schnelle Vorpruefung direkt im Upload-Request (vor dem Anlegen von
    Datenbankzeilen): Zip gueltig, Grenzen eingehalten, echtes DOCX ohne
    Makros. Wirft DocxFehler mit nutzertauglicher Meldung (-> HTTP 400)."""
    try:
        with zipfile.ZipFile(docx_path) as zf:
            _pruefe_zip(zf)
            _inhaltstyp_pruefen(zf)
    except zipfile.BadZipFile:
        raise DocxFehler("Die Datei ist keine gültige Word-Datei (.docx). "
                         "Alte .doc-Dateien bitte in Word als .docx speichern.")


# ---------------------------------------------------------------- Hauptfunktion
def _teile_in_reihenfolge(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(partname, label) — Kopfzeilen, Hauptdokument, Fusszeilen."""
    namen = zf.namelist()
    kopf = sorted(n for n in namen if re.fullmatch(r"word/header\d*\.xml", n))
    fuss = sorted(n for n in namen if re.fullmatch(r"word/footer\d*\.xml", n))
    return [(n, "Kopfzeile") for n in kopf] + [("word/document.xml", "Text")] + [(n, "Fußzeile") for n in fuss]


def analysiere_docx(docx_path: str, output_dir: str | None = None, praefix: str = "docx") -> DocxErgebnis:
    """Liest alle Bildvorkommen. Mit output_dir werden die Bilddateien dorthin
    gespeichert (PNG/JPEG unveraendert, andere Rasterformate nach PNG gewandelt)."""
    erg = DocxErgebnis()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        zf = zipfile.ZipFile(docx_path)
    except zipfile.BadZipFile:
        raise DocxFehler("Die Datei ist keine gültige Word-Datei (.docx). "
                         "Alte .doc-Dateien bitte in Word als .docx speichern.")
    with zf:
        _pruefe_zip(zf)
        _inhaltstyp_pruefen(zf)
        styles = _styles(zf)
        erg.titel = _dokumenttitel(zf)
        gespeichert: dict[str, tuple[str, int, int]] = {}   # hash -> (pfad, w, h)
        order = 0
        for part, label in _teile_in_reihenfolge(zf):
            root = _lese_xml(zf, part)
            rels = _rels(zf, part)
            absaetze = _absaetze_des_teils(root)
            index_von = {id(p): i for i, p in enumerate(absaetze)}
            if part == "word/document.xml":
                erg.volltext_zeichen = sum(len(_text(p)) for p in absaetze)
                erg.ueberschriften = [_text(p) for p in absaetze
                                      if _heading_level(_pstyle(p), styles) is not None and _text(p)]
            for drawing in root.iter(f"{{{NS['w']}}}drawing"):
                container = drawing.find("wp:inline", NS)
                anchored = False
                if container is None:
                    container = drawing.find("wp:anchor", NS); anchored = True
                if container is None:
                    continue
                docpr = container.find("wp:docPr", NS)
                blip = container.find(".//a:blip", NS)
                if docpr is None:
                    continue
                if blip is None:
                    # Diagramm, SmartArt, Form ohne Bild -> Stufe 2
                    erg.uebersprungen.append({"anker": f"{part}|{docpr.get('id')}",
                                              "grund": "kein Rasterbild (Diagramm, SmartArt oder Form)"})
                    continue
                rid = blip.get(f"{{{NS['r']}}}embed") or blip.get(f"{{{NS['r']}}}link")
                media = rels.get(rid or "")
                if not media or media not in zf.namelist():
                    erg.uebersprungen.append({"anker": f"{part}|{docpr.get('id')}",
                                              "grund": "Bilddaten nicht im Dokument (externer Link)"})
                    continue
                try:
                    docpr_id = int(docpr.get("id"))
                except (TypeError, ValueError):
                    continue
                deko = False
                for ext in docpr.findall("a:extLst/a:ext", NS):
                    d = ext.find("adec:decorative", NS)
                    if d is not None and d.get("val") in ("1", "true"):
                        deko = True
                order += 1
                if order > MAX_IMAGES:
                    erg.warnungen.append(f"Mehr als {MAX_IMAGES} Bilder – Rest übersprungen.")
                    break
                daten = zf.read(media)
                h = hashlib.sha256(daten).hexdigest()
                ext_ = os.path.splitext(media)[1].lower()
                bild = DocxBild(anker=f"{part}|{docpr_id}", part=part, docpr_id=docpr_id,
                                name=docpr.get("name", ""), original_alt=(docpr.get("descr") or "").strip(),
                                original_title=(docpr.get("title") or "").strip(), decorative=deko,
                                media_part=media, media_ext=ext_, hash=h, order=order, abschnitt=1,
                                ort=label, anchored=anchored)
                bild.context, bild.caption, bild.ort, bild.abschnitt = _kontext_fuer(
                    drawing, absaetze, index_von, styles, label, erg.titel)
                if ext_ in VEKTOR_EXT:
                    bild.unsupported = f"Vektorgrafik ({ext_[1:].upper()}) – wird noch nicht unterstützt"
                    erg.uebersprungen.append({"anker": bild.anker, "grund": bild.unsupported})
                    erg.bilder.append(bild)
                    continue
                # Bilddatei speichern (einmal pro Hash)
                if h in gespeichert:
                    bild.image_path, bild.width, bild.height = gespeichert[h]
                else:
                    try:
                        with Image.open(io.BytesIO(daten)) as im:
                            im.load()
                            bild.width, bild.height = im.size
                            if output_dir:
                                if ext_ in (".png", ".jpg", ".jpeg") :
                                    ziel = os.path.join(output_dir, f"{praefix}_{order:04d}{'.jpg' if ext_ in ('.jpg', '.jpeg') else '.png'}")
                                    with open(ziel, "wb") as f:
                                        f.write(daten)
                                else:
                                    ziel = os.path.join(output_dir, f"{praefix}_{order:04d}.png")
                                    im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(ziel, "PNG")
                                bild.image_path = ziel
                    except Exception as e:  # kaputtes oder unbekanntes Bild
                        bild.unsupported = f"Bilddaten nicht lesbar ({type(e).__name__})"
                        erg.uebersprungen.append({"anker": bild.anker, "grund": bild.unsupported})
                        erg.bilder.append(bild)
                        continue
                    gespeichert[h] = (bild.image_path, bild.width, bild.height)
                erg.bilder.append(bild)
    return erg


def extract_images_from_docx(docx_path: str, output_dir: str, project_id: int) -> list[dict]:
    """Rueckgabe im selben Muster wie pdf_processor.extract_images_from_pdf, damit
    der Upload-Pfad die Datensaetze unveraendert in `images` schreiben kann.
    Nicht verarbeitbare Bilder (Vektor, kaputt) werden NICHT zurueckgegeben;
    sie stehen in analysiere_docx().uebersprungen (fuer die Nutzer-Meldung)."""
    erg = analysiere_docx(docx_path, output_dir, praefix=f"p{project_id}")
    out = []
    for b in erg.bilder:
        if b.unsupported or not b.image_path:
            continue
        out.append({
            "page_number": b.abschnitt,      # Word hat keine Seiten: Abschnitt (Ueberschrift 1)
            "image_index": b.order,
            "image_path": b.image_path,
            "width": b.width,
            "height": b.height,
            "context_text": b.context,
            "original_alt": b.original_alt,
            "original_title": b.original_title,
            "decorative_hint": b.decorative,
            "docx_anker": b.anker,
            "hash": b.hash,
            "ort": b.ort,
            "caption": b.caption,
        })
    return out


if __name__ == "__main__":     # Schnellprobe: python3 docx_processor.py datei.docx
    import json, sys, tempfile
    erg = analysiere_docx(sys.argv[1], tempfile.mkdtemp())
    print("Titel:", erg.titel, "| Ueberschriften:", erg.ueberschriften)
    for b in erg.bilder:
        print(f"\n#{b.order} {b.anker} ort={b.ort} abschnitt={b.abschnitt} anchored={b.anchored} "
              f"deko={b.decorative} {b.width}x{b.height} {b.media_part} alt={b.original_alt!r} "
              f"title={b.original_title!r} unsupported={b.unsupported!r}\n{b.context}")
    print("\nUebersprungen:", json.dumps(erg.uebersprungen, ensure_ascii=False))
