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

ALTE VML-BILDER (Haertetest 27.08.2026)
  Dokumente aus dem Kompatibilitaetsmodus (aus .doc gewandelt, alte Vorlagen)
  tragen Bilder als <w:pict><v:shape ...><v:imagedata r:id=".."/></v:shape>.
  Der Alt-Text steht dort im Attribut alt des Shapes, der Titel in title. Ein
  Dekorativ-Kennzeichen kennt dieses Altformat nicht. Solche Bilder werden
  gleichwertig gelesen und zurueckgeschrieben; Anker "<part>|v:<shape-id>".

VERSCHACHTELUNG UND DUPLIKATE (Haertetest 27.08.2026)
  - Ein Bild IN einem Textfeld: Word schreibt Textfeld-Drawing > txbxContent >
    Bild-Drawing. Nur der a:blip des Bildes selbst zaehlt (_eigener_blip), das
    Textfeld ist kein Bild.
  - Word schreibt Textfelder doppelt: mc:Choice (modern) und mc:Fallback (VML
    fuer alte Word-Versionen) — mit DENSELBEN docPr-ids. Beim Lesen wird
    mc:Fallback uebersprungen, beim Schreiben bekommen beide Haelften den
    Alt-Text (sonst zeigt eine alte Word-Version den alten Stand).
  - Doppelte docPr-ids ausserhalb von Fallback (kaputte Dokumente nach
    Kopieren/Einfuegen) bekommen Anker "<part>|<id>#<n>" fuer das n-te Vorkommen.

SEITEN (Steve 27.08.2026: "wenn moeglich dieselben Seiten wie im Dokument")
  Word speichert beim Sichern <w:lastRenderedPageBreak/> ueberall dort, wo beim
  letzten Anzeigen eine neue Seite begann. Damit laesst sich die Seite jedes
  Bildes bestimmen, wie Word sie zuletzt gezeigt hat — ohne Rendern. Fehlen
  diese Marken (Datei nicht von Word gesichert), zaehlen manuelle Umbrueche
  (<w:br w:type="page"/>, w:pageBreakBefore, Abschnittswechsel ausser
  "continuous"). Gibt es gar keine Marken, bleibt die Einheit ABSCHNITT
  (Ueberschrift 1) wie bisher; der Aufrufer erfaehrt das ueber
  DocxErgebnis.seiten_bekannt bzw. "docx_einheit" je Bild.

UEBERSPRUNGENE ELEMENTE (Steve 27.08.2026: "in der Oberflaeche anzeigen")
  Alles, was wie ein Bild aussieht, aber in Stufe 1 nicht bearbeitet wird,
  landet mit Art, Ort und Seite/Abschnitt in DocxErgebnis.uebersprungen:
  diagramm, smartart, textfeld, form, gruppe, vektor (EMF/WMF/SVG), extern
  (verknuepft, nicht in der Datei), ole (eingebettetes Objekt, z. B. Excel),
  unlesbar, bild_ohne_daten. main.py speichert die Liste je Dokument
  (documents.hinweise, JSON), die Oberflaeche zeigt sie ueber den Bildern.

ANKER
  Jedes gefundene Bild bekommt einen stabilen Anker "<partname>|<docPr-id>"
  (z. B. "word/document.xml|7"; VML: "word/document.xml|v:_x0000_i1025").
  Damit findet docx_export.py genau dieses Bildelement wieder, ohne auf
  Reihenfolge oder Dateinamen angewiesen zu sein.
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
    "o": "urn:schemas-microsoft-com:office:office",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
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
    vml: bool = False           # altes VML-Bild (w:pict/v:shape) statt DrawingML
    seite: int = 1              # Seite laut Word-Marken (nur gueltig, wenn seiten_bekannt)


@dataclass
class DocxErgebnis:
    bilder: list[DocxBild] = field(default_factory=list)
    titel: str = ""
    ueberschriften: list[str] = field(default_factory=list)
    volltext_zeichen: int = 0
    uebersprungen: list[dict] = field(default_factory=list)   # {anker, art, grund, name, format, ort, abschnitt, seite}
    warnungen: list[str] = field(default_factory=list)
    seiten_bekannt: bool = False    # True, wenn das Dokument Seitenmarken traegt
    seiten_quelle: str = ""         # "word" (lastRenderedPageBreak) | "umbrueche" | ""


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


# ---------------------------------------------------------------- Bildelemente finden
def _in_fallback(el: etree._Element) -> bool:
    """Steht das Element in einem mc:Fallback? (Duplikat des mc:Choice-Inhalts
    fuer alte Word-Versionen — beim Lesen ueberspringen.)"""
    e = el.getparent()
    while e is not None:
        if e.tag == f"{{{NS['mc']}}}Fallback":
            return True
        e = e.getparent()
    return False


def _eigener_blip(container: etree._Element) -> etree._Element | None:
    """Der a:blip des Bildes SELBST — nicht der eines Bildes, das in einem
    Textfeld (w:txbxContent) innerhalb dieses Elements steckt."""
    for blip in container.iter(f"{{{NS['a']}}}blip"):
        e = blip.getparent()
        innen = False
        while e is not None and e is not container:
            if e.tag == f"{{{NS['w']}}}txbxContent":
                innen = True
                break
            e = e.getparent()
        if not innen:
            return blip
    return None


def _drawing_kennungen(root: etree._Element) -> dict[int, tuple[etree._Element, str]]:
    """id(w:drawing) -> (w:drawing, Kennung fuer den Anker). Das Element wird im
    Wert festgehalten, weil lxml-Proxys sonst freigegeben und ihre id() neu
    vergeben werden. Kennung: normal die docPr-id; bei
    doppelten ids ausserhalb von mc:Fallback "<id>#<n>" ab dem 2. Vorkommen.
    Drawings in mc:Fallback bekommen KEINE Kennung (werden nicht gelesen).
    docx_export nutzt dieselbe Funktion, damit Lesen und Schreiben uebereinstimmen."""
    out: dict[int, tuple[etree._Element, str]] = {}
    zaehler: dict[int, int] = {}
    for drawing in root.iter(f"{{{NS['w']}}}drawing"):
        if _in_fallback(drawing):
            continue
        container = drawing.find("wp:inline", NS)
        if container is None:
            container = drawing.find("wp:anchor", NS)
        if container is None:
            continue
        docpr = container.find("wp:docPr", NS)
        if docpr is None:
            continue
        try:
            did = int(docpr.get("id"))
        except (TypeError, ValueError):
            continue
        n = zaehler.get(did, 0) + 1
        zaehler[did] = n
        out[id(drawing)] = (drawing, str(did) if n == 1 else f"{did}#{n}")
    return out


def _vml_bilder(root: etree._Element) -> list[tuple[etree._Element, etree._Element, etree._Element, str]]:
    """Alle VML-Bilder eines Teils in Dokumentreihenfolge als
    (w:pict, Shape, v:imagedata, Kennung). Kennung = Shape-id (Word:
    "_x0000_i1025"), ersatzweise oder bei Doppelung "#<laufende Nummer>".
    mc:Fallback wird uebersprungen; w:object (OLE, z. B. eingebettetes
    Excel-Diagramm) ist kein w:pict und damit kein Bild.
    docx_export nutzt dieselbe Funktion."""
    out = []
    n = 0
    gesehen: set[str] = set()
    for pict in root.iter(f"{{{NS['w']}}}pict"):
        if _in_fallback(pict):
            continue
        n += 1
        shape = imagedata = None
        for kind in pict:
            if not isinstance(kind.tag, str) or not kind.tag.startswith("{" + NS["v"] + "}"):
                continue
            idata = kind.find("v:imagedata", NS)
            if idata is not None:
                shape, imagedata = kind, idata
                break
        if shape is None:
            continue
        sid = (shape.get("id") or shape.get(f"{{{NS['o']}}}spid") or "").strip()
        if not sid or "|" in sid or sid in gesehen:
            sid = f"#{n}"
        gesehen.add(sid)
        out.append((pict, shape, imagedata, sid))
    return out


def _seiten_der_bilder(root: etree._Element) -> tuple[dict[int, tuple[etree._Element, int]], str]:
    """Seite je w:drawing/w:pict im Hauptdokument, wie Word sie zuletzt gezeigt
    hat. Rueckgabe: ({id(el): (el, seite)}, quelle) mit quelle "word"
    (lastRenderedPageBreak), "umbrueche" (nur manuelle Umbrueche) oder ""
    (keine Marken -> Seite unbekannt). Elemente werden im Wert festgehalten
    (lxml-Proxys)."""
    W = NS["w"]
    t_lrpb, t_br, t_pbb = f"{{{W}}}lastRenderedPageBreak", f"{{{W}}}br", f"{{{W}}}pageBreakBefore"
    t_sect, t_ppr, t_p = f"{{{W}}}sectPr", f"{{{W}}}pPr", f"{{{W}}}p"
    t_type, t_val = f"{{{W}}}type", f"{{{W}}}val"
    bild_tags = {f"{{{W}}}drawing", f"{{{W}}}pict"}
    hat_lrpb = root.find(f".//{t_lrpb}") is not None
    seite, offen, explizit = 1, 0, 0
    out: dict[int, tuple[etree._Element, int]] = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag == t_p and offen:          # Abschnittswechsel wirkt ab dem naechsten Absatz
            seite += offen
            offen = 0
        if hat_lrpb:
            if el.tag == t_lrpb and not _in_fallback(el):
                seite += 1
        elif el.tag == t_br and el.get(t_type) == "page" and not _in_fallback(el):
            seite += 1
            explizit += 1
        elif el.tag == t_pbb and el.get(t_val, "1") not in ("0", "false") \
                and el.getparent() is not None and el.getparent().tag == t_ppr:
            seite += 1
            explizit += 1
        elif el.tag == t_sect and el.getparent() is not None and el.getparent().tag == t_ppr:
            typ = el.find(f"{{{W}}}type")
            if typ is None or typ.get(t_val) != "continuous":
                offen += 1
                explizit += 1
        if el.tag in bild_tags:
            out[id(el)] = (el, seite)
    quelle = "word" if hat_lrpb else ("umbrueche" if explizit else "")
    return out, quelle


_URI_ART = (("/chart", "diagramm"), ("/diagram", "smartart"), ("wordprocessingGroup", "gruppe"),
            ("wordprocessingShape", "form"), ("/picture", "bild_ohne_daten"))
_ART_GRUND = {"diagramm": "Diagramm", "smartart": "SmartArt", "textfeld": "Textfeld", "form": "Form",
              "gruppe": "Gruppe", "vektor": "Vektorgrafik", "extern": "Verknüpftes Bild (nicht in der Datei)",
              "ole": "Eingebettetes Objekt", "unlesbar": "Bild nicht lesbar", "bild_ohne_daten": "Bild ohne Bilddaten"}


def _art_ohne_bild(container: etree._Element) -> str:
    """Was ist ein Drawing ohne eigenes Rasterbild? (fuer die Nutzer-Meldung)"""
    gd = container.find("a:graphic/a:graphicData", NS)
    uri = gd.get("uri", "") if gd is not None else ""
    for teil, art in _URI_ART:
        if teil in uri:
            if art == "form" and container.find(".//w:txbxContent", NS) is not None:
                return "textfeld"
            return art
    return "form"


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
            seiten: dict[int, tuple[etree._Element, int]] = {}
            if part == "word/document.xml":
                erg.volltext_zeichen = sum(len(_text(p)) for p in absaetze)
                erg.ueberschriften = [_text(p) for p in absaetze
                                      if _heading_level(_pstyle(p), styles) is not None and _text(p)]
                seiten, erg.seiten_quelle = _seiten_der_bilder(root)
                erg.seiten_bekannt = bool(erg.seiten_quelle)
            drawing_kennung = _drawing_kennungen(root)
            vml = {id(p): (p, shape, idata, sid) for p, shape, idata, sid in _vml_bilder(root)}

            def _uebersprungen(anker: str, art: str, el: etree._Element, name: str = "", fmt: str = "") -> None:
                _k, _c, ort_, abschnitt_ = _kontext_fuer(el, absaetze, index_von, styles, label, erg.titel)
                grund = _ART_GRUND.get(art, art) + (f" ({fmt})" if fmt else "")
                if art in ("diagramm", "smartart", "vektor", "ole"):
                    grund += " – wird noch nicht unterstützt"
                elif art in ("textfeld", "form", "gruppe"):
                    grund += " – enthält kein eigenes Bild"
                erg.uebersprungen.append({"anker": anker, "art": art, "grund": grund, "name": name,
                                          "format": fmt, "ort": ort_, "abschnitt": abschnitt_,
                                          "seite": seiten[id(el)][1] if id(el) in seiten else 1})

            # Eingebettete Objekte (w:object, z. B. Excel-Diagramm als OLE) sind keine Bilder
            for n_obj, obj in enumerate(root.iter(f"{{{NS['w']}}}object"), start=1):
                if _in_fallback(obj):
                    continue
                ole = obj.find(f"{{{NS['o']}}}OLEObject")
                progid = (ole.get("ProgID") or "") if ole is not None else ""
                _uebersprungen(f"{part}|o:{n_obj}", "ole", obj, fmt=progid.split(".")[0] if progid else "")
            for el in root.iter(f"{{{NS['w']}}}drawing", f"{{{NS['w']}}}pict"):
                anchored = False
                deko = False
                ist_vml = el.tag == f"{{{NS['w']}}}pict"
                if ist_vml:
                    if id(el) not in vml:
                        continue            # kein Bild (Form, Textfeld, Fallback)
                    _p, shape, idata, sid = vml[id(el)]
                    anker = f"{part}|v:{sid}"
                    docpr_id = 0
                    name = shape.get("id") or ""
                    original_alt = (shape.get("alt") or "").strip()
                    original_title = (shape.get("title") or "").strip()
                    anchored = "position:absolute" in (shape.get("style") or "")
                    rid = idata.get(f"{{{NS['r']}}}id") or idata.get(f"{{{NS['r']}}}href")
                else:
                    eintrag = drawing_kennung.get(id(el))
                    if eintrag is None:
                        continue            # mc:Fallback-Duplikat oder ohne docPr
                    kennung = eintrag[1]
                    container = el.find("wp:inline", NS)
                    if container is None:
                        container = el.find("wp:anchor", NS)
                        anchored = True
                    docpr = container.find("wp:docPr", NS)
                    anker = f"{part}|{kennung}"
                    blip = _eigener_blip(container)
                    if blip is None:
                        # Diagramm, SmartArt, Form, Textfeld ohne eigenes Bild -> Stufe 2
                        _uebersprungen(anker, _art_ohne_bild(container), el, name=docpr.get("name", ""))
                        continue
                    docpr_id = int(kennung.split("#", 1)[0])
                    name = docpr.get("name", "")
                    original_alt = (docpr.get("descr") or "").strip()
                    original_title = (docpr.get("title") or "").strip()
                    for ext in docpr.findall("a:extLst/a:ext", NS):
                        d = ext.find("adec:decorative", NS)
                        if d is not None and d.get("val") in ("1", "true"):
                            deko = True
                    rid = blip.get(f"{{{NS['r']}}}embed") or blip.get(f"{{{NS['r']}}}link")
                media = rels.get(rid or "")
                if not media or media not in zf.namelist():
                    _uebersprungen(anker, "extern", el, name=name)
                    continue
                order += 1
                if order > MAX_IMAGES:
                    erg.warnungen.append(f"Mehr als {MAX_IMAGES} Bilder – Rest übersprungen.")
                    break
                daten = zf.read(media)
                h = hashlib.sha256(daten).hexdigest()
                ext_ = os.path.splitext(media)[1].lower()
                bild = DocxBild(anker=anker, part=part, docpr_id=docpr_id,
                                name=name, original_alt=original_alt,
                                original_title=original_title, decorative=deko,
                                media_part=media, media_ext=ext_, hash=h, order=order, abschnitt=1,
                                ort=label, anchored=anchored, vml=ist_vml)
                bild.context, bild.caption, bild.ort, bild.abschnitt = _kontext_fuer(
                    el, absaetze, index_von, styles, label, erg.titel)
                bild.seite = seiten[id(el)][1] if id(el) in seiten else 1   # Kopf-/Fusszeile: Seite 1
                if ext_ in VEKTOR_EXT:
                    bild.unsupported = f"Vektorgrafik ({ext_[1:].upper()}) – wird noch nicht unterstützt"
                    _uebersprungen(bild.anker, "vektor", el, name=name, fmt=ext_[1:].upper())
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
                                if ext_ in (".png", ".jpg", ".jpeg"):
                                    ziel = os.path.join(output_dir, f"{praefix}_{order:04d}{'.jpg' if ext_ in ('.jpg', '.jpeg') else '.png'}")
                                    with open(ziel, "wb") as f:
                                        f.write(daten)
                                else:
                                    ziel = os.path.join(output_dir, f"{praefix}_{order:04d}.png")
                                    im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(ziel, "PNG")
                                bild.image_path = ziel
                    except Exception as e:  # kaputtes oder unbekanntes Bild
                        bild.unsupported = f"Bilddaten nicht lesbar ({type(e).__name__})"
                        _uebersprungen(bild.anker, "unlesbar", el, name=name, fmt=type(e).__name__)
                        erg.bilder.append(bild)
                        continue
                    gespeichert[h] = (bild.image_path, bild.width, bild.height)
                erg.bilder.append(bild)
    return erg


def extract_docx(docx_path: str, output_dir: str, project_id: int) -> tuple[list[dict], dict]:
    """(Bilder wie extract_images_from_docx, Hinweise fuer documents.hinweise):
    {"uebersprungen": [...], "warnungen": [...], "seiten": "word"|"umbrueche"|""}."""
    erg = analysiere_docx(docx_path, output_dir, praefix=f"p{project_id}")
    return _bilder_als_dicts(erg), {"uebersprungen": erg.uebersprungen, "warnungen": erg.warnungen,
                                   "seiten": erg.seiten_quelle}


def extract_images_from_docx(docx_path: str, output_dir: str, project_id: int) -> list[dict]:
    """Rueckgabe im selben Muster wie pdf_processor.extract_images_from_pdf, damit
    der Upload-Pfad die Datensaetze unveraendert in `images` schreiben kann.
    Nicht verarbeitbare Bilder (Vektor, kaputt) werden NICHT zurueckgegeben;
    sie stehen in analysiere_docx().uebersprungen (fuer die Nutzer-Meldung)."""
    return _bilder_als_dicts(analysiere_docx(docx_path, output_dir, praefix=f"p{project_id}"))


def _bilder_als_dicts(erg: DocxErgebnis) -> list[dict]:
    out = []
    for b in erg.bilder:
        if b.unsupported or not b.image_path:
            continue
        out.append({
            # Seite wie zuletzt in Word gezeigt (Seitenmarken), sonst Abschnitt (Ueberschrift 1)
            "page_number": b.seite if erg.seiten_bekannt else b.abschnitt,
            "docx_einheit": "seite" if erg.seiten_bekannt else "abschnitt",
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
    print("Titel:", erg.titel, "| Ueberschriften:", erg.ueberschriften, "| Seiten:", erg.seiten_quelle or "unbekannt")
    for b in erg.bilder:
        print(f"\n#{b.order} {b.anker} ort={b.ort} abschnitt={b.abschnitt} seite={b.seite} anchored={b.anchored} vml={b.vml} "
              f"deko={b.decorative} {b.width}x{b.height} {b.media_part} alt={b.original_alt!r} "
              f"title={b.original_title!r} unsupported={b.unsupported!r}\n{b.context}")
    print("\nUebersprungen:", json.dumps(erg.uebersprungen, ensure_ascii=False))
