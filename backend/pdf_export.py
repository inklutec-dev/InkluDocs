"""
PDF Export Module for InkluDocs (Beta).

Contains the PDF/UA-compliant alt-text tagging functionality.
Moved from pdf_processor.py to keep the main processing flow clean.
This is the "beta" PDF tagging feature - importable but separated.

Seit 12.06.2026 zusaetzlich: finalize_export_pdf() — gemeinsamer
Abschluss-Schritt fuer BEIDE Export-Pfade (PDFix und PyMuPDF/fitz).
Setzt Dokumentsprache + Titel und entfernt verwaiste Alt-Text-Altlasten.
"""

import os
import re
import fitz  # PyMuPDF

# Dokument-Eigenschaften der exportierten PDF: Creator (Anwendung) und Producer (erstellt mit).
# EINE Quelle fuer alle PDF-Ausgaenge (Alt-Text-Export beide Wege, Formular-Export,
# barrierefreie PDF aus Word). Author, Subject, Keywords, CreationDate bleiben die des
# Autors. Regel: In den Eigenschaften steht immer nur unser Produktname — nie ein
# Werkzeug oder eine Bibliothek, mit der die Datei technisch geschrieben wurde.
# Aenderbar ueber Umgebung, ohne Code anzufassen.
PDF_CREATOR = os.environ.get("INKLUDOCS_PDF_CREATOR", "inkludocs.de")
PDF_PRODUCER = os.environ.get("INKLUDOCS_PDF_PRODUCER", "InkluDocs")
# Schalter (Michael Karbe 14.09.2026): INKLUDOCS_PDF_METADATEN=aus laesst Creator/Producer der Datei
# unangetastet — auf Staging, um zu pruefen, was LibreOffice/PDFix/PyMuPDF von sich aus eintragen.
# Vorgabe „an“ (Prod: immer nur unser Produktname).
PDF_METADATEN_SETZEN = (os.environ.get("INKLUDOCS_PDF_METADATEN", "an") or "an").strip().lower() \
    not in ("aus", "0", "false", "nein", "off")
CREATOR_MAXLAENGE = 100


def creator_normieren(creator: str | None) -> str | None:
    """Vom Konto hinterlegter Ersteller: eine Zeile, max. 100 Zeichen; leer = Vorgabe."""
    if not creator:
        return None
    t = re.sub(r"[\x00-\x1f\x7f]+", " ", str(creator)).strip()
    return t[:CREATOR_MAXLAENGE] or None


def dokumentinfo_werte(verfahren: str | None = None, creator: str | None = None) -> dict:
    """Creator/Producer fuer eine exportierte PDF — fuer alle Wege dieselben Werte.
    `verfahren` (pdfix/fitz/libreoffice) wird von den Aufrufern weiter mitgegeben, hat aber
    keinen Einfluss auf den Wortlaut: Es steht immer nur unser Produktname in der Datei.
    `creator`: vom Konto hinterlegter Ersteller (Michael Karbe 14.09.2026, Kunde Jens: eigenes
    Unternehmen als Creator) — ersetzt die Vorgabe; der Producer bleibt InkluDocs."""
    return {"creator": creator_normieren(creator) or PDF_CREATOR, "producer": PDF_PRODUCER}


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def xmp_dokumentinfo(xmp: str, werte: dict) -> str:
    """xmp:CreatorTool und pdf:Producer in einem vorhandenen XMP-Paket ersetzen — alles
    andere (dc:title, pdfuaid:part, Datumsangaben ...) bleibt unangetastet. Viewer wie Acrobat
    zeigen bei vorhandenem XMP dessen Werte statt des Info-Dictionarys; die Quell-PDF bringt
    dort oft ihr Erzeuger-Werkzeug mit. Beide Schreibweisen werden abgedeckt: als Element
    (<xmp:CreatorTool>…</xmp:CreatorTool>) und als Attribut (xmp:CreatorTool="…").
    Fehlt ein Eintrag, wird keiner erfunden. Ohne XMP kommt der Text unveraendert zurueck."""
    if not xmp:
        return xmp
    for feld, wert in (("CreatorTool", werte["creator"]), ("Producer", werte["producer"])):
        ersatz = _xml_escape(wert)
        xmp = re.sub(rf"(<([\w\-]+:){feld}(?:\s[^>]*)?>)[^<]*(</\2{feld}>)",
                     lambda m: m.group(1) + ersatz + m.group(3), xmp)
        xmp = re.sub(rf"((?:^|\s)[\w\-]+:{feld}\s*=\s*)\"[^\"]*\"",
                     lambda m: m.group(1) + '"' + ersatz + '"', xmp)
    return xmp


def dokumentinfo_in_doc(doc: "fitz.Document", verfahren: str | None = None,
                        creator: str | None = None) -> dict:
    """Creator/Producer in ein geoeffnetes fitz-Dokument schreiben: Info-Dictionary UND —
    falls vorhanden — XMP-Paket. EINE Stelle fuer alle fitz-basierten Ausgaenge; das
    PDF/UA-Verfahren (pikepdf, Bytes) setzt dieselben Werte auf seinem Weg.
    Bei INKLUDOCS_PDF_METADATEN=aus wird nichts geschrieben; zurueck kommt, was in der Datei steht."""
    meta = doc.metadata or {}
    if not PDF_METADATEN_SETZEN:
        return {"creator": meta.get("creator") or "", "producer": meta.get("producer") or "",
                "unveraendert": True}
    werte = dokumentinfo_werte(verfahren, creator)
    meta["creator"] = werte["creator"]
    meta["producer"] = werte["producer"]
    doc.set_metadata(meta)
    try:
        xmp = doc.get_xml_metadata() or ""
    except Exception:  # noqa: BLE001 — kein/kaputtes XMP: Info-Dictionary reicht
        xmp = ""
    if xmp:
        neu = xmp_dokumentinfo(xmp, werte)
        if neu != xmp:
            doc.set_xml_metadata(neu)
    return werte


def setze_dokumentinfo(pdf_path: str, verfahren: str | None = None,
                       creator: str | None = None) -> dict:
    """Creator/Producer in eine fertige PDF schreiben (Info-Dictionary + XMP), Rest unangetastet.
    Entspricht Heines SetDocInfo-Skript (PutString auf dem Info-Objekt), hier mit PyMuPDF,
    damit alle Ausgaenge dieselbe Stelle nutzen."""
    doc = fitz.open(pdf_path)
    werte = dokumentinfo_in_doc(doc, verfahren, creator)
    if werte.get("unveraendert"):
        doc.close()
        return werte
    # INKREMENTELL speichern: Die Datei wird nur ergaenzt, das Original bleibt byteweise
    # erhalten (der Formular-Export garantiert das ausdruecklich — test_original_ist_praefix;
    # so bleiben Heines PDFix-Ausgabe und die Quickinfos unangetastet). Das PDF/UA-Verfahren
    # arbeitet auf Bytes und hat seinen eigenen Weg (pdfua_export.dokumentinfo_setzen).
    doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    return werte


def _escape_pdf_string(text: str) -> str:
    """Escape special characters for PDF string literals."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_string(text: str) -> str:
    """Encode text as PDF string, using UTF-16BE hex string for non-ASCII characters.
    This fixes the umlaut encoding issue (e.g. ue showing as Ã¼)."""
    if not text:
        return "()"
    try:
        text.encode('ascii')
        return f"({_escape_pdf_string(text)})"
    except UnicodeEncodeError:
        pass
    # UTF-16BE with BOM prefix for full Unicode support
    encoded = text.encode('utf-16-be')
    hex_str = encoded.hex().upper()
    return f"<FEFF{hex_str}>"


def _find_vector_graphic_range(content_str: str, bbox: tuple, page_height: float = 842.0, tolerance: float = 50) -> tuple:
    """
    Find the content stream range belonging to a vector graphic at the given bounding box.

    IMPORTANT: bbox is in fitz coordinates (top-left origin, y increases downward).
    PDF content stream uses bottom-left origin (y increases upward).
    We convert bbox y-coordinates: pdf_y = page_height - fitz_y

    Strategy 1: Look for q...Q blocks with cm transformations in the bbox area.
    Strategy 2: Look for path drawing commands (m, l, c, re) with coordinates in the bbox area.

    Returns (start_pos, end_pos) or (None, None) if not found.
    """
    fitz_x0, fitz_y0, fitz_x1, fitz_y1 = bbox
    # Convert to PDF coordinates (bottom-left origin)
    x0 = fitz_x0
    x1 = fitz_x1
    y0 = page_height - fitz_y1  # fitz bottom -> PDF bottom (lower y in PDF)
    y1 = page_height - fitz_y0  # fitz top -> PDF top (higher y in PDF)

    # Strategy 1: q-blocks with cm transformations (most common for placed graphics)
    cm_pattern = r'q\s+([0-9.\-]+ [0-9.\-]+ [0-9.\-]+ [0-9.\-]+ [0-9.\-]+ [0-9.\-]+) cm'
    matches = list(re.finditer(cm_pattern, content_str))

    blocks_in_range = []
    for m in matches:
        parts = m.group(1).split()
        if len(parts) >= 6:
            tx, ty = float(parts[4]), float(parts[5])
            if x0 - tolerance <= tx <= x1 + tolerance:
                if y0 - tolerance <= ty <= y1 + tolerance:
                    blocks_in_range.append(m.start())

    if blocks_in_range:
        start_pos = min(blocks_in_range)
        last_block_start = max(blocks_in_range)
        remaining = content_str[last_block_start:]
        q_match = re.search(r'Q\s*(?:\n|$)', remaining)
        if not q_match:
            q_match = re.search(r'Q', remaining)
        end_pos = last_block_start + q_match.end() if q_match else last_block_start + 100
        return start_pos, end_pos

    # Strategy 2: Find q...Q blocks containing path operations with matching coordinates
    q_blocks = list(re.finditer(r'q\s+(.*?)\s*Q', content_str, re.DOTALL))

    blocks_in_range = []
    for block in q_blocks:
        block_content = block.group(1)
        coords = re.findall(r'([0-9.\-]+) ([0-9.\-]+) (?:m|l|re)', block_content)
        for cx, cy in coords:
            try:
                px, py = float(cx), float(cy)
                if x0 - tolerance <= px <= x1 + tolerance and y0 - tolerance <= py <= y1 + tolerance:
                    blocks_in_range.append(block.start())
                    break
            except ValueError:
                continue

    if blocks_in_range:
        start_pos = min(blocks_in_range)
        last_start = max(blocks_in_range)
        for block in q_blocks:
            if block.start() == last_start:
                end_pos = block.end()
                return start_pos, end_pos
        return start_pos, last_start + 100

    return None, None


_BDC_RE = re.compile(r"/(\w+)\s*(<<[^>]*?/MCID\s+(\d+)[^>]*?>>|<<.*?>>|/\w+)?\s*BDC|\bBMC\b|\bEMC\b", re.DOTALL)


def layout_vektorbilder(pdf_path: str, images: list) -> set:
    """xrefs der Vektor-„Bilder“, die Seitenlayout sind (Regel pdf_processor._ist_seitenlayout: mindestens die halbe
    Seite UND Fotos darin). Fuer den Export von Projekten, die vor dem 14.09.2026 extrahiert wurden."""
    from pdf_processor import _ist_seitenlayout
    vektor = [i for i in images if i.get("is_vector") and i.get("bbox_x0") is not None]
    if not vektor:
        return set()
    doc = fitz.open(pdf_path)
    raster_je_seite: dict = {}
    treffer = set()
    try:
        for img in vektor:
            pn = int(img.get("page_number") or 1) - 1
            if pn < 0 or pn >= len(doc):
                continue
            page = doc[pn]
            if pn not in raster_je_seite:
                flaechen = []
                for info in page.get_images(full=True):
                    try:
                        flaechen.extend(page.get_image_rects(info[0]))
                    except Exception:
                        pass
                raster_je_seite[pn] = flaechen
            r = fitz.Rect(img["bbox_x0"], img["bbox_y0"], img["bbox_x1"], img["bbox_y1"])
            if _ist_seitenlayout(r, page.rect, raster_je_seite[pn]):
                treffer.add(img["xref"])
    finally:
        doc.close()
    return treffer


def _markierte_bereiche(content_str: str) -> list:
    """Alle Marked-Content-Bereiche eines Inhaltsstroms als (start, end, tag, mcid) — verschachtelt,
    innere Bereiche stehen mit kleinerem Umfang in der Liste. mcid ist None bei BMC oder BDC ohne MCID."""
    bereiche, stapel = [], []
    for m in _BDC_RE.finditer(content_str):
        text = m.group(0)
        if text.endswith("EMC"):
            if stapel:
                start, tag, mcid, bdc_ende = stapel.pop()
                bereiche.append((start, m.end(), tag, mcid, bdc_ende, m.start()))
        elif text.endswith("BMC"):
            stapel.append((m.start(), None, None, m.end()))
        else:
            mcid = int(m.group(3)) if m.group(3) else None
            stapel.append((m.start(), m.group(1), mcid, m.end()))
    return bereiche


def _umhuellung_planen(content_str: str, bereiche: list, start: int, end: int, mcid_start: int):
    """Plant das Figure-Tag um einen Bereich von Zeichenbefehlen in einem TAGGED Dokument so, dass die
    Marked-Content-Verschachtelung gueltig bleibt (14.09.2026, Prod-Dokument 430: der geschaetzte
    Bereich einer Vektorgrafik schnitt Artefakt-Bloecke an — veraPDF „tagged content inside Artifact“).

    Regeln: (1) Bereich an angeschnittenen Bloecken ausrichten (ganz hinein). (2) Artefakt-Marker
    innerhalb aufheben — die Grafik IST Inhalt, sonst gaebe es keinen Alt-Text. (3) Fremde getaggte
    Bloecke innerhalb (z. B. Beschriftungen einer Infografik als /P) bleiben unangetastet; getaggt werden
    nur die FREIEN Stuecke dazwischen — ein Figure darf mehrere Inhaltsstuecke (MCIDs) haben.
    (4) Liegt der Bereich in einem Artefakt-Block, wird dieser davor geschlossen und danach wieder
    geoeffnet. (5) Liegt er in fremder Struktur (z. B. in einem Absatz), wird nicht getaggt (None).
    Rueckgabe: (start, end, ersatztext, anzahl_mcids) oder None."""
    veraendert = True
    while veraendert:
        veraendert = False
        for b in bereiche:
            bs, be = b[0], b[1]
            if (bs < start < be < end) or (start < bs < end < be):
                start, end = min(start, bs), max(end, be)
                veraendert = True
    aussen = [b for b in bereiche if b[0] < start and end < b[1]]
    eltern_mcid = None
    fremd_aussen = [b for b in aussen if b[2] != "Artifact"]
    if fremd_aussen:
        # Bild liegt in fremder Struktur (z. B. in einem Absatz): das Figure wird KIND dieses Elements —
        # moeglich, wenn der Bereich selbst keine fremden Bloecke enthaelt und das Element eine MCID hat.
        innerst_fremd = min(fremd_aussen, key=lambda b: b[1] - b[0])
        if innerst_fremd[3] is None or any(b[2] != "Artifact" for b in bereiche if start <= b[0] and b[1] <= end):
            return None
        eltern_mcid = innerst_fremd[3]
        aussen = [b for b in aussen if b[0] > innerst_fremd[0]]   # nur Artefakte INNERHALB des Elternblocks
    innen = [b for b in bereiche if start <= b[0] and b[1] <= end]
    fremd = [b for b in innen if b[2] != "Artifact"]
    fremd_top = sorted(
        (b for b in fremd if not any(o is not b and o[0] <= b[0] and b[1] <= o[1] for o in fremd)),
        key=lambda b: b[0])
    artefakte = [b for b in innen if b[2] == "Artifact"
                 and not any(t[0] <= b[0] and b[1] <= t[1] for t in fremd_top)]

    def frei_stueck(a: int, z: int) -> str:
        seg = content_str[a:z]
        schnitte = []
        for b in artefakte:
            if a <= b[0] and b[1] <= z:
                schnitte.append((b[0] - a, b[4] - a))
                schnitte.append((b[5] - a, b[1] - a))
        for x, y in sorted(schnitte, reverse=True):
            seg = seg[:x] + seg[y:]
        return seg

    teile, n, pos = [], 0, start
    for t in fremd_top + [None]:
        z = t[0] if t else end
        seg = frei_stueck(pos, z)
        if seg.strip():
            teile.append(f"/Figure <</MCID {mcid_start + n}>> BDC\n{seg}\nEMC\n")
            n += 1
        else:
            teile.append(content_str[pos:z])   # nichts zu taggen: Stueck unveraendert lassen (Marker bleiben)
        if t:
            teile.append(content_str[t[0]:t[1]])
            pos = t[1]
    if n == 0:
        return None
    ersatz = "".join(teile)
    if aussen:
        innerst = min(aussen, key=lambda b: b[1] - b[0])
        bdc_text = content_str[innerst[0]:innerst[4]]
        ersatz = "EMC\n" + ersatz + bdc_text + "\n"
    return start, end, ersatz, n, eltern_mcid


def _figure_tags(doc: fitz.Document, struct_root_xref: int) -> set:
    """Alle Strukturtypen, die „Figure“ bedeuten: Figure selbst plus RoleMap-Eintraege darauf
    (InDesign schreibt z. B. /PlacedGraphic /Figure). 14.09.2026."""
    tags = {"Figure"}
    rm = doc.xref_get_key(struct_root_xref, "RoleMap")
    text = ""
    if rm[0] == "dict":
        text = rm[1]
    elif rm[0] == "xref":
        text = doc.xref_object(int(rm[1].split()[0]), compressed=True)
    for eigen, ziel in re.findall(r"/(\w+)\s*/(\w+)", text):
        if ziel in tags and eigen != "Type":
            tags.add(eigen)
    return tags


def _innerstes_figure(bereiche: list, start: int, end: int, figure_tags: set | None = None):
    """Kleinster Figure-Bereich (auch RoleMap-Typen) mit MCID, der [start, end] vollstaendig enthaelt."""
    figure_tags = figure_tags or {"Figure"}
    beste = None
    for b in bereiche:
        b_start, b_end, tag, mcid = b[0], b[1], b[2], b[3]
        if tag in figure_tags and mcid is not None and b_start <= start and end <= b_end:
            if beste is None or (b_end - b_start) < (beste[1] - beste[0]):
                beste = b
    return beste


def _parenttree_nums(doc: fitz.Document, node_xref: int, gefunden: dict, tiefe: int = 0) -> None:
    """Liest den ParentTree (Number-Tree, /Nums direkt oder ueber /Kids) in gefunden[key] = Rohwert."""
    if tiefe > 8:
        return
    kids = doc.xref_get_key(node_xref, "Kids")
    if kids[0] == "array":
        for ref in _XREF_REF_RE.findall(kids[1]):
            _parenttree_nums(doc, int(ref), gefunden, tiefe + 1)
    nums = doc.xref_get_key(node_xref, "Nums")
    if nums[0] == "array":
        text = nums[1].strip()[1:-1]
    elif nums[0] == "xref":
        text = doc.xref_object(int(nums[1].split()[0]), compressed=True).strip()[1:-1]
    else:
        return
    # Paare: <key> <wert>, wert = "n 0 R" oder inline "[ ... ]"
    pos = 0
    # PyMuPDF normalisiert ohne Leerzeichen: "[0[12 0 R]]" — deshalb \s* zwischen Schluessel und Wert
    token = re.compile(r"\s*(\d+)\s*(\d+\s+0\s+R|\[(?:[^\[\]]|\[[^\]]*\])*\])", re.DOTALL)
    while True:
        m = token.match(text, pos)
        if not m:
            break
        gefunden[int(m.group(1))] = m.group(2)
        pos = m.end()


def _parenttree_elemente(doc: fitz.Document, struct_root_xref: int, struct_parents: int) -> list:
    """Die StructElem-xrefs einer Seite nach MCID (Index = MCID; None fuer Luecken)."""
    pt = doc.xref_get_key(struct_root_xref, "ParentTree")
    if pt[0] != "xref":
        return []
    gefunden: dict = {}
    _parenttree_nums(doc, int(pt[1].split()[0]), gefunden)
    roh = gefunden.get(struct_parents)
    if roh is None:
        return []
    if not roh.startswith("["):
        # Referenz auf ein Array-Objekt (oder auf ein einzelnes Element bei Annotationen)
        obj = doc.xref_object(int(roh.split()[0]), compressed=True).strip()
        if not obj.startswith("["):
            return []
        roh = obj
    eintraege = []
    for t in re.finditer(r"(\d+)\s+0\s+R|null", roh[1:-1]):
        eintraege.append(int(t.group(1)) if t.group(1) else None)
    return eintraege


def _parenttree_anhaengen(doc: fitz.Document, pt_root_xref: int, struct_parents: int, refs: list) -> bool:
    """Haengt StructElem-Referenzen an das ParentTree-Array einer Seite an (Number-Tree mit /Nums direkt
    oder ueber /Kids; Array inline oder als eigenes Objekt). True, wenn der Eintrag gefunden wurde."""
    if not refs:
        return True
    neu = " ".join(f"{x} 0 R" for x in refs)
    stapel, gesehen = [pt_root_xref], set()
    while stapel:
        node = stapel.pop()
        if node in gesehen:
            continue
        gesehen.add(node)
        kids = doc.xref_get_key(node, "Kids")
        if kids[0] == "array":
            stapel.extend(int(r) for r in _XREF_REF_RE.findall(kids[1]))
        nums = doc.xref_get_key(node, "Nums")
        if nums[0] == "array":
            text, nums_obj = nums[1], None
        elif nums[0] == "xref":
            nums_obj = int(nums[1].split()[0]); text = doc.xref_object(nums_obj, compressed=True)
        else:
            continue
        # Paare SEQUENZIELL lesen (kein re.search: „1“ wuerde sonst mitten in „12 0 R“ gefunden)
        inner_start = text.find("[") + 1 if nums[0] == "array" or text.lstrip().startswith("[") else 0
        token = re.compile(r"\s*(\d+)\s*(\d+\s+0\s+R|\[(?:[^\[\]]|\[[^\]]*\])*\])", re.DOTALL)
        pos, treffer = inner_start, None
        while True:
            mt = token.match(text, pos)
            if not mt:
                break
            if int(mt.group(1)) == struct_parents:
                treffer = mt
                break
            pos = mt.end()
        if not treffer:
            continue
        m = treffer
        wert = m.group(2)
        if wert.startswith("["):
            ersatz = wert.rstrip()[:-1].rstrip() + " " + neu + " ]"
            text = text[:m.start(2)] + ersatz + text[m.end(2):]
            if nums_obj is None:
                doc.xref_set_key(node, "Nums", text)
            else:
                doc.update_object(nums_obj, text)
            return True
        ziel = int(wert.split()[0])
        obj = doc.xref_object(ziel, compressed=True).strip()
        if not obj.startswith("["):
            return False
        doc.update_object(ziel, obj.rstrip()[:-1].rstrip() + " " + neu + " ]")
        return True
    return False


def _mcid_index_aus_baum(doc: fitz.Document, erreichbar: set) -> dict:
    """{(page_xref, mcid): elem_xref} aus den /K-Eintraegen aller erreichbaren Elemente — Ersatz fuer
    fehlende ParentTree-Eintraege (Prod-Dokument 430: drei Fotos mit „null“ im ParentTree)."""
    index = {}
    for x in erreichbar:
        try:
            obj = doc.xref_object(x, compressed=True)
        except Exception:
            continue
        pg = re.search(r"/Pg\s+(\d+)\s+0\s+R", obj)
        k = doc.xref_get_key(x, "K")
        if k[0] == "int" and pg:
            index[(int(pg.group(1)), int(k[1]))] = x
        elif k[0] in ("array", "dict"):
            text = k[1]
            for m in re.finditer(r"<<[^<>]*?/MCID\s+(\d+)[^<>]*?>>", text):
                pgm = re.search(r"/Pg\s+(\d+)\s+0\s+R", m.group(0))
                seite = int(pgm.group(1)) if pgm else (int(pg.group(1)) if pg else None)
                if seite is not None:
                    index[(seite, int(m.group(1)))] = x
            if k[0] == "array" and pg:
                for m in re.finditer(r"(?<![\d/])(\d+)(?![\s\d]*0\s+R)", re.sub(r"<<.*?>>", " ", text)):
                    index.setdefault((int(pg.group(1)), int(m.group(1))), x)
    return index


def _parenttree_setzen(doc: fitz.Document, pt_root_xref: int, struct_parents: int, mcid: int, ref: int) -> bool:
    """Setzt im ParentTree-Array einer Seite den Eintrag fuer eine VORHANDENE MCID (null ersetzen oder
    am Ende anhaengen). Fuer Marked-Content-Bloecke, die im Original kein Strukturelement mehr haben."""
    stapel, gesehen = [pt_root_xref], set()
    token = re.compile(r"\s*(\d+)\s*(\d+\s+0\s+R|\[(?:[^\[\]]|\[[^\]]*\])*\])", re.DOTALL)
    while stapel:
        node = stapel.pop()
        if node in gesehen:
            continue
        gesehen.add(node)
        kids = doc.xref_get_key(node, "Kids")
        if kids[0] == "array":
            stapel.extend(int(r) for r in _XREF_REF_RE.findall(kids[1]))
        nums = doc.xref_get_key(node, "Nums")
        if nums[0] == "array":
            text, nums_obj = nums[1], None
        elif nums[0] == "xref":
            nums_obj = int(nums[1].split()[0]); text = doc.xref_object(nums_obj, compressed=True)
        else:
            continue
        pos, treffer = (text.find("[") + 1 if text.lstrip().startswith("[") else 0), None
        while True:
            mt = token.match(text, pos)
            if not mt:
                break
            if int(mt.group(1)) == struct_parents:
                treffer = mt
                break
            pos = mt.end()
        if not treffer:
            continue
        wert = treffer.group(2)
        if wert.startswith("["):
            eintraege = re.findall(r"\d+\s+0\s+R|null", wert[1:-1])
            if mcid < len(eintraege):
                if eintraege[mcid] != "null":
                    return False
                eintraege[mcid] = f"{ref} 0 R"
            elif mcid == len(eintraege):
                eintraege.append(f"{ref} 0 R")
            else:
                return False
            neu = "[ " + " ".join(eintraege) + " ]"
            text = text[:treffer.start(2)] + neu + text[treffer.end(2):]
            if nums_obj is None:
                doc.xref_set_key(node, "Nums", text)
            else:
                doc.update_object(nums_obj, text)
            return True
        ziel = int(wert.split()[0])
        obj = doc.xref_object(ziel, compressed=True).strip()
        if not obj.startswith("["):
            return False
        eintraege = re.findall(r"\d+\s+0\s+R|null", obj[1:-1])
        if mcid < len(eintraege):
            if eintraege[mcid] != "null":
                return False
            eintraege[mcid] = f"{ref} 0 R"
        elif mcid == len(eintraege):
            eintraege.append(f"{ref} 0 R")
        else:
            return False
        doc.update_object(ziel, "[ " + " ".join(eintraege) + " ]")
        return True
    return False


def _vorhandene_figures_uebernehmen(doc: fitz.Document, struct_root_xref: int, page_images: dict,
                                    doc_elem_xref: int | None = None) -> tuple:
    """Traegt Alt-Texte in VORHANDENE Figure-Elemente des Dokuments ein (wie der PDFix-Weg), statt neue
    Figure-Tags um die Zeichenbefehle zu legen.

    14.09.2026: Ein InDesign-Jahresbericht (Prod, Projekt 430) brachte 300 eigene Figure-Tags ohne Alt-Text
    mit; der Ersatzweg legte 234 neue Figures darueber — die leeren Original-Tags blieben (veraPDF 7.3),
    dazu Artefakt-Verschachtelungen durch die Umhuellung. Jetzt: Bild im Inhaltsstrom finden, das innerste
    umschliessende /Figure-BDC nehmen, ueber ParentTree (StructParents -> MCID) zum StructElem aufloesen,
    /Alt dort setzen. Inhaltsstrom bleibt unveraendert.

    Rueckgabe: (uebernommen: {(page_num, xref_bild): elem_xref}, warnungen, neu_bloecke: {page_xref: [elem, ...]}) —
    neu_bloecke sind die fuer verwaiste Figure-Bloecke angelegten Elemente, die der Aufrufer noch nach Seite
    in den Dokument-Knoten einhaengt."""
    uebernommen, warnungen, neu_bloecke = {}, [], {}
    # Nur Elemente, die vom StructTreeRoot aus erreichbar sind — ein Original-Tag, das selbst schon eine
    # Waise ist (InDesign-Altlast, Prod-Dokument 430: 1 von 300), bekommt wie bisher ein eigenes Tag.
    erreichbar = _collect_reachable_struct_elems(doc, struct_root_xref)
    figure_tags = _figure_tags(doc, struct_root_xref)
    index_baum = None   # erst bei Bedarf aufbauen (teuer)
    for page_num in sorted(page_images.keys()):
        page = doc[page_num]
        content = page.read_contents()
        if not content:
            continue
        content_str = content.decode("latin-1")
        sp = doc.xref_get_key(page.xref, "StructParents")
        if sp[0] != "int":
            continue
        elemente = _parenttree_elemente(doc, struct_root_xref, int(sp[1]))
        bereiche = _markierte_bereiche(content_str)
        if not bereiche:
            continue
        belegt: dict = {}
        for img_info in page_images[page_num]:
            if img_info["is_vector"]:
                if not img_info.get("bbox"):
                    continue
                start, end = _find_vector_graphic_range(content_str, img_info["bbox"], page.rect.height)
                if start is None or end is None:
                    continue
            else:
                name = img_info.get("img_name")
                if not name:
                    continue
                m = re.search(rf"/{re.escape(name)}\s+Do\b", content_str)
                if not m:
                    continue
                start, end = m.start(), m.end()
            fig = _innerstes_figure(bereiche, start, end, figure_tags)
            if not fig:
                continue
            mcid = fig[3]
            elem = elemente[mcid] if mcid < len(elemente) else None
            if elem is None:
                # ParentTree-Eintrag fehlt (null): Element ueber den Tag-Baum suchen
                if index_baum is None:
                    index_baum = _mcid_index_aus_baum(doc, erreichbar)
                elem = index_baum.get((page.xref, mcid))
            if elem is None and doc_elem_xref is not None and fig[2] == "Figure":
                # Figure-Block ohne Strukturelement (InDesign-Altlast): Element fuer die VORHANDENE MCID
                # anlegen, am Dokument-Knoten einhaengen, ParentTree-Eintrag setzen — Inhalt bleibt unangetastet.
                pt = doc.xref_get_key(struct_root_xref, "ParentTree")
                neu_elem = doc.get_new_xref()
                doc.update_object(neu_elem,
                    f"<< /Type /StructElem /S /Figure /P {doc_elem_xref} 0 R /Pg {page.xref} 0 R "
                    f"/Alt {_pdf_string(img_info['alt_text'])} /K {mcid} >>")
                if pt[0] == "xref" and _parenttree_setzen(doc, int(pt[1].split()[0]), int(sp[1]), mcid, neu_elem):
                    belegt[neu_elem] = img_info["alt_text"].strip()
                    uebernommen[(page_num, img_info["xref"])] = neu_elem
                    neu_bloecke.setdefault(page.xref, []).append(neu_elem)
                    erreichbar.add(neu_elem)
                    continue
                doc.update_object(neu_elem, "null")
                continue
            if elem is None or elem not in erreichbar:
                continue  # kein Element bzw. verwaistes Original-Tag: eigenes Tag wie bisher
            try:
                obj = doc.xref_object(elem, compressed=True)
            except Exception:
                continue
            s_typ = re.search(r"/S\s*/(\w+)", obj)
            if not s_typ or s_typ.group(1) not in figure_tags:
                continue
            if elem in belegt:
                # Zweites Bild im selben Figure-Tag (Collage): Texte im EINEN Tag zusammenfuehren statt ein
                # Figure ins Figure zu schachteln.
                bisher = belegt[elem]
                neu = img_info["alt_text"].strip()
                if neu and neu not in bisher:
                    bisher = (bisher + " " + neu).strip() if bisher else neu
                    doc.xref_set_key(elem, "Alt", _pdf_string(bisher))
                belegt[elem] = bisher
                uebernommen[(page_num, img_info["xref"])] = elem
                continue
            doc.xref_set_key(elem, "Alt", _pdf_string(img_info["alt_text"]))
            belegt[elem] = img_info["alt_text"].strip()
            uebernommen[(page_num, img_info["xref"])] = elem
    return uebernommen, warnungen, neu_bloecke


def write_alt_texts_to_pdf(input_path: str, output_path: str, alt_texts: dict, image_metadata: list = None) -> dict:
    """
    PDF/UA-compliant alt-text export.
    Raster images: tags the XObject directly via img_name.
    Vector graphics: tags the original drawing commands in the content stream.
    No screenshot overlays - file size stays unchanged.

    Returns dict with:
        path: output file path
        tagged_count: number of successfully tagged images
        warnings: list of warning strings for images that couldn't be tagged
    """
    doc = fitz.open(input_path)
    cat_xref = doc.pdf_catalog()
    warnings = []

    # Build lookup for image metadata
    metadata_by_xref = {}
    if image_metadata:
        for img in image_metadata:
            metadata_by_xref[img.get("xref")] = img

    # Collect all images (raster + vector) with alt-texts per page
    page_images = {}

    # 1. Collect embedded raster images
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            img_name = img_info[7]
            if xref in alt_texts and alt_texts[xref] is not None:
                alt_text = alt_texts[xref]
                # "dekorativ" = Nutzer hat das Bild explizit als dekorativ
                # markiert -> bewusst leerer Alt-Text. Ein leerer Text OHNE
                # diese Markierung bedeutet dagegen "noch kein Alt-Text
                # vorhanden" -> Bild komplett ueberspringen, NICHT mit leerem
                # /Alt taggen (Befund 12.06.2026: leere /Alt-Eintraege sind
                # schlechter als keine — Pruefwerkzeuge werten sie als Fehler,
                # Screenreader uebergehen das Bild stillschweigend).
                if alt_text == "dekorativ":
                    alt_text = ""
                elif not alt_text.strip():
                    continue
                if page_num not in page_images:
                    page_images[page_num] = []

                meta = metadata_by_xref.get(xref, {})
                bbox = meta.get("bbox")

                page_images[page_num].append({
                    "xref": xref,
                    "img_name": img_name,
                    "alt_text": alt_text,
                    "is_vector": False,
                    "bbox": bbox
                })
                # Hinweis 12.06.2026: Frueher wurde /Alt hier zusaetzlich
                # direkt am Bild-XObject gesetzt ("Fallback"). Das ist nicht
                # standardkonform (/Alt gehoert ans StructElem) und fuehrte zu
                # doppelten /Alt-Eintraegen in der Datei — entfernt.

    # 2. Collect vector graphics (xref >= 900000)
    for xref, alt_text in alt_texts.items():
        if xref >= 900000:
            meta = metadata_by_xref.get(xref, {})
            if not meta:
                warnings.append(
                    "Vektorgrafik (Seite unbekannt): Keine Metadaten vorhanden, "
                    "Alt-Text konnte nicht exportiert werden."
                )
                continue

            page_num = meta.get("page_number", 1) - 1
            bbox = meta.get("bbox")

            if not bbox:
                warnings.append(
                    f"Vektorgrafik auf Seite {page_num + 1}: Keine Positionsdaten vorhanden, "
                    "Alt-Text konnte nicht exportiert werden."
                )
                continue

            # Gleiche Regel wie bei Rasterbildern: "dekorativ" -> bewusst
            # leer, fehlender Text -> Grafik ueberspringen statt leer taggen.
            if alt_text == "dekorativ":
                alt_text = ""
            elif alt_text is None or not alt_text.strip():
                continue

            if page_num not in page_images:
                page_images[page_num] = []

            page_images[page_num].append({
                "xref": xref,
                "img_name": None,
                "alt_text": alt_text,
                "is_vector": True,
                "bbox": bbox
            })

    if not page_images:
        doc.save(output_path)
        doc.close()
        return {"path": output_path, "tagged_count": 0, "warnings": warnings}

    # --- Detect existing PDF structure ---
    existing_struct_root = doc.xref_get_key(cat_xref, "StructTreeRoot")
    has_existing_structure = existing_struct_root[0] == "xref"

    if has_existing_structure:
        # --- MERGE into existing structure ---
        struct_root_xref = int(existing_struct_root[1].split()[0])
        print(f"Existing StructTreeRoot found (xref {struct_root_xref}), merging Figure elements...")

        k_info = doc.xref_get_key(struct_root_xref, "K")
        doc_elem_xref = None
        if k_info[0] == "xref":
            doc_elem_xref = int(k_info[1].split()[0])
        elif k_info[0] == "array":
            first_ref = re.search(r'(\d+)\s+0\s+R', k_info[1])
            if first_ref:
                doc_elem_xref = int(first_ref.group(1))

        if not doc_elem_xref:
            warnings.append(
                "Bestehende PDF-Struktur konnte nicht gelesen werden, erstelle neue Struktur."
            )
            has_existing_structure = False

    uebernommen, neu_bloecke = {}, {}
    if has_existing_structure:
        # Zuerst: Bilder, die schon in einem Figure-Tag des Dokuments liegen, bekommen ihren Alt-Text DORT.
        uebernommen, w_ueb, neu_bloecke = _vorhandene_figures_uebernehmen(doc, struct_root_xref, page_images, doc_elem_xref)
        warnings.extend(w_ueb)
        if uebernommen:
            print(f"Vorhandene Figure-Tags uebernommen: {len(uebernommen)}")
            page_images = {pn: [i for i in imgs if (pn, i["xref"]) not in uebernommen] for pn, imgs in page_images.items()}
            page_images = {pn: imgs for pn, imgs in page_images.items() if imgs}

    page_figures = {}
    if has_existing_structure and page_images:
        pt_info = doc.xref_get_key(struct_root_xref, "ParentTree")
        parent_tree_xref = int(pt_info[1].split()[0]) if pt_info[0] == "xref" else None

        ptk_info = doc.xref_get_key(struct_root_xref, "ParentTreeNextKey")
        parent_tree_next_key = int(ptk_info[1]) if ptk_info[0] == "int" else len(doc)

        page_max_mcid = {}
        for page_num in sorted(page_images.keys()):
            page = doc[page_num]
            content = page.read_contents()
            if content:
                content_str = content.decode('latin-1')
                mcids = re.findall(r'/MCID\s+(\d+)', content_str)
                page_max_mcid[page_num] = max(int(m) for m in mcids) if mcids else -1
            else:
                page_max_mcid[page_num] = -1

        figure_xrefs = []
        page_figures = {}
        index_baum_merge = None

        for page_num in sorted(page_images.keys()):
            page = doc[page_num]
            page_figures[page_num] = []
            next_mcid = page_max_mcid.get(page_num, -1) + 1

            content = page.read_contents()
            cs = content.decode('latin-1') if content else ""
            bereiche = _markierte_bereiche(cs)
            belegte_bereiche = []   # (start, end) bereits geplanter Umhuellungen dieser Seite
            sp_seite = doc.xref_get_key(page.xref, "StructParents")
            elemente_seite = _parenttree_elemente(doc, struct_root_xref, int(sp_seite[1])) if sp_seite[0] == "int" else []

            for img_info in page_images[page_num]:
                mcid = next_mcid
                # Erst den Platz im Inhaltsstrom bestimmen, DANN das Element anlegen — ein Figure mit
                # MCID, die es im Inhalt nicht gibt, ist ein leeres Tag (bis 14.09.2026 so).
                if img_info["is_vector"]:
                    if not img_info["bbox"] or not cs:
                        warnings.append(f"Vektorgrafik auf Seite {page_num + 1}: keine Positionsdaten, nicht getaggt.")
                        continue
                    start, end = _find_vector_graphic_range(cs, img_info["bbox"], page.rect.height)
                    if start is None or end is None:
                        warnings.append(
                            f"Vektorgrafik auf Seite {page_num + 1}: Zeichenbefehle im PDF nicht gefunden, nicht getaggt."
                        )
                        continue
                else:
                    name = img_info.get("img_name")
                    m = re.search(rf"(q\s[\s\S]*?/{re.escape(name)}\s+Do\s*Q)", cs) if name else None
                    if not m:
                        warnings.append(f"Bild auf Seite {page_num + 1}: Zeichenbefehl im Seiteninhalt nicht gefunden, nicht getaggt.")
                        continue
                    start, end = m.start(), m.end()
                plan = _umhuellung_planen(cs, bereiche, start, end, mcid)
                if plan is None:
                    warnings.append(
                        f"{'Vektorgrafik' if img_info['is_vector'] else 'Bild'} auf Seite {page_num + 1}: liegt in fremder "
                        "Struktur des Dokuments (z. B. innerhalb eines Absatzes), nicht getaggt."
                    )
                    continue
                if any(plan[0] < e and b_ < plan[1] for b_, e in belegte_bereiche):
                    warnings.append(
                        f"{'Vektorgrafik' if img_info['is_vector'] else 'Bild'} auf Seite {page_num + 1}: ueberschneidet sich "
                        "mit einer bereits getaggten Grafik, nicht getaggt."
                    )
                    continue
                belegte_bereiche.append((plan[0], plan[1]))
                n_mcids = plan[3]
                eltern = doc_elem_xref
                if plan[4] is not None:
                    # Kind des umschliessenden Elements (z. B. Absatz), nicht des Dokument-Knotens
                    kandidat = elemente_seite[plan[4]] if plan[4] < len(elemente_seite) else None
                    if not kandidat:
                        if index_baum_merge is None:
                            index_baum_merge = _mcid_index_aus_baum(doc, _collect_reachable_struct_elems(doc, struct_root_xref))
                        kandidat = index_baum_merge.get((page.xref, plan[4]))
                    if not kandidat:
                        warnings.append(f"Bild auf Seite {page_num + 1}: umschliessendes Element nicht aufloesbar, nicht getaggt.")
                        continue
                    eltern = kandidat
                next_mcid += n_mcids
                fig_xref = doc.get_new_xref()
                mcrs = " ".join(f"<< /Type /MCR /MCID {mcid + k} /Pg {page.xref} 0 R >>" for k in range(n_mcids))
                k_eintrag = mcrs if n_mcids == 1 else f"[ {mcrs} ]"
                doc.update_object(fig_xref,
                    f"<< /Type /StructElem /S /Figure /P {eltern} 0 R "
                    f"/Pg {page.xref} 0 R /Alt {_pdf_string(img_info['alt_text'])} "
                    f"/K {k_eintrag} >>")
                if eltern == doc_elem_xref:
                    figure_xrefs.append(fig_xref)
                elif not _figures_einhaengen(doc, eltern, [fig_xref]):
                    warnings.append(f"Bild auf Seite {page_num + 1}: konnte nicht in das umschliessende Element eingehaengt werden.")
                    continue
                if img_info["is_vector"]:
                    print(f"Vector graphic tagged: page {page_num + 1}, MCID {mcid}")
                page_figures[page_num].append((mcid, fig_xref, img_info, plan))

        # Neue Figure-Elemente als Kinder des Dokument-Knotens einhaengen. Bis 14.09.2026 wurde nur
        # ein INLINE-Array (/K [ ... ]) erkannt; bei /K als Referenz auf ein Array-Objekt (InDesign,
        # Prod-Kundendokument Projekt 430) oder als Einzelkind blieben die Figures ohne Eltern-Eintrag —
        # unerreichbar im Tag-Baum, und finalize_export_pdf raeumte sie als „verwaist“ wieder weg:
        # Export ohne Alt-Texte bei Meldung „234 getaggt“.
        figures_je_seite: dict = {pg: list(figs) for pg, figs in neu_bloecke.items()}
        neu_bloecke = {}
        for page_num in sorted(page_figures.keys()):
            eigene = [f[1] for f in page_figures[page_num] if f[1] in figure_xrefs]
            if eigene:
                figures_je_seite.setdefault(doc[page_num].xref, []).extend(eigene)
        if not _figures_nach_seite_einhaengen(doc, doc_elem_xref, figures_je_seite, struct_root_xref):
            warnings.append(
                "Die Alt-Texte konnten nicht in den Tag-Baum der PDF eingehaengt werden "
                "(unbekannte Form des /K-Eintrags am Dokument-Knoten)."
            )

        if parent_tree_xref:
            # ParentTree: je neuem Marked-Content-Stueck (MCID) ein Eintrag im Array der Seite — Index = MCID.
            # Bis 14.09.2026 per Regex auf dem Wurzelobjekt (nur inline-Nums mit inline-Arrays); bei
            # InDesign-PDFs (Arrays als eigene Objekte) blieben die Eintraege stumm aus → Pruefwerkzeuge
            # werteten die Stuecke als „nicht getaggt“.
            for page_num, figs in sorted(page_figures.items()):
                page = doc[page_num]
                sp_info = doc.xref_get_key(page.xref, "StructParents")
                if sp_info[0] != "int":
                    continue
                refs = []
                for f in figs:
                    anzahl = f[3][3] if isinstance(f[3], tuple) and len(f[3]) > 3 else 1
                    refs.extend([f[1]] * anzahl)
                if not _parenttree_anhaengen(doc, parent_tree_xref, int(sp_info[1]), refs):
                    warnings.append(
                        f"Seite {page_num + 1}: ParentTree-Eintrag nicht gefunden — Pruefwerkzeuge koennten die neuen "
                        "Alt-Texte als nicht getaggt werten."
                    )

    elif has_existing_structure and neu_bloecke:
        if not _figures_nach_seite_einhaengen(doc, doc_elem_xref, neu_bloecke, struct_root_xref):
            warnings.append("Elemente fuer verwaiste Figure-Bloecke konnten nicht eingehaengt werden.")
        neu_bloecke = {}
    elif not has_existing_structure:
        # --- CREATE new structure (PDF had no tags) ---
        print("No existing structure, creating new StructTreeRoot...")
        struct_root_xref = doc.get_new_xref()
        parent_tree_xref = doc.get_new_xref()
        doc_elem_xref = doc.get_new_xref()

        figure_xrefs = []
        page_figures = {}

        for page_num in sorted(page_images.keys()):
            page = doc[page_num]
            page_figures[page_num] = []

            for img_info in page_images[page_num]:
                fig_xref = doc.get_new_xref()
                mcid = len(page_figures[page_num])
                alt_pdf_str = _pdf_string(img_info["alt_text"])

                doc.update_object(fig_xref,
                    f"<< /Type /StructElem /S /Figure /P {doc_elem_xref} 0 R "
                    f"/Pg {page.xref} 0 R /Alt {alt_pdf_str} "
                    f"/K << /Type /MCR /MCID {mcid} /Pg {page.xref} 0 R >> >>")

                figure_xrefs.append(fig_xref)

                content_range = None
                if img_info["is_vector"] and img_info["bbox"]:
                    content = page.read_contents()
                    if content:
                        cs = content.decode('latin-1')
                        start, end = _find_vector_graphic_range(cs, img_info["bbox"], page.rect.height)
                        if start is not None and end is not None:
                            content_range = (start, end)
                            print(f"Vector graphic tagged: page {page_num + 1}, MCID {mcid}")
                        else:
                            warnings.append(
                                f"Vektorgrafik auf Seite {page_num + 1}: Zeichenbefehle im PDF nicht gefunden, "
                                "Alt-Text wurde als Struktur-Tag gesetzt aber moeglicherweise nicht korrekt verknuepft."
                            )

                page_figures[page_num].append((mcid, fig_xref, img_info, content_range))

        kids_str = " ".join(f"{x} 0 R" for x in figure_xrefs)
        doc.update_object(doc_elem_xref,
            f"<< /Type /StructElem /S /Document /P {struct_root_xref} 0 R "
            f"/K [{kids_str}] >>")

        nums_parts = []
        for page_num, figs in sorted(page_figures.items()):
            refs = " ".join(f"{f[1]} 0 R" for f in figs)
            nums_parts.append(f"{page_num} [{refs}]")

        doc.update_object(parent_tree_xref,
            f"<< /Nums [{' '.join(nums_parts)}] >>")

        doc.update_object(struct_root_xref,
            f"<< /Type /StructTreeRoot /K {doc_elem_xref} 0 R "
            f"/ParentTree {parent_tree_xref} 0 R >>")

        doc.xref_set_key(cat_xref, "StructTreeRoot", f"{struct_root_xref} 0 R")
        doc.xref_set_key(cat_xref, "MarkInfo", "<< /Marked true >>")

    # Mark content streams - wrap operations with BMC/EMC
    for page_num, figs in page_figures.items():
        page = doc[page_num]

        if not has_existing_structure:
            doc.xref_set_key(page.xref, "StructParents", str(page_num))

        content = page.read_contents()
        if not content:
            continue
        content_str = content.decode('latin-1')

        modifications = []

        for mcid, fig_xref, img_info, content_range in figs:
            if has_existing_structure:
                # Merge-Zweig (14.09.2026): content_range ist ein Plan (start, end, ersatztext)
                if content_range:
                    modifications.append({'type': 'plan', 'start': content_range[0], 'end': content_range[1],
                                          'ersatz': content_range[2], 'mcid': mcid})
                continue
            if img_info["is_vector"]:
                if content_range:
                    start, end = content_range
                    modifications.append({
                        'type': 'vector',
                        'start': start,
                        'end': end,
                        'mcid': mcid
                    })
            else:
                img_name = img_info.get("img_name")
                if img_name:
                    escaped_name = re.escape(img_name)
                    pattern = rf'(q\s[\s\S]*?/{escaped_name}\s+Do\s*Q)'
                    match = re.search(pattern, content_str)
                    if match:
                        modifications.append({
                            'type': 'raster',
                            'start': match.start(),
                            'end': match.end(),
                            'mcid': mcid,
                            'original': match.group(1)
                        })

        modifications.sort(key=lambda x: x['start'], reverse=True)

        for mod in modifications:
            if mod['type'] == 'plan':
                content_str = content_str[:mod['start']] + mod['ersatz'] + content_str[mod['end']:]
            elif mod['type'] == 'vector':
                original = content_str[mod['start']:mod['end']]
                wrapped = f"/Figure <</MCID {mod['mcid']}>> BDC\n{original}\nEMC\n"
                content_str = content_str[:mod['start']] + wrapped + content_str[mod['end']:]
            else:
                original = mod['original']
                wrapped = f"/Figure <</MCID {mod['mcid']}>> BDC\n{original}\nEMC"
                content_str = content_str.replace(original, wrapped, 1)

        new_content = content_str.encode('latin-1')
        contents_info = doc.xref_get_key(page.xref, "Contents")
        if contents_info[0] == 'xref':
            cs_xref = int(contents_info[1].split()[0])
            doc.update_stream(cs_xref, new_content)
        else:
            new_xref = doc.get_new_xref()
            doc.update_object(new_xref, "<< >>")
            doc.update_stream(new_xref, new_content)
            doc.xref_set_key(page.xref, "Contents", f"{new_xref} 0 R")

    tagged_count = len(uebernommen)          # in vorhandene Figure-Tags eingetragen
    alle_figuren = list(uebernommen.values())
    for page_num, figs in page_figures.items():
        for mcid, fig_xref, img_info, content_range in figs:
            alle_figuren.append(fig_xref)
            if not img_info["is_vector"] or content_range is not None:
                tagged_count += 1

    # Kontrolle (14.09.2026): Jedes geschriebene Figure-Element muss vom StructTreeRoot aus erreichbar
    # sein — sonst liest kein Screenreader den Alt-Text, und die Zaehlung „getaggt“ waere eine Luege.
    root_info = doc.xref_get_key(cat_xref, "StructTreeRoot")
    erreichbar = set()
    if root_info[0] == "xref":
        erreichbar = _collect_reachable_struct_elems(doc, int(root_info[1].split()[0]))
    unerreichbar = [x for x in alle_figuren if x not in erreichbar]
    if unerreichbar:
        warnings.append(
            f"{len(unerreichbar)} von {len(alle_figuren)} Alt-Texten haengen nicht im Tag-Baum der PDF "
            "und werden von Screenreadern nicht vorgelesen."
        )
        tagged_count = max(0, tagged_count - len(unerreichbar))

    doc.save(output_path)
    doc.close()
    return {"path": output_path, "tagged_count": tagged_count, "warnings": warnings,
            "figure_xrefs": alle_figuren, "unreachable_figures": unerreichbar}


def _seiten_eines_elements(doc: fitz.Document, xref: int, seiten: set, tiefe: int = 0, gesehen: set | None = None) -> None:
    """Sammelt alle Seiten-xrefs (/Pg), die im Teilbaum eines Elements vorkommen."""
    gesehen = gesehen if gesehen is not None else set()
    if xref in gesehen or tiefe > 80:
        return
    gesehen.add(xref)
    try:
        obj = doc.xref_object(xref, compressed=True)
    except Exception:
        return
    for m in re.finditer(r"/Pg\s+(\d+)\s+0\s+R", obj):
        seiten.add(int(m.group(1)))
    k = doc.xref_get_key(xref, "K")
    if k[0] == "null":
        return
    kand = [int(r) for r in _XREF_REF_RE.findall(k[1])]
    while kand:
        ch = kand.pop(0)
        try:
            o = doc.xref_object(ch, compressed=True)
        except Exception:
            continue
        if o.lstrip().startswith("["):
            kand = [int(r) for r in _XREF_REF_RE.findall(o)] + kand
            continue
        if "/StructElem" in o or _STRUCT_ELEM_RE.search(o):
            _seiten_eines_elements(doc, ch, seiten, tiefe + 1, gesehen)


def _letztes_element_je_seite(doc: fitz.Document, struct_root_xref: int) -> dict:
    """{page_xref: elem_xref} — das in LESEREIHENFOLGE letzte Strukturelement, das auf der Seite liegt
    (eigenes /Pg oder MCR-/Pg in /K). Anker fuer das Einfuegen neuer Figures als Geschwister."""
    reihe: list = []          # (xref, page_xref) in Lesereihenfolge
    gesehen: set = set()
    stapel = [(struct_root_xref, 0)]
    # iterativ in Vorordnung, Kinder in Originalreihenfolge
    while stapel:
        xref, tiefe = stapel.pop()
        if xref in gesehen or tiefe > 80:
            continue
        gesehen.add(xref)
        try:
            obj = doc.xref_object(xref, compressed=True)
        except Exception:
            continue
        if xref != struct_root_xref:
            for m in re.finditer(r"/Pg\s+(\d+)\s+0\s+R", obj):
                reihe.append((xref, int(m.group(1))))
                break
        k = doc.xref_get_key(xref, "K")
        if k[0] == "null":
            continue
        kinder = []
        kand = [int(r) for r in _XREF_REF_RE.findall(k[1])]
        while kand:
            ch = kand.pop(0)
            try:
                o = doc.xref_object(ch, compressed=True)
            except Exception:
                continue
            if o.lstrip().startswith("["):
                kand = [int(r) for r in _XREF_REF_RE.findall(o)] + kand
                continue
            if "/StructElem" in o or _STRUCT_ELEM_RE.search(o):
                kinder.append(ch)
        for ch in reversed(kinder):
            stapel.append((ch, tiefe + 1))
    # Anker je Seite: das letzte Element der Seite VOR dem ersten Element einer spaeteren Seite —
    # InDesign fuehrt Reste einer Seite (Fusszeilen, Rahmenfortsetzungen) oft erst nach der naechsten
    # Seite im Baum; „letztes Element der Seite“ allein setzte unsere Figures dann hinter die Folgeseite.
    nummer = {pg.xref: pg.number for pg in doc}
    letztes: dict = {}
    for page_xref in set(px for _, px in reihe):
        nr = nummer.get(page_xref)
        if nr is None:
            continue
        erstes_spaeter = next((i for i, (_, px) in enumerate(reihe) if nummer.get(px, -1) > nr), len(reihe))
        kandidaten = [i for i, (_, px) in enumerate(reihe) if px == page_xref and i < erstes_spaeter]
        if kandidaten:
            letztes[page_xref] = reihe[max(kandidaten)][0]
        elif erstes_spaeter > 0:
            # Doppelseiten-Fall (rechte Seite steht im Baum vor der linken): direkt vor die Folgeseite,
            # also hinter das Element, das der Folgeseite vorausgeht
            letztes[page_xref] = reihe[erstes_spaeter - 1][0]
        else:
            erste = next((i for i, (_, px) in enumerate(reihe) if px == page_xref), None)
            if erste is not None:
                letztes[page_xref] = reihe[erste][0]
    return letztes


def _als_geschwister_einfuegen(doc: fitz.Document, anker_xref: int, neu_xrefs: list) -> bool:
    """Fuegt neue Elemente direkt HINTER anker_xref in die Kinderliste von dessen Elternelement ein und
    setzt ihr /P auf dieses Elternelement. False, wenn die Kinderliste nicht in bekannter Form vorliegt."""
    if not neu_xrefs:
        return True
    p = doc.xref_get_key(anker_xref, "P")
    if p[0] != "xref":
        return False
    eltern = int(p[1].split()[0])
    k = doc.xref_get_key(eltern, "K")
    if k[0] == "array":
        text, ziel_obj = k[1], None
    elif k[0] == "xref":
        ziel_obj = int(k[1].split()[0])
        if ziel_obj == anker_xref:
            # Einzelkind: Kinderliste daraus machen
            doc.xref_set_key(eltern, "K", "[ " + f"{anker_xref} 0 R " + " ".join(f"{x} 0 R" for x in neu_xrefs) + " ]")
            for x in neu_xrefs:
                doc.xref_set_key(x, "P", f"{eltern} 0 R")
            return True
        text = doc.xref_object(ziel_obj, compressed=True).strip()
        if not text.startswith("["):
            return False
    else:
        return False
    m = re.search(rf"(?<![0-9]){anker_xref}\s+0\s+R", text)
    if not m:
        return False
    neu = " " + " ".join(f"{x} 0 R" for x in neu_xrefs)
    text = text[:m.end()] + neu + text[m.end():]
    if ziel_obj is None:
        doc.xref_set_key(eltern, "K", text)
    else:
        doc.update_object(ziel_obj, text)
    for x in neu_xrefs:
        doc.xref_set_key(x, "P", f"{eltern} 0 R")
    return True


def _figures_nach_seite_einhaengen(doc: fitz.Document, doc_elem_xref: int, figures_je_seite: dict,
                                   struct_root_xref: int | None = None) -> bool:
    """Neue Figure-Elemente an der Stelle ihrer Seite einhaengen: bevorzugt als Geschwister hinter dem letzten
    Element der Seite (Lesereihenfolge bleibt seitengenau); sonst ueber die Kinderliste des Dokument-Knotens."""
    if struct_root_xref is not None and figures_je_seite:
        letztes = _letztes_element_je_seite(doc, struct_root_xref)
        rest: dict = {}
        for page_xref, figs in figures_je_seite.items():
            anker = letztes.get(page_xref)
            if anker is None or anker in figs or not _als_geschwister_einfuegen(doc, anker, figs):
                rest[page_xref] = figs
                print(f"Figure-Einfuegen als Geschwister nicht moeglich (Seite xref {page_xref}, Anker {anker}) — Dokument-Knoten")
        if not rest:
            return True
        figures_je_seite = rest
    return _figures_nach_dokumentknoten_einhaengen(doc, doc_elem_xref, figures_je_seite)


def _figures_nach_dokumentknoten_einhaengen(doc: fitz.Document, doc_elem_xref: int, figures_je_seite: dict) -> bool:
    """Haengt neue Figure-Elemente in die Kinderliste des Dokument-Knotens an der Stelle ihrer SEITE ein —
    hinter dem letzten Kind, das diese Seite beruehrt (14.09.2026: angehaengt am Ende laesen Screenreader
    sie erst nach der letzten Seite). figures_je_seite: {page_xref: [fig_xref, ...]} in Seitenreihenfolge.
    Fallback bei unbekannter Kinderform: _figures_einhaengen (ans Ende)."""
    alle = [x for figs in figures_je_seite.values() for x in figs]
    if not alle:
        return True
    k = doc.xref_get_key(doc_elem_xref, "K")
    if k[0] == "array":
        kinder = [int(r) for r in _XREF_REF_RE.findall(k[1])]; ziel_obj = None
        if len(kinder) != len(re.findall(r"\S+\s+0\s+R|(?<![\d/])\d+(?![\s\d]*0\s+R)", k[1].strip()[1:-1])):
            return _figures_einhaengen(doc, doc_elem_xref, alle)   # gemischte Kinder (MCIDs): nicht umsortieren
    elif k[0] == "xref":
        ziel_obj = int(k[1].split()[0])
        o = doc.xref_object(ziel_obj, compressed=True).strip()
        if not o.startswith("["):
            # einzelnes Kind (z. B. /Part): dort weitersuchen, sonst ans Ende
            kk = doc.xref_get_key(ziel_obj, "K")
            if kk[0] in ("array", "xref"):
                return _figures_nach_dokumentknoten_einhaengen(doc, ziel_obj, figures_je_seite)
            return _figures_einhaengen(doc, doc_elem_xref, alle)
        kinder = [int(r) for r in _XREF_REF_RE.findall(o)]
    else:
        return _figures_einhaengen(doc, doc_elem_xref, alle)
    if not kinder:
        return _figures_einhaengen(doc, doc_elem_xref, alle)
    # Seiten je Kind (in Reihenfolge) — teuer, aber nur einmal je Export
    seiten_je_kind = []
    for kind in kinder:
        seiten: set = set()
        _seiten_eines_elements(doc, kind, seiten)
        seiten_je_kind.append(seiten)
    seiten_reihenfolge = {pg.xref: pg.number for pg in doc}
    # Einfuegeposition (Index in der ALTEN Kinderliste) je Seite bestimmen
    einfuegungen: dict = {}   # pos -> [fig_xref, ...]
    for page_xref, figs in sorted(figures_je_seite.items(), key=lambda kv: seiten_reihenfolge.get(kv[0], 10**9)):
        pos = None
        for i in range(len(seiten_je_kind) - 1, -1, -1):
            if page_xref in seiten_je_kind[i]:
                pos = i + 1          # hinter dem letzten Kind dieser Seite
                break
        if pos is None:
            # kein Kind dieser Seite: vor das erste Kind einer SPAETEREN Seite (sonst ans Ende)
            nr = seiten_reihenfolge.get(page_xref, 10**9)
            pos = len(kinder)
            for i, sj in enumerate(seiten_je_kind):
                if sj and min(seiten_reihenfolge.get(x, 10**9) for x in sj) > nr:
                    pos = i
                    break
        einfuegungen.setdefault(pos, []).extend(figs)
    neu = []
    for i in range(len(kinder) + 1):
        neu.extend(einfuegungen.get(i, []))
        if i < len(kinder):
            neu.append(kinder[i])
    text = "[ " + " ".join(f"{x} 0 R" for x in neu) + " ]"
    if ziel_obj is None:
        doc.xref_set_key(doc_elem_xref, "K", text)
    else:
        doc.update_object(ziel_obj, text)
    return True


def _figures_einhaengen(doc: fitz.Document, doc_elem_xref: int, figure_xrefs: list) -> bool:
    """Haengt neue StructElem-xrefs als Kinder an den Dokument-Knoten — in jeder Form, die /K haben kann:
    inline-Array, Referenz auf ein Array-Objekt, einzelne Referenz (ein Kind) oder gar kein /K.
    Liefert False, wenn die Form nicht erkannt wurde (dann wird nichts veraendert)."""
    if not figure_xrefs:
        return True
    neu = " ".join(f"{x} 0 R" for x in figure_xrefs)
    k_info = doc.xref_get_key(doc_elem_xref, "K")
    typ, wert = k_info[0], (k_info[1] or "").strip()
    if typ == "array":
        # inline: /K [ a 0 R b 0 R ]  ->  Eintraege anhaengen
        doc.xref_set_key(doc_elem_xref, "K", wert.rstrip()[:-1].rstrip() + " " + neu + " ]")
        return True
    if typ == "xref":
        ziel = int(wert.split()[0])
        ziel_obj = doc.xref_object(ziel, compressed=True).strip()
        if ziel_obj.startswith("["):
            # Referenz auf ein Array-Objekt (InDesign): das Array-Objekt selbst erweitern
            doc.update_object(ziel, ziel_obj.rstrip()[:-1].rstrip() + " " + neu + " ]")
            return True
        # Einzelnes Kind-Element: zu einem Array machen
        doc.xref_set_key(doc_elem_xref, "K", f"[ {ziel} 0 R {neu} ]")
        return True
    if typ in ("null", "") or not wert:
        doc.xref_set_key(doc_elem_xref, "K", f"[ {neu} ]")
        return True
    if typ in ("int", "dict"):
        # MCID-Zahl oder MCR-Dict als einziges Kind: mit in ein Array nehmen
        doc.xref_set_key(doc_elem_xref, "K", f"[ {wert} {neu} ]")
        return True
    return False


# ─── Abschluss-Schritt fuer beide Export-Pfade (12.06.2026) ──────────────────

_STRUCT_ELEM_RE = re.compile(r"/S\s*/\w+")
_XREF_REF_RE = re.compile(r"(\d+)\s+0\s+R")


def _collect_reachable_struct_elems(doc: fitz.Document, root_xref: int) -> set:
    """Sammelt alle vom StructTreeRoot aus erreichbaren StructElem-xrefs.

    Folgt rekursiv den /K-Eintraegen (Einzelreferenz, Array oder Dict).
    MCR-/OBJR-Dicts und Seiten-Objekte sind keine StructElems und werden
    nicht weiterverfolgt.
    """
    reachable = set()
    stack = [root_xref]
    while stack:
        xref = stack.pop()
        if xref in reachable:
            continue
        reachable.add(xref)
        k_info = doc.xref_get_key(xref, "K")
        if k_info[0] == "null":
            continue
        kandidaten = [int(r) for r in _XREF_REF_RE.findall(k_info[1])]
        gesehen_arrays = set()
        while kandidaten:
            child = kandidaten.pop()
            if child in reachable:
                continue
            try:
                obj = doc.xref_object(child, compressed=True)
            except Exception:
                continue
            # /K darf auch auf ein REINES Array-Objekt zeigen (/K 11 0 R -> [ a 0 R b 0 R ]).
            # Das Array ist kein StructElem, seine Eintraege sind aber Kinder — sie werden
            # weiterverfolgt (14.09.2026; vorher endete die Suche hier, und der Abschluss-Schritt
            # haette alle /Alt-Elemente darunter fuer Waisen gehalten).
            if obj.lstrip().startswith("[") and child not in gesehen_arrays:
                gesehen_arrays.add(child)
                kandidaten.extend(int(r) for r in _XREF_REF_RE.findall(obj))
                continue
            # Nur echte StructElems weiterverfolgen (erkennbar am /S-Typ).
            # MCR-/OBJR-Verweise und Seiten-Objekte haben kein /S und fallen
            # hier automatisch raus. WICHTIG: NICHT per Substring auf "/MCR"
            # filtern — StructElems mit INLINE-K-Dict (/K << /Type /MCR ... >>)
            # enthalten den String auch und wuerden faelschlich ausgeschlossen
            # (haette unsere eigenen Figures als Waisen markiert; von der
            # Testsuite am 12.06.2026 gefunden).
            if "/StructElem" in obj or _STRUCT_ELEM_RE.search(obj):
                stack.append(child)
    return reachable


def _parent_tree_node_xrefs(doc: fitz.Document, root_xref: int) -> list:
    """Liefert die xrefs aller Knoten des ParentTree (Number-Tree, inkl. /Kids)."""
    pt_info = doc.xref_get_key(root_xref, "ParentTree")
    if pt_info[0] != "xref":
        return []
    nodes = []
    stack = [int(pt_info[1].split()[0])]
    while stack:
        xref = stack.pop()
        if xref in nodes:
            continue
        nodes.append(xref)
        kids = doc.xref_get_key(xref, "Kids")
        if kids[0] != "null":
            stack.extend(int(r) for r in _XREF_REF_RE.findall(kids[1]))
    return nodes


def remove_orphaned_alt_elems(doc: fitz.Document, schonen: set | None = None) -> int:
    """Entfernt verwaiste StructElems mit /Alt-Eintrag aus der PDF.

    Hintergrund (Befund 12.06.2026, Demo-Infografik): Erstellungsprogramme
    wie PowerPoint hinterlassen StructElems mit /Alt (z.B. "Bullet", Achsen-
    Beschriftungen), die NICHT mehr im Tag-Baum haengen — nur noch der
    ParentTree referenziert sie. Screenreader lesen sie nicht, aber
    Pruefwerkzeuge und Roh-Inspektion sehen sie: Die Demo-PDF hatte 9
    /Alt-Eintraege fuer 3 Bilder. Diese Altlasten raeumen wir hier weg.

    Vorgehen (bewusst konservativ):
    - Erreichbarkeit vom StructTreeRoot aus bestimmen (ueber /K).
    - NUR unerreichbare StructElems, die ein /Alt tragen, werden entfernt
      (Objekt durch null ersetzt). Andere unerreichbare Elemente bleiben —
      sie stoeren niemanden und jeder zusaetzliche Eingriff ist Risiko.
    - Referenzen in den ParentTree-Knoten werden durch null ersetzt, NICHT
      geloescht: Die Array-Position im ParentTree entspricht der MCID,
      Loeschen wuerde alle folgenden Zuordnungen verschieben.
    - Fail-safe: Wenn der Baum nicht lesbar ist (keine Kinder erreichbar),
      wird NICHTS entfernt — lieber Altlasten behalten als Tags zerstoeren.

    Gibt die Anzahl entfernter Elemente zurueck. Speichert NICHT selbst.
    """
    cat = doc.pdf_catalog()
    root_info = doc.xref_get_key(cat, "StructTreeRoot")
    if root_info[0] != "xref":
        return 0
    root_xref = int(root_info[1].split()[0])

    reachable = _collect_reachable_struct_elems(doc, root_xref)
    if len(reachable) <= 1:
        # Nur die Wurzel erreicht -> Baum unlesbar oder leer. Fail-safe: nichts tun.
        return 0

    orphans = []
    schonen = schonen or set()
    for xref in range(1, doc.xref_length()):
        if xref in reachable or xref in schonen:
            continue
        try:
            obj = doc.xref_object(xref, compressed=True)
        except Exception:
            continue
        if "/Alt" not in obj:
            continue
        # Nur StructElems anfassen — /Alt kann auch an anderen Objekttypen
        # vorkommen (z.B. Bild-XObjects aus aelteren Exporten), die lassen
        # wir bewusst in Ruhe.
        if "/StructElem" not in obj and not _STRUCT_ELEM_RE.search(obj):
            continue
        orphans.append(xref)

    if not orphans:
        return 0

    for xref in orphans:
        doc.update_object(xref, "null")

    # ParentTree-Referenzen auf die entfernten Objekte durch null ersetzen.
    for node_xref in _parent_tree_node_xrefs(doc, root_xref):
        obj = doc.xref_object(node_xref, compressed=True)
        new_obj = obj
        for orphan in orphans:
            new_obj = re.sub(r"(?<![0-9])%d\s+0\s+R" % orphan, "null", new_obj)
        if new_obj != obj:
            doc.update_object(node_xref, new_obj)

    return len(orphans)


def _titel_brauchbar(titel: str, filename_base: str | None) -> bool:
    """Ein vorhandener Titel zaehlt nur, wenn er ein Titel ist: nicht leer, nicht „Untitled“,
    nicht der Dateiname (mit oder ohne .pdf). PAC/veraPDF-Praxis: Titel = Dateiname ist ein
    Befund, kein Titel (Michael Karbe 11.09.2026)."""
    t = (titel or "").strip()
    if not t or t.lower() in ("untitled", "unbenannt", "ohne titel", "document", "dokument"):
        return False
    if filename_base:
        fb = filename_base.strip().lower()
        if t.lower() in (fb, fb + ".pdf", fb + ".docx"):
            return False
    return True


def finalize_export_pdf(pdf_path: str, title: str = None,
                        fallback_title: str = None,
                        lang: str = "de-DE", verfahren: str | None = None,
                        fallback_heading: str | None = None,
                        filename_base: str | None = None,
                        schonen: set | None = None,
                        creator: str | None = None) -> dict:
    """Gemeinsamer Abschluss-Schritt fuer beide Export-Pfade (PDFix + fitz).
    `schonen`: xrefs, die Schritt 3 nie entfernen darf (die vom fitz-Export selbst
    geschriebenen Figure-Elemente, 14.09.2026).

    Erledigt drei Dinge an der fertigen Export-PDF:
    1. Dokumentsprache setzen (WCAG 3.1.1) — nur wenn die PDF noch KEINE
       Sprache hat; eine vorhandene Angabe des Autors bleibt erhalten.
       Aktuell konstant de-DE, da die Pipeline deutsche Texte erzeugt
       (bei kuenftiger Mehrsprachigkeit hier parametrisieren).
    2. Dokumenttitel setzen (WCAG 2.4.2 / PDF/UA 7.1) — Prioritaet seit 11.09.2026
       (Michael Karbe/Steve: ein Dateiname ist kein Titel; die Fassung vom 12.06.
       liess den eingetippten Export-Namen sogar einen echten Titel ueberschreiben):
       a) vorhandener Titel der Quell-PDF, wenn brauchbar (_titel_brauchbar),
       b) `title` = der in InkluDocs vergebene Dokumentname (display_name),
       c) `fallback_heading` = erste Ueberschrift aus dem Inhalt (Lesezeichen/Tags),
       d) `fallback_title` = Dateiname ohne Endung (letzter Ausweg, info title_source
          = "dateiname", damit der Export-Dialog darauf hinweisen kann).
       Der Export-Dateiname aus dem Dialog kommt hier NIE an.
       Dazu ViewerPreferences /DisplayDocTitle true (PDF/UA-Anforderung:
       Anzeigeprogramme sollen den Titel statt des Dateinamens ansagen).
       Hinweis: Der Titel wird ins Info-Dictionary geschrieben; im XMP-Paket
       der Quell-PDF werden nur CreatorTool/Producer angepasst (Schritt 4),
       alles andere bleibt (bewusste Begrenzung).
    3. Verwaiste /Alt-StructElems entfernen (siehe remove_orphaned_alt_elems).

    Der Dokument-INHALT bleibt unangetastet — es geht ausschliesslich um
    Metadaten ("das Etikett der Datei") und tote Strukturobjekte.

    Gibt ein Info-Dict zurueck: {lang_set, title_set, orphan_alts_removed}.
    """
    doc = fitz.open(pdf_path)
    info = {"lang_set": False, "title_set": False, "orphan_alts_removed": 0, "title_source": ""}
    cat = doc.pdf_catalog()

    # 1) Dokumentsprache
    lang_info = doc.xref_get_key(cat, "Lang")
    has_lang = lang_info[0] == "string" and lang_info[1].strip("()<> ") != ""
    if not has_lang and lang:
        doc.xref_set_key(cat, "Lang", _pdf_string(lang))
        info["lang_set"] = True

    # 2) Titel + DisplayDocTitle
    meta = doc.metadata or {}
    existing_title = (meta.get("title") or "").strip()
    new_title = None
    if _titel_brauchbar(existing_title, filename_base):
        info["title_source"] = "quelle"
    elif title and title.strip():
        new_title = title.strip(); info["title_source"] = "dokumentname"
    elif fallback_heading and fallback_heading.strip():
        new_title = fallback_heading.strip()[:250]; info["title_source"] = "ueberschrift"
    elif fallback_title and fallback_title.strip():
        new_title = fallback_title.strip(); info["title_source"] = "dateiname"
    if new_title and new_title != existing_title:
        meta["title"] = new_title
        doc.set_metadata(meta)
        info["title_set"] = True

    vp_info = doc.xref_get_key(cat, "ViewerPreferences")
    if vp_info[0] == "xref":
        vp_xref = int(vp_info[1].split()[0])
        if doc.xref_get_key(vp_xref, "DisplayDocTitle")[1] != "true":
            doc.xref_set_key(vp_xref, "DisplayDocTitle", "true")
    elif vp_info[0] == "dict":
        if "/DisplayDocTitle true" not in vp_info[1]:
            # Vorhandenes Inline-Dict erweitern bzw. false -> true korrigieren
            val = vp_info[1]
            if "/DisplayDocTitle" in val:
                val = val.replace("/DisplayDocTitle false", "/DisplayDocTitle true")
            else:
                val = val.rstrip()[:-2].rstrip() + " /DisplayDocTitle true >>"
            doc.xref_set_key(cat, "ViewerPreferences", val)
    else:
        doc.xref_set_key(cat, "ViewerPreferences", "<< /DisplayDocTitle true >>")

    # 3) Verwaiste Alt-Altlasten
    info["orphan_alts_removed"] = remove_orphaned_alt_elems(doc, schonen=schonen)

    # 4) Dokument-Eigenschaften: Creator/Producer = nur unser Produktname (Info + XMP).
    werte = dokumentinfo_in_doc(doc, verfahren, creator)
    info["creator"] = werte["creator"]
    info["producer"] = werte["producer"]

    # In-place speichern: fitz kann nicht in die geoeffnete Datei schreiben,
    # daher Tempdatei + atomarer Austausch.
    tmp_path = pdf_path + ".finalize.tmp"
    doc.save(tmp_path)
    doc.close()
    os.replace(tmp_path, pdf_path)
    return info


def erste_ueberschrift(pdf_path: str) -> str | None:
    """Erste Ueberschrift aus dem Inhalt einer PDF als Titel-Ersatz (11.09.2026): erstes
    Lesezeichen der obersten Ebene, sonst das erste H1/H2-Strukturelement (getaggte PDF).
    None, wenn beides fehlt — dann bleibt nur der Dateiname."""
    try:
        doc = fitz.open(pdf_path)
    except Exception:  # noqa: BLE001
        return None
    try:
        for lvl, titel, _seite in (doc.get_toc(simple=True) or []):
            t = (titel or "").strip()
            if lvl == 1 and 2 <= len(t) <= 250:
                return t
        # Getaggte PDF: erstes H1/H2 ueber die Seitentexte mit Struktur (fitz liefert
        # keinen Strukturbaum; Naeherung: erste Zeile in groesster Schrift auf Seite 1).
        if doc.page_count:
            seite = doc[0]
            zeilen = []
            for block in seite.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    text = "".join(sp.get("text", "") for sp in line.get("spans", [])).strip()
                    groesse = max((sp.get("size", 0) for sp in line.get("spans", [])), default=0)
                    if 2 <= len(text) <= 120:
                        zeilen.append((groesse, text))
            if zeilen:
                groesste = max(g for g, _ in zeilen)
                normal = sorted(g for g, _ in zeilen)[len(zeilen) // 2]
                if groesste >= normal * 1.4:
                    return next(t for g, t in zeilen if g == groesste)
    except Exception:  # noqa: BLE001
        return None
    finally:
        doc.close()
    return None
