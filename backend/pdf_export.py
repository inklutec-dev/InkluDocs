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


def dokumentinfo_werte(verfahren: str | None = None) -> dict:
    """Creator/Producer fuer eine exportierte PDF — fuer alle Wege dieselben Werte.
    `verfahren` (pdfix/fitz/libreoffice) wird von den Aufrufern weiter mitgegeben, hat aber
    keinen Einfluss auf den Wortlaut: Es steht immer nur unser Produktname in der Datei."""
    return {"creator": PDF_CREATOR, "producer": PDF_PRODUCER}


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


def dokumentinfo_in_doc(doc: "fitz.Document", verfahren: str | None = None) -> dict:
    """Creator/Producer in ein geoeffnetes fitz-Dokument schreiben: Info-Dictionary UND —
    falls vorhanden — XMP-Paket. EINE Stelle fuer alle fitz-basierten Ausgaenge; das
    PDF/UA-Verfahren (pikepdf, Bytes) setzt dieselben Werte auf seinem Weg."""
    werte = dokumentinfo_werte(verfahren)
    meta = doc.metadata or {}
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


def setze_dokumentinfo(pdf_path: str, verfahren: str | None = None) -> dict:
    """Creator/Producer in eine fertige PDF schreiben (Info-Dictionary + XMP), Rest unangetastet.
    Entspricht Heines SetDocInfo-Skript (PutString auf dem Info-Objekt), hier mit PyMuPDF,
    damit alle Ausgaenge dieselbe Stelle nutzen."""
    doc = fitz.open(pdf_path)
    werte = dokumentinfo_in_doc(doc, verfahren)
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

    if has_existing_structure:
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

        for page_num in sorted(page_images.keys()):
            page = doc[page_num]
            page_figures[page_num] = []
            next_mcid = page_max_mcid.get(page_num, -1) + 1

            for img_info in page_images[page_num]:
                fig_xref = doc.get_new_xref()
                mcid = next_mcid
                next_mcid += 1
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

        # Neue Figure-Elemente als Kinder des Dokument-Knotens einhaengen. Bis 14.09.2026 wurde nur
        # ein INLINE-Array (/K [ ... ]) erkannt; bei /K als Referenz auf ein Array-Objekt (InDesign,
        # Prod-Kundendokument Projekt 430) oder als Einzelkind blieben die Figures ohne Eltern-Eintrag —
        # unerreichbar im Tag-Baum, und finalize_export_pdf raeumte sie als „verwaist“ wieder weg:
        # Export ohne Alt-Texte bei Meldung „234 getaggt“.
        if not _figures_einhaengen(doc, doc_elem_xref, figure_xrefs):
            warnings.append(
                "Die Alt-Texte konnten nicht in den Tag-Baum der PDF eingehaengt werden "
                "(unbekannte Form des /K-Eintrags am Dokument-Knoten)."
            )

        if parent_tree_xref:
            pt_obj = doc.xref_object(parent_tree_xref)
            for page_num, figs in sorted(page_figures.items()):
                page = doc[page_num]
                sp_info = doc.xref_get_key(page.xref, "StructParents")
                if sp_info[0] != "int":
                    continue
                sp_val = int(sp_info[1])

                pattern = rf'({sp_val})\s*\[(.*?)\]'
                match = re.search(pattern, pt_obj, re.DOTALL)
                if match:
                    existing_refs = match.group(2).strip()
                    new_refs = " ".join(f"{f[1]} 0 R" for f in figs)
                    updated_refs = f"{existing_refs} {new_refs}"
                    pt_obj = pt_obj[:match.start(2)] + updated_refs + pt_obj[match.end(2):]

            doc.update_object(parent_tree_xref, pt_obj)

    else:
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
            if mod['type'] == 'vector':
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

    tagged_count = 0
    alle_figuren = []
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
                        schonen: set | None = None) -> dict:
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
    werte = dokumentinfo_in_doc(doc, verfahren)
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
