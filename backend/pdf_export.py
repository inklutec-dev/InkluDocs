"""
PDF Export Module for InkluDocs (Beta).

Contains the PDF/UA-compliant alt-text tagging functionality.
Moved from pdf_processor.py to keep the main processing flow clean.
This is the "beta" PDF tagging feature - importable but separated.
"""

import re
import fitz  # PyMuPDF


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
                if alt_text == "dekorativ":
                    alt_text = ""
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

                # Set Alt on XObject as fallback
                try:
                    doc.xref_set_key(xref, "Alt", _pdf_string(alt_text))
                except Exception:
                    pass

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

            if alt_text == "dekorativ":
                alt_text = ""

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

        doc_elem_obj = doc.xref_object(doc_elem_xref)
        k_match = re.search(r'/K\s*\[(.*?)\]', doc_elem_obj, re.DOTALL)
        if k_match:
            existing_kids = k_match.group(1).strip()
            new_kids_str = " ".join(f"{x} 0 R" for x in figure_xrefs)
            updated_kids = f"{existing_kids} {new_kids_str}"
            updated_obj = doc_elem_obj[:k_match.start(1)] + updated_kids + doc_elem_obj[k_match.end(1):]
            doc.update_object(doc_elem_xref, updated_obj)

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
    for page_num, figs in page_figures.items():
        for mcid, fig_xref, img_info, content_range in figs:
            if not img_info["is_vector"] or content_range is not None:
                tagged_count += 1

    doc.save(output_path)
    doc.close()
    return {"path": output_path, "tagged_count": tagged_count, "warnings": warnings}
