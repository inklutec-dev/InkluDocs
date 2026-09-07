import fitz  # PyMuPDF
import os
import json
import hashlib
import httpx
import base64
import time
import re
from PIL import Image
# Register AVIF and HEIF support via pillow-heif
try:
    from pillow_heif import register_avif_opener, register_heif_opener
    register_avif_opener()
    register_heif_opener()
except ImportError:
    pass
from io import BytesIO


# v2.2.3: Innerhalb-Projekt-Cache fuer Duplikate (Hash -> Ergebnis)
_project_image_cache: dict[str, dict] = {}

def _get_image_hash(image_path: str) -> str:
    """Berechnet SHA-256 Hash einer Bilddatei."""
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def clear_project_cache():
    """Cache leeren (am Anfang jedes Projekts aufrufen)."""
    _project_image_cache.clear()

# PDFIX-INTEGRATION (24.04.2026): strukturelle Figure-Extraktion fuer getaggte PDFs
import logging as _logging
import pdfix_roundtrip as _pdfix
_pdfix_log = _logging.getLogger("inkludocs.pdfix")
_pipeline_log = _logging.getLogger("inkludocs.pipeline")
# END PDFIX-INTEGRATION

# Pipeline-Version: seit 07.09.2026 gibt es nur noch v4 (Klassifikation + Combo +
# Pruefpass, Claude ueber Bedrock). Die Konstante bleibt als Salz im Cache-Key,
# damit alte v3.7-Eintraege nie als Treffer zurueckkommen. Der Import auf
# Modulebene ist bewusst Fail-Fast: fehlt der Orchestrator, startet der Container nicht.
PIPELINE_VERSION = "v4"
from pipelines.v4.orchestrator import generate_alt_text_v4 as _v4_entry


def _cluster_drawings(drawings, page_rect, gap=100, min_size=50):
    """Group nearby vector drawings into clusters, return significant bounding boxes."""
    if not drawings:
        return []

    drawing_data = []  # (rect, item_count)
    for d in drawings:
        r = fitz.Rect(d["rect"])
        if r.is_empty or r.is_infinite:
            continue
        # Skip full-width lines (decorative separators)
        if r.height < 5 and r.width > page_rect.width * 0.4:
            continue
        if r.width < 5 and r.height > page_rect.height * 0.4:
            continue
        item_count = len(d.get("items", []))
        drawing_data.append((r, item_count))

    rects = [dd[0] for dd in drawing_data]

    if not rects:
        return []

    # Simple clustering: merge overlapping/nearby rectangles
    clusters = []
    used = set()

    for i, r1 in enumerate(rects):
        if i in used:
            continue
        cluster_rect = fitz.Rect(r1)
        cluster = {i}
        changed = True
        while changed:
            changed = False
            expanded = fitz.Rect(cluster_rect.x0 - gap, cluster_rect.y0 - gap,
                                  cluster_rect.x1 + gap, cluster_rect.y1 + gap)
            for j, r2 in enumerate(rects):
                if j in cluster or j in used:
                    continue
                if expanded.intersects(r2):
                    cluster.add(j)
                    cluster_rect = cluster_rect | r2
                    changed = True
        used.update(cluster)

        # Skip clusters with only 1 drawing (likely decorative line/box)
        if len(cluster) < 2:
            continue

        # Count total path segments in cluster - simple shapes (boxes, lines) have very few
        total_items = sum(drawing_data[idx][1] for idx in cluster)
        if total_items < 3:
            continue  # Simple rectangles/lines, not a real graphic

        # Only keep clusters that are significant
        if cluster_rect.width >= min_size and cluster_rect.height >= min_size:
            # Add generous padding to capture axis labels, legends, titles
            pad = 60
            padded = fitz.Rect(cluster_rect.x0 - pad, cluster_rect.y0 - pad,
                               cluster_rect.x1 + pad, cluster_rect.y1 + pad)
            padded = padded & page_rect  # clip to page
            clusters.append(padded)

    return clusters


def _is_caption(text):
    """Detect if a text block is likely a figure caption."""
    caption_patterns = [
        r'^(?:Abbildung|Abb\.?|Bild|Grafik|Tabelle|Tab\.?|Diagramm|Figur|Figure|Fig\.?|Table|Chart|Image)\s*\d',
        r'^(?:Quelle|Source)\s*:',
    ]
    for pat in caption_patterns:
        if re.match(pat, text.strip(), re.IGNORECASE):
            return True
    return False


def _get_nearby_text(page, bbox, max_chars=1200):
    """Extract text context for an image on a PDF page.
    
    Strategy: Use FULL page text as context (truncated to max_chars).
    This ensures that text at the page bottom (e.g. printed names under signatures,
    footnotes, captions far from the image) is included. The model decides what's relevant.
    
    For pages with bbox: Prioritizes captions and nearby text, then fills up with
    remaining page text until max_chars is reached.
    
    Updated 2026-03-24: Changed from radius-based to full-page context (Claude suggestion).
    """
    page_text = page.get_text()
    if not page_text or not page_text.strip():
        return "Kein Textkontext verfuegbar."

    if not bbox:
        return f"[Seitentext] {page_text[:max_chars]}"

    img_rect = fitz.Rect(bbox)
    # Get all text blocks with positions: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")
    if not blocks:
        return f"[Seitentext] {page_text[:max_chars]}"

    text_blocks = []
    for b in blocks:
        if b[6] != 0:  # block_type 0 = text
            continue
        block_rect = fitz.Rect(b[0], b[1], b[2], b[3])
        block_text = b[4].strip()
        if not block_text:
            continue
        # Calculate vertical distance to image
        if block_rect.y1 <= img_rect.y0:
            distance = img_rect.y0 - block_rect.y1
            position = "before"
        elif block_rect.y0 >= img_rect.y1:
            distance = block_rect.y0 - img_rect.y1
            position = "after"
        else:
            distance = 0
            position = "overlap"

        is_cap = _is_caption(block_text)
        sort_distance = 0 if is_cap else distance
        text_blocks.append((sort_distance, distance, position, block_text, is_cap))

    # Sort by sort_distance (captions first), then by actual distance
    text_blocks.sort(key=lambda x: (x[0], x[1]))

    # Reserve 400 chars specifically for page-end text (signatures, names, dates, footnotes).
    # This ensures text at the bottom of the page always gets into context, even if
    # overlapping blocks fill up the budget from the top.
    PAGE_END_RESERVE = 400
    block_budget = max_chars - PAGE_END_RESERVE

    # Build context: captions and closest blocks first
    context_parts = []
    chars_used = 0

    for sort_dist, distance, position, text, is_cap in text_blocks:
        if chars_used >= block_budget:
            break
        remaining = block_budget - chars_used
        snippet = text[:remaining]
        if is_cap:
            context_parts.append(f"[Bildunterschrift] {snippet}")
        elif position == "before":
            context_parts.append(f"[Text davor] {snippet}")
        elif position == "after":
            context_parts.append(f"[Text danach] {snippet}")
        else:
            context_parts.append(f"[Ueberlappend] {snippet}")
        chars_used += len(snippet) + 20

    # Always append the last PAGE_END_RESERVE chars of the page
    # (catches signatures, printed names, footnotes, dates at page bottom)
    page_end = page_text[-PAGE_END_RESERVE:].strip() if len(page_text) > PAGE_END_RESERVE else page_text.strip()
    if page_end:
        context_parts.append(f"[Text am Seitenende] {page_end}")

    return "\n".join(context_parts) if context_parts else "Kein Textkontext verfuegbar."


def _merge_nearby_images(page, image_list, doc, gap=20):
    """Merge raster images that are spatially close on the same page.
    Returns list of (merged_rect, [xrefs]) tuples for groups, plus individual images."""
    if len(image_list) <= 1:
        return None  # No merging needed

    # Collect bounding boxes for all images
    img_rects = []
    for img_info in image_list:
        xref = img_info[0]
        for rect in page.get_image_rects(xref):
            img_rects.append((rect, xref))

    if len(img_rects) <= 1:
        return None

    # Cluster nearby images (same logic as _cluster_drawings but for raster images)
    used = set()
    groups = []
    for i, (r1, x1) in enumerate(img_rects):
        if i in used:
            continue
        group_rect = fitz.Rect(r1)
        group_xrefs = [x1]
        group_indices = {i}
        changed = True
        while changed:
            changed = False
            expanded = fitz.Rect(group_rect.x0 - gap, group_rect.y0 - gap,
                                  group_rect.x1 + gap, group_rect.y1 + gap)
            for j, (r2, x2) in enumerate(img_rects):
                if j in group_indices or j in used:
                    continue
                if expanded.intersects(r2):
                    group_indices.add(j)
                    group_xrefs.append(x2)
                    group_rect = group_rect | r2
                    changed = True
        used.update(group_indices)
        groups.append((group_rect, group_xrefs))

    # Only return if we actually merged (at least one group with >1 images)
    has_merged = any(len(xrefs) > 1 for _, xrefs in groups)
    return groups if has_merged else None


def _extract_via_pdfix(pdf_path: str, output_dir: str) -> list:
    """PDFIX-INTEGRATION: Figures via PDFix-SDK extrahieren (Heines Script).

    Mappt das von Heines Export gelieferte Format auf die InkluDocs-Struktur,
    die die Alt-Text-Pipeline erwartet. Fuer jede Figure wird ein dict geliefert
    wie beim fitz-Pfad, damit der Downstream-Code unveraendert bleibt.

    Erweiterung 27.05.2026: page_view_path + page_text fuer UI-Vorschau.
    """
    figures = _pdfix.extract_figures_pdfix(pdf_path, output_dir)
    full_text = ""
    page_texts: dict[int, str] = {}  # sortiert (Lesereihenfolge), nur fuer UI-Vorschau
    try:
        _doc = fitz.open(pdf_path)
        _ctx_texts = []  # Standard-Reihenfolge: KI-Kontext byte-identisch lassen
        for i, p in enumerate(_doc):
            page_texts[i + 1] = p.get_text(sort=True)
            _ctx_texts.append(p.get_text())
        full_text = "\n".join(_ctx_texts)
        _doc.close()
    except Exception:
        pass
    images = []
    for fig in figures:
        width, height = 0, 0
        try:
            with Image.open(fig["path"]) as _im:
                width, height = _im.size
        except Exception:
            pass
        page_num = fig.get("page_number", 1)
        # Kontext-Quelle (19.06.2026, Karbe V1004): bevorzugt der Kapitel-Kontext
        # (Abschnitt um das Bild), sonst der Seiteninhalt, sonst der bisherige
        # Ganz-Dokument-Text (full_text). So nie schlechter als zuvor, bei gut
        # getaggten PDFs deutlich bildgenauer. OB dieser Kontext der KI gegeben
        # wird, entscheidet SPAETER der Projekt-Schalter use_context (main.py) --
        # hier wird er nur bereitgestellt und gespeichert.
        _chapter = fig.get("chapter_context", "")
        _page_content = fig.get("page_content", "")
        _ctx = _chapter or _page_content or full_text
        images.append({
            "page_number": page_num,
            "image_index": fig["lfnr"],
            "image_path": fig["path"],
            "image_filename": os.path.basename(fig["path"]),
            "width": width,
            "height": height,
            "xref": -fig["lfnr"],
            "context_text": _ctx,
            "ext": "png",
            "bbox": (0, 0, width, height),
            "is_vector": True,
            "source": "pdfix",
            "original_alt": fig.get("alt", ""),
            "page_view_path": fig.get("page_view_path", ""),
            "page_text": _page_content or page_texts.get(page_num, ""),
        })
    return images


def extract_images_from_pdf(pdf_path: str, output_dir: str, project_id: int) -> list:
    """Extract all images from a PDF, including vector graphics rendered as images."""
    # PDFIX-INTEGRATION (24.04.2026): bei getaggter PDF ueber PDFix-SDK gehen.
    # Feature-Flag PDFIX_ENABLED (default false). Fallback auf fitz bei Fehler.
    if os.getenv("PDFIX_ENABLED", "false").lower() in ("1", "true", "yes"):
        if _pdfix.is_pdfix_available() and _pdfix.is_tagged_pdf(pdf_path):
            try:
                figures = _extract_via_pdfix(pdf_path, output_dir)
                _pdfix_log.info("PDFix-Pfad: %d Figures aus %s (project_id=%s)",
                                len(figures), pdf_path, project_id)
                return figures
            except Exception as e:
                _pdfix_log.warning(
                    "PDFix-Extraktion fehlgeschlagen, Fallback auf fitz: %s", e)
    # END PDFIX-INTEGRATION
    doc = fitz.open(pdf_path)
    images = []
    vector_xref_counter = 900000  # High xref range for vector graphics
    img_idx = 0  # Globaler Counter ueber alle Seiten (27.05.2026)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text(sort=True)  # UI-Vorschau in Lesereihenfolge (28.05.2026)
        image_list = page.get_images(full=True)

        # Seitenansicht-PNG: 1x pro Seite, ~144 DPI (Matrix 2.0) fuer lesbaren Text
        page_view_filename = f"p{page_num + 1}_seitenansicht.png"
        page_view_path = os.path.join(output_dir, page_view_filename)
        if not os.path.exists(page_view_path):
            try:
                _pv_pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                _pv_pix.save(page_view_path)
            except Exception as _e:
                print(f"Seitenansicht-Render fehlgeschlagen Seite {page_num + 1}: {_e}")
                page_view_path = ""

        # Track bounding boxes of raster images to avoid duplicating
        raster_areas = []

        # Check if images should be merged (e.g. split chemical formulas)
        merged_groups = _merge_nearby_images(page, image_list, doc)
        if merged_groups:
            for group_rect, group_xrefs in merged_groups:
                if len(group_xrefs) > 1:
                    # Render merged region as single image
                    img_idx += 1
                    pad = 10
                    clip = fitz.Rect(group_rect.x0 - pad, group_rect.y0 - pad,
                                     group_rect.x1 + pad, group_rect.y1 + pad) & page.rect
                    mat = fitz.Matrix(3, 3)
                    pix = page.get_pixmap(matrix=mat, clip=clip)
                    img_filename = f"p{page_num + 1}_merged{img_idx}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    pix.save(img_path)

                    raster_areas.append(group_rect)
                    context = _get_nearby_text(page, (group_rect.x0, group_rect.y0, group_rect.x1, group_rect.y1))

                    images.append({
                        "page_number": page_num + 1,
                        "image_index": img_idx,
                        "image_path": img_path,
                        "image_filename": img_filename,
                        "width": int(group_rect.width * 2),
                        "height": int(group_rect.height * 2),
                        "xref": group_xrefs[0],
                        "context_text": context,
                        "ext": "png",
                        "bbox": (group_rect.x0, group_rect.y0, group_rect.x1, group_rect.y1),
                        "is_vector": False,
                        "page_view_path": page_view_path,
                        "page_text": page_text,
                    })
                    continue

                # Single image in group – process normally below

        # Collect xrefs that were already merged
        merged_xrefs = set()
        if merged_groups:
            for _, group_xrefs in merged_groups:
                if len(group_xrefs) > 1:
                    merged_xrefs.update(group_xrefs)

        # 1. Extract raster images (skip if already merged)
        for img_info in image_list:
            xref = img_info[0]
            if xref in merged_xrefs:
                continue  # Already processed as merged image
            img_idx += 1

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < 20 or height < 20:
                    continue

                # Convert JPX/JP2 (JPEG 2000) and other exotic formats to PNG
                if image_ext in ("jpx", "jp2", "jbig2", "j2k"):
                    try:
                        tmp_path = os.path.join(output_dir, f"_tmp_{xref}.{image_ext}")
                        with open(tmp_path, "wb") as f:
                            f.write(image_bytes)
                        with Image.open(tmp_path) as pil_img:
                            png_path = os.path.join(output_dir, f"p{page_num + 1}_img{img_idx}.png")
                            pil_img.convert("RGB").save(png_path, format="PNG")
                        os.remove(tmp_path)
                        image_ext = "png"
                        img_filename = f"p{page_num + 1}_img{img_idx}.png"
                        img_path = png_path
                        print(f"Converted JPX/JP2 image (xref {xref}) to PNG")
                    except Exception as e:
                        print(f"JPX/JP2 conversion failed (xref {xref}): {e} – rendering from page instead")
                        # Fallback: render the image area from the page as pixmap
                        try:
                            for img_rect in page.get_image_rects(xref):
                                mat = fitz.Matrix(3, 3)
                                pix = page.get_pixmap(matrix=mat, clip=img_rect)
                                png_path = os.path.join(output_dir, f"p{page_num + 1}_img{img_idx}.png")
                                pix.save(png_path)
                                image_ext = "png"
                                img_filename = f"p{page_num + 1}_img{img_idx}.png"
                                img_path = png_path
                                break
                        except Exception as e2:
                            print(f"Page render fallback also failed (xref {xref}): {e2}")
                            continue
                else:
                    img_filename = f"p{page_num + 1}_img{img_idx}.{image_ext}"
                    img_path = os.path.join(output_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)

                # Get bounding box for this raster image
                img_bbox = None
                for img_rect in page.get_image_rects(xref):
                    raster_areas.append(img_rect)
                    if img_bbox is None:
                        img_bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)

                context = _get_nearby_text(page, img_bbox)

                images.append({
                    "page_number": page_num + 1,
                    "image_index": img_idx,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "width": width,
                    "height": height,
                    "xref": xref,
                    "context_text": context,
                    "ext": image_ext,
                    "bbox": img_bbox,
                    "is_vector": False,
                    "page_view_path": page_view_path,
                    "page_text": page_text,
                })

            except Exception as e:
                print(f"Error extracting image {xref} from page {page_num + 1}: {e}")
                continue

        # 2. Detect and render vector graphics (charts, diagrams, icons)
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []

        if drawings:
            clusters = _cluster_drawings(drawings, page.rect)

            for cluster_rect in clusters:
                # Skip if this area overlaps significantly with a raster image
                overlaps_raster = False
                for ra in raster_areas:
                    intersection = cluster_rect & ra
                    if not intersection.is_empty:
                        overlap_area = intersection.width * intersection.height
                        cluster_area = cluster_rect.width * cluster_rect.height
                        if cluster_area > 0 and overlap_area / cluster_area > 0.5:
                            overlaps_raster = True
                            break
                if overlaps_raster:
                    continue

                # Render this region as a PNG image
                img_idx += 1
                try:
                    # Scale factor: 3x for HD rendering, lower for very large ones
                    cw, ch = cluster_rect.width, cluster_rect.height
                    scale = 3.0
                    if cw * scale > MAX_IMAGE_DIM or ch * scale > MAX_IMAGE_DIM:
                        scale = min(MAX_IMAGE_DIM / cw, MAX_IMAGE_DIM / ch)
                        scale = max(scale, 1.0)  # at least 1x
                    mat = fitz.Matrix(scale, scale)
                    pixmap = page.get_pixmap(matrix=mat, clip=cluster_rect)
                    img_filename = f"p{page_num + 1}_vec{img_idx}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    pixmap.save(img_path)

                    vector_xref_counter += 1
                    vec_bbox = (cluster_rect.x0, cluster_rect.y0, cluster_rect.x1, cluster_rect.y1)
                    context = _get_nearby_text(page, vec_bbox)

                    images.append({
                        "page_number": page_num + 1,
                        "image_index": img_idx,
                        "image_path": img_path,
                        "image_filename": img_filename,
                        "width": int(cluster_rect.width),
                        "height": int(cluster_rect.height),
                        "xref": vector_xref_counter,
                        "context_text": context,
                        "ext": "png",
                        "bbox": (cluster_rect.x0, cluster_rect.y0, cluster_rect.x1, cluster_rect.y1),
                        "is_vector": True,
                        "page_view_path": page_view_path,
                        "page_text": page_text,
                    })
                    print(f"Vector graphic on page {page_num + 1}: {int(cluster_rect.width)}x{int(cluster_rect.height)}px")
                except Exception as e:
                    print(f"Error rendering vector graphic on page {page_num + 1}: {e}")
                    continue

    doc.close()
    return images


MAX_IMAGE_DIM = 1536  # laengste Kante fuer den Modellaufruf (Bedrock/Claude); bei niedrigerer Aufloesung halluziniert das Modell Details auf kleinen Objekten (z.B. orangefarbene Token werden zu "Getraenken"). Seit 28.04.2026 1536 statt 1024.
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB Obergrenze fuer den Modellaufruf
MIN_UPSCALE_DIM = int(os.environ.get("V4_UPSCALE_MIN", "800"))  # kleine Bilder bis zu dieser Kante hochskalieren; 0 = aus


MAX_ALT_TEXT_LENGTH = 400  # Characters - enough for key info, not overwhelming for screen readers


def _combine_alt_text(alt_text: str, langbeschreibung: str) -> str:
    """Return only the short alt-text, trimmed to max 350 chars at sentence boundary."""
    if not alt_text:
        return ""
    text = alt_text.strip()
    if len(text) > 350:
        cut = text[:350]
        last_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if last_end > 60:
            text = cut[:last_end + 1]
        else:
            text = cut.rstrip() + "..."
    return text


def _ocr_extract_text(image_path: str) -> str:
    """Extract text from an image using Tesseract OCR.
    Returns extracted text or empty string if OCR fails or finds nothing."""
    try:
        import pytesseract
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="deu+eng", config="--psm 6")
        text = text.strip()
        if len(text) < 3:
            return ""
        return text[:1000]
    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""


def _resize_image_for_model(image_path: str) -> str:
    """Resize image if too large for the model, return base64 encoded string."""
    try:
        img = Image.open(image_path)
    except Exception as e:
        # If image can't be opened (corrupted, unsupported format), try converting via fitz
        print(f"PIL cannot open {image_path}: {e} – attempting fitz render")
        try:
            import fitz as fitz_fallback
            pix = fitz_fallback.Pixmap(image_path)
            if pix.n > 4:
                pix = fitz_fallback.Pixmap(fitz_fallback.csRGB, pix)
            png_data = pix.tobytes("png")
            return base64.b64encode(png_data).decode()
        except Exception:
            raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")
    # Convert palette/RGBA/LA modes to RGB for JPEG compatibility.
    # Transparente Pixel dabei auf WEISS legen — ein blosses convert("RGB")
    # macht sie schwarz, wodurch dunkle Beschriftungen transparenter PNGs
    # fuer das Modell unsichtbar werden (Befund App-Durchlauf 17.07.2026).
    konvertiert = False  # 07.09.2026 (Astra-Befund): Modus-Konvertierung muss auch ohne Groessenaenderung ankommen
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
        konvertiert = True
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
        konvertiert = True
    # Force-convert AVIF/HEIC to JPEG (Ollama and other tools cannot read these formats)
    if image_path.lower().endswith((".avif", ".heic", ".heif")):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    # Kleine Bilder hochskalieren (07.09.2026, Pruefkorpus-Befund: Word-eingebettete
    # Bilder kommen mit 150 bis 450 px an — Distelfink 151x231, Diagramm 462x200 —
    # und das Modell erfindet dann Details oder liest Werte falsch). Bis zur
    # Zielkante MIN_UPSCALE_DIM mit Lanczos vergroessern, verlustfrei als PNG.
    # Schalter V4_UPSCALE_MIN (0 = aus) fuer A/B-Messungen.
    laengste = max(img.width, img.height)
    if 0 < MIN_UPSCALE_DIM and 0 < laengste < MIN_UPSCALE_DIM:
        faktor = min(4, -(-MIN_UPSCALE_DIM // laengste))  # ceil, hoechstens x4
        img = img.resize((img.width * faktor, img.height * faktor), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    # Resize if dimensions exceed limit
    if img.width > MAX_IMAGE_DIM or img.height > MAX_IMAGE_DIM:
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
        buf = BytesIO()
        fmt = "JPEG" if image_path.lower().endswith((".jpg", ".jpeg")) else "PNG"
        img.save(buf, format=fmt, quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    # Check file size - large PNGs from vector rendering can be huge
    file_size = os.path.getsize(image_path)
    if file_size > MAX_IMAGE_BYTES:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    if konvertiert:
        # Bis 07.09.2026 wurden hier die ORIGINALBYTES gelesen — die auf Weiss gelegte
        # Fassung ging verloren, sobald weder Vergroesserung, Verkleinerung noch
        # Kompression griff (transparente 900x900-PNGs kamen als RGBA beim Modell an).
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_alt_text(image_path: str, context: str = "", image_type: str = None,
                      width: int = 0, height: int = 0, original_alt: str = "",
                      force_regenerate: bool = False, temperature: float = 0.0,
                      language: str = "de", previous_alt: str = "",
                      user_prompt: str = "") -> dict:
    """Front-Door — Cache-Check, dann die v4-Pipeline.

    T6 (03.05.2026): persistenter Content-Hash-Cache vor dem Pipeline-Aufruf.
      - force_regenerate=True: Cache uebersprungen, Pipeline laeuft, Result wird gecacht.
      - force_regenerate=False: Cache-Treffer liefert direkt zurueck (kein Modellaufruf).
    Der Slot user_hint im Cache-Key traegt den eigenen Nutzer-Prompt (06.07.2026).
    """
    from cache import build_cache_key, get_cached, set_cached

    content_hash = _get_image_hash(image_path)
    # Eigener Prompt (06.07.2026) geht in den Key (Slot user_hint): anderer Prompt =
    # anderer Cache-Eintrag; ohne Prompt (None) bleiben alle Alt-Eintraege gueltig.
    cache_key = build_cache_key(content_hash, image_type, (user_prompt or None), PIPELINE_VERSION, language=language, enriched_context=(context or ""))

    if not force_regenerate:
        cached = get_cached(cache_key)
        if cached is not None:
            print(f"T6 cache-hit: {image_path} (key={cache_key[:48]}...)")
            # Abo-Etappe-1: Cache-Treffer als solche markieren — die Verbuchungs-
            # Stellen (billing.verbuche) lassen markierte Ergebnisse kostenlos.
            cached["from_cache"] = True
            return cached

    # Cache-Miss oder force_regenerate -> Pipeline rufen
    result = _v4_entry(
        image_path,
        enriched_context=context,
        image_type_override=image_type,
        width=width,
        height=height,
        original_alt=original_alt,
        temperature=temperature,
        language=language,
        previous_alt=previous_alt,
        user_prompt=user_prompt,
    )

    # Auch bei force_regenerate: Cache neu befuellen, damit nachfolgende Anfragen Hits werden
    set_cached(cache_key, content_hash, PIPELINE_VERSION, image_type, None, result)
    return result


def generate_alt_text_for_image(image_path: str, context_text: str = "", image_type: str = None,
                                width: int = 0, height: int = 0, original_alt: str = "",
                                language: str = "de") -> dict:
    """Generate alt-text for a standalone image. Uses the same v4 pipeline.
    v2.2: Passes through width/height/original_alt for thumbnail/improvement modes.
    """
    return generate_alt_text(image_path, context=context_text, image_type=image_type,
                             width=width, height=height, original_alt=original_alt,
                             language=language)
