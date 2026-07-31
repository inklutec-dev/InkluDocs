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

from context_engine import get_prompt, clean_alt_text, remove_hedge_words

# PDFIX-INTEGRATION (24.04.2026): strukturelle Figure-Extraktion fuer getaggte PDFs
import logging as _logging
import pdfix_roundtrip as _pdfix
_pdfix_log = _logging.getLogger("inkludocs.pdfix")
_pipeline_log = _logging.getLogger("inkludocs.pipeline")
# END PDFIX-INTEGRATION

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")

# v3.3 (14.04.2026): Dreistufige Mistral-Pipeline
# Alle drei Modelle ueber Env-Variablen austauschbar, damit wir ohne Code-Aenderung
# nachsteuern koennen wenn Qualitaetstests andere Ergebnisse zeigen.
MISTRAL_MODEL_CLASSIFY = os.environ.get("MISTRAL_MODEL_CLASSIFY", "mistral-medium-latest")
MISTRAL_MODEL_GENERATE = os.environ.get("MISTRAL_MODEL_GENERATE",
                                         os.environ.get("MISTRAL_MODEL", "pixtral-large-latest"))
MISTRAL_MODEL_VALIDATE = os.environ.get("MISTRAL_MODEL_VALIDATE", "mistral-large-latest")
# "correct" = Validator darf ueberschreiben, "flag" = nur needs_review setzen
VALIDATOR_MODE = os.environ.get("VALIDATOR_MODE", "correct").lower()
VALIDATOR_ENABLED = os.environ.get("VALIDATOR_ENABLED", "true").lower() in ("true", "1", "yes")

# T5 (03.05.2026, Phase A): Pipeline-Version Front-Door-Routing.
# Default v3_7 = aktuelle dreistufige Mistral-Pipeline. v4 = neue 4-Pass-Architektur (in Bau).
# Bei PIPELINE_VERSION=v4 muss das v4-Modul beim Container-Start importierbar sein — sonst
# Fail-Fast (uvicorn startet nicht). Begruendung: stiller Fallback auf v3_7 wuerde Audit-Trail
# verfaelschen (Behoerden-Compliance), und ein Operator der v4 setzt muss wissen wenn v4 fehlt.
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "v3_7").lower()
_KNOWN_PIPELINE_VERSIONS = ("v3_7", "v4")
if PIPELINE_VERSION not in _KNOWN_PIPELINE_VERSIONS:
    raise RuntimeError(
        f"PIPELINE_VERSION={PIPELINE_VERSION!r} unbekannt. "
        f"Gueltige Werte: {_KNOWN_PIPELINE_VERSIONS}. Container-Start abgebrochen."
    )
if PIPELINE_VERSION == "v4":
    try:
        from pipelines.v4.orchestrator import generate_alt_text_v4 as _v4_entry
    except ImportError as e:
        raise RuntimeError(
            f"PIPELINE_VERSION=v4 gesetzt, aber pipelines.v4.orchestrator.generate_alt_text_v4 "
            f"nicht importierbar: {e}. Container-Start abgebrochen."
        ) from e

# v2.2.1: Prompt fuer Dekorativ-Recheck (zweiter Qwen-Call bei unsicherer Klassifikation)
DEKORATIV_RECHECK_PROMPT = """/no_think
Ist dieses Bild rein dekorativ (abstrakte Form, Trennlinie, Farbverlauf, Schmuckelement) oder enthaelt es inhaltliche Information (Personen, Objekte, Text, Symbole)?
Antworte NUR mit diesem JSON:
{"ist_dekorativ": true} oder {"ist_dekorativ": false, "kurzbeschreibung": "Max 150 Zeichen, beschreibt was zu sehen ist"}"""


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
    die die Mistral-Pipeline erwartet. Fuer jede Figure wird ein dict geliefert
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


MAX_IMAGE_DIM = 1536  # v3.7 (28.04.2026): erhoeht von 1024 (Qwen-Legacy) auf 1536. Mistral Large 3 / Pixtral-Large akzeptieren mehr — bei niedrigerer Aufloesung halluziniert das Modell Details auf kleinen Objekten (z.B. orangefarbene Token werden zu "Getraenken").
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB max for Ollama


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
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # Force-convert AVIF/HEIC to JPEG (Ollama and other tools cannot read these formats)
    if image_path.lower().endswith((".avif", ".heic", ".heif")):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
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
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


MIN_IMAGE_SIZE = 50  # Minimum dimension in pixels for KI analysis


def _call_ollama(image_path: str, prompt: str) -> dict:
    """Send an image + prompt to Ollama and return the parsed result dict."""
    # Skip tiny images that crash Ollama (tracking pixels, spacers)
    try:
        with Image.open(image_path) as check_img:
            if check_img.width < MIN_IMAGE_SIZE or check_img.height < MIN_IMAGE_SIZE:
                return {
                    "bildtyp": "dekorativ",
                    "alt_text": "",
                    "ist_dekorativ": True,
                    "konfidenz": "hoch",
                }
    except Exception:
        pass

    img_b64 = _resize_image_for_model(image_path)

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": 4000,
                },
            },
            timeout=300.0,
        )
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")
        thinking_text = result.get("thinking", "")
        text = response_text or thinking_text

        # Strip <think>...</think> blocks
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not clean_text:
            clean_text = text

        # If response is empty but thinking has content, extract alt-text from thinking
        if not response_text and thinking_text:
            alt_patterns = [
                r'"alt_text":\s*"([^"]+)"',
                r'[Aa]lt[_-]?[Tt]ext:\s*"([^"]+)"',
                r'[Aa]lt[_-]?[Tt]ext\s+(?:should|would|could|is|shall)\s+be\s*"([^"]+)"',
                r'the\s+alt[_-]?\s*text\s+(?:is|should be|would be)\s*"([^"]+)"',
                r'[Aa]lt[_-]?[Tt]ext\s+(?:waere|ist|lautet|sollte sein)\s*"([^"]+)"',
                r'[Aa]lt[_-]?[Tt]ext[^"]*"([^"]{15,})"',
                r'[Aa]lt[_-]?[Tt]ext:\s*(.+?)(?:\n|$)',
            ]
            found_alt = None
            for pat in alt_patterns:
                m = re.search(pat, thinking_text)
                if m and len(m.group(1).strip()) > 10:
                    found_alt = m.group(1).strip().strip('"').strip('.')
                    if any(kw in found_alt.lower() for kw in ['should be', 'would be', 'the user', 'according to', 'the rules say']):
                        found_alt = None
                        continue
                    break

            bildtyp = "unbekannt"
            typ_match = re.search(r'"bildtyp":\s*"([^"]+)"', thinking_text)
            if not typ_match:
                typ_match = re.search(r'[Bb]ildtyp[:\s]+["\']?(\w+)', thinking_text)
            if not typ_match:
                typ_map = {'logo': 'logo', 'foto': 'foto', 'photo': 'foto',
                           'diagramm': 'diagramm', 'chart': 'diagramm', 'graph': 'diagramm',
                           'tabelle': 'tabelle', 'table': 'tabelle',
                           'screenshot': 'screenshot', 'banner': 'screenshot',
                           'icon': 'icon', 'dekorativ': 'dekorativ', 'decorative': 'dekorativ'}
                for keyword, typ in typ_map.items():
                    if keyword in thinking_text.lower():
                        bildtyp = typ
                        break
            if typ_match:
                bildtyp = typ_match.group(1).strip()

            if found_alt:
                return {
                    "bildtyp": bildtyp,
                    "alt_text": found_alt,
                    "ist_dekorativ": "dekorativ" in found_alt.lower() or bildtyp == "dekorativ",
                    "raw_response": thinking_text,
                }

        # Try to parse JSON from cleaned response
        try:
            json_matches = list(re.finditer(r'\{[^{}]*"alt_text"[^{}]*\}', clean_text))
            if json_matches:
                parsed = json.loads(json_matches[-1].group())
                alt = _combine_alt_text(parsed.get("alt_text", ""), parsed.get("langbeschreibung", ""))
                return {
                    "bildtyp": parsed.get("bildtyp", "unbekannt"),
                    "alt_text": alt,
                    "langbeschreibung": parsed.get("langbeschreibung", ""),
                    "ist_dekorativ": parsed.get("ist_dekorativ", False),
                    "konfidenz": parsed.get("konfidenz", "mittel"),
                    "raw_response": text,
                }
            start = clean_text.find("{")
            end = clean_text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(clean_text[start:end])
                # Classification response (has bildtyp but no alt_text)
                if parsed.get("bildtyp") and parsed.get("alt_text") is None:
                    return {
                        "bildtyp": parsed["bildtyp"],
                        "alt_text": "",
                        "ist_dekorativ": parsed.get("ist_dekorativ", False),
                        "konfidenz": parsed.get("konfidenz", "mittel"),
                        "raw_response": text,
                    }
                # Generation response (has alt_text)
                if parsed.get("alt_text") is not None:
                    alt = _combine_alt_text(parsed.get("alt_text", ""), parsed.get("langbeschreibung", ""))
                    return {
                        "bildtyp": parsed.get("bildtyp", "unbekannt"),
                        "alt_text": alt,
                        "langbeschreibung": parsed.get("langbeschreibung", ""),
                        "ist_dekorativ": parsed.get("ist_dekorativ", False),
                        "konfidenz": parsed.get("konfidenz", "mittel"),
                        "raw_response": text,
                    }
        except (json.JSONDecodeError, AttributeError):
            pass

        fallback_text = clean_text
        for pattern in [r'```json\s*', r'\s*```', r'^\s*\{.*\}\s*$']:
            fallback_text = re.sub(pattern, '', fallback_text, flags=re.DOTALL)
        fallback_text = fallback_text.strip()

        if not fallback_text or len(fallback_text) < 5:
            fallback_text = clean_text.strip()

        return {
            "bildtyp": "unbekannt",
            "alt_text": fallback_text if fallback_text else f"[Modell-Antwort konnte nicht verarbeitet werden: {text[:200]}]",
            "ist_dekorativ": False,
            "raw_response": text,
        }
    except Exception as e:
        return {
            "bildtyp": "fehler",
            "alt_text": f"Fehler bei der Analyse: {str(e)}",
            "ist_dekorativ": False,
            "raw_response": str(e),
        }


def generate_alt_text(image_path: str, context: str = "", image_type: str = None,
                      width: int = 0, height: int = 0, original_alt: str = "",
                      force_regenerate: bool = False, temperature: float = 0.0,
                      language: str = "de", previous_alt: str = "",
                      user_prompt: str = "") -> dict:
    """Front-Door — Cache-Check + Routing zur konfigurierten Pipeline-Version.

    T5 (03.05.2026, Phase A): explizites Routing v3_7|v4 ueber PIPELINE_VERSION env.
    T6 (03.05.2026, Phase A): persistenter Content-Hash-Cache vor Pipeline-Aufruf.
      - force_regenerate=True: Cache uebersprungen, Pipeline laeuft, Result wird gecacht.
      - force_regenerate=False: Cache-Hit liefert direkt zurueck (kein Mistral-Call).

    user_hint kommt erst mit v4 (Chatbot-Modus, Sektion 8.1 der Architektur-Doku).
    Solange v3.7 keinen user_hint kennt, ist er hier None — der Cache-Key bleibt
    dadurch stabil mit dem zukuenftigen v4-Aufruf-Pattern.
    """
    from cache import build_cache_key, get_cached, set_cached

    content_hash = _get_image_hash(image_path)
    # Eigener Prompt (06.07.2026) geht in den Key (Slot user_hint): anderer Prompt =
    # anderer Cache-Eintrag; ohne Prompt (None) bleiben alle Alt-Eintraege gueltig.
    cache_key = build_cache_key(content_hash, image_type, (user_prompt or None), PIPELINE_VERSION, language=language)

    if not force_regenerate:
        cached = get_cached(cache_key)
        if cached is not None:
            print(f"T6 cache-hit: {image_path} (key={cache_key[:48]}...)")
            # Abo-Etappe-1: Cache-Treffer als solche markieren — die Verbuchungs-
            # Stellen (billing.verbuche) lassen markierte Ergebnisse kostenlos.
            cached["from_cache"] = True
            return cached

    # Cache-Miss oder force_regenerate -> Pipeline rufen
    if PIPELINE_VERSION == "v4":
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
    else:
        # Hinweis Ausgabesprache: der v3.7-Mistral-Fallback kennt keinen
        # language-Parameter und liefert immer Deutsch (bewusst so belassen).
        from pipelines.v3_7 import generate_alt_text_v3_7
        result = generate_alt_text_v3_7(image_path, context, image_type, width, height, original_alt)

    # Auch bei force_regenerate: Cache neu befuellen, damit nachfolgende Anfragen Hits werden
    set_cached(cache_key, content_hash, PIPELINE_VERSION, image_type, None, result)
    return result


def _call_mistral_json(image_path: str, prompt: str, model: str, max_tokens: int = 400) -> dict | None:
    """v3.3: Generic Mistral API call returning parsed JSON dict.
    Used for Stufe 1 (Klassifikation) and Stufe 3 (Validierung).
    Returns None on any failure."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return None

    # Strip Qwen-specific /no_think directive
    clean_prompt = prompt.replace("/no_think", "").strip()

    try:
        img_b64 = _resize_image_for_model(image_path)
        response = httpx.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": clean_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        # Strip code fences and control chars
        clean = re.sub(r"```json\s*", "", text)
        clean = re.sub(r"\s*```", "", clean)
        clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", clean)
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            return json.loads(clean[start:end].replace("\n", " ").replace("\t", " "))
    except Exception as e:
        print(f"Mistral JSON-Call Fehler ({model}) fuer {image_path}: {type(e).__name__}: {e}")
        return None


def _call_mistral_classify(image_path: str, classification_prompt: str) -> dict:
    """v3.3 Stufe 1: Klassifikation ueber Mistral API.
    Fallback-Default bei Fehler: bildtyp='foto' damit die Pipeline weiterlaeuft."""
    parsed = _call_mistral_json(image_path, classification_prompt, MISTRAL_MODEL_CLASSIFY, max_tokens=200)
    if not parsed:
        return {
            "bildtyp": "foto",
            "alt_text": "",
            "ist_dekorativ": False,
            "konfidenz": "niedrig",
            "original_alt_brauchbar": False,
            "_classify_failed": True,
        }
    return {
        "bildtyp": parsed.get("bildtyp", "foto"),
        "alt_text": "",
        "ist_dekorativ": bool(parsed.get("ist_dekorativ", False)),
        "konfidenz": parsed.get("konfidenz", "mittel"),
        "original_alt_brauchbar": bool(parsed.get("original_alt_brauchbar", False)),
    }


def _call_mistral_validate(image_path: str, alt_text: str, langbeschreibung: str = "", context: str = "") -> dict | None:
    """v3.4 Stufe 3: Validierung des generierten Alt-Texts.
    v3.4 (14.04.2026): Bekommt jetzt den gleichen Kontext wie der Generator,
    damit Link-Verweise ("verweist auf: ...") nicht faelschlicherweise als
    Halluzinationen markiert werden.
    v3.7 (24.04.2026): Validator korrigiert jetzt auch die Langbeschreibung
    (vorher nur Alt-Text). max_tokens auf 2000, damit Platz fuer Lang-Korrektur.
    Returns dict mit validierung_ok/probleme/korrektur_alt_text/korrektur_langbeschreibung,
    oder None on failure."""
    if not VALIDATOR_ENABLED:
        return None
    if not alt_text or len(alt_text.strip()) < 5:
        return None
    from context_engine import get_validation_prompt
    prompt = get_validation_prompt(alt_text, langbeschreibung, context=context)
    parsed = _call_mistral_json(image_path, prompt, MISTRAL_MODEL_VALIDATE, max_tokens=2000)
    if not parsed:
        return None
    # v3.7: beide Korrekturfelder; Fallback auf alten Schluessel fuer Rueckwaertskompat.
    korrektur_alt = (parsed.get("korrektur_alt_text")
                     or parsed.get("korrektur_vorschlag")
                     or "").strip()
    korrektur_lang = (parsed.get("korrektur_langbeschreibung") or "").strip()
    return {
        "validierung_ok": bool(parsed.get("validierung_ok", True)),
        "probleme": parsed.get("probleme", []) or [],
        "korrektur_alt_text": korrektur_alt,
        "korrektur_langbeschreibung": korrektur_lang,
    }


def _apply_validator(image_path: str, result: dict, context: str = "") -> dict:
    """v3.4 Stufe 3: Run validator and mutate result dict accordingly.
    - In VALIDATOR_MODE=correct: Validator may overwrite alt_text AND langbeschreibung if he flags AND provides corrections.
    - In VALIDATOR_MODE=flag: Validator only sets needs_review, never overwrites.
    On validator failure: result is returned unchanged, needs_review stays False.
    Always records pipeline_steps + validation_result on the dict.
    v3.4 (14.04.2026): context wird an den Validator durchgereicht.
    v3.7 (24.04.2026): Korrektur erstreckt sich jetzt auf alt_text UND langbeschreibung,
    damit beide Felder konsistent bleiben. Fallback: wenn Validator nur Alt-Text korrigiert,
    wird langbeschreibung geleert (lieber leer als widerspruechlich zum korrigierten Alt).
    Plus: bei konfidenz mittel/niedrig immer needs_review=True (Bug #2)."""
    result.setdefault("pipeline_steps", "classified,generated")
    result.setdefault("needs_review", False)
    result.setdefault("validation_result", "")

    # Bug #2 (24.04.2026): mittel/niedrig-Konfidenz immer zur Review markieren,
    # unabhaengig vom Validator-Ergebnis. Das ist die semantische Bedeutung des Feldes.
    konfidenz = (result.get("konfidenz") or "").strip().lower()
    if konfidenz in ("mittel", "niedrig"):
        result["needs_review"] = True

    alt = result.get("alt_text", "") or ""
    lang = result.get("langbeschreibung", "") or ""

    validation = _call_mistral_validate(image_path, alt, lang, context=context)
    if validation is None:
        result["pipeline_steps"] = "classified,generated,validation_failed"
        return result

    result["pipeline_steps"] = f"classified,generated,validated:{MISTRAL_MODEL_VALIDATE}"
    try:
        result["validation_result"] = json.dumps(validation, ensure_ascii=False)[:1500]
    except Exception:
        result["validation_result"] = ""

    if validation["validierung_ok"]:
        return result

    # Probleme gefunden
    correction_alt = validation.get("korrektur_alt_text", "").strip()
    correction_lang = validation.get("korrektur_langbeschreibung", "").strip()
    if VALIDATOR_MODE == "correct" and correction_alt and len(correction_alt) > 10:
        print(f"v3.7 Validator-Korrektur fuer {image_path}:")
        print(f"  alt: '{alt[:60]}' -> '{correction_alt[:60]}'")
        if correction_lang:
            print(f"  lang: '{lang[:60]}' -> '{correction_lang[:60]}'")
        else:
            print(f"  lang: '{lang[:60]}' -> (geleert, da Validator keine Lang-Korrektur lieferte)")
        result["alt_text"] = correction_alt[:400]
        # v3.7: Langbeschreibung mit korrigieren, leer als sicherer Fallback
        result["langbeschreibung"] = correction_lang[:1500] if correction_lang else ""
        result["needs_review"] = True  # Korrektur wurde angewendet, Mensch sollte trotzdem gegenlesen
    else:
        print(f"v3.7 Validator-Flag fuer {image_path}: {validation.get('probleme', [])}")
        result["needs_review"] = True

    return result


def _call_mistral_generate(image_path: str, bildtyp: str, context: str,
                           width: int = 0, height: int = 0,
                           original_alt: str = "", original_alt_brauchbar: bool = False) -> dict | None:
    """Call Mistral Pixtral API for Insight-First alt-text generation (Stufe 2).
    v2.2: Passes image dimensions and original alt for thumbnail/improvement modes.
    """
    from context_engine import get_generation_prompt

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return None

    try:
        img_b64 = _resize_image_for_model(image_path)
        prompt = get_generation_prompt(
            bildtyp=bildtyp, context_text=context,
            width=width, height=height,
            original_alt=original_alt,
            original_alt_brauchbar=original_alt_brauchbar
        )

        response = httpx.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MISTRAL_MODEL_GENERATE,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }],
                "max_tokens": 2500,
                "temperature": 0,
            },
            timeout=60.0
        )
        response.raise_for_status()

        # Parse Mistral API response – handle control chars
        try:
            resp_data = response.json()
        except Exception:
            raw = response.text
            raw = raw.replace('\r\n', '\n').replace('\r', '\n')
            raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
            resp_data = json.loads(raw)

        choice = resp_data["choices"][0]
        text = choice["message"]["content"]

        # v3.7 Debug-Logging fuer Truncation-Audit (negative Bewertungen 22.-26.04.2026):
        # finish_reason='length' = Token-Cap-Treffer; 'stop' = Modell entschied selbst.
        # Zusammen mit completion_tokens und repr(text) klar diagnostizierbar.
        # print() statt logger.info() weil Custom-Logger nicht konfiguriert ist
        # und uvicorn die Default-Level auf WARN haelt.
        if os.getenv("DEBUG_GEN_RAW", "false").lower() == "true":
            _finish_reason = choice.get("finish_reason", "?")
            _usage = resp_data.get("usage", {}) or {}
            print(
                f"[GEN-RAW] {image_path}: finish_reason={_finish_reason} "
                f"completion_tokens={_usage.get('completion_tokens', '?')} "
                f"text_len={len(text)} content={text!r}",
                flush=True,
            )

        # Parse Mistral response (expects {"alt_text": "...", "langbeschreibung": "..."})
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        clean = re.sub(r'```json\s*', '', clean)
        clean = re.sub(r'\s*```', '', clean)
        # Remove control characters that break json.loads
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)

        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(clean[start:end])
            except json.JSONDecodeError:
                try:
                    parsed = json.loads(clean[start:end].replace('\n', ' ').replace('\t', ' '))
                except json.JSONDecodeError:
                    # Bei abgeschnittenem/kaputtem JSON: Text bereinigen statt rohen JSON ausgeben
                    fallback_text = clean[start:end]
                    fallback_text = re.sub(r'[{}\[\]"\\]', '', fallback_text)
                    fallback_text = re.sub(r'(alt_text|langbeschreibung)\s*:', '', fallback_text)
                    fallback_text = re.sub(r',\s*$', '', fallback_text).strip()
                    fallback_text = re.sub(r'\s+', ' ', fallback_text)
                    if len(fallback_text) > 10:
                        print(f"WARNUNG: JSON-Parse fehlgeschlagen, verwende bereinigten Fallback-Text")
                        return {
                            "alt_text": _combine_alt_text(fallback_text[:350], ""),
                            "langbeschreibung": "",
                            "ist_dekorativ": False,
                        }
                    parsed = {}
            alt_text = parsed.get("alt_text", "").strip()
            if alt_text and len(alt_text) > 5:
                return {
                    "alt_text": _combine_alt_text(alt_text, ""),
                    "langbeschreibung": parsed.get("langbeschreibung", ""),
                    "ist_dekorativ": False,
                }

        # If JSON parsing fails, use the raw text – clean up JSON artifacts
        if len(clean) > 10:
            cleaned_fallback = re.sub(r'[{}\[\]"\\]', '', clean)
            cleaned_fallback = re.sub(r'(alt_text|langbeschreibung)\s*:', '', cleaned_fallback)
            cleaned_fallback = re.sub(r'\s+', ' ', cleaned_fallback).strip()
            return {
                "alt_text": _combine_alt_text(cleaned_fallback[:350], ""),
                "langbeschreibung": "",
                "ist_dekorativ": False,
            }

        return None
    except Exception as e:
        print(f"Mistral API Fehler fuer {image_path}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _should_escalate_to_mistral(result: dict) -> bool:
    """Decide if we should escalate to Mistral API based on Qwen result quality."""
    MISTRAL_ENABLED = os.environ.get("MISTRAL_ENABLED", "false").lower() in ("true", "1")
    if not MISTRAL_ENABLED:
        return False

    konfidenz = result.get("konfidenz", "mittel")
    bildtyp = result.get("bildtyp", "")
    alt_text = result.get("alt_text", "")

    # Escalate if quality indicators suggest Qwen struggled
    if konfidenz == "niedrig":
        return True
    if konfidenz == "mittel" and bildtyp in ("strukturformel", "karte", "infografik", "diagramm"):
        return True
    if bildtyp in ("strukturformel", "diagramm"):
        return True
    if "nicht lesbar" in alt_text or "nicht erkennbar" in alt_text:
        return True
    if len(alt_text) < 20 and bildtyp not in ("logo", "dekorativ"):
        return True

    return False


def _call_mistral(image_path: str, context: str, image_type: str = None, qwen_result: dict = None) -> dict | None:
    """Call Mistral Pixtral API as intelligent fallback.
    Uses Qwen's pre-analysis as context for better results.
    Returns result dict or None if Mistral is not configured/fails."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return None

    try:
        img_b64 = _resize_image_for_model(image_path)

        # Build enriched context with Qwen's pre-analysis
        mistral_context = context
        if qwen_result:
            qwen_info = (
                f"[Voranalyse lokales Modell] Bildtyp: {qwen_result.get('bildtyp', 'unbekannt')}, "
                f"Vorlaeufiger Alt-Text: {qwen_result.get('alt_text', '')[:200]}. "
                f"WICHTIG: Pruefe das Bild vollstaendig selbst. Wenn im vorlaeufigen Alt-Text 'nicht lesbar' oder 'nicht erkennbar' steht, "
                f"versuche es trotzdem selbst zu lesen. Lies alle Texte, Zahlen und Beschriftungen direkt aus dem Bild."
            )
            mistral_context = f"{qwen_info}\n{context}"

        prompt = get_prompt(image_type=image_type or qwen_result.get("bildtyp"), context_text=mistral_context)

        response = httpx.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "pixtral-large-latest",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }],
                "max_tokens": 2500,
                "temperature": 0,
            },
            timeout=60.0
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return _parse_response(text)
    except Exception as e:
        print(f"Mistral API Fehler: {e}")
        return None


def _apply_postfilter(result: dict, image_hash: str = "") -> dict:
    """v2.2.1: Apply postfilter to clean context interpretations from any alt-text.
    Runs AFTER all generation paths (Mistral, Qwen, Dekorativ-Recheck).
    v2.2.3: Also writes to project cache if image_hash is provided."""
    if not result or result.get("ist_dekorativ"):
        # Cache dekorative Ergebnisse auch
        if image_hash and result:
            _project_image_cache[image_hash] = result.copy()
        return result
    alt = result.get("alt_text", "")
    if alt:
        result["alt_text"] = clean_alt_text(alt)
        # v2.2.3: Hedge-Woerter auch innerhalb des Satzes entfernen
        result["alt_text"] = remove_hedge_words(result["alt_text"])
    # v2.2.3: Gleicher Filter auch fuer Langbeschreibung
    lang = result.get("langbeschreibung", "")
    if lang:
        result["langbeschreibung"] = clean_alt_text(lang)
        result["langbeschreibung"] = remove_hedge_words(result["langbeschreibung"])
        # v2.2.4: Truncation-Fix – unvollstaendige Saetze am Ende abschneiden
        cleaned_lang = result["langbeschreibung"].strip()
        if cleaned_lang and cleaned_lang[-1] not in '.!?)"”':
            # Find last complete sentence
            last_period = cleaned_lang.rfind('.')
            last_excl = cleaned_lang.rfind('!')
            last_paren = cleaned_lang.rfind(')')
            cut_pos = max(last_period, last_excl, last_paren)
            if cut_pos > len(cleaned_lang) * 0.5:
                result["langbeschreibung"] = cleaned_lang[:cut_pos + 1]
    # v2.2.3: Cache result for duplicate detection
    if image_hash:
        _project_image_cache[image_hash] = result.copy()
    return result


def generate_alt_text_for_image(image_path: str, context_text: str = "", image_type: str = None,
                                width: int = 0, height: int = 0, original_alt: str = "",
                                language: str = "de") -> dict:
    """Generate alt-text for a standalone image. Uses the same dual-model pipeline.
    v2.2: Passes through width/height/original_alt for thumbnail/improvement modes.
    """
    return generate_alt_text(image_path, context=context_text, image_type=image_type,
                             width=width, height=height, original_alt=original_alt,
                             language=language)
