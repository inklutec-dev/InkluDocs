import fitz  # PyMuPDF
import os
import json
import httpx
import base64
import time
import re
from PIL import Image
from io import BytesIO

from context_engine import get_prompt, detect_type_from_context

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")


def _cluster_drawings(drawings, page_rect, gap=50, min_size=50):
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
        if total_items < 5:
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


def _get_nearby_text(page, bbox, max_chars=600):
    """Extract text near an image bounding box - text BEFORE and AFTER the image.
    Prioritizes captions and headings over generic paragraphs.
    This gives the model context like captions, headings, and surrounding paragraphs.
    Inspired by Michael Karbe's suggestion (2026-03-10)."""
    if not bbox:
        page_text = page.get_text()
        return page_text[:max_chars] if page_text else "Kein Textkontext verfuegbar."

    img_rect = fitz.Rect(bbox)
    # Get all text blocks with positions: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")
    if not blocks:
        return "Kein Textkontext verfuegbar."

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
            # Block is ABOVE image
            distance = img_rect.y0 - block_rect.y1
            position = "before"
        elif block_rect.y0 >= img_rect.y1:
            # Block is BELOW image
            distance = block_rect.y0 - img_rect.y1
            position = "after"
        else:
            # Block overlaps with image vertically
            distance = 0
            position = "overlap"

        # Detect captions - they get priority (distance bonus)
        is_cap = _is_caption(block_text)
        # Captions get a sort bonus: treated as distance 0 (closest)
        sort_distance = 0 if is_cap else distance

        text_blocks.append((sort_distance, distance, position, block_text, is_cap))

    # Sort by sort_distance (captions first), then by actual distance
    text_blocks.sort(key=lambda x: (x[0], x[1]))

    # Build context: captions and closest blocks first
    context_parts = []
    chars_used = 0
    for sort_dist, distance, position, text, is_cap in text_blocks:
        if chars_used >= max_chars:
            break
        remaining = max_chars - chars_used
        snippet = text[:remaining]
        if is_cap:
            context_parts.append(f"[Bildunterschrift] {snippet}")
        elif position == "before":
            context_parts.append(f"[Text davor] {snippet}")
        elif position == "after":
            context_parts.append(f"[Text danach] {snippet}")
        else:
            context_parts.append(f"[Ueberlappend] {snippet}")
        chars_used += len(snippet) + 20  # account for label

    return "\n".join(context_parts) if context_parts else "Kein Textkontext verfuegbar."


def extract_images_from_pdf(pdf_path: str, output_dir: str, project_id: int) -> list:
    """Extract all images from a PDF, including vector graphics rendered as images."""
    doc = fitz.open(pdf_path)
    images = []
    vector_xref_counter = 900000  # High xref range for vector graphics

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        image_list = page.get_images(full=True)

        # Track bounding boxes of raster images to avoid duplicating
        raster_areas = []
        img_idx = 0

        # 1. Extract raster images (as before)
        for img_info in image_list:
            xref = img_info[0]
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
                    # Scale factor: 2x for normal graphics, lower for very large ones
                    cw, ch = cluster_rect.width, cluster_rect.height
                    scale = 2.0
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
                    })
                    print(f"Vector graphic on page {page_num + 1}: {int(cluster_rect.width)}x{int(cluster_rect.height)}px")
                except Exception as e:
                    print(f"Error rendering vector graphic on page {page_num + 1}: {e}")
                    continue

    doc.close()
    return images


MAX_IMAGE_DIM = 1024
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


def _resize_image_for_model(image_path: str) -> str:
    """Resize image if too large for the model, return base64 encoded string."""
    img = Image.open(image_path)
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


def _call_ollama(image_path: str, prompt: str) -> dict:
    """Send an image + prompt to Ollama and return the parsed result dict."""
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
                    "temperature": 0.3,
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


def generate_alt_text(image_path: str, context: str = "", image_type: str = None) -> dict:
    """Generate alt-text for a single image using Qwen3-VL via Ollama.

    Args:
        image_path: Path to the image file.
        context: Surrounding text context from the document.
        image_type: Optional specific image type for specialized prompt.
                    If None, uses the general prompt (first pass).
    """
    prompt = get_prompt(image_type=image_type, context_text=context)
    return _call_ollama(image_path, prompt)


def generate_alt_text_for_image(image_path: str, context_text: str = "", image_type: str = None) -> dict:
    """Generate alt-text for a standalone image (not from PDF).

    Works with uploaded images or images downloaded from URLs.
    Uses context_engine for prompt selection.

    Args:
        image_path: Path to the image file on disk.
        context_text: Optional context text (e.g. from the web page).
        image_type: Optional specific image type for specialized prompt.

    Returns:
        Dict with bildtyp, alt_text, langbeschreibung, ist_dekorativ, konfidenz.
    """
    # If no explicit type given, try to detect from context
    effective_type = image_type
    if not effective_type and context_text:
        effective_type = detect_type_from_context(context_text)

    prompt = get_prompt(image_type=effective_type, context_text=context_text)
    return _call_ollama(image_path, prompt)
