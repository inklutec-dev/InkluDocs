import fitz  # PyMuPDF
import os
import json
import httpx
import base64
import time
import re
from PIL import Image
from io import BytesIO

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")

ALT_TEXT_PROMPT = """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
ZIEL: Blinde Nutzer erhalten die GLEICHE INFORMATION wie Sehende.

Antworte NUR mit diesem JSON:
{{"bildtyp": "foto|diagramm|tabelle|screenshot|icon|logo|karte|dekorativ", "alt_text": "...", "ist_dekorativ": true/false, "konfidenz": "hoch|mittel|niedrig"}}

FORMAT: Beginne mit dem Bildtyp als kurzes Praefix, dann Bindestrich, dann die Kernaussage.
So weiss der Screenreader-Nutzer sofort, WAS fuer ein Bild kommt, und dann den Inhalt.

LAENGE: 2-4 Saetze (150-350 Zeichen). Kernaussage + wichtigste Zahlen. Nicht jedes Detail.

BEISPIELE PERFEKTER ALT-TEXTE:

"Logo Nationaler Normenkontrollrat"

"Kreisdiagramm – Die groesste Buerokratie-Entlastung bringt das Wachstumschancengesetz (BMF) mit 39%, gefolgt von der Schwellenwert-Anhebung (BMJ) mit 18%. Groesster Kostentreiber ist die EU-CSRD-Richtlinie (BMJ) mit 39%, gefolgt vom Waermeplanungsgesetz mit 20%."

"Balkendiagramm – Die 'One in one out'-Bilanz ergibt eine Nettoentlastung von 1,5 Milliarden Euro. Der Umstellungsaufwand stieg von unter 5 Milliarden (2011-2019) auf rund 23 Milliarden Euro in 2022/23, getragen vor allem von der Wirtschaft."

"Balkendiagramm – Aufmerksamkeit und Energie empfinden 90,2% der Befragten als sehr hohe Buerokratiebelastung, den Zeitaufwand 83,3% und den Kostenaufwand 66,4% (IfM Bonn, 2023)."

"Foto – Drei Personen am Rednerpult bei einer Pressekonferenz des Normenkontrollrats."

"QR-Code zur NKR-Stellungnahme 'Vereinfachung von Sozialleistungen'."

Dekorativ (abstrakte Formen, Hintergruende, kleine Icons): ist_dekorativ=true, alt_text=""

REGELN:
- Deutsch, professionell, wie ein Nachrichtensprecher
- WISSEN vermitteln, nicht Aussehen beschreiben
- Bei Zeitreihen: IMMER den Trend benennen (gestiegen/gefallen/stabil/schwankend) und Anfangs- und Endwert nennen
- Bei Vergleichen: IMMER benennen wer fuehrt und wer abgeschlagen ist
- Keine Farben (ausser informationstragend)
- Erfinde NICHTS. Wenn unleserlich: "teilweise nicht lesbar"
- konfidenz: hoch = klar lesbar, mittel = manches unsicher, niedrig = vieles unklar
- SOFORT JSON ausgeben

Kontext: {context}"""


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
    """Combine short alt-text with long description, respecting max length."""
    if not alt_text:
        return ""
    if not langbeschreibung:
        text = alt_text.strip()
    elif langbeschreibung.strip().startswith(alt_text.strip()[:30]):
        text = langbeschreibung.strip()
    else:
        text = alt_text.rstrip(". ") + ". " + langbeschreibung.strip()
    # Trim to max length at sentence boundary
    if len(text) > MAX_ALT_TEXT_LENGTH:
        cut = text[:MAX_ALT_TEXT_LENGTH]
        last_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if last_end > 80:
            text = cut[:last_end + 1]
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


def generate_alt_text(image_path: str, context: str = "") -> dict:
    """Generate alt-text for a single image using Qwen3-VL via Ollama."""
    img_b64 = _resize_image_for_model(image_path)

    prompt = ALT_TEXT_PROMPT.format(context=context[:500] if context else "Kein Kontext.")

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
            # Patterns ordered from most specific (JSON) to least specific (natural language)
            alt_patterns = [
                # JSON format: "alt_text": "..."
                r'"alt_text":\s*"([^"]+)"',
                # Key-value: alt_text: "..." or Alt-Text: "..."
                r'[Aa]lt[_-]?[Tt]ext:\s*"([^"]+)"',
                # Natural language: alt_text should/would/could be "..."
                r'[Aa]lt[_-]?[Tt]ext\s+(?:should|would|could|is|shall)\s+be\s*"([^"]+)"',
                # Natural language: the alt_text is "..." or alt text: "..."
                r'the\s+alt[_-]?\s*text\s+(?:is|should be|would be)\s*"([^"]+)"',
                # German: Alt-Text waere/ist/lautet "..."
                r'[Aa]lt[_-]?[Tt]ext\s+(?:waere|ist|lautet|sollte sein)\s*"([^"]+)"',
                # Last quoted string after "alt" mention (catches most remaining cases)
                r'[Aa]lt[_-]?[Tt]ext[^"]*"([^"]{15,})"',
                # Key-value without quotes: alt_text: some text here
                r'[Aa]lt[_-]?[Tt]ext:\s*(.+?)(?:\n|$)',
            ]
            found_alt = None
            for pat in alt_patterns:
                m = re.search(pat, thinking_text)
                if m and len(m.group(1).strip()) > 10:
                    found_alt = m.group(1).strip().strip('"').strip('.')
                    # Skip if it looks like code/reasoning, not actual alt-text
                    if any(kw in found_alt.lower() for kw in ['should be', 'would be', 'the user', 'according to', 'the rules say']):
                        found_alt = None
                        continue
                    break

            bildtyp = "unbekannt"
            typ_match = re.search(r'"bildtyp":\s*"([^"]+)"', thinking_text)
            if not typ_match:
                typ_match = re.search(r'[Bb]ildtyp[:\s]+["\']?(\w+)', thinking_text)
            if not typ_match:
                # Detect type from natural language in thinking
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


def _escape_pdf_string(text: str) -> str:
    """Escape special characters for PDF string literals."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_string(text: str) -> str:
    """Encode text as PDF string, using UTF-16BE hex string for non-ASCII characters.
    This fixes the umlaut encoding issue (e.g. ü showing as Ã¼)."""
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
    # Use \s+ instead of \n to work with both clean_contents() and raw content streams
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
    # Pattern: q followed by drawing commands (m=moveto, l=lineto, c=curveto, re=rectangle)
    q_blocks = list(re.finditer(r'q\s+(.*?)\s*Q', content_str, re.DOTALL))

    blocks_in_range = []
    for block in q_blocks:
        block_content = block.group(1)
        # Extract coordinates from path operations: "x y m", "x y l", "x y w h re"
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
        # Find the Q closing the last block
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
    # NOTE: Do NOT call page.clean_contents() here - it destroys newlines
    # in the content stream, which breaks vector graphic matching later.
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
                warnings.append(f"Vektorgrafik (Seite unbekannt): Keine Metadaten vorhanden, Alt-Text konnte nicht exportiert werden.")
                continue

            page_num = meta.get("page_number", 1) - 1
            bbox = meta.get("bbox")

            if not bbox:
                warnings.append(f"Vektorgrafik auf Seite {page_num + 1}: Keine Positionsdaten vorhanden, Alt-Text konnte nicht exportiert werden.")
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

        # Find the Document element (root K child)
        k_info = doc.xref_get_key(struct_root_xref, "K")
        doc_elem_xref = None
        if k_info[0] == "xref":
            doc_elem_xref = int(k_info[1].split()[0])
        elif k_info[0] == "array":
            first_ref = re.search(r'(\d+)\s+0\s+R', k_info[1])
            if first_ref:
                doc_elem_xref = int(first_ref.group(1))

        if not doc_elem_xref:
            warnings.append("Bestehende PDF-Struktur konnte nicht gelesen werden, erstelle neue Struktur.")
            has_existing_structure = False

    if has_existing_structure:
        # Find existing ParentTree
        pt_info = doc.xref_get_key(struct_root_xref, "ParentTree")
        parent_tree_xref = int(pt_info[1].split()[0]) if pt_info[0] == "xref" else None

        # Get ParentTreeNextKey
        ptk_info = doc.xref_get_key(struct_root_xref, "ParentTreeNextKey")
        parent_tree_next_key = int(ptk_info[1]) if ptk_info[0] == "int" else len(doc)

        # For each page, find max existing MCID
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

        # Create Figure elements and assign MCIDs after existing ones
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

                # For vector graphics: find the content stream range
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
                            warnings.append(f"Vektorgrafik auf Seite {page_num + 1}: Zeichenbefehle im PDF nicht gefunden, Alt-Text wurde als Struktur-Tag gesetzt aber moeglicherweise nicht korrekt verknuepft.")

                page_figures[page_num].append((mcid, fig_xref, img_info, content_range))

        # Update Document element: append Figure xrefs to K array
        doc_elem_obj = doc.xref_object(doc_elem_xref)
        k_match = re.search(r'/K\s*\[(.*?)\]', doc_elem_obj, re.DOTALL)
        if k_match:
            existing_kids = k_match.group(1).strip()
            new_kids_str = " ".join(f"{x} 0 R" for x in figure_xrefs)
            updated_kids = f"{existing_kids} {new_kids_str}"
            updated_obj = doc_elem_obj[:k_match.start(1)] + updated_kids + doc_elem_obj[k_match.end(1):]
            doc.update_object(doc_elem_xref, updated_obj)

        # Update ParentTree: append Figure refs to each page's entry
        if parent_tree_xref:
            pt_obj = doc.xref_object(parent_tree_xref)
            for page_num, figs in sorted(page_figures.items()):
                page = doc[page_num]
                sp_info = doc.xref_get_key(page.xref, "StructParents")
                if sp_info[0] != "int":
                    continue
                sp_val = int(sp_info[1])

                # Find the existing array for this StructParents value in /Nums
                # Pattern: sp_val [ refs... ]
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
                            warnings.append(f"Vektorgrafik auf Seite {page_num + 1}: Zeichenbefehle im PDF nicht gefunden, Alt-Text wurde als Struktur-Tag gesetzt aber moeglicherweise nicht korrekt verknuepft.")

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

        # Only set StructParents for new-structure PDFs (existing ones already have it)
        if not has_existing_structure:
            doc.xref_set_key(page.xref, "StructParents", str(page_num))

        content = page.read_contents()
        if not content:
            continue
        content_str = content.decode('latin-1')

        # Collect all modifications
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

        # Apply modifications in reverse order to preserve positions
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

    # Count successfully tagged images
    tagged_count = 0
    for page_num, figs in page_figures.items():
        for mcid, fig_xref, img_info, content_range in figs:
            if not img_info["is_vector"] or content_range is not None:
                tagged_count += 1

    doc.save(output_path)
    doc.close()
    return {"path": output_path, "tagged_count": tagged_count, "warnings": warnings}
