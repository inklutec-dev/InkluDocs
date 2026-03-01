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

                context = page_text[:500] if page_text else "Kein Textkontext verfuegbar."

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
                })

                # Record where this raster image is on the page
                for img_rect in page.get_image_rects(xref):
                    raster_areas.append(img_rect)

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
                    context = page_text[:500] if page_text else "Kein Textkontext verfuegbar."

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


def write_alt_texts_to_pdf(input_path: str, output_path: str, alt_texts: dict) -> str:
    """Write alt-texts into a proper Tagged PDF structure for screen reader accessibility."""
    doc = fitz.open(input_path)
    cat_xref = doc.pdf_catalog()

    # Collect images with alt-texts per page
    page_images = {}  # page_num -> [(xref, img_name, alt_text)]
    for page_num in range(len(doc)):
        page = doc[page_num]
        page.clean_contents()
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            img_name = img_info[7]
            if xref in alt_texts and alt_texts[xref] is not None:
                alt_text = alt_texts[xref]
                if alt_text == "dekorativ":
                    alt_text = ""
                if page_num not in page_images:
                    page_images[page_num] = []
                page_images[page_num].append((xref, img_name, alt_text))

                # Also set Alt on XObject as fallback for some readers
                try:
                    escaped = _escape_pdf_string(alt_text)
                    doc.xref_set_key(xref, "Alt", f"({escaped})" if escaped else "()")
                except Exception:
                    pass

    if not page_images:
        doc.save(output_path)
        doc.close()
        return output_path

    # --- Build Tagged PDF Structure ---
    struct_root_xref = doc.get_new_xref()
    parent_tree_xref = doc.get_new_xref()
    doc_elem_xref = doc.get_new_xref()

    figure_xrefs = []
    page_figures = {}  # page_num -> [(mcid, fig_xref, img_name)]

    for page_num in sorted(page_images.keys()):
        page = doc[page_num]
        page_figures[page_num] = []

        for xref, img_name, alt_text in page_images[page_num]:
            fig_xref = doc.get_new_xref()
            mcid = len(page_figures[page_num])
            escaped = _escape_pdf_string(alt_text)

            doc.update_object(fig_xref,
                f"<< /Type /StructElem /S /Figure /P {doc_elem_xref} 0 R "
                f"/Pg {page.xref} 0 R /Alt ({escaped}) "
                f"/K << /Type /MCR /MCID {mcid} /Pg {page.xref} 0 R >> >>")

            figure_xrefs.append(fig_xref)
            page_figures[page_num].append((mcid, fig_xref, img_name))

    # Document structure element
    kids_str = " ".join(f"{x} 0 R" for x in figure_xrefs)
    doc.update_object(doc_elem_xref,
        f"<< /Type /StructElem /S /Document /P {struct_root_xref} 0 R "
        f"/K [{kids_str}] >>")

    # ParentTree
    nums_parts = []
    for page_num, figs in sorted(page_figures.items()):
        refs = " ".join(f"{f[1]} 0 R" for f in figs)
        nums_parts.append(f"{page_num} [{refs}]")

    doc.update_object(parent_tree_xref,
        f"<< /Nums [{' '.join(nums_parts)}] >>")

    # StructTreeRoot
    doc.update_object(struct_root_xref,
        f"<< /Type /StructTreeRoot /K {doc_elem_xref} 0 R "
        f"/ParentTree {parent_tree_xref} 0 R >>")

    # Set catalog entries
    doc.xref_set_key(cat_xref, "StructTreeRoot", f"{struct_root_xref} 0 R")
    doc.xref_set_key(cat_xref, "MarkInfo", "<< /Marked true >>")

    # Mark content streams - wrap image operations with BMC/EMC
    for page_num, figs in page_figures.items():
        page = doc[page_num]
        doc.xref_set_key(page.xref, "StructParents", str(page_num))

        content = page.read_contents()
        if not content:
            continue
        content_str = content.decode('latin-1')

        for mcid, fig_xref, img_name in figs:
            escaped_name = re.escape(img_name)
            pattern = rf'(q\s[^Q]*?/{escaped_name}\s+Do\s*Q)'
            match = re.search(pattern, content_str)
            if match:
                original = match.group(1)
                wrapped = f"/Figure <</MCID {mcid}>> BDC\n{original}\nEMC"
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

    doc.save(output_path)
    doc.close()
    return output_path
