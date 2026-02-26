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

ALT_TEXT_PROMPT = """Du bist ein Experte fuer barrierefreie Alt-Texte nach WCAG 2.2.
Dein Ziel: Blinde Menschen sollen durch deinen Alt-Text die GLEICHE Information erhalten wie sehende Menschen.

Analysiere dieses Bild und antworte NUR mit diesem JSON:
{{"bildtyp": "foto|diagramm|tabelle|screenshot|icon|logo|dekorativ", "alt_text": "...", "ist_dekorativ": true/false}}

Regeln je nach Bildtyp:
- Diagramm/Chart: Nenne den Titel, die dargestellten Werte, den Trend und die Kernaussage. Beispiel: "Balkendiagramm zeigt Umsatz 2020-2025. Anstieg von 1,2 auf 3,4 Mio EUR, staerkster Zuwachs 2024."
- Tabelle: Fasse die wichtigsten Zeilen und Spalten zusammen, nenne Extremwerte.
- Foto: Beschreibe was zu sehen ist und warum es im Kontext relevant ist.
- Logo: Nenne den Firmennamen und ggf. Slogan.
- Screenshot/Banner: Lies jeden sichtbaren Text vor und beschreibe das Layout.
- Dekorativ (Linien, Hintergruende, Schmuckelemente): ist_dekorativ=true, alt_text=""

Allgemein:
- Maximal 3 Saetze, auf Deutsch
- Vermittle die INFORMATION, nicht nur das Aussehen
- Lies jeden sichtbaren Text im Bild vor
- Keine Erklaerung, keine Begruendung, NUR das JSON

Kontext aus dem Dokument: {context}"""


def extract_images_from_pdf(pdf_path: str, output_dir: str, project_id: int) -> list:
    """Extract all images from a PDF with their context text."""
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

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

                img_filename = f"p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                img_path = os.path.join(output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                context = page_text[:500] if page_text else "Kein Textkontext verfuegbar."

                images.append({
                    "page_number": page_num + 1,
                    "image_index": img_idx + 1,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "width": width,
                    "height": height,
                    "xref": xref,
                    "context_text": context,
                    "ext": image_ext,
                })
            except Exception as e:
                print(f"Error extracting image {xref} from page {page_num + 1}: {e}")
                continue

    doc.close()
    return images


MAX_IMAGE_WIDTH = 800
MAX_IMAGE_HEIGHT = 800


def _resize_image_for_model(image_path: str) -> str:
    """Resize image if too large, return base64 encoded string."""
    img = Image.open(image_path)
    if img.width > MAX_IMAGE_WIDTH or img.height > MAX_IMAGE_HEIGHT:
        img.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.LANCZOS)
        buf = BytesIO()
        fmt = "JPEG" if image_path.lower().endswith((".jpg", ".jpeg")) else "PNG"
        img.save(buf, format=fmt, quality=85)
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
                    "num_predict": 1000,
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
                r'[Aa]lt[_-]?[Tt]ext:\s*(.+?)(?:\n|$)',
            ]
            found_alt = None
            for pat in alt_patterns:
                m = re.search(pat, thinking_text)
                if m and len(m.group(1)) > 10:
                    found_alt = m.group(1).strip().strip('"')
                    break

            bildtyp = "unbekannt"
            typ_match = re.search(r'"bildtyp":\s*"([^"]+)"', thinking_text)
            if not typ_match:
                typ_match = re.search(r'[Bb]ildtyp[:\s]+["\']?(\w+)', thinking_text)
            if typ_match:
                bildtyp = typ_match.group(1).strip()

            if found_alt:
                return {
                    "bildtyp": bildtyp,
                    "alt_text": found_alt,
                    "ist_dekorativ": "dekorativ" in found_alt.lower(),
                    "raw_response": thinking_text,
                }

        # Try to parse JSON from cleaned response
        try:
            json_matches = list(re.finditer(r'\{[^{}]*"alt_text"[^{}]*\}', clean_text))
            if json_matches:
                parsed = json.loads(json_matches[-1].group())
                return {
                    "bildtyp": parsed.get("bildtyp", "unbekannt"),
                    "alt_text": parsed.get("alt_text", ""),
                    "ist_dekorativ": parsed.get("ist_dekorativ", False),
                    "raw_response": text,
                }
            start = clean_text.find("{")
            end = clean_text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(clean_text[start:end])
                if parsed.get("alt_text"):
                    return {
                        "bildtyp": parsed.get("bildtyp", "unbekannt"),
                        "alt_text": parsed.get("alt_text", ""),
                        "ist_dekorativ": parsed.get("ist_dekorativ", False),
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
