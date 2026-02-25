import fitz  # PyMuPDF
import os
import json
import httpx
import base64
import time
from PIL import Image
from io import BytesIO

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")

ALT_TEXT_PROMPT = """Analysiere dieses Bild aus einem PDF-Dokument.

Kontext aus dem umgebenden Text: {context}

Aufgaben:
1. BILDTYP bestimmen: Ist es ein Foto, Diagramm/Chart, Tabelle, Screenshot, Icon, Logo oder ein dekoratives Element?
2. ALT-TEXT generieren: Schreibe einen informativen, barrierefreien Alt-Text nach WCAG 2.2 Richtlinien.

Regeln fuer den Alt-Text:
- Bei Diagrammen/Charts: Nenne die dargestellten Daten, Trends und Werte
- Bei Fotos: Beschreibe was zu sehen ist und warum es relevant ist
- Bei Tabellen: Fasse die wichtigsten Daten zusammen
- Bei dekorativen Elementen: Antworte mit "dekorativ"
- Bei Logos: Nenne den Firmennamen
- Maximal 2-3 Saetze
- Deutsch

Antworte NUR in diesem JSON-Format:
{{"bildtyp": "...", "alt_text": "...", "ist_dekorativ": true/false}}"""


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

                # Skip very small images (likely decorative/spacers)
                if width < 20 or height < 20:
                    continue

                # Save image
                img_filename = f"p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                img_path = os.path.join(output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # Get surrounding text as context (first 500 chars)
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


def generate_alt_text(image_path: str, context: str = "") -> dict:
    """Generate alt-text for a single image using Qwen3-VL via Ollama."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    prompt = ALT_TEXT_PROMPT.format(context=context[:500] if context else "Kein Kontext.")

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=600.0,
        )
        response.raise_for_status()
        result = response.json()
        text = result.get("response", "")

        # Try to parse JSON from response
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return {
                    "bildtyp": parsed.get("bildtyp", "unbekannt"),
                    "alt_text": parsed.get("alt_text", text),
                    "ist_dekorativ": parsed.get("ist_dekorativ", False),
                    "raw_response": text,
                }
        except json.JSONDecodeError:
            pass

        # Fallback: use raw text
        return {
            "bildtyp": "unbekannt",
            "alt_text": text.strip(),
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


def write_alt_texts_to_pdf(input_path: str, output_path: str, alt_texts: dict) -> str:
    """Write alt-texts back into the PDF structure.

    alt_texts: dict mapping xref -> alt_text string
    """
    doc = fitz.open(input_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            if xref in alt_texts and alt_texts[xref]:
                alt_text = alt_texts[xref]
                if alt_text == "dekorativ":
                    # Mark as decorative (empty alt-text in PDF/UA)
                    alt_text = ""
                # Set the alt-text via the image's struct element
                # PyMuPDF approach: modify the /Alt entry
                try:
                    # Get the xref object and add /Alt text
                    xref_str = doc.xref_get_key(xref, "")
                    doc.xref_set_key(xref, "Alt", fitz.get_text_length(alt_text) and f"({fitz.utils.escape_pdftext(alt_text)})" or "()")
                except Exception:
                    pass

    doc.save(output_path)
    doc.close()
    return output_path
