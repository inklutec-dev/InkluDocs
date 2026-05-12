"""v3.7 Pipeline (klassische dreistufige Mistral-Pipeline).

Bewahrungs-Modul fuer die v3.7-Pipeline-Logik (Klassifikation -> Generierung -> Validator).
Wird via PIPELINE_VERSION=v3_7 (Default) gerufen, kann parallel zu v4 laufen.

T4 (03.05.2026, Phase A): Body wurde 1:1 aus pdf_processor.generate_alt_text() umgezogen.
KEINE Verhaltensaenderung — nur Modul-Standort hat sich geaendert. Wird in T5 ueber den
Front-Door-Router in pdf_processor.generate_alt_text() angesprochen.
"""
import json

from pdf_processor import (
    _get_image_hash,
    _project_image_cache,
    _ocr_extract_text,
    _call_mistral_classify,
    _call_mistral_json,
    _call_mistral_generate,
    _apply_validator,
    _apply_postfilter,
    DEKORATIV_RECHECK_PROMPT,
    MISTRAL_MODEL_CLASSIFY,
)
from context_engine import (
    get_classification_prompt,
    should_use_mistral,
    PIPELINE_MODE,
    validate_dekorativ,
    is_thumbnail,
)


def generate_alt_text_v3_7(image_path: str, context: str = "", image_type: str = None,
                           width: int = 0, height: int = 0, original_alt: str = "") -> dict:
    """Generate alt-text using the dual-model pipeline.

    Pipeline v3.2:
      1. OCR extracts text from image
      2. Qwen classifies the image (Stufe 1)
      3. v2.2: Validate decorative classification (safety net)
      4. v2.2: Functional elements -> keep original alt-text if good
      5. v2.2: Brauchbar original alt -> improvement mode
      6. Based on pipeline mode: Mistral or Qwen generates alt-text (Stufe 2)
      7. Fallback to Qwen if Mistral fails
    """
    # v2.2.3: Check project cache for duplicate images
    img_hash = _get_image_hash(image_path)
    if img_hash in _project_image_cache:
        cached = _project_image_cache[img_hash].copy()
        print(f"v2.2.3 Cache-Hit: {image_path} (Hash {img_hash[:12]}...)")
        return cached

    # OCR: Extract text from the image
    ocr_text = _ocr_extract_text(image_path)
    enriched_context = context
    if ocr_text:
        enriched_context = f"[OCR-Text im Bild] {ocr_text}\n{context}"

    # ─── Stufe 1: Mistral classifies (v3.3, 14.04.2026: ersetzt Qwen/Ollama) ───
    classification_prompt = get_classification_prompt(
        enriched_context, width=width, height=height, original_alt=original_alt
    )
    classification = _call_mistral_classify(image_path, classification_prompt)

    bildtyp = classification.get("bildtyp", image_type or "foto")
    konfidenz = classification.get("konfidenz", "mittel")
    ist_dekorativ = classification.get("ist_dekorativ", False)
    original_alt_brauchbar = classification.get("original_alt_brauchbar", False)

    # ─── v2.2: Validate decorative classification (safety net) ───
    if ist_dekorativ:
        corrected_type = validate_dekorativ(classification, original_alt, width, height)
        if corrected_type != "dekorativ":
            print(f"v2.2 Dekorativ-Korrektur: {image_path} -> {corrected_type} (war dekorativ)")
            bildtyp = corrected_type
            ist_dekorativ = False
        else:
            # v2.2.1: If original_alt is useless AND image is >50px, do a Qwen recheck
            # to catch images like web_19.jpg (visual metaphor classified as decorative)
            useless_alts = {"", "bild", "grafik", "foto", "image", "img"}
            clean_orig = (original_alt or "").strip().lower()
            if clean_orig in useless_alts and width > 50 and height > 50:
                recheck_result = _call_mistral_json(image_path, DEKORATIV_RECHECK_PROMPT, MISTRAL_MODEL_CLASSIFY, max_tokens=200) or {}
                recheck_result = {"raw_response": json.dumps(recheck_result), **recheck_result}

                # Parse recheck result directly from raw_response (avoids _call_ollama JSON ambiguity)
                raw = recheck_result.get("raw_response", "") or recheck_result.get("alt_text", "")
                recheck_dekorativ = True
                recheck_alt = ""
                try:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(raw[start:end])
                        recheck_dekorativ = parsed.get("ist_dekorativ", True)
                        recheck_alt = parsed.get("kurzbeschreibung", "").strip()
                except Exception:
                    recheck_dekorativ = recheck_result.get("ist_dekorativ", True)

                if not recheck_dekorativ and recheck_alt and len(str(recheck_alt)) > 5:
                    clean_alt = str(recheck_alt).strip().strip('"')
                    print(f"v2.2.1 Dekorativ-Recheck: {image_path} -> NICHT dekorativ: '{clean_alt[:60]}'")
                    return _apply_postfilter({
                        "bildtyp": "foto",
                        "alt_text": clean_alt,
                        "langbeschreibung": "",
                        "ist_dekorativ": False,
                        "konfidenz": "mittel",
                    }, image_hash=img_hash)
                else:
                    print(f"v2.2.1 Dekorativ-Recheck: {image_path} -> bestaetigt dekorativ")

            return {
                "bildtyp": "dekorativ",
                "alt_text": "",
                "langbeschreibung": "",
                "ist_dekorativ": True,
                "konfidenz": konfidenz,
            }

    # ─── v2.2: Functional elements → keep original alt-text ───
    if bildtyp == "funktional":
        clean_alt = (original_alt or "").strip()
        useless_alts = {"", "bild", "grafik", "foto", "image", "img"}
        if clean_alt and clean_alt.lower() not in useless_alts:
            print(f"v2.2 Funktional-Bypass: {image_path} -> behalte Original-Alt: '{clean_alt}'")
            return {
                "bildtyp": "funktional",
                "alt_text": clean_alt,
                "langbeschreibung": "",
                "ist_dekorativ": False,
                "konfidenz": konfidenz,
            }
        # No good original → let Mistral/Qwen generate functional alt-text

    # ─── Stufe 2: Generate alt-text ───
    if should_use_mistral(bildtyp, konfidenz):
        # Mistral generates (v2.2: with thumbnail + improvement mode)
        mistral_result = _call_mistral_generate(
            image_path, bildtyp, enriched_context,
            width=width, height=height,
            original_alt=original_alt,
            original_alt_brauchbar=original_alt_brauchbar
        )
        if mistral_result:
            mistral_result["bildtyp"] = bildtyp
            mistral_result["konfidenz"] = konfidenz
            mistral_result["ist_dekorativ"] = False
            # v2.2.1: Force empty langbeschreibung for thumbnails
            if is_thumbnail(width, height) and mistral_result.get("langbeschreibung"):
                mistral_result["langbeschreibung"] = ""
            # v3.4 Stufe 3: Validator (mit Kontext, damit Link-Verweise nicht entfernt werden)
            mistral_result = _apply_validator(image_path, mistral_result, context=enriched_context)
            return _apply_postfilter(mistral_result, image_hash=img_hash)
        # Mistral failed and Qwen-Fallback is retired (29.04.2026, GPU-Server gekuendigt).
        # Statt einen Connection-Error-String als Alt-Text zu speichern: leerer Alt-Text
        # + needs_review=1, damit der User das Bild manuell pruefen / re-generieren kann.
        print(f"Mistral fehlgeschlagen fuer {image_path} - kein Fallback, needs_review=1", flush=True)
        return _apply_postfilter({
            "bildtyp": bildtyp,
            "alt_text": "",
            "langbeschreibung": "",
            "ist_dekorativ": False,
            "konfidenz": konfidenz,
            "needs_review": True,
            "pipeline_steps": "mistral_failed",
            "validation_result": "",
        }, image_hash=img_hash)

    # PIPELINE_MODE != mistral_primary darf nicht mehr vorkommen (qwen_only/hybrid retired).
    # Defensiv: gleiches Verhalten wie Mistral-Fail.
    print(f"PIPELINE_MODE={PIPELINE_MODE!r} nicht unterstuetzt - retired am 29.04.2026", flush=True)
    return _apply_postfilter({
        "bildtyp": bildtyp,
        "alt_text": "",
        "langbeschreibung": "",
        "ist_dekorativ": False,
        "konfidenz": konfidenz,
        "needs_review": True,
        "pipeline_steps": "pipeline_mode_retired",
        "validation_result": "",
    }, image_hash=img_hash)
