"""Chat-Kern: nimmt User-Nachricht im Projekt-Kontext entgegen,
klassifiziert Intent, dispatched zu passendem Handler.

Pfade:
- smalltalk:        freier Chat ohne Bild (Mistral-Medium oder Pixtral)
- generate_fresh:   ruft InkluDocs-Pipeline pro Bild auf
- modify_existing:  Pixtral-Single-Shot mit User-Hint, Bild + bestehender Text
- set_text:         Bestaetigungs-Rueckfrage, dann DB-Update
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from . import storage
from .router import classify_intent
from .providers.mistral import MistralProvider, MistralProviderError
from .providers.bedrock import BedrockProvider, BedrockProviderError
from .adapters.inkludocs import (
    get_project_context,
    resolve_image_refs,
    run_pipeline_for_image,
    update_alt_text,
    build_project_summary,
    MAX_IMAGES_PER_REQUEST,
)
from .prompts.system_smalltalk import SYSTEM_SMALLTALK
from .prompts.system_modify import SYSTEM_MODIFY
from .prompts.system_modify_extras import (
    SYSTEM_MODIFY_BEHAELTER_BLOCK,
    CONTAINER_WORDS,
    CONTAINER_IMAGE_TYPES,
)
from .sanitize import sanitize_markdown


log = logging.getLogger(__name__)

_PROVIDER_NAME = os.environ.get("INKLUAGENT_PROVIDER", "mistral").lower().strip()
if _PROVIDER_NAME == "bedrock":
    _provider = BedrockProvider()
    log.info("InkluAgent: BedrockProvider aktiv (Claude via Frankfurt)")
else:
    _provider = MistralProvider()
    log.info("InkluAgent: MistralProvider aktiv")

# Agentic-Modus (Tool-Use-Loop) — Default an wenn Bedrock-Provider, sonst aus.
# Override via INKLUAGENT_AGENTIC=true|false.
_AGENTIC_ENABLED = os.environ.get(
    "INKLUAGENT_AGENTIC",
    "true" if _PROVIDER_NAME == "bedrock" else "false",
).lower().strip() == "true"
if _AGENTIC_ENABLED:
    log.info("InkluAgent: agentic Tool-Use-Loop aktiviert")

_HISTORY_TURNS = 30  # letzte 30 Nachrichten als Kontext mitgeben (05.05.2026 von 8 erhoeht)


def process_message(project_id: int, user_message: str, user_id: int) -> dict:
    """Hauptfunktion. Returns:
    {"reply": str, "intent": str, "image_refs": list[int]|None,
     "actions": list[dict]}.
    """
    project = get_project_context(project_id, user_id)
    if project is None:
        return {
            "reply": "Projekt nicht gefunden oder kein Zugriff.",
            "intent": "error",
            "image_refs": None,
            "actions": [],
        }

    # Neuer agentic Pfad: Sonnet entscheidet selbst welche Tools er nutzt.
    # Klassischer 4-Pfad-Dispatcher bleibt unten als Fallback (z.B. Mistral-Provider).
    if _AGENTIC_ENABLED and _PROVIDER_NAME == "bedrock":
        from .agent_loop import run_agent
        try:
            return run_agent(project_id, user_id, user_message, project, _provider)
        except Exception as e:
            log.exception("agentic run_agent crashte — Fallback auf klassischen Dispatcher")
            # Fallthrough zum klassischen Pfad statt User-Fehler

    intent = classify_intent(user_message, _provider)
    log.info("InkluAgent intent=%s project=%s user=%s", intent, project_id, user_id)

    if intent == "smalltalk":
        return _handle_smalltalk(project_id, user_message, project)
    if intent == "generate_fresh":
        return _handle_generate_fresh(project, user_message, user_id)
    if intent == "modify_existing":
        return _handle_modify_existing(project, user_message)
    if intent == "set_text":
        return _handle_set_text(project, user_message)

    return _handle_smalltalk(project_id, user_message, project)


def _load_history_messages(project_id: int) -> list[dict]:
    """Lade die letzten N Nachrichten als Mistral-kompatibles Format."""
    full = storage.get_history(project_id, limit=200)
    recent = full[-_HISTORY_TURNS:]
    return [{"role": m["role"], "content": m["content"]} for m in recent
            if m["role"] in ("user", "assistant")]


def _resolve_refs_with_history_fallback(
    user_message: str,
    project: dict,
) -> tuple[list[dict], str | None]:
    """Versuche Bild-Refs aus der aktuellen Nachricht zu lesen.

    Wenn keine drin sind, schaue in die letzten Verlauf-Nachrichten — die
    juengste mit einer Bild-Referenz gewinnt. So versteht der Bot Folge-
    Anweisungen wie 'mach das nochmal' oder 'pruef das' ohne dass der
    User die Nummer wiederholen muss.
    """
    refs, error = resolve_image_refs(user_message, project["images"])
    if error or refs:
        return refs, error
    history = storage.get_history(project["id"], limit=10)
    for msg in reversed(history):
        prior_refs, _ = resolve_image_refs(msg["content"], project["images"])
        if prior_refs:
            return prior_refs, None
    return [], None


def _handle_smalltalk(project_id: int, user_message: str, project: dict) -> dict:
    history = _load_history_messages(project_id)
    project_summary = build_project_summary(project)
    project_context_msg = (
        "PROJEKT-KONTEXT (zur Beantwortung von Fragen ueber dieses Projekt):\n\n"
        + project_summary
        + "\n\nDu kannst dich beim Antworten auf diese Bilder-Uebersicht stuetzen, "
          "z.B. Alt-Texte zitieren oder erklaeren welche Bilder noch keinen Text haben. "
          "Wenn der Nutzer einen Bild-Inhalt visuell beurteilen lassen will (z.B. 'passt der "
          "Alt-Text zum Bild?'), erklaere ihm freundlich, dass er dafuer eine konkrete "
          "Bild-Anweisung formulieren soll wie 'Bild 3 pruefen' oder 'Bild 3 in leichter Sprache'."
    )
    messages = (
        [{"role": "system", "content": SYSTEM_SMALLTALK}]
        + [{"role": "system", "content": project_context_msg}]
        + history
        + [{"role": "user", "content": user_message}]
    )
    try:
        reply = _provider.chat(messages=messages, max_tokens=600, temperature=0.5)
    except (MistralProviderError, BedrockProviderError) as e:
        log.error("Smalltalk-Call fehlgeschlagen: %s", e)
        reply = ("Entschuldige, ich kann gerade nicht antworten. "
                 "Bitte versuch es in einem Moment nochmal.")
    return {"reply": sanitize_markdown(reply), "intent": "smalltalk",
            "image_refs": None, "actions": []}


def _handle_generate_fresh(project: dict, user_message: str, user_id: int) -> dict:
    refs, error = _resolve_refs_with_history_fallback(user_message, project)
    if error:
        return {"reply": error, "intent": "generate_fresh",
                "image_refs": None, "actions": []}
    if not refs:
        return {
            "reply": ("Welche Bilder soll ich neu generieren? Bitte nenne sie "
                      "per Nummer, z.B. 'Bild 3' oder 'Bilder 1-5'."),
            "intent": "generate_fresh",
            "image_refs": None,
            "actions": [],
        }

    results = []
    for img in refs:
        try:
            res = run_pipeline_for_image(img["id"], project["id"], user_id)
            if res:
                results.append((img["nr"], res["alt_text"]))
        except Exception as e:
            log.exception("Pipeline-Fehler fuer Bild %s: %s", img["nr"], e)
            results.append((img["nr"], f"[Fehler: {e}]"))

    lines = [f"Pipeline ausgefuehrt fuer {len(results)} Bild(er):"]
    for nr, txt in results:
        lines.append(f"\nBild {nr}: {txt}")
    return {
        "reply": "\n".join(lines),
        "intent": "generate_fresh",
        "image_refs": [img["id"] for img in refs],
        "actions": [{"type": "refresh_image", "image_id": img["id"]} for img in refs],
    }


def _handle_modify_existing(project: dict, user_message: str) -> dict:
    refs, error = _resolve_refs_with_history_fallback(user_message, project)
    if error:
        return {"reply": error, "intent": "modify_existing",
                "image_refs": None, "actions": []}
    if not refs:
        return {
            "reply": ("Welches Bild soll ich ueberarbeiten? Bitte nenne es "
                      "per Nummer, z.B. 'Bild 3 in leichter Sprache'."),
            "intent": "modify_existing",
            "image_refs": None,
            "actions": [],
        }

    parts = []
    for img in refs:
        text = _modify_one_image(img, user_message, project["id"])
        parts.append(f"Bild {img['nr']}:\n{text}")

    reply = sanitize_markdown("\n\n".join(parts))
    if len(refs) == 1:
        reply += ("\n\nWenn der Vorschlag passt, sag z.B. 'trag das bei Bild "
                  f"{refs[0]['nr']} ein'.")
    return {
        "reply": reply,
        "intent": "modify_existing",
        "image_refs": [img["id"] for img in refs],
        "actions": [],
    }


_RE_PREV_PROPOSAL = re.compile(
    r"^Bild\s+\d+:\s*\n(.+?)(?:\n\(Begruendung:|\Z)",
    re.DOTALL | re.MULTILINE,
)


# Heuristische Sprach-Erkennung: zaehle Marker-Woerter in den ersten Tokens.
# Reicht voellig fuer DE/EN/FR/ES; bei Unsicherheit Fallback "Deutsch".
_LANG_MARKERS = {
    "Deutsch": {"der", "die", "das", "und", "mit", "ein", "eine", "ist", "auf",
                "viele", "kleine", "schalen", "fluessigkeit", "tuch", "im", "von"},
    "Englisch": {"the", "and", "with", "a", "an", "of", "is", "are", "on",
                 "many", "small", "bowls", "liquid", "cloth"},
    "Franzoesisch": {"le", "la", "les", "et", "avec", "un", "une", "des", "sur",
                     "petits", "petites", "bols", "tissu"},
    "Spanisch": {"el", "la", "los", "las", "y", "con", "un", "una", "en",
                 "pequenos", "pequenas", "cuencos", "tela"},
}


def _detect_language_label(text: str) -> str:
    """Heuristik: gib eine deutsche Sprach-Bezeichnung zurueck."""
    words = re.findall(r"[a-zA-ZaeoeueAEOEUEss]+", text.lower())
    if not words:
        return "Deutsch"
    scores = {lang: sum(1 for w in words if w in markers)
              for lang, markers in _LANG_MARKERS.items()}
    best_lang, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "Deutsch"  # Fallback
    return best_lang


def _get_previous_modify_proposal_for_image(project_id: int, image_id: int) -> Optional[str]:
    """Hole den letzten alt_text_neu-Vorschlag des Bots fuer dieses Bild.

    Sucht in den letzten Verlauf-Nachrichten nach assistant-modify_existing-
    Antworten, die das Bild referenzieren, und extrahiert den Vorschlag aus
    dem 'Bild N:\n{vorschlag}\n(Begruendung: ...)' Format.
    """
    history = storage.get_history(project_id, limit=30)
    for msg in reversed(history):
        if msg["role"] != "assistant":
            continue
        if msg.get("intent") != "modify_existing":
            continue
        refs = msg.get("image_refs") or []
        if image_id not in refs:
            continue
        m = _RE_PREV_PROPOSAL.search(msg["content"])
        if m:
            return m.group(1).strip()
    return None


def _modify_one_image(img: dict, user_message: str, project_id: int) -> str:
    """Pixtral-Single-Shot: Bild + bestehender Text + User-Wunsch.

    Wenn fuer dieses Bild bereits ein vorheriger Bot-Vorschlag im Verlauf
    existiert, wird er Pixtral als Iteration-Basis mitgegeben.
    """
    if not os.path.exists(img["image_path"]):
        return f"[Bild-Datei nicht gefunden: {img['image_path']}]"
    try:
        with open(img["image_path"], "rb") as f:
            img_bytes = f.read()
    except OSError as e:
        return f"[Lese-Fehler: {e}]"

    prev_proposal = _get_previous_modify_proposal_for_image(project_id, img["id"])

    prompt_lines = [
        f"AKTUELLER ALT-TEXT (in der Datenbank gespeichert): \"{img['alt_effective']}\"",
    ]
    if prev_proposal:
        prev_lang = _detect_language_label(prev_proposal)
        prompt_lines.append("")
        prompt_lines.append(
            f"DEIN VORHERIGER VORSCHLAG IN DIESEM GESPRAECH (Sprache: {prev_lang}): "
            f"\"{prev_proposal}\""
        )
        prompt_lines.append(
            "Wenn der User eine Folge-Anweisung gibt (z.B. 'mach das nochmal anders', "
            "'kuerzer', 'das war falsch', 'laenger'), beziehe dich auf diesen vorherigen "
            "Vorschlag, nicht auf den Original-Datenbank-Text. WICHTIG: behalte die "
            f"Sprache des vorherigen Vorschlags ({prev_lang}) bei, ausser der User "
            "verlangt ausdruecklich eine andere Sprache."
        )
    prompt_lines.append("")
    prompt_lines.append(f"USER-WUNSCH: {user_message}")
    prompt_lines.append("")
    prompt_lines.append("Ueberarbeite den Alt-Text gemaess Wunsch. Antworte als JSON wie spezifiziert.")
    prompt_user = "\n".join(prompt_lines)

    messages = [{"role": "system", "content": SYSTEM_MODIFY}]
    if _needs_behaelter_block(img):
        messages.append({"role": "system", "content": SYSTEM_MODIFY_BEHAELTER_BLOCK})
    messages.append({"role": "user", "content": prompt_user})

    try:
        raw = _provider.chat(messages=messages, images=[img_bytes],
                             max_tokens=600, temperature=0.4)
    except (MistralProviderError, BedrockProviderError) as e:
        return f"[Mistral-Fehler: {e}]"

    return _parse_modify_response(raw)


def _needs_behaelter_block(img: dict) -> bool:
    """Entscheide ob der Behaelter-Zusatzblock fuer dieses Bild gebraucht wird.

    Greift bei:
    - Bildtyp aus Pipeline-Klassifikation in CONTAINER_IMAGE_TYPES (z.B. foto_objekte)
    - oder Behaelter-Wort im aktuellen Alt-Text (Heuristik fuer v3.7-Stand
      wo der Bildtyp nur 'foto' ist)
    """
    image_type = (img.get("image_type") or "").lower()
    if image_type in CONTAINER_IMAGE_TYPES:
        return True
    alt = (img.get("alt_effective") or "").lower()
    if not alt:
        return False
    return any(re.search(rf"\b{re.escape(w)}\b", alt) for w in CONTAINER_WORDS)


def _parse_modify_response(raw: str) -> str:
    """Strikte JSON-Validierung der Modify-Antwort.

    Erwartet: {"alt_text_neu": "...", "begruendung": "..."} mit beiden
    Feldern als nicht-leere Strings. Markdown wird in beiden Strings
    sanitisiert. Bei Verletzung sauberer Fallback statt rohem Text.
    """
    if not raw:
        return "[Mistral lieferte eine leere Antwort.]"
    cleaned = raw.strip()
    # Codefences abstreifen, falls Mistral sie trotz Verbot setzt
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return ("[Mistral hat kein valides JSON geliefert. Antwort:\n"
                + sanitize_markdown(cleaned)[:500] + "]")
    if not isinstance(data, dict):
        return "[Modify-Antwort ist kein JSON-Objekt.]"
    alt_neu = data.get("alt_text_neu")
    begr = data.get("begruendung")
    if not isinstance(alt_neu, str) or not alt_neu.strip():
        return "[Modify-Antwort hat keinen alt_text_neu — bitte erneut versuchen.]"
    if not isinstance(begr, str) or not begr.strip():
        begr = "(keine Begruendung geliefert)"
    alt_neu = sanitize_markdown(alt_neu).strip()
    begr = sanitize_markdown(begr).strip()
    return f"{alt_neu}\n(Begruendung: {begr})"


def _handle_set_text(project: dict, user_message: str) -> dict:
    """Bei set_text fragt der Bot zurueck (kein Direkt-Ueberschreiben).

    Dieser erste Wurf antwortet mit einer Bestaetigungs-Rueckfrage. Die
    eigentliche Eintragung erfolgt in einem Folge-Schritt, wenn der User
    bestaetigt — das wird in einer spaeteren Iteration mit
    State-Tracking implementiert. Aktuell weist der Bot darauf hin.
    """
    refs, error = _resolve_refs_with_history_fallback(user_message, project)
    if error:
        return {"reply": error, "intent": "set_text",
                "image_refs": None, "actions": []}
    if not refs:
        return {
            "reply": ("Bei welchem Bild soll ich den Text eintragen? "
                      "Bitte nenne es per Nummer, z.B. 'trag das bei Bild 3 ein'."),
            "intent": "set_text",
            "image_refs": None,
            "actions": [],
        }

    img = refs[0]
    return {
        "reply": (
            f"Im Feld fuer Bild {img['nr']} steht aktuell: "
            f"\"{img['alt_effective']}\". \n\n"
            f"Das automatische Eintragen folgt in einer naechsten Version. "
            f"Aktuell kannst du den Vorschlag aus meiner letzten Antwort "
            f"manuell in das Alt-Text-Feld kopieren."
        ),
        "intent": "set_text",
        "image_refs": [img["id"]],
        "actions": [],
    }
