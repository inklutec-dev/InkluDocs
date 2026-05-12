"""Intent-Router: Mistral-Medium klassifiziert die User-Nachricht.

Bei Mistral-Fehler oder unverstaendlicher Antwort: Fallback auf
'smalltalk' (sicher, kein Schaden).
"""
import logging
from typing import Literal

from .prompts.system_intent import SYSTEM_INTENT


log = logging.getLogger(__name__)

Intent = Literal["smalltalk", "generate_fresh", "modify_existing", "set_text"]
_VALID_INTENTS: tuple[Intent, ...] = ("smalltalk", "generate_fresh", "modify_existing", "set_text")


def classify_intent(user_message: str, provider) -> Intent:
    """Klassifiziere die Absicht des Users via Mistral-Medium.

    Bei Fehlern oder unklarer Antwort: Fallback 'smalltalk'.
    """
    try:
        raw = provider.chat(
            messages=[
                {"role": "system", "content": SYSTEM_INTENT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=10,
            temperature=0.0,
        )
    except Exception as e:
        log.warning("Intent-Klassifikation Mistral-Call fehlgeschlagen: %s", e)
        return "smalltalk"

    label = raw.strip().lower().strip(".\"' ")
    if label in _VALID_INTENTS:
        return label  # type: ignore[return-value]
    log.info("Intent-Klassifikator lieferte unbekanntes Label %r → fallback smalltalk", raw)
    return "smalltalk"
