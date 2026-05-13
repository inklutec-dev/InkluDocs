"""Intent-Router: kleines LLM klassifiziert die User-Nachricht.

Bei Provider-Fehler oder unverstaendlicher Antwort: Fallback auf
'smalltalk' (sicher, kein Schaden).

Modell-Override per ENV INKLUAGENT_INTENT_MODEL (provider-spezifischer
Identifier, z.B. Haiku-4.5 fuer Bedrock zur Kostenoptimierung).
"""
import logging
import os
from typing import Literal

from .prompts.system_intent import SYSTEM_INTENT


log = logging.getLogger(__name__)

Intent = Literal["smalltalk", "generate_fresh", "modify_existing", "set_text"]
_VALID_INTENTS: tuple[Intent, ...] = ("smalltalk", "generate_fresh", "modify_existing", "set_text")


def classify_intent(user_message: str, provider) -> Intent:
    """Klassifiziere die Absicht des Users.

    Optional: INKLUAGENT_INTENT_MODEL setzt ein spezifisches Modell
    (z.B. Haiku-4.5 bei Bedrock). Sonst nutzt der Provider seinen Default.

    Bei Fehlern oder unklarer Antwort: Fallback 'smalltalk'.
    """
    intent_model = os.environ.get("INKLUAGENT_INTENT_MODEL", "").strip() or None
    try:
        raw = provider.chat(
            messages=[
                {"role": "system", "content": SYSTEM_INTENT},
                {"role": "user", "content": user_message},
            ],
            model=intent_model,
            max_tokens=10,
            temperature=0.0,
        )
    except Exception as e:
        log.warning("Intent-Klassifikation Provider-Call fehlgeschlagen: %s", e)
        return "smalltalk"

    label = raw.strip().lower().strip(".\"' ")
    if label in _VALID_INTENTS:
        return label  # type: ignore[return-value]
    log.info("Intent-Klassifikator lieferte unbekanntes Label %r → fallback smalltalk", raw)
    return "smalltalk"
