"""LLM-Provider-Switch fuer die v4-Pipeline.

Per ENV-Variable LLM_PROVIDER waehlt der Orchestrator zwischen 'mistral'
(Default, bestehend) und 'bedrock' (Anthropic Claude via AWS Frankfurt,
DSGVO-konform). Beide Provider bieten dieselbe Funktion call_with_schema
und dieselben MODEL_*-Konstanten — Orchestrator bleibt Provider-agnostisch.

Der Switch erfolgt zur Modul-Lade-Zeit. Provider-Wechsel im laufenden
Container ist nicht vorgesehen — ENV setzen, Container restart, fertig.
Das ist ausreichend, weil wir per Provider testen wollen, nicht per Bild.
"""
from __future__ import annotations

import os

_PROVIDER = os.environ.get('LLM_PROVIDER', 'mistral').lower().strip()


class LLMCallError(Exception):
    """Generischer LLM-Call-Fehler — Orchestrator faengt nur diesen Typ."""


if _PROVIDER == 'bedrock':
    from .bedrock_client import (
        BEDROCK_MODEL_CLASSIFY as MODEL_CLASSIFY,
        BEDROCK_MODEL_GENERATE as MODEL_GENERATE,
        BEDROCK_MODEL_INVENTAR as MODEL_INVENTAR,
        BEDROCK_MODEL_VALIDATE as MODEL_VALIDATE,
        BedrockCallError,
        call_bedrock_with_schema as _provider_call,
    )
    _ProviderError = BedrockCallError
elif _PROVIDER == 'mistral':
    from .mistral_client import (
        MISTRAL_MODEL_CLASSIFY as MODEL_CLASSIFY,
        MISTRAL_MODEL_GENERATE as MODEL_GENERATE,
        MISTRAL_MODEL_INVENTAR as MODEL_INVENTAR,
        MISTRAL_MODEL_VALIDATE as MODEL_VALIDATE,
        MistralCallError,
        call_mistral_with_schema as _provider_call,
    )
    _ProviderError = MistralCallError
else:
    raise RuntimeError(
        f"Unbekannter LLM_PROVIDER '{_PROVIDER}' — erlaubt: 'mistral', 'bedrock'."
    )


def call_with_schema(model, prompt, image_path, schema, max_tokens=1500, temperature=0.0, system=None):
    """Provider-agnostischer LLM-Call mit Strict-JSON-Schema.

    Wirft generisch LLMCallError, damit Orchestrator nur einen Exception-Typ
    fangen muss (statt MistralCallError ODER BedrockCallError).
    """
    try:
        return _provider_call(
            model=model,
            prompt=prompt,
            image_path=image_path,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
    except _ProviderError as e:
        raise LLMCallError(str(e)) from e


def get_provider_name() -> str:
    return _PROVIDER
