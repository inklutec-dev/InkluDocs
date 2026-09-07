"""LLM-Zugang der v4-Pipeline: Claude ueber Amazon Bedrock (EU-Region).

Seit 07.09.2026 ist Bedrock der einzige Anbieter. Der fruehere Provider-
Schalter LLM_PROVIDER (mistral/bedrock) und der Mistral-Client wurden
abgebaut; der letzte Stand mit Mistral liegt unter dem Git-Tag
sicherung-vor-mistral-abdockung-20260907.

Dieses Modul bleibt als duenne Fassade bestehen, damit Orchestrator, Eval-
Runner und Tests weiterhin provider-neutrale Namen importieren koennen
(MODEL_*, LLMCallError, call_with_schema).
"""
from __future__ import annotations

from .bedrock_client import (
    BEDROCK_MODEL_CLASSIFY as MODEL_CLASSIFY,
    BEDROCK_MODEL_GENERATE as MODEL_GENERATE,
    BEDROCK_MODEL_INVENTAR as MODEL_INVENTAR,
    BEDROCK_MODEL_VALIDATE as MODEL_VALIDATE,
    BedrockCallError,
    call_bedrock_with_schema as _provider_call,
)


class LLMCallError(Exception):
    """Generischer LLM-Call-Fehler — Orchestrator faengt nur diesen Typ."""


def call_with_schema(model, prompt, image_path, schema, max_tokens=1500, temperature=0.0, system=None):
    """LLM-Call mit erzwungenem Ausgabeschema (Tool-Use).

    Wirft generisch LLMCallError, damit der Orchestrator nur einen
    Exception-Typ fangen muss.
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
    except BedrockCallError as e:
        raise LLMCallError(str(e)) from e


def get_provider_name() -> str:
    return 'bedrock'
