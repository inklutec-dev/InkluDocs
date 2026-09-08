"""LLM-Zugang der v4-Pipeline: Anbieter je Rolle.

Seit 07.09.2026 abends gibt es zwei Anbieter: Claude über Amazon Bedrock (EU-Region) und Google
Gemini. Der Orchestrator ruft weiter provider-neutral `call_with_schema(model, ...)` auf; der
Anbieter ergibt sich aus der Modellkennung (gemini-* -> Gemini, sonst Bedrock). Welche
Modellkennung eine Rolle bekommt, steuert die Umgebung:

  LLM_PROVIDER            Standard für alle Rollen: bedrock (Vorgabe) oder gemini
  LLM_PROVIDER_CLASSIFY   Anbieter nur für den Klassifikator (optional)
  LLM_PROVIDER_INVENTAR   Anbieter nur für den Inventar-Schritt (optional)
  LLM_PROVIDER_GENERATE   Anbieter nur für Beschreibung, Werte-, Zähl- und Faktenschritt (optional)
  LLM_PROVIDER_VALIDATE   Anbieter nur für den Prüfpass (optional)

So ist ein Kreuzmodell (Gemini erzeugt, Sonnet prüft) ein reiner Umgebungsschalter.
Die Modellkennungen je Anbieter kommen aus BEDROCK_MODEL_* bzw. GEMINI_MODEL_*.
Der Mistral-Client wurde am 07.09.2026 abgebaut (Git-Tag sicherung-vor-mistral-abdockung-20260907).
"""
from __future__ import annotations

import os

from . import bedrock_client as _bedrock
from . import gemini_client as _gemini


class LLMCallError(Exception):
    """Generischer LLM-Call-Fehler — Orchestrator faengt nur diesen Typ."""


def _anbieter(rolle: str) -> str:
    wert = os.environ.get(f'LLM_PROVIDER_{rolle}', '') or os.environ.get('LLM_PROVIDER', 'bedrock')
    wert = wert.strip().lower()
    return 'gemini' if wert == 'gemini' else 'bedrock'


def _modell(rolle: str) -> str:
    if _anbieter(rolle) == 'gemini':
        return getattr(_gemini, f'GEMINI_MODEL_{rolle}')
    return getattr(_bedrock, f'BEDROCK_MODEL_{rolle}')


MODEL_CLASSIFY = _modell('CLASSIFY')
MODEL_INVENTAR = _modell('INVENTAR')
MODEL_GENERATE = _modell('GENERATE')
MODEL_VALIDATE = _modell('VALIDATE')


def ist_gemini(model: str) -> bool:
    return (model or '').lower().startswith('gemini')


def call_with_schema(model, prompt, image_path, schema, max_tokens=1500, temperature=0.0, system=None):
    """LLM-Call mit erzwungenem Ausgabeschema; der Anbieter folgt der Modellkennung.

    Wirft generisch LLMCallError, damit der Orchestrator nur einen Exception-Typ fangen muss.
    """
    try:
        if ist_gemini(model):
            return _gemini.call_gemini_with_schema(model=model, prompt=prompt, image_path=image_path, schema=schema,
                                                   max_tokens=max_tokens, temperature=temperature, system=system)
        return _bedrock.call_bedrock_with_schema(model=model, prompt=prompt, image_path=image_path, schema=schema,
                                                 max_tokens=max_tokens, temperature=temperature, system=system)
    except (_bedrock.BedrockCallError, _gemini.GeminiCallError) as e:
        raise LLMCallError(str(e)) from e


def call_text_with_schema(model, prompt, schema, max_tokens=2000, temperature=0.0, system=None):
    """Textaufruf ohne Bild mit Schema; Anbieter folgt der Modellkennung."""
    try:
        if ist_gemini(model):
            return _gemini.call_gemini_text_with_schema(model=model, prompt=prompt, schema=schema,
                                                        max_tokens=max_tokens, temperature=temperature, system=system)
        return _bedrock.call_bedrock_text_with_schema(model=model, prompt=prompt, schema=schema,
                                                      max_tokens=max_tokens, temperature=temperature, system=system)
    except (_bedrock.BedrockCallError, _gemini.GeminiCallError) as e:
        raise LLMCallError(str(e)) from e


def get_provider_name() -> str:
    """Anbieter des Erzeugers (für Anzeigen und Logs)."""
    return _anbieter('GENERATE')
