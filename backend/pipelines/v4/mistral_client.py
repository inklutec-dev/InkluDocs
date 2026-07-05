"""Mistral-API-Client mit Strict JSON Schema Mode für die v4-Pipeline.

Strict Mode (Mistral 2026-Feature):
  response_format = {
      'type': 'json_schema',
      'json_schema': {
          'name': SchemaClass.__name__,
          'schema': SchemaClass.model_json_schema(),
          'strict': True,
      }
  }
Mistral erzwingt das Schema serverseitig — der Output ist garantiert gültiges
JSON, das dem Schema entspricht. Belt-and-Suspenders: clientseitige Pydantic-
Validierung als zusätzliche Sicherheit gegen Edge-Cases.

Retry-Strategie:
- Strict Mode hat selten Schema-Verletzungen (sehr tiefe verschachtelte Schemas
  ab 4+ Ebenen können laut Mistral-Doku gelegentlich relaxieren).
- Bei ValidationError: EIN Retry mit erweitertem Prompt ('Du hast Feld X
  übersprungen, fülle es zwingend'). Mehr Retries kosten Geld ohne Mehrwert
  (gleicher Modell-Aufruf-Determinismus bei temperature=0).

Modell-Wahl:
- MISTRAL_MODEL_CLASSIFY = mistral-medium-latest (schnell + günstig für simple Wahl)
- MISTRAL_MODEL_INVENTAR = pixtral-large-latest (gute Vision für Forensik)
- MISTRAL_MODEL_GENERATE = mistral-large-latest (Reasoning-stark für Output-Formulierung)
- MISTRAL_MODEL_VALIDATE = pixtral-large-latest (Modell-Diversität gegen GENERATE)

Modell-Diversität gegen korrelierte Halluzinationen: GENERATE und VALIDATE
nutzen UNTERSCHIEDLICHE Modelle. Wenn beide Pixtral wären, würden die
gleichen Halluzinationen durchrutschen.
"""
from __future__ import annotations

import logging
import os
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

# _resize_image_for_model wird lazy in den Funktionen importiert (Circular-
# Import-Vermeidung: pdf_processor importiert orchestrator).

log = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Modell-Konstanten — können später per ENV überschrieben werden, falls
# Mistral neue Modelle veröffentlicht und wir A/B-testen wollen.
MISTRAL_MODEL_CLASSIFY = os.environ.get('V4_MODEL_CLASSIFY', 'mistral-medium-latest')
MISTRAL_MODEL_INVENTAR = os.environ.get('V4_MODEL_INVENTAR', 'pixtral-large-latest')
MISTRAL_MODEL_GENERATE = os.environ.get('V4_MODEL_GENERATE', 'mistral-large-latest')
MISTRAL_MODEL_VALIDATE = os.environ.get('V4_MODEL_VALIDATE', 'pixtral-large-latest')

# Endpoint und Timeout
_MISTRAL_ENDPOINT = 'https://api.mistral.ai/v1/chat/completions'
_HTTP_TIMEOUT = 90.0  # großzügig — komplexe Inventar-Pässe können länger dauern


class MistralCallError(Exception):
    """Erhoben wenn der Mistral-Call fundamental schief geht (kein Retry-würdig).

    Beispiele: kein API-Key, HTTP-5xx, Schema-Validation auch nach Retry fehlgeschlagen.
    """


def _http_call_mistral(
    model: str,
    prompt: str,
    image_b64: str,
    max_tokens: int,
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> str:
    """HTTP-Call zu Mistral. Gibt rohen content-String zurück.

    Bei response_format=json_schema strict liefert Mistral garantiert
    valides JSON gegen das Schema; sonst liefert es plain text.
    """
    api_key = os.environ.get('MISTRAL_API_KEY', '')
    if not api_key:
        raise MistralCallError(
            'MISTRAL_API_KEY env-Variable nicht gesetzt — Mistral-Call unmöglich.'
        )

    payload: dict = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
            ],
        }],
        'max_tokens': max_tokens,
        'temperature': temperature,  # 0 = deterministisch (Default); >0 nur bei explizitem Neu-Generieren
    }
    if response_format is not None:
        payload['response_format'] = response_format

    try:
        response = httpx.post(
            _MISTRAL_ENDPOINT,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=_HTTP_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise MistralCallError(f'HTTP-Fehler bei Mistral-Call: {e}') from e

    try:
        data = response.json()
        return data['choices'][0]['message']['content']
    except (KeyError, IndexError, ValueError) as e:
        raise MistralCallError(
            f'Unerwartetes Antwort-Format von Mistral: {e}; raw: {response.text[:300]}'
        ) from e


def call_mistral_with_schema(
    model: str,
    prompt: str,
    image_path: str,
    schema: Type[T],
    max_tokens: int = 1500,
    system: 'str | None' = None,
    temperature: float = 0.0,
) -> T:
    """Mistral-Call mit Strict JSON Schema Mode + Pydantic-Validierung.

    Bei seltenem Schema-Bypass: EIN Retry mit härterer Anweisung. Wenn auch
    der Retry fehlschlägt, wird MistralCallError geworfen — der Orchestrator
    soll die Stelle mit needs_review=True markieren oder die Pipeline
    abbrechen (je nach Pass).

    Args:
      model: Modell-ID (siehe MISTRAL_MODEL_*-Konstanten).
      prompt: Vollständiger Builder-Output (siehe prompts/builders/).
      image_path: Pfad zum Bild (wird auf max 1024x1024 für Mistral skaliert).
      schema: Pydantic-Klasse für strict JSON Schema + clientseitige Validierung.
      max_tokens: Tokens-Budget für die Antwort.

    Returns:
      Pydantic-Instanz vom übergebenen Schema, garantiert validiert.
    """
    from pdf_processor import _resize_image_for_model  # lazy zur Vermeidung von Circular-Import
    img_b64 = _resize_image_for_model(image_path)
    if system:
        # Mistral-Fallback: System-Anteil dem Prompt voranstellen (kein separates Feld noetig).
        prompt = system + '\n\n' + prompt
    schema_dict = schema.model_json_schema()

    response_format = {
        'type': 'json_schema',
        'json_schema': {
            'name': schema.__name__,
            'schema': schema_dict,
            'strict': True,
        },
    }

    raw_text = _http_call_mistral(
        model=model,
        prompt=prompt,
        image_b64=img_b64,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )

    try:
        return schema.model_validate_json(raw_text)
    except ValidationError as e:
        # Sehr seltener Fall: Strict Mode hat etwas durchgehen lassen.
        # EIN Retry mit explizitem Hinweis auf den konkreten Fehler.
        log.warning(
            'Strict-Schema-Bypass für %s: %s — versuche Retry mit Korrektur-Hinweis.',
            schema.__name__, e,
        )
        retry_prompt = (
            prompt
            + '\n\n--- KRITISCHE KORREKTUR ---\n'
            + f'Vorherige Antwort verletzte das Schema: {e}\n'
            + 'Liefere die Antwort exakt nach Schema, jedes Pflichtfeld ausgefüllt.'
        )
        retry_raw = _http_call_mistral(
            model=model,
            prompt=retry_prompt,
            image_b64=img_b64,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        try:
            return schema.model_validate_json(retry_raw)
        except ValidationError as e2:
            raise MistralCallError(
                f'Strict-Schema verletzt auch nach Retry für {schema.__name__}: {e2}; '
                f'raw retry response: {retry_raw[:300]}'
            ) from e2
