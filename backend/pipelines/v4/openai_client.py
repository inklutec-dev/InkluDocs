"""OpenAI-Anbieter der v4-Pipeline (Responses-API mit JSON-Schema).

Eingeführt am 08.09.2026 für das Kreuzmodell: ein Modell einer anderen Familie prüft die Texte des
Erzeugers. Gleicher Vertrag wie bedrock_client.call_bedrock_with_schema. Schlüssel: OPENAI_API_KEY.
Endpunkt umschaltbar (OPENAI_ENDPOINT), für EU-Datenverarbeitung https://eu.api.openai.com/v1 mit
einem EU-Projekt.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from .gemini_client import _media_type, _prompt_ohne_marker

log = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

OPENAI_MODEL_CLASSIFY = os.environ.get('OPENAI_MODEL_CLASSIFY', 'gpt-5.6-terra')
OPENAI_MODEL_INVENTAR = os.environ.get('OPENAI_MODEL_INVENTAR', 'gpt-5.5')
OPENAI_MODEL_GENERATE = os.environ.get('OPENAI_MODEL_GENERATE', 'gpt-5.5')
OPENAI_MODEL_VALIDATE = os.environ.get('OPENAI_MODEL_VALIDATE', 'gpt-5.6-terra')

_HTTP_TIMEOUT = 180
_VERSUCHE = 3


class OpenAICallError(Exception):
    """Fehler beim OpenAI-Aufruf (Netz, Kontingent, Schema)."""


def _endpunkt() -> str:
    return os.environ.get('OPENAI_ENDPOINT', 'https://api.openai.com/v1').rstrip('/') + '/responses'


def _invoke_openai(model, prompt, image_b64, schema_name, schema_dict, max_tokens, temperature, system) -> dict:
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        raise OpenAICallError('OPENAI_API_KEY fehlt in der Umgebung')
    inhalt = [{'type': 'input_text', 'text': _prompt_ohne_marker(prompt)}]
    if image_b64:
        inhalt.append({'type': 'input_image', 'image_url': f'data:{_media_type(image_b64)};base64,{image_b64}', 'detail': 'high'})
    body = {
        'model': model,
        'input': [{'role': 'user', 'content': inhalt}],
        'text': {'format': {'type': 'json_schema', 'name': schema_name, 'schema': schema_dict, 'strict': False}},
        'max_output_tokens': max(int(max_tokens) * 2, 2000),
    }
    if system:
        body['instructions'] = system
    daten = json.dumps(body).encode('utf-8')
    letzter = None
    for versuch in range(_VERSUCHE):
        req = urllib.request.Request(_endpunkt(), data=daten, headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
                antwort = json.load(r)
            break
        except urllib.error.HTTPError as e:
            text = ''
            try:
                text = e.read().decode('utf-8', 'ignore')[:400]
            except Exception:
                pass
            letzter = OpenAICallError(f'OpenAI HTTP {e.code} ({model}): {text}')
            if e.code in (429, 500, 502, 503, 504) and versuch < _VERSUCHE - 1:
                time.sleep(4 * (versuch + 1)); continue
            raise letzter from e
        except Exception as e:
            letzter = OpenAICallError(f'OpenAI-Aufruf fehlgeschlagen ({model}): {e}')
            if versuch < _VERSUCHE - 1:
                time.sleep(4 * (versuch + 1)); continue
            raise letzter from e
    else:
        raise letzter or OpenAICallError('OpenAI-Aufruf fehlgeschlagen')
    if os.getenv('DEBUG_GEN_RAW', 'false').lower() == 'true':
        u = antwort.get('usage', {}) or {}
        print(f"[OPENAI-USAGE] model={model} schema={schema_name} in={u.get('input_tokens', '?')} out={u.get('output_tokens', '?')}", flush=True)
    text = ''
    for item in antwort.get('output', []) or []:
        for c in item.get('content', []) or []:
            if c.get('type') == 'output_text':
                text += c.get('text', '')
    if not text:
        raise OpenAICallError(f'Keine Textantwort von OpenAI ({model}, {schema_name}); raw: {str(antwort)[:300]}')
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise OpenAICallError(f'OpenAI-Antwort kein JSON ({model}, {schema_name}): {e}; text: {text[:300]}')


def _mit_validierung(model, prompt, img_b64, schema, max_tokens, temperature, system):
    schema_dict = schema.model_json_schema()
    schema_name = schema.__name__
    raw = _invoke_openai(model, prompt, img_b64, schema_name, schema_dict, max_tokens, temperature, system)
    try:
        return schema.model_validate(raw)
    except ValidationError as e:
        log.warning('Schema-Verstoß (OpenAI) bei %s: %s — Retry mit Hinweis.', schema_name, e)
        retry_prompt = (prompt + '\n\n--- KORREKTUR ---\n' + f'Die vorherige Antwort verletzte das Schema: {e}\n'
                        + 'Liefere die Antwort exakt nach Schema, jedes Pflichtfeld ausgefüllt.')
        retry_raw = _invoke_openai(model, retry_prompt, img_b64, schema_name, schema_dict, max_tokens, temperature, system)
        try:
            return schema.model_validate(retry_raw)
        except ValidationError as e2:
            raise OpenAICallError(f'Schema verletzt auch nach Retry für {schema_name}: {e2}; raw: {str(retry_raw)[:300]}') from e2


def call_openai_with_schema(model: str, prompt: str, image_path: str, schema: Type[T], max_tokens: int = 1500,
                            temperature: float = 0.0, system: str | None = None) -> T:
    from pdf_processor import _resize_image_for_model  # lazy: pdf_processor importiert den Orchestrator
    return _mit_validierung(model, prompt, _resize_image_for_model(image_path), schema, max_tokens, temperature, system)


def call_openai_text_with_schema(model: str, prompt: str, schema: Type[T], max_tokens: int = 2000,
                                 temperature: float = 0.0, system: str | None = None) -> T:
    return _mit_validierung(model, prompt, None, schema, max_tokens, temperature, system)
