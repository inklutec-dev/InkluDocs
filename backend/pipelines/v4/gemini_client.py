"""Gemini-Anbieter der v4-Pipeline (Google Gemini API, generateContent mit JSON-Schema).

Eingeführt am 07.09.2026 nach dem Modellvergleich auf dem Prüfkorpus: Gemini 3.1 Pro lieferte mit
einem einzigen Aufruf und unserem Prompt-Gerüst eine Wahrnehmung (Zählen, kleine Zahlen, Montagen)
auf dem Niveau von GPT-6 Astra, für rund 2 Cent je Bild. Gleicher Vertrag wie
bedrock_client.call_bedrock_with_schema: Prompt + Bild + Pydantic-Schema -> validiertes Objekt.

Schlüssel: GEMINI_API_KEY. Endpunkt: Gemini Developer API (generativelanguage.googleapis.com).
Für den Produktivbetrieb mit EU-Datenverarbeitung ist Vertex AI vorgesehen (GEMINI_ENDPOINT
umschaltbar, siehe _endpunkt); die Entwickler-API ist für Staging und Messungen gedacht.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

GEMINI_MODEL_CLASSIFY = os.environ.get('GEMINI_MODEL_CLASSIFY', 'gemini-3.8-flash')
GEMINI_MODEL_INVENTAR = os.environ.get('GEMINI_MODEL_INVENTAR', 'gemini-3.1-pro-preview')
GEMINI_MODEL_GENERATE = os.environ.get('GEMINI_MODEL_GENERATE', 'gemini-3.1-pro-preview')
GEMINI_MODEL_VALIDATE = os.environ.get('GEMINI_MODEL_VALIDATE', 'gemini-3.1-pro-preview')

_HTTP_TIMEOUT = 180
_VERSUCHE = 3
# Denk-Reserve über der Ausgabegrenze (Gemini 3.x: thoughtsTokenCount zählt gegen maxOutputTokens), siehe _invoke_gemini.
_DENKRESERVE = int(os.environ.get('GEMINI_DENKRESERVE', '8000'))


class GeminiCallError(Exception):
    """Fehler beim Gemini-Aufruf (Netz, Kontingent, Schema)."""


def _endpunkt(model: str) -> str:
    from . import gemini_auth
    return gemini_auth.endpunkt(model)


_ERLAUBTE_SCHLUESSEL = {'type', 'description', 'properties', 'required', 'items', 'enum', 'nullable',
                        'minimum', 'maximum', 'minItems', 'maxItems', 'format'}


def _schema_fuer_gemini(schema: dict) -> dict:
    """Pydantic-JSON-Schema -> Gemini-Schema: $defs auflösen, Optional[X] zu nullable X, nur die
    Felder behalten, die Gemini kennt (title, default, minLength usw. weglassen)."""
    s = copy.deepcopy(schema)
    defs = s.pop('$defs', {}) or {}

    def aufloesen(node):
        if isinstance(node, dict):
            if '$ref' in node:
                ziel = copy.deepcopy(defs.get(node['$ref'].split('/')[-1], {}))
                for k in ('description',):
                    if k in node and k not in ziel:
                        ziel[k] = node[k]
                return aufloesen(ziel)
            if 'anyOf' in node:
                kandidaten = [k for k in node['anyOf'] if not (isinstance(k, dict) and k.get('type') == 'null')]
                nullable = len(kandidaten) < len(node['anyOf'])
                if len(kandidaten) == 1:
                    merged = aufloesen(kandidaten[0])
                    if 'description' in node:
                        merged['description'] = node['description']
                    if nullable:
                        merged['nullable'] = True
                    return merged
            out = {}
            for k, v in node.items():
                if k == 'properties' and isinstance(v, dict):
                    out[k] = {name: aufloesen(feld) for name, feld in v.items()}
                elif k in _ERLAUBTE_SCHLUESSEL:
                    out[k] = aufloesen(v)
            if out.get('type') == 'object' and 'properties' in out and 'required' not in out:
                out['required'] = list(out['properties'].keys())
            return out
        if isinstance(node, list):
            return [aufloesen(x) for x in node]
        return node

    return aufloesen(s)


def _media_type(image_b64: str) -> str:
    import base64
    try:
        head = base64.b64decode(image_b64[:32])
    except Exception:
        return 'image/jpeg'
    if head.startswith(b'\x89PNG'):
        return 'image/png'
    if head.startswith(b'GIF8'):
        return 'image/gif'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/jpeg'


def _prompt_ohne_marker(prompt: str) -> str:
    from prompts.builders.helpers import BILDDATEN_MARKER
    return prompt.replace(BILDDATEN_MARKER, '\n\n') if BILDDATEN_MARKER in prompt else prompt


def _invoke_gemini(model: str, prompt: str, image_b64: str | None, schema_name: str, schema_dict: dict,
                   max_tokens: int, temperature: float, system: str | None) -> dict:
    from . import gemini_auth
    try:
        kopf = gemini_auth.kopfzeilen()
    except Exception as e:
        raise GeminiCallError(f'Gemini-Zugang: {e}') from e
    from .anbieter_profil import profil
    p = profil('gemini')
    text_teil = {'text': _prompt_ohne_marker(prompt)}
    if image_b64:
        bild_teil = {'inlineData': {'mimeType': _media_type(image_b64), 'data': image_b64}}
        teile = [bild_teil, text_teil] if p.bild_zuerst else [text_teil, bild_teil]
    else:
        teile = [text_teil]
    body = {
        'contents': [{'role': 'user', 'parts': teile}],
        'generationConfig': {
            'temperature': temperature if p.temperatur is None else p.temperatur,
            # Schema-JSON ist ausführlicher als Tool-Use (×2, mindestens 2000) PLUS Denk-Reserve: Gemini 3.x
            # denkt vor der Antwort, und diese Denk-Tokens zählen gegen maxOutputTokens. Ohne Reserve blieb bei
            # komplexen Grafiken (Prod 14.09.2026, Kundenlauf, 13 von 234 Bildern) nach ~1.900 Denk-Tokens nur
            # Platz für ~30 Ausgabe-Tokens → finishReason=MAX_TOKENS, abgeschnittenes JSON, immer an derselben
            # Stelle (mit dem echten Klassifikator-Prompt auf Staging nachgestellt). GEMINI_DENKRESERVE: Tokens.
            'maxOutputTokens': max(int(max_tokens) * 2, 2000) + _DENKRESERVE,
            'responseMimeType': 'application/json',
            'responseSchema': _schema_fuer_gemini(schema_dict),
        },
    }
    if p.bildaufloesung and image_b64:
        body['generationConfig']['mediaResolution'] = p.bildaufloesung
    if system:
        body['systemInstruction'] = {'parts': [{'text': system}]}
    daten = json.dumps(body).encode('utf-8')
    # Wiederholung (bis _VERSUCHE) bei Transportfehlern UND bei unbrauchbarer Antwort (abgeschnittenes
    # oder leeres JSON). Bis 14.09.2026 wurde nur der Transport wiederholt; eine abgeschnittene
    # Antwort führte sofort zum Bildfehler — auf Prod 13 von 234 Bildern eines Kundenlaufs
    # („Unterminated string“ vom Flash-Klassifikator nach ~90 Zeichen). Ein zweiter Aufruf mit
    # demselben Prompt liefert in der Regel eine vollständige Antwort. Nicht wiederholt wird eine
    # Sperre durch Gemini (promptFeedback.blockReason) — die kommt beim zweiten Mal genauso.
    letzter: Exception | None = None
    for versuch in range(_VERSUCHE):
        req = urllib.request.Request(_endpunkt(model), data=daten, headers=kopf)
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
                antwort = json.load(r)
        except urllib.error.HTTPError as e:
            text = ''
            try:
                text = e.read().decode('utf-8', 'ignore')[:400]
            except Exception:
                pass
            letzter = GeminiCallError(f'Gemini HTTP {e.code} ({model}): {text}')
            if e.code in (429, 500, 502, 503, 504) and versuch < _VERSUCHE - 1:
                time.sleep(4 * (versuch + 1))
                continue
            raise letzter from e
        except Exception as e:
            letzter = GeminiCallError(f'Gemini-Aufruf fehlgeschlagen ({model}): {e}')
            if versuch < _VERSUCHE - 1:
                time.sleep(4 * (versuch + 1))
                continue
            raise letzter from e

        if os.getenv('DEBUG_GEN_RAW', 'false').lower() == 'true':
            u = antwort.get('usageMetadata', {}) or {}
            print(f"[GEMINI-USAGE] model={model} schema={schema_name} in={u.get('promptTokenCount', '?')} "
                  f"out={u.get('candidatesTokenCount', '?')} denk={u.get('thoughtsTokenCount', 0)} "
                  f"cached={u.get('cachedContentTokenCount', 0)}", flush=True)
        try:
            return _antwort_auswerten(antwort, model, schema_name)
        except _AntwortUnbrauchbar as e:
            letzter = GeminiCallError(str(e))
            if versuch < _VERSUCHE - 1:
                log.warning('Gemini-Antwort unbrauchbar (%s, %s), Versuch %d von %d: %s — Wiederholung.',
                            model, schema_name, versuch + 1, _VERSUCHE, str(e)[:200])
                time.sleep(4 * (versuch + 1))
                continue
            raise letzter from e
    raise letzter or GeminiCallError('Gemini-Aufruf fehlgeschlagen')


class _AntwortUnbrauchbar(Exception):
    """Antwort kam an, ist aber nicht verwertbar (leer, abgeschnitten, kein JSON) — wiederholbar."""


def _antwort_auswerten(antwort: dict, model: str, schema_name: str) -> dict:
    """Kandidatentext aus der Gemini-Antwort holen und als JSON lesen.

    Wirft _AntwortUnbrauchbar (wiederholbar) bei fehlendem Kandidaten ohne Sperrgrund, leerem Text
    oder unlesbarem JSON; GeminiCallError (endgültig) bei einer Sperre durch Gemini (blockReason).
    Der finishReason (z. B. MAX_TOKENS, SAFETY, RECITATION) steht in jeder Meldung, damit die Ursache
    im Log erkennbar ist."""
    kandidaten = antwort.get('candidates') or []
    kandidat = kandidaten[0] if kandidaten else {}
    finish = kandidat.get('finishReason')
    sperre = (antwort.get('promptFeedback') or {}).get('blockReason')
    if sperre:
        raise GeminiCallError(f'Gemini hat die Anfrage gesperrt ({model}, {schema_name}): {sperre}; raw: {str(antwort)[:300]}')
    try:
        text = ''.join(p.get('text', '') for p in kandidat['content']['parts'])
    except (KeyError, IndexError, TypeError) as e:
        raise _AntwortUnbrauchbar(f'Keine Antwort von Gemini ({model}, {schema_name}): {finish or e}; raw: {str(antwort)[:300]}')
    if not text.strip():
        raise _AntwortUnbrauchbar(f'Leere Antwort von Gemini ({model}, {schema_name}): finishReason={finish}; raw: {str(antwort)[:300]}')
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise _AntwortUnbrauchbar(f'Gemini-Antwort kein JSON ({model}, {schema_name}, finishReason={finish}): {e}; text: {text[:300]}')


def call_gemini_with_schema(model: str, prompt: str, image_path: str, schema: Type[T], max_tokens: int = 1500,
                            temperature: float = 0.0, system: str | None = None) -> T:
    """Gemini-Aufruf mit Bild und erzwungenem Schema plus Pydantic-Validierung (ein Retry mit Hinweis)."""
    from pdf_processor import _resize_image_for_model  # lazy: pdf_processor importiert den Orchestrator
    img_b64 = _resize_image_for_model(image_path)
    return _mit_validierung(model, prompt, img_b64, schema, max_tokens, temperature, system)


def call_gemini_text_with_schema(model: str, prompt: str, schema: Type[T], max_tokens: int = 2000,
                                 temperature: float = 0.0, system: str | None = None) -> T:
    """Textaufruf ohne Bild (Quickinfo-Werkzeug)."""
    return _mit_validierung(model, prompt, None, schema, max_tokens, temperature, system)


def _max_len(schema: Type[BaseModel], feld) -> str:
    try:
        info = schema.model_fields[str(feld)]
        for m in info.metadata:
            if getattr(m, 'max_length', None):
                return str(m.max_length)
    except Exception:
        pass
    return '?'


def _mit_validierung(model, prompt, img_b64, schema, max_tokens, temperature, system):
    schema_dict = schema.model_json_schema()
    schema_name = schema.__name__
    raw = _invoke_gemini(model, prompt, img_b64, schema_name, schema_dict, max_tokens, temperature, system)
    try:
        return schema.model_validate(raw)
    except ValidationError as e:
        log.warning('Schema-Verstoß (Gemini) bei %s: %s — Retry mit Hinweis.', schema_name, e)
        # Gemini erzwingt keine Zeichengrenzen im Schema (minLength/maxLength werden nicht
        # übertragen). Bei Überlänge bekommt das Modell eine gezielte Kürzungsanweisung,
        # statt nur der Fehlermeldung.
        zu_lang = [err.get('loc', ('?',))[0] for err in e.errors() if err.get('type') == 'string_too_long']
        if zu_lang:
            grenzen = ', '.join(f'{feld} höchstens {_max_len(schema, feld)} Zeichen' for feld in zu_lang)
            hinweis = (f'Die vorherige Antwort war zu lang ({grenzen}). Kürze die betroffenen Felder: '
                       'behalte alle belegten Kernfakten, streiche Nebendetails und Wiederholungen, '
                       'füge nichts Neues hinzu.')
        else:
            hinweis = (f'Die vorherige Antwort verletzte das Schema: {e}\n'
                       'Liefere die Antwort exakt nach Schema, jedes Pflichtfeld ausgefüllt.')
        retry_prompt = prompt + '\n\n--- KORREKTUR ---\n' + hinweis
        retry_raw = _invoke_gemini(model, retry_prompt, img_b64, schema_name, schema_dict, max_tokens, temperature, system)
        try:
            return schema.model_validate(retry_raw)
        except ValidationError as e2:
            raise GeminiCallError(f'Schema verletzt auch nach Retry für {schema_name}: {e2}; raw: {str(retry_raw)[:300]}') from e2
