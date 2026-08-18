"""AWS-Bedrock-Client fuer Anthropic Claude — v4-Pipeline-Provider.

DSGVO-konform via Bedrock Frankfurt (eu-central-1). Kein Training auf Inputs
(vertraglich abgesichert, AWS-AVV + Anthropic-Sub-Prozessor).

Strict-JSON via Anthropic-Tool-Use:
  Anthropic akzeptiert kein 'response_format=json_schema' wie Mistral, sondern
  erzwingt strukturierten Output ueber 'tools' + 'tool_choice'. Wir definieren
  ein einziges Output-Tool, dessen input_schema das Pydantic-Schema ist, und
  zwingen das Modell mit tool_choice='tool' es zu nutzen. Resultat: garantiert
  schema-valides JSON im tool_use-Block der Antwort.

EU-Inferenz-Profil:
  Modell-IDs muessen mit 'eu.' praefigiert sein (z.B. 'eu.anthropic.claude-
  sonnet-4-6'), damit AWS das Cross-Region-EU-Profil nutzt und der Aufruf
  garantiert in EU-Regionen bleibt. Ohne Praefix wuerde das Modell zwar evtl.
  funktionieren, aber das EU-Routing waere nicht garantiert.

Retry-Strategie analog zu mistral_client: EIN Retry mit Korrektur-Hinweis bei
Pydantic-ValidationError. Anthropic ist via Tool-Use sehr robust, Schema-
Bypaesse sind extrem selten.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

# _resize_image_for_model wird lazy in den Funktionen importiert (Circular-
# Import-Vermeidung: pdf_processor importiert orchestrator).

log = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Modell-Konstanten — per ENV ueberschreibbar fuer A/B-Tests.
# Default: Sonnet fuer Vision-lastige Paesse, Haiku fuer Klassifikation.
BEDROCK_MODEL_CLASSIFY = os.environ.get('BEDROCK_MODEL_CLASSIFY', 'eu.anthropic.claude-haiku-4-5')
BEDROCK_MODEL_INVENTAR = os.environ.get('BEDROCK_MODEL_INVENTAR', 'eu.anthropic.claude-sonnet-4-6')
BEDROCK_MODEL_GENERATE = os.environ.get('BEDROCK_MODEL_GENERATE', 'eu.anthropic.claude-sonnet-4-6')
BEDROCK_MODEL_VALIDATE = os.environ.get('BEDROCK_MODEL_VALIDATE', 'eu.anthropic.claude-sonnet-4-6')

_AWS_REGION = os.environ.get('AWS_REGION', 'eu-central-1')
_HTTP_TIMEOUT = 90


class BedrockCallError(Exception):
    """Erhoben wenn Bedrock-Call fundamental schief geht (kein Retry-wuerdig)."""


_client = None


def _get_client():
    """Lazy-init des Bedrock-Runtime-Clients (boto3)."""
    global _client
    if _client is None:
        try:
            import boto3
        except ImportError as e:
            raise BedrockCallError(
                'boto3 nicht installiert — Bedrock-Pfad nicht verfuegbar.'
            ) from e
        # AWS-Credentials werden aus den Standard-ENV-Vars geholt:
        # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
        # 17.08.2026: Ohne Retry-Vorgabe gilt boto3s alter Standardmodus.
        # Am 17./20.07. bekamen zehn Bilder eines Kunden eine
        # ThrottlingException (AWS-Drosselung) als Alt-Text — eine
        # Verkehrsregelung, kein Defekt. Der adaptive Modus bremst
        # clientseitig und wiederholt gedrosselte Aufrufe mit wachsenden
        # Pausen. Kosten: keine, ein gedrosselter Aufruf wurde nicht
        # ausgefuehrt und wird nicht berechnet.
        from botocore.config import Config as _BotoConfig
        _client = boto3.client(
            'bedrock-runtime',
            region_name=_AWS_REGION,
            config=_BotoConfig(retries={'max_attempts': 8, 'mode': 'adaptive'}),
        )
    return _client


def _detect_media_type(image_path: str) -> str:
    """Leitet AWS-akzeptierten media_type aus Bild-Endung ab.

    Anthropic via Bedrock akzeptiert: image/jpeg, image/png, image/gif, image/webp.
    AVIF/HEIC/HEIF werden von _resize_image_for_model auf JPEG konvertiert,
    daher gilt fuer diese Endungen image/jpeg.
    """
    p = image_path.lower()
    if p.endswith('.png'):
        return 'image/png'
    if p.endswith('.gif'):
        return 'image/gif'
    if p.endswith('.webp'):
        return 'image/webp'
    return 'image/jpeg'  # Default: jpg/jpeg + AVIF/HEIC/HEIF (vom Resize konvertiert)


def _detect_media_type_from_bytes(image_b64: str, image_path: str) -> str:
    """Media-Type aus den ECHTEN gesendeten Bytes statt der Datei-Endung.

    Budni-Befund 05.07.2026: _resize_image_for_model re-encodiert grosse
    Bilder als PNG. Bei .webp-Quellen (>1536px) passte die aus der Endung
    abgeleitete Deklaration ('image/webp') nicht mehr zu den PNG-Bytes ->
    Bedrock ValidationException, Bild scheiterte (4 von 37 auf budni.de).
    Byte-Signatur ist die Wahrheit; die Endungs-Heuristik bleibt Fallback.
    """
    import base64
    try:
        head = base64.b64decode(image_b64[:32])
    except Exception:
        return _detect_media_type(image_path)
    if head.startswith(b'\x89PNG'):
        return 'image/png'
    if head.startswith(b'\xff\xd8'):
        return 'image/jpeg'
    if head.startswith(b'GIF8'):
        return 'image/gif'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'image/webp'
    return _detect_media_type(image_path)


def _invoke_bedrock(
    model: str,
    prompt: str,
    image_b64: str,
    image_path: str,
    schema_name: str,
    schema_dict: dict,
    max_tokens: int,
    temperature: float = 0.0,
    system: str | None = None,
) -> dict:
    """Bedrock-Invoke-Call mit Anthropic-Tool-Use fuer Strict-JSON.

    Returns:
        dict aus dem tool_use-Input — garantiert schema-konform laut Anthropic-
        Tool-Use-Vertrag.

    Raises:
        BedrockCallError bei AWS-/Auth-/Use-Case-Fehlern.
    """
    client = _get_client()
    body = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': max_tokens,
        'temperature': temperature,  # 0 = deterministisch (Default); >0 nur bei explizitem Neu-Generieren (Variation)
        'messages': [{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': _detect_media_type_from_bytes(image_b64, image_path),
                        'data': image_b64,
                    },
                },
                {'type': 'text', 'text': prompt},
            ],
        }],
        'tools': [{
            'name': schema_name,
            'description': f'Output gemaess {schema_name}-Schema',
            'input_schema': schema_dict,
        }],
        'tool_choice': {'type': 'tool', 'name': schema_name},
    }
    # System-Prompt (05.07.2026): Mission/Rolle auf System-Ebene — wirksamste
    # Position bei Anthropic-Modellen; noetig u.a. fuer die legitime Benennung
    # oeffentlicher Personen in Alternativtexten (Barrierefreiheits-Auftrag).
    if system:
        body['system'] = system

    try:
        response = client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json',
        )
    except Exception as e:
        raise BedrockCallError(
            f'Bedrock invoke_model fehlgeschlagen ({model}): {e}'
        ) from e

    try:
        payload = json.loads(response['body'].read())
    except (KeyError, ValueError) as e:
        raise BedrockCallError(f'Bedrock-Antwort nicht parsebar: {e}') from e

    # Anthropic-Antwort-Struktur: content ist Liste von Blöcken; bei tool_choice
    # ist ein Block vom Typ 'tool_use' mit dem schema-validen Input drin.
    for block in payload.get('content', []):
        if block.get('type') == 'tool_use' and block.get('name') == schema_name:
            return block.get('input', {})

    raise BedrockCallError(
        f'Kein tool_use-Block in Antwort ({schema_name}). Raw: {str(payload)[:300]}'
    )


def call_bedrock_with_schema(
    model: str,
    prompt: str,
    image_path: str,
    schema: Type[T],
    max_tokens: int = 1500,
    temperature: float = 0.0,
    system: str | None = None,
) -> T:
    """Bedrock-Call mit Anthropic-Tool-Use + Pydantic-Validierung.

    Identische Signatur zu mistral_client.call_mistral_with_schema, damit der
    Orchestrator Provider-agnostisch bleibt.
    """
    from pdf_processor import _resize_image_for_model  # lazy zur Vermeidung von Circular-Import
    img_b64 = _resize_image_for_model(image_path)
    schema_dict = schema.model_json_schema()
    schema_name = schema.__name__

    raw = _invoke_bedrock(
        model=model,
        prompt=prompt,
        image_b64=img_b64,
        image_path=image_path,
        schema_name=schema_name,
        schema_dict=schema_dict,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
    )

    try:
        return schema.model_validate(raw)
    except ValidationError as e:
        log.warning(
            'Tool-Use-Schema-Bypass für %s: %s — Retry mit Korrektur-Hinweis.',
            schema_name, e,
        )
        retry_prompt = (
            prompt
            + '\n\n--- KRITISCHE KORREKTUR ---\n'
            + f'Vorherige Antwort verletzte das Schema: {e}\n'
            + 'Liefere die Antwort exakt nach Schema, jedes Pflichtfeld ausgefuellt.'
        )
        retry_raw = _invoke_bedrock(
            model=model,
            prompt=retry_prompt,
            image_b64=img_b64,
            image_path=image_path,
            schema_name=schema_name,
            schema_dict=schema_dict,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
        try:
            return schema.model_validate(retry_raw)
        except ValidationError as e2:
            raise BedrockCallError(
                f'Schema verletzt auch nach Retry für {schema_name}: {e2}; '
                f'raw: {str(retry_raw)[:300]}'
            ) from e2
