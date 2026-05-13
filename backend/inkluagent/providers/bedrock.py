"""Bedrock-Provider für InkluAgent. Anthropic-Claude via AWS Bedrock
Frankfurt (eu-central-1). Folgt der gleichen LLMProvider-Schnittstelle
wie mistral.py — wird in chat_engine.py per INKLUAGENT_PROVIDER=bedrock
aktiviert.

Modell-Defaults (über ENV überschreibbar):
- INKLUAGENT_BEDROCK_MODEL_TEXT   → eu.anthropic.claude-sonnet-4-6
- INKLUAGENT_BEDROCK_MODEL_VISION → eu.anthropic.claude-sonnet-4-6

Für günstige Klassifikator-Calls kann der Caller `model=...` explizit
setzen (z.B. eu.anthropic.claude-haiku-4-5-20251001-v1:0).

Plus invoke_with_tools() für agentic Tool-Use-Loops (12.05.2026).
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .base import LLMProvider


log = logging.getLogger(__name__)

_DEFAULT_MODEL_TEXT = os.environ.get(
    "INKLUAGENT_BEDROCK_MODEL_TEXT",
    "eu.anthropic.claude-sonnet-4-6",
)
_DEFAULT_MODEL_VISION = os.environ.get(
    "INKLUAGENT_BEDROCK_MODEL_VISION",
    "eu.anthropic.claude-sonnet-4-6",
)

_ANTHROPIC_VERSION = "bedrock-2023-05-31"


class BedrockProviderError(Exception):
    pass


def _detect_media_type(img_bytes: bytes) -> str:
    """Sehr leichtes Format-Sniffing — JPEG/PNG/GIF/WebP."""
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if img_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class BedrockProvider(LLMProvider):
    """boto3-bedrock-runtime-Client. Konfiguriert sich aus AWS_REGION,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (gleiche ENVs wie v4-Pipeline).
    """

    def __init__(self) -> None:
        region = os.environ.get("AWS_REGION", "eu-central-1")
        try:
            self._client = boto3.client("bedrock-runtime", region_name=region)
        except (BotoCoreError, ClientError) as e:
            raise BedrockProviderError(
                f"Konnte bedrock-runtime-Client nicht initialisieren: {e}"
            ) from e

    def chat(
        self,
        messages: list[dict],
        images: Optional[list[bytes]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        chosen_model = model or (_DEFAULT_MODEL_VISION if images else _DEFAULT_MODEL_TEXT)

        system_text, anthropic_messages = self._build_anthropic_messages(messages, images)

        body: dict = {
            "anthropic_version": _ANTHROPIC_VERSION,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
        }
        if system_text:
            body["system"] = system_text

        try:
            response = self._client.invoke_model(
                modelId=chosen_model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except (BotoCoreError, ClientError) as e:
            raise BedrockProviderError(
                f"Bedrock invoke_model fehlgeschlagen ({chosen_model}): {e}"
            ) from e

        try:
            payload = json.loads(response["body"].read())
            content_blocks = payload.get("content", [])
            text_parts = [
                b.get("text", "") for b in content_blocks if b.get("type") == "text"
            ]
            return "".join(text_parts)
        except (KeyError, ValueError, TypeError) as e:
            raise BedrockProviderError(
                f"Unerwartetes Bedrock-Antwortformat: {e}"
            ) from e

    def invoke_with_tools(
        self,
        anthropic_messages: list[dict],
        tools: list[dict],
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Direct Bedrock invoke mit Tool-Use. Gibt das rohe Anthropic-Payload zurück
        (mit content-Blocks und stop_reason — der Caller muss tool_use-Blocks
        auswerten und in einem Loop wieder reinschicken).

        anthropic_messages: Liste im Anthropic-Format ({"role": ..., "content": ...}).
            Bilder + Tool-Results müssen vom Caller schon vor-formatiert sein.
        tools: Liste von Tool-Definitions im Anthropic-Format
            (name, description, input_schema).
        """
        chosen_model = model or _DEFAULT_MODEL_TEXT
        body: dict = {
            "anthropic_version": _ANTHROPIC_VERSION,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "tools": tools,
            "temperature": temperature,
        }
        if system:
            body["system"] = system

        try:
            response = self._client.invoke_model(
                modelId=chosen_model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except (BotoCoreError, ClientError) as e:
            raise BedrockProviderError(
                f"Bedrock invoke_with_tools fehlgeschlagen ({chosen_model}): {e}"
            ) from e

        try:
            return json.loads(response["body"].read())
        except (ValueError, TypeError) as e:
            raise BedrockProviderError(
                f"Tool-Use Response nicht JSON: {e}"
            ) from e

    def _build_anthropic_messages(
        self,
        messages: list[dict],
        images: Optional[list[bytes]],
    ) -> tuple[str, list[dict]]:
        """Trennt System-Prompts ab (Anthropic-API hat eigenen system-Parameter)
        und konvertiert restliche messages ins Anthropic-Format. Bilder werden
        an die LETZTE user-Nachricht als image-content-Blocks angehängt.
        """
        system_chunks: list[str] = []
        non_system: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if content:
                    system_chunks.append(content)
            else:
                non_system.append(m)

        last_user_idx = -1
        for i, m in enumerate(non_system):
            if m.get("role") == "user":
                last_user_idx = i

        anthropic_messages: list[dict] = []
        for i, m in enumerate(non_system):
            role = m.get("role", "user")
            text = m.get("content", "")

            if role == "user" and i == last_user_idx and images:
                content_blocks: list[dict] = []
                for img_bytes in images:
                    media_type = _detect_media_type(img_bytes)
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(img_bytes).decode("ascii"),
                        },
                    })
                content_blocks.append({"type": "text", "text": text})
                anthropic_messages.append({"role": "user", "content": content_blocks})
            else:
                anthropic_messages.append({"role": role, "content": text})

        system_text = "\n\n".join(system_chunks).strip()
        return system_text, anthropic_messages
