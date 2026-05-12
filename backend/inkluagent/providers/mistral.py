"""Mistral-Provider. Nutzt direkten HTTP-Call (gleiches Pattern wie
pipelines/v4/mistral_client.py), aber Multi-Turn + Multi-Image fuer
Chat-Konversationen.

Modelle:
- pixtral-large-latest fuer Vision-Calls (mit Bildern)
- mistral-medium-latest fuer reine Text-Calls (Smalltalk, Intent)
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import httpx

from .base import LLMProvider


log = logging.getLogger(__name__)

_MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
_HTTP_TIMEOUT = 90.0

_DEFAULT_MODEL_VISION = os.environ.get("INKLUAGENT_MODEL_VISION", "pixtral-large-latest")
_DEFAULT_MODEL_TEXT = os.environ.get("INKLUAGENT_MODEL_TEXT", "mistral-medium-latest")


class MistralProviderError(Exception):
    pass


class MistralProvider(LLMProvider):
    def chat(
        self,
        messages: list[dict],
        images: Optional[list[bytes]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise MistralProviderError("MISTRAL_API_KEY env-Variable nicht gesetzt.")

        chosen_model = model or (_DEFAULT_MODEL_VISION if images else _DEFAULT_MODEL_TEXT)

        payload_messages = self._build_payload_messages(messages, images)

        payload = {
            "model": chosen_model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = httpx.post(
                _MISTRAL_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=_HTTP_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise MistralProviderError(f"HTTP-Fehler bei Mistral-Call: {e}") from e

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise MistralProviderError(
                f"Unerwartetes Antwort-Format von Mistral: {e}; raw: {response.text[:300]}"
            ) from e

    def _build_payload_messages(
        self,
        messages: list[dict],
        images: Optional[list[bytes]],
    ) -> list[dict]:
        """Baut das Mistral-API messages-Array.

        Bilder werden an die LETZTE user-Nachricht angehaengt (Mistral-
        Konvention fuer Vision-Multi-Turn).
        """
        payload = []
        last_user_idx = -1
        for i, m in enumerate(messages):
            if m["role"] == "user":
                last_user_idx = i

        for i, m in enumerate(messages):
            if m["role"] == "user" and i == last_user_idx and images:
                content = [{"type": "text", "text": m["content"]}]
                for img_bytes in images:
                    img_b64 = base64.b64encode(img_bytes).decode("ascii")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                payload.append({"role": "user", "content": content})
            else:
                payload.append({"role": m["role"], "content": m["content"]})
        return payload
