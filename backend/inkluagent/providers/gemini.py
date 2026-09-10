"""Gemini-Provider für InkluAgent (Chat und Werkzeugaufrufe über die Gemini-API).

Eingeführt am 08.09.2026 mit der Umstellung der Bildpipeline auf Gemini. Der Chatbot spricht intern das
Anthropic-Nachrichtenformat (Textblöcke, Bildblöcke, tool_use und tool_result); dieser Provider übersetzt
in beide Richtungen, damit agent_loop.py und chat_engine.py unverändert bleiben:
- invoke_with_tools() liefert ein Anthropic-förmiges Payload mit content-Blöcken und stop_reason.
- Gemini-Funktionsaufrufe werden zu tool_use-Blöcken (mit erzeugter id); der rohe Gemini-Part wird im
  Block unter "_gemini_part" mitgeführt, weil Gemini 3 bei Folgeaufrufen die Gedankensignatur des
  ursprünglichen Funktionsaufrufs zurück erwartet.
Fehler erben von BedrockProviderError, damit die bestehenden except-Zweige greifen.
Schlüssel: GEMINI_API_KEY; Modelle: INKLUAGENT_GEMINI_MODEL_TEXT / _VISION (Vorgabe gemini-3.1-pro-preview).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from pipelines.v4.gemini_client import _schema_fuer_gemini

from .base import LLMProvider
from .bedrock import BedrockProviderError, _detect_media_type

log = logging.getLogger(__name__)

_DEFAULT_MODEL_TEXT = os.environ.get("INKLUAGENT_GEMINI_MODEL_TEXT", "gemini-3.1-pro-preview")
_DEFAULT_MODEL_VISION = os.environ.get("INKLUAGENT_GEMINI_MODEL_VISION", "gemini-3.1-pro-preview")
# Denkstufe fuer den Chat (10.09.2026): leer = Vorgabe des Modells (Gemini 3: mittel). Werte low|medium|high.
# Denk-Tokens werden als Ausgabe abgerechnet — fuer Kostenmessungen umschaltbar.
_THINKING_LEVEL = os.environ.get("INKLUAGENT_GEMINI_THINKING", "").strip().lower()
_HTTP_TIMEOUT = 180
_VERSUCHE = 3


class GeminiProviderError(BedrockProviderError):
    pass


def _endpunkt(model: str) -> str:
    from pipelines.v4 import gemini_auth
    return gemini_auth.endpunkt(model)


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        from pipelines.v4 import gemini_auth
        try:
            gemini_auth.kopfzeilen()  # prüft Schlüssel bzw. Dienstkonto beim Start
        except Exception as e:
            raise GeminiProviderError(f"Gemini-Zugang: {e}") from e
        self._zaehler = 0

    # ---------------------------------------------------------------- HTTP
    def _aufruf(self, model: str, body: dict) -> dict:
        daten = json.dumps(body).encode("utf-8")
        letzter: Optional[Exception] = None
        for versuch in range(_VERSUCHE):
            from pipelines.v4 import gemini_auth
            req = urllib.request.Request(_endpunkt(model), data=daten, headers=gemini_auth.kopfzeilen())
            try:
                with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                text = ""
                try:
                    text = e.read().decode("utf-8", "ignore")[:400]
                except Exception:
                    pass
                letzter = GeminiProviderError(f"Gemini HTTP {e.code} ({model}): {text}")
                if e.code in (429, 500, 502, 503, 504) and versuch < _VERSUCHE - 1:
                    time.sleep(4 * (versuch + 1))
                    continue
                raise letzter from e
            except Exception as e:
                letzter = GeminiProviderError(f"Gemini-Aufruf fehlgeschlagen ({model}): {e}")
                if versuch < _VERSUCHE - 1:
                    time.sleep(4 * (versuch + 1))
                    continue
                raise letzter from e
        raise letzter or GeminiProviderError("Gemini-Aufruf fehlgeschlagen")

    # ------------------------------------------------------------ Umsetzung
    def _teile_aus_content(self, content: Any, namen: dict[str, str]) -> list[dict]:
        """Anthropic-content (str oder Blockliste) -> Gemini-Parts."""
        if isinstance(content, str):
            return [{"text": content}] if content else []
        teile: list[dict] = []
        for b in content or []:
            typ = b.get("type")
            if typ == "text":
                if b.get("text"):
                    teile.append({"text": b["text"]})
            elif typ == "image":
                src = b.get("source", {})
                teile.append({"inlineData": {"mimeType": src.get("media_type", "image/jpeg"), "data": src.get("data", "")}})
            elif typ == "tool_use":
                namen[b.get("id", "")] = b.get("name", "")
                roh = b.get("_gemini_part")
                if roh:
                    teile.append(roh)
                else:
                    teile.append({"functionCall": {"name": b.get("name", ""), "args": b.get("input", {}) or {}}})
            elif typ == "tool_result":
                inhalt = b.get("content", "")
                if isinstance(inhalt, list):
                    inhalt = "\n".join(x.get("text", "") for x in inhalt if isinstance(x, dict))
                try:
                    antwort = json.loads(inhalt) if isinstance(inhalt, str) and inhalt.strip().startswith(("{", "[")) else None
                except Exception:
                    antwort = None
                if not isinstance(antwort, dict):
                    antwort = {"result": inhalt}
                teile.append({"functionResponse": {"name": namen.get(b.get("tool_use_id", ""), "werkzeug"), "response": antwort}})
        return teile

    def _contents(self, anthropic_messages: list[dict]) -> list[dict]:
        namen: dict[str, str] = {}
        contents: list[dict] = []
        for m in anthropic_messages:
            rolle = "model" if m.get("role") == "assistant" else "user"
            teile = self._teile_aus_content(m.get("content", ""), namen)
            if not teile:
                continue
            if contents and contents[-1]["role"] == rolle:
                contents[-1]["parts"].extend(teile)  # Gemini erwartet abwechselnde Rollen
            else:
                contents.append({"role": rolle, "parts": teile})
        return contents

    @staticmethod
    def _werkzeuge(tools: list[dict]) -> list[dict]:
        decl = []
        for t in tools or []:
            params = _schema_fuer_gemini(t.get("input_schema") or {"type": "object", "properties": {}})
            params.pop("description", None)
            decl.append({"name": t["name"], "description": t.get("description", ""), "parameters": params})
        return [{"functionDeclarations": decl}] if decl else []

    # --------------------------------------------------------------- API
    def chat(self, messages: list[dict], images: Optional[list[bytes]] = None, model: Optional[str] = None,
             max_tokens: int = 1024, temperature: float = 0.4) -> str:
        chosen = model or (_DEFAULT_MODEL_VISION if images else _DEFAULT_MODEL_TEXT)
        system_chunks = [m.get("content", "") for m in messages if m.get("role") == "system" and m.get("content")]
        rest = [m for m in messages if m.get("role") != "system"]
        if images and rest:
            letzter_user = max((i for i, m in enumerate(rest) if m.get("role") == "user"), default=-1)
            if letzter_user >= 0:
                m = rest[letzter_user]
                bloecke = [{"type": "image", "source": {"type": "base64", "media_type": _detect_media_type(b),
                                                         "data": base64.b64encode(b).decode("ascii")}} for b in images]
                bloecke.append({"type": "text", "text": m.get("content", "")})
                rest[letzter_user] = {"role": "user", "content": bloecke}
        body: dict = {"contents": self._contents(rest),
                      "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens + 2048}}
        if system_chunks:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
        antwort = self._aufruf(chosen, body)
        try:
            return "".join(p.get("text", "") for p in antwort["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError) as e:
            raise GeminiProviderError(f"Unerwartetes Gemini-Antwortformat: {e}; raw: {str(antwort)[:300]}") from e

    def invoke_with_tools(self, anthropic_messages: list[dict], tools: list[dict], system: Optional[str] = None,
                          model: Optional[str] = None, max_tokens: int = 4096, temperature: float = 0.3) -> dict[str, Any]:
        chosen = model or _DEFAULT_MODEL_TEXT
        body: dict = {"contents": self._contents(anthropic_messages),
                      "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens + 2048}}
        if _THINKING_LEVEL in ("low", "medium", "high"):
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": _THINKING_LEVEL.upper()}
        werkzeuge = self._werkzeuge(tools)
        if werkzeuge:
            body["tools"] = werkzeuge
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        antwort = self._aufruf(chosen, body)
        if os.getenv("DEBUG_GEN_RAW", "false").lower() == "true":
            u = antwort.get("usageMetadata", {}) or {}
            print(f"[GEMINI-USAGE] model={chosen} schema=agent in={u.get('promptTokenCount', '?')} out={u.get('candidatesTokenCount', '?')} "
                  f"denk={u.get('thoughtsTokenCount', 0)}", flush=True)
        try:
            parts = antwort["candidates"][0]["content"].get("parts", [])
            finish = antwort["candidates"][0].get("finishReason", "")
        except (KeyError, IndexError, TypeError) as e:
            grund = (antwort.get("promptFeedback") or {}).get("blockReason")
            raise GeminiProviderError(f"Keine Antwort von Gemini: {grund or e}; raw: {str(antwort)[:300]}") from e
        content: list[dict] = []
        for p in parts:
            if "functionCall" in p:
                self._zaehler += 1
                fc = p["functionCall"]
                content.append({"type": "tool_use", "id": f"gemini_call_{self._zaehler}", "name": fc.get("name", ""),
                                "input": fc.get("args", {}) or {}, "_gemini_part": p})
            elif p.get("text"):
                content.append({"type": "text", "text": p["text"]})
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else ("max_tokens" if finish == "MAX_TOKENS" else "end_turn")
        return {"content": content, "stop_reason": stop, "model": chosen}
