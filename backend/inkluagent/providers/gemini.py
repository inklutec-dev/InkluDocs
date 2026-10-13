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

import ki_kosten  # KI-Kosten je Aufruf (05.10.2026)

from .base import LLMProvider
from .bedrock import BedrockProviderError, _detect_media_type

log = logging.getLogger(__name__)

_DEFAULT_MODEL_TEXT = os.environ.get("INKLUAGENT_GEMINI_MODEL_TEXT", "gemini-3.1-pro-preview")
_DEFAULT_MODEL_VISION = os.environ.get("INKLUAGENT_GEMINI_MODEL_VISION", "gemini-3.1-pro-preview")
# Denkstufe fuer den Chat (10.09.2026): leer = Vorgabe des Modells (Gemini 3: mittel). Werte low|medium|high.
# Denk-Tokens werden als Ausgabe abgerechnet — fuer Kostenmessungen umschaltbar.
_THINKING_LEVEL = os.environ.get("INKLUAGENT_GEMINI_THINKING", "").strip().lower()
# Temperatur (10.09.2026): Google empfiehlt fuer Gemini 3 ausdruecklich die Vorgabe 1.0 — niedrigere Werte
# fuehren zu Schleifen und Entgleisungen (auf Staging gesehen: 6.000 Tokens erfundene Metadaten bei 0.3).
# Der Aufrufer-Wert (agent_loop: 0.3) wird deshalb ignoriert; Override nur per Umgebung.
_TEMPERATUR = float(os.environ.get("INKLUAGENT_GEMINI_TEMPERATURE", "1.0") or "1.0")
_HTTP_TIMEOUT = 180
_VERSUCHE = 3


class GeminiProviderError(BedrockProviderError):
    pass


def _endpunkt(model: str) -> str:
    from pipelines.v4 import gemini_auth
    return gemini_auth.endpunkt(model)


def _entgleist(text: str, finish: str) -> bool:
    """Erkennt Schleifen/Entgleisungen: Ausgabelimit erreicht, viele Zeilen mit gleichem Anfang,
    oder eine Nicht-Antwort aus ein, zwei Woertern (auf Staging: 'proposal.')."""
    zeilen = [z.strip() for z in text.splitlines() if z.strip()]
    if finish == "MAX_TOKENS" and len(text) > 3000:
        return True
    if len(zeilen) >= 16:
        anfaenge: dict[str, int] = {}
        for z in zeilen:
            k = z[:14]
            anfaenge[k] = anfaenge.get(k, 0) + 1
        if max(anfaenge.values()) >= 12:
            return True
    woerter = text.split()
    if 0 < len(woerter) <= 2 and len(text) < 25:
        return True
    return False


def _gekuerzt(text: str) -> str:
    """Schneidet eine entgleiste Antwort vor der ersten Wiederholung ab."""
    zeilen = text.splitlines()
    gesehen: dict[str, int] = {}
    for i, z in enumerate(zeilen):
        k = z.strip()[:14]
        if not k:
            continue
        gesehen[k] = gesehen.get(k, 0) + 1
        if gesehen[k] >= 6:
            return "\n".join(zeilen[:i]).strip()[:1500] or "Entschuldigung, die Antwort ist nicht sauber zustande gekommen. Bitte die Frage noch einmal stellen."
    return text[:1500]


class GeminiAntwortGesperrt(GeminiProviderError):
    """Gemini hat die Anfrage beantwortet, aber ohne Kandidaten (promptFeedback.blockReason): endgueltig, kein zweiter Versuch
    (Pruefung 13.10.2026, K10 — vorher wurde daraus im Stream eine leere, als normale Antwort gespeicherte Nachricht)."""


class GeminiProvider(LLMProvider):
    # Livestream (Arbeitsfenster, 12.10.2026): invoke_with_tools nimmt on_text und liefert den Antworttext Stueck fuer Stueck
    kann_streamen = True

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
                    antwort = json.load(r)
                # KI-Kosten (05.10.2026): Chat-Aufrufe zaehlen als „chatbot“, sofern der Aufrufer nichts anderes setzt.
                ki_kosten.erfasse_gemini(model, antwort, schritt="chat", standard_zweck="chatbot")
                return antwort
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

    def _aufruf_strom(self, model: str, body: dict, on_text) -> dict:
        """Wie _aufruf, aber ueber streamGenerateContent (SSE): Textteile gehen sofort an on_text. Ergebnis wie generateContent —
        zusammenhaengender Text als ein Teil, Funktionsaufrufe unveraendert (mit Gedankensignatur), finishReason, usageMetadata.
        Kosten werden auch fuer einen abgebrochenen Stream erfasst, soweit Gemini sie schon gemeldet hat (Pruefung 13.10.2026,
        S12: sonst zaehlte der Tages-Kostendeckel zu wenig). Ein Fehlerobjekt im Stream ist ein Fehler (der Aufrufer nimmt den
        Weg ohne Stream); eine Antwort ganz ohne Kandidaten mit blockReason ist endgueltig (GeminiAntwortGesperrt)."""
        from pipelines.v4 import gemini_auth
        url = _endpunkt(model).replace(":generateContent", ":streamGenerateContent")
        url += ("&" if "?" in url else "?") + "alt=sse"
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=gemini_auth.kopfzeilen())
        parts: list[dict] = []
        finish = ""
        usage: dict = {}
        feedback = None
        kandidaten = False
        fertig = False
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
                for roh in r:
                    zeile = roh.decode("utf-8", "ignore").strip()
                    if not zeile.startswith("data:"):
                        continue
                    try:
                        stueck = json.loads(zeile[5:].strip())
                    except ValueError:
                        continue
                    if stueck.get("error"):
                        raise GeminiProviderError(f"Gemini-Stream meldet einen Fehler ({model}): {str(stueck['error'])[:300]}")
                    usage = stueck.get("usageMetadata") or usage
                    feedback = stueck.get("promptFeedback") or feedback
                    kandidaten = kandidaten or bool(stueck.get("candidates"))
                    self._stueck_aufnehmen(stueck, parts, on_text)
                    finish = ((stueck.get("candidates") or [{}])[0] or {}).get("finishReason") or finish
            fertig = True
        finally:
            if not fertig and usage:   # abgebrochen: bezahlt ist, was Gemini schon gemeldet hat
                ki_kosten.erfasse_gemini(model, {"usageMetadata": usage}, schritt="chat", erfolg=False, fehler="stream_abgebrochen",
                                         standard_zweck="chatbot")
        if not kandidaten and (feedback or {}).get("blockReason"):
            ki_kosten.erfasse_gemini(model, {"usageMetadata": usage}, schritt="chat", standard_zweck="chatbot")
            raise GeminiAntwortGesperrt(f"Keine Antwort von Gemini: {feedback.get('blockReason')}")
        antwort: dict = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}], "usageMetadata": usage}
        if feedback:
            antwort["promptFeedback"] = feedback
        ki_kosten.erfasse_gemini(model, antwort, schritt="chat", standard_zweck="chatbot")
        return antwort

    @staticmethod
    def _stueck_aufnehmen(stueck: dict, parts: list[dict], on_text) -> None:
        """Ein SSE-Stueck in die Teile der Antwort: Text zusammenhaengend (und sofort an on_text), Gedankensignaturen am Text,
        Funktionsaufrufe und alles andere unveraendert. Gedanken (thought=true) werden weder gesendet noch Text."""
        kandidat = (stueck.get("candidates") or [{}])[0] or {}
        for p in ((kandidat.get("content") or {}).get("parts") or []):
            letzter_text = bool(parts) and "text" in parts[-1] and not parts[-1].get("thought") and "functionCall" not in parts[-1]
            if p.get("text") and not p.get("thought") and "functionCall" not in p:
                if letzter_text:
                    parts[-1]["text"] += p["text"]
                    if p.get("thoughtSignature"):
                        parts[-1]["thoughtSignature"] = p["thoughtSignature"]
                else:
                    parts.append(dict(p))
                on_text(p["text"])
            elif p.get("thoughtSignature") and not p.get("text") and "functionCall" not in p and letzter_text:
                parts[-1]["thoughtSignature"] = p["thoughtSignature"]   # Signatur im letzten Stueck des Textes
            elif p:
                parts.append(dict(p))

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
                bild = {"inlineData": {"mimeType": src.get("media_type", "image/jpeg"), "data": src.get("data", "")}}
                fr = next((x for x in reversed(teile) if "functionResponse" in x), None)
                if fr is not None:
                    # Bild gehoert zum Werkzeugergebnis (view_image / view_field): als multimodale Function-Response
                    # (functionResponse.parts) mitgeben, nicht als lose Nutzereingabe — sonst echot Gemini 3
                    # Bruchstuecke des Werkzeugergebnisses in den Antworttext (Staging 10.09.2026).
                    fr["functionResponse"].setdefault("parts", []).append(bild)
                else:
                    teile.append(bild)
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
                      "generationConfig": {"temperature": _TEMPERATUR if temperature > 0 else 0.0, "maxOutputTokens": max_tokens + 2048}}
        if system_chunks:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
        antwort = self._aufruf(chosen, body)
        try:
            return "".join(p.get("text", "") for p in antwort["candidates"][0]["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError, TypeError) as e:
            raise GeminiProviderError(f"Unerwartetes Gemini-Antwortformat: {e}; raw: {str(antwort)[:300]}") from e

    def invoke_with_tools(self, anthropic_messages: list[dict], tools: list[dict], system: Optional[str] = None,
                          model: Optional[str] = None, max_tokens: int = 4096, temperature: float = 0.3,
                          on_text=None) -> dict[str, Any]:
        chosen = model or _DEFAULT_MODEL_TEXT
        body: dict = {"contents": self._contents(anthropic_messages),
                      "generationConfig": {"temperature": _TEMPERATUR, "maxOutputTokens": max_tokens + 2048}}
        if _THINKING_LEVEL in ("low", "medium", "high"):
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": _THINKING_LEVEL.upper()}
        werkzeuge = self._werkzeuge(tools)
        if werkzeuge:
            body["tools"] = werkzeuge
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        debug = os.getenv("DEBUG_GEN_RAW", "false").lower() == "true"
        for versuch in range(2):
            antwort = None
            if on_text:
                try:
                    antwort = self._aufruf_strom(chosen, body, on_text)
                except GeminiAntwortGesperrt:
                    on_text(None)
                    raise
                except Exception as e:  # noqa: BLE001
                    # Bricht der Stream ab (Netz, Fehlerobjekt, kaputte Daten), verwirft die Seite den Entwurf, und derselbe Aufruf
                    # laeuft ohne Stream — auch wenn schon Text kam: der Nutzer bekommt eine vollstaendige Antwort statt einer
                    # Fehlermeldung, nach der er ohnehin neu fragen muesste. Die Kosten des abgebrochenen Teils sind erfasst.
                    log.warning("Gemini-Stream fehlgeschlagen (%s) — ohne Stream", e)
                    on_text(None)
                    antwort = None
            if antwort is None:
                antwort = self._aufruf(chosen, body)
            try:
                parts = antwort["candidates"][0]["content"].get("parts", [])
                finish = antwort["candidates"][0].get("finishReason", "")
            except (KeyError, IndexError, TypeError) as e:
                grund = (antwort.get("promptFeedback") or {}).get("blockReason")
                raise GeminiProviderError(f"Keine Antwort von Gemini: {grund or e}; raw: {str(antwort)[:300]}") from e
            if debug:
                u = antwort.get("usageMetadata", {}) or {}
                arten = ["call" if "functionCall" in p else ("denk" if p.get("thought") else ("text" if p.get("text") else "?")) for p in parts]
                print(f"[GEMINI-USAGE] model={chosen} schema=agent in={u.get('promptTokenCount', '?')} out={u.get('candidatesTokenCount', '?')} "
                      f"denk={u.get('thoughtsTokenCount', 0)} finish={finish} parts={','.join(arten)}", flush=True)
            text_roh = "".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))
            hat_call = any("functionCall" in p for p in parts)
            if not hat_call and _entgleist(text_roh, finish):
                log.warning("Gemini-Antwort entgleist (Versuch %d, finish=%s, %d Zeichen) — %s", versuch + 1, finish, len(text_roh),
                            "wiederhole" if versuch == 0 else "kuerze")
                if versuch == 0:
                    if on_text:
                        on_text(None)   # der Entwurf war entgleist: verwerfen, der zweite Versuch streamt neu
                    continue
                parts = [{"text": _gekuerzt(text_roh)}]
            break
        content: list[dict] = []
        for p in parts:
            if p.get("thought"):
                continue  # Gedanken-Zusammenfassungen (thought=true) sind kein Antworttext (10.09.2026: landeten sonst im Chat)
            if "functionCall" in p:
                self._zaehler += 1
                fc = p["functionCall"]
                content.append({"type": "tool_use", "id": f"gemini_call_{self._zaehler}", "name": fc.get("name", ""),
                                "input": fc.get("args", {}) or {}, "_gemini_part": p})
            elif p.get("text"):
                content.append({"type": "text", "text": p["text"]})
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else ("max_tokens" if finish == "MAX_TOKENS" else "end_turn")
        return {"content": content, "stop_reason": stop, "model": chosen}
