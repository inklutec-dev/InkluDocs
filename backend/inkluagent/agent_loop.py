"""Agentic Tool-Loop für InkluAgent (Sonnet 4.6 via Bedrock, 12.05.2026).

Aktiviert wenn INKLUAGENT_PROVIDER=bedrock und INKLUAGENT_AGENTIC nicht 'false'.

Der Loop:
  1. Baut messages aus storage-History + neuer user-Nachricht
  2. Ruft provider.invoke_with_tools(messages, tools, system)
  3. Inspiziert response.content auf tool_use-Blocks
     - Wenn nur text: fertig, return
     - Wenn tool_use: ToolExecutor ausführen, tool_result anfügen, goto 2
  4. Max _MAX_ITERATIONS Tool-Roundtrips pro User-Turn

Spezialfall view_image: Tool-Result kommt zurück mit zusätzlichem
`image_bytes`-Key. Wir hängen das Bild als image-content-Block direkt
an die nächste user-Nachricht an, damit Claude es in seinem nächsten
Schritt sieht.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

from . import storage
from .providers.bedrock import BedrockProvider, BedrockProviderError
from .tools.definitions import TOOL_DEFINITIONS, ToolExecutor
from .prompts.system_agent import SYSTEM_AGENT
from .adapters.inkludocs import build_project_summary

log = logging.getLogger(__name__)

_MAX_ITERATIONS = 6
_HISTORY_TURNS = 20  # weniger als 30, weil Tool-Calls + Images Context-Hungry sind
_DEFAULT_MAX_TOKENS = 4096


def _detect_media_type(img_bytes: bytes) -> str:
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _build_initial_messages(
    project_id: int,
    user_message: str,
    project: dict,
) -> list[dict]:
    """Baut die Anthropic-Format messages-Liste aus storage-History + user_message.

    History-Format aus storage: [{"role": "user|assistant", "content": "..."}].
    Tool-Calls werden NICHT in History gespeichert (vereinfacht den Reload).
    """
    history = storage.get_history(project_id, limit=_HISTORY_TURNS) or []
    messages: list[dict] = []

    # Projekt-Summary als erste user-Nachricht (Context-Vorgabe)
    summary = build_project_summary(project)
    messages.append({
        "role": "user",
        "content": f"[Projekt-Kontext]\n{summary}",
    })
    messages.append({
        "role": "assistant",
        "content": "Verstanden, Projekt-Kontext angenommen. Was kann ich für dich tun?",
    })

    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


def _append_tool_results(
    messages: list[dict],
    assistant_content_blocks: list[dict],
    tool_results: list[dict],
    pending_image_bytes: list[bytes],
) -> None:
    """Anthropic-Pattern: assistant-Turn mit tool_use Blocks UND
    nächster user-Turn mit tool_result Blocks (+ optional image-Blocks).
    """
    messages.append({"role": "assistant", "content": assistant_content_blocks})

    next_user_content: list[dict] = list(tool_results)  # tool_result blocks zuerst
    for img_bytes in pending_image_bytes:
        next_user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _detect_media_type(img_bytes),
                "data": base64.b64encode(img_bytes).decode("ascii"),
            },
        })
    if not next_user_content:
        return
    # Ein leerer text-block am Ende ist optional, hilft aber bei manchen Klienten
    messages.append({"role": "user", "content": next_user_content})


def run_agent(
    project_id: int,
    user_id: int,
    user_message: str,
    project: dict,
    provider: BedrockProvider,
) -> dict[str, Any]:
    """Führt den agentic Loop aus. Returns:
        {"reply": str, "intent": "agentic", "image_refs": [...] or None,
         "actions": [{"tool": "...", "args": {...}, "ok": bool}]}.

    Hinweis zur Persistenz: User-Message + Bot-Reply werden NICHT hier
    gespeichert — das übernimmt main.py:_chat_endpoint vor/nach diesem
    Aufruf zentral.
    """
    executor = ToolExecutor(project_id=project_id, user_id=user_id)
    messages = _build_initial_messages(project_id, user_message, project)

    actions_log: list[dict] = []
    image_refs_seen: set[int] = set()
    last_reply_text = ""

    for iteration in range(_MAX_ITERATIONS):
        try:
            payload = provider.invoke_with_tools(
                anthropic_messages=messages,
                tools=TOOL_DEFINITIONS,
                system=SYSTEM_AGENT,
                max_tokens=_DEFAULT_MAX_TOKENS,
                temperature=0.3,
            )
        except BedrockProviderError as e:
            log.exception("Bedrock-Call im agentic Loop fehlgeschlagen")
            return {
                "reply": f"Entschuldigung, Verbindung zur KI gerade unterbrochen ({e}).",
                "intent": "error",
                "image_refs": None,
                "actions": actions_log,
            }

        content_blocks = payload.get("content", []) or []
        stop_reason = payload.get("stop_reason", "")

        # Text-Anteile sammeln
        text_chunks = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        last_reply_text = "".join(text_chunks).strip()

        # Tool-Use-Blocks sammeln
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_use_blocks:
            # Reines Text-Response, Loop fertig
            log.info("Agent fertig nach %d Iterationen, stop=%s", iteration, stop_reason)
            break

        # Tool-Calls ausführen
        tool_results: list[dict] = []
        pending_images: list[bytes] = []
        for tu in tool_use_blocks:
            name = tu.get("name", "")
            args = tu.get("input", {}) or {}
            tu_id = tu.get("id", "")

            log.info("Agent ruft Tool %s mit args=%s", name, args)
            result = executor.execute(name, args)
            actions_log.append({
                "tool": name,
                "args": args,
                "ok": bool(result.get("ok")),
                "error": result.get("error", "") if not result.get("ok") else "",
            })

            # image_bytes-Sonderfall (view_image)
            img_bytes = result.pop("image_bytes", None) if isinstance(result, dict) else None
            if img_bytes:
                pending_images.append(img_bytes)
                if isinstance(args, dict) and "image_id" in args:
                    image_refs_seen.add(int(args["image_id"]))

            # Andere referenzierte image_ids tracken
            if isinstance(args, dict) and "image_id" in args:
                try:
                    image_refs_seen.add(int(args["image_id"]))
                except (TypeError, ValueError):
                    pass

            # Serialisiere result als kompaktes JSON für tool_result
            try:
                content_text = json.dumps(result, ensure_ascii=False)[:8000]
            except (TypeError, ValueError):
                content_text = str(result)[:8000]

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu_id,
                "content": content_text,
            })

        _append_tool_results(messages, content_blocks, tool_results, pending_images)

        # Wenn stop_reason explizit "end_turn" trotz Tool-Use: dann ist Claude
        # eigentlich fertig, sollte nicht passieren — Schutz gegen Endlos-Loop:
        if stop_reason == "end_turn":
            log.warning("stop_reason=end_turn aber tool_use vorhanden — breche ab")
            break

    else:
        # Loop voll durchgelaufen ohne textuelle Antwort
        log.warning("Agent-Loop max %d Iterationen ohne text-Antwort", _MAX_ITERATIONS)
        if not last_reply_text:
            last_reply_text = (
                "Ich habe mehrere Werkzeuge benutzt aber komme zu keinem klaren Ergebnis. "
                "Kannst du deine Frage etwas eingrenzen?"
            )

    final_reply = last_reply_text or (
        "Habe gerade keine konkrete Antwort dafür. Frag mich nochmal anders?"
    )

    image_refs_list = sorted(image_refs_seen) if image_refs_seen else None
    return {
        "reply": final_reply,
        "intent": "agentic",
        "image_refs": image_refs_list,
        "actions": actions_log,
    }
