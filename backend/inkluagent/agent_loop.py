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
import sqlite3
from typing import Any, Optional

from . import storage
from .providers.bedrock import BedrockProvider, BedrockProviderError
from .tools.definitions import TOOL_DEFINITIONS, ToolExecutor
from .tools.definitions_formular import TOOL_DEFINITIONS_FORMULAR, ToolExecutorFormular
from .prompts.system_agent import SYSTEM_AGENT
from .prompts.system_formular import SYSTEM_FORMULAR
from .adapters.inkludocs import build_project_summary
from .sanitize import sanitize_markdown

log = logging.getLogger(__name__)

_MAX_ITERATIONS = 6
_HISTORY_TURNS = 20  # weniger als 30, weil Tool-Calls + Images Context-Hungry sind
_DEFAULT_MAX_TOKENS = 4096
_MAX_TOOL_RESULT_CHARS = 40000  # Werkzeug-Ergebnis im tool_result (28.08.2026, vorher 8000)


def _detect_media_type(img_bytes: bytes) -> str:
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


# ─── Werkzeugsatz je Projektart (28.08.2026) ───
# Ein Formular-Projekt (Quickinfo-Werkzeug) hat Felder statt Bilder: eigener
# System-Prompt, eigener Werkzeugsatz, eigene Projekt-Zusammenfassung. Die
# Bild-Werkzeuge existieren fuer den Formular-Agenten nicht (und umgekehrt),
# damit sich die Werkzeuge nie vermischen. Neue Werkzeuge = neuer Eintrag hier.
def _ist_formular(project: dict) -> bool:
    return (project or {}).get("tool") == "formular"


def _werkzeugsatz(project: dict, project_id: int, user_id: int):
    """(tool_definitions, executor, system_prompt) fuer dieses Projekt."""
    if _ist_formular(project):
        return TOOL_DEFINITIONS_FORMULAR, ToolExecutorFormular(project_id=project_id, user_id=user_id), SYSTEM_FORMULAR
    return TOOL_DEFINITIONS, ToolExecutor(project_id=project_id, user_id=user_id), SYSTEM_AGENT


def _formular_summary(project_id: int) -> str:
    """Kompakter Projekt-Kontext fuer Formular-Projekte (Gegenstueck zu build_project_summary)."""
    try:
        from .tools.formular import list_form_fields
        r = list_form_fields(project_id, _summary_user_id_holder.get(project_id, 0))
    except Exception:
        return "Formular-Projekt (Quickinfo-Werkzeug). Hol den Stand mit list_form_fields."
    if not r.get("ok"):
        return "Formular-Projekt (Quickinfo-Werkzeug). Hol den Stand mit list_form_fields."
    res = r["result"]
    p = res["project"]
    return (f"Formular-Projekt „{p['name']}“ (Quickinfo-Werkzeug), Status {p['status']}, Sprache der Quickinfos "
            f"{p['sprache_der_quickinfos']}: {res['feld_count']} Felder in {len(res['documents'])} Dokument(en), "
            f"{res['offen']} ohne Quickinfo, {res['beschrieben']} beschrieben; {res['stammdaten_eintraege']} Stammdaten-Eintraege "
            "im Konto. Details je Feld: list_form_fields / get_field_details.")


_summary_user_id_holder: dict[int, int] = {}


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
    summary = _formular_summary(project_id) if _ist_formular(project) else build_project_summary(project)
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
    system_suffix: str = None,
    on_tool=None,
) -> dict[str, Any]:
    """Führt den agentic Loop aus. Returns:
        {"reply": str, "intent": "agentic", "image_refs": [...] or None,
         "actions": [{"tool": "...", "args": {...}, "ok": bool}]}.

    Hinweis zur Persistenz: User-Message + Bot-Reply werden NICHT hier
    gespeichert — das übernimmt main.py:_chat_endpoint vor/nach diesem
    Aufruf zentral.

    system_suffix: optionaler Text, der NUR im Demo-Modus an den System-Prompt
    angehängt wird (thematische Leitplanke). Default None = normaler Pfad,
    System-Prompt unverändert.
    on_tool (28.08.2026, Werkzeug-Transparenz): Callback (name, args) VOR jedem
    Werkzeugaufruf — der Chat-Endpunkt streamt daraus die Live-Zeile „Ruft gerade
    auf: …“ an die Oberflaeche. Ergebnis enthaelt zusaetzlich "werkzeuge": Namen
    in Aufrufreihenfolge (leer = ohne Werkzeug aus dem Verlauf geantwortet).
    """
    _summary_user_id_holder[project_id] = user_id
    tool_defs, executor, system_base = _werkzeugsatz(project, project_id, user_id)
    messages = _build_initial_messages(project_id, user_message, project)
    system_text = system_base if not system_suffix else system_base + "\n\n" + system_suffix

    actions_log: list[dict] = []
    werkzeuge: list[str] = []
    image_refs_seen: set[int] = set()
    last_reply_text = ""

    for iteration in range(_MAX_ITERATIONS):
        try:
            payload = provider.invoke_with_tools(
                anthropic_messages=messages,
                tools=tool_defs,
                system=system_text,
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
            werkzeuge.append(name)
            if on_tool:
                try:
                    on_tool(name, args)
                except Exception:
                    log.exception("on_tool-Callback fehlgeschlagen (ignoriert)")
            result = executor.execute(name, args)
            actions_log.append({
                "tool": name,
                "args": args,
                "ok": bool(result.get("ok")),
                "error": result.get("error", "") if not result.get("ok") else "",
            })

            # Refresh-Marker fuer Frontend: nach erfolgreichem DB-Update den
            # frischen Anzeige-Text mitliefern, damit das Textarea live updaten
            # kann ohne Liste-Reload. Fallback-Kette gleich wie main._display_alt_text.
            if name in ("update_alt_text", "revert_alt_text") and result.get("ok"):
                img_id_raw = args.get("image_id") if isinstance(args, dict) else None
                if img_id_raw is not None:
                    try:
                        img_id_int = int(img_id_raw)
                        _conn = sqlite3.connect("/app/data/inkludocs.db")
                        _conn.row_factory = sqlite3.Row
                        _row = _conn.execute(
                            "SELECT alt_text_edited, alt_text, original_alt, langbeschreibung "
                            "FROM images WHERE id = ?",
                            (img_id_int,),
                        ).fetchone()
                        _conn.close()
                        if _row is not None:
                            actions_log.append({
                                "type": "refresh_image",
                                "image_id": img_id_int,
                                "alt_text": _row["alt_text_edited"] or _row["alt_text"] or _row["original_alt"] or "",
                                "langbeschreibung": _row["langbeschreibung"] or "",
                            })
                    except Exception:
                        log.exception("refresh_image-Action nicht angefuegt (img=%s)", img_id_raw)

            # Formular (28.08.2026): nach Aenderung einer Quickinfo den frischen Stand
            # mitliefern, damit formular.js Textfeld + Badge live setzt (refresh_feld).
            if name in ("update_quickinfo", "revert_quickinfo", "generate_quickinfo") and result.get("ok"):
                r = result.get("result") or {}
                actions_log.append({
                    "type": "refresh_feld", "feld_id": r.get("feld_id"), "quickinfo": r.get("quickinfo", ""),
                    "quelle": r.get("quelle", ""), "sicherheit": r.get("sicherheit", ""), "beleg": r.get("beleg", ""),
                    "ki_hinweise": r.get("hinweise", []),
                    "status": "beschrieben" if (r.get("quickinfo") or "").strip() else "offen",
                })

            # image_bytes-Sonderfall (view_image / view_field)
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

            # Serialisiere result als kompaktes JSON für tool_result.
            # 28.08.2026: Kappe 8000 -> 40000 Zeichen — bei 26 Formularfeldern war die
            # Liste abgeschnitten, der Agent pruefte nur die Haelfte (und sagte es).
            try:
                content_text = json.dumps(result, ensure_ascii=False)[:_MAX_TOOL_RESULT_CHARS]
            except (TypeError, ValueError):
                content_text = str(result)[:_MAX_TOOL_RESULT_CHARS]
            if len(content_text) >= _MAX_TOOL_RESULT_CHARS:
                content_text += ' …[Ergebnis gekuerzt — Werkzeug mit engerer Auswahl erneut aufrufen]'
                log.warning("Tool-Result %s auf %d Zeichen gekappt", name, _MAX_TOOL_RESULT_CHARS)

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

    # 28.08.2026: Markdown raus (Sonnet setzt trotz Verbot gelegentlich **fett** —
    # das Frontend zeigt Text roh, VoiceOver liest sonst "Stern Stern"). Gleicher
    # Filter wie im klassischen Pfad (sanitize_markdown), fuer alle Werkzeuge.
    final_reply = sanitize_markdown(last_reply_text) or (
        "Habe gerade keine konkrete Antwort dafür. Frag mich nochmal anders?"
    )

    image_refs_list = sorted(image_refs_seen) if image_refs_seen else None
    return {
        "reply": final_reply,
        "intent": "agentic",
        "image_refs": image_refs_list,
        "actions": actions_log,
        "werkzeuge": werkzeuge,
    }
