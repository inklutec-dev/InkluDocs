"""Web-Search-Tools — Tavily für LLM-optimierte Suche.

Tavily API: https://docs.tavily.com/docs/rest-api/api-reference
Antwort enthält 'answer' (kurze KI-Zusammenfassung) + 'results'
(Liste mit url, title, content, score).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_HTTP_TIMEOUT = 25.0


def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    include_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Web-Suche via Tavily. Liefert kurze KI-Zusammenfassung + Top-Hits.

    Args:
        query: Suchanfrage in natürlicher Sprache.
        max_results: Anzahl Suchergebnisse (1-10).
        search_depth: 'basic' (schnell) oder 'advanced' (mehr Snippet-Tiefe).
        include_domains: optional Liste von Domains zum Filtern,
            z.B. ['w3.org', 'bitv-test.de', 'bik-fuer-alle.de'].

    Returns:
        {"ok": True, "result": {"answer": "...", "results": [...]}}
        oder {"ok": False, "error": "..."}.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY ist nicht gesetzt."}

    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "Leere Suchanfrage."}

    payload = {
        "api_key": api_key,
        "query": q,
        "max_results": max(1, min(int(max_results), 10)),
        "search_depth": search_depth if search_depth in ("basic", "advanced") else "advanced",
        "include_answer": True,
        "include_raw_content": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    try:
        resp = httpx.post(_TAVILY_ENDPOINT, json=payload, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("Tavily HTTP-Fehler: %s", e)
        return {"ok": False, "error": f"Tavily HTTP-Fehler: {e}"}

    try:
        data = resp.json()
    except ValueError as e:
        return {"ok": False, "error": f"Tavily Antwort nicht JSON: {e}"}

    results = []
    for r in data.get("results", []) or []:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": (r.get("content") or "")[:1200],
            "score": r.get("score"),
        })

    return {
        "ok": True,
        "result": {
            "query": q,
            "answer": data.get("answer") or "",
            "results": results,
        },
    }
