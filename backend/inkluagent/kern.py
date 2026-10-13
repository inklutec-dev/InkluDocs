"""Gemeinsame Helfer fuer Werkzeuge, die DENSELBEN Kern wie ein Knopf der Oberflaeche aufrufen (InkluAgent-Ausbau Runde 2).

Viele Knoepfe stecken direkt in FastAPI-Endpunkten (create_project, rename_project, delete_project, scan_url, Abbrechen in
formular_api/uebersetzung_api). Statt ihre Logik zu kopieren, ruft der InkluAgent genau diese Endpunkt-Funktionen auf:
- endpunkt(methode, pfad): die registrierte Funktion zu einer Route (auch aus Routern wie formular_api)
- Anfrage(daten): ersetzt das Request-Objekt, wenn ein Endpunkt nur `await request.json()` liest
- warten(coro): eine async Endpunkt-Funktion aus einem Werkzeug-Faden heraus ausfuehren (auf der Server-Schleife, wenn sie
  laeuft; sonst — Unit-Tests — in einer eigenen Schleife)
- fehlertext(e, satz): was ein Werkzeug bei einem Fehler an Modell und Seite gibt (Pruefung 13.10.2026, S10)
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


def fehlertext(e: BaseException, satz: str) -> str:
    """Den Satz einer HTTPException (fuer Menschen geschrieben, derselbe wie beim Knopf), sonst nur `satz`. Pfade, Datenbank- und
    Anbieterfehler gehen ins Log, nicht an das Modell, in den Chat oder in den Livestream (Pruefung 13.10.2026, S10)."""
    from fastapi import HTTPException
    if isinstance(e, HTTPException):
        d = (e.detail.get("text") or e.detail.get("detail")) if isinstance(e.detail, dict) else e.detail
        if d:
            return str(d)
    log.warning("%s (%r)", satz, e)
    return satz


def _main():
    return importlib.import_module("main")


def endpunkt(methode: str, pfad: str) -> Optional[Callable]:
    for r in getattr(_main().app, "routes", []):
        if getattr(r, "path", None) == pfad and methode.upper() in (getattr(r, "methods", None) or ()):
            return r.endpoint
    return None


class Anfrage:
    """Kleiner Ersatz fuer starlette.Request: nur json() — fuer Endpunkte, die sonst nichts aus der Anfrage lesen."""

    def __init__(self, daten: dict) -> None:
        self._daten = dict(daten or {})
        self.headers: dict = {}
        self.cookies: dict = {}

    async def json(self) -> dict:
        return dict(self._daten)


def warten(coro, timeout: float = 180.0) -> Any:
    schleife = getattr(_main(), "_HAUPT_LOOP", None)
    if schleife is not None and schleife.is_running():
        try:
            if asyncio.get_running_loop() is schleife:
                raise RuntimeError("warten() darf nicht auf der Server-Schleife selbst laufen")
        except RuntimeError as e:
            if "Server-Schleife" in str(e):
                coro.close()
                raise
        return asyncio.run_coroutine_threadsafe(coro, schleife).result(timeout)
    return asyncio.run(coro)
