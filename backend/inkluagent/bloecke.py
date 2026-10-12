"""Antworten als Bloecke auf dem Server (fuer die Lese-Seite „Frühere Gespräche“, 12.10.2026) — dieselben Regeln wie
frontend/inkluagent.js inkluagentBloeckeBauen: Leerzeile = neuer Absatz, Zeilen mit „• “ = Liste, mindestens zwei Zeilen
„1. “, „2. “ = nummerierte Liste (eine einzelne bleibt Absatz). Die Vorlage gibt nur Text aus (Jinja maskiert)."""
from __future__ import annotations

import re

_PUNKT = re.compile(r"^\s*•\s+(.*)$")
_NUMMER = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")


def bloecke(text: str) -> list[dict]:
    """[{"art": "p"|"ul"|"ol", "zeilen": [...], "start": n}]"""
    out: list[dict] = []
    for absatz in re.split(r"\n[ \t]*\n", text or ""):
        gruppen: list[dict] = []
        for z in absatz.split("\n"):
            punkt, nr = _PUNKT.match(z), _NUMMER.match(z)
            art = "ul" if punkt else ("ol" if nr else "p")
            inhalt = punkt.group(1) if punkt else (nr.group(2) if nr else z)
            if not inhalt.strip():
                continue
            if gruppen and gruppen[-1]["art"] == art:
                gruppen[-1]["zeilen"].append(inhalt)
            else:
                gruppen.append({"art": art, "zeilen": [inhalt], "start": int(nr.group(1)) if nr else 1, "roh": []})
            gruppen[-1]["roh"].append(z.strip())
        for g in gruppen:
            if g["art"] == "ol" and len(g["zeilen"]) < 2:
                g["art"], g["zeilen"] = "p", g["roh"]
            if g["art"] == "p":
                g["zeilen"] = ["\n".join(g["zeilen"])]
            out.append({"art": g["art"], "zeilen": g["zeilen"], "start": g["start"]})
    return out
