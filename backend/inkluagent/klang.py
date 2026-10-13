"""Klangspur des InkluAgent (Paket 1, Steve 13.10.2026; Schalter funktionen.AGENT_KLANG).

Zustaende hoerbar statt angesagt: der Browser spielt kurze Klaenge (frontend/klang.js) — ein Wusch, wenn eine Karte ins
Gespraech fliegt, ein Tick je erledigtem Schritt, ein Klang fuer „fertig“ und einer fuer „ging schief“. Das ersetzt keine Ansage,
die Information steht immer auch im Text; es macht den Stand schneller und leiser erfassbar (Pruefung 13.10.2026, B9: zu viele
Ansagen bei langen Laeufen).

Je Konto abschaltbar („Töne“ in den Einstellungen), gespeichert in users.agent_toene (Vorgabe an). Der Wert steht schon beim
Seitenbau am <body data-toene>, damit der erste Klang der Seite der Einstellung folgt.
"""
from __future__ import annotations

from typing import Optional

import funktionen


def an() -> bool:
    return bool(funktionen.an("AGENT_KLANG") and funktionen.an("AGENT_UEBERALL"))


def fuer_konto(user_id: int) -> bool:
    from database import get_db
    conn = get_db()
    try:
        row = conn.execute("SELECT agent_toene FROM users WHERE id = ?", (int(user_id),)).fetchone()
    finally:
        conn.close()
    return True if not row or row[0] is None else bool(row[0])


def setzen(user_id: int, wert: bool) -> bool:
    from database import get_db
    conn = get_db()
    try:
        conn.execute("UPDATE users SET agent_toene = ? WHERE id = ?", (1 if wert else 0, int(user_id)))
        conn.commit()
    finally:
        conn.close()
    return bool(wert)


def seiten_wert(user_id: Optional[int]) -> Optional[str]:
    """Fuer den Seitenbau (data-toene): None, solange der Schalter aus ist — dann aendert sich am Seitenrahmen nichts."""
    if not an() or not user_id:
        return None
    return "an" if fuer_konto(int(user_id)) else "aus"
