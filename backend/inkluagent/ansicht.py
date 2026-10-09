"""Ansicht je Konto (InkluAgent-Ausbau Runde 1, Schritt 5, 09.10.2026; Konzept Wunsch 1; Schalter funktionen.AGENT_ANSICHT).

Steve (09.10.2026): EINE Einstellung „Ansicht“, Hauptort sind die Einstellungen, dazu ein schneller Umschalter in der
Seitenleiste.
- „Manuelle Ansicht“ (intern „klassisch“, Vorgabe auch fuer neue Konten): die gewohnte Oberflaeche zum haendischen Arbeiten;
  der InkluAgent steht wie bisher eingeklappt in jedem Projekt und laesst sich fuer Fragen aufklappen.
- „Agentenansicht“ (intern „agent“): in einem Projekt nur der InkluAgent gross, die Seitenleiste bleibt. Seiten ohne Projekt
  bleiben vorerst, wie sie sind (der Agent fuer das ganze Konto kommt in Runde 2).

Gespeichert in users.oberflaeche (bewusst nicht „ansicht“: projects.letzte_ansicht ist etwas anderes). Der Wert steht in
/api/me und wird schon beim Seitenbau auf dem Server gesetzt (<body data-oberflaeche>), damit nichts springt.
Erweiterbar: ein weiterer Wert = ein Eintrag in OBERFLAECHEN, NAMEN und BESCHREIBUNG (und seine Darstellung in
frontend/inkluagent.js und style.css); die Einstellungsseite und die Pruefung lesen die Liste von hier.
"""
from __future__ import annotations

from typing import Optional

import funktionen

OBERFLAECHEN = ("klassisch", "agent")     # Reihenfolge = Reihenfolge in den Einstellungen
VORGABE = "klassisch"
NAMEN = {"klassisch": "Manuelle Ansicht", "agent": "Agentenansicht"}
BESCHREIBUNG = {
    "klassisch": "Die gewohnte Oberfläche zum Arbeiten von Hand. Der InkluAgent steht eingeklappt in jedem Projekt und hilft dir auf Wunsch.",
    "agent": "In deinen Projekten siehst du nur den InkluAgent, groß und geöffnet; die Seitenleiste bleibt. Die übrigen Seiten bleiben vorerst, wie sie sind.",
}


def an() -> bool:
    return funktionen.an("AGENT_ANSICHT")


def gueltig(wert: Optional[str]) -> bool:
    return wert in OBERFLAECHEN


def fuer_konto(user_id: int) -> str:
    """Gespeicherte Ansicht des Kontos; unbekannt oder leer = Vorgabe."""
    from database import get_db
    conn = get_db()
    try:
        row = conn.execute("SELECT oberflaeche FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    wert = (row[0] if row else None) or VORGABE
    return wert if gueltig(wert) else VORGABE


def setzen(user_id: int, wert: str) -> str:
    if not gueltig(wert):
        raise ValueError(wert)
    from database import get_db
    conn = get_db()
    try:
        conn.execute("UPDATE users SET oberflaeche = ? WHERE id = ?", (wert, user_id))
        conn.commit()
    finally:
        conn.close()
    return wert


def seiten_wert(user_id: Optional[int]) -> Optional[str]:
    """Fuer den Seitenbau (data-oberflaeche): None, solange der Schalter aus ist — dann aendert sich am Seitenrahmen nichts."""
    if not an() or not user_id:
        return None
    return fuer_konto(int(user_id))


def auswahl(_=None) -> list[dict]:
    """Die Einstellungsseite: [{"wert", "name", "beschreibung"}] in der Reihenfolge von OBERFLAECHEN, uebersetzt."""
    _ = _ or (lambda s: s)
    return [{"wert": w, "name": _(NAMEN[w]), "beschreibung": _(BESCHREIBUNG[w])} for w in OBERFLAECHEN]
