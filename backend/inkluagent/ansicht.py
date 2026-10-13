"""Ansicht je Konto (InkluAgent-Ausbau Runde 1, Schritt 5, 09.10.2026; Konzept Wunsch 1; Schalter funktionen.AGENT_ANSICHT).

Steve (09.10.2026): EINE Einstellung „Ansicht“, Hauptort sind die Einstellungen, dazu ein schneller Umschalter in der
Seitenleiste.
- „Manuelle Ansicht“ (intern „klassisch“, Vorgabe auch fuer neue Konten): die gewohnte Oberflaeche zum haendischen Arbeiten;
  der InkluAgent steht wie bisher eingeklappt in jedem Projekt und laesst sich fuer Fragen aufklappen.
- „Agentenansicht“ (intern „agent“): in einem Projekt nur der InkluAgent gross, die Seitenleiste bleibt. Mit dem Agenten auf jeder
  Seite (Runde 2) gilt das fuer jede Seite; mit Arbeitsfenster (Stufe A, 13.10.2026) arbeitet man im Gespraech, und die Seiten oeffnen
  sich im Arbeitsfenster. Welcher Satz gilt, entscheidet beschreibung() nach den Schaltern.

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
# Seit der InkluAgent auf jeder Seite steht (Runde 2, Schritt 3) und die Agentenansicht wie ChatGPT arbeitet (Arbeitsfenster,
# 12.10.2026), stimmen die Saetze oben nur noch ohne diese Schalter (Steve 12.10.2026: „die übrigen Seiten bleiben“ war veraltet).
BESCHREIBUNG_UEBERALL = {
    "klassisch": "Die gewohnte Oberfläche zum Arbeiten von Hand. Der InkluAgent steht eingeklappt unten auf jeder Seite und hilft dir auf Wunsch.",
    "agent": "Auf jeder Seite siehst du nur die Überschrift und den InkluAgent, groß und geöffnet; die Seitenleiste bleibt. Den Inhalt einer Seite holst du jederzeit mit einem Knopf zurück.",
}
# Pruefung 13.10.2026 (B13/K11): den Knopf „… in der manuellen Ansicht zeigen“ gibt es im Chatstil nicht mehr (Stufe A), und „rechts“
# stimmte unter 1100 px nicht (dort steht das Fenster unter dem Gespraech) — jetzt ohne Lageangabe, mit dem Weg zurueck.
BESCHREIBUNG_ARBEITSFENSTER = ("Du arbeitest im Gespräch mit dem InkluAgent, wie in einem Chat. Das Arbeitsfenster zeigt, woran er gerade "
                               "arbeitet, und öffnet die Seiten der Navigation, ohne dass du das Gespräch verlässt. Zur manuellen Ansicht "
                               "wechselst du mit dem Knopf unter der Navigation.")


def beschreibung(wert: str) -> str:
    """Beschreibung einer Ansicht passend zu den Schaltern (msgid; uebersetzt der Aufrufer)."""
    ueberall = funktionen.an("AGENT_UEBERALL") and funktionen.an("AGENT_KONTO")
    if wert == "agent" and ueberall and funktionen.an("AGENT_ARBEITSFENSTER"):
        return BESCHREIBUNG_ARBEITSFENSTER
    return (BESCHREIBUNG_UEBERALL if ueberall else BESCHREIBUNG)[wert]


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
    return [{"wert": w, "name": _(NAMEN[w]), "beschreibung": _(beschreibung(w))} for w in OBERFLAECHEN]
