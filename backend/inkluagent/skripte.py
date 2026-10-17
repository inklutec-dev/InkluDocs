"""Die Skripte des InkluAgent in der Seite (17.10.2026: aus EINER Datei frontend/inkluagent.js in Teile je Thema; Steve: „so
strukturieren, dass er nicht verbastelt wirkt und wir ihn immer weiter intelligenter machen und erweitern koennen“).

EINE Liste in Ladereihenfolge: die Vorlagen binden genau diese Dateien ein (base_app.html und app.html ueber die Jinja-Variable
inkluagent_skripte, gesetzt in main.py), die Tests lesen sie zusammen (quelltext). Ein neuer Teil = eine Datei in frontend/ und ein
Eintrag hier. Auf oberster Ebene stehen in allen Teilen nur Funktionen (dazu zwei Merker im Kern); die einzige Ausfuehrung beim
Laden, inkluagentBereichAufbauen(), steht am Ende des Kerns und braucht alle Teile — darum laedt der Kern ZULETZT. Die Reihenfolge
der uebrigen Teile ist beliebig (sie rufen sich erst auf, wenn alle geladen sind); sie folgt dem Weg durch die Oberflaeche.
"""
from __future__ import annotations

import os

TEILE: tuple = (
    "inkluagent-ansicht.js",          # Agentenansicht und Chatstil, Aufbau des Arbeitsfensters
    "inkluagent-arbeitsfenster.js",   # Stand, Schritte, Entwurf, Vergroessern, „Stand vorlesen“
    "inkluagent-fenster.js",          # Seiten im Arbeitsfenster, das Projekt auf der Buehne (nur ueber window.InkluBuehne)
    "inkluagent-live.js",             # Live-Fenster aus dem Laufbuch
    "inkluagent-hochladen.js",        # Dateien mitschicken und anhaengen
    "inkluagent-gespraeche.js",       # Gespraech des Kontos, Verlauf, fruehere Gespraeche
    "inkluagent-nachrichten.js",      # Nachrichten bauen: Anhaenge, Bloecke, „Anhoeren“
    "inkluagent-karten.js",           # Formular-Karten (Bestellen, Bezahlen, Auftraege)
    "inkluagent-seitenwechsel.js",    # Seitenwechsel durch den Agenten
    "inkluagent-rueckmeldung.js",     # der Agent meldet sich, wenn etwas fertig ist
    "inkluagent.js",                  # Kern: Bereich, Binden, Senden, Status, Fokus — zuletzt (Startaufruf am Ende)
)


def quelltext(frontend: str) -> str:
    """Alle Teile zusammen, in Ladereihenfolge (fuer Pruefungen am Quelltext)."""
    out = []
    for name in TEILE:
        with open(os.path.join(frontend, name), encoding="utf-8") as f:
            out.append(f.read())
    return "\n".join(out)
