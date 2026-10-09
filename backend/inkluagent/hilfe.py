"""Hilfe-Seite „Alles, was der InkluAgent kann“ (InkluAgent-Ausbau Runde 1, Schritt 3, Schalter funktionen.AGENT_HILFE).

Die Seite entsteht aus DENSELBEN Quellen wie der Agent selbst: dem Werkzeugsatz je Projektart (agent_loop._werkzeugsatz,
Schalter in funktionen.py schon angewendet), den Anzeigenamen der Werkzeuge (tools/namen.py, sechs Sprachen) und daraus,
ob ein Werkzeug vorher fragt (Parameter bestaetigt = Angebot und Zustimmung, tools/ausgaben.py). So verspricht sie nie mehr,
als gerade da ist: ein ausgeschaltetes Werkzeug fehlt auch hier.
"""
from __future__ import annotations

from typing import Callable, Optional

# Projektarten der Oberflaeche (Anlege-Menue, tools.py) in derselben Reihenfolge; das alte Formular-Projekt wird nicht mehr
# angelegt und steht deshalb nicht hier.
PROJEKTARTEN = (
    ("pdf", "PDF-Dokumente", {"project_type": "pdf", "tool": "pdf"}),
    ("word", "Word-Dokumente", {"project_type": "docx", "tool": "word"}),
    ("grafik", "Grafiken", {"project_type": "images", "tool": "grafik"}),
    ("web", "Webseiten", {"project_type": "url", "tool": "web"}),
)

# Was bewusst ausserhalb des Agenten bleibt (Steve 09.10.2026, Konzept Frage 9): erledigt man selbst in den Einstellungen.
AUSSERHALB = ("Abo und Zahlung", "Konto löschen", "Passwort und E-Mail-Adresse ändern", "API-Schlüssel",
              "Team und Gäste einladen")


def fragt_vorher(definition: dict) -> bool:
    return "bestaetigt" in ((definition.get("input_schema") or {}).get("properties") or {})


def seite(_: Optional[Callable[[str], str]] = None) -> dict:
    """{"projektarten": [{"schluessel", "titel", "ohne_rueckfrage": [Namen], "mit_rueckfrage": [Namen]}], "ausserhalb": [...]}"""
    _ = _ or (lambda s: s)
    from .agent_loop import _werkzeugsatz
    from .tools.namen import werkzeug_name
    arten = []
    for schluessel, titel, projekt in PROJEKTARTEN:
        defs, _executor, _system = _werkzeugsatz(projekt, 0, 0)
        ohne, mit = [], []
        for d in defs:
            name = werkzeug_name(d["name"], _)
            ziel = mit if fragt_vorher(d) else ohne
            if name not in ziel:
                ziel.append(name)
        arten.append({"schluessel": schluessel, "titel": _(titel), "ohne_rueckfrage": ohne, "mit_rueckfrage": mit})
    return {"projektarten": arten, "ausserhalb": [_(t) for t in AUSSERHALB]}
