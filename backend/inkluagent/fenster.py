"""Welche Seiten im Arbeitsfenster der Agentenansicht oeffnen duerfen — EINE Liste fuer Server und Oberflaeche (Pruefung
13.10.2026, Runde 2, S-N6/S15 und der Wunsch des Oberflaechen-Helfers).

Vorher pflegten Server und Oberflaeche je eine eigene Sperrliste: der Server verbot das Einrahmen fuer ein paar Seiten
(Anmeldung, Passwort, Freigaben …) und erlaubte es fuer alle anderen, die Oberflaeche oeffnete alles ausser einer eigenen Liste im
Fenster. Neue Seiten waren damit erst einmal einrahmbar, /forgot fehlte. Jetzt eine Positivliste: nur diese Seiten (und die
Anmeldung, damit die Agentenansicht sie bei abgelaufener Sitzung nach vorn holen kann, K7) darf die eigene Seite einrahmen;
die Oberflaeche bekommt dieselbe Liste (window.INKLUAGENT.fensterSeiten) und oeffnet alles andere als ganze Seite.

Aufgenommen sind Seiten mit eingebetteter Fassung: alles auf base_app.html (eingebettet() laesst Seitenleiste, Fusszeile und
Agent weg) und die oeffentlichen Seiten auf base_oeffentlich.html, die die Agentenansicht oeffnet. Statische Rechtstexte
(Datenschutz, Impressum, Nutzungsbedingungen, AVV, Datensicherheit), Projekte, Startseite, Sprache und Anmeldung bleiben ganze Seite.
"""
from __future__ import annotations

FENSTER_SEITEN: tuple[str, ...] = (
    "/ablage", "/abo", "/api-schluessel", "/einstellungen", "/express", "/gespraeche", "/geteilte-projekte", "/hilfe", "/konto",
    "/kuendigen", "/preise", "/projekt-neu", "/projekte", "/prompts", "/stammdaten", "/team", "/ueber-uns", "/vorlagen",
    "/widerrufen",
)
# Einrahmen darf die eigene Seite zusaetzlich die Anmeldung: dort landet der Rahmen, wenn die Sitzung ablaeuft (K7)
EINRAHMBAR: tuple[str, ...] = FENSTER_SEITEN + ("/login",)


def _passt(pfad: str, liste: tuple[str, ...]) -> bool:
    pfad = (pfad or "").split("?", 1)[0].split("#", 1)[0]
    return any(pfad == p or pfad.startswith(p + "/") for p in liste)


def im_fenster(pfad: str) -> bool:
    return _passt(pfad, FENSTER_SEITEN)


def einrahmbar(pfad: str) -> bool:
    return _passt(pfad, EINRAHMBAR)
