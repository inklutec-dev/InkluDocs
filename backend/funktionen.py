"""Funktionsschalter: EIN Ort je Funktion, der Oberflaeche, Chatbot und Server-Endpunkte gemeinsam steuert.

Steve 30.09.2026: „Alles, was man händisch macht, soll über den InkluAgent gehen“ — und umgekehrt bietet der InkluAgent
nichts an, was die Oberflaeche nicht anbietet. Vorher hatte jede Seite ihren eigenen Schalter (dokument.js
ZEIGE_KI_PRUEFUNG, abschluss.js ZEIGE_KI, abschluss.EIGENE_PRUEFUNGEN …) und der Chatbot gar keinen: KI-Pruefung und
Korrektur waren in der Oberflaeche aus, im Chatbot aber aktiv (Audit 30.09.2026, HOCH 1).

Ein Schalter hier wirkt auf drei Stellen:
  - Oberflaeche: app.html setzt window.FUNKTIONEN (fuer_oberflaeche()); dokument.js, abschluss.js und app.html lesen daraus.
  - Chatbot: agent_loop._werkzeugsatz laesst Werkzeuge weg (werkzeug_erlaubt), der ToolExecutor fuehrt sie nicht aus,
    und die Systemprompts erwaehnen sie nicht (system_pdf.system_pdf()).
  - Server: die Endpunkte antworten mit 404, solange der Schalter aus ist (endpunkt_frei).
Zum Einschalten den Wert hier auf True setzen — sonst nichts.
"""
from __future__ import annotations

import os

# KI-basierte Pruefung (experimentell): Block in „Tagging“ und „Barrierefreiheitsprüfung“, Chatbot pruefung_starten /
# pruefbericht_lesen, POST …/pruefung, GET …/pruefung/befunde.csv. Aus seit 24.09.2026 (Steve; Michael, Feedback 28.09.).
KI_PRUEFUNG = False
# Korrektur nach der KI-Pruefung und „Korrektur rückgängig“: schreibt die Arbeitsdatei um. Knoepfe, Chatbot korrektur_*,
# POST …/korrektur, …/korrektur/rueckgaengig. Aus (Audit 30.09.2026: kann Ebenenspruenge erzeugen, Michaels Punkt 13).
KORREKTUR = False
# Eigene Pruefungen (Struktur, Vollstaendigkeit, KI-Befunde) in der Problemliste der Barrierefreiheitsprüfung; aus = nur
# veraPDF (Michael Karbe, Feedback 28.09.2026 - 2).
EIGENE_PRUEFUNGEN = False
# Gesamturteil in „Tagging“ (Feedback 24.09.2026, Punkt 6: Anwender bilden sich ihr Urteil selbst).
URTEIL = False
# „Komplett barrierefrei machen“ (Kette Tagging → Alt-Texte → Quickinfos) und der Ablage-Knopf im Kopf von „Dokument“
# (Feedback 24.09.2026, Punkt 1: vorerst aus). Chatbot komplett_barrierefrei_machen, POST /api/projects/{id}/kette.
KETTE = False
# „Vorherigen Text zurückholen“ je Bild (Sicherheitsnetz nach einem Sammellauf) und das Gegenstueck im Chatbot
# (revert_alt_text: Hand-Text verwerfen, KI-Text wieder aktiv). In der Oberflaeche nie eingeblendet.
TEXT_ZURUECK = False
# Link „Strukturansicht öffnen“ in „Tagging“ (die Seite selbst bleibt fuer „Mit eigenem Screenreader prüfen“).
STRUKTURANSICHT = False
# EXPRESS-SERVICE Stufe 1 (05.10.2026): Bereich „Express-Service“, Link im Projekt, Verwaltung „Express-Aufträge“.
# Anders als die Schalter oben haengt er an der UMGEBUNG, weil derselbe Code auf Staging an und auf Prod aus sein muss:
# EXPRESS_SERVICE=an (docker-compose.staging.yml). Prod/ohne Variable: aus. In der Demo nie.
EXPRESS = ((os.environ.get("EXPRESS_SERVICE") or "aus").strip().lower() in ("an", "on", "1", "true", "ja")
           and (os.environ.get("DEMO_MODE") or "off").strip().lower() not in ("on", "true", "1", "yes"))

# Chatbot-Werkzeuge, die an einem Schalter haengen. Alle anderen sind immer da.
WERKZEUG_SCHALTER = {
    "pruefung_starten": "KI_PRUEFUNG",
    "pruefbericht_lesen": "KI_PRUEFUNG",
    "korrektur_anwenden": "KORREKTUR",
    "korrektur_rueckgaengig": "KORREKTUR",
    "komplett_barrierefrei_machen": "KETTE",
    "revert_alt_text": "TEXT_ZURUECK",
}


def an(name: str) -> bool:
    return bool(globals().get(name, False))


def werkzeug_erlaubt(werkzeug: str) -> bool:
    schalter = WERKZEUG_SCHALTER.get(werkzeug)
    return True if schalter is None else an(schalter)


def werkzeuge_filtern(definitionen: list) -> list:
    return [d for d in definitionen if werkzeug_erlaubt(d.get("name", ""))]


def endpunkt_frei(schalter: str) -> None:
    """In einem Endpunkt hinter einem Schalter: 404, solange er aus ist (die Funktion gibt es dann nicht)."""
    if not an(schalter):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Nicht gefunden")


def fuer_oberflaeche() -> dict:
    """window.FUNKTIONEN in app.html."""
    return {"ki_pruefung": KI_PRUEFUNG, "korrektur": KORREKTUR, "eigene_pruefungen": EIGENE_PRUEFUNGEN, "urteil": URTEIL,
            "kette": KETTE, "text_zurueck": TEXT_ZURUECK, "strukturansicht": STRUKTURANSICHT, "express": EXPRESS}
