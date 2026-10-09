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
# PROFESSIONELLES TAGGING (09.10.2026, Steve nach Absprache mit Michael Karbe): „Barrierefrei machen“ / „Neu taggen“ =
# bezahltes Tagging der Originaldatei. Ohne PDFix-Tagging-Lizenz liefe es im Testmodus (Logo, fehlender Text unter dem
# Logo, Hersteller „Trial version“) — bis die Lizenz da ist, bleibt es gesperrt. Wie EXPRESS an der UMGEBUNG:
# TAGGING_PROFESSIONELL=an (docker-compose bzw. .env.staging). Vorgabe AUS — auch wenn Prod die Variable nicht setzt.
# Aus sperrt Knopf, Kette, Chatbot (barrierefrei_machen) und Server (tagging_api.professionell_gesperrt_text); es werden
# nie Credits abgebucht. „Testweise taggen“ und der Download der Testfassung bleiben immer erlaubt.
TAGGING_PROFESSIONELL = (os.environ.get("TAGGING_PROFESSIONELL") or "aus").strip().lower() in ("an", "on", "1", "true", "ja")
# Derselbe Wortlaut in der Karte (dokument.js), am Server (403) und im Chatbot — msgid in den .po-Katalogen.
TAGGING_PROFESSIONELL_GESPERRT = ("Das professionelle Tagging schalten wir in Kürze frei. Bis dahin kannst du dein Dokument "
                                  "kostenlos testweise taggen und die Testfassung herunterladen.")
# Unterschalter des Express-Service ohne Codeaenderung (Zusatz 05.10.2026, Steve): Knopf „In den Express-Warenkorb“ am
# Dokument und Eintrag „Express-Warenkorb“ in der Navigation (immer / nur mit Inhalt / aus) stehen in der Verwaltung unter
# „Einstellungen des Express-Service“ (express.EINSTELLUNGEN_STANDARD korb_knopf, korb_navigation) und wirken nur, wenn
# EXPRESS an ist. window.FUNKTIONEN.express_korb_knopf kommt aus express_api.fuer_oberflaeche().



def _umgebung_an(name: str) -> bool:
    return (os.environ.get(name) or "aus").strip().lower() in ("an", "on", "1", "true", "ja")


# INKLUAGENT-AUSBAU Runde 1 (09.10.2026, Konzept „InkluAgent ausbauen“): je Schritt EIN Schalter, Vorgabe AUS, gesetzt ueber
# die UMGEBUNG wie EXPRESS (derselbe Code kann auf Staging an und auf Prod aus sein). Doku: docs/INKLUAGENT.md, „Ausbau Runde 1“.
# Schritt 2 — Sicherheitsfundament (inkluagent/sicherheit.py): hoechstens eine bezahlte Einzelaktion je Nachricht ohne Karte,
# Ja-Pruefung auf dem Server, Tagesgrenze ueber eigenen Zaehler, Tages-Kostendeckel je Konto. INKLUAGENT_SICHERHEIT=an.
AGENT_SICHERHEIT = _umgebung_an("INKLUAGENT_SICHERHEIT")
# Schritt 3 — Erklaerung und Hilfe: drei kurze Stichpunkte vor dem Chat statt der langen Einleitung (der KI-Hinweis nach
# KI-Verordnung Art. 50 bleibt der erste Punkt), Hilfe-Seite /hilfe/inkluagent (erzeugt aus Werkzeugsatz, Werkzeugnamen und
# Schaltern, inkluagent/hilfe.py) und der Link „Hilfe“ in der Seitenleiste. INKLUAGENT_HILFE=an.
AGENT_HILFE = _umgebung_an("INKLUAGENT_HILFE")
# Schritt 4 — Werkzeugluecke in Grafik- und Webseiten-Projekten: dort hat der Agent sonst nur sechs Werkzeuge. Mit Schalter
# alles, was die Oberflaeche dort anbietet (Alt-Texte fuer alle, Alt-Texte herunterladen, KI-Kontext, gespeicherter Prompt,
# Sprache der Alt-Texte, Bild umbenennen und loeschen, bei Webseiten Webseite umbenennen und loeschen), jedes Werkzeug mit
# demselben Kern wie der Knopf und derselben Rueckfrage. INKLUAGENT_BILD_WERKZEUGE=an.
AGENT_BILD_WERKZEUGE = _umgebung_an("INKLUAGENT_BILD_WERKZEUGE")
# Schritt 5 — Ansicht je Konto (inkluagent/ansicht.py): Einstellung „Ansicht“ (Manuelle Ansicht / Agentenansicht) in den
# Einstellungen und als schneller Umschalter in der Seitenleiste, gespeichert in users.oberflaeche, beim Seitenbau gesetzt.
# Neue Konten: manuelle Ansicht. INKLUAGENT_ANSICHT=an.
AGENT_ANSICHT = _umgebung_an("INKLUAGENT_ANSICHT")

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


def tagging_professionell_gesperrt(_=None):
    """None, solange das professionelle (bezahlte) Tagging frei ist; sonst der freundliche Hinweis, uebersetzt mit _."""
    if an("TAGGING_PROFESSIONELL"):
        return None
    return (_ or (lambda s: s))(TAGGING_PROFESSIONELL_GESPERRT)


def fuer_oberflaeche() -> dict:
    """window.FUNKTIONEN in app.html."""
    return {"ki_pruefung": KI_PRUEFUNG, "korrektur": KORREKTUR, "eigene_pruefungen": EIGENE_PRUEFUNGEN, "urteil": URTEIL,
            "kette": KETTE, "text_zurueck": TEXT_ZURUECK, "strukturansicht": STRUKTURANSICHT, "express": EXPRESS,
            "tagging_professionell": TAGGING_PROFESSIONELL, "agent_hilfe": AGENT_HILFE}
