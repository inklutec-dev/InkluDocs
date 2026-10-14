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
# Schritt 2 — Sicherheitsfundament (inkluagent/sicherheit.py): Tagesgrenze ueber eigenen Zaehler, Tages-Kostendeckel je Konto.
# INKLUAGENT_SICHERHEIT=an. Der Deckel fuer bezahlte Einzelaktionen und die Ja-Pruefung gelten seit der Pruefung 13.10.2026
# immer (sie sind Teil der Zustimmung, nicht einer Grenze).
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

# RUNDE 2 (09.10.2026, Steve: „parallel am Chatbot bauen“) — wieder je Schritt EIN Schalter, Vorgabe AUS.
# Runde 2, Schritt 1 — Hochladefeld im InkluAgent (inkluagent/hochladen.py): Dateien und Webadresse direkt im Agenten, der
# Server ordnet zu und legt Projekte an, der Agent schlaegt mit Preisen vor. INKLUAGENT_HOCHLADEN=an.
AGENT_HOCHLADEN = _umgebung_an("INKLUAGENT_HOCHLADEN")
# Das FELD „Dateien mitschicken“ im InkluAgent (Michael Karbe, Feedback 20261012 - 1, Punkt 3: „erst einmal weglassen, das verwirrt
# die Anwender derzeit noch“): eigener Schalter fuer die Oberflaeche, Vorgabe aus. Der Server-Teil (AGENT_HOCHLADEN: Endpunkt,
# Webadressen in der Nachricht, Zuordnung) bleibt davon unberuehrt. INKLUAGENT_HOCHLADEFELD=an zeigt das Feld wieder.
AGENT_HOCHLADEFELD = _umgebung_an("INKLUAGENT_HOCHLADEFELD")
# Runde 2, Schritt 2 — EIN Gespraech je Konto (inkluagent/storage.py, Tabellen agent_nachrichten und agent_fokus): der Agent
# fuehrt ein durchgehendes Gespraech ueber alle Projekte, mit Projektmarken („Ab hier: Projekt …“); alte Projektgespraeche
# bleiben im Projekt lesbar, beim Loeschen eines Projekts bleibt das Gespraech (Projektbezug als geloescht markiert).
# Steves Entscheidung 09.10.2026 (Konzept, Abschnitt 6, Fragen 1 und 2). INKLUAGENT_KONTO=an.
AGENT_KONTO = _umgebung_an("INKLUAGENT_KONTO")
# Runde 2, Schritt 3 — der InkluAgent auf JEDER Seite: ein fester Bereich nach dem Hauptinhalt (aside „InkluAgent“, Sprunglink
# „Zum InkluAgent“), in allen Ansichten eines Projekts (auch Dokument und Barrierefreiheitspruefung — Michael 12.10.2026) und auf
# Seiten ohne Projekt mit lesenden Konto-Werkzeugen (inkluagent/konto.py). Eine laufende Antwort uebersteht den Seitenwechsel
# (GET /api/agent/status). Wirkt nur zusammen mit AGENT_KONTO. INKLUAGENT_UEBERALL=an.
AGENT_UEBERALL = _umgebung_an("INKLUAGENT_UEBERALL")
# Runde 2, Schritt 4 — Konto-Werkzeuge, die etwas aendern (inkluagent/konto.py): Projekt anlegen (Name, Art; Bremse wie beim
# Hochladefeld), umbenennen, loeschen (NUR ueber den Knopf der Karte, nie per getipptem Ja, nicht waehrend eines Laufs) und
# laufende Arbeit abbrechen — je ueber denselben Endpunkt wie der Knopf der Oberflaeche. Wirkt nur mit AGENT_UEBERALL und
# AGENT_KONTO. INKLUAGENT_KONTO_WERKZEUGE=an.
AGENT_KONTO_WERKZEUGE = _umgebung_an("INKLUAGENT_KONTO_WERKZEUGE")
# Runde 2, Schritt 6 — Antworten als Bloecke: Absaetze und echte Listen statt eines Textblocks (Konzept Abschnitt 6, Frage 13:
# „gerade Projektlisten werden so fuer VoiceOver navigierbar“). Der Server macht aus Aufzaehlungen des Modells „• “-Zeilen
# (statt sie zu loeschen), die Seite baut daraus p/ul/ol mit textContent — kein Markdown, kein innerHTML. INKLUAGENT_BLOECKE=an.
AGENT_BLOECKE = _umgebung_an("INKLUAGENT_BLOECKE")
# Gespraeche (Steve 12.10.2026): Knopf „Neues Gespräch“ und Link „Frühere Gespräche“ im InkluAgent, eigene Seite mit schlichter
# Liste (neueste zuerst) und Lese-Seite mit „Hier weitermachen“; auf Seiten ohne Projekt findet der Agent fruehere Gespraeche
# zu einem Projekt und verlinkt sie. Braucht AGENT_KONTO. INKLUAGENT_GESPRAECHE=an.
AGENT_GESPRAECHE = _umgebung_an("INKLUAGENT_GESPRAECHE")
# Agentenansicht wie ChatGPT/Claude mit Arbeitsfenster (Michael Karbe, Feedback 20261012 - 1, Punkt 6; Steves Vision vom
# 12.10.2026): in der Agentenansicht Gespraech in der Mitte (Eingabe unten), rechts das Arbeitsfenster (Stand zum Nachlesen,
# Schritte, Vorschau des Bildes, an dem der Agent arbeitet), in der Seitenleiste der Abschnitt „Gespräche“. Die manuelle Ansicht
# bleibt, wie sie ist. Braucht AGENT_ANSICHT und AGENT_UEBERALL. INKLUAGENT_ARBEITSFENSTER=an.
AGENT_ARBEITSFENSTER = _umgebung_an("INKLUAGENT_ARBEITSFENSTER")
# Begleiter durch InkluDocs (Steve 12.10.2026): der InkluAgent bringt den Nutzer zu jeder Stelle der Oberflaeche („Bring mich zu
# den Einstellungen“), weiss, auf welcher Seite er ist („Was kann ich hier machen?“), und aendert auf ausdruecklichen Wunsch die
# Sprache der Oberflaeche und die Ansicht (inkluagent/begleiter.py). Braucht AGENT_UEBERALL. INKLUAGENT_BEGLEITER=an.
AGENT_BEGLEITER = _umgebung_an("INKLUAGENT_BEGLEITER")
# Klangspur des InkluAgent (Paket 1, Steve 13.10.2026): Zustaende hoerbar statt angesagt — ein Wusch, wenn eine Karte ins Gespraech
# fliegt (von rechts in die Mitte, wie die Karte), ein Tick je erledigtem Schritt, ein Klang, wenn er fertig ist oder etwas schiefging.
# Erzeugt im Browser (frontend/klang.js), spaeter Steves eigene Klaenge. Abschaltbar je Konto („Töne“ in den Einstellungen,
# inkluagent/klang.py). Braucht AGENT_UEBERALL. INKLUAGENT_KLANG=an.
AGENT_KLANG = _umgebung_an("INKLUAGENT_KLANG")
# Formular-Karten (Paket 1, Steve 13.10.2026: „der Agent bringt die Knöpfe zu einem“): statt auf eine Seite zu schicken, legt der
# InkluAgent das Formular vorbelegt ins Gespraech — „Zur Prüfung einladen“, „Zahlungspflichtig bestellen“ (Express). Abschicken
# nur ueber den Knopf der Karte, ueber DIESELBE Funktion wie die Seite (inkluagent/karten.py). INKLUAGENT_KARTEN=an.
AGENT_KARTEN = _umgebung_an("INKLUAGENT_KARTEN")
# Rueckmeldung (Steve 14.10.2026, erster Baustein der Auftraege: „er meldet sich, wenn er fertig ist“): startet der InkluAgent
# einen Lauf im Hintergrund (Alt-Texte, Quickinfos, Tagging, Kette, KI-Pruefung, Korrektur, Testlauf, Uebersetzung), passt der
# Server ohne KI auf und schreibt, wenn er zu Ende ist, eine persoenliche Nachricht des Agenten ins Gespraech; auf Wunsch
# dieselbe Nachricht per Mail an die Adresse des eigenen Kontos (inkluagent/rueckmeldung.py). Braucht AGENT_KONTO; die Seite
# meldet sie live mit AGENT_UEBERALL (GET /api/agent/status). INKLUAGENT_RUECKMELDUNG=an.
AGENT_RUECKMELDUNG = _umgebung_an("INKLUAGENT_RUECKMELDUNG")

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
