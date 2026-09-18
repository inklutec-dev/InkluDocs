"""Werkzeug-Registry für InkluDocs.

Zentrale, einzige Definition aller im Dashboard angebotenen Werkzeuge
("Funktionen"). Das Dashboard rendert daraus die Werkzeug-Auswahl, und der
Projekt-Anlege-Endpunkt validiert die gewählte Kennung gegen diese Liste.

EIN NEUES WERKZEUG HINZUFÜGEN:
    1. Unten in TOOLS einen Tool(...)-Eintrag ergänzen.
    2. Status auf ToolStatus.IN_VORBEREITUNG lassen, solange die Arbeitsfläche fehlt.
    3. Sobald die Seite unter `route` existiert, Status auf ToolStatus.VERFUEGBAR setzen.
Mehr ist nicht nötig – Dashboard und Anlege-Dialog ziehen automatisch nach.

Ein Werkzeug entfernen: Eintrag löschen. Ein Werkzeug vorübergehend sperren,
ohne es zu verbergen: Status auf IN_VORBEREITUNG setzen. Ein Werkzeug aus dem
Anlege-Menü nehmen, aber die Kennung für alte Projekte und die API behalten:
sichtbar=False.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ToolStatus(str, Enum):
    """Lebenszyklus eines Werkzeugs. Interne Kennungen (ASCII), das UI zeigt
    stattdessen die Klartext-Etiketten aus STATUS_LABEL."""
    VERFUEGBAR = "verfuegbar"
    BETA = "beta"
    IN_VORBEREITUNG = "in_vorbereitung"


# Menschlich lesbares Etikett pro Status (wird im UI angezeigt).
STATUS_LABEL = {
    ToolStatus.VERFUEGBAR: "Verfügbar",
    ToolStatus.BETA: "Beta",
    ToolStatus.IN_VORBEREITUNG: "In Vorbereitung",
}


@dataclass(frozen=True)
class Tool:
    """Ein im Dashboard angebotenes Werkzeug."""
    key: str          # stabile Kennung, landet in projects.tool – NIE ändern
    name: str         # Anzeigename im Dashboard / Auswahlmenü
    description: str  # ein Satz Erklärung
    route: str        # Ziel-Adresse der Arbeitsfläche ("" wenn noch keine)
    status: ToolStatus
    sichtbar: bool = True  # False: nicht im Anlege-Menü, Kennung bleibt gültig (alte Projekte, API)

    @property
    def is_available(self) -> bool:
        """Anlegbar? (verfügbar oder beta, aber nicht in Vorbereitung)"""
        return self.status in (ToolStatus.VERFUEGBAR, ToolStatus.BETA)

    @property
    def status_label(self) -> str:
        return STATUS_LABEL[self.status]


# --- Die Registry. Reihenfolge = Anzeige-Reihenfolge im Dashboard. ---
# Hinweis: route zeigt aktuell noch auf die gemeinsame Arbeitsfläche /app.
# Sobald die eigenen Modul-Seiten existieren (Service-Trennung Teil 3/4),
# werden die routes auf /app/pdf, /app/web, /app/grafik umgestellt.
TOOLS: list[Tool] = [
    Tool(
        key="pdf",
        name="Alt-Texte für PDFs",
        description="PDF-Dokument hochladen und enthaltene Bilder mit barrierefreien Alt-Texten versehen.",
        route="/app",
        status=ToolStatus.VERFUEGBAR,
    ),
    Tool(
        key="web",
        name="Alt-Texte für Webseiten",
        description="Eine Webseite über ihre Adresse scannen und ihre Bilder mit Alt-Texten versehen.",
        route="/app",
        status=ToolStatus.VERFUEGBAR,  # scharfgeschaltet 04.06.2026: scan_url an Projekte angebunden
    ),
    Tool(
        key="grafik",
        name="Alt-Texte für Grafiken",
        description="Einzelne Bilder hochladen und mit barrierefreien Alt-Texten versehen.",
        route="/app",
        status=ToolStatus.VERFUEGBAR,
    ),
    # WORD-WERKZEUG (26.08.2026, Steve + Michael): eigenes Werkzeug, NICHT ins
    # PDF-Werkzeug (Entscheidung 14.08.2026). Backend: docx_processor.py (lesen)
    # + docx_export.py (zurueckschreiben), Doku docs/WORD.md.
    # TESTUMBAU 18.09.2026 (Steve): Projekt = Dateityp, Faehigkeit = Ansicht. Ein Word-
    # Projekt hat die Ansichten „Alt-Texte" und „Uebersetzung" (Zeile „Ansicht" im
    # Projektkopf, app.html ansichtWahlHtml). Deshalb heisst das Werkzeug nach der
    # Datei-Art „Word-Dokumente", nicht mehr nach einer Faehigkeit (Steve 18.09.2026:
    # „Wir nehmen nur das eine Werkzeug"). Doku docs/UEBERSETZEN.md, Abschnitt Testumbau.
    Tool(
        key="word",
        name="Word-Dokumente",
        description="Word-Datei (.docx) hochladen, die Bilder mit Alt-Texten versehen, den Text übersetzen lassen und die Datei als Word oder barrierefreie PDF herunterladen.",
        route="/app",
        status=ToolStatus.VERFUEGBAR,  # Beta-Etikett entfällt seit 09.09.2026 (Steve)
    ),
    # QUICKINFO-WERKZEUG (27.08.2026, Steve + Michael Karbe/Joerg Heine, Actino):
    # PDF-Formulare — jedes Eingabefeld bekommt eine Quickinfo (/TU), den Text,
    # den Screenreader beim Erreichen des Feldes vorlesen. Eigene Tabellen
    # (formularfelder, stammdaten), eigener Router formular_api.py, Leser
    # formular_processor.py, Schreiber formular_export.py, Doku docs/FORMULAR.md.
    # Beta, bis echte Kundenformulare durch sind (KI-Vorschlaege = Stufe 2).
    Tool(
        key="formular",
        name="Quickinfos für PDF-Formulare",
        description="PDF-Formular hochladen, jedes Eingabefeld mit einer Quickinfo (Hilfetext für Screenreader) versehen, Stammdaten für künftige Formulare speichern und die PDF mit Quickinfos herunterladen.",
        route="/app",
        status=ToolStatus.VERFUEGBAR,  # Beta-Etikett entfällt seit 09.09.2026 (Steve)
    ),
    # UEBERSETZEN (18.09.2026, Anlass Mark Hounschild): Kern uebersetzung.py (Segmente/
    # Marken/byteidentischer Rueckschreiber), Router uebersetzung_api.py, Ansicht
    # frontend/uebersetzen.js, Doku docs/UEBERSETZEN.md. Seit dem Testumbau 18.09.2026
    # KEIN eigener Eintrag im Anlege-Menue mehr (sichtbar=False): Uebersetzen ist die
    # Ansicht „Uebersetzung" jedes Word-Projekts. Die Kennung bleibt fuer bestehende
    # Projekte und die Public API gueltig; ein so angelegtes Projekt ist ein Word-Projekt
    # (Dateityp docx), das in der Ansicht Uebersetzung startet.
    Tool(
        key="uebersetzen",
        name="Word-Dokumente (Start: Übersetzung)",
        description="Wie „Word-Dokumente“, öffnet nach dem Anlegen die Ansicht Übersetzung. Nur über die API anlegbar.",
        route="/app",
        status=ToolStatus.VERFUEGBAR,
        sichtbar=False,
    ),
    Tool(
        key="pdf-a11y",
        name="Barrierefreie PDFs erstellen",
        description="Aus bestehenden Dokumenten barrierefreie, getaggte PDFs erzeugen.",
        route="",
        status=ToolStatus.IN_VORBEREITUNG,
    ),
    # Sammel-Werkzeug vor der Aufteilung. Nicht mehr wählbar, nur als Anzeige-
    # Label fuer evtl. nicht zugeordnete Altprojekte (Migration deckt alle ab).
    Tool(
        key="alttext",
        name="Alt-Texte (ältere Sammel-Projekte)",
        description="Frühere Projekte vor der Aufteilung in einzelne Werkzeuge.",
        route="/app",
        status=ToolStatus.IN_VORBEREITUNG,
    ),
]

# Schneller Zugriff per Kennung.
TOOLS_BY_KEY: dict[str, Tool] = {t.key: t for t in TOOLS}


def get_tool(key: str) -> Tool | None:
    """Werkzeug per Kennung holen, oder None wenn unbekannt."""
    return TOOLS_BY_KEY.get(key)


def is_valid_tool_key(key: str) -> bool:
    """True, wenn key ein bekanntes, anlegbares Werkzeug ist (nicht in Vorbereitung)."""
    tool = TOOLS_BY_KEY.get(key)
    return tool is not None and tool.is_available
