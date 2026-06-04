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
ohne es zu verbergen: Status auf IN_VORBEREITUNG setzen.
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
