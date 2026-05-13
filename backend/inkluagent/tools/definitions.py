"""Anthropic-Tool-Definitions im Bedrock-Claude-Format
(https://docs.anthropic.com/en/docs/build-with-claude/tool-use).

Jedes Tool hier korrespondiert mit einer Funktion in project.py/altext.py/search.py.
Project-Context (project_id, user_id) wird NICHT von Claude bestimmt sondern
zur Laufzeit vom ToolExecutor injected — Claude darf nur funktionale Args
angeben (image_id, query, ...).
"""
from __future__ import annotations

from typing import Any, Callable

from . import project as project_tools
from . import altext as altext_tools
from . import search as search_tools


TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "list_project_images",
        "description": (
            "Gibt eine Übersicht aller Bilder im aktuellen Projekt zurück mit Metadaten "
            "(image_id, page, image_type, alt_text, needs_review, Konfidenz, Bildmaße). "
            "Nutze das zu Beginn der Konversation oder wenn der User pauschal über "
            "das Projekt spricht ('Wie viele Bilder?', 'Welche brauchen Review?'). "
            "Keine Args nötig."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_image_metadata",
        "description": (
            "Detail-Infos zu einem einzelnen Bild: aktueller Alt-Text + Langbeschreibung + "
            "Validation-Result + Pipeline-Steps + Kontext-Text. Brauche das wenn du einen "
            "konkreten Alt-Text vor dem Ändern lesen willst, oder den Validation-Status prüfen "
            "musst."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "Die image_id (aus list_project_images).",
                },
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "view_image",
        "description": (
            "Lädt die Bilddatei und zeigt sie dir im NÄCHSTEN Turn als image-content-Block. "
            "Brauche das wenn du den Alt-Text inhaltlich beurteilen oder ändern sollst — "
            "ohne das Bild gesehen zu haben kannst du keine fundierte Modifikation vorschlagen. "
            "Eine einzige view_image-Anfrage pro Konversation reicht für ein Bild — "
            "danach bleibt es in deinem visuellen Kontext."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "Die image_id des Bildes zum Anschauen.",
                },
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "generate_alt_text",
        "description": (
            "Ruft die InkluDocs-Pipeline auf, um einen neuen Alt-Text + Langbeschreibung "
            "für ein Bild komplett neu zu generieren (force_regenerate=True, umgeht Cache). "
            "Nutze das wenn der User explizit 'neu generieren' will. Die Pipeline ist "
            "Claude Sonnet 4.6 via Bedrock und nutzt die Premium-Builder. Dauer ~15-30 Sek."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "Die image_id des Bildes für Neu-Generierung.",
                },
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "update_alt_text",
        "description": (
            "Speichert einen vom User abgenommenen Alt-Text + optional Langbeschreibung in der DB. "
            "Alt-Text geht in Spalte 'alt_text_edited' (Original-Pipeline-Ausgabe bleibt in 'alt_text' "
            "erhalten — Rollback via revert_alt_text möglich). "
            "Langbeschreibung überschreibt direkt das langbeschreibung-Feld (kein _edited-Feld). "
            "Nur aufrufen wenn der User klar bestätigt hat ('ja speichern', 'übernehmen', 'passt so'). "
            "Alt-Text-Validierung: 5-500 Zeichen, kein 'Bild von...'/'Foto von...'-Präfix."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "Die image_id des Bildes.",
                },
                "new_alt_text": {
                    "type": "string",
                    "description": "Der neue Alt-Text, BITV-konform, präzise, ohne 'Bild von...'-Präfix.",
                },
                "new_langbeschreibung": {
                    "type": "string",
                    "description": (
                        "Optional: neue Langbeschreibung. Wenn nicht gesetzt, bleibt die bestehende "
                        "Langbeschreibung erhalten. Sinnvoll wenn der User eine konkrete inhaltliche "
                        "Änderung wünscht oder du beide Texte gleichzeitig überarbeitest."
                    ),
                },
            },
            "required": ["image_id", "new_alt_text"],
        },
    },
    {
        "name": "revert_alt_text",
        "description": (
            "Setzt die manuelle Alt-Text-Bearbeitung zurück (alt_text_edited = NULL). "
            "Das Frontend zeigt danach wieder den Pipeline-Original-Alt-Text. "
            "Nutze das wenn der User sagt 'nimm wieder das Original', 'mach das rückgängig', "
            "'verwerfe meine Änderung'. Achtung: Langbeschreibung wird NICHT zurückgesetzt — "
            "wenn das auch zurück soll, generate_alt_text neu aufrufen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "Die image_id des Bildes für Rollback.",
                },
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "tavily_search",
        "description": (
            "Web-Suche via Tavily — gibt KI-Zusammenfassung + Top-Treffer mit Snippets zurück. "
            "Nutze das für:\n"
            "- BITV/WCAG/EN-301-549-Recherche (aktuelle Stände, nicht aus deinem Training)\n"
            "- Eigennamen-Verifikation (Personen, Orte, Produkte, Logos)\n"
            "- Fachbegriffe aus speziellen Domänen (Medizin, Recht, Architektur)\n"
            "- Aktuelle Ereignisse oder Daten\n"
            "Tipp: include_domains=['w3.org','bitv-test.de','bik-fuer-alle.de'] für gezielten "
            "Barrierefreiheits-Lookup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Die Suchanfrage in natürlicher Sprache.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Anzahl Treffer (1-10, Default 5).",
                    "default": 5,
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: Liste von Domains zum Filtern (z.B. ['w3.org']).",
                },
            },
            "required": ["query"],
        },
    },
]


class ToolExecutor:
    """Führt Tool-Calls aus mit injected Project-/User-Context.

    Claude darf nur funktionale Args bestimmen — user_id+project_id
    werden hier sicher aus dem Session-Kontext genommen, nie aus den
    Claude-Args, damit kein Cross-Projekt-Zugriff möglich ist.
    """

    def __init__(self, project_id: int, user_id: int) -> None:
        self.project_id = project_id
        self.user_id = user_id

    def execute(self, name: str, args: dict) -> dict[str, Any]:
        try:
            handler = self._handlers().get(name)
            if not handler:
                return {"ok": False, "error": f"Unbekanntes Tool: {name}"}
            return handler(args)
        except Exception as e:
            return {"ok": False, "error": f"Tool-Ausführung crashte: {e}"}

    def _handlers(self) -> dict[str, Callable[[dict], dict]]:
        p, u = self.project_id, self.user_id
        return {
            "list_project_images": lambda _a: project_tools.list_project_images(p, u),
            "get_image_metadata": lambda a: project_tools.get_image_metadata(int(a["image_id"]), p, u),
            "view_image": lambda a: project_tools.view_image(int(a["image_id"]), p, u),
            "generate_alt_text": lambda a: altext_tools.generate_alt_text(int(a["image_id"]), p, u),
            "update_alt_text": lambda a: altext_tools.update_alt_text(
                int(a["image_id"]), p, u,
                str(a.get("new_alt_text", "")),
                a.get("new_langbeschreibung") if a.get("new_langbeschreibung") is not None else None,
            ),
            "revert_alt_text": lambda a: altext_tools.revert_alt_text(
                int(a["image_id"]), p, u,
            ),
            "tavily_search": lambda a: search_tools.tavily_search(
                str(a["query"]),
                max_results=int(a.get("max_results", 5)),
                include_domains=a.get("include_domains"),
            ),
        }
