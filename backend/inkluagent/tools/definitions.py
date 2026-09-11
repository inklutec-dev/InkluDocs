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
from . import ausgaben as ausgaben_tools


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
            "Nutze das wenn der User explizit 'neu generieren' will. Die Pipeline ist dieselbe "
            "wie beim Upload (mehrstufig, mit Bildtyp-Erkennung und Stilregeln). Dauer ~15-30 Sek."
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
            "Alt-Text-Validierung: 5-500 Zeichen, kein 'Bild von...'/'Foto von...'-Präfix. "
            "Vor dem Speichern prüft derselbe Bild-Verify wie in der Pipeline den Text gegen das Bild. "
            "Bei einer Beanstandung wird NICHT gespeichert und du bekommst die strittigen Aussagen plus "
            "ggf. einen Korrektur-Vorschlag zurück — lege beides dem User vor. Nur wenn der User "
            "ausdrücklich auf seiner Fassung besteht (er weiß z.B. etwas, das im Bild nicht sichtbar ist), "
            "rufst du das Tool erneut mit force=true auf."
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
                "force": {
                    "type": "boolean",
                    "description": (
                        "Nur nach einer Verify-Beanstandung UND ausdrücklichem Beharren des Users: "
                        "true speichert ohne erneute Bild-Prüfung. Standard: false."
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


# ─── Word-Projekte: Meine Ausgaben (Schritt 2, 11.09.2026) ───
# Nur fuer Word-Projekte (project_type docx) an TOOL_DEFINITIONS angehaengt (agent_loop._werkzeugsatz).
# Kostenpflichtige Werkzeuge verlangen bestaetigt=true — der SERVER liefert beim ersten Aufruf nur
# Preis + Guthaben zurueck, die Rueckfrage an den Nutzer ist damit erzwungen, nicht nur erbeten.
TOOL_DEFINITIONS_WORD: list[dict] = [
    {
        "name": "pruefe_word_dokument",
        "description": (
            "Prüfbericht des Word-Dokuments mit den aktuellen Alt-Texten (Titel, Sprache, Überschriften-"
            "Hierarchie, Tabellenköpfe, Bilder ohne Alt-Text) plus ein Auszug der Hörprobe (was ein "
            "Screenreader liest). Kostenlos. Immer der erste Schritt vor einer Umwandlung. Ohne document_id "
            "alle Dokumente des Projekts."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional: nur dieses Dokument (documents[].id aus list_project_images)."},
        }, "required": []},
    },
    {
        "name": "konvertiere_zu_pdfua",
        "description": (
            "Wandelt das Word-Dokument mit den aktuellen Alt-Texten in eine barrierefreie PDF (PDF/UA) um "
            "und prüft sie (veraPDF). Kostet Credits. ZWEI SCHRITTE: Erster Aufruf OHNE bestaetigt liefert nur "
            "Preis und Guthaben (rueckfrage_noetig) — nenne dem Nutzer den Preis und frage. Erst nach seinem "
            "klaren Ja erneut mit bestaetigt=true aufrufen. Ergebnis: ausgabe_id, Zusammenfassung, Bereiche mit "
            "Hinweisen; die Datei liegt unter „Meine Ausgaben“ und der Nutzer sieht unter deiner Antwort einen "
            "Download-Knopf. Ohne document_id alle Dokumente (ZIP)."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional: nur dieses Dokument."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers zum genannten Preis. Standard false."},
        }, "required": []},
    },
    {
        "name": "exportiere_word",
        "description": (
            "Gibt die Word-Datei mit den aktuellen Alt-Texten aus (Eintrag unter „Meine Ausgaben“, Download-Knopf "
            "unter deiner Antwort). Kostet Credits. Gleiche zwei Schritte wie konvertiere_zu_pdfua: erst ohne "
            "bestaetigt (Preis nennen, fragen), dann mit bestaetigt=true."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional: nur dieses Dokument."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers. Standard false."},
        }, "required": []},
    },
    {
        "name": "liste_ausgaben",
        "description": (
            "Alle fertigen Ausgaben dieses Projekts (barrierefreie PDFs, Word-Dateien) mit Datum, Prüfstand und "
            "Verfügbarkeit der Datei — das Regal „Meine Ausgaben“. Keine Args."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "lies_ausgabe",
        "description": (
            "Liest zu einer Ausgabe (ausgabe_id aus liste_ausgaben oder konvertiere_zu_pdfua) den Bericht: "
            "teil=bericht (Prüfung je Bereich in Klartext + Prüfbericht des Word-Dokuments), teil=pruefbericht "
            "(nur Word-Prüfbericht), teil=hoerprobe (vollständige Hörprobe, Zeile für Zeile), teil=alles."
        ),
        "input_schema": {"type": "object", "properties": {
            "ausgabe_id": {"type": "integer", "description": "Die ausgabe_id."},
            "teil": {"type": "string", "enum": ["bericht", "pruefbericht", "hoerprobe", "alles"], "description": "Standard bericht."},
        }, "required": ["ausgabe_id"]},
    },
]


class ToolExecutor:
    """Führt Tool-Calls aus mit injected Project-/User-Context.

    Claude darf nur funktionale Args bestimmen — user_id+project_id
    werden hier sicher aus dem Session-Kontext genommen, nie aus den
    Claude-Args, damit kein Cross-Projekt-Zugriff möglich ist.
    """

    def __init__(self, project_id: int, user_id: int, word: bool = False) -> None:
        self.project_id = project_id
        self.user_id = user_id
        self.word = word   # Word-Projekt: Werkzeuge „Meine Ausgaben“ freigeschaltet (11.09.2026)

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
        handlers: dict[str, Callable[[dict], dict]] = {
            "list_project_images": lambda _a: project_tools.list_project_images(p, u),
            "get_image_metadata": lambda a: project_tools.get_image_metadata(int(a["image_id"]), p, u),
            "view_image": lambda a: project_tools.view_image(int(a["image_id"]), p, u),
            "generate_alt_text": lambda a: altext_tools.generate_alt_text(int(a["image_id"]), p, u),
            "update_alt_text": lambda a: altext_tools.update_alt_text(
                int(a["image_id"]), p, u,
                str(a.get("new_alt_text", "")),
                a.get("new_langbeschreibung") if a.get("new_langbeschreibung") is not None else None,
                force=bool(a.get("force", False)),
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
        if self.word:
            def _doc(a):
                return int(a["document_id"]) if a.get("document_id") not in (None, "", 0) else None
            handlers.update({
                "pruefe_word_dokument": lambda a: ausgaben_tools.pruefe_word_dokument(p, u, _doc(a)),
                "konvertiere_zu_pdfua": lambda a: ausgaben_tools.konvertiere_zu_pdfua(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False))),
                "exportiere_word": lambda a: ausgaben_tools.exportiere_word(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False))),
                "liste_ausgaben": lambda _a: ausgaben_tools.liste_ausgaben(p, u),
                "lies_ausgabe": lambda a: ausgaben_tools.lies_ausgabe(p, u, int(a["ausgabe_id"]), str(a.get("teil") or "bericht")),
            })
        return handlers
