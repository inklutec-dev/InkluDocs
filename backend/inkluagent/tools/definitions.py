"""Anthropic-Tool-Definitions im Bedrock-Claude-Format
(https://docs.anthropic.com/en/docs/build-with-claude/tool-use).

Jedes Tool hier korrespondiert mit einer Funktion in project.py/altext.py/search.py.
Project-Context (project_id, user_id) wird NICHT von Claude bestimmt sondern
zur Laufzeit vom ToolExecutor injected — Claude darf nur funktionale Args
angeben (image_id, query, ...).
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from . import project as project_tools
from . import altext as altext_tools
from . import search as search_tools
from . import ausgaben as ausgaben_tools
from . import pdf as pdf_tools   # PDF-Werkzeuge (Werkzeugsatz nach Dateiart, 22.09.2026)


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


# Word-Werkzeuge: eigene Datei (22.09.2026), hier nur weitergereicht (agent_loop importiert von hier).
from .definitions_word import TOOL_DEFINITIONS_WORD  # noqa: E402,F401


class ToolExecutor:
    """Führt Tool-Calls aus mit injected Project-/User-Context.

    Claude darf nur funktionale Args bestimmen — user_id+project_id
    werden hier sicher aus dem Session-Kontext genommen, nie aus den
    Claude-Args, damit kein Cross-Projekt-Zugriff möglich ist.
    """

    def __init__(self, project_id: int, user_id: int, word: bool = False, pdf: bool = False) -> None:
        self.project_id = project_id
        self.user_id = user_id
        self.word = word   # Word-Projekt: Werkzeuge „Meine Ausgaben“ freigeschaltet (11.09.2026)
        self.pdf = pdf     # PDF-Projekt: Feld-Werkzeuge + PDF-Werkzeuge (Werkzeugsatz nach Dateiart, 22.09.2026)
        # Ein Executor je Nutzer-Nachricht (agent_loop): turn_id trennt Preisauskunft und Zustimmung,
        # kostenpflichtig zaehlt bezahlte Aktionen dieser Nachricht (Review 12.09.2026, ausgaben._freigabe).
        self.turn_id = uuid.uuid4().hex
        self.kostenpflichtig = 0

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
                "konvertiere_zu_pdfua": lambda a: ausgaben_tools.konvertiere_zu_pdfua(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "exportiere_word": lambda a: ausgaben_tools.exportiere_word(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "analysiere_word_struktur": lambda a: ausgaben_tools.analysiere_word_struktur(p, u, _doc(a)),
                "uebersetze_dokument": lambda a: ausgaben_tools.uebersetze_dokument(p, u, str(a.get("zielsprache") or ""), bestaetigt=bool(a.get("bestaetigt", False)), alt_texte=bool(a.get("alt_texte", True)), turn=self),
                "uebersetzung_stand": lambda _a: ausgaben_tools.uebersetzung_stand(p, u),
                "exportiere_uebersetzung": lambda _a: ausgaben_tools.exportiere_uebersetzung(p, u),
                "liste_ausgaben": lambda _a: ausgaben_tools.liste_ausgaben(p, u),
                "lies_ausgabe": lambda a: ausgaben_tools.lies_ausgabe(p, u, int(a["ausgabe_id"]), str(a.get("teil") or "bericht")),
            })
        if self.pdf:
            # Werkzeugsatz nach Dateiart (22.09.2026): EIN Gespraech je PDF-Projekt ueber alle drei Stationen —
            # Bild-Werkzeuge + Feld-Werkzeuge (Quickinfos) + PDF-Werkzeuge (Tagging, Kette, Hoerprobe, Pruefung, Export).
            from .definitions_formular import ToolExecutorFormular   # spaet: definitions_formular importiert dieses Modul

            def _doc(a):
                return int(a["document_id"]) if a.get("document_id") not in (None, "", 0) else None
            for name, h in ToolExecutorFormular(project_id=p, user_id=u)._handlers().items():
                handlers.setdefault(name, h)
            handlers.update({
                "dokument_stand": lambda a: pdf_tools.dokument_stand(p, u, _doc(a)),
                "barrierefrei_machen": lambda a: pdf_tools.barrierefrei_machen(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "komplett_barrierefrei_machen": lambda a: pdf_tools.komplett_barrierefrei_machen(p, u, bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "hoerprobe_lesen": lambda a: pdf_tools.hoerprobe_lesen(p, u, _doc(a), von=int(a.get("von") or 1), anzahl=int(a.get("anzahl") or 80)),
                "pruefung_starten": lambda a: pdf_tools.pruefung_starten(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "pruefbericht_lesen": lambda a: pdf_tools.pruefbericht_lesen(p, u, _doc(a)),
                "exportiere_fertige_pdf": lambda a: pdf_tools.exportiere_fertige_pdf(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "dokument_umbenennen": lambda a: pdf_tools.dokument_umbenennen(p, u, _doc(a), str(a.get("name") or "")),
                "dokument_loeschen": lambda a: pdf_tools.dokument_loeschen(p, u, _doc(a), bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "alt_sprache_setzen": lambda a: pdf_tools.alt_sprache_setzen(p, u, str(a.get("sprache") or "")),
                "korrektur_anwenden": lambda a: pdf_tools.korrektur_anwenden(p, u, _doc(a), erneut_pruefen=bool(a.get("erneut_pruefen", False)), bestaetigt=bool(a.get("bestaetigt", False)), turn=self),
                "korrektur_rueckgaengig": lambda a: pdf_tools.korrektur_rueckgaengig(p, u, _doc(a)),
                "liste_ausgaben": lambda _a: ausgaben_tools.liste_ausgaben(p, u),
                "lies_ausgabe": lambda a: ausgaben_tools.lies_ausgabe(p, u, int(a["ausgabe_id"]), str(a.get("teil") or "bericht")),
            })
        return handlers
