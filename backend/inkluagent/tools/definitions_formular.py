"""Werkzeugsatz des InkluAgent fuer Formular-Projekte (Quickinfos), 28.08.2026.

Gegenstueck zu definitions.py (Bild-Projekte). Der Agent-Loop waehlt den Satz
nach project.tool: "formular" -> dieser Satz + prompts/system_formular.py,
sonst Bild-Satz + prompts/system_agent.py. Beide Saetze teilen tavily_search.

Aufbau je Werkzeug wie bei den Bildern: Anthropic-Tool-Schema (Claude bestimmt
nur fachliche Argumente wie feld_id, Text, Beleg) + Dispatcher, der project_id
und user_id aus dem Sitzungskontext injiziert.
"""
from __future__ import annotations

from typing import Any, Callable

from . import formular as formular_tools
from . import search as search_tools
from .definitions import TOOL_DEFINITIONS as _BILD_DEFINITIONS

_TAVILY = next(t for t in _BILD_DEFINITIONS if t["name"] == "tavily_search")

TOOL_DEFINITIONS_FORMULAR: list[dict] = [
    {
        "name": "list_form_fields",
        "description": (
            "Uebersicht aller Formularfelder im aktuellen Projekt: feld_id, ui_label (so heisst das Feld in der "
            "Oberflaeche: 'Feld 3' bzw. 'Dokument 2, Feld 3'), Feldart, Seite, Beschriftung, Abschnitt, Pflicht, "
            "aktuelle Quickinfo, Status (offen/beschrieben), Quelle (PDF/Hand/Stammdaten/KI/Gast), Sicherheit, "
            "Pruefstatus des Gastes. Nutze das zu Beginn und immer, wenn du eine feld_id brauchst. Keine Argumente."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_field_details",
        "description": (
            "Alles zu EINEM Feld: Beschriftung mit Lage (links/oben/rechts/innen), Abschnitt, Umfeldtext, Optionen "
            "(bei Auswahlfeldern), Original-Quickinfo aus der PDF, Beleg und Hinweise des Feld-Passes, Anmerkung des "
            "Gastes, technischer Feldname und der komplette Seitentext als Kontext. Brauchst du, bevor du eine "
            "Quickinfo formulierst oder bewertest — der Seitentext ist die Grundlage fuer den Beleg."
        ),
        "input_schema": {"type": "object", "properties": {
            "feld_id": {"type": "integer", "description": "Die feld_id (aus list_form_fields)."}},
            "required": ["feld_id"]},
    },
    {
        "name": "view_field",
        "description": (
            "Zeigt dir im NAECHSTEN Turn den Bild-Ausschnitt des Feldes (Feld mit Umgebung, ohne eingetragene "
            "Werte) oder mit ganze_seite=true die ganze Formularseite mit nummerierten Feldrahmen. Nutze das, wenn "
            "der Seitentext nicht reicht (z. B. Beschriftung unklar, Tabellenlayout, Kaestchen-Gruppen). "
            "Bilder bleiben NICHT ueber Turns hinweg im Kontext — bei Bedarf erneut aufrufen."
        ),
        "input_schema": {"type": "object", "properties": {
            "feld_id": {"type": "integer", "description": "Die feld_id."},
            "ganze_seite": {"type": "boolean", "description": "true = Seitenansicht statt Ausschnitt. Standard false."}},
            "required": ["feld_id"]},
    },
    {
        "name": "generate_quickinfo",
        "description": (
            "Laesst den Feld-Pass von InkluDocs (Sonnet, Seitentext mit Positionen, deterministische Nachpruefung) "
            "eine Quickinfo fuer EIN Feld neu erzeugen und speichert sie sofort (quelle KI, Sicherheit hoch/mittel/"
            "niedrig, Beleg). Identisch zum Knopf 'Generieren' in der Oberflaeche, ueberschreibt bewusst. "
            "Nutze das bei 'generieren', 'neu generieren', 'lass die KI vorschlagen'. 1 Credit. "
            "Fuer redaktionelle Aenderungen (kuerzer, anders formuliert, Gruppe voranstellen) formulierst DU selbst "
            "und speicherst mit update_quickinfo."
        ),
        "input_schema": {"type": "object", "properties": {
            "feld_id": {"type": "integer", "description": "Die feld_id."}},
            "required": ["feld_id"]},
    },
    {
        "name": "update_quickinfo",
        "description": (
            "Speichert eine vom Nutzer abgenommene Quickinfo fuer ein Feld. NUR nach klarer Zustimmung ('ja speichern', "
            "'uebernehmen', 'passt so'). Vor dem Speichern laeuft dieselbe Nachpruefung wie im Feld-Pass: Der Parameter "
            "beleg muss die WOERTLICHE Textstelle der Formularseite sein, aus der die Quickinfo folgt (Beschriftung "
            "neben dem Feld, Abschnittsueberschrift) — hole sie aus get_field_details (seitentext). Ohne belegbaren "
            "Text wird NICHT gespeichert; du bekommst die Hinweise zurueck und legst sie dem Nutzer vor. Nur wenn der "
            "Nutzer ausdruecklich auf seiner Fassung besteht (er weiss etwas, das nicht auf der Seite steht), rufst du "
            "das Werkzeug erneut mit force=true auf. Ein Satz, hoechstens 200 Zeichen, keine Anleitung, keine Feldart. "
            "1 Credit."
        ),
        "input_schema": {"type": "object", "properties": {
            "feld_id": {"type": "integer", "description": "Die feld_id."},
            "new_quickinfo": {"type": "string", "description": "Die neue Quickinfo, ein Satz."},
            "beleg": {"type": "string", "description": "Woertliche Textstelle der Seite, die die Quickinfo belegt."},
            "force": {"type": "boolean", "description": "Nur nach Beanstandung UND ausdruecklichem Beharren des Nutzers. Standard false."}},
            "required": ["feld_id", "new_quickinfo"]},
    },
    {
        "name": "revert_quickinfo",
        "description": (
            "Setzt das Feld auf die Original-Quickinfo aus der PDF zurueck (leer, wenn die PDF keine hatte). "
            "Nutze das bei 'zurueck auf Original', 'rueckgaengig', 'nimm wieder das Original'. Kostenlos."
        ),
        "input_schema": {"type": "object", "properties": {
            "feld_id": {"type": "integer", "description": "Die feld_id."}},
            "required": ["feld_id"]},
    },
    {
        "name": "search_master_data",
        "description": (
            "Sucht in den Stammdaten des Kontos (Beschriftung, Feldname, Quickinfo — Teiltreffer). Banken und "
            "Versicherungen haben viele Formulare mit gleichen Feldern; hier steht der abgestimmte Wortlaut. "
            "Nutze das, bevor du fuer ein haeufiges Feld (Name, Geburtsdatum, IBAN, Anschrift) einen eigenen Text "
            "formulierst. Leere Anfrage = alle Eintraege (max. 20)."
        ),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Suchbegriff, z. B. 'Geburtsdatum' oder 'IBAN'."},
            "feld_art": {"type": "string", "description": "Optional: nur diese Feldart (text, checkbox, radio, dropdown, liste, signatur)."}},
            "required": ["query"]},
    },
    {
        "name": "save_to_master_data",
        "description": (
            "Nimmt die aktuelle Quickinfo eines Feldes in die Stammdaten des Kontos auf (wie der Knopf 'In Stammdaten "
            "uebernehmen'; gleicher Schluessel Beschriftung+Feldart wird aktualisiert, keine Dublette). Nur auf Wunsch "
            "des Nutzers ('merk dir das', 'in die Stammdaten'). Kostenlos."
        ),
        "input_schema": {"type": "object", "properties": {
            "feld_id": {"type": "integer", "description": "Die feld_id."}},
            "required": ["feld_id"]},
    },
    _TAVILY,
]


class ToolExecutorFormular:
    """Dispatcher fuer den Formular-Werkzeugsatz — project_id/user_id aus dem Sitzungskontext."""

    def __init__(self, project_id: int, user_id: int) -> None:
        self.project_id = project_id
        self.user_id = user_id

    def execute(self, name: str, args: dict) -> dict[str, Any]:
        try:
            handler = self._handlers().get(name)
            if not handler:
                return {"ok": False, "error": f"Unbekanntes Werkzeug: {name}"}
            return handler(args)
        except Exception as e:
            return {"ok": False, "error": f"Werkzeug-Ausfuehrung crashte: {e}"}

    def _handlers(self) -> dict[str, Callable[[dict], dict]]:
        p, u = self.project_id, self.user_id
        return {
            "list_form_fields": lambda _a: formular_tools.list_form_fields(p, u),
            "get_field_details": lambda a: formular_tools.get_field_details(int(a["feld_id"]), p, u),
            "view_field": lambda a: formular_tools.view_field(int(a["feld_id"]), p, u, ganze_seite=bool(a.get("ganze_seite", False))),
            "generate_quickinfo": lambda a: formular_tools.generate_quickinfo(int(a["feld_id"]), p, u),
            "update_quickinfo": lambda a: formular_tools.update_quickinfo(
                int(a["feld_id"]), p, u, str(a.get("new_quickinfo", "")), beleg=str(a.get("beleg", "") or ""),
                force=bool(a.get("force", False))),
            "revert_quickinfo": lambda a: formular_tools.revert_quickinfo(int(a["feld_id"]), p, u),
            "search_master_data": lambda a: formular_tools.search_master_data(str(a.get("query", "")), p, u,
                                                                              feld_art=str(a.get("feld_art", "") or "")),
            "save_to_master_data": lambda a: formular_tools.save_to_master_data(int(a["feld_id"]), p, u),
            "tavily_search": lambda a: search_tools.tavily_search(
                str(a["query"]), max_results=int(a.get("max_results", 5)), include_domains=a.get("include_domains")),
        }
