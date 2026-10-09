# Werkzeugdefinitionen fuer die Funktionen der Oberflaeche, die der InkluAgent seit 30.09.2026 zusaetzlich hat (Steve:
# „alles, was man händisch macht, soll über den InkluAgent gehen“). Umsetzung: tools/oberflaeche.py (derselbe Kern wie der
# Knopf). Welche Projektart welche Werkzeuge bekommt: agent_loop._werkzeugsatz.
_BESTAETIGT = {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers in einer eigenen Nachricht. Standard false."}
_DOC = {"type": "integer", "description": "Optional: nur dieses Dokument (Liste über dokument_stand bzw. list_project_images)."}

_TESTWEISE = {
    "name": "testweise_taggen",
    "description": ("Wie „Testweise taggen“ in der Ansicht Tagging: kostenlos, im Testmodus, eigene Testfassung — das Dokument "
                    "bleibt unverändert. Die Testfassung trägt ein Wasserzeichen von PDFix und ist kostenlos herunterladbar "
                    "(Knopf unter der Antwort). Wartet auf das Ende; sonst Ergebnis über dokument_stand (testlauf)."),
    "input_schema": {"type": "object", "properties": {"document_id": _DOC}, "required": []},
}
_PRUEFDATEI_ERSTELLEN = {
    "name": "pruefdatei_erstellen",
    "description": ("Wie „Prüfung starten“ bzw. „Prüfung erneut starten“ in der Barrierefreiheitsprüfung: baut die fertige Datei (wie beim Herunterladen, mit "
                    "Alt-Texten und Quickinfos) und prüft sie mit veraPDF. Kostenlos, keine Ablage. Dauert einige Sekunden. "
                    "Liefert das Ergebnis wie pruefdatei_lesen. Nur für getaggte Dokumente."),
    "input_schema": {"type": "object", "properties": {"document_id": _DOC}, "required": []},
}
_PRUEFDATEI_LESEN = {
    "name": "pruefdatei_lesen",
    "description": ("Ergebnis der Barrierefreiheitsprüfung (Prüfdatei) zum Vorlesen: teil=probleme (Standard) = Urteil von veraPDF "
                    "und die Problemstellen mit Seiten; teil=hoerprobe = die Hörprobe der FERTIGEN Datei, seitenweise über "
                    "von/anzahl. Kostenlos."),
    "input_schema": {"type": "object", "properties": {
        "document_id": _DOC,
        "teil": {"type": "string", "enum": ["probleme", "hoerprobe"], "description": "Standard probleme."},
        "von": {"type": "integer", "description": "Erste Zeile der Hörprobe (ab 1)."},
        "anzahl": {"type": "integer", "description": "Höchstens so viele Zeilen (Standard 80)."},
    }, "required": []},
}
_ALT_TEXTE_HERUNTERLADEN = {
    "name": "exportiere_alt_texte",
    "description": ("Wie „Alt-Texte herunterladen“: die Alt-Texte als Tabelle (csv, xlsx = Excel, json), fester Preis je Vorgang. Ohne "
                    "document_id alle Dokumente (mehrere als ZIP). ZWEI SCHRITTE: erst ohne bestaetigt (Preis nennen, fragen), "
                    "nach dem Ja mit bestaetigt=true. Download-Knopf unter deiner Antwort."),
    "input_schema": {"type": "object", "properties": {
        "format": {"type": "string", "enum": ["csv", "xlsx", "json"], "description": "Standard csv."},
        "document_id": _DOC, "bestaetigt": _BESTAETIGT,
    }, "required": []},
}
_QUICKINFOS_HERUNTERLADEN = {
    "name": "exportiere_quickinfos",
    "description": ("Wie „Quickinfos herunterladen“: die Feldliste mit Quickinfos als CSV, fester Preis je Vorgang. ZWEI SCHRITTE wie "
                    "exportiere_alt_texte. Download-Knopf unter deiner Antwort."),
    "input_schema": {"type": "object", "properties": {"document_id": _DOC, "bestaetigt": _BESTAETIGT}, "required": []},
}
_ALT_TEXTE_GENERIEREN = {
    "name": "alt_texte_generieren",
    "description": ("Wie „Alt-Texte generieren“: Alt-Texte für alle Bilder des Projekts oder eines Dokuments, Preis je Bild, läuft im "
                    "Hintergrund. Überschreibt vorhandene Texte (auch eigene) — der erste Aufruf nennt Anzahl, Preis und wie viele "
                    "eigene Texte überschrieben würden. ZWEI SCHRITTE: erst ohne bestaetigt, nach dem Ja mit bestaetigt=true. "
                    "Für ein einzelnes Bild: generate_alt_text."),
    "input_schema": {"type": "object", "properties": {"document_id": _DOC, "bestaetigt": _BESTAETIGT}, "required": []},
}
_QUICKINFOS_GENERIEREN = {
    "name": "quickinfos_generieren",
    "description": ("Wie „Quickinfos generieren“: Quickinfos für alle benannten Felder des Projekts oder eines Dokuments, Preis je Feld, "
                    "läuft im Hintergrund, überschreibt vorhandene Quickinfos. ZWEI SCHRITTE wie alt_texte_generieren. Für ein "
                    "einzelnes Feld: generate_quickinfo."),
    "input_schema": {"type": "object", "properties": {"document_id": _DOC, "bestaetigt": _BESTAETIGT}, "required": []},
}
_STAMMDATEN = {
    "name": "stammdaten_anwenden",
    "description": ("Wie „Stammdaten auf alle Felder anwenden“: offene Felder bekommen die passende Quickinfo aus den Stammdaten "
                    "des Kontos. Kostenlos."),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}
_KI_KONTEXT = {
    "name": "ki_kontext_setzen",
    "description": ("Wie das Kästchen „KI-Kontext aus dem Dokument verwenden“: an=true gibt der KI beim Generieren den Text rund um "
                    "das Bild mit, an=false nicht. Gilt für Alt-Texte, die ab jetzt erzeugt werden. Kostenlos."),
    "input_schema": {"type": "object", "properties": {"an": {"type": "boolean"}}, "required": ["an"]},
}
_EIGENER_PROMPT = {
    "name": "eigener_prompt",
    "description": ("Wie die Auswahl „Gespeicherte Prompts“: ohne prompt_id (oder auflisten=true) die eigenen Prompts und die aktuelle "
                    "Wahl; mit prompt_id einen setzen, 0 = kein eigener Prompt. Gilt für alles, was ab jetzt generiert wird. Kostenlos."),
    "input_schema": {"type": "object", "properties": {
        "prompt_id": {"type": "integer"}, "auflisten": {"type": "boolean"},
    }, "required": []},
}
_ABLAGE_LOESCHEN = {
    "name": "ausgabe_loeschen",
    "description": ("Wie „Löschen“ in der Ablage: einen Eintrag dieses Projekts samt Datei löschen. Unumkehrbar — ZWEI SCHRITTE: erst "
                    "ohne bestaetigt (sagen, was weg wäre, fragen), Ja in einer eigenen Nachricht, dann bestaetigt=true. Kostenlos."),
    "input_schema": {"type": "object", "properties": {
        "ausgabe_id": {"type": "integer", "description": "Aus liste_ausgaben."}, "bestaetigt": _BESTAETIGT,
    }, "required": ["ausgabe_id"]},
}

# PDF-Projekt: alles aus den Ansichten Dokument, Tagging, Alt-Texte, Quickinfos, Barrierefreiheitsprüfung
TOOL_DEFINITIONS_OBERFLAECHE_PDF: list[dict] = [
    _TESTWEISE, _PRUEFDATEI_ERSTELLEN, _PRUEFDATEI_LESEN, _ALT_TEXTE_HERUNTERLADEN, _QUICKINFOS_HERUNTERLADEN,
    _ALT_TEXTE_GENERIEREN, _QUICKINFOS_GENERIEREN, _STAMMDATEN, _KI_KONTEXT, _EIGENER_PROMPT, _ABLAGE_LOESCHEN,
]
# Word-Projekt: Alt-Texte herunterladen, Sammellauf, Einstellungen, Ablage; Umbenennen/Löschen/Sprache aus definitions_pdf
TOOL_DEFINITIONS_OBERFLAECHE_WORD: list[dict] = [
    _ALT_TEXTE_HERUNTERLADEN, _ALT_TEXTE_GENERIEREN, _KI_KONTEXT, _EIGENER_PROMPT, _ABLAGE_LOESCHEN,
]
# Altes Formular-Projekt (Quickinfo-Werkzeug): Quickinfos herunterladen (CSV), Sammellauf, Stammdaten
TOOL_DEFINITIONS_OBERFLAECHE_FORMULAR: list[dict] = [_QUICKINFOS_HERUNTERLADEN, _QUICKINFOS_GENERIEREN, _STAMMDATEN, _EIGENER_PROMPT]
