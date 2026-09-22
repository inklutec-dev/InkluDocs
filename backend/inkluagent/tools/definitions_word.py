"""Werkzeugdefinitionen fuer WORD-PROJEKTE (Meine Ablage, Uebersetzen) — eigene Datei seit 22.09.2026,
wie definitions_formular.py und definitions_pdf.py: je Dateiart eine Definitionsdatei (Geruest, docs/GERUEST.md).
Angehaengt in agent_loop._werkzeugsatz fuer project_type docx; Handler in definitions.ToolExecutor (word=True).
"""
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
            "Hinweisen; die Datei liegt in der Ablage und der Nutzer sieht unter deiner Antwort einen "
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
            "Gibt die Word-Datei mit den aktuellen Alt-Texten aus (Download-Knopf unter deiner Antwort; "
            "kein Eintrag in der Ablage — dort liegen nur umgewandelte PDFs). Kostet Credits. Gleiche zwei Schritte wie konvertiere_zu_pdfua: erst ohne "
            "bestaetigt (Preis nennen, fragen), dann mit bestaetigt=true."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional: nur dieses Dokument."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers. Standard false."},
        }, "required": []},
    },
    {
        "name": "analysiere_word_struktur",
        "description": (
            "Struktur-Lektor (Lesestufe): prüft, ob die Struktur, die ein Screenreader bekommt, der Struktur entspricht, "
            "die ein Sehender sieht. Liefert Gliederung, Absatz-Auszug (Formatvorlage, fett, Schriftgröße, Liste, Tabelle) "
            "und Befunde mit Absatznummer, Sicherheit (hoch = belegt, mittel = Vermutung) und Vorschlag: fette/größere "
            "Zeilen ohne Überschriften-Vorlage, getippte Listen, Leerabsätze als Abstand, Großbuchstaben, Linktexte, "
            "Layouttabellen, verschachtelte Tabellen. Kostenlos. Nutze es, wenn der Nutzer den Aufbau bewerten lassen "
            "will („ist das Dokument gut strukturiert?“, „bewerte den Aufbau“). Ohne document_id alle Dokumente."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional: nur dieses Dokument."},
        }, "required": []},
    },
    # Übersetzen als Fähigkeit des Word-Projekts (Testumbau 18.09.2026).
    {
        "name": "uebersetze_dokument",
        "description": (
            "Übersetzt das ganze Word-Dokument (alle Dokumente des Projekts) in eine Zielsprache; Struktur und Formatierung "
            "bleiben unverändert, Alt-Texte werden mitübersetzt, die Dokumentsprache wird gesetzt. Kostet Credits (1 je "
            "angefangene 100 Wörter). ZWEI SCHRITTE wie konvertiere_zu_pdfua: erst OHNE bestaetigt (Umfang, Preis, Guthaben "
            "nennen und fragen), nach dem Ja mit bestaetigt=true. Läuft im Hintergrund; Stand über uebersetzung_stand, Datei "
            "über exportiere_uebersetzung. Zielsprachen: en-gb, en, en-au, de, de-at, de-ch, fr, fr-ch, es, es-419, pt, pt-br, "
            "da, sv, it, nl, nl-be, pl, tr, uk, ru, ar. Sagt der Nutzer nur „Englisch“, nimm en-gb und sag ihm das."
        ),
        "input_schema": {"type": "object", "properties": {
            "zielsprache": {"type": "string", "description": "Kennung der Zielsprache, z. B. en-gb, en, fr, es-419."},
            "alt_texte": {"type": "boolean", "description": "Alt-Texte der Bilder mitübersetzen. Standard true."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers zum genannten Preis. Standard false."},
        }, "required": ["zielsprache"]},
    },
    {
        "name": "uebersetzung_stand",
        "description": "Stand der Übersetzung dieses Projekts: Zielsprache, fertige und gesamte Absätze, Hinweise, ob ein Lauf läuft. Kostenlos.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "exportiere_uebersetzung",
        "description": (
            "Gibt die übersetzte Word-Datei aus (Download-Knopf unter deiner Antwort; kostenlos, keine Ablage). Nur sinnvoll, "
            "wenn uebersetzung_stand fertig > 0 meldet und kein Lauf mehr läuft."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "liste_ausgaben",
        "description": (
            "Alle fertigen Ausgaben dieses Projekts (barrierefreie PDFs, Word-Dateien) mit Datum, Prüfstand und "
            "Verfügbarkeit der Datei — die Ablage. Keine Args."
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
