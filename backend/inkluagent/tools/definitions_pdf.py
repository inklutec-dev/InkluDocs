# Werkzeugdefinitionen fuer PDF-PROJEKTE (Werkzeugsatz nach Dateiart, 22.09.2026). Wird in
# agent_loop._werkzeugsatz an TOOL_DEFINITIONS (Bilder) + TOOL_DEFINITIONS_FORMULAR (Felder) angehaengt.
# Kostenpflichtige Werkzeuge verlangen bestaetigt=true — der SERVER liefert beim ersten Aufruf nur Preis
# und Guthaben, die Rueckfrage an den Nutzer ist damit erzwungen (tools/pdf._freigabe).
TOOL_DEFINITIONS_PDF: list[dict] = [
    {
        "name": "dokument_stand",
        "description": (
            "Stand der Dokumente dieses PDF-Projekts: Seiten, getaggt oder nicht, Sprache, Struktur (Überschriften, "
            "Listen, Tabellen, Bilder), Bilder mit Alt-Text, Felder mit Quickinfo, PDF/UA-Prüfung, Stand der "
            "automatischen Prüfung, laufende Kette, Einträge in der Ablage. Kostenlos. Immer der erste Schritt, wenn "
            "der Nutzer etwas zum Dokument will oder nach einem Lauf fragt. Ohne document_id alle Dokumente."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional: nur dieses Dokument."},
        }, "required": []},
    },
    {
        "name": "barrierefrei_machen",
        "description": (
            "Tagging eines PDF-Dokuments (Strukturbaum, Lesereihenfolge, Dokumentsprache, veraPDF-Prüfung). Kostet "
            "Credits je Seite. ZWEI SCHRITTE: Erster Aufruf OHNE bestaetigt liefert Seiten, Preis und Guthaben "
            "(rueckfrage_noetig) — nenne dem Nutzer den Preis und frage. Erst nach seinem klaren Ja erneut mit "
            "bestaetigt=true. Läuft dann im Hintergrund; Stand über dokument_stand. Bei nur einem Dokument ohne document_id."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional bei einem Dokument; Pflicht bei mehreren."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers zum genannten Preis. Standard false."},
        }, "required": []},
    },
    {
        "name": "komplett_barrierefrei_machen",
        "description": (
            "Kette für das ganze Projekt: Tagging aller ungetaggten Dokumente, dann Alt-Texte für alle Bilder, dann "
            "Quickinfos für alle Felder. Erster Aufruf OHNE bestaetigt liefert den Plan je Station mit Preis und den "
            "Gesamtpreis — nennen und fragen. Nach dem Ja mit bestaetigt=true. Läuft im Hintergrund (Minuten); Stand "
            "über dokument_stand (kette)."
        ),
        "input_schema": {"type": "object", "properties": {
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers zum Gesamtpreis. Standard false."},
        }, "required": []},
    },
    {
        "name": "hoerprobe_lesen",
        "description": (
            "Hörprobe der getaggten PDF: Zeilen in Lesereihenfolge, wie ein Screenreader sie bekommt (Überschriften mit "
            "Ebene, Absätze, Listen, Tabellen mit Zeilen/Spalten, Grafiken mit Alt-Text, Formularfelder mit Quickinfo, "
            "Seitenmarken). Kostenlos, nur für getaggte Dokumente. Seitenweise: von (Zeilennummer, Standard 1) und "
            "anzahl (Standard 80, höchstens 120)."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional bei einem Dokument."},
            "von": {"type": "integer", "description": "Erste Zeile (1-basiert). Standard 1."},
            "anzahl": {"type": "integer", "description": "Zeilen je Aufruf. Standard 80."},
        }, "required": []},
    },
    {
        "name": "pruefung_starten",
        "description": (
            "Automatische Prüfung eines getaggten Dokuments: ein KI-Modell vergleicht je Seite das Seitenbild mit den Tags "
            "und meldet nur belegbare Abweichungen (Überschrift als Listenpunkt, falsche Ebene, Tabelle ohne Kopfzeile, "
            "Alt-Text passt nicht zum Bild, sichtbarer Text ohne Tag). Kostet Credits je Seite. ZWEI SCHRITTE wie "
            "barrierefrei_machen (erst ohne bestaetigt: Preis nennen, fragen; dann bestaetigt=true). Läuft im Hintergrund; "
            "Ergebnis über pruefbericht_lesen. Ändert nichts an der Datei."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional bei einem Dokument."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers zum genannten Preis. Standard false."},
        }, "required": []},
    },
    {
        "name": "pruefbericht_lesen",
        "description": (
            "Bericht der automatischen Prüfung: Zahl der Befunde nach Sicherheit und jeder Befund mit Seite, Rolle, "
            "Textanfang, Befund, Vorschlag, Beleg, Sicherheit (hoch = belegt, mittel/niedrig = Vermutung). Läuft die "
            "Prüfung noch, kommt der Stand (Seite a von b). Kostenlos."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional bei einem Dokument."},
        }, "required": []},
    },
    {
        "name": "exportiere_fertige_pdf",
        "description": (
            "Fertige PDF eines getaggten Dokuments mit Struktur, Alt-Texten und Quickinfos: Download-Knopf unter deiner "
            "Antwort und Eintrag in der Ablage (mit Bericht). Kostet Credits (Export-Staffel nach Bildern). ZWEI SCHRITTE "
            "wie barrierefrei_machen. Nur für getaggte Dokumente."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional bei einem Dokument."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers zum genannten Preis. Standard false."},
        }, "required": []},
    },
    {
        "name": "dokument_umbenennen",
        "description": "Anzeigename eines Dokuments setzen (wie der Knopf „Umbenennen“). Leerer Name = zurück auf den Dateinamen. Kostenlos.",
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional bei einem Dokument."},
            "name": {"type": "string", "description": "Der neue Anzeigename (höchstens 200 Zeichen)."},
        }, "required": ["name"]},
    },
    {
        "name": "dokument_loeschen",
        "description": (
            "Dokument samt Bildern, Alt-Texten, Feldern, Quickinfos und Dateien aus dem Projekt löschen — unumkehrbar. "
            "ZWEI SCHRITTE wie bei kostenpflichtigen Werkzeugen: erst OHNE bestaetigt (sagt, was gelöscht würde; frage den "
            "Nutzer), erst nach seinem klaren Ja in einer eigenen Nachricht mit bestaetigt=true."
        ),
        "input_schema": {"type": "object", "properties": {
            "document_id": {"type": "integer", "description": "Optional bei einem Dokument; Pflicht bei mehreren."},
            "bestaetigt": {"type": "boolean", "description": "true NUR nach ausdrücklichem Ja des Nutzers. Standard false."},
        }, "required": []},
    },
    {
        "name": "alt_sprache_setzen",
        "description": (
            "Sprache der Alt-Texte und Quickinfos dieses Projekts setzen (de, en, da, fr, es, sv …) — gilt für alles, was ab "
            "jetzt erzeugt wird; vorhandene Texte bleiben. Kostenlos."
        ),
        "input_schema": {"type": "object", "properties": {
            "sprache": {"type": "string", "description": "Sprachkürzel, z. B. de, en, fr."},
        }, "required": ["sprache"]},
    },
]
