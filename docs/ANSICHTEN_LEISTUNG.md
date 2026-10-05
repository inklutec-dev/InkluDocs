# Ansichtswechsel und Texte einmal je Projekt

Stand 05.10.2026. Auslöser: Michael Karbe, 05.10.2026: „Beim mehreren PDF wird der Wechsel der Ansicht sehr langsam.
Projekt mit 7 PDF: mehr als 5 Sekunden.“ Bauplan mit allen Messdaten: `/home/claude/ansichtswechsel-1005/BAUPLAN.md`
(Server, Nutzer claude).

## Ursache

Jeder Wechsel zwischen den Ansichten (Dokument, Tagging, Alt-Texte, Quickinfos, Barrierefreiheitsprüfung, Übersetzung)
lud zuerst die komplette Projektantwort `GET /api/projects/{id}` (main.get_project, `SELECT i.*` aller Bilder). Darin
stand je Bild der KI-Kontext. Beim PDFix-Weg ist das der Abschnitt um die Abbildung, in PDFs ohne Überschrift-Tags also
das ganze Dokument, kopiert in jede Bildzeile. Bei 7 PDF mit 269 Bildern waren das 43,8 MB je Wechsel (41 Mio. Zeichen
Kontext), obwohl die Oberfläche den Kontext nirgends anzeigt und die Ansichten Dokument, Tagging und Prüfung aus der
Antwort nur Typ und gemerkte Ansicht brauchen.

Dazu kamen mehrere Bremsen:
- feste Wartezeit von 900 ms in `wechsleAnsicht` für das Auto-Speichern;
- fehlender Index auf `images.document_id`: Jede Abfrage nach Dokument las die ganze Tabelle samt der Kontexte aller Kunden;
- `tag_statistik` las bei jedem Aufruf der Ansicht „Dokument“ den ganzen Strukturbaum;
- get_project und dokument-ansicht liefen synchron im Event-Loop und hielten alle anderen Anfragen an;
- beim ersten Öffnen der Alt-Texte wurden alle Seitenansichten sofort geladen (bei 7 PDF 119 Bilder, 46 MB).

Gemessen vorher (Nachbau mit 7 fiktiven PDF):
- ohne Leitungsgrenze 2,1 bis 2,8 s je Wechsel;
- bei 50 Mbit/s 9,1 bis 9,8 s;
- bei 16 Mbit/s rund 25 s.

## Lösung

### Jede Ansicht bekommt nur, was sie braucht

- `GET /api/projects/{id}` (main.get_project) liefert die Bildliste schlank (`_bildliste`). Alle Spalten bleiben außer
  `context_text`, `page_text`, `pipeline_steps`, `validation_result`, `kontext_id`, `seitentext_id` und den Serverpfaden
  `image_path`, `page_view_path`. Neu sind die Merker `hat_seitenansicht` und `hat_seitentext`. Die Gastansicht
  (`/api/freigabe/{token}`) nutzt dieselbe Liste.
- `GET /api/projects/{id}/kopf`: Projektkopf ohne Bildliste (wenige KB) für die Weiche der Ansichten und für Polls.
  Felder: Projekt mit lauf_art und hat_felder, Dokumente ohne Serverpfad, `bilder_status`, `bilder_gesamt`,
  Freigabe-Rollen und Ablage-Zähler.
- `GET /api/images/{id}/seitentext` und `GET /api/freigabe/{token}/images/{id}/seitentext`: Seitentext erst beim Aufklappen.
- Alle drei laufen im Executor. Die Route get_project bleibt `async def`, weil die Public API v1 sie direkt mit await
  aufruft (`api_dokumente_v1._items_list`).
- app.html `showProject`: zuerst `/kopf`, die Bildliste nur in der Ansicht Alt-Texte. Steht `?ansicht=alttexte` in der
  Adresse, wird gleich die Bildliste geholt. Ein Zähler `_showProjectLauf` sorgt dafür, dass bei schnellem Wechseln nur
  der jüngste Aufruf zeichnet.
- Seitenansicht mit `loading="lazy"`: In einer zugeklappten Klappe lädt der Browser sie erst beim Aufklappen.
- Seitentext: Wird die Seite aufgeklappt, holt die Ansicht ihn im Hintergrund (`seitentextVorladen`). Spätestens beim
  Aufklappen der Klappe „Seitentext anzeigen“ wird er geholt (`seitentextLaden`).
  - Weg für Screenreader: Der Fokus bleibt auf dem Schalter. Bis der Text da ist, steht im Bereich (role=region, Name
    „Seitentext“) „Seitentext wird geladen …“ mit aria-busy. Nur wenn das Laden länger als 1 s dauert, kommt die kurze
    Meldung „Seitentext geladen.“
  - Bei einem Fehler steht ein Satz im Bereich; erneutes Aufklappen versucht es noch einmal.
- Die Polls (Upload `projectSnapshot`/`waitForExtraction`, `updateProjectHeader`, formular.js) fragen `/kopf` ab statt der
  ganzen Antwort.
- Feste 900 ms entfallen. Textfelder melden ihre anstehende Speicherung bei `speicherungVormerken(schluessel, fn)` an
  (app.html: Alt-Text; formular.js: Quickinfo; uebersetzen.js: Übersetzung). `wechsleAnsicht` und Browser-Zurück/Vor
  rufen `alleAusstehendenSpeichern()`: Offene Speicherungen gehen sofort raus, abgeschickte werden abgewartet
  (Obergrenze 8 s). Ohne offene Eingabe kostet das 0 ms.
- Ansicht „Dokument“: `tagging_api` merkt Seitenzahl, Struktur (`tag_statistik`), Metadaten und bei Rohdatei
  `quelle_getaggt` in `documents.struktur_json`.
  - Gültig, solange `struktur_stand` passt (Pfad, Änderungszeit und Größe von original_path und roh_path, plus
    `_STRUKTUR_VERSION`).
  - Schreibt Tagging, Korrektur oder ein Upload die Datei neu, wird automatisch neu gerechnet.
  - Die Route läuft im Executor (`_ansicht_sync`).
- Index `idx_images_document (document_id, page_number, image_index)`.
- `get_image_file` und `freigabe_image_file` lesen nur den Pfad. API v1 `_dokument_status` liest nur die Felder für die
  Zähler. Der Chatbot liest in `get_project_context` keinen Kontext mehr (er wurde bei jeder Nachricht für alle Bilder
  gelesen, aber nicht genutzt).

### Texte einmal je Projekt (Datenmodell)

- Tabelle `projekt_texte`: id, project_id, sha256, text, zeichen, angelegt; `UNIQUE(project_id, sha256)`.
- Spalten `images.kontext_id` und `images.seitentext_id` mit Indizes. Die alten Spalten `context_text` und `page_text`
  bleiben: Nach der Migration sind sie leer, vorher sind sie der Rückfall.
- Je Projekt statt global: Mandantentrennung und Löschen bleiben einfach. Die Dubletten liegen ohnehin innerhalb eines
  Dokuments.
- Schlüssel ist der Inhalt (SHA-256), nicht „Dokument + Seite“. So bildet er jede Kontextquelle ab (PDFix-Kapitel über
  Seiten hinweg, Seite, Volltext, fitz-Umgebung, Word, Web), und der Text bleibt byte-gleich. KI-Eingabe,
  Cache-Schlüssel (`cache.build_cache_key` mit enriched_context), Alt-Texte und Credits ändern sich nicht.
- Kern: `backend/projekt_texte.py`:
  - `text_ablegen`;
  - `bild_kontext` und `bild_seitentext` (Verweis, sonst alte Spalte; NULL bleibt NULL);
  - `kontext_sql` (Ausdruck und JOIN für Abfragen);
  - `texte_aufraeumen` und `projekt_texte_loeschen`;
  - Migration: `probe`, `phase_a`, `pruefen`, `phase_b`, `zurueck`, `fingerabdruck`.
- Schreibwege: `main._bilder_uebernehmen` (Upload PDF/Word und Neu-Extraktion nach dem Tagging) und `main.scan_url`
  (Web). Grafik-Upload und versteckte Web-Bilder haben keinen Kontext.
- Lesewege:
  - `main._process_project_lauf` (Sammellauf) und `main.regenerate_image`;
  - `inkluagent/adapters/inkludocs.run_pipeline_for_image`;
  - `inkluagent/tools/altext._verify_gegen_bild`;
  - `inkluagent/tools/project` (Bildliste mit 200 Zeichen, Bilddetail voll);
  - Seitentext-Abrufe.
- Aufräumen: `_dokument_loeschen_sync`, `delete_image`, Tagging-Neu-Extraktion (`texte_aufraeumen` in derselben
  Transaktion); `delete_project` und `database.delete_user_data` (`projekt_texte_loeschen`).

## Migration und Rückweg

Ablauf: Container mit dem neuen Code starten (init_db legt Tabelle, Spalten und Indizes an), dann im Container:

    docker exec -w /app <container> python3 scripts/texte_migration.py                    # Probe, ändert nichts
    docker exec -w /app <container> python3 scripts/texte_migration.py --alles --vacuum   # Sicherung, A, Prüfung, B

Was `--alles` tut:
1. Sicherung `inkludocs.db.bak-pre-texte-<Zeit>` über die SQLite-Backup-Funktion (konsistent bei WAL und laufendem
   Betrieb), danach `integrity_check`. Ohne gültige Sicherung bricht das Skript ab.
2. Phase A: Verweise setzen, idempotent, eine kurze Transaktion je Projekt. Die alten Spalten bleiben gefüllt.
3. Prüfung: Jeder Verweis liefert byte-gleich den Text der alten Spalte, keiner zeigt ins Leere oder in ein fremdes
   Projekt. Bei einem Befund bricht das Skript vor Phase B ab, es geht nichts verloren.
4. Phase B: alte Spalten leeren, nur wo der Verweis nachweislich denselben Text liefert (Bedingung im UPDATE selbst).
5. Fingerabdruck: SHA-256 des wirksamen Textes je Bild vor A und nach B, muss gleich sein.
6. `--vacuum` gibt den Platz frei: kurz exklusiv, bei 100 MB unter 1 s.

Einzelschritte: `--sicherung`, `--phase-a`, `--pruefen`, `--phase-b`, `--vacuum`.

Rückweg ohne Wiederherstellen der Sicherung: `--zurueck` füllt `context_text` und `page_text` aus `projekt_texte` wieder
auf (vorher wieder eine Sicherung). Ein erneutes `--alles` leert sie wieder.

WICHTIG beim Zurückrollen des CODES: Der alte Code liest nur die alten Spalten. Wer nach Phase B auf einen Stand vor
diesem Umbau zurückgeht, muss VORHER `--zurueck` laufen lassen, sonst bekäme die KI keinen Kontext mehr. Neuer Code mit
nicht migrierter Datenbank ist dagegen jederzeit in Ordnung (Rückfall auf die alten Spalten).

Probe auf einer Kopie der Staging-Datenbank (05.10.2026, 2.717 Bilder):
- Phase A 1,1 s, Phase B 0,5 s, Prüfung 0, Fingerabdruck 0 Abweichungen;
- 1.762 Texte mit 3,1 Mio. Zeichen statt 83 Mio. Zeichen Kontext und 3,4 Mio. Zeichen Seitentext;
- Datei 97,6 → 8,1 MB.

## Messreihe und Zielwerte

- `tests/e2e/mess_ansichtswechsel.py alles` legt ein Projekt mit 7 klar fiktiven PDF an (Generator
  `tests/fixtures/make_messpdfs.py`; 269 Bilder, rund 41 Mio. Zeichen Kontext wie das Kundenprojekt), misst und löscht
  es wieder.
  - Profile: ohne Leitungsgrenze, 50 Mbit/s (20 ms), 16 Mbit/s (30 ms).
  - Zielwerte (Median je Ansicht, Klick bis die H1 der neuen Ansicht den Fokus hat): 300 ms, 500 ms und 1000 ms.
  - Dokument, Tagging und Prüfung laden keine Bildliste. Projektantwort höchstens 400 KB.
  - Erstes Öffnen der Alt-Texte ohne Seitenansicht-Anfrage.
  - Aufklappen von Seite, Seitentext und Seitenansicht höchstens 300 ms bis Inhalt da (ohne Leitungsgrenze und bei
    50 Mbit/s); dazu Fokus und Region beim Seitentext.
- `tests/e2e/ui_ansichtswechsel.py`: tippen und sofort wechseln (Alt-Text zu Dokument, Browser-Vor, Quickinfo zu
  Dokument), Wechsel ohne feste Wartezeit, Seitentext und Seitenansicht beim Aufklappen (Besitzer und Gast).
- Unit: `tests/test_texte_migration.py` (Schema, Rückfall, Migration, Rückweg, Mandantentrennung, Skript) und
  `tests/test_projekt_schlank.py` (schlanke Antwort, Kopf, Seitentext, API v1, Chatbot und Pipeline mit byte-gleichem
  Kontext, Struktur-Merker).

Staging 05.10.2026 (nach Rebuild und Migration, 7 PDF, Median je Ansicht):
- vorher ohne Leitungsgrenze 2,05 bis 2,69 s (bei 50 Mbit/s 9,1 bis 9,8 s, bei 16 Mbit/s 24 bis 25 s, Messung vom Mittag);
- nachher ohne Leitungsgrenze 130 bis 202 ms, bei 50 Mbit/s 153 bis 246 ms, bei 16 Mbit/s 185 bis 269 ms;
- Projektantwort 195 KB statt 43,8 MB; beim ersten Öffnen der Alt-Texte keine Seitenansicht-Anfrage;
- Aufklappen: Seite 55 bis 66 ms, Seitentext ohne Vorlauf 10 bis 41 ms, Seitenansicht 34 ms (50 Mbit/s 117 ms, 16 Mbit/s 300 ms).

Migration auf Staging (05.10.2026, 2.448 Bilder in 100 Projekten):
- Phase A 0,8 s, Phase B 0,3 s, Prüfung ohne Befund, Fingerabdruck 0 Abweichungen;
- 1.639 Texte mit 2,2 Mio. Zeichen statt 41,7 Mio. Zeichen Kontext und 2,4 Mio. Zeichen Seitentext;
- Datei 97,6 → 6,9 MB, gesamt 2 s;
- Kundenprojekt 957: Projektantwort 233 KB statt 40,2 MB.

Gemessen nach dem Umbau (Wegwerf-Container mit Kopie der Staging-Daten, 7 PDF, nach der Migration):
- ohne Leitungsgrenze 150 bis 210 ms;
- 50 Mbit/s 155 bis 220 ms;
- 16 Mbit/s 155 bis 305 ms;
- Projektantwort 193 KB statt 43,8 MB;
- Aufklappen: Seite 41 ms, Seitentext vorgeladen 0 ms, Seitenansicht 35 ms (50 Mbit/s: 133 ms).

## Regeln für Weiterbau

- Keine großen Felder in die Bildliste (`_BILDLISTE_OHNE`). Was nur beim Einzelbild gebraucht wird, bekommt einen eigenen
  Abruf.
- Neue lange Texte je Bild (Kontext, Seitentext und Ähnliches) über `projekt_texte.text_ablegen` ablegen, lesen mit
  `bild_kontext`/`bild_seitentext` bzw. `kontext_sql`. Nie `img["context_text"]` direkt lesen.
- Neue Auto-Speicher-Felder melden sich bei `speicherungVormerken` an, damit ein Ansichtswechsel nichts verliert.
- Ansichten laden ihre Daten selbst über eigene Abrufe; `showProject` braucht für die Weiche nur `/kopf`.
- Synchrone Datei- und Datenbankarbeit in Routen gehört in den Executor (Verbindung im Worker öffnen).
