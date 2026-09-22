# PDF-Tagging: „Barrierefrei machen“ mit PDFix (22.09.2026)

Ungetaggte oder schlecht getaggte PDFs bekommen einen Strukturbaum (Überschriften, Absätze,
Listen, Tabellen, Figures), Titel, Sprache, Lesezeichen und die PDF/UA-1-Kennung. Danach
laufen die bekannten Wege (Alt-Texte über den Strukturbaum, Export mit AltTag_Import) auf
der getaggten Datei.

Beteiligte: Steve Weidel (InkluTec), Michael Karbe (Actino, Produkt), Jörg Heine (Actino,
Skript), PDFix (SDK und eingebaute Aktion). Backend, Endpunkte, Oberfläche (Ansicht
„Dokument“, Abschnitt unten) und Tests: hier.

## Worum es geht

- Jörg Heines Skript `Make_Accessible_01.py` (Mail 21.09.2026 an kontakt@) lädt die in PDFix
  eingebaute Aktion `make_accessible` und führt sie aus. Original unverändert unter
  `backend/pdfix_scripts/original_heine/Make_Accessible_01.py`, Betriebsfassung
  `backend/pdfix_scripts/Make_Accessible.py` (mechanisch erzeugt mit
  `tests/werkzeuge/baue_make_accessible.py`, Drift-Test `tests/test_pdfix_skript_drift.py`).
  Regel wie bei den Formular-Skripten (Steve 17.09.): Heines Skript ist die Vorlage, wir tragen
  nur markierte Zeilen auf (`# InkluDocs`).
- Die Aktion ist eine JSON-Konfiguration mit 37 Teilschritten (Version 0.7.3, 13.08.2026).
  Die aus SDK 9.3.0 exportierte Voreinstellung liegt unverändert unter
  `backend/pdfix_scripts/make_accessible_pdfix_default.json`; ein Test vergleicht sie mit der
  im SDK eingebauten Fassung (Drift-Wache bei SDK-Updates).
- `backend/pdf_tagging.py` erzeugt je Lauf eine angepasste Konfiguration und ruft das Skript als
  Subprocess auf (Zeitlimit `PDFIX_TAGGING_TIMEOUT`, Standard 600 s; Obergrenze
  `PDFIX_TAGGING_MAX_SEITEN`, Standard 500).
- `backend/tagging_api.py`: Endpunkte, Hintergrundlauf, Neu-Extraktion der Bilder, Übernahme
  vorhandener Alt-Texte, Credits.

## Was wir an der PDFix-Voreinstellung ändern (und warum)

1. **Dokumentsprache.** PDFix erkennt keine Sprache; der Schritt „Set Document Language“ trägt
   fest `en-US` ein, wenn nichts gesetzt ist. Wir erkennen die Sprache aus dem Text (dieselbe
   Erkennung wie im Word-Prüfbericht, `docx_hoerprobe.erkenne_sprache`, sechs Sprachen) und
   setzen sie als BCP-47-Wert (`de-DE`, `en-US`, `da-DK`, `fr-FR`, `es-ES`, `sv-SE`).
   Regel: Text sicher erkannt (mindestens 20 Treffer, doppelt so viele wie die zweitbeste
   Sprache) → diese Sprache; weicht die im Dokument gesetzte Sprache ab, wird sie ersetzt
   (Hinweis im Bericht). Nicht sicher erkannt → vorhandene Dokumentsprache bleibt; fehlt auch
   die, gilt die Projektsprache (Hinweis im Bericht).
2. **Keine Alt-Texte von PDFix.** Vier Schritte „Set Alt“ für Figure/Formula kopieren
   Bildunterschriften oder Nachbarabsätze in den Alt-Text oder schreiben das feste Wort
   „Decorative“ hinein. PDFix schaut das Bild nie an. Diese Schritte entfallen, ebenso der
   „Decorative“-Rückfall für Anmerkungen (Set Annotation Contents, Auto-generated). Die
   Alt-Texte schreibt InkluDocs über die Ansicht „Alt-Texte“ und den Export. „Set Alt“ für
   Formularfelder (aus dem zugehörigen Inhalt) bleibt.
3. Alles andere bleibt: Aufräumen, Tags hinzufügen, Tabellen und Überschriften reparieren,
   Titel (Title-Tag → H1 → Info → Dateiname, nie überschreiben), Lesezeichen aus H1–H3,
   PDF/UA-1-Kennung.

## Lizenz und Testmodus (wichtig für Prod)

Stand 22.09.2026 ist der Teilschritt `add_tags` in der Actino-Lizenz **nicht** freigeschaltet.
Mit aktivierter Lizenz bricht die Aktion bei ungetaggten PDFs ab („Invalid initial element
type or initial element parent“). **Ohne** Lizenz läuft das SDK im Testmodus und taggt; die
Datei trägt dann „Trial version of PDFix SDK | www.pdfix.net“ als Producer (kein sichtbares
Wasserzeichen im Seiteninhalt gefunden).

Deshalb aktiviert `inkludocs_betrieb.lizenz_fuer_tagging` die Lizenz **nur bei
`PDFIX_TAGGING_LIZENZ=on`**. Alle anderen Skripte (Export, Import, Formulare) aktivieren sie
immer. Der Bericht nennt den Modus (`modus`, `testmodus`), die Oberfläche soll ihn zeigen.
**Prod bekommt das Tagging erst, wenn Actino/PDFix den Schritt freischalten** (Michael Karbe
klärt mit Joseph, Seitenpreis). WhatsApp an Michael mit diesem Stand: 22.09.2026.

## Endpunkte (nur Besitzer, nur PDF-Projekte: Werkzeuge `pdf` und `formular`)

- `GET /api/projects/{id}/documents/{doc}/tagging` → Stand: `status` (leer | laeuft | fertig |
  fehler), `getaggt`, `seiten`, `preis`, `verfuegbar_credits`, `erlaubt`, `fehlend`,
  `hat_alt_texte`, `neu_taggen` (roh_path vorhanden), `modus`, `bericht`, `projekt_status`.
- `POST /api/projects/{id}/documents/{doc}/tagging` → startet den Lauf. 400 kein PDF /
  zu viele Seiten, 402 Credits (`credits_fehlen`-Body wie überall), 409 läuft bereits oder
  Projekt in Verarbeitung, 503 nicht eingerichtet. Antwort `{gestartet, seiten, preis, modus}`.
- `GET /api/projects/{id}/documents/{doc}/tagging/datei` → das getaggte PDF (nur bei
  `fertig`), Dateiname `<Dokument>_getaggt.pdf`.

Kein Gastweg, keine Public-API-Route (folgt mit der Oberfläche). In der Demo gibt es keine
Konten, also keinen Aufruf.

## Ablauf eines Laufs (tagging_api._lauf_sync, im Executor)

1. Projekt auf `extracting` (Generierung und Export warten, wie beim Upload), Dokument auf
   `laeuft`.
2. Quelle = `roh_path` (unveränderte Kundendatei), sonst `original_path`. Ziel =
   `<Stamm>_getaggt.pdf` im selben Upload-Ordner (der Export liest nur von dort). Der Lauf
   schreibt erst eine `.tmp.pdf`.
3. `pdf_tagging.taggen`: Sprache bestimmen, Konfiguration schreiben, Skript ausführen, Ergebnis
   prüfen (Strukturbaum vorhanden), Tag-Statistik vorher/nachher.
4. veraPDF (PDF/UA-1) über den Konverter-Dienst, in Klartext wie beim Word-Weg. Ausfall des
   Prüfdienstes ist kein Fehler.
5. Bilder des Dokuments **neu extrahieren** (jetzt über den Strukturbaum, `extraction_method`
   pdfix), in **einer Transaktion**: alte Bildzeilen löschen, neue eintragen
   (`main._bilder_uebernehmen`, derselbe Code wie beim Upload), vorhandene Alt-Texte übernehmen
   (`alt_texte_uebernehmen`: gleiche Seite UND Rechteck-Überlappung ≥ 0,5 bei Seitenkoordinaten,
   sonst Bild-Hash dHash mit Abstand ≤ 12 — der PDFix-Weg speichert als bbox nur die Bildmaße —,
   sonst Eindeutigkeit „ein altes Bild mit Text, ein neues Bild auf der Seite“; übernommen werden
   Alt-Text, Handtext, Langbeschreibung, Bildtyp, Status, Bewertung), Dokument umhängen (`original_path` →
   getaggte Datei, `roh_path` bleibt/wird gesetzt, `getaggt = 1`, Bericht), Projekt auf
   `extracted` mit neuen Zählern.
6. Alte Bilddateien, die kein neuer Eintrag nutzt, werden gelöscht. Credits werden **nur jetzt**
   verbucht (`usage_events` Quelle `tagging`, Aktion `pdf_tagging`, Preis je Seite).
7. Fehler: Dokument `fehler` mit nutzertauglichem Grund (nie Pfade oder Tracebacks), Projekt
   zurück auf den vorherigen Status, Temp-Datei weg, Bilder und Datei unverändert. Nach einem
   Server-Neustart gelten `laeuft`-Dokumente als abgebrochen (Start-Reparatur).

Neu-Taggen setzt immer auf `roh_path` auf (nie auf eine schon getaggte Fassung), die
Alt-Texte werden nach denselben Regeln wieder übernommen (E2E `verify_tagging_uebernahme.py`).

## Preis

`billing.AKTIONS_PREISE["pdf_tagging"] = 1` Credit je Seite — **vorläufig** (Steve 22.09.:
„preislich reden wir nochmal“; PDFix nennt ~1 Cent je Seite als eigene Kosten). Die Wache
vor dem Lauf verlangt das volle Guthaben, verbucht wird nach Erfolg.

## Bekannte Grenzen (Stand 22.09.2026)

- Auto-Tagging macht Fehler, die wir schon gesehen haben: nummerierte Überschriften
  („1. Ausgangslage“) werden Listenpunkte; nur die Schriftgröße entscheidet. Der Tagging-Schritt
  hat ein Feld `template` (PDFix-Tagging-Vorlagen mit Regeln), das bei Jörg leer ist → Frage an
  Jörg. Die zweite Stufe (Bauplan Schritt 5) ist die KI-Korrektur nach dem Tagging (Modell
  sieht Seitenbild und Tag-Liste, ändert über die SDK-Funktionen SetType/MoveChild nur bei
  hoher Sicherheit).
- „Make Accessible Docling“ (KI-Layoutanalyse von PDFix) meldet bei uns „Invalid input
  parameter“ → braucht Einrichtung, Frage an Jörg.
- Der Titel-Rückfall auf den Dateinamen nimmt den Servernamen der Datei; bei Dokumenten ohne
  H1 und ohne Info-Titel entsteht so ein technischer Titel (KI-Korrektur später).
- Prod: siehe Lizenz und Testmodus.

## Tests

```
# Unit (Container): Konfiguration, Sprache, Übernahme, Lauf im Testmodus, Drift der Voreinstellung
docker cp tests/test_pdf_tagging.py inkludocs-staging:/app/tests/ && \
docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_pdf_tagging.py -v
# Drift-Test der Betriebsfassung (Original + markierte Zeilen)
docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_pdfix_skript_drift.py
# End-to-End gegen Staging (Projekt anlegen, ungetaggtes PDF hochladen, taggen, prüfen, Datei laden):
python3 tests/e2e/verify_tagging.py https://staging.inkludocs.inklutec.de <mail> <pw> [--behalten]
```

Regression der bestehenden PDFix-Skripte beim SDK-Update 8.7.10 → 9.3.0 (22.09.2026): Alt-Text-
Export/Import und Formular-Export/Import liefern unter 9.3.0 identische CSVs, Alt-Texte und
Quickinfos (Vergleich Container 8.7.10 gegen venv 9.3.0 auf denselben Dateien).

## Oberfläche: Ansicht „Dokument“ (Schritt 4, 22.09.2026)

Ein PDF-Projekt (Werkzeug `pdf`) hat jetzt zwei Ansichten wie ein Word-Projekt: **„Dokument“**
(`frontend/dokument.js`) und **„Alt-Texte“** (app.html). Die Zeile „Ansicht“ im Projektkopf
listet „Dokument“ ganz oben (Michael 21.09.); Adresse `?ansicht=dokument`, Browser-Zurück geht.
**Startansicht bleibt vorerst „Alt-Texte“** (eine Zeile in `aktuelleAnsicht()`, Entscheidung
Steve/Michael offen). Formulare behalten ihre eigene Ansicht (formular.js), Web/Grafik haben keine
Ansichten. Gäste sehen keine Ansichts-Wahl.

Aufbau (Screenreader-Kette): H1 Projekt · Ansichts-Wahl (select + Knopf „Öffnen“) · H2 „PDF
hinzufügen“ (Upload, Wortlaut je Ansicht) · H2 „Dokumente (n)“ · je Datei eine Karte
(`section.dok-karte`, wie die Ablage): Vorschaubild der ersten Seite mit Alt-Text, H3 „Dokument
n: Name“ mit Stand-Badge, Beschreibungsliste (Stand, Seiten, Sprache, Struktur, Bilder),
Knöpfe „Barrierefrei machen“ / „Neu taggen“ (im Namen: Seiten und Credits), „Alt-Texte
bearbeiten“ (wechselt die Ansicht), „Getaggte PDF herunterladen“ (nach dem Lauf), „Umbenennen“,
„Löschen“, ein `<output>` je Karte für den Laufstatus, darunter die Klappe „Bericht lesen“
(Zeit, Sprache mit Quelle, Struktur vorher/nachher, Titel, Bilder/Übernahme, Hinweise,
Testmodus, PDF/UA-Prüfpunkte). Rückfrage als `<dialog id="dkLaufDialog">` (Umfang, Preis,
Guthaben, Neu-Taggen-Hinweis, Testmodus-Hinweis; Abbrechen links / Start rechts; 402 → die
gemeinsame Credits-Meldung). Während eines Laufs pollt die Ansicht alle 2,5 s, aktualisiert nur
bei Zustandswechsel, und meldet das Ende in der Laufmeldung (Fokus, `<output>`).

Endpunkte dazu (tagging_api.py): `GET /api/projects/{id}/dokument-ansicht` (Projekt ohne
Serverpfade, je Dokument Anzeige-Felder, Seiten, Struktur, Tagging-Stand, Ablage-Zähler) und
`GET /api/projects/{id}/documents/{doc}/vorschau` (PNG der ersten Seite: Seitenansicht aus der
Extraktion, sonst eigenes Rendering, gecacht).

Weitere Änderungen in app.html: `istPdfDateityp()`, `aktuelleAnsicht()` für PDF, Optionen der
Ansichts-Wahl je Dateityp, Ansage „Ansicht Dokument geöffnet.“, Weiche in `showProject`,
Upload-Wartetext „Die PDF wird gelesen …“ und Abschluss-Ansage in der Dokument-Ansicht, Fokus
auf die H3 der neuen Karte, Upload-Hinweistext je Ansicht (behebt auch: in der
Übersetzungs-Ansicht eines Word-Projekts stand noch „Daraus werden die Bilder extrahiert“, Steve
22.09.), Status-Badge „Wird gelesen“ bei Projektstatus `extracting`.

Klicktest: `tests/e2e/ui_dokument.py` (legt sein Projekt selbst an; Upload in der Ansicht,
Rückfrage, Lauf bis fertig, Bericht, Download, Wechsel der Ansichten, axe).

Offen: Startansicht, Ansicht „Dokument“ auch für Word und Formulare (dann „Quickinfos“ als
dritte Ansicht), Knopf „Komplett barrierefrei machen“ (Kette Tagging → Alt-Texte → Quickinfos),
Ablage-Eintrag nach dem Tagging, Namen der Ansichten (Steve klärt mit Michael).

## Station „Quickinfos“ und Werkzeugliste nach Dateiart (22.09.2026, Steves Go)

- **Werkzeugliste** (`backend/tools.py`): „PDF-Dokumente“ (Stationen Dokument, Alt-Texte, Quickinfos),
  „Word-Dokumente“, „Webseiten“, „Grafiken“. Die Kennung `formular` bleibt gültig (alte Projekte,
  API), ist aber nicht mehr im Anlege-Menü (`sichtbar=False`); der Platzhalter „Barrierefreie PDFs
  erstellen“ (`pdf-a11y`) entfällt. Namen sind Anzeigetexte und jederzeit änderbar.
- **PDF-Projekt mit Formularfeldern:** Beim Upload liest `_extract_document` nach den Bildern
  zusätzlich die Felder (`formular_api.felder_fuer_dokument_extrahieren`, nur Werkzeug `pdf`, nur
  wenn `validiere_formular` Felder meldet). Ein Fehler dort löscht nichts; es gibt dann nur keine
  Station. `documents.extraction_method` bleibt der Bild-Weg. `/api/projects/{id}` und
  `dokument-ansicht` liefern `hat_felder` bzw. `felder` je Dokument; die Ansichts-Wahl zeigt
  „Quickinfos“ nur bei `hat_felder > 0`, die Karte den Knopf „Quickinfos bearbeiten“.
- **Quickinfo-Ansicht = formular.js** (`Formular.showProject`), unverändert in Form und Funktion;
  die Formular-Endpunkte akzeptieren jetzt Projekte mit Werkzeug `pdf` (`_projekt_des_nutzers`).
  Eigenständige Formular-Projekte (`pdfform`) laufen wie bisher ohne Ansichts-Wahl.
- Offen: Gast-Prüfung der Quickinfos in PDF-Projekten (Freigabe öffnet heute die Alt-Text-Prüfung),
  Chatbot-Werkzeugsatz je Ansicht (heute nach `project.tool`), Kette „Komplett barrierefrei machen“.
- Tests: `tests/e2e/verify_pdf_quickinfos.py` (API), `tests/e2e/ui_pdf_quickinfos.py` (Klick).

## Kette „Komplett barrierefrei machen“ (22.09.2026, Steves Go)

`backend/kette_api.py`. Ein Knopf im Kopf der Ansicht „Dokument“ arbeitet die Stationen
nacheinander ab: Tagging für jede Datei ohne Struktur (`tagging_api.lauf_synchron`), Alt-Texte für
alle Bilder (`main.alttexte_lauf_fuer_kette`, derselbe Lauf wie „Alt-Texte generieren“), Quickinfos
für alle benannten Felder (`formular_api.quickinfos_lauf_fuer_kette`). Eine Rückfrage vorher
(`GET /api/projects/{id}/kette`: Umfang und Preis je Station aus denselben Zählungen wie die
Einzelstationen, Gesamtpreis, Guthaben), Start mit `POST` (402 wenn das Guthaben nicht für alles
reicht, 409 wenn etwas läuft, 429 Tageslimit). Jede Station bucht ihre Credits selbst. Eine
gescheiterte Station stoppt die Kette nicht; die Gründe stehen in der Zusammenfassung. Stand in
`projects.kette_json` (Statuskarte in der Ansicht, Polling alle 2,5 s nur in der Karte, am Ende
Laufmeldung); nach einem Neustart gilt eine laufende Kette als abgebrochen. Die Zahl der Bilder wird
nach dem Tagging neu bestimmt (Struktur-Extraktion), die Rückfrage sagt das. Test:
`tests/e2e/verify_kette.py` (echte Modellaufrufe). Offen: Ablage-Eintrag am Ende, Zusammenfassung
mehrsprachig (heute deutsch aus dem Server).

## Fertige PDF und Ablage (Stufe 1, 22.09.2026)

- **Ein Export für alles:** `main._build_pdf_for_document` schreibt nach den Alt-Texten auch die
  Quickinfos des Dokuments in dieselbe Datei (`_quickinfos_in_export`, Schreiber
  `formular_export.write_quickinfos_to_pdf`); danach wie bisher Sprache/Titel/Abschluss und Abnahme.
  Der PDF-Export („Als PDF“ in der Alt-Text-Ansicht, Knopf „Fertige PDF herunterladen“ auf der
  Karte der Dokument-Ansicht) liefert damit Struktur, Alt-Texte und Quickinfos in einer PDF.
- **Ablage:** Jeder PDF-Export (Einzeldokument, im ZIP je Dokument) legt einen Ablage-Eintrag an
  (`_pdf_in_ablage`, `art = pdf`): Kopie im Ablage-Ordner, PDF/UA-Prüfung über den Konverter
  (Ausfall = Hinweis), Zusammenfassung, Vorschaubild, Preis am ersten Eintrag. Antwort-Header
  `X-Ausgabe-Id`. Der rohe Tagging-Download (`…/tagging/datei`) bleibt als Endpunkt, ist aber
  nicht mehr auf der Karte.
- **Tagging-Konfiguration:** zusätzlich „Create Web Links“ (Adressen im Text werden Links).
  Getestet ohne Wirkung: „Überschriften-Ebenen aus dem Stil“ und „sequential headings“ ändern
  nichts am Fehler „nummerierte Überschrift als Liste“ (die Elemente sind schon als Liste getaggt);
  das bleibt Aufgabe der Korrektur (Stufe 2) oder von Docling.
- Tests: `tests/e2e/verify_export_komplett.py`, Klicktest `ui_dokument.py` (echter Download,
  Statuszeile, Ablage-Eintrag).
