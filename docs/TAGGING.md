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

## Bekannte Grenzen (Stand 22.09.2026, PDFix-Weg)

Gilt für den reinen PDFix-Weg (`PDF_TAGGING_WEG` nicht gesetzt). Der Weg „Struktur zuerst“ löst die Überschriften-Fehler und hat eigene Grenzen, siehe dort.


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

## Strukturlesung, Hörprobe und Strukturansicht (22.09.2026, Steves Go)

Steves Vorgabe: kein PDF-Viewer im Browser (nicht verlässlich), sondern das, was in den Tags steht,
selbst lesen und hörbar machen. Der echte Test bleibt Acrobat + Screenreader beim Kunden; ein
NVDA-Protokoll auf einem Windows-Server ist als Premium-Stufe vorgemerkt.

- **Eigenes PDFix-Skript** `backend/pdfix_scripts/Struktur_Export.py` (kein Heine-Skript, die Regel
  „Original unverändert“ gilt hier nicht): läuft den Tag-Baum in Lesereihenfolge ab und schreibt je
  Element `id` (Pfad im Baum), `typ`, `tiefe`, `seite`, `text` (über MCIDs wie im Alt-Text-Export, mit
  Heuristik für Einzelzeichen-PDFs), `alt`, `actual`, `lang`, bei Tabellen `zeilen`/`spalten`, bei
  Formularfeldern `feldname`/`quickinfo` (OBJR → Anmerkung → `/T`/`/TU`, bei Optionsfeldern vom
  Elternfeld). Elemente mit gesammeltem Text (LI, TD, TH, Caption, Note, Figure, Form) werden nicht
  weiter abgestiegen (sonst stünde der Text doppelt). Aufruf `-i <pdf> -o <json>`, Exit 3 = keine Tags.
- **Modul `backend/pdf_struktur.py`:** `lesen()` (Subprocess, Cache `<pdf>.struktur.json` im
  Upload-Ordner, gültig solange die PDF nicht neuer ist), `hoerprobe()` (Zeilen wie beim Word-Weg:
  „Überschrift Ebene 1: …“, „Liste mit n Einträgen“, „Tabelle mit r Zeilen und c Spalten“, „Kopfzeile:
  … | …“, „Grafik: Alt-Text“ / „Grafik ohne Alt-Text“, „Formularfeld vorname: Quickinfo“, Seitenmarken,
  Zusammenfassung als dritte Zeile), `html_ansicht()` (semantisches HTML: h1–h6 mit Versatz, p, ul/li,
  table/th/td, figure, Formularfelder als Absätze mit Rolle; alles escaped).
- **Endpunkt** `GET /api/projects/{id}/documents/{doc}/struktur` (`?erneuern=1` liest neu): `verfuegbar`,
  `grund` (ungetaggt / keine Tags / Zeitüberschreitung), `info` (Seiten, Sprache, Elemente), `hoerprobe`,
  `zusammenfassung`, `seite_url`. Quickinfos aus der Datenbank ergänzen die `/TU`-Werte der Datei.
- **Seite** `/struktur/{projekt}/{dokument}` (`templates/struktur.html`, nur Besitzer, sonst Login/404):
  H1 „Strukturansicht: Name“, Einleitung (ehrlich: was hier fehlt, fehlt auch im Screenreader), „Zurück
  zum Projekt“, Kennzahlen, Inhalt als Webseite (PDF-H1 wird h2), Hörprobe als Klappe.
- **Karte in der Ansicht „Dokument“:** Link „Strukturansicht öffnen“ und Klappe „Hörprobe lesen“ (lädt
  erst beim Aufklappen), beides nur bei getaggten Dokumenten.
- **Grenzen:** Verschachtelte Listen erscheinen flach; Text wird je Element auf 600 Zeichen gekürzt;
  Artefakte (ausgeblendete Inhalte) sind absichtlich nicht dabei — genau wie im Screenreader.
- Tests: `tests/test_pdf_struktur.py` (Hörprobe, HTML, Escaping, Cache), `tests/e2e/verify_struktur.py`
  (getaggtes Formular: Felder, Tabelle, Seite, Rechte), Klicktest `ui_dokument.py` Abschnitt B2.

## Automatische Prüfung (Schritt 5, erste Fassung, 22.09.2026, Steves Go)

Steves Rahmen: Gemini 3.1 Pro, Qualität vor Kosten; das Modell zieht die Infos aus dem Tagging UND
schaut sich die Seite optisch an; es darf nichts „korrigieren“, was richtig ist; die Prüfung wird mit
Credits berechnet (Preis vorläufig 2 je Seite, `billing.AKTIONS_PREISE["pdf_pruefung"]`).
Warum der Schritt nötig ist: veraPDF prüft die Form (jedes Element getaggt, Alt-Text da, Sprache
gesetzt), nicht den Inhalt der Tags. „1. Ausgangslage“ als Listenpunkt statt Überschrift besteht
veraPDF; ein Screenreader-Nutzer verliert die Überschriften-Navigation.

- **Modul `backend/pdf_pruefung.py`:** `MODELL` (EINE Stelle, ENV `PDF_PRUEFUNG_MODEL`, sonst
  `llm_client.MODEL_GENERATE`; Modellrouter später), `seitenbild()` (PyMuPDF 110 dpi, Cache
  `<pdf>.pruef_p<n>.png`), `zeilen_fuer_seite()` (Strukturliste `E<id> ROLLE: Text`, Alt-Texte, Felder),
  `nachpruefung()` (Kennung muss auf der Seite existieren, sonst „niedrig“ + Hinweis; Doppelmeldungen
  weg), `pruefe_dokument()` (bis `MAX_SEITEN` = 60; Seitenfehler werden Hinweise, alle Seiten
  fehlgeschlagen = Fehler).
- **Prompt:** `prompts/builders/pdf_pruefung.py` (Prüfauftrag 1–7: Rollen, Ebenen, Reihenfolge,
  Tabellen, Grafiken, Fehlt, Sprache; Regel „im Zweifel kein Befund“; Strukturliste als Datenblock),
  Schema `prompts/components/schemas/pdf_pruefung.py` (`PruefSeiteOutput`: befunde mit element, art,
  befund, vorschlag, beleg, sicherheit; zusammenfassung).
- **Endpunkte** `GET/POST /api/projects/{id}/documents/{doc}/pruefung`: nur getaggte Dokumente (400),
  409 bei laufender Prüfung oder laufendem Tagging, 402 Guthaben, 429 Tageslimit; Lauf im Executor,
  Stand in `documents.pruefung_status/pruefung_bericht`, Fortschritt (Seite a von b) über
  `dokument-ansicht` → `tagging.pruefung`. Credits erst nach erfolgreichem Lauf (Quelle `pruefung`).
- **Karte:** Klappe „Automatische Prüfung“ mit Erklärung, Knopf „Prüfung starten“ / „Erneut prüfen“
  (Seiten und Credits im Namen), Statuszeile (output, Fokus beim Start), Bericht als nummerierte
  Liste: Seite, Rolle, Text, Befund, Vorschlag, Beleg, Sicherheit-Badge; Hinweise; Satz „ändert nichts
  an der Datei, ersetzt keinen echten Screenreader-Test“. Laufmeldung am Ende wie beim Tagging.
- **Noch nicht:** Korrektur über PDFix-Befehle (rename/move tags) — erst nach dem Messlauf mit zehn
  echten Dokumenten und nur für Befunde mit hoher Sicherheit; Chatbot-Werkzeug; Preisentscheidung.
- Tests: `tests/test_pdf_pruefung.py` (Nachprüfung, Strukturliste, Bericht mit Modell-Attrappe),
  `tests/e2e/verify_pruefung.py` (echter Modelllauf), Klicktest `ui_dokument.py` Abschnitt B3.

## Chatbot: EIN Werkzeugsatz für PDF-Projekte (22.09.2026, Steves Go)

Steves Frage: „Kann der Chatbot über die Ansichten arbeiten?“ Befund: Der Verlauf hängt schon immer am
Projekt (chat_messages.project_id), nicht an der Ansicht. Die Lücke war der Werkzeugsatz: er hing am
Projekttyp, und PDF-Projekte bekamen nur die Bild-Werkzeuge. Jetzt (Michaels Wunsch vom 18.09.,
„Werkzeugsatz nach Dateiart“):

- `agent_loop._werkzeugsatz`: PDF-Projekt (project_type pdf, tool pdf) = Bild-Werkzeuge + Feld-Werkzeuge
  (Quickinfos, `definitions_formular`) + PDF-Werkzeuge (`tools/definitions_pdf.py`), Executor
  `ToolExecutor(pdf=True)`, Systemprompt `SYSTEM_AGENT + prompts/system_pdf.SYSTEM_PDF`. Alte
  Formular-Projekte (tool formular) und Word-Projekte unverändert. Neue Dateiarten (PowerPoint …) = ein
  weiterer Zweig hier, ein Definitions-Modul, ein Prompt-Zusatz.
- `tools/pdf.py`: `dokument_stand` (kostenlos, erster Schritt), `barrierefrei_machen`,
  `komplett_barrierefrei_machen` (Kette, `kette_api.starten_von_aussen` auf der Hauptschleife),
  `hoerprobe_lesen` (seitenweise), `pruefung_starten`, `pruefbericht_lesen`, `exportiere_fertige_pdf`
  (Anhang unter der Antwort + Ablage-Eintrag, Auslöser `bot`). Dieselben Kernfunktionen wie die
  Oberfläche; Rückfrage in zwei Schritten mit derselben Angebots-Logik wie bei Word
  (`ausgaben._angebot_merken/_angebot_einloesen`): erst ohne `bestaetigt` nur Preis + Guthaben, Ja in
  eigener Nachricht, 15 Minuten, Preis unverändert, eine bezahlte Aktion je Nachricht.
- Lange Läufe (Tagging, Kette, Prüfung) starten im Hintergrund (Thread bzw. Hauptschleife); der Bot
  meldet „läuft“ und liest den Stand mit `dokument_stand`. Er behauptet nie, etwas sei fertig, was das
  Werkzeug nicht als fertig gemeldet hat.
- `tools/formular._projekt` erlaubt jetzt `tool IN ('formular','pdf')`, damit die Feld-Werkzeuge im
  PDF-Projekt arbeiten.
- Tests: `tests/e2e/verify_chat_pdf.py` (LLM-gesteuert: Stand, Tagging mit Rückfrage und Ja, Hörprobe,
  Prüfung mit Rückfrage, Bericht, fertige PDF mit Anhang und Ablage; Quickinfo-Werkzeug im PDF-Projekt).
- Später: Chat als Seitenleiste außerhalb des Hauptbereichs (bleibt beim Ansichtswechsel stehen; eigener
  Landmark, Fokus-Regeln, Live-Region), Kennung je Nachricht, aus welcher Ansicht sie kam.

## Messwerte, Doppelbeleg und Korrektur (Stufe 2, 22.09.2026, Steves Go)

Steves Frage „bist du dir sicher?“ und sein Wunsch: nur reparieren, was wirklich falsch ist, jedes Dokument
sieht anders aus, der Kunde wählt, was er bezahlt.

- **Messwerte** (`backend/pdf_messung.py`, PyMuPDF, deterministisch): je Textzeile Schriftgröße, fett,
  Lage, „allein“ (eigener Block mit Luft darüber und darunter), je Seite die Fließtextgröße; je Element über
  den normalisierten Textanfang zugeordnet, dazu die Zahl der Textzeilen im Element (zusammengezogene
  Zellen). Stehen in der Strukturliste des Prompts (`[16 pt fett, allein]`) und im Bericht (`messung`).
- **Doppelbeleg** (`pdf_pruefung.doppelbeleg`): ein Befund ist `auto` (automatisch korrigierbar) nur, wenn
  das Modell „hoch“ sagt UND eine unabhängige Quelle dieselbe Richtung zeigt: Absatz→Überschrift nur bei
  allein stehend und hervorgehoben (≥ 1,15 × Fließtext oder fett); Überschrift→Absatz nur bei nicht
  hervorgehoben; Kopfzelle→Datenzelle nur aus der Tabellenlage (Wert rechts neben einer Kopfzelle, oder
  Zahl in einer Datenzeile unter einer Kopfzeile). Alles andere bleibt Hinweis (`doppelbeleg` erklärt warum).
- **Ebene aus Schriftgröße** (`ebenen_aus_groesse`): Das Modell entscheidet „ist eine Überschrift“, die
  Messung „welche Ebene“ — Rang der Größe im Dokument zusammen mit den vorhandenen Überschriften
  (24 pt = H1, 16 pt = H2, 11 pt = H3). Messlauf Rechnungen: Modell schlug H1 vor, Messung setzte H3.
- **Korrektur** (`backend/pdf_korrektur.py` + eigenes Skript `pdfix_scripts/Korrektur_Anwenden.py`): nur
  `SetType` auf Elemente, die über die Objektnummer (`obj`, Strukturlesung Version 2) gefunden werden;
  Sicherung `<pdf>.vor_korrektur.pdf` vorher, veraPDF danach, Bericht in `documents.korrektur_bericht`,
  Prüfbericht bekommt `korrigiert_am` („von vor der Korrektur“). Kostenlos. `rueckgaengig()` stellt die
  Sicherung her (Zeitstempel auf jetzt, damit die Zwischenspeicher neu lesen).
- **Endpunkte:** `POST …/documents/{doc}/korrektur` (Body `erneut_pruefen`: hängt die bezahlte
  Nachprüfung an — 402/429 wie bei der Prüfung; 400 ohne Prüfung oder ohne Doppelbeleg; 409 schon
  korrigiert oder Lauf aktiv), `POST …/korrektur/rueckgaengig`. Stand in `pruefung.korrektur`.
- **Karte:** je Befund Badge „Automatisch korrigierbar“, Messung und Doppelbeleg; Knöpfe „n Befunde
  korrigieren (kostenlos)“ und „Korrigieren und erneut prüfen (c Credits)“; Korrektur-Block mit Änderungen,
  veraPDF danach, „Korrektur rückgängig machen“; Hinweis, wenn der Prüfbericht von vor der Korrektur stammt.
- **Chatbot:** `korrektur_anwenden` (Zwei-Schritt, nennt die Änderungen, fragt nach der Nachprüfung),
  `korrektur_rueckgaengig`; `pruefbericht_lesen` liefert Messung, Doppelbeleg und Korrektur-Stand.
- **Messlauf Rechnungen (Projekt 953 auf Staging):** eigene Rechnung 10 Befunde, 8 mit Doppelbeleg, 8
  korrigiert in 1,6 s (Adresszeilen → P, Rechnungsnummer → H2, Zwischenüberschriften → H3, drei Kopfzellen
  → Werte); Haunschild 3 Befunde, 1 korrigiert. Hinweise blieben: fette erste Adresszeile (Messung und
  Modell uneins), zusammengezogene Zahlungszellen (kein Umbenennen — nächste Stufe: Tabelle aus der
  Zeilenlage neu setzen).
- Tests: `tests/test_pdf_messung.py`, Doppelbeleg in `tests/test_pdf_pruefung.py`, `tests/e2e/verify_korrektur.py`,
  Klicktest `ui_dokument.py` Abschnitt B4 (nur bei Doppelbeleg-Befunden).

## Weg „Struktur zuerst“ (23.09.2026, Steves Go) — `PDF_TAGGING_WEG=struktur`

Statt PDFix die Struktur raten zu lassen und hinterher zu flicken, wird die Struktur VORHER bestimmt und PDFix schreibt sie nur noch. Modul `backend/pdf_struktur_tagging.py`, eigenes PDFix-Skript `backend/pdfix_scripts/Struktur_Schreiben.py` (kein Heine-Skript), Prompt `prompts/builders/pdf_struktur.py`, Schema `prompts/components/schemas/pdf_struktur.py`.

Ablauf je Dokument (in `tagging_api._lauf_sync`, wenn `PDF_TAGGING_WEG=struktur`):

1. **Struktur-HTML** rein rechnerisch aus der PDF (PyMuPDF): je Textzeile Kennung `s<Seite>z<n>`, Schriftgröße, Fettdruck, Lage; je Bild `s<Seite>b<n>`. Keine KI.
2. **Zuordnung** durch das Modell, ein Aufruf je Seite mit Seitenbild: nur Überschriften, Artefakte (Kolumnentitel, Seitenzahl, Verlagszeile), Bildunterschriften; je Bild inhaltlich oder Schmuck; hat die Seite eine Tabelle. Das Modell wählt nur Kennung und Rolle, es schreibt keinen Text.
3. **Nachprüfung + Stilprofil**: unbekannte Kennungen fallen weg, vergessene Bilder gelten als inhaltlich. Die Überschriften-EBENE kommt nicht vom Modell (seitenlokal), sondern aus dem Stil dokumentweit (Größe, fett), Titelseiten-Stile schieben nichts nach unten; Klammer-Pass in Lesereihenfolge: kein Ebenensprung, gleicher Stil im selben Abschnitt = gleiche Ebene.
4. **Schreiben** (`Struktur_Schreiben.py`): ganzseitige Form-XObjects (Hintergrund) von der Erkennung ausschließen und als Artefakt markieren (sonst hängt PDFix den Text darauf in ein Bild), Pfade/Schattierungen und Schmuckbilder im Inhalt als Artefakt markieren (Stand abends, siehe unten), Tabellen aus dem Vordurchgang mit Zellen aus unseren Zeilen, Listen, mehrzeilige oder überlappende Rollen und Artefakt-Zeilen als initiale Elemente, `CreateElements`, übrige Rollen per `SetTag`, Bilder als Figure (Alt bleibt leer, der Export trägt ihn nach), Tabellenköpfe: erkannte Kopfzeile/-spalte, sonst erste Zeile ab 3 Spalten, sonst erste Spalte; `AddTags`. Seiteninhalt bleibt byteweise gleich.
5. **Technische Schritte**: Jörgs Make Accessible mit `konfig_erzeugen(..., struktur_vorgegeben=True)` — ohne `add_tags` und ohne `fix_headings` (füllt Sprünge mit LEEREN H-Tags, die ein Screenreader als „Überschrift, leer“ liest).

Bericht wie beim PDFix-Weg plus `weg: "struktur"` und `struktur: {modell, modell_dauer_s, ueberschriften, artefakte, bilder_inhaltlich, bilder_schmuck, verworfen, stilprofil, geklammert, geschrieben}`; Hinweise nennen Zeilen ohne Element (Vollständigkeit) und Seiten ohne KI-Zuordnung.

Ergebnis am Ritterturnier (Michael Karbe, 15 Seiten): veraPDF PDF/UA-1 ohne Befund, 39 Überschriften ohne Sprung und ohne leere Tags, verlorener Text von Seite 3 im Baum; KI-Gegenprobe 40 statt 76 Befunde (Rest: Alt-Text-Platzhalter, umbrochene Listenpunkte, verschmolzene Bilder).

Grenzen (Stand 23.09. abends): gescannte PDFs ohne Textebene gehen nicht (Fehlermeldung); Text ÜBER Bildern und Beschriftungen in Karten/Diagrammen landen bei PDFix im Bild (gelesen wird der Alt-Text — für Karten muss er die Beschriftung tragen); benachbarte Zeichnungen werden zu einer Figure verschmolzen; verschachtelte Listen erscheinen flach; Tabellen-KANDIDATEN findet PDFix, ob sie übernommen werden und welche Zellen sie haben, bestimmen wir. Testmodus verfälscht PDFix-Text mit „*“, daher läuft aller Textabgleich über PyMuPDF. Schalter/Env: `PDF_TAGGING_WEG`, `PDF_STRUKTUR_MODEL`, `PDF_STRUKTUR_DPI`, `PDF_STRUKTUR_HINTERGRUND_ANTEIL`, `PDF_STRUKTUR_PARALLEL` (gleichzeitige Modellaufrufe, Vorgabe 4), `PDFIX_STRUKTUR_TIMEOUT`. Tests: `tests/test_pdf_struktur_tagging.py` (Struktur-HTML, Nachprüfung, Stilprofil, Plan, Konfiguration, Schreibweg mit Modell-Ersatz).

### Formulare, Vektorgrafik, Tabellen, mehrzeilige Überschriften (23.09.2026 abends, Mannheimer-Antrag)

Anlass: Der Mannheimer-Antrag (Versicherungsformular, 6 Seiten) lief technisch durch, die Hörprobe war aber schlecht: Text in Grafiken, Ankreuzkästchen als „Grafik: Bullet“, Titel auf drei Überschriften verteilt, Beitragstabelle in Einzelabsätze zerfallen. Vergleich mit Jörgs Originalweg am selben Dokument: dort dieselben Fehler, zusätzlich eine leere H1. Ursache: Der Weg „Struktur zuerst“ gab PDFix noch nicht genug vor.

- **Vektorgrafik ist kein Bild.** `Struktur_Schreiben.py` schließt Pfad- und Schattierungsobjekte (Kästchen, Rahmen, Linien, Flächen) vor der Erkennung aus und markiert sie als Artefakt; sonst hält PDFix einen Rahmen mit Text darin für ein Bild und hängt den Text hinein. Komplexe Zeichnungen (Kurven oder mehr als 8 Linienzüge, zu Gruppen zusammengefasst, mindestens 0,4 % der Seite) kommen im Struktur-HTML als `<img data-art="vektorzeichnung">` zum Modell; was es als inhaltlich einstuft (Diagramm, Illustration), bleibt für PDFix sichtbar und wird Figure. Schalter im Plan: `vektor_artefakt` (Vorgabe an).
- **Steuerzeichen-Zeilen** (Formulare setzen z. B. `\x08` als Platzhalter) fallen aus dem Struktur-HTML (`druckbar()`); sie verfälschten die Vollständigkeitsprüfung.
- **Tabellen:** Vordurchgang immer. Tabelle mit Linienraster (`kTableGraphic`) wird übernommen; ohne Raster nur mit Zustimmung des Modells oder bei eindeutiger Geometrie (mind. 3 Zeilen, 3 Spalten, 80 % der Zeilen mehrfach gefüllt) — Gemini hatte die Beitragstabelle übersehen. Zellen weiterhin aus unseren Zeilen.
- **Mehrzeilige Überschriften/Bildunterschriften** (`rollen_gruppen`): direkt untereinander stehende Zeilen mit gleicher Rolle und gleichem Stil (höchstens eine Zeilenhöhe Abstand) sind EIN Element; getrennt wird nur, wenn in derselben Spalte etwas dazwischen steht (Nachbarspalten wie „GS-Nr.:“ trennen nicht). Das Skript legt solche Gruppen als initiales Element mit Rolle an.
- **Seiten parallel:** Modellaufrufe laufen zu `PDF_STRUKTUR_PARALLEL` (Vorgabe 4) gleichzeitig, danach ein zweiter Anlauf für fehlgeschlagene Seiten.
- **Vollständigkeit im Urteil:** Bericht `struktur.zeilen_gesamt` / `zeilen_ohne_element`; mehr als 2 % (mindestens 3 Zeilen) ohne Element → Gesamturteil „Struktur unvollständig, bitte Hörprobe prüfen“, nie „in Ordnung“.
- **CSV-Export:** Zellen, die mit `=`, `+`, `-`, `@`, Tab oder CR beginnen, bekommen ein Hochkomma (Formel-Injektion; Text stammt aus fremden PDFs).

Messung (gleiche Gemini-Zuordnung, nur Schreibweg geändert): Mannheimer 96 → 0 Scheingrafiken, 23 → 0 Grafiken mit Text, 35 → 0 Zeilen ohne Element, Titel eine H1, 4 Tabellen (Beitragsermittlung als Tabelle); Infografik Diagramm bleibt Figure; Hofor-Bericht 7 Tabellen, 20 Grafiken (18 Fotos + Diagramme); Ritterturnier unverändert. Tests: `FormularUndVektorTest`, `UrteilTest`, `test_csv_formel_injektion_entschaerft`.

### Diagramme, Karten und überlappende Überschriften (23.09.2026 nachts, Hofor-Bericht)

- **Vektorgruppen:** einzelne Zeichnungen werden auf den sichtbaren Seitenteil beschnitten; eine Einzelzeichnung über 30 % der Seite ist Fläche/Hintergrund und kein Diagramm-Baustein. Eine Gruppe entfällt nur, wenn ein Rasterbild sie zu mindestens 50 % abdeckt (dann vertritt das Foto sie); kleine eingebettete Bildstücke machen ein Diagramm nicht überflüssig.
- **Ganzseitige Karten/Diagramme:** Eine Gruppe über `PDF_STRUKTUR_HINTERGRUND_ANTEIL` (60 %) der Seite gilt nur dann als Hintergrund, wenn sie aus weniger als `MIN_ZEICHNUNGEN_GROSSE_GRAFIK` (40) komplexen Zeichnungen besteht. Hofor S. 11 (Versorgungskarte): vorher rund 50 Ortsnamen als lose Absätze, jetzt zwei Karten-Figures.
- **Überlappende Überschriften:** Rollen, deren Rahmen sich überschneiden (Hofor-Titel: „Årsrapport“ im Glyphenrahmen der 242-pt-Ziffern „2025“), legt `Struktur_Schreiben.py` als initiale Elemente an, kleinste zuerst — sonst verschmolz PDFix beide und eine Überschrift ging verloren. Das Stilprofil gibt überlappenden Überschriften derselben Seite dieselbe Ebene, damit die Folge nicht von der Lesereihenfolge abhängt (vorher 2, 1, 3 = Sprung).
- Messung danach (gleiche Gemini-Zuordnung): Mannheimer, Infografik, Ritterturnier unverändert; Hofor 1 Vektorkarte, 14 Grafiken, 23 Überschriften; alle vier ohne Ebenensprung. Tests: `test_ganzseitige_karte_bleibt_grafik`, `test_ueberlappende_ueberschriften_gleiche_ebene`.
