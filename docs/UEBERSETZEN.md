# Übersetzen-Werkzeug: Word-Dokumente übersetzen, Formatierung bleibt

**Stand:** 18.09.2026, Stufe 1 (Word) auf Staging. Autor: Claude (InkluTec), Vorgaben Steve Weidel. Anlass: Mark Hounschild („das ganze Dokument auf Englisch, aber die Formatierung bleibt, wie sie ist“).

## Worum es geht

Kunden laden ein Word-Dokument hoch, wählen eine Zielsprache, InkluDocs übersetzt den
gesamten Text, und der Kunde lädt **seine eigene Datei in der Zielsprache** herunter.
Absätze, Formatvorlagen, Fettung, Links, Listen, Tabellen, Kopf-/Fußzeilen, Fußnoten,
Bilder, Kommentare, Änderungsverfolgung: alles bleibt Byte für Byte, wo es war. Nur
der Text ändert sich, dazu die Alternativtexte der Bilder, der Dokumenttitel und die
Sprachkennung (`w:lang`, `dc:language`), damit ein Screenreader das Dokument in der
richtigen Sprache vorliest (WCAG 3.1.1).

Entscheidungen (Steve, 18.09.2026):

- **Eigenes Werkzeug** `uebersetzen` mit **einer** Aufgabe und **einer** Ansicht; kein
  Umschalter im Word-Werkzeug (Steve: „überladen“).
- Der **Kern kennt keine Werkzeuge, nur Segmente** und ist überall anschließbar
  (PowerPoint, Chatbot, Übergabe-Knöpfe zwischen Werkzeugen, Public API).
- Übergabe zwischen Werkzeugen („Weiter im Word-Werkzeug“) ist der **zweite Schritt**:
  jedes Werkzeug gibt die ganze Datei als nächste Fassung weiter, Export an der letzten Station.
- Bearbeiten des **Originaltexts**: später. Die Durchsicht zeigt Original und Übersetzung;
  korrigiert werden kann **nur die Übersetzung**.
- Abrechnung: **1 Credit je angefangene 100 Wörter**, der Download ist kostenlos.

## Grundsatz: Struktur ist heilig, Text ist austauschbar

Eine `.docx` ist ein Zip mit XML-Teilen. Der Text steht in Textknoten `<w:t>` innerhalb
von Läufen `<w:r>`, die jeweils eine Formatierung tragen. Ein fettes Wort mitten im Satz
ist ein eigener Lauf:

```xml
<w:p>
  <w:r><w:t xml:space="preserve">Rund </w:t></w:r>
  <w:r><w:rPr><w:b/></w:rPr><w:t>zehn Prozent</w:t></w:r>
  <w:r><w:t xml:space="preserve"> der Bevölkerung …</w:t></w:r>
</w:p>
```

Werkzeuge wie ChatGPT erzeugen den Text neu und verlieren diese Struktur. InkluDocs
tauscht nur den Inhalt der Textknoten aus. Dem Modell geht der Absatz mit **Marken** je
Stück:

```
[[1]]Rund [[/1]][[2]]zehn Prozent[[/2]][[3]] der Bevölkerung …[[/3]]
```

Es übersetzt und behält die Marken (Reihenfolge darf sich ändern, wenn die Zielsprache
es verlangt); wir setzen Stück für Stück zurück. Kommt eine Marke nicht genau einmal
zurück, folgt **ein Korrekturversuch**, danach der **Ersatzweg**: die ganze Übersetzung
in das erste Stück, die anderen leer, Status „zusammengelegt“ mit sichtbarem Hinweis.
Der Ersatzweg ist ehrlich, nie still.

**Normalisierung:** Word speichert nach jedem Tippen neue Läufe mit gleicher Formatierung
(rsid) und Rechtschreibmarken (`w:proofErr`). Vor dem Segmentieren werden direkt
benachbarte reine Textläufe mit identischem `w:rPr` verbunden und die Rechtschreibmarken
entfernt (Word legt sie neu an). Dieselbe Normalisierung läuft beim Schreiben, deshalb
passen die Stücknummern. `strukturvergleich()` prüft am Ende, dass Original und
Ergebnis nach Normalisierung **bis auf die Texte identisch** sind; weicht etwas ab,
bricht der Export mit Fehler ab statt eine veränderte Datei auszuliefern.

**Was nicht übersetzt wird:** Feldbefehle (`w:instrText`), gelöschter Text der
Änderungsverfolgung, Absätze ohne Buchstaben (Zahlen, Zeichen), Text in Bildern
(nur Hinweis), Kommentare (Autorennotizen). Feldergebnisse (z. B. das
Inhaltsverzeichnis) werden übersetzt; ein Hinweis rät, in Word F9 zu drücken.

## Module

| Datei | Aufgabe |
|---|---|
| `backend/uebersetzung.py` | Kern: `segmentiere_docx()` (Segmente mit Stücken, Marken, Trennern, Kontext, Ort, Abschnitt), `text_mit_marken()`, `marken_zerlegen()`, `uebersetze_batch()` (Gemini über `llm_client.call_text_with_schema`, Schema `UebersetzungBatchOutput`, Korrekturversuch, Ersatzweg), `schreibe_uebersetzung()` (byteidentischer Rückschreiber, Sprachkennung), `strukturvergleich()`, `credits_fuer()`, `ZIELSPRACHEN` (22 Einträge mit regionalen Varianten: Englisch GB/USA/Australien, Deutsch DE/AT/CH, Französisch FR/CH, Spanisch Spanien/Lateinamerika, Portugiesisch PT/BR, Niederländisch NL/BE; Kennung = Word-Sprachkennung, Modell bekommt Schreibweise/Datumsformat der Variante). |
| `backend/uebersetzung_api.py` | Router: Upload-Handler + Hintergrund-Segmentierung, Lesen, Vorschau, Lauf (Pakete je Dokument, Guthaben und Tageslimit je Paket, Abbruch, Handarbeit während des Laufs gewinnt), Handkorrektur, Export (einzeln/ZIP, im Executor). Anschlusspunkte: `segmentiere_und_speichere()`, `lauf_starten()`, `export_vorbereiten()` + `export_bauen()`. |
| `backend/database.py` | Tabelle `uebersetzung_segmente`; Löschung bei Konto/Projekt/Dokument. |
| `backend/tools.py` | Werkzeug `uebersetzen` „Dokumente übersetzen“ (Beta). |
| `backend/billing.py` | Aktion `uebersetzung` (1 Credit je 100 Wörter), zählt im Tageslimit. |
| `backend/main.py` | `TOOL_PROJECT_TYPE["uebersetzen"] = "docx-uebersetzung"`, Upload-Weiche (nur .docx), Router, Löschpfade. |
| `frontend/uebersetzen.js` | Ansicht: H1 Projekt, H2 Dokument, H3 Abschnitt (nach Überschrift 1; dazu Kopfzeile, Fußzeile, Fußnoten, Bilder, Dokumenttitel), H4 „{Art} {n}: {Anfang}“; je Absatz Original als Text, Textarea „Übersetzung“ (Auto-Save 800 ms), Status-Badge, Hinweis; Filter-Fieldset Alle / Nur mit Hinweis / Nur noch nicht übersetzt; Dialog „Übersetzen“ (Zielsprache, zwei Schalter, Rückfrage mit Wörtern/Credits/Guthaben); Export-Dialog; Fortschrittskarte mit Abbrechen; Laufmeldung. |
| `backend/templates/app.html` | Skript eingebunden, Upload-Block, Weiche in `showProject`, Wartetexte, Dialogtexte Umbenennen/Löschen (`uebdoc`). |
| `backend/locales/*` | 76 Texte × 6 Sprachen. |
| `tests/test_uebersetzung_kern.py` | 23 Unit-Tests ohne Modell (Marken, Whitespace, Ersatzweg, Rundreise über alle Word-Fixtures, Sprache, Idempotenz, XXE) plus die Regressionsfälle des Reviews vom 18.09. (Ersatzweg leert feste Stücke, Streu-Token, Textfeld-Fallback, styles ohne docDefaults, Sprachkennung, Titel gekappt). |
| `tests/e2e/verify_uebersetzen.py` | 49 End-to-End-Prüfungen gegen Staging mit echtem Modell (Negativfälle, Lauf, 409 während des Laufs, Handkorrektur, Export, Rücklesen: Fettung/Link/Tabelle erhalten, Sprachkennung, Fremdzugriff mit Zweitkonto). |
| `tests/e2e/ui_uebersetzen.py` | Klicktest (Playwright + axe): Werkzeugauswahl, Ansicht, Filter, Dialoge, Download, Tastaturweg. |
| `tests/fixtures/testvortrag_inkludocs.docx` | Fiktiver Vortrag mit Fettung/Kursiv mitten im Satz, Hyperlink, Liste, Tabelle, Kopfzeile (Generator `make_testvortrag.py`, braucht python-docx). |

## Datenfluss

1. **Projekt anlegen** mit Werkzeug `uebersetzen`.
2. **Upload** `.docx` → `validiere_docx` im Request (400 mit Meldung), Dokumentzeile
   `extraction_method = uebersetzung`, Antwort `extracting`, Segmentierung im Hintergrund.
   Andere Dateitypen werden mit Meldung abgewiesen.
3. **Segmente** in `uebersetzung_segmente` (Position im Lesefluss: Haupttext, dann
   Kopf-/Fußzeilen, Fuß-/Endnoten; Alt-Texte/Bildtitel/Dokumenttitel als eigene Segmente).
   `documents.hinweise` trägt Quellsprache, Wörter, Absätze, Hinweise (z. B. Inhaltsverzeichnis).
4. **Rückfrage** `POST …/uebersetzung/vorschau`: Absätze, Wörter, Preis, Guthaben.
5. **Lauf** `POST …/uebersetzung/starten {zielsprache, alt_texte, sprache_setzen, document_id?}`:
   Pakete zu höchstens 30 Absätzen / 6.000 Zeichen, ein Modellaufruf je Paket
   (Gemini, Temperatur 0, Systemprompt mit Marken-Regeln, Dokumentblock als Daten).
   Guthaben und Tageslimit je Paket; Credits erst nach dem Schreiben; Absätze, die
   während des Laufs von Hand geändert wurden, bleiben (Schutz über `updated_at`).
   Vorhandene Übersetzungen werden ersetzt, Handkorrekturen nicht.
6. **Durchsicht**: Original und Übersetzung je Absatz; Korrektur der Übersetzung von Hand
   (`PATCH /api/uebersetzung/segmente/{id}`), Status `hand`; bei mehreren Formatstücken
   im Absatz gilt die Handkorrektur für den ganzen Absatz (Hinweis).
7. **Export** `POST …/export/uebersetzung {document_id?, filename?}`: je Dokument die
   übersetzte Datei (Name `<Dokument>_<Sprache>.docx`), mehrere als ZIP; kostenlos;
   Strukturvergleich vor der Auslieferung.

## Sicherheit

- XML-Parser ohne Entities/Netzwerk, Zip-Grenzen wie im Word-Werkzeug, nichts wird entpackt.
- Alle Endpunkte prüfen den Besitz über `projects.user_id`; kein Gastzugang.
- Zielsprache nur aus der festen Liste; JSON-Körper geprüft; Handtexte längenbegrenzt und
  von Steuerzeichen befreit; Texte im Frontend nur über `escHtml()`.
- Export liest Originale nur unter `UPLOAD_DIR`, schreibt atomar unter
  `RESULTS_DIR/…/_export/u_*` (die drei jüngsten Arbeitsordner bleiben), läuft im Executor;
  Kopfzeilen ASCII, Dateiname nach RFC 6266.
- Der Dokumenttext geht in einem abgegrenzten Datenblock an das Modell; der Systemprompt
  behandelt ihn als Daten (Prompt-Injection aus fremden Dokumenten).
- Serverpfade und Marken-Innereien gehen nie nach außen.

## Messung 18.09.2026 (Staging, Gemini 3.1 Pro)

- Fixture „Testdokument“ (24 Absätze, 207 Wörter): 27/27 Segmente mit sauberen Marken,
  ein Aufruf, 24 s, Struktur identisch.
- Fixture „Testvortrag“ (Fettung/Kursiv mitten im Satz, Link, Liste, Tabelle, Kopfzeile;
  15 Absätze, 91 Wörter): 15/15 mit sauberen Marken, 30 s, Fettung/Link/Tabelle im Export
  erhalten, Sprachkennung en-US gesetzt.
- Kosten: rund 1.400 Eingabe- und 700 Ausgabe-Token je Paket, Cent-Bereich je Dokument.

## Grenzen und was folgt

- Sprachvarianten (Steve 18.09.2026): „Welches Englisch?“ ist keine Kleinigkeit (colour/color, Datumsformat). Vorgabe im Dialog ist Englisch (Großbritannien); der Test prüft, dass keine US-Schreibweise entsteht.
- Word-Sprachkennung der Quelle ist oft falsch (Vorlagen mit en-US); das Modell erkennt die
  Ausgangssprache selbst, die Oberfläche zeigt nur die Zielsprache.
- Text in Bildern, Diagrammen und SmartArt wird nicht übersetzt.
- Wortstellung über Formatgrenzen (z. B. fettes Wort, das im Englischen ans Satzende
  wandert) übernimmt das Modell durch Umordnen der Marken; das ist erlaubt und getestet.
- Nächste Stufen: Chatbot-Werkzeug („übersetze das Dokument ins Englische“), PowerPoint
  über denselben Kern, Übergabe-Knöpfe zwischen Werkzeugen, Public API.

## Tests

```
# Unit (Container, 17 Tests; Fixtures vorher nach /app/tests/fixtures kopieren)
docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_uebersetzung_kern.py -v
# End-to-End gegen Staging (echtes Modell, wenige Credits; --behalten für den Klicktest)
INKLUDOCS_E2E_MAIL=… INKLUDOCS_E2E_PW=… python3 tests/e2e/verify_uebersetzen.py --behalten
# Klicktest (Playwright + axe)
INKLUDOCS_E2E_MAIL=… INKLUDOCS_E2E_PW=… /home/claude/.venv-pw/bin/python tests/e2e/ui_uebersetzen.py <projekt-id>
```

## Unabhängiges Review 18.09.2026 (zweiter Agent, 1 kritisch / 7 mittel / 12 niedrig) — Stand

Behoben: K1 Ersatzweg und Handkorrektur leerten feste Stücke nicht (Zahlen/Feldergebnisse
standen doppelt im Absatz); M1 Lauf überschrieb Handkorrekturen und berechnete schon fertige
Absätze erneut (jetzt: nur was in der Zielsprache fehlt, `quelle = hand` nie); M2 Doppelstart
(atomares Status-Update), Upload während des Laufs (409), Abrechnung nur für tatsächlich
geschriebene Segmente; M3 Textfelder doppelt (mc:Fallback wird nicht segmentiert, sondern aus
mc:Choice gespiegelt); M4 verrutschte Marken/Token im Stück landeten im Dokument; M5 Export-500
bei styles.xml ohne docDefaults; M6 Oberfläche baute #main alle 2,5 s neu (jetzt leichter
Statusabruf `?leicht=1`, Fortschritt in place, Timer stoppt beim Projektwechsel, Textfelder
während des Laufs gesperrt); M7 gescheiterte Segmentierung ohne Grund (jetzt `lauf_hinweis`
als JSON, Anzeige im Kopf); N1 Titel/Sprachkennung in den Datenblock bzw. validiert; N2
Längengrenzen auch für Alt-Texte/Titel; N3 Export nach reiner Handübersetzung (Sprache aus
Projekt, kein „None“ im Namen); N4 erstes Paket muss bezahlbar sein, Fortschritt ohne
abgewählte Alt-Texte; N5 Arbeitsordner bei Export-Fehler weg; N6 ZIP-Namen eindeutig; N7
Abbruchzustand bleibt sichtbar; N8 einsame Surrogate → 400; N9 Waisen-Upload beim Start-
Aufräumen gelöscht; N10 Download-Name aus `filename*`; N11 Arabisch: Hinweis zur
Schreibrichtung; H8 `total_images` bleibt 0.

Offen (bewusst, repo-weite Muster): H2 „Gespeichert“ im Label (wie formular.js); H4
Guthabenprüfung und Buchung nicht atomar; H5 Originale angehängter Dokumente beim Projekt-
Löschen; N12 Quelltext, der selbst „[[1]]“ enthält (sehr selten). Als sauber bestätigt:
Zugriffskontrolle, SQL, Pfade, XML/Zip, Header, XSS, Abrechnungspfade, Nebenläufigkeit,
Logs, Barrierefreiheit.
