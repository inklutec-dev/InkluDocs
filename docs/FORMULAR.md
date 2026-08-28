# Quickinfo-Werkzeug: Quickinfos für PDF-Formulare

Stand 27.08.2026 (Stufe 1, Staging). Steve Weidel (INKLUTEC) mit Fable 5;
Skripte und fachlicher Anstoß von Michael Karbe und Jörg Heine (Actino Software GmbH).

## Worum es geht

Ein PDF-Formular ist für Screenreader-Nutzer nur bedienbar, wenn jedes
Eingabefeld eine **Quickinfo** trägt (PDF-Eintrag `/TU`, „Tooltip“): den
zugänglichen Namen, den der Screenreader vorliest, sobald der Nutzer in das
Feld springt. Er sieht die Beschriftung daneben nicht. Formulare von Banken
und Versicherungen haben 50 bis 100 Felder; sie von Hand in Acrobat zu
beschriften ist Tagesarbeit — und dieselben Felder kommen in über 100
Formularen wieder vor.

Das Werkzeug liest alle Felder, zeigt sie mit Beschriftung, Abschnitt und
Bildausschnitt, lässt Quickinfos von Hand eintragen oder aus einer
**Stammdaten-Bibliothek** des Kontos übernehmen und schreibt sie in eine Kopie
der PDF zurück. Stufe 2 (KI-Vorschläge mit Beleg) und Stufe 3 (Lernen, Team,
API) folgen; die Bausteine sind darauf angelegt.

## Was eine gute Quickinfo ist

Kurzfassung von WCAG 3.3.2 / 4.1.2 und Matterhorn-Protokoll 28 (PDF/UA):
- sagt, **was einzugeben ist**, nicht nur, wie das Feld heißt („Name des
  Kontoinhabers“ statt „Name“);
- trägt den **Gruppenkontext**, wenn das Feld allein nicht eindeutig ist
  („Antragsteller: Vorname“);
- nennt das **Format**, wenn das Formular eines vorgibt („Geburtsdatum,
  Format TT.MM.JJJJ“);
- bei Kästchen und Auswahlknöpfen **Frage und Option** („Zahlungsweise:
  monatlich“);
- Pflichtfelder als Pflicht kennzeichnen; Sprache des Formulars; kurz.

## Module

| Datei | Aufgabe |
|---|---|
| `backend/formular_processor.py` | Leser: Feldliste über PDFix (Heines Export-Skript), Geometrie/Beschriftung/Abschnitt/Seitentext über PyMuPDF, Bildausschnitt und Seitenansicht mit nummerierten Rahmen. Vorprüfung `validiere_formular()`. |
| `backend/formular_export.py` | Schreiber: zwei Wege (Heines PDFix-Import mit Lizenz, sonst PyMuPDF inkrementell), Nachprüfung (Felder, Seitentext, Rücklesen über PDFix, Präfix beim PyMuPDF-Weg). |
| `backend/formular_api.py` | Eigener FastAPI-Router: Felder, Quickinfos, Stammdaten, Bilder, Export. Upload-Handler + Hintergrund-Extraktion. |
| `backend/pdfix_scripts/Formular_Export_Quickinfo.py` | Jörg Heines Export-Skript (Version 1.0.0.2, 25.08.2026), Linux-Anpassungen im Kopf dokumentiert. Original: `original_heine/Formulare_Export_07_r.py`. |
| `backend/pdfix_scripts/Formular_Import_Quickinfo.py` | Jörg Heines Import-Skript (Version 1.0.0.1, 19.08.2026), Standard-Schreibweg mit Lizenz (siehe Schreibweg unten). Original: `original_heine/Formulare_Import_03.py`. |
| `backend/database.py` | Tabellen `formularfelder` und `stammdaten`; DSGVO-Löschung. |
| `backend/tools.py` | Werkzeug `formular` (Beta). |
| `backend/billing.py` | Aktion `formular_export` (5 Credits, nur Bezahlkonten). |
| `backend/main.py` | Projektanlage (`project_type` `pdfform`), Upload-Weiche, Router einhängen, Löschen von Projekt/Dokument räumt Felder und Bilddateien mit ab. |
| `frontend/formular.js` | Projektansicht für Formular-Projekte (eigene Datei, gleiche Form wie die Bild-Ansicht). |
| `backend/templates/stammdaten.html` | Seite „Meine Stammdaten“ (Route `/stammdaten`, Seitenleiste unter „Meine Prompts“): Liste mit Suche, Anlegen/Bearbeiten/Löschen, CSV-Import/-Export; Muster `prompts.html`. |
| `backend/templates/app.html` | Upload-Block für Formulare, Weiche in `showProject`, Wartetexte. |
| `backend/locales/*/messages.po` | 78 neue Texte in de/en/fr/es/da/sv. |
| `tests/test_formular_roundtrip.py` | Unit-Tests (Container). Fixture `tests/fixtures/testformular_inkludocs.pdf` (fiktiv, Generator `make_testformular.py`). |
| `/home/claude/verify_formular.py` | End-to-End gegen Staging (Login, Negativfälle, Upload, Felder, Stammdaten, Export, Fremdzugriff). |
| `/home/claude/ui_formular.py`, `ui_stammdaten.py` | Playwright-Klicktests mit axe (Projektansicht, Stammdaten-Seite). |

## Datenmodell

**`formularfelder`** — ein Eintrag je Feld eines Dokuments. Schlüssel zum
Zurückschreiben ist `anker` = voller Feldname (in gültigen PDFs eindeutig);
namenlose Felder bekommen `#<n>` und können nicht beschrieben werden.
Wichtige Spalten: `feld_art` (text, checkbox, radio, dropdown, liste, button,
signatur, unbekannt), `page_number` (erste Erscheinung), `seiten` (alle),
`rect_*`, `beschriftung` + `beschriftung_lage` (links/oben/rechts/innen),
`gruppe` (Abschnittsüberschrift), `umfeld`, `optionen`, `pflicht`,
`ausgefuellt` (nur ja/nein — **nie der Wert**), `quickinfo_original` (aus der
PDF), `quickinfo` (aktuell), `quelle` (pdf, hand, stammdaten, ki),
Bildpfade, `page_text`.

**`stammdaten`** — Bibliothek je Konto (`user_id`): `beschriftung` +
`beschriftung_norm` (Vergleichsform: klein, ohne Doppelpunkt/Sternchen am
Ende), `feld_art`, `feld_name`, `quickinfo`, `sprache`, `herkunft` (hand, feld,
import, ki), `verwendet`. Schlüssel für Aktualisierung statt Dublette:
Beschriftung+Feldart, sonst Feldname+Feldart.

Ein Formular-Projekt hat `projects.tool = 'formular'` und
`project_type = 'pdfform'`; Dokumente wie beim PDF-Werkzeug
(`documents`, `extraction_method` `formular-pdfix` bzw. `formular`, Hinweise
als JSON in `documents.hinweise`). Die Bildtabelle `images` wird **nicht**
benutzt.

## Datenfluss

1. **Projekt anlegen**: `POST /api/projects {name, tool: "formular"}`.
2. **Upload**: `POST /api/upload` (project_id + PDF). Vorprüfung im Request
   (`validiere_formular`: PDF lesbar, kein Passwort, ≤ 300 Seiten,
   1 … 2000 Felder), sonst 400 mit Meldung. Antwort sofort `extracting`;
   Extraktion im Hintergrund, Frontend pollt `GET /api/projects/{id}`.
3. **Extraktion** (`formular_processor.extract_formular`): Feldliste über
   PDFix-Subprocess (Rückfall PyMuPDF), Geometrie und Texte über eine
   **widgetfreie Arbeitskopie** (siehe Datenschutz), Bilder nach
   `RESULTS_DIR/<user>/<projekt>/doc<n>/` (`feld_<n>.png`,
   `p<n>_seitenansicht.png`). Danach werden die Stammdaten des Kontos auf
   Felder ohne Quickinfo angewendet (Treffer über Feldname, sonst
   Beschriftung, jeweils mit verträglicher Feldart).
4. **Bearbeiten**: `GET /api/projects/{id}/felder` liefert Projekt, Dokumente
   (mit Zählern und Hinweisen), Felder (Seitentext nur am ersten Feld jeder
   Seite) und Stammdaten-Treffer je Feld. `PATCH /api/felder/{id}`
   speichert (Auto-Save 800 ms), `POST …/original`, `POST …/stammdaten`
   (Feld → Bibliothek), `POST …/stammdaten-uebernehmen` (Bibliothek → Feld),
   `POST /api/projects/{id}/stammdaten-anwenden`.
5. **Export**: `POST /api/projects/{id}/export/formular` (einzeln oder ZIP,
   Kopfzeilen `X-Export-Tagged/Total/Open/Warnings`, 5 Credits bei
   Bezahlkonten) und `…/export/formular_csv` (Feldliste, kostenlos; Spalten
   1–5 wie Heines Format).

## Schreibweg und Nachprüfung

Zwei Schreibwege, Auswahl über `FORMULAR_WRITER` (ohne Variable entscheidet
die Lizenz):
- **pdfix** (Standard mit Lizenz): Jörg Heines Import-Skript setzt `/TU` über
  das SDK im Feld-Dictionary und speichert die Datei neu (kSaveFull).
- **pymupdf** (Standard ohne Lizenz): `/TU` als UTF-16-Hexstring in das
  **Feld-Dictionary** (bei `Kids` — Radio-Gruppen, Felder auf mehreren Seiten —
  in das Elternfeld), **inkrementell** gespeichert; die Originalbytes bleiben
  unverändert (Präfix-Vergleich).

Beide Wege durchlaufen dieselbe Nachprüfung: Seitenzahl, Feld-Erscheinungen
(Seite, Name, Typ), sichtbarer Text ohne Widgets, und jede geschriebene
Quickinfo wird mit Heines Export-Skript zurückgelesen. Schlägt ein Punkt
fehl, wird der Export als Fehler gemeldet, nie still ausgeliefert. Felder ohne
Text bleiben unangetastet; eine Quickinfo wird nie gelöscht.

**Befund 27.08.2026 (pdfix-sdk 8.7.10, Michaels Bankformular):** In der
**Testversion** fehlten nach `PutString("TU")` + `Save` zufällig andere
Widget-Annotationen (Lauf 1: Felder 5–8, Lauf 2: keins, Lauf 3: 7–8; Öffnen +
Speichern ohne Änderung sauber). **Mit Lizenz** (Actino, eingetragen 27.08.
abends in `.env.staging`, Durchreichung in `docker-compose.staging.yml`):
8/8 Läufe ohne Verlust, auch über Heines Skript, kein „Trial“-Vermerk mehr.
Der Feldverlust ist eine Eigenheit der Testversion; die Nachprüfung bleibt.

## Datenschutz und Sicherheit

- **Feldwerte werden nie gespeichert** (weder in der Datenbank noch in Bildern
  noch im Seitentext): PyMuPDF liest und rendert Widget-Erscheinungen mit —
  deshalb arbeiten Textextraktion und Bilder auf einer Kopie ohne Widgets;
  die Feldrahmen werden selbst gezeichnet. Gespeichert wird nur
  „ausgefüllt ja/nein“. Belegt durch Unit-Test `test_datenschutz_wert_nirgends`
  und E2E-Check.
- Alle Endpunkte prüfen den Besitz (JOIN projects.user_id); Stammdaten sind
  strikt je Konto. Bilddateien werden nur unterhalb von `RESULTS_DIR`
  ausgeliefert, Originale nur unterhalb von `UPLOAD_DIR` gelesen
  (Realpath-Schutz wie beim Word-Export).
- Eingaben: Steuerzeichen entfernt, Quickinfo ≤ 1000 Zeichen, Textfelder
  ≤ 300, CSV-Import ≤ 1 MB / 5000 Zeilen, höchstens 20 000 Stammdaten je
  Konto, Feldarten als feste Liste, JSON-Körper geprüft (400 statt 500).
  Alle Ausgaben im Frontend laufen durch `escHtml()`.
- CSV-Ausgaben (Feldliste, Stammdaten) sind gegen Formel-Injection geschützt
  (`_csv_safe` aus main.py: `=`, `+`, `-`, `@` am Zellanfang bekommen ein
  Apostroph); der Import nimmt dieses Apostroph wieder ab.
- Kopfzeilen der Export-Antwort sind reines ASCII, Dateinamen nach RFC 6266;
  Credits werden erst nach fertiger Antwort und nur bei tatsächlich
  geschriebenen Quickinfos verbucht. Der Export läuft im Executor (der
  Server bleibt währenddessen bedienbar), jede Anfrage in einem eigenen
  Arbeitsordner (`_export/f_*`, die drei jüngsten bleiben liegen).
- Fehlerpfad der Extraktion: Status `error` (bzw. `extracted` beim Anhängen),
  Dokument, Felder, Upload-Datei und Bildordner werden entfernt; beim Start
  werden hängende `extracting`-Formularprojekte zurückgesetzt.
- Stammdaten ersetzen nie Hand-Texte oder PDF-Originale und nie namenlose
  Felder; `nur_offene=false` ersetzt nur Texte, die selbst aus Stammdaten
  stammen.
- Unabhängige Review-Runde 27.08.2026: 32 Befunde, die kritischen und
  mittleren (Header-Zeichensatz, Credits vor Antwort, CSV-Formeln, Event-Loop,
  Temp-Dateien, hängender Status, Stammdaten-Überschreiben, Abrechnung ohne
  Leistung) behoben; offen geblieben sind geerbte Muster des Altbestands
  (doc_index ohne Sperre bei parallelen Uploads, Serverpfade in
  `GET /api/projects/{id}`).
- PDFix läuft als Subprocess mit Zeitlimit; verschlüsselte PDFs werden mit
  Meldung abgewiesen; Seiten- und Feldzahl sind begrenzt.
- Der Export ändert nichts außer `/TU` — byte-genau belegbar.

## Oberfläche (Barrierefreiheit)

Gleiche Form wie die Alt-Text-Ansicht: H1 Projekt, H2 Dokument (klappbar,
mit Umbenennen/Löschen), H3 Seite (klappbar, Seitenansicht mit nummerierten
Rahmen, Seitentext), H4 „Feld n, Feldart, Seite p“. Je Feld: Statusbadge
(Quickinfo fehlt / vorhandene aus der PDF / von Hand / aus Stammdaten),
Pflichtfeld, „bereits ausgefüllt“; Bildausschnitt (dekorativ, `alt=""`); ein
**Kontextabsatz** (Beschriftung mit Lage, Abschnitt, Optionen, Seiten,
technischer Name) — derselbe Kontext, den in Stufe 2 die KI sieht; die
Textarea „Quickinfo“ (`aria-describedby` auf den Kontext) mit Auto-Save und
„Gespeichert“; Knöpfe „Zurück auf Original“, „In Stammdaten übernehmen“,
„Aus Stammdaten übernehmen“ (mit sichtbarem Vorschlag). Oben: Export-Dialog
(natives `<dialog>`), „Stammdaten auf offene Felder anwenden“, Filter „Nur
offene Felder anzeigen“. Je Dokument die **Hörprobe**: eine Liste, wie ein
Screenreader das Formular Feld für Feld vorliest („ohne Bezeichnung“ bei
offenen Feldern). Alle Zustandswechsel werden über `announce()` angesagt.

## Stufe 2: KI-Vorschläge (Feld-Pass), 27.08.2026

- **Ein Modellaufruf je Seite** (`formular_ki.generiere_seite`): Sonnet über
  Bedrock (`BEDROCK_MODEL_GENERATE`, gleiches Modell wie die Alt-Texte),
  Tool-Use-Schema `QuickinfoSeiteOutput`, Temperatur 0 (Einzel-„Neu
  generieren“ 0,5). Eingabe: Textzeilen der Seite mit Positionen (aus der
  widgetfreien Kopie – nie Feldwerte), Felder mit Positionen, Pflicht,
  Optionen, geometrische Hinweise, schon beschriebene Quickinfos des Projekts
  (Konsistenz), eigener Prompt aus „Meine Prompts“, Sprache aus der
  Projekteinstellung. Kein Bild. Höchstens 40 Felder je Aufruf, Seitentext
  auf 12 000 Zeichen gekappt.
- **Prompt**: `prompts/builders/quickinfo.py` (Systemrolle + STILBLOCK nach
  WCAG 3.3.2/4.1.2 und Matterhorn 28); Schema in
  `prompts/components/schemas/quickinfo.py`. Der Seitentext steht in einem
  abgegrenzten Datenblock; der Systemprompt behandelt ihn als Daten
  (Prompt-Injection aus fremden PDFs).
- **Nachprüfung** (`formular_ki.nachpruefung`, deterministisch, kann die
  Sicherheit nur senken): Beleg steht im Seitentext (sonst *niedrig*), Beleg
  liegt in Feldnähe (sonst höchstens *mittel*), Regeln (Länge ≤ 200,
  Anleitungsfloskeln, Feldart im Text, Formatangabe ohne Vorkommen auf der
  Seite, „Pflichtfeld“ ohne Kennzeichnung). `konsistenz()`: gleiche
  Beschriftung + Feldart + Gruppe → gleicher Wortlaut.
- **Endpunkte**: `POST /api/projects/{id}/quickinfos/generieren` (Hintergrund,
  nur offene Felder, nie namenlose; Status `processing`, Fortschritt in
  `GET …/felder` → `generierung`; 1 Credit je Seite, Kontingent je Seite
  geprüft, Fehler je Seite statt je Projekt; danach Konsistenz-Lauf) und
  `POST /api/felder/{id}/generieren` (überschreibt bewusst, Variation,
  1 Credit). Beim Start werden hängende `processing`-Projekte zurückgesetzt.
- **Spalten** `formularfelder.sicherheit`, `beleg`, `ki_hinweise` (JSON);
  `quelle = 'ki'`.
- **Oberfläche** wie bei den Alt-Texten: „Alle generieren“ (nur Lücken),
  „Generieren“/„Neu generieren“ am Feld, „Zurück auf Original“; Badge
  „KI-Vorschlag, sicher/mittel/unsicher“, Beleg-Satz mit Hinweisen unter dem
  Eingabefeld, Filter „Nur unsichere KI-Vorschläge“, Fortschritt im Kopf,
  Ansage am Ende; Auswahl „Sprache der Quickinfos“ und „Gespeicherte
  Prompts“ über dieselben Endpunkte wie bei den Alt-Texten.
- **Abrechnung**: `quickinfo_generierung` 1 Credit je Seite (vorläufig, mit
  Michael zu klären). Das Tageslimit der Bilder greift nicht.
- **Tests**: `tests/test_formular_ki.py` (Nachprüfung, Konsistenz, Builder,
  Kontext ohne Feldwerte – ohne Modell), E2E `verify_formular.py` Abschnitt G
  (echte Generierung auf Staging), Klicktest.

## Gast-Ansicht: Freigabe zur Prüfung (28.08.2026)

Für Agenturen und Banken: Der Besitzer lädt über „Zur Prüfung freigeben“ einen
Gast ein (Rolle Herausgeber oder Lektorat, gleicher Dialog und gleiche Mail-
Mechanik wie bei Alt-Texten: Token + E-Mail-Gate, `sharing.py`). Der Gast sieht
dieselbe Formular-Ansicht (`frontend/formular.js` im Gast-Modus, `window.GUEST_MODE`)
unter `/freigabe/{token}` – Felder, Kontext, Ausschnitt, Seitenansicht, Hörprobe –
und darf:

- die Quickinfo von Hand ändern (`POST /api/freigabe/{token}/felder/{id}/quickinfo`,
  `quelle = gast`; ohne gesetztes Urteil springt das Feld für seine Rolle auf
  `in_bearbeitung`),
- je Feld ein Urteil setzen: Freigeben / Änderung wünschen (`POST …/review`,
  Status `freigegeben` / `zu_ueberarbeiten`; Lektorat zusätzlich `ruecksprache`),
  mit EINER optionalen Anmerkung (`formularfelder.review_note`),
- die Prüfung abschließen (`POST /api/freigabe/{token}/complete`, gemeinsam mit
  den Bild-Freigaben in `main.py`): Zusammenfassungs-Mail an den Besitzer mit
  Feld-Zählern und allen Anmerkungen.

Der Gast darf NICHT: generieren, Stammdaten sehen oder anwenden, exportieren,
hochladen, umbenennen, löschen. Alle Gast-Endpunkte prüfen Token + bestätigte
Gast-Sitzung (`_require_guest` aus `main.py`, per `Deps.require_guest`) und
liefern nur Felder DES freigegebenen Projekts; Serverpfade bleiben innen.

Datenmodell: Tabelle `feld_reviews (feld_id, role, status, reviewed_at)` –
Gegenstück zu `image_reviews`; `formularfelder.review_status` / `reviewed_at`
spiegeln das jüngste Urteil (Badge beim Besitzer, Zähler), `review_note` die
Anmerkung. Löschen von Dokument/Projekt/Konto räumt `feld_reviews` explizit mit
ab (SQLite erzwingt Fremdschlüssel nicht).

Besitzer-Seite: `GET /api/projects/{id}/felder` liefert `in_review`,
`share_roles` und je Feld `reviews` + `review_note`; die Ansicht zeigt bei
freigegebenem Projekt je Feld das Badge mit dem jüngsten Urteil (dieselben
Begriffe wie bei Bildern: Neu, In Bearbeitung, Herausgeber Freigabe, …) und die
Anmerkung als Klappe. `/api/review-overview` (Dashboard-Knopf „Geteilte
Projekte“, Seite /geteilte-projekte) zählt Formular-Projekte mit.

Barrierefreiheit: Urteil-Knöpfe sind echte Buttons mit `aria-pressed`, Status
steht als Text, Änderungen werden über `announce()` angesagt; Anmerkung als
natives `<details>`; Abschluss als natives `<dialog>` (Fokusfang, Escape).

## InkluAgent im Formular-Projekt (28.08.2026)

Derselbe Chat-Kasten wie bei den Alt-Texten (Knopf „Chatbot“ unter der
Feldliste, nur für den Besitzer), aber mit eigenem Fachteil und eigenem
Werkzeugsatz: `backend/inkluagent/prompts/system_formular.py` und
`backend/inkluagent/tools/formular.py` + `definitions_formular.py`. Die Weiche
liegt in `agent_loop._werkzeugsatz` (`project.tool == "formular"`), Bild- und
Feld-Werkzeuge können sich nicht vermischen. Speichern über den Chat läuft
durch dieselbe Nachprüfung wie der Feld-Pass; Generieren über den Chat ist
derselbe Feld-Pass wie der Knopf. Abrechnung wie in der Oberfläche (1 Credit
je erzeugter oder geänderter Quickinfo, Reden ist frei). Gesamtbild und
Kochrezept für weitere Werkzeuge: `docs/INKLUAGENT.md`.

## „Alle neu generieren“ (28.08.2026)

„Alle generieren“ füllt nur Lücken; sobald jedes Feld einen Text hat, wird
derselbe Knopf zu „Alle neu generieren“ (keine Rückfrage — der Knopf nennt
für Screenreader Anzahl der KI-Vorschläge und Credits, Muster wie das
einzelne „Neu generieren“). Er ruft `POST …/quickinfos/generieren` mit `{"modus": "ki_neu"}`: Der
Sammellauf fasst zusätzlich alle Felder mit `quelle = ki` an; Texte von Hand,
aus der PDF, aus Stammdaten oder vom Gast bleiben unberührt
(`_modus_bedingung`). 1 Credit je betroffener Seite.

## Redundanz-Regel der Nachprüfung (28.08.2026)

„Gruppe: … Gruppe …“ — steht der Präfix vor dem Doppelpunkt im Satz dahinter
noch einmal (Wortkern ohne Klammer-Nummer), entfernt `nachpruefung()` den
Präfix (Hinweis „Doppelte Gruppe entfernt“). Befund Bankformular: „Wirtschaftlich
Berechtigter [1]: Unterschrift des wirtschaftlich Berechtigten [1].“ Gilt für
Feld-Pass und InkluAgent (`update_quickinfo`).

## Seitenbild-Ausnahme im Feld-Pass (28.08.2026)

Der Feld-Pass arbeitet rein mit Text und Positionen. Hat eine Seite ein Feld
OHNE Beschriftung in der Nähe (`beschriftung` leer), geht zusätzlich die
gerenderte Seite mit nummerierten Feldrahmen (`p<n>_seitenansicht.png`, dieselbe
Datei wie die Klappe „Seitenansicht“) als Bild mit (`bedrock_client.
call_bedrock_with_schema` statt `call_bedrock_text_with_schema`); der Prompt
bekommt den Block SEITENBILD (Nummern im Bild = F<n>). Das Modell liest das
Layout wie ein Mensch, der Beleg bleibt die wörtliche Textstelle. Die
Nachprüfung bleibt gleich streng; Hinweise nennen „Zuordnung aus dem
Seitenbild“, betroffene Felder tragen „Seitenbild einbezogen“. Seiten mit
vollständig beschrifteten Feldern laufen unverändert nur mit Text. Gilt für
„Alle generieren“, „Generieren“ am Feld und den InkluAgent (`generate_quickinfo`).
Kosten: rund 1.500 Eingabe-Token je betroffener Seite.

## Grenzen (Stufe 1) und was folgt

- KI-Vorschläge: Eval-Korpus mit Soll-Quickinfos (Michaels Formulare,
  öffentliche Formulare) steht noch aus.
- Stammdaten-Treffer nur exakt (Feldname, Beschriftung); unscharfe Treffer
  und Auto-Lernen in Stufe 3.
- InkluAgent: „Alle offenen generieren“ läuft im Chat Feld für Feld (4–5 je Turn);
  für Massenläufe bleibt der Knopf „Alle generieren“ der bessere Weg.
- Gast-Ansicht: kein Nachrichten-Verlauf je Feld, keine Rücksprache-Liste in der
  Abschluss-Mail (beides Stufe 2 der Gast-Ansicht, wie bei Bildern).
- Beschriftungs-Erkennung ist geometrisch (links/oben/rechts/innen,
  Abschnitt = fette/größere Zeile oder Zeile mit Doppelpunkt); Konstanten in
  `formular_processor.py` sind an echten Kundenformularen nachzujustieren.
- Tab-Reihenfolge, fehlende Tags und Prüfbericht (Matterhorn 28) sind Stufe 4.

## Tests

```
# Unit (Container)
docker exec inkludocs-staging python3 -m unittest /app/tests/test_formular_roundtrip.py -v
# End-to-End gegen Staging (legt Projekt an und löscht es; --behalten für Hör-/Klicktest)
python3 /home/claude/verify_formular.py https://staging.inkludocs.inklutec.de <mail> <pw> /home/claude/testformular_inkludocs.pdf
# Klicktest (Playwright + axe); zweites Argument = Gast-Token für die Gast-Ansicht
/home/claude/.venv-pw/bin/python /home/claude/ui_formular.py <projekt-id> [gast-token]
# alles zusammen
bash /home/claude/formular_tests.sh
```
