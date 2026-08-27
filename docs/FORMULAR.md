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
| `backend/formular_export.py` | Schreiber: `/TU` je Feld auf Objekt-Ebene, inkrementell gespeichert; Nachprüfung (Präfix, Felder, Seitentext, Rücklesen über PDFix). |
| `backend/formular_api.py` | Eigener FastAPI-Router: Felder, Quickinfos, Stammdaten, Bilder, Export. Upload-Handler + Hintergrund-Extraktion. |
| `backend/pdfix_scripts/Formular_Export_Quickinfo.py` | Jörg Heines Export-Skript (Version 1.0.0.2, 25.08.2026), Linux-Anpassungen im Kopf dokumentiert. Original: `original_heine/Formulare_Export_07_r.py`. |
| `backend/pdfix_scripts/Formular_Import_Quickinfo.py` | Jörg Heines Import-Skript (Version 1.0.0.1, 19.08.2026), wählbar über `FORMULAR_WRITER=pdfix` (siehe Befund unten). Original: `original_heine/Formulare_Import_03.py`. |
| `backend/database.py` | Tabellen `formularfelder` und `stammdaten`; DSGVO-Löschung. |
| `backend/tools.py` | Werkzeug `formular` (Beta). |
| `backend/billing.py` | Aktion `formular_export` (5 Credits, nur Bezahlkonten). |
| `backend/main.py` | Projektanlage (`project_type` `pdfform`), Upload-Weiche, Router einhängen, Löschen von Projekt/Dokument räumt Felder und Bilddateien mit ab. |
| `frontend/formular.js` | Projektansicht für Formular-Projekte (eigene Datei, gleiche Form wie die Bild-Ansicht). |
| `backend/templates/app.html` | Upload-Block für Formulare, Weiche in `showProject`, Wartetexte. |
| `backend/locales/*/messages.po` | 78 neue Texte in de/en/fr/es/da/sv. |
| `tests/test_formular_roundtrip.py` | Unit-Tests (Container). Fixture `tests/fixtures/testformular_inkludocs.pdf` (fiktiv, Generator `make_testformular.py`). |
| `/home/claude/verify_formular.py` | End-to-End gegen Staging (Login, Negativfälle, Upload, Felder, Stammdaten, Export, Fremdzugriff). |
| `/home/claude/ui_formular.py` | Playwright-Klicktest mit axe. |

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

Standard ist PyMuPDF auf Objekt-Ebene: `/TU` als UTF-16-Hexstring in das
**Feld-Dictionary** (bei Feldern mit `Kids` — Radio-Gruppen, Felder auf
mehreren Seiten — in das Elternfeld, das den Namen trägt), gespeichert
**inkrementell**. Die Originalbytes bleiben unverändert; das wird in der
Nachprüfung als Präfix-Vergleich belegt. Weiter geprüft: Seitenzahl,
Feld-Erscheinungen (Seite, Name, Typ), sichtbarer Text ohne Widgets, und jede
geschriebene Quickinfo wird mit Heines Export-Skript (PDFix) zurückgelesen.
Schlägt ein Punkt fehl, wird der Export als Fehler gemeldet, nie still
ausgeliefert. Felder ohne Text bleiben unangetastet; eine Quickinfo wird nie
gelöscht.

**Befund 27.08.2026 (pdfix-sdk 8.7.10, Michaels Bankformular):** Nach
`PutString("TU")` + `Save` (kSaveFull wie kSaveIncremental) fehlen in der
gespeicherten Datei zufällig andere Widget-Annotationen (Lauf 1: Felder 5–8,
Lauf 2: keins, Lauf 3: 7–8); Öffnen + Speichern ohne Änderung ist sauber. Die
Nachprüfung fängt das ab. Deshalb ist Heines Import-Skript nur über
`FORMULAR_WRITER=pdfix` aktiv, bis Actino den SDK-Befund geklärt hat.

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
  ≤ 300, CSV-Import ≤ 1 MB / 5000 Zeilen, Feldarten und Quellen als feste
  Listen. Alle Ausgaben im Frontend laufen durch `escHtml()`.
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

## Grenzen (Stufe 1) und was folgt

- Keine KI-Vorschläge (Stufe 2: ein Aufruf je Seite, Text mit Positionen,
  Belegpflicht wie im Verify-Pass, Konsistenz über das Dokument).
- Stammdaten-Treffer nur exakt (Feldname, Beschriftung); unscharfe Treffer
  und Auto-Lernen in Stufe 3; Stammdaten-Seite im Kontobereich folgt.
- Keine Gast-Ansicht für Formular-Projekte; kein Chatbot-Anschluss.
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
# Klicktest (Playwright + axe)
/home/claude/.venv-pw/bin/python /home/claude/ui_formular.py <projekt-id>
# alles zusammen
bash /home/claude/formular_tests.sh
```
