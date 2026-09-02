# Multi-Datei-System (Phase 1, nur PDF)

**Stand:** 08.06.2026 — Branch `feature/multi-datei`
**Autor:** Cody (InkluTec)
**Status:** Phase 1 — nur PDF-Werkzeug. Web- und Grafik-Werkzeug bleiben in dieser
Phase unangetastet und werden separat erweitert.

Dieses Dokument richtet sich an Entwicklerinnen und Entwickler — auch externe —
die InkluDocs warten oder erweitern. Es beschreibt das Datenmodell, die
Migration, die betroffenen Endpunkte und Frontend-Bausteine sowie das
Export-Verhalten der neuen Dokument-Ebene.

## Worum es geht

Vor diesem Branch entsprach ein PDF-Projekt genau einer hochgeladenen PDF-Datei
(Modell „1 Projekt = 1 Datei"). Karbe hat sich gewuenscht, mehrere PDFs in
einem Projekt buendeln zu koennen (z. B. einen Geschaeftsbericht und seinen
Anhang). Multi-Datei Phase 1 dreht das Modell auf „1 Projekt = mehrere
Dokumente = je viele Bilder/Seiten". Jede hochgeladene PDF wird zu einem
Dokument mit eigener Ueberschrift im UI; Sammel-Generierung,
Einzel-Generierung und Export wurden entsprechend nachgezogen.

## Datenmodell

Die Aenderung ist additiv und reversibel — bestehende Projekte (Single-PDF)
funktionieren unveraendert.

### Neue Tabelle `documents`

```
documents (
    id                INTEGER PRIMARY KEY,
    project_id        INTEGER NOT NULL  -> projects.id (ON DELETE CASCADE)
    doc_index         INTEGER NOT NULL  -- 1-basiert, fortlaufend pro Projekt
    original_filename TEXT NOT NULL
    display_name      TEXT NULL         -- optionaler Anzeigename
    original_path     TEXT NOT NULL     -- Pfad der hochgeladenen PDF
    extraction_method TEXT DEFAULT 'fitz'  -- 'fitz' | 'pdfix'
    total_images      INTEGER DEFAULT 0
    created_at        TEXT
)
INDEX idx_documents_project ON documents(project_id, doc_index)
```

Datei: `backend/database.py` — die Tabelle wird im `init_db()`-Block angelegt
und ist idempotent (`CREATE TABLE IF NOT EXISTS`).

### Neue Spalte `images.document_id`

```
ALTER TABLE images ADD COLUMN document_id INTEGER
```

Eine `images`-Zeile zeigt nun auf ihr Dokument. Foreign-Key-Constraint
absichtlich nicht hart erzwungen, weil SQLite-Migrationen sonst ALTER-Probleme
machen — die Konsistenz wird durch die Anwendungslogik gehalten (Inserts setzen
das Feld; Loeschungen raeumen images + documents auf).

### Backfill (bestehende PDF-Projekte)

In `_migrate_columns()` laeuft direkt nach dem Schema-Add ein Idempotent-Backfill:

```python
SELECT id, filename, original_path, extraction_method, total_images
FROM projects
WHERE (project_type = 'pdf' OR tool = 'pdf')
  AND id NOT IN (SELECT project_id FROM documents)
  AND filename IS NOT NULL AND filename != ''
```

Fuer jedes Treffer-Projekt wird genau eine `documents`-Zeile angelegt
(`doc_index=1`, `original_filename=projects.filename`,
`original_path=projects.original_path`) und alle vorhandenen
`images.document_id IS NULL` werden auf die neue Dokument-ID gesetzt.

So funktioniert Single-PDF unveraendert: Es existiert genau ein Dokument 1, das
wie bisher exportiert/angezeigt wird.

## Backend-Aenderungen

Datei: `backend/main.py`.

### `_handle_pdf_upload(file_path, filename, user, project_id=None)`

Neue Logik:

* Wenn `project_id` vorhanden ist und `proj.total_images > 0`, wird die
  bestehende Sperrung gegen ein zweites PDF entfernt (`is_append=True`).
* Es wird der naechste `doc_index` bestimmt
  (`MAX(doc_index)+1` pro Projekt).
* Eine neue `documents`-Zeile wird angelegt und ihre ID gemerkt.
* Bilder werden in einen eigenen Unterordner pro Dokument extrahiert:
  `RESULTS_DIR/<user_id>/<project_id>/doc<doc_index>/`. Das verhindert
  Dateinamens-Kollisionen (`p1_img1.png` zwischen Doc 1 und Doc 2).
* Beim INSERT in `images` wird `document_id` mitgegeben, `image_index`
  laeuft project-weit fortlaufend hoch (kompatibel zur bisherigen
  Bedeutung).
* `projects.total_images` wird zur Summe aller Bilder ueber alle Dokumente;
  `projects.extraction_method` bleibt der Wert vom ersten Dokument
  (nur Anzeige im Projekt-Kopf).
* `projects.filename` und `projects.original_path` werden NUR beim ersten
  Dokument gesetzt, damit der Projekt-Kopf eine stabile „Hauptdatei" hat.

Antwort-JSON (relevant fuer UI):

```json
{
  "ok": true,
  "project_id": 17,
  "document_id": 42,
  "doc_index": 2,
  "filename": "anhang.pdf",
  "total_images": 23,
  "added_images": 9,
  "project_type": "pdf",
  "extraction_method": "fitz",
  "appended": true
}
```

### Sammel-Generierung — `_process_project`

Bewusst **unveraendert** in seiner Selektion:

```sql
SELECT * FROM images WHERE project_id = ? AND status = 'pending' ...
```

Da neu hinzugefuegte Bilder mit `status='pending'` ankommen und bereits
fertige (`status='done'` oder `status='error'`) ignoriert werden, ueberschreibt
der Klick „Alle Alt-Texte generieren" nach dem Anhaengen einer neuen PDF
ausschliesslich die Bilder der neuen PDF — bereits generierte Texte aus
vorherigen Dokumenten bleiben unangetastet. Das ist der Kern des
gewuenschten Verhaltens und musste daher gar nicht erst angepasst werden.

Einzel-Generierung (`regenerate_image`) ueberschreibt wie zuvor genau das
adressierte Bild.

### Neue Endpunkte

* `PATCH /api/projects/{project_id}/documents/{document_id}` —
  Setzt `documents.display_name` (Nice-to-have). Leer = wieder NULL.

### Angepasste Endpunkte (Export)

Alle Exporte gehen jetzt ueber einen JSON-Body und kennen zwei optionale
Parameter:

```json
{ "document_id": 42, "filename": "Geschaeftsbericht" }
```

Verhalten:

* `document_id` gesetzt -> nur dieses Dokument, EINZELNE Datei (direkter
  Download).
* `document_id` nicht gesetzt + mehrere Dokumente -> alle als ZIP. Innerhalb
  der ZIP heisst jede Datei nach dem Schema
  `<doc_index>_<dokumentname>.<endung>` (z. B. `01_Bericht.pdf`).
* `document_id` nicht gesetzt + nur ein Dokument -> Einzeldatei (kein
  unnoetiges ZIP).
* `filename` (ohne Endung) ueberschreibt den vorgeschlagenen Dateinamen.
  Leer oder weggelassen -> Server waehlt einen Standard (Dokumentname bei
  Einzel-Export, Projektname bei Alle-Export).

Betroffene Routen:

* `POST /api/projects/{id}/export`           — PDF/A-Tag-Export
* `POST /api/projects/{id}/export/json`
* `POST /api/projects/{id}/export/csv`
* `POST /api/projects/{id}/export/xlsx`

Die Implementierung kapselt die Erzeugung pro Dokument in den Helfern
`_load_pdf_export_units`, `_build_pdf_for_document`,
`_load_export_units_for_table`, `_build_json_bytes`, `_build_csv_bytes`,
`_build_xlsx_bytes` und `_table_export_dispatch` (Excel-Export 28.08.–02.09.2026
entfernt, auf Kundenwunsch zurück).

### `GET /api/projects/{project_id}`

Liefert jetzt zusaetzlich `documents: [...]` (sortiert nach `doc_index`).
Existierende Felder bleiben unveraendert. Bilder werden nach
`(doc_index, page_number, image_index)` sortiert.

## Frontend (`frontend/app.html` + `frontend/style.css`)

### Upload-Block

`uploadBlockHtml(project)` zeigt fuer PDF-Projekte **dauerhaft** den Knopf
„PDF hinzufuegen" — analog zum URL-Feld beim Web-Werkzeug. Ist bereits ein
Dokument vorhanden, lautet die Ueberschrift „Weitere PDF hinzufuegen" und der
Hinweis erklaert, dass bestehende Alt-Texte unangetastet bleiben.

### Gruppierung der Bilderliste

`renderImages()` gruppiert nun zweistufig: `document_id` -> `page_number` ->
Bilder. Bei PDF-Projekten wird jedes Dokument in ein eigenes
`<details class="doc-section">` mit `<summary>` und `<h2>` gewickelt; darin
liegen die bisherigen `<details class="page-section">` mit `<h3>` fuer die
Seite. Die einzelne Bild-Card nutzt `<h4>`. So entsteht die Hierarchie:

```
h2  Dokument 1: Geschaeftsbericht.pdf
  h3  Seite 1 (3 Bilder)
    h4  Bild 1, Seite 1
    h4  Bild 2, Seite 1
    h4  Bild 3, Seite 1
  h3  Seite 2 (1 Bild)
    h4  Bild 4, Seite 2
h2  Dokument 2: Anhang.pdf
  h3  Seite 1 (2 Bilder)
    h4  Bild 5, Seite 1
    h4  Bild 6, Seite 1
```

Bei Web/Grafik-Projekten oder PDFs ohne `documents`-Eintraege rendert die
Funktion eine flache Liste (h2 Seite, h3 Bild) — abwaertskompatibel.

Der Auf/Zu-Zustand der Dokument- und Seitenklappen wird in den
JS-Variablen `openDocs` und `openPages` pro Projekt gehalten, damit
Polling-Updates oder das Anhaengen einer weiteren PDF die bisherigen
Klappen nicht zuruecksetzen.

### Export-Panel

`renderExportScopeBlock(documents)` baut eine Radio-Gruppe (Fieldset/Legend)
mit „Alle Dokumente (eine ZIP-Datei)" als Default und je einem Eintrag pro
Dokument. Bei einem einzelnen Dokument entfaellt die Auswahl. `doExport()`
liest die Auswahl mit `readExportScope()` und schickt ggf. `document_id`
sowie den optionalen `filename` als JSON-Body an den Server. Der
heruntergeladene Dateiname kommt entweder aus dem `Content-Disposition`-Header
des Servers oder faellt auf den Eingabewert zurueck.

### A11y

* Live-Region `#liveRegion` (`aria-live=polite`) meldet beim Anhaengen einer
  weiteren PDF: „PDF hinzugefuegt als Dokument N. X neue Bilder. Insgesamt Y
  Bilder im Projekt." Bei der ersten PDF wie bisher „N Bilder im Projekt."
* Fokus wandert nach erfolgreichem Append automatisch zur Summary des neuen
  Dokument-Blocks, sodass Tastatur-/Screenreader-Nutzer direkt im Kontext
  der neuen PDF landen.
* Saubere Heading-Hierarchie (s. o.).
* Die Dokument-Auswahl beim Export ist eine Radio-Gruppe in `<fieldset>` mit
  `<legend>` — keine Dropdown-Fummelei.
* `details/summary` bleiben Tastaturzugaenglich; die Doc-Summary erbt die
  vorhandenen Focus-Styles (siehe `style.css` `.doc-section > summary:focus-visible`).

## Filesystem-Layout

Pro Dokument wird ein eigener Bilder-Unterordner angelegt:

```
/app/data/results/<user_id>/<project_id>/doc<doc_index>/
                                              p1_img1.png
                                              p1_seitenansicht.png
                                              ...
                                          _export/
                                              <generierte Export-Dateien>
```

Dadurch koennen mehrere Dokumente Datei-Namen wie `p1_img1.png` parallel
fuehren, ohne sich gegenseitig zu ueberschreiben.

## Was bewusst nicht in dieser Phase steckt

* **Web- und Grafik-Werkzeuge** bleiben unveraendert (siehe Phase 2).
* **PDFix `page_num`-Bug** (getaggte PDFs, alle Bilder kommen als Seite 1) wird
  separat von Karbes Liste angegangen — die neue Dokument-Ebene reicht in
  Phase 1, weil Seitenzahlen ueber Dokumente hinweg sowieso aufgeloest sind.
* **Originaltext-Extraktion (Kauderwelsch)** ist ein separates Thema und nicht
  Teil dieses Branches.

## Migrations- und Rollback-Hinweise

* `init_db()` ist idempotent; Tabelle und Spalte sind via `IF NOT EXISTS`/
  `try/except` abgesichert.
* Backfill laeuft nur fuer Projekte ohne `documents`-Eintrag — laesst sich
  also gefahrlos mehrfach starten.
* Rollback: Die neuen Strukturen sind additiv. Ein altes Backend wuerde die
  Spalten/Tabellen ignorieren; allerdings funktioniert dann das Multi-PDF
  natuerlich nicht mehr. Ein echtes Schema-Down brauchen wir nicht.

## Pruefliste fuer Verifikation

* [x] Zwei PDFs ins selbe Projekt laden -> zwei Dokument-Gruppen sichtbar.
* [x] Seitenzaehlung in jedem Dokument startet wieder bei 1.
* [x] „Alle Alt-Texte generieren" nach Append erzeugt nur die neuen Bilder
      und ueberschreibt keine fertigen Alt-Texte.
* [x] Einzel-Export liefert eine Datei, „Alle exportieren" liefert eine ZIP
      mit `<doc_index>_<name>.<endung>` pro Dokument.
* [x] Single-PDF-Projekte ohne Migration bleiben funktional unveraendert.
* [x] axe-core: keine neuen WCAG-AA-Verstoesse.
