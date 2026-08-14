# Multi-Datei: Dokument-, Webseiten- und Bild-Ebene

Stand: 14.08.2026 (Phase 2). Autor: Claude (InkluTec), Vorgaben Steve/Michael.

## Idee

Ein Projekt fasst mehrere Quellen. Zwischen Projekt und Bild liegt die
Dokument-Ebene (Tabelle `documents`), damit Quellen einzeln benannt,
geloescht und exportiert werden koennen. Was ein "Dokument" ist, haengt
vom Werkzeug ab:

- **PDF** (Phase 1, 06/2026): jede hochgeladene PDF = ein Dokument.
  Struktur in der App: Dokument (h2, klappbar) -> Seite (h3, klappbar,
  mit Seitenansicht + Seitentext) -> Bild (h4).
- **Web** (Phase 2, 08/2026): jede gescannte Adresse = ein Dokument
  ("Webseite N: Seitentitel"). Struktur: Webseite (h2, klappbar) ->
  Link "Webseite im neuen Tab oeffnen" + Seitentext-Klappe -> Bild (h3).
  KEINE Seiten-Ebene; die technische Spalte `page_number` bleibt bei
  Web-Bildern immer 1 und taucht in der Oberflaeche nirgends auf.
- **Grafik** (Phase 2): KEINE documents-Zeilen — das Einzelbild ist die
  atomare Einheit. Jedes Bild ist ein eigener klappbarer Block
  "Bild N: dateiname.jpg" (h2) mit Umbenennen/Loeschen an derselben
  Stelle wie bei Dokumenten. Anzeige-Nummern sind Positionen 1..N und
  ruecken nach Loeschungen nach (wie bei PDF; der Chat-Assistent zaehlt
  in `inkluagent/tools/project.py` genauso positionsbasiert).

## Datenmodell (Phase-2-Zugaenge, alle Migrationen additiv + idempotent)

- `documents.source_url` — Adresse der gescannten Webseite (PDF: leer).
- `documents.page_text` — Textinhalt der Webseite (Deckel 15.000 Zeichen,
  `main._web_page_text`). Bei PDFs haengt der Seitentext wie gehabt an
  `images.page_text`.
- `images.original_filename` — Original-Dateiname beim Grafik-Upload
  (Altbestand leer -> Anzeige faellt auf "Bild N" zurueck).
- `images.display_name` — vom Nutzer vergebener Anzeigename pro Bild
  (NULL = Rueckfall auf original_filename).
- Backfill: bestehende Web-Projekte bekamen EIN Sammel-Dokument
  "Webseite 1" mit der ersten Projekt-Adresse (pro Einzel-URL war der
  Altbestand nicht mehr trennbar, die Quell-URL wurde pro Bild nie
  gespeichert). Neue Scans legen pro Adresse ein eigenes Dokument an.

## Endpunkte

Bestehend (Phase 1, gelten unveraendert auch fuer Webseiten-Dokumente):
- `PATCH  /api/projects/{pid}/documents/{did}` — umbenennen
- `DELETE /api/projects/{pid}/documents/{did}` — loeschen (raeumt seit
  Phase 2 auch die einzelnen Bilddateien ab, weil Web-Dokumente keinen
  doc<N>-Ordner haben; ausserdem Pruefstatus + Nachrichten der Bilder)

Neu (Phase 2, NUR Grafik/Web — PDF liefert 400, Michael-Vorgabe
09.06.2026: bei PDFs ist das Dokument die kleinste Einheit):
- `PATCH  /api/projects/{pid}/images/{iid}` — Bild umbenennen
- `DELETE /api/projects/{pid}/images/{iid}` — Bild loeschen; verschwindet
  dabei das letzte Bild einer Webseite, wird deren documents-Zeile mit
  entfernt (sonst bliebe ein leerer Block stehen)

Alle vier sind mandantensicher (`WHERE ... AND user_id = ?`, kein IDOR).
Gaeste (Freigabe-Token) haben KEINEN Zugriff — ihre Endpunkte unter
`/api/freigabe/` kennen weder Umbenennen noch Loeschen.

## Frontend (backend/templates/app.html)

- `renderDocBlock` rendert PDF-Dokumente UND Webseiten (Weiche
  `isWebProject`); `renderImgBlock` die Grafik-Einzelbloecke.
- Die zwei nativen `<dialog>`-Elemente (Loeschen/Umbenennen) bedienen
  drei Arten; Titel/Label/Hinweis setzt `_kindTexte` beim Oeffnen
  (`data-kind` am Aktionsknopf: 'doc' | 'web' | 'img').
- Auf/Zu-Zustand: `openDocs` (Dokumente/Webseiten) + `openImgs`
  (Grafik-Bloecke, eigenes Set — getrennte ID-Raeume).
- Export-Auswahl (`renderExportScopeBlock`) spricht bei Web-Projekten
  von "Webseite N" / "Alle Webseiten (eine ZIP-Datei)"; die
  Tabellen-Exporte (JSON/CSV/XLSX) liefern pro Webseite eine Datei
  bzw. ein ZIP — dieselbe Logik wie bei PDF-Dokumenten.

## Tests

- Backend: `verify_multidatei2.py` (48 Pruefungen; laeuft im Container,
  eigener Mini-HTTP-Server fuer reproduzierbare Scans ohne Internet).
- Klicktests: `ui_multidatei2.py` + `setup_ui_multidatei2.py`
  (40 Pruefungen, axe 0 auf allen Ansichten und Dialogen).
- Beide sind in `alle_tests.sh` bzw. `ui_tests.sh` eingehaengt.

## Bewusste Grenzen / spaeter

- Der Chat-Assistent nennt Webseiten in seinen Labels weiterhin
  "Dokument N" (inkluagent `ui_label`) — funktional korrekt, sprachlich
  angleichbar (kleiner Folgeauftrag).
- Keine Screenshot-Vorschau fuer Webseiten (bewusst: Link aufs Original
  statt veralteter Server-Screenshots; nachruestbar, falls gewuenscht).
- Alte `.doc`-/Word-Dateien sind hier NICHT Thema — Word wird ein
  eigenes Werkzeug (siehe Fahrplan neue Dokumentformate, 14.08.2026).
