# Word-Werkzeug: Alt-Texte für Word-Dokumente (.docx)

**Stand:** 26.08.2026, Stufe 1 (Alt-Texte). Autor: Claude (InkluTec), Vorgaben Steve Weidel und Michael Karbe.

## Worum es geht

Kunden laden ein Word-Dokument hoch, InkluDocs erkennt alle Bilder samt
Textkontext, erzeugt Alt-Texte über die bestehende Pipeline, und der Kunde
lädt **seine eigene Datei mit eingetragenen Alt-Texten** wieder herunter.
Nichts sonst an der Datei ändert sich: keine Umformatierung, keine
verschobenen Seiten, Formatvorlagen, Kommentare, Änderungsverfolgung und
Eigenschaften bleiben Byte für Byte erhalten.

Entscheidungen (14.08. und 26.08.2026):

- Eigenes Werkzeug `word` in der Werkzeugliste, **nicht** im PDF-Werkzeug
  (Kunden würden dort einen PDF-Export erwarten).
- **Kein** Word-zu-PDF-Export hier. Das gehört ins Werkzeug „Barrierefreie
  PDFs erstellen" (`pdf-a11y`), mit veraPDF-Prüfung.
- Ein Werkzeug, drei Stufen: (1) Alt-Texte — diese Doku, (2) Prüfbericht
  Barrierefreiheit (regelbasiert, ohne KI-Kosten), (3) Aufbereitung.
- Kein Microsoft Office und kein LibreOffice zum Schreiben. Lesen und
  Schreiben passieren direkt im OOXML-Zip (siehe unten). LibreOffice wäre
  nur für eine spätere Seitenvorschau nötig (Rendern, nie Speichern).

## Wie Word Alt-Texte speichert

Eine `.docx` ist ein Zip mit XML-Teilen. Jedes Bild ist ein `<w:drawing>`
mit `<wp:inline>` (im Textfluss) oder `<wp:anchor>` (frei positioniert).
Darin:

```xml
<wp:docPr id="7" name="Grafik 3" descr="Alternativtext" title="Titel">
  <a:extLst>
    <a:ext uri="{C183D7F6-B498-43B3-948B-1728B52AA6E4}">
      <adec:decorative val="1"/>      <!-- "Dekorativ" (Word 2019+) -->
    </a:ext>
  </a:extLst>
</wp:docPr>
...
<a:blip r:embed="rId5"/>              <!-- Relationship -> word/media/image1.png -->
```

- `descr` = das Feld „Beschreibung" in Word (= Alt-Text).
- `title` = optionaler Titel (ältere Word-Versionen zeigen ihn an).
- `adec:decorative` = Kennzeichen „dekorativ". Word leert dabei `descr`.

## Module

| Datei | Aufgabe |
|---|---|
| `backend/docx_processor.py` | Lesen: Bilder, vorhandener Alt-Text/Titel, Dekorativ-Kennzeichen, Kontext. `validiere_docx()` (Vorprüfung im Upload), `analysiere_docx()` (vollständig), `extract_images_from_docx()` (Schnittstelle wie `extract_images_from_pdf`). |
| `backend/docx_export.py` | Schreiben: `write_alt_texts_to_docx(original, ziel, {anker: text})`. Kopiert alle Zip-Mitglieder byteweise, serialisiert nur die XML-Teile mit geänderten Bildern neu. `pruefe_unveraendert()` als Testhilfe. |
| `backend/tools.py` | Werkzeug `word` (Status Beta). |
| `backend/main.py` | Upload-Gate (`.docx` erlaubt, `.doc`/`.docm`/`.dotx`/`.dotm` mit klarer Meldung abgewiesen), `_handle_pdf_upload(art="docx")`, `_extract_document(art="docx")`, `POST /api/projects/{id}/export/docx`. |
| `backend/database.py` | Migration `images.docx_anker TEXT`. |
| `backend/billing.py` | Aktion `docx_export` (5 Credits pro Vorgang, nur Bezahl-Konten — gleiche Regel wie `pdf_export`). |
| `backend/templates/app.html` | Upload-Block für Word, Etiketten „Abschnitt" statt „Seite", Export-Knopf „Als Word (Beta)". |
| `tests/test_docx_roundtrip.py` | 22 Unit-Tests (Lesen, Schreiben, Byte-Identität, Idempotenz, Abwehr). Fixture `tests/fixtures/testdokument_inkludocs.docx` (fiktiv), erzeugt von `tests/fixtures/make_testdoc.py` (braucht python-docx, nur Entwicklung). |

Bewusst **keine** Laufzeit-Abhängigkeit von python-docx: Die Bibliothek
sieht Kopf-/Fußzeilen und frei positionierte Bilder nur über Umwege, und
das byte-identische Zurückschreiben ist mit eigenem Zip-Code sicherer.
Einzige neue Abhängigkeit: `lxml` (XML-Parser ohne Entity-Auflösung).

## Datenfluss

1. **Upload** `POST /api/upload` mit `.docx` und `project_id` eines
   `word`-Projekts. Vorprüfung `validiere_docx()` **im Request**: gültiges
   Zip, Grenzen, echtes DOCX ohne Makros — Fehler kommen als 400 mit
   verständlicher Meldung zurück, bevor Datenbankzeilen entstehen.
2. `documents`-Zeile mit `extraction_method = "docx"`; Extraktion asynchron
   wie bei PDF (`_extract_document`, Executor-Thread).
3. Je Bildvorkommen eine `images`-Zeile: `page_number` = **Abschnitt**
   (laufende Nummer der Überschrift 1 vor dem Bild; Word hat keine Seiten),
   `context_text`, `original_alt` (vorhandener Alt-Text; `"dekorativ"` bei
   gesetztem Kennzeichen), `docx_anker` = `"<part>|<docPr-id>"`.
   Dasselbe Medium mehrfach im Dokument = mehrere Zeilen, eine Bilddatei.
4. **Generierung, Editor, Chatbot, Gast-Review, JSON/CSV/XLSX-Export:**
   unverändert — die Zeilen sehen aus wie PDF-Bilder.
5. **Export** `POST /api/projects/{id}/export/docx` (Body wie PDF-Export:
   optional `document_id`, `filename`): ein Dokument → Download der
   `.docx`, mehrere → ZIP `…_alle_word.zip`. Alt-Text-Auswahl über
   `_exportable_alt_text()`: leer = Bild unberührt lassen (auch ein alter
   Alt-Text bleibt), `"dekorativ"` = Kennzeichen setzen und `descr` leeren.
   Antwort-Header `X-Export-Tagged`, `X-Export-Total`, `X-Export-Warnings`.

## Kontext: was die KI bekommt

Word liefert, was bei PDF geraten werden muss. Pro Bild (Deckel 1.500 Zeichen):

```
Dokument: <dc:title>
Position: Kopfzeile | Fußzeile            (nur außerhalb des Textes)
Abschnitt: Überschrift 1 > Überschrift 2 > …
Tabellenzeile: … | …   /  Tabellenkopf: … | …   (nur in Tabellen)
Bildunterschrift: Abbildung 3: …           (Formatvorlage Caption/Beschriftung
                                            oder Muster "Abbildung N")
Text im Absatz des Bildes: …
Absatz davor: …
Absatz danach: …
Verweis im Text: "… wie Abbildung 3 zeigt …"   (bis zu drei Sätze)
```

Überschriften werden über die Formatvorlagen-Id (`Heading1…9`, auch
`berschrift1`, `Title`) **und** den Anzeigenamen erkannt; damit
funktionieren deutsche, englische, französische, spanische, dänische und
schwedische Word-Installationen.

## Sicherheit

- XML nur über `lxml.XMLParser(resolve_entities=False, no_network=True,
  huge_tree=False)` — externe Entities (XXE) bleiben unaufgelöst (Test).
- Zip-Grenzen: max. 5.000 Mitglieder, 400 MB entpackt, Kompressionsrate
  1:300 bei Mitgliedern > 1 MB (Zip-Bombe), 60 MB je XML-Teil.
- Zip-Eintragsnamen mit `..`, absolutem Pfad oder Laufwerksbuchstaben
  werden abgewiesen. Es wird **nichts** aus dem Zip auf die Platte entpackt;
  Bilder werden im Speicher mit Pillow geprüft und unter eigenem Namen
  (`p<projekt>_<nr>.png|jpg`) gespeichert.
- Makro-Dateien: Endung (`.docm`, `.dotm`, `.dotx`) **und** Inhaltstyp
  (`macroEnabled`) werden geprüft.
- Export liest die Originaldatei nur aus `UPLOAD_DIR` (Realpath-Prüfung)
  und schreibt atomar (Temp-Datei + `os.replace`) nach `RESULTS_DIR/…/_export`.
- Alle Endpunkte mandantensicher (`user_id`), Gäste haben keinen Export.
- Das Tageslimit für Bilder greift wie bei PDF erst bei der Generierung.

## Grenzen (Stufe 1) und was folgt

- **Vektorgrafiken** (EMF/WMF/SVG), **Diagramme**, **SmartArt**, **Formen**
  ohne Rasterbild: werden erkannt, aber übersprungen (`uebersprungen` mit
  Grund). Stufe 2: Rendern über LibreOffice bzw. Diagramme aus den XML-Daten
  beschreiben (Alleinstellungsmerkmal — echte Zahlen statt Bildschätzung).
- Bilder in **Textfeldern** werden gefunden (`ort = "Textfeld"`), VML-Alt-
  Bilder (`<v:imagedata>`, Word ≤ 2007) noch nicht.
- **Keine Seitenvorschau**: Word kennt keine Seiten. Gruppierung nach
  Abschnitt (Überschrift 1). Vorschau = LibreOffice-Rendern (Entscheidung
  offen, ca. 400 MB Image).
- **Langbeschreibung**: Word hat nur `descr`/`title`; die Pipeline
  überspringt sie wie bei PDF (`skip_langbeschreibung`).
- Der Titel (`title`) wird beim Export nicht verändert (nur gesetzt, wenn
  ein Aufrufer explizit Titel übergibt — der Endpunkt tut das nicht).
- PowerPoint (`python-pptx`-freies Gegenstück, gleiche Anker-Idee) ist der
  nächste Ausbau; etwa 70 % des Codes sind wiederverwendbar.

## Tests

```
# Unit (Container, 22 Tests):
docker cp tests/test_docx_roundtrip.py inkludocs-staging:/app/tests/
docker cp tests/fixtures/testdokument_inkludocs.docx inkludocs-staging:/app/tests/fixtures/
docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_docx_roundtrip.py -v

# End-to-End gegen Staging (Login, Projekt, Upload, Extraktion, Bearbeiten,
# Export, Rück-Lesen, Negativfälle): /home/claude/verify_docx.py
# Klicktest mit Screenshots: /home/claude/ui_word.py
```

Regenerieren des Fixtures (Entwicklung, braucht python-docx):
`python3 tests/fixtures/make_testdoc.py`.
