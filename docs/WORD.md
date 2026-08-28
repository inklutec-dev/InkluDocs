# Word-Werkzeug: Alt-Texte für Word-Dokumente (.docx)

**Stand:** 27.08.2026, Stufe 1 (Alt-Texte) nach Härtetest mit echten Word-Dateien und Seitenerkennung. Autor: Claude (InkluTec), Vorgaben Steve Weidel und Michael Karbe.

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

**Altformat VML** (Dokumente aus dem Kompatibilitätsmodus, aus `.doc`
gewandelt, alte Firmenvorlagen — in der Praxis häufig):

```xml
<w:pict>
  <v:shape id="_x0000_i1025" alt="Alternativtext" title="Titel" style="…">
    <v:imagedata r:id="rId5"/>    <!-- Relationship -> word/media/image1.jpeg -->
  </v:shape>
</w:pict>
```

Alt-Text = Attribut `alt` des Shapes, Titel = `title`. Ein Dekorativ-
Kennzeichen kennt VML nicht; „dekorativ" wird dort als leerer `alt`
geschrieben. Anker: `"<part>|v:<shape-id>"`.

**Verschachtelung und Duplikate** (Befunde des Härtetests 27.08.2026):

- Bild **in einem Textfeld**: Word schreibt Textfeld-Drawing → `txbxContent`
  → Bild-Drawing. Nur der `a:blip` des Bildes selbst zählt; das Textfeld ist
  kein Bild (früher wurde es mitgezählt und der Alt-Text wäre am Textfeld
  gelandet).
- Word schreibt Textfelder **doppelt**: `mc:Choice` (modern) und `mc:Fallback`
  (VML für alte Word-Versionen) mit **denselben `docPr`-ids**. Gelesen wird
  nur `mc:Choice`; beim Export bekommen beide Hälften den Alt-Text.
- Doppelte `docPr`-ids außerhalb von Fallback (kaputte Dokumente nach
  Kopieren/Einfügen): Anker `"<part>|<id>#<n>"` ab dem zweiten Vorkommen.

## Module

| Datei | Aufgabe |
|---|---|
| `backend/docx_processor.py` | Lesen: Bilder, vorhandener Alt-Text/Titel, Dekorativ-Kennzeichen, Kontext. `validiere_docx()` (Vorprüfung im Upload), `analysiere_docx()` (vollständig), `extract_images_from_docx()` (Schnittstelle wie `extract_images_from_pdf`). |
| `backend/docx_export.py` | Schreiben: `write_alt_texts_to_docx(original, ziel, {anker: text})`. Kopiert alle Zip-Mitglieder byteweise, serialisiert nur die XML-Teile mit geänderten Bildern neu. `pruefe_unveraendert()` als Testhilfe. |
| `backend/tools.py` | Werkzeug `word` (Status Beta). |
| `backend/main.py` | Upload-Gate (`.docx` erlaubt, `.doc`/`.docm`/`.dotx`/`.dotm` mit klarer Meldung abgewiesen), `_handle_pdf_upload(art="docx")`, `_extract_document(art="docx")`, `POST /api/projects/{id}/export/docx`. |
| `backend/database.py` | Migrationen `images.docx_anker TEXT`, `documents.hinweise TEXT` (JSON, 27.08.). |
| `backend/billing.py` | Aktion `docx_export` (5 Credits pro Vorgang, nur Bezahl-Konten — gleiche Regel wie `pdf_export`). |
| `backend/templates/app.html` | Upload-Block für Word, Etiketten „Abschnitt" statt „Seite", Export-Knopf „Als Word (Beta)". |
| `tests/test_docx_roundtrip.py` | 35 Unit-Tests (Lesen, Schreiben, Byte-Identität, Idempotenz, Abwehr, echte Word-Fälle). Fixture `tests/fixtures/testdokument_inkludocs.docx` (fiktiv), erzeugt von `tests/fixtures/make_testdoc.py` (braucht python-docx, nur Entwicklung); dazu vier von Microsoft Word erzeugte Dateien aus dem LibreOffice-Testkorpus (`word_textfeld_bild`, `word_vml_bild`, `word_vml_kopfzeile`, `word_einfach`, `word_diagramm`, `word_smartart`, `word_excel_objekt`; MPL-2.0). |

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
3. Je Bildvorkommen eine `images`-Zeile: `page_number` = **Seite wie zuletzt
   in Word gezeigt**, wenn das Dokument Seitenmarken trägt (siehe „Seiten"),
   sonst **Abschnitt** (laufende Nummer der Überschrift 1 vor dem Bild);
   `documents.extraction_method` = `"docx-seiten"` bzw. `"docx"`, daran
   erkennt die Oberfläche die Beschriftung „Seite N" bzw. „Abschnitt N";
   `context_text`, `original_alt` (vorhandener Alt-Text; `"dekorativ"` bei
   gesetztem Kennzeichen), `docx_anker` = `"<part>|<docPr-id>"` (VML:
   `"<part>|v:<shape-id>"`).
   Dasselbe Medium mehrfach im Dokument = mehrere Zeilen, eine Bilddatei.
4. **Generierung, Editor, Chatbot, Gast-Review, JSON/CSV-Export:**
   unverändert — die Zeilen sehen aus wie PDF-Bilder.
5. **Export** `POST /api/projects/{id}/export/docx` (Body wie PDF-Export:
   optional `document_id`, `filename`): ein Dokument → Download der
   `.docx`, mehrere → ZIP `…_alle_word.zip`. Alt-Text-Auswahl über
   `_exportable_alt_text()`: leer = Bild unberührt lassen (auch ein alter
   Alt-Text bleibt), `"dekorativ"` = Kennzeichen setzen und `descr` leeren.
   Antwort-Header `X-Export-Tagged`, `X-Export-Total`, `X-Export-Warnings`.

## Seiten (27.08.2026, Steve: „wenn möglich dieselben Seiten wie im Dokument")

Word schreibt beim Sichern `<w:lastRenderedPageBreak/>` an jede Stelle, an
der beim letzten Anzeigen eine neue Seite begann. `_seiten_der_bilder()`
zählt diese Marken in Dokumentreihenfolge und ordnet jedem Bild die Seite zu,
die Word zuletzt gezeigt hat — ohne Rendern, ohne LibreOffice. Regeln:

- Gibt es Word-Marken, zählen **nur** sie (Word setzt sie auch nach manuellen
  Umbrüchen, sonst würde doppelt gezählt).
- Sonst zählen manuelle Umbrüche: `<w:br w:type="page"/>`, `w:pageBreakBefore`,
  Abschnittswechsel (`w:sectPr` in `w:pPr`, außer `continuous`; wirkt ab dem
  nächsten Absatz).
- Gibt es gar keine Marken (Datei nicht von Word gesichert, z. B. aus
  python-docx oder Google Docs), bleibt die Einheit **Abschnitt** — die
  Oberfläche sagt dann „Abschnitt N" statt „Seite N". `docProps/app.xml`
  (`<Pages>`) wird bewusst nicht genutzt: fremde Erzeuger schreiben dort
  Platzhalter.
- Kopf-/Fußzeilenbilder bekommen Seite 1.

Die Seite ist so genau wie Words letzte Anzeige: Wer die Datei in einem
anderen Programm mit anderen Schriften öffnet, kann andere Umbrüche sehen.

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
- Bilder in **Textfeldern** (`ort = "Textfeld"`), in Kopf-/Fußzeilen und
  **VML-Bilder** des Altformats werden gefunden und zurückgeschrieben.
  Eingebettete OLE-Objekte (`w:object`, z. B. Excel-Diagramm als EMF) und
  Bilder mit externem Link (nicht im Dokument) werden übersprungen.
- Übersprungene Elemente stehen je Dokument in `documents.hinweise` (JSON:
  `uebersprungen` mit `art` diagramm|smartart|textfeld|form|gruppe|vektor|
  extern|ole|unlesbar|bild_ohne_daten, `name`, `format`, `ort`, `seite`,
  `abschnitt`; `warnungen`; `seiten`). Die Oberfläche zeigt sie als Klappe
  „N Elemente ohne Alt-Text (in dieser Ausbaustufe nicht unterstützt)" über
  den Bildern des Dokuments, mit Art, Name, Ort und Seite/Abschnitt — in
  allen sechs Sprachen. Textfelder, Formen und Gruppen ohne eigenes Bild
  werden dort ebenfalls genannt, damit kein Element „fehlt".
- **Keine Seitenvorschau** (Bild der Seite): Vorschau = LibreOffice-Rendern
  (Entscheidung offen, ca. 400 MB Image). Die Seitennummer selbst kommt aus
  den Word-Marken (siehe „Seiten").
- **Langbeschreibung**: Word hat nur `descr`/`title`; die Pipeline
  überspringt sie wie bei PDF (`skip_langbeschreibung`).
- Der Titel (`title`) wird beim Export nicht verändert (nur gesetzt, wenn
  ein Aufrufer explizit Titel übergibt — der Endpunkt tut das nicht).
- PowerPoint (`python-pptx`-freies Gegenstück, gleiche Anker-Idee) ist der
  nächste Ausbau; etwa 70 % des Codes sind wiederverwendbar.

## Tests

```
# Unit (Container, 35 Tests):
docker cp tests/test_docx_roundtrip.py inkludocs-staging:/app/tests/
docker cp tests/fixtures/<jede .docx> inkludocs-staging:/app/tests/fixtures/
docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_docx_roundtrip.py -v

# Alles zusammen (Unit + E2E + Klicktest): bash /home/claude/word_tests.sh
# End-to-End gegen Staging (Login, Projekt, Upload, Extraktion, Bearbeiten,
# Export, Rück-Lesen, Negativfälle): /home/claude/verify_docx.py
# Klicktest mit Screenshots: /home/claude/ui_word.py
# Härtetest mit 29 von Word erzeugten Dateien (Lesen → Schreiben → Rück-Lesen,
# Zip-Test, XML wohlgeformt): /home/claude/corpus_test.py (Korpus /home/claude/corpus)
# Praxislauf über die API mit echten Dokumenten inkl. Generierung und ZIP-Export:
# /home/claude/praxislauf.py
# Hinweise (Diagramm/SmartArt/Excel/WMF/Textfeld) über API + Klick-Check: /home/claude/hinweise_e2e.py
# Export-Integrität (Original vs. Export identisch bis auf Alt-Text): /home/claude/text_erhalt.py
```

## Härtetest 27.08.2026

29 von Microsoft Word erzeugte Dateien (LibreOffice-Testkorpus + Karbes
Dokumentation mit drei Screenshots): Bilder inline, frei positioniert, in
Kopfzeilen, in Textfeldern, in Gruppen, zugeschnitten, mit Effekten, in
Inhaltssteuerelementen; Diagramme, SmartArt, OLE, externe Links, WMF.
Ergebnis nach den Korrekturen: 29/29 — jedes erkannte Bild bekommt seinen
Alt-Text zurück, unberührte Zip-Teile bleiben byte-identisch, alle XML-Teile
wohlgeformt. Die drei behobenen Befunde stehen oben unter „Verschachtelung
und Duplikate" und „Altformat VML".

Offen: Öffnungsprobe der exportierten Dateien in Microsoft Word selbst
(Word auf dem Mac ließ sich nicht per Automation steuern).

Regenerieren des Fixtures (Entwicklung, braucht python-docx):
`python3 tests/fixtures/make_testdoc.py`.
