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
| `backend/billing.py` | Aktion `docx_export` (25 Credits + 5 je angefangene 10 Bilder, alle Konten — gleiche Regel wie `pdf_export`; Aktionspreise 29.08.2026, siehe docs/GENERIERUNG.md). `pdfua_export` gleicher Preis. |
| `backend/pdfua_export.py` | Barrierefreie PDF aus Word (29.08.2026): Umwandler-Client, Klartext aus dem veraPDF-Bericht, Titel/Sprache in core.xml. |
| `konverter/` | Eigener Container: LibreOffice Writer (PDF/UA-Filter) + veraPDF (PDF/UA-1), `POST /pdfua`, `GET /health`. Compose-Dienst `inkludocs-konverter`, App erreicht ihn über `KONVERTER_URL`. |

### Prüfer-Version (30.08.2026)

veraPDF kommt per `COPY --from=` aus dem offiziellen Image und ist auf einen
**Digest festgenagelt**: `sha256:5ec181f5…` = veraPDF **1.31.163**, gebaut am
26.08.2026 — genau die Version, mit der die Testdokumente aus Projekt 317
PDF/UA-1 bestanden haben. Vorher stand dort `:latest`.

Warum: Das veraPDF-Urteil ist die Qualitätsaussage, die wir dem Kunden geben
(„besteht PDF/UA-1"). Mit `:latest` hätte ein späterer Neubau still eine andere
Prüfer-Version gezogen, und dasselbe, **unveränderte** Dokument hätte anders
beurteilt werden können, ohne dass es irgendwo steht.

`GET /health` nennt die Version jetzt mit:
`{"ok": true, "soffice": "LibreOffice 7.4.7.2 …", "verapdf": true,
"verapdf_version": "1.31.163"}` — bei einem strittigen Urteil ist damit sofort
sichtbar, wer es gefällt hat.

**Hochziehen ist Handarbeit und ausdrücklich gewollt:** neuen Digest eintragen,
Image bauen, die drei Testdokumente aus Projekt 317 durchschicken und die
Urteile vergleichen. Zwei Dinge dabei wissen: Ein Digest ohne Tag kann auf der
Registry aufgeräumt werden — dann schlägt der Neubau fehl (laut und sichtbar,
kein stiller Fehler). Und auf dem Server liegt zusätzlich ein eigenständiges
veraPDF 1.30.2 (`/opt/verapdf`, `/srv/inklutec/werkzeug/verapdf`) als
Handwerkszeug — **maßgeblich ist allein das im Umwandler** (Steve 30.08.2026).
LibreOffice ist 7.4.7.2 aus `debian:bookworm-slim`; ein neuerer Unterbau ist
eine offene Frage, keine Zusage.
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

## Barrierefreie PDF aus Word (29.08.2026, Michael/Steve)

Im Export-Dialog des Word-Werkzeugs gibt es den dritten Knopf „In barrierefreie
PDF umwandeln (Beta)“. Ablauf: Word-Datei mit Alt-Texten bauen (wie „Als Word“),
Titel und Sprache in `docProps/core.xml` nachtragen, falls leer
(`pdfua_export.dokumenttitel_setzen`), dann `POST /pdfua` am Umwandler
(`konverter/app.py`: LibreOffice headless mit `UseTaggedPDF` + `PDFUACompliance`
+ Lesezeichen, eigenes Profil je Lauf, höchstens zwei Läufe parallel,
Zeitlimit), danach veraPDF gegen PDF/UA-1. Das Ergebnis kommt als JSON
(`POST /api/projects/{id}/export/pdfua`): `zusammenfassung`, je Dokument
`pruefung.punkte` mit Bereich (Struktur und Lesereihenfolge, Text und Sprache,
Bilder und Grafiken, Überschriften, Tabellen …), Status ok/befund und einem
Satz in Alltagssprache (`pdfua_export.klartext`, Regeln nach ISO 14289-1
Klausel 7.x zusammengefasst, die häufigsten Einzelregeln übersetzt). Die
Datei liegt unter `results/<user>/<projekt>/_export/pdfua_<token>.pdf` (ZIP bei
mehreren Dokumenten) und wird über `GET …/export/pdfua/{token}` geladen
(Token 24 Hex, nur der Besitzer). Preis `export_preis(anzahl, "pdfua")` =
25 + 5 je angefangene 10 Bilder, verbucht nach gelungener Umwandlung.
Ohne `KONVERTER_URL` antwortet der Endpunkt 503 (Prod bekommt den Dienst mit
dem Rollout).

Stufe 2 (29.08.2026 abends): (1) `pdfua_export.alt_nachtragen` trägt mit
pikepdf fehlende `/Alt` an Figure-Elementen nach — LibreOffice verliert die
Alt-Texte bei VML-Bildern und Bildern in Textfeldern; zugeordnet wird nur,
wenn die Zahl der Figures genau der Zahl der Bilder im Textkörper entspricht,
danach prüft `POST /pruefe` am Umwandler die fertige Datei erneut.
(2) `docx_hoerprobe.analysiere` liefert die Hörprobe (was ein Screenreader
liest: Titel, Sprache, Überschriften mit Ebene, Absätze, Listenpunkte, Bilder
mit Alt-Text/Schmuckbild, Tabellen mit Kopfzeile) und den Prüfbericht des
Word-Dokuments (Titel, Sprache, Überschriften-Hierarchie, Tabellen ohne
Kopfzeile, Bilder ohne Alt-Text) — kostenlos über
`POST /api/projects/{id}/export/pdfua/vorschau`, Knopf „Hörprobe und
Prüfbericht“, und zusätzlich im Umwandlungsergebnis. (3) Klartext, Hörprobe und
Prüfbericht laufen durch gettext in der Sprache des Nutzers (6 Sprachen).
Grenzen: Layout nicht pixelgleich zu Word; Schmuckbilder werden noch nicht als
Artefakt markiert (7.1-3 kann melden, `nachbearbeitung.dekorativ_offen`);
Seitenvorschau für Sehende folgt. Tests: `tests/test_pdfua_klartext.py`,
`tests/test_docx_hoerprobe.py`, `tests/e2e/verify_pdfua.py` (Staging).

## Vom Autor als dekorativ gekennzeichnet (01.09.2026)

Word-Autoren können ein Bild als dekorativ kennzeichnen (`adec:decorative`).
Bis zum 01.09. wurde das Kennzeichen als `original_alt = "dekorativ"`
mitgeführt, das Bild lief aber trotzdem durch die KI, bekam einen Text (5
Credits), und der Text gewann über das Kennzeichen — Steves Hörtest fand es an
der Zierlinie der Fixture („Dekoratives Gestaltungselement …“).

Seit 01.09. (Michael Karbe: „ob ein Bild dekorativ ist, wird beim Tagging vor
dem Upload entschieden“; Steve: „der Autor geht vor, aber sichtbar und mit
einem Handgriff umstoßbar“):

- Der Import legt das Bild sofort mit `status = done`, `image_type =
  dekorativ`, leerem Text an (`main.py`, docx-Import). Es läuft nicht durch
  die KI und kostet nichts.
- Die Karte trägt das Badge „Vom Autor in der Datei als dekorativ
  gekennzeichnet.“ (`autorDekorativ()` in `app.html`).
- Sammelläufe lassen es aus (`_generier_kandidaten`: `NOT (original_alt =
  'dekorativ' AND image_type = 'dekorativ')`; Oberfläche gleich). Die
  Rückfrage sagt „1 Bild ist vom Autor als dekorativ gekennzeichnet und bleibt
  außen vor.“
- Hält der Kunde die Entscheidung für falsch: „Neu generieren“ am Bild —
  dann beschreibt die KI es (5 Credits), der Bildtyp wird neu gesetzt, und ab
  da ist es ein normales Bild.
- Export: Das Bild wird übersprungen, das Kennzeichen aus der Datei bleibt
  erhalten (wie bisher). Die Zusammenfassung zählt es unter „als dekorativ
  erkannt“.

PDF-Vergleich: PDFix filtert Artefakte beim Einlesen komplett heraus — dort
gewinnt der Autor unsichtbar. Word ist damit die einzige Stelle, an der ein
Autor-Irrtum sichtbar und korrigierbar ist.

## Dokumente ohne Bilder bleiben (01.09.2026, Befund aus Steves Hörtest)

Ein Word-Dokument, das nur ein Diagramm, SmartArt oder ein OLE-Objekt enthält,
liefert 0 Bilder und Hinweise — es ist ein gültiges Dokument. Bis zum
01.09.2026 löschte die Start-Migration „Phantom-Dokument-Cleanup" (08.06.2026,
`database.py`) bei JEDEM Container-Start alle Dokumente ohne Bilder und
nummerierte die übrigen neu. Folge: Steves Diagramm-Dokument verschwand beim
Staging-Neustart, und `doc_index` passte danach nicht mehr zum Bilderordner
(`results/<user>/<projekt>/doc<N>`; auf Staging vier Dokumente, auf Produktion
eines). Die Migration ist stillgelegt (sie hatte ihre Arbeit am 08.06. getan),
die Ordnernummern wurden von Hand repariert, und
`tests/test_phantom_cleanup.py` hält die Migration stumm (rot gegen den alten
Code). Dokumente werden nur noch auf Wunsch des Nutzers gelöscht.

## Bekannte Lücken der barrierefreien PDF (Stand 01.09.2026, aus verify_pdfua)

Mit vollständig beschrifteten Bildern meldet veraPDF für die Testdatei noch zwei
Dinge, die NICHT am Alt-Text-Werkzeug liegen und im Prüfbericht an den Nutzer
ehrlich erscheinen:
1. **Vom Autor als dekorativ gekennzeichnete Bilder** werden in der PDF noch
   nicht zu Artefakten — LibreOffice ignoriert das Word-Kennzeichen, die Figure
   bleibt ohne Alt (`dekorativ_offen` in `nachbearbeitung`). Stufe 3: Figure →
   Artefakt umschreiben (pikepdf, StructTree + Marked Content).
2. **Inhalte ohne Struktur-Markierung** (Bereich „Struktur und Lesereihenfolge“,
   z. B. Rahmen/Linien) stammen aus LibreOffice 7.4 im Konverter; neuere
   LibreOffice-Versionen taggen sauberer → Konverter-Image prüfen.
`tests/e2e/verify_pdfua.py` beschriftet deshalb erst alle Bilder und erwartet
dann: kein Alt-Text-Befund für beschriebene Bilder, keine Befunde außerhalb
dieser beiden Lücken; „vollständig bestanden“ wird nur als Info ausgegeben.
