# Generieren, Neu generieren, Alle neu generieren

Stand 28.08.2026. Gilt gleichlautend für Alt-Texte (PDF, Word, Web, Grafik)
und Quickinfos (Formulare) — Steves Regel: „Bedienung identisch in allen
Werkzeugen“.

## Zwei Fächer je Element

Jedes Bild hat einen KI-Text (`images.alt_text`) und einen Hand-Text
(`images.alt_text_edited`); jedes Formularfeld einen Feldtext
(`formularfelder.quickinfo`, mit `quelle` hand/pdf/stammdaten/gast/ki/chat)
und ein KI-Fach (`formularfelder.quickinfo_ki`). Der Hand-Text gewinnt in
Anzeige und Export; der KI-Text bleibt darunter erhalten und lässt sich mit
„Zurück auf Original“ (Bilder) bzw. „KI-Vorschlag übernehmen“ (Felder) zurückholen.

## Die drei Knöpfe

1. **Alle generieren** (Bilder: „Alle Alt-Texte generieren“) — füllt nur
   Lücken (Bilder mit Status `pending`, Felder ohne Text). Erscheint nur,
   solange Lücken da sind.
2. **Neu generieren** am Element — erzeugt einen neuen KI-Text mit
   Variation (höhere Temperatur, Cache umgangen). Ein Hand-Text bleibt
   obenauf. Bilder: 5 Credits, Felder: 1 Credit (Aktionspreise 29.08.2026).
3. **Derselbe Knopf, sobald keine Lücken mehr da sind** — fasst nur KI-Texte
   ohne Hand-Text an (`POST /api/projects/{id}/generate` bzw.
   `…/quickinfos/generieren` mit `{"modus": "ki_neu"}`). Bilder: 5 Credits je
   Bild, Cache je Bild geräumt (`force_regenerate`); Felder: 1 Credit je Feld.

**Beschriftung seit 31.08./01.09.2026 (Michael Karbe):** Die Sammel-Knöpfe
heißen in beiden Fällen gleich — „Alt-Texte generieren“ (PDF, Word) bzw.
„Quickinfos generieren“ (Formular) — auf Projekt- und auf Dokument-Ebene,
ohne Anzahl und ohne Preis im Knopf. Ein Screenreader unterscheidet sie am
versteckten Zusatz „– ganzes Projekt“ / „– Dokument „x““. Anzahl, Preis und
Guthaben nennt die Rückfrage vor dem Start (nächster Abschnitt).

## Rückfrage vor jedem Sammellauf (31.08./01.09.2026)

Vor jedem Sammellauf öffnet sich EIN Dialog (`#genConfirmDialog` in
`app.html`, Funktion `generierRueckfrage()`), den alle drei Werkzeuge nutzen:

- **Überschrift** = Aktion („Alt-Texte generieren“ / „Quickinfos generieren“).
- **Umfang** („Nur dieses Dokument“ / „Ganzes Projekt, 3 Dokumente“).
- **Kostensatz**: Erstlauf Bilder „{n} Bilder werden beschrieben. Das kostet
  {c} Credits.“ — bewusst die GESAMTZAHL (Michael Karbe 01.09.2026: „lediglich
  die Anzahl der Bilder angeben“). Bringt die Datei schon Alternativtexte mit
  (Word fast immer, PDF manchmal — `images.original_alt`), folgt „{m} davon
  bringen schon einen Text aus der Datei mit; die KI nimmt ihn als Grundlage.“
  Der frühere Satz „haben noch keine Beschreibung“ war an dieser Stelle
  falsch: Die Texte stehen sichtbar im Feld, sie stammen nur nicht von der KI.
  Erneuern: „{n} von der KI erzeugte Texte werden neu erzeugt. Deine eigenen
  Texte bleiben unverändert. …“. Felder analog („{n} Felder ohne Quickinfo …“
  / „{n} KI-Vorschläge …“).
- **Guthaben** (`guthabenSatz()`): unbegrenzt / reicht / „reicht für m von n,
  danach hört der Lauf auf“ (Teil-Lauf, keine Sperre — Steve 31.08.2026) /
  reicht nicht einmal für eins (Knopf gesperrt).
- Umfang UND Kostensatz hängen per `aria-describedby` am Dialog und werden
  beim Öffnen angesagt; Startfokus auf „Abbrechen“; Escape und Abbrechen
  starten nichts. Doppelklick: ein offener Dialog blockt den zweiten Aufruf.

Die Zahlen kommen vom Server, nie aus einer Rechnung im Browser:
`POST /api/projects/{id}/generate/vorschau` bzw. `…/quickinfos/vorschau`
(Body `{"modus": "luecken"|"ki_neu", "document_id"?}`) liefern `anzahl`,
`mit_quelltext` (nur Bilder, Erstlauf), `dokumente`, `preis`, `preis_je`,
`verfuegbar`, `machbar`, `erlaubt`. Beide zählen mit DERSELBEN Funktion wie
der Start (`_generier_kandidaten` in `main.py` bzw. `formular_api.py`) —
nach einem Abbruch nimmt `ki_neu` nur den offenen Rest, das könnte die Seite
nicht wissen. **Antwortet die Vorschau mit einem Fehler (abgelaufene Sitzung,
500), wird NICHT gestartet**, sondern der Fehler angesagt (Prüfbefund
01.09.2026 — vorher lief der Lauf in dem Fall ohne Rückfrage los).

**Bewusst geleert bleibt leer (01.09.2026):** `alt_text_edited = ''` heißt
seit 31.08. „der Nutzer hat den Text absichtlich gelöscht“. So ein Bild ist
KEIN Kandidat für den Sammellauf „neu erzeugen“ (Backend `alt_text_edited IS
NULL`, Oberfläche `alt_text_edited == null` — beide gleich, sonst zeigt die
Seite einen Knopf ohne Lauf dahinter). Vorher zählte es mit, der Lauf schrieb
den neuen KI-Text aber nur nach `alt_text` und ließ das leere Feld stehen:
bezahlt, beschrieben, unsichtbar, nicht exportiert. Wer für ein geleertes Bild
wieder einen Text will, nimmt „Neu generieren“ am Bild (setzt
`alt_text_edited` zurück).

**Word: „Herunterladen (Beta)“ (Michael Karbe 01.09.2026):** Bei
Word-Projekten (`project_type = docx`) tragen der Projekt-Knopf, der
Dokument-Knopf und die Dialog-Überschrift das Beta — die Word-Ausgabe
(docx-Rückschreiben, PDF/UA-Umwandlung) ist noch Beta, die PDF-Ausgabe
nicht. Eine Stelle: `herunterladenName(project)` in `app.html`.

## Was nie angefasst wird

Sammelläufe überschreiben keine Hand-Texte (Bilder: `alt_text_edited`
gesetzt; Felder: `quelle` hand/pdf/stammdaten/gast/chat). Beim Feld-Pass gilt
zusätzlich: Texte, die während des Laufs von Hand gefüllt wurden, bleiben
(`UPDATE … WHERE quickinfo leer` bzw. `quelle = ki`).

## Aktionspreise (Michael Karbe, bestätigt 29.08.2026)

Alle Preise stehen an EINER Stelle: `billing.AKTIONS_PREISE` (je Vorgang) und
`billing.EXPORT_ARTEN` / `EXPORT_SCHRITT` (Export-Staffel). Stand 29.08.2026:

- Alt-Text: 5 Credits je Bild — auf allen Wegen (Sammellauf, Neu generieren,
  Public API, InkluAgent).
- Quickinfo: 1 Credit je Feld (vorher je Seite) — Feld-Pass verbucht
  `aktion_preis("quickinfo_generierung", geschriebene Felder)`.
- InkluAgent: Reden ist kostenlos; ändert er einen Alt-Text 5 Credits, eine
  Quickinfo 1 Credit.
- Datei-Export: `billing.export_preis(anzahl, art)` = 25 Credits Grundpreis +
  5 Credits je angefangene 10 Bilder (PDF, Word, barrierefreie PDF aus Word) bzw. + 1 Credit je angefangene
  10 Felder (Formular-PDF). Beim Alle-Dokumente-ZIP werden die Elemente aller
  Dokumente zusammengezählt, der Grundpreis fällt einmal an. Beispiele: PDF mit
  1 Bild = 30, 26 Bilder = 40, 100 Bilder = 75; Formular mit 1 Feld = 26,
  26 Felder = 28, 50 Felder = 30.
- Tabellen-Export (JSON, CSV, Formular-CSV): 10 Credits je Vorgang, fester
  Preis (`billing.TABELLEN_EXPORTE`). Die Stammdaten-CSV (eigene Bibliothek,
  kein Dokument) bleibt kostenlos. Der Excel-Export wurde am 28.08.2026
  entfernt (CSV öffnet sich in Excel).

Gilt für alle Konten, auch Free (50 Credits): ein 26-Felder-Formular kostet
26 + 28 = 54 Credits — wer größere Formulare braucht, braucht ein Abo oder ein
Paket (Michaels Entscheidung 29.08.2026).

Die Wache vor jeder kostenpflichtigen Aktion ist `billing.aktion_pruefung(user,
aktion, menge)` bzw. `export_pruefung(user, anzahl, art)`: erlaubt nur, wenn das
Guthaben (Monatsrest + Pakete, `verfuegbare_credits`) den vollen Preis deckt.
Reicht es nicht, antworten Export und Generierung mit 402 und
`billing.credits_fehlen_detail` (Code `credits_fehlen`, beide Zahlen); der
InkluAgent antwortet im Chat mit `credits_fehlen_text`. Der Sammellauf prüft je
Bild bzw. je Formularseite (Preis = offene Felder der Seite) und lässt den Rest
offen, wenn das Guthaben nicht mehr reicht.

### Abbruch im Modus „n neu generieren" (30.08.2026)

`ki_neu` schaltet die betroffenen Bilder auf `pending`, damit der vorhandene
Sammellauf sie aufgreift. `pending` heißt seitdem zweierlei: „nie generiert"
und „wartet auf einen neuen Lauf". Damit aus dem zweiten bei einem Abbruch
nicht still das erste wird:

- Der **Start** prüft das Guthaben, **bevor** irgendein Bild umgeschaltet wird,
  und antwortet mit 402 (`billing.aktion_pruefung`). Er steht nach der
  Besitzprüfung des Projekts und feuert nur, wenn es etwas zu tun gibt; das
  Tageslimit greift weiterhin zuerst (429).
- Die Bilder des Laufs gehen **namentlich** an `_process_project`
  (`ki_neu_ids`), statt dort am vorhandenen Alt-Text erkannt zu werden. Das
  Erkennen am Text übersähe dekorative Bilder: die sind zu Recht `done` und
  tragen keinen Text.
- Geht das Guthaben oder das Tageslimit **mitten im Lauf** aus, stellt
  `_ki_neu_zurueck` die noch offenen Bilder wieder auf `done` (nur die
  übergebenen, nur die noch `pending` sind — idempotent).
- Bricht der Lauf **von außen** ab (Ausnahme, oder `CancelledError` beim
  Neustart des Containers), fängt der Mantel `_process_project` das ab und
  ruft `_notaufraeumen`: zurückstellen, frisch zählen, Projekt auf `done`.
  Ohne das bliebe das Projekt auf `processing` stehen und jeder weitere
  Versuch liefe in 409 „Verarbeitung läuft bereits", ohne Ausweg.
- Scheitert ein **einzelnes Bild** im Modus `ki_neu`, bleiben alter Text und
  `done` — der Text ist ja gültig, nur der neue Versuch ist gescheitert. Das
  Bild wird aber `needs_review = 1` gesetzt, und die Zahl der Fehlversuche
  steht im Log. Ohne dieses Signal sähe ein Lauf über 200 Bilder bei einem
  KI-Ausfall wie ein voller Erfolg aus.
- Der Export war von hängengebliebenen Bildern nie betroffen:
  `_display_alt_text` entscheidet am Text, nicht am Status. Falsch waren
  Anzeige und Zähler.
- Offen (bewusst): Ein zurückgestellter Lauf ist von einem vollständigen nicht
  zu unterscheiden. Ein zweiter Lauf generiert alles erneut und verbucht es
  erneut. Das lag schon in der Bauweise von `ki_neu`.

Test: `tests/e2e/verify_ki_neu_abbruch.py` (25 Prüfungen, nur Staging).

Vor dem Export zeigt der Dialog Preis und Guthaben
(`POST /api/projects/{id}/export/preis` bzw. `…/export/summary`, beide liefern
zusätzlich `preis_tabelle` für den CSV/JSON-Satz). Reicht das Guthaben nicht,
antwortet der Export mit 402 und beiden Zahlen; die Oberfläche zeigt die
barrierefreie Meldung „Der Export würde 40 Credits benötigen, du verfügst
derzeit über 37 Credits …“ mit den Knöpfen „Zu Abo & Verbrauch“ und
„Schließen“ (`zeigeCreditsMeldung`, app.html, beide Werkzeuge; dieselbe
Meldung bei 402 aus Generieren/Neu generieren). Jede Export-Antwort — auch
CSV/JSON — trägt `X-Export-Credits`.

## Knöpfe je Dokument (Michael/Steve 28.08.2026)

Neben „Umbenennen“ und „Löschen“ trägt jedes Dokument (PDF, Word, Formular)
zwei weitere Knöpfe: „Alt-Texte generieren“ bzw. „Quickinfos generieren“
(seit 01.09.2026 in beiden Modi gleich benannt — Lücken füllen, solange das
Dokument Lücken hat, sonst nur KI-Texte dieses Dokuments erneuern;
`POST …/generate` bzw. `…/quickinfos/generieren` mit `document_id`; davor
die Rückfrage mit Anzahl und Preis), und „Herunterladen“ (Word:
„Herunterladen (Beta)“), das den Dialog für genau dieses Dokument öffnet
(Einzeldatei, Preis/Guthaben passend). Eine Auswahlliste gibt es im Dialog
nicht mehr (Steve 28.08.2026): der Knopf entscheidet. Der Hauptknopf oben
heißt bei mehreren Dokumenten „Ganzes Projekt herunterladen“ und liefert alle
als ZIP, bei einem Dokument „Herunterladen“; die Dialog-Überschrift nennt das
Ziel („Dokument ‚x.pdf‘ herunterladen“ / „Ganzes Projekt herunterladen (3
Dokumente)“).

**Beschriftung (Michael Karbe / Steve 30.08.2026):** Der Knopf hieß bis dahin
„Exportieren“. Umbenannt in „Herunterladen“ — Michaels Begründung: Die
eigentliche Arbeit, die Alt-Texte, ist vorher schon passiert und steht in der
Oberfläche; der Knopf holt nur noch die Datei. Deutsch ist die Quellsprache,
die msgid selbst hat sich also geändert; alle sechs Kataloge sind mitgezogen
(en/da „Download“, fr „Télécharger“, es „Descargar“, sv „Ladda ner“).
Rein die Beschriftung: Preise, Ablauf und Endpunkte sind unverändert, der
Dialog zeigt weiterhin Preis und Guthaben, bevor etwas passiert. Die
Wortfamilie „Export“ bleibt dort stehen, wo sie den VORGANG benennt
(„Export-Optionen“, „Dieser Export kostet {p} Credits“, „Wird exportiert…“). Die Generier-Knöpfe oben gelten weiterhin für das ganze Projekt. Web-Werkzeug:
ein Dokument = die Seite, daher keine Zusatzknöpfe.


## Abgebrochener „n neu generieren“-Lauf wird fortgesetzt, nicht wiederholt (31.08.2026)

**Vorher:** „n neu generieren“ setzt alle fertigen KI-Bilder auf `pending`, damit der
Sammellauf sie aufgreift. Reichte das Guthaben unterwegs nicht mehr, brach der Lauf ab und
stellte die restlichen Bilder wieder auf `done`. Startete der Nutzer nach dem Aufstocken
erneut, waren wieder ALLE Bilder Kandidaten — auch die gerade frisch generierten. Sie liefen
ein zweites Mal durch die KI und kosteten ein zweites Mal. Seit den Aktionspreisen (5 Credits
je Alt-Text) fällt das ins Gewicht.

**Jetzt:** Endet ein Lauf vorzeitig, merkt sich das Projekt in `projects.ki_neu_rest` die
Bilder, die nicht mehr an der Reihe waren (JSON: `ids` + Zeitstempel `ts`). Der nächste Start
im selben Modus nimmt nur diese.

- Die IDs werden IMMER mit den aktuellen Kandidaten geschnitten. Ein Bild, das inzwischen von
  Hand bearbeitet, gelöscht oder einem anderen Dokument zugeordnet wurde, fällt damit heraus —
  und ein manipulierter Vermerk kann kein fremdes Bild erreichen.
- Läuft ein Lauf vollständig durch, wird der Vermerk gelöscht: der nächste Klick nimmt wieder
  alle Bilder. Gescheiterte Bilder gelten dabei als erledigt — sie haben nichts gekostet.
- Der Vermerk verfällt nach `main.KI_NEU_REST_STUNDEN` (24 h), damit niemand dauerhaft daran
  hängenbleibt.
- Unlesbarer oder veralteter Inhalt führt zum normalen Verhalten (alle Kandidaten), nie zu
  einem Fehler: der Vermerk ist eine Bequemlichkeit, kein Zustand, auf dem etwas aufbaut.

- Mehrere Dokumente im Projekt (31.08.2026, Prüfbefund): Der Vermerk wird fortgeschrieben,
  nicht überschrieben — ein Lauf nimmt nur seine eigenen Bilder heraus und legt seine offenen
  hinein. Ein Lauf in Dokument B lässt den Rest von Dokument A stehen.
- Bewusst offen: Beim Abbruch von außen (Container-Neustart) wird der Rest korrekt vermerkt,
  aber kein Lauf-Hinweis gesetzt — die Ansage läuft nur im laufenden Fortschritts-Poll, und
  nach einem Neustart lädt der Nutzer die Seite ohnehin neu. Ein Hinweis beim Laden der
  Projektansicht wäre ein eigener Umbau.

Beteiligt: `main._ki_neu_rest_lesen`, `main._ki_neu_rest_pflegen`, `main._ki_neu_zurueck`
(schreibt den Vermerk nur, wenn wirklich etwas offen blieb — der geordnete Abschluss ruft sie
ein zweites Mal auf), `main.generate_alt_texts` (liest ihn).
Test: `tests/e2e/verify_doppelkosten.py` (26 Prüfungen, u. a. „vier Bilder, vier Buchungen“).

## Vorzeitiges Ende ist hörbar (31.08.2026)

Ging das Guthaben oder das Tageslimit mitten im Lauf aus, stand der Grund bisher nur im
Server-Log. Die Oberfläche meldete trotzdem „Alle Alt-Texte wurden generiert“ — im Modus
„neu generieren“ sogar mit vollem Zähler, weil die zurückgestellten Bilder wieder als fertig
zählen. Der Nutzer konnte nicht erkennen, dass nur ein Teil neu gemacht wurde.

Jetzt schreibt der Lauf `projects.lauf_hinweis` (JSON: `grund` = `credits` oder `tageslimit`,
`erledigt`, `offen`), `GET /api/projects/{id}/status` reicht ihn als `lauf_hinweis` weiter, und
`app.html` sagt ihn nach dem Lauf in der Live-Region an — in allen sechs Sprachen. Ein neuer
Start räumt den alten Hinweis weg.

## Runde 2 am 01.09.2026 (Steves Hörtest auf Staging)

- **Einzahl:** „1 Bild wird beschrieben“, „Eines davon bringt schon einen Text
  mit“, „1 von der KI erzeugter Text wird neu erzeugt“, „1 Feld ohne
  Quickinfo“, „1 KI-Vorschlag“ — eigene msgids, weil `t()` keine Pluralformen
  kennt. Gehört: „1 davon bringen“, „1 Bilder“.
- **„die KI bezieht ihn ein“** statt „nimmt ihn als Grundlage“: Der
  mitgebrachte Text steht der KI als `ORIGINAL-ALT` im Prompt (alle
  Bausteine), nur bei funktionalen Bildern mit brauchbarem Text wird er ohne
  KI-Aufruf übernommen. Die Fixture `testdokument_inkludocs.docx` trägt
  absichtlich einen FALSCHEN Bestandstext („roter Kreis“ bei einer roten
  Fläche) — die KI hat ihn zu Recht nicht übernommen.
- **Herunterladen-Zusammenfassung mit drei Gründen** (`/export/summary`):
  `dekorativ` zählt jetzt auch Bildtyp `dekorativ` ohne Text (KI oder Autor —
  auf Produktion 986 Bilder, die vorher als „noch nicht generiert“ galten),
  `geleert` = bewusst geleert (`alt_text_edited = ''`), `offen` = nie
  generiert; `uebersprungen = fehler + offen + geleert`. Sätze: „… 2 als
  dekorativ erkannt“, „1 bewusst ohne Beschreibung“, Einzahl „1 wird
  übersprungen“.
- **Vom Autor als dekorativ gekennzeichnet (Word):** siehe `docs/WORD.md`.
  Rückfrage nennt sie („1 Bild ist vom Autor als dekorativ gekennzeichnet und
  bleibt außen vor“, Feld `autor_dekorativ` der Vorschau), Sammelläufe lassen
  sie aus (Backend `_generier_kandidaten`, Oberfläche `autorDekorativ()`).
- **Export-Sperre + Meldung nach dem Herunterladen** (alle Werkzeuge): Während
  des Exports sind alle Knöpfe im Dialog gesperrt, der gedrückte heißt „Wird
  exportiert...“ (Steve hatte doppelt geklickt: zweimal 30 Credits). Danach
  bleibt der Dialog offen, die Statuszeile sagt „Heruntergeladen: „Datei“
  (30 Credits abgebucht).“ (Preis aus `X-Export-Credits`), bekommt den Fokus,
  und „Abbrechen“ heißt „Zurück zum Projekt“; Zusatz „Du findest die Datei bei
  deinen Downloads.“ (die Seite kennt den Speicherort nicht — Browser-Grenze;
  Statuszeile ist ein natives `<output>`). Ein zweiter Klick währenddessen
  sagt „Der Export läuft bereits.“
- **Dateiname-Feld ohne Browser-Ausfüllhilfe** (`autocomplete="off"`): Steves
  ZIP hieß nach einem alten Projekt, weil der Browser einen Verlaufseintrag
  ins optionale Feld gesetzt hatte.
- **Abo-Seite:** Bei gekündigtem Online-Abo steht jetzt beim Knopf „Kündigung
  zurücknehmen“, warum „Plan wechseln“ fehlt (Review-Befund 8 blieb sonst
  unsichtbar).

## PDF-Export über PDFix: laufende Nummer je Dokument (01.09.2026)

Befund Michael Karbe („Fehler beim Export" bei „Als PDF", Produktion Projekt
376): Der PDFix-Rückweg (`pdfix_roundtrip.import_alt_texts_pdfix` →
`pdfix_scripts/AltTag_Import_CSV.py`) ordnet die Alt-Texte über eine laufende
Nummer 1..N den Figures der EINEN PDF zu. Bis zum 01.09. ging
`images.image_index` in die CSV — der ist aber projektweit fortlaufend. Beim
zweiten Dokument eines Projekts (Nummern ab 11) oder nach dem Löschen eines
Dokuments (Nummern ab 3) brach der Import ab: „CSV referenziert laufende
Nummern [11, 12], aber der StructTree enthält nur 10 zählbare Figures". Auf
Produktion waren 6 von 23 PDFix-Projekten betroffen. Jetzt vergibt
`_pdfix_lfnr_je_dokument()` den Rang innerhalb des Dokuments (Sortierung
nach `image_index`), der genau der Figure-Nummer der Extraktion entspricht.
Unit-Test `tests/test_pdfix_lfnr.py`; Beweis auf Staging: Projekt 93,
Dokument 2 exportiert wieder.

## Hand-Text vor dem ersten Lauf = fertig; kein Anhalten (01.09.2026, Michael Karbe Punkt 1/2)

Der Hand-Text hatte in Anzeige und Export immer Vorrang — der erste
Sammellauf schaute aber nur auf `status = pending` und beschrieb ein von Hand
beschriebenes Bild trotzdem (5 Credits für einen Text, der unsichtbar unter
dem Hand-Text lag; der Dialog sagte „2 Bilder werden beschrieben", obwohl
eins schon fertig war). Jetzt macht ein nicht leerer Hand-Text auf einem nie
generierten Bild dieses fertig (`_handtext_macht_fertig`, Besitzer- und
Gast-Endpunkt; die Oberfläche stellt den Knopf am Bild sofort auf „Neu
generieren"). Der Sammellauf lässt es aus, die Rückfrage zählt es nicht mit.
Wer doch einen KI-Text will: „Neu generieren" am Bild — dieselbe Regel wie
beim bewussten Leeren: Was am Bild entschieden wurde, stößt nur ein Klick am
Bild um.

Einen laufenden Sammellauf kann man nicht anhalten; er endet, wenn er fertig
ist, das Guthaben aufgebraucht oder das Tageslimit erreicht ist. Die Rückfrage
sagt das jetzt („Nach dem Start lässt sich der Lauf nicht mehr anhalten; er
endet von selbst."). Ein Abbrechen-Knopf während des Laufs ist offen (eigene
Runde: der Lauf müsste je Bild nachsehen).

## KURSWECHSEL 01.09.2026 NACHMITTAG (Michael Karbe/Steve): Generieren überschreibt alles, Export = Browser

Diese Regeln lösen die Abschnitte „Bewusst geleert bleibt leer", „Hand-Text
vor dem ersten Lauf = fertig" und „drei Gründe" von heute Vormittag ab.

1. **Generieren überschreibt alles.** Der Sammel-Knopf („Alt-Texte
   generieren", Projekt oder Dokument) beschreibt ALLE Bilder des Umfangs mit
   der KI — mitgebrachte Texte, eigene Texte, geleerte Felder, fertige und nie
   generierte Bilder. Es gibt nur noch EINEN Modus (`modus: "alle"`, der
   Server ignoriert andere Werte); `force=True` immer (Cache umgangen, jeder
   Lauf kostet). Einzige Ausnahme: vom Autor in der Datei als dekorativ
   gekennzeichnete Bilder (Word). Rest eines abgebrochenen Laufs wird weiter
   zuerst abgearbeitet (kein Doppelkassieren). Die Rückfrage: „Alle 9 Bilder
   werden von der KI beschrieben; vorhandene Texte werden überschrieben, 3
   davon sind von dir geschrieben. Das kostet 45 Credits. …". Die Vorschau
   liefert dafür `eigene`.
2. **Sicherheitsnetz.** Ein eigener Text, den der Sammellauf überschreibt,
   wandert nach `images.alt_text_vorher`; am Bild erscheint „Vorherigen Text
   zurückholen" (`POST /api/images/{id}/alt-text/zurueck`).
3. **Export ist, was der Kunde sieht.** `_exportable_alt_text`: „dekorativ"
   (Text oder Bildtyp dekorativ ohne Text) → Kennzeichen; Text → Text; leeres
   Feld → `""` = KEIN Alternativtext in der Datei, ein mitgebrachter Text wird
   ENTFERNT (PDFix: Sentinel `__KEIN_ALT__` in der CSV, Import-Skript
   `RemoveKey("Alt")`; Word: `descr` entfernt, Kennzeichen aus; PyMuPDF-Weg:
   Bild nicht getaggt); `None` nur bei technischem Fehlertext (Bild
   unangetastet). Grund (Michael): Acrobat oder ein anderes Werkzeug soll
   später frei einen Text setzen können — „es muss wirklich leer sein".
4. **Zusammenfassung ohne Gründe.** „7 von 9 Bildern haben einen Text, 2
   haben keinen, 2 als dekorativ gekennzeichnet." (`mit_text`, `ohne_text`,
   `dekorativ`; alte Schlüssel bleiben für Skripte).
5. **Dekorativ bleibt dekorativ** (Steve): vom Autor oder von der KI so
   eingestuft → Kennzeichen wie bisher; Aufheben durch den Kunden später.
6. **Quickinfos unverändert** — die Regel gilt für Alt-Texte; das
   Formular-Werkzeug behält seine Quellen (Hand, PDF, Stammdaten, Gast).

Tests: `verify_leeren` (WYSIWYG über die Schnittstelle), `verify_ki_neu_abbruch`
und `verify_doppelkosten` (Kandidaten = alle), `test_export_leer` (CSV-Sentinel,
Word-descr entfernt), `ui_gendialog` (Dialogsätze), `ui_word` (Zusammenfassung).

### Nachtrag 01.09.2026 nachmittags: Wortlaut nach Michael Karbe, Abbruch, Zurückholen ausgeblendet

- **Rückfrage-Wortlaut** (Michaels Mail 01.09.): Dokument „Das Dokument
  beinhaltet insgesamt {n} Bilder, die Erstellung der Alt-Texte benötigt {c}
  Credits."; Projekt „Die Dokumente des Projekts beinhalten insgesamt {n}
  Bilder, …"; Einzahl-Fassungen; Guthaben „Dein Konto verfügt derzeit über
  ein Guthaben von {v} Credits." (Teil-Lauf-Sätze unverändert); danach ggf.
  „{n} Texte davon stammen von dir."; zuletzt „Der Erstellungsprozess kann
  bei Bedarf auch nach dem Start abgebrochen werden."
- **Abbruch** (Michael/Steve): Während des Laufs steht im Projektkopf
  „Generierung abbrechen" (Bilder: `POST /api/projects/{id}/generate/abbrechen`,
  Quickinfos: `POST …/quickinfos/abbrechen`). Das Bild (bzw. die Seite), das
  gerade in Arbeit ist, wird fertig; alle weiteren bleiben unberührt und
  kosten nichts. Der Lauf endet über denselben geordneten Weg wie bei
  Guthaben/Tageslimit: Rest-Vermerk, Lauf-Hinweis `grund: "abbruch"`
  (Ansage „Abgebrochen: {i} Bilder wurden bearbeitet, {n} blieben offen."),
  Rückstellung. Signal: prozesslokales Set `_abbruch_gewuenscht` (main.py)
  bzw. `_generierung[pid]["abbruch"]` (formular_api.py) — der Server läuft
  als EIN uvicorn-Prozess (Compose-CMD ohne --workers); bei mehreren Workern
  müsste das Signal in die Datenbank. Test: `verify_doppelkosten` Abschnitt 7.
- **„Vorherigen Text zurückholen"** ist in der Oberfläche AUSGEBLENDET
  (Steve: erst mal nicht, weniger Knöpfe). Backend, Spalte `alt_text_vorher`
  und Endpunkt bleiben; Einschalten über `window.VORHER_KNOPF = true` in app.html.
