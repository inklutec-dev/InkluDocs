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
3. **{n} … neu generieren, {c} Credits** — derselbe Knopf wie 1, sobald
   keine Lücken mehr da sind: fasst nur KI-Texte ohne Hand-Text an
   (`POST /api/projects/{id}/generate` bzw. `…/quickinfos/generieren` mit
   `{"modus": "ki_neu"}`). Keine Rückfrage: Anzahl und Credits stehen sichtbar
   im Knopf. Bilder: 5 Credits je Bild, Cache je Bild geräumt
   (`force_regenerate`); Felder: 1 Credit je Feld (die Oberfläche rechnet mit
   `window.CREDIT_PREISE` aus `billing.preise_fuer_frontend()`).

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
zwei weitere Knöpfe: „Alle generieren“, solange das Dokument Lücken hat,
sonst „n … neu generieren, c Credits“ (nur KI-Texte dieses Dokuments;
`POST …/generate` bzw. `…/quickinfos/generieren` mit `document_id`), und
„Herunterladen“, das den Dialog für genau dieses Dokument öffnet
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

Beteiligt: `main._ki_neu_rest_lesen`, `main._ki_neu_rest_schreiben`, `main._ki_neu_zurueck`
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
