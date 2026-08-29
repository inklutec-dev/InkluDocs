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
„Exportieren“, das den Export-Dialog für genau dieses Dokument öffnet
(Einzeldatei, Preis/Guthaben passend). Eine Auswahlliste gibt es im Dialog
nicht mehr (Steve 28.08.2026): der Knopf entscheidet. Der Hauptknopf oben
heißt bei mehreren Dokumenten „Ganzes Projekt exportieren“ und liefert alle
als ZIP, bei einem Dokument „Exportieren“; die Dialog-Überschrift nennt das
Ziel („Dokument ‚x.pdf‘ exportieren“ / „Ganzes Projekt exportieren (3
Dokumente)“). Die Generier-Knöpfe oben gelten weiterhin für das ganze Projekt. Web-Werkzeug:
ein Dokument = die Seite, daher keine Zusatzknöpfe.
