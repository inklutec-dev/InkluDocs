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
   obenauf. 1 Credit.
3. **{n} … neu generieren, {c} Credits** — derselbe Knopf wie 1, sobald
   keine Lücken mehr da sind: fasst nur KI-Texte ohne Hand-Text an
   (`POST /api/projects/{id}/generate` bzw. `…/quickinfos/generieren` mit
   `{"modus": "ki_neu"}`). Keine Rückfrage: Anzahl und Credits stehen sichtbar
   im Knopf. Bilder: 1 Credit je Bild, Cache je Bild geräumt
   (`force_regenerate`); Felder: 1 Credit je Seite.

## Was nie angefasst wird

Sammelläufe überschreiben keine Hand-Texte (Bilder: `alt_text_edited`
gesetzt; Felder: `quelle` hand/pdf/stammdaten/gast/chat). Beim Feld-Pass gilt
zusätzlich: Texte, die während des Laufs von Hand gefüllt wurden, bleiben
(`UPDATE … WHERE quickinfo leer` bzw. `quelle = ki`).

## Export-Preis (Staffel, Michael Karbe 28.08.2026)

Jeder Datei-Export (PDF mit Alt-Texten, Word mit Alt-Texten, PDF mit Quickinfos)
kostet `billing.export_preis(anzahl)` = 5 Credits Grundpreis + 1 Credit je
angefangene 10 Bilder bzw. Felder der exportierten Datei; beim
Alle-Dokumente-ZIP werden die Elemente aller Dokumente zusammengezählt, der
Grundpreis fällt einmal an. Beispiele: 1 Feld = 6, 26 Felder = 8, 50 Felder =
10, 100 Bilder = 15 Credits. Gilt für alle Konten (Free hat 10 Credits).
Tabellen-Exporte (JSON, CSV, Excel, Formular-CSV) bleiben kostenlos. Beide
Zahlen stehen in `billing.EXPORT_GRUNDPREIS` / `EXPORT_STAFFEL`.

Vor dem Export zeigt der Dialog Preis und Guthaben
(`POST /api/projects/{id}/export/preis` bzw. `…/export/summary`). Reicht das
Guthaben nicht (`billing.export_pruefung`), antwortet der Export mit 402 und
beiden Zahlen; die Oberfläche zeigt die barrierefreie Meldung „Der Export
würde 8 Credits benötigen, du verfügst derzeit über 7 Credits …“ mit den
Knöpfen „Zu Abo & Verbrauch“ und „Schließen“ (`zeigeCreditsMeldung`,
app.html, beide Werkzeuge). Die Antwort trägt `X-Export-Credits`.
