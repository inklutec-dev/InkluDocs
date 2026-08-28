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
