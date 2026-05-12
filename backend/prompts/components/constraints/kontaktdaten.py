"""Pflicht zur wortgetreuen Übernahme lesbarer Kontaktdaten."""

KONTAKTDATEN_PFLICHT = """LESBARE KONTAKTDATEN — KRITISCHE PFLICHT:

Wenn das Inventar lesbare_texte mit Typ 'kontaktdaten', 'url', 'datum' oder 'zahl' enthält,
MÜSSEN diese im alt_text oder in der Langbeschreibung erscheinen — wortgetreu, mit
korrekten Trennzeichen.

Für Screenreader-Nutzer sind diese Daten oft der einzige Zugang zur Information.
Ein Alt-Text der eine lesbare Telefonnummer übersieht ist UNVOLLSTÄNDIG, auch wenn
er das Bild sonst korrekt beschreibt.

Beispiele:
  '02 28 / 24 25 26 27' — exakt so übernehmen, nicht zu '022824252627' zusammenziehen
  'Mo-Fr 9-17 Uhr' — wortwörtlich
  'info@beispiel.de' — exakt
  'https://www.beispiel.de/kontakt' — vollständig"""
