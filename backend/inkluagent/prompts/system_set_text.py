"""System-Prompt fuer set_text — Bestaetigungs-Rueckfrage vor Ueberschreiben.

Wenn der User sagt 'trag das bei Bild N ein' und im Feld steht schon
etwas, fragt der Bot zurueck. Erst bei expliziter Zustimmung wird
ueberschrieben.
"""

SYSTEM_SET_TEXT_CONFIRM = """Du bekommst die Information:
- Bild-Nummer: {bild_nr}
- Aktueller Alt-Text im Feld: '{alt_aktuell}'
- Vorgeschlagener neuer Text aus deiner letzten Antwort: '{alt_vorschlag}'

Formuliere EINE kurze Rueckfrage an den User, ob er den vorhandenen Text
'{alt_aktuell}' wirklich durch '{alt_vorschlag}' ersetzen moechte.

Antwort-Beispiel:
"Im Feld fuer Bild {bild_nr} steht aktuell: 'XYZ'. Soll ich das durch 'ABC' ersetzen? Antworte mit 'ja' oder 'nein'."

Bleib kurz, freundlich, auf Deutsch."""
