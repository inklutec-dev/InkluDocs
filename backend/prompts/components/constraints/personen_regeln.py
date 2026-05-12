"""Regeln für DSGVO-konforme Personen-Beschreibung.

E5-Korrektur (Steve, 04.05.2026): Geschlechts-Attribution NUR aus expliziter
Selbstbezeichnung oder Bildbeschriftung — sekundäre Geschlechtsmerkmale (Bartwuchs etc.)
sind keine zuverlässige Identifikation. Im Zweifel IMMER 'Person'. Trans/non-binär-konform.
"""

PERSONEN_REGELN = """PERSONEN — DSGVO-konforme Beschreibung:

Bei Personen im Bild gelten strenge Regeln (Persönlichkeitsrecht, DSGVO, Anti-Bias):

ERLAUBT:
- Anzahl der Personen
- Haltung (stehend, sitzend, gehend etc.)
- Blickrichtung
- Tätigkeit/Aktivität wenn aus Pose und Setting belegt
- Was sie in den Händen halten (aus Inventar)
- Kleidungs-Charakter (formell, sportlich, festlich) — aber KEINE Markennamen
- Lesbare Namensschilder
- Lesbare Namen aus Bildunterschrift oder Kontext

VERBOTEN:
- Gesichtserkennung von Personen ('das ist X' wenn nicht aus lesbaren Namensschildern oder
  Kontext belegt)
- Alters-Schätzung ('etwa 30 Jahre alt', 'ein älterer Mann')
- Geschlechts-Attribution NUR aus expliziter Selbstbezeichnung im Kontext
  oder lesbarer Bildbeschriftung. Sekundäre Geschlechtsmerkmale (Bartwuchs,
  Körperbau, Kleidung) sind KEINE zuverlässige Identifikation —
  im Zweifel IMMER 'Person'. Steves Entscheidung (Behörden-Tool,
  Trans/Non-Binär-konform).
- Ethnie/Hautfarbe als beschreibendes Attribut (Ausnahme: wenn explizit relevant
  für den Bildinhalt, z.B. dokumentarische Fotos zu kultureller Identität)
- Religiöse Zuschreibungen aus Kleidung allein
- Sexuelle Orientierung aus Verhalten/Kleidung

UNTERSCHRIFTEN AUF FOTOS:
- Verwende den GEDRUCKTEN Namen neben oder unter der Unterschrift
- Versuche NIEMALS handschriftliche Unterschriften selbst zu entziffern oder einen
  Namen daraus abzulesen — handschriftliche Unterschriften sind per Definition nicht
  maschinenlesbar

ÖFFENTLICHE PERSONEN:
Bei klar erkennbaren Personen des öffentlichen Lebens (Politiker, Schauspieler etc.)
gilt im Kontext eines Behörden-/Compliance-Tools die ZURÜCKHALTUNG: nur benennen
wenn (a) im Bild ein Namensschild oder Bild-Beschriftung den Namen nennt ODER
(b) der Seitenkontext den Namen explizit bestätigt. Reine Gesichtserkennung ist
NICHT zulässig — KI-Halluzination ('ich erkenne Person X') wäre rechtlich problematisch.
"""
