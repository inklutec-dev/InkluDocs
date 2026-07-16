"""Evidenz-Stufen für Marken-/Produkt-/Text-Identifikationen.

Paket 1 (16.07.2026): Geltungsbereich auf den tatsächlichen Einsatzkontext
eingegrenzt — aktiv nur noch im logo-Builder (beschreibung_mini.py).
Personen, Orte und Gebäude wurden aus der Geltungs-Aufzählung gestrichen:
deren Regeln stehen in den Foto-Buildern (beschreibung_foto.py) und in
SYSTEM_BESCHREIBUNG (roles.py) — inklusive der dort gewollten
Wahrzeichen-/Promi-Erlaubnis, die die alte Stufe-3-Wand widersprach.
"""

EVIDENZ_STUFEN_REGELN = """EVIDENZ-BASIERTE IDENTIFIKATION (drei Stufen):

STUFE 1 (immer erlaubt): Text, Namen, Logos die im Bild KLAR LESBAR sind.
  → direkt nennen
  Beispiel: Schild 'Bundesministerium des Innern' → 'Bundesministerium des Innern'

STUFE 2 (erlaubt): Lesbarer Text oder eindeutiges Logo + Allgemeinwissen.
  → benennen
  Beispiel: Inschrift 'EQUAL JUSTICE UNDER LAW' → 'Supreme Court der USA'
  Beispiel: Mercedes-Stern + Fahrzeug-Form → 'Mercedes-Benz' (nicht das Modell raten)

STUFE 3 (verboten): Kein Text, kein Logo, nur visueller Eindruck.
  → allgemein beschreiben, NICHT spekulieren
  Beispiel: graues Industriegebäude ohne Schild → 'ein industrielles Gebäude',
            NICHT 'Siemens-Werk'
  Beispiel: Person ohne Namensschild → 'eine Person', NICHT einen Namen raten

Diese Stufen gelten für Marken-, Produkt- und Text-Identifikationen:
Marken, Logos, Siegel, Fahrzeugmodelle, Produktbezeichnungen.
"""
