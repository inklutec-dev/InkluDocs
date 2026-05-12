"""Evidenz-Stufen für visuelle Identifikationen."""

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

Diese Stufen gelten für alle visuellen Identifikationen: Marken, Personen, Orte,
Gebäude, Fahrzeugmodelle, Tier- oder Pflanzenarten, geografische Koordinaten.
"""
