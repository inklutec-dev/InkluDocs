"""Wieder-exportiert alle Constraint-Konstanten der v4-Pipeline.

Importpfad-Beispiel:
  from prompts.components.constraints import ANTI_HALLUZINATION_REGELN

Paket 1 (16.07.2026): Tote Module entfernt — personen_regeln.py, wcag.py,
kontext_nutzung.py und verbotene_formulierungen.py wurden von keinem Builder
mehr eingebunden (Regel-Inventur, Strukturbefund 2). Der inhaltliche Kern von
kontext_nutzung.py (Anti-Redundanz zur Bildunterschrift, Kontext-Anreicherung
ohne erfundene Handlung / budni-Korrektur) lebt jetzt in
prompts/builders/beschreibung_foto.py (_render_zweck_block) weiter.
"""
from .atmosphere_evidenz import ATMOSPHAERE_REGEL
from .eigennamen import EIGENNAMEN_REGELN
from .halluzination import ANTI_HALLUZINATION_KERN, ANTI_HALLUZINATION_REGELN
from .kunstwerk import KUNSTWERK_REGEL
from .lizenz_logos import LIZENZ_LOGOS_REGELN

# Entfernt (Prompt-Pruefung 09.09.2026): kontaktdaten.py (KONTAKTDATEN_PFLICHT,
# von keinem Builder mehr eingebunden; Inhalt lebt in _LESBARER_TEXT der
# Datenfamilie) und evidenz_stufen.py (leere Konstante seit Paket 1).

__all__ = [
    'ATMOSPHAERE_REGEL',
    'EIGENNAMEN_REGELN',
    'ANTI_HALLUZINATION_KERN',
    'ANTI_HALLUZINATION_REGELN',
    'KUNSTWERK_REGEL',
    'LIZENZ_LOGOS_REGELN',
]
