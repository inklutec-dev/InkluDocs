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
from .evidenz_stufen import EVIDENZ_STUFEN_REGELN
from .halluzination import ANTI_HALLUZINATION_REGELN
from .kontaktdaten import KONTAKTDATEN_PFLICHT
from .lizenz_logos import LIZENZ_LOGOS_REGELN

__all__ = [
    'ATMOSPHAERE_REGEL',
    'EIGENNAMEN_REGELN',
    'EVIDENZ_STUFEN_REGELN',
    'ANTI_HALLUZINATION_REGELN',
    'KONTAKTDATEN_PFLICHT',
    'LIZENZ_LOGOS_REGELN',
]
