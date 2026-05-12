"""Zentraler Dispatcher für Pass-3 (Beschreibungs-Pass).

Routet anhand BildtypEffective auf den passenden Per-Bildtyp-Builder
in beschreibung_foto.py / beschreibung_daten.py / beschreibung_mini.py.

Signatur-Unterschied zwischen den Familien:
- foto-Familie + Daten-Familie: brauchen InventarOutput (vorheriger Pass)
- Mini-Pipelines (logo/icon/funktional): brauchen ClassificationOutput
  + original_alt (kein Inventar-Pass)
- Dekorativ: gar kein Builder — handle_dekorativ_classification gibt
  direkt das Result-Dict zurück (siehe dekorativ.py)

Daher zwei separate Dispatcher-Funktionen statt einer mit verwirrenden
kwargs. Klarer für den Orchestrator-Code.
"""
from __future__ import annotations

from typing import Optional

from prompts.components.schemas import (
    BildtypEffective,
    ClassificationOutput,
    InventarOutput,
)

from .beschreibung_daten import (
    build_beschreibung_prompt_diagramm,
    build_beschreibung_prompt_illustration,
    build_beschreibung_prompt_infografik,
    build_beschreibung_prompt_karte,
    build_beschreibung_prompt_screenshot,
    build_beschreibung_prompt_strukturformel,
    build_beschreibung_prompt_tabelle,
)
from .beschreibung_foto import (
    build_beschreibung_prompt_foto_architektur,
    build_beschreibung_prompt_foto_essen,
    build_beschreibung_prompt_foto_event,
    build_beschreibung_prompt_foto_landschaft,
    build_beschreibung_prompt_foto_objekte,
    build_beschreibung_prompt_foto_personen,
)
from .beschreibung_mini import (
    build_beschreibung_prompt_funktional,
    build_beschreibung_prompt_icon,
    build_beschreibung_prompt_logo,
)


# Dispatcher-Tabelle für die 13 Bildtypen mit Inventar-Pass.
_INVENTAR_BUILDERS = {
    'foto_event': build_beschreibung_prompt_foto_event,
    'foto_personen': build_beschreibung_prompt_foto_personen,
    'foto_objekte': build_beschreibung_prompt_foto_objekte,
    'foto_essen': build_beschreibung_prompt_foto_essen,
    'foto_landschaft': build_beschreibung_prompt_foto_landschaft,
    'foto_architektur': build_beschreibung_prompt_foto_architektur,
    'illustration': build_beschreibung_prompt_illustration,
    'diagramm': build_beschreibung_prompt_diagramm,
    'tabelle': build_beschreibung_prompt_tabelle,
    'karte': build_beschreibung_prompt_karte,
    'infografik': build_beschreibung_prompt_infografik,
    'screenshot': build_beschreibung_prompt_screenshot,
    'strukturformel': build_beschreibung_prompt_strukturformel,
}


# Dispatcher-Tabelle für die 3 Bildtypen ohne Inventar-Pass.
_MINI_BUILDERS = {
    'logo': build_beschreibung_prompt_logo,
    'icon': build_beschreibung_prompt_icon,
    'funktional': build_beschreibung_prompt_funktional,
}


def build_beschreibung_prompt_with_inventar(
    bildtyp: BildtypEffective,
    inventar: InventarOutput,
    enriched_context: str,
    width: int,
    height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Dispatcher für die 13 Bildtypen mit Inventar-Pass (Pass 3 nach Pass 2).

    Raises:
      ValueError wenn bildtyp keinem der 13 Inventar-Builder zugeordnet ist
      (z.B. bei 'logo' — dort gehört build_beschreibung_prompt_mini hin,
      bei 'dekorativ' — dort gibt's keinen Builder, sondern den
      Dict-Return aus handle_dekorativ_classification).
    """
    builder = _INVENTAR_BUILDERS.get(bildtyp)
    if builder is None:
        raise ValueError(
            f"Kein Inventar-Builder für bildtyp='{bildtyp}'. "
            f"Erlaubt: {sorted(_INVENTAR_BUILDERS.keys())}. "
            f"Für logo/icon/funktional → build_beschreibung_prompt_mini(); "
            f"für dekorativ → handle_dekorativ_classification() in dekorativ.py."
        )
    return builder(
        inventar=inventar,
        enriched_context=enriched_context,
        width=width,
        height=height,
        user_hint=user_hint,
    )


def build_beschreibung_prompt_mini(
    bildtyp: BildtypEffective,
    classification: ClassificationOutput,
    enriched_context: str,
    width: int,
    height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Dispatcher für die 3 Mini-Pipeline-Bildtypen (logo/icon/funktional).

    Raises:
      ValueError wenn bildtyp keinem der 3 Mini-Builder zugeordnet ist.
    """
    builder = _MINI_BUILDERS.get(bildtyp)
    if builder is None:
        raise ValueError(
            f"Kein Mini-Builder für bildtyp='{bildtyp}'. "
            f"Erlaubt: {sorted(_MINI_BUILDERS.keys())}. "
            f"Für die 13 Inventar-Bildtypen → build_beschreibung_prompt_with_inventar()."
        )
    return builder(
        classification=classification,
        enriched_context=enriched_context,
        width=width,
        height=height,
        original_alt=original_alt,
        user_hint=user_hint,
    )
