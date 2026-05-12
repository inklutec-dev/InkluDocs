"""Pass-Builder für die v4-Pipeline.

Ein Builder pro Pass:
  Pass 1: build_classification_prompt           (classification.py)
  Pass 2: build_inventar_prompt                 (inventar.py)
  Pass 3: build_beschreibung_prompt_with_inventar  (beschreibung.py — 13 Bildtypen)
          build_beschreibung_prompt_mini           (beschreibung.py — 3 Mini-Pipelines)
          handle_dekorativ_classification          (dekorativ.py — Pipeline-Endpunkt)
  Pass 4: build_validierung_prompt              (validierung.py)

Plus:
  helpers.py — user_hint_block, extract_link_target_from_context, load_examples
  dekorativ.py — handle_dekorativ_classification (Tuple-Return), validate_dekorativ
"""
from .beschreibung import (
    build_beschreibung_prompt_mini,
    build_beschreibung_prompt_with_inventar,
)
from .classification import build_classification_prompt
from .combo import build_combined_inventar_beschreibung_prompt
from .dekorativ import handle_dekorativ_classification, validate_dekorativ
from .helpers import (
    Examples,
    extract_link_target_from_context,
    load_examples,
    resolve_prompt_mode,
    user_hint_block,
)
from .inventar import BILDTYP_INVENTAR_SCHWERPUNKTE, build_inventar_prompt
from .validierung import VALIDIERUNG_SPEZIAL, build_validierung_prompt

__all__ = [
    # Pass 1
    'build_classification_prompt',
    # Pass 2
    'build_inventar_prompt',
    'BILDTYP_INVENTAR_SCHWERPUNKTE',
    # Pass 3
    'build_combined_inventar_beschreibung_prompt',  # Lean-Mode (08.05.2026)
    'build_beschreibung_prompt_with_inventar',
    'build_beschreibung_prompt_mini',
    # Pipeline-Endpunkt für dekorativ (kein Pass 3)
    'handle_dekorativ_classification',
    'validate_dekorativ',
    # Pass 4
    'build_validierung_prompt',
    'VALIDIERUNG_SPEZIAL',
    # Helpers
    'user_hint_block',
    'resolve_prompt_mode',
    'extract_link_target_from_context',
    'load_examples',
    'Examples',
]
