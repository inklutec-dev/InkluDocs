"""Pipeline-Endpunkt für dekorative Bilder + Heuristik-Safety-Net.

Dekorative Bilder bekommen einen leeren Alt-Text — kein Beschreibungs-Pass,
kein Validator. WCAG-konform für rein schmückende Bilder ohne Informationswert.

Mini-Korrektur d (Steve, 04.05.2026): handle_dekorativ_classification
gibt jetzt Tuple (result, corrected_type) zurück statt Optional[dict] +
nachträglichem zweiten validate_dekorativ-Aufruf. Vermeidet doppelte
Heuristik-Auswertung im Override-Pfad.
"""
from __future__ import annotations

from typing import Optional

from prompts.components.schemas import (
    BildtypTopLevel,
    ClassificationOutput,
)


# Heuristische Schwellen für die Plausibilitätsprüfung.
# Dekorative Bilder sind in Production zu 90% kleiner als 100x100 px;
# Bilder mit brauchbarem original_alt sind selten dekorativ.
_DEKORATIV_MAX_SIDE = 100
_GENERIC_ALT_TEXTE = {'', 'bild', 'foto', 'grafik', 'image', 'picture'}


def validate_dekorativ(
    classification: ClassificationOutput,
    original_alt: str,
    width: int,
    height: int,
) -> BildtypTopLevel:
    """Heuristik: ist die 'dekorativ'-Klassifikation plausibel?

    Returns:
      'dekorativ' wenn die Klassifikation plausibel scheint.
      Sonst: ein wahrscheinlicherer Top-Level-Typ ('foto' als Default-
      Fallback). Der Aufrufer (Orchestrator) ruft dann mit dem korrigierten
      Typ den Klassifikator NICHT erneut, sondern setzt ihn direkt
      (Override mit konfidenz='mittel').

    Heuristik:
    1. Bild ist groß (länger als _DEKORATIV_MAX_SIDE in Breite ODER Höhe)
       UND original_alt enthält brauchbaren Text → vermutlich KEIN dekorativ.
    2. Original_alt-Text enthält etwas Substantielles (nicht generisch wie
       'Bild'/'Foto'/leer) → vermutlich KEIN dekorativ.
    3. Sonst → dekorativ bestätigt.

    Default-Override-Typ ist 'foto' — das ist der sicherste Fallback,
    weil der Beschreibungs-Pass für foto die meisten Schutz-Constraints hat.
    Wenn das in Eval-Tests systematisch falsch ist (z.B. immer logo),
    Default hier anpassen.
    """
    alt_normalized = original_alt.strip().lower() if original_alt else ''
    has_meaningful_alt = (
        alt_normalized
        and alt_normalized not in _GENERIC_ALT_TEXTE
        and len(alt_normalized) >= 5
    )
    is_large = width > _DEKORATIV_MAX_SIDE or height > _DEKORATIV_MAX_SIDE

    if is_large and has_meaningful_alt:
        return 'foto'
    if has_meaningful_alt and len(alt_normalized) > 30:
        # Selbst kleine Bilder mit langem alt_text sind selten dekorativ.
        return 'foto'

    return 'dekorativ'


def handle_dekorativ_classification(
    classification: ClassificationOutput,
    image_path: str,
    width: int,
    height: int,
    original_alt: str = '',
) -> tuple[Optional[dict], Optional[str]]:
    """Pipeline-Endpunkt für dekorative Bilder.

    Returns Tuple (result_dict, corrected_type):
      (result_dict, None)         — dekorativ bestätigt durch Heuristik,
                                     Pipeline endet mit dem Result.
      (None, corrected_type)      — Heuristik-Override, Aufrufer
                                     (Orchestrator) klassifiziert mit
                                     corrected_type weiter.

    image_path wird aktuell nicht genutzt — bleibt im Interface für
    spätere Erweiterungen (z.B. 'gleicher Hash wie bekanntes Hintergrund-
    Bild' als zusätzliche Heuristik-Quelle).
    """
    corrected_type = validate_dekorativ(
        classification, original_alt, width, height
    )
    if corrected_type != 'dekorativ':
        # Heuristik-Override: Bild ist doch informativ, weiter im Hauptpfad.
        return (None, corrected_type)

    # Dekorativ bestätigt — leer, WCAG-konform, keine weiteren Pässe.
    return (
        {
            'bildtyp': 'dekorativ',
            'konfidenz': classification.konfidenz,
            'alt_text': '',
            'langbeschreibung': '',
            'needs_review': False,
            'pipeline_steps': 'classified:dekorativ',
            'inventar_json': None,
            'validation_result': None,
        },
        None,
    )
