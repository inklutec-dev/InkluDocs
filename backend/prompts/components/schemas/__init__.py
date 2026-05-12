"""Pydantic-Output-Schemas für die v4-Pipeline.

Jede Pipeline-Stufe gibt ein striktes Schema-Objekt zurück. Das Schema
ist der Vertrag zwischen Stufe und Modell — fehlt ein Pflichtfeld,
schlägt model_validate fehl und der Aufruf wird wiederholt (max. 2x)
mit der Anweisung 'Du hast Feld X übersprungen, fülle es zwingend'.
"""
from .beschreibung import (
    AtmosphaereBeleg,
    BeschreibungOutput,
    IconBeschreibungOutput,
)
from .classification import (
    BildtypEffective,
    BildtypTopLevel,
    ClassificationOutput,
    FotoSubtyp,
    KonfidenzStufe,
)
from .inventar import (
    InventarOutput,
    ObjektInBild,
    PersonInBild,
    TextInBild,
)
from .validierung import InventarVergleich, ValidierungOutput

__all__ = [
    # classification
    "BildtypTopLevel",
    "FotoSubtyp",
    "BildtypEffective",
    "KonfidenzStufe",
    "ClassificationOutput",
    # inventar
    "ObjektInBild",
    "PersonInBild",
    "TextInBild",
    "InventarOutput",
    # beschreibung
    "AtmosphaereBeleg",
    "BeschreibungOutput",
    "IconBeschreibungOutput",
    # validierung
    "InventarVergleich",
    "ValidierungOutput",
]
