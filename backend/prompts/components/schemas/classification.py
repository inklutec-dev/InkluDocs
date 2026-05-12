"""Output-Schema für Pass 1 (Klassifikation) der v4-Pipeline."""
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

BildtypTopLevel = Literal[
    "foto", "diagramm", "tabelle", "karte", "logo", "screenshot",
    "infografik", "strukturformel", "icon", "funktional", "dekorativ",
    "illustration",
]

# K2-Fix: Sub-Typen für foto, vom Inventar-Pass entschieden.
# Stehen separat, weil ClassificationOutput.bildtyp nur Top-Level kennt
# (der Klassifikator entscheidet 'foto', der Inventar-Pass entscheidet 'foto_event').
FotoSubtyp = Literal[
    "foto_personen", "foto_event", "foto_objekte",
    "foto_landschaft", "foto_architektur", "foto_essen",
]

# Effektiver Bildtyp nach dem Inventar-Pass: Top-Level ODER foto-Sub-Typ.
# Wird an Beschreibungs- und Validierungs-Builder übergeben, damit
# Sub-Typ-spezifische Prompt-Bausteine (VALIDIERUNG_SPEZIAL["foto_event"]
# etc.) greifen können. Ohne diesen Type-Alias kollidiert eine Übergabe
# von "foto_event" mit der BildtypTopLevel-Annotation.
BildtypEffective = Union[BildtypTopLevel, FotoSubtyp]

KonfidenzStufe = Literal["hoch", "mittel", "niedrig"]


class ClassificationOutput(BaseModel):
    """Output von Pass 1 (Klassifikation).

    Sub-Typ wird hier NICHT entschieden — das macht der Inventar-Pass,
    der besser sieht ob ein Foto Personen, Landschaft, Objekte etc. zeigt.
    """

    bildtyp: BildtypTopLevel = Field(
        ..., description="Top-Level-Typ des Bildes"
    )
    konfidenz: KonfidenzStufe = Field(
        ..., description="Wie sicher ist die Klassifikation?"
    )
    ist_dekorativ: bool = Field(
        False, description="True nur wenn Bild rein dekorativ ohne Information"
    )
    original_alt_brauchbar: bool = Field(
        False, description="True wenn original_alt eine sinnvolle Beschreibung enthält"
    )
    klassifikations_begruendung: str = Field(
        ..., min_length=10, max_length=200,
        description="Ein Satz: warum dieser Typ? Pflicht zur Selbstbegründung.",
    )

    # Lean-Mode-Erweiterung (08.05.2026): wenn V4_PASS_MODE=lean aktiv ist,
    # entscheidet der Klassifikator den foto-Sub-Typ direkt mit, sodass der
    # separate Inventar-Pass entfällt. Im Multi-Pass-Modus bleibt das Feld
    # None und der Inventar-Pass entscheidet wie bisher.
    foto_subtyp: Optional[FotoSubtyp] = Field(
        None,
        description="Lean-Mode: bei bildtyp=foto direkt den Sub-Typ mitwaehlen. "
                    "Im Multi-Pass-Modus None (entscheidet Inventar-Pass).",
    )
