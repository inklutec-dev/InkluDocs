"""Output-Schema für Pass 2 (Inventar) der v4-Pipeline."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .classification import KonfidenzStufe


class ObjektInBild(BaseModel):
    """Ein einzelnes Objekt/Element im Bild."""

    beschreibung: str = Field(
        ..., min_length=3, max_length=300,
        description="Was ist das? Form, Farbe, Material — KEINE Funktionsdeutung.",
    )
    position: str = Field(
        ..., description="Wo im Bild? (links/Mitte/rechts/oben/unten/Vordergrund/Hintergrund)"
    )
    sicherheit: KonfidenzStufe = Field(
        ..., description="Wie sicher bin ich, dass das wirklich so ist?"
    )
    moegliche_identifikationen: list[str] = Field(
        default_factory=list, max_length=4,
        description="Bei Unsicherheit: 2-4 Hypothesen. Bei Sicherheit: 1 Eintrag.",
    )


class PersonInBild(BaseModel):
    """Eine Person — ohne Identifikation, nur sichtbare Eigenschaften.

    Bewusst NICHT erfasst: Alter, Geschlechts-Attribution, ethnische
    Zuordnung, Namens-Vermutung. DSGVO/Anti-Halluzination.
    """

    position: str
    haltung: str = Field(..., description="stehend/sitzend/Bewegung etc.")
    blickrichtung: Optional[str] = Field(
        None, description="zur Kamera, weg, seitlich etc."
    )
    objekte_in_haenden: list[ObjektInBild] = Field(default_factory=list)
    kleidungs_charakter: Optional[str] = Field(
        None,
        description="formell/sportlich/Arbeitskleidung/festlich — keine Markennamen",
    )


class TextInBild(BaseModel):
    """Ein lesbarer Text im Bild."""

    inhalt: str = Field(..., description="Wortwörtlich was lesbar ist")
    typ: Literal[
        "überschrift", "fließtext", "logo", "beschriftung", "datum",
        "zahl", "kontaktdaten", "url",
    ] = Field(...)
    vollstaendigkeit: Literal["vollständig", "teilweise", "abgeschnitten"] = Field(...)


class InventarOutput(BaseModel):
    """Output von Pass 2 (Inventar).

    Das Modell MUSS jedes Feld ausfüllen, auch wenn leer ([] oder None).
    Leer-Lassen ist eine bewusste Entscheidung, kein Übersehen.
    """

    foto_subtyp: Optional[Literal[
        "foto_personen", "foto_objekte", "foto_landschaft",
        "foto_architektur", "foto_essen", "foto_event",
    ]] = Field(None, description="Nur wenn bildtyp=foto, sonst None")

    personen: list[PersonInBild] = Field(default_factory=list)

    objekte: list[ObjektInBild] = Field(
        default_factory=list,
        description="Alle Nicht-Personen-Objekte mit Beschreibung+Position+Sicherheit",
    )

    lesbare_texte: list[TextInBild] = Field(
        default_factory=list,
        description="Jeder lesbare Text. KEINE Texte erfinden, nur was tatsächlich da steht.",
    )

    setting: dict = Field(
        default_factory=dict,
        description="raum_charakter, beleuchtung, dominante_farben, ungefaehre_szene",
    )

    handlung: Optional[str] = Field(
        None, max_length=300,
        description="Was passiert? Nur belegt durch sichtbare Indikatoren. None erlaubt.",
    )

    halluzinations_warnung: list[str] = Field(
        default_factory=list,
        description=(
            "Klassische Stolperfallen für DIESES Bild, vor denen Pass 3 sich hüten soll. "
            "Beispiel: 'Hellfarbene Glasur könnte als Flüssigkeit fehlinterpretiert werden.' "
            "Beispiel: 'Stilisierte Tierdarstellung — nicht voreilig auf Spezies festlegen.'"
        ),
    )

    inventar_konfidenz_gesamt: KonfidenzStufe = Field(
        default='mittel',
        description='Gesamt-Sicherheit des Inventars (default: mittel, wenn Tool-Use es nicht setzt)',
    )
