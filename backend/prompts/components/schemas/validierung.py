"""Output-Schema für Pass 4 (Validierung) der v4-Pipeline."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class InventarVergleich(BaseModel):
    """Pro Aussage im Beschreibungs-Output: ist sie durch Inventar gedeckt?"""

    aussage: str
    deckung: Literal["inventar_belegt", "atmosphaere_belegt", "nicht_belegt"]
    quelle: Optional[str] = Field(
        None, description="Welches Inventar-Item belegt diese Aussage?"
    )


class ValidierungOutput(BaseModel):
    """Output von Pass 4 (Validierung)."""

    validierung_ok: bool = Field(...)

    inventar_vergleich: list[InventarVergleich] = Field(
        ...,
        description="Pflicht: jede Aussage in alt_text und langbeschreibung einzeln prüfen.",
    )

    nicht_belegte_aussagen: list[str] = Field(
        default_factory=list,
        description=(
            "Aussagen die das Inventar NICHT stützt. "
            "Wenn nicht-leer → validierung_ok=false."
        ),
    )

    fehlende_wichtige_inventar_items: list[str] = Field(
        default_factory=list,
        description=(
            "Inventar-Items die im Output fehlen, obwohl sie wichtig sind. "
            "Z.B. lesbarer Text, Personen-Aktivität, dominante Objekte."
        ),
    )

    korrektur_alt_text: Optional[str] = Field(
        None, max_length=400,
        description=(
            "Wenn validierung_ok=false: Korrektur-Vorschlag. "
            "None wenn keine sichere Korrektur."
        ),
    )

    korrektur_langbeschreibung: Optional[str] = Field(None, max_length=2000)

    needs_review: bool = Field(
        ...,
        description=(
            "Soll ein Mensch nochmal drüberlesen? "
            "Auch true bei mittlerer Inventar-Konfidenz."
        ),
    )
