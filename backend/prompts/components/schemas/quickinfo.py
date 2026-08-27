"""Output-Schema des Feld-Passes (Quickinfo-Werkzeug, Stufe 2, 27.08.2026).

Ein Aufruf je Formularseite liefert fuer jedes angefragte Feld GENAU EINEN
Eintrag. Das Schema ist der Vertrag mit dem Modell (Tool-Use, strikt):
fehlt ein Pflichtfeld oder ist ein Wert ausserhalb der Grenzen, wird der
Aufruf einmal mit Korrekturhinweis wiederholt (bedrock_client).

Die Nachpruefung in formular_ki.py verlaesst sich NICHT auf die
Selbsteinschaetzung des Modells (sicherheit): Beleg, Lage und Regeln werden
deterministisch geprueft und koennen die Sicherheit nur senken, nie heben.
"""
from typing import Literal

from pydantic import BaseModel, Field


class QuickinfoFeldOutput(BaseModel):
    """Ein Feld der Seite."""
    feld_index: int = Field(..., ge=1, description="Laufende Nummer des Feldes, exakt wie in der Feldliste angegeben (F<n>).")
    quickinfo: str = Field(
        ..., min_length=1, max_length=200,
        description="Die Quickinfo: ein Satz, der sagt, was in das Feld einzugeben ist, mit Gruppe und Format, "
                    "in der geforderten Sprache. Keine Anleitung, keine Feldart, kein technischer Name.",
    )
    beleg: str = Field(
        "", max_length=300,
        description="WOERTLICHE Textstelle der Seite (aus dem SEITENTEXT-Block), aus der die Quickinfo folgt — "
                    "meist die Beschriftung neben dem Feld und ggf. die Abschnittsueberschrift. Leer nur, wenn es keinen Beleg gibt.",
    )
    gruppe: str = Field(
        "", max_length=120,
        description="Abschnitt/Gruppe, zu der das Feld gehoert (z. B. 'Antragsteller'), wortgetreu aus der Seite, sonst leer.",
    )
    sicherheit: Literal["hoch", "mittel", "niedrig"] = Field(
        ..., description="hoch = Beschriftung eindeutig neben dem Feld; mittel = aus Abschnitt/Umfeld erschlossen; "
                         "niedrig = kein Beleg, Quickinfo ist eine Vermutung.",
    )
    hinweis: str = Field(
        "", max_length=200,
        description="Kurzer Hinweis fuer den Bearbeiter, z. B. 'Format aus Legende uebernommen' oder 'keine Beschriftung in der Naehe'. Sonst leer.",
    )


class QuickinfoSeiteOutput(BaseModel):
    """Alle angefragten Felder einer Seite."""
    felder: list[QuickinfoFeldOutput] = Field(..., min_length=1, description="Genau ein Eintrag je angefragtem Feld.")
