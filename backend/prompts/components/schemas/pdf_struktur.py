"""Output-Schema der Struktur-Zuordnung (Tagging-Weg „Struktur zuerst“, 23.09.2026, Steves Go).

Ein Aufruf je Seite: das Modell sieht das Seitenbild und das STRUKTUR-HTML der Seite (jede Textzeile
mit Kennung, Schriftgroesse, Fettdruck und Lage; jedes Bild mit Kennung) und ORDNET NUR ZU: welche
Zeilen Ueberschriften, Artefakte oder Bildunterschriften sind, welche Bilder Inhalt tragen. Es schreibt
keinen Text und aendert keinen Text. Die Ueberschriften-EBENE des Modells ist nur seitenlokal; die
dokumentweite Ebene bestimmt pdf_struktur_tagging.stilprofil aus Schriftgroesse und Lesereihenfolge.

Die Nachpruefung verlaesst sich nicht auf die Selbsteinschaetzung: unbekannte Kennungen fallen weg,
vergessene Bilder gelten als inhaltlich (lieber ein Bild zu viel als ein verlorenes).
"""
from typing import Literal

from pydantic import BaseModel, Field


class StrukturZeile(BaseModel):
    """Eine Zeile, die KEIN normaler Absatz ist."""
    id: str = Field(..., max_length=24, description="Kennung der Zeile EXAKT aus dem STRUKTUR-HTML (z. B. s3z4).")
    rolle: Literal["H1", "H2", "H3", "H4", "H5", "H6", "Artefakt", "Caption"] = Field(
        ..., description="H1–H6 = Überschrift dieser Ebene, so wie die Seite sie sichtbar zeigt (größte Überschrift der Seite = "
                         "höchste Ebene, die auf dieser Seite vorkommt); Artefakt = Kolumnentitel (auf jeder Seite wiederholter "
                         "Buch- oder Kapiteltitel), Seitenzahl, Verlags- oder Fußzeile, Schmucktext ohne Informationswert; "
                         "Caption = Bildunterschrift. Normale Absätze und Listenpunkte NICHT nennen, sie bleiben Text.")
    beleg: str = Field("", max_length=200, description="Was auf der Seite sichtbar ist (Größe, Fettdruck, Lage), in einem Halbsatz.")


class StrukturBild(BaseModel):
    """Ein Bild aus dem STRUKTUR-HTML."""
    id: str = Field(..., max_length=24, description="Kennung des Bildes EXAKT aus dem STRUKTUR-HTML (z. B. s3b1).")
    inhaltlich: bool = Field(..., description="true = das Bild trägt Inhalt (Illustration mit Motiv, Foto, Diagramm, Vorlage zum Ausschneiden, "
                                              "Logo mit Text); false = reiner Schmuck ohne Aussage (Rahmen, Hintergrund, Linie, Zierleiste). "
                                              "Im Zweifel true.")
    alt: str = Field("", max_length=300, description="Kurzer Alt-Text-Vorschlag für inhaltliche Bilder in der Sprache des Dokuments; leer bei Schmuck.")


class StrukturSeiteOutput(BaseModel):
    """Zuordnung einer Seite."""
    zeilen: list[StrukturZeile] = Field(default_factory=list, description="Nur Zeilen, die Überschrift, Artefakt oder Bildunterschrift sind.")
    bilder: list[StrukturBild] = Field(default_factory=list, description="Jedes <img> aus dem STRUKTUR-HTML genau einmal.")
    hat_tabelle: bool = Field(..., description="true, wenn die Seite sichtbar eine Tabelle enthält: ein Raster aus Zeilen und Spalten ODER "
                                               "Beschriftung-Wert-Paare in zwei Spalten (wie ein Impressum). Aufzählungen und Text in "
                                               "Kästen sind KEINE Tabelle.")
