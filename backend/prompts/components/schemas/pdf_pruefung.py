"""Output-Schema der automatischen Pruefung getaggter PDFs (Schritt 5, erste Fassung, 22.09.2026).

Ein Aufruf je Seite: das Modell sieht das Seitenbild und die Strukturliste der Seite (Kennung,
Rolle, Text je Tag) und liefert NUR Urteile — Befunde, bei denen Tags und sichtbare Seite nicht
zusammenpassen. Es aendert nichts; die Korrektur ueber PDFix-Befehle ist eine spaetere Stufe und
wird nur bei hoher Sicherheit ausgefuehrt (Steve 22.09.2026: „nicht, dass die KI irgendwas falsch
macht, was eigentlich richtig wäre“).

Die Nachpruefung in pdf_pruefung.py verlaesst sich nicht auf die Selbsteinschaetzung: Kennungen,
die es auf der Seite nicht gibt, werden auf „niedrig“ gesetzt; Doppelmeldungen fallen weg.
"""
from typing import Literal

from pydantic import BaseModel, Field


class PruefBefund(BaseModel):
    """Ein Befund auf der Seite."""
    element: str = Field(
        "", max_length=40,
        description="Kennung des betroffenen Elements EXAKT wie in der STRUKTURLISTE (z. B. E0.3.1). "
                    "Leer nur, wenn der Befund die ganze Seite betrifft (z. B. sichtbarer Text ohne Tag).",
    )
    art: Literal["rolle", "ebene", "reihenfolge", "tabelle", "grafik", "fehlt", "sprache", "sonstiges"] = Field(
        ..., description="rolle = falsche Rolle (z. B. Überschrift als Listenpunkt); ebene = falsche Überschriftenebene; "
                         "reihenfolge = Lesereihenfolge weicht von der sichtbaren Ordnung ab; tabelle = Kopfzeile/Zellen falsch; "
                         "grafik = Alt-Text passt nicht zum Bild oder Bild ohne Tag; fehlt = sichtbarer Inhalt ohne Tag; "
                         "sprache = Sprache passt nicht; sonstiges.",
    )
    befund: str = Field(..., min_length=5, max_length=300, description="Was nicht zusammenpasst, in einem Satz, in der Sprache des Dokuments.")
    vorschlag: str = Field(
        "", max_length=120,
        description="Die richtige Rolle als Tag-Name (H1, H2, H3, H4, P, L, LI, Table, TH, TD, Figure, Artifact) "
                    "oder ein kurzer Alt-Text-Vorschlag bei art=grafik; sonst leer.",
    )
    beleg: str = Field("", max_length=300, description="Was auf der Seite sichtbar ist und den Befund stützt (Schriftgröße, Fettdruck, Lage, Text).")
    sicherheit: Literal["hoch", "mittel", "niedrig"] = Field(
        ..., description="hoch = Seitenbild belegt es eindeutig und die Regel ist eindeutig; mittel = wahrscheinlich, "
                         "aber eine andere Deutung ist vertretbar; niedrig = Vermutung.",
    )


class PruefSeiteOutput(BaseModel):
    """Alle Befunde einer Seite; leer, wenn Tags und Seite zusammenpassen."""
    befunde: list[PruefBefund] = Field(default_factory=list, description="Nur echte Befunde. Keine Befunde = leere Liste.")
    zusammenfassung: str = Field("", max_length=300, description="Ein Satz zur Seite, in der Sprache des Dokuments.")
