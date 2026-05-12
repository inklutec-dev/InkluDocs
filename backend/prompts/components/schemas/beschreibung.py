"""Output-Schema für Pass 3 (Beschreibung) der v4-Pipeline."""
from pydantic import BaseModel, Field


class AtmosphaereBeleg(BaseModel):
    """W4-Fix: eine Wertung/Atmosphäre-Aussage mit ihrem visuellen Beleg.

    Wird vom Beschreibungs-Pass befüllt; vom Validierungs-Pass geprüft
    (referenziert das Inventar — Belege müssen dort wiederzufinden sein).

    Davor: list[dict] (untyped) — Strict Mode konnte die innere Struktur
    nicht erzwingen, Modell hätte beliebige dict-Schemas zurückgeben können.

    Re-Review-Mini-Korrektur b: Untergrenzen verschärft.
    - wertung min_length=4 (vorher 2): blockt 'ja'/'ok', erlaubt 'ernst'/'ruhig'.
    - beleg min_length=20 (vorher 5): zwingt echte Belege wie
      'aufrechte Haltung und formelle Kleidung' statt 'ja ok'.
    Eval-Beobachtung: wenn legitime Kurz-Belege (z.B. 'Hängende Schultern' = 18)
    geblockt werden, ggf. auf 15 senken.
    """

    wertung: str = Field(
        ..., min_length=4, max_length=100,
        description="Die Wertung selbst — z.B. 'professionell', 'konzentriert', 'ruhig'.",
    )
    beleg: str = Field(
        ..., min_length=20, max_length=300,
        description=(
            "Visueller Beleg aus dem Inventar — z.B. "
            "'aufrechte Haltung und formelle Kleidung', "
            "'alle blicken zur Präsentation, niemand spricht'."
        ),
    )


class BeschreibungOutput(BaseModel):
    """Output von Pass 3 (Beschreibung) für Standard-Bildtypen mit Inventar-Pass."""

    alt_text: str = Field(
        ..., min_length=20, max_length=400,
        description=(
            "Kernaussage. Erste Information bild-spezifisch "
            "(siehe SPEZIFITAETS_PFLICHT)."
        ),
    )

    langbeschreibung: str = Field(
        "", max_length=2000,
        description="Vertiefung. Leer wenn alt_text alles wesentliche sagt.",
    )

    verwendete_inventar_items: list[str] = Field(
        ...,
        description="Welche Inventar-Items wurden im Output verwendet? Audit-Trail.",
    )

    nicht_verwendete_inventar_items: list[str] = Field(
        default_factory=list,
        description="Welche bewusst weggelassen, weil unwichtig? (Kein Fehler.)",
    )

    nicht_im_inventar: list[str] = Field(
        default_factory=list,
        description=(
            "MUSS LEER SEIN. Wenn Items im Output stehen die nicht im Inventar sind, "
            "hier auflisten — Pipeline schlägt dann Alarm. Halluzinations-Self-Check."
        ),
    )

    atmosphaere_belege: list[AtmosphaereBeleg] = Field(
        default_factory=list,
        description=(
            "Bei evidenzbasierten Wertungen: jede Wertung mit explizitem visuellem "
            "Beleg. Siehe AtmosphaereBeleg-Submodel."
        ),
    )


class IconBeschreibungOutput(BaseModel):
    """K1-Fix: Output von Pass 3 für Mini-Pipelines (icon, funktional).

    Re-Review-Mini-Korrektur a: dekorativ NICHT mehr in diesem Schema.
    Dekorative Bilder bekommen einen direkten Dict-Return aus
    handle_dekorativ_classification() — siehe §7.15. Das vermeidet,
    dass das Schema mit min_length=0 die Schema-Strenge für icon/funktional
    verliert.

    Schema-Eigenschaften:
    - icon: alt_text 3-50 Zeichen (Funktionsbenennung), KEIN langbeschreibung
    - funktional: alt_text 3-80 Zeichen (Funktion + Zustand), KEIN langbeschreibung

    BeschreibungOutput passt nicht — dessen min_length=20 bricht
    'Suche öffnen' (12 Zeichen) und 'Hauptmenü öffnen' (16); es zwingt
    auch atmosphaere_belege/inventar-Items ein, die für funktionale
    Mini-Pipelines konzeptionell falsch sind.
    """

    alt_text: str = Field(
        ..., min_length=3, max_length=80,
        description=(
            "Funktion (icon: 3-50 Zeichen, funktional: 3-80 Zeichen). "
            "Validierung der Bildtyp-spezifischen Obergrenze erfolgt in der "
            "jeweiligen Mini-Pipeline."
        ),
    )

    verwendete_inventar_items: list[str] = Field(
        default_factory=list,
        description="Audit-Trail. Bei Mini-Pipelines meist leer (kein Inventar-Pass).",
    )

    # Bewusst KEIN langbeschreibung-Feld, KEIN nicht_im_inventar,
    # KEIN atmosphaere_belege — Mini-Pipelines liefern ausschließlich
    # kurze Funktionsbeschreibungen.
