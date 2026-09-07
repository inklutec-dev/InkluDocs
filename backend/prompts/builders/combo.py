"""Combo-Builder: der Produktionsprompt der v4-Pipeline (ein Aufruf, zwei Schritte).

Zusammenbau v2 (07.09.2026). Vorher wurden der komplette Inventar-Prompt
(Pass 2) und der komplette Beschreibungs-Prompt (Pass 3) aneinandergehaengt —
mit doppelter Rolle, doppelter Anti-Halluzinations-Schicht, einer Anweisung
fuer das falsche Ausgabeschema, einem leeren Inventar-JSON und doppeltem
Kontext. Jetzt gibt es EINEN Bauplan:

  1. ROLE_BESCHREIBER (System-Prompt dazu: SYSTEM_BESCHREIBUNG im Orchestrator)
  2. ANTI_HALLUZINATION_REGELN — genau einmal
  3. ARBEITSWEISE — zwei Schritte in einem Aufruf (internes Inventar, dann Output)
  4. der Kategorie-Teil des jeweiligen Builders (Foto-, Daten-Familie), gerendert
     im Combo-Modus: ohne eigenen Kopf, an der Inventar-Stelle steht der Text fuer
     das interne Inventar (Schwerpunkte je Bildtyp, Halluzinationsfallen,
     Montage-Check), Kontext und Bilddaten genau einmal.

Die Regeltexte der Builder bleiben unveraendert; nur der Zusammenbau ist neu.
Die Mini-Familie (logo/icon/funktional) laeuft nicht ueber diesen Builder.
Genutzt von pipelines.v4.orchestrator._run_lean_pipeline().
"""
from __future__ import annotations

from typing import Optional, get_args

from prompts.builders.beschreibung import build_beschreibung_prompt_with_inventar
from prompts.builders.helpers import combo_modus
from prompts.builders.inventar import BILDTYP_INVENTAR_SCHWERPUNKTE
from prompts.components.constraints import ANTI_HALLUZINATION_REGELN
from prompts.components.roles import ROLE_BESCHREIBER
from prompts.components.schemas import (
    BildtypEffective,
    BildtypTopLevel,
    FotoSubtyp,
    InventarOutput,
)


ARBEITSWEISE = """ARBEITSWEISE

Du erledigst zwei Schritte in einem Aufruf.

Schritt 1, inneres Inventar: Bevor du schreibst, erfasst du das Bild vollständig:
Objekte, Personen, lesbare Texte, Umgebung, Form, Farbe, Position, Anzahl. Dieses
Inventar erscheint nicht in der Ausgabe. Es ist die Grundlage für jede Aussage in
Schritt 2.

Schritt 2, Text: Aus dem Inventar schreibst du Alt-Text und Langbeschreibung nach
den folgenden Vorgaben."""


def _internes_inventar_text(bildtyp_top: BildtypTopLevel) -> str:
    """Der Text, der im Kategorie-Teil an der Stelle des Inventars steht."""
    schwerpunkte = BILDTYP_INVENTAR_SCHWERPUNKTE.get(bildtyp_top, '').strip()
    teile = ['DEIN INNERES INVENTAR (Schritt 1)', '']
    if schwerpunkte:
        teile += [schwerpunkte, '']
    if bildtyp_top in ('foto', 'illustration'):
        teile += ["""Benenne dir selbst, welche Fehldeutungen bei diesem Bild naheliegen (helle
Innenfläche als Inhalt, stilisiertes Tier als bestimmte Art, kleine runde
Gegenstände als bestimmte Funktion), und meide sie. Prüfe das Bild Viertel für
Viertel auf Montage-Hinweise (Belegregel 5)."""]
    else:
        teile += ["""Benenne dir selbst, welche Fehldeutungen bei diesem Bild naheliegen (eine Zahl
der falschen Spalte oder Reihe zugeordnet, eine Farbe nach Alltagsbedeutung statt
nach Legende gelesen, ein Beispieltext als Datenangabe), und meide sie."""]
    return '\n'.join(teile)


def build_combined_inventar_beschreibung_prompt(
    bildtyp_top: BildtypTopLevel,
    bildtyp_effective: BildtypEffective,
    enriched_context: str,
    width: int,
    height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Der Produktionsprompt: Kopf + Arbeitsweise + Kategorie-Teil im Combo-Modus.

    Args:
      bildtyp_top: Top-Level-Bildtyp (foto/diagramm/...) fuer die Inventar-Schwerpunkte.
      bildtyp_effective: Effektiver Typ inkl. Sub-Typ (foto_event/diagramm/...) fuer
                         den Kategorie-Builder.
      enriched_context: PDF-/Web-Kontext (Titel, umliegender Text).
      width, height: Bildmasse in Pixel.
      original_alt: vom Autor gesetzter alt-Text (oft leer; im Combo nicht verwendet).
      user_hint: Nutzer-Hinweis (Workflow-Variante 3), optional.

    Returns:
      Prompt-String. Schema fuer den Aufruf: BeschreibungOutput.
    """
    # Die Builder-Signatur verlangt ein InventarOutput; im Combo-Modus wird es
    # nicht gerendert (die Inventar-Stelle traegt den Text fuer Schritt 1).
    foto_sub_values = get_args(FotoSubtyp)
    platzhalter = InventarOutput(foto_subtyp=bildtyp_effective if bildtyp_effective in foto_sub_values else None)
    with combo_modus(_internes_inventar_text(bildtyp_top)):
        kategorie_teil = build_beschreibung_prompt_with_inventar(
            bildtyp=bildtyp_effective,
            inventar=platzhalter,
            enriched_context=enriched_context,
            width=width,
            height=height,
            user_hint=user_hint,
        )
    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{ARBEITSWEISE}

{kategorie_teil.strip()}
"""
