"""Combo-Builder fuer Lean-Mode (1-Pass-Sonnet-Architektur).

Statt zwei separater Paesse (Inventar + Beschreibung) wird im Lean-Mode ein
einziger Sonnet-Aufruf gemacht, der intern beide Aufgaben erledigt:
  1. Bild forensisch inventarisieren (intern, nicht als Output)
  2. Auf Basis dieses Inventars die Beschreibung generieren (Output)

Dieser Builder kombiniert die bestehenden Pass-2- und Pass-3-Prompts mit
einem klar strukturierten Übergang. Sonnet wird angewiesen, das Inventar
"im Kopf" zu erstellen und nur das BeschreibungOutput-Schema zurueckzugeben.

Genutzt von pipelines.v4.orchestrator._run_lean_pipeline().

WICHTIG: Multi-Pass-Code (build_inventar_prompt + build_beschreibung_prompt_*)
bleibt unveraendert. Dieser Builder ist additiv.
"""
from __future__ import annotations

from typing import Optional

from prompts.builders.beschreibung import build_beschreibung_prompt_with_inventar
from prompts.builders.inventar import build_inventar_prompt
from prompts.components.schemas import (
    BildtypEffective,
    BildtypTopLevel,
    InventarOutput,
)


def build_combined_inventar_beschreibung_prompt(
    bildtyp_top: BildtypTopLevel,
    bildtyp_effective: BildtypEffective,
    enriched_context: str,
    width: int,
    height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Kombinierter Inventar+Beschreibung-Prompt fuer einen einzigen Sonnet-Aufruf.

    Args:
      bildtyp_top: Top-Level-Bildtyp (foto/diagramm/...) fuer Inventar-Builder.
      bildtyp_effective: Effektiver Typ inkl. Sub-Typ (foto_event/foto_objekte/
                         diagramm/...) fuer Beschreibungs-Builder mit passendem
                         Premium-Material.
      enriched_context: PDF-/Web-Kontext (Titel, umliegender Text).
      width, height: Bildmaße in Pixel.
      original_alt: vom Autor gesetzter alt-Text (oft leer).
      user_hint: Workflow-Variante 3, optional.

    Returns:
      Combo-Prompt-String. Schema fuer den Sonnet-Aufruf: BeschreibungOutput
      (NICHT InventarOutput) — Inventar entsteht intern.
    """
    inventar_prompt = build_inventar_prompt(
        bildtyp=bildtyp_top,
        enriched_context=enriched_context,
        width=width,
        height=height,
        user_hint=user_hint,
    )

    # Dummy-Inventar damit der Beschreibungs-Builder seine strukturellen
    # Sektionen rendern kann. Sonnet wird angewiesen, das Dummy zu ignorieren
    # und stattdessen das selbst erstellte interne Inventar zu nutzen.
    # foto_subtyp nur setzen wenn bildtyp_effective wirklich ein Sub-Typ ist —
    # 'foto' (Top-Level) ist KEIN gueltiger foto_subtyp und wuerde Pydantic-
    # Validation brechen.
    from typing import get_args
    from prompts.components.schemas import FotoSubtyp
    foto_sub_values = get_args(FotoSubtyp)
    foto_subtyp_for_dummy = bildtyp_effective if bildtyp_effective in foto_sub_values else None
    dummy_inventar = InventarOutput(foto_subtyp=foto_subtyp_for_dummy)
    beschreibung_prompt = build_beschreibung_prompt_with_inventar(
        bildtyp=bildtyp_effective,
        inventar=dummy_inventar,
        enriched_context=enriched_context,
        width=width,
        height=height,
        user_hint=user_hint,
    )

    # Klare Struktur: erst Inventar-Anweisungen, dann Übergangstext, dann
    # Beschreibungs-Anweisungen. Sonnet erledigt beides intern in einem Schritt.
    return f"""DU ERLEDIGST ZWEI AUFGABEN IN EINEM AUFRUF:

== AUFGABE 1: INTERNES INVENTAR ==
Folgende forensische Bildanalyse erledigst Du IM KOPF — sie wird NICHT in
deinem Output stehen, dient aber als Grundlage fuer Aufgabe 2:

{inventar_prompt}

== AUFGABE 2: BESCHREIBUNG (DEIN OUTPUT) ==
Auf Basis Deines internen Inventars erzeugst Du jetzt die Beschreibung
gemaess folgender Vorgaben. Der "Inventar"-JSON-Block weiter unten ist
ein Schema-Platzhalter, ignoriere ihn — nutze stattdessen das Inventar
das Du gerade im Kopf erstellt hast.

{beschreibung_prompt}

== WICHTIG ==
Dein Output muss exakt dem BeschreibungOutput-Schema folgen — also Alt-Text,
Langbeschreibung, verwendete_inventar_items, atmosphaere_belege, nicht_im_inventar.
Liefere KEINEN separaten Inventar-Block im Output. Das Inventar bleibt intern.
"""
