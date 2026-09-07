"""Pass-2-Prompt-Builder: Inventar (forensische Bildanalyse).

Universal für die 8 Inventar-pflichtigen Bildtypen:
foto, illustration, diagramm, tabelle, karte, screenshot, infografik, strukturformel.

Die anderen 4 Bildtypen (logo, icon, funktional, dekorativ) überspringen
den Inventar-Pass — siehe pipelines/v4/orchestrator.py (Phase T9).

Fassung September 2026: Die Schwerpunkte je Bildtyp werden auch vom Combo-Prompt
(combo.py) als Anleitung für das innere Inventar verwendet.
"""
from __future__ import annotations

from typing import Optional

from prompts.components.constraints import ANTI_HALLUZINATION_REGELN
from prompts.components.roles import ROLE_INVENTARISIERER
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import BildtypTopLevel, InventarOutput

from .helpers import bildgroesse_zeile, kontext_werte, user_hint_block


# Bildtyp-spezifische Schwerpunkte für den Inventar-Pass.
# Werden in den Prompt eingespiegelt — ein Schwerpunkt pro Bildtyp,
# damit das Modell weiß WORAUF zu fokussieren ist.
BILDTYP_INVENTAR_SCHWERPUNKTE: dict[str, str] = {
    'foto': """Schwerpunkt Foto: Jede Person einzeln mit Position, Haltung und dem, was sie in
den Händen hält. Personen und Objekte von links nach rechts zählen, auch verdeckte,
angeschnittene und Rückenansichten. Lesbare Texte wortgetreu erfassen (Schilder,
Schriftzüge, Kennzeichen, Namensschilder, Logos). Umgebung benennen: innen oder
außen, Möbel, Geräte, Bühne, Catering.""",

    'illustration': """Schwerpunkt Illustration: Stil (Cartoon, Vektor, gemalt), dargestellte Idee, alle
Text-Elemente wortgetreu, Symbole und Siegel als sichtbare Elemente. Tierart oder
Personentyp nur bei klarer Erkennbarkeit, sonst beide Deutungen.""",

    'diagramm': """Schwerpunkt Diagramm: Diagrammtyp, Titel, Achsen mit Einheit, Legende, Kategorien
und Reihen. Werte einzeln an der Achse ablesen und als Liste notieren, bevor du
einen Trend formulierst. Ohne lesbare Skala nur Rangfolge und Form.""",

    'tabelle': """Schwerpunkt Tabelle: alle Spaltenköpfe wortgetreu, je Zeile die Bezeichnung und alle
Werte, Summenzeilen mit ihrer Beschriftung.""",

    'karte': """Schwerpunkt Karte: Gebiet, Kartenthema, alle markierten Orte mit Beschriftung,
Legende, Maßstab oder Zeitangabe.""",

    'infografik': """Schwerpunkt Infografik: Stationen oder Abschnitte in ihrer Reihenfolge,
Verbindungen (Pfeile, Linien) mit ihrer Bedeutung, alle Zahlen und Beschriftungen
wortgetreu.""",

    'screenshot': """Schwerpunkt Screenshot: Anwendung oder Website (Fenstertitel, Adresszeile, Logo),
gezeigter Zustand, Statusmeldungen, Werte, die wichtigste sichtbare Aktion, dann
Menüs, Eingabefelder und Schaltflächen.""",

    'strukturformel': """Schwerpunkt Strukturformel: Beschriftung und Stoffname, Atome und funktionelle
Gruppen, Bindungstypen, bei Reaktionen Edukte, Bedingungen und Produkte.""",
}


def build_inventar_prompt(
    bildtyp: BildtypTopLevel,
    enriched_context: str,
    width: int,
    height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Pass-2-Prompt: Inventar des Bildes — Was ist sichtbar?

    Inputs:
      bildtyp:          Top-Level-Typ aus Pass 1 (Klassifikation).
                        Bestimmt den BILDTYP_INVENTAR_SCHWERPUNKTE-Block.
      enriched_context: Web-/PDF-Kontext.
      width, height:    Bildmaße.
      user_hint:        Optionaler Nutzer-Hinweis (Workflow-Variante 3).

    Output: prompt-String, der mit InventarOutput-Schema gerufen wird.
    """
    schema_doc = render_schema_for_prompt(InventarOutput)
    bildtyp_hinweis = BILDTYP_INVENTAR_SCHWERPUNKTE.get(bildtyp, '')

    return f"""{ROLE_INVENTARISIERER}

{ANTI_HALLUZINATION_REGELN}

BILDTYP: {bildtyp}
{bildgroesse_zeile(width, height, label='BILDGRÖSSE')}
{bildtyp_hinweis}

KONTEXT
{kontext_werte(enriched_context, user_hint_block(user_hint))}

AUFTRAG
Erstelle ein vollständiges Inventar dieses Bildes. Trage in halluzinations_warnung
die Fehldeutungen ein, die bei diesem Bild naheliegen (helle Innenfläche als
Inhalt, stilisiertes Tier als bestimmte Art, kleine runde Gegenstände als bestimmte
Funktion). Erkennst du Montage-Hinweise, notiere sie dort ebenfalls und liste das
eingefügte Element als eigenes Objekt.

{schema_doc}
"""
