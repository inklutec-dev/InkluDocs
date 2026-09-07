"""Pass-2-Prompt-Builder: Inventar (forensische Bildanalyse).

Universal für die 8 Inventar-pflichtigen Bildtypen:
foto, illustration, diagramm, tabelle, karte, screenshot, infografik, strukturformel.

Die anderen 4 Bildtypen (logo, icon, funktional, dekorativ) überspringen
den Inventar-Pass — siehe pipelines/v4/orchestrator.py (Phase T9).

W1-Korrektur (Steve, 04.05.2026): Sub-Typ-Entscheidungs-Kriterien für foto
explizit im Prompt — vorher 'setze foto_subtyp anhand des Inventars' ohne
Detail. Schwelle ≥2 Personen + Indikator (Re-Review-Mini-Korrektur c).
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
    'foto': """SCHWERPUNKT FOTO:
- Wenn Personen sichtbar: für jede Person separat Position, Haltung, was sie in den Händen hält
- PERSONEN-ZAEHLUNG: Zähle Personen einzeln und systematisch von links nach rechts.
  Auch teilweise verdeckte Personen, Personen im Hintergrund, Rückenansichten und
  angeschnittene Personen zählen, wenn Körper, Kopf, Kleidung oder Haltung eindeutig
  auf eine Person hinweisen. Bei Unsicherheit: die niedrigere SICHERE Zahl nehmen und
  im Text 'mindestens sieben Personen' oder 'acht in einer Reihe, dahinter weitere'
  formulieren statt einer falschen exakten Zahl.
- Lesbare Texte wortgetreu erfassen: Schilder, Schriftzüge, Kennzeichen, Namensschilder, Logos
- Setting-Indikatoren benennen: Innen/Außen, Möbel, Geräte, Schilder, Catering, Bühne""",

    'illustration': """SCHWERPUNKT ILLUSTRATION:
- Stilrichtung benennen (Cartoon, Vektor, gemalt, comic-haft etc.)
- Bei Tieren/Personen: Spezies-Identifikation NUR mit hoher Sicherheit, sonst
  Mehrfach-Hypothesen ('katzenartig oder hundeartig')
- Halluzinations-Warnung explizit für stilisierte Darstellungen formulieren""",

    'diagramm': """SCHWERPUNKT DIAGRAMM:
- Diagrammtyp (Balken, Linie, Kreis, gestapelt, Streu), Titel, Achsen, Einheiten, Legende
- WERTE ZUERST, TREND DANACH: Lies für JEDE Kategorie und JEDE Reihe die Werte
  einzeln an der Achse ab und notiere sie dir als Liste (zum Beispiel
  "Hardware: 2021 2,5 / 2022 4,4 / 2023 2,0"). Erst aus dieser Liste leitest du
  Trends, Vergleiche und Extreme ab — nie aus dem Gesamteindruck. Ein Trendwort
  (steigt, fällt, erholt sich) ist nur erlaubt, wenn die notierten Werte es tragen.
- Wenn keine Werte lesbar sind (keine Achse, keine Zahlen): nur Rangfolge und
  Form beschreiben, keine Zahlen erfinden
- ALLE Achsenbeschriftungen, Legende, Datenpunkte als lesbare Texte erfassen""",

    'tabelle': """SCHWERPUNKT TABELLE:
- ALLE Spaltenköpfe wortgetreu erfassen
- Pro Zeile: erste Spalte (meist Bezeichnung) + alle Wertspalten
- Summen/Bilanzsummen explizit kennzeichnen""",

    'karte': """SCHWERPUNKT KARTE:
- Geografisches Gebiet
- ALLE markierten Standorte mit ihren Beschriftungen
- Legenden-Einträge""",

    'infografik': """SCHWERPUNKT INFOGRAFIK:
- Stationen/Schritte in logischer Reihenfolge
- Beziehungen (Pfeile, Verbindungen)
- Zentrale Datenpunkte""",

    'screenshot': """SCHWERPUNKT SCREENSHOT:
- UI-Anwendung identifizieren wenn möglich (URL-Leiste, Fenstertitel, Logo)
- Sichtbare Menüpunkte, Buttons, Eingabefelder
- Status-Anzeigen, Statusmeldungen""",

    'strukturformel': """SCHWERPUNKT STRUKTURFORMEL:
- Atom-Symbole, funktionelle Gruppen
- Bindungstypen
- Falls Reaktionsgleichung: Edukte → Bedingungen → Produkte""",
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

KONTEXT (vom Web-Scraper, PDF-Extraktion oder API-Aufruf):
{kontext_werte(enriched_context, user_hint_block(user_hint))}

DEINE AUFGABE:
Erstelle ein vollständiges, ehrliches Inventar dieses Bildes. Fülle JEDES Feld
des Schemas aus, auch wenn leer ([] oder None). Das ist eine bewusste Entscheidung,
nicht Vergesslichkeit.

WICHTIG für halluzinations_warnung:
Identifiziere KONKRETE Fehlinterpretationen die für DIESES Bild wahrscheinlich wären.
Beispiele:
- 'Hellfarbene Glasur könnte als Flüssigkeit fehlinterpretiert werden'
- 'Stilisierte Tierdarstellung — Spezies-Festlegung wäre Spekulation'
- 'Personen halten kleine runde Objekte — diese sind nicht eindeutig identifizierbar'

MONTAGE-CHECK:
Achte auf Montage-Indikatoren: harte Freisteller-Kanten, widersprüchliche
Schatten/Perspektive/Maßstäbe, Stilbruch zwischen Foto und Grafik, unmögliche
Kombinationen. SUCHE DABEI AKTIV, Quadrant für Quadrant, auch nach KLEINEN
eingefügten Objekten — ein winziges Bauwerk oder Objekt an einem Ort, an den
es nicht gehört (z.B. eine Kathedrale am Grund einer Schlucht), ist ein
Montage-Beweis; geringe Größe schützt eine Montage nicht vor der Erkennung.
Erkennst du solche Indikatoren, trage einen Eintrag in
halluzinations_warnung ein (z.B. 'Montage-Indikatoren sichtbar: harte
Freisteller-Kante am Gebäude — Bild ist vermutlich eine Fotomontage, nicht als
reales Foto beschreiben') und liste das eingefügte Objekt als eigenes Objekt.

{schema_doc}
"""
