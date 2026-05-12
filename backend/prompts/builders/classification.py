"""Pass-1-Prompt-Builder: Klassifikator (Bildtyp-Wahl + Begründung).

Der Klassifikator entscheidet anhand des Bildes + Kontext + Original-Alt
welcher Top-Level-Bildtyp vorliegt. Sub-Typen für foto entscheidet
später der Inventar-Pass (siehe inventar.py / BildtypEffective).

W3-Fix relevant: original_alt_brauchbar wird HIER vom Klassifikator
gesetzt — der Prompt sagt ihm explizit, wann True/False zu wählen ist.
Ohne diese Anweisung wäre das Schema-Feld funktional leer.
"""
from __future__ import annotations

import os

from typing import Optional

from prompts.components.roles import ROLE_KLASSIFIKATOR
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import ClassificationOutput

from .helpers import user_hint_block


# Top-Level-Bildtypen (12) als nummerierte Liste für den Prompt — synchron
# mit BildtypTopLevel-Literal in classification.py des Schema-Pakets.
_BILDTYP_LISTE_BASIS = """Die 12 möglichen Top-Level-Bildtypen:

1. foto         — Echtes Fotografie-Bild (drinnen, draußen, Personen, Objekte etc.)
2. illustration — Cartoon, Vektor-Grafik, gemalte Illustration, Buch-Bild
3. diagramm     — Balken-, Linien-, Kreis-, gestapeltes Diagramm
4. tabelle      — Tabellarische Daten als Grafik
5. karte        — Landkarte, Stadtplan, Lageplan, Übersichtskarte
6. infografik   — Schaubild, Übersichtsgrafik mit Stationen oder Schritten
7. screenshot   — Bildschirmfoto einer Anwendung, Webseite oder UI
8. strukturformel — Chemische Struktur-, Reaktions- oder Summenformel
9. logo         — Erkennbares Marken-, Organisations- oder Lizenzlogo
10. icon        — Kleines funktionales Symbol (Lupe, Hamburger, Warenkorb etc.)
11. funktional  — Navigations-/Steuerungselement mit Zustand
                  (Paginierungspfeile, Vor/Zurück, Fortschrittsanzeige, Breadcrumb)
12. dekorativ   — Rein schmückendes Bild ohne Information (Trennlinie,
                  Hintergrund, Schmuckelement). Bekommt leeren Alt-Text."""

# Hinweis nur im Multi-Pass-Modus (V4_PASS_MODE=full): dort delegiert der
# Klassifikator die Sub-Typ-Wahl an einen späteren Inventar-Pass. Im Lean-Modus
# gibt es diesen Pass nicht mehr — der Klassifikator entscheidet Sub-Typen mit
# (siehe _foto_subtyp_anweisung), und dieser Hinweis wäre dann irreführend.
_BILDTYP_LISTE_MULTIPASS_HINWEIS = """

Sub-Typen für foto (foto_personen, foto_event etc.) werden NICHT hier
entschieden — das macht später der Inventar-Pass besser."""


def _bildtyp_liste() -> str:
    """Liefert die Bildtyp-Liste passend zum aktiven V4_PASS_MODE."""
    if os.environ.get('V4_PASS_MODE', 'full').lower() == 'lean':
        return _BILDTYP_LISTE_BASIS
    return _BILDTYP_LISTE_BASIS + _BILDTYP_LISTE_MULTIPASS_HINWEIS


def _subtyp_aufgaben_hinweis() -> str:
    """Eine Zeile im 'WICHTIG'-Block, modus-abhängig.

    Im Multi-Pass: Klassifikator soll Sub-Typ NICHT wählen.
    Im Lean-Mode:  Klassifikator soll Sub-Typ HIER mitwählen
    (Details siehe _foto_subtyp_anweisung()-Block weiter unten im Prompt).
    """
    if os.environ.get('V4_PASS_MODE', 'full').lower() == 'lean':
        return "- Sub-Typ für foto wird HIER mitgewählt (siehe Sub-Typ-Block unten)."
    return "- Sub-Typen für foto NICHT hier entscheiden — wähle einfach 'foto'."




def _foto_subtyp_anweisung() -> str:
    """Lean-Mode-spezifische Anweisung: Klassifikator entscheidet bei foto auch Sub-Typ.

    Greift nur wenn V4_PASS_MODE=lean. Im Multi-Pass-Modus bleibt der Sub-Typ
    leer und wird vom Inventar-Pass spaeter entschieden.
    """
    if os.environ.get('V4_PASS_MODE', 'full').lower() != 'lean':
        return ''
    return """

ZUSATZ FÜR LEAN-MODE — foto_subtyp ist PFLICHTFELD bei bildtyp=foto:

WICHTIG: Wenn du bildtyp='foto' wählst, MUSST du foto_subtyp setzen. Output ohne
foto_subtyp bei bildtyp=foto ist UNGÜLTIG und verursacht Pipeline-Fehler.

Wähle exakt EINEN dieser sechs Werte:

- foto_personen    — eine oder mehrere Personen im Mittelpunkt
                     (Porträt, Gruppenfoto, Einzelperson, Pressefoto)
- foto_event       — mehrere Personen + Event-Setting (Workshop, Schulung,
                     Konferenz, Meeting, Feier, Tagung, Seminar)
                     Indikatoren: Bühne, Beamer, Namensschilder, Catering,
                     erkennbare Veranstaltungssituation
- foto_objekte     — Gegenstände im Mittelpunkt
                     (Produkte, Werkstücke, Stillleben, Materialfotos,
                     Werkstattfotos)
- foto_landschaft  — Naturaufnahme, Panorama, Outdoor-Szene OHNE Personen-Fokus
- foto_architektur — Gebäude, Baudetails, Stadtaufnahmen, Innenraum-Architektur
- foto_essen       — Speisen, Gerichte, Lebensmittel im Mittelpunkt

ENTSCHEIDUNGSLOGIK (in dieser Reihenfolge prüfen):
1. Mehrere Personen + Event-Setting (Workshop/Meeting/Konferenz)? → foto_event
2. Eine oder wenige Personen im Vordergrund?                       → foto_personen
3. Vorwiegend Gegenstände im Mittelpunkt?                          → foto_objekte
4. Outdoor-Natur ohne Personen-Fokus?                              → foto_landschaft
5. Gebäude im Vordergrund?                                         → foto_architektur
6. Speisen/Gerichte im Mittelpunkt?                                → foto_essen

HILFSSIGNALE AUS DEM KONTEXT (nutze sie wenn das Bild allein unklar ist):
- 'Workshop', 'Schulung', 'Konferenz', 'Meeting' → foto_event
- 'Porträt', 'Foto von [Name]'                   → foto_personen
- 'Produkt', 'Werkstatt', 'Stillleben'           → foto_objekte
- 'Landschaft', 'Panorama', 'Natur'              → foto_landschaft
- 'Gebäude', 'Architektur', 'Fassade'            → foto_architektur
- 'Speise', 'Gericht', 'Essen', 'Mahlzeit'       → foto_essen

WICHTIG: Wenn du bildtyp NICHT 'foto' wählst (also diagramm/tabelle/karte/etc.),
lasse foto_subtyp leer (None — das Feld bleibt einfach weg).
"""

def build_classification_prompt(
    enriched_context: str,
    width: int,
    height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Pass-1-Prompt: Klassifikation eines Bildes in einen Top-Level-Bildtyp.

    Inputs:
      enriched_context: Web-/PDF-Kontext (Titel, umliegender Text)
      width, height:    Bildmaße in Pixel — Hilfssignal für ist_dekorativ
                        (sehr kleine Bilder oft dekorativ)
      original_alt:     Vom Autor gesetzter alt-Text (leer wenn keiner)
      user_hint:        Workflow-Variante 3: Nutzer-Hinweis mit Vorrang

    Output: prompt-String, der mit ClassificationOutput-Schema gerufen wird.
    """
    schema_doc = render_schema_for_prompt(ClassificationOutput)

    return f"""{ROLE_KLASSIFIKATOR}

{_bildtyp_liste()}

BILDGRÖSSE: {width}x{height} Pixel
ORIGINAL-ALT (vom Autor gesetzt, falls vorhanden): {original_alt or '(keiner)'}

KONTEXT (vom Web-Scraper, PDF-Extraktion oder API-Aufruf):
{enriched_context or '(kein Kontext verfügbar)'}
{user_hint_block(user_hint)}

DEINE AUFGABE:
1. Wähle EINEN der 12 Top-Level-Bildtypen für dieses Bild.
2. Gib deine Konfidenz an (hoch / mittel / niedrig).
3. Setze ist_dekorativ=true NUR wenn das Bild zweifelsfrei dekorativ ist
   (reine Trennlinie, Schmuck-Hintergrund, Designelement ohne Inhalt).
   Bei kleinen Bildern (< 80x80 px) ist Vorsicht geboten — sie sind oft,
   aber nicht immer, dekorativ.
4. Setze original_alt_brauchbar=true WENN original_alt eine sinnvolle
   funktionale Beschreibung enthält. Brauchbare Beispiele:
   - 'Logo Mercedes-Benz', 'Suche öffnen', 'Nächste Seite'
   Unbrauchbare Beispiele (→ False):
   - leer, 'Bild', 'Foto', 'Grafik', 'image001.jpg', 'IMG_2345',
     reiner Dateiname, generischer Platzhalter
5. Begründe deine Wahl in EINEM Satz (10-200 Zeichen).

WICHTIG:
{_subtyp_aufgaben_hinweis()}
- Bei Unsicherheit zwischen zwei Typen: konfidenz=mittel oder niedrig
  und in der Begründung beide Optionen nennen.
- Wenn ein Bild ein Logo ZEIGT aber als Inhaltsfoto verwendet wird
  (z.B. Pressefoto mit Firmenschild im Hintergrund), ist es 'foto',
  nicht 'logo'.
{_foto_subtyp_anweisung()}
{schema_doc}
"""
