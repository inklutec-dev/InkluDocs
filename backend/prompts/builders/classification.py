"""Pass-1-Prompt-Builder: Klassifikator als Router fuer die Premium-Pipeline.

Version 2 vom 13.05.2026 — Routing-fokussierte Neuformulierung
basierend auf ChatGPT-Architektur + Steve-Reviews. Vorher:
definitionslastig in 179 Zeilen, jetzt: konfliktbasiert.

Der Klassifikator ist ROUTER, kein Beschreiber. Er entscheidet:
- bildtyp                       (Top-Level, 12 Optionen)
- foto_subtyp                   (bei lean+foto Pflicht)
- konfidenz                     (hoch/mittel/niedrig)
- ist_dekorativ                 (Bool)
- original_alt_brauchbar        (Bool)
- klassifikations_begruendung   (10-200 Zeichen, technisch)

W3-Fix bleibt: original_alt_brauchbar wird HIER vom Klassifikator
gesetzt und steuert, ob der Premium-Builder laeuft oder der
vorhandene PDF-Alt-Text behalten wird.
"""
from __future__ import annotations

from typing import Optional

from prompts.components.roles import ROLE_KLASSIFIKATOR
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import ClassificationOutput

from .helpers import bilddaten_verlagert, user_hint_block


_BILDTYP_INVENTAR = """DIE ZWÖLF BILDTYPEN

1. foto: echte Fotografie (Personen, Objekte, Räume, Landschaft, Pressefoto),
   auch die Reproduktion eines Kunstwerks (Gemälde, künstlerische Zeichnung,
   Druckgrafik, Skulptur).
2. illustration: Gebrauchsgrafik, die eine Idee oder Aussage bildlich fasst:
   Zeichnung, Cartoon, Vektorgrafik, Produkt- oder Werbegrafik mit Symbolen,
   Kacheln, Sprechblasen oder Siegeln. Ein Kunstwerk ist foto.
3. diagramm: Balken-, Linien-, Kreis-, gestapeltes oder Streudiagramm.
4. tabelle: tabellarische Daten als Grafik.
5. karte: Landkarte, Stadtplan, Lageplan.
6. infografik: Schaubild oder Plakat, das Daten, Prozessschritte oder erklärte
   Zusammenhänge mit Layout verbindet.
7. screenshot: Bildschirmfoto mit sichtbarer Oberfläche (Browser, App, Fenster).
8. strukturformel: chemische Struktur-, Reaktions- oder Summenformel.
9. logo: allein stehendes Marken-, Organisations- oder Lizenzlogo.
10. icon: kleines funktionales Symbol (Lupe, Menü, Warenkorb).
11. funktional: Navigations- oder Steuerelement mit Zustand (Blätterpfeile,
    Fortschrittsanzeige, Brotkrumenpfad).
12. dekorativ: reines Gestaltungselement ohne Informationswert (Trennlinie,
    Farbfläche, Verlauf, Zierrahmen), unabhängig von der Größe."""


_ROUTING_REGELN = """ENTSCHEIDUNGSREGELN (in dieser Reihenfolge)

1. Der Bildinhalt entscheidet, nicht Dateiname oder vorhandener Alt-Text. Der
   Kontext hilft, überschreibt aber nicht, was sichtbar ist.
2. Ist das Bild selbst eine Bildschirmaufnahme (Browserleiste, Fensterrahmen
   oder App-Oberfläche füllen das Bild, keine Kameraperspektive, kein
   Gerätegehäuse), ist es screenshot, auch wenn darin ein Diagramm steht.
3. Ein Foto, auf dem ein Diagramm, ein Logo oder ein Bildschirm zu sehen ist,
   bleibt foto.
4. infografik braucht Daten, Prozessschritte oder erklärte Zusammenhänge. Eine
   Produkt- oder Werbegrafik ohne diese Merkmale ist illustration.
5. Ein allein stehendes Markenzeichen ist logo. Ein kleines Symbol ohne Zustand
   ist icon, mit Zustand (aktiv, Seite 3 von 12, ausgegraut) funktional.
6. dekorativ hängt an der Funktion, nicht an der Größe: Transportiert das Bild
   an seiner Stelle ein Motiv, Text, Navigation, Branding oder Stimmung, ist es
   nicht dekorativ. Ein verlinktes Bild und ein Bild mit Bildunterschrift sind
   nie dekorativ. Im Zweifel nicht dekorativ.
7. Bei Unsicherheit zwischen zwei Typen: konfidenz mittel oder niedrig und beide
   Typen in der Begründung."""


_FOTO_SUBTYP_LEAN = """FOTO-UNTERTYP (Pflichtfeld foto_subtyp, wenn bildtyp foto ist)

Das Feld bildtyp bleibt "foto"; der Untertyp steht getrennt in foto_subtyp.
- foto_event: mehrere Personen und ein erkennbarer Veranstaltungsanlass
  (Workshop, Schulung, Konferenz, Bühne, Beamer, Namensschilder, Catering,
  Moderationsmaterial). Mehrere Personen allein reichen nicht.
- foto_personen: eine oder mehrere Personen im Mittelpunkt ohne
  Veranstaltungsanlass, auch Gruppenfotos und Porträts.
- foto_objekte: Objekte, Produkte, Werkstücke, Sammlungen, Stillleben,
  Kunstwerke (Gemälde, Zeichnung, Druckgrafik, Skulptur).
- foto_architektur: Gebäude, Räume, Fassaden, Baudetails.
- foto_essen: Speisen, Getränke, Lebensmittel.
- foto_landschaft: Natur, Panorama, Außenszene ohne Personen- oder
  Architekturfokus.
Prüfe in dieser Reihenfolge: Personen mit Anlass, Personen ohne Anlass, Objekte,
Architektur, Essen, Landschaft. Bei allen anderen Bildtypen bleibt foto_subtyp leer."""


def _foto_subtyp_block() -> str:
    """foto_subtyp-Block: der Klassifikator waehlt den Subtyp direkt (Lean-Weg).

    Die Multi-Pass-Variante (Subtyp im Inventar-Pass) ist seit 07.09.2026 abgebaut.
    """
    return _FOTO_SUBTYP_LEAN


def _inputs_block(width, height, original_alt, enriched_context, user_hint) -> str:
    """INPUTS inline — oder als Verweis, wenn die Bilddaten ans Ende wandern
    (Prompt-Caching, siehe helpers.bilddaten_am_ende)."""
    if bilddaten_verlagert():
        return 'INPUTS: siehe Block BILDDATEN am Ende dieses Prompts (Bildgroesse, Original-Alt, Kontext, Nutzer-Hinweis).'
    return f"""INPUTS:
- Bildgroesse: {width}x{height} Pixel
- Original-Alt vom Autor: {original_alt or '(keiner)'}
- Kontext (Bildunterschrift, umliegender Text, Angaben des Aufrufers): {enriched_context or '(kein Kontext)'}
{user_hint_block(user_hint)}"""


def build_classification_prompt(
    enriched_context: str,
    width: int,
    height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Pass-1-Prompt: Klassifikation eines Bildes als Routing-Entscheidung.

    Inputs:
      enriched_context: Web-/PDF-Kontext (Titel, umliegender Text)
      width, height:    Bildmasse in Pixel — Hilfssignal fuer ist_dekorativ
      original_alt:     Vom Autor gesetzter alt-Text (leer wenn keiner)
      user_hint:        Workflow-Variante 3: Nutzer-Hinweis mit Vorrang

    Output: prompt-String, der mit ClassificationOutput-Schema gerufen wird.

    Hinweis: dekorativ.py prüft die Entscheidung zusätzlich heuristisch.
    """
    schema_doc = render_schema_for_prompt(ClassificationOutput)

    return f"""{ROLE_KLASSIFIKATOR}

{_BILDTYP_INVENTAR}

{_inputs_block(width, height, original_alt, enriched_context, user_hint)}

{_ROUTING_REGELN}

{_foto_subtyp_block()}

VORHANDENER ALT-TEXT (Feld original_alt_brauchbar)
Wahr, wenn der vom Autor gesetzte Alt-Text die Funktion oder den Inhalt sinnvoll
benennt und zum Bild passt ("Logo Musterwerk", "Nächste Seite", "Diagramm
Quartalsumsatz"). Falsch bei leer, "Bild", "Foto", "Grafik", Dateinamen,
Platzhaltern und bei einem Text, der zwar sinnvoll klingt, aber eine andere
Aktion oder einen anderen Zustand benennt als das Bild zeigt.

DEKORATIV (Feld ist_dekorativ)
Wahr nur, wenn das Bild zweifelsfrei ein reines Gestaltungselement ohne
Informationswert ist. Die Größe ist kein Kriterium; ein kleines Bedienelement
ist nie dekorativ. Im Zweifel falsch.

BEGRÜNDUNG (Feld klassifikations_begruendung)
Ein Satz mit dem Merkmal, das den Ausschlag gab ("Browserleiste und Fensterrahmen
sprechen für screenshot."). Keine Bildbeschreibung.

KONFIDENZ
hoch nur bei klarer Dominanz eines Typs, sonst mittel oder niedrig mit beiden
Typen in der Begründung.

{schema_doc}
"""
