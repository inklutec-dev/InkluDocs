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

import os
from typing import Optional

from prompts.components.roles import ROLE_KLASSIFIKATOR
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import ClassificationOutput

from .helpers import bilddaten_verlagert, user_hint_block


_BILDTYP_INVENTAR = """DIE 12 TOP-LEVEL-BILDTYPEN:

1. foto         — Echte Fotografie (Personen, Objekte, Innen/Aussen, Pressefoto)
2. illustration — Cartoon, Vektor-Grafik, gemalte Illustration, Buchbild
3. diagramm     — Balken-, Linien-, Kreis-, gestapeltes Diagramm (isoliert)
4. tabelle      — Tabellarische Daten als Grafik
5. karte        — Landkarte, Stadtplan, Lageplan
6. infografik   — Schaubild, Uebersicht, Plakat/Flyer mit Layout und Info
7. screenshot   — Bildschirmfoto mit sichtbarer UI (Browser, App, Window)
8. strukturformel — Chemische Struktur-, Reaktions-, Summenformel
9. logo         — Alleinstehendes Marken-/Organisations-/Lizenzlogo
10. icon        — Kleines funktionales Symbol (Lupe, Burger, Warenkorb)
11. funktional  — Navigations-/Steuerungselement mit Zustand
                  (Paginierung, Vor/Zurueck, Fortschrittsanzeige, Breadcrumb)
12. dekorativ   — Reines Designelement ohne Informationswert (leerer
                  Alt-Text) — unabhaengig von der Groesse, auch grosse
                  Trennbanner und Farbflaechen"""


_ROUTING_REGELN = """ROUTING-REGELN (in dieser Prioritaet):

1. Bildinhalt schlaegt Dateiname und generischen Alt-Text.
2. Kontext darf helfen, aber sichtbaren Inhalt nicht ueberschreiben.
3. Foto eines Diagramms an Wand oder Beamer -> foto bzw. foto_event,
   NICHT diagramm.
4. Screenshot mit Browser/UI-Rahmen sichtbar -> screenshot,
   auch wenn ein Diagramm darin zu sehen ist.
   Sichtbare Browserleisten, Fensterrahmen, Toolbars, Tabs oder App-UI
   schlagen eingebettete Inhalte.
5. Pressefoto mit Firmenlogo im Hintergrund -> foto, NICHT logo.
   Logo macht aus einem Bild kein logo.
6. Alleinstehendes Markenzeichen ohne Foto-Kontext -> logo.
7. Plakat oder Flyer mit Layout und Information -> infografik.
8. Kleine UI-Symbole (Lupe, Burger, Pfeil) -> icon oder funktional.
   funktional = Element mit Zustand, icon = generisches Symbol.
9. dekorativ haengt an der FUNKTION, nicht an der Groesse: rein
   schmueckende Designelemente ohne Informationswert (Trennlinien,
   Farbflaechen, Verlaeufe, abstrakte Formen-Banner, Zierrahmen,
   Hintergrund-Texturen) sind dekorativ — auch als bildschirmbreites
   Banner. Massgeblich ist der Kontext der Seite: Transportiert das Bild
   an seiner Stelle Text, Navigation, Branding, ein Motiv (Personen,
   Produkte, Orte) oder Stimmungs-/Fach-Information, ist es NICHT
   dekorativ — ein grosses Stimmungsfoto ist Inhalt, kein Schmuck.
   Bei sehr kleinen Bildern (<100x100 px) genau pruefen — oft, aber
   nicht immer dekorativ. Im Zweifel NICHT dekorativ.
10. Bei Unsicherheit zwischen zwei Typen: konfidenz=mittel/niedrig
    und in der Begruendung beide Optionen nennen."""


_FOTO_SUBTYP_LEAN = """FOTO-SUBTYP (Pflichtfeld bei bildtyp=foto im Lean-Modus):

WICHTIG: foto_event, foto_personen, foto_objekte etc. sind SUB-TYPEN.
Sie gehoeren NICHT ins Feld 'bildtyp'. Das Feld 'bildtyp' bleibt immer
'foto'. Zusaetzlich setzt du das separate Feld 'foto_subtyp'.

Beispiel-Output bei einem Workshop-Foto:
  bildtyp = 'foto'
  foto_subtyp = 'foto_event'

Waehle exakt EINEN der sechs foto_subtyp-Werte:

foto_event:
Mehrere Personen UND klarer Anlass- oder Veranstaltungs-Kontext.
Indikatoren: Workshop, Schulung, Konferenz, Meeting, Buehne, Beamer,
Namensschilder, Catering, Moderationsmaterial, Vortragssetting,
erkennbare Veranstaltungsorganisation.
WICHTIG: Mehrere Personen allein reichen NICHT fuer foto_event.
Es muss mindestens ein erkennbarer Veranstaltungsindikator vorliegen.

foto_personen:
Eine oder mehrere Personen im Mittelpunkt OHNE klare Event-Indikatoren.
Auch Gruppenfoto ohne Veranstaltungskontext gehoert hierhin.
Mehrere Personen ohne erkennbare Veranstaltungsindikatoren bleiben
foto_personen, nicht foto_event.

foto_objekte:
Objekte, Produkte, Werkstuecke, Material, Sammlungen oder Stillleben
im Mittelpunkt.

foto_architektur:
Gebaeude, Raeume, Innenraeume, Fassaden, architektonische Details
im Mittelpunkt.

foto_essen:
Speisen, Getraenke, Gerichte oder Lebensmittel im Mittelpunkt.

foto_landschaft:
Natur, Panorama, Outdoor-Szene ohne Personen- oder Architektur-Fokus.

ENTSCHEIDUNGSREIHENFOLGE FUER foto_subtyp (in dieser Prioritaet pruefen,
bildtyp bleibt dabei immer 'foto'):

1. Mehrere Personen + Event-Indikatoren -> foto_subtyp = 'foto_event'
2. Personen ohne Event-Indikatoren     -> foto_subtyp = 'foto_personen'
3. Objekte im Mittelpunkt              -> foto_subtyp = 'foto_objekte'
4. Architektur/Raum im Mittelpunkt     -> foto_subtyp = 'foto_architektur'
5. Essen/Lebensmittel im Mittelpunkt   -> foto_subtyp = 'foto_essen'
6. Natur/Landschaft im Mittelpunkt     -> foto_subtyp = 'foto_landschaft'

Wenn bildtyp NICHT 'foto' ist, lasse foto_subtyp leer (None)."""


_FOTO_SUBTYP_MULTIPASS = """FOTO-SUBTYP (Multi-Pass-Modus):

Sub-Typen fuer foto (foto_personen, foto_event etc.) werden NICHT hier
entschieden — das macht spaeter der Inventar-Pass. Lasse foto_subtyp leer."""


def _foto_subtyp_block() -> str:
    """Modus-abhaengiger foto_subtyp-Block (lean=Pflicht, full=delegiert)."""
    if os.environ.get('V4_PASS_MODE', 'full').strip().lower() == 'lean':
        return _FOTO_SUBTYP_LEAN
    return _FOTO_SUBTYP_MULTIPASS


def _inputs_block(width, height, original_alt, enriched_context, user_hint) -> str:
    """INPUTS inline — oder als Verweis, wenn die Bilddaten ans Ende wandern
    (Prompt-Caching, siehe helpers.bilddaten_am_ende)."""
    if bilddaten_verlagert():
        return 'INPUTS: siehe Block BILDDATEN am Ende dieses Prompts (Bildgroesse, Original-Alt, Kontext, Nutzer-Hinweis).'
    return f"""INPUTS:
- Bildgroesse: {width}x{height} Pixel
- Original-Alt vom Autor: {original_alt or '(keiner)'}
- Kontext (Web-Scraper, PDF, API): {enriched_context or '(kein Kontext)'}
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

    ACHTUNG Sync-Pflicht: Die Pixel-Schwelle im IST_DEKORATIV-Hinweis
    (<100x100 px) muss mit _DEKORATIV_MAX_SIDE in dekorativ.py synchron
    gehalten werden (angeglichen in Paket 1, 16.07.2026 — vorher 80x80).
    """
    schema_doc = render_schema_for_prompt(ClassificationOutput)

    return f"""{ROLE_KLASSIFIKATOR}

Du bist Router fuer die InkluDocs Premium-Pipeline. Du klassifizierst Bilder
in einen von 12 Top-Level-Typen und triffst weitere Routing-Entscheidungen.
Du beschreibst Bilder nicht — du routest sie.

OUTPUT-STRUKTUR (immer dieses JSON-Format):

  {{
    "bildtyp": "<MUSS einer der 12 Werte sein: foto, illustration, diagramm,
                 tabelle, karte, infografik, screenshot, strukturformel,
                 logo, icon, funktional, dekorativ>",
    "foto_subtyp": "<NUR wenn bildtyp='foto': einer von foto_event,
                     foto_personen, foto_objekte, foto_architektur,
                     foto_essen, foto_landschaft. Sonst null.>",
    "konfidenz": "<hoch | mittel | niedrig>",
    "ist_dekorativ": <true | false>,
    "original_alt_brauchbar": <true | false>,
    "klassifikations_begruendung": "<ein Satz, 10-200 Zeichen>"
  }}

KRITISCH: 'bildtyp' und 'foto_subtyp' sind GETRENNTE Felder.
- Workshop-Foto:   bildtyp='foto',  foto_subtyp='foto_event'  korrekt
- Workshop-Foto:   bildtyp='foto_event'                       FALSCH
- Screenshot:      bildtyp='screenshot', foto_subtyp=null     korrekt
- Stillleben-Foto: bildtyp='foto', foto_subtyp='foto_objekte' korrekt

{_BILDTYP_INVENTAR}

{_inputs_block(width, height, original_alt, enriched_context, user_hint)}

{_ROUTING_REGELN}

{_foto_subtyp_block()}

ORIGINAL_ALT_BRAUCHBAR (Boolean):

Setze True wenn der vom Autor gesetzte original_alt eine sinnvolle
funktionale Beschreibung enthaelt. Beispiele:
- True:  'Logo Mercedes-Benz', 'Suche oeffnen', 'Naechste Seite',
         'Diagramm Quartalsumsatz Q3 2025'
- False: leer, 'Bild', 'Foto', 'Grafik', 'image001.jpg', 'IMG_2345',
         reiner Dateiname, generischer Platzhalter

Wenn True: die Pipeline behaelt den vorhandenen Alt-Text und spart
den Premium-Builder-Lauf. Falsch True kostet uns Qualitaet, falsch
False kostet Geld — beides hat Konsequenzen.

IST_DEKORATIV (Boolean):

True NUR wenn das Bild zweifelsfrei ein reines Designelement ohne
Informationswert ist (Trennlinie, Farbflaeche, Verlauf, abstraktes
Schmuck-Banner, Hintergrund-Textur) — die Groesse ist dabei egal, auch
ein bildschirmbreites Trennbanner kann dekorativ sein. Zeigt das Bild
ein Motiv (Personen, Produkte, Orte), Text, Navigation oder Branding,
ist es NICHT dekorativ. Im Zweifel False. Sehr kleine Bilder
(<100x100 px) sind oft, aber nicht immer dekorativ — pruefen.

KLASSIFIKATIONS_BEGRUENDUNG (10-200 Zeichen, ein Satz):

Technisch und knapp. Welcher Indikator hat den Ausschlag gegeben.

Gute Beispiele:
- 'Mehrere Personen mit Namensschildern und Praesentationsumgebung
   sprechen fuer foto_event.'
- 'UI-Rahmen und Browserleiste sprechen fuer screenshot.'
- 'Isoliertes Balkendiagramm ohne UI spricht fuer diagramm.'
- 'Pressefoto mit Mercedes-Logo im Hintergrund -> foto, nicht logo.'

Schlechte Beispiele (vermeiden):
- 'Das Bild zeigt interessante Personen.' (vage, kein Routing-Grund)
- 'Es wirkt wie ein Workshop.' (Hedge ohne Indikator)
- 'Wahrscheinlich ein Diagramm.' (unsicher ohne Begruendung)
- Mini-Alt-Text statt Routing-Begruendung.

KONFIDENZ:

Waehle hoch, mittel oder niedrig. hoch nur bei klarer Dominanz eines
Typs. Bei mittel/niedrig nenne in der Begruendung beide moeglichen Typen.

{schema_doc}

LETZTE PRUEFUNG VOR DEINER ANTWORT (sehr wichtig):

Pruefe dein JSON gegen diese Regeln, bevor du antwortest:

1. 'bildtyp' MUSS exakt einer dieser 12 Werte sein:
   foto, illustration, diagramm, tabelle, karte, infografik,
   screenshot, strukturformel, logo, icon, funktional, dekorativ.

2. 'foto_event', 'foto_personen', 'foto_objekte', 'foto_architektur',
   'foto_essen' und 'foto_landschaft' sind KEINE gueltigen Werte fuer
   'bildtyp'. Diese gehoeren ausschliesslich ins separate Feld
   'foto_subtyp'.

3. Bei einem Workshop-/Meeting-/Konferenz-Foto lautet die korrekte
   Antwort:
     bildtyp = 'foto'
     foto_subtyp = 'foto_event'
   Niemals: bildtyp = 'foto_event'.

4. Bei allen Bildtypen ausser 'foto' ist 'foto_subtyp' = null.

Wenn dein erster Entwurf bildtyp='foto_event' (oder einen anderen
foto_*-Wert) enthaelt, korrigiere ihn JETZT: bildtyp='foto',
foto_subtyp='foto_event'. Erst dann antworte.
"""
