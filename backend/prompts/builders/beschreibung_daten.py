"""Pass-3-Builder für Daten-Visualisierung + Spezialfälle.

7 Builder: illustration, diagramm, tabelle, karte, infografik, screenshot,
strukturformel. Alle haben Inventar-Pass.

ATMOSPHAERE_REGEL gilt für diese Bildtypen NICHT — Daten sind objektiv,
Wertungen über Atmosphäre haben hier keinen Platz. (Ausnahme: infografik
EINGESCHRÄNKT bei Kampagnen-Infografiken — siehe Builder-Kommentar.)
"""
from __future__ import annotations

from typing import Optional

from prompts.components.constraints import (
    ANTI_HALLUZINATION_REGELN,
    ATMOSPHAERE_REGEL,
    EIGENNAMEN_REGELN,
    KONTAKTDATEN_PFLICHT,
)
from prompts.components.roles import ROLE_BESCHREIBER
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import BeschreibungOutput, InventarOutput

from .helpers import load_examples, user_hint_block


def build_beschreibung_prompt_illustration(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Cartoon, Vektor-Grafik, gemalte Illustration, Buch-Bild.

    Höchstes Halluzinations-Risiko (Spezies-Festlegung, erfundene Interaktionen).
    Daher zusätzliche SPEZIALWARNUNG.
    """
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('illustration')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

BILDTYP: illustration (Cartoon, Vektor-Grafik, gemalte Illustration, Buch-Bild)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

SPEZIALWARNUNG FÜR ILLUSTRATIONEN:
Stilisierte Darstellungen sind die häufigste Quelle für Halluzinationen. Das Modell
neigt dazu, vereinfachte Cartoon-Tiere als Maschinen zu sehen, oder mehrdeutige
Charaktere als das wahrscheinlichste Tier-Klischee zu identifizieren.

SPEZIES-/CHARAKTER-REGEL:
Wenn das Inventar bei einem Charakter Mehrfach-Hypothesen oder niedrige Sicherheit
listet, MUSS der Output diese Unsicherheit abbilden. Verwende:
- 'stilisiertes Tier mit großen Augen, vermutlich [Hypothese 1] oder [Hypothese 2]'
- 'Cartoon-Charakter mit [konkreten sichtbaren Merkmalen]'
- NICHT: einfach die wahrscheinlichste Spezies festlegen

INTERAKTIONS-VERBOT:
Wenn das Inventar nur Objekte nebeneinander listet, schreibe NICHT dass sie
miteinander interagieren. Beispiele:
- Inventar: 'Hundekopf, Mikroskop, Laptop, Tablet, Smartphone — keine Hände sichtbar'
- VERBOTEN: 'Der Hund arbeitet am Laptop und hält ein Tablet.'
- ERLAUBT: 'Cartoon-Illustration eines Hundekopfes; daneben ein Mikroskop und drei
  Geräte (Laptop, Tablet, Smartphone).'

VOLLSTÄNDIGKEIT:
Bei Illustrationen werden Nebenelemente besonders häufig übersehen (z.B. das
Mikroskop im Hund-Bild). Gehe das Inventar VOLLSTÄNDIG durch und benenne ALLE
sichtbaren Elemente — auch wenn sie unscheinbar wirken.

SPEZIFITAETS-PFLICHT (für illustration):
Erster Satz nennt:
- Stilrichtung (Cartoon, Vektor, comic-haft etc.)
- Hauptmotiv mit ehrlicher Spezifität
- Mindestens ein konkretes Element

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_diagramm(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Balken-, Linien-, Kreis-, gestapeltes Diagramm."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('diagramm')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{KONTAKTDATEN_PFLICHT}

# WICHTIG: Bei Daten-Visualisierungen gilt ATMOSPHAERE_REGEL NICHT —
# Wertungen über Atmosphäre haben hier keinen Platz.

BILDTYP: diagramm (Balken, Linie, Kreis, gestapelt, etc.)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST-PFLICHT:
Sehende erkennen Diagramme zuerst nach KERNAUSSAGE, nicht nach Form. Der erste
Satz MUSS:
- Diagrammtyp (Balken, Linie, etc.)
- Titel oder Thema (aus inventar.lesbare_texte)
- 2-3 wichtigste Datenpunkte mit konkreten Werten:
  Spitzenwert, Schlusslicht, Trend oder Spanne

VERBOTEN: vage Aussagen wie 'China führt, gefolgt von den USA' ohne Werte.
PFLICHT: 'China führt mit 251,8 Mrd. Euro, Tschechien bildet mit 115,7 Mrd. das Schlusslicht.'

VOLLSTAENDIGKEITS-PFLICHT FUER LANGBESCHREIBUNG:
ALLE Kategorien aus inventar.lesbare_texte benennen. Pro Kategorie die
lesbaren Werte. Achsenbeschriftungen, Legenden-Werte, Anfangs-/Endwerte
bei Zeitreihen.

KEINE MARKDOWN-TABELLEN im JSON-Output. Fließtext oder strukturierte Liste.

KONSISTENZ-PFLICHT:
Prüfe vor dem Antworten: stimmen die Trends im alt_text mit den Werten
in der langbeschreibung überein? Bei Widerspruch: alt_text korrigieren.

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_tabelle(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Tabellarische Daten als Grafik. Mit Bilanz-Warnung (Buchhaltungstabellen)."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('tabelle')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{KONTAKTDATEN_PFLICHT}

# WICHTIG: ATMOSPHAERE_REGEL gilt NICHT für tabelle — Wertungen über
# Atmosphäre haben hier keinen Platz. Tabellen sind Daten, keine Stimmungen.

BILDTYP: tabelle (tabellarische Daten als Grafik)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — die verbindliche Faktenbasis):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST-PFLICHT FÜR TABELLEN:
Sehende erkennen Tabellen zuerst nach KERNAUSSAGE, nicht nach Form. Der erste
Satz MUSS:
- 'Tabelle —' + Thema (aus inventar.lesbare_texte: Titel-Eintrag)
- Die wichtigste Aussage basierend auf den RICHTIGEN Endwerten/Bilanzsumme
- Maximal 250 Zeichen

VERBOTEN: Nichtssagende Eröffnungen wie 'Eine Tabelle zeigt verschiedene Werte.'

BILANZ-WARNUNG (KRITISCH):
- Unterscheide ABSCHNITTS-Zwischensummen von der GESAMT-Summe
- 'Anlagevermögen' und 'Umlaufvermögen' sind ABSCHNITTE (Zwischensummen)
- 'Bilanzsumme', 'Bilanzsumme Aktiva', 'Gesamtsumme' oder 'Summe Aktiva/Passiva'
  ist das GESAMTERGEBNIS
- Die LETZTE Summenzeile der Tabelle ist fast immer die Bilanzsumme,
  NICHT ein Abschnittsname
- FALSCH: 'Gesamtsumme des Umlaufvermögens beträgt X EUR' wenn X die Bilanzsumme ist
- RICHTIG: 'Bilanzsumme Aktiva beträgt X EUR' oder 'Gesamtsumme Aktiva beträgt X EUR'

VOLLSTÄNDIGKEITS-PFLICHT FÜR LANGBESCHREIBUNG:
1. Gesamtsumme/Bilanzsumme ZUERST nennen — sie ist die wichtigste Zahl, darf
   NICHT durch Token-Limits abgeschnitten werden
2. Spaltenköpfe wortgetreu auflisten
3. ALLE Zeilen mit korrekter Spaltenzuordnung
4. Spitzen- und Tiefstwerte benennen, auffällige Muster
5. Nur bei sehr kleinen Tabellen (max 4x4) alle Werte einzeln auflisten
6. Bei größeren Tabellen: Zusammenfassung statt vollständige Wertliste

KEINE MARKDOWN-TABELLEN im JSON-Output — Fließtext oder strukturierte Liste.

SPALTEN-ZUORDNUNG (KRITISCH):
1. Lies ZUERST alle Spaltenköpfe von links nach rechts
2. Lies DANN jede Zeile und ordne JEDEN Wert seiner EXAKTEN Spalte zu
3. Verwechsle NIEMALS Zwischenwerte (Zugänge, Abschreibungen, Veränderungen)
   mit Bestands- oder Endwerten
4. Wenn eine Zeile Werte in der Spalte '01.01.' UND '31.12.' hat, sind das
   VERSCHIEDENE Werte — nenne BEIDE mit Spaltenzuordnung
5. Bei Buchhaltungstabellen: Anfangs- und Endbestand sind die entscheidenden
   Werte, NICHT die Bewegungen dazwischen

EINHEITEN: %, EUR, Mio., Tsd. penibel übernehmen — nicht weglassen, nicht umformen.

OCR-TEXT: Wenn inventar.lesbare_texte Zellinhalte aus OCR enthält, sind diese
die primäre Wahrheitsquelle. Bei OCR-Werten + visueller Wahrnehmung im Konflikt:
OCR vertrauen.

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_karte(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Landkarte, Stadtplan, Lageplan, Übersichtskarte."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('karte')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{EIGENNAMEN_REGELN}

# WICHTIG: ATMOSPHAERE_REGEL gilt NICHT für karte.

BILDTYP: karte (Landkarte, Stadtplan, Lageplan, Übersichtskarte)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST-PFLICHT FÜR KARTEN:
Der erste Satz MUSS:
- 'Karte —' + Gebiet (Stadt, Region, Land — aus inventar oder Kontext)
- Hauptthema (was wird dargestellt?)
- Räumliche Kernaussage (z.B. 'Konzentration im Süden', 'gleichmäßig verteilt',
  'Cluster in den Großstädten')
- Maximal 350 Zeichen

VOLLSTÄNDIGKEITS-PFLICHT FÜR LANGBESCHREIBUNG:
1. Markierte Standorte VOLLSTÄNDIG auflisten (das sind die Kerninformationen
   einer Karte mit Markierungen)
2. Legende EXPLIZIT erklären (Symbole, Farben, Größen-Bedeutungen)
3. Räumliche Verteilung beschreiben — mit Himmelsrichtungen statt nur 'oben/unten'
   wenn die Karte geografisch ausgerichtet ist (Norden oben)
4. Maßstab nennen wenn lesbar

ORTSNAMEN — Originalsprache beibehalten:
- 'Bordeaux' nicht 'Bordeo'
- 'İstanbul' nicht 'Istanbul' (wenn das I-Punkt-Zeichen lesbar ist)
- 'Köln' nicht 'Cologne' (auch wenn der Kontext englisch ist)

HINTERGRUND-ORTSNAMEN:
Bei Karten mit vielen Hintergrund-Städten zur Orientierung NICHT erschöpfend
auflisten — nur die beschrifteten/markierten relevanten Standorte. Andere
Städte erwähnen wenn sie für die räumliche Einordnung wichtig sind ('zwischen
München und Stuttgart').

SYMBOLIK INTERPRETIEREN:
- Rote Markierungen sind nicht automatisch 'Warnungen' oder 'Gefahren' —
  Bedeutung kommt aus der Legende
- Größenunterschiede von Markern (große vs. kleine Kreise) bedeuten meist
  unterschiedliche Werte — Legende prüfen

ANTI-HALLUZINATION:
- Keine Orte oder Routen erfinden die nicht im Inventar sind
- Bei verschwommenen Details: 'Details teilweise nicht lesbar' — statt zu raten

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_infografik(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Schaubild, Übersichtsgrafik mit Stationen oder Schritten."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('infografik')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{ATMOSPHAERE_REGEL}

# Atmosphäre-Regel gilt für Infografik EINGESCHRÄNKT — nur dort wo
# eine bewusste Designwahl Stimmung transportiert (z.B. Kampagnen-
# Infografik), nicht bei reinen Daten-Visualisierungen.

BILDTYP: infografik (Schaubild, Übersichtsgrafik mit Stationen oder Schritten)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST-PFLICHT FÜR INFOGRAFIK:
Der erste Satz MUSS:
- 'Infografik —' + Hauptthema (aus inventar.lesbare_texte: Titel)
- Zentrale Kernaussage mit konkreten Datenpunkten WENN vorhanden
- Maximal 400 Zeichen

Beispiel RICHTIG: 'Infografik — Bürokratie-Entlastung 2024: Fast die
Hälfte (39%) entfällt auf das Wachstumschancengesetz, gefolgt von
vier weiteren Maßnahmen mit zusammen 61%.'

Beispiel FALSCH: 'Eine Infografik mit verschiedenen Daten.'

VOLLSTÄNDIGKEITS-PFLICHT FÜR LANGBESCHREIBUNG:
1. Alle inhaltlichen Stationen in LOGISCHER Reihenfolge auflisten
   (chronologisch / hierarchisch / kausal — je nach Infografik-Typ)
2. Beziehungen zwischen Stationen benennen ('A führt zu B',
   'X umfasst Y', 'Schritt 1 aktiviert Schritt 2')
3. ALLE konkreten Zahlen, Prozente, Mengenangaben übernehmen
4. Maximal 1500 Zeichen, Fließtext oder strukturierte Liste

VERBOTEN — visuelle Layout-Beschreibungen:
- 'oben links steht...'
- 'ein Pfeil zeigt von X nach Y...'
- 'im Zentrum befindet sich...'
- 'die linke Hälfte des Bildes zeigt...'

Stattdessen INHALTLICH formulieren:
- 'Schritt 1 ist X, daraus folgt Schritt 2 mit Y'
- 'Hauptkategorie A umfasst die Unterkategorien B, C und D'
- 'Im Mittelpunkt steht das Konzept X' (wenn das WIRKLICH die
  inhaltliche Botschaft ist, nicht nur die geometrische Position)

OCR-TEXT als Pflichtquelle:
Wenn inventar.lesbare_texte Beschriftungen enthält, sind diese
wortgetreu zu übernehmen. Bei Konflikt visuelle Wahrnehmung vs.
OCR: OCR-Text vertrauen.

KONTAKTDATEN/URL-PFLICHT:
Wenn inventar.lesbare_texte Typ 'kontaktdaten' oder 'url'
enthält (häufig bei Behörden-Infografiken am unteren Rand),
MÜSSEN diese in der Beschreibung stehen — wortgetreu mit
Trennzeichen.

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_screenshot(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Bildschirmfoto einer Anwendung, Webseite oder UI."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('screenshot')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{KONTAKTDATEN_PFLICHT}

# ATMOSPHAERE_REGEL gilt für Screenshots NICHT — UI-Beschreibungen
# sind funktional, keine emotionalen Wertungen erlaubt.

BILDTYP: screenshot (Bildschirmfoto einer Anwendung, Webseite oder UI)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST FÜR SCREENSHOT:
Der erste Satz MUSS:
- Anwendung (wenn aus URL-Leiste, Titel, Logo identifizierbar) ODER
  generischer Anwendungstyp ('Browser-Fenster', 'Texteditor', 'E-Mail-Programm')
- Zustand oder aktuelle Aktion (was ist gerade sichtbar?)
- Maximal 350 Zeichen

Beispiel RICHTIG: 'Screenshot der InkluDocs-Web-Oberfläche, Projekt-
Übersicht mit drei laufenden Bilduploads und einem fertig analysierten
PDF mit 12 Bildern.'

Beispiel FALSCH: 'Ein Screenshot zeigt eine Anwendung mit verschiedenen
Elementen.'

VOLLSTÄNDIGKEITS-PFLICHT FÜR LANGBESCHREIBUNG:
1. Sichtbare UI-Elemente in funktionaler Hierarchie:
   - Hauptmenü / Navigation
   - Hauptbereich / Inhalt
   - Sekundär-Bereiche / Sidebars
   - Statusleiste / Footer
2. Lesbare Texte WORTGETREU übernehmen — vor allem:
   - URL in der Adressleiste (vollständig)
   - Fenstertitel
   - Buttons / Links die der Nutzer sehen würde
   - Statusmeldungen
   - Eingaben in Formularfeldern
3. Maximal 1000 Zeichen, leer wenn alt_text alles Wesentliche sagt

VERBOTEN — Erfundene Anwendungs-Identifikation:
- Wenn weder URL noch Logo noch Titel die Anwendung benennen,
  schreibe NICHT 'Screenshot von Microsoft Word' — sondern
  'Screenshot eines Texteditors' oder generischer
- Bei unklarer Domain in URL: nur die sichtbare Domain nennen,
  nicht raten was sich dahinter verbirgt

DARK MODE / LIGHT MODE:
Wenn relevant für die Beschreibung (z.B. bei UI-Tutorials),
benennen. Sonst weglassen — ist meist irrelevant für die Funktion.

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_strukturformel(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Chemische Struktur-, Reaktions- oder Summenformel."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('strukturformel')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

# ATMOSPHAERE_REGEL gilt für strukturformel NICHT — Chemie ist
# objektiv, keine Stimmungsbeschreibungen.

BILDTYP: strukturformel (Chemische Struktur-, Reaktions- oder Summenformel)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST FÜR STRUKTURFORMEL:
Der erste Satz MUSS:
- Präfix 'Strukturformel —' ODER 'Reaktionsgleichung —'
- Stoffname falls aus Kontext oder Beschriftung erkennbar
- Summenformel WENN klar lesbar
- Maximal 250 Zeichen

Beispiel RICHTIG: 'Strukturformel — Methanol (CH3OH): Methylgruppe
mit Hydroxylgruppe.'

Beispiel RICHTIG für Reaktion: 'Reaktionsgleichung — Veresterung von
Essigsäure mit Methanol zu Methylacetat und Wasser, katalysiert
durch Schwefelsäure.'

VOLLSTÄNDIGKEITS-PFLICHT FÜR LANGBESCHREIBUNG:
Bei Strukturformeln:
1. Grundgerüst beschreiben (Kette, Ring, verzweigt)
2. Atome und Atomgruppen (CH3, OH, COOH, NH2, Aromaten etc.)
3. Bindungstypen (Einfach-, Doppel-, Dreifach-Bindung) wenn sichtbar
4. Funktionelle Gruppen explizit benennen
5. Stereochemie wenn dargestellt (cis/trans, R/S)

Bei Reaktionsgleichungen:
1. Edukte (links vom Reaktionspfeil)
2. Reaktionsbedingungen (über/unter dem Pfeil — Katalysator,
   Temperatur, Druck, Lösungsmittel)
3. Produkte (rechts vom Reaktionspfeil)
4. Reaktionstyp wenn aus Kontext bekannt (Substitution, Addition,
   Eliminierung, Redox etc.)

Maximal 800 Zeichen, Fließtext.

NOTATION KORREKT WIEDERGEBEN:
- Indizes als normale Zahlen ('CH3' — nicht 'CH₃' weil Screenreader
  Indizes oft schlecht vorlesen)
- Ladungen explizit ('Natrium-Kation' oder 'Na+')
- Reaktionspfeile beschreiben als 'reagiert zu' oder 'ergibt'

ANTI-HALLUZINATION CHEMIE:
- Erfinde KEINE Atome die nicht im Bild sind
- Bei unleserlichen Bindungen: 'Bindungstyp nicht eindeutig
  erkennbar' statt zu raten
- Stoffnamen NUR aus Kontext oder Bildbeschriftung — Strukturen
  visuell zu identifizieren ist fehleranfällig (außer einfachsten
  Molekülen wie H2O, CO2)

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""
