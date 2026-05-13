"""Pass-3-Builder für die foto-Familie (6 Sub-Typen).

Sub-Typen werden vom Inventar-Pass entschieden — siehe BildtypEffective
im Schema-Paket und BILDTYP_INVENTAR_SCHWERPUNKTE['foto'] im inventar.py.

Alle 6 Builder folgen dem gleichen Aufbau:
  ROLE_BESCHREIBER + ANTI_HALLUZINATION_REGELN + (ATMOSPHAERE_REGEL je nach Typ)
  + (PERSONEN_REGELN bei foto_personen/foto_event)
  + (KONTAKTDATEN_PFLICHT bei Typen wo Kontaktdaten häufig vorkommen)
  + Bildtyp-spezifische SPEZIFITAETS-PFLICHT + VOLLSTÄNDIGKEITS-PFLICHT
  + Few-Shot + Schema-Doc

ATMOSPHAERE_REGEL gilt für foto_essen EINGESCHRÄNKT (nur visuell belegbare
Eigenschaften, keine Geschmacks-Adjektive) — siehe Builder-Kommentar.
"""
from __future__ import annotations

from typing import Optional

from .helpers import resolve_prompt_mode

from prompts.components.constraints import (
    ANTI_HALLUZINATION_REGELN,
    ATMOSPHAERE_REGEL,
    EVIDENZ_STUFEN_REGELN,
    KONTAKTDATEN_PFLICHT,
    PERSONEN_REGELN,
)
from prompts.components.roles import ROLE_BESCHREIBER
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import BeschreibungOutput, InventarOutput

from .helpers import load_examples, user_hint_block



# =====================================================================
# Helper-Funktionen fuer Premium-Builder (foto_personen, foto_event)
# =====================================================================
# Diese Helpers extrahieren die wiederverwendeten Sektionen die in
# beiden Premium-Buildern identisch vorkommen. So wird die Personen-
# Logik, Kontext-Logik, Atmosphaere-Logik etc. nur an EINER Stelle
# gepflegt — Drift-Vermeidung.
#
# Sie werden NUR von den Premium-Buildern (foto_personen, foto_event)
# verwendet. Die anderen 4 foto-Builder (foto_objekte, foto_essen,
# foto_landschaft, foto_architektur) nutzen weiter die alten Constraint-
# Imports und sind unveraendert.
#
# Konzeptionell setzen die Helpers ChatGPTs "Personen als Dimension"-
# Idee in Code um: jeder Helper ist eine Dimension die in mehreren
# Bildtypen vorkommen kann.


def _render_personenregeln_block() -> str:
    """Personen-Logik Block — wiederverwendet in foto_personen + foto_event."""
    return """PERSONENREGELN

ERLAUBT:
- Anzahl, Position, Haltung
- sichtbare Taetigkeit
- Blickrichtung
- Interaktion
- Gegenstaende aus Inventar
- Kleidungscharakter (formell, sportlich, festlich, leger)
- Namen/Funktionen bei eindeutiger Zuordnung aus Kontext oder Beschriftung

VERBOTEN:
- Altersschaetzung
- Geschlechtszuschreibung ohne Kontext
- Gesichtserkennung von Personen
- Ethnie, Religion, Gesundheit
- erfundene Beziehungen (z.B. Kolleginnen, Familie, Teilnehmer — nur wenn Kontext das belegt)
- erfundene Emotionen (z.B. gluecklich, begeistert, interessiert)
- psychologische Interpretationen"""


def _render_kontextregeln_block() -> str:
    """Kontext-Logik Block — wiederverwendet in foto_personen + foto_event."""
    return """KONTEXTREGELN

Kontext darf nur verwendet werden, wenn eindeutig zuordenbar.

BILD GEWINNT GEGEN KONTEXT:
Wenn Widerspruch besteht (z.B. Bild zeigt 2 Personen, Kontext sagt 3),
gilt das Inventar/Bild.

NAMEN-PFLICHT:
Wenn ein Name oder eine Funktion im Kontext eindeutig einer Person im
Bild zuzuordnen ist (z.B. einzige Person im Bild, oder Bildunterschrift
nennt sie eindeutig), muss der Name im Output verwendet werden.

OEFFENTLICHE PERSONEN:
Nur benennen bei bestaetigter Zuordnung aus Bildbeschriftung oder
Kontext, keine Gesichtserkennung."""


def _render_unterschriften_block() -> str:
    """Unterschriften-Block — wiederverwendet in foto_personen + foto_event."""
    return """UNTERSCHRIFTEN

Gedruckte Namen neben handschriftlichen Unterschriften duerfen verwendet
werden. Handschriftliche Unterschriften duerfen nicht selbst entziffert
werden."""


def _render_atmosphaere_block() -> str:
    """Atmosphaere-Block — wiederverwendet in foto_personen + foto_event."""
    return """ATMOSPHAERE

Wertungen ueber Atmosphaere (wirkt konzentriert, formell, lebendig)
sind nur erlaubt, wenn durch konkrete sichtbare Belege gestuetzt, die
im selben Satz oder in der Langbeschreibung explizit genannt werden.

GUT (mit Beleg):
'Die Szene wirkt konzentriert: alle blicken nach vorne, niemand
spricht miteinander.'

SCHLECHT (ohne Beleg):
'Die Atmosphaere wirkt formell, aber entspannt.'
'Eine froehliche Stimmung.'

Bei jeder Atmosphaere-Wertung MUSS atmosphaere_belege im Output gesetzt
werden mit wertung und beleg. Keine Atmosphaere ohne Beleg-Eintrag."""


def _render_unsicherheit_block() -> str:
    """Unsicherheits-Block (Hedge-Wort-Verbot) — wiederverwendet in beiden Premium-Buildern.

    Iteration 2 (Steve+ChatGPT 04.05.2026 abends): Verbotsliste erweitert um
    möglich, mögliche, denkbar, könnte sein, Art von. "Ähnelt" / "ähnlich wie"
    sind NICHT in der harten Verbotsliste, sondern werden im FINAL CHECK
    differenziert (sichtbare Form ja, Funktions-/Identitäts-Hypothese nein).
    """
    if resolve_prompt_mode() == 'lean':
        return ''
    return """UNSICHERHEIT

KEINE Hedge-Woerter und keine hypothetischen Identifikationen verwenden.

VERBOTEN (Liste):
vermutlich, wahrscheinlich, scheint, offenbar, koennte, koennte sein,
duerfte, wohl, anscheinend, moeglicherweise, moeglich, moegliche, denkbar,
"Art von"

Verboten ist auch jede Hypothesen-Liste mit oder die Funktion erfindet:
"moegliche Stimmkarten, Namensschilder oder Flyer" → SCHLECHT, weil
das Funktion vermutet die im Inventar nicht belegt ist.

Bei tatsaechlicher Unsicherheit (Inventar listet niedrige Konfidenz oder
Mehrfach-Hypothesen ohne klare Wahl): bevorzugt sichtbare Form, Farbe und
Position beschreiben. KEINE Funktion vermuten.

GUT:
- "orangefarbene rechteckige Gegenstaende"
- "nicht eindeutig erkennbare orangefarbene Gegenstaende"
- "ein rundes Objekt, das einer Tasse aehnelt" (Form-Beschreibung, ok)
- "ein flacher orangefarbener Gegenstand, der einer Karte aehnelt" (Form, ok)

SCHLECHT:
- "moegliche Stimmkarten"
- "vermutlich Namensschilder"
- "aehnelt einer Stimmkarte" (Funktions-Hypothese, schlecht)
- "aehnelt einem Flyer" (Funktions-Hypothese, schlecht)
- "Art von Karte"
"""



def _render_final_check_block() -> str:
    """Final-Check 10-Punkte-Liste — wiederverwendet in beiden Premium-Buildern.

    Iteration 2 (04.05.2026 abends): Punkt 6 verschärft, neuer Punkt 10 für
    aehnelt-Differenzierung. Ziel: das Modell prüft seine Sprache aktiv
    bevor es ausgibt, statt sich auf den Validator zu verlassen.
    """
    if resolve_prompt_mode() == 'lean':
        return _render_final_check_lean()
    return """FINAL CHECK (vor der Ausgabe pruefen):

1. Jede Aussage durch Inventar belegbar?
2. Keine Halluzination (kein Item im Output das nicht im Inventar steht)?
3. Keine Emotion erfunden (gluecklich, interessiert, engagiert)?
4. Keine Beziehung erfunden (Kolleginnen, Familie, Teilnehmer)?
5. Keine Identitaet geraten (Promi-Name ohne Kontext-Beleg)?
6. IRGENDEIN Vermutungswort oder hypothetische Objektidentifikation
   verwendet — egal ob in der expliziten Verbotsliste oder nicht?
   Konkret pruefen: vermutlich, scheint, offenbar, moeglich, moegliche,
   moeglicherweise, denkbar, koennte sein, Art von, oder eine
   Hypothesen-Liste mit oder die Funktion erfindet
   (z.B. \"moegliche Stimmkarten, Namensschilder oder Flyer\")?
   Wenn ja: ohne jede Form von Vermutung neu formulieren. Beschreibe
   nur sichtbare Form, Farbe, Position. Beispiel: statt \"moegliche
   Stimmkarten\" schreibe \"orangefarbene rechteckige Gegenstaende\".
7. Alt-Text nicht generisch (kein "Gruppe von Personen", "Auf dem Bild")?
8. Schema vollstaendig korrekt (alle Pflichtfelder gefuellt)?
9. atmosphaere_belege gefuellt wenn Wertung im Text vorkommt?
10. Falls "aehnelt" oder "aehnlich wie" verwendet wurde: beschreibt
    es eine sichtbare Form (gut, behalten — z.B. "rundes Objekt das
    einer Tasse aehnelt") oder verkleidet es eine Hypothese ueber
    Funktion oder Identitaet (schlecht, neu formulieren — z.B.
    "aehnelt einer Stimmkarte")?

Wenn ein Punkt nicht erfuellt: Output neu formulieren.
"""




def _render_final_check_lean() -> str:
    """Schlanker Final-Check fuer Sonnet — ohne Mistral-Drillregeln (Punkt 6+10)."""
    return """FINAL CHECK (vor der Ausgabe pruefen):

1. Jede Aussage durch Inventar belegbar?
2. Keine Halluzination (kein Item im Output das nicht im Inventar steht)?
3. Keine Emotion erfunden (gluecklich, interessiert, engagiert)?
4. Keine Beziehung erfunden (Kolleginnen, Familie, Teilnehmer)?
5. Keine Identitaet geraten (Promi-Name ohne Kontext-Beleg)?
6. Keine pauschalen Vermutungen formulieren wo Beleg fehlt:
   bei unklaren Objekten lieber sichtbare Form/Farbe/Position
   beschreiben als eine Funktion zu erraten.
7. Alt-Text nicht generisch (kein \"Gruppe von Personen\", \"Auf dem Bild\")?
8. Schema vollstaendig korrekt (alle Pflichtfelder gefuellt)?
9. atmosphaere_belege gefuellt wenn Wertung im Text vorkommt?

Wenn ein Punkt nicht erfuellt: Output neu formulieren.
"""

def build_beschreibung_prompt_foto_event(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder fuer foto_event — Sonnet-iteriert 13.05.2026.

    Neufassung durch ChatGPT + Steve + Claude:
    - Stil: beobachtend statt interpretierend (Sonnet-tauglich)
    - Visuelle Struktur explizit als Alt-Text-Pflicht (rotes Sofa,
      Acer-Logo werden nicht mehr in die Langbeschreibung verschoben)
    - Logo-Logik umgedreht: relevante Logos DUERFEN rein
    - Konkretes Zaehlen statt "mindestens N Personen"
    - Kompakte Personenregeln (rechtliche/ethische Absicherung)
    - Modernisierte Kontextregeln (Bild gewinnt gegen Kontext)
    - Inventar-Block sprachlich modernisiert (primaere faktische Grundlage)
    - Kontext-Quellen-Hinweis (PDF/Web/API)
    - Kein langer Mistral-Drill — Sonnet versteht das ohne 10-Punkte-Liste
    """
    examples = load_examples('foto_event')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_event
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
fuer ein Foto, das eine Veranstaltung, Gruppensituation oder soziale Szene
zeigt (Workshop, Meeting, Schulung, Praesentation, Konferenz).

Der Stil soll fluessig und lesbar sein, aber beobachtend statt
interpretierend.

Nicht beschreiben, was ein Bild "wirkt wie". Nicht interpretieren.
Nicht vermuten. Nur sichtbar belegbare Informationen verwenden.

Der Fokus liegt auf:
- visueller Orientierung
- relevanten Details
- raeumlicher Verstaendlichkeit
- praegnanter Wissensvermittlung

Der Alt-Text soll nicht nur benennen WAS zu sehen ist, sondern die
Szene mental nachvollziehbar machen.


INVENTAR (Pass-2-Beobachtungen)

Das Inventar enthaelt die strukturierten Beobachtungen aus dem
Analyse-Pass. Nutze diese Daten als primaere faktische Grundlage fuer
Alt-Text und Langbeschreibung. Sichtbare Bildinformationen duerfen
ergaenzt werden, aber nicht dem Inventar widersprechen.

{inventar_json}


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen.
Wenn kein oder nur wenig Kontext vorhanden ist, beschreibe
ausschliesslich sichtbar belegbare Bildinformationen. Fehlender Kontext
darf nicht durch Vermutungen ersetzt werden.

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}


ALT-TEXT

Der Alt-Text soll:
- konkret beginnen
- die zentrale Szene sofort verstaendlich machen
- die visuell dominantesten und orientierungsrelevantesten Elemente priorisieren

Wichtiger als allgemeine Formulierungen sind:
- auffaellige Farben
- dominante Moebel oder Raumstrukturen
- sichtbare Projektionsflaechen
- raeumliche Anordnung
- grosse oder klar sichtbare Logos/Marken
- praegende Objekte oder Materialien

Der Alt-Text beschreibt nicht nur die soziale Situation, sondern auch
die visuelle Struktur der Szene.

VERMEIDEN:
- "Das Bild zeigt"
- "Auf dem Bild"
- "Eine Szene"
- "wirkt wie"
- "im Rahmen einer Veranstaltung"
- journalistische oder erzaehlerische Sprache

BEVORZUGEN:
- konkrete Beobachtungen
- klare Hauptmotive
- sichtbare Orientierungspunkte


PERSONENZAHL

Wenn Personen klar sichtbar sind: systematisch zaehlen statt schaetzen.
"Mindestens" oder "etwa" nur verwenden, wenn Personen teilweise
verdeckt, abgeschnitten oder unscharf sind.


EVENT-LOGIK

Eine Veranstaltung darf benannt werden, wenn mindestens eines davon
sichtbar oder im Kontext eindeutig belegt ist:
- Praesentation
- Workshop-Setting
- Schulungssituation
- Moderationsmaterial
- Namensschilder
- Beamer oder Projektionsflaeche
- Buehne oder Vortragsraum
- organisierte Gruppenanordnung

Mehrere Personen allein reichen NICHT fuer foto_event.


LOGOS UND MARKEN

Sichtbare Logos oder Marken duerfen erwaehnt werden, wenn sie:
- visuell auffaellig,
- orientierungsrelevant,
- oder praegend fuer die Szene sind.

Beispiel: Ein sichtbares Acer-Logo auf einem Beamer in einer
Schulungssituation kann relevant sein.


LANGBESCHREIBUNG

Struktur in dieser Reihenfolge:

1. Gesamtueberblick
2. raeumliche Orientierung
3. Personen und Interaktion
4. zentrale Objekte oder Materialien
5. sichtbare Texte oder Logos
6. relevante Kontextinformationen

Die Langbeschreibung soll: nachvollziehbar, klar strukturiert, und
raeumlich verstaendlich sein. Nicht jede Kleinigkeit aufzaehlen —
lieber relevante Zusammenhaenge vermitteln.


PERSONENREGELN

Nur sichtbar belegbare Eigenschaften beschreiben.

Keine:
- Gesichtserkennung
- Identitaetsvermutung
- Zuschreibung von Ethnie, Religion oder Gesundheit
- psychologische Interpretation
- erfundene Beziehungen oder Emotionen

Alter oder Geschlecht nur nennen, wenn eindeutig sichtbar oder durch
Kontext belegt.


KONTEXTREGELN

Kontext darf ergaenzen, aber sichtbare Bildinformationen nicht
ueberschreiben.

Wenn Bild und Kontext widerspruechlich sind, hat das sichtbare Bild
Vorrang.

Namen oder Funktionen aus dem Kontext verwenden, wenn sie eindeutig
einer sichtbaren Person zugeordnet werden koennen.

Oeffentliche Personen nur benennen, wenn die Zuordnung durch Kontext
oder Beschriftung eindeutig belegt ist — nicht durch Gesichtserkennung.


UNTERSCHRIFTEN

Handschriftliche Unterschriften nicht selbst entziffern oder
interpretieren.


HALLUZINATIONSSCHUTZ

Beschreibe nur:
- sichtbare Inhalte
- belegbare Kontextinformationen
- lesbare Texte
- klar erkennbare raeumliche Strukturen

Wenn etwas unklar ist:
- neutral beschreiben
- sichtbare Form/Farbe/Position nennen
- keine Funktion oder Bedeutung erraten

SCHLECHT: "vermutlich", "wirkt wie", "wahrscheinlich", "eine Art von",
"moegliche Flyer", "scheint"

GUT: "orangefarbene rechteckige Gegenstaende", "runde Objekte",
"heller Projektionsbereich", "rotes Sofa im Hintergrund"


ATMOSPHAERE

Atmosphaere oder Stimmung nur beschreiben, wenn sichtbare Belege
vorhanden sind. Keine Emotionen erfinden, keine Motivation
interpretieren, keine Beziehungen annehmen.

ERLAUBT: "Die Szene wirkt konzentriert: mehrere Personen blicken
gleichzeitig zur Praesentationsflaeche."

NICHT ERLAUBT: "lockere Stimmung", "motivierte Teilnehmende",
"entspannte Atmosphaere"


SEMANTISCHE OUTPUT-REGELN

- nicht_im_inventar MUSS LEER SEIN. Steht da etwas drin, ist es eine
  Halluzination.
- atmosphaere_belege nur bei Wertung gefuellt: jede Wertung im Text
  braucht einen Eintrag mit wertung und beleg. Keine Atmosphaere ohne
  Beleg-Eintrag.


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Jede Aussage sichtbar oder im Inventar belegbar?
2. Wurde etwas interpretiert statt beobachtet?
3. Beschreibt der Alt-Text die visuell wichtigsten Elemente zuerst?
4. Ist die Szene raeumlich verstaendlich?
5. Fehlt ein auffaelliges oder orientierungsrelevantes Detail?
6. Enthaelt der Text Vermutungs- oder Interpretationssprache?
7. Ist der Alt-Text konkret statt generisch?

Wenn ein Punkt nicht erfuellt ist: neu formulieren.
"""


# =====================================================================
# Premium-Builder fuer foto_personen (refactored 04.05.2026 abends)
# =====================================================================
# Vorher: alle Bloecke als langer String inline. Jetzt: 6 Helper-
# Funktionen werden aufgerufen fuer die wiederverwendeten Bloecke.
# Inhalt soll identisch zum vor-Refactor-Stand sein (Bogart-Test
# als Sanity-Check).

def build_beschreibung_prompt_foto_personen(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder fuer foto_personen — Pilot 04.05.2026.

    Refactor 04.05.2026 abends: Wiederverwendete Bloecke (Personen,
    Kontext, Atmosphaere, Unsicherheit, Unterschriften, Final Check)
    sind jetzt in Helper-Funktionen ausgelagert. Inhalt unveraendert,
    nur Code-Struktur. Bogart-Test ist der Sanity-Check.

    Bildtyp-spezifisch sind: BILDTYP, ZIEL, ALT-TEXT-Aufbau,
    LANGBESCHREIBUNG-Reihenfolge.
    """
    examples = load_examples('foto_personen')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_personen
BILDGROESSE: {width}x{height} Pixel

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
fuer ein Foto, auf dem eine oder mehrere Personen im Mittelpunkt stehen.

ZIEL

Der Alternativtext vermittelt in einem Satz die zentrale Bildaussage.
Die Langbeschreibung erklaert die Szene vollstaendig und verstaendlich
fuer blinde Nutzer.

Der Stil darf natuerlich sein, aber alle Inhalte muessen strikt belegbar
sein. Locker im Sprachstil, streng in den Fakten.

DATENQUELLEN

Nutze ausschliesslich:
- das INVENTAR aus Pass 2 (siehe unten)
- sichtbaren Text im Bild
- eindeutig zuordenbaren Kontext (siehe unten)
- optionalen Nutzerhinweis (siehe unten)

INVENTAR AUS PASS 2

Nutze ausschliesslich diese strukturierten Daten als Grundlage:

{inventar_json}

Alles, was hier nicht enthalten ist, darf nicht beschrieben werden.

KONTEXT (zur Anreicherung)

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}

ALT-TEXT

Der erste Satz muss konkret sein und enthalten:
- Anzahl der Personen
- zentrale Handlung oder Haltung
- wichtigstes visuelles Element
- ggf. Name oder Funktion (wenn eindeutig belegbar)

Vermeide generische Einleitungen wie 'Auf dem Bild ist zu sehen',
'Das Foto zeigt', 'Eine Gruppe von Personen', 'Mehrere Menschen'.

LANGBESCHREIBUNG

Struktur in dieser Reihenfolge:
1. zentrale Person oder Gruppe
2. Haltung und Interaktion
3. Umgebung
4. Objekte
5. Text oder Kontext

{_render_personenregeln_block()}

{_render_kontextregeln_block()}

{_render_unterschriften_block()}

{_render_atmosphaere_block()}


LESBARE TEXTE IM BILD

Lesbare Texte aus inventar.lesbare_texte differenziert behandeln:
- Typ kontaktdaten, url, datum, zahl: IMMER wortgetreu im Output uebernehmen
- Typ beschriftung, ueberschrift: uebernehmen wenn fuer Bildverstaendnis relevant
- Typ logo (Markenname): nur erwaehnen wenn das Logo fuer das Bildverstaendnis
  sinnvoll ist (z.B. Mercedes-Logo bei Auto-Foto = relevant; "acer" am
  Beamer im Workshop-Foto = irrelevant, weglassen)

AUSGABE-SCHEMA

Fuelle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, prazise und konkret
- langbeschreibung: maximal 2000 Zeichen, leer wenn alt_text alles
  Wesentliche sagt
- verwendete_inventar_items: Liste der genutzten Inventar-Items
  (Audit-Trail)
- nicht_verwendete_inventar_items: Liste der bewusst ausgelassenen
  Inventar-Items
- nicht_im_inventar: MUSS LEER SEIN. Wenn doch was drin steht, ist es
  eine Halluzination die der Validator-Pass faengt.
- atmosphaere_belege: nur bei belegter Atmosphaere, jede Wertung mit
  wertung und beleg

FEW-SHOT BEISPIELE

{examples.format_for_prompt()}

{_render_unsicherheit_block()}

{_render_final_check_block()}
"""


def build_beschreibung_prompt_foto_objekte(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder fuer foto_objekte — Iteration 3 (ChatGPT 04.05.2026 spaetabends).

    Iteration 2 hatte 50/50 Compliance bei der Behaelter-Inhalte-Regel.
    Iteration 3 setzt vier Haerten-Hebel um:
    1. FINAL CHECK haerter mit expliziter Behaelter-Wort-Pruefung
    2. NEUE Sektion INNENRAUM VON BEHAELTERN mit ERLAUBT/VERBOTEN-Listen
    3. halluzinations_warnung als prominenter Block nach Inventar
    4. ALT-TEXT-spezifische harte Behaelter-Regel

    Ziel: 5/5 saubere Wiederholungstests beim Schuesselchen ohne
    Fuellung/Substanz/cremig/Fluessigkeit/Inhalt/gefuellt.
    """
    examples = load_examples('foto_objekte')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)
    halluzinations_warnungen = inventar.halluzinations_warnung if inventar.halluzinations_warnung else []
    halluzinations_block = chr(10).join(f'- {w}' for w in halluzinations_warnungen) if halluzinations_warnungen else '(keine spezifischen Warnungen)'

    return f"""{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_objekte
BILDGROESSE: {width}x{height} Pixel

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
fuer ein Foto, auf dem ein oder mehrere Gegenstaende im Mittelpunkt stehen.

ZIEL

Der Alternativtext vermittelt in einem Satz, welche zentralen Objekte
zu sehen sind und wodurch sie visuell erkennbar sind.

Die Langbeschreibung erklaert Form, Farbe, Material, Position, Anordnung
und sichtbare Details so, dass blinde Nutzer das Objekt oder die
Objektgruppe sinnvoll einordnen koennen.

Der Stil darf natuerlich sein, aber alle Inhalte muessen strikt belegbar
sein. Locker im Sprachstil, streng in den Fakten.

DATENQUELLEN

Nutze ausschliesslich:
- das INVENTAR aus Pass 2
- sichtbaren Text im Bild
- eindeutig zuordenbaren Kontext
- optionalen Nutzerhinweis

INVENTAR AUS PASS 2

Nutze ausschliesslich diese strukturierten Daten als Grundlage:

{inventar_json}

Alles, was weder im Inventar, noch im sichtbaren Bildtext, noch im
eindeutig zuordenbaren Kontext, noch im Nutzerhinweis enthalten ist,
darf nicht beschrieben werden.

HALLUZINATIONS-WARNUNGEN AUS DEM INVENTAR (Iteration 3 — kritisch beachten)

Die folgenden Warnungen sind aktiv zu beachten. Beschreibe genau diese
Punkte NICHT als Tatsache, wenn sie nicht ausdruecklich belegt sind:

{halluzinations_block}

Wenn eine Warnung sagt, dass etwas fehlinterpretiert werden kann, muss
die Beschreibung neutral bleiben und darf diese Fehlinterpretation
NICHT uebernehmen.

Beispiel: Warnung sagt 'Hellfarbene Glasur koennte als Fluessigkeit
fehlinterpretiert werden.' Dann NICHT schreiben: 'Fluessigkeit',
'Fuellung', 'Substanz' oder 'cremig'. Stattdessen: 'helle Glasur',
'helle Innenflaeche' oder 'sichtbarer Innenraum'.

KONTEXT ZUR ANREICHERUNG

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}

ALT-TEXT

Der erste Satz muss konkret sein und enthalten:
- zentrales Objekt oder zentrale Objektgruppe
- wichtigste sichtbare Form/Farbe/Position
- ggf. sichtbarer Text oder eindeutiger Kontext
- keine Funktions- oder Inhaltsvermutung

HARTE ALT-TEXT-REGEL FUER BEHAELTER (Iteration 3):
Bei Behaeltern (Schalen, Tassen, Glaeser, Teller, Dosen, Flaschen, Boxen,
Vasen, Toepfe, Becher) NIEMALS Inhalt/Fuellung/Substanz im Alt-Text
erwaehnen, ausser inventarseitig ausdruecklich als sichtbarer Inhalt
belegt. Bei unsicherem Innenraum nur Objekt, Form, Farbe, Oberflaeche
und Anordnung nennen.

Vermeide generische Einleitungen wie 'Das Foto zeigt',
'Auf dem Bild ist zu sehen', 'Ein Objekt'.

LANGBESCHREIBUNG

Struktur in dieser Reihenfolge:
1. zentrales Objekt oder zentrale Objektgruppe
2. Form, Farbe, Groesse/Proportion, Material nur wenn belegbar
3. Position und Anordnung im Bild
4. sichtbare Details, Oberflaechen, Muster, Oeffnungen, Raender
5. sichtbare Texte, Logos oder Beschriftungen nur wenn relevant
6. Kontext nur wenn eindeutig zuordenbar

OBJEKT-LOGIK

Beschreibe Gegenstaende ueber sichtbare Eigenschaften:
- Form
- Farbe
- Position
- Anordnung
- Oberflaeche
- Muster
- sichtbare Bestandteile
- erkennbare Funktion nur bei eindeutiger Belegbarkeit

INNENRAUM VON BEHAELTERN (Iteration 3 — kritischer Block)

Bei Schalen, Tassen, Glaesern, Tellern, Dosen, Flaschen, Boxen und
aehnlichen Objekten streng unterscheiden:

ERLAUBTE WOERTER:
- Innenraum
- Innenflaeche
- Glasur
- Oberflaeche
- helle Flaeche
- sichtbarer Boden
- Rand
- Vertiefung
- Farbverlauf
- Spiegelung
- Muster

VERBOTENE WOERTER (ausser ausdruecklich im Inventar als Inhalt belegt):
- Fuellung
- gefuellt
- Inhalt
- Substanz
- Fluessigkeit
- Creme
- cremig
- Pulver
- Paste
- Schaum
- Essen
- Getraenk
- Masse

GUTE FORMULIERUNGEN:
- 'Der Innenraum ist sichtbar und hell glasiert.'
- 'Die Innenflaeche erscheint hell und glatt.'
- 'Im Inneren ist eine helle Glasur oder Oberflaeche sichtbar.'
- 'Der sichtbare Innenbereich ist hell, ohne eindeutig erkennbaren Inhalt.'

SCHLECHTE FORMULIERUNGEN (NICHT verwenden):
- 'mit weisser Fuellung'
- 'cremig wirkende Substanz'
- 'mit Fluessigkeit gefuellt'
- 'enthaelt eine helle Masse'

BEHAELTER UND INHALTE (ergaenzend zur INNENRAUM-Sektion)

Inhalte duerfen NUR erwaehnt werden, wenn sie im Inventar ausdruecklich
als sichtbarer Inhalt beschrieben sind.

Wenn KEIN Inhalt im Inventar steht: NICHT schreiben 'gefuellt',
'enthaelt', 'mit Fluessigkeit', 'mit Substanz', 'mit Creme', 'mit Pulver',
'mit Essen' — stattdessen nur das Behaeltnis beschreiben.

MATERIAL

Material nur nennen, wenn im Inventar sicher oder durch Kontext eindeutig
belegt. Bei Unsicherheit: nicht 'Keramik', 'Porzellan', 'Metall', 'Glas'
raten — besser 'helles, glattes Material' oder 'glaenzende Oberflaeche'.

FUNKTION UND ZWECK

Funktion nur nennen, wenn eindeutig belegbar. Nicht aus Form allein
schliessen: keine Stimmkarte, kein Namensschild, kein Flyer, keine
Medikamentendose, keine Nahrung, kein Getraenk. Besser: 'rechteckiger
Gegenstand', 'kleines rundes Gefaess', 'flaches helles Objekt'.

{_render_kontextregeln_block()}

Zusatz fuer foto_objekte:
Wenn der Kontext sagt, dass es sich um ein Keramikschuesselchen handelt,
darf 'Keramikschuesselchen' verwendet werden, sofern das sichtbare Objekt
nicht widerspricht.
Wenn der Kontext Inhalt nennt, der im Bild nicht sichtbar und nicht im
Inventar enthalten ist, darf der Inhalt nicht beschrieben werden.


ATMOSPHAERE

Bei reinen Objektfotos normalerweise KEINE Atmosphaere beschreiben.
Nur wenn Kontext und Bildgestaltung eindeutig relevant sind, darf eine
sehr zurueckhaltende Aussage verwendet werden. Wenn Atmosphaere verwendet
wird, muss atmosphaere_belege gefuellt werden.

AUSGABE-SCHEMA

Fuelle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, prazise und konkret
- langbeschreibung: maximal 2000 Zeichen
- verwendete_inventar_items: Liste der genutzten Inventar-Items
- nicht_verwendete_inventar_items: Liste bewusst ausgelassener Inventar-Items
- nicht_im_inventar: MUSS LEER SEIN
- atmosphaere_belege: bei Objektfotos normalerweise leer

FEW-SHOT BEISPIELE

{examples.format_for_prompt()}

FINAL CHECK (Iteration 3 — foto_objekte-spezifisch, 13 Punkte):

1. Jede Aussage durch Inventar, Kontext, Bildtext oder Nutzerhinweis belegbar?
2. Keine Halluzination?
3. KRITISCH BEI BEHAELTERN: Wurden Woerter wie 'Fuellung', 'gefuellt',
   'Inhalt', 'Substanz', 'cremig', 'Fluessigkeit', 'Pulver', 'Paste',
   'Schaum', 'Masse' oder aehnliche Inhaltsbegriffe verwendet?
   Wenn ja: NUR erlaubt, wenn ein Inhalt im Inventar ausdruecklich als
   sichtbarer Inhalt belegt ist. Sonst neu formulieren als Oberflaeche,
   Innenraum, Glasur, helle Flaeche oder sichtbarer Innenbereich.
4. Kein erfundener Inhalt eines Behaelters?
5. Keine erfundene Substanz?
6. Kein geratenes Material (kein 'Keramik'/'Porzellan'/'Metall' ohne Beleg)?
7. Keine geratenen Funktionen oder Zwecke (kein 'Stimmkarte'/'Flyer'/etc.)?
8. Keine Hedge-Woerter oder Vermutungskonstruktionen (vermutlich, scheint,
   moeglich, moegliche, moeglicherweise, denkbar, koennte sein, Art von)?
9. Alt-Text konkret und nicht generisch?
10. Schema vollstaendig korrekt?
11. nicht_im_inventar leer?
12. atmosphaere_belege leer, ausser Atmosphaere wurde ausdruecklich belegt
    verwendet?
13. Wurden alle halluzinations_warnung-Eintraege aus dem Inventar respektiert
    (also nicht uebernommen als Tatsache)?

Wenn ein Punkt nicht erfuellt: Output neu formulieren.
"""



def build_beschreibung_prompt_foto_essen(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Speisen, Getränke, Tisch-Anrichtungen, Catering.

    ATMOSPHAERE_REGEL gilt EINGESCHRÄNKT — Geschmacks-Adjektive
    ('lecker'/'köstlich') sind subjektive Wertungen ohne visuelle Evidenz
    und damit verboten. Visuell belegbare Eigenschaften ('knusprige Kruste',
    'cremige Soße' wenn die Cremigkeit sichtbar ist) sind ok.
    """
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('foto_essen')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

# ATMOSPHAERE_REGEL gilt für foto_essen EINGESCHRÄNKT — die
# klassischen Geschmacks-Adjektive ('lecker', 'köstlich') sind
# subjektive Wertungen ohne visuelle Evidenz und damit verboten.
# Aber: visuell belegbare Eigenschaften sind ok ('knusprige
# Kruste', 'cremige Soße' wenn die Cremigkeit sichtbar ist).

BILDTYP: foto_essen (Speisen, Getränke, Tisch-Anrichtungen, Catering)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST FÜR foto_essen:
Der erste Satz MUSS:
- Speise / Gericht / Getränk konkret benennen wenn aus visuellem
  Inventar oder Kontext erkennbar
- Servierform (Teller, Schüssel, Tasse, Buffet, Catering-Tisch)
- Maximal 250 Zeichen

ZUTATEN — visuell belegt, nicht geraten:
- Wenn Inventar Zutaten klar listet (z.B. erkennbare Tomatenscheiben,
  Käse, Fleisch): in Output übernehmen
- Wenn Inventar unsicher: ehrliche Unsicherheit ('vermutlich Hühnchen')
- VERBOTEN: Zutaten erfinden die nicht im Inventar sind ('mit frischen
  Kräutern garniert' wenn keine Kräuter sichtbar)

VOLLSTÄNDIGKEITS-PFLICHT FÜR LANG:
1. Anrichtung und Geschirr (Material, Farbe wenn relevant)
2. Erkennbare Beilagen oder Bestandteile
3. Setting wenn relevant (Restaurant-Tisch, häuslich, Catering-
   Buffet)
4. Maximal 800 Zeichen

VERBOTEN — Geschmacks-/Wertungs-Adjektive ohne Evidenz:
- 'lecker', 'köstlich', 'delikat', 'verführerisch'
- 'appetitlich' (Wertung)
- 'frisch zubereitet' (nicht aus Bild ableitbar)

ERLAUBT — visuell belegbare Eigenschaften:
- 'knusprige Kruste' wenn Bräunung sichtbar
- 'cremige Konsistenz' wenn glänzend-weiche Oberfläche
- 'frisch geschnitten' wenn klar erkennbare Schnittflächen
- 'gedünstet/gebraten/gegrillt' wenn aus Erscheinungsbild ableitbar

KULTUR-/HERKUNFTS-IDENTIFIKATION:
- Nur wenn aus Beschriftung, Menükarte im Hintergrund oder Kontext
  belegt
- 'Italienische Pasta' nur wenn Kontext italienisch
- 'Sushi' wenn klar erkennbar (Reisbasis + Belag/Rolle)

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_foto_landschaft(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Außenfoto: Natur, Stadt-Skyline, geografische Aufnahmen."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('foto_landschaft')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{ATMOSPHAERE_REGEL}

BILDTYP: foto_landschaft (Außenfoto: Natur, Stadt-Skyline,
geografische Aufnahmen)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST FÜR foto_landschaft:
Der erste Satz MUSS:
- Geografische Charakteristik (Berge, Küste, Wald, Stadt-Skyline,
  Wüste, Fluss etc.)
- Ein konkretes Element (Wetter, Tageszeit, Jahreszeit wenn
  ableitbar, dominante Farbe, charakteristisches Bauwerk)
- Maximal 250 Zeichen

INHALTLICHE BAUSTEINE FÜR LANG:
1. Topografie (Höhen, Talsenken, Ebenen, Wasser)
2. Vegetation (Wald, Weide, kultivierte Flächen, Jahreszeit)
3. Wetter und Lichtsituation (Sonnenschein, Bewölkung, Nebel,
   Tageszeit)
4. Menschliche Eingriffe (Gebäude, Wege, Felder) wenn vorhanden
5. Maximal 800 Zeichen

ORTSNAMEN — strenge Regel:
- Wenn Schild oder Beschriftung sichtbar: wortgetreu übernehmen
- Wenn Kontext den Ort eindeutig benennt (z.B. Bildunterschrift):
  übernehmen
- SONST: KEINE Ortsspekulation. 'Bergpanorama in den Alpen' geht
  nur wenn Kontext es belegt — sonst 'Bergpanorama mit hohen Gipfeln'
- Ikonische Sichtmotive (Eiffelturm, Brandenburger Tor): erlaubt
  zu benennen wenn klar erkennbar

ATMOSPHÄRE (evidenzbasiert):
Bei Landschaftsfotos häufig relevant — aber mit Beleg.
RICHTIG: 'Die schweren Wolken und das diffuse Licht lassen
den Strand verlassen wirken.'
FALSCH: 'Eine melancholische Strandszene.'

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_foto_architektur(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Gebäude, Innenraum, Brücke, Architektur-Detail."""
    schema_doc = render_schema_for_prompt(BeschreibungOutput)
    examples = load_examples('foto_architektur')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{ATMOSPHAERE_REGEL}

{KONTAKTDATEN_PFLICHT}

{EVIDENZ_STUFEN_REGELN}

BILDTYP: foto_architektur (Gebäude, Innenraum, Brücke, Architektur-
Detail)
BILDGRÖSSE: {width}x{height} Pixel

INVENTAR (von Pass 2 erstellt — nutze AUSSCHLIESSLICH diese Items):
{inventar.model_dump_json(indent=2)}

KONTEXT:
{enriched_context}
{user_hint_block(user_hint)}

INSIGHT-FIRST FÜR foto_architektur:
Der erste Satz MUSS:
- Bautyp (Wohngebäude, Bürogebäude, Kirche, Brücke, Innenraum-Typ)
- Stilrichtung WENN klar erkennbar (modern, Bauhaus, Gotik etc.)
  ODER zentrale visuelle Charakteristik (Glasfassade, Sandsteinmauer)
- Maximal 250 Zeichen

GEBÄUDE-IDENTIFIKATION (drei Stufen wie EVIDENZ_STUFEN_REGELN):
- Stufe 1: Schild oder Beschriftung lesbar → benennen
- Stufe 2: Weltweit eindeutig + Kontext (z.B. Eiffelturm-Form,
  Brandenburger-Tor-Säulen) → benennen
- Stufe 3: Generisches Gebäude → allgemein beschreiben, NICHT raten

LESBARE BESCHRIFTUNGEN PFLICHT:
- Hausnummern, Schilder, Inschriften wortgetreu
- Architekten-/Bauherren-Tafeln
- Öffnungszeiten an Eingängen
- KONTAKTDATEN_PFLICHT für Telefonnummern, URLs

VOLLSTÄNDIGKEITS-PFLICHT FÜR LANG:
1. Material und Bauweise wenn erkennbar (Beton, Holz, Stahl, Glas)
2. Markante architektonische Elemente (Bögen, Säulen, Erker, Türme)
3. Umgebung (Stadtkontext, Park, freistehend)
4. Lichtsituation wenn relevant für die Beschreibung
5. Maximal 1000 Zeichen

ATMOSPHÄRE (evidenzbasiert):
Bei Architektur oft relevant für die Wirkung des Bauwerks.
RICHTIG: 'Die hohen Glasfassaden und der weiße Innenraum lassen
das Foyer großzügig wirken.'

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""
