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


# GELOCKERT 16.06.2026 (Steve): Personen-Identifikation bewusst entschaerft.
# Erkennbare Personen (oeffentliches Leben + per Kontext/Schild/Bildunterschrift
# zuordenbar) duerfen benannt werden; pauschaler Gesichtserkennungs-/
# Identitaetsvermutungs-Bann entfernt, nur Halluzinationsschutz bleibt.
# REVERSIBEL: bei Bedarf Privatpersonen-Schutz wieder verschaerfen.
# Greift in foto_personen + foto_event (Lean=live auf Prod/Staging, Full angeglichen).
def _render_personenregeln_block() -> str:
    """Personen-Logik Block — wiederverwendet in foto_personen + foto_event + foto_objekte.

    Modus-aware: lean = ChatGPT-Stand 15.05.2026 (Alter/Geschlecht bei eindeutiger
    Sichtbarkeit + Bildrelevanz erlaubt, moderner Wortlaut). full = Mistral-Drill.
    """
    if resolve_prompt_mode() == 'lean':
        return """PERSONENREGELN

Personen so vollstaendig und informativ wie moeglich beschreiben.
Erkennbare Personen duerfen benannt werden.

Erlaubt:
- sichtbare Haltung, Position, Blickrichtung
- sichtbare Taetigkeit oder Interaktion
- Kleidungscharakter (formell, sportlich, festlich, leger)
- Gegenstaende aus Inventar
- Namen und Funktionen aus Kontext, Beschriftung oder Bildunterschrift
- erkennbare Personen benennen — Personen des oeffentlichen Lebens
  (Politiker, Staats- und Regierungschefs, bekannte Sportler/Kuenstler)
  ebenso wie Personen, die durch Kontext, Namensschild oder Beschriftung
  zuzuordnen sind

Nicht erfinden (Genauigkeit/Halluzinationsschutz):
- Namen oder Identitaet raten, wenn KEINERLEI Anhaltspunkt vorliegt — dann "Person"
- Ethnie, Religion oder Gesundheit (ausser explizit bildrelevant)
- psychologische Interpretation
- erfundene Beziehungen oder Emotionen

Alter oder Geschlecht nur nennen, wenn eindeutig sichtbar UND fuer die
Bildaussage relevant."""
    return """PERSONENREGELN

ERLAUBT:
- Anzahl, Position, Haltung
- sichtbare Taetigkeit
- Blickrichtung
- Interaktion
- Gegenstaende aus Inventar
- Kleidungscharakter (formell, sportlich, festlich, leger)
- Namen/Funktionen aus Kontext, Beschriftung oder Bildunterschrift
- erkennbare Personen benennen — Personen des oeffentlichen Lebens
  (Politiker, Staats- und Regierungschefs, bekannte Sportler/Kuenstler)
  ebenso wie durch Kontext/Namensschild/Beschriftung zuordenbare Personen

NICHT ERFINDEN (Genauigkeit/Halluzinationsschutz):
- Namen oder Identitaet raten, wenn KEINERLEI Anhaltspunkt vorliegt — dann "Person"
- Altersschaetzung
- Geschlechtszuschreibung ohne Kontext
- Ethnie, Religion, Gesundheit
- erfundene Beziehungen (z.B. Kolleginnen, Familie, Teilnehmer — nur wenn Kontext das belegt)
- erfundene Emotionen (z.B. gluecklich, begeistert, interessiert)
- psychologische Interpretationen"""


def _render_kontextregeln_block() -> str:
    """Kontext-Logik Block — wiederverwendet in foto_personen + foto_event + foto_objekte.

    Modus-aware: lean = ChatGPT-Stand 15.05.2026 mit Bogart-Beispiel. full = wie vorher.
    """
    if resolve_prompt_mode() == 'lean':
        return """KONTEXTREGELN

Kontext darf ergaenzen, aber sichtbare Bildinformationen nicht
ueberschreiben.

BILD GEWINNT GEGEN KONTEXT:
Wenn Bild und Kontext widerspruechlich sind, hat das sichtbare Bild
Vorrang.

NAMEN-PFLICHT:
Namen oder Funktionen aus dem Kontext verwenden, wenn sie eindeutig
einer sichtbaren Person zugeordnet werden koennen.

Beispiel: Wenn die Bildunterschrift "Humphrey Bogart in CASABLANCA (1942)"
lautet und nur eine Person sichtbar ist, soll der Name verwendet werden.

PERSONEN BENENNEN:
Erkennbare Personen duerfen benannt werden — Personen des oeffentlichen
Lebens auch ohne Bildunterschrift. Liegt ein Name aus Kontext, Beschriftung
oder Bildunterschrift vor, ist er zu verwenden. Nur wenn gar kein
Anhaltspunkt vorliegt: "Person"."""
    return """KONTEXTREGELN

Kontext darf nur verwendet werden, wenn eindeutig zuordenbar.

BILD GEWINNT GEGEN KONTEXT:
Wenn Widerspruch besteht (z.B. Bild zeigt 2 Personen, Kontext sagt 3),
gilt das Inventar/Bild.

NAMEN-PFLICHT:
Wenn ein Name oder eine Funktion im Kontext eindeutig einer Person im
Bild zuzuordnen ist (z.B. einzige Person im Bild, oder Bildunterschrift
nennt sie eindeutig), muss der Name im Output verwendet werden.

PERSONEN BENENNEN:
Erkennbare Personen duerfen benannt werden — Personen des oeffentlichen
Lebens auch ohne Bildbeschriftung. Liegt ein Name aus Bildbeschriftung
oder Kontext vor, ist er zu verwenden. Nur ohne jeden Anhaltspunkt: "Person"."""


def _render_unterschriften_block() -> str:
    """Unterschriften-Block — wiederverwendet in foto_personen + foto_event + foto_objekte.

    Modus-aware: lean = kompakt (ChatGPT-Stand 15.05.2026). full = wie vorher.
    """
    if resolve_prompt_mode() == 'lean':
        return """UNTERSCHRIFTEN

Gedruckte Namen oder Beschriftungen duerfen verwendet werden.
Handschriftliche Unterschriften nicht selbst entziffern oder
interpretieren."""
    return """UNTERSCHRIFTEN

Gedruckte Namen neben handschriftlichen Unterschriften duerfen verwendet
werden. Handschriftliche Unterschriften duerfen nicht selbst entziffert
werden."""


def _render_atmosphaere_block() -> str:
    """Atmosphaere-Block — wiederverwendet in foto_personen + foto_event + foto_objekte.

    Modus-aware: lean = ChatGPT-Stand 15.05.2026 (Belege-Pflicht erhalten,
    Wortlaut moderner, Beispiel passend zu Sonnet). full = wie vorher.
    """
    if resolve_prompt_mode() == 'lean':
        return """ATMOSPHAERE

Atmosphaerische Aussagen sind erlaubt, wenn sie durch sichtbare Belege
gestuetzt werden. Der Beleg muss im selben Satz genannt werden UND
zusaetzlich im Feld atmosphaere_belege gesetzt sein.

GUT (mit Beleg):
'Die Szene wirkt konzentriert: alle Personen blicken zur Projektion.'

SCHLECHT (ohne Beleg):
'Die Atmosphaere wirkt locker und motiviert.'
'Eine froehliche Stimmung.'

Keine Emotionen erfinden, keine Motivation interpretieren, keine
Beziehungen annehmen. Bei jeder Atmosphaere-Wertung MUSS
atmosphaere_belege im Output gesetzt werden mit wertung und beleg.
Keine Atmosphaere ohne Beleg-Eintrag."""
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
    """Schlanker Final-Check fuer Sonnet — ChatGPT-Stand 15.05.2026.

    Verschmolzen aus Mittwoch-foto_event-Inline + ChatGPT-heute-Material:
    - Wissensvermittlung-Lehre eingebaut (Punkt 6: konkret + visuell charakteristisch)
    - Punkt 5 stellt Form-Beschreibung explizit ueber Funktions-Raten
    - kompakt 9 Punkte, kein Mistral-Drill (kein Hedge-Wort-Listen-Check)
    """
    return """FINAL CHECK (vor der Ausgabe pruefen):

1. Jede Aussage durch Inventar oder sichtbare Bildinformation belegbar?
2. Keine Halluzination (kein Item im Output das weder im Inventar noch sichtbar belegt ist)?
3. Keine Emotion oder Beziehung erfunden (gluecklich, motiviert, Kolleginnen, Familie)?
4. Keine Identitaet geraten ohne Kontext-Beleg?
5. Bei unklaren Objekten: sichtbare Form/Farbe/Position beschrieben statt Funktion zu erraten?
6. Alt-Text konkret und visuell charakteristisch — nicht nur Personen- oder Inventar-Aufzaehlung?
7. Vermeidet generische Einleitungen (\"Auf dem Bild\", \"Eine Gruppe von Personen\")?
8. Schema vollstaendig korrekt (alle Pflichtfelder gefuellt)?
9. atmosphaere_belege gefuellt wenn Atmosphaere im Text vorkommt?

Wenn ein Punkt nicht erfuellt: Output neu formulieren.
"""

def build_beschreibung_prompt_foto_event(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder fuer foto_event — Auf-den-Punkt + Anti-Markdown 30.06.2026.

    Aenderung gegenueber 13.05.2026-Stand (Befund Querschnitt-Test 30.06.):
    - LANG erzeugte Markdown-Ueberschriften (**Gesamtueberblick**), weil die
      Struktur als nummerierte Abschnittsnamen vorgegeben war. Jetzt: gleiche
      inhaltliche Reihenfolge, aber explizit FLIESSTEXT, keine Ueberschriften.
    - Alt-Text oeffnete generisch ("Etwa zehn Personen...") und lief zu lang
      (>400 Zeichen, Schema-Retry). Jetzt: fuehrt mit Szenen-Art + charakter-
      istischem Element, praegnant, hoechstens 400 Zeichen.
    - Helfer-Bloecke + ANTI_HALL + alle API-Variablen unveraendert.
    Reversibel: Builder-Backup .bak-pre-eventfix-20260630.
    """
    examples = load_examples('foto_event')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_event
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung fuer
ein Foto, das eine Veranstaltung, Gruppensituation oder soziale Szene zeigt
(Workshop, Meeting, Schulung, Praesentation, Konferenz). Ziel ist dichte,
faktenbasierte Wissensvermittlung — praezise, auf den Punkt, beobachtend statt
interpretierend. Nur sichtbar belegbare Informationen; nicht vermuten, nicht
"wirkt wie". Der Text soll die Szene mental nachvollziehbar machen: Art der
Veranstaltung, raeumliche Orientierung, praegende visuelle Elemente.


INVENTAR (Pass-2-Beobachtungen)

Das Inventar enthaelt die strukturierten Beobachtungen aus dem Analyse-Pass.
Nutze diese Daten als primaere faktische Grundlage. Sichtbare
Bildinformationen duerfen ergaenzt werden, aber nicht dem Inventar
widersprechen.

{inventar_json}


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen. Ohne
Kontext beschreibst du ausschliesslich sichtbar belegbare Bildinformationen;
fehlender Kontext wird nicht durch Vermutungen ersetzt.

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}


ALT-TEXT

Der Alt-Text:
- beginnt mit der Art der Szene und dem charakteristischsten, orientierungs-
  relevanten Element, nicht mit einer generischen Personenzaehlung. Beispiel:
  "Workshop in hellem Seminarraum: rund zehn Personen nebeneinander, einige
  halten orange-weisse runde Karten; im Hintergrund Catering-Tisch und Acer-Beamer"
- priorisiert die visuell dominantesten Elemente: auffaellige Farben, praegende
  Moebel/Raumstrukturen, Projektionsflaechen, klar sichtbare Logos/Marken
- beschreibt nicht nur die soziale Situation, sondern auch die visuelle Struktur
- ist praegnant: in der Regel 1-2 Saetze, hoechstens 400 Zeichen

VERMEIDEN: "Das Bild zeigt", "Auf dem Bild", "Eine Szene", "wirkt wie",
"im Rahmen einer Veranstaltung", journalistische/erzaehlerische Sprache.


PERSONENZAHL

Wenn Personen klar sichtbar sind: systematisch zaehlen statt schaetzen.
"Mindestens" oder "etwa" nur, wenn Personen teilweise verdeckt, abgeschnitten
oder unscharf sind.


EVENT-LOGIK

Eine Veranstaltung darf benannt werden, wenn mindestens eines sichtbar oder im
Kontext belegt ist: Praesentation, Workshop-Setting, Schulungssituation,
Moderationsmaterial, Namensschilder, Beamer/Projektionsflaeche, Buehne/
Vortragsraum, organisierte Gruppenanordnung. Mehrere Personen allein reichen NICHT.


LOGOS UND MARKEN

Sichtbare Logos/Marken duerfen erwaehnt werden, wenn sie visuell auffaellig,
orientierungsrelevant oder praegend fuer die Szene sind (z.B. ein Acer-Logo auf
einem Beamer in einer Schulung).


LANGBESCHREIBUNG

Schreibe FLIESSTEXT — keine Markdown-Formatierung, keine Ueberschriften, keine
Aufzaehlungszeichen, keine fettgedruckten Abschnittstitel. Folge inhaltlich
dieser Reihenfolge, ohne sie als Ueberschriften zu setzen: zuerst ein
Gesamtueberblick, dann die raeumliche Orientierung, dann Personen und
Interaktion, dann zentrale Objekte/Materialien, dann sichtbare Texte/Logos,
zuletzt relevante Kontextinformationen. Nachvollziehbar und raeumlich
verstaendlich — nicht jede Kleinigkeit aufzaehlen, lieber Zusammenhaenge
vermitteln.


{_render_personenregeln_block()}


{_render_kontextregeln_block()}


{_render_unterschriften_block()}


HALLUZINATIONSSCHUTZ

Beschreibe nur sichtbare Inhalte, belegbare Kontextinformationen, lesbare Texte
und klar erkennbare raeumliche Strukturen. Wende die Zwei-Wege-Regel an: klar
durch Form UND Setting Getragenes wird benannt, genuin Unklares neutral
beschrieben (Form/Farbe/Position) — nie hedgen.

SCHLECHT (Hedging): "vermutlich", "wirkt wie", "wahrscheinlich", "eine Art von",
"moegliche Flyer", "scheint"
GUT: Klar Getragenes benennen ("orange und weisse Abstimmkarten" im Workshop-
Setting, "Acer-Logo", "Projektionsflaeche"); genuin Unklares neutral ("runde
orangefarbene Gegenstaende, Funktion nicht erkennbar", "rotes Sofa im Hintergrund").


{_render_atmosphaere_block()}


SEMANTISCHE OUTPUT-REGELN

nicht_im_inventar MUSS LEER SEIN. Steht da etwas drin, ist es eine Halluzination.
Der Alt-Text umfasst hoechstens 400 Zeichen.


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


{_render_final_check_block()}
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

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
fuer ein Foto, auf dem eine oder mehrere Personen im Mittelpunkt stehen
(Portraet, Gruppe, Einzelperson in Situation).

Der Stil soll fluessig und lesbar sein, aber beobachtend statt
interpretierend. Nicht beschreiben, was eine Person "wirkt wie".
Nicht Motivation, Beziehungen oder Emotionen vermuten. Nur sichtbar
belegbare Informationen verwenden.

Der Fokus liegt auf:
- visueller Charakterisierung der Person(en)
- Haltung, Blickrichtung, Konstellation
- praegenden visuellen Markern (Kleidung, Hut, charakteristische Objekte)
- praegnanter Wissensvermittlung

Der Alt-Text soll nicht nur benennen WER zu sehen ist, sondern die
Person und ihre sichtbare Situation mental nachvollziehbar machen.


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
- die Person(en) und ihre sichtbare Situation sofort verstaendlich machen
- die visuell dominantesten und orientierungsrelevantesten Elemente priorisieren

Wichtige Bestandteile (wenn sichtbar oder durch Kontext belegt):
- Anzahl der Personen
- zentrale Haltung, Handlung oder Blickrichtung
- praegende visuelle Marker (Kleidung, Hut, charakteristische Objekte)
- praegnante Hintergrund- oder Raumelemente
- Name oder Funktion bei eindeutiger Zuordnung

NAMEN-PFLICHT (Erinnerung):
Wenn der Kontext eine Person eindeutig benennt (z.B. Bildunterschrift
"Humphrey Bogart in CASABLANCA, 1942" und nur eine Person sichtbar),
muss der Name im Alt-Text auftauchen — nicht nur in der Langbeschreibung.

VERMEIDEN:
- "Das Bild zeigt"
- "Auf dem Bild"
- "Eine Gruppe von Personen"
- "Mehrere Menschen"
- "wirkt wie"
- erzaehlerische oder journalistische Einleitungen

BEVORZUGEN:
- konkrete sichtbare Beobachtungen
- praezise Charakterisierung
- visuelle Orientierungspunkte


LANGBESCHREIBUNG

Struktur in dieser Reihenfolge:

1. zentrale Person(en): Anzahl, sichtbare Identifikation, Konstellation
2. Haltung, Blickrichtung, sichtbare Taetigkeit
3. praegende visuelle Marker (Kleidung, Objekte, Hut)
4. Umgebung und Raumwirkung
5. relevante Texte, Logos oder Kontextinformationen

Die Langbeschreibung soll nachvollziehbar und klar strukturiert sein.
Nicht jede Kleinigkeit aufzaehlen — lieber relevante Zusammenhaenge
und visuelle Charakteristika vermitteln.


{_render_personenregeln_block()}


{_render_kontextregeln_block()}


{_render_unterschriften_block()}


{_render_atmosphaere_block()}


LESBARE TEXTE IM BILD

Lesbare Texte aus inventar.lesbare_texte differenziert behandeln:
- Typ kontaktdaten, url, datum, zahl: IMMER wortgetreu im Output uebernehmen
- Typ beschriftung, ueberschrift: uebernehmen wenn fuer Bildverstaendnis relevant


LOGOS UND MARKEN

Sichtbare Logos oder Marken duerfen erwaehnt werden, wenn sie:
- visuell auffaellig
- orientierungsrelevant
- oder praegend fuer die Szene sind

Bei foto_personen sind Logos relevant, wenn sie z.B. Beruf oder
Veranstaltungsort einer Person charakterisieren (Firmen-Polo, Konferenz-
Lanyard, Beamer-Logo im Hintergrund eines Schulungsfotos).

Nicht relevant: Logos die nur klein und am Rand auftauchen ohne
szenenpraegende Wirkung.


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
    """Premium-Builder fuer foto_objekte — Mistral-Altlast-Abspeckung 30.06.2026.

    Hintergrund: Die schweren Behaelter-/Material-Klammern stammen aus der
    Mistral-Zeit (pixtral erfand Schuessel-Inhalte). A/B-Sonde 30.06. mit
    Sonnet 4.6 belegt: das Modell erfindet keine Inhalte mehr. Daher abgespeckt:
    - Riesige VERBOTEN-Wortliste -> kurze Evidenz-Regel (Inhalt nur wenn belegt)
    - MATERIAL-/FUNKTION-Sperre gelockert -> benennen wenn belegt, vage nur bei
      echter Unsicherheit
    - NEU: "fuehre mit der konkretesten belegbaren Benennung" (Typ/Modell/Marke/
      lesbare Bezeichnung) gegen die beobachtete Passivitaet (Flugzeug-Befund 25.06.)
    - Schema, Inventar-Warnungen, Bild-gewinnt-Kontext, alle API-Variablen erhalten.
    Reversibel: Builder-Backup .bak-pre-objektelockern-20260630.
    """
    examples = load_examples('foto_objekte')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)
    halluzinations_warnungen = inventar.halluzinations_warnung if inventar.halluzinations_warnung else []
    halluzinations_block = chr(10).join(f'- {w}' for w in halluzinations_warnungen) if halluzinations_warnungen else '(keine spezifischen Warnungen)'

    return f"""{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_objekte
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung fuer
ein Foto, auf dem Gegenstaende, Materialien oder Objektgruppen im Mittelpunkt
stehen. Ziel ist dichte, faktenbasierte Wissensvermittlung — praezise und auf
den Punkt, nicht banale Aufzaehlung.

Benenne das Objekt so konkret, wie es das Sichtbare und das Inventar hergeben:
Typ, Modell, Marke, Bauart. Was lesbar ist (Schriftzuege, Typenschilder,
Beschriftungen), wird uebernommen. Wo eine konkrete Benennung belegt ist,
beginnt der Text damit — nicht mit einer generischen Umschreibung.


INVENTAR (Pass-2-Beobachtungen)

Das Inventar enthaelt strukturierte Beobachtungen aus dem Analyse-Pass.
Nutze diese Daten als primaere faktische Grundlage. Sichtbare
Bildinformationen duerfen ergaenzt werden, duerfen dem Inventar aber nicht
widersprechen.

{inventar_json}


HALLUZINATIONS-WARNUNGEN AUS DEM INVENTAR
(falls vorhanden — beachten)

Die folgenden Warnungen beschreiben bekannte Fehlinterpretations-Risiken fuer
DIESES Bild. Uebernimm sie nicht als Tatsache:

{halluzinations_block}


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen. Ohne
Kontext beschreibst du ausschliesslich sichtbar belegbare Bildinformationen;
fehlender Kontext wird nicht durch Vermutungen ersetzt.

BILD GEWINNT GEGEN KONTEXT: Bei Widerspruch zwischen Bild und Kontext hat das
sichtbare Bild Vorrang.

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}


ALT-TEXT

Der Alt-Text:
- beginnt mit der konkretesten belegbaren Benennung des zentralen Objekts
  (Typ/Modell/Marke/lesbare Bezeichnung), nicht mit einer generischen Umschreibung
- priorisiert die sichtbar wichtigsten, charakteristischen Eigenschaften
- macht Form und Beschaffenheit nachvollziehbar
- uebernimmt lesbaren Text und relevante Beschriftungen

VERMEIDEN: generische Einleitungen, blosse Inventarlisten, vage Umschreibungen
fuer eindeutig Benennbares.


BENENNEN STATT VAGE BLEIBEN

Benenne Material, Typ und Bauart, wenn sie visuell oder kontextuell hinreichend
belegt sind — z.B. Keramik an Glasur und Form, "Boeing 777" am Schriftzug, eine
Airline an Logo und Lackierung. Weiche nur bei echter Unsicherheit auf eine rein
visuelle Beschreibung aus ("helles glattes Material", "glaenzende Oberflaeche") —
nicht aus Prinzip. Vage zu bleiben, obwohl etwas klar belegt ist, ist ein Fehler.


INHALTE VON BEHAELTERN (Evidenz-Regel)

Bei Behaeltern (Schalen, Tassen, Glaesern, Flaschen, Dosen, Vasen u.ae.):
Inhalte oder Fuellungen nur nennen, wenn das Inventar sie als sichtbaren Inhalt
belegt. Ist nur der Innenraum sichtbar, beschreibe Innenflaeche, Glasur,
Oberflaeche, Boden, Struktur oder Spiegelung — aber erfinde keinen Inhalt
(keine "Fuellung", "Fluessigkeit", "Substanz" oder "cremige Masse" ohne Beleg).


LANGBESCHREIBUNG

Sinnvolle Reihenfolge: zentrales Objekt (konkret benannt) -> Form und Proportion
-> Oberflaeche, Struktur, Material -> raeumliche Anordnung -> sichtbare Details
und Beschriftungen -> relevanter Kontext. Die Langbeschreibung soll die sichtbare
Form mental nachvollziehbar machen, nicht bloss Eigenschaften aufzaehlen.


ATMOSPHAERE

Bei Objektfotos normalerweise KEINE Atmosphaere. Nur wenn Bildgestaltung und
Kontext es eindeutig tragen, eine zurueckhaltende atmosphaerische Aussage —
dann MUSS atmosphaere_belege gesetzt werden.


AUSGABE-SCHEMA

Fuelle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, praezise und konkret
- langbeschreibung: maximal 2000 Zeichen
- verwendete_inventar_items: Audit-Trail der genutzten Inventar-Items
- nicht_verwendete_inventar_items: Audit-Trail der bewusst ausgelassenen Items
- nicht_im_inventar: MUSS leer bleiben
- atmosphaere_belege: bei foto_objekte normalerweise leer


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Ist das zentrale Objekt so konkret benannt, wie Beleg/Inventar es zulassen
   (Typ/Modell/Marke/lesbare Bezeichnung) — statt vager Umschreibung?
2. Ist jede Aussage durch Bild oder Inventar belegt (keine Halluzination)?
3. Behaelter-Inhalt nur genannt, wenn als sichtbarer Inhalt belegt?
4. nicht_im_inventar leer?
5. Wurden vorhandene halluzinations_warnung-Eintraege beachtet?

Wenn ein Punkt nicht erfuellt ist: Output neu formulieren.
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
