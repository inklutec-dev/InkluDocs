"""Pass-3-Builder für die foto-Familie (6 Sub-Typen).

Sub-Typen werden vom Inventar-Pass entschieden — siehe BildtypEffective
im Schema-Paket und BILDTYP_INVENTAR_SCHWERPUNKTE['foto'] im inventar.py.

Alle 6 Builder folgen dem gleichen Aufbau (ROLE_BESCHREIBER war zwischen
den Refactorings 05/06-2026 versehentlich aus allen 6 Buildern herausgefallen —
am 05.07.2026 nach Fable-5-Review wieder eingesetzt, s. Desktop-Doku
Premium-Prompt-Review-Fable5.txt):
  ROLE_BESCHREIBER + ANTI_HALLUZINATION_REGELN + (ATMOSPHAERE_REGEL je nach Typ)
  + geteilte Helper-Blöcke (Personen-, Kontext-, Unterschriften-, Atmosphäre-,
    Zweck-, Kompaktheits-, Zähl-Block — siehe _render_*-Funktionen unten)
  + Bildtyp-spezifische SPEZIFITAETS-PFLICHT + VOLLSTÄNDIGKEITS-PFLICHT
  + Few-Shot + Schema-Doc
  (Paket 1, 16.07.2026: die früher hier genannten Constraint-Module
  PERSONEN_REGELN und KONTAKTDATEN_PFLICHT waren tote Importe und wurden
  entfernt — die Personen-Logik lebt in _render_personenregeln_block.)

ATMOSPHAERE_REGEL gilt für foto_essen EINGESCHRÄNKT (nur visuell belegbare
Eigenschaften, keine Geschmacks-Adjektive) — siehe Builder-Kommentar.
"""
from __future__ import annotations

from typing import Optional

from .helpers import resolve_prompt_mode

# Paket 1 (16.07.2026): tote Importe entfernt — EVIDENZ_STUFEN_REGELN,
# KONTAKTDATEN_PFLICHT und PERSONEN_REGELN wurden in keinem Prompt-String
# dieser Datei verwendet (Regel-Inventur, Strukturbefund 2).
from prompts.components.constraints import (
    ANTI_HALLUZINATION_REGELN,
    ATMOSPHAERE_REGEL,
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
# Historisch nur fuer die Premium-Builder (foto_personen, foto_event);
# seit 05.07.2026 nutzen alle 6 Foto-Builder den Zweck- und Kompaktheits-
# Block, seit Paket 2 (16.07.2026) auch den Zaehl-Block. Personen-/Kontext-/
# Unterschriften-/Atmosphaere-Block bleiben Premium-only.
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

AUSDRUECKLICH ERWUENSCHT — AUCH OHNE KONTEXT:
Dieses Werkzeug erstellt Alternativtexte fuer blinde Nutzer. Sehende erkennen
eine bekannte Persoenlichkeit auf einen Blick — blinde Nutzer haben nur deinen
Text. Das Benennen zweifelsfrei erkennbarer Personen des oeffentlichen Lebens
ist deshalb hier gewuenschter Informationszugang, KEIN Datenschutz-Verstoss:
Es geht ausschliesslich um oeffentlich bekannte Personen in ihrer oeffentlichen
Rolle. Wenn du eine solche Person zweifelsfrei erkennst, benenne sie — auch
ganz ohne Kontext oder Bildunterschrift. Vage Umschreibungen trotz eindeutiger
Erkennbarkeit ("eine Politikerin" statt des Namens) sind hier ein
Qualitaetsfehler. Bei echter Unsicherheit gilt weiter: nicht raten, neutral
beschreiben. Privatpersonen werden NIE per Gesicht identifiziert.

Nicht erfinden (Genauigkeit/Halluzinationsschutz):
- Namen oder Identitaet raten, wenn KEINERLEI Anhaltspunkt vorliegt — dann "Person"
- Ethnie, Religion oder Gesundheit (ausser explizit bildrelevant)
- psychologische Interpretation
- erfundene Beziehungen oder Emotionen

Grobe, eindeutig sichtbare Alters- und Erscheinungs-Kategorien duerfen
benannt werden (Kind, Jugendlicher, Erwachsener, aelterer Mensch; "Mann im
dunklen Anzug", "Frau im blauen Blazer") — sie machen Szenen nachvollziehbar
und sind fast immer bildrelevant. Bei echter Uneindeutigkeit: neutral
"Person". Gleiche Zwei-Wege-Logik wie bei Marken: eindeutig -> benennen,
unklar -> neutral."""
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
- praezise Alterszahlen raten (z.B. "34 Jahre alt")
- Ethnie, Religion, Gesundheit
- erfundene Beziehungen (z.B. Kolleginnen, Familie, Teilnehmer — nur wenn Kontext das belegt)
- erfundene Emotionen (z.B. gluecklich, begeistert, interessiert)
- psychologische Interpretationen

Grobe, eindeutig sichtbare Alters- und Erscheinungs-Kategorien duerfen
benannt werden (Kind, Jugendlicher, Erwachsener, aelterer Mensch; "Mann im
dunklen Anzug").
Bei echter Uneindeutigkeit: neutral "Person"."""
# Hinweis (Paket 1, 16.07.2026): Der Full-Zweig wurde an die 16.06.-Lockerung
# des Lean-Zweigs angeglichen (grobe Alters-/Erscheinungs-Kategorien erlaubt).


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


def _render_zweck_block() -> str:
    """Bild-Zweck-Block — geteilt in allen 6 Foto-Buildern (NEU 05.07.2026).

    Hintergrund: Blog-Befund Jana Wolf (via Michael Karbe, 02.07.2026) + Fable-5-
    Review Teil 5: Roh-KIs beschreiben WAS zu sehen ist, kennen aber den
    kommunikativen ZWECK des Bildes nicht. Unser Kontext wurde bisher nur als
    Fakten-Quelle genutzt (Namen, Orte) — dieser Block macht ihn zur
    Gewichtungs-Quelle. Wichtige Grenze: Der Zweck steuert die GEWICHTUNG,
    er erlaubt keine neuen unbelegten Fakten.
    """
    return """BILD-ZWECK IM DOKUMENT

Der Kontext zeigt, WO und WOZU das Bild verwendet wird. Leite daraus den
kommunikativen Zweck ab: Warum steht dieses Bild an genau dieser Stelle?
Priorisiere die Bildaspekte, die diesen Zweck bedienen — dieselbe Szene braucht
im Produktkatalog eine andere Gewichtung als im Reparatur-Handbuch oder in einer
Pressemitteilung. Der Zweck steuert nur die GEWICHTUNG und Auswahl; er erlaubt
KEINE neuen Fakten, die Bild oder Kontext nicht belegen. Ohne Kontext: neutral
informativ beschreiben.

ANTI-REDUNDANZ ZUR BILDUNTERSCHRIFT: Wiederhole keine beschreibenden Details,
die die Bildunterschrift bereits nennt — Namen, Funktionen und Identitaeten
dagegen IMMER nennen (der Alt-Text muss allein verstaendlich sein).

KONTEXT-ANREICHERUNG OHNE ERFUNDENE HANDLUNG: Der Kontext darf praezisieren,
WAS zu sehen ist ("Filiale der Drogeriekette budni"), aber keine Handlung oder
Absicht erfinden, die das Bild nicht zeigt (NICHT: "beim Einkaufen")."""


def _render_kompaktheit_block() -> str:
    """Kompaktheits-Block — geteilt in allen 6 Foto-Buildern (NEU 05.07.2026).

    Alt-Text-Straffung (Steve-Entscheid 05.07. nach Blog-Abgleich): Richtwerte
    deutlich unter dem 400er-Schema-Limit; Wissens-Tiefe wandert in die
    Langbeschreibung. WCAG-Arbeitsteilung: alt = Essenz, longdesc = Tiefe.
    """
    return """KOMPAKTHEIT (Arbeitsteilung Alt-Text / Langbeschreibung)

Richtwert fuer den Alt-Text: einfache Motive unter 150 Zeichen, komplexe Szenen
bis etwa 250. Die 400 Zeichen des Schemas sind eine harte Obergrenze, KEIN Ziel.
Der Alt-Text traegt die Essenz — Wissens-Tiefe, Nebendetails und raeumliche
Ausfuehrung gehoeren in die Langbeschreibung. Lieber ein praeziser, kurzer
Alt-Text plus dichte Langbeschreibung als ein ueberladener Alt-Text."""


def _render_zaehl_block() -> str:
    """Zaehl-Disziplin-Block — geteilt in allen 6 Foto-Buildern (Paket 2, 16.07.2026).

    Vorher lebte die Zaehlregel nur inline in foto_event (Sektion PERSONENZAHL);
    dieser Baustein ersetzt sie dort und gilt jetzt analog zum Zweck-Block fuer
    die ganze Foto-Familie. Kern: exakt zaehlen statt schaetzen; Schaetz-Woerter
    nur bei echter Verdeckung und dann mit Grund.
    """
    return """ZAEHL-DISZIPLIN

Zaehlbare Personen und Objekte bis etwa 15 exakt zaehlen und die exakte Zahl
nennen — nicht schaetzen. "Circa", "rund", "etwa" oder "mindestens" sind NUR
erlaubt, wenn sichtbare Teile echt verdeckt, abgeschnitten oder unscharf sind;
dann den Grund im Text nennen (z.B. "mindestens sieben Personen, weitere teils
verdeckt"). Bei deutlich mehr als 15 ist eine ehrliche Groessenordnung zulaessig
("ueber zwanzig Personen")."""


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

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

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


{_render_zweck_block()}


{_render_kompaktheit_block()}


ALT-TEXT

Der Alt-Text:
- beginnt mit der Art der Szene und dem charakteristischsten, orientierungs-
  relevanten Element, nicht mit einer generischen Personenzaehlung. Beispiel:
  "Workshop in hellem Seminarraum: zehn Personen nebeneinander, einige
  halten orange-weisse runde Karten; im Hintergrund Catering-Tisch und Acer-Beamer"
- priorisiert die visuell dominantesten Elemente: auffaellige Farben, praegende
  Moebel/Raumstrukturen, Projektionsflaechen, klar sichtbare Logos/Marken
- beschreibt nicht nur die soziale Situation, sondern auch die visuelle Struktur
- STRUKTURGEBENDE PERSON: Gibt es eine herausgehobene Person (moderierend,
  vortragend, der Gruppe zugewandt oder von den Blicken der Gruppe adressiert),
  gehoert sie in den ALT-TEXT — nicht nur in die Langbeschreibung. Auch eine
  Person mit Ruecken zur Kamera kann diese strukturgebende Person sein; benenne
  dann die sichtbare Beziehung (z.B. "alle blicken zu ihr").
- ist praegnant: in der Regel 1-2 Saetze, hoechstens 400 Zeichen

VERMEIDEN: "Das Bild zeigt", "Das Foto zeigt", "Auf dem Bild", "Auf dem Foto", "Zu sehen ist", "Hier sieht man", "Eine Szene", "wirkt wie",
"im Rahmen einer Veranstaltung", journalistische/erzaehlerische Sprache.


{_render_zaehl_block()}


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

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

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


{_render_zweck_block()}


{_render_kompaktheit_block()}


{_render_zaehl_block()}


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
- "Das Foto zeigt"
- "Auf dem Bild"
- "Auf dem Foto"
- "Zu sehen ist"
- "Hier sieht man"
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

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

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


{_render_zweck_block()}


{_render_kompaktheit_block()}


{_render_zaehl_block()}


ALT-TEXT

Der Alt-Text:
- beginnt mit der konkretesten belegbaren Benennung des zentralen Objekts
  (Typ/Modell/Marke/lesbare Bezeichnung), nicht mit einer generischen Umschreibung
- priorisiert die sichtbar wichtigsten, charakteristischen Eigenschaften
- macht Form und Beschaffenheit nachvollziehbar
- uebernimmt lesbaren Text und relevante Beschriftungen
- begrenzt Werbe-Claims der Verpackung auf die zwei bis drei kennzeichnendsten
  (die das Produkt identifizieren oder unterscheiden) — nicht jede Aussage
  der Verpackung abschreiben; weitere Claims gehoeren, wenn ueberhaupt,
  in die Langbeschreibung

VERMEIDEN: generische Einleitungen ("Das Bild zeigt", "Das Foto zeigt", "Auf dem Bild", "Auf dem Foto", "Zu sehen ist", "Hier sieht man"), blosse Inventarlisten, vage Umschreibungen
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

Beginne NICHT mit "Das Bild zeigt", "Das Foto zeigt", "Auf dem Bild", "Auf dem Foto", "Zu sehen ist" oder "Hier sieht man" — steige direkt mit dem Objekt ein.

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
    """Premium-Builder fuer foto_essen — Mistral-Altlast abgespeckt 30.06.2026.

    Vorher Standard-Stand aus der Mistral-Zeit: ROLE_BESCHREIBER im String,
    starre INSIGHT-FIRST-MUSS-Liste, nummerierte VOLLSTAENDIGKEITS-PFLICHT und
    widerspruechliche Zeichen-Caps (250/800 vs. Schema 400/2000). Auf das
    Premium-Muster von foto_objekte gehoben: self-contained, ANTI_HALLUZINATION
    als geteilte Schicht voran, inline ZIEL/AUSGABE-SCHEMA, Few-Shot. Kategorie-
    spezifisch behalten: Geschmacks-Adjektiv-Bann (subjektive Wertung ohne
    visuelle Evidenz) und die Zutaten-Evidenzregel (keine erfundenen Zutaten).
    Reversibel: Backup .bak-pre-fotofamilie-20260630.
    """
    examples = load_examples('foto_essen')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)
    halluzinations_warnungen = inventar.halluzinations_warnung if inventar.halluzinations_warnung else []
    halluzinations_block = chr(10).join(f'- {w}' for w in halluzinations_warnungen) if halluzinations_warnungen else '(keine spezifischen Warnungen)'

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_essen
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung fuer
ein Foto, auf dem Speisen, Gerichte, Getraenke, Tisch-Anrichtungen oder Catering
im Mittelpunkt stehen. Ziel ist dichte, faktenbasierte Wissensvermittlung —
praezise und auf den Punkt, beobachtend statt wertend.

Fuehre mit der Art des Gerichts oder der Speise. Benenne sichtbare Komponenten
und Zutaten selbstbewusst, WENN sie klar erkennbar sind (z.B. "gebratener Lachs
mit gruenem Spargel", "Cappuccino mit Milchschaum-Muster"). Was nicht klar
erkennbar ist, beschreibst du neutral nach Aussehen (z.B. "helle Soße"), statt
es zu raten. Du erfindest keine nicht sichtbaren Zutaten, keine Zubereitung und
keine Rezeptur.

Bei Produkten/Lebensmitteln aus einem Shop: nenne Marke/Hersteller, wenn sie auf
Verpackung oder Etikett sichtbar oder das Produkt eindeutig erkennbar ist. Farben
sind oft wichtig — nenne sie. Halte den Text KOMPAKT; nicht jedes Detail
ausschreiben.


INVENTAR (Pass-2-Beobachtungen)

Nutze diese strukturierten Beobachtungen als primaere faktische Grundlage.
Sichtbare Bildinformationen duerfen ergaenzen, dem Inventar aber nicht
widersprechen.

{inventar_json}


HALLUZINATIONS-WARNUNGEN AUS DEM INVENTAR
(falls vorhanden — beachten, nicht als Tatsache uebernehmen)

{halluzinations_block}


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen. Ohne
Kontext beschreibst du ausschliesslich sichtbar belegbare Bildinformationen.
BILD GEWINNT GEGEN KONTEXT: bei Widerspruch hat das sichtbare Bild Vorrang.

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}


{_render_zweck_block()}


{_render_kompaktheit_block()}


{_render_zaehl_block()}


ALT-TEXT

Der Alt-Text:
- beginnt mit der konkretesten belegbaren Benennung des Gerichts/der Speise und
  der Servierform (Teller, Schuessel, Tasse, Glas, Buffet, Catering-Tisch), nicht
  mit einer generischen Einleitung
- benennt die klar erkennbaren Hauptkomponenten und Zutaten selbstbewusst
- macht die Anrichtung visuell nachvollziehbar
- nennt Marke/Hersteller, wenn sichtbar oder eindeutig erkennbar
- uebernimmt lesbaren Text (Menuekarte, Beschriftung) wenn relevant
- ist so KOMPAKT wie moeglich: in der Regel 1-2 Saetze; das Zeichenlimit ist
  Obergrenze, KEIN Ziel — nimm nur, was zum Verstehen noetig ist

VERMEIDEN: "Das Bild zeigt", "Das Foto zeigt", "Auf dem Foto", "Zu sehen ist", "Auf dem Teller befindet sich", blosse
Inventarlisten, vage Umschreibungen fuer klar Benennbares, sowie mikroskopische
Details (Poren, Lentizellen, einzelne Maserungen) — die gehoeren nicht in einen
kompakten Alt-Text.


ZUTATEN — BENENNEN STATT VAGE, ABER NICHTS ERFINDEN

Benenne sichtbare Komponenten und Zutaten, wenn Inventar oder klar erkennbares
Aussehen sie belegen — z.B. Tomatenscheiben, geriebener Kaese, gruener Spargel,
ein Spiegelei, eine Zitronenspalte. Weiche nur bei echter Unsicherheit auf eine
rein visuelle Beschreibung aus ("helle Soße", "gruenes Blattgemuese", "eine
cremige Komponente") — nicht aus Prinzip vage bleiben.

NICHT erfinden:
- Zutaten, die nicht sichtbar belegt sind (z.B. "mit frischen Kraeutern
  garniert", wenn keine Kraeuter sichtbar sind)
- Rezeptur oder Zubereitung einer Komponente, deren Zusammensetzung nicht
  sichtbar ist (z.B. "hausgemachte Zitronen-Butter-Sauce" — sichtbar ist nur
  eine helle Soße)


GESCHMACK UND WERTUNG

Geschmacks- und Wertungs-Adjektive sind ohne visuelle Evidenz VERBOTEN, weil
subjektiv und aus dem Bild nicht ableitbar: "lecker", "koestlich", "delikat",
"verfuehrerisch", "appetitlich", "frisch zubereitet".

ERLAUBT sind visuell belegbare Eigenschaften:
- "knusprige Kruste", wenn eine Braeunung sichtbar ist
- "cremige Konsistenz", wenn eine glaenzend-weiche Oberflaeche sichtbar ist
- "frisch geschnitten", wenn klare Schnittflaechen sichtbar sind
- "gebraten", "gegrillt", "gedaempft", wenn aus dem Erscheinungsbild ableitbar


HERKUNFT UND KULTUR

Eine kulturelle oder geografische Einordnung ("italienische Pasta", "japanisches
Sushi") nur, wenn sie durch Beschriftung, Menuekarte im Bild oder Kontext belegt
ist — oder wenn das Gericht visuell zweifelsfrei einer Form entspricht (z.B.
Sushi an Reisbasis und Rolle/Belag klar erkennbar). Erfinde keine Herkunft, kein
Restaurant und keinen Anlass, die nicht belegt sind.


LANGBESCHREIBUNG

Schreibe FLIESSTEXT — keine Markdown-Formatierung, keine Ueberschriften, keine
Aufzaehlungszeichen. Beginne NICHT mit "Das Bild zeigt", "Das Foto zeigt", "Auf dem Foto", "Zu sehen ist" oder "Auf dem Teller".
Sinnvolle Reihenfolge ohne sie als Ueberschriften zu setzen:
Gericht (konkret benannt), sichtbare Hauptkomponenten und Beilagen, Anrichtung
und Geschirr (Material/Farbe wenn relevant), Setting wenn relevant (Restaurant-
Tisch, haeuslich, Catering-Buffet), sichtbare Texte. Vermittle Zusammenhaenge,
zaehle nicht jede Kleinigkeit auf — keine Poren, keine einzelnen Maserungen;
konzentriere dich auf das Wesentliche und halte es kompakt.


ATMOSPHAERE

Bei Speisefotos normalerweise KEINE Atmosphaere. Nur wenn Bildgestaltung und
Kontext es eindeutig tragen, eine zurueckhaltende atmosphaerische Aussage — dann
MUSS atmosphaere_belege mit wertung und beleg gesetzt werden. Geschmacks- und
Genuss-Wertungen sind hier KEINE zulaessige Atmosphaere.


AUSGABE-SCHEMA

Fuelle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, praezise und konkret
- langbeschreibung: maximal 2000 Zeichen, leer wenn der Alt-Text alles
  Wesentliche sagt
- verwendete_inventar_items: Audit-Trail der genutzten Inventar-Items
- nicht_verwendete_inventar_items: Audit-Trail der bewusst ausgelassenen Items
- nicht_im_inventar: MUSS LEER SEIN. Steht dort etwas, ist es eine Halluzination.
- atmosphaere_belege: bei foto_essen normalerweise leer


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}

FINAL CHECK (vor der Ausgabe pruefen):

1. Fuehrt der Text mit der Art des Gerichts/der Speise und der Servierform —
   statt mit einer generischen Einleitung?
2. Sind klar erkennbare Komponenten konkret benannt, Unklares neutral nach
   Aussehen beschrieben (keine geratene Zutat)?
3. Keine erfundene Zutat, Garnierung, Rezeptur oder Zubereitung?
4. Kein Geschmacks-/Wertungsadjektiv ohne visuelle Evidenz?
5. Keine erfundene Herkunft, kein erfundenes Restaurant, kein erfundener Anlass?
6. nicht_im_inventar leer, und vorhandene halluzinations_warnung-Eintraege
   beachtet?

Wenn ein Punkt nicht erfuellt ist: Output neu formulieren.
"""

def build_beschreibung_prompt_foto_landschaft(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder fuer foto_landschaft — Mistral-Altlast abgespeckt 30.06.2026.

    Vorher Standard-Stand: ROLE_BESCHREIBER + ATMOSPHAERE_REGEL vorangestellt,
    starre 'INSIGHT-FIRST … MUSS'-Liste, nummerierte Bausteine-Pflicht und
    widerspruechliche Zeichen-Caps (250/800 vs. Schema 400/2000). Auf das
    Premium-Muster von foto_objekte gehoben: self-contained, ANTI_HALLUZINATION
    als geteilte Schicht voran, inline ZIEL/AUSGABE-SCHEMA, Few-Shot. Kategorie-
    spezifisch: keine erfundenen Ortsnamen/Regionen/Gipfel — ikonische
    Sichtmotive (Eiffelturm, Brandenburger Tor) SOLLEN bei eindeutiger
    Erkennbarkeit benannt werden (Paket 2, 16.07.2026: von 'duerfen' auf das
    SOLL-Niveau von foto_architektur gehoben — Koelner-Dom-Fall der
    Regel-Inventur). Reversibel: Backup .bak-pre-fotofamilie-20260630.
    """
    examples = load_examples('foto_landschaft')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)
    halluzinations_warnungen = inventar.halluzinations_warnung if inventar.halluzinations_warnung else []
    halluzinations_block = chr(10).join(f'- {w}' for w in halluzinations_warnungen) if halluzinations_warnungen else '(keine spezifischen Warnungen)'

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_landschaft
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung fuer
ein Aussenfoto, auf dem eine Landschaft oder ein geografischer Raum im
Mittelpunkt steht (Kueste, Gebirge, Wald, Feld, Fluss, Wueste, Stadtpanorama,
Skyline). Ziel ist dichte, faktenbasierte Wissensvermittlung — praezise,
beobachtend statt stimmungsmalend, und so KOMPAKT wie moeglich.

Fuehre mit der Art der Landschaft und benenne ihre praegenden Merkmale so
konkret, wie das Sichtbare und das Inventar sie hergeben (Relief, Gewaesser,
Vegetation, Bebauung, Licht). Was lesbar ist (Orts- oder Wegschilder), wird
uebernommen. Erfinde keinen Ortsnamen, keine Region, keinen Berg- oder
Gewaessernamen und keine Jahreszeit, die nicht belegt sind.


INVENTAR (Pass-2-Beobachtungen)

Nutze diese strukturierten Beobachtungen als primaere faktische Grundlage.
Sichtbare Bildinformationen duerfen ergaenzen, dem Inventar aber nicht
widersprechen.

{inventar_json}


HALLUZINATIONS-WARNUNGEN AUS DEM INVENTAR
(falls vorhanden — beachten, nicht als Tatsache uebernehmen)

{halluzinations_block}


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen. Ohne
Kontext beschreibst du ausschliesslich sichtbar belegbare Bildinformationen.
BILD GEWINNT GEGEN KONTEXT: bei Widerspruch hat das sichtbare Bild Vorrang.

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}


{_render_zweck_block()}


{_render_kompaktheit_block()}


{_render_zaehl_block()}


ALT-TEXT

Der Alt-Text:
- beginnt mit der Art der Landschaft (Kueste, Gebirge, Wald, Feld, Skyline usw.)
  und einem konkreten praegenden Merkmal (dominante Form, Gewaesser, Wetter/
  Licht wenn klar erkennbar), nicht mit einer generischen Einleitung
- benennt die belegten geografischen Hauptelemente und ihre Anordnung
- macht den Raum und die Tiefe der Szene nachvollziehbar
- uebernimmt lesbaren Text (Orts-/Wegschilder) wenn relevant
- ist so KOMPAKT wie moeglich: in der Regel 1-2 Saetze; das Zeichenlimit ist
  Obergrenze, KEIN Ziel — nimm nur, was zum Verstehen noetig ist

VERMEIDEN: "Das Bild zeigt", "Das Foto zeigt", "Auf dem Bild", "Auf dem Foto", "Zu sehen ist", "Hier sieht man", generische Einleitungen, blosse
Inventarlisten, vage Umschreibungen fuer klar Benennbares.


ORTE UND BENENNUNG — BENENNEN STATT RATEN

Benenne die Landschaftsart und ihre Merkmale, wenn sie visuell belegt sind —
Kuestenlinie, schneebedeckte Gipfel, dichter Nadelwald, terrassierte Felder,
Hochhaus-Skyline. Beschreibe Wetter, Tageszeit oder Jahreszeit nur, wenn das
Erscheinungsbild sie klar traegt (kahle Baeume, Schnee, langer Schattenwurf,
warmes Abendlicht).

NICHT erfinden — nur bei Schild- oder Kontext-Beleg nennen:
- konkreter Ortsname, Region oder Land (kein geratenes "die Alpen", "Toskana")
- Eigenname eines Berges, Sees, Flusses oder einer Stadt
- eine Jahreszeit, die nicht sichtbar belegt ist

Eindeutig erkennbare ikonische Motive mit unverwechselbarer Silhouette
(Eiffelturm, Brandenburger Tor, Golden Gate Bridge, Koelner Dom) SOLLEN beim
Namen genannt werden — eine vage Umschreibung trotz eindeutiger Erkennbarkeit
("ein grosser Torbau" statt Brandenburger Tor) ist ein Qualitaetsfehler. Bei
echter Unsicherheit auf die reine sichtbare Beschreibung ausweichen
("Bergpanorama mit hohen, schneebedeckten Gipfeln" statt "die Alpen") — nicht
raten, aber auch nicht aus Prinzip vage bleiben, wenn die Landschaftsart klar
belegt ist.


LANGBESCHREIBUNG

Schreibe FLIESSTEXT — keine Markdown-Formatierung, keine Ueberschriften, keine
Aufzaehlungszeichen. Beginne NICHT mit "Das Bild zeigt", "Das Foto zeigt", "Auf dem Bild", "Auf dem Foto", "Zu sehen ist" oder "Hier sieht man".
Folge inhaltlich dieser Reihenfolge, ohne sie als
Ueberschriften zu setzen: zuerst Landschaftsart und Gesamtraum (Vorder-, Mittel-,
Hintergrund, Tiefe), dann Topografie (Hoehen, Senken, Ebenen, Gewaesser), dann
Vegetation und Bodennutzung (Wald, Weide, Felder), dann Wetter und Licht
(Bewoelkung, Nebel, Tageszeit), dann menschliche Eingriffe (Gebaeude, Wege,
Bruecken) wenn vorhanden, zuletzt lesbare Beschriftungen und Kontext. Mache den
Raum mental nachvollziehbar, statt jede Kleinigkeit aufzuzaehlen.


ATMOSPHAERE

Bei Landschaftsfotos ist eine atmosphaerische Aussage haeufig relevant — aber
nur, wenn durch konkrete sichtbare Belege gestuetzt, die im selben Satz genannt
werden. Bei jeder Atmosphaere-Wertung MUSS atmosphaere_belege mit wertung und
beleg gesetzt werden.
GUT (mit Beleg): "Die schweren Wolken und das diffuse Licht lassen den Strand
verlassen wirken."
SCHLECHT (ohne Beleg): "Eine melancholische Strandszene."


AUSGABE-SCHEMA

Fuelle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, praezise und konkret
- langbeschreibung: maximal 2000 Zeichen, leer wenn der Alt-Text alles
  Wesentliche sagt
- verwendete_inventar_items: Audit-Trail der genutzten Inventar-Items
- nicht_verwendete_inventar_items: Audit-Trail der bewusst ausgelassenen Items
- nicht_im_inventar: MUSS LEER SEIN. Steht dort etwas, ist es eine Halluzination.
- atmosphaere_belege: nur bei belegter Atmosphaere, jede Wertung mit wertung und
  beleg


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}

FINAL CHECK (vor der Ausgabe pruefen):

1. Fuehrt der Alt-Text mit der Art der Landschaft und einem konkreten Merkmal —
   statt generischer Einleitung?
2. Sind die geografischen Hauptelemente konkret benannt, Unklares neutral nach
   Aussehen beschrieben?
3. Kein erfundener Ortsname, keine erfundene Region, kein erfundener Berg-/
   Gewaessername, keine unbelegte Jahreszeit?
4. Ist jede Aussage durch Bild oder Inventar belegt (keine Halluzination)?
5. Atmosphaere nur mit Beleg im selben Satz (atmosphaere_belege gesetzt)?
6. nicht_im_inventar leer, und vorhandene halluzinations_warnung-Eintraege
   beachtet?

Wenn ein Punkt nicht erfuellt ist: Output neu formulieren.
"""

def build_beschreibung_prompt_foto_architektur(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder fuer foto_architektur — Mistral-Altlast abgespeckt 30.06.2026.

    Vorher Standard-Stand: ROLE_BESCHREIBER + ATMOSPHAERE_REGEL +
    KONTAKTDATEN_PFLICHT + EVIDENZ_STUFEN_REGELN vorangestellt, dazu ein harter
    'drei Stufen'-Drill und 'INSIGHT-FIRST … MUSS'-Pflichtsaetze mit
    widerspruechlichen Zeichen-Caps. Auf das Premium-Muster von foto_objekte
    gehoben: self-contained, ANTI_HALLUZINATION als geteilte Schicht voran,
    inline ZIEL/AUSGABE-SCHEMA, Few-Shot. Steve-Vorgabe 30.06.: bekannte
    Wahrzeichen ausdruecklich BEIM NAMEN nennen (Modellwissen nutzen), bei
    unbekannten Bauten die FUNKTION erschliessen (Reithalle, Lagerhalle) statt
    nur 'ein Gebaeude' — kompakt halten. Reversibel: .bak-pre-fotofamilie-20260630.
    """
    examples = load_examples('foto_architektur')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)
    halluzinations_warnungen = inventar.halluzinations_warnung if inventar.halluzinations_warnung else []
    halluzinations_block = chr(10).join(f'- {w}' for w in halluzinations_warnungen) if halluzinations_warnungen else '(keine spezifischen Warnungen)'

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

BILDTYP: foto_architektur
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung fuer
ein Foto, auf dem ein Gebaeude, Bauwerk, Innenraum oder Architektur-Detail im
Mittelpunkt steht (Wohnhaus, Buerogebaeude, Kirche, Bruecke, Hochhaus, Halle,
Innenraum, Fassaden-Ausschnitt). Ziel ist dichte, faktenbasierte
Wissensvermittlung — praezise, beobachtend, und so KOMPAKT wie moeglich.

Fuehre mit dem Namen, wenn das Bauwerk ein bekanntes, eindeutig erkennbares
Wahrzeichen ist — trau dich, dein Wissen ueber bekannte Architektur zu nutzen
(z.B. Brandenburger Tor, Koelner Dom, Eiffelturm). Ist kein eindeutiges
Wahrzeichen erkennbar, schliesse aus dem Sichtbaren auf Bautyp und FUNKTION
(z.B. Reithalle, Lagerhalle, Bahnhofshalle, Buerogebaeude) — auch ohne Kontext.
Erfinde nur keine FALSCHE konkrete Identitaet (keinen geratenen Namen fuer ein
generisches Gebaeude), keinen erfundenen Architekten und kein erfundenes Baujahr.


INVENTAR (Pass-2-Beobachtungen)

Nutze diese strukturierten Beobachtungen als primaere faktische Grundlage.
Sichtbare Bildinformationen duerfen ergaenzen, dem Inventar aber nicht
widersprechen.

{inventar_json}


HALLUZINATIONS-WARNUNGEN AUS DEM INVENTAR
(falls vorhanden — beachten, nicht als Tatsache uebernehmen)

{halluzinations_block}


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen. Ohne
Kontext beschreibst du ausschliesslich sichtbar belegbare Bildinformationen.
BILD GEWINNT GEGEN KONTEXT: bei Widerspruch hat das sichtbare Bild Vorrang.

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}


{_render_zweck_block()}


{_render_kompaktheit_block()}


{_render_zaehl_block()}


ALT-TEXT

Der Alt-Text:
- beginnt mit dem NAMEN, wenn es ein bekanntes Wahrzeichen ist; sonst mit dem
  Bautyp bzw. der erschlossenen FUNKTION und der zentralen visuellen
  Charakteristik (z.B. Glasfassade, Backsteinmauer, geschwungenes Dach)
- benennt knapp die belegten Materialien und die markantesten architektonischen
  Merkmale — nicht jedes Detail, nur das Charakteristische
- uebernimmt lesbaren Text und relevante Beschriftungen
- ist so KOMPAKT wie moeglich: in der Regel 1-2 Saetze. Das Zeichenlimit ist eine
  Obergrenze, KEIN Ziel — nimm nur, was zum Verstehen noetig ist

VERMEIDEN: "Das Bild zeigt", "Das Foto zeigt", "Auf dem Bild", "Auf dem Foto", "Zu sehen ist", "Hier sieht man", generische Einleitungen, blosse
Inventarlisten, das Auslisten jeder Saeule/jedes Fensters.


BENENNEN — TRAU DICH, ABER ERFINDE NICHTS FALSCHES

Nenne ein bekanntes Bauwerk BEIM NAMEN, wenn es eindeutig erkennbar ist — nutze
dafuer dein Wissen ueber bekannte Architektur (Brandenburger Tor, Koelner Dom,
Eiffelturm, Sydney Opera House, Reichstag usw.). Das ist ausdruecklich erwuenscht
und fuer blinde Nutzer wertvoll.

Ist kein eindeutiges Wahrzeichen erkennbar, schliesse aus dem Sichtbaren auf den
Bautyp und die FUNKTION (Reithalle an Sandboden und Bande, Lagerhalle an Toren
und Stahlbau, Kirche an Turm und Portal, Bahnhof an Bahnsteigen und Hallendach) —
auch ohne Kontext. Benenne ebenso belegte Materialien und Bauweise; eine Stil-
Epoche nur, wenn eindeutig belegt.

NICHT erfinden: einen konkreten Eigennamen fuer ein Gebaeude, das du NICHT
eindeutig erkennst; einen Architekten, ein Baujahr oder eine Stil-Epoche, die
nicht belegt sind. Der Unterschied: ein eindeutig erkanntes Wahrzeichen benennen
= richtig und erwuenscht; einem beliebigen Bau einen beruehmten Namen andichten
= falsch.


LESBARE BESCHRIFTUNGEN

Lesbare Texte am Bauwerk wortgetreu uebernehmen, wenn fuer Orientierung oder
Bildverstaendnis relevant: Hausnummern, Strassennamen, Inschriften, Bau- oder
Architekten-Tafeln. Telefonnummern, URLs und Adressen (z.B. an einem Ladenlokal)
immer wortgetreu uebernehmen.


LANGBESCHREIBUNG

Schreibe FLIESSTEXT — keine Markdown-Formatierung, keine Ueberschriften, keine
Aufzaehlungszeichen. Beginne NICHT mit "Das Bild zeigt", "Das Foto zeigt", "Auf dem Bild", "Auf dem Foto", "Zu sehen ist" oder "Hier sieht man".
Halte auch die Langbeschreibung kompakt: Bauwerkstyp/Name
und Gesamtform, dann Fassade/Material, dann die markantesten Elemente (Dachform,
Saeulen, Tuerme), dann die Einbettung in die Umgebung, zuletzt lesbare
Beschriftungen. Mache die Bauform mental nachvollziehbar, ohne jede Saeule und
jedes Fenster einzeln aufzuzaehlen.


ATMOSPHAERE

Eine atmosphaerische Aussage nur, wenn durch konkrete sichtbare Belege gestuetzt,
die im selben Satz genannt werden. Bei jeder Atmosphaere-Wertung MUSS
atmosphaere_belege mit wertung und beleg gesetzt werden.
GUT (mit Beleg): "Die hohen Glasfassaden und der weisse, stuetzenfreie Innenraum
lassen das Foyer grosszuegig wirken."
SCHLECHT (ohne Beleg): "Ein imposantes, ehrwuerdiges Gebaeude."


AUSGABE-SCHEMA

Fuelle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, praezise und KOMPAKT (Limit nicht ausreizen)
- langbeschreibung: maximal 2000 Zeichen, leer wenn der Alt-Text alles
  Wesentliche sagt
- verwendete_inventar_items: Audit-Trail der genutzten Inventar-Items
- nicht_verwendete_inventar_items: Audit-Trail der bewusst ausgelassenen Items
- nicht_im_inventar: MUSS LEER SEIN. Steht dort etwas, ist es eine Halluzination.
- atmosphaere_belege: nur bei belegter Atmosphaere, jede Wertung mit wertung und
  beleg


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}

FINAL CHECK (vor der Ausgabe pruefen):

1. Bekanntes Wahrzeichen beim Namen genannt, falls eindeutig erkennbar?
2. Bei unbekanntem Bau die FUNKTION erschlossen (z.B. Reithalle, Lagerhalle)
   statt nur "ein Gebaeude"?
3. Keine FALSCHE konkrete Identitaet, kein erfundener Architekt, kein erfundenes
   Baujahr, keine unbelegte Stil-Epoche?
4. So kompakt wie moeglich — Limit nicht ausgereizt, kein Auslisten jedes
   Details?
5. Lesbare Beschriftungen und Kontaktdaten wortgetreu uebernommen?
6. nicht_im_inventar leer, und vorhandene halluzinations_warnung-Eintraege
   beachtet?

Wenn ein Punkt nicht erfuellt ist: Output neu formulieren.
"""
