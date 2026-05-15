# Premium-Builder foto_event — Prompt-Modus: full

- **Builder:** `prompts/builders/beschreibung_foto.py:313`
- **Generiert:** 2026-05-15
- **ENV / Modus:**
  - `V4_PROMPT_MODE` = `full`
  - `LLM_PROVIDER` = `mistral`
- **Demo-Werte:**
  - width × height: 1280 × 720
  - inventar: Workshop-Setting (4 Personen, Beamer, Catering)

---

```text
ANTI-HALLUZINATIONS-REGELN (höchste Priorität):

1. EVIDENZ-BASIERT: Eine Aussage darf nur dann im Output stehen, wenn das Inventar sie stützt.
   Plausibel klingen reicht NICHT. 'Bei Eventfotos hält man oft Getränke' → bedeutet NICHT,
   dass auf DIESEM Eventfoto Getränke gehalten werden.

2. EHRLICHE UNSICHERHEIT IST PFLICHT, NICHT VERSAGEN: Wenn das Inventar ein Item mit
   Sicherheit 'niedrig' oder mehreren möglichen Identifikationen aufführt, dann wird
   diese Unsicherheit im Output sprachlich abgebildet. Beispiele:
   - 'orangefarbene Gegenstände, deren Funktion nicht eindeutig erkennbar ist' OK
   - 'vermutlich Stimmkarten' NICHT (Hedge-Wort statt ehrlicher Beschreibung)
   - 'Stimmkarten' NICHT (falsche Sicherheit)

3. KEINE INTERAKTIONS-GESCHICHTEN: Wenn das Inventar nur 'Hund-Cartoon' + 'Laptop' listet,
   schreibe nicht 'Hund arbeitet am Laptop'. Du erfindest eine Handlung. Erlaubt: 'Hund-
   Cartoon, daneben ein Laptop.' Punkt.

4. KEINE SPEZIES-/MARKEN-SPEKULATION: Wenn Inventar 'stilisiertes Tier mit großen Augen,
   gelb-schwarz, Spezies unklar' sagt, schreibe NICHT 'Katze' oder 'Hund' sondern 'Tier'
   oder die im Inventar gelistete Mehrfach-Hypothese.

BILDTYP: foto_event
BILDGROESSE: 1280x720 Pixel

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

{
  "foto_subtyp": "foto_event",
  "personen": [
    {
      "position": "vorn links",
      "haltung": "stehend",
      "blickrichtung": "zur Praesentation",
      "objekte_in_haenden": [],
      "kleidungs_charakter": "Business-casual"
    },
    {
      "position": "Mitte",
      "haltung": "stehend",
      "blickrichtung": "zur Kamera",
      "objekte_in_haenden": [],
      "kleidungs_charakter": "Business-casual"
    },
    {
      "position": "hinten rechts",
      "haltung": "sitzend",
      "blickrichtung": "zur Praesentation",
      "objekte_in_haenden": [],
      "kleidungs_charakter": "legere Kleidung"
    },
    {
      "position": "Mitte rechts",
      "haltung": "stehend",
      "blickrichtung": null,
      "objekte_in_haenden": [],
      "kleidungs_charakter": "Business-casual"
    }
  ],
  "objekte": [
    {
      "beschreibung": "Projektionsflaeche mit hellem Lichtkegel",
      "position": "Hintergrund Mitte",
      "sicherheit": "hoch",
      "moegliche_identifikationen": [
        "Beamer-Projektion"
      ]
    },
    {
      "beschreibung": "rechteckige weisse Karten an Personen befestigt",
      "position": "auf Brusthoehe der Personen",
      "sicherheit": "hoch",
      "moegliche_identifikationen": [
        "Namensschilder"
      ]
    },
    {
      "beschreibung": "Tisch mit Getraenkeflaschen und Glaesern",
      "position": "rechter Bildrand",
      "sicherheit": "hoch",
      "moegliche_identifikationen": [
        "Catering-Tisch"
      ]
    }
  ],
  "lesbare_texte": [
    {
      "inhalt": "acer",
      "typ": "logo",
      "vollstaendigkeit": "vollständig"
    },
    {
      "inhalt": "Workshop Inklusion 2026",
      "typ": "überschrift",
      "vollstaendigkeit": "vollständig"
    }
  ],
  "setting": {
    "raum_charakter": "Seminarraum",
    "beleuchtung": "gedaempft, Projektionslicht",
    "dominante_farben": "blau, weiss, grau",
    "ungefaehre_szene": "Vortragssituation mit Publikum"
  },
  "handlung": "Praesentation vor stehendem und sitzendem Publikum",
  "halluzinations_warnung": [
    "Namensschilder nicht lesbar — keine Identifikationen ableiten.",
    "Karten an Personen nicht als Stimmkarten/Flyer interpretieren."
  ],
  "inventar_konfidenz_gesamt": "hoch"
}


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen.
Wenn kein oder nur wenig Kontext vorhanden ist, beschreibe
ausschliesslich sichtbar belegbare Bildinformationen. Fehlender Kontext
darf nicht durch Vermutungen ersetzt werden.

Workshop-Bericht: Inklusion in der digitalen Arbeitswelt. Am 5. Mai 2026 fand bei INKLUTEC ein eintaegiger Workshop zur barrierefreien Software-Entwicklung statt. Teilnehmende waren Entwickler:innen aus drei Partnerunternehmen.



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
- psychologische Interpretationen


KONTEXTREGELN

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
Kontext, keine Gesichtserkennung.


UNTERSCHRIFTEN

Gedruckte Namen neben handschriftlichen Unterschriften duerfen verwendet
werden. Handschriftliche Unterschriften duerfen nicht selbst entziffert
werden.


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
werden mit wertung und beleg. Keine Atmosphaere ohne Beleg-Eintrag.


SEMANTISCHE OUTPUT-REGELN

nicht_im_inventar MUSS LEER SEIN. Steht da etwas drin, ist es eine
Halluzination.


FEW-SHOT BEISPIELE

(Noch keine Few-Shot-Beispiele für Bildtyp "foto_event" kuratiert.)


FINAL CHECK (vor der Ausgabe pruefen):

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
   (z.B. "moegliche Stimmkarten, Namensschilder oder Flyer")?
   Wenn ja: ohne jede Form von Vermutung neu formulieren. Beschreibe
   nur sichtbare Form, Farbe, Position. Beispiel: statt "moegliche
   Stimmkarten" schreibe "orangefarbene rechteckige Gegenstaende".
7. Alt-Text nicht generisch (kein "Gruppe von Personen", "Auf dem Bild")?
8. Schema vollstaendig korrekt (alle Pflichtfelder gefuellt)?
9. atmosphaere_belege gefuellt wenn Wertung im Text vorkommt?
10. Falls "aehnelt" oder "aehnlich wie" verwendet wurde: beschreibt
    es eine sichtbare Form (gut, behalten — z.B. "rundes Objekt das
    einer Tasse aehnelt") oder verkleidet es eine Hypothese ueber
    Funktion oder Identitaet (schlecht, neu formulieren — z.B.
    "aehnelt einer Stimmkarte")?

Wenn ein Punkt nicht erfuellt: Output neu formulieren.


```
