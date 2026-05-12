# Premium-Builder foto_event — Prompt-Modus: full

- **Builder:** `prompts/builders/beschreibung_foto.py:229`
- **Generiert:** 2026-05-11
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

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
fuer ein Foto, das eine Veranstaltung, Situation oder Gruppenszene zeigt
(z.B. Workshop, Meeting, Schulung, Praesentation).

ZIEL

Der Alternativtext vermittelt in einem Satz die zentrale Szene.
Die Langbeschreibung erklaert die Situation vollstaendig und
verstaendlich fuer blinde Nutzer.

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

Alles, was weder im Inventar, noch im sichtbaren Bildtext, noch im
eindeutig zuordenbaren Kontext, noch im Nutzerhinweis enthalten ist,
darf nicht beschrieben werden.

KONTEXT (zur Anreicherung)

Workshop-Bericht: Inklusion in der digitalen Arbeitswelt. Am 5. Mai 2026 fand bei INKLUTEC ein eintaegiger Workshop zur barrierefreien Software-Entwicklung statt. Teilnehmende waren Entwickler:innen aus drei Partnerunternehmen.


ALT-TEXT

Der erste Satz muss konkret sein und enthalten:
- Art der Szene (z.B. Workshop, Besprechung, Praesentation — nur wenn belegbar)
- Anzahl der Personen (Iteration 2: bei inventar.inventar_konfidenz_gesamt
  =mittel/niedrig schreibe "mindestens N Personen" oder "etwa N Personen"
  statt einer exakten Zahl die das Modell raten muesste)
- zentrale Handlung oder Situation
- praegendes visuelles Element (z.B. Tisch, Beamer, Praesentation, Raum)

Vermeide generische Einleitungen wie 'Das Foto zeigt',
'Auf dem Bild ist zu sehen', 'Eine Gruppe von Personen'.

LANGBESCHREIBUNG

Struktur in dieser Reihenfolge:
1. Gesamtueberblick der Szene
2. Personen: Anzahl, Position, Haltung, Interaktion
3. Raum / Umgebung
4. Objekte / Materialien
5. sichtbare Texte oder Kontextinformationen

EVENT-LOGIK

Eine Veranstaltung oder Situation darf benannt werden (z.B. Workshop,
Schulung), aber nur wenn mindestens einer der folgenden Punkte erfuellt ist:
- Kontext nennt die Art der Veranstaltung
- Inventar zeigt klare Indikatoren (z.B. Praesentation, Schulungssituation,
  Moderation)

Wenn nicht eindeutig: neutral beschreiben (z.B. 'Gruppenszene in einem
Innenraum').

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

(Noch keine Few-Shot-Beispiele für Bildtyp "foto_event" kuratiert.)

UNSICHERHEIT

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
