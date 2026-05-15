# Premium-Builder foto_objekte — Prompt-Modus: lean

- **Builder:** `prompts/builders/beschreibung_foto.py:696`
- **Generiert:** 2026-05-15
- **ENV / Modus:**
  - `V4_PROMPT_MODE` = `lean`
  - `LLM_PROVIDER` = `bedrock`
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

BILDTYP: foto_objekte
BILDGROESSE: 1280x720 Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
fuer ein Foto, auf dem Gegenstaende, Materialien oder Objektgruppen im
Mittelpunkt stehen.

Der Fokus liegt auf sichtbarer Beschaffenheit:
Form, Oberflaeche, Struktur, Anordnung, Materialwirkung und raeumliche
Wirkung sollen nachvollziehbar vermittelt werden.

Wissensvermittlung statt reine Objekt-Aufzaehlung:
Der Text soll helfen, das Objekt mental zu erfassen —
nicht nur Gegenstaende zu benennen.

Beschreibe nur sichtbar belegbare Eigenschaften.
Keine Funktions-, Inhalts- oder Materialvermutungen ohne Beleg.


INVENTAR (Pass-2-Beobachtungen)

Das Inventar enthaelt strukturierte Beobachtungen aus dem Analyse-Pass.
Nutze diese Daten als primaere faktische Grundlage.

Sichtbare Bildinformationen duerfen ergaenzt werden,
duerfen dem Inventar aber nicht widersprechen.

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


HALLUZINATIONS-WARNUNGEN AUS DEM INVENTAR
(KRITISCH — aktiv beachten)

Die folgenden Warnungen beschreiben bekannte Fehlinterpretations-Risiken.
Diese Fehlinterpretationen duerfen NICHT als Tatsache uebernommen werden:

- Namensschilder nicht lesbar — keine Identifikationen ableiten.
- Karten an Personen nicht als Stimmkarten/Flyer interpretieren.

Wenn eine Warnung sagt,
dass eine Oberflaeche oder Innenflaeche als Inhalt fehlinterpretiert werden koennte,
muss die Beschreibung neutral bleiben.

BEISPIEL:

Warnung:
'Hellfarbene Glasur koennte als Fluessigkeit fehlinterpretiert werden.'

ERLAUBT:
- 'helle Innenflaeche'
- 'sichtbarer Innenraum'
- 'helle Glasur'
- 'glaenzende Oberflaeche'

NICHT erlaubt:
- 'Fluessigkeit'
- 'Fuellung'
- 'Substanz'
- 'cremig'


KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt oder API-Aufrufen stammen.

Wenn kein oder nur wenig Kontext vorhanden ist,
beschreibe ausschliesslich sichtbar belegbare Bildinformationen.

Fehlender Kontext darf niemals durch Vermutungen ersetzt werden.

BILD GEWINNT GEGEN KONTEXT:
Wenn Bild und Kontext widerspruechlich sind, hat das sichtbare Bild Vorrang.

Wenn der Kontext sagt, dass es sich um ein Keramikschuesselchen handelt,
darf 'Keramikschuesselchen' verwendet werden, sofern das sichtbare Objekt
nicht widerspricht. Inhalte duerfen trotzdem nur beschrieben werden,
wenn sie sichtbar oder im Inventar belegt sind.

Workshop-Bericht: Inklusion in der digitalen Arbeitswelt. Am 5. Mai 2026 fand bei INKLUTEC ein eintaegiger Workshop zur barrierefreien Software-Entwicklung statt. Teilnehmende waren Entwickler:innen aus drei Partnerunternehmen.



ALT-TEXT

Der Alt-Text soll:
- direkt mit dem zentralen Objekt beginnen
- die sichtbar wichtigsten Eigenschaften priorisieren
- Form und Beschaffenheit nachvollziehbar machen
- visuell charakteristische Merkmale enthalten

Wichtige Bestandteile:
- zentrales Objekt oder Objektgruppe
- Form und Proportion
- Oberflaeche, Muster oder Struktur
- raeumliche Anordnung
- Material nur wenn belegbar
- sichtbarer Text oder relevante Beschriftungen

VERMEIDEN:
- generische Einleitungen
- blosse Inventarlisten
- Funktionsvermutungen
- Inhaltsvermutungen


HARTE ALT-TEXT-REGEL FUER BEHAELTER
(Schuesselchen-Schutz — verpflichtend)

Bei Behaeltern wie:
Schalen, Schuesseln, Tassen, Glaesern, Tellern, Dosen, Flaschen,
Boxen, Vasen, Toepfen oder Bechern

duerfen Inhalte oder Fuellungen NUR erwaehnt werden,
wenn sie im Inventar ausdruecklich als sichtbarer Inhalt belegt sind.

Wenn der Innenraum sichtbar,
aber kein Inhalt eindeutig belegt ist:

Beschreibe nur:
- Innenraum
- Innenflaeche
- Glasur
- Oberflaeche
- sichtbaren Boden
- Spiegelung
- Farbverlauf
- Struktur
- Muster

NICHT verwenden:
- Fuellung
- gefuellt
- Inhalt
- Fluessigkeit
- Substanz
- cremig
- Creme
- Paste
- Pulver
- Schaum
- Masse
- enthaelt
- Essen
- Getraenk

GUTE FORMULIERUNGEN:
- 'helle glaenzende Innenflaeche'
- 'sichtbarer Innenraum mit heller Glasur'
- 'der Innenbereich wirkt glatt und hell'
- 'sichtbarer Boden des Gefaesses'

SCHLECHTE FORMULIERUNGEN:
- 'mit heller Fluessigkeit gefuellt'
- 'cremig wirkende Substanz'
- 'enthaelt eine weisse Masse'


LANGBESCHREIBUNG

Struktur:

1. zentrales Objekt oder Objektgruppe
2. Form und Proportion
3. Oberflaeche, Struktur, Muster oder Materialwirkung
4. raeumliche Anordnung
5. sichtbare Details oder Oeffnungen
6. sichtbare Texte oder Beschriftungen
7. relevanter Kontext

Die Langbeschreibung soll die sichtbare Form mental nachvollziehbar machen —
nicht bloss Eigenschaften aufzaehlen.


OBJEKT-LOGIK

Beschreibe Objekte ueber:
- sichtbare Form
- Proportion
- Oberflaeche
- Struktur
- Anordnung
- sichtbare Bestandteile

Funktion oder Zweck nur nennen,
wenn eindeutig belegbar.


MATERIAL UND FUNKTION
(KRITISCH — nicht raten)

Material nur nennen,
wenn sichtbar oder kontextuell eindeutig belegt.

Bei Unsicherheit:
- 'helles glattes Material'
- 'glaenzende Oberflaeche'
- 'strukturierte Oberflaeche'

statt:
- Keramik
- Porzellan
- Glas
- Metall

Funktion nicht aus Form ableiten.

NICHT:
- Stimmkarte
- Flyer
- Medikamentendose
- Getraenk
- Nahrung

SONDERN:
- flacher rechteckiger Gegenstand
- kleines rundes Gefaess
- heller zylindrischer Behaelter


ATMOSPHAERE

Bei Objektfotos normalerweise KEINE Atmosphaere beschreiben.

Nur wenn Bildgestaltung und Kontext dies eindeutig tragen,
darf eine zurueckhaltende atmosphaerische Aussage verwendet werden.

Dann MUSS atmosphaere_belege gesetzt werden.


AUSGABE-SCHEMA

Fuelle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, prazise und konkret
- langbeschreibung: maximal 2000 Zeichen
- verwendete_inventar_items: Audit-Trail der genutzten Inventar-Items
- nicht_verwendete_inventar_items: Audit-Trail der bewusst ausgelassenen Items
- nicht_im_inventar: MUSS leer bleiben
- atmosphaere_belege: bei foto_objekte normalerweise leer


FEW-SHOT BEISPIELE

(Noch keine Few-Shot-Beispiele für Bildtyp "foto_objekte" kuratiert.)


FINAL CHECK

1. Jede Aussage belegbar?
2. Keine Halluzination?
3. Wurden verbotene Inhalts-/Fuellungsbegriffe verwendet?
   Falls ja: nur erlaubt wenn sichtbarer Inhalt eindeutig belegt ist.
4. Wurde ein Behaelter-Inhalt erfunden?
5. Wurde eine Substanz oder Konsistenz erfunden?
6. Wurde Material geraten statt belegt?
7. Wurde Funktion oder Zweck geraten?
8. Alt-Text konkret und visuell nachvollziehbar?
9. nicht_im_inventar leer?
10. Wurden alle halluzinations_warnung-Eintraege respektiert?

Wenn ein Punkt nicht erfuellt ist:
Output neu formulieren.

```
