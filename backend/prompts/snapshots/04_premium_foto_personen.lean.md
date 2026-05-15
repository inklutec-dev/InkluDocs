# Premium-Builder foto_personen — Prompt-Modus: lean

- **Builder:** `prompts/builders/beschreibung_foto.py:522`
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

BILDTYP: foto_personen
BILDGROESSE: 1280x720 Pixel

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


PERSONENREGELN

Nur sichtbar belegbare Eigenschaften beschreiben.

Erlaubt:
- sichtbare Haltung, Position, Blickrichtung
- sichtbare Taetigkeit oder Interaktion
- Kleidungscharakter (formell, sportlich, festlich, leger)
- Gegenstaende aus Inventar
- Namen oder Funktionen bei eindeutiger Zuordnung aus Kontext oder Beschriftung

Nicht erlaubt:
- Gesichtserkennung
- Identitaetsvermutung
- Ethnie, Religion oder Gesundheit
- psychologische Interpretation
- erfundene Beziehungen (Kolleginnen, Familie, Teilnehmer — nur wenn Kontext das belegt)
- erfundene Emotionen (gluecklich, begeistert, interessiert)

Alter oder Geschlecht nur nennen, wenn eindeutig sichtbar UND fuer die
Bildaussage relevant.


KONTEXTREGELN

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

OEFFENTLICHE PERSONEN:
Nur benennen, wenn die Zuordnung durch Kontext oder Beschriftung
eindeutig belegt ist — nicht durch Gesichtserkennung.


UNTERSCHRIFTEN

Gedruckte Namen oder Beschriftungen duerfen verwendet werden.
Handschriftliche Unterschriften nicht selbst entziffern oder
interpretieren.


ATMOSPHAERE

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
Keine Atmosphaere ohne Beleg-Eintrag.


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

(Noch keine Few-Shot-Beispiele für Bildtyp "foto_personen" kuratiert.)



FINAL CHECK (vor der Ausgabe pruefen):

1. Jede Aussage durch Inventar oder sichtbare Bildinformation belegbar?
2. Keine Halluzination (kein Item im Output das weder im Inventar noch sichtbar belegt ist)?
3. Keine Emotion oder Beziehung erfunden (gluecklich, motiviert, Kolleginnen, Familie)?
4. Keine Identitaet geraten ohne Kontext-Beleg?
5. Bei unklaren Objekten: sichtbare Form/Farbe/Position beschrieben statt Funktion zu erraten?
6. Alt-Text konkret und visuell charakteristisch — nicht nur Personen- oder Inventar-Aufzaehlung?
7. Vermeidet generische Einleitungen ("Auf dem Bild", "Eine Gruppe von Personen")?
8. Schema vollstaendig korrekt (alle Pflichtfelder gefuellt)?
9. atmosphaere_belege gefuellt wenn Atmosphaere im Text vorkommt?

Wenn ein Punkt nicht erfuellt: Output neu formulieren.


```
