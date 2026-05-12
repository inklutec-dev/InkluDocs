"""Pass-4-Prompt-Builder: Validierung (Pixtral-Large, Modell-Diversität).

Universal für alle Bildtypen mit Validierungs-Pflicht (siehe
VALIDIERUNG_PFLICHTIG im v4-Orchestrator). Bildtyp-spezifische
Spezialregeln über VALIDIERUNG_SPEZIAL.

K2-Fix: bildtyp ist BildtypEffective (nicht BildtypTopLevel) — der
Orchestrator übergibt nach dem Inventar-Pass den effektiven Sub-Typ
(z.B. 'foto_event'), damit VALIDIERUNG_SPEZIAL['foto_event'] greifen kann.

W2-Korrektur (Steve, 04.05.2026): VALIDIERUNG_SPEZIAL um 4 Bildtypen
ergänzt (screenshot, infografik, foto_essen, strukturformel). Vorher
hatten diese nur die universellen Regeln.
"""
from __future__ import annotations

import json
from typing import Optional

from prompts.components.constraints import (
    EIGENNAMEN_REGELN,
    LIZENZ_LOGOS_REGELN,
)
from prompts.components.roles import ROLE_VALIDATOR
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import (
    BeschreibungOutput,
    BildtypEffective,
    InventarOutput,
    ValidierungOutput,
)


# Bildtyp-spezifische Validierungs-Schwerpunkte. dict.get(bildtyp, '')
# fällt für nicht-aufgeführte Typen leise auf '' zurück — dort greifen
# nur die universellen Regeln (EIGENNAMEN, LIZENZ_LOGOS, sowie die
# Aufgabenliste im Prompt).
VALIDIERUNG_SPEZIAL: dict[str, str] = {
    'foto_event': """SPEZIAL FÜR foto_event:
Pflicht-Prüfungen:
- Personenanzahl im Output ungefähr inventar.personen.length
- Jedes objekt_in_haenden aus dem Inventar entweder erwähnt ODER bewusst weggelassen
  (nicht_verwendete_inventar_items)
- Atmosphäre-Wertungen nur akzeptieren wenn atmosphaere_belege explizit den
  visuellen Beleg im Inventar referenzieren""",

    'foto_objekte': """SPEZIAL FÜR foto_objekte (Iteration 3 — gehärtet 04.05.2026):
Pflicht-Prüfungen:
- Material-Charakteristik aus Inventar-Objekten übernommen?
- Anordnung/Anzahl korrekt übernommen?

KRITISCH BEI BEHÄLTERN (Schalen, Tassen, Gläser, Teller, Dosen, Flaschen,
Boxen, Vasen, Töpfe, Becher):

Folgende Wörter im Output sind HART als Halluzination zu markieren
(deckung=nicht_belegt), wenn sie OHNE expliziten Inventar-Beleg
verwendet werden:
- Füllung
- gefüllt
- Inhalt (im Sinne von Behälter-Inhalt)
- Substanz
- cremig / Creme
- Flüssigkeit
- Pulver
- Paste
- Schaum
- Masse
- enthält
- Essen / Getränk

Inventar-Beleg heißt: ein objekte-Eintrag oder lesbare_texte-Eintrag mit
explizitem Inhalt-Bezug. Wenn das Inventar nur 'Schale mit heller
Innenglasur' sagt, ist 'cremig wirkende Substanz' eine Halluzination.

WICHTIG zur Validator-Logik (ChatGPT-Klarstellung 04.05.2026 Iter3-Endphase):

"Inhalt" ist NUR dann eine Halluzination, wenn EXISTENZ oder BESCHAFFENHEIT
behauptet wird. Nicht jede Erwaehnung von "Inhalt".

NICHT als Halluzination werten (Negativ-Disclaimer sind erlaubt):
- "ohne erkennbaren Inhalt"
- "kein Inhalt sichtbar"
- "Inhalt nicht eindeutig erkennbar"
- "genaue Beschaffenheit nicht eindeutig erkennbar"

ALS Halluzination werten (Existenz- oder Art-Behauptung ohne Beleg):
- "mit Inhalt gefuellt"
- "der Inhalt ist X"
- "enthaelt X"
- "cremige Substanz"
- "weisse Fuellung"
- "Schale mit Suppe"

Begruendung: Negativ-Disclaimer sind korrekte Unsicherheits-Kommunikation
und MUESSEN erlaubt sein. Sie helfen blinden Nutzern statt sie zu
verwirren.

Inhalte duerfen beschrieben werden, wenn sie im Inventar als sichtbar
enthalten sind. Art des Inhalts (z.B. "Milch", "Creme") darf nur
genannt werden, wenn durch Inventar, sichtbaren Text oder Kontext
eindeutig belegt.

Beispiel: "Glas mit weisser Fluessigkeit" ist erlaubt wenn die
Fluessigkeit sichtbar ist. "Glas Milch" ist nur erlaubt bei
eindeutigem Kontext-Beleg.

Cremefarben (als Farbadjektiv fuer Glasur, Wand, Stoff) ist erlaubt.
Cremig (als Konsistenz-Beschreibung von Substanzen) bleibt verboten.

Plus halluzinations_warnung-Respekt: wenn das Inventar in
halluzinations_warnung explizit warnt (z.B. 'Glasur könnte als Flüssigkeit
fehlinterpretiert werden') und der Output trotzdem 'Flüssigkeit',
'Substanz', 'Füllung' verwendet, ist das eine BESONDERS schwere
Halluzination — markieren plus needs_review=True forcieren.

Plus halluzinations_warnung-Respekt: wenn das Inventar in
halluzinations_warnung explizit warnt (z.B. 'Glasur könnte als Flüssigkeit
fehlinterpretiert werden') und der Output trotzdem 'Flüssigkeit',
'Substanz', 'Füllung' verwendet, ist das eine BESONDERS schwere
Halluzination — markieren plus needs_review=True forcieren.""",

    'illustration': """SPEZIAL FÜR illustration:
Pflicht-Prüfungen:
- Wenn Inventar Mehrfach-Hypothesen für Hauptmotiv listet: hat der Output diese
  Unsicherheit abgebildet ODER eine sichere Festlegung getroffen die NICHT durch
  Inventar gestützt ist?
- Wurden ALLE Nebenelemente aus dem Inventar erwähnt (oder bewusst weggelassen)?
- Wurden erfundene Interaktionen formuliert ('Tier hält Gerät', 'arbeitet am Laptop')
  obwohl Inventar nur 'Objekte nebeneinander' listet?""",

    'diagramm': """SPEZIAL FÜR diagramm:
Pflicht-Prüfungen:
- Spitzenwert + Schlusslicht aus Inventar im alt_text?
- ALLE Kategorien/Legenden-Einträge aus Inventar in langbeschreibung?
- Trends im alt_text konsistent mit Werten in langbeschreibung?
- Achseneinheiten (%, EUR, Mio.) korrekt übernommen?""",

    'tabelle': """SPEZIAL FÜR tabelle:
Pflicht-Prüfungen:
- Spaltenköpfe wortgetreu aus Inventar?
- Bilanzsumme/Gesamtsumme korrekt benannt (NICHT mit Zwischensummen verwechselt)?
- Werte korrekt zugeordnet (keine Spalten-Verwechslung)?""",

    'karte': """SPEZIAL FÜR karte:
Pflicht-Prüfungen:
- ALLE markierten Standorte aus Inventar erwähnt?
- Legende erklärt?
- Ortsnamen wortgetreu (keine OCR-Verwechslungen)?""",

    'screenshot': """SPEZIAL FÜR screenshot:
Pflicht-Prüfungen:
- URL-Leisten-Inhalt + Fenstertitel wortgetreu im Output, falls im
  Inventar als lesbare_texte vorhanden?
- Sichtbare Buttons, Menüpunkte und Statusmeldungen aus Inventar erfasst
  (oder bewusst weggelassen)?
- Verbot: erfundene Anwendungs-Identifikation (z.B. 'Microsoft Word'
  wenn weder URL noch Logo noch Titel das belegt) — als nicht_belegt
  markieren und Korrektur 'Screenshot eines Texteditors' vorschlagen.""",

    'infografik': """SPEZIAL FÜR infografik:
Pflicht-Prüfungen:
- ALLE Stationen/Schritte aus inventar.objekte im Output erwähnt
  (oder bewusst weggelassen mit Begründung)?
- Konkrete Datenpunkte/Prozentwerte aus inventar.lesbare_texte
  wortgetreu übernommen?
- Verbot: visuelle Layout-Beschreibungen ('oben links steht...',
  'ein Pfeil zeigt von X nach Y') — diese als nicht_belegt markieren,
  weil sie blinden Nutzern keinen Mehrwert geben.
- Insight-First: erster Satz nennt Hauptaussage mit Datenpunkten,
  nicht 'Eine Infografik mit verschiedenen Daten'.""",

    'foto_essen': """SPEZIAL FÜR foto_essen:
Pflicht-Prüfungen:
- Geschmacks-/Wertungs-Adjektive ('lecker', 'köstlich', 'delikat',
  'appetitlich', 'verführerisch') als nicht_belegt markieren —
  subjektive Wertungen ohne visuelle Evidenz.
- Erfundene Zutaten markieren: jede Zutat im Output muss in
  inventar.objekte oder den moeglichen_identifikationen stehen.
  'Mit frischen Kräutern garniert' wenn keine Kräuter im Inventar
  → nicht_belegt.
- Kultur-/Herkunfts-Identifikation nur akzeptieren wenn aus Kontext
  oder Beschriftung belegt — sonst nicht_belegt.""",

    'strukturformel': """SPEZIAL FÜR strukturformel:
Pflicht-Prüfungen:
- Atom-Symbole und Bindungstypen gegen inventar.objekte/lesbare_texte
  prüfen — keine Atome erfinden, die nicht im Inventar stehen.
- Stoffname nur akzeptieren wenn aus Kontext oder Bildbeschriftung
  belegt — visuelle Identifikation ist fehleranfällig (außer H2O, CO2).
- Bei Reaktionsgleichungen: Edukte, Reaktionsbedingungen UND Produkte
  alle drei genannt? Wenn etwas fehlt → fehlende_wichtige_inventar_items.
- Notation: Indizes als normale Zahlen ('CH3' nicht 'CH₃') für
  Screenreader-Verträglichkeit.""",
}


def build_validierung_prompt(
    bildtyp: BildtypEffective,
    inventar: Optional[InventarOutput],
    beschreibung: BeschreibungOutput,
    enriched_context: str = '',
) -> str:
    """Pass-4-Prompt: Validierung des generierten Outputs gegen Inventar + Bild.

    K2-Fix: bildtyp ist BildtypEffective — kann Top-Level oder foto-Sub-Typ
    sein. Sub-Typ-spezifische Spezialregeln greifen über VALIDIERUNG_SPEZIAL.

    Inputs:
      bildtyp:          Effektiver Bildtyp (z.B. 'foto_event' nach Inventar-Pass).
      inventar:         InventarOutput aus Pass 2 (None bei simplen Bildtypen
                        ohne Inventar-Pass — logo, icon, funktional).
      beschreibung:     BeschreibungOutput aus Pass 3 — wird hier validiert.
      enriched_context: Optional, hilft bei Kontext-bezogenen Aussagen.

    Output: prompt-String, der mit ValidierungOutput-Schema gerufen wird.
    """
    schema_doc = render_schema_for_prompt(ValidierungOutput)
    bildtyp_spezialregeln = VALIDIERUNG_SPEZIAL.get(bildtyp, '')

    inventar_block = (
        inventar.model_dump_json(indent=2)
        if inventar is not None
        else '(kein Inventar — simpler Bildtyp ohne Inventar-Pass)'
    )

    atmosphaere_block = (
        json.dumps(
            [b.model_dump() for b in beschreibung.atmosphaere_belege],
            ensure_ascii=False,
            indent=2,
        )
        if beschreibung.atmosphaere_belege
        else '(keine)'
    )

    return f"""{ROLE_VALIDATOR}

{LIZENZ_LOGOS_REGELN}

{EIGENNAMEN_REGELN}

BILDTYP: {bildtyp}

INVENTAR (von Pass 2 erstellt — die verbindliche Faktenbasis):
{inventar_block}

GENERIERTER ALT-TEXT:
{beschreibung.alt_text}

GENERIERTE LANGBESCHREIBUNG:
{beschreibung.langbeschreibung or '(leer)'}

VOM GENERATOR DEKLARIERTE ATMOSPHÄRE-BELEGE:
{atmosphaere_block}

KONTEXT (vom Web-Scraper / PDF / API):
{enriched_context or '(kein Kontext)'}

{bildtyp_spezialregeln}

DEINE AUFGABE — IN GENAU DIESER REIHENFOLGE:

1. INVENTAR-VERGLEICH PRO AUSSAGE:
   Gehe alt_text und langbeschreibung Satz für Satz durch. Für jede inhaltliche
   Aussage (nicht jede Konjunktion oder Floskel) prüfe:
   - Ist sie durch ein Inventar-Item gestützt? → 'inventar_belegt'
   - Ist sie eine Atmosphäre-Wertung mit explizitem visuellen Beleg? → 'atmosphaere_belegt'
   - Ist sie weder durch Inventar noch durch Beleg gestützt? → 'nicht_belegt'
   Trage jede Aussage in inventar_vergleich ein.

2. SCHAUE SELBST AUFS BILD:
   Du siehst das Bild auch direkt. Falls du SELBST eine Halluzination erkennst, die das
   Inventar übersehen hat (Inventar-Pass kann fehlerhaft sein), markiere die Aussage
   ebenfalls als 'nicht_belegt' mit Quelle 'vision_check'.

3. FEHLENDE WICHTIGE INVENTAR-ITEMS:
   Welche Inventar-Items wurden im Output NICHT verwendet, obwohl sie wichtig sind?
   Wichtig sind insbesondere:
   - lesbare_texte mit Typ 'kontaktdaten', 'url', 'datum', 'zahl' — IMMER Pflicht
   - Hauptmotiv-Personen oder dominante Objekte
   - Setting-Indikatoren die für die Bildaussage zentral sind
   In fehlende_wichtige_inventar_items eintragen.

4. KORREKTUR-VORSCHLAG (nur bei validierung_ok=false):
   Wenn nicht_belegte_aussagen oder fehlende_wichtige_inventar_items nicht-leer sind:
   formuliere einen korrigierten alt_text und ggf. korrigierte langbeschreibung,
   die NUR Inventar-belegte Aussagen enthalten. Bestehende Atmosphäre-Belege
   beibehalten wenn sie korrekt sind.
   Wenn keine sichere Korrektur möglich (z.B. unklarer Bildinhalt): None setzen.

5. NEEDS_REVIEW-FLAG:
   Setze needs_review=true wenn EINER der folgenden Punkte zutrifft:
   - validierung_ok=false (auch nach Korrektur — Mensch sollte gegenlesen)
   - inventar.inventar_konfidenz_gesamt = 'mittel' oder 'niedrig'
   - Eine Aussage als 'atmosphaere_belegt' markiert wurde — Wertungen brauchen
     menschliches Urteil ob sie passen
   - Du selbst Unsicherheit empfindest

WAS DU NICHT TUST:
- Stilistische Korrekturen ('klingt holprig' ist KEIN Validierungsgrund)
- Plausibilitäts-Vermutungen ('klingt nach Eventfoto, also wird's schon passen') —
  Plausibel ist KEIN Validierungs-Kriterium. Belegt ist das Kriterium.
- Eigenmächtig Inventar-Items hinzufügen, die der Inventar-Pass übersehen hat
  (außer du sagst es explizit als 'vision_check')

{schema_doc}
"""
