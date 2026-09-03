"""v4-Pipeline-Orchestrator: Klassifikation → Inventar → Beschreibung → Validierung.

Entry-Point ist generate_alt_text_v4() — wird vom pdf_processor.generate_alt_text-
Wrapper gerufen wenn PIPELINE_VERSION=v4.

4-Pässe-Architektur (gegen korrelierte Halluzinationen):
  Pass 1 (Klassifikation): mistral-medium-latest, kurzer Top-Level-Pick
  Pass 2 (Inventar):       pixtral-large-latest, forensische Bildanalyse
  Pass 3 (Beschreibung):   mistral-large-latest, Output-Formulierung aus Inventar
  Pass 4 (Validierung):    pixtral-large-latest, Belegtheit prüfen (Modell-Diversität!)

Frühe Exits:
  - dekorativ via handle_dekorativ_classification (Heuristik-Safety-Net)
  - funktional + brauchbarer original_alt → direkter Pass-Through

Sub-Typ-Aktualisierung:
  Wenn Klassifikator 'foto' wählt, übernimmt der Inventar-Pass die Sub-Typ-
  Entscheidung (foto_event/foto_personen/...). Der effektive Bildtyp geht
  dann an Pass 3 + Pass 4.

VALIDATOR_MODE-Schalter (kompatibel mit v3.7):
  'flag'    — Validator-Pass läuft, needs_review=true wenn nicht ok, kein Override
  'correct' — Validator-Pass läuft, korrigierter Output wird übernommen
  'off'     — Validator-Pass übersprungen (für Eval-Vergleiche)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from prompts.builders import (
    build_beschreibung_prompt_mini,
    build_beschreibung_prompt_with_inventar,
    build_classification_prompt,
    build_combined_inventar_beschreibung_prompt,
    build_inventar_prompt,
    build_validierung_prompt,
    handle_dekorativ_classification,
)
from prompts.components.roles import SYSTEM_BESCHREIBUNG
from prompts.components.schemas import (
    BeschreibungOutput,
    BildtypEffective,
    ClassificationOutput,
    IconBeschreibungOutput,
    InventarOutput,
    ValidierungOutput,
)

from .llm_client import (
    MODEL_CLASSIFY as MISTRAL_MODEL_CLASSIFY,
    MODEL_GENERATE as MISTRAL_MODEL_GENERATE,
    MODEL_INVENTAR as MISTRAL_MODEL_INVENTAR,
    MODEL_VALIDATE as MISTRAL_MODEL_VALIDATE,
    LLMCallError as MistralCallError,
    call_with_schema as call_mistral_with_schema,
)

log = logging.getLogger(__name__)

# ── Ausgabesprache (03.07.2026, alirodocs) ────────────────────────────────
# Die Builder-Prompts (Regeln, Few-Shots, Anti-Halluzination) bleiben KOMPLETT
# deutsch — nur die Ausgabe-Felder wechseln die Sprache. Die Anweisung haengt
# zentral HIER am fertigen Beschreibungs-Prompt, damit sie automatisch fuer
# alle heutigen UND kuenftigen Builder gilt (Daten-Familie, Mini, Gesichter).
_OUTPUT_LANGUAGE_NAMES = {
    "en": "Englisch (English)",
    "da": "Dänisch (dansk)",
    "fr": "Französisch (français)",
    "es": "Spanisch (español)",
    "sv": "Schwedisch (svenska)",
}

def _language_suffix(language: str) -> str:
    """Verbindliche Ausgabesprache-Anweisung fuer den Beschreibungs-Prompt.

    Leer fuer Deutsch (Standardverhalten unveraendert) und fuer unbekannte
    Codes (defensiv: lieber deutsch als kaputt).
    """
    name = _OUTPUT_LANGUAGE_NAMES.get((language or "de").lower())
    if not name:
        return ""
    return (
        "\n\nAUSGABESPRACHE (VERBINDLICH): Formuliere die Felder alt_text und "
        f"langbeschreibung ausschließlich auf {name} — fließend und idiomatisch, "
        "keine wörtliche Übersetzung aus dem Deutschen. Das gilt AUSDRÜCKLICH "
        "auch für alle in den Regeln vorgegebenen Schlagwörter, Gattungs-Präfixe "
        "und festen Wendungen: aus 'Tabelle — ' wird auf Englisch 'Table — ', "
        "aus 'Karte — ' 'Map — ', aus 'Infografik — ' 'Infographic — ', aus "
        "'Screenshot der …' 'Screenshot of …', aus 'Fotomontage:' 'Photo "
        "montage:', aus 'Logo X — Link zur Startseite' 'Logo X — link to the "
        "homepage', aus 'Menü öffnen (drei Striche)' 'Open menu (three lines)' — "
        "sinngemäß ebenso in jeder anderen Zielsprache. In alt_text und "
        "langbeschreibung steht KEIN deutsches Wort, außer es ist im Bild "
        "lesbar oder ein Eigenname. Alle obigen Regeln "
        "(Belegbarkeit, Kompaktheit, Zeichenlimits, kein Intro) gelten unverändert. "
        "Alle übrigen Felder der JSON-Antwort (bildtyp, konfidenz, begruendung usw.) "
        "bleiben deutsch."
    )


def _user_prompt_suffix(user_prompt: str) -> str:
    """Gespeicherter eigener Prompt des Nutzers (Prompt-Verwaltung, 06.07.2026).

    Wird wie _language_suffix zentral an den fertigen Beschreibungs-Prompt
    gehaengt (alle Builder, lean + full) — additiv: Stil-/Schwerpunkt-Vorgaben
    des Nutzers gelten verbindlich, Faktentreue/Belegbarkeit und das
    JSON-Antwortformat bleiben unveraendert. Leer = Standardverhalten.
    """
    text = (user_prompt or '').strip()
    if not text:
        return ''
    return (
        '\n\nEIGENE VORGABEN DES NUTZERS (VERBINDLICH): Der Nutzer hat für dieses '
        'Projekt eigene Vorgaben hinterlegt. Setze sie um, soweit sie Stil, Ton, '
        'Wortwahl, Schwerpunkt, Zielgruppe oder Detailgrad betreffen — bei '
        'Längen-Vorgaben haben sie Vorrang vor den Richtwerten oben. Faktentreue '
        'und Belegbarkeit sowie das JSON-Antwortformat bleiben unverändert gültig.\n'
        + text
    )


# ── Verify-Pass (Fable-5-Review Fund 3, 05.07.2026; Redakteur seit Paket 3) ──
# Der Lean-Modus hat keinen Validator mehr; der Selbst-Check (nicht_im_inventar)
# ist wirkungslos, weil dasselbe Modell sich gegen sein eigenes Kopf-Inventar
# nie belastet. Dieser unabhaengige Pruef-Aufruf glich den fertigen Alt-Text
# urspruenglich nur widerlegend gegen das Bild ab (Refuter-Muster); seit
# Paket 3 (16.07.2026) arbeitet er als REDAKTEUR: binaerer Punkt-fuer-Punkt-
# Abgleich, exaktes Nachzaehlen, Vollstaendigkeits- und Montage-Check, bei
# Beanstandung gleich ein korrigierter Alt-Text. Fehler im Verify duerfen die
# Generierung NIE blockieren.
#
# Schaltung per ENV V4_VERIFY_MODE (ob geprueft wird):
#   'off' (Default) — kein Verify (Kosten-Entscheidung: +1 Bild-Aufruf/Pruefung)
#   'kritisch'      — nur risikoreiche Typen (Personen, Events, Objekte, Screenshots)
#   'alle'          — jeder Nicht-Mini-Typ
# Schaltung per ENV V4_VERIFY_KORREKTUR (was mit der Korrektur passiert):
#   'off' (Default) — verhaltensneutral: nur needs_review-Flag wie bisher
#   'on'            — korrigierter_alt_text wird uebernommen + needs_review gesetzt;
#                     Original und Begruendung landen im Log, die Anwendung im
#                     pipeline_steps-Audit-Trail. Langbeschreibung bleibt unangetastet.
from pydantic import BaseModel, Field, field_validator
from prompts.components.stilregeln import STILREGELN_KERN  # Korrekturwache 03.09.2026: gleiche Stilregeln fuer Pruefer-Korrekturen


class VerifyOutput(BaseModel):
    """Schema des Verify-Passes (bewusst klein — Verdikt + Belege + Korrektur).

    Tolerant gebaut (Befund Ersttest 05.07.): Modelle liefern die Liste
    gelegentlich als JSON-String und schreiben laengere Anmerkungen —
    beides wird normalisiert statt hart abgelehnt, damit der Verify nie
    an Formalien scheitert.

    Paket 3 (16.07.2026): zwei optionale Redakteurs-Felder. korrigierter_alt_text
    wird nur uebernommen, wenn ENV V4_VERIFY_KORREKTUR=on (siehe Lean-Pipeline) —
    das Schema selbst ist verhaltensneutral.
    """
    alt_text_belegt: bool = Field(description='True wenn JEDE konkrete Behauptung des Alt-Texts durch das Bild gedeckt ist')
    strittige_aussagen: list[str] = Field(default_factory=list, description='Woertlich zitierte strittige Behauptungen mit kurzem Grund')
    anmerkung: str = Field(default='')
    korrigierter_alt_text: Optional[str] = Field(
        default=None, min_length=20, max_length=1500,
        description='Vollstaendig korrigierter Alt-Text (Ziel unter 250 Zeichen, hoechstens 400), nur bei Beanstandung — sonst leer',
    )
    korrektur_begruendung: Optional[str] = Field(
        default=None,
        description='Kurze Begruendung, was warum korrigiert wurde — nur zusammen mit korrigierter_alt_text',
    )

    @field_validator('strittige_aussagen', mode='before')
    @classmethod
    def _coerce_list(cls, v):
        if isinstance(v, str):
            import json as _json
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except ValueError:
                pass
            return [v] if v.strip() else []
        return v

    @field_validator('korrigierter_alt_text', 'korrektur_begruendung', mode='before')
    @classmethod
    def _coerce_optional_text(cls, v, info):
        # Tolerant wie oben: Modelle liefern statt null gern '' oder ' ' —
        # das ist "keine Korrektur" und darf den Verify nicht scheitern lassen.
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            # Zu kurz fuer einen echten Alt-Text -> als "keine Korrektur" werten,
            # statt den ganzen Verify an min_length scheitern zu lassen.
            if info.field_name == 'korrigierter_alt_text' and len(v) < 20:
                return None
            # Korrekturwache 03.09.2026: NICHT mehr bei 400 kappen (schnitt mitten im
            # Wort ab, Quertest: Koelner Dom, Umleitungsschild). Ueberlange Korrekturen
            # laesst die Wache _korrektur_absichern() kuerzen oder verwirft sie.
            if len(v) > 1500:
                return v[:1500]
        return v


_VERIFY_KRITISCHE_TYPEN = frozenset({'foto_personen', 'foto_event', 'foto_objekte', 'screenshot',
                                     'foto_landschaft', 'foto_architektur'})  # +landschaft/architektur 17.07.: Wahrzeichen- und Montage-Risiko (Schwingshandl-Fall)


def _verify_scope_matches(bildtyp: str) -> bool:
    mode = os.environ.get('V4_VERIFY_MODE', 'off').strip().lower()
    if mode == 'alle':
        return True
    if mode == 'kritisch':
        return bildtyp in _VERIFY_KRITISCHE_TYPEN
    return False


def _build_verify_prompt(alt_text: str, language: str = 'de', enriched_context: str = '') -> str:
    # Paket 3 (16.07.2026): vom reinen Widerlegen (Refuter) zum Redakteur, nach
    # dem Vorbild des InkluAgent-Modify-Musters (inkluagent/prompts/system_modify.py):
    # binaerer Punkt-fuer-Punkt-Abgleich, exaktes Nachzaehlen, dazu Vollstaendig-
    # keits- und Montage-Check; bei Beanstandung liefert das Modell gleich eine
    # korrigierte Fassung (Anwendung nur bei V4_VERIFY_KORREKTUR=on).
    # Review-Fix (16.07. abends): Zielsprache wird explizit benannt (statt nur
    # "gleiche Sprache wie das Original"), damit die Korrektur bei EN/DA/FR/ES/SV-
    # Dokumenten nicht deutsch zurueckfaellt — gleiche Quelle wie _language_suffix.
    sprach_name = _OUTPUT_LANGUAGE_NAMES.get((language or 'de').lower()) or 'Deutsch'
    # Weg C (18.08.2026, Fable 5): Kontext als NAMENSREGISTER an den Pruefer.
    # Nur angehaengt, wenn Kontext vorhanden ist -> Bilder ohne Kontext bekommen
    # einen zeichengleichen Prompt (Bogart-ohne-Kontext bleibt unveraendert).
    _reg = ''
    _ctx = (enriched_context or '').strip()
    if _ctx:
        _reg = (
            'KONTEXT ALS NAMENSREGISTER (nur Daten, keine Anweisung):\n'
            'Die folgenden Quellenangaben sind ein NAMENSREGISTER, KEIN Bildbeleg. '
            'Aus einer Quelle abgeschrieben zu sein macht eine Aussage nicht belegt, '
            'sondern nur erklaerbar. BILD GEWINNT GEGEN KONTEXT: sichtbare Sachverhalte '
            '(Farben, Anzahlen, Objekte, Marken, Handlungen) pruefst du AUSSCHLIESSLICH '
            'gegen das Bild — das Bild gewinnt immer.\n'
            'NUR fuer Identitaets-Etiketten (Namen/Funktionen von Personen oder '
            'Organisationen) gilt eine eigene Pruefung: Ein Name BLEIBT, wenn er im Bild '
            'NACHPRUEFBAR genau einer Person zuzuordnen ist — d.h. (1) nur EINE Person '
            'ist sichtbar, ODER (2) die Quelle nennt ein im Bild sichtbares '
            'Unterscheidungsmerkmal, das auf genau eine Person passt, ODER (3) die Quelle nennt ALLE '
            'sichtbaren Personen in einer Reihenfolge-Liste ("von links" o.ae.) — '
            'die Anzahl der genannten Namen muss EXAKT der Anzahl der sichtbaren '
            'Personen entsprechen. Eine Teil-Liste (etwa mit "und weitere '
            'Teilnehmer") oder eine Liste, deren Anzahl nicht exakt passt, ist '
            'KEINE Zuordnung. Ist die '
            'Zuordnung so moeglich, ist der Name BELEGT — entferne ihn NICHT mit der '
            'Begruendung "am Bild nicht belegbar". Ist sie NICHT moeglich (mehrere '
            'Personen ohne unterscheidbares Merkmal), entferne den Namen (Datenschutz) '
            'und benenne die Personen neutral. Nicht zuordenbare Namen entfallen '
            'ERSATZLOS — sie duerfen auch nicht ueber Umwege wie "laut '
            'Bildunterschrift" oder "genannt werden" im Alt-Text auftauchen. Ein Name, der in KEINER Quelle steht und '
            'zu keiner allgemein bekannten Person gehoert, ist eine Erfindung und wird '
            'entfernt. NIE NAMEN HINZUFUEGEN: Du fuegst niemals einen Namen in '
            'den Alt-Text ein, der dort nicht schon steht — das Register '
            'rechtfertigt nur das BEHALTEN vorhandener Namen, nie das Einfuegen '
            'oder Ersetzen. Muss ein Name entfallen, benenne die Person neutral.\n'
            'FUER KORREKTUREN GILT: Ein nach diesen Regeln BELEGTER Name bleibt '
            'bei JEDER Neufassung woertlich an seiner Position erhalten — eine '
            'Beanstandung anderer Details (Farbe, Marke, Anzahl, Objekte) ist '
            'KEIN Grund, den Namen zu streichen. Nicht-sichtbare Eigenschaften '
            'aus den Quellen (Beruf, Alter, Behinderung, Taetigkeit) sind am Bild '
            'weder belegbar noch widerlegbar — sie zaehlen bei der Zuordnung '
            'weder dafuer noch dagegen und sind KEIN Widerspruch zum Bild.\n'
            'QUELLEN (Namensregister, gekuerzt):\n"' + _ctx[:1500] + '"\n\n'
        )
    _basis = (

        'Du bist ein unabhaengiger Redakteur fuer Alternativtexte. Gleiche den '
        'folgenden Alt-Text Punkt fuer Punkt mit dem Bild ab. Jede Aussage wird '
        'binaer bewertet: belegt oder nicht belegt — Einstufungen wie '
        '"weitgehend korrekt" sind verboten.\n\n'
        'PRUEFE JEDE KONKRETE BEHAUPTUNG EINZELN GEGEN DAS BILD:\n'
        '- Marken-, Produkt- und Personen-Namen (stimmen sie exakt?). Auch ein '
        'unverwechselbares Produktdesign zaehlt als Beleg: Ein Produkt, das ein '
        'durchschnittlicher sehender Mensch am Design sofort erkennt (z.B. ein '
        'MacBook am charakteristischen flachen Aluminiumgehaeuse), gilt als '
        'BELEGT — stufe es NICHT auf die generische Bezeichnung zurueck. '
        'Umgekehrt bleibt ein generisches Geraet ohne solche Merkmale generisch.\n'
        '- wortgetreu zitierte Texte und Aufschriften (Buchstabe fuer Buchstabe)\n'
        '- Personen- und Objekt-Zahlen: zaehle selbst EXAKT nach. "Circa", "rund" '
        'oder "etwa" ohne sichtbaren Verdeckungs-, Anschnitt- oder Unschaerfe-Grund '
        'ist eine Beanstandung. Grenzfaelle (leicht versetzte oder angeschnittene '
        'Personen) werden NICHT weggezaehlt; bei strittiger Zuordnung praezisiert '
        'die Korrektur das Gesamtbild ("X in einer Reihe, weitere im Hintergrund"), '
        'statt eine niedrigere Zahl zu behaupten. AENDERE eine Anzahl NUR, wenn die '
        'urspruengliche Zahl zweifelsfrei falsch ist — ist die Zuordnung strittig '
        'oder beide Zaehlweisen vertretbar, BEHALTE die Zahl des Erzeugers und '
        'ergaenze hoechstens das Gesamtbild.\n'
        '- Farben und eindeutige visuelle Merkmale\n'
        '- VOLLSTAENDIGKEIT: Fehlen zentrale, fuer die Bildfunktion wichtige '
        'Elemente (lesbarer Text, ein Wahrzeichen, praegende Objekte)?\n'
        '- MONTAGE-CHECK: Passen Bildelemente erkennbar nicht zusammen (harte '
        'Freisteller-Kanten, widerspruechliche Schatten/Perspektive/Massstab, '
        'Stilbruch Foto/Grafik, unmoegliche Kombinationen)? Suche dabei AKTIV '
        'Quadrant fuer Quadrant auch nach KLEINEN eingefuegten Objekten (z.B. ein '
        'winziges Bauwerk an unmoeglicher Stelle) — geringe Groesse schuetzt eine '
        'Montage nicht. Dann muss der Alt-Text das Bild WOERTLICH als '
        '"Fotomontage" oder "Collage" benennen — Umschreibungen wie '
        '"aufgesetzte" oder "eingefuegte" Elemente reichen NICHT. Fehlt die '
        'woertliche Kennzeichnung bei erkennbarer Montage, ist das eine '
        'Beanstandung, und deine Korrektur ergaenzt sie.\n\n'
        'alt_text_belegt=false NUR, wenn eine konkrete Behauptung falsch oder im '
        'Bild nicht belegt ist. Stil und Wortwahl sind KEINE Pruefkriterien. '
        'Feine Farb- oder Deutungs-Nuancen sind nur strittig, wenn der Alt-Text '
        'klar danebenliegt. Plausibel reicht nicht als Widerlegung — du brauchst '
        'einen sichtbaren Widerspruch. '
        'Die Benennung zweifelsfrei erkennbarer Personen des oeffentlichen Lebens '
        'und Wahrzeichen ist in diesem Barrierefreiheits-Werkzeug legitim und '
        'erwuenscht — pruefe nur, ob die Benennung RICHTIG ist, nicht OB benannt '
        'werden darf. EIN allgemein bekanntes Kenn-Faktum zu einem korrekt '
        'benannten Wahrzeichen (z.B. "Matterhorn (4.478 m)") gilt als durch die '
        'Benennung gedeckt — beanstande es NUR, wenn es sachlich falsch ist.\n\n'
        'KORREKTUR: Ist etwas falsch, unbelegt, ohne sichtbaren Grund gehedgt, '
        'unvollstaendig oder eine unerkannte Montage, liefere in '
        'korrigierter_alt_text eine vollstaendig korrigierte Fassung — '
        f'ausschliesslich auf {sprach_name} (die Sprache des Original-Alt-Texts). '
        'MINIMALEINGRIFF: Aendere NUR die beanstandeten Stellen. Alle anderen '
        'Formulierungen des Originals uebernimmst du woertlich — du fuehrst keine '
        'neuen Angaben ein (Material, Farbe, Art, Marke, Tageszeit, Deutung), die '
        'du nicht selbst als Beanstandung aufgefuehrt hast, und du ergaenzt keine '
        'Nebensaechlichkeiten. Eine Korrektur ist nie laenger als das Original plus '
        'das, was die Beanstandung zwingend braucht. '
        'Laengen-Regime wie beim Original: einfache Motive unter 150 Zeichen, '
        'komplexe Szenen bis etwa 250, harte Obergrenze 400 — und in '
        'korrektur_begruendung kurz, was warum geaendert wurde. Keine halben '
        'Anpassungen. Ist nichts zu beanstanden, lasse beide Felder leer.\n\n'
        + STILREGELN_KERN + '\n\n'
    )
    return _basis + _reg + f'ALT-TEXT ZUR PRUEFUNG:\n"{alt_text}"'


# ─────────────────────────────────────────────────────────────────────────
# KORREKTURWACHE (Steve 03.09.2026, nach dem Modell-Quertest)
# Befund: Die Korrektur des Pruefers ging ungefiltert in die Datenbank. Bei 400
# Zeichen wurde mitten im Wort gekappt (Koelner Dom, Umleitungsschild), und ein
# Pruefer schrieb Unbeteiligtes um („brauner Metallbock" statt rotem Holzgestell).
# Die Wache: (1) Prompt verlangt Minimaleingriff + Stilregeln (oben), (2) eine
# Korrektur ueber VERIFY_KORREKTUR_MAX Zeichen wird EINMAL vom Pruefmodell
# nachgekuerzt (mit Bild, damit keine Fakten verloren gehen), (3) bleibt sie
# ueber 400 Zeichen oder scheitert das Kuerzen, wird sie VERWORFEN — der
# Original-Alt-Text bleibt, needs_review bleibt gesetzt, der Mensch liest gegen.
# Nie wird abgeschnitten. Gilt fuer die Pipeline UND den Chatbot-Speicherweg.
# ─────────────────────────────────────────────────────────────────────────
VERIFY_KORREKTUR_MAX = int(os.environ.get('VERIFY_KORREKTUR_MAX', '250'))
VERIFY_KORREKTUR_HART = 400


class KuerzungOutput(BaseModel):
    alt_text: str = Field(min_length=20, max_length=1500, description='Gekuerzter Alt-Text')


def _kuerze_korrektur(image_path: str, korrektur: str, language: str = 'de') -> Optional[str]:
    """Ein Aufruf: ueberlange Pruefer-Korrektur unter VERIFY_KORREKTUR_MAX bringen."""
    sprach_name = _OUTPUT_LANGUAGE_NAMES.get((language or 'de').lower()) or 'Deutsch'
    prompt = (
        'Der folgende Alt-Text ist zu lang. Kuerze ihn auf hoechstens '
        f'{VERIFY_KORREKTUR_MAX} Zeichen, ausschliesslich auf {sprach_name}. Behalte alle '
        'am Bild belegten Kernfakten (Anzahlen, Namen, lesbare Texte, Wahrzeichen, '
        'Montage-Kennzeichnung), streiche Nebensaechlichkeiten, Deutungen und '
        'Hintergrund-Details. Fuege NICHTS Neues hinzu. Ein bis zwei Saetze, das '
        'Wichtigste zuerst.\n\n' + STILREGELN_KERN + '\n\nALT-TEXT:\n"' + korrektur + '"'
    )
    try:
        out = call_mistral_with_schema(
            model=MISTRAL_MODEL_VALIDATE, prompt=prompt, image_path=image_path,
            schema=KuerzungOutput, max_tokens=600, system=SYSTEM_BESCHREIBUNG,
        )
        return (out.alt_text or '').strip() or None
    except Exception as e:
        log.warning('Kuerzen der Verify-Korrektur fehlgeschlagen (ignoriert): %s', e)
        return None


def _korrektur_absichern(image_path: str, verify_result, language: str = 'de') -> tuple[Optional[str], str]:
    """Wache vor der Uebernahme einer Pruefer-Korrektur.

    Returns (korrektur_oder_None, schritt) mit schritt in
    'uebernommen' | 'gekuerzt' | 'verworfen' | '' (keine Korrektur vorhanden).
    """
    korr = (getattr(verify_result, 'korrigierter_alt_text', None) or '').strip()
    if not korr:
        return None, ''
    if len(korr) <= VERIFY_KORREKTUR_MAX:
        return korr, 'uebernommen'
    kurz = _kuerze_korrektur(image_path, korr, language=language)
    if kurz and len(kurz) <= VERIFY_KORREKTUR_HART:
        log.info('Verify-Korrektur gekuerzt: %d -> %d Zeichen', len(korr), len(kurz))
        return kurz, 'gekuerzt'
    log.warning('Verify-Korrektur VERWORFEN (%d Zeichen, Kuerzen %s) — Original bleibt, needs_review gesetzt',
                len(korr), 'lieferte %d Zeichen' % len(kurz) if kurz else 'fehlgeschlagen')
    return None, 'verworfen'


def _run_verify_pass(image_path: str, bildtyp: str, alt_text: str, language: str = 'de', enriched_context: str = ''):
    """Fuehrt den Verify-Aufruf aus. Gibt VerifyOutput oder None (Fehler/aus) zurueck."""
    if not alt_text or not _verify_scope_matches(bildtyp):
        return None
    try:
        # 03.09.2026 (Steve): Der Pruefpass nimmt das VALIDATE-Modell (ENV
        # BEDROCK_MODEL_VALIDATE) — bisher lief er im Lean-Mode stillschweigend
        # mit dem Beschreibungsmodell, der Schalter war dort wirkungslos. So kann
        # der Pruefer ein staerkeres Modell bekommen als der Erzeuger (Premium-
        # Weg: Cross-Model-Validator), ohne die Erzeugung zu verteuern.
        return call_mistral_with_schema(
            model=MISTRAL_MODEL_VALIDATE,
            prompt=_build_verify_prompt(alt_text, language=language, enriched_context=enriched_context),
            image_path=image_path,
            schema=VerifyOutput,
            max_tokens=1000,  # Paket 3: Platz fuer korrigierter_alt_text + Begruendung
            system=SYSTEM_BESCHREIBUNG,  # gleiche Legitimation wie die Generierung
        )
    except Exception as e:  # Verify ist Sicherheitsnetz, nie Blocker
        log.warning('Verify-Pass fehlgeschlagen (ignoriert): %s', e)
        return None


def verify_alt_text_extern(image_path: str, bildtyp: str, alt_text: str,
                           language: str = 'de', enriched_context: str = ''):
    """Oeffentlicher Einstieg fuer den Redakteurs-Check AUSSERHALB der Pipeline.

    Qualitaetsrunde 21.08.2026: Der InkluAgent-Speicherweg (update_alt_text)
    laeuft vor dem DB-Write durch DENSELBEN Pruefer wie die Pipeline —
    gleicher Prompt, gleiche ENV-Schalter (V4_VERIFY_MODE-Scope), gleiches
    Namensregister-Verhalten. Rueckgabe: VerifyOutput oder None (Verify aus,
    Bildtyp nicht im Scope oder Pruef-Fehler — der Verify ist Sicherheitsnetz,
    nie Blocker).
    """
    return _run_verify_pass(image_path, bildtyp, alt_text,
                            language=language, enriched_context=enriched_context)


def _variation_suffix(previous_alt: str) -> str:
    """Gezielte Variation beim Einzel-Neu-Generieren (05.07.2026).

    Statt reiner Zufalls-Temperatur bekommt das Modell den bisherigen Alt-Text
    als Abgrenzungs-Vorlage und den Auftrag, sich deutlich davon abzuheben —
    bei identischer Faktenlage. Leer bei Erst-Generierung und im Sammellauf
    (previous_alt kommt nur vom Neu-Generieren-Endpunkt). Haengt wie
    _language_suffix zentral am fertigen Prompt und gilt damit automatisch
    fuer alle heutigen und kuenftigen Builder.
    """
    prev = (previous_alt or '').strip()
    if not prev:
        return ''
    if len(prev) > 600:
        prev = prev[:600] + ' …'
    return (
        '\n\nVARIATION (NEU GENERIEREN): Der Nutzer wünscht eine Alternative zu '
        'diesem bisherigen Alt-Text:\n'
        f'"{prev}"\n'
        'Schreibe eine DEUTLICH anders formulierte und anders gewichtete Fassung: '
        'anderer Satzeinstieg, anderer Satzbau, gern eine andere Reihenfolge oder '
        'ein anderer Schwerpunkt bei gleichwertigen Aspekten. Das gilt für Alt-Text '
        'UND Langbeschreibung. Die Faktenlage bleibt identisch — keine neuen '
        'unbelegten Aussagen, und belegte Kernfakten (Namen, Marken, Typen, lesbare '
        'Texte wie Schild- oder Gate-Aufschriften, und eine vorhandene '
        'Fotomontage-/Collage-Kennzeichnung) bleiben in BEIDEN Feldern der '
        'neuen Fassung erhalten. Alle übrigen Regeln gelten unverändert.'
    )


# Bildtypen die einen Inventar-Pass benötigen.
INVENTAR_PFLICHTIG: frozenset[str] = frozenset({
    'foto', 'illustration', 'diagramm', 'tabelle', 'karte',
    'screenshot', 'infografik', 'strukturformel',
})

# Bildtypen die einen Validierungs-Pass benötigen (alle Inventar-pflichtigen
# plus die Mini-Pipelines, weil auch dort Halluzinationen möglich sind).
VALIDIERUNG_PFLICHTIG: frozenset[str] = INVENTAR_PFLICHTIG | frozenset({
    'logo', 'icon', 'funktional',
})

# Mini-Pipeline-Typen (kein Inventar-Pass).
_MINI_TYPES: frozenset[str] = frozenset({'logo', 'icon', 'funktional'})

# Generic alt_texts die den 'funktional + brauchbarer alt' Frühen-Exit
# nicht triggern dürfen.
_GENERIC_ALT_TEXTE = {'', 'bild', 'foto', 'grafik', 'image', 'picture'}


def _original_alt_brauchbar(original_alt: str) -> bool:
    """Heuristik-Helper für funktional-Frühe-Exit. Spiegelt die Logik aus
    dem Klassifikator-Prompt (siehe build_classification_prompt) für die
    Server-seitige Pre-Check-Variante.

    Verwendet wenn der Klassifikations-Pass selbst übersteuert wurde
    (image_type_override) und classification.original_alt_brauchbar daher
    nicht aus echtem LLM-Reasoning kommt.
    """
    if not original_alt:
        return False
    norm = original_alt.strip().lower()
    return norm not in _GENERIC_ALT_TEXTE and len(norm) >= 5


# foto_subtyp-Werte aus dem ClassificationOutput-Schema. Wenn ein
# image_type_override einer dieser Werte ist (z.B. weil ein Bild bereits
# als 'foto_event' in der DB steht), muessen wir korrekt aufdroeseln:
# bildtyp='foto', foto_subtyp=<wert>. Sonst lehnt Pydantic den Override ab.
_FOTO_SUBTYP_OVERRIDE_VALUES = {
    'foto_personen', 'foto_event', 'foto_objekte',
    'foto_landschaft', 'foto_architektur', 'foto_essen',
}


def _classification_from_override(
    image_type_override: str,
    original_alt: str,
) -> ClassificationOutput:
    """Baut ein ClassificationOutput aus einem image_type_override-String.

    Splittet automatisch, wenn der Wert ein foto_subtyp ist (z.B. 'foto_event'
    aus der DB-Spalte image_type) statt eines Top-Level-Bildtyps. Verhindert
    Pydantic-Validation-Fehler beim 'neu generieren' bereits klassifizierter
    Foto-Sub-Typen ueber den InkluAgent-Chatbot oder den Neu-Generieren-Button.
    """
    if image_type_override in _FOTO_SUBTYP_OVERRIDE_VALUES:
        return ClassificationOutput(
            bildtyp='foto',
            foto_subtyp=image_type_override,
            konfidenz='hoch',
            ist_dekorativ=False,
            original_alt_brauchbar=_original_alt_brauchbar(original_alt),
            klassifikations_begruendung='Manuelle Uebersteuerung durch Nutzer-Wahl (Sub-Typ aus DB).',
        )
    return ClassificationOutput(
        bildtyp=image_type_override,
        konfidenz='hoch',
        ist_dekorativ=False,
        original_alt_brauchbar=_original_alt_brauchbar(original_alt),
        klassifikations_begruendung='Manuelle Uebersteuerung durch Nutzer-Wahl.',
    )


def _format_pipeline_steps(
    classification: ClassificationOutput,
    inventar: Optional[InventarOutput],
    beschreibung: Optional[BeschreibungOutput | IconBeschreibungOutput],
    validierung: Optional[ValidierungOutput],
) -> str:
    """Audit-Trail-String — welche Pässe sind gelaufen, welche übersprungen."""
    steps = [f'classified:{classification.bildtyp}']
    if inventar is not None:
        steps.append(f'inventar:konfidenz={inventar.inventar_konfidenz_gesamt}')
    if beschreibung is not None:
        steps.append('described')
    if validierung is not None:
        steps.append(f'validated:ok={validierung.validierung_ok}')
    return ','.join(steps)


def _run_multipass_pipeline(
    image_path: str,
    enriched_context: str = '',
    image_type_override: Optional[str] = None,
    user_hint: Optional[str] = None,
    width: int = 0,
    height: int = 0,
    original_alt: str = '',
    language: str = 'de',
    previous_alt: str = '',
    user_prompt: str = '',  # Eigener gespeicherter Nutzer-Prompt (Prompt-Verwaltung 06.07.2026)
) -> dict:
    """v4-Entry-Point. Wird vom pdf_processor.generate_alt_text gerufen wenn
    PIPELINE_VERSION=v4.

    Args:
      image_path:          Lokaler Pfad zum Bild.
      enriched_context:    Web-/PDF-Kontext (Titel, umliegender Text, etc.).
      image_type_override: Manuelle Bildtyp-Wahl (überspringt Pass 1).
                           Z.B. wenn Nutzer im Frontend einen Typ erzwingt.
      user_hint:           Workflow-Variante 3: Nutzer-Hinweis (HOHE PRIORITÄT).
      width, height:       Bildmaße in Pixel.
      original_alt:        Vom Autor gesetzter alt-Text aus dem Original-PDF/HTML.

    Returns:
      dict mit dem v4-Result. Kompatibel mit v3.7-Result-Format
      (bildtyp, alt_text, langbeschreibung, needs_review, plus v4-Audit-Felder).

    Raises:
      MistralCallError wenn ein Mistral-Call auch nach Retry fehlschlägt.
      Der Aufrufer sollte das fangen und die Pipeline gnädig abbrechen
      (z.B. Result mit needs_review=True und einer Fehlermeldung).
    """
    validator_mode = os.environ.get('VALIDATOR_MODE', 'flag').lower()

    # === Pass 1: Klassifikation ===
    if image_type_override:
        # Frontend hat den Typ gewählt — Pass 1 überspringen.
        classification = _classification_from_override(image_type_override, original_alt)
    else:
        classify_prompt = build_classification_prompt(
            enriched_context=enriched_context,
            width=width, height=height,
            original_alt=original_alt,
            user_hint=user_hint,
        )
        classification = call_mistral_with_schema(
            model=MISTRAL_MODEL_CLASSIFY,
            prompt=classify_prompt,
            image_path=image_path,
            schema=ClassificationOutput,
            max_tokens=500,  # E10-Korrektur: 5 Felder + 200-Z-Begründung + JSON-Overhead
        )

    # === Frühe Exits für simple Typen ===

    # K3-Fix: dekorativ durch validate_dekorativ()-Heuristik gegenprüfen.
    # Mini-Korrektur d: Tuple-Return spart einen zweiten Heuristik-Aufruf.
    if classification.ist_dekorativ:
        deko_result, corrected_type = handle_dekorativ_classification(
            classification, image_path, width, height, original_alt,
        )
        if deko_result is not None:
            return deko_result
        # Heuristik-Override: corrected_type kommt direkt aus dem Tuple.
        log.info(
            'Dekorativ-Override: Klassifikator hatte dekorativ, Heuristik sagt %s',
            corrected_type,
        )
        classification = ClassificationOutput(
            bildtyp=corrected_type,
            konfidenz='mittel',
            ist_dekorativ=False,
            original_alt_brauchbar=classification.original_alt_brauchbar,
            klassifikations_begruendung=(
                f"Heuristik-Override: Klassifikator hatte 'dekorativ' gewählt, "
                f"validate_dekorativ() korrigierte auf '{corrected_type}'."
            ),
        )

    # W3-Fix: classification.original_alt_brauchbar ist Schema-Feld vom Pass 1.
    # Wenn override gesetzt war, hat _original_alt_brauchbar() es schon gesetzt.
    if classification.bildtyp == 'funktional' and classification.original_alt_brauchbar:
        log.info('Funktional + brauchbarer original_alt → Pass-Through ohne KI-Aufruf.')
        return {
            'bildtyp': 'funktional',
            'konfidenz': classification.konfidenz,
            'alt_text': original_alt,
            'langbeschreibung': '',
            'needs_review': False,
            'pipeline_steps': 'classified:funktional,passthrough:original_alt',
            'inventar_json': None,
            'validation_result': None,
        }

    # === Pass 2: Inventar (nur wenn nötig) ===
    inventar: Optional[InventarOutput] = None
    if classification.bildtyp in INVENTAR_PFLICHTIG:
        inventar_prompt = build_inventar_prompt(
            bildtyp=classification.bildtyp,
            enriched_context=enriched_context,
            width=width, height=height,
            user_hint=user_hint,
        )
        inventar = call_mistral_with_schema(
            model=MISTRAL_MODEL_INVENTAR,
            prompt=inventar_prompt,
            image_path=image_path,
            schema=InventarOutput,
            max_tokens=2500,
        )

    # Sub-Typ-Aktualisierung: bei foto entscheidet der Inventar-Pass den Sub-Typ.
    effective_bildtyp: BildtypEffective
    if (
        classification.bildtyp == 'foto'
        and inventar is not None
        and inventar.foto_subtyp
    ):
        effective_bildtyp = inventar.foto_subtyp
    else:
        effective_bildtyp = classification.bildtyp

    # === Pass 3: Beschreibung ===
    # Andere Modell als Pass 2 (Pixtral) — hier Mistral Large für Reasoning-Stärke.
    beschreibung: BeschreibungOutput | IconBeschreibungOutput
    if effective_bildtyp in _MINI_TYPES:
        # Mini-Pipelines: kein Inventar, nutzt classification + original_alt.
        mini_prompt = build_beschreibung_prompt_mini(
            bildtyp=effective_bildtyp,
            classification=classification,
            enriched_context=enriched_context,
            width=width, height=height,
            original_alt=original_alt,
            user_hint=user_hint,
        )
        mini_prompt += _user_prompt_suffix(user_prompt)
        mini_prompt += _language_suffix(language)
        mini_prompt += _variation_suffix(previous_alt)
        beschreibung = call_mistral_with_schema(
            model=MISTRAL_MODEL_GENERATE,
            prompt=mini_prompt,
            image_path=image_path,
            schema=IconBeschreibungOutput,
            max_tokens=300,
            system=SYSTEM_BESCHREIBUNG,  # Mini-Pipelines liefern <80 Zeichen — 300 Tokens reichen.
        )
    else:
        # Standard-Pfad mit Inventar.
        if inventar is None:
            # Sollte nicht passieren wenn INVENTAR_PFLICHTIG korrekt definiert ist.
            raise RuntimeError(
                f"Bildtyp '{effective_bildtyp}' braucht Inventar, aber Pass 2 wurde übersprungen."
            )
        besch_prompt = build_beschreibung_prompt_with_inventar(
            bildtyp=effective_bildtyp,
            inventar=inventar,
            enriched_context=enriched_context,
            width=width, height=height,
            user_hint=user_hint,
        )
        besch_prompt += _user_prompt_suffix(user_prompt)
        besch_prompt += _language_suffix(language)
        besch_prompt += _variation_suffix(previous_alt)
        beschreibung = call_mistral_with_schema(
            model=MISTRAL_MODEL_GENERATE,
            prompt=besch_prompt,
            image_path=image_path,
            schema=BeschreibungOutput,
            max_tokens=2500,
            system=SYSTEM_BESCHREIBUNG,
        )

    # Halluzinations-Self-Check: hat das Modell Items genannt die nicht im Inventar sind?
    # Sekundäres Sicherheitsnetz — primärer Schutz ist Pass 4.
    needs_review_forced = False
    if isinstance(beschreibung, BeschreibungOutput) and beschreibung.nicht_im_inventar:
        log.warning(
            'Self-Check-Verletzung: Beschreibung enthält Items außerhalb des Inventars: %s',
            beschreibung.nicht_im_inventar,
        )
        needs_review_forced = True

    # === Pass 4: Validierung ===
    validierung: Optional[ValidierungOutput] = None
    if (
        classification.bildtyp in VALIDIERUNG_PFLICHTIG
        and validator_mode != 'off'
    ):
        # Mini-Pipelines liefern IconBeschreibungOutput — der Validator erwartet
        # BeschreibungOutput. Wir bauen ein kompatibles Objekt für die Validierung.
        if isinstance(beschreibung, IconBeschreibungOutput):
            besch_for_validation = BeschreibungOutput(
                alt_text=beschreibung.alt_text if len(beschreibung.alt_text) >= 20
                         else beschreibung.alt_text + ' ' * (20 - len(beschreibung.alt_text)),
                # Padding nur fürs Schema — Validator schaut auf Inhalt, nicht Länge.
                # TODO: ggf. eigenen Validator-Pfad für Mini-Pipelines bauen,
                # der IconBeschreibungOutput direkt akzeptiert.
                verwendete_inventar_items=beschreibung.verwendete_inventar_items,
            )
        else:
            besch_for_validation = beschreibung

        valid_prompt = build_validierung_prompt(
            bildtyp=effective_bildtyp,
            inventar=inventar,
            beschreibung=besch_for_validation,
            enriched_context=enriched_context,
        )
        validierung = call_mistral_with_schema(
            model=MISTRAL_MODEL_VALIDATE,
            prompt=valid_prompt,
            image_path=image_path,
            schema=ValidierungOutput,
            max_tokens=2500,
        )

        # 'correct'-Modus: Korrekturen anwenden auf den BeschreibungOutput-Pfad.
        # Mini-Pipelines bekommen aktuell keine Korrektur — das wäre zu invasiv
        # für eine 80-Zeichen-Beschreibung.
        if (
            validator_mode == 'correct'
            and not validierung.validierung_ok
            and isinstance(beschreibung, BeschreibungOutput)
        ):
            if validierung.korrektur_alt_text:
                beschreibung.alt_text = validierung.korrektur_alt_text
            if validierung.korrektur_langbeschreibung:
                beschreibung.langbeschreibung = validierung.korrektur_langbeschreibung

    # === Result zusammensetzen ===
    needs_review = needs_review_forced
    if validierung is not None and validierung.needs_review:
        needs_review = True

    langbeschreibung = (
        beschreibung.langbeschreibung
        if isinstance(beschreibung, BeschreibungOutput)
        else ''
    )

    return {
        'bildtyp': effective_bildtyp,
        'konfidenz': classification.konfidenz,
        'alt_text': beschreibung.alt_text,
        'langbeschreibung': langbeschreibung,
        'needs_review': needs_review,
        'pipeline_steps': _format_pipeline_steps(
            classification, inventar, beschreibung, validierung,
        ),
        'inventar_json': inventar.model_dump_json() if inventar is not None else None,
        'validation_result': validierung.model_dump_json() if validierung is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCHER + LEAN-PIPELINE — Lean-Mode-Erweiterung (08.05.2026)
# ═══════════════════════════════════════════════════════════════════════════════
#
# generate_alt_text_v4() ist der oeffentliche Eintrittspunkt. Er verzweigt per
# V4_PASS_MODE-ENV in den klassischen Multi-Pass-Modus oder den neuen Lean-Mode.
#
# Multi-Pass-Modus (V4_PASS_MODE=full, default):
#   Pass 1 Klassifikation -> Pass 2 Inventar -> Pass 3 Beschreibung -> Pass 4 Validator
#   Designed fuer Mistral-Modell-Diversitaet. Bleibt unveraendert verfuegbar
#   und ist die richtige Wahl wenn Mistral-Provider aktiv ist (LLM_PROVIDER=mistral).
#
# Lean-Modus (V4_PASS_MODE=lean):
#   Pass 1 Klassifikation (mit foto_subtyp) -> Combo Inventar+Beschreibung in einem Aufruf
#   Designed fuer starke Modelle wie Sonnet die Inventar implizit miterledigen koennen.
#   Empfohlen fuer LLM_PROVIDER=bedrock. Spart 1-2 Paesse, ca. 30% Kosten.
#
# Production (Container inkludocs ohne V4_PASS_MODE-Env): laeuft auf default 'full',
# also unveraendert. Migration auf 'lean' nur nach Test-Phase auf Staging.
# ═══════════════════════════════════════════════════════════════════════════════


def generate_alt_text_v4(
    image_path: str,
    enriched_context: str = '',
    image_type_override: Optional[str] = None,
    user_hint: Optional[str] = None,
    width: int = 0,
    height: int = 0,
    original_alt: str = '',
    temperature: float = 0.0,  # 0 = deterministisch (Default/Bulk); >0 nur beim Einzel-Neu-Generieren
    language: str = 'de',
    previous_alt: str = '',  # bisheriger Alt-Text — nur beim Neu-Generieren gesetzt (gezielte Variation)
    user_prompt: str = '',  # Eigener gespeicherter Nutzer-Prompt (Prompt-Verwaltung 06.07.2026)
) -> dict:
    """v4-Eintrittspunkt mit Mode-Dispatcher.

    Wahl per ENV V4_PASS_MODE:
      'full' (default): Multi-Pass-Pipeline (4 Paesse). Funktioniert mit Mistral.
      'lean':           Lean-Pipeline (Klassifikation + Combo). Empfohlen mit Bedrock.

    Args/Returns: identisch zu beiden Pipelines.
    """
    pass_mode = os.environ.get('V4_PASS_MODE', 'full').strip().lower()
    if pass_mode == 'lean':
        return _run_lean_pipeline(
            image_path=image_path,
            enriched_context=enriched_context,
            image_type_override=image_type_override,
            user_hint=user_hint,
            width=width,
            height=height,
            original_alt=original_alt,
            temperature=temperature,
            language=language,
            previous_alt=previous_alt,
            user_prompt=user_prompt,
        )
    # Default: Multi-Pass (rueckwaertskompatibel zu Mistral-Setup)
    return _run_multipass_pipeline(
        image_path=image_path,
        enriched_context=enriched_context,
        image_type_override=image_type_override,
        user_hint=user_hint,
        width=width,
        height=height,
        original_alt=original_alt,
        language=language,
        previous_alt=previous_alt,
        user_prompt=user_prompt,
    )




# ─────────────────────────────────────────────────────────────────────────
# Sub-Typ-Heuristik fuer Lean-Mode (Variante A, 08.05.2026)
# ─────────────────────────────────────────────────────────────────────────
# Sicherheitsnetz wenn der Klassifikator das foto_subtyp-Feld nicht setzt:
# wir leiten den Sub-Typ aus Schluesselwoertern im enriched_context ab.
# Greift NUR im Lean-Mode wenn classification.foto_subtyp is None.
#
# Reihenfolge der Checks: spezifischste zuerst (foto_event vor foto_personen
# weil 'Workshop' implizit Personen meint, aber als Event klassifiziert
# werden muss).
# ─────────────────────────────────────────────────────────────────────────

_FOTO_SUBTYP_KEYWORDS: dict[str, tuple[str, ...]] = {
    'foto_event': (
        'workshop', 'schulung', 'konferenz', 'meeting', 'tagung',
        'seminar', 'vor-ort', 'event', 'veranstaltung', 'kongress',
        'fortbildung', 'training',
    ),
    'foto_personen': (
        'porträt', 'portrait', 'foto von ', 'bild von ',
        'schauspieler', 'künstler', 'sänger', 'autor', 'redner',
        'gründer', 'vorstand', 'geschäftsführer',
    ),
    'foto_landschaft': (
        'landschaft', 'panorama', 'natur', 'wald', 'berge',
        'meer', 'see', 'küste', 'gebirge', 'wiese',
    ),
    'foto_architektur': (
        'gebäude', 'architektur', 'fassade', 'kirche', 'turm',
        'bauwerk', 'rathaus', 'museum', 'palast',
    ),
    'foto_essen': (
        'speise', 'gericht', 'essen', 'mahlzeit', 'kuchen',
        'brot', 'menü', 'rezept', 'küche',
    ),
    'foto_objekte': (
        'produkt', 'werkstück', 'werkstatt', 'keramik', 'tasse',
        'schüssel', 'gefäß', 'stillleben', 'material',
    ),
}


def _infer_foto_subtyp_from_context(enriched_context: str) -> Optional[str]:
    """Schluesselwort-Heuristik fuer Sub-Typ-Wahl ohne Klassifikator-Hilfe.

    Nur Lean-Mode-Sicherheitsnetz. Reihenfolge entspricht der Spezifitaet
    (foto_event hat Vorrang vor foto_personen, weil 'Workshop' Personen
    impliziert aber als Event klassifiziert werden muss).
    """
    if not enriched_context:
        return None
    ctx = enriched_context.lower()
    for sub_typ, keywords in _FOTO_SUBTYP_KEYWORDS.items():
        if any(kw in ctx for kw in keywords):
            return sub_typ
    return None



def _run_lean_pipeline(
    image_path: str,
    enriched_context: str = '',
    image_type_override: Optional[str] = None,
    user_hint: Optional[str] = None,
    width: int = 0,
    height: int = 0,
    original_alt: str = '',
    temperature: float = 0.0,
    language: str = 'de',
    previous_alt: str = '',
    user_prompt: str = '',  # Eigener gespeicherter Nutzer-Prompt (Prompt-Verwaltung 06.07.2026)
) -> dict:
    """Lean-Pipeline: 1 Klassifikations-Aufruf + 1 Combo-Hauptaufruf.

    Genutzt wenn V4_PASS_MODE=lean. Empfohlen mit LLM_PROVIDER=bedrock (Claude
    Sonnet kann Inventar implizit miterledigen, separater Pass nicht noetig).

    Validator-Pass ist im Lean-Mode NICHT eingebaut — Sonnet halluziniert kaum,
    deshalb Standard-Pipeline ohne Validator. Wenn Validierung gewuenscht:
    entweder V4_PASS_MODE=full nutzen, oder spaeter Cross-Family-Validator
    als optionalen Premium-Tier ergaenzen.
    """
    # === Pass 1: Klassifikation (mit foto_subtyp dank Lean-Builder-Anweisung) ===
    if image_type_override:
        classification = _classification_from_override(image_type_override, original_alt)
    else:
        classify_prompt = build_classification_prompt(
            enriched_context=enriched_context,
            width=width, height=height,
            original_alt=original_alt,
            user_hint=user_hint,
        )
        classification = call_mistral_with_schema(
            model=MISTRAL_MODEL_CLASSIFY,
            prompt=classify_prompt,
            image_path=image_path,
            schema=ClassificationOutput,
            max_tokens=600,  # Lean: +100 fuer foto_subtyp-Feld
        )

    # === Frühe Exits: dekorativ + funktional ===
    if classification.ist_dekorativ:
        deko_result, corrected_type = handle_dekorativ_classification(
            classification, image_path, width, height, original_alt,
        )
        if deko_result is not None:
            return deko_result
        log.info('Dekorativ-Override (Lean): Heuristik korrigiert auf %s', corrected_type)
        classification = ClassificationOutput(
            bildtyp=corrected_type,
            konfidenz='mittel',
            ist_dekorativ=False,
            original_alt_brauchbar=classification.original_alt_brauchbar,
            klassifikations_begruendung=(
                f"Heuristik-Override (Lean): Klassifikator hatte 'dekorativ', "
                f"korrigiert auf '{corrected_type}'."
            ),
        )

    if classification.bildtyp == 'funktional' and classification.original_alt_brauchbar:
        log.info('Lean: funktional + brauchbarer original_alt → Pass-Through.')
        return {
            'bildtyp': 'funktional',
            'konfidenz': classification.konfidenz,
            'alt_text': original_alt,
            'langbeschreibung': '',
            'needs_review': False,
            'pipeline_steps': 'lean:classified:funktional,passthrough:original_alt',
            'inventar_json': None,
            'validation_result': None,
        }

    # === Effektiver Bildtyp aus Klassifikator (Lean: foto_subtyp dort gesetzt) ===
    effective_bildtyp: BildtypEffective
    if classification.bildtyp == 'foto' and classification.foto_subtyp:
        effective_bildtyp = classification.foto_subtyp
    elif classification.bildtyp == 'foto':
        # Klassifikator hat foto erkannt, aber Sub-Typ nicht gesetzt.
        # Variante A (08.05.2026): zwei-stufiges Fallback-Verfahren.
        # Stufe 1: Kontext-Heuristik aus enriched_context
        # Stufe 2: foto_objekte als Last-Resort-Default
        inferred_subtyp = _infer_foto_subtyp_from_context(enriched_context)
        if inferred_subtyp is not None:
            log.info(
                'Lean: Klassifikator ohne foto_subtyp -> Heuristik aus Kontext: %s',
                inferred_subtyp,
            )
            effective_bildtyp = inferred_subtyp
        else:
            log.warning(
                'Lean: Klassifikator ohne foto_subtyp UND Heuristik ohne Match. '
                'Last-Resort-Fallback auf foto_objekte.'
            )
            effective_bildtyp = 'foto_objekte'
    else:
        effective_bildtyp = classification.bildtyp

    # === Beschreibung: Mini-Pipeline oder Combo ===
    beschreibung: BeschreibungOutput | IconBeschreibungOutput
    if effective_bildtyp in _MINI_TYPES:
        # Mini-Pipelines (logo/icon/funktional): unveraendert von Multi-Pass
        mini_prompt = build_beschreibung_prompt_mini(
            bildtyp=effective_bildtyp,
            classification=classification,
            enriched_context=enriched_context,
            width=width, height=height,
            original_alt=original_alt,
            user_hint=user_hint,
        )
        mini_prompt += _user_prompt_suffix(user_prompt)
        mini_prompt += _language_suffix(language)
        mini_prompt += _variation_suffix(previous_alt)
        beschreibung = call_mistral_with_schema(
            model=MISTRAL_MODEL_GENERATE,
            prompt=mini_prompt,
            image_path=image_path,
            schema=IconBeschreibungOutput,
            max_tokens=300,
            system=SYSTEM_BESCHREIBUNG,
        )
    else:
        # Hauptpfad: Combo-Aufruf mit Inventar+Beschreibung in einem Schritt
        combo_prompt = build_combined_inventar_beschreibung_prompt(
            bildtyp_top=classification.bildtyp,
            bildtyp_effective=effective_bildtyp,
            enriched_context=enriched_context,
            width=width, height=height,
            original_alt=original_alt,
            user_hint=user_hint,
        )
        combo_prompt += _user_prompt_suffix(user_prompt)
        combo_prompt += _language_suffix(language)
        combo_prompt += _variation_suffix(previous_alt)
        beschreibung = call_mistral_with_schema(
            model=MISTRAL_MODEL_GENERATE,
            prompt=combo_prompt,
            image_path=image_path,
            schema=BeschreibungOutput,
            max_tokens=3500,  # Lean: mehr Tokens weil Inventar+Beschreibung
            temperature=temperature,
            system=SYSTEM_BESCHREIBUNG,  # Variation nur beim Neu-Generieren; Klassifikation bleibt deterministisch
        )

    # === Self-Check: Halluzinations-Indikator ohne Validator-Pass ===
    needs_review = False
    if isinstance(beschreibung, BeschreibungOutput) and beschreibung.nicht_im_inventar:
        log.warning(
            'Lean Self-Check: Beschreibung enthaelt nicht_im_inventar-Items: %s',
            beschreibung.nicht_im_inventar,
        )
        needs_review = True

    # === Verify-Pass (optional per V4_VERIFY_MODE, s. Bausteine oben) ===
    verify_result = None
    verify_korrektur_applied = False
    if effective_bildtyp not in _MINI_TYPES:
        verify_result = _run_verify_pass(
            image_path, effective_bildtyp, beschreibung.alt_text, language=language,
            enriched_context=enriched_context,
        )
        if verify_result is not None and not verify_result.alt_text_belegt:
            log.warning(
                'Verify-Pass: Alt-Text nicht voll belegt (%s): %s',
                effective_bildtyp, verify_result.strittige_aussagen,
            )
            needs_review = True
        # Paket 3 (16.07.2026): Redakteurs-Korrektur nur bei explizitem Opt-in
        # V4_VERIFY_KORREKTUR=on. Default 'off' ist verhaltensneutral (nur Flag
        # wie bisher). Original + Begruendung werden geloggt (Muster wie oben);
        # der volle Verify-Output steht ohnehin in validation_result. Die
        # Langbeschreibung bleibt in beiden Faellen unangetastet.
        verify_korrektur_schritt = ''
        if (
            verify_result is not None
            and verify_result.korrigierter_alt_text
            and os.environ.get('V4_VERIFY_KORREKTUR', 'off').strip().lower() == 'on'
        ):
            # Korrekturwache 03.09.2026: kuerzen oder verwerfen, nie abschneiden
            sichere_korrektur, verify_korrektur_schritt = _korrektur_absichern(
                image_path, verify_result, language=language,
            )
            needs_review = True  # Beanstandung: Mensch liest gegen — auch bei verworfener Korrektur
            if sichere_korrektur:
                log.warning(
                    'Verify-Korrektur %s (%s). Original: %r — Begruendung: %s',
                    verify_korrektur_schritt, effective_bildtyp,
                    beschreibung.alt_text,
                    verify_result.korrektur_begruendung or '(keine)',
                )
                beschreibung.alt_text = sichere_korrektur
                verify_korrektur_applied = True

    langbeschreibung = (
        beschreibung.langbeschreibung
        if isinstance(beschreibung, BeschreibungOutput)
        else ''
    )

    return {
        'bildtyp': effective_bildtyp,
        'konfidenz': classification.konfidenz,
        'alt_text': beschreibung.alt_text,
        'langbeschreibung': langbeschreibung,
        'needs_review': needs_review,
        'pipeline_steps': (
            f'lean:classified:{classification.bildtyp},combo:{effective_bildtyp}'
            + (f',verify:ok={verify_result.alt_text_belegt}' if verify_result is not None else '')
            + (',verify_korrektur:applied' if verify_korrektur_applied else '')
            + (f',verify_korrektur:{verify_korrektur_schritt}' if verify_korrektur_schritt in ('gekuerzt', 'verworfen') else '')
        ),
        'inventar_json': None,  # Im Lean-Mode kein separates Inventar-Objekt
        'validation_result': verify_result.model_dump_json() if verify_result is not None else None,
    }
