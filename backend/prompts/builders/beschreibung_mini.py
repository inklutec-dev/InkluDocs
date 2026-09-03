"""Pass-3-Builder für Mini-Pipelines: logo, icon, funktional.

Drei Builder, alle nutzen IconBeschreibungOutput (alt_text 3-80 Zeichen,
kein langbeschreibung-Feld) — siehe Mini-Korrektur a in beschreibung.py.

Mini-Pipelines überspringen den Inventar-Pass und gehen direkt vom
Klassifikations-Output in den Beschreibungs-Pass. Keine Atmosphäre,
kein Inventar — nur Funktion oder Markenname.

Dekorativ ist KEIN Builder hier — siehe dekorativ.py
(handle_dekorativ_classification gibt direkt das Result-Dict zurück).
"""
from __future__ import annotations

from typing import Optional

from prompts.components.constraints import (
    ANTI_HALLUZINATION_REGELN,
    EVIDENZ_STUFEN_REGELN,
    LIZENZ_LOGOS_REGELN,
)
from prompts.components.roles import ROLE_BESCHREIBER
from prompts.components.schema_helpers import render_schema_for_prompt
from prompts.components.schemas import (
    ClassificationOutput,
    IconBeschreibungOutput,
)

from .helpers import bildgroesse_zeile, extract_link_target_from_context, kontext_werte, load_examples, original_alt_zeile, user_hint_block


def build_beschreibung_prompt_logo(
    classification: ClassificationOutput,
    enriched_context: str,
    width: int, height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Logo-Builder — Markenname / Lizenz-Code, max 80 Zeichen."""
    schema_doc = render_schema_for_prompt(IconBeschreibungOutput)
    examples = load_examples('logo')
    link_target = extract_link_target_from_context(enriched_context)

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

{LIZENZ_LOGOS_REGELN}

{EVIDENZ_STUFEN_REGELN}

BILDTYP: logo (erkennbares Marken-, Organisations- oder Lizenzlogo)
{bildgroesse_zeile(width, height, label='BILDGRÖSSE')}
{original_alt_zeile(original_alt)}

KONTEXT:
{kontext_werte(enriched_context, user_hint_block(user_hint), extra=(f'LINK-ZIEL DIESES LOGOS: {link_target}' if link_target else ''))}

DEIN AUFTRAG: Der blinde Nutzer muss SOFORT wissen, welche Organisation oder
Marke das Logo repräsentiert. Sonst nichts. Kein visuelles Design, keine Farben,
keine Formen.

FORMAT-PFLICHT:
- Beginne mit 'Logo ' + Markenname (oder 'Lizenz-Logo' / 'Zertifizierungs-Logo'
  bei diesen Sondertypen)
- Optional + Slogan WENN lesbar
- Maximal 80 Zeichen
- Langbeschreibung leer lassen (Schema hat kein langbeschreibung-Feld)

VERBOTEN:
- Visuelle Beschreibung der Logo-Form (Wappen, Tiere, geometrische Formen, Farben)
- Spekulation über die Bedeutung des Logos
- 'Symbol für ...' / 'stilisiertes ...' / 'abstraktes ...'

EVIDENZ-STUFEN FÜR LOGO-IDENTIFIKATION:
- STUFE 1 (immer ok): Markenname als Text im Logo lesbar → direkt nennen
- STUFE 2 (ok): Weltweit eindeutiges Symbol (Apple-Apfel, Mercedes-Stern,
  Coca-Cola-Schriftzug, BMW-Spinner) + Kontext stützt → benennen
- STUFE 3 (verboten): Logo nicht identifizierbar → 'Logo, Text nicht lesbar'
  oder 'Logo eines nicht identifizierbaren Unternehmens'

LIZENZ- UND ZERTIFIZIERUNGS-LOGOS:
- Creative Commons: exakt mit Lizenztyp benennen (siehe LIZENZ_LOGOS_REGELN)
- Bio-Siegel, Fair-Trade, TÜV: konkrete Variante wenn lesbar
- Diese sind NICHT dekorativ — sie tragen rechtliche oder qualitätsbezogene
  Information

VERLINKTE LOGOS:
Wenn LINK-ZIEL gesetzt ist, ergänze: 'Logo [Name] — Link zu [Ziel]' oder
'Logo [Name] — Link zur Startseite' (wenn Link-Ziel die Domain selbst ist).

EIGENNAMEN UND SLOGANS: Im Original belassen, nicht eindeutschen.

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_icon(
    classification: ClassificationOutput,
    enriched_context: str,
    width: int, height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Icon-Builder — kleine funktionale Symbole, alt_text 3-50 Zeichen."""
    schema_doc = render_schema_for_prompt(IconBeschreibungOutput)
    examples = load_examples('icon')
    link_target = extract_link_target_from_context(enriched_context)

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

BILDTYP: icon (kleines funktionales Symbol — Lupe, Hamburger, Warenkorb etc.)
{bildgroesse_zeile(width, height, label='BILDGRÖSSE')}
{original_alt_zeile(original_alt)}

KONTEXT:
{kontext_werte(enriched_context, user_hint_block(user_hint), extra=(f'LINK-ZIEL DIESES ICONS: {link_target}' if link_target else ''))}

DEIN AUFTRAG: Der blinde Nutzer muss SOFORT die Funktion verstehen.
Sonst nichts. Kein visuelles Design und keine Farben — einzige Ausnahme
ist die kurze Formbeschreibung in runden Klammern (siehe FORMAT-PFLICHT).

FORMAT-PFLICHT:
- Die Funktion zuerst, optional gefolgt von einer kurzen Formbeschreibung
  in runden Klammern: 'Suche (Lupe)', 'Menü öffnen (drei Striche)',
  'Warenkorb anzeigen', 'Einstellungen (Zahnrad)', 'Hilfe'
- KEIN Präfix 'Icon —'
- 3-50 Zeichen (Schema-Untergrenze 3, hier max 50 für icon)
- Langbeschreibung leer (Schema hat kein langbeschreibung-Feld)

VERBOTEN:
- Formbeschreibung als Ersatz für die Funktion oder außerhalb der
  Klammer ('Lupe' allein, 'Zahnrad-Symbol')
- Farben
- 'Symbol für ...'
- 'stilisiertes ...'

VERLINKTE ICONS:
Wenn LINK-ZIEL gesetzt: Format '[Funktion] – Link zu [Ziel]'; die
Formbeschreibung in Klammern entfaellt dann (Platz fuer das Link-Ziel,
max 50 Zeichen)
- 'Suche – Link zur Suchseite'
- 'Profil – Link zum Benutzerkonto'
- 'Warenkorb – Link zum Warenkorb (3 Artikel)' wenn Anzahl im
  Bild lesbar

ZWEIFELSFALL:
Wenn Funktion nicht eindeutig ableitbar (weder aus Symbol-Form noch
aus Kontext noch aus original_alt): 'Symbol mit unbekannter Funktion'
ist die ehrliche Antwort. NICHT raten.

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""


def build_beschreibung_prompt_funktional(
    classification: ClassificationOutput,
    enriched_context: str,
    width: int, height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Funktional-Builder — Navigations-/Steuerelemente, alt_text 3-80 Zeichen.

    Pipeline überspringt häufig den KI-Aufruf, wenn original_alt bereits
    eine funktionale Beschreibung enthält (siehe Orchestrator-Frühe-Exit
    mit classification.original_alt_brauchbar).
    """
    schema_doc = render_schema_for_prompt(IconBeschreibungOutput)
    examples = load_examples('funktional')

    return f"""{ROLE_BESCHREIBER}

{ANTI_HALLUZINATION_REGELN}

BILDTYP: funktional (Navigations- oder Steuerungselement mit Zustands-
information — Paginierungspfeile, Vor/Zurück, Fortschrittsanzeigen,
Breadcrumbs)
{original_alt_zeile(original_alt)}

KONTEXT:
{kontext_werte(enriched_context, user_hint_block(user_hint))}

DEIN AUFTRAG: Funktion und ggf. Zustand benennen.

VORRANG: original_alt-Übernahme
Wenn original_alt eine sinnvolle funktionale Beschreibung enthält
(NICHT nur 'Bild' / 'Foto' / 'Grafik'), übernimm ihn wortgetreu oder
mit minimaler Verbesserung. Du verschlechterst NIEMALS einen brauchbaren
Original-Alt.

FORMAT-PFLICHT WENN GENERIERT:
- Funktionsbeschreibung in natürlichem Deutsch
- Zustandsinformation wenn ableitbar:
  - 'Nächste Seite' oder 'Nächste Seite (von 12)' wenn Zahl sichtbar
  - 'Vorheriger Beitrag' oder 'Vorheriger Beitrag: [Titel]' wenn lesbar
  - 'Fortschritt: 3 von 7' bei Fortschrittsanzeigen
- 3-80 Zeichen (Schema-Untergrenze 3, hier max 80 für funktional)
- Langbeschreibung leer (Schema hat kein langbeschreibung-Feld)

BREADCRUMB-SPEZIFIKA:
Bei Breadcrumb-Navigation: lesbare Pfad-Elemente getrennt durch
'›' oder '/' je nach visueller Notation, z.B. 'Startseite › Themen
› Barrierefreiheit'

INAKTIVE / DISABLED-ZUSTÄNDE:
Wenn Element visuell als inaktiv erkennbar (ausgegraut, geringer
Kontrast): 'Keine vorherige Seite' / 'Keine weiteren Seiten' — als
funktionale Beschreibung des Zustands.

FEW-SHOT BEISPIELE:

{examples.format_for_prompt()}

{schema_doc}
"""
