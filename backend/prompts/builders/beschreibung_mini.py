"""Builder der Mini-Familie: logo, icon, funktional.

Alle drei nutzen IconBeschreibungOutput (nur alt_text, 3 bis 80 Zeichen). Sie
überspringen das Inventar und bekommen einen eigenen, kurzen Kopf: Die
allgemeine Beschreiber-Rolle mit Beispielen zu Flugzeugen und Bergen passt nicht
zu einer Ausgabe von wenigen Wörtern. Fassung September 2026.

Dekorativ hat keinen Builder, siehe dekorativ.py.
"""
from __future__ import annotations

from typing import Optional

from prompts.components.constraints import LIZENZ_LOGOS_REGELN
from prompts.components.schemas import ClassificationOutput

from .helpers import bildgroesse_zeile, extract_link_target_from_context, kontext_werte, load_examples, original_alt_zeile, user_hint_block


ROLE_MINI = """Du schreibst Alternativtexte für kleine Bildelemente in Webseiten und Dokumenten:
Logos, Symbole und Bedienelemente. Ein Mensch, der das Element nicht sieht, muss
sofort wissen, wofür es steht oder was es tut. Du benennst, was belegt ist, und
rätst nicht: Ein Name steht im Text, wenn er lesbar ist, ein weltweit eindeutiges
Zeichen ihn trägt oder der Kontext ihn nennt. Sonst beschreibst du neutral. Keine
Vermutungswörter, keine Farben, keine Formbeschreibung um ihrer selbst willen."""


def build_beschreibung_prompt_logo(
    classification: ClassificationOutput,
    enriched_context: str,
    width: int, height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Logo: Marken- oder Organisationsname, Lizenztyp, Linkziel. Höchstens 80 Zeichen."""
    examples = load_examples('logo')
    link_target = extract_link_target_from_context(enriched_context)

    return f"""{ROLE_MINI}

BILDTYP: logo (allein stehendes Marken-, Organisations- oder Lizenzlogo)
{bildgroesse_zeile(width, height, label='BILDGRÖSSE')}
{original_alt_zeile(original_alt)}

AUFTRAG
Nenne die Organisation oder Marke, für die das Logo steht: "Logo Musterwerk GmbH".
Ein lesbarer Slogan darf folgen. Ist der Name weder lesbar noch durch ein weltweit
eindeutiges Zeichen oder den Kontext belegt, schreibe "Logo, Name nicht lesbar".
Eigennamen und Slogans bleiben in ihrer Originalsprache. Höchstens 80 Zeichen.
Ist ein Linkziel angegeben, ergänze es: "Logo Musterwerk GmbH, Link zur Startseite".

{LIZENZ_LOGOS_REGELN}

BEISPIELE

{examples.format_for_prompt()}

KONTEXT
{kontext_werte(enriched_context, user_hint_block(user_hint), extra=(f'LINKZIEL DIESES LOGOS: {link_target}' if link_target else ''))}
"""


def build_beschreibung_prompt_icon(
    classification: ClassificationOutput,
    enriched_context: str,
    width: int, height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Icon: die Funktion, 3 bis 50 Zeichen."""
    examples = load_examples('icon')
    link_target = extract_link_target_from_context(enriched_context)

    return f"""{ROLE_MINI}

BILDTYP: icon (kleines funktionales Symbol wie Lupe, Menü, Warenkorb, Zahnrad)
{bildgroesse_zeile(width, height, label='BILDGRÖSSE')}
{original_alt_zeile(original_alt)}

AUFTRAG
Nenne die Funktion, die das Symbol an dieser Stelle hat: "Suche", "Menü öffnen",
"Warenkorb anzeigen", "Einstellungen". Eine kurze Formangabe in Klammern ist
erlaubt, wenn sie dem Verständnis dient: "Suche (Lupe)". Die Form allein
("Lupe", "Zahnrad-Symbol") ist keine Antwort. Kein Präfix wie "Icon" oder
"Symbol für". 3 bis 50 Zeichen.
Ist ein Linkziel angegeben, gilt die Form "Funktion, Link zu Ziel": "Profil, Link
zum Benutzerkonto". Die Klammer entfällt dann.
Belegt ist die Funktion durch die übliche Bedeutung des Symbols, den Kontext oder
den vorhandenen Alt-Text. Ein Zustand ("Menü schließen" statt "Menü öffnen") nur,
wenn Kontext oder Darstellung ihn zeigen. Ist die Funktion nicht erkennbar:
"Symbol mit unbekannter Funktion".

BEISPIELE

{examples.format_for_prompt()}

KONTEXT
{kontext_werte(enriched_context, user_hint_block(user_hint), extra=(f'LINKZIEL DIESES SYMBOLS: {link_target}' if link_target else ''))}
"""


def build_beschreibung_prompt_funktional(
    classification: ClassificationOutput,
    enriched_context: str,
    width: int, height: int,
    original_alt: str = '',
    user_hint: Optional[str] = None,
) -> str:
    """Bedienelement: Funktion und Zustand, 3 bis 80 Zeichen.

    Die Pipeline überspringt diesen Aufruf oft, wenn der Klassifikator den
    vorhandenen Alt-Text als brauchbar eingestuft hat.
    """
    examples = load_examples('funktional')

    return f"""{ROLE_MINI}

BILDTYP: funktional (Navigations- oder Steuerelement mit Zustand: Blätterpfeile,
Vor und Zurück, Fortschrittsanzeige, Brotkrumenpfad)
{original_alt_zeile(original_alt)}

AUFTRAG
Nenne Funktion und Zustand: "Nächste Seite", "Nächste Seite (von 12)", wenn die
Zahl sichtbar ist, "Vorheriger Beitrag: Titel", wenn der Titel lesbar ist,
"Fortschritt: 3 von 7". Ein ausgegrautes Element beschreibst du als Zustand:
"Keine weiteren Seiten". Bei einem Brotkrumenpfad übernimmst du die lesbaren
Stationen mit ihrem Trennzeichen: "Startseite › Themen › Barrierefreiheit".
Ein Bild einer Seitennavigation beschreibt die sichtbaren Seiten und die aktive
Seite, nicht nur einen Pfeil. 3 bis 80 Zeichen.
Ein vorhandener Alt-Text wird übernommen, wenn Funktion, Ziel und Zustand zum
Element passen. Ein sprachlich sinnvoller Text kann trotzdem die falsche Aktion
oder einen veralteten Zustand nennen; dann schreibst du neu.

BEISPIELE

{examples.format_for_prompt()}

KONTEXT
{kontext_werte(enriched_context, user_hint_block(user_hint))}
"""
