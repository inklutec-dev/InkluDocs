"""Builder der Foto-Familie (sechs Untertypen).

Fassung September 2026 (Prompt-Runde nach dem Prüfkorpus). Aufbau jedes Builders,
siehe docs/PROMPT-STANDARD.md:
  Kopf (Rolle + Belegregeln, im Combo-Aufruf einmal ganz oben)
  BILDTYP, AUFTRAG, inneres Inventar, ALT-TEXT, LANGBESCHREIBUNG,
  höchstens zwei besondere Regeln der Kategorie, STILREGELN, BEISPIELE, KONTEXT.
Was für alle Bildtypen gilt (Beleg, Kontext, Zählen, Wertungen, Montage), steht
nur im Kopf und wird hier nicht wiederholt.
"""
from __future__ import annotations

from typing import Optional

from prompts.components.constraints import ANTI_HALLUZINATION_REGELN, KUNSTWERK_REGEL
from prompts.components.roles import ROLE_BESCHREIBER
from prompts.components.schemas import InventarOutput
from prompts.components.stilregeln import STILREGELN

from .helpers import bildgroesse_zeile, inventar_block, kontext_werte, kopf_schichten, load_examples, user_hint_block


_INVENTAR_EINLEITUNG = (
    "Das Inventar enthält die Beobachtungen des Analyse-Schritts. Es ist die\n"
    "Grundlage jeder Aussage; sichtbare Bildinformationen dürfen ergänzt werden,\n"
    "aber nichts darf dem Inventar widersprechen."
)


def _basis_schichten() -> str:
    """Rolle und Belegregeln im Prompt-Kopf; im Combo-Aufruf leer (stehen dort einmal oben)."""
    return kopf_schichten(f'{ROLE_BESCHREIBER}\n\n{ANTI_HALLUZINATION_REGELN}')


def _render_inventar_block(inventar_json: str) -> str:
    return inventar_block(inventar_json, _INVENTAR_EINLEITUNG)


def _render_personen_block() -> str:
    """Personen: was benannt werden darf und was nicht. Ergänzt Belegregel 6 (Namen)."""
    return """PERSONEN

- Erkennbare Personen benennst du: Personen des öffentlichen Lebens, wenn die
  Erkennung zweifelsfrei ist, und Personen, die Kontext, Namensschild oder
  Bildunterschrift eindeutig zuordnen. Ein Name aus dem Kontext bleibt auch bei
  Kürzungen erhalten.
- Grobe, eindeutig sichtbare Kategorien sind erlaubt und meist hilfreich: Kind,
  Jugendlicher, Erwachsener, älterer Mensch; "Mann im dunklen Anzug", "Frau im
  blauen Blazer". Kleidungscharakter (formell, sportlich, festlich) ebenso.
- Nicht benannt werden Ethnie, Religion und Gesundheit, außer sie sind der
  Gegenstand des Bildes. Keine psychologische Deutung, keine erfundene Beziehung
  oder Emotion. Ein weißer Langstock, ein Rollstuhl oder ein Hörgerät werden als
  sichtbare Gegenstände genannt, wenn sie zum Verständnis der Szene gehören.
- Gedruckte Namen und Beschriftungen darfst du verwenden. Handschriftliche
  Unterschriften entzifferst du nicht."""


def _render_kontext_block(enriched_context: str, user_hint_text: str) -> str:
    """Die Werte zum Kontext. Die Regeln dazu stehen in Belegregel 6."""
    return f"""KONTEXT (Bildunterschrift, umliegender Text, Angaben des Aufrufers)
{kontext_werte(enriched_context, user_hint_text)}"""



def build_beschreibung_prompt_foto_event(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """foto_event: Veranstaltungen, Gruppensituationen, soziale Szenen."""
    examples = load_examples('foto_event')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: foto_event
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Foto einer Veranstaltung oder Gruppensituation: Workshop, Schulung, Konferenz,
Besprechung, Feier, Bühne. Der Text macht die Situation nachvollziehbar: Was für
eine Veranstaltung ist das, wer ist beteiligt, was geschieht sichtbar, wie ist der
Raum aufgebaut. Belegte Angaben aus dem Kontext (Anlass, Veranstalter, Ort, Datum,
Rolle einer Person) gehören in den ersten Satz. Eine Veranstaltung nennst du nur
beim Namen, wenn Bild oder Kontext sie belegen: Präsentation, Moderationsmaterial,
Namensschilder, Beamer, Bühne, organisierte Sitzordnung. Mehrere Personen allein
sind keine Veranstaltung.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginne mit der Art der Situation und dem Merkmal, das sie prägt, nicht mit einer
Personenzählung: "Workshop der Musterwerk GmbH zur Barrierefreiheit: acht Personen
stehen in einer Reihe und halten orange und weiße Abstimmkarten hoch." Dann die
Struktur der Szene: Wer ist wem zugewandt, was tun die Personen sichtbar, welcher
Gegenstand verbindet die Handlung. Gibt es eine Person, die die Szene ordnet (vorn
stehend, der Gruppe zugewandt, von den Blicken der Gruppe adressiert), gehört sie in
den Alt-Text, auch mit dem Rücken zur Kamera; ihre Rolle (moderierend, vortragend)
nennst du nur bei eindeutiger Tätigkeit oder passendem Kontext. Raum, Farben,
Möbel und Logos folgen, soweit sie die Szene unterscheiden oder dem Dokumentzweck
dienen.


LANGBESCHREIBUNG

Fließtext in dieser Reihenfolge, ohne Überschriften: Gesamtüberblick, räumliche
Anordnung, Personen und ihre sichtbare Interaktion, zentrale Gegenstände und
Materialien, lesbare Texte und Logos, belegte Zusatzangaben aus dem Kontext.
Zusammenhänge statt Kleinigkeiten: Die Langbeschreibung erklärt die Szene, sie
zählt sie nicht auf.


{_render_personen_block()}


{STILREGELN}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_foto_personen(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """foto_personen: Porträt, Einzelperson in Situation, kleine Gruppe."""
    examples = load_examples('foto_personen')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: foto_personen
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Foto, auf dem eine oder mehrere Personen im Mittelpunkt stehen: Porträt,
Einzelperson in einer Situation, kleine Gruppe. Der Text beantwortet, wer zu sehen
ist, in welcher belegten Rolle und was die Person sichtbar tut oder in welcher
Situation sie ist. Die Rolle kommt aus dem Kontext (Gründerin der Musterwerk GmbH,
Referentin des Workshops) und steht im ersten Satz. Zu einer zweifelsfrei
benannten Person des öffentlichen Lebens darf ein einzelnes Kenn-Faktum stehen
(Amt und Zeitraum), nicht mehr.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Führe mit der Person: der Name als Subjekt, wenn er belegt ist, sonst eine
sichtbare Kategorie ("eine Frau im blauen Blazer"); dann die belegte Rolle und die
Handlung oder Situation. Dazu höchstens ein bis zwei prägende Merkmale (Kleidung,
ein charakteristischer Gegenstand, die Umgebung). Körperhaltung, Blickrichtung
oder ein Gegenstand gehören in den Alt-Text, wenn sie die Handlung oder Aussage
erst verständlich machen: ein weißer Langstock, ein Rollstuhl, ein Werkzeug in der
Hand, die Geste zur Leinwand. Sonst gehören sie in die Langbeschreibung. Bei
mehreren Personen nennst du Zahl und Konstellation.

Porträt: Nenne den Bildausschnitt (Kopf und Schultern, Halbfigur, ganze Figur), ob
die Person in die Kamera blickt, und den Hintergrund in einem Halbsatz.


LANGBESCHREIBUNG

Fließtext in dieser Reihenfolge, ohne Überschriften: Personen und Konstellation,
sichtbare Tätigkeit, Kleidung und prägende Gegenstände, Haltung und Blickrichtung
dort, wo sie die Szene nachvollziehbarer machen, Umgebung und Raum, lesbare Texte
und Logos, belegte Zusatzangaben aus dem Kontext. Ein Logo zählt, wenn es Beruf,
Organisation oder Ort der Person kennzeichnet (Firmenkleidung, Konferenzband).
Zusammenhänge statt Kleinigkeiten.


{_render_personen_block()}


{KUNSTWERK_REGEL}


{STILREGELN}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_foto_objekte(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """foto_objekte: Gegenstände, Produkte, Materialien, Objektgruppen.

    Fassung September 2026 nach dem Prompt-Standard. Die Behälter-Regel stammt
    aus der Zeit, in der Modelle Schüsselinhalte erfanden; sie bleibt in kurzer
    Form, weil die helle Innenfläche die häufigste Fehldeutung dieser Kategorie
    ist. Die frühere Warnliste aus dem Inventar wird nicht mehr gerendert.
    """
    examples = load_examples('foto_objekte')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: foto_objekte
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Foto, auf dem ein Gegenstand, ein Produkt oder eine Objektgruppe im
Mittelpunkt steht. Der Text benennt das Objekt so konkret, wie Bild und Kontext es
tragen (Typ, Bauart, Modell, Marke, lesbare Bezeichnung), und macht Form und
Beschaffenheit nachvollziehbar. Der Kontext sagt, wozu das Bild dient
(Produktseite, Anleitung, Katalog) und welche Merkmale deshalb zählen.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginne mit der konkretesten belegten Benennung, nicht mit einer Umschreibung:
"Akkubohrschrauber der Beispiel AG mit 18-Volt-Akku" statt "ein Werkzeug". Dann
die ein bis zwei Merkmale, die das Objekt kennzeichnen (Form, Farbe, Oberfläche,
Größenverhältnis), und lesbare Beschriftungen. Von Werbeaussagen auf einer
Verpackung nennst du höchstens die zwei, die das Produkt kennzeichnen; weitere
gehören in die Langbeschreibung. Bei Objektgruppen nennst du Zahl und Anordnung.


LANGBESCHREIBUNG

Fließtext in dieser Reihenfolge, ohne Überschriften: Objekt mit Benennung, Form
und Proportion, Oberfläche und Material, Anordnung im Raum, sichtbare Details und
Beschriftungen, belegte Angaben aus dem Kontext. Sie macht die sichtbare Form
nachvollziehbar, statt Eigenschaften aufzuzählen.


TYP, MATERIAL UND BEHÄLTER

- Typ und Bauart benennst du an unterscheidenden sichtbaren Merkmalen oder aus
  dem Kontext. Eine Materialangabe braucht einen belastbaren Anhaltspunkt
  (Maserung, Glasurriss, Naht, lesbare Angabe); Glanz und Farbe allein reichen
  nicht, dann beschreibst du die Oberfläche ("helle, glänzende Oberfläche").
- Herstellungsweise (handgetöpfert), Herkunft und momentane Nutzung nur mit
  eigenem Beleg.
- Behälter: Eine helle Innenfläche ist Glasur oder Oberfläche, keine Füllung.
  Sichtbar freie Innenräume darfst du leer nennen. Bei gestapelten oder
  verdeckten Behältern behauptest du nicht, alles gesehen zu haben.
- Sammlungen und Gruppen zählst du nach Belegregel 7.


{KUNSTWERK_REGEL}


{STILREGELN}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_foto_essen(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """foto_essen: Speisen, Getränke, Tischanrichtung, Buffet, verpackte Lebensmittel.

    Fassung September 2026 nach dem Prompt-Standard. Die frühere Erlaubt-Liste
    (knusprig, gedämpft) ist durch die Regel ersetzt, dass nur sichtbare
    Eigenschaften in den Text kommen und eine Zubereitungsart einen eigenen Beleg
    braucht.
    """
    examples = load_examples('foto_essen')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: foto_essen
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Foto, auf dem Speisen, Getränke, eine Tischanrichtung oder ein Buffet im
Mittelpunkt stehen. Der Text benennt das Gericht, wenn es erkennbar ist oder der
Kontext es nennt (Speisekarte, Rezepttitel, Bildunterschrift), und macht sichtbar,
woraus es erkennbar besteht und wie es angerichtet ist. Bei verpackten
Lebensmitteln kommen Marke und Produkt aus Etikett oder Aufdruck.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginne mit dem Gericht und der Servierform: "Lachsfilet mit gebräunter Kruste auf
grünem Spargel auf einem weißen Teller". Dann die erkennbaren Hauptkomponenten und
ein Merkmal der Anrichtung. Was du nicht sicher erkennst, beschreibst du nach
Aussehen ("eine helle Soße", "grünes Blattgemüse"). Geschirr und Umgebung in einem
Halbsatz, wenn sie die Szene kennzeichnen (Holztisch, Buffet, Pappschale). Kleinste
Details wie Poren, einzelne Krümel oder eine Maserung gehören nicht in den
Alt-Text.


LANGBESCHREIBUNG

Fließtext in dieser Reihenfolge, ohne Überschriften: Gericht, sichtbare
Komponenten und Beilagen mit ihrer Lage auf dem Teller, Anrichtung und Geschirr,
Umgebung (Restauranttisch, Küche, Buffet), lesbare Texte (Speisekarte, Etikett),
belegte Angaben aus dem Kontext (Rezeptname, Anlass). Kurz und zusammenhängend;
bei einem einfachen Teller darf sie leer bleiben.


SICHTBARES STATT GESCHMACK

Beschreibe, was zu sehen ist: eine gebräunte Kruste, dunkle Röststellen, eine
glänzende Oberfläche, klare Schnittflächen, Grillstreifen, aufsteigender Dampf.
Geschmack, Knusprigkeit, Frische und Zubereitungszeit lassen sich daraus nicht
ablesen; Wörter wie knusprig, frisch, hausgemacht oder lecker stehen nur, wenn der
Kontext sie trägt. Eine Zubereitungsart nennst du, wenn das Bild sie eindeutig
zeigt (Grillstreifen, ein Spieß über Glut) oder der Kontext sie nennt; eine
Bräunung allein reicht nicht. Zutaten nur, soweit sie erkennbar oder benannt
sind: Ohne sichtbare Kräuter gibt es keine Kräutergarnitur, eine helle Soße bleibt
eine helle Soße. Eine Herkunft oder Küche (italienisch, japanisch) nur aus
Beschriftung oder Kontext oder wenn die Form des Gerichts sie zweifelsfrei trägt
(Sushi-Rollen).


{STILREGELN}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_foto_landschaft(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """foto_landschaft: Küste, Gebirge, Wald, Feld, Fluss, Wüste, Stadtpanorama.

    Fassung September 2026 nach dem Prompt-Standard. Der frühere Widerspruch
    zwischen "Bergname nur mit Schild" und "Matterhorn aus Weltwissen" ist durch
    eine einzige Beleg-Definition im Abschnitt ORTE UND NAMEN ersetzt.
    """
    examples = load_examples('foto_landschaft')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: foto_landschaft
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Außenfoto, auf dem eine Landschaft oder ein geografischer Raum im Mittelpunkt
steht: Küste, Gebirge, Wald, Feld, Fluss, Wüste, Stadtpanorama. Der Text nennt die
Landschaftsart und ihre prägenden Merkmale konkret (Relief, Gewässer, Vegetation,
Bebauung, Wetter und Licht) und ordnet den Ort so ein, wie Bild oder Kontext ihn
belegen.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginne mit der Landschaftsart und dem Merkmal, das sie prägt: "Bergpanorama mit
drei schneebedeckten Gipfeln über einem Nadelwald, im Tal ein schmaler See." Dann
die zwei bis drei wichtigsten Elemente in räumlicher Ordnung; lesbare Orts- und
Wegschilder übernimmst du, ein belegter Ortsname steht vorn. Die vollständige
Staffelung des Raums und jedes Nebendetail trägt die Langbeschreibung.


LANGBESCHREIBUNG

Fließtext in dieser Reihenfolge, ohne Überschriften: Landschaftsart und
Gesamtraum (vorn, mittig, hinten, Tiefe), Relief und Gewässer, Vegetation und
Bodennutzung, Wetter und Licht, menschliche Eingriffe (Gebäude, Wege, Brücken),
lesbare Beschriftungen und belegte Angaben aus dem Kontext. Der Raum soll
nachvollziehbar werden; eine Stimmung nur mit dem sichtbaren Beleg im selben Satz.


ORTE UND NAMEN

Ein Name ist auf genau drei Wegen belegt: lesbar im Bild (Schild, Tafel),
ausdrücklich im Kontext, oder als weltbekanntes Wahrzeichen mit eindeutiger,
unverwechselbarer Silhouette (Matterhorn, Uluru, Golden Gate Bridge); dazu darf
ein einzelnes Kenn-Faktum stehen. Passen mehrere Orte plausibel auf das Motiv,
beschreibst du es ohne Eigennamen: "Bergpanorama mit hohen, schneebedeckten
Gipfeln", nicht "die Alpen". Ein erkannter Bergname rechtfertigt keine geratene
Aufnahmeposition, Route, Region oder Ortschaft. Schnee, gelbe Bäume, warmes Licht
oder lange Schatten sind sichtbare Merkmale und stehen als solche im Text; eine
Jahreszeit oder Tageszeit nennst du nur, wenn der Kontext sie belegt.


{STILREGELN}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_foto_architektur(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """foto_architektur: Gebäude, Bauwerk, Innenraum, Fassadendetail.

    Fassung September 2026 nach dem Prompt-Standard. Wahrzeichen werden beim
    Namen genannt (die Erlaubnis steht im System-Prompt); ein Bautyp braucht eine
    unterscheidende Merkmalskombination, eine Nutzung einen eigenen Beleg.
    """
    examples = load_examples('foto_architektur')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: foto_architektur
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Foto, auf dem ein Gebäude, Bauwerk, Innenraum oder Fassadendetail im
Mittelpunkt steht. Der Text benennt das Bauwerk beim Namen, wenn es ein
weltbekanntes Wahrzeichen ist oder Beschriftung oder Kontext den Namen nennen;
sonst nennt er den Bautyp, soweit die sichtbaren Merkmale ihn unterscheiden, und
beschreibt Bauform und Material.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginne mit dem Namen oder dem Bautyp und dem prägenden Merkmal: "Reithalle mit
hellem Sandboden und Holzbanden", "Bürogebäude mit Glasfassade". Zu einem
benannten Wahrzeichen höchstens ein Kenn-Faktum. Ist die Perspektive wichtig,
steht sie ohne Ansage vorn: "Blick von Südwesten auf den Dom". Dann Material und
die zwei bis drei markantesten Elemente (Dachform, Turm, Portal, Fassadenraster);
ein Gerüst, eine Baustelle oder eine Beschädigung nennst du, weil sie das Bild von
anderen Aufnahmen unterscheiden. Lesbare Beschriftungen (Hausnummer, Straßenname,
Inschrift, Tafel) übernimmst du wortgetreu.


LANGBESCHREIBUNG

Fließtext in dieser Reihenfolge, ohne Überschriften: Bauwerk und Gesamtform,
Fassade und Material, markante Elemente, Umgebung und Einbettung (Platz, Straße,
Nachbarbauten), lesbare Beschriftungen, belegte Angaben aus dem Kontext. Die
Bauform soll nachvollziehbar werden, ohne jedes Fenster und jede Säule einzeln zu
zählen.


BAUTYP UND NUTZUNG

Nenne den Bautyp, wenn die Kombination sichtbarer Merkmale ihn unterscheidet:
Sandboden, Banden und Hindernisstangen tragen eine Reithalle, Turm, Portal und
Spitzbogenfenster eine Kirche, Bahnsteige unter einem Hallendach einen Bahnhof.
Eine Halle mit Toren belegt keine Lagerfunktion; bei unklarer Nutzung beschreibst
du Halle, Tragwerk, Tore und Anordnung. Baujahr, Architekt, Stilepoche und heutige
Nutzung kommen aus Beschriftung oder Kontext, sonst fehlen sie. Ein Bauwerk ohne
weltbekannte Silhouette bekommt keinen Namen, auch keinen naheliegenden.


{STILREGELN}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""
