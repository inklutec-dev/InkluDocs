"""Pass-3-Builder für Daten-Visualisierung + Spezialfälle.

7 Builder: illustration, diagramm, tabelle, karte, infografik, screenshot,
strukturformel. Alle haben Inventar-Pass.

PAKET 4 (16.07.2026): Alle 7 Builder auf das Premium-Muster von
beschreibung_foto.py (foto_objekte als Vorlage) gehoben:
  ROLE_BESCHREIBER + ANTI_HALLUZINATION_REGELN (einmal pro gerendertem
  Prompt — siehe _basis_schichten) + BILDTYP/BILDGROESSE + ZIEL + geteilte
  Blöcke (Inventar, Kontext, Zweck, Kompaktheit — siehe _render_*-Helper)
  + kategorie-spezifische Regeln in ruhiger Formulierung + Few-Shot via
  load_examples + AUSGABE-SCHEMA inline + FINAL CHECK als Prüfliste.

- Der frühere diagramm-Zwitter (Premium-Lean-Zweig + Mistral-Full-Fallback
  vom 14.04.2026) ist aufgelöst: EIN Builder für beide Modi, Basis ist der
  Premium-Lean-Zweig vom 15.05.2026. Die Full-Zweig-Pflichten (Kontaktdaten,
  Insight-first mit konkreten Werten, Vollständigkeit der Langbeschreibung,
  Konsistenz alt_text/langbeschreibung, keine Markdown-Tabellen) leben in
  den Sektionen und im FINAL CHECK des einen Builders weiter.
- Fachlich wertvolle Spezialregeln sind INHALTLICH erhalten, nur vom alten
  KRITISCH/MUSS-Drill-Ton auf die ruhige Premium-Formulierung gebracht:
  Bilanz-Regel + Spalten-Zuordnung (tabelle), Ortsnamen wortgetreu in
  Originalsprache + EIGENNAMEN_REGELN (karte), OCR als Primärquelle +
  Kontaktdaten (tabelle/infografik/screenshot), Layout-Beschreibungs-Verbot
  (infografik), Screenreader-Notation CH3 (strukturformel), Screenshot-
  Präfix-Regel, Diagramm-Subtypen + Trend-Vokabular, Illustration-
  Alternativen-Form ("als X oder Y deutbar").
- Längenregime: Richtwerte aus Paket 1 unverändert (tabelle/strukturformel
  unter 250, karte/infografik/screenshot unter 350; Langbeschreibung etwa
  1500/1000/800 für infografik/screenshot/strukturformel); harte Obergrenze
  ist allein das Schema (400/2000).

ATMOSPHAERE_REGEL gilt für diese Bildtypen NICHT — Daten sind objektiv,
Wertungen über Atmosphäre haben hier keinen Platz. (Ausnahme: infografik
EINGESCHRÄNKT bei Kampagnen-Infografiken — die Einschränkung ist seit
Paket 4 direkt in die ATMOSPHAERE-Sektion des infografik-Builders
integriert statt pauschal vorangestellt.)
"""
from __future__ import annotations

import os
from typing import Optional

from prompts.components.constraints import (
    ANTI_HALLUZINATION_REGELN,
    ATMOSPHAERE_REGEL,
    EIGENNAMEN_REGELN,
    KONTAKTDATEN_PFLICHT,
)
from prompts.components.roles import ROLE_BESCHREIBER
from prompts.components.schemas import InventarOutput

from .helpers import load_examples, user_hint_block


# =====================================================================
# Geteilte Helper-Blöcke der Daten-Familie (Paket 4, 16.07.2026)
# =====================================================================
# Der Refactor-Plan aus der Lean-Iteration 15.05.2026 ('Helper-Kandidat'-
# Markierungen im diagramm-Builder) ist damit umgesetzt: die in allen
# 7 Buildern identischen Sektionen leben an EINER Stelle — Drift-Vermeidung
# wie bei den Foto-Helpern in beschreibung_foto.py.


def _basis_schichten() -> str:
    """Rolle + Anti-Halluzinations-Schicht für den Prompt-Kopf.

    Combo-Deduplizierung (Paket 4): Im Lean-Pass-Modus (V4_PASS_MODE=lean)
    wird der Beschreibungs-Prompt ausschließlich von combo.py in den
    kombinierten Prompt eingebaut — dort trägt der Inventar-Teil
    (inventar.py) die ANTI_HALLUZINATION_REGELN bereits. Die Schicht hier
    ein zweites Mal einzubauen würde sie im selben gerenderten Prompt
    duplizieren. Im Full-Pass-Modus (Default) ist dieser Prompt der
    eigenständige Pass 3 und trägt die Schicht selbst.
    combo.py und alle Builder-Signaturen bleiben unverändert — die
    Entscheidung fällt hier über dieselbe ENV, die auch der Orchestrator
    für die Pfad-Wahl liest.
    """
    if os.environ.get('V4_PASS_MODE', 'full').strip().lower() == 'lean':
        return ROLE_BESCHREIBER
    return f'{ROLE_BESCHREIBER}\n\n{ANTI_HALLUZINATION_REGELN}'


def _render_inventar_block(inventar_json: str) -> str:
    """Inventar-Sektion — identisch in allen 7 Daten-Buildern."""
    return f"""INVENTAR (Pass-2-Beobachtungen)

Das Inventar enthält die strukturierten Beobachtungen aus dem Analyse-Pass.
Nutze diese Daten als primäre faktische Grundlage. Sichtbare
Bildinformationen dürfen ergänzt werden, dürfen dem Inventar aber nicht
widersprechen.

{inventar_json}"""


def _render_kontext_block(enriched_context: str, user_hint_text: str) -> str:
    """Kontext-Sektion inkl. Bild-gewinnt-Regel — alle 7 Daten-Builder."""
    return f"""KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt, Bildunterschriften oder
API-Aufrufen stammen. Er hilft, die Grafik fachlich einzuordnen. Ohne
Kontext beschreibst du ausschließlich sichtbar belegbare Bildinformationen;
fehlender Kontext wird nicht durch Vermutungen ersetzt.

BILD GEWINNT GEGEN KONTEXT: Bei Widerspruch zwischen sichtbaren Werten oder
Beschriftungen und dem Kontext gelten die sichtbaren Bildinformationen.

{enriched_context if enriched_context else '(kein Kontext)'}
{user_hint_text}"""


def _render_zweck_block() -> str:
    """Bild-Zweck-Block der Daten-Familie (Pendant zum Foto-Zweck-Block).

    Gleiche Grenze wie in beschreibung_foto.py: der Zweck steuert die
    GEWICHTUNG, er erlaubt keine neuen unbelegten Fakten.
    """
    return """BILD-ZWECK IM DOKUMENT

Der Kontext zeigt, WO und WOZU die Grafik verwendet wird. Leite daraus den
kommunikativen Zweck ab: Warum steht diese Grafik an genau dieser Stelle?
Priorisiere die Aspekte, die diesen Zweck bedienen — dieselbe Tabelle braucht
im Geschäftsbericht eine andere Gewichtung als im Schulbuch oder in einer
Pressemitteilung. Der Zweck steuert nur die GEWICHTUNG und Auswahl; er erlaubt
KEINE neuen Fakten, die Bild oder Kontext nicht belegen. Ohne Kontext: neutral
informativ beschreiben.

ANTI-REDUNDANZ ZUR BILDUNTERSCHRIFT: Wiederhole keine Details, die die
Bildunterschrift bereits nennt — Titel, Thema und Kernaussage dagegen IMMER
nennen (der Alt-Text muss allein verständlich sein)."""


def _render_kompaktheit_block(alt_richtwert: str, lang_richtwert: Optional[str] = None) -> str:
    """Kompaktheits-Block mit kategorie-eigenem Richtwert (Paket-1-Regime).

    Harte Obergrenze ist allein das Schema (400/2000); die Richtwerte sind
    Orientierung, kein Ziel.
    """
    if lang_richtwert:
        lang_satz = (
            f'Richtwert für die Langbeschreibung: {lang_richtwert}; '
            'harte Obergrenze sind die 2000 Zeichen des Schemas.'
        )
    else:
        lang_satz = 'Die Langbeschreibung nutzt maximal 2000 Zeichen (Schema-Obergrenze).'
    return f"""KOMPAKTHEIT (Arbeitsteilung Alt-Text / Langbeschreibung)

Richtwert für den Alt-Text: {alt_richtwert}. Die 400 Zeichen des Schemas sind
eine harte Obergrenze, KEIN Ziel. Der Alt-Text trägt die Kernaussage —
Vollständigkeit, Einzelwerte und Struktur-Tiefe gehören in die
Langbeschreibung. {lang_satz}"""


def _render_atmosphaere_verzicht_block(begruendung: str) -> str:
    """Atmosphäre-Verzicht der Daten-Familie (illustration/tabelle/karte/
    screenshot/strukturformel — diagramm und infografik haben eigene,
    inhaltlich abweichende Sektionen)."""
    return f"""ATMOSPHAERE

{begruendung} Keine Wertungen über Stimmung oder Wirkung;
atmosphaere_belege bleibt leer."""


def _render_ausgabe_schema_block(alt_hinweis: str, atmosphaere_hinweis: str) -> str:
    """AUSGABE-SCHEMA-Sektion inline (Premium-Muster statt schema_doc-Anhängsel)."""
    return f"""AUSGABE-SCHEMA

Fülle exakt das Schema BeschreibungOutput:
- alt_text: 20 bis 400 Zeichen, {alt_hinweis}
- langbeschreibung: maximal 2000 Zeichen, leer wenn alt_text alles
  Wesentliche sagt
- verwendete_inventar_items: Audit-Trail der genutzten Inventar-Items
- nicht_verwendete_inventar_items: Audit-Trail der bewusst ausgelassenen Items
- nicht_im_inventar: MUSS leer bleiben
- atmosphaere_belege: {atmosphaere_hinweis}

Kein Markdown im Output — Fließtext oder einfache Satzlisten, keine
Markdown-Tabellen."""


def build_beschreibung_prompt_illustration(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder für illustration (Paket 4, 16.07.2026).

    Cartoon, Vektor-Grafik, gemalte Illustration, Buch-Bild — höchstes
    Fehldeutungs-Risiko (Spezies-Festlegung, erfundene Interaktionen).
    Erhalten: Spezialwarnung für stilisierte Darstellungen, Spezies-/
    Charakter-Regel mit der Zwei-Wege-konformen Alternativen-Form
    ('als Katze oder Fuchs deutbar', Paket-1-Korrektur), Interaktions-Regel
    mit Hund/Mikroskop-Beispiel, Vollständigkeits-Hinweis für Nebenelemente,
    Spezifitäts-Anforderung an den ersten Satz.
    """
    examples = load_examples('illustration')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: illustration (Cartoon, Vektor-Grafik, gemalte Illustration, Buch-Bild)
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung für
eine Illustration. Ziel ist ehrliche Spezifität: benenne, was die Darstellung
klar trägt, und beschreibe neutral, was genuin mehrdeutig bleibt. Stilisierte
Darstellungen sind die häufigste Quelle für Fehldeutungen — vereinfachte
Cartoon-Motive werden leicht als etwas anderes gesehen, mehrdeutige Charaktere
leicht auf das naheliegendste Klischee festgelegt. Genau das vermeidest du.


{_render_inventar_block(inventar_json)}


{_render_kontext_block(enriched_context, user_hint_text)}


{_render_zweck_block()}


{_render_kompaktheit_block('einfache Motive unter 150 Zeichen, komplexe Illustrationen bis etwa 250')}


ALT-TEXT

Der erste Satz nennt:
- die Stilrichtung (Cartoon, Vektor, gemalt, comic-haft etc.)
- das Hauptmotiv mit ehrlicher Spezifität
- mindestens ein konkretes Element

VERMEIDEN: generische Einleitungen ("Das Bild zeigt", "Eine Illustration von"
als bloße Floskel ohne Inhalt), Festlegung auf eine Deutung, die das Bild
nicht trägt.


SPEZIES- UND CHARAKTER-REGEL

Wenn das Inventar bei einem Charakter Mehrfach-Hypothesen oder niedrige
Sicherheit listet, bildet der Output diese Unsicherheit ab — ohne
Vermutungswörter (kein 'vermutlich', 'wahrscheinlich', 'könnte'). Beschreibe
die Form neutral; wenn zwei Deutungen naheliegend und bildrelevant sind,
nenne beide gleichwertig als Alternativen:
- 'stilisiertes Tier mit großen Augen, als Katze oder Fuchs deutbar'
- 'Cartoon-Charakter mit [konkreten sichtbaren Merkmalen]'
- NICHT: einfach die wahrscheinlichste Spezies festlegen
- NICHT: Hedge-Formulierungen wie 'vermutlich eine Katze'


INTERAKTIONEN NUR MIT BELEG

Bei Illustrationen ist die Interaktions-Regel besonders wichtig: Wenn das
Inventar nur Objekte nebeneinander listet, schreibe nicht, dass sie
miteinander interagieren.
- Inventar: 'Hundekopf, Mikroskop, Laptop, Tablet, Smartphone — keine Hände sichtbar'
- FALSCH: 'Der Hund arbeitet am Laptop und hält ein Tablet.'
- RICHTIG: 'Cartoon-Illustration eines Hundekopfes; daneben ein Mikroskop und drei
  Geräte (Laptop, Tablet, Smartphone).'


VOLLSTÄNDIGKEIT

Bei Illustrationen werden Nebenelemente besonders häufig übersehen (z.B. das
Mikroskop im Hund-Bild). Gehe das Inventar vollständig durch und benenne alle
sichtbaren Elemente — auch wenn sie unscheinbar wirken.


LANGBESCHREIBUNG

Sinnvolle Reihenfolge: Stilrichtung und Hauptmotiv -> zentrale Charaktere
oder Objekte mit ihren sichtbaren Merkmalen -> Nebenelemente vollständig ->
lesbare Texte oder Beschriftungen -> relevanter Kontext. Fließtext, keine
Markdown-Formatierung.


{_render_atmosphaere_verzicht_block('Illustrationen werden sachlich beschrieben.')}


{_render_ausgabe_schema_block('spezifisch und ehrlich', 'bleibt leer')}


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Nennt der erste Satz Stilrichtung, Hauptmotiv und mindestens ein
   konkretes Element?
2. Ist jede Aussage durch Inventar oder sichtbare Bildinformation belegt?
3. Bei mehrdeutigen Charakteren: neutrale Form oder gleichwertige
   Alternativen ('als X oder Y deutbar') statt Festlegung oder
   Vermutungswörtern?
4. Keine Interaktion erfunden, die das Inventar nicht belegt?
5. Alle Inventar-Elemente berücksichtigt — auch unscheinbare Nebenelemente?
6. nicht_im_inventar leer?
7. Schema vollständig korrekt (alle Pflichtfelder gefüllt)?

Wenn ein Punkt nicht erfüllt ist: Output neu formulieren.
"""


def build_beschreibung_prompt_diagramm(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder für diagramm — EIN Builder für beide Modi (Paket 4).

    Basis ist der Premium-Lean-Zweig der Sonnet-Iteration vom 15.05.2026
    (Diagramm-Logik mit 6 Sub-Typen, Trend-Vokabular, Kernaussage-zuerst,
    10-Punkte-Final-Check). Der alte Mistral-Full-Zweig vom 14.04.2026 ist
    ersetzt — seine Pflichten leben weiter: Kontaktdaten/lesbare Texte
    (Sektion LESBARE TEXTE), Kernaussage mit konkreten Werten (ALT-TEXT +
    gute/schlechte Beispiele), Vollständigkeit der Langbeschreibung
    (Struktur-Punkte 6-7), Konsistenz alt_text/langbeschreibung
    (FINAL CHECK Punkt 3), keine Markdown-Tabellen (LESBARE TEXTE +
    AUSGABE-SCHEMA). Neu gegenüber dem Lean-Zweig: ROLE_BESCHREIBER im
    Prompt-Kopf (fehlte dort), Zweck- und Kompaktheits-Block; die
    'Helper-Kandidat'-Pseudokommentare sind aus dem Modell-Text in die
    tatsächlichen Helper-Funktionen dieses Moduls überführt.
    """
    examples = load_examples('diagramm')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: diagramm (Balken, Linie, Kreis, gestapelt, Streu, Heatmap)
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
für ein Diagramm.

Der Fokus liegt NICHT auf bloßer Bildbeschreibung, sondern auf
verständlicher Wissensvermittlung.

Das Ziel ist:
- Trends verständlich machen
- Vergleiche sichtbar machen
- Rangfolgen erklären
- Entwicklungen über Zeit beschreiben
- die Kernaussage des Diagramms erfassbar machen

Der Alt-Text soll die wichtigste Erkenntnis transportieren.
Die Langbeschreibung liefert die vollständige nachvollziehbare Struktur.

Beschreibe nicht nur, WAS sichtbar ist.
Erkläre, welche INFORMATION das Diagramm vermittelt.

KEINE Ursachenbehauptungen erfinden (wirtschaftlich, politisch, fachlich).
KEINE Daten oder Kategorien halluzinieren — wenn unlesbar, ehrlich
benennen.


{_render_inventar_block(inventar_json)}


{_render_kontext_block(enriched_context, user_hint_text)}


{_render_zweck_block()}


{_render_kompaktheit_block('einfache Diagramme unter 150 Zeichen, komplexe bis etwa 250')}


ALT-TEXT

Der Alt-Text:
- beginnt konkret
- nennt Diagrammtyp und Titel oder Thema
- priorisiert 2-3 zentrale Erkenntnisse
- nennt wichtige Werte oder Extreme
- fasst Trends verständlich zusammen

KERNAUSSAGE ZUERST:

Der erste Satz soll die wichtigste Aussage des Diagramms vermitteln —
mit konkreten Werten, wo sie die Aussage tragen.

NICHT:
- reine Aufzählung von Balken oder Linien
- bloße Farbbeschreibung
- isolierte Zahlenlisten ohne Zusammenhang

BEVORZUGEN:
- Trendbeschreibung
- Rangfolge
- Vergleich
- Veränderung über Zeit
- Dominanz oder Verhältnis

VERMEIDEN:
- "Das Diagramm zeigt ..."
- "Zu sehen sind ..."
- generische Formulierungen
- reine Datenpunkt-Aufzählungen
- vage Aussagen ohne Zahlenbezug ('China führt, gefolgt von den USA' —
  besser: 'China führt mit 251,8 Mrd. Euro, Tschechien bildet mit
  115,7 Mrd. das Schlusslicht.')
- unbelegte Interpretationen

GUTE BEISPIELE:
- "Paid Search dominiert zunächst deutlich, fällt ab 2019 jedoch stark ab."
- "AWS bleibt Marktführer mit 34 Prozent Marktanteil vor Azure und Google Cloud."
- "Der Umsatz steigt über drei Jahre kontinuierlich um fast 50 Prozent."

SCHLECHTE BEISPIELE:
- "Mehrere Linien verlaufen durch das Diagramm."
- "Verschiedene Balken mit unterschiedlichen Höhen."
- "Das Diagramm wirkt positiv."


LANGBESCHREIBUNG

Struktur in dieser Reihenfolge:

1. Diagrammtyp und Thema
2. Achsen / Kategorien / Zeiträume
3. Haupttrend oder Hauptstruktur
4. Vergleich der wichtigsten Kategorien
5. Relevante Extremwerte oder Wendepunkte
6. Vollständige Werte oder Reihen — alle Kategorien aus
   inventar.lesbare_texte mit ihren lesbaren Werten, bei Zeitreihen
   Anfangs- und Endwerte
7. Sichtbare Zusatzinformationen (Achsenbeschriftungen, Legenden-Werte)
8. Kontext nur wenn eindeutig passend

Die Langbeschreibung soll:
- nachvollziehbar strukturiert sein
- keine unverbundenen Zahlenlisten erzeugen
- Beziehungen zwischen Werten erklären
- den Verlauf verständlich machen


DIAGRAMM-LOGIK

BALKENDIAGRAMM:
Fokus auf:
- Vergleich
- Rangfolge
- größte/kleinste Kategorie
- Unterschiede zwischen Balken
- Veränderungen zwischen Jahren oder Gruppen

LINIENDIAGRAMM:
Fokus auf:
- Verlauf über Zeit
- Anstieg / Rückgang
- Schwankungen
- Plateaus
- Wendepunkte
- Volatilität
- langfristige Trends

KREISDIAGRAMM:
Fokus auf:
- Anteile
- Dominanz
- größte und kleinste Segmente
- Verhältnis der Gruppen zueinander

GESTAPELTES DIAGRAMM:
Fokus auf:
- Zusammensetzung
- Veränderungen innerhalb der Gesamtmenge
- dominante Teilbereiche

STREUDIAGRAMM:
Fokus auf:
- Cluster
- Ausreißer
- Korrelationen
- Konzentrationen sichtbarer Punkte

HEATMAP / FARBSKALEN:
Fokus auf:
- Intensität
- Verteilung
- Konzentrationsbereiche
- sichtbare Muster

Trend-Vokabular darf genutzt werden wenn sichtbar belegt:
- kontinuierlicher Anstieg
- rückläufig
- stagnierend
- stark schwankend
- Plateau
- Spitzenwert
- Tiefpunkt
- deutlicher Einbruch
- leichte Erholung
- stabil auf Niveau X


LESBARE TEXTE / KONTAKTDATEN

Lesbare Texte aus dem Diagramm differenziert behandeln:

IMMER wortgetreu übernehmen:
- URLs
- Telefonnummern
- Datumsangaben
- Zahlenwerte
- Achsenbeschriftungen
- Kategorienamen

Titel, Legenden und Beschriftungen übernehmen, wenn sie zum Verständnis
des Diagramms beitragen.

Keine Markdown-Tabellen im JSON-Output verwenden.


ATMOSPHAERE

Diagramme haben normalerweise keine Atmosphäre-Beschreibung.
atmosphaere_belege bleibt in der Regel leer. Nur bei eindeutig
gestalterischer Wirkung mit belegbaren visuellen Hinweisen darf eine
sehr zurückhaltende Aussage verwendet werden.


{_render_ausgabe_schema_block('an der Kernaussage orientiert und konkret', 'bei Diagrammen normalerweise leer')}


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Sind alle Aussagen durch sichtbare Daten belegbar?
2. Enthält der Alt-Text eine echte Kernaussage statt bloßer Beschreibung?
3. Stimmen Trend-Aussagen im alt_text mit den konkreten Werten in der
   Langbeschreibung überein?
4. Wurden keine Ursachen oder Bedeutungen erfunden?
5. Sind Diagrammtyp und Struktur korrekt beschrieben?
6. Sind wichtige Werte, Extrempunkte oder Vergleiche enthalten?
7. Wurden keine Daten oder Kategorien halluziniert?
8. Ist nicht_im_inventar leer?
9. Ist der Alt-Text konkret statt generisch?
10. Ist die Langbeschreibung vollständig und nachvollziehbar strukturiert?

Wenn ein Punkt nicht erfüllt ist:
Output neu formulieren.
"""


def build_beschreibung_prompt_tabelle(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder für tabelle (Paket 4, 16.07.2026).

    Tabellarische Daten als Grafik. Erhalten (entdrillt, inhaltlich
    unverändert): Bilanz-Regel (Zwischensummen vs. Bilanzsumme, mit
    FALSCH/RICHTIG-Beispiel), Spalten-Zuordnung (5 Lese-Schritte),
    Einheiten-Treue, OCR als Primärquelle, KONTAKTDATEN_PFLICHT
    (geteiltes Constraint-Modul), Vollständigkeits-Reihenfolge der
    Langbeschreibung inkl. 4x4-Regel, keine Markdown-Tabellen.
    Richtwert 250 (Paket 1) unverändert.
    """
    examples = load_examples('tabelle')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: tabelle (tabellarische Daten als Grafik)
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung für
eine als Grafik vorliegende Tabelle. Sehende erfassen Tabellen zuerst nach
ihrer Kernaussage, nicht nach ihrer Form — der Alt-Text transportiert deshalb
die wichtigste Aussage auf Basis der RICHTIGEN Endwerte, die Langbeschreibung
macht die Struktur mit korrekter Spaltenzuordnung nachvollziehbar. Präzision
bei Zahlen, Summen und Einheiten ist hier der Qualitätsmaßstab.


{_render_inventar_block(inventar_json)}


{_render_kontext_block(enriched_context, user_hint_text)}


{_render_zweck_block()}


{_render_kompaktheit_block('unter 250 Zeichen')}


ALT-TEXT

Der erste Satz:
- beginnt mit 'Tabelle —' + Thema (aus inventar.lesbare_texte: Titel-Eintrag)
- nennt die wichtigste Aussage, gestützt auf die richtigen
  Endwerte bzw. die Bilanzsumme

VERMEIDEN: nichtssagende Eröffnungen wie 'Eine Tabelle zeigt verschiedene
Werte.'


BILANZ-REGEL (bei Buchhaltungs- und Bilanztabellen)

Unterscheide Abschnitts-Zwischensummen von der Gesamtsumme:
- 'Anlagevermögen' und 'Umlaufvermögen' sind ABSCHNITTE (Zwischensummen)
- 'Bilanzsumme', 'Bilanzsumme Aktiva', 'Gesamtsumme' oder 'Summe
  Aktiva/Passiva' ist das GESAMTERGEBNIS
- Die letzte Summenzeile der Tabelle ist fast immer die Bilanzsumme,
  nicht ein Abschnittswert
- FALSCH: 'Gesamtsumme des Umlaufvermögens beträgt X EUR' wenn X die
  Bilanzsumme ist
- RICHTIG: 'Bilanzsumme Aktiva beträgt X EUR' oder 'Gesamtsumme Aktiva
  beträgt X EUR'


SPALTEN-ZUORDNUNG

So liest du die Tabelle korrekt:
1. Lies zuerst alle Spaltenköpfe von links nach rechts
2. Lies dann jede Zeile und ordne jeden Wert seiner exakten Spalte zu
3. Verwechsle Zwischenwerte (Zugänge, Abschreibungen, Veränderungen)
   nicht mit Bestands- oder Endwerten
4. Wenn eine Zeile Werte in der Spalte '01.01.' UND '31.12.' hat, sind das
   verschiedene Werte — nenne beide mit Spaltenzuordnung
5. Bei Buchhaltungstabellen sind Anfangs- und Endbestand die entscheidenden
   Werte, nicht die Bewegungen dazwischen

EINHEITEN: %, EUR, Mio., Tsd. penibel übernehmen — nicht weglassen,
nicht umformen.


LANGBESCHREIBUNG

Reihenfolge und Umfang:
1. Gesamtsumme/Bilanzsumme zuerst nennen — sie ist die wichtigste Zahl und
   darf nicht am Ende abgeschnitten werden
2. Spaltenköpfe wortgetreu auflisten
3. Alle Zeilen mit korrekter Spaltenzuordnung
4. Spitzen- und Tiefstwerte benennen, auffällige Muster
5. Nur bei sehr kleinen Tabellen (max 4x4) alle Werte einzeln auflisten
6. Bei größeren Tabellen: Zusammenfassung statt vollständige Wertliste

Keine Markdown-Tabellen im JSON-Output — Fließtext oder strukturierte Liste.


OCR-TEXT ALS PRIMÄRQUELLE

Wenn inventar.lesbare_texte Zellinhalte aus OCR enthält, sind diese die
primäre Wahrheitsquelle. Bei Konflikt zwischen OCR-Werten und visueller
Wahrnehmung: dem OCR-Text vertrauen.


{KONTAKTDATEN_PFLICHT}


{_render_atmosphaere_verzicht_block('Tabellen sind Daten, keine Stimmungen.')}


{_render_ausgabe_schema_block('beginnt mit Tabelle — + Thema und Kernaussage', 'bleibt leer')}


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Beginnt der Alt-Text mit 'Tabelle —' + Thema und der wichtigsten Aussage?
2. Ist die letzte Summenzeile korrekt eingeordnet (Bilanzsumme/Gesamtsumme,
   nicht als Abschnitts-Zwischensumme ausgegeben)?
3. Ist jeder genannte Wert seiner exakten Spalte zugeordnet (keine
   Zwischenwerte als Endwerte)?
4. Einheiten (%, EUR, Mio., Tsd.) wortgetreu übernommen?
5. Lesbare Kontaktdaten, URLs, Daten und Zahlen wortgetreu enthalten?
6. Keine Markdown-Tabellen im Output?
7. nicht_im_inventar leer?

Wenn ein Punkt nicht erfüllt ist: Output neu formulieren.
"""


def build_beschreibung_prompt_karte(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder für karte (Paket 4, 16.07.2026).

    Landkarte, Stadtplan, Lageplan, Übersichtskarte. Erhalten (entdrillt):
    'Karte —'-Eröffnung mit Gebiet + Thema + räumlicher Kernaussage,
    Ortsnamen wortgetreu in Originalsprache (Bordeaux/İstanbul/Köln),
    EIGENNAMEN_REGELN (TURKU/Turkey — im Bild lesbarer Name schlägt Kontext),
    Standort-Vollständigkeit + Legenden-Erklärung + Himmelsrichtungen,
    Hintergrund-Ortsnamen-Begrenzung, Symbolik-aus-der-Legende-Regel,
    keine erfundenen Orte/Routen. Richtwert 350 (Paket 1) unverändert.
    """
    examples = load_examples('karte')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: karte (Landkarte, Stadtplan, Lageplan, Übersichtskarte)
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung für
eine Karte. Ziel ist räumliche Orientierung aus Text: Gebiet, Thema und die
räumliche Kernaussage zuerst, dann die markierten Standorte und die Legende
so, dass ein blinder Nutzer die Verteilung nachvollziehen kann. Ortsnamen
sind hier heikel — sie werden wortgetreu und in Originalsprache übernommen,
nie geraten und nie aus dem Kontext 'korrigiert'.


{_render_inventar_block(inventar_json)}


{_render_kontext_block(enriched_context, user_hint_text)}


{_render_zweck_block()}


{_render_kompaktheit_block('unter 350 Zeichen')}


ALT-TEXT

Der erste Satz:
- beginnt mit 'Karte —' + Gebiet (Stadt, Region, Land — aus Inventar
  oder Kontext)
- nennt das Hauptthema (was wird dargestellt?)
- nennt die räumliche Kernaussage (z.B. 'Konzentration im Süden',
  'gleichmäßig verteilt', 'Cluster in den Großstädten')


ORTSNAMEN UND EIGENNAMEN

Ortsnamen in Originalsprache beibehalten:
- 'Bordeaux' nicht 'Bordeo'
- 'İstanbul' nicht 'Istanbul' (wenn das I-Punkt-Zeichen lesbar ist)
- 'Köln' nicht 'Cologne' (auch wenn der Kontext englisch ist)

{EIGENNAMEN_REGELN}

Keine Orte oder Routen erfinden, die nicht im Inventar stehen. Bei
verschwommenen Details: 'Details teilweise nicht lesbar' — statt zu raten.


LANGBESCHREIBUNG

Reihenfolge und Umfang:
1. Markierte Standorte vollständig auflisten (das sind die
   Kerninformationen einer Karte mit Markierungen)
2. Legende explizit erklären (Symbole, Farben, Größen-Bedeutungen)
3. Räumliche Verteilung beschreiben — mit Himmelsrichtungen statt nur
   'oben/unten', wenn die Karte geografisch ausgerichtet ist (Norden oben)
4. Maßstab nennen wenn lesbar

HINTERGRUND-ORTSNAMEN:
Bei Karten mit vielen Hintergrund-Städten zur Orientierung NICHT erschöpfend
auflisten — nur die beschrifteten/markierten relevanten Standorte. Andere
Städte erwähnen, wenn sie für die räumliche Einordnung wichtig sind
('zwischen München und Stuttgart').

SYMBOLIK AUS DER LEGENDE:
- Rote Markierungen sind nicht automatisch 'Warnungen' oder 'Gefahren' —
  die Bedeutung kommt aus der Legende
- Größenunterschiede von Markern (große vs. kleine Kreise) bedeuten meist
  unterschiedliche Werte — Legende prüfen


{_render_atmosphaere_verzicht_block('Karten werden sachlich-räumlich beschrieben.')}


{_render_ausgabe_schema_block('beginnt mit Karte — + Gebiet und räumlicher Kernaussage', 'bleibt leer')}


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Beginnt der Alt-Text mit 'Karte —' + Gebiet, Hauptthema und räumlicher
   Kernaussage?
2. Alle Ortsnamen wortgetreu und in Originalsprache (im Bild lesbarer Name
   schlägt Kontext)?
3. Alle markierten Standorte in der Langbeschreibung, Legende erklärt?
4. Symbol-Bedeutungen aus der Legende statt aus Annahmen?
5. Keine Orte oder Routen erfunden; Unlesbares ehrlich benannt?
6. nicht_im_inventar leer?

Wenn ein Punkt nicht erfüllt ist: Output neu formulieren.
"""


def build_beschreibung_prompt_infografik(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder für infografik (Paket 4, 16.07.2026).

    Schaubild, Übersichtsgrafik mit Stationen oder Schritten. Erhalten
    (entdrillt): 'Infografik —'-Eröffnung mit Kernaussage + Datenpunkten
    (RICHTIG/FALSCH-Beispiel), Stationen-Logik mit Beziehungen, Zahlen-
    Vollständigkeit, das Layout-Beschreibungs-Verbot mit den inhaltlichen
    Gegenbeispielen, OCR als Pflichtquelle, Kontaktdaten/URL-Übernahme bei
    Behörden-Infografiken. Die ATMOSPHAERE_REGEL ist mit ihrer bestehenden
    Einschränkung (nur Kampagnen-Infografiken mit bewusster Designwahl)
    sauber in die ATMOSPHAERE-Sektion integriert statt pauschal
    vorangestellt. Richtwerte 350/1500 (Paket 1) unverändert.
    """
    examples = load_examples('infografik')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: infografik (Schaubild, Übersichtsgrafik mit Stationen oder Schritten)
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung für
eine Infografik. Infografiken übersetzen Inhalte in visuelle Anordnung — deine
Aufgabe ist die Rückübersetzung: die inhaltliche Logik (Stationen, Schritte,
Beziehungen, Zahlen) verständlich machen, nicht das Layout nacherzählen.
Der Alt-Text trägt die Kernaussage, die Langbeschreibung die vollständige
inhaltliche Struktur.


{_render_inventar_block(inventar_json)}


{_render_kontext_block(enriched_context, user_hint_text)}


{_render_zweck_block()}


{_render_kompaktheit_block('unter 350 Zeichen', 'etwa 1500 Zeichen')}


ALT-TEXT

Der erste Satz:
- beginnt mit 'Infografik —' + Hauptthema (aus inventar.lesbare_texte: Titel)
- nennt die zentrale Kernaussage mit konkreten Datenpunkten, wenn vorhanden

Beispiel RICHTIG: 'Infografik — Bürokratie-Entlastung 2024: Fast die
Hälfte (39%) entfällt auf das Wachstumschancengesetz, gefolgt von
vier weiteren Maßnahmen mit zusammen 61%.'

Beispiel FALSCH: 'Eine Infografik mit verschiedenen Daten.'


LANGBESCHREIBUNG

Reihenfolge und Umfang:
1. Alle inhaltlichen Stationen in LOGISCHER Reihenfolge auflisten
   (chronologisch / hierarchisch / kausal — je nach Infografik-Typ)
2. Beziehungen zwischen Stationen benennen ('A führt zu B',
   'X umfasst Y', 'Schritt 1 aktiviert Schritt 2')
3. Alle konkreten Zahlen, Prozente, Mengenangaben übernehmen

Fließtext oder strukturierte Liste, keine Markdown-Tabellen.


INHALTLICH STATT LAYOUT

Visuelle Layout-Beschreibungen vermeiden:
- 'oben links steht...'
- 'ein Pfeil zeigt von X nach Y...'
- 'im Zentrum befindet sich...'
- 'die linke Hälfte des Bildes zeigt...'

Stattdessen inhaltlich formulieren:
- 'Schritt 1 ist X, daraus folgt Schritt 2 mit Y'
- 'Hauptkategorie A umfasst die Unterkategorien B, C und D'
- 'Im Mittelpunkt steht das Konzept X' (wenn das WIRKLICH die
  inhaltliche Botschaft ist, nicht nur die geometrische Position)


OCR-TEXT ALS PFLICHTQUELLE

Wenn inventar.lesbare_texte Beschriftungen enthält, sind diese wortgetreu
zu übernehmen. Bei Konflikt zwischen visueller Wahrnehmung und OCR:
dem OCR-Text vertrauen.

KONTAKTDATEN UND URLS:
Enthält inventar.lesbare_texte Einträge vom Typ 'kontaktdaten' oder 'url'
(häufig bei Behörden-Infografiken am unteren Rand), gehören diese
wortgetreu und mit korrekten Trennzeichen in die Beschreibung — für
Screenreader-Nutzer sind sie oft der einzige Zugang zu dieser Information.


ATMOSPHAERE (bei Infografiken EINGESCHRÄNKT)

Bei reinen Daten-Visualisierungen bleibt atmosphaere_belege leer — Daten
haben keine Stimmung. Nur wo eine bewusste Designwahl erkennbar Stimmung
transportiert (z.B. eine Kampagnen-Infografik), gilt die folgende Regel:

{ATMOSPHAERE_REGEL}


{_render_ausgabe_schema_block('beginnt mit Infografik — + Hauptthema und Kernaussage', 'nur bei Kampagnen-Design mit sichtbarem Beleg, sonst leer')}


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Beginnt der Alt-Text mit 'Infografik —' + Hauptthema und Kernaussage
   mit Datenpunkten?
2. Alle Stationen in logischer Reihenfolge, mit ihren Beziehungen?
3. Alle Zahlen, Prozente und Mengenangaben übernommen?
4. Inhaltlich statt Layout formuliert (kein 'oben links steht ...')?
5. OCR-Beschriftungen wortgetreu; Kontaktdaten und URLs enthalten?
6. Atmosphäre nur bei bewusster Designwahl, mit Beleg und gesetztem
   atmosphaere_belege?
7. nicht_im_inventar leer?

Wenn ein Punkt nicht erfüllt ist: Output neu formulieren.
"""


def build_beschreibung_prompt_screenshot(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder für screenshot (Paket 4, 16.07.2026).

    Bildschirmfoto einer Anwendung, Webseite oder UI. Erhalten (entdrillt):
    Präfix-Regel 'Screenshot der/des …' (Paket-1-Entscheidung: die Eröffnung
    ist EXPLIZIT erwünscht, konsistent mit den Präfixen der übrigen
    Daten-Familie; das alte v3.7-Präfix-Verbot wurde bewusst nicht
    übernommen), Anwendungs-Identifikation nur mit Beleg (URL/Logo/Titel,
    sonst generischer Typ), funktionale UI-Hierarchie, wortgetreue
    UI-Texte, KONTAKTDATEN_PFLICHT (geteiltes Constraint-Modul),
    Dark-/Light-Mode-Hinweis, keine emotionalen Wertungen.
    Richtwerte 350/1000 (Paket 1) unverändert.
    """
    examples = load_examples('screenshot')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: screenshot (Bildschirmfoto einer Anwendung, Webseite oder UI)
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung für
einen Screenshot. Screenshots werden funktional beschrieben: Welche Anwendung,
welcher Zustand, welche Bedienelemente — so, dass ein blinder Nutzer versteht,
was auf dem Bildschirm passiert und wo er wäre. Lesbare UI-Texte sind dabei
die verlässlichste Informationsquelle und werden wortgetreu übernommen.


{_render_inventar_block(inventar_json)}


{_render_kontext_block(enriched_context, user_hint_text)}


{_render_zweck_block()}


{_render_kompaktheit_block('unter 350 Zeichen', 'etwa 1000 Zeichen')}


ALT-TEXT

Beginne mit 'Screenshot der/des …'. Der erste Satz:
- nennt die Anwendung (wenn aus URL-Leiste, Titel oder Logo identifizierbar)
  ODER den generischen Anwendungstyp ('Browser-Fenster', 'Texteditor',
  'E-Mail-Programm')
- nennt Zustand oder aktuelle Aktion (was ist gerade sichtbar?)

Beispiel RICHTIG: 'Screenshot der InkluDocs-Web-Oberfläche, Projekt-
Übersicht mit drei laufenden Bilduploads und einem fertig analysierten
PDF mit 12 Bildern.'

Beispiel FALSCH: 'Ein Screenshot zeigt eine Anwendung mit verschiedenen
Elementen.'


ANWENDUNGS-IDENTIFIKATION NUR MIT BELEG

Gleiche Zwei-Wege-Logik wie bei Marken: eindeutig belegt -> benennen,
unklar -> generisch bleiben.
- Wenn weder URL noch Logo noch Titel die Anwendung benennen, schreibe
  nicht 'Screenshot von Microsoft Word' — sondern 'Screenshot eines
  Texteditors' oder generischer
- Bei unklarer Domain in der URL: nur die sichtbare Domain nennen,
  nicht raten, was sich dahinter verbirgt


LANGBESCHREIBUNG

Reihenfolge und Umfang:
1. Sichtbare UI-Elemente in funktionaler Hierarchie:
   - Hauptmenü / Navigation
   - Hauptbereich / Inhalt
   - Sekundär-Bereiche / Sidebars
   - Statusleiste / Footer
2. Lesbare Texte wortgetreu übernehmen — vor allem:
   - URL in der Adressleiste (vollständig)
   - Fenstertitel
   - Buttons / Links, die der Nutzer sehen würde
   - Statusmeldungen
   - Eingaben in Formularfeldern

DARK MODE / LIGHT MODE:
Wenn relevant für die Beschreibung (z.B. bei UI-Tutorials), benennen.
Sonst weglassen — meist irrelevant für die Funktion.


{KONTAKTDATEN_PFLICHT}


{_render_atmosphaere_verzicht_block('UI-Beschreibungen sind funktional — keine emotionalen Wertungen.')}


{_render_ausgabe_schema_block("beginnt mit 'Screenshot der/des …' + Anwendung und Zustand", 'bleibt leer')}


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Beginnt der Alt-Text mit 'Screenshot der/des …' + Anwendung bzw.
   generischem Typ + Zustand?
2. Anwendung nur benannt, wenn URL, Logo oder Titel sie belegen?
3. Lesbare UI-Texte wortgetreu übernommen (URL, Fenstertitel, Buttons,
   Statusmeldungen)?
4. UI-Elemente in funktionaler Hierarchie beschrieben?
5. Funktional beschrieben — keine emotionalen Wertungen?
6. nicht_im_inventar leer?

Wenn ein Punkt nicht erfüllt ist: Output neu formulieren.
"""


def build_beschreibung_prompt_strukturformel(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """Premium-Builder für strukturformel (Paket 4, 16.07.2026).

    Chemische Struktur-, Reaktions- oder Summenformel. Erhalten (entdrillt):
    Präfix 'Strukturformel —' / 'Reaktionsgleichung —' mit beiden
    RICHTIG-Beispielen, die fachlichen Beschreibungs-Listen für Strukturen
    und Reaktionen, die Screenreader-Notation (CH3 statt CH₃, Ladungen
    explizit, Pfeile als 'reagiert zu'), die Chemie-Genauigkeitsregeln
    (keine erfundenen Atome, unleserliche Bindungen ehrlich, Stoffnamen nur
    aus Kontext/Beschriftung außer bei einfachsten Molekülen).
    Richtwerte 250/800 (Paket 1) unverändert.
    """
    examples = load_examples('strukturformel')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: strukturformel (Chemische Struktur-, Reaktions- oder Summenformel)
BILDGROESSE: {width}x{height} Pixel

ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung für
eine chemische Formel-Darstellung. Ziel ist fachliche Verlässlichkeit in
screenreader-tauglicher Notation: Ein blinder Chemie-Lernender muss aus dem
Text das Molekül oder die Reaktion korrekt rekonstruieren können. Was das Bild
belegt, wird präzise benannt; Stoff-Identifikationen kommen aus Kontext oder
Beschriftung, nicht aus visueller Vermutung.


{_render_inventar_block(inventar_json)}


{_render_kontext_block(enriched_context, user_hint_text)}


{_render_zweck_block()}


{_render_kompaktheit_block('unter 250 Zeichen', 'etwa 800 Zeichen')}


ALT-TEXT

Der erste Satz:
- beginnt mit dem Präfix 'Strukturformel —' ODER 'Reaktionsgleichung —'
- nennt den Stoffnamen, falls aus Kontext oder Beschriftung erkennbar
- nennt die Summenformel, wenn klar lesbar

Beispiel RICHTIG: 'Strukturformel — Methanol (CH3OH): Methylgruppe
mit Hydroxylgruppe.'

Beispiel RICHTIG für Reaktion: 'Reaktionsgleichung — Veresterung von
Essigsäure mit Methanol zu Methylacetat und Wasser, katalysiert
durch Schwefelsäure.'


LANGBESCHREIBUNG

Bei Strukturformeln:
1. Grundgerüst beschreiben (Kette, Ring, verzweigt)
2. Atome und Atomgruppen (CH3, OH, COOH, NH2, Aromaten etc.)
3. Bindungstypen (Einfach-, Doppel-, Dreifach-Bindung) wenn sichtbar
4. Funktionelle Gruppen explizit benennen
5. Stereochemie wenn dargestellt (cis/trans, R/S)

Bei Reaktionsgleichungen:
1. Edukte (links vom Reaktionspfeil)
2. Reaktionsbedingungen (über/unter dem Pfeil — Katalysator,
   Temperatur, Druck, Lösungsmittel)
3. Produkte (rechts vom Reaktionspfeil)
4. Reaktionstyp wenn aus Kontext bekannt (Substitution, Addition,
   Eliminierung, Redox etc.)

Fließtext, keine Markdown-Formatierung.


NOTATION (screenreader-tauglich)

- Indizes als normale Zahlen ('CH3' — nicht 'CH₃', weil Screenreader
  Indizes oft schlecht vorlesen)
- Ladungen explizit ('Natrium-Kation' oder 'Na+')
- Reaktionspfeile beschreiben als 'reagiert zu' oder 'ergibt'


CHEMISCHE GENAUIGKEIT

- Erfinde keine Atome oder Gruppen, die nicht im Bild sind
- Bei unleserlichen Bindungen: 'Bindungstyp nicht eindeutig erkennbar' —
  statt zu raten
- Stoffnamen nur aus Kontext oder Bildbeschriftung — Strukturen visuell
  zu identifizieren ist fehleranfällig (außer bei einfachsten Molekülen
  wie H2O, CO2)


{_render_atmosphaere_verzicht_block('Chemie ist objektiv.')}


{_render_ausgabe_schema_block("beginnt mit 'Strukturformel —' oder 'Reaktionsgleichung —'", 'bleibt leer')}


FEW-SHOT BEISPIELE

{examples.format_for_prompt()}


FINAL CHECK

1. Beginnt der Alt-Text mit 'Strukturformel —' oder 'Reaktionsgleichung —'
   + Stoffname (falls belegt) und Summenformel (falls lesbar)?
2. Notation screenreader-tauglich (CH3 statt CH₃, Ladungen explizit,
   Pfeile als 'reagiert zu')?
3. Keine Atome oder Gruppen erfunden; Unleserliches ehrlich benannt?
4. Stoffname nur aus Kontext oder Beschriftung (außer einfachste Moleküle)?
5. Grundgerüst und funktionelle Gruppen in der Langbeschreibung?
6. nicht_im_inventar leer?

Wenn ein Punkt nicht erfüllt ist: Output neu formulieren.
"""
