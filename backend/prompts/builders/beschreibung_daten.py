"""Builder der Datenfamilie: illustration, diagramm, tabelle, karte, infografik,
screenshot, strukturformel.

Fassung September 2026 (Prompt-Runde nach dem Prüfkorpus). Aufbau jedes Builders,
siehe docs/PROMPT-STANDARD.md:
  Kopf (Rolle + Belegregeln, im Combo-Aufruf einmal ganz oben)
  BILDTYP, AUFTRAG, inneres Inventar, ALT-TEXT, LANGBESCHREIBUNG,
  höchstens zwei besondere Regeln der Kategorie, STILREGELN, BEISPIELE, KONTEXT.
Die Langbeschreibung ist bei Datengrafiken Pflicht (Stilregel 5 der sachlichen
Fassung). Was für alle Bildtypen gilt, steht nur im Kopf.

Gattungswort: Tabelle, Karte, Infografik, Screenshot, Strukturformel und
Diagramm beginnen den Alt-Text mit ihrer Gattung als erstem Wort, ohne
Gedankenstrich. Illustration beginnt wie ein Foto mit dem Motiv.
"""
from __future__ import annotations

from typing import Optional

from prompts.components.constraints import (
    ANTI_HALLUZINATION_REGELN,
    ATMOSPHAERE_REGEL,
    EIGENNAMEN_REGELN,
    KUNSTWERK_REGEL,
)
from prompts.components.roles import ROLE_BESCHREIBER
from prompts.components.schemas import InventarOutput
from prompts.components.stilregeln import STILREGELN, STILREGELN_SACHLICH

from .helpers import bildgroesse_zeile, inventar_block, kontext_werte, kopf_schichten, load_examples, user_hint_block


def _basis_schichten() -> str:
    """Rolle und Belegregeln im Prompt-Kopf; im Combo-Aufruf leer (stehen dort einmal oben)."""
    return kopf_schichten(f'{ROLE_BESCHREIBER}\n\n{ANTI_HALLUZINATION_REGELN}')


_INVENTAR_EINLEITUNG = """Das Inventar enthält die Beobachtungen des Analyse-Schritts. Es ist die
Grundlage jeder Aussage; sichtbare Bildinformationen dürfen ergänzt werden,
aber nichts darf dem Inventar widersprechen."""


def _render_inventar_block(inventar_json: str) -> str:
    return inventar_block(inventar_json, _INVENTAR_EINLEITUNG)


def _render_kontext_block(enriched_context: str, user_hint_text: str) -> str:
    """Die Werte zum Kontext. Die Regeln dazu stehen in Belegregel 6."""
    return f"""KONTEXT (Bildunterschrift, umliegender Text, Angaben des Aufrufers)
{kontext_werte(enriched_context, user_hint_text)}"""


_LESBARER_TEXT = """LESBARER TEXT

Lesbare Beschriftungen, Zahlen, Namen und Kontaktdaten übernimmst du wortgetreu
mit ihren Trennzeichen und in ihrer Originalsprache. Prüfe die Zuordnung zur
richtigen Zeile, Spalte, Fläche oder Legende. Erläuternde Absätze fasst du
sinngemäß zusammen. Fehlende oder unleserliche Teile ergänzt du nicht; ein leeres
Feld oder ein Strich ist keine Null."""


def build_beschreibung_prompt_illustration(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """illustration: Cartoon, Vektorgrafik, gemalte Illustration, Buchbild.

    Historie: Die alte Fassung verlangte, alle Inventar-Elemente aufzuzählen,
    und verbot Stimmungsaussagen. Beides ist aufgegeben: Der Alt-Text nennt die
    dargestellte Idee und die dafür unverzichtbaren Elemente, Stimmung folgt
    Belegregel 4 wie bei Fotos. Die Regeln zu Beispieltexten in Sprechblasen
    und zu Siegeln stammen aus einem Kundenbefund (Produktillustration mit
    Muster-Alt-Text und Siegel), die Kunstwerk-Regel aus dem Distelfink-Fall.
    """
    examples = load_examples('illustration')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: illustration (Cartoon, Vektorgrafik, gemalte Illustration, Buchbild)
{bildgroesse_zeile(width, height)}

AUFTRAG

Eine Illustration steht im Dokument, weil sie eine Idee, einen Begriff oder eine
Aussage bildlich fasst. Dein Text nennt zuerst diese Idee, wenn Bild oder Kontext
sie belegen (Symbolbild für Homeoffice, Produktillustration zur Erstellung von
Alt-Texten mit KI), und dann die Elemente, die sie tragen. Stilisierte Motive sind
die häufigste Quelle für Fehldeutungen: Ein vereinfachtes Tier wird schnell zu
einer bestimmten Art, nebeneinander stehende Figuren und Gegenstände werden zu
einer Handlung. Prüfe im inneren Inventar alle Elemente, bevor du auswählst;
genannt wird nur, was die Aussage trägt. Stimmung und Wirkung darfst du wie bei
Fotos benennen, mit dem sichtbaren Beleg im selben Satz.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginnt mit dem Motiv, ohne Gattungswort vorweg: "Symbolbild für Homeoffice: Eine
Frau am Küchentisch mit Laptop, daneben ein Kind mit Malbuch." Die Stilrichtung
(Cartoon, Vektor, Aquarell, Comic) nennst du, wenn sie zur Aussage gehört oder das
Motiv sonst als Foto verstanden würde. Ein Element ist unverzichtbar, wenn sein
Weglassen Aussage, Funktion oder einen wesentlichen Unterschied der Illustration
verändert; kleine dekorative Einzelheiten bleiben weg.

Mehrdeutige Figuren beschreibst du nach Belegregel 2: die sichtbare Form oder
zwei gleichwertige Deutungen, keine Festlegung auf das naheliegendste Klischee.
Handlungen nur, wenn das Bild sie zeigt: Ein Hundekopf neben Laptop, Tablet und
Mikroskop ist keine arbeitende Figur.


LANGBESCHREIBUNG

Pflicht bei mehr als drei bedeutungstragenden Elementen, sonst darf sie leer
bleiben. Sie ergänzt weitere bedeutungstragende Elemente und ihre Anordnung:
zentrale Figuren oder Gegenstände mit ihren sichtbaren Merkmalen, Nebenelemente,
lesbare Beschriftungen, Farbklima.


BEISPIELTEXTE UND SIEGEL

Ein Text in einer Sprechblase, einem Platzhalter oder einer Attrappe ist ein
Beispieltext. Du benennst ihn als solchen ("eine Sprechblase mit einem
Beispiel-Alt-Text") und zitierst ihn nicht als Inhalt oder Datenangabe des
Bildes. Ein Siegel oder Abzeichen ist ein sichtbares Element mit einer Aufschrift
("rundes Siegel mit der Aufschrift Barrierefrei"); es belegt keine Prüfung und
keine Zertifizierung.


{KUNSTWERK_REGEL}


{STILREGELN}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_diagramm(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """diagramm: Balken, Linie, Kreis, gestapelt, Streu, Heatmap.

    Läuft der Werte-Schritt (Orchestrator), hängt er einen Block ABGELESENE WERTE
    mit rechnerischen Kernaussagen an; der Prompt verweist darauf.
    """
    examples = load_examples('diagramm')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: diagramm (Balken, Linie, Kreis, gestapelt, Streu, Heatmap)
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Diagramm steht im Dokument, weil es eine Aussage über Zahlen macht. Dein Text
vermittelt diese Aussage: Trend, Vergleich, Rangfolge, Anteil oder Wendepunkt, mit
den Werten, die sie tragen. Zahlen zuerst, Deutung danach: Lies die Werte an der
Achse ab und notiere sie dir als Liste, bevor du ein Trendwort schreibst. Liegt am
Ende ein Block ABGELESENE WERTE vor, gelten dessen Zahlen und rechnerische
Kernaussagen vor deinem Eindruck. Ohne lesbare Skala nennst du keine Zahl und
keinen Betrag, sondern Rangfolge und Form. Keine Ursachen, keine Prognosen, keine
Bewertung, die das Diagramm nicht enthält.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Diagrammtyp, Thema (Titel oder Kontext) und Zeitraum, dann die Gesamtaussage,
dann jede Reihe oder Kategorie mit ihrer Richtung und dem Wert, der sie trägt:
"Balkendiagramm zur Umsatzentwicklung 2021 bis 2023 in vier Sparten: Mobile
liegt am Ende mit 5,0 vorn, nach einem Einbruch 2022. Software fällt durchgehend
von 4,3 auf 2,0, Hardware steigt 2022 auf 4,4 und fällt dann auf 2,0, Services
sinkt auf 1,8 und erholt sich auf 3,0." Keine Reihe fehlt; bei mehr als etwa
sechs Reihen nennst du Spanne und Ausreißer statt jeder Reihe. Nicht jeden
Zwischenwert, keine Achsenbeschreibung; eine Farbe nur, wenn sie eine Reihe ohne
Legende kenntlich macht.

Trendwörter tragen eine Bedingung: "durchgehend" oder "kontinuierlich" nur, wenn
kein Zwischenschritt widerspricht; "erholt sich" beschreibt einen Anstieg nach
einem Rückgang; "wieder auf dem Ausgangsniveau" nur, wenn Anfangs- und Endwert
gleich sind; bei "höchster" und "zweithöchster" nennst du die Bezugsmenge
(Kategorie, Jahr oder ganzes Diagramm). Prozent und Prozentpunkte werden nicht
vertauscht. Eine Summe oder Differenz darfst du nennen, wenn sie sich aus den
abgelesenen Werten rechnerisch ergibt.


LANGBESCHREIBUNG

Pflicht. Fließtext in dieser Reihenfolge: Diagrammtyp und Thema; Achsen,
Einheiten, Zeitraum und Legende; die Kernaussage mit Werten; je Reihe oder
Kategorie der Verlauf mit Anfangs-, End-, Höchst- und Tiefstwert; Extremwerte und
Wendepunkte des ganzen Diagramms; lesbare Zusatzangaben wie Quelle oder Fußnote.
Beziehungen zwischen Werten erklären, keine unverbundene Zahlenliste. Alle Zahlen
im Alt-Text und in der Langbeschreibung stimmen überein.


{_LESBARER_TEXT}


{STILREGELN_SACHLICH}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_tabelle(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """tabelle: tabellarische Daten als Grafik.

    Historie: Die alte Fassung war auf Bilanzen zugeschnitten (Bilanzsumme,
    Anlage- und Umlaufvermögen, "letzte Zeile ist die Bilanzsumme") und
    erlaubte die vollständige Übertragung nur bis vier mal vier Zellen. Jetzt
    gilt: Summen nach Beschriftung zuordnen, überschaubare Tabellen vollständig
    übertragen, große zusammenfassen. Der Satz zu Zeitraum-Spalten stammt aus
    einem Kundenfall mit Anfangs- und Endbeständen.
    """
    examples = load_examples('tabelle')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: tabelle (tabellarische Daten als Grafik)
{bildgroesse_zeile(width, height)}

AUFTRAG

Eine Tabelle steht im Dokument, weil sie Werte zu einem Thema geordnet
nebeneinanderstellt. Dein Text nennt zuerst Thema, Bezugsgröße (je 100 Gramm, in
Euro, Stand zum Jahresende) und die wichtigste Aussage: eine Gesamtsumme, wenn
sie vorhanden und zentral ist, sonst Rangfolge, Spanne, Ausreißer oder Vergleich.
Danach macht die Langbeschreibung die Struktur mit richtiger Zeilen- und
Spaltenzuordnung nachvollziehbar. Genauigkeit bei Zahlen, Summen und Einheiten
ist hier der Maßstab. Lies zuerst alle Spaltenköpfe von links nach rechts, dann
jede Zeile, und ordne jeden Wert seiner Spalte zu, bevor du formulierst.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginnt mit dem Gattungswort und dem Thema, dann Bezugsgröße und Kernaussage
mit ihren Werten: "Tabelle der Nährwerte je 100 Gramm: 52 Kilokalorien, davon
12 Gramm Kohlenhydrate, 0,3 Gramm Eiweiß und kein Fett." Eine überschaubare
Tabelle (bis etwa fünf Zeilen) trägt der Alt-Text mit allen Werten; eine größere
mit Gesamtsumme, Spanne, Höchst- und Tiefstwert und den Werten, die der
Dokumentzweck braucht. Keine Beschreibung von Rahmen und Zellfarben; eine Farbe
nennst du, wenn sie eine Bedeutung trägt (rot markierte Zeile).

Eine Summe ordnest du nach ihrer Beschriftung und ihrem Abschnitt zu, nicht nach
ihrer Position: Die letzte Zeile ist nicht deshalb die Gesamtsumme, weil sie
unten steht; Zwischensummen von Abschnitten und die Gesamtsumme unterscheidest
du am sichtbaren Zeilentext. Fehlt eine eindeutige Beschriftung, nennst du den
Zeilentext, ohne ihn umzudeuten. Stehen in einer Zeile Werte für zwei Zeitpunkte
(Spalten 01.01. und 31.12., Vorjahr und Berichtsjahr), sind das verschiedene
Werte; nenne beide mit Spaltenzuordnung und verwechsle Bewegungen dazwischen
(Zugänge, Abgänge, Veränderung) nicht mit Beständen.


LANGBESCHREIBUNG

Pflicht. Fließtext in dieser Reihenfolge: Thema und Bezugsgröße mit Einheit; die
Spaltenköpfe wortgetreu; die Kernaussage; dann die Werte. Eine überschaubare
Tabelle (bis etwa acht Zeilen und fünf Spalten) überträgst du vollständig, Zeile
für Zeile mit Zeilenbezeichnung und Spaltenzuordnung. Größere Tabellen fasst du
zusammen: Aufbau, Spannweite, Höchst- und Tiefstwerte, auffällige Muster und die
Werte, die der Dokumentzweck braucht. Einheiten (Prozent, Euro, Mio., Tsd.)
übernimmst du wie gedruckt. Fußnoten, Quelle und Stand gehören ans Ende. Alle
Zahlen in Alt-Text und Langbeschreibung stimmen überein.


{_LESBARER_TEXT}


{STILREGELN_SACHLICH}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_karte(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """karte: Landkarte, Stadtplan, Lageplan, Übersichtskarte, thematische Karte.

    Historie: Die alte Fassung kannte nur Standortkarten ("markierte Standorte
    vollständig auflisten") und trug eine Liste von Ortsnamen-Verwechslungen
    aus der Zeit eines anderen Bildmodells. Jetzt sind Standort-, politische,
    historische und thematische Karten unterschieden; die Ortsnamen-Regel
    steht in EIGENNAMEN_REGELN.
    """
    examples = load_examples('karte')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: karte (Landkarte, Stadtplan, Lageplan, Übersichtskarte, thematische Karte)
{bildgroesse_zeile(width, height)}

AUFTRAG

Eine Karte steht im Dokument, weil sie etwas räumlich verortet: Standorte,
Gebiete, Grenzen, Wege oder Werte je Region. Dein Text gibt räumliche
Orientierung: zuerst Kartenthema, gezeigtes Gebiet und die Bedeutung der
Hervorhebungen, dann die Verteilung so, dass ein Mensch ohne Bild sie
nachvollziehen kann. Farben, Symbole und Größen bedeuten, was die Legende sagt,
nicht, was sie im Alltag bedeuten: Rot ist keine Gefahr, ein großer Kreis steht
für den Wert, den die Legende ihm zuweist.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginnt mit dem Gattungswort, Thema und Gebiet, dann die räumliche Kernaussage:
"Karte der Beratungsstellen in Nordrhein-Westfalen: 14 Standorte, die meisten im
Ruhrgebiet und entlang des Rheins, keiner im Sauerland." Bei politischen oder
historischen Karten trägt der Alt-Text den gezeigten Zeitstand und die wichtigste
Grenze oder Gebietsaufteilung. Eine Jahreszahl nennst du nur, wenn sie im Bild
steht oder der Kontext sie belegt.


LANGBESCHREIBUNG

Pflicht. Fließtext in dieser Reihenfolge: Kartenthema, Gebiet und Ausrichtung;
die Legende mit ihren Symbolen, Farben und Größenstufen; dann die Inhalte nach
Kartentyp. Bei Standortkarten die markierten Orte mit ihrer Kategorie aus der
Legende, nach Lage geordnet und mit Himmelsrichtungen, wenn Norden oben liegt;
Hintergrundorte nur zur Einordnung ("zwischen München und Stuttgart"), nicht als
vollständige Liste. Bei politischen oder historischen Karten die Grenzen,
Gebiete, Zugehörigkeiten und der gezeigte Zeitstand, so wie die Karte ihn
darstellt; heutige Grenzen und Namen ersetzen ihn nicht. Bei thematischen Karten
die Werteklassen je Region mit den Extremen. Maßstab, Quelle und Stand, wenn
lesbar. Keine Orte und keine Wege, die die Karte nicht zeigt; Unlesbares nennst
du unlesbar.


{EIGENNAMEN_REGELN}


{_LESBARER_TEXT}


{STILREGELN_SACHLICH}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_infografik(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """infografik: Schaubild, Ablauf, Übersichtsgrafik mit Stationen oder Kennzahlen.

    Historie: Die alte Fassung verlangte, alle Stationen und alle Zahlen zu
    übernehmen, was bei dichten Grafiken an der Feldgrenze scheiterte
    (Kundenfall einer Raumfahrt-Infografik). Jetzt: Umfang planen, geordnete
    Zusammenfassung, keine Vollständigkeitsbehauptung. Die Pfeil-Regel und die
    Siegel-Regel stammen aus der Prompt-Prüfung; das Beispiel mit "fast die
    Hälfte (39 Prozent)" war selbst ungenau und ist Gegenbeispiel geworden.
    """
    examples = load_examples('infografik')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: infografik (Schaubild, Ablauf, Übersichtsgrafik mit Stationen, Schritten oder Kennzahlen)
{bildgroesse_zeile(width, height)}

AUFTRAG

Eine Infografik übersetzt einen Inhalt in eine visuelle Anordnung. Dein Text
übersetzt zurück: die inhaltliche Logik aus Stationen, Verbindungen und Zahlen,
nicht das Layout. Der Alt-Text trägt Thema, Kernaussage und die Stationen oder
Kennzahlen, die sie tragen; die Langbeschreibung die geordnete Zusammenfassung.
Plane den Umfang, bevor du
formulierst: Zähle im inneren Inventar Stationen und Zahlen und entscheide, was
in 2000 Zeichen Platz hat. Was du auslässt, verdeckst du nicht durch eine
Vollständigkeitsbehauptung; eine gesonderte vollständige Alternative erwähnst du
nur, wenn es sie wirklich gibt.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginnt mit dem Gattungswort und dem Thema, dann die Kernaussage mit ihren
Zahlen und die Stationen in ihrer Reihenfolge: "Infografik zum Ablauf der
Antragstellung: vier Schritte von der Registrierung über Prüfung und Bescheid
bis zur Auszahlung, Bearbeitungszeit sechs Wochen." Bis etwa sechs Stationen
oder Kennzahlen nennst du alle mit ihrer Kernzahl, bei mehr eine geordnete
Auswahl der tragenden. Zahlen exakt wie gedruckt: 39 Prozent, nicht "fast die
Hälfte". Keine Farben, kein Layout.


LANGBESCHREIBUNG

Pflicht. Fließtext in der Ordnung, die die Grafik selbst vorgibt: chronologisch
bei Abläufen, hierarchisch bei Gliederungen, nach Größe bei Kennzahlen. Je
Station ihre Bezeichnung, ihre Zahlen und die Verbindung zur nächsten ("Schritt 1
ist die Registrierung, daraus folgt Schritt 2 mit der Prüfung"; "Hauptkategorie
A umfasst B, C und D"). Bei dichten Grafiken eine geordnete Zusammenfassung mit
den Zahlen, die die Aussage tragen. Lesbare Zusatzangaben wie Quelle, Stand,
Internetadresse und Kontaktdaten am Ende; für Menschen mit Screenreader sind
sie oft der einzige Zugang. Kein Layout-Bericht ("oben links steht", "in der
Mitte befindet sich"); eine Position nennst du nur, wenn sie inhaltlich
bedeutet, dass etwas im Mittelpunkt steht.


PFEILE, SIEGEL UND WERBEAUSSAGEN

Ein Pfeil kann Reihenfolge, Verweis, Bewegung oder Ursache bedeuten. Du nennst
nur die Bedeutung, die Beschriftung und Darstellung tragen; aus räumlicher
Nachbarschaft folgt keine Ursache. Ein Siegel, ein Häkchen oder eine
Werbeaussage belegt, dass die Grafik diese Aufschrift enthält, nicht, dass eine
Prüfung stattgefunden hat: "Siegel mit der Aufschrift Klimaneutral".


{ATMOSPHAERE_REGEL}


{_LESBARER_TEXT}


{STILREGELN_SACHLICH}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_screenshot(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """screenshot: Bildschirmfoto einer Anwendung, Website oder Bedienoberfläche.

    Historie: Die alte Fassung beschrieb rein funktional (Anwendung, Zustand,
    Hierarchie aller Bereiche) ohne Bezug zum Dokumentzweck und nutzte die
    eigene Anwendung als Beispiel. Jetzt: Zustand und wichtigste Aktion in den
    Alt-Text, Bereiche nur soweit nötig, Zweckbezug (Anleitungsschritt,
    Fehlerbild), erfundene Anwendung Musterwerk im Beispiel. Die Regel
    "Abbildung statt Bedienelement" stammt aus einem Prüffall, in dem ein
    Screenshot einer Seitennavigation auf einen Pfeil verkürzt wurde.
    """
    examples = load_examples('screenshot')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: screenshot (Bildschirmfoto einer Anwendung, Website oder Bedienoberfläche)
{bildgroesse_zeile(width, height)}

AUFTRAG

Ein Screenshot steht im Dokument, weil er einen bestimmten Zustand einer
Anwendung belegt: einen Schritt einer Anleitung, ein Fehlerbild, ein Ergebnis,
einen Vorher-Nachher-Vergleich. Dein Text nennt Anwendung oder Website, die
Ansicht, den gezeigten Zustand und die für diesen Zustand wichtigste sichtbare
Aktion, und er sagt, was der Screenshot an dieser Stelle des Dokuments zeigt.
Die Anwendung benennst du, wenn Adressleiste, Fenstertitel, Logo oder Kontext sie
belegen; sonst den Typ ("Browserfenster", "Texteditor", "E-Mail-Programm"). Bei
einer Adresse nennst du die sichtbare Domain, ohne zu deuten, was dahinter steht.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginnt mit dem Gattungswort, Anwendung und Ansicht, dann Zustand und Aktion:
"Screenshot der Anwendung Musterwerk, Ansicht Projektliste: 26 Bilder, 0
verarbeitet, Schaltfläche Alt-Texte generieren." Steht der Screenshot in einer
Anleitung, trägt der Alt-Text den Schritt ("Schritt 3: Dialog Exportieren mit
aktiviertem Kontrollkästchen PDF/UA"); zeigt er einen Fehler, die Fehlermeldung
wortgetreu. Nicht die ganze Kopfzeile, nicht jedes Menü.


LANGBESCHREIBUNG

Pflicht. Fließtext mit den Bereichen, die zum Verständnis des Zustands nötig
sind, in funktionaler Reihenfolge: zuerst der Bereich, in dem die Aktion
stattfindet, dann Navigation, Seitenleisten und Statusleiste, soweit sie den
Zustand erklären. Statusmeldungen, Werte, Eingaben in Feldern, Schaltflächen und
Beschriftungen wortgetreu; eine Adresse in der Adressleiste vollständig. Eine
Abschrift aller Menüs und Randbereiche nur, wenn der Dokumentzweck gerade deren
Inhalt betrifft. Hell- oder Dunkeldarstellung nur, wenn sie für das Dokument
eine Rolle spielt.


ABBILDUNG STATT BEDIENELEMENT

Ein Screenshot ist ein Bild einer Oberfläche, kein bedienbares Element. Ein
Screenshot einer Seitennavigation beschreibt die sichtbaren Seiten und den
aktiven Zustand; er wird nicht auf die Funktion eines einzelnen Pfeils oder
Knopfs verkürzt.


{_LESBARER_TEXT}


{STILREGELN_SACHLICH}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""


def build_beschreibung_prompt_strukturformel(
    inventar: InventarOutput,
    enriched_context: str,
    width: int, height: int,
    user_hint: Optional[str] = None,
) -> str:
    """strukturformel: chemische Struktur-, Reaktions- oder Summenformel.

    Historie: Die alte Fassung erlaubte Stoffnamen nur aus Kontext oder
    Beschriftung (Vorsicht aus der Zeit eines schwächeren Bildmodells) und
    nannte zugleich im Beispiel eine Summenformel, die nirgends belegt war.
    Jetzt: eindeutig erkennbare Strukturen aus Fachwissen benennen, unsichere
    beim Gerüst beschreiben, Summenformel nur mit Beleg, Stoffklasse und
    Reaktionstyp als Einordnung.
    """
    examples = load_examples('strukturformel')
    inventar_json = inventar.model_dump_json(indent=2)
    user_hint_text = user_hint_block(user_hint)

    return f"""{_basis_schichten()}

BILDTYP: strukturformel (chemische Struktur-, Reaktions- oder Summenformel)
{bildgroesse_zeile(width, height)}

AUFTRAG

Eine Formeldarstellung steht im Dokument, weil sie den Aufbau eines Stoffs oder
den Verlauf einer Reaktion zeigt. Dein Text muss so verlässlich sein, dass ein
Mensch, der Chemie lernt, das Molekül oder die Reaktion daraus richtig aufbauen
kann. Den Stoffnamen nimmst du aus Beschriftung oder Kontext; fehlt beides,
benennst du eindeutig erkennbare Strukturen aus deinem Fachwissen
(Acetylsalicylsäure, Koffein und Glucose sind an ihrem Gerüst erkennbar) und
beschreibst unsichere Strukturen beim Gerüst. Eine Summenformel nennst du nur,
wenn sie im Bild oder im Kontext steht; du erzeugst sie nicht als Wissensangabe.
Stoffklasse (aromatische Carbonsäure, Ester, Alkaloid) und Reaktionstyp
(Veresterung, Substitution, Addition, Redoxreaktion) ordnest du ein, wenn
Struktur oder Kontext sie belegen.


{_render_inventar_block(inventar_json)}


ALT-TEXT

Beginnt mit dem Gattungswort und dem Stoff oder der Reaktion, dann die
wesentlichen Bausteine: "Strukturformel von Acetylsalicylsäure: Benzolring mit
Carboxygruppe und benachbarter Acetoxygruppe." Bei Reaktionen: "Reaktionsgleichung
der Veresterung von Essigsäure mit Ethanol zu Essigsäureethylester und Wasser,
Schwefelsäure als Katalysator." Ohne belegten Stoffnamen beginnt der Alt-Text mit
dem Gerüst: "Strukturformel eines Sechsrings mit zwei Hydroxygruppen".


LANGBESCHREIBUNG

Pflicht. Bei Strukturformeln in dieser Reihenfolge: Grundgerüst (Kette, Ring,
verzweigt, Ringsystem); Atome und Atomgruppen mit ihrer Position; Bindungstypen
(Einfach-, Doppel-, Dreifachbindung); funktionelle Gruppen mit Namen; Ladungen
ausgesprochen ("Natrium-Kation" oder "Na plus"); Stereochemie nur, wenn Keil-
und Strichbindungen oder eine Angabe wie cis, trans, R oder S sie darstellen.
Die Anordnung auf dem Papier allein belegt keine Stereochemie. Bei
Reaktionsgleichungen: Edukte links vom Pfeil, Bedingungen über und unter dem
Pfeil (Katalysator, Temperatur, Druck, Lösungsmittel), Produkte rechts,
stöchiometrische Zahlen, dann der Reaktionstyp. Erfinde keine Atome und keine
Gruppen; eine unleserliche Bindung nennst du unleserlich.


SCHREIBWEISE FÜR SCREENREADER

Keine Hoch- und Tiefstellung: Indizes als normale Zahlen und Gruppen
ausgeschrieben ("CH3-Gruppe", "H2O", nicht "CH₃"). Reaktionspfeile als "reagiert
zu" oder "ergibt", Gleichgewichtspfeile als "steht im Gleichgewicht mit".
Griechische Buchstaben und Positionsangaben ausgeschrieben ("alpha-Position",
"Position 2").


{STILREGELN_SACHLICH}


BEISPIELE

{examples.format_for_prompt()}


{_render_kontext_block(enriched_context, user_hint_text)}
"""
