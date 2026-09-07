# InkluDocs Prompt-Standard

Gilt für alle Prompts der Bildbeschreibung (Klassifikator, Beschreibung, Prüfpass, Zähl- und Werte-Schritt). Stand: September 2026.

## Ziel

Ein Alternativtext ersetzt das Bild für Menschen, die es nicht sehen. Er beschreibt nicht nur, er vermittelt Wissen: Was ist das, was sagt es aus, warum steht es an dieser Stelle. Alles, was im Text steht, ist belegt. Nichts, was belegt und wichtig ist, fehlt.

## Aufbau jedes Beschreibungs-Prompts

Jede Regel lebt an genau einer Stelle. Der Kopf gilt für alle Bildtypen, der Kategorie-Teil ergänzt nur das Besondere.

1. Rolle und Auftrag (`ROLE_BESCHREIBER`), einmal.
2. Belegregeln (`ANTI_HALLUZINATION_REGELN`), einmal: Beleg, zwei Wege statt Vermutung, keine erfundenen Handlungen, Wertungen nur mit Beleg, Fotomontage, Kontext und Wissen, Zählen.
3. Arbeitsweise (nur im Produktionsaufruf): erst inneres Inventar, dann Text.
4. Kategorie-Teil mit festen Abschnitten in dieser Reihenfolge:
   BILDTYP, AUFTRAG, ALT-TEXT, LANGBESCHREIBUNG, besondere Regeln der Kategorie (höchstens zwei Abschnitte), STILREGELN, BEISPIELE, KONTEXT mit den Bilddaten.
5. Kein Abschlusscheck, keine Schema-Nacherzählung, keine Formatanweisung. Das Ausgabeformat erzwingt das Werkzeugschema.

## Sprache

- Deutsch mit echten Umlauten. Deutsche Abschnittstitel in Großbuchstaben. Das Modell wird mit „du" angesprochen, Anweisungen im Imperativ.
- Feste Begriffe: Alt-Text, Langbeschreibung, Bildtyp, Kontext, Beleg. Keine englischen Fachwörter, wo ein deutsches Wort existiert.
- Keine Großschreib-Kaskaden, keine Ausrufezeichen, kein „NIEMALS". Eine Regel steht einmal, ruhig formuliert.
- Keine Daten, Personennamen, Ticketnummern, Kundennamen oder Entwicklerkommentare im Modelltext. Historie gehört in den Docstring der Funktion.
- Beispiele verwenden erfundene Marken und Personen (Musterwerk GmbH, Beispiel AG, Anna Reimers). Keine realen Firmen, außer das Beispiel handelt von einem allgemein bekannten Wahrzeichen oder einer historischen Person.

## Beleg

Belegt ist, was im Bild sichtbar oder lesbar ist, was der Kontext (Bildunterschrift, umliegender Text, Nutzerhinweis) ausdrücklich sagt, oder was ein sehender Mensch mit Allgemeinwissen auf einen Blick erkennt (Eiffelturm, ein MacBook am Gehäuse, eine Person des öffentlichen Lebens). Alles andere bleibt neutral beschrieben. Vermutungswörter gibt es nicht: Entweder benennen oder die sichtbare Form beschreiben.

## Wertungen und Körperdetails

Wertungen über Stimmung, Wirkung oder Charakter sind erlaubt, wenn der sichtbare Beleg im selben Satz steht. Kein Wort ist verboten, kein Wort ist vorgeschrieben. Blickrichtung, Haltung und Gestik gehören in den Alt-Text nur, wenn sie eine Beziehung oder Handlung tragen.

## Arbeitsteilung Alt-Text und Langbeschreibung

Der Alt-Text trägt Thema, Kernaussage und die Einordnung, die der Kontext belegt. Richtwert: einfache Motive unter 150 Zeichen, komplexe Motive bis 250, Datengrafiken bis 300. Die Langbeschreibung ergänzt Struktur, Werte, Reihenfolgen und Nebendetails. Sie ist Pflicht bei Diagramm, Tabelle, Infografik, Karte, Strukturformel und Screenshot und bei Fotos mit mehr als drei bedeutungstragenden Elementen. Sie wiederholt den Alt-Text nicht.

## Beispiele

Je Kategorie ein bis zwei gute Beispiele mit vollständiger Antwort und ein Gegenbeispiel mit genau einem lehrreichen Fehler. Bei komplexen Kategorien zeigt das gute Beispiel Alt-Text und Langbeschreibung. Beispiele halten die Stilregeln selbst ein.

## Pflege

`tests/test_prompt_hygiene.py` rendert alle Prompts und meldet Verstöße gegen diesen Standard: doppelte Abschnitte, ASCII-Umlaute, Daten, Namen, verbotene Altlasten, Längen. Die gerenderten Prompts in `backend/prompts/snapshots/` sind der Maßstab und werden nach jeder Änderung neu erzeugt.
