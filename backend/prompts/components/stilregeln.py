"""Stilregeln für Alt-Text und Langbeschreibung (eine Quelle für Pipeline und InkluAgent).

Fassung September 2026. Der Maßstab ist ein guter Redakteur: kurze, natürliche
Sätze, das Wichtigste zuerst, keine Detail-Abhak-Prosa. Fakten-Regeln stehen in
constraints/halluzination.py, hier steht nur Stil.

Drei Exporte:
- STILREGELN_KERN     — Punkte 1 bis 5 ohne Längen (Prüfpass, InkluAgent).
- STILREGELN          — Kern plus Länge für die Foto-Familie und Illustration.
- STILREGELN_SACHLICH — Punkte 1, 2 und 5 mit Datengrafik-Beispiel plus Länge für
                        Diagramm, Tabelle, Karte, Infografik, Screenshot,
                        Strukturformel. Körperdetails und Namensregel entfallen.
Die Mini-Familie (Logo, Icon, Bedienelement) hat eine eigene, kürzere Formel.
"""

_PUNKT_1 = """1. Wichtigstes zuerst. Führe mit der Information, wegen der das Bild an seiner
   Stelle steht: wer oder was, die Aussage, die belegte Einordnung. Jedes weitere
   Detail muss die Frage bestehen: Hilft es, dieses Bild an dieser Stelle zu
   verstehen? Wenn nicht, gehört es in die Langbeschreibung oder nirgendwohin."""

_PUNKT_2_FOTO = """2. Natürlicher Satzbau. Subjekt und Verb stehen früh und nah beieinander, ein
   bis zwei Sätze. Keine Partizip-Einschübe zwischen Subjekt und Verb, keine
   Semikolon-Ketten, keine Lagefloskeln wie "im Bildvordergrund" (stattdessen
   "vor ihr", "dahinter", "auf dem Tisch").
   Gut: "Anna Reimers in schwarzem Blazer sitzt an einem Holztisch mit
   aufgeklapptem Laptop vor einer hellen Wand."
   Schlecht: "Anna Reimers in schwarzem Blazer, den Kopf leicht nach oben links
   gewandt und den Mund leicht geöffnet, sitzt vor einer hellen Wand; im
   Bildvordergrund ein aufgeklapptes Laptop auf einem Holztisch.\""""

_PUNKT_2_SACHLICH = """2. Natürlicher Satzbau. Subjekt und Verb stehen früh und nah beieinander, ein
   bis zwei Sätze im Alt-Text. Keine Semikolon-Ketten, keine Lagefloskeln wie
   "im Bildvordergrund".
   Gut: "Balkendiagramm zur Umsatzentwicklung 2021 bis 2023: Nur Mobile steigt
   und erreicht 2023 mit 5,0 den höchsten Wert."
   Schlecht: "Ein Balkendiagramm, bestehend aus vier Kategorien mit jeweils drei
   Balken, deren Höhen variieren; im Bildvordergrund die Legende.\""""

_PUNKT_3 = """3. Körperdetails nur mit Bedeutung. Kopfhaltung, Blickrichtung, Mundstellung,
   Gestik und Mimik gehören in den Alt-Text nur, wenn sie eine Beziehung oder
   Handlung tragen (die Rednerin zeigt auf die Leinwand; zwei Personen geben
   sich die Hand; alle blicken zu ihr). In der Langbeschreibung nur dort, wo
   sie die Szene nachvollziehbarer machen."""

_PUNKT_4 = """4. Name als Satzanfang. Ein verwendeter Name ist das Subjekt des ersten Satzes
   ("Anna Reimers, Gründerin der Musterwerk GmbH, sitzt an einem Holztisch").
   Falsch ist die Etikett-Struktur "Name, Funktion: Ein Mann ...". Eine benannte
   Person wird danach nicht erneut anonym eingeführt, sondern mit Pronomen oder
   Rolle weitergeführt."""

_PUNKT_5 = """5. Keine Floskeln. Keine Ansage, dass etwas gezeigt wird: nicht "Das Bild
   zeigt", "Die Aufnahme zeigt", "Zu sehen ist", "Hier sieht man" und keine
   sinngemäße Variante, weder am Anfang noch mitten im Text. Steige direkt mit
   dem Motiv ein: statt "Die Aufnahme zeigt den Dom von Südwesten" schreibe
   "Blick von Südwesten auf den Dom". Keine Quellenhinweise wie "laut Kontext"
   oder "laut Bildunterschrift": Eine belegte Angabe wird direkt ausgesagt."""

_LAENGE_FOTO = """6. Länge und Arbeitsteilung. Richtwert für den Alt-Text: einfache Motive unter
   150 Zeichen, komplexe Szenen bis etwa 250. Die 400 Zeichen des Schemas sind
   eine Obergrenze, kein Ziel. Der Alt-Text trägt die Essenz; Nebendetails,
   räumliche Ausführung und Wissenstiefe gehören in die Langbeschreibung. Die
   Langbeschreibung ist Fließtext ohne Überschriften und Aufzählungszeichen,
   beginnt nicht mit einer Ansage und wiederholt den Alt-Text nicht."""

_LAENGE_SACHLICH = """6. Länge und Arbeitsteilung. Richtwert für den Alt-Text: bis etwa 250 Zeichen,
   bei dichten Grafiken bis 300. Die 400 Zeichen des Schemas sind eine
   Obergrenze, kein Ziel. Der Alt-Text trägt Typ, Thema und Kernaussage; die
   Langbeschreibung ist bei diesem Bildtyp Pflicht und trägt Struktur, Werte,
   Reihenfolgen und lesbare Texte in Fließtext ohne Überschriften, Tabellen und
   Aufzählungszeichen, höchstens 2000 Zeichen. Sie wiederholt den Alt-Text nicht."""

STILREGELN_KERN = "STILREGELN (Stil, nicht Fakten)\n\n" + "\n\n".join([_PUNKT_1, _PUNKT_2_FOTO, _PUNKT_3, _PUNKT_4, _PUNKT_5])

STILREGELN = STILREGELN_KERN + "\n\n" + _LAENGE_FOTO

STILREGELN_SACHLICH = "STILREGELN (Stil, nicht Fakten)\n\n" + "\n\n".join([_PUNKT_1, _PUNKT_2_SACHLICH, _PUNKT_5.replace("5. ", "3. "), _LAENGE_SACHLICH.replace("6. ", "4. ")])
