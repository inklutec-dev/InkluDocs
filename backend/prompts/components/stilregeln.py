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
   bis drei kurze Sätze. Keine Partizip-Einschübe zwischen Subjekt und Verb, keine
   Semikolon-Ketten, keine Lagefloskeln wie "im Bildvordergrund" (stattdessen
   "vor ihr", "dahinter", "auf dem Tisch").
   Gut: "Anna Reimers in schwarzem Blazer sitzt an einem Holztisch mit
   aufgeklapptem Laptop vor einer hellen Wand."
   Schlecht: "Anna Reimers in schwarzem Blazer, den Kopf leicht nach oben links
   gewandt und den Mund leicht geöffnet, sitzt vor einer hellen Wand; im
   Bildvordergrund ein aufgeklapptes Laptop auf einem Holztisch.\""""

_PUNKT_2_SACHLICH = """2. Natürlicher Satzbau. Subjekt und Verb stehen früh und nah beieinander, zwei
   bis drei kurze Sätze im Alt-Text. Keine Semikolon-Ketten, keine Lagefloskeln
   wie "im Bildvordergrund".
   Gut: "Balkendiagramm zum Umsatz der Musterwerk GmbH 2021 bis 2023 in drei
   Sparten, in Millionen Euro: Service liegt am Ende mit 6,2 vorn, nach einem
   Einbruch 2022. Produkte fallen durchgehend von 5,1 auf 3,4, Lizenzen steigen
   von 2,0 auf 2,8."
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
   Rolle weitergeführt. Ein Anlass oder ein Gattungswort darf den Satz mit
   Doppelpunkt eröffnen ("Workshop der Musterwerk GmbH: acht Personen stehen in
   einer Reihe"); nur ein Personenname steht nicht als Etikett vor dem
   Doppelpunkt."""

_PUNKT_5 = """5. Keine Floskeln. Keine Ansage, dass etwas gezeigt wird: nicht "Das Bild
   zeigt", "Die Aufnahme zeigt", "Zu sehen ist", "Hier sieht man" und keine
   sinngemäße Variante, weder am Anfang noch mitten im Text. Steige direkt mit
   dem Motiv ein: statt "Die Aufnahme zeigt den Dom von Südwesten" schreibe
   "Blick von Südwesten auf den Dom". Keine Quellenhinweise wie "laut Kontext"
   oder "laut Bildunterschrift": Eine belegte Angabe wird direkt ausgesagt."""

_LAENGE_FOTO = """6. Länge und Arbeitsteilung. Der Alt-Text muss allein tragen: In PDF- und
   Word-Dokumenten ist er der einzige Text, den der Leser bekommt. Er sagt
   zuerst, wer oder was zu sehen ist und was das Bild aussagt, dann die
   Kernfakten, ohne die das Bild nicht verstanden ist. So kurz wie möglich, so
   lang wie nötig: einfache Motive in einem Satz, komplexe Szenen in zwei bis
   drei kurzen Sätzen. Die 400 Zeichen des Schemas sind eine Obergrenze, kein
   Ziel. Schreibe so, wie du es einem Kollegen am Telefon sagst, der das Bild
   nicht sieht und sofort mitreden muss: Alltagssprache, die jeder versteht,
   kein Amtston, keine Aufzählung. Die Langbeschreibung vertieft: Nebendetails,
   räumliche Anordnung, Wissenstiefe, in Fließtext ohne Überschriften und
   Aufzählungszeichen. Sie beginnt nicht mit einer Ansage wie "Das Bild zeigt",
   wiederholt den Alt-Text nicht und widerspricht ihm in keinem Punkt."""

_SCHREIBWEISE_SACHLICH = """5. Schreibweise. Das Gattungswort steht als normales Wort im Satz ("Tabelle der
   Nährwerte je 100 Gramm:", "Karte der Beratungsstellen:"), nicht als Etikett mit
   Gedankenstrich. Senkrechte und waagerechte Balken heißen Balkendiagramm, nicht
   Säulendiagramm. Dezimalzahlen mit Komma (61,3 Prozent), Tausender mit Punkt
   (4.478 Meter), Einheiten ausgeschrieben oder wie im Bild gedruckt."""

_LAENGE_SACHLICH = """6. Länge und Arbeitsteilung. Der Alt-Text muss allein tragen: In PDF- und
   Word-Dokumenten ist er der einzige Text, den der Leser bekommt. Er sagt
   zuerst, was die Grafik ist und was sie aussagt, dann die Kernfakten, ohne die
   die Aussage nicht stimmt: bei Diagrammen jede Reihe mit Richtung und dem
   Wert, der sie trägt, bei Tabellen die tragenden Werte, bei Abläufen die
   Stationen. So kurz wie möglich, so lang wie nötig: meist zwei bis drei kurze
   Sätze, bei dichten Grafiken bis etwa 350 Zeichen. Die 400 Zeichen des Schemas
   sind eine Obergrenze, kein Ziel. Schreibe so, wie du es einem Kollegen am
   Telefon sagst, der die Grafik nicht sieht und sofort mitreden muss:
   Alltagssprache, die jeder versteht, kein Amtston, keine Zahlenliste ohne
   Zusammenhang. Die Langbeschreibung ist bei diesem Bildtyp Pflicht und
   vertieft: Aufbau, Achsen und Legende, alle Werte, Reihenfolgen und lesbare
   Texte in Fließtext ohne Überschriften, Tabellen und Aufzählungszeichen,
   höchstens 2000 Zeichen. Auch sie beginnt nicht mit einer Ansage wie "Das
   Diagramm zeigt" oder "Die Tabelle enthält", sondern mit dem Inhalt. Sie
   wiederholt den Alt-Text nicht und widerspricht ihm in keiner Zahl."""

STILREGELN_KERN = "STILREGELN (Stil, nicht Fakten)\n\n" + "\n\n".join([_PUNKT_1, _PUNKT_2_FOTO, _PUNKT_3, _PUNKT_4, _PUNKT_5])

STILREGELN = STILREGELN_KERN + "\n\n" + _LAENGE_FOTO

STILREGELN_SACHLICH = "STILREGELN (Stil, nicht Fakten)\n\n" + "\n\n".join([_PUNKT_1, _PUNKT_2_SACHLICH, _PUNKT_5.replace("5. ", "3. "), _SCHREIBWEISE_SACHLICH.replace("5. ", "4. "), _LAENGE_SACHLICH.replace("6. ", "5. ")])
