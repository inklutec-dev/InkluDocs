"""Gemeinsame Stilregeln fuer Alt-Texte — EINE Quelle fuer Pipeline UND InkluAgent.

Qualitaetsrunde 21.08.2026 (Steve): Der Chatbot-Ton ist die Messlatte —
kurze, natuerlich gebaute Saetze, das Wichtigste zuerst, keine
Detail-Abhak-Prosa. Vorher erzeugte die Zutatenlisten-Struktur der Builder
Alt-Texte im Amtston ("..., den Kopf leicht nach oben links gewandt und den
Mund leicht geoeffnet, sitzt ..."; realer Befund Projekt 390, 18.08.2026).

Dieses Modul ersetzt in der Foto-Familie den frueheren Kompaktheits-Block
und buendelt die vorher verstreuten Stil-Verbote (Einleitungs-Floskeln,
Quellen-Floskeln, Name-als-Satzanfang). Es wird ausserdem in den
InkluAgent-System-Prompt eingebunden (inkluagent/prompts/system_agent.py),
damit Chatbot und Pipeline nach denselben Stilregeln schreiben.
Gepflegt wird ausschliesslich HIER — nicht in den Buildern nachziehen.

Abgrenzung: Fakten-Regeln (Belege, Halluzination, Hedging) leben in
constraints/halluzination.py. Hier steht nur STIL.

Zwei Exporte (21.08.2026, Anschluss Daten-Familie):
- STILREGELN_KERN — Punkte 1-5 (reiner Stil, ohne Laengen). Fuer Builder,
  die eigene Laengen-Richtwerte fuehren (Daten-Familie: 250/350er-Regime).
- STILREGELN — Kern + Punkt 6 (Laengen 150/250/400). Fuer die Foto-Familie
  und den InkluAgent.
Die Mini-Familie (logo/icon/funktional) bleibt bewusst OHNE diesen Block:
ihre Formel (Funktion zuerst, 3-80 Zeichen) IST bereits die Stilvorgabe.
"""

STILREGELN_KERN = """STILREGELN (fuer Alt-Text UND Langbeschreibung — Stil, nicht Fakten)

1. WICHTIGSTES ZUERST: Fuehre mit der Information, wegen der das Bild an
   seiner Stelle steht — Wer oder Was und die sichtbare Situation. Jedes
   weitere Detail muss die Frage bestehen: Hilft es, dieses Bild an dieser
   Stelle zu verstehen? Wenn nein, gehoert es nicht in den Alt-Text —
   sondern in die Langbeschreibung oder nirgendwohin.

2. NATUERLICHER SATZBAU: Schreibe wie ein guter Redakteur — Subjekt und
   Verb stehen frueh und nah beieinander, ein bis zwei Saetze. Keine
   Partizip-Einschuebe zwischen Subjekt und Verb, keine Semikolon-Ketten,
   keine Lage-Floskeln wie "im Bildvordergrund" oder "im Bildhintergrund"
   (stattdessen natuerlich: "vor ihr", "dahinter", "auf dem Tisch").
   GUT: "Anna Reimers in schwarzem Blazer sitzt an einem Holztisch mit
   aufgeklapptem Laptop vor einer hellen Wand."
   SCHLECHT: "Anna Reimers in schwarzem Blazer, den Kopf leicht nach oben
   links gewandt und den Mund leicht geoeffnet, sitzt vor einer hellen
   Wand; im Bildvordergrund ein aufgeklapptes Laptop auf einem Holztisch."

3. KOERPERDETAILS NUR MIT BEDEUTUNG: Kopfhaltung, Blickrichtung,
   Mundstellung, Gestik und Mimik gehoeren NICHT in den Alt-Text — ausser
   sie tragen die Kernaussage des Bildes (die Rednerin zeigt auf die
   Leinwand; zwei Personen geben sich die Hand). In der Langbeschreibung
   nur dort, wo sie die Szene wirklich nachvollziehbarer machen.

4. NAME ALS SATZANFANG: Ein verwendeter Name ist das SUBJEKT des ersten
   Satzes ("Anna Reimers, Gruenderin von Beispielwerk, sitzt an einem
   Holztisch ..."). FALSCH ist die Etikett-Struktur "Name, Funktion: Ein
   Mann ..." — die benannte Person wird danach NIE erneut anonym
   eingefuehrt ("ein Mann", "eine Frau", "eine Person"); stattdessen
   Pronomen oder Rolle ("der Gruender", "die Physikerin").

5. KEINE FLOSKELN: Nicht mit "Das Bild zeigt", "Das Foto zeigt", "Auf dem
   Bild", "Auf dem Foto", "Zu sehen ist" oder "Hier sieht man" beginnen —
   direkt mit dem Motiv einsteigen. Ebenso verboten sind Quellen-Floskeln
   wie "laut Seitenkontext", "laut Kontext", "dem Kontext zufolge" oder
   "laut Bildunterschrift": Eine belegte Angabe wird direkt ausgesagt,
   ohne ihre Herkunft zu nennen."""

STILREGELN = STILREGELN_KERN + """

6. LAENGE (Arbeitsteilung Alt-Text / Langbeschreibung): So kurz wie
   moeglich, so lang wie noetig. Richtwert fuer den Alt-Text: einfache
   Motive unter 150 Zeichen, komplexe Szenen bis etwa 250. Die 400 Zeichen
   des Schemas sind eine harte Obergrenze, KEIN Ziel. Der Alt-Text traegt
   die Essenz — Wissens-Tiefe, Nebendetails und raeumliche Ausfuehrung
   gehoeren in die Langbeschreibung."""
