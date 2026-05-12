"""Regeln zur Kontext-Anreicherung — erlaubt vs. verboten.

E4-Korrektur (Steve-Review, 04.05.2026): Drogeriemarkt-Beispiel präzisiert.
'Beim Einkaufen' war erfundene Handlung; jetzt 'Innenraum eines Drogeriemarktes'.
"""

KONTEXT_NUTZUNGS_REGELN = """KONTEXT-NUTZUNG — Anreicherung erlaubt, Halluzination verboten:

Der Seiten-/PDF-Kontext (Titel, Überschriften, Meta-Beschreibung, umgebender Text)
hilft dir dabei, das THEMA des Bildes zu verstehen. Er erlaubt dir aber NICHT,
Details ins Bild hineinzuinterpretieren die du nicht siehst.

ERLAUBT — Kontext-Anreicherung bei unspezifischen Bildern:
Wenn das Bild allein betrachtet generisch ist ('Person in einem Raum', 'leere Halle'),
darfst du den Seitenkontext nutzen um dem Bild Bedeutung zu geben:
- Bild zeigt 'leere Halle mit sandigem Boden' + Kontext ist Pferdesport
  → 'Reithalle mit sandigem Boden' OK
- Bild zeigt 'Person in einem Geschäft' + Kontext ist Drogeriemarkt budni
  → 'Person im Innenraum eines Drogeriemarktes (Kette: budni)' OK
  (NICHT: 'Kundin beim Einkaufen' — Person könnte Mitarbeiterin,
   Stöberin oder Lieferantin sein. Handlung NUR wenn aus Pose belegt.)

VERBOTEN — Kontext-Hineininterpretation:
- Bild zeigt unscharfe Felder + Kontext ist 'Bundesanstalt für Landwirtschaft'
  → 'nachhaltige Landwirtschaft symbolisiert' NICHT
  (du erfindest 'nachhaltig', das siehst du nicht)
- Bild zeigt Person + Kontext nennt einen Namen
  → 'Person X' nur wenn die Person eindeutig identifizierbar ist (nicht aus Kontext raten)

FAUSTREGEL: Wenn du einen Bildinhalt nur wegen des Kontexts beschreiben würdest,
aber nicht weil du ihn SIEHST — lass ihn weg.

PFLICHT — Namen aus Bild-Beschriftung oder Kontext:
Wenn der Kontext den NAMEN oder die FUNKTION einer Person nennt UND die Person
identifizierbar ist (z.B. einzige Person im Bild, oder eindeutig zugeordnet via
Bildunterschrift), MUSS dieser Name im Alt-Text stehen. Der Alt-Text muss allein
verständlich sein — der Nutzer sieht den Kontext nicht.

ANTI-REDUNDANZ:
Wiederhole keine BESCHREIBENDEN Details die der Kontext bereits nennt
(z.B. 'ein Buch über Insekten' wenn der Kontext schon das Buch ankündigt).
Aber Namen, Funktionen und Identitäten IMMER nennen — sie sind die Kerninfo.
"""
