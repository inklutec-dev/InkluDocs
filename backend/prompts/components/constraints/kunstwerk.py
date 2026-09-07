"""Kunstwerk-Regel für Gemälde, Zeichnung, Druckgrafik und Skulptur.

Historie (für Menschen): eingeführt am 07.09.2026 nach einem Kundenbefund. Eine
kleine Reproduktion des Distelfinks von Fabritius erhielt eine erfundene
"Holzbank mit Kette". Eingebunden in foto_objekte, foto_personen und illustration.
"""

KUNSTWERK_REGEL = """KUNSTWERKE (Gemälde, Zeichnung, Druckgrafik, Skulptur)

Ist das Bild erkennbar ein Kunstwerk, gilt: weniger ist mehr.
- Benenne Titel, Künstler und Jahr, wenn sie im Bild lesbar sind oder das Werk
  weltbekannt und zweifelsfrei erkennbar ist. Sonst beschreibe nur.
- Dann das Motiv in ein bis zwei Sätzen: wer oder was, Haltung, unmittelbare
  Umgebung, Farbklima. Der Alt-Text bleibt unter 200 Zeichen.
- Keine Detailinventur, keine Deutung von Maltechnik oder Stil, keine
  kunsthistorische Einordnung über die Benennung und ein Kenn-Faktum hinaus.
- Bei kleinen oder unscharfen Reproduktionen lässt du weg, was du nicht sicher
  siehst. Details gehören in die Langbeschreibung nur, wenn sie eindeutig
  sichtbar sind."""
