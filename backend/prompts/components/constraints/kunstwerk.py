"""Kunstwerk-Regel (07.09.2026, Michael Karbe: "Bei Kunstwerken muessen wir nicht jedes
kleine Detail beschreiben. Weniger ist hier mehr.").

Anlass: Der Distelfink (Fabritius, 1654, nur 151x231 px aus einem Word-Dokument) bekam
eine erfundene "Holzbank mit Kette". Gilt fuer Gemaelde, Zeichnungen, Druckgrafik,
Skulpturen — in foto_objekte, foto_personen und illustration eingebunden.
"""

KUNSTWERK_REGEL = """KUNSTWERKE (Gemaelde, Zeichnung, Druckgrafik, Skulptur)

Ist das Bild erkennbar ein Kunstwerk, gilt: weniger ist mehr.
- Benenne Titel, Kuenstler und Jahr, wenn sie im Bild lesbar sind (Signatur,
  Beschriftung) oder das Werk weltbekannt und zweifelsfrei erkennbar ist —
  so, wie ein Sehender es auf einen Blick erkennt. Sonst beschreibe nur.
- Dann das Motiv in ein bis zwei Saetzen: wer oder was, Haltung, unmittelbare
  Umgebung, Farbklima. Der Alt-Text bleibt unter 200 Zeichen.
- KEINE Detailinventur, keine Deutung von Maltechnik oder Stil, keine
  kunsthistorische Einordnung ueber die Benennung hinaus.
- Bei kleinen oder unscharfen Reproduktionen: Was du nicht sicher siehst,
  laesst du weg — ein erfundener Gegenstand (eine Bank, eine Kette, ein Fenster)
  ist der schwerste Fehler, ein fehlendes Detail keiner.
- Details gehoeren in die Langbeschreibung nur, wenn sie eindeutig sichtbar sind."""
