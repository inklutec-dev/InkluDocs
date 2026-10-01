"""Michael Karbe, Feedback 20261001 - 1, Punkt 10 (01.10.2026): „Die Anzeige der Problemstellen wird nicht mehr angezeigt.“
veraPDF nennt Seiten nur, wenn sein Kontextpfad eine Seite enthaelt — bei „Figure ohne Alt-Text“ (7.3-1) nie. Seit „nur veraPDF“
fehlten damit Seitenbild und Problemseite. Jetzt kommen die Seiten aus dem Strukturbaum der Pruefdatei.
Laeuft ohne Server: python3 -m unittest tests/test_problemseiten.py"""
import os
import sys
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import abschluss  # noqa: E402

META = {"verapdf": {"punkte": [
    {"status": "befund", "bereich": "Bilder und Grafiken",
     "einzeln": [{"text": "Ein Bild hat keinen Alternativtext. (12-mal)", "seiten": [], "regeln": ["7.3-1"],
                  "satz": "Ein Bild hat keinen Alternativtext.", "mal": "(12-mal)", "lang": ""}]},
    {"status": "befund", "bereich": "PDF/UA-Kennzeichnung",
     "einzeln": [{"text": "Keine PDF/UA-Kennung.", "seiten": [], "regeln": ["5-1"], "satz": "Keine PDF/UA-Kennung.", "mal": "", "lang": ""}]},
    {"status": "befund", "bereich": "Links",
     "einzeln": [{"text": "Link ohne Beschreibung.", "seiten": [4], "regeln": ["7.18.5-2"], "satz": "Link ohne Beschreibung.", "mal": "", "lang": ""}]},
]}}
STRUKTUR = {"elemente": [
    {"id": "1", "typ": "Figure", "seite": 1, "alt": ""}, {"id": "2", "typ": "Figure", "seite": 1},
    {"id": "3", "typ": "Figure", "seite": 3, "alt": "Ein Diagramm"}, {"id": "4", "typ": "Figure", "seite": 5},
    {"id": "5", "typ": "P", "seite": 2, "text": "x"},
]}


class Problemseiten(unittest.TestCase):
    def test_bild_ohne_alt_bekommt_seiten_aus_der_struktur(self):
        pr = abschluss.probleme_zusammenstellen(META, STRUKTUR, [])
        bild = next(p for p in pr if p["regeln"] == ["7.3-1"])
        self.assertEqual(bild["seiten"], [1, 5], "nur Seiten mit Grafik OHNE Alt-Text")
        self.assertEqual(bild["seite"], 1)

    def test_seite_von_verapdf_bleibt_und_dokumentweite_befunde_bleiben_ohne_seite(self):
        pr = abschluss.probleme_zusammenstellen(META, STRUKTUR, [])
        self.assertEqual(next(p for p in pr if p["regeln"] == ["7.18.5-2"])["seiten"], [4])
        self.assertEqual(next(p for p in pr if p["regeln"] == ["5-1"])["seiten"], [])

    def test_ohne_struktur_wie_bisher(self):
        pr = abschluss.probleme_zusammenstellen(META, None, [])
        self.assertEqual(next(p for p in pr if p["regeln"] == ["7.3-1"])["seiten"], [])


if __name__ == "__main__":
    unittest.main()
