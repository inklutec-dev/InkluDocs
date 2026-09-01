# -*- coding: utf-8 -*-
"""PDFix-Rueckweg: laufende Nummer je DOKUMENT, nicht projektweit (Befund Michael Karbe 01.09.2026).

Projekt 376 auf Produktion: Dokument mit 10 Figures, Bilder mit image_index 3..12
(Dokument 1 war geloescht) -> CSV nannte 11 und 12, Import brach ab. Der Rang im
Dokument (1..10) ist die richtige Nummer. Laeuft im Container:
python3 -m unittest tests/test_pdfix_lfnr.py
"""
import os
import sys
import unittest

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import main  # noqa: E402


class LfnrJeDokument(unittest.TestCase):
    def test_indizes_ab_3_werden_zu_1_bis_10(self):
        images = [{"id": 100 + k, "image_index": k} for k in range(3, 13)]
        lf = main._pdfix_lfnr_je_dokument(images)
        self.assertEqual(sorted(lf.values()), list(range(1, 11)))
        self.assertEqual(lf[103], 1)
        self.assertEqual(lf[112], 10)

    def test_reihenfolge_nach_image_index_nicht_nach_liste(self):
        images = [{"id": 2, "image_index": 20}, {"id": 1, "image_index": 15}, {"id": 3, "image_index": 26}]
        lf = main._pdfix_lfnr_je_dokument(images)
        self.assertEqual((lf[1], lf[2], lf[3]), (1, 2, 3))

    def test_einzeldokument_bleibt_unveraendert(self):
        images = [{"id": k, "image_index": k} for k in range(1, 6)]
        lf = main._pdfix_lfnr_je_dokument(images)
        self.assertEqual([lf[k] for k in range(1, 6)], [1, 2, 3, 4, 5])

    def test_bilder_ohne_index_bleiben_aussen_vor(self):
        images = [{"id": 1, "image_index": 4}, {"id": 2, "image_index": None}]
        lf = main._pdfix_lfnr_je_dokument(images)
        self.assertEqual(lf, {1: 1})


if __name__ == "__main__":
    unittest.main()
