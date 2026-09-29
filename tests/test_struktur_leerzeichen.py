"""Leerzeichen in der Strukturlesung (Struktur_Export.py), 28.09.2026.

Anlass: Michael Karbe, „Die aktuelle Prüfung scheint nicht auf veraPDF zu basieren“. Auf Seiten mit Leerzeichen-Objekten
(Browser-Druck) verband die alte Seitenregel ALLE Textstücke ohne Leerzeichen — aus Diagramm-Beschriftungen wurde
„FörderungfürPlug-In-Hybride“, die Vollständigkeitsprüfung meldete Text als fehlend. Jetzt entscheidet die Worterkennung
(PyMuPDF): verbunden wird nur, was im selben Wort liegt. Seiten ohne Leerzeichen-Objekte bleiben wie bisher.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "pdfix_scripts"))
sys.path.insert(0, "/app/pdfix_scripts")
try:
    import Struktur_Export as se
except Exception as e:  # pdfixsdk fehlt ausserhalb des Containers
    se = None
    GRUND = str(e)


@unittest.skipIf(se is None, "Struktur_Export nicht ladbar (pdfixsdk fehlt)")
class Leerzeichen(unittest.TestCase):
    def setUp(self):
        # Seite 0: Woerter „Förderung“ (Zeile 1), „für“ und „Plug-In-Hybride“ (Zeile 2), „Vertrag“ (Zeile 3, aus 3 Stücken)
        se._woerter_cache.clear()
        se._unterkanten_cache.clear()
        # wie _woerter sie liefert: nach Unterkante sortiert (bisect in _selbes_wort)
        se._woerter_cache[0] = sorted([(10, 100, 60, 110), (10, 85, 25, 95), (30, 85, 90, 95), (10, 70, 50, 80)], key=lambda w: w[1])

    def test_andere_zeile_bekommt_leerzeichen(self):
        self.assertTrue(se._getrennt(0, (10, 100, 60, 110), (10, 85, 25, 95)))

    def test_selbe_zeile_anderes_wort_bekommt_leerzeichen(self):
        # „für“ und „Plug-In-Hybride“: normaler Wortabstand, kein Leerzeichen-Objekt dazwischen
        self.assertTrue(se._getrennt(0, (10, 85, 25, 95), (30, 85, 90, 95)))

    def test_bruchstuecke_eines_worts_bleiben_zusammen(self):
        # Browser-Druck: „Ve“ „rt“ „rag“ sind Stücke EINES Worts
        self.assertFalse(se._getrennt(0, (10, 70, 20, 80), (21, 70, 32, 80)))
        self.assertFalse(se._getrennt(0, (21, 70, 32, 80), (33, 70, 50, 80)))

    def test_verketten_auf_leerzeichen_seite(self):
        out = se._append_fragment("Förderung", "für", 3, True, se._getrennt(0, (10, 100, 60, 110), (10, 85, 25, 95)))
        out = se._append_fragment(out, "Plug-In-Hybride", 15, True, se._getrennt(0, (10, 85, 25, 95), (30, 85, 90, 95)))
        self.assertEqual(out, "Förderung für Plug-In-Hybride")
        wort = se._append_fragment("Ve", "rt", 2, True, se._getrennt(0, (10, 70, 20, 80), (21, 70, 32, 80)))
        self.assertEqual(se._append_fragment(wort, "rag", 3, True, se._getrennt(0, (21, 70, 32, 80), (33, 70, 50, 80))), "Vertrag")

    def test_seite_ohne_leerzeichen_objekte_unveraendert(self):
        self.assertEqual(se._append_fragment("Förderung", "für", 3, False, False), "Förderung für")
        self.assertEqual(se._append_fragment("Auftrags-", "verarbeitung", 12, False, False), "Auftragsverarbeitung")

    def test_ohne_worterkennung_grobe_regel(self):
        se._woerter_cache[1] = None
        self.assertTrue(se._getrennt(1, (10, 100, 60, 110), (10, 85, 25, 95)))    # andere Zeile
        self.assertFalse(se._getrennt(1, (10, 70, 20, 80), (21, 70, 32, 80)))     # dicht dran
        self.assertTrue(se._getrennt(1, (10, 70, 20, 80), (40, 70, 50, 80)))      # grosse Luecke


if __name__ == "__main__":
    unittest.main()
