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
        # seit der Pruefung vom 30.09.2026 (N8) bleibt auch ein Strich am Zeilenende stehen — so steht es im Tag
        self.assertEqual(se._append_fragment("Auftrags-", "verarbeitung", 12, False, False), "Auftrags-verarbeitung")

    def test_echte_bindestriche_bleiben(self):
        """Messlauf 30.09.2026: „KI-gestützte“ wurde „KIgestützte“, „Internet-Services“ „InternetServices“, „E-Mail“ „EMail“.
        Seit der Pruefung vom 30.09.2026 (N8) bleibt jeder Bindestrich (auch U+2010/U+2011), in der Zeile und am
        Zeilenende; „blau-“/„grün“ wird nicht mehr „blaugrün“. Nur der weiche Trennstrich (U+00AD) faellt weg."""
        af = se._append_fragment
        self.assertEqual(af("Die KI-", "gestützte Prüfung", 3, False, False, False), "Die KI-gestützte Prüfung")
        self.assertEqual(af("die Internet-", "Services", 13, False, False, True), "die Internet-Services")
        self.assertEqual(af("per E-", "Mail", 6, False, False, True), "per E-Mail")
        self.assertEqual(af("EU-", "Standardvertragsklauseln", 3, False, False, True), "EU-Standardvertragsklauseln")
        self.assertEqual(af("Seiten 3-", "5", 9, False, False, True), "Seiten 3-5")
        self.assertEqual(af("die KI-", "gestützte", 7, False, False, None), "die KI-gestützte")
        self.assertEqual(af("blau-", "grün", 5, False, False, True), "blau-grün")
        self.assertEqual(af("rot-", "grün", 4, False, False, None), "rot-grün")
        self.assertEqual(af("Auftrags-", "verarbeitung", 12, False, False, True), "Auftrags-verarbeitung")
        self.assertEqual(af("blau\u2010", "grün", 5, False, False, True), "blau\u2010grün")   # U+2010 HYPHEN
        self.assertEqual(af("blau\u2011", "grün", 5, False, False, False), "blau\u2011grün")  # U+2011 geschützt
        self.assertEqual(af("Silben\xad", "trennung", 7, False, False, True), "Silbentrennung")   # weicher Trennstrich
        self.assertEqual(af("Ein-", "und Ausgabe", 4, False, False, True), "Ein- und Ausgabe")
        self.assertEqual(af("Seite 3 -", "5", 9, False, False, False), "Seite 3 - 5")
        self.assertEqual(af("Seite 3 \u2010", "5", 9, False, False, False), "Seite 3 \u2010 5")
        wort = af("KI", "-", 2, True, False, False)
        self.assertEqual(af(wort, "g", 1, True, False, False), "KI-g")

    def test_zeilenwechsel(self):
        self.assertIsNone(se._zeilenwechsel(None, (10, 70, 20, 80)))
        self.assertFalse(se._zeilenwechsel((10, 70, 20, 80), (21, 70, 32, 80)))
        self.assertTrue(se._zeilenwechsel((10, 70, 20, 80), (10, 55, 32, 65)))

    def test_ohne_worterkennung_grobe_regel(self):
        se._woerter_cache[1] = None
        self.assertTrue(se._getrennt(1, (10, 100, 60, 110), (10, 85, 25, 95)))    # andere Zeile
        self.assertFalse(se._getrennt(1, (10, 70, 20, 80), (21, 70, 32, 80)))     # dicht dran
        self.assertTrue(se._getrennt(1, (10, 70, 20, 80), (40, 70, 50, 80)))      # grosse Luecke


if __name__ == "__main__":
    unittest.main()
