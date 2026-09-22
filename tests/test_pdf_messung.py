"""Unit-Tests Messwerte + Doppelbeleg (pdf_messung.py, pdf_pruefung.doppelbeleg, 22.09.2026) — ohne Modell."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "/app")
import pdf_messung  # noqa: E402
import pdf_pruefung  # noqa: E402


def _pdf(pfad):
    import fitz
    d = fitz.open()
    p = d.new_page(width=595, height=842)
    p.insert_text((50, 80), "Jahresbericht", fontsize=20, fontname="hebo")          # Titel, fett, allein
    y = 130
    for i in range(6):
        p.insert_text((50, y), f"Zeile {i} des Fliesstextes mit ein paar Woertern drin.", fontsize=10)
        y += 13
    p.insert_text((50, y + 30), "Zahlungsinformationen", fontsize=11, fontname="hebo")   # Zwischenueberschrift
    p.insert_text((50, y + 60), "Bank:", fontsize=10); p.insert_text((150, y + 60), "N26", fontsize=10)
    p.insert_text((50, y + 75), "IBAN:", fontsize=10); p.insert_text((150, y + 75), "DE59 1001", fontsize=10)
    d.save(pfad)
    d.close()


class MessungTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.pdf = os.path.join(cls.dir, "m.pdf")
        _pdf(cls.pdf)
        cls.m = pdf_messung.seiten_messung(cls.pdf)

    def test_fliesstext(self):
        self.assertEqual(self.m[1]["fliesstext"], 10.0)

    def test_titel(self):
        e = pdf_messung.element_messung("Jahresbericht", self.m[1])
        self.assertEqual(e["groesse"], 20.0)
        self.assertTrue(e["fett"])
        self.assertTrue(e["allein"])
        self.assertEqual(e["verhaeltnis"], 2.0)
        self.assertEqual(pdf_messung.messung_text(e), "20 pt fett, allein")

    def test_fliesszeile_im_block(self):
        e = pdf_messung.element_messung("Zeile 2 des Fliesstextes", self.m[1])
        self.assertEqual(e["groesse"], 10.0)
        self.assertFalse(e["fett"])
        self.assertFalse(e["allein"])

    def test_zusammengezogene_zellen(self):
        e = pdf_messung.element_messung("N26 DE59 1001", self.m[1])
        self.assertGreaterEqual(e["zeilen_im_element"], 2)

    def test_unbekannt(self):
        self.assertIsNone(pdf_messung.element_messung("gibt es nicht", self.m[1]))
        self.assertIsNone(pdf_messung.element_messung("", self.m[1]))


class DoppelbelegTest(unittest.TestCase):
    def _elemente(self):
        return [
            {"id": "0.0", "typ": "P", "text": "Jahresbericht", "_messung": {"groesse": 20.0, "fett": True, "allein": True, "fliesstext": 10.0, "verhaeltnis": 2.0, "zeilen_im_element": 1}},
            {"id": "0.1", "typ": "H3", "text": "Hauptstr. 25", "_messung": {"groesse": 10.0, "fett": False, "allein": False, "fliesstext": 10.0, "verhaeltnis": 1.0, "zeilen_im_element": 1}},
            {"id": "0.2", "typ": "H2", "text": "Firma GmbH", "_messung": {"groesse": 10.0, "fett": True, "allein": False, "fliesstext": 10.0, "verhaeltnis": 1.0, "zeilen_im_element": 1}},
            {"id": "0.3", "typ": "Table", "text": ""},
            {"id": "0.3.0", "typ": "TR", "text": ""},
            {"id": "0.3.0.0", "typ": "TH", "text": "Bank:"},
            {"id": "0.3.0.1", "typ": "TH", "text": "N26"},
            {"id": "0.3.1", "typ": "TR", "text": ""},
            {"id": "0.3.1.0", "typ": "TH", "text": "1"},
            {"id": "0.3.1.1", "typ": "TD", "text": "Beratung"},
        ]

    def test_absatz_zu_ueberschrift(self):
        el = self._elemente()
        auto, begr = pdf_pruefung.doppelbeleg({"art": "rolle", "vorschlag": "H1", "sicherheit": "hoch"}, el[0], el)
        self.assertTrue(auto); self.assertIn("hervorgehoben", begr)
        auto, _ = pdf_pruefung.doppelbeleg({"art": "rolle", "vorschlag": "H1", "sicherheit": "mittel"}, el[0], el)
        self.assertFalse(auto)                                   # nur bei „hoch“

    def test_ueberschrift_zu_absatz(self):
        el = self._elemente()
        auto, _ = pdf_pruefung.doppelbeleg({"art": "rolle", "vorschlag": "P", "sicherheit": "hoch"}, el[1], el)
        self.assertTrue(auto)
        auto, begr = pdf_pruefung.doppelbeleg({"art": "rolle", "vorschlag": "P", "sicherheit": "hoch"}, el[2], el)
        self.assertFalse(auto); self.assertIn("widerspricht", begr)   # fett: Messung und Modell uneins -> Hinweis

    def test_tabelle(self):
        el = self._elemente()
        auto, begr = pdf_pruefung.doppelbeleg({"art": "tabelle", "vorschlag": "TD", "sicherheit": "hoch"}, el[6], el)
        self.assertTrue(auto); self.assertIn("Spalte 2", begr)   # Wert rechts neben der Kopfzelle
        auto, _ = pdf_pruefung.doppelbeleg({"art": "tabelle", "vorschlag": "TD", "sicherheit": "hoch"}, el[8], el)
        self.assertTrue(auto)                                     # Zahl in Datenzeile unter Kopfzeile
        auto, _ = pdf_pruefung.doppelbeleg({"art": "tabelle", "vorschlag": "TD", "sicherheit": "hoch"}, el[5], el)
        self.assertFalse(auto)                                    # „Bank:“ ist eine echte Zeilen-Kopfzelle

    def test_ebenen_aus_groesse(self):
        struktur = {"elemente": [{"id": "0", "typ": "H1", "text": "INKLUTEC", "_messung": {"groesse": 24.0}}]}
        befunde = [
            {"auto": True, "vorschlag": "H1", "_groesse": 16.0, "doppelbeleg": "x"},
            {"auto": True, "vorschlag": "H1", "_groesse": 11.0, "doppelbeleg": "y"},
            {"auto": False, "vorschlag": "H1", "doppelbeleg": ""},
        ]
        pdf_pruefung.ebenen_aus_groesse(befunde, struktur)
        self.assertEqual(befunde[0]["vorschlag"], "H2")
        self.assertEqual(befunde[1]["vorschlag"], "H3")
        self.assertIn("Ebene aus Schriftgröße", befunde[1]["doppelbeleg"])
        self.assertEqual(befunde[2]["vorschlag"], "H1")
        self.assertNotIn("_groesse", befunde[0])


if __name__ == "__main__":
    unittest.main()
