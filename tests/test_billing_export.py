"""Aktionspreise (Michael Karbe, bestaetigt 29.08.2026): Preisliste, Export-Staffel je
Art, Wache vor kostenpflichtigen Aktionen.
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_billing_export.py -v
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import billing  # noqa: E402


class TestAktionspreise(unittest.TestCase):
    def test_preisliste_michael_2908(self):
        self.assertEqual(billing.AKTIONS_PREISE["bild_generierung"], 5)
        self.assertEqual(billing.AKTIONS_PREISE["alt_text_aenderung_chatbot"], 5)
        self.assertEqual(billing.AKTIONS_PREISE["quickinfo_generierung"], 1)
        self.assertEqual(billing.AKTIONS_PREISE["quickinfo_aenderung_chatbot"], 1)
        self.assertEqual(billing.AKTIONS_PREISE["pdf_export"], 25)
        self.assertEqual(billing.AKTIONS_PREISE["docx_export"], 25)
        self.assertEqual(billing.AKTIONS_PREISE["formular_export"], 25)
        self.assertEqual(billing.AKTIONS_PREISE["csv_export"], 10)
        self.assertEqual(billing.AKTIONS_PREISE["json_export"], 10)
        self.assertEqual(billing.AKTIONS_PREISE["formular_csv_export"], 10)
        self.assertEqual(billing.EXPORT_SCHRITT, 10)
        self.assertEqual(billing.EXPORT_ARTEN["pdf"], ("pdf_export", 5))
        self.assertEqual(billing.EXPORT_ARTEN["docx"], ("docx_export", 5))
        self.assertEqual(billing.EXPORT_ARTEN["formular"], ("formular_export", 1))
        self.assertEqual(billing.TABELLEN_EXPORTE, {"csv": "csv_export", "json": "json_export",
                                                    "formular_csv": "formular_csv_export"})

    def test_kontingente_mal_fuenf(self):
        self.assertEqual(billing.PLAN_KONTINGENTE["free"], 50)
        self.assertEqual(billing.PLAN_KONTINGENTE["single"], 250)
        self.assertEqual(billing.PLAN_KONTINGENTE["team"], 500)
        self.assertEqual(billing.PLAN_KONTINGENTE["enterprise"], 1375)

    def test_aktion_preis(self):
        self.assertEqual(billing.aktion_preis("bild_generierung"), 5)
        self.assertEqual(billing.aktion_preis("bild_generierung", 3), 15)
        self.assertEqual(billing.aktion_preis("quickinfo_generierung", 26), 26)
        self.assertEqual(billing.aktion_preis("quickinfo_generierung", 0), 0)
        self.assertEqual(billing.aktion_preis("csv_export"), 10)


class TestExportPreis(unittest.TestCase):
    def test_staffel_bilder(self):
        for art in ("pdf", "docx"):
            self.assertEqual(billing.export_preis(0, art), 25)
            self.assertEqual(billing.export_preis(1, art), 30)
            self.assertEqual(billing.export_preis(10, art), 30)
            self.assertEqual(billing.export_preis(11, art), 35)
            self.assertEqual(billing.export_preis(26, art), 40)
            self.assertEqual(billing.export_preis(50, art), 50)
            self.assertEqual(billing.export_preis(100, art), 75)

    def test_staffel_felder(self):
        self.assertEqual(billing.export_preis(0, "formular"), 25)
        self.assertEqual(billing.export_preis(1, "formular"), 26)
        self.assertEqual(billing.export_preis(12, "formular"), 27)
        self.assertEqual(billing.export_preis(26, "formular"), 28)
        self.assertEqual(billing.export_preis(50, "formular"), 30)

    def test_standard_art_ist_pdf(self):
        self.assertEqual(billing.export_preis(26), billing.export_preis(26, "pdf"))

    def test_unbekannte_art(self):
        with self.assertRaises(KeyError):
            billing.export_preis(1, "xlsx")

    def test_frontend_preisliste(self):
        d = billing.preise_fuer_frontend()
        self.assertEqual(d["bild_generierung"], 5)
        self.assertEqual(d["export_schritt"], 10)
        self.assertEqual(d["export_staffel"], {"pdf": 5, "docx": 5, "formular": 1})


class TestWache(unittest.TestCase):
    """aktion_pruefung / export_pruefung entscheiden allein ueber verfuegbare_credits."""

    def test_reicht(self):
        with mock.patch.object(billing, "verfuegbare_credits", return_value=12):
            p = billing.aktion_pruefung(1, "bild_generierung", 2)
            self.assertEqual((p["preis"], p["erlaubt"], p["fehlend"], p["verfuegbar"]), (10, True, 0, 12))

    def test_reicht_nicht(self):
        with mock.patch.object(billing, "verfuegbare_credits", return_value=3):
            p = billing.aktion_pruefung(1, "bild_generierung")
            self.assertEqual((p["preis"], p["erlaubt"], p["fehlend"]), (5, False, 2))
            d = billing.credits_fehlen_detail(p, "Das Neu-Generieren")
            self.assertEqual(d["code"], "credits_fehlen")
            self.assertIn("5 Credits", d["text"])
            self.assertIn("3 Credits", d["text"])
            self.assertIn("5 Credits", billing.credits_fehlen_text(p))

    def test_unbegrenzt(self):
        with mock.patch.object(billing, "verfuegbare_credits", return_value=None):
            self.assertTrue(billing.aktion_pruefung(1, "pdf_export")["erlaubt"])
            self.assertTrue(billing.export_pruefung(1, 999, "pdf")["erlaubt"])
            self.assertEqual(billing.credits_fehlen_detail(billing.export_pruefung(1, 1, "pdf"))["verfuegbar"], None)

    def test_export_pruefung_art(self):
        with mock.patch.object(billing, "verfuegbare_credits", return_value=27):
            self.assertTrue(billing.export_pruefung(1, 12, "formular")["erlaubt"])
            p = billing.export_pruefung(1, 12, "pdf")
            self.assertEqual((p["preis"], p["erlaubt"], p["fehlend"], p["art"], p["anzahl"]), (35, False, 8, "pdf", 12))

    def test_free_beispiel_michael(self):
        # 26-Felder-Formular im Free-Tarif: 26 (Felder) + 28 (Export) = 54 > 50 -> braucht Abo/Paket
        gen = billing.aktion_preis("quickinfo_generierung", 26)
        exp = billing.export_preis(26, "formular")
        self.assertEqual(gen + exp, 54)
        self.assertGreater(gen + exp, billing.PLAN_KONTINGENTE["free"])


if __name__ == "__main__":
    unittest.main()
