"""Tests fuer pdfix_roundtrip (24.04.2026).

Testet, dass Heines Export/Import-Scripts ueber das Modul-Wrapper korrekt
laufen. Nutzt die Test-PDF aus Heines ZIP (4 Figures).

Ausfuehrung im Container:
  docker exec inkludocs-staging python -m pytest /app/tests/test_pdfix_roundtrip.py -v
"""
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

# Pfad zu Heines Test-PDF im Container. Wenn nicht vorhanden, Tests skippen.
_TEST_PDF_CANDIDATES = [
    "/app/backend/pdfix_scripts/test_data/PDFix_AutoTag_Test_01.pdf",
    "/app/tests/fixtures/PDFix_AutoTag_Test_01.pdf",
    "/opt/pdfix-work/PDFix_AutoTag_Test_01.pdf",
]
_TEST_ZIP = "/opt/pdfix-work/AltTag_Scripte.zip"

import sys
sys.path.insert(0, "/app/backend")

try:
    import pdfix_roundtrip
    _MOD_OK = True
except ImportError as e:
    _MOD_OK = False
    _IMPORT_ERR = str(e)


def _find_test_pdf() -> str | None:
    for c in _TEST_PDF_CANDIDATES:
        if os.path.exists(c):
            return c
    if os.path.exists(_TEST_ZIP):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(_TEST_ZIP) as z:
            for n in z.namelist():
                if n.endswith("PDFix_AutoTag_Test_01.pdf"):
                    z.extract(n, tmp)
                    return os.path.join(tmp, n)
    return None


@unittest.skipUnless(_MOD_OK, f"pdfix_roundtrip nicht importierbar")
class TestPdfixRoundtrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_pdf = _find_test_pdf()
        if not cls.test_pdf:
            raise unittest.SkipTest("Test-PDF nicht gefunden")
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp") and os.path.isdir(cls.tmp):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pdfix_available(self):
        self.assertTrue(pdfix_roundtrip.is_pdfix_available(),
                        "pdfix-sdk muss im Container installiert sein")

    def test_is_tagged_pdf_true(self):
        self.assertTrue(pdfix_roundtrip.is_tagged_pdf(self.test_pdf),
                        "Heines Test-PDF ist getaggt und muss als solche erkannt werden")

    def test_is_tagged_pdf_false_for_missing(self):
        self.assertFalse(pdfix_roundtrip.is_tagged_pdf("/nonexistent.pdf"))

    def test_extract_figures(self):
        out_dir = os.path.join(self.tmp, "extract")
        figures = pdfix_roundtrip.extract_figures_pdfix(self.test_pdf, out_dir)
        self.assertEqual(len(figures), 4, "Heines Test-PDF hat genau 4 Figures")
        for fig in figures:
            self.assertIn("lfnr", fig)
            self.assertIn("path", fig)
            self.assertTrue(os.path.exists(fig["path"]),
                            f"Export-PNG nicht gefunden: {fig['path']}")
            self.assertGreater(os.path.getsize(fig["path"]), 0)
        csv_path = os.path.join(out_dir, "figure_array.csv")
        self.assertTrue(os.path.exists(csv_path), "CSV muss erzeugt werden")

    def test_roundtrip_alt_text_persists(self):
        out_dir = os.path.join(self.tmp, "roundtrip")
        figures = pdfix_roundtrip.extract_figures_pdfix(self.test_pdf, out_dir)
        self.assertEqual(len(figures), 4)
        pdf_out = os.path.join(self.tmp, "with_alt.pdf")
        alt_texts = {1: "Test-Alt 1",
                     2: "Test-Alt 2",
                     3: "Test-Alt 3",
                     4: "Test-Alt 4"}
        count = pdfix_roundtrip.import_alt_texts_pdfix(
            self.test_pdf, pdf_out, alt_texts, work_dir=out_dir)
        self.assertEqual(count, 4)
        self.assertTrue(os.path.exists(pdf_out))
        # Re-Export und pruefen, dass ein Alt-Text drin ist
        reverify_dir = os.path.join(self.tmp, "reverify")
        figures2 = pdfix_roundtrip.extract_figures_pdfix(pdf_out, reverify_dir)
        alts_after = [f["alt"] for f in figures2]
        self.assertIn("Test-Alt 1", alts_after,
                      f"Nach Roundtrip muss Alt-Text drin sein, fand: {alts_after}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
