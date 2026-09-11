"""Meine Ausgaben (11.09.2026): Vorschaubild der ersten Seite.
    docker exec -w /app inkludocs-staging python3 /app/tests/test_ausgaben.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import pdfua_export  # noqa: E402


def _mini_pdf() -> bytes:
    import fitz
    doc = fitz.open()
    seite = doc.new_page(width=595, height=842)
    seite.insert_text((72, 100), "Testdokument InkluDocs (fiktiv)", fontsize=18)
    return doc.tobytes()


class TestVorschau(unittest.TestCase):
    def test_png_erste_seite(self):
        png = pdfua_export.vorschau_png(_mini_pdf(), breite=300)
        self.assertIsNotNone(png)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        import fitz
        pix = fitz.Pixmap(png)
        self.assertTrue(280 <= pix.width <= 320, pix.width)   # Breite ~ gewuenscht, Hoehe im Seitenverhaeltnis
        self.assertGreater(pix.height, pix.width)

    def test_kaputte_pdf_gibt_none(self):
        self.assertIsNone(pdfua_export.vorschau_png(b"kein pdf"))


if __name__ == "__main__":
    unittest.main()
