"""Vektor-Extraktion (Ersatzweg ohne PDFix), 14.09.2026: seitengrosse Vektorbereiche, die Fotos enthalten, sind
Seitenlayout und kein Bild (Prod-Dokument 430: 20 von 62 Vektorbildern waren ganze Zeitschriftenseiten).
Echte Grafiken ohne Fotos (Diagramme) bleiben.
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_vektor_layout.py -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import fitz  # noqa: E402
import pdf_processor  # noqa: E402


def _seite_mit_rahmen_und_foto(doc, mit_foto: bool):
    """Seite: grosser Rahmen + Linien ueber die ganze Seite (wie InDesign-Layout), optional ein Foto darin."""
    page = doc.new_page(width=400, height=600)
    sh = page.new_shape()
    sh.draw_rect(fitz.Rect(20, 20, 380, 580)); sh.finish(color=(0.8, 0, 0), width=2)
    sh.draw_rect(fitz.Rect(30, 30, 370, 570)); sh.finish(color=(0, 0, 0.6), width=1)
    for y in (200, 300, 400):
        sh.draw_line(fitz.Point(40, y), fitz.Point(360, y)); sh.finish(color=(0, 0, 0), width=1)
    sh.commit()
    page.insert_text(fitz.Point(50, 120), "Zeitschriftenseite mit viel Text", fontsize=14)
    if mit_foto:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 200), 0); pix.set_rect(pix.irect, (10, 120, 10))
        page.insert_image(fitz.Rect(60, 220, 340, 380), pixmap=pix)
    return page


def _seite_mit_diagramm(doc):
    """Seite: Balkendiagramm aus vielen Pfaden, ohne Foto — muss als Bild erhalten bleiben."""
    page = doc.new_page(width=400, height=600)
    sh = page.new_shape()
    for i in range(6):
        sh.draw_rect(fitz.Rect(40 + i * 55, 500 - i * 60, 80 + i * 55, 520)); sh.finish(color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    sh.draw_line(fitz.Point(30, 520), fitz.Point(380, 520)); sh.finish(color=(0, 0, 0), width=1)
    sh.draw_line(fitz.Point(30, 520), fitz.Point(30, 100)); sh.finish(color=(0, 0, 0), width=1)
    sh.commit()
    return page


class TestSeitenlayout(unittest.TestCase):
    def test_regel_direkt(self):
        seite = fitz.Rect(0, 0, 400, 600)
        gross = fitz.Rect(10, 10, 390, 590); klein = fitz.Rect(10, 10, 200, 200)
        foto = fitz.Rect(60, 220, 340, 380)
        self.assertTrue(pdf_processor._ist_seitenlayout(gross, seite, [foto]))
        self.assertFalse(pdf_processor._ist_seitenlayout(gross, seite, []), "ohne Foto: echte Grafik (z. B. Organisationsplan)")
        self.assertFalse(pdf_processor._ist_seitenlayout(klein, seite, [foto]), "kleiner als halbe Seite: kein Layout")

    def test_extraktion_ueberspringt_seitenlayout_behaelt_diagramm(self):
        with tempfile.TemporaryDirectory() as d:
            doc = fitz.open()
            _seite_mit_rahmen_und_foto(doc, mit_foto=True)   # Seite 1: Layout mit Foto -> nur das Foto
            _seite_mit_diagramm(doc)                          # Seite 2: Diagramm -> Vektorbild
            _seite_mit_rahmen_und_foto(doc, mit_foto=False)  # Seite 3: Rahmen ohne Foto -> darf als Grafik bleiben
            pfad = os.path.join(d, "t.pdf"); doc.save(pfad); doc.close(); os.makedirs(os.path.join(d, "out"))
            bilder = pdf_processor.extract_images_from_pdf(pfad, os.path.join(d, "out"), 0)
            vek = [b for b in bilder if b.get("is_vector")]
            ras = [b for b in bilder if not b.get("is_vector")]
            self.assertEqual([b["page_number"] for b in ras], [1], bilder)
            self.assertNotIn(1, [b["page_number"] for b in vek], "Seitenlayout mit Foto darf kein Vektorbild werden")
            self.assertIn(2, [b["page_number"] for b in vek], "Diagramm muss Vektorbild bleiben")


if __name__ == "__main__":
    unittest.main()
