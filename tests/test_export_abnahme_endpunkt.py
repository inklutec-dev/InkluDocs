"""Export-Abnahme im Export-Ablauf (15.09.2026): Befund = HTTP 422, keine Datei, keine Credits;
Ausfall der Abnahme selbst = Datei + Hinweis.
    docker exec -w /app inkludocs-staging python3 -m unittest tests.test_export_abnahme_endpunkt -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import fitz  # noqa: E402
from fastapi import HTTPException  # noqa: E402
import main  # noqa: E402


def _quelle(pfad):
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0)
    pix.set_rect(pix.irect, (30, 30, 200))
    page.insert_image(fitz.Rect(20, 20, 120, 120), pixmap=pix)
    doc.save(pfad)
    xref = page.get_images()[0][0]
    doc.close()
    return xref


def _unit(quelle, xref):
    return {"doc": {"id": 7, "project_id": 3, "original_path": quelle, "original_filename": "q.pdf",
                    "extraction_method": "fitz", "display_name": "Testdokument"},
            "images": [_bild(xref)]}


def _bild(xref):
    """Bildzeile wie aus der Tabelle images (alle Spalten, die die Ausgabe-Regeln lesen)."""
    spalten = ["id", "project_id", "page_number", "image_index", "image_path", "image_type", "alt_text",
               "alt_text_edited", "context_text", "width", "height", "xref", "status", "created_at", "konfidenz",
               "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "is_vector", "langbeschreibung", "original_alt",
               "feedback", "needs_review", "pipeline_steps", "validation_result", "page_view_path", "page_text",
               "document_id", "context_mode", "review_status", "reviewed_at", "gen_language", "original_filename",
               "display_name", "docx_anker", "alt_text_vorher"]
    b = {k: None for k in spalten}
    b.update({"id": 1, "project_id": 3, "page_number": 1, "image_index": 1, "image_type": "foto",
              "alt_text": "Blaues Quadrat", "xref": xref, "status": "done", "is_vector": 0, "document_id": 7})
    return b


class TestAbnahmeImExport(unittest.TestCase):
    def test_sauber_liefert_datei_und_abnahme(self):
        with tempfile.TemporaryDirectory() as d:
            q = os.path.join(d, "q.pdf"); xref = _quelle(q)
            out, info = main._build_pdf_for_document(_unit(q, xref), os.path.join(d, "out"))
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(info["abnahme"]["ok"], info)
            self.assertNotIn("warnings", info)

    def test_befund_verweigert_export(self):
        with tempfile.TemporaryDirectory() as d:
            q = os.path.join(d, "q.pdf"); xref = _quelle(q)
            befund = {"ok": False, "befunde": ["Testbefund: 3 Waisen"], "kennzahlen": {}}
            with mock.patch("export_abnahme.abnahme_pdf", return_value=befund):
                with self.assertRaises(HTTPException) as cm:
                    main._build_pdf_for_document(_unit(q, xref), os.path.join(d, "out"))
            self.assertEqual(cm.exception.status_code, 422)
            self.assertIn("Testbefund: 3 Waisen", cm.exception.detail)
            self.assertIn("keine Credits", cm.exception.detail)
            self.assertEqual(os.listdir(os.path.join(d, "out")), [], "Datei mit Befund darf nicht liegen bleiben")

    def test_abnahme_ausfall_liefert_mit_hinweis(self):
        with tempfile.TemporaryDirectory() as d:
            q = os.path.join(d, "q.pdf"); xref = _quelle(q)
            with mock.patch("export_abnahme.abnahme_pdf", side_effect=RuntimeError("pikepdf kaputt")):
                out, info = main._build_pdf_for_document(_unit(q, xref), os.path.join(d, "out"))
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(any("konnte nicht laufen" in w for w in info.get("warnings", [])), info)


if __name__ == "__main__":
    unittest.main()
