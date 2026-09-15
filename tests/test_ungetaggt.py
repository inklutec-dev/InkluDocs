"""PDF ohne Tags -> kein PDF-Download (Michael Karbe 15.09.2026)."""
import os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)
import fitz  # noqa: E402
from fastapi import HTTPException  # noqa: E402
import pdf_export  # noqa: E402
import main  # noqa: E402


def _pdf(pfad, tags):
    doc = fitz.open(); page = doc.new_page()
    if tags:
        root = doc.get_new_xref(); dok = doc.get_new_xref()
        doc.update_object(dok, f"<< /Type /StructElem /S /Document /P {root} 0 R /K [] >>")
        doc.update_object(root, f"<< /Type /StructTreeRoot /K [ {dok} 0 R ] >>")
        doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    doc.save(pfad); doc.close()


class Ungetaggt(unittest.TestCase):
    def test_pdf_hat_tags(self):
        with tempfile.TemporaryDirectory() as d:
            _pdf(os.path.join(d, "t.pdf"), True); _pdf(os.path.join(d, "u.pdf"), False)
            self.assertTrue(pdf_export.pdf_hat_tags(os.path.join(d, "t.pdf")))
            self.assertFalse(pdf_export.pdf_hat_tags(os.path.join(d, "u.pdf")))
            self.assertFalse(pdf_export.pdf_hat_tags(os.path.join(d, "fehlt.pdf")))
            open(os.path.join(d, "kaputt.pdf"), "wb").write(b"x"); self.assertFalse(pdf_export.pdf_hat_tags(os.path.join(d, "kaputt.pdf")))

    def test_export_lehnt_ungetaggt_ab(self):
        units = [{"doc": {"id": None, "getaggt": 0, "original_filename": "ohne.pdf", "display_name": "Ohne Tags"}, "images": []}]
        with self.assertRaises(HTTPException) as cm:
            main._ungetaggte_pruefen(units)
        self.assertEqual(cm.exception.status_code, 422)
        self.assertIn("keine Tags", cm.exception.detail)
        self.assertIn("Ohne Tags", cm.exception.detail)
        main._ungetaggte_pruefen([{"doc": {"id": None, "getaggt": 1, "original_filename": "mit.pdf"}, "images": []}])  # kein Fehler


if __name__ == "__main__":
    unittest.main()
