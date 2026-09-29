"""PDF ohne Tags: seit 29.09.2026 Download unveraendert bzw. nur mit Quickinfos (Michael Karbe, Feedback 20260928 - 2,
Punkt 5); bis dahin 422 (15.09.2026)."""
import os, sys, tempfile, unittest
from unittest import mock
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

    def test_plan_je_dokument(self):
        """getaggt -> PDF-Staffel; ungetaggt ohne Quickinfos -> unveraendert, 0 Credits."""
        units = [{"doc": {"id": None, "getaggt": 1, "original_filename": "mit.pdf"}, "images": [{}, {}, {}]},
                 {"doc": {"id": None, "getaggt": 0, "original_filename": "ohne.pdf"}, "images": [{}]}]
        with mock.patch.object(main.billing, "preis_pruefung", side_effect=lambda uid, preis: {"preis": preis, "verfuegbar": None, "erlaubt": True, "fehlend": 0}):
            plan = main._pdf_export_plan(0, units)
        self.assertEqual([u["doc"]["original_filename"] for u in plan["getaggt"]], ["mit.pdf"])
        self.assertEqual([u["doc"]["original_filename"] for u in plan["unveraendert"]], ["ohne.pdf"])
        self.assertEqual(plan["mit_qi"], [])
        self.assertEqual(plan["preis_pdf"], main.billing.export_preis(3, "pdf"))   # nur die Bilder der getaggten
        self.assertEqual(plan["preis_qi"], 0)
        self.assertEqual(plan["pruefung"]["preis"], plan["preis_pdf"])
        with mock.patch.object(main.billing, "preis_pruefung", side_effect=lambda uid, preis: {"preis": preis, "verfuegbar": None, "erlaubt": True, "fehlend": 0}):
            nur_ohne = main._pdf_export_plan(0, units[1:])
        self.assertEqual(nur_ohne["pruefung"]["preis"], 0)   # nichts zu schreiben, nichts zu bezahlen

    def test_ohne_tags_kommt_byte_gleich_zurueck(self):
        with tempfile.TemporaryDirectory() as d:
            quelle = os.path.join(d, "u.pdf"); _pdf(quelle, False)
            unit = {"doc": {"id": None, "getaggt": 0, "original_filename": "u.pdf", "original_path": quelle}, "images": []}
            pfad, info = main._pdf_ohne_tags(unit, os.path.join(d, "aus"), None, False)
            self.assertEqual(open(pfad, "rb").read(), open(quelle, "rb").read())
            self.assertTrue(info["unveraendert"])
            self.assertEqual(info["quickinfos"]["geschrieben"], 0)
            self.assertNotEqual(os.path.abspath(pfad), os.path.abspath(quelle))   # Original bleibt unberuehrt

    def test_ohne_tags_fehlende_datei(self):
        unit = {"doc": {"id": None, "getaggt": 0, "original_filename": "x.pdf", "original_path": "/gibt/es/nicht.pdf"}, "images": []}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(HTTPException) as cm:
                main._pdf_ohne_tags(unit, d, None, False)
        self.assertEqual(cm.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
