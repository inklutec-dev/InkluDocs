# -*- coding: utf-8 -*-
"""Dokument-Eigenschaften der exportierten PDF (Joerg Heine / Michael Karbe, 01.09.2026):
Creator = InkluDocs, Producer = InkluDocs + Werkzeug — an EINER Stelle (pdf_export) fuer alle
PDF-Ausgaenge; Author/Subject/Keywords/CreationDate des Autors bleiben.
Laeuft im Container: python3 -m unittest tests/test_dokumentinfo.py"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import fitz  # noqa: E402
import pdf_export  # noqa: E402
import pdfua_export  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "testformular_inkludocs.pdf")


class Dokumentinfo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pdf = os.path.join(self.tmp, "t.pdf")
        shutil.copy(FIX, self.pdf)

    def test_werte(self):
        w = pdf_export.dokumentinfo_werte("pdfix")
        self.assertTrue(w["creator"].startswith("InkluDocs"))
        self.assertIn("PDFix", w["producer"])
        self.assertIn("PyMuPDF", pdf_export.dokumentinfo_werte("fitz")["producer"])
        self.assertIn("LibreOffice", pdf_export.dokumentinfo_werte("libreoffice")["producer"])
        self.assertEqual(pdf_export.dokumentinfo_werte(None)["producer"], pdf_export.PDF_PRODUCER_BASIS)

    def test_setze_dokumentinfo_schreibt_creator_producer_und_laesst_rest(self):
        d = fitz.open(self.pdf); alt = dict(d.metadata or {}); d.close()
        pdf_export.setze_dokumentinfo(self.pdf, "pdfix")
        d = fitz.open(self.pdf); m = d.metadata; d.close()
        self.assertEqual(m["creator"], pdf_export.PDF_CREATOR)
        self.assertIn("PDFix", m["producer"])
        for k in ("author", "subject", "keywords", "creationDate"):
            self.assertEqual((m.get(k) or ""), (alt.get(k) or ""), k)

    def test_finalize_setzt_dokumentinfo(self):
        info = pdf_export.finalize_export_pdf(self.pdf, title="Test", fallback_title=None, verfahren="fitz")
        self.assertIn("PyMuPDF", info["producer"])
        d = fitz.open(self.pdf); m = d.metadata; d.close()
        self.assertEqual(m["creator"], pdf_export.PDF_CREATOR)
        self.assertEqual(m["title"], "Test")

    def test_pdfua_bytes(self):
        raw = open(FIX, "rb").read()
        neu = pdfua_export.dokumentinfo_setzen(raw)
        self.assertNotEqual(raw, neu)
        p = os.path.join(self.tmp, "ua.pdf"); open(p, "wb").write(neu)
        d = fitz.open(p); m = d.metadata; d.close()
        self.assertEqual(m["creator"], pdf_export.PDF_CREATOR)
        self.assertIn("LibreOffice", m["producer"])


if __name__ == "__main__":
    unittest.main()
