"""Ersteller (Creator) je Konto + Schalter INKLUDOCS_PDF_METADATEN (Michael Karbe 14.09.2026).
    docker exec -w /app inkludocs-staging python3 -m unittest tests.test_dokumentinfo -v
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
import pdf_export  # noqa: E402
import pdfua_export  # noqa: E402


def _pdf(pfad):
    doc = fitz.open(); doc.new_page(); doc.set_metadata({"creator": "Word", "producer": "LibreOffice 26.2"}); doc.save(pfad); doc.close()


class TestCreator(unittest.TestCase):
    def test_werte_vorgabe_und_uebersteuerung(self):
        self.assertEqual(pdf_export.dokumentinfo_werte("fitz"), {"creator": pdf_export.PDF_CREATOR, "producer": pdf_export.PDF_PRODUCER})
        w = pdf_export.dokumentinfo_werte("fitz", "  Musterfirma\nGmbH  ")
        self.assertEqual(w["creator"], "Musterfirma GmbH")
        self.assertEqual(w["producer"], pdf_export.PDF_PRODUCER)
        self.assertEqual(pdf_export.dokumentinfo_werte("fitz", "   ")["creator"], pdf_export.PDF_CREATOR)
        self.assertEqual(len(pdf_export.creator_normieren("x" * 500)), pdf_export.CREATOR_MAXLAENGE)

    def test_creator_landet_in_datei(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.pdf"); _pdf(p)
            with mock.patch.object(pdf_export, "PDF_METADATEN_SETZEN", True):
                pdf_export.setze_dokumentinfo(p, "fitz", "Musterfirma GmbH")
            m = fitz.open(p).metadata
            self.assertEqual(m["creator"], "Musterfirma GmbH")
            self.assertEqual(m["producer"], pdf_export.PDF_PRODUCER)

    def test_schalter_aus_laesst_datei_unangetastet(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.pdf"); _pdf(p)
            vorher = open(p, "rb").read()
            with mock.patch.object(pdf_export, "PDF_METADATEN_SETZEN", False):
                w = pdf_export.setze_dokumentinfo(p, "fitz", "Musterfirma GmbH")
            self.assertTrue(w.get("unveraendert"))
            self.assertEqual(w["producer"], "LibreOffice 26.2")
            self.assertEqual(open(p, "rb").read(), vorher)
            with mock.patch.object(pdf_export, "PDF_METADATEN_SETZEN", False):
                self.assertEqual(pdfua_export.dokumentinfo_setzen(vorher, "Musterfirma GmbH"), vorher)

    def test_pdfua_weg_mit_creator(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.pdf"); _pdf(p)
            with mock.patch.object(pdf_export, "PDF_METADATEN_SETZEN", True):
                out = pdfua_export.dokumentinfo_setzen(open(p, "rb").read(), "Musterfirma GmbH")
            q = os.path.join(d, "b.pdf"); open(q, "wb").write(out)
            m = fitz.open(q).metadata
            self.assertEqual(m["creator"], "Musterfirma GmbH")
            self.assertEqual(m["producer"], pdf_export.PDF_PRODUCER)


if __name__ == "__main__":
    unittest.main()
