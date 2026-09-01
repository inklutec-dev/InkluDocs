# -*- coding: utf-8 -*-
"""Dokument-Eigenschaften der exportierten PDF: Creator = inkludocs.de, Producer = InkluDocs —
an EINER Stelle (pdf_export) fuer alle PDF-Ausgaenge, immer nur unser Produktname (nie ein
Werkzeug); Author/Subject/Keywords/CreationDate des Autors bleiben.
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
FREMDNAMEN = ("pdfix", "verapdf", "libreoffice", "pymupdf", "mupdf", "fitz", "pikepdf")


class Dokumentinfo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pdf = os.path.join(self.tmp, "t.pdf")
        shutil.copy(FIX, self.pdf)

    def test_werte(self):
        self.assertEqual(pdf_export.PDF_CREATOR, "inkludocs.de")
        self.assertEqual(pdf_export.PDF_PRODUCER, "InkluDocs")
        for weg in ("pdfix", "fitz", "libreoffice", None, "irgendwas"):
            w = pdf_export.dokumentinfo_werte(weg)
            self.assertEqual(w, {"creator": "inkludocs.de", "producer": "InkluDocs"}, weg)

    def test_nur_produktname_kein_werkzeug(self):
        """Regel: In den Eigenschaften steht nur unser Produktname, nie ein Werkzeug."""
        for weg in ("pdfix", "fitz", "libreoffice", None):
            w = pdf_export.dokumentinfo_werte(weg)
            for wert in (w["creator"], w["producer"]):
                for fremd in FREMDNAMEN:
                    self.assertNotIn(fremd, wert.lower(), (weg, wert))

    def test_setze_dokumentinfo_schreibt_creator_producer_und_laesst_rest(self):
        d = fitz.open(self.pdf); alt = dict(d.metadata or {}); d.close()
        pdf_export.setze_dokumentinfo(self.pdf, "pdfix")
        d = fitz.open(self.pdf); m = d.metadata; d.close()
        self.assertEqual(m["creator"], "inkludocs.de")
        self.assertEqual(m["producer"], "InkluDocs")
        for k in ("author", "subject", "keywords", "creationDate"):
            self.assertEqual((m.get(k) or ""), (alt.get(k) or ""), k)

    def test_finalize_setzt_dokumentinfo(self):
        info = pdf_export.finalize_export_pdf(self.pdf, title="Test", fallback_title=None, verfahren="fitz")
        self.assertEqual(info["producer"], "InkluDocs")
        d = fitz.open(self.pdf); m = d.metadata; d.close()
        self.assertEqual(m["creator"], "inkludocs.de")
        self.assertEqual(m["producer"], "InkluDocs")
        self.assertEqual(m["title"], "Test")

    def _mit_xmp(self, path):
        """Fixture mit einem XMP-Paket versehen, wie es eine Quell-PDF mitbringt (Element- und
        Attribut-Schreibweise, dazu ein Feld, das erhalten bleiben muss)."""
        xmp = ('<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
               '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
               '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:pdf="http://ns.adobe.com/pdf/1.3/"'
               ' xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/" pdf:Producer="Trial version of PDFix SDK | www.pdfix.net">'
               '<xmp:CreatorTool>PDFL 18.0.5</xmp:CreatorTool><pdfuaid:part>1</pdfuaid:part>'
               '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')
        d = fitz.open(path); d.set_xml_metadata(xmp); d.save(path + ".x"); d.close()
        shutil.move(path + ".x", path)

    def _xmp_werte(self, path):
        d = fitz.open(path); x = d.get_xml_metadata() or ""; d.close()
        return x

    def test_xmp_wird_mitgezogen(self):
        """Quell-XMP nennt ein Werkzeug — nach dem Export steht auch dort nur unser Name;
        andere XMP-Felder (pdfuaid:part) bleiben."""
        self._mit_xmp(self.pdf)
        self.assertIn("PDFix", self._xmp_werte(self.pdf))
        pdf_export.setze_dokumentinfo(self.pdf, "pdfix")
        x = self._xmp_werte(self.pdf)
        self.assertIn("<xmp:CreatorTool>inkludocs.de</xmp:CreatorTool>", x)
        self.assertIn('pdf:Producer="InkluDocs"', x)
        self.assertIn("<pdfuaid:part>1</pdfuaid:part>", x)
        for fremd in FREMDNAMEN:
            self.assertNotIn(fremd, x.lower())
        d = fitz.open(self.pdf); m = d.metadata; d.close()
        self.assertEqual((m["creator"], m["producer"]), ("inkludocs.de", "InkluDocs"))

    def test_finalize_zieht_xmp_mit(self):
        self._mit_xmp(self.pdf)
        pdf_export.finalize_export_pdf(self.pdf, title="Test", fallback_title=None, verfahren="pdfix")
        x = self._xmp_werte(self.pdf)
        self.assertIn("inkludocs.de", x)
        for fremd in FREMDNAMEN:
            self.assertNotIn(fremd, x.lower())

    def test_xmp_ohne_eintrag_wird_nichts_erfunden(self):
        werte = pdf_export.dokumentinfo_werte()
        x = '<x:xmpmeta><rdf:RDF><rdf:Description><dc:title>T</dc:title></rdf:Description></rdf:RDF></x:xmpmeta>'
        self.assertEqual(pdf_export.xmp_dokumentinfo(x, werte), x)
        self.assertEqual(pdf_export.xmp_dokumentinfo("", werte), "")

    def test_pdfua_bytes(self):
        with open(FIX, "rb") as fh:
            raw = fh.read()
        neu = pdfua_export.dokumentinfo_setzen(raw)
        self.assertNotEqual(raw, neu)
        p = os.path.join(self.tmp, "ua.pdf"); open(p, "wb").write(neu)
        d = fitz.open(p); m = d.metadata; d.close()
        self.assertEqual(m["creator"], "inkludocs.de")
        self.assertEqual(m["producer"], "InkluDocs")
        # XMP muss dieselben Werte tragen (Viewer zeigen sonst den XMP-Producer)
        import pikepdf, io
        with pikepdf.open(io.BytesIO(neu)) as pdf, pdf.open_metadata() as xmp:
            self.assertEqual(str(xmp.get("xmp:CreatorTool", "")), "inkludocs.de")
            self.assertEqual(str(xmp.get("pdf:Producer", "")), "InkluDocs")


if __name__ == "__main__":
    unittest.main()
