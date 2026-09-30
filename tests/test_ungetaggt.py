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
        """getaggt -> Export mit Alt-Texten; ungetaggt ohne Quickinfos -> unveraendert, 0 Credits. Seit 30.09.2026 (Michael
        Karbe, Feedback 202609230 - 1, Punkt 12) kosten nur BEARBEITETE Alt-Texte (hier 2 von 3), die Bilder der
        ungetaggten Datei nie (ausfuehrlich: test_herunterladen_genutzt.py)."""
        def bild(i, **kw):
            d = {"id": i, "image_index": i, "alt_text": "", "alt_text_edited": None, "original_alt": "", "image_type": "unknown", "status": "pending"}
            d.update(kw)
            return d
        units = [{"doc": {"id": None, "getaggt": 1, "original_filename": "mit.pdf", "extraction_method": "pdfix"},
                  "images": [bild(1, alt_text="KI-Text", status="done"), bild(2, alt_text_edited="Von Hand"), bild(3)]},
                 {"doc": {"id": None, "getaggt": 0, "original_filename": "ohne.pdf"}, "images": [bild(4, alt_text_edited="Von Hand")]}]
        with mock.patch.object(main.billing, "preis_pruefung", side_effect=lambda uid, preis: {"preis": preis, "verfuegbar": None, "erlaubt": True, "fehlend": 0}):
            plan = main._pdf_export_plan(0, units)
        self.assertEqual([u["doc"]["original_filename"] for u in plan["getaggt"]], ["mit.pdf"])
        self.assertEqual([u["doc"]["original_filename"] for u in plan["unveraendert"]], ["ohne.pdf"])
        self.assertEqual(plan["mit_qi"], [])
        self.assertEqual(plan["alt_bearbeitet"], 2)
        self.assertEqual(plan["preis_pdf"], main.billing.pdf_download_preis(2, 0)["preis"])   # nur bearbeitete der getaggten
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


class ExportTokensAufraeumen(unittest.TestCase):
    """Pruefung 3 (Entwicklung N2): Sofort-Downloads (bot_*, word_*, pdfua_*.json) in _export werden nach der Frist geloescht,
    frische bleiben, andere Dateien und dl_-Ordner fasst die Funktion nicht an."""

    def test_nur_alte_token_dateien(self):
        with tempfile.TemporaryDirectory() as d:
            exp = os.path.join(d, "7", "11", "_export")
            os.makedirs(os.path.join(exp, "dl_abc"))
            namen = ["bot_alt.csv", "word_alt.docx", "pdfua_alt.json", "bot_neu.zip", "pdfua_neu.json", "anderes_alt.json", "pdfua_alt.pdf"]
            for n in namen:
                open(os.path.join(exp, n), "w").write("x")
            alt = main.time.time() - main.EXPORT_TOKEN_AUFBEWAHREN - 60
            for n in namen:
                if "_alt" in n:
                    os.utime(os.path.join(exp, n), (alt, alt))
            n = main._export_tokens_aufraeumen(os.path.join(d, "*", "*", "_export"))
            self.assertEqual(n, 3)
            self.assertEqual(sorted(os.listdir(exp)), sorted(["dl_abc", "bot_neu.zip", "pdfua_neu.json", "anderes_alt.json", "pdfua_alt.pdf"]))

    def test_ablage_links_bleiben_und_frist_nur_ohne_ablage(self):
        """Pruefung 4 (Entwicklung 1): Metadateien, die in die Ablage zeigen, bleiben; gueltig_bis nur fuer Links ohne Ablage."""
        import json as _json
        with tempfile.TemporaryDirectory() as d:
            exp = os.path.join(d, "7", "11", "_export")
            abl = os.path.join(d, "7", "_ablage")
            os.makedirs(exp)
            os.makedirs(abl)
            open(os.path.join(abl, "pdfua_" + "a" * 24 + ".pdf"), "w").write("x")
            open(os.path.join(exp, "bot_" + "b" * 24 + ".csv"), "w").write("x")
            _json.dump({"pfad": os.path.join(abl, "pdfua_" + "a" * 24 + ".pdf")}, open(os.path.join(exp, "pdfua_" + "a" * 24 + ".json"), "w"))
            _json.dump({"pfad": os.path.join(exp, "bot_" + "b" * 24 + ".csv")}, open(os.path.join(exp, "pdfua_" + "b" * 24 + ".json"), "w"))
            with mock.patch.object(main, "RESULTS_DIR", d):
                self.assertIsNone(main.token_gueltig_bis(7, "/api/projects/11/export/pdfua/" + "a" * 24))
                bis = main.token_gueltig_bis(7, "/api/projects/11/export/pdfua/" + "b" * 24)
                self.assertRegex(bis or "", r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
                self.assertIsNone(main.token_gueltig_bis(7, "/api/ausgaben/5/datei"))
            alt = main.time.time() - main.EXPORT_TOKEN_AUFBEWAHREN - 60
            for n in os.listdir(exp):
                os.utime(os.path.join(exp, n), (alt, alt))
            self.assertEqual(main._export_tokens_aufraeumen(os.path.join(d, "*", "*", "_export")), 2)
            self.assertEqual(os.listdir(exp), ["pdfua_" + "a" * 24 + ".json"])
