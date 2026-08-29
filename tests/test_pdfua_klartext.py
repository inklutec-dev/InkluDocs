"""Klartext aus dem veraPDF-Bericht (29.08.2026).
    docker exec -w /app inkludocs-staging python3 /app/tests/test_pdfua_klartext.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import pdfua_export  # noqa: E402


class TestKlartext(unittest.TestCase):
    def test_bestanden(self):
        k = pdfua_export.klartext({"compliant": True, "profile": "PDF/UA-1 validation profile", "rules": []})
        self.assertTrue(k["bestanden"])
        self.assertEqual(k["regeln_fehlgeschlagen"], 0)
        bereiche = [p["bereich"] for p in k["punkte"]]
        self.assertEqual(bereiche, ["Struktur und Lesereihenfolge", "Text und Sprache", "Bilder und Grafiken",
                                    "Überschriften", "Tabellen"])
        self.assertTrue(all(p["status"] == "ok" for p in k["punkte"]))
        self.assertIn("bestanden", pdfua_export.zusammenfassung(k))

    def test_befunde(self):
        bericht = {"compliant": False, "rules": [
            {"clause": "7.1", "test": 3, "description": "Content shall be marked", "failed": 17},
            {"clause": "7.1", "test": 9, "description": "dc:title", "failed": 1},
            {"clause": "7.3", "test": 1, "description": "Figure alt", "failed": 2},
            {"clause": "7.21", "test": 1, "description": "font embedded", "failed": 1},
            {"clause": "7.99", "test": 4, "description": "Irgendwas Exotisches", "failed": 1},
        ]}
        k = pdfua_export.klartext(bericht)
        self.assertFalse(k["bestanden"])
        self.assertEqual(k["regeln_fehlgeschlagen"], 5)
        d = {p["bereich"]: p for p in k["punkte"]}
        self.assertEqual(d["Struktur und Lesereihenfolge"]["status"], "befund")
        self.assertIn("Schmuck", d["Struktur und Lesereihenfolge"]["text"])
        self.assertIn("Dokumenttitel fehlt in den Metadaten", d["Struktur und Lesereihenfolge"]["text"])
        self.assertEqual(d["Bilder und Grafiken"]["status"], "befund")
        self.assertIn("(2-mal)", d["Bilder und Grafiken"]["text"])
        self.assertEqual(d["Schriften"]["status"], "befund")       # 7.21 laeuft unter Schriften
        self.assertEqual(d["Text und Sprache"]["status"], "ok")   # Kernbereich ohne Befund bleibt sichtbar
        self.assertEqual(d["Überschriften"]["status"], "ok")
        self.assertIn("Weitere Prüfpunkte", d)
        self.assertIn("Irgendwas Exotisches", d["Weitere Prüfpunkte"]["text"])
        self.assertIn("Bereiche mit Hinweisen", pdfua_export.zusammenfassung(k))

    def test_leerer_bericht(self):
        k = pdfua_export.klartext({})
        self.assertFalse(k["bestanden"])
        self.assertEqual(k["regeln_fehlgeschlagen"], 0)

    def test_titel_setzen(self):
        import tempfile, zipfile
        from lxml import etree
        core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title></dc:title></cp:coreProperties>')
        fd, pfad = tempfile.mkstemp(suffix=".docx"); os.close(fd)
        with zipfile.ZipFile(pfad, "w") as z:
            z.writestr("docProps/core.xml", core)
            z.writestr("word/document.xml", "<w/>")
        self.assertTrue(pdfua_export.dokumenttitel_setzen(pfad, "Mein Titel", "de"))
        with zipfile.ZipFile(pfad) as z:
            x = etree.fromstring(z.read("docProps/core.xml"))
            self.assertEqual(x.find("dc:title", pdfua_export._NS).text, "Mein Titel")
            self.assertEqual(x.find("dc:language", pdfua_export._NS).text, "de")
            self.assertEqual(z.read("word/document.xml"), b"<w/>")
        self.assertFalse(pdfua_export.dokumenttitel_setzen(pfad, "Anderer", "de"))   # nichts mehr zu tun
        os.unlink(pfad)


if __name__ == "__main__":
    unittest.main()
