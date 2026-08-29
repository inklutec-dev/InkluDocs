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

    def test_uebersetzung(self):
        k = pdfua_export.klartext({"compliant": False, "rules": [{"clause": "7.3", "test": 1, "description": "x", "failed": 2}]},
                                  lambda s: "X" + s)
        d = {p["bereich"]: p for p in k["punkte"]}
        self.assertIn("XBilder und Grafiken", d)
        self.assertTrue(d["XBilder und Grafiken"]["text"].startswith("XEin Bild hat keinen Alternativtext."))
        self.assertTrue(pdfua_export.zusammenfassung(k, lambda s: "X" + s).startswith("X"))

    def test_alt_nachtragen_ohne_struktur(self):
        # Minimal-PDF ohne Strukturbaum: nichts anfassen, Bytes unveraendert
        import pikepdf, io
        pdf = pikepdf.new(); pdf.add_blank_page(); buf = io.BytesIO(); pdf.save(buf)
        raw = buf.getvalue()
        out, info = pdfua_export.alt_nachtragen(raw, ["Text"])
        self.assertEqual(out, raw)
        self.assertEqual(info["nachgetragen"], 0)
        self.assertFalse(info["zugeordnet"])

    def test_alt_nachtragen_mit_figures(self):
        import pikepdf, io
        pdf = pikepdf.new(); pdf.add_blank_page()
        f1 = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Figure")))
        f2 = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Figure"), Alt=pikepdf.String("schon da")))
        f3 = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Figure")))
        doc = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Document"), K=pikepdf.Array([f1, f2, f3])))
        pdf.Root.StructTreeRoot = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name("/StructTreeRoot"), K=pikepdf.Array([doc])))
        buf = io.BytesIO(); pdf.save(buf)
        out, info = pdfua_export.alt_nachtragen(buf.getvalue(), ["Erstes Bild", "egal", "dekorativ"])
        self.assertTrue(info["zugeordnet"]); self.assertEqual(info["figures"], 3)
        self.assertEqual(info["nachgetragen"], 1); self.assertEqual(info["dekorativ_offen"], 1)
        p2 = pikepdf.open(io.BytesIO(out))
        figs = pdfua_export._figures_in_reihenfolge(p2.Root.StructTreeRoot)
        self.assertEqual([str(f.get("/Alt") or "") for f in figs], ["Erstes Bild", "schon da", ""])
        # Zahl passt nicht -> nichts anfassen
        out2, info2 = pdfua_export.alt_nachtragen(buf.getvalue(), ["a", "b"])
        self.assertFalse(info2["zugeordnet"]); self.assertEqual(info2["nachgetragen"], 0)

    def test_alt_nachtragen_rahmen(self):
        # Textfeld mit Bild: LibreOffice = Figure (Rahmen) mit Figure (Bild) darin.
        import pikepdf, io
        pdf = pikepdf.new(); pdf.add_blank_page()
        innen = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Figure")))
        rahmen = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Figure"), K=pikepdf.Array([innen])))
        doc = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Document"), K=pikepdf.Array([rahmen])))
        pdf.Root.StructTreeRoot = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name("/StructTreeRoot"), K=pikepdf.Array([doc])))
        buf = io.BytesIO(); pdf.save(buf)
        out, info = pdfua_export.alt_nachtragen(buf.getvalue(), ["Bild im Kasten"])
        self.assertEqual((info["rahmen_umgewandelt"], info["figures"], info["nachgetragen"], info["zugeordnet"]), (1, 1, 1, True))
        p2 = pikepdf.open(io.BytesIO(out))
        d2 = p2.Root.StructTreeRoot.K[0]
        self.assertEqual(str(d2.K[0].S), "/Div")
        self.assertEqual(str(d2.K[0].K[0].Alt), "Bild im Kasten")

    def test_alt_nachtragen_libreoffice_rahmen(self):
        # Gemessenes LibreOffice-Muster (29.08.2026): leeres Figure (Rahmen), dann /Div > "/Frame contents" > Figure (Bild)
        import pikepdf, io
        pdf = pikepdf.new(); pdf.add_blank_page()
        rahmen = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Figure")))
        bild = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Figure")))
        fc = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Frame contents"), K=pikepdf.Array([bild])))
        div = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Div"), K=pikepdf.Array([fc])))
        std = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Standard"), K=pikepdf.Array([rahmen, div])))
        doc = pdf.make_indirect(pikepdf.Dictionary(S=pikepdf.Name("/Document"), K=pikepdf.Array([std])))
        pdf.Root.StructTreeRoot = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name("/StructTreeRoot"), K=pikepdf.Array([doc])))
        buf = io.BytesIO(); pdf.save(buf)
        out, info = pdfua_export.alt_nachtragen(buf.getvalue(), ["Skript oder Code"])
        self.assertEqual((info["rahmen_umgewandelt"], info["figures"], info["nachgetragen"], info["zugeordnet"]), (1, 1, 1, True))
        p2 = pikepdf.open(io.BytesIO(out))
        std2 = p2.Root.StructTreeRoot.K[0].K[0]
        self.assertEqual(str(std2.K[0].S), "/Div")
        self.assertEqual(str(std2.K[1].K[0].K[0].Alt), "Skript oder Code")

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
