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
        self.assertEqual(bereiche, ["Struktur und Lesereihenfolge", "Sprache und Aufbau von Tabellen und Listen",
                                    "Bilder und Grafiken", "Überschriften", "Tabellen"])
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
        self.assertEqual(d["Sprache und Aufbau von Tabellen und Listen"]["status"], "ok")   # Kernbereich ohne Befund bleibt sichtbar
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


class SeitenUndDoppelungenTest(unittest.TestCase):
    """23.09.2026 (Michaels Punkt 7): Seitenangabe aus veraPDF, jede Aussage nur einmal, 7.18.5-1 verständlich."""

    def test_seiten_und_keine_doppelungen(self):
        k = pdfua_export.klartext({"compliant": False, "rules": [
            {"clause": "7.18.5", "test": 1, "description": "Links shall be tagged", "failed": 1, "pages": [15]},
            {"clause": "7.18.5", "test": 2, "description": "Links shall contain", "failed": 1, "pages": [15]},
            {"clause": "7.18.1", "test": 2, "description": "annot", "failed": 1, "pages": [15]},
            {"clause": "7.3", "test": 1, "description": "x", "failed": 2, "pages": [3, 10]}]})
        d = {p["bereich"]: p for p in k["punkte"]}
        t = d["Anmerkungen, Formularfelder und Links"]["text"]
        self.assertEqual(t.count("Ein Link hat keine Beschreibung"), 1)
        self.assertIn("Eine Anmerkung (zum Beispiel ein Kommentar oder ein Link) hat keine Beschreibung.", t)   # 7.18.1-2
        self.assertIn("nicht als Link getaggt", t)
        self.assertTrue(t.endswith("(Seite 15)"), t)
        self.assertEqual(d["Anmerkungen, Formularfelder und Links"]["seiten"], [15])
        self.assertTrue(d["Bilder und Grafiken"]["text"].endswith("(Seiten 3, 10)"))

    def test_ohne_seiten_kein_zusatz(self):
        k = pdfua_export.klartext({"compliant": False, "rules": [{"clause": "7.3", "test": 1, "description": "x", "failed": 1}]})
        d = {p["bereich"]: p for p in k["punkte"]}
        self.assertEqual(d["Bilder und Grafiken"]["text"], "Ein Bild hat keinen Alternativtext.")
        self.assertEqual(d["Bilder und Grafiken"]["seiten"], [])

    def test_einzeln_je_regel_ohne_einleitung(self):
        # Michael Karbe, Feedback 24.09.2026, Punkte 11/12: je Pruefpunkt eine Zeile, Originaltext ohne Einleitung
        k = pdfua_export.klartext({"compliant": False, "rules": [
            {"clause": "7.21.4.1", "test": 1, "description": "The font programs for all fonts used for rendering within a conforming file shall be embedded within that file, as defined in ISO 32000-1:2008, 9.9", "failed": 2, "pages": [1, 2]},
            {"clause": "7.21.3.2", "test": 1, "description": "Glyph widths must be consistent.", "failed": 1, "pages": [2]},
            {"clause": "7.18.5", "test": 1, "description": "Links shall be tagged", "failed": 1, "pages": [15]},
            {"clause": "7.18.5", "test": 2, "description": "Links shall contain", "failed": 1, "pages": [15]},
            {"clause": "7.18.1", "test": 2, "description": "annot", "failed": 1, "pages": [15]}]})
        d = {p["bereich"]: p for p in k["punkte"]}
        schrift = d["Schriften"]["einzeln"]
        self.assertEqual(len(schrift), 2)
        self.assertEqual(schrift[0]["text"], "Eine Schrift ist nicht eingebettet. (2-mal) (Seiten 1, 2)")   # 7.21.4.1-1 seit 30.09.
        self.assertEqual(schrift[1]["text"], "Glyph widths must be consistent (Seite 2)")                     # unbekannt: Originaltext
        self.assertTrue(all("technischer Prüfpunkt" not in e["text"] for e in schrift))
        links = d["Anmerkungen, Formularfelder und Links"]["einzeln"]
        self.assertEqual(len(links), 3)   # je Regel ein eigener Satz (7.18.1-2 gilt fuer alle Anmerkungen)
        self.assertEqual(sum("Ein Link hat keine Beschreibung" in e["text"] for e in links), 1)



class RegelwerkTest(unittest.TestCase):
    """Audit 30.09.2026 (MITTEL 2): jeder Klartext-Satz passt zu SEINER Regel im Regelwerk PDFUA-1.xml der veraPDF-Version des
    Konverters (Auszug tests/fixtures/verapdf_pdfua1_regeln.json), keine toten Eintraege, keine toten Bereiche, und
    zusammengelegte Regeln werden nicht addiert. Neue Eintraege brauchen einen Beleg hier (englische Stichworte aus der
    Regelbeschreibung, deutsche aus unserem Satz)."""

    BELEGE = {
        ("5", 1): (["PDF/UA", "Identification"], ["PDF/UA"]),
        ("6.2", 1): (["MarkInfo", "Marked"], ["getaggte PDF"]),
        ("7.1", 1): (["marked as Artifact", "inside tagged content"], ["Artefakt", "innerhalb von ausgezeichnetem"]),
        ("7.1", 2): (["Tagged content", "inside content marked as Artifact"], ["Ausgezeichneter Inhalt", "Artefakt"]),
        ("7.1", 3): (["marked as Artifact or tagged as real content"], ["weder als Struktur noch als Schmuck"]),
        ("7.1", 8): (["Metadata key", "metadata stream"], ["Metadaten"]),
        ("7.1", 9): (["dc:title"], ["Dokumenttitel", "Metadaten"]),
        ("7.1", 10): (["DisplayDocTitle"], ["Titel statt des Dateinamens"]),
        ("7.1", 11): (["StructTreeRoot"], ["Strukturbaum"]),
        ("7.2", 3): (["Table element may contain only TR"], ["Tabelle"]),
        ("7.2", 34): (["Natural language for text in page content"], ["Sprache"]),
        ("7.3", 1): (["Figure", "alternative"], ["Bild", "Alternativtext"]),
        ("7.4.2", 1): (["heading"], ["Überschriften"]),
        ("7.5", 1): (["TH", "Scope", "Headers"], ["Kopfzelle", "Zeile oder die Spalte"]),
        ("7.5", 2): (["undefined Header"], ["Kopfzellen, die es nicht gibt"]),
        ("7.16", 1): (["encrypted", "10th bit"], ["verschlüsselt"]),
        ("7.18.1", 2): (["An annotation (except Widget", "Contents"], ["Anmerkung", "Beschreibung"]),
        ("7.18.1", 3): (["form field", "TU key"], ["Formularfeld", "Beschreibung"]),
        ("7.18.5", 1): (["Links shall be tagged"], ["Link", "getaggt"]),
        ("7.18.5", 2): (["Links shall contain an alternate description"], ["Link", "Beschreibung"]),
        ("7.21.4.1", 1): (["font", "embedded"], ["Schrift", "eingebettet"]),
    }
    BEREICH_BELEGE = {
        "5": (["PDF/UA"], ["PDF/UA"]), "7.1": (["Artifact"], ["Struktur"]), "7.2": (["Table", "Natural language"], ["Sprache", "Tabellen"]),
        "7.3": (["Figure"], ["Bild"]), "7.4": (["heading"], ["Überschriften"]), "7.5": (["Scope"], ["Tabelle"]),
        "7.7": (["mathematical"], ["Formeln"]), "7.9": (["Note"], ["Fußnoten"]), "7.10": (["optional content"], ["Ein- und ausblendbare"]),
        "7.11": (["embedded file", "UF keys"], ["Dateinamen"]), "7.16": (["encrypted"], ["Verschlüsselung"]),
        "7.18": (["annotation", "form field", "Links"], ["Anmerkungen", "Formularfelder", "Links"]),
        "7.20": (["XObject"], ["Inhaltsblöcke"]), "7.21": (["font"], ["Schriften"]),
    }

    @classmethod
    def setUpClass(cls):
        import json
        with open(os.path.join(HERE, "fixtures", "verapdf_pdfua1_regeln.json"), encoding="utf-8") as f:
            cls.regelwerk = {(r["clause"], r["test"]): r for r in json.load(f)["regeln"]}

    def test_jeder_satz_passt_zur_regel(self):
        kt = pdfua_export.REGELN_KLARTEXT
        self.assertEqual(set(kt), set(self.BELEGE), "Jeder Klartext-Eintrag braucht einen Beleg im Test (und umgekehrt)")
        for regel, satz in kt.items():
            with self.subTest(regel=regel):
                self.assertIn(regel, self.regelwerk, f"{regel}: diese Regel gibt es in veraPDF nicht (toter Eintrag)")
                r = self.regelwerk[regel]
                englisch = (r["description"] + " " + r["message"]).lower()
                en, de = self.BELEGE[regel]
                for w in en:
                    self.assertIn(w.lower(), englisch, f"{regel}: Beleg „{w}“ steht nicht in der Regelbeschreibung")
                for w in de:
                    self.assertIn(w, satz, f"{regel}: Satz „{satz}“ nennt „{w}“ nicht")

    def test_bereiche_haben_regeln_und_passen(self):
        praefixe = [b[0] for b in pdfua_export.BEREICHE]
        self.assertEqual(set(praefixe), set(self.BEREICH_BELEGE))
        for praefix, name, gut in pdfua_export.BEREICHE:
            with self.subTest(bereich=praefix):
                regeln = [r for (c, _t), r in self.regelwerk.items() if c == praefix or c.startswith(praefix + ".")]
                self.assertTrue(regeln, f"Bereich {praefix}: keine Regel in veraPDF (toter Bereich)")
                englisch = " ".join(r["description"] + " " + r["message"] for r in regeln).lower()
                en, de = self.BEREICH_BELEGE[praefix]
                for w in en:
                    self.assertIn(w.lower(), englisch)
                for w in de:
                    self.assertIn(w, name + " " + gut)
        self.assertNotIn("7.17", praefixe)
        self.assertNotIn("7.6", praefixe)

    def test_audit_faelle(self):
        """Die im Audit belegten Falschzuordnungen: jetzt der richtige Satz."""
        kt = pdfua_export.REGELN_KLARTEXT
        self.assertIn("Tabelle", kt[("7.2", 3)])
        self.assertNotIn("Sprache", kt[("7.2", 3)])
        self.assertIn("Sprache", kt[("7.2", 34)])
        self.assertIn("verschlüsselt", kt[("7.16", 1)])
        self.assertNotIn("Link hat keine", kt[("7.18.1", 2)])
        k = pdfua_export.klartext({"compliant": False, "rules": [{"clause": "7.16", "test": 1, "description": "x", "failed": 1},
                                                                 {"clause": "7.20", "test": 2, "description": "Form XObject", "failed": 1},
                                                                 {"clause": "6.2", "test": 1, "description": "MarkInfo", "failed": 1}]})
        d = {p["bereich"]: p for p in k["punkte"]}
        self.assertEqual(d["Sicherheit"]["status"], "befund")
        self.assertEqual(d["Eingebettete Inhaltsblöcke (XObjects)"]["status"], "befund")
        self.assertIn("nicht als getaggte PDF gekennzeichnet", d["Struktur und Lesereihenfolge"]["text"])   # 6.2-1 unter Struktur

    def test_lange_saetze_nicht_still_gekuerzt(self):
        """Pruefung 3 (N2): ein langer englischer Regelsatz steht in der Zeile ganz; der Absatz kuerzt nur mit „…“."""
        lang = "Embedded fonts shall define all glyphs referenced for rendering " * 8
        k = pdfua_export.klartext({"compliant": False, "rules": [{"clause": "7.21.4.2", "test": 2, "description": lang, "failed": 1}]})
        p = [x for x in k["punkte"] if x["bereich"] == "Schriften"][0]
        self.assertEqual(p["einzeln"][0]["satz"], " ".join(lang.split()).rstrip("."))
        self.assertIn(" …", p["text"])

    def test_zusammengelegt_nicht_addiert(self):
        """16 Links verletzen zwei Regeln: nie „32-mal“ — je Regel ein Satz mit 16, gleiche Saetze mit der groessten Zahl."""
        k = pdfua_export.klartext({"compliant": False, "rules": [
            {"clause": "7.18.1", "test": 2, "description": "annot", "failed": 16, "pages": [1]},
            {"clause": "7.18.5", "test": 2, "description": "links", "failed": 16, "pages": [1]}]})
        einzeln = [p for p in k["punkte"] if p["bereich"] == "Anmerkungen, Formularfelder und Links"][0]["einzeln"]
        self.assertEqual(len(einzeln), 2)
        self.assertTrue(all(e["mal"] == "(16-mal)" for e in einzeln), einzeln)
        self.assertFalse(any("32" in e["text"] for e in einzeln))
        gleich = pdfua_export._einzeln([{"clause": "9.9", "test": 1, "failed": 16}, {"clause": "9.9", "test": 2, "failed": 7}],
                                       {("9.9", 1): "Gleicher Satz.", ("9.9", 2): "Gleicher Satz."}, lambda s: s)
        self.assertEqual((len(gleich), gleich[0]["mal"], gleich[0]["regeln"]), (1, "(16-mal)", ["9.9-1", "9.9-2"]))


if __name__ == "__main__":
    unittest.main()
