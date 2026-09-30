"""Word-Ansichten „Dokument“ und „Barrierefreiheitsprüfung“ (30.09.2026): Dokumentinfos, Fingerabdruck und Zuordnung
der letzten barrierefreien PDF aus der Ablage (backend/docx_ansicht.py).
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_docx_ansicht.py -v
Braucht tests/fixtures/*.docx (word_tests.sh kopiert sie nach /app/tests/fixtures).
"""
import json
import os
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import docx_ansicht  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
CT = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="xml" ContentType="application/xml"/>'
      '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
APP = ('<?xml version="1.0" encoding="UTF-8"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
       '<Application>{app}</Application><Pages>{pages}</Pages></Properties>')


def _docx(pfad, body, app=None, pages=1):
    with zipfile.ZipFile(pfad, "w") as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("word/document.xml", f'<?xml version="1.0" encoding="UTF-8"?><w:document {W}><w:body>{body}</w:body></w:document>')
        if app is not None:
            z.writestr("docProps/app.xml", APP.format(app=app, pages=pages))


def _p(text, rsid=True, extra=""):
    attr = ' w:rsidR="00AB12CD" w:rsidRDefault="00AB12CD"' if rsid else ""
    return f"<w:p{attr}>{extra}<w:r><w:t>{text}</w:t></w:r></w:p>"


class Dokumentinfo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def pfad(self, name):
        return os.path.join(self.tmp.name, name)

    def test_seitenmarken_von_word(self):
        # Zwei <w:lastRenderedPageBreak/> = drei Seiten, egal was app.xml sagt
        marke = "<w:r><w:lastRenderedPageBreak/></w:r>"
        _docx(self.pfad("a.docx"), _p("Eins") + _p("Zwei", extra=marke) + _p("Drei", extra=marke), app="Microsoft Office Word", pages=7)
        self.assertEqual(docx_ansicht.dokumentinfo(self.pfad("a.docx"))["seiten"], 3)

    def test_app_xml_nur_mit_bearbeitungsspuren(self):
        _docx(self.pfad("word.docx"), _p("Eins") + _p("Zwei"), app="Microsoft Office Word", pages=1)
        info = docx_ansicht.dokumentinfo(self.pfad("word.docx"))
        self.assertEqual(info["seiten"], 1)
        self.assertEqual(info["anwendung"], "Microsoft Office Word")
        # Programmbibliothek: keine rsid-Kennungen -> Vorlagenwert „1“ nicht glauben
        _docx(self.pfad("lib.docx"), _p("Eins", rsid=False) + _p("Zwei", rsid=False), app="Microsoft Macintosh Word", pages=1)
        self.assertIsNone(docx_ansicht.dokumentinfo(self.pfad("lib.docx"))["seiten"])

    def test_app_xml_kleiner_als_umbrueche(self):
        umbruch = '<w:r><w:br w:type="page"/></w:r>'
        _docx(self.pfad("b.docx"), _p("Eins") + _p("Zwei", extra=umbruch) + _p("Drei", extra=umbruch), app="Microsoft Office Word", pages=1)
        self.assertIsNone(docx_ansicht.dokumentinfo(self.pfad("b.docx"))["seiten"])

    def test_ohne_app_xml(self):
        _docx(self.pfad("c.docx"), _p("Eins"))
        info = docx_ansicht.dokumentinfo(self.pfad("c.docx"))
        self.assertTrue(info["lesbar"])
        self.assertEqual(info["anwendung"], "")
        self.assertIsNone(info["seiten"])

    def test_kaputt_und_fehlend(self):
        with open(self.pfad("k.docx"), "wb") as f:
            f.write(b"PK\x03\x04kaputt")
        for p in (self.pfad("k.docx"), self.pfad("gibtsnicht.docx"), ""):
            info = docx_ansicht.dokumentinfo(p)
            self.assertFalse(info["lesbar"])
            self.assertIsNone(info["seiten"])

    def test_zip_bombe_wird_nicht_gelesen(self):
        # Pfad-Trick im Archiv: _pruefe_zip lehnt ab, dokumentinfo liefert „nicht lesbar“ statt Ausnahme
        with zipfile.ZipFile(self.pfad("z.docx"), "w") as z:
            z.writestr("[Content_Types].xml", CT)
            z.writestr("word/document.xml", f'<w:document {W}><w:body>{_p("x")}</w:body></w:document>')
            z.writestr("../boese.txt", "x")
        self.assertFalse(docx_ansicht.dokumentinfo(self.pfad("z.docx"))["lesbar"])

    @unittest.skipUnless(os.path.isfile(os.path.join(FIX, "testdokument_inkludocs.docx")), "Fixture fehlt")
    def test_fixture_testdokument(self):
        info = docx_ansicht.dokumentinfo(os.path.join(FIX, "testdokument_inkludocs.docx"))
        self.assertTrue(info["lesbar"])
        self.assertTrue(info["titel"].startswith("Testdokument"), info["titel"])
        self.assertGreaterEqual(info["ueberschriften"], 1)
        self.assertEqual(info["tabellen"], 1)
        self.assertIsNone(info["seiten"])   # python-docx: keine Seitenmarken, keine rsid -> unbekannt

    @unittest.skipUnless(os.path.isfile(os.path.join(FIX, "word_vml_bild.docx")), "Fixture fehlt")
    def test_fixture_aus_word(self):
        info = docx_ansicht.dokumentinfo(os.path.join(FIX, "word_vml_bild.docx"))
        self.assertEqual(info["seiten"], 1)
        self.assertIn("Word", info["anwendung"])


class Fingerabdruck(unittest.TestCase):
    def test_stabil_und_empfindlich(self):
        a = docx_ansicht.fingerabdruck({"x|1": "Hund", "x|2": "dekorativ"}, "Bericht", "de")
        self.assertEqual(a, docx_ansicht.fingerabdruck({"x|2": "dekorativ", "x|1": "Hund"}, "Bericht", "de"))
        self.assertNotEqual(a, docx_ansicht.fingerabdruck({"x|1": "Katze", "x|2": "dekorativ"}, "Bericht", "de"))
        self.assertNotEqual(a, docx_ansicht.fingerabdruck({"x|1": "Hund", "x|2": "dekorativ"}, "Bericht 2", "de"))
        self.assertNotEqual(a, docx_ansicht.fingerabdruck({"x|1": "Hund", "x|2": "dekorativ"}, "Bericht", "en"))
        # leer ("" = Alt-Text entfernen) und None (Fehlertext, Bild bleibt unberuehrt) sind verschieden
        self.assertNotEqual(docx_ansicht.fingerabdruck({"x|1": ""}, "", "de"), docx_ansicht.fingerabdruck({"x|1": None}, "", "de"))


def _zeile(id_, doc_id, bericht):
    return {"id": id_, "document_id": doc_id, "created_at": "2026-09-30 08:15:00", "bericht": json.dumps(bericht)}


def _eintrag(dokument, doc_id=None, bestanden=True, fp=None, punkte=None):
    e = {"dokument": dokument, "pruefung": {"bestanden": bestanden, "regeln_fehlgeschlagen": len(punkte or []),
                                            "punkte": punkte or [{"bereich": "Struktur", "status": "ok", "text": "gut"}]}}
    if doc_id is not None:
        e["document_id"] = doc_id
    if fp is not None:
        e["fingerabdruck"] = fp
    return e


class PdfuaZuordnung(unittest.TestCase):
    DOCS = [{"id": 11, "doc_index": 1, "display_name": "Bericht"}, {"id": 12, "doc_index": 2, "display_name": "Anhang"}]

    def label(self, d):
        return d["display_name"]

    def test_einzeldokument_und_neuester_gewinnt(self):
        rows = [_zeile(5, 11, [_eintrag("Bericht", 11, bestanden=False, fp="neu",
                                        punkte=[{"bereich": "Bilder", "status": "befund", "text": "Alt fehlt",
                                                 "einzeln": [{"text": "Alt fehlt", "regeln": ["7.3-1"]}]}])]),
                _zeile(4, 11, [_eintrag("Bericht", 11, fp="alt")])]
        out = docx_ansicht.pdfua_je_dokument(rows, self.DOCS, self.label, {11: "neu"})
        self.assertEqual(out[11]["ausgabe_id"], 5)
        self.assertFalse(out[11]["bestanden"])
        self.assertTrue(out[11]["aktuell"])
        self.assertEqual(out[11]["punkte"][0]["einzeln"][0]["regeln"], ["7.3-1"])
        self.assertNotIn(12, out)

    def test_zip_fuer_ganzes_projekt(self):
        rows = [_zeile(7, None, [_eintrag("Bericht", 11, fp="a"), _eintrag("Anhang", 12, fp="b")])]
        out = docx_ansicht.pdfua_je_dokument(rows, self.DOCS, self.label, {11: "a", 12: "anders"})
        self.assertTrue(out[11]["aktuell"])
        self.assertFalse(out[12]["aktuell"])
        self.assertEqual(out[12]["ausgabe_id"], 7)

    def test_alter_zip_nur_ueber_eindeutigen_namen(self):
        rows = [_zeile(3, None, [_eintrag("Bericht"), _eintrag("Anhang")])]
        out = docx_ansicht.pdfua_je_dokument(rows, self.DOCS, self.label, {})
        self.assertEqual(out[11]["ausgabe_id"], 3)
        self.assertIsNone(out[11]["aktuell"])   # Eintrag ohne Fingerabdruck: Stand unbekannt
        doppelt = [_zeile(2, None, [_eintrag("Bericht"), _eintrag("Bericht")])]
        self.assertEqual(docx_ansicht.pdfua_je_dokument(doppelt, self.DOCS, self.label, {}), {})

    def test_nur_befunde_und_kaputte_zeilen(self):
        rows = [{"id": 9, "document_id": 11, "created_at": "", "bericht": "{kaputt"},
                _zeile(8, 11, [_eintrag("Bericht", 11, punkte=[{"bereich": "A", "status": "ok", "text": "x"},
                                                               {"bereich": "B", "status": "befund", "text": "y"}])])]
        out = docx_ansicht.pdfua_je_dokument(rows, self.DOCS, self.label, {})
        self.assertEqual(out[11]["ausgabe_id"], 8)
        self.assertEqual([p["bereich"] for p in out[11]["punkte"]], ["B"])
        self.assertEqual(out[11]["punkte"][0]["einzeln"], [])

    def test_fremdes_dokument_nie(self):
        rows = [_zeile(1, 99, [_eintrag("Bericht", 99)])]
        self.assertEqual(docx_ansicht.pdfua_je_dokument(rows, self.DOCS, self.label, {}), {})


if __name__ == "__main__":
    unittest.main()
