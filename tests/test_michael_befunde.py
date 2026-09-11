"""Michael Karbes Testrunde 11.09.2026 (Omnidocs-Dokument): vier Nachbesserungen.
    docker exec -w /app inkludocs-staging python3 /app/tests/test_michael_befunde.py -v
(1) leeres Alt-Feld loest keine „nicht mehr gefunden“-Warnung aus, (2) Titel aus dem Inhalt
statt Dateiname, (3) Link-Annotationen bekommen /Contents, (4) Klartext kennt 7.4.2/7.18.x,
(5) Sprach-Abgleich im Pruefbericht."""
import os
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import docx_export  # noqa: E402
import docx_hoerprobe  # noqa: E402
import docx_processor  # noqa: E402
import pdfua_export  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "word_einfach.docx")
W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
EN = ("This is a longer paragraph in English with the usual words that you find in a text about accessible documents "
      "and the way screen readers read them to the user of the document for the purpose of this test.")


def _docx(absaetze, lang="da-DK", titel_stil=None):
    body = "".join(absaetze) + "<w:sectPr/>"
    doc = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W}><w:body>{body}</w:body></w:document>'
    styles = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {W}>'
              f'<w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="22"/><w:lang w:val="{lang}"/></w:rPr></w:rPrDefault></w:docDefaults>'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
              '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style>'
              '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style></w:styles>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    fd, pfad = tempfile.mkstemp(suffix=".docx"); os.close(fd)
    with zipfile.ZipFile(pfad, "w") as z:
        z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc); z.writestr("word/styles.xml", styles)
    return pfad


def p(text, style=None):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


class TestMichael(unittest.TestCase):
    def test_leeres_feld_keine_warnung(self):
        if not os.path.isfile(FIX):
            self.skipTest("Fixture fehlt")
        erg = docx_processor.analysiere_docx(FIX)
        bilder = erg.bilder if hasattr(erg, "bilder") else erg[0]
        anker = [(b if isinstance(b, dict) else b.__dict__)["anker"] for b in bilder]
        self.assertTrue(anker)
        out = tempfile.mktemp(suffix=".docx")
        r = docx_export.write_alt_texts_to_docx(FIX, out, {anker[0]: ""})
        self.assertEqual(r.geleert, 1)
        self.assertEqual(r.nicht_gefunden, [])
        self.assertEqual(r.warnungen, [])
        os.remove(out)

    def test_titel_aus_inhalt(self):
        d = _docx([p("PDF vs PDF/UA", "Title"), p("Understanding accessibility", "Heading1"), p(EN)])
        self.assertEqual(pdfua_export.titel_aus_inhalt(d), "PDF vs PDF/UA")
        d2 = _docx([p(EN), p("Understanding accessibility", "Heading1"), p(EN)])
        self.assertEqual(pdfua_export.titel_aus_inhalt(d2), "Understanding accessibility")
        d3 = _docx([p(EN)])
        self.assertEqual(pdfua_export.titel_aus_inhalt(d3), "")

    def test_links_beschriften(self):
        import fitz
        doc = fitz.open(); seite = doc.new_page()
        seite.insert_text((72, 100), "Beispiel-Link (fiktiv)")
        seite.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 90, 200, 110), "uri": "https://www.example.com/test"})
        seite.insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(72, 120, 200, 140), "page": 0})
        pdf = doc.tobytes()
        neu, n = pdfua_export.links_beschriften(pdf)
        self.assertEqual(n, 2)
        import pikepdf, io
        pp = pikepdf.open(io.BytesIO(neu))
        inhalte = sorted(str(a.get("/Contents")) for a in pp.pages[0].obj["/Annots"])
        self.assertEqual(inhalte, ["Verweis im Dokument", "https://www.example.com/test"])
        neu2, n2 = pdfua_export.links_beschriften(neu)
        self.assertEqual(n2, 0)

    def test_klartext_neue_regeln(self):
        k = pdfua_export.klartext({"compliant": False, "rules": [
            {"clause": "7.4.2", "test": 1, "description": "For documents that are not strongly structured", "failed": 4},
            {"clause": "7.18.1", "test": 2, "description": "An annotation shall have Contents", "failed": 32},
            {"clause": "7.18.5", "test": 2, "description": "Links shall contain", "failed": 32},
        ]})
        texte = {pt["bereich"]: pt["text"] for pt in k["punkte"]}
        self.assertIn("Ebenen sind nicht durchgehend", texte["Überschriften"])
        self.assertIn("Ein Link hat keine Beschreibung", texte["Formularfelder und Verknüpfungen"])
        self.assertNotIn("technischer Prüfpunkt", texte["Überschriften"])
        self.assertIn("Prüfbericht des Word-Dokuments", texte["Tabellen"])

    def test_sprache_abgleich_und_titel_ersatz(self):
        d = _docx([p(EN), p(EN), p(EN), p(EN)], lang="da-DK")
        a = docx_hoerprobe.analysiere(d, titel_ersatz="Erste Überschrift (fiktiv)")
        texte = [b["text"] for b in a["pruefbericht"]]
        self.assertTrue(any("da-DK" in t and "Englisch" in t for t in texte), texte)
        self.assertTrue(any("ersatzweise „Erste Überschrift (fiktiv)“" in t for t in texte), texte)
        d2 = _docx([p(EN), p(EN), p(EN)], lang="en-GB")
        a2 = docx_hoerprobe.analysiere(d2)
        self.assertTrue(any("Dokumentsprache ist gesetzt (en-GB)" in t for t in [b["text"] for b in a2["pruefbericht"]]))


if __name__ == "__main__":
    unittest.main()
