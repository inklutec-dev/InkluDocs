"""Struktur-Lektor, Lesestufe (11.09.2026): Befunde an einem handgebauten Word-Dokument.
    docker exec -w /app inkludocs-staging python3 /app/tests/test_docx_struktur.py -v
Das Testdokument wird ohne python-docx als Zip+XML gebaut (klar fiktiv)."""
import os
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import docx_struktur  # noqa: E402

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def p(text, style=None, bold=False, sz=None, num=False, link=None, caps=False):
    ppr = ""
    if style or num:
        ppr = "<w:pPr>" + (f'<w:pStyle w:val="{style}"/>' if style else "") + ('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>' if num else "") + "</w:pPr>"
    rpr = ""
    if bold or sz or caps:
        rpr = "<w:rPr>" + ("<w:b/>" if bold else "") + ("<w:caps/>" if caps else "") + (f'<w:sz w:val="{sz*2}"/>' if sz else "") + "</w:rPr>"
    run = f"<w:r>{rpr}<w:t xml:space=\"preserve\">{text}</w:t></w:r>"
    if link is not None:
        run = f'<w:hyperlink r:id="rId9">{run}</w:hyperlink>'
    return f"<w:p>{ppr}{run}</w:p>"


LANG = "Dies ist ein längerer Absatz mit ausreichend vielen Wörtern, damit die Heuristik ihn als Fließtext erkennt und nicht als Überschrift."


def dokument(absaetze, tabellen_xml=""):
    body = "".join(absaetze) + tabellen_xml + '<w:sectPr/>'
    doc = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W}><w:body>{body}</w:body></w:document>'
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              f'<w:styles {W}><w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
              '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
              '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/></w:rPr></w:style>'
              '</w:styles>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    fd, pfad = tempfile.mkstemp(suffix=".docx"); os.close(fd)
    with zipfile.ZipFile(pfad, "w") as z:
        z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc); z.writestr("word/styles.xml", styles)
    return pfad


class TestStruktur(unittest.TestCase):
    def setUp(self):
        tabelle = ('<w:tbl><w:tblPr/><w:tr><w:tc><w:p><w:r><w:t>Aussen</w:t></w:r></w:p>'
                   '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Innen</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:tc></w:tr></w:tbl>')
        self.pfad = dokument([
            p("Musterbericht der Musterfirma (fiktiv)", style="Heading1"),
            p(LANG),
            p("Einleitung", bold=True),                    # 3: fett + kurz vor langem Absatz -> Ueberschrift ohne Vorlage
            p(LANG),
            p("Weitere Hinweise", sz=16),                  # 5: groesser -> Ueberschrift ohne Vorlage
            p(LANG),
            p("- erster Punkt der getippten Liste"),       # 7-9: getippte Liste
            p("- zweiter Punkt"),
            p("- dritter Punkt"),
            p(""), p(""), p(""),                            # 10-12: Leerabsaetze
            p("ACHTUNG WICHTIGER HINWEIS ZUM VERFAHREN"),   # 13: Grossbuchstaben
            p("https://www.example.com/antrag", link=True),  # 14: Linktext = URL
            p("Echte Liste", style="Heading2"),
            p("Punkt A", num=True), p("Punkt B", num=True),
        ], tabelle)

    def tearDown(self):
        os.remove(self.pfad)

    def test_befunde(self):
        st = docx_struktur.analysiere_struktur(self.pfad)
        arten = {b["art"]: b for b in st["befunde"]}
        self.assertIn("ueberschrift_ohne_vorlage", arten)
        kandidaten = [b for b in st["befunde"] if b["art"] == "ueberschrift_ohne_vorlage"]
        self.assertEqual(sorted(b["absatz"] for b in kandidaten), [3, 5])
        self.assertTrue(all(b["sicherheit"] == "mittel" and b["vorschlag_ebene"] == 2 for b in kandidaten))
        self.assertEqual(arten["getippte_liste"]["absatz"], 7)
        self.assertEqual(arten["getippte_liste"]["anzahl"], 3)
        self.assertEqual(arten["getippte_liste"]["sicherheit"], "hoch")
        self.assertIn("leerabsaetze", arten)
        self.assertEqual(arten["grossbuchstaben"]["absatz"], 13)
        self.assertEqual(arten["linktext"]["absatz"], 14)
        self.assertIn("tabelle_verschachtelt", arten)
        self.assertNotIn("keine_ueberschriften", arten)
        self.assertEqual([g["ebene"] for g in st["gliederung"]], [1, 2])
        self.assertEqual(st["standard_schriftgroesse"], 11.0)
        self.assertEqual(st["zahlen"]["listenpunkte"], 2)

    def test_echte_liste_und_ueberschrift_kein_befund(self):
        st = docx_struktur.analysiere_struktur(dokument([p("Titel", style="Heading1"), p(LANG), p("Punkt", num=True), p("Punkt 2", num=True)]))
        self.assertEqual(st["befunde"], [])

    def test_fixture_laeuft(self):
        fx = os.path.join(HERE, "fixtures", "testdokument_inkludocs.docx")
        if not os.path.isfile(fx):
            self.skipTest("Fixture fehlt")
        st = docx_struktur.analysiere_struktur(fx)
        self.assertGreaterEqual(st["zahlen"]["ueberschriften"], 3)
        self.assertTrue(st["absaetze"])


if __name__ == "__main__":
    unittest.main()
