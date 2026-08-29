"""Hoerprobe + Pruefbericht aus einer synthetischen Word-Datei (29.08.2026).
    docker exec -w /app inkludocs-staging python3 /app/tests/test_docx_hoerprobe.py -v
"""
import os
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import docx_hoerprobe  # noqa: E402

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
WP = 'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
CT = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="xml" ContentType="application/xml"/>'
      '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')


def _p(text, style=None, num=False):
    ppr = ""
    if style or num:
        ppr = "<w:pPr>" + (f'<w:pStyle w:val="{style}"/>' if style else "") + ("<w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr>" if num else "") + "</w:pPr>"
    return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"


def _bild(descr, deko=False):
    ext = ('<a:extLst><a:ext uri="{C183D7F6-B498-43B3-948B-1728B52AA6E4}">'
           '<adec:decorative xmlns:adec="http://schemas.microsoft.com/office/drawing/2017/decorative" val="1"/></a:ext></a:extLst>') if deko else ""
    return (f'<w:p><w:r><w:drawing><wp:inline><wp:docPr id="1" name="Bild" descr="{descr}">{ext}</wp:docPr>'
            '<a:graphic><a:graphicData><a:blip/></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def _textfeld_mit_bild(descr):
    """Rahmen (docPr ohne eigenen blip) mit einem Bild im Textfeld — der Rahmen zaehlt nicht als Bild."""
    return ('<w:p><w:r><w:drawing><wp:anchor><wp:docPr id="7" name="Textfeld 1"/><a:graphic><a:graphicData>'
            '<w:txbxContent>' + _bild(descr) + '</w:txbxContent></a:graphicData></a:graphic></wp:anchor></w:drawing></w:r></w:p>')


def _tabelle(kopf):
    trpr = "<w:trPr><w:tblHeader/></w:trPr>" if kopf else ""
    return ("<w:tbl>" f"<w:tr>{trpr}<w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Wert</w:t></w:r></w:p></w:tc></w:tr>"
            "<w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>")


def _docx(body, titel="", lang=""):
    doc = f'<?xml version="1.0" encoding="UTF-8"?><w:document {W} {WP} {A}><w:body>{body}</w:body></w:document>'
    styles = (f'<?xml version="1.0" encoding="UTF-8"?><w:styles {W}>'
              + (f'<w:docDefaults><w:rPrDefault><w:rPr><w:lang w:val="{lang}"/></w:rPr></w:rPrDefault></w:docDefaults>' if lang else "")
              + '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
              + '<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/></w:style></w:styles>')
    core = ('<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{titel}</dc:title></cp:coreProperties>')
    fd, pfad = tempfile.mkstemp(suffix=".docx"); os.close(fd)
    with zipfile.ZipFile(pfad, "w") as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", styles)
        z.writestr("docProps/core.xml", core)
    return pfad


class TestHoerprobe(unittest.TestCase):
    def test_sauberes_dokument(self):
        pfad = _docx(_p("Kapitel", "Heading1") + _p("Ein Absatz.") + _bild("Ein Hund im Garten") + _tabelle(True),
                     titel="Testdok", lang="de-DE")
        a = docx_hoerprobe.analysiere(pfad)
        os.unlink(pfad)
        h = a["hoerprobe"]
        self.assertEqual(h[0], "Dokumenttitel: Testdok")
        self.assertEqual(h[1], "Sprache: de-DE")
        self.assertIn("Überschrift Ebene 1: Kapitel", h)
        self.assertIn("Absatz: Ein Absatz.", h)
        self.assertIn("Bild: Ein Hund im Garten", h)
        self.assertTrue(any(z.startswith("Tabelle mit 2 Zeilen und 2 Spalten, Kopfzeile: Name, Wert") for z in h), h)
        self.assertTrue(all(b["status"] == "ok" for b in a["pruefbericht"]), a["pruefbericht"])

    def test_befunde(self):
        pfad = _docx(_p("Erst Ebene 3", "Heading3") + _p("Kapitel", "Heading1") + _p("", "Heading1")
                     + _bild("") + _bild("", deko=True) + _tabelle(False) + _p("Punkt", num=True))
        a = docx_hoerprobe.analysiere(pfad)
        os.unlink(pfad)
        h = a["hoerprobe"]
        self.assertEqual(h[0], "Dokumenttitel: fehlt")
        self.assertEqual(h[1], "Sprache: nicht gesetzt")
        self.assertIn("Leere Überschrift (Ebene 1)", h)
        self.assertIn("Bild ohne Beschreibung — ein Screenreader sagt nur „Grafik“", h)
        self.assertIn("Schmuckbild (wird nicht vorgelesen)", h)
        self.assertIn("Listenpunkt: Punkt", h)
        self.assertTrue(any(z.endswith(", ohne Kopfzeile") for z in h), h)
        texte = " | ".join(b["text"] for b in a["pruefbericht"] if b["status"] == "befund")
        self.assertIn("Es fehlt ein Dokumenttitel", texte)
        self.assertIn("Dokumentsprache ist nicht gesetzt", texte)
        self.assertIn("erste Überschrift hat Ebene 3", texte)
        self.assertIn("Eine Überschrift ist leer", texte)
        self.assertIn("1 von 1 Tabellen", texte)
        self.assertIn("1 von 2 Bildern", texte)
        self.assertEqual(a["zahlen"]["dekorativ"], 1)

    def test_textfeld_rahmen_zaehlt_nicht(self):
        pfad = _docx(_textfeld_mit_bild("Bild im Kasten"), titel="T", lang="de")
        a = docx_hoerprobe.analysiere(pfad)
        os.unlink(pfad)
        self.assertEqual(a["zahlen"]["bilder"], 1)
        self.assertIn("Bild: Bild im Kasten", a["hoerprobe"])
        self.assertNotIn("Bild ohne Beschreibung — ein Screenreader sagt nur „Grafik“", a["hoerprobe"])

    def test_uebersetzung(self):
        pfad = _docx(_p("Kapitel", "Heading1"), titel="T", lang="en-GB")
        a = docx_hoerprobe.analysiere(pfad, lambda s: "X" + s)
        os.unlink(pfad)
        self.assertTrue(a["hoerprobe"][0].startswith("XDokumenttitel"))


if __name__ == "__main__":
    unittest.main()
