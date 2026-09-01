# -*- coding: utf-8 -*-
"""Export = was der Kunde sieht (Michael Karbe/Steve 01.09.2026): ein LEERES Feld heisst
KEIN Alternativtext in der Datei — ein mitgebrachter Text wird entfernt.

- pdfix_roundtrip._build_csv_rows: "" -> Sentinel KEIN_ALT (Import entfernt den Alt-Eintrag),
  None -> keine Zeile (Figure unangetastet), "dekorativ" -> leerer Alt (Kennzeichen wie bisher).
- docx_export.write_alt_texts_to_docx: "" -> descr entfernt + Dekorativ-Kennzeichen aus,
  None -> Bild nicht angefasst.
Laeuft im Container: python3 -m unittest tests/test_export_leer.py
"""
import os
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import pdfix_roundtrip  # noqa: E402
import docx_export  # noqa: E402
from docx_processor import extract_images_from_docx  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "testdokument_inkludocs.docx")


class CsvZeilen(unittest.TestCase):
    def test_leer_wird_sentinel_none_bleibt_weg_dekorativ_bleibt_leer(self):
        rows = pdfix_roundtrip._build_csv_rows({1: "Text eins", 2: "", 3: None, 4: "dekorativ"}, "quelle")
        by = {r[0]: r[4] for r in rows}
        self.assertEqual(by[1], "Text eins")
        self.assertEqual(by[2], pdfix_roundtrip.KEIN_ALT)
        self.assertNotIn(3, by)
        self.assertEqual(by[4], "")


class WordLeer(unittest.TestCase):
    def test_leeres_feld_entfernt_descr(self):
        tmp = tempfile.mkdtemp()
        bilder = extract_images_from_docx(FIX, os.path.join(tmp, "img"), 1)
        imgs = bilder[0] if isinstance(bilder, tuple) else bilder
        mit_text = [b for b in imgs if (b.get("original_alt") or "") not in ("", "dekorativ")]
        self.assertTrue(mit_text, "Fixture braucht ein Bild mit mitgebrachtem Alt-Text")
        anker = mit_text[0]["docx_anker"]
        out = os.path.join(tmp, "leer.docx")
        erg = docx_export.write_alt_texts_to_docx(FIX, out, {anker: ""})
        self.assertEqual(erg.geleert, 1)
        part, kennung = anker.rsplit("|", 1)
        xml = zipfile.ZipFile(out).read(part).decode("utf8")
        import re
        m = re.search(r'<wp:docPr[^>]*\bid="%s"[^>]*>' % re.escape(kennung.split("#")[0]), xml)
        self.assertIsNotNone(m, "docPr nicht gefunden")
        self.assertNotIn('descr="', m.group(0), "descr muss entfernt sein: %s" % m.group(0)[:200])

    def test_none_laesst_bild_in_ruhe(self):
        tmp = tempfile.mkdtemp()
        bilder = extract_images_from_docx(FIX, os.path.join(tmp, "img"), 1)
        imgs = bilder[0] if isinstance(bilder, tuple) else bilder
        mit_text = [b for b in imgs if (b.get("original_alt") or "") not in ("", "dekorativ")]
        anker = mit_text[0]["docx_anker"]
        out = os.path.join(tmp, "unangetastet.docx")
        erg = docx_export.write_alt_texts_to_docx(FIX, out, {anker: None})
        self.assertEqual(erg.uebersprungen, 1)
        part, kennung = anker.rsplit("|", 1)
        xml = zipfile.ZipFile(out).read(part).decode("utf8")
        self.assertIn(mit_text[0]["original_alt"][:20], xml)


if __name__ == "__main__":
    unittest.main()
