"""Unit-Tests fuer das Word-Werkzeug (docx_processor + docx_export), 26.08.2026.

Aufruf im Container:
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_docx_roundtrip.py -v
Lokal (Entwicklung):
    cd backend && python3 -m unittest ../tests/test_docx_roundtrip.py -v

Fixture: tests/fixtures/testdokument_inkludocs.docx (FIKTIV, erzeugt von
tests/fixtures/make_testdoc.py mit python-docx; im Container nicht neu
erzeugbar, weil python-docx dort bewusst nicht installiert ist).
Abgedeckt: 8 Bildvorkommen (Kopfzeile, Bildunterschrift, vorhandener Alt-Text,
dekorativ, Tabelle, frei positioniert, JPEG, Wiederholung desselben Mediums),
Zurueckschreiben, Byte-Identitaet unberuehrter Teile, Idempotenz und die
Abwehr von .doc / .docm / Nicht-Word-Zip / Zip-Bombe.

Echte Word-Dokumente (Haertetest 27.08.2026, Dateien aus dem LibreOffice-
Testkorpus sw/qa/extras/ooxmlexport/data, MPL-2.0, von Microsoft Word erzeugt):
  word_textfeld_bild.docx   Bild IN einem Textfeld; Word schreibt das Textfeld
                            doppelt (mc:Choice + mc:Fallback), gleiche docPr-id
  word_vml_bild.docx        altes VML-Bild (w:pict/v:shape/v:imagedata), frei
  word_vml_kopfzeile.docx   VML-Bild in der Kopfzeile
  word_einfach.docx         ein Inline-Bild; dient dem Test doppelter docPr-ids
"""
import io
import os
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import docx_processor as dp   # noqa: E402
import docx_export as de      # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "testdokument_inkludocs.docx")
FIX_TEXTFELD = os.path.join(HERE, "fixtures", "word_textfeld_bild.docx")
FIX_VML = os.path.join(HERE, "fixtures", "word_vml_bild.docx")
FIX_VML_KOPF = os.path.join(HERE, "fixtures", "word_vml_kopfzeile.docx")
FIX_EINFACH = os.path.join(HERE, "fixtures", "word_einfach.docx")


@unittest.skipUnless(os.path.isfile(FIXTURE), "Fixture testdokument_inkludocs.docx fehlt")
class TestDocxLesen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="docx-test-")
        cls.erg = dp.analysiere_docx(FIXTURE, cls.tmp)
        cls.by_order = {b.order: b for b in cls.erg.bilder}

    def test_acht_bilder_gefunden(self):
        self.assertEqual(len(self.erg.bilder), 8)
        self.assertEqual(self.erg.uebersprungen, [])

    def test_kopfzeile_zuerst(self):
        b = self.by_order[1]
        self.assertEqual(b.ort, "Kopfzeile")
        self.assertTrue(b.part.startswith("word/header"))
        self.assertIn("Position: Kopfzeile", b.context)

    def test_bildunterschrift_und_querverweise(self):
        b = self.by_order[2]
        self.assertTrue(b.caption.startswith("Abbildung 1:"))
        self.assertIn("Bildunterschrift: Abbildung 1", b.context)
        self.assertIn("Abschnitt: ", b.context)
        self.assertIn("1 Einleitung", b.context)

    def test_vorhandener_alt_text_und_titel(self):
        b = self.by_order[3]
        self.assertIn("alter Alt-Text", b.original_alt)
        self.assertEqual(b.original_title, "Musterfirma-Logo")

    def test_dekorativ_erkannt(self):
        self.assertTrue(self.by_order[4].decorative)
        self.assertFalse(self.by_order[3].decorative)

    def test_tabelle_mit_kopf(self):
        b = self.by_order[5]
        self.assertEqual(b.ort, "Tabelle")
        self.assertIn("Tabellenkopf: Produkt (fiktiv) | Foto", b.context)
        self.assertIn("Tabellenzeile: Beispielgerät Modell X", b.context)

    def test_frei_positioniert(self):
        self.assertTrue(self.by_order[6].anchored)
        self.assertFalse(self.by_order[2].anchored)

    def test_jpeg_bleibt_jpeg(self):
        b = self.by_order[7]
        self.assertEqual(b.media_ext, ".jpeg")
        self.assertTrue(b.image_path.endswith(".jpg"))
        self.assertTrue(os.path.isfile(b.image_path))

    def test_wiederholung_teilt_datei(self):
        a, a2 = self.by_order[2], self.by_order[8]
        self.assertEqual(a.hash, a2.hash)
        self.assertEqual(a.image_path, a2.image_path)
        self.assertNotEqual(a.anker, a2.anker)

    def test_abschnitte_steigen(self):
        self.assertLessEqual(self.by_order[2].abschnitt, self.by_order[5].abschnitt)
        self.assertLess(self.by_order[5].abschnitt, self.by_order[7].abschnitt)

    def test_extract_images_schnittstelle(self):
        rows = dp.extract_images_from_docx(FIXTURE, tempfile.mkdtemp(), project_id=1)
        self.assertEqual(len(rows), 8)
        for r in rows:
            for key in ("page_number", "image_index", "image_path", "width", "height",
                        "context_text", "original_alt", "decorative_hint", "docx_anker"):
                self.assertIn(key, r)
        self.assertTrue(any(r["decorative_hint"] for r in rows))


@unittest.skipUnless(os.path.isfile(FIXTURE), "Fixture testdokument_inkludocs.docx fehlt")
class TestDocxSchreiben(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="docx-export-")
        self.erg = dp.analysiere_docx(FIXTURE, self.tmp)
        self.alts = {}
        for b in self.erg.bilder:
            if b.ort == "Kopfzeile":
                self.alts[b.anker] = "dekorativ"
            elif b.order == 7:
                self.alts[b.anker] = ""            # bewusst unberuehrt
            else:
                self.alts[b.anker] = f"Alt-Text {b.order} (fiktiv)"
        self.alts["word/document.xml|999"] = "Geist"
        self.out = os.path.join(self.tmp, "export.docx")
        self.r = de.write_alt_texts_to_docx(FIXTURE, self.out, self.alts)

    def test_zaehler(self):
        self.assertEqual(self.r.geschrieben, 7)
        self.assertEqual(self.r.dekorativ, 1)
        self.assertEqual(self.r.uebersprungen, 1)
        self.assertEqual(self.r.nicht_gefunden, ["word/document.xml|999"])
        self.assertEqual(len(self.r.warnungen), 1)

    def test_unberuehrte_teile_byteidentisch(self):
        diff = de.pruefe_unveraendert(FIXTURE, self.out, {"word/document.xml", "word/header1.xml"})
        self.assertEqual(diff, [])

    def test_rueck_lesen(self):
        neu = {b.anker: b for b in dp.analysiere_docx(self.out, tempfile.mkdtemp()).bilder}
        for b in self.erg.bilder:
            n = neu[b.anker]
            soll = self.alts[b.anker]
            if soll == "dekorativ":
                self.assertTrue(n.decorative); self.assertEqual(n.original_alt, "")
            elif soll == "":
                self.assertEqual(n.original_alt, b.original_alt)
            else:
                self.assertEqual(n.original_alt, soll); self.assertFalse(n.decorative)

    def test_idempotent(self):
        out2 = os.path.join(self.tmp, "export2.docx")
        de.write_alt_texts_to_docx(self.out, out2, self.alts)
        with zipfile.ZipFile(self.out) as a, zipfile.ZipFile(out2) as b:
            self.assertEqual(a.read("word/document.xml"), b.read("word/document.xml"))

    def test_word_xml_deklaration_bleibt(self):
        with zipfile.ZipFile(self.out) as z:
            self.assertTrue(z.read("word/document.xml").startswith(b"<?xml"))


@unittest.skipUnless(all(os.path.isfile(f) for f in (FIX_TEXTFELD, FIX_VML, FIX_VML_KOPF, FIX_EINFACH)),
                     "Word-Fixtures des Haertetests fehlen")
class TestEchteWordDokumente(unittest.TestCase):
    """Befunde des Haertetests vom 27.08.2026 mit von Word erzeugten Dateien."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="docx-word-")

    def _export(self, quelle, alts):
        out = os.path.join(self.tmp, "export.docx")
        r = de.write_alt_texts_to_docx(quelle, out, alts)
        return out, r

    # --- Bild im Textfeld, Choice/Fallback-Duplikat
    def test_textfeld_ist_kein_bild(self):
        erg = dp.analysiere_docx(FIX_TEXTFELD, self.tmp)
        self.assertEqual(len(erg.bilder), 1)                 # nicht 3 wie vor dem Fix
        b = erg.bilder[0]
        self.assertEqual(b.anker, "word/document.xml|3")
        self.assertEqual(b.ort, "Textfeld")
        self.assertFalse(b.vml)
        self.assertEqual([u["anker"] for u in erg.uebersprungen], ["word/document.xml|1"])  # das Textfeld

    def test_fallback_bekommt_denselben_alt_text(self):
        out, r = self._export(FIX_TEXTFELD, {"word/document.xml|3": "Bild im Textfeld"})
        self.assertEqual((r.geschrieben, r.nicht_gefunden), (1, []))
        with zipfile.ZipFile(out) as z:
            xml = z.read("word/document.xml").decode("utf-8")
        self.assertEqual(xml.count('descr="Bild im Textfeld"'), 2)   # Choice UND Fallback
        self.assertEqual(de.pruefe_unveraendert(FIX_TEXTFELD, out, {"word/document.xml"}), [])
        neu = dp.analysiere_docx(out, tempfile.mkdtemp()).bilder
        self.assertEqual([(b.anker, b.original_alt) for b in neu], [("word/document.xml|3", "Bild im Textfeld")])

    # --- altes VML-Bild
    def test_vml_bild_lesen(self):
        erg = dp.analysiere_docx(FIX_VML, self.tmp)
        self.assertEqual(len(erg.bilder), 1)
        b = erg.bilder[0]
        self.assertTrue(b.vml)
        self.assertEqual(b.anker, "word/document.xml|v:_x0000_s1029")
        self.assertTrue(b.anchored)
        self.assertEqual(b.media_ext, ".jpeg")
        self.assertTrue(b.image_path.endswith(".jpg"))
        self.assertGreater(b.width, 0)

    def test_vml_bild_schreiben_und_dekorativ(self):
        anker = "word/document.xml|v:_x0000_s1029"
        out, r = self._export(FIX_VML, {anker: "Foto vom Ausflug (fiktiv)"})
        self.assertEqual((r.geschrieben, r.dekorativ, r.nicht_gefunden), (1, 0, []))
        self.assertEqual(de.pruefe_unveraendert(FIX_VML, out, {"word/document.xml"}), [])
        neu = dp.analysiere_docx(out, tempfile.mkdtemp()).bilder[0]
        self.assertEqual(neu.original_alt, "Foto vom Ausflug (fiktiv)")
        with zipfile.ZipFile(out) as z:
            self.assertIn(b'alt="Foto vom Ausflug (fiktiv)"', z.read("word/document.xml"))
        # dekorativ: VML kennt kein Kennzeichen -> leerer Alt-Text
        out2 = os.path.join(self.tmp, "deko.docx")
        r2 = de.write_alt_texts_to_docx(out, out2, {anker: "dekorativ"})
        self.assertEqual((r2.geschrieben, r2.dekorativ), (1, 1))
        self.assertEqual(dp.analysiere_docx(out2, tempfile.mkdtemp()).bilder[0].original_alt, "")

    def test_vml_bild_in_kopfzeile(self):
        erg = dp.analysiere_docx(FIX_VML_KOPF, self.tmp)
        self.assertEqual([(b.anker, b.ort, b.vml) for b in erg.bilder],
                         [("word/header1.xml|v:_x0000_i1025", "Kopfzeile", True)])
        out, r = self._export(FIX_VML_KOPF, {"word/header1.xml|v:_x0000_i1025": "Firmenlogo (fiktiv)"})
        self.assertEqual(r.geschrieben, 1)
        self.assertEqual(de.pruefe_unveraendert(FIX_VML_KOPF, out, {"word/header1.xml"}), [])
        self.assertEqual(dp.analysiere_docx(out, tempfile.mkdtemp()).bilder[0].original_alt, "Firmenlogo (fiktiv)")

    # --- doppelte docPr-ids ausserhalb von mc:Fallback (kaputtes Dokument)
    def test_doppelte_docpr_ids_bekommen_eigene_anker(self):
        kaputt = os.path.join(self.tmp, "doppelt.docx")
        with zipfile.ZipFile(FIX_EINFACH) as zin, zipfile.ZipFile(kaputt, "w") as zout:
            for zi in zin.infolist():
                daten = zin.read(zi.filename)
                if zi.filename == "word/document.xml":
                    xml = daten.decode("utf-8")
                    a = xml.index("<w:drawing"); e = xml.index("</w:drawing>") + len("</w:drawing>")
                    xml = xml[:e] + xml[a:e] + xml[e:]        # Bild-Element direkt verdoppeln
                    daten = xml.encode("utf-8")
                zout.writestr(zi, daten)
        erg = dp.analysiere_docx(kaputt, self.tmp)
        anker = [b.anker for b in erg.bilder]
        self.assertEqual(len(anker), 2)
        self.assertTrue(anker[1].endswith("#2"), anker)
        self.assertEqual(anker[1].split("#")[0], anker[0])
        out = os.path.join(self.tmp, "export.docx")
        r = de.write_alt_texts_to_docx(kaputt, out, {anker[0]: "Erstes", anker[1]: "Zweites"})
        self.assertEqual((r.geschrieben, r.nicht_gefunden), (2, []))
        neu = dp.analysiere_docx(out, tempfile.mkdtemp()).bilder
        self.assertEqual([b.original_alt for b in neu], ["Erstes", "Zweites"])

    def test_unbekannter_anker_wird_gemeldet(self):
        out, r = self._export(FIX_VML, {"word/document.xml|v:gibtsnicht": "x", "word/document.xml|abc": "y"})
        self.assertEqual(sorted(r.nicht_gefunden), ["word/document.xml|abc", "word/document.xml|v:gibtsnicht"])
        self.assertEqual(r.geschrieben, 0)


class TestAbwehr(unittest.TestCase):
    def _zip(self, members: dict[str, bytes]) -> str:
        p = os.path.join(tempfile.mkdtemp(), "x.docx")
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            for n, b in members.items():
                z.writestr(n, b)
        return p

    def test_kein_zip(self):
        p = os.path.join(tempfile.mkdtemp(), "alt.doc")
        with open(p, "wb") as f:
            f.write(b"\xd0\xcf\x11\xe0" + b"\x00" * 64)
        with self.assertRaises(dp.DocxFehler):
            dp.validiere_docx(p)

    def test_kein_word(self):
        with self.assertRaises(dp.DocxFehler):
            dp.validiere_docx(self._zip({"hallo.txt": b"x"}))

    def test_docm_inhaltstyp(self):
        ct = ('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Override PartName="/word/document.xml" ContentType="%s"/></Types>' % dp.CT_DOCM).encode()
        p = self._zip({"[Content_Types].xml": ct, "word/document.xml": b"<x/>"})
        with self.assertRaisesRegex(dp.DocxFehler, "Makros"):
            dp.validiere_docx(p)

    def test_pfad_traversal(self):
        ct = ('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Override PartName="/word/document.xml" ContentType="%s"/></Types>' % dp.CT_DOCX).encode()
        p = self._zip({"[Content_Types].xml": ct, "word/document.xml": b"<x/>", "../boese.txt": b"x"})
        with self.assertRaisesRegex(dp.DocxFehler, "ungültige Pfade"):
            dp.validiere_docx(p)

    def test_zip_bombe(self):
        ct = ('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Override PartName="/word/document.xml" ContentType="%s"/></Types>' % dp.CT_DOCX).encode()
        bombe = b"\x00" * (5 * 1024 * 1024)          # 5 MB Nullen komprimieren ~1:1000
        p = self._zip({"[Content_Types].xml": ct, "word/document.xml": b"<x/>", "word/media/b.bin": bombe})
        with self.assertRaisesRegex(dp.DocxFehler, "komprimiert"):
            dp.validiere_docx(p)

    def test_xxe_wird_nicht_aufgeloest(self):
        ct = ('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Override PartName="/word/document.xml" ContentType="%s"/></Types>' % dp.CT_DOCX).encode()
        doc = (b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
               b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
               b'<w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body></w:document>')
        p = self._zip({"[Content_Types].xml": ct, "word/document.xml": doc})
        erg = dp.analysiere_docx(p)
        self.assertEqual(erg.bilder, [])
        self.assertLess(erg.volltext_zeichen, 50)   # Entity blieb ungeloest, kein Dateiinhalt


if __name__ == "__main__":
    unittest.main(verbosity=2)
