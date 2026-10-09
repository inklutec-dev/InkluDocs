"""Folgen aus „Alt-Texte: Feld = Datei“ (Steve 09.10.2026):
  1. Herunterladen: Bewusst leere Felder und dekorative Bilder sperren den Download nicht mehr (Export-Abnahme), jede
     andere Verschlechterung schon; der Kunde bekommt einen kurzen Hinweis, welche Bilder keinen Alt-Text haben.
  3. Word „Als PDF“: In der PDF steht der Feldtext, nicht „Titel - Beschreibung“ (LibreOffice).
  4. Ersatzweg (fitz): ein leeres Feld nimmt das vorhandene /Alt der Figure weg — bewusst geleert im Export, jedes leere
     Feld nach dem Tagging; ein nie angefasstes leeres Feld laesst den Text der Kundendatei stehen (fitz liest ihn nicht).
(Punkt 2, „Neu generieren“ mit Bildtyp dekorativ, steht in test_dekorativ_neu_generieren.py — eigene Datenbank.)
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_feld_datei_folgen.py -v
"""
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="feld-folgen-")
os.environ["INKLUDOCS_DB"] = os.path.join(_TMP, "test.db")

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import fitz  # noqa: E402
import export_abnahme  # noqa: E402
import pdf_export  # noqa: E402
import pdfua_export  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")


def _main():
    try:
        import main
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"main nicht ladbar: {e}")
    return main


def _bild(i, **kw):
    b = {"id": i, "page_number": 1, "image_index": i, "alt_text": "", "alt_text_edited": None, "original_alt": "",
         "image_type": "unknown", "status": "pending", "xref": -i, "bbox_x0": None, "is_vector": 1, "document_id": 7}
    b.update(kw)
    return b


class Abnahme(unittest.TestCase):
    """export_abnahme.verapdf_vergleich: 7.3-1 darf um erlaubt_ohne_alt zunehmen, alles andere nicht."""

    def _vgl(self, vor, nach, erlaubt):
        with tempfile.TemporaryDirectory() as d:
            q = os.path.join(d, "q.pdf")
            doc = fitz.open(); doc.new_page(); doc.save(q); doc.close()
            with mock.patch.object(export_abnahme, "_verapdf_regeln", side_effect=[vor, nach]):
                return export_abnahme.verapdf_vergleich(q, q, erlaubt)

    def test_leere_felder_kein_befund(self):
        v = self._vgl({("7.3", 1): 1}, {("7.3", 1): 3}, 2)
        self.assertEqual((v["neu"], v["schlechter"]), ([], []))
        self.assertEqual(v["feldstand"], ["7.3-1 (1->3)"])

    def test_neue_regel_nur_aus_leeren_feldern(self):
        v = self._vgl({}, {("7.3", 1): 2}, 2)
        self.assertEqual((v["neu"], v["schlechter"]), ([], []))

    def test_mehr_fehlende_alt_texte_als_gewollt_bleibt_befund(self):
        v = self._vgl({("7.3", 1): 1}, {("7.3", 1): 4}, 2)
        self.assertEqual(v["schlechter"], ["7.3-1 (1->4)"])
        v = self._vgl({}, {("7.3", 1): 1}, 0)
        self.assertEqual(v["neu"], ["7.3-1 (1x)"])

    def test_andere_regeln_bleiben_befund(self):
        v = self._vgl({("7.3", 1): 1}, {("7.3", 1): 2, ("7.18.1", 2): 1, ("7.1", 3): 5}, 5)
        self.assertEqual(v["feldstand"], ["7.3-1 (1->2)"])
        self.assertIn("7.18.1-2 (1x)", v["neu"])
        self.assertIn("7.1-3 (5x)", v["neu"])


class Hinweis(unittest.TestCase):
    """main._ohne_alt_im_export und _ohne_alt_hinweis: welche Bilder ohne Alt-Text in der Datei stehen."""

    @classmethod
    def setUpClass(cls):
        cls.m = _main()

    def _unit(self, bilder, methode="pdfix"):
        return {"doc": {"id": 7, "extraction_method": methode, "original_path": ""}, "images": bilder}

    def test_pdfix_weg(self):
        bilder = [_bild(1, original_alt="Sonnenblumen", page_number=1),
                  _bild(2, original_alt="Kuchen", alt_text_edited="Von Hand", page_number=1),
                  _bild(3, original_alt="Plakat", alt_text_edited="", page_number=2),          # bewusst geleert
                  _bild(4, page_number=2),                                                   # nie etwas da
                  _bild(5, original_alt="Logo", alt_text_edited="dekorativ", page_number=3),  # dekorativ
                  _bild(6, alt_text="Fehler bei der Analyse: Zeitueberschreitung", page_number=3)]
        liste = self.m._ohne_alt_im_export(self._unit(bilder))
        self.assertEqual([(b["nr"], b["seite"], b["dekorativ"], b["mitgebracht"]) for b in liste],
                         [(3, 2, False, True), (4, 2, False, False), (5, 3, True, True)])
        self.assertEqual(self.m._ohne_alt_erlaubt(self._unit(bilder), liste), 2)   # Plakat + dekorativ
        self.assertEqual(self.m._ohne_alt_hinweis(liste),
                         "Diese Bilder haben keinen Alt-Text: Bild 3 auf Seite 2, Bild 4 auf Seite 2, "
                         "Bild 5 auf Seite 3 (dekorativ). Das meldet auch die PDF-Prüfung.")
        self.assertEqual(self.m._ohne_alt_hinweis(liste[:1]),
                         "Bild 3 auf Seite 2 hat keinen Alt-Text. Das meldet auch die PDF-Prüfung.")
        self.assertEqual(self.m._ohne_alt_hinweis([]), "")

    def test_nummer_wie_die_bildkarte(self):
        """Nummer je Dokument nach Seite, dann image_index — wie app.html (display_nr)."""
        bilder = [_bild(11, image_index=30, page_number=2, alt_text_edited=""),
                  _bild(12, image_index=31, page_number=1, alt_text="Text")]
        liste = self.m._ohne_alt_im_export(self._unit(bilder))
        self.assertEqual([(b["nr"], b["seite"]) for b in liste], [(2, 2)])

    def test_ersatzweg_nur_bewusst_geleert_und_dekorativ(self):
        bilder = [_bild(1, xref=101, is_vector=0), _bild(2, xref=102, is_vector=0, alt_text_edited=""),
                  _bild(3, xref=103, is_vector=0, alt_text_edited="dekorativ")]
        with mock.patch("pdf_export.layout_vektorbilder", return_value=set()):
            liste = self.m._ohne_alt_im_export(self._unit(bilder, "fitz"))
        self.assertEqual([b["nr"] for b in liste], [2, 3])

    def test_uebersetzt(self):
        """Die vier Saetze stehen in allen sechs Katalogen, uebersetzt (de = Quelltext)."""
        from babel.messages.pofile import read_po
        ids = ["Bild {n} auf Seite {p}", "{bild} (dekorativ)", "{bild} hat keinen Alt-Text. Das meldet auch die PDF-Prüfung.",
               "Diese Bilder haben keinen Alt-Text: {liste}. Das meldet auch die PDF-Prüfung."]
        basis = os.path.dirname(self.m.__file__)
        for sprache in ("de", "en", "fr", "es", "da", "sv"):
            with open(os.path.join(basis, "locales", sprache, "LC_MESSAGES", "messages.po"), "rb") as f:
                katalog = read_po(f)
            for m in ids:
                eintrag = katalog.get(m)
                self.assertIsNotNone(eintrag, (sprache, m))
                if sprache != "de":
                    self.assertTrue(eintrag.string, (sprache, m))
                    for platzhalter in re.findall(r"\{\w+\}", m):
                        self.assertIn(platzhalter, eintrag.string, (sprache, m))


def _getaggt_mit_alt(pfad, alt="Blaues Quadrat (mitgebracht)"):
    """Seite mit Rasterbild in /Figure <</MCID 0>> BDC … EMC, Figure-Element MIT /Alt (wie eine getaggte Kundendatei)."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0)
    pix.set_rect(pix.irect, (30, 30, 200))
    page.insert_image(fitz.Rect(20, 20, 120, 120), pixmap=pix)
    xref, name = page.get_images(full=True)[0][0], page.get_images(full=True)[0][7]
    roh = page.read_contents().decode("latin-1")
    bild = re.search(rf"(q\s[\s\S]*?/{name}\s+Do\s*Q)", roh).group(1)
    neu = doc.get_new_xref(); doc.update_object(neu, "<< >>")
    doc.update_stream(neu, ("/Figure <</MCID 0>> BDC\n" + bild + "\nEMC\n").encode("latin-1"))
    doc.xref_set_key(page.xref, "Contents", f"{neu} 0 R")
    root, dok, fig, pt, arr = (doc.get_new_xref() for _ in range(5))
    doc.update_object(fig, f"<< /Type /StructElem /S /Figure /P {dok} 0 R /Pg {page.xref} 0 R /K 0 >>")
    doc.xref_set_key(fig, "Alt", pdf_export._pdf_string(alt))
    doc.update_object(dok, f"<< /Type /StructElem /S /Document /P {root} 0 R /K [ {fig} 0 R ] >>")
    doc.update_object(arr, f"[ {fig} 0 R ]")
    doc.update_object(pt, f"<< /Nums [ 0 {arr} 0 R ] >>")
    doc.update_object(root, f"<< /Type /StructTreeRoot /K {dok} 0 R /ParentTree {pt} 0 R /ParentTreeNextKey 1 >>")
    doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    doc.xref_set_key(page.xref, "StructParents", "0")
    doc.save(pfad)
    doc.close()
    return xref, fig


def _alt(pfad, xref):
    doc = fitz.open(pfad)
    try:
        return doc.xref_get_key(xref, "Alt")
    finally:
        doc.close()


class Ersatzweg(unittest.TestCase):
    """Punkt 4: pdf_export.write_alt_texts_to_pdf(alt_entfernen) und main._alt_texte_einsetzen (fitz)."""

    def test_vorhandenes_alt_wird_entfernt(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref, fig = _getaggt_mit_alt(q)
            self.assertIn("mitgebracht", _alt(q, fig)[1])
            r = pdf_export.write_alt_texts_to_pdf(q, z, {}, [{"xref": xref, "page_number": 1, "is_vector": False,
                                                               "bbox": None, "alt_text": "", "image_path": None}],
                                                  alt_entfernen={xref})
            self.assertEqual((r["entfernt"], r["tagged_count"]), (1, 0), r)
            self.assertEqual(_alt(z, fig)[0], "null")
            doc = fitz.open(z)
            self.assertEqual(doc[0].read_contents(), fitz.open(q)[0].read_contents())   # Inhaltsstrom unveraendert
            doc.close()

    def test_ohne_auftrag_bleibt_der_text(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref, fig = _getaggt_mit_alt(q)
            r = pdf_export.write_alt_texts_to_pdf(q, z, {}, [])
            self.assertEqual(r.get("entfernt"), 0)
            self.assertIn("mitgebracht", _alt(z, fig)[1])

    def test_ungetaggte_pdf_bekommt_keine_struktur(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            doc = fitz.open(); page = doc.new_page(width=200, height=200)
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0); pix.set_rect(pix.irect, (30, 30, 200))
            page.insert_image(fitz.Rect(20, 20, 120, 120), pixmap=pix)
            xref = page.get_images()[0][0]; doc.save(q); doc.close()
            r = pdf_export.write_alt_texts_to_pdf(q, z, {}, [], alt_entfernen={xref})
            self.assertEqual(r["entfernt"], 0)
            doc = fitz.open(z)
            self.assertEqual(doc.xref_get_key(doc.pdf_catalog(), "StructTreeRoot")[0], "null")
            doc.close()

    def test_einsetzen_bewusst_geleert_nie_angefasst_und_tagging(self):
        m = _main()
        with tempfile.TemporaryDirectory() as d:
            q = os.path.join(d, "q.pdf")
            xref, fig = _getaggt_mit_alt(q)
            doc = {"extraction_method": "fitz", "original_path": q}
            fall = {"nie angefasst": (_bild(1, xref=xref, is_vector=0), None, "mitgebracht"),
                    "bewusst geleert": (_bild(1, xref=xref, is_vector=0, alt_text_edited=""), None, None),
                    "Tagging, Feld leer": (_bild(1, xref=xref, is_vector=0), (lambda _z: True), None),
                    "Text": (_bild(1, xref=xref, is_vector=0, alt_text="Neuer Text"), None, "Neuer Text")}
            for name, (bild, leer_fn, erwartet) in fall.items():
                z = os.path.join(d, f"z_{len(name)}.pdf")
                with mock.patch("pdf_export.layout_vektorbilder", return_value=set()):
                    info, _t, _s = m._alt_texte_einsetzen(doc, [bild], q, z, d, leer_entfernen=leer_fn)
                art, wert = _alt(z, fig)
                if erwartet is None:
                    self.assertEqual(art, "null", name)
                else:
                    self.assertIn(erwartet, wert, name)


class ExportAblauf(unittest.TestCase):
    """Punkt 1 im Export-Ablauf: _build_pdf_for_document mit einer getaggten Kundendatei, deren Bild bewusst geleert
    wurde (Ersatzweg entfernt das /Alt). veraPDF ist ersetzt: Zunahme von 7.3-1 um 1 -> Datei; dazu eine andere
    Regel -> weiter 422."""

    def _unit(self, q, xref):
        b = _bild(1, xref=xref, is_vector=0, alt_text_edited="", original_alt="", page_number=1)
        return {"doc": {"id": 7, "project_id": 3, "original_path": q, "original_filename": "q.pdf",
                        "extraction_method": "fitz", "display_name": "Testdokument"}, "images": [b]}

    def test_bewusst_geleert_sperrt_nicht(self):
        m = _main()
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as d:
            q = os.path.join(d, "q.pdf")
            xref, fig = _getaggt_mit_alt(q)
            vor, nach = {("7.3", 1): 0}, {("7.3", 1): 1}
            with mock.patch("pdf_export.layout_vektorbilder", return_value=set()), \
                 mock.patch.object(export_abnahme, "_verapdf_regeln", side_effect=[vor, nach]):
                out, info = m._build_pdf_for_document(self._unit(q, xref), os.path.join(d, "out"))
            self.assertTrue(info["abnahme"]["ok"], info["abnahme"])
            self.assertEqual(info["abnahme"]["kennzahlen"].get("verapdf_feldstand"), "7.3-1 (0->1)")
            self.assertEqual([b["nr"] for b in info["ohne_alt"]], [1])
            self.assertEqual(_alt(out, fig)[0], "null")
            with mock.patch("pdf_export.layout_vektorbilder", return_value=set()), \
                 mock.patch.object(export_abnahme, "_verapdf_regeln", side_effect=[vor, {("7.3", 1): 1, ("7.18.1", 2): 1}]):
                with self.assertRaises(HTTPException) as e:
                    m._build_pdf_for_document(self._unit(q, xref), os.path.join(d, "out2"))
            self.assertEqual(e.exception.status_code, 422)
            self.assertIn("7.18.1-2", e.exception.detail)
            self.assertNotIn("7.3-1", e.exception.detail)


class WordAlsPdf(unittest.TestCase):
    """Punkt 3: pdfua_export.bildtitel_entfernen — Titel der Bilder raus, Beschreibung (Feldtext) bleibt."""

    def _docx_mit_titel(self, pfad):
        quelle = os.path.join(FIXTURES, "word_einfach.docx")
        with zipfile.ZipFile(quelle) as z:
            teile = {i.filename: z.read(i.filename) for i in z.infolist()}
        xml = teile["word/document.xml"].decode("utf-8")
        n = len(re.findall(r"<wp:docPr ", xml))
        self.assertGreater(n, 0, "Testdatei ohne Bild")
        xml = re.sub(r"<wp:docPr ([^>]*?)(/?)>",
                     lambda mm: "<wp:docPr " + re.sub(r'\s(title|descr)="[^"]*"', "", mm.group(1))
                                + ' title="Bildtitel (fiktiv)" descr="Feldtext (fiktiv)"' + mm.group(2) + ">", xml)
        teile["word/document.xml"] = xml.encode("utf-8")
        with zipfile.ZipFile(pfad, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in teile.items():
                z.writestr(name, data)
        return n

    def test_titel_weg_beschreibung_bleibt(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.docx")
            n = self._docx_mit_titel(p)
            vorher = zipfile.ZipFile(p).read("word/styles.xml")
            self.assertGreaterEqual(pdfua_export.bildtitel_entfernen(p), 1)
            xml = zipfile.ZipFile(p).read("word/document.xml").decode("utf-8")
            self.assertNotIn("Bildtitel (fiktiv)", xml)
            self.assertEqual(xml.count('descr="Feldtext (fiktiv)"'), n)
            self.assertEqual(zipfile.ZipFile(p).read("word/styles.xml"), vorher)   # andere Teile byte-gleich
            self.assertEqual(pdfua_export.bildtitel_entfernen(p), 0)                # nichts mehr zu tun

    def test_umwandlung_schreibt_nur_den_feldtext(self):
        if not pdfua_export.verfuegbar():
            self.skipTest("kein Umwandler (KONVERTER_URL)")
        import pikepdf
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.docx")
            self._docx_mit_titel(p)
            pdfua_export.bildtitel_entfernen(p)
            pdf, _bericht = pdfua_export.konvertiere(p, "w.docx")
            with pikepdf.open(io.BytesIO(pdf)) as doc:
                alts = [str(o.get("/Alt")) for o in doc.objects
                        if isinstance(o, pikepdf.Dictionary) and str(o.get("/S", "")) == "/Figure" and "/Alt" in o]
            self.assertTrue(alts)
            self.assertTrue(all(a == "Feldtext (fiktiv)" for a in alts), alts)


if __name__ == "__main__":
    unittest.main()
