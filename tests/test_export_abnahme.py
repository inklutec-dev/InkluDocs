"""Export-Abnahme (15.09.2026): misst exportierte PDFs unabhaengig vom Schreibweg.
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_export_abnahme.py -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import fitz  # noqa: E402
import pdf_export  # noqa: E402
from unittest import mock  # noqa: E402
import export_abnahme  # noqa: E402
from export_abnahme import abnahme_pdf, abnahme_loggen, verapdf_vergleich  # noqa: E402


def _quelle_mit_bild(pfad: str, seiten: int = 1) -> int:
    doc = fitz.open()
    xref = None
    for _ in range(seiten):
        page = doc.new_page(width=200, height=200)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0)
        pix.set_rect(pix.irect, (30, 30, 200))
        page.insert_image(fitz.Rect(20, 20, 120, 120), pixmap=pix)
        xref = xref or page.get_images()[0][0]
    doc.save(pfad)
    doc.close()
    return xref


def _export(quelle: str, ziel: str, xref: int, text: str) -> dict:
    meta = [{"xref": xref, "page_number": 1, "is_vector": False, "bbox": None,
             "alt_text": text, "image_path": None}]
    r = pdf_export.write_alt_texts_to_pdf(quelle, ziel, {xref: text}, meta)
    pdf_export.finalize_export_pdf(ziel, title="Test", schonen=set(r.get("figure_xrefs") or []))
    return r


class TestAbnahme(unittest.TestCase):
    def test_sauberer_export_besteht(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q)
            r = _export(q, z, xref, "Blaues Quadrat auf weissem Grund")
            a = abnahme_pdf(z, q, ["Blaues Quadrat auf weissem Grund"], erwartet_getaggt=r["tagged_count"], verapdf=False)
            self.assertTrue(a["ok"], a)
            self.assertEqual(a["kennzahlen"]["figures_mit_alt"], 1)
            self.assertEqual(a["kennzahlen"]["waisen"], 0)
            self.assertEqual(a["kennzahlen"]["seiten_unbalanciert"], 0)
            self.assertEqual(a["kennzahlen"]["texte_gefunden"], 1)
            zeile = abnahme_loggen(a, projekt=1, dokument=2, verfahren="fitz", datei=z)
            self.assertTrue(zeile.startswith("EXPORT-ABNAHME ok "), zeile)

    def test_gemeldeter_text_fehlt(self):
        """Der Export meldet 2 getaggt, in der Datei steht nur einer -> Befund."""
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q)
            _export(q, z, xref, "Blaues Quadrat")
            a = abnahme_pdf(z, q, ["Blaues Quadrat", "Roter Kreis"], erwartet_getaggt=2, verapdf=False)
            self.assertFalse(a["ok"])
            self.assertTrue(any("Nur 1 von 2" in b for b in a["befunde"]), a)
            self.assertIn("FEHLGESCHLAGEN", abnahme_loggen(a))

    def test_waise_wird_erkannt(self):
        """Ein Figure-Element mit Alt ohne Verbindung zum Strukturbaum -> Befund."""
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q)
            _export(q, z, xref, "Blaues Quadrat")
            doc = fitz.open(z)
            w = doc.get_new_xref()
            doc.update_object(w, "<< /Type /StructElem /S /Figure /Alt (Verwaister Text) >>")
            doc.save(z, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
            doc.close()
            a = abnahme_pdf(z, q, ["Blaues Quadrat"], erwartet_getaggt=1, verapdf=False)
            self.assertFalse(a["ok"])
            self.assertEqual(a["kennzahlen"]["waisen"], 1, a)

    def test_seitenzahl_abweichung(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q, seiten=2)
            _export(q, z, xref, "Blaues Quadrat")
            doc = fitz.open(z); doc.delete_page(1); doc.save(os.path.join(d, "z2.pdf")); doc.close()
            a = abnahme_pdf(os.path.join(d, "z2.pdf"), q, ["Blaues Quadrat"], erwartet_getaggt=1, verapdf=False)
            self.assertTrue(any("Seitenzahl 1 statt 2" in b for b in a["befunde"]), a)

    def test_unbalancierte_marker(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q)
            _export(q, z, xref, "Blaues Quadrat")
            doc = fitz.open(z)
            page = doc[0]
            strom = doc.xref_stream(page.get_contents()[0])
            doc.update_stream(page.get_contents()[0], strom + b"\n/Artifact BMC\n")
            doc.save(os.path.join(d, "z3.pdf")); doc.close()
            a = abnahme_pdf(os.path.join(d, "z3.pdf"), q, ["Blaues Quadrat"], erwartet_getaggt=1, verapdf=False)
            self.assertFalse(a["ok"])
            self.assertTrue(any("Marker unbalanciert auf Seite(n) 1" in b for b in a["befunde"]), a)

    def test_dekorativ_und_leer_zaehlen_nicht(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q)
            _export(q, z, xref, "Blaues Quadrat")
            a = abnahme_pdf(z, q, ["Blaues Quadrat", "", "dekorativ", None], erwartet_getaggt=3, verapdf=False)
            self.assertTrue(a["ok"], a)
            self.assertEqual(a["kennzahlen"]["texte_geschrieben"], 1)

    def test_lesereihenfolge_ruecksprung(self):
        """Zwei Bilder: das Element fuer Seite 20 steht im Baum VOR dem fuer Seite 1 -> Befund;
        ein kleiner Ruecksprung (Seite 3 vor Seite 2) bleibt erlaubt."""
        with tempfile.TemporaryDirectory() as d:
            for name, seiten_folge, erwartet_ok in (("gross.pdf", [20, 1], False), ("klein.pdf", [3, 2], True)):
                pfad = os.path.join(d, name)
                doc = fitz.open()
                for _ in range(20):
                    doc.new_page(width=100, height=100)
                root = doc.get_new_xref(); dok = doc.get_new_xref()
                figs = []
                for s_nr in seiten_folge:
                    f = doc.get_new_xref()
                    doc.update_object(f, f"<< /Type /StructElem /S /Figure /P {dok} 0 R /Pg {doc[s_nr - 1].xref} 0 R /Alt (Bild Seite {s_nr}) >>")
                    figs.append(f)
                doc.update_object(dok, f"<< /Type /StructElem /S /Document /P {root} 0 R /K [ {' '.join(f'{f} 0 R' for f in figs)} ] >>")
                doc.update_object(root, f"<< /Type /StructTreeRoot /K {dok} 0 R >>")
                doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
                doc.save(pfad); doc.close()
                a = abnahme_pdf(pfad, pfad, [f"Bild Seite {s}" for s in seiten_folge], erwartet_getaggt=2, verapdf=False)
                self.assertEqual(a["ok"], erwartet_ok, a)
                if not erwartet_ok:
                    self.assertTrue(any("Lesereihenfolge" in b and "20->1" in b for b in a["befunde"]), a)
                    self.assertEqual(a["kennzahlen"]["ruecksprünge_gross"], 1)
                else:
                    self.assertEqual(a["kennzahlen"]["ruecksprünge_klein"], 1)

    def test_verapdf_vergleich_gemockt(self):
        """Regel 7: neue Regel oder haeufigere Verletzung = Befund; gleich oder besser = ok."""
        vor = {("7.1", 1): 10, ("7.3", 2): 5}
        faelle = [
            ({("7.1", 1): 8, ("7.3", 2): 5}, True, "besser"),
            ({("7.1", 1): 10, ("7.3", 2): 5, ("7.18.5", 1): 3}, False, "neue Regel"),
            ({("7.1", 1): 12, ("7.3", 2): 5}, False, "haeufiger"),
        ]
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q); _export(q, z, xref, "Blaues Quadrat")
            for nach, ok, name in faelle:
                with mock.patch.object(export_abnahme, "_verapdf_regeln", side_effect=[vor, nach]):
                    a = abnahme_pdf(z, q, ["Blaues Quadrat"], erwartet_getaggt=1, verapdf=True)
                self.assertEqual(a["ok"], ok, (name, a))
                self.assertEqual(a["kennzahlen"]["verapdf"], f"{len(nach)}/{len(vor)}")
            # Pruefdienst faellt aus -> kein Befund, Kennzahl „nicht moeglich“
            with mock.patch.object(export_abnahme, "_verapdf_regeln", side_effect=RuntimeError("Konverter weg")):
                a = abnahme_pdf(z, q, ["Blaues Quadrat"], erwartet_getaggt=1, verapdf=True)
            self.assertTrue(a["ok"], a)
            self.assertEqual(a["kennzahlen"]["verapdf"], "nicht moeglich")

    def test_verapdf_echt_wenn_konverter_da(self):
        """Laeuft nur mit KONVERTER_URL (Staging-Container): sauberer Export ist nicht schlechter als die Quelle."""
        if not os.environ.get("KONVERTER_URL"):
            self.skipTest("kein Konverter")
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref = _quelle_mit_bild(q); _export(q, z, xref, "Blaues Quadrat")
            v = verapdf_vergleich(q, z)
            self.assertTrue(v["moeglich"], v)
            self.assertEqual(v["neu"], [], v)
            self.assertEqual(v["schlechter"], [], v)

    def test_kaputte_datei(self):
        with tempfile.TemporaryDirectory() as d:
            z = os.path.join(d, "kaputt.pdf")
            with open(z, "wb") as f:
                f.write(b"kein pdf")
            a = abnahme_pdf(z, None, ["x"], verapdf=False)
            self.assertFalse(a["ok"])
            self.assertTrue(any("nicht lesbar" in b for b in a["befunde"]), a)


if __name__ == "__main__":
    unittest.main()
