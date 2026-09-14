"""fitz-Export (Ersatzweg ohne PDFix), 14.09.2026: Figure-Elemente mit Alt-Text muessen in JEDER Form
des /K-Eintrags am Dokument-Knoten eingehaengt werden (inline-Array, Referenz auf Array-Objekt wie bei
InDesign, Einzelkind) und den Abschluss-Schritt ueberleben. Anlass: Prod-Kundendokument mit
/K 2195 0 R — 234 Alt-Texte wurden nicht eingehaengt und von finalize_export_pdf als „verwaist“ entfernt.
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_pdf_export_einhaengen.py -v
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


def _pdf_mit_struktur(pfad: str, k_form: str) -> int:
    """Eine Seite mit einem Rasterbild und einem Tag-Baum: Root -> Document -> (ein P-Element).
    k_form: 'inline' (/K [p]), 'ref' (/K <arr> 0 R mit Array-Objekt), 'einzel' (/K p 0 R)."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0)
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(fitz.Rect(20, 20, 120, 120), pixmap=pix)
    root = doc.get_new_xref(); dokument = doc.get_new_xref(); absatz = doc.get_new_xref(); pt = doc.get_new_xref()
    doc.update_object(absatz, f"<< /Type /StructElem /S /P /P {dokument} 0 R /Pg {page.xref} 0 R /K 0 >>")
    if k_form == "inline":
        k = f"[ {absatz} 0 R ]"
    elif k_form == "ref":
        arr = doc.get_new_xref(); doc.update_object(arr, f"[ {absatz} 0 R ]"); k = f"{arr} 0 R"
    else:
        k = f"{absatz} 0 R"
    doc.update_object(dokument, f"<< /Type /StructElem /S /Document /P {root} 0 R /K {k} >>")
    doc.update_object(pt, f"<< /Nums [ 0 [ {absatz} 0 R ] ] >>")
    doc.update_object(root, f"<< /Type /StructTreeRoot /K {dokument} 0 R /ParentTree {pt} 0 R /ParentTreeNextKey 1 >>")
    doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    doc.xref_set_key(page.xref, "StructParents", "0")
    doc.save(pfad)
    xref = page.get_images()[0][0]
    doc.close()
    return xref


class TestFigureEinhaengen(unittest.TestCase):
    def _lauf(self, k_form):
        with tempfile.TemporaryDirectory() as d:
            quelle = os.path.join(d, "q.pdf"); ziel = os.path.join(d, "z.pdf")
            xref = _pdf_mit_struktur(quelle, k_form)
            meta = [{"xref": xref, "page_number": 1, "is_vector": False, "bbox": None,
                     "alt_text": "Rotes Quadrat", "image_path": None}]
            r = pdf_export.write_alt_texts_to_pdf(quelle, ziel, {xref: "Rotes Quadrat"}, meta)
            self.assertEqual(r["tagged_count"], 1, r)
            self.assertEqual(r["unreachable_figures"], [], r)
            self.assertFalse(any("nicht in den Tag-Baum" in w for w in r["warnings"]), r["warnings"])
            f = pdf_export.finalize_export_pdf(ziel, title="Test", schonen=set(r["figure_xrefs"]))
            self.assertEqual(f["orphan_alts_removed"], 0)
            doc = fitz.open(ziel)
            root = int(doc.xref_get_key(doc.pdf_catalog(), "StructTreeRoot")[1].split()[0])
            erreichbar = pdf_export._collect_reachable_struct_elems(doc, root)
            for fx in r["figure_xrefs"]:
                self.assertIn(fx, erreichbar, f"Figure {fx} nicht erreichbar ({k_form})")
                self.assertIn("Rotes Quadrat", doc.xref_get_key(fx, "Alt")[1])
            doc.close()

    def test_inline_array(self):
        self._lauf("inline")

    def test_referenz_auf_array_objekt_indesign(self):
        self._lauf("ref")

    def test_einzelkind(self):
        self._lauf("einzel")

    def test_schonliste_haelt_eigene_figures(self):
        """Selbst wenn ein Figure-Element unerreichbar waere, darf finalize es mit Schonliste nicht loeschen."""
        with tempfile.TemporaryDirectory() as d:
            pfad = os.path.join(d, "s.pdf")
            _pdf_mit_struktur(pfad, "inline")
            doc = fitz.open(pfad)
            waise = doc.get_new_xref()
            doc.update_object(waise, "<< /Type /StructElem /S /Figure /Alt (Waise) >>")
            doc.saveIncr(); doc.close()
            f = pdf_export.finalize_export_pdf(pfad, title="Test", schonen={waise})
            self.assertEqual(f["orphan_alts_removed"], 0)
            f2 = pdf_export.finalize_export_pdf(pfad, title="Test")
            self.assertEqual(f2["orphan_alts_removed"], 1)


if __name__ == "__main__":
    unittest.main()
