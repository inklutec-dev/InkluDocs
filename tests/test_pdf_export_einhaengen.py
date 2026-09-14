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


def _inhalt_ersetzen(doc, page, text: str) -> None:
    """Seiteninhalt durch EINEN neuen Strom ersetzen (fitz legt nach insert_image mehrere an)."""
    neu = doc.get_new_xref(); doc.update_object(neu, "<< >>"); doc.update_stream(neu, text.encode("latin-1"))
    doc.xref_set_key(page.xref, "Contents", f"{neu} 0 R")


def _pdf_getaggt_mit_inhalt(pfad: str, inhalt_vorlage: str, k_elem: str = "0") -> tuple:
    """Seite mit Rasterbild und selbst geschriebenem Inhaltsstrom (Marked Content), Tag-Baum
    Root -> Document -> Figure (MCID 0) plus ParentTree-Eintrag. Liefert (bild_xref, figure_xref)."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0); pix.set_rect(pix.irect, (30, 30, 200))
    page.insert_image(fitz.Rect(20, 20, 120, 120), pixmap=pix)
    xref = page.get_images(full=True)[0][0]; name = page.get_images(full=True)[0][7]
    roh = page.read_contents().decode("latin-1")
    m = __import__("re").search(rf"(q\s[\s\S]*?/{name}\s+Do\s*Q)", roh)
    neu = inhalt_vorlage.replace("{BILD}", m.group(1))
    _inhalt_ersetzen(doc, page, neu)
    root = doc.get_new_xref(); dokument = doc.get_new_xref(); figur = doc.get_new_xref(); pt = doc.get_new_xref(); arr = doc.get_new_xref()
    doc.update_object(figur, f"<< /Type /StructElem /S /Figure /P {dokument} 0 R /Pg {page.xref} 0 R /K {k_elem} >>")
    doc.update_object(dokument, f"<< /Type /StructElem /S /Document /P {root} 0 R /K [ {figur} 0 R ] >>")
    doc.update_object(arr, f"[ {figur} 0 R ]")
    doc.update_object(pt, f"<< /Nums [ 0 {arr} 0 R ] >>")
    doc.update_object(root, f"<< /Type /StructTreeRoot /K {dokument} 0 R /ParentTree {pt} 0 R /ParentTreeNextKey 1 >>")
    doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    doc.xref_set_key(page.xref, "StructParents", "0")
    doc.save(pfad); doc.close()
    return xref, figur


class TestVorhandeneTagsUndArtefakte(unittest.TestCase):
    def test_bild_in_vorhandenem_figure_bekommt_dort_den_alt(self):
        """Bild liegt in /Figure <</MCID 0>> BDC ... EMC des Originals -> /Alt am vorhandenen Element, kein neues."""
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref, figur = _pdf_getaggt_mit_inhalt(q, "/Figure <</MCID 0>> BDC\n{BILD}\nEMC\n")
            r = pdf_export.write_alt_texts_to_pdf(q, z, {xref: "Blaues Quadrat"},
                                                  [{"xref": xref, "page_number": 1, "is_vector": False, "bbox": None, "alt_text": "Blaues Quadrat", "image_path": None}])
            self.assertEqual(r["tagged_count"], 1, r)
            self.assertEqual(r["figure_xrefs"], [figur], r)
            doc = fitz.open(z)
            self.assertIn("Blaues Quadrat", doc.xref_get_key(figur, "Alt")[1])
            self.assertEqual(sum(1 for x in range(1, doc.xref_length()) if "/S /Figure" in doc.xref_object(x, compressed=True).replace("/S/Figure", "/S /Figure")), 1)
            self.assertNotIn("/MCID 1", doc[0].read_contents().decode("latin-1"), "Inhaltsstrom darf nicht veraendert sein")
            doc.close()

    def test_bild_im_artefakt_wird_herausgeloest_und_marker_bleiben_balanciert(self):
        """Bild liegt in einem /Artifact-Block: Artefakt wird davor geschlossen, Figure eingefuegt, danach
        wieder geoeffnet; BDC/EMC bleiben balanciert; ParentTree bekommt den Eintrag."""
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            xref, figur = _pdf_getaggt_mit_inhalt(q, "/Figure <</MCID 0>> BDC\n0 0 m 1 1 l S\nEMC\n/Artifact <</Type /Pagination>> BDC\n{BILD}\n0 0 m 2 2 l S\nEMC\n")
            r = pdf_export.write_alt_texts_to_pdf(q, z, {xref: "Blaues Quadrat"},
                                                  [{"xref": xref, "page_number": 1, "is_vector": False, "bbox": None, "alt_text": "Blaues Quadrat", "image_path": None}])
            self.assertEqual(r["tagged_count"], 1, r)
            self.assertEqual(r["unreachable_figures"], [], r)
            doc = fitz.open(z)
            cs = doc[0].read_contents().decode("latin-1")
            self.assertEqual(cs.count("BDC"), cs.count("EMC"), cs)
            self.assertIn("/Figure <</MCID 1>> BDC", cs)
            self.assertEqual(cs.count("/Artifact <</Type /Pagination>> BDC"), 2, "Artefakt vor und nach dem Bild wieder geoeffnet")
            root = int(doc.xref_get_key(doc.pdf_catalog(), "StructTreeRoot")[1].split()[0])
            eintraege = pdf_export._parenttree_elemente(doc, root, 0)
            self.assertEqual(len(eintraege), 2); self.assertEqual(eintraege[1], r["figure_xrefs"][0])
            f = pdf_export.finalize_export_pdf(z, title="Test", schonen=set(r["figure_xrefs"]))
            self.assertEqual(f["orphan_alts_removed"], 0)
            doc.close()

    def test_zwei_bilder_in_einem_figure_werden_zusammengefuehrt(self):
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            doc = fitz.open(); page = doc.new_page(width=200, height=200)
            for i, farbe in enumerate([(200, 0, 0), (0, 0, 200)]):
                pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0); pix.set_rect(pix.irect, farbe)
                page.insert_image(fitz.Rect(20 + 90 * i, 20, 100 + 90 * i, 100), pixmap=pix)
            infos = page.get_images(full=True); doc.save(q); doc.close()
            # Tag-Baum + Inhalt ueber die Hilfsfunktion nachbauen (beide Bilder in EINEM Figure-Block)
            doc = fitz.open(q); page = doc[0]; roh = page.read_contents().decode("latin-1")
            _inhalt_ersetzen(doc, page, "/Figure <</MCID 0>> BDC\n" + roh + "\nEMC\n")
            root = doc.get_new_xref(); dok = doc.get_new_xref(); fig = doc.get_new_xref(); pt = doc.get_new_xref()
            doc.update_object(fig, f"<< /Type /StructElem /S /Figure /P {dok} 0 R /Pg {page.xref} 0 R /K 0 >>")
            doc.update_object(dok, f"<< /Type /StructElem /S /Document /P {root} 0 R /K [ {fig} 0 R ] >>")
            doc.update_object(pt, f"<< /Nums [ 0 [ {fig} 0 R ] ] >>")
            doc.update_object(root, f"<< /Type /StructTreeRoot /K {dok} 0 R /ParentTree {pt} 0 R /ParentTreeNextKey 1 >>")
            doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R"); doc.xref_set_key(page.xref, "StructParents", "0")
            doc.save(z + ".src.pdf"); doc.close()
            alt = {infos[0][0]: "Rotes Quadrat", infos[1][0]: "Blaues Quadrat"}
            meta = [{"xref": x, "page_number": 1, "is_vector": False, "bbox": None, "alt_text": a, "image_path": None} for x, a in alt.items()]
            r = pdf_export.write_alt_texts_to_pdf(z + ".src.pdf", z, alt, meta)
            self.assertEqual(r["tagged_count"], 2, r)
            doc = fitz.open(z)
            a = doc.xref_get_key(fig, "Alt")[1]
            self.assertIn("Rotes Quadrat", a); self.assertIn("Blaues Quadrat", a)
            doc.close()


class TestLesereihenfolge(unittest.TestCase):
    def test_neues_figure_steht_bei_seiner_seite_nicht_am_ende(self):
        """Zwei Seiten mit je einem Absatz; Bild auf Seite 1 in einem Artefakt-Block. Das neue Figure muss
        in der Kinderliste des Dokuments VOR dem Absatz von Seite 2 stehen (14.09.2026: vorher am Ende)."""
        with tempfile.TemporaryDirectory() as d:
            q, z = os.path.join(d, "q.pdf"), os.path.join(d, "z.pdf")
            doc = fitz.open(); doc.new_page(width=200, height=200); doc.new_page(width=200, height=200)
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), 0); pix.set_rect(pix.irect, (30, 200, 30))
            doc[0].insert_image(fitz.Rect(20, 20, 120, 120), pixmap=pix)
            name = doc[0].get_images(full=True)[0][7]; xref = doc[0].get_images(full=True)[0][0]
            roh = doc[0].read_contents().decode("latin-1")
            bild = __import__("re").search(rf"(q\s[\s\S]*?/{name}\s+Do\s*Q)", roh).group(1)
            x1, x2 = doc[0].xref, doc[1].xref
            _inhalt_ersetzen(doc, doc[0], "/P <</MCID 0>> BDC\nBT ET\nEMC\n/Artifact BDC\n" + bild + "\nEMC\n")
            _inhalt_ersetzen(doc, doc[1], "/P <</MCID 0>> BDC\nBT ET\nEMC\n")
            root = doc.get_new_xref(); dok = doc.get_new_xref(); a1 = doc.get_new_xref(); a2 = doc.get_new_xref(); pt = doc.get_new_xref()
            doc.update_object(a1, f"<< /Type /StructElem /S /P /P {dok} 0 R /Pg {x1} 0 R /K 0 >>")
            doc.update_object(a2, f"<< /Type /StructElem /S /P /P {dok} 0 R /Pg {x2} 0 R /K 0 >>")
            doc.update_object(dok, f"<< /Type /StructElem /S /Document /P {root} 0 R /K [ {a1} 0 R {a2} 0 R ] >>")
            doc.update_object(pt, f"<< /Nums [ 0 [ {a1} 0 R ] 1 [ {a2} 0 R ] ] >>")
            doc.update_object(root, f"<< /Type /StructTreeRoot /K {dok} 0 R /ParentTree {pt} 0 R /ParentTreeNextKey 2 >>")
            doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
            doc.xref_set_key(x1, "StructParents", "0"); doc.xref_set_key(x2, "StructParents", "1")
            doc.save(q); doc.close()
            r = pdf_export.write_alt_texts_to_pdf(q, z, {xref: "Gruenes Quadrat"},
                                                  [{"xref": xref, "page_number": 1, "is_vector": False, "bbox": None, "alt_text": "Gruenes Quadrat", "image_path": None}])
            self.assertEqual(r["tagged_count"], 1, r)
            doc = fitz.open(z)
            kinder = [int(m) for m in __import__("re").findall(r"(\d+)\s+0\s+R", doc.xref_get_key(dok, "K")[1])]
            fig = r["figure_xrefs"][0]
            self.assertEqual(kinder, [a1, fig, a2], f"Figure muss zwischen Absatz Seite 1 und Absatz Seite 2 stehen: {kinder}")
            self.assertEqual(doc.xref_get_key(fig, "P")[1], f"{dok} 0 R")
            doc.close()
