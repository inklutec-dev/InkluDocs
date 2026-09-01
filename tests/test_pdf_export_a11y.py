"""Tests fuer die Export-Haertung vom 12.06.2026.

Abgedeckt:
- _build_csv_rows (pdfix_roundtrip): leere Alt-Texte ueberspringen,
  "dekorativ" -> leerer Alt-Text, Luecken in den laufenden Nummern erlaubt.
- write_alt_texts_to_pdf (pdf_export, fitz-Pfad): keine leeren /Alt-Eintraege,
  kein /Alt mehr am Bild-XObject.
- finalize_export_pdf (pdf_export): Dokumentsprache, Titel-Prioritaet,
  DisplayDocTitle, Entfernen verwaister /Alt-StructElems.
- Import-Script (AltTag_Import_CSV.py, nur wenn pdfix-sdk + Test-PDF da):
  Luecken lassen Figures unangetastet, ungueltige laufende Nummern
  brechen laut ab (Exit-Code 4).

Ausfuehrung im Container:
  docker exec inkludocs-staging python3 /app/tests/test_pdf_export_a11y.py
"""
import os
import re
import shutil
import tempfile
import unittest

import sys
# Modulpfade: im Container liegen die Backend-Module flach unter /app,
# im Repo unter backend/. Beide Varianten abdecken.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("/app", "/app/backend", os.path.join(_repo_root, "backend")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import fitz

import pdfix_roundtrip
from pdf_export import write_alt_texts_to_pdf, finalize_export_pdf

_TEST_PDF_CANDIDATES = [
    "/app/backend/pdfix_scripts/test_data/PDFix_AutoTag_Test_01.pdf",
    "/app/tests/fixtures/PDFix_AutoTag_Test_01.pdf",
    "/opt/pdfix-work/PDFix_AutoTag_Test_01.pdf",
]


def _find_heine_test_pdf():
    for c in _TEST_PDF_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _make_pdf_with_images(path, n_images=2):
    """Erzeugt eine 1-seitige PDF mit n eingebetteten Rasterbildern.
    Gibt die Liste der Bild-xrefs zurueck (Reihenfolge wie eingefuegt).
    Jedes Bild bekommt einen anderen Farbwert — identische Bilddaten wuerde
    PyMuPDF zu EINEM XObject deduplizieren, dann waeren die xrefs gleich."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(n_images):
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
        pix.clear_with(60 + i * 60)
        rect = fitz.Rect(50, 50 + i * 120, 150, 150 + i * 120)
        page.insert_image(rect, stream=pix.tobytes("png"))
    doc.save(path)
    doc.close()
    doc = fitz.open(path)
    xrefs = sorted({info[0] for info in doc[0].get_images(full=True)})
    doc.close()
    return xrefs


def _alt_entries(path):
    """Listet (xref, ist_struct_elem, alt_wert) fuer alle /Alt-Objekte der PDF."""
    doc = fitz.open(path)
    entries = []
    for x in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(x, compressed=True)
        except Exception:
            continue
        if "/Alt" not in obj:
            continue
        is_elem = "/StructElem" in obj or re.search(r"/S\s*/\w+", obj) is not None
        m = re.search(r"/Alt\s*(\((?:[^()\\]|\\.)*\)|<[0-9A-Fa-f]*>)", obj)
        entries.append((x, is_elem, m.group(1) if m else None))
    doc.close()
    return entries


class TestBuildCsvRows(unittest.TestCase):
    """CSV-Zeilen fuer das PDFix-Import-Script."""

    def test_none_uebersprungen_leer_entfernt_alt(self):
        """Seit 01.09.2026: None (nie angefasst) bekommt keine Zeile — die Figure bleibt
        unangetastet; leer/Whitespace (bewusst geleert) bekommt eine Zeile mit dem
        Sentinel KEIN_ALT, das Import-Script entfernt den Alt-Eintrag (Export = Browser)."""
        rows = pdfix_roundtrip._build_csv_rows(
            {1: "Text eins", 2: "", 3: None, 4: "   ", 5: "Text fuenf"}, "quelle")
        self.assertEqual([r[0] for r in rows], [1, 2, 4, 5])
        by = {r[0]: r[4] for r in rows}
        self.assertEqual(by[1], "Text eins")
        self.assertEqual(by[2], pdfix_roundtrip.KEIN_ALT)
        self.assertEqual(by[4], pdfix_roundtrip.KEIN_ALT)
        self.assertEqual(by[5], "Text fuenf")

    def test_dekorativ_wird_leere_zeile(self):
        rows = pdfix_roundtrip._build_csv_rows({1: "dekorativ"}, "quelle")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][4], "", "dekorativ -> bewusst leerer Alt-Text")

    def test_luecken_bleiben_erhalten(self):
        rows = pdfix_roundtrip._build_csv_rows({3: "Drei", 7: "Sieben"}, "quelle")
        self.assertEqual([r[0] for r in rows], [3, 7],
                         "Laufende Nummern duerfen Luecken haben (lfnr-Matching)")


class TestFitzPfadKeineLeerenAlts(unittest.TestCase):
    """write_alt_texts_to_pdf: leere Texte ueberspringen, kein XObject-/Alt."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "quelle.pdf")
        self.out = os.path.join(self.tmp, "export.pdf")
        self.xrefs = _make_pdf_with_images(self.src, n_images=2)
        self.assertEqual(len(self.xrefs), 2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_leerer_text_wird_nicht_getaggt(self):
        result = write_alt_texts_to_pdf(
            self.src, self.out,
            {self.xrefs[0]: "Ein rotes Quadrat.", self.xrefs[1]: ""})
        self.assertEqual(result["tagged_count"], 1,
                         "Nur das Bild mit Text darf getaggt werden")
        entries = _alt_entries(self.out)
        self.assertEqual(len(entries), 1,
                         f"Genau EIN /Alt-Eintrag erwartet, gefunden: {entries}")
        self.assertTrue(entries[0][1], "/Alt muss am StructElem sitzen")

    def test_kein_alt_am_xobject(self):
        write_alt_texts_to_pdf(self.src, self.out,
                               {self.xrefs[0]: "Ein rotes Quadrat."})
        doc = fitz.open(self.out)
        for xref in self.xrefs:
            obj = doc.xref_object(xref, compressed=True)
            self.assertNotIn("/Alt", obj,
                             "Bild-XObject darf kein /Alt mehr tragen (nicht standardkonform)")
        doc.close()

    def test_dekorativ_ergibt_leeren_alt_am_structelem(self):
        result = write_alt_texts_to_pdf(self.src, self.out,
                                        {self.xrefs[0]: "dekorativ"})
        self.assertEqual(result["tagged_count"], 1)
        entries = [e for e in _alt_entries(self.out) if e[1]]
        self.assertEqual(len(entries), 1)
        self.assertIn(entries[0][2], ("()", "<>"),
                      "dekorativ -> bewusst leerer /Alt am StructElem")


class TestFinalizeExportPdf(unittest.TestCase):
    """Sprache, Titel, DisplayDocTitle, Waisen-Cleanup."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "quelle.pdf")
        self.out = os.path.join(self.tmp, "export.pdf")
        self.xrefs = _make_pdf_with_images(self.src, n_images=1)
        write_alt_texts_to_pdf(self.src, self.out, {self.xrefs[0]: "Ein Bild."})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _catalog_key(self, path, key):
        doc = fitz.open(path)
        val = doc.xref_get_key(doc.pdf_catalog(), key)
        doc.close()
        return val

    def test_sprache_wird_gesetzt(self):
        info = finalize_export_pdf(self.out, fallback_title="egal")
        self.assertTrue(info["lang_set"])
        typ, val = self._catalog_key(self.out, "Lang")
        self.assertEqual(typ, "string")
        self.assertIn("de-DE", val)

    def test_vorhandene_sprache_bleibt(self):
        doc = fitz.open(self.out)
        doc.xref_set_key(doc.pdf_catalog(), "Lang", "(en-US)")
        tmp2 = self.out + ".tmp"
        doc.save(tmp2)
        doc.close()
        os.replace(tmp2, self.out)
        info = finalize_export_pdf(self.out, fallback_title="egal")
        self.assertFalse(info["lang_set"], "Sprachangabe des Autors muss erhalten bleiben")
        self.assertIn("en-US", self._catalog_key(self.out, "Lang")[1])

    def test_titel_prioritaet_explizit_gewinnt(self):
        finalize_export_pdf(self.out, title="Mein Exportname", fallback_title="dateiname")
        doc = fitz.open(self.out)
        self.assertEqual(doc.metadata.get("title"), "Mein Exportname")
        doc.close()

    def test_titel_vorhandener_bleibt_ohne_expliziten(self):
        doc = fitz.open(self.out)
        meta = dict(doc.metadata)
        meta["title"] = "Autorentitel"
        doc.set_metadata(meta)
        tmp2 = self.out + ".tmp"
        doc.save(tmp2)
        doc.close()
        os.replace(tmp2, self.out)
        info = finalize_export_pdf(self.out, fallback_title="dateiname")
        self.assertFalse(info["title_set"], "Vorhandener Quell-Titel muss respektiert werden")
        doc = fitz.open(self.out)
        self.assertEqual(doc.metadata.get("title"), "Autorentitel")
        doc.close()

    def test_titel_fallback_dateiname(self):
        info = finalize_export_pdf(self.out, fallback_title="Geschaeftsbericht")
        self.assertTrue(info["title_set"])
        doc = fitz.open(self.out)
        self.assertEqual(doc.metadata.get("title"), "Geschaeftsbericht")
        doc.close()

    def test_displaydoctitle_gesetzt(self):
        finalize_export_pdf(self.out, fallback_title="egal")
        typ, val = self._catalog_key(self.out, "ViewerPreferences")
        self.assertIn(typ, ("dict", "xref"))
        if typ == "dict":
            self.assertIn("/DisplayDocTitle true", val)

    def test_verwaiste_alt_elems_werden_entfernt(self):
        # Verwaisten StructElem mit /Alt einschleusen (wie PowerPoint-Altlasten:
        # /P zeigt auf ein Element im Baum, aber kein /K referenziert ihn).
        doc = fitz.open(self.out)
        orphan = doc.get_new_xref()
        doc.update_object(orphan, "<< /Type /StructElem /S /Figure /Alt (Junk-Altlast) >>")
        tmp2 = self.out + ".tmp"
        doc.save(tmp2)
        doc.close()
        os.replace(tmp2, self.out)
        self.assertEqual(len(_alt_entries(self.out)), 2, "Setup: Waise + echter Eintrag")

        info = finalize_export_pdf(self.out, fallback_title="egal")
        self.assertEqual(info["orphan_alts_removed"], 1)
        entries = _alt_entries(self.out)
        self.assertEqual(len(entries), 1,
                         f"Nur der erreichbare /Alt-Eintrag darf ueberleben: {entries}")
        doc = fitz.open(self.out)
        alts = [doc.xref_object(e[0], compressed=True) for e in entries]
        doc.close()
        self.assertIn("Ein Bild.", alts[0], "Der echte Alt-Text muss erhalten bleiben")


@unittest.skipUnless(pdfix_roundtrip.is_pdfix_available() and _find_heine_test_pdf(),
                     "pdfix-sdk oder Heine-Test-PDF nicht verfuegbar")
class TestImportScriptLfnrMatching(unittest.TestCase):
    """Gehaertetes Import-Script gegen Heines Test-PDF (4 Figures)."""

    @classmethod
    def setUpClass(cls):
        cls.test_pdf = _find_heine_test_pdf()
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_luecke_laesst_figures_unangetastet(self):
        out_dir = os.path.join(self.tmp, "gap")
        before = pdfix_roundtrip.extract_figures_pdfix(self.test_pdf, out_dir)
        self.assertEqual(len(before), 4)
        pdf_out = os.path.join(self.tmp, "gap.pdf")
        # Nur Figure 2 bekommt einen Text — 1, 3, 4 muessen unveraendert bleiben.
        count = pdfix_roundtrip.import_alt_texts_pdfix(
            self.test_pdf, pdf_out, {2: "Nur Figure zwei"}, work_dir=out_dir)
        self.assertEqual(count, 1)
        after = pdfix_roundtrip.extract_figures_pdfix(
            pdf_out, os.path.join(self.tmp, "gap_check"))
        self.assertEqual(after[1]["alt"], "Nur Figure zwei")
        for i in (0, 2, 3):
            self.assertEqual(after[i]["alt"], before[i]["alt"],
                             f"Figure {i+1} ohne CSV-Zeile darf sich nicht aendern")

    def test_ungueltige_lfnr_bricht_laut_ab(self):
        pdf_out = os.path.join(self.tmp, "fail.pdf")
        with self.assertRaises(RuntimeError,
                               msg="lfnr ausserhalb der Figure-Anzahl muss abbrechen"):
            pdfix_roundtrip.import_alt_texts_pdfix(
                self.test_pdf, pdf_out, {99: "Versatz-Test"},
                work_dir=os.path.join(self.tmp, "fail"))
        self.assertFalse(os.path.exists(pdf_out),
                         "Bei Abbruch darf KEINE Ausgabedatei entstehen")

    def test_leere_texte_erzeugen_keine_csv_zeilen(self):
        out_dir = os.path.join(self.tmp, "empty")
        pdf_out = os.path.join(self.tmp, "empty.pdf")
        before = pdfix_roundtrip.extract_figures_pdfix(self.test_pdf, out_dir)
        count = pdfix_roundtrip.import_alt_texts_pdfix(
            self.test_pdf, pdf_out, {1: "", 2: "", 3: "", 4: ""}, work_dir=out_dir)
        self.assertEqual(count, 0, "Nur leere Texte -> nichts zu setzen")
        after = pdfix_roundtrip.extract_figures_pdfix(
            pdf_out, os.path.join(self.tmp, "empty_check"))
        self.assertEqual([f["alt"] for f in after], [f["alt"] for f in before],
                         "PDF muss unveraendert sein")


if __name__ == "__main__":
    unittest.main(verbosity=2)
