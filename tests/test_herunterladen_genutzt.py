"""Abrechnung nach Michaels Mail „Feedback 202609230 - 1“ (30.09.2026), Punkte 11 und 12, und schon getaggte PDFs
(Messlauf 30.09.2026) — ohne Datenbank und ohne PDFix:
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_herunterladen_genutzt.py -v

Prueft:
  - Tagging kostet 20 Credits je Seite (Punkt 11);
  - Herunterladen kostet nur, was bearbeitet wurde: Alt-Texte (KI oder von Hand) und Quickinfos; EIN Grundpreis, auch wenn
    beides bearbeitet ist; ohne Bearbeitung 0 — auch bei getaggten Dateien (das Tagging ist beim Lauf bezahlt) (Punkt 12);
  - was „bearbeitet“ heisst (Vergleich mit dem, was schon in der Datei steht);
  - derselbe schon bezahlte Stand kostet beim zweiten Herunterladen nichts (keine Doppelabbuchung), ein geaenderter wieder;
  - eine schon getaggte Quelle wird erkannt (kein Tagging, keine Credits), eine selbst getaggte mit ungetaggter Rohdatei nicht.
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import billing  # noqa: E402


def _pdf(pfad, tags):
    import fitz
    doc = fitz.open()
    doc.new_page()
    if tags:
        root = doc.get_new_xref()
        dok = doc.get_new_xref()
        doc.update_object(dok, f"<< /Type /StructElem /S /Document /P {root} 0 R /K [] >>")
        doc.update_object(root, f"<< /Type /StructTreeRoot /K [ {dok} 0 R ] >>")
        doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    doc.save(pfad)
    doc.close()


class Preise(unittest.TestCase):
    def test_tagging_20_je_seite(self):
        self.assertEqual(billing.AKTIONS_PREISE["pdf_tagging"], 20)
        self.assertEqual(billing.aktion_preis("pdf_tagging", 10), 200)

    def test_alt_text_preis_unveraendert(self):
        """Steve 30.09.2026: der KI-Alt-Text bleibt bei 5 Credits (Michaels „20 pro Bild“ klaert er selbst)."""
        self.assertEqual(billing.AKTIONS_PREISE["bild_generierung"], 5)

    def test_download_nur_bearbeitetes(self):
        p = billing.pdf_download_preis
        self.assertEqual(p(0, 0), {"preis": 0, "pdf": 0, "formular": 0})
        self.assertEqual(p(1, 0)["preis"], 25 + 5)
        self.assertEqual(p(26, 0)["preis"], 25 + 15)            # wie das Beispiel auf der Preisseite
        self.assertEqual(p(0, 13), {"preis": 25 + 2, "pdf": 0, "formular": 27})
        # beides bearbeitet: EIN Grundpreis, nie zwei (keine doppelte Abrechnung)
        self.assertEqual(p(26, 13), {"preis": 25 + 15 + 2, "pdf": 40, "formular": 2})
        self.assertEqual(p(-3, None)["preis"], 0)


class Bearbeitet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
        except Exception as e:  # ausserhalb des Containers
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.m = main

    def _img(self, **kw):
        d = {"id": 1, "alt_text": "", "alt_text_edited": None, "original_alt": "", "image_type": "unknown", "status": "pending"}
        d.update(kw)
        return d

    def test_regeln(self):
        b = self.m._alt_text_bearbeitet
        self.assertFalse(b(self._img()))                                            # nie angefasst
        self.assertTrue(b(self._img(alt_text="Ein Hund im Garten", status="done")))  # per KI erzeugt
        self.assertTrue(b(self._img(alt_text_edited="Von Hand geschrieben")))       # von Hand
        self.assertFalse(b(self._img(original_alt="Logo der Firma")))                # Text stand schon in der Datei
        self.assertFalse(b(self._img(original_alt="Logo", alt_text_edited=" Logo ")))  # „bearbeitet“, aber gleich
        self.assertTrue(b(self._img(original_alt="Logo", alt_text_edited="")))      # bewusst geleert: Export entfernt ihn
        self.assertFalse(b(self._img(alt_text_edited="")))                          # geleert, war nie etwas da
        self.assertTrue(b(self._img(image_type="dekorativ", status="done")))         # KI: dekorativ
        self.assertFalse(b(self._img(original_alt="dekorativ", image_type="dekorativ", status="done")))  # Autor: dekorativ
        self.assertFalse(b(self._img(alt_text="Fehler bei der Analyse: Zeitueberschreitung")))  # Fehlertext wird nie geschrieben


class Plan(unittest.TestCase):
    """_pdf_export_plan mit einer Datenbank im Speicher (nur formularfelder/documents) — keine echte Datenbank."""

    @classmethod
    def setUpClass(cls):
        try:
            import main
        except Exception as e:
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.m = main

    def setUp(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE formularfelder (id INTEGER PRIMARY KEY, document_id INTEGER, anker TEXT, quickinfo TEXT, quickinfo_original TEXT)")
        self.db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, export_bezahlt TEXT DEFAULT '')")
        for i in (1, 2, 3):
            self.db.execute("INSERT INTO documents (id) VALUES (?)", (i,))
        self.db.commit()

        class _Conn:
            """Verbindung, deren close() die Speicher-Datenbank offen laesst."""
            def __init__(s, c):
                s.c = c

            def execute(s, *a):
                return s.c.execute(*a)

            def commit(s):
                s.c.commit()

            def close(s):
                pass
        self.p_db = mock.patch.object(self.m, "get_db", side_effect=lambda: _Conn(self.db))
        self.p_pr = mock.patch.object(self.m.billing, "preis_pruefung",
                                      side_effect=lambda uid, preis: {"preis": preis, "verfuegbar": None, "erlaubt": True, "fehlend": 0})
        self.p_db.start()
        self.p_pr.start()

    def tearDown(self):
        self.p_db.stop()
        self.p_pr.stop()
        self.db.close()

    def _unit(self, doc_id, getaggt, images, bezahlt=""):
        return {"doc": {"id": doc_id, "getaggt": 1 if getaggt else 0, "original_filename": f"d{doc_id}.pdf", "export_bezahlt": bezahlt},
                "images": images}

    def _bild(self, i, **kw):
        d = {"id": i, "alt_text": "", "alt_text_edited": None, "original_alt": "", "image_type": "unknown", "status": "pending"}
        d.update(kw)
        return d

    def test_getaggt_ohne_bearbeitung_kostet_nichts(self):
        """Punkt 12: das Tagging nie beim Herunterladen — getaggt, nichts bearbeitet = 0 (vorher 25 + 5 je 10 Bilder)."""
        plan = self.m._pdf_export_plan(0, [self._unit(1, True, [self._bild(1), self._bild(2), self._bild(3, original_alt="Logo")])])
        self.assertEqual(len(plan["getaggt"]), 1)
        self.assertEqual(plan["preis"], 0)
        self.assertEqual((plan["alt_bearbeitet"], plan["qi_bearbeitet"]), (0, 0))

    def test_getaggt_mit_bearbeiteten_alt_texten(self):
        bilder = [self._bild(i, alt_text=f"KI-Text {i}", status="done") for i in range(1, 12)] + [self._bild(99)]
        plan = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual(plan["alt_bearbeitet"], 11)
        self.assertEqual(plan["preis"], 25 + 10)      # 11 bearbeitete Bilder = 2 angefangene Zehner
        self.assertEqual((plan["preis_pdf"], plan["preis_qi"]), (35, 0))

    def test_quickinfos_nur_bearbeitete(self):
        self.db.executemany("INSERT INTO formularfelder (document_id, anker, quickinfo, quickinfo_original) VALUES (?,?,?,?)", [
            (2, "a", "Vorname eingeben", ""),            # bearbeitet (KI/Stammdaten/Hand)
            (2, "b", "Aus der Datei", "Aus der Datei"),  # stand schon in der Datei
            (2, "c", "", "Alt"),                         # geleert: Export schreibt nur Felder mit Text
            (2, "d", "Neu formuliert", "Alt"),           # geaendert
        ])
        plan = self.m._pdf_export_plan(0, [self._unit(2, False, [])])
        self.assertEqual(plan["qi_bearbeitet"], 2)
        self.assertEqual(len(plan["mit_qi"]), 1)
        self.assertEqual(plan["preis"], 25 + 1)
        # ungetaggt, nur Quickinfos aus der Datei: unveraendert, 0 Credits
        self.db.execute("DELETE FROM formularfelder WHERE anker IN ('a','d')")
        plan = self.m._pdf_export_plan(0, [self._unit(2, False, [])])
        self.assertEqual(len(plan["unveraendert"]), 1)
        self.assertEqual(plan["preis"], 0)

    def test_ungetaggt_alt_texte_zaehlen_nicht(self):
        """Ohne Tags kommen keine Alt-Texte in die Datei — also kosten sie beim Herunterladen auch nichts."""
        plan = self.m._pdf_export_plan(0, [self._unit(3, False, [self._bild(1, alt_text_edited="Von Hand")])])
        self.assertEqual(plan["preis"], 0)
        self.assertEqual(len(plan["unveraendert"]), 1)

    def test_zip_ein_grundpreis(self):
        self.db.execute("INSERT INTO formularfelder (document_id, anker, quickinfo, quickinfo_original) VALUES (2, 'a', 'Neu', '')")
        units = [self._unit(1, True, [self._bild(1, alt_text="KI", status="done")]), self._unit(2, False, []), self._unit(3, False, [])]
        plan = self.m._pdf_export_plan(0, units)
        self.assertEqual((plan["alt_bearbeitet"], plan["qi_bearbeitet"]), (1, 1))
        self.assertEqual(plan["preis"], 25 + 5 + 1)   # EIN Grundpreis fuer das ganze ZIP
        self.assertEqual((len(plan["getaggt"]), len(plan["mit_qi"]), len(plan["unveraendert"])), (1, 1, 1))

    def test_keine_doppelabbuchung(self):
        """Derselbe Stand, schon bezahlt heruntergeladen: 0. Aendert sich etwas, gilt wieder der volle Preis."""
        bilder = [self._bild(1, alt_text="KI-Text", status="done")]
        plan = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual(plan["preis"], 30)
        self.m._export_bezahlt_merken(plan, [1])
        stand = self.db.execute("SELECT export_bezahlt FROM documents WHERE id = 1").fetchone()[0]
        self.assertTrue(stand)
        plan2 = self.m._pdf_export_plan(0, [self._unit(1, True, bilder, bezahlt=stand)])
        self.assertEqual((plan2["preis"], plan2["schon_bezahlt"]), (0, 1))
        bilder[0]["alt_text_edited"] = "Nachgebessert"
        plan3 = self.m._pdf_export_plan(0, [self._unit(1, True, bilder, bezahlt=stand)])
        self.assertEqual((plan3["preis"], plan3["schon_bezahlt"]), (30, 0))
        # Schalter aus: jedes Herunterladen kostet (Stand vor dem 30.09.2026)
        with mock.patch.object(self.m.billing, "GLEICHER_STAND_KOSTENLOS", False):
            self.assertEqual(self.m._pdf_export_plan(0, [self._unit(1, True, bilder[:0] + [self._bild(1, alt_text="KI-Text", status="done")], bezahlt=stand)])["preis"], 30)


class SchonGetaggt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tagging_api
        except Exception as e:
            raise unittest.SkipTest(f"tagging_api nicht ladbar: {e}")
        cls.ta = tagging_api

    def test_quelle(self):
        with tempfile.TemporaryDirectory() as d:
            mit, ohne = os.path.join(d, "mit.pdf"), os.path.join(d, "ohne.pdf")
            _pdf(mit, True)
            _pdf(ohne, False)
            q = self.ta.quelle_getaggt
            self.assertTrue(q({"original_path": mit, "roh_path": "", "getaggt": 1}))      # Kunde laedt getaggte PDF hoch
            self.assertFalse(q({"original_path": ohne, "roh_path": "", "getaggt": 0}))
            self.assertTrue(q({"original_path": mit, "roh_path": "", "getaggt": None}))   # Altbestand: aus der Datei
            # selbst getaggt: Arbeitsdatei hat Tags, die Rohdatei nicht -> Neu taggen bleibt moeglich
            self.assertFalse(q({"original_path": mit, "roh_path": ohne, "getaggt": 1}))
            # Rohdatei war schon getaggt (z. B. vor dem 30.09. trotzdem „getaggt“) -> auch Neu taggen taggt nichts
            self.assertTrue(q({"original_path": mit, "roh_path": mit, "getaggt": 1}))

    def test_text(self):
        self.assertIn("schon getaggt", self.ta.schon_getaggt_text())
        self.assertIn("berechnen nichts", self.ta.schon_getaggt_text())


if __name__ == "__main__":
    unittest.main()
