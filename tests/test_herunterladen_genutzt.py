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
        teil = lambda d: {k: d[k] for k in ("preis", "pdf", "formular")}
        self.assertEqual(teil(p(0, 0)), {"preis": 0, "pdf": 0, "formular": 0})
        self.assertEqual(p(1, 0)["preis"], 25 + 5)
        self.assertEqual(p(26, 0)["preis"], 25 + 15)            # wie das Beispiel auf der Preisseite
        self.assertEqual(teil(p(0, 13)), {"preis": 25 + 2, "pdf": 0, "formular": 27})
        # beides bearbeitet: EIN Grundpreis, nie zwei (keine doppelte Abrechnung)
        self.assertEqual(teil(p(26, 13)), {"preis": 25 + 15 + 2, "pdf": 40, "formular": 2})
        self.assertEqual(p(-3, None)["preis"], 0)
        # Zusammensetzung fuer den Dialog (Pruefung 30.09.2026, Punkt 8)
        z = p(26, 13)
        self.assertEqual((z["grund"], z["bilder"], z["felder"], z["alt"], z["qi"]), (25, 15, 2, 26, 13))


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
    """_pdf_export_plan und _export_abrechnen mit einer Wegwerf-Datenbank (Datei im Temp-Ordner, nur formularfelder,
    documents, usage_events) — main.get_db und billing.get_db zeigen darauf, die echte Datenbank bleibt unberuehrt."""

    @classmethod
    def setUpClass(cls):
        try:
            import main
        except Exception as e:
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.m = main

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pfad = os.path.join(self.tmp.name, "t.db")
        c = sqlite3.connect(self.pfad)
        c.execute("CREATE TABLE formularfelder (id INTEGER PRIMARY KEY, document_id INTEGER, anker TEXT, quickinfo TEXT, quickinfo_original TEXT)")
        c.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, export_bezahlt TEXT DEFAULT '')")
        c.execute("CREATE TABLE usage_events (id INTEGER PRIMARY KEY, user_id INTEGER, konto_user_id INTEGER, quelle TEXT, aktion TEXT, credits INTEGER, image_id INTEGER)")
        for i in (1, 2, 3):
            c.execute("INSERT INTO documents (id) VALUES (?)", (i,))
        c.commit()
        c.close()

        def verbindung():
            k = sqlite3.connect(self.pfad, timeout=10)
            k.row_factory = sqlite3.Row
            return k
        self.patches = [
            mock.patch.object(self.m, "get_db", side_effect=verbindung),
            mock.patch.object(self.m.billing, "get_db", side_effect=verbindung),
            mock.patch.object(self.m.billing, "_konto_fuer", side_effect=lambda uid: uid),
            mock.patch.object(self.m.billing, "_pakete_abbuchen", side_effect=lambda conn, konto: None),
            mock.patch.object(self.m.billing, "preis_pruefung",
                              side_effect=lambda uid, preis: {"preis": preis, "verfuegbar": None, "erlaubt": True, "fehlend": 0}),
        ]
        for pt in self.patches:
            pt.start()

    def tearDown(self):
        for pt in self.patches:
            pt.stop()
        self.tmp.cleanup()

    def sql(self, q, *a):
        c = sqlite3.connect(self.pfad)
        try:
            r = c.execute(q, a).fetchall()
            c.commit()
            return r
        finally:
            c.close()

    def gebucht(self):
        return self.sql("SELECT COALESCE(SUM(credits), 0) FROM usage_events")[0][0]

    def stand(self, doc_id):
        return self.sql("SELECT export_bezahlt FROM documents WHERE id = ?", doc_id)[0][0] or ""

    def _unit(self, doc_id, getaggt, images, extraction="pdfix"):
        return {"doc": {"id": doc_id, "getaggt": 1 if getaggt else 0, "original_filename": f"d{doc_id}.pdf",
                        "export_bezahlt": self.stand(doc_id), "extraction_method": extraction}, "images": images}

    def _bild(self, i, **kw):
        d = {"id": i, "image_index": i, "alt_text": "", "alt_text_edited": None, "original_alt": "", "image_type": "unknown", "status": "pending"}
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

    def test_nur_bilder_die_der_export_schreibt(self):
        """N5: Bilder ohne laufende Nummer (PDFix-Weg) bzw. ohne xref (PyMuPDF-Weg) schreibt der Export nicht — kein Preis."""
        pdfix = [self._bild(1, alt_text="KI", status="done"), self._bild(2, alt_text="KI", status="done", image_index=0)]
        self.assertEqual(self.m._pdf_export_plan(0, [self._unit(1, True, pdfix)])["alt_bearbeitet"], 1)
        fitz_weg = [self._bild(1, alt_text="KI", status="done", xref=12), self._bild(2, alt_text="KI", status="done", xref=None),
                    self._bild(3, alt_text_edited="", original_alt="Alt", xref=13)]   # geleert: PyMuPDF-Weg entfernt nichts
        with mock.patch("pdf_export.layout_vektorbilder", return_value=set()):
            self.assertEqual(self.m._pdf_export_plan(0, [self._unit(1, True, fitz_weg, extraction="fitz")])["alt_bearbeitet"], 1)

    def test_quickinfos_nur_bearbeitete_und_momentaufnahme(self):
        self.sql("INSERT INTO formularfelder (document_id, anker, quickinfo, quickinfo_original) VALUES "
                 "(2,'a','Vorname eingeben',''),(2,'b','Aus der Datei','Aus der Datei'),(2,'c','','Alt'),(2,'d','Neu formuliert','Alt')")
        u = self._unit(2, False, [])
        plan = self.m._pdf_export_plan(0, [u])
        self.assertEqual(plan["qi_bearbeitet"], 2)
        self.assertEqual(len(plan["mit_qi"]), 1)
        self.assertEqual(plan["preis"], 25 + 1)
        # M1: die Momentaufnahme haengt am Dokument; eine spaetere Aenderung in der Datenbank kommt NICHT in diese Datei
        self.assertEqual(u["quickinfos"], {"a": "Vorname eingeben", "b": "Aus der Datei", "d": "Neu formuliert"})
        self.sql("UPDATE formularfelder SET quickinfo = 'Waehrend des Baus geaendert' WHERE anker = 'b'")
        geschrieben = {}

        class _Erg:
            geschrieben = 3
            warnungen = []

        def schreiber(quelle, ziel, qi, creator=None):
            geschrieben.update(qi)
            open(ziel, "wb").write(open(quelle, "rb").read())
            return _Erg()
        with tempfile.TemporaryDirectory() as d:
            ziel = os.path.join(d, "x.pdf")
            _pdf(ziel, False)
            with mock.patch("formular_export.write_quickinfos_to_pdf", side_effect=schreiber):
                self.m._quickinfos_in_export(u["doc"], ziel, None, u["quickinfos"])
        self.assertEqual(geschrieben.get("b"), "Aus der Datei")
        # ungetaggt, nur Quickinfos aus der Datei: unveraendert, 0 Credits
        self.sql("DELETE FROM formularfelder WHERE anker IN ('a','d')")
        self.sql("UPDATE formularfelder SET quickinfo = 'Aus der Datei' WHERE anker = 'b'")
        plan = self.m._pdf_export_plan(0, [self._unit(2, False, [])])
        self.assertEqual(len(plan["unveraendert"]), 1)
        self.assertEqual(plan["preis"], 0)

    def test_ungetaggt_alt_texte_zaehlen_nicht(self):
        """Ohne Tags kommen keine Alt-Texte in die Datei — also kosten sie beim Herunterladen auch nichts."""
        plan = self.m._pdf_export_plan(0, [self._unit(3, False, [self._bild(1, alt_text_edited="Von Hand")])])
        self.assertEqual(plan["preis"], 0)
        self.assertEqual(len(plan["unveraendert"]), 1)

    def test_zip_ein_grundpreis(self):
        self.sql("INSERT INTO formularfelder (document_id, anker, quickinfo, quickinfo_original) VALUES (2, 'a', 'Neu', '')")
        units = [self._unit(1, True, [self._bild(1, alt_text="KI", status="done")]), self._unit(2, False, []), self._unit(3, False, [])]
        plan = self.m._pdf_export_plan(0, units)
        self.assertEqual((plan["alt_bearbeitet"], plan["qi_bearbeitet"]), (1, 1))
        self.assertEqual(plan["preis"], 25 + 5 + 1)   # EIN Grundpreis fuer das ganze ZIP
        self.assertEqual((len(plan["getaggt"]), len(plan["mit_qi"]), len(plan["unveraendert"])), (1, 1, 1))

    def test_keine_doppelabbuchung(self):
        """Derselbe Stand, schon bezahlt: 0. Aendert sich etwas, gilt wieder der volle Preis."""
        bilder = [self._bild(1, alt_text="KI-Text", status="done")]
        plan = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual(plan["preis"], 30)
        self.assertEqual(self.m._export_abrechnen(7, plan, {1: False}), 30)
        self.assertEqual(self.gebucht(), 30)
        self.assertTrue(self.stand(1).startswith("a="))
        plan2 = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual((plan2["preis"], plan2["schon_bezahlt"]), (0, 1))
        self.assertEqual(self.m._export_abrechnen(7, plan2, {1: False}), 0)
        self.assertEqual(self.gebucht(), 30)
        bilder[0]["alt_text_edited"] = "Nachgebessert"
        plan3 = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual((plan3["preis"], plan3["schon_bezahlt"]), (30, 0))
        with mock.patch.object(self.m.billing, "GLEICHER_STAND_KOSTENLOS", False):
            self.assertEqual(self.m._pdf_export_plan(0, [self._unit(1, True, [self._bild(1, alt_text="KI-Text", status="done")])])["preis"], 30)

    def test_atomarer_anspruch_zwei_gleichzeitige_plaene(self):
        """M2: Zwei Downloads planen denselben Stand, bevor einer bucht (Chatbot + Knopf, ZIP + Einzel). Nur der erste bucht;
        der zweite findet den Stand beansprucht und bucht nichts."""
        bilder = [self._bild(1, alt_text="KI-Text", status="done")]
        p1 = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        p2 = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual((p1["preis"], p2["preis"]), (30, 30))
        self.assertEqual(self.m._export_abrechnen(7, p1, {1: False}), 30)
        self.assertEqual(self.m._export_abrechnen(7, p2, {1: False}), 0)
        self.assertEqual(self.gebucht(), 30)

    def test_teilbuchung_wird_gemerkt(self):
        """N4: Alt-Texte UND Quickinfos bearbeitet, die Quickinfos landen aber nicht in der Datei: nur der Alt-Text-Teil wird
        gebucht UND gemerkt; beim naechsten Herunterladen kostet nur noch der Quickinfo-Teil (mit Grundpreis)."""
        self.sql("INSERT INTO formularfelder (document_id, anker, quickinfo, quickinfo_original) VALUES (1, 'a', 'Neu', '')")
        bilder = [self._bild(1, alt_text="KI-Text", status="done")]
        plan = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual(plan["preis"], 25 + 5 + 1)
        self.assertEqual(self.m._export_abrechnen(7, plan, {1: False}), 30)   # Quickinfos nicht geschrieben
        plan2 = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        self.assertEqual((plan2["alt_bearbeitet"], plan2["qi_bearbeitet"], plan2["preis"]), (0, 1, 26))
        self.assertEqual(self.m._export_abrechnen(7, plan2, {1: True}), 26)
        self.assertEqual(self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])["preis"], 0)

    def test_zip_quickinfos_je_dokument(self):
        """N4 b: im ZIP zaehlt der Quickinfo-Teil eines Dokuments nur, wenn SEINE Quickinfos geschrieben wurden."""
        self.sql("INSERT INTO formularfelder (document_id, anker, quickinfo, quickinfo_original) VALUES (2, 'a', 'Neu', ''),(3, 'b', 'Aus Datei', 'Aus Datei')")
        plan = self.m._pdf_export_plan(0, [self._unit(2, False, []), self._unit(3, False, [])])
        self.assertEqual(plan["preis"], 26)
        # Dokument 3 hat Quickinfos geschrieben (unbearbeitet), Dokument 2 nicht -> nichts buchen
        self.assertEqual(self.m._export_abrechnen(7, plan, {2: False, 3: True}), 0)
        self.assertEqual(self.gebucht(), 0)
        self.assertEqual(self.stand(2), "")

    def test_buchungsfehler_nichts_gemerkt(self):
        """N6: schlaegt die Buchung fehl, gilt der Stand NICHT als bezahlt (vorher: verschluckter Fehler, Stand gemerkt)."""
        bilder = [self._bild(1, alt_text="KI-Text", status="done")]
        plan = self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])
        with mock.patch.object(self.m.billing, "_pakete_abbuchen", side_effect=RuntimeError("Datenbank gesperrt")):
            self.assertEqual(self.m._export_abrechnen(7, plan, {1: False}), 0)
        self.assertEqual((self.gebucht(), self.stand(1)), (0, ""))
        self.assertEqual(self.m._pdf_export_plan(0, [self._unit(1, True, bilder)])["preis"], 30)


class Bremse(unittest.TestCase):
    """H1: hoechstens ein PDF-Herunterladen je Nutzer gleichzeitig, Drosselung je Nutzer."""

    @classmethod
    def setUpClass(cls):
        try:
            import main
        except Exception as e:
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.m = main

    def test_sperre_und_drossel(self):
        m = self.m
        uid = 987654321
        m._export_zeiten.pop(uid, None)
        self.assertEqual(m._export_belegen(uid), "")
        self.assertEqual(m._export_belegen(uid), "laeuft")
        m._export_freigeben(uid)
        with mock.patch.object(m, "EXPORT_DROSSEL_ANZAHL", 3):
            m._export_zeiten[uid] = []
            for _ in range(3):
                self.assertEqual(m._export_belegen(uid), "")
                m._export_freigeben(uid)
            self.assertEqual(m._export_belegen(uid), "drossel")
        m._export_zeiten.pop(uid, None)
        self.assertIn("warte", m._export_belegt_text("laeuft"))
        self.assertIn("Minuten", m._export_belegt_text("drossel"))


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
