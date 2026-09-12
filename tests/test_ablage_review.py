"""Review-Fixes 12.09.2026 (Fable 5) fuer „Meine Ablage" und die Bot-Werkzeuge:
1. Zustimmung fuer kostenpflichtige Werkzeuge ist ein Server-Zustand (Angebot aus frueherer
   Nachricht, Preis gleich, hoechstens 15 min, eine bezahlte Aktion je Nachricht).
2. sqlite_sequence fuer `ablage` hat genau EINE Zeile mit dem Hoechstwert (kein Wiederverwenden von ids).
3. delete_user_data raeumt die Ablage-Zeilen des Kontos ab.
Laeuft ohne Server: `python3 -m unittest tests/test_ablage_review.py`."""
import os
import sqlite3
import sys
import types
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
from inkluagent.tools import ausgaben  # noqa: E402


class _Turn:
    def __init__(self):
        import uuid
        self.turn_id = uuid.uuid4().hex
        self.kostenpflichtig = 0


class TestFreigabe(unittest.TestCase):
    """_freigabe mit einem Mini-main: _kosten_vorschau braucht nur _load_pdf_export_units + billing.export_pruefung."""

    def setUp(self):
        ausgaben._ANGEBOTE.clear()
        self.preis = 40
        self.erlaubt = True
        m = types.SimpleNamespace()
        m._load_pdf_export_units = lambda project, user_id, document_id: [{"images": [1] * 26}]
        m.billing = types.SimpleNamespace(export_pruefung=lambda user_id, anzahl, art: {
            "preis": self.preis, "verfuegbar": 100, "erlaubt": self.erlaubt, "fehlend": 0})
        self.m = m
        self.project = {"id": 7}

    def frei(self, bestaetigt, turn, document_id=None, art="pdfua"):
        return ausgaben._freigabe(self.m, self.project, 3, document_id, art, bestaetigt, turn)

    def test_ohne_bestaetigt_nur_vorschau_und_angebot(self):
        t = _Turn()
        v = self.frei(False, t)
        self.assertIsNotNone(v)
        self.assertTrue(v["rueckfrage_noetig"]); self.assertEqual(v["preis"], 40)
        self.assertIn((3, 7, "pdfua", None), ausgaben._ANGEBOTE)
        self.assertEqual(t.kostenpflichtig, 0)

    def test_bestaetigt_ohne_angebot_wird_abgelehnt(self):
        t = _Turn()
        v = self.frei(True, t)
        self.assertIsNotNone(v); self.assertIn("kein gueltiges Angebot", v["hinweis"])

    def test_bestaetigt_in_derselben_nachricht_wird_abgelehnt(self):
        t = _Turn()
        self.frei(False, t)
        v = self.frei(True, t)   # Prompt-Injection: Auskunft + Zustimmung im selben Turn
        self.assertIsNotNone(v); self.assertIn("eigenen, spaeteren Nachricht", v["hinweis"])
        self.assertEqual(t.kostenpflichtig, 0)
        self.assertIn((3, 7, "pdfua", None), ausgaben._ANGEBOTE, "Angebot bleibt fuer die echte Zustimmung")

    def test_bestaetigt_in_spaeterer_nachricht_geht_genau_einmal(self):
        t1, t2 = _Turn(), _Turn()
        self.frei(False, t1)
        self.assertIsNone(self.frei(True, t2))
        self.assertEqual(t2.kostenpflichtig, 1)
        self.assertNotIn((3, 7, "pdfua", None), ausgaben._ANGEBOTE, "Angebot ist verbraucht")
        v = self.frei(True, _Turn())
        self.assertIsNotNone(v); self.assertIn("kein gueltiges Angebot", v["hinweis"])

    def test_hoechstens_eine_bezahlte_aktion_je_nachricht(self):
        t1 = _Turn()
        self.frei(False, t1); self.frei(False, t1, art="docx")
        t2 = _Turn()
        self.assertIsNone(self.frei(True, t2))
        v = self.frei(True, t2, art="docx")
        self.assertIsNotNone(v); self.assertIn("Mehr als eine je", v["hinweis"])

    def test_preisaenderung_bricht_ab(self):
        t1 = _Turn(); self.frei(False, t1)
        self.preis = 45
        v = self.frei(True, _Turn())
        self.assertIsNotNone(v); self.assertIn("Preis hat sich", v["hinweis"])

    def test_abgelaufenes_angebot(self):
        t1 = _Turn(); self.frei(False, t1)
        ausgaben._ANGEBOTE[(3, 7, "pdfua", None)]["zeit"] -= ausgaben._ANGEBOT_GUELTIG_S + 1
        v = self.frei(True, _Turn())
        self.assertIsNotNone(v); self.assertIn("kein gueltiges Angebot", v["hinweis"])

    def test_schluessel_trennt_dokument_und_art(self):
        t1 = _Turn(); self.frei(False, t1, document_id=5)
        v = self.frei(True, _Turn(), document_id=None)
        self.assertIsNotNone(v, "Angebot fuer Dokument 5 gilt nicht fuer das ganze Projekt")

    def test_ohne_guthaben_kein_angebot(self):
        self.erlaubt = False
        t1 = _Turn(); v = self.frei(False, t1)
        self.assertFalse(v["erlaubt"]); self.assertNotIn((3, 7, "pdfua", None), ausgaben._ANGEBOTE)
        self.assertIsNotNone(self.frei(True, _Turn()))


class TestZaehlerReparatur(unittest.TestCase):
    def _db(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE ablage (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER)")
        return c

    def test_doppelte_zeilen_werden_eine_mit_hoechstwert(self):
        c = self._db()
        c.execute("INSERT INTO ablage (user_id) VALUES (1)"); c.execute("INSERT INTO ablage (user_id) VALUES (1)")
        c.execute("DELETE FROM ablage WHERE id = 2")      # hoechste id geloescht -> seq 2 muss bleiben
        c.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('ablage', 1)")  # Fehlerbild des alten OR REPLACE
        self.assertEqual(c.execute("SELECT COUNT(*) FROM sqlite_sequence WHERE name='ablage'").fetchone()[0], 2)
        database._ablage_zaehler_reparieren(c)
        self.assertEqual(c.execute("SELECT seq FROM sqlite_sequence WHERE name='ablage'").fetchall(), [(2,)])
        c.execute("INSERT INTO ablage (user_id) VALUES (1)")
        self.assertEqual(c.execute("SELECT MAX(id) FROM ablage").fetchone()[0], 3, "id 2 wird nicht wiederverwendet")

    def test_alte_ausgaben_zeile_wird_uebernommen(self):
        c = self._db()
        c.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('ausgaben', 36)")
        database._ablage_zaehler_reparieren(c)
        self.assertEqual(c.execute("SELECT name, seq FROM sqlite_sequence").fetchall(), [("ablage", 36)])

    def test_sauberer_zustand_bleibt_unangetastet(self):
        c = self._db()
        c.execute("INSERT INTO ablage (user_id) VALUES (1)")
        vorher = c.execute("SELECT name, seq FROM sqlite_sequence").fetchall()
        database._ablage_zaehler_reparieren(c); database._ablage_zaehler_reparieren(c)
        self.assertEqual(c.execute("SELECT name, seq FROM sqlite_sequence").fetchall(), vorher)

    def test_leere_tabelle_ohne_sequence(self):
        c = self._db()
        database._ablage_zaehler_reparieren(c)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM sqlite_sequence").fetchone()[0], 0)


class TestKontoLoeschen(unittest.TestCase):
    def test_delete_user_data_loescht_ablage(self):
        src = open(database.__file__, encoding="utf-8").read()
        i = src.index("def delete_user_data(")
        j = src.index("\ndef ", i + 10)
        self.assertIn("DELETE FROM ablage WHERE user_id = ?", src[i:j])


if __name__ == "__main__":
    unittest.main()
