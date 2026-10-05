"""EXPRESS-SERVICE Stufe 1 (05.10.2026): Kern express.py mit eigener Wegwerf-Datenbank — Warenkorb, Bestellung mit
Vormerkung, Lieferung bucht genau einmal ab, Storno gibt frei, Besitzpruefungen, Zustandswechsel, Upload-Pruefung,
Frist ruht bei Rueckfrage, Erinnerungen genau einmal, Kontoloeschung.
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_express.py -v
"""
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from datetime import timedelta
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="express_test_")
_DB = os.path.join(_TMP, "test.db")
os.environ["INKLUDOCS_DB"] = _DB

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402


def _eigene_db():
    os.environ["INKLUDOCS_DB"] = _DB
    database.DB_PATH = _DB


_eigene_db()
database.init_db()

import billing  # noqa: E402
import express  # noqa: E402

express.RESULTS_DIR = os.path.join(_TMP, "results")


def _pdf(pfad, seiten=2):
    import fitz
    d = fitz.open()
    for i in range(seiten):
        d.new_page().insert_text((72, 72), f"Seite {i + 1} (fiktiv)")
    d.save(pfad)
    d.close()
    return pfad


def _sql(q, *p):
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    try:
        r = [dict(x) for x in conn.execute(q, p).fetchall()]
        conn.commit()
        return r
    finally:
        conn.close()


PERSON = {"id": 900, "name": "Bearbeiterin Test"}


class Basis(unittest.TestCase):
    def setUp(self):
        _eigene_db()
        billing.ABO_ENFORCEMENT = True
        for t in ("express_verlauf", "express_positionen", "express_auftraege", "usage_events", "paket_abbuchungen",
                  "quota_pakete", "documents", "projects", "users", "system_kv"):
            _sql(f"DELETE FROM {t}")
        _sql("INSERT INTO users (id, email, password_hash, display_name, plan) VALUES (1, 'kundin@beispiel.invalid', 'x', 'Kundin', 'free')")
        _sql("INSERT INTO users (id, email, password_hash, display_name, plan) VALUES (2, 'fremd@beispiel.invalid', 'x', 'Fremd', 'free')")
        self.pid = self._projekt(1, "Projekt A")
        self.d1 = self._dokument(self.pid, "Bericht.pdf", 2)
        self.d2 = self._dokument(self.pid, "Flyer.pdf", 1)
        self.fremd_pid = self._projekt(2, "Fremdes Projekt")
        self.fremd_doc = self._dokument(self.fremd_pid, "Geheim.pdf", 1)

    def _projekt(self, uid, name):
        conn = sqlite3.connect(_DB)
        cur = conn.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type) VALUES (?, ?, '', ?, 'pdf', 'pdf')",
                           (uid, name, name))
        conn.commit()
        conn.close()
        return cur.lastrowid

    def _dokument(self, pid, name, seiten):
        pfad = _pdf(os.path.join(_TMP, f"{pid}_{name}"), seiten)
        conn = sqlite3.connect(_DB)
        cur = conn.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, 1, ?, ?)",
                           (pid, name, pfad))
        conn.commit()
        conn.close()
        return cur.lastrowid

    def _guthaben(self, credits):
        _sql("INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) VALUES (1, ?, ?, 'admin', 'Test', NULL)",
             credits, credits)

    def _bestellen(self, idem="abcdefgh-1"):
        return express.bestellen(1, ansprechpartner="Kundin", telefon="+49 40 123", hinweise="bitte Seite 2",
                                 bedingungen=True, bearbeitung=True, idempotenz=idem)


class Warenkorb(Basis):
    def test_hinzufuegen_preis_und_doppelte(self):
        r = express.dokumente_hinzufuegen(1, [self.d1, self.d2])
        self.assertEqual(r["hinzugefuegt"], 2)
        w = r["warenkorb"]
        self.assertEqual((w["dokumente"], w["seiten"], w["credits"]), (2, 3, 150))   # 3 Seiten x 50 (Platzhalter)
        r = express.dokumente_hinzufuegen(1, [self.d1])
        self.assertEqual(r["hinzugefuegt"], 0)
        self.assertTrue(r["hinweise"])

    def test_fremdes_dokument_wird_abgewiesen(self):
        with self.assertRaises(express.ExpressFehler) as e:
            express.dokumente_hinzufuegen(1, [self.d1, self.fremd_doc])
        self.assertEqual(e.exception.status, 404)
        self.assertEqual(express.warenkorb(1)["dokumente"], 0)

    def test_fremde_position_aendern_oder_entfernen_geht_nicht(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        pos = express.warenkorb(1)["positionen"][0]["id"]
        with self.assertRaises(express.NichtGefunden):
            express.leistung_setzen(2, pos, "pruefen")
        with self.assertRaises(express.NichtGefunden):
            express.position_entfernen(2, pos)
        w = express.leistung_setzen(1, pos, "pruefen")
        self.assertEqual(w["credits"], 50)   # 2 Seiten x 25
        with self.assertRaises(express.ExpressFehler):
            express.leistung_setzen(1, pos, "alles")

    def test_keine_pdf_wird_nicht_aufgenommen(self):
        conn = sqlite3.connect(_DB)
        text = os.path.join(_TMP, "kein.pdf")
        with open(text, "w") as f:
            f.write("kein pdf")
        cur = conn.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, 3, 'x.docx', ?)",
                           (self.pid, text))
        conn.commit()
        conn.close()
        r = express.dokumente_hinzufuegen(1, [cur.lastrowid])
        self.assertEqual(r["hinzugefuegt"], 0)

    def test_grenzen(self):
        express.speichere_einstellungen({"preis_aufbereiten": "50", "preis_pruefen": "25", "frist_stunden": "48",
                                         "max_seiten_auftrag": "2", "max_dokumente_auftrag": "50"})
        with self.assertRaises(express.ExpressFehler):
            express.dokumente_hinzufuegen(1, [self.d1, self.d2])


class Bestellen(Basis):
    def test_ohne_haekchen_kein_auftrag(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        with self.assertRaises(express.ExpressFehler):
            express.bestellen(1, ansprechpartner="K", telefon="", hinweise="", bedingungen=True, bearbeitung=False,
                              idempotenz="abcdefgh-2")
        with self.assertRaises(express.ExpressFehler):
            express.bestellen(1, ansprechpartner="K", telefon="", hinweise="", bedingungen="true", bearbeitung=True,
                              idempotenz="abcdefgh-2")

    def test_zu_wenig_guthaben(self):
        express.dokumente_hinzufuegen(1, [self.d1])   # 100 Credits
        # Free hat 50 im Monat, kein Paket
        with self.assertRaises(express.KeinGuthaben) as e:
            self._bestellen()
        self.assertEqual(e.exception.status, 402)
        self.assertEqual(e.exception.extra["preis"], 100)

    def test_bestellen_merkt_vor_und_ist_idempotent(self):
        express.dokumente_hinzufuegen(1, [self.d1, self.d2])
        self._guthaben(1000)
        vorher = billing.verfuegbare_credits(1)
        r = self._bestellen()
        self.assertTrue(r["neu"])
        self.assertEqual(billing.vorgemerkt(1), 150)
        self.assertEqual(billing.verfuegbare_credits(1), vorher - 150)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM usage_events")[0]["n"], 0)   # noch nichts abgebucht
        nochmal = self._bestellen()
        self.assertFalse(nochmal["neu"])
        self.assertEqual(nochmal["auftrag_id"], r["auftrag_id"])
        self.assertEqual(billing.vorgemerkt(1), 150)
        a = express.auftrag_fuer_kunde(1, r["auftrag_id"])
        self.assertEqual(a["status"], "neu")
        self.assertEqual(a["zustimmung"]["fassung"], express.ZUSTIMMUNG_FASSUNG)
        self.assertTrue(a["zustimmung"]["bearbeitung"].startswith("Ich bin einverstanden"))
        for p in _sql("SELECT original_pfad FROM express_positionen"):
            self.assertTrue(os.path.isfile(p["original_pfad"]))
            self.assertTrue(p["original_pfad"].startswith(express.RESULTS_DIR))
        self.assertEqual(express.warenkorb(1)["dokumente"], 0)   # neuer, leerer Warenkorb

    def test_gleichzeitig_nur_ein_auftrag(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        ergebnisse = []

        def los():
            try:
                ergebnisse.append(self._bestellen("gleich-123456"))
            except Exception as e:  # noqa: BLE001
                ergebnisse.append(e)
        threads = [threading.Thread(target=los) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        ids = {r["auftrag_id"] for r in ergebnisse if isinstance(r, dict)}
        self.assertEqual(len(ids), 1)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_auftraege WHERE status = 'neu'")[0]["n"], 1)

    def test_vormerkung_sperrt_andere_ausgaben(self):
        express.dokumente_hinzufuegen(1, [self.d1])   # 100
        self._guthaben(60)                            # 50 Monat + 60 Paket = 110
        self._bestellen()
        self.assertEqual(billing.verfuegbare_credits(1), 10)
        self.assertFalse(billing.aktion_pruefung(1, "pdf_tagging")["erlaubt"])   # 20 > 10
        self.assertTrue(billing.aktion_pruefung(1, "bild_generierung")["erlaubt"])  # 5 <= 10

    def test_telefon_pruefung(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        with self.assertRaises(express.ExpressFehler):
            express.bestellen(1, ansprechpartner="K", telefon="<script>", hinweise="", bedingungen=True,
                              bearbeitung=True, idempotenz="abcdefgh-3")


class Ablauf(Basis):
    def _auftrag(self, dokumente=None):
        express.dokumente_hinzufuegen(1, dokumente or [self.d1, self.d2])
        self._guthaben(1000)
        return self._bestellen()["auftrag_id"]

    def _ergebnis(self, aid):
        for p in express.auftrag_fuer_verwaltung(aid)["positionen"]:
            with open(_pdf(os.path.join(_TMP, f"erg_{p['id']}.pdf"), 1), "rb") as f:
                inhalt = f.read()
            express.datei_speichern(aid, p["id"], "ergebnis", inhalt, "../../etc/passwd.pdf", PERSON)

    def test_kunde_sieht_nur_eigene(self):
        aid = self._auftrag()
        with self.assertRaises(express.NichtGefunden):
            express.auftrag_fuer_kunde(2, aid)
        self.assertEqual(express.auftraege_des_kunden(2), [])
        with self.assertRaises(express.NichtGefunden):
            express.antwort_kunde(2, aid, "hallo")

    def test_liefern_bucht_genau_einmal_ab(self):
        aid = self._auftrag()
        with self.assertRaises(express.ExpressFehler) as e:
            express.liefern(aid, PERSON)            # Ergebnisse fehlen
        self.assertIn("fehlt", e.exception.text)
        self._ergebnis(aid)
        express.liefern(aid, PERSON)
        self.assertEqual(billing.vorgemerkt(1), 0)
        ev = _sql("SELECT quelle, aktion, credits FROM usage_events")
        self.assertEqual(sum(x["credits"] for x in ev), 150)
        self.assertTrue(all(x["quelle"] == "express" for x in ev))
        with self.assertRaises(express.ExpressFehler) as e:
            express.liefern(aid, PERSON)
        self.assertEqual(e.exception.status, 409)
        self.assertEqual(sum(x["credits"] for x in _sql("SELECT credits FROM usage_events")), 150)
        with self.assertRaises(express.ExpressFehler):
            express.stornieren(aid, PERSON, "zu spaet")   # nach der Lieferung kein Storno

    def test_gleichzeitig_liefern_bucht_einmal(self):
        aid = self._auftrag()
        self._ergebnis(aid)
        fehler = []

        def los():
            try:
                express.liefern(aid, PERSON)
            except express.ExpressFehler as e:
                fehler.append(e.status)
        threads = [threading.Thread(target=los) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sum(x["credits"] for x in _sql("SELECT credits FROM usage_events")), 150)
        self.assertEqual(len(fehler), 3)

    def test_storno_gibt_frei(self):
        aid = self._auftrag()
        with self.assertRaises(express.ExpressFehler):
            express.stornieren(aid, PERSON, "")      # Grund ist Pflicht
        express.stornieren(aid, PERSON, "Kunde wollte nicht mehr")
        self.assertEqual(billing.vorgemerkt(1), 0)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM usage_events")[0]["n"], 0)
        a = express.auftrag_fuer_kunde(1, aid)
        self.assertEqual((a["status"], a["credits_stand"]), ("storniert", "frei"))

    def test_dateiname_wird_nie_pfad(self):
        aid = self._auftrag([self.d1])
        self._ergebnis(aid)
        p = express.auftrag_fuer_verwaltung(aid)["positionen"][0]
        pfad = _sql("SELECT ergebnis_pfad FROM express_positionen")[0]["ergebnis_pfad"]
        self.assertTrue(pfad.startswith(express.ordner(1, aid)))
        self.assertNotIn("..", pfad)
        self.assertEqual(p["ergebnis_name"], "passwd.pdf")

    def test_upload_pruefung(self):
        aid = self._auftrag([self.d1])
        pos = express.auftrag_fuer_verwaltung(aid)["positionen"][0]["id"]
        for inhalt in (b"", b"MZ\x90\x00 kein pdf", b"%PDF-1.7 kaputt"):
            with self.assertRaises(express.ExpressFehler):
                express.datei_speichern(aid, pos, "ergebnis", inhalt, "x.pdf", PERSON)
        with self.assertRaises(express.ExpressFehler):
            express.datei_speichern(aid, pos, "skript", b"%PDF-", "x.pdf", PERSON)
        with self.assertRaises(express.NichtGefunden):
            express.datei_speichern(aid + 1000, pos, "ergebnis", b"%PDF-", "x.pdf", PERSON)
        with mock.patch.object(express, "MAX_ERGEBNIS_BYTES", 10):
            with self.assertRaises(express.ExpressFehler) as e:
                express.datei_speichern(aid, pos, "ergebnis", b"%PDF-" + b"x" * 20, "x.pdf", PERSON)
            self.assertEqual(e.exception.status, 413)

    def test_kunde_laedt_erst_nach_lieferung(self):
        aid = self._auftrag([self.d1])
        self._ergebnis(aid)
        pos = express.auftrag_fuer_verwaltung(aid)["positionen"][0]["id"]
        with self.assertRaises(express.NichtGefunden):
            express.datei_fuer_kunde(1, aid, pos, "ergebnis")
        express.liefern(aid, PERSON)
        pfad, name = express.datei_fuer_kunde(1, aid, pos, "ergebnis")
        self.assertTrue(os.path.isfile(pfad))
        self.assertTrue(name.endswith("(barrierefrei).pdf"))
        with self.assertRaises(express.NichtGefunden):
            express.datei_fuer_kunde(2, aid, pos, "ergebnis")
        with self.assertRaises(express.NichtGefunden):
            express.datei_fuer_kunde(1, aid, pos, "original_pfad")

    def test_rueckfrage_frist_ruht(self):
        aid = self._auftrag()
        express.uebernehmen(aid, PERSON)
        vorher = _sql("SELECT faellig_am FROM express_auftraege WHERE id = ?", aid)[0]["faellig_am"]
        express.rueckfrage(aid, PERSON, "Welche Sprache hat Seite 2?")
        _sql("UPDATE express_auftraege SET rueckfrage_seit = datetime('now', '-3 hours') WHERE id = ?", aid)
        a = express.antwort_kunde(1, aid, "Englisch")
        self.assertEqual(a["status"], "in_arbeit")
        nachher = _sql("SELECT faellig_am FROM express_auftraege WHERE id = ?", aid)[0]["faellig_am"]
        delta = (express._als_dt(nachher) - express._als_dt(vorher)).total_seconds()
        self.assertAlmostEqual(delta, 3 * 3600, delta=120)
        with self.assertRaises(express.ExpressFehler) as e:
            express.antwort_kunde(1, aid, "nochmal")
        self.assertEqual(e.exception.status, 409)

    def test_erinnerung_und_ueberfaellig_je_einmal(self):
        aid = self._auftrag()
        self.assertEqual(express.faellige_meldungen(), [])
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?",
             express._utc(express._jetzt() + timedelta(hours=5)), aid)
        self.assertEqual(express.faellige_meldungen(), [("erinnerung", aid)])
        self.assertEqual(express.faellige_meldungen(), [])
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?",
             express._utc(express._jetzt() - timedelta(hours=1)), aid)
        self.assertEqual(express.faellige_meldungen(), [("ueberfaellig", aid)])
        self.assertEqual(express.faellige_meldungen(), [])

    def test_liste_nach_dringlichkeit(self):
        aid = self._auftrag([self.d1])
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?",
             express._utc(express._jetzt() - timedelta(hours=1)), aid)
        g = express.liste_fuer_verwaltung()
        self.assertEqual([a["id"] for a in g["ueberfaellig"]], [aid])
        self.assertEqual(g["neu"], [])

    def test_nachweis_docx(self):
        aid = self._auftrag()
        a = express.auftrag_fuer_kunde(1, aid)
        ziel = os.path.join(_TMP, "nachweis.docx")
        titel = express.nachweis_docx(a, ziel, lambda t: t)
        import zipfile
        with zipfile.ZipFile(ziel) as z:
            doc = z.read("word/document.xml").decode()
            self.assertIn('w:val="Heading1"', doc)
            self.assertIn("keine Rechnung", doc)
            self.assertIn("<dc:language>de-DE</dc:language>", z.read("docProps/core.xml").decode())
        self.assertIn(str(aid), titel)

    def test_kontoloeschung(self):
        aid = self._auftrag()
        database.delete_user_data(1)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_auftraege")[0]["n"], 0)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_positionen WHERE auftrag_id = ?", aid)[0]["n"], 0)


class Einstellungen(Basis):
    def test_pruefung(self):
        for falsch in ({"preis_aufbereiten": "0"}, {"preis_aufbereiten": "viel"}, {"frist_stunden": "-5"},
                       {"team_mail": "kein mail"}):
            daten = {"preis_aufbereiten": "50", "preis_pruefen": "25", "frist_stunden": "48",
                     "max_seiten_auftrag": "500", "max_dokumente_auftrag": "50"}
            daten.update(falsch)
            with self.assertRaises(express.ExpressFehler):
                express.speichere_einstellungen(daten)
        e = express.speichere_einstellungen({"preis_aufbereiten": "60", "preis_pruefen": "30", "frist_stunden": "72",
                                             "max_seiten_auftrag": "300", "max_dokumente_auftrag": "20",
                                             "team_mail": "team@beispiel.invalid", "preise_festgelegt": True})
        self.assertEqual((e["preis_aufbereiten"], e["frist_stunden"], e["preise_festgelegt"]), (60, 72, True))
        self.assertEqual(express.team_empfaenger("support@beispiel.invalid"), ["team@beispiel.invalid"])

    def test_bearbeiter(self):
        self.assertFalse(express.ist_bearbeiter(2))
        express.bearbeiter_setzen(email="FREMD@beispiel.invalid", an=True)
        self.assertTrue(express.ist_bearbeiter(2))
        self.assertIn("fremd@beispiel.invalid", express.team_empfaenger("support@beispiel.invalid"))
        express.bearbeiter_setzen(user_id=2, an=False)
        self.assertFalse(express.ist_bearbeiter(2))
        with self.assertRaises(express.ExpressFehler):
            express.bearbeiter_setzen(email="niemand@beispiel.invalid")


class Schalter(unittest.TestCase):
    def test_umgebung(self):
        import importlib
        import funktionen
        try:
            for umg, erwartet in (({"EXPRESS_SERVICE": "an", "DEMO_MODE": "off"}, True),
                                  ({"EXPRESS_SERVICE": "", "DEMO_MODE": "off"}, False),
                                  ({"EXPRESS_SERVICE": "an", "DEMO_MODE": "on"}, False)):
                with mock.patch.dict(os.environ, umg):
                    self.assertIs(importlib.reload(funktionen).EXPRESS, erwartet, umg)
        finally:
            importlib.reload(funktionen)


if __name__ == "__main__":
    unittest.main()
