"""Testfassung herunterladen, professionelles Tagging sperrbar, Speicher-Hygiene (09.10.2026, Steve nach Absprache mit
Michael Karbe) — auf einer WEGWERF-Datenbank mit Wegwerf-Ordnern, PDFix wird nie aufgerufen:
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_testfassung_download.py -v

Prüft: Download nur für den Besitzer (fremd 404, ohne Testfassung 404), Dateiname <Originalname>_Testfassung_mit_
Wasserzeichen.pdf, Datei bleibt nach dem Herunterladen liegen; Schalter TAGGING_PROFESSIONELL aus = Knopf-Endpunkt 403,
lauf_synchron (Kette/Chatbot) und _lauf_sync (Sicherheitsnetz) laufen nicht, nie Credits; an = Lauf startet; Hygiene:
Testfassung weg nach erfolgreichem Barrierefrei-Machen, beim Löschen von Dokument und Projekt, nach 30 Tagen (frische
und laufende bleiben), ein neuer Testlauf überschreibt.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="testfassung-")
os.environ["INKLUDOCS_DB"] = os.path.join(_TMP, "test.db")

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402

assert database.DB_PATH.startswith(_TMP)
database.init_db()


class Grundlage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import funktionen
            import main
            import tagging_api
            from fastapi.testclient import TestClient
        except Exception as e:  # ausserhalb des Containers
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.ta, cls.fk = main, tagging_api, funktionen
        # SICHERHEIT: nur im Wegwerf-Verzeichnis anlegen und loeschen
        cls._alt = (main.RESULTS_DIR, main.UPLOAD_DIR, tagging_api._d.results_dir)
        main.RESULTS_DIR = os.path.join(_TMP, "results")
        main.UPLOAD_DIR = _TMP
        tagging_api._d.results_dir = main.RESULTS_DIR
        os.makedirs(main.RESULTS_DIR, exist_ok=True)
        cls.TestClient = TestClient

    @classmethod
    def tearDownClass(cls):
        cls.main.RESULTS_DIR, cls.main.UPLOAD_DIR, cls.ta._d.results_dir = cls._alt

    def konto(self, mail):
        uid = database.create_user(mail, "geheim-12345", "Test " + mail.split("@")[0])
        c = self.TestClient(self.main.app)
        c.cookies.set("token", self.main.create_token(uid, mail, 0))
        return uid, c

    def projekt(self, uid, name="Info-Brief Herbst 2026.pdf", seiten=2):
        import fitz
        ordner = os.path.join(_TMP, "uploads", str(uid))
        os.makedirs(ordner, exist_ok=True)
        pfad = os.path.join(ordner, f"{time.time_ns()}.pdf")
        d = fitz.open()
        for _ in range(seiten):
            d.new_page()
        d.save(pfad)
        d.close()
        conn = database.get_db()
        cur = conn.execute("INSERT INTO projects (user_id, name, filename, original_path, status, tool, project_type) VALUES (?,?,?,?,?,?,?)",
                           (uid, "Testfassung", name, pfad, "extracted", "pdf", "pdf"))
        pid = cur.lastrowid
        cur = conn.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?,?,?,?)",
                           (pid, 1, name, pfad))
        did = cur.lastrowid
        conn.commit()
        conn.close()
        return pid, did, pfad

    def lege_testfassung_an(self, uid, pid, did, inhalt=b"%PDF-1.7 Testfassung mit Wasserzeichen", alter_tage=0, fehler=None):
        ordner, pdf, meta = self.ta.test_pfade(uid, pid, did)
        os.makedirs(ordner, exist_ok=True)
        if not fehler:
            with open(pdf, "wb") as f:
                f.write(inhalt)
            with open(pdf + ".struktur.json", "w") as f:
                f.write("{}")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"zeit": "2026-10-09 10:31:15", "seiten": 2, "struktur": {"elemente": 3}, "verapdf": None,
                       **({"fehler": fehler} if fehler else {})}, f)
        if alter_tage:
            alt = time.time() - alter_tage * 86400
            for p in (pdf, meta, pdf + ".struktur.json"):
                if os.path.exists(p):
                    os.utime(p, (alt, alt))
        return ordner, pdf, meta

    def credits_gebucht(self, uid):
        conn = database.get_db()
        try:
            return conn.execute("SELECT COUNT(*) FROM usage_events WHERE user_id = ? AND quelle = 'tagging'", (uid,)).fetchone()[0]
        except Exception:  # noqa: BLE001 — ohne Tabelle: nichts gebucht
            return 0
        finally:
            conn.close()


class Download(Grundlage):
    def test_besitzer_laedt_mit_dateiname_und_datei_bleibt(self):
        uid, c = self.konto("dl-besitzer@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        _o, pdf, _m = self.lege_testfassung_an(uid, pid, did)
        stand = c.get(f"/api/projects/{pid}/documents/{did}/tagging").json()["test"]
        self.assertTrue(stand["datei_verfuegbar"])
        self.assertEqual(stand["aufbewahrung_tage"], self.ta.TESTFASSUNG_AUFBEWAHRUNG_TAGE)
        r = c.get(f"/api/projects/{pid}/documents/{did}/tagging/test/datei")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "application/pdf")
        self.assertEqual(r.content, b"%PDF-1.7 Testfassung mit Wasserzeichen")
        self.assertIn("Info-Brief Herbst 2026_Testfassung_mit_Wasserzeichen.pdf",
                      r.headers["content-disposition"].replace("%20", " "))
        self.assertTrue(os.path.isfile(pdf), "nach dem Herunterladen bleibt die Testfassung (zweites Holen)")
        self.assertEqual(c.get(f"/api/projects/{pid}/documents/{did}/tagging/test/datei").status_code, 200)

    def test_fremd_und_unbekannt_404(self):
        uid, c = self.konto("dl-eigen@testfassung.invalid")
        _fid, cf = self.konto("dl-fremd@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        self.lege_testfassung_an(uid, pid, did)
        r = cf.get(f"/api/projects/{pid}/documents/{did}/tagging/test/datei")
        self.assertEqual(r.status_code, 404)
        self.assertNotEqual(r.headers.get("content-type"), "application/pdf")
        self.assertEqual(c.get(f"/api/projects/{pid}/documents/{did + 999}/tagging/test/datei").status_code, 404)
        self.assertEqual(c.get(f"/api/projects/{pid + 999}/documents/{did}/tagging/test/datei").status_code, 404)
        ohne = self.TestClient(self.main.app)
        self.assertEqual(ohne.get(f"/api/projects/{pid}/documents/{did}/tagging/test/datei").status_code, 401)

    def test_ohne_testfassung_oder_nach_fehler_404(self):
        uid, c = self.konto("dl-ohne@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        self.assertEqual(c.get(f"/api/projects/{pid}/documents/{did}/tagging/test/datei").status_code, 404)
        self.lege_testfassung_an(uid, pid, did, fehler="PDFix hat die Aktion abgebrochen")
        self.assertEqual(c.get(f"/api/projects/{pid}/documents/{did}/tagging/test/datei").status_code, 404)

    def test_pfad_nur_aus_zahlen(self):
        """Der Pfad entsteht aus Konto-, Projekt- und Dokumentnummer — nie aus Anfragedaten."""
        uid, c = self.konto("dl-pfad@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        self.lege_testfassung_an(uid, pid, did)
        for boese in ("../../etc/passwd", "%2e%2e%2f", "1;2"):
            r = c.get(f"/api/projects/{pid}/documents/{boese}/tagging/test/datei")
            self.assertIn(r.status_code, (404, 422), boese)


class Schalter(Grundlage):
    def test_aus_sperrt_knopf_ohne_credits(self):
        uid, c = self.konto("profi-aus@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        gestartet = []
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", False), \
             mock.patch.object(self.ta.pdf_tagging, "verfuegbar", return_value=True), \
             mock.patch.object(self.ta, "_lauf_sync", side_effect=lambda *a, **k: gestartet.append(a)):
            r = c.post(f"/api/projects/{pid}/documents/{did}/tagging")
            self.assertEqual(r.status_code, 403, r.text)
            self.assertIn("professionelle Tagging schalten wir in Kürze frei", r.json()["detail"])
            stand = c.get(f"/api/projects/{pid}/documents/{did}/tagging").json()
            self.assertFalse(stand["professionell"])
            self.assertFalse(self.fk.fuer_oberflaeche()["tagging_professionell"])
            # fremdes Dokument bleibt 404, auch bei gesperrtem Schalter
            _f, cf = self.konto("profi-aus-fremd@testfassung.invalid")
            self.assertEqual(cf.post(f"/api/projects/{pid}/documents/{did}/tagging").status_code, 404)
        self.assertEqual(gestartet, [])
        conn = database.get_db()
        d = dict(conn.execute("SELECT tagging_status FROM documents WHERE id = ?", (did,)).fetchone())
        p = dict(conn.execute("SELECT status FROM projects WHERE id = ?", (pid,)).fetchone())
        conn.close()
        self.assertNotEqual(d["tagging_status"], "laeuft")
        self.assertEqual(p["status"], "extracted")
        self.assertEqual(self.credits_gebucht(uid), 0)

    def test_aus_sperrt_kette_und_chatbot_weg(self):
        uid, _c = self.konto("profi-aus-kette@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", False), \
             mock.patch.object(self.ta, "_lauf_sync") as lauf:
            erg = self.ta.lauf_synchron(pid, did, uid, "de", "de")
        self.assertEqual(erg["status"], "fehler")
        self.assertIn("in Kürze frei", erg["grund"])
        lauf.assert_not_called()
        self.assertEqual(self.credits_gebucht(uid), 0)
        from inkluagent.tools import pdf as bot
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", False), \
             mock.patch.object(bot.threading, "Thread") as faden:
            r = bot.barrierefrei_machen(pid, uid, did, bestaetigt=True)
        faden.assert_not_called()
        self.assertTrue(r["result"]["gesperrt"])
        self.assertIn("testweise_taggen", r["result"]["hinweis"])
        from inkluagent.prompts import system_pdf
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", False):
            self.assertIn("noch nicht freigeschaltet", system_pdf.system_pdf())
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", True):
            self.assertNotIn("noch nicht freigeschaltet", system_pdf.system_pdf())

    def test_aus_sicherheitsnetz_im_lauf(self):
        """Auch wenn ein Aufrufer die Wache uebergeht: _lauf_sync taggt nicht und bucht nichts."""
        uid, _c = self.konto("profi-aus-netz@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        billing = mock.Mock()
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", False), \
             mock.patch.object(self.ta.pdf_tagging, "taggen") as taggen, \
             mock.patch.object(self.ta._d, "billing", billing):
            self.ta._lauf_sync(pid, did, uid, 40, "de", "extracted", "de")
        taggen.assert_not_called()
        billing.verbuche.assert_not_called()
        conn = database.get_db()
        d = dict(conn.execute("SELECT tagging_status, tagging_bericht FROM documents WHERE id = ?", (did,)).fetchone())
        conn.close()
        self.assertEqual(d["tagging_status"], "fehler")
        self.assertIn("in Kürze frei", json.loads(d["tagging_bericht"])["fehler"])

    def test_an_startet_lauf(self):
        uid, c = self.konto("profi-an@testfassung.invalid")
        pid, did, _p = self.projekt(uid)
        gestartet = []
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", True), \
             mock.patch.object(self.ta.pdf_tagging, "verfuegbar", return_value=True), \
             mock.patch.object(self.ta._d.billing, "aktion_pruefung", return_value={"erlaubt": True, "preis": 40, "verfuegbar": 100}), \
             mock.patch.object(self.ta, "_lauf_sync", side_effect=lambda *a, **k: gestartet.append(a)):
            r = c.post(f"/api/projects/{pid}/documents/{did}/tagging")
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["preis"], 40)
            self.assertTrue(c.get(f"/api/projects/{pid}/documents/{did}/tagging").json()["professionell"])
        for _ in range(100):
            if gestartet:
                break
            time.sleep(0.05)
        self.assertEqual(len(gestartet), 1)
        self.ta._laeuft.pop(did, None)

    def test_vorgabe_aus_ohne_variable(self):
        import importlib
        import funktionen as fk
        alt = os.environ.pop("TAGGING_PROFESSIONELL", None)
        try:
            importlib.reload(fk)
            self.assertFalse(fk.TAGGING_PROFESSIONELL)
            os.environ["TAGGING_PROFESSIONELL"] = "an"
            importlib.reload(fk)
            self.assertTrue(fk.TAGGING_PROFESSIONELL)
            os.environ["TAGGING_PROFESSIONELL"] = "irgendwas"
            importlib.reload(fk)
            self.assertFalse(fk.TAGGING_PROFESSIONELL)
        finally:
            if alt is None:
                os.environ.pop("TAGGING_PROFESSIONELL", None)
            else:
                os.environ["TAGGING_PROFESSIONELL"] = alt
            importlib.reload(fk)


class Hygiene(Grundlage):
    def test_nach_erfolgreichem_barrierefrei_machen_weg(self):
        uid, _c = self.konto("hyg-profi@testfassung.invalid")
        pid, did, quelle = self.projekt(uid)
        ordner, pdf, meta = self.lege_testfassung_an(uid, pid, did)

        def fake_taggen(q, ziel, sprache, arbeitsordner=None, tags_ersetzen=False, testmodus=False):
            shutil.copyfile(q, ziel)
            return {"zeit": "x", "seiten": 2, "vorher": {"elemente": 0}, "nachher": {"elemente": 5}, "tags_ersetzt": False}
        billing = mock.Mock()
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", True), \
             mock.patch.object(self.ta.pdf_struktur_tagging, "aktiv", return_value=False), \
             mock.patch.object(self.ta.pdf_tagging, "taggen", side_effect=fake_taggen), \
             mock.patch.object(self.ta.pdf_tagging, "verapdf", return_value=None), \
             mock.patch.object(self.ta._d, "extract_images_from_pdf", return_value=[]), \
             mock.patch.object(self.ta._d, "bilder_uebernehmen", return_value="pdfix"), \
             mock.patch.object(self.ta._d, "billing", billing):
            self.ta._lauf_sync(pid, did, uid, 40, "de", "extracted", "de")
        conn = database.get_db()
        d = dict(conn.execute("SELECT tagging_status FROM documents WHERE id = ?", (did,)).fetchone())
        conn.close()
        self.assertEqual(d["tagging_status"], "fertig")
        billing.verbuche.assert_called_once()
        for p in (pdf, meta, pdf + ".struktur.json"):
            self.assertFalse(os.path.exists(p), p)

    def test_fehlgeschlagener_lauf_laesst_testfassung(self):
        uid, _c = self.konto("hyg-fehler@testfassung.invalid")
        pid, did, _q = self.projekt(uid)
        _o, pdf, _m = self.lege_testfassung_an(uid, pid, did)
        with mock.patch.object(self.fk, "TAGGING_PROFESSIONELL", True), \
             mock.patch.object(self.ta.pdf_struktur_tagging, "aktiv", return_value=False), \
             mock.patch.object(self.ta.pdf_tagging, "taggen", side_effect=self.ta.pdf_tagging.TaggingFehler("kaputt")):
            self.ta._lauf_sync(pid, did, uid, 40, "de", "extracted", "de")
        self.assertTrue(os.path.isfile(pdf))

    def test_dokument_und_projekt_loeschen(self):
        uid, c = self.konto("hyg-loeschen@testfassung.invalid")
        pid, did, _q = self.projekt(uid)
        _o, pdf, meta = self.lege_testfassung_an(uid, pid, did)
        self.main._dokument_loeschen_sync(uid, pid, did)
        self.assertFalse(os.path.exists(pdf))
        self.assertFalse(os.path.exists(meta))
        pid2, did2, _q2 = self.projekt(uid)
        ordner2, pdf2, _m2 = self.lege_testfassung_an(uid, pid2, did2)
        r = c.delete(f"/api/projects/{pid2}")
        self.assertIn(r.status_code, (200, 204), r.text)
        self.assertFalse(os.path.exists(pdf2))
        self.assertFalse(os.path.exists(ordner2))

    def test_neuer_lauf_ueberschreibt(self):
        uid, _c = self.konto("hyg-ueberschreiben@testfassung.invalid")
        pid, did, _q = self.projekt(uid)
        ordner, pdf, _m = self.lege_testfassung_an(uid, pid, did, inhalt=b"%PDF-1.7 ALT")

        def fake_taggen(q, ziel, sprache, arbeitsordner=None, testmodus=False, tags_ersetzen=False):
            with open(ziel, "wb") as f:
                f.write(b"%PDF-1.7 NEU")
            return {"testmodus": True, "zeit": "y", "seiten": 2, "nachher": {"elemente": 4}}
        with self.ta._start_lock:
            self.ta._test_laeuft[did] = {"seit": time.time(), "user_id": uid}
            self.ta._test_nutzer[uid] = did
        # ohne Bilder: Feld = Datei (09.10.2026) hat nichts einzusetzen, die Testfassung ist die PDFix-Ausgabe
        with mock.patch.object(self.ta.pdf_tagging, "taggen", side_effect=fake_taggen), \
             mock.patch.object(self.ta.pdf_tagging, "verapdf", return_value=None), \
             mock.patch.object(self.ta._d, "extract_images_from_pdf", return_value=[]):
            self.ta._test_sync(pid, did, uid, "de", "de")
        with open(pdf, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.7 NEU")
        self.assertFalse([x for x in os.listdir(ordner) if x.endswith(".tmp") or ".tmp." in x])   # Wegwerf-Ordner weg
        self.assertEqual(sorted(x for x in os.listdir(ordner) if x.endswith(".pdf")), [f"doc{did}_testweise.pdf"])

    def test_nach_30_tagen_weg_frische_und_laufende_bleiben(self):
        uid, _c = self.konto("hyg-30tage@testfassung.invalid")
        pid, d_alt, _q = self.projekt(uid)
        _p2, d_neu, _q2 = self.projekt(uid)
        _p3, d_lauf, _q3 = self.projekt(uid)
        o1, pdf_alt, meta_alt = self.lege_testfassung_an(uid, pid, d_alt, alter_tage=31)
        o2, pdf_neu, _m2 = self.lege_testfassung_an(uid, _p2, d_neu, alter_tage=29)
        o3, pdf_lauf, _m3 = self.lege_testfassung_an(uid, _p3, d_lauf, alter_tage=40)
        anderes = os.path.join(o1, "fremde_datei.txt")   # nur doc<id>-Dateien werden angefasst
        with open(anderes, "w") as f:
            f.write("x")
        with self.ta._start_lock:
            self.ta._test_laeuft[d_lauf] = {"seit": time.time(), "user_id": uid}
        try:
            n = self.ta.testfassungen_aufraeumen()
        finally:
            with self.ta._start_lock:
                self.ta._test_laeuft.pop(d_lauf, None)
        self.assertGreaterEqual(n, 1)
        self.assertFalse(os.path.exists(pdf_alt))
        self.assertFalse(os.path.exists(meta_alt))
        self.assertFalse(os.path.exists(pdf_alt + ".struktur.json"))
        self.assertTrue(os.path.exists(pdf_neu))
        self.assertTrue(os.path.exists(pdf_lauf))
        self.assertTrue(os.path.exists(anderes))
        self.assertEqual(self.ta.TESTFASSUNG_AUFBEWAHRUNG_TAGE, int(os.environ.get("TESTFASSUNG_AUFBEWAHRUNG_TAGE", "30")))

    def test_nicht_in_der_ablage(self):
        uid, c = self.konto("hyg-ablage@testfassung.invalid")
        pid, did, _q = self.projekt(uid)
        self.lege_testfassung_an(uid, pid, did)
        r = c.get("/api/ausgaben")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn("Testfassung", r.text)
        self.assertNotIn("_testweise", r.text)


class Chatbot(Grundlage):
    def test_testweise_taggen_nennt_download(self):
        uid, _c = self.konto("bot-dl@testfassung.invalid")
        pid, did, _q = self.projekt(uid)
        self.lege_testfassung_an(uid, pid, did)
        from inkluagent.tools import oberflaeche
        with mock.patch.object(self.ta, "test_starten_fuer", return_value={"gestartet": True, "document_id": did, "seiten": 2, "preis": 0}):
            r = oberflaeche.testweise_taggen(pid, uid, did)
        self.assertTrue(r["ok"], r)
        self.assertIn("Testfassung herunterladen (mit Wasserzeichen)", r["result"]["hinweis"])
        self.assertIn("30 Tage", r["result"]["hinweis"])
        self.assertEqual(r["anhang"]["download_url"], f"/api/projects/{pid}/documents/{did}/tagging/test/datei")
        self.assertTrue(r["anhang"]["dateiname"].endswith("_Testfassung_mit_Wasserzeichen.pdf"))
        self.assertEqual(r["anhang"]["label"], "testfassung")


if __name__ == "__main__":
    unittest.main()
