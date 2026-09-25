"""„Testweise taggen“ (Michael Karbe, Feedback 24.09.2026 - 2, Punkt 3; gebaut 25.09.2026) — auf einer WEGWERF-
Datenbank, PDFix wird nicht aufgerufen (Subprozess und Lauf sind ersetzt):
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_testweise_taggen.py -v

Prüft: Testmodus wird erzwungen (Lizenz aus, auch wenn sie an ist), der Testlauf ändert das Dokument nicht,
ein zweiter Start für dasselbe Dokument/denselben Nutzer wird abgewiesen, fremde Dokumente 404, es gibt keinen
Download-Weg für die Testfassung, Löschen des Dokuments räumt Testfassung und Prüfdatei weg.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="testweise-")
os.environ["INKLUDOCS_DB"] = os.path.join(_TMP, "test.db")

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402

assert database.DB_PATH.startswith(_TMP)
database.init_db()

import pdf_tagging  # noqa: E402


class TestModusErzwungen(unittest.TestCase):
    def test_lizenz_aus_im_testlauf(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen["env"] = kw.get("env") or {}
            raise subprocess.TimeoutExpired(cmd, 1)   # Abbruch direkt nach dem Aufruf: uns interessiert nur die Umgebung
        quelle = os.path.join(_TMP, "q.pdf")
        import fitz
        d = fitz.open(); d.new_page(); d.save(quelle); d.close()
        with mock.patch.dict(os.environ, {"PDFIX_TAGGING_LIZENZ": "on"}), \
             mock.patch.object(pdf_tagging, "verfuegbar", return_value=True), \
             mock.patch.object(pdf_tagging.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(pdf_tagging.TaggingFehler):
                pdf_tagging.taggen(quelle, os.path.join(_TMP, "z.pdf"), "de", arbeitsordner=_TMP, testmodus=True)
            self.assertEqual(seen["env"].get("PDFIX_TAGGING_LIZENZ"), "off")
            with self.assertRaises(pdf_tagging.TaggingFehler):
                pdf_tagging.taggen(quelle, os.path.join(_TMP, "z.pdf"), "de", arbeitsordner=_TMP, testmodus=False)
            self.assertEqual(seen["env"].get("PDFIX_TAGGING_LIZENZ"), "on")   # normaler Lauf unverändert


class TestEndpunkte(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            import tagging_api
            from fastapi.testclient import TestClient
        except Exception as e:  # ausserhalb des Containers
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.ta = main, tagging_api
        # SICHERHEIT: Ergebnis- und Upload-Ordner auf das Wegwerf-Verzeichnis umlenken. Die Nutzer-/Projektnummern
        # dieser Datenbank koennen mit echten Konten uebereinstimmen — geloescht werden darf nur hier drin.
        cls._alt = (main.RESULTS_DIR, main.UPLOAD_DIR, tagging_api._d.results_dir)
        main.RESULTS_DIR = os.path.join(_TMP, "results")
        main.UPLOAD_DIR = _TMP
        tagging_api._d.results_dir = main.RESULTS_DIR
        os.makedirs(main.RESULTS_DIR, exist_ok=True)
        cls.uid = database.create_user("testweise@example.invalid", "geheim-12345", "Testweise")
        cls.fremd = database.create_user("fremd-tw@example.invalid", "geheim-12345", "Fremd")
        conn = database.get_db()
        cls.upload = os.path.join(_TMP, "a.pdf")
        import fitz
        d = fitz.open(); d.new_page(); d.new_page(); d.save(cls.upload); d.close()
        cur = conn.execute("INSERT INTO projects (user_id, name, filename, original_path, status, tool, project_type) VALUES (?,?,?,?,?,?,?)",
                           (cls.uid, "Testweise", "a.pdf", cls.upload, "extracted", "pdf", "pdf"))
        cls.pid = cur.lastrowid
        cur = conn.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?,?,?,?)",
                           (cls.pid, 1, "a.pdf", cls.upload))
        cls.did = cur.lastrowid
        conn.commit(); conn.close()
        cls.c = TestClient(main.app)
        cls.c.cookies.set("token", main.create_token(cls.uid, "testweise@example.invalid", 0))
        cls.cf = TestClient(main.app)
        cls.cf.cookies.set("token", main.create_token(cls.fremd, "fremd-tw@example.invalid", 0))

    @classmethod
    def tearDownClass(cls):
        cls.main.RESULTS_DIR, cls.main.UPLOAD_DIR, cls.ta._d.results_dir = cls._alt

    def _doc(self):
        conn = database.get_db()
        try:
            return dict(conn.execute("SELECT * FROM documents WHERE id = ?", (self.did,)).fetchone())
        finally:
            conn.close()

    def test_1_start_und_doppelstart(self):
        vorher = self._doc()

        def schnell(project_id, document_id, user_id, sprache, ui_lang):
            ordner, pdf, meta = self.ta.test_pfade(user_id, project_id, document_id)
            os.makedirs(ordner, exist_ok=True)
            with open(self.upload, "rb") as q, open(pdf, "wb") as f:
                f.write(q.read())
            with open(meta, "w", encoding="utf-8") as f:
                json.dump({"zeit": "2026-09-25 18:00:00", "seiten": 2, "struktur": {"elemente": 3}, "verapdf": None}, f)
            with self.ta._start_lock:
                self.ta._test_laeuft.pop(document_id, None)
                self.ta._test_nutzer.pop(user_id, None)
        with mock.patch.object(self.ta.pdf_tagging, "verfuegbar", return_value=True), \
             mock.patch.object(self.ta, "_test_sync", side_effect=schnell):
            r = self.c.post(f"/api/projects/{self.pid}/documents/{self.did}/tagging/test")
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["preis"], 0)
            for _ in range(100):   # der Lauf laeuft im eigenen Executor — abwarten, bis er fertig ist
                if self.did not in self.ta._test_laeuft:
                    break
                time.sleep(0.05)
            # Doppelstart: Hinweis — der TestClient wartet am Ende jeder Anfrage auf den Hintergrundlauf; im Server
            # laeuft er weiter. Darum die Sperre direkt pruefen: Lauf als laufend markieren, zweiten Start versuchen.
            with self.ta._start_lock:
                self.ta._test_laeuft[self.did] = {"seit": time.time(), "user_id": self.uid}
            try:
                self.assertEqual(self.c.post(f"/api/projects/{self.pid}/documents/{self.did}/tagging/test").status_code, 409)
                stand = self.c.get(f"/api/projects/{self.pid}/documents/{self.did}/tagging").json()
                self.assertTrue(stand["test"]["laeuft"])
            finally:
                with self.ta._start_lock:
                    self.ta._test_laeuft.pop(self.did, None)
        stand = self.c.get(f"/api/projects/{self.pid}/documents/{self.did}/tagging").json()
        self.assertFalse(stand["test"]["laeuft"])
        self.assertEqual(stand["test"]["zeit"], "2026-09-25 18:00:00")
        self.assertTrue(stand["test"]["hoerprobe_moeglich"])
        nachher = self._doc()
        for feld in ("original_path", "roh_path", "getaggt", "tagging_status", "tagging_bericht"):
            self.assertEqual(vorher.get(feld), nachher.get(feld), feld)   # Dokument unverändert

    def test_2_fremd_und_kein_download(self):
        with mock.patch.object(self.ta.pdf_tagging, "verfuegbar", return_value=True):
            self.assertEqual(self.cf.post(f"/api/projects/{self.pid}/documents/{self.did}/tagging/test").status_code, 404)
        self.assertEqual(self.cf.get(f"/api/projects/{self.pid}/documents/{self.did}/tagging/test/hoerprobe").status_code, 404)
        # Es gibt keinen Endpunkt, der die Testfassung ausliefert
        for pfad in ("/tagging/test/datei", "/tagging/test.pdf", "/tagging/test"):
            r = self.c.get(f"/api/projects/{self.pid}/documents/{self.did}{pfad}")
            self.assertNotEqual(r.headers.get("content-type", ""), "application/pdf", pfad)
        # Der normale Download liefert nie die Testfassung (Dokument ist nicht getaggt -> 404/400)
        r = self.c.get(f"/api/projects/{self.pid}/documents/{self.did}/tagging/datei")
        self.assertIn(r.status_code, (400, 404))

    def test_3_nutzer_nur_ein_testlauf(self):
        with self.ta._start_lock:
            self.ta._test_nutzer[self.uid] = 999999
        try:
            with mock.patch.object(self.ta.pdf_tagging, "verfuegbar", return_value=True):
                self.assertEqual(self.c.post(f"/api/projects/{self.pid}/documents/{self.did}/tagging/test").status_code, 429)
        finally:
            with self.ta._start_lock:
                self.ta._test_nutzer.pop(self.uid, None)

    def test_3b_grenzen(self):
        with mock.patch.object(self.ta.pdf_tagging, "verfuegbar", return_value=True):
            with self.ta._start_lock:
                for i in range(self.ta.TEST_GLEICHZEITIG):
                    self.ta._test_laeuft[900000 + i] = {"seit": time.time(), "user_id": 0}
            try:
                r = self.c.post(f"/api/projects/{self.pid}/documents/{self.did}/tagging/test")
                self.assertEqual(r.status_code, 429, r.text)   # globale Obergrenze
            finally:
                with self.ta._start_lock:
                    for i in range(self.ta.TEST_GLEICHZEITIG):
                        self.ta._test_laeuft.pop(900000 + i, None)
            with self.ta._start_lock:
                self.ta._test_zaehler[self.uid] = (time.strftime("%Y-%m-%d"), self.ta.TEST_JE_TAG)
            try:
                r = self.c.post(f"/api/projects/{self.pid}/documents/{self.did}/tagging/test")
                self.assertEqual(r.status_code, 429, r.text)   # Tagesgrenze
                self.assertIn("Heute", r.json()["detail"])
            finally:
                with self.ta._start_lock:
                    self.ta._test_zaehler.pop(self.uid, None)
        self.assertEqual(self.ta._test_laeuft.get(self.did), None)
        self.assertEqual(self.ta._test_nutzer.get(self.uid), None)

    def test_3c_ohne_testmodus_verworfen(self):
        """Traegt das Ergebnis keinen Testmodus-Vermerk, wird es verworfen: Fehler im Bericht, keine Testfassung,
        Sperren frei."""
        def fake_taggen(quelle, ziel, sprache, arbeitsordner=None, testmodus=False):
            self.assertTrue(testmodus)
            with open(ziel, "wb") as f:
                f.write(b"%PDF-1.7 lizenziert")
            return {"testmodus": False, "zeit": "x", "nachher": {}}
        with self.ta._start_lock:
            self.ta._test_laeuft[self.did] = {"seit": time.time(), "user_id": self.uid}
            self.ta._test_nutzer[self.uid] = self.did
        with mock.patch.object(self.ta.pdf_tagging, "taggen", side_effect=fake_taggen), \
             mock.patch.object(self.ta.pdf_tagging, "verapdf", return_value=None):
            self.ta._test_sync(self.pid, self.did, self.uid, "de", "de")
        ordner, pdf, meta = self.ta.test_pfade(self.uid, self.pid, self.did)
        with open(meta, encoding="utf-8") as f:
            b = json.load(f)
        self.assertIn("Testmodus", b.get("fehler", ""))
        self.assertFalse(os.path.exists(pdf))
        self.assertFalse([x for x in os.listdir(ordner) if x.endswith(".tmp.pdf")])
        self.assertIsNone(self.ta._test_laeuft.get(self.did))
        self.assertIsNone(self.ta._test_nutzer.get(self.uid))
        stand = self.c.get(f"/api/projects/{self.pid}/documents/{self.did}/tagging").json()["test"]
        self.assertFalse(stand.get("hoerprobe_moeglich"))

    def test_4_loeschen_raeumt_auf(self):
        ordner, pdf, meta = self.ta.test_pfade(self.uid, self.pid, self.did)
        ab_ordner = os.path.join(self.main.RESULTS_DIR, str(self.uid), str(self.pid), "_abschluss")
        os.makedirs(ordner, exist_ok=True); os.makedirs(ab_ordner, exist_ok=True)
        dateien = [pdf, meta, pdf + ".struktur.json", os.path.join(ab_ordner, f"doc{self.did}.pdf"),
                   os.path.join(ab_ordner, f"doc{self.did}_s1.png")]
        fremd = os.path.join(ab_ordner, f"doc{self.did}1.pdf")   # anderes Dokument (id mit gleicher Anfangsziffer)
        for p in dateien + [fremd]:
            with open(p, "w") as f:
                f.write("x")
        self.main._dokument_loeschen_sync(self.uid, self.pid, self.did)
        for p in dateien:
            self.assertFalse(os.path.exists(p), p)
        self.assertTrue(os.path.exists(fremd), "Dateien anderer Dokumente bleiben")


if __name__ == "__main__":
    unittest.main()
