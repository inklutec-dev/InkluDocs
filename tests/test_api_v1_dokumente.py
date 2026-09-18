"""Public API v1 Dokumente (17.09.2026): Verdrahtung und Fehlerformat, ohne Netz und ohne Credits.
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_api_v1_dokumente.py
"""
import os
import sys
import unittest

sys.path.insert(0, "/app")
os.chdir("/app")
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
import api_dokumente_v1 as v1  # noqa: E402

ZIELE = [("POST", "/api/upload"), ("POST", "/api/scan-url"), ("DELETE", "/api/projects/{project_id}"),
         ("GET", "/api/projects/{project_id}"), ("POST", "/api/projects/{project_id}/generate"),
         ("POST", "/api/projects/{project_id}/generate/abbrechen"), ("GET", "/api/images/{image_id}/file"),
         ("POST", "/api/images/{image_id}/alt-text"), ("POST", "/api/projects/{project_id}/regenerate/{image_id}"),
         ("GET", "/api/projects/{project_id}/felder"), ("PATCH", "/api/felder/{feld_id}"),
         ("POST", "/api/felder/{feld_id}/generieren"), ("POST", "/api/projects/{project_id}/quickinfos/generieren"),
         ("POST", "/api/projects/{project_id}/quickinfos/abbrechen"), ("POST", "/api/projects/{project_id}/share"),
         ("GET", "/api/projects/{project_id}/shares"), ("POST", "/api/projects/{project_id}/shares/revoke"),
         ("GET", "/api/projects/{project_id}/export/pdfua/{token}")] + [("POST", p) for p in v1._EXPORT_ROUTEN.values()]

V1 = ["/api/v1/documents", "/api/v1/documents/{project_id}", "/api/v1/documents/{project_id}/generate",
      "/api/v1/documents/{project_id}/generate/cancel", "/api/v1/documents/{project_id}/items",
      "/api/v1/documents/{project_id}/items/{item_id}/file", "/api/v1/documents/{project_id}/items/{item_id}",
      "/api/v1/documents/{project_id}/items/{item_id}/generate", "/api/v1/documents/{project_id}/export/{fmt}",
      "/api/v1/documents/{project_id}/export/pdfua/{token}", "/api/v1/documents/{project_id}/review-link",
      "/api/v1/documents/{project_id}/review-links", "/api/v1/documents/{project_id}/review-links/revoke"]


class VerdrahtungTest(unittest.TestCase):
    def test_alle_zielrouten_existieren(self):
        for method, path in ZIELE:
            self.assertTrue(callable(v1._route(method, path)), f"{method} {path}")

    def test_v1_pfade_registriert(self):
        pfade = {getattr(r, "path", None) for r in main.app.routes}
        for p in V1:
            self.assertIn(p, pfade)

    def test_swagger_bleibt_aus(self):
        c = TestClient(main.app)
        self.assertEqual(c.get("/openapi.json").status_code, 404)


class FehlerformatTest(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(main.app)

    def test_ohne_schluessel_401_maschinenlesbar(self):
        for method, url in (("get", "/api/v1/documents"), ("get", "/api/v1/documents/1"),
                            ("post", "/api/v1/documents/1/generate"), ("delete", "/api/v1/documents/1")):
            r = getattr(self.c, method)(url)
            self.assertEqual(r.status_code, 401, url)
            body = r.json()
            self.assertEqual(body["error"]["code"], "unauthorized")
            self.assertEqual(body["error"]["status"], 401)
            self.assertTrue(body["detail"])

    def test_falscher_schluessel_401(self):
        r = self.c.get("/api/v1/documents", headers={"X-API-Key": "idocs_gibtesnicht"})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"]["code"], "unauthorized")

    def test_credits_fehlen_wird_zu_payment_required_mit_zahlen(self):
        from fastapi import HTTPException
        e = HTTPException(status_code=402, detail=main.billing.credits_fehlen_detail(
            {"preis": 25, "verfuegbar": 3, "fehlend": 22}, "Der Export"))
        antwort = v1._aus_http_exception(e)
        import json
        body = json.loads(bytes(antwort.body))
        self.assertEqual(antwort.status_code, 402)
        self.assertEqual(body["error"]["code"], "credits_fehlen")
        self.assertEqual(body["error"]["preis"], 25)
        self.assertEqual(body["error"]["fehlend"], 22)
        self.assertIn("25 Credits", body["detail"])

    def test_werkzeug_nach_dateityp(self):
        self.assertEqual(v1._werkzeug_fuer(".pdf", None), "pdf")
        self.assertEqual(v1._werkzeug_fuer(".PDF", "formular"), "formular")
        self.assertEqual(v1._werkzeug_fuer(".docx", None), "word")
        self.assertEqual(v1._werkzeug_fuer(".png", None), "grafik")
        from fastapi import HTTPException
        for ext, tool in ((".exe", None), (".docx", "pdf"), (".pdf", "word"), (".pdf", "hexerei")):
            with self.assertRaises(HTTPException):
                v1._werkzeug_fuer(ext, tool)

    def test_text_status_folgt_herunterladen_regel(self):
        leer = {"alt_text": "", "alt_text_edited": None, "original_alt": "", "image_type": "foto"}
        self.assertEqual(v1._text_status(leer), "offen")
        self.assertEqual(v1._text_status({**leer, "alt_text": "KI-Text"}), "mit_text")
        self.assertEqual(v1._text_status({**leer, "alt_text": "KI-Text", "alt_text_edited": ""}), "offen")
        self.assertEqual(v1._text_status({**leer, "image_type": "dekorativ"}), "dekorativ")


class FehlergrundUndEinstellungenTest(unittest.TestCase):
    """18.09.2026 (Steve, API-Pruefung): Fehlergrund je Bild, scope, Einstellungen je Lauf."""

    def test_fehler_kurz_liefert_nutzertaugliche_gruende(self):
        self.assertIn("429", main._fehler_kurz(RuntimeError("Gemini HTTP 429 (x): Resource exhausted")))
        self.assertIn("Zeitüberschreitung", main._fehler_kurz(TimeoutError("timed out")))
        self.assertIn("unbrauchbar", main._fehler_kurz(ValueError("Schema verletzt auch nach Retry")))
        self.assertIn("gelesen", main._fehler_kurz(OSError("cannot identify image file")))
        self.assertIn("Unerwarteter", main._fehler_kurz(KeyError("x")))
        for e in (RuntimeError("secret sk-123 /app/data/uploads/9/x.pdf"),):
            self.assertNotIn("sk-123", main._fehler_kurz(e)); self.assertNotIn("/app/", main._fehler_kurz(e))

    def test_item_traegt_error_nur_bei_fehler(self):
        basis = {"id": 1, "status": "done", "alt_text": "x", "alt_text_edited": None, "original_alt": "", "image_type": "foto", "fehler_grund": ""}
        self.assertIsNone(v1._bild_item(5, basis)["error"])
        self.assertEqual(v1._bild_item(5, {**basis, "status": "error", "fehler_grund": "KI-Dienst überlastet (429)"})["error"], "KI-Dienst überlastet (429)")
        self.assertEqual(v1._bild_item(5, {**basis, "status": "error", "fehler_grund": ""})["error"], None)

    def test_einstellungen_je_lauf(self):
        import sqlite3
        conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
        conn.executescript("""CREATE TABLE projects (id INTEGER PRIMARY KEY, user_id INT, alt_language TEXT, use_context INT DEFAULT 1, prompt_id INT);
            CREATE TABLE user_prompts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT, name TEXT, description TEXT, category TEXT, prompt_text TEXT);
            INSERT INTO projects (id, user_id, alt_language) VALUES (7, 1, 'de');
            INSERT INTO user_prompts (user_id, name, description, category, prompt_text) VALUES (2, 'fremd', '', '', 'x');""")
        alt_d = v1._d
        class D: alt_text_languages = ("de", "en", "fr", "es", "da", "sv")
        v1._d = D()
        try:
            g = v1._lauf_einstellungen_anwenden(conn, 1, 7, {"language": "en", "use_context": False, "prompt": "Kurz bitte."})
            self.assertEqual(g["language"], "en"); self.assertIs(g["use_context"], False); self.assertTrue(g["prompt_id"])
            row = conn.execute("SELECT alt_language, use_context, prompt_id FROM projects WHERE id = 7").fetchone()
            self.assertEqual((row[0], row[1]), ("en", 0))
            up = conn.execute("SELECT category, prompt_text, name FROM user_prompts WHERE id = ?", (row[2],)).fetchone()
            self.assertEqual((up[0], up[1]), ("API", "Kurz bitte.")); self.assertTrue(up[2].startswith("API: "))
            g2 = v1._lauf_einstellungen_anwenden(conn, 1, 7, {"prompt": "Kurz bitte."})
            self.assertEqual(g2["prompt_id"], row[2])   # gleicher Text = gleiche Zeile
            with self.assertRaises(Exception):
                v1._lauf_einstellungen_anwenden(conn, 1, 7, {"prompt_id": 1})   # fremder Prompt -> 404
            with self.assertRaises(Exception):
                v1._lauf_einstellungen_anwenden(conn, 1, 7, {"prompt": "x" * 5000})   # zu lang -> 400
            g3 = v1._lauf_einstellungen_anwenden(conn, 1, 7, {"prompt_id": 0})
            self.assertIsNone(g3["prompt_id"])
        finally:
            v1._d = alt_d


class LeseBremseTest(unittest.TestCase):
    def test_lesende_aufrufe_haben_eigene_grenze(self):
        from fastapi import HTTPException
        v1._lese_fenster.pop(-99, None)
        for _ in range(v1.LESE_LIMIT_MINUTE):
            v1._lese_bremse(-99)
        with self.assertRaises(HTTPException) as cm:
            v1._lese_bremse(-99)
        self.assertEqual(cm.exception.status_code, 429)
        self.assertEqual(cm.exception.headers.get("Retry-After"), "60")
        v1._lese_fenster.pop(-99, None)


class SchluesselLoeschenTest(unittest.TestCase):
    """Vorher: IntegrityError, sobald api_usage/api_results am Schluessel hingen (17.09.2026)."""

    def test_benutzter_schluessel_laesst_sich_loeschen(self):
        import database
        conn = database.get_db()
        uid = conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()[0]
        conn.close()
        kid, roh = database.create_api_key(uid, "unittest-loeschen (fiktiv)")
        try:
            database.log_api_usage(kid, uid, model_used="v1.documents.test", success=True)
            database.create_api_result("unittest_res_" + str(kid), uid, kid, "a", "b")
            self.assertTrue(database.delete_api_key(uid, kid))
            conn = database.get_db()
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM api_usage WHERE api_key_id = ?", (kid,)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM api_results WHERE api_key_id = ?", (kid,)).fetchone()[0], 0)
            conn.close()
            self.assertFalse(database.delete_api_key(uid, kid))
        finally:
            conn = database.get_db()
            conn.execute("DELETE FROM api_usage WHERE api_key_id = ?", (kid,)); conn.execute("DELETE FROM api_results WHERE api_key_id = ?", (kid,))
            conn.execute("DELETE FROM api_keys WHERE id = ?", (kid,)); conn.commit(); conn.close()


class VerbrauchTest(unittest.TestCase):
    """Verbrauch je Schluessel (17.09.2026): Zaehler, Dokument-Zuordnung, Endpunkt nur mit Login."""

    def test_stats_endpunkt_braucht_login(self):
        self.assertEqual(TestClient(main.app).get("/api/api-keys/stats").status_code, 401)

    def test_stats_zaehlen_aufrufe_und_dokumente(self):
        import database
        conn = database.get_db()
        uid = conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()[0]
        conn.close()
        kid, roh = database.create_api_key(uid, "unittest-stats (fiktiv)")
        pid = None
        try:
            database.log_api_usage(kid, uid, model_used="v1.documents.create", success=True)
            database.log_api_usage(kid, uid, model_used="v1.documents.export", success=False, error_message="422")
            conn = database.get_db()
            cur = conn.execute("INSERT INTO projects (user_id, name, filename, original_path, status, project_type, tool, api_key_id) "
                               "VALUES (?, 'unittest (fiktiv)', 'x.pdf', '', 'neu', 'pdf', 'pdf', ?)", (uid, kid))
            pid = cur.lastrowid; conn.commit(); conn.close()
            st = database.get_api_key_stats(uid)
            k = st["keys"][str(kid)]
            self.assertEqual((k["calls_total"], k["calls_today"], k["errors_total"], k["documents"]), (2, 2, 1, 1))
            self.assertEqual(k["error_rate"], 50.0)
            self.assertGreaterEqual(st["summary"]["calls_today"], 2)
            self.assertTrue(st["keys_count"] >= 1)
        finally:
            conn = database.get_db()
            if pid:
                conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
            conn.execute("DELETE FROM api_usage WHERE api_key_id = ?", (kid,)); conn.execute("DELETE FROM api_keys WHERE id = ?", (kid,))
            conn.commit(); conn.close()


if __name__ == "__main__":
    unittest.main()
