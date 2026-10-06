"""Ergebnis-Abholung nach Verbindungsabbruch (06.10.2026, backend/ki_abholung.py) — auf einer WEGWERF-Datenbank.
Prueft: Stand „laeuft“/„fertig“, dieselbe Antwort wie der POST, kein zweites Generieren und keine zweite Buchung
beim Abholen, fremde/unbekannte Kennungen, Fehlerantworten (402/404/500), Endpunkte ohne Kopf unveraendert, und dass
FastAPI die dekorierten Endpunkte richtig verdrahtet (formular_api nutzt `from __future__ import annotations`).
    docker exec -w /app <container> python3 -m unittest /app/tests/test_ki_abholung.py
Immer im EIGENEN Prozess starten (nicht zusammen mit anderen Testdateien in einem unittest-Aufruf): `database` liest
den Pfad beim ersten Import; ist es schon geladen, ueberspringt sich der Test, statt fremde Daten anzufassen.
"""
import asyncio
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="ki_abholung_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
sys.path.insert(0, "/app")
os.chdir("/app")
import database  # noqa: E402
# Hat im selben Prozess schon ein anderer Test `database` mit einer anderen Datenbank geladen, nichts anfassen
# (06.10.2026: so landeten zwei Testprojekte in der Staging-Datenbank) — die Klasse ueberspringt sich dann.
FREMDE_DB = os.path.abspath(database.DB_PATH) != os.path.abspath(os.environ["INKLUDOCS_DB"])
if not FREMDE_DB:
    database.init_db()
    import main  # noqa: E402
    import formular_api  # noqa: E402
    import ki_abholung  # noqa: E402
from fastapi import HTTPException, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

NUTZER = {"id": 1, "email": "test@example.invalid", "is_admin": 0}
FREMD = {"id": 2, "email": "fremd@example.invalid", "is_admin": 0}
ERGEBNIS = {"alt_text": "Balkendiagramm Umsatz 2026 (fiktiv)", "bildtyp": "diagramm", "konfidenz": "hoch",
            "langbeschreibung": "", "needs_review": False, "pipeline_steps": "", "validation_result": ""}


class _Anfrage:
    def __init__(self, kopf):
        self.headers = {"x-ki-anfrage": kopf} if kopf else {}


class KiAbholungTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FREMDE_DB:
            raise unittest.SkipTest("database ist schon mit einer anderen Datenbank geladen — bitte im eigenen Prozess starten")
        c = database.get_db()
        for u in (NUTZER, FREMD):
            c.execute("INSERT OR IGNORE INTO users (id, email, password_hash, display_name) VALUES (?, ?, 'x', 'Test')", (u["id"], u["email"]))
        cls.pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                            "VALUES (1, 'a.pdf', '', 'Abholung (fiktiv)', 'pdf', 'pdf', 'completed')").lastrowid
        bild = os.path.join(TMP, "b.png")
        with open(bild, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        cls.iid = c.execute("INSERT INTO images (project_id, page_number, image_index, image_path, status, alt_text) VALUES (?, 1, 1, ?, 'done', 'alt (fiktiv)')",
                            (cls.pid, bild)).lastrowid
        c.commit()
        c.close()
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def setUp(self):
        self.nutzer = NUTZER
        main.app.dependency_overrides[main.get_current_user] = lambda: self.nutzer
        main.app.dependency_overrides[formular_api._user] = lambda: self.nutzer
        self.generiert = mock.Mock(return_value=dict(ERGEBNIS))
        self.gebucht = mock.Mock()
        self.patches = [
            mock.patch.object(main, "generate_alt_text", self.generiert),
            mock.patch.object(main.billing, "verbuche", self.gebucht),
            mock.patch.object(main.billing, "aktion_pruefung", return_value={"erlaubt": True}),
            mock.patch.object(main, "tageslimit_wache", return_value=None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        main.app.dependency_overrides.clear()

    def _neu(self, kennung):
        return self.client.post(f"/api/projects/{self.pid}/regenerate/{self.iid}", json={}, headers={"X-KI-Anfrage": kennung})

    def test_fertig_liefert_dieselbe_antwort_ohne_zweite_generierung_und_buchung(self):
        r = self._neu("test-kennung-0001")
        self.assertEqual(r.status_code, 200)
        a = self.client.get("/api/ki-anfragen/test-kennung-0001")
        self.assertEqual(a.status_code, 200)
        self.assertEqual(a.json(), {"stand": "fertig", "status": 200, "daten": r.json()})
        for _ in range(3):   # mehrfach abholen: nichts wird neu erzeugt oder gebucht
            self.client.get("/api/ki-anfragen/test-kennung-0001")
        self.assertEqual(self.generiert.call_count, 1)
        self.assertEqual(self.gebucht.call_count, 1)

    def test_ohne_kopf_unveraendert_und_nichts_gemerkt(self):
        vorher = dict(ki_abholung._anfragen)
        r = self.client.post(f"/api/projects/{self.pid}/regenerate/{self.iid}", json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json()), {"ok", "alt_text", "bildtyp", "konfidenz", "langbeschreibung"})
        self.assertEqual(set(ki_abholung._anfragen), set(vorher))   # ohne Kopf wird nichts gemerkt

    def test_fremd_unbekannt_ungueltig_404(self):
        self._neu("test-kennung-0002")
        self.nutzer = FREMD
        self.assertEqual(self.client.get("/api/ki-anfragen/test-kennung-0002").status_code, 404)
        self.nutzer = NUTZER
        self.assertEqual(self.client.get("/api/ki-anfragen/gibt-es-nicht-0000").status_code, 404)
        self.assertEqual(self.client.get("/api/ki-anfragen/zu-kurz").status_code, 404)
        self.assertEqual(self.client.get("/api/ki-anfragen/ungueltig_zeichen!").status_code, 404)

    def test_fremde_kennung_wird_nicht_ueberschrieben(self):
        self._neu("test-kennung-0003")
        self.nutzer = FREMD
        self.client.post(f"/api/projects/{self.pid}/regenerate/{self.iid}", json={}, headers={"X-KI-Anfrage": "test-kennung-0003"})
        self.nutzer = NUTZER
        self.assertEqual(self.client.get("/api/ki-anfragen/test-kennung-0003").json()["status"], 200)

    def test_serverfehler_wird_als_fehler_abgeholt(self):
        self.generiert.side_effect = RuntimeError("Modell weg (fiktiv)")
        with mock.patch.object(main, "_fehler_protokoll"):   # kein Traceback im Testprotokoll
            r = self._neu("test-kennung-0004")
        self.assertEqual(r.status_code, 500)
        a = self.client.get("/api/ki-anfragen/test-kennung-0004").json()
        self.assertEqual((a["stand"], a["status"]), ("fertig", 500))
        self.assertEqual(a["daten"], r.json())
        self.assertNotIn("Modell weg", str(a))   # kein roher Ausnahmetext nach aussen
        self.assertEqual(self.gebucht.call_count, 0)

    def test_credits_fehlen_402_wird_abgeholt(self):
        with mock.patch.object(main.billing, "aktion_pruefung", return_value={"erlaubt": False}), \
             mock.patch.object(main.billing, "credits_fehlen_detail", return_value={"preis": 5, "verfuegbar": 0}):
            r = self._neu("test-kennung-0005")
        self.assertEqual(r.status_code, 402)
        a = self.client.get("/api/ki-anfragen/test-kennung-0005").json()
        self.assertEqual((a["status"], a["daten"]), (402, {"detail": {"preis": 5, "verfuegbar": 0}}))
        self.assertEqual(self.generiert.call_count, 0)

    def test_quickinfo_endpunkt_dekoriert_und_verdrahtet(self):
        route = next(r for r in main.app.routes if getattr(r, "path", "") == "/api/felder/{feld_id}/generieren")
        self.assertEqual(route.dependant.request_param_name, "request")
        self.assertEqual([p.name for p in route.dependant.path_params], ["feld_id"])
        self.assertEqual(route.dependant.query_params, [])
        r = self.client.post("/api/felder/999999/generieren", headers={"X-KI-Anfrage": "test-kennung-0006"})
        self.assertEqual(r.status_code, 404)
        a = self.client.get("/api/ki-anfragen/test-kennung-0006").json()
        self.assertEqual((a["stand"], a["status"], a["daten"]), ("fertig", 404, r.json()))

    def test_stand_laeuft_waehrend_der_arbeit(self):
        async def lauf():
            frei = asyncio.Event()

            @ki_abholung.abholbar
            async def endpunkt(x: int, request: Request, user: dict):
                await frei.wait()
                return {"ok": True, "x": x}
            t = asyncio.create_task(endpunkt(x=7, request=_Anfrage("test-kennung-0007"), user=NUTZER))
            await asyncio.sleep(0)
            self.assertEqual(ki_abholung.abfragen("test-kennung-0007", 1), {"stand": "laeuft"})
            frei.set()
            await t
            self.assertEqual(ki_abholung.abfragen("test-kennung-0007", 1), {"stand": "fertig", "status": 200, "daten": {"ok": True, "x": 7}})
        asyncio.run(lauf())

    def test_abgebrochener_lauf_wird_vergessen(self):
        async def lauf():
            @ki_abholung.abholbar
            async def endpunkt(request: Request, user: dict):
                await asyncio.sleep(10)
            t = asyncio.create_task(endpunkt(request=_Anfrage("test-kennung-0008"), user=NUTZER))
            await asyncio.sleep(0)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
            self.assertIsNone(ki_abholung.abfragen("test-kennung-0008", 1))
        asyncio.run(lauf())

    def test_aufbewahrung_laeuft_ab(self):
        ki_abholung.beginnen("test-kennung-0009", 1)
        ki_abholung.abschliessen("test-kennung-0009", 1, 200, {"ok": True})
        with mock.patch.object(ki_abholung.time, "time", return_value=time.time() + ki_abholung.AUFBEWAHREN_S + 5):
            self.assertIsNone(ki_abholung.abfragen("test-kennung-0009", 1))


if __name__ == "__main__":
    unittest.main()
