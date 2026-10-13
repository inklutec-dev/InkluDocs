"""Paket 1 (Steve 13.10.2026): Klangspur des InkluAgent — Schalter funktionen.AGENT_KLANG, Einstellung „Töne“ je Konto,
frontend/klang.js. Wegwerf-Datenbank, keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_paket1_klang.py
"""
import itertools
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="paket1_klang_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_NR = itertools.count(1)


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def schalter(**werte):
    alle = {"AGENT_KONTO": True, "AGENT_UEBERALL": True, "AGENT_KLANG": True}
    alle.update(werte)
    return [mock.patch.object(funktionen, k, v) for k, v in alle.items()]


class Speicher(unittest.TestCase):
    def test_vorgabe_an_und_umschalten(self):
        from inkluagent import klang
        uid = database.create_user(f"klang{next(_NR)}-{time.time_ns()}@beispiel.invalid", "geheim-12345", "Test (fiktiv)")
        self.assertTrue(klang.fuer_konto(uid), "Vorgabe: an")
        klang.setzen(uid, False)
        self.assertFalse(klang.fuer_konto(uid))
        with mock.patch.object(funktionen, "AGENT_KLANG", False):
            self.assertIsNone(klang.seiten_wert(uid), "ohne Schalter nichts am Seitenrahmen")
        with mock.patch.object(funktionen, "AGENT_KLANG", True), mock.patch.object(funktionen, "AGENT_UEBERALL", True):
            self.assertEqual(klang.seiten_wert(uid), "aus")


class MitServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.TestClient = main, TestClient

    def setUp(self):
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        mail = f"klangsrv{next(_NR)}-{time.time_ns()}@beispiel.invalid"
        self.uid = database.create_user(mail, "geheim-12345", "Test (fiktiv)")
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.uid, mail, 0))

    def tearDown(self):
        mock.patch.stopall()

    def test_ohne_schalter_wie_bisher(self):
        html = self.c.get("/einstellungen").text
        self.assertNotIn("data-toene", html)
        self.assertNotIn("/static/klang.js", html)
        self.assertNotIn('id="toeneAn"', html)
        self.assertEqual(self.c.put("/api/me/toene", json={"toene": False}).status_code, 404)
        self.assertIsNone(self.c.get("/api/me").json()["user"]["toene"])

    def test_mit_schalter_seite_einstellung_und_api(self):
        for p in schalter():
            p.start()
        html = self.c.get("/einstellungen").text
        self.assertIn('data-toene="an"', html)
        self.assertIn("/static/klang.js", html)
        self.assertIn('id="toeneAn" checked', html)
        self.assertEqual(self.c.put("/api/me/toene", json={"toene": "ja"}).status_code, 400, "nur true oder false")
        r = self.c.put("/api/me/toene", json={"toene": False})
        self.assertEqual((r.status_code, r.json()["toene"]), (200, False))
        self.assertIn('data-toene="aus"', self.c.get("/dashboard").text)
        self.assertFalse(self.c.get("/api/me").json()["user"]["toene"])
        self.assertTrue(self.main._inkluagent_schalter()["klang"])


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        self.klang = lies(FRONTEND, "klang.js")
        self.js = lies(FRONTEND, "inkluagent.js")

    def test_klaenge_und_regeln(self):
        for name in ("karte:", "schritt:", "fertig:", "frage:", "fehler:"):
            self.assertIn(name, self.klang)
        self.assertIn("document.body.getAttribute('data-toene') === 'an'", self.klang, "ob es klingt, entscheidet das Konto")
        self.assertIn("a.pan.linearRampToValueAtTime(0, t0 + dauer * 0.9);", self.klang, "Wusch wandert von rechts in die Mitte")
        self.assertIn("window.Klang = {", self.klang)

    def test_haken_im_agenten(self):
        self.assertIn("if (window.Klang && typeof window.Klang.spielen === 'function') window.Klang.spielen(name);", self.js)
        self.assertIn("? 'karte' : 'fertig');", self.js, "Karte in der Antwort: Wusch, sonst „fertig“")
        self.assertIn("if (ev.ok) inkluagentKlang('schritt');", self.js)
        self.assertIn("if (isError && text) inkluagentKlang('fehler');", self.js)


if __name__ == "__main__":
    unittest.main()
