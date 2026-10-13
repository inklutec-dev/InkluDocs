"""Sicherheitspruefung des InkluAgent-Ausbaus (13.10.2026): Regressionstests zu den Befunden, Befund-Nummern wie im
Pruefbericht (S1 ...). Wegwerf-Datenbank, keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_sicherheit_pruefung_1310.py
"""
import os
import re
import sys
import tempfile
import unittest

TMP = tempfile.mkdtemp(prefix="sicherheit_1310_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def funktion(quelle, name):
    f = quelle[quelle.index("function " + name + "("):]
    return f[:f.index("\n    }\n")]


class S1ProjektKennung(unittest.TestCase):
    """DOM-XSS ueber ?projekt=: die Seite nimmt nur eine positive ganze Zahl (Verhaltensprobe: lokal/probe_xss_projekt.py)."""

    def setUp(self):
        self.app = lies(BACKEND, "templates", "app.html")

    def test_nur_positive_ganze_zahl(self):
        k = funktion(self.app, "projektKennung")
        self.assertIn(r"/^[1-9]\d{0,14}$/.test(s) ? Number(s) : null", k)
        muster = re.compile(r"^[1-9]\d{0,14}$")
        for gut in ("1", "4711", "999999999999999"):
            self.assertTrue(muster.match(gut), gut)
        for schlecht in ("", "0", "-1", "2abc", " 2", "2 ", "4711?x=\"><img src=x onerror=1>", "1e3", "0x1A", "2/../3"):
            self.assertIsNone(muster.match(schlecht), schlecht)

    def test_adresse_und_zurueck_nur_ueber_die_pruefung(self):
        self.assertIn("const _pid = projektKennung(new URLSearchParams(window.location.search).get('projekt'));", self.app)
        self.assertIn("const pid = projektKennung(new URLSearchParams(window.location.search).get('projekt'));", self.app)
        self.assertNotRegex(self.app, r"showProject\(\s*new URLSearchParams")

    def test_show_project_ist_die_einzige_tuer(self):
        s = funktion(self.app, "showProject")
        pruefung = s.index("const kennung = projektKennung(projectId);")
        self.assertIn("if (kennung === null) { window.location.href = '/dashboard'; return; }", s)
        self.assertLess(pruefung, s.index("currentDocProjectId = projectId;"), "vor jeder Verwendung")
        self.assertLess(pruefung, s.index("fetch("), "vor dem ersten Abruf")

    def test_agent_bereich_maskiert_die_kennung(self):
        self.assertIn("data-project-id=\"' + escHtmlAgent(projectId) + '\"", lies(FRONTEND, "inkluagent.js"))


class S5CorsNurEigeneHerkunft(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.c = main, TestClient(main.app)

    def _preflight(self, herkunft):
        return self.c.options("/api/me", headers={"Origin": herkunft, "Access-Control-Request-Method": "POST"})

    def test_herkunft(self):
        h = self.main._herkunft
        self.assertEqual(h("https://inkludocs.de/pfad?x=1"), "https://inkludocs.de")
        self.assertEqual(h("http://127.0.0.1:8791"), "http://127.0.0.1:8791")
        for leer in ("", None, "inkludocs.de", "javascript:alert(1)", "ftp://x.de"):
            self.assertEqual(h(leer), "", leer)

    def test_nur_die_eigenen_adressen_dieser_umgebung(self):
        erwartet = sorted({self.main._herkunft(self.main.BASE_URL), self.main._herkunft(self.main.API_BASE_URL)} - {""})
        self.assertEqual(self.main.CORS_HERKUENFTE, erwartet)
        self.assertIn("allow_origins=CORS_HERKUENFTE,", lies(BACKEND, "main.py"), "keine fest eingetragene fremde Umgebung")

    def test_fremde_umgebung_bekommt_keine_freigabe(self):
        eigene = self.main.CORS_HERKUENFTE[0]
        self.assertEqual(self._preflight(eigene).headers.get("access-control-allow-origin"), eigene)
        for fremd in ("https://andere-umgebung.inklutec.de", "https://staging.inkludocs.inklutec.de", "https://inkludocs.de.evil.example"):
            if fremd in self.main.CORS_HERKUENFTE:
                continue
            self.assertIsNone(self._preflight(fremd).headers.get("access-control-allow-origin"), fremd)


if __name__ == "__main__":
    unittest.main()
