"""InkluAgent-Ausbau Runde 2, Schritt 6 (12.10.2026): Antworten als Bloecke hinter funktionen.AGENT_BLOECKE — Aufzaehlungen des
Modells bleiben als „• “-Zeilen (inkluagent/sanitize.py), die Seite baut Absaetze und echte Listen (frontend/inkluagent.js,
inkluagentBloeckeBauen, nur textContent). Ohne Schalter wie bisher. Keine KI, keine Datenbank noetig.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_bloecke.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_bloecke_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
os.environ.pop("INKLUAGENT_BLOECKE", None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop  # noqa: E402
from inkluagent.sanitize import LISTENPUNKT, sanitize_markdown  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))

ROH = "Deine Projekte:\n- **Jahresbericht** (PDF)\n* Flyer (Word)\n+ Fotos (Grafik)\n\nSo geht es weiter:\n1. Öffnen\n2. Prüfen"


class Filter(unittest.TestCase):
    def test_ohne_schalter_wie_bisher(self):
        self.assertFalse(funktionen.AGENT_BLOECKE)
        s = sanitize_markdown(ROH)
        self.assertNotIn(LISTENPUNKT, s)
        self.assertIn("\nJahresbericht (PDF)\nFlyer (Word)\nFotos (Grafik)", s)

    def test_listen_bleiben_als_punkte(self):
        s = sanitize_markdown(ROH, listen=True)
        self.assertIn("\n• Jahresbericht (PDF)\n• Flyer (Word)\n• Fotos (Grafik)\n", s)
        self.assertNotIn("**", s)
        self.assertIn("1. Öffnen\n2. Prüfen", s)
        self.assertEqual(sanitize_markdown(s, listen=True), s, "idempotent")
        self.assertEqual(sanitize_markdown("3*5 ist 15", listen=True), "3*5 ist 15")

    def test_agent_mit_schalter(self):
        class Modell:
            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3):
                return {"content": [{"type": "text", "text": ROH}], "stop_reason": "end_turn"}
        c = database.get_db()
        c.execute("INSERT OR IGNORE INTO users (id, email, password_hash, display_name) VALUES (1, 'b@beispiel.invalid', 'x', 'Test')")
        c.commit()
        c.close()
        r = agent_loop.run_agent(None, 1, "Zeig meine Projekte", None, Modell(), konto=True)
        self.assertNotIn(LISTENPUNKT, r["reply"])
        with mock.patch.object(funktionen, "AGENT_BLOECKE", True):
            r = agent_loop.run_agent(None, 1, "Zeig meine Projekte", None, Modell(), konto=True)
        self.assertIn("• Jahresbericht (PDF)", r["reply"])


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(FRONTEND, "inkluagent.js"), encoding="utf-8") as f:
            self.js = f.read()

    def test_bloecke_nur_mit_schalter_und_ohne_innerhtml(self):
        self.assertIn("if (role === 'assistant' && window.INKLUAGENT && window.INKLUAGENT.bloecke) inkluagentBloeckeBauen(contentEl, content || '');",
                      self.js)
        f = self.js[self.js.index("function inkluagentBloeckeBauen("):]
        f = f[:f.index("\n}\n")]
        self.assertNotIn("innerHTML", f)
        self.assertIn("document.createElement('p')", f)
        self.assertIn("if (g.art === 'ol' && g.zeilen.length < 2)", f, "eine einzelne „1. …“-Zeile bleibt ein Absatz")


class Schalter(unittest.TestCase):
    def test_seite_erfaehrt_den_schalter(self):
        try:
            import main
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        self.assertFalse(main._inkluagent_schalter()["bloecke"])
        with mock.patch.object(funktionen, "AGENT_BLOECKE", True):
            self.assertTrue(main._inkluagent_schalter()["bloecke"])


if __name__ == "__main__":
    unittest.main()
