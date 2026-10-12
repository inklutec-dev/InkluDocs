"""InkluAgent: Gespraeche (Steve 12.10.2026, Schalter AGENT_GESPRAECHE + AGENT_KONTO). „Neues Gespräch“ beginnt ein neues, die
frueheren bleiben auf der Seite „Frühere Gespräche“ (schlichte Liste, neueste zuerst) lesbar und lassen sich mit „Hier
weitermachen“ fortsetzen; auf Seiten ohne Projekt findet der Agent fruehere Gespraeche und verlinkt sie. Wegwerf-Datenbank,
keine KI, keine Mails.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_gespraeche.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_gespraeche_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KONTO", "INKLUAGENT_GESPRAECHE", "INKLUAGENT_UEBERALL", "INKLUAGENT_HILFE"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, konto, storage  # noqa: E402
from inkluagent.daten import DATEN_MARKE  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_N = [0]


def nutzer() -> dict:
    _N[0] += 1
    mail = f"gespraech{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, name: str) -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, 'word', 'docx', 'extracted')", (uid, name, name)).lastrowid
    c.commit()
    c.close()
    return pid


class Speicher(unittest.TestCase):
    def test_neues_gespraech_frueheres_bleibt(self):
        u = nutzer()
        p = projekt(u["id"], "Jahresbericht (fiktiv)")
        storage.konto_anhaengen(u["id"], p, "user", "Wie viele Bilder hat der Jahresbericht?\nZweite Zeile")
        storage.konto_anhaengen(u["id"], p, "assistant", "Zwölf (fiktiv).")
        erstes = storage.aktuelles_gespraech(u["id"])
        neu = storage.neues_gespraech(u["id"])
        self.assertNotEqual(erstes, neu)
        self.assertEqual(storage.neues_gespraech(u["id"]), neu, "ein leeres laufendes wird weiterbenutzt")
        self.assertEqual(storage.konto_verlauf(u["id"]), [], "das neue Gespraech beginnt leer")
        storage.konto_anhaengen(u["id"], None, "user", "Hallo im neuen Gespräch")
        liste = storage.gespraeche_liste(u["id"])
        self.assertEqual([g["id"] for g in liste], [neu, erstes], "neueste zuerst, leere fehlen")
        self.assertEqual(liste[1]["titel"], "Wie viele Bilder hat der Jahresbericht?")
        self.assertEqual(liste[1]["anzahl"], 2)
        self.assertEqual([pr["name"] for pr in liste[1]["projekte"]], ["Jahresbericht (fiktiv)"])
        self.assertEqual(liste[1]["letztes_projekt"], p)
        self.assertTrue(liste[0]["aktuell"])
        self.assertEqual([m["content"] for m in storage.gespraech_lesen(u["id"], erstes)][1], "Zwölf (fiktiv).")
        self.assertTrue(storage.gespraech_fortsetzen(u["id"], erstes))
        self.assertEqual(storage.aktuelles_gespraech(u["id"]), erstes)
        self.assertEqual(len(storage.konto_verlauf(u["id"])), 2)

    def test_filter_und_fremde(self):
        u, fremd = nutzer(), nutzer()
        a, b = projekt(u["id"], "Flyer (fiktiv)"), projekt(u["id"], "Satzung (fiktiv)")
        storage.konto_anhaengen(u["id"], a, "user", "Flyer kürzer")
        storage.neues_gespraech(u["id"])
        storage.konto_anhaengen(u["id"], b, "user", "Satzung prüfen")
        self.assertEqual([g["titel"] for g in storage.gespraeche_liste(u["id"], project_id=a)], ["Flyer kürzer"])
        self.assertEqual([g["titel"] for g in storage.gespraeche_liste(u["id"], suche="Satzung")], ["Satzung prüfen"])
        gid = storage.gespraeche_liste(u["id"])[0]["id"]
        self.assertIsNone(storage.gespraech_lesen(fremd["id"], gid))
        self.assertFalse(storage.gespraech_fortsetzen(fremd["id"], gid))

    def test_altbestand_kommt_ins_erste_gespraech(self):
        u = nutzer()
        c = database.get_db()
        c.execute("INSERT INTO agent_nachrichten (user_id, project_id, projekt_name, role, content) VALUES (?, NULL, '', 'user', 'Alte Frage')",
                  (u["id"],))
        c.commit()
        c.close()
        self.assertEqual([m["content"] for m in storage.konto_verlauf(u["id"])], ["Alte Frage"])
        self.assertEqual(storage.gespraeche_liste(u["id"])[0]["titel"], "Alte Frage")

    def test_gedaechtnis_nur_aus_dem_laufenden_gespraech(self):
        u = nutzer()
        p = projekt(u["id"], "Brief (fiktiv)")
        storage.konto_anhaengen(u["id"], p, "user", "Altes Thema")
        storage.konto_anhaengen(u["id"], p, "assistant", "Alte Antwort")
        storage.neues_gespraech(u["id"])
        msgs = konto.nachrichten(u["id"], "Neue Frage")
        self.assertFalse(any("Altes Thema" in m["content"] for m in msgs))

    def test_konto_loeschen_nimmt_die_gespraeche_mit(self):
        u = nutzer()
        storage.konto_anhaengen(u["id"], None, "user", "Hallo")
        database.delete_user_data(u["id"])
        c = database.get_db()
        self.assertEqual(c.execute("SELECT COUNT(*) FROM agent_gespraeche WHERE user_id = ?", (u["id"],)).fetchone()[0], 0)
        c.close()


class Werkzeug(unittest.TestCase):
    def test_fruehere_gespraeche_finden_und_verlinken(self):
        u = nutzer()
        p = projekt(u["id"], "Jahresbericht (fiktiv)")
        storage.konto_anhaengen(u["id"], p, "user", "Alt-Texte im Jahresbericht")
        alt = storage.aktuelles_gespraech(u["id"])
        storage.neues_gespraech(u["id"])
        storage.konto_anhaengen(u["id"], None, "user", "Was gibt es Neues?")
        with mock.patch.object(funktionen, "AGENT_GESPRAECHE", True):
            defs, ex, system = konto.werkzeugsatz(u["id"])
            self.assertIn("fruehere_gespraeche", {d["name"] for d in defs})
            self.assertIn("Frühere Gespräche", system)
            r = ex.execute("fruehere_gespraeche", {"project_id": p})
        self.assertEqual([g["gespraech_id"] for g in r["result"]["gespraeche"]], [alt], "das laufende ist nicht dabei")
        self.assertEqual(r["result"]["gespraeche"][0]["titel_daten"], DATEN_MARKE + "Alt-Texte im Jahresbericht")
        self.assertEqual(r["anhang"][0]["url"], f"/gespraeche/{alt}")
        self.assertNotIn("fruehere_gespraeche", {d["name"] for d in konto.werkzeugsatz(u["id"])[0]}, "ohne Schalter nicht da")

    def test_mehrere_links_aus_einem_werkzeug(self):
        class Modell:
            def __init__(self):
                self.n = 0

            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3):
                self.n += 1
                if self.n == 1:
                    return {"content": [{"type": "tool_use", "id": "t1", "name": "fruehere_gespraeche", "input": {"suche": "Thema"}}],
                            "stop_reason": "tool_use"}
                return {"content": [{"type": "text", "text": "Unten stehen die Links."}], "stop_reason": "end_turn"}
        u = nutzer()
        for i in range(4):
            storage.konto_anhaengen(u["id"], None, "user", f"Thema {i}")
            storage.neues_gespraech(u["id"])
        with mock.patch.object(funktionen, "AGENT_GESPRAECHE", True):
            r = agent_loop.run_agent(None, u["id"], "Worüber haben wir gesprochen?", None, Modell(), konto=True)
        self.assertEqual(len(r["anhang"]), 3, "hoechstens drei Links")
        self.assertTrue(all(a["art"] == "link" for a in r["anhang"]))


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
        self.u = nutzer()
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))

    def tearDown(self):
        mock.patch.stopall()

    def _an(self):
        mock.patch.object(funktionen, "AGENT_KONTO", True).start()
        mock.patch.object(funktionen, "AGENT_GESPRAECHE", True).start()

    def test_schalter_aus(self):
        self.assertEqual(self.c.post("/api/agent/gespraech/neu").status_code, 404)
        self.assertEqual(self.c.get("/api/agent/gespraeche").status_code, 404)
        self.assertEqual(self.c.get("/gespraeche").status_code, 404)

    def test_seiten_und_hier_weitermachen(self):
        self._an()
        p = projekt(self.u["id"], "Jahresbericht (fiktiv)")
        storage.konto_anhaengen(self.u["id"], p, "user", "Was fehlt im Jahresbericht?")
        storage.konto_anhaengen(self.u["id"], p, "assistant", "Zwei Punkte:\n• Titel\n• Sprache")
        alt = storage.aktuelles_gespraech(self.u["id"])
        self.assertEqual(self.c.post("/api/agent/gespraech/neu").status_code, 200)
        html = self.c.get("/gespraeche").text
        self.assertIn("<h1>Frühere Gespräche</h1>", html)
        self.assertIn(f'<h2 class="gespraech-titel"><a href="/gespraeche/{alt}">Was fehlt im Jahresbericht?</a></h2>', html)
        self.assertIn("Jahresbericht (fiktiv)", html)
        lese = self.c.get(f"/gespraeche/{alt}").text
        self.assertIn('id="gespraechWeiter"', lese)
        self.assertIn("<li>Titel</li>", lese, "Antworten als Liste")
        self.assertIn("Ab hier: Projekt „Jahresbericht (fiktiv)“", lese)
        r = self.c.post(f"/api/agent/gespraeche/{alt}/fortsetzen")
        self.assertEqual(r.json()["weiter"], f"/app?projekt={p}&agent=offen")
        self.assertEqual(storage.aktuelles_gespraech(self.u["id"]), alt)
        api = self.c.get("/api/agent/gespraeche").json()["gespraeche"]
        self.assertTrue(api[0]["datum"])

    def test_fremdes_gespraech(self):
        self._an()
        fremd = nutzer()
        storage.konto_anhaengen(fremd["id"], None, "user", "Geheim (fiktiv)")
        gid = storage.aktuelles_gespraech(fremd["id"])
        self.assertEqual(self.c.get(f"/gespraeche/{gid}").status_code, 404)
        self.assertEqual(self.c.post(f"/api/agent/gespraeche/{gid}/fortsetzen").status_code, 404)

    def test_weiter_ohne_projekt_auf_die_uebersicht(self):
        self._an()
        storage.konto_anhaengen(self.u["id"], None, "user", "Allgemeine Frage")
        gid = storage.aktuelles_gespraech(self.u["id"])
        self.assertEqual(self.c.post(f"/api/agent/gespraeche/{gid}/fortsetzen").json()["weiter"], "/dashboard?agent=offen")

    def test_hilfe_seite_beschreibt_das_gespraech(self):
        def hauptinhalt():   # nur <main> — der Uebersetzungskatalog der Seite (window.I18N) enthaelt alle Saetze
            html = self.c.get("/hilfe/inkluagent").text
            return html.split('<main id="main"', 1)[1].split("</main>", 1)[0]
        with mock.patch.object(funktionen, "AGENT_HILFE", True):
            self.assertIn("Jedes Projekt hat sein eigenes Gespräch.", hauptinhalt())
            self._an()
            mock.patch.object(funktionen, "AGENT_UEBERALL", True).start()
            text = hauptinhalt()
        self.assertNotIn("Jedes Projekt hat sein eigenes Gespräch.", text)
        self.assertIn("durchgehendes Gespräch", text)
        self.assertIn("Hier weitermachen", text)
        self.assertIn("Zum InkluAgent", text)


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(FRONTEND, "inkluagent.js"), encoding="utf-8") as f:
            self.js = f.read()

    def test_leiste_nur_mit_schalter(self):
        self.assertIn("(inkluagentGespraecheAn() ? inkluagentGespraechLeisteHtml() : '')", self.js)
        f = self.js[self.js.index("function inkluagentGespraechLeisteHtml("):]
        f = f[:f.index("\n}\n")]
        self.assertIn('<a href="/gespraeche">', f, "die Liste steht auf einer eigenen Seite, nicht im Agenten")

    def test_hier_weitermachen_oeffnet_den_agenten(self):
        self.assertIn("get('agent') === 'offen'", self.js)


if __name__ == "__main__":
    unittest.main()
