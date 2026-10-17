"""InkluAgent-Ausbau Runde 2, Schritt 3 (12.10.2026): der InkluAgent auf JEDER Seite hinter funktionen.AGENT_UEBERALL (braucht
AGENT_KONTO). Fester Bereich im Seitenrahmen (base_app.html), Seiten ohne Projekt mit lesenden Konto-Werkzeugen
(inkluagent/konto.py, POST /api/agent/nachricht), laufende Antwort ueber den Seitenwechsel (GET /api/agent/status), auch in
der Barrierefreiheitspruefung (Michael 12.10.2026). Wegwerf-Datenbank (/tmp), keine KI, keine Mails.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_ueberall.py
"""
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_ueberall_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KONTO", "INKLUAGENT_UEBERALL", "INKLUAGENT_HOCHLADEN", "INKLUAGENT_SICHERHEIT", "INKLUAGENT_HILFE"):
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
from inkluagent.prompts.system_gemeinsam import DATEN_KEINE_ANWEISUNG  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_N = [0]


def nutzer() -> dict:
    _N[0] += 1
    mail = f"ueberall{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, name: str, tool: str = "pdf", typ: str = "pdf", status: str = "extracted") -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, ?, ?, ?)", (uid, name, name, tool, typ, status)).lastrowid
    c.commit()
    c.close()
    return pid


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


class Schalter(unittest.TestCase):
    def test_vorgabe_aus(self):
        self.assertFalse(funktionen.AGENT_UEBERALL)
        with mock.patch.dict(os.environ, {"INKLUAGENT_UEBERALL": "an"}):
            self.assertTrue(funktionen._umgebung_an("INKLUAGENT_UEBERALL"))


class KontoWerkzeuge(unittest.TestCase):
    def setUp(self):
        self.u, self.fremd = nutzer(), nutzer()
        self.a = projekt(self.u["id"], "Jahresbericht 2026 (fiktiv)")
        self.b = projekt(self.u["id"], "Flyer Sommerfest (fiktiv)", "word", "docx", status="processing")
        self.f = projekt(self.fremd["id"], "Geheim (fiktiv)")
        self.ex = konto.KontoExecutor(self.u["id"])

    def test_projekte_auflisten_nur_eigene_mit_suche_und_marke(self):
        r = self.ex.execute("projekte_auflisten", {})["result"]
        ids = [p["project_id"] for p in r["projekte"]]
        self.assertEqual(ids, [self.b, self.a], "neueste zuerst, nur eigene")
        self.assertEqual(r["anzahl_gesamt"], 2)
        self.assertEqual(r["projekte"][1]["name_daten"], DATEN_MARKE + "Jahresbericht 2026 (fiktiv)")
        self.assertEqual(r["projekte"][0]["art"], "Word")
        r2 = self.ex.execute("projekte_auflisten", {"suche": "sommer"})["result"]
        self.assertEqual([p["project_id"] for p in r2["projekte"]], [self.b])

    def test_projekt_oeffnen_setzt_fokus_und_link(self):
        r = self.ex.execute("projekt_oeffnen", {"project_id": self.a})
        self.assertTrue(r["ok"])
        self.assertEqual(storage.fokus(self.u["id"]), self.a)
        self.assertEqual(r["anhang"]["art"], "link")
        self.assertEqual(r["anhang"]["url"], f"/app?projekt={self.a}")
        self.assertIn("Jahresbericht 2026 (fiktiv)", r["anhang"]["text"])

    def test_fremdes_projekt_nicht_oeffnen(self):
        r = self.ex.execute("projekt_oeffnen", {"project_id": self.f})
        self.assertFalse(r["ok"])
        self.assertNotEqual(storage.fokus(self.u["id"]), self.f)
        self.assertFalse(self.ex.execute("projekt_oeffnen", {"project_id": "x"})["ok"])

    def test_guthaben(self):
        import billing
        with mock.patch.object(billing, "verfuegbare_credits", return_value=42):
            self.assertEqual(self.ex.execute("guthaben_lesen", {})["result"]["verfuegbar"], 42)
        with mock.patch.object(billing, "verfuegbare_credits", return_value=None):
            self.assertEqual(self.ex.execute("guthaben_lesen", {})["result"]["verfuegbar"], "unbegrenzt")

    def test_laufende_arbeit(self):
        r = self.ex.execute("laufende_arbeit", {})["result"]
        self.assertEqual(r["anzahl"], 1)
        self.assertEqual(r["laeuft"][0]["project_id"], self.b)
        self.assertTrue(r["laeuft"][0]["projekt_daten"].startswith(DATEN_MARKE))

    def test_unbekanntes_werkzeug(self):
        self.assertFalse(self.ex.execute("dokument_loeschen", {})["ok"])

    def test_prompt_mit_regel_genau_einmal(self):
        _d, _e, system = konto.werkzeugsatz(self.u["id"])
        self.assertEqual(system.count(DATEN_KEINE_ANWEISUNG.split("\n")[0]), 1)
        self.assertIn("Nenne nie interne Werkzeugnamen", system)
        self.assertNotIn("Hochgeladene Dateien und Webadressen", system)
        with mock.patch.object(funktionen, "AGENT_HOCHLADEN", True):
            _d, _e, system = konto.werkzeugsatz(self.u["id"])
        self.assertIn("Hochgeladene Dateien und Webadressen", system)


class Gedaechtnis(unittest.TestCase):
    def test_konto_kontext_marken_und_aktuelle_nicht_doppelt(self):
        u = nutzer()
        p = projekt(u["id"], "Brief (fiktiv)", "word", "docx")
        storage.konto_anhaengen(u["id"], p, "user", "Frage im Brief")
        storage.konto_anhaengen(u["id"], p, "assistant", "Antwort im Brief")
        nid = storage.konto_anhaengen(u["id"], None, "user", "Welche Projekte habe ich?")
        msgs = konto.nachrichten(u["id"], "Welche Projekte habe ich?", ohne_nachricht=nid)
        self.assertTrue(msgs[0]["content"].startswith(konto.KONTO_KOPF))
        self.assertIn(DATEN_MARKE + "„Brief (fiktiv)“", msgs[0]["content"])
        self.assertTrue(msgs[2]["content"].startswith(agent_loop.projekt_marke("Brief (fiktiv)")))
        self.assertEqual(msgs[-1]["content"], agent_loop.OHNE_PROJEKT_MARKE + "\nWelche Projekte habe ich?")
        self.assertEqual(sum("Welche Projekte habe ich?" in m["content"] for m in msgs), 1)

    def test_projektseite_nach_seite_ohne_projekt(self):
        u = nutzer()
        p = projekt(u["id"], "Brief (fiktiv)", "word", "docx")
        storage.konto_anhaengen(u["id"], None, "user", "Hallo")
        storage.konto_anhaengen(u["id"], None, "assistant", "Hallo zurück")
        with mock.patch.object(agent_loop, "build_project_summary", return_value="Zusammenfassung (fiktiv)"):
            c = database.get_db()
            projekt_zeile = dict(c.execute("SELECT * FROM projects WHERE id = ?", (p,)).fetchone())
            c.close()
            msgs = agent_loop._build_initial_messages(p, "Jetzt im Brief", projekt_zeile, u["id"], konto=True)
        self.assertTrue(msgs[2]["content"].startswith(agent_loop.OHNE_PROJEKT_MARKE))
        self.assertTrue(msgs[-1]["content"].startswith(agent_loop.projekt_marke("Brief (fiktiv)")))


class OhneProjektImAgenten(unittest.TestCase):
    """run_agent(None, …): Konto-Werkzeuge, Link-Anhang unter der Antwort (Ersatzmodell statt KI)."""

    def test_projekt_oeffnen_ueber_den_agenten(self):
        u = nutzer()
        p = projekt(u["id"], "Brief (fiktiv)", "word", "docx")

        class Modell:
            def __init__(self):
                self.aufrufe = []

            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3):
                self.aufrufe.append(([t["name"] for t in tools], system))
                if isinstance(anthropic_messages[-1]["content"], list):
                    return {"content": [{"type": "text", "text": "Unter dieser Antwort steht der Link."}], "stop_reason": "end_turn"}
                return {"content": [{"type": "tool_use", "id": "t1", "name": "projekt_oeffnen", "input": {"project_id": p}}],
                        "stop_reason": "tool_use"}
        m = Modell()
        r = agent_loop.run_agent(None, u["id"], "Öffne den Brief", None, m, konto=True)
        self.assertEqual(sorted(m.aufrufe[0][0]), sorted(d["name"] for d in konto.TOOL_DEFINITIONS_KONTO))
        self.assertIn("Seite OHNE Projekt", m.aufrufe[0][1])
        self.assertEqual(r["werkzeuge"], ["projekt_oeffnen"])
        self.assertEqual(r["anhang"][0]["url"], f"/app?projekt={p}")
        self.assertEqual(storage.fokus(u["id"]), p)


class Endpunkte(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
            from inkluagent import chat_engine
        except Exception as e:  # ausserhalb des Containers
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.TestClient, cls.chat_engine = main, TestClient, chat_engine

    def setUp(self):
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        self.user = nutzer()
        self.p = projekt(self.user["id"], "Bericht (fiktiv)")
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.user["id"], self.user["email"], 0))

    def tearDown(self):
        mock.patch.stopall()

    def _an(self):
        mock.patch.object(funktionen, "AGENT_KONTO", True).start()
        mock.patch.object(funktionen, "AGENT_UEBERALL", True).start()

    def test_schalter_aus(self):
        self.assertEqual(self.c.post("/api/agent/nachricht", json={"message": "Hallo"}).status_code, 404)
        self.assertEqual(self.c.get("/api/agent/status").status_code, 404)
        html = self.c.get("/dashboard").text
        self.assertNotIn('id="inkluagentBereich"', html)
        self.assertNotIn("/static/inkluagent", html)
        app = self.c.get(f"/app?projekt={self.p}").text
        from inkluagent import skripte   # die Teile des Agenten-Skripts (17.10.2026)
        for teil in skripte.TEILE:
            self.assertEqual(app.count(f"/static/{teil}?v="), 1, f"ohne Schalter laedt app.html das Skript wie bisher: {teil}")

    def test_nur_ueberall_ohne_konto_bleibt_aus(self):
        with mock.patch.object(funktionen, "AGENT_UEBERALL", True):
            self.assertFalse(self.main._inkluagent_schalter()["ueberall"])
            self.assertEqual(self.c.post("/api/agent/nachricht", json={"message": "Hallo"}).status_code, 404)

    def test_seitenrahmen_mit_schalter(self):
        self._an()
        html = self.c.get("/dashboard").text
        self.assertEqual(html.count('id="inkluagentBereich"'), 1)
        self.assertIn('href="#inkluagentBereich"', html)
        self.assertIn("Zum InkluAgent", html)
        self.assertLess(html.index('id="main"'), html.index('id="inkluagentBereich"'), "der Bereich steht nach dem Hauptinhalt")
        from inkluagent import skripte   # die Teile des Agenten-Skripts (17.10.2026), in Ladereihenfolge
        app = self.c.get(f"/app?projekt={self.p}").text
        for seite in (html, app):
            for teil in skripte.TEILE:
                self.assertEqual(seite.count(f"/static/{teil}?v="), 1, f"genau einmal, sonst doppelte let-Deklaration: {teil}")
            stellen = [seite.index(f"/static/{teil}?v=") for teil in skripte.TEILE]
            self.assertEqual(stellen, sorted(stellen), "in der Reihenfolge der Liste (der Kern mit dem Startaufruf zuletzt)")
        self.assertIn('"ueberall": true', app)
        self.assertIn('"werkzeug_namen"', app)

    def test_nachricht_ohne_projekt(self):
        self._an()
        laeuft = []

        def antwort(user_id, message, **kw):
            laeuft.append(self.main._agent_laeuft(user_id))
            return {"reply": "Du hast ein Projekt (fiktiv).", "intent": "agentic", "werkzeuge": ["projekte_auflisten"],
                    "anhang": [{"art": "link", "url": f"/app?projekt={self.p}", "text": "Projekt öffnen"}]}
        with mock.patch.object(self.chat_engine, "process_konto_message", side_effect=antwort) as pm:
            r = self.c.post("/api/agent/nachricht", json={"message": "Welche Projekte habe ich?"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["reply"], "Du hast ein Projekt (fiktiv).")
        self.assertEqual(laeuft, [True], "waehrend der Antwort meldet der Status „laeuft“")
        self.assertFalse(self.main._agent_laeuft(self.user["id"]))
        self.assertFalse(self.c.get("/api/agent/status").json()["laeuft"])
        kw = pm.call_args.kwargs
        self.assertTrue(kw["ohne_nachricht"])
        v = storage.konto_verlauf(self.user["id"])
        self.assertEqual([(m["role"], m["project_id"]) for m in v], [("user", None), ("assistant", None)])
        self.assertEqual(v[-1]["anhang"][0]["art"], "link")
        self.assertIsNone(storage.fokus(self.user["id"]), "eine Seite ohne Projekt aendert den Fokus nicht")
        d = self.c.get("/api/agent/verlauf").json()
        self.assertEqual(len(d["messages"]), 2)
        self.assertEqual(d["frueher"], 0)

    def test_status_nach_fehler_wieder_frei(self):
        self._an()
        with mock.patch.object(self.chat_engine, "process_konto_message", side_effect=RuntimeError("kaputt (fiktiv)")):
            with self.assertRaises(RuntimeError):
                self.c.post("/api/agent/nachricht", json={"message": "Hallo"})
        self.assertFalse(self.main._agent_laeuft(self.user["id"]))


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        from inkluagent import skripte   # alle Teile des Agenten-Skripts (17.10.2026)
        self.js = skripte.quelltext(FRONTEND)
        self.ab = lies(FRONTEND, "abschluss.js")

    def test_barrierefreiheitspruefung_bindet_den_agenten(self):
        self.assertEqual(self.ab.count("inkluagentHierBinden(project.id);"), 2, "PDF und Word")
        self.assertIn("if (typeof inkluagentUeberall === 'function' && inkluagentUeberall()) inkluagentInit(projectId);", self.ab)

    def test_ansichten_betten_im_festen_bereich_nichts_ein(self):
        f = self.js[self.js.index("function inkluagentSectionHtml("):]
        f = f[:f.index("\n}\n")]
        self.assertIn("if (inkluagentUeberall()) { window._inkluagentVariante = variante || 'bilder'; return ''; }", f)

    def test_bereich_ohne_eigene_section_landmark(self):
        f = self.js[self.js.index("function inkluagentInhaltHtml("):]
        self.assertIn("'<div class=\"inkluagent-section\"", f[:800])

    def test_link_nur_auf_dieser_seite(self):
        # Seit der Pruefung 13.10.2026 (S14) prueft EINE Hilfsfunktion (dashboard.js) Links, Downloads und Rahmen-Ziele
        self.assertIn("if (a && a.art === 'link' && eigenerPfad(a.url)) {", self.js)
        self.assertIn("if (!a || !eigenerPfad(a.download_url)) return;", self.js)

    def test_ohne_projekt_an_den_konto_endpunkt(self):
        f = self.js[self.js.index("async function inkluagentSend("):]
        self.assertIn("ohneProjekt ? '/api/agent/nachricht'", f)
        self.assertIn("if (projectId != null) fd.append('project_id', String(projectId));", self.js)


if __name__ == "__main__":
    unittest.main()
