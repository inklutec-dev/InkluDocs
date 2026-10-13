"""InkluAgent-Ausbau Runde 2, Schritt 4 (12.10.2026): Konto-Werkzeuge, die etwas aendern, hinter
funktionen.AGENT_KONTO_WERKZEUGE (inkluagent/konto.py, POST /api/agent/bestaetigen). Projekt anlegen, umbenennen, loeschen (NUR
ueber den Knopf der Karte, nie per getipptem Ja, nicht waehrend eines Laufs), Lauf abbrechen — je ueber denselben Endpunkt wie
der Knopf der Oberflaeche. Wegwerf-Datenbank (/tmp), keine KI, keine Mails.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_konto_werkzeuge.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_konto_werkzeuge_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KONTO", "INKLUAGENT_UEBERALL", "INKLUAGENT_KONTO_WERKZEUGE", "INKLUAGENT_HOCHLADEN",
              "INKLUAGENT_SICHERHEIT", "INKLUAGENT_MAX_NEUE_PROJEKTE_TAG"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import konto, storage  # noqa: E402
from inkluagent.daten import DATEN_MARKE  # noqa: E402
from inkluagent.tools import ausgaben  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_N = [0]


def nutzer() -> dict:
    _N[0] += 1
    mail = f"kontowerkzeug{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, name: str, status: str = "extracted", tool: str = "pdf", typ: str = "pdf") -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, ?, ?, ?)", (uid, name, name, tool, typ, status)).lastrowid
    c.commit()
    c.close()
    return pid


def gibt_es(pid: int) -> bool:
    c = database.get_db()
    r = c.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone()
    c.close()
    return bool(r)


def name_von(pid: int) -> str:
    c = database.get_db()
    r = c.execute("SELECT name FROM projects WHERE id = ?", (pid,)).fetchone()
    c.close()
    return r["name"] if r else ""


def leeren():
    for d in (ausgaben._ANGEBOTE, ausgaben._LETZTES, ausgaben._NACH_ID, ausgaben._ERLEDIGT, ausgaben._VERBRAUCHT):
        d.clear()


AN = mock.patch.object(funktionen, "AGENT_KONTO_WERKZEUGE", True)
OHNE_MAIN = mock.MagicMock()
OHNE_MAIN.get_gettext.return_value = (lambda s: s)
OHNE_MAIN.token_gueltig_bis.return_value = None


class Schalter(unittest.TestCase):
    def test_vorgabe_aus_und_werkzeugsatz(self):
        self.assertFalse(funktionen.AGENT_KONTO_WERKZEUGE)
        defs, _e, system = konto.werkzeugsatz(1)
        self.assertEqual({d["name"] for d in defs}, {d["name"] for d in konto.TOOL_DEFINITIONS_KONTO})
        self.assertIn("das geht über die Oberfläche", system)
        self.assertNotIn("Was du hier außerdem ändern kannst", system)
        r = konto.KontoExecutor(1).execute("projekt_anlegen", {"name": "x", "art": "pdf"})
        self.assertFalse(r["ok"], "ohne Schalter fuehrt der Executor nichts Aenderndes aus")
        with AN:
            defs, _e, system = konto.werkzeugsatz(1)
        self.assertEqual(len(defs), 8)
        self.assertIn("Was du hier außerdem ändern kannst", system)
        self.assertIn("NUR der Knopf löscht", system)
        self.assertNotIn("bestaetigt", str([d["input_schema"] for d in konto.TOOL_DEFINITIONS_KONTO_AENDERN]),
                         "das Modell kann Loeschen nicht selbst bestaetigen")


class MitServer(unittest.TestCase):
    """Die Werkzeuge rufen die Endpunkte der Knoepfe (main) — ausserhalb des Containers uebersprungen."""

    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.TestClient = main, TestClient

    def setUp(self):
        leeren()
        AN.start()
        mock.patch.object(ausgaben, "_main", return_value=OHNE_MAIN).start()
        self.u = nutzer()
        self.ex = konto.KontoExecutor(self.u["id"])

    def tearDown(self):
        mock.patch.stopall()

    def test_anlegen_mit_link_und_bremse(self):
        r = self.ex.execute("projekt_anlegen", {"name": "Jahresbericht 2027 (fiktiv)", "art": "word"})
        self.assertTrue(r["ok"], r)
        pid = r["result"]["project_id"]
        self.assertEqual(name_von(pid), "Jahresbericht 2027 (fiktiv)")
        self.assertEqual(r["result"]["name_daten"], DATEN_MARKE + "Jahresbericht 2027 (fiktiv)")
        self.assertEqual(r["anhang"]["url"], f"/app?projekt={pid}")
        self.assertFalse(self.ex.execute("projekt_anlegen", {"name": "x", "art": "powerpoint"})["ok"])
        with mock.patch.dict(os.environ, {"INKLUAGENT_MAX_NEUE_PROJEKTE_TAG": "1"}):
            r2 = self.ex.execute("projekt_anlegen", {"name": "Zweites (fiktiv)", "art": "pdf"})
        self.assertFalse(r2["ok"])
        self.assertIn("Morgen", r2["error"])

    def test_umbenennen_nur_eigene(self):
        pid = projekt(self.u["id"], "Alt (fiktiv)")
        r = self.ex.execute("projekt_umbenennen", {"project_id": pid, "name": "Neu (fiktiv)"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(name_von(pid), "Neu (fiktiv)")
        fremd = projekt(nutzer()["id"], "Fremd (fiktiv)")
        self.assertFalse(self.ex.execute("projekt_umbenennen", {"project_id": fremd, "name": "Gekapert"})["ok"])
        self.assertEqual(name_von(fremd), "Fremd (fiktiv)")

    def test_loeschen_nur_ueber_den_knopf(self):
        pid = projekt(self.u["id"], "Weg damit (fiktiv)")
        r = self.ex.execute("projekt_loeschen", {"project_id": pid})
        karte = r.get("anhang") or {}
        self.assertEqual(karte.get("art"), "bestaetigung")
        self.assertTrue(karte.get("konto"))
        self.assertIn("Weg damit (fiktiv)", karte.get("text", ""))
        self.assertIn("nicht rückgängig", karte.get("text", ""))
        self.assertNotIn("dokument", r["result"], "der Name geht nur gekennzeichnet ans Modell")
        self.assertTrue(gibt_es(pid))
        # das Modell versucht es mit bestaetigt (getipptes Ja): nichts passiert
        ex2 = konto.KontoExecutor(self.u["id"])
        r2 = ex2.execute("projekt_loeschen", {"project_id": pid, "bestaetigt": True})
        self.assertTrue(r2["result"]["rueckfrage_noetig"])
        self.assertIn("nur der Knopf", r2["result"]["hinweis"])
        self.assertTrue(gibt_es(pid))
        # die Karte steht im Gespraech; der Knopf loescht
        storage.konto_anhaengen(self.u["id"], None, "assistant", "Bitte bestätigen (fiktiv)", anhang=[karte])
        c = self.TestClient(self.main.app)
        c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))
        with mock.patch.object(self.main, "INKLUAGENT_ENABLED", True), mock.patch.object(funktionen, "AGENT_KONTO", True), \
                mock.patch.object(funktionen, "AGENT_UEBERALL", True):
            # Runde 2 (S-N3): arbeitet der Agent fuer das Konto, wartet der Knopf (409), und die Karte bleibt gueltig
            start = self.main._lauf_anmelden(self.u["id"])
            try:
                besetzt = c.post("/api/agent/bestaetigen", json={"angebot_id": karte["angebot_id"]})
            finally:
                self.main._lauf_abmelden(self.u["id"], start)
            self.assertEqual(besetzt.status_code, 409, besetzt.text)
            self.assertIn("arbeitet noch", besetzt.json()["detail"])
            self.assertTrue(gibt_es(pid))
            antwort = c.post("/api/agent/bestaetigen", json={"angebot_id": karte["angebot_id"]})
            self.assertEqual(antwort.status_code, 200, antwort.text)
            d = antwort.json()
            self.assertFalse(gibt_es(pid))
            self.assertIn("ist gelöscht", d["reply"])
            self.assertIn({"type": "projekt_geloescht", "project_id": pid}, d["actions"])
            self.assertEqual(d["karte"]["zustand"], "erledigt")
            zweimal = c.post("/api/agent/bestaetigen", json={"angebot_id": karte["angebot_id"]})
            self.assertEqual(zweimal.status_code, 409)
        v = storage.konto_verlauf(self.u["id"])
        self.assertEqual(v[0]["anhang"][0]["zustand"], "erledigt", "die Karte im Verlauf zeigt „Erledigt“")
        self.assertEqual([(m["role"], m["project_id"]) for m in v[1:]], [("user", None), ("assistant", None)])
        # Runde 2 (S-N1/K-N2): die Zeile „Bestätigt über den Knopf: …“ ist eine Zeile des SERVERS — nicht getippt, im Gedaechtnis
        # als Daten gekennzeichnet (der Projektname darin ist Fremdtext)
        frage = v[1]
        self.assertEqual(frage["text"], "")
        self.assertTrue(frage["zusatz"].startswith("Bestätigt über den Knopf:"), frage)
        self.assertIn("Weg damit (fiktiv)", frage["zusatz"])
        self.assertTrue(storage.fuer_modell(frage).startswith(DATEN_MARKE), storage.fuer_modell(frage))

    def test_nicht_waehrend_eines_laufs(self):
        pid = projekt(self.u["id"], "Läuft (fiktiv)", status="processing")
        r = self.ex.execute("projekt_loeschen", {"project_id": pid})
        self.assertFalse(r["ok"])
        self.assertIn("läuft gerade", r["error"])
        self.assertNotIn("anhang", r)

    def test_fremdes_projekt_nicht_loeschen(self):
        fremd = projekt(nutzer()["id"], "Fremd (fiktiv)")
        self.assertFalse(self.ex.execute("projekt_loeschen", {"project_id": fremd})["ok"])
        self.assertTrue(gibt_es(fremd))

    def test_knopf_ohne_schalter(self):
        c = self.TestClient(self.main.app)
        c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))
        with mock.patch.object(self.main, "INKLUAGENT_ENABLED", True), mock.patch.object(funktionen, "AGENT_KONTO_WERKZEUGE", False):
            self.assertEqual(c.post("/api/agent/bestaetigen", json={"angebot_id": "x"}).status_code, 404)

    def test_lauf_abbrechen(self):
        pid = projekt(self.u["id"], "Generierung (fiktiv)", status="processing")
        r = self.ex.execute("lauf_abbrechen", {"project_id": pid})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["abgebrochen"], ["Alt-Text-Generierung"])
        self.assertIn(pid, self.main._abbruch_gewuenscht)
        self.main._abbruch_gewuenscht.discard(pid)
        ruhig = projekt(self.u["id"], "Ruhig (fiktiv)")
        r2 = self.ex.execute("lauf_abbrechen", {"project_id": ruhig})
        self.assertEqual(r2["result"]["abgebrochen"], [])
        self.assertIn("lief nichts", r2["result"]["hinweis"])


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(FRONTEND, "inkluagent.js"), encoding="utf-8") as f:
            self.js = f.read()

    def test_konto_karte_an_den_konto_endpunkt(self):
        self.assertIn("konto ? '/api/agent/bestaetigen'", self.js)
        self.assertIn("inkluagentBestaetigen(a.angebot_id, b, st, !!a.konto)", self.js)

    def test_geloeschtes_projekt_ohne_innerhtml_und_ohne_sprung(self):
        f = self.js[self.js.index("function inkluagentProjektWeg("):]
        f = f[:f.index("\n}\n")]
        self.assertNotIn("innerHTML", f)
        self.assertNotIn("location", f, "kein automatischer Seitenwechsel")


if __name__ == "__main__":
    unittest.main()
