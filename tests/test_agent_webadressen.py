"""InkluAgent: Webadressen in der EIGENEN Nachricht (Steve 12.10.2026, Schalter AGENT_HOCHLADEN + AGENT_KONTO). Kein eigenes
Feld mehr: der Server erkennt Adressen im Text (inkluagent/hochladen.finde_webadressen), legt Karten „Als neues Projekt laden“
und — im Webseiten-Projekt — „Zum Projekt … hinzufügen“ vor und gibt dem Modell eine Notiz. Geladen wird NUR ueber den Knopf
(POST /api/agent/bestaetigen, konto.KontoExecutor „webseite_laden“); die andere Karte derselben Adresse verfaellt.
Wegwerf-Datenbank, keine KI, scan_url als Attrappe.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_webadressen.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_webadressen_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KONTO", "INKLUAGENT_HOCHLADEN", "INKLUAGENT_UEBERALL", "INKLUAGENT_KONTO_WERKZEUGE"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import hochladen, konto, storage  # noqa: E402
from inkluagent.daten import DATEN_MARKE  # noqa: E402
from inkluagent.tools import ausgaben  # noqa: E402

_N = [0]


def nutzer() -> dict:
    _N[0] += 1
    mail = f"webadresse{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, name: str, tool: str, typ: str) -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, ?, ?, 'completed')", (uid, name, name, tool, typ)).lastrowid
    c.commit()
    c.close()
    return pid


def leeren():
    for d in (ausgaben._ANGEBOTE, ausgaben._LETZTES, ausgaben._NACH_ID, ausgaben._ERLEDIGT, ausgaben._VERBRAUCHT):
        d.clear()


class Erkennen(unittest.TestCase):
    def test_adressen_im_text(self):
        f = hochladen.finde_webadressen
        self.assertEqual(f("Schau dir https://www.beispiel.invalid/ueber-uns an."), ["https://www.beispiel.invalid/ueber-uns"])
        self.assertEqual(f("Bitte www.beispiel.invalid, danke"), ["https://www.beispiel.invalid"])
        self.assertEqual(f("(https://a.invalid/x?y=1)"), ["https://a.invalid/x?y=1"])
        self.assertEqual(f("„https://a.invalid“"), ["https://a.invalid"])
        self.assertEqual(f("beispiel.invalid und a@www.b.invalid"), [], "blosse Domain und E-Mail sind keine Webadresse")
        self.assertEqual(f("https://a.invalid https://A.invalid"), ["https://a.invalid"], "doppelt nur einmal")
        self.assertEqual(len(f(" ".join(f"https://s{i}.invalid" for i in range(6)))), 3, "hoechstens drei")
        self.assertEqual(f(""), [])


class Karten(unittest.TestCase):
    def setUp(self):
        leeren()
        self.u = nutzer()

    def test_neues_projekt_und_im_webprojekt_auch_hinzufuegen(self):
        karten, notiz = hochladen.webadressen_karten(self.u["id"], ["https://www.beispiel.invalid/"], None)
        self.assertEqual(len(karten), 1)
        self.assertEqual(karten[0]["knopf"], "Als neues Projekt laden")
        self.assertTrue(karten[0]["konto"])
        self.assertIn(DATEN_MARKE + "„https://www.beispiel.invalid/“", notiz)
        web = {"id": projekt(self.u["id"], "Vereinsseite (fiktiv)", "web", "url"), "name": "Vereinsseite (fiktiv)", "tool": "web"}
        karten, _n = hochladen.webadressen_karten(self.u["id"], ["https://www.beispiel.invalid/"], web)
        self.assertEqual([k["knopf"] for k in karten], ["Als neues Projekt laden", "Zum Projekt „Vereinsseite (fiktiv)“ hinzufügen"])
        pdf = {"id": projekt(self.u["id"], "PDF (fiktiv)", "pdf", "pdf"), "name": "PDF (fiktiv)", "tool": "pdf"}
        karten, _n = hochladen.webadressen_karten(self.u["id"], ["https://www.beispiel.invalid/"], pdf)
        self.assertEqual(len(karten), 1, "in ein PDF-Projekt passt keine Webseite")

    def test_das_modell_kann_nicht_selbst_laden(self):
        self.assertNotIn("webseite_laden", {d["name"] for d in konto.werkzeugsatz(self.u["id"])[0]})
        with mock.patch.object(funktionen, "AGENT_HOCHLADEN", True):
            r = konto.KontoExecutor(self.u["id"]).execute("webseite_laden", {"url": "https://www.beispiel.invalid/"})
        self.assertFalse(r["ok"])
        self.assertIn("Knopf", r["error"])


class MitServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
            from inkluagent import chat_engine
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.TestClient, cls.chat_engine = main, TestClient, chat_engine

    def setUp(self):
        leeren()
        for name in ("AGENT_HOCHLADEN", "AGENT_KONTO", "AGENT_UEBERALL"):
            mock.patch.object(funktionen, name, True).start()
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        self.u = nutzer()
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))
        self.modell = []
        self.scans = []

        async def scan(request, user):
            daten = await request.json()
            self.scans.append(daten)
            pid = daten.get("project_id") or projekt(user["id"], daten["url"], "web", "url")
            return {"project_id": pid}
        mock.patch.object(self.main, "scan_url", side_effect=scan).start()

    def tearDown(self):
        mock.patch.stopall()

    def _antwort(self, *a, **kw):
        self.modell.append((a, kw))
        return {"reply": "Was möchtest du mit der Seite machen? (fiktiv)", "intent": "agentic", "werkzeuge": []}

    def test_adresse_in_der_nachricht_karten_und_knopf_laedt(self):
        web = projekt(self.u["id"], "Vereinsseite (fiktiv)", "web", "url")
        with mock.patch.object(self.chat_engine, "process_message", side_effect=self._antwort):
            r = self.c.post(f"/api/projects/{web}/chat", json={"message": "Nimm bitte https://www.beispiel.invalid/fest dazu."})
        self.assertEqual(r.status_code, 200, r.text)
        karten = [a for a in r.json()["anhang"] if a.get("art") == "bestaetigung"]
        self.assertEqual(len(karten), 2)
        modell_text = self.modell[-1][0][1]
        self.assertIn("[Vom Server: Webadressen in der Nachricht", modell_text)
        self.assertTrue(self.modell[-1][1]["nur_mit_karte"])
        self.assertEqual(self.scans, [], "vor dem Knopf wird nichts geladen")
        v = storage.konto_verlauf(self.u["id"])
        self.assertEqual(v[-1]["anhang"][0]["angebot_id"], karten[0]["angebot_id"], "die Karten stehen im Verlauf")
        # Knopf „Zum Projekt … hinzufügen“
        d = self.c.post("/api/agent/bestaetigen", json={"angebot_id": karten[1]["angebot_id"]})
        self.assertEqual(d.status_code, 200, d.text)
        self.assertEqual(self.scans, [{"url": "https://www.beispiel.invalid/fest", "project_id": web}])
        self.assertIn({"type": "ansicht_aktualisieren"}, d.json()["actions"])
        zustaende = {k["angebot_id"]: k["zustand"] for k in d.json()["actions"] if k.get("type") == "karte"}
        self.assertEqual(zustaende.get(karten[0]["angebot_id"]), "abgelaufen", "die andere Karte verfaellt")
        # die verfallene Karte laedt nicht noch einmal
        d2 = self.c.post("/api/agent/bestaetigen", json={"angebot_id": karten[0]["angebot_id"]})
        self.assertNotEqual(d2.status_code, 200)
        self.assertEqual(len(self.scans), 1)

    def test_ohne_projekt_als_neues_projekt(self):
        with mock.patch.object(self.chat_engine, "process_konto_message", side_effect=self._antwort):
            r = self.c.post("/api/agent/nachricht", json={"message": "www.beispiel.invalid"})
        karten = [a for a in r.json()["anhang"] if a.get("art") == "bestaetigung"]
        self.assertEqual([k["knopf"] for k in karten], ["Als neues Projekt laden"])
        d = self.c.post("/api/agent/bestaetigen", json={"angebot_id": karten[0]["angebot_id"]})
        self.assertEqual(d.status_code, 200, d.text)
        self.assertEqual(self.scans, [{"url": "https://www.beispiel.invalid"}])
        self.assertEqual(d.json()["anhang"][0]["art"], "link")

    def test_ohne_schalter_keine_karten(self):
        with mock.patch.object(funktionen, "AGENT_HOCHLADEN", False), \
                mock.patch.object(self.chat_engine, "process_konto_message", side_effect=self._antwort):
            r = self.c.post("/api/agent/nachricht", json={"message": "https://www.beispiel.invalid"})
        self.assertFalse([a for a in r.json()["anhang"] if a.get("art") == "bestaetigung"])
        self.assertNotIn("[Vom Server: Webadressen", self.modell[-1][0][1])


if __name__ == "__main__":
    unittest.main()
