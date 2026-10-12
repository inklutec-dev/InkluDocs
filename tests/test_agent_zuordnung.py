"""InkluAgent: Zuordnung zu vorhandenen Projekten und Seitenwechsel (Steve 12.10.2026). Neues Projekt ueber den Agenten: erst
fragen, wenn es ein aehnliches gibt. Datei auf einer Seite ohne Projekt: gibt es ein aehnliches, wartet sie (Eingang, 30 Minuten)
auf die Karte „In … hochladen“ / „Als neues Projekt hochladen“. Geht es danach um genau ein Projekt, wechselt die Seite dorthin.
Wegwerf-Datenbank, keine KI, Upload als Attrappe.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_zuordnung.py
"""
import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_zuordnung_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KONTO", "INKLUAGENT_HOCHLADEN", "INKLUAGENT_UEBERALL", "INKLUAGENT_KONTO_WERKZEUGE",
              "INKLUAGENT_MAX_NEUE_PROJEKTE_TAG"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, hochladen, konto, zuordnung  # noqa: E402
from inkluagent.tools import ausgaben  # noqa: E402

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_N = [0]


def nutzer() -> dict:
    _N[0] += 1
    mail = f"zuordnung{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, name: str, tool: str = "pdf", typ: str = "pdf") -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, ?, ?, 'extracted')", (uid, name, name, tool, typ)).lastrowid
    c.commit()
    c.close()
    return pid


def leeren():
    for d in (ausgaben._ANGEBOTE, ausgaben._LETZTES, ausgaben._NACH_ID, ausgaben._ERLEDIGT, ausgaben._VERBRAUCHT):
        d.clear()


class Aehnlich(unittest.TestCase):
    def test_namen(self):
        a = zuordnung.aehnlich
        self.assertTrue(a("Jahresbericht_2026_v2.pdf", "Jahresbericht 2026"))
        self.assertTrue(a("Flyer Sommerfest", "Sommerfest Flyer"))
        self.assertTrue(a("antrag_pflege.pdf", "Antrag Pflege"))
        self.assertFalse(a("Satzung", "Jahresbericht"))
        self.assertFalse(a("", "Jahresbericht"))
        self.assertFalse(a("ab", "ab cd ef"), "unter vier Zeichen zaehlt Enthalten nicht")

    def test_nur_gleiche_art_und_eigene(self):
        u, fremd = nutzer(), nutzer()
        p = projekt(u["id"], "Jahresbericht 2026")
        projekt(u["id"], "Jahresbericht 2026 Word", "word", "docx")
        projekt(fremd["id"], "Jahresbericht 2026")
        self.assertEqual([x["project_id"] for x in zuordnung.aehnliche_projekte(u["id"], "Jahresbericht_2026.pdf", "pdf")], [p])
        self.assertEqual(len(zuordnung.aehnliche_projekte(u["id"], "Jahresbericht 2026.docx", "docx")), 1)
        self.assertEqual(zuordnung.aehnliche_projekte(u["id"], "Foto.png", "bild"), [])


class AnlegenUndOeffnen(unittest.TestCase):
    def setUp(self):
        leeren()
        mock.patch.object(funktionen, "AGENT_KONTO_WERKZEUGE", True).start()
        self.u = nutzer()
        self.ex = konto.KontoExecutor(self.u["id"])

    def tearDown(self):
        mock.patch.stopall()

    def test_aehnliches_projekt_erst_fragen(self):
        try:
            import main  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        vorhanden = projekt(self.u["id"], "Jahresbericht 2026")
        r = self.ex.execute("projekt_anlegen", {"name": "Jahresbericht 2026 neu", "art": "pdf"})
        self.assertTrue(r["result"]["nichts_angelegt"])
        self.assertEqual(r["result"]["aehnliche_projekte"][0]["project_id"], vorhanden)
        self.assertNotIn("aktionen", r)
        r2 = self.ex.execute("projekt_anlegen", {"name": "Jahresbericht 2026 neu", "art": "pdf", "trotzdem": True})
        self.assertTrue(r2["ok"], r2)
        neu = r2["result"]["project_id"]
        self.assertEqual(r2["aktionen"], [{"type": "seite_wechseln", "url": f"/app?projekt={neu}", "project_id": neu}])

    def test_oeffnen_wechselt_die_seite(self):
        p = projekt(self.u["id"], "Brief")
        r = self.ex.execute("projekt_oeffnen", {"project_id": p})
        self.assertEqual(r["aktionen"][0]["url"], f"/app?projekt={p}")

    def test_aktionen_kommen_aus_dem_agenten(self):
        p = projekt(self.u["id"], "Brief")

        class Modell:
            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3):
                if isinstance(anthropic_messages[-1]["content"], list):
                    return {"content": [{"type": "text", "text": "Ich öffne den Brief."}], "stop_reason": "end_turn"}
                return {"content": [{"type": "tool_use", "id": "t", "name": "projekt_oeffnen", "input": {"project_id": p}}],
                        "stop_reason": "tool_use"}
        r = agent_loop.run_agent(None, self.u["id"], "Öffne den Brief", None, Modell(), konto=True)
        self.assertIn({"type": "seite_wechseln", "url": f"/app?projekt={p}", "project_id": p}, r["actions"])


class Eingang(unittest.TestCase):
    """Datei auf einer Seite ohne Projekt mit aehnlichem Projekt: wartet, Karten, Knopf legt ab."""

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
        mock.patch.object(self.main, "RESULTS_DIR", os.path.join(TMP, "results")).start()
        self.u = nutzer()
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))
        self.uploads = []

        async def upload(file, project_id, user, request=None):
            await file.read()
            self.uploads.append((file.filename, project_id))
            pid = project_id or projekt(user["id"], file.filename)
            return {"project_id": pid, "status": "extracted"}
        mock.patch.object(self.main, "upload_file", side_effect=upload).start()
        mock.patch.object(hochladen, "_fakten", side_effect=lambda pid, did: {
            "projekt": f"Projekt {pid}", "status": "extracted", "bilder": 0, "felder": 0, "seiten": 1, "getaggt": False, "tool": "pdf"}).start()

    def tearDown(self):
        mock.patch.stopall()

    def _hochladen(self, name):
        return self.c.post("/api/agent/hochladen", files=[("dateien", (name, PDF, "application/pdf"))]).json()

    def _antwort(self, *a, **kw):
        return {"reply": "Wohin soll die Datei? (fiktiv)", "intent": "agentic", "werkzeuge": []}

    def test_wartet_und_knopf_legt_ins_vorhandene_projekt(self):
        vorhanden = projekt(self.u["id"], "Jahresbericht 2026")
        erg = self._hochladen("Jahresbericht_2026_v2.pdf")
        e = erg["ergebnisse"][0]
        self.assertTrue(e["wartet"])
        self.assertEqual(self.uploads, [], "noch nichts abgelegt")
        self.assertIn("Wartet auf deine Entscheidung: Jahresbericht_2026_v2.pdf", erg["kurz"])
        with mock.patch.object(self.chat_engine, "process_konto_message", side_effect=self._antwort):
            r = self.c.post("/api/agent/nachricht", json={"message": "", "hochgeladen": erg["notiz_id"]}).json()
        karten = [a for a in r["anhang"] if a.get("art") == "bestaetigung"]
        self.assertEqual([k["knopf"] for k in karten], ["In „Jahresbericht 2026“ hochladen", "Als neues Projekt hochladen"])
        self.assertFalse([a for a in r["actions"] if a.get("type") == "seite_wechseln"], "erst nach der Wahl wechseln")
        d = self.c.post("/api/agent/bestaetigen", json={"angebot_id": karten[0]["angebot_id"]})
        self.assertEqual(d.status_code, 200, d.text)
        self.assertEqual(self.uploads, [("Jahresbericht_2026_v2.pdf", vorhanden)])
        self.assertIn({"type": "seite_wechseln", "url": f"/app?projekt={vorhanden}", "project_id": vorhanden}, d.json()["actions"])
        zustaende = {k["angebot_id"]: k["zustand"] for k in d.json()["actions"] if k.get("type") == "karte"}
        self.assertEqual(zustaende.get(karten[1]["angebot_id"]), "abgelaufen", "die andere Karte verfaellt")
        self.assertEqual(os.listdir(os.path.join(TMP, "results", str(self.u["id"]), "_agent_eingang")), [], "Eingang geleert")

    def test_ohne_aehnliches_sofort_und_seite_wechselt(self):
        erg = self._hochladen("Ganz neu.pdf")
        self.assertFalse(erg["ergebnisse"][0].get("wartet"))
        self.assertEqual(len(self.uploads), 1)
        with mock.patch.object(self.chat_engine, "process_konto_message", side_effect=self._antwort):
            r = self.c.post("/api/agent/nachricht", json={"message": "", "hochgeladen": erg["notiz_id"]}).json()
        pid = erg["projekte"][0]
        self.assertIn({"type": "seite_wechseln", "url": f"/app?projekt={pid}", "project_id": pid}, r["actions"])

    def test_im_projekt_kein_warten(self):
        vorhanden = projekt(self.u["id"], "Jahresbericht 2026")
        aktuell = projekt(self.u["id"], "Anderes PDF")
        r = self.c.post("/api/agent/hochladen", data={"project_id": str(aktuell)},
                        files=[("dateien", ("Jahresbericht_2026_v2.pdf", PDF, "application/pdf"))]).json()
        self.assertFalse(r["ergebnisse"][0].get("wartet"))
        self.assertEqual(self.uploads, [("Jahresbericht_2026_v2.pdf", aktuell)])
        self.assertNotEqual(vorhanden, aktuell)

    def test_abgelaufene_oder_fremde_datei(self):
        self.assertIsNone(hochladen.eingang_pfad(self.u["id"], "../../etc/passwd"))
        self.assertIsNone(hochladen.eingang_pfad(self.u["id"], "0" * 32 + ".pdf"))
        datei = hochladen.eingang_ablegen(self.u["id"], "x.pdf", PDF)
        self.assertTrue(hochladen.eingang_pfad(self.u["id"], datei))
        pfad = os.path.join(TMP, "results", str(self.u["id"]), "_agent_eingang", datei)
        os.utime(pfad, (1, 1))
        self.assertIsNone(hochladen.eingang_pfad(self.u["id"], datei), "nach 30 Minuten weg")
        self.assertFalse(os.path.exists(pfad))


class Oberflaeche(unittest.TestCase):
    def test_seitenwechsel_nur_auf_projektseiten(self):
        with open(os.path.join(FRONTEND, "inkluagent.js"), encoding="utf-8") as f:
            js = f.read()
        self.assertIn("/^\\/app\\?projekt=\\d+$/.test(x.url)", js)
        self.assertIn("window.location.href = a.url + '&agent=offen';", js)
        self.assertIn("if (inkluagentSeiteWechseln(data.actions))", js)


if __name__ == "__main__":
    unittest.main()
