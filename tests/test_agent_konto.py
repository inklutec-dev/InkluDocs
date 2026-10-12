"""InkluAgent-Ausbau Runde 2, Schritt 2 (12.10.2026): EIN Gespraech je Konto hinter funktionen.AGENT_KONTO (inkluagent/storage.py,
Tabellen agent_nachrichten und agent_fokus, GET /api/agent/verlauf). Dazu zwei Fehler im Gedaechtnis des Agenten, die es auch
ohne Schalter gab: er bekam die AELTESTEN statt der juengsten Nachrichten, und die eben gespeicherte Nachricht doppelt.
Wegwerf-Datenbank (/tmp), keine KI, keine Mails.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_konto.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_konto_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KONTO", "INKLUAGENT_HOCHLADEN", "INKLUAGENT_SICHERHEIT"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, storage  # noqa: E402
from inkluagent.daten import DATEN_MARKE  # noqa: E402

_N = [0]


def konto() -> dict:
    _N[0] += 1
    mail = f"konto{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, name: str, tool: str = "word", typ: str = "docx") -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, ?, ?, 'extracted')", (uid, name, name, tool, typ)).lastrowid
    c.commit()
    c.close()
    return pid


def projekt_zeile(pid: int) -> dict:
    c = database.get_db()
    r = dict(c.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone())
    c.close()
    return r


class Schalter(unittest.TestCase):
    def test_vorgabe_aus(self):
        self.assertFalse(funktionen.AGENT_KONTO)
        with mock.patch.dict(os.environ, {"INKLUAGENT_KONTO": "an"}):
            self.assertTrue(funktionen._umgebung_an("INKLUAGENT_KONTO"))


class Speicher(unittest.TestCase):
    def test_gespraech_ueber_projekte_mit_marke_und_fokus(self):
        u = konto()
        a, b = projekt(u["id"], "Jahresbericht (fiktiv)"), projekt(u["id"], "Flyer (fiktiv)")
        storage.konto_anhaengen(u["id"], a, "user", "Frage zu A")
        storage.konto_anhaengen(u["id"], a, "assistant", "Antwort zu A", werkzeuge=[], anhang=[{"art": "pdf"}])
        storage.konto_anhaengen(u["id"], b, "user", "Frage zu B")
        v = storage.konto_verlauf(u["id"])
        self.assertEqual([m["content"] for m in v], ["Frage zu A", "Antwort zu A", "Frage zu B"])
        self.assertEqual([m["projekt_name"] for m in v], ["Jahresbericht (fiktiv)"] * 2 + ["Flyer (fiktiv)"])
        self.assertEqual(v[1]["werkzeuge"], [])
        self.assertEqual(v[1]["anhang"], [{"art": "pdf"}])
        self.assertEqual(storage.konto_verlauf(konto()["id"]), [], "anderes Konto sieht nichts")
        self.assertEqual([m["content"] for m in storage.konto_verlauf(u["id"], limit=2)], ["Antwort zu A", "Frage zu B"],
                         "juengste zuerst abgeschnitten, aelteste zuerst geliefert")
        self.assertIsNone(storage.fokus(u["id"]))
        storage.fokus_setzen(u["id"], a)
        storage.fokus_setzen(u["id"], b)
        self.assertEqual(storage.fokus(u["id"]), b)

    def test_fremdes_projekt_gibt_keinen_namen_preis(self):
        u, fremd = konto(), konto()
        p = projekt(fremd["id"], "Geheim (fiktiv)")
        storage.konto_anhaengen(u["id"], p, "user", "x")
        self.assertEqual(storage.konto_verlauf(u["id"])[-1]["projekt_name"], "")

    def test_projekt_loeschen_markiert_und_loest_den_fokus(self):
        u = konto()
        p = projekt(u["id"], "Alt (fiktiv)")
        storage.konto_anhaengen(u["id"], p, "user", "Hallo")
        storage.fokus_setzen(u["id"], p)
        c = database.get_db()
        storage.projekt_geloescht_markieren(c, p)
        c.commit()
        c.close()
        m = storage.konto_verlauf(u["id"])[-1]
        self.assertTrue(m["projekt_geloescht"])
        self.assertEqual(m["projekt_name"], "Alt (fiktiv)")
        self.assertIsNone(storage.fokus(u["id"]))

    def test_juengste_nachrichten_eines_projekts(self):
        u = konto()
        p = projekt(u["id"], "Lang (fiktiv)")
        for i in range(30):
            storage.append_message(p, "user" if i % 2 == 0 else "assistant", f"N{i}")
        self.assertEqual([m["content"] for m in storage.get_recent(p, 3)], ["N27", "N28", "N29"])
        self.assertEqual(storage.anzahl(p), 30)

    def test_karte_im_gespraech_des_kontos_aktualisieren(self):
        u = konto()
        p = projekt(u["id"], "Karte (fiktiv)")
        storage.konto_anhaengen(u["id"], p, "assistant", "Angebot", anhang=[{"art": "bestaetigung", "angebot_id": "abc123"}])
        self.assertEqual(storage.karte_aktualisieren(p, "abc123", {"zustand": "erledigt"}), 1)
        self.assertEqual(storage.konto_verlauf(u["id"])[-1]["anhang"][0]["zustand"], "erledigt")

    def test_tagesgrenze_zaehlt_das_gespraech_des_kontos(self):
        u = konto()
        p = projekt(u["id"], "Zaehlen (fiktiv)")
        storage.append_message(p, "user", "alt")
        storage.konto_anhaengen(u["id"], p, "user", "neu")
        storage.konto_anhaengen(u["id"], p, "assistant", "Antwort zaehlt nicht")
        self.assertEqual(database.get_daily_chat_count(u["id"]), 2)

    def test_konto_loeschen_nimmt_alles_mit(self):
        u = konto()
        p = projekt(u["id"], "Weg (fiktiv)")
        storage.konto_anhaengen(u["id"], p, "user", "Hallo")
        storage.fokus_setzen(u["id"], p)
        c = database.get_db()
        c.execute("INSERT INTO agent_zaehler (user_id, schluessel, anzahl) VALUES (?, 'projekte:2026-10-12', 3)", (u["id"],))
        c.execute("INSERT INTO agent_tageszaehler (user_id, tag, nachrichten) VALUES (?, '2026-10-12', 4)", (u["id"],))
        c.commit()
        c.close()
        database.delete_user_data(u["id"])
        c = database.get_db()
        for tabelle in ("agent_nachrichten", "agent_fokus", "agent_zaehler", "agent_tageszaehler"):
            with self.subTest(tabelle=tabelle):
                self.assertEqual(c.execute(f"SELECT COUNT(*) FROM {tabelle} WHERE user_id = ?", (u["id"],)).fetchone()[0], 0)
        c.close()


class Gedaechtnis(unittest.TestCase):
    """_build_initial_messages: juengste Nachrichten, die aktuelle nicht doppelt, Projektmarken, abwechselnde Rollen."""

    def _bauen(self, *a, **kw):
        with mock.patch.object(agent_loop, "build_project_summary", return_value="Zusammenfassung (fiktiv)"):
            return agent_loop._build_initial_messages(*a, **kw)

    def test_projekt_juengste_und_aktuelle_nicht_doppelt(self):
        u = konto()
        p = projekt(u["id"], "P (fiktiv)")
        for i in range(30):
            storage.append_message(p, "user" if i % 2 == 0 else "assistant", f"N{i}")
        nid = storage.append_message(p, "user", "Jetzt")
        msgs = self._bauen(p, "Jetzt", projekt_zeile(p), u["id"], ohne_nachricht=nid)
        texte = [m["content"] for m in msgs]
        self.assertEqual(texte[-1], "Jetzt")
        self.assertEqual(texte.count("Jetzt"), 1, "die eben gespeicherte Nachricht nur einmal")
        self.assertIn("N29", texte)
        self.assertNotIn("N0", texte, "nicht mehr die aeltesten")
        self.assertEqual(msgs[2]["role"], "user", "Gedaechtnis beginnt mit einer Nutzer-Nachricht")
        for x, y in zip(msgs, msgs[1:]):
            self.assertNotEqual(x["role"], y["role"])

    def test_zwei_nutzer_nachrichten_hintereinander_zusammengefasst(self):
        u = konto()
        p = projekt(u["id"], "Q (fiktiv)")
        storage.append_message(p, "user", "Erste (Antwort ging verloren)")
        msgs = self._bauen(p, "Zweite", projekt_zeile(p), u["id"])
        self.assertEqual(msgs[-1], {"role": "user", "content": "Erste (Antwort ging verloren)\n\nZweite"})

    def test_konto_mit_projektmarken(self):
        u = konto()
        a, b = projekt(u["id"], "Jahresbericht (fiktiv)"), projekt(u["id"], "Flyer (fiktiv)")
        storage.konto_anhaengen(u["id"], a, "user", "Frage A")
        storage.konto_anhaengen(u["id"], a, "assistant", "Antwort A")
        nid = storage.konto_anhaengen(u["id"], b, "user", "Frage B")
        msgs = self._bauen(b, "Frage B", projekt_zeile(b), u["id"], konto=True, ohne_nachricht=nid)
        texte = [m["content"] for m in msgs]
        self.assertTrue(texte[2].startswith(agent_loop.projekt_marke("Jahresbericht (fiktiv)")), texte[2])
        self.assertIn(DATEN_MARKE + "„Jahresbericht (fiktiv)“", texte[2], "Projektname als Daten gekennzeichnet")
        self.assertEqual(texte[3], "Antwort A")
        self.assertTrue(texte[-1].startswith(agent_loop.projekt_marke("Flyer (fiktiv)")), "Wechsel vor der neuen Nachricht")
        self.assertTrue(texte[-1].endswith("Frage B"))
        self.assertEqual(sum(t.count("Frage B") for t in texte), 1)

    def test_konto_gleiches_projekt_ohne_neue_marke(self):
        u = konto()
        a = projekt(u["id"], "Nur A (fiktiv)")
        storage.konto_anhaengen(u["id"], a, "user", "Eins")
        storage.konto_anhaengen(u["id"], a, "assistant", "Antwort")
        msgs = self._bauen(a, "Zwei", projekt_zeile(a), u["id"], konto=True)
        self.assertEqual(msgs[-1]["content"], "Zwei")
        self.assertEqual(sum("[Projektwechsel" in m["content"] for m in msgs), 1)

    def test_geloeschtes_projekt_in_der_marke(self):
        self.assertIn("inzwischen gelöscht", agent_loop.projekt_marke("X", True))


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
        self.user = konto()
        self.a = projekt(self.user["id"], "Jahresbericht (fiktiv)")
        self.b = projekt(self.user["id"], "Flyer (fiktiv)")
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.user["id"], self.user["email"], 0))
        self.aufrufe = []

    def tearDown(self):
        mock.patch.stopall()

    def _antwort(self, project_id, message, user_id, **kw):
        self.aufrufe.append((project_id, message, kw))
        return {"reply": f"Antwort zu {project_id} (fiktiv)", "intent": "agentic", "werkzeuge": [],
                "anhang": [{"art": "bestaetigung", "angebot_id": f"ang{project_id}", "zustand": "offen", "titel": "Karte"}]}

    def test_schalter_aus_wie_bisher(self):
        r = self.c.get("/api/agent/verlauf")
        self.assertEqual(r.status_code, 404)
        with mock.patch.object(self.chat_engine, "process_message", side_effect=self._antwort):
            self.c.post(f"/api/projects/{self.a}/chat", json={"message": "Hallo"})
        self.assertEqual([m["content"] for m in storage.get_history(self.a)], ["Hallo", f"Antwort zu {self.a} (fiktiv)"])
        self.assertEqual(storage.konto_verlauf(self.user["id"]), [])
        kw = self.aufrufe[-1][2]
        self.assertFalse(kw.get("konto"))
        self.assertTrue(kw.get("ohne_nachricht"), "auch ohne Schalter: die gespeicherte Nachricht nicht doppelt")

    def test_gespraech_je_konto(self):
        storage.append_message(self.a, "user", "Alt 1")
        storage.append_message(self.a, "assistant", "Alt 2")
        with mock.patch.object(funktionen, "AGENT_KONTO", True), \
                mock.patch.object(self.chat_engine, "process_message", side_effect=self._antwort):
            self.c.post(f"/api/projects/{self.a}/chat", json={"message": "Frage A"})
            self.c.post(f"/api/projects/{self.b}/chat", json={"message": "Frage B"})
            self.assertTrue(self.aufrufe[-1][2].get("konto"))
            self.assertEqual(storage.fokus(self.user["id"]), self.b)
            self.assertEqual(storage.anzahl(self.a), 2, "das alte Projektgespraech bleibt, es kommt nichts dazu")
            # beide Angebote gelten noch (sonst zeigt karten_im_verlauf sie als „Nicht mehr gültig“)
            from inkluagent.tools import ausgaben
            with mock.patch.object(ausgaben, "angebot_nach_id", return_value={"gilt": True}):
                r = self.c.get(f"/api/agent/verlauf?projekt={self.b}")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual([m["content"] for m in d["messages"]],
                         ["Frage A", f"Antwort zu {self.a} (fiktiv)", "Frage B", f"Antwort zu {self.b} (fiktiv)"])
        self.assertEqual([m["projekt_name"] for m in d["messages"]], ["Jahresbericht (fiktiv)"] * 2 + ["Flyer (fiktiv)"] * 2)
        self.assertEqual(d["projekt_name"], "Flyer (fiktiv)")
        self.assertEqual(d["frueher"], 0)
        # offene Karte eines anderen Projekts: ohne Knopf, mit Hinweis auf ihr Projekt; die eigene bleibt offen
        karte_a = d["messages"][1]["anhang"][0]
        karte_b = d["messages"][3]["anhang"][0]
        self.assertEqual(karte_a["zustand"], "anderes_projekt")
        self.assertIn("Jahresbericht (fiktiv)", karte_a["titel"])
        self.assertEqual(karte_b["zustand"], "offen", "die Karte dieses Projekts behaelt ihren Knopf")
        with mock.patch.object(funktionen, "AGENT_KONTO", True):
            d2 = self.c.get(f"/api/agent/verlauf?projekt={self.a}").json()
        self.assertEqual(d2["frueher"], 2, "das fruehere Gespraech dieses Projekts bleibt lesbar")
        self.assertEqual(self.c.get(f"/api/projects/{self.a}/chat/history").json()["messages"][0]["content"], "Alt 1")

    def test_fremdes_projekt_im_verlauf_abgefragt(self):
        fremd = konto()
        p = projekt(fremd["id"], "Geheim (fiktiv)")
        storage.append_message(p, "user", "geheim")
        with mock.patch.object(funktionen, "AGENT_KONTO", True):
            d = self.c.get(f"/api/agent/verlauf?projekt={p}").json()
        self.assertEqual(d["frueher"], 0)
        self.assertEqual(d["projekt_name"], "")

    def test_projekt_loeschen_laesst_das_gespraech_stehen(self):
        with mock.patch.object(funktionen, "AGENT_KONTO", True), \
                mock.patch.object(self.chat_engine, "process_message", side_effect=self._antwort):
            self.c.post(f"/api/projects/{self.a}/chat", json={"message": "Frage A"})
            r = self.c.delete(f"/api/projects/{self.a}")
            self.assertEqual(r.status_code, 200, r.text)
            d = self.c.get(f"/api/agent/verlauf?projekt={self.b}").json()
        self.assertEqual(len(d["messages"]), 2)
        self.assertTrue(all(m["projekt_geloescht"] for m in d["messages"]))


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        backend = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
        frontend = next(k for k in (os.path.normpath(os.path.join(backend, "..", "frontend")), os.path.join(backend, "frontend"))
                        if os.path.isdir(k))
        with open(os.path.join(frontend, "inkluagent.js"), encoding="utf-8") as f:
            self.js = f.read()

    def test_verlauf_des_kontos_nur_mit_schalter(self):
        self.assertIn("window.INKLUAGENT && window.INKLUAGENT.konto", self.js)
        self.assertIn("if (inkluagentKonto()) return inkluagentKontoVerlaufLaden(projectId, log);", self.js)

    def test_frueheres_gespraech_nativ_und_ohne_innerhtml(self):
        start = self.js.index("function inkluagentFrueherZeigen(")
        f = self.js[start:self.js.index("\nasync function inkluagentLoadHistory(")]
        self.assertIn("document.createElement('details')", f)
        self.assertIn("document.createElement('summary')", f)
        self.assertNotIn("innerHTML", f)

    def test_marke_vor_der_eigenen_nachricht(self):
        start = self.js.index("async function inkluagentSend(")
        f = self.js[start:]
        self.assertLess(f.index("inkluagentMarkeVorNachricht(projectId);"), f.index("inkluagentAppendMessage('user', verlaufText);"))


if __name__ == "__main__":
    unittest.main()
