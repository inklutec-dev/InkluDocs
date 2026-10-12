"""Agentenansicht wie ChatGPT/Claude mit Arbeitsfenster (Michael Karbe, Feedback 20261012 - 1, Punkt 6; Steve 12.10.2026;
Schalter funktionen.AGENT_ARBEITSFENSTER). Server: woran der Agent arbeitet (inkluagent/arbeitsfenster.py) — Nummern wie in der
Oberflaeche, nur eigene Daten des Projekts, nur ganze Zahlen aus den Argumenten; der Stream nennt es nur mit Schalter.
Oberflaeche am Quelltext (Aufbau, stille Schritte, Seitenleiste). Wegwerf-Datenbank, keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_arbeitsfenster.py
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_arbeitsfenster_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_ARBEITSFENSTER", "INKLUAGENT_ANSICHT", "INKLUAGENT_KONTO", "INKLUAGENT_UEBERALL", "INKLUAGENT_GESPRAECHE"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import arbeitsfenster  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_N = [0]


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def nutzer() -> dict:
    _N[0] += 1
    mail = f"arbeitsfenster{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, dokumente=("Jahresbericht (fiktiv).pdf",)) -> tuple:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, 'x.pdf', '', 'Projekt (fiktiv)', 'pdf', 'pdf', 'extracted')", (uid,)).lastrowid
    docs = [c.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, ?, ?, '')",
                      (pid, i + 1, name)).lastrowid for i, name in enumerate(dokumente)]
    c.commit()
    c.close()
    return pid, docs


def bild(pid: int, did: int, seite: int, index: int = 0, seitenansicht: str = "") -> int:
    c = database.get_db()
    iid = c.execute("INSERT INTO images (project_id, document_id, page_number, image_index, image_path, page_view_path) "
                    "VALUES (?, ?, ?, ?, '/x.png', ?)", (pid, did, seite, index, seitenansicht)).lastrowid
    c.commit()
    c.close()
    return iid


class Ziel(unittest.TestCase):
    def setUp(self):
        self.u, self.fremd = nutzer(), nutzer()
        self.p, (self.d1, self.d2) = projekt(self.u["id"], ("Jahresbericht (fiktiv).pdf", "Anhang (fiktiv).pdf"))
        # Reihenfolge wie die Oberflaeche: je Dokument nach Seite, Index, Kennung — b ist „Bild 1“, a „Bild 2“
        self.a = bild(self.p, self.d1, 2, 0, "/data/seite2.png")
        self.b = bild(self.p, self.d1, 1, 0)
        self.c = bild(self.p, self.d2, 1, 0)

    def test_bild_mit_nummer_seite_und_dokument_wie_die_oberflaeche(self):
        z = arbeitsfenster.ziel(self.u["id"], self.p, {"image_id": self.a, "new_alt_text": "egal"})
        self.assertEqual(z, {"art": "bild", "nummer": 2, "seite": 2, "bild_url": f"/api/images/{self.a}/file",
                             "seite_url": f"/api/images/{self.a}/page-view", "dokument": "Jahresbericht (fiktiv).pdf",
                             "dokument_nr": 1})
        z = arbeitsfenster.ziel(self.u["id"], self.p, {"image_id": self.c})
        self.assertEqual((z["nummer"], z["dokument_nr"], z["seite_url"]), (1, 2, None), "Nummer je Dokument, ohne Seitenansicht keine")

    def test_ein_dokument_ohne_dokumentnummer(self):
        p, (d,) = projekt(self.u["id"])
        z = arbeitsfenster.ziel(self.u["id"], p, {"image_id": bild(p, d, 1)})
        self.assertIsNone(z["dokument_nr"])
        self.assertEqual(z["nummer"], 1)

    def test_feld_und_dokument(self):
        c = database.get_db()
        fid = c.execute("INSERT INTO formularfelder (project_id, document_id, feld_index, anker, page_number) VALUES (?, ?, 4, 'Name', 3)",
                        (self.p, self.d2)).lastrowid
        c.commit()
        c.close()
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {"feld_id": fid}),
                         {"art": "feld", "nummer": 4, "seite": 3, "dokument": "Anhang (fiktiv).pdf", "dokument_nr": 2})
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {"document_id": self.d1}),
                         {"art": "dokument", "dokument": "Jahresbericht (fiktiv).pdf", "dokument_nr": 1})

    def test_nur_eigene_daten_dieses_projekts(self):
        self.assertIsNone(arbeitsfenster.ziel(self.fremd["id"], self.p, {"image_id": self.a}), "fremdes Konto")
        anderes, _ = projekt(self.u["id"])
        self.assertIsNone(arbeitsfenster.ziel(self.u["id"], anderes, {"image_id": self.a}), "Bild aus einem anderen Projekt")
        self.assertIsNone(arbeitsfenster.ziel(self.u["id"], anderes, {"document_id": self.d1}))
        self.assertIsNone(arbeitsfenster.ziel(self.u["id"], self.p, {"image_id": 999999}))

    def test_nur_ganze_zahlen_aus_den_argumenten(self):
        for args in ({"image_id": True}, {"image_id": 1.5}, {"image_id": "../1"}, {"image_id": None}, {}, "image_id", None):
            self.assertIsNone(arbeitsfenster.ziel(self.u["id"], self.p, args), args)
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {"image_id": str(self.b)})["nummer"], 1)
        self.assertIsNone(arbeitsfenster.ziel(self.u["id"], None, {"image_id": self.a}), "Seiten ohne Projekt: kein Ziel")

    def test_nie_eine_ausnahme(self):
        with mock.patch("database.get_db", side_effect=RuntimeError("kaputt")):
            self.assertIsNone(arbeitsfenster.ziel(self.u["id"], self.p, {"image_id": self.a}))


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
        for name in ("AGENT_KONTO", "AGENT_UEBERALL", "AGENT_GESPRAECHE"):
            mock.patch.object(funktionen, name, True).start()
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        self.u = nutzer()
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))
        self.p, (d,) = projekt(self.u["id"])
        self.b = bild(self.p, d, 1)

    def tearDown(self):
        mock.patch.stopall()

    def _strom(self):
        def antwort(*a, on_tool=None, **kw):
            on_tool("view_image", {"image_id": self.b})
            on_tool("list_project_images", {})
            return {"reply": "Fertig (fiktiv).", "intent": "agentic", "werkzeuge": ["view_image", "list_project_images"]}
        with mock.patch.object(self.chat_engine, "process_message", side_effect=antwort):
            r = self.c.post(f"/api/projects/{self.p}/chat", json={"message": "Schau dir Bild 1 an."},
                            headers={"Accept": "application/x-ndjson"})
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(z) for z in r.text.splitlines() if z.strip()]

    def test_stream_nennt_das_ziel_nur_mit_schalter(self):
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            zeilen = self._strom()
        werkzeuge = [z for z in zeilen if z["type"] == "tool"]
        self.assertEqual(werkzeuge[0]["ziel"]["art"], "bild")
        self.assertEqual(werkzeuge[0]["ziel"]["nummer"], 1)
        self.assertNotIn("ziel", werkzeuge[1], "Werkzeug ohne Bild, Feld oder Dokument: kein Ziel")
        self.assertEqual(zeilen[-1]["type"], "reply")
        zeilen = self._strom()
        self.assertTrue(all("ziel" not in z for z in zeilen if z["type"] == "tool"), "ohne Schalter wie bisher")

    def test_schalter_fuer_die_seite(self):
        s = self.main._inkluagent_schalter
        self.assertFalse(funktionen.AGENT_ARBEITSFENSTER, "Vorgabe aus")
        self.assertFalse(s()["arbeitsfenster"])
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            self.assertFalse(s()["arbeitsfenster"], "braucht die Agentenansicht")
            with mock.patch.object(funktionen, "AGENT_ANSICHT", True):
                self.assertTrue(s()["arbeitsfenster"])
                with mock.patch.object(funktionen, "AGENT_UEBERALL", False):
                    self.assertFalse(s()["arbeitsfenster"], "braucht den Agenten auf jeder Seite")

    def test_gespraeche_fuer_die_seitenleiste_begrenzt(self):
        from inkluagent import storage
        for i in range(4):
            storage.konto_anhaengen(self.u["id"], None, "user", f"Frage {i} (fiktiv)")
            storage.neues_gespraech(self.u["id"])
        r = self.c.get("/api/agent/gespraeche?anzahl=2")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([g["titel"] for g in r.json()["gespraeche"]], ["Frage 3 (fiktiv)", "Frage 2 (fiktiv)"])
        self.assertEqual(len(self.c.get("/api/agent/gespraeche").json()["gespraeche"]), 4)


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")

    def funktion(self, name):
        f = self.js[self.js.index("function " + name + "("):]
        return f[:f.index("\n}\n")]

    def test_nur_in_der_agentenansicht_und_umkehrbar(self):
        self.assertIn("if (inkluagentArbeitsfensterAn()) { inkluagentChatstil(true);", self.funktion("inkluagentAgentenansicht"))
        self.assertIn("inkluagentChatstil(false);", self.funktion("inkluagentAgentenansichtAufheben"))
        an = self.funktion("inkluagentArbeitsfensterAn")
        self.assertIn("window.INKLUAGENT.arbeitsfenster", an)
        self.assertIn("inkluagentUeberall()", an)

    def test_aufbau_verlauf_eingabe_beschreibung_dann_arbeitsfenster(self):
        f = self.funktion("inkluagentChatstil")
        self.assertIn("[frueher, log, form, leiste, trenner, beschreibung].forEach", f, "Chatstil: Verlauf vor der Eingabe")
        self.assertIn("[form, log, leiste, frueher, trenner, beschreibung].forEach", f, "zurueck: Eingabe zuerst (Michael, Punkt 1)")
        self.assertIn("spalten.appendChild(inkluagentArbeitsfensterBauen())", f)
        bau = self.funktion("inkluagentArbeitsfensterBauen")
        self.assertIn("document.createElement('section')", bau)
        self.assertIn("aria-labelledby', 'arbeitsfensterTitel'", bau)
        self.assertIn("'<h2 id=\"arbeitsfensterTitel\">' + t('Arbeitsfenster')", bau)
        self.assertIn("t('Stand vorlesen')", bau)

    def test_schritte_still_ansagen_nur_bei_start_ende_rueckfrage_fehler(self):
        send = self.funktion("inkluagentSend")
        self.assertIn("if (chatstil) inkluagentArbeitsSchritt(ev.name, ev.ziel);\n                        else inkluagentSetStatus(", send)
        self.assertIn("t('InkluAgent arbeitet. Was er gerade tut, steht im Arbeitsfenster.')", send)
        self.assertIn("inkluagentArbeitsEnde(inkluagentWartetAufDich(data.anhang))", send)
        self.assertIn("inkluagentArbeitsFehler(text)", self.funktion("inkluagentSetStatus"))
        self.assertNotIn("announce(", self.funktion("inkluagentArbeitsSchritt"))
        self.assertIn("announce(", self.funktion("inkluagentStandVorlesen"))

    def test_daten_nur_als_text_und_nur_eigene_bildadressen(self):
        zeigen = self.funktion("inkluagentArbeitsZeigen")
        self.assertNotIn("innerHTML", zeigen)
        self.assertIn(r"/^\/api\/images\/\d+\/(file|page-view)$/", zeigen)
        self.assertIn("img alt=\"\"", self.funktion("inkluagentArbeitsfensterBauen"), "Bildunterschrift traegt den Inhalt")

    def test_seitenleiste_gespraeche(self):
        dash = lies(FRONTEND, "dashboard.js")
        f = dash[dash.index("function seitenleisteGespraecheAn("):]
        self.assertIn("currentUser.oberflaeche === 'agent'", f[:f.index("\n}\n")])
        self.assertIn("fetch('/api/agent/gespraeche?anzahl=8')", dash)
        self.assertIn("a.setAttribute('aria-current', 'true')", dash)
        self.assertIn("'/api/agent/gespraeche/' + id + '/fortsetzen'", dash)
        self.assertIn("const liGespraeche = seitenleisteGespraecheBauen(path);", dash)
        self.assertLess(dash.index("ul.appendChild(liGespraeche)"), dash.index("ul.appendChild(liKonto)"), "vor „Konto“")

    def test_angeheftete_eingabe_verdeckt_nichts(self):
        css = lies(FRONTEND, "style.css")
        self.assertIn("html.agentenansicht-chatstil { scroll-padding-bottom: 13rem; }", css)
        self.assertIn("@media (max-height: 560px)", css)
        self.assertIn("html.mit-seitenleiste-gespraeche .inkluagent-chatstil .inkluagent-gespraech-leiste { display: none; }", css)



class Anhaengen(unittest.TestCase):
    """Knopf „Dateien anhängen“ in der Agentenansicht (Steve 12.10.2026): derselbe Weg wie das Feld, nur im Chatstil."""

    def setUp(self):
        self.u = nutzer()
        from inkluagent import ansicht, hochladen
        self.ansicht, self.hochladen = ansicht, hochladen
        for name in ("AGENT_HOCHLADEN", "AGENT_ARBEITSFENSTER", "AGENT_ANSICHT", "AGENT_UEBERALL", "AGENT_KONTO"):
            mock.patch.object(funktionen, name, True).start()

    def tearDown(self):
        mock.patch.stopall()

    def test_prompt_je_nach_weg(self):
        hier = self.hochladen.prompt_hochladen_hier
        self.assertIn("kein Feld zum Hochladen", hier(self.u["id"]), "manuelle Ansicht: kein Weg im Agenten")
        self.assertEqual(hier(self.u["id"], im_projekt=True), "", "im Projekt gibt es den Knopf des Projekts")
        self.ansicht.setzen(self.u["id"], "agent")
        self.assertIn("„Dateien anhängen“", hier(self.u["id"]))
        self.assertIn("„Dateien anhängen“", hier(self.u["id"], im_projekt=True))
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", False):
            self.assertIn("kein Feld zum Hochladen", hier(self.u["id"]))
        with mock.patch.object(funktionen, "AGENT_HOCHLADEN", False):
            self.assertNotIn("Dateien anhängen", hier(self.u["id"]), "ohne Server-Teil kein Knopf")
        with mock.patch.object(funktionen, "AGENT_HOCHLADEFELD", True):
            self.assertIn("„Dateien mitschicken“", hier(self.u["id"]), "das Feld geht vor")
        from inkluagent import konto
        self.assertIn("„Dateien anhängen“", konto.werkzeugsatz(self.u["id"])[2])

    def test_schalter_fuer_die_seite(self):
        try:
            import main
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        with mock.patch.object(main, "INKLUAGENT_ENABLED", True):
            self.assertTrue(main._inkluagent_schalter()["anhaengen"])
            self.assertFalse(main._inkluagent_schalter()["hochladen"], "das Feld bleibt aus (Michael, Punkt 3)")
            with mock.patch.object(funktionen, "AGENT_HOCHLADEN", False):
                self.assertFalse(main._inkluagent_schalter()["anhaengen"])
            with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", False):
                self.assertFalse(main._inkluagent_schalter()["anhaengen"])

    def test_projekt_prompt_nimmt_den_absatz(self):
        quelle = lies(BACKEND, "inkluagent", "agent_loop.py")
        self.assertIn("hochladen.prompt_hochladen_hier(user_id, im_projekt=True)", quelle)

    def test_oberflaeche(self):
        js = lies(FRONTEND, "inkluagent.js")

        def funktion(name):
            f = js[js.index("function " + name + "("):]
            return f[:f.index("\n}\n")]
        an = funktion("inkluagentAnhaengenAn")
        self.assertIn("window.INKLUAGENT.anhaengen", an)
        self.assertIn("inkluagentArbeitsfensterAn()", an)
        self.assertIn("!inkluagentHochladenAn()", an, "nur ohne das Feld")
        bau = funktion("inkluagentAnhangBauen")
        for teil in ("knopf.type = 'button'", "t('Dateien anhängen')", "datei.type = 'file'", "datei.multiple = true",
                     "liste.setAttribute('aria-label', t('Angehängte Dateien'))", "'dragover'", "'drop'", "'paste'",
                     "send.parentNode.insertBefore(knopf, send)"):
            self.assertIn(teil, bau)
        zeigen = funktion("inkluagentAnhaengeZeigen")
        self.assertNotIn("innerHTML", zeigen)
        self.assertIn("weg.setAttribute('aria-label', t('„{name}“ entfernen'", zeigen)
        self.assertIn("ziel.focus()", funktion("inkluagentAnhangEntfernen"))
        send = funktion("inkluagentSend")
        self.assertIn("inkluagentAnhangAktiv() ? (window._inkluagentAnhaenge || []).slice()", send)
        self.assertIn("if (inkluagentAnhangAktiv()) inkluagentAnhaengeLeeren();", send)
        stil = funktion("inkluagentChatstil")
        self.assertIn("inkluagentAnhangBauen(form, gespraech);", stil)
        self.assertIn("inkluagentAnhangEntfernenAlles();", stil)
        self.assertNotIn("inkluagentSetStatus", funktion("inkluagentAnhaengeHinzu"), "Meldungen nie ueber die Statuszeile des Auftrags")


if __name__ == "__main__":
    unittest.main()
