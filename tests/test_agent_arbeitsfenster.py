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
        # Kennungen von Bild und Dokument seit dem Live-Fenster (16.10.2026): damit es die Karte findet
        self.assertEqual(z, {"art": "bild", "image_id": self.a, "nummer": 2, "seite": 2, "bild_url": f"/api/images/{self.a}/file",
                             "seite_url": f"/api/images/{self.a}/page-view", "dokument": "Jahresbericht (fiktiv).pdf",
                             "dokument_nr": 1, "document_id": self.d1, "project_id": self.p})   # Projekt seit 17.10.2026
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
                         {"art": "feld", "feld_id": fid, "nummer": 4, "seite": 3, "dokument": "Anhang (fiktiv).pdf", "dokument_nr": 2,
                          "document_id": self.d2, "project_id": self.p})
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {"document_id": self.d1}),
                         {"art": "dokument", "dokument": "Jahresbericht (fiktiv).pdf", "dokument_nr": 1, "document_id": self.d1,
                          "project_id": self.p})

    def test_ansicht_je_werkzeug(self):
        # Live-Fenster „zum Anklemmen“ (17.10.2026): EINE Liste Werkzeug -> Ansicht; ohne Eintrag keine Ansicht (die Seite nimmt die Zielart)
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {"image_id": self.a}, "generate_alt_text")["ansicht"], "alttexte")
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {"document_id": self.d1}, "pruefung_starten")["ansicht"], "abschluss")
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {"document_id": self.d1}, "barrierefrei_machen")["ansicht"], "tagging")
        self.assertNotIn("ansicht", arbeitsfenster.ziel(self.u["id"], self.p, {"document_id": self.d1}, "gibt_es_nicht"))
        # ein Werkzeug fuer das ganze Projekt (17.10.2026): die Ansicht selbst ist das Ziel — nur im eigenen Projekt, nur ohne Kennung
        self.assertEqual(arbeitsfenster.ziel(self.u["id"], self.p, {}, "konvertiere_zu_pdfua"),
                         {"art": "ansicht", "project_id": self.p, "ansicht": "dokument"})
        self.assertIsNone(arbeitsfenster.ziel(self.u["id"], self.p, {}, "gibt_es_nicht"))
        self.assertIsNone(arbeitsfenster.ziel(self.fremd["id"], self.p, {}, "konvertiere_zu_pdfua"), "fremdes Projekt")
        self.assertIsNone(arbeitsfenster.ziel(self.u["id"], self.p, {"image_id": 999999}, "generate_alt_text"), "falsche Kennung")
        self.assertIsNone(arbeitsfenster.ziel(self.u["id"], None, {}, "konvertiere_zu_pdfua"))
        gueltig = {"dokument", "tagging", "alttexte", "quickinfos", "uebersetzung", "abschluss"}
        self.assertLessEqual(set(arbeitsfenster.ANSICHT_JE_WERKZEUG.values()), gueltig)
        self.assertLessEqual(set(arbeitsfenster.ANSICHT_JE_LAUF.values()), gueltig)
        import laufbuch
        self.assertEqual(set(arbeitsfenster.ANSICHT_JE_LAUF), set(laufbuch.ARTEN), "jede Art des Laufbuchs hat ihre Ansicht")

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

    def test_livestream_zeilen_und_der_lauf_endet(self):
        """Regression 12.10.2026: eine Hilfsfunktion „_fertig“ ueberdeckte die gleichnamige des Endpunkts — der Stream endete nie."""
        def antwort(*a, on_tool=None, **kw):
            on_tool.text("Hallo ")
            on_tool.text(None)
            on_tool.text("Neu ")
            on_tool.zwischen()
            on_tool("revert_alt_text", {"image_id": self.b})
            on_tool.fertig("revert_alt_text", {"image_id": self.b}, {"ok": True}, [{"type": "refresh_image", "alt_text": "Neu (fiktiv)"}])
            return {"reply": "Fertig (fiktiv).", "intent": "agentic", "werkzeuge": ["revert_alt_text"]}
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True), \
                mock.patch.object(self.chat_engine, "process_message", side_effect=antwort):
            r = self.c.post(f"/api/projects/{self.p}/chat", json={"message": "Setz Bild 1 zurück."},
                            headers={"Accept": "application/x-ndjson"})
        zeilen = [json.loads(z) for z in r.text.splitlines() if z.strip()]
        self.assertEqual([z["type"] for z in zeilen], ["text", "text_neu", "text", "zwischen", "tool", "werkzeug_fertig", "reply"])
        self.assertEqual(zeilen[5], {"type": "werkzeug_fertig", "name": "revert_alt_text", "ok": True, "text": "Neu (fiktiv)"})
        self.assertFalse(self.main._agent_laeuft(self.u["id"]) if hasattr(self.main, "_agent_laeuft")
                         else self.main._AGENT_LAEUFT.get(self.u["id"]), "der Lauf ist ausgetragen")
        self.assertNotIn("def _fertig(name", lies(BACKEND, "main.py"))

    def test_fortschritt_fuer_eine_neue_seite(self):
        """Steve 12.10.2026: nach einem Seiten- oder Ansichtswechsel zeigt die neue Seite die Schritte weiter (GET /api/agent/status)."""
        m = self.main
        start = 12345.0
        m._AGENT_LAEUFT[self.u["id"]] = __import__("time").time()
        try:
            m._fortschritt_neu(self.u["id"], start, self.p)
            m._fortschritt_aendern(self.u["id"], start, lambda f: m._fortschritt_schritt(f, "view_image", {"art": "bild", "nummer": 1}))
            m._fortschritt_aendern(self.u["id"], start, lambda f: m._fortschritt_fertig(f, False))
            m._fortschritt_aendern(self.u["id"], start, lambda f: m._fortschritt_schritt(f, "list_project_images", None))
            m._fortschritt_aendern(self.u["id"], 1.0, lambda f: m._fortschritt_schritt(f, "fremder_lauf", None))   # alter Lauf: nichts
            with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
                st = self.c.get("/api/agent/status").json()
            self.assertEqual(st["projekt"], self.p)
            self.assertEqual([(x["name"], x["fertig"], x["ok"]) for x in st["schritte"]],
                             [("view_image", True, False), ("list_project_images", False, None)])
            self.assertFalse(st["schreibt"])
            self.assertEqual(self.c.get("/api/agent/status").json(), {"laeuft": True}, "ohne Schalter wie bisher")
            m._fortschritt_ende(self.u["id"], start)
            self.assertNotIn(self.u["id"], m._AGENT_FORTSCHRITT)
        finally:
            m._AGENT_LAEUFT.pop(self.u["id"], None)
            m._AGENT_FORTSCHRITT.pop(self.u["id"], None)

    def test_stream_fuellt_den_fortschritt_und_raeumt_ihn_ab(self):
        gesehen = {}

        def antwort(*a, on_tool=None, **kw):
            on_tool("view_image", {"image_id": self.b})
            gesehen["waehrend"] = self.c.get("/api/agent/status").json()
            return {"reply": "Fertig (fiktiv).", "intent": "agentic", "werkzeuge": ["view_image"]}
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True), \
                mock.patch.object(self.chat_engine, "process_message", side_effect=antwort):
            self.c.post(f"/api/projects/{self.p}/chat", json={"message": "Schau dir Bild 1 an."}, headers={"Accept": "application/x-ndjson"})
        w = gesehen["waehrend"]
        self.assertTrue(w["laeuft"])
        self.assertEqual(w["schritte"][0]["name"], "view_image")
        self.assertEqual(w["schritte"][0]["ziel"]["nummer"], 1)
        self.assertNotIn(self.u["id"], self.main._AGENT_FORTSCHRITT, "nach dem Lauf weg")

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
        from inkluagent import skripte   # alle Teile des Agenten-Skripts (17.10.2026)
        self.js = skripte.quelltext(FRONTEND)

    def funktion(self, name):
        f = self.js[self.js.index("function " + name + "("):]
        return f[:f.index("\n}\n")]

    def test_nur_in_der_agentenansicht(self):
        # Pruefung 13.10.2026 (HK2): der Chatstil hat keinen Rueckweg mehr — der Zweig war nie erreichbar (die manuelle Ansicht kommt
        # ueber den Umschalter mit Neuladen); inkluagentChatstil() schaltet nur ein
        self.assertIn("if (inkluagentArbeitsfensterAn()) { inkluagentChatstil();", self.funktion("inkluagentAgentenansicht"))
        self.assertNotIn("inkluagentChatstil(", self.funktion("inkluagentAgentenansichtAufheben"))
        self.assertNotIn("function inkluagentAnhangEntfernenAlles(", self.js)
        an = self.funktion("inkluagentArbeitsfensterAn")
        self.assertIn("window.INKLUAGENT.arbeitsfenster", an)
        self.assertIn("inkluagentUeberall()", an)

    def test_aufbau_verlauf_eingabe_beschreibung_dann_arbeitsfenster(self):
        f = self.funktion("inkluagentChatstil")
        # Pruefung 13.10.2026 (B6): Meldungen (Ergebnis des Hochladens, Statuszeile) im Fluss zwischen Verlauf und angehefteter Eingabe
        self.assertIn("[frueher, log, meldungen, form, trenner, beschreibung].forEach", f, "Chatstil: Verlauf vor der Eingabe")
        self.assertIn("['inkluagentHochgeladen', 'inkluagentStatus'].forEach", f)
        self.assertIn("log.removeAttribute('tabindex')", f, "B22: kein leerer Tab-Stopp im Chatstil")
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
        # Steve 12.10.2026 (spaeter): Schritte nur mit dem Kaestchen „Jeden Schritt ansagen“ ansagen — seit der Pruefung 13.10.2026
        # (B9) gebuendelt, hoechstens einmal je Sekunde
        self.assertIn("inkluagentSchrittAnsagen(text);", self.funktion("inkluagentArbeitsSchritt"))
        buendel = self.funktion("inkluagentSchrittAnsagen")
        self.assertIn("if (!text || !inkluagentSchritteAnsagen()) return;", buendel)
        self.assertIn("b.zuletzt + 1000 - Date.now()", buendel)
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
        # Pruefung 13.10.2026 (B6): der Platz folgt der gemessenen Hoehe (ResizeObserver in inkluagent.js); Verhalten: lokal/probe_pr13_*
        self.assertIn("html.agentenansicht-chatstil { scroll-padding-bottom: calc(var(--agent-eingabe-hoehe, 13rem) + 1rem); }", css)
        self.assertIn("--agent-eingabe-hoehe", self.funktion("inkluagentEingabeHoeheBeobachten"))
        self.assertIn("@media (max-height: 560px)", css)
        self.assertIn("html.mit-seitenleiste-gespraeche .inkluagent-chatstil .inkluagent-gespraech-knoepfe { display: none; }", css)



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
        """Im Projekt steht der Absatz im Prompt, sobald es im Agenten einen Weg gibt (seit 15.10.2026 ueber das Schubfach
        „hochladen_hier“, prompts/schubfaecher.py — vorher stand die Zeile in agent_loop und wurde hier als Text gesucht)."""
        from inkluagent import agent_loop
        pdf = {"project_type": "pdf", "tool": "pdf"}
        self.assertNotIn("Dateien anhängen", agent_loop._werkzeugsatz(pdf, 1, self.u["id"])[2], "manuelle Ansicht: kein Absatz")
        self.ansicht.setzen(self.u["id"], "agent")
        self.assertIn("„Dateien anhängen“", agent_loop._werkzeugsatz(pdf, 1, self.u["id"])[2])

    def test_oberflaeche(self):
        from inkluagent import skripte   # alle Teile des Agenten-Skripts (17.10.2026)
        js = skripte.quelltext(FRONTEND)

        def funktion(name):
            f = js[js.index("function " + name + "("):]
            return f[:f.index("\n}\n")]
        an = funktion("inkluagentAnhaengenAn")
        self.assertIn("window.INKLUAGENT.anhaengen", an)
        self.assertIn("inkluagentArbeitsfensterAn()", an)
        self.assertIn("!inkluagentHochladenAn()", an, "nur ohne das Feld")
        bau = funktion("inkluagentAnhangBauen")
        for teil in ("knopf.type = 'button'", "t('Dateien anhängen')", "datei.type = 'file'", "datei.multiple = true",
                     "liste.setAttribute('aria-label', t('Angehängte Dateien'))", "'paste'", "inkluagentAblageEinrichten();",
                     "send.parentNode.insertBefore(knopf, send)"):
            self.assertIn(teil, bau)
        # Hineinziehen ueberall auf der Seite (Pruefung 13.10.2026, HK7) und sichtbare Grenzen (B21)
        ablage = funktion("inkluagentAblageEinrichten")
        for teil in ("'dragover'", "'drop'", "if (e.defaultPrevented || !inkluagentAnhangAktiv() || !mitDateien(e)) return;"):
            self.assertIn(teil, ablage)
        self.assertNotIn("'visually-hidden'", bau)
        zeigen = funktion("inkluagentAnhaengeZeigen")
        self.assertNotIn("innerHTML", zeigen)
        self.assertIn("weg.setAttribute('aria-label', t('„{name}“ entfernen'", zeigen)
        self.assertIn("ziel.focus()", funktion("inkluagentAnhangEntfernen"))
        send = funktion("inkluagentSend")
        self.assertIn("inkluagentAnhangAktiv() ? (window._inkluagentAnhaenge || []).slice()", send)
        self.assertIn("if (inkluagentAnhangAktiv()) inkluagentAnhaengeLeeren(dateien);", send, "B12: nur die mitgeschickten")
        stil = funktion("inkluagentChatstil")
        self.assertIn("inkluagentAnhangBauen(form, meldungen);", stil)
        self.assertNotIn("inkluagentSetStatus", funktion("inkluagentAnhaengeHinzu"), "Meldungen nie ueber die Statuszeile des Auftrags")



class AnsichtTexte(unittest.TestCase):
    """Steve 12.10.2026: „Die übrigen Seiten bleiben vorerst, wie sie sind“ war veraltet — die Beschreibungen folgen den Schaltern."""

    def test_beschreibung_je_nach_schalter(self):
        from inkluagent import ansicht
        self.assertIn("in jedem Projekt", ansicht.beschreibung("klassisch"), "ohne Schalter wie bisher")
        self.assertIn("vorerst", ansicht.beschreibung("agent"))
        with mock.patch.object(funktionen, "AGENT_UEBERALL", True), mock.patch.object(funktionen, "AGENT_KONTO", True):
            self.assertIn("unten auf jeder Seite", ansicht.beschreibung("klassisch"))
            self.assertIn("Auf jeder Seite siehst du nur die Überschrift", ansicht.beschreibung("agent"))
            with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
                self.assertIn("Arbeitsfenster", ansicht.beschreibung("agent"))
                self.assertEqual([a["beschreibung"] for a in ansicht.auswahl()][1], ansicht.BESCHREIBUNG_ARBEITSFENSTER)

    def test_texte_in_allen_sprachen(self):
        from inkluagent import ansicht
        texte = list(ansicht.BESCHREIBUNG_UEBERALL.values()) + [ansicht.BESCHREIBUNG_ARBEITSFENSTER]
        for sprache in ("de", "en", "fr", "es", "da", "sv"):
            po = lies(BACKEND, "locales", sprache, "LC_MESSAGES", "messages.po")
            for text in texte:
                self.assertIn('msgid "' + text + '"', po, (sprache, text[:40]))
        self.assertIn("inkluagent_schalter.ueberall", lies(BACKEND, "templates", "hilfe_inkluagent.html"))



class Livestream(unittest.TestCase):
    """Livestream im Arbeitsfenster (Steve 12.10.2026): Ergebnis je Werkzeug, Antwort Wort fuer Wort, Zwischenstand."""

    def test_ergebnis_je_werkzeug(self):
        e = arbeitsfenster.ergebnis
        self.assertEqual(e("view_image", {"ok": False, "error": "Bild nicht gefunden." + "x" * 300}, [])["fehler"][:20], "Bild nicht gefunden.")
        self.assertEqual(len(e("view_image", {"ok": False, "error": "y" * 900}, [])["fehler"]), 200)
        self.assertEqual(e("revert_alt_text", {"ok": True}, [{"type": "refresh_image", "alt_text": "Neu (fiktiv)"}]),
                         {"ok": True, "text": "Neu (fiktiv)"})
        self.assertEqual(e("generate_quickinfo", {"ok": True}, [{"type": "refresh_feld", "quickinfo": "Q (fiktiv)"}])["text"], "Q (fiktiv)")
        self.assertEqual(e("generate_alt_text", {"ok": True, "result": {"alt_text": "G (fiktiv)"}}, [])["text"], "G (fiktiv)")
        self.assertEqual(e("list_project_images", {"ok": True, "result": {"x": 1}}, []), {"ok": True})
        self.assertEqual(e("x", "kaputt", []), {"ok": False, "fehler": ""})

    def test_schleife_streamt_meldet_zwischenstand_und_ergebnis(self):
        from inkluagent import agent_loop
        u = nutzer()
        ereignisse = []

        class Modell:
            kann_streamen = True

            def __init__(self):
                self.n = 0

            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3, on_text=None):
                self.n += 1
                if self.n == 1:
                    on_text("Ich sehe nach. ")
                    return {"content": [{"type": "text", "text": "Ich sehe nach. "},
                                        {"type": "tool_use", "id": "t1", "name": "projekte_auflisten", "input": {}}], "stop_reason": "tool_use"}
                on_text("Fertig ")
                on_text("(fiktiv).")
                return {"content": [{"type": "text", "text": "Fertig (fiktiv)."}], "stop_reason": "end_turn"}

        def melder(name, args):
            ereignisse.append(("tool", name))
        melder.text = lambda t: ereignisse.append(("text", t))
        melder.zwischen = lambda: ereignisse.append(("zwischen",))
        melder.fertig = lambda name, args, result, neu: ereignisse.append(("fertig", name, bool(result.get("ok"))))
        with mock.patch.object(funktionen, "AGENT_KONTO", True), mock.patch.object(funktionen, "AGENT_UEBERALL", True):
            r = agent_loop.run_agent(None, u["id"], "Welche Projekte habe ich?", None, Modell(), on_tool=melder)
        self.assertEqual(ereignisse, [("text", "Ich sehe nach. "), ("zwischen",), ("tool", "projekte_auflisten"),
                                      ("fertig", "projekte_auflisten", True), ("text", "Fertig "), ("text", "(fiktiv).")])
        self.assertEqual(r["reply"], "Fertig (fiktiv).")

    def test_ohne_stream_faehigkeit_kein_on_text(self):
        from inkluagent import agent_loop
        u = nutzer()

        class Alt:   # wie BedrockProvider: kennt on_text nicht
            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3):
                return {"content": [{"type": "text", "text": "Alt (fiktiv)."}], "stop_reason": "end_turn"}

        def melder(name, args):
            pass
        melder.text = lambda t: self.fail("kein Stream ohne kann_streamen")
        with mock.patch.object(funktionen, "AGENT_KONTO", True), mock.patch.object(funktionen, "AGENT_UEBERALL", True):
            self.assertEqual(agent_loop.run_agent(None, u["id"], "Hallo", None, Alt(), on_tool=melder)["reply"], "Alt (fiktiv).")


class GeminiStrom(unittest.TestCase):
    """GeminiProvider ueber streamGenerateContent (SSE): Text an on_text, Ergebnis wie generateContent, Kosten wie bisher."""

    def setUp(self):
        try:
            from inkluagent.providers import gemini
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"Gemini-Provider nicht ladbar: {e}")
        self.gemini = gemini
        self.p = object.__new__(gemini.GeminiProvider)
        self.p._zaehler = 0
        self.urls = []
        self.kosten = []
        mock.patch.object(gemini, "_endpunkt", return_value="https://modell.invalid/v1/models/m:generateContent").start()
        mock.patch("pipelines.v4.gemini_auth.kopfzeilen", return_value={}).start()
        mock.patch.object(gemini.ki_kosten, "erfasse_gemini", side_effect=lambda m, a, **k: self.kosten.append(a)).start()

    def tearDown(self):
        mock.patch.stopall()

    def _sse(self, *stuecke):
        zeilen = [("data: " + json.dumps(x) + "\n").encode() for x in stuecke]
        test = self

        class Antwort:
            def __enter__(self):
                return iter(zeilen)

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout=None):
            test.urls.append(req.full_url)
            return Antwort()
        return mock.patch("urllib.request.urlopen", side_effect=urlopen)

    def test_text_stueckweise(self):
        stuecke = []
        with self._sse({"candidates": [{"content": {"parts": [{"text": "Hallo "}]}}]},
                       {"candidates": [{"content": {"parts": [{"text": "Welt, hier schreibt der Test (fiktiv).", "thoughtSignature": "sig"}]}, "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}}):
            r = self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=stuecke.append)
        self.assertEqual(stuecke, ["Hallo ", "Welt, hier schreibt der Test (fiktiv)."])
        self.assertEqual(r["content"], [{"type": "text", "text": "Hallo Welt, hier schreibt der Test (fiktiv)."}])
        self.assertEqual(r["stop_reason"], "end_turn")
        self.assertTrue(self.urls[0].endswith(":streamGenerateContent?alt=sse"), self.urls)
        self.assertEqual(self.kosten[0]["usageMetadata"]["candidatesTokenCount"], 3, "Kosten wie bisher erfasst")

    def test_funktionsaufruf_mit_signatur(self):
        teil = {"functionCall": {"name": "projekte_auflisten", "args": {}}, "thoughtSignature": "sig2"}
        with self._sse({"candidates": [{"content": {"parts": [teil]}, "finishReason": "STOP"}]}):
            r = self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=lambda t: self.fail("kein Text"))
        self.assertEqual(r["stop_reason"], "tool_use")
        self.assertEqual(r["content"][0]["_gemini_part"], teil, "Gedankensignatur bleibt fuer den Folgeaufruf")

    def test_stream_scheitert_ohne_text_weg_ohne_stream(self):
        stuecke = []
        normal = {"candidates": [{"content": {"parts": [{"text": "Ohne Stream (fiktiv)."}]}, "finishReason": "STOP"}]}
        with mock.patch("urllib.request.urlopen", side_effect=OSError("weg")), \
                mock.patch.object(self.gemini.GeminiProvider, "_aufruf", return_value=normal):
            r = self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=stuecke.append)
        self.assertEqual(stuecke, [None], "angefangener Entwurf wird verworfen")
        self.assertEqual(r["content"][0]["text"], "Ohne Stream (fiktiv).")

    # Sonderfaelle (Pruefung 13.10.2026, K10/S12)
    def test_gesperrt_ohne_kandidat_ist_ein_fehler_ohne_zweiten_aufruf(self):
        stuecke = []
        with self._sse({"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 12}}), \
                mock.patch.object(self.gemini.GeminiProvider, "_aufruf", side_effect=AssertionError("kein zweiter Aufruf")):
            with self.assertRaises(self.gemini.GeminiAntwortGesperrt):
                self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=stuecke.append)
        self.assertEqual(stuecke, [None])
        self.assertEqual(self.kosten[0]["usageMetadata"]["promptTokenCount"], 12, "auch die gesperrte Anfrage kostet")

    def test_fehlerobjekt_im_stream_nimmt_den_weg_ohne_stream(self):
        normal = {"candidates": [{"content": {"parts": [{"text": "Ohne Stream (fiktiv)."}]}, "finishReason": "STOP"}]}
        with self._sse({"error": {"code": 500, "message": "intern (fiktiv)"}}), \
                mock.patch.object(self.gemini.GeminiProvider, "_aufruf", return_value=normal) as ohne:
            r = self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=lambda t: None)
        ohne.assert_called_once()
        self.assertEqual(r["content"][0]["text"], "Ohne Stream (fiktiv).")

    def test_abbruch_nach_text_erfasst_kosten_und_holt_die_ganze_antwort(self):
        stuecke = []
        erstes = ("data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hallo "}]}}],
                                          "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 2}}) + "\n").encode()

        def zeilen():
            yield erstes
            raise OSError("Verbindung weg (fiktiv)")

        class Antwort:
            def __enter__(self):
                return zeilen()

            def __exit__(self, *a):
                return False
        normal = {"candidates": [{"content": {"parts": [{"text": "Ganze Antwort (fiktiv)."}]}, "finishReason": "STOP"}]}
        with mock.patch("urllib.request.urlopen", return_value=Antwort()), \
                mock.patch.object(self.gemini.GeminiProvider, "_aufruf", return_value=normal):
            r = self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=stuecke.append)
        self.assertEqual(stuecke, ["Hallo ", None], "Entwurf gezeigt, dann verworfen")
        self.assertEqual(r["content"][0]["text"], "Ganze Antwort (fiktiv).")
        self.assertEqual(self.kosten[0]["usageMetadata"]["promptTokenCount"], 30, "der abgebrochene Teil zaehlt fuer den Deckel")

    # Stille im Stream (Steve 15.10.2026: Gemini hing drei Minuten ohne ein Byte, die Nachricht lief in die Gesamtfrist)
    def test_stille_begrenzt_den_lesevorgang(self):
        gesehen = []

        class Antwort:
            def __enter__(self):
                return iter([("data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hallo, hier schreibt der Test (fiktiv)."}]},
                                                                      "finishReason": "STOP"}]}) + "\n").encode()])

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout=None):
            gesehen.append(timeout)
            return Antwort()
        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=lambda t: None,
                                     frist=__import__("time").monotonic() + 300)
        self.assertEqual(gesehen, [self.gemini._STILLE_S], "je Lesevorgang hoechstens die Stille, nicht die ganze Frist")
        self.assertLessEqual(self.gemini._STILLE_S, 60)

    def test_stille_nimmt_sofort_den_weg_ohne_stream(self):
        stuecke = []

        def zeilen():
            raise TimeoutError("The read operation timed out")
            yield b""   # noqa: unreachable — macht die Funktion zum Generator

        class Antwort:
            def __enter__(self):
                return zeilen()

            def __exit__(self, *a):
                return False
        normal = {"candidates": [{"content": {"parts": [{"text": "Ohne Stream (fiktiv)."}]}, "finishReason": "STOP"}]}
        with mock.patch("urllib.request.urlopen", return_value=Antwort()), \
                mock.patch.object(self.gemini.GeminiProvider, "_aufruf", return_value=normal) as ohne, \
                self.assertLogs(self.gemini.log, level="WARNING") as protokoll:
            r = self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=stuecke.append)
        ohne.assert_called_once()
        self.assertEqual(r["content"][0]["text"], "Ohne Stream (fiktiv).")
        self.assertEqual(stuecke, [None])
        self.assertIn("ohne Daten", protokoll.output[0])

    def test_stille_erkennen(self):
        import urllib.error
        self.assertTrue(self.gemini._still(TimeoutError("The read operation timed out")))
        self.assertTrue(self.gemini._still(urllib.error.URLError(TimeoutError("timed out"))))
        self.assertFalse(self.gemini._still(OSError("Verbindung weg (fiktiv)")))

    def test_text_und_zwei_funktionsaufrufe(self):
        a = {"functionCall": {"name": "projekte_auflisten", "args": {}}, "thoughtSignature": "s1"}
        b = {"functionCall": {"name": "guthaben_lesen", "args": {}}}
        stuecke = []
        with self._sse({"candidates": [{"content": {"parts": [{"text": "Ich schaue nach (fiktiv)."}]}}]},
                       {"candidates": [{"content": {"parts": [a, b]}, "finishReason": "STOP"}]}):
            r = self.p.invoke_with_tools([{"role": "user", "content": "Hi"}], [], on_text=stuecke.append)
        self.assertEqual(stuecke, ["Ich schaue nach (fiktiv)."])
        self.assertEqual([x["type"] for x in r["content"]], ["text", "tool_use", "tool_use"])
        self.assertEqual([x.get("name") for x in r["content"][1:]], ["projekte_auflisten", "guthaben_lesen"])
        self.assertEqual(r["stop_reason"], "tool_use")



class AgentenansichtKorrekturen(unittest.TestCase):
    """Steve 12.10.2026 auf Staging: spaeter eingeblendete Teile rutschten durch, der Agent liess sich auf- und zuklappen,
    „Gespräche“ stand als Ueberschrift 2 vor der Ueberschrift 1, die neue Seite zeigte nach einem Wechsel keine Schritte."""

    def setUp(self):
        from inkluagent import skripte   # alle Teile des Agenten-Skripts (17.10.2026)
        self.js = skripte.quelltext(FRONTEND)

    def funktion(self, name):
        f = self.js[self.js.index("function " + name + "("):]
        return f[:f.index("\n}\n")]

    def test_spaeter_eingeblendetes_bleibt_versteckt(self):
        f = self.funktion("inkluagentAgentenansicht")
        aus = self.funktion("inkluagentAgentenansichtAusblenden")
        self.assertIn("if (k.hidden) { k.setAttribute('data-agentenansicht', 'war-versteckt'); return; }", aus)
        self.assertIn("window._agentenansichtBeobachter.observe(main, { childList: true, subtree: true });", f)
        # Pruefung 13.10.2026 (K19): der Beobachter blendet nur aus, er baut das Arbeitsfenster nicht neu
        beobachter = f[f.index("new MutationObserver("):f.index(".observe(main")]
        self.assertIn("inkluagentAgentenansichtAusblenden()", beobachter)
        self.assertNotIn("inkluagentArbeitsZeigen", beobachter)
        self.assertIn("[data-agentenansicht] {\n    display: none !important;", lies(FRONTEND, "style.css"))
        auf = self.funktion("inkluagentAgentenansichtAufheben")
        self.assertIn("if (el.getAttribute('data-agentenansicht') === 'aus') el.hidden = false;", auf)

    def test_kein_auf_und_zu_in_der_agentenansicht(self):
        f = self.funktion("inkluagentChatstil")
        self.assertIn("toggle.hidden = true;", f)
        self.assertIn("panel.hidden = false;", f)
        self.assertIn(".inkluagent-toggle[hidden] { display: none; }", lies(FRONTEND, "style.css"))

    def test_gespraeche_ohne_ueberschrift_als_benannter_bereich(self):
        dash = lies(FRONTEND, "dashboard.js")
        f = dash[dash.index("function seitenleisteGespraecheBauen("):]
        f = f[:f.index("\n}\n")]
        self.assertNotIn("createElement('h2')", f)
        self.assertIn("bereich.setAttribute('aria-labelledby', 'navGespraecheTitel')", f)

    def test_neue_seite_zeigt_die_schritte_weiter(self):
        f = self.funktion("inkluagentLaufVerfolgen")
        self.assertIn("inkluagentFortschrittZeigen(st, gesehen, false)", f)
        self.assertIn("if (projekt && hier === projekt) await inkluagentAnsichtNachziehen(window._inkluagentProjectId);", f)
        z = self.funktion("inkluagentFortschrittZeigen")
        self.assertIn("neu.forEach(inkluagentSchrittAnsagen)", z, "neue Schritte gebuendelt ansagen (B9)")
        self.assertIn("t('Ruft gerade auf: {w}'", z)


if __name__ == "__main__":
    unittest.main()
