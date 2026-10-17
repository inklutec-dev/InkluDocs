"""Live-Fenster im Arbeitsfenster (Steve 16.10.2026: „wie einen Livestream … wenn er einen Alternativtext für die Bilder macht, dass
man in dem Moment das Bild eingeblendet sieht“; 17.10.2026 „zum Anklemmen“: jedes Werkzeug, jede Kette, „natürlich soll nur das
gestreamt werden, was der Agent auch an Aufgaben macht“; Schalter funktionen.AGENT_ARBEITSFENSTER).

Server: GET /api/agent/live (inkluagent/arbeitsfenster.live) liest das Laufbuch (laufbuch.py, tests/test_laufbuch.py): nur Laeufe des
Agenten, nur Projekte DIESES Kontos; je Lauf Art, Station, Ansicht, Fortschritt, Stelle (aus der Datenbank), das Stueck davor mit
seinem Text (wie die Oberflaeche ihn zeigt) und das Ergebnis in der Ablage (nur eigene). Die Laeufe melden sich an und ab — auch bei
Fehler und Abbruch (Alt-Texte, die Maentel um Herunterladen, Umwandeln, Word pruefen, Pruefdatei); jeder Ort, der einen Lauf anmeldet,
nimmt eine bekannte Art; Threads und fremde Schleifen nehmen den Agenten mit.
Oberflaeche am Quelltext: die Seite draussen spricht mit der Seite im Fenster nur ueber window.InkluBuehne (Vertrag: test_buehne.py),
zieht niemanden weg, sagt nichts an, setzt alles mit textContent, Bewegung abschaltbar.
Wegwerf-Datenbank, keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_live.py
"""
import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_live_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_ARBEITSFENSTER", "INKLUAGENT_ANSICHT", "INKLUAGENT_KONTO", "INKLUAGENT_UEBERALL"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
import laufbuch  # noqa: E402
from inkluagent import arbeitsfenster  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_N = [0]


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def funktion(text, kopf):
    t = text[text.index(kopf):]
    return t[:t.index("\n}\n")]


def nutzer() -> dict:
    _N[0] += 1
    mail = f"live{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, dokumente=("Jahresbericht (fiktiv).pdf",), name="Vereinsbrief (fiktiv)") -> tuple:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, 'x.pdf', '', ?, 'pdf', 'pdf', 'processing')", (uid, name)).lastrowid
    docs = [c.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, ?, ?, '')",
                      (pid, i + 1, n)).lastrowid for i, n in enumerate(dokumente)]
    c.commit()
    c.close()
    return pid, docs


def bild(pid, did, seite, index=0, status="pending", alt=None, alt_edited=None) -> int:
    c = database.get_db()
    iid = c.execute("INSERT INTO images (project_id, document_id, page_number, image_index, image_path, page_view_path, status, "
                    "alt_text, alt_text_edited) VALUES (?, ?, ?, ?, '/x.png', '/s.png', ?, ?, ?)",
                    (pid, did, seite, index, status, alt, alt_edited)).lastrowid
    c.commit()
    c.close()
    return iid


def status(iid, wert):
    c = database.get_db()
    c.execute("UPDATE images SET status = ? WHERE id = ?", (wert, iid))
    c.commit()
    c.close()


def anzeige(img):   # wie main._display_alt_text
    return img["alt_text_edited"] if img["alt_text_edited"] is not None else (img["alt_text"] or img["original_alt"])




def feld(pid, did, seite, index=1, quickinfo="") -> int:
    c = database.get_db()
    fid = c.execute("INSERT INTO formularfelder (project_id, document_id, feld_index, anker, page_number, quickinfo) "
                    "VALUES (?, ?, ?, ?, ?, ?)", (pid, did, index, f"feld-{index} (fiktiv)", seite, quickinfo)).lastrowid
    c.commit()
    c.close()
    return fid


def ausgabe(uid, pid=None) -> int:
    c = database.get_db()
    aid = c.execute("INSERT INTO ablage (user_id, project_id, art, dateiname) VALUES (?, ?, 'pdfua', 'Vereinsbrief (fiktiv).pdf')",
                    (uid, pid)).lastrowid
    c.commit()
    c.close()
    return aid


def agent_lauf(art, uid, pid, **kw):
    with laufbuch.agent_kontext():
        return laufbuch.beginnen(art, user_id=uid, project_id=pid, **kw)


class Live(unittest.TestCase):
    def setUp(self):
        laufbuch._leeren()
        self.u = nutzer()
        self.p, (self.d,) = projekt(self.u["id"])
        self.b1 = bild(self.p, self.d, 1, 0, "done", alt="Ein Haus mit rotem Dach (fiktiv).")
        self.b2 = bild(self.p, self.d, 2, 0, "processing")
        self.b3 = bild(self.p, self.d, 3, 0, "pending")

    def tearDown(self):
        laufbuch._leeren()

    def live(self, uid=None):
        return arbeitsfenster.live(self.u["id"] if uid is None else uid, None, anzeige)["laeufe"]

    def test_ohne_lauf_nichts(self):
        # ein haengender Status 'processing' (Projekt und Bild) zeigt nichts, solange kein Lauf im Laufbuch steht
        self.assertEqual(arbeitsfenster.live(self.u["id"], None, anzeige), {"laeufe": []})

    def test_bild_in_arbeit_mit_fortschritt_und_dem_fertigen_davor(self):
        l = agent_lauf("alt_texte", self.u["id"], self.p, gesamt=2)
        l.stueck(image_id=self.b1)
        l.weiter()
        l.stueck(image_id=self.b2)
        (x,) = self.live()
        self.assertEqual((x["art"], x["was"], x["ansicht"], x["project_id"], x["projekt"]),
                         ("alt_texte", "alt_texte", "alttexte", self.p, "Vereinsbrief (fiktiv)"))
        self.assertEqual((x["gesamt"], x["fertig"], x["laeuft"]), (2, 1, True))
        s = x["stelle"]
        self.assertEqual((s["art"], s["image_id"], s["nummer"], s["seite"], s["project_id"], s["ansicht"]),
                         ("bild", self.b2, 2, 2, self.p, "alttexte"))
        self.assertEqual((s["bild_url"], s["seite_url"]), (f"/api/images/{self.b2}/file", f"/api/images/{self.b2}/page-view"))
        self.assertEqual((s["dokument"], s["document_id"]), ("Jahresbericht (fiktiv).pdf", self.d))
        self.assertNotIn("text", s, "die Stelle in Arbeit hat noch keinen Text")
        v = x["vorher"]
        self.assertEqual((v["image_id"], v["status"], v["text"], v["nummer"]), (self.b1, "done", "Ein Haus mit rotem Dach (fiktiv).", 1))
        l.weiter()
        l.beenden()
        (x,) = self.live()
        self.assertEqual((x["laeuft"], x["ok"], x["stelle"], x["vorher"]["image_id"], x["fertig"]), (False, True, None, self.b2, 2))

    def test_eigener_text_geht_vor_und_fehler_bleibt_fehler(self):
        b = bild(self.p, self.d, 4, 0, "done", alt="KI-Text (fiktiv).", alt_edited="Eigener Text (fiktiv).")
        e = bild(self.p, self.d, 5, 0, "error")
        l = agent_lauf("alt_texte", self.u["id"], self.p, gesamt=3)
        l.stueck(image_id=b)
        l.stueck(image_id=e)
        self.assertEqual(self.live()[0]["vorher"]["text"], "Eigener Text (fiktiv).", "wie die Oberflaeche: der eigene Text geht vor")
        l.stueck(image_id=self.b3)
        self.assertEqual(self.live()[0]["vorher"]["status"], "error")

    def test_feld_mit_quickinfo(self):
        f1 = feld(self.p, self.d, 1, 1, "Bitte den Vornamen eintragen (fiktiv).")
        f2 = feld(self.p, self.d, 2, 2)
        l = agent_lauf("quickinfos", self.u["id"], self.p, gesamt=2)
        l.stueck(feld_id=f1, document_id=self.d)
        l.weiter()
        l.stueck(feld_id=f2, document_id=self.d)
        (x,) = self.live()
        self.assertEqual((x["ansicht"], x["stelle"]["art"], x["stelle"]["feld_id"], x["stelle"]["nummer"], x["stelle"]["seite"]),
                         ("quickinfos", "feld", f2, 2, 2))
        self.assertEqual((x["vorher"]["feld_id"], x["vorher"]["text"]), (f1, "Bitte den Vornamen eintragen (fiktiv)."))

    def test_seite_und_dokument_als_stelle(self):
        l = agent_lauf("tagging", self.u["id"], self.p, document_id=self.d)
        l.stueck(document_id=self.d, seite=3)
        s = self.live()[0]["stelle"]
        self.assertEqual((s["art"], s["document_id"], s["seite"], s["ansicht"]), ("seite", self.d, 3, "tagging"))
        l.stueck(document_id=self.d)
        s = self.live()[0]["stelle"]
        self.assertEqual((s["art"], s["document_id"], s["dokument"]), ("dokument", self.d, "Jahresbericht (fiktiv).pdf"))
        fremd_p, (fremd_d,) = projekt(nutzer()["id"])
        l.stueck(document_id=fremd_d)
        self.assertEqual(self.live()[0]["stelle"], {"art": "ansicht", "project_id": self.p, "ansicht": "tagging"},
                         "ein Dokument eines anderen Projekts ist keine Stelle: nur die Ansicht des eigenen Projekts")

    def test_ergebnis_nur_aus_der_eigenen_ablage(self):
        mein = ausgabe(self.u["id"], self.p)
        l = agent_lauf("umwandeln", self.u["id"], self.p)
        l.ergebnis(ausgabe_id=mein)
        l.beenden()
        self.assertEqual(self.live()[0]["ergebnis"], {"art": "ausgabe", "ausgabe_id": mein})
        fremd = ausgabe(nutzer()["id"])
        l2 = agent_lauf("export", self.u["id"], self.p)
        l2.ergebnis(ausgabe_id=fremd)
        l2.beenden()
        self.assertIsNone(next(x for x in self.live() if x["nr"] == l2.nr)["ergebnis"], "fremde Ablage")

    def test_auftrag_zeigt_die_station(self):
        l = agent_lauf("auftrag", self.u["id"], self.p, gesamt=3)
        l.station_setzen("tagging")
        x = self.live()[0]
        self.assertEqual((x["art"], x["station"], x["was"], x["ansicht"]), ("auftrag", "tagging", "tagging", "tagging"))
        self.assertEqual(x["stelle"], {"art": "ansicht", "project_id": self.p, "ansicht": "tagging"},
                         "eine Station ohne einzelne Stelle: die Ansicht selbst (das Fenster wechselt Station fuer Station)")
        l.station_setzen("alt_texte")
        self.assertEqual(self.live()[0]["ansicht"], "alttexte")
        self.assertEqual(self.live()[0]["stelle"]["ansicht"], "alttexte")
        l.beenden()
        self.assertIsNone(self.live()[0]["stelle"], "beendet: keine Stelle mehr")

    def test_nur_agent_nur_eigene_projekte_hoechstens_drei(self):
        laufbuch.beginnen("export", user_id=self.u["id"], project_id=self.p)          # Oberflaeche: nicht gestreamt
        self.assertEqual(self.live(), [])
        fremd = nutzer()
        fremd_p, _ = projekt(fremd["id"])
        agent_lauf("export", self.u["id"], fremd_p)                                      # Projekt eines anderen Kontos
        agent_lauf("export", self.u["id"], None)                                         # ohne Projekt
        self.assertEqual(self.live(), [])
        self.assertEqual(self.live(fremd["id"]), [], "fremdes Konto sieht nichts")
        nrn = [agent_lauf("export", self.u["id"], self.p).nr for _ in range(4)]
        self.assertEqual([x["nr"] for x in self.live()], list(reversed(nrn))[:3], "juengste zuerst, hoechstens drei")

    def test_nie_eine_ausnahme(self):
        agent_lauf("alt_texte", self.u["id"], self.p).stueck(image_id=self.b1)
        with mock.patch("database.get_db", side_effect=RuntimeError("weg")):
            self.assertEqual(arbeitsfenster.live(self.u["id"], None, anzeige), {"laeufe": []})
        kaputt = [{"project_id": self.p, "stueck": {"art": "bild", "image_id": "x"}, "ergebnis": "kaputt"}]
        self.assertEqual(arbeitsfenster.live(self.u["id"], kaputt, anzeige), {"laeufe": []})


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
        laufbuch._leeren()
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        self.u = nutzer()
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))
        self.p, (self.d,) = projekt(self.u["id"])
        self.b1 = bild(self.p, self.d, 1, 0, "done", alt="Fertiger Text (fiktiv).")
        self.b2 = bild(self.p, self.d, 2, 0, "processing")

    def tearDown(self):
        mock.patch.stopall()
        laufbuch._leeren()

    def test_endpunkt_nur_mit_schalter(self):
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", False):
            self.assertEqual(self.c.get("/api/agent/live").status_code, 404)

    def test_endpunkt_liest_das_laufbuch(self):
        l = agent_lauf("alt_texte", self.u["id"], self.p, gesamt=2)
        l.stueck(image_id=self.b1)
        l.weiter()
        l.stueck(image_id=self.b2)
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            r = self.c.get("/api/agent/live")
        self.assertEqual(r.status_code, 200, r.text)
        (x,) = r.json()["laeufe"]
        self.assertEqual((x["stelle"]["image_id"], x["fertig"], x["gesamt"], x["vorher"]["text"]), (self.b2, 1, 2, "Fertiger Text (fiktiv)."))

    def test_alt_text_lauf_meldet_sich_an_und_ab_auch_bei_abbruch(self):
        main = self.main
        c = database.get_db()
        c.execute("UPDATE images SET status = 'pending' WHERE id = ?", (self.b2,))
        c.commit()
        c.close()
        gesehen = {}

        async def lauf(project_id, *a, lauf=None, **kw):
            gesehen["drin"] = laufbuch.fuer_konto(self.u["id"], nur_agent=False)[0]
            lauf.stueck(image_id=self.b2)
            lauf.weiter()
        with mock.patch.object(main, "_process_project_lauf", side_effect=lauf):
            with laufbuch.agent_kontext():
                asyncio.run(main._process_project(self.p, self.u["id"]))
        self.assertEqual((gesehen["drin"]["art"], gesehen["drin"]["gesamt"], gesehen["drin"]["laeuft"]), ("alt_texte", 1, True),
                         "beim Start gezaehlt: ein Bild 'pending'")
        d = laufbuch.fuer_konto(self.u["id"])[0]
        self.assertEqual((d["laeuft"], d["ok"], d["fertig"], d["vorher"]["image_id"]), (False, True, 1, self.b2), "vom Agenten, abgemeldet")

        laufbuch._leeren()

        async def kaputt(*a, **kw):
            raise RuntimeError("Abbruch (fiktiv)")
        with mock.patch.object(main, "_process_project_lauf", side_effect=kaputt), \
                mock.patch.object(main, "_notaufraeumen") as aufraeumen:
            with self.assertRaises(RuntimeError):
                asyncio.run(main._process_project(self.p, self.u["id"]))
        aufraeumen.assert_called_once()
        d = laufbuch.fuer_konto(self.u["id"], nur_agent=False)[0]
        self.assertEqual((d["laeuft"], d["ok"], d["quelle"]), (False, False, "oberflaeche"), "auch nach Abbruch abgemeldet")

    def test_zaehlt_nur_die_bilder_des_laufs(self):
        main = self.main
        _, (d2,) = projekt(self.u["id"])
        c = database.get_db()
        c.execute("UPDATE images SET status = 'pending' WHERE id = ?", (self.b2,))
        c.commit()
        c.close()
        self.assertEqual(main._alt_lauf_gesamt(self.p, None), 1)
        self.assertEqual(main._alt_lauf_gesamt(self.p, d2), 0, "nur Bilder dieses Dokuments")
        with mock.patch.object(main, "get_db", side_effect=RuntimeError("weg")):
            self.assertEqual(main._alt_lauf_gesamt(self.p, None), 0, "die Anzeige haelt den Lauf nie auf")

    def test_maentel_melden_ergebnis_und_fehler(self):
        main = self.main
        projekt_ = {"id": self.p}
        maentel = (
            ("_pdf_export_sync", "_pdf_export_kern", "export", lambda d: (self.u["id"], projekt_, d, None)),
            ("_pdfua_umwandeln_sync", "_pdfua_umwandeln_kern", "umwandeln", lambda d: (projekt_, self.u["id"], d, None, "de")),
            ("_pdfua_vorschau_sync", "_pdfua_vorschau_kern", "pruefung", lambda d: (projekt_, self.u["id"], d, "de")),
            ("_word_export_ausgabe_sync", "_word_export_ausgabe_kern", "export", lambda d: (projekt_, self.u["id"], d, None, "de")),
        )
        for mantel, kern, art, argumente in maentel:
            with self.subTest(mantel):
                laufbuch._leeren()
                aid = ausgabe(self.u["id"], self.p)
                with mock.patch.object(main, kern, return_value={"ausgabe_ids": [aid]}) as k, laufbuch.agent_kontext():
                    self.assertEqual(getattr(main, mantel)(*argumente(self.d)), {"ausgabe_ids": [aid]})
                k.assert_called_once()
                d = laufbuch.fuer_konto(self.u["id"])[0]
                self.assertEqual((d["art"], d["ok"], d["fertig"], d["ergebnis"], d["vorher"]),
                                 (art, True, 1, {"art": "ausgabe", "ausgabe_id": aid}, {"art": "dokument", "document_id": self.d}))
                laufbuch._leeren()
                with mock.patch.object(main, kern, side_effect=RuntimeError("LibreOffice weg (fiktiv)")), laufbuch.agent_kontext():
                    with self.assertRaises(RuntimeError):
                        getattr(main, mantel)(*argumente(None))
                d = laufbuch.fuer_konto(self.u["id"])[0]
                self.assertEqual((d["art"], d["laeuft"], d["ok"], d["ergebnis"]), (art, False, False, None))

    def test_pruefdatei_meldet_sich_an_und_ab(self):
        main = self.main
        with mock.patch.object(main, "_abschluss_bauen", return_value=True), laufbuch.agent_kontext():
            self.assertTrue(main._abschluss_erstellen_sync({"id": self.p}, self.u["id"], self.d))
        d = laufbuch.fuer_konto(self.u["id"])[0]
        self.assertEqual((d["art"], d["ok"], d["vorher"]), ("pruefung", True, {"art": "dokument", "document_id": self.d}))
        self.assertNotIn(self.d, main._abschluss_laeuft, "die Sperre ist wieder frei")


class Anschluss(unittest.TestCase):
    """Jeder Ort, der einen Lauf anmeldet, nimmt eine Art, die das Laufbuch kennt; Werkzeuge laufen als Agent; Threads, Rechenbereiche
    und fremde Schleifen nehmen das mit (ohne laufbuch.mitnehmen erbt ein Thread nichts — dann waere der Lauf „Oberflaeche“)."""

    def quellen(self):
        import glob
        pfade = glob.glob(os.path.join(BACKEND, "*.py")) + glob.glob(os.path.join(BACKEND, "inkluagent", "**", "*.py"), recursive=True)
        return {os.path.relpath(p, BACKEND): lies(p) for p in pfade if os.path.basename(p) != "laufbuch.py"}

    def test_jede_angemeldete_art_ist_bekannt_und_jede_art_angeschlossen(self):
        import re
        gefunden: dict = {}
        for name, text in self.quellen().items():
            for art in re.findall(r'laufbuch\.(?:beginnen|lauf)\(\s*"([a-z_]+)"', text):
                gefunden.setdefault(art, set()).add(name)
        self.assertLessEqual(set(gefunden), set(laufbuch.ARTEN), gefunden)
        self.assertEqual(set(gefunden), set(laufbuch.ARTEN), "jede Art hat (mindestens) einen Lauf, der sich anmeldet")
        for art, wo in (("alt_texte", "main.py"), ("quickinfos", "formular_api.py"), ("tagging", "tagging_api.py"),
                        ("korrektur", "tagging_api.py"), ("pruefung", "tagging_api.py"), ("pruefung", "main.py"),
                        ("uebersetzen", "uebersetzung_api.py"), ("auftrag", "kette_api.py"),
                        ("auftrag", os.path.join("inkluagent", "auftraege.py"))):
            self.assertIn(wo, gefunden.get(art, set()), (art, wo))

    def test_werkzeuge_laufen_als_agent(self):
        d = lies(BACKEND, "inkluagent", "tools", "definitions.py")
        self.assertIn("with ki_kosten.kontext(user_id=self.user_id, project_id=self.project_id), laufbuch.agent_kontext():\n"
                      "                        ergebnis = handler(args)", d)
        self.assertIn("with laufbuch.agent_kontext():", lies(BACKEND, "inkluagent", "tools", "definitions_formular.py"))
        self.assertIn('beginnen("auftrag", user_id=a["user_id"]', lies(BACKEND, "inkluagent", "auftraege.py"))
        self.assertIn('quelle="agent"', lies(BACKEND, "inkluagent", "auftraege.py"), "Auftraege sind immer Arbeit des Agenten")

    def test_threads_und_schleifen_nehmen_den_agenten_mit(self):
        pdf = lies(BACKEND, "inkluagent", "tools", "pdf.py")
        self.assertEqual(pdf.count("threading.Thread("), pdf.count("threading.Thread(target=laufbuch.mitnehmen("))
        self.assertGreaterEqual(pdf.count("threading.Thread(target=laufbuch.mitnehmen("), 3)
        self.assertIn("asyncio.run_coroutine_threadsafe(laufbuch.mitnehmen_coro(coro), _HAUPT_LOOP)", lies(BACKEND, "main.py"))
        self.assertIn("_test_executor.submit(laufbuch.mitnehmen(_test_sync)", lies(BACKEND, "tagging_api.py"))
        self.assertIn("asyncio.run_coroutine_threadsafe(laufbuch.mitnehmen_coro(lauf_starten(", lies(BACKEND, "uebersetzung_api.py"))
        k = lies(BACKEND, "kette_api.py")
        self.assertIn("laufbuch.mitnehmen(_d.tagging_lauf)", k)
        self.assertIn("asyncio.run_coroutine_threadsafe(laufbuch.mitnehmen_coro(_kette(", k)

    def test_stream_nennt_die_ansicht_des_werkzeugs(self):
        self.assertIn('_z = _af.ziel(user["id"], project_id, args, name)', lies(BACKEND, "main.py"))
        self.assertIn('return await asyncio.to_thread(_af.live, user["id"], None, _display_alt_text)', lies(BACKEND, "main.py"))
        code = [z for z in lies(BACKEND, "main.py").splitlines() if "_ALT_LAEUFE" in z and not z.strip().startswith("#")]
        self.assertEqual(code, [], "die eigene Liste vom 16.10. ist im Laufbuch aufgegangen")


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")
        self.zeigen = funktion(self.js, "function inkluagentLiveZeigen(")
        self.beob = funktion(self.js, "function inkluagentLiveBeobachten(")

    def test_anstoss_wie_die_rueckmeldung(self):
        rm = funktion(self.js, "async function inkluagentRueckmeldungenBeobachten(")
        self.assertLess(rm.index("inkluagentLiveBeobachten();"), rm.index("if (!inkluagentRueckmeldungAn()"),
                        "auch ohne Rueckmeldungs-Schalter")
        self.assertIn("if (!inkluagentArbeitsfensterAn()) return;", self.beob)
        self.assertIn("fetch('/api/agent/live')", self.beob)
        self.assertIn("d.laeufe.some(x => x && x.laeuft)", self.beob, "weiter, solange ein Lauf arbeitet")
        self.assertIn("document.visibilityState === 'hidden'", self.beob, "unsichtbarer Tab fragt nicht")
        self.assertIn("setTimeout(runde, 2500)", self.beob)

    def test_nur_eigene_bildadressen_und_textcontent(self):
        self.assertIn(r"/^\/api\/images\/\d+\/(file|page-view)$/", self.zeigen)
        self.assertIn("q.textContent = String(v.text);", self.zeigen)
        self.assertIn('<img alt="">', funktion(self.js, "function inkluagentArbeitsfensterBauen("))
        for kopf in ("function inkluagentLiveZeigen(", "function inkluagentLiveTexte(", "function inkluagentLiveStelleText(",
                     "function inkluagentLiveTitel(", "function inkluagentBuehneGemeldet("):
            self.assertNotIn("innerHTML", funktion(self.js, kopf), kopf)

    def test_keine_ansagen_tick_je_stueck_hand_nur_beim_ersten_bild(self):
        for f in (self.zeigen, self.beob, funktion(self.js, "function inkluagentFensterFolgen(")):
            self.assertNotIn("announce(", f)
        self.assertIn("if (derselbe && neu.vorher && neu.vorher !== alt.vorher) { inkluagentKlang('schritt'); inkluagentRahmenNachziehen(); }",
                      self.zeigen, "ein Tick je fertigem Stueck, die echte Oberflaeche zeichnet neu")
        self.assertIn("if (wechsel && !derselbe && typeof zeigeHand === 'function') zeigeHand(fig);", self.zeigen)

    def test_kamera_folgt_der_stelle_und_am_ende_dem_ergebnis(self):
        self.assertIn("if (l.laeuft && l.stelle && (!derselbe || neu.stelle !== alt.stelle)) inkluagentFensterFolgen(l.stelle);", self.zeigen)
        # auch ein Lauf, der zwischen zwei Nachfragen begann und endete (Umwandeln im Werkzeug) — nicht beim ersten Blick, nur einmal
        ende = self.zeigen[self.zeigen.index("if ((derselbe && alt.lauf.laeuft && !l.laeuft) || (!derselbe && !erster && !l.laeuft)) {"):]
        self.assertIn("inkluagentRahmenNachziehen();\n        if (window._arbeitsLiveGefolgt !== l.nr) {\n"
                      "            window._arbeitsLiveGefolgt = l.nr;\n            if (l.ergebnis) inkluagentFensterFolgen(l.ergebnis);", ende)
        # Rauchtest 17.10.2026 (Word herunterladen ohne Ablage, ein Schritt von Sekunden): wenigstens die Station zeigen
        self.assertIn("else if (!derselbe && l.ansicht && l.project_id != null) inkluagentFensterFolgen({ art: 'ansicht', "
                      "project_id: l.project_id, ansicht: l.ansicht });", ende)
        self.assertIn("const derselbe = alt.lauf && alt.nr === l.nr;", self.zeigen)
        self.assertIn("const erster = !('lauf' in alt);", self.zeigen)

    def test_stand_vorlesen_und_kopfzeile(self):
        self.assertIn("inkluagentLiveTexte().forEach(x => teile.push(x));", funktion(self.js, "function inkluagentStandVorlesen("))
        z = funktion(self.js, "function inkluagentArbeitsZeigen(")
        self.assertIn("t('Arbeitet im Hintergrund: {was}, {n} von {m} fertig.'", z)
        self.assertIn("f.dataset.zustand = z.laeuft ? 'laeuft' : (lv ? 'hinten' : 'ruhe');", z)
        w = funktion(self.js, "function inkluagentLiveWas(")
        for art in laufbuch.ARTEN:
            self.assertIn(art + ": t(", w, f"die Art {art} hat einen Namen")

    def test_nach_dem_knopf_wartet_nichts_mehr(self):
        e = funktion(self.js, "function inkluagentEntschieden(")
        self.assertIn("inkluagentWartetAufDich(anhang) ? t('Wartet auf dich: Bitte entscheide unter der Antwort.') : "
                      "t('Fertig. Die Antwort steht im Gespräch.')", e)
        self.assertIn("if (z.laeuft) return;", e)
        best = funktion(self.js, "async function inkluagentBestaetigen(")
        self.assertLess(best.index("inkluagentEntschieden(data.anhang);"), best.index("if (wechsel && await inkluagentWechseln("))
        self.assertIn("if (res.ok) inkluagentEntschieden(data.anhang);", self.js, "auch nach einer Formular-Karte")

    def test_gestaltung_ohne_text_im_schmuck_und_bewegung_abschaltbar(self):
        css = lies(FRONTEND, "style.css")
        teil = css[css.index("/* ─── Gesicht wie das Sichtfenster in InkluEdit"):css.index("/* Eingebettete Seite (im Rahmen")]
        import re
        for inhalt in re.findall(r'content:\s*("[^"]*")', teil):
            self.assertEqual(inhalt, '""', "Schmuck ohne Text (Screenreader lesen generierten Text vor)")
        reduziert = teil[teil.index("@media (prefers-reduced-motion: reduce)"):]
        for anim in ("arbeitsfenster-stand::before", "arbeitsfenster-live h3::before", "arbeitsfenster-live-bild img",
                     "arbeitsfenster-live-alt"):
            self.assertIn(anim, reduziert[:reduziert.index("}\n")])
        self.assertIn("@media (forced-colors: active)", teil)
        self.assertIn("Lucide (ISC-Lizenz", teil)


class Buehne(unittest.TestCase):
    """Das Projekt selbst im Arbeitsfenster (echte Oberflaeche), Kamerafahrt, Mitsehen und Mithoeren (16.10.2026) — seit 17.10.2026
    nur ueber die feste Schnittstelle window.InkluBuehne (Kern dashboard.js, Vertrag tests/test_buehne.py)."""

    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")

    def test_projektseite_einrahmbar_aber_keine_fensterseite(self):
        from inkluagent import fenster
        self.assertTrue(fenster.einrahmbar("/app"))
        self.assertFalse(fenster.im_fenster("/app"), "Links auf Projekte wechseln weiter die ganze Seite")
        self.assertFalse(fenster.einrahmbar("/api/projects/1"))

    def test_eingebettet_ist_die_projektseite_nur_buehne(self):
        b = funktion(self.js, "async function inkluagentBinden(")
        self.assertTrue(b.split("{", 1)[1].lstrip().startswith("// Im Arbeitsfenster"))
        self.assertIn("if (window.EINGEBETTET === true) { window._inkluagentProjectId = projectId == null ? null : projectId; return; }", b)
        self.assertLess(b.index("window.EINGEBETTET === true"), b.index("inkluagentRueckmeldungenBeobachten();"))

    def test_eingebettet_gar_kein_agent(self):
        # Klickprobe 16.10.2026: ohne festen Bereich fiel die eingebettete Projektseite auf den alten eingebetteten Agenten zurueck,
        # dessen Agentenansicht die Bildliste im Fenster versteckte
        sh = funktion(self.js, "function inkluagentSectionHtml(")
        self.assertIn("if (window.EINGEBETTET === true) return '';", sh)
        self.assertLess(sh.index("window.EINGEBETTET === true"), sh.index("inkluagentInhaltHtml("))
        ini = funktion(self.js, "function inkluagentInit(")
        self.assertLess(ini.index("if (window.EINGEBETTET === true) {"), ini.index("inkluagentInitLaden(projectId)"))

    def test_nur_ueber_die_schnittstelle(self):
        import re
        # keine Kennungen, Funktionen oder Rollen der Seite im Fenster mehr (frueher: showProject, img_heading_, scrollTo von aussen)
        for weg in ("inkluagentZielKarte", "inkluagentImRahmenZeigen", "inkluagentRahmenRollen", "inkluagentRahmenNeuGezeichnet",
                    "inkluagentProjektGeladen", "contentWindow.showProject", "win.showProject", "win.scrollTo"):
            self.assertNotIn(weg, self.js, weg)
        self.assertIsNone(re.search(r"(doc|contentDocument)\.getElementById\('[a-z]", self.js), "keine festen Kennungen im Rahmen")
        self.assertEqual(self.js.count("contentWindow.InkluBuehne"), 2, "Folgen und Nachziehen")
        self.assertIn("const b = win && win.InkluBuehne;", funktion(self.js, "function inkluagentBuehneGeladen("))

    def test_projekt_im_rahmen_nur_dieses_oder_vom_agenten_geoeffnet(self):
        f = funktion(self.js, "function inkluagentIstProjektImRahmen(")
        self.assertIn(r"if (!loc || !/^\/app\/?$/.test(loc.pathname)) return false;", f)
        self.assertIn("p === String(window._inkluagentProjectId)", f)
        self.assertIn("s && s.projekt && String(s.projektId) === p", f)
        g = funktion(self.js, "function inkluagentRahmenGeladen(")
        self.assertIn("const projekt = eigene && eingebettet === true && inkluagentIstProjektImRahmen(loc);", g)
        self.assertIn("(fensterSeite(loc.pathname) || projekt)", g)
        self.assertIn("vomAgenten: !!(vorher.vomAgenten && gleich)", g, "geht man im Rahmen woanders hin, ist es die eigene Seite")
        self.assertIn("inkluagentBuehneGeladen(win, doc);", g)

    def test_kamerafahrt_zieht_niemanden_weg(self):
        f = funktion(self.js, "function inkluagentFensterFolgen(")
        self.assertIn("if (!inkluagentFolgenAn()) return;", f)
        self.assertIn("if (!rahmen || document.activeElement === rahmen) return;", f)
        self.assertIn("if (s && s.offen && !s.projekt && !s.vomAgenten) return;", f, "eine selbst geoeffnete Seite bleibt")
        self.assertIn("Promise.resolve(b.zeigen(stelle, { weich: true })).catch(() => {});", f)
        g = funktion(self.js, "function inkluagentBuehneGeladen(")
        self.assertIn("['pointerdown', 'wheel', 'keydown', 'touchstart']", g)
        self.assertIn("if (e.isTrusted) inkluagentFolgenSetzen(false, false);", g)
        self.assertIn("if (window._arbeitsFolgeZiel) inkluagentFensterFolgen(window._arbeitsFolgeZiel);", g)
        n = funktion(self.js, "function inkluagentRahmenNachziehen(")
        self.assertIn("document.activeElement === rahmen) return;", n, "nie neu zeichnen, solange man im Fenster arbeitet")

    def test_folgt_in_andere_eigene_projekte_und_in_die_ablage(self):
        f = funktion(self.js, "function inkluagentFensterFolgen(")
        self.assertIn("if (!(s && s.ablage)) { inkluagentAblageImFenster(); return; }", f)
        self.assertIn("if (!(s && s.projekt && String(s.projektId) === pid)) { inkluagentProjektImFenster(pid, stelle.ansicht); return; }", f)
        p = funktion(self.js, "function inkluagentProjektImFenster(")
        self.assertIn("vomAgenten: !eigenes", p)
        self.assertIn("const darf = projektId == null ? inkluagentProjektFensterAn() : (inkluagentChatstilAktiv() && !window.GUEST_MODE);", p,
                      "auch von der Startseite aus, nie im Gastzugang")
        self.assertIn("inkluagentProjektPfad(pid, ansicht) + '&eingebettet=1'", p)
        self.assertIn("Object.assign(window._arbeitsSeite, { ablage: true, vomAgenten: true });",
                      funktion(self.js, "function inkluagentAblageImFenster("))
        z = funktion(self.js, "function inkluagentSeiteZeigen(")
        self.assertIn("f.querySelector('#arbeitsSeiteZu').hidden = eigenes;", z, "ein fremdes Projekt schliesst man wieder")
        self.assertIn("f.querySelector('#arbeitsFolgen').hidden = !(s.projekt || s.vomAgenten);", z)
        self.assertIn("if (!s || s.projekt || s.vomAgenten) return;", funktion(self.js, "function inkluagentSeiteMerken("))

    def test_mithoeren_aus_der_buehne(self):
        g = funktion(self.js, "function inkluagentBuehneGemeldet(")
        self.assertIn("try { if (rahmen.contentDocument !== doc) return; } catch (e) { return; }", g, "nur die Seite, die im Fenster steht")
        self.assertIn("inkluagentMithoerenSetzen({ art: 'bild', alt: g.text });", g)
        self.assertIn("inkluagentMithoerenSetzen({ art: 'feld', info: g.text });", g)
        m = funktion(self.js, "function inkluagentMithoerenSetzen(")
        self.assertIn("inkluagentTextSetzen(box.querySelector('#arbeitsHoerenText'), text);", m)
        self.assertNotIn("innerHTML", m)
        bau = funktion(self.js, "function inkluagentArbeitsfensterBauen(")
        self.assertIn("inkluagentAnhoerenKnopf(s.querySelector('#arbeitsHoerenTitel'), s.querySelector('#arbeitsHoerenText'));", bau)
        self.assertIn('id="arbeitsFolgen" class="btn btn-secondary" aria-pressed="true" hidden>', bau)

    def test_fenster_nimmt_beim_laden_keinen_fokus(self):
        dash = lies(FRONTEND, "dashboard.js")
        self.assertIn("return !imRahmen() || document.hasFocus();", funktion(dash, "function fokusHierErlaubt("))
        for datei, n in (("dokument.js", 1), ("abschluss.js", 2), ("formular.js", 1), ("uebersetzen.js", 1)):
            q = lies(FRONTEND, datei)
            self.assertEqual(q.count("if (h1 && !erneut && (typeof fokusHierErlaubt !== 'function' || fokusHierErlaubt())) h1.focus();"), n, datei)
            self.assertNotIn("if (h1 && !erneut) h1.focus();", q, datei)

    def test_ansichtslinks_im_fenster_bleiben_im_fenster(self):
        dash = lies(FRONTEND, "dashboard.js")
        e = funktion(dash, "function eingebettetEinrichten(")
        self.assertIn("if (u.origin !== window.location.origin || fensterSeite(u.pathname) || gleichesProjekt(u)) return;", e)
        g = funktion(dash, "function gleichesProjekt(")
        self.assertIn("u.searchParams.get('projekt') === hier", g)

    def test_gespraechstitel_ueberlappt_nicht(self):
        dash = lies(FRONTEND, "dashboard.js")
        self.assertIn("text.className = 'app-nav-gespraech-text';", dash)
        css = lies(FRONTEND, "dashboard.css")
        regel = css[css.index(".app-nav-gespraeche-liste a {"):]
        self.assertNotIn("line-clamp", regel[:regel.index("}")], "Kuerzung nicht am Link mit Innenabstand")
        self.assertIn(".app-nav-gespraech-text { display:-webkit-box; -webkit-line-clamp:2;", css)

    def test_projekt_hat_vorrang_beim_wiederherstellen(self):
        w = funktion(self.js, "function inkluagentFensterWiederherstellen(")
        self.assertIn("frisch && (!inkluagentProjektFensterAn() || w.seite === inkluagentPfadOhne(hier.split('#')[0], ['hin']))", w)
        self.assertIn("inkluagentProjektImFenster();", w)
        zu = funktion(self.js, "function inkluagentSeiteSchliessen(")
        self.assertIn("if (inkluagentProjektFensterAn()) inkluagentProjektImFenster();", zu)
        # Klickprobe 17.10.2026: ohne das holte das Laden des eigenen Projekts das andere Projekt bzw. die Ablage gleich zurueck
        self.assertLess(zu.index("window._arbeitsFolgeZiel = null;"), zu.index("if (inkluagentProjektFensterAn()) inkluagentProjektImFenster();"))

    def test_schritte_fahren_hin_und_die_ansicht_zieht_nach(self):
        self.assertIn("if (ziel && typeof ziel === 'object') inkluagentFensterFolgen(ziel);", funktion(self.js, "function inkluagentArbeitsSchritt("))
        self.assertIn("inkluagentRahmenNachziehen();", funktion(self.js, "async function inkluagentAnsichtNachziehen("))

    def test_gestaltung(self):
        css = lies(FRONTEND, "style.css")
        self.assertIn(".arbeitsfenster-hoeren {", css)
        self.assertIn("#arbeitsFolgen { --af-symbol:", css)
        self.assertIn('#arbeitsFolgen[aria-pressed="true"]', css)
        self.assertIn(".arbeitsfenster [hidden] { display: none !important; }", css, "Pillen duerfen hidden nicht ueberstimmen")


if __name__ == "__main__":
    unittest.main()
