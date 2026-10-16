"""Live-Fenster im Arbeitsfenster (Steve 16.10.2026: „wie einen Livestream … wenn er einen Alternativtext für die Bilder macht, dass
man in dem Moment das Bild eingeblendet sieht“; Schalter funktionen.AGENT_ARBEITSFENSTER).

Server: main._ALT_LAEUFE kennt nur Alt-Text-Laeufe, die in diesem Prozess wirklich arbeiten (eingetragen und ausgetragen von
_process_project, auch bei Abbruch) — ein haengender Projektstatus 'processing' zeigt nichts. GET /api/agent/live
(inkluagent/arbeitsfenster.live): nur Projekte und Bilder DIESES Kontos; Bild in Arbeit mit Nummer je Dokument, Seite, Dokument,
Adressen; Fortschritt des Laufs; vorher=<Bild> bringt dessen fertigen Alt-Text (wie die Oberflaeche ihn zeigt) bzw. „error“.
Oberflaeche am Quelltext: Anstoss mit der Rueckmeldung, Adressen nur der eigenen Bild-Endpunkte, alles mit textContent, keine
Ansagen, Hand nur beim ersten Bild, „Stand vorlesen“ mit Live-Angaben, Bewegung abschaltbar.
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


class Live(unittest.TestCase):
    def setUp(self):
        self.u = nutzer()
        self.p, (self.d,) = projekt(self.u["id"])
        self.b1 = bild(self.p, self.d, 1, 0, "done", alt="Ein Haus mit rotem Dach (fiktiv).")
        self.b2 = bild(self.p, self.d, 2, 0, "processing")
        self.b3 = bild(self.p, self.d, 3, 0, "pending")

    def test_ohne_eingetragenen_lauf_nichts(self):
        # ein haengender Status 'processing' (Projekt und Bild) zeigt nichts, solange kein Lauf eingetragen ist
        self.assertEqual(arbeitsfenster.live(self.u["id"], {}, None, anzeige), {"laeuft": None, "vorher": None})

    def test_bild_in_arbeit_mit_nummer_seite_dokument_und_fortschritt(self):
        d = arbeitsfenster.live(self.u["id"], {self.p: {"gesamt": 2, "document_id": None}}, None, anzeige)
        l = d["laeuft"]
        self.assertEqual((l["art"], l["project_id"], l["projekt"]), ("alt_texte", self.p, "Vereinsbrief (fiktiv)"))
        self.assertEqual((l["gesamt"], l["fertig"]), (2, 0), "Bild 2 in Arbeit und Bild 3 offen: noch keins fertig")
        b = l["bild"]
        self.assertEqual((b["image_id"], b["nummer"], b["seite"]), (self.b2, 2, 2))
        self.assertEqual((b["bild_url"], b["seite_url"]), (f"/api/images/{self.b2}/file", f"/api/images/{self.b2}/page-view"))
        self.assertEqual(b["dokument"], "Jahresbericht (fiktiv).pdf")
        status(self.b2, "done")
        status(self.b3, "processing")
        l = arbeitsfenster.live(self.u["id"], {self.p: {"gesamt": 2, "document_id": None}}, None, anzeige)["laeuft"]
        self.assertEqual((l["fertig"], l["bild"]["image_id"]), (1, self.b3))

    def test_gesamt_nie_kleiner_als_offen(self):
        l = arbeitsfenster.live(self.u["id"], {self.p: {"gesamt": 0, "document_id": None}}, None, anzeige)["laeuft"]
        self.assertEqual((l["gesamt"], l["fertig"]), (2, 0))

    def test_nur_eigene_projekte(self):
        fremd = nutzer()
        self.assertIsNone(arbeitsfenster.live(fremd["id"], {self.p: {"gesamt": 2, "document_id": None}}, None, anzeige)["laeuft"])

    def test_vorher_bringt_den_fertigen_alt_text(self):
        v = arbeitsfenster.live(self.u["id"], {}, self.b1, anzeige)["vorher"]
        self.assertEqual((v["image_id"], v["status"], v["alt"], v["nummer"]), (self.b1, "done", "Ein Haus mit rotem Dach (fiktiv).", 1))
        # wie die Oberflaeche: der eigene Text geht vor
        b = bild(self.p, self.d, 4, 0, "done", alt="KI-Text (fiktiv).", alt_edited="Eigener Text (fiktiv).")
        self.assertEqual(arbeitsfenster.live(self.u["id"], {}, b, anzeige)["vorher"]["alt"], "Eigener Text (fiktiv).")
        e = bild(self.p, self.d, 5, 0, "error")
        self.assertEqual(arbeitsfenster.live(self.u["id"], {}, e, anzeige)["vorher"]["status"], "error")

    def test_vorher_nur_fertig_und_nur_eigen(self):
        self.assertIsNone(arbeitsfenster.live(self.u["id"], {}, self.b2, anzeige)["vorher"], "noch in Arbeit")
        self.assertIsNone(arbeitsfenster.live(self.u["id"], {}, self.b3, anzeige)["vorher"], "noch offen")
        fremd = nutzer()
        self.assertIsNone(arbeitsfenster.live(fremd["id"], {}, self.b1, anzeige)["vorher"], "fremdes Bild")
        for unsinn in ("abc", True, 1.5, None, {"id": self.b1}):
            self.assertIsNone(arbeitsfenster.live(self.u["id"], {}, unsinn, anzeige)["vorher"], unsinn)

    def test_nie_eine_ausnahme(self):
        with mock.patch("database.get_db", side_effect=RuntimeError("weg")):
            self.assertEqual(arbeitsfenster.live(self.u["id"], {self.p: {}}, self.b1, anzeige), {"laeuft": None, "vorher": None})


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
        self.p, (d,) = projekt(self.u["id"])
        self.b1 = bild(self.p, d, 1, 0, "done", alt="Fertiger Text (fiktiv).")
        self.b2 = bild(self.p, d, 2, 0, "processing")

    def tearDown(self):
        mock.patch.stopall()
        self.main._ALT_LAEUFE.clear()

    def test_endpunkt_nur_mit_schalter(self):
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", False):
            self.assertEqual(self.c.get("/api/agent/live").status_code, 404)

    def test_endpunkt_mit_lauf_und_vorher(self):
        self.main._ALT_LAEUFE[self.p] = {"gesamt": 2, "document_id": None, "seit": 0}
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            r = self.c.get(f"/api/agent/live?vorher={self.b1}")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual((d["laeuft"]["bild"]["image_id"], d["laeuft"]["fertig"], d["laeuft"]["gesamt"]), (self.b2, 1, 2))
        self.assertEqual(d["vorher"]["alt"], "Fertiger Text (fiktiv).")
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            self.assertEqual(self.c.get("/api/agent/live?vorher=abc").status_code, 422, "nur ganze Zahlen")

    def test_lauf_traegt_sich_ein_und_aus_auch_bei_abbruch(self):
        main = self.main
        gesehen = {}

        async def lauf(project_id, *a, **kw):
            gesehen["drin"] = dict(main._ALT_LAEUFE.get(project_id) or {})
        with mock.patch.object(main, "_process_project_lauf", side_effect=lauf):
            asyncio.run(main._process_project(self.p, self.u["id"]))
        self.assertEqual(gesehen["drin"]["gesamt"], 0, "beim Start gezaehlt: kein Bild 'pending'")
        self.assertNotIn(self.p, main._ALT_LAEUFE, "nach dem Lauf ausgetragen")

        async def kaputt(*a, **kw):
            raise RuntimeError("Abbruch (fiktiv)")
        with mock.patch.object(main, "_process_project_lauf", side_effect=kaputt), \
                mock.patch.object(main, "_notaufraeumen") as aufraeumen:
            with self.assertRaises(RuntimeError):
                asyncio.run(main._process_project(self.p, self.u["id"]))
        aufraeumen.assert_called_once()
        self.assertNotIn(self.p, main._ALT_LAEUFE, "auch nach einem Abbruch ausgetragen")

    def test_zaehlt_nur_die_bilder_des_laufs(self):
        main = self.main
        _, (d2,) = projekt(self.u["id"])
        c = database.get_db()
        c.execute("UPDATE images SET status = 'pending' WHERE id = ?", (self.b2,))
        c.commit()
        c.close()
        e = main._alt_lauf_eintragen(self.p, None)
        self.assertEqual(e["gesamt"], 1)
        e = main._alt_lauf_eintragen(self.p, d2)
        self.assertEqual(e["gesamt"], 0, "nur Bilder dieses Dokuments")


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
        self.assertIn("fetch('/api/agent/live'", self.beob)
        self.assertIn("document.visibilityState === 'hidden'", self.beob, "unsichtbarer Tab fragt nicht")
        self.assertIn("setTimeout(runde, 2500)", self.beob)

    def test_nur_eigene_bildadressen_und_textcontent(self):
        self.assertIn(r"/^\/api\/images\/\d+\/(file|page-view)$/", self.zeigen)
        self.assertNotIn("innerHTML", self.zeigen)
        self.assertIn("q.textContent = String(v.alt);", self.zeigen)
        self.assertIn('<img alt="">', funktion(self.js, "function inkluagentArbeitsfensterBauen("))

    def test_keine_ansagen_hand_nur_beim_ersten_bild(self):
        for f in (self.zeigen, self.beob):
            self.assertNotIn("announce(", f)
        self.assertIn("if (wechsel && !alt.bild && !buehne && typeof zeigeHand === 'function') zeigeHand(fig);", self.zeigen)
        self.assertIn("inkluagentKlang('schritt');", self.zeigen, "ein Tick je fertigem Bild")

    def test_stand_vorlesen_und_kopfzeile(self):
        self.assertIn("inkluagentLiveTexte().forEach(x => teile.push(x));", funktion(self.js, "function inkluagentStandVorlesen("))
        z = funktion(self.js, "function inkluagentArbeitsZeigen(")
        self.assertIn("t('Arbeitet im Hintergrund: Alt-Texte, {n} von {m} fertig.'", z)
        self.assertIn("f.dataset.zustand = z.laeuft ? 'laeuft' : (lv ? 'hinten' : 'ruhe');", z)

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
    """Das Projekt selbst im Arbeitsfenster (echte Oberflaeche), Kamerafahrt, Mitsehen und Mithoeren (16.10.2026)."""

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

    def test_nur_dieses_projekt_zaehlt_als_erfolg(self):
        f = funktion(self.js, "function inkluagentIstProjektImRahmen(")
        self.assertIn("new URLSearchParams(loc.search).get('projekt') === String(window._inkluagentProjectId)", f)
        g = funktion(self.js, "function inkluagentRahmenGeladen(")
        self.assertIn("const projekt = eigene && eingebettet === true && inkluagentIstProjektImRahmen(loc);", g)
        self.assertIn("(fensterSeite(loc.pathname) || projekt)", g)

    def test_kamerafahrt_zieht_niemanden_weg(self):
        f = funktion(self.js, "function inkluagentFensterFolgen(")
        self.assertIn("if (!rahmen || document.activeElement === rahmen) return;", f)
        self.assertIn("if (!inkluagentFolgenAn() || !s || !s.projekt || s.laedt) return;", f)
        self.assertIn("lauf !== window._arbeitsFolgeLauf || document.activeElement === rahmen", f)
        z = funktion(self.js, "function inkluagentImRahmenZeigen(")
        self.assertNotIn("scrollIntoView", z, "scrollIntoView rollte auch die Seite draussen")
        self.assertIn("inkluagentRahmenRollen(win, ziel, !ruhig);", z)
        self.assertIn("prefers-reduced-motion: reduce", z)
        self.assertIn("win.scrollTo(", funktion(self.js, "function inkluagentRahmenRollen("))
        n2 = funktion(self.js, "function inkluagentRahmenNeuGezeichnet(")
        self.assertIn("if (k && Date.now() < m.halten && inkluagentFolgenAn() && document.activeElement !== rahmen) {", n2)
        g = funktion(self.js, "function inkluagentProjektGeladen(")
        self.assertIn("['pointerdown', 'wheel', 'keydown', 'touchstart']", g)
        self.assertIn("if (e.isTrusted) inkluagentFolgenSetzen(false, false);", g)
        n = funktion(self.js, "function inkluagentRahmenNachziehen(")
        self.assertIn("document.activeElement === rahmen) return;", n, "nie neu zeichnen, solange man im Fenster arbeitet")

    def test_neu_gezeichnet_ueberschrift_marke_mithoeren(self):
        g = funktion(self.js, "function inkluagentProjektGeladen(")
        self.assertIn("}).observe(main, { childList: true, subtree: true });", g)
        n = funktion(self.js, "function inkluagentRahmenNeuGezeichnet(")
        self.assertIn("if (rahmen.contentDocument !== doc) return;", n, "nur die Seite, die gerade im Fenster steht")
        self.assertIn("if (k && Date.now() < m.bis && !k.classList.contains('inkluagent-hin-ziel')) {", n)
        self.assertIn("inkluagentMithoerenSetzen({ art: 'bild', alt: feld.value });", n)
        f = funktion(self.js, "function inkluagentFensterFolgen(")
        self.assertIn("if (!nachLaden && karte.ansicht && aktuell && aktuell !== karte.ansicht) { inkluagentProjektImFenster(karte.ansicht); return; }", f)
        self.assertIn("const neu = s.projekt ? s.name : t('Seite: {name}', { name: s.name });", funktion(self.js, "function inkluagentSeiteZeigen("))

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

    def test_fertiges_bild_und_ende_zeichnen_das_fenster_neu(self):
        self.assertIn("if (neu.vorher !== alt.vorher || (alt.laeuft && !l)) inkluagentRahmenNachziehen();",
                      funktion(self.js, "function inkluagentLiveZeigen("))

    def test_karten_der_ziele(self):
        k = funktion(self.js, "function inkluagentZielKarte(")
        for teil in ("'img_heading_' + id", "'alttext_' + id", "ansicht: 'alttexte'", "'feld_heading_' + id", "ansicht: 'quickinfos'",
                     "'doc_heading_' + id", "ansicht: 'dokument'"):
            self.assertIn(teil, k)

    def test_mithoeren_mit_textcontent_und_anhoeren(self):
        m = funktion(self.js, "function inkluagentMithoerenSetzen(")
        self.assertIn("inkluagentTextSetzen(box.querySelector('#arbeitsHoerenText'), text);", m)
        self.assertNotIn("innerHTML", m)
        bau = funktion(self.js, "function inkluagentArbeitsfensterBauen(")
        self.assertIn("inkluagentAnhoerenKnopf(s.querySelector('#arbeitsHoerenTitel'), s.querySelector('#arbeitsHoerenText'));", bau)
        self.assertIn('id="arbeitsFolgen" class="btn btn-secondary" aria-pressed="true" hidden>', bau)

    def test_projekt_hat_vorrang_beim_wiederherstellen(self):
        w = funktion(self.js, "function inkluagentFensterWiederherstellen(")
        self.assertIn("frisch && (!inkluagentProjektFensterAn() || w.seite === inkluagentPfadOhne(hier.split('#')[0], ['hin']))", w)
        self.assertIn("inkluagentProjektImFenster();", w)
        self.assertIn("if (!s || s.projekt) return;", funktion(self.js, "function inkluagentSeiteMerken("), "das Projekt wird nicht gemerkt")
        self.assertIn("if (inkluagentProjektFensterAn()) inkluagentProjektImFenster();", funktion(self.js, "function inkluagentSeiteSchliessen("))

    def test_schritte_und_lauf_fahren_hin(self):
        self.assertIn("if (ziel && typeof ziel === 'object') inkluagentFensterFolgen(ziel);", funktion(self.js, "function inkluagentArbeitsSchritt("))
        self.assertIn("if (wechsel) inkluagentFensterFolgen({ art: 'bild', image_id: b.image_id, bild_url: b.bild_url });",
                      funktion(self.js, "function inkluagentLiveZeigen("))
        self.assertIn("inkluagentRahmenNachziehen();", funktion(self.js, "async function inkluagentAnsichtNachziehen("))

    def test_gestaltung(self):
        css = lies(FRONTEND, "style.css")
        self.assertIn(".arbeitsfenster-hoeren {", css)
        self.assertIn("#arbeitsFolgen { --af-symbol:", css)
        self.assertIn('#arbeitsFolgen[aria-pressed="true"]', css)
        self.assertIn(".arbeitsfenster [hidden] { display: none !important; }", css, "Pillen duerfen hidden nicht ueberstimmen")


if __name__ == "__main__":
    unittest.main()
