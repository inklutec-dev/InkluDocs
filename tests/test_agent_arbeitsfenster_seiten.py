"""Seiten im Arbeitsfenster der Agentenansicht (Steve 12./13.10.2026; mit Schalter funktionen.AGENT_ARBEITSFENSTER): die
Agentenansicht zeigt nur Gespraech + Arbeitsfenster; Seiten oeffnen sich eingebettet im Arbeitsfenster (ohne Seitenleiste,
Fusszeile, Agent), „Arbeitsfenster vergrößern“, Rahmen-Schutz nur noch fuer FREMDE Seiten. Wegwerf-Datenbank, keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_arbeitsfenster_seiten.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_fenster_seiten_")
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

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def funktion(quelle, name):
    f = quelle[quelle.index("function " + name + "("):]
    return f[:f.index("\n}\n")]


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
        for name in ("AGENT_KONTO", "AGENT_UEBERALL", "AGENT_ANSICHT"):
            mock.patch.object(funktionen, name, True).start()
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        mail = f"fenster{id(self)}@beispiel.invalid"
        self.uid = database.create_user(mail, "geheim-12345", "Test (fiktiv)")
        self.c = self.TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.uid, mail, 0))

    def tearDown(self):
        mock.patch.stopall()

    def test_rahmen_schutz_nur_noch_fuer_fremde(self):
        r = self.c.get("/login")
        self.assertEqual((r.headers.get("x-frame-options"), r.headers.get("content-security-policy")), ("DENY", "frame-ancestors 'none'"),
                         "ohne Schalter wie bisher")
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            r = self.c.get("/login")
        self.assertEqual((r.headers.get("x-frame-options"), r.headers.get("content-security-policy")),
                         ("SAMEORIGIN", "frame-ancestors 'self'"), "nur die eigene Seite darf einrahmen")

    def test_eingebettete_seite_ohne_seitenleiste_fusszeile_agent(self):
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            normal = self.c.get("/einstellungen").text
            im_rahmen = self.c.get("/einstellungen", headers={"Sec-Fetch-Dest": "iframe"}).text
            per_parameter = self.c.get("/einstellungen?eingebettet=1").text
        self.assertIn('id="appSidebar"', normal)
        self.assertIn('id="inkluagentBereich"', normal)
        self.assertIn('class="dash-footer"', normal)
        for html in (im_rahmen, per_parameter):
            self.assertIn('class="eingebettet"', html)
            self.assertNotIn('id="appSidebar"', html)
            self.assertNotIn('id="inkluagentBereich"', html)
            self.assertNotIn('class="dash-footer"', html)
            self.assertIn('id="zumGespraech"', html)
            self.assertIn("window.EINGEBETTET = true;", html)
            self.assertIn('id="lang-h"', html, "der Inhalt der Seite ist da")
        ohne = self.c.get("/einstellungen", headers={"Sec-Fetch-Dest": "iframe"}).text
        self.assertNotIn('class="eingebettet"', ohne, "ohne Schalter nie eingebettet")


class Oberflaeche(unittest.TestCase):
    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")
        self.dash = lies(FRONTEND, "dashboard.js")

    def test_seite_im_fenster_nur_eigene_pfade_ohne_projekte(self):
        m = funktion(self.js, "inkluagentSeiteImFensterMoeglich")
        self.assertIn("if (!inkluagentChatstilAktiv()", m)
        self.assertIn(r"/^\/[A-Za-z0-9\/_-]*(\?[A-Za-z0-9=&_%-]*)?$/.test(pfad)", m)
        self.assertIn(r"!/^\/(app|dashboard)(\/|\?|$)/.test(pfad)", m)
        f = funktion(self.js, "inkluagentSeiteImFenster")
        self.assertIn("rahmen.title = t('Arbeitsfenster: {name}'", f, "der Rahmen hat einen Namen")
        self.assertIn("announce(t('Im Arbeitsfenster geöffnet: {name}'", f)
        self.assertIn("rahmen.contentWindow.focus();", f, "Fokus in die Seite")
        self.assertIn("'eingebettet=1'", f)
        self.assertNotIn("innerHTML", f)

    def test_vergroessern_umschaltknopf_und_inert_ueber_dem_gespraech(self):
        bau = funktion(self.js, "inkluagentArbeitsfensterBauen")
        self.assertIn('id="arbeitsGross" class="btn btn-secondary" aria-pressed="false">\' + t(\'Arbeitsfenster vergrößern\')', bau)
        self.assertIn("e.key === 'Escape'", bau)
        g = funktion(self.js, "inkluagentFensterGross")
        self.assertIn("knopf.setAttribute('aria-pressed', an ? 'true' : 'false');", g)
        self.assertIn("el.inert = ueber;", g)
        self.assertIn("html.arbeitsfenster-gross .arbeitsfenster { position: fixed; inset: 0;", lies(FRONTEND, "style.css"))

    def test_hinfuehren_und_navigation_oeffnen_im_fenster(self):
        self.assertIn("if (inkluagentSeiteImFensterMoeglich(a.url)) { inkluagentSeiteImFenster(a.url, a.name || ''); return false; }",
                      funktion(self.js, "inkluagentBegleiterAktion"))
        self.assertIn("a.addEventListener('click', (e) => imArbeitsfensterOeffnen(e, a));", self.dash)
        self.assertIn("window.inkluagentSeiteImFenster(pfad, a.textContent.trim());", funktion(self.dash, "imArbeitsfensterOeffnen"))

    def test_im_rahmen_ganze_seite_fuer_projekte_sprache_anmeldung_zahlung(self):
        self.assertIn(r"/^\/(app|dashboard)(\/|\?|$)/.test(pfad) || /^\/(set-language|login|logout|register)(\/|\?|$)/.test(pfad)",
                      funktion(self.dash, "gehoertGanzeSeite"))
        self.assertIn("ganzeSeite(data.weiter);", lies(BACKEND, "templates", "gespraech.html"))
        self.assertIn("ganzeSeite('/app?projekt='", lies(BACKEND, "templates", "projekt-neu.html"))
        self.assertEqual(lies(BACKEND, "templates", "abo.html").count("ganzeSeite(d.url);"), 3, "Zahlung nie im Rahmen")
        self.assertIn("window.top.location.reload();", lies(BACKEND, "templates", "einstellungen.html"), "Ansicht im Fenster gewechselt")


if __name__ == "__main__":
    unittest.main()
