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
            r = self.c.get("/einstellungen")
            anmelden = self.c.get("/register")
        self.assertEqual((r.headers.get("x-frame-options"), r.headers.get("content-security-policy")),
                         ("SAMEORIGIN", "frame-ancestors 'self'"), "nur die eigene Seite darf einrahmen")
        self.assertEqual((anmelden.headers.get("x-frame-options"), anmelden.headers.get("content-security-policy")),
                         ("DENY", "frame-ancestors 'none'"), "Registrieren nie im Rahmen (Pruefung 13.10.2026, S15)")

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

    def test_parameter_gilt_nicht_im_hauptfenster(self):
        """Pruefung 13.10.2026 (K7): ?eingebettet=1 wanderte ueber die Anmeldung ins Hauptfenster — dort gilt er nicht mehr."""
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            haupt = self.c.get("/einstellungen?eingebettet=1", headers={"Sec-Fetch-Dest": "document"}).text
            rahmen = self.c.get("/einstellungen?eingebettet=1", headers={"Sec-Fetch-Dest": "iframe"}).text
            alt = self.c.get("/einstellungen?eingebettet=1").text   # Browser ohne Fetch-Metadaten
        self.assertNotIn('class="eingebettet"', haupt)
        self.assertIn('id="appSidebar"', haupt)
        self.assertIn('class="eingebettet"', rahmen)
        self.assertIn('class="eingebettet"', alt)

    def test_agent_ist_in_der_agentenansicht_hauptinhalt(self):
        """Pruefung 13.10.2026 (B17): Agentenansicht = <main> um Seite und Agent (section), manuell = aside nach main; #main bleibt."""
        from inkluagent import ansicht
        manuell = self.c.get("/einstellungen").text
        self.assertIn('<main id="main" class="dash-main" tabindex="-1">', manuell)
        self.assertIn('<aside id="inkluagentBereich" class="inkluagent-bereich" aria-labelledby="inkluagentHeading"></aside>', manuell)
        self.assertNotIn('agent-hauptbereich', manuell)
        ansicht.setzen(self.uid, "agent")
        agent = self.c.get("/einstellungen").text
        self.assertIn('<main class="agent-hauptbereich">', agent)
        self.assertIn('<div id="main" class="dash-main" tabindex="-1">', agent)
        self.assertIn('<section id="inkluagentBereich" class="inkluagent-bereich" aria-labelledby="inkluagentHeading"></section>', agent)
        self.assertLess(agent.index('<div id="main"'), agent.index('<section id="inkluagentBereich"'))
        self.assertLess(agent.index('<section id="inkluagentBereich"'), agent.index('</main>'))
        self.assertEqual(agent.count("<main"), 1)
        self.assertNotIn('tabindex="-1"></aside>', manuell, "B19: kein dauerhaftes tabindex am Bereich")
        with mock.patch.object(funktionen, "AGENT_UEBERALL", False):
            ohne = self.c.get("/einstellungen").text
        self.assertIn('<main id="main" class="dash-main" tabindex="-1">', ohne, "ohne Agent auf jeder Seite wie bisher")
        self.assertNotIn('id="inkluagentBereich"', ohne)


class Oberflaeche(unittest.TestCase):
    """Quelltext-Pruefungen; das Verhalten pruefen die Klickproben lokal/probe_pr13_*.py (docs/INKLUAGENT.md, Pruefung 13.10.2026)."""

    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")
        self.dash = lies(FRONTEND, "dashboard.js")

    def test_seite_im_fenster_nur_eigene_pfade_ohne_projekte(self):
        m = funktion(self.js, "inkluagentSeiteImFensterMoeglich")
        self.assertIn("if (!inkluagentChatstilAktiv() || typeof pfad !== 'string' || !eigenerPfad(pfad)) return false;", m)
        self.assertIn(r"/^\/[A-Za-z0-9\/_-]*(\?[A-Za-z0-9=&_%-]*)?(#[A-Za-z0-9_-]*)?$/.test(pfad)", m)
        self.assertIn("return !gehoertGanzeSeite(pfad);", m, "eine Liste fuer Elternseite und Rahmen (dashboard.js)")
        zeigen = funktion(self.js, "inkluagentSeiteZeigen")
        self.assertIn(".title = t('Arbeitsfenster: {name}'", zeigen, "der Rahmen hat einen Namen")
        f = funktion(self.js, "inkluagentSeiteImFenster")
        self.assertIn("'eingebettet=1'", f)
        self.assertNotIn("announce(", f, "B15: keine eigene Ansage, wenn der Fokus ohnehin in die Seite geht")
        self.assertNotIn("innerHTML", f)
        self.assertIn("rahmen.contentWindow.location.replace(url)", funktion(self.js, "inkluagentRahmenLaden"), "K6: ohne Verlaufseintrag")
        fokus = funktion(self.js, "inkluagentRahmenFokus")
        self.assertIn("win.focus();", fokus, "Fokus in die Seite")

    def test_rahmen_folgt_dem_geladenen_dokument(self):
        """K6/K7/B14: ein dauerhafter load-Hoerer liest Pfad und Ueberschrift aus dem Dokument und bricht aus, wo noetig."""
        bau = funktion(self.js, "inkluagentArbeitsfensterBauen")
        self.assertIn("rahmen.addEventListener('load', () => inkluagentRahmenGeladen(rahmen));", bau)
        self.assertNotIn("{ once: true }", bau)
        g = funktion(self.js, "inkluagentRahmenGeladen")
        self.assertIn("doc.querySelector('#main h1')", g)
        self.assertIn("gehoertGanzeSeite(loc.pathname) || !win || win.EINGEBETTET !== true", g)
        self.assertIn("if (/^\\/login(\\/|$)/.test(loc.pathname)) { inkluagentZurAnmeldung(); return; }", g)
        self.assertIn("document.activeElement === rahmen", g, "Link im Rahmen: Fokus auf die neue Seite")

    def test_vergroessern_ein_zustand_alles_abgeleitet(self):
        bau = funktion(self.js, "inkluagentArbeitsfensterBauen")
        self.assertIn('id="arbeitsGross" class="btn btn-secondary" aria-pressed="false">\' + t(\'Arbeitsfenster vergrößern\')', bau)
        g = funktion(self.js, "inkluagentFensterGross")
        self.assertIn("knopf.setAttribute('aria-pressed', an ? 'true' : 'false');", g)
        self.assertIn("inkluagentUeberlagerungAnwenden();", g)
        u = funktion(self.js, "inkluagentUeberlagerungAnwenden")
        self.assertIn("k === ansage", u, "die Ansage-Region bleibt hoerbar (B3)")
        self.assertIn("f.setAttribute('role', 'dialog'); f.setAttribute('aria-modal', 'true');", u)
        e = funktion(self.js, "inkluagentUeberlagerungEinrichten")
        self.assertIn("mq.addEventListener('change', inkluagentUeberlagerungAnwenden)", e, "B2/K8: Breitenwechsel")
        self.assertIn("e.key === 'Escape'", e)
        self.assertIn("window.parent.inkluagentEscape()", self.dash, "B7: Escape aus der Seite im Rahmen")
        self.assertIn("html.arbeitsfenster-gross .arbeitsfenster { position: fixed; inset: 0;", lies(FRONTEND, "style.css"))

    def test_hinfuehren_und_links_oeffnen_im_fenster(self):
        # B1: EIN delegierter Klick-Handler fuer alle Links der Seite statt einzelner Handler an der Navigation
        self.assertIn("document.addEventListener('click', inkluagentLinkImFenster);", funktion(self.js, "inkluagentChatstil"))
        link = funktion(self.js, "inkluagentLinkImFenster")
        self.assertIn("if (e.defaultPrevented || e.button !== 0", link)
        self.assertIn("a.hasAttribute('download')", link)
        self.assertNotIn("imArbeitsfensterOeffnen", self.dash)
        w = funktion(self.js, "inkluagentWechseln")
        self.assertIn("if (frei && ziel.fenster) { inkluagentSeiteImFenster(ziel.url, ziel.name); return true; }", w)

    def test_im_rahmen_ganze_seite_fuer_projekte_sprache_anmeldung_zahlung(self):
        g = funktion(self.dash, "gehoertGanzeSeite")
        for teil in ("app|dashboard|set-language|login|logout|register", "datensicherheit|impressum-app", "ueber-uns", "preise"):
            self.assertIn(teil, g)
        self.assertIn("ganzeSeite(data.weiter);", lies(BACKEND, "templates", "gespraech.html"))
        self.assertIn("ganzeSeite('/app?projekt='", lies(BACKEND, "templates", "projekt-neu.html"))
        self.assertEqual(lies(BACKEND, "templates", "abo.html").count("ganzeSeite(d.url);"), 3, "Zahlung nie im Rahmen")
        self.assertIn("window.top.location.reload();", lies(BACKEND, "templates", "einstellungen.html"), "Ansicht im Fenster gewechselt")


if __name__ == "__main__":
    unittest.main()
