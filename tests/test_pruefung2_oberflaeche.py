"""Pruefung 13.10.2026, Runde 2: Oberflaeche des InkluAgent (docs/INKLUAGENT.md, Abschnitt „Prüfung 13.10.2026, Runde 2: Oberfläche“).

Drei Arten von Pruefungen, wie in tests/test_pruefung_1310_oberflaeche.py:
- Reine Funktionen aus frontend/*.js in Node: fensterSeite (S-N6/A-N6, Liste des Servers), hinZieleListe (S-N9), announce mit
  Mindestabstand und Verwerfen (A-N8), Kennung des Hochlade-Stapels (K12), „Nicht gesendet:“ in der Statuszeile (A-N4), lang am
  Nachrichtentext (B11, Punkt 4). Ohne Node (z. B. im Container) uebersprungen.
- Server und Vorlagen: eingebettete Fassung der oeffentlichen Seiten (A-N6), lang auf der Gespraechsseite, Hilfe nennt die Ausnahmen.
- Quelltext, wo Verhalten nur im Browser pruefbar ist; das Verhalten selbst pruefen die Klickproben lokal/probe_r2_*.py.
    python3 -m unittest tests/test_pruefung2_oberflaeche.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
NODE = shutil.which("node") or shutil.which("nodejs")

# Die Liste aus der Schnittstelle des Servers (main._inkluagent_schalter, Runde 2)
FENSTER_SEITEN = ["/ablage", "/abo", "/api-schluessel", "/einstellungen", "/express", "/gespraeche", "/geteilte-projekte", "/hilfe",
                  "/konto", "/kuendigen", "/preise", "/projekt-neu", "/projekte", "/prompts", "/stammdaten", "/team", "/ueber-uns",
                  "/vorlagen", "/widerrufen"]


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def funktion(quelle, name):
    m = re.search(r"^(async )?function " + re.escape(name) + r"\(", quelle, re.M)
    f = quelle[m.start():]
    return f[:f.index("\n}\n") + 2]


def ohne_kommentare(css):
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def node(skript):
    """Fuehrt JavaScript in Node aus und liefert die letzte Zeile der Ausgabe als JSON."""
    r = subprocess.run([NODE, "-e", skript], capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise AssertionError(r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "Node fehlt (z. B. im Container) — die Funktionen pruefen dann nur die Klickproben")
class ReineFunktionen(unittest.TestCase):
    def setUp(self):
        self.dash = lies(FRONTEND, "dashboard.js")
        self.js = lies(FRONTEND, "inkluagent.js")

    def _fenster(self, fenster_js, pfade):
        teile = "\n".join(funktion(self.dash, n) for n in ("imRahmen", "fensterSeitenListe", "fensterSeite", "gehoertGanzeSeite"))
        return node(fenster_js + "\n" + teile + "\nconsole.log(JSON.stringify(%s.map(fensterSeite)));" % json.dumps(pfade))

    def test_fenster_seite_nach_der_liste_des_servers(self):
        """S-N6/A-N6: gleich oder Praefix + „/“; Abfrage und Anker zaehlen nicht; „/ablagex“ ist nicht „/ablage“."""
        ja = ["/ablage", "/ablage?projekt=1#ausgabe-3", "/hilfe/inkluagent", "/express/warenkorb", "/gespraeche/3", "/ueber-uns#kontakt",
              "/preise", "/kuendigen", "/widerrufen", "/api-schluessel?hin=newKeyName"]
        nein = ["/", "/app?projekt=1", "/dashboard", "/ablagex", "/abonnement", "/login?weiter=%2Fablage", "/datensicherheit",
                "/impressum", "/datenschutz", "/verwaltung/kunden", "/kontakt", "/set-language/en", "/api/v1/docs", "/register"]
        w = ("const window = { location: { origin: 'https://inkludocs.example' }, EINGEBETTET: false, "
             "INKLUAGENT: { fensterSeiten: %s.concat(['ohne-schraegstrich', 5]) } }; window.top = window;" % json.dumps(FENSTER_SEITEN))
        self.assertEqual(self._fenster(w, ja + nein), [True] * len(ja) + [False] * len(nein))

    def test_fenster_seite_ohne_liste_die_bisherige_regel(self):
        """Server ohne Liste: alles ausser gehoertGanzeSeite — seit A-N6 mit Über uns, Preise, Kündigen, Widerrufen."""
        w = "const window = { location: { origin: 'https://inkludocs.example' }, EINGEBETTET: false, INKLUAGENT: {} }; window.top = window;"
        self.assertEqual(self._fenster(w, ["/ueber-uns", "/preise", "/kuendigen", "/widerrufen", "/verwaltung/kunden", "/app?projekt=2", "/impressum"]),
                         [True, True, True, True, True, False, False])

    def test_eingebettete_oeffentliche_seite_nimmt_die_liste_des_hauptfensters(self):
        """A-N6: base_oeffentlich.html hat kein window.INKLUAGENT — im Rahmen gilt die Liste der Elternseite."""
        w = ("const eltern = { INKLUAGENT: { fensterSeiten: %s } };"
             "const window = { location: { origin: 'https://inkludocs.example' }, EINGEBETTET: true, parent: eltern, top: eltern };"
             % json.dumps(FENSTER_SEITEN))
        self.assertEqual(self._fenster(w, ["/preise", "/kontakt", "/dashboard", "/verwaltung/kunden"]), [True, False, False, False])

    def test_hin_ziele_fail_closed(self):
        """S-N9: ohne Liste (oder kein Array) keine; eingebettet die des Hauptfensters; die Pruefung nimmt nur ids aus der Liste."""
        teile = funktion(self.dash, "imRahmen") + funktion(self.dash, "hinZieleListe")
        skript = ("let window;\n" + teile + """
const eltern = { INKLUAGENT: { hinZiele: ['ueberschrift', 'kontakt'] } };
const erg = [];
window = { INKLUAGENT: {}, EINGEBETTET: false }; window.top = window; erg.push(hinZieleListe());
window = { INKLUAGENT: { hinZiele: 'kontakt' }, EINGEBETTET: false }; window.top = window; erg.push(hinZieleListe());
window = { INKLUAGENT: { hinZiele: ['newKeyName'] }, EINGEBETTET: false }; window.top = window; erg.push(hinZieleListe());
window = { EINGEBETTET: true, parent: eltern, top: eltern }; erg.push(hinZieleListe());
console.log(JSON.stringify(erg));""")
        self.assertEqual(node(skript), [None, None, ["newKeyName"], ["ueberschrift", "kontakt"]])
        h = funktion(self.dash, "hinfuehrenBeimLaden")
        self.assertIn("if (!liste || liste.indexOf(ziel) < 0) return;", h)
        self.assertNotIn("ziel !== 'ueberschrift'", h, "auch „ueberschrift“ nur aus der Liste (sie steht darin)")

    def _announce(self):
        a = self.dash[self.dash.index("function announce(msg, opts) {"):]
        return a[:a.index("};\n", a.index("announce.verwerfen")) + 3]

    def test_announce_mindestabstand_und_verwerfen(self):
        """A-N8: eine Ansage leert die Region erst nach dem Mindestabstand (hier 400 ms statt 1 s), was dazukommt, haengt an;
        verwerfen streicht nur wartende Zwischenstaende (verwerfbar), ein Fehler bleibt."""
        skript = ("const el = { textContent: '' }; const byId = () => el;\n" + self._announce() + """
announce.ABSTAND_MS = 400;
const spur = [];
announce('A.');
setTimeout(() => { spur.push(el.textContent); announce('B.'); spur.push(el.textContent); }, 150);
setTimeout(() => { spur.push(el.textContent); }, 300);
setTimeout(() => { spur.push(el.textContent); }, 600);
setTimeout(() => { announce('Schritt.', { verwerfbar: true }); announce('Fehler.'); announce.verwerfen(); }, 650);
setTimeout(() => { spur.push(el.textContent); announce('Nur Schritt.', { verwerfbar: true }); announce.verwerfen(); }, 1150);
setTimeout(() => { spur.push(el.textContent); console.log(JSON.stringify(spur)); }, 1900);""")
        # 150 ms: „A.“ steht und wird fuer „B.“ NICHT sofort geleert (vorher: ''); 600 ms: „B.“; 1150 ms: nur „Fehler.“;
        # 1900 ms: der verworfene Schritt kam nie (die Region wurde geleert und blieb leer)
        self.assertEqual(node(skript), ["A.", "A.", "A.", "B.", "Fehler.", ""])

    def test_hochladen_kennung_je_stapel(self):
        """K12: dieselben Dateien = dieselbe Kennung (bis zur Annahme), ein anderer Stapel = neue; Form [A-Za-z0-9_-], hoechstens 64."""
        skript = ("const window = { crypto: require('crypto').webcrypto };\n" + funktion(self.js, "inkluagentHochladenKennung")
                  + funktion(self.js, "inkluagentNeueKennung") + """
const a = {}, b = {}, c = {};
const k1 = inkluagentHochladenKennung([a, b]), k2 = inkluagentHochladenKennung([a, b]), k3 = inkluagentHochladenKennung([a, b, c]);
const k4 = inkluagentHochladenKennung([a, b, c]), k5 = inkluagentHochladenKennung([b, a, c]);
window._inkluagentStapel = null;   // angenommen (inkluagentSend)
const k6 = inkluagentHochladenKennung([b, a, c]);
console.log(JSON.stringify([k1 === k2, k3 !== k1, k4 === k3, k5 !== k4, k6 !== k5,
  [k1, k3, k5, k6].every(k => /^[A-Za-z0-9_-]{1,64}$/.test(k))]));""")
        self.assertEqual(node(skript), [True] * 6)
        send = funktion(self.js, "inkluagentSend")
        self.assertIn("inkluagentHochladen(projectId, dateien, inkluagentHochladenKennung(dateien))", send)
        self.assertLess(send.index("angenommen = true;"), send.index("window._inkluagentStapel = null;"))
        self.assertIn("if (kennung) fd.append('hochladen_id', kennung);", funktion(self.js, "inkluagentHochladen"))

    def test_nicht_gesendet_in_der_statuszeile(self):
        """A-N4: die Statuszeile beginnt mit „Nicht gesendet:“, nennt den Grund und endet mit dem Satz zum Eingabefeld."""
        skript = ("""const document = { createElement: () => ({}) };
function t(s, p) { let o = s; if (p) Object.keys(p).forEach(k => { o = o.replace('{' + k + '}', p[k]); }); return o; }
const spur = []; function inkluagentSetStatus(text, fehler) { spur.push([text, fehler]); }
""" + funktion(self.js, "inkluagentNichtGesendet") + """
const blase = { querySelector: () => null, classList: { add() {} }, appendChild() {} };
inkluagentNichtGesendet(blase, true, 'Tagesgrenze erreicht (fiktiv)');
inkluagentNichtGesendet(null, false, 'Verbindungsfehler. Bitte erneut versuchen.');
console.log(JSON.stringify(spur));""")
        self.assertEqual(node(skript), [
            ["Nicht gesendet: Tagesgrenze erreicht (fiktiv). Dein Text steht noch im Eingabefeld.", True],
            ["Nicht gesendet: Verbindungsfehler. Bitte erneut versuchen.", True]])

    def test_sprache_am_nachrichtentext(self):
        """B11, Punkt 4: lang nur mit gueltigem Sprachkuerzel."""
        skript = (funktion(self.js, "inkluagentSpracheSetzen") + """
const erg = ['de', 'en-GB', 'de" onmouseover="x', '', undefined, 5].map(s => { const el = {}; inkluagentSpracheSetzen(el, s); return el.lang || null; });
console.log(JSON.stringify(erg));""")
        self.assertEqual(node(skript), ["de", "en-GB", None, None, None, None])


class Vorlagen(unittest.TestCase):
    """A-N6 am Server: die oeffentlichen Seiten im Rahmen ohne Seitenleiste und Fusszeile, mit „Zurück zum Gespräch“."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pruefung2_oberflaeche_")
        os.environ.setdefault("INKLUDOCS_DB", os.path.join(cls.tmp, "test.db"))
        try:
            import database
            database.init_db()
            import main
            from fastapi.testclient import TestClient
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.c = main, TestClient(main.app)

    def test_oeffentliche_seiten_eingebettet(self):
        import funktionen
        for pfad in ("/ueber-uns", "/preise", "/kuendigen", "/widerrufen"):
            with self.subTest(pfad=pfad):
                with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
                    normal = self.c.get(pfad).text
                    rahmen = self.c.get(pfad, headers={"Sec-Fetch-Dest": "iframe"}).text
                    haupt = self.c.get(pfad + "?eingebettet=1", headers={"Sec-Fetch-Dest": "document"}).text
                ohne = self.c.get(pfad, headers={"Sec-Fetch-Dest": "iframe"}).text
                for html in (normal, haupt, ohne):
                    self.assertIn('id="appSidebar"', html)
                    self.assertIn('class="dash-footer"', html)
                    self.assertIn("window.EINGEBETTET = false;", html)
                    self.assertNotIn('class="eingebettet"', html)
                self.assertIn('<body class="eingebettet">', rahmen)
                self.assertNotIn('id="appSidebar"', rahmen)
                self.assertNotIn('class="dash-footer"', rahmen)
                self.assertIn('id="zumGespraech" class="dash-skip"', rahmen)
                self.assertNotIn('<a href="#main" class="dash-skip">', rahmen, "nur „Zurück zum Gespräch“ (window.I18N enthaelt alle Texte)")
                self.assertIn("window.EINGEBETTET = true;", rahmen)
                self.assertIn('<main id="main"', rahmen)
                self.assertRegex(rahmen, r"<h1>[^<]+</h1>", "die Seite selbst ist da")

    def test_gespraechsseite_sprache_am_text(self):
        """B11, Punkt 4: die Gespraechsseite setzt lang aus n.sprache am Text — nur Sprachen der Oberflaeche, nicht an der Ueberschrift."""
        import i18n
        nachrichten = [{"role": r, "content": c, "project_id": None, "marke": False, "projekt_name": "", "projekt_geloescht": False,
                        "sprache": s, "bloecke": [{"art": "p", "zeilen": [c], "start": 1}]}
                       for r, c, s in (("user", "Frage (fiktiv)", "de"), ("assistant", "Antwort (fiktiv)", "en"),
                                       ("assistant", "Ohne (fiktiv)", None), ("assistant", "Boese (fiktiv)", 'de" onmouseover="x'))]
        html = self.main.templates.env.get_template("gespraech.html").render(
            _=i18n.get_gettext("fr"), current_lang="fr", supported_languages=i18n.SUPPORTED_LANGUAGES, is_staging=False, asset_version="1",
            i18n_json="{}", eingebettet=False, inkluagent_schalter={}, oberflaeche=None, toene=None,
            gespraech={"id": 1, "titel": "Frage (fiktiv)", "datum": "", "aktuell": False}, nachrichten=nachrichten)
        inhalte = re.findall(r'<div class="inkluagent-message-content"( lang="[^"]*")?>', html)
        self.assertEqual(inhalte, [' lang="de"', ' lang="en"', "", ""])
        self.assertNotIn("onmouseover", html)
        self.assertEqual(len(re.findall(r'<h3 class="inkluagent-message-role">', html)), 4, "Absender ohne lang")


class Quelltext(unittest.TestCase):
    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")
        self.dash = lies(FRONTEND, "dashboard.js")
        self.css = lies(FRONTEND, "style.css")

    def test_ueberlagerung_ohne_aria_modal_mit_ansage(self):
        """A-N1: keine aria-modal mehr (nahm die Ansage-Region aus dem Baum); A-N2: Ansage beim Oeffnen, Fokus bleibt."""
        self.assertNotIn("'aria-modal'", self.js)
        u = funktion(self.js, "inkluagentUeberlagerungAnwenden")
        self.assertIn("if (ueber) f.setAttribute('role', 'dialog'); else f.removeAttribute('role');", u)
        self.assertIn("if (!war) announce(t('Das Arbeitsfenster liegt jetzt über dem Gespräch.", u)
        self.assertIn("k === ansage", u, "die Ansage-Region bleibt ausserhalb von inert")

    def test_statuszeile_nie_display_none(self):
        """A-N3: leer ohne Hoehe und Abstand, aber im Baum."""
        css = ohne_kommentare(self.css)
        self.assertNotRegex(css, r"#inkluagentStatus:empty\s*\{[^}]*display:\s*none")
        self.assertIn(".inkluagent-meldungen > :empty { margin: 0; }", css)
        self.assertNotIn("gap: 0.35rem", css[css.index(".inkluagent-meldungen {"):][:120])

    def test_fokus_bleibt_nach_nicht_gesendet(self):
        """A-N4: ins Feld nur, wenn der Fokus auf body gefallen ist."""
        send = funktion(self.js, "inkluagentSend")
        self.assertIn("else if ((!ae || ae === document.body || ae === document.documentElement) && inkluagentFokusImChat()) { input.focus(); }", send)
        self.assertNotIn("document.activeElement !== input) { input.focus(); }", send)
        self.assertIn("inkluagentNichtGesendet(nutzerEl, !!message, t('Verbindungsfehler. Bitte erneut versuchen.'));", send)

    def test_gesperrt_ohne_opacity(self):
        """A-N5: gedaempfte Farben, der Fokusring bleibt voll."""
        css = ohne_kommentare(self.css)
        regel = css[css.index('.btn[aria-disabled="true"], .btn[aria-disabled="true"]:hover {'):]
        regel = regel[:regel.index("}")]
        self.assertNotIn("opacity", regel)
        self.assertIn("border: 2px dashed", regel, "gesperrt nicht nur an der Farbe erkennbar")
        self.assertNotRegex(css, r'aria-disabled="true"\][^{]*\{[^}]*opacity')

    def test_seitenleiste_hinweis_hidden(self):
        """A-N7."""
        b = funktion(self.dash, "seitenleisteGespraecheBauen")
        self.assertIn("hinweis.hidden = true;", b)
        self.assertNotIn("hinweis.className = 'visually-hidden'", b)

    def test_anker_wie_hin(self):
        """A-N9: Sprungmarke zaehlt wie ?hin=; die Ablage nutzt fokusAuf."""
        self.assertIn("hin: /[?&]hin=/.test(pfad) || /#./.test(pfad)", funktion(self.js, "inkluagentSeiteImFenster"))
        fokus = funktion(self.js, "inkluagentRahmenFokus")
        self.assertIn("doc.getElementById(decodeURIComponent(anker.slice(1)))", fokus)
        ablage = lies(BACKEND, "templates", "ablage.html")
        self.assertIn("fokusAuf(ziel, { preventScroll: true });", ablage)
        self.assertNotIn("ziel.setAttribute('tabindex', '-1')", ablage)

    def test_fehlerseite_bleibt_im_fenster(self):
        """K-N1: erst nach dem Laden merken; Fehlerantwort schliesst im Fenster, statt wegzunavigieren."""
        self.assertNotIn("sessionStorage", funktion(self.js, "inkluagentSeiteZeigen"))
        self.assertIn("sessionStorage.setItem('inkluagent.fenster'", funktion(self.js, "inkluagentSeiteMerken"))
        g = funktion(self.js, "inkluagentRahmenGeladen")
        erfolg = g.index("if (eigene && eingebettet === true && fensterSeite(loc.pathname)) {")
        self.assertGreater(g.index("inkluagentSeiteMerken();"), erfolg)
        self.assertEqual(g.count("inkluagentSeiteMerken();"), 1)
        self.assertLess(g.index("inkluagentSeiteZustandZu();\n        window.location.href = pfad + loc.hash;"), g.rindex("inkluagentSeiteNichtGeladen(rahmen);"))
        self.assertNotIn("window.location.href", funktion(self.js, "inkluagentSeiteNichtGeladen"))
        self.assertIn("inkluagentSeiteZustandZu();", funktion(self.js, "inkluagentSeiteNichtGeladen"))

    def test_grenzen_hinweis_im_fluss(self):
        """HN1: der Hinweis steht in der Beschreibung, nicht im angehefteten Formular; der Knopf verweist weiter darauf."""
        b = funktion(self.js, "inkluagentAnhangBauen")
        self.assertIn("[datei, liste].forEach(el => form.appendChild(el));", b)
        self.assertIn("beschreibung.insertBefore(hinweis, beschreibung.firstChild)", b)
        self.assertIn("knopf.setAttribute('aria-describedby', 'inkluagentAnhangHinweis');", b)

    def test_fokus_merker_am_rueckweg(self):
        """HN3: „Zurück zu …“ setzt inkluagent.fokusAntwort wie „Sprache wechseln: …“ (eine Funktion)."""
        a = funktion(self.js, "inkluagentAnhangEl")
        self.assertIn("if (/^\\/set-language\\//.test(a.url)) l.addEventListener('click', inkluagentFokusAntwortMerken);", a)
        self.assertEqual(self.js.count("sessionStorage.setItem('inkluagent.fokusAntwort', '1')"), 1)

    def test_sprache_an_allen_nachrichten(self):
        """B11, Punkt 4: Konto-Verlauf, Projektverlauf, frueheres Gespraech, neue Antworten und die Gespraechsseite."""
        for aufruf in ("m.anhang, m.sprache);", "data.anhang, data.sprache);", "data.anhang || [], data.sprache);",
                       "inkluagentSpracheSetzen(c, m.sprache);", "inkluagentSpracheSetzen(contentEl, sprache);"):
            self.assertIn(aufruf, self.js)
        self.assertEqual(self.js.count("m.anhang, m.sprache);"), 3)   # dazu die angehaengte Rueckmeldung (14.10.2026)
        self.assertIn('{% if n.sprache and n.sprache in supported_languages %} lang="{{ n.sprache }}"{% endif %}',
                      lies(BACKEND, "templates", "gespraech.html"))

    def test_hilfe_nennt_alle_ausnahmen(self):
        """A-N6: die Hilfe nennt die Seiten, die als ganze Seite oeffnen, vollstaendig."""
        h = lies(BACKEND, "templates", "hilfe_inkluagent.html")
        self.assertNotIn("Projekte und die Startseite öffnen als ganze Seite.", h)
        for teil in ("deine Projekte und die Startseite", "der Wechsel der Sprache, Anmelden und Abmelden",
                     "die Rechtstexte: Impressum, Datenschutz, Datensicherheit, Nutzungsbedingungen, Widerrufsbelehrung und Auftragsverarbeitung",
                     "die API-Dokumentation"):
            self.assertIn(teil, h)

    def test_einstellungen_erst_vergleichen(self):
        """HN2: das Hauptfenster laedt nur bei geaenderter Ansicht neu."""
        e = lies(BACKEND, "templates", "einstellungen.html")
        self.assertLess(e.index("if (wahl.value !== vorher) {"), e.index("window.top"))
        self.assertNotIn("window.top.location.reload();", e)


if __name__ == "__main__":
    unittest.main()
