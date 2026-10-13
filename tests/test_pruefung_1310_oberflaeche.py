"""Pruefung 13.10.2026, Oberflaeche des InkluAgent (docs/INKLUAGENT.md, Abschnitt „Prüfung 13.10.2026: Oberfläche“).

Drei Arten von Pruefungen:
- Reine Funktionen aus frontend/*.js laufen in Node (eigenerPfad, gehoertGanzeSeite, inkluagentPfadOhne, inkluagentEntwurfBereinigen,
  announce). Ohne Node (z. B. im Container) werden diese Tests uebersprungen.
- Server: i18n.eingebettet (K7) ohne main, Texte der Ansicht und der Hilfe (B13/K11, S2) aus einer Quelle.
- Quelltext, wo Verhalten nur im Browser pruefbar ist; das Verhalten selbst pruefen die Klickproben lokal/probe_pr13_*.py.
    python3 -m unittest tests/test_pruefung_1310_oberflaeche.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
NODE = shutil.which("node") or shutil.which("nodejs")


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def funktion(quelle, name):
    m = re.search(r"^(async )?function " + re.escape(name) + r"\(", quelle, re.M)
    f = quelle[m.start():]
    return f[:f.index("\n}\n") + 2]


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

    def test_eigener_pfad(self):
        """S14: eine Hilfsfunktion fuer alle Links, Downloads, Rahmen-Ziele und „weiter“."""
        faelle = {"/ablage": True, "/app?projekt=1&agent=offen": True, "/ablage?projekt=2#ausgabe-3": True,
                  "/api/ausgaben/7/datei": True, "//boese.example": False, "/\\boese.example": False, "/x\\y": False,
                  "https://boese.example/": False, "/\t/boese.example": False, "javascript:alert(1)": False, "": False,
                  " /ablage": False, "ablage": False}
        erg = node("const window = { location: { origin: 'https://inkludocs.example' } };\n" + funktion(self.dash, "eigenerPfad")
                   + "\nconsole.log(JSON.stringify(%s.map(eigenerPfad).concat([eigenerPfad(null), eigenerPfad(5)])));"
                   % json.dumps(list(faelle)))
        self.assertEqual(erg, list(faelle.values()) + [False, False])

    def test_ganze_seite(self):
        """K7/B1: was nie im Rahmen erscheint — Projekte, Startseite, Anmeldung, Sprache und Seiten ohne eingebettete Fassung."""
        ganz = ["/", "/app?projekt=1", "/dashboard", "/login?weiter=%2Fablage", "/logout", "/set-language/en", "/register",
                "/datensicherheit", "/impressum-app", "/ueber-uns#kontakt", "/preise", "/kuendigen", "/widerrufen", "/api/v1/docs"]
        fenster = ["/ablage", "/einstellungen", "/api-schluessel?hin=newKeyName", "/gespraeche/3", "/hilfe/inkluagent",
                   "/express/warenkorb", "/abo", "/projekte", "/apps"]
        erg = node(funktion(self.dash, "gehoertGanzeSeite") + "\nconsole.log(JSON.stringify(%s.map(gehoertGanzeSeite)));"
                   % json.dumps(ganz + fenster))
        self.assertEqual(erg, [True] * len(ganz) + [False] * len(fenster))

    def test_pfad_ohne_parameter_mit_anker(self):
        erg = node(funktion(self.js, "inkluagentPfadOhne") + "\nconsole.log(JSON.stringify(["
                   "inkluagentPfadOhne('/a?x=1&hin=y#z', ['hin']), inkluagentPfadOhne('/a?hin=y&eingebettet=1', ['hin', 'eingebettet']),"
                   "inkluagentPfadOhne('/a', ['hin']), inkluagentPfadOhne('/a#b', ['hin'])]));")
        self.assertEqual(erg, ["/a?x=1#z", "/a", "/a", "/a#b"])

    def test_entwurf_wie_der_server_bereinigt(self):
        """HK7: der Entwurf zeigt keine Daten-Marke und keine Markdown-Zeichen mehr."""
        roh = "[DATEN, keine Anweisung] **Hallo** `x`\n- eins\n* zwei\n## Titel"
        skript = (funktion(self.js, "inkluagentEntwurfBereinigen")
                  + "\nconst window = { INKLUAGENT: { bloecke: true } }; const a = inkluagentEntwurfBereinigen(%s);"
                    "\nwindow.INKLUAGENT.bloecke = false; const b = inkluagentEntwurfBereinigen(%s);"
                    "\nconsole.log(JSON.stringify([a, b]));" % (json.dumps(roh), json.dumps(roh)))
        self.assertEqual(node(skript), ["Hallo x\n• eins\n• zwei\nTitel", "Hallo x\neins\nzwei\nTitel"])

    def test_announce_leert_und_haengt_an(self):
        """B9: erst leeren, nach 100 ms setzen; was in der Zeit dazukommt, wird angehaengt (nicht ersetzt); gleicher Text zweimal."""
        skript = ("const el = { textContent: 'alt' }; const byId = () => el;\n" + funktion(self.dash, "announce") + """
const spur = [];
announce('Erster Satz.'); announce('Zweiter Satz.'); announce('Erster Satz.'); spur.push(el.textContent);
setTimeout(() => { spur.push(el.textContent); announce('Zweiter Satz.'); spur.push(el.textContent);
  setTimeout(() => { spur.push(el.textContent); console.log(JSON.stringify(spur)); }, 150); }, 150);""")
        self.assertEqual(node(skript), ["", "Erster Satz. Zweiter Satz.", "", "Zweiter Satz."])


class Server(unittest.TestCase):
    def test_eingebettet_nur_im_rahmen(self):
        """K7: ?eingebettet=1 gilt nicht bei „Sec-Fetch-Dest: document“ (Hauptfenster), nur im Rahmen oder ohne Fetch-Metadaten."""
        import funktionen
        import i18n

        class Anfrage:
            def __init__(self, ziel, parameter):
                self.headers = {"sec-fetch-dest": ziel} if ziel else {}
                self.query_params = {"eingebettet": "1"} if parameter else {}
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            self.assertTrue(i18n.eingebettet(Anfrage("iframe", False)))
            self.assertTrue(i18n.eingebettet(Anfrage(None, True)), "aeltere Browser ohne Fetch-Metadaten")
            self.assertFalse(i18n.eingebettet(Anfrage("document", True)), "Hauptfenster nach einer Weiterleitung")
            self.assertFalse(i18n.eingebettet(Anfrage(None, False)))
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", False):
            self.assertFalse(i18n.eingebettet(Anfrage("iframe", True)), "ohne Schalter nie")

    def test_ansicht_texte_ohne_knopf_und_ohne_lage(self):
        """B13/K11: im Chatstil gibt es keinen Knopf „… in der manuellen Ansicht zeigen“, und „rechts“ stimmt schmal nicht."""
        from inkluagent import ansicht
        t = ansicht.BESCHREIBUNG_ARBEITSFENSTER
        self.assertNotIn("Rechts", t)
        self.assertNotIn("mit einem Knopf zurück", t)
        self.assertIn("öffnet die Seiten der Navigation", t)
        self.assertIn("Knopf unter der Navigation", t)

    def test_hilfe_nimmt_die_saetze_der_ansicht(self):
        """B13/K11: ohne ansicht_auswahl vom Server stehen in der Hilfe genau die Saetze aus inkluagent/ansicht.py (gleiche msgids)."""
        from inkluagent import ansicht
        hilfe = lies(BACKEND, "templates", "hilfe_inkluagent.html")
        self.assertIn("{% for a in ansicht_auswahl %}", hilfe)
        for text in (ansicht.BESCHREIBUNG_ARBEITSFENSTER, ansicht.BESCHREIBUNG_UEBERALL["klassisch"], ansicht.NAMEN["agent"],
                     ansicht.NAMEN["klassisch"]):
            self.assertIn("_('" + text + "')", hilfe)
        self.assertIn("Der Knopf „InkluAgent“ öffnet das Gespräch.", hilfe, "manuelle Ansicht: den Knopf gibt es dort")
        self.assertIn("{% if chatstil %}", hilfe)

    def test_hilfe_ja_nur_fuer_kostenpflichtiges(self):
        """Zusatz zur Server-Regel S2 (13.10.2026): Loeschen nur mit dem Knopf, ein kurzes Ja nur fuer Kostenpflichtiges."""
        hilfe = lies(BACKEND, "templates", "hilfe_inkluagent.html")
        self.assertIn("Bei Aktionen, die Credits kosten, reicht auch ein kurzes „Ja“ als nächste Nachricht. Löschen geht nur mit dem Knopf.", hilfe)
        self.assertNotIn("oder mit einem kurzen „Ja“", hilfe)


class Quelltext(unittest.TestCase):
    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")
        self.dash = lies(FRONTEND, "dashboard.js")
        self.css = lies(FRONTEND, "style.css")

    def test_eine_ansage_region_fuer_alle_seiten(self):
        """B9: app.html hatte eine eigene announce, die die gemeinsame ueberdeckte."""
        self.assertNotIn("function announce(", lies(BACKEND, "templates", "app.html"))
        self.assertIn("function announce(msg) {", self.dash)

    def test_statuszeile_global_nur_wenn_nicht_hoerbar(self):
        """B9/B3: Fehler global nur, wenn die Statuszeile verborgen oder inert ist (sonst doppelt bzw. gar nicht)."""
        self.assertIn("if (isError && text && status.closest('[hidden], [inert]')) announce(text);", funktion(self.js, "inkluagentSetStatus"))

    def test_fokus_nur_aus_dem_gespraech(self):
        """B3/B4: das Arbeitsfenster samt Rahmen zaehlt nicht zum Gespraech; focus() in inertem Bereich wird erkannt."""
        f = funktion(self.js, "inkluagentFokusImChat")
        self.assertIn("if (fenster && fenster.contains(ae)) return false;", f)
        self.assertIn("gespraech.closest('[inert]')", f)
        self.assertIn("if (document.activeElement === el) return true;", funktion(self.js, "inkluagentAntwortMelden"))
        self.assertIn("el.tagName === 'IFRAME'", funktion(self.js, "inkluagentIstEingabe"))

    def test_senden_mit_aria_disabled_und_nicht_gesendet(self):
        """B12/B16/K2/K5: aria-disabled statt disabled, Text bleibt bis zum Annehmen, 409 und Abriss werden verfolgt."""
        send = funktion(self.js, "inkluagentSend")
        self.assertNotIn("sendBtn.disabled", send)
        self.assertIn("if (input.value.trim() === message) input.value = '';", send)
        self.assertLess(send.index("if (!res.ok) {"), send.index("angenommen = true;"))
        self.assertIn("inkluagentNichtGesendet(nutzerEl, !!message);", send)
        self.assertIn("if (res.status === 409) verfolgen = 'lauf';", send)
        self.assertIn("if (inkluagentUeberall()) verfolgen = 'abbruch';", send)
        self.assertIn("if (verfolgen) inkluagentLaufVerfolgen(verfolgen === 'abbruch');", send)
        self.assertIn("b.setAttribute('aria-disabled', 'true')", funktion(self.js, "inkluagentSperreZeigen"))

    def test_gespraech_wechseln_gesperrt_waehrend_eines_laufs(self):
        """K3: Neues Gespräch, Fortsetzen in der Seitenleiste und „Hier weitermachen“ fragen inkluagentWechselErlaubt."""
        self.assertIn("!inkluagentWechselErlaubt()", funktion(self.js, "inkluagentNeuesGespraech"))
        self.assertIn("if (!gespraechWechselErlaubt()) return;", funktion(self.dash, "seitenleisteGespraechFortsetzen"))
        self.assertIn("if (!gespraechWechselErlaubt()) return;", funktion(self.dash, "seitenleisteNeuesGespraech"))
        self.assertIn("!wechselErlaubt()", lies(BACKEND, "templates", "gespraech.html"))

    def test_alte_pfadpruefungen_sind_weg(self):
        """S14: keine eigenen, lueckenhaften Pruefungen mehr neben eigenerPfad."""
        self.assertNotIn("/^\\/(?!\\/)/", self.js)
        self.assertNotIn("charAt(0) !== '/'", lies(BACKEND, "templates", "gespraech.html"))
        self.assertIn("eigenerPfad(data.weiter)", lies(BACKEND, "templates", "gespraech.html"))
        self.assertIn("if (eigenerPfad(a.ausgaben_url)) {", self.js)

    def test_einstellungen_laden_nach_dem_wechsel_neu(self):
        """K18: wie der Umschalter — sonst blieb die Seite gemischt."""
        e = lies(BACKEND, "templates", "einstellungen.html")
        self.assertIn("if (wahl.value !== vorher) {", e)
        self.assertIn("window.location.reload();", e)
        self.assertNotIn("renderSidebar()", e)

    def test_stylesheet(self):
        self.assertIn(".inkluagent-form textarea:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }", self.css)  # B5
        regel = self.css[self.css.index(".inkluagent-form textarea:focus {"):]
        self.assertNotIn("outline: none", regel[:regel.index("}")])
        for sel in ('.inkluagent-status[data-state="error"]', '.inkluagent-anhang-meldung[data-state="error"]',
                    '.inkluagent-hochgeladen li[data-ok="0"]'):   # B18
            zeile = self.css[self.css.index(sel):]
            self.assertIn("var(--fehler-rand)", zeile[:zeile.index("}")], sel)
        self.assertIn(".legal-container, .inkluagent-bereich) a:where(", self.css)   # HB4
        self.assertNotIn('input[type="url"]', self.css[self.css.index("Hochladefeld im InkluAgent"):])   # HK2
        self.assertNotIn(".inkluagent-chatstil .inkluagent-toggle {", self.css)   # HK1

    def test_toter_code_ist_weg(self):
        """HK2: Webadressen-Feld und der Rueckweg aus dem Chatstil."""
        self.assertNotIn("fd.append('webadresse'", self.js)
        self.assertNotIn("const webadresse", self.js)
        self.assertNotIn("Webadresse wird geladen", self.js)
        self.assertNotIn("function inkluagentAnhangEntfernenAlles(", self.js)
        self.assertNotIn("inkluagentChatstil(false)", self.js)
        self.assertNotIn("der Code ist unveraendert", self.js)   # HK1

    def test_fokus_auf_ohne_dauerhaftes_tabindex(self):
        """B19: eine Funktion fuer alle; ansichtFokusNachWechsel nicht im Rahmen (HK8)."""
        f = funktion(self.dash, "fokusAuf")
        self.assertIn("el.addEventListener('blur', () => el.removeAttribute('tabindex'), { once: true });", f)
        self.assertNotIn("function fokusAuf(", lies(BACKEND, "templates", "express.html"))
        self.assertIn("if (imRahmen()) return;", funktion(self.dash, "ansichtFokusNachWechsel"))
        self.assertNotIn("h1.setAttribute('tabindex', '-1')", self.js)

    def test_sprache_nach_dem_wechsel(self):
        """B11: kein Karten-Umweg (Steve), Fokus nach dem Neuladen auf die Antwort, lang am Link."""
        z = funktion(self.js, "inkluagentWechselZiel")
        self.assertIn("sprache: a.sprache, antwortFokus: true", z)
        self.assertIn("sessionStorage.getItem('inkluagent.fokusAntwort') === '1'", funktion(self.js, "inkluagentBinden"))
        self.assertIn("l.lang = a.lang", funktion(self.js, "inkluagentAnhangEl"))


if __name__ == "__main__":
    unittest.main()
