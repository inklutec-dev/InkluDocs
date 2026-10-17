"""Konzept InkluAgent, Schritt 1 (09.10.2026): das Skript des Chat-Bereichs steht in frontend/inkluagent.js statt in
backend/templates/app.html. Ohne Verhaltensaenderung: dieselben Funktionen, genau einmal definiert, vor dem Seitenskript
geladen, Texte weiter aus window.I18N (check_i18n.py findet sie). Laeuft ohne Server:
    python3 -m unittest tests/test_inkluagent_js.py"""
import os
import re
import sys
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
from inkluagent import skripte  # noqa: E402   die Teile des Agenten-Skripts (17.10.2026)

FUNKTIONEN = (
    "inkluagentSectionHtml", "inkluagentInit", "inkluagentInitLaden", "inkluagentOpen", "inkluagentClose",
    "inkluagentLoadHistory", "inkluagentIstEingabe", "inkluagentFokusImChat", "inkluagentLetzteAntwort",
    "inkluagentNeuMarkieren", "inkluagentAntwortMelden", "inkluagentKarteSetzen", "inkluagentAnsichtNachziehen",
    "inkluagentFokusMerken", "inkluagentFokusFinden", "inkluagentAktionenKarten", "inkluagentWerkzeugLabel",
    "inkluagentWerkzeugText", "inkluagentBestaetigen", "inkluagentAnhangEl", "inkluagentAppendMessage",
    "inkluagentSetStatus", "inkluagentSend",
)


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


class InkluagentJs(unittest.TestCase):
    """Seit 17.10.2026 in Teilen je Thema (backend/inkluagent/skripte.py): dieselben Regeln gelten fuer alle Teile zusammen."""

    def setUp(self):
        self.js = skripte.quelltext(FRONTEND)
        self.teile = {n: lies(FRONTEND, n) for n in skripte.TEILE}
        self.app = lies(BACKEND, "templates", "app.html")

    def test_funktionen_genau_einmal_und_nur_in_den_agenten_dateien(self):
        alle_js = {n: lies(FRONTEND, n) for n in os.listdir(FRONTEND) if n.endswith(".js") and ".bak" not in n}
        for f in FUNKTIONEN:
            with self.subTest(funktion=f):
                muster = re.compile(r"\b(?:async\s+)?function\s+" + f + r"\s*\(")
                self.assertEqual(len(muster.findall(self.js)), 1)
                self.assertEqual(len(muster.findall(self.app)), 0, "steht noch in app.html")
                for name, text in alle_js.items():
                    if name not in skripte.TEILE:
                        self.assertEqual(len(muster.findall(text)), 0, f"auch in {name}")

    def test_jede_funktion_der_teile_genau_einmal(self):
        # Klassische Skripte teilen sich den globalen Bereich: eine zweite Funktion gleichen Namens ueberschriebe still die erste
        alle_js = {n: lies(FRONTEND, n) for n in os.listdir(FRONTEND) if n.endswith(".js") and ".bak" not in n}
        namen = re.findall(r"^(?:async )?function (\w+)\(", self.js, re.M)
        self.assertGreater(len(namen), 150)
        doppelt = sorted({n for n in namen if namen.count(n) > 1})
        self.assertEqual(doppelt, [], "in zwei Teilen")
        for name, text in alle_js.items():
            if name in skripte.TEILE:
                continue
            fremd = set(re.findall(r"^(?:async )?function (\w+)\(", text, re.M)) & set(namen)
            self.assertEqual(fremd, set(), f"auch in {name}")

    def test_liste_der_teile(self):
        auf_platte = sorted(n for n in os.listdir(FRONTEND) if re.fullmatch(r"inkluagent(-[a-z]+)?\.js", n))
        self.assertEqual(auf_platte, sorted(skripte.TEILE), "jeder Teil in der Liste, keine Datei ohne Eintrag")
        self.assertEqual(skripte.TEILE[-1], "inkluagent.js", "der Kern laedt zuletzt (Startaufruf an seinem Ende)")
        self.assertEqual(len(set(skripte.TEILE)), len(skripte.TEILE))

    def test_vor_dem_seitenskript_geladen(self):
        tag = '{% for s in inkluagent_skripte %}<script src="/static/{{ s }}?v={{ asset_version }}"></script>{% endfor %}'
        self.assertEqual(self.app.count(tag), 1)
        self.assertLess(self.app.index(tag), self.app.index("{% raw %}"), "muss vor dem Seitenskript (init) stehen")
        self.assertIn("inkluagentSectionHtml(projectId)", self.app)   # der Aufrufer bleibt in app.html
        self.assertIn("inkluagentInit(projectId)", self.app)
        self.assertEqual(lies(BACKEND, "templates", "base_app.html").count(tag), 1, "fester Bereich auf jeder Seite: dieselbe Liste")
        self.assertIn('templates.env.globals["inkluagent_skripte"] = _inkluagent_skripte.TEILE', lies(BACKEND, "main.py"))

    def test_beim_laden_nichts_ausfuehren(self):
        """Nur Funktionen und Merker auf oberster Ebene — die Teile laufen vor dem Seitenskript, dessen Konstanten
        (main, liveRegion …) es dann noch nicht gibt. Einzige Ausnahme (Ausbau Runde 2, Schritt 3): inkluagentBereichAufbauen()
        baut den festen Bereich im Seitenrahmen; es braucht nur t() aus dashboard.js und kehrt ohne Bereich sofort zurueck. Es
        steht als letzte Anweisung im Kern, und der Kern laedt zuletzt — dann sind alle Teile da."""
        for name, text in self.teile.items():
            oben = [z for z in text.split("\n") if z and not z.startswith((" ", "}", "//", "\t"))]
            for z in oben:
                with self.subTest(teil=name, zeile=z[:60]):
                    if name == "inkluagent.js":
                        self.assertRegex(z, r"^(async function |function |let _inkluagentNachziehenWartet = false;$|"
                                            r"let _inkluagentVerfolgt = false;$|inkluagentBereichAufbauen\(\);$)")
                    else:
                        self.assertRegex(z, r"^(async function |function )", "in den Teilen nur Funktionen")
        kern = self.teile["inkluagent.js"].rstrip("\n").split("\n")
        self.assertEqual(kern[-1], "inkluagentBereichAufbauen();", "der Startaufruf ist das Letzte im Kern")
        aufbau = self.js[self.js.index("function inkluagentBereichAufbauen("):]
        aufbau = aufbau[:aufbau.index("\n}\n")]
        self.assertIn("if (!bereich || !inkluagentUeberall() || bereich.dataset.gebaut) return;", aufbau)

    def test_texte_aus_i18n(self):
        """check_i18n.py liest nur JS-Dateien, die „I18N“ enthalten — sonst fielen die Texte aus der Pruefung. Darum nennt jeder Teil
        window.I18N in seinem Kopf."""
        for name, text in self.teile.items():
            self.assertIn("I18N", text, name)
        self.assertGreater(len(re.findall(r"\bt\('", self.js)), 40)


if __name__ == "__main__":
    unittest.main()
