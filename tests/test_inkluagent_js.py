"""Konzept InkluAgent, Schritt 1 (09.10.2026): das Skript des Chat-Bereichs steht in frontend/inkluagent.js statt in
backend/templates/app.html. Ohne Verhaltensaenderung: dieselben Funktionen, genau einmal definiert, vor dem Seitenskript
geladen, Texte weiter aus window.I18N (check_i18n.py findet sie). Laeuft ohne Server:
    python3 -m unittest tests/test_inkluagent_js.py"""
import os
import re
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))

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
    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")
        self.app = lies(BACKEND, "templates", "app.html")

    def test_funktionen_genau_einmal_und_nur_in_der_eigenen_datei(self):
        alle_js = {n: lies(FRONTEND, n) for n in os.listdir(FRONTEND) if n.endswith(".js") and ".bak" not in n}
        for f in FUNKTIONEN:
            with self.subTest(funktion=f):
                muster = re.compile(r"\b(?:async\s+)?function\s+" + f + r"\s*\(")
                self.assertEqual(len(muster.findall(self.js)), 1)
                self.assertEqual(len(muster.findall(self.app)), 0, "steht noch in app.html")
                for name, text in alle_js.items():
                    if name != "inkluagent.js":
                        self.assertEqual(len(muster.findall(text)), 0, f"auch in {name}")

    def test_vor_dem_seitenskript_geladen(self):
        tag = '<script src="/static/inkluagent.js?v={{ asset_version }}"></script>'
        self.assertEqual(self.app.count(tag), 1)
        self.assertLess(self.app.index(tag), self.app.index("{% raw %}"), "muss vor dem Seitenskript (init) stehen")
        self.assertIn("inkluagentSectionHtml(projectId)", self.app)   # der Aufrufer bleibt in app.html
        self.assertIn("inkluagentInit(projectId)", self.app)

    def test_beim_laden_nichts_ausfuehren(self):
        """Nur Funktionen und ein Merker auf oberster Ebene — die Datei laeuft vor dem Seitenskript, dessen Konstanten
        (main, liveRegion …) es dann noch nicht gibt."""
        oben = [z for z in self.js.split("\n") if z and not z.startswith((" ", "}", "//", "\t"))]
        for z in oben:
            with self.subTest(zeile=z[:60]):
                self.assertRegex(z, r"^(async function |function |let _inkluagentNachziehenWartet = false;$)")

    def test_texte_aus_i18n(self):
        """check_i18n.py liest nur JS-Dateien, die „I18N“ enthalten — sonst fielen die Texte aus der Pruefung."""
        self.assertIn("I18N", self.js)
        self.assertGreater(len(re.findall(r"\bt\('", self.js)), 40)


if __name__ == "__main__":
    unittest.main()
