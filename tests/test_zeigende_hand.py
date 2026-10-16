"""Die zeigende Hand beim Hinfuehren (16.10.2026, Steve: „dass der Agent das mit der Hand zeigt“, wie InkluEdit seit 15.10.).

Die Hand ist nur fuers Auge: Fokus und Markierung bleiben wie vorher (test_agent_begleiter), die Hand ist aria-hidden, nimmt keine
Klicks, wird bei reduzierter Bewegung nicht gebaut, es gibt nie zwei, und sie verschwindet von selbst. Der Klang kommt aus klang.js.
"""
import os
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "frontend")), os.path.join(HIER, "frontend"),
                            "/app/frontend") if os.path.isdir(k))


def lies(name):
    with open(os.path.join(FRONTEND, name), encoding="utf-8") as f:
        return f.read()


def funktion(text, kopf):
    t = text[text.index(kopf):]
    return t[:t.index("\n}\n")]


class ZeigendeHand(unittest.TestCase):
    def setUp(self):
        self.dash = lies("dashboard.js")
        self.hand = funktion(self.dash, "function zeigeHand(")

    def test_hinfuehren_zeigt_nach_dem_fokus(self):
        h = funktion(self.dash, "function hinfuehrenBeimLaden(")
        self.assertIn("fokusAuf(el, { preventScroll: true });\n    zeigeHand(el);", h)
        # S14 bleibt: nur Ziele aus der Liste des Servers — die Hand aendert daran nichts
        self.assertIn("if (!liste || liste.indexOf(ziel) < 0) return;", h)

    def test_nur_fuers_auge(self):
        self.assertIn("hand.setAttribute('aria-hidden', 'true')", self.hand)
        self.assertIn("ring.setAttribute('aria-hidden', 'true')", self.hand)
        self.assertIn("svg.setAttribute('focusable', 'false')", self.hand)
        self.assertNotIn(".focus(", self.hand)   # nimmt dem Ziel den Fokus nicht

    def test_reduzierte_bewegung_ohne_hand_aber_mit_klang(self):
        i_klang = self.hand.index("zeigeKlang();")
        i_ruhe = self.hand.index("(prefers-reduced-motion: reduce)")
        self.assertLess(i_klang, i_ruhe)          # Klang ist keine Bewegung: kommt auch bei reduzierter Bewegung
        self.assertIn("matches) return;", self.hand[i_ruhe:i_ruhe + 80])

    def test_nie_zwei_und_raeumt_auf(self):
        self.assertIn("document.querySelectorAll('.inkluagent-hand, .inkluagent-hand-ring').forEach(x => x.remove())", self.hand)
        self.assertIn("hand.remove()", self.hand)
        self.assertIn("ring.remove()", self.hand)

    def test_fehler_brechen_nichts(self):
        self.assertIn("} catch (e) { /* ohne Hand: Markierung und Fokus bleiben */ }", self.hand)

    def test_klang_aus_klang_js(self):
        k = funktion(self.dash, "function zeigeKlang(")
        self.assertIn("klang.spielen('karte')", k)
        self.assertIn("imRahmen() ? k(window.parent) : null", k)
        self.assertIn("'karte'", lies("klang.js"))

    def test_lizenzhinweis(self):
        self.assertIn("Lucide", self.dash)
        self.assertIn("ISC-Lizenz", self.dash)

    def test_css_nimmt_keine_klicks(self):
        css = lies("style.css")
        for klasse in (".inkluagent-hand {", ".inkluagent-hand-ring {"):
            regel = css[css.index(klasse):]
            regel = regel[:regel.index("}")]
            self.assertIn("pointer-events: none", regel)
            self.assertIn("position: fixed", regel)


if __name__ == "__main__":
    unittest.main()
