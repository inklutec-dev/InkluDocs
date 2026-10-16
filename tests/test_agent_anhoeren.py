"""„Anhören“ unter jeder Antwort des InkluAgent (16.10.2026, Steve: „wie bei InkluEdit auch im Chat einen Button zum
Antwortvorlesen“).

Prueft am Quelltext (frontend/inkluagent.js, style.css): Knopf nur bei Antworten des Agenten, direkt unter dem Absender, auch im
aelteren Gespraech; Start/Stopp ueber denselben Knopf (aria-pressed); Escape haelt an; immer nur eine Stimme (gemeinsame Klasse
tts-btn mit „Anhören“ beim Alt-Text, app.html); NUR Stimmen auf dem Geraet (localService, Datenschutz wie app.html); gelesen wird
nur der Inhalt der Antwort in ihrer Sprache; ohne Sprachausgabe kein Knopf. Das Verhalten im Browser prueft die Klickprobe
(Playwright, nachgebaute Sprachausgabe).
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


class Anhoeren(unittest.TestCase):
    def setUp(self):
        self.js = lies("inkluagent.js")
        self.knopf = funktion(self.js, "function inkluagentAnhoerenKnopf(")
        self.lesen = funktion(self.js, "async function inkluagentAnhoeren(")
        self.stopp = funktion(self.js, "function inkluagentAnhoerenStopp(")

    def test_nur_bei_antworten_unter_dem_absender(self):
        anfuegen = funktion(self.js, "function inkluagentAppendMessage(")
        self.assertIn("wrapper.appendChild(roleEl);\n    if (role === 'assistant') inkluagentAnhoerenKnopf(roleEl, contentEl);\n"
                      "    wrapper.appendChild(contentEl);", anfuegen)
        self.assertIn("roleEl.insertAdjacentElement('afterend', b);", self.knopf)
        self.assertIn("if (m.role === 'assistant') inkluagentAnhoerenKnopf(h, c);", self.js, "auch im aelteren Gespraech")
        self.assertNotIn("inkluagentAnhoerenKnopf", funktion(self.js, "function inkluagentEntwurfText("),
                         "nicht am wachsenden Entwurf")

    def test_echter_knopf_mit_zustand(self):
        self.assertIn("document.createElement('button')", self.knopf)
        self.assertIn("b.type = 'button';", self.knopf)
        self.assertIn("b.setAttribute('aria-pressed', 'false');", self.knopf)
        self.assertIn("tts-btn inkluagent-anhoeren", self.knopf)
        self.assertIn("b.textContent = t('Anhören');", self.knopf)
        self.assertIn("b.setAttribute('aria-pressed', 'true');\n    b.textContent = t('Stopp');", self.lesen)
        self.assertIn("if (b.getAttribute('aria-pressed') === 'true') { inkluagentAnhoerenStopp(); return; }", self.lesen)

    def test_ohne_sprachausgabe_kein_knopf(self):
        self.assertTrue(self.knopf.split("{", 1)[1].lstrip().startswith(
            "if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) return;"))

    def test_nur_stimmen_auf_dem_geraet(self):
        self.assertIn("v.localService && v.lang", self.lesen)
        self.assertIn("Aus Datenschutzgründen lesen wir nicht über einen Sprachdienst im Internet vor.", self.lesen)
        self.assertNotIn("u.lang = kennung", self.lesen, "ohne passende lokale Stimme nie die Vorgabe des Browsers nehmen")

    def test_nur_der_inhalt_in_seiner_sprache(self):
        self.assertIn("contentEl.innerText || contentEl.textContent", self.lesen)
        self.assertIn("contentEl.closest('[lang]')", self.lesen)

    def test_immer_nur_eine_stimme_und_escape(self):
        self.assertIn("window.speechSynthesis.cancel()", self.stopp)
        self.assertIn(".tts-btn[aria-pressed=\"true\"]", self.stopp, "setzt auch „Anhören“ beim Alt-Text zurueck")
        self.assertIn("inkluagentAnhoerenStopp();\n    const lauf = window._inkluagentAnhoerenLauf;", self.lesen)
        self.assertIn("e.key === 'Escape' && document.querySelector('.inkluagent-anhoeren[aria-pressed=\"true\"]')", self.knopf)

    def test_gestaltung(self):
        css = lies("style.css")
        self.assertIn(".inkluagent-anhoeren {", css)


if __name__ == "__main__":
    unittest.main()
