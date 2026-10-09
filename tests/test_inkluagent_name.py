"""Der KI-Agent heisst in der Oberflaeche „InkluAgent“ (Steve 09.10.2026, Michael einverstanden; vorher „Chatbot“).

Geprueft ohne Server: Auf/Zu-Knopf und Ansagen im Werkzeug, Demo, Ablage, Verwaltung und alle sechs Sprachkataloge.
Der Name wird nicht uebersetzt. Interne Schluessel (zweck „chatbot“, CSS-Klassen, API-Pfade) bleiben, wie sie sind.
Bewusst NICHT geprueft und unveraendert: Rechtstexte (Datenschutz, Nutzungsbedingungen), der DSGVO-Hinweis in der
Fusszeile (Kurzfassung der Datenschutzerklaerung) und die Erklaertexte vor dem Chat („Der InkluAgent ist ein
KI-Assistent …“), die in einer eigenen Runde ersetzt werden.
Aufruf: `python3 -m unittest tests/test_inkluagent_name.py`."""
import os
import re
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
SPRACHEN = ("de", "en", "fr", "es", "da", "sv")

# Dieselben Muster wie backend/scripts/check_i18n.py
RE_GETTEXT = re.compile(r"""\b_\(\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")\s*[,)]""")
RE_T = re.compile(r"""\bt\(\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")\s*[,)]""")

# Ausnahmen mit Grund: Erklaertext vor dem Chat (eigene Runde) und DSGVO-Hinweis (gehoert zur Datenschutzerklaerung).
ERLAUBT = (
    lambda s: s.startswith("Der InkluAgent ist ein KI-Assistent."),
    lambda s: s.startswith("Der InkluAgent ist der KI-Assistent in deinem Projekt."),
    lambda s: s.startswith(" Hosting bei Hetzner Online"),
)


def lies(pfad):
    with open(pfad, encoding="utf-8") as f:
        return f.read()


def oberflaechen_texte():
    """msgid -> Fundstellen aus Templates (_() und t()) und den an I18N angebundenen JS-Dateien."""
    gefunden = {}
    tpl = os.path.join(BACKEND, "templates")
    for name in sorted(os.listdir(tpl)):
        if not name.endswith(".html") or ".bak" in name:
            continue
        text = lies(os.path.join(tpl, name))
        for rx in (RE_GETTEXT, RE_T):
            for m in rx.finditer(text):
                gefunden.setdefault(m.group(1) if m.group(1) is not None else m.group(2), []).append(name)
    for name in sorted(os.listdir(FRONTEND)):
        if not name.endswith(".js") or ".bak" in name:
            continue
        text = lies(os.path.join(FRONTEND, name))
        if "I18N" not in text:
            continue
        for m in RE_T.finditer(text):
            gefunden.setdefault(m.group(1) if m.group(1) is not None else m.group(2), []).append(name)
    return gefunden


def katalog(sprache):
    from babel.messages.pofile import read_po
    with open(os.path.join(BACKEND, "locales", sprache, "LC_MESSAGES", "messages.po"), "rb") as f:
        return {m.id: (m.string or "") for m in read_po(f) if m.id and isinstance(m.id, str)}


class Werkzeug(unittest.TestCase):
    def setUp(self):
        self.app = lies(os.path.join(BACKEND, "templates", "app.html"))

    def test_knopf_heisst_inkluagent(self):
        """Beschriftung beim Aufbau, beim Oeffnen und beim Schliessen: „InkluAgent“; Auf/Zu meldet aria-expanded."""
        self.assertEqual(self.app.count("t('InkluAgent')"), 3)
        self.assertIn('<button id="inkluagentToggle" type="button" aria-expanded="false" aria-controls="inkluagentPanel"',
                      self.app)
        self.assertRegex(self.app, r"<h2 id=\"inkluagentHeading\"[^>]*>'\s*\+\s*'<button id=\"inkluagentToggle\"")

    def test_ki_hinweis_bleibt(self):
        """KI-VO Art. 50: der Name allein sagt nicht, dass es eine KI ist — der Satz unter dem Knopf sagt es."""
        self.assertEqual(self.app.count("t('Der InkluAgent ist ein KI-Assistent."), 2)

    def test_absender_und_ansagen(self):
        for text in ("Nachricht an den InkluAgent", "Antwort vom InkluAgent ist da.", "InkluAgent denkt nach..."):
            with self.subTest(text=text):
                self.assertIn("t('" + text + "')", self.app)
        self.assertIn("(role === 'user' ? t('Du') : 'InkluAgent')", self.app)


class KeinChatbotMehr(unittest.TestCase):
    def test_kein_chatbot_oder_assistent_in_oberflaechentexten(self):
        falsch = {}
        for s, wo in oberflaechen_texte().items():
            if any(erlaubt(s) for erlaubt in ERLAUBT):
                continue
            if re.search(r"chat-?bot", s, re.I) or re.search(r"\bAssistent", s):
                falsch[s] = sorted(set(wo))
        self.assertEqual(falsch, {})

    def test_demo(self):
        demo = lies(os.path.join(BACKEND, "templates", "demo.html"))
        for text in ("Mit dem InkluAgent verfeinern", "Nachricht an den InkluAgent", "Nachricht an den InkluAgent …"):
            with self.subTest(text=text):
                self.assertIn("_('" + text + "')", demo)
        js = lies(os.path.join(FRONTEND, "demo.js"))
        self.assertIn('t("InkluAgent denkt nach...")', js)
        self.assertIn('t("Antwort vom InkluAgent ist da.")', js)

    def test_ablage_und_verwaltung(self):
        self.assertIn("t('über den InkluAgent')", lies(os.path.join(BACKEND, "templates", "ablage.html")))
        kosten = lies(os.path.join(BACKEND, "templates", "verwaltung_ki_kosten.html"))
        self.assertIn("chatbot: t('InkluAgent')", kosten)   # Schluessel „chatbot“ bleibt (Datenbank), nur der Anzeigename
        self.assertIn("Änderungen durch den InkluAgent", kosten)
        self.assertRegex(lies(os.path.join(BACKEND, "ki_kosten.py")), r'"chatbot": "InkluAgent"')

    def test_server_antworten(self):
        """Antworten, die im Chat erscheinen, nennen den InkluAgent, nicht „den Assistenten“."""
        self.assertNotRegex(lies(os.path.join(BACKEND, "inkluagent", "chat_engine.py")), r'"[^"\n]*\bAssistent\b[^"\n]*"')
        self.assertIn("dann hilft der InkluAgent.", lies(os.path.join(BACKEND, "main.py")))


class Kataloge(unittest.TestCase):
    def test_name_wird_nicht_uebersetzt(self):
        for sprache in SPRACHEN:
            kat = katalog(sprache)
            with self.subTest(sprache=sprache):
                self.assertEqual(kat.get("InkluAgent"), "InkluAgent")
                self.assertNotIn("Chatbot", kat)
                for mid, ms in kat.items():
                    if "InkluAgent" in mid and ms:
                        self.assertIn("InkluAgent", ms, mid[:80])

    def test_keine_chatbot_uebersetzung_fuer_den_agenten(self):
        """In keiner Sprache heisst der Agent noch chatbot/chatbotten/chattbot (Ausnahme: DSGVO-Hinweis s. o.)."""
        for sprache in SPRACHEN:
            for mid, ms in katalog(sprache).items():
                if any(erlaubt(mid) for erlaubt in ERLAUBT):
                    continue
                with self.subTest(sprache=sprache, msgid=mid[:60]):
                    self.assertNotRegex(ms, r"(?i)chatt?bot")


if __name__ == "__main__":
    unittest.main()
