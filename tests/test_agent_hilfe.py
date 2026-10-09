"""InkluAgent-Ausbau Runde 1, Schritt 3 (09.10.2026): drei Stichpunkte vor dem Chat, Hilfe-Seite /hilfe/inkluagent aus
denselben Quellen wie Werkzeugsatz, Werkzeugnamen und Schalter (inkluagent/hilfe.py), Link „Hilfe“ in der Seitenleiste.
Schalter funktionen.AGENT_HILFE (Umgebung INKLUAGENT_HILFE=an, Vorgabe aus). Laeuft ohne Server:
    python3 -m unittest tests/test_agent_hilfe.py"""
import os
import re
import sys
import unittest
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)
os.environ.pop("INKLUAGENT_HILFE", None)

import funktionen  # noqa: E402
from inkluagent import hilfe  # noqa: E402

BACKEND = os.path.dirname(os.path.abspath(funktionen.__file__))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
SPRACHEN = ("de", "en", "fr", "es", "da", "sv")
KURZ = ("Der InkluAgent ist eine KI und arbeitet mit deinen Projekten und Dateien.",
        "Sag ihm in eigenen Worten, was du brauchst.",
        "Bevor etwas Credits kostet oder sich nicht rückgängig machen lässt, fragt er dich.")


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def katalog(sprache):
    from babel.messages.pofile import read_po
    with open(os.path.join(BACKEND, "locales", sprache, "LC_MESSAGES", "messages.po"), "rb") as f:
        return {m.id: (m.string or "") for m in read_po(f) if m.id and isinstance(m.id, str)}


class Schalter(unittest.TestCase):
    def test_vorgabe_aus_und_ueberall_derselbe(self):
        self.assertFalse(funktionen.AGENT_HILFE)
        self.assertIs(funktionen.fuer_oberflaeche()["agent_hilfe"], funktionen.AGENT_HILFE)
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            funktionen.endpunkt_frei("AGENT_HILFE")
        self.assertEqual(cm.exception.status_code, 404)
        main = lies(BACKEND, "main.py")
        route = main[main.index('@app.get("/hilfe/inkluagent"'):]
        self.assertIn('funktionen.endpunkt_frei("AGENT_HILFE")', route[:900])
        self.assertIn('"inkluagent_hilfe": bool(funktionen.AGENT_HILFE)', main)


class Seite(unittest.TestCase):
    def test_vier_projektarten_aus_dem_werkzeugsatz(self):
        s = hilfe.seite()
        self.assertEqual([a["titel"] for a in s["projektarten"]], ["PDF-Dokumente", "Word-Dokumente", "Grafiken", "Webseiten"])
        pdf = s["projektarten"][0]
        self.assertIn("Dokumentstand", pdf["ohne_rueckfrage"])
        self.assertIn("Dokument löschen", pdf["mit_rueckfrage"])
        self.assertIn("PDF herunterladen", pdf["mit_rueckfrage"])
        alle = [n for a in s["projektarten"] for n in a["ohne_rueckfrage"] + a["mit_rueckfrage"]]
        for aus in ("KI-basierte Prüfung", "Komplett barrierefrei machen", "Korrektur", "Alt-Text zurücksetzen"):
            self.assertNotIn(aus, alle, "ausgeschaltete Funktion darf die Hilfe nicht versprechen")
        self.assertNotIn("Werkzeug", alle, "jedes Werkzeug hat einen Anzeigenamen")

    def test_folgt_den_schaltern(self):
        with mock.patch.object(funktionen, "KETTE", True):
            pdf = hilfe.seite()["projektarten"][0]
        self.assertIn("Komplett barrierefrei machen", pdf["mit_rueckfrage"])
        with mock.patch.object(funktionen, "AGENT_SICHERHEIT", True):
            pdf = hilfe.seite()["projektarten"][0]
        self.assertIn("Alt-Text speichern", pdf["mit_rueckfrage"], "mit Deckel fragt der Agent auch bei Einzelaktionen")

    def test_seite_in_allen_sprachen(self):
        from i18n import get_gettext, get_templates
        tpl = get_templates().get_template("hilfe_inkluagent.html")
        for sprache in SPRACHEN:
            _ = get_gettext(sprache)
            html = tpl.render(_=_, current_lang=sprache, i18n_json="{}", asset_version="t", language_labels={},
                              supported_languages=SPRACHEN, hilfe=hilfe.seite(_), tageslimit=100, max_zeichen=5000,
                              einzel_deckel=False)
            with self.subTest(sprache=sprache):
                self.assertEqual(html.count("<h1>"), 1)
                self.assertEqual(len(re.findall(r"<h3 id=\"hilfe-", html)), 4)
                self.assertIn('href="/einstellungen"', html)
                self.assertNotIn("{n}", html)
                if sprache != "de":
                    self.assertNotIn("Alles, was der InkluAgent kann", html)

    def test_texte_in_allen_katalogen(self):
        ids = list(KURZ) + ["Alles, was der InkluAgent kann", "Hilfe"] + [t for _k, t, _p in hilfe.PROJEKTARTEN] + list(hilfe.AUSSERHALB)
        tpl = lies(BACKEND, "templates", "hilfe_inkluagent.html")
        ids += [m.group(1) for m in re.finditer(r"_\('((?:[^'\\]|\\.)*)'\)", tpl)]
        for sprache in SPRACHEN:
            kat = katalog(sprache)
            for mid in ids:
                with self.subTest(sprache=sprache, msgid=mid[:50]):
                    self.assertIn(mid, kat)
                    if sprache != "de":
                        self.assertTrue(kat[mid])


class Oberflaeche(unittest.TestCase):
    def test_drei_stichpunkte_mit_ki_hinweis_und_link(self):
        js = lies(FRONTEND, "inkluagent.js")
        for text in KURZ + ("Alles, was der InkluAgent kann",):
            self.assertIn("t('" + text + "')", js)
        self.assertIn("eine KI", KURZ[0], "KI-VO Art. 50: der erste Punkt sagt, dass es eine KI ist")
        self.assertIn('<a href="/hilfe/inkluagent">', js)
        self.assertIn("window.FUNKTIONEN.agent_hilfe", js)
        self.assertEqual(js.count("t('Der InkluAgent ist ein KI-Assistent."), 2, "ohne Schalter bleibt die alte Einleitung")

    def test_link_hilfe_in_der_seitenleiste(self):
        dash = lies(FRONTEND, "dashboard.js")
        self.assertIn("{ href: '/hilfe/inkluagent', label: t('Hilfe'), hilfe: true", dash)
        self.assertIn("if (it.hilfe && !(currentUser && currentUser.inkluagent_hilfe)) return;", dash)
        # feste Stelle: direkt nach „Über uns und Kontakt“, vor den Eintraegen fuer Admins und Bearbeiter
        self.assertLess(dash.index("label: t('Über uns und Kontakt')"), dash.index("label: t('Hilfe')"))
        self.assertLess(dash.index("label: t('Hilfe')"), dash.index("label: t('Verwaltung')"))


if __name__ == "__main__":
    unittest.main()
