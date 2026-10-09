"""InkluAgent-Ausbau Runde 1, Schritt 5 (09.10.2026): Ansicht je Konto — „Manuelle Ansicht“ (intern klassisch, Vorgabe) und
„Agentenansicht“ (intern agent), Hauptort Einstellungen, schneller Umschalter in der Seitenleiste, Wert in users.oberflaeche,
beim Seitenbau gesetzt. Schalter funktionen.AGENT_ANSICHT (Umgebung INKLUAGENT_ANSICHT=an, Vorgabe aus).
Wegwerf-Datenbank (/tmp), keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_ansicht.py
"""
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_ansicht_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
os.environ.pop("INKLUAGENT_ANSICHT", None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import ansicht  # noqa: E402

BACKEND = os.path.dirname(os.path.abspath(database.__file__))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
SPRACHEN = ("de", "en", "fr", "es", "da", "sv")
AN = mock.patch.object(funktionen, "AGENT_ANSICHT", True)


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def spalten():
    c = database.get_db()
    try:
        return {r[1]: r[4] for r in c.execute("PRAGMA table_info(users)").fetchall()}
    finally:
        c.close()


class Migration(unittest.TestCase):
    def test_vorwaerts_spalte_mit_vorgabe(self):
        self.assertEqual(spalten().get("oberflaeche"), "'klassisch'")
        uid = database.create_user("neu@example.invalid", "x" * 12, "Neu (fiktiv)")
        self.assertEqual(ansicht.fuer_konto(uid), "klassisch", "neue Konten: manuelle Ansicht")

    def test_rueckwaerts_und_wieder_vorwaerts(self):
        """Rueckweg laut Doku (DROP COLUMN): alter Code arbeitet ohne die Spalte; ein erneuter Start legt sie wieder an."""
        db = os.path.join(TMP, "rueckweg.db")
        with mock.patch.object(database, "DB_PATH", db), mock.patch.dict(os.environ, {"INKLUDOCS_DB": db}):
            database.init_db()
            c = sqlite3.connect(db)
            c.execute("ALTER TABLE users DROP COLUMN oberflaeche")
            c.commit()
            c.close()
            uid = database.create_user("alt@example.invalid", "x" * 12, "Alt (fiktiv)")   # alter Weg ohne Spalte
            self.assertTrue(uid)
            database.init_db()
            database.init_db()                                                             # idempotent
            c = sqlite3.connect(db)
            try:
                wert = c.execute("SELECT oberflaeche FROM users WHERE id = ?", (uid,)).fetchone()[0]
            finally:
                c.close()
            self.assertEqual(wert, "klassisch")


class Wert(unittest.TestCase):
    def test_schalter_aus(self):
        self.assertFalse(funktionen.AGENT_ANSICHT)
        self.assertIsNone(ansicht.seiten_wert(1), "ohne Schalter bleibt der Seitenrahmen, wie er ist")
        main = lies(BACKEND, "main.py")
        self.assertIn("return _ansicht.fuer_konto(user_id) if _ansicht.an() else _ansicht.VORGABE", main)

    def test_setzen_und_lesen(self):
        uid = database.create_user("wechsel@example.invalid", "x" * 12, "Wechsel (fiktiv)")
        self.assertEqual(ansicht.setzen(uid, "agent"), "agent")
        self.assertEqual(ansicht.fuer_konto(uid), "agent")
        with AN:
            self.assertEqual(ansicht.seiten_wert(uid), "agent")
        with self.assertRaises(ValueError):
            ansicht.setzen(uid, "ohne_agent")
        c = database.get_db()
        c.execute("UPDATE users SET oberflaeche = 'unbekannt' WHERE id = ?", (uid,))
        c.commit()
        c.close()
        self.assertEqual(ansicht.fuer_konto(uid), "klassisch", "Unbekanntes faellt auf die Vorgabe")

    def test_erweiterbar_an_einer_stelle(self):
        """Ein dritter Wert (spaeter moeglich) = ein Eintrag in OBERFLAECHEN, NAMEN, BESCHREIBUNG."""
        with mock.patch.object(ansicht, "OBERFLAECHEN", ansicht.OBERFLAECHEN + ("drei",)), \
                mock.patch.dict(ansicht.NAMEN, {"drei": "Dritte (fiktiv)"}), mock.patch.dict(ansicht.BESCHREIBUNG, {"drei": "…"}):
            self.assertTrue(ansicht.gueltig("drei"))
            self.assertEqual([a["wert"] for a in ansicht.auswahl()], ["klassisch", "agent", "drei"])
        self.assertEqual([a["name"] for a in ansicht.auswahl()], ["Manuelle Ansicht", "Agentenansicht"])

    def test_endpunkt(self):
        main = lies(BACKEND, "main.py")
        teil = main[main.index('@app.put("/api/me/oberflaeche")'):]
        teil = teil[:teil.index("\n\n\n")]
        self.assertIn('funktionen.endpunkt_frei("AGENT_ANSICHT")', teil)
        self.assertIn("if not _ansicht.gueltig(wert):", teil)
        self.assertIn('"oberflaeche_waehlbar": bool(funktionen.AGENT_ANSICHT)', main)
        self.assertIn('extra.setdefault("oberflaeche", _ansicht.seiten_wert(_nutzer.get("id")))', main)


class Darstellung(unittest.TestCase):
    def _render(self, name, sprache="de", **kw):
        from i18n import get_gettext, get_templates
        _ = get_gettext(sprache)
        return get_templates().get_template(name).render(_=_, current_lang=sprache, i18n_json="{}", asset_version="t",
                                                          language_labels={s: s for s in SPRACHEN}, supported_languages=SPRACHEN, **kw)

    def test_einstellungen_mit_und_ohne_schalter(self):
        from i18n import get_gettext
        for sprache in SPRACHEN:
            html = self._render("einstellungen.html", sprache, ansicht_waehlbar=True, oberflaeche="agent",
                                ansicht_auswahl=ansicht.auswahl(get_gettext(sprache)))
            with self.subTest(sprache=sprache):
                self.assertEqual(len(re.findall(r'type="radio" name="oberflaeche"', html)), 2)
                self.assertRegex(html, r'id="ansicht-agent" value="agent"\s+aria-describedby="ansicht-agent-text" checked')
                self.assertIn("<fieldset>", html)
                self.assertIn('role="status"', html)
        aus = self._render("einstellungen.html", ansicht_waehlbar=False, ansicht_auswahl=[])
        self.assertNotIn("ansichtForm", aus)
        neu = self._render("einstellungen.html", ansicht_waehlbar=True, oberflaeche="klassisch", ansicht_auswahl=ansicht.auswahl())
        self.assertRegex(neu, r'id="ansicht-klassisch" value="klassisch"\s+aria-describedby="ansicht-klassisch-text" checked')

    def test_seitenrahmen(self):
        self.assertIn('<body data-oberflaeche="agent">', self._render("vorlagen.html", oberflaeche="agent"))
        self.assertIn("<body>", self._render("vorlagen.html", oberflaeche=None))

    def test_oberflaeche_js(self):
        js = lies(FRONTEND, "inkluagent.js")
        self.assertIn("function inkluagentAgentenansicht()", js)
        self.assertIn("if (inkluagentAgentenansicht()) { await inkluagentOpen(projectId, false, true); return; }", js)
        self.assertIn('[data-agentenansicht="aus"]', lies(FRONTEND, "style.css"))
        self.assertIn("k.hidden = true;", js)
        self.assertIn("function inkluagentAgentenansichtAufheben()", js)   # nur diese Seite manuell (Hochladen, Knoepfe)
        self.assertIn("t('Dieses Projekt in der manuellen Ansicht zeigen')", js)
        dash = lies(FRONTEND, "dashboard.js")
        self.assertIn("if (!currentUser || !currentUser.oberflaeche_waehlbar) return;", dash)
        self.assertIn("t('Zur manuellen Ansicht wechseln') : t('Zur Agentenansicht wechseln')", dash)
        self.assertIn("ansichtFokusNachWechsel();", dash)

    def test_texte_in_allen_katalogen(self):
        from babel.messages.pofile import read_po
        ids = list(ansicht.NAMEN.values()) + list(ansicht.BESCHREIBUNG.values())
        for datei in (("templates", "einstellungen.html"),):
            ids += [m.group(1) for m in re.finditer(r"_\('((?:[^'\\]|\\.)*)'\)", lies(BACKEND, *datei))]
        ids += ["Zur manuellen Ansicht wechseln", "Zur Agentenansicht wechseln", "Gespeichert: {ansicht}.",
                "Die Ansicht konnte nicht gespeichert werden. Bitte versuch es noch einmal.", "Diese Ansicht gibt es nicht."]
        for sprache in SPRACHEN:
            with open(os.path.join(BACKEND, "locales", sprache, "LC_MESSAGES", "messages.po"), "rb") as f:
                kat = {m.id: m.string for m in read_po(f) if m.id and isinstance(m.id, str)}
            for mid in ids:
                with self.subTest(sprache=sprache, msgid=mid[:40]):
                    self.assertIn(mid, kat)
                    if sprache != "de":
                        self.assertTrue(kat[mid])


if __name__ == "__main__":
    unittest.main()
