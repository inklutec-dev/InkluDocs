"""InkluAgent-Ausbau Runde 1, Schritt 4 (09.10.2026): Werkzeugluecke in Grafik- und Webseiten-Projekten geschlossen, hinter
funktionen.AGENT_BILD_WERKZEUGE (Umgebung INKLUAGENT_BILD_WERKZEUGE=an, Vorgabe aus). Jedes Werkzeug ruft denselben Kern wie der
Knopf; Loeschen mit derselben Rueckfrage wie dokument_loeschen. Wegwerf-Datenbank (/tmp), keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_agent_bild_werkzeuge.py
"""
import inspect
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_bild_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
os.environ.pop("INKLUAGENT_BILD_WERKZEUGE", None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, sicherheit  # noqa: E402
from inkluagent.tools import ausgaben, namen, oberflaeche  # noqa: E402

GRAFIK = {"project_type": "images", "tool": "grafik"}
WEB = {"project_type": "url", "tool": "web"}
AN = mock.patch.object(funktionen, "AGENT_BILD_WERKZEUGE", True)
SECHS = {"list_project_images", "get_image_metadata", "view_image", "generate_alt_text", "update_alt_text", "tavily_search"}
NEU = {"alt_texte_generieren", "exportiere_alt_texte", "ki_kontext_setzen", "eigener_prompt", "alt_sprache_setzen",
       "bild_umbenennen", "bild_loeschen"}


def satz(projekt):
    d, ex, system = agent_loop._werkzeugsatz(projekt, 1, 1)
    return {x["name"]: x for x in d}, ex, system


class Werkzeugsatz(unittest.TestCase):
    def test_schalter_aus_wie_bisher(self):
        self.assertFalse(funktionen.AGENT_BILD_WERKZEUGE)
        for projekt in (GRAFIK, WEB):
            je, _ex, system = satz(projekt)
            self.assertEqual(set(je), SECHS)
            self.assertNotIn("Grafik- und Webseiten-Projekte", system)

    def test_grafik_und_web_mit_schalter(self):
        with AN:
            g, gex, gsys = satz(GRAFIK)
            w, wex, wsys = satz(WEB)
        self.assertEqual(set(g), SECHS | NEU)
        self.assertEqual(set(w), SECHS | NEU | {"dokument_umbenennen", "dokument_loeschen"})
        for je, ex in ((g, gex), (w, wex)):
            handler = ex._handlers()
            for name in je:
                self.assertIn(name, handler, f"{name}: beschrieben, aber ohne Ausfuehrung")
                self.assertIn(name, namen.WERKZEUG_NAMEN)
            self.assertNotIn("ausgabe_loeschen", handler, "keine Ablage in Grafik- und Webseiten-Projekten")
            for name in ("alt_texte_generieren", "exportiere_alt_texte", "bild_loeschen"):
                self.assertIn("bestaetigt", je[name]["input_schema"]["properties"], name)
        self.assertIn("bestaetigt", w["dokument_loeschen"]["input_schema"]["properties"])
        self.assertEqual(gsys.count("Grafik- und Webseiten-Projekte: was du zusätzlich kannst"), 1)
        self.assertIn("Grafik-Projekt (einzelne Bilder)", gsys)
        self.assertNotIn("dokument_loeschen", gsys)
        self.assertIn("Webseiten-Projekt (gescannte Webseiten)", wsys)
        self.assertIn("dokument_loeschen", wsys)

    def test_pdf_und_word_unveraendert(self):
        for projekt in ({"project_type": "pdf", "tool": "pdf"}, {"project_type": "docx", "tool": "word"},
                        {"project_type": "pdfform", "tool": "formular"}):
            ohne, _e, s1 = satz(projekt)
            with AN:
                mit, _e2, s2 = satz(projekt)
            self.assertEqual(set(ohne), set(mit))
            self.assertEqual(s1, s2)

    def test_derselbe_kern_wie_der_knopf(self):
        with open(os.path.join(os.path.dirname(database.__file__), "main.py"), encoding="utf-8") as f:
            main = f.read()
        umb = main[main.index("async def rename_image"):main.index("def _bild_loeschen_sync")]
        self.assertIn("return _bild_umbenennen_sync(user[\"id\"], project_id, image_id", umb)
        loe = main[main.index("async def delete_image"):main.index('@app.delete("/api/projects/{project_id}")')]
        self.assertIn("return _bild_loeschen_sync(user[\"id\"], project_id, image_id)", loe)
        self.assertIn("_main()._bild_umbenennen_sync(", inspect.getsource(oberflaeche.bild_umbenennen))
        self.assertIn("_main()._bild_loeschen_sync(", inspect.getsource(oberflaeche.bild_loeschen))


class BildLoeschenUndUmbenennen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(os.path.dirname(database.__file__))
        import main  # noqa: F401  (Kern der Knoepfe)
        c = database.get_db()
        c.execute("INSERT OR IGNORE INTO users (id, email, password_hash, display_name) VALUES (1, 'test@example.invalid', 'x', 'Test (fiktiv)')")
        cls.pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status, total_images) "
                            "VALUES (1, 'foto.png', '', 'Fotos (fiktiv)', 'grafik', 'images', 'completed', 2)").lastrowid
        cls.bilder = [c.execute("INSERT INTO images (project_id, page_number, image_index, image_path, original_filename, width, height, status) "
                                "VALUES (?, 1, ?, ?, ?, 10, 10, 'done')", (cls.pid, i, os.path.join(TMP, f"b{i}.png"), f"foto{i}.png")).lastrowid
                      for i in range(2)]
        c.commit()
        c.close()

    def setUp(self):
        for d in (ausgaben._ANGEBOTE, ausgaben._LETZTES, ausgaben._NACH_ID, ausgaben._ERLEDIGT, ausgaben._VERBRAUCHT):
            d.clear()

    def _ex(self):
        with AN:
            _je, ex, _s = satz(GRAFIK)
        ex.project_id = self.pid
        return ex

    def test_umbenennen(self):
        r = self._ex().execute("bild_umbenennen", {"image_id": self.bilder[0], "name": "Sommerfest (fiktiv)"})
        self.assertTrue(r["ok"], r)
        c = database.get_db()
        self.assertEqual(c.execute("SELECT display_name FROM images WHERE id = ?", (self.bilder[0],)).fetchone()[0], "Sommerfest (fiktiv)")
        c.close()

    def test_loeschen_in_zwei_schritten(self):
        ex = self._ex()
        r1 = ex.execute("bild_loeschen", {"image_id": self.bilder[1]})
        self.assertTrue(r1["result"]["rueckfrage_noetig"])
        karte = r1.get("anhang") or {}
        self.assertEqual(karte.get("art"), "bestaetigung")
        self.assertIn("„Bild 2 (foto1.png)“", karte.get("text", ""))
        self.assertIn("nicht rückgängig", karte.get("text", ""))
        r2 = ex.execute("bild_loeschen", {"image_id": self.bilder[1], "bestaetigt": True})
        self.assertTrue(r2["result"]["rueckfrage_noetig"], "nicht in derselben Nachricht")
        ex2 = self._ex()
        sicherheit.nachricht_merken(ex2.turn_id, "Ja")
        r3 = ex2.execute("bild_loeschen", {"image_id": self.bilder[1], "bestaetigt": True})
        self.assertTrue(r3["ok"] and r3["result"].get("geloescht"), r3)
        c = database.get_db()
        self.assertEqual(c.execute("SELECT COUNT(*) FROM images WHERE id = ?", (self.bilder[1],)).fetchone()[0], 0)
        self.assertEqual(c.execute("SELECT total_images FROM projects WHERE id = ?", (self.pid,)).fetchone()[0], 1)
        c.close()

    def test_fremdes_bild_nicht(self):
        r = self._ex().execute("bild_loeschen", {"image_id": 999999, "bestaetigt": True})
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
