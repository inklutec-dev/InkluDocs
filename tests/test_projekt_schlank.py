"""Ansichtswechsel (05.10.2026, docs/ANSICHTEN_LEISTUNG.md): schlanke Projektantwort, Projektkopf, Seitentext beim
Aufklappen, Schreibweg mit projekt_texte und gemerkte Struktur-Kennzahlen — auf einer WEGWERF-Datenbank (/tmp).
    docker exec -w /app <container> python3 -m unittest /app/tests/test_projekt_schlank.py
"""
import asyncio
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="projekt_schlank_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
sys.path.insert(0, "/app")
os.chdir("/app")
import database  # noqa: E402
database.init_db()
import main  # noqa: E402
import projekt_texte  # noqa: E402
import tagging_api  # noqa: E402
import api_dokumente_v1  # noqa: E402

KAPITEL = "Kapiteltext ohne Ueberschrift. " * 5000
SEITE = "Seitentext Seite 1 (fiktiv)."
SCHWER = {"context_text", "page_text", "pipeline_steps", "validation_result", "image_path", "page_view_path",
          "kontext_id", "seitentext_id"}
# Felder, die app.html (renderImages, Filter, Gastansicht) und Public API v1 (_bild_item) lesen
GEBRAUCHT = {"id", "status", "page_number", "image_type", "image_index", "review_note", "document_id", "reviews", "thread",
             "konfidenz", "fehler_grund", "alt_text_edited", "alt_text", "review_status", "langbeschreibung",
             "original_filename", "original_alt", "feedback", "display_name", "context_mode", "needs_review", "width",
             "height", "gen_language", "alt_text_vorher", "reviewed_at", "hat_seitenansicht", "hat_seitentext"}


def _pdf(pfad, seiten=2):
    import fitz
    d = fitz.open()
    for i in range(seiten):
        d.new_page().insert_text((50, 70), f"Seite {i + 1} (fiktiv)")
    d.save(pfad)
    d.close()


class ProjektSchlankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = database.get_db()
        c.execute("INSERT OR IGNORE INTO users (id, email, password_hash, display_name) VALUES (1, 'test@example.invalid', 'x', 'Test')")
        cls.pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                            "VALUES (1, 'a.pdf', '', 'Schlank (fiktiv)', 'pdf', 'pdf', 'extracted')").lastrowid
        cls.pdf = os.path.join(TMP, "a.pdf")
        _pdf(cls.pdf)
        cls.did = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path, getaggt) "
                            "VALUES (?, 1, 'a.pdf', ?, 0)", (cls.pid, cls.pdf)).lastrowid
        bild = os.path.join(TMP, "b.png")
        with open(bild, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        ansicht = os.path.join(TMP, "p1.png")
        with open(ansicht, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        bilder = [{"page_number": 1, "image_path": bild, "width": 10, "height": 10, "xref": -i, "context_text": KAPITEL,
                   "page_view_path": ansicht, "page_text": SEITE, "original_alt": ""} for i in range(1, 4)]
        main._bilder_uebernehmen(c, cls.pid, cls.did, bilder, "pdf", cls.pdf)
        c.commit()
        cls.bild_ids = [r[0] for r in c.execute("SELECT id FROM images WHERE project_id = ? ORDER BY id", (cls.pid,))]
        c.close()

    def test_schreibweg_legt_text_einmal_ab(self):
        c = database.get_db()
        zeilen = c.execute("SELECT * FROM images WHERE project_id = ?", (self.pid,)).fetchall()
        self.assertEqual(len(zeilen), 3)
        self.assertTrue(all(z["context_text"] == "" and z["page_text"] == "" for z in zeilen))
        self.assertEqual(len({z["kontext_id"] for z in zeilen}), 1)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM projekt_texte WHERE project_id = ?", (self.pid,)).fetchone()[0], 2)
        self.assertEqual(projekt_texte.bild_kontext(c, zeilen[0]), KAPITEL)   # byte-gleich
        c.close()

    def test_projektantwort_schlank(self):
        d = main._projekt_daten(self.pid, 1)
        self.assertEqual(len(d["images"]), 3)
        for img in d["images"]:
            self.assertFalse(SCHWER & set(img), SCHWER & set(img))
            self.assertTrue(GEBRAUCHT <= set(img), GEBRAUCHT - set(img))
            self.assertTrue(img["hat_seitenansicht"] and img["hat_seitentext"])
        self.assertTrue({"project", "documents", "show_langbeschreibung", "in_review", "share_roles", "ausgaben_anzahl"} <= set(d))
        self.assertNotIn("original_path", d["documents"][0])
        import json
        self.assertLess(len(json.dumps(d)), 20000)   # vorher 3 x 150.000 Zeichen Kontext

    def test_route_bleibt_async_fuer_api_v1(self):
        self.assertTrue(asyncio.iscoroutinefunction(main.get_project))
        erg = asyncio.run(main.get_project(self.pid, user={"id": 1}))
        items = [api_dokumente_v1._bild_item(self.pid, img) for img in erg["images"]]
        self.assertEqual(len(items), 3)
        self.assertEqual(set(items[0]), {"id", "type", "document_id", "page", "index", "filename", "width", "height", "status",
                                         "text_status", "alt_text", "alt_text_ki", "alt_text_edited", "langbeschreibung",
                                         "bildtyp", "konfidenz", "needs_review", "error", "language", "file_url"})

    def test_kopf(self):
        k = main._projekt_kopf(self.pid, 1)
        self.assertNotIn("images", k)
        self.assertEqual(sum(k["bilder_status"].values()), 3)   # Zaehler je Status (ein anderer Test setzt ein Bild auf done)
        self.assertEqual(k["bilder_gesamt"], 3)
        self.assertEqual(k["documents"][0]["id"], self.did)
        self.assertTrue({"lauf_art", "hat_felder", "letzte_ansicht"} <= set(k["project"]))

    def test_fremdes_projekt_404(self):
        from fastapi import HTTPException
        for f in (lambda: main._projekt_daten(self.pid, 2), lambda: main._projekt_kopf(self.pid, 2),
                  lambda: main._seitentext_lesen(self.bild_ids[0], 2, None),
                  lambda: main._seitentext_lesen(self.bild_ids[0], None, self.pid + 999)):
            with self.assertRaises(HTTPException) as e:
                f()
            self.assertEqual(e.exception.status_code, 404)

    def test_seitentext_beim_aufklappen(self):
        self.assertEqual(main._seitentext_lesen(self.bild_ids[0], 1, None)["text"], SEITE)
        self.assertEqual(main._seitentext_lesen(self.bild_ids[0], None, self.pid)["text"], SEITE)   # Gast

    def test_gast_freigabe_schlank(self):
        d = main._freigabe_daten(self.pid, "kunde")
        self.assertTrue(d["guest"])
        self.assertFalse(SCHWER & set(d["images"][0]))
        self.assertTrue(d["images"][0]["hat_seitentext"])

    def test_dokument_loeschen_raeumt_texte_auf(self):
        c = database.get_db()
        pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type) VALUES (1, 'b.pdf', '', 'Loeschen (fiktiv)', 'pdf', 'pdf')").lastrowid
        did = c.execute("INSERT INTO documents (project_id, doc_index, original_filename) VALUES (?, 1, 'b.pdf')", (pid,)).lastrowid
        main._bilder_uebernehmen(c, pid, did, [{"page_number": 1, "image_path": "/tmp/x.png", "width": 1, "height": 1,
                                                "context_text": "eigener Kontext", "page_text": "eigene Seite"}], "pdf", "/tmp/b.pdf")
        c.commit()
        c.close()
        main._dokument_loeschen_sync(1, pid, did)
        c = database.get_db()
        self.assertEqual(c.execute("SELECT COUNT(*) FROM projekt_texte WHERE project_id = ?", (pid,)).fetchone()[0], 0)
        c.close()

    def test_chatbot_und_pipeline_bekommen_den_kontext_byte_gleich(self):
        """Chatbot-Werkzeuge (Bildliste: erste 200 Zeichen, Bilddetail: voll), Chat-Verify und der Pipeline-Trichter
        des Chatbots lesen den Kontext jetzt aus projekt_texte — derselbe Text wie vorher in der Bildzeile."""
        import inkluagent.tools.project as tp
        import inkluagent.tools.altext as ta
        import inkluagent.adapters.inkludocs as ad
        db = os.environ["INKLUDOCS_DB"]
        with mock.patch.object(tp, "_DB_PATH", db), mock.patch.object(ta, "_DB_PATH", db):
            liste = tp.list_project_images(self.pid, 1)
            self.assertTrue(liste["ok"], liste)
            self.assertEqual(liste["result"]["images"][0]["context_text"], KAPITEL[:200])
            detail = tp.get_image_metadata(self.bild_ids[0], self.pid, 1)
            self.assertEqual(detail["result"]["context_text"], KAPITEL)
            gesehen = {}
            import pipelines.v4.orchestrator as orch
            with mock.patch.object(orch, "verify_alt_text_extern", side_effect=lambda *a, **k: gesehen.update(k) or None), \
                 mock.patch.dict(os.environ, {"INKLUAGENT_VERIFY": "on"}):
                ta._verify_gegen_bild(self.bild_ids[0], self.pid, "Ein fiktiver Text")
            self.assertEqual(gesehen.get("enriched_context"), KAPITEL)
        aufruf = {}

        def generieren(*a, **k):
            aufruf["kontext"] = a[1]
            return {"alt_text": "fiktiv", "bildtyp": "foto", "konfidenz": "hoch", "langbeschreibung": ""}
        with mock.patch.object(ad.billing, "aktion_pruefung", return_value={"erlaubt": True, "preis": 0, "verfuegbar": 99}), \
             mock.patch("pdf_processor.generate_alt_text", side_effect=generieren):
            ad.run_pipeline_for_image(self.bild_ids[1], self.pid, 1)
        self.assertEqual(aufruf.get("kontext"), KAPITEL)
        self.assertFalse(any("context_text" in b for b in ad.get_project_context(self.pid, 1)["images"]))

    def test_taegliche_texte_pruefung(self):
        """Befund 3 (Entwicklung): die Datenpruefung laeuft im Tageslauf und haelt ihr Ergebnis in system_kv fest."""
        erg = main._texte_pruefung_tageslauf()
        self.assertTrue(erg.get("ok"), erg)
        c = database.get_db()
        import json
        gespeichert = json.loads(c.execute("SELECT value FROM system_kv WHERE key = 'texte_pruefung'").fetchone()[0])
        c.close()
        self.assertTrue(gespeichert["ok"])
        self.assertEqual(gespeichert["ins_leere"], 0)

    def test_bildliste_spalten_tupel(self):
        """Hinweis 5 (Entwicklung): Spaltenliste als fertig gebautes Tupel (threadsicher ohne Sperre)."""
        c = database.get_db()
        sp = main._bildliste_spalten(c)
        c.close()
        self.assertIsInstance(sp, tuple)
        self.assertEqual(len(sp), len(set(sp)))
        self.assertFalse(SCHWER & set(sp))

    def test_struktur_kennzahlen_gemerkt(self):
        c = database.get_db()
        doc = dict(c.execute("SELECT * FROM documents WHERE id = ?", (self.did,)).fetchone())
        erst = tagging_api._struktur_daten(c, doc)
        self.assertEqual(erst["seiten"], 2)
        doc = dict(c.execute("SELECT * FROM documents WHERE id = ?", (self.did,)).fetchone())
        self.assertTrue(doc["struktur_stand"])
        with mock.patch.object(tagging_api.pdf_tagging, "tag_statistik", side_effect=AssertionError("nicht neu rechnen")):
            self.assertEqual(tagging_api._struktur_daten(c, doc), erst)   # aus dem Gedaechtnis
        time.sleep(0.01)
        _pdf(self.pdf, seiten=3)                                          # neue Datei -> anderer Stempel
        neu = tagging_api._struktur_daten(c, doc)
        self.assertEqual(neu["seiten"], 3)
        c.close()


if __name__ == "__main__":
    unittest.main()
