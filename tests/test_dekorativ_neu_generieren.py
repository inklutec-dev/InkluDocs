"""„Neu generieren“ mit Bildtyp „dekorativ“ (Befund 09.10.2026: 500, v4 hatte keinen Inventar-Builder fuer dekorativ) —
auf einer WEGWERF-Datenbank, ohne Modell:
  - Knopf mit Bildtyp dekorativ: 200, Feld leer ('' = Entscheidung am Bild, auch bei mitgebrachtem Text), Bildtyp
    dekorativ, kein Modellaufruf, kein Credit; der Export schreibt das Bild als dekorativ.
  - Danach „Neu generieren“ mit anderem Bildtyp: wieder ein KI-Text, Feld wieder offen (NULL).
  - Langbeschreibung bzw. Chatbot bei einem dekorativen Bild: keine Vorgabe „dekorativ“, die KI stuft frisch ein.
  - Orchestrator: Vorgabe dekorativ liefert das Ergebnis ohne call_with_schema.
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_dekorativ_neu_generieren.py -v
Immer im EIGENEN Prozess starten: `database` liest den Pfad beim ersten Import.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="deko_neu_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)
import database  # noqa: E402
FREMDE_DB = os.path.abspath(database.DB_PATH) != os.path.abspath(os.environ["INKLUDOCS_DB"])
if not FREMDE_DB:
    database.init_db()
    import main  # noqa: E402
    from pipelines.v4 import orchestrator  # noqa: E402
    from inkluagent.adapters import inkludocs as adapter  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

NUTZER = {"id": 1, "email": "test@example.invalid", "is_admin": 0}
MITGEBRACHT = "Logo des Nachbarschaftsvereins Musterstadt (fiktiv)."
KI = {"alt_text": "Grünes Blatt auf hellem Kreis (fiktiv)", "bildtyp": "logo", "konfidenz": "hoch",
      "langbeschreibung": "", "needs_review": False, "pipeline_steps": "", "validation_result": ""}


class DekorativNeuGenerieren(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FREMDE_DB:
            raise unittest.SkipTest("database ist schon mit einer anderen Datenbank geladen — eigenen Prozess nutzen")
        from PIL import Image
        c = database.get_db()
        c.execute("INSERT OR IGNORE INTO users (id, email, password_hash, display_name) VALUES (1, ?, 'x', 'Test')",
                  (NUTZER["email"],))
        cls.pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                            "VALUES (1, 'a.pdf', '', 'Dekorativ (fiktiv)', 'pdf', 'pdf', 'completed')").lastrowid
        cls.bild = os.path.join(TMP, "logo.png")
        Image.new("RGB", (60, 60), (20, 110, 50)).save(cls.bild)
        cls.iid = c.execute("INSERT INTO images (project_id, page_number, image_index, image_path, status, alt_text, "
                            "original_alt) VALUES (?, 1, 1, ?, 'pending', '', ?)", (cls.pid, cls.bild, MITGEBRACHT)).lastrowid
        c.commit()
        c.close()
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def setUp(self):
        main.app.dependency_overrides[main.get_current_user] = lambda: NUTZER
        self.verbuche = mock.Mock()
        self.modell = mock.Mock(side_effect=AssertionError("kein Modellaufruf erwartet"))
        self.patches = [
            mock.patch.object(main.billing, "verbuche", self.verbuche),
            mock.patch.object(main.billing, "aktion_pruefung", return_value={"erlaubt": True, "preis": 1, "verfuegbar": 9}),
            mock.patch.object(main, "tageslimit_wache", return_value=None),
            mock.patch.object(orchestrator, "call_with_schema", self.modell),
            mock.patch("cache.get_cached", return_value=None),
            mock.patch("cache.set_cached"),
            mock.patch("cache.evict_by_content_hash", return_value=0),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        main.app.dependency_overrides.clear()

    def _zeile(self):
        c = database.get_db()
        try:
            return dict(c.execute("SELECT * FROM images WHERE id = ?", (self.iid,)).fetchone())
        finally:
            c.close()

    def test_1_bildtyp_dekorativ(self):
        r = self.client.post(f"/api/projects/{self.pid}/regenerate/{self.iid}", json={"image_type": "dekorativ"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["bildtyp"], r.json()["alt_text"]), ("dekorativ", ""))
        z = self._zeile()
        self.assertEqual((z["image_type"], z["alt_text"], z["alt_text_edited"], z["status"]), ("dekorativ", "", "", "done"))
        self.modell.assert_not_called()
        self.verbuche.assert_not_called()                       # kein Modell, kein Credit
        self.assertEqual(main._display_alt_text(z), "")          # das mitgebrachte Original kommt nicht zurueck ins Feld
        self.assertEqual(main._exportable_alt_text(z), "dekorativ")

    def test_2_danach_anderer_bildtyp(self):
        with mock.patch.object(main, "generate_alt_text", return_value=dict(KI)) as gen:
            r = self.client.post(f"/api/projects/{self.pid}/regenerate/{self.iid}", json={"image_type": "logo"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(gen.call_args.args[2], "logo")
        z = self._zeile()
        self.assertEqual((z["image_type"], z["alt_text"], z["alt_text_edited"]), ("logo", KI["alt_text"], None))
        self.verbuche.assert_called_once()

    def test_3_langbeschreibung_bei_dekorativem_bild(self):
        c = database.get_db()
        c.execute("UPDATE images SET image_type = 'dekorativ', alt_text = '', alt_text_edited = NULL WHERE id = ?", (self.iid,))
        c.commit()
        c.close()
        with mock.patch.object(main, "generate_alt_text", return_value=dict(KI)) as gen:
            r = self.client.post(f"/api/projects/{self.pid}/regenerate/{self.iid}", json={"long_description": True})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsNone(gen.call_args.args[2])                 # frische Einstufung statt Vorgabe dekorativ

    def test_4_orchestrator_ohne_modell(self):
        erg = orchestrator.generate_alt_text_v4(self.bild, image_type_override="dekorativ", original_alt=MITGEBRACHT)
        self.assertEqual((erg["bildtyp"], erg["alt_text"], erg["needs_review"], erg["ohne_ki"]), ("dekorativ", "", False, True))
        self.modell.assert_not_called()

    def test_5_chatbot_gibt_dekorativ_nicht_als_vorgabe(self):
        c = database.get_db()
        c.execute("UPDATE images SET image_type = 'dekorativ', alt_text = '', alt_text_edited = NULL WHERE id = ?", (self.iid,))
        c.commit()
        c.close()
        with mock.patch("pdf_processor.generate_alt_text", return_value=dict(KI)) as gen, \
             mock.patch.object(adapter.billing, "aktion_pruefung", return_value={"erlaubt": True, "preis": 5, "verfuegbar": 9}), \
             mock.patch.object(adapter.billing, "verbuche"):
            erg = adapter.run_pipeline_for_image(self.iid, self.pid, NUTZER["id"])
        self.assertEqual(erg["alt_text"], KI["alt_text"])
        self.assertIsNone(gen.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
