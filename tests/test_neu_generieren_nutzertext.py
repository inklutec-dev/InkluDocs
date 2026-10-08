"""Neu generieren mit einem Text, den der Nutzer bearbeitet hat (08.10.2026) — auf einer WEGWERF-Datenbank.

Anlass: Der Variationszusatz erklärte den bisherigen Text pauschal zu "kein Beleg". Hatte der Nutzer von Hand
einen Namen oder einen Anlass ergänzt, fiel der beim Neu-Generieren eher weg. Jetzt reichen beide Wege (Knopf
"Neu generieren" in main.py und Chatbot über inkluagent/adapters/inkludocs.py) zusätzlich den zuletzt von der
KI erzeugten Text (images.alt_text) als previous_alt_ki durch; der Orchestrator erkennt daran, was vom Nutzer
stammt (herkunft_vorlage). Ein solches Ergebnis landet nicht im kontoübergreifenden Ergebnis-Cache.
Randfälle (08.10.2026): Feld bewusst geleert und dann Neu generieren (frisch anfangen, keine Hand-Fakten),
Feld geleert und komplett eigenen Text geschrieben (Angaben bleiben), Sammellauf ohne jede Vorlage.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_neu_generieren_nutzertext.py
Immer im EIGENEN Prozess starten (wie test_ki_abholung.py): `database` liest den Pfad beim ersten Import.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="neu_nutzertext_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
sys.path.insert(0, "/app")
os.chdir("/app")
import database  # noqa: E402
FREMDE_DB = os.path.abspath(database.DB_PATH) != os.path.abspath(os.environ["INKLUDOCS_DB"])
if not FREMDE_DB:
    database.init_db()
    import main  # noqa: E402
    import pdf_processor  # noqa: E402
    from inkluagent.adapters import inkludocs as adapter  # noqa: E402
    from pipelines.v4.orchestrator import _variation_suffix, herkunft_vorlage  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

NUTZER = {"id": 1, "email": "test@example.invalid", "is_admin": 0}
KI_TEXT = "Karte zum Landabtausch im Areal Beispielfeld: Ein Fluss teilt das Gebiet (fiktiv)."
VON_HAND = "Karte zum Landabtausch im Areal Beispielfeld, Vorlage für die Gemeindeversammlung: Ein Fluss teilt das Gebiet (fiktiv)."
ERGEBNIS = {"alt_text": "Karte des Areals Beispielfeld (fiktiv)", "bildtyp": "karte", "konfidenz": "hoch",
            "langbeschreibung": "", "needs_review": False, "pipeline_steps": "", "validation_result": ""}


class NeuGenerierenNutzertextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FREMDE_DB:
            raise unittest.SkipTest("database ist schon mit einer anderen Datenbank geladen — bitte im eigenen Prozess starten")
        c = database.get_db()
        c.execute("INSERT OR IGNORE INTO users (id, email, password_hash, display_name) VALUES (1, ?, 'x', 'Test')",
                  (NUTZER["email"],))
        cls.pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                            "VALUES (1, 'a.pdf', '', 'Nutzertext (fiktiv)', 'pdf', 'pdf', 'completed')").lastrowid
        cls.bild = os.path.join(TMP, "b.png")
        with open(cls.bild, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        cls.iid = c.execute("INSERT INTO images (project_id, page_number, image_index, image_path, status, alt_text) "
                            "VALUES (?, 1, 1, ?, 'done', ?)", (cls.pid, cls.bild, KI_TEXT)).lastrowid
        c.commit()
        c.close()
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def setUp(self):
        main.app.dependency_overrides[main.get_current_user] = lambda: NUTZER
        self.generiert = mock.Mock(return_value=dict(ERGEBNIS))
        self.patches = [
            mock.patch.object(main, "generate_alt_text", self.generiert),
            mock.patch.object(main.billing, "verbuche"),
            mock.patch.object(main.billing, "aktion_pruefung", return_value={"erlaubt": True, "preis": 1, "verfuegbar": 9}),
            mock.patch.object(main, "tageslimit_wache", return_value=None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        main.app.dependency_overrides.clear()

    def _setze(self, alt_text, bearbeitet):
        c = database.get_db()
        c.execute("UPDATE images SET alt_text = ?, alt_text_edited = ? WHERE id = ?", (alt_text, bearbeitet, self.iid))
        c.commit()
        c.close()

    def _knopf(self):
        r = self.client.post(f"/api/projects/{self.pid}/regenerate/{self.iid}", json={})
        self.assertEqual(r.status_code, 200)
        args = self.generiert.call_args.args
        return args[9], args[11]   # previous_alt, previous_alt_ki

    def test_knopf_reicht_bearbeiteten_und_ki_text_durch(self):
        self._setze(KI_TEXT, VON_HAND)
        self.assertEqual(self._knopf(), (VON_HAND, KI_TEXT))

    def test_knopf_ohne_bearbeitung(self):
        self._setze(KI_TEXT, None)
        self.assertEqual(self._knopf(), (KI_TEXT, KI_TEXT))

    def test_feld_bewusst_geleert_dann_neu_generieren(self):
        """Randfall 1: Feld geleert ('') und Neu generieren = frisch anfangen. Keine Hand-Fakten; der alte
        KI-Text geht nur als Abgrenzungs-Vorlage mit."""
        self._setze(KI_TEXT, "")
        vorlage, ki = self._knopf()
        self.assertEqual((vorlage, ki), (KI_TEXT, KI_TEXT))
        self.assertEqual(herkunft_vorlage(vorlage, ki), ("ki", []))
        zusatz = " ".join(_variation_suffix(vorlage, ki).split())
        self.assertIn("Den bisherigen Text hat die KI geschrieben. Er zeigt dir nur, wovon sich die neue Fassung "
                      "abheben soll, und ist kein Beleg", zusatz)
        self.assertNotIn("Nutzer selbst", zusatz)
        self.assertNotIn("Ergänzt oder geändert hat er", zusatz)

    def test_feld_geleert_und_eigenen_text_geschrieben(self):
        """Randfall 2: Feld geleert und einen komplett eigenen Text hineingeschrieben. Seine Angaben gelten,
        die Formulierung darf sich ändern."""
        eigen = "Lageplan für die Gemeindeversammlung am 14. November, erstellt von Erika Muster (fiktiv)."
        self._setze(KI_TEXT, eigen)
        vorlage, ki = self._knopf()
        self.assertEqual((vorlage, ki), (eigen, KI_TEXT))
        self.assertEqual(herkunft_vorlage(vorlage, ki), ("nutzer", []))
        zusatz = " ".join(_variation_suffix(vorlage, ki).split())
        self.assertIn("Den bisherigen Text hat der Nutzer selbst geschrieben. Seine Angaben übernimmst du "
                      "inhaltlich in BEIDE Felder der neuen Fassung, auch wenn das Bild sie nicht zeigt", zusatz)
        self.assertIn("nur die Formulierung darf sich ändern", zusatz)
        self.assertNotIn("kein Beleg", zusatz)

    def test_knopf_handtext_ohne_ki_text(self):
        self._setze(None, "Eigener Text des Nutzers (fiktiv).")
        self.assertEqual(self._knopf(), ("Eigener Text des Nutzers (fiktiv).", ""))

    def test_sammellauf_gibt_keinen_alten_text_mit(self):
        """Randfall 3: Der Sammellauf (Generieren, Alle neu generieren, je Projekt oder Dokument) gibt bewusst keinen
        alten Text mit und ueberschreibt alles; ein Hand-Text wandert nach alt_text_vorher (Michael Karbe 01.09.2026)."""
        self._setze(KI_TEXT, VON_HAND)
        c = database.get_db()
        c.execute("UPDATE images SET status = 'pending', alt_text_vorher = NULL WHERE id = ?", (self.iid,))
        c.commit()
        c.close()
        import asyncio
        asyncio.run(main._process_project(self.pid, NUTZER["id"], force=True))
        self.assertGreaterEqual(self.generiert.call_count, 1)
        for aufruf in self.generiert.call_args_list:
            self.assertEqual(aufruf.args[9], "")          # previous_alt leer
            self.assertEqual(len(aufruf.args), 11)        # kein previous_alt_ki
            self.assertEqual(aufruf.kwargs, {})
            self.assertNotIn(VON_HAND, repr(aufruf))
            self.assertNotIn(KI_TEXT, repr(aufruf))
        c = database.get_db()
        zeile = c.execute("SELECT alt_text, alt_text_edited, alt_text_vorher FROM images WHERE id = ?", (self.iid,)).fetchone()
        c.close()
        self.assertEqual(zeile["alt_text_vorher"], VON_HAND)
        self.assertIsNone(zeile["alt_text_edited"])
        self.assertEqual(zeile["alt_text"], ERGEBNIS["alt_text"])

    def test_chatbot_weg_reicht_ki_text_durch(self):
        self._setze(KI_TEXT, VON_HAND)
        with mock.patch.object(adapter.billing, "aktion_pruefung", return_value={"erlaubt": True}), \
                mock.patch.object(adapter.billing, "verbuche"), \
                mock.patch("pdf_processor.generate_alt_text", return_value=dict(ERGEBNIS)) as gen:
            adapter.run_pipeline_for_image(self.iid, self.pid, NUTZER["id"])
        self.assertEqual(gen.call_args.kwargs["previous_alt"], VON_HAND)
        self.assertEqual(gen.call_args.kwargs["previous_alt_ki"], KI_TEXT)

    def test_ergebnis_mit_nutzerangaben_nicht_im_cache(self):
        with mock.patch.object(pdf_processor, "_v4_entry", return_value=dict(ERGEBNIS)) as lauf, \
                mock.patch("cache.set_cached") as gespeichert:
            pdf_processor.generate_alt_text(self.bild, "", "karte", force_regenerate=True, temperature=0.5,
                                            previous_alt=VON_HAND, previous_alt_ki=KI_TEXT)
            self.assertEqual(lauf.call_args.kwargs["previous_alt_ki"], KI_TEXT)
            gespeichert.assert_not_called()
            pdf_processor.generate_alt_text(self.bild, "", "karte", force_regenerate=True, temperature=0.5,
                                            previous_alt=KI_TEXT, previous_alt_ki=KI_TEXT)
            gespeichert.assert_called_once()
            pdf_processor.generate_alt_text(self.bild, "", "karte", force_regenerate=True)   # Sammellauf
            self.assertEqual(gespeichert.call_count, 2)


if __name__ == "__main__":
    unittest.main()
