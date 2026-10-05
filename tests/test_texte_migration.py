"""Texte einmal je Projekt (05.10.2026, docs/ANSICHTEN_LEISTUNG.md): Schema, Rueckfall, Migration (Probe, Phase A,
Pruefung, Phase B, Rueckweg), Aufraeumen und Mandantentrennung — auf einer WEGWERF-Datenbank (eigene Datei in /tmp).
    docker exec -w /app <container> python3 -m unittest /app/tests/test_texte_migration.py
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

TMP = tempfile.mkdtemp(prefix="texte_migration_")
DB = os.path.join(TMP, "test.db")
os.environ["INKLUDOCS_DB"] = DB   # vor dem Import von database/projekt_texte setzen
sys.path.insert(0, "/app")
import database  # noqa: E402
import projekt_texte  # noqa: E402

KAPITEL = "Kapiteltext ohne Ueberschrift. " * 4000           # rund 124.000 Zeichen, wie der PDFix-Kapitelkontext
SEITE1 = "Seite 1: Text in Lesereihenfolge.\nZweite Zeile mit Umlauten äöüß und „Anführungszeichen“."
SEITE2 = "Seite 2: anderer Text."


def verbindung():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


class TexteMigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()

    def setUp(self):
        self.c = verbindung()
        self.c.execute("DELETE FROM images")
        self.c.execute("DELETE FROM projekt_texte")
        self.c.execute("DELETE FROM documents")
        self.c.execute("DELETE FROM projects")
        self.c.execute("INSERT OR IGNORE INTO users (id, email, password_hash, display_name) VALUES (1, 'test@example.invalid', 'x', 'Test')")
        self.c.commit()

    def tearDown(self):
        self.c.close()

    def _projekt(self, name="Testprojekt (fiktiv)"):
        cur = self.c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type) VALUES (1, 'a.pdf', '', ?, 'pdf', 'pdf')", (name,))
        pid = cur.lastrowid
        did = self.c.execute("INSERT INTO documents (project_id, doc_index, original_filename) VALUES (?, 1, 'a.pdf')", (pid,)).lastrowid
        return pid, did

    def _altbild(self, pid, did, nr, kontext, seitentext, seite=1):
        """Bildzeile wie VOR dem Umbau: Text in der Zeile, kein Verweis."""
        return self.c.execute(
            "INSERT INTO images (project_id, document_id, page_number, image_index, image_path, context_text, page_text) "
            "VALUES (?, ?, ?, ?, '/tmp/x.png', ?, ?)", (pid, did, seite, nr, kontext, seitentext)).lastrowid

    def _bild(self, iid):
        return self.c.execute("SELECT * FROM images WHERE id = ?", (iid,)).fetchone()

    # --- Schema -------------------------------------------------------------------------------------------------
    def test_schema_angelegt(self):
        spalten = {r[1] for r in self.c.execute("PRAGMA table_info(images)")}
        self.assertTrue({"kontext_id", "seitentext_id", "context_text", "page_text"} <= spalten)
        self.assertTrue({"struktur_json", "struktur_stand"} <= {r[1] for r in self.c.execute("PRAGMA table_info(documents)")})
        indizes = {r[1] for r in self.c.execute("SELECT type, name FROM sqlite_master WHERE type = 'index'")}
        self.assertTrue({"idx_images_document", "idx_images_kontext", "idx_images_seitentext"} <= indizes)
        plan = " ".join(str(r[3]) for r in self.c.execute("EXPLAIN QUERY PLAN SELECT COUNT(*) FROM images WHERE document_id = 1"))
        self.assertIn("idx_images_document", plan)
        database.init_db()   # zweiter Lauf: idempotent, kein Fehler

    # --- Rueckfall ----------------------------------------------------------------------------------------------
    def test_rueckfall_auf_alte_spalte(self):
        pid, did = self._projekt()
        a = self._altbild(pid, did, 1, KAPITEL, SEITE1)
        b = self.c.execute("INSERT INTO images (project_id, document_id, page_number, image_index, image_path) VALUES (?, ?, 1, 2, '/x')",
                           (pid, did)).lastrowid
        self.c.execute("UPDATE images SET context_text = NULL WHERE id = ?", (b,))
        self.assertEqual(projekt_texte.bild_kontext(self.c, self._bild(a)), KAPITEL)
        self.assertEqual(projekt_texte.bild_seitentext(self.c, self._bild(a)), SEITE1)
        self.assertIsNone(projekt_texte.bild_kontext(self.c, self._bild(b)))   # NULL bleibt NULL (wie vorher)
        self.assertIsNone(projekt_texte.bild_kontext(self.c, {"context_text": None}))   # dict ohne Spalte

    # --- Migration ----------------------------------------------------------------------------------------------
    def test_migration_byte_gleich_und_idempotent(self):
        pid, did = self._projekt()
        ids = [self._altbild(pid, did, i, KAPITEL, SEITE1 if i < 3 else SEITE2, seite=1 if i < 3 else 2) for i in range(1, 6)]
        leer = self._altbild(pid, did, 6, "", "")
        self.c.commit()
        vorher = projekt_texte.fingerabdruck(self.c)
        p = projekt_texte.probe(self.c)
        self.assertEqual(p["context_text"]["bilder_ohne_verweis"], 5)
        self.assertEqual(p["context_text"]["verschiedene_texte"], 1)
        self.assertEqual(p["page_text"]["verschiedene_texte"], 2)
        a1 = projekt_texte.phase_a(self.c)
        self.assertEqual(a1["bilder"], 5)
        self.assertEqual(a1["neue_texte"], 3)                       # 1 Kontext + 2 Seitentexte
        self.assertEqual(projekt_texte.phase_a(self.c)["bilder"], 0)   # zweiter Lauf: nichts mehr zu tun
        self.assertEqual(projekt_texte.pruefen(self.c), {"abweichend": 0, "ins_leere": 0, "fremdes_projekt": 0, "noch_ohne_verweis": 0})
        b = projekt_texte.phase_b(self.c)
        self.assertEqual(b, {"context_text": 5, "page_text": 5})
        for i in ids:
            z = self._bild(i)
            self.assertEqual(z["context_text"], "")
            self.assertEqual(projekt_texte.bild_kontext(self.c, z), KAPITEL)
        self.assertEqual(projekt_texte.bild_seitentext(self.c, self._bild(ids[0])), SEITE1)
        self.assertEqual(projekt_texte.bild_seitentext(self.c, self._bild(ids[4])), SEITE2)
        self.assertEqual(projekt_texte.bild_kontext(self.c, self._bild(leer)), "")
        self.assertEqual(projekt_texte.fingerabdruck(self.c), vorher)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM projekt_texte").fetchone()[0], 3)
        self.assertEqual(projekt_texte.pruefen(self.c)["abweichend"], 0)

    def test_phase_b_leert_nur_nachweislich_gleiches(self):
        pid, did = self._projekt()
        a = self._altbild(pid, did, 1, KAPITEL, SEITE1)
        projekt_texte.phase_a(self.c)
        # Text hinter dem Verweis verfaelscht (simulierter Fehler): Pruefung meldet es, Phase B laesst die Zeile stehen
        self.c.execute("UPDATE projekt_texte SET text = 'falsch' WHERE id = (SELECT kontext_id FROM images WHERE id = ?)", (a,))
        self.c.commit()
        self.assertEqual(projekt_texte.pruefen(self.c)["abweichend"], 1)
        projekt_texte.phase_b(self.c)
        self.assertEqual(self._bild(a)["context_text"], KAPITEL)       # nichts verloren
        self.assertEqual(self._bild(a)["page_text"], "")               # Seitentext war gleich -> geleert

    def test_rueckweg(self):
        pid, did = self._projekt()
        a = self._altbild(pid, did, 1, KAPITEL, SEITE1)
        projekt_texte.phase_a(self.c)
        projekt_texte.phase_b(self.c)
        self.assertEqual(projekt_texte.zurueck(self.c), {"context_text": 1, "page_text": 1})
        z = self._bild(a)
        self.assertEqual((z["context_text"], z["page_text"]), (KAPITEL, SEITE1))

    def test_mandantentrennung_und_aufraeumen(self):
        p1, d1 = self._projekt("Projekt eins (fiktiv)")
        p2, d2 = self._projekt("Projekt zwei (fiktiv)")
        a = self._altbild(p1, d1, 1, KAPITEL, SEITE1)
        b = self._altbild(p1, d1, 2, "anderer Kontext", SEITE1)
        c = self._altbild(p2, d2, 1, KAPITEL, SEITE1)
        projekt_texte.phase_a(self.c)
        ka, kc = self._bild(a)["kontext_id"], self._bild(c)["kontext_id"]
        self.assertNotEqual(ka, kc)   # gleicher Text, verschiedene Projekte: je Projekt eine Zeile
        self.assertEqual(projekt_texte.pruefen(self.c)["fremdes_projekt"], 0)
        # Bild a loeschen: sein Kontext gehoert sonst niemandem mehr -> weg; Seitentext nutzt b noch -> bleibt
        self.c.execute("DELETE FROM images WHERE id = ?", (a,))
        self.assertEqual(projekt_texte.texte_aufraeumen(self.c, p1), 1)
        self.assertIsNone(self.c.execute("SELECT 1 FROM projekt_texte WHERE id = ?", (ka,)).fetchone())
        self.assertEqual(projekt_texte.bild_seitentext(self.c, self._bild(b)), SEITE1)
        self.assertEqual(projekt_texte.bild_kontext(self.c, self._bild(c)), KAPITEL)   # Projekt zwei unberuehrt
        projekt_texte.projekt_texte_loeschen(self.c, p2)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM projekt_texte WHERE project_id = ?", (p2,)).fetchone()[0], 0)

    def test_neu_ablegen_einmal_je_projekt(self):
        pid, _ = self._projekt()
        t1 = projekt_texte.text_ablegen(self.c, pid, KAPITEL)
        t2 = projekt_texte.text_ablegen(self.c, pid, KAPITEL)
        self.assertEqual(t1, t2)
        self.assertIsNone(projekt_texte.text_ablegen(self.c, pid, ""))
        self.assertIsNone(projekt_texte.text_ablegen(self.c, pid, None))

    # --- Korrektur nach den Pruefungen (05.10.2026) -------------------------------------------------------------
    def test_nicht_migrierte_bilder_sind_kein_befund(self):
        """Befund 2 (Entwicklung): Ein Bild, das waehrend der Migration von altem Code ohne Verweis angelegt wurde, zaehlt
        getrennt als noch_ohne_verweis, nicht als abweichend; Phase B laesst es unangetastet."""
        pid, did = self._projekt()
        self._altbild(pid, did, 1, KAPITEL, SEITE1)
        projekt_texte.phase_a(self.c)
        spaet = self._altbild(pid, did, 2, "spaeter Kontext von altem Code", SEITE2)   # nach Phase A, ohne Verweis
        p = projekt_texte.pruefen(self.c)
        self.assertEqual((p["abweichend"], p["ins_leere"], p["fremdes_projekt"], p["noch_ohne_verweis"]), (0, 0, 0, 1))
        projekt_texte.phase_b(self.c)
        self.assertEqual(self._bild(spaet)["context_text"], "spaeter Kontext von altem Code")
        self.assertEqual(projekt_texte.bild_kontext(self.c, self._bild(spaet)), "spaeter Kontext von altem Code")
        self.assertEqual(projekt_texte.pruefen(self.c)["abweichend"], 0)

    def test_fingerabdruck_nur_ueber_gemeinsame_bilder(self):
        v = projekt_texte.fingerabdruck_vergleich({1: ("a", "b"), 2: ("c", None), 3: ("d", "e")},
                                                  {1: ("a", "b"), 3: ("d", "e"), 4: ("x", None)})
        self.assertEqual(v, {"verglichen": 2, "abweichend": [], "geloescht": 1, "neu": 1})
        v = projekt_texte.fingerabdruck_vergleich({1: ("a", "b")}, {1: ("a", "anders")})
        self.assertEqual(v["abweichend"], [1])

    def test_verweis_ins_leere_wird_geloggt(self):
        """Befund 3 (Entwicklung): Verweis ins Leere liefert nicht still „kein Kontext“, sondern eine Log-Warnung;
        die Pruefung zaehlt ihn."""
        pid, did = self._projekt()
        a = self._altbild(pid, did, 1, KAPITEL, SEITE1)
        projekt_texte.phase_a(self.c)
        projekt_texte.phase_b(self.c)
        self.c.execute("PRAGMA foreign_keys=OFF")
        self.c.execute("DELETE FROM projekt_texte WHERE id = (SELECT kontext_id FROM images WHERE id = ?)", (a,))
        self.c.commit()
        with self.assertLogs("inkludocs.projekt_texte", level="WARNING") as cm:
            projekt_texte.bild_kontext(self.c, self._bild(a))
        self.assertIn("Verweis ins Leere", cm.output[0])
        self.assertEqual(projekt_texte.pruefen(self.c)["ins_leere"], 1)

    # --- Skript ------------------------------------------------------------------------------------------------
    def test_skript_alles_mit_sicherung(self):
        pid, did = self._projekt()
        for i in range(1, 4):
            self._altbild(pid, did, i, KAPITEL, SEITE1)
        self.c.commit()
        skript = os.path.join("/app", "scripts", "texte_migration.py")
        probe = subprocess.run([sys.executable, skript, "--db", DB], capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
        self.assertIn("Nur Probe", probe.stdout)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM images WHERE kontext_id IS NOT NULL").fetchone()[0], 0)
        lauf = subprocess.run([sys.executable, skript, "--db", DB, "--alles", "--vacuum"], capture_output=True, text=True)
        self.assertEqual(lauf.returncode, 0, lauf.stdout + lauf.stderr)
        self.assertIn("0 Abweichungen", lauf.stdout)
        self.assertTrue([f for f in os.listdir(TMP) if ".bak-pre-texte-" in f], os.listdir(TMP))
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM images WHERE context_text <> ''").fetchone()[0], 0)
        zweit = subprocess.run([sys.executable, skript, "--db", DB, "--alles", "--ohne-sicherung"], capture_output=True, text=True)
        self.assertEqual(zweit.returncode, 0, zweit.stdout + zweit.stderr)   # zweiter Lauf: nichts zu tun, kein Fehler


if __name__ == "__main__":
    unittest.main()
