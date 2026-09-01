# -*- coding: utf-8 -*-
"""Die Start-Migration darf keine Dokumente mehr loeschen oder umnummerieren (01.09.2026).

Hintergrund: Der „Phantom-Dokument-Cleanup" (08.06.2026) loeschte bei jedem
Container-Start Dokumente ohne Bilder und nummerierte den Rest neu. Ein
Word-Dokument, das nur ein Diagramm enthaelt, hat 0 Bilder — und verschwand
beim naechsten Neustart (Steves Projekt 491 auf Staging). Die Neunummerierung
trennte ausserdem doc_index vom Bilderordner (results/<user>/<projekt>/doc<N>).

Der Test legt in einer frischen Datenbank ein Word-Projekt mit zwei Dokumenten
an — eines ohne Bilder — und ruft die Migration auf: Beide Dokumente muessen
bleiben, die Nummern unveraendert. Laeuft im Container: python3 -m unittest tests/test_phantom_cleanup.py
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class PhantomCleanupStumm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "test.db")
        import database
        database.DB_PATH = self.db
        self.database = database
        database.init_db()

    def _conn(self):
        c = sqlite3.connect(self.db)
        c.row_factory = sqlite3.Row
        return c

    def test_word_dokument_ohne_bilder_ueberlebt_die_migration(self):
        c = self._conn()
        uid = c.execute("INSERT INTO users (email, password_hash, display_name) VALUES ('t@example.invalid', 'x', 'T')").lastrowid
        pid = c.execute("INSERT INTO projects (user_id, filename, original_path, status, project_type, tool, name) "
                        "VALUES (?, 'a.docx', '/tmp/a.docx', 'extracted', 'docx', 'word', 'Test')", (uid,)).lastrowid
        d1 = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, extraction_method, total_images) "
                       "VALUES (?, 1, 'mit-bild.docx', 'docx', 1)", (pid,)).lastrowid
        d2 = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, extraction_method, total_images) "
                       "VALUES (?, 2, 'nur-diagramm.docx', 'docx', 0)", (pid,)).lastrowid
        d3 = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, extraction_method, total_images) "
                       "VALUES (?, 3, 'drittes.docx', 'docx', 1)", (pid,)).lastrowid
        c.execute("INSERT INTO images (project_id, document_id, page_number, image_index, image_path, context_text, width, height) "
                  "VALUES (?, ?, 1, 1, ?, '', 10, 10)", (pid, d1, f"/app/data/results/{uid}/{pid}/doc1/p1.png"))
        c.execute("INSERT INTO images (project_id, document_id, page_number, image_index, image_path, context_text, width, height) "
                  "VALUES (?, ?, 1, 2, ?, '', 10, 10)", (pid, d3, f"/app/data/results/{uid}/{pid}/doc3/p2.png"))
        c.commit()
        # Migration erneut laufen lassen — wie bei jedem Container-Start.
        self.database.init_db()
        rows = c.execute("SELECT id, doc_index FROM documents WHERE project_id = ? ORDER BY id", (pid,)).fetchall()
        self.assertEqual([(r["id"], r["doc_index"]) for r in rows], [(d1, 1), (d2, 2), (d3, 3)],
                         "Dokument ohne Bilder wurde geloescht oder Nummern wurden veraendert")

    def test_pdf_dokument_ohne_bilder_bleibt_ebenfalls(self):
        c = self._conn()
        uid = c.execute("INSERT INTO users (email, password_hash, display_name) VALUES ('p@example.invalid', 'x', 'P')").lastrowid
        pid = c.execute("INSERT INTO projects (user_id, filename, original_path, status, project_type, tool, name) "
                        "VALUES (?, 'a.pdf', '/tmp/a.pdf', 'extracted', 'pdf', 'pdf', 'Test')", (uid,)).lastrowid
        d1 = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, extraction_method, total_images) "
                       "VALUES (?, 1, 'leer.pdf', 'fitz', 0)", (pid,)).lastrowid
        c.commit()
        self.database.init_db()
        self.assertEqual(c.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (d1,)).fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
