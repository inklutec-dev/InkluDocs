"""Texte eines Projekts EINMAL speichern (Ansichtswechsel-Umbau, 05.10.2026, docs/ANSICHTEN_LEISTUNG.md).

Hintergrund: Der KI-Kontext eines Bildes (images.context_text) und der Seitentext (images.page_text) standen bis
zum 05.10.2026 in JEDER Bildzeile. Beim PDFix-Weg ist der Kontext der ganze Abschnitt um die Figure — in PDFs ohne
Ueberschrift-Tags das ganze Dokument. So lag derselbe Text von bis zu 380.000 Zeichen 71-mal in der Datenbank
(Prod: 35,7 Mio. Zeichen Kontext, davon 3,2 Mio. verschieden) und ging bei jedem Ansichtswechsel an den Browser.

Jetzt: Tabelle `projekt_texte` (je Projekt, Schluessel SHA-256 des Textes); das Bild verweist darauf
(images.kontext_id, images.seitentext_id). Der Text bleibt BYTE-GLEICH — KI-Eingabe, Cache-Schluessel
(cache.build_cache_key mit enriched_context) und Credits aendern sich nicht.

Rueckfall: Die alten Spalten bleiben bestehen. Solange ein Bild keinen Verweis hat (Altbestand vor der Migration),
liest `bild_kontext` die alte Spalte — Code und Datenbank lassen sich so in beliebiger Reihenfolge ausrollen.
Migration und Rueckweg: scripts/texte_migration.py (Funktionen unten, damit die Tests sie direkt pruefen).
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Optional

# Spaltenpaare (alte Spalte, Verweis-Spalte) in images.
PAARE = (("context_text", "kontext_id"), ("page_text", "seitentext_id"))


def schema_anlegen(conn) -> None:
    """Tabelle und Indizes anlegen (idempotent; database.init_db ruft es NACH _migrate_columns, weil die
    Verweis-Spalten dort per ALTER entstehen)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projekt_texte (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            text TEXT NOT NULL,
            zeichen INTEGER NOT NULL,
            angelegt TEXT DEFAULT (datetime('now')),
            UNIQUE (project_id, sha256),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        )""")
    spalten = {r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
    if "kontext_id" in spalten:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_kontext ON images(kontext_id)")
    if "seitentext_id" in spalten:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_seitentext ON images(seitentext_id)")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_ablegen(conn, project_id: int, text: Optional[str]) -> Optional[int]:
    """Text einmal je Projekt ablegen und seine id liefern; leer/None -> None (kein Verweis, wie bisher '')."""
    if not text:
        return None
    sha = _sha(text)
    conn.execute("INSERT OR IGNORE INTO projekt_texte (project_id, sha256, text, zeichen) VALUES (?, ?, ?, ?)",
                 (project_id, sha, text, len(text)))
    row = conn.execute("SELECT id FROM projekt_texte WHERE project_id = ? AND sha256 = ?", (project_id, sha)).fetchone()
    return int(row[0])


def _wert(img, schluessel):
    """Feld aus sqlite3.Row oder dict; fehlt die Spalte (Altschema, Tests), None."""
    try:
        keys = img.keys()
    except AttributeError:
        return None
    return img[schluessel] if schluessel in keys else None


def _text_zu(conn, img, alt_spalte: str, id_spalte: str):
    tid = _wert(img, id_spalte)
    if tid:
        row = conn.execute("SELECT text FROM projekt_texte WHERE id = ?", (tid,)).fetchone()
        if row is not None:
            return row[0]
    # Altbestand (vor der Migration) oder kein Text: unveraendert die alte Spalte (auch None bleibt None)
    return _wert(img, alt_spalte)


def bild_kontext(conn, img):
    """KI-Kontext eines Bildes, byte-gleich zum frueheren images.context_text."""
    return _text_zu(conn, img, "context_text", "kontext_id")


def bild_seitentext(conn, img):
    """Seitentext eines Bildes, byte-gleich zum frueheren images.page_text."""
    return _text_zu(conn, img, "page_text", "seitentext_id")


def kontext_sql(alias: str = "i", tabelle_alias: str = "kt") -> tuple[str, str]:
    """(Ausdruck, JOIN) fuer Abfragen, die den Kontext gleich mitlesen (Chat-Werkzeuge):
    Ausdruck liefert den Text hinter dem Verweis, sonst die alte Spalte."""
    return (f"COALESCE({tabelle_alias}.text, {alias}.context_text)",
            f"LEFT JOIN projekt_texte {tabelle_alias} ON {tabelle_alias}.id = {alias}.kontext_id")


def texte_aufraeumen(conn, project_id: int) -> int:
    """Texte des Projekts loeschen, auf die kein Bild mehr verweist (nach Dokument-/Bild-Loeschen und nach der
    Neu-Extraktion beim Tagging). Loescht nie einen Text, auf den noch ein Bild zeigt. Rueckgabe: Zahl."""
    cur = conn.execute(
        "DELETE FROM projekt_texte WHERE project_id = ? "
        "AND id NOT IN (SELECT kontext_id FROM images WHERE project_id = ? AND kontext_id IS NOT NULL) "
        "AND id NOT IN (SELECT seitentext_id FROM images WHERE project_id = ? AND seitentext_id IS NOT NULL)",
        (project_id, project_id, project_id))
    return cur.rowcount or 0


def projekt_texte_loeschen(conn, project_id: int) -> None:
    """Alle Texte eines Projekts (Projekt bzw. Konto wird geloescht; nicht auf ON DELETE CASCADE verlassen,
    vgl. database.delete_user_data)."""
    conn.execute("DELETE FROM projekt_texte WHERE project_id = ?", (project_id,))


# ---------------------------------------------------------------------------
# Migration der Bestandsdaten (scripts/texte_migration.py)
# ---------------------------------------------------------------------------

@contextmanager
def _transaktion(conn):
    """Eigene, kurze Schreib-Transaktion (BEGIN IMMEDIATE), unabhaengig vom isolation_level der Verbindung."""
    alt = conn.isolation_level
    if conn.in_transaction:
        conn.commit()
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.isolation_level = alt


def probe(conn) -> dict:
    """Was wuerde Phase A/B aendern? Nur lesend."""
    erg = {}
    for alt, ref in PAARE:
        r = conn.execute(f"SELECT COUNT(*), COALESCE(SUM(LENGTH({alt})), 0) FROM images "
                         f"WHERE {ref} IS NULL AND COALESCE({alt}, '') <> ''").fetchone()
        verschieden = conn.execute(f"SELECT COUNT(*), COALESCE(SUM(LENGTH(t)), 0) FROM (SELECT DISTINCT project_id, {alt} AS t "
                                   f"FROM images WHERE {ref} IS NULL AND COALESCE({alt}, '') <> '')").fetchone()
        erg[alt] = {"bilder_ohne_verweis": r[0], "zeichen": r[1], "verschiedene_texte": verschieden[0],
                    "zeichen_nachher": verschieden[1]}
        gefuellt = conn.execute(f"SELECT COUNT(*) FROM images WHERE {ref} IS NOT NULL AND COALESCE({alt}, '') <> ''").fetchone()[0]
        erg[alt]["mit_verweis_noch_gefuellt"] = gefuellt
    erg["projekt_texte"] = conn.execute("SELECT COUNT(*), COALESCE(SUM(zeichen), 0) FROM projekt_texte").fetchone()[:]
    return erg


def phase_a(conn, fortschritt=None) -> dict:
    """Verweise setzen, alte Spalten bleiben gefuellt. Je Projekt eine Transaktion (kurze Schreibsperren),
    idempotent: nur Bilder ohne Verweis mit nicht-leerem Text."""
    bilder = texte = 0
    projekte = [r[0] for r in conn.execute(
        "SELECT DISTINCT project_id FROM images WHERE (kontext_id IS NULL AND COALESCE(context_text, '') <> '') "
        "OR (seitentext_id IS NULL AND COALESCE(page_text, '') <> '') ORDER BY project_id").fetchall()]
    for pid in projekte:
        vorher = conn.execute("SELECT COUNT(*) FROM projekt_texte WHERE project_id = ?", (pid,)).fetchone()[0]
        with _transaktion(conn):
            zeilen = conn.execute(
                "SELECT id, context_text, page_text, kontext_id, seitentext_id FROM images WHERE project_id = ? AND "
                "((kontext_id IS NULL AND COALESCE(context_text, '') <> '') OR (seitentext_id IS NULL AND COALESCE(page_text, '') <> ''))",
                (pid,)).fetchall()
            for z in zeilen:
                kid = z[3] if z[3] is not None else text_ablegen(conn, pid, z[1])
                sid = z[4] if z[4] is not None else text_ablegen(conn, pid, z[2])
                conn.execute("UPDATE images SET kontext_id = ?, seitentext_id = ? WHERE id = ?", (kid, sid, z[0]))
                bilder += 1
        texte += conn.execute("SELECT COUNT(*) FROM projekt_texte WHERE project_id = ?", (pid,)).fetchone()[0] - vorher
        if fortschritt:
            fortschritt(pid, len(zeilen))
    return {"projekte": len(projekte), "bilder": bilder, "neue_texte": texte}


def pruefen(conn) -> dict:
    """Jeder Verweis muss byte-gleich den Text der alten Spalte liefern (solange die alte Spalte gefuellt ist),
    und kein Verweis darf ins Leere zeigen. Ergebnis 0/0 ist Pflicht vor Phase B."""
    abweichend = conn.execute(
        "SELECT COUNT(*) FROM images i LEFT JOIN projekt_texte k ON k.id = i.kontext_id "
        "LEFT JOIN projekt_texte s ON s.id = i.seitentext_id "
        "WHERE (COALESCE(i.context_text, '') <> '' AND (k.text IS NULL OR k.text <> i.context_text OR k.project_id <> i.project_id)) "
        "OR (COALESCE(i.page_text, '') <> '' AND (s.text IS NULL OR s.text <> i.page_text OR s.project_id <> i.project_id))").fetchone()[0]
    ins_leere = conn.execute(
        "SELECT COUNT(*) FROM images i WHERE (i.kontext_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM projekt_texte k WHERE k.id = i.kontext_id)) "
        "OR (i.seitentext_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM projekt_texte s WHERE s.id = i.seitentext_id))").fetchone()[0]
    fremd = conn.execute(
        "SELECT COUNT(*) FROM images i JOIN projekt_texte k ON k.id = i.kontext_id WHERE k.project_id <> i.project_id").fetchone()[0]
    fremd += conn.execute(
        "SELECT COUNT(*) FROM images i JOIN projekt_texte s ON s.id = i.seitentext_id WHERE s.project_id <> i.project_id").fetchone()[0]
    ohne_verweis = conn.execute(
        "SELECT COUNT(*) FROM images WHERE (kontext_id IS NULL AND COALESCE(context_text, '') <> '') "
        "OR (seitentext_id IS NULL AND COALESCE(page_text, '') <> '')").fetchone()[0]
    return {"abweichend": abweichend, "ins_leere": ins_leere, "fremdes_projekt": fremd, "noch_ohne_verweis": ohne_verweis}


def phase_b(conn) -> dict:
    """Alte Spalten leeren — NUR wo der Verweis nachweislich denselben Text liefert (Bedingung im UPDATE selbst)."""
    erg = {}
    with _transaktion(conn):
        for alt, ref in PAARE:
            cur = conn.execute(
                f"UPDATE images SET {alt} = '' WHERE {ref} IS NOT NULL AND COALESCE({alt}, '') <> '' "
                f"AND {alt} = (SELECT text FROM projekt_texte WHERE id = images.{ref} AND project_id = images.project_id)")
            erg[alt] = cur.rowcount
    return erg


def zurueck(conn) -> dict:
    """Rueckweg ohne Wiederherstellen der Sicherung: alte Spalten aus projekt_texte wieder fuellen. Die Verweise
    bleiben (harmlos; ein erneutes Phase B leert wieder)."""
    erg = {}
    with _transaktion(conn):
        for alt, ref in PAARE:
            cur = conn.execute(
                f"UPDATE images SET {alt} = (SELECT text FROM projekt_texte WHERE id = images.{ref}) "
                f"WHERE {ref} IS NOT NULL AND COALESCE({alt}, '') = ''")
            erg[alt] = cur.rowcount
    return erg


def fingerabdruck(conn) -> dict:
    """{image_id: (sha256 Kontext, sha256 Seitentext)} des WIRKSAMEN Textes (Verweis, sonst alte Spalte) — vor und
    nach der Migration verglichen beweist er, dass KI-Eingabe und Anzeige byte-gleich bleiben."""
    expr_k, join_k = kontext_sql("i", "kt")
    rows = conn.execute(
        f"SELECT i.id, {expr_k}, COALESCE(st.text, i.page_text) FROM images i {join_k} "
        "LEFT JOIN projekt_texte st ON st.id = i.seitentext_id").fetchall()

    def h(t):
        return None if t is None else _sha(t)
    return {r[0]: (h(r[1]), h(r[2])) for r in rows}
