"""Persistente Cache-Schicht fuer Alt-Text-Resultate (T6, Phase A v4-Architektur).

Schluessel-Design (siehe prompt-architektur-v4.md §8.2):
- content_hash (SHA-256 der Bild-Bytes) statt image_path
  -> dasselbe Bild in verschiedenen Projekten teilt EINEN Cache-Eintrag
- image_type_override als Salt (User-Override hat Vorrang vor Auto-Klassifikation)
- user_hint als Salt (Chatbot-Modus aendert das Ergebnis)
- pipeline_version als Salt (v3_7 vs v4 sollen NICHT gegenseitig Caches treffen)

NICHT im Cache-Key: enriched_context (Bild-Inhalt ist primaere Erkenntnisquelle),
project_id (sonst keine Cross-Projekt-Hits), timestamps.
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime
from typing import Optional


_DB_PATH = os.environ.get("INKLUDOCS_DB", "/app/data/inkludocs.db")


def _conn() -> sqlite3.Connection:
    """SQLite-Verbindung zur Haupt-DB. WAL-Mode wie in database.py."""
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_schema() -> None:
    """Idempotent: legt Tabelle + Indizes an. Aufgerufen aus database.init_db()."""
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS alt_text_cache (
            cache_key TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            pipeline_version TEXT NOT NULL,
            image_type_override TEXT,
            user_hint_hash TEXT,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_accessed_at TEXT NOT NULL,
            hits INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_alt_text_cache_content
            ON alt_text_cache(content_hash);
        CREATE INDEX IF NOT EXISTS idx_alt_text_cache_accessed
            ON alt_text_cache(last_accessed_at);
    """)
    c.commit()
    c.close()


def _hash_user_hint(user_hint: Optional[str]) -> str:
    """SHA-256-basierter Hash fuer user_hint im Cache-Key.

    NICHT Pythons eingebautes hash() verwenden: das ist seit Python 3.3 randomisiert
    (PYTHONHASHSEED), wuerde bei jedem Container-Restart andere Werte liefern und den
    Cache stillschweigend invalidieren. Bei Chatbot-Variante mit haeufigen user_hints
    waere das ein Performance-Killer ohne sichtbares Symptom.

    SHA-256 ist deterministisch ueber Prozess- und Container-Grenzen hinweg.
    16 Hex-Zeichen (= 64 Bit) genuegen fuer Kollisionsfreiheit bei realistischen Volumen.
    """
    if not user_hint:
        return "no_hint"
    return hashlib.sha256(user_hint.encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    content_hash: str,
    image_type_override: Optional[str],
    user_hint: Optional[str],
    pipeline_version: str = "v3_7",
) -> str:
    """Cache-Key zusammenbauen. Reihenfolge: pipeline:content:type:hinthash."""
    parts = [
        pipeline_version,
        content_hash,
        image_type_override or "auto",
        _hash_user_hint(user_hint),
    ]
    return ":".join(parts)


def get_cached(cache_key: str) -> Optional[dict]:
    """Gibt gecachten Result-Dict zurueck oder None.

    Nebeneffekt: bumpt `hits` und aktualisiert `last_accessed_at` (LRU-Tracking).
    """
    c = _conn()
    row = c.execute(
        "SELECT result_json FROM alt_text_cache WHERE cache_key = ?",
        (cache_key,)
    ).fetchone()
    if not row:
        c.close()
        return None
    c.execute(
        "UPDATE alt_text_cache SET last_accessed_at = ?, hits = hits + 1 WHERE cache_key = ?",
        (datetime.utcnow().isoformat(), cache_key)
    )
    c.commit()
    c.close()
    try:
        return json.loads(row["result_json"])
    except (json.JSONDecodeError, TypeError):
        return None


def set_cached(
    cache_key: str,
    content_hash: str,
    pipeline_version: str,
    image_type_override: Optional[str],
    user_hint: Optional[str],
    result: dict,
) -> None:
    """Speichert Result. INSERT OR REPLACE — bestehende Eintraege werden ueberschrieben,
    hits-Counter bleibt erhalten."""
    now = datetime.utcnow().isoformat()
    c = _conn()
    c.execute(
        """INSERT OR REPLACE INTO alt_text_cache
           (cache_key, content_hash, pipeline_version, image_type_override,
            user_hint_hash, result_json, created_at, last_accessed_at, hits)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                   COALESCE((SELECT hits FROM alt_text_cache WHERE cache_key = ?), 0))""",
        (cache_key, content_hash, pipeline_version, image_type_override,
         _hash_user_hint(user_hint), json.dumps(result), now, now, cache_key)
    )
    c.commit()
    c.close()


def evict_by_content_hash(content_hash: str) -> int:
    """Loescht alle Cache-Eintraege fuer diesen content_hash — alle Varianten
    (verschiedene image_type_override, user_hint, pipeline_version).

    Wird vom regenerate_image-Endpoint gerufen: wenn der User explizit "neu generieren"
    klickt, soll das wirklich eine frische Pipeline-Ausfuehrung erzwingen, auch fuer
    spaetere Anfragen mit anderen Override-Varianten desselben Bilds.

    Return: Anzahl geloeschter Zeilen.
    """
    c = _conn()
    cursor = c.execute(
        "DELETE FROM alt_text_cache WHERE content_hash = ?",
        (content_hash,)
    )
    deleted = cursor.rowcount
    c.commit()
    c.close()
    return deleted
