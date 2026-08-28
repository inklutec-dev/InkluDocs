"""CRUD fuer chat_messages. Kennt nichts von Bildern oder Pipeline,
arbeitet nur ueber project_id."""
import json
from typing import Optional

from database import get_db


_VALID_ROLES = ("user", "assistant", "system")


def append_message(
    project_id: int,
    role: str,
    content: str,
    image_refs: Optional[list[int]] = None,
    intent: Optional[str] = None,
    werkzeuge: Optional[list[str]] = None,
) -> int:
    """werkzeuge (28.08.2026): Namen der aufgerufenen Werkzeuge in Reihenfolge — None = unbekannt
    (Altbestand), [] = ausdruecklich ohne Werkzeug geantwortet."""
    if role not in _VALID_ROLES:
        raise ValueError(f"Ungueltige Rolle: {role!r}")
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO chat_messages (project_id, role, content, image_refs, intent, werkzeuge) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                project_id,
                role,
                content,
                json.dumps(image_refs) if image_refs else None,
                intent,
                json.dumps(werkzeuge) if werkzeuge is not None else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_history(project_id: int, limit: int = 200) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, role, content, image_refs, intent, created_at, werkzeuge "
            "FROM chat_messages WHERE project_id = ? "
            "ORDER BY created_at ASC, id ASC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "image_refs": json.loads(r["image_refs"]) if r["image_refs"] else None,
            "intent": r["intent"],
            "created_at": r["created_at"],
            "werkzeuge": (json.loads(r["werkzeuge"]) if r["werkzeuge"] else None),
        }
        for r in rows
    ]


def clear_history(project_id: int) -> int:
    conn = get_db()
    try:
        cursor = conn.execute(
            "DELETE FROM chat_messages WHERE project_id = ?", (project_id,)
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
