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
    anhang: Optional[list[dict]] = None,
) -> int:
    """werkzeuge (28.08.2026): Namen der aufgerufenen Werkzeuge in Reihenfolge — None = unbekannt
    (Altbestand), [] = ausdruecklich ohne Werkzeug geantwortet.
    anhang (11.09.2026): Download-Knoepfe unter der Antwort (Meine Ausgaben), None = keine."""
    if role not in _VALID_ROLES:
        raise ValueError(f"Ungueltige Rolle: {role!r}")
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO chat_messages (project_id, role, content, image_refs, intent, werkzeuge, anhang) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                project_id,
                role,
                content,
                json.dumps(image_refs) if image_refs else None,
                intent,
                json.dumps(werkzeuge) if werkzeuge is not None else None,
                json.dumps(anhang, ensure_ascii=False) if anhang else None,
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
            "SELECT id, role, content, image_refs, intent, created_at, werkzeuge, anhang "
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
            "anhang": (json.loads(r["anhang"]) if r["anhang"] else []),
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


def karte_aktualisieren(project_id: int, angebot_id: str, felder: dict) -> int:
    """Bestaetigungs-Karte im gespeicherten Verlauf auf ihren neuen Zustand stellen (Pruefung 4, M3: „Erledigt“ oder „Nicht
    mehr gültig“), damit sie nach dem Neuladen nicht wieder als offene Frage mit Knopf erscheint. Rueckgabe: geaenderte Zeilen."""
    if not angebot_id:
        return 0
    conn = get_db()
    try:
        rows = conn.execute("SELECT id, anhang FROM chat_messages WHERE project_id = ? AND anhang LIKE ?",
                            (project_id, f"%{angebot_id}%")).fetchall()
        n = 0
        for r in rows:
            try:
                liste = json.loads(r["anhang"]) or []
            except ValueError:
                continue
            geaendert = False
            for a in liste:
                if isinstance(a, dict) and a.get("art") == "bestaetigung" and a.get("angebot_id") == angebot_id:
                    a.update(felder)
                    geaendert = True
            if geaendert:
                conn.execute("UPDATE chat_messages SET anhang = ? WHERE id = ?", (json.dumps(liste, ensure_ascii=False), r["id"]))
                n += 1
        conn.commit()
        return n
    finally:
        conn.close()
