"""CRUD fuer den Gespraechsverlauf des InkluAgent. Kennt nichts von Bildern oder Pipeline.

- chat_messages: das Gespraech je Projekt (bisher; mit dem Projekt geloescht).
- agent_nachrichten (Ausbau Runde 2, Schritt 2, Schalter funktionen.AGENT_KONTO): EIN Gespraech je Konto, das Projekt nur als
  Marke (Momentaufnahme des Namens, beim Loeschen des Projekts als geloescht markiert). agent_fokus: woran es gerade arbeitet.
"""
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
    return [_zeile(r) for r in rows]


def _zeile(r) -> dict:
    return {
        "id": r["id"],
        "role": r["role"],
        "content": r["content"],
        "image_refs": json.loads(r["image_refs"]) if r["image_refs"] else None,
        "intent": r["intent"],
        "created_at": r["created_at"],
        "werkzeuge": (json.loads(r["werkzeuge"]) if r["werkzeuge"] else None),
        "anhang": (json.loads(r["anhang"]) if r["anhang"] else []),
    }


def get_recent(project_id: int, limit: int = 20) -> list[dict]:
    """Die JUENGSTEN `limit` Nachrichten eines Projekts, aelteste zuerst. (get_history mit LIMIT liefert die aeltesten —
    fuer das Gedaechtnis des Agenten falsch, 12.10.2026.)"""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, role, content, image_refs, intent, created_at, werkzeuge, anhang FROM chat_messages "
            "WHERE project_id = ? ORDER BY created_at DESC, id DESC LIMIT ?", (project_id, limit)).fetchall()
    finally:
        conn.close()
    return [_zeile(r) for r in reversed(rows)]


def anzahl(project_id: int) -> int:
    conn = get_db()
    try:
        return conn.execute("SELECT COUNT(*) FROM chat_messages WHERE project_id = ?", (project_id,)).fetchone()[0]
    finally:
        conn.close()


# ─── Gespraech je Konto (Ausbau Runde 2, Schritt 2) ─────────────────────────────────────────────────────────────────────────
def konto_anhaengen(
    user_id: int,
    project_id: Optional[int],
    role: str,
    content: str,
    image_refs: Optional[list[int]] = None,
    intent: Optional[str] = None,
    werkzeuge: Optional[list[str]] = None,
    anhang: Optional[list[dict]] = None,
) -> int:
    """Wie append_message, aber ins Gespraech des Kontos; der Projektname wird als Marke festgehalten."""
    if role not in _VALID_ROLES:
        raise ValueError(f"Ungueltige Rolle: {role!r}")
    conn = get_db()
    try:
        name = ""
        if project_id:
            p = conn.execute("SELECT name, filename FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
            name = ((p["name"] or p["filename"]) if p else "") or ""
        cursor = conn.execute(
            "INSERT INTO agent_nachrichten (user_id, project_id, projekt_name, role, content, image_refs, intent, werkzeuge, anhang) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id, project_id, name, role, content,
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


def konto_verlauf(user_id: int, limit: int = 200) -> list[dict]:
    """Die juengsten `limit` Nachrichten des Kontos, aelteste zuerst, je mit Projektmarke (project_id, projekt_name,
    projekt_geloescht)."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, role, content, image_refs, intent, created_at, werkzeuge, anhang, project_id, projekt_name, projekt_geloescht "
            "FROM agent_nachrichten WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    finally:
        conn.close()
    out = []
    for r in reversed(rows):
        z = _zeile(r)
        z.update({"project_id": r["project_id"], "projekt_name": r["projekt_name"] or "",
                  "projekt_geloescht": bool(r["projekt_geloescht"])})
        out.append(z)
    return out


def projekt_geloescht_markieren(conn, project_id: int) -> None:
    """delete_project (in seiner Verbindung): das Gespraech des Kontos bleibt, der Projektbezug gilt als geloescht."""
    conn.execute("UPDATE agent_nachrichten SET projekt_geloescht = 1 WHERE project_id = ?", (project_id,))
    conn.execute("UPDATE agent_fokus SET project_id = NULL, geaendert_am = datetime('now') WHERE project_id = ?", (project_id,))


def fokus_setzen(user_id: int, project_id: Optional[int]) -> None:
    conn = get_db()
    try:
        conn.execute("INSERT INTO agent_fokus (user_id, project_id, geaendert_am) VALUES (?, ?, datetime('now')) "
                     "ON CONFLICT(user_id) DO UPDATE SET project_id = excluded.project_id, geaendert_am = excluded.geaendert_am",
                     (user_id, project_id))
        conn.commit()
    finally:
        conn.close()


def fokus(user_id: int) -> Optional[int]:
    conn = get_db()
    try:
        r = conn.execute("SELECT project_id FROM agent_fokus WHERE user_id = ?", (user_id,)).fetchone()
        return r["project_id"] if r else None
    finally:
        conn.close()


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
        # Gespraech je Konto (Runde 2, Schritt 2): dieselbe Karte steht dort, wenn der Schalter an war
        for r in conn.execute("SELECT id, anhang FROM agent_nachrichten WHERE project_id = ? AND anhang LIKE ?",
                              (project_id, f"%{angebot_id}%")).fetchall():
            try:
                liste = json.loads(r["anhang"]) or []
            except ValueError:
                continue
            treffer = [a for a in liste if isinstance(a, dict) and a.get("art") == "bestaetigung" and a.get("angebot_id") == angebot_id]
            for a in treffer:
                a.update(felder)
            if treffer:
                conn.execute("UPDATE agent_nachrichten SET anhang = ? WHERE id = ?", (json.dumps(liste, ensure_ascii=False), r["id"]))
                n += 1
        conn.commit()
        return n
    finally:
        conn.close()
