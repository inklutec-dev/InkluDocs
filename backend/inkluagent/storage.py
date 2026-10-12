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
        gid = _aktuelles_gespraech(conn, user_id)
        cursor = conn.execute(
            "INSERT INTO agent_nachrichten (user_id, project_id, projekt_name, role, content, image_refs, intent, werkzeuge, anhang, "
            "gespraech_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id, project_id, name, role, content,
                json.dumps(image_refs) if image_refs else None,
                intent,
                json.dumps(werkzeuge) if werkzeuge is not None else None,
                json.dumps(anhang, ensure_ascii=False) if anhang else None,
                gid,
            ),
        )
        # Titel = die erste Frage (eine Zeile, gekuerzt); zuletzt = jetzt
        conn.execute("UPDATE agent_gespraeche SET zuletzt_am = datetime('now'), titel = CASE WHEN titel = '' AND ? = 'user' "
                     "THEN ? ELSE titel END WHERE id = ?", (role, _titel(content), gid))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


# ─── Gespraeche (Steve 12.10.2026): „Neues Gespräch“, „Frühere Gespräche“, „Hier weitermachen“ ─────────────────────────────
TITEL_LAENGE = 80


def _titel(text: str) -> str:
    """Die erste Frage als Titel: erste Zeile, ohne die Kurzzeile des Hochladefelds, hoechstens 80 Zeichen."""
    zeile = next((z.strip() for z in (text or "").splitlines() if z.strip()), "")
    return zeile if len(zeile) <= TITEL_LAENGE else zeile[:TITEL_LAENGE - 1].rstrip() + "…"


def _aktuelles_gespraech(conn, user_id: int) -> int:
    """Das laufende Gespraech des Kontos (agent_fokus.gespraech_id). Gibt es keins, wird eins angelegt; Nachrichten von vor den
    Gespraechen (gespraech_id leer) gehoeren dann dazu, und die erste Frage wird sein Titel."""
    r = conn.execute("SELECT f.gespraech_id FROM agent_fokus f JOIN agent_gespraeche g ON g.id = f.gespraech_id "
                     "WHERE f.user_id = ? AND g.user_id = ?", (user_id, user_id)).fetchone()
    if r and r["gespraech_id"]:
        return int(r["gespraech_id"])
    neu = conn.execute("SELECT id FROM agent_gespraeche WHERE user_id = ? ORDER BY zuletzt_am DESC, id DESC LIMIT 1", (user_id,)).fetchone()
    gid = int(neu["id"]) if neu else int(conn.execute("INSERT INTO agent_gespraeche (user_id) VALUES (?)", (user_id,)).lastrowid)
    alt = conn.execute("SELECT content FROM agent_nachrichten WHERE user_id = ? AND gespraech_id IS NULL AND role = 'user' "
                       "ORDER BY id LIMIT 1", (user_id,)).fetchone()
    conn.execute("UPDATE agent_nachrichten SET gespraech_id = ? WHERE user_id = ? AND gespraech_id IS NULL", (gid, user_id))
    if alt:
        conn.execute("UPDATE agent_gespraeche SET titel = ? WHERE id = ? AND titel = ''", (_titel(alt["content"]), gid))
    _gespraech_merken(conn, user_id, gid)
    return gid


def _gespraech_merken(conn, user_id: int, gid: Optional[int]) -> None:
    conn.execute("INSERT INTO agent_fokus (user_id, gespraech_id, geaendert_am) VALUES (?, ?, datetime('now')) "
                 "ON CONFLICT(user_id) DO UPDATE SET gespraech_id = excluded.gespraech_id", (user_id, gid))


def aktuelles_gespraech(user_id: int) -> int:
    conn = get_db()
    try:
        gid = _aktuelles_gespraech(conn, user_id)
        conn.commit()
        return gid
    finally:
        conn.close()


def neues_gespraech(user_id: int) -> int:
    """Beginnt ein neues Gespraech (das bisherige bleibt unter „Frühere Gespräche“). Ein leeres laufendes wird weiterbenutzt."""
    conn = get_db()
    try:
        alt = _aktuelles_gespraech(conn, user_id)
        leer = not conn.execute("SELECT 1 FROM agent_nachrichten WHERE gespraech_id = ? LIMIT 1", (alt,)).fetchone()
        gid = alt if leer else int(conn.execute("INSERT INTO agent_gespraeche (user_id) VALUES (?)", (user_id,)).lastrowid)
        _gespraech_merken(conn, user_id, gid)
        conn.commit()
        return gid
    finally:
        conn.close()


def gespraech_fortsetzen(user_id: int, gid: int) -> bool:
    """„Hier weitermachen“: dieses (eigene) Gespraech wird wieder das laufende."""
    conn = get_db()
    try:
        if not conn.execute("SELECT 1 FROM agent_gespraeche WHERE id = ? AND user_id = ?", (gid, user_id)).fetchone():
            return False
        _gespraech_merken(conn, user_id, gid)
        conn.execute("UPDATE agent_gespraeche SET zuletzt_am = datetime('now') WHERE id = ?", (gid,))
        conn.commit()
        return True
    finally:
        conn.close()


def gespraeche_liste(user_id: int, project_id: Optional[int] = None, suche: str = "", limit: int = 100) -> list[dict]:
    """Gespraeche mit Nachrichten, neueste zuerst: Titel, Datum, Zahl der Nachrichten, beteiligte Projekte, laufendes. Mit
    project_id nur Gespraeche, in denen es um dieses Projekt ging; mit suche nur solche, deren Titel, Projektname oder Text sie
    enthaelt."""
    conn = get_db()
    try:
        aktuell = _aktuelles_gespraech(conn, user_id)
        conn.commit()
        sql = ("SELECT g.id, g.titel, g.angelegt_am, g.zuletzt_am, COUNT(n.id) AS anzahl, MAX(n.project_id) AS irgendein_projekt "
               "FROM agent_gespraeche g JOIN agent_nachrichten n ON n.gespraech_id = g.id WHERE g.user_id = ?")
        werte: list = [user_id]
        if project_id:
            sql += " AND g.id IN (SELECT gespraech_id FROM agent_nachrichten WHERE user_id = ? AND project_id = ?)"
            werte += [user_id, project_id]
        if suche:
            sql += (" AND g.id IN (SELECT gespraech_id FROM agent_nachrichten WHERE user_id = ? AND (content LIKE ? OR projekt_name LIKE ?)"
                    " UNION SELECT id FROM agent_gespraeche WHERE user_id = ? AND titel LIKE ?)")
            werte += [user_id, f"%{suche}%", f"%{suche}%", user_id, f"%{suche}%"]
        rows = conn.execute(sql + " GROUP BY g.id ORDER BY g.zuletzt_am DESC, g.id DESC LIMIT ?", (*werte, limit)).fetchall()
        out = []
        for r in rows:
            projekte = conn.execute(
                "SELECT project_id, projekt_name, MAX(projekt_geloescht) AS geloescht FROM agent_nachrichten WHERE gespraech_id = ? "
                "AND project_id IS NOT NULL GROUP BY project_id ORDER BY MIN(id)", (r["id"],)).fetchall()
            letzte = conn.execute("SELECT project_id FROM agent_nachrichten WHERE gespraech_id = ? AND project_id IS NOT NULL "
                                  "AND projekt_geloescht = 0 ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
            out.append({"id": r["id"], "titel": r["titel"] or "", "angelegt_am": r["angelegt_am"], "zuletzt_am": r["zuletzt_am"],
                        "anzahl": r["anzahl"], "aktuell": r["id"] == aktuell,
                        "projekte": [{"project_id": p["project_id"], "name": p["projekt_name"] or "", "geloescht": bool(p["geloescht"])}
                                     for p in projekte],
                        "letztes_projekt": letzte["project_id"] if letzte else None})
        return out
    finally:
        conn.close()


def gespraech_lesen(user_id: int, gid: int) -> Optional[list[dict]]:
    """Alle Nachrichten eines eigenen Gespraechs, aelteste zuerst (wie konto_verlauf) — None, wenn es nicht dem Konto gehoert."""
    conn = get_db()
    try:
        if not conn.execute("SELECT 1 FROM agent_gespraeche WHERE id = ? AND user_id = ?", (gid, user_id)).fetchone():
            return None
    finally:
        conn.close()
    return konto_verlauf(user_id, limit=1000, gespraech_id=gid)


def konto_verlauf(user_id: int, limit: int = 200, gespraech_id: Optional[int] = None) -> list[dict]:
    """Die juengsten `limit` Nachrichten des LAUFENDEN Gespraechs (oder von gespraech_id), aelteste zuerst, je mit Projektmarke
    (project_id, projekt_name, projekt_geloescht)."""
    conn = get_db()
    try:
        gid = gespraech_id or _aktuelles_gespraech(conn, user_id)
        conn.commit()
        rows = conn.execute(
            "SELECT id, role, content, image_refs, intent, created_at, werkzeuge, anhang, project_id, projekt_name, projekt_geloescht "
            "FROM agent_nachrichten WHERE user_id = ? AND gespraech_id = ? ORDER BY id DESC LIMIT ?", (user_id, gid, limit)).fetchall()
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
        # Gespraech je Konto (Runde 2, Schritt 2): dieselbe Karte steht dort, wenn der Schalter an war. Ohne Projekt-Filter: auch
        # Karten aus Seiten ohne Projekt (Schritt 4, Projekt 0 im Angebot, NULL im Gespraech); Angebots-Kennungen sind eindeutig.
        for r in conn.execute("SELECT id, anhang FROM agent_nachrichten WHERE anhang LIKE ?", (f"%{angebot_id}%",)).fetchall():
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
