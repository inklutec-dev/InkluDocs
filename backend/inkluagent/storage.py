"""CRUD fuer den Gespraechsverlauf des InkluAgent. Kennt nichts von Bildern oder Pipeline.

- chat_messages: das Gespraech je Projekt (bisher; mit dem Projekt geloescht).
- agent_nachrichten (Ausbau Runde 2, Schritt 2, Schalter funktionen.AGENT_KONTO): EIN Gespraech je Konto, das Projekt nur als
  Marke (Momentaufnahme des Namens, beim Loeschen des Projekts als geloescht markiert). agent_fokus: woran es gerade arbeitet.
- content ist, was der Nutzer bzw. der Agent geschrieben hat; zusatz (Pruefung 13.10.2026, S3) eine Zeile des SERVERS zu einer
  Nutzer-Nachricht („Hochgeladen: …“, „Bestätigt über den Knopf: …“) mit Datei- und Projektnamen. Gelesene Zeilen tragen beides
  getrennt (text, zusatz) und zusammengesetzt fuer die Anzeige (content); das Gedaechtnis des Agenten nimmt fuer_modell().
"""
import json
import re
from typing import Optional

from database import get_db

from .daten import daten


_VALID_ROLES = ("user", "assistant", "system")


def append_message(
    project_id: int,
    role: str,
    content: str,
    image_refs: Optional[list[int]] = None,
    intent: Optional[str] = None,
    werkzeuge: Optional[list[str]] = None,
    anhang: Optional[list[dict]] = None,
    zusatz: Optional[str] = None,
    sprache: Optional[str] = None,
) -> int:
    """werkzeuge (28.08.2026): Namen der aufgerufenen Werkzeuge in Reihenfolge — None = unbekannt
    (Altbestand), [] = ausdruecklich ohne Werkzeug geantwortet.
    anhang (11.09.2026): Download-Knoepfe unter der Antwort (Meine Ausgaben), None = keine.
    zusatz (13.10.2026, S3): Zeile des Servers zur Nachricht, getrennt vom getippten Text (siehe Modulkopf).
    sprache (13.10.2026, B11): Sprache der Oberflaeche beim Schreiben, z. B. "de" — die Seite setzt sie als lang."""
    if role not in _VALID_ROLES:
        raise ValueError(f"Ungueltige Rolle: {role!r}")
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO chat_messages (project_id, role, content, zusatz, sprache, image_refs, intent, werkzeuge, anhang) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                project_id,
                role,
                content or "",
                (zusatz or "").strip() or None,
                _sprache(sprache),
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
            "SELECT " + _SPALTEN + " FROM chat_messages WHERE project_id = ? "
            "ORDER BY created_at ASC, id ASC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [_zeile(r) for r in rows]


_SPALTEN = "id, role, content, zusatz, sprache, image_refs, intent, created_at, werkzeuge, anhang"


def _sprache(wert: Optional[str]) -> Optional[str]:
    """Nur Sprachkuerzel wie „de“ oder „de-CH“ — die Seite setzt den Wert als lang-Attribut."""
    w = (wert or "").strip()
    return w if re.fullmatch(r"[a-z]{2}(-[A-Z]{2})?", w) else None


def anzeige(text: Optional[str], zusatz: Optional[str]) -> str:
    """Was Menschen im Verlauf lesen: der getippte Text, darunter die Zeile des Servers."""
    text, zusatz = (text or "").strip(), (zusatz or "").strip()
    return (text + "\n" + zusatz) if (text and zusatz) else (text or zusatz)


def fuer_modell(h: dict) -> str:
    """Was das Gedaechtnis des Agenten aus einer gespeicherten Nachricht liest (Pruefung 13.10.2026, S3): den getippten Text
    unveraendert, die Zeile des Servers als Daten gekennzeichnet. Sonst stuende „Hochgeladen: <fremder Dateiname>“ ab der
    naechsten Runde als eigene Bitte des Nutzers im Gedaechtnis."""
    if "text" not in h:   # Eintrag ohne getrennte Felder (eigene Dicts von Aufrufern)
        return h.get("content") or ""
    zusatz = daten(h.get("zusatz"))
    return "\n".join(t for t in ((h.get("text") or "").strip(), zusatz) if t)


def _zeile(r) -> dict:
    return {
        "id": r["id"],
        "role": r["role"],
        "content": anzeige(r["content"], r["zusatz"]),
        "text": r["content"] or "",
        "zusatz": r["zusatz"] or "",
        "sprache": r["sprache"] or "",
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
            "SELECT " + _SPALTEN + " FROM chat_messages "
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
    zusatz: Optional[str] = None,
    gespraech_id: Optional[int] = None,
    sprache: Optional[str] = None,
) -> int:
    """Wie append_message, aber ins Gespraech des Kontos; der Projektname wird als Marke festgehalten.
    gespraech_id (Pruefung 13.10.2026, K3): die Antwort gehoert in das Gespraech ihrer Frage — auch wenn der Nutzer waehrend des
    Laufs ein neues Gespraech begonnen oder ein frueheres fortgesetzt hat. Ohne (oder fremd): das laufende Gespraech."""
    if role not in _VALID_ROLES:
        raise ValueError(f"Ungueltige Rolle: {role!r}")
    conn = get_db()
    try:
        name = ""
        if project_id:
            p = conn.execute("SELECT name, filename FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
            name = ((p["name"] or p["filename"]) if p else "") or ""
        gid = None
        if gespraech_id and conn.execute("SELECT 1 FROM agent_gespraeche WHERE id = ? AND user_id = ?",
                                         (int(gespraech_id), user_id)).fetchone():
            gid = int(gespraech_id)
        if gid is None:
            gid = _aktuelles_gespraech(conn, user_id)
        cursor = conn.execute(
            "INSERT INTO agent_nachrichten (user_id, project_id, projekt_name, role, content, zusatz, sprache, image_refs, intent, "
            "werkzeuge, anhang, gespraech_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id, project_id, name, role, content or "",
                (zusatz or "").strip() or None,
                _sprache(sprache),
                json.dumps(image_refs) if image_refs else None,
                intent,
                json.dumps(werkzeuge) if werkzeuge is not None else None,
                json.dumps(anhang, ensure_ascii=False) if anhang else None,
                gid,
            ),
        )
        # Titel = die erste Frage (eine Zeile, gekuerzt; ohne getippten Text die Zeile des Servers, z. B. „Hochgeladen: a.pdf“)
        conn.execute("UPDATE agent_gespraeche SET zuletzt_am = datetime('now'), titel = CASE WHEN titel = '' AND ? = 'user' "
                     "THEN ? ELSE titel END WHERE id = ?", (role, _titel(content or zusatz or ""), gid))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


# ─── Gespraeche (Steve 12.10.2026): „Neues Gespräch“, „Frühere Gespräche“, „Hier weitermachen“ ─────────────────────────────
TITEL_LAENGE = 80


def _titel(text: str) -> str:
    """Die erste Frage als Titel: erste Zeile, hoechstens 80 Zeichen."""
    zeile = next((z.strip() for z in (text or "").splitlines() if z.strip()), "")
    return zeile if len(zeile) <= TITEL_LAENGE else zeile[:TITEL_LAENGE - 1].rstrip() + "…"


def _gespraech_im_fokus(conn, user_id: int) -> Optional[int]:
    r = conn.execute("SELECT f.gespraech_id FROM agent_fokus f JOIN agent_gespraeche g ON g.id = f.gespraech_id "
                     "WHERE f.user_id = ? AND g.user_id = ?", (user_id, user_id)).fetchone()
    return int(r["gespraech_id"]) if r and r["gespraech_id"] else None


def _juengstes_gespraech(conn, user_id: int) -> Optional[int]:
    r = conn.execute("SELECT id FROM agent_gespraeche WHERE user_id = ? ORDER BY zuletzt_am DESC, id DESC LIMIT 1", (user_id,)).fetchone()
    return int(r["id"]) if r else None


def _laufendes_gespraech(conn, user_id: int) -> Optional[int]:
    """Nur lesen (Pruefung 13.10.2026, K15: lesende Anfragen legen nichts an): das laufende Gespraech (agent_fokus), sonst das
    juengste; None, solange das Konto keins hat."""
    return _gespraech_im_fokus(conn, user_id) or _juengstes_gespraech(conn, user_id)


def _aktuelles_gespraech(conn, user_id: int) -> int:
    """Zum Schreiben: das laufende Gespraech des Kontos; gibt es keins, das juengste oder ein neues. Nachrichten von vor den
    Gespraechen uebernimmt die Start-Migration (database.init_db), nicht diese Funktion (Runde 2, K-N9: hier war es doppelt)."""
    gid = _gespraech_im_fokus(conn, user_id)
    if gid:
        return gid
    gid = _juengstes_gespraech(conn, user_id) or int(conn.execute("INSERT INTO agent_gespraeche (user_id) VALUES (?)", (user_id,)).lastrowid)
    _gespraech_merken(conn, user_id, gid)
    return gid


def _gespraech_merken(conn, user_id: int, gid: Optional[int]) -> None:
    conn.execute("INSERT INTO agent_fokus (user_id, gespraech_id, geaendert_am) VALUES (?, ?, datetime('now')) "
                 "ON CONFLICT(user_id) DO UPDATE SET gespraech_id = excluded.gespraech_id", (user_id, gid))


def aktuelles_gespraech(user_id: int) -> int:
    """Kennung des laufenden Gespraechs (legt eins an, wenn es keins gibt) — fuer Tests und Werkzeuge ausserhalb dieses Moduls."""
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
    enthaelt. Nur lesen, und die Projekte aller Gespraeche in EINER Abfrage (Pruefung 13.10.2026, K15: vorher legte die Liste
    ein Gespraech an und fragte je Gespraech zweimal ueber die ganze Tabelle)."""
    conn = get_db()
    try:
        aktuell = _laufendes_gespraech(conn, user_id)
        sql = ("SELECT g.id, g.titel, g.angelegt_am, g.zuletzt_am, COUNT(n.id) AS anzahl "
               "FROM agent_gespraeche g JOIN agent_nachrichten n ON n.gespraech_id = g.id WHERE g.user_id = ?")
        werte: list = [user_id]
        if project_id:
            sql += " AND g.id IN (SELECT gespraech_id FROM agent_nachrichten WHERE user_id = ? AND project_id = ?)"
            werte += [user_id, project_id]
        if suche:
            muster = "%" + suche.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            sql += (" AND g.id IN (SELECT gespraech_id FROM agent_nachrichten WHERE user_id = ? AND (content LIKE ? ESCAPE '\\' "
                    "OR zusatz LIKE ? ESCAPE '\\' OR projekt_name LIKE ? ESCAPE '\\') "
                    "UNION SELECT id FROM agent_gespraeche WHERE user_id = ? AND titel LIKE ? ESCAPE '\\')")
            werte += [user_id, muster, muster, muster, user_id, muster]
        rows = conn.execute(sql + " GROUP BY g.id ORDER BY g.zuletzt_am DESC, g.id DESC LIMIT ?", (*werte, limit)).fetchall()
        projekte: dict[int, list] = {}
        if rows:
            ids = [r["id"] for r in rows]
            for p in conn.execute(
                    "SELECT gespraech_id, project_id, projekt_name, MAX(projekt_geloescht) AS geloescht, MIN(id) AS erste, "
                    "MAX(CASE WHEN projekt_geloescht = 0 THEN id END) AS letzte_offen FROM agent_nachrichten "
                    "WHERE user_id = ? AND project_id IS NOT NULL AND gespraech_id IN (" + ",".join("?" * len(ids)) + ") "
                    "GROUP BY gespraech_id, project_id", (user_id, *ids)).fetchall():
                projekte.setdefault(p["gespraech_id"], []).append(p)
        out = []
        for r in rows:
            beteiligt = sorted(projekte.get(r["id"], []), key=lambda p: p["erste"])
            offen = [p for p in beteiligt if p["letzte_offen"] is not None]
            out.append({"id": r["id"], "titel": r["titel"] or "", "angelegt_am": r["angelegt_am"], "zuletzt_am": r["zuletzt_am"],
                        "anzahl": r["anzahl"], "aktuell": r["id"] == aktuell,
                        "projekte": [{"project_id": p["project_id"], "name": p["projekt_name"] or "", "geloescht": bool(p["geloescht"])}
                                     for p in beteiligt],
                        "letztes_projekt": max(offen, key=lambda p: p["letzte_offen"])["project_id"] if offen else None})
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
    (project_id, projekt_name, projekt_geloescht). Nur lesen: ohne Gespraech die Nachrichten von vor den Gespraechen."""
    conn = get_db()
    try:
        gid = gespraech_id or _laufendes_gespraech(conn, user_id)
        bedingung, wert = ("gespraech_id = ?", (gid,)) if gid else ("gespraech_id IS NULL", ())
        rows = conn.execute(
            "SELECT " + _SPALTEN + ", project_id, projekt_name, projekt_geloescht FROM agent_nachrichten "
            "WHERE user_id = ? AND " + bedingung + " ORDER BY id DESC LIMIT ?", (user_id, *wert, limit)).fetchall()
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


def datum(wert: Optional[str]) -> str:
    """„2026-10-12 11:45:00“ (UTC, SQLite) -> „12.10.2026, 13:45“ (Berlin) — fuer Gespraechslisten in Seite und Agent (vorher je
    eine Kopie in main und konto, Pruefung 13.10.2026, H2)."""
    try:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        d = datetime.strptime(str(wert)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Berlin"))
        return d.strftime("%d.%m.%Y, %H:%M")
    except Exception:  # noqa: BLE001
        return str(wert or "")


def gespraech_der_nachricht(user_id: int, nachricht_id: Optional[int]) -> Optional[int]:
    """In welches Gespraech eine gespeicherte Nachricht des Kontos kam (K3: die Antwort folgt ihrer Frage)."""
    if not nachricht_id:
        return None
    conn = get_db()
    try:
        r = conn.execute("SELECT gespraech_id FROM agent_nachrichten WHERE id = ? AND user_id = ?", (int(nachricht_id), user_id)).fetchone()
        return int(r["gespraech_id"]) if r and r["gespraech_id"] else None
    finally:
        conn.close()


def karte_aktualisieren(project_id: int, angebot_id: str, felder: dict, user_id: Optional[int] = None) -> int:
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
        # Karten aus Seiten ohne Projekt (Schritt 4, Projekt 0 im Angebot, NULL im Gespraech). Nur im eigenen Konto (Pruefung
        # 13.10.2026, K15: vorher ein LIKE ueber die Nachrichten aller Konten).
        if user_id is None:
            conn.commit()
            return n
        for r in conn.execute("SELECT id, anhang FROM agent_nachrichten WHERE user_id = ? AND anhang LIKE ?",
                              (int(user_id), f"%{angebot_id}%")).fetchall():
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
