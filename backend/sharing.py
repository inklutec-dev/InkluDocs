"""Gastzugang / Projekt-Freigabe fuer InkluDocs (19.06.2026).

Erzeugt und prueft Einladungs-Token, ueber die ein Kunde (Gast) ein einzelnes
Projekt OHNE Login lesen + pruefen kann. Sicherheitsmodell (siehe Memo
project_inkludocs_gastzugang):
  1. Token = lang + zufaellig (secrets.token_urlsafe) -> nicht erratbar.
  2. E-Mail-Gate: der Gast muss einmal die Einladungs-E-Mail eingeben; nur bei
     Uebereinstimmung wird die Gast-Sitzung freigeschaltet.
  3. Jede Freigabe ist an GENAU EIN Projekt + EINE E-Mail + EINEN Besitzer
     gebunden -> gleichzeitige Gaeste/Projekte kommen sich nie in die Quere.

Reine Datenlogik (keine Routen/kein FastAPI). Endpunkte folgen in main.py
(spaetere Etappe). Bewusst isoliert + testbar gehalten.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from database import get_db

DEFAULT_EXPIRY_DAYS = 30
# shares.status: active=gueltig | revoked=widerrufen | expired=abgelaufen |
#                completed=Gast hat abgeschlossen (bleibt lesbar)
_USABLE_STATES = ("active", "completed")


def create_share(project_id, guest_email, created_by, expiry_days=DEFAULT_EXPIRY_DAYS, guest_name=""):
    """Legt eine Freigabe an und gibt den geheimen Token zurueck.
    guest_email wird normalisiert (trim+lowercase) gespeichert; guest_name optional (Klarname)."""
    token = secrets.token_urlsafe(32)
    expires_at = None
    if expiry_days:
        expires_at = (datetime.utcnow() + timedelta(days=expiry_days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO shares (project_id, token, guest_email, created_by, expires_at, guest_name) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, token, (guest_email or "").strip().lower(), created_by, expires_at, (guest_name or "").strip()),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def _is_expired(share):
    if not share["expires_at"]:
        return False
    try:
        return datetime.utcnow() > datetime.strptime(share["expires_at"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return False


def get_share(token):
    """Freigabe-Zeile zum Token oder None. Abgelaufene aktive Token werden
    einmalig auf expired gesetzt und als ungueltig (None) behandelt."""
    if not token:
        return None
    conn = get_db()
    try:
        share = conn.execute("SELECT * FROM shares WHERE token = ?", (token,)).fetchone()
        if not share:
            return None
        if share["status"] == "active" and _is_expired(share):
            conn.execute("UPDATE shares SET status = 'expired' WHERE id = ?", (share["id"],))
            conn.commit()
            return None
        return share
    finally:
        conn.close()


def is_valid_share(token):
    """True, wenn der Token existiert und nutzbar ist (active/completed)."""
    share = get_share(token)
    return bool(share and share["status"] in _USABLE_STATES)


def confirm_guest_email(token, email):
    """E-Mail-Gate: prueft, ob die eingegebene E-Mail zur Einladung passt.
    Bei Erfolg email_confirmed_at setzen und True zurueckgeben."""
    share = get_share(token)
    if not share or share["status"] not in _USABLE_STATES:
        return False
    if (email or "").strip().lower() != share["guest_email"]:
        return False
    conn = get_db()
    try:
        conn.execute("UPDATE shares SET email_confirmed_at = datetime('now') WHERE id = ?", (share["id"],))
        conn.commit()
    finally:
        conn.close()
    return True


def mark_completed(token):
    """Setzt eine aktive Freigabe auf completed (bleibt lesbar)."""
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE shares SET status = 'completed', completed_at = datetime('now') "
            "WHERE token = ? AND status = 'active'",
            (token,),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def revoke_share(token, owner_id):
    """Widerruft eine Freigabe (nur durch den Besitzer)."""
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE shares SET status = 'revoked' WHERE token = ? AND created_by = ?",
            (token, owner_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_shares_for_project(project_id, owner_id):
    """Alle Freigaben eines Projekts (Besitzer-Ansicht)."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM shares WHERE project_id = ? AND created_by = ? ORDER BY created_at DESC",
            (project_id, owner_id),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
