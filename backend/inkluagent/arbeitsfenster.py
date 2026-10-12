"""Arbeitsfenster der Agentenansicht (Steve 12.10.2026, Michael Karbe Feedback 20261012 - 1, Punkt 6; Schalter
funktionen.AGENT_ARBEITSFENSTER): woran der InkluAgent gerade arbeitet — welches Dokument, welche Seite, welches Bild oder Feld.

Der Chat-Stream meldet je Werkzeugaufruf eine Zeile {"type": "tool", "name": …}; mit dem Schalter kommt `ziel` dazu, wenn das
Werkzeug ein Bild, ein Feld oder ein Dokument DIESES Projekts nennt. Die Angaben kommen aus der Datenbank, nie aus dem Modell:
aus den Argumenten zaehlen nur die Kennungen (ganze Zahlen), und nur, wenn sie zum Projekt des Laufs und zum Konto gehoeren. Namen
(Dokument) sind Nutzerdaten — die Seite setzt sie mit textContent. Nummern wie in der Oberflaeche: Bild n bzw. Feld n je Dokument.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

_NAME_MAX = 120


def _zahl(wert) -> Optional[int]:
    if isinstance(wert, bool):
        return None
    if isinstance(wert, int):
        return wert
    if isinstance(wert, str) and wert.strip().isdigit():
        return int(wert.strip())
    return None


def _dokument(conn, project_id: int, document_id) -> dict:
    """Name und Nummer des Dokuments (Nummer nur, wenn das Projekt mehrere hat — wie die Oberflaeche)."""
    if not document_id:
        return {}
    d = conn.execute("SELECT id, doc_index, display_name, original_filename FROM documents WHERE id = ? AND project_id = ?",
                     (document_id, project_id)).fetchone()
    if not d:
        return {}
    mehrere = conn.execute("SELECT COUNT(*) FROM documents WHERE project_id = ?", (project_id,)).fetchone()[0] > 1
    return {"dokument": (d["display_name"] or d["original_filename"] or "")[:_NAME_MAX],
            "dokument_nr": d["doc_index"] if mehrere and d["doc_index"] else None}


def ziel(user_id: int, project_id: Optional[int], args) -> Optional[dict]:
    """Woran ein Werkzeugaufruf arbeitet, oder None. Nie eine Ausnahme (der Stream darf daran nicht scheitern)."""
    if not project_id or not isinstance(args, dict):
        return None
    try:
        from database import get_db
        conn = get_db()
        try:
            if not conn.execute("SELECT 1 FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone():
                return None
            bild = _zahl(args.get("image_id"))
            if bild is not None:
                return _bild(conn, project_id, bild)
            feld = _zahl(args.get("feld_id"))
            if feld is not None:
                return _feld(conn, project_id, feld)
            dok = _zahl(args.get("document_id"))
            if dok is not None:
                out = _dokument(conn, project_id, dok)
                return dict(out, art="dokument") if out else None
            return None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        log.exception("Arbeitsfenster: Ziel nicht bestimmbar (ignoriert)")
        return None


def _bild(conn, project_id: int, image_id: int) -> Optional[dict]:
    r = conn.execute("SELECT id, document_id, page_number, page_view_path FROM images WHERE id = ? AND project_id = ?",
                     (image_id, project_id)).fetchone()
    if not r:
        return None
    # Nummer je Dokument in derselben Reihenfolge wie list_project_images (tools/project.py) und die Ansicht Alt-Texte
    if r["document_id"]:
        ids = [x[0] for x in conn.execute("SELECT id FROM images WHERE project_id = ? AND document_id = ? "
                                          "ORDER BY page_number, image_index, id", (project_id, r["document_id"])).fetchall()]
    else:
        ids = [x[0] for x in conn.execute("SELECT id FROM images WHERE project_id = ? AND document_id IS NULL "
                                          "ORDER BY page_number, image_index, id", (project_id,)).fetchall()]
    out = {"art": "bild", "nummer": ids.index(r["id"]) + 1 if r["id"] in ids else None,
           "seite": r["page_number"] if r["page_number"] and r["page_number"] > 0 else None,
           "bild_url": f"/api/images/{r['id']}/file",
           "seite_url": f"/api/images/{r['id']}/page-view" if r["page_view_path"] else None}
    out.update(_dokument(conn, project_id, r["document_id"]))
    return out


def _feld(conn, project_id: int, feld_id: int) -> Optional[dict]:
    r = conn.execute("SELECT id, document_id, page_number, feld_index FROM formularfelder WHERE id = ? AND project_id = ?",
                     (feld_id, project_id)).fetchone()
    if not r:
        return None
    out = {"art": "feld", "nummer": r["feld_index"],
           "seite": r["page_number"] if r["page_number"] and r["page_number"] > 0 else None}
    out.update(_dokument(conn, project_id, r["document_id"]))
    return out
