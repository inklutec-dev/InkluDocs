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
            "dokument_nr": d["doc_index"] if mehrere and d["doc_index"] else None,
            "document_id": d["id"]}   # Live-Fenster: damit es die Karte des Dokuments findet (doc_heading_<id>)


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
    out = {"art": "bild", "image_id": r["id"], "nummer": ids.index(r["id"]) + 1 if r["id"] in ids else None,
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
    out = {"art": "feld", "feld_id": r["id"], "nummer": r["feld_index"],
           "seite": r["page_number"] if r["page_number"] and r["page_number"] > 0 else None}
    out.update(_dokument(conn, project_id, r["document_id"]))
    return out


_ALT_MAX = 1000


def live(user_id: int, laeufe: dict, vorher=None, alt_text_von=None) -> dict:
    """Live-Fenster (16.10.2026): {"laeuft": {...} oder None, "vorher": {...} oder None}.
    laeufe: die Alt-Text-Laeufe des Prozesses (main._ALT_LAEUFE, Projekt -> {gesamt, document_id}); gezeigt wird der juengste in
    einem Projekt DIESES Kontos — Projektname, gesamt, fertig (gesamt minus noch offene) und das Bild, an dem er gerade ist (wie
    ziel: Nummer je Dokument, Seite, Dokument, Adressen). vorher: Kennung eines Bildes, das die Seite zuletzt als „in Arbeit“
    gezeigt hat — ist es fertig (done/error) und gehoert zum Konto, kommt sein Alt-Text (alt_text_von, wie die Oberflaeche ihn
    zeigt) mit. Nie eine Ausnahme nach aussen."""
    out: dict = {"laeuft": None, "vorher": None}
    try:
        from database import get_db
        conn = get_db()
        try:
            ids = [pid for pid in laeufe if _zahl(pid) is not None]
            if ids:
                platz = ",".join("?" * len(ids))
                p = conn.execute(f"SELECT id, COALESCE(NULLIF(TRIM(name), ''), filename) AS name FROM projects "
                                 f"WHERE user_id = ? AND id IN ({platz}) ORDER BY id DESC LIMIT 1", (user_id, *ids)).fetchone()
                if p:
                    e = laeufe.get(p["id"]) or {}
                    dok = e.get("document_id")
                    if dok is not None:
                        offen = conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ? AND document_id = ? AND status IN "
                                             "('pending', 'processing')", (p["id"], dok)).fetchone()[0]
                    else:
                        offen = conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ? AND status IN ('pending', 'processing')",
                                             (p["id"],)).fetchone()[0]
                    gesamt = max(int(e.get("gesamt") or 0), int(offen))
                    jetzt = conn.execute("SELECT id FROM images WHERE project_id = ? AND status = 'processing' ORDER BY id DESC LIMIT 1",
                                         (p["id"],)).fetchone()
                    out["laeuft"] = {"art": "alt_texte", "project_id": p["id"], "projekt": (p["name"] or "")[:_NAME_MAX],
                                     "gesamt": gesamt, "fertig": max(0, gesamt - int(offen)),
                                     "bild": dict(_bild(conn, p["id"], jetzt["id"]) or {}, image_id=jetzt["id"]) if jetzt else None}
            v = _zahl(vorher)
            if v is not None:
                r = conn.execute("SELECT i.* FROM images i JOIN projects p ON p.id = i.project_id WHERE i.id = ? AND p.user_id = ? "
                                 "AND i.status IN ('done', 'error')", (v, user_id)).fetchone()
                if r:
                    text = alt_text_von(dict(r)) if alt_text_von else (r["alt_text"] or "")
                    out["vorher"] = dict(_bild(conn, r["project_id"], r["id"]) or {}, image_id=r["id"], status=r["status"],
                                         alt=str(text or "")[:_ALT_MAX])
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        log.exception("Live-Fenster: Stand nicht lesbar (ignoriert)")
    return out


def ergebnis(name: str, result, neue_aktionen) -> dict:
    """Livestream (12.10.2026): Kurzfassung nach einem Werkzeug — ok, bei Fehler der Grund (gekuerzt), bei Alt-Text und Quickinfo
    der neue Text (aus den Aktionen fuer die Oberflaeche bzw. dem Ergebnis von generate_alt_text). Die Seite setzt alles mit
    textContent."""
    ok = isinstance(result, dict) and bool(result.get("ok"))
    out: dict = {"ok": ok}
    if not ok:
        out["fehler"] = str(result.get("error") or "")[:200] if isinstance(result, dict) else ""
        return out
    text = ""
    for a in neue_aktionen or []:
        if isinstance(a, dict) and a.get("type") == "refresh_image":
            text = a.get("alt_text") or ""
        elif isinstance(a, dict) and a.get("type") == "refresh_feld":
            text = a.get("quickinfo") or ""
    if not text and name == "generate_alt_text" and isinstance(result.get("result"), dict):
        text = result["result"].get("alt_text") or ""
    if text:
        out["text"] = str(text)[:500]
    return out
