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

# ─── Welche Ansicht der Projektseite zu welcher Arbeit gehoert (Live-Fenster „zum Anklemmen“, 17.10.2026) ───
# EINE Liste je Richtung: ein neues Werkzeug ist ein Eintrag in ANSICHT_JE_WERKZEUG, ein neuer Lauf einer in ANSICHT_JE_LAUF (und
# eine Art in laufbuch.ARTEN). Die Ansichten sind die der Projektseite (app.html ansichtenFuer): dokument, tagging, alttexte,
# quickinfos, uebersetzung, abschluss. Kennt ein Projekt die Ansicht nicht (Word ohne Tagging, Grafik nur Alt-Texte), bleibt die
# Seite im Fenster, wo sie ist (InkluBuehne in app.html prueft das). Werkzeuge ohne Eintrag: die Ansicht folgt der Art des Ziels.
ANSICHT_JE_WERKZEUG: dict = {
    # Bilder und Alt-Texte
    "view_image": "alttexte", "generate_alt_text": "alttexte", "update_alt_text": "alttexte", "revert_alt_text": "alttexte",
    "get_image_metadata": "alttexte", "bild_umbenennen": "alttexte", "alt_texte_generieren": "alttexte",
    "exportiere_alt_texte": "alttexte",
    # Formularfelder und Quickinfos
    "view_field": "quickinfos", "get_field_details": "quickinfos", "generate_quickinfo": "quickinfos", "update_quickinfo": "quickinfos",
    "revert_quickinfo": "quickinfos", "save_to_master_data": "quickinfos", "quickinfos_generieren": "quickinfos",
    "exportiere_quickinfos": "quickinfos",
    # PDF: Dokument, Tagging, Pruefung
    "dokument_stand": "dokument", "dokument_umbenennen": "dokument", "exportiere_fertige_pdf": "dokument",
    "komplett_barrierefrei_machen": "dokument",
    "barrierefrei_machen": "tagging", "testweise_taggen": "tagging",
    "pruefung_starten": "abschluss", "pruefbericht_lesen": "abschluss", "pruefdatei_erstellen": "abschluss",
    "pruefdatei_lesen": "abschluss", "hoerprobe_lesen": "abschluss", "korrektur_anwenden": "abschluss",
    "korrektur_rueckgaengig": "abschluss",
    # Word
    "analysiere_word_struktur": "dokument", "konvertiere_zu_pdfua": "dokument", "exportiere_word": "dokument",
    "pruefe_word_dokument": "abschluss",
    "uebersetze_dokument": "uebersetzung", "uebersetzung_stand": "uebersetzung", "exportiere_uebersetzung": "uebersetzung",
}
ANSICHT_JE_LAUF: dict = {
    "alt_texte": "alttexte", "quickinfos": "quickinfos", "tagging": "tagging", "korrektur": "abschluss", "pruefung": "abschluss",
    "umwandeln": "dokument", "uebersetzen": "uebersetzung", "export": "dokument", "auftrag": "dokument",
    "uebernehmen": "dokument",   # aus der Ablage in ein Projekt: das Ziel in der Ansicht Dokument
}


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


def ziel(user_id: int, project_id: Optional[int], args, werkzeug: Optional[str] = None) -> Optional[dict]:
    """Woran ein Werkzeugaufruf arbeitet, oder None — mit der Ansicht, in der man es sieht (ANSICHT_JE_WERKZEUG) und dem Projekt.
    Nie eine Ausnahme (der Stream darf daran nicht scheitern)."""
    z = _ziel_ohne_ansicht(user_id, project_id, args)
    a = ANSICHT_JE_WERKZEUG.get(werkzeug or "")
    if z is not None:
        z["project_id"] = int(project_id)
        if a:
            z["ansicht"] = a
    elif a and isinstance(args, dict) and not any(k in args for k in ("image_id", "feld_id", "document_id")) \
            and _eigenes_projekt(user_id, project_id):
        # Ein Werkzeug fuer das ganze Projekt (z. B. In PDF umwandeln ohne Dokument): die Ansicht selbst ist die Stelle
        z = {"art": "ansicht", "project_id": int(project_id), "ansicht": a}
    return z


def _eigenes_projekt(user_id: int, project_id) -> bool:
    if not _zahl(project_id):
        return False
    try:
        from database import get_db
        conn = get_db()
        try:
            return conn.execute("SELECT 1 FROM projects WHERE id = ? AND user_id = ?", (int(project_id), user_id)).fetchone() is not None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        log.exception("Arbeitsfenster: Projekt nicht pruefbar (ignoriert)")
        return False


def _ziel_ohne_ansicht(user_id: int, project_id: Optional[int], args) -> Optional[dict]:
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
_LIVE_HOECHSTENS = 3


def _stelle(conn, user_id: int, project_id: int, stueck: Optional[dict], ansicht: Optional[str], alt_text_von=None,
            mit_text: bool = False) -> Optional[dict]:
    """Ein Stueck aus dem Laufbuch (nur Kennungen) als Stelle fuer die Seite — aus der Datenbank, nur in diesem Projekt."""
    if not stueck:
        return None
    art = stueck.get("art")
    out: Optional[dict] = None
    if art == "bild" and stueck.get("image_id"):
        out = _bild(conn, project_id, int(stueck["image_id"]))
        if out and mit_text:
            r = conn.execute("SELECT * FROM images WHERE id = ? AND project_id = ?", (int(stueck["image_id"]), project_id)).fetchone()
            if r:
                text = alt_text_von(dict(r)) if alt_text_von else (r["alt_text"] or "")
                out.update(status=r["status"], text=str(text or "")[:_ALT_MAX])
    elif art == "feld" and stueck.get("feld_id"):
        out = _feld(conn, project_id, int(stueck["feld_id"]))
        if out:
            out["feld_id"] = int(stueck["feld_id"])
            if mit_text:
                r = conn.execute("SELECT quickinfo FROM formularfelder WHERE id = ? AND project_id = ?",
                                 (int(stueck["feld_id"]), project_id)).fetchone()
                out["text"] = str((r["quickinfo"] if r else "") or "")[:_ALT_MAX]
    elif art in ("dokument", "seite") and stueck.get("document_id"):
        d = _dokument(conn, project_id, int(stueck["document_id"]))
        if d:
            out = dict(d, art=art)
            if art == "seite" and stueck.get("seite"):
                out["seite"] = int(stueck["seite"])
    if out is None:
        return None
    out["project_id"] = project_id
    if ansicht:
        out["ansicht"] = ansicht
    return out


def live(user_id: int, eintraege: Optional[list] = None, alt_text_von=None) -> dict:
    """Live-Fenster (16./17.10.2026): {"laeufe": [...]} — was der Agent im Hintergrund gerade tut, juengstes Laufendes zuerst
    (hoechstens _LIVE_HOECHSTENS). Quelle ist das Laufbuch (nur Laeufe des Agenten, laufbuch.fuer_konto); Projekt, Stelle und Texte
    kommen aus der Datenbank, nur aus Projekten DIESES Kontos. Je Lauf: art, station (bei Auftraegen), ansicht, projekt, gesamt,
    fertig, laeuft, ok, stelle (woran er gerade ist), vorher (das Stueck davor, mit Text — Alt-Text bzw. Quickinfo), ergebnis
    (z. B. die fertige Datei in der Ablage). Nie eine Ausnahme nach aussen."""
    out: dict = {"laeufe": []}
    try:
        import laufbuch
        from database import get_db
        eintraege = laufbuch.fuer_konto(user_id) if eintraege is None else eintraege
        conn = get_db()
        try:
            for e in eintraege:
                if len(out["laeufe"]) >= _LIVE_HOECHSTENS:
                    break
                pid = _zahl(e.get("project_id"))
                if pid is None:
                    continue
                p = conn.execute("SELECT id, COALESCE(NULLIF(TRIM(name), ''), filename) AS name FROM projects WHERE id = ? AND user_id = ?",
                                 (pid, user_id)).fetchone()
                if not p:
                    continue
                was = e.get("station") or e.get("art")
                ansicht = ANSICHT_JE_LAUF.get(was)
                ergebnis_ = None
                erg = e.get("ergebnis") or {}
                if erg.get("art") == "ausgabe" and _zahl(erg.get("ausgabe_id")) is not None:
                    a = conn.execute("SELECT id FROM ablage WHERE id = ? AND user_id = ?", (int(erg["ausgabe_id"]), user_id)).fetchone()
                    if a:
                        ergebnis_ = {"art": "ausgabe", "ausgabe_id": int(a["id"])}
                out["laeufe"].append({
                    "nr": e.get("nr"), "art": e.get("art"), "station": e.get("station"), "was": was, "ansicht": ansicht,
                    "project_id": pid, "projekt": (p["name"] or "")[:_NAME_MAX],
                    "gesamt": int(e.get("gesamt") or 0), "fertig": int(e.get("fertig") or 0),
                    "laeuft": bool(e.get("laeuft")), "ok": e.get("ok"),
                    "stelle": _stelle(conn, user_id, pid, e.get("stueck"), ansicht) or (
                        # eine Station ohne einzelne Stelle (Auftrag, Umwandeln des ganzen Projekts): die Ansicht selbst
                        {"art": "ansicht", "project_id": pid, "ansicht": ansicht} if e.get("laeuft") and ansicht else None),
                    "vorher": _stelle(conn, user_id, pid, e.get("vorher"), ansicht, alt_text_von, mit_text=True),
                    "ergebnis": ergebnis_})
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
