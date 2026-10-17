"""Aus der Ablage in ein Projekt (Steve 17.10.2026: „wäre natürlich cool, wenn der Agent das sogar in ein anderes Projekt laden
könnte, wenn man möchte“). Die Brücke für Ketten über Dateitypen: Word → barrierefreie PDF in „Meine Ablage“ → PDF-Projekt →
prüfen, bei Bedarf neu taggen. Vorher ging das nur über Herunterladen und neu Hochladen.

EINE Funktion für beide Wege (Chatbot = Oberfläche): der Knopf „In ein Projekt laden“ in Meine Ablage (POST
/api/ausgaben/{id}/in-projekt, main.py) und das Werkzeug ausgabe_in_projekt des InkluAgent (inkluagent/tools/ausgaben.py). Die
Datei geht denselben Weg wie ein Hochladen (main.upload_file: Dateityp am Inhalt, Größe, Passwort, Auslesen) — in ein NEUES Projekt
oder als weiteres Dokument in ein vorhandenes EIGENES Projekt derselben Art (PDF zu PDF, Word zu Word). Ein ZIP (Umwandlung mehrerer
Dokumente) wird ausgepackt: jede PDF- bzw. Word-Datei darin wird ein Dokument desselben Projekts. Kostet nichts (wie Hochladen).
Über den InkluAgent zählt ein neues Projekt zur Tagesgrenze wie beim Hochladen im Gespräch. Laufbuch: art „uebernehmen“ (das
Live-Fenster zeigt das Ziel, während es ausgelesen wird).
"""
from __future__ import annotations

import importlib
import io
import os
import zipfile
from typing import Optional

from fastapi import HTTPException

_AUSLESEN_S = 60.0


def _main():
    return importlib.import_module("main")


def _hochladen():
    return importlib.import_module("inkluagent.hochladen")


def projektname(row, art: str, _=None) -> str:
    """Name des neuen Projekts: Dokument bzw. Projekt der Herkunft, bei einer PDF mit dem Zusatz, was es ist."""
    _ = _ or (lambda s: s)
    basis = ((row["dokument_name"] or "").strip() or (row["projekt_name"] or "").strip()
             or os.path.splitext(row["dateiname"] or "")[0].strip() or _("Aus der Ablage"))
    return (_("{name} (barrierefreie PDF)").format(name=basis) if art == "pdf" else basis)[:120]


def dateien(pfad: str, dateiname: str, media: str, grenze: int, hoechstens: int, _=None) -> list:
    """[(Name, Inhalt, Art)] der Datei eines Eintrags; ein ZIP ausgepackt (nur PDF und Word, flach, je Datei hoechstens `grenze`)."""
    _ = _ or (lambda s: s)
    h = _hochladen()
    zu_gross = _("Die Datei ist zu groß (höchstens {n} MB).").format(n=grenze // (1024 * 1024))
    with open(pfad, "rb") as f:
        daten = f.read(grenze + 1)
    if media == "application/zip" or (daten[:4] == b"PK\x03\x04" and h.art_aus_inhalt(daten) is None):
        out = []
        try:
            with zipfile.ZipFile(io.BytesIO(daten)) as z:
                for info in z.infolist():
                    name = os.path.basename(info.filename or "")
                    if info.is_dir() or not name or os.path.splitext(name)[1].lower() not in (".pdf", ".docx"):
                        continue
                    if len(out) >= hoechstens:
                        raise HTTPException(status_code=400, detail=_("Höchstens {n} Dateien auf einmal. Bitte teile die Auswahl auf.")
                                            .format(n=hoechstens))
                    if info.file_size > grenze:
                        raise HTTPException(status_code=413, detail=zu_gross)
                    with z.open(info) as zf:
                        inhalt = zf.read(grenze + 1)
                    if len(inhalt) > grenze:
                        raise HTTPException(status_code=413, detail=zu_gross)
                    art = h.art_aus_inhalt(inhalt)
                    if art in ("pdf", "docx") and art == h.art_aus_endung(name):
                        out.append((name, inhalt, art))
        except zipfile.BadZipFile:
            out = []
        if not out:
            raise HTTPException(status_code=400, detail=_("In diesem Eintrag ist keine PDF- oder Word-Datei."))
        return out
    if len(daten) > grenze:
        raise HTTPException(status_code=413, detail=zu_gross)
    art = h.art_aus_inhalt(daten)
    if art not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail=_("In diesem Eintrag ist keine PDF- oder Word-Datei."))
    endung = ".pdf" if art == "pdf" else ".docx"
    name = dateiname if (dateiname or "").lower().endswith(endung) else (os.path.splitext(dateiname or "ablage")[0] + endung)
    return [(name, daten, art)]


async def uebernehmen(user: dict, ausgabe_id: int, project_id: Optional[int] = None, request=None, _=None,
                      durch_agent: bool = False) -> dict:
    """-> {"ok": True, "project_id", "neues_projekt", "projekt", "dokumente": [ids], "status", "url", "art"}. Fehler als
    HTTPException mit einem Satz für Menschen (404 Eintrag, Datei oder Projekt; 400 Art passt nicht; 413 zu groß; 429 Tagesgrenze)."""
    import laufbuch
    _ = _ or (lambda s: s)
    m = _main()
    h = _hochladen()
    uid = int(user["id"])
    row = m._ausgabe_row(uid, int(ausgabe_id))
    if not row:
        raise HTTPException(status_code=404, detail=_("Eintrag nicht gefunden"))
    if not m._ablage_pfad_ok(uid, row["datei_pfad"] or ""):
        raise HTTPException(status_code=404, detail=_("Die Datei ist nicht mehr vorhanden. Der Bericht bleibt."))
    liste = dateien(row["datei_pfad"], row["dateiname"] or "", row["media"] or "", m.MAX_UPLOAD_SIZE, h.max_dateien(), _)
    arten = {a for _n, _d, a in liste}
    if len(arten) > 1:
        raise HTTPException(status_code=400, detail=_("In diesem Eintrag liegen PDF- und Word-Dateien. Lade sie einzeln hoch."))
    art = arten.pop()
    ziel: Optional[int] = None
    if project_id is not None:
        conn = m.get_db()
        try:
            p = conn.execute("SELECT id, name, filename, tool, project_type FROM projects WHERE id = ? AND user_id = ?",
                             (int(project_id), uid)).fetchone()
        finally:
            conn.close()
        if not p:
            raise HTTPException(status_code=404, detail=_("Projekt nicht gefunden"))
        if not h._passt(dict(p), art):
            raise HTTPException(status_code=400, detail=(_("Eine PDF passt nur in ein PDF-Projekt. Lade sie in ein neues Projekt.")
                                                         if art == "pdf" else
                                                         _("Eine Word-Datei passt nur in ein Word-Projekt. Lade sie in ein neues Projekt.")))
        ziel = int(p["id"])
    elif durch_agent and h._zaehler(uid, h._tag()) >= h.max_neue_projekte_tag():
        raise HTTPException(status_code=429, detail=_("Heute sind schon {n} neue Projekte über den InkluAgent angelegt worden. Morgen "
                                                      "geht es weiter; in ein bestehendes Projekt kannst du weiter hochladen.")
                            .format(n=h.max_neue_projekte_tag()))
    neu = ziel is None
    doks: list = []
    for i, (name, inhalt, _art) in enumerate(liste):
        r = await m.upload_file(file=h._Datei(name, inhalt), project_id=ziel, user=user, request=request)
        ziel = int(r["project_id"])
        if r.get("document_id"):
            doks.append(int(r["document_id"]))
        if neu and i == 0:   # das neue Projekt entstand mit der ersten Datei; die weiteren kommen als Dokumente dazu
            h._namen_setzen(ziel, uid, projektname(row, art, _))
            if durch_agent:
                h._zaehlen(uid, h._tag())
    # Das Dokument heisst wie in der Herkunft, nicht wie die Datei („inkludocs_….pdf“) — bei einer Datei, ohne eigenen Namen
    if len(liste) == 1 and doks and (row["dokument_name"] or "").strip():
        conn = m.get_db()
        try:
            conn.execute("UPDATE documents SET display_name = ? WHERE id = ? AND (display_name IS NULL OR display_name = '')",
                         (row["dokument_name"].strip()[:200], doks[0]))
            conn.commit()
        finally:
            conn.close()
    # Laufbuch: waehrend die Datei ausgelesen wird, zeigt das Live-Fenster das Ziel (nur, wenn der Agent es tut)
    with laufbuch.lauf("uebernehmen", user_id=uid, project_id=ziel, document_id=(doks[0] if doks else None), gesamt=1) as lauf:
        if doks:
            lauf.stueck(document_id=doks[0])
        status = await h._warten_auf_auslesen(ziel, _AUSLESEN_S)
        lauf.weiter()
    conn = m.get_db()
    try:
        p = conn.execute("SELECT name, filename FROM projects WHERE id = ?", (ziel,)).fetchone()
    finally:
        conn.close()
    return {"ok": True, "project_id": ziel, "neues_projekt": neu, "projekt": ((p["name"] or p["filename"]) if p else "") or "",
            "dokumente": doks, "status": status, "art": art, "url": f"/app?projekt={ziel}"}
