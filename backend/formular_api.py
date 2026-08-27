"""Quickinfo-Werkzeug: Schnittstellen (API) fuer PDF-Formulare (27.08.2026).

Eigener Router mit eigenen Tabellen (formularfelder, stammdaten) — bewusst
KEINE Wiederverwendung der Bild-/Alt-Text-Tabellen und -Endpunkte (Steve
27.08.2026: "professionelle Loesung, kein Pflaster"). Ein Formularfeld ist
kein Bild; Chatbot, Pipeline, Statistik und Abrechnung sollen die beiden
Dinge nie verwechseln koennen.

Einbindung: main.py ruft build_router(Abhaengigkeiten) auf und haengt den
Router an die App. Die Abhaengigkeiten (Auth, DB, Verzeichnisse, Abrechnung,
Export-Helfer) kommen von main.py — so gibt es keinen Ringimport und die
Regeln (Login-Cookie, Realpath-Schutz, Credits) bleiben an EINER Stelle.

Endpunkte (alle nur fuer den eingeloggten Besitzer des Projekts):
  GET    /api/projects/{pid}/felder                  Felder + Dokumente + Stammdaten-Treffer
  PATCH  /api/felder/{fid}                            Quickinfo speichern (Hand)
  POST   /api/felder/{fid}/original                   zurueck auf die Quickinfo aus der PDF
  POST   /api/felder/{fid}/stammdaten                 Quickinfo dieses Feldes in die Stammdaten
  POST   /api/felder/{fid}/stammdaten-uebernehmen     Stammdaten-Eintrag in dieses Feld
  POST   /api/projects/{pid}/stammdaten-anwenden      Stammdaten auf alle offenen Felder
  GET    /api/felder/{fid}/ausschnitt                 Bildausschnitt (PNG)
  GET    /api/felder/{fid}/page-view                  Seitenansicht mit Rahmen (PNG)
  POST   /api/projects/{pid}/export/formular          PDF mit Quickinfos (einzeln/ZIP), kostet Credits
  POST   /api/projects/{pid}/export/formular_csv      Feldliste als CSV (Heine-kompatibel), kostenlos
  GET    /api/stammdaten                              Bibliothek des Kontos
  POST   /api/stammdaten                              Eintrag anlegen
  PATCH  /api/stammdaten/{sid}                        Eintrag aendern
  DELETE /api/stammdaten/{sid}                        Eintrag loeschen
  GET    /api/stammdaten/export.csv                   Bibliothek als CSV
  POST   /api/stammdaten/import                       CSV einspielen

DATENSCHUTZ: Feldwerte werden nie gespeichert (siehe formular_processor);
Stammdaten gehoeren dem Konto (user_id) und werden nie kontouebergreifend
gelesen. Alle Eingaben werden laengenbegrenzt und von Steuerzeichen befreit.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

import formular_export
from formular_processor import extract_formular

# Grenzen fuer Eingaben (Abwehr, nicht Fachlichkeit).
MAX_QUICKINFO = 1000
MAX_TEXTFELD = 300
MAX_IMPORT_BYTES = 1024 * 1024
MAX_IMPORT_ZEILEN = 5000
FELDARTEN = ("", "text", "checkbox", "radio", "dropdown", "liste", "button", "signatur", "unbekannt")
QUELLEN = ("", "pdf", "hand", "stammdaten", "ki")


@dataclass
class Deps:
    get_current_user: Callable
    get_db: Callable
    upload_dir: str
    results_dir: str
    billing: object
    read_export_options: Callable      # async (request) -> (document_id, filename)
    safe_filename_component: Callable
    doc_label: Callable


_d: Optional[Deps] = None


def _user(request: Request) -> dict:
    return _d.get_current_user(request)


# --------------------------------------------------------------------------- Helfer

def _sauber(text: Optional[str], max_len: int) -> str:
    """Einzeiliger Text: Steuerzeichen raus, Zeilenumbrueche und Tabs zu
    Leerzeichen (eine Quickinfo ist ein Satz, kein Absatz — PDF-/TU-Strings
    kennen keine sinnvollen Umbrueche), Whitespace normalisieren, Laenge begrenzen."""
    if text is None:
        return ""
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(text))
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r" {2,}", " ", t).strip()
    return t[:max_len]


def _norm_beschriftung(text: str) -> str:
    """Vergleichsform einer Beschriftung: klein, ohne Doppelpunkt/Sternchen am
    Ende, ohne Mehrfach-Leerzeichen. Klammern bleiben (Formatangaben zaehlen)."""
    t = _sauber(text, MAX_TEXTFELD).lower()
    t = re.sub(r"[\s:*]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _feld_dict(row) -> dict:
    d = dict(row)
    for k in ("seiten", "optionen"):
        try:
            d[k] = json.loads(d.get(k) or "[]")
        except Exception:
            d[k] = []
    d["pflicht"] = bool(d.get("pflicht"))
    d["ausgefuellt"] = bool(d.get("ausgefuellt"))
    d["status"] = "beschrieben" if (d.get("quickinfo") or "").strip() else "offen"
    # Pfade sind intern — nach aussen nur "vorhanden ja/nein".
    d["hat_ausschnitt"] = bool(d.pop("ausschnitt_path", ""))
    d["hat_seitenansicht"] = bool(d.pop("page_view_path", ""))
    return d


def _projekt_des_nutzers(conn, project_id: int, user_id: int) -> dict:
    p = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
    if not p:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    if p["tool"] != "formular":
        raise HTTPException(status_code=400, detail="Dieses Projekt ist kein Formular-Projekt")
    return dict(p)


def _feld_des_nutzers(conn, feld_id: int, user_id: int):
    row = conn.execute(
        """SELECT f.* FROM formularfelder f JOIN projects p ON p.id = f.project_id
           WHERE f.id = ? AND p.user_id = ?""", (feld_id, user_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Feld nicht gefunden")
    return row


def ist_formular_projekt(project_id: int, user_id: int) -> bool:
    conn = _d.get_db()
    try:
        p = conn.execute("SELECT tool FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
        return bool(p and p["tool"] == "formular")
    finally:
        conn.close()


# --------------------------------------------------------------------------- Stammdaten-Abgleich

def _stammdaten_treffer(eintraege: list[dict], feld: dict, sprache: str) -> list[dict]:
    """Passende Stammdaten fuer ein Feld, beste zuerst.
    Stufe 1: sicher = gleicher Feldname (+ vertraegliche Feldart),
             wahrscheinlich = gleiche Beschriftung (+ vertraegliche Feldart).
    Sprache: Eintraege in der Projektsprache zuerst, andere danach."""
    name = (feld.get("feld_name") or "").strip()
    norm = _norm_beschriftung(feld.get("beschriftung") or "")
    art = feld.get("feld_art") or ""
    treffer = []
    for e in eintraege:
        art_ok = (not e["feld_art"]) or (e["feld_art"] == art)
        if not art_ok:
            continue
        if name and e["feld_name"] and e["feld_name"] == name:
            treffer.append(("name", e))
        elif norm and e["beschriftung_norm"] and e["beschriftung_norm"] == norm:
            treffer.append(("beschriftung", e))
    rang = {"name": 0, "beschriftung": 1}
    treffer.sort(key=lambda t: (rang[t[0]], 0 if t[1]["sprache"] == sprache else 1, -int(t[1]["verwendet"] or 0)))
    return [dict(t[1], treffer_art=t[0]) for t in treffer]


def _stammdaten_laden(conn, user_id: int) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM stammdaten WHERE user_id = ? ORDER BY verwendet DESC, id", (user_id,)).fetchall()]


def _stammdaten_anwenden(conn, project: dict, felder: list, nur_offene: bool = True) -> int:
    """Traegt Stammdaten in Felder ein (quelle 'stammdaten'). Gibt die Anzahl zurueck."""
    eintraege = _stammdaten_laden(conn, project["user_id"])
    if not eintraege:
        return 0
    sprache = project.get("alt_language") or "de"
    anzahl = 0
    for f in felder:
        if nur_offene and (f["quickinfo"] or "").strip():
            continue
        treffer = _stammdaten_treffer(eintraege, f, sprache)
        if not treffer:
            continue
        e = treffer[0]
        conn.execute("UPDATE formularfelder SET quickinfo = ?, quelle = 'stammdaten', updated_at = datetime('now') WHERE id = ?",
                     (e["quickinfo"], f["id"]))
        conn.execute("UPDATE stammdaten SET verwendet = verwendet + 1 WHERE id = ?", (e["id"],))
        anzahl += 1
    return anzahl


def _stammdaten_upsert(conn, user_id: int, beschriftung: str, feld_art: str, feld_name: str,
                       quickinfo: str, sprache: str, herkunft: str) -> int:
    """Legt einen Eintrag an oder aktualisiert den gleichen Schluessel
    (Beschriftung+Feldart, sonst Feldname+Feldart). Gibt die id zurueck."""
    beschriftung = _sauber(beschriftung, MAX_TEXTFELD)
    feld_name = _sauber(feld_name, MAX_TEXTFELD)
    quickinfo = _sauber(quickinfo, MAX_QUICKINFO)
    norm = _norm_beschriftung(beschriftung)
    if not quickinfo:
        raise HTTPException(status_code=400, detail="Die Quickinfo darf nicht leer sein")
    if not norm and not feld_name:
        raise HTTPException(status_code=400, detail="Bitte eine Beschriftung oder einen Feldnamen angeben")
    if norm:
        vorhanden = conn.execute(
            "SELECT id FROM stammdaten WHERE user_id = ? AND beschriftung_norm = ? AND feld_art = ?",
            (user_id, norm, feld_art)).fetchone()
    else:
        vorhanden = conn.execute(
            "SELECT id FROM stammdaten WHERE user_id = ? AND beschriftung_norm = '' AND feld_name = ? AND feld_art = ?",
            (user_id, feld_name, feld_art)).fetchone()
    if vorhanden:
        conn.execute(
            """UPDATE stammdaten SET quickinfo = ?, feld_name = CASE WHEN ? != '' THEN ? ELSE feld_name END,
               sprache = ?, herkunft = ?, updated_at = datetime('now') WHERE id = ?""",
            (quickinfo, feld_name, feld_name, sprache, herkunft, vorhanden["id"]))
        return int(vorhanden["id"])
    cur = conn.execute(
        """INSERT INTO stammdaten (user_id, beschriftung, beschriftung_norm, feld_art, feld_name, quickinfo, sprache, herkunft)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, beschriftung, norm, feld_art, feld_name, quickinfo, sprache, herkunft))
    return int(cur.lastrowid)


# --------------------------------------------------------------------------- Upload + Extraktion

async def handle_upload(file_path: str, filename: str, user: dict, project_id: int) -> dict:
    """Spiegel von main._handle_pdf_upload fuer Formular-Projekte: Dokument-Zeile
    anlegen, Antwort sofort, Feldextraktion im Hintergrund (Frontend pollt)."""
    conn = _d.get_db()
    try:
        proj = _projekt_des_nutzers(conn, project_id, user["id"])
        is_append = bool(conn.execute("SELECT COUNT(*) FROM documents WHERE project_id = ?", (project_id,)).fetchone()[0])
        if is_append:
            conn.execute("UPDATE projects SET status = 'extracting' WHERE id = ?", (project_id,))
        else:
            conn.execute("UPDATE projects SET filename = ?, original_path = ?, status = 'extracting' WHERE id = ?",
                         (filename, file_path, project_id))
        doc_index = (conn.execute("SELECT COALESCE(MAX(doc_index), 0) FROM documents WHERE project_id = ?",
                                  (project_id,)).fetchone()[0] or 0) + 1
        cur = conn.execute(
            """INSERT INTO documents (project_id, doc_index, original_filename, original_path, extraction_method, total_images)
               VALUES (?, ?, ?, ?, 'formular', 0)""", (project_id, doc_index, filename, file_path))
        document_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    asyncio.create_task(_extract_in_background(project_id, document_id, doc_index, file_path, user["id"], is_append))
    return {"ok": True, "project_id": project_id, "document_id": document_id, "doc_index": doc_index,
            "filename": filename, "project_type": "pdfform", "appended": is_append, "status": "extracting"}


async def _extract_in_background(project_id: int, document_id: int, doc_index: int,
                                 file_path: str, user_id: int, is_append: bool):
    out_dir = os.path.join(_d.results_dir, str(user_id), str(project_id), f"doc{doc_index}")
    os.makedirs(out_dir, exist_ok=True)
    loop = asyncio.get_event_loop()
    try:
        felder, hinweise = await loop.run_in_executor(None, extract_formular, file_path, out_dir, project_id)
    except Exception as e:
        print(f"[formular] Extraktion fehlgeschlagen (Projekt {project_id}, Dokument {document_id}): {e}")
        conn = _d.get_db()
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.execute("UPDATE projects SET status = ? WHERE id = ?", ("extracted" if is_append else "error", project_id))
        conn.commit()
        conn.close()
        return

    conn = _d.get_db()
    try:
        for f in felder:
            rect = f.get("rect") or (None, None, None, None)
            quickinfo = f.get("quickinfo_original") or ""
            conn.execute(
                """INSERT INTO formularfelder
                   (project_id, document_id, feld_index, anker, feld_name, feld_art, page_number, seiten,
                    rect_x0, rect_y0, rect_x1, rect_y1, beschriftung, beschriftung_lage, gruppe, umfeld,
                    optionen, pflicht, ausgefuellt, quickinfo_original, quickinfo, quelle,
                    ausschnitt_path, page_view_path, page_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (project_id, document_id, f["feld_index"], f["anker"], f.get("feld_name") or "", f.get("feld_art") or "unbekannt",
                 f.get("page_number") or 0, json.dumps(f.get("seiten") or []),
                 rect[0], rect[1], rect[2], rect[3],
                 _sauber(f.get("beschriftung"), MAX_TEXTFELD), f.get("beschriftung_lage") or "",
                 _sauber(f.get("gruppe"), MAX_TEXTFELD), f.get("umfeld") or "",
                 json.dumps(f.get("optionen") or [], ensure_ascii=False), 1 if f.get("pflicht") else 0,
                 1 if f.get("ausgefuellt") else 0, quickinfo, quickinfo, "pdf" if quickinfo.strip() else "",
                 f.get("ausschnitt_path") or "", f.get("page_view_path") or "", f.get("page_text") or ""))
        hinweise_json = json.dumps(hinweise, ensure_ascii=False) if (hinweise.get("uebersprungen") or hinweise.get("warnungen")) else ""
        conn.execute("UPDATE documents SET extraction_method = ?, hinweise = ? WHERE id = ?",
                     ("formular-pdfix" if hinweise.get("quelle_liste") == "pdfix" else "formular", hinweise_json, document_id))
        # Stammdaten des Kontos direkt beim Hochladen anwenden (Michaels Wunsch:
        # "in jedes neue Formular importieren") — nur auf Felder ohne Quickinfo.
        project = dict(conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone())
        neue = [dict(r) for r in conn.execute("SELECT * FROM formularfelder WHERE document_id = ?", (document_id,)).fetchall()]
        _stammdaten_anwenden(conn, project, neue, nur_offene=True)
        conn.execute("UPDATE projects SET status = 'extracted' WHERE id = ?", (project_id,))
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- Router

def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    router = APIRouter()

    # ---- Felder lesen
    @router.get("/api/projects/{project_id}/felder")
    async def felder_lesen(project_id: int, user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            docs = [dict(d) for d in conn.execute(
                """SELECT id, doc_index, original_filename, display_name, extraction_method, created_at, hinweise
                   FROM documents WHERE project_id = ? ORDER BY doc_index""", (project_id,)).fetchall()]
            rows = conn.execute(
                """SELECT f.* FROM formularfelder f LEFT JOIN documents d ON d.id = f.document_id
                   WHERE f.project_id = ? ORDER BY COALESCE(d.doc_index, 0), f.page_number, f.feld_index""",
                (project_id,)).fetchall()
            felder = [_feld_dict(r) for r in rows]
            eintraege = _stammdaten_laden(conn, user["id"])
        finally:
            conn.close()
        # Seitentext nur einmal je Seite mitschicken (Bandbreite): am ersten Feld der Seite.
        gesehen = set()
        for f in felder:
            key = (f["document_id"], f["page_number"])
            if key in gesehen:
                f["page_text"] = ""
            gesehen.add(key)
        sprache = project.get("alt_language") or "de"
        treffer = {}
        if eintraege:
            for f in felder:
                t = _stammdaten_treffer(eintraege, f, sprache)
                if t:
                    treffer[f["id"]] = [{"id": e["id"], "quickinfo": e["quickinfo"], "treffer_art": e["treffer_art"]} for e in t[:3]]
        for d in docs:
            d["felder_gesamt"] = sum(1 for f in felder if f["document_id"] == d["id"])
            d["felder_offen"] = sum(1 for f in felder if f["document_id"] == d["id"] and f["status"] == "offen")
            try:
                d["hinweise"] = json.loads(d["hinweise"]) if d.get("hinweise") else None
            except Exception:
                d["hinweise"] = None
        return {"project": project, "documents": docs, "felder": felder, "stammdaten_treffer": treffer,
                "stammdaten_anzahl": len(eintraege)}

    # ---- Quickinfo speichern
    @router.patch("/api/felder/{feld_id}")
    async def feld_speichern(feld_id: int, request: Request, user: dict = Depends(_user)):
        data = await request.json()
        if not isinstance(data, dict) or "quickinfo" not in data:
            raise HTTPException(status_code=400, detail="quickinfo fehlt")
        text = _sauber(data.get("quickinfo"), MAX_QUICKINFO)
        conn = _d.get_db()
        try:
            feld = _feld_des_nutzers(conn, feld_id, user["id"])
            quelle = "hand" if text else ""
            if text and text == (feld["quickinfo_original"] or ""):
                quelle = "pdf"
            conn.execute("UPDATE formularfelder SET quickinfo = ?, quelle = ?, updated_at = datetime('now') WHERE id = ?",
                         (text, quelle, feld_id))
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "quickinfo": text, "quelle": quelle, "status": "beschrieben" if text else "offen"}

    @router.post("/api/felder/{feld_id}/original")
    async def feld_original(feld_id: int, user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            feld = _feld_des_nutzers(conn, feld_id, user["id"])
            text = feld["quickinfo_original"] or ""
            conn.execute("UPDATE formularfelder SET quickinfo = ?, quelle = ?, updated_at = datetime('now') WHERE id = ?",
                         (text, "pdf" if text else "", feld_id))
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "quickinfo": text, "quelle": "pdf" if text else "", "status": "beschrieben" if text else "offen"}

    # ---- Stammdaten <-> Feld
    @router.post("/api/felder/{feld_id}/stammdaten")
    async def feld_in_stammdaten(feld_id: int, user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            feld = _feld_des_nutzers(conn, feld_id, user["id"])
            if not (feld["quickinfo"] or "").strip():
                raise HTTPException(status_code=400, detail="Dieses Feld hat noch keine Quickinfo")
            project = dict(conn.execute("SELECT alt_language FROM projects WHERE id = ?", (feld["project_id"],)).fetchone())
            sid = _stammdaten_upsert(conn, user["id"], feld["beschriftung"], feld["feld_art"], feld["feld_name"],
                                     feld["quickinfo"], project.get("alt_language") or "de", "feld")
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "stammdaten_id": sid}

    @router.post("/api/felder/{feld_id}/stammdaten-uebernehmen")
    async def feld_aus_stammdaten(feld_id: int, request: Request, user: dict = Depends(_user)):
        data = await request.json()
        try:
            sid = int(data.get("stammdaten_id"))
        except (TypeError, ValueError, AttributeError):
            raise HTTPException(status_code=400, detail="stammdaten_id fehlt")
        conn = _d.get_db()
        try:
            _feld_des_nutzers(conn, feld_id, user["id"])
            e = conn.execute("SELECT * FROM stammdaten WHERE id = ? AND user_id = ?", (sid, user["id"])).fetchone()
            if not e:
                raise HTTPException(status_code=404, detail="Stammdaten-Eintrag nicht gefunden")
            conn.execute("UPDATE formularfelder SET quickinfo = ?, quelle = 'stammdaten', updated_at = datetime('now') WHERE id = ?",
                         (e["quickinfo"], feld_id))
            conn.execute("UPDATE stammdaten SET verwendet = verwendet + 1 WHERE id = ?", (sid,))
            conn.commit()
            text = e["quickinfo"]
        finally:
            conn.close()
        return {"ok": True, "quickinfo": text, "quelle": "stammdaten", "status": "beschrieben"}

    @router.post("/api/projects/{project_id}/stammdaten-anwenden")
    async def stammdaten_anwenden(project_id: int, request: Request, user: dict = Depends(_user)):
        try:
            data = await request.json()
        except Exception:
            data = {}
        nur_offene = bool(data.get("nur_offene", True)) if isinstance(data, dict) else True
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            felder = [dict(r) for r in conn.execute("SELECT * FROM formularfelder WHERE project_id = ?", (project_id,)).fetchall()]
            anzahl = _stammdaten_anwenden(conn, project, felder, nur_offene=nur_offene)
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "uebernommen": anzahl}

    # ---- Bilder
    def _datei_des_feldes(feld_id: int, user_id: int, spalte: str) -> str:
        conn = _d.get_db()
        try:
            feld = _feld_des_nutzers(conn, feld_id, user_id)
        finally:
            conn.close()
        pfad = feld[spalte] or ""
        wurzel = os.path.realpath(_d.results_dir) + os.sep
        if not pfad or not os.path.realpath(pfad).startswith(wurzel) or not os.path.isfile(pfad):
            raise HTTPException(status_code=404, detail="Bild nicht gefunden")
        return pfad

    @router.get("/api/felder/{feld_id}/ausschnitt")
    async def feld_ausschnitt(feld_id: int, user: dict = Depends(_user)):
        return FileResponse(_datei_des_feldes(feld_id, user["id"], "ausschnitt_path"), media_type="image/png")

    @router.get("/api/felder/{feld_id}/page-view")
    async def feld_seitenansicht(feld_id: int, user: dict = Depends(_user)):
        return FileResponse(_datei_des_feldes(feld_id, user["id"], "page_view_path"), media_type="image/png")

    # ---- Export
    def _export_einheiten(conn, project: dict, document_id: Optional[int]) -> list[dict]:
        docs = [dict(d) for d in conn.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index",
                                              (project["id"],)).fetchall()]
        if document_id is not None:
            docs = [d for d in docs if d["id"] == document_id]
            if not docs:
                raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
        einheiten = []
        for d in docs:
            felder = [dict(r) for r in conn.execute(
                "SELECT * FROM formularfelder WHERE document_id = ? ORDER BY feld_index", (d["id"],)).fetchall()]
            einheiten.append({"doc": d, "felder": felder})
        return einheiten

    def _pdf_fuer_dokument(einheit: dict, output_dir: str, custom_title: Optional[str]) -> tuple[str, dict]:
        doc = einheit["doc"]
        src = doc.get("original_path") or ""
        wurzel = os.path.realpath(_d.upload_dir) + os.sep
        if not src or not os.path.realpath(src).startswith(wurzel) or not os.path.isfile(src):
            raise HTTPException(status_code=404, detail="Die Originaldatei dieses Dokuments ist nicht mehr vorhanden")
        quickinfos = {f["anker"]: (f["quickinfo"] or "") for f in einheit["felder"]}
        base = custom_title or _d.doc_label(doc)
        out_path = os.path.join(output_dir, "inkludocs_" + _d.safe_filename_component(base) + "_quickinfos.pdf")
        try:
            erg = formular_export.write_quickinfos_to_pdf(src, out_path, quickinfos)
        except formular_export.FormularExportFehler as e:
            raise HTTPException(status_code=500, detail=str(e))
        return out_path, {"geschrieben": erg.geschrieben, "writer": erg.writer, "gesamt": len(einheit["felder"]),
                          "offen": sum(1 for f in einheit["felder"] if not (f["quickinfo"] or "").strip()),
                          "warnungen": erg.warnungen}

    @router.post("/api/projects/{project_id}/export/formular")
    async def export_formular(project_id: int, request: Request, user: dict = Depends(_user)):
        document_id, custom_name = await _d.read_export_options(request)
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            einheiten = _export_einheiten(conn, project, document_id)
        finally:
            conn.close()
        output_dir = os.path.join(_d.results_dir, str(user["id"]), str(project_id), "_export")
        os.makedirs(output_dir, exist_ok=True)
        abo = _d.billing.pruefe_kontingent(user["id"])
        kostenpflichtig = abo.get("plan") != "free"
        if kostenpflichtig and not abo.get("erlaubt", True):
            raise HTTPException(status_code=429, detail="Credit-Kontingent erschoepft. Bitte Credits nachbuchen oder bis zum Monatswechsel warten")

        if document_id is not None or len(einheiten) == 1:
            einheit = einheiten[0]
            out_path, info = _pdf_fuer_dokument(einheit, output_dir, custom_name)
            headers = {"X-Export-Method": "formular", "X-Export-Writer": info.get("writer", ""),
                       "X-Export-Tagged": str(info["geschrieben"]),
                       "X-Export-Total": str(info["gesamt"]), "X-Export-Open": str(info["offen"])}
            if info["warnungen"]:
                headers["X-Export-Warnings"] = json.dumps(info["warnungen"], ensure_ascii=False)
            if kostenpflichtig:
                _d.billing.verbuche(user["id"], "export", aktion="formular_export")
            name = f"inkludocs_{custom_name or _d.doc_label(einheit['doc'])}_quickinfos.pdf"
            return FileResponse(out_path, filename=name, media_type="application/pdf", headers=headers)

        zip_base = custom_name or _d.safe_filename_component(project.get("name") or project.get("filename") or "projekt")
        zip_path = os.path.join(output_dir, f"{zip_base}_alle_formulare.zip")
        warnungen, gesamt, geschrieben = [], 0, 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pos, einheit in enumerate(einheiten, start=1):
                out_path, info = _pdf_fuer_dokument(einheit, output_dir, None)
                inner = f"{pos:02d}_{_d.doc_label(einheit['doc'])}_quickinfos.pdf"
                zf.write(out_path, arcname=inner)
                gesamt += info["gesamt"]
                geschrieben += info["geschrieben"]
                warnungen += [f"[{inner}] {w}" for w in info["warnungen"]]
        headers = {"X-Export-Method": "formular", "X-Export-Tagged": str(geschrieben), "X-Export-Total": str(gesamt)}
        if warnungen:
            headers["X-Export-Warnings"] = json.dumps(warnungen, ensure_ascii=False)
        if kostenpflichtig:
            _d.billing.verbuche(user["id"], "export", aktion="formular_export")
        return FileResponse(zip_path, filename=f"{zip_base}_alle_formulare.zip", media_type="application/zip", headers=headers)

    @router.post("/api/projects/{project_id}/export/formular_csv")
    async def export_formular_csv(project_id: int, request: Request, user: dict = Depends(_user)):
        """Feldliste als CSV — Spalten 1-5 wie Heines Format, danach unsere
        Zusatzspalten. Ohne Feldwerte. Kostenlos (kein Schreibvorgang)."""
        document_id, custom_name = await _d.read_export_options(request)
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            einheiten = _export_einheiten(conn, project, document_id)
        finally:
            conn.close()
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";")
        w.writerow(["Nummer", "Name", "Quickinfo", "Type-Nr", "Type", "Seite", "Dokument", "Beschriftung", "Abschnitt", "Pflicht", "Quelle"])
        typ_nr = {"unbekannt": 0, "button": 1, "radio": 2, "checkbox": 3, "text": 4, "dropdown": 5, "liste": 6, "signatur": 7}
        for einheit in einheiten:
            label = _d.doc_label(einheit["doc"])
            for f in einheit["felder"]:
                w.writerow([f["feld_index"], f["feld_name"], f["quickinfo"] or "", typ_nr.get(f["feld_art"], 0), f["feld_art"],
                            f["page_number"] or "", label, f["beschriftung"] or "", f["gruppe"] or "",
                            1 if f["pflicht"] else 0, f["quelle"] or ""])
        base = custom_name or _d.safe_filename_component(project.get("name") or "formular")
        return Response(content="﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{base}_quickinfos.csv"'})

    # ---- Stammdaten-Bibliothek
    @router.get("/api/stammdaten")
    async def stammdaten_liste(user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            rows = conn.execute("SELECT * FROM stammdaten WHERE user_id = ? ORDER BY beschriftung, feld_name, id",
                                (user["id"],)).fetchall()
        finally:
            conn.close()
        return {"stammdaten": [dict(r) for r in rows]}

    def _eintrag_aus_body(data: dict) -> dict:
        art = (data.get("feld_art") or "").strip()
        if art not in FELDARTEN:
            raise HTTPException(status_code=400, detail="Unbekannte Feldart")
        sprache = (data.get("sprache") or "de").strip().lower()[:5]
        return {"beschriftung": _sauber(data.get("beschriftung"), MAX_TEXTFELD), "feld_art": art,
                "feld_name": _sauber(data.get("feld_name"), MAX_TEXTFELD),
                "quickinfo": _sauber(data.get("quickinfo"), MAX_QUICKINFO), "sprache": sprache}

    @router.post("/api/stammdaten")
    async def stammdaten_anlegen(request: Request, user: dict = Depends(_user)):
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Ungueltige Daten")
        e = _eintrag_aus_body(data)
        conn = _d.get_db()
        try:
            sid = _stammdaten_upsert(conn, user["id"], e["beschriftung"], e["feld_art"], e["feld_name"], e["quickinfo"], e["sprache"], "hand")
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "id": sid}

    @router.patch("/api/stammdaten/{sid}")
    async def stammdaten_aendern(sid: int, request: Request, user: dict = Depends(_user)):
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Ungueltige Daten")
        conn = _d.get_db()
        try:
            alt = conn.execute("SELECT * FROM stammdaten WHERE id = ? AND user_id = ?", (sid, user["id"])).fetchone()
            if not alt:
                raise HTTPException(status_code=404, detail="Eintrag nicht gefunden")
            e = _eintrag_aus_body({**dict(alt), **data})
            if not e["quickinfo"]:
                raise HTTPException(status_code=400, detail="Die Quickinfo darf nicht leer sein")
            conn.execute(
                """UPDATE stammdaten SET beschriftung = ?, beschriftung_norm = ?, feld_art = ?, feld_name = ?, quickinfo = ?,
                   sprache = ?, updated_at = datetime('now') WHERE id = ?""",
                (e["beschriftung"], _norm_beschriftung(e["beschriftung"]), e["feld_art"], e["feld_name"], e["quickinfo"], e["sprache"], sid))
            conn.commit()
        finally:
            conn.close()
        return {"ok": True}

    @router.delete("/api/stammdaten/{sid}")
    async def stammdaten_loeschen(sid: int, user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            cur = conn.execute("DELETE FROM stammdaten WHERE id = ? AND user_id = ?", (sid, user["id"]))
            conn.commit()
            if not cur.rowcount:
                raise HTTPException(status_code=404, detail="Eintrag nicht gefunden")
        finally:
            conn.close()
        return {"ok": True}

    @router.get("/api/stammdaten/export.csv")
    async def stammdaten_export(user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            rows = conn.execute("SELECT * FROM stammdaten WHERE user_id = ? ORDER BY beschriftung, feld_name, id", (user["id"],)).fetchall()
        finally:
            conn.close()
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";")
        w.writerow(["Beschriftung", "Feldart", "Feldname", "Quickinfo", "Sprache"])
        for r in rows:
            w.writerow([r["beschriftung"], r["feld_art"], r["feld_name"], r["quickinfo"], r["sprache"]])
        return Response(content="﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="inkludocs_stammdaten.csv"'})

    @router.post("/api/stammdaten/import")
    async def stammdaten_import(file: UploadFile = File(...), user: dict = Depends(_user)):
        """CSV (Semikolon oder Komma, UTF-8, Kopfzeile Beschriftung;Feldart;Feldname;Quickinfo;Sprache).
        Gleiche Schluessel werden aktualisiert, nichts wird geloescht."""
        inhalt = await file.read()
        if len(inhalt) > MAX_IMPORT_BYTES:
            raise HTTPException(status_code=413, detail="Die CSV ist zu gross (maximal 1 MB)")
        text = inhalt.decode("utf-8-sig", errors="replace")
        try:
            dialekt = csv.Sniffer().sniff(text[:2000], delimiters=";,")
        except Exception:
            dialekt = csv.excel
            dialekt.delimiter = ";"
        reader = csv.reader(io.StringIO(text), dialekt)
        kopf = next(reader, None)
        if not kopf:
            raise HTTPException(status_code=400, detail="Die CSV ist leer")
        spalten = {(k or "").strip().lower(): i for i, k in enumerate(kopf)}
        if "quickinfo" not in spalten or ("beschriftung" not in spalten and "feldname" not in spalten):
            raise HTTPException(status_code=400, detail="Die CSV braucht die Spalten Beschriftung (oder Feldname) und Quickinfo")

        def zelle(zeile, name):
            i = spalten.get(name)
            return zeile[i] if i is not None and i < len(zeile) else ""
        conn = _d.get_db()
        try:
            uebernommen, uebersprungen = 0, 0
            for n, zeile in enumerate(reader):
                if n >= MAX_IMPORT_ZEILEN:
                    break
                art = zelle(zeile, "feldart").strip().lower()
                if art not in FELDARTEN:
                    art = ""
                try:
                    _stammdaten_upsert(conn, user["id"], zelle(zeile, "beschriftung"), art, zelle(zeile, "feldname"),
                                       zelle(zeile, "quickinfo"), (zelle(zeile, "sprache") or "de").strip().lower()[:5], "import")
                    uebernommen += 1
                except HTTPException:
                    uebersprungen += 1
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "uebernommen": uebernommen, "uebersprungen": uebersprungen}

    return router
