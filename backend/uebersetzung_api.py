"""Uebersetzen-Werkzeug: Schnittstellen (API), Upload, Hintergrundlauf, Export (18.09.2026).

Eigener Router mit eigener Tabelle (uebersetzung_segmente) — wie das Quickinfo-Werkzeug
(formular_api.py) bewusst KEINE Wiederverwendung der Bild-/Alt-Text-Tabellen: ein
Absatz ist kein Bild. Der Kern (uebersetzung.py) kennt keine Werkzeuge, nur Segmente;
dieser Router ist der erste Anschluss. Weitere Anschluesse (PowerPoint, Chatbot,
Uebergabe-Knoepfe zwischen Werkzeugen, Public API) haengen sich an dieselben Funktionen:
    segmentiere_und_speichere(), lauf_starten(), export_vorbereiten() + export_bauen().

Einbindung: main.py ruft build_router(Deps) auf. Abhaengigkeiten (Auth, DB, Verzeichnisse,
Abrechnung, Tageslimit, Dateinamen-Helfer) kommen aus main.py — kein Ringimport, die
Regeln bleiben an EINER Stelle.

Endpunkte (nur fuer den eingeloggten Besitzer des Projekts):
  GET    /api/projects/{pid}/uebersetzung                 Stand: Dokumente, Segmente (Original + Uebersetzung), Lauf
  POST   /api/projects/{pid}/uebersetzung/vorschau        Rueckfrage: Woerter, Absaetze, Preis, Guthaben (aendert nichts)
  POST   /api/projects/{pid}/uebersetzung/starten         Lauf im Hintergrund (Zielsprache, Schalter), Credits je Segment-Paket
  POST   /api/projects/{pid}/uebersetzung/abbrechen       laufenden Lauf nach dem aktuellen Paket beenden
  PATCH  /api/uebersetzung/segmente/{sid}                 Uebersetzung eines Absatzes von Hand korrigieren
  POST   /api/projects/{pid}/export/uebersetzung          Word-Datei in der Zielsprache (einzeln oder ZIP)

ABRECHNUNG: 1 Credit je angefangene 100 Woerter (uebersetzung.WOERTER_JE_CREDIT), verbucht
je fertigem Paket, nie fuer Pakete, die nicht kamen. Der Export der Datei ist kostenlos
(der Lauf ist die Leistung). Tageslimit: jeder Modellaufruf zaehlt wie ein Alt-Text.

DATENSCHUTZ: Original- und Zieltext liegen in der Datenbank des Kontos (wie Alt-Texte);
Loeschen von Dokument/Projekt/Konto raeumt sie mit ab. Serverpfade gehen nie nach aussen.

SICHERHEIT: Zugriff immer ueber projects.user_id; JSON-Koerper geprueft (400 statt 500);
Zielsprache nur aus der festen Liste; Handkorrekturen laengenbegrenzt und von
Steuerzeichen befreit; Export liest Originale nur unter UPLOAD_DIR und schreibt atomar
unter RESULTS_DIR; Kopfzeilen ASCII, Dateinamen nach RFC 6266; Export im Executor.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response

import uebersetzung as ue
from docx_processor import DocxFehler, validiere_docx

log = logging.getLogger(__name__)

MAX_HANDTEXT = 20000
TOOL_KEY = "uebersetzen"
PROJECT_TYPE = "docx-uebersetzung"
AKTION = "uebersetzung"               # billing.AKTIONS_PREISE["uebersetzung"] = 1 (je 100 Woerter)


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
    tageslimit_wache: Callable = None
    tageslimit_text: Callable = None
    get_user_by_id: Callable = None


_d: Optional[Deps] = None
# Laufstatus je Projekt (nur im Prozess; nach Neustart setzt _startup den Status zurueck).
_lauf: dict[int, dict] = {}


def _user(request: Request) -> dict:
    return _d.get_current_user(request)


async def _json_body(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Ungültiger JSON-Körper")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON-Objekt erwartet")
    return data


def _content_disposition(dateiname: str) -> str:
    ascii_name = dateiname.encode("ascii", "ignore").decode("ascii").replace('"', "") or "download"
    return "attachment; filename=\"%s\"; filename*=UTF-8''%s" % (ascii_name, urllib.parse.quote(dateiname))


def _sauber(text: Optional[str], max_len: int) -> str:
    if text is None:
        return ""
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(text))
    return t[:max_len]


def _projekt_des_nutzers(conn, project_id: int, user_id: int) -> dict:
    p = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
    if not p:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    if p["tool"] != TOOL_KEY:
        raise HTTPException(status_code=400, detail="Dieses Projekt ist kein Übersetzungsprojekt")
    return dict(p)


def ist_uebersetzungsprojekt(project_id: int, user_id: int) -> bool:
    conn = _d.get_db()
    try:
        p = conn.execute("SELECT tool FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
        return bool(p and p["tool"] == TOOL_KEY)
    finally:
        conn.close()


def _originalpfad(doc: dict) -> str:
    src = doc.get("original_path") or ""
    wurzel = os.path.realpath(_d.upload_dir) + os.sep
    if not src or not os.path.realpath(src).startswith(wurzel) or not os.path.isfile(src):
        raise HTTPException(status_code=404, detail="Die Originaldatei dieses Dokuments ist nicht mehr vorhanden")
    return src


def zielsprachen_aussen() -> list[dict]:
    return [{"code": k, "name": v[0]} for k, v in ue.ZIELSPRACHEN.items()]


# --------------------------------------------------------------------------- Upload + Segmentierung

async def handle_upload(file_path: str, filename: str, user: dict, project_id: int) -> dict:
    """Spiegel von formular_api.handle_upload: Dokumentzeile anlegen, sofort antworten,
    Segmentierung im Hintergrund (das Frontend pollt)."""
    try:
        validiere_docx(file_path)
    except DocxFehler as e:
        try:
            os.unlink(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=str(e))
    conn = _d.get_db()
    try:
        _projekt_des_nutzers(conn, project_id, user["id"])
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
               VALUES (?, ?, ?, ?, 'uebersetzung', 0)""", (project_id, doc_index, filename, file_path))
        document_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    asyncio.create_task(_segmentiere_im_hintergrund(project_id, document_id, file_path, is_append))
    return {"ok": True, "project_id": project_id, "document_id": document_id, "doc_index": doc_index,
            "filename": filename, "project_type": PROJECT_TYPE, "appended": is_append, "status": "extracting"}


def _segmentierung_fehlgeschlagen(project_id: int, document_id: int, file_path: str, is_append: bool, grund: str) -> None:
    log.error("[uebersetzung] Segmentierung fehlgeschlagen (Projekt %s, Dokument %s): %s", project_id, document_id, grund)
    conn = _d.get_db()
    try:
        conn.execute("DELETE FROM uebersetzung_segmente WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.execute("UPDATE projects SET status = ? WHERE id = ?", ("extracted" if is_append else "error", project_id))
        conn.commit()
    finally:
        conn.close()
    wurzel = os.path.realpath(_d.upload_dir) + os.sep
    try:
        if file_path and os.path.realpath(file_path).startswith(wurzel) and os.path.isfile(file_path):
            os.unlink(file_path)
    except OSError:
        pass


def segmentiere_und_speichere(conn, project_id: int, document_id: int, seg: ue.Segmentierung) -> int:
    """Segmente eines Dokuments in die Tabelle schreiben (Anschlusspunkt fuer andere Werkzeuge)."""
    n = 0
    for pos, s in enumerate(seg.segmente, start=1):
        conn.execute(
            """INSERT INTO uebersetzung_segmente
               (project_id, document_id, position, anker, art, ort, abschnitt, abschnitt_titel, kontext, stil,
                ueberschrift_ebene, uebersetzbar, woerter, stuecke, marken, trenner, original, uebersetzung, ziel_stuecke, status, quelle, hinweis)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '{}', 'offen', '', '')""",
            (project_id, document_id, pos, s.anker, s.art, s.ort, s.abschnitt, s.abschnitt_titel[:300], s.kontext[:500], s.stil,
             s.ueberschrift_ebene, 1 if s.uebersetzbar else 0, s.woerter, json.dumps(s.stuecke, ensure_ascii=False),
             json.dumps(s.marken), json.dumps({str(k): v for k, v in s.trenner.items()}), s.text))
        n += 1
    hinweise = {"hinweise": seg.hinweise, "quellsprache": seg.quellsprache, "titel": seg.titel,
                "woerter": seg.woerter, "absaetze": seg.absaetze, "teile": seg.teile}
    conn.execute("UPDATE documents SET hinweise = ?, total_images = ? WHERE id = ?",
                 (json.dumps(hinweise, ensure_ascii=False), seg.absaetze, document_id))
    return n


async def _segmentiere_im_hintergrund(project_id: int, document_id: int, file_path: str, is_append: bool):
    loop = asyncio.get_running_loop()
    try:
        seg = await loop.run_in_executor(None, ue.segmentiere_docx, file_path)
    except Exception as e:
        _segmentierung_fehlgeschlagen(project_id, document_id, file_path, is_append, repr(e))
        return
    conn = _d.get_db()
    try:
        segmentiere_und_speichere(conn, project_id, document_id, seg)
        conn.execute("UPDATE projects SET status = 'extracted' WHERE id = ?", (project_id,))
        conn.commit()
    except Exception as e:
        conn.close()
        conn = None
        _segmentierung_fehlgeschlagen(project_id, document_id, file_path, is_append, "DB-Phase: " + repr(e))
        return
    finally:
        if conn is not None:
            conn.close()


def haengende_laeufe_zuruecksetzen() -> int:
    conn = _d.get_db()
    try:
        conn.execute(f"UPDATE projects SET status = 'extracted' WHERE tool = '{TOOL_KEY}' AND status = 'processing'")
        rows = conn.execute(f"SELECT id FROM projects WHERE tool = '{TOOL_KEY}' AND status = 'extracting'").fetchall()
        for r in rows:
            pid = r["id"]
            conn.execute("""DELETE FROM documents WHERE project_id = ? AND NOT EXISTS
                            (SELECT 1 FROM uebersetzung_segmente s WHERE s.document_id = documents.id)""", (pid,))
            hat = conn.execute("SELECT COUNT(*) FROM uebersetzung_segmente WHERE project_id = ?", (pid,)).fetchone()[0]
            conn.execute("UPDATE projects SET status = ? WHERE id = ?", ("extracted" if hat else "error", pid))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


# --------------------------------------------------------------------------- Lesen

def _segment_dict(row) -> dict:
    d = dict(row)
    for k in ("stuecke", "marken"):
        try:
            d[k] = json.loads(d.get(k) or "[]")
        except Exception:
            d[k] = []
    try:
        d["ziel_stuecke"] = {int(k): v for k, v in (json.loads(d.get("ziel_stuecke") or "{}")).items()}
    except Exception:
        d["ziel_stuecke"] = {}
    d["uebersetzbar"] = bool(d.get("uebersetzbar"))
    # Nach aussen: Original, Uebersetzung, Status, Hinweis, Kontext — keine Marken-Innereien.
    aussen = {k: d[k] for k in ("id", "document_id", "position", "anker", "art", "ort", "abschnitt", "abschnitt_titel",
                                "kontext", "ueberschrift_ebene", "uebersetzbar", "woerter", "original", "uebersetzung",
                                "status", "quelle", "hinweis", "updated_at")}
    return aussen


def _lade(conn, project_id: int):
    docs = [dict(r) for r in conn.execute(
        """SELECT id, doc_index, original_filename, display_name, created_at, hinweise, total_images
           FROM documents WHERE project_id = ? ORDER BY doc_index""", (project_id,)).fetchall()]
    rows = conn.execute(
        """SELECT s.* FROM uebersetzung_segmente s LEFT JOIN documents d ON d.id = s.document_id
           WHERE s.project_id = ? ORDER BY COALESCE(d.doc_index, 0), s.position""", (project_id,)).fetchall()
    segmente = [_segment_dict(r) for r in rows]
    for d in docs:
        try:
            d["hinweise"] = json.loads(d["hinweise"]) if d.get("hinweise") else {}
        except Exception:
            d["hinweise"] = {}
        eigene = [s for s in segmente if s["document_id"] == d["id"]]
        d["absaetze"] = sum(1 for s in eigene if s["uebersetzbar"])
        d["woerter"] = sum(s["woerter"] for s in eigene if s["uebersetzbar"])
        d["fertig"] = sum(1 for s in eigene if s["uebersetzbar"] and s["status"] in ("fertig", "zusammengelegt", "hand"))
        d["mit_hinweis"] = sum(1 for s in eigene if s["hinweis"])
    return docs, segmente


def _projekt_aussen(project: dict) -> dict:
    aussen = {k: project.get(k) for k in ("id", "name", "filename", "status", "tool", "project_type", "alt_language",
                                           "created_at", "updated_at", "lauf_hinweis")}
    aussen["hat_original"] = bool(project.get("original_path"))
    try:
        aussen["einstellungen"] = json.loads(project.get("ki_neu_rest") or "{}")   # Zielsprache/Schalter des letzten Laufs
    except Exception:
        aussen["einstellungen"] = {}
    return aussen


# --------------------------------------------------------------------------- Lauf

def _kandidaten(conn, project_id: int, document_id: Optional[int]) -> list[dict]:
    sql = "SELECT * FROM uebersetzung_segmente WHERE project_id = ? AND uebersetzbar = 1"
    args: list = [project_id]
    if document_id is not None:
        sql += " AND document_id = ?"
        args.append(document_id)
    return [dict(r) for r in conn.execute(sql + " ORDER BY document_id, position", args).fetchall()]


def _segment_aus_zeile(z: dict) -> ue.Segment:
    stuecke = json.loads(z["stuecke"] or "[]")
    marken = json.loads(z["marken"] or "[]")
    trenner = {int(k): v for k, v in json.loads(z["trenner"] or "{}").items()}
    return ue.Segment(anker=z["anker"], part=z["anker"].rsplit("|", 1)[0], art=z["art"], stuecke=stuecke, marken=marken,
                      trenner=trenner, kontext=z.get("kontext") or "", ort=z.get("ort") or "Text",
                      abschnitt=z.get("abschnitt") or 1, abschnitt_titel=z.get("abschnitt_titel") or "",
                      stil=z.get("stil") or "", ueberschrift_ebene=z.get("ueberschrift_ebene"),
                      uebersetzbar=bool(z.get("uebersetzbar")), woerter=z.get("woerter") or 0)


def _einstellungen_pruefen(data: dict) -> dict:
    ziel = str(data.get("zielsprache") or "").strip().lower()
    if ziel not in ue.ZIELSPRACHEN:
        raise HTTPException(status_code=400, detail="Bitte eine Zielsprache aus der Liste wählen")
    return {"zielsprache": ziel,
            "alt_texte": bool(data.get("alt_texte", True)),
            "sprache_setzen": bool(data.get("sprache_setzen", True))}


def _vorschau(conn, user_id: int, project_id: int, document_id: Optional[int]) -> dict:
    kand = _kandidaten(conn, project_id, document_id)
    woerter = sum(int(k["woerter"] or 0) for k in kand)
    preis = ue.credits_fuer(woerter)
    verf = _d.billing.verfuegbare_credits(user_id)
    dokumente = conn.execute("SELECT COUNT(*) FROM documents WHERE project_id = ?", (project_id,)).fetchone()[0]
    machbar = len(kand) if verf is None else (len(kand) if verf >= preis else int(len(kand) * (verf / preis)) if preis else 0)
    return {"anzahl": len(kand), "woerter": woerter, "preis": preis, "preis_je": 1,
            "woerter_je_credit": ue.WOERTER_JE_CREDIT, "verfuegbar": verf,
            "erlaubt": verf is None or verf >= 1, "machbar": machbar, "dokumente": dokumente}


async def lauf_starten(project_id: int, user: dict, einstellungen: dict, document_id: Optional[int]) -> dict:
    """Anschlusspunkt: prueft Guthaben und Tageslimit, setzt Status, startet den Hintergrundlauf."""
    conn = _d.get_db()
    try:
        project = _projekt_des_nutzers(conn, project_id, user["id"])
        if project.get("status") in ("extracting", "processing"):
            raise HTTPException(status_code=409, detail="Für dieses Projekt läuft gerade eine Verarbeitung")
        v = _vorschau(conn, user["id"], project_id, document_id)
        if not v["anzahl"]:
            return {"ok": True, "gestartet": False, "anzahl": 0}
        wache = _d.billing.aktion_pruefung(user["id"], AKTION, 1)
        if not wache["erlaubt"]:
            raise HTTPException(status_code=402, detail=_d.billing.credits_fehlen_detail(wache, "Das Übersetzen"))
        tl = _d.tageslimit_wache(user) if _d.tageslimit_wache else None
        if tl:
            raise HTTPException(status_code=429, detail=_d.tageslimit_text(tl))
        conn.execute("UPDATE projects SET status = 'processing', ki_neu_rest = ?, lauf_hinweis = '' WHERE id = ?",
                     (json.dumps(einstellungen), project_id))
        conn.commit()
    finally:
        conn.close()
    _lauf[project_id] = {"laeuft": True, "pakete_gesamt": 0, "pakete_fertig": 0, "segmente_fertig": 0,
                         "segmente_gesamt": v["anzahl"], "credits": 0, "fehler": [], "abbruch": False}
    asyncio.create_task(_uebersetze_projekt(project_id, user["id"], einstellungen, document_id))
    return {"ok": True, "gestartet": True, "anzahl": v["anzahl"], "woerter": v["woerter"], "preis": v["preis"]}


async def _uebersetze_projekt(project_id: int, user_id: int, einstellungen: dict, document_id: Optional[int]) -> None:
    st = _lauf.setdefault(project_id, {"laeuft": True, "pakete_gesamt": 0, "pakete_fertig": 0, "segmente_fertig": 0,
                                       "segmente_gesamt": 0, "credits": 0, "fehler": [], "abbruch": False})
    loop = asyncio.get_running_loop()
    ziel = einstellungen["zielsprache"]
    conn = _d.get_db()
    try:
        kand = _kandidaten(conn, project_id, document_id)
        if not einstellungen.get("alt_texte", True):
            kand = [k for k in kand if k["art"] not in ("alt", "titel")]
        docs = {d["id"]: dict(d) for d in conn.execute("SELECT * FROM documents WHERE project_id = ?", (project_id,)).fetchall()}
    finally:
        conn.close()
    stand_by_id = {k["id"]: (k.get("updated_at") or "") for k in kand}
    lauf_user = _d.get_user_by_id(user_id) if _d.get_user_by_id else None
    # Pakete je Dokument (Kontext/Titel je Dokument)
    je_doc: dict[int, list] = {}
    for k in kand:
        je_doc.setdefault(k["document_id"], []).append(k)
    pakete: list[tuple[int, list]] = []
    for did, zeilen in je_doc.items():
        nummeriert = [(z["id"], _segment_aus_zeile(z)) for z in zeilen]
        for b in ue._batches(nummeriert):
            pakete.append((did, b))
    st["pakete_gesamt"] = len(pakete)
    try:
        for did, batch in pakete:
            if st.get("abbruch"):
                st["fehler"].append("Vom Nutzer abgebrochen – der Rest bleibt unübersetzt.")
                break
            woerter = sum(seg.woerter for _nr, seg in batch)
            preis = ue.credits_fuer(woerter)
            wache = _d.billing.aktion_pruefung(user_id, AKTION, preis)
            if not wache["erlaubt"]:
                st["fehler"].append(f"Guthaben reicht nicht für das nächste Paket ({preis} Credits nötig, "
                                    f"{0 if wache['verfuegbar'] is None else wache['verfuegbar']} vorhanden) – der Rest bleibt unübersetzt.")
                break
            tl = _d.tageslimit_wache(lauf_user) if (_d.tageslimit_wache and lauf_user) else None
            if tl:
                st["fehler"].append(f"Tageslimit erreicht ({tl['limit']} KI-Aufrufe pro Tag) – der Rest bleibt unübersetzt.")
                break
            doc = docs.get(did) or {}
            try:
                h = json.loads(doc.get("hinweise") or "{}")
            except Exception:
                h = {}
            try:
                ergebnisse = await loop.run_in_executor(
                    None, lambda: ue.uebersetze_batch(batch, ziel, h.get("quellsprache") or "", h.get("titel") or ""))
            except Exception as e:  # nie den ganzen Lauf abbrechen
                log.exception("[uebersetzung] Paket fehlgeschlagen (Projekt %s): %r", project_id, e)
                st["fehler"].append(f"Ein Paket mit {len(batch)} Absätzen konnte nicht übersetzt werden ({type(e).__name__}).")
                st["pakete_fertig"] += 1
                continue
            if st.get("abbruch"):
                st["fehler"].append("Vom Nutzer abgebrochen – das laufende Paket wurde verworfen.")
                break
            conn = _d.get_db()
            geschrieben = 0
            try:
                by_nr = {nr: seg for nr, seg in batch}
                for e in ergebnisse:
                    seg = by_nr[e.nr]
                    text = "".join(e.ziel_stuecke.get(i, seg.stuecke[i]) for i in range(len(seg.stuecke))) if e.ziel_stuecke else ""
                    cur = conn.execute(
                        """UPDATE uebersetzung_segmente SET uebersetzung = ?, ziel_stuecke = ?, status = ?, quelle = 'ki', hinweis = ?,
                           zielsprache = ?, updated_at = datetime('now') WHERE id = ? AND COALESCE(updated_at, '') = ?""",
                        (text, json.dumps({str(k): v for k, v in e.ziel_stuecke.items()}, ensure_ascii=False), e.status,
                         e.hinweis, ziel, e.nr, stand_by_id.get(e.nr, "")))
                    if cur.rowcount and e.status != "fehler":
                        geschrieben += 1
                conn.commit()
            finally:
                conn.close()
            st["segmente_fertig"] += geschrieben
            if geschrieben:
                # Preis anteilig nach tatsaechlich geschriebenen Segmenten (nichts geschrieben = nichts verbucht)
                anteil = ue.credits_fuer(sum(seg.woerter for nr, seg in batch if any(r.nr == nr and r.status != "fehler" for r in ergebnisse)))
                if anteil:
                    _d.billing.verbuche(user_id, "sammellauf", aktion=AKTION, credits=anteil)
                    st["credits"] += anteil
            st["pakete_fertig"] += 1
    finally:
        conn = _d.get_db()
        try:
            hinweis = " ".join(st["fehler"])[:1000]
            conn.execute("UPDATE projects SET status = 'extracted', lauf_hinweis = ? WHERE id = ? AND status = 'processing'",
                         (hinweis, project_id))
            conn.commit()
        finally:
            conn.close()
        st["laeuft"] = False


# --------------------------------------------------------------------------- Export

def export_vorbereiten(conn, project: dict, document_id: Optional[int]) -> list[dict]:
    """Liest alles, was der Export braucht, auf der Anfrage-Verbindung (SQLite-Verbindungen
    duerfen nicht in den Executor-Thread): je Dokument Quelle, Ziele, Sprache."""
    docs = [dict(d) for d in conn.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index", (project["id"],)).fetchall()]
    if document_id is not None:
        docs = [d for d in docs if d["id"] == document_id]
        if not docs:
            raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    try:
        einst = json.loads(project.get("ki_neu_rest") or "{}")
    except Exception:
        einst = {}
    auftraege = []
    for doc in docs:
        src = _originalpfad(doc)
        rows = conn.execute(
            """SELECT anker, ziel_stuecke, status, zielsprache FROM uebersetzung_segmente
               WHERE document_id = ? AND status IN ('fertig', 'zusammengelegt', 'hand')""", (doc["id"],)).fetchall()
        ziele = {}
        sprache = None
        for r in rows:
            try:
                stk = {int(k): v for k, v in json.loads(r["ziel_stuecke"] or "{}").items()}
            except Exception:
                continue
            if stk:
                ziele[r["anker"]] = stk
            sprache = sprache or r["zielsprache"]
        if not ziele:
            raise HTTPException(status_code=400, detail=f"„{_d.doc_label(doc)}“ ist noch nicht übersetzt.")
        sprache = sprache or einst.get("zielsprache")
        auftraege.append({"doc_id": doc["id"], "doc_index": doc["doc_index"], "src": src, "ziele": ziele, "sprache": sprache,
                          "name": f"{_d.doc_label(doc)}_{sprache}.docx",
                          "sprache_setzen": einst.get("sprache_setzen", True)})
    return auftraege


def export_bauen(auftraege: list[dict], ziel_dir: str) -> list[tuple[str, str, dict]]:
    """Anschlusspunkt: baut je Dokument die uebersetzte Datei (ohne Datenbank, executor-tauglich).
    Rueckgabe [(dateipfad, anzeigename, info)]. Bricht ab, wenn die Struktur abweicht."""
    out = []
    for a in auftraege:
        pfad = os.path.join(ziel_dir, f"doc{a['doc_index']}_{a['sprache']}.docx")
        erg = ue.schreibe_uebersetzung(a["src"], pfad, a["ziele"], sprache_ziel=a["sprache"] if a["sprache_setzen"] else None)
        abweichung = ue.strukturvergleich(a["src"], pfad)
        if abweichung:
            # Darf nicht passieren — lieber Fehler als eine Datei mit veraenderter Struktur ausliefern.
            log.error("[uebersetzung] Strukturabweichung beim Export (Dokument %s): %s", a["doc_id"], abweichung)
            raise HTTPException(status_code=500, detail="Die Struktur der übersetzten Datei weicht vom Original ab. Der Export wurde abgebrochen.")
        out.append((pfad, a["name"], {"geschrieben": erg.geschrieben, "warnungen": erg.warnungen}))
    return out


# --------------------------------------------------------------------------- Router

def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    router = APIRouter()

    @router.on_event("startup")
    async def _startup():
        try:
            haengende_laeufe_zuruecksetzen()
        except Exception as e:
            log.warning("[uebersetzung] Start-Reparatur übersprungen: %r", e)

    @router.get("/api/projects/{project_id}/uebersetzung")
    async def lesen(project_id: int, user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            docs, segmente = _lade(conn, project_id)
        finally:
            conn.close()
        return {"project": _projekt_aussen(project), "documents": docs, "segmente": segmente,
                "lauf": _lauf.get(project_id), "zielsprachen": zielsprachen_aussen()}

    @router.post("/api/projects/{project_id}/uebersetzung/vorschau")
    async def vorschau(project_id: int, request: Request, user: dict = Depends(_user)):
        try:
            data = await request.json()
        except Exception:
            data = {}
        document_id = data.get("document_id") if isinstance(data, dict) else None
        try:
            document_id = int(document_id) if document_id is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="document_id ungültig")
        conn = _d.get_db()
        try:
            _projekt_des_nutzers(conn, project_id, user["id"])
            return _vorschau(conn, user["id"], project_id, document_id)
        finally:
            conn.close()

    @router.post("/api/projects/{project_id}/uebersetzung/starten")
    async def starten(project_id: int, request: Request, user: dict = Depends(_user)):
        data = await _json_body(request)
        einst = _einstellungen_pruefen(data)
        document_id = data.get("document_id")
        try:
            document_id = int(document_id) if document_id is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="document_id ungültig")
        return await lauf_starten(project_id, user, einst, document_id)

    @router.post("/api/projects/{project_id}/uebersetzung/abbrechen")
    async def abbrechen(project_id: int, user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            _projekt_des_nutzers(conn, project_id, user["id"])
        finally:
            conn.close()
        st = _lauf.get(project_id)
        if st and st.get("laeuft"):
            st["abbruch"] = True
            return {"ok": True, "angefordert": True}
        return {"ok": True, "angefordert": False}

    @router.patch("/api/uebersetzung/segmente/{segment_id}")
    async def segment_speichern(segment_id: int, request: Request, user: dict = Depends(_user)):
        data = await _json_body(request)
        if "uebersetzung" not in data:
            raise HTTPException(status_code=400, detail="uebersetzung fehlt")
        text = _sauber(data.get("uebersetzung"), MAX_HANDTEXT)
        conn = _d.get_db()
        try:
            row = conn.execute(
                """SELECT s.* FROM uebersetzung_segmente s JOIN projects p ON p.id = s.project_id
                   WHERE s.id = ? AND p.user_id = ?""", (segment_id, user["id"])).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Absatz nicht gefunden")
            seg = _segment_aus_zeile(dict(row))
            if not seg.uebersetzbar:
                raise HTTPException(status_code=400, detail="Dieser Absatz enthält keinen übersetzbaren Text")
            if text.strip():
                # Handtext: die Formatierung im Absatz wird zusammengelegt (ein Stueck), ehrlich gemeldet
                ziel = ue.ersatz_zusammenlegen(seg, text)
                status = "hand"
                hinweis = ("Von Hand korrigiert. Die Formatierung innerhalb dieses Absatzes (z. B. Fettung einzelner Wörter) "
                           "gilt für den ganzen Absatz.") if len(seg.marken) > 1 else "Von Hand korrigiert."
                quelle = "hand"
            else:
                ziel, status, hinweis, quelle = {}, "offen", "", ""
            conn.execute(
                """UPDATE uebersetzung_segmente SET uebersetzung = ?, ziel_stuecke = ?, status = ?, quelle = ?, hinweis = ?,
                   updated_at = datetime('now') WHERE id = ?""",
                ("".join(ziel.get(i, seg.stuecke[i]) for i in range(len(seg.stuecke))) if ziel else "",
                 json.dumps({str(k): v for k, v in ziel.items()}, ensure_ascii=False), status, quelle, hinweis, segment_id))
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "status": status, "quelle": quelle, "hinweis": hinweis, "uebersetzung": text if ziel else ""}

    @router.post("/api/projects/{project_id}/export/uebersetzung")
    async def export(project_id: int, request: Request, user: dict = Depends(_user)):
        document_id, wunschname = await _d.read_export_options(request)
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            if project.get("status") in ("extracting", "processing"):
                raise HTTPException(status_code=409, detail="Bitte warten, bis die Übersetzung fertig ist")
            auftraege = export_vorbereiten(conn, project, document_id)
        finally:
            conn.close()
        ziel_dir = os.path.join(_d.results_dir, str(user["id"]), str(project_id), "_export")
        os.makedirs(ziel_dir, exist_ok=True)
        arbeits = tempfile.mkdtemp(prefix="u_", dir=ziel_dir)
        loop = asyncio.get_running_loop()
        dateien = await loop.run_in_executor(None, export_bauen, auftraege, arbeits)
        _alte_arbeitsordner_aufraeumen(ziel_dir, behalten=arbeits)
        warnungen = [w for _p, _n, info in dateien for w in info.get("warnungen", [])]
        kopf = {"X-Export-Total": str(len(dateien)), "X-Export-Warnings": json.dumps(warnungen, ensure_ascii=True)}
        if len(dateien) == 1:
            pfad, name, _info = dateien[0]
            if wunschname:
                name = f"{_d.safe_filename_component(wunschname)}.docx"
            kopf["Content-Disposition"] = _content_disposition(name)
            return FileResponse(pfad, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                headers=kopf)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for pfad, name, _info in dateien:
                z.write(pfad, name)
        zipname = f"{_d.safe_filename_component(wunschname or project.get('name') or 'projekt')}_uebersetzt.zip"
        kopf["Content-Disposition"] = _content_disposition(zipname)
        return Response(buf.getvalue(), media_type="application/zip", headers=kopf)

    return router


def _alte_arbeitsordner_aufraeumen(ziel_dir: str, behalten: str, max_alt: int = 3) -> None:
    """Die drei juengsten Export-Arbeitsordner bleiben (laufende Downloads), aeltere weg."""
    try:
        ordner = sorted((os.path.join(ziel_dir, n) for n in os.listdir(ziel_dir) if n.startswith("u_")),
                        key=os.path.getmtime, reverse=True)
        for o in ordner[max_alt:]:
            if o != behalten:
                shutil.rmtree(o, ignore_errors=True)
    except OSError:
        pass
