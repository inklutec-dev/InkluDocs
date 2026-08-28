"""Quickinfo-Werkzeug: Schnittstellen (API) fuer PDF-Formulare (27.08.2026).

Eigener Router mit eigenen Tabellen (formularfelder, stammdaten) — bewusst
KEINE Wiederverwendung der Bild-/Alt-Text-Tabellen und -Endpunkte (Steve
27.08.2026: "professionelle Loesung, kein Pflaster"). Ein Formularfeld ist
kein Bild; Chatbot, Pipeline, Statistik und Abrechnung sollen die beiden
Dinge nie verwechseln koennen.

Einbindung: main.py ruft build_router(Abhaengigkeiten) auf und haengt den
Router an die App. Die Abhaengigkeiten (Auth, DB, Verzeichnisse, Abrechnung,
Export-Helfer, CSV-Schutz) kommen von main.py — so gibt es keinen Ringimport
und Login-Regel, Credits und Dateinamen-Helfer bleiben an EINER Stelle. Die
Realpath-Pruefungen fuer Bild- und Originaldateien stehen hier (gleiche
Regel wie im Word-Export: nur unterhalb von RESULTS_DIR bzw. UPLOAD_DIR).

Review-Runde 27.08.2026 (unabhaengiger Reviewer, 32 Befunde, die kritischen
und mittleren hier behoben): Kopfzeilen nur ASCII (sonst 500 nach Verbuchung),
Credits erst nach fertiger Antwort und nur bei geschriebenen Quickinfos,
CSV-Formelschutz, Export im Executor (Event-Loop bleibt frei), Temp-Dateien je
Anfrage, Fehlerpfad der Extraktion setzt Status "error" und raeumt auf,
haengende "extracting"-Projekte werden beim Start zurueckgesetzt, Stammdaten
ueberschreiben nie Hand-/PDF-Texte und nie namenlose Felder, JSON-Koerper
werden geprueft (400 statt 500), Dubletten-Schutz im PATCH, Kappe je Konto.

Endpunkte (alle nur fuer den eingeloggten Besitzer des Projekts):
  GET    /api/projects/{pid}/felder                  Felder + Dokumente + Stammdaten-Treffer
  PATCH  /api/felder/{fid}                            Quickinfo speichern (Hand)
  POST   /api/felder/{fid}/original                   zurueck auf die Quickinfo aus der PDF
  POST   /api/felder/{fid}/stammdaten                 Quickinfo dieses Feldes in die Stammdaten
  POST   /api/felder/{fid}/stammdaten-uebernehmen     Stammdaten-Eintrag in dieses Feld
  POST   /api/projects/{pid}/stammdaten-anwenden      Stammdaten auf alle offenen Felder
  GET    /api/felder/{fid}/ausschnitt                 Bildausschnitt (PNG)
  GET    /api/felder/{fid}/page-view                  Seitenansicht mit Rahmen (PNG)
  POST   /api/projects/{pid}/quickinfos/generieren    Stufe 2: KI-Vorschlaege fuer offene Felder (Hintergrund, je Seite 1 Credit)
  POST   /api/felder/{fid}/generieren                 Stufe 2: ein Feld neu generieren (ueberschreibt, 1 Credit)
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
import logging
import os
import re
import shutil
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

import formular_export
import formular_ki
from formular_processor import extract_formular

log = logging.getLogger(__name__)

# Grenzen fuer Eingaben (Abwehr, nicht Fachlichkeit).
MAX_QUICKINFO = 1000
MAX_TEXTFELD = 300
MAX_IMPORT_BYTES = 1024 * 1024
MAX_IMPORT_ZEILEN = 5000
MAX_STAMMDATEN_JE_KONTO = 20000
FELDARTEN = ("", "text", "checkbox", "radio", "dropdown", "liste", "button", "signatur", "unbekannt")
# Werte von formularfelder.quelle: "" (offen), pdf, hand, stammdaten, ki, gast (28.08.: vom Gast bearbeitet).
# Stammdaten duerfen nur "" und "stammdaten" ersetzen — nie Hand oder PDF-Original.
QUELLEN_ERSETZBAR = ("", "stammdaten")


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
    csv_safe: Callable                 # Formel-Injection-Schutz fuer CSV-Zellen (main._csv_safe)
    # GAST-ANSICHT (28.08.2026): Wachposten der Freigabe aus main.py — (request, token)
    # -> shares-Zeile oder HTTPException(401); guest_session(request, token) -> dict|None.
    require_guest: Callable = None
    guest_session: Callable = None


_d: Optional[Deps] = None


def _user(request: Request) -> dict:
    return _d.get_current_user(request)


async def _json_body(request: Request) -> dict:
    """JSON-Koerper als dict, sonst 400 (statt eines 500 aus request.json())."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Ungueltiger JSON-Koerper")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON-Objekt erwartet")
    return data


def _ascii_header(text: str) -> str:
    """HTTP-Kopfzeilen sind latin-1; alles andere wuerde die Antwort mit 500 abbrechen."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _content_disposition(dateiname: str) -> str:
    """RFC 6266: ASCII-Fallback + filename* in UTF-8 fuer Namen mit Umlauten."""
    ascii_name = dateiname.encode("ascii", "ignore").decode("ascii").replace('"', "") or "download"
    return "attachment; filename=\"%s\"; filename*=UTF-8''%s" % (ascii_name, urllib.parse.quote(dateiname))


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
    try:
        d["ki_hinweise"] = json.loads(d.get("ki_hinweise") or "[]")
    except Exception:
        d["ki_hinweise"] = []
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
    if str(feld.get("anker") or "").startswith("#"):
        return []   # namenlose Felder koennen nie zurueckgeschrieben werden
    name = (feld.get("feld_name") or "").strip()
    norm = _norm_beschriftung(feld.get("beschriftung") or "")
    art = feld.get("feld_art") or ""
    index = _stammdaten_index(eintraege)
    treffer = []
    gesehen = set()
    for e in (index["name"].get(name, []) if name else []):
        if (not e["feld_art"]) or e["feld_art"] == art:
            treffer.append(("name", e)); gesehen.add(e["id"])
    for e in (index["norm"].get(norm, []) if norm else []):
        if e["id"] not in gesehen and ((not e["feld_art"]) or e["feld_art"] == art):
            treffer.append(("beschriftung", e))
    rang = {"name": 0, "beschriftung": 1}
    treffer.sort(key=lambda t: (rang[t[0]], 0 if t[1]["sprache"] == sprache else 1, -int(t[1]["verwendet"] or 0)))
    return [dict(t[1], treffer_art=t[0]) for t in treffer]


def _stammdaten_index(eintraege: list[dict]) -> dict:
    """Nachschlage-Indizes (Feldname, Beschriftungs-Norm) — einmal je Liste,
    damit der Abgleich O(Felder) statt O(Felder x Eintraege) bleibt."""
    if eintraege and "_index" in eintraege[0]:
        return eintraege[0]["_index"]
    idx = {"name": {}, "norm": {}}
    for e in eintraege:
        if e["feld_name"]:
            idx["name"].setdefault(e["feld_name"], []).append(e)
        if e["beschriftung_norm"]:
            idx["norm"].setdefault(e["beschriftung_norm"], []).append(e)
    if eintraege:
        eintraege[0]["_index"] = idx
    return idx


def _stammdaten_laden(conn, user_id: int) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM stammdaten WHERE user_id = ? ORDER BY verwendet DESC, id", (user_id,)).fetchall()]


def _stammdaten_anwenden(conn, project: dict, felder: list, nur_offene: bool = True) -> int:
    """Traegt Stammdaten in Felder ein (quelle 'stammdaten'). Gibt die Anzahl zurueck.
    nur_offene=True: nur Felder ohne Quickinfo. nur_offene=False: zusaetzlich
    Felder, deren Text selbst aus Stammdaten kam. Hand-Texte und PDF-Originale
    werden NIE ersetzt; namenlose Felder (Anker "#n") nie befuellt."""
    eintraege = _stammdaten_laden(conn, project["user_id"])
    if not eintraege:
        return 0
    sprache = project.get("alt_language") or "de"
    anzahl = 0
    for f in felder:
        hat_text = bool((f["quickinfo"] or "").strip())
        if hat_text and (nur_offene or (f.get("quelle") or "") not in QUELLEN_ERSETZBAR):
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
    if feld_art not in FELDARTEN:
        raise HTTPException(status_code=400, detail="Unbekannte Feldart")
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
    anzahl = conn.execute("SELECT COUNT(*) FROM stammdaten WHERE user_id = ?", (user_id,)).fetchone()[0]
    if anzahl >= MAX_STAMMDATEN_JE_KONTO:
        raise HTTPException(status_code=400, detail=f"Die Stammdaten-Bibliothek ist voll (maximal {MAX_STAMMDATEN_JE_KONTO} Eintraege)")
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


def _extraktion_fehlgeschlagen(project_id: int, document_id: int, file_path: str, out_dir: str,
                               is_append: bool, grund: str) -> None:
    """Fehlerpfad der Extraktion: Dokumentzeile und angefangene Felder weg, Status
    zurueck ('extracted' beim Anhaengen, sonst 'error'), Upload-Datei und
    Bildordner entfernen (sonst Waisen auf der Platte)."""
    log.error("[formular] Extraktion fehlgeschlagen (Projekt %s, Dokument %s): %s", project_id, document_id, grund)
    conn = _d.get_db()
    try:
        conn.execute("DELETE FROM formularfelder WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.execute("UPDATE projects SET status = ? WHERE id = ?", ("extracted" if is_append else "error", project_id))
        conn.commit()
    finally:
        conn.close()
    wurzel_up = os.path.realpath(_d.upload_dir) + os.sep
    wurzel_res = os.path.realpath(_d.results_dir) + os.sep
    try:
        if file_path and os.path.realpath(file_path).startswith(wurzel_up) and os.path.isfile(file_path):
            os.unlink(file_path)
        if out_dir and os.path.realpath(out_dir).startswith(wurzel_res) and os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
    except OSError:
        pass


async def _extract_in_background(project_id: int, document_id: int, doc_index: int,
                                 file_path: str, user_id: int, is_append: bool):
    out_dir = os.path.join(_d.results_dir, str(user_id), str(project_id), f"doc{doc_index}")
    os.makedirs(out_dir, exist_ok=True)
    loop = asyncio.get_running_loop()
    try:
        felder, hinweise = await loop.run_in_executor(None, extract_formular, file_path, out_dir, project_id)
    except Exception as e:
        _extraktion_fehlgeschlagen(project_id, document_id, file_path, out_dir, is_append, repr(e))
        return

    conn = _d.get_db()
    try:
        seiten_mit_text: set = set()
        for f in felder:
            # Seitentext nur am ersten Feld jeder Seite speichern (Bandbreite und
            # Speicher: bei 2000 Feldern sonst 2000 Kopien des Seitentextes).
            page_text = ""
            if f.get("page_number") and f.get("page_number") not in seiten_mit_text:
                page_text = f.get("page_text") or ""
                seiten_mit_text.add(f.get("page_number"))
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
                 f.get("ausschnitt_path") or "", f.get("page_view_path") or "", page_text))
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
    except Exception as e:
        conn.close()
        conn = None
        _extraktion_fehlgeschlagen(project_id, document_id, file_path, out_dir, is_append, "DB-Phase: " + repr(e))
        return
    finally:
        if conn is not None:
            conn.close()


def haengende_extraktionen_zuruecksetzen() -> int:
    """Beim Start: Formular-Projekte, die (z. B. durch einen Neustart mitten in der
    Extraktion) auf 'extracting' stehen geblieben sind, auf 'extracted' bzw.
    'error' setzen; Dokumente ohne Felder werden entfernt. Gibt die Anzahl zurueck."""
    conn = _d.get_db()
    try:
        # Auch 'processing' (Stufe 2, Generierung lief beim Neustart) zurueck auf 'extracted'.
        conn.execute("UPDATE projects SET status = 'extracted' WHERE tool = 'formular' AND status = 'processing'")
        rows = conn.execute("SELECT id FROM projects WHERE tool = 'formular' AND status = 'extracting'").fetchall()
        for r in rows:
            pid = r["id"]
            conn.execute("""DELETE FROM documents WHERE project_id = ? AND NOT EXISTS
                            (SELECT 1 FROM formularfelder f WHERE f.document_id = documents.id)""", (pid,))
            hat_felder = conn.execute("SELECT COUNT(*) FROM formularfelder WHERE project_id = ?", (pid,)).fetchone()[0]
            conn.execute("UPDATE projects SET status = ? WHERE id = ?", ("extracted" if hat_felder else "error", pid))
        conn.commit()
        if rows:
            log.warning("[formular] %d haengende Extraktion(en) beim Start zurueckgesetzt", len(rows))
        return len(rows)
    finally:
        conn.close()


# --------------------------------------------------------------------------- Stufe 2: Feld-Pass im Hintergrund

# Laufstatus je Projekt (nur im Prozess; nach Neustart setzt _startup den Status zurueck).
_generierung: dict[int, dict] = {}


def _originalpfad(doc: dict) -> str:
    src = doc.get("original_path") or ""
    wurzel = os.path.realpath(_d.upload_dir) + os.sep
    if not src or not os.path.realpath(src).startswith(wurzel) or not os.path.isfile(src):
        raise HTTPException(status_code=404, detail="Die Originaldatei dieses Dokuments ist nicht mehr vorhanden")
    return src


def _seitenzahl(doc: dict) -> int:
    try:
        h = json.loads(doc.get("hinweise") or "{}")
        return int(h.get("seiten") or 1)
    except Exception:
        return 1


def _feld_fuer_ki(f: dict) -> dict:
    """Feld-Zeile -> Eingabe fuer den Feld-Pass (ohne Pfade, ohne Werte)."""
    try:
        optionen = json.loads(f.get("optionen") or "[]")
    except Exception:
        optionen = []
    try:
        seiten = json.loads(f.get("seiten") or "[]")
    except Exception:
        seiten = []
    rect = None
    if f.get("rect_x0") is not None:
        rect = (f["rect_x0"], f["rect_y0"], f["rect_x1"], f["rect_y1"])
    return {"id": f["id"], "feld_index": f["feld_index"], "feld_art": f.get("feld_art") or "unbekannt", "rect": rect,
            "pflicht": bool(f.get("pflicht")), "optionen": optionen, "beschriftung": f.get("beschriftung") or "",
            "beschriftung_lage": f.get("beschriftung_lage") or "", "gruppe": f.get("gruppe") or "", "seiten": seiten,
            "quickinfo_original": f.get("quickinfo_original") or "", "anker": f.get("anker") or ""}


def _bestaetigte_quickinfos(conn, project_id: int, ausser_feld: Optional[int] = None) -> list[tuple[str, str]]:
    """(Beschriftung, Quickinfo) aller beschriebenen Felder des Projekts — Konsistenz-Vorgabe."""
    rows = conn.execute(
        """SELECT beschriftung, quickinfo FROM formularfelder WHERE project_id = ? AND TRIM(COALESCE(quickinfo,'')) != ''
           AND TRIM(COALESCE(beschriftung,'')) != '' ORDER BY document_id, page_number, feld_index""", (project_id,)).fetchall()
    out, gesehen = [], set()
    for r in rows:
        key = r["beschriftung"].strip().lower()
        if key in gesehen:
            continue
        gesehen.add(key)
        out.append((r["beschriftung"], r["quickinfo"]))
    return out


def _user_prompt(conn, project_id: int) -> str:
    row = conn.execute(
        "SELECT up.prompt_text FROM user_prompts up JOIN projects p ON p.prompt_id = up.id AND p.user_id = up.user_id WHERE p.id = ?",
        (project_id,)).fetchone()
    return (row["prompt_text"] if row and row["prompt_text"] else "")


def _modus_bedingung(modus: str) -> str:
    """SQL-Bedingung, welche Felder ein Sammellauf anfasst. 'luecken' (Standard) = nur Felder
    ohne Quickinfo; 'ki_neu' (28.08.2026, Steve: Knopf „Alle neu generieren“) = zusaetzlich alle
    KI-Vorschlaege — Texte von Hand, aus der PDF, aus Stammdaten oder vom Gast bleiben unberuehrt."""
    if modus == "ki_neu":
        return "(TRIM(COALESCE(quickinfo,'')) = '' OR quelle = 'ki')"
    return "TRIM(COALESCE(quickinfo,'')) = ''"


async def _generiere_projekt(project_id: int, user_id: int, document_id: Optional[int], modus: str = "luecken") -> None:
    """Hintergrundlauf: je Seite ein Feld-Pass fuer die OFFENEN Felder (modus 'ki_neu': auch
    KI-Vorschlaege), Kontingent je Seite geprueft, 1 Credit je Seite; Fehler je Seite, nicht je Projekt."""
    st = _generierung.setdefault(project_id, {"laeuft": True, "seiten_gesamt": 0, "seiten_fertig": 0, "felder_neu": 0, "fehler": []})
    loop = asyncio.get_running_loop()
    conn = _d.get_db()
    try:
        project = dict(conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone())
        docs = {d["id"]: dict(d) for d in conn.execute("SELECT * FROM documents WHERE project_id = ?", (project_id,)).fetchall()}
        sql = "SELECT * FROM formularfelder WHERE project_id = ? AND " + _modus_bedingung(modus) + " AND anker NOT LIKE '#%'"
        args = [project_id]
        if document_id is not None:
            sql += " AND document_id = ?"; args.append(document_id)
        offen = [dict(r) for r in conn.execute(sql + " ORDER BY document_id, page_number, feld_index", args).fetchall()]
        user_prompt = _user_prompt(conn, project_id)
    finally:
        conn.close()
    seiten: dict[tuple, list] = {}
    for f in offen:
        if f.get("page_number"):
            seiten.setdefault((f["document_id"], f["page_number"]), []).append(f)
    st["seiten_gesamt"] = len(seiten)
    alle_vorschlaege: list = []
    felder_by_id = {f["id"]: _feld_fuer_ki(f) for f in offen}
    try:
        for (doc_id, page), felder in seiten.items():
            abo = _d.billing.pruefe_kontingent(user_id)
            if not abo.get("erlaubt", True):
                st["fehler"].append("Credit-Kontingent erschöpft – Rest bleibt offen.")
                break
            doc = docs.get(doc_id) or {}
            try:
                src = _originalpfad(doc)
                conn = _d.get_db()
                try:
                    bestaetigte = _bestaetigte_quickinfos(conn, project_id)
                finally:
                    conn.close()
                vorschlaege = await loop.run_in_executor(
                    None, lambda: formular_ki.generiere_seite(
                        src, page, [felder_by_id[f["id"]] for f in felder], sprache=project.get("alt_language") or "de",
                        formular_titel=_d.doc_label(doc), seiten_gesamt=_seitenzahl(doc), bestaetigte=bestaetigte,
                        user_prompt=user_prompt, variation=False,
                        seitenbild_path=(felder[0].get("page_view_path") or None)))
            except (formular_ki.FeldPassFehler, HTTPException) as e:
                st["fehler"].append(f"Seite {page}: {getattr(e, 'detail', None) or e}")
                st["seiten_fertig"] += 1
                continue
            except Exception as e:  # nie den ganzen Lauf abbrechen
                log.exception("[formular] Feld-Pass Seite %s Projekt %s: %r", page, project_id, e)
                st["fehler"].append(f"Seite {page}: unerwarteter Fehler")
                st["seiten_fertig"] += 1
                continue
            alle_vorschlaege.extend(vorschlaege)
            conn = _d.get_db()
            try:
                for v in vorschlaege:
                    # Nur schreiben, wenn das Feld INZWISCHEN nicht von Hand gefuellt wurde.
                    cur = conn.execute(
                        """UPDATE formularfelder SET quickinfo = ?, quelle = 'ki', sicherheit = ?, beleg = ?, ki_hinweise = ?,
                           updated_at = datetime('now') WHERE id = ? AND """ + _modus_bedingung(modus),
                        (v.quickinfo, v.sicherheit, v.beleg, json.dumps(v.hinweise, ensure_ascii=False), v.feld_id))
                    st["felder_neu"] += cur.rowcount
                conn.commit()
            finally:
                conn.close()
            _d.billing.verbuche(user_id, "generierung", aktion="quickinfo_generierung")
            st["seiten_fertig"] += 1
        # Konsistenz ueber das ganze Dokument (nur KI-Texte dieses Laufs).
        if alle_vorschlaege:
            angeglichen = formular_ki.konsistenz(alle_vorschlaege, felder_by_id)
            conn = _d.get_db()
            try:
                for v in angeglichen:
                    conn.execute("UPDATE formularfelder SET quickinfo = ?, ki_hinweise = ? WHERE id = ? AND quelle = 'ki'",
                                 (v.quickinfo, json.dumps(v.hinweise, ensure_ascii=False), v.feld_id))
                conn.commit()
            finally:
                conn.close()
    finally:
        conn = _d.get_db()
        try:
            conn.execute("UPDATE projects SET status = 'extracted' WHERE id = ? AND status = 'processing'", (project_id,))
            conn.commit()
        finally:
            conn.close()
        st["laeuft"] = False


# --------------------------------------------------------------------------- Router

def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    router = APIRouter()

    @router.on_event("startup")
    async def _startup():
        try:
            haengende_extraktionen_zuruecksetzen()
        except Exception as e:  # Start darf daran nie scheitern (Tabelle fehlt bei Erstlauf o. ae.)
            log.warning("[formular] Start-Reparatur uebersprungen: %r", e)

    # ---- Felder lesen
    def _lade_felder(conn, project_id: int):
        """Dokumente + Felder eines Formular-Projekts in Anzeige-Reihenfolge, mit
        Pruefstatus je Rolle (feld_reviews) und den aktiven Gast-Rollen der Freigaben.
        Gemeinsame Grundlage fuer Besitzer- UND Gast-Ansicht (28.08.2026)."""
        docs = [dict(d) for d in conn.execute(
            """SELECT id, doc_index, original_filename, display_name, extraction_method, created_at, hinweise
               FROM documents WHERE project_id = ? ORDER BY doc_index""", (project_id,)).fetchall()]
        rows = conn.execute(
            """SELECT f.* FROM formularfelder f LEFT JOIN documents d ON d.id = f.document_id
               WHERE f.project_id = ? ORDER BY COALESCE(d.doc_index, 0), f.page_number, f.feld_index""",
            (project_id,)).fetchall()
        felder = [_feld_dict(r) for r in rows]
        reviews = {}
        for r in conn.execute(
                """SELECT r.feld_id, r.role, r.status, r.reviewed_at FROM feld_reviews r
                   JOIN formularfelder f ON f.id = r.feld_id WHERE f.project_id = ?""", (project_id,)).fetchall():
            reviews.setdefault(r["feld_id"], {})[r["role"]] = {"status": r["status"], "reviewed_at": r["reviewed_at"]}
        share_roles = [r["role"] for r in conn.execute(
            "SELECT DISTINCT role FROM shares WHERE project_id = ? AND status IN ('active', 'completed')",
            (project_id,)).fetchall()]
        # Seitentext nur einmal je Seite mitschicken (Bandbreite): am ersten Feld der Seite.
        gesehen = set()
        for f in felder:
            f["reviews"] = reviews.get(f["id"], {})
            key = (f["document_id"], f["page_number"])
            if key in gesehen:
                f["page_text"] = ""
            gesehen.add(key)
        for d in docs:
            d["felder_gesamt"] = sum(1 for f in felder if f["document_id"] == d["id"])
            d["felder_offen"] = sum(1 for f in felder if f["document_id"] == d["id"] and f["status"] == "offen")
            d["felder_unsicher"] = sum(1 for f in felder if f["document_id"] == d["id"] and f["quelle"] == "ki" and f["sicherheit"] == "niedrig")
            try:
                d["hinweise"] = json.loads(d["hinweise"]) if d.get("hinweise") else None
            except Exception:
                d["hinweise"] = None
        return docs, felder, share_roles

    def _projekt_aussen(project: dict) -> dict:
        # Nur, was die Ansicht braucht — keine Serverpfade (original_path) nach aussen.
        aussen = {k: project.get(k) for k in ("id", "name", "filename", "status", "tool", "project_type",
                                               "alt_language", "prompt_id", "created_at", "updated_at")}
        aussen["hat_original"] = bool(project.get("original_path"))
        return aussen

    @router.get("/api/projects/{project_id}/felder")
    async def felder_lesen(project_id: int, user: dict = Depends(_user)):
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            docs, felder, share_roles = _lade_felder(conn, project_id)
            eintraege = _stammdaten_laden(conn, user["id"])
        finally:
            conn.close()
        sprache = project.get("alt_language") or "de"
        treffer = {}
        if eintraege:
            for f in felder:
                t = _stammdaten_treffer(eintraege, f, sprache)
                if t:
                    treffer[f["id"]] = [{"id": e["id"], "quickinfo": e["quickinfo"], "treffer_art": e["treffer_art"]} for e in t[:3]]
        return {"project": _projekt_aussen(project), "documents": docs, "felder": felder, "stammdaten_treffer": treffer,
                "stammdaten_anzahl": len(eintraege), "generierung": _generierung.get(project_id),
                # Pruef-Badges nur, wenn das Projekt ueberhaupt freigegeben wurde (wie bei Bildern).
                "in_review": bool(share_roles), "share_roles": share_roles}

    # ---- Quickinfo speichern
    @router.patch("/api/felder/{feld_id}")
    async def feld_speichern(feld_id: int, request: Request, user: dict = Depends(_user)):
        data = await _json_body(request)
        if "quickinfo" not in data:
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
        data = await _json_body(request)
        try:
            sid = int(data.get("stammdaten_id"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="stammdaten_id fehlt")
        conn = _d.get_db()
        try:
            feld = _feld_des_nutzers(conn, feld_id, user["id"])
            if str(feld["anker"] or "").startswith("#"):
                raise HTTPException(status_code=400, detail="Dieses Feld hat keinen Feldnamen und kann nicht beschrieben werden")
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

    # ---- Stufe 2: KI-Vorschlaege
    @router.post("/api/projects/{project_id}/quickinfos/generieren")
    async def quickinfos_generieren(project_id: int, request: Request, user: dict = Depends(_user)):
        """Startet den Feld-Pass fuer alle OFFENEN Felder (optional nur ein Dokument)
        im Hintergrund; das Frontend pollt GET /felder ("generierung"). Vorhandene
        Texte werden nie ueberschrieben (Regel wie "alle generieren" bei Alt-Texten) —
        AUSSER modus 'ki_neu' (Knopf „Alle neu generieren“, 28.08.2026): dann werden
        KI-Vorschlaege ueberschrieben, Hand/PDF/Stammdaten/Gast bleiben."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        modus = "ki_neu" if data.get("modus") == "ki_neu" else "luecken"
        document_id = data.get("document_id") if isinstance(data, dict) else None
        try:
            document_id = int(document_id) if document_id is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="document_id ungueltig")
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            if project.get("status") in ("extracting", "processing"):
                raise HTTPException(status_code=409, detail="Für dieses Projekt läuft gerade eine Verarbeitung")
            abo = _d.billing.pruefe_kontingent(user["id"])
            if not abo.get("erlaubt", True):
                raise HTTPException(status_code=429, detail="Credit-Kontingent erschoepft. Bitte Credits nachbuchen oder bis zum Monatswechsel warten")
            sql = "SELECT COUNT(*) FROM formularfelder WHERE project_id = ? AND " + _modus_bedingung(modus) + " AND anker NOT LIKE '#%'"
            args = [project_id]
            if document_id is not None:
                sql += " AND document_id = ?"; args.append(document_id)
            offen = conn.execute(sql, args).fetchone()[0]
            if not offen:
                return {"ok": True, "gestartet": False, "offen": 0, "modus": modus}
            conn.execute("UPDATE projects SET status = 'processing' WHERE id = ?", (project_id,))
            conn.commit()
        finally:
            conn.close()
        _generierung[project_id] = {"laeuft": True, "seiten_gesamt": 0, "seiten_fertig": 0, "felder_neu": 0, "fehler": []}
        asyncio.create_task(_generiere_projekt(project_id, user["id"], document_id, modus))
        return {"ok": True, "gestartet": True, "offen": offen, "modus": modus}

    @router.post("/api/felder/{feld_id}/generieren")
    async def feld_generieren(feld_id: int, user: dict = Depends(_user)):
        """Ein Feld (neu) generieren — ueberschreibt bewusst (Variation), 1 Credit.
        "Zurueck auf Original" bleibt moeglich (quickinfo_original)."""
        conn = _d.get_db()
        try:
            feld = dict(_feld_des_nutzers(conn, feld_id, user["id"]))
            if str(feld["anker"] or "").startswith("#"):
                raise HTTPException(status_code=400, detail="Dieses Feld hat keinen Feldnamen und kann nicht beschrieben werden")
            project = dict(conn.execute("SELECT * FROM projects WHERE id = ?", (feld["project_id"],)).fetchone())
            if project.get("status") in ("extracting", "processing"):
                raise HTTPException(status_code=409, detail="Für dieses Projekt läuft gerade eine Verarbeitung")
            abo = _d.billing.pruefe_kontingent(user["id"])
            if not abo.get("erlaubt", True):
                raise HTTPException(status_code=429, detail="Credit-Kontingent erschoepft. Bitte Credits nachbuchen oder bis zum Monatswechsel warten")
            doc = dict(conn.execute("SELECT * FROM documents WHERE id = ?", (feld["document_id"],)).fetchone())
            seite_felder = [dict(r) for r in conn.execute(
                "SELECT * FROM formularfelder WHERE document_id = ? AND page_number = ? ORDER BY feld_index",
                (feld["document_id"], feld["page_number"])).fetchall()]
            bestaetigte = _bestaetigte_quickinfos(conn, feld["project_id"], ausser_feld=feld_id)
            user_prompt = _user_prompt(conn, feld["project_id"])
        finally:
            conn.close()
        src = _originalpfad(doc)
        loop = asyncio.get_running_loop()
        try:
            vorschlaege = await loop.run_in_executor(
                None, lambda: formular_ki.generiere_seite(
                    src, feld["page_number"], [_feld_fuer_ki(feld)], sprache=project.get("alt_language") or "de",
                    formular_titel=_d.doc_label(doc), seiten_gesamt=_seitenzahl(doc), bestaetigte=bestaetigte,
                    user_prompt=user_prompt, variation=True, seitenbild_path=(feld.get("page_view_path") or None)))
        except formular_ki.FeldPassFehler as e:
            raise HTTPException(status_code=502, detail=str(e))
        if not vorschlaege:
            raise HTTPException(status_code=502, detail="Die KI hat für dieses Feld keinen Vorschlag geliefert")
        v = vorschlaege[0]
        conn = _d.get_db()
        try:
            conn.execute("""UPDATE formularfelder SET quickinfo = ?, quelle = 'ki', sicherheit = ?, beleg = ?, ki_hinweise = ?,
                            updated_at = datetime('now') WHERE id = ?""",
                         (v.quickinfo, v.sicherheit, v.beleg, json.dumps(v.hinweise, ensure_ascii=False), feld_id))
            conn.commit()
        finally:
            conn.close()
        _d.billing.verbuche(user["id"], "generierung", aktion="quickinfo_generierung")
        return {"ok": True, "quickinfo": v.quickinfo, "quelle": "ki", "status": "beschrieben",
                "sicherheit": v.sicherheit, "beleg": v.beleg, "ki_hinweise": v.hinweise}

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

    # ---- Gast-Ansicht (Freigabe-Link), 28.08.2026
    # Gegenstueck zu den /api/freigabe/{token}/images/...-Endpunkten in main.py:
    # derselbe Wachposten (Token + bestaetigte Gast-Sitzung), strikt auf DAS eine
    # Projekt der Freigabe begrenzt. Der Gast darf lesen, die Quickinfo von Hand
    # aendern und je Feld ein Urteil setzen (freigegeben / zu_ueberarbeiten, Lektorat
    # zusaetzlich ruecksprache) mit EINER Anmerkung. KEINE KI, keine Stammdaten,
    # kein Export, kein Upload, keine Serverpfade nach aussen.
    GAST_STATUS = ("offen", "in_bearbeitung", "freigegeben", "zu_ueberarbeiten", "ruecksprache")

    def _gast_share(request: Request, token: str) -> dict:
        if not _d.require_guest:
            raise HTTPException(status_code=404, detail="Gastzugang nicht verfuegbar")
        return dict(_d.require_guest(request, token))

    def _gast_feld(conn, share: dict, feld_id: int):
        row = conn.execute("SELECT * FROM formularfelder WHERE id = ? AND project_id = ?",
                           (feld_id, share["project_id"])).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Feld nicht gefunden")
        return row

    def _gast_wieder_aktiv(conn, token: str):
        # Wieder-Einstieg (wie bei Bildern, 15.07.2026): arbeitet ein Gast nach dem
        # Abschluss weiter, springt seine Freigabe zurueck auf 'active'.
        conn.execute("UPDATE shares SET status = 'active' WHERE token = ? AND status = 'completed'", (token,))

    @router.get("/api/freigabe/{token}/felder")
    async def gast_felder(token: str, request: Request):
        share = _gast_share(request, token)
        conn = _d.get_db()
        try:
            project = conn.execute("SELECT * FROM projects WHERE id = ?", (share["project_id"],)).fetchone()
            if not project or project["tool"] != "formular":
                raise HTTPException(status_code=404, detail="Kein Formular-Projekt")
            docs, felder, share_roles = _lade_felder(conn, share["project_id"])
        finally:
            conn.close()
        return {"project": _projekt_aussen(dict(project)), "documents": docs, "felder": felder,
                "guest": True, "role": share.get("role") or "kunde", "in_review": True,
                "share_roles": share_roles}

    def _gast_datei(request: Request, token: str, feld_id: int, spalte: str) -> str:
        share = _gast_share(request, token)
        conn = _d.get_db()
        try:
            feld = _gast_feld(conn, share, feld_id)
        finally:
            conn.close()
        pfad = feld[spalte] or ""
        wurzel = os.path.realpath(_d.results_dir) + os.sep
        if not pfad or not os.path.realpath(pfad).startswith(wurzel) or not os.path.isfile(pfad):
            raise HTTPException(status_code=404, detail="Bild nicht gefunden")
        return pfad

    @router.get("/api/freigabe/{token}/felder/{feld_id}/ausschnitt")
    async def gast_ausschnitt(token: str, feld_id: int, request: Request):
        return FileResponse(_gast_datei(request, token, feld_id, "ausschnitt_path"), media_type="image/png")

    @router.get("/api/freigabe/{token}/felder/{feld_id}/page-view")
    async def gast_seitenansicht(token: str, feld_id: int, request: Request):
        return FileResponse(_gast_datei(request, token, feld_id, "page_view_path"), media_type="image/png")

    @router.post("/api/freigabe/{token}/felder/{feld_id}/quickinfo")
    async def gast_quickinfo(token: str, feld_id: int, request: Request):
        """Gast aendert die Quickinfo von Hand (quelle 'gast'). Ohne gesetztes Urteil
        springt das Feld fuer DIESE Rolle auf 'in_bearbeitung'; ein Urteil wird nie
        ueberschrieben (Spiegel von freigabe_save_alttext)."""
        share = _gast_share(request, token)
        data = await _json_body(request)
        if "quickinfo" not in data:
            raise HTTPException(status_code=400, detail="quickinfo fehlt")
        text = _sauber(data.get("quickinfo"), MAX_QUICKINFO)
        role = share.get("role") or "kunde"
        conn = _d.get_db()
        try:
            feld = _gast_feld(conn, share, feld_id)
            if (feld["anker"] or "").startswith("#"):
                raise HTTPException(status_code=400, detail="Dieses Feld hat keinen Feldnamen und kann nicht beschrieben werden.")
            quelle = "gast" if text else ""
            if text and text == (feld["quickinfo_original"] or ""):
                quelle = "pdf"
            conn.execute("UPDATE formularfelder SET quickinfo = ?, quelle = ?, updated_at = datetime('now') WHERE id = ?",
                         (text, quelle, feld_id))
            cur = conn.execute("SELECT status FROM feld_reviews WHERE feld_id = ? AND role = ?", (feld_id, role)).fetchone()
            auto_status = None
            if not cur or (cur["status"] or "offen") == "offen":
                conn.execute(
                    "INSERT INTO feld_reviews (feld_id, role, status, reviewed_at) VALUES (?, ?, 'in_bearbeitung', datetime('now')) "
                    "ON CONFLICT(feld_id, role) DO UPDATE SET status = 'in_bearbeitung', reviewed_at = datetime('now')",
                    (feld_id, role))
                conn.execute("UPDATE formularfelder SET review_status = 'in_bearbeitung', reviewed_at = datetime('now') WHERE id = ?",
                             (feld_id,))
                auto_status = "in_bearbeitung"
            _gast_wieder_aktiv(conn, token)
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "quickinfo": text, "quelle": quelle, "status": "beschrieben" if text else "offen",
                "auto_status": auto_status, "role": role}

    @router.post("/api/freigabe/{token}/felder/{feld_id}/review")
    async def gast_review(token: str, feld_id: int, request: Request):
        """Gast setzt das Urteil zu einem Feld + optional die eine Anmerkung
        (leer = Anmerkung loeschen). Spiegel von freigabe_set_review."""
        share = _gast_share(request, token)
        data = await _json_body(request)
        role = share.get("role") or "kunde"
        status = data.get("status", "offen")
        if status not in GAST_STATUS:
            raise HTTPException(status_code=400, detail="Ungueltiger Status.")
        if status == "ruecksprache" and role != "lektorat":
            raise HTTPException(status_code=403, detail="Ruecksprache kann nur das Lektorat setzen.")
        comment = _sauber(data.get("comment"), 2000) if data.get("comment") else ""
        conn = _d.get_db()
        try:
            _gast_feld(conn, share, feld_id)
            conn.execute(
                "INSERT INTO feld_reviews (feld_id, role, status, reviewed_at) VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(feld_id, role) DO UPDATE SET status = excluded.status, reviewed_at = excluded.reviewed_at",
                (feld_id, role, status))
            conn.execute("UPDATE formularfelder SET review_status = ?, reviewed_at = datetime('now'), review_note = ? WHERE id = ?",
                         (status, comment, feld_id))
            _gast_wieder_aktiv(conn, token)
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "review_status": status, "comment": comment, "role": role}

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
        """Synchron (laeuft im Executor): Originalpfad pruefen, Quickinfos schreiben."""
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

    def _export_bauen(einheiten: list, work_dir: str, custom_name: Optional[str], project: dict,
                      einzeln: bool) -> tuple[str, str, dict]:
        """Synchron (Executor): erzeugt PDF oder ZIP in work_dir. Liefert (Pfad, Download-Name, Info)."""
        if einzeln:
            einheit = einheiten[0]
            out_path, info = _pdf_fuer_dokument(einheit, work_dir, custom_name)
            name = f"inkludocs_{custom_name or _d.doc_label(einheit['doc'])}_quickinfos.pdf"
            return out_path, name, info
        zip_base = custom_name or _d.safe_filename_component(project.get("name") or project.get("filename") or "projekt")
        zip_path = os.path.join(work_dir, f"{zip_base}_alle_formulare.zip")
        warnungen, gesamt, geschrieben, offen = [], 0, 0, 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pos, einheit in enumerate(einheiten, start=1):
                out_path, info = _pdf_fuer_dokument(einheit, work_dir, None)
                inner = f"{pos:02d}_{_d.doc_label(einheit['doc'])}_quickinfos.pdf"
                zf.write(out_path, arcname=inner)
                gesamt += info["gesamt"]; geschrieben += info["geschrieben"]; offen += info["offen"]
                warnungen += [f"[{inner}] {w}" for w in info["warnungen"]]
        return zip_path, f"{zip_base}_alle_formulare.zip", {"geschrieben": geschrieben, "gesamt": gesamt, "offen": offen,
                                                            "warnungen": warnungen, "writer": formular_export.aktiver_writer()}

    def _alte_exportordner_aufraeumen(export_root: str, behalten: int = 3) -> None:
        """Nur die juengsten Export-Arbeitsordner (f_*) behalten."""
        try:
            ordner = sorted((os.path.join(export_root, n) for n in os.listdir(export_root) if n.startswith("f_")),
                            key=os.path.getmtime)
            for alt in ordner[:-behalten]:
                shutil.rmtree(alt, ignore_errors=True)
        except OSError:
            pass

    @router.post("/api/projects/{project_id}/export/formular")
    async def export_formular(project_id: int, request: Request, user: dict = Depends(_user)):
        document_id, custom_name = await _d.read_export_options(request)
        conn = _d.get_db()
        try:
            project = _projekt_des_nutzers(conn, project_id, user["id"])
            einheiten = _export_einheiten(conn, project, document_id)
        finally:
            conn.close()
        if not einheiten:
            raise HTTPException(status_code=400, detail="Dieses Projekt enthaelt noch kein Formular")
        abo = _d.billing.pruefe_kontingent(user["id"])
        kostenpflichtig = abo.get("plan") != "free"
        if kostenpflichtig and not abo.get("erlaubt", True):
            raise HTTPException(status_code=429, detail="Credit-Kontingent erschoepft. Bitte Credits nachbuchen oder bis zum Monatswechsel warten")

        # Eigener Arbeitsordner je Anfrage (parallele Exporte desselben Projekts
        # kommen sich sonst in die Quere); die Datei bleibt fuer den Download liegen,
        # aeltere Arbeitsordner werden beim naechsten Export aufgeraeumt.
        export_root = os.path.join(_d.results_dir, str(user["id"]), str(project_id), "_export")
        os.makedirs(export_root, exist_ok=True)
        _alte_exportordner_aufraeumen(export_root)
        work_dir = tempfile.mkdtemp(prefix="f_", dir=export_root)
        einzeln = document_id is not None or len(einheiten) == 1
        loop = asyncio.get_running_loop()
        try:
            out_path, name, info = await loop.run_in_executor(
                None, _export_bauen, einheiten, work_dir, custom_name, project, einzeln)
        except HTTPException:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            log.exception("[formular] Export fehlgeschlagen (Projekt %s): %r", project_id, e)
            raise HTTPException(status_code=500, detail="Der Export ist fehlgeschlagen")

        headers = {"X-Export-Method": "formular", "X-Export-Writer": _ascii_header(info.get("writer") or ""),
                   "X-Export-Tagged": str(info["geschrieben"]), "X-Export-Total": str(info["gesamt"]),
                   "X-Export-Open": str(info["offen"])}
        if info["warnungen"]:
            headers["X-Export-Warnings"] = json.dumps(info["warnungen"], ensure_ascii=True)
        # Antwort zuerst bauen, dann verbuchen — und nur, wenn wirklich etwas geschrieben wurde.
        response = FileResponse(out_path, filename=name, media_type="application/pdf" if einzeln else "application/zip",
                                headers=headers)
        if kostenpflichtig and info["geschrieben"] > 0:
            _d.billing.verbuche(user["id"], "export", aktion="formular_export")
        return response

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
        cs = _d.csv_safe   # Formel-Injection-Schutz: Feldnamen/Texte stammen aus fremden PDFs
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";")
        w.writerow(["Nummer", "Name", "Quickinfo", "Type-Nr", "Type", "Seite", "Dokument", "Beschriftung", "Abschnitt", "Pflicht", "Quelle"])
        typ_nr = {"unbekannt": 0, "button": 1, "radio": 2, "checkbox": 3, "text": 4, "dropdown": 5, "liste": 6, "signatur": 7}
        for einheit in einheiten:
            label = _d.doc_label(einheit["doc"])
            for f in einheit["felder"]:
                w.writerow([f["feld_index"], cs(f["feld_name"]), cs(f["quickinfo"] or ""), typ_nr.get(f["feld_art"], 0), f["feld_art"],
                            f["page_number"] or "", cs(label), cs(f["beschriftung"] or ""), cs(f["gruppe"] or ""),
                            1 if f["pflicht"] else 0, f["quelle"] or ""])
        base = custom_name or _d.safe_filename_component(project.get("name") or "formular")
        return Response(content="﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": _content_disposition(f"{base}_quickinfos.csv")})

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
        art = _sauber(data.get("feld_art"), 20).lower()
        if art not in FELDARTEN:
            raise HTTPException(status_code=400, detail="Unbekannte Feldart")
        sprache = (_sauber(data.get("sprache"), 5) or "de").lower()
        return {"beschriftung": _sauber(data.get("beschriftung"), MAX_TEXTFELD), "feld_art": art,
                "feld_name": _sauber(data.get("feld_name"), MAX_TEXTFELD),
                "quickinfo": _sauber(data.get("quickinfo"), MAX_QUICKINFO), "sprache": sprache}

    @router.post("/api/stammdaten")
    async def stammdaten_anlegen(request: Request, user: dict = Depends(_user)):
        data = await _json_body(request)
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
        data = await _json_body(request)
        conn = _d.get_db()
        try:
            alt = conn.execute("SELECT * FROM stammdaten WHERE id = ? AND user_id = ?", (sid, user["id"])).fetchone()
            if not alt:
                raise HTTPException(status_code=404, detail="Eintrag nicht gefunden")
            e = _eintrag_aus_body({**dict(alt), **data})
            if not e["quickinfo"]:
                raise HTTPException(status_code=400, detail="Die Quickinfo darf nicht leer sein")
            norm = _norm_beschriftung(e["beschriftung"])
            if not norm and not e["feld_name"]:
                raise HTTPException(status_code=400, detail="Bitte eine Beschriftung oder einen Feldnamen angeben")
            # Dubletten-Schutz: derselbe Schluessel darf nicht auf einen zweiten Eintrag zeigen.
            if norm:
                dublette = conn.execute("SELECT id FROM stammdaten WHERE user_id = ? AND beschriftung_norm = ? AND feld_art = ? AND id != ?",
                                        (user["id"], norm, e["feld_art"], sid)).fetchone()
            else:
                dublette = conn.execute("SELECT id FROM stammdaten WHERE user_id = ? AND beschriftung_norm = '' AND feld_name = ? AND feld_art = ? AND id != ?",
                                        (user["id"], e["feld_name"], e["feld_art"], sid)).fetchone()
            if dublette:
                raise HTTPException(status_code=409, detail="Es gibt schon einen Eintrag mit dieser Beschriftung und Feldart")
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
        cs = _d.csv_safe
        for r in rows:
            w.writerow([cs(r["beschriftung"]), r["feld_art"], cs(r["feld_name"]), cs(r["quickinfo"]), r["sprache"]])
        return Response(content="﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": _content_disposition("inkludocs_stammdaten.csv")})

    @router.post("/api/stammdaten/import")
    async def stammdaten_import(file: UploadFile = File(...), user: dict = Depends(_user)):
        """CSV (Semikolon oder Komma, UTF-8, Kopfzeile Beschriftung;Feldart;Feldname;Quickinfo;Sprache).
        Gleiche Schluessel werden aktualisiert, nichts wird geloescht."""
        inhalt = await file.read()
        if len(inhalt) > MAX_IMPORT_BYTES:
            raise HTTPException(status_code=413, detail="Die CSV ist zu gross (maximal 1 MB)")
        text = inhalt.decode("utf-8-sig", errors="replace")
        class _Semikolon(csv.excel):
            delimiter = ";"
        try:
            dialekt = csv.Sniffer().sniff(text[:2000], delimiters=";,")
        except Exception:
            dialekt = _Semikolon
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
                # CSV-Zellen koennen aus unserem eigenen Export mit Apostroph-Schutz stammen — den wieder abnehmen.
                def ohne_schutz(v: str) -> str:
                    return v[1:] if v[:1] == "'" and v[1:2] in ("=", "+", "-", "@") else v
                try:
                    _stammdaten_upsert(conn, user["id"], ohne_schutz(zelle(zeile, "beschriftung")), art, ohne_schutz(zelle(zeile, "feldname")),
                                       ohne_schutz(zelle(zeile, "quickinfo")), (zelle(zeile, "sprache") or "de").strip().lower()[:5], "import")
                    uebernommen += 1
                except HTTPException as e:
                    if "voll" in str(e.detail):
                        raise
                    uebersprungen += 1
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "uebernommen": uebernommen, "uebersprungen": uebersprungen}

    return router
