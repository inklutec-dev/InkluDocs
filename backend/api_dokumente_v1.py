"""InkluDocs Public API v1 — DOKUMENTE (17.09.2026, Steve: „die API soll alles abdecken“).

Anlass: Jens (alirodocs) will InkluDocs in seine Portalloesung einbauen; Steve will die API
unabhaengig davon als eigenes Produkt (Einzelbilder UND Dokumente, spaeter MCP). Bisher gab
es nur die Einzelbild-Endpunkte /api/v1/alt-text (main.py). Dieses Modul haengt die ganze
Dokument-Pipeline der App an denselben Schluessel: hochladen, extrahieren, generieren,
Texte lesen und korrigieren, exportieren, Gast-Freigabe, loeschen.

GRUNDSATZ: KEINE zweite Pipeline. Jeder Endpunkt hier ist eine duenne Schicht (Schluessel-
Auth, Rate-Limit, maschinenlesbare Fehler, Antwortform) ueber den BESTEHENDEN App-Routen,
die per Pfad aus der App nachgeschlagen und direkt aufgerufen werden (_route). Besitz-
pruefung, Credits, Tageslimit, Export-Abnahme, Dateipruefung — alles bleibt an EINER Stelle
in main.py/formular_api.py. Aendert sich die App, aendert sich die API mit.

Ein „Dokument“ der API = ein Projekt der App mit genau einer Datei (bzw. einer Webadresse).
Werkzeug nach Dateityp: .pdf -> pdf (oder formular, wenn tool=formular), .docx -> word,
Bilddatei -> grafik, JSON {url} -> web. Jedes Dokument hat dieselbe Gestalt: id, kind,
status, counts, items. Ein Item ist ein Bild (type image) oder ein Formularfeld (type field).

Endpunkte (alle mit Kopfzeile X-API-Key; * = schreibend, zaehlt im Minuten-/Tageslimit je Schluessel;
lesende Aufrufe haben eine eigene Bremse von LESE_LIMIT_MINUTE pro Minute):
  POST   /api/v1/documents                          * Datei (multipart) oder {url} (JSON) -> 202/201
  GET    /api/v1/documents                            eigene Dokumente, ?limit=&offset=
  GET    /api/v1/documents/{id}                       Status, Zaehler, Fortschritt
  POST   /api/v1/documents/{id}/generate            * Sammellauf starten (Hintergrund) -> 202
  POST   /api/v1/documents/{id}/generate/cancel     * laufenden Sammellauf beenden
  GET    /api/v1/documents/{id}/items                 Bilder bzw. Felder mit Texten
  GET    /api/v1/documents/{id}/items/{item_id}/file  Bilddatei (zum Anzeigen beim Partner)
  PATCH  /api/v1/documents/{id}/items/{item_id}     * Text von Hand setzen (Korrektur)
  POST   /api/v1/documents/{id}/items/{item_id}/generate * ein Bild/Feld neu generieren
  POST   /api/v1/documents/{id}/export/{format}     * pdf | docx | pdfua | xlsx | csv | json | formular | formular_csv
  GET    /api/v1/documents/{id}/export/pdfua/{token}  Datei der PDF/UA-Umwandlung
  POST   /api/v1/documents/{id}/review-link         * Gast-Freigabe (unsere Pruefansicht) anlegen
  GET    /api/v1/documents/{id}/review-links          Freigaben des Dokuments
  POST   /api/v1/documents/{id}/review-links/revoke * Freigabe zurueckziehen
  DELETE /api/v1/documents/{id}                     * Dokument samt Dateien loeschen

Fehler kommen IMMER als JSON {"error": {"code", "message", "status"}, "detail": ...}; die
Codes sind stabil und englisch (unauthorized, not_found, payment_required, rate_limited, ...),
die Meldung ist der Klartext der App (Etappe 2: mehrsprachig). Guthaben-Fehler tragen
zusaetzlich preis/verfuegbar/fehlend.

Status eines Dokuments (aus projects.status uebersetzt): uploaded, extracting, ready,
generating, done, error. Status eines Items: pending, generating, done, failed; dazu
text_status offen | mit_text | dekorativ nach der Herunterladen-Regel (main._exportable_alt_text).

Sicherheit: kein Endpunkt nimmt Pfade entgegen; Item-IDs werden gegen das Dokument geprueft
(sonst 404), obwohl die App-Routen den Besitz ohnehin pruefen; Werkzeug und Sprache sind
Whitelists; Dateigroesse/-typ prueft /api/upload; Download-Token der PDF/UA prueft die App-Route
(Hex, Besitz, Pfad unter Export-/Ablage-Ordner). Das Nutzer-Dict der API traegt is_admin=0.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

log = logging.getLogger("inkludocs.api.v1")

API_VERSION = "1"


@dataclass
class Deps:
    app: Any                          # FastAPI-App: die Ziel-Routen werden per Pfad nachgeschlagen
    get_api_user: Callable            # (request) -> {"id","email","is_admin":0,"api_key_id"} oder 401
    check_api_rate_limit: Callable    # (api_key_id) -> {"minute_remaining","day_remaining"} oder 429
    log_api_usage: Callable           # (api_key_id, user_id, ..., success, error_message)
    get_db: Callable
    billing: Any
    display_alt_text: Callable        # main._display_alt_text (sichtbarer Text)
    exportable_alt_text: Callable     # main._exportable_alt_text (Herunterladen-Regel)
    image_extensions: set
    alt_text_languages: tuple
    tool_project_type: dict           # {"pdf": "pdf", "web": "url", "grafik": "images", "word": "docx", "formular": "pdfform"}
    is_valid_tool_key: Callable
    base_url: str


_d: Optional[Deps] = None
_routen: dict = {}

# Lesende Aufrufe (Status-Polling, Items, Bilddatei) laufen NICHT durch das Minuten-/Tageslimit der
# schreibenden Aufrufe, brauchen aber eine eigene, grosszuegige Bremse: jeder Aufruf prueft den
# Schluessel in der Datenbank. 300 pro Minute je Schluessel = ein Poll alle 200 ms — weit ueber der
# empfohlenen Abfrage alle fuenf Sekunden, aber eine Grenze fuer Endlosschleifen.
LESE_LIMIT_MINUTE = 300
_lese_fenster: dict = {}


def _lese_bremse(key_id: int) -> None:
    import time as _time
    jetzt = _time.time()
    liste = [t for t in _lese_fenster.get(key_id, []) if jetzt - t < 60]
    if len(liste) >= LESE_LIMIT_MINUTE:
        raise HTTPException(status_code=429, detail=f"Rate-Limit ueberschritten: max. {LESE_LIMIT_MINUTE} lesende Anfragen pro Minute.",
                            headers={"Retry-After": "60", "X-RateLimit-Limit": str(LESE_LIMIT_MINUTE), "X-RateLimit-Remaining": "0"})
    liste.append(jetzt)
    _lese_fenster[key_id] = liste

# ---------------------------------------------------------------- Fehlerformat
_CODES = {400: "bad_request", 401: "unauthorized", 402: "payment_required", 403: "forbidden",
          404: "not_found", 405: "method_not_allowed", 409: "conflict", 413: "payload_too_large",
          415: "unsupported_media_type", 422: "unprocessable", 429: "rate_limited",
          500: "internal_error", 502: "upstream_error", 503: "unavailable", 504: "timeout"}

_STATUS = {"neu": "uploaded", "extracting": "extracting", "extracted": "ready", "processing": "generating",
           "done": "done", "error": "error"}
_ITEM_STATUS = {"pending": "pending", "processing": "generating", "done": "done", "completed": "done",
                "error": "failed"}
_EXPORT_ROUTEN = {
    "pdf": "/api/projects/{project_id}/export",
    "docx": "/api/projects/{project_id}/export/docx",
    "pdfua": "/api/projects/{project_id}/export/pdfua",
    "xlsx": "/api/projects/{project_id}/export/xlsx",
    "csv": "/api/projects/{project_id}/export/csv",
    "json": "/api/projects/{project_id}/export/json",
    "formular": "/api/projects/{project_id}/export/formular",
    "formular_csv": "/api/projects/{project_id}/export/formular_csv",
}
_MAX_NAME = 120


def _fehlerantwort(status: int, message: str, code: Optional[str] = None, extra: Optional[dict] = None,
                   headers: Optional[dict] = None) -> JSONResponse:
    body = {"error": {"code": code or _CODES.get(status, "error"), "message": message, "status": status},
            "detail": message}
    if extra:
        body["error"].update(extra)
    return JSONResponse(status_code=status, content=body, headers=headers or None)


def _aus_http_exception(e: HTTPException) -> JSONResponse:
    detail = e.detail
    if isinstance(detail, dict):
        # z. B. billing.credits_fehlen_detail: {"code": "credits_fehlen", "preis", "verfuegbar", "fehlend", "text"}
        code = str(detail.get("code") or _CODES.get(e.status_code, "error"))
        message = str(detail.get("text") or detail.get("message") or detail.get("detail") or "")
        extra = {k: v for k, v in detail.items() if k not in ("code", "text", "message", "detail")}
        return _fehlerantwort(e.status_code, message, code, extra, headers=dict(e.headers or {}))
    return _fehlerantwort(e.status_code, str(detail), headers=dict(e.headers or {}))


class _Body:
    """Request-Ersatz fuer delegierte App-Routen: request.json() liefert GENAU die Felder, die die
    API geprueft hat (nie den rohen Body); alles andere (headers, cookies, url, client — z. B. fuer
    resolve_ui_language beim PDF/UA-Export) kommt unveraendert vom echten Request."""

    def __init__(self, daten: dict, original: Request):
        self._daten = daten
        self._original = original

    async def json(self):
        return self._daten

    async def body(self):
        import json as _json
        return _json.dumps(self._daten).encode("utf-8")

    def __getattr__(self, name):
        return getattr(self._original, name)


def _route(method: str, path: str) -> Callable:
    """App-Route per Methode+Pfad nachschlagen (einmal, dann aus dem Cache)."""
    key = (method, path)
    if key not in _routen:
        for r in _d.app.routes:
            if getattr(r, "path", None) == path and method in (getattr(r, "methods", None) or ()):
                _routen[key] = r.endpoint
                break
        else:
            raise RuntimeError(f"API v1: App-Route {method} {path} nicht gefunden")
    return _routen[key]


async def _json_body(request: Request) -> dict:
    try:
        body = await request.body()
        if not body:
            return {}
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Ungueltiger JSON-Koerper")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON-Objekt erwartet")
    return data


def _als_bool(wert, standard: Optional[bool] = None) -> Optional[bool]:
    if wert is None or wert == "":
        return standard
    if isinstance(wert, bool):
        return wert
    return str(wert).strip().lower() in ("1", "true", "yes", "ja", "on")


def _sprache(wert) -> Optional[str]:
    if wert is None or str(wert).strip() == "":
        return None
    s = str(wert).strip().lower()
    if s not in _d.alt_text_languages:
        raise HTTPException(status_code=400, detail="Unbekannte Sprache. Erlaubt: " + ", ".join(_d.alt_text_languages))
    return s


def _name(wert, fallback: str) -> str:
    n = (str(wert).strip() if wert is not None else "") or fallback
    n = "".join(ch for ch in n if ch.isprintable())
    return n[:_MAX_NAME] or "Dokument"


# ---------------------------------------------------------------- Aufruf-Mantel
async def _sicher(request: Request, op: str, schreibend: bool, fn: Callable, *args):
    """Auth -> (Rate-Limit bei schreibenden Aufrufen) -> fn(user, ...) -> Antwort + Kopfzeilen;
    HTTPException -> Fehler-JSON; alles andere -> 500 mit Protokoll. Schreibende Aufrufe und alle
    Fehler landen in api_usage (Verbrauchsanzeige je Schluessel); reines Lesen nicht."""
    try:
        user = _d.get_api_user(request)
    except HTTPException as e:
        return _aus_http_exception(e)
    key_id = user["api_key_id"]
    rate = None
    try:
        if schreibend:
            rate = _d.check_api_rate_limit(key_id)
        else:
            _lese_bremse(key_id)
        ergebnis = await fn(user, *args)
        if isinstance(ergebnis, Response):
            antwort = ergebnis
        else:
            status = 200
            if isinstance(ergebnis, tuple):
                ergebnis, status = ergebnis
            antwort = JSONResponse(status_code=status, content=ergebnis)
        antwort.headers["X-API-Version"] = API_VERSION
        if rate:
            antwort.headers["X-RateLimit-Remaining-Minute"] = str(rate["minute_remaining"])
            antwort.headers["X-RateLimit-Remaining-Day"] = str(rate["day_remaining"])
        # Verbrauchsprotokoll (17.09.2026): nur schreibende Aufrufe zaehlen als „Aufrufe" — lesende
        # (Status-Polling alle 5 s, Items, Bilddatei) wuerden die Statistik und api_usage aufblaehen.
        # Fehler werden immer vermerkt (auch bei GET), damit die Fehlerquote stimmt.
        if schreibend:
            _protokoll(key_id, user["id"], op, True, "")
        return antwort
    except HTTPException as e:
        _protokoll(key_id, user["id"], op, False, f"{e.status_code}: {str(e.detail)[:300]}")
        return _aus_http_exception(e)
    except Exception as e:  # noqa: BLE001
        log.exception("API v1 %s fehlgeschlagen (user=%s)", op, user["id"])
        _protokoll(key_id, user["id"], op, False, f"500: {type(e).__name__}")
        return _fehlerantwort(500, "Interner Fehler")


def _protokoll(key_id: int, user_id: int, op: str, ok: bool, fehler: str) -> None:
    try:
        _d.log_api_usage(key_id, user_id, model_used=f"v1.documents.{op}", success=ok, error_message=fehler)
    except Exception:  # noqa: BLE001 — Statistik darf die Antwort nie brechen
        log.exception("api_usage-Eintrag fehlgeschlagen")


# ---------------------------------------------------------------- Daten lesen und formen
def _projekt(conn, project_id: int, user_id: int) -> dict:
    p = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
    if not p:
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    return dict(p)


def _ist_formular(projekt: dict) -> bool:
    return (projekt.get("tool") or "") == "formular" or (projekt.get("project_type") or "") == "pdfform"


def _text_status(img: dict) -> str:
    ausgabe = _d.exportable_alt_text(img)
    if ausgabe == "dekorativ":
        return "dekorativ"
    return "mit_text" if (ausgabe or "").strip() else "offen"


def _bild_item(project_id: int, img: dict) -> dict:
    sichtbar = _d.display_alt_text(img) or ""
    return {
        "id": img["id"], "type": "image", "document_id": img.get("document_id"),
        "page": img.get("page_number"), "index": img.get("image_index"),
        "filename": img.get("display_name") or img.get("original_filename") or None,
        "width": img.get("width"), "height": img.get("height"),
        "status": _ITEM_STATUS.get(img.get("status") or "", img.get("status") or ""),
        "text_status": _text_status(img),
        "alt_text": sichtbar,
        "alt_text_ki": img.get("alt_text") or "",
        "alt_text_edited": img.get("alt_text_edited"),
        "langbeschreibung": img.get("langbeschreibung") or "",
        "bildtyp": img.get("image_type") or "",
        "konfidenz": img.get("konfidenz") or "",
        "needs_review": bool(img.get("needs_review")),
        # 18.09.2026 (Steve): Grund des letzten Fehlschlags (images.fehler_grund), sonst null.
        "error": (img.get("fehler_grund") or None) if (img.get("status") == "error" or img.get("fehler_grund")) else None,
        "language": img.get("gen_language") or None,
        "file_url": f"{_d.base_url}/api/v1/documents/{project_id}/items/{img['id']}/file",
    }


def _feld_item(f: dict) -> dict:
    return {
        "id": f["id"], "type": "field", "document_id": f.get("document_id"),
        "page": f.get("page_number"), "index": f.get("feld_index"),
        "name": f.get("feld_name"), "field_type": f.get("feld_art"), "label": f.get("beschriftung") or "",
        "group": f.get("gruppe") or "", "required": bool(f.get("pflicht")),
        "status": "done" if (f.get("quickinfo") or "").strip() else "pending",
        "text_status": "mit_text" if (f.get("quickinfo") or "").strip() else "offen",
        "quickinfo": f.get("quickinfo") or "",
        "quickinfo_ki": f.get("quickinfo_ki") or "",
        "quickinfo_original": f.get("quickinfo_original") or "",
        "source": f.get("quelle") or "",
        "konfidenz": f.get("sicherheit") or "",
    }


async def _dokument_status(user: dict, project_id: int) -> dict:
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
        if _ist_formular(p):
            rows = [dict(r) for r in conn.execute(
                "SELECT quickinfo, quelle FROM formularfelder WHERE project_id = ?", (project_id,)).fetchall()]
            counts = {"items": len(rows),
                      "with_text": sum(1 for r in rows if (r.get("quickinfo") or "").strip()),
                      "without_text": sum(1 for r in rows if not (r.get("quickinfo") or "").strip()),
                      "decorative": 0, "done": None, "pending": None, "failed": None}
        else:
            rows = [dict(r) for r in conn.execute("SELECT * FROM images WHERE project_id = ?", (project_id,)).fetchall()]
            st = [_text_status(r) for r in rows]
            counts = {"items": len(rows),
                      "with_text": st.count("mit_text"), "without_text": st.count("offen"), "decorative": st.count("dekorativ"),
                      "done": sum(1 for r in rows if r.get("status") in ("done", "completed")),
                      "pending": sum(1 for r in rows if r.get("status") == "pending"),
                      "failed": sum(1 for r in rows if r.get("status") == "error")}
        docs = [dict(d) for d in conn.execute(
            "SELECT id, doc_index, original_filename, display_name, extraction_method, total_images, created_at, source_url, getaggt "
            "FROM documents WHERE project_id = ? ORDER BY doc_index", (project_id,)).fetchall()]
        links = conn.execute("SELECT COUNT(*) FROM shares WHERE project_id = ? AND status IN ('active', 'completed')",
                             (project_id,)).fetchone()[0]
    finally:
        conn.close()
    hinweis = None
    if p.get("lauf_hinweis"):
        try:
            import json as _json
            hinweis = _json.loads(p["lauf_hinweis"])
        except Exception:  # noqa: BLE001
            hinweis = None
    return {
        "id": p["id"], "tool": p.get("tool"), "kind": p.get("project_type"),
        "name": p.get("name") or p.get("filename") or "",
        "status": _STATUS.get(p.get("status") or "", p.get("status") or ""),
        "language": p.get("alt_language") or "de",
        "use_context": bool(p.get("use_context", 1)) if p.get("use_context") is not None else True,
        "source_url": p.get("source_url") or None,
        "created_at": p.get("created_at"), "updated_at": p.get("updated_at"),
        "counts": counts,
        "progress": {"processed": p.get("processed_images") or 0, "total": p.get("total_images") or 0},
        "run_note": hinweis,
        "documents": [{"id": d["id"], "index": d["doc_index"], "filename": d.get("original_filename"),
                       "display_name": d.get("display_name"), "items": d.get("total_images"),
                       "tagged": (None if d.get("getaggt") is None else bool(d.get("getaggt"))),
                       "source_url": d.get("source_url"), "created_at": d.get("created_at")} for d in docs],
        "review_links": links,
        "links": {"self": f"{_d.base_url}/api/v1/documents/{p['id']}",
                  "items": f"{_d.base_url}/api/v1/documents/{p['id']}/items"},
    }


def _item_gehoert_zum_dokument(conn, projekt: dict, item_id: int) -> None:
    tabelle = "formularfelder" if _ist_formular(projekt) else "images"
    r = conn.execute(f"SELECT 1 FROM {tabelle} WHERE id = ? AND project_id = ?", (item_id, projekt["id"])).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Item nicht gefunden")


# ---------------------------------------------------------------- Endpunkt-Kerne
def _werkzeug_fuer(ext: str, gewuenscht: Optional[str]) -> str:
    ext = (ext or "").lower()
    if gewuenscht:
        g = gewuenscht.strip().lower()
        if g not in ("pdf", "word", "grafik", "formular"):
            raise HTTPException(status_code=400, detail="Unbekanntes Werkzeug. Erlaubt: pdf, word, grafik, formular (Dateien) oder web (JSON mit url)")
        if g in ("pdf", "formular") and ext != ".pdf":
            raise HTTPException(status_code=400, detail="Fuer die Werkzeuge pdf und formular wird eine PDF-Datei erwartet")
        if g == "word" and ext != ".docx":
            raise HTTPException(status_code=400, detail="Fuer das Werkzeug word wird eine .docx-Datei erwartet")
        if g == "grafik" and ext not in _d.image_extensions:
            raise HTTPException(status_code=400, detail="Fuer das Werkzeug grafik wird eine Bilddatei erwartet")
        return g
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "word"
    if ext in _d.image_extensions:
        return "grafik"
    raise HTTPException(status_code=400, detail="Nur PDF-, Word- und Bilddateien erlaubt (PDF, DOCX, JPG, PNG, GIF, SVG, WebP, HEIC, BMP, TIFF)")


def _projekt_anlegen(user_id: int, name: str, tool: str, language: Optional[str], use_context: Optional[bool],
                     api_key_id: Optional[int] = None) -> int:
    if not _d.is_valid_tool_key(tool):
        raise HTTPException(status_code=400, detail="Dieses Werkzeug ist auf dieser Instanz nicht verfuegbar")
    project_type = _d.tool_project_type.get(tool)
    if not project_type:
        raise HTTPException(status_code=400, detail="Unbekanntes Werkzeug")
    conn = _d.get_db()
    try:
        cur = conn.execute(
            "INSERT INTO projects (user_id, name, filename, original_path, status, project_type, tool, api_key_id) VALUES (?, ?, ?, '', 'neu', ?, ?, ?)",
            (user_id, name, name, project_type, tool, api_key_id))
        pid = cur.lastrowid
        if language:
            conn.execute("UPDATE projects SET alt_language = ? WHERE id = ?", (language, pid))
        if use_context is not None:
            conn.execute("UPDATE projects SET use_context = ? WHERE id = ?", (1 if use_context else 0, pid))
        conn.commit()
    finally:
        conn.close()
    return pid


async def _projekt_verwerfen(user: dict, project_id: int) -> None:
    """Aufraeumen, wenn der Upload nach dem Anlegen scheitert (kein leeres Projekt zuruecklassen)."""
    try:
        await _route("DELETE", "/api/projects/{project_id}")(project_id=project_id, user=user)
    except Exception:  # noqa: BLE001
        log.exception("API v1: Projekt %s nach Upload-Fehler nicht aufgeraeumt", project_id)


async def _documents_create(user: dict, request: Request):
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        data = await _json_body(request)
        url = (data.get("url") or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="Feld 'url' fehlt (oder eine Datei als multipart/form-data hochladen)")
        language = _sprache(data.get("language"))
        use_context = _als_bool(data.get("use_context"))
        _prompt_daten = {k: data.get(k) for k in ("prompt_id", "prompt") if data.get(k) not in (None, "")}
        # Web-Werkzeug: die App-Route legt das Projekt selbst an und laedt die Bilder synchron.
        erg = await _route("POST", "/api/scan-url")(request=_Body({"url": url}, request), user=user)
        pid = int(erg["project_id"])
        conn = _d.get_db()
        try:
            # Verbrauch je Schluessel (17.09.2026): das Web-Projekt dem Schluessel zuordnen.
            conn.execute("UPDATE projects SET api_key_id = ? WHERE id = ? AND user_id = ?", (user.get("api_key_id"), pid, user["id"]))
            if data.get("name"):
                conn.execute("UPDATE projects SET name = ? WHERE id = ? AND user_id = ?", (_name(data.get("name"), url), pid, user["id"]))
            if language:
                conn.execute("UPDATE projects SET alt_language = ? WHERE id = ? AND user_id = ?", (language, pid, user["id"]))
            if use_context is not None:
                conn.execute("UPDATE projects SET use_context = ? WHERE id = ? AND user_id = ?", (1 if use_context else 0, pid, user["id"]))
            conn.commit()
            if _prompt_daten:
                _lauf_einstellungen_anwenden(conn, user["id"], pid, _prompt_daten)
        finally:
            conn.close()
        return await _dokument_status(user, pid), 201

    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=415, detail="Erwartet multipart/form-data (Datei) oder application/json (url)")
    form = await request.form()
    datei = form.get("file") or form.get("document")
    if datei is None or not hasattr(datei, "filename"):
        raise HTTPException(status_code=400, detail="Bitte eine Datei als 'file' hochladen")
    filename = datei.filename or "dokument"
    ext = os.path.splitext(filename)[1].lower()
    tool = _werkzeug_fuer(ext, form.get("tool"))
    language = _sprache(form.get("language"))
    use_context = _als_bool(form.get("use_context"))
    name = _name(form.get("name"), os.path.basename(filename))
    pid = _projekt_anlegen(user["id"], name, tool, language, use_context, api_key_id=user.get("api_key_id"))
    _prompt_daten = {k: form.get(k) for k in ("prompt_id", "prompt") if form.get(k) not in (None, "")}
    if _prompt_daten:
        conn = _d.get_db()
        try:
            _lauf_einstellungen_anwenden(conn, user["id"], pid, _prompt_daten)
        finally:
            conn.close()
    try:
        await _route("POST", "/api/upload")(file=datei, project_id=pid, user=user)
    except HTTPException:
        await _projekt_verwerfen(user, pid)
        raise
    except Exception:
        await _projekt_verwerfen(user, pid)
        raise
    status = await _dokument_status(user, pid)
    return status, (201 if status["status"] in ("ready", "done") else 202)


async def _documents_list(user: dict, limit: int, offset: int):
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    conn = _d.get_db()
    try:
        rows = conn.execute(
            "SELECT id, name, filename, status, tool, project_type, total_images, processed_images, created_at, updated_at "
            "FROM projects WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?", (user["id"], limit, offset)).fetchall()
        gesamt = conn.execute("SELECT COUNT(*) FROM projects WHERE user_id = ?", (user["id"],)).fetchone()[0]
    finally:
        conn.close()
    return {"total": gesamt, "limit": limit, "offset": offset,
            "documents": [{"id": r["id"], "name": r["name"] or r["filename"] or "", "tool": r["tool"], "kind": r["project_type"],
                           "status": _STATUS.get(r["status"] or "", r["status"] or ""),
                           "progress": {"processed": r["processed_images"] or 0, "total": r["total_images"] or 0},
                           "created_at": r["created_at"], "updated_at": r["updated_at"],
                           "links": {"self": f"{_d.base_url}/api/v1/documents/{r['id']}"}} for r in rows]}


async def _documents_get(user: dict, project_id: int):
    return await _dokument_status(user, project_id)


PROMPT_TEXT_MAX = 4000   # wie main.PROMPT_TEXT_MAX


def _lauf_einstellungen_anwenden(conn, user_id: int, project_id: int, data: dict) -> dict:
    """Sprache, Kontext und eigener Prompt je Lauf (18.09.2026, Steve + Michael): dieselben
    Projekt-Einstellungen wie in der App, hier beim Start mitgegeben. Rueckgabe = was gesetzt wurde.
    prompt_id: gespeicherter Prompt des Kontos („Meine Prompts“). prompt: freier Text — wird als
    gespeicherter Prompt in der Kategorie „API“ angelegt (gleicher Text = gleiche Zeile), damit
    Kontoinhaber ihn in der App sehen und der Lauf ihn wie jeden eigenen Prompt nutzt."""
    gesetzt = {}
    language = _sprache(data.get("language"))
    use_context = _als_bool(data.get("use_context"))
    if language:
        conn.execute("UPDATE projects SET alt_language = ? WHERE id = ? AND user_id = ?", (language, project_id, user_id))
        gesetzt["language"] = language
    if use_context is not None:
        conn.execute("UPDATE projects SET use_context = ? WHERE id = ? AND user_id = ?", (1 if use_context else 0, project_id, user_id))
        gesetzt["use_context"] = use_context
    prompt_id = data.get("prompt_id")
    prompt = data.get("prompt")
    if prompt_id not in (None, ""):
        try:
            prompt_id = int(prompt_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="prompt_id ungültig")
        if prompt_id == 0:
            conn.execute("UPDATE projects SET prompt_id = NULL WHERE id = ? AND user_id = ?", (project_id, user_id))
            gesetzt["prompt_id"] = None
        else:
            if not conn.execute("SELECT 1 FROM user_prompts WHERE id = ? AND user_id = ?", (prompt_id, user_id)).fetchone():
                raise HTTPException(status_code=404, detail="prompt_id nicht gefunden (nur eigene gespeicherte Prompts)")
            conn.execute("UPDATE projects SET prompt_id = ? WHERE id = ? AND user_id = ?", (prompt_id, project_id, user_id))
            gesetzt["prompt_id"] = prompt_id
    elif prompt is not None:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(prompt)).strip()
        if len(text) > PROMPT_TEXT_MAX:
            raise HTTPException(status_code=400, detail=f"prompt darf höchstens {PROMPT_TEXT_MAX} Zeichen lang sein")
        if not text:
            conn.execute("UPDATE projects SET prompt_id = NULL WHERE id = ? AND user_id = ?", (project_id, user_id))
            gesetzt["prompt_id"] = None
        else:
            row = conn.execute("SELECT id FROM user_prompts WHERE user_id = ? AND category = 'API' AND prompt_text = ?",
                               (user_id, text)).fetchone()
            if row:
                pid = row["id"]
            else:
                name = ("API: " + re.sub(r"\s+", " ", text)[:60]).strip()
                cur = conn.execute("INSERT INTO user_prompts (user_id, name, description, category, prompt_text) VALUES (?, ?, ?, 'API', ?)",
                                   (user_id, name, "Über die API mitgegebener Prompt", text))
                pid = cur.lastrowid
            conn.execute("UPDATE projects SET prompt_id = ? WHERE id = ? AND user_id = ?", (pid, project_id, user_id))
            gesetzt["prompt_id"] = pid
    conn.commit()
    return gesetzt


async def _documents_generate(user: dict, request: Request, project_id: int):
    """Sammellauf. scope (18.09.2026, Steve): "open" (Vorgabe) = nur Eintraege ohne fertigen Text —
    nie generierte und fehlgeschlagene; "all" = alle Eintraege neu (kostet alle erneut). Dazu je Lauf
    language, use_context, prompt_id oder prompt. Formulare kennen nur "all" (alle benannten Felder)."""
    data = await _json_body(request)
    scope = str(data.get("scope") or "open").strip().lower()
    if scope not in ("open", "all"):
        raise HTTPException(status_code=400, detail="scope: open oder all")
    body = {}
    if data.get("document_id") is not None:
        body["document_id"] = data.get("document_id")
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
        gesetzt = _lauf_einstellungen_anwenden(conn, user["id"], project_id, data)
    finally:
        conn.close()
    if _ist_formular(p):
        erg = await _route("POST", "/api/projects/{project_id}/quickinfos/generieren")(project_id=project_id, request=_Body(body, request), user=user)
        anzahl = erg.get("offen", 0)
        scope = "all"
    else:
        if scope == "open":
            body["nur_offen"] = True
        erg = await _route("POST", "/api/projects/{project_id}/generate")(project_id=project_id, request=_Body(body, request), user=user)
        anzahl = erg.get("anzahl", 0)
    status = await _dokument_status(user, project_id)
    status["started"] = bool(erg.get("gestartet"))
    status["queued_items"] = int(anzahl or 0)
    status["scope"] = scope
    if gesetzt:
        status["settings"] = gesetzt
    if not status["started"] and scope == "open":
        status["hint"] = "Keine offenen Einträge. Mit scope=all werden alle Einträge neu beschrieben (kostet erneut Credits)."
    return status, (202 if status["started"] else 200)


async def _documents_cancel(user: dict, project_id: int):
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
    finally:
        conn.close()
    pfad = "/api/projects/{project_id}/quickinfos/abbrechen" if _ist_formular(p) else "/api/projects/{project_id}/generate/abbrechen"
    erg = await _route("POST", pfad)(project_id=project_id, user=user)
    return {"id": project_id, "cancel_requested": bool(erg.get("angefordert")), "reason": erg.get("grund")}


async def _items_list(user: dict, project_id: int, request: Optional[Request] = None):
    """?status=pending|generating|done|failed und ?text_status=offen|mit_text|dekorativ (18.09.2026)
    filtern die Liste; `count` ist die Zahl der zurueckgegebenen, `total` die Zahl aller Eintraege."""
    filt_status = filt_text = ""
    if request is not None:
        filt_status = (request.query_params.get("status") or "").strip().lower()
        filt_text = (request.query_params.get("text_status") or "").strip().lower()
        if filt_status and filt_status not in ("pending", "generating", "done", "failed"):
            raise HTTPException(status_code=400, detail="status: pending, generating, done oder failed")
        if filt_text and filt_text not in ("offen", "mit_text", "dekorativ"):
            raise HTTPException(status_code=400, detail="text_status: offen, mit_text oder dekorativ")
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
    finally:
        conn.close()
    if _ist_formular(p):
        erg = await _route("GET", "/api/projects/{project_id}/felder")(project_id=project_id, user=user)
        items = [_feld_item(f) for f in erg.get("felder", [])]
    else:
        erg = await _route("GET", "/api/projects/{project_id}")(project_id=project_id, user=user)
        items = [_bild_item(project_id, img) for img in erg.get("images", [])]
    total = len(items)
    if filt_status:
        items = [i for i in items if i.get("status") == filt_status]
    if filt_text:
        items = [i for i in items if i.get("text_status") == filt_text]
    return {"id": project_id, "kind": p.get("project_type"), "count": len(items), "total": total, "items": items}


async def _item_file(user: dict, project_id: int, item_id: int):
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
        if _ist_formular(p):
            raise HTTPException(status_code=404, detail="Formularfelder haben keine Bilddatei")
        _item_gehoert_zum_dokument(conn, p, item_id)
    finally:
        conn.close()
    return await _route("GET", "/api/images/{image_id}/file")(image_id=item_id, user=user)


async def _item_patch(user: dict, request: Request, project_id: int, item_id: int):
    data = await _json_body(request)
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
        _item_gehoert_zum_dokument(conn, p, item_id)
        if _ist_formular(p):
            if "quickinfo" not in data:
                raise HTTPException(status_code=400, detail="Feld 'quickinfo' fehlt")
            if not isinstance(data["quickinfo"], str):
                raise HTTPException(status_code=400, detail="'quickinfo' muss ein String sein")
            body = {"quickinfo": data["quickinfo"]}
        else:
            if "alt_text" not in data and "langbeschreibung" not in data:
                raise HTTPException(status_code=400, detail="Mindestens 'alt_text' oder 'langbeschreibung' muss angegeben werden")
            for k in ("alt_text", "langbeschreibung"):
                if k in data and not isinstance(data[k], str):
                    raise HTTPException(status_code=400, detail=f"'{k}' muss ein String sein")
            body = {}
            if "alt_text" in data:
                body["alt_text"] = data["alt_text"]
            else:
                # Die App-Route setzt alt_text_edited immer — ohne diesen Schutz wuerde ein PATCH nur
                # der Langbeschreibung den sichtbaren Alt-Text auf „bewusst geleert" stellen.
                img = dict(conn.execute("SELECT * FROM images WHERE id = ?", (item_id,)).fetchone())
                body["alt_text"] = _d.display_alt_text(img) or ""
            if "langbeschreibung" in data:
                body["langbeschreibung"] = data["langbeschreibung"]
    finally:
        conn.close()
    if _ist_formular(p):
        await _route("PATCH", "/api/felder/{feld_id}")(feld_id=item_id, request=_Body(body, request), user=user)
    else:
        await _route("POST", "/api/images/{image_id}/alt-text")(image_id=item_id, request=_Body(body, request), user=user)
    erg = await _items_list(user, project_id)
    for it in erg["items"]:
        if it["id"] == item_id:
            return it
    raise HTTPException(status_code=404, detail="Item nicht gefunden")


async def _item_generate(user: dict, request: Request, project_id: int, item_id: int):
    data = await _json_body(request)
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
        _item_gehoert_zum_dokument(conn, p, item_id)
    finally:
        conn.close()
    if data.get("language") is not None or data.get("prompt_id") not in (None, "") or data.get("prompt") is not None or data.get("use_context") is not None:
        conn = _d.get_db()
        try:
            _lauf_einstellungen_anwenden(conn, user["id"], project_id, data)
        finally:
            conn.close()
    if _ist_formular(p):
        await _route("POST", "/api/felder/{feld_id}/generieren")(feld_id=item_id, user=user)
    else:
        body = {}
        if data.get("image_type"):
            body["image_type"] = str(data["image_type"])
        await _route("POST", "/api/projects/{project_id}/regenerate/{image_id}")(
            project_id=project_id, image_id=item_id, request=_Body(body, request), user=user)
    erg = await _items_list(user, project_id)
    for it in erg["items"]:
        if it["id"] == item_id:
            return it
    raise HTTPException(status_code=404, detail="Item nicht gefunden")


async def _documents_export(user: dict, request: Request, project_id: int, fmt: str):
    fmt = (fmt or "").lower()
    if fmt not in _EXPORT_ROUTEN:
        raise HTTPException(status_code=400, detail="Unbekanntes Format. Erlaubt: " + ", ".join(_EXPORT_ROUTEN))
    data = await _json_body(request)
    body = {}
    if data.get("document_id") is not None:
        body["document_id"] = data.get("document_id")
    if data.get("filename"):
        body["filename"] = str(data["filename"])[:120]
    conn = _d.get_db()
    try:
        p = _projekt(conn, project_id, user["id"])
    finally:
        conn.close()
    if fmt in ("formular", "formular_csv") and not _ist_formular(p):
        raise HTTPException(status_code=400, detail="Dieses Format gibt es nur fuer Formular-Dokumente")
    if fmt not in ("formular", "formular_csv") and _ist_formular(p):
        raise HTTPException(status_code=400, detail="Formular-Dokumente exportieren als formular (PDF) oder formular_csv")
    route = _route("POST", _EXPORT_ROUTEN[fmt])
    erg = await route(project_id=project_id, request=_Body(body, request), user=user)
    if fmt == "pdfua":
        # Die App-Route antwortet mit JSON (Bericht) + Token; der Download laeuft ueber unseren v1-Pfad.
        inhalt = _json_von(erg)
        token = inhalt.get("token")
        inhalt["download_url"] = f"{_d.base_url}/api/v1/documents/{project_id}/export/pdfua/{token}"
        inhalt.pop("ausgaben_anzahl", None)
        return inhalt
    return erg


def _json_von(antwort) -> dict:
    """JSONResponse einer App-Route wieder als dict (fuer Nachbearbeitung)."""
    if isinstance(antwort, dict):
        return dict(antwort)
    import json as _json
    return _json.loads(bytes(antwort.body).decode("utf-8"))


async def _export_pdfua_download(user: dict, project_id: int, token: str):
    return await _route("GET", "/api/projects/{project_id}/export/pdfua/{token}")(project_id=project_id, token=token, user=user)


async def _review_link_create(user: dict, request: Request, project_id: int):
    data = await _json_body(request)
    body = {"guest_email": (data.get("guest_email") or "").strip(),
            "guest_name": str(data.get("guest_name") or "")[:120],
            "message": str(data.get("message") or "")[:2000],
            "notify": _als_bool(data.get("notify"), True),
            "role": str(data.get("role") or "kunde").strip().lower()}
    erg = await _route("POST", "/api/projects/{project_id}/share")(project_id=project_id, request=_Body(body, request), user=user)
    return {"id": project_id, "token": erg["token"], "url": erg["url"], "guest_email": erg["guest_email"],
            "role": erg["role"], "email_sent": bool(erg.get("sent"))}, 201


async def _review_links_list(user: dict, project_id: int):
    erg = await _route("GET", "/api/projects/{project_id}/shares")(project_id=project_id, user=user)
    return {"id": project_id, "review_links": erg.get("shares", [])}


async def _review_link_revoke(user: dict, request: Request, project_id: int):
    data = await _json_body(request)
    token = str(data.get("token") or "")
    if not token:
        raise HTTPException(status_code=400, detail="Feld 'token' fehlt")
    conn = _d.get_db()
    try:
        _projekt(conn, project_id, user["id"])
    finally:
        conn.close()
    await _route("POST", "/api/projects/{project_id}/shares/revoke")(project_id=project_id, request=_Body({"token": token}, request), user=user)
    return {"id": project_id, "revoked": True}


async def _documents_delete(user: dict, project_id: int):
    await _route("DELETE", "/api/projects/{project_id}")(project_id=project_id, user=user)
    return {"id": project_id, "deleted": True}


# ---------------------------------------------------------------- Router
def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    router = APIRouter()

    @router.post("/api/v1/documents")
    async def v1_documents_create(request: Request):
        return await _sicher(request, "create", True, _documents_create, request)

    @router.get("/api/v1/documents")
    async def v1_documents_list(request: Request, limit: int = 50, offset: int = 0):
        return await _sicher(request, "list", False, _documents_list, limit, offset)

    @router.get("/api/v1/documents/{project_id}")
    async def v1_documents_get(project_id: int, request: Request):
        return await _sicher(request, "get", False, _documents_get, project_id)

    @router.post("/api/v1/documents/{project_id}/generate")
    async def v1_documents_generate(project_id: int, request: Request):
        return await _sicher(request, "generate", True, _documents_generate, request, project_id)

    @router.post("/api/v1/documents/{project_id}/generate/cancel")
    async def v1_documents_cancel(project_id: int, request: Request):
        return await _sicher(request, "cancel", True, _documents_cancel, project_id)

    @router.get("/api/v1/documents/{project_id}/items")
    async def v1_items_list(project_id: int, request: Request):
        return await _sicher(request, "items", False, _items_list, project_id, request)

    @router.get("/api/v1/documents/{project_id}/items/{item_id}/file")
    async def v1_item_file(project_id: int, item_id: int, request: Request):
        return await _sicher(request, "item_file", False, _item_file, project_id, item_id)

    @router.patch("/api/v1/documents/{project_id}/items/{item_id}")
    async def v1_item_patch(project_id: int, item_id: int, request: Request):
        return await _sicher(request, "item_patch", True, _item_patch, request, project_id, item_id)

    @router.post("/api/v1/documents/{project_id}/items/{item_id}/generate")
    async def v1_item_generate(project_id: int, item_id: int, request: Request):
        return await _sicher(request, "item_generate", True, _item_generate, request, project_id, item_id)

    @router.post("/api/v1/documents/{project_id}/export/{fmt}")
    async def v1_documents_export(project_id: int, fmt: str, request: Request):
        return await _sicher(request, "export", True, _documents_export, request, project_id, fmt)

    @router.get("/api/v1/documents/{project_id}/export/pdfua/{token}")
    async def v1_export_pdfua_download(project_id: int, token: str, request: Request):
        return await _sicher(request, "export_download", False, _export_pdfua_download, project_id, token)

    @router.post("/api/v1/documents/{project_id}/review-link")
    async def v1_review_link_create(project_id: int, request: Request):
        return await _sicher(request, "review_link", True, _review_link_create, request, project_id)

    @router.get("/api/v1/documents/{project_id}/review-links")
    async def v1_review_links_list(project_id: int, request: Request):
        return await _sicher(request, "review_links", False, _review_links_list, project_id)

    @router.post("/api/v1/documents/{project_id}/review-links/revoke")
    async def v1_review_link_revoke(project_id: int, request: Request):
        return await _sicher(request, "review_link_revoke", True, _review_link_revoke, request, project_id)

    @router.delete("/api/v1/documents/{project_id}")
    async def v1_documents_delete(project_id: int, request: Request):
        return await _sicher(request, "delete", True, _documents_delete, project_id)

    return router
