"""Project + Image Tools: was hat das Projekt, wie sieht ein Bild aus.

Eigene DB-Queries mit allen relevanten Spalten (statt nur was
get_project_context aus adapters/inkludocs.py rausgibt).
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from ..adapters.inkludocs import get_project_context

log = logging.getLogger(__name__)

_DB_PATH = "/app/data/inkludocs.db"


def _check_project_access(project_id: int, user_id: int) -> bool:
    """Prüft ob das Projekt dem User gehört."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        row = conn.execute(
            "SELECT 1 FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _project_image_ids(project_id: int) -> list[int]:
    """Liefert die echten image_ids im Projekt (sortiert nach UI-Reihenfolge)."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id FROM images WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _wrong_image_id_hint(image_id: int, project_id: int) -> str:
    """Standardisierte Fehlermeldung wenn die image_id nicht passt — der Bot
    bekommt die echten ids zurück, damit er den Tool-Call selbstkorrigieren
    kann statt zu halluzinieren das Bild würde nicht laden.
    """
    real_ids = _project_image_ids(project_id)
    return (
        f"image_id={image_id} existiert nicht im Projekt {project_id}. "
        f"Die echten image_ids in diesem Projekt sind: {real_ids}. "
        f"Vermutlich hast du eine UI-Position (z.B. Bild 1 = erstes Bild "
        f"in der Liste) statt der echten image_id verwendet. Die echten "
        f"image_ids sind nicht-konsekutive Zahlen aus der DB, nicht 1, 2 oder 3. "
        f"Mappe UI-Position auf die echte image_id und rufe das Tool erneut."
    )


def list_project_images(project_id: int, user_id: int) -> dict[str, Any]:
    """Gibt alle Bilder im Projekt mit Metadaten zurück.

    Multi-Datei (08.06.2026): liefert pro Bild zusaetzlich doc_index,
    doc_position und ui_label, damit Claude die im Frontend pro Dokument
    nummerierten Bilder eindeutig referenzieren kann ("Dokument 2, Bild 1"
    statt "Bild 5"). Plus separate documents-Sektion mit der bekannten
    Reihenfolge (doc_index, original_filename, display_name, Bild-Zahl).
    """
    if not _check_project_access(project_id, user_id):
        return {"ok": False, "error": "Projekt nicht gefunden oder kein Zugriff."}

    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        proj_row = conn.execute(
            "SELECT id, filename, status, total_images, processed_images, extraction_method "
            "FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if not proj_row:
            return {"ok": False, "error": "Projekt nicht gefunden."}

        doc_rows = conn.execute(
            "SELECT id, doc_index, original_filename, display_name, total_images "
            "FROM documents WHERE project_id = ? ORDER BY doc_index",
            (project_id,),
        ).fetchall()
        img_rows = conn.execute(
            "SELECT i.id, i.page_number, i.image_index, i.image_type, i.alt_text, "
            "i.alt_text_edited, i.langbeschreibung, i.context_text, i.width, i.height, "
            "i.status, i.konfidenz, i.needs_review, i.pipeline_steps, "
            "i.document_id, d.doc_index AS doc_index "
            "FROM images i "
            "LEFT JOIN documents d ON d.id = i.document_id "
            "WHERE i.project_id = ? "
            "ORDER BY COALESCE(d.doc_index, 0), i.page_number, i.image_index, i.id ASC",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()

    documents = [dict(d) for d in doc_rows]
    multi_doc = len(documents) > 1

    images = []
    per_doc_counter: dict[int, int] = {}
    for img in img_rows:
        d = dict(img)
        alt = d.get("alt_text_edited") or d.get("alt_text") or ""
        di = d.get("doc_index") or 0
        per_doc_counter[di] = per_doc_counter.get(di, 0) + 1
        pos = per_doc_counter[di]
        ui_label = (f"Dokument {di}, Bild {pos}" if multi_doc and di
                    else f"Bild {pos}")
        images.append({
            "image_id": d["id"],
            "doc_index": di if di else None,
            "doc_position": pos,
            "ui_label": ui_label,
            "page": d.get("page_number"),
            "index_on_page": d.get("image_index"),
            "image_type": d.get("image_type") or "",
            "konfidenz": d.get("konfidenz") or "",
            "needs_review": bool(d.get("needs_review")),
            "alt_text": alt[:300],
            "alt_text_length": len(alt),
            "langbeschreibung_length": len(d.get("langbeschreibung") or ""),
            "width": d.get("width"),
            "height": d.get("height"),
            "pipeline_steps": d.get("pipeline_steps") or "",
            "context_text": (d.get("context_text") or "")[:200],
            "status": d.get("status") or "",
        })

    proj = dict(proj_row)
    return {
        "ok": True,
        "result": {
            "project": {
                "id": proj["id"],
                "filename": proj.get("filename"),
                "total_images": proj.get("total_images"),
                "processed_images": proj.get("processed_images"),
                "extraction_method": proj.get("extraction_method"),
                "status": proj.get("status"),
                "multi_doc": multi_doc,
            },
            "documents": [
                {
                    "doc_index": d["doc_index"],
                    "original_filename": d["original_filename"],
                    "display_name": d.get("display_name"),
                    "total_images": d.get("total_images"),
                }
                for d in documents
            ],
            "image_count": len(images),
            "images": images,
        },
    }


def get_image_metadata(image_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Volldaten zu einem einzelnen Bild."""
    if not _check_project_access(project_id, user_id):
        return {"ok": False, "error": "Projekt nicht gefunden oder kein Zugriff."}

    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id, project_id, page_number, image_index, image_path, image_type, "
            "alt_text, alt_text_edited, langbeschreibung, context_text, width, height, "
            "status, konfidenz, needs_review, pipeline_steps, validation_result, feedback "
            "FROM images WHERE id = ? AND project_id = ?",
            (image_id, project_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return {"ok": False, "error": _wrong_image_id_hint(image_id, project_id)}

    d = dict(row)
    return {
        "ok": True,
        "result": {
            "image_id": d["id"],
            "page": d.get("page_number"),
            "image_type": d.get("image_type") or "",
            "konfidenz": d.get("konfidenz") or "",
            "needs_review": bool(d.get("needs_review")),
            "alt_text": d.get("alt_text") or "",
            "alt_text_edited": d.get("alt_text_edited") or "",
            "langbeschreibung": d.get("langbeschreibung") or "",
            "context_text": d.get("context_text") or "",
            "width": d.get("width"),
            "height": d.get("height"),
            "pipeline_steps": d.get("pipeline_steps") or "",
            "validation_result": d.get("validation_result") or "",
            "feedback": d.get("feedback") or "",
            "image_path": d.get("image_path"),
        },
    }


def view_image(image_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Lädt die Bilddatei + gibt Bytes zurück, damit chat_engine das Bild
    als image-block an die nächste user-message anhängen kann.
    """
    if not _check_project_access(project_id, user_id):
        return {"ok": False, "error": "Projekt nicht gefunden oder kein Zugriff."}

    conn = sqlite3.connect(_DB_PATH)
    try:
        row = conn.execute(
            "SELECT image_path, width, height FROM images WHERE id = ? AND project_id = ?",
            (image_id, project_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return {"ok": False, "error": _wrong_image_id_hint(image_id, project_id)}

    path = row[0] or ""
    if not os.path.isabs(path):
        path = "/app/data/" + path
    if not os.path.exists(path):
        return {"ok": False, "error": f"Bilddatei nicht auf Disk: {path}"}

    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        return {"ok": False, "error": f"Datei nicht lesbar: {e}"}

    return {
        "ok": True,
        "result": {
            "image_id": image_id,
            "size_bytes": len(data),
            "width": row[1],
            "height": row[2],
            "info": "Bild wurde geladen und ist im naechsten Turn fuer dich sichtbar.",
        },
        "image_bytes": data,
    }
