"""Alt-Text Tools: generieren + speichern + rückgängig machen.

update_alt_text speichert direkt via SQL, weil der bestehende Adapter
keine Langbeschreibung unterstützt — hier wollen wir beides in einem
Tool-Aufruf abdecken.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

from ..adapters.inkludocs import run_pipeline_for_image as _run_pipeline

log = logging.getLogger(__name__)

_DB_PATH = "/app/data/inkludocs.db"


def _check_project_access(project_id: int, user_id: int) -> bool:
    conn = sqlite3.connect(_DB_PATH)
    try:
        row = conn.execute(
            "SELECT 1 FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def generate_alt_text(image_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Ruft die v4-Pipeline für ein Bild auf (force_regenerate=True via Pipeline-Adapter).

    Returns:
        {"ok": True, "result": {alt_text, langbeschreibung, image_type, pipeline_steps, konfidenz, needs_review}}
        oder {"ok": False, "error": "..."}.
    """
    try:
        result = _run_pipeline(image_id, project_id, user_id)
    except Exception as e:
        log.exception("generate_alt_text Pipeline-Fehler")
        return {"ok": False, "error": f"Pipeline-Fehler: {e}"}

    if result is None:
        return {"ok": False, "error": f"Bild {image_id} nicht in Projekt oder Pipeline lieferte nichts."}

    return {
        "ok": True,
        "result": {
            "image_id": image_id,
            "alt_text": result.get("alt_text") or "",
            "langbeschreibung": result.get("langbeschreibung") or "",
            "image_type": result.get("image_type") or result.get("bildtyp") or "",
            "pipeline_steps": result.get("pipeline_steps") or "",
            "konfidenz": result.get("konfidenz") or "",
            "needs_review": bool(result.get("needs_review")),
        },
    }


def update_alt_text(
    image_id: int,
    project_id: int,
    user_id: int,
    new_alt_text: str,
    new_langbeschreibung: Optional[str] = None,
) -> dict[str, Any]:
    """Speichert Alt-Text (Pflicht) + optional Langbeschreibung in der DB.

    - Alt-Text: in Spalte `alt_text_edited` (Original-Pipeline-Ausgabe bleibt
      in `alt_text` erhalten).
    - Langbeschreibung: überschreibt direkt `langbeschreibung` (kein
      separates _edited-Feld). Wenn nicht gesetzt, bleibt die bestehende
      Langbeschreibung unangetastet.

    Validierungs-Mindeststandards Alt-Text:
    - Mindestens 5 Zeichen
    - Maximal 500 Zeichen (BITV-Empfehlung kompakter Alt-Text)
    - Kein „Bild von..." / „Foto von..." (zu generisch)
    """
    if not _check_project_access(project_id, user_id):
        return {"ok": False, "error": "Projekt nicht gefunden oder kein Zugriff."}

    text = (new_alt_text or "").strip()
    if not text:
        return {"ok": False, "error": "Alt-Text darf nicht leer sein."}
    if len(text) < 5:
        return {"ok": False, "error": f"Alt-Text zu kurz ({len(text)} Zeichen, mindestens 5)."}
    if len(text) > 500:
        return {
            "ok": False,
            "error": (
                f"Alt-Text zu lang ({len(text)} Zeichen). InkluDocs erlaubt bis 500 "
                "Zeichen, damit Standard-Alt-Texte kompakt bleiben. Kuerze deinen "
                "Vorschlag und biete dem User an, fuer die Details eine separate "
                "Langbeschreibung zu formulieren (Parameter new_langbeschreibung in "
                "update_alt_text, bis 5000 Zeichen) — frage aber zuerst, ob das "
                "gewuenscht ist."
            ),
        }

    lower = text.lower()
    bad_prefixes = ("bild von ", "bild eines ", "bild einer ", "foto von ", "foto eines ", "foto einer ")
    for p in bad_prefixes:
        if lower.startswith(p):
            return {"ok": False, "error": f"Alt-Text beginnt generisch mit '{p.strip()}'. Beschreibe direkt was zu sehen ist."}

    lang_text: Optional[str] = None
    if new_langbeschreibung is not None:
        lang_text = new_langbeschreibung.strip()
        if lang_text and len(lang_text) > 5000:
            return {"ok": False, "error": f"Langbeschreibung zu lang ({len(lang_text)} Zeichen, max 5000)."}

    updates = ["alt_text_edited = ?"]
    params: list[Any] = [text]
    if lang_text is not None:
        updates.append("langbeschreibung = ?")
        params.append(lang_text)
    params.extend([image_id, project_id])

    conn = sqlite3.connect(_DB_PATH)
    try:
        cursor = conn.execute(
            f"UPDATE images SET {', '.join(updates)} WHERE id = ? AND project_id = ?",
            params,
        )
        conn.commit()
        affected = cursor.rowcount
    except sqlite3.Error as e:
        log.exception("update_alt_text DB-Fehler")
        return {"ok": False, "error": f"DB-Fehler beim Speichern: {e}"}
    finally:
        conn.close()

    if affected == 0:
        return {"ok": False, "error": f"Bild {image_id} nicht in Projekt {project_id}."}

    result = {
        "image_id": image_id,
        "saved_alt_text": text,
        "saved_alt_text_length": len(text),
        "info": "Alt-Text in DB-Feld 'alt_text_edited' gespeichert. Original-Pipeline-Ausgabe bleibt in 'alt_text'.",
    }
    if lang_text is not None:
        result["saved_langbeschreibung_length"] = len(lang_text)
        result["langbeschreibung_info"] = "Langbeschreibung wurde ueberschrieben (kein Rollback-Feld vorhanden)."

    return {"ok": True, "result": result}


def revert_alt_text(image_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Setzt `alt_text_edited` auf NULL → Frontend zeigt wieder Pipeline-Original
    (`alt_text`). Manuelle Bearbeitung wird verworfen.

    Achtung: Langbeschreibung wird NICHT zurückgesetzt (kein _edited-Feld
    dafür). Wenn Langbeschreibung auch zurück muss, generate_alt_text neu
    aufrufen.
    """
    if not _check_project_access(project_id, user_id):
        return {"ok": False, "error": "Projekt nicht gefunden oder kein Zugriff."}

    conn = sqlite3.connect(_DB_PATH)
    try:
        cursor = conn.execute(
            "UPDATE images SET alt_text_edited = NULL WHERE id = ? AND project_id = ?",
            (image_id, project_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return {"ok": False, "error": f"Bild {image_id} nicht in Projekt {project_id}."}

        row = conn.execute(
            "SELECT alt_text FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        original = (row[0] if row else "") or ""
    except sqlite3.Error as e:
        log.exception("revert_alt_text DB-Fehler")
        return {"ok": False, "error": f"DB-Fehler beim Rollback: {e}"}
    finally:
        conn.close()

    return {
        "ok": True,
        "result": {
            "image_id": image_id,
            "restored_alt_text": original[:400],
            "restored_alt_text_length": len(original),
            "info": "Manuelle Bearbeitung entfernt. Pipeline-Original ist jetzt wieder aktiv.",
        },
    }
