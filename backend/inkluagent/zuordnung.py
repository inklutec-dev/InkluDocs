"""Zuordnung zu vorhandenen Projekten (Steve 12.10.2026): bevor der InkluAgent ein neues Projekt anlegt oder eine Datei auf
einer Seite ohne Projekt in ein neues Projekt legt, schaut er, ob es schon ein passendes gibt — gleiche Art, aehnlicher Name.
Dann fragt er (Projekt anlegen) bzw. legt Karten vor („In … hochladen“ / „Als neues Projekt hochladen“, inkluagent/hochladen.py).

Aehnlich heisst (nach Normalisieren: Kleinbuchstaben, Dateiendung weg, _ - . als Leerzeichen): gleich, einer enthaelt den
anderen (ab 4 Zeichen), Zeichenfolge zu mindestens 80 % gleich (difflib) oder die Woerter ab 3 Zeichen zu mindestens 60 % gleich.
"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

# Art einer Datei bzw. eines neuen Projekts -> Werkzeuge der Projekte, in die es passt
WERKZEUGE_JE_ART = {
    "pdf": ("pdf", "formular"), "docx": ("word", "uebersetzen"), "bild": ("grafik",), "web": ("web",),
    # projekt_anlegen (konto.py) nennt die Art wie das Werkzeug
    "word": ("word", "uebersetzen"), "formular": ("formular",), "grafik": ("grafik",),
}
_ENDUNGEN = {".pdf", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
MAX_VORSCHLAEGE = 3


def normalisieren(name: str) -> str:
    s = (name or "").strip()
    stamm, endung = os.path.splitext(s)
    if endung.lower() in _ENDUNGEN:
        s = stamm
    s = re.sub(r"[_\-.]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def aehnlich(a: str, b: str) -> bool:
    na, nb = normalisieren(a), normalisieren(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    kurz, lang = sorted((na, nb), key=len)
    if len(kurz) >= 4 and kurz in lang:
        return True
    if SequenceMatcher(None, na, nb).ratio() >= 0.8:
        return True
    wa = {w for w in na.split() if len(w) >= 3}
    wb = {w for w in nb.split() if len(w) >= 3}
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.6


def aehnliche_projekte(user_id: int, name: str, art: str) -> list[dict]:
    """Eigene Projekte derselben Art mit aehnlichem Namen, neueste zuerst, hoechstens drei: [{project_id, name, tool}]."""
    werkzeuge = WERKZEUGE_JE_ART.get(art) or ()
    if not werkzeuge or not normalisieren(name):
        return []
    from database import get_db
    conn = get_db()
    try:
        rows = conn.execute("SELECT id, name, filename, tool FROM projects WHERE user_id = ? AND tool IN (%s) ORDER BY id DESC LIMIT 500"
                            % ",".join("?" * len(werkzeuge)), (user_id, *werkzeuge)).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        pname = r["name"] or r["filename"] or ""
        if aehnlich(name, pname):
            out.append({"project_id": r["id"], "name": pname, "tool": r["tool"]})
            if len(out) >= MAX_VORSCHLAEGE:
                break
    return out
