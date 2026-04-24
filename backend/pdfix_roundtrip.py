"""PDFix-Roundtrip fuer InkluDocs.

Kooperation mit Actino Software GmbH (www.actino.de), April 2026.
Die Extraktions- und Rueckschreibe-Logik stammt aus den Scripten von
Joerg Heine (heine@actino.de), die hier als Subprocess aufgerufen werden
(siehe backend/pdfix_scripts/). Dieses Modul ist ein schmaler Wrapper.
"""
from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_EXPORT_SCRIPT = _SCRIPT_DIR / "AltTag_Export_CSV_PNG.py"
_IMPORT_SCRIPT = _SCRIPT_DIR / "AltTag_Import_CSV.py"
_TIMEOUT_SECONDS = 120

try:
    from pdfixsdk import GetPdfix
    _PDFIX_AVAILABLE = True
except ImportError:
    _PDFIX_AVAILABLE = False


def is_pdfix_available() -> bool:
    return _PDFIX_AVAILABLE


def is_tagged_pdf(pdf_path: str) -> bool:
    """True wenn die PDF einen StructTree mit mindestens einem Kind hat."""
    if not _PDFIX_AVAILABLE:
        return False
    try:
        pdfix = GetPdfix()
        doc = pdfix.OpenDoc(pdf_path, "")
        if not doc:
            return False
        try:
            st = doc.GetStructTree()
            return st is not None and st.GetNumChildren() > 0
        finally:
            doc.Close()
    except Exception as e:
        log.warning("is_tagged_pdf failed for %s: %s", pdf_path, e)
        return False


def extract_figures_pdfix(pdf_path: str, out_dir: str) -> list[dict]:
    """Ruft Heines Export-Script auf und gibt die Figures als Liste zurueck."""
    if not _PDFIX_AVAILABLE:
        raise RuntimeError("pdfix-sdk nicht installiert")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, str(_EXPORT_SCRIPT),
           "-i", pdf_path, "-o", pdf_path, "-d", out_dir]
    log.info("PDFix-Export aufrufen: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR))
    if result.returncode != 0:
        raise RuntimeError(
            f"PDFix-Export fehlgeschlagen (rc={result.returncode}): "
            f"stderr={result.stderr[:500]}"
        )
    csv_path = os.path.join(out_dir, "figure_array.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"PDFix-Export lief, aber CSV fehlt: {csv_path}")
    figures: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=";"):
            if not row or row[0] in ("laufende Nummer", ""):
                continue
            if len(row) >= 6:
                try:
                    figures.append({
                        "lfnr": int(row[0]),
                        "path": row[1],
                        "title": row[2],
                        "actual_text": row[3],
                        "alt": row[4],
                        "source_pdf": row[5],
                    })
                except ValueError:
                    log.warning("PDFix-CSV: Zeile uebersprungen: %s", row)
    log.info("PDFix-Export: %d Figures aus %s", len(figures), pdf_path)
    return figures


def import_alt_texts_pdfix(pdf_in: str, pdf_out: str,
                           alt_texts_by_lfnr: dict[int, str],
                           work_dir: str | None = None) -> int:
    """Schreibt Alt-Texte zurueck in eine getaggte PDF.

    alt_texts_by_lfnr ist {1: "Alt 1", 2: "Alt 2", ...}, Reihenfolge wie
    bei extract_figures_pdfix (= StructTree-Traversierung).
    Intern: CSV im Heine-Format schreiben, sein Import-Script aufrufen.
    Gibt die Anzahl gesetzter Alt-Texte zurueck.
    """
    if not _PDFIX_AVAILABLE:
        raise RuntimeError("pdfix-sdk nicht installiert")
    if work_dir is None:
        work_dir = os.path.dirname(pdf_out) or "."
    os.makedirs(work_dir, exist_ok=True)
    csv_path = os.path.join(work_dir, "_pdfix_import.csv")
    source_stem = Path(pdf_in).stem
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["laufende Nummer", "Pfad mit Dateinamen", "Titel",
                    "Echter Text", "Alternativer Text", "Dateiname"])
        w.writerow([])
        for lfnr in sorted(alt_texts_by_lfnr):
            w.writerow([lfnr, "", "", "", alt_texts_by_lfnr[lfnr], source_stem])
    cmd = [sys.executable, str(_IMPORT_SCRIPT),
           "-i", pdf_in, "-o", pdf_out, "-c", csv_path]
    log.info("PDFix-Import aufrufen: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR))
    if result.returncode != 0:
        raise RuntimeError(
            f"PDFix-Import fehlgeschlagen (rc={result.returncode}): "
            f"stderr={result.stderr[:500]}"
        )
    log.info("PDFix-Import: %d Alt-Texte gesetzt, Output=%s",
             len(alt_texts_by_lfnr), pdf_out)
    return len(alt_texts_by_lfnr)
