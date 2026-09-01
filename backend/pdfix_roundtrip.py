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
import re
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
                    fig = {
                        "lfnr": int(row[0]),
                        "path": row[1],
                        "title": row[2],
                        "actual_text": row[3],
                        "alt": row[4],
                        "source_pdf": row[5],
                    }
                    # Erweiterung 27.05.2026: Seitennummer + Seitenansicht-Pfad
                    if len(row) > 6 and row[6]:
                        try:
                            fig["page_number"] = int(row[6])
                        except ValueError:
                            pass
                    if len(row) > 7 and row[7]:
                        fig["page_view_path"] = row[7]
                    # Erweiterung 19.06.2026 (Karbe V1004): Spalte 9 = Seiteninhalt
                    # (Text der Bildseite), Spalte 10 = Kapitel-Kontext (Abschnitt
                    # um das Bild). Defensiv gelesen -> aeltere 6-/8-Spalten-CSVs
                    # bleiben gueltig.
                    if len(row) > 8 and row[8]:
                        fig["page_content"] = row[8]
                    if len(row) > 9 and row[9]:
                        fig["chapter_context"] = row[9]
                    figures.append(fig)
                except ValueError:
                    log.warning("PDFix-CSV: Zeile uebersprungen: %s", row)
    log.info("PDFix-Export: %d Figures aus %s", len(figures), pdf_path)
    return figures


KEIN_ALT = "__KEIN_ALT__"   # Konvention mit pdfix_scripts/AltTag_Import_CSV.py (01.09.2026)


def _build_csv_rows(alt_texts_by_lfnr: dict[int, str],
                    source_stem: str) -> list[list]:
    """Baut die CSV-Datenzeilen fuer das Import-Script (ohne Header).

    Regeln (12.06.2026, siehe AltTag_Import_CSV.py Kopfkommentar):
    - Bilder OHNE Text (None oder leer/Whitespace) bekommen KEINE Zeile.
      Das Import-Script laesst Figures ohne Zeile unangetastet — ein evtl.
      vorhandener Original-Alt-Text der Quell-PDF bleibt erhalten, und es
      entstehen keine leeren /Alt-Eintraege mehr (Befund 12.06.2026: Export
      eines teilbearbeiteten Projekts schrieb 6 leere Alt-Texte in die PDF).
    - Die Konvention "dekorativ" (Nutzer hat das Bild explizit als dekorativ
      markiert) wird zu einem leeren Alt-Text uebersetzt — das ist eine
      BEWUSSTE leere Zeile, das Import-Script setzt dann Alt="".
    - Zuordnung erfolgt im Import-Script ueber die laufende Nummer (Spalte 1),
      Luecken sind deshalb erlaubt.
    """
    rows = []
    for lfnr in sorted(alt_texts_by_lfnr):
        alt = alt_texts_by_lfnr[lfnr]
        if alt is None:
            continue  # nicht angefasst -> Figure unangetastet lassen
        if alt == "dekorativ":
            alt = ""
        elif not alt.strip():
            # 01.09.2026 (Michael Karbe): leeres Feld = KEIN Alt-Text in der Datei.
            # Das Import-Script entfernt den Eintrag (Sentinel KEIN_ALT). Vorher wurde
            # die Figure ausgelassen und ein Original-Text der Quelle blieb stehen.
            alt = KEIN_ALT
        rows.append([lfnr, "", "", "", alt, source_stem])
    return rows


def import_alt_texts_pdfix(pdf_in: str, pdf_out: str,
                           alt_texts_by_lfnr: dict[int, str],
                           work_dir: str | None = None) -> int:
    """Schreibt Alt-Texte zurueck in eine getaggte PDF.

    alt_texts_by_lfnr ist {1: "Alt 1", 2: "Alt 2", ...}; die laufende Nummer
    entspricht der Zaehlung von extract_figures_pdfix (= StructTree-Traversierung).
    Eintraege ohne Text werden uebersprungen (Figure bleibt unangetastet),
    "dekorativ" wird zu Alt="" — Details siehe _build_csv_rows.
    Intern: CSV im Heine-Format schreiben, Import-Script als Subprocess.
    Das Script bricht mit Exit-Code 4 ab, wenn CSV-Nummern nicht zur PDF passen
    (Schutz gegen stillen Versatz bei Script-Updates).
    Gibt die Anzahl tatsaechlich gesetzter Alt-Texte zurueck.
    """
    if not _PDFIX_AVAILABLE:
        raise RuntimeError("pdfix-sdk nicht installiert")
    if work_dir is None:
        work_dir = os.path.dirname(pdf_out) or "."
    os.makedirs(work_dir, exist_ok=True)
    csv_path = os.path.join(work_dir, "_pdfix_import.csv")
    rows = _build_csv_rows(alt_texts_by_lfnr, Path(pdf_in).stem)
    if not rows:
        # Nichts zu schreiben: Quell-PDF unveraendert als Export uebernehmen
        # (kein Subprocess noetig, byte-genaue Kopie).
        import shutil
        shutil.copyfile(pdf_in, pdf_out)
        log.info("PDFix-Import: keine Alt-Texte zu setzen, PDF unveraendert kopiert.")
        return 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["laufende Nummer", "Pfad mit Dateinamen", "Titel",
                    "Echter Text", "Alternativer Text", "Dateiname"])
        w.writerow([])
        w.writerows(rows)
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
    # Ergebniszeile des Scripts auswerten ("ALT_APPLIED=n FIGURES_FOUND=m").
    applied = len(rows)
    m = re.search(r"ALT_APPLIED=(\d+)", result.stdout or "")
    if m:
        applied = int(m.group(1))
        if applied != len(rows):
            # Nach dem Konsistenz-Check des Scripts eigentlich unmoeglich —
            # falls doch, soll es im Log auffallen.
            log.warning("PDFix-Import: %d Zeilen geschrieben, aber %d gesetzt.",
                        len(rows), applied)
    log.info("PDFix-Import: %d Alt-Texte gesetzt, Output=%s", applied, pdf_out)
    return applied
