"""Quickinfo-Werkzeug: Quickinfos in eine PDF zurueckschreiben (27.08.2026).

Gegenstueck zu docx_export.write_alt_texts_to_docx bzw. pdfix_roundtrip.
import_alt_texts_pdfix — hier fuer Formularfelder: je Feld (Anker = voller
Feldname) wird der Eintrag /TU (Quickinfo) gesetzt. Der Schreibweg ist Joerg
Heines Import-Skript (pdfix_scripts/Formular_Import_Quickinfo.py) als
Subprocess; siehe dessen Kopf fuer die Herkunft.

Regeln:
  - Felder OHNE Text (None/leer) bleiben unangetastet — auch eine im Original
    vorhandene Quickinfo bleibt erhalten. Es wird nie eine Quickinfo geloescht.
  - Es wird nichts ausser /TU geaendert (keine Werte, keine Flags, keine Tags).
  - Namenlose Felder (Anker "#n") koennen nicht adressiert werden -> Warnung.
  - Die Quell-PDF wird nie veraendert; das Ziel wird atomar geschrieben
    (erst .tmp, dann Umbenennen), damit nie eine halbe Datei ausgeliefert wird.

Nachpruefung (Ergebnis-Integritaet, wie beim Word-Export):
  - Seitenzahl gleich, Anzahl der Feld-Erscheinungen gleich,
  - sichtbarer Text aller Seiten identisch (page.get_text),
  - jede geschriebene Quickinfo ist in der Zieldatei zurueckzulesen.
  Schlaegt ein Punkt fehl, wird der Export als Fehler gemeldet, nicht still
  ausgeliefert.

SCHREIBWEG (Entscheidung 27.08.2026, zwei Wege, Auswahl ueber FORMULAR_WRITER):
  "pdfix"   = Joerg Heines Import-Skript (pdfix_scripts/Formular_Import_Quickinfo.py):
              /TU ueber das SDK in das Feld-Dictionary, Speichern mit kSaveFull
              (Datei wird neu aufgebaut). STANDARD, sobald eine PDFix-Lizenz
              gesetzt ist (PDFIX_LICENSE_USER/KEY) — siehe Befund unten.
  "pymupdf" = PyMuPDF auf Objekt-Ebene: /TU als UTF-16-String in das FELD-
              Dictionary (bei Feldern mit Kids in das Elternfeld, sonst in das
              gemeinsame Feld/Widget-Dictionary — der Ort, den Acrobat und
              Screenreader lesen), INKREMENTELL gespeichert: die Originalbytes
              bleiben vollstaendig erhalten (byte-genau belegbar, Praefix-
              Vergleich). STANDARD ohne Lizenz (Testversion).
  Ohne gesetzte Variable entscheidet die Lizenz; beide Wege durchlaufen dieselbe
  Nachpruefung und das Ruecklesen ueber Heines Export-Skript.

  BEFUND 27.08.2026 (pdfix-sdk 8.7.10, Michaels Bankformular testformular.pdf):
  In der TESTVERSION (ohne Lizenz) fehlten nach PutString("TU") + Save zufaellig
  andere Widget-Annotationen (Lauf 1: Felder 5-8, Lauf 2: keins, Lauf 3: 7-8;
  Oeffnen + Speichern ohne Aenderung sauber 3/3). MIT LIZENZ (Actino, eingetragen
  27.08. abends): 8/8 Laeufe auf beiden Testformularen ohne Verlust, auch ueber
  Heines Import-Skript, und der Producer-Vermerk "Trial version" entfaellt. Der
  Feldverlust ist also eine Eigenheit der Testversion. Die Nachpruefung bleibt
  fuer beide Wege bestehen (sie hat den Befund gefunden).
"""
from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from formular_processor import pdfix_moeglich

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_IMPORT_SCRIPT = _SCRIPT_DIR / "Formular_Import_Quickinfo.py"
_EXPORT_SCRIPT = _SCRIPT_DIR / "Formular_Export_Quickinfo.py"
_TIMEOUT_SECONDS = 180


class FormularExportFehler(RuntimeError):
    """Export konnte nicht sicher erzeugt werden (Meldung fuer den Nutzer)."""


@dataclass
class FormularExportErgebnis:
    path: str
    geschrieben: int = 0
    writer: str = ""
    nicht_gefunden: list = field(default_factory=list)
    warnungen: list = field(default_factory=list)


def _subprocess(cmd: list, was: str) -> subprocess.CompletedProcess:
    """Subprocess mit Zeitlimit; ein Timeout wird zur Nutzer-Meldung statt zum 500."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR))
    except subprocess.TimeoutExpired:
        raise FormularExportFehler(f"{was} hat zu lange gedauert (Zeitlimit {_TIMEOUT_SECONDS} s).")


def _temp_pfad(verzeichnis: str, praefix: str, endung: str) -> str:
    """Eindeutiger Temp-Dateiname im Arbeitsordner (parallele Exporte stoeren sich nicht)."""
    fd, pfad = tempfile.mkstemp(prefix=praefix, suffix=endung, dir=verzeichnis)
    os.close(fd)
    return pfad


def _quickinfos_lesen(pdf_path: str, work_dir: str) -> dict[str, str]:
    """Liest Feldname -> Quickinfo (Feld-Ebene) ueber Heines Export-Skript."""
    csv_path = _temp_pfad(work_dir, "_pruefung_", ".csv")
    try:
        result = _subprocess([sys.executable, str(_EXPORT_SCRIPT), "-i", pdf_path, "-c", csv_path], "Die Nachprüfung")
        if result.returncode != 0 or not os.path.exists(csv_path):
            raise FormularExportFehler("Die exportierte PDF konnte nicht nachgeprüft werden.")
        out = {}
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter=";"):
                if row and row[0] != "Nummer" and len(row) > 2:
                    out[row[1]] = row[2]
        return out
    finally:
        try:
            os.unlink(csv_path)
        except OSError:
            pass


def _widget_liste(doc) -> list:
    """(Seite, Feldname, Typ) aller Erscheinungen — Vergleichsbasis der Nachpruefung."""
    return sorted((pn + 1, w.field_name or "", w.field_type_string) for pn, page in enumerate(doc) for w in page.widgets())


def _seitentexte_ohne_widgets(doc) -> list[str]:
    """Sichtbarer Text je Seite OHNE Widget-Erscheinungen (Arbeitskopie im Speicher;
    Wortliste sortiert, damit eine andere Objekt-Reihenfolge nicht als Aenderung zaehlt)."""
    kopie = fitz.open("pdf", doc.tobytes())
    try:
        texte = []
        for page in kopie:
            for w in list(page.widgets()):
                page.delete_widget(w)
            texte.append(" ".join(sorted(t[4] for t in page.get_text("words"))))
        return texte
    finally:
        kopie.close()


def aktiver_writer() -> str:
    """"pdfix" oder "pymupdf" — FORMULAR_WRITER, sonst pdfix bei gesetzter Lizenz
    und verfuegbarem SDK, sonst pymupdf (siehe Modulkopf)."""
    moeglich = pdfix_moeglich()
    gewaehlt = os.environ.get("FORMULAR_WRITER", "").strip().lower()
    if gewaehlt in ("pdfix", "pymupdf"):
        return "pdfix" if (gewaehlt == "pdfix" and moeglich) else "pymupdf"
    lizenz = bool(os.environ.get("PDFIX_LICENSE_USER") and os.environ.get("PDFIX_LICENSE_KEY"))
    return "pdfix" if (moeglich and lizenz) else "pymupdf"


def _pdf_string_utf16(text: str) -> str:
    """PDF-Textstring als Hex-String mit UTF-16BE-BOM (<FEFF...>) — vertraegt
    jedes Zeichen, keine Escape-Regeln noetig."""
    return "<" + ("\ufeff" + text).encode("utf-16-be").hex().upper() + ">"


def _schreibe_mit_pymupdf(input_path: str, tmp_path: str, zu_schreiben: dict[str, str]) -> tuple[int, list]:
    """Setzt /TU je Feld auf Objekt-Ebene und speichert inkrementell.

    Ziel des /TU-Eintrags ist das FELD-Dictionary: hat das Widget einen /Parent,
    ist das Elternfeld der Traeger des Namens und der Quickinfo (Kids-Struktur,
    z. B. Radio-Gruppen oder ein Feld auf mehreren Seiten); ohne /Parent sind
    Feld und Widget dasselbe Dictionary. Ein Feld wird genau einmal beschrieben,
    auch wenn es mehrere Erscheinungen hat.
    """
    shutil.copyfile(input_path, tmp_path)   # inkrementell = an die Kopie anhaengen
    doc = fitz.open(tmp_path)
    try:
        gesehen: set[str] = set()
        beschrieben_xrefs: set[int] = set()
        for page in doc:
            for w in page.widgets():
                name = w.field_name or ""
                if name not in zu_schreiben:
                    continue
                gesehen.add(name)
                ziel_xref = w.xref
                # Zum Elternfeld aufsteigen, das den Namen traegt (/T). Bei
                # hierarchischen Namen ("a.b") traegt jedes Level ein /T; der
                # volle Name gehoert zum ERSTEN Dictionary mit /T von unten.
                x = w.xref
                for _ in range(8):
                    typ_t, _val = doc.xref_get_key(x, "T")
                    if typ_t != "null":
                        ziel_xref = x
                        break
                    typ_p, val_p = doc.xref_get_key(x, "Parent")
                    if typ_p != "xref":
                        break
                    x = int(val_p.split()[0])
                if ziel_xref in beschrieben_xrefs:
                    continue
                doc.xref_set_key(ziel_xref, "TU", _pdf_string_utf16(zu_schreiben[name]))
                beschrieben_xrefs.add(ziel_xref)
        nicht_gefunden = [n for n in zu_schreiben if n not in gesehen]
        doc.save(tmp_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        return len(beschrieben_xrefs), nicht_gefunden
    finally:
        doc.close()


def write_quickinfos_to_pdf(input_path: str, output_path: str,
                            quickinfos: dict[str, str]) -> FormularExportErgebnis:
    """Schreibt Quickinfos in eine Kopie der PDF. quickinfos: Anker -> Text.
    Leere Texte werden uebersprungen (Feld bleibt wie im Original)."""
    zu_schreiben = {k: v for k, v in quickinfos.items()
                    if v is not None and str(v).strip() and not str(k).startswith("#")}
    warnungen = [f"Feld Nr. {k[1:]} hat keinen Feldnamen und kann nicht beschrieben werden."
                 for k, v in quickinfos.items() if str(k).startswith("#") and v and str(v).strip()]
    # Steuerzeichen raus (PDF-Strings vertragen keine Zeilenumbrueche sinnvoll),
    # Laenge begrenzen — eine Quickinfo ist ein Satz, kein Absatz.
    for k in list(zu_schreiben):
        t = re.sub(r"[\r\n\t]+", " ", str(zu_schreiben[k])).strip()
        zu_schreiben[k] = t[:1000]

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = _temp_pfad(out_dir, "_export_", ".pdf.tmp")

    try:
        return _schreiben_und_pruefen(input_path, output_path, tmp_path, out_dir, zu_schreiben, warnungen)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)   # bei Erfolg schon nach output_path verschoben, bei Fehler Rest entfernen
        except OSError:
            pass


def _schreiben_und_pruefen(input_path: str, output_path: str, tmp_path: str, out_dir: str,
                           zu_schreiben: dict[str, str], warnungen: list) -> FormularExportErgebnis:
    if not zu_schreiben:
        shutil.copyfile(input_path, tmp_path)
        os.replace(tmp_path, output_path)
        return FormularExportErgebnis(path=output_path, geschrieben=0, writer=aktiver_writer(), warnungen=warnungen)

    writer = aktiver_writer()
    if writer == "pdfix":
        csv_path = _temp_pfad(out_dir, "_import_", ".csv")
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["Nummer", "Name", "Quickinfo"])
                for i, (name, text) in enumerate(zu_schreiben.items(), start=1):
                    w.writerow([i, name, text])
            result = _subprocess([sys.executable, str(_IMPORT_SCRIPT), "-i", input_path, "-o", tmp_path, "-c", csv_path],
                                 "Das Schreiben der Quickinfos")
        finally:
            try:
                os.unlink(csv_path)
            except OSError:
                pass
        m = re.search(r"TU_APPLIED=(\d+) FIELDS_FOUND=(\d+) NOT_FOUND=(.*)", result.stdout or "")
        nicht_gefunden = [n for n in (m.group(3).split("|") if m else []) if n]
        if result.returncode not in (0, 4) or not m or not os.path.exists(tmp_path):
            log.error("Formular-Import fehlgeschlagen rc=%s stderr=%s", result.returncode, (result.stderr or "")[:500])
            raise FormularExportFehler("Die Quickinfos konnten nicht in die PDF geschrieben werden.")
        geschrieben = int(m.group(1))
        for n in nicht_gefunden:
            warnungen.append(f"Feld „{n}“ wurde in der PDF nicht gefunden (Quickinfo nicht geschrieben).")
        # Nachpruefung
        gelesen = _quickinfos_lesen(tmp_path, out_dir)
        fehlend = [n for n, t in zu_schreiben.items() if n not in nicht_gefunden and gelesen.get(n) != t]
        if fehlend:
            log.error("Formular-Export: %d Quickinfos nach dem Schreiben nicht zurueckzulesen: %s", len(fehlend), fehlend[:5])
            raise FormularExportFehler("Die Nachprüfung der exportierten PDF ist fehlgeschlagen (Quickinfos nicht lesbar).")
    else:
        geschrieben, nicht_gefunden = _schreibe_mit_pymupdf(input_path, tmp_path, zu_schreiben)
        for n in nicht_gefunden:
            warnungen.append(f"Feld „{n}“ wurde in der PDF nicht gefunden (Quickinfo nicht geschrieben).")
        # Byte-Beleg: die Originaldatei ist unveraenderter Praefix der Ausgabe.
        with open(input_path, "rb") as fa, open(tmp_path, "rb") as fb:
            a = fa.read()
            if fb.read(len(a)) != a:
                raise FormularExportFehler("Nachprüfung fehlgeschlagen: Originaldaten wurden verändert.")
        if pdfix_moeglich():
            # Gegenprobe mit Heines Export-Skript: liest /TU auf Feld-Ebene.
            gelesen = _quickinfos_lesen(tmp_path, out_dir)
            fehlend = [n for n, t in zu_schreiben.items() if n not in nicht_gefunden and gelesen.get(n) != t]
            if fehlend:
                log.error("Formular-Export: Quickinfos nach dem Schreiben nicht zurueckzulesen: %s", fehlend[:5])
                raise FormularExportFehler("Die Nachprüfung der exportierten PDF ist fehlgeschlagen (Quickinfos nicht lesbar).")

    # Struktur-Nachpruefung: Seiten, Feld-Erscheinungen (Namen je Seite),
    # sichtbarer Seiteninhalt. Der Text wird OHNE Widget-Erscheinungen
    # verglichen (page.get_text liest Erscheinungen mit; die enthalten
    # eingetragene Werte und werden von Schreibwerkzeugen neu erzeugt).
    quelle, ziel = fitz.open(input_path), fitz.open(tmp_path)
    try:
        if quelle.page_count != ziel.page_count:
            raise FormularExportFehler("Nachprüfung fehlgeschlagen: Seitenzahl weicht ab.")
        if _widget_liste(quelle) != _widget_liste(ziel):
            raise FormularExportFehler("Nachprüfung fehlgeschlagen: Formularfelder weichen ab (Anzahl oder Namen).")
        tq, tz = _seitentexte_ohne_widgets(quelle), _seitentexte_ohne_widgets(ziel)
        for i, (a, b) in enumerate(zip(tq, tz)):
            if a != b:
                raise FormularExportFehler(f"Nachprüfung fehlgeschlagen: Text auf Seite {i + 1} weicht ab.")
    finally:
        quelle.close()
        ziel.close()

    os.replace(tmp_path, output_path)
    # Dokument-Eigenschaften (Heine/Karbe 01.09.2026): Creator InkluDocs, Producer = Weg. Darf den
    # Export nicht scheitern lassen — die Quickinfos stehen zu diesem Zeitpunkt bereits in der PDF.
    try:
        import pdf_export as _pe
        _pe.setze_dokumentinfo(output_path, "pdfix" if writer == "pdfix" else "fitz")
    except Exception as e:  # noqa: BLE001
        log.warning("Formular-Export: Dokument-Eigenschaften nicht gesetzt: %s", e)
    return FormularExportErgebnis(path=output_path, geschrieben=geschrieben, writer=writer,
                                  nicht_gefunden=nicht_gefunden, warnungen=warnungen)
