"""Automatische Korrektur getaggter PDFs — Stufe 2 der Pruefung (22.09.2026, Steves Go).

Fuehrt nur Befunde aus, die den DOPPELBELEG tragen (pdf_pruefung.doppelbeleg: Modell „hoch“ UND Messung
in derselben Richtung). Heute: Rollen (P <-> H1..H6, LI -> Hn, TH <-> TD) ueber das eigene PDFix-Skript
pdfix_scripts/Korrektur_Anwenden.py. Zusammengezogene Zellen, fehlende Inhalte, Grafiken bleiben Hinweise.

Rueckweg: vor jeder Korrektur eine Sicherung (<pdf>.vor_korrektur.pdf); rueckgaengig() stellt sie wieder her.
Kosten: keine Credits (rein mechanisch). Die Nachpruefung ist ein eigener, bezahlter Schritt (tagging_api).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_SCRIPT = _SCRIPT_DIR / "Korrektur_Anwenden.py"
_TIMEOUT_SECONDS = int(os.environ.get("PDFIX_KORREKTUR_TIMEOUT", "300"))


class KorrekturFehler(Exception):
    """Nutzertauglicher Grund."""


def verfuegbar() -> bool:
    return _SCRIPT.is_file()


def sicherung_pfad(pdf_pfad: str) -> str:
    return pdf_pfad + ".vor_korrektur.pdf"


def anwenden(pdf_pfad: str, befunde: list[dict], uebers: Optional[Callable[[str], str]] = None) -> dict:
    """Wendet die auto-korrigierbaren Befunde an. befunde: Eintraege des Pruefberichts mit obj + vorschlag.
    Rueckgabe: Korrektur-Bericht (zeit, angewendet[], nicht_gefunden, sicherung, verapdf)."""
    if not verfuegbar():
        raise KorrekturFehler("Die Korrektur ist auf diesem Server nicht eingerichtet")
    if not os.path.isfile(pdf_pfad):
        raise KorrekturFehler("Die Datei fehlt")
    plan = []
    for b in befunde:
        if not b.get("auto") or not b.get("obj"):
            continue
        typ = (b.get("vorschlag") or "").strip()
        if not re.fullmatch(r"H[1-6]|P|TH|TD|LI|L|Figure|Caption", typ):
            continue
        plan.append({"obj": int(b["obj"]), "typ": typ, "befund": b})
    if not plan:
        raise KorrekturFehler("Kein Befund mit Doppelbeleg — nichts zu korrigieren")
    sicherung = sicherung_pfad(pdf_pfad)
    shutil.copyfile(pdf_pfad, sicherung)
    ordner = os.path.dirname(pdf_pfad)
    stamm = os.path.basename(pdf_pfad)
    kdatei = os.path.join(ordner, f"{stamm}.korrekturen.json")
    tmp = os.path.join(ordner, f"{stamm}.korr.tmp.pdf")
    with open(kdatei, "w", encoding="utf-8") as f:
        json.dump([{"obj": p["obj"], "typ": p["typ"]} for p in plan], f)
    cmd = [sys.executable, str(_SCRIPT), "-i", pdf_pfad, "-o", tmp, "-k", kdatei]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR))
    except subprocess.TimeoutExpired:
        raise KorrekturFehler("Die Korrektur hat zu lange gedauert")
    finally:
        try:
            os.remove(kdatei)
        except OSError:
            pass
    if r.returncode != 0 or not os.path.isfile(tmp):
        log.warning("[korrektur] rc=%s stderr=%s", r.returncode, (r.stderr or "")[-300:])
        raise KorrekturFehler("Die Korrektur konnte nicht ausgeführt werden")
    try:
        ergebnis = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        ergebnis = {"angewendet": 0, "nicht_gefunden": [], "unveraendert": []}
    os.replace(tmp, pdf_pfad)
    fehlt = set(ergebnis.get("nicht_gefunden") or []) | set(ergebnis.get("unveraendert") or [])
    angewendet = []
    for p in plan:
        b = p["befund"]
        eintrag = {"seite": b.get("seite"), "element": b.get("element"), "obj": p["obj"], "typ_vorher": b.get("typ"),
                   "typ_nachher": p["typ"], "text": b.get("text"), "begruendung": b.get("doppelbeleg") or "",
                   "beleg": b.get("beleg") or ""}
        if p["obj"] in fehlt:
            eintrag["status"] = "nicht_gefunden"
        else:
            eintrag["status"] = "angewendet"
        angewendet.append(eintrag)
    verapdf = None
    try:
        import pdf_tagging
        verapdf = pdf_tagging.verapdf(pdf_pfad, uebers)
    except Exception as e:  # noqa: BLE001
        log.warning("[korrektur] veraPDF nach der Korrektur nicht moeglich: %r", e)
    return {
        "zeit": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dauer_s": round(time.time() - t0, 1),
        "angewendet": angewendet,
        "anzahl": int(ergebnis.get("angewendet") or 0),
        "sicherung": sicherung,
        "verapdf": verapdf,
    }


def rueckgaengig(pdf_pfad: str, sicherung: Optional[str] = None) -> None:
    s = sicherung or sicherung_pfad(pdf_pfad)
    if not os.path.isfile(s):
        raise KorrekturFehler("Es gibt keine Sicherung von vor der Korrektur")
    os.replace(s, pdf_pfad)
    # Zeitstempel auf jetzt: die Sicherung ist aelter als die Zwischenspeicher der Strukturlesung und der
    # Seitenbilder (<pdf>.struktur.json, .pruef_p<n>.png) — sonst zeigten Hoerprobe und Pruefung den alten Stand.
    os.utime(pdf_pfad, None)
