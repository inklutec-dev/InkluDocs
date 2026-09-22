"""Automatische Pruefung getaggter PDFs — Schritt 5, erste Fassung: nur Urteile (22.09.2026, Steve).

Ablauf je Dokument (pruefe_dokument): Strukturlesung (pdf_struktur) nach Seiten aufteilen, jede Seite
als Bild rendern (PyMuPDF, gecacht), je Seite EIN Modellaufruf mit Bild + Strukturliste
(prompts/builders/pdf_pruefung.py, Schema PruefSeiteOutput), dann Nachpruefung (Kennungen muessen
existieren, Doppelmeldungen weg) und Bericht.

Was die erste Fassung NICHT tut: Sie aendert die Datei nicht. Die Korrektur ueber PDFix-Befehle
(rename/move tags) ist eine spaetere Stufe und nur fuer Befunde mit hoher Sicherheit vorgesehen.

Modell: EINE Stelle (MODELL); Steve 22.09.: Gemini 3.1 Pro, Qualitaet vor Kosten. Ein Modellrouter
kommt spaeter — dann wird nur diese Zeile ersetzt. Preis: billing.AKTIONS_PREISE["pdf_pruefung"]
je Seite (vorlaeufig), verbucht in tagging_api nach erfolgreichem Lauf.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

from pipelines.v4 import llm_client
from prompts.builders.pdf_pruefung import build_pruefung_prompt
from prompts.components.schemas.pdf_pruefung import PruefSeiteOutput

log = logging.getLogger(__name__)

MODELL = os.environ.get("PDF_PRUEFUNG_MODEL") or llm_client.MODEL_GENERATE   # Gemini 3.1 Pro (Stand 22.09.2026)
MAX_SEITEN = int(os.environ.get("PDF_PRUEFUNG_MAX_SEITEN", "60"))
DPI = int(os.environ.get("PDF_PRUEFUNG_DPI", "110"))
MAX_TEXT = 200
MAX_ZEILEN_JE_SEITE = 400
_ARTEN = ("rolle", "ebene", "reihenfolge", "tabelle", "grafik", "fehlt", "sprache", "sonstiges")


class PruefFehler(Exception):
    """Nutzertauglicher Grund."""


def seitenbild(pdf_pfad: str, seite: int, ordner: str) -> str:
    """PNG der Seite (1-basiert), gecacht in <ordner>/<pdfname>.pruef_p<n>.png solange die PDF nicht neuer ist."""
    os.makedirs(ordner, exist_ok=True)
    ziel = os.path.join(ordner, f"{os.path.basename(pdf_pfad)}.pruef_p{seite}.png")
    if os.path.isfile(ziel) and os.path.getmtime(ziel) >= os.path.getmtime(pdf_pfad):
        return ziel
    import fitz
    with fitz.open(pdf_pfad) as pdf:
        if seite < 1 or seite > len(pdf):
            raise PruefFehler(f"Seite {seite} gibt es nicht")
        pdf[seite - 1].get_pixmap(dpi=DPI).save(ziel)
    return ziel


def _kurz(text: str, n: int = MAX_TEXT) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + " …"


def zeilen_fuer_seite(struktur: dict, seite: int) -> tuple[list[str], dict]:
    """Strukturliste einer Seite fuer den Prompt + {Kennung: Element} fuer die Nachpruefung."""
    zeilen: list[str] = []
    kennungen: dict = {}
    for e in struktur.get("elemente") or []:
        if (e.get("seite") or 0) != seite:
            continue
        typ = e.get("typ") or "?"
        if typ in ("Document", "Part", "Art", "Sect", "Div", "NonStruct", "Private", "TBody", "THead", "TFoot", "LBody", "Lbl", "Span"):
            continue
        kenn = "E" + (e.get("id") or "")
        kennungen[kenn] = e
        teile = [f"{kenn} {typ}"]
        if typ == "Table":
            teile.append(f"({e.get('zeilen', '?')} Zeilen, {e.get('spalten', '?')} Spalten)")
        text = _kurz(e.get("text") or "")
        if typ in ("Figure", "Formula"):
            alt = e.get("alt") or e.get("actual") or ""
            teile.append(": Alt-Text: " + (_kurz(alt) if alt else "(keiner)"))
        elif typ == "Form":
            teile.append(f": Feld {e.get('feldname') or '(ohne Namen)'}" + (f", Quickinfo: {_kurz(e.get('quickinfo') or '')}" if e.get("quickinfo") else ""))
        elif text:
            teile.append(": " + text)
        if e.get("lang"):
            teile.append(f" [lang={e['lang']}]")
        zeilen.append("".join(teile))
        if len(zeilen) >= MAX_ZEILEN_JE_SEITE:
            zeilen.append("… (weitere Elemente gekürzt)")
            break
    return zeilen, kennungen


def nachpruefung(befunde: list, kennungen: dict, seite: int) -> list[dict]:
    """Deterministische Nachpruefung: unbekannte Kennung -> niedrig + Hinweis; Doppelmeldungen weg;
    Werte auf die erlaubten Mengen gezogen."""
    out: list[dict] = []
    gesehen: set = set()
    for b in befunde:
        kenn = (b.element or "").strip()
        art = b.art if b.art in _ARTEN else "sonstiges"
        sicherheit = b.sicherheit if b.sicherheit in ("hoch", "mittel", "niedrig") else "niedrig"
        hinweis = ""
        elem = kennungen.get(kenn) if kenn else None
        if kenn and elem is None:
            sicherheit = "niedrig"
            hinweis = "Kennung nicht in der Strukturliste dieser Seite"
        schluessel = (kenn, art, (b.befund or "").strip().lower()[:80])
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        out.append({
            "seite": seite,
            "element": kenn,
            "typ": (elem or {}).get("typ", ""),
            "text": _kurz((elem or {}).get("text") or (elem or {}).get("alt") or "", 120),
            "art": art,
            "befund": (b.befund or "").strip(),
            "vorschlag": (b.vorschlag or "").strip(),
            "beleg": (b.beleg or "").strip(),
            "sicherheit": sicherheit,
            "hinweis": hinweis,
        })
    return out


def pruefe_dokument(pdf_pfad: str, struktur: dict, ordner: str, *, sprache_ausgabe: str = "de",
                    dokument_name: str = "", fortschritt: Optional[Callable[[int, int], None]] = None) -> dict:
    """Prueft alle Seiten (bis MAX_SEITEN). Rueckgabe: Bericht-Dict fuer documents.pruefung_bericht."""
    t0 = time.time()
    info = struktur.get("info") or {}
    seiten = int(info.get("seiten") or 0)
    if seiten <= 0:
        raise PruefFehler("Die PDF hat keine Seiten")
    zu_pruefen = min(seiten, MAX_SEITEN)
    befunde: list[dict] = []
    je_seite: list[dict] = []
    hinweise: list[str] = []
    fehlgeschlagen = 0
    for seite in range(1, zu_pruefen + 1):
        zeilen, kennungen = zeilen_fuer_seite(struktur, seite)
        try:
            bild = seitenbild(pdf_pfad, seite, ordner)
        except Exception as e:  # noqa: BLE001
            log.warning("[pruefung] Seite %s nicht renderbar: %r", seite, e)
            hinweise.append(f"Seite {seite}: Seitenbild konnte nicht erzeugt werden")
            fehlgeschlagen += 1
            continue
        system, prompt = build_pruefung_prompt(zeilen, seite=seite, seiten_gesamt=seiten, sprache_dokument=info.get("lang") or "",
                                               sprache_ausgabe=sprache_ausgabe, dokument_name=dokument_name)
        try:
            out = llm_client.call_with_schema(model=MODELL, prompt=prompt, image_path=bild, schema=PruefSeiteOutput,
                                              max_tokens=4000, temperature=0.0, system=system)
        except llm_client.LLMCallError as e:
            log.error("[pruefung] Seite %s: KI-Anfrage fehlgeschlagen: %s", seite, e)
            hinweise.append(f"Seite {seite}: KI-Anfrage fehlgeschlagen")
            fehlgeschlagen += 1
            continue
        seiten_befunde = nachpruefung(list(out.befunde or []), kennungen, seite)
        befunde.extend(seiten_befunde)
        je_seite.append({"seite": seite, "anzahl": len(seiten_befunde), "zusammenfassung": (out.zusammenfassung or "").strip()})
        if fortschritt:
            try:
                fortschritt(seite, zu_pruefen)
            except Exception:  # noqa: BLE001
                pass
    if fehlgeschlagen and fehlgeschlagen >= zu_pruefen:
        raise PruefFehler("Die KI-Anfrage ist fehlgeschlagen. Bitte später erneut versuchen.")
    if seiten > zu_pruefen:
        hinweise.append(f"Nur die ersten {zu_pruefen} von {seiten} Seiten geprüft")
    anzahl = {"hoch": 0, "mittel": 0, "niedrig": 0}
    for b in befunde:
        anzahl[b["sicherheit"]] = anzahl.get(b["sicherheit"], 0) + 1
    return {
        "zeit": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modell": MODELL,
        "seiten": seiten,
        "seiten_geprueft": zu_pruefen - fehlgeschlagen,
        "dauer_s": round(time.time() - t0, 1),
        "befunde": befunde,
        "je_seite": je_seite,
        "anzahl": anzahl,
        "hinweise": hinweise,
    }
