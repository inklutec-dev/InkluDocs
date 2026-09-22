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
import re
import time
from typing import Callable, Optional

import pdf_messung
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


_STILL = ("Document", "Part", "Art", "Sect", "Div", "NonStruct", "Private", "TBody", "THead", "TFoot", "LBody", "Lbl", "Span")


def elemente_der_seite(struktur: dict, seite: int) -> list[dict]:
    return [e for e in (struktur.get("elemente") or []) if (e.get("seite") or 0) == seite and (e.get("typ") or "?") not in _STILL]


def zeilen_fuer_seite(struktur: dict, seite: int, messung: Optional[dict] = None) -> tuple[list[str], dict]:
    """Strukturliste einer Seite fuer den Prompt + {Kennung: Element} fuer die Nachpruefung.
    messung: pdf_messung.seiten_messung()[seite] — dann steht je Element „[16 pt fett, allein]“ dabei (22.09.2026)."""
    zeilen: list[str] = []
    kennungen: dict = {}
    if messung and messung.get("fliesstext"):
        zeilen.append(f"(Messwerte aus der PDF: Fließtext dieser Seite {messung['fliesstext']:g} pt; je Element in eckigen Klammern Schriftgröße, fett/normal, allein stehend oder im Block, ggf. Zahl der Textzeilen im Element)")
    for e in elemente_der_seite(struktur, seite):
        typ = e.get("typ") or "?"
        kenn = "E" + (e.get("id") or "")
        kennungen[kenn] = e
        teile = [f"{kenn} {typ}"]
        m = pdf_messung.element_messung(e.get("text") or "", messung) if messung else None
        if m:
            e["_messung"] = m
            teile.append(" [" + pdf_messung.messung_text(m) + "]")
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


def _tabellen_lage(elem: dict, elemente: list) -> Optional[dict]:
    """Zeile/Spalte einer Zelle in ihrer Tabelle (aus den Kennungen): {zeile, spalte, erste_zelle_th, kopfzeile_th}."""
    eid = elem.get("id") or ""
    if "." not in eid:
        return None
    zeile_id = eid.rsplit(".", 1)[0]
    zellen = [x for x in elemente if (x.get("id") or "").rsplit(".", 1)[0] == zeile_id and x.get("typ") in ("TH", "TD")]
    if not zellen or elem not in zellen:
        return None
    tabelle_id = zeile_id.rsplit(".", 1)[0] if "." in zeile_id else ""
    zeilen = [x for x in elemente if x.get("typ") == "TR" and (x.get("id") or "").startswith(tabelle_id + ".") and (x.get("id") or "").count(".") == zeile_id.count(".")]
    zeile_idx = next((i for i, z in enumerate(zeilen) if z.get("id") == zeile_id), 0)
    kopf = [x for x in elemente if (x.get("id") or "").rsplit(".", 1)[0] == (zeilen[0].get("id") if zeilen else "") and x.get("typ") in ("TH", "TD")]
    return {"zeile": zeile_idx, "spalte": zellen.index(elem), "erste_zelle_th": zellen[0].get("typ") == "TH",
            "kopfzeile_th": bool(kopf) and all(x.get("typ") == "TH" for x in kopf)}


def doppelbeleg(befund: dict, elem: Optional[dict], elemente: list) -> tuple[bool, str]:
    """DOPPELBELEG (Steve 22.09.2026): automatisch korrigiert wird nur, wenn das Modell „hoch“ sagt UND eine
    unabhaengige Quelle (Messung der PDF oder Tabellenlage) dieselbe Richtung zeigt. Alles andere bleibt Hinweis."""
    if not elem or befund.get("sicherheit") != "hoch":
        return False, ""
    typ = elem.get("typ") or ""
    v = (befund.get("vorschlag") or "").strip()
    m = elem.get("_messung")
    art = befund.get("art")
    if art in ("rolle", "ebene") and re.fullmatch(r"H[1-6]", v) and typ in ("P", "LI", "Caption"):
        if m and m.get("allein") and ((m.get("verhaeltnis") or 0) >= 1.15 or m.get("fett")):
            return True, f"Messung: {pdf_messung.messung_text(m)} (Fließtext {m['fliesstext']:g} pt) — steht allein und ist hervorgehoben"
        return False, ("Messung widerspricht oder fehlt: " + pdf_messung.messung_text(m)) if m else "keine Messung möglich"
    if art in ("rolle", "ebene") and v == "P" and typ.startswith("H"):
        if m and not m.get("fett") and (m.get("verhaeltnis") or 9) <= 1.05:
            return True, f"Messung: {pdf_messung.messung_text(m)} (Fließtext {m['fliesstext']:g} pt) — nicht hervorgehoben"
        return False, ("Messung widerspricht oder fehlt: " + pdf_messung.messung_text(m)) if m else "keine Messung möglich"
    if art == "tabelle" and typ == "TH" and v == "TD":
        lage = _tabellen_lage(elem, elemente)
        if lage and lage["spalte"] > 0 and lage["erste_zelle_th"]:
            return True, f"Tabellenlage: Spalte {lage['spalte'] + 1}, erste Zelle der Zeile ist die Kopfzelle — diese Zelle ist ein Wert"
        if lage and lage["spalte"] == 0 and lage["zeile"] > 0 and lage["kopfzeile_th"] and re.fullmatch(r"[\d.,/ -]+", (elem.get("text") or "").strip() or "x"):
            return True, f"Tabellenlage: Datenzeile {lage['zeile'] + 1} unter einer Kopfzeile, Inhalt ist eine Zahl"
        return False, "Tabellenlage stützt die Änderung nicht eindeutig (mögliche Zeilen-Kopfzelle)"
    return False, ""


def ebenen_aus_groesse(befunde: list[dict], struktur: dict) -> None:
    """Die EBENE einer automatisch korrigierbaren Ueberschrift kommt nicht vom Modell, sondern aus dem Rang der
    Schriftgroesse im Dokument (groesste = H1, naechste = H2 …), gemeinsam mit den vorhandenen Ueberschriften.
    Das Modell entscheidet „ist eine Ueberschrift“, die Messung entscheidet „welche Ebene“ (22.09.2026)."""
    groessen: set = set()
    for e in struktur.get("elemente") or []:
        m = e.get("_messung")
        if m and (e.get("typ") or "").startswith("H") and (e.get("typ") or "")[1:].isdigit():
            groessen.add(m["groesse"])
    kandidaten = [b for b in befunde if b.get("auto") and re.fullmatch(r"H[1-6]", b.get("vorschlag") or "") and b.get("_groesse")]
    for b in kandidaten:
        groessen.add(b["_groesse"])
    if not groessen:
        return
    rang = {g: i + 1 for i, g in enumerate(sorted(groessen, reverse=True))}
    for b in kandidaten:
        ebene = min(6, rang[b["_groesse"]])
        if b["vorschlag"] != f"H{ebene}":
            b["doppelbeleg"] += f"; Ebene aus Schriftgröße: {b['_groesse']:g} pt → H{ebene} (Modell: {b['vorschlag']})"
            b["vorschlag"] = f"H{ebene}"
    for b in befunde:
        b.pop("_groesse", None)


def nachpruefung(befunde: list, kennungen: dict, seite: int, elemente: Optional[list] = None) -> list[dict]:
    """Deterministische Nachpruefung: unbekannte Kennung -> niedrig + Hinweis; Doppelmeldungen weg;
    Werte auf die erlaubten Mengen gezogen; Doppelbeleg fuer die Korrektur (auto + Begruendung)."""
    out: list[dict] = []
    gesehen: set = set()
    elemente = elemente if elemente is not None else list(kennungen.values())
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
        eintrag = {
            "seite": seite,
            "element": kenn,
            "obj": (elem or {}).get("obj"),
            "typ": (elem or {}).get("typ", ""),
            "text": _kurz((elem or {}).get("text") or (elem or {}).get("alt") or "", 120),
            "art": art,
            "befund": (b.befund or "").strip(),
            "vorschlag": (b.vorschlag or "").strip(),
            "beleg": (b.beleg or "").strip(),
            "sicherheit": sicherheit,
            "hinweis": hinweis,
            "messung": pdf_messung.messung_text((elem or {}).get("_messung")),
        }
        auto, begr = doppelbeleg(eintrag, elem, elemente)
        eintrag["auto"] = auto
        eintrag["doppelbeleg"] = begr
        if auto and (elem or {}).get("_messung"):
            eintrag["_groesse"] = elem["_messung"]["groesse"]   # fuer ebenen_aus_groesse, wird danach entfernt
        out.append(eintrag)
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
    try:
        messungen = pdf_messung.seiten_messung(pdf_pfad, max_seiten=zu_pruefen)
    except Exception as e:  # noqa: BLE001
        log.warning("[pruefung] Messung nicht moeglich: %r", e)
        messungen = {}
    befunde: list[dict] = []
    je_seite: list[dict] = []
    hinweise: list[str] = []
    fehlgeschlagen = 0
    for seite in range(1, zu_pruefen + 1):
        zeilen, kennungen = zeilen_fuer_seite(struktur, seite, messungen.get(seite))
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
        seiten_befunde = nachpruefung(list(out.befunde or []), kennungen, seite, elemente_der_seite(struktur, seite))
        befunde.extend(seiten_befunde)
        je_seite.append({"seite": seite, "anzahl": len(seiten_befunde), "zusammenfassung": (out.zusammenfassung or "").strip()})
        if fortschritt:
            try:
                fortschritt(seite, zu_pruefen)
            except Exception:  # noqa: BLE001
                pass
    if fehlgeschlagen and fehlgeschlagen >= zu_pruefen:
        raise PruefFehler("Die KI-Anfrage ist fehlgeschlagen. Bitte später erneut versuchen.")
    ebenen_aus_groesse(befunde, struktur)
    if seiten > zu_pruefen:
        hinweise.append(f"Nur die ersten {zu_pruefen} von {seiten} Seiten geprüft")
    anzahl = {"hoch": 0, "mittel": 0, "niedrig": 0, "auto": 0}
    for b in befunde:
        anzahl[b["sicherheit"]] = anzahl.get(b["sicherheit"], 0) + 1
        if b.get("auto"):
            anzahl["auto"] += 1
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
