"""Messwerte aus der PDF fuer Pruefung und Korrektur (22.09.2026, Steve: „die Messwerte bekommt er bei Word
doch auch“). Deterministisch mit PyMuPDF, ohne Modell: je Textzeile Schriftgroesse, Fettdruck, Lage und ob
die Zeile allein steht; je Seite die Fliesstext-Groesse (haeufigste Groesse nach Zeichen gewichtet).

Zweck: Das Modell der automatischen Pruefung bekommt die Werte neben der Tag-Liste („16 pt fett, allein“)
und begruendet mit Zahlen statt mit Eindruck; die Korrektur fuehrt einen Befund nur aus, wenn Modell
UND Messung dieselbe Richtung zeigen (Doppelbeleg, pdf_pruefung.doppelbeleg).

Zuordnung Element -> Zeile ueber den normalisierten Textanfang (Strukturlesung liefert den Text je Tag).
Grenzen: gescannte PDFs ohne Textebene liefern nichts (dann keine Messung, keine Auto-Korrektur).
"""
from __future__ import annotations

import re
from typing import Optional

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = (s or "").replace("\xad", "").replace(" ", " ")
    s = _WS.sub(" ", s).strip().lower()
    return s


def seiten_messung(pdf_pfad: str, max_seiten: int = 200) -> dict:
    """{seite (1-basiert): {"fliesstext": pt, "zeilen": [{text, norm, groesse, fett, y0, y1, x0, allein}]}}"""
    import fitz
    out: dict = {}
    with fitz.open(pdf_pfad) as d:
        for pno, page in enumerate(d, 1):
            if pno > max_seiten:
                break
            zeilen = []
            gewicht: dict = {}
            for b in page.get_text("dict").get("blocks", []):
                lines = b.get("lines", [])
                for l in lines:
                    spans = [s for s in l.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in l["spans"]).strip()
                    groesse = round(max(s["size"] for s in spans), 1)
                    fett = any((s.get("flags", 0) & 16) or "bold" in (s.get("font") or "").lower() or "black" in (s.get("font") or "").lower() for s in spans)
                    zeilen.append({"text": text, "norm": _norm(text), "groesse": groesse, "fett": bool(fett),
                                   "y0": round(l["bbox"][1], 1), "y1": round(l["bbox"][3], 1), "x0": round(l["bbox"][0], 1),
                                   "block_zeilen": len(lines)})
                    gewicht[groesse] = gewicht.get(groesse, 0) + len(text)
            zeilen.sort(key=lambda z: (z["y0"], z["x0"]))
            fliesstext = max(gewicht.items(), key=lambda kv: kv[1])[0] if gewicht else 0.0
            # allein: eigener Block mit einer Zeile ODER Luft darueber und darunter (> 0,6 Zeilenhoehe)
            for i, z in enumerate(zeilen):
                h = max(z["y1"] - z["y0"], 1.0)
                oben = z["y0"] - zeilen[i - 1]["y1"] if i > 0 else 999
                unten = zeilen[i + 1]["y0"] - z["y1"] if i + 1 < len(zeilen) else 999
                z["allein"] = bool(z["block_zeilen"] == 1 and oben > 0.6 * h and unten > 0.6 * h)
            out[pno] = {"fliesstext": fliesstext, "zeilen": zeilen}
    return out


def element_messung(text: str, seite: Optional[dict]) -> Optional[dict]:
    """Messwerte fuer ein Element ueber seinen Textanfang; None, wenn nichts zuzuordnen ist."""
    if not seite or not text:
        return None
    norm = _norm(text)
    if len(norm) < 3:
        return None
    kopf = norm[:40]
    treffer = None
    for z in seite["zeilen"]:
        zn = z["norm"]
        if not zn:
            continue
        n = min(len(kopf), len(zn), 40)
        if n >= 3 and kopf[:n] == zn[:n]:
            treffer = z
            break
    if treffer is None:
        return None
    # Zeilen im Element: wie viele getrennte Textzeilen der Seite stecken im Elementtext (zusammengezogene Zellen)
    zeilen_im_element = 0
    for z in seite["zeilen"]:
        zn = z["norm"]
        if len(zn) >= 3 and zn in norm:
            zeilen_im_element += 1
    fl = seite.get("fliesstext") or 0.0
    verh = round(treffer["groesse"] / fl, 2) if fl else None
    return {"groesse": treffer["groesse"], "fett": treffer["fett"], "allein": treffer["allein"], "fliesstext": fl,
            "verhaeltnis": verh, "zeilen_im_element": max(1, zeilen_im_element)}


def messung_text(m: Optional[dict]) -> str:
    """Kurzform fuer Prompt und Bericht: „16 pt fett, allein“ / „10 pt normal, im Block, 3 Zeilen“."""
    if not m:
        return ""
    teile = [f"{m['groesse']:g} pt {'fett' if m['fett'] else 'normal'}", "allein" if m["allein"] else "im Block"]
    if m.get("zeilen_im_element", 1) > 1:
        teile.append(f"{m['zeilen_im_element']} Zeilen")
    return ", ".join(teile)
