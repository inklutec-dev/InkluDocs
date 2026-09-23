"""Tagging-Weg „Struktur zuerst“ (23.09.2026, Steves Go): erst verstehen, dann taggen.

Statt PDFix die Struktur raten zu lassen und hinterher zu flicken, wird die Struktur des Dokuments
VORHER bestimmt und PDFix schreibt sie nur noch (pdfix_scripts/Struktur_Schreiben.py, eigenes Skript):

  A  STRUKTUR-HTML   rein rechnerisch aus der PDF (PyMuPDF): je Textzeile Kennung s<Seite>z<n>, Schrift-
                     groesse, Fettdruck, Lage; je Bild Kennung s<Seite>b<n>. Keine KI, kein Raten.
  B  ZUORDNUNG       ein Modellaufruf je Seite (Seitenbild + STRUKTUR-HTML): welche Zeilen Ueberschrift,
                     Artefakt, Bildunterschrift sind; welche Bilder Inhalt tragen; ob die Seite eine Tabelle hat.
                     Das Modell WAEHLT nur (Kennung + Rolle), es schreibt nichts.
  B2 NACHPRUEFUNG    unbekannte Kennungen fallen weg; vergessene Bilder gelten als inhaltlich.
     STILPROFIL      die Ebene der Ueberschriften kommt NICHT vom Modell (seitenlokal, inkonsistent), sondern
                     aus dem Stil (Schriftgroesse, fett) dokumentweit; Titelseiten-Stile schieben nichts nach
                     unten; KLAMMER-PASS in Lesereihenfolge: kein Ebenensprung, gleicher Stil im selben
                     Abschnitt = gleiche Ebene. Damit sind leere Auffuell-Tags (fix_headings) ueberfluessig.
  C  SCHREIBEN       Struktur_Schreiben.py: Hintergrund-Formulare ausschliessen, Artefakte vorgeben, PDFix-
                     Layout, Rollen setzen, AddTags — in derselben Datei, Seiteninhalt unveraendert.
  D  TECHNIK         Joerg Heines Make Accessible OHNE add_tags/fix_headings/set_alt (pdf_tagging.konfig_erzeugen
                     mit struktur_vorgegeben=True): Metadaten, Sprache, Schriften, Links, PDF/UA-Kennung.
     Alt-Texte bleiben leer; sie kommen wie bisher aus der Alt-Text-Pipeline und dem Export (alt_nachtragen).

Ergebnis am Ritterturnier (Michael Karbe, 15 Seiten, 23.09.2026): veraPDF PDF/UA-1 ohne Befund, 39 Ueber-
schriften ohne Sprung und ohne leere Tags, verlorener Text von Seite 3 wieder im Baum.

Schalter: PDF_TAGGING_WEG=struktur (sonst der bisherige PDFix-Weg). Modell: PDF_STRUKTUR_MODEL (Vorgabe
llm_client.MODEL_GENERATE). Fortschritt je Seite ueber Callback. Wirft StrukturFehler mit nutzertauglichem Grund.
"""
from __future__ import annotations

import collections
import concurrent.futures
import html as html_mod
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import pdf_tagging
from pipelines.v4 import llm_client
from prompts.builders.pdf_struktur import build_struktur_prompt
from prompts.components.schemas.pdf_struktur import StrukturSeiteOutput

log = logging.getLogger(__name__)

WEG_ENV = "PDF_TAGGING_WEG"
MODELL = os.environ.get("PDF_STRUKTUR_MODEL") or llm_client.MODEL_GENERATE
DPI = int(os.environ.get("PDF_STRUKTUR_DPI", "110"))
PARALLEL = int(os.environ.get("PDF_STRUKTUR_PARALLEL", "4"))   # gleichzeitige Seitenaufrufe an das Modell
HINTERGRUND_ANTEIL = float(os.environ.get("PDF_STRUKTUR_HINTERGRUND_ANTEIL", "0.6"))
MAX_ZEILEN_JE_SEITE = 400
_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_SCRIPT = _SCRIPT_DIR / "Struktur_Schreiben.py"
_TIMEOUT_SECONDS = int(os.environ.get("PDFIX_STRUKTUR_TIMEOUT", "600"))
_ROLLE_H = re.compile(r"H[1-6]")


class StrukturFehler(pdf_tagging.TaggingFehler):
    """Nutzertauglicher Grund (nie Pfade, Tracebacks oder Schluessel). Erbt von TaggingFehler, damit
    tagging_api._lauf_sync den Grund an den Nutzer weitergibt (Pruefbericht 23.09.2026, Befund 6)."""


def aktiv() -> bool:
    """PDF_TAGGING_WEG=struktur schaltet diesen Weg ein (Vorgabe: bisheriger PDFix-Weg)."""
    return os.environ.get(WEG_ENV, "pdfix").strip().lower() == "struktur"


def verfuegbar() -> bool:
    return _SCRIPT.is_file() and pdf_tagging.verfuegbar()


# ---------------------------------------------------------------------------
# A: Struktur-HTML (deterministisch, PyMuPDF)
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xad", "")).strip()


def druckbar(text: str) -> bool:
    """Hat die Zeile mindestens ein sichtbares Zeichen? Formulare setzen z. B. \\x08 als Platzhalter (Mannheimer 23.09.)."""
    return any(ch.isprintable() and not ch.isspace() for ch in (text or ""))


def _pdf_bbox(rect, rueck) -> list:
    """PyMuPDF-Rechteck (Ursprung oben links, CropBox und /Rotate eingerechnet) -> PDF-Benutzerkoordinaten
    [l, b, r, t] (Ursprung unten links, wie PDFix sie erwartet). rueck = ~page.transformation_matrix.
    Pruefbericht 23.09.2026, Befund 1: vorher hoehe - y, falsch bei versetzter CropBox und gedrehten Seiten."""
    import fitz
    r = fitz.Rect(rect) * rueck
    r.normalize()
    return [round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1)]


MIN_ZEICHNUNGEN_GROSSE_GRAFIK = 40   # ab so vielen komplexen Zeichnungen ist eine (fast) seitenfuellende Gruppe Inhalt
MAX_ZEICHNUNGEN_JE_SEITE = 4000   # darueber (technische Zeichnung, Karte) keine Vektorgruppen suchen (Rechenzeit)


def vektor_gruppen(rechtecke: list, abstand: float = 6.0, mit_anzahl: bool = False) -> list:
    """Rechtecke, die sich (mit Abstand) beruehren, zu Gruppen vereinigen — ueber ein Raster und Union-Find,
    also etwa linear statt kubisch (Pruefbericht Befund 2 + 13: vorher ging „|=“ ins Leere und die Schleife begann
    nach jeder Verschmelzung von vorn). Rueckgabe: Liste vereinigter fitz.Rect; mit_anzahl=True: Liste von
    (fitz.Rect, Zahl der Zeichnungen in der Gruppe)."""
    import fitz
    n = len(rechtecke)
    eltern = list(range(n))

    def wurzel(i):
        while eltern[i] != i:
            eltern[i] = eltern[eltern[i]]
            i = eltern[i]
        return i

    zelle = 24.0
    raster: dict = {}
    grossen = [fitz.Rect(r) + (-abstand, -abstand, abstand, abstand) for r in rechtecke]
    for i, g in enumerate(grossen):
        for cx in range(int(g.x0 // zelle), int(g.x1 // zelle) + 1):
            for cy in range(int(g.y0 // zelle), int(g.y1 // zelle) + 1):
                raster.setdefault((cx, cy), []).append(i)
    for kandidaten in raster.values():
        for a in range(len(kandidaten)):
            for b in range(a + 1, len(kandidaten)):
                i, j = kandidaten[a], kandidaten[b]
                if grossen[i].intersects(rechtecke[j]):
                    wi, wj = wurzel(i), wurzel(j)
                    if wi != wj:
                        eltern[wj] = wi
    gruppen: dict = {}
    anzahl: dict = {}
    for i, r in enumerate(rechtecke):
        w = wurzel(i)
        gruppen[w] = (gruppen[w] | fitz.Rect(r)) if w in gruppen else fitz.Rect(r)
        anzahl[w] = anzahl.get(w, 0) + 1
    if mit_anzahl:
        return [(g, anzahl[w]) for w, g in gruppen.items()]
    return list(gruppen.values())


def struktur_html(pdf_pfad: str) -> list[dict]:
    """Je Seite: {seite, breite, hoehe, fliesstext, zeilen[{id, block, text, size, bold, bbox_pdf, top, left}],
    bilder[{id, bbox_pdf, breite, hoehe, top, left, vektor?}], felder, html}. bbox_pdf = (l, b, r, t) in
    PDF-Benutzerkoordinaten (Ursprung unten links, CropBox/Rotate beruecksichtigt); top/left sind Anzeige-
    koordinaten (oben links) und dienen nur der Lesereihenfolge."""
    import fitz
    seiten: list[dict] = []
    with fitz.open(pdf_pfad) as d:
        for pno, page in enumerate(d, 1):
            hoehe, breite = page.rect.height, page.rect.width
            rueck = ~page.transformation_matrix
            zeilen: list[dict] = []
            gewicht: dict = {}
            voll = False
            for bi, b in enumerate(page.get_text("dict").get("blocks", [])):
                if voll:
                    break   # Pruefbericht Befund 11: das Zeilenlimit gilt fuer die ganze Seite
                if b.get("type") != 0:
                    continue
                for l in b.get("lines", []):
                    spans = [s for s in l.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = _norm("".join(s["text"] for s in l["spans"]))
                    if not druckbar(text):
                        continue   # nur Steuerzeichen (z. B. \x08 als Platzhalter in Formularen, Mannheimer-Antrag 23.09.)
                    size = round(max(s["size"] for s in spans), 1)
                    bold = any((s.get("flags", 0) & 16) or "bold" in (s.get("font") or "").lower() or "black" in (s.get("font") or "").lower() for s in spans)
                    x0, y0, x1, y1 = l["bbox"]
                    zeilen.append({"id": f"s{pno}z{len(zeilen) + 1}", "block": bi, "text": text, "size": size, "bold": bool(bold),
                                   "bbox_pdf": _pdf_bbox(l["bbox"], rueck), "top": round(y0), "left": round(x0)})
                    gewicht[size] = gewicht.get(size, 0) + len(text)
                    if len(zeilen) >= MAX_ZEILEN_JE_SEITE:
                        voll = True
                        break
            fliesstext = max(gewicht.items(), key=lambda kv: kv[1])[0] if gewicht else 0.0
            bilder: list[dict] = []
            raster_rects: list = []
            try:
                infos = page.get_image_info()
            except Exception:  # noqa: BLE001
                infos = []
            for info in infos:
                x0, y0, x1, y1 = info["bbox"]
                if (x1 - x0) < 20 or (y1 - y0) < 20 or (x1 - x0) * (y1 - y0) > HINTERGRUND_ANTEIL * breite * hoehe:
                    continue   # Winzlinge und ganzseitige Hintergruende sind keine Bilder fuer die Zuordnung
                raster_rects.append(fitz.Rect(info["bbox"]))
                bilder.append({"id": f"s{pno}b{len(bilder) + 1}", "bbox_pdf": _pdf_bbox(info["bbox"], rueck),
                               "breite": round(x1 - x0), "hoehe": round(y1 - y0), "top": round(y0), "left": round(x0)})
            # VEKTORGRAFIK-KANDIDATEN (23.09.2026): komplexe Zeichnungen (Kurven oder viele Linien) werden zu Gruppen
            # zusammengefasst und als Bild-Kandidat ins HTML gegeben (Diagramm, Logo, Illustration); einfache Rechtecke
            # und Linien (Kaestchen, Rahmen, Tabellenlinien) sind keine Bilder und werden beim Schreiben Artefakt.
            try:
                zeichnungen = page.get_drawings()
            except Exception:  # noqa: BLE001
                zeichnungen = []
            komplexe = []
            if len(zeichnungen) <= MAX_ZEICHNUNGEN_JE_SEITE:
                for dr in zeichnungen:
                    items = dr.get("items") or []
                    r = dr.get("rect")
                    if r is None:
                        continue
                    r = fitz.Rect(r) & page.rect   # nur der sichtbare Teil (Anschnitt ueber den Seitenrand zaehlt nicht)
                    if r.is_empty or r.width < 3 or r.height < 3 or r.width * r.height > 0.3 * breite * hoehe:
                        continue   # grosse Einzelzeichnung = Flaeche/Hintergrund, kein Diagramm-Baustein (Hofor 23.09.)
                    if any(it[0] == "c" for it in items) or len(items) > 8:
                        komplexe.append(r)
            for g, n_zeichnungen in vektor_gruppen(komplexe, mit_anzahl=True):
                flaeche = g.width * g.height
                if flaeche < 0.004 * breite * hoehe or g.width < 20 or g.height < 20:
                    continue
                if flaeche > HINTERGRUND_ANTEIL * breite * hoehe and n_zeichnungen < MIN_ZEICHNUNGEN_GROSSE_GRAFIK:
                    continue   # grosse Gruppe aus wenigen Formen = Hintergrund/Zierrahmen; aus vielen Pfaden dagegen
                               # eine ganzseitige Karte oder ein Diagramm (Hofor S. 11, Versorgungskarte, 23.09.)
                if any((rr & g).get_area() >= 0.5 * g.get_area() for rr in raster_rects):
                    continue   # ein Rasterbild deckt die Zeichnung ueberwiegend ab und vertritt sie; kleine eingebettete
                               # Bildstuecke (Symbole, Beschriftungskaestchen) machen ein Diagramm NICHT ueberfluessig (Hofor 23.09.)
                bilder.append({"id": f"s{pno}b{len(bilder) + 1}", "bbox_pdf": _pdf_bbox(g, rueck),
                               "breite": round(g.width), "hoehe": round(g.height), "top": round(g.y0), "left": round(g.x0), "vektor": True})
            felder = []
            try:
                for w in page.widgets():
                    felder.append(_pdf_bbox(w.rect, rueck))
            except Exception:  # noqa: BLE001
                pass
            seiten.append({"seite": pno, "breite": breite, "hoehe": hoehe, "fliesstext": fliesstext,
                           "zeilen": zeilen, "bilder": bilder, "felder": felder})
            seiten[-1]["html"] = _html_der_seite(seiten[-1])
    return seiten


def _html_der_seite(s: dict) -> str:
    teile = [f'<section data-seite="{s["seite"]}" data-fliesstext="{s["fliesstext"]:g}pt">']
    letzter = None
    for z in s["zeilen"]:
        if z["block"] != letzter:
            if letzter is not None:
                teile.append("</div>")
            teile.append('<div class="block">')
            letzter = z["block"]
        teile.append(f'<p id="{z["id"]}" data-size="{z["size"]:g}pt" data-bold="{"ja" if z["bold"] else "nein"}" '
                     f'data-top="{z["top"]}" data-left="{z["left"]}">{html_mod.escape(z["text"])}</p>')
    if letzter is not None:
        teile.append("</div>")
    for b in s["bilder"]:
        teile.append(f'<img id="{b["id"]}" data-top="{b["top"]}" data-left="{b["left"]}" data-breite="{b["breite"]}" data-hoehe="{b["hoehe"]}"'
                     + (' data-art="vektorzeichnung"' if b.get("vektor") else '') + ' alt="">')
    teile.append("</section>")
    return "\n".join(teile)


# ---------------------------------------------------------------------------
# B: Zuordnung durch das Modell (ein Aufruf je Seite)
# ---------------------------------------------------------------------------

def seitenbild(pdf_pfad: str, seite: int, ordner: str) -> str:
    """PNG der Seite, gecacht wie bei der Pruefung (<pdf>.struktur_p<n>.png)."""
    os.makedirs(ordner, exist_ok=True)
    ziel = os.path.join(ordner, f"{os.path.basename(pdf_pfad)}.struktur_p{seite}.png")
    if os.path.isfile(ziel) and os.path.getmtime(ziel) >= os.path.getmtime(pdf_pfad):
        return ziel
    import fitz
    with fitz.open(pdf_pfad) as pdf:
        pdf[seite - 1].get_pixmap(dpi=DPI).save(ziel)
    return ziel


def _seitenbilder_loeschen(pfade: dict) -> None:
    for p in pfade.values():
        try:
            os.remove(p)
        except OSError:
            pass


def zuordnung_je_seite(s: dict, seiten_gesamt: int, bild_pfad: str, sprache_dokument: str = "",
                       dokument_name: str = "", modell: Optional[str] = None) -> dict:
    system, prompt = build_struktur_prompt(s["html"], seite=s["seite"], seiten_gesamt=seiten_gesamt,
                                           fliesstext=s["fliesstext"], sprache_dokument=sprache_dokument, dokument_name=dokument_name)
    out = llm_client.call_with_schema(model=modell or MODELL, prompt=prompt, image_path=bild_pfad, schema=StrukturSeiteOutput,
                                      max_tokens=3000, temperature=0.0, system=system)   # Schleifen kamen vom Freitextfeld zusammenfassung (entfernt 23.09.), nicht vom Limit
    return out.model_dump()


# ---------------------------------------------------------------------------
# B2: Nachpruefung, Stilprofil, Klammer-Pass (deterministisch, testbar)
# ---------------------------------------------------------------------------

def nachpruefung(seiten: list[dict], zuordnungen: dict) -> dict:
    """Aus den Modellantworten (seite -> dict) die belegten Zuordnungen: rollen {zeilen_id: Rolle},
    bilder {bild_id: {inhaltlich, alt}}, tabellen {seite: bool}, verworfen [(seite, id, grund)], vergessene_bilder n."""
    ids_text = {z["id"]: z for s in seiten for z in s["zeilen"]}
    ids_bild = {b["id"]: b for s in seiten for b in s["bilder"]}
    rollen: dict = {}
    bilder: dict = {}
    tabellen: dict = {}
    verworfen: list = []
    for pno, d in zuordnungen.items():
        tabellen[pno] = bool(d.get("hat_tabelle"))
        for z in d.get("zeilen") or []:
            zid = (z.get("id") or "").strip()
            if zid not in ids_text or not zid.startswith(f"s{pno}z"):
                verworfen.append((pno, zid, "unbekannte Kennung"))
                continue
            rollen[zid] = z.get("rolle")
        for b in d.get("bilder") or []:
            bid = (b.get("id") or "").strip()
            if bid not in ids_bild or not bid.startswith(f"s{pno}b"):
                verworfen.append((pno, bid, "unbekanntes Bild"))
                continue
            bilder[bid] = {"inhaltlich": bool(b.get("inhaltlich")), "alt": _norm(b.get("alt") or "")[:300]}
    vergessen = 0
    for pno in zuordnungen:
        for bid, b in ids_bild.items():
            if bid.startswith(f"s{pno}b") and bid not in bilder:
                bilder[bid] = {"inhaltlich": True, "alt": ""}   # lieber ein Bild zu viel als ein verlorenes
                vergessen += 1
    return {"rollen": rollen, "bilder": bilder, "tabellen": tabellen, "verworfen": verworfen, "vergessene_bilder": vergessen}


def stilprofil(seiten: list[dict], rollen: dict) -> dict:
    """EBENEN dokumentweit (23.09.2026): Das Modell entscheidet nur „ist Ueberschrift“. Die Ebene kommt aus dem Stil
    (Schriftgroesse, fett): Rang nach Groesse ueber die Stile, die ausserhalb der Titelseite vorkommen; Stile nur
    auf der Titelseite: der groesste = H1, die anderen = Ebene des naechstkleineren wiederkehrenden Stils.
    Dann KLAMMER-PASS in Lesereihenfolge: keine Ebene tiefer als Vorgaenger + 1 (kein Sprung, keine leeren Tags);
    gleicher Stil im selben Abschnitt = gleiche Ebene. Rueckgabe: {rollen, profil (Text), geklammert}."""
    ids_text = {z["id"]: z for s in seiten for z in s["zeilen"]}
    rollen = dict(rollen)
    stil_alle: dict = collections.defaultdict(list)
    for zid, r in rollen.items():
        if _ROLLE_H.fullmatch(r or ""):
            z = ids_text[zid]
            stil_alle[(z["size"], z["bold"])].append(int(zid[1:].split("z")[0]))
    if not stil_alle:
        return {"rollen": rollen, "profil": "", "geklammert": 0}
    wiederkehrend = sorted([k for k, ps in stil_alle.items() if any(pg != 1 for pg in ps)], key=lambda k: (-k[0], not k[1]))
    nur_titel = [k for k in stil_alle if k not in wiederkehrend]
    ebene: dict = {}
    groesster = max(stil_alle, key=lambda k: (k[0], k[1]))
    ebene[groesster] = 1
    start = 2 if groesster not in wiederkehrend else 1
    for i, k in enumerate(wiederkehrend):
        ebene.setdefault(k, start + i)
    for k in nur_titel:
        if k in ebene:
            continue
        kleinere = [w for w in wiederkehrend if w[0] < k[0]]
        ebene[k] = ebene[kleinere[0]] if kleinere else 2
    profil = ", ".join(f"{k[0]:g} pt{' fett' if k[1] else ''} = H{ebene[k]} ({len(stil_alle[k])}{', nur Titelseite' if k in nur_titel else ''})"
                       for k in sorted(ebene, key=lambda k: (-k[0], not k[1])))
    reihenfolge = sorted([zid for zid in rollen if _ROLLE_H.fullmatch(rollen[zid] or "")],
                         key=lambda zid: (int(zid[1:].split("z")[0]), ids_text[zid]["top"], ids_text[zid]["left"]))
    vorher = 0
    geklammert = 0
    letzte_eff: dict = {}   # Stil -> zuletzt vergebene Ebene im laufenden Abschnitt
    vergeben: dict = {}     # zid -> Ebene (fuer ueberlappende Ueberschriften)

    def _ueberlappt(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    for zid in reihenfolge:
        z = ids_text[zid]
        stil = (z["size"], z["bold"])
        e = min(6, ebene[stil])
        # Ueberlappt der Rahmen eine schon vergebene Ueberschrift derselben Seite, ist es optisch EINE Ueberschrift
        # (Hofor-Titelseite: „Årsrapport“ im Glyphenrahmen von „2025“): gleiche Ebene. Sonst haengt die Folge von
        # der Lesereihenfolge ab, und PDFix ordnet solche Zeilen anders als wir (Ebene 2, 1, 3 = Sprung).
        seite_z = zid.split("z")[0]
        partner = [v for k, v in vergeben.items() if k.split("z")[0] == seite_z and _ueberlappt(ids_text[k]["bbox_pdf"], z["bbox_pdf"])]
        if partner:
            e = min(partner)
        elif stil in letzte_eff:
            e = letzte_eff[stil]
        elif e > vorher + 1:
            e = vorher + 1
            geklammert += 1
        for t_, v in list(letzte_eff.items()):
            if v > e:
                del letzte_eff[t_]
        letzte_eff[stil] = e
        rollen[zid] = f"H{e}"
        vergeben[zid] = e
        vorher = e
    return {"rollen": rollen, "profil": profil, "geklammert": geklammert}


# ---------------------------------------------------------------------------
# B3: Listen deterministisch aus dem Struktur-HTML (23.09.2026, Listentest Seite 4 Ritterturnier)
# ---------------------------------------------------------------------------

# Aufzaehlungszeichen ODER Nummer/Buchstabe mit Punkt/Klammer, danach Leerraum und KEINE Ziffer (sonst waere „23.09.2026“ ein
# Listenpunkt, Korpus-Lauf 23.09.: Rechnungen zeigten „Liste mit 1 Eintraegen“ fuer Datumszeilen).
# Pruefbericht Befund 15: Bindestrich/Sternchen/Gedankenstrich nur MIT Leerraum danach (\u201e-5 %\u201c ist kein Punkt);
# Einzelbuchstabe nur, wenn nicht eine Abkuerzung folgt (\u201ez. B.\u201c, \u201ed. h.\u201c, \u201eu. a.\u201c, \u201eA. M\u00fcller\u201c mit Grossbuchstabe).
_AUFZAEHLUNG = re.compile(r"^(?:[\u2022\u25a0\u25cf\u25cb\u25aa\u2043\u25ba\u27a2\u2713\u2714]\s*(?=\S)"
                          r"|[\-\*\u2013\u2014]\s+(?=\S)"
                          r"|(?:\d{1,3}[.)]|\(\d{1,3}\))\s+(?!\d)"
                          r"|[a-z][.)]\s+(?![a-zA-Z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc]\.)(?=[a-z\u00e4\u00f6\u00fc\u00df(\u201e\"])"
                          r"|[A-Z]\)\s+(?=\S))")
_SATZENDE = re.compile(r"[.!?:;]\s*$")
LISTEN_EINZUG_MIN = 4.0      # Fortsetzungszeile: mindestens so viel weiter rechts als das Aufzaehlungszeichen
LISTEN_ABSTAND_MAX = 2.2     # ... und hoechstens so viele Zeilenhoehen unter der letzten Zeile des Punktes


def listen_erkennen(s: dict, rollen: Optional[dict] = None) -> list[list[dict]]:
    """Listen einer Seite ohne KI: Zeile mit Aufzaehlungszeichen oder Nummer = Listenpunkt; eingerueckte
    Folgezeilen dicht darunter = Fortsetzung desselben Punktes (der Umbruch, den PDFix als eigenen Absatz taggt).
    Eine nicht eingerueckte Zeile ohne Zeichen beendet die Liste. Rueckgabe: [[{ids, bboxes}, ...], ...]."""
    zeilen = sorted(s.get("zeilen") or [], key=lambda z: (z["top"], z["left"]))
    rollen = rollen or {}
    listen: list[list[dict]] = []
    akt: list[dict] = []
    punkt: Optional[dict] = None
    for z in zeilen:
        hoehe = max(z["bbox_pdf"][3] - z["bbox_pdf"][1], 1.0)
        if rollen.get(z["id"]):
            # Ueberschrift/Artefakt/Bildunterschrift beendet jede Liste und ist nie Fortsetzung (Ritterturnier S. 2:
            # „Materialliste …“ wurde sonst an „Station 6 …“ angehaengt)
            if akt:
                listen.append(akt)
            akt, punkt = [], None
            continue
        if _AUFZAEHLUNG.match(z["text"]) and len(z["text"]) > 1:
            punkt = {"ids": [z["id"]], "bboxes": [list(z["bbox_pdf"])], "x0": z["left"], "unten": z["bbox_pdf"][1], "letzter_text": z["text"],
                     "size": z["size"], "bold": z["bold"]}
            akt.append(punkt)
            continue
        # Pruefbericht Befund 8: nur Zeilen UNTER dem Punkt (Abstand >= 0) und in derselben Spalte (waagerechte
        # Ueberlappung) sind Fortsetzung — sonst haengt sich bei zweispaltigem Satz die Nachbarspalte an.
        abstand = punkt["unten"] - z["bbox_pdf"][3] if punkt is not None else -1
        gleiche_spalte = punkt is not None and z["bbox_pdf"][0] < max(bb[2] for bb in punkt["bboxes"]) and z["bbox_pdf"][2] > min(bb[0] for bb in punkt["bboxes"])
        dicht = punkt is not None and gleiche_spalte and -0.5 <= abstand <= LISTEN_ABSTAND_MAX * hoehe
        haengend = punkt is not None and z["left"] >= punkt["x0"] + LISTEN_EINZUG_MIN
        # gleicher Stil = gleiche Schriftgroesse (Fettdruck nicht: bei „(1) …“-Absaetzen ist oft nur die Nummer fett;
        # Ueberschriften werden ueber ihre Rolle abgefangen, AVV-Nachlauf 23.09.)
        gleicher_stil = punkt is not None and abs(z["size"] - punkt["size"]) <= 0.6
        # Fortsetzung: gleicher Stil UND (haengender Einzug ODER dicht darunter und der Punkt endet nicht mit Satzzeichen)
        # (Korpus-Lauf 23.09.: AVV mit „(1) …“-Absaetzen ohne Einzug zerfiel in 36 Listen mit je einem Punkt)
        if punkt is not None and not gleiche_spalte:
            continue   # Zeile einer anderen Spalte: beruehrt die laufende Liste nicht (Pruefbericht Befund 8)
        if dicht and gleicher_stil and (haengend or not _SATZENDE.search(punkt["letzter_text"])):
            punkt["ids"].append(z["id"]); punkt["bboxes"].append(list(z["bbox_pdf"])); punkt["unten"] = z["bbox_pdf"][1]; punkt["letzter_text"] = z["text"]
            continue
        if akt:
            listen.append(akt)
        akt, punkt = [], None
    if akt:
        listen.append(akt)
    return [l for l in listen if l]


# ---------------------------------------------------------------------------
# C: Plan + Schreiben (PDFix im Unterprozess)
# ---------------------------------------------------------------------------

def rollen_gruppen(s: dict, rollen: dict) -> list[dict]:
    """Ueberschriften und Bildunterschriften als GRUPPEN (23.09.2026, Mannheimer-/Ritterturnier-Titel): direkt
    aufeinanderfolgende Zeilen mit derselben Rolle und demselben Stil, hoechstens eine Zeilenhoehe Abstand, sind EIN
    Element (ein Titel ueber drei Zeilen ist eine Ueberschrift, nicht drei). Rueckgabe: [{bbox (Vereinigung), tag, zeilen}]."""
    gruppen: list[dict] = []
    for z in sorted(s["zeilen"], key=lambda z: (z["top"], z["left"])):
        r = rollen.get(z["id"])
        if not (r and (_ROLLE_H.fullmatch(r) or r == "Caption")):
            continue
        l, b, rr, t = z["bbox_pdf"]
        hoehe = max(t - b, 1.0)
        g = next((gg for gg in reversed(gruppen) if gg["tag"] == r and gg["stil"] == (z["size"], z["bold"])
                  and 0 <= gg["bbox"][1] - t <= 1.0 * hoehe and abs(gg["bbox"][0] - l) <= 40), None)
        if g:
            g["bbox"] = [min(g["bbox"][0], l), min(g["bbox"][1], b), max(g["bbox"][2], rr), max(g["bbox"][3], t)]
            g["zeilen_ids"].append(z["id"]); g["zeilen"] += 1
            continue
        gruppen.append({"bbox": [l, b, rr, t], "tag": r, "stil": (z["size"], z["bold"]), "zeilen": 1, "zeilen_ids": [z["id"]]})
    # Getrennt wird nur, wenn in DERSELBEN Spalte (waagerechte Ueberlappung) eine andere Zeile zwischen den Zeilen der
    # Gruppe steht; Nachbarspalten (z. B. „GS-Nr.:“ rechts neben einem Formulartitel) trennen nicht.
    ergebnis = []
    for g in gruppen:
        l, b, r, t = g["bbox"]
        eigene = set(g["zeilen_ids"])
        dazwischen = [z for z in s["zeilen"] if z["id"] not in eigene and b < (z["bbox_pdf"][1] + z["bbox_pdf"][3]) / 2 < t
                      and z["bbox_pdf"][0] < r and z["bbox_pdf"][2] > l]
        if g["zeilen"] > 1 and dazwischen:
            for i in g["zeilen_ids"]:
                z = next(zz for zz in s["zeilen"] if zz["id"] == i)
                ergebnis.append({"bbox": list(z["bbox_pdf"]), "tag": g["tag"], "zeilen": 1})
        else:
            ergebnis.append({"bbox": g["bbox"], "tag": g["tag"], "zeilen": g["zeilen"]})
    return ergebnis


def plan_erzeugen(seiten: list[dict], rollen: dict, bilder: dict, tabellen: dict, sprache: str) -> dict:
    plan = {"sprache": sprache, "hintergrund_anteil": HINTERGRUND_ANTEIL, "seiten": []}
    for s in seiten:
        pno = s["seite"]
        eintrag = {"seite": pno, "artefakte": [], "rollen": [], "bilder": [], "tabellen": tabellen.get(pno, True), "zeilen": [], "listen": [],
                   "felder": list(s.get("felder") or [])}
        listen_ids: set = set()
        for liste in listen_erkennen(s, rollen):
            punkte = []
            for pkt in liste:
                punkte.append({"bboxes": pkt["bboxes"]}); listen_ids.update(pkt["ids"])
            if punkte:
                eintrag["listen"].append(punkte)
        for z in s["zeilen"]:
            r = rollen.get(z["id"])
            if r == "Artefakt":
                eintrag["artefakte"].append(z["bbox_pdf"])
                continue
            eintrag["zeilen"].append(z["bbox_pdf"])
        eintrag["rollen"] = rollen_gruppen(s, rollen)
        for b in s["bilder"]:
            u = bilder.get(b["id"]) or {"inhaltlich": True, "alt": ""}
            # Alt bleibt LEER: der Export (pdfua_export.alt_nachtragen) fuellt nur leere /Alt — die Alt-Texte der
            # Pipeline (Ansicht „Alt-Texte“, vom Nutzer geprueft) sind besser als der kurze Modell-Vorschlag hier.
            eintrag["bilder"].append({"bbox": b["bbox_pdf"], "alt": "", "artefakt": not u.get("inhaltlich", True), "vektor": bool(b.get("vektor"))})
        plan["seiten"].append(eintrag)
    return plan


def _grund_aus_ausgabe(stdout: str, stderr: str) -> str:
    text = (stderr or "") + "\n" + (stdout or "")
    if "konnte nicht geoeffnet" in text:
        return "Die PDF konnte nicht geöffnet werden (beschädigt oder verschlüsselt?)"
    if "Speichern fehlgeschlagen" in text:
        return "Die getaggte PDF konnte nicht gespeichert werden"
    return "Das Schreiben der Struktur ist fehlgeschlagen"


def schreiben(pdf_in: str, pdf_out: str, plan: dict, arbeitsordner: str) -> dict:
    """Struktur_Schreiben.py ausfuehren; Rueckgabe: dessen Statistik (rollen, artefakte, bilder, tabellen, zeilen_ohne_element)."""
    # eindeutiger Name (Pruefbericht Befund 10: parallele Laeufe desselben Nutzers teilen den Upload-Ordner)
    fd, plan_pfad = tempfile.mkstemp(prefix="_struktur_plan_", suffix=".json", dir=arbeitsordner)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False)
    cmd = [sys.executable, str(_SCRIPT), "-i", pdf_in, "-o", pdf_out, "-k", plan_pfad]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR))
    except subprocess.TimeoutExpired:
        raise StrukturFehler(f"Das Schreiben der Struktur hat zu lange gedauert (Zeitlimit {_TIMEOUT_SECONDS} s)")
    finally:
        try:
            os.remove(plan_pfad)
        except OSError:
            pass
    if r.returncode != 0 or not os.path.isfile(pdf_out):
        log.warning("[struktur] Schreiben fehlgeschlagen rc=%s stdout=%s stderr=%s", r.returncode, (r.stdout or "")[-400:], (r.stderr or "")[-400:])
        raise StrukturFehler(_grund_aus_ausgabe(r.stdout, r.stderr))
    try:
        return json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# D: Joerg Heines technische Schritte (ohne Strukturerkennung)
# ---------------------------------------------------------------------------

def technische_schritte(pdf_in: str, pdf_out: str, sprache: dict, arbeitsordner: str) -> dict:
    fd, konfig_pfad = tempfile.mkstemp(prefix="_make_accessible_struktur_", suffix=".json", dir=arbeitsordner)
    os.close(fd)
    konfig = pdf_tagging.konfig_erzeugen(sprache["lang"], sprache["overwrite"], konfig_pfad, struktur_vorgegeben=True)
    cmd = [sys.executable, str(pdf_tagging._SCRIPT), "-i", pdf_in, "-o", pdf_out, "-k", konfig_pfad]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=str(pdf_tagging._SCRIPT_DIR))
    except subprocess.TimeoutExpired:
        raise StrukturFehler(f"Die technischen Schritte haben zu lange gedauert (Zeitlimit {_TIMEOUT_SECONDS} s)")
    finally:
        try:
            os.remove(konfig_pfad)
        except OSError:
            pass
    if r.returncode != 0 or not os.path.isfile(pdf_out):
        log.warning("[struktur] technische Schritte fehlgeschlagen rc=%s stdout=%s stderr=%s", r.returncode, (r.stdout or "")[-400:], (r.stderr or "")[-400:])
        raise StrukturFehler("Die technischen Schritte (Metadaten, Sprache, Schriften) sind fehlgeschlagen")
    return konfig


# ---------------------------------------------------------------------------
# Gesamtlauf
# ---------------------------------------------------------------------------

def taggen(pdf_in: str, pdf_out: str, sprache_vorgabe: str = "de", arbeitsordner: Optional[str] = None,
           fortschritt: Optional[Callable[[int, int], None]] = None, dokument_name: str = "") -> dict:
    """Wie pdf_tagging.taggen (gleiche Berichtsform), aber ueber den Weg „Struktur zuerst“.
    pdf_out wird nur bei Erfolg geschrieben. Wirft StrukturFehler / pdf_tagging.TaggingFehler."""
    if not verfuegbar():
        raise StrukturFehler("PDF-Tagging ist auf diesem Server nicht eingerichtet")
    if not os.path.isfile(pdf_in):
        raise StrukturFehler("Die Quelldatei fehlt")
    try:
        seitenzahl = pdf_tagging.seitenzahl(pdf_in)
    except Exception:  # noqa: BLE001
        raise StrukturFehler("Die PDF konnte nicht gelesen werden")
    if seitenzahl > pdf_tagging.MAX_SEITEN:
        raise StrukturFehler(f"Die PDF hat {seitenzahl} Seiten; das Tagging ist auf {pdf_tagging.MAX_SEITEN} Seiten begrenzt")
    arbeitsordner = arbeitsordner or os.path.dirname(pdf_out) or "."
    os.makedirs(arbeitsordner, exist_ok=True)
    t0 = time.time()
    vorher = pdf_tagging.tag_statistik(pdf_in)
    sprache = pdf_tagging.sprache_bestimmen(pdf_in, sprache_vorgabe)
    hinweise: list[str] = [h for h in (sprache.get("hinweis"),) if h]
    # A
    try:
        seiten = struktur_html(pdf_in)
    except Exception as e:  # noqa: BLE001
        log.warning("[struktur] Struktur-HTML nicht moeglich: %r", e)
        raise StrukturFehler("Die PDF konnte nicht gelesen werden")
    if not any(s["zeilen"] for s in seiten):
        raise StrukturFehler("Die PDF enthält keinen lesbaren Text (gescannt?) — der Weg „Struktur zuerst“ braucht eine Textebene")
    # B — Seiten PARALLEL (23.09.2026: 100-Seiten-Dokumente; Drosselung 429 faengt der Gemini-Client mit Wartezeit ab)
    t_modell = time.time()
    zuordnungen: dict = {}
    fehler_seiten: dict = {}
    zu_fragen = [s for s in seiten if s["zeilen"] or s["bilder"]]
    for s in seiten:
        if not (s["zeilen"] or s["bilder"]):
            zuordnungen[s["seite"]] = {"zeilen": [], "bilder": [], "hat_tabelle": False}
    fertig = [0]
    sperre = threading.Lock()
    # Seitenbilder VORAB im Hauptthread (Pruefbericht Befund 7: PyMuPDF ist nicht threadsicher); nur die
    # Modellaufrufe laufen parallel. Die Bilder werden am Ende geloescht (Befund 19: Kundendaten, Plattenplatz).
    bilder_pfade: dict = {}
    for s in zu_fragen:
        try:
            bilder_pfade[s["seite"]] = seitenbild(pdf_in, s["seite"], arbeitsordner)
        except Exception as e:  # noqa: BLE001
            log.warning("[struktur] Seite %s: Seitenbild nicht moeglich: %r", s["seite"], e)

    def _eine_seite(s: dict):
        bild = bilder_pfade.get(s["seite"])
        if not bild:
            raise StrukturFehler("Seitenbild fehlt")
        return zuordnung_je_seite(s, len(seiten), bild, sprache_dokument=sprache.get("lang") or "", dokument_name=dokument_name)

    def _melden():
        with sperre:
            fertig[0] += 1
            stand = fertig[0]
        if fortschritt:
            try:
                fortschritt(stand, len(zu_fragen))
            except Exception:  # noqa: BLE001
                pass

    def _runde(liste: list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, PARALLEL)) as pool:
            auftraege = {pool.submit(_eine_seite, s): s for s in liste}
            for f in concurrent.futures.as_completed(auftraege):
                s = auftraege[f]
                try:
                    zuordnungen[s["seite"]] = f.result()
                    fehler_seiten.pop(s["seite"], None)
                except Exception as e:  # noqa: BLE001
                    log.warning("[struktur] Seite %s: KI-Zuordnung fehlgeschlagen: %r", s["seite"], e)
                    fehler_seiten[s["seite"]] = e
                _melden()

    try:
        _runde(zu_fragen)
    except BaseException:
        _seitenbilder_loeschen(bilder_pfade)
        raise
    # Zweiter Anlauf fuer Seiten ohne Zuordnung (meist Drosselung des Anbieters), nacheinander — auch wenn in der
    # ersten Runde ALLE Seiten scheiterten (Befund 18: ein einseitiges Dokument scheiterte sonst am ersten 429)
    offen = [s for s in zu_fragen if s["seite"] in fehler_seiten]
    if offen:
        time.sleep(8)
        for s in offen:
            try:
                zuordnungen[s["seite"]] = _eine_seite(s)
                fehler_seiten.pop(s["seite"], None)
            except Exception as e:  # noqa: BLE001
                log.warning("[struktur] Seite %s: auch der zweite Anlauf schlug fehl: %r", s["seite"], e)
    _seitenbilder_loeschen(bilder_pfade)
    fehlgeschlagen = len(fehler_seiten)
    for seite in sorted(fehler_seiten):
        hinweise.append(f"Seite {seite}: KI-Zuordnung fehlgeschlagen, Seite ohne Überschriften-Vorgabe getaggt")
    if zu_fragen and fehlgeschlagen >= len(zu_fragen):
        raise StrukturFehler("Die KI-Anfrage ist fehlgeschlagen. Bitte später erneut versuchen.")
    modell_dauer = round(time.time() - t_modell, 1)
    # B2
    np_ = nachpruefung(seiten, zuordnungen)
    sp = stilprofil(seiten, np_["rollen"])
    rollen = sp["rollen"]
    # C
    plan = plan_erzeugen(seiten, rollen, np_["bilder"], np_["tabellen"], sprache["lang"])
    zwischen = pdf_out + ".struktur.tmp.pdf"
    try:
        stat = schreiben(pdf_in, zwischen, plan, arbeitsordner)
        # D
        konfig = technische_schritte(zwischen, pdf_out, sprache, arbeitsordner)
    finally:
        try:
            os.remove(zwischen)
        except OSError:
            pass
    from pdf_export import pdf_hat_tags
    if not pdf_hat_tags(pdf_out):
        raise StrukturFehler("Es wurde keine Struktur erzeugt")
    nachher = pdf_tagging.tag_statistik(pdf_out)
    zeilen_gesamt = sum(len(e["zeilen"]) for e in plan["seiten"])
    ohne = int(stat.get("zeilen_ohne_element") or 0)
    if ohne:
        hinweise.append(f"{ohne} von {zeilen_gesamt} Textzeilen konnten keinem Element zugeordnet werden — bitte in der Hörprobe prüfen")
    if np_["vergessene_bilder"]:
        hinweise.append(f"{np_['vergessene_bilder']} Bilder ohne KI-Urteil wurden als inhaltlich getaggt")
    zaehl = collections.Counter(rollen.values())
    return {
        "zeit": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dauer_s": round(time.time() - t0, 1),
        "seiten": seitenzahl,
        "modus": pdf_tagging.lizenz_modus(),
        "testmodus": nachher["testmodus"],
        "sprache": sprache,
        "konfig": konfig,
        "vorher": {k: vorher[k] for k in ("elemente", "ueberschriften", "listen", "tabellen", "bilder", "absaetze", "titel", "lang")},
        "nachher": nachher,
        "hinweise": ["Weg „Struktur zuerst“: KI-Zuordnung je Seite, Ebenen aus dem Stilprofil, PDFix schreibt den Baum"] + hinweise,
        "weg": "struktur",
        "struktur": {
            "modell": MODELL, "modell_dauer_s": modell_dauer, "seiten_ohne_zuordnung": fehlgeschlagen,
            "ueberschriften": sum(v for k, v in zaehl.items() if _ROLLE_H.fullmatch(k or "")),
            "artefakte": zaehl.get("Artefakt", 0), "captions": zaehl.get("Caption", 0),
            "bilder_inhaltlich": sum(1 for b in np_["bilder"].values() if b["inhaltlich"]),
            "bilder_schmuck": sum(1 for b in np_["bilder"].values() if not b["inhaltlich"]),
            "verworfen": len(np_["verworfen"]), "stilprofil": sp["profil"], "geklammert": sp["geklammert"],
            "listen": sum(len(e["listen"]) for e in plan["seiten"]),
            "listenpunkte_umbrochen": sum(1 for e in plan["seiten"] for l in e["listen"] for pkt in l if len(pkt["bboxes"]) > 1),
            "geschrieben": {k: stat.get(k) for k in ("rollen", "artefakte", "bilder", "tabellen", "hintergrund", "zeilen_ohne_element")},
            "zeilen_gesamt": zeilen_gesamt, "zeilen_ohne_element": ohne, "seiten_parallel": PARALLEL,
        },
    }
