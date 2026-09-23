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
import html as html_mod
import json
import logging
import os
import re
import subprocess
import sys
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
HINTERGRUND_ANTEIL = float(os.environ.get("PDF_STRUKTUR_HINTERGRUND_ANTEIL", "0.6"))
MAX_ZEILEN_JE_SEITE = 400
_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_SCRIPT = _SCRIPT_DIR / "Struktur_Schreiben.py"
_TIMEOUT_SECONDS = int(os.environ.get("PDFIX_STRUKTUR_TIMEOUT", "600"))
_ROLLE_H = re.compile(r"H[1-6]")


class StrukturFehler(Exception):
    """Nutzertauglicher Grund (nie Pfade, Tracebacks oder Schluessel)."""


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


def struktur_html(pdf_pfad: str) -> list[dict]:
    """Je Seite: {seite, breite, hoehe, fliesstext, zeilen[{id, block, text, size, bold, bbox_pdf, top, left}],
    bilder[{id, bbox_pdf, breite, hoehe, top, left}], html}. bbox_pdf = (l, b, r, t) mit Ursprung unten links."""
    import fitz
    seiten: list[dict] = []
    with fitz.open(pdf_pfad) as d:
        for pno, page in enumerate(d, 1):
            hoehe, breite = page.rect.height, page.rect.width
            zeilen: list[dict] = []
            gewicht: dict = {}
            for bi, b in enumerate(page.get_text("dict").get("blocks", [])):
                if b.get("type") != 0:
                    continue
                for l in b.get("lines", []):
                    spans = [s for s in l.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = _norm("".join(s["text"] for s in l["spans"]))
                    if not text:
                        continue
                    size = round(max(s["size"] for s in spans), 1)
                    bold = any((s.get("flags", 0) & 16) or "bold" in (s.get("font") or "").lower() or "black" in (s.get("font") or "").lower() for s in spans)
                    x0, y0, x1, y1 = l["bbox"]
                    zeilen.append({"id": f"s{pno}z{len(zeilen) + 1}", "block": bi, "text": text, "size": size, "bold": bool(bold),
                                   "bbox_pdf": [round(x0, 1), round(hoehe - y1, 1), round(x1, 1), round(hoehe - y0, 1)],
                                   "top": round(y0), "left": round(x0)})
                    gewicht[size] = gewicht.get(size, 0) + len(text)
                    if len(zeilen) >= MAX_ZEILEN_JE_SEITE:
                        break
            fliesstext = max(gewicht.items(), key=lambda kv: kv[1])[0] if gewicht else 0.0
            bilder: list[dict] = []
            try:
                infos = page.get_image_info()
            except Exception:  # noqa: BLE001
                infos = []
            for info in infos:
                x0, y0, x1, y1 = info["bbox"]
                if (x1 - x0) < 20 or (y1 - y0) < 20 or (x1 - x0) * (y1 - y0) > HINTERGRUND_ANTEIL * breite * hoehe:
                    continue   # Winzlinge und ganzseitige Hintergruende sind keine Bilder fuer die Zuordnung
                bilder.append({"id": f"s{pno}b{len(bilder) + 1}", "bbox_pdf": [round(x0, 1), round(hoehe - y1, 1), round(x1, 1), round(hoehe - y0, 1)],
                               "breite": round(x1 - x0), "hoehe": round(y1 - y0), "top": round(y0), "left": round(x0)})
            seiten.append({"seite": pno, "breite": breite, "hoehe": hoehe, "fliesstext": fliesstext,
                           "zeilen": zeilen, "bilder": bilder})
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
        teile.append(f'<img id="{b["id"]}" data-top="{b["top"]}" data-left="{b["left"]}" data-breite="{b["breite"]}" data-hoehe="{b["hoehe"]}" alt="">')
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


def zuordnung_je_seite(s: dict, seiten_gesamt: int, bild_pfad: str, sprache_dokument: str = "",
                       dokument_name: str = "", modell: Optional[str] = None) -> dict:
    system, prompt = build_struktur_prompt(s["html"], seite=s["seite"], seiten_gesamt=seiten_gesamt,
                                           fliesstext=s["fliesstext"], sprache_dokument=sprache_dokument, dokument_name=dokument_name)
    out = llm_client.call_with_schema(model=modell or MODELL, prompt=prompt, image_path=bild_pfad, schema=StrukturSeiteOutput,
                                      max_tokens=3000, temperature=0.0, system=system)
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
    for zid in reihenfolge:
        z = ids_text[zid]
        stil = (z["size"], z["bold"])
        e = min(6, ebene[stil])
        if stil in letzte_eff:
            e = letzte_eff[stil]
        elif e > vorher + 1:
            e = vorher + 1
            geklammert += 1
        for t_, v in list(letzte_eff.items()):
            if v > e:
                del letzte_eff[t_]
        letzte_eff[stil] = e
        rollen[zid] = f"H{e}"
        vorher = e
    return {"rollen": rollen, "profil": profil, "geklammert": geklammert}


# ---------------------------------------------------------------------------
# B3: Listen deterministisch aus dem Struktur-HTML (23.09.2026, Listentest Seite 4 Ritterturnier)
# ---------------------------------------------------------------------------

_AUFZAEHLUNG = re.compile(r"^(?:[\u2022\u25a0\u25cf\u25cb\u25aa\u2013\u2014\-\*\u2043\u25ba\u27a2\u2713\u2714]|\d{1,3}[.)]|[a-zA-Z][.)]|\(\d{1,3}\))\s*")
LISTEN_EINZUG_MIN = 4.0      # Fortsetzungszeile: mindestens so viel weiter rechts als das Aufzaehlungszeichen
LISTEN_ABSTAND_MAX = 2.2     # ... und hoechstens so viele Zeilenhoehen unter der letzten Zeile des Punktes


def listen_erkennen(s: dict) -> list[list[dict]]:
    """Listen einer Seite ohne KI: Zeile mit Aufzaehlungszeichen oder Nummer = Listenpunkt; eingerueckte
    Folgezeilen dicht darunter = Fortsetzung desselben Punktes (der Umbruch, den PDFix als eigenen Absatz taggt).
    Eine nicht eingerueckte Zeile ohne Zeichen beendet die Liste. Rueckgabe: [[{ids, bboxes}, ...], ...]."""
    zeilen = sorted(s.get("zeilen") or [], key=lambda z: (z["top"], z["left"]))
    listen: list[list[dict]] = []
    akt: list[dict] = []
    punkt: Optional[dict] = None
    for z in zeilen:
        hoehe = max(z["bbox_pdf"][3] - z["bbox_pdf"][1], 1.0)
        if _AUFZAEHLUNG.match(z["text"]) and len(z["text"]) > 1:
            punkt = {"ids": [z["id"]], "bboxes": [list(z["bbox_pdf"])], "x0": z["left"], "unten": z["bbox_pdf"][1]}
            akt.append(punkt)
            continue
        if punkt is not None and z["left"] >= punkt["x0"] + LISTEN_EINZUG_MIN and (punkt["unten"] - z["bbox_pdf"][3]) <= LISTEN_ABSTAND_MAX * hoehe:
            punkt["ids"].append(z["id"]); punkt["bboxes"].append(list(z["bbox_pdf"])); punkt["unten"] = z["bbox_pdf"][1]
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

def plan_erzeugen(seiten: list[dict], rollen: dict, bilder: dict, tabellen: dict, sprache: str) -> dict:
    plan = {"sprache": sprache, "hintergrund_anteil": HINTERGRUND_ANTEIL, "seiten": []}
    for s in seiten:
        pno = s["seite"]
        eintrag = {"seite": pno, "artefakte": [], "rollen": [], "bilder": [], "tabellen": tabellen.get(pno, True), "zeilen": [], "listen": []}
        listen_ids: set = set()
        for liste in listen_erkennen(s):
            punkte = []
            for pkt in liste:
                if any(rollen.get(i) for i in pkt["ids"]):
                    continue   # Zeilen mit Rolle (Ueberschrift/Artefakt) sind keine Listenpunkte
                punkte.append({"bboxes": pkt["bboxes"]}); listen_ids.update(pkt["ids"])
            if punkte:
                eintrag["listen"].append(punkte)
        for z in s["zeilen"]:
            r = rollen.get(z["id"])
            if r == "Artefakt":
                eintrag["artefakte"].append(z["bbox_pdf"])
                continue
            eintrag["zeilen"].append(z["bbox_pdf"])
            if r and (_ROLLE_H.fullmatch(r) or r == "Caption"):
                eintrag["rollen"].append({"bbox": z["bbox_pdf"], "tag": r})
        for b in s["bilder"]:
            u = bilder.get(b["id"]) or {"inhaltlich": True, "alt": ""}
            # Alt bleibt LEER: der Export (pdfua_export.alt_nachtragen) fuellt nur leere /Alt — die Alt-Texte der
            # Pipeline (Ansicht „Alt-Texte“, vom Nutzer geprueft) sind besser als der kurze Modell-Vorschlag hier.
            eintrag["bilder"].append({"bbox": b["bbox_pdf"], "alt": "", "artefakt": not u.get("inhaltlich", True)})
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
    plan_pfad = os.path.join(arbeitsordner, f"_struktur_plan_{int(time.time())}.json")
    with open(plan_pfad, "w", encoding="utf-8") as f:
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
    konfig_pfad = os.path.join(arbeitsordner, f"_make_accessible_struktur_{int(time.time())}.json")
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
    # B
    t_modell = time.time()
    zuordnungen: dict = {}
    fehlgeschlagen = 0
    for s in seiten:
        if not s["zeilen"] and not s["bilder"]:
            zuordnungen[s["seite"]] = {"zeilen": [], "bilder": [], "hat_tabelle": False}
            continue
        try:
            bild = seitenbild(pdf_in, s["seite"], arbeitsordner)
            zuordnungen[s["seite"]] = zuordnung_je_seite(s, len(seiten), bild, sprache_dokument=sprache.get("lang") or "", dokument_name=dokument_name)
        except llm_client.LLMCallError as e:
            log.error("[struktur] Seite %s: KI-Anfrage fehlgeschlagen: %s", s["seite"], e)
            fehlgeschlagen += 1
            hinweise.append(f"Seite {s['seite']}: KI-Zuordnung fehlgeschlagen, Seite ohne Überschriften-Vorgabe getaggt")
        except Exception as e:  # noqa: BLE001
            log.warning("[struktur] Seite %s: Seitenbild/Zuordnung nicht moeglich: %r", s["seite"], e)
            fehlgeschlagen += 1
            hinweise.append(f"Seite {s['seite']}: KI-Zuordnung nicht möglich, Seite ohne Überschriften-Vorgabe getaggt")
        if fortschritt:
            try:
                fortschritt(s["seite"], len(seiten))
            except Exception:  # noqa: BLE001
                pass
    # Zweiter Anlauf fuer Seiten ohne Zuordnung (meist Drosselung des Anbieters, 23.09.2026: Seite 6 im API-Lauf)
    offen = [s for s in seiten if s["seite"] not in zuordnungen and (s["zeilen"] or s["bilder"])]
    if offen and fehlgeschlagen < len(seiten):
        time.sleep(8)
        for s in offen:
            try:
                bild = seitenbild(pdf_in, s["seite"], arbeitsordner)
                zuordnungen[s["seite"]] = zuordnung_je_seite(s, len(seiten), bild, sprache_dokument=sprache.get("lang") or "", dokument_name=dokument_name)
                fehlgeschlagen -= 1
                hinweise = [h for h in hinweise if not h.startswith(f"Seite {s['seite']}: KI-Zuordnung")]
            except Exception as e:  # noqa: BLE001
                log.warning("[struktur] Seite %s: auch der zweite Anlauf schlug fehl: %r", s["seite"], e)
    if fehlgeschlagen and fehlgeschlagen >= len(seiten):
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
    if stat.get("zeilen_ohne_element"):
        hinweise.append(f"{stat['zeilen_ohne_element']} Textzeilen konnten keinem Element zugeordnet werden — bitte in der Hörprobe prüfen")
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
        },
    }
