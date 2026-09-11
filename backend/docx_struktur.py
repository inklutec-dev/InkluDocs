"""Struktur-Lektor fuer Word-Dokumente, Lesestufe (11.09.2026, Steve + Fable 5).

Der Pruefbericht (docx_hoerprobe) sagt, was FORMAL fehlt (Titel, Sprache, Ebenen-
Spruenge, Tabellenkoepfe, Alt-Texte). Dieser Lektor schaut eine Ebene tiefer: Ist die
Struktur, die ein Screenreader bekommt, die Struktur, die ein Sehender SIEHT? Typische
Luecken in Kundendokumenten sind nicht falsche Formatvorlagen, sondern GAR KEINE —
fette, groessere Zeilen als Ueberschriften, mit Bindestrichen getippte Listen,
Leerabsaetze als Abstand, Tabellen als Layout, Grossbuchstaben statt Auszeichnung.

Was hier passiert, ist rein lesend und deterministisch (kein KI-Aufruf): Absatz fuer
Absatz werden Formatvorlage, Fettung, Schriftgroesse, Nummerierung, Tabellenlage und
Text erfasst, dann laufen Heuristiken darueber. Jede Heuristik liefert Absatznummer,
Textanfang, Befund, Vorschlag und eine SICHERHEIT: "hoch" = aus dem XML belegt
(z. B. Liste ohne Nummerierung, die mit Aufzaehlungszeichen beginnt), "mittel" =
Vermutung aus Optik (kurze fette Zeile vor einem langen Absatz). Der Chatbot bekommt
Befunde UND den Absatz-Auszug und darf eigene Beobachtungen ergaenzen — als
Einschaetzung gekennzeichnet, nie als Tatsache.

Umgebaut wird hier NICHTS. Der Umbau (Formatvorlagen zuweisen, Listen anlegen) ist die
naechste Stufe; heute sagt der Lektor dem Nutzer, was er in Word tun kann.
"""
from __future__ import annotations

import re
import zipfile
from typing import Optional

from lxml import etree

from docx_processor import (NS, DocxFehler, _pruefe_zip, _lese_xml, _text, _pstyle, _heading_level, _styles,
                            _dokumenttitel)

W = NS["w"]
MAX_ABSAETZE = 400        # Absatz-Auszug fuer den Chatbot (Tokens!)
MAX_TEXT = 140            # Zeichen je Absatz im Auszug
_AUFZAEHLUNG_RE = re.compile(r"^\s*(?:[-–—•·*▪■○●➢➤►]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+\S")
_ABSCHLUSS_RE = re.compile(r"[.!?…]\s*$")
_URL_RE = re.compile(r"^(https?://|www\.)", re.I)
_LINK_FLOSKELN = {"hier", "klicken sie hier", "click here", "mehr", "link", "weiter", "hier klicken", "here"}


def _wv(el: Optional[etree._Element], attr: str = "val") -> Optional[str]:
    return None if el is None else el.get(f"{{{W}}}{attr}")


def _bool_prop(rpr: Optional[etree._Element], tag: str) -> Optional[bool]:
    """w:b / w:caps in einem rPr: None = nicht gesetzt, True/False = gesetzt (val 0/false = aus)."""
    if rpr is None:
        return None
    el = rpr.find(f"{{{W}}}{tag}")
    if el is None:
        return None
    v = (_wv(el) or "").lower()
    return v not in ("0", "false", "off")


def _sz(rpr: Optional[etree._Element]) -> Optional[float]:
    """Schriftgroesse in Punkt aus w:sz (Halbpunkte)."""
    if rpr is None:
        return None
    v = _wv(rpr.find(f"{{{W}}}sz"))
    try:
        return float(v) / 2.0 if v else None
    except ValueError:
        return None


class _Stile:
    """Formatvorlagen mit Vererbung (basedOn): fett, Groesse, Grossbuchstaben, Gliederungsebene."""

    def __init__(self, zf: zipfile.ZipFile):
        self.roh: dict[str, dict] = {}
        self.standard_sz = 11.0
        self.standard_ids: dict[str, str] = {}
        if "word/styles.xml" not in zf.namelist():
            return
        root = _lese_xml(zf, "word/styles.xml")
        dd = root.find(f"{{{W}}}docDefaults/{{{W}}}rPrDefault/{{{W}}}rPr")
        s = _sz(dd)
        if s:
            self.standard_sz = s
        for st in root.findall(f"{{{W}}}style"):
            sid = st.get(f"{{{W}}}styleId") or ""
            rpr = st.find(f"{{{W}}}rPr")
            ppr = st.find(f"{{{W}}}pPr")
            lvl = _wv(ppr.find(f"{{{W}}}outlineLvl")) if ppr is not None else None
            self.roh[sid] = {
                "typ": st.get(f"{{{W}}}type") or "",
                "basedOn": _wv(st.find(f"{{{W}}}basedOn")) or "",
                "b": _bool_prop(rpr, "b"), "caps": _bool_prop(rpr, "caps"), "sz": _sz(rpr),
                "outline": int(lvl) if (lvl or "").isdigit() else None,
                "default": (st.get(f"{{{W}}}default") or "") in ("1", "true"),
            }
            if self.roh[sid]["default"]:
                self.standard_ids[self.roh[sid]["typ"]] = sid

    def eigenschaft(self, sid: str, key: str):
        gesehen = set()
        while sid and sid not in gesehen and sid in self.roh:
            gesehen.add(sid)
            v = self.roh[sid].get(key)
            if v is not None:
                return v
            sid = self.roh[sid].get("basedOn") or ""
        return None


def _absatz_eigenschaften(p: etree._Element, stile: _Stile) -> dict:
    """Fettung/Groesse/Caps ueber die Textlaeufe eines Absatzes, mit Vererbung aus
    Zeichen- und Absatzvorlage. fett = ALLE Laeufe mit Text fett."""
    ps = _pstyle(p)
    ppr = p.find(f"{{{W}}}pPr")
    ppr_rpr = ppr.find(f"{{{W}}}rPr") if ppr is not None else None   # Absatzmarke, nicht Text
    stil_b = stile.eigenschaft(ps, "b") if ps else None
    stil_caps = stile.eigenschaft(ps, "caps") if ps else None
    stil_sz = stile.eigenschaft(ps, "sz") if ps else None
    std_p = stile.standard_ids.get("paragraph", "")
    if stil_sz is None and std_p:
        stil_sz = stile.eigenschaft(std_p, "sz")
    laeufe = 0
    fette = 0
    caps_laeufe = 0
    max_sz = None
    for r in p.iter(f"{{{W}}}r"):
        if not "".join(t.text or "" for t in r.findall(f"{{{W}}}t")).strip():
            continue
        laeufe += 1
        rpr = r.find(f"{{{W}}}rPr")
        rs = _wv(rpr.find(f"{{{W}}}rStyle")) if rpr is not None else None
        b = _bool_prop(rpr, "b")
        if b is None and rs:
            b = stile.eigenschaft(rs, "b")
        if b is None:
            b = stil_b
        if b:
            fette += 1
        c = _bool_prop(rpr, "caps")
        if c is None and rs:
            c = stile.eigenschaft(rs, "caps")
        if c is None:
            c = stil_caps
        if c:
            caps_laeufe += 1
        sz = _sz(rpr)
        if sz is None and rs:
            sz = stile.eigenschaft(rs, "sz")
        if sz is None:
            sz = stil_sz
        if sz is None:
            sz = stile.standard_sz
        max_sz = sz if max_sz is None else max(max_sz, sz)
    jc = _wv(ppr.find(f"{{{W}}}jc")) if ppr is not None else None
    numpr = ppr.find(f"{{{W}}}numPr") if ppr is not None else None
    ilvl = _wv(numpr.find(f"{{{W}}}ilvl")) if numpr is not None else None
    return {
        "laeufe": laeufe, "fett": laeufe > 0 and fette == laeufe, "teilfett": 0 < fette < laeufe,
        "caps_format": laeufe > 0 and caps_laeufe == laeufe,
        "groesse": max_sz if max_sz is not None else (stil_sz or stile.standard_sz),
        "zentriert": jc == "center", "liste": numpr is not None,
        "listenebene": int(ilvl) if (ilvl or "").isdigit() else (0 if numpr is not None else None),
        "umbrueche": sum(1 for br in p.iter(f"{{{W}}}br") if (br.get(f"{{{W}}}type") or "") != "page"),
        "seitenumbruch": any((br.get(f"{{{W}}}type") or "") == "page" for br in p.iter(f"{{{W}}}br")),
        "bilder": sum(1 for _ in p.iter(f"{{{NS['wp']}}}docPr")),
    }


def _links(p: etree._Element) -> list[str]:
    return [_text(h).strip() for h in p.findall(f".//{{{W}}}hyperlink")]


def _tabelle_info(tbl: etree._Element) -> dict:
    zeilen = tbl.findall(f"{{{W}}}tr")
    spalten = max((len(tr.findall(f"{{{W}}}tc")) for tr in zeilen), default=0)
    kopf = False
    if zeilen:
        trpr = zeilen[0].find(f"{{{W}}}trPr")
        kopf = trpr is not None and trpr.find(f"{{{W}}}tblHeader") is not None
    verschachtelt = tbl.find(f".//{{{W}}}tbl") is not None
    tblpr = tbl.find(f"{{{W}}}tblPr")
    ohne_rahmen = False
    if tblpr is not None:
        borders = tblpr.find(f"{{{W}}}tblBorders")
        if borders is not None:
            werte = [(_wv(b) or "").lower() for b in borders]
            ohne_rahmen = bool(werte) and all(v in ("none", "nil") for v in werte)
    zellen_text = [_text(tc).strip() for tr in zeilen for tc in tr.findall(f"{{{W}}}tc")]
    laengste = max((len(z) for z in zellen_text), default=0)
    return {"zeilen": len(zeilen), "spalten": spalten, "kopf": kopf, "verschachtelt": verschachtelt,
            "ohne_rahmen": ohne_rahmen, "laengste_zelle": laengste,
            "erste_zeile": [_text(tc).strip()[:40] for tc in zeilen[0].findall(f"{{{W}}}tc")] if zeilen else []}


def analysiere_struktur(docx_path: str) -> dict:
    """Liefert {"absaetze": [...], "gliederung": [...], "tabellen": [...], "befunde": [...],
    "zahlen": {...}, "standard_schriftgroesse": pt, "titel": str}."""
    try:
        zf = zipfile.ZipFile(docx_path)
    except zipfile.BadZipFile:
        raise DocxFehler("Keine gültige Word-Datei")
    with zf:
        _pruefe_zip(zf)
        namen = _styles(zf)
        stile = _Stile(zf)
        titel = _dokumenttitel(zf)
        doc = _lese_xml(zf, "word/document.xml")
        body = doc.find(f"{{{W}}}body")
        if body is None:
            raise DocxFehler("Word-Datei ohne Textkörper")

        absaetze: list[dict] = []
        tabellen: list[dict] = []
        nr = 0
        for el in body:
            if el.tag == f"{{{W}}}p":
                nr += 1
                text = _text(el).strip()
                ebene = _heading_level(_pstyle(el), namen)
                if ebene is None:
                    ol = stile.eigenschaft(_pstyle(el), "outline") if _pstyle(el) else None
                    if ol is not None and 0 <= ol <= 8:
                        ebene = ol + 1
                e = _absatz_eigenschaften(el, stile)
                e.update({"nr": nr, "text": text, "woerter": len(text.split()), "stil": _pstyle(el) or "",
                          "stilname": namen.get(_pstyle(el), ""), "ebene": ebene, "in_tabelle": False,
                          "links": _links(el), "beschriftung": bool(re.match(
                              r"^\s*(abbildung|abb\.|bild|grafik|figure|fig\.|tabelle|table)\s*\d+", text, re.I))})
                absaetze.append(e)
            elif el.tag == f"{{{W}}}tbl":
                t = _tabelle_info(el)
                t["nr"] = len(tabellen) + 1
                t["nach_absatz"] = nr
                tabellen.append(t)
                for p in el.iter(f"{{{W}}}p"):
                    text = _text(p).strip()
                    if not text:
                        continue
                    nr += 1
                    e = _absatz_eigenschaften(p, stile)
                    e.update({"nr": nr, "text": text, "woerter": len(text.split()), "stil": _pstyle(p) or "",
                              "stilname": namen.get(_pstyle(p), ""), "ebene": None, "in_tabelle": True,
                              "links": _links(p), "beschriftung": False, "tabelle": t["nr"]})
                    absaetze.append(e)

    std = stile.standard_sz
    befunde: list[dict] = []
    ueberschriften = [a for a in absaetze if a["ebene"] is not None]
    n_ueber = len([a for a in ueberschriften if a["ebene"] and a["text"]])

    def ausz(a: dict, n: int = 80) -> str:
        return a["text"] if len(a["text"]) <= n else a["text"][:n].rstrip() + " …"

    # 1. Fette/grosse kurze Zeilen ohne Ueberschriften-Vorlage vor einem laengeren Absatz
    letzte_ebene = 0
    for i, a in enumerate(absaetze):
        if a["ebene"]:
            letzte_ebene = a["ebene"]
        if a["ebene"] is not None or a["in_tabelle"] or a["liste"] or a["beschriftung"] or not a["text"]:
            continue
        if not (1 <= a["woerter"] <= 12) or _ABSCHLUSS_RE.search(a["text"]) or a["bilder"]:
            continue
        auffaellig = a["fett"] or a["groesse"] >= std + 2 or a["caps_format"] or (a["text"].isupper() and a["woerter"] >= 2)
        if not auffaellig:
            continue
        folgende = next((b for b in absaetze[i + 1:i + 3] if b["text"]), None)
        if folgende is None or folgende["ebene"] is not None:
            continue
        if not (folgende["woerter"] >= 12 or folgende["liste"] or folgende["in_tabelle"]):
            continue
        gruende = []
        if a["fett"]:
            gruende.append("fett")
        if a["groesse"] >= std + 2:
            gruende.append(f"{a['groesse']:g} pt statt {std:g} pt")
        if a["caps_format"] or a["text"].isupper():
            gruende.append("Großbuchstaben")
        vorschlag_ebene = min(max(letzte_ebene + 1, 1), 3) if n_ueber else (1 if a["groesse"] >= std + 4 else 2)
        befunde.append({
            "art": "ueberschrift_ohne_vorlage", "sicherheit": "mittel", "absatz": a["nr"], "text": ausz(a),
            "befund": f"Kurze Zeile ({', '.join(gruende)}) vor einem längeren Absatz, aber ohne Überschriften-Formatvorlage — ein Screenreader liest sie als normalen Text, niemand kann dorthin springen.",
            "vorschlag": f"In Word markieren und die Formatvorlage „Überschrift {vorschlag_ebene}“ zuweisen (Start → Formatvorlagen). Wenn das nur eine Hervorhebung und keine Überschrift ist, so lassen.",
            "vorschlag_ebene": vorschlag_ebene,
        })

    # 2. Getippte Listen: >= 2 aufeinanderfolgende Absaetze mit Aufzaehlungszeichen ohne Nummerierung
    lauf: list[dict] = []

    def _lauf_abschliessen():
        if len(lauf) >= 2:
            befunde.append({
                "art": "getippte_liste", "sicherheit": "hoch", "absatz": lauf[0]["nr"], "bis_absatz": lauf[-1]["nr"],
                "text": ausz(lauf[0], 60),
                "befund": f"{len(lauf)} Absätze beginnen mit einem getippten Aufzählungszeichen oder einer Zahl, sind aber keine Word-Liste — ein Screenreader sagt weder „Liste mit {len(lauf)} Einträgen“ noch die Position.",
                "vorschlag": "Die Absätze markieren und eine echte Aufzählung oder Nummerierung zuweisen (Start → Aufzählungszeichen / Nummerierung); die getippten Zeichen entfallen dann.",
                "anzahl": len(lauf),
            })
        lauf.clear()

    for a in absaetze:
        if a["text"] and not a["liste"] and not a["in_tabelle"] and a["ebene"] is None and _AUFZAEHLUNG_RE.match(a["text"]):
            lauf.append(a)
        else:
            if a["text"] or a["in_tabelle"]:
                _lauf_abschliessen()
    _lauf_abschliessen()

    # 3. Ueberschriften, die wie Absaetze aussehen (sehr lang oder mit Punkt-Satzende bei > 15 Woertern)
    for a in ueberschriften:
        if a["ebene"] and a["text"] and (a["woerter"] > 25 or (a["woerter"] > 15 and _ABSCHLUSS_RE.search(a["text"]))):
            befunde.append({
                "art": "absatz_als_ueberschrift", "sicherheit": "mittel", "absatz": a["nr"], "text": ausz(a),
                "befund": f"Überschrift Ebene {a['ebene']} mit {a['woerter']} Wörtern — das ist eher ein Absatz als eine Überschrift; in der Gliederung wirkt er als Sprungziel mit sehr langem Namen.",
                "vorschlag": "Prüfen, ob der Text als „Standard“-Absatz gehört und die Überschrift auf eine kurze Zeile gekürzt wird.",
            })

    # 4. Leerabsaetze als Abstand (>= 2 hintereinander)
    leer_lauf = 0
    leer_starts: list[int] = []
    for a in absaetze:
        if not a["text"] and not a["bilder"] and not a["in_tabelle"] and not a["seitenumbruch"]:
            leer_lauf += 1
            if leer_lauf == 2:
                leer_starts.append(a["nr"] - 1)
        else:
            leer_lauf = 0
    if leer_starts:
        befunde.append({
            "art": "leerabsaetze", "sicherheit": "hoch", "absatz": leer_starts[0], "text": "",
            "befund": f"An {len(leer_starts)} Stellen stehen mehrere leere Absätze hintereinander (Abstand per Enter) — Screenreader lesen jeden als „leer“.",
            "vorschlag": "Abstände über Absatzformat (Abstand vor/nach) oder Seitenumbruch statt leerer Absätze.",
            "stellen": leer_starts[:20],
        })

    # 5. Grossbuchstaben-Absaetze (getippt, nicht per Format)
    for a in absaetze:
        if a["text"] and a["woerter"] >= 3 and a["text"].isupper() and not a["caps_format"] and a["ebene"] is None:
            befunde.append({
                "art": "grossbuchstaben", "sicherheit": "hoch", "absatz": a["nr"], "text": ausz(a),
                "befund": "Der Absatz ist in Großbuchstaben getippt — manche Screenreader buchstabieren solche Wörter, und die Hervorhebung ist nicht als Struktur erkennbar.",
                "vorschlag": "Normal schreiben und, wenn es eine Überschrift ist, eine Überschriften-Vorlage zuweisen; Optik notfalls über die Schriftoption „Großbuchstaben“.",
            })

    # 6. Manuelle Zeilenumbrueche als Layout (>= 3 im Absatz)
    for a in absaetze:
        if a["umbrueche"] >= 3 and a["text"]:
            befunde.append({
                "art": "manuelle_umbrueche", "sicherheit": "mittel", "absatz": a["nr"], "text": ausz(a),
                "befund": f"{a['umbrueche']} manuelle Zeilenumbrüche in einem Absatz — vermutlich Zeilen, die eigentlich eigene Absätze oder Listenpunkte sind (Adresse, Aufzählung).",
                "vorschlag": "Je Zeile ein Absatz oder eine Liste; Zeilenumbrüche nur innerhalb eines Gedankens.",
            })

    # 7. Linktexte: URL als Text oder Floskel
    for a in absaetze:
        for lt in a["links"]:
            l = lt.strip().lower().rstrip(".")
            if not l:
                continue
            if _URL_RE.match(l) or l in _LINK_FLOSKELN:
                befunde.append({
                    "art": "linktext", "sicherheit": "hoch", "absatz": a["nr"], "text": lt[:80],
                    "befund": "Der Linktext ist die Adresse selbst oder eine Floskel — vorgelesen in einer Linkliste sagt er nicht, wohin er führt.",
                    "vorschlag": "Den Link mit dem Ziel beschriften (zum Beispiel „Antragsformular der Stadt Musterstadt“) statt mit der Adresse oder „hier“.",
                })

    # 8. Tabellen als Layout / verschachtelt
    for t in tabellen:
        if t["verschachtelt"]:
            befunde.append({
                "art": "tabelle_verschachtelt", "sicherheit": "hoch", "absatz": t["nach_absatz"], "text": f"Tabelle {t['nr']}",
                "befund": f"Tabelle {t['nr']} enthält eine weitere Tabelle — Screenreader-Nutzer verlieren darin die Orientierung.",
                "vorschlag": "Verschachtelung auflösen: eine Tabelle je Sachverhalt, Layout über Absätze und Überschriften.",
            })
        elif (t["zeilen"] == 1 or t["spalten"] == 1) and t["laengste_zelle"] > 200:
            befunde.append({
                "art": "layouttabelle", "sicherheit": "mittel", "absatz": t["nach_absatz"], "text": f"Tabelle {t['nr']}",
                "befund": f"Tabelle {t['nr']} hat {t['zeilen']} Zeile(n) und {t['spalten']} Spalte(n) mit langem Fließtext — das sieht nach einer Layout-Tabelle aus, nicht nach Daten.",
                "vorschlag": "Fließtext aus der Tabelle in normale Absätze holen; Tabellen nur für Daten mit Zeilen und Spalten.",
            })
        elif t["ohne_rahmen"] and t["zeilen"] <= 2 and t["laengste_zelle"] > 120:
            befunde.append({
                "art": "layouttabelle", "sicherheit": "mittel", "absatz": t["nach_absatz"], "text": f"Tabelle {t['nr']}",
                "befund": f"Tabelle {t['nr']} ohne Rahmen mit wenigen Zeilen und langem Text — vermutlich Layout (Spaltensatz).",
                "vorschlag": "Prüfen, ob der Inhalt als Absätze oder Spalten (Layout → Spalten) gesetzt werden kann.",
            })

    # 9. Keine Ueberschriften bei laengerem Dokument
    n_text = len([a for a in absaetze if a["text"] and not a["in_tabelle"]])
    if n_ueber == 0 and n_text >= 8:
        befunde.append({
            "art": "keine_ueberschriften", "sicherheit": "hoch", "absatz": 1, "text": "",
            "befund": f"{n_text} Absätze, aber keine einzige Überschrift mit Formatvorlage — die Gliederung existiert nur optisch.",
            "vorschlag": "Die Abschnittstitel mit den Vorlagen „Überschrift 1/2/3“ auszeichnen; die Befunde oben nennen die Kandidaten.",
        })

    befunde.sort(key=lambda b: (b.get("absatz") or 0))
    gliederung = [{"absatz": a["nr"], "ebene": a["ebene"], "text": ausz(a, 100)} for a in ueberschriften if a["text"]]
    auszug = []
    for a in absaetze[:MAX_ABSAETZE]:
        if not a["text"] and not a["bilder"]:
            continue
        eintrag = {"nr": a["nr"], "text": a["text"][:MAX_TEXT] + (" …" if len(a["text"]) > MAX_TEXT else "")}
        if a["ebene"] is not None:
            eintrag["ueberschrift"] = a["ebene"]
        if a["liste"]:
            eintrag["liste"] = a["listenebene"]
        if a["in_tabelle"]:
            eintrag["tabelle"] = a.get("tabelle")
        if a["fett"]:
            eintrag["fett"] = True
        if a["groesse"] and a["groesse"] >= std + 2:
            eintrag["pt"] = a["groesse"]
        if a["bilder"]:
            eintrag["bilder"] = a["bilder"]
        if a["stil"] and a["ebene"] is None and a["stil"].lower() not in ("normal", "standard", ""):
            eintrag["stil"] = a["stilname"] or a["stil"]
        auszug.append(eintrag)
    zahlen = {
        "absaetze": n_text, "ueberschriften": n_ueber, "listenpunkte": len([a for a in absaetze if a["liste"]]),
        "tabellen": len(tabellen), "bilder": sum(a["bilder"] for a in absaetze),
        "befunde": len(befunde), "befunde_sicher": len([b for b in befunde if b["sicherheit"] == "hoch"]),
        "befunde_vermutet": len([b for b in befunde if b["sicherheit"] == "mittel"]),
    }
    return {"titel": titel, "standard_schriftgroesse": std, "gliederung": gliederung, "absaetze": auszug,
            "tabellen": [{k: v for k, v in t.items() if k != "laengste_zelle"} for t in tabellen],
            "befunde": befunde, "zahlen": zahlen, "auszug_gekuerzt": len(absaetze) > MAX_ABSAETZE}
