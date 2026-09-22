# =============================================================================
#  Struktur_Export.py — InkluDocs Strukturlesung einer getaggten PDF (22.09.2026)
# =============================================================================
#  EIGENES Skript von InkluDocs (kein Skript von Joerg Heine; die Regel „Heines Skripte
#  bleiben unangetastet“ gilt hier nicht). Es laeuft den Strukturbaum (Tag-Baum) einer PDF in
#  Lesereihenfolge ab und schreibt je Element eine Zeile in eine JSON-Datei: Kennung (Pfad im
#  Baum), Rolle, Tiefe, Seite, Text, Alt-Text, Ersatztext, Sprache, bei Tabellen Zeilen und
#  Spalten, bei Formularfeldern Feldname und Quickinfo. Das ist die Grundlage fuer die Hoerprobe
#  (was ein Screenreader in welcher Reihenfolge bekommt), die Strukturansicht und die
#  automatische Pruefung (pdf_struktur.py, pdf_pruefung.py).
#
#  Text je Element kommt ueber die MCID-Zuordnung (Tag -> markierter Inhalt), dieselbe Technik
#  wie im Alt-Text-Export (Karbe V1006): eigene MCIDs plus die der Inline-Kinder (Span, Link …),
#  Fragmente mit Silbentrennungs-Heuristik verkettet.
#
#  Aufruf: python3 Struktur_Export.py -i <pdf> -o <struktur.json> [--max-text 600]
#  Exit 0 = geschrieben; 2 = PDF nicht lesbar; 3 = kein Strukturbaum. Lizenz ueber
#  PDFIX_LICENSE_USER/KEY (inkludocs_betrieb.lizenz_aktivieren); ohne Lizenz Testmodus (lesen geht).
# =============================================================================
import argparse
import json
import sys
import time

from pdfixsdk import *  # noqa: F401,F403

import inkludocs_betrieb as betrieb

_INLINE = {"Span", "Link", "Code", "Quote", "Reference", "BibEntry", "Sub", "Em", "Strong", "Annot", "Ruby", "Warichu"}
_CONTAINER = {"Document", "Part", "Art", "Sect", "Div", "NonStruct", "Private", "L", "Table", "TR", "TBody", "THead", "TFoot", "LBody", "TOC", "Aside", "DocumentFragment"}
_KOPPELWOERTER = ("und ", "oder ", "og ", "eller ", "and ", "or ", "&")
# Elemente, die IMMER einen eigenen Eintrag bekommen, auch wenn sie in einer Zelle, einem Listenpunkt
# oder einer Bildunterschrift stecken (Formularfelder in Tabellenzellen sind bei Formularen der Normalfall).
_STOPP = {"L", "Table", "Figure", "Formula", "Form"}

_page_cache: dict = {}
_page_objects_cache: dict = {}
_doc = None
_tree = None


def _sauber(s):
    if not isinstance(s, str):
        return s
    return s.encode("utf-8", "replace").decode("utf-8")


def _get_page(page_num):
    p = _page_cache.get(page_num)
    if p is None:
        p = _doc.AcquirePage(page_num)
        _page_cache[page_num] = p
    return p


def _page_text_objects(page_num):
    objs = _page_objects_cache.get(page_num)
    if objs is None:
        objs = []
        try:
            content = _get_page(page_num).GetContent()
            for dd in range(content.GetNumObjects()):
                obj = content.GetObject(dd)
                if obj.GetObjectType() == kPdsPageText:
                    mcid = obj.GetMcid()
                    if mcid != -1:
                        objs.append((mcid, _sauber(obj.GetText())))   # auch " " (Worttrenner bei Einzelzeichen-PDFs)
        except Exception as e:  # noqa: BLE001
            print(f"WARNUNG: Seite {page_num + 1} nicht lesbar: {e}", file=sys.stderr)
        _page_objects_cache[page_num] = objs
    return objs


def _append_fragment(out, frag, letzte_laenge=0, leerzeichen_objekte=False):
    """Fragmente verketten. Manche Erzeuger (z. B. Browser-Druck) legen jeden Buchstaben als eigenes
    Textobjekt ab und Leerzeichen als eigene Objekte — dann werden Einzelzeichen OHNE Leerzeichen
    angehaengt und die Leerzeichen-Objekte als Worttrenner uebernommen; sonst wie im Alt-Text-Export."""
    if frag == " ":
        return out if out.endswith(" ") or not out else out + " "
    if not out:
        return frag
    if out.endswith("\xad"):
        return out[:-1] + frag
    if out.endswith("-") and frag and not frag.lower().startswith(_KOPPELWOERTER):
        return out[:-1] + frag
    if out.endswith(" "):
        return out + frag
    # Seite mit eigenen Leerzeichen-Objekten (Browser-Druck u. a.): Textobjekte sind Bruchstuecke von
    # Woertern, die Worttrenner kommen als " " — deshalb ohne Leerzeichen anhaengen.
    if leerzeichen_objekte:
        return out + frag
    if len(frag) == 1 and letzte_laenge == 1:
        return out + frag
    return out + " " + frag


def _eigene_mcids(elem, deep=False):
    """[(seite, mcid)] eigener markierter Inhalte + der Inline-Kinder (deep: ganzer Teilbaum)."""
    out = []
    for i in range(elem.GetNumChildren()):
        mcid = elem.GetChildMcid(i)
        if mcid != -1:
            out.append((elem.GetChildPageNumber(i), mcid))
        elif elem.GetChildType(i) == kPdsStructChildElement:
            child = _tree.GetStructElementFromObject(elem.GetChildObject(i))
            if child is None:
                continue
            ctyp = child.GetType(True)
            if deep and ctyp in _STOPP:
                continue   # bekommt einen eigenen Eintrag
            if deep or ctyp in _INLINE:
                out.extend(_eigene_mcids(child, deep))
    return out


def _text(elem, deep=False, max_text=600):
    paare = _eigene_mcids(elem, deep)
    if not paare:
        return "", -1
    seiten = sorted({p for p, _m in paare if p >= 0})
    out = ""
    letzte = 0
    for p in seiten:
        wanted = {m for pp, m in paare if pp == p}
        objekte = _page_text_objects(p)
        leer = any(t == " " for _m, t in objekte)
        for mcid, text in objekte:
            if mcid in wanted and text:
                out = _append_fragment(out, text, letzte, leer)
                letzte = len(text)
    out = " ".join(out.replace("\xad", "").split())
    if max_text and len(out) > max_text:
        out = out[:max_text].rstrip() + " …"
    return out, (seiten[0] if seiten else -1)


def _feld_info(elem):
    """Formularfeld: Feldname (/T, ggf. vom Elternfeld) und Quickinfo (/TU) aus der Anmerkung des
    Form-Elements. Das Kindobjekt ist ein OBJR-Woerterbuch {/Type /OBJR /Obj <Anmerkung>}."""
    for i in range(elem.GetNumChildren()):
        if elem.GetChildType(i) != kPdsStructChildObject:
            continue
        try:
            d = elem.GetChildObject(i)
            if d is None:
                continue
            if d.GetObjectType() != kPdsDictionary:
                continue
            d = PdsDictionary(d.obj)
            annot = d.GetDictionary("Obj") if d.Get("Obj") is not None else d
            if annot is None:
                continue
            name, tu = "", ""
            knoten, tiefe = annot, 0
            # Feldname/Quickinfo stehen bei Optionsfeldern (Kids) am Elternfeld -> Parent-Kette hinauf
            while knoten is not None and tiefe < 8:
                if not name:
                    name = _sauber(knoten.GetText("T") or "")
                if not tu:
                    tu = _sauber(knoten.GetText("TU") or "")
                if name and tu:
                    break
                knoten = knoten.GetDictionary("Parent") if knoten.Get("Parent") is not None else None
                tiefe += 1
            return name or "", tu or ""
        except Exception:  # noqa: BLE001
            return "", ""
    return "", ""


def _walk(elem, tiefe, pfad, out, max_text, zaehler, im_text=False):
    """im_text: der Text dieses Teilbaums steckt schon im Eintrag eines Vorfahren (Zelle, Listenpunkt …);
    dann bekommen nur _STOPP-Elemente (Felder, Grafiken, Tabellen, Listen) eigene Eintraege."""
    typ = elem.GetType(True) or "?"
    zaehler[0] += 1
    if zaehler[0] > 50000:
        return
    if im_text and typ not in _STOPP:
        n = 0
        for i in range(elem.GetNumChildren()):
            if elem.GetChildType(i) == kPdsStructChildElement:
                child = _tree.GetStructElementFromObject(elem.GetChildObject(i))
                if child is not None:
                    _walk(child, tiefe + 1, f"{pfad}.{n}", out, max_text, zaehler, True)
                    n += 1
        return
    eintrag = {"id": pfad, "typ": typ, "tiefe": tiefe}
    deep = typ in ("TH", "TD", "LI", "Lbl", "LBody", "Caption", "Note", "TOCI", "Formula", "Figure", "Form") and typ not in _CONTAINER
    text, seite = _text(elem, deep=deep, max_text=max_text)
    if typ in _CONTAINER and typ not in ("LBody",):
        text, seite = "", seite
    eintrag["text"] = text
    eintrag["seite"] = (seite + 1) if seite >= 0 else 0
    try:
        alt = _sauber(elem.GetAlt() or "")
        act = _sauber(elem.GetActualText() or "")
        lang = _sauber(elem.GetLang() or "")
        if alt:
            eintrag["alt"] = alt
        if act:
            eintrag["actual"] = act
        if lang:
            eintrag["lang"] = lang
    except Exception:  # noqa: BLE001
        pass
    if typ == "Table":
        try:
            eintrag["zeilen"] = elem.GetNumRow()
            eintrag["spalten"] = elem.GetNumCol()
        except Exception:  # noqa: BLE001
            pass
    if typ == "Form":
        name, tu = _feld_info(elem)
        if name:
            eintrag["feldname"] = name
        if tu:
            eintrag["quickinfo"] = tu
    kinder = 0
    for i in range(elem.GetNumChildren()):
        if elem.GetChildType(i) == kPdsStructChildElement:
            kinder += 1
    eintrag["kinder"] = kinder
    out.append(eintrag)
    n = 0
    for i in range(elem.GetNumChildren()):
        if elem.GetChildType(i) == kPdsStructChildElement:
            child = _tree.GetStructElementFromObject(elem.GetChildObject(i))
            if child is None:
                continue
            ctyp = child.GetType(True) or ""
            if not deep and ctyp in _INLINE and typ not in _CONTAINER:
                continue   # Inline-Text steckt schon im Text des Elternelements
            # deep: Text der Nachfahren steckt schon in diesem Eintrag -> nur noch Felder/Grafiken/Tabellen/Listen
            _walk(child, tiefe + 1, f"{pfad}.{n}", out, max_text, zaehler, im_text=deep)
            n += 1


def main():
    global _doc, _tree
    ap = argparse.ArgumentParser(description="Strukturlesung einer getaggten PDF (InkluDocs)")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--max-text", type=int, default=600)
    args = ap.parse_args()
    t0 = time.time()
    pdfix = GetPdfix()
    if pdfix is None:
        raise SystemExit("Pdfix Initialization fail")
    betrieb.lizenz_aktivieren(pdfix)
    _doc = pdfix.OpenDoc(args.input, "")
    if _doc is None:
        sys.exit(betrieb.pdf_nicht_geoeffnet(pdfix))
    try:
        _tree = _doc.GetStructTree()
        if _tree is None or _tree.GetNumChildren() == 0:
            print("Kein Strukturbaum", file=sys.stderr)
            sys.exit(3)
        out: list = []
        zaehler = [0]
        for i in range(_tree.GetNumChildren()):
            elem = _tree.GetStructElementFromObject(_tree.GetChildObject(i))
            if elem is not None:
                _walk(elem, 0, str(i), out, args.max_text, zaehler)
        # Container ohne eigene Seite: Seite des ersten Nachfolgers mit Seite
        for idx, e in enumerate(out):
            if e["seite"] == 0:
                for f in out[idx + 1:]:
                    if f["id"].startswith(e["id"] + ".") and f["seite"]:
                        e["seite"] = f["seite"]
                        break
                    if not f["id"].startswith(e["id"] + "."):
                        break
        # Tabellen: liefert das SDK keine Zeilen/Spalten (-1), aus TR und TH/TD nachzaehlen
        for idx, e in enumerate(out):
            if e["typ"] != "Table" or (e.get("zeilen") or -1) >= 0:
                continue
            nach = []
            for f in out[idx + 1:]:
                if not f["id"].startswith(e["id"] + "."):
                    break
                nach.append(f)
            zeilen = [f for f in nach if f["typ"] == "TR"]
            spalten = 0
            for z in zeilen:
                t = z["id"].count(".") + 1
                spalten = max(spalten, sum(1 for f in nach if f["id"].startswith(z["id"] + ".") and f["id"].count(".") == t and f["typ"] in ("TH", "TD")))
            e["zeilen"], e["spalten"] = len(zeilen), spalten
        info = {"seiten": _doc.GetNumPages(), "elemente": len(out), "dauer_s": round(time.time() - t0, 2)}
        try:
            info["lang"] = _sauber(_doc.GetLang() or "")
        except Exception:  # noqa: BLE001
            pass
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"info": info, "elemente": out}, f, ensure_ascii=False)
        print(f"STRUKTUR_ELEMENTE={len(out)}")
    finally:
        for p in _page_cache.values():
            try:
                p.Release()
            except Exception:  # noqa: BLE001
                pass
        _doc.Close()


if __name__ == "__main__":
    main()
