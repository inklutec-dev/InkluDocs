# =============================================================================
#  Struktur_Schreiben.py — InkluDocs: Tag-Baum aus UNSERER Vorgabe schreiben (23.09.2026)
# =============================================================================
#  EIGENES Skript von InkluDocs (kein Skript von Joerg Heine). Weg „Struktur zuerst“ (Steves Go 23.09.2026):
#  Die Struktur der Seite kommt von aussen (pdf_struktur_tagging: Messung + Modell-Zuordnung), PDFix schreibt
#  daraus den Strukturbaum in DIESELBE Datei. Der Seiteninhalt bleibt byteweise gleich; nur der Baum ist neu.
#
#  Was das Skript je Seite tut (Reihenfolge ist wichtig, Erkenntnisse vom Schreibtest 23.09.2026):
#    1. grosse Form-XObjects (Hintergrund, > anteil der Seitenflaeche) von der Erkennung AUSSCHLIESSEN und im
#       Inhalt als Artefakt markieren — sonst haengt PDFix den Text darauf als Kinder in ein Bild (Seite 3 des
#       Ritterturniers: ganzer Text wurde zum Alt-Text einer Figure);
#    2. Tabellenerkennung ueber die Vorlage (Template) abschalten, wenn die Seite laut Vorgabe keine Tabelle hat;
#    3. Artefakt-Zeilen (Kolumnentitel, Seitenzahl, Verlagszeile) und Schmuckbilder als INITIALE Elemente mit
#       kElemInitial|kElemArtifact vorgeben — nur so werden sie Artefakt (SetFlags nach CreateElements wirkt nicht);
#    4. CreateElements (PDFix-Layout: Zeilen, Absaetze, Listen, Tabellen, Bilder);
#    5. Rollen setzen: Textelement, dessen Rahmen die Mitte einer Vorgabezeile enthaelt, bekommt SetTag(H1..H6/Caption);
#       Bilder: SetTag("Figure") + SetAlt (nur wenn vorgegeben; leer bleibt leer, der Export traegt Alt-Texte nach);
#       Tabellen: erste Spalte als Kopfzellen;
#    6. Vollstaendigkeit: jede Vorgabezeile, die kein Textelement trifft, wird gezaehlt (zeilen_ohne_element);
#    7. AddTags in das Document-Element.
#
#  Aufruf: python3 Struktur_Schreiben.py -i <pdf> -o <pdf_out> -k <plan.json>
#     plan.json: {"sprache": "de-DE", "hintergrund_anteil": 0.6, "seiten": [
#        {"seite": 1, "artefakte": [[l,b,r,t], ...], "rollen": [{"bbox": [l,b,r,t], "tag": "H2"}, ...],
#         "bilder": [{"bbox": [l,b,r,t], "alt": "", "artefakt": false}, ...], "tabellen": true,
#         "zeilen": [[l,b,r,t], ...]}]}   (PDF-Koordinaten, Ursprung unten links)
#  Ausgabe (stdout, letzte Zeile): JSON je Seite {seite, hintergrund, artefakte, rollen, bilder, tabellen,
#     zeilen_ohne_element} + Summen.  Exit 0 = geschrieben; 2 = PDF nicht lesbar; 4 = Speichern fehlgeschlagen.
# =============================================================================
import argparse
import ctypes
import copy
import json
import sys

from pdfixsdk import *  # noqa: F401,F403

import inkludocs_betrieb as betrieb

_ERLAUBT = {"H1", "H2", "H3", "H4", "H5", "H6", "Caption", "P"}


def _rect(l, b, r, t):
    x = PdfRect()
    x.left, x.bottom, x.right, x.top = float(l), float(b), float(r), float(t)
    return x


def _mitte_drin(bb, box, rand=2.0):
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return bb.left - rand <= cx <= bb.right + rand and bb.bottom - rand <= cy <= bb.top + rand


def _walk(el, fn):
    fn(el)
    for i in range(el.GetNumChildren()):
        c = el.GetChild(i)
        if c and c.GetType() not in (kPdeWord, kPdeTextRun, kPdeTextLine):
            _walk(c, fn)


def _daten(b: bytearray):
    return (ctypes.c_ubyte * len(b)).from_buffer(b)


def _template_lesen(pdfix, doc) -> dict:
    s = pdfix.CreateMemStream()
    try:
        doc.GetTemplate().SaveToStream(s, kDataFormatJson, 0)
        n = s.GetSize()
        raw = bytearray(n)
        s.Read(0, _daten(raw), n)
        return json.loads(bytes(raw).decode("utf-8", "replace"))
    finally:
        s.Destroy()


def _template_setzen(pdfix, doc, vorlage: dict, aenderungen: dict) -> bool:
    j = copy.deepcopy(vorlage)
    try:
        for k, v in aenderungen.items():
            j["template"]["pagemap"][0][k] = v
    except (KeyError, IndexError, TypeError):
        return False
    data = json.dumps(j).encode()
    s = pdfix.CreateMemStream()
    try:
        s.Write(0, _daten(bytearray(data)), len(data))
        t = doc.GetTemplate()
        return bool(t.LoadFromStream(s, kDataFormatJson) and t.Update())
    finally:
        s.Destroy()


def main():
    ap = argparse.ArgumentParser(description="Tag-Baum aus Vorgabe schreiben (InkluDocs)")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-k", "--plan", required=True)
    args = ap.parse_args()
    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)
    pdfix = GetPdfix()
    if pdfix is None:
        raise SystemExit("Pdfix Initialization fail")
    betrieb.lizenz_fuer_tagging(pdfix)
    doc = pdfix.OpenDoc(args.input, "")
    if doc is None:
        sys.exit(betrieb.pdf_nicht_geoeffnet(pdfix))
    anteil = float(plan.get("hintergrund_anteil") or 0.6)
    ergebnis = {"seiten": [], "rollen": 0, "artefakte": 0, "bilder": 0, "tabellen": 0, "zeilen_ohne_element": 0, "hintergrund": 0}
    try:
        doc.RemoveTags()
        tree = doc.CreateStructTree()
        if tree is None:
            print("Strukturbaum konnte nicht angelegt werden", file=sys.stderr)
            sys.exit(4)
        doc_el = tree.AddNewChild("Document", 0)
        vorlage = _template_lesen(pdfix, doc)
        je_seite = {int(s.get("seite") or 0): s for s in plan.get("seiten") or []}
        for pno in range(doc.GetNumPages()):
            vorgabe = je_seite.get(pno + 1, {})
            page = doc.AcquirePage(pno)
            if page is None:
                continue
            crop = page.GetCropBox()
            breite, hoehe = crop.right - crop.left, crop.top - crop.bottom
            stat = {"seite": pno + 1, "hintergrund": 0, "artefakte": 0, "rollen": 0, "bilder": 0, "tabellen": 0, "zeilen_ohne_element": 0}
            # 1. Hintergrund-Formulare ausschliessen + Artefakt-Marke
            content = page.GetContent()
            for i in range(content.GetNumObjects()):
                o = content.GetObject(i)
                if o.GetObjectType() != kPdsPageForm:
                    continue
                bb = o.GetBBox()
                if (bb.right - bb.left) * (bb.top - bb.bottom) > anteil * breite * hoehe:
                    o.SetStateFlags(kStateExclude)
                    d = doc.CreateDictObject(False)
                    d.PutName("Type", "Layout")
                    cm = o.GetContentMark()
                    if cm is not None:
                        cm.AddTag("Artifact", d, False)   # NIE mit None als Objekt (Absturz)
                    stat["hintergrund"] += 1
            # 2. Tabellenerkennung je Seite
            _template_setzen(pdfix, doc, vorlage, {} if vorgabe.get("tabellen", True) else {"text_table_detect": "0", "graphic_table_detect": "0"})
            pm = page.AcquirePageMap()
            # 3. initiale Artefakte
            for box in vorgabe.get("artefakte") or []:
                e = pm.CreateElement(kPdeText, None)
                if e:
                    e.SetBBox(_rect(box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3))
                    e.SetFlags(kElemInitial | kElemArtifact)
                    stat["artefakte"] += 1
            for b in vorgabe.get("bilder") or []:
                if b.get("artefakt"):
                    e = pm.CreateElement(kPdeImage, None)
                    if e:
                        e.SetBBox(_rect(*b["bbox"]))
                        e.SetFlags(kElemInitial | kElemArtifact)
                        stat["artefakte"] += 1
            # 4. Layout
            pm.CreateElements()
            # 5. Rollen, Bilder, Tabellen; 6. Vollstaendigkeit
            rollen = [r for r in (vorgabe.get("rollen") or []) if r.get("tag") in _ERLAUBT]
            bilder = [b for b in (vorgabe.get("bilder") or []) if not b.get("artefakt")]
            zeilen = list(vorgabe.get("zeilen") or [])
            getroffen = [False] * len(zeilen)

            def bearbeiten(el):
                t = el.GetType()
                bb = el.GetBBox()
                if t == kPdeText:
                    for r in rollen:
                        if _mitte_drin(bb, r["bbox"]):
                            if el.SetTag(r["tag"]):
                                stat["rollen"] += 1
                            break
                    for i, z in enumerate(zeilen):
                        if not getroffen[i] and _mitte_drin(bb, z):
                            getroffen[i] = True
                elif t == kPdeImage and el.GetNumChildren() == 0:
                    alt = ""
                    for b in bilder:
                        if _mitte_drin(bb, b["bbox"], rand=6.0) and (b.get("alt") or "").strip():
                            alt = b["alt"].strip()
                            break
                    el.SetTag("Figure")
                    if alt:
                        el.SetAlt(alt)
                    stat["bilder"] += 1
                elif t == kPdeTable:
                    tb = PdeTable(el.obj)
                    stat["tabellen"] += 1
                    for rr in range(tb.GetNumRows()):
                        for cc in range(tb.GetNumCols()):
                            cell = tb.GetCell(rr, cc)
                            if cell:
                                PdeCell(cell.obj).SetHeader(cc == 0)

            root = pm.GetElement()
            if root:
                _walk(root, bearbeiten)
            stat["zeilen_ohne_element"] = sum(1 for g in getroffen if not g)
            # 7. Tags schreiben
            pm.AddTags(doc_el, False, PdfTagsParams())
            pm.Release()
            page.Release()
            ergebnis["seiten"].append(stat)
            for k in ("hintergrund", "artefakte", "rollen", "bilder", "tabellen", "zeilen_ohne_element"):
                ergebnis[k] += stat[k]
        if plan.get("sprache"):
            doc.SetLang(str(plan["sprache"]))
        if not doc.Save(args.output, kSaveFull):
            print("Speichern fehlgeschlagen", file=sys.stderr)
            sys.exit(4)
        print(json.dumps(ergebnis, ensure_ascii=False))
    finally:
        doc.Close()


if __name__ == "__main__":
    main()
