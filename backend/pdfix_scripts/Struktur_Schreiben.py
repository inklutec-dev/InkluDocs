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
#    7. AddTags in das Document-Element;
#    9. TABELLENZELLEN AUS UNSEREN ZEILEN (23.09.2026, Tabellentest Rechnung): Pass 1 laesst PDFix die Tabellen-
#       RAHMEN finden; die ZELLEN kommen dann aus den Zeilen des Struktur-HTML (Zeilenbaender x Spaltencluster) und
#       werden als initiale kPdeTable/kPdeCell vorgegeben — zusammengezogene Zellen (IBAN+BIC+Verwendungszweck in
#       einer Zelle) sind damit getrennt. Kopfzellen: wie PDFix in Pass 1, sonst Kopfzeile ab 3 Spalten, sonst Kopfspalte.
#    8. BILDER JE OBJEKT (23.09.2026, Bildtest): Bildobjekte, die im Plan stehen, werden VOR der Erkennung
#       ausgeschlossen (sonst haengt PDFix Text darauf ins Bild und verschmilzt Nachbarbilder) und NACH AddTags
#       je Objekt als eigene Figure (AddNewChild + AddPageObject) in Lesereihenfolge eingehaengt; Schmuckbilder
#       werden im Inhalt als Artefakt markiert. Alt bleibt leer (Export/Pipeline).
#
#  Aufruf: python3 Struktur_Schreiben.py -i <pdf> -o <pdf_out> -k <plan.json>
#     plan.json (Listen: "listen": [[{"bboxes": [[l,b,r,t], ...]}, ...], ...] — je Liste die Punkte, je Punkt seine
#       Zeilen; sie werden VOR CreateElements als kPdeList mit kPdeText-Kindern (kElemInitial|kElemNoSplit) vorgegeben,
#       so bleibt ein umbrochener Punkt EIN Listeneintrag, Listentest 23.09.2026):
#                {"sprache": "de-DE", "hintergrund_anteil": 0.6, "seiten": [
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


def _zellen_aus_zeilen(tb, zeilen):
    """(zeilen_n, spalten_n, {(r, c): [bbox...]}) fuer Zeilen-Rahmen innerhalb des Tabellenrahmens tb=(l,b,r,t)."""
    drin = [z for z in zeilen if tb[0] - 2 <= z[0] and z[2] <= tb[2] + 2 and tb[1] - 2 <= z[1] and z[3] <= tb[3] + 2]
    if len(drin) < 2:
        return 0, 0, {}
    drin.sort(key=lambda z: (-z[3], z[0]))
    baender = []
    for z in drin:
        if baender and abs(baender[-1][0] - z[3]) < 4:
            baender[-1][1].append(z)
        else:
            baender.append([z[3], [z]])
    spalten = []
    for x in sorted(set(round(z[0]) for z in drin)):
        if spalten and x - spalten[-1][-1] < 15:
            spalten[-1].append(x)
        else:
            spalten.append([x])
    sx = [s[0] for s in spalten]

    def _spalte(z):
        return max(i for i, x in enumerate(sx) if z[0] >= x - 2)

    # Ein Zeilenband ohne Eintrag in der ERSTEN Spalte setzt die Zelle(n) der Zeile darueber fort
    # (umbrochene Beschreibung in einer Rechnungsposition) — es wird keine eigene Tabellenzeile.
    reihen = []
    for _top, zs in baender:
        if reihen and not any(_spalte(z) == 0 for z in zs):
            reihen[-1].extend(zs)
        else:
            reihen.append(list(zs))
    zellen = {}
    for r, zs in enumerate(reihen):
        for z in zs:
            zellen.setdefault((r, _spalte(z)), []).append(z)
    return len(reihen), len(sx), zellen


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
            # 9. Pass 1 (eigene Seitenkarte): Tabellenrahmen + Kopfzellen von PDFix; die Zellen kommen aus unseren
            #    Zeilen. Ein „Tabellen“-Kandidat, dessen Zeilen meist nur EINE Zelle haben (Formular, Beschriftungen mit
            #    Feldern), ist keine Tabelle: dann wird die Tabellenerkennung fuer diese Seite abgeschaltet.
            tabellen_vorgabe = []
            echte, unechte = 0, 0
            if vorgabe.get("tabellen", True) and vorgabe.get("zeilen"):
                _template_setzen(pdfix, doc, vorlage, {})
                pm1 = page.AcquirePageMap()
                pm1.CreateElements()
                kandidaten = []

                def _sammeln(el):
                    if el.GetType() == kPdeTable:
                        bb = el.GetBBox()
                        tb = PdeTable(el.obj)
                        koepfe = set()
                        for rr in range(tb.GetNumRows()):
                            for cc in range(tb.GetNumCols()):
                                cell = tb.GetCell(rr, cc)
                                if cell and PdeCell(cell.obj).GetHeader():
                                    koepfe.add((rr, cc))
                        kandidaten.append(((bb.left, bb.bottom, bb.right, bb.top), tb.GetNumRows(), tb.GetNumCols(), koepfe))
                    for i in range(el.GetNumChildren()):
                        c2 = el.GetChild(i)
                        if c2 and c2.GetType() not in (kPdeWord, kPdeTextRun, kPdeTextLine):
                            _sammeln(c2)

                root1 = pm1.GetElement()
                if root1:
                    _sammeln(root1)
                pm1.RemoveElements()
                pm1.Release()
                # Seite neu holen: die Vorlage (Template) wirkt nur auf eine frisch angelegte Seitenkarte
                page.Release()
                page = doc.AcquirePage(pno)
                felder = [tuple(f) for f in (vorgabe.get("felder") or [])]
                struktur_boxen = [r["bbox"] for r in (vorgabe.get("rollen") or [])] + [bb for liste in (vorgabe.get("listen") or []) for pkt in liste for bb in (pkt.get("bboxes") or [])]

                def _drin(tb, box):
                    return tb[0] - 2 <= (box[0] + box[2]) / 2 <= tb[2] + 2 and tb[1] - 2 <= (box[1] + box[3]) / 2 <= tb[3] + 2

                for (tb, r1, c1, koepfe) in kandidaten:
                    # Formularfelder im Kandidaten: ein Formular ist keine Tabelle
                    if sum(1 for f in felder if _drin(tb, f)) >= 2:
                        unechte += 1
                        continue
                    # Ueberschriften oder Listenpunkte im Kandidaten (unsere eigene Struktur): zwei Aufzaehlungsspalten
                    # nebeneinander sind keine Tabelle (Ritterturnier S. 2, 23.09.2026)
                    if any(_drin(tb, box) for box in struktur_boxen):
                        unechte += 1
                        continue
                    nr, nc, zellen = _zellen_aus_zeilen(tb, vorgabe.get("zeilen") or [])
                    if nr < 2 or nc < 2:
                        unechte += 1
                        continue
                    volle = sum(1 for rr in range(nr) if sum(1 for cc in range(nc) if (rr, cc) in zellen) >= 2)
                    if volle < 0.5 * nr:
                        unechte += 1
                        continue
                    echte += 1
                    tabellen_vorgabe.append((tb, r1, c1, koepfe, nr, nc, zellen))
            # 1. Hintergrund-Formulare ausschliessen + Artefakt-Marke; Bildobjekte aus dem Plan ausschliessen
            content = page.GetContent()
            plan_bilder = list(vorgabe.get("bilder") or [])
            eigene_bilder = []   # (PdsPageObject, bbox) -> nach AddTags als Figure
            for i in range(content.GetNumObjects()):
                o = content.GetObject(i)
                typ = o.GetObjectType()
                bb = o.GetBBox()
                if typ == kPdsPageForm:
                    if (bb.right - bb.left) * (bb.top - bb.bottom) > anteil * breite * hoehe:
                        o.SetStateFlags(kStateExclude)
                        d = doc.CreateDictObject(False)
                        d.PutName("Type", "Layout")
                        cm = o.GetContentMark()
                        if cm is not None:
                            cm.AddTag("Artifact", d, False)   # NIE mit None als Objekt (Absturz)
                        stat["hintergrund"] += 1
                elif typ == kPdsPageImage and (bb.right - bb.left) > 20 and (bb.top - bb.bottom) > 20:
                    treffer = None
                    for b in plan_bilder:
                        if _mitte_drin(bb, b["bbox"], rand=6.0):
                            treffer = b
                            break
                    if treffer is None:
                        continue   # nicht im Plan (z. B. Winzling): PDFix entscheidet
                    o.SetStateFlags(kStateExclude)
                    if treffer.get("artefakt"):
                        d = doc.CreateDictObject(False)
                        d.PutName("Type", "Layout")
                        cm = o.GetContentMark()
                        if cm is not None:
                            cm.AddTag("Artifact", d, False)
                        stat["artefakte"] += 1
                    else:
                        eigene_bilder.append((o, (bb.left, bb.bottom, bb.right, bb.top)))
            # 2. Tabellenerkennung je Seite: aus, wenn das Modell keine Tabelle sieht oder nur unechte Kandidaten da sind
            keine_tabellen = (not vorgabe.get("tabellen", True)) or (unechte > 0 and echte == 0)
            _template_setzen(pdfix, doc, vorlage, {"text_table_detect": "0", "graphic_table_detect": "0", "form_table_detect": "0"} if keine_tabellen else {})
            if keine_tabellen and unechte:
                stat["tabellen_verworfen"] = unechte
            pm = page.AcquirePageMap()
            for (tb, r1, c1, koepfe, nr, nc, zellen) in tabellen_vorgabe:
                T = pm.CreateElement(kPdeTable, None)
                if not T:
                    continue
                T.SetBBox(_rect(*tb))
                T.SetFlags(kElemInitial)
                PdeTable(T.obj).SetNumRows(nr)
                PdeTable(T.obj).SetNumCols(nc)
                gleich = (nr == r1 and nc == c1)
                for (rr, cc), zs in sorted(zellen.items()):
                    cell = pm.CreateElement(kPdeCell, T)
                    if not cell:
                        continue
                    cell.SetBBox(_rect(min(z[0] for z in zs), min(z[1] for z in zs), max(z[2] for z in zs), max(z[3] for z in zs)))
                    cell.SetFlags(kElemInitial | kElemNoSplit)
                    pc = PdeCell(cell.obj)
                    pc.SetRowNum(rr)
                    pc.SetColNum(cc)
                    pc.SetHeader((rr, cc) in koepfe if gleich and koepfe else (rr == 0 if nc >= 3 else cc == 0))
                stat["tabellen_vorgegeben"] = stat.get("tabellen_vorgegeben", 0) + 1
            # 3. initiale Artefakte
            for box in vorgabe.get("artefakte") or []:
                e = pm.CreateElement(kPdeText, None)
                if e:
                    e.SetBBox(_rect(box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3))
                    e.SetFlags(kElemInitial | kElemArtifact)
                    stat["artefakte"] += 1
            # (Schmuckbilder sind oben bereits als Artefakt im Inhalt markiert)
            # 3b. Listen vorgeben (Aufzaehlungszeichen + Einzug aus dem Struktur-HTML)
            for liste in vorgabe.get("listen") or []:
                alle = [bb for pkt in liste for bb in (pkt.get("bboxes") or [])]
                if not alle:
                    continue
                L = pm.CreateElement(kPdeList, None)
                if not L:
                    continue
                L.SetBBox(_rect(min(b[0] for b in alle), min(b[1] for b in alle), max(b[2] for b in alle), max(b[3] for b in alle)))
                L.SetFlags(kElemInitial)
                for pkt in liste:
                    bbs = pkt.get("bboxes") or []
                    if not bbs:
                        continue
                    e = pm.CreateElement(kPdeText, L)
                    if e:
                        e.SetBBox(_rect(min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs)))
                        e.SetFlags(kElemInitial | kElemNoSplit)
                        stat["listenpunkte"] = stat.get("listenpunkte", 0) + 1
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
                    if not tabellen_vorgabe:   # ohne Vorgabe: Kopfspalte als Rueckfall
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
            # 8. Bilder je Objekt als eigene Figure in Lesereihenfolge (vor dem ersten Kind, das unter der Bildoberkante beginnt)
            for (o, bb) in sorted(eigene_bilder, key=lambda x: (-x[1][3], x[1][0])):
                idx = doc_el.GetNumChildren()
                for k in range(doc_el.GetNumChildren()):
                    if doc_el.GetChildType(k) != kPdsStructChildElement:
                        continue
                    kind = tree.GetStructElementFromObject(doc_el.GetChildObject(k))
                    if kind is None:
                        continue
                    try:
                        seite_kind = kind.GetPageNumber(0)
                    except Exception:  # noqa: BLE001
                        seite_kind = -1
                    if seite_kind != pno:
                        if seite_kind > pno:
                            idx = k
                            break
                        continue
                    try:
                        cb = kind.GetBBox(pno)
                    except Exception:  # noqa: BLE001
                        continue
                    if cb is not None and cb.top < bb[3] - 1.0:
                        idx = k
                        break
                fig = doc_el.AddNewChild("Figure", idx)
                if fig is not None and fig.AddPageObject(o, -1):
                    stat["bilder"] += 1
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
