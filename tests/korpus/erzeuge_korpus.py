"""Prüfkorpus erzeugen (15.09.2026, Steve: „echte Qualität auch ohne PDFix“).

Eigene Dokumente, keine Kundendateien. Laeuft im Staging-Container:
    docker exec -w /app inkludocs-staging python3 tests/korpus/erzeuge_korpus.py tests/korpus/dokumente

Dokumente:
  01_lo_testdokument.pdf   Word (tests/fixtures/testdokument_inkludocs.docx) -> PDF/UA ueber den Konverter (LibreOffice, getaggt)
  02_lo_diagramm.pdf       Word mit Diagramm -> Konverter
  03_synth_satz.pdf        Satzprogramm-Stand-in (bis Michaels InDesign-Datei da ist): Figures mit MCIDs in
                           vorhandenen Tags, /K als Referenz auf Array-Objekt, RoleMap PlacedGraphic, Collage
                           (zwei Bilder in einem Figure), ein ungetaggtes Bild, ParentTree, Doppelseiten-Reihenfolge,
                           dasselbe Logo (gleiches XObject) auf zwei Seiten
  04_untagged_raster.pdf   drei Seiten Text + Rasterbilder, kein Strukturbaum (wie ReportLab-Ausgaben)
  05_scan.pdf              nur seitenfuellende Bilder, kein Text, keine Tags
  06_vektor_layout.pdf     ungetaggt: seitengrosser Rahmen mit Foto (Layout, kein Bild) + kleines Balkendiagramm (Bild)
  07_formular.pdf          tests/fixtures/testformular_inkludocs.pdf (Formular, keine Bilder)
"""
import os
import shutil
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(os.path.dirname(HERE)), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)
FIX = None
for k in ("/app/tests/fixtures", os.path.join(os.path.dirname(HERE), "fixtures")):
    if os.path.isdir(k):
        FIX = k
        break

FARBEN = [(200, 30, 30), (30, 30, 200), (30, 160, 30), (220, 160, 20), (120, 40, 160), (20, 160, 160)]
_ZAEHLER = [0]   # jede Platzierung ein EIGENES Bild (sonst dedupliziert PyMuPDF auf ein XObject) — ausser Logo


def _pix(i, w=24, h=24):
    """Eindeutiges Bild je Aufruf: Farbe plus eine Markierung, die sich mit jedem Bild aendert."""
    p = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, h), 0)
    p.set_rect(p.irect, FARBEN[i % len(FARBEN)])
    _ZAEHLER[0] += 1
    n = _ZAEHLER[0]
    p.set_rect(fitz.IRect(0, 0, 1 + n % (w - 1), 1 + (n // 7) % (h - 1)), (255, 255, 255))
    return p


_LOGO = [None]


def _logo():
    """Dasselbe Logo auf jeder Seite = dasselbe XObject (gleicher xref) — Korpus-Fall „wiederholtes Bild“."""
    if _LOGO[0] is None:
        p = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), 0); p.set_rect(p.irect, (10, 10, 10)); _LOGO[0] = p
    return _LOGO[0]


def _bild_op(page, rect, i, logo=False):
    """Bild als XObject registrieren; liefert den Operator-Block fuer den Inhaltsstrom."""
    page.insert_image(rect, pixmap=_logo() if logo else _pix(i))
    name = page.get_images(full=True)[-1][7]
    h = page.rect.height
    return f"q {rect.width:.2f} 0 0 {rect.height:.2f} {rect.x0:.2f} {h - rect.y1:.2f} cm /{name} Do Q\n"


def _text_op(page, x, y, text, groesse=12):
    page.insert_font(fontname="helv", fontfile=None)
    h = page.rect.height
    t = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /helv {groesse} Tf {x:.2f} {h - y:.2f} Td ({t}) Tj ET\n"


def _setze_inhalt(doc, page, strom: str):
    page.clean_contents()
    xrefs = page.get_contents()
    doc.update_stream(xrefs[0], strom.encode("latin-1"))


def synth_satz(pfad):
    """Satzprogramm-Stand-in mit echtem Tag-Baum."""
    doc = fitz.open()
    root = doc.get_new_xref(); dokument = doc.get_new_xref(); karr = doc.get_new_xref(); pt = doc.get_new_xref()
    elemente = []          # (xref, seitenindex) in Dokumentreihenfolge
    nums = []              # ParentTree: StructParents -> Array der Elemente je MCID
    seiten_elemente = {}   # seite -> [xref je MCID]

    def struct(s, page, k, extra=""):
        x = doc.get_new_xref()
        doc.update_object(x, f"<< /Type /StructElem /S /{s} /P {dokument} 0 R /Pg {page.xref} 0 R /K {k} {extra}>>")
        return x

    def seite(nr, bausteine):
        """bausteine: Liste von ('P'|'Figure'|'PlacedGraphic'|'ohne'|'Collage', rect(s), text)."""
        page = doc.new_page(width=420, height=595)
        doc.xref_set_key(page.xref, "StructParents", str(nr))
        strom = ""; mcid = 0; je_mcid = []; bilder = 0
        for art, rects, text in bausteine:
            if art == "P":
                strom += f"/P <</MCID {mcid}>> BDC\n" + _text_op(page, 40, rects, text) + "EMC\n"
                je_mcid.append(struct("P", page, mcid)); mcid += 1
            elif art in ("Figure", "PlacedGraphic"):
                strom += f"/{art} <</MCID {mcid}>> BDC\n" + _bild_op(page, rects, bilder) + "EMC\n"
                je_mcid.append(struct(art, page, mcid)); mcid += 1; bilder += 1
            elif art == "Collage":
                x = None
                ks = []
                for r in rects:
                    strom += f"/Figure <</MCID {mcid}>> BDC\n" + _bild_op(page, r, bilder) + "EMC\n"
                    ks.append(mcid); mcid += 1; bilder += 1
                x = struct("Figure", page, "[ " + " ".join(str(k) for k in ks) + " ]")
                je_mcid.extend([x] * len(rects))
            elif art == "Artefakt":
                strom += "/Artifact BMC\n" + _text_op(page, 40, rects, text, 8) + "EMC\n"
            elif art == "ohne":
                strom += _bild_op(page, rects, bilder); bilder += 1
            elif art == "Logo":   # wiederholtes Logo (gleiches XObject auf jeder Seite), getaggt
                strom += f"/Figure <</MCID {mcid}>> BDC\n" + _bild_op(page, rects, bilder, logo=True) + "EMC\n"
                je_mcid.append(struct("Figure", page, mcid)); mcid += 1; bilder += 1
        _setze_inhalt(doc, page, strom)
        seiten_elemente[nr] = je_mcid
        # Elemente in Dokumentreihenfolge (ohne Doppelungen der Collage)
        gesehen = set()
        for x in je_mcid:
            if x not in gesehen:
                gesehen.add(x); elemente.append((x, nr))
        nums.append(f"{nr} [ " + " ".join(f"{x} 0 R" for x in je_mcid) + " ]")

    R = fitz.Rect
    seite(0, [("Logo", R(360, 20, 400, 60), None), ("P", 60, "Jahresbericht Musterverein (fiktiv)"), ("Figure", R(40, 80, 200, 200), None),
              ("P", 230, "Bildunterschrift zum Titelbild"), ("Artefakt", 570, "Seite 1")])
    seite(1, [("Logo", R(360, 20, 400, 60), None), ("P", 60, "Kapitel eins"), ("PlacedGraphic", R(40, 80, 180, 180), None),
              ("Figure", R(220, 80, 380, 180), None), ("Artefakt", 570, "Seite 2")])
    seite(2, [("P", 60, "Collage aus zwei Bildern in einem Tag"),
              ("Collage", [R(40, 80, 200, 200), R(220, 80, 380, 200)], None), ("Artefakt", 570, "Seite 3")])
    seite(3, [("P", 60, "Ungetaggtes Bild (Altlast)"), ("ohne", R(40, 80, 200, 200), None), ("Artefakt", 570, "Seite 4")])
    seite(4, [("P", 60, "Doppelseite rechts"), ("Figure", R(40, 80, 200, 200), None), ("Artefakt", 570, "Seite 5")])
    seite(5, [("P", 60, "Doppelseite links"), ("Figure", R(40, 80, 200, 200), None), ("Artefakt", 570, "Seite 6")])
    # Doppelseiten-Reihenfolge wie InDesign: Elemente der Seite 6 stehen im Baum VOR Seite 5 (kleiner Ruecksprung)
    reihenfolge = [x for x, s in elemente if s < 4] + [x for x, s in elemente if s == 5] + [x for x, s in elemente if s == 4]
    doc.update_object(karr, "[ " + " ".join(f"{x} 0 R" for x in reihenfolge) + " ]")
    doc.update_object(dokument, f"<< /Type /StructElem /S /Document /P {root} 0 R /K {karr} 0 R >>")
    doc.update_object(pt, "<< /Nums [ " + " ".join(nums) + " ] >>")
    doc.update_object(root, f"<< /Type /StructTreeRoot /K {dokument} 0 R /ParentTree {pt} 0 R "
                            f"/ParentTreeNextKey {len(nums)} /RoleMap << /PlacedGraphic /Figure >> >>")
    doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    doc.xref_set_key(doc.pdf_catalog(), "MarkInfo", "<< /Marked true >>")
    doc.set_metadata({"title": "Synthetischer Satz-Testfall (fiktiv)", "creator": "Satzprogramm-Stand-in"})
    doc.save(pfad, garbage=0, deflate=True)
    doc.close()


def untagged_raster(pfad):
    doc = fitz.open()
    for s in range(3):
        page = doc.new_page(width=420, height=595)
        strom = _text_op(page, 40, 60, f"Seite {s + 1}: Text ohne Tags (fiktiv)")
        strom += _bild_op(page, fitz.Rect(40, 80, 200, 200), s)
        if s == 1:
            strom += _bild_op(page, fitz.Rect(220, 80, 380, 200), s + 3)
        strom += _text_op(page, 40, 230, "Bildunterschrift: Testbild")
        _setze_inhalt(doc, page, strom)
    doc.set_metadata({"title": "Ungetaggter Testfall (fiktiv)", "producer": "ReportLab-Stand-in"})
    doc.save(pfad); doc.close()


def scan(pfad):
    doc = fitz.open()
    for s in range(2):
        page = doc.new_page(width=420, height=595)
        _setze_inhalt(doc, page, _bild_op(page, page.rect, s))
    doc.save(pfad); doc.close()


def vektor_layout(pfad):
    doc = fitz.open()
    page = doc.new_page(width=420, height=595)
    strom = _text_op(page, 40, 40, "Vektor-Testfall (fiktiv)")
    # seitengrosser Rahmen + Linien (Layout) mit Foto darin -> darf KEIN Bild werden
    strom += "q 0.8 0.8 0.8 RG 2 w 20 20 380 555 re S 20 300 m 400 300 l S Q\n"
    strom += _bild_op(page, fitz.Rect(60, 60, 220, 220), 0)
    # kleines Balkendiagramm (Vektor) -> soll ein Bild werden
    strom += "q 0.2 0.4 0.8 rg 260 350 20 80 re f 290 350 20 120 re f 320 350 20 50 re f 350 350 20 100 re f Q\n"
    strom += "q 0 0 0 RG 1 w 255 350 m 380 350 l S Q\n"
    _setze_inhalt(doc, page, strom)
    doc.save(pfad); doc.close()


def main(ziel):
    os.makedirs(ziel, exist_ok=True)
    import pdfua_export
    for nr, docx in (("01_lo_testdokument", "testdokument_inkludocs.docx"), ("02_lo_diagramm", "word_diagramm.docx")):
        out = os.path.join(ziel, nr + ".pdf")
        if os.path.exists(out):
            print(nr, "vorhanden"); continue
        if not pdfua_export.verfuegbar():
            print(nr, "UEBERSPRUNGEN: kein Konverter"); continue
        pdf, bericht = pdfua_export.konvertiere(os.path.join(FIX, docx), docx)
        open(out, "wb").write(pdf); print(nr, len(pdf), "Bytes, veraPDF-Regeln:", len(bericht.get("rules") or []))
    synth_satz(os.path.join(ziel, "03_synth_satz.pdf")); print("03_synth_satz ok")
    untagged_raster(os.path.join(ziel, "04_untagged_raster.pdf")); print("04_untagged_raster ok")
    scan(os.path.join(ziel, "05_scan.pdf")); print("05_scan ok")
    vektor_layout(os.path.join(ziel, "06_vektor_layout.pdf")); print("06_vektor_layout ok")
    shutil.copyfile(os.path.join(FIX, "testformular_inkludocs.pdf"), os.path.join(ziel, "07_formular.pdf")); print("07_formular ok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dokumente"))
