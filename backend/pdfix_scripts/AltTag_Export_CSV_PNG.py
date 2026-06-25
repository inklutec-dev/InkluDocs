# =============================================================================
#  AltTag_Export_CSV_PNG.py  —  InkluDocs PDFix-Export (getaggte PDFs)
# =============================================================================
#  HERKUNFT / PROVENANCE (fuer spaetere Code-Reviews wichtig):
#
#  Dieses Skript ist eine ZUSAMMENFUEHRUNG zweier Staende:
#
#   (A) BASIS = unsere Linux-Anpassung von Joerg Heines/Michael Karbes Export-
#       Skript, Stand "Karbe V1002" (27.05.2026). Davon stammt:
#         - StructTree-Walk + Figure-Erkennung
#         - Figure-Bild-Rendering (BoundingBox -> PNG)
#         - die CSV (Spalten 1-8)
#         - UNSERE Zusatzverbesserung: Seitenansicht-PNG 1x pro Seite gecacht,
#           in 144 DPI (2.0x), _render_page_view(). Die hat Karbes Vorlage NICHT.
#         - Linux-Anpassungen: kein input(), Pfade via CLI --data, kein Lizenz-
#           Block (Trial-/Wasserzeichen-Modus bis Verkaufsstart).
#
#   (B) NEU EINGEPFROPFT aus Karbes Version V1004 (Mail 17.06.2026): die tag-
#       basierte Text-Extraktion. Daraus stammen die zwei NEUEN Ausgaben je Bild:
#         - "Seiteninhalt": Text DER SEITE, auf der das Bild steht (Lesereihen-
#           folge aus den Tags rekonstruiert).
#         - "Kontext":      Text des ganzen ABSCHNITTS um das Bild (von Ueber-
#           schrift zu Ueberschrift, kann mehrere Seiten umfassen).
#       Diese zwei Texte landen als CSV-Spalten 9 + 10 und dienen spaeter der
#       KI als "enriched_context" (qualitativ besser als der bisherige
#       Ganz-Dokument-Text). Karbes Original-Methode (AcquireWordList +
#       Wort-BoundingBox-Treffer) ist 1:1 uebernommen.
#
#  WICHTIGER VERIFIZIERTER BEFUND (19.06.2026):
#       Die Text-Extraktion via AcquireWordList liefert AUCH im PDFix-Trial-
#       Modus (ohne Lizenz) SAUBEREN Text — KEINE '*'-Verstuemmelung. Getestet
#       an actino-master.pdf und Naturavetal-Katalog.pdf (18 Seiten). Die
#       1.320-EUR-Lizenz ist fuer diese Funktion NICHT noetig. (Im Gegensatz zur
#       frueheren PageMap/GetText-Methode, die im Trial verstuemmelt — siehe
#       Memo project_inkludocs_seitenvorschau, Befund 28.05.2026.)
#
#  UNSERE ZUSATZ-OPTIMIERUNG ueber V1004 hinaus: Wort-Liste 1x pro Seite cachen
#       (_wordlist_cache) statt pro Tag neu zu holen — spart auf grossen PDFs
#       viel Zeit. Dokumentiert an Ort und Stelle.
#
#   (C) KARBE V1.05 (Mail Michael Karbe, 24.06.2026): Lesbarkeit des
#       "Seiteninhalts". Hinter jedem Tag wird ein Zeilenumbruch gesetzt (vorher
#       ein Fliesstext-Blob mit Leerzeichen), Listen-Label bleiben inline.
#       Umgesetzt in _join_reading_order(), genutzt von _page_content() (der
#       ANGEZEIGTE Seitentext). Der KI-Kontext _chapter_context() bleibt bewusst
#       flach — Begruendung dort. Frontend zeigt es bereits via
#       white-space:pre-wrap (style.css .page-text-content), kein UI-Eingriff.
#
#  Aufruf (unveraendert ggue. V1002-Linux):
#       python3 AltTag_Export_CSV_PNG.py -i <input.pdf> -o <egal.pdf> -d <outdir>
# =============================================================================

import os
import csv
import time
import argparse
import math

start = time.time()

from Utils import inputPath, outputPath
from pdfixsdk import *
from pathlib import Path

pdfix = GetPdfix()

# Lizenz: Karbe besorgt sie zum Verkaufsstart. Jetzt Default-Modus (Wasserzeichen).
# if not pdfix.GetAccountAuthorization().Authorize("Benutzer", "Seriennummer"):
#     print("PDFix SDK not authorized")

# -----------------------------------------------------------------------------
#  TEIL A — unveraendert aus unserer V1002-Linux-Basis: Seitenansicht-Cache
# -----------------------------------------------------------------------------
_rendered_pages = {}     # page_num -> page_view_path (1 Seitenansicht pro Seite)
PAGE_VIEW_SCALE = 2.0    # ~144 DPI fuer lesbare Seitenansicht


def _render_page_view(page, page_num, crop_box):
    """Rendert die ganze Seite als PNG. Cache: 1x pro Seite (Schluessel page_num)."""
    if page_num in _rendered_pages:
        return _rendered_pages[page_num]
    pv = page.AcquirePageView(PAGE_VIEW_SCALE, kRotate0)
    dr = pv.RectToDevice(crop_box)
    dr.right -= dr.left
    dr.left = 0
    dr.bottom -= dr.top
    dr.top = 0
    img = pdfix.CreateImage(pv.GetDeviceWidth(), pv.GetDeviceHeight(), kImageDIBFormatArgb)
    rp = PdfPageRenderParams()
    rp.clip_box = crop_box
    rp.image = img
    rp.matrix = pv.GetDeviceMatrix()
    page.DrawContent(rp)
    path = os.path.join(data_dir, f"{filename2}_p{page_num + 1}_seitenansicht.png")
    ip = PdfImageParams()
    img.SaveRect(path, ip, dr)
    _rendered_pages[page_num] = path
    return path


# -----------------------------------------------------------------------------
#  TEIL B — NEU (aus Karbe V1004): tag-basierte Text-Extraktion
# -----------------------------------------------------------------------------
# Welche Tag-Typen tragen Text? (aus V1004 uebernommen)
_TEXT_TAGS = {"P", "H", "H1", "H2", "H3", "H4", "H5", "H6", "LBody", "Lbl"}
_HEADING_TAGS = {"H", "H1", "H2", "H3", "H4", "H5", "H6"}

_doc = None              # Haupt-Dokumenthandle (in main() gesetzt)
_page_cache = {}         # page_num -> Page (mehrfach-Acquire vermeiden)
_wordlist_cache = {}     # page_num -> WordList  (UNSERE Optimierung ggue. V1004)

# tagarray: in DFS-Reihenfolge ein Eintrag pro Struktur-Element:
#   [tag_typ, page_num, text]   text="" bei Nicht-Text-Tags/Figures.
# Daraus berechnen wir spaeter Seiteninhalt (= Text der Seite) und Kontext
# (= Text zwischen den zwei umgebenden Ueberschriften).
tagarray = []
# figure_rows: Merker je Figure, um Kontext/Seiteninhalt der CSV-Zeile zuzuordnen.
figure_rows = []         # [{"lfnr": int, "page_num": int, "tagidx": int}, ...]


def _get_page(page_num):
    p = _page_cache.get(page_num)
    if p is None:
        p = _doc.AcquirePage(page_num)
        _page_cache[page_num] = p
    return p


def _tag_text(elem, page_num):
    """Rekonstruiert den Text EINES Tags aus den Wort-Positionen.

    Methode 1:1 aus Karbe V1004: hole die Wortliste der Seite (in Lesereihen-
    folge) und nimm jene Woerter, deren BoundingBox in der BoundingBox des Tags
    liegt. Die kleinen +1.01/-1.01-Korrekturen sind Karbes Toleranzwerte.
    Liefert auch im Trial-Modus sauberen Text (verifiziert 19.06.2026).
    """
    bbox = elem.GetBBox(page_num)
    tagl = math.floor(bbox.left)
    tagu = math.floor(bbox.bottom)
    tago = math.ceil(bbox.top)

    wl = _wordlist_cache.get(page_num)
    if wl is None:
        # UNSERE Optimierung: Wortliste nur 1x pro Seite holen (V1004 holte sie
        # pro Tag neu -> auf grossen PDFs deutlich langsamer).
        page = _get_page(page_num)
        wl = page.AcquireWordList(kWordFinderAlgLatest, 0)
        _wordlist_cache[page_num] = wl

    treffer = []
    for k in range(wl.GetNumWords()):
        w = wl.GetWord(k)
        wb = w.GetBBox()
        textl = math.ceil(wb.left + 1.01)
        textu = math.ceil(wb.bottom + 1.01)
        texto = math.floor(wb.top - 1.01)
        if textl >= tagl and textu >= tagu and texto <= tago:
            treffer.append(w.GetText())
    return " ".join(treffer).strip()


# -----------------------------------------------------------------------------
#  Struktur-Walk: rendert Figures (Teil A) UND sammelt Tag-Texte (Teil B)
# -----------------------------------------------------------------------------
def process_struct_elem(elem: PdsStructElement):
    page_num = -1
    bbox = None
    for i in range(elem.GetNumPages()):
        page_num = elem.GetPageNumber(i)
        bbox = elem.GetBBox(page_num)

    etype = elem.GetType(True)

    # --- Teil B: einen tagarray-Eintrag je Element anlegen ---
    if page_num >= 0 and etype in _TEXT_TAGS:
        tagarray.append([etype, page_num, _tag_text(elem, page_num)])
    else:
        # Figures und Nicht-Text-Tags: Platzhalter (Text leer). Figures dienen
        # als Anker fuer die Kapitel-Suche, deshalb auch sie in den tagarray.
        tagarray.append([etype, page_num, ""])
    this_tagidx = len(tagarray) - 1

    # --- Teil A: Figure rendern (unveraendert aus V1002-Linux) ---
    if etype == "Figure" and page_num >= 0:
        bboxfigure = elem.GetBBox(page_num)
        doc = pdfix.OpenDoc(aaadatei, "")
        page = doc.AcquirePage(page_num)
        crop_box = page.GetCropBox()

        pageView = page.AcquirePageView(1.0, kRotate0)
        devRect = pageView.RectToDevice(bboxfigure)
        devRect.right -= devRect.left
        devRect.left = 0
        devRect.bottom -= devRect.top
        devRect.top = 0
        psImage = pdfix.CreateImage(pageView.GetDeviceWidth(), pageView.GetDeviceHeight(), kImageDIBFormatArgb)
        renderParams = PdfPageRenderParams()
        renderParams.clip_box = bbox
        renderParams.image = psImage
        renderParams.matrix = pageView.GetDeviceMatrix()
        page.DrawContent(renderParams)

        global lfn_counter
        lfn_counter = lfn_counter + 1
        figure_path = os.path.join(data_dir, f"{filename2}_ExtractImages_{lfn_counter}.png")
        imageParams = PdfImageParams()
        psImage.SaveRect(figure_path, imageParams, devRect)

        page_view_path = _render_page_view(page, page_num, crop_box)

        matrix.append([lfn_counter, figure_path,
                       elem.GetTitle(), elem.GetActualText(), elem.GetAlt(),
                       filename2, page_num + 1, page_view_path])
        # Merker fuer die spaetere Kontext-Zuordnung (Teil B)
        figure_rows.append({"lfnr": lfn_counter, "page_num": page_num, "tagidx": this_tagidx})

    # Kinder rekursiv
    for i in range(elem.GetNumChildren()):
        child_type = elem.GetChildType(i)
        if child_type == kPdsStructChildElement:
            obj = elem.GetChildObject(i)
            child_elem = elem.GetStructTree().GetStructElementFromObject(obj)
            process_struct_elem(child_elem)


# -----------------------------------------------------------------------------
#  Teil B: Seiteninhalt + Kontext berechnen (nach dem Walk)
# -----------------------------------------------------------------------------
def _join_reading_order(typed_parts):
    """Fuegt Tag-Texte in Lesereihenfolge zu lesbarem Seitentext zusammen.

    Aenderung Karbe V1.05 (Mail Michael Karbe, 24.06.2026): hinter jedem Tag
    einen Zeilenumbruch setzen, damit die ANZEIGE des Seiteninhalts strukturiert
    erscheint und nicht als ein einziger Fliesstext-Blob (vorher " ".join). So
    bekommt jeder Absatz/jede Ueberschrift/jeder Listeneintrag eine eigene Zeile.

    AUSNAHME (1:1 aus Karbes Vorlage): Listen-Label (Tag-Typ "Lbl") und einzelne
    Aufzaehlungszeichen (Text aus genau einem Zeichen, z.B. "•") bleiben INLINE,
    d.h. sie werden mit Leerzeichen statt Umbruch angehaengt. Dadurch klebt das
    Label "1." / "•" am Anfang seines Listeneintrags, statt allein auf einer
    Zeile zu stehen.

    typed_parts: Liste von (tag_typ, text) in Lesereihenfolge.
    """
    out = []
    for etype, text in typed_parts:
        inline = (etype == "Lbl") or (len(text) == 1)
        out.append(text + (" " if inline else "\n"))
    return "".join(out).strip()


def _page_content(page_num):
    """Text aller Text-Tags auf einer Seite (Lesereihenfolge = DFS-Reihenfolge).

    Wird im Werkzeug als "Seitentext" ANGEZEIGT (Weg: CSV-Spalte 9 ->
    pdf_processor page_text -> Frontend "Seitentext anzeigen", CSS
    white-space:pre-wrap). Deshalb hier die zeilenweise, lesbare Formatierung
    nach Karbe V1.05 (siehe _join_reading_order)."""
    typed = [(t[0], t[2]) for t in tagarray if t[1] == page_num and t[2]]
    return _join_reading_order(typed)


def _chapter_context(tagidx):
    """Text des Abschnitts um den Figure-Tag: von der naechsten Ueberschrift
    OBERHALB bis zur naechsten Ueberschrift UNTERHALB (exklusive). Logik wie
    in Karbe V1004 (dort ueber figurearray geloest).

    BEWUSST weiterhin flacher " ".join (NICHT die zeilenweise V1.05-Formatierung):
    Dieser Text wird NICHT angezeigt, sondern geht als enriched_context an die KI
    (pdf_processor _ctx). Karbes V1.05 betrifft ausdruecklich nur die ANZEIGE des
    Seiteninhalts. Eine Whitespace-Umstellung hier wuerde nur die Prompt-Bytes und
    damit potenziell die Generierung veraendern, ohne sichtbaren Nutzen."""
    start = 0
    for a in range(tagidx, -1, -1):
        if tagarray[a][0] in _HEADING_TAGS:
            start = a
            break
    end = len(tagarray)
    for b in range(tagidx + 1, len(tagarray)):
        if tagarray[b][0] in _HEADING_TAGS:
            end = b
            break
    parts = [tagarray[z][2] for z in range(start, end) if tagarray[z][2]]
    return " ".join(parts).strip()


def main():
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')
    parser.add_argument('-d', '--data', required=True, help='LINUX: Output dir for PNGs and CSV')

    args = parser.parse_args()
    global aaadatei, data_dir, _doc
    aaadatei = args.input
    data_dir = args.data
    os.makedirs(data_dir, exist_ok=True)
    _doc = pdfix.OpenDoc(args.input, "")
    path = Path("" + args.input)
    global filename2
    filename2 = path.stem
    struct_tree = _doc.GetStructTree()
    for i in range(struct_tree.GetNumChildren()):
        obj = struct_tree.GetChildObject(i)
        elem = struct_tree.GetStructElementFromObject(obj)
        process_struct_elem(elem)
    print(lfn_counter, " Bilder + ", len(_rendered_pages), " Seitenansichten extrahiert")


lfn_counter = 0
matrix = []
# CSV-Kopf: Spalten 1-8 wie bisher, NEU Spalten 9-10 (Seiteninhalt, Kontext)
matrix.append(["laufende Nummer", "Pfad mit Dateinamen", "Titel",
               "Echter Text", "Alternativer Text", "Dateiname",
               "Seitennummer", "Pfad Seitenansicht",
               "Seiteninhalt", "Kontext"])
matrix.append([])

main()

# --- Teil B: Seiteninhalt + Kontext je Figure an die CSV-Zeilen anhaengen ---
context_by_lfnr = {}
for fr in figure_rows:
    si = _page_content(fr["page_num"])
    ko = _chapter_context(fr["tagidx"])
    context_by_lfnr[fr["lfnr"]] = (si, ko)

for row in matrix[1:]:
    if not row:           # die leere Trennzeile unveraendert lassen
        continue
    lfnr = row[0]
    si, ko = context_by_lfnr.get(lfnr, ("", ""))
    row.append(si)
    row.append(ko)

pfadcsv = os.path.join(data_dir, "figure_array.csv")
with open(pfadcsv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(matrix)

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")
print("CSV:", pfadcsv)
