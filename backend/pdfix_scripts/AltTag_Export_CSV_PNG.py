# Linux-Anpassung von Joerg Heines Export-Script (Original: AltTag_Export_CSV_PNG.py)
# Basis: Karbes V1002 (27.05.2026)
# Anpassungen:
#   1. input("...") auskommentiert (Server-Betrieb, kein interaktives stdin)
#   2. Windows-Pfade C:\... durch CLI-Parameter --data ersetzt
#   3. Lizenz-Block weggelassen (Karbe: spaeter zum Verkaufsstart, jetzt Wasserzeichen-Modus)
#   4. NEU 27.05.2026: Seitenansicht-PNG, 1x pro Seite (gecached), nicht pro Figure
#   5. NEU: CSV-Spalten 7+8 = page_number + page_view_path

# input("Druecke ENTER, um fortzufahren...")  # LINUX: stdin nicht verfuegbar

import os
import csv
import time
import argparse

start = time.time()

from Utils import inputPath, outputPath
from pdfixsdk import *
import uuid
from pathlib import Path

pdfix = GetPdfix()

# Lizenz: Karbe besorgt sie zum Verkaufsstart. Jetzt Default-Modus (Wasserzeichen).
# if not pdfix.GetAccountAuthorization().Authorize("Benutzer", "Seriennummer"):
#     print("PDFix SDK not authorized")

_rendered_pages = {}  # page_num -> page_view_path (1 Seitenansicht pro Seite, nicht pro Figure)
PAGE_VIEW_SCALE = 2.0  # ~144 DPI fuer lesbare Seitenansicht


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


def process_struct_elem(elem: PdsStructElement):
    alt = elem.GetAlt()
    id = elem.GetId()
    page_num = -1
    bbox = None
    for i in range(elem.GetNumPages()):
        page_num = elem.GetPageNumber(i)
        bbox = elem.GetBBox(page_num)

    if elem.GetType(True) == "Figure" and page_num >= 0:
        bboxfigure = elem.GetBBox(page_num)
        doc = pdfix.OpenDoc(aaadatei, "")
        page = doc.AcquirePage(page_num)
        crop_box = page.GetCropBox()

        # Figure-Bild rendern (1.0 Skala wie V1000)
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

        # Seitenansicht (1x pro Seite, gecached)
        page_view_path = _render_page_view(page, page_num, crop_box)

        matrix.append([lfn_counter, figure_path,
                       elem.GetTitle(), elem.GetActualText(), elem.GetAlt(),
                       filename2, page_num + 1, page_view_path])

    # process children
    for i in range(elem.GetNumChildren()):
        child_type = elem.GetChildType(i)
        if child_type == kPdsStructChildElement:
            obj = elem.GetChildObject(i)
            child_elem = elem.GetStructTree().GetStructElementFromObject(obj)
            process_struct_elem(child_elem)


def main():
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')
    parser.add_argument('-d', '--data', required=True, help='LINUX: Output dir for PNGs and CSV')

    args = parser.parse_args()
    global aaadatei, data_dir
    aaadatei = args.input
    data_dir = args.data
    os.makedirs(data_dir, exist_ok=True)
    doc = pdfix.OpenDoc(args.input, "")
    path = Path("" + args.input)
    global filename2
    filename2 = path.stem
    struct_tree = doc.GetStructTree()
    for i in range(struct_tree.GetNumChildren()):
        obj = struct_tree.GetChildObject(i)
        elem = struct_tree.GetStructElementFromObject(obj)
        process_struct_elem(elem)
    print(lfn_counter, " Bilder + ", len(_rendered_pages), " Seitenansichten im Unterordner gespeichert")


lfn_counter = 0
matrix = []
matrix.append(["laufende Nummer", "Pfad mit Dateinamen", "Titel",
               "Echter Text", "Alternativer Text", "Dateiname",
               "Seitennummer", "Pfad Seitenansicht"])
matrix.append([])

main()

pfadcsv = os.path.join(data_dir, "figure_array.csv")
with open(pfadcsv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(matrix)

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")
print("CSV:", pfadcsv)
