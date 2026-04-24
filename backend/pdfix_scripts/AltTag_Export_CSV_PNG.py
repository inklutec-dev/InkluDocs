# Linux-Anpassung von Jörg Heines Export-Script (Original: AltTag_Export_CSV_PNG.py)
# Nur zwei Änderungen:
#   1. input("...") auskommentiert (Server-Betrieb, kein interaktives stdin)
#   2. Windows-Pfade C:\... durch CLI-Parameter --data ersetzt
# Alle Kern-Logik (StructTree-Traversierung, Figure-Erkennung, PNG/CSV) 1:1 übernommen.

# input("Drücke ENTER, um fortzufahren...")  # LINUX: Server-Container hat keinen stdin

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

def process_struct_elem(elem: PdsStructElement):
  alt = elem.GetAlt()
  id = elem.GetId()
  for i in range(elem.GetNumPages()):
    page_num = elem.GetPageNumber(i)
    bbox = elem.GetBBox(page_num)


  if elem.GetType(True) == "Figure":
        print("------------------------------------------")
        print(elem.GetType(True))
        bboxfigure = elem.GetBBox(page_num)
        count = elem.GetNumChildren()
        page_num = elem.GetPageNumber(i)
        doc = pdfix.OpenDoc(aaadatei, "")
        page = doc.AcquirePage(page_num)
        crop_box = page.GetCropBox()
        width = crop_box.right - crop_box.left
        height = crop_box.top - crop_box.bottom
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

        # save image to file
        global lfn_counter
        lfn_counter=lfn_counter+1
        global filename2
        # LINUX: data_dir statt C:\Daten\python_scripte\AltTag\Bilder_und_CSV\
        path = os.path.join(data_dir, f"{filename2}_ExtractImages_{lfn_counter}.png")
        imageParams = PdfImageParams()
        psImage.SaveRect(path, imageParams, devRect)
        matrix.append([lfn_counter,path,elem.GetTitle(),elem.GetActualText(),elem.GetAlt(),filename2])

  # process children
  for i in range(elem.GetNumChildren()):
    child_type = elem.GetChildType(i)
    if child_type == kPdsStructChildElement:
      obj = elem.GetChildObject(i)
      child_elem = elem.GetStructTree().GetStructElementFromObject(obj)
      process_struct_elem(child_elem)
      pass
    elif child_type == kPdsStructChildObject:
      pass
    elif child_type == kPdsStructChildPageContent:
      pass
    elif child_type == kPdsStructChildStreamContent:
      pass


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
    path = Path(""+args.input)
    global filename2
    filename2 = path.stem
    struct_tree = doc.GetStructTree()
    for i in range(struct_tree.GetNumChildren()):
        obj = struct_tree.GetChildObject(i)
        elem = struct_tree.GetStructElementFromObject(obj)
        process_struct_elem(elem)
    print(lfn_counter," Bilder und eine csv im Unterordner gespeichert")


lfn_counter = 0

matrix = []

matrix.append(["laufende Nummer","Pfad mit Dateinamen","Titel","Echter Text","Alternativer Text","Dateiname"])
matrix.append([])


main()

# LINUX: pfadcsv nach main() aufgebaut, weil data_dir dann gesetzt ist
pfadcsv = os.path.join(data_dir, "figure_array.csv")

with open(pfadcsv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerows(matrix)

end = time.time()

print("Dauer:", round((end - start), 2), "Sekunden")
print("CSV:", pfadcsv)
