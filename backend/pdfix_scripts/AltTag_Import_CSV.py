# Linux-Anpassung von Jörg Heines Import-Script (Original: AltTag_Import_CSV.py)
# Nur zwei Änderungen:
#   1. input("...") auskommentiert (Server-Betrieb, kein interaktives stdin)
#   2. Windows-Pfad für CSV durch CLI-Parameter --csv ersetzt
# Kern-Logik (StructTree-Walk, elem.SetAlt(), doc.Save()) 1:1 übernommen.

# input("Drücke ENTER, um fortzufahren...")  # LINUX: Server-Container hat keinen stdin

import os
import csv
import argparse

from Utils import inputPath, outputPath
from pdfixsdk import *
import uuid
from pathlib import Path

pdfix = GetPdfix()

# LINUX: CSV-Pfad kommt jetzt vom CLI-Parameter --csv
parser = argparse.ArgumentParser(description="Import Alt-Texte aus CSV in PDF.")
parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')
parser.add_argument('-c', '--csv', required=True, help='LINUX: CSV-Datei mit Alt-Texten')
args = parser.parse_args()

daten = []
anz = 0
with open(args.csv, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    for zeile in reader:
        daten.append(zeile)
        anz = anz + 1


def csvimport_struct_elem(elem: PdsStructElement):
  if elem.GetType(True) == "Figure":
                global lfn_counter
                lfn_counter=lfn_counter+1
                alttextneu = daten[lfn_counter+1][4]
                elem.SetAlt(alttextneu)

  # process children
  for i in range(elem.GetNumChildren()):
    child_type = elem.GetChildType(i)
    if child_type == kPdsStructChildElement:
        obj = elem.GetChildObject(i)
        child_elem = elem.GetStructTree().GetStructElementFromObject(obj)
        csvimport_struct_elem(child_elem)


def main():
    global aaadatei
    aaadatei = args.input
    doc = pdfix.OpenDoc(args.input, "")
    path = Path(""+args.input)
    filename = path.name
    struct_tree = doc.GetStructTree()
    for i in range(struct_tree.GetNumChildren()):
        obj = struct_tree.GetChildObject(i)
        elem = struct_tree.GetStructElementFromObject(obj)
        csvimport_struct_elem(elem)
    doc.Save(args.output, kSaveFull)


lfn_counter = 0
main()

print("fertig -", args.output)
