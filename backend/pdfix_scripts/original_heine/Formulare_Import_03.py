
# 19.08.2026
# Version 1.0.0.1
# Import der Quickinfos der Formularfelder aus einer csv


input("Drücke ENTER, um fortzufahren...")

import os
import csv
import time
import math
import copy

start = time.time()

from Utils import inputPath, outputPath
from pdfixsdk import *
import uuid
from pathlib import Path

pdfix = GetPdfix()

# if not pdfix.GetAccountAuthorization().Authorize("Benutzer", "Seriennummer"):
#   print("dummy message: PDFix SDK not authorized")

def main():
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')

    args = parser.parse_args()
    # print("args.input : ",args.input)
    # print("args.output : ",args.output)
    global aaadatei
    aaadatei = args.input
    doc = pdfix.OpenDoc(args.input, "")
    
    global filename
    path = Path(""+args.input)
    filename = path.name
    # print("filename", filename)     
    pfad3 = Path(""+args.input).parent
    # print("pfad3", pfad3) 
    global filename2
    filename2 = path.stem
    # print("filename2", filename2)
    global pfad5
    pfad5 = str(pfad3)+"\\"+filename2   
    # print("pfad5", pfad5)
    pfadcsv = pfad5+"_formulararray.csv"
    
    datenim = []
    with open(pfadcsv, newline="", encoding="utf-8") as csvfile:
    # with open(r"C:\Daten\20260226_python_scripte\datei_matrix_bitte_aendern.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for zeile in reader:
            datenim.append(zeile) 
             
    print("------------------------")
    num_fields = doc.GetNumFormFields()
    print("Anzahl Felder:", num_fields)

    for i in range(num_fields):
        field = doc.GetFormField(i)
        for a in range(0, len(datenim)-0):
            if datenim[a][1] == field.GetFullName():
                obj = field.GetObject()    
                page_obj = obj.Get("P") 
                neuertooltipp = datenim[a][2]
                obj.PutString("TU", neuertooltipp)                
                
    doc.Save(args.output, kSaveFull)
 
    print("-------------------")

import argparse 
lfn_counter = 0
lfn_tagnummer = 0

matrix = []

main() 

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")
