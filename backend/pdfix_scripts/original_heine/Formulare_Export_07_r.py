
# 25.08.2026
# Version 1.0.0.2


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

fieldarray = [["Nummer", "Name", "Quickinfo", "Type-Nr", "Type", "Value", "Seite"]]

def main():
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')

    args = parser.parse_args()

    global aaadatei
    aaadatei = args.input
    doc = pdfix.OpenDoc(args.input, "")

    print("------------------------")
    num_fields = doc.GetNumFormFields()
    print("Anzahl Felder:", num_fields)
    print("-------------------")

    for ff in range(doc.GetNumFormFields()):

        feld1 = doc.GetFormField(ff)
        obj = feld1.GetObject()
        vorher = feld1.GetTooltip()
        feldname = feld1.GetFullName()
        
        feldart = "unbekannt"
        if feld1.GetType()==0:
            feldart = "Unknown field"
        if feld1.GetType()==1:
            feldart = "Button"
        if feld1.GetType()==2:
            feldart = "Radio button"
        if feld1.GetType()==3:
            feldart = "Check box"
        if feld1.GetType()==4:
            feldart = "Text field"
        if feld1.GetType()==5:
            feldart = "Dropdown field"
        if feld1.GetType()==6:
            feldart = "Listenfeld"
        if feld1.GetType()==7:
            feldart = "Signatur field"

        field_dict = feld1.GetObject()     
        kids = field_dict.GetArray("Kids")
        
        if kids is not None:             
            aufseiten = ""
            auf1seite = ""
            for page_num in range(doc.GetNumPages()):
                page = doc.AcquirePage(page_num)
                for i in range(page.GetNumAnnots()):
                    annot = page.GetAnnot(i)
                    subtype = annot.GetSubtype()
                    if subtype == 20 :
                        field = annot.GetFormField()
                        if field:
                            if field.GetFullName() == feld1.GetFullName():
                                aufseiten = aufseiten+str(page_num + 1)+" , "
                                if auf1seite == "":
                                    auf1seite = auf1seite+str(page_num + 1)       

        else:           
            page_obj = field_dict.Get("P")           
            p = field_dict.Get("P")
            if p is not None:
                for i in range(doc.GetNumPages()):
                    page = doc.AcquirePage(i)
                    page_dict = page.GetObject()
                    if page_dict.GetId() == p.GetId():
                        auf1seite = (i + 1)

                page.Release()
                
        feldwert = feld1.GetValue()
        if feldwert == "" :
            feldwert = "kein Wert"
        # fieldarray.append([(ff+1), feld1.GetFullName(), feld1.GetTooltip(), feld1.GetType(), feldart, feld1.GetValue(),auf1seite ])                 
        fieldarray.append([(ff+1), feld1.GetFullName(), feld1.GetTooltip(), feld1.GetType(), feldart, feldwert,auf1seite ])       

    pfad3 = Path(""+args.input).parent
      
    global filename
    path = Path(""+args.input)
    filename = path.name

    global filename2
    filename2 = path.stem
    
    pfad4 = str(pfad3)+"\\"+filename   

    global pfad5
    pfad5 = str(pfad3)+"\\"+filename2   
   
import argparse 
    
main() 

pfadcsv = r"C:\Daten\20260709_Formularfelder\Formular_array.csv"
pfadcsv = pfad5+"_formulararray.csv"
with open(pfadcsv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerows(fieldarray)

print("csv gespeichert unter : ",pfadcsv)
print()

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")
