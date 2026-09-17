
# 17.09.2026
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

# fieldarray = [["Nummer", "Name", "Quickinfo", "Type-Nr", "Type", "Value"]]
fieldarray = [["Nummer", "Name", "Quickinfo", "Type-Nr", "Type", "Value", "Seite","left","bottom2" ,"right","top","Anzahl Felder mit identischem Namen"]]

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
            # print("kids is not none - ",feld1.GetFullName())
            anzfelder = kids.GetNumObjects()
            # print("Anzahl Kids:", kids.GetNumObjects())
            for i in range(kids.GetNumObjects()):
                kid = kids.GetDictionary(i)
                arrayr = kid.GetRect("Rect") 
                left = round(arrayr.left)
                bottom = round(arrayr.bottom)
                right = round(arrayr.right)
                top = round(arrayr.top)
        
            aufseiten = ""
            auf1seite = ""
            for page_num in range(doc.GetNumPages()):
                page = doc.AcquirePage(page_num)
                for i in range(page.GetNumAnnots()):
                    annot = page.GetAnnot(i)
                    subtype = annot.GetSubtype()
                    # print(i ,"  subtype : ",subtype)
                    if subtype == 20 :
                        field = annot.GetFormField()        
                        if field:
                            if field.GetFullName() == feld1.GetFullName():
                                aufseiten = aufseiten+str(page_num + 1)+" , "
                                # print(i ,"  aufseiten : ",aufseiten)
                                if auf1seite == "":
                                    # der erste Treffer wird gespeichert
                                    auf1seite = auf1seite+str(page_num + 1)       

        else:  
            # print("kids is none - ",feld1.GetFullName())
            anzfelder = 1
            arrayr = field_dict.GetRect("Rect")
            left = round(arrayr.left)
            bottom = round(arrayr.bottom)
            right = round(arrayr.right)
            top = round(arrayr.top)
            
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
        
        fieldarray.append([(ff+1), feld1.GetFullName(), feld1.GetTooltip(), feld1.GetType(), feldart, feldwert,auf1seite, left, bottom , right, top, anzfelder])

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

# Sortierung nach Seite und dann mit Tolleranz 5 von oben nach unten und von links nach rechts

kopf = fieldarray[0]
daten = fieldarray[1:]

# Sortierung zunächst nach Seite und top
daten.sort(key=lambda x: (int(x[6]), -int(x[10])))

# Zeilen bilden
sortiert = []
zeile = []
letzter_top = None
letzte_seite = None
toleranz = 5

for feld in daten:
    seite = int(feld[6])
    top = int(feld[10])

    if (letzte_seite is None or
        seite != letzte_seite or
        letzter_top is None or
        abs(top - letzter_top) > toleranz):

        # bisherige Zeile abschließen
        if zeile:
            zeile.sort(key=lambda x: int(x[7]))  # left
            sortiert.extend(zeile)

        zeile = [feld]
        letzter_top = top
        letzte_seite = seite

    else:
        zeile.append(feld)

# letzte Zeile
if zeile:
    zeile.sort(key=lambda x: int(x[7]))
    sortiert.extend(zeile)

fieldarray = [kopf] + sortiert

# in der ersten Spalte wieder fortlaufen von oben  
for i in range(1, len(fieldarray)):
    fieldarray[i][0] = i

    
pfadcsv = r"C:\Daten\20260709_Formularfelder\Formular_array.csv"
pfadcsv = pfad5+"_formulararray.csv"
with open(pfadcsv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerows(fieldarray)

print("csv gespeichert unter : ",pfadcsv)
print()

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")

