# =============================================================================
#  Formular_Export_Quickinfo.py — InkluDocs Quickinfo-Werkzeug (PDF-Formulare)
# =============================================================================
#  HERKUNFT / PROVENANCE:
#
#  Dies ist Joerg Heines Skript "Formulare_Export_07_r.py" (Actino Software
#  GmbH, heine@actino.de), Version 1.0.0.2 vom 25.08.2026, gesendet an
#  kontakt@inklutec.de. Das unveraenderte Original liegt daneben unter
#  original_heine/Formulare_Export_07_r.py. Die Logik (Feld-Aufzaehlung,
#  Feldart-Zuordnung, Seitenermittlung ueber Kids/Annotationen bzw. "P",
#  CSV-Spalten) ist 1:1 uebernommen.
#
#  Nur die folgenden Anpassungen fuer den Serverbetrieb (27.08.2026,
#  Steve + Fable 5), jeweils mit "# InkluDocs:" markiert:
#    1. input("Druecke ENTER") entfernt (kein stdin auf dem Server).
#    2. CSV-Pfad als Parameter -c/--csv statt Windows-Pfad neben der PDF;
#       -o/--output ist beim Export nicht mehr Pflicht.
#    3. DATENSCHUTZ: Spalte "Value" enthaelt nicht mehr den eingetragenen
#       Feldwert, sondern nur "kein Wert" bzw. "Wert vorhanden". Formulare
#       koennen bereits personenbezogene Eingaben enthalten; fuer Quickinfos
#       brauchen wir sie nicht und speichern sie deshalb nirgends.
#    4. Lizenzaktivierung ueber PDFIX_LICENSE_USER/PDFIX_LICENSE_KEY (wie in
#       AltTag_Import_CSV.py), Ergebniszeile "FIELDS_FOUND=n" fuer den Wrapper.
#    5. Ungepaarte UTF-16-Surrogate in Feldnamen/Quickinfos werden vor dem
#       CSV-Schreiben ersetzt (Lehre aus dem PDFix-Befund vom 27.08.2026,
#       KBV_Formeln.pdf: sonst UnicodeEncodeError und Abbruch).
#    6. "aufseiten"/"auf1seite" werden vor der Verzweigung initialisiert (in der
#       Vorlage nur im Kids-Zweig), und im Kids-Zweig wird jede Seite nach der
#       Annotationsschleife mit page.Release() freigegeben (Speicher bei
#       grossen Formularen). "aufseiten" (alle Seiten) wird wie in der Vorlage
#       berechnet, aber nicht ausgegeben — die Seitenliste kommt in InkluDocs
#       aus PyMuPDF (formular_processor.py).
#    7. doc.Close() nach dem Lesen; Fehlermeldung + Exit-Code 2, wenn die PDF
#       nicht geoeffnet werden kann.
#
#  CSV-Format (Semikolon, UTF-8): Nummer;Name;Quickinfo;Type-Nr;Type;Value;Seite
# =============================================================================

# 25.08.2026
# Version 1.0.0.2


# InkluDocs: kein input()-Stopp auf dem Server
# input("Drücke ENTER, um fortzufahren...")

import os
import csv
import time
import math
import copy
import sys  # InkluDocs: fuer stderr/Exit-Code

start = time.time()

from Utils import inputPath, outputPath
from pdfixsdk import *
import uuid
from pathlib import Path

pdfix = GetPdfix()

# InkluDocs: Lizenz (27.08.2026). Ohne die beiden Variablen laeuft das SDK als
# Testversion weiter; eine abgelehnte Lizenz bricht den Lauf nicht ab.
_lu, _lk = os.environ.get("PDFIX_LICENSE_USER", ""), os.environ.get("PDFIX_LICENSE_KEY", "")
if _lu and _lk:
    try:
        if not pdfix.GetAccountAuthorization().Authorize(_lu, _lk):
            print("PDFix-Lizenz nicht angenommen: " + str(pdfix.GetError()), file=sys.stderr)
    except Exception as _e:
        print("PDFix-Lizenz: Fehler bei der Aktivierung: " + repr(_e), file=sys.stderr)


def _sauber(s):
    """InkluDocs: ungepaarte Surrogate durch U+FFFD ersetzen (CSV-sicher)."""
    if not isinstance(s, str):
        return s
    return s.encode("utf-8", "replace").decode("utf-8")


fieldarray = [["Nummer", "Name", "Quickinfo", "Type-Nr", "Type", "Value", "Seite"]]

def main():
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-o', '--output', required=False, help='(unbenutzt, Kompatibilitaet)')  # InkluDocs
    parser.add_argument('-c', '--csv', required=True, help='Pfad der zu schreibenden CSV')      # InkluDocs

    args = parser.parse_args()

    global aaadatei
    aaadatei = args.input
    doc = pdfix.OpenDoc(args.input, "")
    if not doc:  # InkluDocs: klare Meldung statt AttributeError
        print("PDF konnte nicht geoeffnet werden: " + str(pdfix.GetError()), file=sys.stderr)
        sys.exit(2)

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

        aufseiten = ""   # InkluDocs: initialisiert, damit die Variable in jedem Zweig existiert
        auf1seite = ""
        if kids is not None:
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
                page.Release()  # InkluDocs: Seite wieder freigeben (Speicher bei grossen Formularen)

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
        # InkluDocs (Datenschutz): nie den Wert selbst, nur ob einer vorhanden ist.
        # "Off" ist bei Checkbox/Radio der Nicht-Ausgewaehlt-Zustand, also kein Wert.
        if feldwert == "" or feldwert == "Off":
            feldwert = "kein Wert"
        else:
            feldwert = "Wert vorhanden"
        # fieldarray.append([(ff+1), feld1.GetFullName(), feld1.GetTooltip(), feld1.GetType(), feldart, feld1.GetValue(),auf1seite ])
        fieldarray.append([(ff+1), _sauber(feld1.GetFullName()), _sauber(feld1.GetTooltip()), feld1.GetType(), feldart, feldwert,auf1seite ])

    # InkluDocs: Pfad-Ableitung aus der Vorlage entfaellt, CSV-Pfad kommt aus -c
    global pfadcsv
    pfadcsv = args.csv
    doc.Close()  # InkluDocs

import argparse

main()

# InkluDocs: pfadcsv aus main() (Parameter -c) statt Windows-Pfad
with open(pfadcsv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerows(fieldarray)

print("csv gespeichert unter : ",pfadcsv)
print()

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")
print("FIELDS_FOUND=%d" % (len(fieldarray) - 1))  # InkluDocs: Ergebniszeile fuer den Wrapper
