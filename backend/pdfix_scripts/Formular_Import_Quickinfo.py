# =============================================================================
#  Formular_Import_Quickinfo.py — InkluDocs Quickinfo-Werkzeug (PDF-Formulare)
# =============================================================================
#  HERKUNFT / PROVENANCE:
#
#  Dies ist Joerg Heines Skript "Formulare_Import_03.py" (Actino Software
#  GmbH, heine@actino.de), Version 1.0.0.1 vom 19.08.2026. Das unveraenderte
#  Original liegt unter original_heine/Formulare_Import_03.py. Die Logik
#  (CSV lesen, Feld ueber den vollen Namen finden, Quickinfo als /TU in das
#  Feld-Dictionary schreiben, Speichern mit kSaveFull) ist 1:1 uebernommen.
#
#  Nur die folgenden Anpassungen fuer den Serverbetrieb (27.08.2026,
#  Steve + Fable 5), jeweils mit "# InkluDocs:" markiert:
#    1. input("Druecke ENTER") entfernt.
#    2. CSV-Pfad als Parameter -c/--csv statt "<pdf>_formulararray.csv" neben
#       der PDF (Windows-Pfadlogik).
#    3. Zeilen mit LEERER Quickinfo werden uebersprungen: ein Feld ohne Text
#       bleibt exakt wie im Original, eine dort vorhandene Quickinfo bleibt
#       erhalten. Die Vorlage haette einen leeren /TU geschrieben. Es wird
#       also nie eine Quickinfo geloescht, nur gesetzt oder ersetzt.
#    4. Kopfzeile der CSV wird uebersprungen (die Vorlage verglich sie mit,
#       traf aber nie, weil kein Feld "Name" heisst — hier ausdruecklich).
#    5. Lizenzaktivierung ueber Umgebungsvariablen; Ergebniszeile
#       "TU_APPLIED=n FIELDS_FOUND=m NOT_FOUND=name1|name2" fuer den Wrapper
#       und Exit-Code 4, wenn ein CSV-Name in der PDF nicht vorkommt.
#    6. Es wird ausser /TU nichts geaendert (keine Werte, keine Flags) — das
#       gilt schon fuer die Vorlage und wird hier nur festgehalten.
# =============================================================================

# 19.08.2026
# Version 1.0.0.1
# Import der Quickinfos der Formularfelder aus einer csv


# InkluDocs: kein input()-Stopp auf dem Server
# input("Drücke ENTER, um fortzufahren...")

import os
import csv
import time
import math
import copy
import sys  # InkluDocs

start = time.time()

from Utils import inputPath, outputPath
from pdfixsdk import *
import uuid
from pathlib import Path

pdfix = GetPdfix()

# if not pdfix.GetAccountAuthorization().Authorize("Benutzer", "Seriennummer"):
#   print("dummy message: PDFix SDK not authorized")
# InkluDocs: Lizenz aus der Umgebung (27.08.2026), Vorlage-Schnipsel oben belassen.
_lu, _lk = os.environ.get("PDFIX_LICENSE_USER", ""), os.environ.get("PDFIX_LICENSE_KEY", "")
if _lu and _lk:
    try:
        if not pdfix.GetAccountAuthorization().Authorize(_lu, _lk):
            print("PDFix-Lizenz nicht angenommen: " + str(pdfix.GetError()), file=sys.stderr)
    except Exception as _e:
        print("PDFix-Lizenz: Fehler bei der Aktivierung: " + repr(_e), file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')
    parser.add_argument('-c', '--csv', required=True, help='CSV mit Kopfzeile Nummer;Name;Quickinfo;...')  # InkluDocs

    args = parser.parse_args()
    # print("args.input : ",args.input)
    # print("args.output : ",args.output)
    global aaadatei
    aaadatei = args.input
    doc = pdfix.OpenDoc(args.input, "")
    if not doc:  # InkluDocs
        print("PDF konnte nicht geoeffnet werden: " + str(pdfix.GetError()), file=sys.stderr)
        sys.exit(2)

    # InkluDocs: CSV-Pfad aus -c statt "<pdf>_formulararray.csv"
    pfadcsv = args.csv

    datenim = []
    with open(pfadcsv, newline="", encoding="utf-8") as csvfile:
    # with open(r"C:\Daten\20260226_python_scripte\datei_matrix_bitte_aendern.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for zeile in reader:
            datenim.append(zeile)
    # InkluDocs: Kopfzeile ausdruecklich uebergehen
    if datenim and datenim[0] and datenim[0][0] == "Nummer":
        datenim = datenim[1:]

    print("------------------------")
    num_fields = doc.GetNumFormFields()
    print("Anzahl Felder:", num_fields)

    gesetzt = 0            # InkluDocs
    gefunden = set()       # InkluDocs
    for i in range(num_fields):
        field = doc.GetFormField(i)
        for a in range(0, len(datenim)-0):
            if len(datenim[a]) < 3:   # InkluDocs: unvollstaendige Zeile ueberspringen
                continue
            if datenim[a][1] == field.GetFullName():
                gefunden.add(datenim[a][1])
                neuertooltipp = datenim[a][2]
                if not neuertooltipp.strip():   # InkluDocs: leer = Feld unangetastet lassen
                    continue
                obj = field.GetObject()
                page_obj = obj.Get("P")
                obj.PutString("TU", neuertooltipp)
                gesetzt += 1

    if not doc.Save(args.output, kSaveFull):  # InkluDocs: Fehler beim Speichern melden
        print("PDF konnte nicht gespeichert werden: " + str(pdfix.GetError()), file=sys.stderr)
        sys.exit(2)
    doc.Close()

    print("-------------------")
    # InkluDocs: Ergebniszeile + Konsistenzcheck
    nicht_gefunden = [z[1] for z in datenim if len(z) >= 3 and z[1] and z[1] not in gefunden]
    print("TU_APPLIED=%d FIELDS_FOUND=%d NOT_FOUND=%s" % (gesetzt, num_fields, "|".join(nicht_gefunden)))
    if nicht_gefunden:
        sys.exit(4)

import argparse
lfn_counter = 0
lfn_tagnummer = 0

matrix = []

main()

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")
