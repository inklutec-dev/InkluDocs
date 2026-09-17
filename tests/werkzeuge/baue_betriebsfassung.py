#!/usr/bin/env python3
"""Erzeugt die InkluDocs-Betriebsfassung von Joerg Heines Formulare_Export_08.py MECHANISCH:
Original + markierte Zeilen. Regeln (dieselben, die tests/test_pdfix_skript_drift.py rueckwaerts anwendet):
  - Kopfblock bis zur Zeile "# === InkluDocs-Kopf Ende ===" ist unser Kommentar.
  - "# InkluDocs-Original: <Zeile>" ersetzt eine Originalzeile (die Originalzeile steht wortgleich dahinter).
  - Jede sonstige Zeile mit "# InkluDocs" ist von uns ergaenzt.
  - Alle anderen Zeilen sind unveraendert Heines Original.
Aufruf: baue_betriebsfassung.py <original> <ziel>"""
import sys

ORIG, ZIEL = sys.argv[1], sys.argv[2]
zeilen = open(ORIG, encoding="utf-8").read().split("\n")

KOPF = """# =============================================================================
#  Formular_Export_Quickinfo.py — InkluDocs Quickinfo-Werkzeug (PDF-Formulare)
# =============================================================================
#  HERKUNFT / PROVENANCE:
#
#  Dies ist Joerg Heines Skript "Formulare_Export_08.py" (Actino Software GmbH,
#  heine@actino.de), Version 1.0.0.2 vom 17.09.2026, gesendet an
#  steve.weidel@inklutec.de ("Script fuer den Export der QuickInfo bei
#  Formularfeldern"). Das unveraenderte Original liegt daneben unter
#  original_heine/Formulare_Export_08.py. Neu gegenueber Formulare_Export_07_r.py
#  (25.08.2026): Rechteck je Feld (left, bottom, right, top), Anzahl der Felder
#  mit identischem Namen, und die Sortierung je Seite von oben nach unten
#  (Toleranz 5) und links nach rechts mit fortlaufender Neunummerierung —
#  Michael Karbe 16.09.2026: die Reihenfolge der Quickinfos soll der sichtbaren
#  Reihenfolge der Felder entsprechen.
#
#  REGEL (Steve 17.09.2026): Heines Skript ist die Vorlage, wir tragen nur einen
#  markierten Aufsatz fuer den Serverbetrieb darauf. Diese Datei entsteht
#  mechanisch aus dem Original (tests/werkzeuge/baue_betriebsfassung.py), und
#  tests/test_pdfix_skript_drift.py prueft, dass sie ohne die markierten Zeilen
#  byteidentisch mit dem Original ist. Markierungen:
#    "# InkluDocs-Original: <Zeile>"  ersetzt genau diese Originalzeile
#    "... # InkluDocs"                 von uns ergaenzte Zeile
#  Die Betriebslogik selbst steht in inkludocs_betrieb.py.
#
#  Anpassungen (seit 27.08.2026, heute unveraendert uebernommen):
#    1. input("Druecke ENTER") entfaellt (kein stdin auf dem Server).
#    2. CSV-Pfad als Parameter -c/--csv; -o/--output nicht mehr Pflicht.
#    3. DATENSCHUTZ: Spalte "Value" nur "kein Wert" / "Wert vorhanden".
#    4. Lizenz ueber PDFIX_LICENSE_USER/PDFIX_LICENSE_KEY; Ergebniszeile FIELDS_FOUND=n.
#    5. Ungepaarte UTF-16-Surrogate in Name/Quickinfo werden ersetzt.
#    6. aufseiten/auf1seite/Rechteck vor der Verzweigung initialisiert; im
#       Kids-Zweig page.Release() je Seite.
#    7. doc.Close(); Fehlermeldung + Exit-Code 2, wenn die PDF nicht zu oeffnen ist.
#    8. NEU 17.09.2026: Vor Heines Sortierung werden leere Seiten-/Rechteckwerte
#       auf 0 gesetzt (sonst ValueError in int()), siehe seiten_absichern.
#
#  CSV-Format (Semikolon, UTF-8):
#    Nummer;Name;Quickinfo;Type-Nr;Type;Value;Seite;left;bottom2;right;top;Anzahl Felder mit identischem Namen
#  formular_processor.py liest die ersten sieben Spalten ueber feste Positionen.
# =============================================================================
# === InkluDocs-Kopf Ende ==="""

def original(z):
    return "# InkluDocs-Original: " + z

out = KOPF.split("\n")
i = 0
def take():
    global i
    z = zeilen[i]; i += 1; return z

while i < len(zeilen):
    z = zeilen[i]
    if z == 'input("Drücke ENTER, um fortzufahren...")':
        out.append(original(take()))
    elif z == "import copy":
        out.append(take()); out.append("import sys  # InkluDocs: fuer stderr/Exit-Code")
        out.append("import inkludocs_betrieb as betrieb  # InkluDocs: Betriebshelfer (Lizenz, Datenschutz, CSV-Sicherheit)")
    elif z == "pdfix = GetPdfix()":
        out.append(take()); out.append("betrieb.lizenz_aktivieren(pdfix)  # InkluDocs: Lizenz aus der Umgebung")
    elif z == "    parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')":
        out.append(original(take()))
        out.append("    parser.add_argument('-o', '--output', required=False, help='(unbenutzt, Kompatibilitaet)')  # InkluDocs")
        out.append("    parser.add_argument('-c', '--csv', required=True, help='Pfad der zu schreibenden CSV')  # InkluDocs")
    elif z == '    doc = pdfix.OpenDoc(args.input, "")':
        out.append(take()); out.append("    if not doc: sys.exit(betrieb.pdf_nicht_geoeffnet(pdfix))  # InkluDocs: klare Meldung statt AttributeError")
    elif z == "        if kids is not None:":
        out.append('        aufseiten = ""; auf1seite = ""; left = bottom = right = top = 0; anzfelder = 0  # InkluDocs: in jedem Zweig definiert')
        out.append(take())
    elif z == "                                    auf1seite = auf1seite+str(page_num + 1)       ":
        out.append(take()); out.append("                page.Release()  # InkluDocs: Seite wieder freigeben (Speicher bei grossen Formularen)")
    elif z == '        if feldwert == "" :':
        out.append(original(take())); out.append(original(take()))   # auch die Zeile 'feldwert = "kein Wert"'
        out.append("        feldwert = betrieb.feldwert_maskieren(feldwert)  # InkluDocs (Datenschutz): nie der Wert selbst")
    elif z.startswith("        fieldarray.append([(ff+1), feld1.GetFullName(), feld1.GetTooltip(),"):
        out.append(original(take()))
        out.append("        fieldarray.append([(ff+1), betrieb.sauber(feld1.GetFullName()), betrieb.sauber(feld1.GetTooltip()), feld1.GetType(), feldart, feldwert,auf1seite, left, bottom , right, top, anzfelder])  # InkluDocs: Surrogate-sicher")
    elif z == '    pfad5 = str(pfad3)+"\\\\"+filename2   ':
        out.append(take())
        out.append("    global pfadcsv  # InkluDocs")
        out.append("    pfadcsv = args.csv  # InkluDocs: CSV-Pfad aus -c statt Windows-Pfad neben der PDF")
        out.append("    doc.Close()  # InkluDocs")
    elif z == "daten = fieldarray[1:]":
        out.append(take()); out.append("daten = betrieb.seiten_absichern(daten)  # InkluDocs: leere Seite/Rechteck -> 0 statt ValueError")
    elif z.startswith('pfadcsv = r"C:') or z == 'pfadcsv = pfad5+"_formulararray.csv"':
        out.append(original(take()))
    elif z == 'print("Dauer:", round((end - start), 2), "Sekunden")':
        out.append(take()); out.append('print("FIELDS_FOUND=%d" % (len(fieldarray) - 1))  # InkluDocs: Ergebniszeile fuer den Wrapper')
    else:
        out.append(take())

open(ZIEL, "w", encoding="utf-8").write("\n".join(out))
print("geschrieben:", ZIEL, len(out), "Zeilen")
