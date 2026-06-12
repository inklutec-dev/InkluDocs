# Linux-Anpassung von Jörg Heines Import-Script (Original: AltTag_Import_CSV.py)
# Historie:
#   - April 2026: input() auskommentiert, CSV-Pfad als CLI-Parameter (Server-Betrieb).
#   - 12.06.2026: Gehärtete Zuordnung (InkluDocs). Hintergrund und Verhalten siehe unten.
#
# WARUM DIE HÄRTUNG (12.06.2026):
# Das Original ordnete CSV-Zeilen rein positionell zu: n-te CSV-Zeile -> n-te Figure
# im StructTree-Walk. Die Spalte "laufende Nummer" wurde ignoriert. Das funktionierte
# nur, solange (a) für JEDE Figure eine Zeile existierte und (b) Export- und
# Import-Script die Figures in exakt derselben Reihenfolge zählten. Fehlte eine
# Zeile, verrutschten ALLE folgenden Alt-Texte stillschweigend auf falsche Bilder.
#
# NEUES VERHALTEN:
#   1. Zuordnung über die Spalte "laufende Nummer" (lfnr) statt blinder Position.
#      Lücken sind erlaubt: Figures ohne CSV-Zeile bleiben UNANGETASTET
#      (deren Original-Alt-Text aus der Quell-PDF überlebt).
#   2. Eine vorhandene Zeile mit leerem Alt-Text setzt Alt explizit auf ""
#      (Konvention für "dekorativ" — der Aufrufer entscheidet, siehe
#      pdfix_roundtrip._build_csv_rows). Bilder OHNE Text bekommen keine Zeile.
#   3. Die Figure-Zählung spiegelt exakt die Zählregel des Export-Scripts
#      (AltTag_Export_CSV_PNG.py): GetType(True) == "Figure" UND mindestens eine
#      Seitenzuordnung (page_num >= 0). Das Original zählte hier abweichend
#      ALLE Figures — eine latente Verschiebungsquelle bei Figures ohne Seite.
#   4. Lauter Abbruch (Exit-Code 4) statt stillem Fehler, wenn die CSV eine
#      lfnr enthält, die größer ist als die Anzahl gefundener Figures.
#      Das passiert z. B., wenn eine neue Version des Export-Scripts die
#      Traversierung ändert — dann soll der Export FEHLSCHLAGEN, nicht
#      falsche Texte auf falsche Bilder schreiben.
#   5. Maschinenlesbare Ergebniszeile auf stdout ("ALT_APPLIED=n FIGURES_FOUND=m"),
#      die der Wrapper (pdfix_roundtrip.import_alt_texts_pdfix) auswertet.

import csv
import argparse
import sys

from pdfixsdk import *
from pathlib import Path

pdfix = GetPdfix()

parser = argparse.ArgumentParser(description="Import Alt-Texte aus CSV in PDF (lfnr-basiert).")
parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')
parser.add_argument('-c', '--csv', required=True, help='CSV-Datei mit Alt-Texten (Heine-Format)')
args = parser.parse_args()

# CSV einlesen: {lfnr: alt_text}. Header- und Leerzeilen überspringen.
# Spalten (Heine-Format): 0=laufende Nummer, 1=Pfad, 2=Titel, 3=Echter Text, 4=Alt-Text, 5=Dateiname
alt_by_lfnr = {}
with open(args.csv, newline="", encoding="utf-8") as csvfile:
    for zeile in csv.reader(csvfile, delimiter=";"):
        if not zeile or zeile[0] in ("laufende Nummer", ""):
            continue
        if len(zeile) < 5:
            continue
        try:
            lfnr = int(zeile[0])
        except ValueError:
            print("WARNUNG: Zeile mit ungueltiger laufender Nummer uebersprungen: %s" % zeile[:2],
                  file=sys.stderr)
            continue
        if lfnr in alt_by_lfnr:
            print("FEHLER: laufende Nummer %d kommt in der CSV doppelt vor." % lfnr, file=sys.stderr)
            sys.exit(4)
        alt_by_lfnr[lfnr] = zeile[4]

figure_counter = 0   # zählt Figures in StructTree-Reihenfolge — MUSS der Zählung des Export-Scripts entsprechen
applied_counter = 0  # tatsächlich gesetzte Alt-Texte


def _figure_page_num(elem):
    """Seitenzuordnung exakt wie im Export-Script ermitteln (letzte Seite der Schleife)."""
    page_num = -1
    for i in range(elem.GetNumPages()):
        page_num = elem.GetPageNumber(i)
    return page_num


def csvimport_struct_elem(elem):
    global figure_counter, applied_counter
    # Zählregel identisch zum Export-Script: nur Figures MIT Seitenzuordnung
    # bekommen eine laufende Nummer.
    if elem.GetType(True) == "Figure" and _figure_page_num(elem) >= 0:
        figure_counter += 1
        if figure_counter in alt_by_lfnr:
            elem.SetAlt(alt_by_lfnr[figure_counter])
            applied_counter += 1
        # Keine CSV-Zeile fuer diese Figure -> bewusst unangetastet lassen
        # (Original-Alt-Text der Quell-PDF bleibt erhalten).

    for i in range(elem.GetNumChildren()):
        if elem.GetChildType(i) == kPdsStructChildElement:
            obj = elem.GetChildObject(i)
            child_elem = elem.GetStructTree().GetStructElementFromObject(obj)
            csvimport_struct_elem(child_elem)


def main():
    doc = pdfix.OpenDoc(args.input, "")
    if not doc:
        print("FEHLER: PDF konnte nicht geoeffnet werden: %s" % args.input, file=sys.stderr)
        sys.exit(2)
    struct_tree = doc.GetStructTree()
    if not struct_tree:
        print("FEHLER: PDF hat keinen StructTree (nicht getaggt).", file=sys.stderr)
        sys.exit(2)
    for i in range(struct_tree.GetNumChildren()):
        obj = struct_tree.GetChildObject(i)
        elem = struct_tree.GetStructElementFromObject(obj)
        csvimport_struct_elem(elem)

    # Konsistenz-Check: jede CSV-lfnr muss auf eine real existierende Figure zeigen.
    # Sonst stimmt die Zuordnung Export <-> Import nicht mehr -> laut abbrechen,
    # NICHT speichern (lieber gar kein Export als ein stiller Versatz).
    out_of_range = [n for n in alt_by_lfnr if n < 1 or n > figure_counter]
    if out_of_range:
        print(
            "FEHLER: CSV referenziert laufende Nummern %s, aber der StructTree "
            "enthaelt nur %d zaehlbare Figures. Export- und Import-Traversierung "
            "passen nicht zusammen — Abbruch ohne Speichern."
            % (sorted(out_of_range), figure_counter),
            file=sys.stderr)
        sys.exit(4)

    if not doc.Save(args.output, kSaveFull):
        print("FEHLER: PDF konnte nicht gespeichert werden: %s" % args.output, file=sys.stderr)
        sys.exit(2)

    # Maschinenlesbares Ergebnis fuer den Wrapper (pdfix_roundtrip).
    print("ALT_APPLIED=%d FIGURES_FOUND=%d" % (applied_counter, figure_counter))
    print("fertig -", args.output)


main()
