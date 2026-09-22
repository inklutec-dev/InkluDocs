#!/usr/bin/env python3
"""Erzeugt die InkluDocs-Betriebsfassung von Joerg Heines Make_Accessible_01.py MECHANISCH
(22.09.2026): Original + markierte Zeilen, nach denselben Regeln wie baue_betriebsfassung.py
(Formular-Export) und tests/test_pdfix_skript_drift.py:
  - Kopfblock bis zur Zeile "# === InkluDocs-Kopf Ende ===" ist unser Kommentar.
  - "# InkluDocs-Original: <Zeile>" ersetzt eine Originalzeile (die Ersatzzeile traegt "# InkluDocs").
  - Jede sonstige Zeile mit "# InkluDocs" ist von uns ergaenzt.
  - Alle anderen Zeilen sind unveraendert Heines Original.
Aufruf: baue_make_accessible.py <original> <ziel>"""
import sys

ORIG, ZIEL = sys.argv[1], sys.argv[2]
zeilen = open(ORIG, encoding="utf-8").read().split("\n")

KOPF = """# =============================================================================
#  Make_Accessible.py — InkluDocs PDF-Tagging (PDF barrierefrei machen)
# =============================================================================
#  HERKUNFT / PROVENANCE:
#
#  Dies ist Joerg Heines Skript "Make_Accessible_01.py" (Actino Software GmbH,
#  heine@actino.de), Version 1.0.0.0 vom 21.09.2026, gesendet an
#  kontakt@inklutec.de ("Skript fuer das Ausfuehren von Make Accessible"). Das
#  unveraenderte Original liegt daneben unter original_heine/Make_Accessible_01.py.
#  Es laedt die in PDFix eingebaute Aktion "make_accessible" (37 Teilschritte:
#  Aufraeumen, Tags hinzufuegen, Tabellen/Ueberschriften reparieren, Titel,
#  Sprache, PDF/UA-Kennung, Lesezeichen) und fuehrt sie auf einer PDF aus.
#
#  REGEL (Steve 17.09.2026): Heines Skript ist die Vorlage, wir tragen nur einen
#  markierten Aufsatz fuer den Serverbetrieb darauf. Diese Datei entsteht
#  mechanisch aus dem Original (tests/werkzeuge/baue_make_accessible.py), und
#  tests/test_pdfix_skript_drift.py prueft, dass sie ohne die markierten Zeilen
#  byteidentisch mit dem Original ist. Markierungen:
#    "# InkluDocs-Original: <Zeile>"  ersetzt genau diese Originalzeile
#    "... # InkluDocs"                 von uns ergaenzte Zeile
#  Die Betriebslogik selbst steht in inkludocs_betrieb.py.
#
#  Anpassungen (22.09.2026):
#    1. Parameter -k/--konfig: Pfad einer JSON-Konfiguration der Aktion. Ohne
#       Parameter laeuft wie im Original die eingebaute Voreinstellung; mit
#       Parameter die von pdf_tagging.py je Lauf erzeugte Fassung (Dokument-
#       sprache aus dem Text, keine "Decorative"-Alt-Texte von PDFix — die
#       Alt-Texte schreibt InkluDocs selbst).
#    2. Lizenz ueber inkludocs_betrieb.lizenz_fuer_tagging: NUR aktiv bei
#       PDFIX_TAGGING_LIZENZ=on. Stand 22.09.2026 ist der Teilschritt add_tags
#       in der Actino-Lizenz nicht freigeschaltet (mit Lizenz bricht die Aktion
#       ab); ohne Lizenz taggt das SDK im Testmodus (Producer "Trial version").
#
#  Aufruf: python3 Make_Accessible.py -i <ein.pdf> -o <aus.pdf> [-k <konfig.json>]
#  Exit-Code 0 = gespeichert; sonst Traceback + "ERROR: <Grund>" auf stdout.
# =============================================================================
# === InkluDocs-Kopf Ende ==="""


def original(z):
    return "# InkluDocs-Original: " + z


out = KOPF.split("\n")
i = 0
ersetzt = {"konfig": 0, "commandPath": 0, "lizenz": 0, "import": 0}
while i < len(zeilen):
    z = zeilen[i]
    if z == "import Utils":
        out.append(z)
        out.append("import inkludocs_betrieb as betrieb  # InkluDocs")
        ersetzt["import"] += 1
    elif z == "parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')":
        out.append(z)
        out.append("parser.add_argument('-k', '--konfig', required=False, help='JSON-Konfiguration der Aktion (InkluDocs)')  # InkluDocs")
        ersetzt["konfig"] += 1
    elif z == 'commandPath = ""  # inputPath + "/make-accessible.json"':
        out.append(original(z))
        out.append('commandPath = args.konfig or ""  # InkluDocs')
        ersetzt["commandPath"] += 1
    elif z == '    raise Exception("Pdfix Initialization fail")':
        out.append(z)
        out.append("betrieb.lizenz_fuer_tagging(pdfix)  # InkluDocs")
        ersetzt["lizenz"] += 1
    else:
        out.append(z)
    i += 1

assert all(v == 1 for v in ersetzt.values()), ersetzt
open(ZIEL, "w", encoding="utf-8").write("\n".join(out))
print("geschrieben:", ZIEL, ersetzt)
