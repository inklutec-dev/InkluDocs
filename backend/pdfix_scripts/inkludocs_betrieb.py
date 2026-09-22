"""Betriebshelfer fuer Joerg Heines PDFix-Skripte (InkluDocs, 17.09.2026).

Heines Skripte (Actino Software GmbH) bleiben inhaltlich unangetastet; alles, was nur der
Serverbetrieb bei InkluDocs braucht, steht hier und wird im Skript ueber einzelne, mit
"# InkluDocs" markierte Zeilen aufgerufen. tests/test_pdfix_skript_drift.py prueft, dass die
Betriebsfassung ohne diese markierten Zeilen byteidentisch mit Heines Original ist.
"""
import os
import sys


def lizenz_aktivieren(pdfix) -> None:
    """PDFix-Lizenz aus PDFIX_LICENSE_USER/PDFIX_LICENSE_KEY (wie in AltTag_Import_CSV.py).
    Ohne die Variablen laeuft das SDK als Testversion weiter; eine abgelehnte Lizenz bricht
    den Lauf nicht ab, sie wird nur auf stderr gemeldet."""
    lu, lk = os.environ.get("PDFIX_LICENSE_USER", ""), os.environ.get("PDFIX_LICENSE_KEY", "")
    if not (lu and lk):
        return
    try:
        if not pdfix.GetAccountAuthorization().Authorize(lu, lk):
            print("PDFix-Lizenz nicht angenommen: " + str(pdfix.GetError()), file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print("PDFix-Lizenz: Fehler bei der Aktivierung: " + repr(e), file=sys.stderr)


def lizenz_fuer_tagging(pdfix) -> None:
    """PDF-Tagging (22.09.2026, Make_Accessible.py): Der Teilschritt add_tags ist in der Actino-Lizenz
    (Stand 22.09.2026) NICHT freigeschaltet — mit aktivierter Lizenz bricht die Aktion bei ungetaggten
    PDFs ab ("Invalid initial element type"), ohne Lizenz taggt das SDK im Testmodus (Producer
    "Trial version of PDFix SDK"). Darum wird die Lizenz hier NUR bei PDFIX_TAGGING_LIZENZ=on aktiviert
    (sobald Actino/PDFix das Tagging freischalten). Alle anderen Skripte aktivieren sie immer."""
    an = os.environ.get("PDFIX_TAGGING_LIZENZ", "off").strip().lower() in ("on", "1", "true", "yes")
    if an:
        lizenz_aktivieren(pdfix)
    else:
        print("PDFix-Tagging im Testmodus (PDFIX_TAGGING_LIZENZ ist nicht gesetzt)", file=sys.stderr)


def pdf_nicht_geoeffnet(pdfix) -> int:
    """Klare Meldung + Exit-Code 2 statt AttributeError, wenn OpenDoc None liefert."""
    print("PDF konnte nicht geoeffnet werden: " + str(pdfix.GetError()), file=sys.stderr)
    return 2


def sauber(s):
    """Ungepaarte UTF-16-Surrogate ersetzen (encode/replace -> "?"), damit die CSV schreibbar bleibt
    (Befund KBV_Formeln.pdf 27.08.2026: sonst UnicodeEncodeError und Abbruch)."""
    if not isinstance(s, str):
        return s
    return s.encode("utf-8", "replace").decode("utf-8")


def feldwert_maskieren(feldwert) -> str:
    """DATENSCHUTZ: nie den eingetragenen Wert, nur ob einer vorhanden ist. "Off" ist bei
    Checkbox/Radio der Nicht-Ausgewaehlt-Zustand, also kein Wert."""
    return "kein Wert" if feldwert in ("", "Off", None) else "Wert vorhanden"


def seiten_absichern(daten: list) -> list:
    """Heines Sortierung (Version 1.0.0.2 vom 17.09.2026) rechnet mit int(Seite) und int(top).
    Ein Feld ohne gefundene Seite (leerer String) oder ohne Rechteck wuerde den ganzen Export
    mit ValueError abbrechen. Hier bekommt es Seite 0 und top 0 und landet damit vorn; die
    Seitenliste ermittelt InkluDocs ohnehin selbst ueber PyMuPDF (formular_processor.py)."""
    for zeile in daten:
        try:
            zeile[6] = int(zeile[6])
        except (TypeError, ValueError):
            zeile[6] = 0
        for idx in (7, 8, 9, 10):
            try:
                zeile[idx] = int(zeile[idx])
            except (TypeError, ValueError, IndexError):
                if len(zeile) > idx:
                    zeile[idx] = 0
    return daten
