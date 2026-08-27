"""Erzeugt das FIKTIVE Testformular fuer das Quickinfo-Werkzeug (27.08.2026).

Aufruf (im Container oder mit installiertem PyMuPDF):
    python3 make_testformular.py [zielpfad]
Standardziel: testformular_inkludocs.pdf neben diesem Skript.

Alle Namen, Firmen und Daten sind erfunden (Musterbank Beispielstadt). Das
Formular deckt die Feldfaelle ab, die das Werkzeug unterscheiden muss:
  Seite 1: Textfelder mit Beschriftung links (Vorname, Nachname), Textfeld mit
           Beschriftung oberhalb (Anschrift), Pflichtfeld mit Sternchen und
           Formatangabe (Geburtsdatum), Textfeld MIT vorhandener Quickinfo
           (E-Mail), Checkbox mit Beschriftung rechts (Newsletter), Radio-Gruppe
           (Zahlungsweise: monatlich/jaehrlich), Dropdown (Anrede), ein
           ausgefuelltes Feld (Kundennummer, Wert darf nie exportiert werden).
  Seite 2: Abschnittsueberschrift "Zweiter Kontoinhaber" mit Vorname/Nachname
           (gleiche Beschriftung wie Seite 1, andere Feldnamen -> Stammdaten-
           Treffer ueber die Beschriftung), Unterschriftsfeld.
"""
import os
import re
import sys

import fitz  # PyMuPDF


def _radio_gruppe(doc, kids_mit_wert, name: str) -> None:
    """Macht aus einzelnen Checkbox-Widgets EINE Radio-Gruppe (Eltern-Feld
    /FT/Btn mit Radio-Flag, /Kids = die Widgets, Name nur am Elternteil).
    Jedes Widget bekommt statt des Zustands "Yes" seinen Exportwert als
    Erscheinungszustand (/AP/N/<wert>), verliert /T und bekommt /Parent; in
    /AcroForm/Fields ersetzt das Elternfeld die Einzelfelder. So sieht die
    Struktur aus, die Acrobat fuer Radio-Gruppen schreibt."""
    kid_xrefs = [x for x, _ in kids_mit_wert]
    parent = doc.get_new_xref()
    kids = " ".join(f"{x} 0 R" for x in kid_xrefs)
    # Ff Bit 16 (32768) = Radio, Bit 15 (16384) = NoToggleToOff
    doc.update_object(parent, f"<< /FT /Btn /Ff 49152 /T ({name}) /V /Off /Kids [ {kids} ] >>")
    for x, wert in kids_mit_wert:
        # Objekt als Text umbauen: xref_set_key(..., "null") hinterlaesst
        # "/T null" als Eintrag, und PDFix haelt ein Kind mit /FT-Schluessel
        # fuer das Feld selbst. Deshalb die Schluessel wirklich entfernen.
        obj = doc.xref_object(x, compressed=True)
        obj = re.sub(r"/(FT|T|Ff|V|AS)\s*(\([^)]*\)|/\w+|\d+)", "", obj)
        obj = obj.replace("/Yes", "/" + wert)                 # Erscheinungszustand = Exportwert
        obj = obj[:obj.rindex(">>")] + f"/AS/Off/Parent {parent} 0 R>>"
        doc.update_object(x, obj)
    # /AcroForm/Fields: die Einzelfelder durch das Elternfeld ersetzen. AcroForm
    # kann im Katalog eingebettet (dict) oder ein eigenes Objekt (xref) sein.
    cat = doc.pdf_catalog()
    typ, acro = doc.xref_get_key(cat, "AcroForm")
    if typ == "xref":
        holder, key = int(acro.split()[0]), "Fields"
    else:
        holder, key = cat, "AcroForm/Fields"
    ftyp, fields = doc.xref_get_key(holder, key)
    if ftyp == "xref":  # Fields-Array als eigenes Objekt
        holder, key = int(fields.split()[0]), None
        fields = doc.xref_object(holder)
    refs = []
    for teil in fields.strip("[] ").split("R"):
        teil = teil.strip()
        if teil and int(teil.split()[0]) not in kid_xrefs:
            refs.append(teil + " R")
    refs.append(f"{parent} 0 R")
    neu = "[ " + " ".join(refs) + " ]"
    if key is None:
        doc.update_object(holder, neu)
    else:
        doc.xref_set_key(holder, key, neu)


def erzeuge(ziel: str) -> str:
    doc = fitz.open()
    # ---------------------------------------------------------------- Seite 1
    p = doc.new_page(width=595, height=842)
    p.insert_text((50, 60), "Musterbank Beispielstadt – Kontoeröffnung (FIKTIVES TESTFORMULAR)", fontsize=13)
    p.insert_text((50, 95), "Angaben zum Kontoinhaber", fontsize=12)

    def textfeld(page, name, rect, label=None, label_pos="links", tooltip=None, value=None, required=False):
        if label:
            if label_pos == "links":
                page.insert_text((rect.x0 - 150, rect.y1 - 5), label, fontsize=10)
            elif label_pos == "oben":
                page.insert_text((rect.x0, rect.y0 - 6), label, fontsize=10)
        w = fitz.Widget()
        w.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        w.field_name = name
        w.rect = rect
        if tooltip:
            w.field_label = tooltip
        if value:
            w.field_value = value
        if required:
            w.field_flags = fitz.PDF_FIELD_IS_REQUIRED
        page.add_widget(w)

    textfeld(p, "vorname", fitz.Rect(200, 120, 500, 140), "Vorname")
    textfeld(p, "nachname", fitz.Rect(200, 150, 500, 170), "Nachname")
    textfeld(p, "anschrift", fitz.Rect(200, 200, 500, 220), "Straße und Hausnummer", label_pos="oben")
    textfeld(p, "geburtsdatum", fitz.Rect(200, 240, 500, 260), "Geburtsdatum (TT.MM.JJJJ) *", required=True)
    textfeld(p, "email", fitz.Rect(200, 270, 500, 290), "E-Mail", tooltip="E-Mail-Adresse für Kontoauszüge")
    textfeld(p, "kundennummer", fitz.Rect(200, 300, 500, 320), "Kundennummer", value="K-0000-TEST")

    # Checkbox, Beschriftung RECHTS vom Kaestchen
    cb = fitz.Widget()
    cb.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
    cb.field_name = "newsletter"
    cb.rect = fitz.Rect(200, 340, 214, 354)
    p.add_widget(cb)
    p.insert_text((222, 351), "Ich möchte den Newsletter erhalten", fontsize=10)

    # Radio-Gruppe Zahlungsweise. PyMuPDF 1.24 kann keine Radio-Widgets anlegen
    # ("bad xref" in add_widget); die Gruppe wird deshalb aus zwei Checkboxen
    # von Hand zu einer Radio-Gruppe umgebaut (Eltern-Feld mit Kids und
    # Radio-Flag, Exportwerte als Erscheinungszustaende — so wie Acrobat es
    # schreibt). Siehe _radio_gruppe().
    p.insert_text((50, 391), "Zahlungsweise", fontsize=10)
    radio_kids = []
    for i, (val, lab) in enumerate([("monatlich", "monatlich"), ("jaehrlich", "jährlich")]):
        r = fitz.Widget()
        r.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
        r.field_name = f"zw_tmp_{i}"
        r.rect = fitz.Rect(200 + i * 120, 380, 214 + i * 120, 394)
        annot = p.add_widget(r)
        radio_kids.append((annot.xref, val))
        p.insert_text((222 + i * 120, 391), lab, fontsize=10)

    # Dropdown Anrede
    p.insert_text((50, 431), "Anrede", fontsize=10)
    dd = fitz.Widget()
    dd.field_type = fitz.PDF_WIDGET_TYPE_COMBOBOX
    dd.field_name = "anrede"
    dd.rect = fitz.Rect(200, 418, 320, 438)
    dd.choice_values = ["Frau", "Herr", "keine Angabe"]
    p.add_widget(dd)

    # ---------------------------------------------------------------- Seite 2
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 60), "Zweiter Kontoinhaber", fontsize=12)
    textfeld(p2, "vorname_2", fitz.Rect(200, 100, 500, 120), "Vorname")
    textfeld(p2, "nachname_2", fitz.Rect(200, 130, 500, 150), "Nachname")
    p2.insert_text((50, 700), "Unterschrift Kontoinhaber", fontsize=10)
    # Unterschriftsfeld: PyMuPDF 1.24 kann keine Signaturfelder anlegen
    # (pdf_new_nt fehlt) -> Textfeld anlegen und per Objekt-Edit zu /FT/Sig machen.
    sig = fitz.Widget()
    sig.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    sig.field_name = "unterschrift"
    sig.rect = fitz.Rect(200, 680, 500, 720)
    sig_annot = p2.add_widget(sig)
    obj = doc.xref_object(sig_annot.xref, compressed=True)
    obj = re.sub(r"/(FT|Ff|V|DA)\s*(\([^)]*\)|/\w+|\d+)", "", obj)
    doc.update_object(sig_annot.xref, obj[:obj.rindex(">>")] + "/FT/Sig>>")

    _radio_gruppe(doc, radio_kids, "zahlungsweise")

    doc.set_metadata({"title": "Kontoeröffnung Musterbank (fiktives Testformular InkluDocs)"})
    doc.save(ziel, garbage=3, deflate=True)
    doc.close()
    return ziel


if __name__ == "__main__":
    ziel = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "testformular_inkludocs.pdf")
    print(erzeuge(ziel))
