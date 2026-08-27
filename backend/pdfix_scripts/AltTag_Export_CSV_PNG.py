# =============================================================================
#  AltTag_Export_CSV_PNG.py  —  InkluDocs PDFix-Export (getaggte PDFs)
# =============================================================================
#  HERKUNFT / PROVENANCE (fuer spaetere Code-Reviews wichtig):
#
#  Dieses Skript ist eine ZUSAMMENFUEHRUNG mehrerer Staende:
#
#   (A) BASIS = unsere Linux-Anpassung von Joerg Heines/Michael Karbes Export-
#       Skript, Stand "Karbe V1002" (27.05.2026). Davon stammt:
#         - StructTree-Walk + Figure-Erkennung
#         - Figure-Bild-Rendering (BoundingBox -> PNG)
#         - die CSV (Spalten 1-8)
#         - UNSERE Zusatzverbesserung: Seitenansicht-PNG 1x pro Seite gecacht,
#           in 144 DPI (2.0x), _render_page_view(). Die hat Karbes Vorlage NICHT.
#         - Linux-Anpassungen: kein input(), Pfade via CLI --data, kein Lizenz-
#           Block (Trial-/Wasserzeichen-Modus bis Verkaufsstart).
#
#   (B) Karbe V1004 (Mail 17.06.2026): die tag-basierte Text-Extraktion mit den
#       zwei Ausgaben je Bild, die als CSV-Spalten 9 + 10 an die KI gehen:
#         - "Seiteninhalt": Text DER SEITE, auf der das Bild steht.
#         - "Kontext":      Text des ABSCHNITTS um das Bild (Ueberschrift bis
#           Ueberschrift, kann mehrere Seiten umfassen) = enriched_context.
#       V1004 holte den Text ueber Wort-BoundingBoxen (AcquireWordList) — seit
#       (D) ERSETZT durch die praezisere MCID-Methode, siehe unten.
#
#   (C) Karbe V1.05 (Mail 24.06.2026): Lesbarkeit des ANGEZEIGTEN Seiteninhalts
#       — Zeilenumbruch hinter jedem Tag, Listen-Label bleiben inline.
#       Umgesetzt in _join_reading_order(), genutzt von _page_content().
#
#   (D) Karbe V1006 (Mail 07.07.2026, "Script Update 1006"): BUGFIX Doppelungen.
#       Bei Absaetzen ueber SPALTEN- oder SEITENUMBRUECHE hinweg las die alte
#       Wort-BoundingBox-Methode fremden Text mit ein (das Tag-Rechteck wird
#       bei solchen Absaetzen zu gross und faengt Nachbarspalten ein) — Texte
#       erschienen doppelt. Kern der V1006, hier uebernommen:
#         1. MCID-METHODE statt Wort-Rechtecken: jedes Struktur-Tag kennt die
#            MCIDs (Marked-Content-IDs) seiner Inhalte; wir nehmen von den
#            Text-Objekten der Seite exakt jene, deren MCID zum Tag gehoert.
#            Praezise Zuordnung, keine geometrischen Fehltreffer mehr.
#            (Gemessen am hofor-Jahresbericht, 92 Bilder, mehrspaltig:
#            Doppelungen Seiteninhalt 851 -> 14, Kontext 303 -> 1; der Rest
#            sind echte Wiederholungen im Dokument.)
#         2. Seitenuebergreifende Tags erzeugen PRO SEITE einen tagarray-
#            Eintrag (nur mit dem Textanteil dieser Seite); im Kontext werden
#            die Teile per Fortsetzungs-Regel wieder zusammengefuegt.
#         3. Tag-Typen "Link" und "Span" liefern jetzt auch Text (noetig bei
#            der MCID-Methode, sonst ginge verlinkter Text verloren).
#         4. Bereinigung weicher Trennzeichen (U+00AD) und Silbentrennung.
#         5. Fussnoten ("Note"-Teilbaeume) fliegen aus dem KI-Kontext raus.
#         6. Der KI-Kontext bekommt wie der Seiteninhalt Zeilenumbrueche
#            zwischen den Absaetzen (Absatzgrenzen sichtbar fuer die KI;
#            Entscheidung Steve 07.07.2026 — vorher bewusst flach).
#
#       UNSERE ABWEICHUNGEN von Karbes V1006-Code (Absicht identisch, alle am
#       07.07.2026 im A/B-Test gegen den alten Stand verifiziert):
#         a) Span/Link-Text wird REKURSIV dem umgebenden Text-Tag zugeschlagen
#            (_collect_mcids) statt als eigene Zeile ausgegeben — sonst wuerde
#            z.B. "Besuchen Sie [unsere Seite] heute" auseinandergerissen.
#         b) TABELLEN: Karbes Typenliste kennt kein TD/TH — Tabellentext waere
#            KOMPLETT weggefallen (Alt-Stand hatte ihn geometrisch mit drin;
#            hofor-Test: ueber 3.000 Zellen). Wir sammeln je Tabellenzeile
#            (TR) den Text ALLER Zellen als eine Zeile ein (_CONSUME_SUBTREE);
#            TD/TH einzeln nur als Fallback ohne umgebendes TR. Dazu "Caption"
#            (Bildunterschrift!) und "Note" (nur Seiteninhalt, nie Kontext).
#         c) SILBENTRENNUNG nur an FRAGMENTGRENZEN statt Karbes pauschalem
#            replace("- ",""): das Pauschale macht aus "rense- og" (daenisch)
#            bzw. "Vor- und" (deutsch) Kunstwoerter ("renseog"/"Vorund").
#            Wir fuegen nur zusammen, wenn ein Fragment mit "-" endet, und
#            lassen den Bindestrich stehen, wenn das naechste Fragment mit
#            einem Koppelwort beginnt (und/oder/og/eller/and/or) —
#            "Wasser- und Abwasser" bleibt intakt (_append_fragment).
#         d) Text-Objekte der Seite werden 1x pro Seite gecacht
#            (_page_text_objects), nicht pro Tag neu geholt.
#         e) Fussnoten-Erkennung ueber Rekursionstiefe (note_depth) statt
#            Pfad-String.
#
#  WICHTIGER VERIFIZIERTER BEFUND (07.07.2026, A/B im Staging-Container):
#       Auch die MCID-Methode liefert im PDFix-TRIAL-Modus (ohne Lizenz)
#       sauberen Text — KEINE '*'-Verstuemmelung (Sternchen im Output waren
#       echte Fussnoten-Zeichen des Dokuments). Getestet an
#       V105-Test-Actino.pdf + hofor_aar_2025 (daenisch, 92 Bilder). Die
#       Verstuemmelungs-Warnung vom 28.05.2026 betraf die PageMap-Methode.
#
#  Aufruf (unveraendert ggue. V1002-Linux):
#       python3 AltTag_Export_CSV_PNG.py -i <input.pdf> -o <egal.pdf> -d <outdir>
# =============================================================================

import os
import csv
import time
import argparse

start = time.time()

from Utils import inputPath, outputPath
from pdfixsdk import *
from pathlib import Path

pdfix = GetPdfix()

# Lizenz: Karbe besorgt sie zum Verkaufsstart. Jetzt Default-Modus (Wasserzeichen).
# if not pdfix.GetAccountAuthorization().Authorize("Benutzer", "Seriennummer"):
#     print("PDFix SDK not authorized")

# -----------------------------------------------------------------------------
#  TEIL A — unveraendert aus unserer V1002-Linux-Basis: Seitenansicht-Cache
# -----------------------------------------------------------------------------
_rendered_pages = {}     # page_num -> page_view_path (1 Seitenansicht pro Seite)
PAGE_VIEW_SCALE = 2.0    # ~144 DPI fuer lesbare Seitenansicht


import re


# 27.08.2026 (Befund Karbe, KBV_Formeln.pdf): Text aus Formel-/Symbolschriften
# kann ungepaarte UTF-16-Surrogate enthalten (kaputte Zeichenzuordnung im PDF).
# Die lassen sich nicht als UTF-8 schreiben -> csv.writer warf UnicodeEncodeError,
# der ganze PDFix-Export brach ab und InkluDocs fiel still auf fitz zurueck
# (1 Bild statt aller Figures). Jedes solche Zeichen wird durch U+FFFD ersetzt.
_SURROGAT = re.compile("[\ud800-\udfff]")


def _sauber(wert):
    return _SURROGAT.sub("\ufffd", wert) if isinstance(wert, str) else wert

def _render_page_view(page, page_num, crop_box):
    """Rendert die ganze Seite als PNG. Cache: 1x pro Seite (Schluessel page_num)."""
    if page_num in _rendered_pages:
        return _rendered_pages[page_num]
    pv = page.AcquirePageView(PAGE_VIEW_SCALE, kRotate0)
    dr = pv.RectToDevice(crop_box)
    dr.right -= dr.left
    dr.left = 0
    dr.bottom -= dr.top
    dr.top = 0
    img = pdfix.CreateImage(pv.GetDeviceWidth(), pv.GetDeviceHeight(), kImageDIBFormatArgb)
    rp = PdfPageRenderParams()
    rp.clip_box = crop_box
    rp.image = img
    rp.matrix = pv.GetDeviceMatrix()
    page.DrawContent(rp)
    path = os.path.join(data_dir, f"{filename2}_p{page_num + 1}_seitenansicht.png")
    ip = PdfImageParams()
    img.SaveRect(path, ip, dr)
    _rendered_pages[page_num] = path
    return path


# -----------------------------------------------------------------------------
#  TEIL B — tag-basierte Text-Extraktion (Karbe V1004, Methode seit V1006: MCID)
# -----------------------------------------------------------------------------
# Welche Tag-Typen tragen Text?
#   P/H*/LBody/Lbl:  seit V1004 (Karbe)
#   Link/Span:       NEU in V1006 (Karbe)
#   TR/TD/TH/Caption/Note: UNSERE Ergaenzung (Abweichung b im Kopf) — sonst
#                    fiele Tabellen-/Bildunterschrift-Text weg, den der alte
#                    geometrische Ansatz mitgeliefert hat.
_TEXT_TAGS = {"P", "H", "H1", "H2", "H3", "H4", "H5", "H6", "LBody", "Lbl",
              "Link", "Span", "TR", "TD", "TH", "Caption", "Note"}
_HEADING_TAGS = {"H", "H1", "H2", "H3", "H4", "H5", "H6"}
# Inline-Tags: ihr Text gehoert in den Lesefluss des UMGEBENDEN Tags. Ihre
# MCIDs sammelt _collect_mcids() beim Eltern-Tag ein; als eigenstaendige Zeile
# erscheinen sie nur, wenn sie NICHT in einem Text-Tag stecken (Randfall).
_INLINE_TEXT_TAGS = {"Span", "Link"}
# Teilbaum-Sammler: eine Tabellenzeile (TR) sammelt den Text ALLER Nachfahren
# (TD/TH samt deren P/Span/...) zu EINER Zeile ein. Die Nachfahren werden im
# Walk als "konsumiert" markiert und legen keine eigenen Zeilen mehr an.
_CONSUME_SUBTREE = {"TR"}

# Konsum-Stufen (Parameter "consumed" im Walk):
_C_NO = 0        # Element darf eigene Textzeile(n) anlegen
_C_INLINE = 1    # von Eltern-Text-Tag als Inline-Kind (Span/Link) eingesammelt
_C_SUBTREE = 2   # im Teilbaum eines Sammlers (TR) — gilt fuer ALLE Nachfahren

_doc = None              # Haupt-Dokumenthandle (in main() gesetzt)
_page_cache = {}         # page_num -> Page (mehrfach-Acquire vermeiden)
_page_objects_cache = {} # page_num -> [(mcid, text), ...] in Inhalts-Reihenfolge

# tagarray: in DFS-Reihenfolge ein Eintrag pro Struktur-Element UND SEITE
# (seit V1006: ein Tag ueber 2 Seiten -> 2 Eintraege mit demselben "serial").
#   tag:    Tag-Typ (z.B. "P", "H1", "TR", "Figure")
#   page:   Seitennummer (0-basiert), -1 wenn unbekannt
#   text:   extrahierter Text dieses Tags AUF DIESER SEITE ("" bei Nicht-Text)
#   note:   True, wenn das Tag in einer Fussnote steckt (Vorfahre/selbst "Note")
#   serial: laufende Element-Nummer — gleiche Nummer = gleiches Element
#           (erkennt seitenuebergreifende Absaetze beim Zusammenfuegen)
tagarray = []
# figure_rows: Merker je Figure, um Kontext/Seiteninhalt der CSV-Zeile zuzuordnen.
figure_rows = []         # [{"lfnr": int, "page_num": int, "tagidx": int}, ...]
_elem_serial = 0


def _get_page(page_num):
    p = _page_cache.get(page_num)
    if p is None:
        p = _doc.AcquirePage(page_num)
        _page_cache[page_num] = p
    return p


def _page_text_objects(page_num):
    """Alle Text-Objekte einer Seite als [(mcid, text), ...] in der Reihenfolge
    des Inhaltsstroms. 1x pro Seite gecacht (Abweichung d; Karbes V1006 laeuft
    pro Tag neu ueber den ganzen Seiteninhalt)."""
    objs = _page_objects_cache.get(page_num)
    if objs is None:
        content = _get_page(page_num).GetContent()
        objs = []
        for dd in range(content.GetNumObjects()):
            obj = content.GetObject(dd)
            # GetObjectType()==1 ist Text (Wert aus Karbes V1006 uebernommen).
            if obj.GetObjectType() == 1:
                mcid = obj.GetMcid()
                if mcid != -1:
                    objs.append((mcid, _sauber(obj.GetText())))
        _page_objects_cache[page_num] = objs
    return objs


def _collect_mcids(elem, deep=False):
    """MCIDs eines Tags einsammeln.

    deep=False: eigene MCIDs + die der Inline-Kinder (Span/Link), rekursiv —
                so bleibt verlinkter Text an seiner Stelle im Absatz
                (Abweichung a im Kopf).
    deep=True:  MCIDs des GESAMTEN Teilbaums (fuer TR = Tabellenzeile,
                Abweichung b im Kopf).
    """
    mcids = set()
    for uu in range(elem.GetNumChildren()):
        mcid = elem.GetChildMcid(uu)
        if mcid != -1:
            mcids.add(mcid)
        elif elem.GetChildType(uu) == kPdsStructChildElement:
            obj = elem.GetChildObject(uu)
            child = elem.GetStructTree().GetStructElementFromObject(obj)
            if deep or child.GetType(True) in _INLINE_TEXT_TAGS:
                mcids |= _collect_mcids(child, deep)
    return mcids


# Koppelwoerter: beginnt das Folge-Fragment so, ist ein Bindestrich am
# Fragment-Ende KEINE Silbentrennung, sondern eine Wortkopplung wie
# "Wasser- und Abwasser" / "rense- og spildevand" -> Bindestrich BLEIBT.
_KOPPELWOERTER = ("und ", "oder ", "og ", "eller ", "and ", "or ", "&")


def _append_fragment(out, frag):
    """Fragment-Verkettung mit Silbentrennungs-Heuristik (Abweichung c).

    Endet der bisherige Text auf "-" oder ein weiches Trennzeichen (U+00AD),
    stammt das in aller Regel von einem Zeilen-/Spaltenumbruch mitten im Wort:
    Trennzeichen entfernen und OHNE Leerzeichen anschliessen ("kom-" +
    "mercielt" -> "kommercielt"). AUSSER das neue Fragment beginnt mit einem
    Koppelwort ("Vor-" + "und Nachteile" -> "Vor- und Nachteile").
    Karbes V1006 loest das pauschal mit replace("- ","") und zerstoert damit
    Wortkopplungen; deshalb hier gezielter.
    """
    if not out:
        return frag
    if out.endswith("\xad"):
        return out[:-1] + frag
    if out.endswith("-") and frag and not frag.lower().startswith(_KOPPELWOERTER):
        return out[:-1] + frag
    if out.endswith(" "):
        return out + frag
    return out + " " + frag


def _tag_text(elem, page_num, deep=False):
    """Text EINES Tags auf EINER Seite ueber die MCID-Zuordnung (Karbe V1006).

    Vorher (V1004): Wort-BoundingBox-Treffer — bei mehrspaltigen oder seiten-
    uebergreifenden Absaetzen wurde das Tag-Rechteck zu gross und fing fremden
    Text ein -> Doppelungen. Die MCID-Zuordnung ist die im PDF hinterlegte,
    exakte Verknuepfung Tag -> Inhalt und hat dieses Problem nicht.
    """
    mcids = _collect_mcids(elem, deep)
    if not mcids:
        return ""
    out = ""
    for mcid, text in _page_text_objects(page_num):
        if mcid in mcids and text != " ":
            out = _append_fragment(out, text)
    # Uebrig gebliebene weiche Trennzeichen MITTEN in Fragmenten entfernen
    # (unsichtbares Layout-Zeichen, gehoert nicht in den KI-Text).
    out = out.rstrip().replace("\xad", "")
    return out


# -----------------------------------------------------------------------------
#  Struktur-Walk: rendert Figures (Teil A) UND sammelt Tag-Texte (Teil B)
# -----------------------------------------------------------------------------
def process_struct_elem(elem: PdsStructElement, note_depth=0, consumed=_C_NO):
    """DFS ueber den Tag-Baum.

    note_depth: >0 = wir sind innerhalb eines "Note"-Elements (Fussnote);
                solche Texte bleiben im Seiteninhalt, fliegen aber aus dem
                KI-Kontext (_chapter_context) raus. (Karbe V1006)
    consumed:   _C_INLINE/_C_SUBTREE = der Text dieses Elements wurde bereits
                vom umgebenden Tag via _collect_mcids eingesammelt -> keine
                eigene Textzeile; Kinder laufen trotzdem weiter (falls z.B.
                eine Figure in einem Link oder einer Tabellenzelle steckt).
    """
    global _elem_serial
    _elem_serial += 1
    serial = _elem_serial

    etype = elem.GetType(True)
    is_note = note_depth > 0 or etype == "Note"
    emits_text = etype in _TEXT_TAGS and consumed == _C_NO
    deep = emits_text and etype in _CONSUME_SUBTREE

    # --- Teil B: tagarray-Eintraege anlegen (seit V1006: einen JE SEITE) ---
    page_num = -1
    text_rows = 0
    if emits_text:
        for i in range(elem.GetNumPages()):
            page_num = elem.GetPageNumber(i)   # kann -1 sein (kaputtes Tag)
            if page_num < 0:
                continue
            tagarray.append({"tag": etype, "page": page_num,
                             "text": _tag_text(elem, page_num, deep),
                             "note": is_note, "serial": serial})
            text_rows += 1
    if text_rows == 0:
        # Figures, Nicht-Text-Tags, konsumierte Tags: Platzhalter ohne Text.
        # Figures dienen als Anker fuer die Kapitel-Suche, deshalb auch sie
        # in den tagarray.
        for i in range(elem.GetNumPages()):
            page_num = elem.GetPageNumber(i)
        tagarray.append({"tag": etype, "page": page_num, "text": "",
                         "note": is_note, "serial": serial})
    this_tagidx = len(tagarray) - 1

    # --- Teil A: Figure rendern (unveraendert aus V1002-Linux) ---
    if etype == "Figure" and page_num >= 0:
        bbox = elem.GetBBox(page_num)
        bboxfigure = elem.GetBBox(page_num)
        doc = pdfix.OpenDoc(aaadatei, "")
        page = doc.AcquirePage(page_num)
        crop_box = page.GetCropBox()

        pageView = page.AcquirePageView(1.0, kRotate0)
        devRect = pageView.RectToDevice(bboxfigure)
        devRect.right -= devRect.left
        devRect.left = 0
        devRect.bottom -= devRect.top
        devRect.top = 0
        psImage = pdfix.CreateImage(pageView.GetDeviceWidth(), pageView.GetDeviceHeight(), kImageDIBFormatArgb)
        renderParams = PdfPageRenderParams()
        renderParams.clip_box = bbox
        renderParams.image = psImage
        renderParams.matrix = pageView.GetDeviceMatrix()
        page.DrawContent(renderParams)

        global lfn_counter
        lfn_counter = lfn_counter + 1
        figure_path = os.path.join(data_dir, f"{filename2}_ExtractImages_{lfn_counter}.png")
        imageParams = PdfImageParams()
        psImage.SaveRect(figure_path, imageParams, devRect)

        page_view_path = _render_page_view(page, page_num, crop_box)

        matrix.append([lfn_counter, figure_path,
                       elem.GetTitle(), elem.GetActualText(), elem.GetAlt(),
                       filename2, page_num + 1, page_view_path])
        # Merker fuer die spaetere Kontext-Zuordnung (Teil B)
        figure_rows.append({"lfnr": lfn_counter, "page_num": page_num, "tagidx": this_tagidx})

    # Kinder rekursiv. Konsum-Stufe weiterreichen:
    #   - im TR-Teilbaum (oder darunter) ist ALLES konsumiert (_C_SUBTREE),
    #   - Inline-Kinder (Span/Link) eines Text-Tags sind konsumiert (_C_INLINE);
    #     deren NICHT-inline Kinder (Randfall, z.B. P in einem Link) legen
    #     wieder eigene Zeilen an — ihre MCIDs wurden oben nicht eingesammelt.
    for i in range(elem.GetNumChildren()):
        child_type = elem.GetChildType(i)
        if child_type == kPdsStructChildElement:
            obj = elem.GetChildObject(i)
            child_elem = elem.GetStructTree().GetStructElementFromObject(obj)
            if deep or consumed == _C_SUBTREE:
                child_consumed = _C_SUBTREE
            elif (emits_text or consumed == _C_INLINE) and \
                    child_elem.GetType(True) in _INLINE_TEXT_TAGS:
                child_consumed = _C_INLINE
            else:
                child_consumed = _C_NO
            process_struct_elem(child_elem,
                                note_depth + (1 if etype == "Note" else 0),
                                child_consumed)


# -----------------------------------------------------------------------------
#  Teil B: Seiteninhalt + Kontext berechnen (nach dem Walk)
# -----------------------------------------------------------------------------
def _join_reading_order(typed_parts):
    """Fuegt Tag-Texte in Lesereihenfolge zu lesbarem Seitentext zusammen.

    Karbe V1.05 (24.06.2026): hinter jedem Tag ein Zeilenumbruch, damit die
    ANZEIGE des Seiteninhalts strukturiert erscheint (Absatz/Ueberschrift/
    Listeneintrag/Tabellenzeile = eigene Zeile).

    AUSNAHME (1:1 aus Karbes Vorlage): Listen-Label (Tag-Typ "Lbl") und
    einzelne Aufzaehlungszeichen (Text aus genau einem Zeichen, z.B. "•")
    bleiben INLINE, d.h. Leerzeichen statt Umbruch — das Label "1." / "•"
    klebt am Anfang seines Listeneintrags.

    typed_parts: Liste von (tag_typ, text) in Lesereihenfolge.
    """
    out = []
    for etype, text in typed_parts:
        inline = (etype == "Lbl") or (len(text) == 1)
        out.append(text + (" " if inline else "\n"))
    return "".join(out).strip()


def _page_content(page_num):
    """Text aller Text-Tags auf einer Seite (Lesereihenfolge = DFS-Reihenfolge).

    Wird im Werkzeug als "Seitentext" ANGEZEIGT (Weg: CSV-Spalte 9 ->
    pdf_processor page_text -> Frontend "Seitentext anzeigen", CSS
    white-space:pre-wrap). Seit V1006 traegt ein seitenuebergreifender Absatz
    hier automatisch nur noch seinen Anteil DIESER Seite (ein tagarray-Eintrag
    je Seite) — vorher stand der Absatz inkl. eingefangenem Fremdtext doppelt
    drin."""
    typed = [(r["tag"], r["text"]) for r in tagarray
             if r["page"] == page_num and r["text"]]
    return _join_reading_order(typed)


def _chapter_context(tagidx):
    """Text des Abschnitts um den Figure-Tag: von der naechsten Ueberschrift
    OBERHALB bis zur naechsten Ueberschrift UNTERHALB (exklusive). Geht als
    enriched_context an die KI (pdf_processor _ctx), wird nirgends angezeigt.

    Seit V1006 (vorher bewusst flacher " ".join, Umstellung Steve 07.07.2026):
      - Zeilenumbruch zwischen den Tags (Absatzgrenzen fuer die KI sichtbar),
        Listen-Label/Aufzaehlungszeichen inline wie in _join_reading_order().
      - FORTSETZUNGS-REGEL: Folgt ein Eintrag DESSELBEN Elements (gleiches
        serial = seitenuebergreifender Absatz), werden die Teile wieder zu
        EINEM Text verbunden — inkl. Silbentrennungs-Heuristik am Uebergang
        (_append_fragment), falls der erste Teil mitten im Wort endet.
      - Fussnoten (note=True) bleiben draussen: sie stehen im Tag-Baum oft
        weit weg von ihrer Seite und wuerden den Abschnitts-Text verfaelschen.
    """
    start = 0
    for a in range(tagidx, -1, -1):
        if tagarray[a]["tag"] in _HEADING_TAGS:
            start = a
            break
    end = len(tagarray)
    for b in range(tagidx + 1, len(tagarray)):
        if tagarray[b]["tag"] in _HEADING_TAGS:
            end = b
            break
    out = ""
    pending_join = False   # naechster Eintrag ist Fortsetzung desselben Elements
    for z in range(start, end):
        r = tagarray[z]
        if r["note"] or not r["text"]:
            pending_join = False
            continue
        if pending_join:
            # Uebergang Seite->Seite desselben Absatzes: _append_fragment
            # fuegt zusammen (inkl. Silbentrennung am Uebergang).
            out = _append_fragment(out, r["text"])
        else:
            out += r["text"]
        inline = (r["tag"] == "Lbl") or (len(r["text"]) == 1)
        pending_join = (z + 1 < end
                        and tagarray[z + 1]["serial"] == r["serial"])
        if not pending_join:
            out += " " if inline else "\n"
    return out.strip()


def main():
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')
    parser.add_argument('-d', '--data', required=True, help='LINUX: Output dir for PNGs and CSV')

    args = parser.parse_args()
    global aaadatei, data_dir, _doc
    aaadatei = args.input
    data_dir = args.data
    os.makedirs(data_dir, exist_ok=True)
    _doc = pdfix.OpenDoc(args.input, "")
    path = Path("" + args.input)
    global filename2
    filename2 = path.stem
    struct_tree = _doc.GetStructTree()
    for i in range(struct_tree.GetNumChildren()):
        obj = struct_tree.GetChildObject(i)
        elem = struct_tree.GetStructElementFromObject(obj)
        process_struct_elem(elem)
    print(lfn_counter, " Bilder + ", len(_rendered_pages), " Seitenansichten extrahiert")


lfn_counter = 0
matrix = []
# CSV-Kopf: Spalten 1-8 wie bisher, Spalten 9-10 (Seiteninhalt, Kontext)
matrix.append(["laufende Nummer", "Pfad mit Dateinamen", "Titel",
               "Echter Text", "Alternativer Text", "Dateiname",
               "Seitennummer", "Pfad Seitenansicht",
               "Seiteninhalt", "Kontext"])
matrix.append([])

main()

# --- Teil B: Seiteninhalt + Kontext je Figure an die CSV-Zeilen anhaengen ---
context_by_lfnr = {}
for fr in figure_rows:
    si = _page_content(fr["page_num"])
    ko = _chapter_context(fr["tagidx"])
    context_by_lfnr[fr["lfnr"]] = (si, ko)

for row in matrix[1:]:
    if not row:           # die leere Trennzeile unveraendert lassen
        continue
    lfnr = row[0]
    si, ko = context_by_lfnr.get(lfnr, ("", ""))
    row.append(si)
    row.append(ko)

pfadcsv = os.path.join(data_dir, "figure_array.csv")
with open(pfadcsv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows([[_sauber(zelle) for zelle in zeile] for zeile in matrix])

end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")
print("CSV:", pfadcsv)
