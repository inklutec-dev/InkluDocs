"""Hoerprobe und Pruefbericht fuer Word-Dokumente (29.08.2026, Stufe 2 der
barrierefreien PDF aus Word).

Hoerprobe: der Text, den ein Screenreader aus dem Dokument liest — Titel,
Sprache, Ueberschriften mit Ebene, Absaetze, Listenpunkte, Bilder mit ihrem
Alt-Text (bzw. "Schmuckbild"), Tabellen mit Groesse und Kopfzeile. Gelesen wird
die EXPORTIERTE Word-Datei (mit unseren Alt-Texten), Teil word/document.xml in
Dokumentreihenfolge; Kopf-/Fusszeilen werden bewusst nicht vorgelesen (das tun
Screenreader in der PDF auch nicht, sie sind Artefakte).

Pruefbericht: was am Word-Dokument selbst die Barrierefreiheit der PDF
beschraenkt und was der Kunde in Word selbst richten kann — Titel, Sprache,
Ueberschriften-Hierarchie, Tabellen ohne Kopfzeile, Bilder ohne Alt-Text.
Reines Lesen, nichts wird veraendert.

Alle Texte gehen durch `_` (gettext), damit die Oberflaeche sie in der Sprache
des Nutzers zeigt.
"""
from __future__ import annotations

import re
import zipfile
from typing import Callable

from lxml import etree

from docx_processor import (NS, DECORATIVE_EXT_URI, DocxFehler, _pruefe_zip, _lese_xml, _text, _pstyle,
                            _heading_level, _styles, _dokumenttitel, _eigener_blip)

W = NS["w"]
_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_V = "urn:schemas-microsoft-com:vml"
_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"

MAX_ABSATZ = 400          # Zeichen je vorgelesenem Absatz (Hoerprobe soll ein Eindruck sein, kein Volltext)
MAX_ZEILEN = 400          # Zeilen der Hoerprobe insgesamt


def _identitaet(s: str) -> str:
    return s


def _sprache(zf: zipfile.ZipFile) -> str:
    """Dokumentsprache: w:lang in den Standardwerten von styles.xml, sonst erste w:lang im Text."""
    try:
        if "word/styles.xml" in zf.namelist():
            st = _lese_xml(zf, "word/styles.xml")
            for lang in st.iter(f"{{{W}}}lang"):
                v = lang.get(f"{{{W}}}val") or lang.get(f"{{{W}}}eastAsia")
                if v:
                    return v
        doc = _lese_xml(zf, "word/document.xml")
        for lang in doc.iter(f"{{{W}}}lang"):
            v = lang.get(f"{{{W}}}val")
            if v:
                return v
    except Exception:  # noqa: BLE001
        pass
    return ""


def _ist_dekorativ(docpr: etree._Element) -> bool:
    for ext in docpr.iter(f"{{{_A}}}ext"):
        if (ext.get("uri") or "").upper() == DECORATIVE_EXT_URI.upper():
            for kind in ext:
                if kind.get("val") in ("1", "true"):
                    return True
    return False


def _in_fallback(el: etree._Element) -> bool:
    p = el.getparent()
    while p is not None:
        if p.tag == f"{{{_MC}}}Fallback":
            return True
        p = p.getparent()
    return False


def _bilder_im_absatz(p: etree._Element) -> list[tuple[str, bool]]:
    """(alt, dekorativ) je Bild im Absatz, Dokumentreihenfolge; mc:Fallback-Duplikate ausgelassen."""
    out = []
    for docpr in p.iter(f"{{{_WP}}}docPr"):
        if _in_fallback(docpr):
            continue
        # Nur echte Bilder: der Rahmen eines Textfelds hat auch ein docPr, aber
        # keinen eigenen a:blip (das Bild darin zaehlt fuer sich, ueber txbxContent).
        container = docpr.getparent()
        if container is None or _eigener_blip(container) is None:
            continue
        out.append(((docpr.get("descr") or "").strip(), _ist_dekorativ(docpr)))
    for shape in p.iter(f"{{{_V}}}shape"):
        if _in_fallback(shape):
            continue
        if shape.find(f"{{{_V}}}imagedata") is None:
            continue
        alt = (shape.get("alt") or "").strip()
        out.append((alt, False))
    return out


def _tabelle(tbl: etree._Element) -> dict:
    zeilen = tbl.findall(f"{{{W}}}tr")
    spalten = 0
    kopf = False
    kopfzellen: list[str] = []
    for i, tr in enumerate(zeilen):
        zellen = tr.findall(f"{{{W}}}tc")
        spalten = max(spalten, len(zellen))
        if i == 0:
            trpr = tr.find(f"{{{W}}}trPr")
            kopf = trpr is not None and trpr.find(f"{{{W}}}tblHeader") is not None
            kopfzellen = [_text(tc).strip() for tc in zellen]
    return {"zeilen": len(zeilen), "spalten": spalten, "kopf": kopf, "kopfzellen": kopfzellen}


# Sprach-Abgleich (Michael Karbe 11.09.2026: Dokument auf da-DK gestellt, Text englisch —
# der Pruefbericht sagte „Sprache gesetzt“, der Screenreader haette daenisch ausgesprochen).
# Kleine Stoppwort-Zaehlung ueber den Anfang des Textes; nur bei klarem Abstand ein Hinweis.
_SPRACH_MARKER = {
    "de": {"der", "die", "das", "und", "nicht", "mit", "ist", "ein", "eine", "auf", "für", "wird", "sind", "auch", "werden", "sich", "dem", "den", "von", "zu"},
    "en": {"the", "and", "with", "is", "are", "for", "this", "that", "of", "to", "in", "you", "your", "not", "be", "as", "it", "on", "have", "will"},
    "da": {"og", "det", "er", "til", "af", "en", "at", "for", "med", "på", "som", "den", "der", "ikke", "kan", "skal", "har", "også", "eller", "være"},
    "fr": {"le", "la", "les", "et", "des", "est", "pour", "dans", "une", "que", "qui", "pas", "sur", "vous", "avec", "sont", "ce", "cette", "par", "plus"},
    "es": {"el", "la", "los", "las", "y", "es", "para", "que", "con", "una", "por", "del", "se", "no", "en", "más", "como", "son", "este", "esta"},
    "sv": {"och", "att", "det", "är", "för", "som", "med", "den", "till", "inte", "av", "ett", "kan", "på", "har", "eller", "ska", "vi", "de", "från"},
}
_SPRACH_NAMEN = {"de": "Deutsch", "en": "Englisch", "da": "Dänisch", "fr": "Französisch", "es": "Spanisch", "sv": "Schwedisch"}


def erkenne_sprache(text: str) -> tuple[str, int, int]:
    """Oeffentlicher Name der Spracherkennung (22.09.2026, auch fuer das PDF-Tagging, pdf_tagging.py):
    (Sprachkuerzel oder "", Treffer, Treffer der zweitbesten Sprache)."""
    return _text_sprache(text)


def _text_sprache(text: str) -> tuple[str, int, int]:
    """(Sprachkuerzel, Treffer, Treffer der zweitbesten) ueber die ersten ~600 Woerter."""
    woerter = re.findall(r"[a-zA-ZäöüÄÖÜßæøåÆØÅéèêàçñ]+", text.lower())[:600]
    if len(woerter) < 40:
        return "", 0, 0
    punkte = {k: sum(1 for w in woerter if w in m) for k, m in _SPRACH_MARKER.items()}
    best = max(punkte, key=punkte.get)
    rest = sorted(punkte.values(), reverse=True)
    return best, rest[0], (rest[1] if len(rest) > 1 else 0)


def analysiere(docx_path: str, _: Callable[[str], str] = _identitaet, titel_ersatz: str = "") -> dict:
    """Liefert {"hoerprobe": [zeilen], "pruefbericht": [{"status","text"}], "titel", "sprache"}.
    titel_ersatz (11.09.2026): was die PDF als Titel bekommt, wenn im Dokument keiner steht
    (erste Ueberschrift / Dokumentname) — damit Vorschau und Umwandlung dasselbe sagen."""
    try:
        zf = zipfile.ZipFile(docx_path)
    except zipfile.BadZipFile:
        raise DocxFehler("Keine gültige Word-Datei")
    hoer: list[str] = []
    befunde: list[dict] = []
    with zf:
        _pruefe_zip(zf)
        styles = _styles(zf)
        titel = _dokumenttitel(zf)
        sprache = _sprache(zf)
        doc = _lese_xml(zf, "word/document.xml")
        body = doc.find(f"{{{W}}}body")
        if body is None:
            raise DocxFehler("Word-Datei ohne Textkörper")

        # Kopf
        hoer.append(_("Dokumenttitel: {t}").format(t=titel) if titel else _("Dokumenttitel: fehlt"))
        hoer.append(_("Sprache: {s}").format(s=sprache) if sprache else _("Sprache: nicht gesetzt"))

        letzte_ebene = 0
        texte: list[str] = []      # Rohtext fuer den Sprach-Abgleich (unabhaengig von der UI-Sprache)
        n_ueberschriften = 0
        n_absaetze = 0
        n_bilder = 0
        n_bilder_ohne = 0
        n_dekorativ = 0
        n_tabellen = 0
        n_tabellen_ohne_kopf = 0
        spruenge = 0
        leere_ueberschriften = 0
        erste_ebene = None

        def _bilder(p):
            nonlocal n_bilder, n_bilder_ohne, n_dekorativ
            for alt, deko in _bilder_im_absatz(p):
                n_bilder += 1
                if deko:
                    n_dekorativ += 1
                    hoer.append(_("Schmuckbild (wird nicht vorgelesen)"))
                elif alt:
                    hoer.append(_("Bild: {alt}").format(alt=alt))
                else:
                    n_bilder_ohne += 1
                    hoer.append(_("Bild ohne Beschreibung — ein Screenreader sagt nur „Grafik“"))

        for el in body:
            if len(hoer) > MAX_ZEILEN:
                hoer.append(_("… (Hörprobe gekürzt)"))
                break
            if el.tag == f"{{{W}}}p":
                text = _text(el).strip()
                if text and len(texte) < 400:
                    texte.append(text)
                ebene = _heading_level(_pstyle(el), styles)
                if ebene is not None:
                    n_ueberschriften += 1
                    if not text:
                        leere_ueberschriften += 1
                        hoer.append(_("Leere Überschrift (Ebene {n})").format(n=max(ebene, 1)))
                    elif ebene == 0:
                        hoer.append(_("Titel: {t}").format(t=text))
                    else:
                        if erste_ebene is None:
                            erste_ebene = ebene
                        if letzte_ebene and ebene > letzte_ebene + 1:
                            spruenge += 1
                        letzte_ebene = ebene
                        hoer.append(_("Überschrift Ebene {n}: {t}").format(n=ebene, t=text))
                    _bilder(el)
                    continue
                ist_liste = el.find(f"{{{W}}}pPr/{{{W}}}numPr") is not None
                if text:
                    n_absaetze += 1
                    kurz = text if len(text) <= MAX_ABSATZ else text[:MAX_ABSATZ].rstrip() + " …"
                    hoer.append((_("Listenpunkt: {t}") if ist_liste else _("Absatz: {t}")).format(t=kurz))
                _bilder(el)
            elif el.tag == f"{{{W}}}tbl":
                n_tabellen += 1
                t = _tabelle(el)
                if not t["kopf"]:
                    n_tabellen_ohne_kopf += 1
                hoer.append(_("Tabelle mit {r} Zeilen und {c} Spalten").format(r=t["zeilen"], c=t["spalten"])
                            + (_(", Kopfzeile: {k}").format(k=", ".join(k for k in t["kopfzellen"] if k) or "—")
                               if t["kopf"] else _(", ohne Kopfzeile")))
                for p in el.iter(f"{{{W}}}p"):
                    _bilder(p)

        # Pruefbericht
        def ok(text):
            befunde.append({"status": "ok", "text": text})

        def hinweis(text):
            befunde.append({"status": "befund", "text": text})

        if titel:
            ok(_("Der Dokumenttitel ist gesetzt."))
        elif titel_ersatz:
            hinweis(_("Es fehlt ein Dokumenttitel (Datei → Informationen → Titel). Die PDF bekommt ersatzweise „{t}“ als Titel.").format(t=titel_ersatz))
        else:
            hinweis(_("Es fehlt ein Dokumenttitel (Datei → Informationen → Titel). Die PDF bekommt sonst nur den Dateinamen als Namen."))
        if not sprache:
            hinweis(_("Die Dokumentsprache ist nicht gesetzt. Screenreader wählen dann eine falsche Aussprache."))
        else:
            erkannt, treffer, zweite = _text_sprache(" ".join(texte))
            deklariert = sprache.split("-")[0].lower()
            if erkannt and erkannt != deklariert and treffer >= 12 and treffer >= 2 * max(zweite, 1) and deklariert in _SPRACH_NAMEN:
                hinweis(_("Die Dokumentsprache ist auf {s} gesetzt, der Text wirkt aber {name} — Screenreader würden ihn falsch aussprechen (Überprüfen → Sprache).").format(s=sprache, name=_(_SPRACH_NAMEN[erkannt])))
            else:
                ok(_("Die Dokumentsprache ist gesetzt ({s}).").format(s=sprache))
        if n_ueberschriften == 0 and n_absaetze > 5:
            hinweis(_("Das Dokument nutzt keine Überschriften-Formatvorlagen. Ohne Überschriften kann niemand im Dokument springen."))
        else:
            if erste_ebene not in (None, 1):
                hinweis(_("Die erste Überschrift hat Ebene {n} statt 1.").format(n=erste_ebene))
            if spruenge:
                hinweis(_("{n}-mal springt die Überschriften-Ebene (zum Beispiel von 1 auf 3).").format(n=spruenge) if spruenge > 1
                        else _("Einmal springt die Überschriften-Ebene (zum Beispiel von 1 auf 3)."))
            if leere_ueberschriften:
                hinweis(_("{n} Überschriften sind leer.").format(n=leere_ueberschriften) if leere_ueberschriften > 1
                        else _("Eine Überschrift ist leer."))
            if n_ueberschriften and not spruenge and not leere_ueberschriften and erste_ebene in (None, 1):
                ok(_("{n} Überschriften in sauberer Reihenfolge.").format(n=n_ueberschriften))
        if n_tabellen:
            if n_tabellen_ohne_kopf:
                hinweis(_("{n} von {m} Tabellen haben keine als Kopfzeile markierte erste Zeile (Tabellentools → Kopfzeile wiederholen).").format(n=n_tabellen_ohne_kopf, m=n_tabellen))
            else:
                ok(_("Alle {n} Tabellen haben eine Kopfzeile.").format(n=n_tabellen))
        if n_bilder:
            # „im Textkörper“ (11.09.2026): die Hoerprobe liest Kopf-/Fusszeilen nicht — sonst passen die
            # Zahlen nicht zur Bildliste des Projekts (Michael: 10 im Projekt, 9 hier, 4 in der PDF).
            if n_bilder_ohne:
                hinweis(_("{n} von {m} Bildern im Textkörper haben keinen Alternativtext (Kopf- und Fußzeilen nicht mitgezählt) — in InkluDocs generieren oder als Schmuckbild markieren.").format(n=n_bilder_ohne, m=n_bilder))
            else:
                ok(_("Alle {n} Bilder im Textkörper haben einen Alternativtext oder sind Schmuckbilder (Kopf- und Fußzeilen nicht mitgezählt).").format(n=n_bilder))
    return {"hoerprobe": hoer, "pruefbericht": befunde, "titel": titel, "sprache": sprache,
            "zahlen": {"ueberschriften": n_ueberschriften, "absaetze": n_absaetze, "bilder": n_bilder,
                       "bilder_ohne_alt": n_bilder_ohne, "dekorativ": n_dekorativ, "tabellen": n_tabellen,
                       "tabellen_ohne_kopf": n_tabellen_ohne_kopf}}
