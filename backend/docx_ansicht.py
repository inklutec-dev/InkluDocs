"""Word-Projekt: Ansichten „Dokument“ und „Barrierefreiheitsprüfung“ (30.09.2026).

Steve: „Word soll die gleiche Ansicht wie PDF bekommen, auch mit der Dokumentenverwaltung.“
Ein Word-Projekt hat seitdem dieselben Ansichten wie ein PDF-Projekt: Dokument (Datei-
verwaltung), Alt-Texte, Übersetzung, Barrierefreiheitsprüfung. Dieses Modul liefert die
Daten, die es dafür NEU braucht — alles ohne KI, reines Lesen:

* dokumentinfo(pfad): Dokumentinfos der hochgeladenen Word-Datei für die Karte in
  „Dokument“ (Titel, Sprache, Anwendung, Seiten, Zahl der Überschriften, Tabellen,
  Absätze). Titel, Sprache und Zahlen kommen aus docx_hoerprobe.analysiere — derselben
  Lesung, die auch Hörprobe und Prüfbericht machen —, Anwendung und Seiten aus
  docProps/app.xml bzw. den Seitenmarken von Word.
* pdfua_je_dokument(...): das Ergebnis der letzten barrierefreien PDF je Dokument aus
  der Ablage (veraPDF-Prüfung der Umwandlung), mit „aktuell ja/nein“ über einen
  Fingerabdruck der Alt-Texte.
* fingerabdruck(...): was die PDF eines Word-Dokuments bestimmt (Alt-Texte, Dokumentname,
  Sprache der Alt-Texte) als Prüfsumme — main._pdfua_umwandeln_sync speichert ihn im
  Ablage-Bericht, die Ansicht vergleicht ihn mit dem heutigen Stand.

Sicherheit: dieselben Schutzmaßnahmen wie docx_processor (_pruefe_zip gegen Zip-Bomben und
Pfad-Tricks, _lese_xml mit sicherem Parser ohne Entitäten/Netz). Fehler beim Lesen geben
leere Infos statt einer Ausnahme — die Karte zeigt dann „nicht angegeben“.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import zipfile
from typing import Callable, Optional

import docx_hoerprobe
from docx_processor import NS, _in_fallback, _lese_xml, _pruefe_zip

log = logging.getLogger(__name__)

_EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
_W = NS["w"]

# Kleiner Zwischenspeicher: die Karte wird bei jedem Neuzeichnen (Upload, Umbenennen, Polling) neu geholt;
# die Word-Datei aendert sich nach dem Hochladen nicht mehr. Schluessel = (Pfad, mtime, Groesse).
_CACHE_MAX = 256
_cache: dict = {}
_cache_lock = threading.Lock()


def _anwendung(zf: zipfile.ZipFile) -> str:
    """Programm, mit dem die Datei zuletzt gespeichert wurde (docProps/app.xml <Application>), wie „Anwendung“ bei PDF
    (dort der Creator-Eintrag). Angezeigt wird, was die Datei sagt."""
    if "docProps/app.xml" not in zf.namelist():
        return ""
    try:
        el = _lese_xml(zf, "docProps/app.xml").find(f"{{{_EP}}}Application")
        return ((el.text or "").strip() if el is not None else "")[:160]
    except Exception:  # noqa: BLE001
        return ""


def _app_seiten(zf: zipfile.ZipFile) -> Optional[int]:
    if "docProps/app.xml" not in zf.namelist():
        return None
    try:
        el = _lese_xml(zf, "docProps/app.xml").find(f"{{{_EP}}}Pages")
        n = int((el.text or "").strip()) if el is not None else 0
        return n if n > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _seitenzahl(zf: zipfile.ZipFile, root) -> Optional[int]:
    """Seiten, wie Word das Dokument zuletzt gezaehlt hat — nur, wenn das belegbar ist:
    1. Die Angabe in docProps/app.xml <Pages>, die Word bei jedem Speichern schreibt — aber nur, wenn die Datei die
       Bearbeitungsspuren eines Textprogramms traegt (w:rsid… an mindestens der Haelfte der Absaetze) und die Angabe
       nicht kleiner ist als das, was die ausdruecklichen Seiten- und Abschnittsumbrueche mindestens ergeben.
       Programmbibliotheken (python-docx) kopieren dort eine feste „1“ aus ihrer Vorlage und schreiben keine rsid-
       Kennungen (gemessen 30.09.2026: 29 Word-Dateien alle Absaetze mit rsid, die zwei python-docx-Testdateien keinen).
    2. Sonst die Seitenmarken <w:lastRenderedPageBreak/> im Hauptdokument (dieselbe Quelle wie „Seite N“ an den Bildern,
       docx_processor._seiten_der_bilder): Seiten = Marken + 1. Laeuft eine Tabellenzeile ueber eine Seitengrenze, setzt
       Word die Marke in JEDE Zelle — Marken derselben Tabellenzeile zaehlen darum nur einmal (Pruefung 30.09.2026).
    3. Sonst unbekannt (None).
    Andere Programme und Schriften koennen anders umbrechen; das steht in der Doku (docs/WORD.md)."""
    t_lrpb, t_br, t_pbb = f"{{{_W}}}lastRenderedPageBreak", f"{{{_W}}}br", f"{{{_W}}}pageBreakBefore"
    t_sect, t_ppr, t_p, t_tr = f"{{{_W}}}sectPr", f"{{{_W}}}pPr", f"{{{_W}}}p", f"{{{_W}}}tr"
    rsid = (f"{{{_W}}}rsidR", f"{{{_W}}}rsidRDefault", f"{{{_W}}}rsidP")
    marken = 0
    umbrueche = 0
    absaetze = 0
    mit_rsid = 0
    zeilen_mit_marke: dict = {}   # id -> Element; das Element im Wert halten, sonst ist id() eines lxml-Proxys nicht stabil
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag == t_p:
            absaetze += 1
            if any(el.get(a) for a in rsid):
                mit_rsid += 1
        elif el.tag == t_lrpb:
            if _in_fallback(el):
                continue
            zeile = next((a for a in el.iterancestors(t_tr)), None)
            if zeile is not None:
                if id(zeile) in zeilen_mit_marke:
                    continue
                zeilen_mit_marke[id(zeile)] = zeile
            marken += 1
        elif el.tag == t_br:
            if el.get(f"{{{_W}}}type") == "page" and not _in_fallback(el):
                umbrueche += 1
        elif el.tag == t_pbb:
            if el.get(f"{{{_W}}}val", "1") not in ("0", "false") and el.getparent() is not None \
                    and el.getparent().tag == t_ppr:
                umbrueche += 1
        elif el.tag == t_sect and el.getparent() is not None and el.getparent().tag == t_ppr:
            typ = el.find(f"{{{_W}}}type")
            if typ is None or typ.get(f"{{{_W}}}val") != "continuous":
                umbrueche += 1
    if absaetze and mit_rsid * 2 >= absaetze:
        angabe = _app_seiten(zf)
        if angabe and angabe >= umbrueche + 1:
            return angabe
    if marken:
        return marken + 1
    return None


def _core_sprache(zf: zipfile.ZipFile) -> str:
    """dc:language aus docProps/core.xml — steht sie dort, uebernimmt die Umwandlung sie in die PDF und nicht die
    Sprache der Alt-Texte (pdfua_export.dokumenttitel_setzen fuellt nur leere Felder)."""
    if "docProps/core.xml" not in zf.namelist():
        return ""
    try:
        el = _lese_xml(zf, "docProps/core.xml").find("dc:language", NS)
        return ((el.text or "").strip() if el is not None else "")[:35]
    except Exception:  # noqa: BLE001
        return ""


def _lesen(pfad: str) -> dict:
    info = {"lesbar": False, "titel": "", "sprache": "", "anwendung": "", "seiten": None,
            "ueberschriften": 0, "tabellen": 0, "absaetze": 0, "core_sprache": ""}
    if not pfad or not os.path.isfile(pfad):
        return info
    try:
        analyse = docx_hoerprobe.analysiere(pfad)
        zahlen = analyse.get("zahlen") or {}
        with zipfile.ZipFile(pfad) as zf:
            _pruefe_zip(zf)
            root = _lese_xml(zf, "word/document.xml")
            info.update({
                "lesbar": True,
                "titel": (analyse.get("titel") or "").strip()[:300],
                "sprache": (analyse.get("sprache") or "").strip()[:35],
                "anwendung": _anwendung(zf),
                "seiten": _seitenzahl(zf, root),
                "ueberschriften": int(zahlen.get("ueberschriften") or 0),
                "tabellen": int(zahlen.get("tabellen") or 0),
                "absaetze": int(zahlen.get("absaetze") or 0),
                "core_sprache": _core_sprache(zf),
            })
    except Exception as e:  # noqa: BLE001 — kaputte Datei: Karte zeigt „nicht angegeben“, nie einen Absturz
        log.warning("[word-ansicht] Dokumentinfos nicht lesbar (%s): %r", os.path.basename(pfad), e)
    return info


def dokumentinfo(pfad: str) -> dict:
    """Dokumentinfos einer Word-Datei (siehe Modulkopf), zwischengespeichert je Datei-Stand."""
    try:
        st = os.stat(pfad)
        schluessel = (os.path.realpath(pfad), st.st_mtime_ns, st.st_size)
    except OSError:
        return _lesen("")
    with _cache_lock:
        treffer = _cache.get(schluessel)
    if treffer is not None:
        return dict(treffer)
    info = _lesen(pfad)
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[schluessel] = info
    return dict(info)


def fingerabdruck(alt_texte: dict, name: str, sprache: str) -> str:
    """Pruefsumme ueber alles, was die barrierefreie PDF eines Word-Dokuments bestimmt und sich nach dem Hochladen
    aendern kann: die exportierten Alt-Texte je Bild ({anker: text|"dekorativ"|""|None}), den Dokumentnamen
    (wird Titel der PDF, wenn die Datei keinen hat) und die Sprache der Alt-Texte (Dokumentsprache der PDF, wenn die
    Datei keine hat). Die Word-Datei selbst aendert sich nicht (neue Datei = neues Dokument)."""
    roh = json.dumps({"alt": sorted((str(k), v if v is None else str(v)) for k, v in (alt_texte or {}).items()),
                      "name": (name or "").strip(), "sprache": (sprache or "").strip()},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(roh.encode("utf-8")).hexdigest()[:32]


def fingerabdruck_word(alt_texte: dict, name: str, sprache: str, info: dict) -> str:
    """Fingerabdruck nur ueber das, was die PDF WIRKLICH aendert: Der Dokumentname wird nur dann Titel der PDF, wenn die
    Word-Datei keinen eigenen Titel hat, die Sprache der Alt-Texte nur dann Dokumentsprache, wenn core.xml keine nennt
    (pdfua_export.dokumenttitel_setzen fuellt nur leere Felder). Sonst meldete die Pruefansicht nach jedem Umbenennen
    „nicht mehr aktuell“, obwohl eine neue PDF gleich waere. info = dokumentinfo() der hochgeladenen Word-Datei."""
    info = info or {}
    return fingerabdruck(alt_texte, "" if info.get("titel") else name, "" if info.get("core_sprache") else sprache)


def pdfua_je_dokument(rows: list, docs: list, label: Callable[[dict], str], fingerabdruecke: dict) -> dict:
    """Letzte barrierefreie PDF je Dokument aus Ablage-Zeilen (neueste zuerst, art 'pdfua').
    Zuordnung: Eintrag fuer genau dieses Dokument (ablage.document_id), sonst ein Eintrag fuer das ganze Projekt (ZIP),
    dessen Bericht das Dokument enthaelt (document_id im Bericht seit 30.09.2026; aeltere ZIP-Berichte nur ueber den
    Dateinamen-Teil, und nur bei genau einem Treffer). Eintraege, die VOR dem Hochladen des Dokuments entstanden sind,
    gehoeren nie zu ihm (gleichnamige Datei neu hochgeladen, Pruefung 30.09.2026). aktuell: True/False ueber den
    Fingerabdruck, None = unbekannt (Eintrag von vor dem 30.09.2026). Berichte werden erst bei Bedarf gelesen, und die
    Suche endet, sobald jedes Dokument sein Ergebnis hat. Rueckgabe {doc_id: {...}} — ohne Serverpfade."""
    offen = [d for d in (docs or []) if d.get("id") is not None]
    out: dict = {}
    for r in rows or []:
        if not offen:
            break
        bericht = None
        for doc in list(offen):
            did = doc["id"]
            if r.get("document_id") not in (did, None):
                continue
            angelegt, entstanden = str(doc.get("created_at") or ""), str(r.get("created_at") or "")
            if angelegt and entstanden and entstanden < angelegt:
                continue
            if bericht is None:
                try:
                    b = json.loads(r.get("bericht") or "[]")
                except Exception:  # noqa: BLE001
                    b = []
                bericht = [x for x in b if isinstance(x, dict)] if isinstance(b, list) else []
            if not bericht:
                break
            if r.get("document_id") == did:
                e = next((x for x in bericht if x.get("document_id") == did), bericht[0] if len(bericht) == 1 else None)
            else:
                e = next((x for x in bericht if x.get("document_id") == did), None)
                if e is None and not any("document_id" in x for x in bericht):
                    treffer = [x for x in bericht if x.get("dokument") == label(doc)]
                    e = treffer[0] if len(treffer) == 1 else None
            if e is None:
                continue
            pr = e.get("pruefung") or {}
            fp = e.get("fingerabdruck")
            out[did] = {
                "ausgabe_id": r.get("id"),
                "erstellt_am": r.get("created_at") or "",
                "bestanden": bool(pr.get("bestanden")),
                "regeln_verletzt": int(pr.get("regeln_fehlgeschlagen") or 0),
                "punkte": [{"bereich": p.get("bereich") or "", "text": p.get("text") or "",
                            "einzeln": [{"text": x.get("text") or "", "regeln": list(x.get("regeln") or [])}
                                        for x in (p.get("einzeln") or []) if isinstance(x, dict)]}
                           for p in (pr.get("punkte") or []) if isinstance(p, dict) and p.get("status") == "befund"],
                "aktuell": (None if not fp else fp == fingerabdruecke.get(did)),
            }
            offen.remove(doc)
    return out
