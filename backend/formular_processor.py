"""Quickinfo-Werkzeug: PDF-Formularfelder lesen (27.08.2026, Steve + Fable 5).

Gegenstueck zu pdf_processor.extract_images_from_pdf, aber fuer FORMULARFELDER
statt Bilder. Ein Formularfeld braucht fuer Screenreader eine Quickinfo (PDF-
Eintrag /TU, "Tooltip"): den zugaenglichen Namen, der vorgelesen wird, wenn der
Nutzer in das Feld springt. Dieses Modul liefert je Feld alles, was ein Mensch
oder die KI braucht, um diese Quickinfo zu schreiben — und NICHTS, was nicht
noetig ist (siehe Datenschutz).

Zwei Quellen, ein Ergebnis:
  1. PDFix (Joerg Heines Skript pdfix_scripts/Formular_Export_Quickinfo.py,
     Subprocess wie in pdfix_roundtrip.py): die massgebliche Feldliste mit
     vollem Feldnamen, vorhandener Quickinfo auf FELD-Ebene, Feldart und ob ein
     Wert eingetragen ist. Diese Liste bestimmt Reihenfolge und Anker.
  2. PyMuPDF (fitz): Geometrie und Umfeld — auf welcher Seite und wo jedes Feld
     erscheint, Beschriftung links/oberhalb/rechts, Abschnittsueberschrift,
     Text der Seite, Pflicht-Flag, Optionen von Auswahlfeldern; dazu der
     Bildausschnitt (Feld mit Beschriftung) und die Seitenansicht mit
     nummerierten Rahmen.
  Ist PDFix nicht verfuegbar (oder scheitert), liefert fitz allein die Liste
  (Quickinfo dann von der Erscheinung statt vom Feld — in der Praxis gleich).

Anker: der VOLLE Feldname (in gueltigen PDFs eindeutig). Namenlose Felder
bekommen "#<laufende Nummer>" und koennen nicht zurueckgeschrieben werden
(Hinweis im Ergebnis).

DATENSCHUTZ (bewusste Entscheidung, Steve 27.08.2026): Der eingetragene WERT
eines Feldes wird nirgends gespeichert oder weitergegeben — weder in der
Datenbank noch an die KI. Gespeichert wird nur "ausgefuellt ja/nein". Eine
Quickinfo beschreibt, was einzugeben ist, nicht, was drinsteht. Weil PyMuPDF
beim Textlesen und Rendern die Widget-Erscheinungen (und damit eingetragene
Werte) MIT ausgibt (belegt 27.08.2026 am Testformular), arbeiten Textextraktion
und Bilder auf einer Arbeitskopie im Speicher, aus der alle Widgets entfernt
sind; die Feldrahmen werden auf den Bildern selbst gezeichnet.

SICHERHEIT: Es werden nur Dateien aus dem Upload-Verzeichnis gelesen (Aufrufer
prueft den Pfad), PDFix laeuft als Subprocess mit Zeitlimit, verschluesselte
PDFs werden mit klarer Meldung abgewiesen, Seitenzahl und Feldzahl sind
begrenzt (ein Formular mit 5.000 Feldern ist kein Anwendungsfall, sondern
ein Angriff auf die Verarbeitungszeit).
"""
from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_EXPORT_SCRIPT = _SCRIPT_DIR / "Formular_Export_Quickinfo.py"
_TIMEOUT_SECONDS = 120

# Grenzen (Abwehr von Ressourcen-Missbrauch, nicht fachliche Grenzen).
MAX_SEITEN = 300
MAX_FELDER = 2000

# Geometrie-Parameter (PDF-Punkte, 1 pt = 1/72 Zoll). Erfahrungswerte fuer
# Formulare mit 9-11 pt Schrift; bewusst als Konstanten, damit sie an echten
# Kundenformularen nachjustiert werden koennen, ohne die Logik anzufassen.
LINKS_MAX_ABSTAND = 260     # Beschriftung links: hoechstens so weit vom Feldrand entfernt
RECHTS_MAX_ABSTAND = 220    # Beschriftung rechts (Kaestchen): hoechstens so weit
OBEN_MAX_ABSTAND = 42       # Beschriftung oberhalb: hoechstens so weit ueber dem Feld
GRUPPE_MAX_ABSTAND = 260    # Abschnittsueberschrift: hoechstens so weit oberhalb
UMFELD_LINKS, UMFELD_OBEN, UMFELD_RECHTS, UMFELD_UNTEN = 260, 70, 80, 25
AUSSCHNITT_ZOOM = 2.0       # ~144 dpi wie die Seitenansicht in pdf_processor

try:
    from pdfixsdk import GetPdfix  # noqa: F401
    _PDFIX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDFIX_AVAILABLE = False


def pdfix_moeglich() -> bool:
    """SDK installiert und nicht per PDFIX_ENABLED=false abgeschaltet (eine Stelle
    fuer Leser und Schreiber)."""
    return _PDFIX_AVAILABLE and os.environ.get("PDFIX_ENABLED", "true").lower() != "false"


class FormularFehler(ValueError):
    """Fachlicher Fehler mit Meldung fuer den Nutzer (400 im Upload)."""


# Feldarten: maschinenlesbare Schluessel. Anzeige-Texte kommen aus dem Frontend
# (uebersetzt), die Datenbank speichert nur diese Schluessel.
FELDART_PDFIX = {
    0: "unbekannt", 1: "button", 2: "radio", 3: "checkbox",
    4: "text", 5: "dropdown", 6: "liste", 7: "signatur",
}
FELDART_FITZ = {
    "Text": "text", "CheckBox": "checkbox", "RadioButton": "radio",
    "ComboBox": "dropdown", "ListBox": "liste", "Button": "button",
    "Signature": "signatur",
}
AUSWAHL_ARTEN = ("checkbox", "radio", "dropdown", "liste")

_FLAG_REQUIRED = 1 << 1   # /Ff Bit 2 (PDF 32000-1, Tabelle 221)


@dataclass
class Feld:
    """Ein Formularfeld mit allem, was Quickinfo-Arbeit braucht (ohne Wert!)."""
    feld_index: int                 # laufende Nummer im Dokument (PDFix-Reihenfolge)
    anker: str                      # voller Feldname, oder "#n" bei namenlosen Feldern
    feld_name: str
    feld_art: str                   # text|checkbox|radio|dropdown|liste|button|signatur|unbekannt
    quickinfo_original: str = ""    # vorhandene /TU aus der PDF
    page_number: int = 0            # erste Seite der Erscheinung (1-basiert), 0 = ohne Darstellung
    seiten: list = field(default_factory=list)
    rect: Optional[tuple] = None    # (x0, y0, x1, y1) auf page_number, fitz-Koordinaten
    beschriftung: str = ""          # erkannte Beschriftung im Formular
    beschriftung_lage: str = ""     # links|oben|rechts|innen|""
    gruppe: str = ""                # Abschnittsueberschrift oberhalb
    umfeld: str = ""                # Text rund um das Feld (Lesereihenfolge)
    optionen: list = field(default_factory=list)   # Exportwerte / Auswahlwerte
    pflicht: bool = False
    ausgefuellt: bool = False
    ausschnitt_path: str = ""
    page_view_path: str = ""
    page_text: str = ""


@dataclass
class FormularErgebnis:
    felder: list
    hinweise: dict                  # {"uebersprungen": [...], "warnungen": [...]}
    quelle_liste: str
    seiten: int


# --------------------------------------------------------------------------- Vorpruefung

def validiere_formular(pdf_path: str) -> int:
    """Prueft im Upload-Request, ob die PDF als Formular taugt. Gibt die Anzahl
    der Feld-Erscheinungen (Widgets) zurueck. Wirft FormularFehler mit einer
    Meldung, die der Nutzer versteht."""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        raise FormularFehler("Die Datei konnte nicht als PDF gelesen werden.")
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise FormularFehler("Die PDF ist mit einem Passwort geschützt. Bitte den Schutz in Acrobat entfernen und erneut hochladen.")
        if doc.page_count > MAX_SEITEN:
            raise FormularFehler(f"Die PDF hat {doc.page_count} Seiten. Unterstützt werden bis zu {MAX_SEITEN} Seiten je Datei.")
        anzahl = 0
        for page in doc:
            anzahl += sum(1 for _ in page.widgets())
            if anzahl > MAX_FELDER:
                raise FormularFehler(f"Die PDF enthält mehr als {MAX_FELDER} Formularfelder. Bitte das Formular aufteilen.")
        if anzahl == 0:
            raise FormularFehler("Diese PDF enthält keine ausfüllbaren Formularfelder. Das Quickinfo-Werkzeug braucht ein PDF-Formular (AcroForm) mit Eingabefeldern.")
        return anzahl
    finally:
        doc.close()


# --------------------------------------------------------------------------- PDFix-Liste

def _pdfix_feldliste(pdf_path: str, work_dir: str) -> list[dict]:
    """Ruft Heines Export-Skript auf (Subprocess) und liest die CSV.
    Spalten: Nummer;Name;Quickinfo;Type-Nr;Type;Value;Seite
    (Value ist bei uns nur "kein Wert"/"Wert vorhanden", siehe Skriptkopf)."""
    if not _PDFIX_AVAILABLE:
        raise RuntimeError("pdfix-sdk nicht installiert")
    os.makedirs(work_dir, exist_ok=True)
    csv_path = os.path.join(work_dir, "_formular_felder.csv")
    cmd = [sys.executable, str(_EXPORT_SCRIPT), "-i", pdf_path, "-c", csv_path]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR))
    if result.returncode != 0 or not os.path.exists(csv_path):
        raise RuntimeError(f"PDFix-Feldexport fehlgeschlagen (rc={result.returncode}): {result.stderr[:500]}")
    felder = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if not row or row[0] == "Nummer":
                continue
            try:
                typ_nr = int(row[3])
            except (ValueError, IndexError):
                typ_nr = 0
            seiten = []
            if len(row) > 6 and row[6].strip():
                try:
                    seiten = [int(row[6].strip())]
                except ValueError:
                    seiten = []
            felder.append({
                "nummer": int(row[0]),
                "name": row[1],
                "quickinfo": row[2] if len(row) > 2 else "",
                "feld_art": FELDART_PDFIX.get(typ_nr, "unbekannt"),
                "ausgefuellt": (len(row) > 5 and row[5] == "Wert vorhanden"),
                "seiten": seiten,
            })
    try:
        os.unlink(csv_path)
    except OSError:
        pass
    return felder


# --------------------------------------------------------------------------- fitz-Geometrie

def _zeilen_der_seite(page) -> list[dict]:
    """Textzeilen der Seite mit Rechteck, Text, groesster Schriftgroesse und
    Fett-Kennzeichen (aus dem "dict"-Extrakt). Reihenfolge: oben nach unten."""
    zeilen = []
    try:
        d = page.get_text("dict")
    except Exception:
        return zeilen
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans)
            text = re.sub(r"[\x00-\x1f\x7f\ufffd]", " ", text)   # Steuerzeichen (z. B. \x08 aus Symbolschriften)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            groesse = max(float(s.get("size", 0)) for s in spans)
            fett = any((int(s.get("flags", 0)) & 16) or ("bold" in str(s.get("font", "")).lower()) for s in spans)
            x0, y0, x1, y1 = line["bbox"]
            zeilen.append({"rect": fitz.Rect(x0, y0, x1, y1), "text": text, "groesse": groesse, "fett": fett})
    zeilen.sort(key=lambda z: (round(z["rect"].y0, 1), z["rect"].x0))
    return zeilen


def _ueberlappung_y(a: fitz.Rect, b: fitz.Rect) -> float:
    """Anteil (0..1) der Hoehe von a, der mit b ueberlappt."""
    h = max(a.height, 0.1)
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0)) / h


def _beschriftung(rect: fitz.Rect, zeilen: list[dict], feld_art: str) -> tuple[str, str]:
    """Findet die Beschriftung eines Feldes aus den Textzeilen der Seite.

    Reihenfolge je Feldart: Kaestchen (checkbox/radio) tragen ihre Beschriftung
    meist RECHTS, alle anderen LINKS in derselben Zeile, sonst OBERHALB,
    sonst INNEN (Platzhalter im Feldrahmen). Gibt (Text, Lage) zurueck.
    """
    links, rechts, oben, innen = [], [], [], []
    for z in zeilen:
        zr = z["rect"]
        if _ueberlappung_y(zr, rect) >= 0.5:
            if zr.x1 <= rect.x0 + 3 and rect.x0 - zr.x1 <= LINKS_MAX_ABSTAND:
                links.append((rect.x0 - zr.x1, z))
            elif zr.x0 >= rect.x1 - 3 and zr.x0 - rect.x1 <= RECHTS_MAX_ABSTAND:
                rechts.append((zr.x0 - rect.x1, z))
            elif rect.contains(zr):
                innen.append((0, z))
        elif zr.y1 <= rect.y0 + 3 and rect.y0 - zr.y1 <= OBEN_MAX_ABSTAND:
            # oberhalb: horizontal ueberlappend oder linksbuendig beginnend
            if zr.x1 > rect.x0 - 5 and zr.x0 < rect.x1:
                oben.append((rect.y0 - zr.y1, z))
    reihen = [("rechts", rechts), ("links", links), ("oben", oben), ("innen", innen)] \
        if feld_art in ("checkbox", "radio") else \
        [("links", links), ("oben", oben), ("innen", innen), ("rechts", rechts)]
    for lage, kand in reihen:
        if kand:
            kand.sort(key=lambda k: k[0])
            return kand[0][1]["text"], lage
    return "", ""


def _gruppe(rect: fitz.Rect, zeilen: list[dict], median_groesse: float, beschriftung: str) -> str:
    """Naechste Abschnittsueberschrift oberhalb des Feldes: fett, groesser als
    der Fliesstext oder eine Zeile, die mit Doppelpunkt endet — und nicht die
    Beschriftung selbst."""
    beste = None
    for z in zeilen:
        zr = z["rect"]
        if zr.y1 > rect.y0 + 3 or rect.y0 - zr.y1 > GRUPPE_MAX_ABSTAND:
            continue
        if z["text"] == beschriftung:
            continue
        ist_kopf = z["fett"] or z["groesse"] >= median_groesse * 1.15 or z["text"].endswith(":")
        if not ist_kopf or len(z["text"]) > 90:
            continue
        abstand = rect.y0 - zr.y1
        if beste is None or abstand < beste[0]:
            beste = (abstand, z["text"])
    return beste[1] if beste else ""


def _umfeld(rect: fitz.Rect, zeilen: list[dict], page_rect: fitz.Rect) -> str:
    box = fitz.Rect(rect.x0 - UMFELD_LINKS, rect.y0 - UMFELD_OBEN, rect.x1 + UMFELD_RECHTS, rect.y1 + UMFELD_UNTEN) & page_rect
    teile = [z["text"] for z in zeilen if z["rect"].intersects(box)]
    text = " ".join(teile)
    return text[:600]


def _fitz_widgets(doc) -> dict[str, dict]:
    """Alle Widget-Erscheinungen, gruppiert nach vollem Feldnamen.
    Liefert je Name: seiten (sortiert), erste Erscheinung (page, rect), Typ,
    TU der Erscheinung, Pflicht, Optionen, ausgefuellt."""
    felder: dict[str, dict] = {}
    for pn, page in enumerate(doc):
        for w in page.widgets():
            name = w.field_name or ""
            typ = FELDART_FITZ.get(w.field_type_string, "unbekannt")
            eintrag = felder.setdefault(name, {
                "seiten": [], "page": None, "rect": None, "rects": [], "feld_art": typ,
                "tu": "", "pflicht": False, "optionen": [], "ausgefuellt": False,
            })
            if pn + 1 not in eintrag["seiten"]:
                eintrag["seiten"].append(pn + 1)
            if eintrag["page"] is None:
                eintrag["page"], eintrag["rect"] = pn + 1, fitz.Rect(w.rect)
            if pn + 1 == eintrag["page"]:
                eintrag["rects"].append(fitz.Rect(w.rect))   # alle Erscheinungen der ersten Seite (Radio-Optionen)
            if not eintrag["tu"] and w.field_label:
                eintrag["tu"] = w.field_label
            try:
                eintrag["pflicht"] = eintrag["pflicht"] or bool(int(w.field_flags or 0) & _FLAG_REQUIRED)
            except Exception:
                pass
            try:
                if typ in ("checkbox", "radio"):
                    on = w.on_state()
                    if on and on not in eintrag["optionen"]:
                        eintrag["optionen"].append(str(on))
                elif typ in ("dropdown", "liste") and w.choice_values:
                    for v in w.choice_values:
                        s = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else str(v)
                        if s not in eintrag["optionen"]:
                            eintrag["optionen"].append(s)
            except Exception:
                pass
            # Wert wird NICHT gespeichert — nur, ob einer da ist.
            try:
                wert = w.field_value
                if wert not in (None, "", "Off") and typ != "button":
                    eintrag["ausgefuellt"] = True
            except Exception:
                pass
    return felder


# --------------------------------------------------------------------------- Bilder

def _ohne_widgets(doc):
    """Arbeitskopie des Dokuments im Speicher OHNE Widget-Erscheinungen (siehe
    DATENSCHUTZ im Modulkopf). Das Original bleibt unberuehrt."""
    kopie = fitz.open("pdf", doc.tobytes())
    for page in kopie:
        for w in list(page.widgets()):
            page.delete_widget(w)
    return kopie


def _render_ausschnitt(page, rect: fitz.Rect, feld_art: str, pfad: str) -> str:
    """Bildausschnitt: Feld mit seiner Beschriftung (links bzw. rechts bei
    Kaestchen), als PNG mit gezeichnetem Feldrahmen (die Seite ist widgetfrei).
    Gibt den Pfad oder "" zurueck."""
    if feld_art in ("checkbox", "radio"):
        clip = fitz.Rect(rect.x0 - 40, rect.y0 - 30, rect.x1 + RECHTS_MAX_ABSTAND + 40, rect.y1 + 14)
    else:
        clip = fitz.Rect(rect.x0 - LINKS_MAX_ABSTAND - 20, rect.y0 - OBEN_MAX_ABSTAND - 4, rect.x1 + 40, rect.y1 + 14)
    clip = clip & page.rect
    if clip.is_empty or clip.width < 4 or clip.height < 4:
        return ""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(AUSSCHNITT_ZOOM, AUSSCHNITT_ZOOM), clip=clip)
        try:
            from PIL import Image, ImageDraw
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            z = AUSSCHNITT_ZOOM
            ImageDraw.Draw(img).rectangle(((rect.x0 - clip.x0) * z, (rect.y0 - clip.y0) * z,
                                           (rect.x1 - clip.x0) * z, (rect.y1 - clip.y0) * z),
                                          outline=(199, 80, 0), width=3)
            img.save(pfad)
        except ImportError:  # pragma: no cover
            pix.save(pfad)
        return pfad
    except Exception as e:
        log.warning("Ausschnitt-Render fehlgeschlagen (%s): %s", pfad, e)
        return ""


def _render_seitenansicht(page, markierungen: list[tuple[int, fitz.Rect]], pfad: str) -> str:
    """Seitenansicht mit nummerierten Rahmen um alle Felder der Seite.
    Zeichnet auf eine Kopie der Seite (Shape), das Original bleibt unberuehrt."""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(AUSSCHNITT_ZOOM, AUSSCHNITT_ZOOM))
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover
            pix.save(pfad)
            return pfad
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(img)
        z = AUSSCHNITT_ZOOM
        for nr, r in markierungen:
            box = (r.x0 * z, r.y0 * z, r.x1 * z, r.y1 * z)
            draw.rectangle(box, outline=(199, 80, 0), width=3)
            etikett = str(nr)
            tx, ty = max(0, box[0] - 2), max(0, box[1] - 16)
            draw.rectangle((tx, ty, tx + 8 * len(etikett) + 8, ty + 15), fill=(199, 80, 0))
            draw.text((tx + 4, ty + 1), etikett, fill=(255, 255, 255))
        img.save(pfad)
        return pfad
    except Exception as e:
        log.warning("Seitenansicht-Render fehlgeschlagen (%s): %s", pfad, e)
        return ""


# --------------------------------------------------------------------------- Hauptfunktion

def analysiere_formular(pdf_path: str, output_dir: Optional[str] = None,
                        praefix: str = "feld") -> FormularErgebnis:
    """Liest alle Formularfelder einer PDF. output_dir=None: keine Bilder."""
    hinweise: dict = {"uebersprungen": [], "warnungen": []}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Massgebliche Liste ueber PDFix, Rueckfall fitz.
    pdfix_liste: list[dict] = []
    quelle = "fitz"
    if pdfix_moeglich():
        try:
            pdfix_liste = _pdfix_feldliste(pdf_path, output_dir or os.path.dirname(pdf_path) or ".")
            quelle = "pdfix"
        except Exception as e:
            log.warning("PDFix-Feldliste nicht verfuegbar, Rueckfall auf PyMuPDF: %s", e)
            hinweise["warnungen"].append("Feldliste ohne PDFix gelesen (Rückfall auf PyMuPDF).")

    doc = fitz.open(pdf_path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise FormularFehler("Die PDF ist mit einem Passwort geschützt.")
        widgets = _fitz_widgets(doc)

        if not pdfix_liste:
            # Rueckfall: Reihenfolge = Seite, dann Position (oben->unten, links->rechts)
            namen = sorted(widgets.keys(), key=lambda n: (widgets[n]["page"] or 0,
                                                          widgets[n]["rect"].y0 if widgets[n]["rect"] else 0,
                                                          widgets[n]["rect"].x0 if widgets[n]["rect"] else 0))
            pdfix_liste = [{"nummer": i + 1, "name": n, "quickinfo": widgets[n]["tu"],
                            "feld_art": widgets[n]["feld_art"], "ausgefuellt": widgets[n]["ausgefuellt"],
                            "seiten": widgets[n]["seiten"]} for i, n in enumerate(namen)]

        # Alle Seitenzugriffe (Text, Bilder) laufen ueber die widgetfreie Kopie.
        lese = _ohne_widgets(doc)
        # Seiten-Caches: Zeilen, Median-Schriftgroesse, Seitentext, Markierungen
        zeilen_cache: dict[int, list] = {}
        median_cache: dict[int, float] = {}
        text_cache: dict[int, str] = {}
        markierungen: dict[int, list] = {}

        def _seite(pn: int):
            if pn not in zeilen_cache:
                page = lese[pn - 1]
                zl = _zeilen_der_seite(page)
                zeilen_cache[pn] = zl
                groessen = sorted(z["groesse"] for z in zl) or [10.0]
                median_cache[pn] = groessen[len(groessen) // 2]
                try:
                    text_cache[pn] = page.get_text(sort=True)
                except Exception:
                    text_cache[pn] = ""
            return lese[pn - 1], zeilen_cache[pn], median_cache[pn], text_cache[pn]

        felder: list[Feld] = []
        gesehen: set[str] = set()
        for eintrag in pdfix_liste:
            name = eintrag["name"]
            w = widgets.get(name)
            anker = name if name else f"#{eintrag['nummer']}"
            if anker in gesehen:
                # Doppelter Feldname (ungueltige PDF): nur einmal fuehren, Hinweis.
                hinweise["warnungen"].append(f"Feldname „{name}“ kommt mehrfach vor; die Quickinfo gilt für alle Vorkommen.")
                continue
            gesehen.add(anker)
            feld_art = eintrag["feld_art"]
            if feld_art == "unbekannt" and w:
                feld_art = w["feld_art"]
            f = Feld(
                feld_index=eintrag["nummer"], anker=anker, feld_name=name, feld_art=feld_art,
                quickinfo_original=eintrag.get("quickinfo") or (w["tu"] if w else ""),
                ausgefuellt=bool(eintrag.get("ausgefuellt")) or bool(w and w["ausgefuellt"]),
            )
            if not name:
                hinweise["uebersprungen"].append({"art": "ohne_name", "nummer": eintrag["nummer"], "feld_art": feld_art})
            if w and w["page"]:
                f.page_number = w["page"]
                f.seiten = sorted(w["seiten"])
                f.rect = (round(w["rect"].x0, 2), round(w["rect"].y0, 2), round(w["rect"].x1, 2), round(w["rect"].y1, 2))
                f.pflicht = w["pflicht"]
                f.optionen = list(w["optionen"])
                page, zeilen, median, ptext = _seite(f.page_number)
                f.beschriftung, f.beschriftung_lage = _beschriftung(w["rect"], zeilen, feld_art)
                f.gruppe = _gruppe(w["rect"], zeilen, median, f.beschriftung)
                f.umfeld = _umfeld(w["rect"], zeilen, page.rect)
                f.page_text = ptext
                for r in (w.get("rects") or [w["rect"]]):
                    markierungen.setdefault(f.page_number, []).append((f.feld_index, r))
                if output_dir:
                    f.ausschnitt_path = _render_ausschnitt(page, w["rect"], feld_art,
                                                           os.path.join(output_dir, f"{praefix}_{f.feld_index}.png"))
            else:
                # Feld ohne sichtbare Erscheinung (versteckt / ohne Widget): Seite aus PDFix, kein Bild.
                f.seiten = list(eintrag.get("seiten") or [])
                f.page_number = f.seiten[0] if f.seiten else 0
                hinweise["uebersprungen"].append({"art": "ohne_darstellung", "name": name, "feld_art": feld_art, "seite": f.page_number})
            felder.append(f)

        # Seitenansichten mit Rahmen (einmal je Seite), Pfad an alle Felder der Seite.
        if output_dir:
            for pn, marks in markierungen.items():
                pfad = _render_seitenansicht(lese[pn - 1], marks, os.path.join(output_dir, f"p{pn}_seitenansicht.png"))
                for f in felder:
                    if f.page_number == pn:
                        f.page_view_path = pfad

        lese.close()
        return FormularErgebnis(felder=felder, hinweise=hinweise, quelle_liste=quelle, seiten=doc.page_count)
    finally:
        doc.close()


def extract_formular(pdf_path: str, output_dir: str, project_id: int) -> tuple[list[dict], dict]:
    """Schnittstelle fuer main.py (Muster extract_docx): (Felder als dicts, Hinweise)."""
    erg = analysiere_formular(pdf_path, output_dir, praefix="feld")
    return [asdict(f) for f in erg.felder], dict(erg.hinweise, quelle_liste=erg.quelle_liste, seiten=erg.seiten)
