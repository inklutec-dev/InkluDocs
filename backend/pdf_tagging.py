"""PDF-Tagging: ungetaggte (oder schlecht getaggte) PDFs mit PDFix barrierefrei machen (22.09.2026).

Weg: Joerg Heines Skript Make_Accessible (pdfix_scripts/Make_Accessible.py, Betriebsfassung
seines Make_Accessible_01.py vom 21.09.2026) fuehrt die in PDFix eingebaute Aktion
"make_accessible" aus. Die Aktion ist eine JSON-Konfiguration mit 37 Teilschritten; die
Voreinstellung liegt unveraendert unter pdfix_scripts/make_accessible_pdfix_default.json
(aus SDK 9.3.0 exportiert). Dieses Modul erzeugt je Lauf eine angepasste Fassung:

  1. Dokumentsprache: PDFix kennt keine Spracherkennung, die Voreinstellung traegt fest
     "en-US" ein (nur wenn keine Sprache gesetzt ist). Wir erkennen die Sprache aus dem
     Text (dieselbe Erkennung wie im Word-Pruefbericht) und setzen sie; steht im Dokument
     eine andere Sprache als der Text nahelegt, wird sie ersetzt (Hinweis im Bericht).
  2. Fuenf der 37 Teilschritte entfallen (_ENTFAELLT; richtiggestellt nach dem Audit vom 30.09.2026):
     - die vier "Set Alt (Figure/Formula: …)": sie schreiben Bildunterschrift, Nachbartext ("Ort, Datum", "A") oder
       das feste Wort "Decorative" als Alt-Text in Bilder und Formeln — das wuerde unsere Alt-Texte verdraengen, die
       InkluDocs ueber die Ansicht "Alt-Texte" schreibt (Export ueber AltTag_Import_CSV);
     - "Set Annotation Contents (Auto-generated)" (alt_type 3): er schreibt in ALLE Anmerkungen einen erzeugten Text,
       auch in Formularfelder. Laut Messung im Audit (11 Dateien, Joergs Original gegen diesen Aufsatz) ist das bei
       Formularfeldern der technische Feldname als Quickinfo (z. B. "Antrag_VGV_Vermittleradresse", bei einem Antrag
       407 von 407 Feldern) und bei Notizen "Text annotation". Dieser Feldnamen-Rueckfall ist damit AUCH abgeschaltet.
     Es bleiben: "Set Annotation Contents (BBox text)" und "(Action destination)" (Text unter der Anmerkung bzw. ihr
     Ziel, mit "Decorative" als Rueckfall) und "Set Alt (Form: From associated content)" (Beschriftung aus dem Tag).
     Folge: veraPDF meldet bei uns gegenueber Joergs Original zusaetzlich nur 7.3-1, 7.18.1-2 und 7.18.1-3 — Luecken, die
     InkluDocs mit echten Alt-Texten und Quickinfos schliesst; Joergs Automatik fuellt sie mit Platzhaltern, die veraPDF
     genuegen, einem Screenreader-Nutzer aber nicht. Ob bei Formularen ohne Quickinfo-Station der Feldname als Rueckfall
     zurueckkommen soll, klaert Michael Karbe (bewusst nicht geaendert).
  3. Zusaetzlich "Create Web Links" vor "Tag Annotations" (siehe konfig_erzeugen).

Lizenz (Stand 22.09.2026): Der Teilschritt add_tags ist in der Actino-Lizenz NICHT
freigeschaltet — mit aktivierter Lizenz bricht die Aktion bei ungetaggten PDFs ab
("Invalid initial element type"). Ohne Lizenz taggt das SDK im Testmodus, die Datei
traegt dann "Trial version of PDFix SDK" als Producer. Der Lauf aktiviert die Lizenz
deshalb nur bei PDFIX_TAGGING_LIZENZ=on (inkludocs_betrieb.lizenz_fuer_tagging); der
Bericht nennt den Modus, die Oberflaeche zeigt ihn. Prod erst mit freigeschaltetem
Tagging (Michael Karbe / PDFix).

Das Skript laeuft als Subprocess (Zeitlimit PDFIX_TAGGING_TIMEOUT, Standard 600 s). Der
Bericht (taggen()) ist JSON-tauglich und wird an documents.tagging_bericht gespeichert.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_SCRIPT = _SCRIPT_DIR / "Make_Accessible.py"
_PRESET = _SCRIPT_DIR / "make_accessible_pdfix_default.json"
_TIMEOUT_SECONDS = int(os.environ.get("PDFIX_TAGGING_TIMEOUT", "600"))
MAX_SEITEN = int(os.environ.get("PDFIX_TAGGING_MAX_SEITEN", "500"))
TRIAL_PRODUCER = "Trial version of PDFix SDK"

# Sprachkuerzel der Erkennung -> BCP-47-Wert fuer /Lang (PDF/UA verlangt einen gueltigen Tag).
SPRACHEN_BCP47 = {"de": "de-DE", "en": "en-US", "da": "da-DK", "fr": "fr-FR", "es": "es-ES", "sv": "sv-SE"}
# Form einer Sprachangabe, die wir an PDFix weitergeben (konfig_erzeugen) und aus dem Dokument uebernehmen (sprache_bestimmen).
_LANG_MUSTER = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*")
# Teilschritte, die je Lauf entfallen (name, Bedingung ueber die Parameter). Siehe Modulkopf.
_ENTFAELLT = (
    ("set_alt", lambda p: bool(re.search(r"Figure|Formula", p.get("tag_names", "")))),
    ("set_annot_contents", lambda p: p.get("alt_type") == "3"),
)


class TaggingFehler(Exception):
    """Nutzertauglicher Grund (nie Pfade, Tracebacks oder Schluessel)."""


def lizenz_modus() -> str:
    """'lizenz' (PDFIX_TAGGING_LIZENZ=on) oder 'testmodus'."""
    an = os.environ.get("PDFIX_TAGGING_LIZENZ", "off").strip().lower() in ("on", "1", "true", "yes")
    return "lizenz" if an else "testmodus"


def verfuegbar() -> bool:
    """Skript, Voreinstellung und SDK vorhanden?"""
    if not (_SCRIPT.is_file() and _PRESET.is_file()):
        return False
    try:
        import pdfixsdk  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# Dokument lesen: Seiten, Sprache, Titel, Tag-Statistik
# ---------------------------------------------------------------------------

def seitenzahl(pdf_pfad: str) -> int:
    import fitz
    with fitz.open(pdf_pfad) as d:
        return len(d)


def _katalog_lang(pdf_pfad: str) -> str:
    """/Lang aus dem Dokumentkatalog (leer, wenn nicht gesetzt)."""
    try:
        import pikepdf
        with pikepdf.open(pdf_pfad) as pdf:
            v = pdf.Root.get("/Lang")
            return str(v).strip() if v is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def _text_probe(pdf_pfad: str, max_seiten: int = 5, max_zeichen: int = 20000) -> str:
    import fitz
    teile: list[str] = []
    n = 0
    with fitz.open(pdf_pfad) as d:
        for seite in d:
            teile.append(seite.get_text())
            n += 1
            if n >= max_seiten or sum(len(t) for t in teile) >= max_zeichen:
                break
    return "\n".join(teile)[:max_zeichen]


def sprache_erkennen(text: str) -> dict:
    """{"code": "de"|..|"", "treffer": n, "zweite": m, "sicher": bool} — sicher = mindestens 20 Treffer
    und doppelt so viele wie die zweitbeste Sprache (Erkennung aus docx_hoerprobe, 6 Sprachen)."""
    from docx_hoerprobe import erkenne_sprache
    code, treffer, zweite = erkenne_sprache(text or "")
    sicher = bool(code) and treffer >= 20 and treffer >= 2 * max(zweite, 1)
    return {"code": code, "treffer": int(treffer), "zweite": int(zweite), "sicher": sicher}


def sprache_bestimmen(pdf_pfad: str, vorgabe: str = "de") -> dict:
    """Welche Sprache bekommt das Dokument?
    {"lang": BCP-47, "quelle": "erkannt"|"dokument"|"vorgabe", "overwrite": bool, "hinweis": str, "vorher": str}
    - Text sicher erkannt: diese Sprache; weicht /Lang ab, wird sie ersetzt (Hinweis).
    - Sonst vorhandenes /Lang behalten, wenn es eine gueltige Sprachangabe ist.
    - Sonst die Vorgabe (Projektsprache), mit Hinweis.
    Ungueltiges /Lang (z. B. "de_DE" oder "Deutsch", Skriptpruefung 09.10.2026): vorher brach der Lauf bei unsicherer
    Erkennung mit „Ungueltige Sprachangabe“ ab (konfig_erzeugen). Jetzt wird es wie ein fehlendes behandelt und mit
    overwrite ersetzt — sonst liesse PDFix (overwrite=false) den ungueltigen Wert stehen."""
    vorher = _katalog_lang(pdf_pfad)
    ungueltig = bool(vorher) and not _LANG_MUSTER.fullmatch(vorher)
    erk = sprache_erkennen(_text_probe(pdf_pfad))
    if erk["sicher"]:
        lang = SPRACHEN_BCP47[erk["code"]]
        if vorher and (ungueltig or vorher.split("-")[0].lower() != erk["code"]):
            return {"lang": lang, "quelle": "erkannt", "overwrite": True, "vorher": vorher,
                    "hinweis": f"Die Dokumentsprache war auf {vorher[:40]} gesetzt, der Text ist {erk['code']} — auf {lang} gesetzt."}
        return {"lang": lang, "quelle": "erkannt", "overwrite": False, "vorher": vorher, "hinweis": ""}
    if vorher and not ungueltig:
        return {"lang": vorher, "quelle": "dokument", "overwrite": False, "vorher": vorher, "hinweis": ""}
    lang = SPRACHEN_BCP47.get((vorgabe or "de").split("-")[0].lower(), "de-DE")
    if ungueltig:
        hinweis = (f"Die Dokumentsprache „{vorher[:40]}“ ist keine gültige Sprachangabe und die Sprache des Textes war nicht "
                   f"sicher erkennbar; {lang} wurde gesetzt (Projektsprache).")
    else:
        hinweis = f"Die Sprache des Textes war nicht sicher erkennbar; {lang} wurde angenommen (Projektsprache)."
    return {"lang": lang, "quelle": "vorgabe", "overwrite": ungueltig, "vorher": vorher, "hinweis": hinweis}


def tag_statistik(pdf_pfad: str) -> dict:
    """Zaehlt Strukturelemente je Typ (pikepdf), dazu Kennzahlen fuer den Bericht."""
    import pikepdf
    zaehler: dict[str, int] = {}

    def lauf(knoten):
        if isinstance(knoten, pikepdf.Dictionary) and "/S" in knoten:
            typ = str(knoten.S).lstrip("/")
            zaehler[typ] = zaehler.get(typ, 0) + 1
        k = knoten.get("/K") if isinstance(knoten, pikepdf.Dictionary) else None
        if k is None:
            return
        for kind in (k if isinstance(k, pikepdf.Array) else [k]):
            lauf(kind)

    with pikepdf.open(pdf_pfad) as pdf:
        wurzel = pdf.Root.get("/StructTreeRoot")
        if wurzel is not None:
            lauf(wurzel)
        info = pdf.docinfo
        producer = str(info.get("/Producer", "")) if info is not None else ""
        titel = str(info.get("/Title", "")) if info is not None else ""
        # PDFix schreibt Producer/Titel (Schritt fix_metadata) in die XMP-Metadaten; bei Dateien ohne
        # Info-Woerterbuch steht der Testmodus-Hinweis nur dort.
        try:
            with pdf.open_metadata() as xmp:
                producer = producer or str(xmp.get("pdf:Producer") or "")
                titel = titel or str(xmp.get("dc:title") or "")
        except Exception:  # noqa: BLE001
            pass
        lang = str(pdf.Root.get("/Lang", "") or "")
        lesezeichen = "/Outlines" in pdf.Root
    ueberschriften = sum(n for t, n in zaehler.items() if re.fullmatch(r"H[1-6]?", t))
    return {
        "elemente": sum(zaehler.values()),
        "je_typ": dict(sorted(zaehler.items(), key=lambda kv: -kv[1])),
        "ueberschriften": ueberschriften,
        "listen": zaehler.get("L", 0),
        "tabellen": zaehler.get("Table", 0),
        "bilder": zaehler.get("Figure", 0) + zaehler.get("Formula", 0),
        "absaetze": zaehler.get("P", 0),
        "producer": producer,
        "testmodus": TRIAL_PRODUCER.lower() in producer.lower(),
        "titel": titel,
        "lang": lang,
        "lesezeichen": lesezeichen,
    }


# ---------------------------------------------------------------------------
# Konfiguration der Aktion
# ---------------------------------------------------------------------------

def voreinstellung() -> dict:
    with open(_PRESET, encoding="utf-8") as f:
        return json.load(f)


def _params(aktion: dict) -> dict:
    return {p.get("name"): str(p.get("value")) for p in (aktion.get("params") or []) if isinstance(p, dict)}


def konfig_erzeugen(lang: str, overwrite_lang: bool, ziel_pfad: str, struktur_vorgegeben: bool = False,
                    tags_ersetzen: bool = False) -> dict:
    """Schreibt die Lauf-Konfiguration (Voreinstellung + unsere Aenderungen) nach ziel_pfad.
    struktur_vorgegeben=True (Weg „Struktur zuerst“, pdf_struktur_tagging, 23.09.2026): der Baum steht schon —
    add_tags (Strukturerkennung) und fix_headings (fuellt Ebenenspruenge mit LEEREN H-Tags, die ein Screenreader
    als „Ueberschrift, leer“ liest) entfallen; alle technischen Schritte bleiben.
    tags_ersetzen=True („Neu taggen“ einer schon getaggten PDF, Michael Karbe, Feedback 20261001 - 1, Punkt 1): add_tags mit
    overwrite=true — PDFix verwirft den vorhandenen Strukturbaum und taggt neu. Mit der Voreinstellung („Preserve Existing
    Tags“) blieb der Baum unveraendert (Messlauf 30.09.2026, und 01.10.2026: Actino Master Word 175 Elemente vorher und
    nachher; mit overwrite 135 Elemente, andere Struktur).
    Rueckgabe: {"entfernt": [Titel...], "sprache": lang, "schritte": n}"""
    if not _LANG_MUSTER.fullmatch(lang or ""):
        raise TaggingFehler("Ungueltige Sprachangabe")
    konfig = copy.deepcopy(voreinstellung())
    behalten, entfernt = [], []
    sprache_gesetzt = False
    for aktion in konfig.get("actions", []):
        name = aktion.get("name")
        p = _params(aktion)
        if any(name == n and bed(p) for n, bed in _ENTFAELLT):
            entfernt.append(aktion.get("title") or name)
            continue
        if struktur_vorgegeben and name in ("add_tags", "fix_headings"):
            entfernt.append(aktion.get("title") or name)
            continue
        if tags_ersetzen and name == "add_tags":
            for q in aktion.get("params", []):
                if q.get("name") == "overwrite":
                    q["value"] = "true"
            aktion["title"] = "AutoTag (Preflight, Replace Existing Tags)"
        if name == "set_language":
            for q in aktion.get("params", []):
                if q.get("name") == "lang":
                    q["value"] = lang
                elif q.get("name") == "overwrite":
                    q["value"] = "true" if overwrite_lang else "false"
            aktion["title"] = f"Set Document Language ({lang})"
            sprache_gesetzt = True
        behalten.append(aktion)
    if not sprache_gesetzt:
        raise TaggingFehler("Die Voreinstellung enthaelt keinen Sprachschritt")
    # Web-Links (22.09.2026, aus der Befehlsliste des SDK): Adressen und Mailadressen im Text werden
    # klickbare, getaggte Links — sonst liest ein Screenreader nur die Zeichenkette.
    # 23.09.2026 (Michaels Befund 10, „Ritterturnier“ S. 15): Der Schritt muss VOR tag_annot und set_annot_contents
    # stehen — sonst bleibt der neu erzeugte Link ungetaggt und ohne Contents (veraPDF 7.18.1-2, 7.18.5-1, 7.18.5-2).
    web_links = {"name": "create_web_links", "title": "Create Web Links (InkluDocs)", "params": [
        {"name": "url_regex", "value": "^(((http(s)?|ftp):\\/\\/)|(mailto:)|www.)[^\\s\\/$.?#].[^\\s]*"},
        {"name": "url_prefix", "value": ""}, {"name": "url", "value": ""}]}
    pos = next((i for i, a in enumerate(behalten) if a.get("name") == "tag_annot"), None)
    if pos is None:
        raise TaggingFehler("Die Voreinstellung enthaelt keinen Schritt zum Taggen von Anmerkungen")
    behalten.insert(pos, web_links)
    konfig["actions"] = behalten
    konfig["title"] = "Make Accessible (InkluDocs)"
    with open(ziel_pfad, "w", encoding="utf-8") as f:
        json.dump(konfig, f, ensure_ascii=False, indent=1)
    return {"entfernt": entfernt, "sprache": lang, "schritte": len(behalten)}


# ---------------------------------------------------------------------------
# Lauf
# ---------------------------------------------------------------------------

def _grund_aus_ausgabe(stdout: str, stderr: str) -> str:
    """Letzte ERROR-Zeile des Skripts als Grund, ohne Pfade."""
    for zeile in reversed((stdout or "").splitlines()):
        if zeile.startswith("ERROR:"):
            g = zeile[len("ERROR:"):].strip()
            g = re.sub(r"/[^\s]+", "…", g)
            return g[:200] or "PDFix hat die Aktion abgebrochen"
    if "Timeout" in (stderr or ""):
        return "Zeitueberschreitung"
    return "PDFix hat die Aktion abgebrochen"


def taggen(pdf_in: str, pdf_out: str, sprache_vorgabe: str = "de", arbeitsordner: Optional[str] = None,
           testmodus: bool = False, tags_ersetzen: bool = False) -> dict:
    """Fuehrt die Aktion aus und liefert den Bericht. Wirft TaggingFehler mit nutzertauglichem Grund.
    pdf_out wird nur bei Erfolg geschrieben (das Skript speichert am Ende, bei Abbruch nicht).
    testmodus=True (25.09.2026, „Testweise taggen“): Lizenz ausdruecklich AUS, egal was PDFIX_TAGGING_LIZENZ sagt —
    das SDK taggt dann im Testmodus (Wasserzeichen/„Trial version“ als Hersteller)."""
    if not verfuegbar():
        raise TaggingFehler("PDF-Tagging ist auf diesem Server nicht eingerichtet")
    if not os.path.isfile(pdf_in):
        raise TaggingFehler("Die Quelldatei fehlt")
    try:
        seiten = seitenzahl(pdf_in)
    except Exception:  # noqa: BLE001
        raise TaggingFehler("Die PDF konnte nicht gelesen werden")
    if seiten > MAX_SEITEN:
        raise TaggingFehler(f"Die PDF hat {seiten} Seiten; das Tagging ist auf {MAX_SEITEN} Seiten begrenzt")
    arbeitsordner = arbeitsordner or os.path.dirname(pdf_out) or "."
    os.makedirs(arbeitsordner, exist_ok=True)
    t0 = time.time()
    vorher = tag_statistik(pdf_in)
    sprache = sprache_bestimmen(pdf_in, sprache_vorgabe)
    konfig_pfad = os.path.join(arbeitsordner, "_make_accessible_konfig.json")
    konfig = konfig_erzeugen(sprache["lang"], sprache["overwrite"], konfig_pfad, tags_ersetzen=tags_ersetzen)
    cmd = [sys.executable, str(_SCRIPT), "-i", pdf_in, "-o", pdf_out, "-k", konfig_pfad]
    umgebung = os.environ.copy()
    if testmodus:
        umgebung["PDFIX_TAGGING_LIZENZ"] = "off"   # inkludocs_betrieb.lizenz_fuer_tagging aktiviert dann nichts
    modus = "testmodus" if testmodus else lizenz_modus()
    log.info("PDFix-Tagging aufrufen (%s, %d Seiten, Sprache %s)", modus, seiten, sprache["lang"])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR), env=umgebung)
    except subprocess.TimeoutExpired:
        raise TaggingFehler(f"Das Tagging hat zu lange gedauert (Zeitlimit {_TIMEOUT_SECONDS} s)")
    finally:
        try:
            os.remove(konfig_pfad)
        except OSError:
            pass
    if r.returncode != 0 or not os.path.isfile(pdf_out):
        log.warning("PDFix-Tagging fehlgeschlagen rc=%s stdout=%s stderr=%s", r.returncode, (r.stdout or "")[-400:], (r.stderr or "")[-400:])
        raise TaggingFehler(_grund_aus_ausgabe(r.stdout, r.stderr))
    from pdf_export import pdf_hat_tags
    if not pdf_hat_tags(pdf_out):
        raise TaggingFehler("PDFix hat keine Struktur erzeugt")
    nachher = tag_statistik(pdf_out)
    return {
        "zeit": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dauer_s": round(time.time() - t0, 1),
        "seiten": seiten,
        "modus": modus,
        "testmodus": nachher["testmodus"],
        "sprache": sprache,
        "konfig": konfig,
        "vorher": {k: vorher[k] for k in ("elemente", "ueberschriften", "listen", "tabellen", "bilder", "absaetze", "titel", "lang")},
        "nachher": nachher,
        "hinweise": [h for h in (sprache.get("hinweis"),) if h],
        "tags_ersetzt": bool(tags_ersetzen),
    }


def zwischenstand_satz(v: Optional[dict], _=None) -> str:
    """veraPDF direkt nach dem Taggen in einem Satz — ausdruecklich als Zwischenstand vor Alt-Texten und Quickinfos (Audit
    30.09.2026, MITTEL 4). Vorher stand hier pdfua_export.zusammenfassung („Deine PDF ist fertig und hat die Prüfung auf
    PDF/UA bestanden“) — nicht wahr: die fertige Datei prueft erst die Barrierefreiheitsprüfung."""
    _ = _ or (lambda s: s)
    if not v:
        return _("Die PDF/UA-Prüfung nach dem Taggen war nicht möglich.")
    # Michael Karbe, Feedback 20261001 - 2, Punkt 5 (wörtlich), ohne Abweichung der Gegensatz im selben Wortlaut
    if v.get("bestanden"):
        satz = _("Die automatische PDF/UA-Prüfung hat keine Abweichungen vom Standard gefunden.")
    else:
        satz = _("Die automatische PDF/UA-Prüfung hat Abweichungen vom Standard identifiziert.")
    return satz + " " + _("Die fertige Datei prüfst du in der Barrierefreiheitsprüfung.")


def verapdf(pdf_pfad: str, _=None) -> Optional[dict]:
    """PDF/UA-1-Pruefung ueber den Konverter-Dienst (veraPDF); None, wenn nicht pruefbar
    (kein Befund — ein Ausfall des Pruefdienstes sperrt das Tagging nicht)."""
    try:
        import pdfua_export
        if not pdfua_export.verfuegbar():
            return None
        with open(pdf_pfad, "rb") as f:
            rep = pdfua_export.pruefe(f.read())
        uebers = _ or (lambda s: s)
        klar = pdfua_export.klartext(rep, uebers)
        return {"bestanden": bool(klar.get("bestanden")), "profil": klar.get("profil"),
                "regeln_fehlgeschlagen": int(klar.get("regeln_fehlgeschlagen") or 0),
                "punkte": klar.get("punkte") or [], "zusammenfassung": zwischenstand_satz(klar, uebers)}
    except Exception as e:  # noqa: BLE001
        log.warning("veraPDF nach Tagging nicht moeglich: %r", e)
        return None
