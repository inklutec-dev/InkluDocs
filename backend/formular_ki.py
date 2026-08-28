"""Quickinfo-Werkzeug, Stufe 2: Feld-Pass mit Nachpruefung (27.08.2026, Steve + Fable 5).

Erzeugt Quickinfo-Vorschlaege fuer die Felder EINER Formularseite mit einem
einzigen Modellaufruf (Text, kein Bild) und prueft das Ergebnis
deterministisch nach, bevor es gespeichert wird:

  1. Kontext bauen: Textzeilen der Seite mit Positionen aus der widgetfreien
     Arbeitskopie (formular_processor — nie Feldwerte), Felder mit Positionen.
  2. Modell (Sonnet ueber Bedrock, Tool-Use-Schema QuickinfoSeiteOutput,
     Temperatur 0; bei "Neu generieren" 0.5 fuer Variation).
  3. Nachpruefung je Feld — kann die Sicherheit nur senken:
       - Beleg-Pruefung: Beleg steht (normalisiert) im Seitentext, sonst niedrig.
       - Lage-Pruefung: Belegzeile liegt in Feldnaehe (links/oberhalb, rechts bei
         Kaestchen, innen, oder Ueberschrift darueber), sonst hoechstens mittel.
       - Regel-Pruefung: Laenge, Anleitungsfloskeln, Feldart im Text, Format ohne
         Vorkommen auf der Seite, "Pflichtfeld" ohne Kennzeichnung.
  4. Konsistenz ueber das Dokument (formular_api ruft konsistenz() nach dem
     Lauf): gleiche Beschriftung + Feldart + Gruppe -> gleicher Wortlaut.

Was das Modell NIE sieht: Feldwerte (werden nicht gespeichert), Bilder.
Der Seitentext steht in einem Datenblock; Texte ohne Beleg werden markiert.
Grenzen: hoechstens 40 Felder je Aufruf (grosse Seiten werden geteilt),
Seitentext auf 12.000 Zeichen gekappt, Zeitlimit des Bedrock-Clients.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import fitz

from formular_processor import _ohne_widgets, _zeilen_der_seite  # widgetfreie Kopie, Zeilen mit Position
from pipelines.v4 import bedrock_client
from prompts.builders.quickinfo import build_quickinfo_prompt
from prompts.components.schemas.quickinfo import QuickinfoSeiteOutput

log = logging.getLogger(__name__)

MAX_FELDER_JE_AUFRUF = 40
MAX_SEITENTEXT_ZEICHEN = 12000
MAX_QUICKINFO_LAENGE = 200
TEMPERATUR_NORMAL = 0.0
TEMPERATUR_VARIATION = 0.5
STUFEN = {"hoch": 3, "mittel": 2, "niedrig": 1}
_ANLEITUNGSFLOSKELN = ("bitte hier", "hier eingeben", "hier eintragen", "tragen sie", "geben sie", "please enter", "enter here")
_FELDART_WOERTER = ("textfeld", "kontrollkästchen", "auswahlknopf", "auswahlliste", "listenfeld", "text field", "checkbox", "radio button")
_FORMAT_MUSTER = re.compile(r"(TT|DD|JJJJ|YYYY|MM|Format|dd/mm|dd\.mm|mm/dd|IBAN|BIC|PLZ)", re.I)


@dataclass
class FeldVorschlag:
    feld_id: int
    quickinfo: str
    beleg: str = ""
    gruppe: str = ""
    sicherheit: str = "niedrig"
    hinweise: list = field(default_factory=list)


class FeldPassFehler(RuntimeError):
    """Modell- oder Kontextfehler mit Meldung fuer den Nutzer."""


# --------------------------------------------------------------------------- Kontext

def _norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[\s ]+", " ", t)
    t = re.sub(r"[^\w\s/().,:-]", "", t)
    return t.strip(" .:,;")


def seiten_zeilen(pdf_path: str, page_number: int) -> tuple[list[dict], str]:
    """Textzeilen (rect als Tupel, text, fett, groesse) und Seitentext der Seite,
    aus einer Kopie OHNE Widgets (keine Feldwerte)."""
    doc = fitz.open(pdf_path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise FeldPassFehler("Die PDF ist mit einem Passwort geschützt.")
        if page_number < 1 or page_number > doc.page_count:
            raise FeldPassFehler("Seite nicht vorhanden.")
        lese = _ohne_widgets(doc)
        try:
            page = lese[page_number - 1]
            zeilen = _zeilen_der_seite(page)
            out = [{"rect": (round(z["rect"].x0, 1), round(z["rect"].y0, 1), round(z["rect"].x1, 1), round(z["rect"].y1, 1)),
                    "text": z["text"], "fett": bool(z["fett"]), "groesse": float(z["groesse"])} for z in zeilen]
            text = " ".join(z["text"] for z in out)
            if len(text) > MAX_SEITENTEXT_ZEICHEN:
                # Zeilen kappen, damit der Prompt begrenzt bleibt (Riesenseiten sind
                # kein Formular-Alltag; die Felder selbst bleiben vollstaendig).
                gesamt, gekappt = 0, []
                for z in out:
                    gesamt += len(z["text"]) + 1
                    if gesamt > MAX_SEITENTEXT_ZEICHEN:
                        break
                    gekappt.append(z)
                out = gekappt
                text = " ".join(z["text"] for z in out)
            return out, text
        finally:
            lese.close()
    finally:
        doc.close()


# --------------------------------------------------------------------------- Nachpruefung

def _zeilen_mit_beleg(beleg: str, zeilen: list[dict]) -> list[dict]:
    nb = _norm(beleg)
    if not nb:
        return []
    treffer = [z for z in zeilen if nb in _norm(z["text"])]
    if treffer:
        return treffer
    # Beleg kann ueber zwei Zeilen gehen: erste 4 Woerter reichen fuer die Lage.
    kurz = " ".join(nb.split()[:4])
    return [z for z in zeilen if kurz and kurz in _norm(z["text"])] if len(nb.split()) > 4 else []


def _in_feldnaehe(z: dict, rect, feld_art: str) -> bool:
    x0, y0, x1, y1 = rect
    zx0, zy0, zx1, zy1 = z["rect"]
    ueberlappt_y = min(y1, zy1) - max(y0, zy0) > 0
    links = ueberlappt_y and zx1 <= x0 + 3 and (x0 - zx1) <= 300
    rechts = ueberlappt_y and zx0 >= x1 - 3 and (zx0 - x1) <= 260
    innen = zx0 >= x0 - 2 and zx1 <= x1 + 2 and zy0 >= y0 - 2 and zy1 <= y1 + 2
    oberhalb = zy1 <= y0 + 3 and (y0 - zy1) <= 300 and zx1 > x0 - 40 and zx0 < x1 + 40
    ueberschrift = zy1 <= y0 + 3 and (y0 - zy1) <= 400 and (z.get("fett") or z.get("groesse", 0) >= 12)
    return links or rechts or innen or oberhalb or ueberschrift


def nachpruefung(vorschlag: FeldVorschlag, feld: dict, zeilen: list[dict], seitentext: str,
                 mit_seitenbild: bool = False) -> FeldVorschlag:
    """Deterministische Pruefung; senkt die Sicherheit bei Verstoessen und sammelt Hinweise.
    mit_seitenbild: der Vorschlag entstand MIT Seitenbild (Ausnahme fuer Felder ohne Beschriftung in
    der Naehe) — die Stufen bleiben gleich streng, nur der Hinweis sagt, woher die Zuordnung kommt."""
    stufe = STUFEN.get(vorschlag.sicherheit, 1)
    hinweise = list(vorschlag.hinweise)
    qi = (vorschlag.quickinfo or "").strip()
    nt = _norm(seitentext)

    # Beleg-Pruefung
    if not _norm(vorschlag.beleg):
        stufe = min(stufe, 1)
        hinweise.append("Kein Beleg auf der Seite angegeben.")
    elif _norm(vorschlag.beleg) not in nt:
        stufe = min(stufe, 1)
        hinweise.append("Beleg steht nicht im Seitentext.")
    else:
        # Lage-Pruefung
        rect = feld.get("rect")
        if rect:
            belegzeilen = _zeilen_mit_beleg(vorschlag.beleg, zeilen)
            if belegzeilen and not any(_in_feldnaehe(z, rect, feld.get("feld_art") or "") for z in belegzeilen):
                stufe = min(stufe, 2)
                hinweise.append("Beleg liegt nicht in der Nähe des Feldes; Zuordnung aus dem Seitenbild." if mit_seitenbild
                                else "Beleg liegt nicht in der Nähe des Feldes.")

    # Redundanz (Befund Bankformular 28.08.2026, vom Chatbot gefunden): „Gruppe: … Gruppe …“ —
    # steht der Praefix vor dem Doppelpunkt im Satz dahinter noch einmal (Kern ohne Klammer-
    # Nummer), fliegt der Praefix; der Satz dahinter traegt die Gruppe bereits.
    m = re.match(r"^\s*([^:]{3,60}?)\s*:\s*(.+)$", qi)
    if m:
        kern = re.sub(r"\[[^\]]*\]|\d+", " ", m.group(1)).lower()
        woerter = re.findall(r"[^\W\d_]{4,}", kern)          # alle Schriften (é, ø, å), Review 28.08.
        rest_woerter = {w[:6] for w in re.findall(r"[^\W\d_]{4,}", m.group(2).lower())}
        # Ganze Woerter vergleichen (Stamm 6 Zeichen: Berechtigter/Berechtigten), kein Substring —
        # „Kontoinhaber: Konto-Nr.“ bleibt stehen.
        if woerter and all(w[:6] in rest_woerter for w in woerter):
            qi = m.group(2).strip()
            hinweise.append("Doppelte Gruppe entfernt.")

    # Regel-Pruefung
    if len(qi) > MAX_QUICKINFO_LAENGE:
        qi = qi[:MAX_QUICKINFO_LAENGE].rstrip()
        hinweise.append("Text gekürzt.")
    ql = qi.lower()
    if any(f in ql for f in _ANLEITUNGSFLOSKELN):
        stufe = min(stufe, 2)
        hinweise.append("Anleitungsfloskel im Text.")
    if any(w in ql for w in _FELDART_WOERTER):
        stufe = min(stufe, 2)
        hinweise.append("Feldart im Text (sagt der Screenreader selbst).")
    if _FORMAT_MUSTER.search(qi) and not _FORMAT_MUSTER.search(seitentext or ""):
        stufe = min(stufe, 2)
        hinweise.append("Formatangabe ohne Vorkommen auf der Seite.")
    if "pflichtfeld" in ql or "required" in ql:
        # Kennzeichnung zaehlt nur AM FELD (Pflicht-Flag oder Sternchen an der eigenen
        # Beschriftung) — ein Sternchen irgendwo auf der Seite reicht nicht.
        stern = "*" in (feld.get("beschriftung") or "")
        if not feld.get("pflicht") and not stern:
            stufe = min(stufe, 2)
            hinweise.append("„Pflichtfeld“ ohne Kennzeichnung im Formular.")

    vorschlag.quickinfo = qi
    vorschlag.sicherheit = {3: "hoch", 2: "mittel", 1: "niedrig"}[stufe]
    vorschlag.hinweise = hinweise
    return vorschlag


# Zaehl-Merkmale eines Textes: Ziffern, eckige Klammern mit Nummer, Ordnungswoerter (de/en/fr/es/da/sv).
# Unterscheiden sich zwei Quickinfos darin, meinen sie verschiedene Personen/Bloecke
# ("Berechtigter [1]" vs. "[2]", "erster" vs. "zweiter Vertreter") — dann darf die
# Konsistenz-Angleichung NICHT greifen (Befund Bankformular 28.08.2026: drei gleiche
# Bloecke, die Angleichung ueberschrieb die richtigen Nummern des Modells).
_ORDNUNG = re.compile(
    r"\d+|\b(?:erste[rsn]?|zweite[rsn]?|dritte[rsn]?|vierte[rsn]?|fuenfte[rsn]?|fünfte[rsn]?|first|second|third|fourth|fifth|"
    r"premier|première|deuxième|troisième|primer[oa]?|segund[oa]|tercer[oa]?|første|anden|andet|tredje|första|andra)\b", re.I)


def _zaehlmerkmale(text: str) -> tuple:
    return tuple(m.lower() for m in _ORDNUNG.findall(text or ""))


def konsistenz(vorschlaege: list[FeldVorschlag], felder_by_id: dict[int, dict]) -> list[FeldVorschlag]:
    """Gleiche Beschriftung + Feldart + Gruppe -> gleicher Wortlaut (erste Fassung gewinnt) —
    AUSSER die Texte unterscheiden sich in Zahlen oder Ordnungswoertern: dann bezeichnen sie
    verschiedene Bloecke, und der Wortlaut des Modells bleibt (28.08.2026)."""
    gruppen: dict[tuple, str] = {}
    for v in vorschlaege:
        f = felder_by_id.get(v.feld_id, {})
        key = (_norm(f.get("beschriftung") or ""), f.get("feld_art") or "", _norm(v.gruppe or f.get("gruppe") or ""))
        if not key[0]:
            continue
        if key in gruppen:
            if gruppen[key] != v.quickinfo:
                if _zaehlmerkmale(gruppen[key]) != _zaehlmerkmale(v.quickinfo):
                    continue   # andere Nummer/Ordnung = anderer Block, nicht angleichen
                v.quickinfo = gruppen[key]
                v.hinweise.append("Wortlaut an gleiche Beschriftung angeglichen.")
        else:
            gruppen[key] = v.quickinfo
    return vorschlaege


# --------------------------------------------------------------------------- Feld-Pass

def generiere_seite(pdf_path: str, page_number: int, felder: list[dict], *, sprache: str = "de",
                    formular_titel: str = "", seiten_gesamt: int = 1, bestaetigte: list[tuple[str, str]] | None = None,
                    user_prompt: str = "", variation: bool = False, seitenbild_path: str | None = None) -> list[FeldVorschlag]:
    """Ein Modellaufruf je Seite (bei > 40 Feldern mehrere), mit Nachpruefung.

    felder: Feld-Dicts (id, feld_index, feld_art, rect als Tupel, pflicht, optionen,
    beschriftung, beschriftung_lage, gruppe, seiten, quickinfo_original).
    seitenbild_path: SEITENBILD-AUSNAHME (28.08.2026, Steve): Hat der Aufruf ein Feld OHNE
    Beschriftung in der Naehe, geht die gerenderte Seite mit nummerierten Feldrahmen
    (formular_processor._render_seitenansicht) als Bild mit — das Modell liest das
    Layout wie ein Mensch. Seiten, deren Felder alle beschriftet sind, laufen
    unveraendert nur mit Text (billiger, belegbarer).
    """
    if not felder:
        return []
    zeilen, seitentext = seiten_zeilen(pdf_path, page_number)
    ergebnisse: list[FeldVorschlag] = []
    by_index = {f["feld_index"]: f for f in felder}
    for start in range(0, len(felder), MAX_FELDER_JE_AUFRUF):
        teil = felder[start:start + MAX_FELDER_JE_AUFRUF]
        mit_bild = bool(seitenbild_path) and os.path.isfile(seitenbild_path or "") \
            and any(not (f.get("beschriftung") or "").strip() for f in teil)
        system, prompt = build_quickinfo_prompt(
            zeilen, teil, formular_titel=formular_titel, seite=page_number, seiten_gesamt=seiten_gesamt,
            sprache=sprache, bestaetigte=bestaetigte, user_prompt=user_prompt, variation=variation,
            mit_seitenbild=mit_bild)
        temperatur = TEMPERATUR_VARIATION if variation else TEMPERATUR_NORMAL
        try:
            if mit_bild:
                log.info("Feld-Pass Seite %s: Seitenbild-Ausnahme (Feld ohne Beschriftung)", page_number)
                out = bedrock_client.call_bedrock_with_schema(
                    model=bedrock_client.BEDROCK_MODEL_GENERATE, prompt=prompt, image_path=seitenbild_path,
                    schema=QuickinfoSeiteOutput, max_tokens=4000, temperature=temperatur, system=system)
            else:
                out = bedrock_client.call_bedrock_text_with_schema(
                    model=bedrock_client.BEDROCK_MODEL_GENERATE, prompt=prompt, schema=QuickinfoSeiteOutput,
                    max_tokens=4000, temperature=temperatur, system=system)
        except bedrock_client.BedrockCallError as e:
            log.error("Feld-Pass Seite %s fehlgeschlagen: %s", page_number, e)
            raise FeldPassFehler("Die KI-Anfrage ist fehlgeschlagen. Bitte später erneut versuchen.")
        gesehen = set()
        for o in out.felder:
            f = by_index.get(o.feld_index)
            if not f or o.feld_index in gesehen or f not in teil:
                continue   # unbekannte oder doppelte Nummer: ignorieren
            gesehen.add(o.feld_index)
            v = FeldVorschlag(feld_id=f["id"], quickinfo=o.quickinfo, beleg=o.beleg or "", gruppe=o.gruppe or "",
                              sicherheit=o.sicherheit, hinweise=[o.hinweis] if o.hinweis else [])
            if mit_bild and not (f.get("beschriftung") or "").strip():
                v.hinweise.append("Seitenbild einbezogen (keine Beschriftung in der Nähe).")
            ergebnisse.append(nachpruefung(v, f, zeilen, seitentext, mit_seitenbild=mit_bild))
        fehlend = [f["feld_index"] for f in teil if f["feld_index"] not in gesehen]
        if fehlend:
            log.warning("Feld-Pass Seite %s: keine Antwort fuer Felder %s", page_number, fehlend)
    return ergebnisse
