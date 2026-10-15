"""Auftraege des InkluAgent, Baustein 2 (Steve 14.10.2026: „bitte nimm dir die Aufgabe — er sagt, das kostet so und so viel,
dann kann man ihn beauftragen, und dann macht er im Hintergrund und man kann die Seite zumachen“). Schalter
funktionen.AGENT_AUFTRAEGE (braucht die Formular-Karten AGENT_KARTEN und die Rueckmeldung, die wiederum AGENT_KONTO braucht).

1. Planen (Werkzeug auftrag_vorbereiten): der Agent nennt die gewuenschten Schritte. Der SERVER prueft sie gegen den Katalog
   (KATALOG: was es in diesem Projekt als Werkzeug gibt), ordnet sie in die feste Reihenfolge des Katalogs, rechnet je Schritt
   den Preis mit DENSELBEN Funktionen wie das einzelne Werkzeug und legt EINE Karte vor (karten.AUFTRAG): die Schritte mit
   Umfang und Preis, die Summe, das Guthaben und einen Haken „per Mail melden“. Geht ein gewuenschter Schritt nicht oder reicht
   das Guthaben nicht, gibt es keine Karte, sondern den Grund fuer das Modell.
2. Erteilen: NUR der Knopf der Karte (POST /api/agent/karte -> karten.absenden -> karte_senden). Ein getipptes „Ja“ erteilt
   keinen Auftrag. Der Server rechnet den Plan neu; ist die Summe eine andere, gilt die Karte nicht mehr. Sonst legt er den
   Auftrag an (Tabelle agent_auftraege) und in DERSELBEN Transaktion die Rueckmeldung (Art „auftrag“, rueckmeldung.py).
3. Abarbeiten OHNE KI (schleife, alle PRUEF_S Sekunden, eigener Faden): je Auftrag den naechsten Schritt starten — ueber
   DASSELBE Werkzeug wie im Gespraech, mit bestaetigt=true. Die Zustimmung gibt der Auftrag (tools/ausgaben.auftrag_schritt):
   nur fuer die Art dieses Schritts, hoechstens einmal und hoechstens bis zur Rest-Summe der Karte. Laeuft der Schritt im
   Hintergrund weiter, wartet der Server an denselben Zustaenden wie die Rueckmeldung und wertet ihn danach aus.
4. Melden: ist der Auftrag zu Ende, schreibt die Rueckmeldung die persoenliche Nachricht des Agenten aus den Ergebnissen aller
   Schritte (fakten), mit den entstandenen Dateien darunter (anhaenge) und auf Wunsch per Mail an das eigene Konto.

Regeln:
- Nie mehr als die Summe der Karte. Jeder Schritt wird mit der Rest-Summe freigegeben (Summe minus die freigegebenen Preise der
  Schritte davor). Wuerde ein Schritt jetzt mehr kosten (etwa weil das Tagging andere Bilder gefunden hat), haelt der Auftrag an,
  und die Rueckmeldung sagt es. Ein Schritt, dessen Preis eine Obergrenze ist (Herunterladen nach dem Bearbeiten: es kostet nur,
  was bearbeitet wurde), steht auf der Karte als „höchstens“.
- Ein Schritt, der scheitert, haelt den Auftrag an — was danach kaeme, startet nicht und kostet nichts. Ein Schritt, der nur
  teilweise klappt (einzelne Bilder ohne Alt-Text), haelt ihn nicht an; so wie die Kette (Steve und Michael 22.09.2026).
  Bricht der Nutzer einen Lauf selbst ab (Knopf in der Oberflaeche), endet auch der Auftrag.
- Laeuft im Projekt gerade etwas anderes, wartet der Schritt (hoechstens WARTEN_HOECHSTENS_S).
- Ein laufender Auftrag je Projekt, hoechstens JE_KONTO je Konto. auftrag_abbrechen: der Schritt, der gerade laeuft, laeuft zu
  Ende, danach startet nichts mehr.
- Ein Neustart des Servers bricht einen laufenden Schritt ab (die Laeufe leben im Prozess): der Schritt gilt als gescheitert,
  der Auftrag endet, die Rueckmeldung sagt ehrlich, was fertig ist. Ein Auftrag zwischen zwei Schritten laeuft weiter.
- Nutzer und Projekt kommen aus der Sitzung bzw. der Karte; der Agent kann nur Schritte des Katalogs fuer eigene Projekte
  vorschlagen. Das Loeschen des Kontos nimmt die Tabelle mit (database.delete_user_data).
Doku: docs/INKLUAGENT.md, „Aufträge, Baustein 2: Auftrag mit Gesamtpreis“.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import funktionen
from database import get_db

from .daten import daten

log = logging.getLogger(__name__)

PRUEF_S = 10                       # Abstand der Rundgaenge
ANLAUF_S = 45                      # so lange darf ein Schritt brauchen, bis er als laufend sichtbar ist (wie die Rueckmeldung)
SCHRITT_HOECHSTENS_S = 6 * 3600    # laeuft ein Schritt laenger, gilt er als gescheitert (vor den zwoelf Stunden der Rueckmeldung)
WARTEN_HOECHSTENS_S = 2 * 3600     # so lange wartet ein Schritt, solange im Projekt etwas anderes laeuft
JE_KONTO = 3                       # laufende Auftraege je Konto
JE_RUNDGANG = 50                   # so viele laufende Auftraege je Rundgang, reihum
UEBERGAENGE_JE_RUNDGANG = 4        # so viele Schritte eines Auftrags hoechstens je Rundgang (synchrone Schritte nacheinander)
FEHLVERSUCHE = 3                   # so viele unerwartete Fehler hintereinander, dann gilt der Auftrag als gescheitert
STAND_LISTE = 5                    # so viele Auftraege nennt auftrag_stand
PAUSE_DROSSEL_S = 120              # Herunterladen gerade belegt oder gedrosselt: so lange wartet der Schritt, dann noch einmal
AUFBEWAHREN_TAGE = 30              # beendete Auftraege (mit Dokument- und Dateinamen) so lange, danach geloescht (Datenschutz)
AUFRAEUMEN_S = 3600                # so oft sieht der Rundgang nach alten Auftraegen

# Endzustaende eines Auftrags; „laeuft“ ist der einzige andere
ENDE = ("fertig", "gescheitert", "abgebrochen")


# ─── Katalog ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Schritt:
    """Ein Schritt, den ein Auftrag enthalten kann. titel ist ein msgid (dieselben Namen wie die Werkzeuge in der Oberflaeche).
    freigabe: die Art im Angebots-Schluessel des Werkzeugs ({ziel} und {format} werden ersetzt) — nur dafuer gibt der Auftrag
    die Zustimmung. hintergrund: Art fuer rueckmeldung.laeuft; leer = fertig, wenn das Werkzeug zurueckkehrt."""
    art: str
    werkzeug: str
    titel: str
    freigabe: str
    hintergrund: str = ""
    je_dokument: bool = False


# Reihenfolge = Reihenfolge im Auftrag (erst taggen, dann beschreiben, dann pruefen, dann herunterladen — wie die Kette)
KATALOG: tuple = (
    Schritt("tagging", "barrierefrei_machen", "Barrierefrei machen", "tagging", "tagging", True),
    Schritt("alt_texte", "alt_texte_generieren", "Alt-Texte generieren", "alt_texte", "alt_texte"),
    Schritt("quickinfos", "quickinfos_generieren", "Quickinfos generieren", "quickinfos", "quickinfos"),
    Schritt("pruefung", "pruefung_starten", "KI-basierte Prüfung", "pruefung", "pruefung", True),
    Schritt("uebersetzung", "uebersetze_dokument", "Übersetzen", "uebersetzung:{ziel}", "uebersetzung"),
    Schritt("alt_texte_tabelle", "exportiere_alt_texte", "Alt-Texte herunterladen", "tabelle_{format}"),
    Schritt("quickinfos_tabelle", "exportiere_quickinfos", "Quickinfos herunterladen", "quickinfo_csv"),
    Schritt("word", "exportiere_word", "Als Word herunterladen", "docx"),
    Schritt("pdfua", "konvertiere_zu_pdfua", "In barrierefreie PDF umwandeln", "pdfua"),
    Schritt("pdf", "exportiere_fertige_pdf", "PDF herunterladen", "pdf_export"),
)
NACH_ART = {s.art: s for s in KATALOG}
# Schritte, die etwas am Projekt aendern, das ein spaeteres Herunterladen berechnet (billing.pdf_download_preis)
_BEARBEITEN = frozenset({"tagging", "alt_texte", "quickinfos"})
# Schritte ohne Dokument-Auswahl (ihr Werkzeug nimmt das ganze Projekt) — mit document_id nur, wenn das Projekt nur dieses hat
_NUR_GANZES_PROJEKT = frozenset({"uebersetzung", "alt_texte_tabelle", "quickinfos_tabelle", "word", "pdfua"})
_FORMATE = {"csv": "CSV", "xlsx": "Excel", "json": "JSON"}
_WUNSCH_MAX = 20   # so viele Eintraege liest der Server aus „schritte“ (der Katalog hat zehn; der Rest waere Unsinn)

# Zustaende eines Schritts (msgids fuer auftrag_stand und die Ergebnisse an das Modell der Rueckmeldung)
SCHRITT_ZUSTAND = {"offen": "wartet", "laeuft": "läuft", "fertig": "fertig", "teilweise": "teilweise fertig",
                   "fehler": "gescheitert", "uebersprungen": "nichts zu tun", "ausgelassen": "nicht mehr gestartet"}
AUFTRAG_ZUSTAND = {"laeuft": "läuft", "fertig": "fertig", "gescheitert": "angehalten, weil ein Schritt nicht ging",
                   "abgebrochen": "abgebrochen"}

T_NEUSTART = "Der Server wurde neu gestartet, während dieser Schritt lief. Was bis dahin fertig war, bleibt."
T_ZU_LANG = "Der Schritt lief länger als sechs Stunden und wurde nicht weiter abgewartet."
T_BELEGT = "Im Projekt lief zwei Stunden lang etwas anderes; der Schritt wurde nicht gestartet."
T_PROJEKT_WEG = "Das Projekt wurde gelöscht."
T_DOKUMENT_WEG = "Das Dokument wurde gelöscht."
T_INTERN = "Ein unerwarteter Fehler auf dem Server hat den Auftrag angehalten."
T_UNTERBROCHEN = "Der Start dieses Schritts wurde unterbrochen; ob er lief, ist nicht sicher. Sieh bitte im Projekt nach."
T_NICHT_DA = "Diese Funktion gibt es in diesem Projekt nicht mehr."
T_SUMME = "Dieser Schritt würde jetzt {n} Credits kosten; der Auftrag gibt dafür noch {rest} frei. Er wurde nicht gestartet."
T_GUTHABEN = "Das Guthaben reichte für diesen Schritt nicht mehr."
T_VOM_NUTZER = "Der Lauf wurde in der Oberfläche abgebrochen."
T_NICHTS = "Als der Schritt dran war, gab es dafür nichts mehr zu tun."
T_SCHON_GETAGGT = "Das Dokument war inzwischen schon getaggt."
T_AUS = "Aufträge wurden auf dem Server ausgeschaltet; der Auftrag wurde angehalten."
T_AUFGEGEBEN = "Der Auftrag lief länger als zwei Tage und wurde nicht weiter abgewartet."
T_KEIN_ALT_TEXT = "Für kein Bild ist ein neuer Alt-Text entstanden."
T_KEINE_QUICKINFO = "Es ist keine neue Quickinfo entstanden."


def an() -> bool:
    """Auftraege brauchen die Karten (Erteilen nur ueber den Knopf) und die Rueckmeldung (Melden am Ende)."""
    from . import karten, rueckmeldung
    return bool(funktionen.an("AGENT_AUFTRAEGE") and karten.an() and rueckmeldung.an())


def _modul(name: str):
    return importlib.import_module(name)


def _jetzt() -> float:
    return time.time()


# ─── Was es im Projekt gibt ──────────────────────────────────────────────────────────────────────────────────────────────────
def _werkzeugsatz(project: dict, user_id: int):
    """(Namen der Werkzeuge, Executor) dieses Projekts — derselbe Satz wie im Gespraech und am Knopf einer Karte."""
    from .agent_loop import _werkzeugsatz as satz
    defs, executor, _system = satz(project, int(project["id"]), int(user_id))
    return {d["name"] for d in defs}, executor


def moeglich(project: dict, user_id: int) -> list[str]:
    """Die Schritte, die es in diesem Projekt gibt (Katalog-Reihenfolge)."""
    namen, _e = _werkzeugsatz(project, user_id)
    out = [s.art for s in KATALOG if s.werkzeug in namen]
    if "tagging" in out and funktionen.tagging_professionell_gesperrt():
        out.remove("tagging")
    return out


# ─── 1. Planen ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
class _Nicht(Exception):
    """Ein gewuenschter Schritt geht in diesem Projekt nicht (Grund fuer das Modell)."""


@dataclass
class _Rechnen:
    """Was die Vorschau eines Schritts braucht: Projekt, Dokumente, Wunsch, was im Plan davor steht, Hinweise fuer die Karte."""
    project: dict
    user_id: int
    docs: list
    wunsch: dict
    _: Callable
    davor: set = field(default_factory=set)               # Arten, die im Plan vor diesem Schritt stehen
    davor_eintraege: list = field(default_factory=list)   # ihre Eintraege (z. B. welche Dokumente getaggt werden)
    hinweise: list = field(default_factory=list)

    @property
    def pid(self) -> int:
        return int(self.project["id"])

    @property
    def dok(self) -> Optional[int]:
        """Das eine Dokument, auf das der Auftrag beschraenkt ist (document_id), sonst None = alle Dokumente des Projekts."""
        return self.wunsch.get("document_id")

    def nur_dok(self) -> dict:
        """Argumente fuer ein Werkzeug, das ein Dokument annimmt: mit Beschraenkung dessen document_id, sonst nichts."""
        return {"document_id": self.dok} if self.dok else {}


def _eintrag(c: _Rechnen, art: str, preis: int, umfang: str, args: Optional[dict] = None, document_id: Optional[int] = None,
             hoechstens: bool = False) -> dict:
    s = NACH_ART[art]
    return {"art": art, "werkzeug": s.werkzeug, "titel": s.titel, "hintergrund": s.hintergrund,
            "freigabe": s.freigabe.format(ziel=c.wunsch.get("zielsprache") or "", format=c.wunsch.get("format") or "csv"),
            "args": dict(args or {}), "document_id": document_id, "preis_geplant": int(preis or 0), "hoechstens": bool(hoechstens),
            "umfang": umfang, "zustand": "offen", "preis": 0}


def _menge(_, n: int, eins: str, viele: str, **mehr) -> str:
    """„ein Bild“ bzw. „3 Bilder“ (zwei msgids, damit es auch bei genau eins in jeder Sprache stimmt)."""
    return (_(eins) if int(n or 0) == 1 else _(viele)).format(n=n, **mehr)


def _dokname(d: dict) -> str:
    return d.get("display_name") or d.get("original_filename") or f"Dokument {d.get('doc_index') or ''}".strip()


def _ist_pdf(d: dict) -> bool:
    return (d.get("original_path") or "").lower().endswith(".pdf")


def _v_tagging(c: _Rechnen) -> list:
    _ = c._
    gesperrt = funktionen.tagging_professionell_gesperrt()
    if gesperrt:
        raise _Nicht(gesperrt)
    t, m, billing, pt = _modul("tagging_api"), _modul("main"), _modul("billing"), _modul("pdf_tagging")
    if not pt.verfuegbar():
        raise _Nicht("Das Tagging ist auf diesem Server nicht eingerichtet.")
    out = []
    for d in c.docs:
        if not _ist_pdf(d) or m._dokument_getaggt(d):
            continue
        grund = t.lesbar_grund(d)
        if grund:
            raise _Nicht(f"„{_dokname(d)}“: {grund}")
        seiten = t._seiten(d)
        if not seiten:
            raise _Nicht(f"„{_dokname(d)}“: Die PDF konnte nicht gelesen werden.")
        if seiten > pt.MAX_SEITEN:   # dieselbe Grenze wie das Tagging selbst (sonst scheitert der Schritt erst im Lauf)
            raise _Nicht(f"„{_dokname(d)}“ hat {seiten} Seiten; barrierefrei machen geht bis {pt.MAX_SEITEN} Seiten.")
        out.append(_eintrag(c, "tagging", billing.aktion_preis(t.AKTION, seiten),
                            _menge(_, seiten, "„{name}“, eine Seite", "„{name}“, {n} Seiten", name=_dokname(d)),
                            args={"document_id": d["id"]}, document_id=d["id"]))
    if not out:
        if c.dok:
            c.hinweise.append(_("Barrierefrei machen entfällt: „{name}“ ist schon getaggt.").format(name=_dokname(c.docs[0])))
        else:
            c.hinweise.append(_("Barrierefrei machen entfällt: Alle Dokumente sind schon getaggt."))
    return out


def _v_alt_texte(c: _Rechnen) -> list:
    _ = c._
    v = _modul("main")._generierung_vorschau_daten(c.pid, c.user_id, dict({"modus": "alle"}, **c.nur_dok()))
    if not v.get("anzahl"):
        c.hinweise.append(_("Alt-Texte generieren entfällt: Es gibt keine Bilder dafür."))
        return []
    if v.get("eigene"):
        c.hinweise.append(_("Alt-Texte generieren überschreibt {n} eigene Texte.").format(n=v["eigene"]))
    if "tagging" in c.davor:
        c.hinweise.append(_("Nach dem Tagging kann die Zahl der Bilder eine andere sein. Kosten die Alt-Texte dann mehr, hält der "
                            "Auftrag vorher an und ich sage es dir."))
    return [_eintrag(c, "alt_texte", v["preis"], _menge(_, v["anzahl"], "ein Bild", "{n} Bilder"), args=c.nur_dok(),
                     document_id=c.dok)]


def _v_quickinfos(c: _Rechnen) -> list:
    _ = c._
    v = _modul("formular_api").quickinfos_vorschau_daten(c.pid, c.user_id, c.dok)
    if not v.get("anzahl"):
        c.hinweise.append(_("Quickinfos generieren entfällt: Es gibt keine benannten Formularfelder."))
        return []
    return [_eintrag(c, "quickinfos", v["preis"], _menge(_, v["anzahl"], "ein Feld", "{n} Felder"), args=c.nur_dok(),
                     document_id=c.dok)]


def _v_pruefung(c: _Rechnen) -> list:
    _ = c._
    t, m, billing = _modul("tagging_api"), _modul("main"), _modul("billing")
    pdf_pruefung = _modul("pdf_pruefung")
    getaggt_im_plan = {e["document_id"] for e in c.davor_eintraege if e["art"] == "tagging"}
    out = []
    for d in c.docs:
        if not _ist_pdf(d) or not (m._dokument_getaggt(d) or d["id"] in getaggt_im_plan):
            continue
        seiten = min(t._seiten(d), pdf_pruefung.MAX_SEITEN)
        if not seiten:
            continue
        out.append(_eintrag(c, "pruefung", billing.aktion_preis(t.AKTION_PRUEFUNG, seiten),
                            _menge(_, seiten, "„{name}“, eine Seite", "„{name}“, {n} Seiten", name=_dokname(d)),
                            args={"document_id": d["id"]}, document_id=d["id"]))
    if not out:
        raise _Nicht("Die KI-basierte Prüfung geht nur für getaggte Dokumente. Nimm „Barrierefrei machen“ (tagging) dazu.")
    return out


def _v_uebersetzung(c: _Rechnen) -> list:
    _ = c._
    ue = _modul("uebersetzung_api")
    ziel = (c.wunsch.get("zielsprache") or "").strip().lower()
    if ziel not in ue.ue.ZIELSPRACHEN:
        raise _Nicht("Für das Übersetzen fehlt die Zielsprache (zielsprache, z. B. en-gb, fr, es).")
    v = ue.bot_vorschau(c.user_id, c.pid, ziel, True)
    sprache = ue.ue.ZIELSPRACHEN[ziel][0]
    if not v.get("anzahl"):
        c.hinweise.append(_("Übersetzen entfällt: In dieser Sprache ist schon alles übersetzt, oder es gibt keinen Text."))
        return []
    umfang = _menge(_, v["woerter"], "{sprache}, ein Wort", "{sprache}, {n} Wörter", sprache=_(sprache))
    return [_eintrag(c, "uebersetzung", v["preis"], umfang, args={"zielsprache": ziel, "alt_texte": True})]


def _v_alt_texte_tabelle(c: _Rechnen) -> list:
    billing = _modul("billing")
    fmt = c.wunsch.get("format") or "csv"
    p = billing.aktion_pruefung(c.user_id, billing.TABELLEN_EXPORTE[fmt])
    return [_eintrag(c, "alt_texte_tabelle", p["preis"], _FORMATE[fmt], args={"format": fmt})]


def _v_quickinfos_tabelle(c: _Rechnen) -> list:
    billing = _modul("billing")
    p = billing.aktion_pruefung(c.user_id, billing.TABELLEN_EXPORTE["formular_csv"])
    return [_eintrag(c, "quickinfos_tabelle", p["preis"], "CSV")]


def _v_export(art: str, kosten_art: str):
    def vorschau(c: _Rechnen) -> list:
        _ = c._
        from .tools import ausgaben
        v = ausgaben._kosten_vorschau(_modul("main"), c.project, c.user_id, None, kosten_art)
        return [_eintrag(c, art, v["preis"], _menge(_, v["bilder"], "ein Bild", "{n} Bilder"))]
    return vorschau


def _v_pdf(c: _Rechnen) -> list:
    """„PDF herunterladen“ kostet nur, was in InkluDocs bearbeitet wurde (main._pdf_export_plan). Bearbeitet der Auftrag vorher
    (Tagging, Alt-Texte, Quickinfos), ist der Preis jetzt noch nicht bekannt: dann die Obergrenze — als waeren alle Bilder der
    getaggten (oder im Auftrag getaggten) Dokumente und alle Felder bearbeitet. Mehr kann es nicht werden, solange die Zahl
    der Bilder bleibt (billing.pdf_download_preis waechst mit beiden; Alt-Texte stehen nur in getaggten Dateien)."""
    _ = c._
    m, billing = _modul("main"), _modul("billing")
    alle = len(c.docs) > 1   # mit document_id steht in c.docs nur dieses Dokument
    args = {"alle": True} if alle else c.nur_dok()
    umfang = _("alle Dokumente als ZIP") if alle else _("ein Dokument")
    if c.davor & _BEARBEITEN:
        getaggt = ({d["id"] for d in c.docs if m._dokument_getaggt(d)}
                   | {e["document_id"] for e in c.davor_eintraege if e["art"] == "tagging"})
        conn = get_db()
        try:
            bilder = conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ? AND document_id IN (%s)"
                                  % ",".join("?" * len(getaggt)), (c.pid, *getaggt)).fetchone()[0] if getaggt else 0
            felder = conn.execute("SELECT COUNT(*) FROM formularfelder WHERE project_id = ?" + (" AND document_id = ?" if c.dok else ""),
                                  (c.pid, *([c.dok] if c.dok else []))).fetchone()[0]
        finally:
            conn.close()
        return [_eintrag(c, "pdf", billing.pdf_download_preis(bilder, felder)["preis"], umfang, args=args, document_id=c.dok,
                         hoechstens=True)]
    units = m._load_pdf_export_units(c.project, c.user_id, c.dok)
    return [_eintrag(c, "pdf", m._pdf_export_plan(c.user_id, units)["preis"], umfang, args=args, document_id=c.dok)]


_VORSCHAU: dict[str, Callable[[_Rechnen], list]] = {
    "tagging": _v_tagging, "alt_texte": _v_alt_texte, "quickinfos": _v_quickinfos, "pruefung": _v_pruefung,
    "uebersetzung": _v_uebersetzung, "alt_texte_tabelle": _v_alt_texte_tabelle, "quickinfos_tabelle": _v_quickinfos_tabelle,
    "word": _v_export("word", "docx"), "pdfua": _v_export("pdfua", "pdfua"), "pdf": _v_pdf,
}


def _dokument_wunsch(wert: Any) -> Optional[int]:
    """document_id aus den Argumenten: None = alle Dokumente; eine ganze Zahl (auch „12“ oder 12.0, wie Modelle sie schicken);
    alles andere -1 (planen() nennt es als Fehler, statt still das ganze Projekt zu nehmen)."""
    if wert is None or wert == "":
        return None
    if isinstance(wert, bool):
        return -1
    try:
        f = float(str(wert).strip())
    except (TypeError, ValueError):
        return -1
    return int(f) if f.is_integer() and f > 0 else -1


def wunsch_aus(args: dict) -> dict:
    """Die Argumente des Modells als Wunsch: bekannte Arten in der genannten Reihenfolge (doppelte einmal), Zielsprache, Format,
    Dokument (document_id, Michael 15.10.2026: „Mach das Dokument 2 barrierefrei …“). Unbekannte Arten bleiben drin — planen()
    nennt sie als Fehler."""
    roh = (args or {}).get("schritte")
    roh = roh if isinstance(roh, list) else ([roh] if isinstance(roh, str) else [])
    schritte: list[str] = []
    for a in roh[:_WUNSCH_MAX]:
        a = str(a or "").strip().lower()[:40]
        if a and a not in schritte:
            schritte.append(a)
    fmt = str((args or {}).get("format") or "csv").strip().lower()[:10]
    return {"schritte": schritte, "zielsprache": str((args or {}).get("zielsprache") or "").strip().lower()[:10], "format": fmt,
            "document_id": _dokument_wunsch((args or {}).get("document_id"))}


def planen(user_id: int, project: dict, wunsch: dict, _: Optional[Callable[[str], str]] = None) -> dict:
    """Plan fuer die Karte und fuer das Erteilen (dieselbe Rechnung zweimal: Karte und Knopf). Rueckgabe:
    {eintraege, summe, hoechstens, verfuegbar, erlaubt, hinweise, fehler}. fehler (fuer das Modell): Schritte, die es nicht gibt
    oder die hier nicht gehen — dann gibt es keine Karte."""
    _ = _ or (lambda s: s)
    billing = _modul("billing")
    fehler: list[str] = []
    gibt_es = moeglich(project, user_id)
    gewuenscht = []
    for a in wunsch.get("schritte") or []:
        if a not in NACH_ART:
            fehler.append(f"Einen Schritt „{a}“ gibt es nicht.")
        elif a == "tagging" and funktionen.tagging_professionell_gesperrt():
            fehler.append(funktionen.tagging_professionell_gesperrt())
        elif a not in gibt_es:
            fehler.append(f"„{NACH_ART[a].titel}“ gibt es in diesem Projekt nicht.")
        else:
            gewuenscht.append(a)
    if wunsch.get("format") not in _FORMATE and "alt_texte_tabelle" in gewuenscht:
        fehler.append("Format für die Alt-Texte: csv, xlsx (Excel) oder json.")
    conn = get_db()
    try:
        docs = [dict(r) for r in conn.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index, id",
                                              (project["id"],)).fetchall()]
    finally:
        conn.close()
    dokument = None
    if wunsch.get("document_id") is not None:
        # Nur dieses Dokument: Tagging, Alt-Texte, Quickinfos, Pruefung und Herunterladen nehmen seine document_id; was es nur
        # fuers ganze Projekt gibt (Uebersetzen, Tabellen, Word, Umwandeln), geht dann nur, wenn das Projekt nur dieses hat
        nur = [d for d in docs if d["id"] == wunsch["document_id"]]
        if not nur:
            fehler.append("Dieses Dokument gibt es in diesem Projekt nicht. Nimm die document_id aus dokument_stand oder lass "
                          "document_id weg (dann alle Dokumente).")
        else:
            if len(docs) > 1:
                fehler += [f"„{NACH_ART[a].titel}“ geht im Auftrag nur für das ganze Projekt, nicht für ein einzelnes Dokument."
                           for a in gewuenscht if a in _NUR_GANZES_PROJEKT]
            docs, dokument = nur, _dokname(nur[0])
    c = _Rechnen(project=project, user_id=int(user_id), docs=docs, wunsch=wunsch, _=_)
    eintraege: list[dict] = []
    if not fehler:
        for s in KATALOG:
            if s.art not in gewuenscht:
                continue
            try:
                neu = _VORSCHAU[s.art](c)
            except _Nicht as e:
                fehler.append(str(e))
                continue
            except Exception as e:  # noqa: BLE001 — HTTPException der Kernfunktion (404, 409 …) oder Unerwartetes
                detail = getattr(e, "detail", None)
                if detail is None:
                    log.exception("Auftrag: Vorschau %s fehlgeschlagen", s.art)
                fehler.append(f"„{s.titel}“: {detail if isinstance(detail, str) else 'geht gerade nicht.'}")
                continue
            eintraege += neu
            c.davor_eintraege += neu
            if neu:
                c.davor.add(s.art)
    summe = sum(e["preis_geplant"] for e in eintraege)
    verf = billing.verfuegbare_credits(int(user_id))
    return {"eintraege": eintraege, "summe": summe, "hoechstens": any(e["hoechstens"] for e in eintraege), "verfuegbar": verf,
            "erlaubt": verf is None or verf >= summe, "hinweise": c.hinweise, "fehler": fehler, "dokument": dokument}


def _preis_text(e: dict, _) -> str:
    if not e["preis_geplant"]:
        return _("kostenlos")
    return (_("höchstens {n} Credits") if e["hoechstens"] else _("{n} Credits")).format(n=e["preis_geplant"])


def karte_zeilen(plan: dict, _) -> list[str]:
    """Die Schritte fuer die Karte mit Umfang und Preis — die Oberflaeche nummeriert sie (nummerierte Liste)."""
    return [f"{_(e['titel'])} ({e['umfang']}): {_preis_text(e, _)}" if e["umfang"] else f"{_(e['titel'])}: {_preis_text(e, _)}"
            for e in plan["eintraege"]]


def karte_knopf(plan: dict, _) -> str:
    """„Auftrag erteilen (45 Credits)“: der Knopf nennt die Summe — wer per Tab oder Rotor „Formulare“ hinkommt, hoert den Preis
    beim Knopf (wie die Bestaetigungs-Karten seit Pruefung 4; Pruefung Auftraege N7)."""
    summe = {"preis_geplant": plan["summe"], "hoechstens": plan["hoechstens"]}
    return _("Auftrag erteilen ({preis})").format(preis=_preis_text(summe, _))


def fingerabdruck(plan: dict) -> list:
    """Was die Karte zeigte: je Schritt Art, Dokument und Preis — das Erteilen vergleicht mehr als die Summe (Pruefung N4: Dokument
    A geloescht, B mit gleicher Seitenzahl hochgeladen, gleiche Summe — die Karte zeigte A)."""
    return [[e["art"], e.get("document_id"), e["preis_geplant"]] for e in plan["eintraege"]]


def karte_text(projekt: str, plan: dict, _) -> str:
    teile = [_("Projekt „{projekt}“, Dokument „{dokument}“.").format(projekt=projekt, dokument=plan["dokument"])
             if plan.get("dokument") else _("Projekt „{projekt}“.").format(projekt=projekt)]
    summe = (_("Zusammen höchstens {n} Credits.") if plan["hoechstens"] else _("Zusammen {n} Credits.")).format(n=plan["summe"])
    teile.append(summe)
    if plan["verfuegbar"] is not None:
        teile.append(_("Dein Guthaben: {n} Credits.").format(n=plan["verfuegbar"]))
    teile.append(_("Die Schritte laufen nacheinander im Hintergrund; du kannst die Seite danach schließen."))
    return " ".join(teile)


# ─── Werkzeuge des Agenten ───────────────────────────────────────────────────────────────────────────────────────────────────
WERKZEUG_VORBEREITEN = "auftrag_vorbereiten"
WERKZEUG_STAND = "auftrag_stand"
WERKZEUG_ABBRECHEN = "auftrag_abbrechen"
NAMEN = frozenset({WERKZEUG_VORBEREITEN, WERKZEUG_STAND, WERKZEUG_ABBRECHEN})

_SCHRITTE_BESCHREIBUNG = (
    "tagging = Barrierefrei machen (alle ungetaggten PDFs), alt_texte = Alt-Texte für alle Bilder generieren, quickinfos = "
    "Quickinfos für alle benannten Formularfelder, pruefung = KI-basierte Prüfung der getaggten PDFs, uebersetzung = Word-Dokument "
    "übersetzen (zielsprache nötig; die übersetzte Datei kommt dazu), alt_texte_tabelle = Alt-Texte als Tabelle herunterladen "
    "(format), quickinfos_tabelle = Quickinfos als CSV, word = Word-Datei mit Alt-Texten, pdfua = Word in barrierefreie PDF "
    "umwandeln, pdf = fertige PDF herunterladen (mehrere Dokumente als ZIP). Mit document_id gelten tagging, alt_texte, quickinfos, "
    "pruefung und pdf nur für dieses Dokument.")


def werkzeuge(project_id: Optional[int]) -> list[dict]:
    pid = {"project_id": {"type": "integer", "description": "Welches Projekt (Kennung aus projekte_auflisten)"}} if project_id is None else {}
    return [
        {"name": WERKZEUG_VORBEREITEN,
         "description": ("Mehrere Arbeitsschritte in einem Projekt als EIN Auftrag: Der Server ordnet die Schritte, rechnet jeden "
                         "Preis und die Summe und legt dem Nutzer EINE Karte mit Knopf „Auftrag erteilen“ vor. Nach dem Knopf "
                         "arbeitet der Server die Schritte im Hintergrund ab (die Seite darf zu), und der InkluAgent meldet sich, "
                         "wenn alles fertig ist. Erteilen kann NUR der Nutzer mit dem Knopf, nicht mit einem getippten Ja. Für "
                         "einen einzelnen Schritt nimm sein eigenes Werkzeug. Schritte: " + _SCHRITTE_BESCHREIBUNG),
         "input_schema": {"type": "object", "properties": dict(pid, **{
             "schritte": {"type": "array", "items": {"type": "string", "enum": [s.art for s in KATALOG]},
                          "description": "Die gewünschten Schritte; die Reihenfolge legt der Server fest."},
             "document_id": {"type": "integer", "description": (
                 "Nur wenn der Nutzer ein bestimmtes Dokument meint (z. B. „Dokument 2“): dessen Kennung aus dokument_stand. Ohne "
                 "gilt der Auftrag für alle Dokumente des Projekts.")},
             "zielsprache": {"type": "string", "description": "Nur für uebersetzung: z. B. en-gb, en, fr, es-419."},
             "format": {"type": "string", "enum": list(_FORMATE), "description": "Nur für alt_texte_tabelle. Standard csv."},
             "per_mail": {"type": "boolean", "description": (
                 "true nur, wenn der Nutzer selbst eine Mail haben will, wenn alles fertig ist (Haken auf der Karte vorbelegt; "
                 "die Mail geht immer an die Adresse seines Kontos).")}}),
             "required": ["schritte"] + (["project_id"] if project_id is None else [])}},
        {"name": WERKZEUG_STAND,
         "description": ("Stand der Aufträge" + (" in diesem Projekt" if project_id is not None else " des Kontos") +
                         ": je Auftrag die Schritte mit ihrem Zustand (wartet, läuft, fertig, gescheitert). Kostenlos."),
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": WERKZEUG_ABBRECHEN,
         "description": ("Den laufenden Auftrag " + ("in diesem Projekt" if project_id is not None else "eines Projekts") +
                         " anhalten: Der Schritt, der gerade läuft, läuft zu Ende; danach startet nichts mehr, und was nicht mehr "
                         "läuft, kostet nichts. Nur auf ausdrücklichen Wunsch des Nutzers. Kostenlos."),
         "input_schema": {"type": "object", "properties": dict(pid), "required": ["project_id"] if project_id is None else []}},
    ]


PROMPT = """Aufträge mit mehreren Schritten

Will der Nutzer mehrere Arbeitsschritte in einem Projekt nacheinander erledigt haben (zum Beispiel Alt-Texte generieren, dann als Word herunterladen; oder barrierefrei machen, Alt-Texte, dann die fertige PDF), leg ihm mit auftrag_vorbereiten EINE Karte vor, statt jeden Schritt einzeln anzubieten. Das gilt auch, wenn er das Wort „Auftrag“ nicht benutzt. Meint er ein bestimmtes Dokument, gib dessen document_id mit; sonst gilt der Auftrag für alle Dokumente des Projekts. Der Server ordnet die Schritte, rechnet die Preise und die Summe und zeigt alles auf der Karte. Erteilen kann nur der Nutzer mit dem Knopf „Auftrag erteilen“ — frag nicht nach einem Ja, sondern sag in einem Satz, was der Auftrag umfasst, was er zusammen kostet und dass er danach die Seite schließen kann. Der Server arbeitet die Schritte im Hintergrund ab, und du meldest dich von selbst, wenn alles fertig ist; eine Mail gibt es, wenn er den Haken auf der Karte setzt. Geht ein Schritt nicht oder reicht das Guthaben nicht, sagt dir das Werkzeug den Grund: Nenne ihn und frag, wie er weitermachen will. Fragt er, wie weit der Auftrag ist: auftrag_stand. Will er ihn anhalten: auftrag_abbrechen."""

# PDF: das Beispiel aus Michaels Fall (15.10.2026) und das schon getaggte Dokument — nur dort, wo es PDF-Projekte gibt. Ohne
# Projekt gibt es barrierefrei_machen nicht (der Werkzeug-Waechter fand den Namen in Word-, Grafik- und Konto-Gespraechen).
_PDF = ("In PDF-Projekten: „Mach die PDF barrierefrei, erzeug die Alt-Texte und gib mir dann die fertige PDF“ ist ein Auftrag mit den "
        "Schritten tagging, alt_texte und pdf. Ist ein Dokument schon getaggt, lässt der Server das Taggen weg und sagt es auf der "
        "Karte; neu taggen geht nur als einzelner Schritt")


def prompt(projektart: str) -> str:
    """PROMPT fuer die Projektart (prompts/schubfaecher.py): im PDF-Projekt und ohne Projekt mit den PDF-Saetzen."""
    if projektart == "pdf":
        return PROMPT + "\n\n" + _PDF + " (barrierefrei_machen)."
    if projektart == "konto":
        return PROMPT + "\n\n" + _PDF + " im Projekt."
    return PROMPT


def ausfuehren(name: str, args: dict, *, user_id: int, project_id: Optional[int], gespraech_id: Optional[int],
               sprache: Optional[str] = None, auftrag: str = "", _t: Optional[Callable[[str], str]] = None) -> dict:
    """Werkzeuge des Agenten (agent_loop). Wie jedes Werkzeug: ein Fehler wird ein Ergebnis, nie ein Absturz der Runde."""
    _ = _t or (lambda s: s)
    args = args if isinstance(args, dict) else {}
    try:
        if name == WERKZEUG_VORBEREITEN:
            return _vorbereiten(args, user_id, project_id, gespraech_id, sprache, auftrag, _)
        if name == WERKZEUG_STAND:
            return _stand_werkzeug(user_id, project_id, _)
        if name == WERKZEUG_ABBRECHEN:
            return _abbrechen_werkzeug(args, user_id, project_id)
    except Exception:  # noqa: BLE001
        log.exception("Auftrag: Werkzeug %s fehlgeschlagen", name)
        return {"ok": False, "error": "Das ging gerade nicht. Bitte versuch es gleich noch einmal."}
    return {"ok": False, "error": f"Unbekanntes Tool: {name}"}


def _eigenes_projekt(user_id: int, project_id: Any) -> Optional[dict]:
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return None
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (pid, int(user_id))).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def _projektname(p: dict) -> str:
    return (p.get("name") or "").strip() or p.get("filename") or ""


def _laufender(conn, user_id: int, project_id: int) -> Optional[dict]:
    r = conn.execute("SELECT * FROM agent_auftraege WHERE user_id = ? AND project_id = ? AND zustand = 'laeuft' ORDER BY id DESC "
                     "LIMIT 1", (int(user_id), int(project_id))).fetchone()
    return dict(r) if r else None


def _vorbereiten(args: dict, user_id: int, project_id: Optional[int], gespraech_id: Optional[int], sprache: Optional[str],
                 auftrag: str, _) -> dict:
    from . import karten, rueckmeldung
    p = _eigenes_projekt(user_id, project_id if project_id is not None else args.get("project_id"))
    if not p:
        return {"ok": False, "error": "Projekt nicht gefunden (nur eigene Projekte). Frag, welches Projekt gemeint ist."}
    if not gespraech_id:
        return {"ok": False, "error": "Aufträge gibt es nur im Gespräch des Kontos."}
    conn = get_db()
    try:
        schon = _laufender(conn, user_id, p["id"])
    finally:
        conn.close()
    projekt_daten = daten(_projektname(p))
    if schon:
        return {"ok": True, "result": {"karte": None, "projekt_daten": projekt_daten, "hinweis": (
            "In diesem Projekt läuft schon ein Auftrag. Sag das dem Nutzer; den Stand liest du mit auftrag_stand. Einen neuen "
            "Auftrag gibt es, wenn er fertig ist.")}}
    wunsch = wunsch_aus(args)
    if not wunsch["schritte"]:
        return {"ok": False, "error": "Nenne die Schritte des Auftrags (schritte)."}
    plan = planen(user_id, p, wunsch, _)
    if plan["fehler"]:
        return {"ok": True, "result": {"karte": None, "projekt_daten": projekt_daten, "nicht_moeglich": plan["fehler"],
                                       "moegliche_schritte": moeglich(p, user_id), "hinweis": (
            "Daraus wird kein Auftrag. Nenne dem Nutzer den Grund in einfachen Worten und frag, wie er weitermachen will "
            "(etwa ohne diesen Schritt).")}}
    if not plan["eintraege"]:
        return {"ok": True, "result": {"karte": None, "projekt_daten": projekt_daten, "nichts_zu_tun": plan["hinweise"],
                                       "hinweis": "Es gibt nichts zu tun. Sag dem Nutzer, warum."}}
    schritte = [{"schritt": e["titel"], "umfang_daten": daten(e["umfang"]), "preis": e["preis_geplant"],
                 "hoechstens": e["hoechstens"]} for e in plan["eintraege"]]
    if not plan["erlaubt"]:
        fehlend = max(0, plan["summe"] - int(plan["verfuegbar"] or 0))
        kaufen = (" Frag, ob du ihm die Karte „Zusatz-Credits kaufen“ bringen sollst (credits_kaufen_vorbereiten, paket "
                  f"{_passendes_paket(fehlend)}); sagt er ja, bring sie." if karten.kaufen_an() else
                  " Verweise auf „Abo & Verbrauch“.")
        return {"ok": True, "result": {"karte": None, "projekt_daten": projekt_daten, "schritte": schritte, "summe": plan["summe"],
                                       "verfuegbar": plan["verfuegbar"], "fehlend": fehlend, "guthaben_reicht_nicht": True,
                                       "hinweis": "Das Guthaben reicht für diesen Auftrag nicht. Nenne Summe und Guthaben." + kaufen}}
    kontext = {"project_id": int(p["id"]), "wunsch": wunsch, "summe": plan["summe"], "fingerabdruck": fingerabdruck(plan),
               "gespraech_id": int(gespraech_id), "sprache": sprache or "de", "auftrag": (auftrag or "").strip()[:1000]}
    kid = karten._merken(user_id, karten.AUFTRAG, None if project_id is None else int(p["id"]), kontext)
    anhang = karten.karte(karten.AUFTRAG, kid, {"per_mail": rueckmeldung._ja(args.get("per_mail"))}, _,
                          text=karte_text(_projektname(p), plan, _), zeilen=karte_zeilen(plan, _), nummeriert=True,
                          hinweise=plan["hinweise"], knopf=karte_knopf(plan, _))
    return {"ok": True,
            "result": {"karte": "auftrag", "projekt_daten": projekt_daten, "schritte": schritte, "summe": plan["summe"],
                       "hoechstens": plan["hoechstens"], "verfuegbar": plan["verfuegbar"], "hinweise": plan["hinweise"],
                       "hinweis": ("Die Karte steht unter deiner Antwort. Sag in ein bis zwei Sätzen, was der Auftrag umfasst und was "
                                   "er zusammen kostet; erteilen kann nur der Nutzer mit „Auftrag erteilen“, danach darf er die "
                                   "Seite schließen.")},
            "anhang": anhang}


def _passendes_paket(fehlend: int) -> int:
    """Das kleinste Zusatzpaket, das die fehlenden Credits deckt (sonst das groesste) — Vorschlag fuer die Kauf-Karte."""
    groessen = sorted(_modul("billing").PAKET_PREISE)
    return next((g for g in groessen if g >= fehlend), groessen[-1])


def _schritte_kurz(a: dict) -> list:
    return [dict({"schritt": s["titel"], "umfang_daten": daten(s.get("umfang") or ""), "zustand": SCHRITT_ZUSTAND.get(s["zustand"],
                                                                                                                  s["zustand"])},
                 **({"grund": s["grund"]} if s.get("grund") else {})) for s in a["schritte"]]


def _stand_werkzeug(user_id: int, project_id: Optional[int], _) -> dict:
    conn = get_db()
    try:
        sql = ("SELECT a.*, COALESCE(NULLIF(TRIM(p.name), ''), p.filename, '') AS projekt FROM agent_auftraege a "
               "LEFT JOIN projects p ON p.id = a.project_id WHERE a.user_id = ?")
        werte: list = [int(user_id)]
        if project_id is not None:
            sql += " AND a.project_id = ?"
            werte.append(int(project_id))
        rows = [_zeile(dict(r)) for r in conn.execute(sql + " ORDER BY (a.zustand = 'laeuft') DESC, a.id DESC LIMIT ?",
                                                      werte + [STAND_LISTE]).fetchall()]
    finally:
        conn.close()
    if not rows:
        return {"ok": True, "result": {"auftraege": [], "hinweis": "Es gibt keinen Auftrag. Sag das dem Nutzer."}}
    return {"ok": True, "result": {"auftraege": [
        {"auftrag_id": a["id"], "project_id": a["project_id"], "projekt_daten": daten(a.get("projekt") or ""),
         "zustand": AUFTRAG_ZUSTAND.get(a["zustand"], a["zustand"]), "abbrechen_angefordert": bool(a["abbrechen"]),
         "credits_freigegeben": a["freigegeben"], "angelegt_am": a["angelegt_am"], "schritte": _schritte_kurz(a)}
        for a in rows], "hinweis": ("Sag dem Nutzer knapp, welcher Schritt gerade läuft und was schon fertig ist. Zeiten "
                                    "schätzt du nicht.")}}


def _abbrechen_werkzeug(args: dict, user_id: int, project_id: Optional[int]) -> dict:
    p = _eigenes_projekt(user_id, project_id if project_id is not None else args.get("project_id"))
    if not p:
        return {"ok": False, "error": "Projekt nicht gefunden (nur eigene Projekte)."}
    if not abbrechen(user_id, p["id"]):
        return {"ok": True, "result": {"angehalten": False, "projekt_daten": daten(_projektname(p)),
                                       "hinweis": "In diesem Projekt läuft kein Auftrag. Sag das dem Nutzer."}}
    return {"ok": True, "result": {"angehalten": True, "projekt_daten": daten(_projektname(p)), "hinweis": (
        "Der Schritt, der gerade läuft, läuft zu Ende; danach startet nichts mehr. Du meldest dich wie angekündigt, wenn er zu "
        "Ende ist. Sag das dem Nutzer in einem Satz.")}}


def abbrechen(user_id: int, project_id: int) -> bool:
    """Den laufenden Auftrag des Projekts anhalten (auch von konto.lauf_abbrechen). True = es lief einer."""
    conn = get_db()
    try:
        n = conn.execute("UPDATE agent_auftraege SET abbrechen = 1 WHERE user_id = ? AND project_id = ? AND zustand = 'laeuft'",
                         (int(user_id), int(project_id))).rowcount
        conn.commit()
    finally:
        conn.close()
    if n:
        _wecken()
    return bool(n)


# ─── 2. Erteilen (Knopf der Karte) ───────────────────────────────────────────────────────────────────────────────────────────
async def karte_senden(e: dict, user: dict, werte: dict, sprache: str, _) -> tuple[int, dict]:
    """karten.absenden fuer die Auftragskarte: (HTTP-Status, Antwort) wie die anderen Karten."""
    return await asyncio.get_running_loop().run_in_executor(None, erteilen, e["kontext"], user, werte, _)


def _abgelaufen(text: str, status: int = 409) -> tuple[int, dict]:
    return status, {"detail": text, "karte": {"zustand": "abgelaufen", "ergebnis": text}}


def erteilen(kontext: dict, user: dict, werte: dict, _) -> tuple[int, dict]:
    """Den Auftrag anlegen — nur mit dem Knopf der Karte. Der Plan wird neu gerechnet; eine andere Summe = die Karte gilt nicht
    mehr. Guthaben, ein Auftrag je Projekt, JE_KONTO und die Rueckmeldung werden geprueft; Auftrag und Rueckmeldung entstehen in
    EINER Transaktion. 402/409 ohne „abgelaufen“ lassen die Karte offen (Guthaben aufladen, den laufenden Auftrag abwarten)."""
    from . import rueckmeldung
    if not an():
        return _abgelaufen(_("Diese Karte gilt nicht mehr. Bitte frag im Chat neu."), 404)
    uid = int(user["id"])
    p = _eigenes_projekt(uid, kontext.get("project_id"))
    if not p:
        return _abgelaufen(_("Das Projekt gibt es nicht mehr."), 404)
    plan = planen(uid, p, kontext["wunsch"], _)
    if (plan["fehler"] or not plan["eintraege"] or plan["summe"] != int(kontext["summe"])
            or fingerabdruck(plan) != kontext.get("fingerabdruck")):
        return _abgelaufen(_("Umfang oder Preis haben sich geändert. Bitte frag im Chat neu."))
    if not plan["erlaubt"]:
        return 402, {"detail": _("Dein Guthaben reicht nicht: {f} Credits fehlen.").format(
            f=max(0, plan["summe"] - int(plan["verfuegbar"] or 0)))}
    per_mail = werte.get("per_mail") is True
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM projects WHERE id = ? AND user_id = ?", (p["id"], uid)).fetchone():
            conn.rollback()
            return _abgelaufen(_("Das Projekt gibt es nicht mehr."), 404)
        if _laufender(conn, uid, p["id"]):
            conn.rollback()
            return 409, {"detail": _("In diesem Projekt läuft schon ein Auftrag. Warte bitte, bis er fertig ist.")}
        if conn.execute("SELECT COUNT(*) FROM agent_auftraege WHERE user_id = ? AND zustand = 'laeuft'", (uid,)).fetchone()[0] >= JE_KONTO:
            conn.rollback()
            return 409, {"detail": _("Es laufen schon {n} Aufträge. Warte bitte, bis einer fertig ist.").format(n=JE_KONTO)}
        aid = conn.execute("INSERT INTO agent_auftraege (user_id, project_id, gespraech_id, sprache, schritte, freigegeben) "
                           "VALUES (?, ?, ?, ?, ?, ?)",
                           (uid, p["id"], kontext.get("gespraech_id"), kontext.get("sprache") or "de",
                            json.dumps(plan["eintraege"], ensure_ascii=False), plan["summe"])).lastrowid
        if not rueckmeldung.anmelden_in(conn, uid, kontext.get("gespraech_id"), p["id"], "auftrag", per_mail,
                                        kontext.get("sprache"), kontext.get("auftrag") or "", auftrag_id=aid):
            conn.rollback()
            return 409, {"detail": _("Es warten schon zu viele Rückmeldungen. Warte bitte, bis eine da ist.")}
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    log.info("Auftrag %s: erteilt (Konto %s, Projekt %s, %d Schritte, %d Credits)", aid, uid, p["id"], len(plan["eintraege"]),
             plan["summe"])
    _wecken()
    text = (_("Auftrag erteilt. Die Schritte laufen jetzt nacheinander im Hintergrund; du kannst die Seite schließen.") + " "
            + _(rueckmeldung.T_MELDE_MICH_MAIL if per_mail else rueckmeldung.T_MELDE_MICH))
    return 200, {"ok": True, "text": text, "karte": {"zustand": "gesendet", "ergebnis": text}, "auftrag_id": aid}


# ─── 3. Abarbeiten (ohne KI) ─────────────────────────────────────────────────────────────────────────────────────────────────
def _laden(aid: int) -> Optional[dict]:
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM agent_auftraege WHERE id = ?", (int(aid),)).fetchone()
    finally:
        conn.close()
    return _zeile(dict(r)) if r else None


def _zeile(a: dict) -> dict:
    a["schritte_roh"] = a.get("schritte") or "[]"   # fuer _beanspruchen: genau dieser Stand in der Datenbank
    try:
        a["schritte"] = json.loads(a.get("schritte") or "[]")
    except ValueError:
        a["schritte"] = []
    return a


def _speichern(a: dict, **felder) -> bool:
    """Schritte, Verbrauch und uebergebene Felder schreiben — nie „abbrechen“ (das setzt nur auftrag_abbrechen) und nur, solange
    der Auftrag laeuft: wurde er inzwischen anderswo beendet (aufgegeben, ausgeschaltet), bleibt dessen Stand (Pruefung M4).
    True = geschrieben."""
    roh = json.dumps(a["schritte"], ensure_ascii=False)
    felder = dict(felder, schritte=roh, verbraucht=int(a["verbraucht"]))
    conn = get_db()
    try:
        n = conn.execute("UPDATE agent_auftraege SET " + ", ".join(f"{k} = ?" for k in felder) + ", geprueft_am = datetime('now') "
                         "WHERE id = ? AND zustand = 'laeuft'", (*felder.values(), a["id"])).rowcount
        conn.commit()
    finally:
        conn.close()
    if n:
        a["schritte_roh"] = roh
    return bool(n)


def _beanspruchen(a: dict) -> bool:
    """Den Start des naechsten Schritts in EINEM bedingten Schritt festhalten: nur, wenn die Schritte noch genau so in der
    Datenbank stehen, wie dieser Rundgang sie gelesen hat. Zwei Abarbeiter (zwei Prozesse, ein ueberlappender Container beim
    Neustart) starten so nie denselben Schritt (Pruefung N1). True = beansprucht."""
    roh = json.dumps(a["schritte"], ensure_ascii=False)
    conn = get_db()
    try:
        n = conn.execute("UPDATE agent_auftraege SET schritte = ?, geprueft_am = datetime('now') WHERE id = ? AND zustand = 'laeuft' "
                         "AND schritte = ?", (roh, a["id"], a["schritte_roh"])).rowcount
        conn.commit()
    finally:
        conn.close()
    if n:
        a["schritte_roh"] = roh
    return bool(n)


def _beenden(a: dict, zustand: str) -> None:
    for s in a["schritte"]:
        if s["zustand"] == "offen":
            s["zustand"] = "ausgelassen"
    a["zustand"] = zustand
    _speichern(a, zustand=zustand, beendet_am=_iso())
    log.info("Auftrag %s: %s", a["id"], zustand)


def _iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def durchlauf() -> int:
    """Ein Rundgang (im eigenen Faden): jeden laufenden Auftrag einen Schritt weiter bringen. Reihum nach dem letzten Blick.
    Rueckgabe: Zahl der angesehenen Auftraege."""
    _aufraeumen()
    conn = get_db()
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM agent_auftraege WHERE zustand = 'laeuft' "
                                          "ORDER BY COALESCE(geprueft_am, ''), id LIMIT ?", (JE_RUNDGANG,)).fetchall()]
    finally:
        conn.close()
    for aid in ids:
        try:
            weiter(aid)
            _fehlversuch(aid, zuruecksetzen=True)
        except Exception:  # noqa: BLE001 — ein Auftrag darf den Rundgang der anderen nie aufhalten
            log.exception("Auftrag %s: Rundgang fehlgeschlagen", aid)
            _fehlversuch(aid)
    return len(ids)


_AUFGERAEUMT_AM = 0.0


def _aufraeumen() -> None:
    """Beendete Auftraege nach AUFBEWAHREN_TAGE loeschen (Pruefung N9: die Schritte nennen Dokument- und Dateinamen); hoechstens
    einmal je AUFRAEUMEN_S. Die Rueckmeldung braucht sie nur bis zum Melden — eine noch wartende behaelt ihren Auftrag."""
    global _AUFGERAEUMT_AM
    if _jetzt() - _AUFGERAEUMT_AM < AUFRAEUMEN_S:
        return
    _AUFGERAEUMT_AM = _jetzt()
    conn = get_db()
    try:
        n = conn.execute("DELETE FROM agent_auftraege WHERE zustand <> 'laeuft' AND beendet_am < datetime('now', ?) AND id NOT IN "
                         "(SELECT auftrag_id FROM agent_rueckmeldungen WHERE auftrag_id IS NOT NULL AND zustand IN ('wartet', 'meldet'))",
                         (f"-{AUFBEWAHREN_TAGE} days",)).rowcount
        conn.commit()
    finally:
        conn.close()
    if n:
        log.info("Auftraege: %d alte geloescht (aelter als %d Tage)", n, AUFBEWAHREN_TAGE)


def _fehlversuch(aid: int, zuruecksetzen: bool = False) -> None:
    """Unerwartete Fehler eines Auftrags zaehlen (zuruecksetzen: der Rundgang lief sauber). Nach FEHLVERSUCHE hintereinander
    gilt er als gescheitert — sonst stuende er ewig als laufend da, und seine Rueckmeldung kaeme nie."""
    n = None
    conn = get_db()
    try:
        if zuruecksetzen:
            conn.execute("UPDATE agent_auftraege SET fehlversuche = 0 WHERE id = ? AND fehlversuche > 0", (aid,))
        else:
            conn.execute("UPDATE agent_auftraege SET fehlversuche = fehlversuche + 1, geprueft_am = datetime('now') WHERE id = ?",
                         (aid,))
            n = conn.execute("SELECT fehlversuche FROM agent_auftraege WHERE id = ?", (aid,)).fetchone()
        conn.commit()
    finally:
        conn.close()
    if n and n[0] >= FEHLVERSUCHE:
        a = _laden(aid)
        if a and a["zustand"] == "laeuft":
            laufend = next((s for s in a["schritte"] if s["zustand"] in ("offen", "laeuft")), None)
            if laufend:
                laufend.update(zustand="fehler", grund=T_INTERN)
            _beenden(a, "gescheitert")


def weiter(aid: int) -> None:
    """Den Auftrag so weit bringen, wie es jetzt geht: einen laufenden Schritt auswerten, den naechsten starten."""
    for _runde in range(UEBERGAENGE_JE_RUNDGANG):
        a = _laden(aid)
        if not a or a["zustand"] != "laeuft":
            return
        if not _eigenes_projekt(a["user_id"], a["project_id"]):
            laufend = next((s for s in a["schritte"] if s["zustand"] in ("offen", "laeuft")), None)
            if laufend:
                laufend.update(zustand="fehler", grund=T_PROJEKT_WEG)
            _beenden(a, "gescheitert")
            return
        s = next((s for s in a["schritte"] if s["zustand"] in ("offen", "laeuft")), None)
        if s is None:
            _beenden(a, "fertig")
            return
        if s["zustand"] == "laeuft":
            if not _zu_ende(a, s):
                _speichern(a)
                return
            _speichern(a)
            if s["zustand"] == "fehler":
                _beenden(a, "gescheitert")
                return
            if s.get("vom_nutzer") or a["abbrechen"]:
                _beenden(a, "abgebrochen")
                return
            continue
        if a["abbrechen"]:
            _beenden(a, "abgebrochen")
            return
        if _belegt(a, s):
            _speichern(a)
            return
        if s["zustand"] != "fehler":
            # Erst festhalten, dass er startet (gestartet=False): bricht danach etwas ab (Datenbank, Absturz), startet ihn der
            # naechste Rundgang nicht noch einmal — _zu_ende sagt dann ehrlich „unterbrochen“, statt doppelt zu buchen. Bedingt
            # geschrieben: hat ein anderer Abarbeiter den Stand schon geaendert, startet dieser hier nichts (Pruefung N1).
            s.update(zustand="laeuft", gestartet=False, gestartet_am=_jetzt())
            if not _beanspruchen(a):
                log.warning("Auftrag %s: Schritt %s schon von einem anderen Abarbeiter beansprucht", a["id"], s["art"])
                return
            try:
                s.update(_starten(a, s))
            except Exception:  # noqa: BLE001 — ein unerwarteter Fehler beim Start haelt den Auftrag an, statt ihn zu wiederholen
                log.exception("Auftrag %s: Start von %s fehlgeschlagen", a["id"], s["art"])
                s.update(zustand="fehler", grund=T_INTERN, preis=0)
            a["verbraucht"] = int(a["verbraucht"]) + int(s.get("preis") or 0)
        _speichern(a)
        if s["zustand"] == "fehler":
            _beenden(a, "gescheitert")
            return
        if s["zustand"] in ("laeuft", "offen"):   # offen: Herunterladen gerade belegt oder gedrosselt — spaeter noch einmal
            return


def _belegt(a: dict, s: dict) -> bool:
    """Laeuft im Projekt gerade etwas anderes (auch aus der Oberflaeche gestartet)? Dann wartet der Schritt — hoechstens
    WARTEN_HOECHSTENS_S, danach gilt er als gescheitert (zustand fehler, Rueckgabe False)."""
    from . import rueckmeldung
    jetzt = _jetzt()
    if not (rueckmeldung.was_laeuft(a["project_id"]) or _export_belegt(a["user_id"]) or float(s.get("pause_bis") or 0) > jetzt
            or _rueckmeldung_wartet(a)):
        s.pop("wartet_seit", None)
        s.pop("pause_bis", None)
        return False
    s.setdefault("wartet_seit", jetzt)
    if jetzt - float(s["wartet_seit"]) > WARTEN_HOECHSTENS_S:
        s.update(zustand="fehler", grund=T_BELEGT)
        return False
    return True


def _export_belegt(user_id: int) -> bool:
    """Baut gerade ein Herunterladen fuer dieses Konto (main._export_belegen, je Nutzer EINES)? Dann wartet „PDF herunterladen“,
    statt sofort zu scheitern — am Ende eines Auftrags, nachdem alles davor bezahlt ist (Pruefung N3)."""
    m = sys.modules.get("main")
    return bool(m is not None and int(user_id) in getattr(m, "_export_nutzer", ()))


def _rueckmeldung_wartet(a: dict) -> bool:
    """Wartet im Projekt noch eine Rueckmeldung zu einem Lauf ausserhalb des Auftrags (Baustein 1)? Dann wartet der Schritt, bis
    sie gemeldet ist — sonst saehe sie den Lauf des Auftrags als ihren und meldete dessen Zahlen ein zweites Mal (Pruefung N6)."""
    conn = get_db()
    try:
        return bool(conn.execute("SELECT 1 FROM agent_rueckmeldungen WHERE user_id = ? AND project_id = ? AND auftrag_id IS NULL "
                                 "AND zustand IN ('wartet', 'meldet') LIMIT 1", (a["user_id"], a["project_id"])).fetchone())
    finally:
        conn.close()


def _starten(a: dict, s: dict) -> dict:
    """Den Schritt ueber sein Werkzeug starten — mit bestaetigt=true, freigegeben durch den Auftrag (nur seine Art, hoechstens
    die Rest-Summe). Rueckgabe: neue Felder fuer den Schritt."""
    from .tools import ausgaben
    p = _eigenes_projekt(a["user_id"], a["project_id"])
    if not p:
        return {"zustand": "fehler", "grund": T_PROJEKT_WEG}
    doc = _dokument(s["document_id"], p["id"]) if s.get("document_id") else None
    if s.get("document_id") and not doc:
        return {"zustand": "fehler", "grund": T_DOKUMENT_WEG}
    if s["art"] == "tagging" and _modul("main")._dokument_getaggt(doc):
        # Inzwischen getaggt (etwa per Knopf, waehrend der Auftrag ein anderes Dokument taggte): nicht noch einmal taggen und
        # bezahlen — „Neu taggen“ ersetzte sonst die gerade entstandenen Tags (Pruefung N2)
        return {"zustand": "uebersprungen", "grund": T_SCHON_GETAGGT, "preis": 0}
    namen, executor = _werkzeugsatz(p, a["user_id"])
    if s["werkzeug"] not in namen:
        return {"zustand": "fehler", "grund": T_NICHT_DA}
    rest = max(0, int(a["freigegeben"]) - int(a["verbraucht"]))
    with ausgaben.auftrag_schritt({s["freigabe"]}, rest) as f:
        r = executor.execute(s["werkzeug"], dict(s.get("args") or {}, bestaetigt=True))
    preis = int(f["preis"] or 0)
    felder: dict[str, Any] = {"gestartet_am": _jetzt(), "preis": preis}
    res = r.get("result") if isinstance(r, dict) else None
    if not (isinstance(r, dict) and r.get("ok")):
        if _herunterladen_belegt(r):   # nichts gebaut, nichts gebucht: spaeter noch einmal (wartet hoechstens WARTEN_HOECHSTENS_S)
            return {"zustand": "offen", "gestartet": False, "preis": 0, "pause_bis": _jetzt() + PAUSE_DROSSEL_S,
                    "wartet_seit": s.get("wartet_seit") or _jetzt()}
        return dict(felder, zustand="fehler", preis=0, grund=str((r or {}).get("error") or T_INTERN))
    res = res if isinstance(res, dict) else {}
    if res.get("rueckfrage_noetig"):
        if f["grund"] == "summe":
            grund = T_SUMME.format(n=f.get("preis_noetig"), rest=rest)
        elif res.get("erlaubt") is False or res.get("grund") == "guthaben":
            grund = T_GUTHABEN
        else:
            grund = str(res.get("hinweis") or T_INTERN)
        return dict(felder, zustand="fehler", preis=0, grund=grund)
    if s["hintergrund"]:
        if res.get("gestartet") is True:
            log.info("Auftrag %s: %s gestartet (%d Credits freigegeben)", a["id"], s["art"], preis)
            # Sammellaeufe setzen ihren Zustand schon beim Start: dann gilt die Anlaufzeit nicht, und ein schneller Lauf wird
            # beim naechsten Rundgang ausgewertet statt erst nach ANLAUF_S
            return dict(felder, zustand="laeuft", gestartet=True, gesehen=_laeuft_schon(s, a["project_id"]))
        if res.get("gestartet") is False and not res.get("gesperrt"):   # nichts zu tun (inzwischen alles erledigt)
            return dict(felder, zustand="uebersprungen", preis=0, grund=T_NICHTS)
        return dict(felder, zustand="fehler", preis=0, grund=T_NICHT_DA if res.get("gesperrt") else T_INTERN)
    anhang = r.get("anhang") if isinstance(r.get("anhang"), dict) else None
    log.info("Auftrag %s: %s fertig (%d Credits freigegeben)", a["id"], s["art"], preis)
    return dict(felder, zustand="fertig", anhang=anhang, beendet_am=_jetzt())


def _herunterladen_belegt(r: Any) -> bool:
    """Scheiterte ein Herunterladen nur, weil fuer das Konto gerade eines gebaut wird oder es gedrosselt ist (main._export_belegt_text)?"""
    m = sys.modules.get("main")
    if m is None or not isinstance(r, dict):
        return False
    return str(r.get("error") or "") in {m._export_belegt_text("laeuft"), m._export_belegt_text("drossel")}


def _laeuft_schon(s: dict, project_id: int) -> bool:
    from . import rueckmeldung
    try:
        return rueckmeldung.laeuft(s["hintergrund"], project_id)
    except Exception:  # noqa: BLE001 — Zustand nicht lesbar: dann eben mit Anlaufzeit
        log.exception("Auftrag: Zustand nach dem Start nicht lesbar")
        return False


def _dokument(document_id: int, project_id: int) -> Optional[dict]:
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM documents WHERE id = ? AND project_id = ?", (int(document_id), int(project_id))).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def _zu_ende(a: dict, s: dict) -> bool:
    """Ist der laufende Schritt zu Ende? Dann ist er ausgewertet (zustand, grund, anhang). Dieselben Zustaende wie die
    Rueckmeldung (rueckmeldung.laeuft); Anlaufzeit wie dort; laenger als SCHRITT_HOECHSTENS_S = gescheitert."""
    from . import rueckmeldung
    if not s.get("gestartet"):   # der Start wurde unterbrochen (siehe weiter): ob er geschah, ist nicht sicher
        s.update(zustand="fehler", grund=T_UNTERBROCHEN, beendet_am=_jetzt())
        return True
    alter = _jetzt() - float(s.get("gestartet_am") or _jetzt())
    if rueckmeldung.laeuft(s["hintergrund"], a["project_id"]):
        s["gesehen"] = True
        if alter > SCHRITT_HOECHSTENS_S:
            s.update(zustand="fehler", grund=T_ZU_LANG, beendet_am=_jetzt())
            return True
        return False
    if not s.get("gesehen") and alter < ANLAUF_S:
        return False
    s.update(_auswerten(a, s))
    s["beendet_am"] = _jetzt()
    return True


def _hinweis_des_laufs(project_id: int) -> dict:
    conn = get_db()
    try:
        r = conn.execute("SELECT lauf_hinweis FROM projects WHERE id = ?", (project_id,)).fetchone()
    finally:
        conn.close()
    try:
        h = json.loads(r["lauf_hinweis"]) if r and r["lauf_hinweis"] else {}
        return h if isinstance(h, dict) else {}
    except (TypeError, ValueError):
        return {}


def _auswerten(a: dict, s: dict) -> dict:
    """Ergebnis eines Schritts, der im Hintergrund lief: fertig, teilweise (vom_nutzer: in der Oberflaeche abgebrochen) oder
    fehler — aus denselben Daten, die Oberflaeche und Werkzeuge lesen."""
    from . import rueckmeldung
    pid, art = a["project_id"], s["art"]
    if art in ("tagging", "pruefung"):
        t = _modul("tagging_api")
        d = _dokument(s["document_id"], pid)
        if not d:
            return {"zustand": "fehler", "grund": T_DOKUMENT_WEG}
        if art == "tagging":
            if d.get("tagging_status") == t.STATUS_FERTIG and d.get("getaggt") in (1, True):
                return {"zustand": "fertig"}
            return {"zustand": "fehler", "grund": str(t._bericht(d).get("fehler") or "Das Tagging ist fehlgeschlagen.")}
        if d.get("pruefung_status") == t.STATUS_FERTIG:
            return {"zustand": "fertig"}
        return {"zustand": "fehler", "grund": str(t._pruef_bericht(d).get("fehler") or "Die Prüfung ist fehlgeschlagen.")}
    if art == "alt_texte":
        # Neu ist ein Text, wenn das Bild fertig ist und keinen Fehlergrund traegt: der Lauf leert fehler_grund bei Erfolg und
        # setzt ihn bei einem Fehlversuch — auch wenn ein alter Text bleibt (Rettung, Status 'done' mit needs_review). Vorher
        # zaehlte jedes fertige Bild; fiel die KI fuer alle aus, hiess der Schritt „fertig“, und der Auftrag lief weiter zum
        # bezahlten Herunterladen (Pruefung M3). Vom Autor als Schmuck gekennzeichnete Bilder laufen nie mit.
        # Mit document_id (Auftrag fuer ein Dokument) zaehlen nur dessen Bilder — offene Bilder anderer Dokumente sind nicht Teil
        # des Schritts und machten ihn sonst „teilweise“
        nur_dok = s.get("document_id")
        conn = get_db()
        try:
            z = dict(conn.execute(
                "SELECT COUNT(*) AS n, SUM(CASE WHEN status = 'done' AND COALESCE(fehler_grund, '') = '' THEN 1 ELSE 0 END) AS neu, "
                "SUM(CASE WHEN status = 'error' OR COALESCE(fehler_grund, '') <> '' THEN 1 ELSE 0 END) AS kaputt, "
                "SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS offen FROM images WHERE project_id = ? "
                "AND NOT (COALESCE(original_alt, '') = 'dekorativ' AND COALESCE(image_type, '') = 'dekorativ')"
                + (" AND document_id = ?" if nur_dok else ""), (pid, *([nur_dok] if nur_dok else []))).fetchone())
        finally:
            conn.close()
        h = _hinweis_des_laufs(pid)
        if h.get("grund") == "abbruch":
            return {"zustand": "teilweise", "vom_nutzer": True, "grund": T_VOM_NUTZER}
        grund = {"credits": T_GUTHABEN, "tageslimit": "Das Tageslimit für KI-Aufrufe war erreicht."}.get(h.get("grund"), "")
        if z["n"] and not z["neu"]:
            return {"zustand": "fehler", "grund": grund or T_KEIN_ALT_TEXT}
        if grund or z["kaputt"] or z["offen"]:
            return {"zustand": "teilweise", "grund": grund or "Nicht für alle Bilder ist ein Alt-Text entstanden."}
        return {"zustand": "fertig"}
    if art == "quickinfos":
        st = _modul("formular_api")._generierung.get(pid) or {}
        if st.get("abbruch"):
            return {"zustand": "teilweise", "vom_nutzer": True, "grund": T_VOM_NUTZER}
        if st.get("fehler"):
            # keine neue Quickinfo = gescheitert (haelt an), sonst teilweise (Pruefung M3)
            return {"zustand": "teilweise" if st.get("felder_neu") else "fehler", "grund": str(st["fehler"][0])}
        if st and not st.get("felder_neu"):
            return {"zustand": "fehler", "grund": T_KEINE_QUICKINFO}
        return {"zustand": "fertig"}
    if art == "uebersetzung":
        st = _modul("uebersetzung_api")._lauf.get(pid) or {}
        if st.get("abbruch"):
            return {"zustand": "teilweise", "vom_nutzer": True, "grund": T_VOM_NUTZER}
        if st.get("fehler") and not st.get("segmente_fertig"):
            return {"zustand": "fehler", "grund": str(st["fehler"][0])}
        ergebnis = {"zustand": "teilweise", "grund": str(st["fehler"][0])} if st.get("fehler") else {"zustand": "fertig"}
        # Die uebersetzte Datei gehoert zum Schritt (kostenlos, derselbe Weg wie exportiere_uebersetzung im Gespraech)
        p = _eigenes_projekt(a["user_id"], pid)
        namen, executor = _werkzeugsatz(p, a["user_id"]) if p else (set(), None)
        if executor is not None and "exportiere_uebersetzung" in namen:
            # im Faden des Schritts (ohne Freigabe — die Datei ist kostenlos): so fasst das Werkzeug keine Angebote und Karten des
            # Gespraechs an (Pruefung M1, tools/ausgaben.nachbereiten)
            from .tools import ausgaben
            with ausgaben.auftrag_schritt((), 0):
                r = executor.execute("exportiere_uebersetzung", {})
            if isinstance(r, dict) and r.get("ok") and isinstance(r.get("anhang"), dict):
                ergebnis["anhang"] = r["anhang"]
            else:
                ergebnis.update(zustand="teilweise", grund="Die übersetzte Datei ließ sich nicht erstellen.")
        return ergebnis
    return {"zustand": "fertig"}


def laeuft(auftrag_id: Optional[int]) -> bool:
    """Fuer die Rueckmeldung: laeuft der Auftrag noch? Ein unbekannter Auftrag laeuft nicht."""
    if not auftrag_id:
        return False
    conn = get_db()
    try:
        r = conn.execute("SELECT zustand FROM agent_auftraege WHERE id = ?", (int(auftrag_id),)).fetchone()
    finally:
        conn.close()
    return bool(r and r["zustand"] == "laeuft")


def laufende(user_id: int) -> list[dict]:
    """Laufende Auftraege des Kontos fuer konto.laufende_arbeit: Projekt und der Schritt, der dran ist."""
    conn = get_db()
    try:
        rows = [_zeile(dict(r)) for r in conn.execute(
            "SELECT a.*, COALESCE(NULLIF(TRIM(p.name), ''), p.filename, '') AS projekt FROM agent_auftraege a "
            "JOIN projects p ON p.id = a.project_id WHERE a.user_id = ? AND a.zustand = 'laeuft' ORDER BY a.id", (int(user_id),))]
    finally:
        conn.close()
    out = []
    for a in rows:
        dran = next((i for i, s in enumerate(a["schritte"], 1) if s["zustand"] in ("offen", "laeuft")), len(a["schritte"]))
        titel = a["schritte"][dran - 1]["titel"] if a["schritte"] else ""
        # „was“ deutsch fuer das Modell; „auftrag“ fuer die Oberflaeche, die den Satz in ihrer Sprache baut (Pruefung N8)
        out.append({"project_id": a["project_id"], "projekt": a["projekt"],
                    "was": f"Auftrag, Schritt {dran} von {len(a['schritte'])}: {titel}",
                    "auftrag": {"schritt": dran, "von": len(a["schritte"]), "titel": titel}})
    return out


# ─── 4. Fuer die Rueckmeldung ────────────────────────────────────────────────────────────────────────────────────────────────
def fakten(auftrag_id: int) -> dict:
    """Was der Server ueber den Auftrag weiss — fuer die persoenliche Nachricht am Ende (rueckmeldung.fakten)."""
    a = _laden(auftrag_id)
    if not a:
        return {"auftrag": {"fehler": "nicht mehr lesbar"}}
    schritte = []
    for s in a["schritte"]:
        e = {"schritt": s["titel"], "umfang": s.get("umfang") or "", "ergebnis": SCHRITT_ZUSTAND.get(s["zustand"], s["zustand"])}
        if s.get("grund"):
            e["grund"] = s["grund"]
        if isinstance(s.get("anhang"), dict) and s["anhang"].get("dateiname"):
            e["datei"] = s["anhang"]["dateiname"]
            if s["anhang"].get("gueltig_bis"):   # Sofort-Downloads gelten 24 Stunden; die Ablage bleibt (Pruefung N10)
                e["datei_herunterladbar_bis"] = s["anhang"]["gueltig_bis"]
        schritte.append(e)
    return {"auftrag": {"ergebnis": AUFTRAG_ZUSTAND.get(a["zustand"], a["zustand"]), "schritte": schritte,
                        "credits_freigegeben": a["freigegeben"]}}


def schlichte_meldung(auftrag_id: int, projekt: str, _) -> str:
    """Die Meldung ohne Modell (Modell ausgefallen, Kostendeckel): Ende des Auftrags und je Schritt sein Ergebnis in der Sprache
    der Oberflaeche — vorher hiess es auch bei einem angehaltenen Auftrag nur „ist zu Ende“ (Pruefung N5)."""
    from . import rueckmeldung
    a = _laden(auftrag_id)
    kopf = _(rueckmeldung.T_FERTIG).format(was=_(rueckmeldung.WAS["auftrag"]), projekt=projekt)
    if not a:
        return kopf
    return kopf + " " + " ".join(_("{schritt}: {ergebnis}.").format(schritt=_(s["titel"]),
                                                                    ergebnis=_(SCHRITT_ZUSTAND.get(s["zustand"], s["zustand"])))
                                 for s in a["schritte"])


def aufgeben(auftrag_id: Optional[int], grund: str = T_AUFGEGEBEN) -> bool:
    """Einen Auftrag von aussen beenden (die Rueckmeldung nach zwei Tagen, rueckmeldung.AUFTRAG_HOECHSTENS_S): ein Abarbeiter,
    der in einem Werkzeug haengt, haelt sonst den Auftrag, sein Projekt und die Rueckmeldung fuer immer fest (Pruefung M4).
    True = er lief noch und ist jetzt beendet."""
    a = _laden(auftrag_id) if auftrag_id else None
    if not a or a["zustand"] != "laeuft":
        return False
    laufend = next((s for s in a["schritte"] if s["zustand"] in ("offen", "laeuft")), None)
    if laufend:
        laufend.update(zustand="fehler", grund=grund, beendet_am=_jetzt())
    _beenden(a, "gescheitert")
    return True


def ausgeschaltet() -> int:
    """Beim Start ohne Schalter (main.lifespan): laufende Auftraege werden nicht mehr abgearbeitet — beenden, damit Projekt und
    Rueckmeldung nicht fuer immer warten (Pruefung M4). Rueckgabe: Zahl der beendeten."""
    conn = get_db()
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM agent_auftraege WHERE zustand = 'laeuft'").fetchall()]
    finally:
        conn.close()
    return sum(1 for aid in ids if aufgeben(aid, T_AUS))


def arten(auftrag_id: int) -> set:
    """Welche Arbeiten der Auftrag enthielt, die gelaufen sind (fuer die Zahlen in rueckmeldung.fakten)."""
    a = _laden(auftrag_id)
    return {s["art"] for s in (a or {}).get("schritte", []) if s["zustand"] in ("fertig", "teilweise", "fehler")}


def anhaenge(auftrag_id: int) -> list[dict]:
    """Die Dateien aus den Schritten (Herunterladen, Ablage) fuer unter die Rueckmeldung — wie unter einer Antwort."""
    a = _laden(auftrag_id)
    return [dict(s["anhang"]) for s in (a or {}).get("schritte", []) if isinstance(s.get("anhang"), dict)]


# ─── Schleife ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Ein eigener Faden fuer die Schritte: Werkzeuge wie Umwandeln und Herunterladen arbeiten Sekunden bis Minuten, und die
# Freigabe des Auftrags (ausgaben.auftrag_schritt) gilt nur in DIESEM Faden — kein Gespraech und kein Knopf sieht sie.
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inkluagent-auftrag")
_AUFGABE: Optional[asyncio.Task] = None   # starke Referenz (asyncio sammelt Aufgaben ohne Referenz ein)
_LOOP: Optional[asyncio.AbstractEventLoop] = None
_WECKER: Optional[asyncio.Event] = None


def _wecken() -> None:
    """Nach dem Erteilen oder Abbrechen nicht bis zum naechsten Rundgang warten (aus jedem Faden)."""
    if _LOOP is not None and _WECKER is not None:
        try:
            _LOOP.call_soon_threadsafe(_WECKER.set)
        except RuntimeError:   # Schleife schon beendet
            pass


async def schleife() -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            await loop.run_in_executor(_POOL, durchlauf)
        except Exception:  # noqa: BLE001 — die Schleife darf nie enden
            log.exception("Auftraege: Rundgang fehlgeschlagen")
        try:
            await asyncio.wait_for(_WECKER.wait(), timeout=PRUEF_S)
        except asyncio.TimeoutError:
            pass
        _WECKER.clear()


def starten() -> None:
    """Aus main.lifespan (ein uvicorn-Worker): nach einem Neustart aufraeumen, dann die Schleife — einmal je Prozess."""
    global _AUFGABE, _LOOP, _WECKER
    try:
        nach_neustart()
    except Exception:  # noqa: BLE001 — etwa Datenbank gesperrt: die Schleife startet trotzdem (Pruefung M4)
        log.exception("Auftraege: Aufraeumen nach dem Neustart fehlgeschlagen")
    _LOOP = asyncio.get_running_loop()
    if _WECKER is None:
        _WECKER = asyncio.Event()
    if _AUFGABE is None or _AUFGABE.done():
        _AUFGABE = _LOOP.create_task(schleife())


def nach_neustart() -> None:
    """Die Laeufe der Schritte leben im Prozess: lief beim Neustart ein Schritt, ist er abgebrochen. Er gilt als gescheitert,
    der Auftrag endet (die Rueckmeldung sagt, was fertig ist). Ein Auftrag zwischen zwei Schritten laeuft weiter."""
    conn = get_db()
    try:
        rows = [_zeile(dict(r)) for r in conn.execute("SELECT * FROM agent_auftraege WHERE zustand = 'laeuft'").fetchall()]
    finally:
        conn.close()
    for a in rows:
        s = next((s for s in a["schritte"] if s["zustand"] == "laeuft"), None)
        if s:
            s.update(zustand="fehler", grund=T_NEUSTART, beendet_am=_jetzt())
            _beenden(a, "gescheitert")
