"""Werkzeuge „Meine Ablage" fuer den Chatbot (Schritt 2, 11.09.2026, Steve + Fable 5).

Der InkluAgent kann ein Word-Projekt zu Ende bringen: Pruefbericht und Hoerprobe lesen,
in eine barrierefreie PDF umwandeln (Umwandler + veraPDF), die Word-Datei mit Alt-Texten
ausgeben, das Regal lesen. Alles laeuft ueber DIESELBEN Kernfunktionen wie der
Export-Bereich (main._pdfua_umwandeln_sync, _pdfua_vorschau_sync, _word_export_ausgabe_sync)
— ein Weg, zwei Bediener. Jede Umwandlung landet als Eintrag unter „Meine Ablage"
(ausloeser 'bot'); die Oberflaeche zeigt unter der Antwort den Download-Knopf (Anhang).

Drei Regeln, die der SERVER durchsetzt (nicht nur der Prompt):
- Kostenpflichtige Werkzeuge verlangen bestaetigt=true. Der erste Aufruf liefert nur
  Preis und Guthaben (rueckfrage_noetig) — der Bot muss den Nutzer fragen.
- Zustimmung in zwei Schritten (Review 12.09.2026, Fable 5): bestaetigt=true gilt NUR, wenn
  derselbe Server vorher ein Angebot (Nutzer, Projekt, Art, Dokument, Preis) abgelegt hat,
  dieses Angebot aus einer FRUEHEREN Nutzer-Nachricht stammt (anderer Turn), hoechstens
  15 Minuten alt ist und der Preis unveraendert ist. Je Nutzer-Nachricht hoechstens EINE
  kostenpflichtige Aktion. Grund: bestaetigt kommt als Modell-Argument — eine Anweisung im
  Dokumenttext (Prompt-Injection ueber Hoerprobe/Struktur) darf keine Credits ausgeben koennen.
- Projekt- und Nutzerkontext kommen aus der Sitzung (ToolExecutor), nie aus Modell-Argumenten.

main wird zur Laufzeit importiert (sys.modules), weil main die Agenten-Module selbst erst
in den Endpunkten laedt — ein Import beim Modulstart waere zirkulaer.
"""
from __future__ import annotations

import importlib
import logging
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

from fastapi import HTTPException

from ..daten import daten, daten_zeilen, text_kennzeichnen   # Fremdtext ist keine Anweisung (09.10.2026)

log = logging.getLogger(__name__)


def _db_path() -> str:
    # Gleicher Pfad wie database.DB_PATH (INKLUDOCS_DB), kein fester /app-Pfad (Review 12.09.2026).
    try:
        return importlib.import_module("database").DB_PATH
    except Exception:  # noqa: BLE001
        return "/app/data/inkludocs.db"


# Angebote fuer kostenpflichtige Werkzeuge: Schluessel (user_id, project_id, art, document_id) ->
# {"preis", "turn", "zeit"}. Im Prozess gehalten — die Zustimmung muss ohnehin binnen Minuten kommen.
_ANGEBOTE: dict[tuple, dict] = {}
_ANGEBOT_GUELTIG_S = 15 * 60
_KOSTENPFLICHTIG_JE_TURN = 1
# Zustimmung an die konkrete Aktion gebunden (Pruefung 3, 30.09.2026, Entwicklung N1): jedes Angebot hat eine Kennung; der
# Server merkt sich je Nutzer und Projekt das ZULETZT gemachte Angebot. Ein getipptes „Ja“ (bestaetigt=true vom Modell) loest
# nur dieses aus; die Oberflaeche zeigt den Angebotstext des SERVERS als Karte mit eigenem Bestaetigungsknopf, der genau das
# gespeicherte Angebot (Werkzeug, Argumente, Preis) ausfuehrt — nicht das, was das Modell dem Nutzer vielleicht schildert.
_LETZTES: dict[tuple, str] = {}          # (user_id, project_id) -> Angebots-Kennung
_NACH_ID: dict[str, tuple] = {}          # Angebots-Kennung -> Schluessel in _ANGEBOTE
# Pruefung 4 (30.09.2026, Entwicklung 2/3, Barrierefreiheit M3): Einloesen unter einer Sperre (genau ein Verbrauch), der Knopf
# der Karte belegt sein Angebot vorher (ein Doppelklick bekommt „Schon bestätigt.“ und schreibt nichts in den Verlauf), und
# ausgefuehrte Angebote bleiben bekannt, damit ihre Karte „Erledigt“ zeigt statt „gilt nicht mehr“.
_SPERRE = threading.RLock()
_ERLEDIGT: dict[str, tuple] = {}         # Kennung -> (user_id, project_id, zeit, Kartenfelder)
_ERLEDIGT_BEHALTEN_S = 24 * 3600
_VERBRAUCHT: dict[tuple, list] = {}      # (user_id, project_id) -> [(Kennung, Angebot)] im laufenden Aufruf verbraucht
_KARTE = threading.local()               # Kennung, deren Knopf gerade in diesem Faden ausfuehrt

# Gruende einer abgelehnten Zustimmung: Text fuer das Modell (Anweisung) und Kennwort fuer die Karte (eigener Satz fuer Menschen)
_G_KEIN = ("Es liegt kein gueltiges Angebot vor: erst OHNE bestaetigt aufrufen, dem Nutzer Preis und Guthaben "
           "nennen und auf sein Ja warten.")
_G_LETZTES = ("Das ist nicht das zuletzt genannte Angebot. Ein Ja gilt nur für das letzte Angebot: frage für diese Aktion "
              "neu (ohne bestaetigt) oder lass den Nutzer die Karte unter deiner Antwort bestätigen.")
_G_TURN = ("Die Zustimmung muss vom Nutzer in einer eigenen, spaeteren Nachricht kommen — nicht in derselben "
           "Nachricht wie die Preisauskunft. Nenne den Preis und warte auf sein Ja.")
_G_PREIS = "Der Preis hat sich seit der Auskunft geaendert. Nenne dem Nutzer den neuen Preis und frage erneut."
_GRUND_KENNWORT = {_G_KEIN: "angebot", _G_LETZTES: "letztes", _G_TURN: "nachricht", _G_PREIS: "preis"}
# Ja-Pruefung auf dem Server (Ausbau Runde 1, Schritt 2, Schalter funktionen.AGENT_SICHERHEIT): ein getipptes Ja zaehlt nur,
# wenn die Nachricht des Nutzers ein kurzes, eindeutiges Ja ist; der Knopf der Karte gilt immer.
from ..sicherheit import G_JA as _G_JA, ja_grund as _ja_grund  # noqa: E402
_GRUND_KENNWORT[_G_JA] = "ja"


def _angebot_merken(schluessel: tuple, preis: int, turn_id: str) -> str:
    with _SPERRE:
        return _angebot_merken_ungesperrt(schluessel, preis, turn_id)


def _angebot_merken_ungesperrt(schluessel: tuple, preis: int, turn_id: str) -> str:
    jetzt = time.time()
    for k in [k for k, e in _ERLEDIGT.items() if jetzt - e[2] > _ERLEDIGT_BEHALTEN_S]:
        _ERLEDIGT.pop(k, None)
    for k in [k for k, a in _ANGEBOTE.items() if jetzt - a["zeit"] > _ANGEBOT_GUELTIG_S]:
        _NACH_ID.pop(_ANGEBOTE[k].get("id"), None)
        _ANGEBOTE.pop(k, None)
    alt = _ANGEBOTE.get(schluessel)
    if alt:
        _NACH_ID.pop(alt.get("id"), None)
        if alt.get("werkzeug"):   # hatte eine Karte: die ist jetzt veraltet
            _VERBRAUCHT.setdefault((schluessel[0], schluessel[1]), []).append((alt.get("id"), alt, None))
    angebot_id = uuid.uuid4().hex
    _ANGEBOTE[schluessel] = {"preis": int(preis or 0), "turn": turn_id, "zeit": jetzt, "id": angebot_id}
    _NACH_ID[angebot_id] = schluessel
    _LETZTES[(schluessel[0], schluessel[1])] = angebot_id
    return angebot_id


def angebot_nach_id(angebot_id: str) -> Optional[tuple]:
    """(Schluessel, Angebot) zu einer Kennung, solange es gilt."""
    k = _NACH_ID.get(angebot_id or "")
    a = _ANGEBOTE.get(k) if k else None
    if not a or time.time() - a["zeit"] > _ANGEBOT_GUELTIG_S:
        return None
    return k, a


def _angebot_einloesen(schluessel: tuple, preis: int, turn_id: str) -> Optional[str]:
    """None = Zustimmung gueltig (Angebot wird verbraucht). Sonst der Grund, warum nicht. Unter der Sperre: Pruefen und
    Verbrauchen sind ein Schritt, zwei gleichzeitige Zustimmungen fuehren hoechstens eine aus (Pruefung 4, Entwicklung 3)."""
    with _SPERRE:
        a = _ANGEBOTE.get(schluessel)
        if not a or time.time() - a["zeit"] > _ANGEBOT_GUELTIG_S:
            _ANGEBOTE.pop(schluessel, None)
            return _G_KEIN
        if a.get("in_arbeit") and getattr(_KARTE, "id", None) != a.get("id"):
            return _G_KEIN          # der Knopf der Karte fuehrt dieses Angebot gerade aus
        if _LETZTES.get((schluessel[0], schluessel[1])) != a.get("id"):
            return _G_LETZTES
        if a["turn"] == turn_id:
            return _G_TURN
        if getattr(_KARTE, "id", None) != a.get("id"):
            g = _ja_grund(turn_id)          # None, solange der Schalter aus ist
            if g:
                return g
        if a["preis"] != int(preis or 0):
            _ANGEBOTE.pop(schluessel, None)
            _NACH_ID.pop(a.get("id"), None)
            _VERBRAUCHT.setdefault((schluessel[0], schluessel[1]), []).append((a.get("id"), a, False))
            return _G_PREIS
        _ANGEBOTE.pop(schluessel, None)
        _NACH_ID.pop(a.get("id"), None)
        _VERBRAUCHT.setdefault((schluessel[0], schluessel[1]), []).append((a.get("id"), a, True))
        return None


def angebot_reservieren(angebot_id: str, user_id: int, project_id: int) -> tuple:
    """Knopf der Karte: das Angebot fuer genau EINEN Klick belegen. Rueckgabe (treffer, grund) mit grund '' (belegt),
    'erledigt' (schon ausgefuehrt oder laeuft gerade) oder 'ungueltig' (abgelaufen, unbekannt, fremd)."""
    with _SPERRE:
        e = _ERLEDIGT.get(angebot_id or "")
        if e and e[0] == int(user_id) and e[1] == int(project_id):
            return None, "erledigt"
        t = angebot_nach_id(angebot_id)
        if not t or t[0][0] != int(user_id) or t[0][1] != int(project_id) or not t[1].get("werkzeug"):
            return None, "ungueltig"
        if t[1].get("in_arbeit"):
            return None, "erledigt"
        t[1]["in_arbeit"] = True
        return t, ""


def angebot_loslassen(angebot_id: str) -> None:
    """Nach dem Klick: steht das Angebot noch (nicht verbraucht, z. B. Werkzeug hier nicht verfuegbar), wieder frei geben."""
    with _SPERRE:
        t = angebot_nach_id(angebot_id)
        if t:
            t[1].pop("in_arbeit", None)


def erledigte_karte(angebot_id: str, user_id: int, project_id: int) -> Optional[dict]:
    e = _ERLEDIGT.get(angebot_id or "")
    return e[3] if e and e[0] == int(user_id) and e[1] == int(project_id) else None


def karte_ausfuehren(angebot_id: str, werkzeug: str, args: dict, executor) -> dict:
    """Fuehrt das belegte Angebot aus (Faden des Aufrufers): nur DIESES Angebot darf sein belegtes Angebot einloesen."""
    _KARTE.id = angebot_id
    try:
        return executor.execute(werkzeug, dict(args or {}, bestaetigt=True))
    finally:
        _KARTE.id = None


# Karte unter der Antwort (Pruefung 3): Text und Knopf vom SERVER, nicht vom Modell
_UNUMKEHRBAR = ("ausgabe_loeschen", "dokument_loeschen", "bild_loeschen")


def karte_anhaengen(result: dict, werkzeug: str, args: dict, user_id: int, project_id: int, vorher_id: Optional[str]) -> dict:
    """Hat dieser Werkzeugaufruf ein NEUES Angebot abgelegt, haengt der Server eine Bestaetigungs-Karte an (anhang): Aktion,
    Ziel und Preis aus dem gespeicherten Angebot und dem Werkzeug-Ergebnis, Knopf „<Aktion> bestätigen“. Das Angebot merkt
    sich Werkzeug und Argumente — der Knopf fuehrt genau das aus (main: POST /api/projects/{id}/chat/bestaetigen)."""
    angebot_id = _LETZTES.get((int(user_id), int(project_id)))
    if not angebot_id or angebot_id == vorher_id or not isinstance(result, dict) or not result.get("ok"):
        return result
    treffer = angebot_nach_id(angebot_id)
    if not treffer:
        return result
    _k, a = treffer
    a["werkzeug"] = werkzeug
    a["args"] = {k: v for k, v in (args or {}).items() if k != "bestaetigt"}
    r = result.get("result") or {}
    try:
        from .namen import werkzeug_name
        _ = _main().get_gettext(_ui_lang(user_id))
    except Exception:  # noqa: BLE001
        werkzeug_name, _ = (lambda n, u=None: n), (lambda s: s)
    aktion = werkzeug_name(werkzeug, _)
    ziel = r.get("dokument") or r.get("dateiname") or ""
    preis = int(a.get("preis") or 0)
    beschreibung = _beschreibung(werkzeug, a["args"], aktion, ziel, preis, _)
    if preis:
        text = (_("{aktion}: „{ziel}“ für {p} Credits.").format(aktion=aktion, ziel=ziel, p=preis) if ziel
                else _("{aktion} für {p} Credits.").format(aktion=aktion, p=preis))
    elif werkzeug in _UNUMKEHRBAR:
        text = (_("{aktion}: „{ziel}“. Das lässt sich nicht rückgängig machen.").format(aktion=aktion, ziel=ziel) if ziel
                else _("{aktion}. Das lässt sich nicht rückgängig machen.").format(aktion=aktion))
    else:
        text = (_("{aktion}: „{ziel}“, kostenlos.").format(aktion=aktion, ziel=ziel) if ziel
                else _("{aktion}, kostenlos.").format(aktion=aktion))
    a["text"] = text
    a["beschreibung"] = beschreibung
    a["erledigt_text"] = (_("„{ziel}“ ist gelöscht.").format(ziel=ziel) if (werkzeug in _UNUMKEHRBAR and ziel)
                          else beschreibung + ".")
    a["titel_erledigt"], a["titel_abgelaufen"] = _("Erledigt"), _("Nicht mehr gültig")
    # Knopf nennt, was genau passiert (Pruefung 4, NIEDRIG 2): „Alt-Texte als Excel herunterladen (10 Credits) bestätigen“
    karte = {"art": "bestaetigung", "angebot_id": angebot_id, "titel": _("Bestätigung nötig"), "text": text,
             "knopf": _("{aktion} bestätigen").format(aktion=beschreibung), "zustand": "offen"}
    if not result.get("anhang"):
        result["anhang"] = karte
    return result


_FORMATE = {"csv": "CSV", "xlsx": "Excel", "json": "JSON"}


def _beschreibung(werkzeug: str, args: dict, aktion: str, ziel: str, preis: int, _) -> str:
    """Was der Knopf ausfuehrt, in einem Stueck: Aktion (mit Format), Ziel, Preis."""
    fmt = _FORMATE.get(str((args or {}).get("format") or "").lower())
    if werkzeug == "exportiere_alt_texte" and fmt:
        kern = _("Alt-Texte als {format} herunterladen").format(format=fmt)
    elif werkzeug == "exportiere_quickinfos":
        kern = _("Quickinfos als CSV herunterladen")
    else:
        kern = aktion
    teile = (["„" + ziel + "“"] if ziel else []) + ([_("{p} Credits").format(p=preis)] if preis else [])
    return kern + (" (" + ", ".join(teile) + ")" if teile else "")


def _kartenfelder(a: dict, zustand: str) -> dict:
    if zustand == "erledigt":
        return {"zustand": "erledigt", "titel": a.get("titel_erledigt") or "Erledigt", "text": a.get("erledigt_text") or a.get("text") or ""}
    return {"zustand": "abgelaufen", "titel": a.get("titel_abgelaufen") or "Nicht mehr gültig", "text": a.get("text") or ""}


def _karte_speichern(project_id: int, angebot_id: str, felder: dict) -> None:
    """Zustand der Karte im gespeicherten Verlauf festhalten (ueberlebt Neuladen und Neustart)."""
    try:
        from inkluagent import storage
        storage.karte_aktualisieren(int(project_id), angebot_id, felder)
    except Exception:  # noqa: BLE001 — ohne Datenbank (Unit-Tests) bleibt es beim Zustand im Speicher
        log.debug("Karte %s: Zustand nicht gespeichert", angebot_id, exc_info=True)


# Werkzeuge, nach denen die offene Ansicht nachgezogen wird (Pruefung 4, M2). Nicht dabei: Alt-Text/Quickinfo je Bild oder
# Feld — die setzt die Oberflaeche schon selbst ein (refresh_image / refresh_feld), ohne die Ansicht neu zu zeichnen.
AENDERT_ANSICHT = frozenset({
    "barrierefrei_machen", "komplett_barrierefrei_machen", "pruefung_starten", "korrektur_anwenden", "korrektur_rueckgaengig",
    "dokument_umbenennen", "dokument_loeschen", "alt_sprache_setzen", "konvertiere_zu_pdfua", "exportiere_fertige_pdf",
    "exportiere_word", "uebersetze_dokument", "exportiere_uebersetzung", "testweise_taggen", "pruefdatei_erstellen",
    "exportiere_alt_texte", "exportiere_quickinfos", "alt_texte_generieren", "quickinfos_generieren", "stammdaten_anwenden",
    "ki_kontext_setzen", "eigener_prompt", "ausgabe_loeschen", "save_to_master_data",
    "bild_umbenennen", "bild_loeschen",   # Grafik- und Webseiten-Projekte (Ausbau Runde 1, Schritt 4)
})


def nachbereiten(result: dict, werkzeug: str, args: dict, user_id: int, project_id: int, vorher_id: Optional[str]) -> dict:
    """Nach JEDEM Werkzeugaufruf (ToolExecutor): Karte fuer ein neues Angebot anhaengen; Karten verbrauchter oder ersetzter
    Angebote auf „Erledigt“ bzw. „Nicht mehr gültig“ stellen (gespeichert + als Aktion an die Oberflaeche); melden, ob die
    offene Ansicht nachgezogen werden muss; Download-Links mit Frist (gueltig_bis) kennzeichnen."""
    result = karte_anhaengen(result, werkzeug, args, user_id, project_id, vorher_id)
    with _SPERRE:
        verbraucht = _VERBRAUCHT.pop((int(user_id), int(project_id)), [])
    if not isinstance(result, dict):
        return result
    erfolg = bool(result.get("ok")) and not (result.get("result") or {}).get("rueckfrage_noetig")
    karten = []
    for aid, a, eingeloest in verbraucht:
        zustand = "erledigt" if (eingeloest and erfolg) else "abgelaufen"
        felder = _kartenfelder(a, zustand)
        if zustand == "erledigt":
            with _SPERRE:
                _ERLEDIGT[aid] = (int(user_id), int(project_id), time.time(), felder)
        _karte_speichern(project_id, aid, felder)
        karten.append(dict(felder, angebot_id=aid))
    if karten:
        result["karten"] = karten
    if erfolg and werkzeug in AENDERT_ANSICHT and not (werkzeug == "eigener_prompt" and (args or {}).get("auflisten")):
        result["aktualisieren"] = True
    anh = result.get("anhang")
    if isinstance(anh, dict) and anh.get("download_url"):
        try:
            bis = _main().token_gueltig_bis(int(user_id), anh["download_url"])
        except Exception:  # noqa: BLE001
            bis = None
        if bis:
            anh["gueltig_bis"] = bis
            if isinstance(result.get("result"), dict):
                result["result"]["gueltig_bis"] = bis
    return result


def karten_im_verlauf(messages: list, _) -> list:
    """GET /chat/history: offene Karten, deren Angebot nicht mehr gilt (abgelaufen, ersetzt, Neustart), als „Nicht mehr
    gültig“ zeigen — ohne Knopf. Erledigte stehen schon so im gespeicherten Verlauf."""
    for nachricht in messages or []:
        for a in (nachricht.get("anhang") or []):
            if not (isinstance(a, dict) and a.get("art") == "bestaetigung") or a.get("zustand") in ("erledigt", "abgelaufen"):
                continue
            if angebot_nach_id(a.get("angebot_id")):
                a["zustand"] = "offen"
                continue
            e = _ERLEDIGT.get(a.get("angebot_id") or "")
            if e:
                a.update(e[3])
            else:
                a.update({"zustand": "abgelaufen", "titel": _("Nicht mehr gültig")})
    return messages


def _turn_id(turn) -> str:
    return getattr(turn, "turn_id", None) or uuid.uuid4().hex


def _freigabe(m, project: dict, user_id: int, document_id: Optional[int], art: str,
              bestaetigt: bool, turn) -> Optional[dict]:
    """Server-Rueckfrage fuer kostenpflichtige Werkzeuge. Liefert die Kostenvorschau (= Antwort an das
    Modell, keine Aktion), oder None, wenn die Aktion jetzt ausgefuehrt werden darf."""
    vorschau = _kosten_vorschau(m, project, user_id, document_id, art)
    schluessel = (int(user_id), int(project["id"]), art, document_id)
    tid = _turn_id(turn)
    if not bestaetigt:
        if vorschau.get("erlaubt"):
            _angebot_merken(schluessel, vorschau.get("preis") or 0, tid)
        return vorschau
    if getattr(turn, "kostenpflichtig", 0) >= _KOSTENPFLICHTIG_JE_TURN:
        vorschau["hinweis"] = ("In dieser Nachricht wurde schon eine kostenpflichtige Aktion ausgefuehrt. Mehr als eine je "
                               "Nachricht laesst der Server nicht zu — sag dem Nutzer, was erledigt ist, und frage fuer "
                               "das Weitere neu.")
        vorschau["grund"] = "je_nachricht"
        return vorschau
    if not vorschau.get("erlaubt"):
        vorschau["grund"] = "guthaben"
        return vorschau
    grund = _angebot_einloesen(schluessel, vorschau.get("preis") or 0, tid)
    if grund:
        vorschau["hinweis"] = grund
        vorschau["grund"] = _GRUND_KENNWORT.get(grund, "angebot")
        return vorschau
    if turn is not None and hasattr(turn, "kostenpflichtig"):
        turn.kostenpflichtig += 1
    return None
_HOERPROBE_AUSZUG = 40      # Zeilen, die pruefe_word_dokument direkt mitliefert
_HOERPROBE_MAX = 400        # Zeilen je Dokument bei lies_ausgabe(teil=hoerprobe)


def _main():
    return importlib.import_module("main")


def _ui_lang(user_id: int) -> str:
    conn = sqlite3.connect(_db_path())
    try:
        row = conn.execute("SELECT language FROM users WHERE id = ?", (user_id,)).fetchone()
        return (row[0] if row and row[0] else "de")
    except sqlite3.Error:
        return "de"
    finally:
        conn.close()


def _fehler(e: HTTPException) -> dict[str, Any]:
    d = e.detail
    if isinstance(d, dict):
        # 402 aus billing.credits_fehlen_detail: {"code","preis","verfuegbar","fehlend","text"}
        d = d.get("text") or d.get("meldung") or d.get("detail") or str(d)
    return {"ok": False, "error": str(d)}


def _doc_kurz(d: dict) -> dict:
    """Ein Dokument des Umwandlungsberichts, ohne die lange Hoerprobe."""
    return {
        "dokument": d.get("dokument"), "bilder": d.get("bilder"), "alt_texte": d.get("alt_texte"),
        "punkte": [{"bereich": p.get("bereich"), "status": p.get("status"), "text": p.get("text")}
                   for p in ((d.get("pruefung") or {}).get("punkte") or [])],
        # Saetze des Word-Pruefberichts zitieren das Dokument (Titel, Ueberschriften): gekennzeichnet (daten.py)
        "pruefbericht_hinweise_daten": [daten(b.get("text")) for b in (d.get("pruefbericht") or []) if b.get("status") != "ok"],
        "warnungen": d.get("warnungen") or [],
    }


def _anhang(art: str, r: dict, project_id: int) -> dict:
    ist_zip = (r.get("media") == "application/zip")
    return {
        "art": art, "ausgabe_id": r["ausgabe_id"], "dateiname": r.get("dateiname") or "",
        "download_url": f"/api/ausgaben/{r['ausgabe_id']}/datei",
        "ausgaben_url": f"/ablage?projekt={project_id}#ausgabe-{r['ausgabe_id']}",
        "project_id": project_id, "ausgaben_anzahl": r.get("ausgaben_anzahl"),
        "label": ("zip" if ist_zip else art),
    }


def pruefe_word_dokument(project_id: int, user_id: int, document_id: Optional[int] = None) -> dict[str, Any]:
    """Pruefbericht + Hoerprobe des Word-Dokuments mit den aktuellen Alt-Texten (kostenlos)."""
    m = _main()
    try:
        project = m._pdfua_projekt_laden(project_id, user_id, meldung="Die Hörprobe gibt es nur für Word-Projekte")
        r = m._pdfua_vorschau_sync(project, user_id, document_id, _ui_lang(user_id))
    except HTTPException as e:
        return _fehler(e)
    doks = []
    hinweise_gesamt = 0
    bilder_ohne = 0
    for d in r.get("dokumente") or []:
        hoer = d.get("hoerprobe") or []
        hinweise = [b for b in (d.get("pruefbericht") or []) if b.get("status") != "ok"]
        hinweise_gesamt += len(hinweise)
        ohne = max(0, int(d.get("bilder") or 0) - int(d.get("alt_texte") or 0))
        bilder_ohne += ohne
        doks.append({
            "dokument": d.get("dokument"), "document_id": d.get("document_id"),
            "bilder": d.get("bilder"), "alt_texte": d.get("alt_texte"), "bilder_ohne_alt_text": ohne,
            # Pruefbericht-Saetze und Hoerprobe zitieren das Dokument: gekennzeichnet (daten.py, 09.10.2026)
            "pruefbericht": text_kennzeichnen(d.get("pruefbericht") or []),
            "zahlen": d.get("zahlen") or {},
            "hoerprobe_auszug_daten": daten_zeilen(hoer[:_HOERPROBE_AUSZUG]), "hoerprobe_zeilen": len(hoer),
        })
    return {"ok": True, "result": {
        "dokumente": doks, "hinweise_gesamt": hinweise_gesamt, "bilder_ohne_alt_text": bilder_ohne,
        "hinweis": ("Die vollstaendige Hoerprobe bekommst du nach einer Umwandlung mit lies_ausgabe(teil='hoerprobe'). "
                    "Gibst du Hoerprobe-Zeilen wieder, dann ohne die Markierung. "
                    "Fehlen Alt-Texte, schlage dem Nutzer vor, sie zuerst zu erzeugen (generate_alt_text je Bild "
                    "oder Sammellauf in der Oberflaeche), bevor du umwandelst."),
    }}


def _kosten_vorschau(m, project: dict, user_id: int, document_id: Optional[int], art: str) -> dict:
    units = m._load_pdf_export_units(project, user_id, document_id)
    anzahl = sum(len(u["images"]) for u in units)
    p = m.billing.export_pruefung(user_id, anzahl, art)
    return {
        "rueckfrage_noetig": True, "preis": p.get("preis"), "verfuegbar": p.get("verfuegbar"),
        "erlaubt": bool(p.get("erlaubt")), "fehlend": p.get("fehlend"),
        "dokumente": len(units), "bilder": anzahl,
        "hinweis": ("Nenne dem Nutzer den Preis in Credits (und das Guthaben, wenn nicht unbegrenzt) und frage, "
                    "ob du fortfahren sollst. Unter deiner Antwort steht eine Karte mit genau diesem Angebot und einem Knopf "
                    "zum Bestaetigen. Schreibt der Nutzer „Ja“, erneut mit bestaetigt=true aufrufen (gilt nur fuer dieses "
                    "zuletzt genannte Angebot)."
                    if p.get("erlaubt") else
                    "Das Guthaben reicht nicht. Sag dem Nutzer Preis und Guthaben und verweise auf Abo & Verbrauch."),
    }


def konvertiere_zu_pdfua(project_id: int, user_id: int, document_id: Optional[int] = None,
                         bestaetigt: bool = False, turn=None) -> dict[str, Any]:
    """Word-Projekt in eine barrierefreie PDF (PDF/UA) umwandeln und pruefen. Kostenpflichtig:
    ohne bestaetigt=true nur Preis + Guthaben (Rueckfrage); bestaetigt=true nur mit gueltigem
    Angebot aus einer frueheren Nachricht (siehe _freigabe). `turn` = ToolExecutor der Nachricht."""
    m = _main()
    try:
        project = m._pdfua_projekt_laden(project_id, user_id)
        vorschau = _freigabe(m, project, user_id, document_id, "pdfua", bestaetigt, turn)
        if vorschau is not None:
            return {"ok": True, "result": vorschau}
        r = m._pdfua_umwandeln_sync(project, user_id, document_id, None, _ui_lang(user_id), "bot")
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {
        "ausgabe_id": r["ausgabe_id"], "dateiname": r["dateiname"], "preis": r["preis"],
        "bestanden": r["bestanden"], "zusammenfassung": r["zusammenfassung"],
        "dokumente": [_doc_kurz(d) for d in r.get("dokumente") or []],
        "download_url": f"/api/ausgaben/{r['ausgabe_id']}/datei",
        "ausgaben_url": f"/ablage?projekt={project_id}#ausgabe-{r['ausgabe_id']}",
        "ausgaben_anzahl": r.get("ausgaben_anzahl"), "aufbewahrung_tage": r.get("aufbewahrung_tage"),
        "hinweis": ("Der Nutzer sieht unter deiner Antwort einen Knopf zum Herunterladen; die Datei liegt "
                    "ausserdem unter „Meine Ablage“ (Seitenleiste), das Prüfergebnis auch in der Ansicht Barrierefreiheitsprüfung. Fasse das Ergebnis in Worten "
                    "zusammen: bestanden oder welche Bereiche Hinweise haben, und was der Nutzer dagegen tun kann."),
    }, "anhang": _anhang("pdfua", r, project_id)}


def exportiere_word(project_id: int, user_id: int, document_id: Optional[int] = None,
                    bestaetigt: bool = False, turn=None) -> dict[str, Any]:
    """Word-Datei mit den aktuellen Alt-Texten ausgeben — Download-Knopf unter der Antwort, KEIN Ablage-
    Eintrag (Steve 11.09.: Ablage nur fuer umgewandelte PDFs). Kostenpflichtig: ohne bestaetigt=true nur
    Preis + Guthaben."""
    m = _main()
    try:
        project = m._pdfua_projekt_laden(project_id, user_id, meldung="Der Word-Export ist nur fuer Word-Projekte verfuegbar")
        vorschau = _freigabe(m, project, user_id, document_id, "docx", bestaetigt, turn)
        if vorschau is not None:
            return {"ok": True, "result": vorschau}
        r = m._word_export_ausgabe_sync(project, user_id, document_id, None, _ui_lang(user_id), "bot")
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {
        "dateiname": r["dateiname"], "preis": r["preis"],
        "alt_texte": r["alt_texte"], "hinweise": r["hinweise"], "zusammenfassung": r["zusammenfassung"],
        "dokumente": [{"dokument": d.get("dokument"), "bilder": d.get("bilder"), "alt_texte": d.get("alt_texte"),
                       "pruefbericht_hinweise_daten": [daten(b.get("text")) for b in (d.get("pruefbericht") or []) if b.get("status") != "ok"],
                       "warnungen": d.get("warnungen") or []} for d in r.get("dokumente") or []],
        "download_url": r["download_url"],
        "hinweis": ("Der Nutzer sieht unter deiner Antwort einen Knopf zum Herunterladen der Word-Datei"
                    + (" und einen Link in die Ablage." if r.get("ausgabe_id") else ". Sie liegt nicht in der Ablage (dort nur umgewandelte PDFs).")
                    + " Nenne nur die Befunde des Word-Pruefberichts, nicht das Gute."),
    }, "anhang": dict({"art": "docx", "dateiname": r.get("dateiname") or "", "download_url": r["download_url"],
                       "project_id": project_id, "label": ("zip" if r.get("media") == "application/zip" else "docx")},
                      **({"ausgabe_id": r["ausgabe_id"], "ausgaben_url": f"/ablage?projekt={project_id}#ausgabe-{r['ausgabe_id']}",
                          "ausgaben_anzahl": r.get("ausgaben_anzahl")} if r.get("ausgabe_id") else {}))}


def liste_ausgaben(project_id: int, user_id: int) -> dict[str, Any]:
    """Alle Eintraege unter Meine Ablage fuer dieses Projekt (neueste zuerst)."""
    m = _main()
    try:
        m._require_project_owned(project_id, user_id)   # Ablage gibt es fuer Word- UND PDF-Projekte (22.09.2026)
    except HTTPException as e:
        return _fehler(e)
    m._ablage_dateien_einsammeln(user_id)
    eintraege = m._ausgaben_des_projekts(user_id, project_id)
    kurz = [{"ausgabe_id": a["id"], "art": a["art"], "dokument": a["dokument"] or "alle Dokumente",
             "erstellt": a["created_at"], "ausloeser": a["ausloeser"], "bestanden": a["bestanden"],
             "hinweise": a["hinweise"], "zusammenfassung": a["zusammenfassung"],
             "datei_verfuegbar": a["datei_verfuegbar"], "dateiname": a["dateiname"],
             "download_url": f"/api/ausgaben/{a['id']}/datei" if a["datei_verfuegbar"] else None}
            for a in eintraege]
    return {"ok": True, "result": {"anzahl": len(kurz), "ausgaben": kurz,
                                   "ausgaben_url": f"/ablage?projekt={project_id}",
                                   "hinweis": "Details (Befunde, Hoerprobe) je Eintrag mit lies_ausgabe(ausgabe_id, teil)."}}


def lies_ausgabe(project_id: int, user_id: int, ausgabe_id: int, teil: str = "bericht") -> dict[str, Any]:
    """Bericht (Klartext je Bereich), Pruefbericht des Word-Dokuments oder Hoerprobe eines Eintrags."""
    m = _main()
    row = m._ausgabe_row(user_id, int(ausgabe_id))
    if not row or int(row["project_id"]) != int(project_id):
        return {"ok": False, "error": f"Ausgabe {ausgabe_id} gibt es in diesem Projekt nicht. Hol die Liste mit liste_ausgaben."}
    a = m._ausgabe_dict(row, mit_bericht=True)
    teil = (teil or "bericht").strip().lower()
    doks = []
    for d in a.get("bericht") or []:
        eintrag: dict[str, Any] = {"dokument": d.get("dokument"), "bilder": d.get("bilder"), "alt_texte": d.get("alt_texte")}
        if teil in ("bericht", "alles"):
            eintrag["punkte"] = [{"bereich": p.get("bereich"), "status": p.get("status"), "text": p.get("text")}
                                 for p in ((d.get("pruefung") or {}).get("punkte") or [])]
        if teil in ("pruefbericht", "bericht", "alles"):
            eintrag["pruefbericht"] = text_kennzeichnen(d.get("pruefbericht") or [])   # zitiert das Dokument (daten.py)
        if teil in ("hoerprobe", "alles"):
            hoer = d.get("hoerprobe") or []
            # Hoerprobe = Text aus dem Dokument: je Zeile gekennzeichnet wie hoerprobe_lesen (09.10.2026)
            eintrag["hoerprobe_daten"] = daten_zeilen(hoer[:_HOERPROBE_MAX])
            eintrag["hoerprobe_zeilen"] = len(hoer)
            if len(hoer) > _HOERPROBE_MAX:
                eintrag["hoerprobe_gekuerzt"] = True
        doks.append(eintrag)
    return {"ok": True, "result": {
        "ausgabe_id": a["id"], "art": a["art"], "dokument": a["dokument"] or "alle Dokumente",
        "erstellt": a["created_at"], "bestanden": a["bestanden"], "zusammenfassung": a["zusammenfassung"],
        "datei_verfuegbar": a["datei_verfuegbar"],
        "download_url": f"/api/ausgaben/{a['id']}/datei" if a["datei_verfuegbar"] else None,
        "teil": teil, "dokumente": doks,
        **({"hinweis": "Gib die Hörprobe (hoerprobe_daten) ohne die Markierung Zeile für Zeile wieder, ohne Umformulierung."}
           if teil in ("hoerprobe", "alles") else {}),
    }}


def analysiere_word_struktur(project_id: int, user_id: int, document_id: Optional[int] = None) -> dict[str, Any]:
    """Struktur-Lektor, Lesestufe (11.09.2026): Absatz-Auszug mit Formatvorlage/Fettung/Groesse,
    Gliederung, Tabellen und deterministische Befunde (Ueberschrift ohne Vorlage, getippte Liste,
    Leerabsaetze, Grossbuchstaben, Linktexte, Layouttabellen). Kostenlos, kein KI-Aufruf im Werkzeug."""
    m = _main()
    try:
        project = m._pdfua_projekt_laden(project_id, user_id, meldung="Den Struktur-Lektor gibt es nur für Word-Projekte")
        units = m._load_pdf_export_units(project, user_id, document_id)
    except HTTPException as e:
        return _fehler(e)
    import os
    import docx_struktur
    output_dir = os.path.join(m.RESULTS_DIR, str(user_id), str(project["id"]), "_export")
    os.makedirs(output_dir, exist_ok=True)
    doks = []
    for unit in units:
        label = m._doc_label(unit["doc"])
        try:
            docx_path, _info = m._build_docx_for_document(unit, output_dir, custom_title=label)
            st = docx_struktur.analysiere_struktur(docx_path)
        except Exception as e:  # noqa: BLE001
            log.exception("analysiere_word_struktur: %s", e)
            doks.append({"dokument": label, "fehler": f"Struktur nicht lesbar: {e}"})
            continue
        try:
            doc_id = int(unit["doc"]["id"])
        except (KeyError, TypeError, ValueError):
            doc_id = None
        # Titel, Gliederung, Absaetze, Zellen und Textanfaenge der Befunde sind Text aus dem Dokument: gekennzeichnet
        # (inkluagent/daten.py, 09.10.2026); Zahlen, Stil und Befund-Saetze des Lektors bleiben, wie sie sind.
        tabellen = [dict({k: v for k, v in t.items() if k != "erste_zeile"}, erste_zeile_daten=daten_zeilen(t.get("erste_zeile")))
                    if isinstance(t, dict) and "erste_zeile" in t else t for t in (st["tabellen"] or [])]
        doks.append({"dokument": label, "document_id": doc_id, "titel_daten": daten(st["titel"]),
                     "standard_schriftgroesse_pt": st["standard_schriftgroesse"], "zahlen": st["zahlen"],
                     "gliederung": text_kennzeichnen(st["gliederung"]), "tabellen": tabellen,
                     "befunde": text_kennzeichnen(st["befunde"]),
                     "absaetze": text_kennzeichnen(st["absaetze"]), "auszug_gekuerzt": st["auszug_gekuerzt"]})
    return {"ok": True, "result": {"dokumente": doks, "hinweis": (
        "Befunde mit sicherheit=hoch sind aus dem Dokument belegt — nenne sie als Tatsache mit Absatznummer und "
        "Textanfang (text_daten, ohne die Markierung). Befunde mit sicherheit=mittel sind Vermutungen aus der Optik — nenne sie als Vermutung und frage, "
        "ob es eine Überschrift sein soll. Du darfst aus dem Absatz-Auszug eigene Beobachtungen ergänzen, gekennzeichnet "
        "als Einschätzung. Umbauen kannst du nichts; sag, was der Nutzer in Word tut (Formatvorlage zuweisen, echte "
        "Liste anlegen). Bewertung am Ende in einem Satz: gut aufgebaut / brauchbar mit n Stellen / ohne Struktur.")}}


# ─── Übersetzen als Fähigkeit des Word-Projekts (Testumbau 18.09.2026, Steve + Michael) ──────────
# Dieselben Kernfunktionen wie die Übersetzungs-Ansicht (uebersetzung_api.bot_*): Vorschau, Lauf,
# Stand, Datei. Gleiche Zwei-Schritt-Zustimmung wie bei der Umwandlung (Angebot je Nutzer/Projekt/
# Zielsprache, erst ohne bestaetigt, dann mit).

def _ueb():
    return importlib.import_module("uebersetzung_api")


def _lauf_user(user_id: int) -> dict:
    m = _main()
    u = m.get_user_by_id(user_id)
    return dict(u) if u else {"id": user_id}


def uebersetze_dokument(project_id: int, user_id: int, zielsprache: str, bestaetigt: bool = False,
                        alt_texte: bool = True, turn=None) -> dict[str, Any]:
    """Startet die Übersetzung des ganzen Projekts in die Zielsprache (Kennung aus ZIELSPRACHEN).
    Erster Aufruf ohne bestaetigt: Umfang, Preis, Guthaben; mit bestaetigt=true nach dem Ja des Nutzers:
    Lauf im Hintergrund, Ergebnis über uebersetzung_stand."""
    ue = _ueb()
    zielsprache = str(zielsprache or "").strip().lower()
    try:
        v = ue.bot_vorschau(user_id, project_id, zielsprache, alt_texte)
    except HTTPException as e:
        return _fehler(e)
    sprache_name = ue.ue.ZIELSPRACHEN[zielsprache][0]
    if not v.get("anzahl") and not (v.get("stand") or {}).get("absaetze"):
        # Nichts zu uebersetzen, weil es keine Absaetze gibt (kein Dokument, Original fehlt/unlesbar) —
        # NICHT „schon alles uebersetzt“ (Review 2, Befund 7).
        return {"ok": True, "result": {"gestartet": False, "anzahl": 0, "zielsprache": zielsprache, "sprache_name": sprache_name,
                                       "stand": v.get("stand"),
                                       "hinweis": "Es gibt keine übersetzbaren Absätze: entweder ist noch kein Word-Dokument im Projekt, "
                                                  "oder die Datei konnte nicht gelesen werden. Bitte dem Nutzer sagen, er soll ein Word-Dokument hochladen."}}
    if not v.get("anzahl"):
        return {"ok": True, "result": {"gestartet": False, "anzahl": 0, "zielsprache": zielsprache, "sprache_name": sprache_name,
                                       "stand": v.get("stand"),
                                       "hinweis": "In dieser Sprache ist schon alles übersetzt (von Hand korrigierte Absätze bleiben). "
                                                  "Der Nutzer kann die Datei herunterladen (exportiere_uebersetzung) oder eine andere Sprache wählen."}}
    schluessel = (int(user_id), int(project_id), f"uebersetzung:{zielsprache}", None)
    tid = _turn_id(turn)
    vorschau = {"rueckfrage_noetig": True, "zielsprache": zielsprache, "sprache_name": sprache_name,
                "absaetze": v["anzahl"], "woerter": v["woerter"], "preis": v["preis"], "verfuegbar": v["verfuegbar"],
                "erlaubt": bool(v["erlaubt"]), "woerter_je_credit": v["woerter_je_credit"],
                "hinweis": ("Nenne dem Nutzer Zielsprache, Absätze, Wörter und Preis in Credits (1 Credit je angefangene "
                            f"{v['woerter_je_credit']} Wörter; Guthaben nennen, wenn nicht unbegrenzt) und frage, ob du übersetzen sollst. "
                            "Erst nach ausdrücklichem Ja erneut mit bestaetigt=true aufrufen."
                            if v["erlaubt"] else "Das Guthaben reicht nicht. Sag dem Nutzer Preis und Guthaben und verweise auf Abo & Verbrauch.")}
    if not bestaetigt:
        if v["erlaubt"]:
            _angebot_merken(schluessel, v["preis"], tid)
        return {"ok": True, "result": vorschau}
    if getattr(turn, "kostenpflichtig", 0) >= _KOSTENPFLICHTIG_JE_TURN:
        vorschau["hinweis"] = ("In dieser Nachricht wurde schon eine kostenpflichtige Aktion ausgeführt. Mehr als eine je "
                               "Nachricht lässt der Server nicht zu — sag dem Nutzer, was erledigt ist, und frage für das Weitere neu.")
        return {"ok": True, "result": vorschau}
    if not v["erlaubt"]:
        return {"ok": True, "result": vorschau}
    grund = _angebot_einloesen(schluessel, v["preis"], tid)
    if grund:
        vorschau["hinweis"] = grund
        return {"ok": True, "result": vorschau}
    if turn is not None and hasattr(turn, "kostenpflichtig"):
        turn.kostenpflichtig += 1
    try:
        erg = ue.bot_starten(_lauf_user(user_id), project_id, zielsprache, alt_texte)
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {
        "gestartet": bool(erg.get("gestartet")), "anzahl": erg.get("anzahl"), "woerter": erg.get("woerter"),
        "preis": erg.get("preis"), "zielsprache": zielsprache, "sprache_name": sprache_name,
        "hinweis": ("Die Übersetzung läuft im Hintergrund (etwa eine halbe Minute je 30 Absätze). Sag dem Nutzer, dass sie "
                    "läuft, und dass du mit uebersetzung_stand nachsehen kannst; danach exportiere_uebersetzung für die Datei. "
                    "Die Übersetzung ist auch in der Ansicht „Übersetzung“ des Projekts zu sehen und zu korrigieren."),
    }}


def uebersetzung_stand(project_id: int, user_id: int) -> dict[str, Any]:
    """Stand der Übersetzung: Zielsprache, fertige/gesamte Absätze, Hinweise, laufender Lauf."""
    ue = _ueb()
    try:
        st = ue.bot_stand(user_id, project_id)
    except HTTPException as e:
        return _fehler(e)
    lauf = st.pop("lauf", None) or {}
    st["laeuft"] = bool(lauf.get("laeuft"))
    if lauf:
        st["lauf"] = {"pakete_fertig": lauf.get("pakete_fertig"), "pakete_gesamt": lauf.get("pakete_gesamt"),
                      "segmente_fertig": lauf.get("segmente_fertig"), "segmente_gesamt": lauf.get("segmente_gesamt"),
                      "credits": lauf.get("credits"), "fehler": lauf.get("fehler") or []}
    return {"ok": True, "result": st}


def exportiere_uebersetzung(project_id: int, user_id: int) -> dict[str, Any]:
    """Übersetzte Word-Datei ausgeben (kostenlos, Download-Knopf unter der Antwort; keine Ablage)."""
    ue = _ueb()
    try:
        r = ue.bot_export(user_id, project_id)
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {"dateiname": r["dateiname"], "dokumente": r["dokumente"], "warnungen": r["warnungen"],
                                   "download_url": r["download_url"],
                                   "hinweis": "Der Nutzer sieht unter deiner Antwort einen Knopf zum Herunterladen der übersetzten Word-Datei. "
                                              "Struktur und Formatierung sind unverändert, nur der Text, die Alt-Texte, der Titel und die Dokumentsprache."},
            "anhang": {"art": "docx", "dateiname": r["dateiname"], "download_url": r["download_url"], "project_id": project_id,
                       "label": "zip" if r["media"] == "application/zip" else "docx"}}
