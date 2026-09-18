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
import time
import uuid
from typing import Any, Optional

from fastapi import HTTPException

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


def _angebot_merken(schluessel: tuple, preis: int, turn_id: str) -> None:
    jetzt = time.time()
    for k in [k for k, a in _ANGEBOTE.items() if jetzt - a["zeit"] > _ANGEBOT_GUELTIG_S]:
        _ANGEBOTE.pop(k, None)
    _ANGEBOTE[schluessel] = {"preis": int(preis or 0), "turn": turn_id, "zeit": jetzt}


def _angebot_einloesen(schluessel: tuple, preis: int, turn_id: str) -> Optional[str]:
    """None = Zustimmung gueltig (Angebot wird verbraucht). Sonst der Grund, warum nicht."""
    a = _ANGEBOTE.get(schluessel)
    if not a or time.time() - a["zeit"] > _ANGEBOT_GUELTIG_S:
        _ANGEBOTE.pop(schluessel, None)
        return ("Es liegt kein gueltiges Angebot vor: erst OHNE bestaetigt aufrufen, dem Nutzer Preis und Guthaben "
                "nennen und auf sein Ja warten.")
    if a["turn"] == turn_id:
        return ("Die Zustimmung muss vom Nutzer in einer eigenen, spaeteren Nachricht kommen — nicht in derselben "
                "Nachricht wie die Preisauskunft. Nenne den Preis und warte auf sein Ja.")
    if a["preis"] != int(preis or 0):
        _ANGEBOTE.pop(schluessel, None)
        return "Der Preis hat sich seit der Auskunft geaendert. Nenne dem Nutzer den neuen Preis und frage erneut."
    _ANGEBOTE.pop(schluessel, None)
    return None


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
        return vorschau
    if not vorschau.get("erlaubt"):
        return vorschau
    grund = _angebot_einloesen(schluessel, vorschau.get("preis") or 0, tid)
    if grund:
        vorschau["hinweis"] = grund
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
        "pruefbericht_hinweise": [b.get("text") for b in (d.get("pruefbericht") or []) if b.get("status") != "ok"],
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
            "pruefbericht": d.get("pruefbericht") or [],
            "zahlen": d.get("zahlen") or {},
            "hoerprobe_auszug": hoer[:_HOERPROBE_AUSZUG], "hoerprobe_zeilen": len(hoer),
        })
    return {"ok": True, "result": {
        "dokumente": doks, "hinweise_gesamt": hinweise_gesamt, "bilder_ohne_alt_text": bilder_ohne,
        "hinweis": ("Die vollstaendige Hoerprobe bekommst du nach einer Umwandlung mit lies_ausgabe(teil='hoerprobe'). "
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
                    "ob du fortfahren sollst. Erst nach ausdruecklichem Ja erneut mit bestaetigt=true aufrufen."
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
                    "ausserdem unter „Meine Ablage“ (Knopf „Ablage“ neben Herunterladen). Fasse das Ergebnis in Worten "
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
                       "pruefbericht_hinweise": [b.get("text") for b in (d.get("pruefbericht") or []) if b.get("status") != "ok"],
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
        m._pdfua_projekt_laden(project_id, user_id, meldung="Ausgaben gibt es nur fuer Word-Projekte")
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
            eintrag["pruefbericht"] = d.get("pruefbericht") or []
        if teil in ("hoerprobe", "alles"):
            hoer = d.get("hoerprobe") or []
            eintrag["hoerprobe"] = hoer[:_HOERPROBE_MAX]
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
        doks.append({"dokument": label, "document_id": doc_id, "titel": st["titel"],
                     "standard_schriftgroesse_pt": st["standard_schriftgroesse"], "zahlen": st["zahlen"],
                     "gliederung": st["gliederung"], "tabellen": st["tabellen"], "befunde": st["befunde"],
                     "absaetze": st["absaetze"], "auszug_gekuerzt": st["auszug_gekuerzt"]})
    return {"ok": True, "result": {"dokumente": doks, "hinweis": (
        "Befunde mit sicherheit=hoch sind aus dem Dokument belegt — nenne sie als Tatsache mit Absatznummer und "
        "Textanfang. Befunde mit sicherheit=mittel sind Vermutungen aus der Optik — nenne sie als Vermutung und frage, "
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
