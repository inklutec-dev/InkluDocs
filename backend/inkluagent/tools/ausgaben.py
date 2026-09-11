"""Werkzeuge „Meine Ausgaben" fuer den Chatbot (Schritt 2, 11.09.2026, Steve + Fable 5).

Der InkluAgent kann ein Word-Projekt zu Ende bringen: Pruefbericht und Hoerprobe lesen,
in eine barrierefreie PDF umwandeln (Umwandler + veraPDF), die Word-Datei mit Alt-Texten
ausgeben, das Regal lesen. Alles laeuft ueber DIESELBEN Kernfunktionen wie der
Export-Bereich (main._pdfua_umwandeln_sync, _pdfua_vorschau_sync, _word_export_ausgabe_sync)
— ein Weg, zwei Bediener. Jede Umwandlung landet als Eintrag unter „Meine Ausgaben"
(ausloeser 'bot'); die Oberflaeche zeigt unter der Antwort den Download-Knopf (Anhang).

Zwei Regeln, die der SERVER durchsetzt (nicht nur der Prompt):
- Kostenpflichtige Werkzeuge verlangen bestaetigt=true. Der erste Aufruf liefert nur
  Preis und Guthaben (rueckfrage_noetig) — der Bot muss den Nutzer fragen.
- Projekt- und Nutzerkontext kommen aus der Sitzung (ToolExecutor), nie aus Modell-Argumenten.

main wird zur Laufzeit importiert (sys.modules), weil main die Agenten-Module selbst erst
in den Endpunkten laedt — ein Import beim Modulstart waere zirkulaer.
"""
from __future__ import annotations

import importlib
import logging
import sqlite3
from typing import Any, Optional

from fastapi import HTTPException

log = logging.getLogger(__name__)

_DB_PATH = "/app/data/inkludocs.db"
_HOERPROBE_AUSZUG = 40      # Zeilen, die pruefe_word_dokument direkt mitliefert
_HOERPROBE_MAX = 400        # Zeilen je Dokument bei lies_ausgabe(teil=hoerprobe)


def _main():
    return importlib.import_module("main")


def _ui_lang(user_id: int) -> str:
    conn = sqlite3.connect(_DB_PATH)
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
        "ausgaben_url": f"/ausgaben?projekt={project_id}#ausgabe-{r['ausgabe_id']}",
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
                         bestaetigt: bool = False) -> dict[str, Any]:
    """Word-Projekt in eine barrierefreie PDF (PDF/UA) umwandeln und pruefen. Kostenpflichtig:
    ohne bestaetigt=true nur Preis + Guthaben (Rueckfrage)."""
    m = _main()
    try:
        project = m._pdfua_projekt_laden(project_id, user_id)
        if not bestaetigt:
            return {"ok": True, "result": _kosten_vorschau(m, project, user_id, document_id, "pdfua")}
        r = m._pdfua_umwandeln_sync(project, user_id, document_id, None, _ui_lang(user_id), "bot")
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {
        "ausgabe_id": r["ausgabe_id"], "dateiname": r["dateiname"], "preis": r["preis"],
        "bestanden": r["bestanden"], "zusammenfassung": r["zusammenfassung"],
        "dokumente": [_doc_kurz(d) for d in r.get("dokumente") or []],
        "download_url": f"/api/ausgaben/{r['ausgabe_id']}/datei",
        "ausgaben_url": f"/ausgaben?projekt={project_id}#ausgabe-{r['ausgabe_id']}",
        "ausgaben_anzahl": r.get("ausgaben_anzahl"), "aufbewahrung_tage": r.get("aufbewahrung_tage"),
        "hinweis": ("Der Nutzer sieht unter deiner Antwort einen Knopf zum Herunterladen; die Datei liegt "
                    "ausserdem unter „Meine Ausgaben“ (Reiter Ausgaben im Projekt). Fasse das Ergebnis in Worten "
                    "zusammen: bestanden oder welche Bereiche Hinweise haben, und was der Nutzer dagegen tun kann."),
    }, "anhang": _anhang("pdfua", r, project_id)}


def exportiere_word(project_id: int, user_id: int, document_id: Optional[int] = None,
                    bestaetigt: bool = False) -> dict[str, Any]:
    """Word-Datei mit den aktuellen Alt-Texten ausgeben (Eintrag unter Meine Ausgaben).
    Kostenpflichtig: ohne bestaetigt=true nur Preis + Guthaben."""
    m = _main()
    try:
        project = m._pdfua_projekt_laden(project_id, user_id, meldung="Der Word-Export ist nur fuer Word-Projekte verfuegbar")
        if not bestaetigt:
            return {"ok": True, "result": _kosten_vorschau(m, project, user_id, document_id, "docx")}
        r = m._word_export_ausgabe_sync(project, user_id, document_id, None, _ui_lang(user_id), "bot")
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {
        "ausgabe_id": r["ausgabe_id"], "dateiname": r["dateiname"], "preis": r["preis"],
        "alt_texte": r["alt_texte"], "hinweise": r["hinweise"], "zusammenfassung": r["zusammenfassung"],
        "dokumente": [{"dokument": d.get("dokument"), "bilder": d.get("bilder"), "alt_texte": d.get("alt_texte"),
                       "pruefbericht_hinweise": [b.get("text") for b in (d.get("pruefbericht") or []) if b.get("status") != "ok"],
                       "warnungen": d.get("warnungen") or []} for d in r.get("dokumente") or []],
        "download_url": f"/api/ausgaben/{r['ausgabe_id']}/datei",
        "ausgaben_url": f"/ausgaben?projekt={project_id}#ausgabe-{r['ausgabe_id']}",
        "ausgaben_anzahl": r.get("ausgaben_anzahl"),
        "hinweis": "Der Nutzer sieht unter deiner Antwort einen Knopf zum Herunterladen; die Datei liegt ausserdem unter „Meine Ausgaben“.",
    }, "anhang": _anhang("docx", r, project_id)}


def liste_ausgaben(project_id: int, user_id: int) -> dict[str, Any]:
    """Alle Eintraege unter Meine Ausgaben fuer dieses Projekt (neueste zuerst)."""
    m = _main()
    try:
        m._pdfua_projekt_laden(project_id, user_id, meldung="Ausgaben gibt es nur fuer Word-Projekte")
    except HTTPException as e:
        return _fehler(e)
    m._ausgaben_aufraeumen(user_id)
    eintraege = m._ausgaben_des_projekts(user_id, project_id)
    kurz = [{"ausgabe_id": a["id"], "art": a["art"], "dokument": a["dokument"] or "alle Dokumente",
             "erstellt": a["created_at"], "ausloeser": a["ausloeser"], "bestanden": a["bestanden"],
             "hinweise": a["hinweise"], "zusammenfassung": a["zusammenfassung"],
             "datei_verfuegbar": a["datei_verfuegbar"], "dateiname": a["dateiname"],
             "download_url": f"/api/ausgaben/{a['id']}/datei" if a["datei_verfuegbar"] else None}
            for a in eintraege]
    return {"ok": True, "result": {"anzahl": len(kurz), "ausgaben": kurz,
                                   "ausgaben_url": f"/ausgaben?projekt={project_id}",
                                   "hinweis": "Details (Bericht, Hoerprobe) je Eintrag mit lies_ausgabe(ausgabe_id, teil)."}}


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
