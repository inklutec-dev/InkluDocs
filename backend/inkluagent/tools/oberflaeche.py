"""Werkzeuge fuer alles, was man in der Oberflaeche von Hand macht (Steve 30.09.2026: „alles, was man händisch macht, soll
über den InkluAgent gehen“ — und umgekehrt nichts, was die Oberflaeche nicht anbietet, siehe funktionen.py).

Jedes Werkzeug ruft DENSELBEN Kern wie der Knopf (main, formular_api, tagging_api) — keine zweite Logik: dieselben Preise,
dieselben Sperren und Grenzen, dieselbe Rueckfrage vor kostenpflichtigen oder unumkehrbaren Schritten (Angebot ->
Bestaetigung in einer eigenen Nachricht, pdf._freigabe). Dateien kommen als Download-Knopf unter der Antwort
(main.sofort_download_ablegen, derselbe Token-Weg wie „Als Word“ im Chatbot).
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

from fastapi import HTTPException

from . import ausgaben as _ausg
from . import pdf as _pdf

log = logging.getLogger(__name__)

_FORMATE = {"csv": "CSV", "xlsx": "Excel", "json": "JSON"}
TESTLAUF_WARTEN_S = 150   # so lange wartet testweise_taggen auf das Ende, bevor es „laeuft noch“ meldet


def _main():
    return importlib.import_module("main")


def _formular():
    return importlib.import_module("formular_api")


def _fehler(e: HTTPException) -> dict[str, Any]:
    return _ausg._fehler(e)


def _uebersetzer(user_id: int):
    try:
        return _main().get_gettext(_ausg._ui_lang(user_id))
    except Exception:  # noqa: BLE001
        return None


def _doc_id(project_id: int, user_id: int, document_id: Optional[int]) -> Optional[int]:
    """Ein genanntes Dokument pruefen (gehoert zum Projekt); None bleibt None (= ganzes Projekt)."""
    if document_id in (None, 0, ""):
        return None
    conn = _pdf._get_db()
    try:
        _pdf._projekt(conn, project_id, user_id, jede_art=True)
        return _pdf._dokument(conn, project_id, document_id)["id"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tagging: „Testweise taggen“ (kostenlos)
# ---------------------------------------------------------------------------

def testweise_taggen(project_id: int, user_id: int, document_id: Optional[int] = None) -> dict[str, Any]:
    """Wie der Knopf „Testweise taggen“: kostenlos, im Testmodus, eigene Testfassung; das Dokument bleibt unveraendert."""
    t = _pdf._tagging()
    conn = _pdf._get_db()
    try:
        _pdf._projekt(conn, project_id, user_id)
        doc = _pdf._dokument(conn, project_id, document_id)
    except HTTPException as e:
        return _fehler(e)
    finally:
        conn.close()
    lang = _ausg._ui_lang(user_id)
    try:
        r = t.test_starten_fuer(project_id, doc["id"], {"id": user_id, "language": lang}, lang, _uebersetzer(user_id))
    except HTTPException as e:
        return _fehler(e)
    # Auf das Ende warten (Pruefung 3 Barrierefreiheit, N6: der Chat meldete das Ende nicht von selbst) — ein Testlauf
    # dauert meist unter einer Minute; laeuft er laenger, sagt die Antwort, dass er noch laeuft.
    import time as _time
    ende = _time.time() + TESTLAUF_WARTEN_S
    while doc["id"] in t._test_laeuft and _time.time() < ende:
        _time.sleep(2)
    if doc["id"] in t._test_laeuft:
        return {"ok": True, "result": dict(r, dokument=_pdf._name(doc), fertig=False, hinweis=(
            "Der Testlauf läuft noch (kostenlos). Das Dokument bleibt unverändert; die Testfassung ist nicht zum Herunterladen. "
            "Sag das; das Ergebnis steht gleich in der Ansicht „Tagging“ und in dokument_stand (Feld testlauf)."))}
    st = _pdf.dokument_stand(project_id, user_id, doc["id"])
    testlauf = (((st.get("result") or {}).get("dokumente") or [{}])[0]).get("testlauf") if st.get("ok") else None
    return {"ok": True, "result": dict(r, dokument=_pdf._name(doc), fertig=True, testlauf=testlauf, hinweis=(
        "Der Testlauf ist fertig (kostenlos, das Dokument bleibt unverändert, die Testfassung ist nicht zum Herunterladen). "
        "Nenne die Struktur und ob die PDF/UA-Prüfung der Testfassung bestanden ist; bei einem Fehler den Grund."))}


# ---------------------------------------------------------------------------
# Barrierefreiheitspruefung: Pruefdatei erstellen und lesen (kostenlos)
# ---------------------------------------------------------------------------

def _pruef_daten(project_id: int, user_id: int, document_id: Optional[int]) -> tuple:
    m = _main()
    project = m._abschluss_projekt(project_id, user_id)
    conn = _pdf._get_db()
    try:
        doc = _pdf._dokument(conn, project_id, document_id)
    finally:
        conn.close()
    units = m._load_pdf_export_units(project, user_id, doc["id"])
    _ = _uebersetzer(user_id) or (lambda s: s)
    return project, doc, m._abschluss_dokument(project, units[0], user_id, _, True)


def _probleme_kurz(d: dict) -> list:
    out = []
    for p in d.get("probleme") or []:
        tl = p.get("teile") or {}
        out.append({"nr": p.get("nr"), "seiten": p.get("seiten") or [], "bereich": tl.get("bereich") or "",
                    "satz": tl.get("satz") or p.get("text") or "", "anzahl": tl.get("mal") or "",
                    "regeln": p.get("regeln") or [], "englisch": tl.get("lang") == "en"})
    return out


def pruefdatei_erstellen(project_id: int, user_id: int, document_id: Optional[int] = None) -> dict[str, Any]:
    """Wie „Prüfung starten“ in der Barrierefreiheitsprüfung: die fertige Datei bauen (wie beim Herunterladen, mit
    Alt-Texten und Quickinfos) und mit veraPDF pruefen. Kostenlos, keine Ablage. Dieselbe Sperre wie der Knopf."""
    m = _main()
    try:
        project = m._abschluss_projekt(project_id, user_id)
        conn = _pdf._get_db()
        try:
            doc = _pdf._dokument(conn, project_id, document_id)
        finally:
            conn.close()
        neu = m._abschluss_erstellen_sync(project, user_id, doc["id"])
    except HTTPException as e:
        return _fehler(e)
    r = pruefdatei_lesen(project_id, user_id, doc["id"])
    if r.get("ok"):
        r["result"]["neu_gebaut"] = bool(neu)
    return r


def pruefdatei_lesen(project_id: int, user_id: int, document_id: Optional[int] = None, teil: str = "probleme",
                     von: int = 1, anzahl: int = 80) -> dict[str, Any]:
    """Ergebnis der Pruefdatei zum Vorlesen: teil="probleme" = veraPDF-Urteil und die Problemstellen (je Regel eine Zeile,
    mit Seiten und Regelnummer); teil="hoerprobe" = die Hoerprobe der FERTIGEN Datei (mit Alt-Texten und Quickinfos),
    seitenweise ueber von/anzahl, begrenzt wie hoerprobe_lesen."""
    try:
        _project, doc, d = _pruef_daten(project_id, user_id, document_id)
    except HTTPException as e:
        return _fehler(e)
    pd = d.get("pruefdatei")
    if d.get("laeuft"):
        return {"ok": True, "result": {"status": "laeuft", "dokument": _pdf._name(doc),
                                       "hinweis": "Die Prüfdatei wird gerade erstellt. Sag das und sieh gleich noch einmal nach."}}
    if not pd:
        return {"ok": True, "result": {"status": "keine_pruefdatei", "dokument": _pdf._name(doc), "getaggt": d.get("getaggt"),
                                       "hinweis": ("Noch keine Prüfdatei. Biete pruefdatei_erstellen an (kostenlos)."
                                                   if d.get("getaggt") else
                                                   "Die PDF hat keine Tags; eine Prüfdatei gibt es erst nach dem Tagging.")}}
    kopf = {"dokument": _pdf._name(doc), "erstellt_am": pd.get("erstellt_am"), "aktuell": pd.get("aktuell"),
            "verapdf_moeglich": pd.get("verapdf_moeglich"), "bestanden": pd.get("bestanden"),
            "pruefpunkte_erfuellt": pd.get("pruefpunkte_erfuellt"), "pruefpunkte_verletzt": pd.get("pruefpunkte_verletzt"),
            "anzahl_probleme": pd.get("anzahl_probleme")}
    if not pd.get("aktuell"):
        kopf["hinweis_veraltet"] = "Die Prüfdatei ist älter als der Stand (Alt-Texte, Quickinfos oder Name geändert): neu erstellen."
    if teil == "hoerprobe":
        hp = d.get("hoerprobe") or {"kopf": [], "seiten": []}
        zeilen = list(hp.get("kopf") or [])
        for sd in hp.get("seiten") or []:
            zeilen.append(f"— Seite {sd.get('seite')} —")
            zeilen.extend(sd.get("zeilen") or ["(Auf dieser Seite liest ein Screenreader nichts vor.)"])
        zeilen = _pdf._hoerprobe_teile(zeilen)
        von = max(1, int(von or 1))
        anzahl = max(1, min(int(anzahl or 80), _pdf._HOERPROBE_MAX))
        stueck, zeichen = [], 0
        for z in zeilen[von - 1: von - 1 + anzahl]:
            laenge = _pdf._antwort_laenge(z)
            if stueck and zeichen + laenge > _pdf._HOERPROBE_ZEICHEN:
                break
            stueck.append(z)
            zeichen += laenge
        return {"ok": True, "result": dict(kopf, teil="hoerprobe", zeilen_gesamt=len(zeilen), von=von, bis=von - 1 + len(stueck),
                                           zeilen_daten=[_pdf._DATEN_MARKE + z for z in stueck], hinweis=(
            "Das ist die Hörprobe der fertigen Datei (wie beim Herunterladen). Gib die Zeilen ohne Markierung Zeile für Zeile "
            "wieder, ohne Umformulierung. Sind noch Zeilen übrig, biete an weiterzulesen (von = bis + 1)."))}
    return {"ok": True, "result": dict(kopf, teil="probleme", probleme=_probleme_kurz(d), hinweis=(
        "Nenne zuerst das Urteil von veraPDF (bestanden oder nicht, Zahl der Problemstellen), dann jede Problemstelle mit "
        "Seiten und Satz; die Regelnummer nur, wenn der Nutzer danach fragt. Sätze mit englisch=true sind der Originaltext von "
        "veraPDF — sag das. Die Hörprobe der fertigen Datei liest du mit teil=\"hoerprobe\"."))}


# ---------------------------------------------------------------------------
# Herunterladen: Alt-Texte als Tabelle, Quickinfos als CSV (kostenpflichtig, fester Preis)
# ---------------------------------------------------------------------------

def exportiere_alt_texte(project_id: int, user_id: int, format: str = "csv", document_id: Optional[int] = None,
                         bestaetigt: bool = False, turn=None) -> dict[str, Any]:
    """Wie „Alt-Texte herunterladen“ (Als CSV / Als Excel / Als JSON): fester Preis je Vorgang, ohne document_id alle
    Dokumente (bei mehreren als ZIP). Download-Knopf unter der Antwort."""
    m = _main()
    fmt = (format or "csv").strip().lower()
    if fmt not in _FORMATE:
        return {"ok": False, "error": "Format: csv, xlsx (Excel) oder json"}
    try:
        doc_id = _doc_id(project_id, user_id, document_id)
        p = m.billing.aktion_pruefung(user_id, m.billing.TABELLEN_EXPORTE[fmt])
    except HTTPException as e:
        return _fehler(e)
    vorschau = {"format": _FORMATE[fmt], "document_id": doc_id, "preis": p.get("preis"), "verfuegbar": p.get("verfuegbar"),
                "erlaubt": bool(p.get("erlaubt")), "fehlend": p.get("fehlend")}
    grund = _pdf._freigabe(user_id, project_id, "tabelle_" + fmt, doc_id, int(p.get("preis") or 0), bool(p.get("erlaubt")), bestaetigt, turn)
    if grund == "rueckfrage":
        return {"ok": True, "result": _pdf._rueckfrage(vorschau, f"die Alt-Texte als {_FORMATE[fmt]}")}
    if grund:
        return {"ok": True, "result": dict(vorschau, hinweis=grund, rueckfrage_noetig=True)}
    try:
        erg = m._tabellen_export_bauen(project_id, user_id, fmt, doc_id, None)
        url = m.sofort_download_ablegen(user_id, project_id, erg["dateiname"], erg["media"], pfad=erg.get("pfad"), daten=erg.get("daten"))
    except HTTPException as e:
        return _fehler(e)
    m.billing.verbuche(user_id, "export", aktion=erg["aktion"], credits=erg["preis"])
    ist_zip = erg["media"] == "application/zip"
    return {"ok": True, "result": {"dateiname": erg["dateiname"], "preis": erg["preis"], "download_url": url,
                                   "hinweis": ("Der Nutzer sieht unter deiner Antwort einen Knopf zum Herunterladen. Nenne Format und "
                                               "Preis" + (f"; bei mehreren Dokumenten ist es ein ZIP mit je einer {_FORMATE[fmt]}-Datei "
                                                          "pro Dokument — sag das." if ist_zip else "."))},
            "anhang": {"art": ("zip" if ist_zip else fmt), "dateiname": erg["dateiname"], "download_url": url,
                       "label": ("zip" if ist_zip else fmt), "format": fmt}}


def exportiere_quickinfos(project_id: int, user_id: int, document_id: Optional[int] = None, bestaetigt: bool = False,
                          turn=None) -> dict[str, Any]:
    """Wie „Quickinfos herunterladen“ (Als CSV, Feldliste): fester Preis je Vorgang. Download-Knopf unter der Antwort."""
    m = _main()
    fa = _formular()
    try:
        doc_id = _doc_id(project_id, user_id, document_id)
        p = m.billing.aktion_pruefung(user_id, "formular_csv_export")
    except HTTPException as e:
        return _fehler(e)
    vorschau = {"format": "CSV", "document_id": doc_id, "preis": p.get("preis"), "verfuegbar": p.get("verfuegbar"),
                "erlaubt": bool(p.get("erlaubt")), "fehlend": p.get("fehlend")}
    grund = _pdf._freigabe(user_id, project_id, "quickinfo_csv", doc_id, int(p.get("preis") or 0), bool(p.get("erlaubt")), bestaetigt, turn)
    if grund == "rueckfrage":
        return {"ok": True, "result": _pdf._rueckfrage(vorschau, "die Quickinfos als CSV")}
    if grund:
        return {"ok": True, "result": dict(vorschau, hinweis=grund, rueckfrage_noetig=True)}
    try:
        inhalt, dateiname, preis = fa.quickinfo_csv_bauen(project_id, user_id, doc_id, None)
        url = m.sofort_download_ablegen(user_id, project_id, dateiname, "text/csv; charset=utf-8", daten=inhalt.encode("utf-8"))
    except HTTPException as e:
        return _fehler(e)
    m.billing.verbuche(user_id, "export", aktion="formular_csv_export", credits=preis)
    return {"ok": True, "result": {"dateiname": dateiname, "preis": preis, "download_url": url,
                                   "hinweis": "Der Nutzer sieht unter deiner Antwort einen Knopf zum Herunterladen. Nenne den Preis."},
            "anhang": {"art": "csv", "dateiname": dateiname, "download_url": url, "label": "csv"}}


# ---------------------------------------------------------------------------
# Sammellaeufe: „Alt-Texte generieren“, „Quickinfos generieren“ (kostenpflichtig je Bild/Feld)
# ---------------------------------------------------------------------------

def alt_texte_generieren(project_id: int, user_id: int, document_id: Optional[int] = None, bestaetigt: bool = False,
                         turn=None) -> dict[str, Any]:
    """Wie „Alt-Texte generieren“ (Projekt oder ein Dokument): dieselbe Auswahl, derselbe Preis je Bild, derselbe Lauf im
    Hintergrund. Ueberschreibt vorhandene Texte (wie der Knopf; der Nutzer muss das wissen)."""
    m = _main()
    body = {"modus": "alle"}
    try:
        doc_id = _doc_id(project_id, user_id, document_id)
        if doc_id:
            body["document_id"] = doc_id
        v = m._generierung_vorschau_daten(project_id, user_id, dict(body))
    except HTTPException as e:
        return _fehler(e)
    if not v.get("anzahl"):
        return {"ok": True, "result": {"gestartet": False, "anzahl": 0, "hinweis": "Nichts zu generieren."}}
    vorschau = {"bilder": v["anzahl"], "eigene_texte_werden_ueberschrieben": v.get("eigene"), "mit_text_aus_der_datei": v.get("mit_quelltext"),
                "preis": v["preis"], "preis_je_bild": v.get("preis_je"), "verfuegbar": v.get("verfuegbar"),
                "machbar": v.get("machbar"), "erlaubt": bool(v.get("erlaubt")), "fehlend": v.get("fehlend"), "document_id": doc_id}
    grund = _pdf._freigabe(user_id, project_id, "alt_texte", doc_id, int(v.get("preis") or 0), bool(v.get("erlaubt")), bestaetigt, turn)
    if grund == "rueckfrage":
        r = _pdf._rueckfrage(vorschau, f"die Alt-Texte für {v['anzahl']} Bilder")
        if v.get("eigene"):
            r["hinweis"] += f" Sag dazu, dass {v['eigene']} eigene Texte überschrieben würden."
        return {"ok": True, "result": r}
    if grund:
        return {"ok": True, "result": dict(vorschau, hinweis=grund, rueckfrage_noetig=True)}
    try:
        user = m.get_user_by_id(user_id) or {"id": user_id}
        erg = m._generierung_vorbereiten(project_id, dict(user), body)
        ids = erg.pop("_ids", None)
        if erg.pop("_lauf", None):
            m.im_hauptloop(m._process_project(project_id, user_id, force=True, document_id=erg["document_id"], ki_neu_ids=ids))
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": dict(erg, hinweis=(
        "Die Generierung läuft im Hintergrund (je Bild einige Sekunden). Sag das; den Stand zeigt die Ansicht „Alt-Texte“, "
        "und du siehst mit list_project_images nach. Behaupte nicht, es sei fertig."))}


def quickinfos_generieren(project_id: int, user_id: int, document_id: Optional[int] = None, bestaetigt: bool = False,
                          turn=None) -> dict[str, Any]:
    """Wie „Quickinfos generieren“: alle benannten Felder (Projekt oder ein Dokument), Preis je Feld, Lauf im Hintergrund.
    Ueberschreibt vorhandene Quickinfos (wie der Knopf)."""
    m = _main()
    fa = _formular()
    try:
        doc_id = _doc_id(project_id, user_id, document_id)
        v = fa.quickinfos_vorschau_daten(project_id, user_id, doc_id)
    except HTTPException as e:
        return _fehler(e)
    if not v.get("anzahl"):
        return {"ok": True, "result": {"gestartet": False, "anzahl": 0, "hinweis": "Keine Felder zu beschreiben."}}
    vorschau = {"felder": v["anzahl"], "preis": v["preis"], "preis_je_feld": v.get("preis_je"), "verfuegbar": v.get("verfuegbar"),
                "machbar": v.get("machbar"), "erlaubt": bool(v.get("erlaubt")), "fehlend": v.get("fehlend"), "document_id": doc_id}
    grund = _pdf._freigabe(user_id, project_id, "quickinfos", doc_id, int(v.get("preis") or 0), bool(v.get("erlaubt")), bestaetigt, turn)
    if grund == "rueckfrage":
        return {"ok": True, "result": _pdf._rueckfrage(vorschau, f"die Quickinfos für {v['anzahl']} Felder")}
    if grund:
        return {"ok": True, "result": dict(vorschau, hinweis=grund, rueckfrage_noetig=True)}
    try:
        user = m.get_user_by_id(user_id) or {"id": user_id}
        erg = fa.quickinfos_vorbereiten(project_id, dict(user), doc_id)
        if erg.get("gestartet"):
            m.im_hauptloop(fa._generiere_projekt(project_id, user_id, doc_id, erg["modus"]))
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": dict(erg, hinweis=(
        "Die Quickinfos werden im Hintergrund erzeugt. Sag das; den Stand zeigt die Ansicht „Quickinfos“, und du siehst mit "
        "list_form_fields nach. Behaupte nicht, es sei fertig."))}


# ---------------------------------------------------------------------------
# Kostenlose Einstellungen und Kleinigkeiten
# ---------------------------------------------------------------------------

def stammdaten_anwenden(project_id: int, user_id: int) -> dict[str, Any]:
    """Wie „Stammdaten auf alle Felder anwenden“: offene Felder bekommen die passende Quickinfo aus den Stammdaten."""
    try:
        n = _formular().stammdaten_auf_felder(project_id, user_id, True)
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {"uebernommen": n, "hinweis": "Nenne die Zahl der übernommenen Quickinfos (kostenlos)."}}


def ki_kontext_setzen(project_id: int, user_id: int, an: bool) -> dict[str, Any]:
    """Wie das Kästchen „KI-Kontext aus dem Dokument verwenden“: gilt für Alt-Texte, die ab jetzt erzeugt werden."""
    try:
        wert = _main()._ki_kontext_setzen(project_id, user_id, bool(an))
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {"ki_kontext": wert, "hinweis": "Gilt für Alt-Texte, die ab jetzt erzeugt werden; vorhandene bleiben."}}


def eigener_prompt(project_id: int, user_id: int, prompt_id: Optional[int] = None, auflisten: bool = False) -> dict[str, Any]:
    """Wie die Auswahl „Gespeicherte Prompts“: auflisten=true zeigt die eigenen Prompts und die aktuelle Wahl; prompt_id
    setzt einen (0 = kein eigener Prompt)."""
    m = _main()
    conn = _pdf._get_db()
    try:
        project = _pdf._projekt(conn, project_id, user_id, jede_art=True)
        prompts = [{"prompt_id": r["id"], "name": r["name"], "beschreibung": r["description"] or ""}
                   for r in conn.execute("SELECT id, name, description FROM user_prompts WHERE user_id = ? ORDER BY name", (user_id,)).fetchall()]
    except HTTPException as e:
        return _fehler(e)
    finally:
        conn.close()
    if auflisten or prompt_id is None:
        return {"ok": True, "result": {"prompts": prompts, "aktuell": project.get("prompt_id"),
                                       "hinweis": "Nenne die Prompts mit Namen; setzen mit prompt_id (0 = kein eigener Prompt)."}}
    try:
        wert = m._prompt_setzen(project_id, user_id, prompt_id)
    except HTTPException as e:
        return _fehler(e)
    return {"ok": True, "result": {"prompt_id": wert, "hinweis": "Gilt für alles, was ab jetzt generiert wird."}}


def ausgabe_loeschen(project_id: int, user_id: int, ausgabe_id: int, bestaetigt: bool = False, turn=None) -> dict[str, Any]:
    """Wie „Löschen“ in der Ablage: Eintrag samt Datei loeschen — unumkehrbar, deshalb erst Rueckfrage, Ja in eigener
    Nachricht, dann bestaetigt=true. Nur Eintraege dieses Projekts."""
    m = _main()
    row = m._ausgabe_row(user_id, int(ausgabe_id))
    if not row or row["project_id"] != project_id:
        return {"ok": False, "error": f"Einen Ablage-Eintrag {ausgabe_id} gibt es in diesem Projekt nicht. Liste über liste_ausgaben."}
    a = m._ausgabe_dict(row)
    vorschau = {"ausgabe_id": a["id"], "dateiname": a.get("dateiname"), "art": a.get("art_label"), "erstellt": a.get("created_at"),
                "preis_damals": a.get("preis")}
    grund = _pdf._freigabe(user_id, project_id, "ablage_loeschen", int(ausgabe_id), 0, True, bestaetigt, turn)
    if grund == "rueckfrage":
        return {"ok": True, "result": dict(vorschau, rueckfrage_noetig=True, hinweis=(
            "Löschen ist unumkehrbar: Datei, Prüfbericht und Hörprobe dieses Eintrags sind danach weg. Sag das und frage. "
            "Unter deiner Antwort steht eine Karte mit genau diesem Angebot und einem Knopf zum Bestätigen; ein getipptes Ja "
            "gilt nur für dieses letzte Angebot (dann erneut mit bestaetigt=true aufrufen)."))}
    if grund:
        return {"ok": True, "result": dict(vorschau, rueckfrage_noetig=True, hinweis=grund)}
    m._ablage_eintrag_weg(user_id, row)
    return {"ok": True, "result": {"geloescht": True, "ausgabe_id": int(ausgabe_id)}}
