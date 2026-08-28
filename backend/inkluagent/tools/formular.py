"""Formular-Werkzeuge des InkluAgent: Quickinfos fuer PDF-Formularfelder (28.08.2026, Steve + Fable 5).

Gegenstueck zu project.py/altext.py fuer das Quickinfo-Werkzeug (docs/FORMULAR.md).
Ein Formular-Projekt hat keine Bilder, sondern Felder (Tabelle formularfelder);
der Agent bekommt deshalb einen EIGENEN Werkzeugsatz (definitions_formular.py)
und einen eigenen System-Prompt (prompts/system_formular.py). Die Weiche liegt
in agent_loop.run_agent (project.tool == "formular") — Bild-Werkzeuge existieren
fuer den Formular-Agenten nicht, Formular-Werkzeuge nicht fuer den Bild-Agenten.

Konventionen wie bei den Bild-Werkzeugen:
  - Rueckgabe {"ok": True, "result": ...} oder {"ok": False, "error": "..."}.
  - project_id + user_id kommen aus dem Sitzungskontext (ToolExecutor), NIE aus
    den Modell-Argumenten -> kein Zugriff auf fremde Projekte.
  - Falsche feld_id -> Fehlermeldung mit den echten ids (Selbstkorrektur statt
    Halluzination), Muster _wrong_image_id_hint.
  - Keine Serverpfade, keine Feldwerte nach aussen (Feldwerte werden ohnehin nie
    gespeichert, siehe formular_processor).

Fachliche Absicherung beim Speichern (Gegenstueck zum Bild-Verify des
Alt-Text-Agenten): update_quickinfo laeuft durch DIESELBE deterministische
Nachpruefung wie der Feld-Pass (formular_ki.nachpruefung) — Beleg im Seitentext,
Lage in Feldnaehe, Regeln (Floskel, Feldart, Format, Pflicht). Ergebnis
"niedrig" wird NICHT gespeichert, ausser der Nutzer besteht darauf (force=true).
So schreibt der Chatbot nach denselben Massstaeben wie die Pipeline.

Abrechnung (Regel Steve 31.07.2026, uebertragen): Reden ist frei; sobald der
Agent eine Quickinfo ERZEUGT (generate_quickinfo, Feld-Pass) oder AENDERT
(update_quickinfo), kostet es 1 Credit — dieselben Aktionen wie in der Oberflaeche.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import billing
import formular_api                                   # Helfer + Deps (_d) des Quickinfo-Werkzeugs
import formular_ki
from database import get_db

log = logging.getLogger(__name__)

MAX_FELDER_LISTE = 400          # Schutz gegen Riesenformulare im Tool-Result (8000-Zeichen-Kappe im Loop)
MAX_SEITENTEXT = 6000           # Seitentext je Feld-Detail
QUELLE_TEXT = {"": "offen", "pdf": "vorhanden (aus der PDF)", "hand": "von Hand", "stammdaten": "aus Stammdaten",
               "ki": "KI-Vorschlag", "gast": "vom Gast bearbeitet"}
FELDART_TEXT = {"text": "Textfeld", "checkbox": "Kontrollkästchen", "radio": "Auswahlknopf", "dropdown": "Auswahlliste",
                "liste": "Listenfeld", "button": "Schaltfläche", "signatur": "Unterschriftsfeld", "unbekannt": "Feld"}


# --------------------------------------------------------------------------- Zugriff / Helfer

def _projekt(conn, project_id: int, user_id: int):
    return conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ? AND tool = 'formular'",
                        (project_id, user_id)).fetchone()


def _feld_ids(conn, project_id: int) -> list[int]:
    return [r["id"] for r in conn.execute(
        """SELECT f.id FROM formularfelder f LEFT JOIN documents d ON d.id = f.document_id
           WHERE f.project_id = ? ORDER BY COALESCE(d.doc_index, 0), f.page_number, f.feld_index""", (project_id,)).fetchall()]


def _wrong_feld_id_hint(conn, feld_id: int, project_id: int) -> str:
    ids = _feld_ids(conn, project_id)
    return (f"feld_id={feld_id} existiert nicht im Projekt {project_id}. Die echten feld_ids sind: {ids}. "
            "Vermutlich hast du die UI-Nummer (Feld 1 = erstes Feld) statt der echten feld_id benutzt. "
            "Ordne die UI-Nummer ueber list_form_fields der echten feld_id zu und rufe das Werkzeug erneut.")


def _feld(conn, feld_id: int, project_id: int):
    return conn.execute("SELECT * FROM formularfelder WHERE id = ? AND project_id = ?", (feld_id, project_id)).fetchone()


def _ui_labels(conn, project_id: int) -> dict[int, str]:
    """feld_id -> Anzeige wie in der Oberflaeche: 'Feld n' (ein Dokument) bzw. 'Dokument d, Feld n'."""
    docs = conn.execute("SELECT id, doc_index FROM documents WHERE project_id = ? ORDER BY doc_index", (project_id,)).fetchall()
    multi = len(docs) > 1
    labels = {}
    for r in conn.execute(
            """SELECT f.id, f.feld_index, f.document_id, COALESCE(d.doc_index, 0) AS doc_index FROM formularfelder f
               LEFT JOIN documents d ON d.id = f.document_id WHERE f.project_id = ?""", (project_id,)).fetchall():
        labels[r["id"]] = (f"Dokument {r['doc_index']}, Feld {r['feld_index']}" if multi and r["doc_index"]
                           else f"Feld {r['feld_index']}")
    return labels


def _namenlos(feld) -> bool:
    return str(feld["anker"] or "").startswith("#")


def _liste(text: str) -> list:
    try:
        v = json.loads(text or "[]")
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _kurz(feld, label: str) -> dict:
    """Kompakte Feldzeile fuer die Uebersicht (ohne Umfeld/Seitentext)."""
    qi = feld["quickinfo"] or ""
    return {
        "feld_id": feld["id"], "ui_label": label, "feld_index": feld["feld_index"],
        "feld_art": FELDART_TEXT.get(feld["feld_art"] or "unbekannt", "Feld"), "page": feld["page_number"],
        "beschriftung": feld["beschriftung"] or "", "gruppe": feld["gruppe"] or "",
        "pflicht": bool(feld["pflicht"]), "ausgefuellt": bool(feld["ausgefuellt"]),
        "quickinfo": qi[:200], "status": "beschrieben" if qi.strip() else "offen",
        "quelle": QUELLE_TEXT.get(feld["quelle"] or "", feld["quelle"] or ""),
        "sicherheit": feld["sicherheit"] or "", "namenlos": _namenlos(feld),
        "pruefstatus": feld["review_status"] or "offen",
    }


# --------------------------------------------------------------------------- Werkzeuge

def list_form_fields(project_id: int, user_id: int) -> dict[str, Any]:
    """Uebersicht: Projekt, Dokumente, alle Felder mit ui_label, Beschriftung, Status, Quelle."""
    conn = get_db()
    try:
        p = _projekt(conn, project_id, user_id)
        if not p:
            return {"ok": False, "error": "Projekt nicht gefunden, kein Zugriff oder kein Formular-Projekt."}
        docs = [dict(d) for d in conn.execute(
            "SELECT id, doc_index, original_filename, display_name FROM documents WHERE project_id = ? ORDER BY doc_index",
            (project_id,)).fetchall()]
        labels = _ui_labels(conn, project_id)
        rows = conn.execute(
            """SELECT f.* FROM formularfelder f LEFT JOIN documents d ON d.id = f.document_id
               WHERE f.project_id = ? ORDER BY COALESCE(d.doc_index, 0), f.page_number, f.feld_index""",
            (project_id,)).fetchall()
        anzahl_stammdaten = conn.execute("SELECT COUNT(*) FROM stammdaten WHERE user_id = ?", (user_id,)).fetchone()[0]
    finally:
        conn.close()
    felder = [_kurz(r, labels.get(r["id"], f"Feld {r['feld_index']}")) for r in rows[:MAX_FELDER_LISTE]]
    offen = sum(1 for f in felder if f["status"] == "offen")
    return {"ok": True, "result": {
        "project": {"id": p["id"], "name": p["name"] or p["filename"], "status": p["status"],
                    "sprache_der_quickinfos": p["alt_language"] or "de", "multi_doc": len(docs) > 1},
        "documents": [{"doc_index": d["doc_index"], "original_filename": d["original_filename"],
                       "display_name": d["display_name"]} for d in docs],
        "feld_count": len(rows), "offen": offen, "beschrieben": len(rows) - offen,
        "stammdaten_eintraege": anzahl_stammdaten,
        "gekuerzt": len(rows) > MAX_FELDER_LISTE,
        "felder": felder,
    }}


def get_field_details(feld_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Alles zu einem Feld: Beschriftung mit Lage, Abschnitt, Umfeld, Optionen, Original-Quickinfo,
    KI-Beleg/Hinweise, Pruefstatus + Anmerkung des Gastes, Seitentext (Kontext) — wie der Kontextabsatz
    in der Oberflaeche, nur vollstaendiger."""
    conn = get_db()
    try:
        if not _projekt(conn, project_id, user_id):
            return {"ok": False, "error": "Projekt nicht gefunden, kein Zugriff oder kein Formular-Projekt."}
        f = _feld(conn, feld_id, project_id)
        if not f:
            return {"ok": False, "error": _wrong_feld_id_hint(conn, feld_id, project_id)}
        label = _ui_labels(conn, project_id).get(feld_id, f"Feld {f['feld_index']}")
        # Seitentext liegt am ersten Feld der Seite (formular_processor speichert ihn je Feld, formular_api
        # kuerzt nur beim Ausliefern) — hier direkt aus dem Feld, sonst vom Nachbarfeld derselben Seite.
        seitentext = f["page_text"] or ""
        if not seitentext:
            r = conn.execute("SELECT page_text FROM formularfelder WHERE document_id = ? AND page_number = ? AND page_text != '' LIMIT 1",
                             (f["document_id"], f["page_number"])).fetchone()
            seitentext = (r["page_text"] if r else "") or ""
    finally:
        conn.close()
    d = _kurz(f, label)
    d.update({
        "beschriftung_lage": f["beschriftung_lage"] or "", "umfeld": f["umfeld"] or "",
        "optionen": _liste(f["optionen"]), "seiten": _liste(f["seiten"]),
        "quickinfo": f["quickinfo"] or "", "quickinfo_original": f["quickinfo_original"] or "",
        "beleg": f["beleg"] or "", "ki_hinweise": _liste(f["ki_hinweise"]),
        "anmerkung_des_gastes": f["review_note"] or "",
        "technischer_feldname": f["feld_name"] or "",
        "seitentext": seitentext[:MAX_SEITENTEXT],
        "hat_ausschnitt": bool(f["ausschnitt_path"]), "hat_seitenansicht": bool(f["page_view_path"]),
    })
    return {"ok": True, "result": d}


def view_field(feld_id: int, project_id: int, user_id: int, ganze_seite: bool = False) -> dict[str, Any]:
    """Laedt den Feld-Ausschnitt (oder die ganze Seite mit nummerierten Rahmen) als Bild in den
    naechsten Turn — Muster view_image (image_bytes-Sonderfall im Agent-Loop)."""
    conn = get_db()
    try:
        if not _projekt(conn, project_id, user_id):
            return {"ok": False, "error": "Projekt nicht gefunden, kein Zugriff oder kein Formular-Projekt."}
        f = _feld(conn, feld_id, project_id)
        if not f:
            return {"ok": False, "error": _wrong_feld_id_hint(conn, feld_id, project_id)}
    finally:
        conn.close()
    pfad = (f["page_view_path"] if ganze_seite else f["ausschnitt_path"]) or ""
    wurzel = os.path.realpath(formular_api._d.results_dir) + os.sep
    if not pfad or not os.path.realpath(pfad).startswith(wurzel) or not os.path.isfile(pfad):
        return {"ok": False, "error": "Fuer dieses Feld gibt es keinen Bild-Ausschnitt." if not ganze_seite
                else "Fuer diese Seite gibt es keine Seitenansicht."}
    try:
        with open(pfad, "rb") as fh:
            data = fh.read()
    except OSError as e:
        return {"ok": False, "error": f"Datei nicht lesbar: {e}"}
    return {"ok": True, "result": {"feld_id": feld_id, "art": "seitenansicht" if ganze_seite else "ausschnitt",
                                   "size_bytes": len(data),
                                   "info": "Bild geladen, im naechsten Turn fuer dich sichtbar. Feldwerte sind NICHT enthalten "
                                           "(widgetfreie Arbeitskopie)."},
            "image_bytes": data}


def _kontingent_ok(user_id: int) -> Optional[str]:
    if not billing.pruefe_kontingent(user_id).get("erlaubt", True):
        return "Das Monatskontingent dieses Kontos ist aufgebraucht. Unter Einstellungen → Abo & Verbrauch gibt es Zusatz-Credits."
    return None


def generate_quickinfo(feld_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Feld-Pass fuer EIN Feld (formular_ki.generiere_seite mit Variation) — identisch zum Knopf
    „Generieren“ in der Oberflaeche: ueberschreibt bewusst, quelle 'ki', 1 Credit."""
    fehler = _kontingent_ok(user_id)
    if fehler:
        return {"ok": False, "error": fehler}
    conn = get_db()
    try:
        p = _projekt(conn, project_id, user_id)
        if not p:
            return {"ok": False, "error": "Projekt nicht gefunden, kein Zugriff oder kein Formular-Projekt."}
        f = _feld(conn, feld_id, project_id)
        if not f:
            return {"ok": False, "error": _wrong_feld_id_hint(conn, feld_id, project_id)}
        if _namenlos(f):
            return {"ok": False, "error": "Dieses Feld hat keinen Feldnamen; eine Quickinfo kann dafuer nicht in die PDF geschrieben werden."}
        if p["status"] in ("extracting", "processing"):
            return {"ok": False, "error": "Fuer dieses Projekt laeuft gerade eine Verarbeitung (Alle generieren). Bitte kurz warten."}
        doc = dict(conn.execute("SELECT * FROM documents WHERE id = ?", (f["document_id"],)).fetchone())
        bestaetigte = formular_api._bestaetigte_quickinfos(conn, project_id, ausser_feld=feld_id)
        user_prompt = formular_api._user_prompt(conn, project_id)
        feld_ki = formular_api._feld_fuer_ki(dict(f))
        sprache = p["alt_language"] or "de"
    finally:
        conn.close()
    try:
        vorschlaege = formular_ki.generiere_seite(
            formular_api._originalpfad(doc), f["page_number"], [feld_ki], sprache=sprache,
            formular_titel=formular_api._d.doc_label(doc), seiten_gesamt=formular_api._seitenzahl(doc),
            bestaetigte=bestaetigte, user_prompt=user_prompt, variation=True)
    except formular_ki.FeldPassFehler as e:
        return {"ok": False, "error": f"Feld-Pass fehlgeschlagen: {e}"}
    except Exception as e:
        log.exception("generate_quickinfo: Feld-Pass crashte")
        return {"ok": False, "error": f"Feld-Pass fehlgeschlagen: {e}"}
    if not vorschlaege:
        return {"ok": False, "error": "Die KI hat fuer dieses Feld keinen Vorschlag geliefert."}
    v = vorschlaege[0]
    conn = get_db()
    try:
        conn.execute("""UPDATE formularfelder SET quickinfo = ?, quelle = 'ki', sicherheit = ?, beleg = ?, ki_hinweise = ?,
                        updated_at = datetime('now') WHERE id = ?""",
                     (v.quickinfo, v.sicherheit, v.beleg, json.dumps(v.hinweise, ensure_ascii=False), feld_id))
        conn.commit()
    finally:
        conn.close()
    billing.verbuche(user_id, "chatbot", aktion="quickinfo_generierung")
    return {"ok": True, "result": {"feld_id": feld_id, "quickinfo": v.quickinfo, "sicherheit": v.sicherheit,
                                   "beleg": v.beleg, "hinweise": v.hinweise, "quelle": "ki",
                                   "info": "Quickinfo gespeichert (quelle KI). Zurueck auf Original bleibt moeglich."}}


def _nachpruefen(feld, doc: dict, quickinfo: str, beleg: str) -> formular_ki.FeldVorschlag:
    """Dieselbe deterministische Nachpruefung wie im Feld-Pass — fuer einen vom Agenten formulierten Text."""
    zeilen, seitentext = formular_ki.seiten_zeilen(formular_api._originalpfad(doc), int(feld["page_number"] or 1))
    v = formular_ki.FeldVorschlag(feld_id=feld["id"], quickinfo=quickinfo, beleg=beleg, sicherheit="hoch")
    return formular_ki.nachpruefung(v, formular_api._feld_fuer_ki(dict(feld)), zeilen, seitentext)


def update_quickinfo(feld_id: int, project_id: int, user_id: int, new_quickinfo: str, beleg: str = "",
                     force: bool = False) -> dict[str, Any]:
    """Speichert eine vom Nutzer abgenommene Quickinfo. Vorher dieselbe Nachpruefung wie der Feld-Pass
    (Beleg im Seitentext, Lage, Regeln). Ergebnis 'niedrig' wird NICHT gespeichert (ausser force=true
    nach ausdruecklichem Beharren). quelle 'ki' mit Sicherheit/Beleg — das Badge in der Oberflaeche zeigt
    „KI-Vorschlag, sicher/mittel“. 1 Credit (Aenderungs-Fall)."""
    fehler = _kontingent_ok(user_id)
    if fehler:
        return {"ok": False, "error": fehler}
    text = formular_api._sauber(new_quickinfo, formular_api.MAX_QUICKINFO)
    if not text:
        return {"ok": False, "error": "Die Quickinfo darf nicht leer sein."}
    if len(text) < 3:
        return {"ok": False, "error": "Die Quickinfo ist zu kurz."}
    if len(text) > formular_ki.MAX_QUICKINFO_LAENGE:
        return {"ok": False, "error": f"Quickinfo zu lang ({len(text)} Zeichen, hoechstens {formular_ki.MAX_QUICKINFO_LAENGE}). "
                                      "Eine Quickinfo ist ein Satz — kuerze sie."}
    conn = get_db()
    try:
        p = _projekt(conn, project_id, user_id)
        if not p:
            return {"ok": False, "error": "Projekt nicht gefunden, kein Zugriff oder kein Formular-Projekt."}
        f = _feld(conn, feld_id, project_id)
        if not f:
            return {"ok": False, "error": _wrong_feld_id_hint(conn, feld_id, project_id)}
        if _namenlos(f):
            return {"ok": False, "error": "Dieses Feld hat keinen Feldnamen; eine Quickinfo kann dafuer nicht in die PDF geschrieben werden."}
        doc = dict(conn.execute("SELECT * FROM documents WHERE id = ?", (f["document_id"],)).fetchone())
    finally:
        conn.close()

    sicherheit, beleg_txt, hinweise = "hoch", (beleg or "").strip(), []
    if not force:
        try:
            v = _nachpruefen(f, doc, text, beleg_txt)
            text, sicherheit, hinweise = v.quickinfo, v.sicherheit, list(v.hinweise)
        except Exception:
            log.exception("update_quickinfo: Nachpruefung fehlgeschlagen (ignoriert, kein Blocker)")
        if sicherheit == "niedrig":
            return {"ok": False, "error": (
                "NICHT gespeichert: Die Nachpruefung (gleiche Pruefung wie der Feld-Pass) findet keinen Beleg fuer "
                f"diesen Text auf der Formularseite. Hinweise: {'; '.join(hinweise) or 'kein Beleg'}. Gib beim naechsten "
                "Aufruf im Parameter beleg die WOERTLICHE Textstelle der Seite an (Beschriftung neben dem Feld, Abschnitt), "
                "oder lege dem Nutzer die Beanstandung vor. NUR wenn der Nutzer ausdruecklich auf seiner Fassung besteht "
                "(er weiss z. B. etwas, das nicht auf der Seite steht), rufe update_quickinfo erneut mit force=true auf.")}
    else:
        sicherheit, hinweise = "mittel", ["Auf ausdrueckliche Nutzer-Entscheidung ohne Nachpruefung gespeichert."]

    conn = get_db()
    try:
        conn.execute("""UPDATE formularfelder SET quickinfo = ?, quelle = 'ki', sicherheit = ?, beleg = ?, ki_hinweise = ?,
                        updated_at = datetime('now') WHERE id = ? AND project_id = ?""",
                     (text, sicherheit, beleg_txt, json.dumps(hinweise, ensure_ascii=False), feld_id, project_id))
        conn.commit()
    finally:
        conn.close()
    billing.verbuche(user_id, "chatbot", aktion="quickinfo_aenderung_chatbot")
    return {"ok": True, "result": {"feld_id": feld_id, "quickinfo": text, "sicherheit": sicherheit, "beleg": beleg_txt,
                                   "hinweise": hinweise, "quelle": "ki",
                                   "info": ("Gespeichert. Nachpruefung: " + ("alles belegt." if not hinweise else "; ".join(hinweise)))
                                   if not force else "Gespeichert (force, ohne Nachpruefung)."}}


def revert_quickinfo(feld_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Zurueck auf das Original aus der PDF (leer, wenn die PDF keine Quickinfo hatte) — wie der Knopf."""
    conn = get_db()
    try:
        if not _projekt(conn, project_id, user_id):
            return {"ok": False, "error": "Projekt nicht gefunden, kein Zugriff oder kein Formular-Projekt."}
        f = _feld(conn, feld_id, project_id)
        if not f:
            return {"ok": False, "error": _wrong_feld_id_hint(conn, feld_id, project_id)}
        orig = f["quickinfo_original"] or ""
        conn.execute("UPDATE formularfelder SET quickinfo = ?, quelle = ?, sicherheit = '', beleg = '', ki_hinweise = '', "
                     "updated_at = datetime('now') WHERE id = ?", (orig, "pdf" if orig else "", feld_id))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "result": {"feld_id": feld_id, "quickinfo": orig, "quelle": "pdf" if orig else "",
                                   "info": "Original wiederhergestellt." if orig else "Die PDF hatte keine Quickinfo — das Feld ist jetzt leer (offen)."}}


def search_master_data(query: str, project_id: int, user_id: int, feld_art: str = "") -> dict[str, Any]:
    """Sucht in den Stammdaten des Kontos (Beschriftung, Feldname, Quickinfo, Teiltreffer)."""
    q = formular_api._norm_beschriftung(query or "")
    conn = get_db()
    try:
        rows = conn.execute("SELECT id, beschriftung, feld_art, feld_name, quickinfo, sprache, herkunft FROM stammdaten "
                            "WHERE user_id = ? ORDER BY beschriftung", (user_id,)).fetchall()
    finally:
        conn.close()
    treffer = []
    for r in rows:
        if feld_art and (r["feld_art"] or "") != feld_art:
            continue
        heu = " ".join([formular_api._norm_beschriftung(r["beschriftung"] or ""), (r["feld_name"] or "").lower(),
                        (r["quickinfo"] or "").lower()])
        if not q or q in heu:
            treffer.append({"stammdaten_id": r["id"], "beschriftung": r["beschriftung"], "feld_art": r["feld_art"],
                            "feld_name": r["feld_name"], "quickinfo": r["quickinfo"], "sprache": r["sprache"],
                            "herkunft": r["herkunft"]})
    return {"ok": True, "result": {"anzahl_gesamt": len(rows), "treffer": treffer[:20],
                                   "info": "Uebernahme in ein Feld: update_quickinfo mit dem Wortlaut aus der Quickinfo "
                                           "(quelle bleibt KI); in die Stammdaten aufnehmen: save_to_master_data."}}


def save_to_master_data(feld_id: int, project_id: int, user_id: int) -> dict[str, Any]:
    """Nimmt die aktuelle Quickinfo eines Feldes in die Stammdaten des Kontos auf — wie der Knopf
    „In Stammdaten uebernehmen“ (gleicher Schluessel wird aktualisiert, keine Dublette)."""
    conn = get_db()
    try:
        p = _projekt(conn, project_id, user_id)
        if not p:
            return {"ok": False, "error": "Projekt nicht gefunden, kein Zugriff oder kein Formular-Projekt."}
        f = _feld(conn, feld_id, project_id)
        if not f:
            return {"ok": False, "error": _wrong_feld_id_hint(conn, feld_id, project_id)}
        qi = (f["quickinfo"] or "").strip()
        if not qi:
            return {"ok": False, "error": "Dieses Feld hat noch keine Quickinfo — erst eine setzen, dann in die Stammdaten uebernehmen."}
        if not (f["beschriftung"] or "").strip() and not (f["feld_name"] or "").strip():
            return {"ok": False, "error": "Ohne Beschriftung und Feldname laesst sich kein Stammdaten-Schluessel bilden."}
        sid = formular_api._stammdaten_upsert(conn, user_id, f["beschriftung"] or "", f["feld_art"] or "text",
                                              f["feld_name"] or "", qi, p["alt_language"] or "de", "feld")
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "result": {"stammdaten_id": sid, "beschriftung": f["beschriftung"] or "", "feld_art": f["feld_art"],
                                   "quickinfo": qi, "info": "In die Stammdaten des Kontos uebernommen."}}
