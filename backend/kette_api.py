"""Kette „Komplett barrierefrei machen“ (22.09.2026, Steve + Michael Karbe).

Ein Knopf in der Ansicht „Dokument“ eines PDF-Projekts arbeitet die Stationen nacheinander ab:
  1. Barrierefrei machen (PDFix-Tagging, tagging_api) fuer jede Datei ohne Struktur,
  2. Alt-Texte fuer alle Bilder (Sammellauf wie „Alt-Texte generieren“, main),
  3. Quickinfos fuer alle benannten Formularfelder (Feld-Pass wie „Quickinfos generieren“, formular_api).
EINE Rueckfrage vorher (Umfang je Station, Preis je Station, Gesamtpreis, Guthaben), EINE Statuskarte
waehrend des Laufs (welche Station, wie weit), EINE Laufmeldung am Ende. Jede Station bucht ihre Credits
selbst wie bisher; die Kette prueft vorher, ob das Guthaben fuer alles reicht. Eine Station, die
scheitert, stoppt die Kette nicht — die naechste laeuft, der Grund steht in der Zusammenfassung.

Stand der Kette: projects.kette_json (JSON, ueberlebt Neuladen) + Prozess-Merker; nach einem Neustart
gilt eine laufende Kette als abgebrochen (Start-Reparatur).
  GET  /api/projects/{id}/kette   Vorschau (Umfang, Preise, Guthaben) + Stand
  POST /api/projects/{id}/kette   Kette starten (400/402/409/429 wie die Einzelstationen)
Nur Besitzer, nur PDF-Projekte (Werkzeug pdf), nie im Gastweg.
"""
from __future__ import annotations

import asyncio
import threading
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

log = logging.getLogger(__name__)

SCHRITTE = ("tagging", "alttexte", "quickinfos")


@dataclass
class Deps:
    get_current_user: Callable
    get_db: Callable
    billing: object
    tageslimit_wache: Callable                 # (user) -> None | dict
    tageslimit_text: Callable                  # (dict) -> str
    dokument_getaggt: Callable                 # (doc) -> bool
    seitenzahl: Callable                       # (doc) -> int
    tagging_lauf: Callable                     # SYNC (project_id, document_id, user_id, sprache, ui_lang) -> dict
    alttexte_kandidaten: Callable              # (conn, project_id) -> int
    alttexte_lauf: Callable                    # async (project_id, user_id) -> dict
    quickinfos_kandidaten: Callable            # (conn, project_id) -> int
    quickinfos_lauf: Callable                  # async (project_id, user_id) -> dict
    resolve_ui_language: Callable = None
    preis_tagging: str = "pdf_tagging"
    preis_alttexte: str = "bild_generierung"
    preis_quickinfos: str = "quickinfo_generierung"


_d: Optional[Deps] = None
_laeuft: dict[int, dict] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None
_start_lock = threading.Lock()   # Start aus Endpunkt UND Chatbot-Thread: pruefen-und-markieren atomar (22.09.2026)   # Hauptschleife (fuer Starts aus Threads, z. B. Chatbot)


def _user():
    return _d.get_current_user


def _projekt(conn, project_id: int, user_id: int) -> dict:
    p = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
    if not p:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    p = dict(p)
    if p.get("tool") != "pdf" or p.get("project_type") != "pdf":
        raise HTTPException(status_code=400, detail="Die Kette gibt es nur für PDF-Projekte")
    return p


def _stand(project: dict) -> dict:
    try:
        k = json.loads(project.get("kette_json") or "{}")
        return k if isinstance(k, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _speichern(project_id: int, stand: dict) -> None:
    conn = _d.get_db()
    try:
        conn.execute("UPDATE projects SET kette_json = ? WHERE id = ?", (json.dumps(stand, ensure_ascii=False), project_id))
        conn.commit()
    finally:
        conn.close()


def vorschau(conn, project: dict, user_id: int) -> dict:
    """Umfang und Preis je Station — dieselben Zaehlungen wie die Einzelstationen."""
    docs = [dict(r) for r in conn.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index", (project["id"],)).fetchall()]
    zu_taggen = [d for d in docs if (d.get("original_path") or "").lower().endswith(".pdf") and not _d.dokument_getaggt(d)]
    seiten = sum(_d.seitenzahl(d) for d in zu_taggen)
    bilder = _d.alttexte_kandidaten(conn, project["id"])
    felder = _d.quickinfos_kandidaten(conn, project["id"])
    b = _d.billing
    preise = {
        "tagging": b.aktion_preis(_d.preis_tagging, seiten),
        "alttexte": b.aktion_preis(_d.preis_alttexte, bilder),
        "quickinfos": b.aktion_preis(_d.preis_quickinfos, felder),
    }
    gesamt = sum(preise.values())
    verf = b.verfuegbare_credits(user_id)
    stand = _stand(project)
    return {
        "dokumente": len(docs),
        "tagging": {"dokumente": len(zu_taggen), "schon_getaggt": len(docs) - len(zu_taggen), "seiten": seiten, "preis": preise["tagging"],
                    "namen": [(d.get("display_name") or d.get("original_filename") or "") for d in zu_taggen]},
        "alttexte": {"bilder": bilder, "preis": preise["alttexte"]},
        "quickinfos": {"felder": felder, "preis": preise["quickinfos"]},
        "gesamt": gesamt,
        "verfuegbar": verf,
        "erlaubt": (verf is None or verf >= gesamt) and gesamt > 0,
        "nichts_zu_tun": gesamt == 0 and not zu_taggen and not bilder and not felder,
        "laeuft": bool(stand.get("laeuft")) or project["id"] in _laeuft,
        "stand": stand,
        "projekt_status": project.get("status"),
    }


# ---------------------------------------------------------------------------
# Der Lauf
# ---------------------------------------------------------------------------

def _neuer_stand(plan: dict) -> dict:
    return {
        "laeuft": True, "gestartet": time.strftime("%Y-%m-%d %H:%M:%S"), "beendet": "", "schritt": "tagging",
        "schritte": {
            "tagging": {"status": "offen" if plan["tagging"]["dokumente"] else "uebersprungen", "geplant": plan["tagging"]["dokumente"], "fertig": 0, "fehler": []},
            "alttexte": {"status": "offen" if plan["alttexte"]["bilder"] else "uebersprungen", "geplant": plan["alttexte"]["bilder"], "fertig": 0, "fehler": []},
            "quickinfos": {"status": "offen" if plan["quickinfos"]["felder"] else "uebersprungen", "geplant": plan["quickinfos"]["felder"], "fertig": 0, "fehler": []},
        },
        "zusammenfassung": "",
    }


async def _kette(project_id: int, user_id: int, plan: dict, ui_lang: str, sprache: str) -> None:
    stand = _neuer_stand(plan)
    _laeuft[project_id] = stand
    _speichern(project_id, stand)
    loop = asyncio.get_running_loop()
    try:
        # 1. Tagging je Datei ohne Struktur
        s = stand["schritte"]["tagging"]
        if s["status"] != "uebersprungen":
            s["status"] = "laeuft"
            stand["schritt"] = "tagging"
            _speichern(project_id, stand)
            conn = _d.get_db()
            try:
                docs = [dict(r) for r in conn.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index", (project_id,)).fetchall()]
            finally:
                conn.close()
            for d in docs:
                if not (d.get("original_path") or "").lower().endswith(".pdf") or _d.dokument_getaggt(d):
                    continue
                name = d.get("display_name") or d.get("original_filename") or f"Dokument {d.get('doc_index')}"
                try:
                    erg = await loop.run_in_executor(None, _d.tagging_lauf, project_id, d["id"], user_id, sprache, ui_lang)
                except Exception as e:  # noqa: BLE001
                    log.exception("[kette] Tagging %s/%s", project_id, d["id"])
                    erg = {"status": "fehler", "grund": "Unerwarteter Fehler beim Tagging"}
                if erg.get("status") == "fertig":
                    s["fertig"] += 1
                else:
                    s["fehler"].append(f"{name}: {erg.get('grund') or 'Tagging fehlgeschlagen'}")
                _speichern(project_id, stand)
            s["status"] = "fertig" if not s["fehler"] else ("teilweise" if s["fertig"] else "fehler")
            _speichern(project_id, stand)
        # 2. Alt-Texte
        s = stand["schritte"]["alttexte"]
        conn = _d.get_db()
        try:
            s["geplant"] = _d.alttexte_kandidaten(conn, project_id)   # nach dem Tagging koennen es andere Bilder sein
        finally:
            conn.close()
        if s["geplant"]:
            s["status"] = "laeuft"
            stand["schritt"] = "alttexte"
            _speichern(project_id, stand)
            try:
                erg = await _d.alttexte_lauf(project_id, user_id)
            except Exception as e:  # noqa: BLE001
                log.exception("[kette] Alt-Texte %s", project_id)
                erg = {"gestartet": False, "fehler_grund": "Unerwarteter Fehler beim Generieren"}
            s["fertig"] = int(erg.get("fertig") or 0)
            if erg.get("hinweis"):
                s["fehler"].append(str(erg["hinweis"]))
            if erg.get("fehler_grund"):
                s["fehler"].append(erg["fehler_grund"])
            if erg.get("fehlgeschlagen"):
                s["fehler"].append(f"{erg['fehlgeschlagen']} Bilder ohne Alt-Text (Fehler bei der Generierung)")
            s["status"] = "fertig" if s["fertig"] >= s["geplant"] and not s["fehler"] else ("teilweise" if s["fertig"] else "fehler")
        else:
            s["status"] = "uebersprungen"
        _speichern(project_id, stand)
        # 3. Quickinfos
        s = stand["schritte"]["quickinfos"]
        conn = _d.get_db()
        try:
            s["geplant"] = _d.quickinfos_kandidaten(conn, project_id)
        finally:
            conn.close()
        if s["geplant"]:
            s["status"] = "laeuft"
            stand["schritt"] = "quickinfos"
            _speichern(project_id, stand)
            try:
                erg = await _d.quickinfos_lauf(project_id, user_id)
            except Exception as e:  # noqa: BLE001
                log.exception("[kette] Quickinfos %s", project_id)
                erg = {"gestartet": False, "fehler": ["Unerwarteter Fehler beim Generieren der Quickinfos"]}
            s["fertig"] = int(erg.get("felder_neu") or 0)
            s["fehler"] = [str(x) for x in (erg.get("fehler") or [])]
            s["status"] = "fertig" if s["fertig"] >= s["geplant"] and not s["fehler"] else ("teilweise" if s["fertig"] else "fehler")
        else:
            s["status"] = "uebersprungen"
        stand["schritt"] = "fertig"
    except asyncio.CancelledError:
        stand["schritt"] = "fehler"
        stand["zusammenfassung"] = "Die Kette wurde durch einen Neustart des Servers abgebrochen."

        stand["zusammenfassung"] = (stand["zusammenfassung"] + " Bitte Alt-Texte und Quickinfos in den Ansichten prüfen, dann exportieren.").strip()   # Steve 23.09.2026
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("[kette] Projekt %s", project_id)
        stand["schritt"] = "fehler"
        stand["zusammenfassung"] = "Unerwarteter Fehler in der Kette."
    finally:
        stand["laeuft"] = False
        stand["beendet"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if not stand.get("zusammenfassung"):
            stand["zusammenfassung"] = zusammenfassung(stand)
        _laeuft.pop(project_id, None)
        _speichern(project_id, stand)
        # Projektstatus nie in einem Laufzustand zuruecklassen
        conn = _d.get_db()
        try:
            conn.execute("UPDATE projects SET status = 'extracted' WHERE id = ? AND status IN ('extracting', 'processing')", (project_id,))
            conn.commit()
        finally:
            conn.close()


def zusammenfassung(stand: dict) -> str:
    """Ein Satz je Station, deutsch (die Oberflaeche uebersetzt die Bausteine selbst, siehe dokument.js)."""
    teile = []
    s = stand["schritte"]
    t = s["tagging"]
    if t["status"] != "uebersprungen":
        teile.append(f"Tagging: {t['fertig']} von {t['geplant']} Dokumenten.")
    a = s["alttexte"]
    if a["status"] != "uebersprungen":
        teile.append(f"Alt-Texte: {a['fertig']} von {a['geplant']} Bildern.")
    q = s["quickinfos"]
    if q["status"] != "uebersprungen":
        teile.append(f"Quickinfos: {q['fertig']} von {q['geplant']} Feldern.")
    fehler = [f for k in SCHRITTE for f in s[k]["fehler"]]
    if fehler:
        teile.append("Hinweise: " + " ".join(fehler))
    return " ".join(teile) or "Nichts zu tun."


def starten_von_aussen(project_id: int, user_id: int, ui_lang: str) -> dict:
    """Kette aus einem Thread starten (Chatbot-Werkzeug komplett_barrierefrei_machen, 22.09.2026): dieselben
    Wachen wie POST /kette (laeuft, Projektstatus, nichts zu tun, Guthaben, Tageslimit), dann _kette auf der
    Hauptschleife. Wirft HTTPException wie der Endpunkt."""
    conn = _d.get_db()
    try:
        project = _projekt(conn, project_id, user_id)
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if project_id in _laeuft or _stand(project).get("laeuft"):
            raise HTTPException(status_code=409, detail="Die Kette läuft bereits")
        if project.get("status") in ("extracting", "processing"):
            raise HTTPException(status_code=409, detail="Das Projekt wird gerade verarbeitet. Bitte warte, bis der Lauf fertig ist.")
        plan = vorschau(conn, project, user_id)
    finally:
        conn.close()
    if plan["nichts_zu_tun"]:
        return {"gestartet": False, "grund": "nichts_zu_tun", "plan": plan}
    if not plan["erlaubt"]:
        fehlend = max(0, plan["gesamt"] - int(plan["verfuegbar"] or 0))
        raise HTTPException(status_code=402, detail=_d.billing.credits_fehlen_detail(
            {"preis": plan["gesamt"], "verfuegbar": plan["verfuegbar"], "fehlend": fehlend}, "Komplett barrierefrei machen"))
    if (plan["alttexte"]["bilder"] or plan["quickinfos"]["felder"]) and _d.tageslimit_wache and user is not None:
        tl = _d.tageslimit_wache(dict(user))
        if tl:
            raise HTTPException(status_code=429, detail=_d.tageslimit_text(tl))
    if _loop is None:
        raise HTTPException(status_code=503, detail="Die Kette kann gerade nicht gestartet werden")
    sprache = project.get("alt_language") or (dict(user).get("language") if user is not None else None) or "de"
    with _start_lock:
        if project_id in _laeuft:
            raise HTTPException(status_code=409, detail="Die Kette läuft bereits")
        _laeuft[project_id] = {"laeuft": True}
    asyncio.run_coroutine_threadsafe(_kette(project_id, user_id, plan, ui_lang or "", sprache), _loop)
    return {"gestartet": True, "plan": plan}


def haengende_ketten_zuruecksetzen() -> None:
    conn = _d.get_db()
    try:
        rows = conn.execute("SELECT id, kette_json FROM projects WHERE kette_json LIKE '%\"laeuft\": true%'").fetchall()
        for r in rows:
            try:
                st = json.loads(r["kette_json"] or "{}")
            except Exception:  # noqa: BLE001
                st = {}
            st["laeuft"] = False
            st["schritt"] = "fehler"
            st["zusammenfassung"] = "Die Kette wurde durch einen Neustart des Servers abgebrochen. Die fertigen Schritte bleiben erhalten."
            conn.execute("UPDATE projects SET kette_json = ? WHERE id = ?", (json.dumps(st, ensure_ascii=False), r["id"]))
        if rows:
            conn.commit()
            log.warning("[kette] %d haengende Ketten zurueckgesetzt", len(rows))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    router = APIRouter()

    @router.on_event("startup")
    async def _startup():
        global _loop
        _loop = asyncio.get_running_loop()
        try:
            haengende_ketten_zuruecksetzen()
        except Exception as e:  # noqa: BLE001
            log.warning("[kette] Start-Reparatur uebersprungen: %r", e)

    @router.get("/api/projects/{project_id}/kette")
    async def lesen(project_id: int, user: dict = Depends(_user())):
        conn = _d.get_db()
        try:
            project = _projekt(conn, project_id, user["id"])
            return vorschau(conn, project, user["id"])
        finally:
            conn.close()

    @router.post("/api/projects/{project_id}/kette")
    async def starten(project_id: int, request: Request, user: dict = Depends(_user())):
        conn = _d.get_db()
        try:
            project = _projekt(conn, project_id, user["id"])
            if project_id in _laeuft or _stand(project).get("laeuft"):
                raise HTTPException(status_code=409, detail="Die Kette läuft bereits")
            if project.get("status") in ("extracting", "processing"):
                raise HTTPException(status_code=409, detail="Das Projekt wird gerade verarbeitet. Bitte warte, bis der Lauf fertig ist.")
            plan = vorschau(conn, project, user["id"])
        finally:
            conn.close()
        if plan["nichts_zu_tun"]:
            return {"gestartet": False, "grund": "nichts_zu_tun", "plan": plan}
        if not plan["erlaubt"]:
            fehlend = max(0, plan["gesamt"] - int(plan["verfuegbar"] or 0))
            raise HTTPException(status_code=402, detail=_d.billing.credits_fehlen_detail(
                {"preis": plan["gesamt"], "verfuegbar": plan["verfuegbar"], "fehlend": fehlend}, "Komplett barrierefrei machen"))
        if (plan["alttexte"]["bilder"] or plan["quickinfos"]["felder"]) and _d.tageslimit_wache:
            tl = _d.tageslimit_wache(user)
            if tl:
                raise HTTPException(status_code=429, detail=_d.tageslimit_text(tl))
        ui_lang = _d.resolve_ui_language(request) if _d.resolve_ui_language else ""
        sprache = project.get("alt_language") or user.get("language") or "de"
        with _start_lock:
            if project_id in _laeuft:
                raise HTTPException(status_code=409, detail="Die Kette läuft bereits")
            _laeuft[project_id] = {"laeuft": True}
        asyncio.create_task(_kette(project_id, user["id"], plan, ui_lang, sprache))
        return {"gestartet": True, "plan": plan}

    return router
