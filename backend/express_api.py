"""EXPRESS-SERVICE Stufe 1 — Endpunkte und Seiten (05.10.2026). Kern: express.py, Doku: docs/EXPRESS_SERVICE.md.

Kunde (angemeldet):
  GET  /express                                   Seite: neuer Auftrag (Warenkorb) + Meine Auftraege
  GET  /express/auftrag/{id}                      Seite: Auftragsuebersicht (Nachweis, druckbar)
  GET  /express/warenkorb                         Seite: dieselbe, geoeffnet bei „Deine Auswahl“ (Navigation, Zusatz 05.10.)
  GET  /express/bedingungen                       Seite: Bedingungen (ENTWURF)
  GET  /api/express/stand                         Warenkorb, Leistungen und Preise, Dateitypen, Frist, Guthaben
  GET  /api/express/projekte                      eigene Projekte mit Dokumenten eines angebotenen Dateityps
  GET  /api/express/projekte/{pid}/dokumente      Dokumente eines eigenen Projekts mit Seitenzahl
  POST /api/express/warenkorb/dokumente           {document_ids}
  POST /api/express/warenkorb/hochladen           Datei ohne Projekt (legt „Express-Auftrag <Nr>“ an)
  POST /api/express/warenkorb/positionen/{id}/leistung   {leistung}
  DELETE /api/express/warenkorb/positionen/{id}
  POST /api/express/bestellen                     {ansprechpartner, telefon, hinweise, bedingungen, idempotenz,
                                                   korb_id, erwartete_credits, fassung}
  GET  /api/express/auftraege, /api/express/auftraege/{id}
  POST /api/express/auftraege/{id}/antwort        {text}
  POST /api/express/auftraege/{id}/name           {name} eigener Name (leer = „Auftrag <Nr>“)
  DELETE /api/express/auftraege/{id}              nur geliefert/storniert; intern bleibt ein Buchungsnachweis
  GET  /api/express/auftraege/{id}/positionen/{pos}/(ergebnis|bericht)
  GET  /api/express/auftraege/{id}/nachweis.pdf   Auftragsuebersicht als PDF/UA (LibreOffice-Umwandler)
Verwaltung (Admins lesen; Voll-Admins und Express-Bearbeiter arbeiten; Einstellungen und Bearbeiter nur Voll-Admins):
  GET  /verwaltung/express, /verwaltung/express/{id}
  GET  /api/admin/express/auftraege, /api/admin/express/auftraege/{id}
  POST …/{id}/uebernehmen | rueckfrage {text} | notiz {text} | liefern {trotz_befunden} | stornieren {grund}
  POST …/{id}/positionen/{pos}/(ergebnis|bericht)   Datei hochladen (Ergebnis: automatische Pruefung je Dateityp)
  GET  …/{id}/positionen/{pos}/(original|ergebnis|bericht), …/{id}/originale.zip
  GET/POST /api/admin/express/einstellungen; GET/POST/DELETE /api/admin/express/bearbeiter

Schalter (funktionen.EXPRESS; Pruefung Entwicklung 05.10.2026, Befund 9 — die sichere Variante):
  - NEU BESTELLEN (Warenkorb, Projekte, Hochladen, Bestellen) nur, wenn der Schalter an ist.
  - Alles andere — Auftragsseiten der Kunden, Downloads, Antwort auf Rueckfragen, die ganze Verwaltung — bleibt
    erreichbar, sobald es einen bestellten Auftrag gibt, auch bei ausgeschaltetem Schalter. So lassen sich offene
    Auftraege nach dem Abschalten weiter liefern oder stornieren, die Vormerkung bleibt nicht haengen, Kunden kommen an
    ihre Ergebnisse, und die Erinnerungen laufen weiter. Solange es nie einen Auftrag gab (Prod, Demo), antwortet
    alles mit 404 wie bisher.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import glob
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.parse
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.background import BackgroundTask

import billing
import express
import funktionen
import umsatz

log = logging.getLogger("express_api")


@dataclass
class Deps:
    get_current_user: Callable          # (request) -> {"id", "email", "is_admin"}
    get_user_by_id: Callable            # (id) -> dict | None
    require_full_admin: Callable        # (request) -> user
    send_email: Callable                # (to, subject, html, bcc_admin=…) -> bool
    base_url: str
    notification_email: str
    results_dir: str
    upload_dir: str
    max_upload_size: int
    upload_uebernehmen: Callable        # async (dateityp, file_path, filename, user, project_id) -> {"project_id", "document_id", …}
    upload_vorpruefung: Callable        # (dateityp, file_path, gettext) -> Grund | None
    get_gettext: Callable               # (lang) -> _
    resolve_ui_language: Callable       # (request) -> lang
    render_seite: Callable              # (request, template, **extra) -> Response
    verwaltung_seite: Callable          # (request, template, bereich, **extra) -> Response
    absender_kennung: Callable          # (request) -> str


_d: Optional[Deps] = None
_bremse: dict = defaultdict(list)       # (art, user_id) -> [Zeitpunkte]
_BREMSEN = {"bestellen": (10, 3600), "hochladen": (30, 3600), "antwort": (20, 3600), "nachweis": (30, 3600),
            "zip": (20, 3600)}
# Befund 16: Der Nachweis wartet auf den Umwandler-Dienst — in einem EIGENEN, kleinen Pool und mit kurzem Zeitlimit,
# damit ein haengender Umwandler nicht die Threads belegt, die alle anderen Anfragen brauchen.
_nachweis_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="express_nachweis")
NACHWEIS_TIMEOUT = 60
ZIP_PREFIX = "express_zip_"
ZIP_PLATZ_RESERVE = 512 * 1024 * 1024   # so viel muss nach dem ZIP auf der Platte frei bleiben
# Befund 10: Eine Meldung, die an niemanden rausging, wird so oft erneut versucht (alle 10 Minuten), dann aufgegeben;
# der Zaehler steht in der Datenbank (express.meldung_freigeben).
MELDUNG_VERSUCHE = express.MELDUNG_VERSUCHE


def _bremsen(art: str, user_id: int) -> None:
    anzahl, fenster = _BREMSEN[art]
    jetzt = time.time()
    liste = [t for t in _bremse[(art, user_id)] if jetzt - t < fenster]
    if len(liste) >= anzahl:
        _bremse[(art, user_id)] = liste
        raise HTTPException(status_code=429, detail="Zu viele Anfragen in kurzer Zeit. Bitte später erneut versuchen.")
    liste.append(jetzt)
    _bremse[(art, user_id)] = liste


def _fehler(e: express.ExpressFehler):
    detail = {"text": e.text, **e.extra} if e.extra else e.text
    return HTTPException(status_code=e.status, detail=detail)


async def _json_dict(request: Request) -> dict:
    """Befund 8: kaputter oder falsch geformter JSON-Koerper -> 400 statt 500."""
    try:
        daten = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Ungültige Anfrage")
    if not isinstance(daten, dict):
        raise HTTPException(status_code=400, detail="Ungültige Anfrage")
    return daten


def _neu_an() -> None:
    """Neu bestellen: nur bei eingeschaltetem Schalter."""
    funktionen.endpunkt_frei("EXPRESS")


def aktiv() -> bool:
    """Auftragsseiten und Verwaltung: Schalter an ODER es gibt schon bestellte Auftraege (Befund 9)."""
    return bool(funktionen.EXPRESS) or express.gibt_bestellte()


def _an() -> None:
    if not aktiv():
        raise HTTPException(status_code=404, detail="Not Found")


def fuer_oberflaeche() -> dict:
    """Zusatz zu funktionen.fuer_oberflaeche() (window.FUNKTIONEN in app.html): Knopf „In den Express-Warenkorb“ am
    Dokument — nur bei eingeschaltetem Neu-Bestellen UND Einstellung korb_knopf (Verwaltung, ohne Codeaenderung)."""
    try:
        an = bool(funktionen.EXPRESS) and bool(express.einstellungen().get("korb_knopf"))
    except Exception:  # noqa: BLE001
        an = False
    return {"express_korb_knopf": an}


def korb_fuer_me(user_id: int):
    """Fuer /api/me: {"modus", "dokumente"} des Navigations-Eintrags „Express-Warenkorb“ oder None (Eintrag aus)."""
    if not funktionen.EXPRESS:
        return None
    try:
        modus = express.einstellungen().get("korb_navigation") or "immer"
        if modus == "aus":
            return None
        return {"modus": modus, "dokumente": express.korb_kurz(user_id)["dokumente"]}
    except Exception:  # noqa: BLE001
        log.exception("Express-Warenkorb fuer /api/me nicht lesbar (Konto %s)", user_id)
        return None


def _person(user_id: int) -> dict:
    u = _d.get_user_by_id(user_id) or {}
    return {"id": int(user_id), "name": (u.get("display_name") or "").strip() or u.get("email") or "Unbekannt"}


def _kunde_neu(request: Request) -> dict:
    _neu_an()
    return _d.get_current_user(request)


def _kunde(request: Request) -> dict:
    _an()
    return _d.get_current_user(request)


def _leser(request: Request) -> dict:
    """Verwaltung lesen: jeder Admin (frisch aus der Datenbank) und Express-Bearbeiter."""
    _an()
    user = _d.get_current_user(request)
    if user.get("is_admin") or express.ist_bearbeiter(user["id"]):
        return user
    raise HTTPException(status_code=403, detail="Nur für Administratoren und Express-Bearbeiter")


def _bearbeiter(request: Request) -> dict:
    """Verwaltung arbeiten: Voll-Admins und Express-Bearbeiter (Nur-Einsicht-Admins lesen nur)."""
    _an()
    user = _d.get_current_user(request)
    db = _d.get_user_by_id(user["id"]) or {}
    if (db.get("is_admin") and db.get("admin_level") == "full") or express.ist_bearbeiter(user["id"]):
        return user
    raise HTTPException(status_code=403, detail="Nur für Voll-Administratoren und Express-Bearbeiter")


def _voll(request: Request) -> dict:
    _an()
    return _d.require_full_admin(request)


def _ohne_cache(antwort):
    """Befund 4: Seiten mit Bestell-Zustand nicht aus dem Zurueck-Speicher des Browsers (bfcache) zeigen — sonst
    bestellt eine alte Seite mit altem Stand. Die Seite erneuert ihren Stand zusaetzlich bei pageshow."""
    try:
        antwort.headers["Cache-Control"] = "no-store"
    except AttributeError:
        pass
    return antwort


def _disposition(name: str) -> str:
    """attachment mit ASCII-Ersatz und UTF-8-Namen (RFC 6266/5987)."""
    ascii_name = "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in name)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{urllib.parse.quote(name)}"


def _datei(pfad: str, name: str) -> FileResponse:
    return FileResponse(pfad, media_type=express.mime_der_datei(pfad),
                        headers={"Content-Disposition": _disposition(name), "Cache-Control": "no-store",
                                 "X-Content-Type-Options": "nosniff"})


async def _im_thread(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kwargs))


def _mail_senden(an: str, betreff_html: tuple) -> bool:
    """True, wenn die Mail rausging. main.send_email meldet Fehler mit False (und wirft nicht)."""
    betreff, inhalt = betreff_html
    try:
        return _d.send_email(an, betreff, inhalt, bcc_admin=False) is not False
    except Exception:  # noqa: BLE001 — eine Mail darf keinen Auftrag scheitern lassen
        log.exception("Express-Mail nicht versandt (%s)", betreff)
        return False


def _mails_nach(art_kunde: str = "", art_team: str = "", auftrag_id: int = 0, text: str = "") -> int:
    """Mails NACH einem erfolgreichen Schritt (im Thread aufrufen). Rueckgabe: Zahl der Team-Mails, die rausgingen
    (fuer die Wiederholung der Erinnerungen, Befund 10); -1 = Auftrag nicht lesbar."""
    try:
        a = express.auftrag_fuer_verwaltung(auftrag_id)
    except Exception:  # noqa: BLE001
        log.exception("Express-Mail: Auftrag %s nicht lesbar", auftrag_id)
        return -1
    if art_kunde and a.get("kunde_email"):
        _mail_senden(a["kunde_email"], express.mail_kunde(art_kunde, a, _d.base_url, text))
    erfolge = 0
    if art_team:
        inhalt = express.mail_team(art_team, a, _d.base_url, text)
        for adresse in express.team_empfaenger(_d.notification_email):
            erfolge += 1 if _mail_senden(adresse, inhalt) else 0
    return erfolge


def _hintergrund(fn, *args, **kwargs) -> None:
    """Mails nicht auf der Antwort warten lassen."""
    asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kwargs))


def _meldung_senden(art: str, auftrag_id: int) -> None:
    """Erinnerung/Ueberfaellig an das Team. Ging sie an niemanden raus, wird sie freigegeben und beim naechsten
    Durchlauf erneut versucht — hoechstens MELDUNG_VERSUCHE-mal (Befund 10). Doppelmails gibt es nicht: freigegeben wird
    nur, wenn KEINE Mail rausging."""
    erfolge = _mails_nach("", art, auftrag_id)
    if erfolge != 0:
        if erfolge > 0:
            express.meldung_erfolg(auftrag_id)
        return
    if express.meldung_freigeben(art, auftrag_id):
        log.warning("Express-%s fuer Auftrag %s ging an niemanden raus — neuer Versuch im naechsten Durchlauf", art, auftrag_id)
    else:
        log.error("Express-%s fuer Auftrag %s nach %d Versuchen aufgegeben", art, auftrag_id, MELDUNG_VERSUCHE)


def _zip_reste_wegraeumen(alter_s: int = 3600) -> None:
    """ZIPs, deren Download abgebrochen wurde (BackgroundTask lief nicht), nach einer Stunde loeschen."""
    grenze = time.time() - alter_s
    for pfad in glob.glob(os.path.join(tempfile.gettempdir(), ZIP_PREFIX + "*.zip")):
        try:
            if os.path.getmtime(pfad) < grenze:
                os.remove(pfad)
        except OSError:
            pass


async def erinnerungs_schleife():
    """Alle 10 Minuten: Erinnerung 12 Stunden vor der Frist und Meldung bei Ueberfaelligkeit (je einmal, bei
    Versandfehler erneut), Aufbewahrungsfrist der Dateien (express.aufraeumen), Reste abgebrochener ZIP-Downloads.
    Gestartet aus main.lifespan (ein APIRouter-„startup“ laeuft neben einem eigenen lifespan nicht) — IMMER, auch bei
    ausgeschaltetem Schalter (Befund 9); ohne Auftraege tut sie nichts."""
    while True:
        try:
            for art, auftrag_id in await _im_thread(express.faellige_meldungen):
                await _im_thread(_meldung_senden, art, auftrag_id)
            await _im_thread(express.aufraeumen)
            await _im_thread(_zip_reste_wegraeumen)
        except Exception:  # noqa: BLE001
            log.exception("Express-Erinnerungen: Durchlauf fehlgeschlagen")
        await asyncio.sleep(600)


# ─── Kontoloeschung (Befund 6) ───────────────────────────────────────────

def vor_kontoloeschung(user_id: int) -> dict:
    """VOR database.delete_user_data aufrufen (main): welche offenen Auftraege betroffen sind."""
    try:
        return express.vor_kontoloeschung(user_id)
    except Exception:  # noqa: BLE001
        log.exception("Express: Auftraege vor dem Loeschen von Konto %s nicht lesbar", user_id)
        return {"topf": [], "eigene": []}


def nach_kontoloeschung(info: dict) -> None:
    """NACH dem Loeschen: Storno-Mails (Kunde und Team) fuer die stornierten Topf-Auftraege, Team-Hinweis fuer die
    entfallenen eigenen Auftraege. Im Hintergrund; aus einem async-Endpunkt aufrufen."""
    if _d is None or not info or not (info.get("topf") or info.get("eigene")):
        return

    def senden():
        for auftrag_id in info.get("topf") or []:
            _mails_nach("storniert", "storniert", auftrag_id, express.STORNO_TOPF_GRUND)
        for a in info.get("eigene") or []:
            inhalt = express.mail_team("entfallen", a, _d.base_url)
            for adresse in express.team_empfaenger(_d.notification_email):
                _mail_senden(adresse, inhalt)
    _hintergrund(senden)


def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    express.RESULTS_DIR = deps.results_dir
    router = APIRouter()

    # ─── Seiten ───
    @router.get("/express", response_class=HTMLResponse)
    async def seite_express(request: Request):
        _an()
        return _ohne_cache(_d.render_seite(request, "express.html", express_bestellen=bool(funktionen.EXPRESS)))

    @router.get("/express/warenkorb", response_class=HTMLResponse)
    async def seite_warenkorb(request: Request):
        """Ziel des Navigations-Eintrags „Express-Warenkorb“ und des Links „Zum Warenkorb“: dieselbe Seite wie /express,
        geoeffnet bei „2. Deine Auswahl“ (eigene Adresse, damit die Navigation sie als aktuelle Seite markieren kann)."""
        _neu_an()
        return _ohne_cache(_d.render_seite(request, "express.html", express_bestellen=True, warenkorb_ansicht=True))

    @router.get("/express/auftrag/{auftrag_id}", response_class=HTMLResponse)
    async def seite_auftrag(auftrag_id: int, request: Request):
        _an()
        return _ohne_cache(_d.render_seite(request, "express_auftrag.html", auftrag_id=int(auftrag_id)))

    @router.get("/express/bedingungen", response_class=HTMLResponse)
    async def seite_bedingungen(request: Request):
        _an()
        return _d.render_seite(request, "express_bedingungen.html", zustimmung_fassung=express.ZUSTIMMUNG_FASSUNG,
                               express_typen=express.typen_text())

    @router.get("/verwaltung/express", response_class=HTMLResponse)
    async def seite_verwaltung(request: Request):
        _an()
        return _ohne_cache(_d.verwaltung_seite(request, "verwaltung_express.html", "express"))

    @router.get("/verwaltung/express/{auftrag_id}", response_class=HTMLResponse)
    async def seite_verwaltung_auftrag(auftrag_id: int, request: Request):
        _an()
        return _ohne_cache(_d.verwaltung_seite(request, "verwaltung_express_auftrag.html", "express",
                                               auftrag_id=int(auftrag_id)))

    # ─── Kunde: Warenkorb (nur bei eingeschaltetem Schalter) ───
    @router.get("/api/express/stand")
    async def stand(request: Request, user: dict = Depends(_kunde_neu)):
        e = await _im_thread(express.einstellungen)
        korb = await _im_thread(express.warenkorb, user["id"])
        verf = await _im_thread(billing.verfuegbare_credits, user["id"])
        db = _d.get_user_by_id(user["id"]) or {}
        _ = _d.get_gettext(_d.resolve_ui_language(request))
        typen = express.angebotene_dateitypen()
        return {"warenkorb": korb, "preise": dict(e["preise"]), "leistungen": express.leistungen_liste(e),
                "dateitypen": [{"schluessel": t.schluessel, "name": t.name, "accept": t.accept} for t in typen],
                "frist_stunden": e["frist_stunden"], "max_seiten": e["max_seiten_auftrag"],
                "max_dokumente": e["max_dokumente_auftrag"], "guthaben": verf, "max_upload_mb": _d.max_upload_size // (1024 * 1024),
                "ansprechpartner": (db.get("display_name") or "").strip(),
                "texte": {"bedingungen": _(express.TEXT_BEDINGUNGEN)}}

    @router.get("/api/express/projekte")
    async def projekte(user: dict = Depends(_kunde_neu)):
        """Nur Projekte, deren Typ der Express-Service bearbeiten kann (heute PDF), mit mindestens einem Dokument
        (Pruefung Barrierefreiheit 05.10.2026, Befund 15). angelegt_am: damit die Seite gleichnamige unterscheidet."""
        projekt_typen = sorted({pt for t in express.angebotene_dateitypen() for pt in t.projekt_typen})

        def lesen():
            from database import get_db
            conn = get_db()
            try:
                platz = ",".join("?" * len(projekt_typen))
                return [dict(r) for r in conn.execute(
                    "SELECT p.id, COALESCE(NULLIF(TRIM(p.name), ''), p.filename) AS name, p.created_at AS angelegt_am, "
                    "(SELECT COUNT(*) FROM documents d WHERE d.project_id = p.id) AS dokumente "
                    f"FROM projects p WHERE p.user_id = ? AND COALESCE(p.project_type, 'pdf') IN ({platz}) "
                    "AND EXISTS (SELECT 1 FROM documents d WHERE d.project_id = p.id) "
                    "ORDER BY p.updated_at DESC, p.id DESC LIMIT 500", (user["id"], *projekt_typen))]
            finally:
                conn.close()
        liste = await _im_thread(lesen) if projekt_typen else []
        for p in liste:
            p["angelegt_am"] = umsatz.lokal(p["angelegt_am"]) if p.get("angelegt_am") else ""
        return {"projekte": liste}

    @router.get("/api/express/projekte/{project_id}/dokumente")
    async def projekt_dokumente(project_id: int, user: dict = Depends(_kunde_neu)):
        def lesen():
            from database import get_db
            conn = get_db()
            try:
                if not conn.execute("SELECT 1 FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])).fetchone():
                    raise express.NichtGefunden("Projekt nicht gefunden")
                docs = [dict(r) for r in conn.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index, id",
                                                      (project_id,))]
            finally:
                conn.close()
            korb = {p["document_id"] for p in express.warenkorb(user["id"])["positionen"]}
            out = []
            for d in docs:
                quelle = express._quelle_des_dokuments(d)
                typ = express.dateityp_der_datei(quelle) if quelle else None
                eintrag = {"id": d["id"], "name": express._name_des_dokuments(d), "im_warenkorb": d["id"] in korb,
                           "seiten": None, "grund": "", "dateityp": typ.schluessel if typ else ""}
                if not typ:
                    # Code statt Text: die Seite sagt es in der Sprache des Kunden (mit den moeglichen Typen).
                    eintrag["grund"] = "dateityp"
                else:
                    try:
                        eintrag["seiten"] = typ.seiten(quelle)
                    except express.ExpressFehler as fe:
                        eintrag["grund"] = fe.text
                out.append(eintrag)
            return out
        try:
            return {"dokumente": await _im_thread(lesen), "moeglich": express.typen_text()}
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/warenkorb/dokumente")
    async def korb_dokumente(request: Request, user: dict = Depends(_kunde_neu)):
        daten = await _json_dict(request)
        try:
            return await _im_thread(express.dokumente_hinzufuegen, user["id"], daten.get("document_ids"))
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/warenkorb/hochladen")
    async def korb_hochladen(request: Request, file: UploadFile = File(...), user: dict = Depends(_kunde_neu)):
        _bremsen("hochladen", user["id"])
        name = os.path.basename(file.filename or "dokument")
        inhalt = await file.read(_d.max_upload_size + 1)
        if len(inhalt) > _d.max_upload_size:
            raise HTTPException(status_code=413, detail=f"Datei zu groß. Maximum: {_d.max_upload_size // (1024 * 1024)} MB")
        try:
            typ = express.dateityp_fuer_upload(name, inhalt[:64])
        except express.ExpressFehler as e:
            raise _fehler(e)
        ordner = os.path.join(_d.upload_dir, str(int(user["id"])))
        os.makedirs(ordner, exist_ok=True)
        stamm = "".join(c for c in os.path.splitext(name)[0] if c.isalnum() or c in " ._-")[:80].strip() or "dokument"
        pfad = os.path.join(ordner, f"{time.strftime('%Y%m%d_%H%M%S')}_express_{stamm}{typ.endung}")
        with open(pfad, "wb") as f:
            f.write(inhalt)
        grund = await _im_thread(_d.upload_vorpruefung, typ.schluessel, pfad, _d.get_gettext(_d.resolve_ui_language(request)))
        if grund:
            os.unlink(pfad)
            raise HTTPException(status_code=400, detail=grund)
        projekt = await _im_thread(express.auto_projekt, user["id"], typ)
        try:
            erg = await _d.upload_uebernehmen(typ.schluessel, pfad, name, user, projekt)
        except HTTPException:
            if projekt is None:
                raise
            # Das gemerkte Projekt taugt nicht mehr (z. B. gerade in Arbeit): ein neues anlegen.
            erg = await _d.upload_uebernehmen(typ.schluessel, pfad, name, user, None)
            projekt = None
        if projekt is None:
            await _im_thread(express.auto_projekt_merken, user["id"], int(erg["project_id"]))
        try:
            return await _im_thread(express.dokumente_hinzufuegen, user["id"], [int(erg["document_id"])])
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/warenkorb/positionen/{pos_id}/leistung")
    async def korb_leistung(pos_id: int, request: Request, user: dict = Depends(_kunde_neu)):
        daten = await _json_dict(request)
        try:
            return await _im_thread(express.leistung_setzen, user["id"], pos_id, str(daten.get("leistung") or ""))
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.delete("/api/express/warenkorb/positionen/{pos_id}")
    async def korb_entfernen(pos_id: int, user: dict = Depends(_kunde_neu)):
        try:
            return await _im_thread(express.position_entfernen, user["id"], pos_id)
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/bestellen")
    async def bestellen(request: Request, user: dict = Depends(_kunde_neu)):
        daten = await _json_dict(request)
        _bremsen("bestellen", user["id"])
        sprache = _d.resolve_ui_language(request)
        _ = _d.get_gettext(sprache)
        try:
            erg = await _im_thread(
                express.bestellen, user["id"], ansprechpartner=daten.get("ansprechpartner"), telefon=daten.get("telefon"),
                hinweise=daten.get("hinweise"), bedingungen=daten.get("bedingungen"),
                idempotenz=daten.get("idempotenz"), korb_id=daten.get("korb_id"),
                erwartete_credits=daten.get("erwartete_credits"), fassung=daten.get("fassung"), sprache=sprache,
                texte={"bedingungen": _(express.TEXT_BEDINGUNGEN)},
                absender=_d.absender_kennung(request))
        except express.ExpressFehler as e:
            raise _fehler(e)
        if erg["neu"]:
            log.info("Express-Auftrag %s bestellt (Konto %s)", erg["auftrag_id"], user["id"])
            _hintergrund(_mails_nach, "bestellt", "bestellt", erg["auftrag_id"])
        return {"ok": True, "auftrag_id": erg["auftrag_id"], "schon_bestellt": not erg["neu"]}

    # ─── Kunde: Auftraege (auch bei ausgeschaltetem Schalter, siehe oben) ───
    @router.get("/api/express/auftraege")
    async def meine_auftraege(user: dict = Depends(_kunde)):
        return {"auftraege": await _im_thread(express.auftraege_des_kunden, user["id"]),
                "bestellen_moeglich": bool(funktionen.EXPRESS)}

    @router.get("/api/express/auftraege/{auftrag_id}")
    async def mein_auftrag(auftrag_id: int, user: dict = Depends(_kunde)):
        try:
            return await _im_thread(express.auftrag_fuer_kunde, user["id"], auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/auftraege/{auftrag_id}/antwort")
    async def antwort(auftrag_id: int, request: Request, user: dict = Depends(_kunde)):
        daten = await _json_dict(request)
        _bremsen("antwort", user["id"])
        text = str(daten.get("text") or "")
        try:
            a = await _im_thread(express.antwort_kunde, user["id"], auftrag_id, text)
        except express.ExpressFehler as e:
            raise _fehler(e)
        _hintergrund(_mails_nach, "", "antwort", auftrag_id, text.strip())
        return a

    @router.post("/api/express/auftraege/{auftrag_id}/name")
    async def umbenennen(auftrag_id: int, request: Request, user: dict = Depends(_kunde)):
        """Eigener Name des Kunden fuer den Auftrag (leer = „Auftrag <Nr>“)."""
        daten = await _json_dict(request)
        try:
            return await _im_thread(express.umbenennen, user["id"], auftrag_id, daten.get("name"))
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.delete("/api/express/auftraege/{auftrag_id}")
    async def kunde_loeschen(auftrag_id: int, user: dict = Depends(_kunde)):
        """Nur gelieferte oder stornierte Auftraege; intern bleibt ein Buchungsnachweis (express.kunde_loeschen)."""
        try:
            await _im_thread(express.kunde_loeschen, user["id"], auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)
        log.info("Express-Auftrag %s vom Kunden %s geloescht", auftrag_id, user["id"])
        return {"ok": True}

    @router.get("/api/express/auftraege/{auftrag_id}/positionen/{pos_id}/{art}")
    async def kunde_datei(auftrag_id: int, pos_id: int, art: str, user: dict = Depends(_kunde)):
        try:
            pfad, name = await _im_thread(express.datei_fuer_kunde, user["id"], auftrag_id, pos_id, art)
        except express.ExpressFehler as e:
            raise _fehler(e)
        return _datei(pfad, name)

    @router.get("/api/express/auftraege/{auftrag_id}/nachweis.pdf")
    async def nachweis(auftrag_id: int, user: dict = Depends(_kunde)):
        _bremsen("nachweis", user["id"])
        try:
            a = await _im_thread(express.auftrag_fuer_kunde, user["id"], auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)
        import pdfua_export
        if not pdfua_export.verfuegbar():
            raise HTTPException(status_code=503, detail="Die PDF kann gerade nicht erstellt werden. Bitte die Druckansicht nutzen.")

        def bauen():
            with tempfile.TemporaryDirectory(prefix="express_nachweis_") as tmp:
                docx = os.path.join(tmp, f"Express-Auftrag-{a['id']}.docx")
                # Zeiten stehen in der Kundenansicht in deutscher Zeit (express._lokal); im PDF ausgeschrieben wie auf
                # der Webseite („5. Oktober 2026, 12:04“, Pruefung Barrierefreiheit 05.10.2026, Befund 11).
                express.nachweis_docx(a, docx, express.datum_deutsch)
                pdf, bericht = pdfua_export.konvertiere(docx, os.path.basename(docx), timeout=NACHWEIS_TIMEOUT)
                klar = pdfua_export.klartext(bericht or {})
                if not klar.get("bestanden"):
                    # Nur eine barrierefreie PDF ausliefern (Steve) — sonst bleibt die Druckansicht.
                    log.warning("Express-Nachweis %s: PDF/UA-Pruefung nicht bestanden (%s)", a["id"], klar.get("zusammenfassung"))
                    raise pdfua_export.UmwandlungFehlgeschlagen("PDF/UA nicht bestanden")
                return pdf
        try:
            pdf = await asyncio.get_running_loop().run_in_executor(_nachweis_pool, bauen)
        except pdfua_export.UmwandlungFehlgeschlagen:
            log.exception("Express-Nachweis %s: Umwandlung fehlgeschlagen", auftrag_id)
            raise HTTPException(status_code=503, detail="Die PDF kann gerade nicht erstellt werden. Bitte die Druckansicht nutzen.")
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": _disposition(f"Express-Auftrag {a['id']} – Auftragsübersicht.pdf"),
                                 "Cache-Control": "no-store"})

    # ─── Verwaltung ───
    @router.get("/api/admin/express/auftraege")
    async def v_liste(user: dict = Depends(_leser)):
        db = _d.get_user_by_id(user["id"]) or {}
        return {"gruppen": await _im_thread(express.liste_fuer_verwaltung),
                "darf_arbeiten": bool((db.get("is_admin") and db.get("admin_level") == "full") or express.ist_bearbeiter(user["id"])),
                "darf_einstellen": bool(db.get("is_admin") and db.get("admin_level") == "full"),
                "ist_admin": bool(db.get("is_admin")), "bestellen_moeglich": bool(funktionen.EXPRESS)}

    async def _auftrag_fuer(user: dict, auftrag_id: int) -> dict:
        a = await _im_thread(express.auftrag_fuer_verwaltung, auftrag_id)
        db = _d.get_user_by_id(user["id"]) or {}
        a["darf_arbeiten"] = bool((db.get("is_admin") and db.get("admin_level") == "full") or express.ist_bearbeiter(user["id"]))
        a["ist_admin"] = bool(db.get("is_admin"))
        a["lieferbar"] = await _im_thread(express.lieferbar, a)
        if not a["ist_admin"]:
            # Bearbeiter (z. B. Partner) brauchen keinen Konto-Bezug: keine Kunden-ID fuer Verwaltungs-Links.
            a.pop("user_id", None)
            a.pop("konto_user_id", None)
        return a

    @router.get("/api/admin/express/auftraege/{auftrag_id}")
    async def v_auftrag(auftrag_id: int, user: dict = Depends(_leser)):
        try:
            return await _auftrag_fuer(user, auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)

    async def _schritt(user, fn, auftrag_id, *args, mail_kunde="", mail_team="", text=""):
        try:
            await _im_thread(fn, auftrag_id, *args)
        except express.ExpressFehler as e:
            raise _fehler(e)
        if mail_kunde or mail_team:
            _hintergrund(_mails_nach, mail_kunde, mail_team, auftrag_id, text)
        return await _auftrag_fuer(user, auftrag_id)

    @router.post("/api/admin/express/auftraege/{auftrag_id}/uebernehmen")
    async def v_uebernehmen(auftrag_id: int, user: dict = Depends(_bearbeiter)):
        return await _schritt(user, express.uebernehmen, auftrag_id, _person(user["id"]))

    @router.post("/api/admin/express/auftraege/{auftrag_id}/rueckfrage")
    async def v_rueckfrage(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        text = str((await _json_dict(request)).get("text") or "")
        return await _schritt(user, express.rueckfrage, auftrag_id, _person(user["id"]), text, mail_kunde="rueckfrage",
                              text=text.strip())

    @router.post("/api/admin/express/auftraege/{auftrag_id}/notiz")
    async def v_notiz(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        text = str((await _json_dict(request)).get("text") or "")
        return await _schritt(user, express.notiz_setzen, auftrag_id, _person(user["id"]), text)

    @router.post("/api/admin/express/auftraege/{auftrag_id}/liefern")
    async def v_liefern(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        trotz = (await _json_dict(request)).get("trotz_befunden") is True
        a = await _schritt(user, express.liefern, auftrag_id, _person(user["id"]), trotz, mail_kunde="geliefert")
        log.info("Express-Auftrag %s geliefert von Konto %s", auftrag_id, user["id"])
        return a

    @router.post("/api/admin/express/auftraege/{auftrag_id}/stornieren")
    async def v_stornieren(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        grund = str((await _json_dict(request)).get("grund") or "")
        a = await _schritt(user, express.stornieren, auftrag_id, _person(user["id"]), grund, mail_kunde="storniert",
                           text=grund.strip())
        log.info("Express-Auftrag %s storniert von Konto %s", auftrag_id, user["id"])
        return a

    @router.post("/api/admin/express/auftraege/{auftrag_id}/positionen/{pos_id}/{art}")
    async def v_hochladen(auftrag_id: int, pos_id: int, art: str, file: UploadFile = File(...),
                          user: dict = Depends(_bearbeiter)):
        if art not in ("ergebnis", "bericht"):
            raise HTTPException(status_code=404, detail="Nicht gefunden")
        inhalt = await file.read(express.MAX_ERGEBNIS_BYTES + 1)
        try:
            erg = await _im_thread(express.datei_speichern, auftrag_id, pos_id, art, inhalt, file.filename, _person(user["id"]))
        except express.ExpressFehler as e:
            raise _fehler(e)
        if erg["pruef_kennung"]:
            # Automatische Pruefung des Ergebnisses je Dateityp (PDF: veraPDF) als Information, nicht als Sperre: beim
            # Liefern fragt die Oberflaeche bei Abweichungen einmal nach.
            await _im_thread(express.ergebnis_pruefen, pos_id, erg["pfad"], erg["dateityp"], erg["pruef_kennung"])
        return await _auftrag_fuer(user, auftrag_id)

    @router.get("/api/admin/express/auftraege/{auftrag_id}/positionen/{pos_id}/{art}")
    async def v_datei(auftrag_id: int, pos_id: int, art: str, user: dict = Depends(_leser)):
        try:
            pfad, name = await _im_thread(express.datei_fuer_verwaltung, auftrag_id, pos_id, art)
        except express.ExpressFehler as e:
            raise _fehler(e)
        return _datei(pfad, name)

    @router.get("/api/admin/express/auftraege/{auftrag_id}/originale.zip")
    async def v_zip(auftrag_id: int, user: dict = Depends(_leser)):
        _bremsen("zip", user["id"])
        try:
            dateien = await _im_thread(express.originale_fuer_zip, auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)
        if not dateien:
            raise HTTPException(status_code=404, detail="Keine Originale vorhanden")

        def packen():
            # Befund 16: PDFs sind schon komprimiert — ZIP_STORED spart Rechenzeit; vorher Platz pruefen und Reste
            # abgebrochener Downloads wegraeumen.
            _zip_reste_wegraeumen()
            groesse = sum(os.path.getsize(q) for q, _ in dateien if os.path.isfile(q))
            if shutil.disk_usage(tempfile.gettempdir()).free - groesse < ZIP_PLATZ_RESERVE:
                raise express.ExpressFehler("Gerade ist zu wenig Speicherplatz für das ZIP frei. Bitte die Originale "
                                            "einzeln herunterladen.", 507)
            fd, pfad = tempfile.mkstemp(prefix=ZIP_PREFIX, suffix=".zip")
            os.close(fd)
            try:
                with zipfile.ZipFile(pfad, "w", zipfile.ZIP_STORED) as z:
                    for quelle, name in dateien:
                        z.write(quelle, arcname=name)
            except BaseException:
                os.remove(pfad)
                raise
            return pfad
        try:
            pfad = await _im_thread(packen)
        except express.ExpressFehler as e:
            raise _fehler(e)
        return FileResponse(pfad, media_type="application/zip", background=BackgroundTask(os.remove, pfad),
                            headers={"Content-Disposition": _disposition(f"Express-Auftrag {int(auftrag_id)} – Originale.zip"),
                                     "Cache-Control": "no-store"})

    @router.get("/api/admin/express/einstellungen")
    async def v_einstellungen(user: dict = Depends(_leser)):
        e = await _im_thread(express.einstellungen)
        return {**e, "leistungen": express.leistungen_liste(e)}

    @router.post("/api/admin/express/einstellungen")
    async def v_einstellungen_setzen(request: Request, user: dict = Depends(_voll)):
        daten = await _json_dict(request)
        try:
            e = await _im_thread(express.speichere_einstellungen, daten)
        except express.ExpressFehler as fe:
            raise _fehler(fe)
        log.info("Express-Einstellungen geaendert von %s: %s", user.get("email"), json.dumps(e, ensure_ascii=False))
        return {"ok": True, "einstellungen": {**e, "leistungen": express.leistungen_liste(e)},
                "message": "Einstellungen gespeichert."}

    @router.get("/api/admin/express/bearbeiter")
    async def v_bearbeiter(user: dict = Depends(_voll)):
        return {"bearbeiter": await _im_thread(express.bearbeiter_liste)}

    @router.post("/api/admin/express/bearbeiter")
    async def v_bearbeiter_dazu(request: Request, user: dict = Depends(_voll)):
        mail = str((await _json_dict(request)).get("email") or "").strip()
        if not mail:
            raise HTTPException(status_code=400, detail="Bitte die E-Mail-Adresse eines bestehenden Kontos angeben.")
        try:
            u = await _im_thread(express.bearbeiter_setzen, email=mail, an=True)
        except express.ExpressFehler as e:
            raise _fehler(e)
        log.info("Express-Bearbeiter gesetzt von %s: Konto %s", user.get("email"), u["id"])
        return {"ok": True, "bearbeiter": await _im_thread(express.bearbeiter_liste),
                "message": f"{u['display_name'] or u['email']} ist jetzt Express-Bearbeiter."}

    @router.delete("/api/admin/express/bearbeiter/{user_id}")
    async def v_bearbeiter_weg(user_id: int, user: dict = Depends(_voll)):
        try:
            u = await _im_thread(express.bearbeiter_setzen, user_id=user_id, an=False)
        except express.ExpressFehler as e:
            raise _fehler(e)
        return {"ok": True, "bearbeiter": await _im_thread(express.bearbeiter_liste),
                "message": f"{u['display_name'] or u['email']} ist kein Express-Bearbeiter mehr."}

    return router
