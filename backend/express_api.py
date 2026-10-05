"""EXPRESS-SERVICE Stufe 1 — Endpunkte und Seiten (05.10.2026). Kern: express.py, Doku: docs/EXPRESS_SERVICE.md.

Kunde (angemeldet):
  GET  /express                                   Seite: neuer Auftrag (Warenkorb) + Meine Auftraege
  GET  /express/auftrag/{id}                      Seite: Auftragsuebersicht (Nachweis, druckbar)
  GET  /express/bedingungen                       Seite: Bedingungen (ENTWURF)
  GET  /api/express/stand                         Warenkorb, Preise, Frist, Guthaben, Vorbelegung
  GET  /api/express/projekte                      eigene Projekte (Auswahl Schritt 1)
  GET  /api/express/projekte/{pid}/dokumente      PDFs eines eigenen Projekts mit Seitenzahl
  POST /api/express/warenkorb/dokumente           {document_ids}
  POST /api/express/warenkorb/hochladen           PDF ohne Projekt (legt „Express-Auftrag <Nr>“ an)
  POST /api/express/warenkorb/positionen/{id}/leistung   {leistung}
  DELETE /api/express/warenkorb/positionen/{id}
  POST /api/express/bestellen                     {ansprechpartner, telefon, hinweise, bedingungen, bearbeitung, idempotenz}
  GET  /api/express/auftraege, /api/express/auftraege/{id}
  POST /api/express/auftraege/{id}/antwort        {text}
  GET  /api/express/auftraege/{id}/positionen/{pos}/(ergebnis|bericht)
  GET  /api/express/auftraege/{id}/nachweis.pdf   Auftragsuebersicht als PDF/UA (LibreOffice-Umwandler)
Verwaltung (Admins lesen; Voll-Admins und Express-Bearbeiter arbeiten; Einstellungen und Bearbeiter nur Voll-Admins):
  GET  /verwaltung/express, /verwaltung/express/{id}
  GET  /api/admin/express/auftraege, /api/admin/express/auftraege/{id}
  POST …/{id}/uebernehmen | rueckfrage {text} | notiz {text} | liefern {trotz_befunden} | stornieren {grund}
  POST …/{id}/positionen/{pos}/(ergebnis|bericht)   PDF hochladen (Ergebnis: veraPDF laeuft automatisch)
  GET  …/{id}/positionen/{pos}/(original|ergebnis|bericht), …/{id}/originale.zip
  GET/POST /api/admin/express/einstellungen; GET/POST/DELETE /api/admin/express/bearbeiter
Alles antwortet mit 404, solange funktionen.EXPRESS aus ist (Prod, Demo).
"""
from __future__ import annotations

import asyncio
import logging
import os
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
    handle_pdf_upload: Callable         # async (file_path, filename, user, project_id) -> {"project_id", "document_id", …}
    pdf_vorpruefung: Callable           # (file_path, gettext) -> Grund | None
    get_gettext: Callable               # (lang) -> _
    resolve_ui_language: Callable       # (request) -> lang
    render_seite: Callable              # (request, template, **extra) -> Response
    verwaltung_seite: Callable          # (request, template, bereich, **extra) -> Response
    absender_kennung: Callable          # (request) -> str


_d: Optional[Deps] = None
_bremse: dict = defaultdict(list)       # (art, user_id) -> [Zeitpunkte]
_BREMSEN = {"bestellen": (10, 3600), "hochladen": (30, 3600), "antwort": (20, 3600), "nachweis": (30, 3600)}


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


def _an() -> None:
    funktionen.endpunkt_frei("EXPRESS")


def _person(user_id: int) -> dict:
    u = _d.get_user_by_id(user_id) or {}
    return {"id": int(user_id), "name": (u.get("display_name") or "").strip() or u.get("email") or "Unbekannt"}


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


def _disposition(name: str) -> str:
    """attachment mit ASCII-Ersatz und UTF-8-Namen (RFC 6266/5987)."""
    ascii_name = "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in name)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{urllib.parse.quote(name)}"


def _datei(pfad: str, name: str) -> FileResponse:
    return FileResponse(pfad, media_type="application/pdf",
                        headers={"Content-Disposition": _disposition(name), "Cache-Control": "no-store",
                                 "X-Content-Type-Options": "nosniff"})


async def _im_thread(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kwargs))


def _mail_senden(an: str, betreff_html: tuple) -> None:
    betreff, inhalt = betreff_html
    try:
        _d.send_email(an, betreff, inhalt, bcc_admin=False)
    except Exception:  # noqa: BLE001 — eine Mail darf keinen Auftrag scheitern lassen
        log.exception("Express-Mail nicht versandt (%s)", betreff)


def _mails_nach(art_kunde: str = "", art_team: str = "", auftrag_id: int = 0, text: str = "") -> None:
    """Mails NACH einem erfolgreichen Schritt (im Thread aufrufen)."""
    try:
        a = express.auftrag_fuer_verwaltung(auftrag_id)
    except Exception:  # noqa: BLE001
        log.exception("Express-Mail: Auftrag %s nicht lesbar", auftrag_id)
        return
    if art_kunde and a.get("kunde_email"):
        _mail_senden(a["kunde_email"], express.mail_kunde(art_kunde, a, _d.base_url, text))
    if art_team:
        inhalt = express.mail_team(art_team, a, _d.base_url, text)
        for adresse in express.team_empfaenger(_d.notification_email):
            _mail_senden(adresse, inhalt)


def _hintergrund(fn, *args, **kwargs) -> None:
    """Mails nicht auf der Antwort warten lassen."""
    asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kwargs))


async def erinnerungs_schleife():
    """Alle 10 Minuten: Erinnerung 12 Stunden vor der Frist und Meldung bei Ueberfaelligkeit (je einmal). Gestartet aus
    main.lifespan (ein APIRouter-„startup“ laeuft neben einem eigenen lifespan nicht), nur wenn funktionen.EXPRESS an ist."""
    while True:
        try:
            for art, auftrag_id in await _im_thread(express.faellige_meldungen):
                await _im_thread(_mails_nach, "", art, auftrag_id)
        except Exception:  # noqa: BLE001
            log.exception("Express-Erinnerungen: Durchlauf fehlgeschlagen")
        await asyncio.sleep(600)


def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    express.RESULTS_DIR = deps.results_dir
    router = APIRouter()

    # ─── Seiten ───
    @router.get("/express", response_class=HTMLResponse)
    async def seite_express(request: Request):
        _an()
        return _d.render_seite(request, "express.html")

    @router.get("/express/auftrag/{auftrag_id}", response_class=HTMLResponse)
    async def seite_auftrag(auftrag_id: int, request: Request):
        _an()
        return _d.render_seite(request, "express_auftrag.html", auftrag_id=int(auftrag_id))

    @router.get("/express/bedingungen", response_class=HTMLResponse)
    async def seite_bedingungen(request: Request):
        _an()
        return _d.render_seite(request, "express_bedingungen.html", zustimmung_fassung=express.ZUSTIMMUNG_FASSUNG)

    @router.get("/verwaltung/express", response_class=HTMLResponse)
    async def seite_verwaltung(request: Request):
        _an()
        return _d.verwaltung_seite(request, "verwaltung_express.html", "express")

    @router.get("/verwaltung/express/{auftrag_id}", response_class=HTMLResponse)
    async def seite_verwaltung_auftrag(auftrag_id: int, request: Request):
        _an()
        return _d.verwaltung_seite(request, "verwaltung_express_auftrag.html", "express", auftrag_id=int(auftrag_id))

    # ─── Kunde: Warenkorb ───
    @router.get("/api/express/stand")
    async def stand(request: Request, user: dict = Depends(_kunde)):
        e = await _im_thread(express.einstellungen)
        korb = await _im_thread(express.warenkorb, user["id"])
        verf = await _im_thread(billing.verfuegbare_credits, user["id"])
        db = _d.get_user_by_id(user["id"]) or {}
        _ = _d.get_gettext(_d.resolve_ui_language(request))
        return {"warenkorb": korb, "preise": {"aufbereiten": e["preis_aufbereiten"], "pruefen": e["preis_pruefen"]},
                "frist_stunden": e["frist_stunden"], "max_seiten": e["max_seiten_auftrag"],
                "max_dokumente": e["max_dokumente_auftrag"], "guthaben": verf,
                "ansprechpartner": (db.get("display_name") or "").strip(),
                "texte": {"bedingungen": _(express.TEXT_BEDINGUNGEN), "bearbeitung": _(express.TEXT_BEARBEITUNG)}}

    @router.get("/api/express/projekte")
    async def projekte(user: dict = Depends(_kunde)):
        def lesen():
            from database import get_db
            conn = get_db()
            try:
                return [dict(r) for r in conn.execute(
                    "SELECT p.id, COALESCE(NULLIF(TRIM(p.name), ''), p.filename) AS name, "
                    "(SELECT COUNT(*) FROM documents d WHERE d.project_id = p.id) AS dokumente "
                    "FROM projects p WHERE p.user_id = ? AND EXISTS (SELECT 1 FROM documents d WHERE d.project_id = p.id) "
                    "ORDER BY p.updated_at DESC, p.id DESC LIMIT 500", (user["id"],))]
            finally:
                conn.close()
        return {"projekte": await _im_thread(lesen)}

    @router.get("/api/express/projekte/{project_id}/dokumente")
    async def projekt_dokumente(project_id: int, user: dict = Depends(_kunde)):
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
                eintrag = {"id": d["id"], "name": express._name_des_dokuments(d), "im_warenkorb": d["id"] in korb,
                           "seiten": None, "grund": ""}
                if not quelle or not express.ist_pdf_datei(quelle):
                    eintrag["grund"] = "keine PDF"
                else:
                    try:
                        eintrag["seiten"] = express.seitenzahl(quelle)
                    except express.ExpressFehler as fe:
                        eintrag["grund"] = fe.text
                out.append(eintrag)
            return out
        try:
            return {"dokumente": await _im_thread(lesen)}
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/warenkorb/dokumente")
    async def korb_dokumente(request: Request, user: dict = Depends(_kunde)):
        daten = await request.json()
        if not isinstance(daten, dict):
            raise HTTPException(status_code=400, detail="Ungültige Anfrage")
        try:
            return await _im_thread(express.dokumente_hinzufuegen, user["id"], daten.get("document_ids"))
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/warenkorb/hochladen")
    async def korb_hochladen(request: Request, file: UploadFile = File(...), user: dict = Depends(_kunde)):
        _bremsen("hochladen", user["id"])
        name = os.path.basename(file.filename or "dokument.pdf")
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Im Express-Service können zurzeit nur PDF-Dateien bearbeitet werden.")
        inhalt = await file.read(_d.max_upload_size + 1)
        if len(inhalt) > _d.max_upload_size:
            raise HTTPException(status_code=413, detail=f"Datei zu groß. Maximum: {_d.max_upload_size // (1024 * 1024)} MB")
        if not inhalt.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Die Datei ist keine PDF.")
        ordner = os.path.join(_d.upload_dir, str(int(user["id"])))
        os.makedirs(ordner, exist_ok=True)
        stamm = "".join(c for c in os.path.splitext(name)[0] if c.isalnum() or c in " ._-")[:80].strip() or "dokument"
        pfad = os.path.join(ordner, f"{time.strftime('%Y%m%d_%H%M%S')}_express_{stamm}.pdf")
        with open(pfad, "wb") as f:
            f.write(inhalt)
        grund = await _im_thread(_d.pdf_vorpruefung, pfad, _d.get_gettext(_d.resolve_ui_language(request)))
        if grund:
            os.unlink(pfad)
            raise HTTPException(status_code=400, detail=grund)
        projekt = await _im_thread(express.auto_projekt, user["id"])
        try:
            erg = await _d.handle_pdf_upload(pfad, name, user, projekt)
        except HTTPException:
            if projekt is None:
                raise
            # Das gemerkte Projekt taugt nicht mehr (z. B. gerade in Arbeit): ein neues anlegen.
            erg = await _d.handle_pdf_upload(pfad, name, user, None)
            projekt = None
        if projekt is None:
            await _im_thread(express.auto_projekt_merken, user["id"], int(erg["project_id"]))
        try:
            return await _im_thread(express.dokumente_hinzufuegen, user["id"], [int(erg["document_id"])])
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/warenkorb/positionen/{pos_id}/leistung")
    async def korb_leistung(pos_id: int, request: Request, user: dict = Depends(_kunde)):
        daten = await request.json()
        try:
            return await _im_thread(express.leistung_setzen, user["id"], pos_id, str((daten or {}).get("leistung") or ""))
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.delete("/api/express/warenkorb/positionen/{pos_id}")
    async def korb_entfernen(pos_id: int, user: dict = Depends(_kunde)):
        try:
            return await _im_thread(express.position_entfernen, user["id"], pos_id)
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/bestellen")
    async def bestellen(request: Request, user: dict = Depends(_kunde)):
        daten = await request.json()
        if not isinstance(daten, dict):
            raise HTTPException(status_code=400, detail="Ungültige Anfrage")
        _bremsen("bestellen", user["id"])
        sprache = _d.resolve_ui_language(request)
        _ = _d.get_gettext(sprache)
        try:
            erg = await _im_thread(
                express.bestellen, user["id"], ansprechpartner=daten.get("ansprechpartner"), telefon=daten.get("telefon"),
                hinweise=daten.get("hinweise"), bedingungen=daten.get("bedingungen"), bearbeitung=daten.get("bearbeitung"),
                idempotenz=daten.get("idempotenz"), sprache=sprache,
                texte={"bedingungen": _(express.TEXT_BEDINGUNGEN), "bearbeitung": _(express.TEXT_BEARBEITUNG)},
                absender=_d.absender_kennung(request))
        except express.ExpressFehler as e:
            raise _fehler(e)
        if erg["neu"]:
            log.info("Express-Auftrag %s bestellt (Konto %s)", erg["auftrag_id"], user["id"])
            _hintergrund(_mails_nach, "bestellt", "bestellt", erg["auftrag_id"])
        return {"ok": True, "auftrag_id": erg["auftrag_id"], "schon_bestellt": not erg["neu"]}

    # ─── Kunde: Auftraege ───
    @router.get("/api/express/auftraege")
    async def meine_auftraege(user: dict = Depends(_kunde)):
        return {"auftraege": await _im_thread(express.auftraege_des_kunden, user["id"])}

    @router.get("/api/express/auftraege/{auftrag_id}")
    async def mein_auftrag(auftrag_id: int, user: dict = Depends(_kunde)):
        try:
            return await _im_thread(express.auftrag_fuer_kunde, user["id"], auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)

    @router.post("/api/express/auftraege/{auftrag_id}/antwort")
    async def antwort(auftrag_id: int, request: Request, user: dict = Depends(_kunde)):
        daten = await request.json()
        _bremsen("antwort", user["id"])
        text = str((daten or {}).get("text") or "")
        try:
            a = await _im_thread(express.antwort_kunde, user["id"], auftrag_id, text)
        except express.ExpressFehler as e:
            raise _fehler(e)
        _hintergrund(_mails_nach, "", "antwort", auftrag_id, text.strip())
        return a

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
                # Zeiten stehen in der Kundenansicht schon in deutscher Zeit (express._lokal).
                express.nachweis_docx(a, docx, lambda t: t or "")
                pdf, bericht = pdfua_export.konvertiere(docx, os.path.basename(docx))
                klar = pdfua_export.klartext(bericht or {})
                if not klar.get("bestanden"):
                    # Nur eine barrierefreie PDF ausliefern (Steve) — sonst bleibt die Druckansicht.
                    log.warning("Express-Nachweis %s: PDF/UA-Pruefung nicht bestanden (%s)", a["id"], klar.get("zusammenfassung"))
                    raise pdfua_export.UmwandlungFehlgeschlagen("PDF/UA nicht bestanden")
                return pdf
        try:
            pdf = await _im_thread(bauen)
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
                "ist_admin": bool(db.get("is_admin"))}

    @router.get("/api/admin/express/auftraege/{auftrag_id}")
    async def v_auftrag(auftrag_id: int, user: dict = Depends(_leser)):
        try:
            a = await _im_thread(express.auftrag_fuer_verwaltung, auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)
        db = _d.get_user_by_id(user["id"]) or {}
        a["darf_arbeiten"] = bool((db.get("is_admin") and db.get("admin_level") == "full") or express.ist_bearbeiter(user["id"]))
        a["ist_admin"] = bool(db.get("is_admin"))
        a["lieferbar"] = express.lieferbar(a)
        if not a["ist_admin"]:
            # Bearbeiter (z. B. Partner) brauchen keinen Konto-Bezug: keine Kunden-ID fuer Verwaltungs-Links.
            a.pop("user_id", None)
            a.pop("konto_user_id", None)
        return a

    async def _schritt(fn, auftrag_id, *args, mail_kunde="", mail_team="", text=""):
        try:
            a = await _im_thread(fn, auftrag_id, *args)
        except express.ExpressFehler as e:
            raise _fehler(e)
        if mail_kunde or mail_team:
            _hintergrund(_mails_nach, mail_kunde, mail_team, auftrag_id, text)
        return a

    @router.post("/api/admin/express/auftraege/{auftrag_id}/uebernehmen")
    async def v_uebernehmen(auftrag_id: int, user: dict = Depends(_bearbeiter)):
        return await _schritt(express.uebernehmen, auftrag_id, _person(user["id"]))

    @router.post("/api/admin/express/auftraege/{auftrag_id}/rueckfrage")
    async def v_rueckfrage(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        text = str(((await request.json()) or {}).get("text") or "")
        return await _schritt(express.rueckfrage, auftrag_id, _person(user["id"]), text, mail_kunde="rueckfrage", text=text.strip())

    @router.post("/api/admin/express/auftraege/{auftrag_id}/notiz")
    async def v_notiz(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        text = str(((await request.json()) or {}).get("text") or "")
        return await _schritt(express.notiz_setzen, auftrag_id, _person(user["id"]), text)

    @router.post("/api/admin/express/auftraege/{auftrag_id}/liefern")
    async def v_liefern(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        trotz = ((await request.json()) or {}).get("trotz_befunden") is True
        a = await _schritt(express.liefern, auftrag_id, _person(user["id"]), trotz, mail_kunde="geliefert")
        log.info("Express-Auftrag %s geliefert von Konto %s", auftrag_id, user["id"])
        return a

    @router.post("/api/admin/express/auftraege/{auftrag_id}/stornieren")
    async def v_stornieren(auftrag_id: int, request: Request, user: dict = Depends(_bearbeiter)):
        grund = str(((await request.json()) or {}).get("grund") or "")
        a = await _schritt(express.stornieren, auftrag_id, _person(user["id"]), grund, mail_kunde="storniert", text=grund.strip())
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
        if art == "ergebnis":
            # veraPDF als Information (nicht als Sperre): beim Liefern fragt die Oberflaeche bei Abweichungen einmal nach.
            import pdf_tagging
            try:
                ergebnis = await _im_thread(pdf_tagging.verapdf, erg["pfad"])
            except Exception:  # noqa: BLE001
                log.exception("Express: veraPDF fuer Position %s fehlgeschlagen", pos_id)
                ergebnis = None
            await _im_thread(express.verapdf_merken, pos_id, ergebnis)
        a = await _im_thread(express.auftrag_fuer_verwaltung, auftrag_id)
        a["lieferbar"] = express.lieferbar(a)
        return a

    @router.get("/api/admin/express/auftraege/{auftrag_id}/positionen/{pos_id}/{art}")
    async def v_datei(auftrag_id: int, pos_id: int, art: str, user: dict = Depends(_leser)):
        try:
            pfad, name = await _im_thread(express.datei_fuer_verwaltung, auftrag_id, pos_id, art)
        except express.ExpressFehler as e:
            raise _fehler(e)
        return _datei(pfad, name)

    @router.get("/api/admin/express/auftraege/{auftrag_id}/originale.zip")
    async def v_zip(auftrag_id: int, user: dict = Depends(_leser)):
        try:
            dateien = await _im_thread(express.originale_fuer_zip, auftrag_id)
        except express.ExpressFehler as e:
            raise _fehler(e)
        if not dateien:
            raise HTTPException(status_code=404, detail="Keine Originale vorhanden")

        def packen():
            fd, pfad = tempfile.mkstemp(prefix="express_zip_", suffix=".zip")
            os.close(fd)
            with zipfile.ZipFile(pfad, "w", zipfile.ZIP_DEFLATED) as z:
                for quelle, name in dateien:
                    z.write(quelle, arcname=name)
            return pfad
        pfad = await _im_thread(packen)
        return FileResponse(pfad, media_type="application/zip", background=BackgroundTask(os.remove, pfad),
                            headers={"Content-Disposition": _disposition(f"Express-Auftrag {int(auftrag_id)} – Originale.zip"),
                                     "Cache-Control": "no-store"})

    @router.get("/api/admin/express/einstellungen")
    async def v_einstellungen(user: dict = Depends(_leser)):
        return await _im_thread(express.einstellungen)

    @router.post("/api/admin/express/einstellungen")
    async def v_einstellungen_setzen(request: Request, user: dict = Depends(_voll)):
        daten = await request.json()
        if not isinstance(daten, dict):
            raise HTTPException(status_code=400, detail="Ungültige Anfrage")
        try:
            e = await _im_thread(express.speichere_einstellungen, daten)
        except express.ExpressFehler as fe:
            raise _fehler(fe)
        log.info("Express-Einstellungen geaendert von %s", user.get("email"))
        return {"ok": True, "einstellungen": e, "message": "Einstellungen gespeichert."}

    @router.get("/api/admin/express/bearbeiter")
    async def v_bearbeiter(user: dict = Depends(_voll)):
        return {"bearbeiter": await _im_thread(express.bearbeiter_liste)}

    @router.post("/api/admin/express/bearbeiter")
    async def v_bearbeiter_dazu(request: Request, user: dict = Depends(_voll)):
        mail = str(((await request.json()) or {}).get("email") or "").strip()
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
