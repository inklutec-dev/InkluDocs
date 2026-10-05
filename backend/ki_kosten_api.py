"""KI-KOSTEN — Endpunkte der Verwaltung (05.10.2026, Steve).

Bereich „KI-Kosten“: was die KI-Aufrufe im Monat gekostet haben — gesamt, nach Zweck, nach Kunde bis zum einzelnen
Bild — neben dem Umsatz; dazu die Preisliste. Lesen: jeder Admin (wie Umsatz). Preisliste aendern: nur Voll-Admins.
Kern ohne HTTP: ki_kosten.py (Tabelle ki_aufrufe). Doku: docs/KI_KOSTEN.md.

  GET  /api/admin/ki-kosten?jahr&monat[&umgebung]           Monatsbericht
  GET  /api/admin/ki-kosten/kunde?konto=<id|ohne>&jahr&monat Projekte eines Kontos
  GET  /api/admin/ki-kosten/projekt?projekt=&konto=&jahr&monat Bilder eines Projekts
  GET  /api/admin/ki-preise                                  Preisliste
  POST /api/admin/ki-preise/modell   {modell, ein, aus, cache?, ab, quelle}   (Voll-Admin)
  POST /api/admin/ki-preise/kurs     {usd_eur, quelle}                        (Voll-Admin)
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

import ki_kosten
import umsatz

log = logging.getLogger("ki_kosten_api")

UMGEBUNGEN = ("prod", "staging", "demo", "test")


@dataclass
class Deps:
    require_admin: Callable          # (request) -> user; jeder Admin
    require_full_admin: Callable     # (request) -> user; nur Voll-Admins
    pauschale_eur_je_credit: Callable  # () -> float (billing.KOSTEN_PRO_CREDIT_EUR) fuer den Vergleich


_d: Optional[Deps] = None


def _monat(jahr, monat) -> tuple:
    """(jahr, monat) aus der Anfrage; ohne Angabe der laufende Monat (deutsche Zeit)."""
    if jahr in (None, "") and monat in (None, ""):
        jetzt = umsatz.jetzt_lokal()
        return jetzt.year, jetzt.month
    try:
        j, m = int(jahr), int(monat)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Jahr und Monat müssen Zahlen sein")
    if not (2026 <= j <= 2100 and 1 <= m <= 12):
        raise HTTPException(status_code=400, detail="Diesen Zeitraum gibt es nicht")
    return j, m


def _umgebung(umgebung):
    if umgebung in (None, "", "alle"):
        return None
    if umgebung not in UMGEBUNGEN:
        raise HTTPException(status_code=400, detail="Unbekannte Umgebung")
    return [umgebung]


def _kennung(wert, name: str):
    """Konto- bzw. Projektkennung aus der Anfrage: Zahl oder 'ohne' (= ohne Zuordnung)."""
    if wert == "ohne":
        return None
    try:
        return int(wert)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{name} ungültig")


def _preis_zahl(wert, name: str, pflicht: bool = True):
    """'2,00' / '2.00' / 2 -> 2.0 (USD je 1 Mio. Tokens); leer -> None, wenn nicht Pflicht."""
    if wert in (None, "") and not pflicht:
        return None
    text = str(wert if wert is not None else "").strip().replace(" ", "").replace(",", ".")
    if not re.fullmatch(r"\d{1,4}(\.\d{1,6})?", text):
        raise HTTPException(status_code=400, detail=f"{name}: bitte eine Zahl, zum Beispiel 2,50")
    return float(text)


def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    router = APIRouter()

    def _admin(request: Request):
        return _d.require_admin(request)

    def _voll(request: Request):
        return _d.require_full_admin(request)

    @router.get("/api/admin/ki-kosten")
    async def ki_kosten_monat(jahr: str = None, monat: str = None, umgebung: str = "alle", user: dict = Depends(_admin)):
        """Monatsbericht der KI-Kosten mit Umsatz, Zwecken, Modellen und Kunden."""
        j, m = _monat(jahr, monat)
        umg = _umgebung(umgebung)
        loop = asyncio.get_running_loop()
        bericht = await loop.run_in_executor(None, ki_kosten.monatsbericht, j, m, umg)
        zeitraeume = await loop.run_in_executor(None, ki_kosten.zeitraeume)
        beginn = await loop.run_in_executor(None, ki_kosten.messbeginn)
        bericht.update({"zeitraeume": zeitraeume, "messbeginn": umsatz.lokal(beginn) or None,
                        "umgebung": umgebung or "alle", "zwecke": ki_kosten.ZWECKE,
                        "pauschale_cent_je_credit": _d.pauschale_eur_je_credit() * 100})
        return bericht

    @router.get("/api/admin/ki-kosten/kunde")
    async def ki_kosten_kunde(konto: str, jahr: str = None, monat: str = None, umgebung: str = "alle",
                              user: dict = Depends(_admin)):
        """Projekte eines Kontos (oder „ohne Zuordnung“) mit ihren KI-Kosten im Monat."""
        j, m = _monat(jahr, monat)
        return await asyncio.get_running_loop().run_in_executor(
            None, ki_kosten.kunde_bericht, _kennung(konto, "Konto"), j, m, _umgebung(umgebung))

    @router.get("/api/admin/ki-kosten/projekt")
    async def ki_kosten_projekt(projekt: str, konto: str, jahr: str = None, monat: str = None, umgebung: str = "alle",
                                user: dict = Depends(_admin)):
        """Bilder eines Projekts mit ihren KI-Kosten im Monat, dazu die Aufrufe ohne Bild (z. B. Chatbot)."""
        j, m = _monat(jahr, monat)
        return await asyncio.get_running_loop().run_in_executor(
            None, ki_kosten.projekt_bericht, _kennung(projekt, "Projekt"), _kennung(konto, "Konto"), j, m, _umgebung(umgebung))

    @router.get("/api/admin/ki-preise")
    async def ki_preise(user: dict = Depends(_admin)):
        """Preisliste der KI-Modelle (USD je 1 Mio. Tokens) und Wechselkurs."""
        return ki_kosten.preise_fuer_anzeige()

    @router.post("/api/admin/ki-preise/modell")
    async def ki_preis_setzen(request: Request, user: dict = Depends(_voll)):
        """Preisstufe eines Modells anlegen oder ersetzen (gleiches „gültig ab“). Neue Modelle sind erlaubt. Die Kosten
        schon erfasster Aufrufe bleiben unveraendert (festgeschrieben beim Aufruf)."""
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Ungültige Anfrage")
        modell = str(data.get("modell") or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9._:\-]{1,99}", modell):
            raise HTTPException(status_code=400, detail="Modellkennung ungültig (Kleinbuchstaben, Ziffern, Punkt, Bindestrich)")
        ab = str(data.get("ab") or "").strip() or datetime.utcnow().strftime("%Y-%m-%d")
        try:
            datetime.strptime(ab, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="„Gültig ab“ bitte als Datum JJJJ-MM-TT")
        stufe = {"ab": ab, "ein": _preis_zahl(data.get("ein"), "Eingabe"), "aus": _preis_zahl(data.get("aus"), "Ausgabe")}
        for feld, name in (("cache", "Zwischenspeicher"), ("cache_schreiben", "Zwischenspeicher schreiben")):
            wert = _preis_zahl(data.get(feld), name, pflicht=False)
            if wert is not None:
                stufe[feld] = wert
        quelle = str(data.get("quelle") or "").strip()
        if not quelle:
            raise HTTPException(status_code=400, detail="Bitte die Quelle des Preises angeben (z. B. Preisseite und Datum)")
        liste = ki_kosten.preise()
        eintrag = liste["modelle"].setdefault(modell, {"stufen": []})
        alt = next((s for s in eintrag.get("stufen") or [] if s.get("ab") == ab), None)
        if alt:
            # Lange-Eingabe-Staffel (Gemini Pro) und Cache-Schreibpreis behalten, wenn nur die Grundpreise geaendert werden.
            for k in ("grenze", "ein_lang", "aus_lang", "cache_lang", "cache_schreiben"):
                if k in alt and k not in stufe:
                    stufe[k] = alt[k]
        eintrag["stufen"] = sorted([s for s in eintrag.get("stufen") or [] if s.get("ab") != ab] + [stufe],
                                   key=lambda s: s["ab"])
        eintrag["quelle"] = quelle[:300]
        try:
            ki_kosten.speichere_preise(liste)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        log.info("KI-Preis geaendert von %s: %s ab %s", user.get("email"), modell, ab)
        return {"ok": True, "preise": ki_kosten.preise_fuer_anzeige(),
                "message": f"Preis für {modell} gespeichert (gültig ab {ab})."}

    @router.post("/api/admin/ki-preise/kurs")
    async def ki_kurs_setzen(request: Request, user: dict = Depends(_voll)):
        """Wechselkurs USD -> EUR fuer neue Aufrufe."""
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Ungültige Anfrage")
        text = str(data.get("usd_eur") or "").strip().replace(",", ".")
        if not re.fullmatch(r"\d(\.\d{1,6})?", text) or not (0.2 <= float(text) <= 5.0):
            raise HTTPException(status_code=400, detail="Wechselkurs bitte als Zahl, zum Beispiel 0,8909")
        quelle = str(data.get("quelle") or "").strip()
        if not quelle:
            raise HTTPException(status_code=400, detail="Bitte die Quelle des Kurses angeben (z. B. EZB-Referenzkurs und Datum)")
        liste = ki_kosten.preise()
        liste["usd_eur"], liste["usd_eur_quelle"] = float(text), quelle[:300]
        try:
            ki_kosten.speichere_preise(liste)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        log.info("KI-Wechselkurs geaendert von %s: %s", user.get("email"), text)
        return {"ok": True, "preise": ki_kosten.preise_fuer_anzeige(), "message": "Wechselkurs gespeichert."}

    return router
