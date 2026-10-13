"""Formular-Karten des InkluAgent (Paket 1, Steve 13.10.2026; Schalter funktionen.AGENT_KARTEN, Umgebung INKLUAGENT_KARTEN=an).

Steve: „der Agent bringt die Knöpfe zu einem“. Statt den Nutzer auf eine Seite zu schicken, legt der InkluAgent das Formular als
Karte in das Gespraech — vorbelegt mit dem, was der Nutzer gesagt hat. Der Nutzer prueft, aendert und schickt mit EINEM Knopf ab.
Fuer Sehende fliegt die Karte ins Gespraech (frontend/inkluagent.js), dazu der Wusch der Klangspur (frontend/klang.js).

Karten (KARTENTYPEN):
- „einladen“: Zur Prüfung einladen (Projekt freigeben) — dieselbe Funktion wie der Dialog im Projekt (main._freigabe_anlegen).
- „express_bestellen“: Express-Warenkorb zahlungspflichtig bestellen — dieselbe Funktion wie der Warenkorb
  (express_api.bestellung) mit Aufstellung, Preis und Bedingungen. Nur mit funktionen.EXPRESS.

Sicherheit:
- Was zaehlt, haelt der SERVER in der Karte: Projekt, Warenkorb, erwarteter Preis, Fassung, Idempotenz. Der Browser schickt nur
  die Werte der Felder. Eine Karte gehoert einem Konto, gilt GUELTIG_S und wird hoechstens einmal erfolgreich abgeschickt.
- Das Modell schlaegt nur Werte vor (Werkzeuge …_vorbereiten). Ein Werkzeug zum Abschicken gibt es nicht; abschicken kann nur
  der Knopf der Karte (POST /api/agent/karte). Vorschlaege des Modells prueft der Server wie Eingaben (eine E-Mail-Adresse, die
  keine ist, bleibt leer).
- Die Felder prueft der Server nach dem Schema der Karte (Pflicht, Laenge, Art, Auswahl, Haken); danach prueft die Funktion
  dahinter wie auf der Seite.
"""
from __future__ import annotations

import importlib
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import funktionen

GUELTIG_S = 30 * 60
HOECHSTENS_JE_KONTO = 20          # offene Karten je Konto im Speicher (die aeltesten fallen weg)
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}$")


def an() -> bool:
    return funktionen.an("AGENT_KARTEN")


def _main():
    return importlib.import_module("main")


# ─── Schema ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Feld:
    """Ein Feld der Karte. art: text, email, tel, textarea, auswahl, haken. label und Texte sind msgids (uebersetzt beim Bauen)."""
    name: str
    art: str
    label: str
    pflicht: bool = False
    max: int = 200
    auswahl: tuple = ()            # ((wert, label-msgid), …)
    autocomplete: str = ""
    link: Optional[tuple] = None   # (url, text-msgid) neben dem Feld, z. B. die Bedingungen


@dataclass(frozen=True)
class Kartentyp:
    art: str
    titel: str                     # msgid
    knopf: str                     # msgid
    felder: tuple = field(default_factory=tuple)


EINLADEN = Kartentyp(
    art="einladen", titel="Zur Prüfung einladen", knopf="Einladung per E-Mail senden",
    felder=(
        Feld("name", "text", "Name (optional)", max=120, autocomplete="name"),
        Feld("email", "email", "E-Mail-Adresse (Pflichtfeld)", pflicht=True, max=254, autocomplete="email"),
        Feld("rolle", "auswahl", "Rolle des Empfängers", pflicht=True, auswahl=(("kunde", "Herausgeber"), ("lektorat", "Lektorat"))),
        Feld("nachricht", "textarea", "Nachricht (optional)", max=2000),
    ))

EXPRESS_BESTELLEN = Kartentyp(
    art="express_bestellen", titel="Express-Auftrag bestellen", knopf="Zahlungspflichtig bestellen",
    felder=(
        Feld("ansprechpartner", "text", "Ansprechpartner (Pflicht)", pflicht=True, max=120, autocomplete="name"),
        Feld("telefon", "tel", "Telefon für Rückfragen (freiwillig)", max=40, autocomplete="tel"),
        Feld("hinweise", "textarea", "Hinweise für uns (freiwillig)", max=2000),
        Feld("bedingungen", "haken", "Ich akzeptiere die Bedingungen für den Express-Service.", pflicht=True,
             link=("/express/bedingungen", "Bedingungen für den Express-Service lesen (öffnet in neuem Tab)")),
    ))

KARTENTYPEN = {k.art: k for k in (EINLADEN, EXPRESS_BESTELLEN)}


def _uebersetzt(typ: Kartentyp, werte: dict, _: Callable[[str], str]) -> list[dict]:
    """Felder fuer die Oberflaeche: Beschriftungen in der Sprache des Nutzers, Werte vorbelegt."""
    out = []
    for f in typ.felder:
        d: dict[str, Any] = {"name": f.name, "art": f.art, "label": _(f.label), "pflicht": f.pflicht, "max": f.max,
                             "wert": werte.get(f.name, False if f.art == "haken" else "")}
        if f.auswahl:
            d["auswahl"] = [{"wert": w, "label": _(l)} for w, l in f.auswahl]
        if f.autocomplete:
            d["autocomplete"] = f.autocomplete
        if f.link:
            d["link"] = {"url": f.link[0], "text": _(f.link[1])}
        out.append(d)
    return out


def werte_pruefen(typ: Kartentyp, roh: Any, _: Callable[[str], str]) -> tuple[dict, dict]:
    """Eingaben nach dem Schema: (saubere Werte, Fehler je Feld). Unbekannte Felder fallen weg."""
    roh = roh if isinstance(roh, dict) else {}
    werte: dict[str, Any] = {}
    fehler: dict[str, str] = {}
    for f in typ.felder:
        v = roh.get(f.name)
        if f.art == "haken":
            werte[f.name] = v is True
            if f.pflicht and v is not True:
                fehler[f.name] = _("Bitte bestätige das.")
            continue
        text = v.strip() if isinstance(v, str) else ""
        werte[f.name] = text
        if f.pflicht and not text:
            fehler[f.name] = _("Bitte fülle dieses Feld aus.")
        elif len(text) > f.max:
            fehler[f.name] = _("Höchstens {n} Zeichen.").format(n=f.max)
        elif f.art == "email" and text and not _EMAIL.match(text):
            fehler[f.name] = _("Bitte gib eine gültige E-Mail-Adresse an.")
        elif f.auswahl and text not in {w for w, _l in f.auswahl}:
            fehler[f.name] = _("Bitte wähle einen Eintrag aus der Liste.")
    return werte, fehler


# ─── Karten im Speicher ──────────────────────────────────────────────────────────────────────────────────────────────────────
# Nur im Prozess (ein uvicorn-Worker, wie die Angebote in tools/ausgaben.py). Nach einem Neustart sagt der Knopf „nicht mehr
# gültig“, und der Agent legt die Karte neu vor.
_KARTEN: dict[str, dict] = {}
_SPERRE = threading.Lock()


def _aufraeumen(jetzt: float) -> None:
    for k in [k for k, e in _KARTEN.items() if jetzt - e["zeit"] > GUELTIG_S]:
        _KARTEN.pop(k, None)


def _merken(user_id: int, typ: Kartentyp, project_id: Optional[int], kontext: dict) -> str:
    jetzt = time.time()
    kid = "k" + secrets.token_hex(12)
    with _SPERRE:
        _aufraeumen(jetzt)
        eigene = sorted((e["zeit"], k) for k, e in _KARTEN.items() if e["user_id"] == user_id)
        for _zeit, alt in eigene[:max(0, len(eigene) - HOECHSTENS_JE_KONTO + 1)]:
            _KARTEN.pop(alt, None)
        _KARTEN[kid] = {"user_id": int(user_id), "art": typ.art, "project_id": project_id, "kontext": kontext, "zeit": jetzt,
                        "zustand": "offen"}
    return kid


def _belegen(kid: str, user_id: int) -> tuple[Optional[dict], str]:
    """Die Karte fuer GENAU einen Klick belegen: (Eintrag, '') oder (None, Grund) — 'gesendet', 'laeuft' oder 'ungueltig'."""
    with _SPERRE:
        _aufraeumen(time.time())
        e = _KARTEN.get(kid)
        if not e or e["user_id"] != int(user_id):
            return None, "ungueltig"
        if e["zustand"] != "offen":
            return None, e["zustand"]
        e["zustand"] = "laeuft"
        return e, ""


def _zustand(kid: str, zustand: str) -> None:
    with _SPERRE:
        e = _KARTEN.get(kid)
        if e:
            e["zustand"] = zustand


def karte(typ: Kartentyp, kid: str, werte: dict, _: Callable[[str], str], text: str = "", zeilen: Optional[list] = None) -> dict:
    """Anhang fuer die Antwort (art „formular“): so steht die Karte im Gespraech und im gespeicherten Verlauf. Abgeschickt wird
    jede Karte ueber POST /api/agent/karte, auf jeder Seite."""
    return {"art": "formular", "karte": typ.art, "karten_id": kid, "titel": _(typ.titel), "text": text,
            "zeilen": list(zeilen or []), "felder": _uebersetzt(typ, werte, _), "knopf": _(typ.knopf), "zustand": "offen"}


# ─── Werkzeuge des Agenten ───────────────────────────────────────────────────────────────────────────────────────────────────
def werkzeuge(project_id: Optional[int]) -> list[dict]:
    defs = [{
        "name": "einladen_vorbereiten",
        "description": ("Legt dem Nutzer das Formular „Zur Prüfung einladen“ als Karte ins Gespräch, vorbelegt mit dem, was er "
                        "gesagt hat (Name, E-Mail-Adresse, Rolle, Nachricht). Abschicken kann NUR der Nutzer mit dem Knopf der "
                        "Karte — du verschickst nichts. Erfinde keine E-Mail-Adresse: fehlt sie, lass das Feld leer."
                        + ("" if project_id else " Auf Seiten ohne Projekt: project_id aus projekte_auflisten.")),
        "input_schema": {"type": "object", "properties": {
            "project_id": {"type": "integer", "description": "nur ohne geöffnetes Projekt nötig"},
            "email": {"type": "string"}, "name": {"type": "string"},
            "rolle": {"type": "string", "enum": ["kunde", "lektorat"],
                      "description": "kunde = Herausgeber (prüft und gibt frei), lektorat = Lektorat"},
            "nachricht": {"type": "string", "description": "persönliche Nachricht; ersetzt den Einleitungssatz der Einladung"}}},
    }]
    if funktionen.an("EXPRESS"):
        defs.append({
            "name": "express_bestellen_vorbereiten",
            "description": ("Legt dem Nutzer das Formular „Zahlungspflichtig bestellen“ für seinen Express-Warenkorb als Karte "
                            "ins Gespräch: Aufstellung, Preis in Credits, Ansprechpartner, Bedingungen. Bestellen kann NUR der "
                            "Nutzer mit dem Knopf der Karte. Ist der Warenkorb leer, sag ihm, wie Dokumente hineinkommen "
                            "(am Dokument „In den Express-Warenkorb“)."),
            "input_schema": {"type": "object", "properties": {
                "ansprechpartner": {"type": "string"}, "telefon": {"type": "string"}, "hinweise": {"type": "string"}}},
        })
    return defs


WERKZEUG_KARTE = {"einladen_vorbereiten": EINLADEN, "express_bestellen_vorbereiten": EXPRESS_BESTELLEN}
NAMEN = frozenset(WERKZEUG_KARTE)

PROMPT = """Formulare im Gespräch (Karten)

Will der Nutzer jemanden zum Prüfen einladen oder seinen Express-Warenkorb bestellen, bring ihm das Formular als Karte ins Gespräch (einladen_vorbereiten, express_bestellen_vorbereiten), statt ihn auf eine Seite zu schicken. Fülle vor, was er gesagt hat; erfinde nichts. Abschicken kann nur der Nutzer mit dem Knopf der Karte — frag also nicht nach einem Ja, sondern sag in einem Satz, was vorbelegt ist und welcher Knopf abschickt."""


def ausfuehren(name: str, args: dict, user_id: int, project_id: Optional[int], _t: Optional[Callable[[str], str]] = None) -> dict:
    """Werkzeug des Agenten -> Werkzeug-Ergebnis mit der Karte als anhang (agent_loop haengt sie an die Antwort)."""
    _ = _t or (lambda s: s)
    args = args if isinstance(args, dict) else {}
    try:
        if name == "einladen_vorbereiten":
            return _einladen_vorbereiten(args, user_id, project_id, _)
        if name == "express_bestellen_vorbereiten" and funktionen.an("EXPRESS"):
            return _express_vorbereiten(args, user_id, _)
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).exception("Karte %s nicht vorbereitet", name)
        return {"ok": False, "error": "Die Karte ließ sich gerade nicht vorbereiten."}
    return {"ok": False, "error": f"Unbekanntes Tool: {name}"}


def _eigenes_projekt(user_id: int, project_id: Any) -> Optional[dict]:
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return None
    from database import get_db
    conn = get_db()
    try:
        r = conn.execute("SELECT id, COALESCE(NULLIF(TRIM(name), ''), filename) AS name FROM projects WHERE id = ? AND user_id = ?",
                         (pid, int(user_id))).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def _vorschlag(v: Any, laenge: int) -> str:
    return v.strip()[:laenge] if isinstance(v, str) else ""


def _einladen_vorbereiten(args: dict, user_id: int, project_id: Optional[int], _) -> dict:
    from .daten import daten
    p = _eigenes_projekt(user_id, project_id or args.get("project_id"))
    if not p:
        return {"ok": False, "error": "Projekt nicht gefunden (nur eigene Projekte). Frag, welches Projekt gemeint ist."}
    email = _vorschlag(args.get("email"), 254)
    werte = {"name": _vorschlag(args.get("name"), 120), "email": email if _EMAIL.match(email) else "",
             "rolle": args.get("rolle") if args.get("rolle") in ("kunde", "lektorat") else "kunde",
             "nachricht": _vorschlag(args.get("nachricht"), 2000)}
    kid = _merken(user_id, EINLADEN, None if project_id is None else int(p["id"]), {"project_id": int(p["id"])})
    text = _("Projekt „{name}“").format(name=p["name"])
    return {"ok": True,
            "result": {"karte": "einladen", "projekt_daten": daten(p["name"]), "vorbelegt": [k for k, v in werte.items() if v],
                       "hinweis": "Die Karte steht unter deiner Antwort. Sag in einem Satz, was vorbelegt ist und dass der Knopf "
                                  "„Einladung per E-Mail senden“ die Einladung verschickt."},
            "anhang": karte(EINLADEN, kid, werte, _, text=text)}


def _express_vorbereiten(args: dict, user_id: int, _) -> dict:
    import billing
    import express
    from database import get_user_by_id
    korb = express.warenkorb(user_id)
    if not korb.get("id") or not korb.get("positionen"):
        return {"ok": True, "result": {"warenkorb_leer": True, "hinweis": (
            "Der Express-Warenkorb ist leer. Sag dem Nutzer, dass er Dokumente im Projekt mit „In den Express-Warenkorb“ "
            "hineinlegt; dann bringst du ihm das Bestellformular.")}}
    e = express.einstellungen()
    guthaben = billing.verfuegbare_credits(user_id)
    leistung = {l["schluessel"]: l["name"] for l in express.leistungen_liste(e)}
    zeilen = [_("„{name}“: {seiten} Seiten, {leistung}, {credits} Credits").format(
        name=p.get("dokument_name") or "", seiten=p.get("seiten") or 0, leistung=_(leistung.get(p.get("leistung"), "")),
        credits=p.get("credits") or 0) for p in korb["positionen"]]
    text = _("{d} Dokumente, {s} Seiten, zusammen {c} Credits. Dein Guthaben: {g} Credits. Fertig in {h} Stunden.").format(
        d=korb["dokumente"], s=korb["seiten"], c=korb["credits"], g=guthaben, h=e["frist_stunden"])
    if guthaben < korb["credits"]:
        return {"ok": True, "result": {"guthaben_reicht_nicht": True, "credits": korb["credits"], "guthaben": guthaben,
                                       "hinweis": "Das Guthaben reicht nicht. Nenne Preis und Guthaben und verweise auf "
                                                  "„Abo & Verbrauch“."}}
    nutzer = get_user_by_id(user_id) or {}
    werte = {"ansprechpartner": _vorschlag(args.get("ansprechpartner"), 120) or (nutzer.get("display_name") or "").strip()[:120],
             "telefon": _vorschlag(args.get("telefon"), 40), "hinweise": _vorschlag(args.get("hinweise"), 2000),
             "bedingungen": False}
    kid = _merken(user_id, EXPRESS_BESTELLEN, None, {
        "korb_id": int(korb["id"]), "erwartete_credits": int(korb["credits"]), "fassung": korb["fassung"],
        "idempotenz": "karte-" + secrets.token_hex(12)})
    return {"ok": True,
            "result": {"karte": "express_bestellen", "dokumente": korb["dokumente"], "seiten": korb["seiten"],
                       "credits": korb["credits"], "guthaben": guthaben,
                       "hinweis": "Die Karte steht unter deiner Antwort. Nenne Preis und Frist in einem Satz; bestellen kann nur "
                                  "der Nutzer mit „Zahlungspflichtig bestellen“, nachdem er die Bedingungen akzeptiert hat."},
            "anhang": karte(EXPRESS_BESTELLEN, kid, werte, _, text=text, zeilen=zeilen)}


# ─── Abschicken (Knopf der Karte) ────────────────────────────────────────────────────────────────────────────────────────────
async def absenden(kid: str, user: dict, roh: Any, sprache: str, absender: str = "") -> tuple[int, dict]:
    """POST /api/agent/karte: (HTTP-Status, Antwort). Antwort bei Erfolg {"ok", "text", "karte": {"zustand", "ergebnis"},
    "anhang"?}; sonst {"detail", "fehler": {feld: text}?, "karte"?}. Felder mit Fehlern: die Karte bleibt offen, der Nutzer
    korrigiert. „karte“ geht auch in den gespeicherten Verlauf (Titel und Felder der Karte bleiben, dazu Zustand und Ergebnis)."""
    m = _main()
    _ = m.get_gettext(sprache)
    e, grund = _belegen(str(kid or ""), user["id"])
    if not e:
        if grund in ("gesendet", "laeuft"):
            return 409, {"detail": _("Schon abgeschickt."), "karte": {"zustand": "gesendet"}}
        text = _("Diese Karte gilt nicht mehr. Bitte frag im Chat neu.")
        return 404, {"detail": text, "karte": {"zustand": "abgelaufen", "ergebnis": text}}
    typ = KARTENTYPEN[e["art"]]
    werte, fehler = werte_pruefen(typ, roh, _)
    if fehler:
        _zustand(kid, "offen")
        return 400, {"detail": _("Bitte prüfe die markierten Felder."), "fehler": fehler}
    try:
        if typ is EINLADEN:
            status, antwort = await _einladen_senden(e, user, werte, _)
        else:
            status, antwort = await _express_senden(e, user, werte, sprache, absender, _)
    except Exception:
        _zustand(kid, "offen")
        raise
    _zustand(kid, "gesendet" if status == 200 else ("abgelaufen" if (antwort.get("karte") or {}).get("zustand") == "abgelaufen"
                                                     else "offen"))
    if status == 200 or (antwort.get("karte") or {}).get("zustand") == "abgelaufen":
        _im_verlauf_aendern(user["id"], e, kid, dict(antwort.get("karte") or {}, werte=werte))
    return status, antwort


async def _einladen_senden(e: dict, user: dict, werte: dict, _) -> tuple[int, dict]:
    import asyncio
    from fastapi import HTTPException
    m = _main()
    daten = {"guest_email": werte["email"], "guest_name": werte["name"], "message": werte["nachricht"], "role": werte["rolle"],
             "notify": True}
    try:
        erg = await asyncio.get_running_loop().run_in_executor(
            None, lambda: m._freigabe_anlegen(user, int(e["kontext"]["project_id"]), daten))
    except HTTPException as fehler:
        if fehler.status_code == 404:
            text = _("Das Projekt gibt es nicht mehr.")
            return 404, {"detail": text, "karte": {"zustand": "abgelaufen", "ergebnis": text}}
        return fehler.status_code, {"detail": str(fehler.detail)}
    if erg.get("sent"):
        text = _("Die Einladung an {email} ist verschickt.").format(email=werte["email"])
    else:
        text = _("Die Freigabe ist angelegt, aber die E-Mail ging nicht raus. Den Link findest du im Projekt unter „Zur Prüfung "
                 "freigeben“.")
    return 200, {"ok": True, "text": text, "karte": {"zustand": "gesendet", "ergebnis": text}}


async def _express_senden(e: dict, user: dict, werte: dict, sprache: str, absender: str, _) -> tuple[int, dict]:
    import express
    import express_api
    from fastapi import HTTPException
    k = e["kontext"]
    daten = {"ansprechpartner": werte["ansprechpartner"], "telefon": werte["telefon"], "hinweise": werte["hinweise"],
             "bedingungen": werte["bedingungen"], "idempotenz": k["idempotenz"], "korb_id": k["korb_id"],
             "erwartete_credits": k["erwartete_credits"], "fassung": k["fassung"]}
    try:
        erg = await express_api.bestellung(user["id"], daten, sprache, absender)
    except express.ExpressFehler as fehler:
        text = _(fehler.text)
        if fehler.extra.get("veraltet"):
            text = text + " " + _("Frag den InkluAgent nach einem neuen Bestellformular.")
            return 409, {"detail": text, "karte": {"zustand": "abgelaufen", "ergebnis": text}}
        feld = fehler.extra.get("feld")
        return fehler.status, ({"detail": text, "fehler": {feld: text}} if feld else {"detail": text})
    except HTTPException as fehler:   # Bremse (429) oder Schalter aus (404)
        return fehler.status_code, {"detail": str(fehler.detail)}
    text = _("Bestellt: Express-Auftrag {nr}. Die Bestätigung kommt per E-Mail.").format(nr=erg["auftrag_id"])
    return 200, {"ok": True, "text": text, "karte": {"zustand": "gesendet", "ergebnis": text},
                 "anhang": {"art": "link", "url": f"/express/auftrag/{int(erg['auftrag_id'])}", "text": _("Auftrag ansehen")}}


def _im_verlauf_aendern(user_id: int, e: dict, kid: str, felder: dict) -> None:
    """Die Karte im gespeicherten Gespraech auf ihren Zustand stellen, damit sie nach dem Neuladen nicht wieder offen steht."""
    from . import storage
    try:
        storage.karte_aktualisieren(e.get("project_id") or 0, kid, felder, user_id=user_id)
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).exception("Karte %s im Verlauf nicht aktualisiert", kid)
