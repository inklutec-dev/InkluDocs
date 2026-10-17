"""Formular-Karten des InkluAgent (Paket 1, Steve 13.10.2026; Schalter funktionen.AGENT_KARTEN, Umgebung INKLUAGENT_KARTEN=an).

Steve: „der Agent bringt die Knöpfe zu einem“. Statt den Nutzer auf eine Seite zu schicken, legt der InkluAgent das Formular als
Karte in das Gespraech — vorbelegt mit dem, was der Nutzer gesagt hat. Der Nutzer prueft, aendert und schickt mit EINEM Knopf ab.
Fuer Sehende fliegt die Karte ins Gespraech (frontend/inkluagent.js), dazu der Wusch der Klangspur (frontend/klang.js).

Karten (KARTENTYPEN):
- „einladen“: Zur Prüfung einladen (Projekt freigeben) — dieselbe Funktion wie der Dialog im Projekt (main._freigabe_anlegen).
- „express_bestellen“: Express-Warenkorb zahlungspflichtig bestellen — dieselbe Funktion wie der Warenkorb
  (express_api.bestellung) mit Aufstellung, Preis und Bedingungen. Nur mit funktionen.EXPRESS.
- „auftrag“: einen Auftrag mit mehreren Schritten erteilen (Baustein 2, 14.10.2026) — Plan, Preise und Summe haelt der Server
  (inkluagent/auftraege.py, Werkzeug auftrag_vorbereiten); die Karte hat nur den Haken „per Mail melden“.
- „credits_kaufen“ und „abo_buchen“: Kaufen im Gespraech (Baustein 3, 14.10.2026, Schalter funktionen.AGENT_KAUFEN, nur mit
  Online-Zahlung) — dieselben Wege wie die Seite „Abo & Verbrauch“ (main._paket_checkout_url, main._abo_checkout_url). Der Knopf
  fuehrt zur Bezahlseite von Stripe; bezahlt wird nur dort. Wortlaut, Pflicht-Haken und Links wie auf der Seite (§ 312j BGB).

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

import billing
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
    links: tuple = ()              # weitere Links daneben: ((url, text-msgid), …)


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

# Auftrag mit mehreren Schritten (Baustein 2): Schritte, Preise und Summe stehen als Text und Zeilen auf der Karte; abgeschickt
# wird nur der Haken. Werkzeug und Erteilen: inkluagent/auftraege.py (auftrag_vorbereiten, karte_senden).
AUFTRAG = Kartentyp(
    art="auftrag", titel="Dein Auftrag", knopf="Auftrag erteilen",
    felder=(
        Feld("per_mail", "haken", "Zusätzlich per Mail an mein Konto melden, wenn der Auftrag fertig ist"),
    ))

# Kaufen im Gespraech (Baustein 3, Steve 14.10.2026: „man kann auch mit dem Bot Pakete und Credits buchen — aber man muss das
# ausdrücklich bestätigen“). Wortlaut wie auf „Abo & Verbrauch“ (abo.html): Knopf „Kostenpflichtig kaufen“ bzw. „Kostenpflichtig
# buchen“, Preis direkt beim Knopf (Paket: in der Auswahl; Abo: Plan, Laufzeit, Gesamtpreis und Verlaengerung fest auf der Karte),
# der Haken zum sofortigen Leistungsbeginn NICHT vorbelegt, Widerrufsbelehrung und Nutzungsbedingungen daneben.
def _paket_auswahl() -> tuple:
    """Paketgroessen mit Betrag aus billing.PAKET_PREISE — dieselbe Liste wie auf „Abo & Verbrauch“."""
    return tuple((str(groesse), f"{text} — {betrag}") for groesse, text, betrag in billing.pakete_fuer_anzeige())


CREDITS_KAUFEN = Kartentyp(
    art="credits_kaufen", titel="Zusatz-Credits kaufen", knopf="Kostenpflichtig kaufen",
    felder=(Feld("paket", "auswahl", "Paketgröße (Credits)", pflicht=True, auswahl=_paket_auswahl()),))

ZUSTIMMUNG_LEISTUNGSBEGINN = (
    "Ich verlange ausdrücklich, dass InkluDocs sofort nach der Buchung bereitsteht — also vor Ablauf der 14-tägigen "
    "Widerrufsfrist. Mir ist bekannt, dass ich bei einem Widerruf Wertersatz für die bis dahin verbrauchten Credits schulde und "
    "mein Widerrufsrecht bei vollständiger Vertragserfüllung erlischt.")
ABO_BUCHEN = Kartentyp(
    art="abo_buchen", titel="Abo buchen", knopf="Kostenpflichtig buchen",
    felder=(Feld("widerruf", "haken", ZUSTIMMUNG_LEISTUNGSBEGINN, pflicht=True,
                 link=("/widerruf", "Widerrufsbelehrung und Muster-Widerrufsformular"),
                 links=(("/nutzungsbedingungen", "Nutzungsbedingungen"),)),))

KARTENTYPEN = {k.art: k for k in (EINLADEN, EXPRESS_BESTELLEN, AUFTRAG, CREDITS_KAUFEN, ABO_BUCHEN)}


def kaufen_an() -> bool:
    """Kaufen im Gespraech: Schalter AGENT_KAUFEN, die Karten und die Online-Zahlung (Stripe konfiguriert)."""
    if not (an() and funktionen.an("AGENT_KAUFEN")):
        return False
    try:
        import stripe_zahlung
        return bool(stripe_zahlung.AKTIV)
    except Exception:  # noqa: BLE001 — ohne Stripe-Modul kein Kaufen
        return False


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
        if f.links:
            d["links"] = [{"url": u, "text": _(t)} for u, t in f.links]
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


def offene_karte(user_id: int, art: str, gespraech_id: Optional[int] = None) -> Optional[tuple]:
    """Die juengste offene Karte dieser Art fuer dieses Konto (und dieses Gespraech): (Kennung, Eintrag) oder None — fuer ein
    getipptes Ja (auftraege.auftrag_erteilen, 17.10.2026)."""
    with _SPERRE:
        _aufraeumen(time.time())
        offen = [(e["zeit"], k, e) for k, e in _KARTEN.items()
                 if e["user_id"] == int(user_id) and e["art"] == art and e["zustand"] == "offen"
                 and (gespraech_id is None or (e.get("kontext") or {}).get("gespraech_id") in (None, int(gespraech_id)))]
    if not offen:
        return None
    _zeit, k, e = max(offen, key=lambda x: x[0])
    return k, e


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


def gilt(kid: Optional[str]) -> bool:
    """Kennt der Server diese Karte noch als offen (oder gerade beim Abschicken)? Nach Ablauf (GUELTIG_S), Neustart oder
    Verdraengung nicht mehr — dann zeigt der Verlauf sie als „nicht mehr gültig“ (tools/ausgaben.karten_im_verlauf)."""
    with _SPERRE:
        _aufraeumen(time.time())
        e = _KARTEN.get(kid or "")
        return bool(e) and e["zustand"] in ("offen", "laeuft")


def _zustand(kid: str, zustand: str) -> None:
    with _SPERRE:
        e = _KARTEN.get(kid)
        if e:
            e["zustand"] = zustand


def karte(typ: Kartentyp, kid: str, werte: dict, _: Callable[[str], str], text: str = "", zeilen: Optional[list] = None,
          nummeriert: bool = False, hinweise: Optional[list] = None, knopf: Optional[str] = None) -> dict:
    """Anhang fuer die Antwort (art „formular“): so steht die Karte im Gespraech und im gespeicherten Verlauf. Abgeschickt wird
    jede Karte ueber POST /api/agent/karte, auf jeder Seite. nummeriert: die Zeilen sind Schritte (nummerierte Liste, ohne „1.“
    im Text); hinweise: Saetze nach der Liste als eigene Absaetze; knopf: eigene Beschriftung (etwa mit dem Preis), sonst die
    des Kartentyps (Auftraege, Pruefung N7)."""
    k = {"art": "formular", "karte": typ.art, "karten_id": kid, "titel": _(typ.titel), "text": text, "zeilen": list(zeilen or []),
         "felder": _uebersetzt(typ, werte, _), "knopf": knopf or _(typ.knopf), "zustand": "offen"}
    if nummeriert:
        k["nummeriert"] = True
    if hinweise:
        k["hinweise"] = [str(h) for h in hinweise]
    return k


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
                            "ins Gespräch: Aufstellung, Preis in Credits, Ansprechpartner, Telefon, Hinweise, Bedingungen. Will der "
                            "Nutzer Dokumente an den Express-Service geben, nimm sie mit document_ids gleich mit: Der Server legt "
                            "sie selbst in den Warenkorb, schick den Nutzer dafür nicht zum Knopf am Dokument. Bestellen kann NUR "
                            "der Nutzer mit dem Knopf der Karte."),
            "input_schema": {"type": "object", "properties": {
                "document_ids": {"type": "array", "items": {"type": "integer"}, "description": (
                    "Dokumente, die in den Express-Warenkorb sollen: ihre Kennungen aus dokument_stand (z. B. für „Dokument 2“). "
                    "Hineinlegen kostet nichts, der Nutzer kann sie wieder entfernen; die Karte zeigt dann den ganzen Warenkorb. "
                    "Ohne: die Karte für das, was schon im Warenkorb liegt.")},
                "ansprechpartner": {"type": "string"}, "telefon": {"type": "string"},
                "hinweise": {"type": "string", "description": "Hinweise des Nutzers für das Team, z. B. „Seite 1 prüfen“."}}},
        })
    if kaufen_an():
        defs += [{
            "name": "credits_kaufen_vorbereiten",
            "description": ("Legt dem Nutzer die Karte „Zusatz-Credits kaufen“ ins Gespräch: Paketgröße mit Preis (vorbelegt) und "
                            "Knopf „Kostenpflichtig kaufen“. Der Knopf führt zur Bezahlseite von Stripe; bezahlt wird nur dort — "
                            "du kaufst nichts. Nur, wenn der Nutzer Credits kaufen will."),
            "input_schema": {"type": "object", "properties": {
                # Ohne „enum“: Gemini nimmt Auswahllisten nur aus Texten (HTTP 400 bei Zahlen, Rauchtest Staging 14.10.2026);
                # der Server nimmt ohnehin nur eine Groesse aus billing.PAKET_PREISE (sonst die kleinste)
                "paket": {"type": "integer", "description": (
                    "Vorschlag für die Paketgröße in Credits: " + ", ".join(str(g) for g in sorted(billing.PAKET_PREISE))
                    + " (der Nutzer kann auf der Karte eine andere wählen).")}}},
        }, {
            "name": "abo_buchen_vorbereiten",
            "description": ("Legt dem Nutzer die Karte „Abo buchen“ ins Gespräch: Plan und Laufzeit fest, Gesamtpreis, Leistung, "
                            "Verlängerung, der Pflicht-Haken zum sofortigen Leistungsbeginn und Knopf „Kostenpflichtig buchen“. Der "
                            "Knopf führt zur Bezahlseite von Stripe; bezahlt wird nur dort. Nur, wenn der Nutzer ein Abo buchen will "
                            "und Plan und Laufzeit feststehen (frag sonst nach). Für Konten ohne Abo; Wechsel und Kündigung gehen auf "
                            "der Seite „Abo & Verbrauch“."),
            "input_schema": {"type": "object", "properties": {
                "plan": {"type": "string", "enum": sorted(billing.PLAN_PREISE_EUR)},
                "laufzeit_monate": {"type": "integer",   # ohne „enum“ (Gemini, siehe paket); der Server prueft PLAN_LAUFZEITEN
                                    "description": "1 = Monatsabo (monatlich kündbar), sonst feste Laufzeit 3, 6 oder 12 Monate."}},
                "required": ["plan", "laufzeit_monate"]},
        }]
    return defs


WERKZEUG_KARTE = {"einladen_vorbereiten": EINLADEN, "express_bestellen_vorbereiten": EXPRESS_BESTELLEN,
                  "credits_kaufen_vorbereiten": CREDITS_KAUFEN, "abo_buchen_vorbereiten": ABO_BUCHEN}
NAMEN = frozenset(WERKZEUG_KARTE)

PROMPT = """Formulare im Gespräch (Karten)

Will der Nutzer jemanden zum Prüfen einladen oder Dokumente an den Express-Service geben, bring ihm das Formular als Karte ins Gespräch (einladen_vorbereiten, express_bestellen_vorbereiten), statt ihn auf eine Seite zu schicken; die Dokumente legst du dabei selbst in den Express-Warenkorb (document_ids). Fülle vor, was er gesagt hat; erfinde nichts. Abschicken kann nur der Nutzer mit dem Knopf der Karte — frag also nicht nach einem Ja, sondern sag in einem Satz, was vorbelegt ist und welcher Knopf abschickt."""

# Ohne Express-Service (funktionen.EXPRESS aus) fehlt das Werkzeug express_bestellen_vorbereiten — der Text nennt es dann
# nicht (gefunden vom Werkzeug-Waechter, 15.10.2026)
PROMPT_OHNE_EXPRESS = PROMPT.replace(
    "Will der Nutzer jemanden zum Prüfen einladen oder Dokumente an den Express-Service geben, bring ihm das Formular als Karte "
    "ins Gespräch (einladen_vorbereiten, express_bestellen_vorbereiten), statt ihn auf eine Seite zu schicken; die Dokumente "
    "legst du dabei selbst in den Express-Warenkorb (document_ids).",
    "Will der Nutzer jemanden zum Prüfen einladen, bring ihm das Formular als Karte ins Gespräch (einladen_vorbereiten), statt "
    "ihn auf eine Seite zu schicken.")
assert PROMPT_OHNE_EXPRESS != PROMPT

# Kaufen im Gespraech (Baustein 3): nur mit kaufen_an() im Systemprompt (eigenes Schubfach „kaufen“, prompts/schubfaecher.py)
PROMPT_KAUFEN = """Kaufen im Gespräch

Will der Nutzer Credits nachkaufen oder ein Abo buchen, bring ihm die Karte (credits_kaufen_vorbereiten, abo_buchen_vorbereiten). Kaufen kann nur er selbst: Der Knopf der Karte führt zur Bezahlseite von Stripe, bezahlt wird nur dort. Biete Käufe nicht von dir aus an; reicht das Guthaben für etwas nicht, sag es und frag, ob du ihm die Karte bringen sollst. Ein Abo wechseln oder kündigen, Rechnungen und Zahlungsart gehen auf der Seite „Abo & Verbrauch“."""


def ausfuehren(name: str, args: dict, user_id: int, project_id: Optional[int], _t: Optional[Callable[[str], str]] = None) -> dict:
    """Werkzeug des Agenten -> Werkzeug-Ergebnis mit der Karte als anhang (agent_loop haengt sie an die Antwort)."""
    _ = _t or (lambda s: s)
    args = args if isinstance(args, dict) else {}
    try:
        if name == "einladen_vorbereiten":
            return _einladen_vorbereiten(args, user_id, project_id, _)
        if name == "express_bestellen_vorbereiten" and funktionen.an("EXPRESS"):
            return _express_vorbereiten(args, user_id, _)
        if name == "credits_kaufen_vorbereiten" and kaufen_an():
            return _credits_vorbereiten(args, user_id, project_id, _)
        if name == "abo_buchen_vorbereiten" and kaufen_an():
            return _abo_vorbereiten(args, user_id, project_id, _)
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
    # Steve 16.10.2026 (Michael: „Mach Du dies“ — der Agent konnte nichts in den Warenkorb legen): Dokumente legt der Server
    # selbst hinein, ueber dieselbe Funktion wie der Knopf am Dokument (nur Dokumente dieses Kontos, Grenzen, Dateityp). Das
    # Modell bekommt nur Zahlen; die Hinweise mit Dokumentnamen stehen auf der Karte (textContent), nicht im Werkzeug-Ergebnis.
    ids, hinzu, schon_drin = args.get("document_ids"), None, 0
    if ids not in (None, "", []):
        ids = ids if isinstance(ids, list) else [ids]
        if any(isinstance(x, bool) or not isinstance(x, (int, str)) for x in ids):
            return {"ok": False, "error": "document_ids: bitte nur Kennungen aus dokument_stand."}
        vorher = {p["document_id"] for p in express.warenkorb(user_id).get("positionen") or []}
        try:
            hinzu = express.dokumente_hinzufuegen(user_id, list(ids))
        except express.ExpressFehler as fe:
            return {"ok": False, "error": fe.text}
        schon_drin = len({int(x) for x in ids} & vorher)
    korb = express.warenkorb(user_id)
    if not korb.get("id") or not korb.get("positionen"):
        return {"ok": True, "result": {"warenkorb_leer": True, "nicht_moeglich": len(hinzu["hinweise"]) if hinzu else 0, "hinweis": (
            "Der Express-Warenkorb ist leer. Frag, welche Dokumente in den Express-Service sollen, und ruf dann "
            "express_bestellen_vorbereiten mit ihren Kennungen (document_ids) auf." if not hinzu else
            "Keines der Dokumente konnte in den Express-Warenkorb (heute nur PDF). Sag das dem Nutzer.")}}
    e = express.einstellungen()
    guthaben = billing.verfuegbare_credits(user_id)
    leistung = {l["schluessel"]: l["name"] for l in express.leistungen_liste(e)}
    zeilen = [_("„{name}“: {seiten} Seiten, {leistung}, {credits} Credits").format(
        name=p.get("dokument_name") or "", seiten=p.get("seiten") or 0, leistung=_(leistung.get(p.get("leistung"), "")),
        credits=p.get("credits") or 0) for p in korb["positionen"]]
    # None = unbegrenzt (Admin, Enterprise; billing.verfuegbare_credits). Vorher brach der Vergleich mit None ab und die Karte kam
    # nie — gefunden beim Rauchtest 16.10.2026 mit einem Admin-Konto.
    if guthaben is None:
        text = _("{d} Dokumente, {s} Seiten, zusammen {c} Credits. Dein Guthaben ist unbegrenzt. Fertig in {h} Stunden.").format(
            d=korb["dokumente"], s=korb["seiten"], c=korb["credits"], h=e["frist_stunden"])
    else:
        text = _("{d} Dokumente, {s} Seiten, zusammen {c} Credits. Dein Guthaben: {g} Credits. Fertig in {h} Stunden.").format(
            d=korb["dokumente"], s=korb["seiten"], c=korb["credits"], g=guthaben, h=e["frist_stunden"])
    if guthaben is not None and guthaben < korb["credits"]:
        knapp = {"guthaben_reicht_nicht": True, "credits": korb["credits"], "guthaben": guthaben,
                 "hinweis": "Das Guthaben reicht nicht. Nenne Preis und Guthaben und verweise auf „Abo & Verbrauch“."}
        if hinzu is not None:
            knapp.update(hineingelegt=hinzu["hinzugefuegt"], schon_im_warenkorb=schon_drin,
                         nicht_moeglich=len(hinzu["hinweise"]) - schon_drin)
        return {"ok": True, "result": knapp}
    nutzer = get_user_by_id(user_id) or {}
    # Was der Nutzer gesagt hat, geht vor; sonst Ansprechpartner und Telefon vom letzten Auftrag (Steve 16.10.2026: „dass er sich
    # das merkt“), sonst der Anzeigename
    letzte = express.letzte_kontaktdaten(user_id)
    werte = {"ansprechpartner": (_vorschlag(args.get("ansprechpartner"), 120) or letzte["ansprechpartner"][:120]
                                 or (nutzer.get("display_name") or "").strip()[:120]),
             "telefon": _vorschlag(args.get("telefon"), 40) or letzte["telefon"][:40],
             "hinweise": _vorschlag(args.get("hinweise"), 2000), "bedingungen": False}
    kid = _merken(user_id, EXPRESS_BESTELLEN, None, {
        "korb_id": int(korb["id"]), "erwartete_credits": int(korb["credits"]), "fassung": korb["fassung"],
        "idempotenz": "karte-" + secrets.token_hex(12)})
    ergebnis = {"karte": "express_bestellen", "dokumente": korb["dokumente"], "seiten": korb["seiten"],
                "credits": korb["credits"], "guthaben": guthaben, "vorbelegt": [k for k, v in werte.items() if v],
                "vom_letzten_auftrag": [k for k in ("ansprechpartner", "telefon")
                                        if letzte[k] and not _vorschlag(args.get(k), 200) and werte[k]],
                "hinweis": "Die Karte steht unter deiner Antwort. Nenne Preis und Frist in einem Satz; bestellen kann nur "
                           "der Nutzer mit „Zahlungspflichtig bestellen“, nachdem er die Bedingungen akzeptiert hat."}
    if hinzu is not None:
        nicht_moeglich = len(hinzu["hinweise"]) - schon_drin
        ergebnis.update(hineingelegt=hinzu["hinzugefuegt"], schon_im_warenkorb=schon_drin, nicht_moeglich=nicht_moeglich)
        if nicht_moeglich > 0:
            ergebnis["hinweis"] += " Nicht jedes Dokument konnte hinein (heute nur PDF); warum, steht auf der Karte."
    return {"ok": True, "result": ergebnis,
            "anhang": karte(EXPRESS_BESTELLEN, kid, werte, _, text=text, zeilen=zeilen,
                            hinweise=(hinzu or {}).get("hinweise") or None)}


def _zahl(n: int) -> str:
    """Credits mit Tausenderpunkt wie auf „Abo & Verbrauch“ (pakete_fuer_anzeige)."""
    return f"{int(n):,}".replace(",", ".")


def _euro(betrag: float) -> str:
    return f"{betrag:.2f}".replace(".", ",")


T_STRIPE = ("Bezahlt wird auf der sicheren Stripe-Seite. Welche Zahlungsarten dort erscheinen, hängt davon ab, was für dein Land "
            "freigeschaltet ist; Kartenzahlung geht immer.")
T_ENDPREISE = "Preise sind Endpreise. Nach § 19 UStG wird keine Umsatzsteuer erhoben und daher nicht ausgewiesen."


def _credits_vorbereiten(args: dict, user_id: int, project_id: Optional[int], _) -> dict:
    try:
        paket = int(args.get("paket") or min(billing.PAKET_PREISE))
    except (TypeError, ValueError):
        paket = min(billing.PAKET_PREISE)
    if paket not in billing.PAKET_PREISE:
        paket = min(billing.PAKET_PREISE)
    guthaben = billing.verfuegbare_credits(int(user_id))
    kid = _merken(user_id, CREDITS_KAUFEN, project_id, {})
    text = (_("Dein Guthaben: {n} Credits.").format(n=_zahl(guthaben)) + " " if guthaben is not None else "") + _(T_STRIPE)
    return {"ok": True,
            "result": {"karte": "credits_kaufen", "pakete": {g: p for g, p in billing.PAKET_PREISE.items()}, "vorbelegt": paket,
                       "guthaben": guthaben,
                       "hinweis": ("Die Karte steht unter deiner Antwort. Sag in einem Satz, welches Paket vorbelegt ist und was es "
                                   "kostet, und dass „Kostenpflichtig kaufen“ zur Bezahlseite von Stripe führt — bezahlt wird nur dort.")},
            "anhang": karte(CREDITS_KAUFEN, kid, {"paket": str(paket)}, _, text=text, zeilen=[_(T_ENDPREISE)])}


def _abo_zeilen(plan: str, monate: int, _) -> list[str]:
    """Bestellung wie „Deine Bestellung“ auf „Abo & Verbrauch“ (abo.html, aktualisiereSumme): Plan mit Leistung, Gesamtpreis,
    Leistung, Laufzeit und Verlaengerung — dieselben Saetze, dieselben Zahlen aus billing."""
    name = {"single": "Single", "team": "Team", "enterprise": "Enterprise"}[plan]
    credits, plaetze = _zahl(billing.PLAN_KONTINGENTE[plan]), billing.PLAN_SITZE.get(plan)
    rate = billing.preis_pro_monat(plan, monate)
    summe = _euro(rate * monate)
    alt, qi = billing.AKTIONS_PREISE["bild_generierung"], billing.AKTIONS_PREISE["quickinfo_generierung"]
    return [
        (_("{plan} — {credits} Credits im Monat, bis {plaetze} Personen").format(plan=name, credits=credits, plaetze=plaetze)
         if plaetze else _("{plan} — {credits} Credits im Monat, ein Konto").format(plan=name, credits=credits)),
        (_("Gesamtpreis: {summe} € pro Monat, monatlich kündbar.").format(summe=summe) if monate == 1 else
         _("Gesamtpreis: {summe} € für {monate} Monate — {rate} € pro Monat.").format(summe=summe, monate=monate, rate=_euro(rate))),
        (_("Leistung: {credits} Credits pro Monat für bis zu {plaetze} Personen (du eingerechnet). Ein Alt-Text kostet {alt} Credits, "
           "eine Quickinfo {qi} Credit.").format(credits=credits, plaetze=plaetze, alt=alt, qi=qi) if plaetze else
         _("Leistung: {credits} Credits pro Monat. Ein Alt-Text kostet {alt} Credits, eine Quickinfo {qi} Credit.").format(
             credits=credits, alt=alt, qi=qi)),
        (_("Monatsabo: {summe} € werden jeweils für einen Monat im Voraus abgebucht. Das Abo verlängert sich monatlich und ist "
           "jederzeit zum Ende des laufenden Monats kündbar.").format(summe=summe) if monate == 1 else
         _("Laufzeit {monate} Monate, im Voraus zu zahlen. Wenn du nicht zum Laufzeitende kündigst, läuft das Abo danach als "
           "Monatsabo zur Treuekondition weiter: {treue} € pro Monat, jederzeit monatlich kündbar.").format(
             monate=monate, treue=_euro(billing.treue_preis_eur(plan)))),
        _(T_ENDPREISE),
    ]


def _abo_vorbereiten(args: dict, user_id: int, project_id: Optional[int], _) -> dict:
    from database import get_user_by_id
    plan = str(args.get("plan") or "").strip().lower()
    try:
        monate = int(args.get("laufzeit_monate"))
    except (TypeError, ValueError):
        monate = 0
    if plan not in billing.PLAN_PREISE_EUR or monate not in billing.PLAN_LAUFZEITEN:
        return {"ok": False, "error": "Plan (single, team, enterprise) und Laufzeit (1, 3, 6 oder 12 Monate) fehlen. Frag nach."}
    nutzer = get_user_by_id(int(user_id)) or {}
    jetzt = billing.effektiver_plan(nutzer)
    if jetzt != "free":
        return {"ok": True, "result": {"karte": None, "plan_jetzt": jetzt, "hinweis": (
            "Das Konto hat schon ein Abo. Ein Wechsel oder eine Kündigung geht auf der Seite „Abo & Verbrauch“ — sag das und "
            "biete an, ihn hinzubringen.")}}
    kid = _merken(user_id, ABO_BUCHEN, project_id, {"plan": plan, "monate": monate})
    zeilen = _abo_zeilen(plan, monate, _)
    return {"ok": True,
            "result": {"karte": "abo_buchen", "plan": plan, "laufzeit_monate": monate,
                       "gesamtpreis_eur": round(billing.preis_pro_monat(plan, monate) * monate, 2),
                       "hinweis": ("Die Karte steht unter deiner Antwort. Sag in einem Satz Plan, Laufzeit und Gesamtpreis und dass "
                                   "der Nutzer den Haken zum sofortigen Leistungsbeginn setzen und dann „Kostenpflichtig buchen“ "
                                   "drücken muss; bezahlt wird auf der Seite von Stripe.")},
            "anhang": karte(ABO_BUCHEN, kid, {"widerruf": False}, _, text=_(T_STRIPE), zeilen=zeilen)}


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
        elif typ is AUFTRAG:
            from . import auftraege
            status, antwort = await auftraege.karte_senden(e, user, werte, sprache, _)
        elif typ in (CREDITS_KAUFEN, ABO_BUCHEN):
            status, antwort = await _kaufen_senden(typ, e, user, werte, absender, _)
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


T_WEITER = "Weiter zur Bezahlseite von Stripe. Bezahlt wird dort; danach kommst du zurück zu „Abo & Verbrauch“."


async def _kaufen_senden(typ: Kartentyp, e: dict, user: dict, werte: dict, absender: str, _) -> tuple[int, dict]:
    """Knopf „Kostenpflichtig kaufen“ / „Kostenpflichtig buchen“: derselbe Kern wie „Abo & Verbrauch“ (Stripe-Checkout). Die Seite
    leitet zu „weiter“ (nur eine Adresse von Stripe-Checkout) — bezahlt wird nur dort. Plan und Laufzeit kommen aus der Karte des
    Servers, nicht vom Browser; das Paket aus der geprueften Auswahl."""
    import asyncio
    from fastapi import HTTPException
    m = _main()
    if not kaufen_an():
        text = _("Diese Karte gilt nicht mehr. Bitte frag im Chat neu.")
        return 404, {"detail": text, "karte": {"zustand": "abgelaufen", "ergebnis": text}}
    nutzer = m.get_user_by_id(user["id"])
    try:
        if typ is CREDITS_KAUFEN:
            url = await asyncio.get_running_loop().run_in_executor(None, m._paket_checkout_url, nutzer, int(werte["paket"]))
        else:
            k = e["kontext"]
            url = await asyncio.get_running_loop().run_in_executor(None, m._abo_checkout_url, nutzer, k["plan"], int(k["monate"]),
                                                                   absender)
    except HTTPException as fehler:
        return fehler.status_code, {"detail": _(str(fehler.detail))}
    text = _(T_WEITER)
    return 200, {"ok": True, "text": text, "karte": {"zustand": "gesendet", "ergebnis": text}, "weiter": url}


def _im_verlauf_aendern(user_id: int, e: dict, kid: str, felder: dict) -> None:
    """Die Karte im gespeicherten Gespraech auf ihren Zustand stellen, damit sie nach dem Neuladen nicht wieder offen steht."""
    from . import storage
    try:
        storage.karte_aktualisieren(e.get("project_id") or 0, kid, felder, user_id=user_id)
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).exception("Karte %s im Verlauf nicht aktualisiert", kid)
