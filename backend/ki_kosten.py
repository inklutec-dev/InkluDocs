"""KI-KOSTEN (05.10.2026, Steve): JEDER KI-Aufruf als eigene Zeile — wer, wofuer, welches Modell, wie viele
Tokens, was er gekostet hat. Grundlage fuer den Bereich „KI-Kosten“ der Verwaltung (Monat, Zweck, Kunde bis
zum einzelnen Bild) und fuer die Kundenseite, die bis dahin nur eine Pauschale je Credit kannte
(billing.KOSTEN_PRO_CREDIT_EUR).

Warum hier und nicht an den Aufrufstellen: Erfasst wird an den drei Stellen, an denen InkluDocs ueberhaupt mit
einem KI-Anbieter spricht (pipelines/v4/gemini_client, pipelines/v4/bedrock_client + openai_client,
inkluagent/providers). Jede kuenftige KI-Funktion, die ueber diese Clients laeuft, zaehlt damit von selbst mit.

Wer und wofuer: Die Clients kennen weder Kunde noch Projekt. Das reicht ein Kontext (contextvars) durch, den
die Einstiegspunkte setzen (Sammellauf je Bild, Neu-Generieren, Chat, Quickinfos, Uebersetzen, API …):
    with ki_kosten.kontext(user_id=..., project_id=..., image_id=...):
        ...
Den Zweck setzt die fachliche Funktion selbst (pdf_processor.generate_alt_text -> 'alttext',
formular_ki -> 'quickinfo' …), damit z. B. ein Alt-Text, den der Chatbot erzeugen laesst, als Alt-Text zaehlt.
Damit der Kontext in Threads ankommt, ist der Standard-Executor des Servers ein KontextExecutor (kopiert den
Kontext bei submit); eigene ThreadPools nutzen mit_kontext(). Fehlt der Kontext trotzdem, wird der Aufruf
„ohne Zuordnung“ gezaehlt — nie verworfen.

Was er kostet: Tokens aus der Antwort des Anbieters (Gemini usageMetadata, Bedrock usage) mal Preisliste
(USD je 1 Mio. Tokens) mal Wechselkurs. Die Preisliste ist pflegbar (Verwaltung, system_kv 'ki_preise') und
traegt Quelle und Stand. Kosten werden beim Aufruf mit dem DANN gueltigen Preis festgeschrieben — eine
spaetere Preisaenderung schreibt die Vergangenheit nicht um. Kein Preis bekannt -> Kosten NULL („unbekannt“),
nie 0. Denk-Tokens rechnet Google als Ausgabe ab, darum auch hier.

Nie-Stoeren-Garantie: erfasse() wirft NIE. Ein Fehler beim Mitschreiben darf keinen KI-Lauf brechen (Log-Warnung,
fertig). Ohne Datenbankdatei (Eval-Laeufe auf dem Host) wird gar nichts geschrieben.
"""
from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import inspect
import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

log = logging.getLogger("ki_kosten")

# ─── Kontext ─────────────────────────────────────────────────────────────

_KONTEXT: contextvars.ContextVar = contextvars.ContextVar("ki_kosten_kontext", default=None)
FELDER = ("user_id", "konto_user_id", "project_id", "document_id", "image_id", "zweck")

# Zwecke, wie sie in der Verwaltung erscheinen (Reihenfolge = Anzeige). Neue Zwecke brauchen nur hier eine Zeile.
ZWECKE = {
    "alttext": "Alt-Texte",
    "chatbot": "Chatbot",
    "quickinfo": "Quickinfos",
    "uebersetzung": "Übersetzen",
    "tagging_ki": "Tagging mit KI",
    "ki_pruefung": "KI-Prüfung",
    "unbekannt": "Sonstiges",
}


def aktuell() -> dict:
    """Der gerade gesetzte Kontext (Kopie)."""
    return dict(_KONTEXT.get() or {})


def _gemischt(felder: dict) -> dict:
    neu = aktuell()
    for k, v in felder.items():
        if k not in FELDER:
            raise ValueError(f"unbekanntes Kontextfeld {k!r}")
        neu[k] = v
    return neu


def setze(**felder) -> None:
    """Kontext fuer den Rest der laufenden Aufgabe (asyncio-Task bzw. Thread) ergaenzen. Fuer Schleifen in
    Hintergrund-Aufgaben (je Bild ein neuer image_id), wo ein with-Block unhandlich waere."""
    _KONTEXT.set(_gemischt(felder))


@contextlib.contextmanager
def kontext(**felder):
    """Kontext fuer einen Block ergaenzen; danach gilt wieder der vorherige."""
    token = _KONTEXT.set(_gemischt(felder))
    try:
        yield
    finally:
        _KONTEXT.reset(token)


def fuer_zweck(zweck: str):
    """Dekorator fuer die fachliche Funktion, die eine KI-Leistung erbringt: setzt den Zweck fuer ihre Laufzeit
    (Kunde, Projekt und Bild kommen weiter vom Aufrufer)."""
    if zweck not in ZWECKE:
        raise ValueError(f"unbekannter Zweck {zweck!r}")

    def deko(fn):
        @functools.wraps(fn)
        def innen(*args, **kwargs):
            with kontext(zweck=zweck):
                return fn(*args, **kwargs)
        return innen
    return deko


def mit_kunde(fn):
    """Dekorator fuer SYNCHRONE Arbeitsfunktionen mit den Parametern user_id/project_id/document_id (Tagging-Lauf,
    Pruefung …): setzt daraus den Kontext fuer ihre Laufzeit — egal, ob sie im Executor, in einem eigenen Thread
    oder direkt laufen."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def innen(*args, **kwargs):
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
        except TypeError:
            b = {}
        felder = {k: b[k] for k in ("user_id", "project_id", "document_id") if k in b}
        with kontext(image_id=None, **felder):
            return fn(*args, **kwargs)
    return innen


def mit_kontext(fn):
    """fn so verpacken, dass sie im JETZT gueltigen Kontext laeuft — fuer eigene ThreadPools/Threads.
    Je Aufruf ein eigener Kontext (ein Context-Objekt darf nicht in zwei Threads gleichzeitig laufen)."""
    ctx = contextvars.copy_context()

    def _lauf(*args, **kwargs):
        return ctx.copy().run(fn, *args, **kwargs)
    return _lauf


class KontextExecutor(ThreadPoolExecutor):
    """ThreadPool, der den Kontext des Aufrufers in den Thread mitnimmt. Wird als Standard-Executor des Servers
    gesetzt (main.lifespan): loop.run_in_executor(None, …) kopiert den Kontext sonst NICHT (asyncio.to_thread
    taete es, run_in_executor nicht) — ohne das kaeme kein Kunde und kein Projekt bei den KI-Clients an."""

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


# ─── Umgebung ────────────────────────────────────────────────────────────

def umgebung() -> str:
    """'prod' | 'staging' | 'demo' | 'test' — damit Tests und Demo getrennt ausgewiesen werden koennen."""
    wert = (os.environ.get("INKLUDOCS_UMGEBUNG") or "").strip().lower()
    if wert in ("prod", "staging", "demo", "test"):
        return wert
    if (os.environ.get("DEMO_MODE") or "off").strip().lower() in ("on", "true", "1", "yes"):
        return "demo"
    if "staging" in (os.environ.get("BASE_URL") or ""):
        return "staging"
    return "prod"


# ─── Preisliste ──────────────────────────────────────────────────────────
# USD je 1 Mio. Tokens. Je Modell eine Liste von Preisstufen mit Gueltigkeitsbeginn „ab“ (ISO-Datum): es gilt die
# juengste Stufe mit ab <= heute. Felder je Stufe:
#   ein              Eingabe (Text und Bild), nicht aus dem Zwischenspeicher
#   aus              Ausgabe INKLUSIVE Denk-Tokens (so rechnen Google und Anthropic ab)
#   cache            Eingabe aus dem Zwischenspeicher (Gemini Context Caching / Anthropic Cache-Lesen)
#   cache_schreiben  nur Anthropic: Eingabe, die in den Zwischenspeicher geschrieben wird
#   grenze, ein_lang, aus_lang, cache_lang   Gemini Pro: andere Preise ab einer Eingabe ueber `grenze` Tokens
# Quellen, abgerufen am 05.10.2026:
#   Gemini: https://ai.google.dev/gemini-api/docs/pricing (Bezahlstufe „Standard“; Vertex AI fuehrt dieselben
#     Listenpreise, Rechnung bei uns in EUR). Gemini 3.8 Flash hat einen Einfuehrungspreis bis 31.12.2026,
#     ab 01.01.2027 doppelt so hoch — beides steht als eigene Stufe drin.
#   Claude Sonnet 4.6 ueber Bedrock: EU-Regionsinferenz (Kennung eu.…) kostet 10 % mehr als die globale
#     Inferenz (3,30/16,50 statt 3,00/15,00 USD); Cache-Lesen 10 %, Cache-Schreiben 125 % des Eingabepreises.
# Wechselkurs: EZB-Referenzkurs vom 02.10.2026, 1 EUR = 1,1225 USD -> 1 USD = 0,8909 EUR.
PREISE_STANDARD = {
    "usd_eur": 0.8909,
    "usd_eur_quelle": "EZB-Referenzkurs 02.10.2026 (1 EUR = 1,1225 USD)",
    "modelle": {
        "gemini-3.1-pro-preview": {
            "quelle": "ai.google.dev/gemini-api/docs/pricing, abgerufen 05.10.2026",
            "stufen": [{"ab": "2026-01-01", "ein": 2.00, "aus": 12.00, "cache": 0.20,
                        "grenze": 200000, "ein_lang": 4.00, "aus_lang": 18.00, "cache_lang": 0.40}],
        },
        "gemini-3.8-flash": {
            "quelle": "ai.google.dev/gemini-api/docs/pricing, abgerufen 05.10.2026 (Einführungspreis bis 31.12.2026)",
            "stufen": [{"ab": "2026-01-01", "ein": 0.75, "aus": 3.75, "cache": 0.075},
                       {"ab": "2027-01-01", "ein": 1.50, "aus": 7.50, "cache": 0.15}],
        },
        "eu.anthropic.claude-sonnet-4-6": {
            "quelle": "AWS Bedrock, Anthropic Claude Sonnet 4.6, EU-Regionsinferenz (+10 %), Stand 05.10.2026",
            "stufen": [{"ab": "2026-01-01", "ein": 3.30, "aus": 16.50, "cache": 0.33, "cache_schreiben": 4.125}],
        },
        "claude-sonnet-4-6": {
            "quelle": "AWS Bedrock, Anthropic Claude Sonnet 4.6, globale Inferenz, Stand 05.10.2026",
            "stufen": [{"ab": "2026-01-01", "ein": 3.00, "aus": 15.00, "cache": 0.30, "cache_schreiben": 3.75}],
        },
    },
}

PREIS_FELDER = ("ein", "aus", "cache", "cache_schreiben", "ein_lang", "aus_lang", "cache_lang")
PREIS_MAX = 1000.0                       # USD je 1 Mio. Tokens — alles darueber ist ein Tippfehler
_MODELL_RE = re.compile(r"^[a-z0-9][a-z0-9._:\-]{1,99}$")
_DATUM_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_KV_PREISE = "ki_preise"
_preis_cache: dict = {"wert": None, "zeit": 0.0}
_preis_lock = threading.Lock()
_PREIS_CACHE_S = 60


def _db_pfad() -> str:
    return os.environ.get("INKLUDOCS_DB", "/app/data/inkludocs.db")


def _verbindung():
    """Eigene kurze Verbindung; None, wenn es die Datenbank (noch) nicht gibt — dann wird nichts angelegt."""
    pfad = _db_pfad()
    if not os.path.exists(pfad):
        return None
    conn = sqlite3.connect(pfad, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def preise() -> dict:
    """Gueltige Preisliste: gespeicherte Fassung aus der Verwaltung, sonst PREISE_STANDARD. 60 s zwischengespeichert."""
    with _preis_lock:
        if _preis_cache["wert"] is not None and time.time() - _preis_cache["zeit"] < _PREIS_CACHE_S:
            return copy.deepcopy(_preis_cache["wert"])
    wert = None
    try:
        conn = _verbindung()
        if conn is not None:
            try:
                r = conn.execute("SELECT value FROM system_kv WHERE key = ?", (_KV_PREISE,)).fetchone()
            finally:
                conn.close()
            if r and r["value"]:
                wert = json.loads(r["value"])
                pruefe_preisliste(wert)
    except Exception:  # noqa: BLE001 — kaputte gespeicherte Liste: Standard nehmen, nicht abbrechen
        log.exception("Gespeicherte KI-Preisliste unlesbar — Standardliste gilt")
        wert = None
    if wert is None:
        wert = copy.deepcopy(PREISE_STANDARD)
    with _preis_lock:
        _preis_cache["wert"], _preis_cache["zeit"] = wert, time.time()
    return copy.deepcopy(wert)


def _zahl(wert, feld: str, minimum: float = 0.0, maximum: float = PREIS_MAX) -> float:
    if isinstance(wert, bool) or not isinstance(wert, (int, float)):
        raise ValueError(f"{feld}: bitte eine Zahl")
    w = float(wert)
    if not (minimum <= w <= maximum):
        raise ValueError(f"{feld}: Wert außerhalb des erlaubten Bereichs")
    return w


def pruefe_preisliste(liste: dict) -> dict:
    """Wirft ValueError mit verstaendlichem Text, wenn die Liste nicht taugt. Rueckgabe: die Liste."""
    if not isinstance(liste, dict):
        raise ValueError("Preisliste ungültig")
    _zahl(liste.get("usd_eur"), "Wechselkurs", 0.2, 5.0)
    modelle = liste.get("modelle")
    if not isinstance(modelle, dict) or not modelle:
        raise ValueError("Preisliste ohne Modelle")
    if len(modelle) > 100:
        raise ValueError("Zu viele Modelle")
    for name, m in modelle.items():
        if not isinstance(name, str) or not _MODELL_RE.match(name):
            raise ValueError(f"Modellkennung ungültig: {str(name)[:60]}")
        stufen = (m or {}).get("stufen")
        if not isinstance(stufen, list) or not stufen or len(stufen) > 50:
            raise ValueError(f"{name}: keine Preisstufen")
        for s in stufen:
            if not isinstance(s, dict) or not _DATUM_RE.match(str(s.get("ab") or "")):
                raise ValueError(f"{name}: „gültig ab“ bitte als Datum JJJJ-MM-TT")
            for feld in ("ein", "aus"):
                _zahl(s.get(feld), f"{name} {feld}")
            for feld in PREIS_FELDER[2:]:
                if s.get(feld) is not None:
                    _zahl(s.get(feld), f"{name} {feld}")
            if s.get("grenze") is not None:
                _zahl(s.get("grenze"), f"{name} grenze", 1, 100_000_000)
        if len(str((m or {}).get("quelle") or "")) > 300:
            raise ValueError(f"{name}: Quelle zu lang")
    return liste


def speichere_preise(liste: dict) -> None:
    """Preisliste dauerhaft speichern (Verwaltung, nur Voll-Admins — die Rechte prueft der Endpunkt)."""
    pruefe_preisliste(liste)
    conn = _verbindung()
    if conn is None:
        raise RuntimeError("Datenbank fehlt")
    try:
        conn.execute("INSERT INTO system_kv (key, value, updated_at) VALUES (?, ?, datetime('now')) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                     (_KV_PREISE, json.dumps(liste, ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()
    with _preis_lock:
        _preis_cache["wert"], _preis_cache["zeit"] = None, 0.0


def _modell_schluessel(modell: str, liste: dict):
    """Eintrag der Preisliste zu einer Modellkennung: erst exakt, dann ohne Regions-/Herstellervorsatz und
    Versionsendung (eu.anthropic.claude-sonnet-4-6-v1:0 -> claude-sonnet-4-6)."""
    modelle = liste.get("modelle") or {}
    m = (modell or "").strip().lower()
    ohne_version = re.sub(r"-v\d+(:\d+)?$", "", m)
    ohne_region = re.sub(r"^(eu|us|apac|global|jp|au)\.", "", ohne_version)
    for kandidat in (m, ohne_version, ohne_region, re.sub(r"^anthropic\.", "", ohne_region)):
        if kandidat in modelle:
            return kandidat
    return None


def preisstufe(modell: str, liste: dict = None, heute: str = None):
    """Die heute gueltige Preisstufe eines Modells (dict) oder None."""
    liste = liste or preise()
    schluessel = _modell_schluessel(modell, liste)
    if not schluessel:
        return None
    heute = heute or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    gueltig = [s for s in liste["modelle"][schluessel].get("stufen") or [] if str(s.get("ab")) <= heute]
    if not gueltig:
        return None
    return max(gueltig, key=lambda s: str(s.get("ab")))


def kosten_usd(modell: str, n: dict, liste: dict = None, heute: str = None):
    """Kosten in USD fuer die Token-Mengen n = {ein, aus, denk, cache, cache_schreiben, eingabe_gesamt};
    None, wenn fuer das Modell kein Preis bekannt ist."""
    stufe = preisstufe(modell, liste, heute)
    if stufe is None:
        return None
    lang = bool(stufe.get("grenze")) and int(n.get("eingabe_gesamt") or 0) > float(stufe["grenze"])

    def p(feld: str) -> float:
        if lang and stufe.get(feld + "_lang") is not None:
            return float(stufe[feld + "_lang"])
        if stufe.get(feld) is not None:
            return float(stufe[feld])
        # Kein eigener Cache-Preis: wie normale Eingabe (vorsichtig, eher zu hoch als zu niedrig).
        return p("ein") if feld in ("cache", "cache_schreiben") else 0.0

    summe = (int(n.get("ein") or 0) * p("ein") + int(n.get("cache") or 0) * p("cache")
             + int(n.get("cache_schreiben") or 0) * p("cache_schreiben")
             + (int(n.get("aus") or 0) + int(n.get("denk") or 0)) * p("aus"))
    return summe / 1_000_000


# ─── Erfassen ────────────────────────────────────────────────────────────

def _int(wert) -> int:
    try:
        return max(0, int(wert or 0))
    except (TypeError, ValueError):
        return 0


def nutzung_gemini(antwort: dict) -> dict:
    """Gemini usageMetadata -> Token-Mengen. promptTokenCount enthaelt die gecachten Tokens; die werden
    getrennt (guenstiger) gerechnet. toolUsePromptTokenCount (Werkzeug-Eingaben) zaehlt als Eingabe."""
    u = (antwort or {}).get("usageMetadata") or {}
    prompt = _int(u.get("promptTokenCount")) + _int(u.get("toolUsePromptTokenCount"))
    cache = min(_int(u.get("cachedContentTokenCount")), prompt)
    return {"ein": prompt - cache, "cache": cache, "cache_schreiben": 0,
            "aus": _int(u.get("candidatesTokenCount")), "denk": _int(u.get("thoughtsTokenCount")),
            "eingabe_gesamt": prompt}


def nutzung_anthropic(payload: dict) -> dict:
    """Anthropic-Nachrichtenformat (Bedrock invoke_model): input_tokens ist OHNE Cache-Anteile."""
    u = (payload or {}).get("usage") or {}
    ein, cache, schreiben = _int(u.get("input_tokens")), _int(u.get("cache_read_input_tokens")), _int(u.get("cache_creation_input_tokens"))
    return {"ein": ein, "cache": cache, "cache_schreiben": schreiben, "aus": _int(u.get("output_tokens")), "denk": 0,
            "eingabe_gesamt": ein + cache + schreiben}


def nutzung_converse(resp: dict) -> dict:
    """Bedrock Converse: inputTokens ohne Cache-Anteile, Cache getrennt."""
    u = (resp or {}).get("usage") or {}
    ein, cache, schreiben = _int(u.get("inputTokens")), _int(u.get("cacheReadInputTokens")), _int(u.get("cacheWriteInputTokens"))
    return {"ein": ein, "cache": cache, "cache_schreiben": schreiben, "aus": _int(u.get("outputTokens")), "denk": 0,
            "eingabe_gesamt": ein + cache + schreiben}


def nutzung_openai(antwort: dict) -> dict:
    """OpenAI (Responses-API input_tokens/output_tokens, aeltere Chat-API prompt_tokens/completion_tokens): die
    Ausgabe enthaelt die Denk-Tokens dort schon — nicht doppelt zaehlen."""
    u = (antwort or {}).get("usage") or {}
    prompt = _int(u.get("input_tokens") if "input_tokens" in u else u.get("prompt_tokens"))
    details = u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}
    cache = min(_int(details.get("cached_tokens")), prompt)
    aus = _int(u.get("output_tokens") if "output_tokens" in u else u.get("completion_tokens"))
    return {"ein": prompt - cache, "cache": cache, "cache_schreiben": 0, "aus": aus, "denk": 0, "eingabe_gesamt": prompt}


_warnung_gezeigt = {"tabelle": False}


def erfasse(anbieter: str, modell: str, nutzung: dict, *, schritt: str = "", erfolg: bool = True,
            fehler: str = "", standard_zweck: str = "unbekannt") -> None:
    """EINEN KI-Aufruf festschreiben. Wirft nie (siehe Modulkopf)."""
    try:
        k = aktuell()
        user_id = k.get("user_id")
        konto = k.get("konto_user_id")
        if user_id and not konto:
            try:
                import billing
                konto = billing._konto_fuer(int(user_id))
            except Exception:  # noqa: BLE001
                konto = user_id
        liste = preise()
        usd = kosten_usd(modell, nutzung, liste)
        kurs = float(liste.get("usd_eur") or 0)
        eur_cent = None if usd is None else usd * kurs * 100
        conn = _verbindung()
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO ki_aufrufe (umgebung, user_id, konto_user_id, project_id, document_id, image_id, zweck, "
                "schritt, anbieter, modell, tokens_ein, tokens_cache, tokens_cache_schreiben, tokens_aus, tokens_denk, "
                "kosten_usd, kosten_eur_cent, kurs_usd_eur, erfolg, fehler) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (umgebung(), user_id, konto, k.get("project_id"), k.get("document_id"), k.get("image_id"),
                 (k.get("zweck") or standard_zweck or "unbekannt")[:40], (schritt or "")[:80], (anbieter or "")[:40],
                 (modell or "")[:100], _int(nutzung.get("ein")), _int(nutzung.get("cache")),
                 _int(nutzung.get("cache_schreiben")), _int(nutzung.get("aus")), _int(nutzung.get("denk")),
                 usd, eur_cent, kurs if usd is not None else None, 1 if erfolg else 0, (fehler or "")[:200]))
            conn.commit()
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            if not _warnung_gezeigt["tabelle"]:
                _warnung_gezeigt["tabelle"] = True
                log.warning("ki_aufrufe fehlt (Datenbank noch nicht migriert) — KI-Kosten werden nicht erfasst")
        else:
            log.warning("KI-Kosten nicht erfasst (%s, %s): %s", anbieter, modell, e)
    except Exception as e:  # noqa: BLE001
        log.warning("KI-Kosten nicht erfasst (%s, %s): %r", anbieter, modell, e)


def erfasse_gemini(modell: str, antwort: dict, *, schritt: str = "", erfolg: bool = True, fehler: str = "",
                   standard_zweck: str = "unbekannt") -> None:
    try:
        n = nutzung_gemini(antwort)
    except Exception:  # noqa: BLE001
        n = {}
    erfasse("gemini", modell, n, schritt=schritt, erfolg=erfolg, fehler=fehler, standard_zweck=standard_zweck)


def erfasse_anthropic(modell: str, payload: dict, *, schritt: str = "", erfolg: bool = True, fehler: str = "",
                      standard_zweck: str = "unbekannt") -> None:
    try:
        n = nutzung_anthropic(payload)
    except Exception:  # noqa: BLE001
        n = {}
    erfasse("bedrock", modell, n, schritt=schritt, erfolg=erfolg, fehler=fehler, standard_zweck=standard_zweck)


def erfasse_converse(modell: str, resp: dict, *, schritt: str = "", erfolg: bool = True, fehler: str = "",
                     standard_zweck: str = "unbekannt") -> None:
    try:
        n = nutzung_converse(resp)
    except Exception:  # noqa: BLE001
        n = {}
    erfasse("bedrock", modell, n, schritt=schritt, erfolg=erfolg, fehler=fehler, standard_zweck=standard_zweck)


def erfasse_openai(modell: str, antwort: dict, *, schritt: str = "", erfolg: bool = True, fehler: str = "",
                   standard_zweck: str = "unbekannt") -> None:
    try:
        n = nutzung_openai(antwort)
    except Exception:  # noqa: BLE001
        n = {}
    erfasse("openai", modell, n, schritt=schritt, erfolg=erfolg, fehler=fehler, standard_zweck=standard_zweck)


# ─── Auswertung fuer die Verwaltung ──────────────────────────────────────
# Monatsgrenzen in deutscher Zeit wie beim Umsatz (umsatz.zeitraum). Betraege in Cent (Gleitkomma — ein
# einzelner Aufruf kostet Bruchteile eines Cents; gerundet wird erst in der Anzeige).

def _db():
    from database import get_db
    return get_db()


def messbeginn():
    """Zeitpunkt (UTC-Text) des ersten erfassten Aufrufs oder None."""
    conn = _db()
    try:
        return conn.execute("SELECT MIN(created_at) FROM ki_aufrufe").fetchone()[0]
    finally:
        conn.close()


def zeitraeume() -> list:
    """Monate mit erfassten Aufrufen, neuester zuerst; der laufende Monat steht immer drin."""
    import umsatz
    jetzt = umsatz.jetzt_lokal()
    erster = messbeginn()
    start = (jetzt.year, jetzt.month)
    if erster:
        lok = umsatz.lokal(erster)
        start = min(start, (int(lok[:4]), int(lok[5:7])))
    out = []
    j, m = jetzt.year, jetzt.month
    conn = _db()
    try:
        while (j, m) >= start:
            von, bis = umsatz.zeitraum(j, m)
            r = conn.execute("SELECT COUNT(*) AS n, SUM(kosten_eur_cent) AS c FROM ki_aufrufe "
                             "WHERE created_at >= ? AND created_at < ?", (von, bis)).fetchone()
            out.append({"jahr": j, "monat": m, "aufrufe": int(r["n"] or 0), "kosten_cent": float(r["c"] or 0)})
            m -= 1
            if m == 0:
                j, m = j - 1, 12
    finally:
        conn.close()
    return out


def _bedingung(von: str, bis: str, umgebungen) -> tuple:
    sql = "a.created_at >= ? AND a.created_at < ?"
    werte = [von, bis]
    if umgebungen:
        sql += " AND a.umgebung IN (" + ",".join("?" * len(umgebungen)) + ")"
        werte += list(umgebungen)
    return sql, werte


def _summen(row) -> dict:
    return {"aufrufe": int(row["n"] or 0), "kosten_cent": float(row["c"] or 0),
            "unbekannt": int(row["u"] or 0), "fehlgeschlagen": int(row["f"] or 0),
            "tokens": int(row["t"] or 0)}


_SUMMEN_SQL = ("COUNT(*) AS n, SUM(a.kosten_eur_cent) AS c, SUM(CASE WHEN a.kosten_eur_cent IS NULL THEN 1 ELSE 0 END) AS u, "
               "SUM(CASE WHEN a.erfolg = 0 THEN 1 ELSE 0 END) AS f, "
               "SUM(a.tokens_ein + a.tokens_cache + a.tokens_cache_schreiben + a.tokens_aus + a.tokens_denk) AS t")


def monatsbericht(jahr: int, monat: int, umgebungen=None) -> dict:
    """Alles fuer die Seite „KI-Kosten“ eines Monats. umgebungen: None = alle."""
    import umsatz
    von, bis = umsatz.zeitraum(int(jahr), int(monat))
    bed, werte = _bedingung(von, bis, umgebungen)
    conn = _db()
    try:
        gesamt = _summen(conn.execute(f"SELECT {_SUMMEN_SQL} FROM ki_aufrufe a WHERE {bed}", werte).fetchone())
        nach_zweck = [dict(_summen(r), zweck=r["zweck"]) for r in conn.execute(
            f"SELECT a.zweck, {_SUMMEN_SQL} FROM ki_aufrufe a WHERE {bed} GROUP BY a.zweck ORDER BY c DESC", werte)]
        nach_modell = [dict(_summen(r), modell=r["modell"], anbieter=r["anbieter"]) for r in conn.execute(
            f"SELECT a.anbieter, a.modell, {_SUMMEN_SQL} FROM ki_aufrufe a WHERE {bed} "
            "GROUP BY a.anbieter, a.modell ORDER BY c DESC", werte)]
        nach_umgebung = [dict(_summen(r), umgebung=r["umgebung"]) for r in conn.execute(
            f"SELECT a.umgebung, {_SUMMEN_SQL} FROM ki_aufrufe a WHERE {bed} GROUP BY a.umgebung ORDER BY c DESC", werte)]
        kunden = []
        for r in conn.execute(
                f"SELECT a.konto_user_id AS konto, {_SUMMEN_SQL}, "
                "COALESCE(NULLIF(TRIM(u.display_name), ''), u.email) AS name, u.email AS email "
                f"FROM ki_aufrufe a LEFT JOIN users u ON u.id = a.konto_user_id WHERE {bed} "
                "GROUP BY a.konto_user_id ORDER BY c DESC", werte):
            kunden.append(dict(_summen(r), konto_user_id=r["konto"], name=r["name"] or "", email=r["email"] or "",
                               konto_geloescht=bool(r["konto"]) and not r["email"]))
        # Credits und Umsatz je Konto im selben Zeitraum (Umsatz wie auf der Umsatz-Seite: ohne Bonus und Ruecklastschrift).
        # Express-Credits (05.10.2026) bezahlen Handarbeit, keine KI — sie wuerden die Kosten je Credit verwaessern.
        credits = {r["k"]: int(r["s"] or 0) for r in conn.execute(
            "SELECT konto_user_id AS k, SUM(credits) AS s FROM usage_events "
            "WHERE created_at >= ? AND created_at < ? AND quelle != 'express' GROUP BY konto_user_id", (von, bis))}
        umsatz_je = {r["k"]: int(r["s"] or 0) for r in conn.execute(
            f"SELECT konto_user_id AS k, SUM(betrag_cent) AS s FROM buchungen WHERE {umsatz._ZAEHLT} "
            "AND gebucht_am >= ? AND gebucht_am < ? GROUP BY konto_user_id", (von, bis))}
        credits_gesamt = sum(credits.values())
        umsatz_cent = umsatz._summe(conn, von, bis)
    finally:
        conn.close()
    for k in kunden:
        k["credits"] = credits.get(k["konto_user_id"], 0) if k["konto_user_id"] else 0
        k["umsatz_cent"] = umsatz_je.get(k["konto_user_id"], 0) if k["konto_user_id"] else 0
    # Kosten je Credit: nur Kundenkosten (ohne „ohne Zuordnung“) durch die Credits, die dafuer verbraucht wurden.
    kunden_kosten = sum(k["kosten_cent"] for k in kunden if k["konto_user_id"])
    return {
        "jahr": int(jahr), "monat": int(monat), "von": von, "bis": bis,
        "gesamt": gesamt, "umsatz_cent": umsatz_cent, "bleibt_cent": umsatz_cent - gesamt["kosten_cent"],
        "credits": credits_gesamt,
        "kosten_je_credit_cent": (kunden_kosten / credits_gesamt) if credits_gesamt else None,
        "nach_zweck": nach_zweck, "nach_modell": nach_modell, "nach_umgebung": nach_umgebung, "kunden": kunden,
    }


def kunde_bericht(konto_id, jahr: int, monat: int, umgebungen=None) -> dict:
    """Projekte eines Kontos (oder „ohne Zuordnung“ bei konto_id None) mit ihren Kosten im Monat."""
    import umsatz
    von, bis = umsatz.zeitraum(int(jahr), int(monat))
    bed, werte = _bedingung(von, bis, umgebungen)
    if konto_id is None:
        bed += " AND a.konto_user_id IS NULL"
    else:
        bed += " AND a.konto_user_id = ?"
        werte.append(int(konto_id))
    conn = _db()
    try:
        projekte = [dict(_summen(r), project_id=r["pid"], name=r["name"] or "", geloescht=bool(r["pid"]) and r["vorhanden"] is None)
                    for r in conn.execute(
                        f"SELECT a.project_id AS pid, {_SUMMEN_SQL}, "
                        "COALESCE(NULLIF(TRIM(p.name), ''), p.filename) AS name, p.id AS vorhanden "
                        f"FROM ki_aufrufe a LEFT JOIN projects p ON p.id = a.project_id WHERE {bed} "
                        "GROUP BY a.project_id ORDER BY c DESC", werte)]
        zwecke = [dict(_summen(r), zweck=r["zweck"]) for r in conn.execute(
            f"SELECT a.zweck, {_SUMMEN_SQL} FROM ki_aufrufe a WHERE {bed} GROUP BY a.zweck ORDER BY c DESC", werte)]
    finally:
        conn.close()
    return {"konto_user_id": konto_id, "projekte": projekte, "nach_zweck": zwecke}


def projekt_bericht(project_id, konto_id, jahr: int, monat: int, umgebungen=None) -> dict:
    """Bilder eines Projekts mit ihren Kosten (und die Aufrufe ohne Bild, z. B. Chatbot) im Monat."""
    import umsatz
    von, bis = umsatz.zeitraum(int(jahr), int(monat))
    bed, werte = _bedingung(von, bis, umgebungen)
    if project_id is None:
        bed += " AND a.project_id IS NULL"
    else:
        bed += " AND a.project_id = ?"
        werte.append(int(project_id))
    if konto_id is None:
        bed += " AND a.konto_user_id IS NULL"
    else:
        bed += " AND a.konto_user_id = ?"
        werte.append(int(konto_id))
    conn = _db()
    try:
        bilder = [dict(_summen(r), image_id=r["iid"], seite=r["seite"], dokument=r["dok"] or "",
                       zweck=r["zwecke"] or "") for r in conn.execute(
            f"SELECT a.image_id AS iid, {_SUMMEN_SQL}, i.page_number AS seite, "
            "COALESCE(NULLIF(TRIM(d.display_name), ''), d.original_filename) AS dok, GROUP_CONCAT(DISTINCT a.zweck) AS zwecke "
            "FROM ki_aufrufe a LEFT JOIN images i ON i.id = a.image_id LEFT JOIN documents d ON d.id = i.document_id "
            f"WHERE {bed} AND a.image_id IS NOT NULL GROUP BY a.image_id ORDER BY c DESC LIMIT 500", werte)]
        ohne_bild = [dict(_summen(r), zweck=r["zweck"]) for r in conn.execute(
            f"SELECT a.zweck, {_SUMMEN_SQL} FROM ki_aufrufe a WHERE {bed} AND a.image_id IS NULL "
            "GROUP BY a.zweck ORDER BY c DESC", werte)]
    finally:
        conn.close()
    return {"project_id": project_id, "bilder": bilder, "ohne_bild": ohne_bild}


def konto_summen(konto_id: int, monate: int = 12) -> dict:
    """Fuer die Kundenseite: gemessene Kosten des Kontos insgesamt und je Monat (deutsche Zeit) seit Messbeginn."""
    import umsatz
    conn = _db()
    try:
        gesamt = conn.execute("SELECT COUNT(*) AS n, SUM(kosten_eur_cent) AS c FROM ki_aufrufe WHERE konto_user_id = ?",
                              (int(konto_id),)).fetchone()
        anfang = conn.execute("SELECT MIN(created_at) FROM ki_aufrufe").fetchone()[0]
        je_monat = {}
        jetzt = umsatz.jetzt_lokal()
        j, m = jetzt.year, jetzt.month
        for _ in range(max(1, min(int(monate or 12), 36))):
            von, bis = umsatz.zeitraum(j, m)
            r = conn.execute("SELECT COUNT(*) AS n, SUM(kosten_eur_cent) AS c FROM ki_aufrufe "
                             "WHERE konto_user_id = ? AND created_at >= ? AND created_at < ?",
                             (int(konto_id), von, bis)).fetchone()
            if r["n"]:
                je_monat[f"{j:04d}-{m:02d}"] = float(r["c"] or 0)
            m -= 1
            if m == 0:
                j, m = j - 1, 12
    finally:
        conn.close()
    return {"aufrufe": int(gesamt["n"] or 0), "kosten_cent": float(gesamt["c"] or 0),
            "messbeginn": anfang, "je_monat": je_monat}


def preise_fuer_anzeige() -> dict:
    """Preisliste mit der heute gueltigen Stufe je Modell (fuer die Verwaltung)."""
    liste = preise()
    heute = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    modelle = []
    for name, m in sorted((liste.get("modelle") or {}).items()):
        stufe = preisstufe(name, liste, heute)
        kuenftig = sorted((s for s in m.get("stufen") or [] if str(s.get("ab")) > heute), key=lambda s: s["ab"])
        modelle.append({"modell": name, "quelle": m.get("quelle") or "", "aktuell": stufe,
                        "kuenftig": kuenftig, "stufen": m.get("stufen") or []})
    return {"usd_eur": liste.get("usd_eur"), "usd_eur_quelle": liste.get("usd_eur_quelle") or "",
            "modelle": modelle, "standard": liste == PREISE_STANDARD}
