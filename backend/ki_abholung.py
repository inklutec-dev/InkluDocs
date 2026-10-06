"""Ergebnis-Abholung fuer KI-Knoepfe nach einem Verbindungsabbruch (06.10.2026).

Anlass: Bricht die Verbindung waehrend „Alt-Text (neu) generieren“ oder „Quickinfo generieren“ ab (WLAN-/VPN-
Wechsel, Rechner wacht auf; auf dem Testserver ein Docker-Netzwechsel), rechnet der Server zu Ende, speichert
das Ergebnis und bucht die Credits — die Seite bekam aber nur „Verbindungsfehler“. Wer dann noch einmal klickte,
generierte und bezahlte doppelt.

Ablauf: Die Seite schickt mit dem POST eine eigene Kennung im Kopf „X-KI-Anfrage“. Der Server merkt sich je
Kennung den Stand (laeuft / fertig mit Status und Antwort) fuer AUFBEWAHREN_S Sekunden. Nach einem Abbruch fragt
die Seite GET /api/ki-anfragen/<kennung> ab: „laeuft“ -> weiter warten, „fertig“ -> dieselbe Antwort, die der POST
geliefert haette, anzeigen; unbekannt (404) -> die Anfrage kam nie an, bisherige Fehlermeldung. Das Abholen
generiert nichts und bucht nichts.

Nur im Speicher dieses Prozesses (InkluDocs laeuft mit EINEM uvicorn-Worker). Nach einem Neustart sind die
Eintraege weg — dann meldet die Seite den Verbindungsfehler wie bisher. Ohne Kopf verhalten sich die Endpunkte
unveraendert (API-Kunden, alte Seiten im Browser-Cache)."""
import functools
import inspect
import re
import threading
import time

from fastapi import HTTPException

KOPF = "x-ki-anfrage"
AUFBEWAHREN_S = 600          # 10 Minuten: genug fuer die laengste KI-Antwort plus Abholen
HOECHSTZAHL = 5000           # Schutz vor Speicherwachstum (je Eintrag ein paar hundert Byte)
_MUSTER = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_sperre = threading.Lock()
_anfragen: dict = {}         # kennung -> {"user_id", "zeit", "stand", "status", "daten"}


def anfrage_id(request) -> str | None:
    """Kennung aus dem Kopf X-KI-Anfrage, nur wenn sie dem Muster entspricht (sonst None = keine Abholung)."""
    if request is None:
        return None
    wert = (request.headers.get(KOPF) or "").strip()
    return wert if _MUSTER.match(wert) else None


def _aufraeumen(jetzt: float) -> None:
    for k in [k for k, v in _anfragen.items() if jetzt - v["zeit"] > AUFBEWAHREN_S]:
        del _anfragen[k]
    if len(_anfragen) > HOECHSTZAHL:
        for k in sorted(_anfragen, key=lambda k: _anfragen[k]["zeit"])[:len(_anfragen) - HOECHSTZAHL]:
            del _anfragen[k]


def beginnen(kennung: str | None, user_id) -> None:
    if not kennung:
        return
    with _sperre:
        jetzt = time.time()
        _aufraeumen(jetzt)
        alt = _anfragen.get(kennung)
        if alt and alt["user_id"] != user_id:
            return   # fremde Kennung nie ueberschreiben
        _anfragen[kennung] = {"user_id": user_id, "zeit": jetzt, "stand": "laeuft", "status": None, "daten": None}


def abschliessen(kennung: str | None, user_id, status: int, daten) -> None:
    if not kennung:
        return
    with _sperre:
        e = _anfragen.get(kennung)
        if not e or e["user_id"] != user_id:
            return
        e.update(stand="fertig", status=int(status), daten=daten, zeit=time.time())


def vergessen(kennung: str | None, user_id) -> None:
    if not kennung:
        return
    with _sperre:
        e = _anfragen.get(kennung)
        if e and e["user_id"] == user_id:
            del _anfragen[kennung]


def abfragen(kennung: str, user_id) -> dict | None:
    """{"stand": "laeuft"} | {"stand": "fertig", "status": int, "daten": dict} | None (unbekannt/fremd/abgelaufen)."""
    if not _MUSTER.match(kennung or ""):
        return None
    with _sperre:
        _aufraeumen(time.time())
        e = _anfragen.get(kennung)
        if not e or e["user_id"] != user_id:
            return None
        if e["stand"] == "laeuft":
            return {"stand": "laeuft"}
        return {"stand": "fertig", "status": e["status"], "daten": e["daten"]}


def abholbar(fn):
    """Dekorator fuer einen KI-Endpunkt mit den Parametern `request` und `user`: merkt sich Stand und Antwort
    unter der Kennung des Aufrufs. FastAPI sieht dieselben Parameter: Die Signatur wird mit AUFGELOESTEN Typen
    uebernommen — FastAPI wertet Text-Annotationen (from __future__ import annotations, z. B. formular_api) sonst
    im Namensraum dieses Moduls aus, wo `Request` & Co. fehlen."""
    signatur = inspect.signature(fn, eval_str=True)

    @functools.wraps(fn)
    async def innen(*args, **kwargs):
        kennung = anfrage_id(kwargs.get("request"))
        uid = (kwargs.get("user") or {}).get("id")
        beginnen(kennung, uid)
        try:
            erg = await fn(*args, **kwargs)
        except HTTPException as e:
            abschliessen(kennung, uid, e.status_code, {"detail": e.detail})
            raise
        except Exception:
            abschliessen(kennung, uid, 500, {"detail": "Interner Fehler"})
            raise
        except BaseException:
            vergessen(kennung, uid)   # abgebrochen (Server faehrt herunter): nichts abzuholen
            raise
        abschliessen(kennung, uid, 200, erg if isinstance(erg, dict) else {})
        return erg
    innen.__signature__ = signatur
    return innen
