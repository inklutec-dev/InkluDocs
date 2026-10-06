"""KI-Knopf klicken und auf das ERGEBNIS warten statt auf eine feste Zeit (Klicktests, 06.10.2026).

Anlass: ui_word („Generierter Alt-Text vorhanden“) und ui_formular („KI-Text im Feld … Verbindungsfehler.“)
wackelten am 05./06.10.2026. Ursache laut Staging-, Proxy- und Kernel-Protokoll, beide AUSSERHALB des App-Codes:
1. KI-Latenz: Eine Generierung dauert meist ~20 s, Ausreisser 54 s (Quickinfo) und 92 s (Alt-Text). Die festen
   Wartezeiten der Tests (45 s / 90 s) waren dafuer zu knapp.
2. Netzwechsel auf dem Testrechner: Der Klicktest-Chromium laeuft auf dem Server, auf dem parallel Docker-Container
   und -Netze angelegt und entfernt werden. Deren Schnittstellen tragen IPv6-Link-lokal-Adressen; ihr Kommen und
   Gehen meldet Chromium als Netzwechsel und schliesst alle HTTP/2-Verbindungen mit net::ERR_NETWORK_CHANGED. Die
   laufende KI-Anfrage bricht ab (Seite: „Verbindungsfehler.“), der Server rechnet trotzdem zu Ende. Belegt am
   06.10.2026: 07:12:53,72 „docker rm -f“ eines Wegwerf-Containers -> Abbruch 07:12:53 (ui_word); 07:16:14,06
   „docker network create“ direkt nach einem „docker build“ -> Abbruch 07:16:14 (ui_formular); der Proxy meldete
   jeweils 499 (Anfrage vom Browser geschlossen). Nachbau ohne Staging: tests/e2e/netzwechsel_nachweis/nachweis.sh.

Darum wartet ki_klick() auf Antwort oder Abbruch DER Anfrage (grosszuegige Obergrenze), gibt Dauer und Ursache aus
(Zeilen beginnen mit „KI-Anfrage“, die Aufrufer-Skripte reichen sie durch). Seit dem Produkt-Fix vom 06.10.2026 holt
die Seite nach einem Abbruch das Ergebnis selbst ab (app.html kiErgebnisAbholen, backend/ki_abholung.py): ki_klick
wartet dann, bis der Knopf wieder frei ist, und prueft mit `ergebnis_da`, ob das Ergebnis angekommen ist — dann
KEIN zweiter Klick (kein doppeltes Generieren). Nur wenn nichts ankam, klickt es bei einem Transportabbruch
des Browsers noch einmal. Antworten des Servers mit Fehlerstatus (4xx/5xx) werden nie wiederholt — sie bleiben ein
Befund. Loest der Klick gar keine Anfrage aus, steht auch das im Protokoll (dann liegt es an der Seite).

Optional (nur zur Diagnose):
- INKLUDOCS_E2E_MITSCHNITT=<Datei>: mitschnitt_anhaengen() schreibt alle Anfragen, Antworten, Abbrueche und
  Konsolenmeldungen mit Zeitstempel als JSON-Zeilen in die Datei.
- INKLUDOCS_E2E_STOERUNG=<Shell-Befehl>: wird beim ERSTEN KI-Klick eines Laufs 3 s nach dem Klick ausgefuehrt
  (z. B. ein Docker-Netz anlegen und loeschen), um den Netzwechsel absichtlich auszuloesen.
"""
import json
import os
import subprocess
import threading
import time

# Abbrueche durch das Netz des Testrechners (nicht durch den Server): nur diese loesen einen neuen Klick aus.
TRANSPORT_ABBRUECHE = (
    "net::ERR_NETWORK_CHANGED", "net::ERR_CONNECTION_RESET", "net::ERR_CONNECTION_CLOSED",
    "net::ERR_CONNECTION_ABORTED", "net::ERR_HTTP2_PROTOCOL_ERROR", "net::ERR_HTTP2_PING_FAILED",
    "net::ERR_HTTP2_SERVER_REFUSED_STREAM", "net::ERR_INTERNET_DISCONNECTED", "net::ERR_NETWORK_IO_SUSPENDED",
)
_stoerung_erledigt = False


def _sek(s):
    return f"{s:.1f}".replace(".", ",") + " s"


def mitschnitt_anhaengen(pg, kennung=""):
    """Haengt bei gesetztem INKLUDOCS_E2E_MITSCHNITT einen Netz- und Konsolenmitschnitt an die Seite."""
    datei = os.environ.get("INKLUDOCS_E2E_MITSCHNITT", "")
    if not datei:
        return

    def schreiben(art, **daten):
        daten.update(art=art, zeit=time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}", seite=kennung)
        with open(datei, "a", encoding="utf-8") as f:
            f.write(json.dumps(daten, ensure_ascii=False) + "\n")

    pg.on("request", lambda r: schreiben("anfrage", methode=r.method, url=r.url))
    pg.on("response", lambda r: schreiben("antwort", methode=r.request.method, url=r.url, status=r.status))
    pg.on("requestfailed", lambda r: schreiben("abbruch", methode=r.method, url=r.url, grund=r.failure))
    pg.on("console", lambda m: schreiben("konsole", typ=m.type, text=m.text[:500]))
    pg.on("pageerror", lambda e: schreiben("js-fehler", text=str(e)[:500]))
    pg.on("framenavigated", lambda fr: schreiben("navigation", url=fr.url) if fr == pg.main_frame else None)


def _stoerung_planen():
    """Diagnose-Schalter INKLUDOCS_E2E_STOERUNG: Befehl einmal je Lauf 3 s nach dem ersten KI-Klick ausfuehren."""
    global _stoerung_erledigt
    befehl = os.environ.get("INKLUDOCS_E2E_STOERUNG", "")
    if not befehl or _stoerung_erledigt:
        return
    _stoerung_erledigt = True

    def los():
        time.sleep(3)
        subprocess.run(befehl, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"      KI-Anfrage: Stoerung ausgeloest ({befehl[:60]})", flush=True)
    threading.Thread(target=los, daemon=True).start()


def ki_klick(pg, knopf, url_teil, name, grenze_s=180, wiederholungen=1, bild=None, ergebnis_da=None):
    """Klickt `knopf` und wartet auf die POST-Anfrage, deren URL `url_teil` enthaelt — bis Antwort oder Abbruch.

    Rueckgabe: dict(status=int|None, grund=str|None, sekunden=float, versuche=int, kurz=str)
      status   HTTP-Status der Antwort (None: keine Antwort binnen grenze_s oder Abbruch)
      grund    Abbruchgrund des Browsers (z. B. net::ERR_NETWORK_CHANGED) oder Beschreibung des Ausbleibens
      kurz     ein Satz fuer die Fehlermeldung des Checks
      abgeholt True, wenn die Seite das Ergebnis nach einem Abbruch selbst abgeholt hat (ergebnis_da() wurde wahr)
      posts    Zahl der POST-Anfragen, die der Browser fuer diesen Knopf geschickt hat
    ergebnis_da: Funktion ohne Argumente -> bool, „die Ansicht zeigt das neue Ergebnis“ (fuer die Abholung).
    Nach einer Antwort zeichnet die Seite das Ergebnis selbst; der Aufrufer prueft danach die Ansicht."""
    ereignisse = []

    def passt(req):
        return req.method == "POST" and url_teil in req.url

    def bei_anfrage(req):
        if passt(req):
            ereignisse.append(("anfrage", None, time.monotonic()))

    def bei_antwort(resp):
        if passt(resp.request):
            ereignisse.append(("antwort", resp.status, time.monotonic()))

    def bei_abbruch(req):
        if passt(req):
            ereignisse.append(("abbruch", req.failure, time.monotonic()))

    pg.on("request", bei_anfrage)
    pg.on("response", bei_antwort)
    pg.on("requestfailed", bei_abbruch)
    erg = dict(status=None, grund=None, sekunden=0.0, versuche=0, kurz="", abgeholt=False, posts=0)
    try:
        for versuch in range(1, wiederholungen + 2):
            erg["versuche"] = versuch
            ab = len(ereignisse)
            t0 = time.monotonic()
            knopf.click()
            if versuch == 1:
                _stoerung_planen()
            ende = None
            gesendet = False
            while time.monotonic() - t0 < grenze_s:
                pg.wait_for_timeout(250)   # laesst Playwright die Ereignisse zustellen
                neu = ereignisse[ab:]
                gesendet = gesendet or any(e[0] == "anfrage" for e in neu)
                ende = next((e for e in neu if e[0] in ("antwort", "abbruch")), None)
                if ende:
                    break
                if not gesendet and time.monotonic() - t0 > 10:
                    break   # 10 s nach dem Klick noch keine Anfrage: der Klick kam nicht an
            erg["sekunden"] = (ende[2] if ende else time.monotonic()) - t0
            if ende and ende[0] == "antwort":
                erg.update(status=ende[1], grund=None)
                erg["kurz"] = f"HTTP {ende[1]} nach {_sek(erg['sekunden'])} (Versuch {versuch})"
                print(f"      KI-Anfrage {name}: {erg['kurz']}", flush=True)
                break
            if ende:   # Abbruch durch den Browser
                erg.update(status=None, grund=ende[1])
                transport = any(ende[1] and ende[1].startswith(t) for t in TRANSPORT_ABBRUECHE)
                erg["kurz"] = (f"vom Browser abgebrochen nach {_sek(erg['sekunden'])}: {ende[1]}"
                               + (" (Netzwechsel/Verbindungsabbruch auf dem Testrechner, nicht der Server)" if transport else ""))
                nochmal = transport and versuch <= wiederholungen
                print(f"      KI-Anfrage {name}: Versuch {versuch} {erg['kurz']}", flush=True)
                if bild:
                    pg.screenshot(path=bild)
                # Die Seite holt das Ergebnis ab und gibt den Knopf danach (finally) wieder frei — darauf warten.
                t1 = time.monotonic()
                while time.monotonic() - t1 < grenze_s and not knopf.is_enabled():
                    pg.wait_for_timeout(250)
                if ergebnis_da and ergebnis_da():
                    erg.update(abgeholt=True)
                    erg["kurz"] += f"; Seite holte das Ergebnis nach {_sek(time.monotonic() - t1)} ab (kein zweiter Klick)"
                    n_posts = sum(1 for e in ereignisse if e[0] == "anfrage")
                    print(f"      KI-Anfrage {name}: Ergebnis von der Seite abgeholt nach {_sek(time.monotonic() - t1)}, "
                          f"kein zweiter Klick (POST-Anfragen insgesamt: {n_posts})", flush=True)
                    break
                if not nochmal:
                    break
                # Ein Netzwechsel kommt selten allein (Container-Start = Netz, veth, Adressen): erst zur Ruhe kommen
                # lassen, sonst faellt der neue Klick in dieselbe Stoerung (Stoerlauf 06.10.2026, 09:24).
                print(f"      KI-Anfrage {name}: kein Ergebnis angekommen — neuer Klick in 10 s", flush=True)
                pg.wait_for_timeout(10000)
                continue
            # Keine Antwort binnen Grenze oder gar keine Anfrage
            erg.update(status=None, grund=("keine Anfrage nach dem Klick" if not gesendet
                                           else f"keine Antwort binnen {grenze_s} s"))
            erg["kurz"] = (f"Klick loeste binnen 10 s keine Anfrage aus (Versuch {versuch})" if not gesendet
                           else f"keine Antwort binnen {grenze_s} s (KI-Latenz, Versuch {versuch})")
            print(f"      KI-Anfrage {name}: {erg['kurz']}", flush=True)
            if bild:
                pg.screenshot(path=bild)
            break
    finally:
        erg["posts"] = sum(1 for e in ereignisse if e[0] == "anfrage")
        pg.remove_listener("request", bei_anfrage)
        pg.remove_listener("response", bei_antwort)
        pg.remove_listener("requestfailed", bei_abbruch)
    return erg


ANSAGEN_JS = """(() => { if (window.__ansagen) return; window.__ansagen = [];
  const r = document.getElementById('liveRegion'); if (!r) return;
  new MutationObserver(() => { const t = (r.textContent || '').trim(); if (t) window.__ansagen.push(t); })
    .observe(r, { childList: true, characterData: true, subtree: true }); })()"""


def ansagen_beobachten(pg):
    """Sammelt ab jetzt alle Ansagen der Live-Region (#liveRegion) in window.__ansagen (bis zum Neuladen)."""
    pg.evaluate(ANSAGEN_JS)


def ansagen(pg):
    return pg.evaluate("window.__ansagen || []")


def warten_bis(pg, bedingung, sekunden=10):
    """Nach der Antwort: kurz warten, bis die Seite das Ergebnis gezeichnet hat (bedingung() -> bool)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < sekunden:
        if bedingung():
            return True
        pg.wait_for_timeout(200)
    return bedingung()
