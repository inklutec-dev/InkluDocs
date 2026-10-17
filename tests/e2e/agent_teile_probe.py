#!/usr/bin/env python3
"""Klickprobe nach dem Aufteilen des Agenten-Skripts (17.10.2026, backend/inkluagent/skripte.py): auf jeder Seite mit Agent laden
alle Teile genau einmal und in der Reihenfolge der Liste, ohne JavaScript-Fehler; aus jedem Teil ist eine Funktion da; der Bereich
des Agenten ist aufgebaut — in der Agentenansicht und in der manuellen Ansicht. Ohne KI, ohne Aenderungen am Konto ausser der
Ansicht (danach wie vorher). Aufruf (vor jedem Einspielen auf Staging, Playwright): python3 tests/e2e/agent_teile_probe.py
<mail> <passwort>   (Server: INKLUDOCS_BASIS, Vorgabe Staging — nie gegen Production)"""
import os
import sys

from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_BASIS", "https://staging.inkludocs.inklutec.de")   # nie gegen Production
TEILE = ("inkluagent-ansicht.js", "inkluagent-arbeitsfenster.js", "inkluagent-fenster.js", "inkluagent-live.js", "inkluagent-hochladen.js",
         "inkluagent-gespraeche.js", "inkluagent-nachrichten.js", "inkluagent-karten.js", "inkluagent-seitenwechsel.js",
         "inkluagent-rueckmeldung.js", "inkluagent.js")   # wie backend/inkluagent/skripte.py TEILE
# je Teil eine Funktion, die es nur dort gibt
AUS_JEDEM_TEIL = ("inkluagentChatstil", "inkluagentArbeitsZeigen", "inkluagentFensterFolgen", "inkluagentLiveZeigen", "inkluagentAnhangBauen",
                  "inkluagentLoadHistory", "inkluagentAppendMessage", "inkluagentFormularKarte", "inkluagentWechseln",
                  "inkluagentRueckmeldungenBeobachten", "inkluagentSend")
ok = fehler = 0


def check(n, c, info=""):
    global ok, fehler
    ok, fehler = (ok + 1, fehler) if c else (ok, fehler + 1)
    print(("  OK   " if c else "  FEHLT ") + n + ("" if c else " -- " + str(info)[:500]))


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(locale="de-DE", viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    fehlerliste = []
    pg.on("pageerror", lambda e: fehlerliste.append(f"{pg.url}: {e}"))
    pg.on("console", lambda m: fehlerliste.append(f"{pg.url}: {m.text}") if m.type == "error" and "favicon" not in m.text else None)
    check("Anmeldung", pg.request.post(B + "/api/login", data={"email": sys.argv[1], "password": sys.argv[2]}).ok)
    vorher = pg.request.get(B + "/api/me").json()["user"]["oberflaeche"]
    projekte = pg.request.get(B + "/api/projects").json()
    projekte = projekte.get("projects", projekte) if isinstance(projekte, dict) else projekte
    pid = projekte[0]["id"] if projekte else None
    seiten = ["/dashboard", "/ablage", "/einstellungen"] + ([f"/app?projekt={pid}"] if pid else [])
    try:
        for ansicht in ("agent", "klassisch"):
            pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": ansicht})
            for s in seiten:
                fehlerliste.clear()
                pg.goto(B + s, wait_until="networkidle")
                pg.wait_for_timeout(1500)
                d = pg.evaluate("""(arg) => {
                    const srcs = [...document.querySelectorAll('script[src]')].map(x => x.getAttribute('src').split('?')[0].replace('/static/', ''));
                    const teile = srcs.filter(x => /^inkluagent(-[a-z]+)?\\.js$/.test(x));
                    return { teile, fehlen: arg.filter(f => typeof window[f] !== 'function'),
                             bereich: !!document.getElementById('inkluagentInput'), version: (document.querySelector('script[src*="inkluagent.js"]') || {}).src || '' }; }""",
                                list(AUS_JEDEM_TEIL))
                with_agent = bool(d["teile"])
                check(f"{ansicht} {s}: alle Teile genau einmal, in der Reihenfolge der Liste", d["teile"] == list(TEILE), d["teile"])
                check(f"{ansicht} {s}: aus jedem Teil ist die Funktion da", with_agent and not d["fehlen"], d["fehlen"])
                check(f"{ansicht} {s}: Bereich des Agenten aufgebaut", d["bereich"], d)
                check(f"{ansicht} {s}: keine JavaScript-Fehler", not fehlerliste, fehlerliste)
        # der Agent antwortet auf Bedienung (ohne KI): Gespräch öffnen und das Feld nimmt Text
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": "agent"})
        pg.goto(B + "/dashboard", wait_until="networkidle")
        pg.fill("#inkluagentInput", "Probe (fiktiv), wird nicht gesendet")
        check("Eingabefeld des Agenten bedienbar", pg.input_value("#inkluagentInput") == "Probe (fiktiv), wird nicht gesendet")
    finally:
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": vorher})
        print("  Ansicht zurueck auf: " + pg.request.get(B + "/api/me").json()["user"]["oberflaeche"])
    br.close()
print(f"ERGEBNIS: {ok} OK, {fehler} FEHLT")
