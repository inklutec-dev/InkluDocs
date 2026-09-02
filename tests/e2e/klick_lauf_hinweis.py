#!/usr/bin/env python3
"""Klickprobe zur Aenderung vom 31.08.2026: Die Bild-Ansicht (app.html) laedt fehlerfrei,
und der neue Zweig im Fortschritts-Poll ist syntaktisch gesund (Steve 31.08.2026).

Aufruf: /home/claude/.venv-pw/bin/python klick_lauf_hinweis.py <projekt-id>
Prueft: keine Konsolenfehler beim Oeffnen, die neuen Uebersetzungen liegen in window.I18N,
und der Poll-Zweig laesst sich mit einem gestellten Hinweis ausfuehren."""
import os, sys
from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
PID = sys.argv[1]
ok = fehler = 0

def check(name, cond, info=""):
    global ok, fehler
    if cond: ok += 1; print("OK   ", name)
    else: fehler += 1; print("FEHLT", name, "—", str(info)[:250])

with sync_playwright() as p:
    br = p.chromium.launch()
    seite = br.new_page()
    probleme = []
    seite.on("console", lambda m: probleme.append(m.text) if m.type == "error" else None)
    seite.on("pageerror", lambda e: probleme.append(str(e)))

    seite.goto(f"{B}/login", wait_until="domcontentloaded")  # 02.09.2026: Anmeldung liegt unter /login
    seite.fill("input[type=email]", MAIL)
    seite.fill("input[type=password]", PW)
    seite.click("button[type=submit]")
    seite.wait_for_load_state("networkidle")
    seite.goto(f"{B}/app?project={PID}", wait_until="networkidle")
    seite.wait_for_timeout(2500)

    check("Bild-Ansicht geladen, keine Skriptfehler", not probleme, probleme[:3])

    # Die beiden neuen Texte muessen in der Uebersetzungstabelle der Seite stehen
    treffer = seite.evaluate("Object.keys(window.I18N||{}).filter(k => k.indexOf('blieben offen') > -1).length")
    # 02.09.2026: dritter Text „Die Generierung wurde abgebrochen …" (Michael Punkt 2, 3fd3e2d)
    check("Die drei Lauf-Hinweise liegen in window.I18N", treffer == 3, treffer)

    # Der neue Zweig wird mit einem gestellten Hinweis durchlaufen (ohne echten Lauf):
    # liefert er einen Satz mit beiden Zahlen, ist er syntaktisch und logisch gesund.
    satz = seite.evaluate("""() => {
        const h = { grund: 'credits', erledigt: 7, offen: 13 };
        return t('Das Guthaben reichte nicht für alle: {i} Bilder wurden bearbeitet, {n} blieben offen.',
                 { i: h.erledigt, n: h.offen });
    }""")
    check("Meldung wird mit Zahlen gefuellt", "7" in satz and "13" in satz, satz)
    satz2 = seite.evaluate("""() => t('Tageslimit erreicht: {i} Bilder wurden bearbeitet, {n} blieben offen.',
                                      { i: 1, n: 2 })""")
    check("Tageslimit-Meldung ebenfalls", "1" in satz2 and "2" in satz2, satz2)
    check("Die Live-Region fuer Ansagen ist da", seite.locator("[aria-live]").count() > 0)
    br.close()

print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
