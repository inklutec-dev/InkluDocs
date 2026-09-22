#!/usr/bin/env python3
"""Gast- und Besitzer-Pfad der Filterleiste bei aktiver Freigabe (17.09.2026).
Aufruf: ui_filter_gast.py <projekt-id> <token> <gast-mail> — Freigabe wird vorher/nachher per DB gestellt/entfernt."""
import os, sys
from playwright.sync_api import sync_playwright
B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
PID, TOKEN, GMAIL = sys.argv[1], sys.argv[2], sys.argv[3]
ERW_S = ["alle", "offen", "mit_text"]; ERW_P = ["alle", "neu", "in_bearbeitung", "lek_frei", "lek_aend", "her_frei", "her_aend"]
ok = fehler = 0
def check(name, cond, info=""):
    global ok, fehler
    if cond: ok += 1; print("OK   ", name)
    else: fehler += 1; print("FEHLT", name, "—", str(info)[:250])
with sync_playwright() as p:
    br = p.chromium.launch()
    # Gast
    seite = br.new_page(); probleme = []
    seite.on("console", lambda m: probleme.append(m.text) if m.type == "error" else None)
    seite.on("pageerror", lambda e: probleme.append(str(e)))
    seite.goto(f"{B}/freigabe/{TOKEN}", wait_until="networkidle")
    seite.fill("#gateEmail", GMAIL); seite.click("#gateForm button[type=submit]")
    seite.wait_for_selector("#imageFilterBar", timeout=15000); seite.wait_for_timeout(1500)
    ks = seite.evaluate("[Array.from(document.querySelectorAll('input[name=imgFilterStand]')).map(i => i.value), Array.from(document.querySelectorAll('input[name=imgFilterPruef]')).map(i => i.value)]")
    check("Gast: zwei Felder (Bearbeitungsstand + Freigabestatus)", ks[0] == ERW_S and ks[1] == ERW_P, ks)
    check("Gast: keine Skriptfehler (401 von /api/me = anonyme Huelle)", not [x for x in probleme if "401" not in x], probleme[:3])
    seite.check("input[name=imgFilterStand][value=offen]"); seite.wait_for_timeout(300)
    check("Gast: Filtern laeuft (Ergebniszeile)", "Bildern" in seite.locator("#filterResult").inner_text(), seite.locator("#filterResult").inner_text())
    seite.check("input[name=imgFilterPruef][value=neu]"); seite.wait_for_timeout(300)
    n_beide = seite.evaluate("document.querySelectorAll('section.image-review:not([hidden])').length")
    n_erw = seite.evaluate("filterImages.filter(i => imageMatchesKey(i, 'offen') && imageMatchesKey(i, 'neu')).length")
    check("Gast: beide Felder wirken zusammen (UND)", n_beide == n_erw, (n_beide, n_erw))
    # Besitzer mit aktiver Freigabe (in_review)
    s2 = br.new_page(); probleme2 = []
    s2.on("console", lambda m: probleme2.append(m.text) if m.type == "error" else None)
    s2.on("pageerror", lambda e: probleme2.append(str(e)))
    s2.goto(f"{B}/login", wait_until="domcontentloaded"); s2.fill("input[type=email]", MAIL); s2.fill("input[type=password]", PW)
    s2.click("button[type=submit]"); s2.wait_for_load_state("networkidle")
    s2.goto(f"{B}/app?projekt={PID}&ansicht=alttexte", wait_until="networkidle"); s2.wait_for_timeout(2000)
    ks2 = s2.evaluate("[Array.from(document.querySelectorAll('input[name=imgFilterStand]')).map(i => i.value), Array.from(document.querySelectorAll('input[name=imgFilterPruef]')).map(i => i.value)]")
    check("Besitzer mit Freigabe: zwei Felder", ks2[0] == ERW_S and ks2[1] == ERW_P, ks2)
    check("Besitzer: keine Skriptfehler", not probleme2, probleme2[:3])
    br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER"); sys.exit(1 if fehler else 0)
