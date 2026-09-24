#!/usr/bin/env python3
"""Gastansicht nach dem Testumbau (18.09.2026): Ein Gast sieht bei einem Word-Projekt weiter nur die
Alt-Text-Pruefung — keine Ansichts-Wahl, keine Uebersetzungs-Ansicht, ?ansicht=uebersetzung wirkungslos.
Aufruf: ui_gast_word_testumbau.py <projekt-id eines Word-Projekts> <token> <gast-mail>"""
import os, sys
from playwright.sync_api import sync_playwright
B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
PID, TOKEN, GMAIL = sys.argv[1], sys.argv[2], sys.argv[3]
ok = fehler = 0
def check(name, cond, info=""):
    global ok, fehler
    if cond: ok += 1; print("OK   ", name)
    else: fehler += 1; print("FEHLT", name, "—", str(info)[:250])
with sync_playwright() as p:
    br = p.chromium.launch(); pg = br.new_page(); probleme = []
    pg.on("pageerror", lambda e: probleme.append(str(e)))
    pg.goto(f"{B}/freigabe/{TOKEN}?ansicht=uebersetzung", wait_until="networkidle")
    pg.fill("#gateEmail", GMAIL); pg.click("#gateForm button[type=submit]")
    pg.wait_for_selector("#imageFilterBar", timeout=15000); pg.wait_for_timeout(1500)
    check("Gast: Alt-Text-Pruefung mit Bilderkarten", pg.locator("section.image-review").count() >= 1, pg.locator("section.image-review").count())
    check("Gast: KEINE Ansichts-Wahl (keine Ansichts-Knöpfe)", pg.locator(".ansicht-knoepfe").count() == 0 and pg.locator("#ansichtSelect").count() == 0)
    check("Gast: KEINE Uebersetzungs-Ansicht trotz ?ansicht=uebersetzung", pg.locator("#segFilterBar").count() == 0 and pg.locator("textarea.seg-ziel").count() == 0)
    r = pg.request.get(f"{B}/api/projects/{PID}/uebersetzung?leicht=1")
    check("Gast: Uebersetzungs-API ohne Login gesperrt (401/403)", r.status in (401, 403), r.status)
    r2 = pg.request.get(f"{B}/api/freigabe/{TOKEN}/uebersetzung")
    check("Gast: kein Uebersetzungs-Endpunkt unter /api/freigabe (404)", r2.status == 404, r2.status)
    pg.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"); pg.wait_for_timeout(500)
    res = pg.evaluate("axe.run(document, {runOnly:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}).then(r => r.violations.filter(v => ['serious','critical'].includes(v.impact)).map(v => v.id + ': ' + v.nodes.length))")
    check("Gast: axe 0 ernste Verstoesse", not res, res)
    check("Gast: keine Skriptfehler", not probleme, probleme[:3])
    br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER"); sys.exit(1 if fehler else 0)
