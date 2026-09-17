#!/usr/bin/env python3
"""Klickprobe (17.09.2026): Seite „API-Schluessel" mit Verbrauchskarte, Dashboard-Kachel, Doku-Seite in de/en.
Aufruf: /home/claude/.venv-pw/bin/python ui_api_seiten.py (Creds aus ~/.e2e.env)."""
import os, sys
from playwright.sync_api import sync_playwright
B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ["INKLUDOCS_E2E_MAIL"], os.environ["INKLUDOCS_E2E_PW"]
ok = fehler = 0
def check(name, cond, info=""):
    global ok, fehler
    if cond: ok += 1; print("OK   ", name)
    else: fehler += 1; print("FEHLT", name, "—", str(info)[:250])
with sync_playwright() as p:
    br = p.chromium.launch(); s = br.new_page(); probleme = []
    s.on("console", lambda m: probleme.append(m.text) if m.type == "error" else None)
    s.on("pageerror", lambda e: probleme.append(str(e)))
    s.goto(f"{B}/api/v1/docs", wait_until="networkidle")
    check("Doku-Seite laedt im oeffentlichen Geruest", s.locator("#h-dok").count() == 1 and s.locator(".app-sidebar").count() == 1)
    check("Doku: Inhaltsverzeichnis mit 9 Eintraegen", s.locator(".api-doc nav ul li a").count() == 9, s.locator(".api-doc nav ul li a").count())
    s.goto(f"{B}/login", wait_until="domcontentloaded"); s.fill("input[type=email]", MAIL); s.fill("input[type=password]", PW)
    s.click("button[type=submit]"); s.wait_for_load_state("networkidle")
    s.goto(f"{B}/api-schluessel", wait_until="networkidle"); s.wait_for_timeout(1500)
    check("Schluesselseite: Verbrauchskarte mit 5 Zeilen", s.locator("#apiUsageList li").count() == 5, s.locator("#apiUsageList li").count())
    check("Schluesselseite: Link zur Doku", s.locator("a[href='/api/v1/docs']").count() >= 1)
    check("Schluesselseite: Zeile je Schluessel mit Aufrufen", s.locator("#apiKeyList li").count() == 0 or "Aufrufe" in s.locator("#apiKeyList").inner_text(), s.locator("#apiKeyList").inner_text()[:200])
    s.goto(f"{B}/dashboard", wait_until="networkidle"); s.wait_for_timeout(1500)
    hat_keys = s.evaluate("fetch('/api/api-keys/stats').then(r => r.json()).then(d => d.keys_count)")
    sichtbar = s.evaluate("!document.getElementById('apiSection').hidden")
    check("Dashboard: API-Kachel genau dann sichtbar, wenn Schluessel existieren", sichtbar == (hat_keys > 0), (hat_keys, sichtbar))
    if sichtbar:
        check("Dashboard: Kachel nennt Aufrufe und zwei Wege", "Aufrufe" in s.locator("#apiContainer").inner_text() and s.locator("#apiContainer a").count() == 2, s.locator("#apiContainer").inner_text())
    # Der 401 von /api/me gehoert zur oeffentlichen Huelle (dashboard.js prueft die Anmeldung; auch auf /preise so).
    echte = [x for x in probleme if "401" not in x]
    check("Keine Skriptfehler (401 von /api/me ist die anonyme Huelle)", not echte, echte[:3])
    br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER"); sys.exit(1 if fehler else 0)
