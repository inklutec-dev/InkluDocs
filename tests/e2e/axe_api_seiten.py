#!/usr/bin/env python3
"""axe-Pruefung der API-Seiten (17.09.2026): Doku (anonym, de+en), API-Schluessel, Dashboard mit Kachel."""
import os, sys, urllib.request
from playwright.sync_api import sync_playwright
B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ["INKLUDOCS_E2E_MAIL"], os.environ["INKLUDOCS_E2E_PW"]
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"
axe_js = urllib.request.urlopen(AXE, timeout=20).read().decode()
gesamt = 0
def pruefe(pg, name):
    global gesamt
    pg.add_script_tag(content=axe_js)
    e = pg.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}})")
    v = e["violations"]; gesamt += len(v)
    print(f"{name}: {len(v)} Verstoesse")
    for x in v:
        print(f"   - [{x['impact']}] {x['id']}: {x['help']} -> {[t for k in x['nodes'] for t in k['target']][:3]}")
with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ("de", "en"):
        pg = b.new_context(locale=lang, extra_http_headers={"Accept-Language": lang}).new_page()
        pg.goto(f"{B}/api/v1/docs", wait_until="networkidle"); pg.wait_for_timeout(800)
        pruefe(pg, f"API-Doku anonym ({lang})"); pg.close()
    pg = b.new_context().new_page()
    pg.goto(f"{B}/login", wait_until="domcontentloaded"); pg.fill("#email", MAIL); pg.fill("#password", PW); pg.click("button[type=submit]")
    pg.wait_for_url("**/dashboard", timeout=15000); pg.wait_for_timeout(1500); pruefe(pg, "Dashboard mit API-Kachel")
    pg.goto(f"{B}/api-schluessel", wait_until="networkidle"); pg.wait_for_timeout(1500); pruefe(pg, "API-Schluessel mit Verbrauch")
    pg.goto(f"{B}/api/v1/docs", wait_until="networkidle"); pg.wait_for_timeout(800); pruefe(pg, "API-Doku eingeloggt")
    b.close()
print(f"\nERGEBNIS: {gesamt} axe-Verstoesse insgesamt"); sys.exit(1 if gesamt else 0)
