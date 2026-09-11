#!/usr/bin/env python3
"""Klicktest Chatbot + Meine Ausgaben (11.09.2026): Umwandlung im Chat anstossen, Rueckfrage, Ja,
Download-Knopf und Link ins Regal unter der Antwort, Reiter-Zaehler, Anhang nach Neuladen, axe.
Aufruf: /home/claude/.venv-pw/bin/python ui_chat_ausgaben.py <projekt-id eines Word-Projekts mit Alt-Texten>
Zugangsdaten aus INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW (~/.e2e.env)."""
import os, sys, re
from playwright.sync_api import sync_playwright
B = os.environ.get("INKLUDOCS_E2E_URL") or "https://staging.inkludocs.inklutec.de"
PID = sys.argv[1]
MAIL = os.environ.get("INKLUDOCS_E2E_MAIL", ""); PW = os.environ.get("INKLUDOCS_E2E_PW", "")
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"
ok = fehler = 0
def check(n, c, i=""):
    global ok, fehler
    if c: ok += 1; print("  OK ", n)
    else: fehler += 1; print("  FEHLT", n, "--", str(i)[:300])
def axe(pg, name):
    pg.add_script_tag(url=AXE); pg.wait_for_timeout(500)
    r = pg.evaluate("async () => { const r = await axe.run(document, {runOnly:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}); return r.violations.map(v => ({id:v.id, impact:v.impact, n:v.nodes.length, html:v.nodes[0].html.slice(0,120)})); }")
    check(f"axe {name}: 0 Verstoesse", len(r) == 0, r)
def senden(pg, text, warte=150):
    pg.fill("#inkluagentInput", text); pg.keyboard.press("Enter")
    n = pg.locator(".inkluagent-message.assistant").count()
    for _ in range(warte):
        pg.wait_for_timeout(1000)
        if pg.locator(".inkluagent-message.assistant").count() > n: break
    pg.wait_for_timeout(500)
    return pg.locator(".inkluagent-message.assistant").last
os.makedirs("/home/claude/shots", exist_ok=True)
with sync_playwright() as p:
    br = p.chromium.launch(); ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE"); pg = ctx.new_page()
    pg.goto(B + "/login"); pg.fill("#email", MAIL); pg.fill("#password", PW); pg.keyboard.press("Enter"); pg.wait_for_timeout(2500)
    # Alle Bilder ohne Text beschriften (wie verify_chat_ausgaben): sonst weigert sich der Bot zu Recht,
    # vor der Umwandlung zu fragen, und schlaegt erst Alt-Texte vor (Prompt-Regel 1). Gleiche Sitzung wie der Browser.
    proj = pg.request.get(B + f"/api/projects/{PID}").json()
    for i in proj.get("images") or []:
        if not (i.get("alt_text_edited") or i.get("alt_text")) and i.get("original_alt") != "dekorativ":
            pg.request.post(B + f"/api/images/{i['id']}/alt-text", data={"alt_text": f"Testtext für Bild {i['id']} (fiktiv, E2E)"})
    pg.request.delete(B + f"/api/projects/{PID}/chat")
    pg.goto(B + f"/app?projekt={PID}"); pg.wait_for_timeout(3500)
    vorher = int(re.search(r"\((\d+)\)", pg.locator("#ausgabenTab").inner_text()).group(1))
    pg.locator("#inkluagentToggle").click(); pg.wait_for_timeout(1200)
    print("== A. Umwandlung im Chat ==")
    a1 = senden(pg, "Wandle das Dokument in eine barrierefreie PDF um.")
    check("Rueckfrage nennt Credits, noch kein Anhang", "Credit" in a1.inner_text() and a1.locator(".inkluagent-message-anhang").count() == 0, a1.inner_text()[:200])
    a2 = senden(pg, "Ja, bitte.")
    anh = a2.locator(".inkluagent-message-anhang")
    check("Anhang unter der Antwort", anh.count() == 1, a2.inner_text()[:200])
    dl = anh.locator("a.btn-primary")
    check("Download-Knopf (PDF/ZIP herunterladen) mit /api/ausgaben/<id>/datei", dl.count() == 1 and dl.inner_text().strip() in ("PDF herunterladen", "ZIP herunterladen") and re.search(r"/api/ausgaben/\d+/datei", dl.get_attribute("href") or ""), (dl.inner_text() if dl.count() else "", dl.get_attribute("href") if dl.count() else ""))
    zu = anh.locator("a.btn-secondary")
    check("Link „Zur Ablage“ mit #ausgabe-<id>", zu.count() == 1 and zu.inner_text().strip() == "Zur Ablage" and re.search(r"/ablage\?projekt=\d+#ausgabe-\d+", zu.get_attribute("href") or ""), zu.get_attribute("href") if zu.count() else "")
    check("Zeile „Geprüft mit“ nennt die Umwandlung", "In barrierefreie PDF umwandeln" in a2.locator(".inkluagent-message-tools").inner_text(), a2.locator(".inkluagent-message-tools").inner_text())
    check("Ablage-Zaehler um 1 erhoeht", pg.locator("#ausgabenTab").inner_text().strip() == f"Ablage ({vorher + 1})", pg.locator("#ausgabenTab").inner_text())
    check("Fokus auf der Antwort", pg.evaluate("() => document.activeElement && document.activeElement.classList.contains('inkluagent-message')"))
    axe(pg, "Projektansicht mit Chat-Anhang")
    pg.screenshot(path="/home/claude/shots/f_chat_anhang.png", full_page=False)
    aid = re.search(r"#ausgabe-(\d+)", zu.get_attribute("href")).group(1)
    print("== B. Nach Neuladen bleibt der Anhang ==")
    pg.reload(); pg.wait_for_timeout(3500)
    if pg.locator("#inkluagentToggle").get_attribute("aria-expanded") != "true": pg.locator("#inkluagentToggle").click(); pg.wait_for_timeout(1500)
    check("Anhang im geladenen Verlauf", pg.locator(f".inkluagent-message-anhang a[href='/api/ausgaben/{aid}/datei']").count() == 1)
    print("== C. Link fuehrt ins Regal ==")
    pg.locator(f".inkluagent-message-anhang a[href*='#ausgabe-{aid}']").first.click(); pg.wait_for_timeout(2500)
    li = pg.locator(f"#ausgabe-{aid}")
    check("Eintrag im Regal mit Vermerk „über den Chatbot“", li.count() == 1 and "über den Chatbot" in li.inner_text(), li.inner_text()[:200] if li.count() else "")
    check("Fokus auf der H2 des Eintrags", pg.evaluate("() => document.activeElement && document.activeElement.tagName") == "H2")
    # Aufraeumen: Eintrag loeschen
    li.locator("button:has-text('Löschen')").click(); pg.wait_for_timeout(300); li.locator("button:has-text('Ja, löschen')").click(); pg.wait_for_timeout(1500)
    check("Testeintrag geloescht", pg.locator(f"#ausgabe-{aid}").count() == 0)
    br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
