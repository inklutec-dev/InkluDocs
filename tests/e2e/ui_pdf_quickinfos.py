#!/usr/bin/env python3
"""Klicktest Station „Quickinfos“ im PDF-Projekt (22.09.2026): Projekt anlegen, Formular in der Ansicht Dokument
hochladen, Karte zeigt Formularfelder + Knopf „Quickinfos bearbeiten“, Ansichts-Wahl hat drei Stationen, Wechsel
in die Quickinfo-Ansicht (Feldliste, Ansichts-Wahl steht auf Quickinfos), Upload-Hinweis dort, axe, keine Skriptfehler.
Aufruf: ui_pdf_quickinfos.py <testformular.pdf> [--behalten]"""
import os, sys, time
from playwright.sync_api import sync_playwright
B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
FORM = sys.argv[1]; BEHALTEN = "--behalten" in sys.argv
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"
ok = fehler = 0
def check(n, c, i=""):
    global ok, fehler
    if c: ok += 1; print("  OK ", n)
    else: fehler += 1; print("  FEHLT", n, "--", str(i)[:300])
def axe(pg, name):
    pg.add_script_tag(url=AXE); pg.wait_for_timeout(500)
    r = pg.evaluate("async () => { const r = await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']}); return r.violations.map(v => ({id: v.id, impact: v.impact, n: v.nodes.length})); }")
    ernst = [v for v in r if v["impact"] in ("serious", "critical")]
    check(f"axe {name}: 0 ernste Verstoesse", not ernst, ernst)
with sync_playwright() as p:
    br = p.chromium.launch(); ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE"); pg = ctx.new_page()
    fehler_js = []; pg.on("pageerror", lambda e: fehler_js.append(str(e)))
    pg.goto(B + "/login"); pg.fill("#email", MAIL); pg.fill("#password", PW); pg.keyboard.press("Enter"); pg.wait_for_timeout(2500)
    pg.goto(B + "/projekt-neu"); pg.wait_for_timeout(1200)
    opts = pg.locator("#toolSelect option").all_text_contents()
    check("Anlege-Menue: PDF-Dokumente, Word-Dokumente, Webseiten, Grafiken — keine Quickinfos-Kachel, kein Platzhalter", "PDF-Dokumente" in opts and "Webseiten" in opts and "Grafiken" in opts and not any("Quickinfos" in o or "Barrierefreie PDFs" in o for o in opts), opts)
    r = pg.request.post(B + "/api/projects", data={"name": "Klicktest PDF-Quickinfos " + time.strftime("%H:%M"), "tool": "pdf"})
    pid = r.json().get("id") or r.json().get("project_id")
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle"); pg.wait_for_timeout(1000)
    fehler_js_vorher = list(fehler_js)
    check("Neues PDF-Projekt: Knopf Quickinfos da, aber ausgegraut (noch keine Felder)", pg.locator(".ansicht-knoepfe [data-ansicht]").count() == 4 and pg.locator(".ansicht-knoepfe .ansicht-aus[data-ansicht=quickinfos]").count() == 1, pg.locator(".ansicht-knoepfe [data-ansicht]").all_inner_texts())
    pg.set_input_files("#projUpload", FORM); pg.wait_for_selector("section.dok-karte", timeout=90000); pg.wait_for_timeout(1500)
    dl = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    check("Karte nennt Formularfelder (12)", "Formularfelder: 12 Felder" in dl, dl)
    # Keine Wechsel-Knoepfe auf der Karte (Michael Karbe, Mail 22.09.2026, Punkt 3): gewechselt wird ueber die Ansichts-Wahl
    check("Kein Knopf „Quickinfos bearbeiten“ auf der Karte", pg.locator("section.dok-karte button:has-text('Quickinfos bearbeiten')").count() == 0)
    check("Ansichts-Knöpfe: alle vier Stationen als Links (mit Abschlussprüfung)", [x.strip() for x in pg.locator(".ansicht-knoepfe a[data-ansicht]").all_inner_texts()] == ["Dokument", "Alt-Texte", "Quickinfos", "Abschlussprüfung"] and pg.locator(".ansicht-knoepfe .ansicht-aus").count() == 0, pg.locator(".ansicht-knoepfe [data-ansicht]").all_inner_texts())
    axe(pg, "Dokument-Ansicht mit Formular")
    pg.click(".ansicht-knoepfe a[data-ansicht=quickinfos]")
    pg.wait_for_selector("#feldListe", timeout=20000); pg.wait_for_timeout(1000)
    check("Quickinfo-Ansicht: Feldliste mit 12 Feldern", pg.locator("#feldListe textarea").count() == 12, pg.locator("#feldListe textarea").count())
    check("Adresse ansicht=quickinfos, Ansichts-Wahl steht auf Quickinfos", "ansicht=quickinfos" in pg.url and pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("data-ansicht") == "quickinfos")
    check("Quickinfo-Ansicht: kein Upload-Feld (Michael Karbe, Punkt 8: Hochladen nur in der Ansicht Dokument)", pg.locator("#projUploadZone").count() == 0 and pg.locator("#projUpload").count() == 0)
    check("Knopf „Quickinfos generieren“ im Feld „Funktionen und Einstellungen“", pg.locator("section.projekt-funktionen #fGenAllBtn").count() == 1)
    check("Quickinfo-Ansicht: Projektkopf mit PDF-Symbol + Ansichts-Wahl, ohne Statusanzeige", pg.locator(".projekt-kopf img.projekt-dateityp").count() == 1 and pg.locator(".projekt-kopf .ansicht-knoepfe").count() == 1 and pg.locator("#projectStatusBadge").count() == 0)
    axe(pg, "Quickinfo-Ansicht im PDF-Projekt")
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle"); pg.wait_for_timeout(1000)
    check("Ohne ?ansicht: gemerkte Ansicht Quickinfos", pg.locator("#feldListe").count() == 1)
    pg.click(".ansicht-knoepfe a[data-ansicht=alttexte]"); pg.wait_for_selector("#bilderHeading", timeout=15000); pg.wait_for_timeout(500)
    # Das Testformular hat keine Bilder: die Alt-Text-Ansicht zeigt dann nur den Bilder-Anker, keine Feldliste
    # und seit 24.09.2026 auch kein Upload-Feld mehr (Punkt 8).
    check("Wechsel zu Alt-Texte aus der Quickinfo-Ansicht", pg.locator("#bilderHeading").count() == 1 and pg.locator("#feldListe").count() == 0 and pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("data-ansicht") == "alttexte")
    check("Alt-Text-Ansicht ohne Bilder: kein Upload-Feld, kein leeres Funktionen-Feld", pg.locator("#projUpload").count() == 0 and pg.locator("section.projekt-funktionen").count() == 0)
    check("keine Skriptfehler", not fehler_js, fehler_js[:3])
    if not BEHALTEN: print("  Testprojekt geloescht:", pg.request.delete(B + f"/api/projects/{pid}").status)
    else: print("  Projekt bleibt stehen:", pid)
    br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER"); sys.exit(1 if fehler else 0)
