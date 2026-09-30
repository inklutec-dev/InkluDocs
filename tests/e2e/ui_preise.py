#!/usr/bin/env python3
"""Klicktest oeffentliche Preisseite /preise (12.09.2026): Preisliste je Aktion
(Michael Karbe, WhatsApp 12.09.2026 — Alt-Text, Quickinfo, Herunterladen als Liste),
Zahlen aus billing.py, Englisch ueber das Sprach-Cookie, axe, keine JS-Fehler.
Aufruf: /home/claude/.venv-pw/bin/python ui_preise.py [BASIS]  (ohne Anmeldung)"""
import os, sys, re
from playwright.sync_api import sync_playwright
B = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("INKLUDOCS_E2E_URL")) or "https://staging.inkludocs.inklutec.de"
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
js = []
with sync_playwright() as p:
    br = p.chromium.launch(); ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE"); pg = ctx.new_page()
    pg.on("pageerror", lambda e: js.append(str(e)))
    print("== A. Deutsch: Preisliste je Aktion ==")
    pg.goto(B + "/preise", wait_until="networkidle"); pg.wait_for_timeout(500)
    sek = pg.locator("section[aria-labelledby='aktionen-h']")
    check("Abschnitt „Was eine Aktion kostet“ mit H2", sek.count() == 1 and pg.locator("#aktionen-h").inner_text().strip() == "Was eine Aktion kostet")
    items = [li.inner_text().strip() for li in sek.locator("ul > li").all()]
    check("9 Listenpunkte (seit 30.09.2026 mit Tagging, PDF und Word getrennt)", len(items) == 9, items)
    # Wortlaut seit 14.09.2026 (Michael Karbe, Mail 12.09.): „per KI“ + „Eingabe von Hand: kostenlos“,
    # „Übernahme … in das Dokument/Formular“, Tabellen-Export beschreibt den Inhalt (Excel mit Bildern).
    check("Alt-Text per KI 5 Credits je Bild + Eingabe von Hand kostenlos", any(i.startswith("Alt-Text per KI: 5 Credits je Bild") and i.endswith("Eingabe von Hand: kostenlos.") for i in items), items[:1])
    check("Quickinfo per KI 1 Credit je Formularfeld + Eingabe von Hand kostenlos", any(i.startswith("Quickinfo per KI: 1 Credit je Formularfeld") and i.endswith("Eingabe von Hand: kostenlos.") for i in items), items[1:2])
    # Michael Karbe, Feedback 202609230 - 1, Punkte 11 und 12 (30.09.2026)
    check("Tagging 20 Credits je Seite, nur beim Ausführen, schon getaggt kostenlos", any(i.startswith("PDF barrierefrei machen (Tagging): 20 Credits je Seite") and "schon getaggte PDF" in i for i in items), items[2:3])
    check("Herunterladen PDF: nur Bearbeitetes, ohne Bearbeitung kostenlos, Stand nur einmal, Beispiel 26 bearbeitete Alt-Texte = 40", any(i.startswith("Herunterladen einer PDF mit deinen Alt-Texten und Quickinfos: 25 Credits plus 5 Credits je angefangene 10 bearbeitete Bilder und 1 Credit je angefangene 10 bearbeitete Formularfelder") and "ohne Bearbeitung kostet das Herunterladen nichts" in i and "26 bearbeiteten Alt-Texten kostet 40 Credits" in i for i in items), items[3:4])
    check("Word: Übernahme Alt-Texte + Herunterladen als Word 25 plus 5 je 10 Bilder", any(i.startswith("Übernahme der Alt-Texte in das Dokument und Herunterladen als Word: 25 Credits plus 5 Credits je angefangene 10 Bilder") for i in items), items[4:5])
    check("Word in barrierefreie PDF 25 plus 5 je 10 Bilder", any(i.startswith("Word in barrierefreie PDF umwandeln: 25 Credits plus 5 Credits je angefangene 10 Bilder") for i in items), items[5:6])
    check("Eigenständige Formular-Projekte: Quickinfos + Formular-PDF 25 plus 1 je 10 Felder", any(i.startswith("Eigenständige Formular-Projekte: Übernahme der Quickinfos in das Formular und Herunterladen als PDF: 25 Credits plus 1 Credit je angefangene 10 Felder") for i in items), items[6:7])
    check("Tabelle mit Alt-Texten/Quickinfos (Excel mit Bildern) 10 Credits je Datei", any(i.startswith("Tabelle mit Alt-Texten oder Quickinfos erzeugen und herunterladen (CSV, JSON, Excel — Excel mit Bildern): 10 Credits je Datei") for i in items), items[5:6])
    check("InkluAgent reden kostenlos", any(i == "Mit dem InkluAgent reden: kostenlos." for i in items), items[6:7])
    check("Einleitung nennt Credits und § 19 UStG", "§ 19 UStG" in pg.locator("main").inner_text() and "Eine Währung für alles: Credits." in pg.locator("main").inner_text())
    check("Liste steht VOR „Kostenlos starten“", pg.evaluate("() => document.getElementById('aktionen-h').compareDocumentPosition(document.getElementById('free-h')) & Node.DOCUMENT_POSITION_FOLLOWING") != 0)
    check("Genau eine H1 „Preise“", pg.locator("h1").count() == 1 and pg.locator("h1").inner_text().strip() == "Preise", pg.locator("h1").all_inner_texts())
    check("Navigation markiert Preise als aktuelle Seite", pg.locator("nav a[aria-current='page']").inner_text().strip() == "Preise")
    axe(pg, "/preise (de)")
    print("== B. Englisch ueber Sprach-Cookie ==")
    ctx.add_cookies([{"name": "lang", "value": "en", "url": B}])
    pg.goto(B + "/preise", wait_until="networkidle"); pg.wait_for_timeout(500)
    check("H2 „What an action costs“", pg.locator("#aktionen-h").inner_text().strip() == "What an action costs", pg.locator("#aktionen-h").inner_text())
    items_en = [li.inner_text().strip() for li in pg.locator("section[aria-labelledby='aktionen-h'] ul > li").all()]
    check("Englisch: 9 Punkte, Alt text by AI 5 credits, Tagging 20 per page, Beispiel 40", len(items_en) == 9 and items_en[0].startswith("Alt text by AI: 5 credits per image") and items_en[2].startswith("Make a PDF accessible (tagging): 20 credits per page") and "26 edited alt texts costs 40 credits" in items_en[3], items_en[:4])
    check("Keine deutschen Reste in der Liste", not any(re.search(r"\bCredits je\b|Herunterladen", i) for i in items_en), items_en)
    check("keine JavaScript-Fehler", not js, js[:2])
    br.close()
print(f"\nErgebnis ui_preise: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
