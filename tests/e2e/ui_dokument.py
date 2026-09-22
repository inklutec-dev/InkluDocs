#!/usr/bin/env python3
"""Klicktest Ansicht „Dokument“ eines PDF-Projekts (22.09.2026): Ansichts-Wahl (Dokument ganz oben),
Struktur (H1 Projekt, H2 Upload, H2 Dokumente, H3 je Datei, dl, Knoepfe), Rueckfrage-Dialog mit
Umfang und Preis, Tagging-Lauf bis „fertig“ (Badge, Bericht-Klappe, Download-Link), Wechsel zur
Ansicht Alt-Texte und zurueck (Browser-Zurueck), Upload-Hinweistext je Ansicht, axe in Ansicht und
Dialog, keine Skriptfehler. Legt sein Projekt selbst an und loescht es (ausser --behalten).
Aufruf: /home/claude/.venv-pw/bin/python ui_dokument.py [--behalten]
Braucht INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW (Testkonto auf Staging)."""
import os
import sys
import time
from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
if not MAIL or not PW:
    sys.exit("Zugangsdaten fehlen: INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW setzen")
BEHALTEN = "--behalten" in sys.argv
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"
ok = fehler = 0


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK ", n)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:300])


def testpdf() -> bytes:
    import fitz
    d = fitz.open()
    p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Klicktest Dokument-Ansicht", fontsize=20)
    text = ("Der Bericht fasst die Massnahmen des Landes zusammen und richtet sich an die Oeffentlichkeit. "
            "Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen, und die Mittel stammen aus dem Green Bond. ") * 3
    y = 110
    for zeile in [text[i:i + 95] for i in range(0, len(text), 95)]:
        p.insert_text((50, y), zeile, fontsize=10)
        y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0)
    for yy in range(80):
        for xx in range(120):
            pix.set_pixel(xx, yy, (int(255 * xx / 120), int(255 * yy / 80), 90 + ((xx // 8 + yy // 8) % 2) * 100))
    p.insert_image(fitz.Rect(50, y + 20, 250, y + 150), pixmap=pix)
    p.insert_text((50, y + 170), "Abbildung 1: Testbild mit Farbverlauf.", fontsize=10)
    p2 = d.new_page(width=595, height=842)
    p2.insert_text((50, 70), "2. Ausblick", fontsize=14)
    p2.insert_text((50, 100), "Absatz des Ausblicks mit der Planung fuer das kommende Jahr.", fontsize=10)
    out = d.tobytes()
    d.close()
    return out


def axe(pg, name):
    pg.add_script_tag(url=AXE)
    pg.wait_for_timeout(500)
    r = pg.evaluate("async () => { const r = await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']}); return r.violations.map(v => ({id: v.id, impact: v.impact, n: v.nodes.length, html: v.nodes.slice(0,2).map(x => x.html.slice(0,120))})); }")
    ernst = [v for v in r if v["impact"] in ("serious", "critical")]
    check(f"axe {name}: 0 ernste Verstoesse", not ernst, ernst)
    if r:
        print("     axe-Hinweise (alle):", r)


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
    pg = ctx.new_page()
    fehler_js = []
    pg.on("pageerror", lambda e: fehler_js.append(str(e)))
    pg.goto(B + "/login")
    pg.fill("#email", MAIL)
    pg.fill("#password", PW)
    pg.keyboard.press("Enter")
    pg.wait_for_timeout(2500)

    print("== A. Projekt anlegen, PDF in der Ansicht Dokument hochladen ==")
    r = pg.request.post(B + "/api/projects", data={"name": "Klicktest Dokument-Ansicht " + time.strftime("%H:%M"), "tool": "pdf"})
    check("Projekt angelegt", r.ok, r.status)
    pid = r.json().get("id") or r.json().get("project_id")
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
    pg.wait_for_timeout(1200)
    check("H1 Projekt vorhanden", pg.locator("h1#projectName").count() == 1)
    check("Neues PDF-Projekt startet ohne ?ansicht in der Ansicht Dokument (Steve 22.09.)", pg.locator("#dokumenteHeading").count() == 1)
    opts = pg.locator("#ansichtSelect option").all_text_contents()
    check("Ansichts-Wahl: „Dokument“ ganz oben, dann „Alt-Texte“, „Quickinfos“ ausgegraut (keine Felder)", [o.strip() for o in opts][:2] == ["Dokument", "Alt-Texte"] and len(opts) == 3 and "nicht verfügbar" in opts[2] and pg.locator("#ansichtSelect option[value=quickinfos]").get_attribute("disabled") is not None, opts)
    check("Ansicht Dokument ist gewaehlt", pg.locator("#ansichtSelect").input_value() == "dokument")
    check("Knopf „Öffnen“ vorhanden (kein Wechsel per Pfeiltaste, WCAG 3.2.2)", pg.locator("#ansichtOeffnen").count() == 1)
    hint = pg.locator("#projUploadHint").inner_text()
    check("Upload-Hinweis spricht vom Dokument, nicht von Bildern", "barrierefrei" in hint and "Bilder extrahiert" not in hint, hint)
    check("H2 Dokumente (0) + Leerhinweis", "Dokumente (0)" in pg.locator("#dokumenteHeading").inner_text() and pg.locator("text=Noch kein Dokument hochgeladen.").count() == 1)
    pg.set_input_files("#projUpload", {"name": "klicktest_roh.pdf", "mimeType": "application/pdf", "buffer": testpdf()})
    pg.wait_for_selector("section.dok-karte", timeout=60000)
    pg.wait_for_timeout(1500)
    check("Nach dem Upload: eine Dokument-Karte", pg.locator("section.dok-karte").count() == 1)
    h3 = pg.locator("section.dok-karte h3").first.inner_text()
    check("H3 „Dokument 1: klicktest_roh.pdf“ mit Stand-Badge „Ungetaggt“", h3.startswith("Dokument 1: klicktest_roh.pdf") and "Ungetaggt" in h3, h3)
    check("Fokus nach dem Upload auf der Dokument-H3", pg.evaluate("document.activeElement && document.activeElement.id") .startswith("dok_heading_"), pg.evaluate("document.activeElement && document.activeElement.id"))
    dl = pg.locator("section.dok-karte dl.dok-meta").first.inner_text()
    check("Beschreibungsliste: Stand, Seiten 2, Sprache, Struktur, Bilder", all(k in dl for k in ("Stand", "Seiten", "Sprache", "Struktur", "Bilder")) and "\n2" in dl, dl)
    check("Vorschaubild mit Alt-Text", pg.locator("section.dok-karte img.ausgabe-vorschau").first.get_attribute("alt").startswith("Vorschau der ersten Seite"))
    check("Knopf „Barrierefrei machen“ mit Seiten und Credits im Namen", pg.locator("button[id^=dok_tag_]").count() == 1 and "2 Seiten, 2 Credits" in pg.locator("button[id^=dok_tag_]").first.inner_text())
    check("Knoepfe Alt-Texte bearbeiten / Umbenennen / Löschen", pg.locator("section.dok-karte button:has-text('Alt-Texte bearbeiten')").count() == 1 and pg.locator("section.dok-karte button:has-text('Umbenennen')").count() == 1 and pg.locator("section.dok-karte button:has-text('Löschen')").count() == 1)
    check("Noch kein Knopf „Fertige PDF herunterladen“ (ungetaggt)", pg.locator("section.dok-karte button:has-text('Fertige PDF herunterladen')").count() == 0)
    axe(pg, "Ansicht Dokument vor dem Lauf")

    print("== B. Rueckfrage und Lauf ==")
    pg.click("button[id^=dok_tag_]")
    pg.wait_for_selector("#dkLaufDialog[open]", timeout=5000)
    check("Dialog offen, Fokus auf Abbrechen", pg.evaluate("document.activeElement && document.activeElement.id") == "dkLaufCancel")
    umfang = pg.locator("#dkLaufUmfang").inner_text()
    summary = pg.locator("#dkLaufSummary").inner_text()
    check("Umfang nennt Dokument und 2 Seiten", "klicktest_roh.pdf" in umfang and "2 Seiten" in umfang, umfang)
    check("Preis 2 Credits genannt", "2 Credits" in summary, summary)
    axe(pg, "Dialog Barrierefrei machen")
    pg.keyboard.press("Escape")
    check("Escape schliesst den Dialog, Fokus zurueck auf dem Knopf", not pg.locator("#dkLaufDialog[open]").count() and (pg.evaluate("document.activeElement && document.activeElement.id") or "").startswith("dok_tag_"))
    pg.click("button[id^=dok_tag_]")
    pg.wait_for_selector("#dkLaufDialog[open]")
    pg.click("#dkLaufOk")
    pg.wait_for_timeout(1500)
    check("Dialog zu, Karte zeigt „Wird barrierefrei gemacht“", not pg.locator("#dkLaufDialog[open]").count() and "barrierefrei gemacht" in pg.locator("section.dok-karte").first.inner_text())
    check("Knopf waehrend des Laufs ausgeblendet", pg.locator("button[id^=dok_tag_]").count() == 0)
    fertig = False
    for _ in range(60):
        pg.wait_for_timeout(2000)
        if pg.locator("#dkLaufMeldung:not([hidden])").count():
            fertig = True
            break
    meld = pg.locator("#dkLaufMeldungText").inner_text() if fertig else ""
    check("Laufmeldung erscheint und nennt Struktur", fertig and "getaggt" in meld and "Elemente" in meld, meld)
    check("Fokus auf der Laufmeldung", pg.evaluate("document.activeElement && document.activeElement.id") == "dkLaufMeldung")
    h3 = pg.locator("section.dok-karte h3").first.inner_text()
    check("Badge jetzt „Getaggt …“", "Getaggt" in h3, h3)
    check("Knopf heisst jetzt „Neu taggen“", pg.locator("button[id^=dok_tag_]").first.inner_text().startswith("Neu taggen"))
    check("Knopf „Fertige PDF herunterladen“ nach dem Tagging", pg.locator("section.dok-karte button:has-text('Fertige PDF herunterladen')").count() == 1)
    pg.click("details.dok-bericht > summary")
    pg.wait_for_timeout(300)
    ber = pg.locator("details.dok-bericht").first.inner_text()
    check("Bericht: Sprache de-DE, Struktur, PDF/UA-Prüfung", "de-DE" in ber and "Struktur" in ber and "PDF/UA" in ber, ber[:300])
    dl = pg.locator("section.dok-karte dl.dok-meta").first.inner_text()
    check("Beschreibungsliste nach dem Lauf: Sprache de-DE, 1 Bild", "de-DE" in dl and "1 Bilder" in dl, dl)
    with pg.expect_download(timeout=60000) as dl_info:
        pg.click("section.dok-karte button:has-text('Fertige PDF herunterladen')")
    dl = dl_info.value
    pfad = dl.path()
    check("Fertige PDF heruntergeladen (Datei beginnt mit %PDF)", pfad is not None and open(pfad, "rb").read(5) == b"%PDF-", dl.suggested_filename)
    pg.wait_for_timeout(2500)
    st = pg.locator("output.dok-status").first.inner_text()
    check("Statuszeile nennt Download und Ablage", "Heruntergeladen" in st and "Ablage" in st, st)
    check("Ablage-Knopf im Kopf zeigt einen Eintrag", "Ablage (1)" in (pg.locator("#ausgabenTab").inner_text() if pg.locator("#ausgabenTab").count() else ""), pg.locator("#ausgabenTab").count())
    r = pg.request.get(B + f"/api/ausgaben?projekt={pid}")
    eintraege = r.json().get("ausgaben", []) if r.ok else []
    check("Ablage-Eintrag art pdf mit Datei und Bericht", len(eintraege) == 1 and eintraege[0].get("art") == "pdf" and eintraege[0].get("datei_verfuegbar") is True, eintraege[:1])
    # Nach dem Export wurde die Ansicht neu gezeichnet; die Laufmeldung des Taggings ist dann schon zu.
    if pg.locator("#dkLaufMeldung:not([hidden]) button").count():
        pg.click("#dkLaufMeldung button")
    axe(pg, "Ansicht Dokument nach dem Lauf")

    print("== B2. Hoerprobe und Strukturansicht (22.09.) ==")
    check("Link „Strukturansicht öffnen“ nach dem Tagging", pg.locator("section.dok-karte a:has-text('Strukturansicht öffnen')").count() == 1)
    check("Klappe „Hörprobe lesen“ vorhanden", pg.locator("details.dok-hoerprobe > summary").count() == 1)
    pg.click("details.dok-hoerprobe > summary")
    hp = ""
    for _ in range(20):
        pg.wait_for_timeout(1000)
        hp = pg.locator("details.dok-hoerprobe .ausgabe-hoerprobe").inner_text()
        if "wird geladen" not in hp:
            break
    check("Hörprobe geladen: Sprache, Seiten, Zusammenfassung, Grafik", all(k in hp for k in ("Sprache", "Seiten", "Zusammenfassung", "Grafik")), hp[:300])
    check("Hörprobe nennt die Seiten in Lesereihenfolge", "Seite 1" in hp and "Seite 2" in hp, hp[:300])
    axe(pg, "Ansicht Dokument mit offener Hörprobe")
    pg.click("section.dok-karte a:has-text('Strukturansicht öffnen')")
    pg.wait_for_selector("h1#strukturTitel", timeout=30000)
    pg.wait_for_timeout(500)
    check("Strukturansicht: Adresse /struktur/<projekt>/<dokument>", f"/struktur/{pid}/" in pg.url, pg.url)
    check("Strukturansicht: H1 mit Dokumentname", pg.locator("h1#strukturTitel").inner_text().startswith("Strukturansicht: klicktest_roh.pdf"), pg.locator("h1#strukturTitel").inner_text())
    check("Strukturansicht: Inhalt mit Absaetzen und Grafik", pg.locator("#strukturInhalt p").count() >= 3 and pg.locator("#strukturInhalt figure").count() >= 1, pg.locator("#strukturInhalt p").count())
    check("Strukturansicht: genau eine H1, Hörprobe als H2", pg.locator("h1").count() == 1 and pg.locator("h2#strukturHoerprobe").count() == 1)
    check("Strukturansicht: kein Skript-Text im Inhalt", "<script" not in pg.locator("#strukturInhalt").inner_html().lower())
    axe(pg, "Strukturansicht")
    pg.click("#strukturZurueck")
    pg.wait_for_selector("section.dok-karte", timeout=15000)
    pg.wait_for_timeout(800)
    check("Zurück zum Projekt fuehrt in die Ansicht Dokument", pg.locator("section.dok-karte").count() == 1 and "ansicht=dokument" in pg.url, pg.url)

    print("== C. Wechsel zur Ansicht Alt-Texte und zurueck ==")
    pg.click("section.dok-karte button:has-text('Alt-Texte bearbeiten')")
    pg.wait_for_selector("#imageFilterBar", timeout=15000)
    pg.wait_for_timeout(800)
    check("Alt-Text-Ansicht mit Filterleiste, Adresse ansicht=alttexte", "ansicht=alttexte" in pg.url and pg.locator("section.image-review").count() == 1)
    check("Ansichts-Wahl steht auf Alt-Texte", pg.locator("#ansichtSelect").input_value() == "alttexte")
    check("Dokument im Alt-Text-Modus ueber PDFix extrahiert", "PDFix" in pg.locator("h2.doc-heading").first.inner_text())
    check("Alt-Text-Ansicht: keine Dokument-Karten", pg.locator("section.dok-karte").count() == 0)
    pg.go_back()
    pg.wait_for_selector("section.dok-karte", timeout=15000)
    check("Browser-Zurueck fuehrt zur Ansicht Dokument", pg.locator("section.dok-karte").count() == 1 and "ansicht=dokument" in pg.url)
    pg.select_option("#ansichtSelect", "alttexte")
    pg.click("#ansichtOeffnen")
    pg.wait_for_selector("#imageFilterBar", timeout=15000)
    check("Auswahl + Öffnen wechselt ebenfalls", pg.locator("section.image-review").count() == 1)
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
    pg.wait_for_timeout(1000)
    check("Ohne ?ansicht oeffnet das Projekt in der zuletzt gewaehlten Ansicht (Alt-Texte, gemerkt am Projekt)", pg.locator("#imageFilterBar").count() == 1 and pg.locator("section.dok-karte").count() == 0)
    pg.select_option("#ansichtSelect", "dokument")
    pg.click("#ansichtOeffnen")
    pg.wait_for_selector("section.dok-karte", timeout=15000)
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
    pg.wait_for_timeout(1000)
    check("Nach Wechsel zurueck: ohne ?ansicht wieder die Ansicht Dokument", pg.locator("section.dok-karte").count() == 1)
    r = pg.request.post(B + f"/api/projects/{pid}/ansicht", data={"ansicht": "quatsch"})
    check("Unbekannte Ansicht wird abgewiesen (400)", r.status == 400, r.status)
    r = pg.request.post(B + "/api/projects/999999/ansicht", data={"ansicht": "dokument"})
    check("Fremdes Projekt: Ansicht speichern 404", r.status == 404, r.status)

    print("== C2. Kette „Komplett barrierefrei machen“: Rueckfrage ==")
    pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle"); pg.wait_for_timeout(1000)
    check("Knopf „Komplett barrierefrei machen“ im Kopf", pg.locator("#dkKetteBtn").count() == 1)
    pg.click("#dkKetteBtn"); pg.wait_for_selector("#dkKetteDialog[open]", timeout=5000); pg.wait_for_timeout(1500)
    plan = pg.locator("#dkKettePlan").inner_text(); summ = pg.locator("#dkKetteSummary").inner_text()
    check("Rueckfrage nennt die drei Stationen mit Zahlen", "Tagging:" in plan and "Alt-Texte:" in plan and "Quickinfos:" in plan, plan)
    check("Dokument ist schon getaggt: Tagging 0 Dokumente, 1 schon getaggt", "Tagging: 0 Dokumente" in plan and "1 Dokumente sind schon getaggt" in plan, plan)
    check("Gesamtpreis genannt", "Gesamt:" in summ and "Credits" in summ, summ)
    check("Fokus auf Abbrechen", pg.evaluate("document.activeElement && document.activeElement.id") == "dkKetteCancel")
    axe(pg, "Dialog Komplett barrierefrei machen")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    check("Escape schliesst die Rueckfrage, Fokus auf dem Knopf", not pg.locator("#dkKetteDialog[open]").count() and pg.evaluate("document.activeElement && document.activeElement.id") == "dkKetteBtn")

    print("== D. Gast und Negativfaelle ==")
    r = pg.request.get(B + f"/api/projects/{pid}/dokument-ansicht")
    check("dokument-ansicht liefert 1 Dokument mit tagging.status fertig", r.ok and len(r.json()["documents"]) == 1 and r.json()["documents"][0]["tagging"]["status"] == "fertig", r.status)
    r = pg.request.get(B + "/api/projects/999999/dokument-ansicht")
    check("fremdes Projekt 404", r.status == 404, r.status)
    check("keine Skriptfehler", not fehler_js, fehler_js[:3])

    if not BEHALTEN:
        r = pg.request.delete(B + f"/api/projects/{pid}")
        print("  Testprojekt geloescht:", r.status)
    else:
        print("  Projekt bleibt stehen:", pid)
    br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
