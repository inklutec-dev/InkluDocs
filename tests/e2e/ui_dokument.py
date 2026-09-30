#!/usr/bin/env python3
"""Klicktest Ansicht „Dokument“ eines PDF-Projekts (22.09.2026): Ansichts-Wahl (Dokument ganz oben),
Struktur (H1 Projekt, H2 Upload, H2 Dokumente, H3 je Datei, dl, Knoepfe), Rueckfrage-Dialog mit
Umfang und Preis, Tagging-Lauf bis „fertig“ (Badge, Bericht-Klappe, Download-Link), Wechsel zur
Ansicht Alt-Texte und zurueck (Browser-Zurueck), Upload-Hinweistext je Ansicht, axe in Ansicht und
Dialog, keine Skriptfehler. Legt sein Projekt selbst an und loescht es (ausser --behalten).
Aufruf: /home/claude/.venv-pw/bin/python ui_dokument.py [--behalten]
Braucht INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW (Testkonto auf Staging)."""
import os
import re
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
    p2.insert_text((50, 70), "2. Ausblick", fontsize=14, fontname="hebo")
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
    # Ansichts-Wahl als Knoepfe (Michael Karbe, Feedback 24.09.2026, Punkt 8): Links, aktuelle mit aria-current
    opts = [x.strip() for x in pg.locator(".ansicht-knoepfe [data-ansicht]").all_inner_texts()]
    check("Ansichts-Knöpfe: Dokument, Tagging, Alt-Texte, Quickinfos (ausgegraut, keine Felder), Barrierefreiheitsprüfung", [o.split("(")[0].strip() for o in opts] == ["Dokument", "Tagging", "Alt-Texte", "Quickinfos", "Barrierefreiheitsprüfung"] and pg.locator(".ansicht-knoepfe .ansicht-aus[data-ansicht=quickinfos]").count() == 1 and pg.locator(".ansicht-knoepfe a[data-ansicht=quickinfos]").count() == 0, opts)
    # Michael Karbe, Feedback 24.09.2026 - 3: „Ansicht:“ nur für Screenreader (1), nicht verfügbar mit durchgezogener Linie (2)
    check("„Ansicht:“ sichtbar weg, für Screenreader Name der Knopfliste", "visually-hidden" in (pg.locator("#ansichtTitel").get_attribute("class") or "") and pg.locator("ul.ansicht-knoepfe[aria-labelledby=ansichtTitel]").count() == 1)
    check("Nicht verfügbare Ansicht mit durchgezogener Linie", pg.evaluate("getComputedStyle(document.querySelector('.ansicht-aus')).borderTopStyle") == "solid")
    check("Aktuelle Ansicht Dokument: dunkler Knopf mit aria-current=page", pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("data-ansicht") == "dokument" and "btn-primary" in (pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("class") or ""))
    check("Andere Ansichten sind echte Links mit eigener Adresse", "ansicht=alttexte" in (pg.locator(".ansicht-knoepfe a[data-ansicht=alttexte]").get_attribute("href") or ""))
    # Projektkopf nach Michael Karbe (Mails 21.09./22.09.2026): Name + PDF-Symbol + Ansichts-Wahl in EINEM Feld,
    # keine Statusanzeige oben rechts
    check("Projektkopf: H1 und Ansichts-Wahl im selben Feld", pg.locator(".projekt-kopf h1#projectName").count() == 1 and pg.locator(".projekt-kopf .ansicht-knoepfe").count() == 1)
    check("Projektkopf: rotes PDF-Symbol mit Alt-Text „PDF-Projekt“", pg.locator(".projekt-kopf img.projekt-dateityp").count() == 1 and pg.locator(".projekt-kopf img.projekt-dateityp").get_attribute("alt") == "PDF-Projekt" and "icon-pdf" in (pg.locator(".projekt-kopf img.projekt-dateityp").get_attribute("src") or ""))
    check("Keine Statusanzeige oben rechts (Punkt 9)", pg.locator("#projectStatusBadge").count() == 0)
    check("Station steht vorne in der H1 und im Seitentitel (Steve 24.09.)", pg.locator("h1#projectName").inner_text().startswith("Dokument – Projekt: ") and pg.title().startswith("Dokument – Projekt: "), (pg.locator("h1#projectName").inner_text(), pg.title()))
    check("Leeres Projekt: kein leeres Feld „Funktionen und Einstellungen“", pg.locator("section.projekt-funktionen").count() == 0)
    hint = pg.locator("#projUploadHint").inner_text()
    check("Upload-Hinweis spricht vom Dokument, nicht von Bildern", "barrierefrei" in hint and "Bilder extrahiert" not in hint, hint)
    check("H2 Dokumente (0) + Leerhinweis", "Dokumente (0)" in pg.locator("#dokumenteHeading").inner_text() and pg.locator("text=Noch kein Dokument hochgeladen.").count() == 1)
    roh_bytes = testpdf()
    pg.set_input_files("#projUpload", {"name": "klicktest_roh.pdf", "mimeType": "application/pdf", "buffer": roh_bytes})
    pg.wait_for_selector("section.dok-karte", timeout=60000)
    pg.wait_for_timeout(1500)
    check("Nach dem Upload: eine Dokument-Karte", pg.locator("section.dok-karte").count() == 1)
    h3 = pg.locator("section.dok-karte h3").first.inner_text()
    check("H3 „Dokument 1: klicktest_roh.pdf“ mit Stand-Badge „Nicht getaggt“ (Punkt 4)", h3.startswith("Dokument 1: klicktest_roh.pdf") and "Nicht getaggt" in h3, h3)
    rb = pg.evaluate("() => { const h = document.querySelector('section.dok-karte h3').getBoundingClientRect(); const b = document.querySelector('section.dok-karte h3 .badge').getBoundingClientRect(); return [Math.round(h.right - b.right), Math.round(b.left - h.left)]; }")
    check("Stand-Abzeichen rechtsbündig in der Überschriftszeile (Mail - 2, Punkt 1)", rb[0] <= 4 and rb[1] > 150, rb)
    check("Fokus nach dem Upload auf dem Schalter der Dokument-Karte (H3 im summary)", pg.evaluate("!!(document.activeElement && document.activeElement.tagName === 'SUMMARY' && document.activeElement.querySelector('[id^=dok_heading_]'))"), pg.evaluate("document.activeElement && document.activeElement.outerHTML.slice(0,120)"))
    check("Einzelnes Dokument: Karte ist aufgeklappt", pg.locator("section.dok-karte details.dok-klappe[open]").count() == 1)
    dl = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    check("Dokument = Metadaten (Feedback 20260928 - 2, Punkt 1): Stand, Seiten: 2, Sprache — ohne Struktur und Bilder (Punkt 7)", all(k in dl for k in ("Stand: ", "Seiten: 2", "Sprache: ")) and "Struktur: " not in dl and "Bilder: " not in dl, dl)
    fs = pg.evaluate("() => [getComputedStyle(document.querySelector('ul.dok-meta')).fontSize, getComputedStyle(document.querySelector('ul.dok-meta')).fontFamily]")
    check("Dokumentinfo in Schrift und Groesse des Berichts (Punkt 4)", fs[0] == pg.evaluate("() => { const d = document.createElement('div'); d.className = 'page-text-content'; document.body.appendChild(d); const f = getComputedStyle(d).fontSize; d.remove(); return f; }"), fs)
    check("Dokumentansicht ohne Feld „Funktionen und Einstellungen“ (Feedback 24.09., Punkt 1)", pg.locator("section.projekt-funktionen").count() == 0)
    check("Vorschaubild mit Alt-Text", pg.locator("section.dok-karte img.ausgabe-vorschau").first.get_attribute("alt").startswith("Vorschau der ersten Seite"))
    check("Dokument: kein Tagging (Barrierefrei machen, Testweise taggen gibt es in „Tagging“)", pg.locator("button[id^=dok_tag_]").count() == 0 and pg.locator("button[id^=dok_test_]").count() == 0)
    check("Kein Knopf „Alt-Texte bearbeiten“; Umbenennen / Löschen da", pg.locator("section.dok-karte button:has-text('Alt-Texte bearbeiten')").count() == 0 and pg.locator("section.dok-karte button:has-text('Umbenennen')").count() == 1 and pg.locator("section.dok-karte button:has-text('Löschen')").count() == 1)
    check("Ungetaggt: keine Hörprobe, aber „PDF herunterladen“ (Feedback 20260928 - 2, Punkt 5)", pg.locator("button[id^=dok_hp_]").count() == 0 and pg.locator("button[id^=dok_export_]").count() == 1)
    kn0 = [x.split("\n")[0].strip() for x in pg.locator("section.dok-karte .dok-werkbank .ausgabe-aktionen button").all_inner_texts()]
    check("Ungetaggt: Knöpfe PDF herunterladen, Umbenennen, Löschen", kn0 == ["PDF herunterladen", "Umbenennen", "Löschen"], kn0)
    verbraucht0 = pg.request.get(B + "/api/me").json().get("abo", {}).get("verbraucht")
    pg.click("button[id^=dok_export_]"); pg.wait_for_timeout(800)
    for _ in range(20):
        zs = pg.locator("#exportSummary").inner_text()
        if "Credits" in zs or "unverändert" in zs:
            break
        pg.wait_for_timeout(500)
    check("Dialog sagt vorher: keine Tags, unverändert, keine Credits (Punkt 5)", "keine Tags" in zs and "unverändert" in zs and "keine Credits" in zs, zs)
    check("Hinweis nicht doppelt", pg.locator("#exportPdfHinweis").is_hidden())
    axe(pg, "Herunterladen-Dialog, ungetaggte PDF")
    with pg.expect_download(timeout=90000) as dl0:
        pg.click("#exportPdfBtn")
    pfad0 = dl0.value.path()
    check("Ungetaggte PDF kommt byte-gleich zurück", pfad0 is not None and open(pfad0, "rb").read() == roh_bytes)
    pg.wait_for_timeout(1500)
    st0 = pg.locator("#exportStatus").inner_text()
    check("Meldung: unverändert, keine Credits; Fokus auf der Meldung", "unverändert" in st0 and "keine Credits" in st0 and "abgebucht" not in st0 and pg.evaluate("document.activeElement && document.activeElement.id") == "exportStatus", st0)
    check("Keine Credits verbraucht", pg.request.get(B + "/api/me").json().get("abo", {}).get("verbraucht") == verbraucht0)
    check("Kein Ablage-Eintrag für die unveränderte Datei", len((pg.request.get(B + f"/api/ausgaben?projekt={pid}").json() or {}).get("ausgaben", [])) == 0)
    pg.click("#exportCancelBtn"); pg.wait_for_timeout(400)
    check("Knöpfe unter der Linie, darunter nichts weiter (Punkt 2)",
          pg.locator("section.dok-karte .ausgabe-text button").count() == 0
          and pg.evaluate("getComputedStyle(document.querySelector('.dok-werkbank')).borderTopStyle") == "solid"
          and pg.evaluate("document.querySelector('.dok-werkbank').children.length") == 1)
    axe(pg, "Ansicht Dokument vor dem Lauf")

    print("== A1. Ansicht Tagging (Feedback 20260928 - 2, Punkte 1, 6, 7) ==")
    pg.click(".ansicht-knoepfe a[data-ansicht=tagging]")
    # beide Ansichten haben Dokument-Karten: auf die H1 der neuen Ansicht warten (der Wechsel wartet bewusst ~1 s aufs Speichern)
    pg.wait_for_function("() => (document.getElementById('projectName') || {}).textContent.startsWith('Tagging')", timeout=15000); pg.wait_for_timeout(500)
    check("Adresse ansicht=tagging, H1 „Tagging – Projekt: …“, Knopf aktuell", "ansicht=tagging" in pg.url and pg.locator("h1#projectName").inner_text().startswith("Tagging – Projekt: ") and pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("data-ansicht") == "tagging", pg.locator("h1#projectName").inner_text())
    check("Tagging: kein Upload-Feld (Hochladen nur in „Dokument“)", pg.locator("#projUpload").count() == 0)
    tl = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    check("Tagging-Karte zeigt Struktur und Bilder, keine Metadaten (Punkt 7)", "Struktur: " in tl and "Bilder: " in tl and "Titel: " not in tl and "Anwendung: " not in tl, tl)
    # 20 Credits je Seite (Michael Karbe, Feedback 202609230 - 1, Punkt 11)
    check("Knopf „Barrierefrei machen“ mit Seiten und Credits im Namen (20 je Seite)", pg.locator("button[id^=dok_tag_]").count() == 1 and "2 Seiten, 40 Credits" in pg.locator("button[id^=dok_tag_]").first.inner_text(), pg.locator("button[id^=dok_tag_]").first.inner_text() if pg.locator("button[id^=dok_tag_]").count() else "")
    check("Ungetaggte Quelle: kein Hinweis „schon getaggt“", pg.locator("[id^=dok_schon_getaggt_]").count() == 0)
    check("Tagging: keine Dateiknöpfe (Umbenennen, Löschen, Herunterladen)", pg.locator("section.dok-karte button:has-text('Umbenennen')").count() == 0 and pg.locator("section.dok-karte button:has-text('Löschen')").count() == 0 and pg.locator("button[id^=dok_export_]").count() == 0)
    check("Knöpfe unter einer Linie über die volle Breite, nicht neben dem Vorschaubild (Mail - 3, Punkt 3)",
          pg.locator("section.dok-karte .dok-werkbank .ausgabe-aktionen button[id^=dok_tag_]").count() == 1
          and pg.locator("section.dok-karte .ausgabe-text button").count() == 0)
    axe(pg, "Ansicht Tagging vor dem Lauf")

    print("== A2. Testweise taggen (Mail - 2, Punkt 3): kostenlos, Testmodus, Original bleibt ==")
    kn = pg.locator("button[id^=dok_test_]")
    check("Knopf „Testweise taggen“ (kostenlos, im Testmodus)", kn.count() == 1 and "kostenlos, im Testmodus" in kn.first.inner_text(), kn.first.inner_text() if kn.count() else "")
    guthaben_vorher = pg.request.get(B + "/api/me").json().get("abo", {})
    kn.first.click()
    pg.wait_for_timeout(1200)
    check("Statuszeile „Testlauf gestartet“ mit Fokus", "Testlauf gestartet" in pg.locator("output[id^=dok_status_]").first.inner_text() and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_status_"))
    fertig = False
    for _ in range(90):
        pg.wait_for_timeout(2000)
        if pg.locator("section.dok-karte .dok-ergebnis").count():
            fertig = True
            break
    meld = pg.locator("section.dok-karte .dok-ergebnis p").first.inner_text() if fertig else ""
    check("Ergebnis des Testlaufs in der Karte, Fokus darauf", fertig and meld.startswith("Testlauf vom") and "Elemente" in meld and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_ergebnis_text_"), meld)
    check("Dokument bleibt „Nicht getaggt“ (Original unverändert)", "Nicht getaggt" in pg.locator("section.dok-karte h3").first.inner_text())
    check("Kein Herunterladen der Testfassung", pg.locator("button[id^=dok_export_]").count() == 0 and pg.locator("section.dok-karte a[href*='test']").count() == 0)
    pg.click("section.dok-karte .dok-ergebnis button"); pg.wait_for_timeout(300)
    pg.click("details.dok-test > summary")
    hp = ""
    for _ in range(20):
        pg.wait_for_timeout(1000)
        hp = pg.locator("details.dok-test .dok-test-hoerprobe").inner_text() if pg.locator("details.dok-test .dok-test-hoerprobe").count() else ""
        if hp and "wird geladen" not in hp:
            break
    check("Klappe „Ergebnis des Testlaufs“ mit Hörprobe der Testfassung", "Zusammenfassung" in hp and "Seite 1" in hp, hp[:200])
    check("Testlauf kostet nichts", pg.request.get(B + "/api/me").json().get("abo", {}).get("verbraucht") == guthaben_vorher.get("verbraucht"))
    axe(pg, "Ansicht Tagging mit Ergebnis des Testlaufs")

    print("== B. Rueckfrage und Lauf ==")
    pg.click("button[id^=dok_tag_]")
    pg.wait_for_selector("#dkLaufDialog[open]", timeout=5000)
    check("Dialog offen, Fokus auf Abbrechen", pg.evaluate("document.activeElement && document.activeElement.id") == "dkLaufCancel")
    umfang = pg.locator("#dkLaufUmfang").inner_text()
    summary = pg.locator("#dkLaufSummary").inner_text()
    check("Umfang nennt Dokument und 2 Seiten", "klicktest_roh.pdf" in umfang and "2 Seiten" in umfang, umfang)
    check("Preis 40 Credits genannt, 20 Credits je Seite, Tagging nicht noch einmal beim Herunterladen (Punkte 11, 12)", "40 Credits" in summary and "20 Credits je Seite" in summary and "beim Herunterladen wird es nicht noch einmal berechnet" in summary, summary)
    verbraucht_vor_tagging = pg.request.get(B + "/api/me").json().get("abo", {}).get("verbraucht")
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
        if pg.locator("section.dok-karte .dok-ergebnis").count():
            fertig = True
            break
    meld = pg.locator("section.dok-karte .dok-ergebnis p").first.inner_text() if fertig else ""
    # Mail - 2, Punkt 2: Ergebnis UNTER dem Dokument in der Karte, farbig, mit Fokus
    check("Ergebnis in der Karte, nennt Struktur", fertig and "getaggt" in meld and "Elemente" in meld, meld)
    vb_nach = pg.request.get(B + "/api/me").json().get("abo", {}).get("verbraucht")
    check("Tagging hat genau 40 Credits gebucht (2 Seiten × 20)", isinstance(vb_nach, int) and isinstance(verbraucht_vor_tagging, int) and vb_nach - verbraucht_vor_tagging == 40, (verbraucht_vor_tagging, vb_nach))
    check("Fokus auf dem Ergebnis in der Karte", str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_ergebnis_text_"))
    check("Ergebnis grün hinterlegt (Erfolg)", pg.evaluate("getComputedStyle(document.querySelector('.dok-ergebnis')).backgroundColor") == "rgb(240, 253, 244)")
    check("Keine Meldung mehr oben über der Liste", pg.locator("#dkLaufMeldung:not([hidden])").count() == 0)
    h3 = pg.locator("section.dok-karte h3").first.inner_text()
    check("Badge jetzt „Getaggt …“", "Getaggt" in h3, h3)
    check("Knopf heisst jetzt „Neu taggen“", pg.locator("button[id^=dok_tag_]").first.inner_text().startswith("Neu taggen"))
    check("Tagging: kein „PDF herunterladen“ (das gibt es in „Dokument“), Knopf „Hörprobe“ da", pg.locator("button[id^=dok_export_]").count() == 0 and pg.locator("button[id^=dok_hp_]").count() == 1)
    pg.click("details.dok-bericht > summary")
    pg.wait_for_timeout(300)
    ber = pg.locator("details.dok-bericht").first.inner_text()
    check("Bericht: Zeit + PDF/UA-Prüfung, ohne Dokumentinfos und ohne „Hinweis“ (Punkte 9, 10)", "Getaggt am" in ber and "PDF/UA" in ber and "Dokumentsprache" not in ber and "Struktur:" not in ber and "Hinweis –" not in ber, ber[:300])
    check("Kein Testmodus-Hinweis im Bericht (Punkt 7)", "Testmodus" not in ber)
    check("Bericht ohne „In Ordnung“-Zeilen (Punkt 6)", "In Ordnung –" not in ber, ber[-400:])
    tl = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    check("Tagging-Karte nach dem Lauf: Struktur und 1 Bild", "1 Bilder" in tl and "Struktur: " in tl, tl)
    axe(pg, "Ansicht Tagging nach dem Lauf (mit Ergebnis)")
    pg.click("section.dok-karte .dok-ergebnis button")
    pg.wait_for_timeout(300)
    check("„Meldung schließen“: weg, Fokus auf dem Schalter der Karte", pg.locator(".dok-ergebnis").count() == 0 and pg.evaluate("document.activeElement && document.activeElement.tagName") == "SUMMARY")

    print("== B1. Hörprobe als Dialog (Punkte 1 und 6) ==")
    pg.click("button[id^=dok_hp_]")
    pg.wait_for_selector("#dkHoerprobeDialog[open]", timeout=5000)
    hp = ""
    for _ in range(20):
        pg.wait_for_timeout(1000)
        hp = pg.locator("#dkHpInhalt").inner_text()
        if "wird geladen" not in hp:
            break
    check("Hörprobe-Dialog: Sprache, Seiten, Zusammenfassung, Grafik, Seiten in Lesereihenfolge", all(k in hp for k in ("Sprache", "Seiten", "Zusammenfassung", "Grafik", "Seite 1", "Seite 2")), hp[:300])
    check("Hörprobe-Dialog: Überschrift mit Dokumentname, Fokus im Dialog", "klicktest_roh.pdf" in pg.locator("#dkHpHeading").inner_text() and pg.evaluate("document.getElementById('dkHoerprobeDialog').contains(document.activeElement)"))
    axe(pg, "Hörprobe-Dialog")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(400)
    check("Escape schließt, Fokus zurück auf „Hörprobe“", not pg.locator("#dkHoerprobeDialog[open]").count() and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_hp_"))

    print("== B1b. Ansicht Dokument nach dem Lauf: Metadaten, Knöpfe Hörprobe / Herunterladen / Umbenennen / Löschen ==")
    pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
    pg.wait_for_function("() => (document.getElementById('projectName') || {}).textContent.startsWith('Dokument')", timeout=15000); pg.wait_for_timeout(500)
    mt = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    check("Dokumentinfos: Titel, Anwendung, Erstellt mit, Stand: Getaggt, PDF-Standard (Punkte 2-5)", all(k in mt for k in ("Titel: ", "Anwendung: ", "Erstellt mit: ", "Stand: Getaggt", "PDF-Standard: ")) and mt.index("Titel:") < mt.index("Anwendung:") < mt.index("Erstellt mit:") < mt.index("Stand:"), mt)
    check("PDF-Standard nach dem Tagging: PDF/UA-1, Sprache de-DE", "PDF-Standard: PDF/UA-1" in mt and "de-DE" in mt, mt)
    check("Kein Testmodus-Hinweis in den Dokumentinfos (Punkt 7)", "Testmodus" not in mt)
    kn = [x.split("\n")[0].strip() for x in pg.locator("section.dok-karte .dok-werkbank .ausgabe-aktionen button").all_inner_texts()]
    check("Knöpfe genau: Hörprobe, PDF herunterladen, Umbenennen, Löschen (Punkt 1)", kn == ["Hörprobe", "PDF herunterladen", "Umbenennen", "Löschen"], kn)
    check("Unter den Knöpfen nichts weiter: kein Ergebnis, kein Bericht, kein Testlauf, kein Hinweis (Punkt 2)", pg.evaluate("document.querySelector('.dok-werkbank').children.length") == 1 and pg.locator("section.dok-karte details.dok-bericht, section.dok-karte details.dok-test, section.dok-karte .dok-ergebnis").count() == 0)
    axe(pg, "Ansicht Dokument nach dem Lauf")
    # Feedback 28.09.2026 - 1, Punkt 4: „PDF herunterladen“ öffnet dieselbe Rückfrage wie früher in „Alt-Texte“ — nur mit der PDF
    pg.click("button[id^=dok_export_]")
    pg.wait_for_timeout(1500)
    check("„PDF herunterladen“ öffnet den Herunterladen-Dialog (modal)", pg.locator("#exportPanel").evaluate("d => d.open") is True)
    check("Dialog-Überschrift „PDF herunterladen“", pg.locator("#exportPanelHeading").inner_text().strip() == "PDF herunterladen", pg.locator("#exportPanelHeading").inner_text())
    sichtbar = [b.inner_text().strip() for b in pg.locator("#exportPanel button").all() if b.is_visible()]
    check("Im Dialog nur „Als PDF“ und „Abbrechen“ (keine Tabellen, die gibt es in „Alt-Texte“)", sichtbar == ["Als PDF", "Abbrechen"], sichtbar)
    for _ in range(20):
        zs = pg.locator("#exportSummary").inner_text()
        if "kostet" in zs:
            break
        pg.wait_for_timeout(500)
    # Punkt 12: getaggt, nichts bearbeitet -> das Herunterladen kostet nichts; der Dialog sagt es VORHER
    check("Zusammenfassung nennt Bilder mit Text, kein Tabellen-Export, und dass das Herunterladen nichts kostet (nichts bearbeitet)", "Text" in zs and "CSV" not in zs and "kostet nichts" in zs and "keine Alt-Texte und keine Quickinfos bearbeitet" in zs and "Dieser Export kostet" not in zs, zs)
    verbraucht_vor_dl = pg.request.get(B + "/api/me").json().get("abo", {}).get("verbraucht")
    check("Fokus liegt im Dialog", pg.evaluate("document.getElementById('exportPanel').contains(document.activeElement)"))
    axe(pg, "Herunterladen-Dialog in der Ansicht Dokument")
    with pg.expect_download(timeout=90000) as dl_info:
        pg.click("#exportPdfBtn")
    pfad = dl_info.value.path()
    check("PDF aus der Ansicht Dokument heruntergeladen (%PDF)", pfad is not None and open(pfad, "rb").read(5) == b"%PDF-", dl_info.value.suggested_filename)
    pg.wait_for_timeout(2500)
    st = pg.locator("#exportStatus").inner_text()
    check("Statuszeile im Dialog nennt den Download, Fokus darauf", "Heruntergeladen" in st and pg.evaluate("document.activeElement && document.activeElement.id") == "exportStatus", st)
    check("Statuszeile: keine Credits berechnet (Punkt 12)", "Es wurden keine Credits berechnet." in st and "abgebucht" not in st, st)
    check("Herunterladen ohne Bearbeitung hat nichts gebucht", pg.request.get(B + "/api/me").json().get("abo", {}).get("verbraucht") == verbraucht_vor_dl)
    check("Abbrechen heißt jetzt „Zurück zum Projekt“", pg.locator("#exportCancelBtn").inner_text().strip() == "Zurück zum Projekt")
    axe(pg, "Herunterladen-Dialog nach dem Download")
    pg.click("#exportCancelBtn")
    pg.wait_for_timeout(500)
    check("Dialog zu, Fokus zurück auf „PDF herunterladen“", pg.locator("#exportPanel").evaluate("d => d.open") is False and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_export_"))
    r = pg.request.get(B + f"/api/ausgaben?projekt={pid}")
    eintraege = r.json().get("ausgaben", []) if r.ok else []
    check("Ablage-Eintrag art pdf mit Datei", len(eintraege) == 1 and eintraege[0].get("art") == "pdf" and eintraege[0].get("datei_verfuegbar") is True, eintraege[:1])

    print("== B2. Strukturansicht-Link ausgeblendet (22.09.) ==")
    check("Strukturansicht-Link in der Dokument-Karte ausgeblendet (Steve 24.09.; gibt es in der Prüfung)", pg.locator("section.dok-karte a[id^=dok_struktur_]").count() == 0)

    print("== B3. KI-basierte Pruefung nicht mehr in „Dokument“ (Mail - 3, Punkt 13) ==")
    check("Keine KI-Prüfung in der Dokument-Karte", pg.locator("details.dok-pruefung").count() == 0 and pg.locator("button[id^=dok_pruef_]").count() == 0)

    check("Kein Urteil an der Karte (Feedback 24.09., Punkt 6)", pg.locator("p.dok-urteil").count() == 0)

    print("== C. Wechsel zur Ansicht Alt-Texte und zurueck ==")
    pg.click(".ansicht-knoepfe a[data-ansicht=alttexte]")
    pg.wait_for_selector("#imageFilterBar", timeout=15000)
    pg.wait_for_timeout(800)
    check("Alt-Text-Ansicht mit Filterleiste, Adresse ansicht=alttexte", "ansicht=alttexte" in pg.url and pg.locator("section.image-review").count() == 1)
    check("Ansichts-Knopf Alt-Texte ist jetzt der aktuelle", pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("data-ansicht") == "alttexte")
    check("Dokument im Alt-Text-Modus ueber PDFix extrahiert", "PDFix" in pg.locator("h2.doc-heading").first.inner_text())
    check("Alt-Text-Ansicht: keine Dokument-Karten", pg.locator("section.dok-karte").count() == 0)
    check("Alt-Text-Ansicht: kein Upload-Feld (Punkt 8)", pg.locator("#projUploadZone").count() == 0 and pg.locator("#projUpload").count() == 0)
    check("Alt-Text-Ansicht: gleicher Projektkopf mit PDF-Symbol, ohne Statusanzeige", pg.locator(".projekt-kopf img.projekt-dateityp").count() == 1 and pg.locator("#projectStatusBadge").count() == 0)
    check("Alt-Text-Ansicht: „Alt-Texte generieren“ und Einstellungen im Feld „Funktionen und Einstellungen“", pg.locator("section.projekt-funktionen #generateBtn").count() == 1 and pg.locator("section.projekt-funktionen #altLangSelect").count() == 1 and pg.locator("section.projekt-funktionen #useContextToggle").count() == 1)
    axe(pg, "Ansicht Alt-Texte mit neuem Kopf")
    # Feedback 28.09.2026 - 1, Punkte 1-3: in „Alt-Texte“ kein Umbenennen/Löschen und keine PDF; eigener Knopf für die Textliste
    da = pg.locator(".doc-block .doc-actions").first
    check("Alt-Texte: am Dokument kein Umbenennen/Löschen, dafür Generieren + „Alt-Texte herunterladen“ (Punkt 1, 2)", da.locator("button:has-text('Umbenennen')").count() == 0 and da.locator("button:has-text('Löschen')").count() == 0 and "Alt-Texte generieren" in da.inner_text() and "Alt-Texte herunterladen" in da.inner_text(), da.inner_text())
    check("Alt-Texte: Projektknopf heißt „Alt-Texte herunterladen“", pg.locator("#exportOpenBtn").inner_text().split("\n")[0].strip() == "Alt-Texte herunterladen", pg.locator("#exportOpenBtn").inner_text())
    pg.click("#exportOpenBtn")
    pg.wait_for_timeout(1500)
    sichtbar = [b.inner_text().strip() for b in pg.locator("#exportPanel button").all() if b.is_visible()]
    check("Alt-Texte-Dialog: nur Excel, JSON, CSV — keine PDF (Punkt 3)", sichtbar == ["Als Excel", "Als JSON", "Als CSV", "Abbrechen"] and pg.locator("#exportPanelHeading").inner_text().strip() == "Alt-Texte herunterladen", (sichtbar, pg.locator("#exportPanelHeading").inner_text()))
    for _ in range(20):
        zs = pg.locator("#exportSummary").inner_text()
        if "Credits" in zs:
            break
        pg.wait_for_timeout(500)
    check("Alt-Texte-Dialog: Preis der Textliste, kein PDF-Preis", "CSV" in zs and "Dieser Export kostet" not in zs, zs)
    axe(pg, "Herunterladen-Dialog in der Ansicht Alt-Texte")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(400)
    check("Escape schließt, Fokus zurück auf den Knopf", pg.locator("#exportPanel").evaluate("d => d.open") is False and pg.evaluate("document.activeElement && document.activeElement.id") == "exportOpenBtn")
    pg.go_back()
    pg.wait_for_selector("section.dok-karte", timeout=15000)
    check("Browser-Zurueck fuehrt zur Ansicht Dokument", pg.locator("section.dok-karte").count() == 1 and "ansicht=dokument" in pg.url)
    pg.click(".ansicht-knoepfe a[data-ansicht=alttexte]")
    pg.wait_for_selector("#imageFilterBar", timeout=15000)
    check("Ein Klick auf den Ansichts-Knopf wechselt", pg.locator("section.image-review").count() == 1)
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
    pg.wait_for_timeout(1000)
    check("Ohne ?ansicht oeffnet das Projekt in der zuletzt gewaehlten Ansicht (Alt-Texte, gemerkt am Projekt)", pg.locator("#imageFilterBar").count() == 1 and pg.locator("section.dok-karte").count() == 0)
    pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
    pg.wait_for_selector("section.dok-karte", timeout=15000)
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
    pg.wait_for_timeout(1000)
    check("Nach Wechsel zurueck: ohne ?ansicht wieder die Ansicht Dokument", pg.locator("section.dok-karte").count() == 1)
    r = pg.request.post(B + f"/api/projects/{pid}/ansicht", data={"ansicht": "quatsch"})
    check("Unbekannte Ansicht wird abgewiesen (400)", r.status == 400, r.status)
    r = pg.request.post(B + "/api/projects/999999/ansicht", data={"ansicht": "dokument"})
    check("Fremdes Projekt: Ansicht speichern 404", r.status == 404, r.status)

    print("== C2. Kein Feld „Funktionen und Einstellungen“ in der Dokumentansicht (Feedback 24.09., Punkt 1) ==")
    pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle"); pg.wait_for_timeout(1000)
    check("Kein „Komplett barrierefrei machen“, keine Ablage, kein Funktionen-Feld", pg.locator("#dkKetteBtn").count() == 0 and pg.locator("#ausgabenTab").count() == 0 and pg.locator("section.projekt-funktionen").count() == 0)
    if False:   # Kette-Rueckfrage: Knopf ausgeblendet (ZEIGE_PROJEKT_KNOEPFE), Pruefungen fuer spaeter behalten
        pg.click("#dkKetteBtn"); pg.wait_for_selector("#dkKetteDialog[open]", timeout=5000); pg.wait_for_timeout(1500)
        plan = pg.locator("#dkKettePlan").inner_text(); summ = pg.locator("#dkKetteSummary").inner_text()
        check("Rueckfrage nennt die drei Stationen mit Zahlen", "Tagging:" in plan and "Alt-Texte:" in plan and "Quickinfos:" in plan, plan)
        check("Dokument ist schon getaggt: Tagging 0 Dokumente, 1 schon getaggt", "Tagging: 0 Dokumente" in plan and "1 Dokumente sind schon getaggt" in plan, plan)
        check("Gesamtpreis genannt", "Gesamt:" in summ and "Credits" in summ, summ)
        check("Fokus auf Abbrechen", pg.evaluate("document.activeElement && document.activeElement.id") == "dkKetteCancel")
        axe(pg, "Dialog Komplett barrierefrei machen")
        pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
        check("Escape schliesst die Rueckfrage, Fokus auf dem Knopf", not pg.locator("#dkKetteDialog[open]").count() and pg.evaluate("document.activeElement && document.activeElement.id") == "dkKetteBtn")

    print("== C2b. Station „Barrierefreiheitsprüfung“ (nur veraPDF, Michaels Feedback 20260928 - 2) ==")
    opts = [x.strip() for x in pg.locator(".ansicht-knoepfe [data-ansicht]").all_inner_texts()]
    check("Ansichts-Knöpfe nennen „Barrierefreiheitsprüfung“ als letzte Station (Punkt 3)", opts[-1] == "Barrierefreiheitsprüfung", opts)
    pg.click(".ansicht-knoepfe a[data-ansicht=abschluss]")
    pg.wait_for_selector("section.ab-karte", timeout=15000); pg.wait_for_timeout(800)
    check("H1 nennt die Station „Barrierefreiheitsprüfung“", pg.locator("h1#projectName").inner_text().startswith("Barrierefreiheitsprüfung – Projekt: "), pg.locator("h1#projectName").inner_text())
    check("Adresse ansicht=abschluss, Kopf mit PDF-Symbol, eine Karte offen", "ansicht=abschluss" in pg.url and pg.locator(".projekt-kopf img.projekt-dateityp").count() == 1 and pg.locator("details.ab-klappe[open]").count() == 1)
    check("Karte: „Noch keine Prüfdatei“, Knopf „Prüfdatei erstellen“ (kostenlos)", "Noch keine Prüfdatei" in pg.locator("section.ab-karte h3").inner_text() and pg.locator("button[id^=ab_erstellen_]").count() == 1 and "kostenlos" in pg.locator("button[id^=ab_erstellen_]").inner_text())
    check("Kein Herunterladen in der Prüfung (Punkt 6), kein Upload-Feld", pg.locator("button[id^=ab_export_]").count() == 0 and pg.locator("#abAlleBtn").count() == 0 and "PDF herunterladen" not in pg.locator("main").inner_text() and pg.locator("#projUpload").count() == 0)
    check("Keine KI-basierte Prüfung (ausgeblendet, Punkt 9)", pg.locator("section.ab-ki").count() == 0 and pg.locator("button[id^=dok_pruef_]").count() == 0 and "KI-basierte" not in pg.locator("main").inner_text())
    axe(pg, "Prüfung vor der Prüfdatei")
    pg.click("button[id^=ab_erstellen_]")
    for _ in range(60):
        pg.wait_for_timeout(1500)
        if "Prüfdatei erstellt" in (pg.locator("output[id^=ab_status_]").first.inner_text() if pg.locator("output[id^=ab_status_]").count() else ""):
            break
    st = pg.locator("output[id^=ab_status_]").first.inner_text()
    check("Prüfdatei erstellt, Statuszeile mit Fokus", "Prüfdatei erstellt" in st and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("ab_status_"), st)
    meta = pg.locator("section.ab-karte ul.dok-meta").first.inner_text()
    check("Stand: Prüfdatei erstellt am …, aktuell, Norm-Prüfung (veraPDF), Problemstellen", all(k in meta for k in ("Prüfdatei: erstellt am", "Stand: aktuell", "Norm-Prüfung PDF/UA-1 (veraPDF): ", "Problemstellen: ")), meta)
    check("veraPDF nennt Ergebnis mit Zahl der Prüfpunkte (Michael Karbe 28.09.2026)", re.search(r"Norm-Prüfung PDF/UA-1 \(veraPDF\): (bestanden, [\d.]+ Prüfpunkte erfüllt|nicht bestanden, [\d.]+ Prüfpunkte verletzt)", meta) is not None, meta)
    karte = pg.locator("section.ab-karte").first.inner_text()
    # Michael Karbe, Feedback 202609230 - 1, Punkt 6: gekürzter Satz
    check("Hinweis gekürzt: „Geprüft wird mit veraPDF gegen PDF/UA-1. Jede Problemstelle nennt die Regelnummer von veraPDF.“ (Punkt 6)", "Geprüft wird mit veraPDF gegen PDF/UA-1. Jede Problemstelle nennt die Regelnummer von veraPDF." in karte and "demselben Werkzeug" not in karte and "Zusätzlich prüft InkluDocs" not in karte, karte[:400])
    kopf = pg.locator(".projekt-kopf").inner_text()
    check("Satz oben unter den Ansichts-Knöpfen, gekürzt (Punkt 9)", "Hier prüfst du die fertige Datei mit veraPDF. Anzeige der Problemstellen im Prüfbericht." in kopf and pg.evaluate("(() => { const k = document.querySelector('.projekt-kopf .ansicht-wahl'); const h = document.getElementById('abKopfHinweis'); return !!(k && h && (k.compareDocumentPosition(h) & Node.DOCUMENT_POSITION_FOLLOWING)); })()"), kopf)
    check("Unter „Dokumente (n)“ kein Hinweissatz mehr (Punkt 9)", pg.evaluate("(() => { const n = document.getElementById('dokumenteHeading').nextElementSibling; return n && n.id; })()") == "abListe")
    check("Linie über „Prüfdatei neu erstellen“ (Punkt 2)", pg.evaluate("(() => { const b = document.querySelector('button[id^=ab_erstellen_]'); const w = b && b.closest('.ab-werkbank'); return !!w && getComputedStyle(w).borderTopStyle === 'solid'; })()"))
    check("Kein Satz „Keine der Problemstellen gehört zu einer bestimmten Seite.“ (Punkt 7)", "Keine der Problemstellen gehört zu einer bestimmten Seite" not in karte)
    pg.wait_for_timeout(1500)
    check("Keine Sprache/Zusammenfassung in der Prüfung (Punkt 10)", pg.locator("ul[id^=ab_kopf_]").count() == 0)
    probleme = pg.locator("ol.ab-problemliste > li").all_inner_texts()
    # Feedback 202609230 - 1, Punkt 8: kein „veraPDF (PDF/UA-1):“ mehr in der Zeile, die Regelnummer bleibt
    check("Nur veraPDF-Problemstellen, jede mit Regelnummer, ohne „veraPDF“ in der Zeile (Punkt 8)", all("veraPDF" not in x and "(Regel" in x for x in probleme) and not any(q in " ".join(probleme) for q in ("Vollständigkeit:", "Struktur:", "KI-basierte")), probleme[:4])
    if probleme:
        st_ab = pg.evaluate("""(() => { const m = document.querySelector('section.ab-karte ul.dok-meta li'); const p = document.querySelector('ol.ab-problemliste > li');
            const a = getComputedStyle(m), b = getComputedStyle(p); const s = document.querySelector('.ab-problemklappe > summary'); const o = document.querySelector('ol.ab-problemliste');
            const alle = document.querySelectorAll('ol.ab-problemliste > li');
            return {gleich: a.fontFamily === b.fontFamily && a.fontSize === b.fontSize && a.lineHeight === b.lineHeight, anzahl: alle.length, li_abstand: alle.length > 1 ? parseFloat(getComputedStyle(alle[0]).marginBottom) : null, ol_oben: parseFloat(getComputedStyle(o).marginTop), sum_unten: parseFloat(getComputedStyle(s).marginBottom)}; })()""")
        check("Problemtext in Schrift, Größe und Zeilenhöhe der Dokumentinfos (Punkt 4)", st_ab["gleich"], st_ab)
        check("Abstand zwischen den Problemstellen (Punkt 5, bei mehr als einer) und unter „Problemstellen“ (Punkt 3)", (st_ab["li_abstand"] is None or st_ab["li_abstand"] >= 8) and st_ab["ol_oben"] + st_ab["sum_unten"] >= 14, st_ab)
    check("Kein Filter „Ganzes Dokument / Nur Problemstellen“ mehr (Punkt 12)", pg.locator("fieldset.ab-filter").count() == 0)
    check("Kein „!“ vor den Problemen (Punkt 7)", pg.locator(".ab-marke").count() == 0)
    n_prob = pg.locator("ol.ab-problemliste > li").count()
    if n_prob:
        pg.wait_for_selector("section.ab-seite", timeout=15000)
        kopfzeile = pg.locator("h4[id^=ab_seite_heading_]").inner_text()
        check("Seitenansicht zeigt nur Problemseiten: „Problemseite 1 von n: Seite x“", kopfzeile.startswith("Problemseite 1 von "), kopfzeile)
        check("Linie über der seitenweisen Anzeige (Punkt 11)", pg.evaluate("getComputedStyle(document.querySelector('section.ab-seite')).borderTopStyle") == "solid")
        nav = pg.locator(".ab-seitennav > button").all_inner_texts()
        check("„Vorherige Seite“ und „Nächste Seite“ direkt nebeneinander (Punkt 8)", nav[:2] == ["Vorherige Seite", "Nächste Seite"], nav)
        check("Seitenwahl nennt nur Problemseiten", all("Problemstelle" in o for o in pg.locator("select[id^=ab_seitenwahl_] option").all_inner_texts()))
        pg.wait_for_timeout(1000)
        check("Seitenbild geladen, mit Alt-Text", pg.evaluate("(() => { const i = document.querySelector('img.ab-seitenbild'); return !!(i && i.complete && i.naturalWidth > 0 && i.alt.startsWith('Seitenbild von Seite')); })()"))
        check("Hörprobe: Inhalt mit lang-Attribut der Dokumentsprache (Punkt 9)", pg.locator(".ab-hoerprobe span[lang]").count() > 0 and (pg.locator(".ab-hoerprobe span[lang]").first.get_attribute("lang") or "").lower().startswith("de"), pg.locator(".ab-hoerprobe").first.inner_html()[:200])
        check("Knopf „Seite vorlesen“ (aria-pressed)", pg.locator("button[id^=ab_vorlesen_]").get_attribute("aria-pressed") == "false")
        pg.click("button[id^=ab_vorlesen_]"); pg.wait_for_timeout(1500)
        check("Vorlesen ohne Stimme auf dem Geraet: kein Absturz, Knopf bleibt bedienbar", pg.locator("button[id^=ab_vorlesen_]").count() == 1 and not fehler_js, fehler_js[:2])
    else:
        check("Ohne Probleme: Hinweis „Keine Problemstellen gefunden“, keine Seitenansicht", "Keine Problemstellen gefunden" in pg.locator("div.ab-detail").inner_text() and pg.locator("section.ab-seite").count() == 0)
    axe(pg, "Prüfung mit Prüfdatei")

    # C2c/C2d (KI-Pruefung und Korrektur in der Station) entfallen, solange die KI-Pruefung ausgeblendet ist
    # (Michael Karbe, Feedback 20260928 - 2, Punkt 9; Schalter ZEIGE_KI in frontend/abschluss.js).
    pg.click("a[id^=ab_struktur_]")
    pg.wait_for_selector("h1#strukturTitel", timeout=30000)
    check("Mit eigenem Screenreader prüfen: Strukturansicht der fertigen Datei", "(fertige Datei)" in pg.locator("h1#strukturTitel").inner_text() and "quelle=abschluss" in pg.url, pg.locator("h1#strukturTitel").inner_text())
    check("Strukturansicht: genau eine H1, Hörprobe als H2, Inhalt mit Absätzen, kein Skript-Text", pg.locator("h1").count() == 1 and pg.locator("h2#strukturHoerprobe").count() == 1 and pg.locator("#strukturInhalt p").count() >= 3 and "<script" not in pg.locator("#strukturInhalt").inner_html().lower())
    axe(pg, "Strukturansicht der fertigen Datei")
    pg.click("#strukturZurueck"); pg.wait_for_selector("section.ab-karte", timeout=15000)
    check("Zurück führt in die Prüfung", "ansicht=abschluss" in pg.url)
    # Alt-Text aendern -> Pruefdatei nicht mehr aktuell
    bilder = pg.request.get(B + f"/api/projects/{pid}").json().get("images") or []
    if bilder:
        pg.request.post(B + f"/api/images/{bilder[0]['id']}/alt-text", data={"alt_text": "Geänderter Alt-Text " + time.strftime("%H%M%S")})
        pg.goto(B + f"/app?projekt={pid}&ansicht=abschluss", wait_until="networkidle"); pg.wait_for_timeout(1200)
        check("Nach Alt-Text-Änderung: „Prüfdatei nicht mehr aktuell“, Neu-erstellen-Knopf ist Hauptknopf", "nicht mehr aktuell" in pg.locator("section.ab-karte h3").inner_text() and "btn-primary" in (pg.locator("button[id^=ab_erstellen_]").get_attribute("class") or ""), pg.locator("section.ab-karte h3").inner_text())
    doc_id = pg.request.get(B + f"/api/projects/{pid}/abschluss").json()["documents"][0]["id"]
    check("Seitenbild ausserhalb des Bereichs: 404", pg.request.get(B + f"/api/projects/{pid}/documents/{doc_id}/abschluss/seite/99").status == 404)
    check("Prüfung fremdes Projekt: 404", pg.request.get(B + "/api/projects/999999/abschluss").status == 404)
    check("Ansicht „abschluss“ lässt sich am Projekt merken", pg.request.post(B + f"/api/projects/{pid}/ansicht", data={"ansicht": "abschluss"}).ok)
    pg.request.post(B + f"/api/projects/{pid}/ansicht", data={"ansicht": "dokument"})

    print("== C3. Zweite PDF: Karten zum Aufklappen (Michael Karbe, PS 24.09.2026) ==")
    pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle"); pg.wait_for_timeout(1000)
    pg.set_input_files("#projUpload", {"name": "zweite_datei.pdf", "mimeType": "application/pdf", "buffer": testpdf()})
    for _ in range(40):
        pg.wait_for_timeout(1500)
        if pg.locator("section.dok-karte").count() == 2:
            break
    pg.wait_for_timeout(1500)
    check("Zwei Dokument-Karten", pg.locator("section.dok-karte").count() == 2)
    offen = pg.evaluate("() => Array.from(document.querySelectorAll('details.dok-klappe')).map(d => d.open)")
    check("Neue Karte aufgeklappt, erste zu", offen == [False, True], offen)
    check("Fokus auf dem Schalter der neuen Karte", pg.evaluate("!!(document.activeElement && document.activeElement.tagName === 'SUMMARY' && (document.activeElement.textContent || '').includes('zweite_datei.pdf'))"), pg.evaluate("document.activeElement && document.activeElement.textContent.slice(0,80)"))
    pg.click("details.dok-klappe >> nth=0 >> summary")
    pg.wait_for_timeout(300)
    check("Erste Karte per Klick aufgeklappt", pg.evaluate("document.querySelectorAll('details.dok-klappe')[0].open"))
    pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle"); pg.wait_for_timeout(1200)
    offen = pg.evaluate("() => Array.from(document.querySelectorAll('details.dok-klappe')).map(d => d.open)")
    check("Neu geladen mit zwei Dokumenten: beide Karten zu", offen == [False, False], offen)
    check("Zugeklappt: Überschriften bleiben lesbar (H3 im summary)", pg.locator("details.dok-klappe > summary h3").count() == 2)
    axe(pg, "Ansicht Dokument mit zwei zugeklappten Karten")
    pg.keyboard.press("Tab")
    r = pg.request.get(B + f"/api/projects/{pid}/dokument-ansicht")
    zweite = [d for d in (r.json().get("documents") or []) if (d.get("display_name") or d.get("original_filename") or "").startswith("zweite_datei")] if r.ok else []
    if zweite:
        rd = pg.request.delete(B + f"/api/projects/{pid}/documents/{zweite[0]['id']}")
        check("Zweite Datei wieder geloescht", rd.ok, rd.status)

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
