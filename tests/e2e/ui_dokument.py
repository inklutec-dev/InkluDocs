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
    check("Ansichts-Knöpfe: Dokument, Alt-Texte, Quickinfos (ausgegraut, keine Felder), Prüfung", [o.split("(")[0].strip() for o in opts] == ["Dokument", "Alt-Texte", "Quickinfos", "Prüfung"] and pg.locator(".ansicht-knoepfe .ansicht-aus[data-ansicht=quickinfos]").count() == 1 and pg.locator(".ansicht-knoepfe a[data-ansicht=quickinfos]").count() == 0, opts)
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
    pg.set_input_files("#projUpload", {"name": "klicktest_roh.pdf", "mimeType": "application/pdf", "buffer": testpdf()})
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
    check("Dokumentinfo je Zeile mit Doppelpunkt (Punkt 2): Stand: …, Seiten: 2, Sprache, Struktur, Bilder", all(k in dl for k in ("Stand: ", "Seiten: 2", "Sprache: ", "Struktur: ", "Bilder: ")), dl)
    fs = pg.evaluate("() => [getComputedStyle(document.querySelector('ul.dok-meta')).fontSize, getComputedStyle(document.querySelector('ul.dok-meta')).fontFamily]")
    check("Dokumentinfo in Schrift und Groesse des Berichts (Punkt 4)", fs[0] == pg.evaluate("() => { const d = document.createElement('div'); d.className = 'page-text-content'; document.body.appendChild(d); const f = getComputedStyle(d).fontSize; d.remove(); return f; }"), fs)
    check("Dokumentansicht ohne Feld „Funktionen und Einstellungen“ (Feedback 24.09., Punkt 1)", pg.locator("section.projekt-funktionen").count() == 0)
    check("Vorschaubild mit Alt-Text", pg.locator("section.dok-karte img.ausgabe-vorschau").first.get_attribute("alt").startswith("Vorschau der ersten Seite"))
    check("Knopf „Barrierefrei machen“ mit Seiten und Credits im Namen", pg.locator("button[id^=dok_tag_]").count() == 1 and "2 Seiten, 2 Credits" in pg.locator("button[id^=dok_tag_]").first.inner_text())
    check("Kein Knopf „Alt-Texte bearbeiten“ mehr (Punkt 3); Umbenennen / Löschen da", pg.locator("section.dok-karte button:has-text('Alt-Texte bearbeiten')").count() == 0 and pg.locator("section.dok-karte button:has-text('Umbenennen')").count() == 1 and pg.locator("section.dok-karte button:has-text('Löschen')").count() == 1)
    check("Noch kein Knopf „PDF herunterladen“ (ungetaggt)", pg.locator("button[id^=dok_export_]").count() == 0)
    check("Knöpfe unter einer Linie über die volle Breite, nicht neben dem Vorschaubild (Mail - 3, Punkt 3)",
          pg.locator("section.dok-karte .dok-werkbank .ausgabe-aktionen button[id^=dok_tag_]").count() == 1
          and pg.locator("section.dok-karte .ausgabe-text button").count() == 0
          and pg.evaluate("getComputedStyle(document.querySelector('.dok-werkbank')).borderTopStyle") == "solid")
    axe(pg, "Ansicht Dokument vor dem Lauf")

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
    axe(pg, "Ansicht Dokument mit Ergebnis des Testlaufs")

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
        if pg.locator("section.dok-karte .dok-ergebnis").count():
            fertig = True
            break
    meld = pg.locator("section.dok-karte .dok-ergebnis p").first.inner_text() if fertig else ""
    # Mail - 2, Punkt 2: Ergebnis UNTER dem Dokument in der Karte, farbig, mit Fokus
    check("Ergebnis in der Karte, nennt Struktur", fertig and "getaggt" in meld and "Elemente" in meld, meld)
    check("Fokus auf dem Ergebnis in der Karte", str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_ergebnis_text_"))
    check("Ergebnis grün hinterlegt (Erfolg)", pg.evaluate("getComputedStyle(document.querySelector('.dok-ergebnis')).backgroundColor") == "rgb(240, 253, 244)")
    check("Keine Meldung mehr oben über der Liste", pg.locator("#dkLaufMeldung:not([hidden])").count() == 0)
    h3 = pg.locator("section.dok-karte h3").first.inner_text()
    check("Badge jetzt „Getaggt …“", "Getaggt" in h3, h3)
    check("Knopf heisst jetzt „Neu taggen“", pg.locator("button[id^=dok_tag_]").first.inner_text().startswith("Neu taggen"))
    # Herunterladen wieder in „Dokument“ (Mail - 3, Punkt 4); Hinweis auf die Station „Prüfung“
    check("Knopf „PDF herunterladen“ in der Dokument-Karte, Hinweis auf „Prüfung“", pg.locator("button[id^=dok_export_]").count() == 1 and "Ansicht „Prüfung“" in pg.locator("section.dok-karte").first.inner_text())
    pg.click("details.dok-bericht > summary")
    pg.wait_for_timeout(300)
    ber = pg.locator("details.dok-bericht").first.inner_text()
    check("Bericht: Zeit + PDF/UA-Prüfung, ohne Dokumentinfos und ohne „Hinweis“ (Punkte 9, 10)", "Getaggt am" in ber and "PDF/UA" in ber and "Dokumentsprache" not in ber and "Struktur:" not in ber and "Hinweis –" not in ber, ber[:300])
    mt = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    check("Dokumentinfos: Titel, Anwendung, Erstellt mit, Stand: Getaggt, PDF-Standard (Punkte 2-5)", all(k in mt for k in ("Titel: ", "Anwendung: ", "Erstellt mit: ", "Stand: Getaggt", "PDF-Standard: ")) and mt.index("Titel:") < mt.index("Anwendung:") < mt.index("Erstellt mit:") < mt.index("Stand:"), mt)
    check("PDF-Standard nach dem Tagging: PDF/UA-1", "PDF-Standard: PDF/UA-1" in mt, mt)
    # Punkt 7 (Feedback 24.09.) meint das echte Tagging: kein Testmodus-Hinweis in Dokumentinfos und Bericht. Der Knopf
    # „Testweise taggen“ und sein Ergebnis nennen den Testmodus bewusst (Mail - 2, Punkt 3).
    check("Kein Testmodus-Hinweis in Dokumentinfos und Bericht (Punkt 7)", "Testmodus" not in pg.locator("section.dok-karte ul.dok-meta").first.inner_text() and "Testmodus" not in pg.locator("section.dok-karte details.dok-bericht").first.inner_text())
    check("Bericht ohne „In Ordnung“-Zeilen (Punkt 6)", "In Ordnung –" not in ber, ber[-400:])
    dl = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    check("Beschreibungsliste nach dem Lauf: Sprache de-DE, 1 Bild", "de-DE" in dl and "1 Bilder" in dl, dl)
    axe(pg, "Ansicht Dokument nach dem Lauf (mit Ergebnis)")
    pg.click("section.dok-karte .dok-ergebnis button")
    pg.wait_for_timeout(300)
    check("„Meldung schließen“: weg, Fokus auf dem Schalter der Karte", pg.locator(".dok-ergebnis").count() == 0 and pg.evaluate("document.activeElement && document.activeElement.tagName") == "SUMMARY")
    with pg.expect_download(timeout=90000) as dl_info:
        pg.click("button[id^=dok_export_]")
    pfad = dl_info.value.path()
    check("PDF aus der Ansicht Dokument heruntergeladen (%PDF)", pfad is not None and open(pfad, "rb").read(5) == b"%PDF-", dl_info.value.suggested_filename)
    pg.wait_for_timeout(2500)
    st = pg.locator("output[id^=dok_status_]").first.inner_text()
    check("Statuszeile nennt Download und Ablage, Fokus darauf", "Heruntergeladen" in st and "Ablage" in st and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_status_"), st)
    r = pg.request.get(B + f"/api/ausgaben?projekt={pid}")
    eintraege = r.json().get("ausgaben", []) if r.ok else []
    check("Ablage-Eintrag art pdf mit Datei", len(eintraege) == 1 and eintraege[0].get("art") == "pdf" and eintraege[0].get("datei_verfuegbar") is True, eintraege[:1])

    print("== B2. Hoerprobe und Strukturansicht (22.09.) ==")
    check("Strukturansicht-Link in der Dokument-Karte ausgeblendet (Steve 24.09.; gibt es in der Abschlussprüfung)", pg.locator("section.dok-karte a[id^=dok_struktur_]").count() == 0)
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

    print("== C2b. Station „Prüfung“ (24.09.2026, umgebaut 25.09. nach Michaels Mail - 3) ==")
    opts = [x.strip() for x in pg.locator(".ansicht-knoepfe [data-ansicht]").all_inner_texts()]
    check("Ansichts-Knöpfe nennen „Prüfung“ als letzte Station (Punkt 5)", opts[-1] == "Prüfung", opts)
    pg.click(".ansicht-knoepfe a[data-ansicht=abschluss]")
    pg.wait_for_selector("section.ab-karte", timeout=15000); pg.wait_for_timeout(800)
    check("H1 nennt die Station „Prüfung“", pg.locator("h1#projectName").inner_text().startswith("Prüfung – Projekt: "), pg.locator("h1#projectName").inner_text())
    check("Adresse ansicht=abschluss, Kopf mit PDF-Symbol, eine Karte offen", "ansicht=abschluss" in pg.url and pg.locator(".projekt-kopf img.projekt-dateityp").count() == 1 and pg.locator("details.ab-klappe[open]").count() == 1)
    check("Karte: „Noch keine Prüfdatei“, Knopf „Prüfdatei erstellen“ (kostenlos)", "Noch keine Prüfdatei" in pg.locator("section.ab-karte h3").inner_text() and pg.locator("button[id^=ab_erstellen_]").count() == 1 and "kostenlos" in pg.locator("button[id^=ab_erstellen_]").inner_text())
    check("Kein Herunterladen in der Prüfung (Punkt 6), kein Upload-Feld", pg.locator("button[id^=ab_export_]").count() == 0 and pg.locator("#abAlleBtn").count() == 0 and "PDF herunterladen" not in pg.locator("main").inner_text() and pg.locator("#projUpload").count() == 0)
    check("KI-basierte Prüfung als Abschnitt mit Knopf (Punkt 13)", pg.locator("section.ab-ki h4").count() == 1 and "KI-basierte Prüfung (experimentell)" in pg.locator("section.ab-ki h4").inner_text() and pg.locator("section.ab-ki button[id^=dok_pruef_]").count() == 1 and "2 Seiten, 4 Credits" in pg.locator("section.ab-ki button[id^=dok_pruef_]").inner_text())
    axe(pg, "Prüfung vor der Prüfdatei")
    pg.click("button[id^=ab_erstellen_]")
    for _ in range(60):
        pg.wait_for_timeout(1500)
        if "Prüfdatei erstellt" in (pg.locator("output[id^=ab_status_]").first.inner_text() if pg.locator("output[id^=ab_status_]").count() else ""):
            break
    st = pg.locator("output[id^=ab_status_]").first.inner_text()
    check("Prüfdatei erstellt, Statuszeile mit Fokus", "Prüfdatei erstellt" in st and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("ab_status_"), st)
    meta = pg.locator("section.ab-karte ul.dok-meta").first.inner_text()
    check("Stand: Prüfdatei erstellt am …, aktuell, PDF/UA-Prüfung, Problemstellen", all(k in meta for k in ("Prüfdatei: erstellt am", "Stand: aktuell", "PDF/UA-Prüfung: ", "Problemstellen: ")), meta)
    pg.wait_for_timeout(1500)
    kopf = pg.locator("ul[id^=ab_kopf_]").first.inner_text()
    check("Sprache und Zusammenfassung oben bei den Infos (Punkt 10)", "Sprache" in kopf and "Zusammenfassung" in kopf, kopf)
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

    print("== C2c. KI-basierte Pruefung in der Station „Prüfung“ ==")
    kn = pg.locator("section.ab-ki button[id^=dok_pruef_]")
    kn.first.click()
    pg.wait_for_timeout(1500)
    st = pg.locator("output[id^=dok_pruef_status_]").first.inner_text()
    check("Statuszeile „Prüfung läuft“ und Fokus darauf", "Prüfung läuft" in st and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_pruef_status_"), st)
    fertig = False
    for _ in range(90):
        pg.wait_for_timeout(2000)
        if "fertig" in (pg.locator("output[id^=dok_pruef_status_]").first.inner_text() if pg.locator("output[id^=dok_pruef_status_]").count() else ""):
            fertig = True
            break
    st = pg.locator("output[id^=dok_pruef_status_]").first.inner_text() if fertig else ""
    check("Statuszeile „Prüfung … fertig“ mit Fokus", fertig and str(pg.evaluate("document.activeElement && document.activeElement.id")).startswith("dok_pruef_status_"), st)
    check("Kurzfassung statt zweiter Befundliste, Knopf „KI-Prüfung erneut starten“", "Letzte Prüfung am" in pg.locator("section.ab-ki").inner_text() and pg.locator("section.ab-ki ol.dok-befunde").count() == 0 and pg.locator("section.ab-ki button[id^=dok_pruef_]").first.inner_text().startswith("KI-Prüfung erneut starten"), pg.locator("section.ab-ki").inner_text()[:300])
    axe(pg, "Prüfung nach der KI-Prüfung")
    if pg.locator("button[id^=dok_korr_]").count():
        print("== C2d. Korrektur mit Doppelbeleg (in der Prüfung) ==")
        pg.locator("button[id^=dok_korr_]").first.click()
        fertig = False
        for _ in range(30):
            pg.wait_for_timeout(2000)
            if "Korrektur" in (pg.locator("output[id^=dok_pruef_status_]").first.inner_text() if pg.locator("output[id^=dok_pruef_status_]").count() else "") and pg.locator("button[id^=dok_korr_undo_]").count():
                fertig = True
                break
        check("Korrektur fertig, Rückgängig-Knopf da", fertig, pg.locator("section.ab-ki").inner_text()[-300:])
        if fertig:
            pg.click("button[id^=dok_korr_undo_]"); pg.wait_for_timeout(2500)
            check("Rückgängig: Meldung in der Statuszeile", "rückgängig" in pg.locator("output[id^=dok_pruef_status_]").first.inner_text().lower())
    else:
        print("   (keine Befunde mit Doppelbeleg in diesem Lauf — Korrektur-Teil uebersprungen)")
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
