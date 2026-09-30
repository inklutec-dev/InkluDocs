#!/usr/bin/env python3
"""Klicktest: Word-Projekt mit denselben Ansichten wie ein PDF-Projekt (30.09.2026, Steve: „Word soll die gleiche Ansicht
wie PDF bekommen, auch mit der Dokumentenverwaltung“).

Prüft: Projektkopf (H1 mit Station, Word-Symbol, Ansichts-Knöpfe als Links mit aria-current), Startansicht „Dokument“,
Hochladen nur dort, Dokument-Karte (details/summary, H3, Dokumentinfos „Bezeichnung: Wert“ ohne KI, Knöpfe Hörprobe /
Herunterladen / Umbenennen / Löschen unter der Linie), Hörprobe-Dialog (Word-Datei mit Alt-Texten, lang am Inhalt, Fokus),
Herunterladen-Dialog im Modus 'word' (Word-Download, barrierefreie PDF), Umbenennen mit Fokus, Ansicht „Alt-Texte“ (kein
Upload, keine Dateiknöpfe, „Alt-Texte herunterladen“ nur Excel/JSON/CSV), Ansicht „Übersetzung“ (kein Upload, kein
Umbenennen/Löschen, „Übersetzung herunterladen“), Ansicht „Barrierefreiheitsprüfung“ (Prüfbericht, veraPDF-Ergebnis der
PDF mit „aktuell“/„nicht mehr aktuell“, Hörprobe), Browser-Zurück, gemerkte Ansicht, Löschen mit Fokus, axe ohne ernste
Verstöße in jeder Ansicht und in den Dialogen, keine Skriptfehler. Legt sein Projekt selbst an und löscht es (außer --behalten).

Aufruf: /home/claude/.venv-pw/bin/python ui_word_ansichten.py <testdokument_inkludocs.docx> [--behalten]
Braucht INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW (Testkonto auf Staging). Kostet Credits des Testkontos (Word-Download und
eine Umwandlung in barrierefreie PDF, wie verify_pdfua/ui_word); keine KI-Aufrufe.
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
if not MAIL or not PW:
    sys.exit("Zugangsdaten fehlen: INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW setzen")
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DOCX = ARGS[0] if ARGS else "/home/claude/testdokument_inkludocs.docx"
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


def axe(pg, name):
    pg.add_script_tag(url=AXE)
    pg.wait_for_timeout(500)
    r = pg.evaluate("async () => { const r = await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']}); return r.violations.map(v => ({id: v.id, impact: v.impact, n: v.nodes.length, html: v.nodes.slice(0,2).map(x => x.html.slice(0,120))})); }")
    ernst = [v for v in r if v["impact"] in ("serious", "critical")]
    check(f"axe {name}: 0 ernste Verstoesse", not ernst, ernst)
    if r:
        print("     axe-Hinweise (alle):", r)


def aktiv(pg):
    return pg.evaluate("document.activeElement ? (document.activeElement.tagName + '#' + (document.activeElement.id || '') + ' ' + (document.activeElement.textContent || '').slice(0, 60)) : ''")


def ansicht(pg, key, h1_anfang):
    pg.click(f".ansicht-knoepfe a[data-ansicht={key}]")
    pg.wait_for_function("(a) => (document.getElementById('projectName') || {}).textContent.startsWith(a)", arg=h1_anfang, timeout=20000)
    pg.wait_for_timeout(800)


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE", accept_downloads=True)
    pg = ctx.new_page()
    fehler_js = []
    pg.on("pageerror", lambda e: fehler_js.append(str(e)))
    pg.goto(B + "/login")
    pg.fill("#email", MAIL)
    pg.fill("#password", PW)
    pg.keyboard.press("Enter")
    pg.wait_for_timeout(2500)

    print("== A. Neues Word-Projekt startet in der Ansicht Dokument ==")
    r = pg.request.post(B + "/api/projects", data={"name": "Klicktest Word-Ansichten " + time.strftime("%H:%M"), "tool": "word"})
    check("Projekt angelegt", r.ok, r.status)
    pid = r.json().get("id") or r.json().get("project_id")
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
    pg.wait_for_timeout(1500)
    check("Startansicht Dokument (Adresse ansicht=dokument, H2 Dokumente)", "ansicht=dokument" in pg.url and pg.locator("#dokumenteHeading").count() == 1, pg.url)
    opts = [x.split("(")[0].strip() for x in pg.locator(".ansicht-knoepfe [data-ansicht]").all_inner_texts()]
    check("Ansichts-Knöpfe: Dokument, Alt-Texte, Übersetzung, Barrierefreiheitsprüfung", opts == ["Dokument", "Alt-Texte", "Übersetzung", "Barrierefreiheitsprüfung"], opts)
    check("Knöpfe sind Links; aktuelle Ansicht Dokument mit aria-current=page und dunkel",
          pg.locator(".ansicht-knoepfe a").count() == 4 and pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("data-ansicht") == "dokument"
          and "btn-primary" in (pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("class") or ""))
    check("„Ansicht:“ nur für Screenreader, Name der Knopfliste", "visually-hidden" in (pg.locator("#ansichtTitel").get_attribute("class") or "") and pg.locator("ul.ansicht-knoepfe[aria-labelledby=ansichtTitel]").count() == 1)
    ikon = pg.locator(".projekt-kopf img.projekt-dateityp")
    check("Projektkopf: blaues Word-Symbol mit Alt-Text „Word-Projekt“", ikon.count() == 1 and ikon.get_attribute("alt") == "Word-Projekt" and "icon-word" in (ikon.get_attribute("src") or ""))
    check("H1 und Seitentitel beginnen mit „Dokument – Projekt: “", pg.locator("h1#projectName").inner_text().startswith("Dokument – Projekt: ") and pg.title().startswith("Dokument – Projekt: "), (pg.locator("h1#projectName").inner_text(), pg.title()))
    check("Fokus nach dem Öffnen auf der H1", aktiv(pg).startswith("H1#projectName"), aktiv(pg))
    check("Keine Statusanzeige oben rechts, kein Feld „Funktionen und Einstellungen“", pg.locator("#projectStatusBadge").count() == 0 and pg.locator("section.projekt-funktionen").count() == 0)
    check("Upload-Feld „Word-Datei hinzufügen“ mit .docx", pg.locator("#addHeading").inner_text().strip() == "Word-Datei hinzufügen" and pg.locator("#projUpload").get_attribute("accept") == ".docx")
    hint = pg.locator("#projUploadHint").inner_text()
    check("Upload-Hinweis spricht vom Dokument (übersetzen, barrierefreie PDF), nicht von „Bilder extrahiert“", "barrierefreie PDF" in hint and "übersetzen" in hint and "Bilder extrahiert" not in hint, hint)
    check("H2 Dokumente (0) + Leerhinweis", "Dokumente (0)" in pg.locator("#dokumenteHeading").inner_text() and pg.locator("text=Noch kein Dokument hochgeladen.").count() == 1)
    axe(pg, "Ansicht Dokument, leeres Word-Projekt")
    pg.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="networkidle")
    pg.wait_for_timeout(1500)
    check("Leeres Projekt, Ansicht Alt-Texte: sagt, dass man in „Dokument“ hochlädt (kein Upload hier)", pg.locator("text=Noch kein Dokument hochgeladen. Das geht in der Ansicht „Dokument“.").count() == 1 and pg.locator("#projUpload").count() == 0)
    pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
    pg.wait_for_timeout(1500)

    print("== B. Hochladen und Dokument-Karte ==")
    pg.set_input_files("#projUpload", DOCX)
    pg.wait_for_selector("section.dok-karte", timeout=90000)
    pg.wait_for_timeout(2500)
    check("Eine Dokument-Karte", pg.locator("section.dok-karte").count() == 1)
    h3 = pg.locator("section.dok-karte h3").first.inner_text()
    name = os.path.basename(DOCX)
    check(f"H3 „Dokument 1: {name}“ im summary, ohne Stand-Abzeichen (Word kennt kein Tagging)", h3.strip() == f"Dokument 1: {name}" and pg.locator("section.dok-karte summary h3").count() == 1 and pg.locator("section.dok-karte h3 .badge").count() == 0, h3)
    check("Fokus nach dem Upload auf dem Schalter der Karte", pg.evaluate("!!(document.activeElement && document.activeElement.tagName === 'SUMMARY' && document.activeElement.querySelector('[id^=dok_heading_]'))"), aktiv(pg))
    check("Einzelnes Dokument: Karte aufgeklappt", pg.locator("section.dok-karte details.dok-klappe[open]").count() == 1)
    check("Upload-Feld heißt jetzt „Weitere Word-Datei hinzufügen“", pg.locator("#addHeading").inner_text().strip() == "Weitere Word-Datei hinzufügen")
    meta = pg.locator("section.dok-karte ul.dok-meta").first.inner_text()
    zeilen = [z.strip() for z in meta.split("\n") if z.strip()]
    check("Dokumentinfos je Zeile „Bezeichnung: Wert“", all(": " in z for z in zeilen) and len(zeilen) >= 6, zeilen)
    check("Titel aus der Datei, Anwendung, Sprache, Überschriften, Tabellen, Bilder", meta.startswith("Titel: Testdokument") and all(k in meta for k in ("Anwendung: ", "Sprache: en-US", "Überschriften: ", "Tabellen: 1", "Bilder: ")), meta)
    check("Seiten nur, wenn belegbar (python-docx-Datei: keine Zeile „Seiten“)", "Seiten:" not in meta, meta)
    check("Kein Vorschaubild, kein Tagging-Knopf", pg.locator("section.dok-karte img.ausgabe-vorschau").count() == 0 and pg.locator("button[id^=dok_tag_]").count() == 0)
    kn = [x.split("\n")[0].strip() for x in pg.locator("section.dok-karte .dok-werkbank .ausgabe-aktionen button").all_inner_texts()]
    check("Knöpfe unter der Linie: Hörprobe, Herunterladen, Umbenennen, Löschen", kn == ["Hörprobe", "Herunterladen", "Umbenennen", "Löschen"], kn)
    check("Linie über der Knopfleiste, darunter nichts weiter", pg.evaluate("getComputedStyle(document.querySelector('.dok-werkbank')).borderTopStyle") == "solid" and pg.evaluate("document.querySelector('.dok-werkbank').children.length") == 1)
    check("Knopfnamen eindeutig für Screenreader (Dokumentname im Namen)", name in pg.locator("button[id^=dok_export_]").first.inner_text() and "barrierefreie PDF" in pg.locator("button[id^=dok_export_]").first.inner_text())
    axe(pg, "Ansicht Dokument mit Word-Karte")

    print("== C. Hörprobe (Word-Datei mit Alt-Texten, ohne KI) ==")
    pg.click("button[id^=dok_hp_]")
    pg.wait_for_selector("#dkHoerprobeDialog[open]", timeout=10000)
    inhalt = ""
    for _ in range(30):
        pg.wait_for_timeout(500)
        inhalt = pg.locator("#dkHpInhalt").inner_text()
        if inhalt and "wird geladen" not in inhalt:
            break
    check("Dialog „Hörprobe: <Name>“ mit Hinweis für Word (keine Tags)", pg.locator("#dkHpHeading").inner_text().startswith("Hörprobe: ") and "Word-Dokument" in pg.locator("#dkHpHinweis").inner_text() and "Tags" not in pg.locator("#dkHpHinweis").inner_text(), pg.locator("#dkHpHinweis").inner_text())
    check("Hörprobe liest die Word-Datei: Dokumenttitel, Überschrift, Bild", "Dokumenttitel: Testdokument" in inhalt and "Überschrift Ebene" in inhalt and ("Bild" in inhalt or "Schmuckbild" in inhalt), inhalt[:300])
    check("Inhalt mit lang der Dokumentsprache", pg.locator("#dkHpInhalt span[lang='en-US']").count() > 3)
    check("Statuszeile „Hörprobe geladen, n Zeilen.“ und Fokus im Vorlesetext", "Hörprobe geladen" in pg.locator("#dkHpStatus").inner_text() and aktiv(pg).startswith("DIV#dkHpInhalt"), (pg.locator("#dkHpStatus").inner_text(), aktiv(pg)))
    axe(pg, "Hörprobe-Dialog Word")
    pg.click("#dkHpZu")
    pg.wait_for_timeout(300)
    check("Schließen: Fokus zurück auf „Hörprobe“", aktiv(pg).startswith("BUTTON#dok_hp_"), aktiv(pg))

    print("== D. Umbenennen ==")
    pg.click("section.dok-karte button:has-text('Umbenennen')")
    pg.wait_for_selector("#docRenameDialog[open]", timeout=5000)
    check("Umbenennen-Dialog, Fokus im Namensfeld (vorbelegt)", aktiv(pg).startswith("INPUT#docRenameInput") and pg.input_value("#docRenameInput") == name, aktiv(pg))
    pg.fill("#docRenameInput", "Klicktest umbenannt")
    pg.keyboard.press("Enter")
    pg.wait_for_function("() => (document.querySelector('section.dok-karte h3') || {}).textContent.includes('Klicktest umbenannt')", timeout=15000)
    pg.wait_for_timeout(600)
    check("Neuer Name in der Karte, Fokus auf ihrem Schalter", pg.evaluate("!!(document.activeElement && document.activeElement.tagName === 'SUMMARY' && document.activeElement.textContent.includes('Klicktest umbenannt'))"), aktiv(pg))

    print("== E. Herunterladen (Modus word) ==")
    pg.click("button[id^=dok_export_]")
    pg.wait_for_selector("#exportPanel[open]", timeout=10000)
    zs = ""
    for _ in range(20):
        zs = pg.locator("#exportSummary").inner_text()
        if "Credits" in zs:
            break
        pg.wait_for_timeout(500)
    sichtbar = [b.inner_text().strip() for b in pg.locator("#exportPanel button").all() if b.is_visible()]
    check("Überschrift „Dokument herunterladen“", pg.locator("#exportPanelHeading").inner_text().strip() == "Dokument herunterladen", pg.locator("#exportPanelHeading").inner_text())
    check("Sichtbar: Als Word, In barrierefreie PDF umwandeln, Abbrechen — keine Excel/JSON/CSV, kein „Hörprobe und Prüfbericht“",
          "Als Word" in sichtbar and any("barrierefreie PDF" in s for s in sichtbar) and not any(s in sichtbar for s in ("Als Excel", "Als JSON", "Als CSV")) and not any("Prüfbericht" in s for s in sichtbar), sichtbar)
    check("Zusammenfassung mit Preis, ohne CSV-Satz", "Credits" in zs and "CSV" not in zs, zs)
    axe(pg, "Herunterladen-Dialog Word")
    with pg.expect_download(timeout=90000) as dl:
        pg.click("#exportDocxBtn")
    d = dl.value
    check("Word-Datei kommt an (.docx)", d.suggested_filename.endswith(".docx"), d.suggested_filename)
    pg.wait_for_timeout(1200)
    st = pg.locator("#exportStatus").inner_text()
    check("Meldung „Heruntergeladen …“ mit Fokus, Abbrechen heißt „Zurück zum Projekt“", st.startswith("Heruntergeladen:") and pg.evaluate("document.activeElement && document.activeElement.id") == "exportStatus" and pg.locator("#exportCancelBtn").inner_text().strip() == "Zurück zum Projekt", st)
    # Barrierefreie PDF (vorhandener Weg): Grundlage für das veraPDF-Ergebnis in der Barrierefreiheitsprüfung
    pg.click("#pdfuaBtn")
    erg = ""
    for _ in range(90):
        pg.wait_for_timeout(1000)
        erg = pg.locator("#pdfuaResult").inner_text()
        if pg.locator("#pdfuaDownload").count() or "fehlgeschlagen" in erg or "nicht eingerichtet" in erg:
            break
    check("Umwandlung in barrierefreie PDF liefert Ergebnis + „PDF herunterladen“ mit Fokus", pg.locator("#pdfuaDownload").count() == 1 and aktiv(pg).startswith("BUTTON#pdfuaDownload"), erg[:200])
    pg.click("#exportCancelBtn")
    pg.wait_for_timeout(400)
    check("Dialog geschlossen", pg.evaluate("!document.getElementById('exportPanel').open"))

    print("== F. Ansicht Alt-Texte ==")
    ansicht(pg, "alttexte", "Alt-Texte")
    check("Adresse ansicht=alttexte, aktueller Knopf Alt-Texte", "ansicht=alttexte" in pg.url and pg.locator(".ansicht-knoepfe a[aria-current=page]").get_attribute("data-ansicht") == "alttexte")
    check("Fokus nach dem Wechsel auf der H1", aktiv(pg).startswith("H1#projectName"), aktiv(pg))
    check("Kein Upload-Feld (Hochladen nur in „Dokument“)", pg.locator("#projUpload").count() == 0)
    check("Dokument-Block mit H2 und Bildern wie bisher", pg.locator("h2.doc-heading").count() == 1 and "Klicktest umbenannt" in pg.locator("h2.doc-heading").first.inner_text())
    da_kn = [x.split("\n")[0].strip() for x in pg.locator(".doc-actions button").all_inner_texts()]
    check("Je Dokument: Alt-Texte generieren, Alt-Texte herunterladen — kein Umbenennen, Löschen, Datei-Download", da_kn == ["Alt-Texte generieren", "Alt-Texte herunterladen"], da_kn)
    check("Oben „Alt-Texte herunterladen“ statt „Herunterladen“", pg.locator("#exportOpenBtn").inner_text().split("\n")[0].strip().startswith("Alt-Texte herunterladen"), pg.locator("#exportOpenBtn").inner_text())
    pg.click("#exportOpenBtn")
    pg.wait_for_selector("#exportPanel[open]", timeout=10000)
    pg.wait_for_timeout(1500)
    sichtbar = [b.inner_text().strip() for b in pg.locator("#exportPanel button").all() if b.is_visible()]
    check("Dialog „Alt-Texte herunterladen“: nur Excel, JSON, CSV (keine Word-Datei, keine PDF, keine Übersetzung)",
          pg.locator("#exportPanelHeading").inner_text().strip() == "Alt-Texte herunterladen" and all(s in sichtbar for s in ("Als Excel", "Als JSON", "Als CSV"))
          and not any(s.startswith("Als Word") or "barrierefreie PDF" in s for s in sichtbar), sichtbar)
    axe(pg, "Alt-Texte-Dialog Word")
    pg.click("#exportCancelBtn")
    pg.wait_for_timeout(300)
    axe(pg, "Ansicht Alt-Texte Word")

    print("== G. Ansicht Übersetzung ==")
    ansicht(pg, "uebersetzung", "Übersetzung")
    pg.wait_for_timeout(2500)
    check("Kein Upload-Feld", pg.locator("#projUpload").count() == 0)
    ue_kn = [x.split("\n")[0].strip() for x in pg.locator(".doc-actions button").all_inner_texts()]
    check("Kein Umbenennen/Löschen in der Übersetzung", not any(k in ("Umbenennen", "Löschen") for k in ue_kn), ue_kn)
    check("Dokument-Block der Übersetzung da", pg.locator("h2.doc-heading").count() == 1, pg.locator("main").inner_text()[:200])
    axe(pg, "Ansicht Übersetzung Word")

    print("== H. Ansicht Barrierefreiheitsprüfung ==")
    ansicht(pg, "abschluss", "Barrierefreiheitsprüfung")
    pg.wait_for_selector("section.ab-karte", timeout=30000)
    pg.wait_for_timeout(800)
    h3p = pg.locator("section.ab-karte h3").first.inner_text()
    check("Karte mit H3 „Dokument 1: …“ und Abzeichen (Befunde)", h3p.startswith("Dokument 1: Klicktest umbenannt") and pg.locator("section.ab-karte h3 .badge").count() == 1 and "Befund" in pg.locator("section.ab-karte h3 .badge").inner_text(), h3p)
    pm = pg.locator("section.ab-karte ul.dok-meta").first.inner_text()
    check("Infos: Prüfbericht, Barrierefreie PDF erstellt am, Stand aktuell, Norm-Prüfung veraPDF", all(k in pm for k in ("Prüfbericht des Word-Dokuments: ", "Barrierefreie PDF: erstellt am", "Stand: aktuell", "Norm-Prüfung PDF/UA-1 (veraPDF): ")), pm)
    h4 = [x.strip() for x in pg.locator("section.ab-karte h4").all_inner_texts()]
    check("Abschnitte H4: Prüfbericht, Norm-Prüfung der barrierefreien PDF, Hörprobe", h4 == ["Prüfbericht des Word-Dokuments", "Norm-Prüfung der barrierefreien PDF (veraPDF)", "Hörprobe"], h4)
    txt = pg.locator("section.ab-karte").first.inner_text()
    check("veraPDF-Teil: Befunde mit Regelnummer oder „Keine Problemstellen“ + Hinweis", ("(veraPDF-Regel" in txt) or ("Keine Problemstellen gefunden" in txt and "Wichtig: veraPDF prüft" in txt), txt[:600])
    check("Keine KI: kein KI-Knopf, Hinweis „ohne KI“", pg.locator("section.ab-karte button:has-text('KI')").count() == 0 and "ohne KI" in txt)
    check("Kein Herunterladen in der Prüfung", pg.locator("section.ab-karte button:has-text('Herunterladen')").count() == 0)
    vk = pg.locator("button[id^=ab_wvorlesen_][aria-pressed=false]")
    check("„Hörprobe vorlesen“ (aria-pressed=false) mit Dokumentname im Namen, Klappe „Hörprobe lesen – Dokument …“",
          vk.count() == 1 and "ab_wname_" in (vk.get_attribute("aria-labelledby") or "") and pg.locator("details.ab-whp > summary").inner_text().strip().startswith("Hörprobe lesen")
          and "Klicktest umbenannt" in pg.locator("details.ab-whp > summary").inner_text())
    check("Region der Hörprobe heißt „Hörprobe von „…““", (pg.locator("details.ab-whp [role=region]").get_attribute("aria-label") or "").startswith("Hörprobe von „Klicktest umbenannt"))
    pg.click("details.ab-whp > summary")
    pg.wait_for_timeout(300)
    check("Hörprobe in der Prüfung: Zeilen mit lang am Inhalt", "Dokumenttitel" in pg.locator("details.ab-whp").inner_text() and pg.locator("details.ab-whp span[lang='en-US']").count() > 3)
    axe(pg, "Ansicht Barrierefreiheitsprüfung Word")
    # Alt-Text ändern -> die PDF passt nicht mehr zum Stand
    bild = pg.request.get(B + f"/api/projects/{pid}").json().get("images", [])
    if bild:
        pg.request.post(B + f"/api/images/{bild[0]['id']}/alt-text", data={"alt_text": "Geänderter Alt-Text aus dem Klicktest " + time.strftime("%H:%M:%S")})
    pg.reload(wait_until="networkidle")
    pg.wait_for_selector("section.ab-karte", timeout=30000)
    pg.wait_for_timeout(800)
    pm2 = pg.locator("section.ab-karte ul.dok-meta").first.inner_text()
    check("Nach einer Alt-Text-Änderung: „Stand: nicht mehr aktuell …“ und Hinweis zum Neuerstellen", "Stand: nicht mehr aktuell" in pm2 and "nicht mehr aktuell." in pg.locator("section.ab-karte").first.inner_text(), pm2)

    print("== I. Zurück, gemerkte Ansicht ==")
    pg.go_back()
    pg.wait_for_function("() => (document.getElementById('projectName') || {}).textContent.startsWith('Übersetzung')", timeout=20000)
    check("Browser-Zurück führt zur vorherigen Ansicht (Übersetzung)", "ansicht=uebersetzung" in pg.url)
    pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
    pg.wait_for_timeout(1500)
    check("Ohne ?ansicht öffnet die zuletzt gewählte Ansicht (Barrierefreiheitsprüfung)", pg.locator("h1#projectName").inner_text().startswith("Barrierefreiheitsprüfung"), pg.locator("h1#projectName").inner_text())

    print("== J. Löschen ==")
    ansicht(pg, "dokument", "Dokument")
    pg.wait_for_selector("section.dok-karte", timeout=15000)
    pg.click("section.dok-karte button:has-text('Löschen')")
    pg.wait_for_selector("#docDeleteDialog[open]", timeout=5000)
    check("Lösch-Dialog nennt Dokument und dass Alt-Texte und Übersetzungen verloren gehen, Fokus auf Abbrechen",
          "Klicktest umbenannt" in pg.locator("#docDeleteBody").inner_text() and "Alt-Texte und Übersetzungen" in pg.locator("#docDeleteBody").inner_text()
          and "0 Bildern" not in pg.locator("#docDeleteBody").inner_text() and aktiv(pg).startswith("BUTTON#docDeleteCancel"), (pg.locator("#docDeleteBody").inner_text(), aktiv(pg)))
    axe(pg, "Lösch-Dialog Word")
    pg.click("#docDeleteConfirm")
    pg.wait_for_function("() => (document.getElementById('dokumenteHeading') || {}).textContent === 'Dokumente (0)'", timeout=15000)
    pg.wait_for_timeout(500)
    check("Karte weg, „Dokumente (0)“, Fokus auf der Überschrift", pg.locator("section.dok-karte").count() == 0 and aktiv(pg).startswith("H2#dokumenteHeading"), aktiv(pg))

    check("Keine Skriptfehler", not fehler_js, fehler_js[:3])
    if not BEHALTEN:
        rr = pg.request.delete(B + f"/api/projects/{pid}")
        print("  Testprojekt gelöscht:", rr.status)
    else:
        print("  Projekt bleibt:", pid)
    br.close()
print(f"\nErgebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
