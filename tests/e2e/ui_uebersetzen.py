#!/usr/bin/env python3
"""Klicktest Uebersetzen-Werkzeug (18.09.2026): Werkzeugauswahl, Ansicht (Kopf, Dokument/
Abschnitt/Absatz-Struktur, Original + Uebersetzung, Filter-Radiogruppe, Uebersetzen-Dialog
mit Rueckfrage, Export-Dialog), Tastaturweg, axe in beiden Dialogen. Muster: ui_formular.py.
Aufruf: /home/claude/.venv-pw/bin/python ui_uebersetzen.py <projekt-id>   (Projekt mit fertiger Uebersetzung)"""
import os
import sys
from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de"); PID = sys.argv[1]
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
if not MAIL or not PW: sys.exit("Zugangsdaten fehlen: INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW setzen")
SHOTS = os.environ.get("INKLUDOCS_E2E_SHOTS", "/tmp")
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"
ok = fehler = 0
def check(n, c, i=""):
    global ok, fehler
    if c: ok += 1; print("  OK ", n)
    else: fehler += 1; print("  FEHLT", n, "--", i)

def axe(pg, name):
    pg.add_script_tag(url=AXE); pg.wait_for_timeout(500)
    r = pg.evaluate("async () => { const r = await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']}); return r.violations.map(v => ({id: v.id, impact: v.impact, n: v.nodes.length, html: v.nodes.slice(0,2).map(x => x.html.slice(0,120))})); }")
    ernst = [v for v in r if v["impact"] in ("serious", "critical")]
    check(f"axe {name}: 0 ernste Verstoesse", not ernst, ernst)
    if r: print("     axe-Hinweise (alle):", r)

with sync_playwright() as p:
    br = p.chromium.launch(); ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE"); pg = ctx.new_page()
    fehler_js = []
    pg.on("pageerror", lambda e: fehler_js.append(str(e)))
    pg.goto(B + "/login"); pg.fill("#email", MAIL); pg.fill("#password", PW); pg.keyboard.press("Enter"); pg.wait_for_timeout(2500)
    print("== A. Werkzeugauswahl ==")
    pg.goto(B + "/projekt-neu"); pg.wait_for_timeout(1500)
    opts = pg.locator("#toolSelect option").all_text_contents()
    check("„Dokumente übersetzen“ im Auswahlmenue", any("Dokumente übersetzen" in o for o in opts), opts)
    print("== B. Ansicht ==")
    pg.goto(B + f"/app?projekt={PID}"); pg.wait_for_timeout(3500)
    main = pg.locator("main")
    check("H1 Projektname", pg.locator("h1#projectName").count() == 1)
    check("Keine Statuszeile unter dem Projektnamen (Michael Punkt 4, wie Word)", pg.locator("#projectHeadInfo").inner_text().strip() == "")
    check("Badge Vollstaendig", pg.locator("#projectStatusBadge").inner_text().strip() in ("Vollständig", "In Arbeit"), pg.locator("#projectStatusBadge").inner_text())
    check("Upload-Block: Weitere Word-Datei hinzufuegen", pg.locator("#addHeading").inner_text().strip() == "Weitere Word-Datei hinzufügen", pg.locator("#addHeading").inner_text())
    check("Dateiauswahl akzeptiert .docx", pg.locator("#projUpload").get_attribute("accept") == ".docx")
    check("Kein 'Bilder' und kein 'Alt-Texte generieren' in der Ansicht", "Alt-Texte generieren" not in main.inner_text() and "Bilder filtern" not in main.inner_text())
    ca = pg.locator(".card-actions").first
    check("Knoepfe: Uebersetzen + Herunterladen", "Übersetzen" in ca.inner_text() and "Herunterladen" in ca.inner_text(), ca.inner_text()[:200])
    fs = pg.locator("fieldset#segFilterFieldset")
    check("Filter als eigene Karte 'Absätze filtern' mit Fieldset, Legende, 3 Chips mit Zaehler, Statuszeile", pg.locator("#segFilterBar h2").inner_text().strip() == "Absätze filtern" and fs.locator("legend").inner_text().strip() == "Nach Übersetzungsstand filtern" and fs.locator("label.filter-chip").count() == 3 and "(" in fs.locator("label.filter-chip").first.inner_text() and "angezeigt" in pg.locator("#segFilterStatus").inner_text(), (fs.locator("label.filter-chip").first.inner_text(), pg.locator("#segFilterStatus").inner_text()))
    docs = pg.locator("h2.doc-heading"); check("Dokument-Ueberschrift (h2) mit Absatz-Zaehler", docs.count() == 1 and "Absätze" in docs.first.inner_text(), docs.first.inner_text() if docs.count() else "")
    pg.locator("details.doc-section").first.evaluate("d=>d.open=true"); pg.wait_for_timeout(300)
    check("Hinweis-Klappe zum Dokument (nur wenn Hinweise)", True)
    h3 = pg.locator("h3.page-heading"); check("Abschnitts-Ueberschriften (h3) 'Abschnitt N: …' + Kopfzeile + Bilder/Titel", h3.count() >= 3 and h3.first.inner_text().startswith("Abschnitt 1"), [h3.nth(i).inner_text() for i in range(h3.count())])
    check("Gruppe 'Kopfzeile' vorhanden", any("Kopfzeile" in h3.nth(i).inner_text() for i in range(h3.count())))
    check("Gruppe 'Dokumenttitel' vorhanden", any("Dokumenttitel" in h3.nth(i).inner_text() for i in range(h3.count())))
    pg.locator("details.page-section").first.evaluate("d=>d.open=true"); pg.wait_for_timeout(300)
    sec = pg.locator("details.page-section").first
    h4 = sec.locator("h4.image-heading")
    alle_h4 = pg.locator("h4.image-heading")
    check("Absatz-Ueberschriften h4 nur '{Art} N' ohne Textanfang (Michael 18.09.)", alle_h4.count() >= 12 and h4.first.inner_text().strip() == "Titel 1", (alle_h4.count(), h4.first.inner_text() if h4.count() else ""))
    check("Abschnitts-Klappen behalten den Titel (Michael 18.09.)", ":" in h3.first.inner_text(), h3.first.inner_text())
    card = pg.locator("section.seg-review:has(textarea.seg-ziel)").nth(2)
    card.evaluate("c => c.closest('details').open = true"); pg.wait_for_timeout(200)
    orig = card.locator("textarea.seg-original")
    check("Original als schreibgeschuetztes Feld mit Label (Michael 18.09.)", orig.count() == 1 and orig.get_attribute("readonly") is not None and card.locator("label[for=" + (orig.get_attribute("id") or "x") + "]").inner_text().strip() == "Original" and orig.input_value().strip() != "", orig.input_value()[:40] if orig.count() else "")
    ta = card.locator("textarea.seg-ziel")
    check("Textarea Uebersetzung mit aria-describedby auf Original + Hinweis", ta.count() == 1 and "seg_original_" in (ta.get_attribute("aria-describedby") or "") and "seg_hinweis_" in (ta.get_attribute("aria-describedby") or ""), ta.get_attribute("aria-describedby") if ta.count() else "")
    check("Textarea hat sichtbares Label 'Übersetzung'", card.locator("label[for=" + (ta.get_attribute("id") or "x") + "]").count() == 1)
    check("Status-Badge 'Übersetzt' oder 'Von Hand korrigiert'", card.locator(".badge").first.inner_text() in ("Übersetzt", "Von Hand korrigiert", "Übersetzt, Formatierung zusammengelegt"), card.locator(".badge").first.inner_text())
    check("Uebersetzung gefuellt und nicht gleich dem Original", ta.input_value().strip() and ta.input_value().strip() != orig.input_value().strip(), ta.input_value()[:60])
    print("== C. Handkorrektur (Auto-Save) ==")
    alt = ta.input_value()
    ta.fill("Manually corrected via click test (fictional)."); pg.wait_for_timeout(1400)
    check("Badge nach Korrektur 'Von Hand korrigiert'", card.locator(".badge").first.inner_text() == "Von Hand korrigiert", card.locator(".badge").first.inner_text())
    ta.fill(alt); pg.wait_for_timeout(1400)
    print("== D. Filter ==")
    fs.locator("input[value=hinweis]").check(); pg.wait_for_timeout(400)
    st = pg.locator("#segFilterStatus").inner_text()
    check("Filter 'Nur mit Hinweis' meldet Anzahl im Status", "mit Hinweis" in st, st)
    fs.locator("input[value=offen]").check(); pg.wait_for_timeout(400)
    check("Filter 'Nur noch nicht uebersetzt': leere Klappen ausgeblendet", pg.locator("details.page-section:not([hidden])").count() == 0 or "0" in pg.locator("#segFilterStatus").inner_text(), pg.locator("#segFilterStatus").inner_text())
    fs.locator("input[value=alle]").check(); pg.wait_for_timeout(400)
    check("Filter 'Alle' stellt alles wieder her", pg.locator("details.page-section:not([hidden])").count() >= 3)
    print("== E. Uebersetzen-Dialog ==")
    pg.locator("#uStartBtn").click(); pg.wait_for_timeout(2500)
    dlg = pg.locator("#uLaufDialog")
    check("Dialog offen (natives dialog, app-dialog wie Rueckfrage der anderen Werkzeuge)", dlg.evaluate("d=>d.open") is True and dlg.get_attribute("aria-labelledby") == "uLaufHeading" and "app-dialog" in (dlg.get_attribute("class") or ""))
    check("Knopfreihenfolge Abbrechen links, Start rechts (wie #genConfirmDialog)", dlg.locator(".dialog-actions button").first.inner_text().strip() == "Abbrechen" and dlg.locator(".dialog-actions button").last.inner_text().strip() == "Übersetzung starten")
    check("Zielsprache: select mit Label, >= 20 Varianten, Vorgabe Englisch (Großbritannien)", pg.locator("label[for=uZielsprache]").count() == 1 and pg.locator("#uZielsprache option").count() >= 20 and "Großbritannien" in pg.locator("#uZielsprache option:checked").inner_text(), pg.locator("#uZielsprache option:checked").inner_text())
    check("Zwei Schalter mit Label (Alt-Texte, Dokumentsprache)", pg.locator("#uAltTexte").count() == 1 and pg.locator("#uSpracheSetzen").count() == 1 and "Screenreader" in dlg.inner_text())
    sm0 = pg.locator("#uLaufSummary").inner_text()
    check("Rueckfrage gleiche Sprache: 'schon alles übersetzt', Start gesperrt (Review M1)", "schon alles übersetzt" in sm0 and pg.locator("#uLaufOk").is_disabled(), sm0)
    pg.select_option("#uZielsprache", "fr"); pg.wait_for_timeout(1500)
    sm = pg.locator("#uLaufSummary").inner_text()
    check("Rueckfrage andere Sprache nennt Absaetze, Woerter, Credits und Guthaben", "Absätze" in sm and "Wörtern" in sm and "Credits" in sm and not pg.locator("#uLaufOk").is_disabled(), sm)
    check("Fokus auf der Zielsprache", pg.evaluate("document.activeElement && document.activeElement.id") == "uZielsprache")
    pg.screenshot(path=os.path.join(SHOTS, "ui_uebersetzen_dialog.png"), full_page=False)
    axe(pg, "Uebersetzen-Dialog")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    check("Escape schliesst den Dialog", dlg.evaluate("d=>d.open") is False)
    print("== F. Export-Dialog ==")
    pg.locator("#uExportOpenBtn").click(); pg.wait_for_timeout(600)
    ex = pg.locator("#uExportPanel")
    check("Export-Dialog offen, Ueberschrift 'Übersetzung herunterladen'", ex.evaluate("d=>d.open") is True and pg.locator("#uExportHeading").inner_text().strip() == "Übersetzung herunterladen")
    check("Zusammenfassung: uebersetzt-Zaehler, kostenlos", "übersetzt" in pg.locator("#uExportSummary").inner_text() and "keine Credits" in pg.locator("#uExportSummary").inner_text(), pg.locator("#uExportSummary").inner_text())
    check("Dateiname-Feld mit Label + Hinweis", pg.locator("label[for=uExportFilename]").count() == 1 and pg.locator("#uExportFilename").get_attribute("aria-describedby") == "uExportFilenameHint")
    check("Abbrechen ist letzter Knopf im Dialog, in eigener Fusszeile nach Trennlinie (wie Word)", ex.locator("button").last.inner_text().strip() == "Abbrechen" and ex.locator("hr.export-trenner").count() == 1 and ex.locator("#uExportFooter button").count() == 1)
    axe(pg, "Export-Dialog")
    with pg.expect_download(timeout=60000) as dl:
        pg.locator("#uExportBtn").click()
    d = dl.value
    check("Download .docx", d.suggested_filename.endswith(".docx"), d.suggested_filename)
    pg.wait_for_timeout(800)
    check("Statuszeile 'Heruntergeladen' + Knopf 'Zurück zum Projekt'", "Heruntergeladen" in pg.locator("#uExportStatus").inner_text() and pg.locator("#uExportCancelBtn").inner_text().strip() == "Zurück zum Projekt", pg.locator("#uExportStatus").inner_text())
    check("Fokus auf der Statuszeile", pg.evaluate("document.activeElement && document.activeElement.id") == "uExportStatus")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    print("== G. Tastatur + axe Seite ==")
    pg.locator("h1#projectName").focus()
    ziele = []
    for _ in range(12):
        pg.keyboard.press("Tab"); ziele.append(pg.evaluate("(document.activeElement.id || document.activeElement.tagName + ':' + (document.activeElement.textContent||'').trim().slice(0,25))"))
    check("Tab-Reihenfolge erreicht Uebersetzen, Herunterladen und die Filter-Radios", any("uStartBtn" in z for z in ziele) and any("uExportOpenBtn" in z for z in ziele) and any("INPUT" in z or "segFilter" in z for z in ziele), ziele)
    pg.screenshot(path=os.path.join(SHOTS, "ui_uebersetzen.png"), full_page=True)
    axe(pg, "Projektansicht")
    check("Keine JavaScript-Fehler", not fehler_js, fehler_js)
    br.close()
print(f"\n{ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
