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


def zweitdokumente_loeschen(pg):
    """Nur das erste Dokument bleibt: der B2-Block laedt ein zweites hoch, der Test muss wiederholbar sein."""
    r = pg.request.get(B + f"/api/projects/{PID}")
    if not r.ok: return
    docs = sorted(r.json().get("documents", []), key=lambda d: d.get("doc_index", 0))
    for d in docs[1:]:
        pg.request.delete(B + f"/api/projects/{PID}/documents/{d['id']}")

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
    check("EIN Werkzeug „Word-Dokumente“ im Auswahlmenue, kein „Dokumente übersetzen“ mehr (Steve 18.09.)", any(o.strip() == "Word-Dokumente" for o in opts) and not any("übersetzen" in o for o in opts), opts)
    zweitdokumente_loeschen(pg); pg.goto(B + f"/app?projekt={PID}&ansicht=uebersetzung"); pg.wait_for_timeout(3000)
    print("== B. Ansicht ==")
    pg.goto(B + f"/app?projekt={PID}&ansicht=uebersetzung"); pg.wait_for_timeout(3500)
    main = pg.locator("main")
    check("H1 Projektname", pg.locator("h1#projectName").count() == 1)
    check("Keine Statuszeile unter dem Projektnamen (Michael Punkt 4, wie Word)", pg.locator("#projectHeadInfo").inner_text().strip() == "")
    # Projektkopf nach Michael Karbe (Mails 21.09./22.09.2026): blaues Word-Symbol statt Statusanzeige
    check("Word-Symbol mit Alt-Text „Word-Projekt“, keine Statusanzeige", pg.locator(".projekt-kopf img.projekt-dateityp").count() == 1 and pg.locator(".projekt-kopf img.projekt-dateityp").get_attribute("alt") == "Word-Projekt" and pg.locator("#projectStatusBadge").count() == 0)
    check("Upload-Block: Weitere Word-Datei hinzufuegen", pg.locator("#addHeading").inner_text().strip() == "Weitere Word-Datei hinzufügen", pg.locator("#addHeading").inner_text())
    check("Dateiauswahl akzeptiert .docx", pg.locator("#projUpload").get_attribute("accept") == ".docx")
    check("Kein 'Bilder' und kein 'Alt-Texte generieren' in der Ansicht", "Alt-Texte generieren" not in main.inner_text() and "Bilder filtern" not in main.inner_text())
    ca_text = " ".join(pg.locator(".card-actions").all_inner_texts())
    check("Knoepfe: Uebersetzen + Herunterladen", "Übersetzen" in ca_text and "Herunterladen" in ca_text, ca_text[:200])
    check("Chatbot (InkluAgent) auch in der Uebersetzungs-Ansicht (Steve 18.09.)", pg.locator(".inkluagent-section").count() == 1 and pg.locator("#inkluagentToggle").count() == 1)
    # Seit 24.09.2026 (Michael Karbe, Mail 21.09.2026): Ansichts-Wahl direkt unter dem Projektnamen, VOR den
    # Knoepfen, in allen Ansichten gleich; die Knoepfe stehen im Feld „Funktionen und Einstellungen“.
    check("Ansichts-Wahl VOR den Hauptknoepfen, Knoepfe im Feld „Funktionen und Einstellungen“", pg.evaluate("(() => { const a = document.getElementById('uStartBtn'), b = document.getElementById('ansichtSelect'); return !!(a && b) && !!(b.compareDocumentPosition(a) & Node.DOCUMENT_POSITION_FOLLOWING); })()") and pg.locator("section.projekt-funktionen #uStartBtn").count() == 1)
    check("Keine leere Live-Region #ansichtStatus mehr (Review 2, Befund 9)", pg.locator("#ansichtStatus").count() == 0)
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
    fs = pg.locator("fieldset#segFilterFieldset")
    card = pg.locator("section.seg-review:has(textarea.seg-ziel)").nth(2)
    card.evaluate("c => c.closest('details').open = true"); pg.wait_for_timeout(200)
    ta = card.locator("textarea.seg-ziel")
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
    print("== B2. Ansichts-Wahl (Testumbau 18.09.) ==")
    # Fuer die Alt-Text-Ansicht braucht das Projekt ein Dokument MIT Bildern: fiktives Testdokument dazuladen.
    BILDDOC = os.environ.get("INKLUDOCS_E2E_BILDDOC", "/home/claude/work/repo/tests/fixtures/testdokument_inkludocs.docx")
    if pg.locator("h2.doc-heading").count() < 2 and os.path.isfile(BILDDOC):
        pg.set_input_files("#projUpload", BILDDOC); pg.wait_for_timeout(12000)
        pg.goto(B + f"/app?projekt={PID}&ansicht=uebersetzung"); pg.wait_for_timeout(3500)
    check("Zweites Dokument (mit Bildern) im Projekt", pg.locator("h2.doc-heading").count() >= 2, pg.locator("h2.doc-heading").count())
    sel = pg.locator("#ansichtSelect")
    check("Ansichts-Wahl: Ausklappliste mit Label + Knopf Oeffnen, aktuell Uebersetzung", sel.count() == 1 and pg.locator("label[for=ansichtSelect]").inner_text().strip() == "Ansicht" and sel.input_value() == "uebersetzung" and pg.locator("#ansichtOeffnen").count() == 1, sel.input_value() if sel.count() else "")
    pg.select_option("#ansichtSelect", "alttexte"); pg.wait_for_timeout(300)
    check("Pfeil/Auswahl allein wechselt NICHT (WCAG 3.2.2)", pg.locator("#segFilterBar").count() == 1)
    pg.locator("#ansichtOeffnen").click(); pg.wait_for_timeout(2500)
    check("Fokus nach dem Wechsel auf der H1 der neuen Ansicht (Review 2, Befund 6)", pg.evaluate("document.activeElement && document.activeElement.id") == "projectName", pg.evaluate("document.activeElement && document.activeElement.id"))
    check("Oeffnen wechselt zur Alt-Text-Ansicht (Bilder filtern, Sprache der Alt-Texte, Adresse ?ansicht=alttexte)", pg.locator("#imageFilterBar").count() == 1 and pg.locator("#altLangSelect").count() == 1 and "ansicht=alttexte" in pg.url and pg.locator("#segFilterBar").count() == 0, (pg.url, pg.locator("#imageFilterBar").count(), pg.locator("#altLangSelect").count()))
    check("Alt-Text-Ansicht unveraendert: Bilderkarten, Filterkarte, Upload-Block, Chatbot", pg.locator("section.image-review").count() >= 1 and pg.locator("#projUploadZone").count() == 1 and pg.locator(".inkluagent-section").count() == 1)
    check("Alt-Text-Ansicht hat dieselbe Ansichts-Wahl", pg.locator("#ansichtSelect").count() == 1 and pg.locator("#ansichtSelect").input_value() == "alttexte")
    pg.locator(".card-actions button:has-text('Herunterladen')").first.click(); pg.wait_for_timeout(1500)
    check("Herunterladen-Dialog der Alt-Text-Ansicht bietet „Als Word, Englisch (Großbritannien)“", pg.locator("#exportUebersetzungBtn").count() == 1 and "Großbritannien" in pg.locator("#exportUebersetzungBtn").inner_text(), pg.locator("#exportUebersetzungBtn").inner_text() if pg.locator("#exportUebersetzungBtn").count() else "")
    axe(pg, "Alt-Text-Ansicht mit Export-Dialog")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    pg.go_back(); pg.wait_for_timeout(2500)
    check("Browser Zurueck fuehrt in die Uebersetzungs-Ansicht", pg.locator("#segFilterBar").count() == 1, pg.url)
    if pg.locator("#segFilterBar").count() == 0:
        pg.goto(B + f"/app?projekt={PID}&ansicht=uebersetzung"); pg.wait_for_timeout(3000)
    # Ganzprojekt-Export mit einem nicht uebersetzten Dokument: Download laeuft, das Dokument wird ausgelassen und benannt.
    pg.locator("#uExportOpenBtn").click(); pg.wait_for_timeout(600)
    with pg.expect_download(timeout=60000) as dl2:
        pg.locator("#uExportBtn").click()
    pg.wait_for_timeout(800)
    st = pg.locator("#uExportStatus").inner_text()
    check("Export mit nicht uebersetztem 2. Dokument: Download + Hinweis 'ausgelassen'", dl2.value.suggested_filename.endswith(".docx") and "ausgelassen" in st, (dl2.value.suggested_filename, st[:160]))
    pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    zweitdokumente_loeschen(pg); pg.goto(B + f"/app?projekt={PID}&ansicht=uebersetzung"); pg.wait_for_timeout(3000)
    check("Aufgeraeumt: wieder ein Dokument", pg.locator("h2.doc-heading").count() == 1, pg.locator("h2.doc-heading").count())
    pg.screenshot(path=os.path.join(SHOTS, "ui_uebersetzen.png"), full_page=True)
    axe(pg, "Projektansicht")
    check("Keine JavaScript-Fehler", not fehler_js, fehler_js)
    br.close()
print(f"\n{ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
