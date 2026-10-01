#!/usr/bin/env python3
"""Klicktest zur Behebungsrunde 30.09.2026 (Prüfung Barrierefreiheit, Punkte 1, 3, 4, 5, 7, 8, 11, 12) an schon getaggten
PDFs (Actino Master Word: 9 Bilder, 8 mit Alt-Text in der Datei; dieselbe Datei mit Dokumentsprache en-US; Antrag Pflege:
Formular, keine Bilder). Die Sprachausgabe des Browsers ist nachgebildet (Stimmen Deutsch + Englisch auf dem Gerät). Legt ein Projekt an
und löscht es am Ende (ausser --behalten). Lädt nichts kostenpflichtig herunter.
Aufruf: /home/claude/.venv-pw/bin/python ui_michael_0930.py <ordner mit actino_master_word.pdf und antrag_pflege.pdf> [--behalten]
Zugang aus INKLUDOCS_E2E_URL / _MAIL / _PW."""
import os
import sys
import time

from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ["INKLUDOCS_E2E_MAIL"], os.environ["INKLUDOCS_E2E_PW"]
K = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/home/claude/michael-0930"
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
        print("  FEHLT", n, "--", str(i)[:400])


def axe(pg, name):
    pg.add_script_tag(url=AXE)
    pg.wait_for_timeout(400)
    r = pg.evaluate("async () => { const r = await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']}); return r.violations.map(v => ({id: v.id, impact: v.impact, n: v.nodes.length})); }")
    check(f"axe {name}: 0 ernste Verstoesse", not [v for v in r if v["impact"] in ("serious", "critical")], r)


def breite_ok(pg):
    return pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), pg.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
    # Sprachausgabe nachgebildet (headless Chromium hat keine Stimmen): zwei Stimmen AUF DEM GERÄT (Deutsch, Englisch) und
    # eine Netz-Stimme, die nie genommen werden darf. speak() merkt sich Text und Sprache jeder Äußerung.
    ctx.add_init_script("""
      window.__gesprochen = [];
      class U { constructor(t) { this.text = t; this.lang = ''; this.voice = null; this.onend = null; this.onerror = null; } }
      window.SpeechSynthesisUtterance = U;
      const stimmen = [{ name: 'Netz', lang: 'de-DE', localService: false }, { name: 'Anna', lang: 'de-DE', localService: true },
                       { name: 'Samantha', lang: 'en-US', localService: true }];
      Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: {
        getVoices: () => stimmen, addEventListener: () => {}, cancel: () => {},
        speak: (u) => window.__gesprochen.push({ text: u.text, lang: u.lang, stimme: u.voice && u.voice.name }) } });
    """)
    pg = ctx.new_page()
    js = []
    pg.on("pageerror", lambda e: js.append(str(e)))
    pg.goto(B + "/login")
    pg.fill("#email", MAIL)
    pg.fill("#password", PW)
    pg.keyboard.press("Enter")
    pg.wait_for_timeout(2500)
    r = pg.request.post(B + "/api/projects", data={"name": "Klicktest Behebung 30.09. (Test) " + time.strftime("%H:%M"), "tool": "pdf"})
    pid = r.json().get("id") or r.json().get("project_id")
    try:
        pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(800)
        for n, f in enumerate(("actino_master_word.pdf", "antrag_pflege.pdf"), 1):
            pg.set_input_files("#projUpload", os.path.join(K, f))
            for _ in range(80):
                pg.wait_for_timeout(1500)
                j = pg.request.get(B + f"/api/projects/{pid}").json()
                if len(j.get("documents") or []) >= n and (j.get("project") or {}).get("status") not in ("extracting", "processing"):
                    break
            pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
            pg.wait_for_timeout(1200)
        # Englische Kopie der Actino-Datei (nur die Dokumentsprache umgestellt): zeigt, dass beim Vorlesen die eigenen Zeilen von
        # InkluDocs in der Kontosprache und der Inhalt in der Dokumentsprache gesprochen werden (Prüfung Barrierefreiheit, Punkt 1)
        import fitz
        e = fitz.open(os.path.join(K, "actino_master_word.pdf"))
        e.set_language("en-US")
        en_bytes = e.tobytes()
        e.close()
        pg.set_input_files("#projUpload", {"name": "actino_englisch.pdf", "mimeType": "application/pdf", "buffer": en_bytes})
        for _ in range(80):
            pg.wait_for_timeout(1500)
            j = pg.request.get(B + f"/api/projects/{pid}").json()
            if len(j.get("documents") or []) >= 3 and (j.get("project") or {}).get("status") not in ("extracting", "processing"):
                break
        docs = pg.request.get(B + f"/api/projects/{pid}/dokument-ansicht").json()["documents"]
        aid = [d["id"] for d in docs if d["original_filename"] == "actino_master_word.pdf"][0]
        eid = [d["id"] for d in docs if d["original_filename"] == "actino_englisch.pdf"][0]
        fid = [d["id"] for d in docs if d["original_filename"].startswith("antrag")][0]

        print("== Punkt 5 und 11: schon getaggte PDF in „Tagging“ ==")
        pg.goto(B + f"/app?projekt={pid}&ansicht=tagging", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        pg.locator(f"#dok_karte_{aid} summary").first.click()
        pg.wait_for_timeout(500)
        # Michael Karbe, Feedback 20261001 - 2, Punkt 3: der Satz „beim Hochladen schon getaggt …“ entfällt ganz
        check("Kein Info-Satz „beim Hochladen schon getaggt“ (Punkt 3)", pg.locator(f"#dok_schon_getaggt_{aid}").count() == 0
              and "beim Hochladen" not in pg.locator(f"#dok_karte_{aid}").inner_text())
        tag_knopf = pg.locator(f"#dok_tag_{aid}")
        check("Knöpfe „Neu taggen“ (Seiten, Credits) und „Testweise taggen“, „Hörprobe“ da (Feedback 20261001 - 1, Punkt 1)",
              tag_knopf.count() == 1 and tag_knopf.inner_text().startswith("Neu taggen") and "Credits" in tag_knopf.inner_text()
              and pg.locator(f"#dok_test_{aid}").count() == 1 and pg.locator(f"#dok_hp_{aid}").count() == 1, tag_knopf.inner_text() if tag_knopf.count() else "")
        tag_knopf.click()
        pg.wait_for_timeout(500)
        dlg_text = pg.locator("#dkLaufDialog").inner_text()
        check("Dialog „Neu taggen“ nur mit dem Preissatz (Punkt 4)", pg.locator("#dkLaufHeading").inner_text() == "Neu taggen"
              and "Credits (20 Credits je Seite). Das Tagging bezahlst du nur in diesem Moment; beim Herunterladen wird es nicht noch einmal berechnet." in dlg_text
              and "Verfügbar" not in dlg_text and "Testmodus" not in dlg_text and "Seiten." not in pg.locator("#dkLaufUmfang").inner_text(), dlg_text)
        pg.click("#dkLaufCancel")
        pg.wait_for_timeout(300)
        badge = pg.locator(f"#dok_badge_{aid}").inner_text()
        check("Abzeichen nur „Getaggt“ (Feedback 20261001 - 2, Punkt 2)", badge.strip() == "Getaggt", badge)
        meta = pg.locator(f"#dok_karte_{aid} ul.dok-meta").inner_text()
        check("Bilder: 9 Bilder, 8 mit Alt-Text (wie Dialog und Hörprobe; vorher 0)", "9 Bilder, 8 mit Alt-Text" in meta, meta)
        axe(pg, "Tagging mit schon getaggter PDF")

        print("== Punkt 1: „Hörprobe vorlesen“ im Hörprobe-Dialog (Sprachausgabe nachgebildet) ==")
        if pg.locator(f"#dok_karte_{eid} details.dok-klappe:not([open])").count():
            pg.locator(f"#dok_karte_{eid} summary").first.click()
            pg.wait_for_timeout(300)
        pg.click(f"#dok_hp_{eid}")
        pg.wait_for_function("() => { const b = document.getElementById('dkHpInhalt'); return b && b.querySelectorAll('p').length > 5; }", timeout=120000)
        kn = pg.locator("#dkHpVorlesen")
        check("Knopf „Hörprobe vorlesen“ im Dialog, per Tab erreichbar vor „Schließen“",
              kn.is_visible() and pg.evaluate("() => { const a = [...document.querySelectorAll('#dkHoerprobeDialog button')].map(b => b.id); return a.indexOf('dkHpVorlesen') >= 0 && a.indexOf('dkHpVorlesen') < a.indexOf('dkHpZu'); }"))
        kn.focus()
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(1500)
        g = pg.evaluate("window.__gesprochen")
        check("Start: aria-pressed=true, Knopf heißt „Stopp“", kn.get_attribute("aria-pressed") == "true" and kn.inner_text().strip() == "Stopp", (kn.get_attribute("aria-pressed"), kn.inner_text()))
        check("Es wird vorgelesen, nur mit Stimmen auf dem Gerät (keine Netz-Stimme)", len(g) > 10 and all(x["stimme"] in ("Anna", "Samantha") for x in g), g[:3])
        check("Erste Zeile „Sprache: Englisch (en)“ ganz in der Kontosprache (eigene Zeile von InkluDocs, nicht englisch)",
              len(g) > 1 and g[0]["text"] == "Sprache:" and g[1]["text"].startswith("Englisch (en") and g[0]["lang"] == g[1]["lang"] == "de-DE", g[:2])
        ansagen = [x for x in g if x["text"].endswith(":") and x["text"].split()[0].rstrip(":") in
                   ("Sprache", "Seiten", "Zusammenfassung", "Überschrift", "Absatz", "Bild", "Liste", "Listenpunkt", "Tabelle", "Zeile", "Kopfzeile", "Link")]
        englisch = [x for x in g if x["lang"] == "en-US"]
        check("Ansagen („Überschrift Ebene 1:“) deutsch, Dokumentinhalt englisch", ansagen and englisch and all(x["lang"] == "de-DE" for x in ansagen), (ansagen[:3], englisch[:2]))
        check("Keine Meldung unter dem Knopf, wenn eine Stimme da ist", pg.locator("#dkHpVorleseStatus").inner_text().strip() == "")
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(300)
        check("Zweiter Druck stoppt: aria-pressed=false, wieder „Hörprobe vorlesen“", kn.get_attribute("aria-pressed") == "false" and kn.inner_text().strip() == "Hörprobe vorlesen", (kn.get_attribute("aria-pressed"), kn.inner_text()))
        pg.click("#dkHpZu")
        pg.wait_for_timeout(300)

        print("== Punkte 3 und 8: Herunterladen-Dialog ==")
        pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(1500)

        def dialog(doc_id):
            if pg.locator(f"#dok_karte_{doc_id} details.dok-klappe:not([open])").count():
                pg.locator(f"#dok_karte_{doc_id} summary").first.click()
                pg.wait_for_timeout(300)
            pg.click(f"#dok_export_{doc_id}")
            zs = ""
            for _ in range(30):
                pg.wait_for_timeout(400)
                zs = pg.locator("#exportSummary").inner_text()
                if "kostet" in zs:
                    break
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(300)
            return zs
        zs = dialog(fid)
        check("Antrag (keine Bilder, schon getaggt, nichts bearbeitet): kein Bildsatz, kostet nichts, kein Tagging-Satz",
              "Bilder" not in zs and "kostet nichts" in zs and "Tagging" not in zs, zs)
        zs = dialog(aid)
        check("Actino: „8 von 9 Bildern haben einen Text“, kostet nichts, kein Tagging-Satz (nie in InkluDocs getaggt)",
              "8 von 9 Bildern" in zs and "kostet nichts" in zs and "Tagging" not in zs, zs)
        bilder = [b for b in (pg.request.get(B + f"/api/projects/{pid}").json().get("images") or []) if b.get("document_id") == aid]
        pg.request.post(B + f"/api/images/{bilder[0]['id']}/alt-text", data={"alt_text": "Fiktiver Alt-Text für den Klicktest"})
        pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        zs = dialog(aid)
        check("Preis mit Zusammensetzung: „30 Credits: 25 Grundpreis und 5 für bearbeitete Alt-Texte (1)“",
              "kostet 30 Credits: 25 Grundpreis und 5 für bearbeitete Alt-Texte (1)." in zs and "Berechnet wird nur" in zs, zs)

        print("== Punkte 1, 4 und 7: Barrierefreiheitsprüfung ==")
        pg.request.post(B + f"/api/projects/{pid}/documents/{aid}/abschluss", timeout=180000)
        pg.goto(B + f"/app?projekt={pid}&ansicht=abschluss", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        pg.locator(f"#ab_karte_{aid} summary").first.click()
        pg.wait_for_timeout(3000)
        pg.locator(f"#ab_karte_{aid} details.ab-ergebnis > summary").click()   # Ergebnis zum Aufklappen (Feedback 20261001 - 2, Punkt 8)
        pg.wait_for_timeout(400)
        ad = pg.request.get(B + f"/api/projects/{pid}/documents/{aid}/abschluss").json()
        englisch = sum(1 for pr in (ad.get("probleme") or []) if ((pr.get("teile") or {}).get("lang") == "en"))
        check(f"Nicht übersetzte veraPDF-Sätze ({englisch}) stehen mit lang=\"en\" in der Liste",
              pg.locator(f"#ab_karte_{aid} ol.ab-problemliste span[lang=en]").count() == englisch, (englisch, list(ad.keys())))
        seitentext = [x for x in pg.locator(f"#ab_karte_{aid} ol.ab-problemliste li").all_inner_texts()]
        check("Keine doppelte Seitenangabe in den Problemzeilen", not any(x.count("Seite") > 2 for x in seitentext), seitentext[:4])
        kn = [b.inner_text() for b in pg.locator(f"#ab_karte_{aid} ol.ab-problemliste button").all()]
        check("„Zur Seite“-Knöpfe eindeutig benannt", len(kn) == len(set(kn)), kn)
        check("Hörprobe mit „Hörprobe vorlesen“ in der Karte", pg.locator(f"#ab_dvorlesen_{aid}").count() == 1)
        name = pg.evaluate(f"() => {{ const b = document.getElementById('ab_dvorlesen_{aid}'); return b.getAttribute('aria-labelledby').split(' ').map(i => document.getElementById(i).textContent).join(' '); }}")
        check("Knopfname mit Dokumentnamen („Hörprobe vorlesen – Dokument „…““)", name.startswith("Hörprobe vorlesen – Dokument „actino_master_word"), name)
        pg.evaluate("window.__gesprochen = []")
        pg.click(f"#ab_dvorlesen_{aid}")
        pg.wait_for_timeout(1500)
        g = pg.evaluate("window.__gesprochen")
        seiten = [x["text"] for x in g if x["text"].startswith("— Seite ")]
        check("Prüfung: das ganze Dokument wird vorgelesen, mit Seitenmarken (auch ohne Problemseiten)", len(g) > 10 and seiten[:2] == ["— Seite 1 —", "— Seite 2 —"], (len(g), seiten[:3]))
        check("Prüfung: Knopf zeigt „Stopp“, keine Meldung", pg.locator(f"#ab_dvorlesen_{aid}").get_attribute("aria-pressed") == "true" and pg.locator(f"#ab_dvstatus_{aid}").inner_text().strip() == "")
        pg.click(f"#ab_dvorlesen_{aid}")
        pg.wait_for_timeout(300)
        pg.set_viewport_size({"width": 320, "height": 800})
        pg.wait_for_timeout(800)
        b = breite_ok(pg)
        check("320 px: kein seitliches Scrollen in der Prüfung (Punkt 4)", b[0], b[1])
        pg.set_viewport_size({"width": 640, "height": 800})
        pg.wait_for_timeout(500)
        b = breite_ok(pg)
        check("640 px (200 %): kein seitliches Scrollen in der Prüfung", b[0], b[1])
        pg.set_viewport_size({"width": 1280, "height": 900})

        print("== Punkt 12: Quickinfos und Alt-Texte bei 320 px ==")
        pg.goto(B + f"/app?projekt={pid}&ansicht=quickinfos", wait_until="networkidle")
        pg.wait_for_timeout(2000)
        pg.set_viewport_size({"width": 320, "height": 800})
        pg.wait_for_timeout(800)
        b = breite_ok(pg)
        check("320 px: kein seitliches Scrollen in „Quickinfos“ (Auswahl „Gespeicherte Prompts“ mit langem Namen)", b[0], b[1])
        pg.set_viewport_size({"width": 1280, "height": 900})
        pg.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="networkidle")
        pg.wait_for_timeout(2000)
        pg.set_viewport_size({"width": 320, "height": 800})
        pg.wait_for_timeout(800)
        b = breite_ok(pg)
        check("320 px: kein seitliches Scrollen in „Alt-Texte“", b[0], b[1])
        pg.set_viewport_size({"width": 1280, "height": 900})
        check("keine Skriptfehler", not js, js[:3])
    finally:
        if not BEHALTEN:
            ab = pg.request.get(B + "/api/ausgaben").json()
            for e in (ab.get("ausgaben") if isinstance(ab, dict) else ab) or []:
                if e.get("project_id") == pid:
                    pg.request.delete(B + f"/api/ausgaben/{e['id']}")
            print("  Testprojekt geloescht:", pg.request.delete(B + f"/api/projects/{pid}").status)
        else:
            print("  Projekt bleibt stehen:", pid)
        br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
