#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Klicktest: einheitliche Knopfnamen + Rueckfrage vor dem Generieren.

Michael Karbe, 31.08.2026: alle Generieren-Knoepfe sollen gleich heissen, und
vor dem Start soll der Verbrauch genannt werden, mit Weg zurueck („Ich habe
jetzt beim Testen einiges Credits verbraucht, weil ich nicht Abbrechen
konnte").

Geprueft wird:
  A. Knopfnamen: „Alt-Texte generieren" und „Herunterladen", ohne Anzahl,
     ohne Preis, ohne „Alle"/„Ganzes"/„neu".
  B. Der Dialog nennt Umfang, Anzahl, Preis und Guthaben.
  C. Was ein Screenreader beim Oeffnen ansagt (Name + Beschreibung des
     Dialogs) — dafuer sind aria-labelledby und aria-describedby gesetzt.
  D. Abbrechen und Escape schliessen, ohne etwas zu starten.
  E. axe: 0 Verstoesse bei geoeffnetem Dialog.
  F. (01.09.2026) Word-Projekt: „Herunterladen (Beta)" auf den Knoepfen, der
     Kostensatz nennt die Gesamtzahl („werden beschrieben") und die
     mitgebrachten Texte („davon bringen schon einen Text … mit").
  G. (01.09.2026) Formular-Projekt: „Quickinfos generieren" auf beiden Ebenen,
     derselbe Dialog mit Umfang, Anzahl, Preis; Abbrechen startet nichts.
  H. (01.09.2026) Die Ansage beim Oeffnen enthaelt Umfang UND Kostensatz
     (aria-describedby nennt beide).
Es wird NICHTS generiert — der Test bricht immer ab.
Aufruf: ui_gendialog.py [basis] [pdf-projekt] [word-projekt] [formular-projekt]
"""
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
PROJEKT = int(sys.argv[2]) if len(sys.argv) > 2 else 69
WORD = int(sys.argv[3]) if len(sys.argv) > 3 else 320
FORMULAR = int(sys.argv[4]) if len(sys.argv) > 4 else 326
MAIL = "steve.weidel@inklutec.de"
PW = "Ewigwind-2026"
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"

ok = fail = 0
js_fehler = []


def check(name, bedingung, zusatz=""):
    global ok, fail
    if bedingung:
        ok += 1
        print("  OK   %s" % name)
    else:
        fail += 1
        print("  FAIL %s %s" % (name, zusatz))


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 1000})
    page.on("pageerror", lambda e: js_fehler.append(str(e)))

    page.goto(BASE + "/", wait_until="domcontentloaded")
    page.fill("#email", MAIL)
    page.fill("#password", PW)
    page.click("button[type=submit]")
    page.wait_for_url("**/dashboard", timeout=20000)
    page.goto("%s/app?projekt=%d" % (BASE, PROJEKT), wait_until="networkidle")
    page.wait_for_timeout(2500)

    print("== A. Knopfnamen ==")
    namen = [x.inner_text().strip().split("\n")[0]
             for x in page.locator("button").all() if x.is_visible()]
    gen = [n for n in namen if "generieren" in n.lower()]
    dl = [n for n in namen if "erunterladen" in n]
    print("     Generieren-Knoepfe: %s" % gen)
    print("     Herunterladen:      %s" % dl)
    check("alle Generieren-Knoepfe heissen gleich",
          gen and all(n == "Alt-Texte generieren" for n in gen), str(gen))
    check("kein Preis und keine Anzahl auf den Knoepfen",
          not any("Credits" in n or n[:1].isdigit() for n in gen + dl), str(gen + dl))
    check("Herunterladen ohne den Zusatz Ganzes Projekt",
          all(n == "Herunterladen" for n in dl), str(dl))

    print("== B. Dialog oeffnen ==")
    page.locator("button:has-text('Alt-Texte generieren')").first.click()
    page.wait_for_timeout(1500)
    dlg = page.locator("#genConfirmDialog")
    check("Dialog ist offen", dlg.evaluate("d => d.open") is True)
    kopf = page.locator("#genConfirmTitle").inner_text().strip()
    umfang = page.locator("#genConfirmScope").inner_text().strip()
    text = page.locator("#genConfirmBody").inner_text().strip()
    print("     Ueberschrift: %r" % kopf)
    print("     Umfang:       %r" % umfang)
    print("     Kostensatz:   %r" % text)
    check("Ueberschrift benennt die Aktion", kopf == "Alt-Texte generieren", kopf)
    check("Umfang wird benannt", bool(umfang), umfang)
    check("Kostensatz nennt Anzahl und Credits",
          "Credits" in text and any(c.isdigit() for c in text), text)

    print("== C. Was ein Screenreader ansagt ==")
    # aria-describedby darf mehrere IDs tragen (seit 01.09.2026: Umfang + Kostensatz) —
    # ein Screenreader liest sie in dieser Reihenfolge hintereinander.
    ansage = page.evaluate("""() => {
        const d = document.getElementById('genConfirmDialog');
        const name = document.getElementById(d.getAttribute('aria-labelledby'));
        const teile = (d.getAttribute('aria-describedby') || '').split(/\\s+/).filter(Boolean)
            .map(id => { const e = document.getElementById(id); return e ? e.textContent.trim() : ''; });
        return {name: name ? name.textContent.trim() : null,
                beschreibung: teile.length ? teile.join(' ') : null};
    }""")
    check("Dialog hat einen Namen (aria-labelledby)", bool(ansage["name"]), str(ansage))
    check("Dialog hat eine Beschreibung (aria-describedby) mit dem Kostensatz",
          bool(ansage["beschreibung"]) and "Credits" in ansage["beschreibung"], str(ansage))
    # H. (01.09.2026): aria-describedby nennt BEIDE Absaetze — Umfang und Kostensatz —,
    # damit die Entscheidung „ganzes Projekt oder nur dieses Dokument?" hoerbar ist.
    beschr_ids = page.evaluate("document.getElementById('genConfirmDialog').getAttribute('aria-describedby')")
    check("aria-describedby nennt Umfang und Kostensatz",
          beschr_ids == "genConfirmScope genConfirmBody", repr(beschr_ids))
    check("Die Ansage beginnt mit dem Umfang",
          (ansage["beschreibung"] or "").startswith(umfang), str(ansage))
    # Generieren ueberschreibt alles (Michael Karbe/Steve 01.09.2026): Gesamtzahl + Hinweis auf Ueberschreiben.
    check("Kostensatz nach Michael: „beinhalten insgesamt … Bilder … benötigt … Credits“", "beinhalten insgesamt" in text and "benötigt" in text, text)
    check("Kostensatz: Abbruch moeglich (Knopf existiert seit 01.09.)", "abgebrochen werden" in text, text)
    check("Kostensatz nennt eigene Texte NICHT mehr (Michael Punkt 3, 02.09.)",
          "stammen von" not in text and "stammt von dir" not in text, text)
    check("Kostensatz nennt keinen Rest eines abgebrochenen Laufs mehr (Michael Punkt 1, 02.09.)",
          "abgebrochenen Lauf" not in text, text)
    check("Statusmeldungs-Box vorhanden und verborgen (Michael Punkt 2, 02.09.)",
          page.locator("#laufMeldung").count() == 1 and page.locator("#laufMeldung").is_hidden(), None)
    # Einzahl (Steve 01.09.2026, gehoert: „1 davon bringen“, „1 Bilder“): kein „1 <Mehrzahl>“ im Satz.
    import re as _re
    check("Keine Einzahl-Mehrzahl-Panne im Kostensatz",
          not _re.search(r"\b1 (davon bringen|Texte davon|Bilder|Felder|KI-Vorschläge)\b", text), text)
    print("     Angesagt wird: %r — %r" % (ansage["name"], (ansage["beschreibung"] or "")[:90]))
    check("Fokus liegt im Dialog",
          page.evaluate("document.getElementById('genConfirmDialog').contains(document.activeElement)"))
    print("     Fokus auf: %r" % page.evaluate("document.activeElement && document.activeElement.textContent.trim()"))

    print("== D. axe bei offenem Dialog ==")
    page.add_script_tag(url=AXE)
    r = page.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',"
                      "values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
    check("axe: 0 Verstoesse", len(r["violations"]) == 0,
          str([(v["id"], len(v["nodes"])) for v in r["violations"]]))

    print("== E. Abbrechen und Escape ==")
    page.locator("#genConfirmCancel").click()
    page.wait_for_timeout(700)
    check("Abbrechen schliesst den Dialog", dlg.evaluate("d => d.open") is False)
    check("Projekt laeuft NICHT", page.locator("text=Wird analysiert").count() == 0)

    page.locator("button:has-text('Alt-Texte generieren')").first.click()
    page.wait_for_timeout(1200)
    page.keyboard.press("Escape")
    page.wait_for_timeout(600)
    check("Escape schliesst den Dialog", dlg.evaluate("d => d.open") is False)

    print("== F. Word-Projekt: Beta auf dem Knopf, Gesamtzahl im Kostensatz ==")
    page.goto("%s/app?projekt=%d" % (BASE, WORD), wait_until="networkidle")
    page.wait_for_timeout(2500)
    namen_w = [x.inner_text().strip().split("\n")[0]
               for x in page.locator("button").all() if x.is_visible()]
    dl_w = [n for n in namen_w if "erunterladen" in n]
    print("     Herunterladen (Word): %s" % dl_w)
    check("Word: alle Herunterladen-Knoepfe heissen „Herunterladen (Beta)“",
          dl_w and all(n == "Herunterladen (Beta)" for n in dl_w), str(dl_w))
    page.locator("button:has-text('Herunterladen (Beta)')").first.click()
    page.wait_for_timeout(1200)
    kopf_dl = page.locator("#exportPanelHeading").inner_text().strip()
    check("Word: Dialog-Ueberschrift traegt das Beta", kopf_dl.endswith("(Beta)"), kopf_dl)
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    gen_w = page.locator("button:has-text('Alt-Texte generieren')")
    if gen_w.count():
        gen_w.first.click()
        page.wait_for_timeout(1500)
        text_w = page.locator("#genConfirmBody").inner_text().strip()
        print("     Kostensatz (Word): %r" % text_w)
        check("Word: Kostensatz nach Michael (Projekt: „Die Dokumente des Projekts beinhalten insgesamt“)",
              "Die Dokumente des Projekts beinhalten insgesamt" in text_w and "benötigt" in text_w, text_w)
        check("Word: keine Einzahl-Mehrzahl-Panne",
              not _re.search(r"\b1 (Texte davon|Bilder)\b", text_w), text_w)
        vorschau = page.evaluate("""async (id) => {
            const r = await fetch('/api/projects/' + id + '/generate/vorschau', {method:'POST',
                headers:{'Content-Type':'application/json'}, body: JSON.stringify({modus:'alle'})});
            return await r.json(); }""", WORD)
        print("     Vorschau (Word): anzahl=%s eigene=%s modus=%s" % (vorschau.get("anzahl"), vorschau.get("eigene"), vorschau.get("modus")))
        check("Vorschau: ein Modus „alle“, liefert eigene", vorschau.get("modus") == "alle" and "eigene" in vorschau, str(vorschau)[:120])
        check("Vorschau zaehlt ALLE Bilder des Projekts", vorschau.get("anzahl") == page.evaluate("document.querySelectorAll('section.image-review').length"), (vorschau.get("anzahl"), page.evaluate("document.querySelectorAll('section.image-review').length")))
        page.locator("#genConfirmCancel").click()
        page.wait_for_timeout(500)
        check("Word: Abbrechen startet nichts", page.locator("#genConfirmDialog").evaluate("d => d.open") is False)
    else:
        print("     (kein Generieren-Knopf im Word-Projekt %d — Kostensatz-Pruefung uebersprungen)" % WORD)

    print("== G. Formular-Projekt: Quickinfos generieren mit derselben Rueckfrage ==")
    page.goto("%s/app?projekt=%d" % (BASE, FORMULAR), wait_until="networkidle")
    page.wait_for_timeout(2500)
    namen_f = [x.inner_text().strip().split("\n")[0]
               for x in page.locator("button").all() if x.is_visible()]
    gen_f = [n for n in namen_f if "generieren" in n.lower() and n not in ("Generieren", "Neu generieren")]
    print("     Generieren-Knoepfe (Formular): %s" % gen_f)
    check("Formular: Sammel-Knoepfe heissen „Quickinfos generieren“",
          gen_f and all(n == "Quickinfos generieren" for n in gen_f), str(gen_f))
    check("Formular: kein Preis und keine Anzahl auf den Knoepfen",
          not any("Credits" in n or n[:1].isdigit() for n in gen_f), str(gen_f))
    zus = page.evaluate("""() => [...document.querySelectorAll('#fGenAllBtn .visually-hidden')].map(e => e.textContent.trim())""")
    check("Formular: Projekt-Knopf traegt den versteckten Zusatz „– ganzes Projekt“",
          any("ganzes Projekt" in z for z in zus), str(zus))
    if gen_f:
        page.locator("button:has-text('Quickinfos generieren')").first.click()
        page.wait_for_timeout(1500)
        dlg_f = page.locator("#genConfirmDialog")
        check("Formular: Dialog ist offen", dlg_f.evaluate("d => d.open") is True)
        kopf_f = page.locator("#genConfirmTitle").inner_text().strip()
        umfang_f = page.locator("#genConfirmScope").inner_text().strip()
        text_f = page.locator("#genConfirmBody").inner_text().strip()
        print("     Ueberschrift: %r | Umfang: %r" % (kopf_f, umfang_f))
        print("     Kostensatz:   %r" % text_f)
        check("Formular: Ueberschrift „Quickinfos generieren“", kopf_f == "Quickinfos generieren", kopf_f)
        check("Formular: Umfang wird benannt", bool(umfang_f), umfang_f)
        check("Formular: Kostensatz nennt Felder, Anzahl und Credits",
              "Credits" in text_f and any(c.isdigit() for c in text_f) and ("Feld" in text_f or "KI-Vorschl" in text_f), text_f)
        check("Formular: Fokus liegt im Dialog",
              page.evaluate("document.getElementById('genConfirmDialog').contains(document.activeElement)"))
        page.add_script_tag(url=AXE)
        r = page.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',"
                          "values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
        check("Formular: axe 0 Verstoesse bei offenem Dialog", len(r["violations"]) == 0,
              str([(v["id"], len(v["nodes"])) for v in r["violations"]]))
        page.locator("#genConfirmCancel").click()
        page.wait_for_timeout(700)
        check("Formular: Statusmeldungs-Box vorhanden und verborgen (Michael Punkt 2, 02.09.)",
              page.locator("#fLaufMeldung").count() == 1 and page.locator("#fLaufMeldung").is_hidden(), None)
        check("Formular: Abbrechen schliesst, nichts laeuft",
              dlg_f.evaluate("d => d.open") is False and page.locator("text=KI generiert").count() == 0)
    else:
        print("     (kein Sammel-Knopf im Formular-Projekt %d — alle Felder gefuellt und keine KI-Felder?)" % FORMULAR)

    check("keine JavaScript-Fehler", not js_fehler, str(js_fehler[:2]))
    b.close()

print()
print("%d/%d Pruefungen bestanden" % (ok, ok + fail))
sys.exit(1 if fail else 0)
