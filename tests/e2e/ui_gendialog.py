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
Es wird NICHTS generiert — der Test bricht immer ab.
"""
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
PROJEKT = int(sys.argv[2]) if len(sys.argv) > 2 else 69
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
    ansage = page.evaluate("""() => {
        const d = document.getElementById('genConfirmDialog');
        const name = document.getElementById(d.getAttribute('aria-labelledby'));
        const besch = document.getElementById(d.getAttribute('aria-describedby'));
        return {name: name ? name.textContent.trim() : null,
                beschreibung: besch ? besch.textContent.trim() : null};
    }""")
    check("Dialog hat einen Namen (aria-labelledby)", bool(ansage["name"]), str(ansage))
    check("Dialog hat eine Beschreibung (aria-describedby) mit dem Kostensatz",
          bool(ansage["beschreibung"]) and "Credits" in ansage["beschreibung"], str(ansage))
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

    check("keine JavaScript-Fehler", not js_fehler, str(js_fehler[:2]))
    b.close()

print()
print("%d/%d Pruefungen bestanden" % (ok, ok + fail))
sys.exit(1 if fail else 0)
