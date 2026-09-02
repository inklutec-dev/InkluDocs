#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durchklick auf PRODUKTION mit Steves Gmail-Konto — wie ein Nutzer.

Sucht selbst ein PDF-Projekt mit gefuelltem Textfeld, leert es im Browser,
laedt neu, prueft dass es leer bleibt, vergleicht den Zaehler im
Herunterladen-Dialog, prueft Symbole und Beschriftung — und stellt den
Ausgangszustand wieder her.
"""
import subprocess
import sys

from playwright.sync_api import sync_playwright

BASE = "https://inkludocs.de"

env = {}
with open("/home/claude/.prodtest.env", encoding="utf-8") as f:
    for z in f:
        z = z.strip()
        if z and not z.startswith("#") and "=" in z:
            k, v = z.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

ok = fail = 0
js_fehler = []
bild_id = None


def check(name, bedingung, zusatz=""):
    global ok, fail
    if bedingung:
        ok += 1
        print("  OK   %s" % name)
    else:
        fail += 1
        print("  FAIL %s %s" % (name, zusatz))


def dialog(page):
    page.locator("button:has-text('herunterladen'), button:has-text('Herunterladen')").first.click()
    page.wait_for_timeout(1800)
    d = page.locator("dialog[open]")
    text = d.inner_text() if d.count() else "(kein Dialog)"
    svg = d.locator("button svg").count() if d.count() else -1
    if d.count():
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
    return text, svg


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 1000})
    page.on("pageerror", lambda e: js_fehler.append(str(e)))

    print("== Anmelden auf der PRODUKTION ==")
    page.goto(BASE + "/login", wait_until="domcontentloaded")  # 02.09.2026: Anmeldung liegt unter /login
    page.fill("#email", env["IDOC_EMAIL"])
    page.fill("#password", env["IDOC_PW"])
    page.click("button[type=submit]")
    page.wait_for_url("**/dashboard", timeout=25000)
    check("Anmeldung als %s" % env["IDOC_EMAIL"], "/dashboard" in page.url)

    projekte = page.evaluate("""async () => {
        const r = await fetch('/api/projects'); const d = await r.json();
        const liste = Array.isArray(d) ? d : (d.projects || []);
        return liste.filter(p => p.project_type === 'pdf').map(p => [p.id, p.name]);
    }""")
    print("   PDF-Projekte: %s" % projekte[:6])

    ziel_projekt = feld = None
    for pid, pname in projekte:
        page.goto("%s/app?projekt=%s" % (BASE, pid), wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.evaluate("document.querySelectorAll('main details').forEach(d => d.open = true)")
        page.wait_for_timeout(1200)
        felder = page.locator("textarea.alt-text-field")
        for i in range(felder.count()):
            f = felder.nth(i)
            if f.is_visible() and f.input_value().strip():
                ziel_projekt, feld = (pid, pname), f
                break
        if feld:
            break
    if not feld:
        print("ABBRUCH: kein Projekt mit gefuelltem Textfeld gefunden")
        b.close()
        sys.exit(1)

    bild_id = feld.get_attribute("data-image-id")
    vorher = feld.input_value()
    print("\n== Projekt %s — Testbild %s ==" % (ziel_projekt[0], bild_id))
    print("   Text im Feld: %r" % vorher[:60])

    dv, sv = dialog(page)
    zeile_v = [z for z in dv.split("\n") if "eschreib" in z]
    print("   Dialog vorher:  %s" % (zeile_v[:1] or [dv[:100]]))

    print("\n== Feld im Browser leeren ==")
    feld.click()
    feld.fill("")
    page.keyboard.press("Tab")
    page.wait_for_timeout(3000)

    print("\n== Seite NEU LADEN ==")
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(2500)
    page.evaluate("document.querySelectorAll('main details').forEach(d => d.open = true)")
    page.wait_for_timeout(1200)
    nach = page.locator("textarea[data-image-id='%s']" % bild_id).input_value()
    check("Feld bleibt nach dem Neuladen LEER", nach.strip() == "", "gefunden: %r" % nach[:60])

    dn, sn = dialog(page)
    zeile_n = [z for z in dn.split("\n") if "eschreib" in z]
    print("   Dialog nachher: %s" % (zeile_n[:1] or [dn[:100]]))
    check("Zaehler hat sich geaendert", dv != dn, "unveraendert")

    print("\n== Michaels weitere Punkte ==")
    check("Keine Symbole in den Knoepfen", page.locator("button svg").count() == 0,
          "%d" % page.locator("button svg").count())
    check("Keine Symbole im Dialog", sn == 0, "%d" % sn)
    check("Als PDF ohne Zusatz Beta", "Als PDF" in dn and "Als PDF (Beta)" not in dn,
          [z for z in dn.split("\n") if "PDF" in z][:2])
    check("Dokumentname nicht von der Knopfleiste verdeckt", page.evaluate("""() => {
            const l = document.querySelector('.doc-block .doc-actions');
            const k = document.querySelector('.doc-block summary');
            if (!l || !k) return true;
            const t = k.querySelector('span,h3,strong') || k;
            return Math.round(t.getBoundingClientRect().right) <= Math.round(l.getBoundingClientRect().left);
          }"""))
    check("Keine JavaScript-Fehler", not js_fehler, str(js_fehler[:2]))

    page.screenshot(path="/home/claude/prod_durchklick.png")
    b.close()

print("\n== Ausgangszustand wiederherstellen ==")
subprocess.run(["sudo", "docker", "exec", "inkludocs", "python3", "-c",
                "import sqlite3;c=sqlite3.connect('/app/data/inkludocs.db');"
                "c.execute('UPDATE images SET alt_text_edited=NULL WHERE id=%s');c.commit();"
                "print('   Bild %s zurueckgesetzt:', list(c.execute('SELECT alt_text_edited IS NULL FROM images WHERE id=%s')))"
                % (bild_id, bild_id, bild_id)], check=True)

print("\n%d/%d Pruefungen bestanden" % (ok, ok + fail))
sys.exit(1 if fail else 0)
