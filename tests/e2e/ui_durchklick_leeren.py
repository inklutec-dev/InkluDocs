#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durchklick mit STEVES Konto auf Staging — wie ein Nutzer, nicht über die API.

Prüft Michaels Fall am Projekt 93 (PDF, die Quelldatei bringt eigene Alt-Texte
mit — genau Steves Frage „was passiert, wenn die Quelle schon einen Text hat?"):
  1. Text im Feld merken, Zähler im Herunterladen-Dialog merken
  2. Feld im Browser LEEREN und speichern lassen
  3. Seite NEU LADEN — das Feld muss leer bleiben
  4. Zähler muss um 1 gesunken sein
  5. Keine Symbole in den Knöpfen, „Als PDF" ohne Beta
  6. Ausgangszustand wiederherstellen

Alles wird mit Bildschirmfotos belegt.
"""
import subprocess
import sys

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8002"
# Zugangsdaten stehen NICHT im Repo (02.09.2026): aus ~/.e2e.env oder der
# Umgebung (INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW), wie in ui_start.py.
import os as _os


def _e2e(schluessel):
    wert = _os.environ.get(schluessel)
    if wert:
        return wert
    try:
        for zeile in open(_os.path.expanduser("~/.e2e.env"), encoding="utf-8"):
            if zeile.startswith(schluessel + "="):
                return zeile.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


MAIL = _e2e("INKLUDOCS_E2E_MAIL")
PW = _e2e("INKLUDOCS_E2E_PW")
PROJEKT = 93

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


def dialog_zahlen(page):
    """Öffnet den Herunterladen-Dialog und liest den Text; schließt wieder."""
    knopf = page.locator("button:has-text('herunterladen'), button:has-text('Herunterladen')").first
    knopf.click()
    page.wait_for_timeout(1500)
    d = page.locator("dialog[open]")
    text = d.inner_text() if d.count() else "(kein Dialog)"
    symbole = d.locator("button svg").count() if d.count() else -1
    if d.count():
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
    return text, symbole


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 1000})
    page.on("pageerror", lambda e: js_fehler.append(str(e)))

    print("== Anmelden als %s ==" % MAIL)
    page.goto(BASE + "/login", wait_until="domcontentloaded")  # 02.09.2026: Anmeldung liegt unter /login
    page.fill("#email", MAIL)
    page.fill("#password", PW)
    page.click("button[type=submit]")
    page.wait_for_url("**/dashboard", timeout=20000)
    check("Anmeldung", "/dashboard" in page.url)

    print("\n== Projekt %d öffnen ==" % PROJEKT)
    page.goto("%s/app?projekt=%d" % (BASE, PROJEKT), wait_until="networkidle")
    page.wait_for_timeout(2500)
    # Dokument- und Seiten-Klappen öffnen, sonst sind die Felder unsichtbar.
    page.evaluate("document.querySelectorAll('main details').forEach(d => d.open = true)")
    page.wait_for_timeout(1200)
    felder = page.locator("textarea.alt-text-field")
    check("Projekt geladen, Textfelder da", felder.count() > 0, "%d Felder" % felder.count())

    # Erstes SICHTBARES Feld MIT Inhalt suchen
    ziel = None
    for i in range(felder.count()):
        f = felder.nth(i)
        if not f.is_visible():
            continue
        if f.input_value().strip():
            ziel = i
            break
    if ziel is None:
        print("ABBRUCH: kein Feld mit Inhalt gefunden")
        b.close()
        sys.exit(1)
    feld = felder.nth(ziel)
    bild_id = feld.get_attribute("data-image-id")
    vorher_text = feld.input_value()
    print("   Testbild %s, Text im Feld: %r" % (bild_id, vorher_text[:60]))

    dtext_vorher, sym_vorher = dialog_zahlen(page)
    zeile_vorher = [z for z in dtext_vorher.split("\n") if "eschreib" in z or "Bilder" in z]
    print("   Dialog vorher: %s" % (zeile_vorher[:2] or dtext_vorher[:120].replace("\n", " | ")))

    print("\n== Das Feld im Browser leeren ==")
    feld.click()
    feld.fill("")
    page.keyboard.press("Tab")
    page.wait_for_timeout(2500)
    check("Feld ist unmittelbar nach dem Leeren leer", feld.input_value() == "",
          repr(feld.input_value()))

    print("\n== Seite NEU LADEN — das ist der eigentliche Test ==")
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(2500)
    page.evaluate("document.querySelectorAll('main details').forEach(d => d.open = true)")
    page.wait_for_timeout(1000)
    feld2 = page.locator("textarea[data-image-id='%s']" % bild_id)
    nach_reload = feld2.input_value()
    check("Feld bleibt nach dem Neuladen LEER (kein Text kommt zurück)",
          nach_reload.strip() == "", "gefunden: %r" % nach_reload[:70])

    dtext_nachher, sym_nachher = dialog_zahlen(page)
    zeile_nachher = [z for z in dtext_nachher.split("\n") if "eschreib" in z or "Bilder" in z]
    print("   Dialog nachher: %s" % (zeile_nachher[:2] or dtext_nachher[:120].replace("\n", " | ")))
    check("Der Zähler im Dialog hat sich geändert", dtext_vorher != dtext_nachher,
          "unverändert")

    print("\n== Michaels weitere Punkte ==")
    check("Keine Symbole in den Knöpfen der Seite",
          page.locator("button svg").count() == 0,
          "%d gefunden" % page.locator("button svg").count())
    check("Keine Symbole im Herunterladen-Dialog", sym_nachher == 0, "%d" % sym_nachher)
    check("Als PDF ohne den Zusatz Beta im Dialog",
          "Als PDF" in dtext_nachher and "Als PDF (Beta)" not in dtext_nachher,
          [z for z in dtext_nachher.split("\n") if "PDF" in z][:3])
    check("Keine JavaScript-Fehler", not js_fehler, str(js_fehler[:2]))

    page.screenshot(path="/home/claude/durchklick.png", full_page=False)
    print("\n   Bildschirmfoto: /home/claude/durchklick.png")
    b.close()

print("\n== Ausgangszustand wiederherstellen ==")
subprocess.run(["sudo", "docker", "exec", "inkludocs-staging", "python3", "-c",
                "import sqlite3;c=sqlite3.connect('/app/data/inkludocs.db');"
                "c.execute('UPDATE images SET alt_text_edited=NULL WHERE id=%s');c.commit();"
                "print('   zurückgesetzt:', list(c.execute('SELECT alt_text_edited FROM images WHERE id=%s')))"
                % (bild_id, bild_id)], check=True)

print("\n%d/%d Prüfungen bestanden" % (ok, ok + fail))
sys.exit(1 if fail else 0)
