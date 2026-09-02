#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rauchtest ALLER angemeldeten Seiten: Zeigt die Seite ueberhaupt etwas an?

Anlass (31.08.2026): Die Abo-Seite war zwei Tage lang tot — ein Jinja-Ausdruck
im raw-Block brach das Seitenskript ab, die Seite zeigte nur "Wird geladen ...".
Kein Test hat das gemerkt: die API-Tests laufen ohne Browser, und axe meldet auf
einer leeren Seite null Verstoesse. Genau diese Luecke schliesst dieser Test.

Je Seite wird geprueft:
  - HTTP-Antwort 200, keine Weiterleitung auf die Anmeldung
  - KEIN JavaScript-Fehler (pageerror) und kein console.error
  - kein haengender Platzhalter mehr ("Wird geladen", "Lade ...")
  - genau eine H1, und der Hauptbereich hat echten Inhalt (> 40 Zeichen)
  - keine Netz-Antwort ab 400 (ausser bewusst erlaubten)

Aufruf: /home/claude/.venv-pw/bin/python ui_smoke.py [BASIS]
Standard-Basis ist die Staging-Instanz im Container-Netz.
"""
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
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


LOGIN_MAIL = _e2e("INKLUDOCS_E2E_MAIL")
LOGIN_PW = _e2e("INKLUDOCS_E2E_PW")

# Alle Seiten hinter der Anmeldung (aus den Routen in main.py).
SEITEN = [
    ("/dashboard", "Startseite"),
    ("/projekte", "Meine Projekte"),
    ("/projekt-neu", "Neues Projekt"),
    ("/app", "Projektansicht"),
    ("/einstellungen", "Einstellungen"),
    ("/konto", "E-Mail & Passwort"),
    ("/abo", "Abo & Verbrauch"),
    ("/team", "Team"),
    ("/api-schluessel", "API-Schlüssel"),
    ("/prompts", "Meine Prompts"),
    ("/stammdaten", "Meine Stammdaten"),
    ("/geteilte-projekte", "Geteilte Projekte"),
    ("/benutzer", "Benutzerverwaltung"),
    ("/datensicherheit", "Datensicherheit"),
    ("/impressum-app", "Impressum"),
    ("/nutzungsbedingungen-app", "Nutzungsbedingungen"),
    ("/widerruf-app", "Widerrufsbelehrung"),
]

PLATZHALTER = ("Wird geladen", "wird geladen", "Lade ", "Loading")
# 404 auf das Browser-Symbol ist harmlos und projektueblich.
ERLAUBT_404 = ("/favicon.ico",)

ok = fail = 0
befunde = []


def check(seite, name, bedingung, zusatz=""):
    global ok, fail
    if bedingung:
        ok += 1
    else:
        fail += 1
        befunde.append("%-26s %s %s" % (seite, name, zusatz))
    return bedingung


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    js_fehler, netz = [], []
    page.on("pageerror", lambda e: js_fehler.append(str(e)))
    page.on("console", lambda m: js_fehler.append("console.error: " + m.text)
            if m.type == "error" else None)
    page.on("response", lambda r: netz.append("%d %s" % (r.status, r.url))
            if r.status >= 400 and not any(a in r.url for a in ERLAUBT_404) else None)

    page.goto(BASE + "/login", wait_until="domcontentloaded")  # 02.09.2026: Anmeldung liegt unter /login
    page.fill("#email", LOGIN_MAIL)
    page.fill("#password", LOGIN_PW)
    page.click("button[type=submit]")
    page.wait_for_url("**/dashboard", timeout=20000)
    print("Angemeldet an %s" % BASE)

    # Die Benutzerverwaltung ist Admins vorbehalten. Meldet sich der Test mit
    # einem normalen Konto an, antwortet /api/admin/* korrekt mit 403 — das ist
    # richtiges Verhalten, kein Befund. Also nur pruefen, wenn das Konto Admin ist.
    ist_admin = page.evaluate("""async () => {
        const r = await fetch('/api/me');
        if (!r.ok) return false;
        const d = await r.json();
        return !!(d.is_admin || (d.user && d.user.is_admin));
    }""")
    if not ist_admin:
        SEITEN[:] = [(p, n) for p, n in SEITEN if p != "/benutzer"]
        print("Konto ist kein Admin — Benutzerverwaltung uebersprungen")

    # /app ohne Projekt leitet planmaessig aufs Dashboard. Die Projektansicht ist
    # aber die wichtigste Seite der App — deshalb mit einem echten Projekt pruefen.
    projekte = page.evaluate("""async () => {
        const r = await fetch('/api/projects');
        if (!r.ok) return [];
        const d = await r.json();
        return (Array.isArray(d) ? d : (d.projects || [])).map(p => p.id);
    }""")
    if projekte:
        SEITEN[:] = [("/app?projekt=%d" % projekte[0], n) if p == "/app" else (p, n)
                     for p, n in SEITEN]
        print("Projektansicht wird mit Projekt %d geprueft" % projekte[0])
    else:
        SEITEN[:] = [(p, n) for p, n in SEITEN if p != "/app"]
        print("Kein Projekt vorhanden — Projektansicht uebersprungen")
    print()

    for pfad, name in SEITEN:
        js_fehler.clear()
        netz.clear()
        antwort = page.goto(BASE + pfad, wait_until="networkidle")
        page.wait_for_timeout(1200)          # Zeit fuer Nachladen per fetch

        status = antwort.status if antwort else 0
        ziel = pfad.split("?")[0].rstrip("/")
        umgeleitet = ziel not in page.url
        haupt = page.locator("main")
        text = haupt.inner_text().strip() if haupt.count() else ""
        h1 = page.locator("h1")
        haengt = [w for w in PLATZHALTER if w in text]

        gut = True
        gut &= check(pfad, "HTTP 200", status == 200, "-> %d" % status)
        gut &= check(pfad, "keine Umleitung", not umgeleitet, "-> %s" % page.url)
        gut &= check(pfad, "keine JS-Fehler", not js_fehler, str(js_fehler[:2]))
        gut &= check(pfad, "kein haengender Platzhalter", not haengt,
                     "-> %s" % haengt)
        gut &= check(pfad, "genau eine H1", h1.count() == 1, "-> %d" % h1.count())
        gut &= check(pfad, "Hauptbereich hat Inhalt", len(text) > 40,
                     "-> %d Zeichen" % len(text))
        gut &= check(pfad, "keine Fehlerantworten", not netz, str(netz[:2]))

        print("  %-26s %-24s %s" % (pfad, name, "ok" if gut else "FEHLER"))
        if not gut:
            print("        Textanfang: %r" % text[:120])

    browser.close()

print()
if befunde:
    print("BEFUNDE (%d):" % len(befunde))
    for b in befunde:
        print("  " + b)
print("%d Pruefungen ok, %d fehlgeschlagen" % (ok, fail))
sys.exit(1 if fail else 0)
