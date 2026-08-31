#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preis-Klicktest fuer die Buchungsstrecke auf /abo (31.08.2026).

Anlass: In der Plan-Liste stand fest "Single — 9,95 € / Monat". Diese 9,95 €
gelten aber nur fuer die festen Laufzeiten; beim Monatsabo sind es 11,95 €.
Auf der Bezahlseite standen damit zwei verschiedene Preise fuer dieselbe Sache.
Seitdem nennt die Plan-Liste keinen Preis mehr, und der Preis steht genau
einmal in der Zusammenfassung.

Geprueft wird deshalb:
  A. Die Plan-Liste nennt KEINEN Preis (kein Euro-Zeichen, keine Preiszahl).
  B. Sie nennt Plan, Credits und Plaetze — und zwar die Werte aus billing.py.
  C. Fuer JEDE Kombination aus Plan und Laufzeit stimmt die Zusammenfassung:
     Gesamtpreis = Monatsrate x Monate, und die Monatsrate wird genannt.
     Monatsabo nimmt die hoehere Rate, feste Laufzeiten die niedrigere.
  D. Keine JavaScript-Fehler.

Aufruf: /home/claude/.venv-pw/bin/python ui_abo_preise.py [BASIS]
Meldet sich mit einem FREE-Konto an — nur dann erscheint der Buchungsbereich.
"""
import re
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
LOGIN_MAIL = "steve.weidel@gmail.com"
LOGIN_PW = "Ewigwind-2026"

# Sollwerte = billing.py. Weichen sie ab, ist entweder billing geaendert
# worden (dann hier nachziehen) oder die Seite tippt wieder eigene Werte.
PREIS_LAUFZEIT = {"single": 9.95, "team": 19.95, "enterprise": 49.95}
PREIS_MONATLICH = {"single": 11.95, "team": 23.95, "enterprise": 59.95}
CREDITS = {"single": 250, "team": 500, "enterprise": 1375}
SITZE = {"team": 5, "enterprise": 25}
LAUFZEITEN = [1, 3, 6, 12]

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


def euro(wert):
    return ("%.2f" % wert).replace(".", ",")


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page()
    page.on("pageerror", lambda e: js_fehler.append(str(e)))

    page.goto(BASE + "/", wait_until="domcontentloaded")
    page.fill("#email", LOGIN_MAIL)
    page.fill("#password", LOGIN_PW)
    page.click("button[type=submit]")
    page.wait_for_url("**/dashboard", timeout=20000)
    page.goto(BASE + "/abo", wait_until="networkidle")
    page.wait_for_timeout(1500)

    print("== A. Plan-Liste nennt keinen Preis ==")
    optionen = {o.get_attribute("value"): o.inner_text().strip()
                for o in page.locator("#buchenPlan option").all()}
    for plan, text in optionen.items():
        check("%-10s ohne Preis: %r" % (plan, text),
              "€" not in text and not re.search(r"\d+,\d{2}", text))

    print("== B. Plan-Liste nennt die Werte aus billing.py ==")
    for plan, text in optionen.items():
        check("%-10s nennt den Plannamen" % plan, plan.capitalize() in text, text)
        # 1375 darf lokalisiert als 1.375 stehen
        zahl = str(CREDITS[plan])
        lokal = "{:,}".format(CREDITS[plan]).replace(",", ".")
        check("%-10s nennt %s Credits" % (plan, zahl),
              zahl in text.replace(".", "") or lokal in text, text)
        if plan in SITZE:
            check("%-10s nennt %d Plaetze" % (plan, SITZE[plan]),
                  str(SITZE[plan]) in text, text)
        else:
            check("%-10s nennt ein Konto" % plan,
                  "Konto" in text or "account" in text.lower(), text)

    print("== C. Preis stimmt fuer jede Kombination ==")
    for plan in ("single", "team", "enterprise"):
        page.select_option("#buchenPlan", plan)
        for monate in LAUFZEITEN:
            page.select_option("#buchenLaufzeit", str(monate))
            page.wait_for_timeout(120)
            summe_text = page.locator("#buchenSumme").inner_text()
            rate = PREIS_MONATLICH[plan] if monate == 1 else PREIS_LAUFZEIT[plan]
            gesamt = euro(rate * monate)
            check("%-10s %2d Monate: Gesamtpreis %s €" % (plan, monate, gesamt),
                  gesamt in summe_text, "-> %r" % summe_text)
            if monate == 1:
                check("%-10s  1 Monat: als monatlich kuendbar benannt" % plan,
                      "kündbar" in summe_text or "cancel" in summe_text.lower(),
                      "-> %r" % summe_text)
            else:
                check("%-10s %2d Monate: Monatsrate %s € genannt"
                      % (plan, monate, euro(rate)),
                      euro(rate) in summe_text, "-> %r" % summe_text)

    print("== D. Der teure Monatspreis taucht nur beim Monatsabo auf ==")
    page.select_option("#buchenPlan", "single")
    page.select_option("#buchenLaufzeit", "6")
    page.wait_for_timeout(150)
    text6 = page.locator("#buchenSumme").inner_text()
    check("6 Monate nennen NICHT den Monatsabo-Preis 11,95 €",
          "11,95" not in text6, "-> %r" % text6)
    page.select_option("#buchenLaufzeit", "1")
    page.wait_for_timeout(150)
    text1 = page.locator("#buchenSumme").inner_text()
    check("Monatsabo nennt 11,95 €", "11,95" in text1, "-> %r" % text1)

    check("keine JavaScript-Fehler", not js_fehler, str(js_fehler[:2]))
    b.close()

print()
print("%d/%d Pruefungen bestanden" % (ok, ok + fail))
sys.exit(1 if fail else 0)
