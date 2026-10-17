#!/usr/bin/env python3
"""Klickprobe „In ein Projekt laden“ in Meine Ablage (17.10.2026, backend/ablage_projekt.py), mit der Tastatur bedient: Knopf →
beschriftete Auswahl (Fokus darauf) → „Abbrechen“ bringt den Fokus zurück → erneut, „Laden“ mit Enter → das neue PDF-Projekt öffnet
sich mit der Datei als Dokument. Danach wird das Projekt gelöscht — und die Datei in der Ablage ist noch da (das Löschen eines
Projekts entfernt dessen Ursprungsdatei; die Brücke kopiert darum über den Hochladeweg). Nimmt einen vorhandenen PDF-Eintrag des
Kontos (nur lesend), ohne KI. Aufruf (vor jedem Einspielen auf Staging, Playwright): python3 tests/e2e/ablage_bruecke_probe.py
<mail> <passwort>   (Server: INKLUDOCS_BASIS, Vorgabe Staging — nie gegen Production)"""
import os
import re
import sys

from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_BASIS", "https://staging.inkludocs.inklutec.de")   # nie gegen Production
ok = fehler = 0


def check(n, c, info=""):
    global ok, fehler
    ok, fehler = (ok + 1, fehler) if c else (ok, fehler + 1)
    print(("  OK   " if c else "  FEHLT ") + n + ("" if c else " -- " + str(info)[:500]))


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    fehlerliste = []
    pg.on("pageerror", lambda e: fehlerliste.append(str(e)))
    check("Anmeldung", pg.request.post(B + "/api/login", data={"email": sys.argv[1], "password": sys.argv[2]}).ok)
    vorher = pg.request.get(B + "/api/me").json()["user"]["oberflaeche"]
    pid = None
    try:
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": "klassisch"})   # die Ablage selbst, ohne Agentenansicht
        ausgaben = pg.request.get(B + "/api/ausgaben").json().get("ausgaben") or []
        a = next((x for x in ausgaben if x.get("datei_verfuegbar") and x.get("media") == "application/pdf"
                  and x.get("art") in ("pdf", "pdfua")), None)
        check("Ein PDF-Eintrag in der Ablage zum Ausprobieren", bool(a), [x.get("art") for x in ausgaben][:10])
        if a:
            projekte_vorher = {x["id"] for x in pg.request.get(B + "/api/projects").json()["projects"]}
            pg.goto(B + f"/ablage#ausgabe-{a['id']}", wait_until="networkidle")
            pg.wait_for_timeout(1000)
            eintrag = pg.locator(f"#ausgabe-{a['id']}")
            knopf = eintrag.get_by_role("button", name="In ein Projekt laden")
            check("Knopf „In ein Projekt laden“ am Eintrag", knopf.count() == 1)
            knopf.focus()
            pg.keyboard.press("Enter")
            pg.wait_for_timeout(1200)
            auswahl = eintrag.get_by_label("Ziel")
            fokus = pg.evaluate("() => document.activeElement && document.activeElement.id")
            check("Beschriftete Auswahl „Ziel“, Fokus darauf", auswahl.count() == 1 and fokus == f"ziel_{a['id']}", fokus)
            optionen = auswahl.locator("option").all_text_contents()
            check("Erste Wahl „Neues PDF-Projekt“", bool(optionen) and optionen[0] == "Neues PDF-Projekt", optionen[:3])
            eintrag.get_by_role("button", name="Abbrechen").click()
            pg.wait_for_timeout(300)
            check("„Abbrechen“: Fokus zurück auf den Knopf", pg.evaluate("() => document.activeElement && document.activeElement.textContent") == "In ein Projekt laden")
            pg.keyboard.press("Enter")
            pg.wait_for_timeout(1200)
            pg.keyboard.press("Tab")
            check("Mit Tab von der Auswahl zu „Laden“", pg.evaluate("() => document.activeElement && document.activeElement.textContent") == "Laden")
            with pg.expect_navigation(url=re.compile(r"/app\?projekt=\d+"), timeout=90000):
                pg.keyboard.press("Enter")
            pid = int(re.search(r"projekt=(\d+)", pg.url).group(1))
            check("Neues Projekt geöffnet", pid not in projekte_vorher, pid)
            pg.wait_for_timeout(2500)
            h1 = pg.evaluate("() => (document.querySelector('#main h1') || {}).textContent || ''")
            check("Name aus der Herkunft „… (barrierefreie PDF)“", "(barrierefreie PDF)" in h1, h1)
            d = pg.request.get(B + f"/api/projects/{pid}").json()
            proj = d.get("project", d)
            check("PDF-Projekt mit der Datei als Dokument", proj.get("tool") == "pdf" and len(d.get("documents") or []) == 1,
                  (proj.get("tool"), len(d.get("documents") or [])))
            check("Keine JavaScript-Fehler", not fehlerliste, fehlerliste)
    finally:
        if pid:
            check("Projekt wieder gelöscht", pg.request.delete(B + f"/api/projects/{pid}").ok)
            if a:
                check("Die Datei in der Ablage ist noch da", pg.request.get(B + f"/api/ausgaben/{a['id']}/datei").status == 200)
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": vorher})
        print("  Ansicht zurueck auf: " + pg.request.get(B + "/api/me").json()["user"]["oberflaeche"])
    br.close()
print(f"ERGEBNIS: {ok} OK, {fehler} FEHLT")
