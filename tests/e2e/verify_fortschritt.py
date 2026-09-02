#!/usr/bin/env python3
"""Fortschrittsanzeige beim Sammellauf (Michael Karbe, 02.09.2026).

Michaels Befund: Beim ersten Klick auf „Alt-Texte generieren" stand „0 von 8"
(gemeint ist das Bild, das gerade dran ist: 1 von 8). Beim erneuten Lauf fuer
dieselbe PDF stand am Anfang „8 von 8", obwohl wieder Bild 1 bearbeitet wurde.
Word (gleicher Sammellauf) genauso.

Der Test faehrt ZWEI ECHTE Laeufe auf einem kleinen Staging-Projekt (Standard
230, zwei Bilder — kostet Credits des E2E-Kontos) und prueft:
  A. Direkt nach dem Start: /status liefert processed_images = 0 (nicht die
     Zahl des letzten Laufs), die Seite sagt „Bild 1 von n wird verarbeitet."
  B. Waehrend des Laufs steigt die laufende Nummer nie ueber n, und die
     angesagte Nummer ist immer fertige + 1 (gedeckelt).
  C. Nach dem Ende: Projekt done, processed_images = Zahl der wirklich fertigen
     Bilder (Projekt 230 ist der Fehlerpfad-Test: ein Bild fehlt absichtlich,
     also 1 von 2 — deshalb wird nicht n erwartet, sondern der Stand gemerkt).
  D. Zweiter Lauf (Michaels Fall): wieder processed_images = 0 und
     „Bild 1 von n" — nicht „n von n". Danach wird abgebrochen, der Rest
     kehrt auf done zurueck (ki_neu): processed_images = Stand nach Lauf 1.

Aufruf: /home/claude/.venv-pw/bin/python tests/e2e/verify_fortschritt.py [basis] [projekt]
Konto aus ~/.e2e.env (INKLUDOCS_E2E_MAIL/_PW) oder Umgebung.
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
PROJEKT = int(sys.argv[2]) if len(sys.argv) > 2 else 230


def env_datei(pfad=os.path.expanduser("~/.e2e.env")):
    werte = {}
    try:
        for zeile in open(pfad, encoding="utf-8"):
            if "=" in zeile and not zeile.startswith("#"):
                k, v = zeile.strip().split("=", 1)
                werte[k] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return werte


E = env_datei()
MAIL = os.environ.get("INKLUDOCS_E2E_MAIL") or E.get("INKLUDOCS_E2E_MAIL")
PW = os.environ.get("INKLUDOCS_E2E_PW") or E.get("INKLUDOCS_E2E_PW")
if not (MAIL and PW):
    print("Kein E2E-Konto (INKLUDOCS_E2E_MAIL/_PW)")
    sys.exit(2)

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  OK  {name}")
    else:
        fail += 1
        print(f"FEHLT {name} {extra}")


def nummer(text):
    m = re.search(r"Bild (\d+) von (\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page()
    js_fehler = []
    page.on("pageerror", lambda e: js_fehler.append(str(e)))
    page.goto(f"{BASE}/login", wait_until="domcontentloaded")
    page.fill("#email", MAIL)
    page.fill("#password", PW)
    page.click("button[type=submit]")
    page.wait_for_url("**/dashboard", timeout=20000)

    def status():
        return page.evaluate(f"async () => (await fetch('/api/projects/{PROJEKT}/status')).json()")

    def starten():
        return page.evaluate(f"""async () => {{
            const r = await fetch('/api/projects/{PROJEKT}/generate', {{method:'POST',
                headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{modus:'alle'}})}});
            return {{status: r.status, body: await r.json()}};
        }}""")

    def warten_bis_fertig(max_s=600):
        t0 = time.time()
        while time.time() - t0 < max_s:
            s = status()
            if s["status"] != "processing":
                return s
            time.sleep(5)
        return status()

    vorher = status()
    check("Testprojekt ist frei (nicht processing)", vorher["status"] != "processing", vorher)
    n = vorher["total_images"]
    check("Testprojekt hat Bilder", n and n >= 1, n)

    fertig1 = None
    for lauf in (1, 2):
        print(f"== Lauf {lauf} ==")
        r = starten()
        check("Lauf gestartet (200)", r["status"] == 200 and r["body"].get("gestartet"), r)
        s0 = status()
        check("direkt nach dem Start: processed_images = 0 (nicht die Zahl des letzten Laufs)",
              s0["processed_images"] == 0 and s0["status"] == "processing", s0)
        page.goto(f"{BASE}/app?projekt={PROJEKT}", wait_until="networkidle")
        page.wait_for_timeout(1500)
        info = page.locator("#processingInfo")
        text = info.inner_text() if info.count() else ""
        i, g = nummer(text)
        check(f"Anzeige sagt Bild 1 von {n} (nicht 0, nicht {n})", i == 1 and g == n, text)
        check("Balken bei 0 Prozent", page.locator(".progress-bar[aria-valuenow='0']").count() == 1)
        # waehrend des Laufs: angesagte Nummer = fertige + 1, nie > n
        beobachtet = set()
        t0 = time.time()
        while time.time() - t0 < 240:
            s = status()
            if s["status"] != "processing":
                break
            page.wait_for_timeout(2500)
            if page.locator("#processingInfo").count():
                i, g = nummer(page.locator("#processingInfo").inner_text())
                if i is not None:
                    beobachtet.add(i)
                    if i != min(s["processed_images"] + 1, n) and i != min(status()["processed_images"] + 1, n):
                        check("laufende Nummer = fertige + 1", False, f"angesagt {i}, fertig {s['processed_images']}")
            if lauf == 2 and s["processed_images"] >= 1 and n >= 2:
                break
        check("laufende Nummer blieb im Bereich 1..n", beobachtet and max(beobachtet) <= n and min(beobachtet) >= 1, beobachtet)
        if lauf == 1:
            ende = warten_bis_fertig()
            fertig1 = ende["processed_images"]
            check("Lauf 1 beendet: status done, 1..n Bilder fertig",
                  ende["status"] == "done" and 1 <= fertig1 <= n, ende)
        else:
            # Michaels Fall ist geprueft — Rest abbrechen, spart Credits; ki_neu stellt zurueck.
            page.evaluate(f"async () => (await fetch('/api/projects/{PROJEKT}/generate/abbrechen', {{method:'POST'}})).status")
            ende = warten_bis_fertig(120)
            check("Lauf 2 abgebrochen: status done, Zaehler wieder auf dem Stand nach Lauf 1 (ki_neu)",
                  ende["status"] == "done" and ende["processed_images"] == fertig1, ende)
    check("keine JavaScript-Fehler", not js_fehler, str(js_fehler[:1]))
    b.close()

print(f"\nErgebnis: {ok} OK, {fail} FEHLER")
sys.exit(1 if fail else 0)
