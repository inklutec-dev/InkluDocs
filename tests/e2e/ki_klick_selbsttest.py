#!/usr/bin/env python3
"""Selbsttest fuer tests/e2e/ki_klick.py (06.10.2026) — ohne Staging, ohne KI, ohne Kosten.
Eine kleine Seite bildet den KI-Knopf von formular.js nach (fetch POST, Ergebnis ins Feld, im catch
„Verbindungsfehler.“). Per Playwright-Route werden die Faelle erzeugt, die der Helfer unterscheiden muss:
  1. Antwort nach 1,5 s                      -> HTTP 200, ein Versuch, Feld gefuellt
  2. erste Anfrage bricht im Netz ab        -> Ursache protokolliert, EIN neuer Klick, dann HTTP 200
  3. Server antwortet 500                    -> HTTP 500, KEIN neuer Klick (ein Serverfehler bleibt ein Befund)
  4. Antwort dauert laenger als die Grenze   -> „keine Antwort binnen …“ (Latenz)
  5. Klick loest keine Anfrage aus           -> „keine Anfrage“ (dann laege es an der Seite)
  6. Abbruch, die Seite holt das Ergebnis selbst ab (wie app.html seit 06.10.2026) -> KEIN zweiter Klick
Aufruf: python tests/e2e/ki_klick_selbsttest.py   (Python mit Playwright; laeuft auch in formular_tests.sh)"""
import http.server
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ki_klick import ki_klick, warten_bis  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

SEITE = b"""<!doctype html><html lang="de"><title>KI-Knopf</title>
<textarea id="ta"></textarea><p id="msg"></p>
<button type="button" id="gen" onclick="gen()">Generieren</button>
<button type="button" id="tot">Ohne Anfrage</button>
<button type="button" id="gen2" onclick="gen2()">Generieren mit Abholung</button>
<script>
async function gen() {
  const b = document.getElementById('gen'); b.disabled = true;
  try {
    const res = await fetch('/api/felder/1/generieren', { method: 'POST', body: '{}' });
    const d = await res.json().catch(() => ({}));
    if (!res.ok) { document.getElementById('msg').textContent = d.detail || 'Fehler'; return; }
    document.getElementById('ta').value = d.text; document.getElementById('msg').textContent = 'fertig';
  } catch (e) { document.getElementById('msg').textContent = 'Verbindungsfehler.'; }
  finally { b.disabled = false; }
}
async function gen2() {
  const b = document.getElementById('gen2'); b.disabled = true;
  try {
    let res;
    try { res = await fetch('/api/felder/2/generieren', { method: 'POST', body: '{}' }); }
    catch (e) { await new Promise(r => setTimeout(r, 1000)); res = await fetch('/abholen'); }
    const d = await res.json();
    document.getElementById('ta').value = d.text;
  } finally { b.disabled = false; }
}
</script></html>"""


MODUS = {"art": "ok", "verzoegerung": 0.0, "n": 0}   # Verhalten von POST /api/felder/1/generieren


class Seite(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        MODUS["n"] += 1
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if MODUS["art"] == "fehler500":
            self.send_response(500); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"detail": "Serverfehler (Test)"}'); return
        time.sleep(MODUS["verzoegerung"])
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(('{"text": "KI-Text %d"}' % MODUS["n"]).encode())

    def do_GET(self):
        if self.path.startswith("/abholen"):
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"text": "abgeholt"}'); return
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(SEITE)

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Seite)
threading.Thread(target=srv.serve_forever, daemon=True).start()
B = f"http://127.0.0.1:{srv.server_address[1]}/"
ok = fehler = 0


def check(n, c, i=""):
    global ok, fehler
    if c: ok += 1; print("  OK ", n)
    else: fehler += 1; print("  FEHLT", n, "--", i)


with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_context().new_page()
    def fall(art, verzoegerung=0.0):
        """art = ok | abbruch_dann_ok | fehler500. Der Server zaehlt die POSTs, die ihn erreichen (MODUS["n"]);
        beim Abbruch-Fall kappt eine Route die ERSTE Anfrage im Browser mit einem Verbindungsabbruch."""
        MODUS.update(art=art, verzoegerung=verzoegerung, n=0)
        pg.unroute("**/api/felder/*/generieren")
        if art == "abbruch_dann_ok":
            gekappt = {"n": 0}

            def kappen(route):
                gekappt["n"] += 1
                if gekappt["n"] == 1:
                    route.abort("connectionreset")
                else:
                    route.continue_()
            pg.route("**/api/felder/*/generieren", kappen)

    pg.goto(B)
    print("== 1. Antwort nach 1,5 s ==")
    fall("ok", 1.5); pg.evaluate("document.getElementById('ta').value=''")
    e = ki_klick(pg, pg.locator("#gen"), "/generieren", "Fall 1", grenze_s=20)
    check("HTTP 200 im ersten Versuch", e["status"] == 200 and e["versuche"] == 1, e)
    check("Gewartet wurde auf die Antwort (1,5 s, nicht auf eine feste Zeit)", 1.3 <= e["sekunden"] < 4, e["sekunden"])
    check("Feld gefuellt", warten_bis(pg, lambda: pg.locator("#ta").input_value() == "KI-Text 1", 5), pg.locator("#ta").input_value())
    print("== 2. Netzabbruch, dann Erfolg ==")
    fall("abbruch_dann_ok", 1.0); pg.evaluate("document.getElementById('ta').value=''")
    e = ki_klick(pg, pg.locator("#gen"), "/generieren", "Fall 2", grenze_s=20)
    check("Neuer Klick nach Transportabbruch, dann HTTP 200", e["status"] == 200 and e["versuche"] == 2 and MODUS["n"] == 1, (e, MODUS))
    check("Feld zeigt das Ergebnis des zweiten Klicks (der erste erreichte den Server nie)", warten_bis(pg, lambda: pg.locator("#ta").input_value() == "KI-Text 1", 5), pg.locator("#ta").input_value())
    print("== 3. Serverfehler 500 ==")
    fall("fehler500")
    e = ki_klick(pg, pg.locator("#gen"), "/generieren", "Fall 3", grenze_s=20)
    check("HTTP 500 gemeldet, KEIN neuer Klick", e["status"] == 500 and e["versuche"] == 1 and MODUS["n"] == 1, (e, MODUS))
    print("== 4. Latenz ueber der Grenze ==")
    fall("ok", 4.0)
    e = ki_klick(pg, pg.locator("#gen"), "/generieren", "Fall 4", grenze_s=2)
    check("Keine Antwort binnen Grenze erkannt", e["status"] is None and "keine Antwort binnen 2 s" in e["kurz"], e)
    pg.wait_for_timeout(3000)   # die langsame Antwort ablaufen lassen
    check("Seite zeigt danach das Ergebnis (der Server war nur langsam)", warten_bis(pg, lambda: pg.locator("#msg").inner_text() == "fertig", 5), pg.locator("#msg").inner_text())
    print("== 5. Klick ohne Anfrage ==")
    e = ki_klick(pg, pg.locator("#tot"), "/generieren", "Fall 5", grenze_s=20)
    check("Kein Request erkannt (Klick kam nicht an)", e["status"] is None and "keine Anfrage" in e["kurz"], e)
    print("== 6. Abbruch, Seite holt das Ergebnis selbst ab ==")
    MODUS.update(art="ok", verzoegerung=0.0, n=0)
    pg.route("**/api/felder/2/generieren", lambda route: route.abort("connectionreset"))
    pg.evaluate("document.getElementById('ta').value=''")
    e = ki_klick(pg, pg.locator("#gen2"), "/api/felder/2/generieren", "Fall 6", grenze_s=20,
                 ergebnis_da=lambda: pg.locator("#ta").input_value() == "abgeholt")
    check("Abgeholt erkannt, KEIN zweiter Klick (ein POST)", e["abgeholt"] and e["versuche"] == 1 and e["posts"] == 1, e)
    br.close()
srv.shutdown()
print(f"\nErgebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
