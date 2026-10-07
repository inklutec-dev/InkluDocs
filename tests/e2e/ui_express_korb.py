#!/usr/bin/env python3
"""Klicktest EXPRESS-WARENKORB (Zusatz 05.10.2026, Steve) im echten Browser mit axe:
- Knopf „In den Express-Warenkorb“ an jedem PDF-Dokument der Projektansicht „Dokument“ (neben Umbenennen/Löschen):
  sichtbare Bestätigung mit Link „Zum Warenkorb“, Fokus darauf, keine zusätzliche Ansage; zweiter Klick sagt „schon drin“.
- Eintrag „Express-Warenkorb“ in der Hauptnavigation: Zahl als Text, wechselt still, aria-current nur auf
  /express/warenkorb; Modi „immer“, „nur wenn etwas im Warenkorb liegt“, „aus“ und Knopf aus — umgeschaltet über die
  Express-Einstellungen (ohne Codeänderung).
NUR gegen Staging. Ein fiktives Kundenkonto auf .invalid wird im Container angelegt und am Ende gelöscht; die
Express-Einstellungen werden danach wiederhergestellt.
Aufruf auf dem Server: /home/claude/.venv-pw/bin/python ui_express_korb.py [BASIS]
"""
import subprocess
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://staging.inkludocs.inklutec.de"
if "staging" not in BASE and "localhost" not in BASE:
    sys.exit("ABBRUCH: nur gegen Staging.")
KUNDE, KPW = "korb-kunde@express-korb.invalid", "Express-Korb-2026!x"
AXE = urllib.request.urlopen("https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js", timeout=20).read().decode()
ok = fehler = 0


def check(name, bedingung, info=""):
    global ok, fehler
    if bedingung:
        ok += 1
        print("OK   ", name)
    else:
        fehler += 1
        print("FEHLT", name, "—", str(info)[:300])


def axe(pg, name):
    pg.add_script_tag(content=AXE)
    e = pg.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',values:"
                    "['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}})")
    v = e["violations"]
    check(f"axe {name}: 0 Verstöße", not v, [(x["id"], [t for n in x["nodes"] for t in n["target"]][:3]) for x in v])


def im_container(code):
    return subprocess.run(["sudo", "docker", "exec", "-w", "/app", "inkludocs-staging", "python3", "-c", code],
                          capture_output=True, text=True, check=True).stdout.strip()


AUFRAEUMEN = f"""
import shutil
from database import get_user_by_email, delete_user_data
u = get_user_by_email({KUNDE!r})
if u:
    for p in ('/app/data/uploads/%d' % u['id'], '/app/data/results/%d' % u['id']):
        shutil.rmtree(p, ignore_errors=True)
    delete_user_data(u['id'])
"""
ANLEGEN = AUFRAEUMEN + f"""
import fitz, httpx
from database import create_user, get_db
create_user({KUNDE!r}, {KPW!r}, 'Korb Test (fiktiv)')
uid = get_user_by_email({KUNDE!r})['id']
k = httpx.Client(base_url='http://127.0.0.1:8001', timeout=120)
r = k.post('/api/login', json={{'email': {KUNDE!r}, 'password': {KPW!r}}})
k.headers['Cookie'] = 'token=' + r.cookies.get('token')
import time
pid = None
# Runde 7: kein Hochladen ohne Projekt mehr — die Dokumente kommen in ein Projekt.
for name, seiten in (('Bericht fiktiv.pdf', 2), ('Flyer fiktiv.pdf', 1)):
    d = fitz.open()
    for i in range(seiten):
        d.new_page().insert_text((72, 72), 'Fiktives Dokument, Seite %d' % (i + 1))
    r = k.post('/api/upload', files={{'file': (name, d.tobytes(), 'application/pdf')}}, data={{'project_id': str(pid)}} if pid else None)
    r.raise_for_status()
    pid = r.json()['project_id']
    for _ in range(120):
        if k.get('/api/projects/%d/status' % pid).json().get('status') != 'extracting':
            break
        time.sleep(0.5)
print(pid)
"""
SICHERN = """
from database import get_db
c = get_db(); r = c.execute("SELECT value FROM system_kv WHERE key = 'express_einstellungen'").fetchone(); c.close()
print(r[0] if r else '')
"""


def einstellen(**werte):
    """Express-Einstellungen im Container setzen (wie „Einstellungen speichern“ in der Verwaltung)."""
    im_container(f"""
import json
from database import get_db
c = get_db(); r = c.execute("SELECT value FROM system_kv WHERE key = 'express_einstellungen'").fetchone()
e = json.loads(r[0]) if r and r[0] else {{}}
e.update({werte!r})
c.execute("INSERT INTO system_kv (key, value) VALUES ('express_einstellungen', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (json.dumps(e),))
c.commit(); c.close()
""")


vorher = im_container(SICHERN)
pid = int(im_container(ANLEGEN).splitlines()[-1])
try:
    einstellen(korb_knopf=True, korb_navigation="immer")
    with sync_playwright() as p:
        b = p.chromium.launch()
        js_fehler = []
        ctx = b.new_context(locale="de-DE")
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: js_fehler.append(str(e)))
        pg.on("console", lambda m: js_fehler.append(m.text) if m.type == "error" and "status of 4" not in m.text else None)

        def fokus():
            return pg.evaluate("() => document.activeElement && (document.activeElement.id || document.activeElement.textContent.trim().slice(0, 60))")

        def live():
            return pg.evaluate("() => (document.getElementById('liveRegion') || {}).textContent || ''")

        def nav_korb():
            a = pg.locator(".app-nav a[data-express-korb]")
            return a.inner_text().strip() if a.count() else None

        def projekt_oeffnen():
            pg.goto(f"{BASE}/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
            pg.wait_for_timeout(1500)
            pg.evaluate("() => document.querySelectorAll('details.dok-klappe').forEach((d) => { d.open = true; })")
            pg.wait_for_timeout(200)

        pg.goto(f"{BASE}/login", wait_until="domcontentloaded")
        pg.fill("#email", KUNDE)
        pg.fill("#password", KPW)
        pg.click("button[type=submit]")
        pg.wait_for_url("**/dashboard", timeout=20000)
        pg.wait_for_timeout(800)
        check("Navigation: „Express-Warenkorb“ (Modus immer, leer)", nav_korb() == "Express-Warenkorb", nav_korb())

        projekt_oeffnen()
        knoepfe = pg.locator("button[id^=dok_express_]")
        check("Knopf „In den Express-Warenkorb“ an jedem PDF-Dokument", knoepfe.count() == 2, knoepfe.count())
        erster = knoepfe.first
        check("Knopf neben Umbenennen/Löschen, Name nennt das Dokument", erster.locator("xpath=..").locator("button", has_text="Umbenennen").count() == 1
              and "Dokument „" in erster.inner_text(), erster.inner_text())
        axe(pg, "Projekt Dokument mit Knopf")
        did = erster.get_attribute("id").split("_")[-1]
        live_vorher = live()
        erster.click()
        pg.wait_for_timeout(1200)
        m = pg.locator(f"#dok_express_meldung_{did}")
        check("Bestätigung sichtbar, Fokus darauf", m.is_visible() and fokus() == f"dok_express_meldung_{did}", fokus())
        check("Bestätigung nennt Dokument, Warenkorb und Credits", "liegt jetzt im Express-Warenkorb" in m.inner_text()
              and "Im Warenkorb: 1 Dokument" in m.inner_text(), m.inner_text())
        check("Link „Zum Warenkorb“ in der Bestätigung", m.locator("a[href='/express/warenkorb']").inner_text() == "Zum Warenkorb")
        check("Keine zusätzliche Ansage (Live-Region unverändert)", live() == live_vorher, live())
        check("Navigation zählt still mit: „Express-Warenkorb: 1 Dokument“", nav_korb() == "Express-Warenkorb: 1 Dokument", nav_korb())
        erster.click()
        pg.wait_for_timeout(1200)
        check("Zweiter Klick: „liegt schon im Express-Warenkorb“, Fokus auf der Meldung",
              "liegt schon im Express-Warenkorb" in m.inner_text() and fokus() == f"dok_express_meldung_{did}", m.inner_text())
        check("Navigation unverändert 1 Dokument", nav_korb() == "Express-Warenkorb: 1 Dokument", nav_korb())
        axe(pg, "Projekt nach dem Hinzufügen")

        m.locator("a").click()
        pg.wait_for_url("**/express/warenkorb", timeout=10000)
        pg.wait_for_timeout(1800)
        check("Warenkorb-Seite: Titel und Fokus auf „2. Deine Auswahl“", pg.title().startswith("Express-Warenkorb") and fokus() == "h-auswahl",
              (pg.title(), fokus()))
        check("Warenkorb-Seite: H1 passt zum Titel (N5)", pg.locator("h1").inner_text().strip() == "Express-Warenkorb", pg.locator("h1").inner_text())
        check("Warenkorb-Seite: das Dokument ist in der Auswahl", "1 Dokument" in pg.locator("#exSumme").inner_text(), pg.locator("#exSumme").inner_text())
        check("aria-current nur am Eintrag „Express-Warenkorb“", pg.locator(".app-nav a[aria-current=page]").count() == 1
              and pg.locator(".app-nav a[aria-current=page]").get_attribute("data-express-korb") is not None)
        axe(pg, "Warenkorb-Seite")
        pg.locator("#exAuswahl button", has_text="Entfernen").first.click()
        pg.wait_for_timeout(1200)
        check("Entfernen: Navigation still auf „Express-Warenkorb“", nav_korb() == "Express-Warenkorb" and fokus() == "exMeldung", (nav_korb(), fokus()))
        pg.goto(f"{BASE}/express", wait_until="networkidle")
        pg.wait_for_timeout(1000)
        check("Auf /express: aria-current am „Meine Aufträge“, nicht am Warenkorb",
              pg.locator(".app-nav a[aria-current=page]").inner_text().strip() == "Meine Aufträge")

        # Modus „nur wenn etwas im Warenkorb liegt“
        einstellen(korb_navigation="mit_inhalt")
        projekt_oeffnen()
        check("Modus „nur mit Inhalt“: leer -> kein Eintrag", nav_korb() is None, nav_korb())
        pg.locator("button[id^=dok_express_]").first.click()
        pg.wait_for_timeout(1200)
        check("Modus „nur mit Inhalt“: nach dem Hinzufügen erscheint „Express-Warenkorb: 1 Dokument“", nav_korb() == "Express-Warenkorb: 1 Dokument", nav_korb())
        axe(pg, "Projekt, Modus nur mit Inhalt")

        # Modus „aus“ und Knopf aus
        einstellen(korb_navigation="aus", korb_knopf=False)
        projekt_oeffnen()
        check("Modus „aus“: kein Eintrag in der Navigation", nav_korb() is None, nav_korb())
        check("Knopf aus: kein „In den Express-Warenkorb“ am Dokument", pg.locator("button[id^=dok_express_]").count() == 0)
        check("Keine JS-Fehler", not js_fehler, js_fehler)
        b.close()
finally:
    if vorher:
        im_container(f"""
from database import get_db
c = get_db(); c.execute("UPDATE system_kv SET value = ? WHERE key = 'express_einstellungen'", ({vorher!r},)); c.commit(); c.close()
""")
    else:
        im_container("""
from database import get_db
c = get_db(); c.execute("DELETE FROM system_kv WHERE key = 'express_einstellungen'"); c.commit(); c.close()
""")
    im_container(AUFRAEUMEN)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
