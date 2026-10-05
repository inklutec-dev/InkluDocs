#!/usr/bin/env python3
"""Klicktest EXPRESS-SERVICE (05.10.2026) im echten Browser mit axe: Kunde (Link im Projekt, Auswahl über Projekt und
„Alle Dokumente“, Leistung, Entfernen, Pflichtfelder mit Fokus, zahlungspflichtig bestellen, Auftragsübersicht,
Startseite) und Verwaltung (Liste, Auftrag, Rückfrage-Dialog, Ergebnis hochladen, Liefern mit Nachfrage), danach der
Download beim Kunden. NUR gegen Staging. Ein fiktives Kundenkonto auf .invalid wird im Container angelegt und am Ende
mit allen Aufträgen gelöscht; die Verwaltung bedient das E2E-Konto (Admin) aus ~/.e2e.env.
Aufruf auf dem Server: /home/claude/.venv-pw/bin/python ui_express.py [BASIS]
"""
import os
import subprocess
import sys
import tempfile
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://staging.inkludocs.inklutec.de"
if "staging" not in BASE and "localhost" not in BASE:
    sys.exit("ABBRUCH: nur gegen Staging.")


def _e2e(schluessel):
    wert = os.environ.get(schluessel)
    if wert:
        return wert
    for zeile in open(os.path.expanduser("~/.e2e.env"), encoding="utf-8"):
        if zeile.startswith(schluessel + "="):
            return zeile.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return ""


ADMIN_MAIL, ADMIN_PW = _e2e("INKLUDOCS_E2E_MAIL"), _e2e("INKLUDOCS_E2E_PW")
KUNDE, KPW = "ui-kunde@express-ui.invalid", "Express-UI-2026!x"
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
create_user({KUNDE!r}, {KPW!r}, 'Kim Muster (fiktiv)')
uid = get_user_by_email({KUNDE!r})['id']
c = get_db(); c.execute("INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) VALUES (?, 1000, 1000, 'admin', 'UI-Test', NULL)", (uid,)); c.commit(); c.close()
k = httpx.Client(base_url='http://127.0.0.1:8001', timeout=120)
r = k.post('/api/login', json={{'email': {KUNDE!r}, 'password': {KPW!r}}})
k.headers['Cookie'] = 'token=' + r.cookies.get('token')
for name, seiten in (('Jahresbericht fiktiv.pdf', 2), ('Flyer fiktiv.pdf', 1)):
    d = fitz.open()
    for i in range(seiten):
        d.new_page().insert_text((72, 72), 'Fiktives Dokument, Seite %d' % (i + 1))
    k.post('/api/express/warenkorb/hochladen', files={{'file': (name, d.tobytes(), 'application/pdf')}}).raise_for_status()
for p in k.get('/api/express/stand').json()['warenkorb']['positionen']:
    k.delete('/api/express/warenkorb/positionen/%d' % p['id']).raise_for_status()
c = get_db(); pid = c.execute('SELECT id FROM projects WHERE user_id = ?', (uid,)).fetchone()[0]; c.close()
print(pid)
"""

pid = int(im_container(ANLEGEN).splitlines()[-1])
fd, ergebnis_pdf = tempfile.mkstemp(suffix=".pdf")
os.close(fd)
im_container("import fitz; d = fitz.open(); d.new_page().insert_text((72, 72), 'Aufbereitet (fiktiv)'); "
             "open('/tmp/ui_express_ergebnis.pdf', 'wb').write(d.tobytes())")
subprocess.run(["sudo", "docker", "cp", "inkludocs-staging:/tmp/ui_express_ergebnis.pdf", ergebnis_pdf], check=True)
subprocess.run(["sudo", "chmod", "644", ergebnis_pdf], check=True)

try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        js_fehler = []

        def seite(ctx):
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: js_fehler.append(str(e)))
            pg.on("console", lambda m: js_fehler.append(m.text) if m.type == "error" and "status of 4" not in m.text else None)
            return pg

        def anmelden(pg, mail, pw):
            pg.goto(f"{BASE}/login", wait_until="domcontentloaded")
            pg.fill("#email", mail)
            pg.fill("#password", pw)
            pg.click("button[type=submit]")
            pg.wait_for_url("**/dashboard", timeout=20000)

        def fokus(pg):
            return pg.evaluate("() => document.activeElement && (document.activeElement.id || document.activeElement.textContent.trim().slice(0, 60))")

        # ── Kunde ──
        kc = b.new_context(locale="de-DE")
        pg = seite(kc)
        anmelden(pg, KUNDE, KPW)
        pg.wait_for_timeout(800)
        check("Seitenleiste: „Express-Service“", pg.locator(".app-nav a[href='/express']").count() == 1)
        pg.goto(f"{BASE}/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        link = pg.locator("a", has_text="Vom Express-Service bearbeiten lassen")
        check("Projekt: Link „Vom Express-Service bearbeiten lassen“", link.count() == 1)
        link.click()
        pg.wait_for_url("**/express?projekt=*")
        pg.wait_for_timeout(1500)
        check("Genau eine H1 „Express-Service“", pg.locator("h1").count() == 1 and pg.locator("h1").inner_text() == "Express-Service")
        check("Projekt vorgewählt, alle Dokumente angehakt", pg.locator("#exProjekt").input_value() == str(pid)
              and pg.locator("#exAlle").is_checked() and pg.locator("input[name=exDok]:checked").count() == 2)
        check("Keine Zahlen-Stepper, keine Tabellen", pg.locator("input[type=number]").count() == 0 and pg.locator("main table").count() == 0)
        check("Lieferung ohne Uhrzeit genannt", "innerhalb von 48 Stunden" in pg.locator("#exIntro").inner_text())
        axe(pg, "Express-Seite")
        pg.click("#exHinzu")
        pg.wait_for_timeout(1200)
        check("Auswahl: 2 Dokumente, 3 Seiten, 150 Credits", pg.locator("#exSumme").inner_text() == "Deine Auswahl: 2 Dokumente, 3 Seiten, 150 Credits.",
              pg.locator("#exSumme").inner_text())
        check("Dokumente als „schon in deiner Auswahl“ gesperrt", pg.locator("input[name=exDok]:disabled").count() == 2)
        sel = pg.locator("#exAuswahl select").first
        sel.select_option("pruefen")
        pg.wait_for_timeout(800)
        check("Leistung geändert: Summe 100", "100 Credits" in pg.locator("#exSumme").inner_text(), pg.locator("#exSumme").inner_text())
        pg.locator("#exAuswahl button", has_text="Entfernen").nth(1).click()
        pg.wait_for_timeout(800)
        check("Entfernen: Fokus auf „2. Deine Auswahl“", fokus(pg) == "h-auswahl", fokus(pg))
        check("Nach Entfernen: „1 Dokument“ (Einzahl)", pg.locator("#exSumme").inner_text() == "Deine Auswahl: 1 Dokument, 2 Seiten, 50 Credits.",
              pg.locator("#exSumme").inner_text())
        pg.locator("#exAuswahl select").first.select_option("aufbereiten")
        pg.wait_for_timeout(800)
        check("Wieder „aufbereiten“: 100 Credits", pg.locator("#exSumme").inner_text() == "Deine Auswahl: 1 Dokument, 2 Seiten, 100 Credits.",
              pg.locator("#exSumme").inner_text())
        axe(pg, "Express-Seite mit Auswahl")
        pg.fill("#exName", "")
        pg.click("#exBestellen")
        pg.wait_for_timeout(400)
        check("Pflichtfeld: Fehler am Feld, Fokus dorthin", fokus(pg) == "exName" and pg.locator("#exName").get_attribute("aria-invalid") == "true"
              and "Ansprechpartner" in pg.locator("#exNameFehler").inner_text(), fokus(pg))
        pg.fill("#exName", "Kim Muster (fiktiv)")
        pg.click("#exBestellen")
        pg.wait_for_timeout(400)
        check("Häkchen fehlen: Fokus auf das erste", fokus(pg) == "exBedingungen" and "Häkchen" in pg.locator("#exZustimmungFehler").inner_text(), fokus(pg))
        check("Häkchen nicht vorab gesetzt", not pg.locator("#exBedingungen").is_checked() and not pg.locator("#exBearbeitung").is_checked())
        check("Knopf „Zahlungspflichtig bestellen“", pg.locator("#exBestellen").inner_text() == "Zahlungspflichtig bestellen")
        pg.check("#exBedingungen")
        pg.check("#exBearbeitung")
        pg.fill("#exHinweise", "Bitte Seite 2 genau prüfen (Test)")
        pg.click("#exBestellen")
        pg.wait_for_url("**/express/auftrag/*", timeout=20000)
        pg.wait_for_timeout(1200)
        aid = int(pg.url.rstrip("/").split("/")[-1].split("?")[0])
        check("Danke-Meldung mit Fokus", fokus(pg) == "exaNeuText" and "ist eingegangen" in pg.locator("#exaNeuText").inner_text())
        check("Übersicht: vorgemerkt, Einverständnis, Verlauf", all(w in pg.locator("main").inner_text() for w in
              ("vorgemerkt, abgebucht wird erst bei der Lieferung", "Ich bin einverstanden", "Bestellt", "keine Rechnung")))
        check("Drucken und PDF angeboten", pg.locator("#exaDrucken").count() == 1 and pg.locator("#exaPdf").count() == 1)
        axe(pg, "Auftragsübersicht")
        pg.goto(f"{BASE}/dashboard", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        check("Startseite: „Meine Express-Aufträge“", pg.locator("#expressSection").is_visible() and f"Auftrag {aid}" in pg.locator("#expressSection").inner_text())
        check("Startseite: vorgemerkte Credits genannt", "Express-Aufträge vorgemerkt: 100" in pg.locator("main").inner_text(), pg.locator("#dailyLimitInfo").inner_text())
        axe(pg, "Startseite mit Express")

        # ── Verwaltung ──
        ac = b.new_context(locale="de-DE")
        ap = seite(ac)
        anmelden(ap, ADMIN_MAIL, ADMIN_PW)
        ap.goto(f"{BASE}/verwaltung/express", wait_until="networkidle")
        ap.wait_for_timeout(1500)
        check("Verwaltung: „Express-Aufträge“ aktuelle Seite", ap.locator(".verwaltung-nav a[aria-current=page]").inner_text().strip() == "Express-Aufträge")
        check("Auftrag in „Neu“", ap.locator(f"a[href='/verwaltung/express/{aid}']").count() == 1)
        check("Platzhalter-Hinweis bei Preisen", ap.locator("#exvPlatzhalter").is_visible())
        axe(ap, "Verwaltung Express-Liste")
        ap.goto(f"{BASE}/verwaltung/express/{aid}", wait_until="networkidle")
        ap.wait_for_timeout(1500)
        axe(ap, "Verwaltung Auftrag")
        ap.click("button:has-text('Übernehmen')")
        ap.wait_for_timeout(1000)
        check("Übernommen: Meldung mit Fokus", fokus(ap) == "verwaltungMeldung" and "bearbeitest" in ap.locator("#verwaltungMeldung").inner_text())
        ap.click("button:has-text('Rückfrage stellen')")
        ap.wait_for_timeout(400)
        check("Rückfrage-Dialog: Fokus im Textfeld", fokus(ap) == "exdFrageText")
        axe(ap, "Rückfrage-Dialog")
        ap.keyboard.press("Escape")
        ap.wait_for_timeout(300)
        feld = ap.locator("input[type=file]").first
        feld.set_input_files(ergebnis_pdf)
        ap.locator("form:has(input[type=file]) button[type=submit]").first.click()
        ap.wait_for_timeout(5000)
        check("Ergebnis hochgeladen: Meldung nennt veraPDF", "veraPDF" in ap.locator("#verwaltungMeldung").inner_text(), ap.locator("#verwaltungMeldung").inner_text())
        ap.click("button:has-text('Liefern')")
        ap.wait_for_timeout(1500)
        if ap.locator("#exdLieferDialog").evaluate("d => d.open"):
            check("Nachfrage-Dialog bei Abweichungen, Fokus auf „Abbrechen“", fokus(ap) == "exdLieferAbbrechen")
            axe(ap, "Liefern-Nachfrage")
            ap.click("#exdLieferTrotz")
            ap.wait_for_timeout(1500)
        check("Geliefert: Meldung", "Geliefert" in ap.locator("#verwaltungMeldung").inner_text(), ap.locator("#verwaltungMeldung").inner_text())

        # ── Kunde: Download ──
        pg.goto(f"{BASE}/express/auftrag/{aid}", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        dl = pg.locator("a", has_text="Barrierefreie PDF herunterladen")
        check("Kunde: Download-Link mit Dokumentnamen", dl.count() == 1 and "Jahresbericht" in dl.inner_text())
        with pg.expect_download() as info:
            dl.click()
        check("Download kommt an", info.value.suggested_filename.endswith("(barrierefrei).pdf"), info.value.suggested_filename)
        check("Stand „Geliefert“, abgebucht", "Geliefert" in pg.locator("main").inner_text() and "abgebucht" in pg.locator("main").inner_text())
        axe(pg, "Auftragsübersicht geliefert")
        check("Keine JS-Fehler", not js_fehler, js_fehler)
        b.close()
finally:
    im_container(AUFRAEUMEN)
    subprocess.run(["sudo", "rm", "-f", ergebnis_pdf], check=False)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
