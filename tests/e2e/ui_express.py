#!/usr/bin/env python3
"""Klicktest EXPRESS-SERVICE (05.10.2026) im echten Browser mit axe: Kunde (Link im Projekt, Auswahl über Projekt und
„Alle Dokumente“, Leistung, Entfernen, Pflichtfelder mit Fokus, zahlungspflichtig bestellen, Auftragsübersicht,
Startseite) und Verwaltung (Liste, Auftrag, Rückfrage-Dialog, Ergebnis hochladen, Liefern mit Nachfrage), danach der
Download beim Kunden.
Korrekturrunde 05.10.2026: Hochladefeld = Komponente der Projekte (Etikett-Knopf, Fokusring sichtbar, Dateiname in der
Statuszeile, Fehler am Feld) beim Kunden und in der Verwaltung; Entfernen mit sichtbarer Meldung und Fokus; Leistung
entprellt (eine Anfrage); Häkchen mit required und Fehler am Kästchen; keine Ansage beim Laden; Fokusring an .btn;
Rahmen der Eingabefelder; PDF-Nachweis per fetch; Danke-Kasten nicht im Druck; Überschrift im summary. NUR gegen Staging. Ein fiktives Kundenkonto auf .invalid wird im Container angelegt und am Ende
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


def axe_regel(pg, name, regel):
    """Einzelne axe-Regel, auch aus „best-practice“ (z. B. heading-order, Runde 8)."""
    pg.add_script_tag(content=AXE)
    e = pg.evaluate("async (r) => await axe.run(document, {runOnly:{type:'rule',values:[r]}})", regel)
    v = e["violations"]
    check(f"axe {regel} {name}: 0 Verstöße", not v, [(x["id"], [t for n in x["nodes"] for t in n["target"]][:3]) for x in v])


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
import time
pid = None
# Runde 7: kein Hochladen ohne Projekt mehr — die Dokumente kommen wie beim Kunden in ein Projekt.
for name, seiten in (('Jahresbericht fiktiv.pdf', 2), ('Flyer fiktiv.pdf', 1)):
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

        def live(pg):
            return pg.evaluate("() => (document.getElementById('liveRegion') || {}).textContent || ''")

        def ring(pg, sel):
            """Berechneter Fokusring: 3px solid in #c75000."""
            return pg.evaluate("(s) => { const e = document.querySelector(s); const c = getComputedStyle(e); "
                               "return c.outlineStyle + ' ' + c.outlineWidth + ' ' + c.outlineColor; }", sel)

        # ── Kunde ──
        kc = b.new_context(locale="de-DE")
        pg = seite(kc)
        anmelden(pg, KUNDE, KPW)
        pg.wait_for_timeout(800)
        check("Seitenleiste: „Meine Aufträge“ (Runde 7)", pg.locator(".app-nav a[href='/express']").count() == 1
              and pg.locator(".app-nav a[href='/express']").inner_text().strip() == "Meine Aufträge")
        pg.goto(f"{BASE}/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        link = pg.locator("a", has_text="Vom Express-Service bearbeiten lassen")
        check("Projekt: Link „Vom Express-Service bearbeiten lassen“", link.count() == 1)
        link.click()
        pg.wait_for_url("**/express/warenkorb?projekt=*")
        pg.wait_for_timeout(1500)
        check("Link aus dem Projekt führt in den Express-Warenkorb, genau eine H1", pg.locator("h1").count() == 1
              and pg.locator("h1").inner_text() == "Express-Warenkorb", pg.locator("h1").inner_text())
        check("Projekt vorgewählt, alle Dokumente angehakt, Fokus auf Schritt 1", pg.locator("#exProjekt").input_value() == str(pid)
              and pg.locator("#exAlle").is_checked() and pg.locator("input[name=exDok]:checked").count() == 2 and fokus(pg) == "h-schritt1", fokus(pg))
        check("Keine Zahlen-Stepper, keine Tabellen", pg.locator("input[type=number]").count() == 0 and pg.locator("main table").count() == 0)
        check("Lieferung ohne Uhrzeit genannt", "innerhalb von 48 Stunden" in pg.locator("#exIntro").inner_text())
        check("Preis: 50 je Seite plus 100 je Dokument (Runde 7)", pg.locator("#exPreise").inner_text() == "Preis: 50 Credits je Seite plus 100 Credits je Dokument.",
              pg.locator("#exPreise").inner_text())
        check("Keine Ansage beim Laden (Vorwahl ist sichtbar)", live(pg) == "", live(pg))
        check("Legende mit Projektname ohne Anzahl", pg.locator("#exDokLegende").inner_text().startswith("Dokumente im Projekt „")
              and "Dokumente)" not in pg.locator("#exDokLegende").inner_text(), pg.locator("#exDokLegende").inner_text())
        check("Rahmen der Eingabefelder dunkel (#767f8f)", pg.evaluate("() => getComputedStyle(document.getElementById('exName')).borderColor") == "rgb(118, 127, 143)")
        # Runde 7 (Punkt 2): kein Hochladen bei der Auswahl — Hinweis aufs Projekt
        check("Kein Hochladefeld, Hinweis „zuerst in ein Projekt“ mit Link", pg.locator("#exDateiZone").count() == 0
              and pg.locator("input[type=file]").count() == 0 and "zuerst in ein Projekt" in pg.locator("#exProjektHinweis").inner_text()
              and pg.locator("#exProjektHinweis a[href='/projekt-neu']").count() == 1)
        axe(pg, "Express-Warenkorb")
        pg.click("#exHinzu")
        pg.wait_for_timeout(1200)
        check("Auswahl: 2 Dokumente, 3 Seiten, 350 Credits", pg.locator("#exSumme").inner_text() == "Deine Auswahl: 2 Dokumente, 3 Seiten, 350 Credits.",
              pg.locator("#exSumme").inner_text())
        check("Hinzufügen: EINE Ansage (Bestätigung), keine Neuladung-Ansage darüber", live(pg).startswith("Hinzugefügt: 2 Dokumente."), live(pg))
        check("Dokumente als „schon in deiner Auswahl“ gesperrt", pg.locator("input[name=exDok]:disabled").count() == 2)
        check("Auswahl je Dokument mit Credits", "Jahresbericht fiktiv.pdf, 2 Seiten, 200 Credits" in pg.locator("#exAuswahl").inner_text(),
              pg.locator("#exAuswahl").inner_text())
        # Michael Karbe 05.10.2026: nur Aufbereitung — reine Dokumentliste ohne Leistungswahl (Punkt 3), unter „Prüfen und
        # bestellen“ nur Summen (Punkt 4) — seit Runde 7 mit Zusammensetzung —, nirgends „nur PDF“ (Punkt 6)
        check("Auswahl: reine Dokumentliste, keine Leistungswahl", pg.locator("#exAuswahl select").count() == 0)
        zeilen = [z.strip() for z in pg.locator("#exGesamt p").all_inner_texts()]
        check("Prüfen und bestellen: Rechenweg in kurzen Zeilen, nur die Summe fett (Runde 8)", pg.locator("#exAufstellung").count() == 0
              and zeilen == ["2 Dokumente, 3 Seiten", "Seiten: 3 × 50 Credits = 150 Credits", "Grundpreis: 2 × 100 Credits = 200 Credits",
                             "Summe: 350 Credits"]
              and pg.locator("#exGesamt strong").all_inner_texts() == ["Summe: 350 Credits"] and pg.locator("#exGesamt table").count() == 0, zeilen)
        check("Kein „derzeit nur PDF“ auf der Seite", "nur PDF" not in pg.locator("main").inner_text()
              and "Zurzeit" not in pg.locator("main").inner_text())
        pg.locator("#exAuswahl button", has_text="Entfernen").nth(1).click()
        pg.wait_for_timeout(800)
        check("Entfernen: Fokus auf die sichtbare Meldung", fokus(pg) == "exMeldung" and "„Flyer fiktiv.pdf“ entfernt." in pg.locator("#exMeldung").inner_text(),
              (fokus(pg), pg.locator("#exMeldung").inner_text()))
        check("Nach Entfernen: „1 Dokument“ (Einzahl)", pg.locator("#exSumme").inner_text() == "Deine Auswahl: 1 Dokument, 2 Seiten, 200 Credits.",
              pg.locator("#exSumme").inner_text())
        axe(pg, "Express-Warenkorb mit Auswahl")
        pg.fill("#exName", "")
        pg.click("#exBestellen")
        pg.wait_for_timeout(400)
        check("Pflichtfeld: Fehler am Feld, Fokus dorthin", fokus(pg) == "exName" and pg.locator("#exName").get_attribute("aria-invalid") == "true"
              and "Ansprechpartner" in pg.locator("#exNameFehler").inner_text(), fokus(pg))
        pg.fill("#exName", "Kim Muster (fiktiv)")
        pg.click("#exBestellen")
        pg.wait_for_timeout(400)
        # EIN Pflicht-Häkchen (Punkt 5): Fehler am Kästchen, Fokus dorthin, required, Legende nennt Pflicht
        check("Häkchen fehlt: Fehler am Kästchen, Fokus dorthin", fokus(pg) == "exBedingungen"
              and pg.locator("#exBedingungen").get_attribute("aria-invalid") == "true"
              and pg.locator("#exBedingungen").get_attribute("aria-describedby") == "exBedingungenFehler"
              and "Bedingungen akzeptieren" in pg.locator("#exBedingungenFehler").inner_text(), fokus(pg))
        check("Nur ein Häkchen, required, Legende „Zustimmung (Pflicht)“", pg.locator("#exBestellForm input[type=checkbox]").count() == 1
              and pg.locator("#exBedingungen").get_attribute("required") is not None
              and pg.locator("#exBestellForm legend").inner_text().strip() == "Zustimmung (Pflicht)")
        check("Häkchen nicht vorab gesetzt", not pg.locator("#exBedingungen").is_checked())
        pg.check("#exBedingungen")
        pg.wait_for_timeout(200)
        check("Fehler verschwindet beim Ankreuzen (N3)", pg.locator("#exBedingungen").get_attribute("aria-invalid") is None
              and pg.locator("#exBedingungenFehler").inner_text().strip() == "", pg.locator("#exBedingungenFehler").inner_text())
        # Mit der Tastatur auf den Knopf (Tab vom Häkchen): :focus-visible greift wie bei echter Tastaturbedienung.
        pg.evaluate("() => document.getElementById('exBedingungen').focus()")
        pg.keyboard.press("Tab")
        check("Fokusring an „Zahlungspflichtig bestellen“ (3px #c75000)", fokus(pg) == "exBestellen"
              and ring(pg, "#exBestellen").startswith("solid 3px rgb(199, 80, 0)"), (fokus(pg), ring(pg, "#exBestellen")))
        check("Knopf „Zahlungspflichtig bestellen“", pg.locator("#exBestellen").inner_text() == "Zahlungspflichtig bestellen")
        pg.fill("#exHinweise", "Bitte Seite 2 genau prüfen (Test)")
        pg.click("#exBestellen")
        pg.wait_for_url(f"{BASE}/express*", timeout=20000)
        pg.wait_for_timeout(1500)
        # Runde 7: nach dem Bestellen in „Meine Aufträge“, neuer Auftrag aufgeklappt, Danke oben mit Fokus
        karten = pg.locator("section.ex-auftrag-karte")
        aid = int((karten.first.get_attribute("id") or "exa_karte_0").split("_")[-1])
        check("Nach dem Bestellen: „Meine Aufträge“, Danke-Meldung mit Fokus, Adresse ohne ?neu", pg.locator("h1").inner_text() == "Meine Aufträge"
              and fokus(pg) == "exAuftragMeldung" and f"Dein Auftrag {aid} ist eingegangen" in pg.locator("#exAuftragMeldung").inner_text()
              and "neu=" not in pg.url, (fokus(pg), pg.url))
        check("Neuer Auftrag aufgeklappt", karten.count() == 1 and karten.first.locator("details.dok-klappe").first.evaluate("d => d.open"))
        check("Keine Ansage zusätzlich zur Danke-Meldung", live(pg) in ("", "Bestellung wird gesendet …"), live(pg))
        pg.goto(f"{BASE}/express", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        check("Michaels Fall: auch EIN Auftrag ist als Karte zu (aufklappbar erkennbar)", pg.locator("section.ex-auftrag-karte").count() == 1
              and not pg.locator("section.ex-auftrag-karte details.dok-klappe").first.evaluate("d => d.open")
              and pg.locator("section.ex-auftrag-karte details.dok-klappe > summary h2").count() == 1)
        check("Unter der Liste: Weg zum Express-Warenkorb, kein Bestellformular", pg.locator("#exBestellForm").count() == 0
              and pg.locator("main a[href='/express/warenkorb']", has_text="Zum Express-Warenkorb").count() == 1)
        axe(pg, "Meine Aufträge")
        check("Überschriften ohne Sprung: H1 „Meine Aufträge“, Karte H2, „Neuer Express-Auftrag“ H2 (Runde 8)",
              pg.locator("section.ex-auftrag-karte summary h2").count() == 1 and pg.locator("h2#h-neuer-auftrag").count() == 1)
        axe_regel(pg, "Meine Aufträge", "heading-order")
        pg.goto(f"{BASE}/express?auftrag={aid}#exa_karte_{aid}", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        check("Sprung auf einen Auftrag: Karte offen, Fokus auf ihrer Überschrift", pg.locator(f"#exa_karte_{aid} details.dok-klappe").first.evaluate("d => d.open")
              and pg.evaluate("() => document.activeElement && document.activeElement.tagName") == "SUMMARY", fokus(pg))
        pg.goto(f"{BASE}/express#h-auftraege", wait_until="networkidle")
        check("Alter Link /express#h-auftraege trifft die Überschrift", pg.locator("#h-auftraege").inner_text() == "Meine Aufträge")
        pg.goto(f"{BASE}/express/auftrag/{aid}?neu=1", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        check("Alter Link mit ?neu=1: Danke-Meldung in der Übersicht", fokus(pg) == "exaNeuText" and "ist eingegangen" in pg.locator("#exaNeuText").inner_text())
        check("Übersicht: vorgemerkt, keine Rechnung", all(w in pg.locator("main").inner_text() for w in
              ("vorgemerkt, abgebucht wird erst bei der Lieferung", "keine Rechnung")))
        check("Übersicht: Preis je Dokument zusammengesetzt", "200 (2 × 50 Credits je Seite plus 100 Credits je Dokument)" in pg.locator("main").inner_text())
        # Runde 7 (Punkt 1+3): Angaben, Einverständnis, Verlauf aufklappbar — zu, Überschrift im summary; Druck klappt alles auf
        klappen = pg.locator("details.ex-abschnitt-klappe")
        check("Angaben, Einverständnis, Verlauf aufklappbar und zu", klappen.count() == 3
              and [k.locator("summary h2").inner_text() for k in klappen.all()] == ["Angaben", "Einverständnis", "Verlauf"]
              and not any(k.evaluate("d => d.open") for k in klappen.all()))
        check("Druckkopf auf dem Bildschirm unsichtbar", not pg.locator("#exaDruckKopf").is_visible())
        pg.emulate_media(media="print")
        pg.evaluate("() => window.dispatchEvent(new Event('beforeprint'))")
        kopf = pg.locator("#exaDruckKopf").inner_text() if pg.locator("#exaDruckKopf").is_visible() else ""
        check("Druck: Kopfzeile „InkluDocs · Auftragsübersicht“, Konto und Druckdatum (Runde 8)", "InkluDocs · Auftragsübersicht" in kopf
              and f"Konto: {KUNDE}" in kopf and "Gedruckt am " in kopf, kopf)
        pg.emulate_media(media="screen")
        check("Druck: alles aufgeklappt (Einverständnis-Text lesbar)", all(k.evaluate("d => d.open") for k in klappen.all())
              and "Ich akzeptiere die Bedingungen" in pg.locator("main").inner_text() and "Ich bin einverstanden" not in pg.locator("main").inner_text())
        pg.evaluate("() => window.dispatchEvent(new Event('afterprint'))")
        check("Nach dem Druck wieder zu", not any(k.evaluate("d => d.open") for k in klappen.all()))
        check("Drucken und PDF angeboten", pg.locator("#exaDrucken").count() == 1 and pg.locator("#exaPdf").count() == 1)
        pg.emulate_media(media="print")
        check("Druck: Danke-Kasten nicht dabei", not pg.locator("#exaNeu").is_visible())
        pg.emulate_media(media="screen")
        axe(pg, "Auftragsübersicht")
        pg.goto(f"{BASE}/express/auftrag/{aid}", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        check("Auftragsübersicht ohne Ansage beim Laden", live(pg) == "", live(pg))
        pg.goto(f"{BASE}/dashboard", wait_until="networkidle")
        pg.wait_for_timeout(1200)
        check("Startseite: „Meine Aufträge“ mit Sprung auf die Karte und „Alle Aufträge“", pg.locator("#express-h").inner_text() == "Meine Aufträge"
              and f"Auftrag {aid}" in pg.locator("#expressSection").inner_text()
              and pg.locator(f"#expressSection a[href='/express?auftrag={aid}#exa_karte_{aid}']").count() == 1
              and pg.locator("#expressSection a[href='/express']", has_text="Alle Aufträge").count() == 1)
        check("Startseite: vorgemerkte Credits genannt", "Express-Aufträge vorgemerkt: 200" in pg.locator("main").inner_text(), pg.locator("#dailyLimitInfo").inner_text())
        axe(pg, "Startseite mit Express")

        # ── Verwaltung ──
        ac = b.new_context(locale="de-DE")
        ap = seite(ac)
        anmelden(ap, ADMIN_MAIL, ADMIN_PW)
        ap.goto(f"{BASE}/verwaltung/express", wait_until="networkidle")
        ap.wait_for_timeout(1500)
        check("Verwaltung: „Express-Aufträge“ aktuelle Seite", ap.locator(".verwaltung-nav a[aria-current=page]").inner_text().strip() == "Express-Aufträge")
        check("Auftrag in „Neu“", ap.locator(f"a[href='/verwaltung/express/{aid}']").count() == 1)
        check("Preise ohne Platzhalter-Hinweis, Grundpreis je Dokument als eigenes Feld (Runde 7)", ap.locator("#exvPlatzhalter").count() == 0
              and ap.locator("#exvFest").count() == 0 and ap.locator("#exvGrund_aufbereiten").input_value() == "100"
              and ap.locator("label[for=exvGrund_aufbereiten]").inner_text() == "Barrierefrei aufbereiten (mit Prüfung): Grundpreis je Dokument (Credits)")
        check("Verwaltung: keine Ansage beim Laden", live(ap) == "", live(ap))
        check("Geliefert/Storniert: Überschrift im summary", ap.locator("details summary h2").count() == 2)
        check("Preisfelder aus der Liste der Leistungen (nur eingeschaltete)", ap.locator("#exvPreis_aufbereiten").count() == 1 and ap.locator("#exvPreis_pruefen").count() == 0)
        ap.fill("#exvFrist", "bald")
        ap.click("#exvEinstKnopf")
        ap.wait_for_timeout(1000)
        check("Einstellungen: Fehler am Feld „Lieferfrist in Stunden“, Fokus dorthin", fokus(ap) == "exvFrist"
              and ap.locator("#exvFrist").get_attribute("aria-invalid") == "true"
              and ap.locator("#exvFristFehler").inner_text().startswith("Lieferfrist in Stunden"), (fokus(ap), ap.locator("#exvFristFehler").inner_text() if ap.locator("#exvFristFehler").count() else ""))
        ap.reload(wait_until="networkidle")
        ap.wait_for_timeout(1200)
        axe(ap, "Verwaltung Express-Liste")
        ap.goto(f"{BASE}/verwaltung/express/{aid}", wait_until="networkidle")
        ap.wait_for_timeout(1500)
        axe(ap, "Verwaltung Auftrag")
        ap.click("button:has-text('Übernehmen')")
        ap.wait_for_timeout(1000)
        check("Übernommen: Meldung mit Fokus", fokus(ap) == "verwaltungMeldung" and "bearbeitest" in ap.locator("#verwaltungMeldung").inner_text())
        ap.click("button:has-text('Rückfrage stellen')")
        ap.wait_for_timeout(400)
        check("Rückfrage-Dialog: Fokus im Textfeld, Beschreibung, Pflicht", fokus(ap) == "exdFrageText"
              and ap.locator("#exdFrageDialog").get_attribute("aria-describedby") == "exdFrageInfo"
              and ap.locator("#exdFrageText").get_attribute("required") is not None)
        ap.click("#exdFrageForm button[type=submit]")
        ap.wait_for_timeout(300)
        check("Rückfrage leer: Fehler am Feld", fokus(ap) == "exdFrageText" and ap.locator("#exdFrageText").get_attribute("aria-invalid") == "true"
              and "Frage" in ap.locator("#exdFrageTextFehler").inner_text())
        axe(ap, "Rückfrage-Dialog")
        ap.keyboard.press("Escape")
        ap.wait_for_timeout(300)
        zonen = ap.locator(".proj-dropzone.hochladefeld")
        check("Verwaltung: Hochladefelder = Komponente der Projekte (Ergebnis und Prüfbericht)", zonen.count() == 2
              and zonen.first.locator("label.upload-btn").is_visible()
              and zonen.first.locator("label.upload-btn").evaluate("e => e.firstChild.textContent.trim()") == "PDF-Datei auswählen"
              and "Ergebnis für „Jahresbericht fiktiv.pdf“ hochladen" in zonen.first.locator("h4").inner_text(), zonen.count())
        namen = [z.locator("input[type=file]").evaluate("e => e.labels[0].textContent.trim()") for z in zonen.all()]
        check("Hochladeknöpfe eindeutig benannt (versteckter Zusatz, N2), keine eigene Landmarke je Fläche",
              namen == ["PDF-Datei auswählen: Ergebnis für „Jahresbericht fiktiv.pdf“", "PDF-Datei auswählen: Prüfbericht für „Jahresbericht fiktiv.pdf“"]
              and all(z.get_attribute("aria-labelledby") is None for z in zonen.all()), namen)
        feld = zonen.first.locator("input[type=file]")
        fid = feld.get_attribute("id")
        ap.evaluate("(i) => document.getElementById(i).focus()", fid)
        check("Verwaltung: Fokusring am Knopf des Hochladefelds", ring(ap, f"label[for={fid}]").startswith("solid 3px rgb(199, 80, 0)"), ring(ap, f"label[for={fid}]"))
        feld.set_input_files(files=[{"name": "kein.pdf", "mimeType": "application/pdf", "buffer": b"kein pdf"}])
        ap.wait_for_timeout(1500)
        check("Verwaltung: falsche Datei -> Fehler am Feld", feld.get_attribute("aria-invalid") == "true"
              and ap.locator(f"#{fid}Status").inner_text().startswith("Fehler:"), ap.locator(f"#{fid}Status").inner_text())
        feld.set_input_files(ergebnis_pdf)                  # startet das Hochladen sofort (wie in den Projekten)
        ap.wait_for_timeout(6000)
        check("Ergebnis hochgeladen: Meldung nennt Datei und veraPDF", "veraPDF" in ap.locator("#verwaltungMeldung").inner_text()
              and "als Ergebnis für" in ap.locator("#verwaltungMeldung").inner_text(), ap.locator("#verwaltungMeldung").inner_text())
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
        check("Prüfergebnis für Kunden ohne Technik-Zusammenfassung", "Automatische Prüfung (veraPDF):" in pg.locator("main").inner_text()
              and "identifiziert" not in pg.locator("main").inner_text())
        # PDF-Nachweis per fetch: Erfolg als Download, Fehler als Satz neben dem Link (nicht als JSON-Seite)
        pg.route("**/nachweis.pdf", lambda route: route.fulfill(status=503, content_type="application/json",
                                                                 body='{"detail": "Die PDF kann gerade nicht erstellt werden. Bitte die Druckansicht nutzen."}'))
        pg.click("#exaPdf")
        pg.wait_for_timeout(800)
        check("PDF-Fehler: Meldung neben dem Link mit Fokus, Seite bleibt", fokus(pg) == "exaPdfFehler" and "Druckansicht" in pg.locator("#exaPdfFehler").inner_text()
              and pg.url.endswith(f"/express/auftrag/{aid}"), (fokus(pg), pg.url))
        pg.unroute("**/nachweis.pdf")
        js_fehler[:] = [x for x in js_fehler if "status of 503" not in x]     # die 503 oben war gewollt (nachgestellt)
        axe(pg, "Auftragsübersicht geliefert")

        # Ein stornierter Auftrag daneben (Nachkontrolle Runde 3: Abzeichen „Storniert“ hatte 4,34:1) — direkt im Container
        # bestellt und storniert, ohne Mails.
        aid_storno = int(im_container(f"""
import express
from database import get_user_by_email, get_db
uid = get_user_by_email({KUNDE!r})['id']
c = get_db(); did = c.execute("SELECT d.id FROM documents d JOIN projects p ON p.id = d.project_id WHERE p.user_id = ? "
                              "AND d.original_filename = 'Flyer fiktiv.pdf'", (uid,)).fetchone()[0]; c.close()
express.dokumente_hinzufuegen(uid, [did])
w = express.warenkorb(uid)
r = express.bestellen(uid, ansprechpartner='Kim Muster (fiktiv)', telefon='', hinweise='', bedingungen=True,
                      idempotenz='ui-storno-%d' % w['id'], korb_id=w['id'], erwartete_credits=w['credits'], fassung=w['fassung'])
express.stornieren(r['auftrag_id'], {{'id': 0, 'name': 'UI-Test'}}, 'Test: Abzeichen Storniert')
print(r['auftrag_id'])
""").splitlines()[-1])
        # ── Kunde: „Meine Aufträge“ als Karten wie die Dokumente, umbenennen und löschen (Punkte 1 und 2) ──
        pg.goto(f"{BASE}/express", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        karte = pg.locator(f"#exa_karte_{aid}")
        check("Meine Aufträge: Karte mit <details>, H3 im summary, Stand-Abzeichen", karte.count() == 1
              and karte.locator("details.dok-klappe > summary h2").count() == 1 and "Geliefert" in karte.locator(".badge").inner_text())
        karte.locator("details.dok-klappe > summary").first.evaluate("e => { e.parentElement.open = true; }")
        inhalte = karte.locator("details.ex-dok-klappe")
        check("Inhalte des Auftrags: Überschrift H3 unter der Karten-H2", inhalte.locator("summary h3").count() == 1)
        axe_regel(pg, "Meine Aufträge, Karte offen", "heading-order")
        check("Inhalte des Auftrags aufklappbar (Dokumente mit Seiten, Stand, Download)", inhalte.count() == 1
              and "Dokumente dieses Auftrags (1)" in inhalte.locator("summary").inner_text())
        inhalte.evaluate("e => { e.open = true; }")
        check("Dokument in der Karte: Seiten, Stand, Download", all(w in inhalte.inner_text() for w in ("Jahresbericht fiktiv.pdf", "Seiten: 2", "fertig, zum Herunterladen"))
              and inhalte.locator("a", has_text="Barrierefreie PDF herunterladen").count() == 1, inhalte.inner_text())
        knopf = karte.locator("button", has_text="Umbenennen")
        check("Knöpfe Umbenennen und Löschen (geliefert)", knopf.count() == 1 and karte.locator("button", has_text="Löschen").count() == 1)
        abz = pg.locator(f"#exa_karte_{aid_storno} .badge")
        check("Abzeichen „Storniert“ mit dunklerer Schrift (#475569, 6,9:1)", abz.count() == 1 and "Storniert" in abz.inner_text()
              and abz.evaluate("e => getComputedStyle(e).color") == "rgb(71, 85, 105)", abz.count() and abz.evaluate("e => getComputedStyle(e).color"))
        axe(pg, "Meine Aufträge als Karten (mit storniertem Auftrag)")
        knopf.click()
        pg.wait_for_timeout(300)
        check("Umbenennen: Dialog, Fokus im Feld", pg.locator("#exaRenameDialog").evaluate("d => d.open") and fokus(pg) == "exaRenameName")
        axe(pg, "Dialog Umbenennen")
        pg.fill("#exaRenameName", "Jahresberichte (fiktiv)")
        pg.click("#exaRenameForm button[type=submit]")
        pg.wait_for_timeout(1200)
        check("Umbenannt: Meldung mit Fokus, Name in der Karte", fokus(pg) == "exAuftragMeldung"
              and "Jahresberichte (fiktiv)" in pg.locator("#exAuftragMeldung").inner_text()
              and "Jahresberichte (fiktiv)" in pg.locator(f"#exa_karte_{aid} summary").first.inner_text(), fokus(pg))
        pg.locator(f"#exa_karte_{aid} details.dok-klappe").first.evaluate("e => { e.open = true; }")
        pg.locator(f"#exa_karte_{aid} button", has_text="Löschen").click()
        pg.wait_for_timeout(300)
        check("Löschen: Bestätigungsdialog, Fokus auf „Abbrechen“", pg.locator("#exaLoeschDialog").evaluate("d => d.open")
              and fokus(pg) == "exaLoeschAbbrechen" and "Buchhaltung" in pg.locator("#exaLoeschText").inner_text(), fokus(pg))
        axe(pg, "Dialog Löschen")
        pg.click("#exaLoeschJa")
        pg.wait_for_timeout(1500)
        check("Gelöscht: Meldung mit Fokus, Karte weg", fokus(pg) == "exAuftragMeldung" and "ist gelöscht" in pg.locator("#exAuftragMeldung").inner_text()
              and pg.locator(f"#exa_karte_{aid}").count() == 0, fokus(pg))
        ap.goto(f"{BASE}/verwaltung/express/{aid}", wait_until="networkidle")
        ap.wait_for_timeout(1200)
        check("Verwaltung sieht „vom Kunden gelöscht“", "Vom Kunden gelöscht am" in ap.locator("main").inner_text())
        axe(ap, "Verwaltung: vom Kunden gelöschter Auftrag")

        # ── Runde 7 (Punkt 4): Anmeldung über Links mit Rücksprung, Auftrag eines anderen Kontos ──
        # (mit dem stornierten Auftrag — den gelieferten hat die Kundin oben gelöscht)
        za = aid_storno
        fc = b.new_context(locale="de-DE")
        fp = seite(fc)
        fp.goto(f"{BASE}/express/auftrag/{za}", wait_until="networkidle")
        check("Ohne Sitzung: zur Anmeldung mit Rücksprung und Hinweis", f"/login?weiter=%2Fexpress%2Fauftrag%2F{za}" in fp.url
              and fp.locator("#weiterHinweis").is_visible(), fp.url)
        check("Hinweis nennt das Ziel und hängt am E-Mail-Feld (aria-describedby, Runde 8)",
              fp.locator("#weiterHinweis").inner_text() == "Nach der Anmeldung geht es weiter zu deinem Express-Auftrag."
              and fp.locator("#email").get_attribute("aria-describedby") == "weiterHinweis"
              and fokus(fp) == "email", fp.locator("#weiterHinweis").inner_text())
        axe(fp, "Anmeldung mit Rücksprung")
        fp.fill("#email", ADMIN_MAIL)
        fp.fill("#password", ADMIN_PW)
        fp.click("button[type=submit]")
        fp.wait_for_url(f"**/express/auftrag/{za}", timeout=20000)
        fp.wait_for_timeout(1500)
        meld = fp.locator("#exaFremd")
        check("Anderes Konto: Meldung mit der eigenen Adresse, nichts vom Auftrag", meld.count() == 1 and f"({ADMIN_MAIL})" in meld.inner_text()
              and "Melde dich mit dem Konto an, mit dem du bestellt hast." in meld.inner_text()
              and "Kim Muster" not in fp.locator("main").inner_text() and not fp.locator("#exaAktionen").is_visible(),
              meld.inner_text() if meld.count() else fp.locator("main").inner_text())
        check("Fremdes Konto: Fokus auf dem Meldungssatz (tabindex -1), keine Live-Ansage (Runde 8)", fokus(fp) == "exaFremdText"
              and fp.locator("#exaFremdText").get_attribute("tabindex") == "-1" and live(fp) == "", (fokus(fp), live(fp)))
        axe(fp, "Auftrag eines anderen Kontos")
        fp.click("#exaAndersAnmelden")
        fp.wait_for_url("**/login?weiter=*", timeout=20000)
        check("„Abmelden und anders anmelden“: Anmeldung mit Rücksprung", f"weiter=%2Fexpress%2Fauftrag%2F{za}" in fp.url, fp.url)
        fp.fill("#email", KUNDE)
        fp.fill("#password", KPW)
        fp.click("button[type=submit]")
        fp.wait_for_url(f"**/express/auftrag/{za}", timeout=20000)
        fp.wait_for_timeout(1500)
        check("Mit dem Konto der Bestellung: der Auftrag", fp.locator("#exaFremd").count() == 0 and "Überblick" in fp.locator("main").inner_text())
        fp.evaluate("() => fetch('/api/logout', { method: 'POST' })")
        fp.goto(f"{BASE}/login?weiter=//example.com/boese", wait_until="networkidle")
        check("Fremdes Ziel: kein Hinweis auf Rücksprung", fp.locator("#weiterHinweis").count() == 0)
        fp.fill("#email", KUNDE)
        fp.fill("#password", KPW)
        fp.click("button[type=submit]")
        fp.wait_for_timeout(3000)
        check("Kein offener Redirect: nach der Anmeldung bei InkluDocs", fp.url.startswith(BASE) and "example.com" not in fp.url, fp.url)
        fc.close()

        # ── Feld-Fokus auf Anmelden, Registrieren, Passwort vergessen (Nachprüfung Barrierefreiheit, N1) ──
        oc = b.new_context(locale="de-DE")
        op = oc.new_page()
        for pfad in ("/login", "/register", "/forgot"):
            op.goto(f"{BASE}{pfad}", wait_until="networkidle")
            feld = op.locator("main input[type=email], form input[type=email]").first
            if not feld.count():
                check(f"{pfad}: E-Mail-Feld gefunden", False)
                continue
            feld.evaluate("e => e.blur()")
            op.keyboard.press("Tab")                         # Tastatur-Modus
            op.evaluate("() => document.querySelector('form input[type=email]').focus()")
            st = op.evaluate("() => { const c = getComputedStyle(document.activeElement); return c.outlineStyle + ' ' + c.outlineWidth + ' ' + c.outlineColor; }")
            check(f"{pfad}: Textfeld mit 3-px-Fokusring", st.startswith("solid 3px rgb(199, 80, 0)"), st)
            box = op.locator("form input[type=checkbox]").first
            if box.count():
                box.evaluate("e => e.focus()")
                st = op.evaluate("() => { const c = getComputedStyle(document.activeElement); return c.outlineStyle + ' ' + c.outlineWidth; }")
                check(f"{pfad}: Kästchen mit Fokusring", st.startswith("solid 3px"), st)
        oc.close()
        check("Keine JS-Fehler", not js_fehler, js_fehler)
        b.close()
finally:
    im_container(AUFRAEUMEN)
    subprocess.run(["sudo", "rm", "-f", ergebnis_pdf], check=False)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
