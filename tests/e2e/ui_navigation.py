#!/usr/bin/env python3
"""Klicktest NAVIGATION AUFGERÄUMT (Runde 10, 08.10.2026) im echten Browser mit axe:
- Hauptnavigation für Kunden: Startseite, Meine Projekte, Meine Ablage, Meine Aufträge (Express), Meine Vorlagen,
  Über uns und Kontakt, „Konto“ (natives <details>: zu, offen auf Einstellungen/Datensicherheit samt Unterseiten,
  Tastatur Enter/Leertaste, Abmelden darin). Handy 375/320 px ohne Querscrollen.
- „Neues Projekt anlegen“ als Primärknopf auf Startseite und „Meine Projekte“; /projekt-neu mit Weg zurück.
- „Meine Vorlagen“ mit zwei Bereichen und Zahlen; Prompts/Stammdaten mit Ansichtswahl (aria-current) und Weg zurück;
  „Prompts verwalten“ neben der Prompt-Auswahl im Projekt.
- „Über uns und Kontakt“: gemeinsame Seite, H2 „Kontakt“ (id=kontakt), /kontakt → 301, Kopfzeile der öffentlichen
  Seiten; Kontaktformular barrierefrei (Pflicht/freiwillig im Label, required, Fehler am Feld mit aria-invalid +
  aria-describedby und Fokus, Fehler ohne Feld als role=alert, Bestätigung mit Fokus, autocomplete, Fokusring,
  320 px). Das Formular wird hier NIE echt abgeschickt (Antworten des Servers nachgestellt) — es geht keine Mail raus.
NUR gegen Staging. Ein fiktives Kundenkonto auf .invalid wird im Container angelegt und am Ende gelöscht.
Aufruf auf dem Server: /home/claude/.venv-pw/bin/python ui_navigation.py [BASIS]
"""
import subprocess
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://staging.inkludocs.inklutec.de"
if "staging" not in BASE and "localhost" not in BASE:
    sys.exit("ABBRUCH: nur gegen Staging.")

KUNDE, KPW = "nav-kunde@navigation-r10.invalid", "Navigation-R10-2026!x"
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
import fitz, httpx, time
from database import create_user
create_user({KUNDE!r}, {KPW!r}, 'Kim Muster (fiktiv)')
k = httpx.Client(base_url='http://127.0.0.1:8001', timeout=120)
r = k.post('/api/login', json={{'email': {KUNDE!r}, 'password': {KPW!r}}})
k.headers['Cookie'] = 'token=' + r.cookies.get('token')
# Ein Projekt mit einem Bild (dann zeigt die Alt-Text-Ansicht die Prompt-Auswahl), ein Prompt, zwei Stammdaten
d = fitz.open(); s = d.new_page(); s.insert_text((72, 72), 'Fiktiver Jahresbericht')
pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0); pm.set_rect(pm.irect, (200, 80, 0))
s.insert_image(fitz.Rect(72, 100, 312, 260), stream=pm.tobytes('png'))
r = k.post('/api/upload', files={{'file': ('Jahresbericht fiktiv.pdf', d.tobytes(), 'application/pdf')}}); r.raise_for_status()
pid = r.json()['project_id']
for _ in range(120):
    if k.get('/api/projects/%d/status' % pid).json().get('status') != 'extracting':
        break
    time.sleep(0.5)
k.post('/api/prompts', json={{'name': 'Einfache Sprache (fiktiv)', 'prompt_text': 'Formuliere in einfacher Sprache.', 'category': '', 'description': ''}}).raise_for_status()
for b, q in (('Nachname', 'Familienname, wie im Ausweis'), ('Vorname', 'Vorname, wie im Ausweis')):
    k.post('/api/stammdaten', json={{'beschriftung': b, 'feld_art': 'text', 'quickinfo': q, 'sprache': 'de'}}).raise_for_status()
print(pid)
"""

pid = int(im_container(ANLEGEN).splitlines()[-1])

try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        js_fehler = []

        def seite(ctx):
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: js_fehler.append(str(e)))
            # 401 von /api/me gehört zur öffentlichen Hülle; die nachgestellte 429 ist gewollt
            pg.on("console", lambda m: js_fehler.append(m.text) if m.type == "error" and "status of 4" not in m.text else None)
            return pg

        def fokus(pg):
            return pg.evaluate("() => document.activeElement && document.activeElement.id")

        def live(pg):
            return pg.evaluate("() => (document.getElementById('liveRegion') || {}).textContent || ''").strip()

        def quer(pg):
            return pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")

        def kopf(pg):
            return [(a.get_attribute("href"), a.inner_text().strip()) for a in pg.locator("header.start-header nav a").all()]

        def oben(pg):
            """Einträge der ersten Ebene der Hauptnavigation (ohne „Konto“)."""
            return [a.inner_text().strip() for a in pg.locator("#appSidebar .app-nav > li > a").all()]

        def aktuell(pg):
            return [a.inner_text().strip() for a in pg.locator("#appSidebar a[aria-current=page]").all()]

        def konto_offen(pg):
            return pg.locator("#navKonto").evaluate("d => d.open")

        # ── A. Ohne Anmeldung ──
        ctx = b.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
        pg = seite(ctx)
        erwartet_kopf = [("/preise", "Preise"), ("/ueber-uns", "Über uns und Kontakt"), ("/login", "Anmelden"), ("/register", "Kostenlos starten")]
        for pfad in ("/", "/login"):
            pg.goto(f"{BASE}{pfad}", wait_until="networkidle")
            check(f"Kopfzeile {pfad}: Preise, Über uns und Kontakt, Anmelden, Kostenlos starten", kopf(pg) == erwartet_kopf, kopf(pg))
        r = pg.request.get(f"{BASE}/kontakt", max_redirects=0)
        check("/kontakt: 301 auf /ueber-uns#kontakt", r.status == 301 and r.headers.get("location") == "/ueber-uns#kontakt",
              (r.status, r.headers.get("location")))
        r = pg.request.get(f"{BASE}/kontakt?lang=en", max_redirects=0)
        check("/kontakt?lang=en: Sprachwahl geht mit", r.headers.get("location") == "/ueber-uns?lang=en#kontakt", r.headers.get("location"))
        pg.goto(f"{BASE}/kontakt", wait_until="networkidle")
        check("Alter Link /kontakt landet beim Abschnitt „Kontakt“", pg.url.endswith("/ueber-uns#kontakt"), pg.url)
        check("Über uns und Kontakt: Titel und eine H1", pg.locator("h1").count() == 1 and pg.locator("h1").inner_text() == "Über uns und Kontakt"
              and pg.title().startswith("InkluDocs - Über uns und Kontakt"), (pg.locator("h1").all_inner_texts(), pg.title()))
        check("Seitenleiste ohne Anmeldung: Preise, Über uns und Kontakt, Anmelden oder registrieren",
              [a.inner_text().strip() for a in pg.locator("#appSidebar nav a").all()] == ["Preise", "Über uns und Kontakt", "Anmelden oder registrieren"]
              and aktuell(pg) == ["Über uns und Kontakt"], [a.inner_text().strip() for a in pg.locator("#appSidebar nav a").all()])
        ueber = [h.inner_text().strip() for h in pg.locator("main h2, main h3").all()]
        check("Aufbau: Über-uns-Karten (H2), dann H2 „Kontakt“ mit H3 je Karte", pg.locator("h2#kontakt").inner_text() == "Kontakt"
              and ueber[:4] == ["Steve Weidel — InkluTec", "Michael Karbe — Actino Software", "Was uns verbindet", "Kontakt"]
              and "So erreichst du uns" in ueber and "Oder schreib uns direkt hier" in ueber
              and pg.locator("h3#formular-h").count() == 1, ueber)
        check("Sprung „Direkt zu Kontakt und Kontaktformular“ zeigt auf #kontakt",
              pg.locator("a.kontakt-sprung, .kontakt-sprung a").first.get_attribute("href") == "#kontakt")
        axe(pg, "Über uns und Kontakt (ohne Anmeldung)")
        axe_regel(pg, "Über uns und Kontakt", "heading-order")

        # Kontaktformular — Barrierefreiheit
        labels = [pg.locator(f"label[for={i}]").inner_text().strip() for i in ("ko_name", "ko_email", "ko_betreff", "ko_nachricht")]
        check("Labels nennen Pflicht bzw. freiwillig (kein Sternchen)", labels == ["Dein Name (freiwillig)", "Deine E-Mail-Adresse (Pflicht, für die Antwort)",
              "Betreff (freiwillig)", "Deine Nachricht (Pflicht)"] and "*" not in "".join(labels), labels)
        check("required nur an E-Mail und Nachricht", [pg.locator(f"#{i}").get_attribute("required") is not None for i in
              ("ko_name", "ko_email", "ko_betreff", "ko_nachricht")] == [False, True, False, True])
        check("autocomplete name und email", pg.locator("#ko_name").get_attribute("autocomplete") == "name"
              and pg.locator("#ko_email").get_attribute("autocomplete") == "email" and pg.locator("#ko_email").get_attribute("type") == "email")
        check("Fehlersätze am Feld verknüpft (aria-describedby)", pg.locator("#ko_email").get_attribute("aria-describedby") == "ko_emailFehler"
              and pg.locator("#ko_nachricht").get_attribute("aria-describedby") == "ko_nachrichtFehler")
        check("Spam-Schutz ohne CAPTCHA: Honigtopf unsichtbar, nicht fokussierbar", not pg.locator("#ko_firma").is_visible()
              and pg.locator("#ko_firma").get_attribute("tabindex") == "-1" and pg.locator("iframe[src*=captcha], .g-recaptcha, .h-captcha").count() == 0)
        pg.click("#ko_senden")
        pg.wait_for_timeout(300)
        check("Leer abgeschickt: Fokus auf E-Mail, beide Felder aria-invalid mit Satz, keine Ansage, nichts unter dem Knopf",
              fokus(pg) == "ko_email" and pg.locator("#ko_email").get_attribute("aria-invalid") == "true"
              and pg.locator("#ko_nachricht").get_attribute("aria-invalid") == "true"
              and pg.locator("#ko_emailFehler").inner_text() == "Bitte gib deine E-Mail-Adresse an, damit wir antworten können."
              and pg.locator("#ko_nachrichtFehler").inner_text() == "Bitte schreib uns eine Nachricht."
              and live(pg) == "" and pg.locator("#kontaktErr").inner_text().strip() == "", (fokus(pg), live(pg)))
        pg.locator("#ko_email").type("kein-at")
        check("Beim Tippen verschwindet der Fehler am Feld", pg.locator("#ko_email").get_attribute("aria-invalid") is None
              and pg.locator("#ko_emailFehler").inner_text().strip() == "")
        pg.fill("#ko_nachricht", "Testnachricht (fiktiv)")
        pg.click("#ko_senden")
        pg.wait_for_timeout(300)
        check("Ungültige Adresse: Satz mit Beispiel, Fokus auf E-Mail", fokus(pg) == "ko_email"
              and "gültige E-Mail-Adresse" in pg.locator("#ko_emailFehler").inner_text(), pg.locator("#ko_emailFehler").inner_text())
        pg.fill("#ko_email", "kim.muster@beispiel.invalid")
        pg.fill("#ko_nachricht", "")
        pg.click("#ko_senden")
        pg.wait_for_timeout(300)
        check("Nur Nachricht fehlt: Fokus auf die Nachricht", fokus(pg) == "ko_nachricht" and pg.locator("#ko_email").get_attribute("aria-invalid") is None)
        axe(pg, "Kontaktformular mit Fehlern")
        # Fehler ohne Feld (Server: zu viele Anfragen) — nachgestellt, es geht nichts raus
        pg.route("**/api/kontakt", lambda rt: rt.fulfill(status=429, content_type="application/json",
                 body='{"detail": "Zu viele Anfragen von dieser Verbindung. Bitte später erneut versuchen oder direkt an support@inkludocs.de schreiben."}'))
        pg.fill("#ko_nachricht", "Testnachricht (fiktiv)")
        pg.click("#ko_senden")
        pg.wait_for_timeout(600)
        check("Fehler ohne Feld: Satz unter dem Knopf (role=alert), Fokus bleibt am Knopf", "Zu viele Anfragen" in pg.locator("#kontaktErr").inner_text()
              and pg.locator("#kontaktErr").get_attribute("role") == "alert" and fokus(pg) == "ko_senden", fokus(pg))
        pg.unroute("**/api/kontakt")
        pg.route("**/api/kontakt", lambda rt: rt.fulfill(status=200, content_type="application/json", body='{"ok": true}'))
        pg.click("#ko_senden")
        pg.wait_for_timeout(600)
        check("Erfolg: Bestätigung sichtbar mit Fokus, Formular weg, keine zusätzliche Ansage", pg.locator("#schrittFertig").is_visible()
              and not pg.locator("#schrittFormular").is_visible() and fokus(pg) == "kontaktErfolg"
              and pg.locator("#kontaktErfolg").get_attribute("role") is None and live(pg) == "", fokus(pg))
        axe(pg, "Kontaktformular nach dem Senden")
        pg.unroute("**/api/kontakt")
        pg.goto(f"{BASE}/ueber-uns", wait_until="networkidle")     # neu laden (nur der Anker hätte die Bestätigung stehen lassen)
        pg.evaluate("() => document.getElementById('ko_name').focus()")
        pg.keyboard.press("Tab")
        ring = pg.evaluate("() => { const s = getComputedStyle(document.activeElement); return [document.activeElement.id, s.outlineStyle, s.outlineWidth]; }")
        check("Sichtbarer Fokus im Formular (Tastatur): 3 px", ring[0] == "ko_email" and ring[1] == "solid" and ring[2] == "3px", ring)
        for breite in (320, 375):
            pg.set_viewport_size({"width": breite, "height": 800})
            pg.goto(f"{BASE}/ueber-uns", wait_until="networkidle")
            felder = pg.evaluate("() => ['ko_name','ko_email','ko_betreff','ko_nachricht'].map((i) => document.getElementById(i).getBoundingClientRect().right)")
            check(f"{breite} px (≈ 400 % Zoom): kein Querscrollen, Felder im Bild", not quer(pg) and max(felder) <= breite, (quer(pg), felder))
        pg.set_viewport_size({"width": 1280, "height": 900})
        ctx.close()

        # ── B. Angemeldet (fiktives Kundenkonto) ──
        ctx = b.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
        pg = seite(ctx)
        pg.goto(f"{BASE}/login", wait_until="domcontentloaded")
        pg.fill("#email", KUNDE)
        pg.fill("#password", KPW)
        pg.click("button[type=submit]")
        pg.wait_for_url("**/dashboard", timeout=20000)
        pg.wait_for_timeout(1200)
        eintraege = oben(pg)
        soll = ["Startseite", "Meine Projekte", "Meine Ablage"] + (["Meine Aufträge"] if "Meine Aufträge" in eintraege else []) \
            + ["Meine Vorlagen", "Über uns und Kontakt"]
        check("Navigation: Startseite, Meine Projekte, Meine Ablage, (Meine Aufträge), Meine Vorlagen, Über uns und Kontakt",
              eintraege == soll, eintraege)
        check("Nicht mehr in der Navigation: Neues Projekt anlegen, Meine Prompts, Meine Stammdaten, Kontakt, Über uns",
              not set(eintraege) & {"Neues Projekt anlegen", "Meine Prompts", "Meine Stammdaten", "Kontakt", "Über uns", "Einstellungen", "Datensicherheit"})
        check("„Konto“ als letzter Eintrag, natives details, zu",
              pg.locator("#appSidebar .app-nav > li:last-child > details#navKonto > summary").inner_text().strip() == "Konto"
              and not konto_offen(pg) and not pg.locator("#logoutBtn").is_visible())
        unter = [(x.text_content() or "").strip() for x in pg.locator("#navKonto .app-nav-unter > li > *").all()]
        check("Unter „Konto“: Einstellungen, Datensicherheit, Abmelden", unter == ["Einstellungen", "Datensicherheit", "Abmelden"], unter)
        check("Startseite: aktuell nur „Startseite“", aktuell(pg) == ["Startseite"], aktuell(pg))
        live_vorher = live(pg)        # die Startseite sagt beim Laden schon etwas an — „Konto“ darf nichts hinzufügen
        pg.focus("#navKonto > summary")
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(200)
        offen_enter = konto_offen(pg)
        pg.keyboard.press("Tab")
        tab_ziel = pg.evaluate("() => document.activeElement.textContent.trim()")
        pg.focus("#navKonto > summary")
        pg.keyboard.press(" ")
        pg.wait_for_timeout(200)
        check("Tastatur: Enter klappt auf, Tab führt zu „Einstellungen“, Leertaste klappt zu, keine Ansage",
              offen_enter and tab_ziel == "Einstellungen" and not konto_offen(pg) and live(pg) == live_vorher, (offen_enter, tab_ziel, live(pg)))
        ring = pg.evaluate("() => { const s = getComputedStyle(document.activeElement); return [document.activeElement.tagName, s.outlineStyle, s.outlineWidth]; }")
        check("Fokusring an „Konto“ sichtbar (3 px)", ring == ["SUMMARY", "solid", "3px"], ring)
        knopf = pg.locator("main a.btn-primary[href='/projekt-neu']")
        check("Startseite: Primärknopf „Neues Projekt anlegen“ oben", knopf.count() == 1 and knopf.inner_text().strip() == "Neues Projekt anlegen"
              and pg.evaluate("() => !!(document.querySelector(\"main a[href='/projekt-neu']\").compareDocumentPosition(document.getElementById('recent-h')) & Node.DOCUMENT_POSITION_FOLLOWING)")
              and "links" not in pg.locator("main .dash-sub").first.inner_text())
        axe(pg, "Startseite")
        pg.locator("#navKonto > summary").click()
        axe(pg, "Startseite, Konto offen")
        pg.goto(f"{BASE}/projekte", wait_until="networkidle")
        check("Meine Projekte: Primärknopf direkt unter der H1", pg.evaluate("() => { const h = document.querySelector('main h1'); "
              "const n = h.nextElementSibling; return !!n && !!n.querySelector(\"a.btn-primary[href='/projekt-neu']\"); }")
              and aktuell(pg) == ["Meine Projekte"])
        axe(pg, "Meine Projekte")
        pg.click("main a.btn-primary[href='/projekt-neu']")
        pg.wait_for_url("**/projekt-neu")
        check("Neues Projekt anlegen: Weg „Zu meinen Projekten“, in der Navigation „Meine Projekte“ aktuell",
              pg.locator("main .verwaltung-zurueck a[href='/projekte']").inner_text().strip() == "Zu meinen Projekten" and aktuell(pg) == ["Meine Projekte"], aktuell(pg))
        axe(pg, "Neues Projekt anlegen")
        for pfad, wer in (("/einstellungen", "Einstellungen"), ("/abo", "Einstellungen"), ("/konto", "Einstellungen"), ("/datensicherheit", "Datensicherheit")):
            pg.goto(f"{BASE}{pfad}", wait_until="networkidle")
            pg.wait_for_timeout(500)
            check(f"{pfad}: „Konto“ offen, „{wer}“ aktuell", konto_offen(pg) and aktuell(pg) == [wer], (konto_offen(pg), aktuell(pg)))
        pg.goto(f"{BASE}/einstellungen", wait_until="networkidle")
        axe(pg, "Einstellungen (Konto offen)")
        pg.goto(f"{BASE}/vorlagen", wait_until="networkidle")
        pg.wait_for_timeout(800)
        check("Meine Vorlagen: H1, aktuell, zwei Bereiche Prompts/Stammdaten", pg.locator("h1").inner_text() == "Meine Vorlagen"
              and aktuell(pg) == ["Meine Vorlagen"] and pg.locator("main h2").all_inner_texts() == ["Prompts", "Stammdaten"], aktuell(pg))
        check("Meine Vorlagen: Zahlen „1 Prompt gespeichert.“ / „2 Einträge gespeichert.“",
              pg.locator("#vorlagenPromptsZahl").inner_text() == "1 Prompt gespeichert." and pg.locator("#vorlagenStammdatenZahl").inner_text() == "2 Einträge gespeichert.",
              (pg.locator("#vorlagenPromptsZahl").inner_text(), pg.locator("#vorlagenStammdatenZahl").inner_text()))
        check("Meine Vorlagen: Wege „Prompts öffnen“ und „Stammdaten öffnen“", pg.locator("main a[href='/prompts']").inner_text().strip() == "Prompts öffnen"
              and pg.locator("main a[href='/stammdaten']").inner_text().strip() == "Stammdaten öffnen")
        axe(pg, "Meine Vorlagen")
        axe_regel(pg, "Meine Vorlagen", "heading-order")
        pg.click("main a[href='/prompts']")
        pg.wait_for_url("**/prompts")
        pg.wait_for_timeout(800)

        def wahl(pg):
            return [(a.inner_text().strip(), a.get_attribute("aria-current"), "btn-primary" in (a.get_attribute("class") or ""))
                    for a in pg.locator(".vorlagen-wahl a").all()]
        check("Prompts: zwei Schritte von der Navigation, Ansichtswahl „Prompts“ aktuell, Weg zurück, Navigation „Meine Vorlagen“",
              wahl(pg) == [("Prompts", "page", True), ("Stammdaten", None, False)]
              and pg.locator(".vorlagen-wahl ul").get_attribute("aria-labelledby") == "vorlagenWahlTitel"
              and pg.locator("#vorlagenWahlTitel").inner_text().strip() == "Ansicht:"
              and pg.locator("main .verwaltung-zurueck a[href='/vorlagen']").inner_text().strip() == "Zu meinen Vorlagen"
              and aktuell(pg) == ["Meine Vorlagen"], (wahl(pg), aktuell(pg)))
        axe(pg, "Meine Prompts mit Ansichtswahl")
        pg.click(".vorlagen-wahl a[href='/stammdaten']")
        pg.wait_for_url("**/stammdaten")
        pg.wait_for_timeout(800)
        check("Stammdaten: Ansichtswahl „Stammdaten“ aktuell, Navigation „Meine Vorlagen“", wahl(pg) == [("Prompts", None, False), ("Stammdaten", "page", True)]
              and aktuell(pg) == ["Meine Vorlagen"], (wahl(pg), aktuell(pg)))
        axe(pg, "Meine Stammdaten mit Ansichtswahl")
        pg.goto(f"{BASE}/ueber-uns", wait_until="networkidle")
        check("Angemeldet auf Über uns und Kontakt: Eintrag aktuell, kein „Anmelden oder registrieren“", aktuell(pg) == ["Über uns und Kontakt"]
              and pg.locator("#appSidebar a[href='/login']").count() == 0, aktuell(pg))
        axe(pg, "Über uns und Kontakt (angemeldet)")
        pg.goto(f"{BASE}/app?projekt={pid}&ansicht=alttexte", wait_until="networkidle")
        try:
            pg.wait_for_selector("#ownPromptSelect", timeout=20000)
        except Exception:
            pass
        verwalten = pg.locator("a.prompt-verwalten")
        check("Projekt (Alt-Texte): „Prompts verwalten“ neben der Prompt-Auswahl", verwalten.count() == 1 and verwalten.inner_text().strip() == "Prompts verwalten"
              and verwalten.get_attribute("href") == "/prompts" and pg.locator("#ownPromptSelect").count() == 1)

        # Handy
        for breite in (375, 320):
            pg.set_viewport_size({"width": breite, "height": 800})
            pg.goto(f"{BASE}/dashboard", wait_until="networkidle")
            pg.wait_for_timeout(800)
            zu = not quer(pg) and pg.locator("#navKonto > summary").is_visible() and not pg.locator("#logoutBtn").is_visible()
            pg.locator("#navKonto > summary").click()
            pg.wait_for_timeout(200)
            check(f"Handy {breite} px: Navigation ohne Querscrollen, „Konto“ zu und aufklappbar, Abmelden sichtbar",
                  zu and not quer(pg) and pg.locator("#logoutBtn").is_visible()
                  and pg.locator("#navKonto a[href='/einstellungen']").is_visible(), (zu, quer(pg)))
            box = pg.locator("#logoutBtn").bounding_box()
            check(f"Handy {breite} px: Zielgröße Abmelden ≥ 24 px", box and box["height"] >= 24 and box["width"] >= 24, box)
        axe(pg, "Startseite 320 px, Konto offen")
        pg.set_viewport_size({"width": 1280, "height": 900})
        pg.goto(f"{BASE}/dashboard", wait_until="networkidle")
        pg.locator("#navKonto > summary").click()
        pg.click("#logoutBtn")
        pg.wait_for_url(f"{BASE}/", timeout=20000)
        check("Abmelden unter „Konto“ führt auf die Startseite", pg.url.rstrip("/") == BASE.rstrip("/"), pg.url)
        check("Keine JS-Fehler", not js_fehler, js_fehler)
        b.close()
finally:
    im_container(AUFRAEUMEN)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
