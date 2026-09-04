#!/usr/bin/env python3
"""Klicktest + axe fuer STARTSEITE und STARTGERUEST (base_start.html,
02.09.2026). Doku: docs/STARTSEITE.md

Aufruf: /home/claude/.venv-pw/bin/python tests/e2e/ui_start.py [BASE]
  BASE Standard http://localhost:8002 (Staging-Container). Teil D meldet sich
  mit dem E2E-Konto aus ~/.e2e.env an (INKLUDOCS_E2E_MAIL/_PW), aendert nichts.

Prueft:
  A. Startseite / anonym: Skip-Link, Kopfzeile (Marke -> /, Navigation Preise/
     Kontakt/Ueber uns/Anmelden/Kostenlos starten), genau eine H1, neun
     Abschnitte mit H2, Karten als H3, sechs FAQ als <details>, Knoepfe Demo +
     Konto, keine Seitenleiste, kein Rest der Login-Karte, Fusszeile = 7 Links
     wie im oeffentlichen Geruest, Meta-Beschreibung/canonical/OG/JSON-LD,
     noindex auf Staging, html lang + englische Fassung per Accept-Language,
     keine JS-Fehler, axe 0 Verstoesse
  B. Login-Karten /login, /register, /forgot, /reset: Karte, H1 = Thema,
     Kopfzeile mit aria-current, Fusszeile, noindex, Links auf /login, axe 0
  C. Weiterleitungen: /app, /dashboard, /projekte ohne Login -> /login;
     /login?geloescht=1 zeigt die fokussierte Bestaetigung
  D. eingeloggt: / -> /dashboard; nach Abmelden landet man auf /
  E. robots.txt + sitemap.xml
  F. schmaler Bildschirm (375 px): kein horizontales Scrollen, Navigation da
"""
import json
import os
import sys
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"


def env_datei(pfad=os.path.expanduser("~/.e2e.env")):
    werte = {}
    try:
        for zeile in open(pfad, encoding="utf-8"):
            zeile = zeile.strip()
            if "=" in zeile and not zeile.startswith("#"):
                k, v = zeile.split("=", 1)
                werte[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return werte


E = env_datei()
LOGIN_MAIL = os.environ.get("INKLUDOCS_E2E_MAIL") or E.get("INKLUDOCS_E2E_MAIL")
LOGIN_PW = os.environ.get("INKLUDOCS_E2E_PW") or E.get("INKLUDOCS_E2E_PW")

H1_DE = "Alt-Texte per KI für barrierefreie PDF-, Word- und Formulardokumente"
H1_EN_TEILE = ("One document. Many people. Equal opportunities.", "AI alt text for accessible PDF, Word and forms")
KOPF_NAV = [("/preise", "Preise"), ("/kontakt", "Kontakt"), ("/ueber-uns", "Über uns"),
            ("/login", "Anmelden"), ("/register", "Kostenlos starten")]
FUSSZEILE = [("/impressum", "Impressum"), ("/datenschutz", "Datenschutz"),
             ("/nutzungsbedingungen", "Nutzungsbedingungen"), ("/preise", "Preise"),
             ("/widerruf", "Widerrufsbelehrung"), ("/kuendigen", "Vertrag kündigen"),
             ("/widerrufen", "Vertrag widerrufen")]
KARTEN = [("/login", "Anmeldung"), ("/register", "Kostenloses Konto erstellen"),
          ("/forgot", "Passwort zurücksetzen"), ("/reset", "Neues Passwort setzen")]

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  OK  {name}")
    else:
        fail += 1
        print(f"FEHLT {name} {extra}")


def axe_lauf(page, name):
    page.add_script_tag(content=axe_js)
    e = page.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',values:"
                      "['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}})")
    v = e["violations"]
    check(f"axe 0 Verstoesse: {name}", len(v) == 0)
    for x in v:
        ziele = [t for k in x["nodes"] for t in k["target"]]
        print(f"   - [{x['impact']}] {x['id']}: {x['help']} -> {ziele[:3]}")


def roh(pfad, headers=None):
    req = urllib.request.Request(f"{BASE}{pfad}", headers=headers or {})
    return urllib.request.urlopen(req, timeout=20).read().decode()


class KeinRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def status_und_ziel(pfad):
    opener = urllib.request.build_opener(KeinRedirect)
    try:
        r = opener.open(f"{BASE}{pfad}", timeout=20)
        return r.status, r.headers.get("location")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("location")


def kopf_nav(page):
    return [(a.get_attribute("href"), a.inner_text().strip())
            for a in page.locator("header.start-header nav a").all()]


def fusszeile(page):
    return [(a.get_attribute("href"), a.inner_text().strip())
            for a in page.locator("footer.start-footer .dash-legal-links a").all()]


axe_js = urllib.request.urlopen(AXE, timeout=20).read().decode()
staging = "staging" in BASE or "8002" in BASE

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(locale="de-DE")
    page = ctx.new_page()
    js_fehler = []
    page.on("pageerror", lambda err: js_fehler.append(str(err)))

    print("== A. Startseite / ohne Anmeldung ==")
    page.goto(f"{BASE}/", wait_until="networkidle")
    check("keine JavaScript-Fehler", not js_fehler, str(js_fehler[:1]))
    check("html lang=de", page.locator("html").get_attribute("lang") == "de")
    check("Fenstertitel nennt InkluDocs und Alt-Texte",
          "InkluDocs" in page.title() and "Alt-Texte" in page.title(), page.title())
    check("Skip-Link auf #main als erstes Element",
          page.locator("body > a.dash-skip[href='#main']").count() == 1)
    check("main#main vorhanden", page.locator("main#main").count() == 1)
    check("Landmarken: header, nav mit Beschriftung, footer",
          page.locator("header.start-header").count() == 1
          and page.locator("header.start-header nav").count() == 1
          and page.locator("footer.start-footer").count() == 1)
    check("Marken-Link zeigt auf /", page.locator("header a.start-brand[href='/']").count() == 1)
    check("Kopfzeile: Preise, Kontakt, Ueber uns, Anmelden, Kostenlos starten", kopf_nav(page) == KOPF_NAV, str(kopf_nav(page)))
    check("kein aria-current auf der Startseite", page.locator("header nav a[aria-current]").count() == 0)
    h1s = page.locator("h1")
    check("genau eine H1 mit dem Versprechen",
          h1s.count() == 1 and " ".join(h1s.first.inner_text().split()) == H1_DE,
          str([x.inner_text().strip() for x in h1s.all()]))
    h2 = [x.inner_text().strip() for x in page.locator("main h2").all()]
    check("zehn Abschnitte, neun mit H2", len(h2) == 9 and page.locator("main section").count() == 10, str(h2))
    check("zehn Abschnitte mit id, KEIN aria-labelledby (Steve 03.09.: ARIA nur wo noetig)",
          page.locator("main section[id]").count() == 10 and page.locator("main section[aria-labelledby]").count() == 0)
    check("nur vier Landmarken: header, nav, main, footer — kein role=region, kein aria-label an nav/Marke",
          page.locator("[role=region], section[aria-label], section[aria-labelledby], nav[aria-label], a.start-brand[aria-label]").count() == 0)
    check("Werkzeug-Karten als H3", page.locator("section#werkzeuge h3").count() == 3)
    check("drei Schritte als nummerierte Liste", page.locator("ol.start-schritte > li").count() == 3)
    check("Abschnitt 'letztes Wort': von Hand aendern + InkluAgent + Prompts (Steve 02.09.)",
          page.locator("section#kontrolle h3").count() == 3
          and "InkluAgent" in page.locator("section#kontrolle").inner_text())
    check("Vorher/Nachher vorhanden", page.locator(".start-vergleich .vorher").count() == 1
          and page.locator(".start-vergleich .nachher blockquote").count() == 1)
    check("vier Zielgruppen + Satz fuer alle anderen",
          page.locator("ul.start-zielgruppen > li").count() == 4
          and "Privatpersonen" in page.locator("section#zielgruppen").inner_text())
    check("Preise mit Link auf /preise",
          page.locator("section#preise a[href='/preise']").count() == 1)
    check("sieben FAQ als natives details/summary",
          page.locator(".start-faq details > summary").count() == 7)
    check("Knopf Demo (ohne Anmeldung) x2, Knopf Konto x2",
          page.locator("main a.btn-start[href='https://demo.inkludocs.de']").count() == 2
          and page.locator("main a.btn-start[href='/register']").count() == 2)
    check("Hero: dritter Knopf 'Anmelden' nach /login (Michael 03.09.)",
          page.locator("section.start-hero a.btn-start[href='/login']").count() == 1
          and page.locator("section.start-hero a.btn-start[href='/login']").inner_text().strip() == "Anmelden")
    check("Demo-Knopf heisst 'Ohne Anmeldung selbst erleben' (Michael 03.09.)",
          page.locator("main a.btn-start[href='https://demo.inkludocs.de']").first.inner_text().strip() == "Ohne Anmeldung selbst erleben")
    check("Zielgruppen: vier H3 statt fettem Text (Cody 02.09.)",
          page.locator("ul.start-zielgruppen > li > h3").count() == 4)
    check("Hero: drei Paare, Satz direkt VOR dem Link (Steve 03.09.: 'Du hast schon ein Konto.' dann 'Anmelden')",
          page.locator("section.start-hero .start-aktion").count() == 3
          and all(page.locator("section.start-hero .start-aktion").nth(i).locator("p.start-hinweis + a.btn-start").count() == 1 for i in range(3))
          and "50 Credits im Monat, ohne Zahlungsdaten." in page.locator("section.start-hero .start-aktion").nth(1).inner_text())
    check("keine Seitenleiste, kein Rest der Login-Karte",
          page.locator("#appSidebar, .auth-container, .legal-footer, .subtitle").count() == 0)
    check("Fusszeile: 7 Links wie im oeffentlichen Geruest", fusszeile(page) == FUSSZEILE, str(fusszeile(page)))
    check("Bilder haben Alt-Text (oder es gibt keine)",
          page.locator("img:not([alt])").count() == 0)
    check("Hero-Bild: WebP-Quelle, JPEG-Rueckfall, Breite/Hoehe gesetzt, Alt-Text aus InkluDocs (Michael 04.09.)",
          page.locator("section.start-hero picture source[type='image/webp']").count() == 1
          and page.locator("section.start-hero img[src$='.jpg'][width='734'][height='498']").count() == 1
          and "Illustration" in (page.locator("section.start-hero img").get_attribute("alt") or ""))
    check("Hero-Bild wird wirklich geladen (naturalWidth > 0)",
          page.locator("section.start-hero img").evaluate("i => i.complete && i.naturalWidth > 0"))
    axe_lauf(page, "/")

    print("== A2. Startseite: Suchmaschinen-Angaben im rohen HTML ==")
    html = roh("/")
    check("Meta-Beschreibung", 'name="description" content="Alt-Texte per KI' in html)
    check("canonical", '<link rel="canonical" href="https://' in html)
    check("Open Graph Titel/Beschreibung/URL/Locale",
          all(s in html for s in ('property="og:title"', 'property="og:description"',
                                  'property="og:url"', 'property="og:locale" content="de_DE"')))
    ld_start = html.find('<script type="application/ld+json">')
    ld_ende = html.find("</script>", ld_start)
    ld_ok = False
    if ld_start > -1:
        try:
            ld = json.loads(html[ld_start + len('<script type="application/ld+json">'):ld_ende])
            ld_ok = ld.get("@type") == "SoftwareApplication" and ld["provider"]["name"] == "InkluTec"
        except Exception as e:  # noqa: BLE001
            print("   JSON-LD:", e)
    check("JSON-LD SoftwareApplication mit Anbieter InkluTec", ld_ok)
    check("Staging: noindex gesetzt" if staging else "Prod: kein noindex",
          ('name="robots" content="noindex' in html) == staging)
    check("Kopfzeile und Fusszeile stehen ohne JavaScript im HTML",
          'class="start-header"' in html and all(f'href="{h}"' in html for h, _ in FUSSZEILE))
    html_en = roh("/", {"Accept-Language": "en-GB,en;q=0.9"})
    check("Accept-Language en: html lang=en, englische H1, og:locale en_GB",
          '<html lang="en">' in html_en and all(t in html_en for t in H1_EN_TEILE) and 'content="en_GB"' in html_en)

    print("== B. Login-Karten im Startgeruest ==")
    for pfad, h1 in KARTEN:
        js_fehler.clear()
        url = f"{BASE}{pfad}" + ("?token=test" if pfad == "/reset" else "")
        page.goto(url, wait_until="networkidle")
        check(f"{pfad}: keine JavaScript-Fehler", not js_fehler, str(js_fehler[:1]))
        check(f"{pfad}: Login-Karte vorhanden", page.locator("main.start-main-auth .auth-container").count() == 1)
        h1s = page.locator("h1")
        check(f"{pfad}: genau eine H1 '{h1}', Marke nicht in der H1",
              h1s.count() == 1 and h1s.first.inner_text().strip() == h1,
              str([x.inner_text().strip() for x in h1s.all()]))
        check(f"{pfad}: Kopfzeile vollstaendig", kopf_nav(page) == KOPF_NAV, str(kopf_nav(page)))
        if pfad in ("/login", "/register"):
            check(f"{pfad}: aria-current auf dem eigenen Kopfzeilen-Eintrag",
                  page.locator(f"header nav a[href='{pfad}'][aria-current='page']").count() == 1)
        check(f"{pfad}: Fusszeile 7 Links, keine alte .legal-footer",
              fusszeile(page) == FUSSZEILE and page.locator(".legal-footer").count() == 0)
        check(f"{pfad}: noindex", 'name="robots" content="noindex' in roh(pfad))
        if pfad != "/login":
            check(f"{pfad}: Link zur Anmeldung zeigt auf /login",
                  page.locator(".auth-links a[href='/login']").count() == 1
                  and page.locator(".auth-links a[href='/']").count() == 0)
        axe_lauf(page, pfad)
    page.goto(f"{BASE}/login", wait_until="domcontentloaded")
    check("/login: Links Konto erstellen (/register) und Passwort vergessen (/forgot)",
          page.locator(".auth-links a[href='/register']").count() == 1
          and page.locator(".auth-links a[href='/forgot']").count() == 1)
    check("/login: Demo-Hinweis bleibt", page.locator("a[href='https://demo.inkludocs.de']").count() == 1)

    print("== C. Weiterleitungen ohne Login ==")
    for pfad in ("/app", "/dashboard", "/projekte", "/einstellungen"):
        st, ziel = status_und_ziel(pfad)
        check(f"{pfad} -> /login", st in (302, 303, 307) and ziel == "/login", f"{st} {ziel}")
    st, ziel = status_und_ziel("/")
    check("/ liefert 200 ohne Weiterleitung", st == 200, f"{st} {ziel}")
    page.goto(f"{BASE}/login?geloescht=1", wait_until="networkidle")
    check("/login?geloescht=1: Bestaetigung sichtbar und fokussiert",
          page.locator("#geloeschtHinweis").count() == 1
          and page.evaluate("document.activeElement.id") == "geloeschtHinweis")

    print("== D. Eingeloggt ==")
    if LOGIN_MAIL and LOGIN_PW:
        page.goto(f"{BASE}/login", wait_until="domcontentloaded")
        page.fill("#email", LOGIN_MAIL)
        page.fill("#password", LOGIN_PW)
        page.click("button[type=submit]")
        page.wait_for_url("**/dashboard", timeout=15000)
        check("Anmeldung ueber /login fuehrt ins Dashboard", page.url.endswith("/dashboard"))
        page.goto(f"{BASE}/", wait_until="domcontentloaded")
        check("eingeloggt: / leitet ins Dashboard", page.url.endswith("/dashboard"), page.url)
        page.goto(f"{BASE}/kontakt", wait_until="networkidle")
        check("eingeloggt auf /kontakt: kein 'Anmelden oder registrieren' in der Seitenleiste",
              page.locator("#appSidebar nav a[href='/login']").count() == 0)
        page.click("#logoutBtn")
        page.wait_for_url(f"{BASE}/", timeout=15000)
        check("Abmelden landet auf der Startseite", page.url.rstrip("/") == BASE.rstrip("/")
              and " ".join(page.locator("h1").first.inner_text().split()) == H1_DE)
        page.goto(f"{BASE}/preise", wait_until="networkidle")
        nav = [(a.get_attribute("href"), a.inner_text().strip()) for a in page.locator("#appSidebar nav a").all()]
        check("oeffentliche Seitenleiste: 'Anmelden oder registrieren' zeigt auf /login",
              ("/login", "Anmelden oder registrieren") in nav, str(nav))
    else:
        print("  (uebersprungen: kein E2E-Konto in ~/.e2e.env)")

    print("== E. robots.txt + sitemap.xml ==")
    robots = roh("/robots.txt")
    if staging:
        check("Staging robots.txt sperrt alles", "Disallow: /\n" in robots and "Sitemap:" not in robots)
    else:
        check("robots.txt: Sitemap-Verweis, App gesperrt", "Sitemap:" in robots and "Disallow: /app" in robots)
    sitemap = roh("/sitemap.xml")
    check("sitemap.xml: urlset mit Startseite und Preisen",
          "<urlset" in sitemap and "/preise</loc>" in sitemap and "/ueber-uns</loc>" in sitemap)
    check("sitemap.xml: keine Login- oder App-Seiten",
          all(s not in sitemap for s in ("/login", "/app", "/dashboard")))

    print("== F. Schmaler Bildschirm ==")
    mobil = browser.new_context(viewport={"width": 375, "height": 740}, locale="de-DE").new_page()
    mobil.goto(f"{BASE}/", wait_until="networkidle")
    check("375 px: kein horizontales Scrollen",
          mobil.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
    check("375 px: Kopfzeilen-Navigation sichtbar", mobil.locator("header nav a[href='/login']").is_visible())
    check("375 px: Knoepfe sichtbar", mobil.locator("main a.btn-start").first.is_visible())
    mobil.goto(f"{BASE}/login", wait_until="networkidle")
    check("375 px /login: kein horizontales Scrollen",
          mobil.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))

    browser.close()

print(f"\nErgebnis: {ok} OK, {fail} FEHLER")
sys.exit(1 if fail else 0)
