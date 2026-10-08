#!/usr/bin/env python3
"""Klicktest DARSTELLUNG OHNE SCREENREADER (Runde 11, 08.10.2026 — Steve: „barrierefreiheitstechnisch soll alles
stimmen“, auch für Menschen mit Seheinschränkung ohne Screenreader). Je Seite (angemeldet und öffentlich):
- keine sichtbare Schrift unter 14 px;
- EIN Linkstil: Inhaltslinks in #a94200 und unterstrichen (Navigation und Knopf-Links ausgenommen);
- Fokusring 3 px an allen per Tab erreichbaren Elementen;
- Hochkontrast (forced-colors, Chromium-Emulation): jeder Knopf mit sichtbarem Rahmen; in der Navigation trägt nur die
  aktuelle Seite den Balken; die aktuelle Ansicht („Prompts | Stammdaten“, Projekt-Ansichten) ist ohne Farbe erkennbar;
- Sekundärknöpfe mit Rand ≥ 3:1; „Konto“ mit deutlichem Pfeil;
- Formularfehler: Feld mit aria-invalid hat roten Rand, der Fehlertext beginnt mit „Fehler:“;
- axe (WCAG + landmark-unique) 0.
Das Kontaktformular wird NICHT abgeschickt. NUR gegen Staging; fiktives Konto auf .invalid, am Ende gelöscht.
Aufruf auf dem Server: /home/claude/.venv-pw/bin/python ui_darstellung.py [BASIS]
"""
import subprocess
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://staging.inkludocs.inklutec.de"
if "staging" not in BASE and "localhost" not in BASE:
    sys.exit("ABBRUCH: nur gegen Staging.")
KUNDE, KPW = "darstellung@darstellung-r11.invalid", "Darstellung-R11-2026!x"


def _e2e(schluessel):
    import os
    wert = os.environ.get(schluessel)
    if wert:
        return wert
    for zeile in open(os.path.expanduser("~/.e2e.env"), encoding="utf-8"):
        if zeile.startswith(schluessel + "="):
            return zeile.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return ""


ADMIN_MAIL, ADMIN_PW = _e2e("INKLUDOCS_E2E_MAIL"), _e2e("INKLUDOCS_E2E_PW")
AXE = urllib.request.urlopen("https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js", timeout=20).read().decode()
ok = fehler = 0


def check(name, bedingung, info=""):
    global ok, fehler
    if bedingung:
        ok += 1
        print("OK   ", name)
    else:
        fehler += 1
        print("FEHLT", name, "—", str(info)[:400])


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
d = fitz.open(); s = d.new_page(); s.insert_text((72, 72), 'Fiktiver Jahresbericht')
pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0); pm.set_rect(pm.irect, (200, 80, 0))
s.insert_image(fitz.Rect(72, 100, 312, 260), stream=pm.tobytes('png'))
r = k.post('/api/upload', files={{'file': ('Jahresbericht fiktiv.pdf', d.tobytes(), 'application/pdf')}}); r.raise_for_status()
pid = r.json()['project_id']; did = r.json()['document_id']
for _ in range(120):
    if k.get('/api/projects/%d/status' % pid).json().get('status') != 'extracting':
        break
    time.sleep(0.5)
k.post('/api/prompts', json={{'name': 'Einfache Sprache (fiktiv)', 'prompt_text': 'Formuliere einfach.', 'category': '', 'description': 'Beispiel'}}).raise_for_status()
k.post('/api/express/warenkorb/dokumente', json={{'document_ids': [did]}})
print(pid)
"""

KLEIN = """() => {
  const aus = [];
  const sichtbar = (e) => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none' && !e.closest('.sr-only, .visually-hidden, [hidden]'); };
  for (const e of document.querySelectorAll('body *')) {
    if (['SCRIPT', 'STYLE', 'SVG', 'svg', 'NOSCRIPT', 'OPTION'].includes(e.tagName)) continue;
    const text = Array.from(e.childNodes).filter((n) => n.nodeType === 3).map((n) => n.textContent.trim()).join(' ').trim();
    if (!text || !sichtbar(e)) continue;
    const px = parseFloat(getComputedStyle(e).fontSize);
    if (px < 13.95) aus.push(e.tagName.toLowerCase() + (e.className && typeof e.className === 'string' ? '.' + e.className.split(' ')[0] : '') + ' ' + px.toFixed(1) + 'px „' + text.slice(0, 30) + '“');
  }
  return aus;
}"""
LINKS = """() => {
  const aus = [];
  for (const a of document.querySelectorAll('main a[href], footer a[href]')) {
    if (/btn/.test(a.className || '') || a.closest('nav, .app-sidebar, .start-header, .sr-only, .visually-hidden')) continue;
    const r = a.getBoundingClientRect(); if (r.width < 1 || r.height < 1) continue;
    const s = getComputedStyle(a);
    if (s.color !== 'rgb(169, 66, 0)' || !s.textDecorationLine.includes('underline'))
      aus.push((a.textContent || '').trim().slice(0, 30) + ' ' + s.color + ' ' + s.textDecorationLine);
  }
  return aus;
}"""
KNOEPFE_HC = """() => {
  const aus = [];
  for (const b of document.querySelectorAll('main button, main a[class*="btn"], main .btn, footer button, dialog[open] button')) {
    const r = b.getBoundingClientRect(); if (r.width < 1 || r.height < 1 || b.closest('.sr-only, .visually-hidden, [hidden]')) continue;
    const s = getComputedStyle(b);
    if (s.borderTopStyle === 'none' || parseFloat(s.borderTopWidth) < 1) aus.push((b.textContent || '').trim().slice(0, 30));
  }
  return aus;
}"""
NAV_HC = """() => {
  const links = Array.from(document.querySelectorAll('#appSidebar .app-nav a'));
  const quer = window.innerWidth <= 768;
  const balken = (a) => { const s = getComputedStyle(a); const bg = getComputedStyle(a.closest('.app-sidebar')).backgroundColor;
    return quer ? s.borderBottomColor !== bg && s.borderBottomColor !== s.backgroundColor && s.borderBottomStyle !== 'none'
                : s.borderLeftColor !== bg && s.borderLeftColor !== s.backgroundColor && s.borderLeftStyle !== 'none'; };
  return links.filter(balken).map((a) => a.textContent.trim());
}"""


def tab_lauf(pg, schritte=45):
    """Fokusring je Tab-Halt: Breite der Umrandung (outline) in px."""
    pg.evaluate("() => { document.activeElement && document.activeElement.blur && document.activeElement.blur(); window.scrollTo(0, 0); }")
    pg.locator("body").focus() if pg.locator("body").count() else None
    duenn = []
    for _ in range(schritte):
        pg.keyboard.press("Tab")
        info = pg.evaluate("""() => { const e = document.activeElement; if (!e || e === document.body) return null;
            const s = getComputedStyle(e); return [e.tagName.toLowerCase() + (e.id ? '#' + e.id : ''), (e.textContent || e.value || '').trim().slice(0, 25),
              s.outlineStyle, parseFloat(s.outlineWidth)]; }""")
        if not info:
            break
        if info[2] == "none" or info[3] < 3:
            duenn.append(f"{info[0]} „{info[1]}“ {info[2]} {info[3]}px")
    return duenn


def axe(pg, name):
    pg.add_script_tag(content=AXE)
    e = pg.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}})")
    u = pg.evaluate("async () => await axe.run(document, {runOnly:{type:'rule',values:['landmark-unique','heading-order']}})")
    v = e["violations"] + u["violations"]
    check(f"axe {name}: 0 Verstöße (WCAG, landmark-unique, heading-order)", not v,
          [(x["id"], [t for n in x["nodes"] for t in n["target"]][:3]) for x in v])


pid = int(im_container(ANLEGEN).splitlines()[-1])
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        js_fehler = []

        def seite(ctx):
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: js_fehler.append(str(e)))
            return pg

        def pruefe(pg, name, pfad, warte=1200, tab=True):
            pg.goto(f"{BASE}{pfad}", wait_until="networkidle")
            pg.wait_for_timeout(warte)
            klein = pg.evaluate(KLEIN)
            check(f"{name}: keine Schrift unter 14 px", not klein, klein[:8])
            links = pg.evaluate(LINKS)
            check(f"{name}: Inhaltslinks #a94200 und unterstrichen", not links, links[:8])
            if tab:
                duenn = tab_lauf(pg)
                check(f"{name}: Fokusring 3 px an jedem Tab-Halt", not duenn, duenn[:6])
            axe(pg, name)
            pg.emulate_media(forced_colors="active")
            pg.wait_for_timeout(200)
            ohne = pg.evaluate(KNOEPFE_HC)
            check(f"{name} (Hochkontrast): jeder Knopf mit sichtbarem Rahmen", not ohne, ohne[:8])
            if pg.locator("#appSidebar .app-nav").count():
                balken = pg.evaluate(NAV_HC)
                aktuell = [a.inner_text().strip() for a in pg.locator("#appSidebar .app-nav a[aria-current=page]").all()]
                check(f"{name} (Hochkontrast): Balken nur an der aktuellen Seite", balken == aktuell, (balken, aktuell))
            pg.emulate_media(forced_colors="none")

        # ── Öffentlich ──
        ctx = b.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
        pg = seite(ctx)
        for name, pfad in (("Start", "/"), ("Preise", "/preise"), ("Über uns und Kontakt (öffentlich)", "/ueber-uns"),
                           ("Anmeldung", "/login"), ("Registrierung", "/register"), ("Passwort vergessen", "/forgot"),
                           ("Impressum", "/impressum"), ("Datenschutz", "/datenschutz")):
            pruefe(pg, name, pfad, warte=600, tab=(pfad in ("/login", "/ueber-uns", "/register")))
        # Formularfehler: roter Rand + „Fehler:“
        pg.goto(f"{BASE}/ueber-uns", wait_until="networkidle")
        pg.click("#ko_senden")
        pg.wait_for_timeout(300)
        rand = pg.evaluate("() => { const s = getComputedStyle(document.getElementById('ko_email')); return [s.borderTopColor, s.boxShadow]; }")
        vor = pg.evaluate("() => getComputedStyle(document.getElementById('ko_emailFehler'), '::before').content")
        check("Fehlerfeld: roter Rand (#b91c1c), Fehlertext beginnt mit „Fehler:“", rand[0] == "rgb(185, 28, 28)" and vor == '"Fehler: "', (rand, vor))
        pg.emulate_media(forced_colors="active")
        hc = pg.evaluate("() => { const s = getComputedStyle(document.getElementById('ko_email')); return [s.borderTopStyle, s.borderTopWidth]; }")
        check("Fehlerfeld im Hochkontrast: gestrichelter 3-px-Rand", hc == ["dashed", "3px"], hc)
        pg.emulate_media(forced_colors="none")
        en = b.new_context(locale="en-GB", viewport={"width": 1280, "height": 900}).new_page()   # Sprache über den Browser
        en.goto(f"{BASE}/ueber-uns", wait_until="networkidle")
        en.click("#ko_senden")
        en.wait_for_timeout(300)
        vor = en.evaluate("() => [document.documentElement.lang, getComputedStyle(document.getElementById('ko_emailFehler'), '::before').content]")
        check("Englisch: „Error:“ vor dem Fehlertext", vor == ["en", '"Error: "'], vor)
        en.context.close()
        pg.goto(f"{BASE}/login?lang=de", wait_until="networkidle")
        pg.fill("#email", "niemand@darstellung-r11.invalid")
        pg.fill("#password", "falsch-falsch-1")
        pg.click("button[type=submit]")
        pg.wait_for_timeout(1500)
        vor = pg.evaluate("() => getComputedStyle(document.getElementById('errorMsg'), '::before').content")
        text = pg.locator("#errorMsg").inner_text()      # ohne das vorangestellte „Fehler: “ (CSS)
        check("Anmeldung: Fehlerkasten beginnt mit „Fehler:“ (ohne Doppelung)", vor == '"Fehler: "' and text and not text.startswith("Fehler"), (vor, text))
        ctx.close()

        # ── Angemeldet ──
        ctx = b.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
        pg = seite(ctx)
        pg.goto(f"{BASE}/login", wait_until="domcontentloaded")
        pg.fill("#email", KUNDE)
        pg.fill("#password", KPW)
        pg.click("button[type=submit]")
        pg.wait_for_url("**/dashboard", timeout=20000)
        for name, pfad, tab in (("Startseite", "/dashboard", True), ("Meine Projekte", "/projekte", True), ("Neues Projekt", "/projekt-neu", True),
                                ("Meine Vorlagen", "/vorlagen", True), ("Prompts", "/prompts", True), ("Stammdaten", "/stammdaten", True),
                                ("Über uns und Kontakt", "/ueber-uns", False), ("Einstellungen", "/einstellungen", True), ("Konto", "/konto", False),
                                ("Abo & Verbrauch", "/abo", False), ("Datensicherheit", "/datensicherheit", True), ("Meine Aufträge", "/express", True),
                                ("Express-Warenkorb", "/express/warenkorb", True), ("Meine Ablage", "/ablage", False),
                                ("Projekt Dokument", f"/app?projekt={pid}&ansicht=dokument", True), ("Projekt Alt-Texte", f"/app?projekt={pid}&ansicht=alttexte", True),
                                ("Team", "/team", False), ("API-Schlüssel", "/api-schluessel", False), ("Geteilte Projekte", "/geteilte-projekte", False),
                                ("Express-Bedingungen", "/express/bedingungen", False), ("Impressum (App)", "/impressum-app", False),
                                ("Widerrufsbelehrung (App)", "/widerruf-app", False), ("Nutzungsbedingungen (App)", "/nutzungsbedingungen-app", False)):
            pruefe(pg, name, pfad, warte=2500 if pfad.startswith("/app") else 1200, tab=tab)
        # Sekundärknopf-Rand >= 3:1, Ansichtswahl im Hochkontrast, Konto-Pfeil
        pg.goto(f"{BASE}/prompts", wait_until="networkidle")
        pg.wait_for_timeout(800)
        rand = pg.evaluate("() => getComputedStyle(document.querySelector('.vorlagen-wahl a.btn-secondary')).borderTopColor")
        check("Sekundärknopf: Rand #767f8f (4,0:1)", rand == "rgb(118, 127, 143)", rand)
        unter = pg.evaluate("() => [...document.querySelectorAll('.vorlagen-wahl a')].map((a) => [a.textContent.trim(), getComputedStyle(a).textDecorationLine, getComputedStyle(a).textDecorationThickness])")
        check("Ansichtswahl: aktuelle Ansicht dick unterstrichen, die andere nicht", unter[0][1] == "underline" and unter[0][2] == "3px" and unter[1][1] == "none", unter)
        pg.emulate_media(forced_colors="active")
        hc = pg.evaluate("() => [...document.querySelectorAll('.vorlagen-wahl a')].map((a) => [getComputedStyle(a).borderTopWidth, getComputedStyle(a).textDecorationLine])")
        check("Ansichtswahl im Hochkontrast unterscheidbar (4-px-Rahmen + Unterstreichung)", hc[0] == ["4px", "underline"] and hc[1][0] != "4px" and hc[1][1] == "none", hc)
        pg.emulate_media(forced_colors="none")
        pg.goto(f"{BASE}/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(2500)
        akt = pg.evaluate("() => { const a = document.querySelector('.ansicht-knoepfe a[aria-current=page]'); return a ? [getComputedStyle(a).textDecorationLine, getComputedStyle(a).textDecorationThickness] : null; }")
        check("Projekt-Ansichten: aktuelle Ansicht dick unterstrichen", akt == ["underline", "3px"], akt)
        pg.goto(f"{BASE}/dashboard", wait_until="networkidle")
        pfeil = pg.evaluate("() => { const s = getComputedStyle(document.querySelector('#navKonto > summary'), '::before'); return [s.borderRightWidth, s.width, s.transform]; }")
        pg.locator("#navKonto > summary").click()
        pg.wait_for_timeout(400)                          # Drehung ist animiert (0,15 s)
        pfeil2 = pg.evaluate("() => getComputedStyle(document.querySelector('#navKonto > summary'), '::before').transform")
        check("„Konto“: deutlicher Pfeil (3 px, ≈ 10 px), dreht sich beim Aufklappen", pfeil[0] == "3px" and float(pfeil[1].rstrip("px")) >= 9 and pfeil[2] != pfeil2, (pfeil, pfeil2))
        ctx.close()

        # ── Verwaltung (Admin-Testkonto aus ~/.e2e.env) ──
        if ADMIN_MAIL and ADMIN_PW:
            ctx = b.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
            pg = seite(ctx)
            pg.goto(f"{BASE}/login", wait_until="domcontentloaded")
            pg.fill("#email", ADMIN_MAIL)
            pg.fill("#password", ADMIN_PW)
            pg.click("button[type=submit]")
            pg.wait_for_url("**/dashboard", timeout=20000)
            for name, pfad in (("Verwaltung Kunden", "/verwaltung/kunden"), ("Verwaltung Umsatz", "/verwaltung/umsatz"),
                               ("Verwaltung KI-Kosten", "/verwaltung/ki-kosten"), ("Verwaltung Express", "/verwaltung/express"),
                               ("Verwaltung API", "/verwaltung/api"), ("Verwaltung Einstellungen", "/verwaltung/einstellungen")):
                pruefe(pg, name, pfad, warte=1800, tab=False)
            ctx.close()
        check("Keine JS-Fehler", not js_fehler, js_fehler)
        b.close()
finally:
    im_container(AUFRAEUMEN)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
