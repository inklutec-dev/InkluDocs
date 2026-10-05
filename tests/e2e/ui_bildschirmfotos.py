#!/usr/bin/env python3
"""Bildschirmfotos VOR und NACH den globalen CSS-Aenderungen der Korrekturrunde (05.10.2026): Dateifeld-Regel nur noch an
der Hochladeflaeche, deutlicher Fokusring an .btn und am Etikett-Knopf .upload-btn, dunklerer Rahmen der Eingabefelder.
Die ganze App im Blick: Projekt-Hochladeflaeche (PDF), Knoepfe und Formulare auf Anmeldung, Startseite, Einstellungen,
Verwaltung, dazu die Express-Seiten. Je Stelle ein Foto ohne und eines mit Tastaturfokus.
NUR gegen Staging. Ein fiktives Konto auf .invalid wird im Container angelegt und am Ende geloescht; die Verwaltung
bedient das E2E-Konto aus ~/.e2e.env.
    /home/claude/.venv-pw/bin/python ui_bildschirmfotos.py <vorher|nachher> [BASIS]
Fotos: /home/claude/umsetzung-1005/bilder/<phase>/; dazu messwerte.txt (berechnete Stile, zum Vergleich)."""
import json
import os
import subprocess
import sys

from playwright.sync_api import sync_playwright

PHASE = sys.argv[1] if len(sys.argv) > 1 else "nachher"
BASE = sys.argv[2] if len(sys.argv) > 2 else "https://staging.inkludocs.inklutec.de"
if "staging" not in BASE and "localhost" not in BASE:
    sys.exit("ABBRUCH: nur gegen Staging.")
ZIEL = f"/home/claude/umsetzung-1005/bilder/{PHASE}"
os.makedirs(ZIEL, exist_ok=True)


def _e2e(schluessel):
    for zeile in open(os.path.expanduser("~/.e2e.env"), encoding="utf-8"):
        if zeile.startswith(schluessel + "="):
            return zeile.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return ""


ADMIN_MAIL, ADMIN_PW = _e2e("INKLUDOCS_E2E_MAIL"), _e2e("INKLUDOCS_E2E_PW")
KUNDE, KPW = "fotos@css-fotos.invalid", "Fotos-CSS-2026!x"


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
import httpx
from database import create_user, get_db
create_user({KUNDE!r}, {KPW!r}, 'Foto Konto (fiktiv)')
k = httpx.Client(base_url='http://127.0.0.1:8001', timeout=120)
r = k.post('/api/login', json={{'email': {KUNDE!r}, 'password': {KPW!r}}})
k.headers['Cookie'] = 'token=' + r.cookies.get('token')
r = k.post('/api/projects', json={{'tool': 'pdf', 'name': 'Foto-Projekt (fiktiv)'}})
print(r.json()['project_id'])
"""

messwerte = []


def stil(pg, sel, felder=("outlineStyle", "outlineWidth", "outlineColor", "boxShadow", "borderColor", "position", "clip", "width")):
    werte = pg.evaluate("""([sel, felder]) => { const e = document.querySelector(sel); if (!e) return null;
        const s = getComputedStyle(e); const o = {}; felder.forEach((f) => { o[f] = s[f]; }); return o; }""", [sel, list(felder)])
    messwerte.append(f"{PHASE} {pg.url.replace(BASE, '')} {sel}: {json.dumps(werte, ensure_ascii=False)}")
    return werte


def foto(pg, name, sel=None, rand=24):
    pfad = os.path.join(ZIEL, name + ".png")
    if sel and pg.locator(sel).count():
        el = pg.locator(sel).first
        el.scroll_into_view_if_needed()
        pg.wait_for_timeout(150)
        box = el.bounding_box()
        vp = pg.viewport_size
        if box:
            x, y = max(0, box["x"] - rand), max(0, box["y"] - rand)
            w = min(vp["width"] - x, box["width"] + 2 * rand)
            h = min(vp["height"] - y, box["height"] + 2 * rand)
            if w > 0 and h > 0:
                pg.screenshot(path=pfad, clip={"x": x, "y": y, "width": w, "height": h})
                return
    pg.screenshot(path=pfad, full_page=False)


def mit_fokus(pg, sel, name, rahmen_sel=None, mess_sel=None):
    """Fokus wie mit der Tabulatortaste (programmatisch, ohne vorherigen Mausklick: :focus-visible greift)."""
    if not pg.locator(sel).count():
        messwerte.append(f"{PHASE} {name}: {sel} nicht gefunden")
        return
    pg.evaluate("(s) => document.querySelector(s).focus()", sel)
    pg.wait_for_timeout(200)
    stil(pg, mess_sel or sel)
    foto(pg, name, rahmen_sel or sel)


pid = int(im_container(ANLEGEN).splitlines()[-1])
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        # Anmeldung (nicht angemeldet): Eingabefelder und Knopf
        ctx = b.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        pg.goto(f"{BASE}/login", wait_until="networkidle")
        foto(pg, "01-login", "form")
        stil(pg, "#email")
        mit_fokus(pg, "button[type=submit]", "02-login-knopf-fokus")
        pg.fill("#email", KUNDE)
        pg.fill("#password", KPW)
        pg.click("button[type=submit]")
        pg.wait_for_url("**/dashboard", timeout=20000)
        pg.goto(f"{BASE}/dashboard", wait_until="networkidle")
        pg.wait_for_timeout(1000)
        foto(pg, "03-startseite")
        mit_fokus(pg, "main .btn", "04-startseite-knopf-fokus")
        # Projekt-Hochladeflaeche (PDF, Ansicht Dokument)
        pg.goto(f"{BASE}/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        foto(pg, "05-projekt-hochladen", "#projUploadZone")
        stil(pg, "#projUpload")
        mit_fokus(pg, "#projUpload", "06-projekt-hochladen-fokus", rahmen_sel="#projUploadZone", mess_sel="label[for=projUpload]")
        # Einstellungen (Formular mit Eingabefeldern)
        pg.goto(f"{BASE}/einstellungen", wait_until="networkidle")
        pg.wait_for_timeout(1000)
        foto(pg, "07-einstellungen")
        feld = "main input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=file])"
        mit_fokus(pg, feld, "08-einstellungen-feld-fokus", rahmen_sel="main form")
        # Express (Kunde): Hochladefeld
        pg.goto(f"{BASE}/express", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        foto(pg, "09-express-schritt1", "#exNeu" if pg.locator("#exNeu").count() else "main section")
        mit_fokus(pg, "#exDatei", "10-express-datei-fokus", rahmen_sel="#exDateiZone" if pg.locator("#exDateiZone").count() else "#exUploadForm",
                  mess_sel="label[for=exDatei]")
        stil(pg, "#exName")
        mit_fokus(pg, "#exName", "11-express-feld-fokus", rahmen_sel="#exBestellForm")
        mit_fokus(pg, "#exBestellen", "12-express-bestellen-fokus")
        ctx.close()
        # Verwaltung (E2E-Admin): Kundenliste mit Suchfeld, Express-Einstellungen
        ac = b.new_context(locale="de-DE", viewport={"width": 1280, "height": 900})
        ap = ac.new_page()
        ap.goto(f"{BASE}/login", wait_until="networkidle")
        ap.fill("#email", ADMIN_MAIL)
        ap.fill("#password", ADMIN_PW)
        ap.click("button[type=submit]")
        ap.wait_for_url("**/dashboard", timeout=20000)
        ap.goto(f"{BASE}/verwaltung/kunden", wait_until="networkidle")
        ap.wait_for_timeout(1500)
        foto(ap, "13-verwaltung-kunden")
        mit_fokus(ap, "main input", "14-verwaltung-suchfeld-fokus", rahmen_sel="main form")
        ap.goto(f"{BASE}/verwaltung/express", wait_until="networkidle")
        ap.wait_for_timeout(1500)
        foto(ap, "15-verwaltung-express-einstellungen", "#exvEinstellungen")
        b.close()
finally:
    im_container(AUFRAEUMEN)
open(os.path.join(ZIEL, "messwerte.txt"), "w", encoding="utf-8").write("\n".join(messwerte) + "\n")
print("\n".join(messwerte))
print("Fotos:", ZIEL)
