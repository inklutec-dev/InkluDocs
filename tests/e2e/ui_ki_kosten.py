#!/usr/bin/env python3
"""Klicktest VERWALTUNG — KI-KOSTEN (05.10.2026) im echten Browser, mit axe auf der Seite, nach dem Aufklappen und in
beiden Dialogen. NUR gegen Staging. Legt fiktive Kostenzeilen (Umgebung „test“) an und entfernt sie am Ende wieder;
speichert KEINE Preise (das prueft verify_ki_kosten.py ueber die API).
Aufruf auf dem Server: /home/claude/.venv-pw/bin/python ui_ki_kosten.py [BASIS]
"""
import os
import subprocess
import sys
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


MAIL, PW = _e2e("INKLUDOCS_E2E_MAIL"), _e2e("INKLUDOCS_E2E_PW")
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
    return subprocess.run(["sudo", "docker", "exec", "inkludocs-staging", "python3", "-c", code],
                          capture_output=True, text=True, check=True).stdout.strip()


ANLEGEN = f"""
import sqlite3
c = sqlite3.connect('/app/data/inkludocs.db')
uid = c.execute("SELECT id FROM users WHERE email = ?", ({MAIL!r},)).fetchone()[0]
for konto, projekt, bild, zweck, cent in ((uid, 999001, 999101, 'alttext', 2.5), (uid, 999001, 999102, 'alttext', 1.25),
                                         (uid, None, None, 'chatbot', 0.4), (None, None, None, 'unbekannt', 0.1)):
    c.execute("INSERT INTO ki_aufrufe (umgebung, user_id, konto_user_id, project_id, image_id, zweck, schritt, anbieter, modell, "
              "tokens_ein, tokens_aus, kosten_usd, kosten_eur_cent, kurs_usd_eur, fehler) "
              "VALUES ('test', ?, ?, ?, ?, ?, 'UI-Test', 'gemini', 'gemini-3.8-flash', 1000, 100, ?, ?, 0.8909, 'ui_ki_kosten fiktiv')",
              (konto, konto, projekt, bild, zweck, cent / 89.09, cent))
c.commit()
print(uid)
"""
ENTFERNEN = """
import sqlite3
c = sqlite3.connect('/app/data/inkludocs.db')
c.execute("DELETE FROM ki_aufrufe WHERE fehler = 'ui_ki_kosten fiktiv'")
c.commit()
"""

uid = int(im_container(ANLEGEN))
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_context(locale="de-DE").new_page()
        js_fehler = []
        pg.on("pageerror", lambda e: js_fehler.append(str(e)))
        # Die absichtlich falsche Preiseingabe beantwortet der Server mit 400 — das meldet der Browser als
        # Ladefehler; das ist erwartet und kein Skriptfehler.
        pg.on("console", lambda m: js_fehler.append(m.text) if m.type == "error" and "status of 400" not in m.text else None)
        pg.goto(f"{BASE}/login", wait_until="domcontentloaded")
        pg.fill("#email", MAIL)
        pg.fill("#password", PW)
        pg.click("button[type=submit]")
        pg.wait_for_url("**/dashboard", timeout=15000)

        pg.goto(f"{BASE}/verwaltung/ki-kosten", wait_until="networkidle")
        pg.wait_for_timeout(800)
        check("Genau eine H1 „Verwaltung: KI-Kosten“", pg.locator("h1").count() == 1 and pg.locator("h1").inner_text() == "Verwaltung: KI-Kosten")
        check("Bereichs-Navigation: „KI-Kosten“ ist aktuelle Seite",
              pg.locator(".verwaltung-nav a[aria-current=page]").inner_text().strip() == "KI-Kosten")
        check("Monatsüberschrift nennt den Monat", pg.locator("#h-ki-monat").inner_text().startswith("KI-Kosten im "), pg.locator("#h-ki-monat").inner_text())
        zahlen = pg.locator("#kiZahlen").inner_text()
        check("Kennzahlen: KI-Kosten, Umsatz, Bleibt, Aufrufe", all(w in zahlen for w in ("KI-Kosten:", "Umsatz:", "Bleibt nach KI-Kosten:", "KI-Aufrufe:")), zahlen)
        check("Monatsauswahl hat Einträge", pg.locator("#kiZeitraum option").count() >= 1)
        check("Zweck Alt-Texte gelistet", "Alt-Texte:" in pg.locator("#kiZwecke").inner_text())
        check("„Ohne Zuordnung“ gelistet", "Ohne Zuordnung" in pg.locator("#kiKunden").inner_text())
        check("Keine Tabellen auf der Seite", pg.locator("main table").count() == 0)
        axe(pg, "KI-Kosten-Seite")

        # Aufklappen: Testkonto -> Projekte -> Bilder (das fiktive Projekt 999001 gibt es nicht -> „Gelöschtes Projekt“,
        # 2 Aufrufe; andere Zeilen des Testkontos aus früheren Läufen stören nicht)
        eintrag = pg.locator("#kiKunden li.verwaltung-zeile", has=pg.locator(f"a[href='/verwaltung/kunden/{uid}']"))
        check("Testkonto in der Kundenliste verlinkt", eintrag.count() == 1, eintrag.count())
        kunde = eintrag.locator("details").first
        kunde.locator("summary").first.click()
        pg.wait_for_timeout(1000)
        projekt = kunde.locator("details", has=pg.locator("summary", has_text="2 Aufrufe"))
        check("Projekte des Kunden geladen (fiktives Projekt mit 2 Aufrufen)", projekt.count() == 1, kunde.inner_text())
        projekt.locator("summary").click()
        pg.wait_for_timeout(1000)
        check("Bilder des Projekts geladen", "Bild 999101" in projekt.inner_text() and "Bild 999102" in projekt.inner_text(), projekt.inner_text())
        axe(pg, "nach dem Aufklappen")

        # Preisliste + Dialoge (ohne zu speichern)
        check("Preisliste nennt gemini-3.1-pro-preview", "gemini-3.1-pro-preview" in pg.locator("#kiPreise").inner_text())
        check("Wechselkurs genannt", "1 US-Dollar =" in pg.locator("#kiPreise").inner_text())
        knopf = pg.locator("#kiPreise button", has_text="Preis ändern").first
        check("Voll-Admin sieht „Preis ändern“", knopf.count() == 1)
        knopf.click()
        pg.wait_for_timeout(300)
        check("Preisdialog offen, Fokus auf „Eingabe“", pg.evaluate("() => document.activeElement.id") == "kiPreisEin")
        check("Modellfeld schreibgeschützt", pg.locator("#kiPreisModell").evaluate("e => e.readOnly"))
        check("Keine Zahlen-Stepper", pg.locator("dialog input[type=number]").count() == 0)
        axe(pg, "Preisdialog")
        pg.fill("#kiPreisEin", "viel")
        pg.fill("#kiPreisQuelle", "UI-Test")
        pg.click("#kiPreisForm button[type=submit]")
        pg.wait_for_timeout(600)
        check("Fehlermeldung bei Unsinn", "Zahl" in pg.locator("#kiPreisFehler").inner_text(), pg.locator("#kiPreisFehler").inner_text())
        pg.click("#kiPreisAbbrechen")
        pg.wait_for_timeout(300)
        check("Abbrechen: Fokus zurück auf den Knopf", pg.evaluate("() => document.activeElement.textContent.trim()") == "Preis ändern")
        pg.click("#kiKursAendern")
        pg.wait_for_timeout(300)
        check("Kursdialog offen, Fokus im Kursfeld", pg.evaluate("() => document.activeElement.id") == "kiKursWert")
        axe(pg, "Kursdialog")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(300)
        check("Escape: Fokus zurück auf „Wechselkurs ändern“", pg.evaluate("() => document.activeElement.id") == "kiKursAendern")

        # Kundenseite zeigt gemessene Kosten
        pg.goto(f"{BASE}/verwaltung/kunden/{uid}", wait_until="networkidle")
        pg.wait_for_timeout(1000)
        check("Kundenseite: „KI-Kosten dieses Kontos“ gemessen", "KI-Kosten dieses Kontos:" in pg.locator("main").inner_text() and "gemessen" in pg.locator("main").inner_text())
        axe(pg, "Kundenseite")

        # Englische Oberfläche: Texte übersetzt
        pg.goto(f"{BASE}/set-language/en", wait_until="networkidle")
        pg.goto(f"{BASE}/verwaltung/ki-kosten", wait_until="networkidle")
        pg.wait_for_timeout(800)
        check("Englisch: Überschrift übersetzt", "AI costs" in pg.locator("h1").inner_text(), pg.locator("h1").inner_text())
        pg.goto(f"{BASE}/set-language/de", wait_until="networkidle")
        check("Keine JS-Fehler", not js_fehler, js_fehler)
        b.close()
finally:
    im_container(ENTFERNEN)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
