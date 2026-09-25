#!/usr/bin/env python3
"""Klicktest VERWALTUNG (Steve 25.09.2026): Kunden, Kundenseite, Gutschrift, Abo, Berichtigen,
Umsatz, Export, API, Einstellungen — im echten Browser, mit axe auf jeder Seite und in jedem
Dialog. NUR gegen Staging: legt ein Testkonto auf example.invalid an (Mails werden unterdrückt)
und löscht es am Ende wieder. Die Buchungen des Testkontos bleiben absichtlich stehen (Kontolöschung
behält Buchungen) und werden danach im Container entfernt:
    docker exec inkludocs-staging python3 -c "import sqlite3;c=sqlite3.connect('/app/data/inkludocs.db');
    c.execute(\\"DELETE FROM buchungen WHERE kunde_email LIKE 'verwaltung-test-%@example.invalid'\\");c.commit()"

Aufruf: /home/claude/.venv-pw/bin/python ui_verwaltung.py [BASIS]
"""
import os
import re
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://staging.inkludocs.inklutec.de"
if "staging" not in BASE and "localhost" not in BASE:
    sys.exit("ABBRUCH: Dieser Test bucht Credits und darf nur gegen Staging laufen.")


def _e2e(schluessel):
    wert = os.environ.get(schluessel)
    if wert:
        return wert
    try:
        for zeile in open(os.path.expanduser("~/.e2e.env"), encoding="utf-8"):
            if zeile.startswith(schluessel + "="):
                return zeile.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
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
    check(f"axe {name}: 0 Verstöße", not v,
          [(x["id"], [t for n in x["nodes"] for t in n["target"]][:3]) for x in v])


def fokus_id(pg):
    return pg.evaluate("() => document.activeElement && document.activeElement.id")


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context(locale="de-DE").new_page()
    js_fehler = []
    pg.on("pageerror", lambda e: js_fehler.append(str(e)))
    pg.on("console", lambda m: js_fehler.append(m.text) if m.type == "error" else None)

    pg.goto(f"{BASE}/login", wait_until="domcontentloaded")
    pg.fill("#email", MAIL)
    pg.fill("#password", PW)
    pg.click("button[type=submit]")
    pg.wait_for_url("**/dashboard", timeout=15000)

    # Testkonto anlegen (über die Oberfläche der Kundenliste)
    stempel = str(int(time.time()))
    test_mail = f"verwaltung-test-{stempel}@example.invalid"
    pg.goto(f"{BASE}/benutzer", wait_until="networkidle")
    check("Alte Adresse /benutzer führt zur Kundenliste", pg.url.endswith("/verwaltung/kunden"), pg.url)
    pg.wait_for_timeout(800)
    check("Seitenleiste: „Verwaltung“ ist aktuelle Seite",
          pg.locator(".app-nav a[aria-current=page]").inner_text().strip() == "Verwaltung")
    check("Bereichs-Navigation: „Kunden“ ist aktuelle Seite",
          pg.locator(".verwaltung-nav a[aria-current=page]").inner_text().strip() == "Kunden")
    check("Genau eine H1 „Verwaltung: Kunden“", pg.locator("h1").count() == 1 and pg.locator("h1").inner_text() == "Verwaltung: Kunden")
    check("Kundenliste nennt Auswahl und Anzahl", re.match(r"Alle Kunden: \d+ Kunden", pg.locator("#kundenStatus").inner_text()) is not None,
          pg.locator("#kundenStatus").inner_text())
    check("Kunden als Links, keine Überschrift je Kunde",
          pg.locator("#kundenListe a").count() > 0 and pg.locator("#kundenListe h3, #kundenListe h2").count() == 0)
    axe(pg, "Kundenliste")

    pg.click("#neuerKundeKlappe summary")
    pg.fill("#neuName", "Verwaltung Testkunde")
    pg.fill("#neuEmail", test_mail)
    pg.fill("#neuPw", "Testpasswort-123")
    pg.fill("#neuPw2", "Testpasswort-123")
    pg.click("#neuerKundeForm button[type=submit]")
    pg.wait_for_selector("#neuErgebnis a", timeout=10000)
    check("Neuer Kunde angelegt, Link zur Kundenseite", "Zur Kundenseite" in pg.locator("#neuErgebnis").inner_text())

    # Suche
    pg.fill("#suchText", "verwaltung-test-" + stempel)
    pg.click("#kundenSuche button[type=submit]")
    pg.wait_for_function("() => document.querySelector('#kundenStatus').textContent.trim().endsWith(': 1 Kunde')", timeout=10000)
    check("Überschrift nennt die Suche", pg.locator("#kundenStatus").inner_text().startswith("Suche „verwaltung-test-"), pg.locator("#kundenStatus").inner_text())
    check("Suche findet genau das Testkonto", pg.locator("#kundenListe li").count() == 1)
    check("Suche steht in der Adresse", "q=verwaltung-test-" in pg.url, pg.url)
    pg.click("#kundenListe li a")
    pg.wait_for_selector("#h-ueberblick", timeout=10000)
    kunde_url = pg.url
    kid = int(kunde_url.rstrip("/").split("/")[-1])
    check("Kundenseite: H1 mit Bereich und Namen", pg.locator("h1").inner_text() == "Verwaltung: Kunde Verwaltung Testkunde", pg.locator("h1").inner_text())
    check("Kundenseite: Zurück führt zur Suche", "q=verwaltung-test-" in (pg.locator("#zurListe").get_attribute("href") or ""))
    check("Kundenseite: noch keine Buchungen", "Noch keine Käufe oder Gutschriften" in pg.locator("main").inner_text())
    axe(pg, "Kundenseite")

    # Gutschrift: Verkauf 2.500
    pg.click("text=Credits gutschreiben")
    check("Gutschrift-Dialog: Fokus auf „Art“", fokus_id(pg) == "gsArt", fokus_id(pg))
    axe(pg, "Gutschrift-Dialog")
    pg.click("#gsForm button[type=submit]")
    check("Ohne Art: Fehlermeldung", "Bitte die Art wählen" in pg.locator("#gsFehler").inner_text())
    pg.select_option("#gsArt", "verkauf")
    pg.select_option("#gsMenge", "2500")
    check("Betrag mit Listenpreis vorbelegt (87,50)", pg.input_value("#gsBetrag") == "87,50", pg.input_value("#gsBetrag"))
    pg.select_option("#gsMenge", "frei")
    pg.fill("#gsFrei", "150")
    check("Freie Menge 150: Vorschlag 6,00", pg.input_value("#gsBetrag") == "6,00", pg.input_value("#gsBetrag"))
    pg.select_option("#gsMenge", "2500")
    pg.fill("#gsNummer", "RE-TEST-1")
    pg.click("#gsForm button[type=submit]")
    satz = pg.locator("#gsSatz").inner_text()
    check("Bestätigungssatz nennt Menge, Art und Betrag",
          "2.500 Credits" in satz and "Verkauf auf Rechnung" in satz and "87,50" in satz, satz)
    check("Fokus auf dem Bestätigungssatz", fokus_id(pg) == "gsSatz", fokus_id(pg))
    pg.click("#gsBuchen")
    pg.wait_for_function("() => !document.querySelector('#gsDialog').open", timeout=10000)
    pg.wait_for_function("() => (document.querySelector('#kundeInhalt') || {}).textContent.includes('RE-TEST-1')", timeout=10000)
    text = pg.locator("main").inner_text()
    check("Buchung erscheint: Verkauf auf Rechnung, 87,50 €, Rechnung RE-TEST-1",
          "Verkauf auf Rechnung, 87,50" in text and "RE-TEST-1" in text, text[:600])
    check("Umsatz mit diesem Kunden: 87,50 €", "Umsatz mit diesem Kunden insgesamt: 87,50" in text)

    # Bonus über der Grenze braucht das Häkchen
    pg.click("text=Credits gutschreiben")
    pg.select_option("#gsArt", "bonus")
    pg.select_option("#gsMenge", "frei")
    pg.fill("#gsFrei", "600")
    pg.fill("#gsGrund", "Klicktest großer Bonus")
    pg.click("#gsForm button[type=submit]")
    check("Großer Bonus: Häkchen wird angeboten", pg.locator("#gsGrossFeld").is_visible())
    pg.click("#gsBuchen")
    check("Ohne Häkchen: Fehlermeldung", "Häkchen" in pg.locator("#gsFehler2").inner_text())
    pg.check("#gsGross")
    pg.click("#gsBuchen")
    pg.wait_for_function("() => !document.querySelector('#gsDialog').open", timeout=10000)
    pg.wait_for_timeout(500)
    check("Bonus-Buchung erscheint", "600 Credits · Bonus (kostenlos)" in pg.locator("main").inner_text())

    # Berichtigen: Bonus -> Verkauf 24,00
    zeile = pg.locator("#kundeInhalt li", has_text="Klicktest großer Bonus")
    zeile.locator("button", has_text="Berichtigen").click()
    axe(pg, "Berichtigen-Dialog")
    pg.select_option("#korrekturArt", "verkauf")
    pg.fill("#korrekturBetrag", "24,00")
    pg.fill("#korrekturGrund", "Klicktest Korrektur")
    pg.click("#korrekturForm button[type=submit]")
    pg.wait_for_function("() => !document.querySelector('#korrekturDialog').open", timeout=10000)
    pg.wait_for_timeout(600)
    text = pg.locator("main").inner_text()
    check("Berichtigt: Verkauf 24,00 € mit Vermerk",
          "Verkauf auf Rechnung, 24,00" in text and "berichtigt" in text, text[:800])
    check("Umsatz mit diesem Kunden jetzt 111,50 €", "insgesamt: 111,50" in text)

    # Stornieren: Bonus 100, dann zurücknehmen
    pg.click("text=Credits gutschreiben")
    pg.select_option("#gsArt", "bonus")
    pg.select_option("#gsMenge", "frei")
    pg.fill("#gsFrei", "100")
    pg.fill("#gsGrund", "Klicktest Storno")
    pg.click("#gsForm button[type=submit]")
    pg.click("#gsBuchen")
    pg.wait_for_function("() => !document.querySelector('#gsDialog').open", timeout=10000)
    pg.wait_for_timeout(600)
    zeile = pg.locator("#kundeInhalt li", has_text="Klicktest Storno")
    zeile.locator("button", has_text="Stornieren").click()
    check("Storno-Dialog nennt die zurückgenommenen Credits",
          "100 von 100 Credits" in pg.locator("#stornoFolgen").inner_text(), pg.locator("#stornoFolgen").inner_text())
    axe(pg, "Storno-Dialog")
    pg.click("#stornoForm button[type=submit]")
    check("Storno ohne Grund: Fehlermeldung", "Grund" in pg.locator("#stornoFehler").inner_text())
    pg.fill("#stornoGrund", "Klicktest falsche Menge")
    pg.click("#stornoForm button[type=submit]")
    pg.wait_for_function("() => !document.querySelector('#stornoDialog').open", timeout=10000)
    pg.wait_for_timeout(600)
    zeile = pg.locator("#kundeInhalt li", has_text="Klicktest Storno")
    check("Storniert, kein Knopf mehr an der Buchung",
          "storniert, zählt nicht zum Umsatz" in zeile.inner_text() and zeile.locator("button").count() == 0, zeile.inner_text())
    check("Umsatz unverändert 111,50 € (Bonus zählte nie)", "insgesamt: 111,50" in pg.locator("main").inner_text())

    # Sperren mit Rückfrage, dann wieder entsperren
    pg.click("text=Konto sperren")
    check("Sperr-Rückfrage erklärt die Folgen", "nicht mehr anmelden" in pg.locator("#sperrText").inner_text())
    axe(pg, "Sperr-Dialog")
    pg.click("#sperrJa")
    pg.wait_for_selector("text=Konto entsperren", timeout=10000)
    check("Status: gesperrt", "Status: gesperrt" in pg.locator("main").inner_text())
    pg.click("text=Konto entsperren")
    pg.wait_for_selector("text=Konto sperren", timeout=10000)
    check("Wieder aktiv", "Status: aktiv" in pg.locator("main").inner_text())

    # Abo auf Rechnung
    pg.click("text=Abo zuweisen oder ändern")
    axe(pg, "Abo-Dialog")
    pg.select_option("#aboPlan", "single")
    pg.select_option("#aboLaufzeit", "6")
    pg.select_option("#aboArt", "verkauf")
    check("Abo-Betrag vorbelegt (59,70)", pg.input_value("#aboBetrag") == "59,70", pg.input_value("#aboBetrag"))
    pg.click("#aboForm button[type=submit]")
    check("Abo-Bestätigung nennt Plan und Betrag", "Single" in pg.locator("#aboSatz").inner_text()
          and "59,70" in pg.locator("#aboSatz").inner_text())
    pg.click("#aboSpeichern")
    pg.wait_for_function("() => !document.querySelector('#aboDialog').open", timeout=15000)
    pg.wait_for_timeout(600)
    text = pg.locator("main").inner_text()
    check("Überblick: Single auf Rechnung", "Single auf Rechnung" in text)
    check("Abo-Buchung erscheint", "Abo Single, 6 Monate" in text and "59,70" in text)

    # Limit-Dialog
    pg.click("text=Limit ändern")
    axe(pg, "Limit-Dialog")
    pg.keyboard.press("Escape")

    # Umsatz-Seite
    pg.goto(f"{BASE}/verwaltung/umsatz", wait_until="networkidle")
    pg.wait_for_timeout(800)
    check("Umsatz: vier Kennzahlen", pg.locator(".umsatz-zahl").count() == 4)
    check("Umsatz: Testbuchungen im laufenden Monat", "Verwaltung Testkunde" in pg.locator("#umsatzBuchungen").inner_text())
    check("Umsatz: Kundenname ist Link", pg.locator("#umsatzBuchungen a", has_text="Verwaltung Testkunde").count() >= 1)
    href = pg.locator("#dlExcel").get_attribute("href")
    r = pg.request.get(BASE + href)
    check("Excel-Download: 200 und XLSX", r.status == 200 and r.body()[:2] == b"PK", (r.status, href))
    r = pg.request.get(BASE + pg.locator("#dlCsv").get_attribute("href"))
    check("CSV-Download: 200, enthält Testkunde", r.status == 200 and "Verwaltung Testkunde" in r.body().decode("utf-8-sig"))
    axe(pg, "Umsatz")
    optionen = pg.locator("#wahlZeitraum option").all_inner_texts()
    check("Zeitraum-Liste: Jahr zuerst, jede Option mit Betrag",
          optionen and optionen[0].startswith("Ganzes Jahr") and all("€" in o for o in optionen), optionen[:4])
    check("Keine getrennten Felder Jahr/Monat und keine Monatsliste mehr",
          pg.locator("#wahlJahr, #wahlMonat, #umsatzMonate").count() == 0)
    pg.select_option("#wahlZeitraum", index=0)
    pg.click("#zeitraumForm button[type=submit]")
    pg.wait_for_timeout(800)
    check("Ganzes Jahr: Überschrift passt",
          "ganzen Jahr" in pg.locator("#h-buchungen").inner_text(), pg.locator("#h-buchungen").inner_text())
    check("Nach „Anzeigen“: Fokus auf den Buchungen", fokus_id(pg) == "h-buchungen", fokus_id(pg))
    check("Jahres-Download in der Adresse", "monat=" not in (pg.locator("#dlExcel").get_attribute("href") or ""))

    # API und Einstellungen
    pg.goto(f"{BASE}/verwaltung/api", wait_until="networkidle")
    pg.wait_for_timeout(800)
    check("API-Seite lädt", "Konten mit API-Schlüssel" in pg.locator("main").inner_text())
    axe(pg, "API")
    pg.goto(f"{BASE}/verwaltung/einstellungen", wait_until="networkidle")
    pg.wait_for_timeout(800)
    check("Einstellungen: Administratoren gelistet", pg.locator("#adminListe li").count() >= 1)
    axe(pg, "Einstellungen")

    # Seiten der Kundenliste
    pg.goto(f"{BASE}/verwaltung/kunden", wait_until="networkidle")
    pg.wait_for_timeout(800)
    if pg.locator("#kundenSeiten").is_visible():
        check("Seite 1: „Vorherige Seite“ gesperrt", pg.locator("#seiteZurueck").is_disabled())
        pg.click("#seiteVor")
        pg.wait_for_timeout(800)
        check("Nächste Seite: Fokus auf der Überschrift mit „Seite 2“",
              fokus_id(pg) == "kundenStatus" and "Seite 2" in pg.locator("#kundenStatus").inner_text(),
              (fokus_id(pg), pg.locator("#kundenStatus").inner_text()))
        check("Höchstens 25 Kunden je Seite", pg.locator("#kundenListe li").count() <= 25)
    else:
        print("(nur eine Seite Kunden — Blättern nicht prüfbar)")

    # Aufräumen: Testkonto löschen (Buchungen bleiben absichtlich stehen)
    pg.goto(kunde_url, wait_until="networkidle")
    pg.wait_for_timeout(800)
    pg.click("text=Konto löschen")
    pg.click("#loeschJa")
    pg.wait_for_url("**/verwaltung/kunden**", timeout=10000)
    check("Konto gelöscht, zurück in der Kundenliste", "/verwaltung/kunden" in pg.url)
    r = pg.request.get(f"{BASE}/api/admin/umsatz")
    rest = [x for x in r.json()["buchungen"] if x["kunde_email"] == test_mail]
    check("Buchungen des gelöschten Kontos bleiben erhalten (ohne Kontoverweis)",
          rest and all(x["konto_user_id"] is None for x in rest), rest[:1])
    check("Keine JavaScript-Fehler", not js_fehler, js_fehler[:3])
    b.close()

print(f"\n{ok} ok, {fehler} fehlgeschlagen — Testkonto {test_mail} (id {kid})")
sys.exit(1 if fehler else 0)
