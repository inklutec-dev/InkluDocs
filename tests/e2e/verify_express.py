#!/usr/bin/env python3
"""E2E EXPRESS-SERVICE Stufe 1 (05.10.2026) im Staging-Container ueber HTTP: ganzer Ablauf (Upload ohne Projekt, Auswahl,
Bestellen mit Vormerkung, Uebernehmen, Rueckfrage, Antwort, Ergebnis mit veraPDF, Liefern, Download, Nachweis-PDF,
ZIP, Storno) und die Sicherheitsfaelle: fremde Kunden (IDOR), Nur-Einsicht-Admin, Express-Bearbeiter ohne Admin-Recht,
Upload-Pruefungen, Doppel-Bestellung, Doppel-Lieferung, Storno nach Lieferung.
Korrekturrunde 05.10.2026: Bestellen nur mit Korb, Summe und Fassung (409 bei geaendertem Preis oder altem Schluessel),
kaputter JSON-Koerper 400, Fehler der Einstellungen mit Feld, no-store auf den Seiten, Projektliste nur mit PDF-Projekten,
ZIP ungepackt (ZIP_STORED), automatische Pruefung beim Hochladen fertig, bevor die Antwort kommt.
    docker cp tests/e2e/verify_express.py inkludocs-staging:/tmp/ && docker exec inkludocs-staging python3 /tmp/verify_express.py
Testkonten auf .invalid (Mails werden unterdrueckt); Team-Mails gehen waehrend des Tests an eine .invalid-Adresse.
Alles wird am Ende entfernt, die Einstellungen werden zurueckgesetzt."""
import io
import json
import time
import os
import shutil
import sys
import zipfile

sys.path.insert(0, "/app")
os.chdir("/app")

import fitz  # noqa: E402
import httpx  # noqa: E402

from database import create_user, delete_user_data, get_db, get_user_by_email  # noqa: E402

BASE = "http://127.0.0.1:8001"
PW = "Express-Test-2026!x"
DOM = "express-test.invalid"
KUNDE, FREMD, VOLL, SICHT, BEARB = (f"{n}@{DOM}" for n in ("kunde", "fremd", "voll", "sicht", "bearbeiter"))
ok = fehler = 0


def check(name, cond, extra=""):
    global ok, fehler
    if cond:
        ok += 1
        print(f"  OK  {name}")
    else:
        fehler += 1
        print(f"FEHLT {name} {str(extra)[:300]}")


def sql(q, *p):
    conn = get_db()
    try:
        r = [dict(x) for x in conn.execute(q, p).fetchall()]
        conn.commit()
        return r
    finally:
        conn.close()


def aufraeumen():
    for mail in (KUNDE, FREMD, VOLL, SICHT, BEARB):
        u = get_user_by_email(mail)
        if u:
            for pfad in (f"/app/data/uploads/{u['id']}", f"/app/data/results/{u['id']}"):
                shutil.rmtree(pfad, ignore_errors=True)
            delete_user_data(u["id"])


def client(mail):
    c = httpx.Client(base_url=BASE, timeout=300)
    r = c.post("/api/login", json={"email": mail, "password": PW})
    assert r.status_code == 200, f"Login {mail}: {r.status_code} {r.text[:200]}"
    c.headers["Cookie"] = "token=" + r.cookies.get("token")
    return c


def projekt_bereit(c, pid, sekunden=60):
    """Bis die Bild-Extraktion des Projekts fertig ist (Anhaengen waehrend „extracting“ lehnt /api/upload mit 409 ab)."""
    ende = time.time() + sekunden
    while time.time() < ende:
        if c.get(f"/api/projects/{pid}/status").json().get("status") != "extracting":
            return True
        time.sleep(0.5)
    return False


def pdf_bytes(seiten=2, text="Fiktives Testdokument"):
    d = fitz.open()
    for i in range(seiten):
        d.new_page().insert_text((72, 72), f"{text}, Seite {i + 1}")
    b = d.tobytes()
    d.close()
    return b


aufraeumen()
einst_vorher = sql("SELECT value FROM system_kv WHERE key = 'express_einstellungen'")
for mail in (KUNDE, FREMD, VOLL, SICHT, BEARB):
    create_user(mail, PW, "Testperson Express (fiktiv)")
k_id, f_id, b_id = (get_user_by_email(m)["id"] for m in (KUNDE, FREMD, BEARB))
sql("UPDATE users SET is_admin = 1, admin_level = 'full' WHERE email = ?", VOLL)
sql("UPDATE users SET is_admin = 1, admin_level = 'view' WHERE email = ?", SICHT)

try:
    kunde, fremd, voll, sicht, bearb = (client(m) for m in (KUNDE, FREMD, VOLL, SICHT, BEARB))
    EINST = {"preise": {"aufbereiten": "50", "pruefen": "25"}, "frist_stunden": "48", "max_seiten_auftrag": "500",
             "max_dokumente_auftrag": "50", "team_mail": f"team@{DOM}",
             # Grundpreis je Dokument hier 0: die Zahlen dieser Reihe rechnen mit 50 je Seite. Den Standard (50 je Seite
             # + 100 je Dokument, Runde 7) prueft Abschnitt B2.
             "grundpreise": {"aufbereiten": "0"},
             # Anzeige-Modi fest fuer diese Reihe (auf Staging kann die Verwaltung sie umgestellt haben; am Ende wird der
             # vorige Stand wiederhergestellt)
             "korb_knopf": True, "korb_navigation": "immer"}
    r = voll.post("/api/admin/express/einstellungen", json=EINST)
    check("Voll-Admin setzt Einstellungen (Team-Mail auf Testadresse)", r.status_code == 200, r.text)
    r = voll.post("/api/admin/express/einstellungen", json=dict(EINST, frist_stunden="bald"))
    d = r.json().get("detail") or {}
    check("Einstellungen: Fehler nennt Feld und Beschriftung", r.status_code == 400 and d.get("feld") == "frist_stunden"
          and d.get("text", "").startswith("Lieferfrist in Stunden"), r.text)
    e = voll.get("/api/admin/express/einstellungen").json()
    check("Einstellungen: nur eingeschaltete Leistungen („Nur prüfen“ aus), Aufbewahrung 0", [l["schluessel"] for l in e["leistungen"]] == ["aufbereiten"]
          and e["aufbewahrung_tage"] == 0, e)

    print("== A. Seiten und Stand ==")
    for pfad in ("/express", "/express/bedingungen"):
        r = kunde.get(pfad)
        check(f"Seite {pfad} 200", r.status_code == 200 and ("Express" in r.text or "Aufträge" in r.text), r.status_code)
    check("Seite /express: Cache-Control no-store (Zurueck-Speicher)", kunde.get("/express").headers.get("cache-control") == "no-store")
    check("Bedingungen: Datenschutz und Mail verlinkt", 'href="/datensicherheit"' in kunde.get("/express/bedingungen").text
          and 'href="mailto:support@inkludocs.de"' in kunde.get("/express/bedingungen").text)
    r = kunde.get("/api/express/stand")
    st = r.json()
    check("Stand: Preise, Frist, Texte", r.status_code == 200 and st["preise"]["aufbereiten"] == 50 and st["frist_stunden"] == 48
          and set(st["texte"]) == {"bedingungen"}, st)
    check("Stand: Leistungen und Dateitypen aus der Liste (nur Aufbereiten)", [l["schluessel"] for l in st["leistungen"]] == ["aufbereiten"]
          and [t["schluessel"] for t in st["dateitypen"]] == ["pdf"], (st.get("leistungen"), st.get("dateitypen")))
    check("/api/me meldet express", kunde.get("/api/me").json()["user"]["express"] is True)
    # Zusatz 05.10.2026: Navigations-Eintrag „Express-Warenkorb“ und Knopf am Dokument (Einstellungen, ohne Codeaenderung)
    check("/api/me: Express-Warenkorb (Modus immer, leer)", kunde.get("/api/me").json()["user"]["express_warenkorb"] == {"modus": "immer", "dokumente": 0})
    r = kunde.get("/express/warenkorb")
    check("Seite /express/warenkorb 200, Titel „Express-Warenkorb“, no-store", r.status_code == 200 and "<title>Express-Warenkorb" in r.text
          and r.headers.get("cache-control") == "no-store", r.status_code)
    check("Projektseite: window.FUNKTIONEN mit express_korb_knopf", '"express_korb_knopf": true' in kunde.get("/app").text)
    r = voll.post("/api/admin/express/einstellungen", json=dict(EINST, korb_knopf=False, korb_navigation="aus"))
    check("Einstellungen: Knopf aus, Navigation aus", r.status_code == 200 and r.json()["einstellungen"]["korb_navigation"] == "aus"
          and kunde.get("/api/me").json()["user"]["express_warenkorb"] is None and '"express_korb_knopf": false' in kunde.get("/app").text, r.text[:200])
    r = voll.post("/api/admin/express/einstellungen", json=dict(EINST, korb_navigation="manchmal"))
    check("Einstellungen: unbekannter Navigations-Modus 400 mit Feld", r.status_code == 400 and r.json()["detail"].get("feld") == "korb_navigation")
    voll.post("/api/admin/express/einstellungen", json=dict(EINST, korb_knopf=True, korb_navigation="immer"))

    print("== B. Hochladen nur im Projekt (Runde 7) + Pruefungen ==")
    r = kunde.post("/api/express/warenkorb/hochladen", files={"file": ("Jahresbericht fiktiv.pdf", pdf_bytes(3), "application/pdf")})
    check("Hochladen ohne Projekt entfallen: 410 mit Hinweis aufs Projekt", r.status_code == 410 and "Projekt" in r.text
          and not sql("SELECT id FROM projects WHERE user_id = ?", k_id), r.text[:200])
    r = kunde.post("/api/upload", files={"file": ("Jahresbericht fiktiv.pdf", pdf_bytes(3), "application/pdf")})
    check("PDF im Projekt hochgeladen", r.status_code == 200 and r.json().get("project_id"), r.text[:200])
    pid, d1 = r.json()["project_id"], r.json()["document_id"]
    check("Projekt fertig eingelesen", projekt_bereit(kunde, pid))
    r = kunde.post("/api/upload", files={"file": ("Flyer fiktiv.pdf", pdf_bytes(1), "application/pdf")}, data={"project_id": str(pid)})
    check("Zweite PDF im selben Projekt", r.status_code == 200 and r.json()["project_id"] == pid, r.text[:200])
    d2 = r.json()["document_id"]
    projekt_bereit(kunde, pid)
    r = kunde.post("/api/upload", files={"file": ("notiz.txt", b"kein pdf", "text/plain")}, data={"project_id": str(pid)})
    check("Falscher Dateityp im PDF-Projekt: positiv formulierte Meldung", r.status_code == 400 and r.json()["detail"] == "Bitte wähle eine PDF-Datei aus.", r.text)
    r = kunde.post("/api/express/warenkorb/dokumente", json={"document_ids": [d1, d2]})
    check("Aus dem Projekt in die Auswahl", r.status_code == 200 and r.json()["hinzugefuegt"] == 2, r.text[:200])
    korb = kunde.get("/api/express/stand").json()["warenkorb"]
    check("Auswahl: 2 Dokumente, 4 Seiten, 200 Credits (Grundpreis hier 0)", (korb["dokumente"], korb["seiten"], korb["credits"]) == (2, 4, 200), korb)
    docs = kunde.get(f"/api/express/projekte/{pid}/dokumente").json()["dokumente"]
    check("Dokumente des Projekts: beide schon in der Auswahl", all(d["im_warenkorb"] for d in docs), docs)

    print("== B2. Preis: 50 je Seite plus 100 je Dokument (Runde 7, Michaels Richtpreis) ==")
    e = voll.post("/api/admin/express/einstellungen", json=dict(EINST, grundpreise={"aufbereiten": "100"})).json()["einstellungen"]
    check("Einstellungen: Grundpreis je Dokument gespeichert, kein Platzhalter-Kennzeichen mehr",
          e["grundpreise"]["aufbereiten"] == 100 and e["leistungen"][0]["grundpreis"] == 100 and "preise_festgelegt" not in e, e)
    st = kunde.get("/api/express/stand").json()
    w = st["warenkorb"]
    check("Aufstellung: 4 × 50 + 2 × 100 = 400 Credits, Teilsummen", (w["credits"], w["credits_seiten"], w["credits_grund"]) == (400, 200, 200)
          and [(p["preis_seite"], p["grundpreis"]) for p in w["positionen"]] == [(50, 100), (50, 100)], w)
    r = voll.post("/api/admin/express/einstellungen", json=dict(EINST, grundpreise={"aufbereiten": "-3"}))
    check("Grundpreis -3: 400 mit Feld grundpreis_aufbereiten", r.status_code == 400 and r.json()["detail"].get("feld") == "grundpreis_aufbereiten", r.text)
    for wert in ("1.5", "100.00"):
        r = voll.post("/api/admin/express/einstellungen", json=dict(EINST, grundpreise={"aufbereiten": wert}))
        check(f"Grundpreis „{wert}“: 400 mit Feld statt still umgedeutet (Runde 8)", r.status_code == 400
              and r.json()["detail"].get("feld") == "grundpreis_aufbereiten", r.text)
    r = voll.post("/api/admin/express/einstellungen", json=dict(EINST, max_seiten_auftrag="1.000"))
    check("Tausenderpunkt „1.000“ bleibt erlaubt", r.status_code == 200 and r.json()["einstellungen"]["max_seiten_auftrag"] == 1000, r.text[:200])
    voll.post("/api/admin/express/einstellungen", json=EINST)
    pos2 = korb["positionen"][1]["id"]
    r = kunde.post(f"/api/express/warenkorb/positionen/{pos2}/leistung", json={"leistung": "pruefen"})
    check("„Nur prüfen“ ist abgeschaltet: 400, Auswahl unverändert 200 Credits", r.status_code == 400
          and kunde.get("/api/express/stand").json()["warenkorb"]["credits"] == 200, r.text)
    for koerper in ("{kaputt", "[1]"):
        r = kunde.post(f"/api/express/warenkorb/positionen/{pos2}/leistung", content=koerper, headers={"Content-Type": "application/json"})
        check(f"Kaputter JSON-Koerper {koerper!r}: 400 statt 500", r.status_code == 400, r.status_code)
    # Projektliste: nur PDF-Projekte mit Dokumenten (Word-Projekt und leeres Projekt erscheinen nicht).
    sql("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type) VALUES (?, 'w', '', 'Word fiktiv', 'word', 'docx')", k_id)
    sql("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type) VALUES (?, 'l', '', 'Leer fiktiv', 'pdf', 'pdf')", k_id)
    pl = kunde.get("/api/express/projekte").json()["projekte"]
    check("Projektliste: nur PDF-Projekte mit Dokumenten, mit Anlagedatum", [p["id"] for p in pl] == [pid] and pl[0]["angelegt_am"], pl)

    print("== C. Fremde Kunden (IDOR) ==")
    check("Fremder sieht Projekt-Dokumente nicht (404)", fremd.get(f"/api/express/projekte/{pid}/dokumente").status_code == 404)
    r = fremd.post("/api/express/warenkorb/dokumente", json={"document_ids": [docs[0]["id"]]})
    check("Fremder kann Dokument nicht in seine Auswahl legen (404)", r.status_code == 404, r.text)
    check("Fremder kann Position nicht aendern (404)", fremd.post(f"/api/express/warenkorb/positionen/{pos2}/leistung", json={"leistung": "aufbereiten"}).status_code == 404)
    check("Fremder kann Position nicht entfernen (404)", fremd.delete(f"/api/express/warenkorb/positionen/{pos2}").status_code == 404)
    check("Ohne Anmeldung: 401", httpx.get(BASE + "/api/express/stand").status_code == 401)

    print("== D. Bestellen ==")

    def mit_korb(daten):
        """Wie die Seite: Korb, angezeigte Summe und Fassung mitschicken."""
        w = kunde.get("/api/express/stand").json()["warenkorb"]
        return dict(daten, korb_id=w["id"], erwartete_credits=w["credits"], fassung=w["fassung"])
    bestellung = mit_korb({"ansprechpartner": "Kim Muster (fiktiv)", "telefon": "+49 40 0000", "hinweise": "Seite 2 bitte genau",
                           "bedingungen": True, "idempotenz": "e2e-express-0001"})
    r = kunde.post("/api/express/bestellen", json=dict(bestellung, bedingungen=False))
    check("Ohne Häkchen „Bedingungen“: 400 mit Feld", r.status_code == 400 and r.json()["detail"].get("feld") == "bedingungen", r.text)
    r = kunde.post("/api/express/bestellen", content="{kaputt", headers={"Content-Type": "application/json"})
    check("Bestellen mit kaputtem JSON: 400", r.status_code == 400, r.status_code)
    r = kunde.post("/api/express/bestellen", json=bestellung)
    check("Zu wenig Guthaben: 402 mit Zahlen", r.status_code == 402 and r.json()["detail"]["preis"] == 200, r.text)
    sql("INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) VALUES (?, 1000, 1000, 'admin', 'Express-Test', NULL)", k_id)
    # Preis aendert sich nach dem Anzeigen: nicht bestellen, 409 mit dem neuen Betrag (§ 312j BGB).
    voll.post("/api/admin/express/einstellungen", json=dict(EINST, preise={"aufbereiten": "60", "pruefen": "25"}))
    r = kunde.post("/api/express/bestellen", json=dict(bestellung, idempotenz="e2e-express-0000"))
    d = r.json().get("detail") or {}
    check("Geaenderter Preis: 409 mit neuem Betrag, nichts bestellt", r.status_code == 409 and d.get("veraltet") and d.get("neu_credits") == 240
          and not sql("SELECT id FROM express_auftraege WHERE user_id = ? AND status NOT IN ('entwurf', 'bestellung')", k_id), r.text)
    voll.post("/api/admin/express/einstellungen", json=EINST)
    verf_vorher = kunde.get("/api/express/stand").json()["guthaben"]
    r = kunde.post("/api/express/bestellen", json=bestellung)
    aid = r.json().get("auftrag_id")
    check("Bestellt", r.status_code == 200 and aid and not r.json()["schon_bestellt"], r.text)
    r = kunde.post("/api/express/bestellen", json=bestellung)
    check("Doppelklick: derselbe Auftrag, nicht zweimal", r.status_code == 200 and r.json()["auftrag_id"] == aid and r.json()["schon_bestellt"], r.text)
    check("Genau ein Auftrag", len(sql("SELECT id FROM express_auftraege WHERE user_id = ? AND status != 'entwurf'", k_id)) == 1)
    st = kunde.get("/api/express/stand").json()
    check("Guthaben um 200 gemindert (vorgemerkt)", st["guthaben"] == verf_vorher - 200, (verf_vorher, st["guthaben"]))
    check("/api/me: 200 vorgemerkt", kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 200)
    zu = kunde.get(f"/api/express/auftraege/{aid}").json()["zustimmung"]
    check("Zustimmung: Fassung -3, ein Häkchen (Bedingungen)", zu["fassung"] == "2026-10-05-entwurf-3" and zu["bearbeitung"] == ""
          and zu["bedingungen"].startswith("Ich akzeptiere"), zu)
    check("Noch nichts abgebucht", sql("SELECT COUNT(*) AS n FROM usage_events WHERE quelle = 'express'")[0]["n"] == 0)
    a = kunde.get(f"/api/express/auftraege/{aid}").json()
    check("Auftragsuebersicht: eingegangen, vorgemerkt, Einverstaendnis mit Zeitpunkt",
          a["status"] == "neu" and a["credits_stand"] == "vorgemerkt" and a["zustimmung"]["am"], a)
    check("Seite Auftragsuebersicht 200", kunde.get(f"/express/auftrag/{aid}").status_code == 200)
    check("Fremder: Auftrag 404", fremd.get(f"/api/express/auftraege/{aid}").status_code == 404)
    check("Fremder: Nachweis 404", fremd.get(f"/api/express/auftraege/{aid}/nachweis.pdf").status_code == 404)
    check("Fremder: Antwort 404", fremd.post(f"/api/express/auftraege/{aid}/antwort", json={"text": "x"}).status_code == 404)

    print("== E. Rechte in der Verwaltung ==")
    for pfad in ("/api/admin/express/auftraege", f"/api/admin/express/auftraege/{aid}", "/api/admin/express/einstellungen"):
        check(f"Kunde: {pfad} 403", kunde.get(pfad).status_code == 403)
    check("Nur-Einsicht liest die Liste", sicht.get("/api/admin/express/auftraege").status_code == 200)
    check("Nur-Einsicht darf nicht uebernehmen (403)", sicht.post(f"/api/admin/express/auftraege/{aid}/uebernehmen").status_code == 403)
    check("Nur-Einsicht darf keine Einstellungen (403)", sicht.post("/api/admin/express/einstellungen", json={}).status_code == 403)
    check("Bearbeiter vorher: kein Zugriff (403)", bearb.get("/api/admin/express/auftraege").status_code == 403)
    r = sicht.post("/api/admin/express/bearbeiter", json={"email": BEARB})
    check("Nur-Einsicht vergibt kein Bearbeiter-Recht (403)", r.status_code == 403)
    r = voll.post("/api/admin/express/bearbeiter", json={"email": BEARB})
    check("Voll-Admin macht Konto zum Express-Bearbeiter", r.status_code == 200, r.text)
    check("Bearbeiter liest die Liste", bearb.get("/api/admin/express/auftraege").status_code == 200)
    for pfad in ("/api/admin/kunden", "/api/admin/umsatz", "/api/admin/ki-kosten", f"/api/admin/users/{k_id}/report", "/api/admin/express/bearbeiter"):
        check(f"Bearbeiter: {pfad} 403", bearb.get(pfad).status_code == 403, bearb.get(pfad).status_code)
    check("Bearbeiter: keine Einstellungen aendern (403)", bearb.post("/api/admin/express/einstellungen", json={}).status_code == 403)
    va = bearb.get(f"/api/admin/express/auftraege/{aid}").json()
    check("Bearbeiter sieht keine Konto-IDs", "user_id" not in va and "konto_user_id" not in va, list(va)[:8])
    seite = bearb.get("/verwaltung/express").text
    check("Bearbeiter: Navigation nur „Express-Aufträge“", "/verwaltung/express" in seite and "/verwaltung/umsatz" not in seite)
    check("Bearbeiter in /api/me", bearb.get("/api/me").json()["user"]["express_bearbeiter"] is True)

    print("== F. Ablauf ==")
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/uebernehmen")
    check("Uebernommen -> in Arbeit", r.status_code == 200 and r.json()["status"] == "in_arbeit"
          and r.json()["bearbeiter_name"], r.text)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/rueckfrage", json={"text": ""})
    check("Leere Rueckfrage: 400", r.status_code == 400)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/rueckfrage", json={"text": "Ist Seite 3 ein Formular? (Test)"})
    check("Rueckfrage gestellt", r.status_code == 200 and r.json()["status"] == "rueckfrage", r.text)
    a = kunde.get(f"/api/express/auftraege/{aid}").json()
    check("Kunde sieht die Rueckfrage, ohne Bearbeiter-Namen", a["rueckfrage_offen"]
          and any(v["art"] == "rueckfrage" and v["von_name"] == "" for v in a["verlauf"]), a["verlauf"])
    r = kunde.post(f"/api/express/auftraege/{aid}/antwort", json={"text": "Nein, nur Text (Test)"})
    check("Antwort -> wieder in Arbeit", r.status_code == 200 and r.json()["status"] == "in_arbeit", r.text)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/notiz", json={"text": "intern: Test"})
    check("Interne Notiz", r.status_code == 200)
    check("Kunde sieht die interne Notiz nicht", "intern: Test" not in json.dumps(kunde.get(f"/api/express/auftraege/{aid}").json(), ensure_ascii=False))
    va = bearb.get(f"/api/admin/express/auftraege/{aid}").json()
    p1, p2 = va["positionen"]
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/liefern", json={})
    check("Liefern ohne Ergebnis: 400 mit Liste", r.status_code == 400 and r.json()["detail"]["fehlt"], r.text)
    r = bearb.get(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/original")
    check("Original herunterladen", r.status_code == 200 and r.content.startswith(b"%PDF") and "attachment" in r.headers.get("content-disposition", ""), r.status_code)
    r = bearb.get(f"/api/admin/express/auftraege/{aid}/originale.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content)) if r.status_code == 200 else None
    check("Originale als ZIP (2 Dateien, keine Pfade)", z and len(z.namelist()) == 2 and all("/" not in n for n in z.namelist()), z and z.namelist())
    check("ZIP ungepackt gespeichert (PDFs sind schon komprimiert)", z and all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist()))
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis", files={"file": ("x.pdf", b"kein pdf", "application/pdf")})
    check("Ergebnis: keine PDF -> 400", r.status_code == 400, r.text)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/skript", files={"file": ("x.pdf", pdf_bytes(1), "application/pdf")})
    check("Unbekannte Dateiart -> 404", r.status_code == 404)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis",
                   files={"file": ("../../../etc/ergebnis.pdf", pdf_bytes(3, "Aufbereitet"), "application/pdf")})
    pos = next(p for p in r.json()["positionen"] if p["id"] == p1["id"]) if r.status_code == 200 else {}
    check("Ergebnis hochgeladen, veraPDF gelaufen (nicht mehr „laeuft“)", r.status_code == 200 and pos.get("ergebnis_da") and pos.get("verapdf")
          and not pos["verapdf"].get("laeuft") and pos.get("pruef_name") == "veraPDF", r.text[:300])
    check("Hochladefelder je Leistung: Ergebnis PDF, Pruefbericht PDF", [t["schluessel"] for t in pos.get("ergebnis_typen", [])] == ["pdf"]
          and [t["schluessel"] for t in pos.get("bericht_typen", [])] == ["pdf"], pos)
    check("Dateiname ohne Pfad", pos.get("ergebnis_name") == "ergebnis.pdf", pos.get("ergebnis_name"))
    pfad = sql("SELECT ergebnis_pfad FROM express_positionen WHERE id = ?", p1["id"])[0]["ergebnis_pfad"]
    check("Ablage im Auftragsordner", pfad.startswith(f"/app/data/results/{k_id}/_express/{aid}/") and ".." not in pfad, pfad)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p2['id']}/ergebnis", files={"file": ("flyer.pdf", pdf_bytes(1, "Aufbereitet"), "application/pdf")})
    check("Ergebnis fuer das zweite Dokument", r.status_code == 200, r.text[:200])
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p2['id']}/bericht", files={"file": ("bericht.pdf", pdf_bytes(1, "Pruefbericht"), "application/pdf")})
    check("Pruefbericht (freiwillig) dazu", r.status_code == 200, r.text[:200])
    check("Kunde: Ergebnis vor der Lieferung 404", kunde.get(f"/api/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis").status_code == 404)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/liefern", json={})
    if r.status_code == 409 and r.json().get("detail", {}).get("nachfrage"):
        check("veraPDF-Abweichungen: Nachfrage statt Lieferung", True)
        check("Nichts abgebucht vor der Bestaetigung", sql("SELECT COUNT(*) AS n FROM usage_events WHERE quelle = 'express'")[0]["n"] == 0)
        r = bearb.post(f"/api/admin/express/auftraege/{aid}/liefern", json={"trotz_befunden": True})
    check("Geliefert", r.status_code == 200 and r.json()["status"] == "geliefert", r.text[:300])
    ev = sql("SELECT aktion, credits, konto_user_id FROM usage_events WHERE quelle = 'express' AND user_id = ?", k_id)
    check("Abgebucht: 200 Credits (aufbereiten) auf den Topf der Bestellung",
          sorted((e["aktion"], e["credits"]) for e in ev) == [("express_aufbereiten", 200)]
          and all(e["konto_user_id"] == k_id for e in ev), ev)
    check("Vormerkung weg", kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 0)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/liefern", json={"trotz_befunden": True})
    check("Zweites Liefern: 409, nichts doppelt", r.status_code == 409 and len(sql("SELECT id FROM usage_events WHERE quelle = 'express' AND user_id = ?", k_id)) == 1)
    check("Storno nach Lieferung: 409", bearb.post(f"/api/admin/express/auftraege/{aid}/stornieren", json={"grund": "Test"}).status_code == 409)
    check("Upload nach Lieferung: 409", bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis",
                                                  files={"file": ("x.pdf", pdf_bytes(1), "application/pdf")}).status_code == 409)
    r = kunde.get(f"/api/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis")
    check("Kunde laedt das Ergebnis", r.status_code == 200 and r.content.startswith(b"%PDF")
          and "barrierefrei" in r.headers.get("content-disposition", ""), r.headers.get("content-disposition"))
    check("Kunde: Pruefbericht", kunde.get(f"/api/express/auftraege/{aid}/positionen/{p2['id']}/bericht").status_code == 200)
    pk = next(p for p in kunde.get(f"/api/express/auftraege/{aid}").json()["positionen"] if p["id"] == p1["id"])
    check("Kunde sieht nur bestanden ja/nein, keine Technik-Zusammenfassung", set((pk.get("verapdf") or {"bestanden": 1}).keys()) == {"bestanden"}, pk)
    check("Kunde: Original ist kein Kunden-Download (404)", kunde.get(f"/api/express/auftraege/{aid}/positionen/{p1['id']}/original").status_code == 404)
    check("Fremder: Ergebnis 404", fremd.get(f"/api/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis").status_code == 404)
    r = kunde.get(f"/api/express/auftraege/{aid}/nachweis.pdf")
    if r.status_code == 200:
        check("Nachweis als PDF (PDF/UA geprueft)", r.content.startswith(b"%PDF") and r.headers["content-type"] == "application/pdf")
        with fitz.open(stream=r.content, filetype="pdf") as d:
            check("Nachweis: Titel und Text", "Express-Auftrag" in (d.metadata.get("title") or "") and "keine Rechnung" in d[0].get_text(), d.metadata)
    else:
        check("Nachweis als PDF", False, f"{r.status_code} {r.text[:200]}")
    a = kunde.get("/api/express/auftraege").json()["auftraege"]
    check("Meine Auftraege: geliefert", a and a[0]["status"] == "geliefert", a)

    print("== G. Storno gibt frei ==")
    r = kunde.post("/api/express/warenkorb/dokumente", json={"document_ids": [docs[0]["id"]]})
    check("Neue Auswahl", r.status_code == 200 and r.json()["hinzugefuegt"] == 1, r.text)
    r = kunde.post("/api/express/bestellen", json=mit_korb(dict(bestellung, idempotenz="e2e-express-0001")))
    check("Alter Schluessel mit neuem Korb (Zurueck-Speicher): 409, nicht still der alte Auftrag",
          r.status_code == 409 and (r.json().get("detail") or {}).get("veraltet"), r.text[:200])
    r = kunde.post("/api/express/bestellen", json=mit_korb(dict(bestellung, idempotenz="e2e-express-0002")))
    aid2 = r.json().get("auftrag_id")
    check("Zweiter Auftrag, 150 vorgemerkt", r.status_code == 200 and kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 150, r.text)
    check("Storno ohne Grund: 400", voll.post(f"/api/admin/express/auftraege/{aid2}/stornieren", json={"grund": ""}).status_code == 400)
    r = voll.post(f"/api/admin/express/auftraege/{aid2}/stornieren", json={"grund": "Test-Storno"})
    check("Storniert, Vormerkung frei, nichts abgebucht", r.status_code == 200 and kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 0
          and len(sql("SELECT id FROM usage_events WHERE quelle = 'express' AND user_id = ?", k_id)) == 1, r.text[:200])
    check("Kunde sieht Storno mit Grund", kunde.get(f"/api/express/auftraege/{aid2}").json()["storno_grund"] == "Test-Storno")

    print("== G2. Umbenennen und Löschen (Michael Karbe, Punkt 1) ==")
    r = kunde.post(f"/api/express/auftraege/{aid}/name", json={"name": "Jahresberichte (fiktiv)"})
    check("Umbenennen: Name in Auftrag und Liste", r.status_code == 200 and r.json()["auftrag_name"] == "Jahresberichte (fiktiv)"
          and kunde.get("/api/express/auftraege").json()["auftraege"][-1]["auftrag_name"] == "Jahresberichte (fiktiv)", r.text[:200])
    check("Umbenennen fremder Auftrag: 404", fremd.post(f"/api/express/auftraege/{aid}/name", json={"name": "x"}).status_code == 404)
    check("Umbenennen kaputter JSON: 400", kunde.post(f"/api/express/auftraege/{aid}/name", content="[1]",
                                                      headers={"Content-Type": "application/json"}).status_code == 400)
    liste = {a["id"]: a for a in kunde.get("/api/express/auftraege").json()["auftraege"]}
    check("Liste: Dokumente je Auftrag zum Aufklappen, Löschen nur für geliefert/storniert",
          len(liste[aid]["positionen"]) == 2 and liste[aid]["loeschbar"] and liste[aid2]["loeschbar"], list(liste))
    check("Löschen fremder Auftrag: 404", fremd.delete(f"/api/express/auftraege/{aid}").status_code == 404)
    r = kunde.delete(f"/api/express/auftraege/{aid}")
    check("Gelieferten Auftrag gelöscht", r.status_code == 200, r.text[:200])
    check("Kunde: Auftrag, Downloads und Nachweis weg (404)", kunde.get(f"/api/express/auftraege/{aid}").status_code == 404
          and kunde.get(f"/api/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis").status_code == 404
          and kunde.get(f"/api/express/auftraege/{aid}/nachweis.pdf").status_code == 404
          and aid not in [a["id"] for a in kunde.get("/api/express/auftraege").json()["auftraege"]])
    check("Dateien des Auftrags gelöscht", not os.path.isdir(f"/app/data/results/{k_id}/_express/{aid}"))
    va = voll.get(f"/api/admin/express/auftraege/{aid}").json()
    check("Verwaltung: Buchungsnachweis mit Vermerk „vom Kunden gelöscht“", va.get("kunde_geloescht_am") and va["credits_gesamt"] == 200
          and va["status"] == "geliefert" and va["positionen"] == [] and va["ansprechpartner"] == "", va)
    check("Credits-Buchung bleibt", len(sql("SELECT id FROM usage_events WHERE quelle = 'express' AND user_id = ?", k_id)) == 1)

    print("== G3. Zweiter Tab nach der Bestellung, Rahmen-Schutz ==")
    kunde.post("/api/express/warenkorb/dokumente", json={"document_ids": [docs[1]["id"]]})
    alter_stand = mit_korb(dict(bestellung, idempotenz="e2e-express-tab1"))
    r = kunde.post("/api/express/bestellen", json=alter_stand)
    aid3 = r.json().get("auftrag_id")
    r = kunde.post("/api/express/bestellen", json=dict(alter_stand, idempotenz="e2e-express-tab2"))
    check("Zweiter Tab mit dem bestellten Korb: 409 veraltet mit leerem Korb (N4)", r.status_code == 409
          and r.json()["detail"].get("veraltet") and r.json()["detail"]["warenkorb"]["dokumente"] == 0, r.text[:200])
    voll.post(f"/api/admin/express/auftraege/{aid3}/stornieren", json={"grund": "Test"})
    for pfad in ("/express", "/express/warenkorb", "/app", "/api/express/stand"):
        h = kunde.get(pfad).headers
        check(f"Rahmen-Schutz {pfad}: X-Frame-Options DENY, frame-ancestors 'none' (N6)", h.get("x-frame-options") == "DENY"
              and "frame-ancestors 'none'" in h.get("content-security-policy", ""), dict(h))

    print("== G4. Vormonats-Bestellung: Startseite und Guthaben aus einer Rechnung (Nachkontrolle Runde 3, R1/R2) ==")
    import billing  # noqa: E402
    from datetime import datetime, timezone
    kunde.post("/api/express/warenkorb/dokumente", json={"document_ids": [docs[1]["id"]]})
    r = kunde.post("/api/express/bestellen", json=mit_korb(dict(bestellung, idempotenz="e2e-express-vormonat")))
    aid4 = r.json().get("auftrag_id")
    jetzt = datetime.now(timezone.utc)
    j, m = (jetzt.year, jetzt.month - 1) if jetzt.month > 1 else (jetzt.year - 1, 12)
    sql("UPDATE express_auftraege SET bestellt_am = ? WHERE id = ?", f"{j:04d}-{m:02d}-28 10:00:00", aid4)
    z = billing.pruefe_kontingent(k_id)
    abo = kunde.get("/api/me").json()["abo"]
    check("/api/me nennt nur die Vormerkung, die diesen Monat bindet (R2)", r.status_code == 200
          and z["vorgemerkt"] > z["vorgemerkt_laufend"] and abo["vorgemerkt"] == z["vorgemerkt_laufend"], (z, abo))
    check("/api/me: Rest + Zusatz-Credits - vorgemerkt = verfügbar = Sperre", z["verfuegbar_gesamt"]
          == max(0, abo["rest"] + abo["pakete_rest"] - abo["vorgemerkt"]) and z["erlaubt"] == (z["verfuegbar_gesamt"] > 0), (z, abo))
    voll.post(f"/api/admin/express/auftraege/{aid4}/stornieren", json={"grund": "Test"})

    print("== R7. Meine Aufträge, Warenkorb, Anmeldung über Links (Runde 7) ==")
    seite = kunde.get("/express").text
    check("/express = „Meine Aufträge“ (H1 mit Sprungziel h-auftraege), kein Bestellformular, Weg zum Warenkorb",
          '<h1 id="h-auftraege"' in seite and "<title>Meine Aufträge" in seite and 'id="exBestellForm"' not in seite
          and 'href="/express/warenkorb"' in seite)
    seite = kunde.get("/express/warenkorb").text
    check("/express/warenkorb: Bestellformular ohne Hochladen", 'id="exBestellForm"' in seite and "exHochladen" not in seite
          and "hochladefeld.js" not in seite)
    r = kunde.get(f"/express?projekt={pid}", follow_redirects=False)
    check("Alter Link /express?projekt= führt zum Warenkorb", r.status_code == 302 and r.headers.get("location") == f"/express/warenkorb?projekt={pid}",
          (r.status_code, r.headers.get("location")))
    anonym = httpx.Client(base_url=BASE, timeout=60)
    r = anonym.get(f"/express/auftrag/{aid4}", follow_redirects=False)
    check("Ohne Sitzung: Auftragsübersicht leitet zur Anmeldung mit Rücksprung", r.status_code in (302, 307)
          and r.headers.get("location") == f"/login?weiter=%2Fexpress%2Fauftrag%2F{aid4}", (r.status_code, r.headers.get("location")))
    r = anonym.get("/app?projekt=957", follow_redirects=False)
    check("Ohne Sitzung: Projektseite mit Abfrage behält das Ziel", r.headers.get("location") == "/login?weiter=%2Fapp%3Fprojekt%3D957",
          r.headers.get("location"))
    check("Login übernimmt nur interne Ziele", 'const WEITER = "/express/auftrag/5"' in anonym.get("/login?weiter=/express/auftrag/5").text
          and all('const WEITER = ""' in anonym.get("/login", params={"weiter": z}).text
                  for z in ("//boese.example", "https://boese.example", "/\\boese", "javascript:alert(1)", "/login")))
    r = anonym.post("/api/login", json={"email": KUNDE, "password": PW})
    check("Anmeldung klappt (Rücksprung macht die Seite selbst)", r.status_code == 200)
    anonym.close()
    r = fremd.get(f"/api/express/auftraege/{aid4}")
    check("Fremdes Konto: Auftrags-API 404 (nichts preisgegeben), Seite nur Hülle", r.status_code == 404
          and "Kim Muster" not in fremd.get(f"/express/auftrag/{aid4}").text, r.status_code)
    pos = sql("SELECT preis_seite, grundpreis FROM express_positionen WHERE auftrag_id = ?", aid4)
    check("Bestellte Positionen merken Preis je Seite und Grundpreis", pos and all(p["preis_seite"] == 50 and p["grundpreis"] == 0 for p in pos), pos)

    print("== H. Recht wieder entziehen ==")
    r = voll.delete(f"/api/admin/express/bearbeiter/{b_id}")
    check("Recht entzogen", r.status_code == 200)
    check("Bearbeiter danach 403", bearb.get("/api/admin/express/auftraege").status_code == 403)
finally:
    print("== Aufraeumen ==")
    if einst_vorher:
        sql("UPDATE system_kv SET value = ? WHERE key = 'express_einstellungen'", einst_vorher[0]["value"])
    else:
        sql("DELETE FROM system_kv WHERE key = 'express_einstellungen'")
    aufraeumen()
    check("Testkonten und Auftraege entfernt", sql("SELECT COUNT(*) AS n FROM users WHERE email LIKE ?", f"%@{DOM}")[0]["n"] == 0)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
