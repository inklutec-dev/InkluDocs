#!/usr/bin/env python3
"""E2E EXPRESS-SERVICE Stufe 1 (05.10.2026) im Staging-Container ueber HTTP: ganzer Ablauf (Upload ohne Projekt, Auswahl,
Bestellen mit Vormerkung, Uebernehmen, Rueckfrage, Antwort, Ergebnis mit veraPDF, Liefern, Download, Nachweis-PDF,
ZIP, Storno) und die Sicherheitsfaelle: fremde Kunden (IDOR), Nur-Einsicht-Admin, Express-Bearbeiter ohne Admin-Recht,
Upload-Pruefungen, Doppel-Bestellung, Doppel-Lieferung, Storno nach Lieferung.
    docker cp tests/e2e/verify_express.py inkludocs-staging:/tmp/ && docker exec inkludocs-staging python3 /tmp/verify_express.py
Testkonten auf .invalid (Mails werden unterdrueckt); Team-Mails gehen waehrend des Tests an eine .invalid-Adresse.
Alles wird am Ende entfernt, die Einstellungen werden zurueckgesetzt."""
import io
import json
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
    r = voll.post("/api/admin/express/einstellungen", json={
        "preis_aufbereiten": "50", "preis_pruefen": "25", "frist_stunden": "48", "max_seiten_auftrag": "500",
        "max_dokumente_auftrag": "50", "team_mail": f"team@{DOM}", "preise_festgelegt": False})
    check("Voll-Admin setzt Einstellungen (Team-Mail auf Testadresse)", r.status_code == 200, r.text)

    print("== A. Seiten und Stand ==")
    for pfad in ("/express", "/express/bedingungen"):
        r = kunde.get(pfad)
        check(f"Seite {pfad} 200", r.status_code == 200 and "Express" in r.text, r.status_code)
    r = kunde.get("/api/express/stand")
    st = r.json()
    check("Stand: Preise, Frist, Texte", r.status_code == 200 and st["preise"]["aufbereiten"] == 50 and st["frist_stunden"] == 48
          and st["texte"]["bearbeitung"].startswith("Ich bin einverstanden"), st)
    check("/api/me meldet express", kunde.get("/api/me").json()["user"]["express"] is True)

    print("== B. Upload ohne Projekt + Pruefungen ==")
    r = kunde.post("/api/express/warenkorb/hochladen", files={"file": ("bild.png", b"\x89PNG\r\n", "image/png")})
    check("Nicht-PDF abgewiesen (400)", r.status_code == 400, r.text)
    r = kunde.post("/api/express/warenkorb/hochladen", files={"file": ("tarnung.pdf", b"MZ\x90\x00 kein pdf", "application/pdf")})
    check("Getarnte Datei abgewiesen (400)", r.status_code == 400, r.text)
    r = kunde.post("/api/express/warenkorb/hochladen", files={"file": ("Jahresbericht fiktiv.pdf", pdf_bytes(3), "application/pdf")})
    check("PDF hochgeladen und in der Auswahl", r.status_code == 200 and r.json()["warenkorb"]["seiten"] == 3, r.text)
    proj = sql("SELECT id, name FROM projects WHERE user_id = ?", k_id)
    check("Projekt „Express-Auftrag <Nr>“ angelegt", len(proj) == 1 and proj[0]["name"].startswith("Express-Auftrag "), proj)
    pid = proj[0]["id"]
    r = kunde.post("/api/express/warenkorb/hochladen", files={"file": ("Flyer fiktiv.pdf", pdf_bytes(1), "application/pdf")})
    check("Zweite PDF landet im selben Projekt", r.status_code == 200 and len(sql("SELECT id FROM projects WHERE user_id = ?", k_id)) == 1, r.text)
    korb = kunde.get("/api/express/stand").json()["warenkorb"]
    check("Auswahl: 2 Dokumente, 4 Seiten, 200 Credits", (korb["dokumente"], korb["seiten"], korb["credits"]) == (2, 4, 200), korb)
    docs = kunde.get(f"/api/express/projekte/{pid}/dokumente").json()["dokumente"]
    check("Dokumente des Projekts: beide schon in der Auswahl", all(d["im_warenkorb"] for d in docs), docs)
    pos2 = korb["positionen"][1]["id"]
    r = kunde.post(f"/api/express/warenkorb/positionen/{pos2}/leistung", json={"leistung": "pruefen"})
    check("Leistung „Nur prüfen“ gesetzt", r.status_code == 200 and r.json()["credits"] == 175, r.text)

    print("== C. Fremde Kunden (IDOR) ==")
    check("Fremder sieht Projekt-Dokumente nicht (404)", fremd.get(f"/api/express/projekte/{pid}/dokumente").status_code == 404)
    r = fremd.post("/api/express/warenkorb/dokumente", json={"document_ids": [docs[0]["id"]]})
    check("Fremder kann Dokument nicht in seine Auswahl legen (404)", r.status_code == 404, r.text)
    check("Fremder kann Position nicht aendern (404)", fremd.post(f"/api/express/warenkorb/positionen/{pos2}/leistung", json={"leistung": "aufbereiten"}).status_code == 404)
    check("Fremder kann Position nicht entfernen (404)", fremd.delete(f"/api/express/warenkorb/positionen/{pos2}").status_code == 404)
    check("Ohne Anmeldung: 401", httpx.get(BASE + "/api/express/stand").status_code == 401)

    print("== D. Bestellen ==")
    bestellung = {"ansprechpartner": "Kim Muster (fiktiv)", "telefon": "+49 40 0000", "hinweise": "Seite 2 bitte genau",
                  "bedingungen": True, "bearbeitung": True, "idempotenz": "e2e-express-0001"}
    r = kunde.post("/api/express/bestellen", json=dict(bestellung, bearbeitung=False))
    check("Ohne Einverstaendnis: 400", r.status_code == 400, r.text)
    r = kunde.post("/api/express/bestellen", json=bestellung)
    check("Zu wenig Guthaben: 402 mit Zahlen", r.status_code == 402 and r.json()["detail"]["preis"] == 175, r.text)
    sql("INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) VALUES (?, 1000, 1000, 'admin', 'Express-Test', NULL)", k_id)
    verf_vorher = kunde.get("/api/express/stand").json()["guthaben"]
    r = kunde.post("/api/express/bestellen", json=bestellung)
    aid = r.json().get("auftrag_id")
    check("Bestellt", r.status_code == 200 and aid and not r.json()["schon_bestellt"], r.text)
    r = kunde.post("/api/express/bestellen", json=bestellung)
    check("Doppelklick: derselbe Auftrag, nicht zweimal", r.status_code == 200 and r.json()["auftrag_id"] == aid and r.json()["schon_bestellt"], r.text)
    check("Genau ein Auftrag", len(sql("SELECT id FROM express_auftraege WHERE user_id = ? AND status != 'entwurf'", k_id)) == 1)
    st = kunde.get("/api/express/stand").json()
    check("Guthaben um 175 gemindert (vorgemerkt)", st["guthaben"] == verf_vorher - 175, (verf_vorher, st["guthaben"]))
    check("/api/me: 175 vorgemerkt", kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 175)
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
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis", files={"file": ("x.pdf", b"kein pdf", "application/pdf")})
    check("Ergebnis: keine PDF -> 400", r.status_code == 400, r.text)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/skript", files={"file": ("x.pdf", pdf_bytes(1), "application/pdf")})
    check("Unbekannte Dateiart -> 404", r.status_code == 404)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis",
                   files={"file": ("../../../etc/ergebnis.pdf", pdf_bytes(3, "Aufbereitet"), "application/pdf")})
    pos = next(p for p in r.json()["positionen"] if p["id"] == p1["id"]) if r.status_code == 200 else {}
    check("Ergebnis hochgeladen, veraPDF gelaufen", r.status_code == 200 and pos.get("ergebnis_da") and pos.get("verapdf"), r.text[:300])
    check("Dateiname ohne Pfad", pos.get("ergebnis_name") == "ergebnis.pdf", pos.get("ergebnis_name"))
    pfad = sql("SELECT ergebnis_pfad FROM express_positionen WHERE id = ?", p1["id"])[0]["ergebnis_pfad"]
    check("Ablage im Auftragsordner", pfad.startswith(f"/app/data/results/{k_id}/_express/{aid}/") and ".." not in pfad, pfad)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p2['id']}/bericht", files={"file": ("bericht.pdf", pdf_bytes(1, "Pruefbericht"), "application/pdf")})
    check("Pruefbericht fuer „Nur pruefen“", r.status_code == 200, r.text[:200])
    check("Kunde: Ergebnis vor der Lieferung 404", kunde.get(f"/api/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis").status_code == 404)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/liefern", json={})
    if r.status_code == 409 and r.json().get("detail", {}).get("nachfrage"):
        check("veraPDF-Abweichungen: Nachfrage statt Lieferung", True)
        check("Nichts abgebucht vor der Bestaetigung", sql("SELECT COUNT(*) AS n FROM usage_events WHERE quelle = 'express'")[0]["n"] == 0)
        r = bearb.post(f"/api/admin/express/auftraege/{aid}/liefern", json={"trotz_befunden": True})
    check("Geliefert", r.status_code == 200 and r.json()["status"] == "geliefert", r.text[:300])
    ev = sql("SELECT aktion, credits, konto_user_id FROM usage_events WHERE quelle = 'express' AND user_id = ?", k_id)
    check("Abgebucht: 175 Credits (150 aufbereiten + 25 pruefen) auf den Topf der Bestellung",
          sorted((e["aktion"], e["credits"]) for e in ev) == [("express_aufbereiten", 150), ("express_pruefen", 25)]
          and all(e["konto_user_id"] == k_id for e in ev), ev)
    check("Vormerkung weg", kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 0)
    r = bearb.post(f"/api/admin/express/auftraege/{aid}/liefern", json={"trotz_befunden": True})
    check("Zweites Liefern: 409, nichts doppelt", r.status_code == 409 and len(sql("SELECT id FROM usage_events WHERE quelle = 'express' AND user_id = ?", k_id)) == 2)
    check("Storno nach Lieferung: 409", bearb.post(f"/api/admin/express/auftraege/{aid}/stornieren", json={"grund": "Test"}).status_code == 409)
    check("Upload nach Lieferung: 409", bearb.post(f"/api/admin/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis",
                                                  files={"file": ("x.pdf", pdf_bytes(1), "application/pdf")}).status_code == 409)
    r = kunde.get(f"/api/express/auftraege/{aid}/positionen/{p1['id']}/ergebnis")
    check("Kunde laedt das Ergebnis", r.status_code == 200 and r.content.startswith(b"%PDF")
          and "barrierefrei" in r.headers.get("content-disposition", ""), r.headers.get("content-disposition"))
    check("Kunde: Pruefbericht", kunde.get(f"/api/express/auftraege/{aid}/positionen/{p2['id']}/bericht").status_code == 200)
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
    r = kunde.post("/api/express/bestellen", json=dict(bestellung, idempotenz="e2e-express-0002"))
    aid2 = r.json().get("auftrag_id")
    check("Zweiter Auftrag, 150 vorgemerkt", r.status_code == 200 and kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 150, r.text)
    check("Storno ohne Grund: 400", voll.post(f"/api/admin/express/auftraege/{aid2}/stornieren", json={"grund": ""}).status_code == 400)
    r = voll.post(f"/api/admin/express/auftraege/{aid2}/stornieren", json={"grund": "Test-Storno"})
    check("Storniert, Vormerkung frei, nichts abgebucht", r.status_code == 200 and kunde.get("/api/me").json()["abo"]["vorgemerkt"] == 0
          and len(sql("SELECT id FROM usage_events WHERE quelle = 'express' AND user_id = ?", k_id)) == 2, r.text[:200])
    check("Kunde sieht Storno mit Grund", kunde.get(f"/api/express/auftraege/{aid2}").json()["storno_grund"] == "Test-Storno")

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
