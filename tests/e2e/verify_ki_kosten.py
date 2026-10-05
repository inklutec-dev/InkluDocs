#!/usr/bin/env python3
"""E2E KI-KOSTEN (05.10.2026): ein echter Alt-Text-Lauf auf Staging landet mit Kunde, Projekt, Bild, Zweck, Tokens und
Kosten in ki_aufrufe; die Verwaltung zeigt ihn; Rechte (Kunde/Nur-Einsicht/Voll-Admin) und Eingabepruefungen der
Preisliste stimmen. Laeuft IM Staging-Container (braucht die echte Staging-Datenbank und einen KI-Aufruf):
    docker cp tests/e2e/verify_ki_kosten.py inkludocs-staging:/tmp/ && docker exec inkludocs-staging python3 /tmp/verify_ki_kosten.py
Testkonten auf der reservierten Domain .invalid (nie Mails), alles wird am Ende entfernt; die Preisliste wird auf den
Stand vor dem Test zurueckgesetzt. Kosten: ein bis zwei Alt-Text-Aufrufe (wenige Cent)."""
import io
import json
import os
import sys
import time

sys.path.insert(0, "/app")
os.chdir("/app")

import httpx  # noqa: E402

from database import create_user, delete_user_data, get_db, get_user_by_email  # noqa: E402

BASE = "http://127.0.0.1:8001"
PW = "KiKosten-Test-2026!x"
DOM = "ki-kosten-test.invalid"
KUNDE, VOLL, SICHT = f"kunde@{DOM}", f"voll@{DOM}", f"sicht@{DOM}"
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
        rows = [dict(r) for r in conn.execute(q, p).fetchall()]
        conn.commit()
        return rows
    finally:
        conn.close()


def aufraeumen():
    for mail in (KUNDE, VOLL, SICHT):
        u = get_user_by_email(mail)
        if u:
            sql("DELETE FROM ki_aufrufe WHERE user_id = ? OR konto_user_id = ?", u["id"], u["id"])
            delete_user_data(u["id"])


def client(mail):
    c = httpx.Client(base_url=BASE, timeout=180)
    r = c.post("/api/login", json={"email": mail, "password": PW})
    assert r.status_code == 200, f"Login {mail}: {r.status_code} {r.text[:200]}"
    c.headers["Cookie"] = "token=" + r.cookies.get("token")
    return c


aufraeumen()
preise_vorher = sql("SELECT value FROM system_kv WHERE key = 'ki_preise'")
for mail in (KUNDE, VOLL, SICHT):
    create_user(mail, PW, "Testperson KI-Kosten (fiktiv)")
k_id = get_user_by_email(KUNDE)["id"]
sql("UPDATE users SET is_admin = 1, admin_level = 'full' WHERE email = ?", VOLL)
sql("UPDATE users SET is_admin = 1, admin_level = 'view' WHERE email = ?", SICHT)
# Genug Guthaben fuer den Lauf (Bonus-Paket, ohne Umsatz).
sql("INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) VALUES (?, 100, 100, 'admin', 'KI-Kosten-Test', NULL)", k_id)

try:
    kunde, voll, sicht = client(KUNDE), client(VOLL), client(SICHT)
    from PIL import Image, ImageDraw
    bild = Image.new("RGB", (320, 200), (250, 250, 250))
    zeichnung = ImageDraw.Draw(bild)
    zeichnung.rectangle((40, 40, 280, 160), outline=(20, 60, 160), width=6)
    zeichnung.ellipse((120, 70, 200, 130), fill=(200, 40, 40))
    puffer = io.BytesIO()
    bild.save(puffer, format="PNG")

    print("== A. Echter Alt-Text-Lauf ==")
    r = kunde.post("/api/projects", json={"name": "KI-Kosten-Test (fiktiv)", "tool": "grafik"})
    pid = (r.json() or {}).get("id") or (r.json() or {}).get("project_id")
    check("Projekt angelegt", r.status_code == 200 and pid, r.text)
    r = kunde.post("/api/upload", data={"project_id": str(pid)}, files={"file": ("kreis_im_rahmen_fiktiv.png", puffer.getvalue(), "image/png")})
    check("Bild hochgeladen", r.status_code == 200, r.text)
    iid = None
    for _ in range(30):
        imgs = kunde.get(f"/api/projects/{pid}").json().get("images") or []
        if imgs:
            iid = imgs[0]["id"]
            break
        time.sleep(1)
    check("Bild im Projekt", iid, imgs)
    status = sql("SELECT status FROM images WHERE id = ?", iid)[0]["status"]
    if status == "pending":
        r = kunde.post(f"/api/projects/{pid}/generate", json={})
        check("Lauf gestartet", r.status_code == 200, r.text)
    for _ in range(120):
        status = sql("SELECT status FROM images WHERE id = ?", iid)[0]["status"]
        if status in ("done", "error"):
            break
        time.sleep(2)
    check("Alt-Text erzeugt", status == "done", status)
    zeilen = sql("SELECT * FROM ki_aufrufe WHERE project_id = ? ORDER BY id", pid)
    check("KI-Aufrufe erfasst", len(zeilen) >= 1, zeilen)
    check("alle mit Kunde, Konto, Bild und Zweck alttext",
          all(z["user_id"] == k_id and z["konto_user_id"] == k_id and z["image_id"] == iid and z["zweck"] == "alttext" for z in zeilen), zeilen)
    check("Tokens und Kosten > 0", all(z["tokens_ein"] > 0 and (z["kosten_eur_cent"] or 0) > 0 for z in zeilen if z["erfolg"]), zeilen)
    check("Umgebung staging, Anbieter gemini", all(z["umgebung"] == "staging" and z["anbieter"] == "gemini" for z in zeilen), zeilen)
    anzahl_lauf = len(zeilen)

    print("== B. Einzel-Neu-Generieren zaehlt dazu ==")
    r = kunde.post(f"/api/projects/{pid}/regenerate/{iid}", json={})
    check("Neu generiert", r.status_code == 200, r.text)
    zeilen = sql("SELECT * FROM ki_aufrufe WHERE project_id = ? ORDER BY id", pid)
    check("weitere Aufrufe mit demselben Bild", len(zeilen) > anzahl_lauf and zeilen[-1]["image_id"] == iid, zeilen[-1:] if zeilen else zeilen)

    print("== C. Verwaltung: Bericht, Drill-down, Kundenseite ==")
    r = voll.get("/api/admin/ki-kosten")
    d = r.json()
    check("Monatsbericht 200", r.status_code == 200, r.text)
    eintrag = next((k for k in d.get("kunden", []) if k["konto_user_id"] == k_id), None)
    check("Kunde im Bericht mit Kosten", eintrag and eintrag["kosten_cent"] > 0 and eintrag["aufrufe"] == len(zeilen), eintrag)
    check("Zweck alttext im Bericht", any(z["zweck"] == "alttext" for z in d.get("nach_zweck", [])), d.get("nach_zweck"))
    r = voll.get("/api/admin/ki-kosten/kunde", params={"konto": k_id})
    check("Projekte des Kunden", r.status_code == 200 and r.json()["projekte"][0]["project_id"] == pid, r.text)
    r = voll.get("/api/admin/ki-kosten/projekt", params={"konto": k_id, "projekt": pid})
    check("Bild im Projekt-Drill-down", r.status_code == 200 and r.json()["bilder"][0]["image_id"] == iid, r.text)
    r = voll.get(f"/api/admin/users/{k_id}/report")
    k = r.json().get("kosten", {})
    check("Kundenseite: gemessene Kosten", r.status_code == 200 and k.get("gemessen_cent", 0) > 0 and k.get("messbeginn"), k)
    r = voll.get("/verwaltung/ki-kosten")
    check("Seite /verwaltung/ki-kosten", r.status_code == 200 and "KI-Kosten" in r.text, r.status_code)
    r = sicht.get("/api/admin/ki-kosten")
    check("Nur-Einsicht darf lesen", r.status_code == 200, r.status_code)
    r = voll.get("/api/admin/ki-kosten", params={"jahr": "1999", "monat": "1"})
    check("unsinniger Zeitraum -> 400", r.status_code == 400, r.status_code)
    r = voll.get("/api/admin/ki-kosten/kunde", params={"konto": "1 OR 1=1"})
    check("Konto-Kennung wird geprueft", r.status_code == 400, r.status_code)

    print("== D. Rechte ==")
    check("Kunde sieht nichts", kunde.get("/api/admin/ki-kosten").status_code == 403)
    check("Kunde: Drill-down 403", kunde.get("/api/admin/ki-kosten/kunde", params={"konto": k_id}).status_code == 403)
    check("ohne Anmeldung 401", httpx.get(BASE + "/api/admin/ki-kosten").status_code == 401)
    preis = {"modell": "test-modell-e2e", "ein": "1,50", "aus": "6", "quelle": "E2E-Test (fiktiv)", "ab": "2026-01-01"}
    check("Nur-Einsicht darf keine Preise aendern", sicht.post("/api/admin/ki-preise/modell", json=preis).status_code == 403)
    check("Kunde darf keine Preise aendern", kunde.post("/api/admin/ki-preise/modell", json=preis).status_code == 403)
    check("Kunde darf den Kurs nicht aendern", kunde.post("/api/admin/ki-preise/kurs", json={"usd_eur": "0,9", "quelle": "x"}).status_code == 403)

    print("== E. Preisliste ==")
    r = voll.post("/api/admin/ki-preise/modell", json=preis)
    check("Voll-Admin traegt Preis ein", r.status_code == 200, r.text)
    m = next((x for x in r.json()["preise"]["modelle"] if x["modell"] == "test-modell-e2e"), None)
    check("Preis gespeichert (1,5 / 6)", m and m["aktuell"]["ein"] == 1.5 and m["aktuell"]["aus"] == 6.0, m)
    for name, falsch in (("Text statt Zahl", dict(preis, ein="viel")), ("negativ", dict(preis, aus="-1")),
                         ("Modellname mit Leerzeichen", dict(preis, modell="böses modell")), ("ohne Quelle", dict(preis, quelle="")),
                         ("Datum falsch", dict(preis, ab="05.10.2026")), ("Riesenwert", dict(preis, ein="99999"))):
        check(f"abgelehnt: {name}", voll.post("/api/admin/ki-preise/modell", json=falsch).status_code == 400)
    check("Kurs ausserhalb des Bereichs abgelehnt", voll.post("/api/admin/ki-preise/kurs", json={"usd_eur": "9", "quelle": "x"}).status_code == 400)
    check("Kurs ohne Quelle abgelehnt", voll.post("/api/admin/ki-preise/kurs", json={"usd_eur": "0,9"}).status_code == 400)
finally:
    print("== Aufraeumen ==")
    if preise_vorher:
        sql("UPDATE system_kv SET value = ? WHERE key = 'ki_preise'", preise_vorher[0]["value"])
    else:
        sql("DELETE FROM system_kv WHERE key = 'ki_preise'")
    aufraeumen()
    rest = sql("SELECT COUNT(*) AS n FROM users WHERE email LIKE ?", f"%@{DOM}")[0]["n"]
    check("Testkonten entfernt", rest == 0, rest)

print(f"Ergebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
