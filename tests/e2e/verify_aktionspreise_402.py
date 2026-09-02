"""Aktionspreise 29.08.2026 — echter 402-Weg mit ABO_ENFORCEMENT=on.
Laeuft IM Staging-Container gegen eine zweite uvicorn-Instanz (Port 8099, ABO_ENFORCEMENT=on),
damit das laufende Staging (Enforcement aus) unberuehrt bleibt. Test-Konto: steve.weidel@gmail.com
(Free, von Steve fuer Tests freigegeben). Synthetische Verbrauchs-Zeilen werden am Ende geloescht.
"""
import os, sys, sqlite3, httpx, time

BASE = "http://127.0.0.1:8099"
MAIL, PW = "steve.weidel@gmail.com", os.environ.get("TEST_PW", "")
DB = "/app/data/inkludocs.db"
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", info)


con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
uid = con.execute("SELECT id FROM users WHERE email=?", (MAIL,)).fetchone()["id"]
sys.path.insert(0, "/app")
import billing  # noqa: E402
billing.ABO_ENFORCEMENT = True   # nur fuer die Rechnung in DIESEM Prozess

# Ausgangslage: alles, was das Konto diesen Monat schon verbraucht hat
START_MAX_ID = con.execute("SELECT COALESCE(MAX(id), 0) FROM usage_events").fetchone()[0]
z = billing.pruefe_kontingent(uid)
print("Ausgangslage:", {k: z[k] for k in ("plan", "kontingent", "verbraucht", "rest", "pakete_rest")})
check("Testkonto ist Free mit 50 Credits", z["plan"] == "free" and z["kontingent"] == 50, z)


def setze_verfuegbar(ziel):
    """Synthetische Verbrauchszeilen einfuegen/entfernen, bis genau `ziel` Credits uebrig sind."""
    con.execute("DELETE FROM usage_events WHERE konto_user_id=? AND aktion='test_aktionspreise'", (uid,))
    con.commit()
    z = billing.pruefe_kontingent(uid)
    rest = int(z["rest"]) + int(z["pakete_rest"])
    if rest > ziel:
        con.execute("INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits) VALUES (?,?,?,?,?)",
                    (uid, uid, "einzeln", "test_aktionspreise", rest - ziel))
        con.commit()
    v = billing.verfuegbare_credits(uid)
    return v


c = httpx.Client(base_url=BASE, timeout=60)
for _ in range(30):
    try:
        if c.get("/").status_code < 500:
            break
    except Exception:
        time.sleep(1)
r = c.post("/api/login", json={"email": MAIL, "password": PW})
check("Login am Enforcement-Server", r.status_code == 200, r.text[:120])
tok = r.cookies.get("token")
if tok:
    c.headers["Cookie"] = "token=" + tok

PID_WEB = 18   # Website-Projekt des Testkontos, 21 Bilder, Status done
img = con.execute("SELECT id FROM images WHERE project_id=? AND status='done' LIMIT 1", (PID_WEB,)).fetchone()
img_id = img["id"] if img else None

print("== A. 7 Credits uebrig: alles Kostenpflichtige muss 402 liefern ==")
v = setze_verfuegbar(7)
check("Guthaben auf 7 gesetzt", v == 7, v)
r = c.post(f"/api/projects/{PID_WEB}/export/preis", json={})
b = r.json()
check("export/preis: 21 Bilder -> 25 + 3x5 = 40, erlaubt False, fehlend 33, Tabellenpreis 10",
      r.status_code == 200 and b.get("anzahl") == 21 and b.get("preis") == 40 and b.get("erlaubt") is False
      and b.get("fehlend") == 33 and b.get("preis_tabelle") == 10 and b.get("verfuegbar") == 7, b)
r = c.post(f"/api/projects/{PID_WEB}/export/summary", json={})
b = r.json()
check("export/summary: preis 40, erlaubt False, preis_tabelle 10", b.get("preis") == 40 and b.get("erlaubt") is False and b.get("preis_tabelle") == 10, b)
r = c.post(f"/api/projects/{PID_WEB}/export/csv", json={})
d = r.json().get("detail", {}) if r.status_code == 402 else {}
check("CSV-Export bei 7 Credits -> 402 credits_fehlen (10 noetig, 7 da, fehlend 3)",
      r.status_code == 402 and d.get("code") == "credits_fehlen" and d.get("preis") == 10 and d.get("verfuegbar") == 7 and d.get("fehlend") == 3, (r.status_code, d))
check("402-Text nennt beide Zahlen", "10 Credits" in d.get("text", "") and "7 Credits" in d.get("text", ""), d.get("text"))
r = c.post(f"/api/projects/{PID_WEB}/export/json", json={})
check("JSON-Export bei 7 Credits -> 402", r.status_code == 402, r.status_code)
r = c.post(f"/api/projects/{PID_WEB}/export/xlsx", json={})
check("Excel-Export bei 7 Credits -> 402 (zurueck am 02.09.2026)", r.status_code == 402, r.status_code)
# Ein einzelner Alt-Text (5) waere bei 7 noch erlaubt — nur rechnerisch pruefen, kein echter Bedrock-Lauf.
w = billing.aktion_pruefung(uid, "bild_generierung")
check("Rechnerisch: 1 Alt-Text (5) bei 7 Credits erlaubt, 2 Alt-Texte (10) nicht",
      w["erlaubt"] is True and billing.aktion_pruefung(uid, "bild_generierung", 2)["erlaubt"] is False, w)
if not img_id:
    print("     (kein fertiges Bild im Projekt 18 — Regenerate-Check in B uebersprungen)")
vorher = con.execute("SELECT COUNT(*) FROM usage_events WHERE konto_user_id=?", (uid,)).fetchone()[0]

print("== B. 4 Credits uebrig: auch ein einzelner Alt-Text (5) ist gesperrt ==")
v = setze_verfuegbar(4)
check("Guthaben auf 4 gesetzt", v == 4, v)
if img_id:
    r = c.post(f"/api/projects/{PID_WEB}/regenerate/{img_id}", json={})
    d = r.json().get("detail", {}) if r.status_code == 402 else {}
    check("Neu generieren bei 4 Credits -> 402 credits_fehlen (5 noetig, 4 da)",
          r.status_code == 402 and d.get("code") == "credits_fehlen" and d.get("preis") == 5 and d.get("verfuegbar") == 4, (r.status_code, d))
r = c.post(f"/api/projects/{PID_WEB}/generate", json={"modus": "ki_neu"})
print("     generate (Sammellauf-Start) ->", r.status_code, r.text[:100])
time.sleep(2)
nachher = con.execute("SELECT COUNT(*) FROM usage_events WHERE konto_user_id=?", (uid,)).fetchone()[0]
check("Sammellauf bei 4 Credits verbucht NICHTS (Wache je Bild)", nachher == vorher, (vorher, nachher))

print("== C. 12 Credits uebrig: CSV geht (10), danach bleiben 2, JSON ist gesperrt ==")
v = setze_verfuegbar(12)
check("Guthaben auf 12 gesetzt", v == 12, v)
r = c.post(f"/api/projects/{PID_WEB}/export/csv", json={})
check("CSV-Export bei 12 Credits -> 200 mit X-Export-Credits 10",
      r.status_code == 200 and r.headers.get("x-export-credits") == "10" and r.content[:3] == b"\xef\xbb\xbf", (r.status_code, dict(r.headers)))
ev = con.execute("SELECT quelle, aktion, credits FROM usage_events WHERE konto_user_id=? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
check("Verbucht: quelle export, aktion csv_export, 10 Credits", ev and ev["quelle"] == "export" and ev["aktion"] == "csv_export" and ev["credits"] == 10, dict(ev) if ev else None)
check("Danach 2 Credits uebrig", billing.verfuegbare_credits(uid) == 2, billing.verfuegbare_credits(uid))
r = c.post(f"/api/projects/{PID_WEB}/export/json", json={})
check("JSON-Export bei 2 Credits -> 402", r.status_code == 402, r.status_code)

print("== C2. Excel (zurueck am 02.09.2026): 10 Credits wie CSV/JSON ==")
v = setze_verfuegbar(12)
check("Guthaben auf 12 gesetzt", v == 12, v)
r = c.post(f"/api/projects/{PID_WEB}/export/xlsx", json={})
check("Excel-Export bei 12 Credits -> 200, X-Export-Credits 10, ZIP/XLSX-Magic",
      r.status_code == 200 and r.headers.get("x-export-credits") == "10" and r.content[:2] == b"PK", (r.status_code, r.content[:8]))
ev = con.execute("SELECT quelle, aktion, credits FROM usage_events WHERE konto_user_id=? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
check("Verbucht: quelle export, aktion xlsx_export, 10 Credits", ev and ev["quelle"] == "export" and ev["aktion"] == "xlsx_export" and ev["credits"] == 10, dict(ev) if ev else None)
check("Danach 2 Credits uebrig", billing.verfuegbare_credits(uid) == 2, billing.verfuegbare_credits(uid))
r = c.post(f"/api/projects/{PID_WEB}/export/xlsx", json={})
check("Excel-Export bei 2 Credits -> 402", r.status_code == 402, r.status_code)

print("== D. Aufraeumen ==")
con.execute("DELETE FROM usage_events WHERE konto_user_id=? AND id > ?", (uid, START_MAX_ID))
con.commit()
z2 = billing.pruefe_kontingent(uid)
check("Verbrauch wieder wie am Anfang", z2["verbraucht"] == z["verbraucht"], (z["verbraucht"], z2["verbraucht"]))
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
