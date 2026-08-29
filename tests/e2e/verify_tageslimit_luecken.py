"""Tageslimit-Luecken (29.08.2026) — im Staging-Container gegen die laufende App:
    docker cp tests/e2e/verify_tageslimit_luecken.py inkludocs-staging:/tmp/ && docker exec inkludocs-staging python3 /tmp/verify_tageslimit_luecken.py
Setzt dem Gmail-Testkonto (Free, von Steve freigegeben) voruebergehend api_tageslimit=1, erzeugt ein
synthetisches KI-Ereignis fuer heute und erwartet 429 beim Neu-Generieren und beim Sammellauf-Start;
raeumt alles wieder auf. Kein echter Bedrock-Aufruf."""
import os, sqlite3, sys, httpx

BASE = os.environ.get("BASE", "http://127.0.0.1:8001")
MAIL, PW = "steve.weidel@gmail.com", os.environ.get("TEST_PW", "")
DB = "/app/data/inkludocs.db"
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", str(info)[:300])


sys.path.insert(0, "/app")
import billing  # noqa: E402
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
uid = con.execute("SELECT id FROM users WHERE email=?", (MAIL,)).fetchone()["id"]
alt_limit = con.execute("SELECT api_tageslimit FROM users WHERE id=?", (uid,)).fetchone()["api_tageslimit"]
start_max = con.execute("SELECT COALESCE(MAX(id),0) FROM usage_events").fetchone()[0]
c = httpx.Client(base_url=BASE, timeout=60)
r = c.post("/api/login", json={"email": MAIL, "password": PW})
check("Login", r.status_code == 200, r.text[:120])
tok = r.cookies.get("token")
if tok:
    c.headers["Cookie"] = "token=" + tok
PID = 18
img = con.execute("SELECT id FROM images WHERE project_id=? AND status='done' LIMIT 1", (PID,)).fetchone()
try:
    check("Zaehler vorher: heute 0 KI-Aufrufe fuer das Testkonto oder mehr (nur Info)", True, billing.tagesverbrauch_ki(uid))
    con.execute("UPDATE users SET api_tageslimit=1 WHERE id=?", (uid,)); con.commit()
    con.execute("INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits) VALUES (?,?,?,?,?)",
                (uid, uid, "einzeln", "bild_generierung", 5)); con.commit()
    check("tagesverbrauch_ki zaehlt das Ereignis", billing.tagesverbrauch_ki(uid) >= 1, billing.tagesverbrauch_ki(uid))
    if img:
        r = c.post(f"/api/projects/{PID}/regenerate/{img['id']}", json={})
        check("Luecke 2: Neu generieren bei erreichtem Tageslimit -> 429", r.status_code == 429 and "Tageslimit" in r.text, (r.status_code, r.text[:120]))
    r = c.post(f"/api/projects/{PID}/generate", json={"modus": "ki_neu"})
    check("Luecke 1: Sammellauf-Start bei erreichtem Tageslimit -> 429", r.status_code == 429 and "Tageslimit" in r.text, (r.status_code, r.text[:120]))
    con.execute("UPDATE users SET api_tageslimit=5 WHERE id=?", (uid,)); con.commit()
    if img:
        # Limit 5, genutzt 1 -> Neu generieren darf (wir brechen den echten KI-Lauf nicht an: nur Wache pruefen)
        from main import tageslimit_wache
        row = dict(con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())
        check("Wache: Limit 5, genutzt 1 -> darf", tageslimit_wache(row) is None)
        con.execute("UPDATE users SET api_tageslimit=1 WHERE id=?", (uid,)); con.commit()
        check("Wache: Limit 1, genutzt 1 -> gesperrt", tageslimit_wache(row) is not None)
        check("Wache: Admin ist ausgenommen (is_admin=True)", tageslimit_wache(row, is_admin=True) is None)
finally:
    con.execute("UPDATE users SET api_tageslimit=? WHERE id=?", (alt_limit, uid))
    con.execute("DELETE FROM usage_events WHERE user_id=? AND id > ?", (uid, start_max))
    con.commit()
    check("Aufgeraeumt: Limit und Ereignisse wie vorher",
          con.execute("SELECT api_tageslimit FROM users WHERE id=?", (uid,)).fetchone()["api_tageslimit"] == alt_limit
          and con.execute("SELECT COUNT(*) FROM usage_events WHERE user_id=? AND id > ?", (uid, start_max)).fetchone()[0] == 0)
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
