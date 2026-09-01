#!/usr/bin/env python3
"""Einzel-Neu-Generieren, wenn die Bilddatei fehlt (Prod-Befund 01.09.2026, Projekt 406):
statt 500 ein ehrliches 404 mit Hinweis, Bildstatus bleibt wie er war. Laeuft IM Container
(braucht Dateisystem + DB): BASE, INKLUDOCS_E2E_MAIL, INKLUDOCS_E2E_PW aus der Umgebung.
Alle Daten fiktiv; das Testprojekt wird am Ende geloescht."""
import http.cookiejar, io, json, os, sqlite3, sys, time, urllib.request, uuid

BASE = os.environ["BASE"]; MAIL = os.environ["INKLUDOCS_E2E_MAIL"]; PW = os.environ["INKLUDOCS_E2E_PW"]
DB = os.environ.get("INKLUDOCS_DB", "/app/data/inkludocs.db")
cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
ok = fehler = 0
def check(name, cond, info=""):
    global ok, fehler
    if cond: ok += 1; print("  OK ", name)
    else: fehler += 1; print("  FEHLT", name, "--", str(info)[:300])
def req(m, path, data=None, files=None):
    if files:
        b = uuid.uuid4().hex; body = io.BytesIO()
        for k, v in (data or {}).items():
            body.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fn, content) in files.items():
            body.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode())
            body.write(content); body.write(b"\r\n")
        body.write(f"--{b}--\r\n".encode())
        r = urllib.request.Request(BASE + path, data=body.getvalue(), method=m, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    else:
        r = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data is not None else None, method=m, headers={"Content-Type": "application/json"})
    try:
        with op.open(r, timeout=120) as resp:
            c = resp.read(); return resp.status, (json.loads(c) if c else {})
    except urllib.error.HTTPError as e:
        c = e.read()
        try: return e.code, json.loads(c)
        except Exception: return e.code, {"raw": c[:300].decode(errors="replace")}

from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (240, 160), (30, 90, 160)).save(buf, format="PNG"); png = buf.getvalue()

print("== A. Login + Grafik-Projekt mit einem Bild ==")
s, _ = req("POST", "/api/login", {"email": MAIL, "password": PW}); check("Login", s == 200)
s, b = req("POST", "/api/projects", {"name": "Datei-weg-Test (fiktiv) " + time.strftime("%H:%M:%S"), "tool": "grafik"})
pid = b.get("id") or b.get("project_id"); check("Grafik-Projekt angelegt", s == 200 and pid, b)
s, b = req("POST", "/api/upload", {"project_id": pid}, {"file": ("testbild_fiktiv.png", png)}); check("Bild hochgeladen", s == 200, b)
for _ in range(20):
    s, p = req("GET", f"/api/projects/{pid}"); imgs = p.get("images") or []
    if imgs: break
    time.sleep(0.5)
check("Bild in der Projektansicht", len(imgs) == 1, imgs)
iid = imgs[0]["id"]
conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
row = conn.execute("SELECT image_path, status FROM images WHERE id = ?", (iid,)).fetchone()
check("Bilddatei liegt auf der Platte", row and os.path.isfile(row["image_path"]), dict(row) if row else None)
status_vorher = row["status"]

print("== B. Datei weg -> 404 statt 500, Status unveraendert ==")
os.remove(row["image_path"])
s, b = req("POST", f"/api/projects/{pid}/regenerate/{iid}", {})
check("Neu generieren antwortet 404 (nicht 500)", s == 404, (s, b))
check("Hinweis nennt das Neuladen", "neu laden" in str(b.get("detail", "")).lower(), b)
nach = conn.execute("SELECT status FROM images WHERE id = ?", (iid,)).fetchone()["status"]
check(f"Bildstatus bleibt '{status_vorher}' (nicht processing/error)", nach == status_vorher, nach)
s, b = req("POST", f"/api/projects/{pid}/regenerate/{iid + 100000}", {})
check("Unbekanntes Bild: 404", s == 404, (s, b))
s, b = req("POST", f"/api/projects/{pid + 100000}/regenerate/{iid}", {})
check("Fremdes/unbekanntes Projekt: 404", s == 404, (s, b))
conn.close()

print("== C. Aufraeumen ==")
s, _ = req("DELETE", f"/api/projects/{pid}"); check("Testprojekt geloescht (fehlende Datei ist kein Fehler)", s == 200, s)
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
