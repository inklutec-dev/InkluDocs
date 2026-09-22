#!/usr/bin/env python3
"""End-to-End Automatische Pruefung gegen Staging (22.09.2026, ECHTER Modelllauf, kostet Credits):
ungetaggtes PDF (nummerierte Ueberschrift, Bild) hochladen -> Pruefung vor dem Tagging = 400 ->
taggen -> Stand (Preis 2 Credits je Seite) -> Pruefung starten -> 409 waehrend des Laufs -> warten ->
Bericht (Struktur, Befunde mit Kennung/Seite/Sicherheit) -> Credits verbucht -> fremdes Projekt 404.
Aufruf: verify_pruefung.py <URL> <mail> <pw> [--behalten]
"""
import http.cookiejar
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

B, MAIL, PW = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
BEHALTEN = "--behalten" in sys.argv
ok_n, fehlt = 0, []


def check(bed, text):
    global ok_n
    if bed:
        ok_n += 1
    else:
        fehlt.append(text)
        print("FEHLT:", text)


cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def req(method, pfad, body=None, files=None):
    headers = {}
    data = None
    if files:
        bnd = uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in (body or {}).items():
            buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fn, inhalt, mt) in files.items():
            buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: {mt}\r\n\r\n".encode())
            buf.write(inhalt)
            buf.write(b"\r\n")
        buf.write(f"--{bnd}--\r\n".encode())
        data = buf.getvalue()
        headers["Content-Type"] = f"multipart/form-data; boundary={bnd}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(B + pfad, data=data, method=method, headers=headers)
    try:
        with op.open(r, timeout=180) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def js(raw):
    try:
        return json.loads(raw.decode())
    except Exception:
        return {}


def testpdf() -> bytes:
    import fitz
    d = fitz.open()
    p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Jahresbericht Naturschutz 2025", fontsize=20)
    y = 110
    text = ("Der Bericht fasst die Massnahmen des Landes im Jahr 2025 zusammen und richtet sich an die "
            "Oeffentlichkeit. Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen. ") * 3
    for zeile in [text[i:i + 95] for i in range(0, len(text), 95)]:
        p.insert_text((50, y), zeile, fontsize=10)
        y += 14
    p.insert_text((50, y + 24), "1. Ausgangslage", fontsize=14)
    y += 54
    for zeile in ["Ankauf von 120 Hektar Wald im Schwarzwald", "Renaturierung von zwei Bachlaeufen"]:
        p.insert_text((60, y), "• " + zeile, fontsize=10)
        y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0)
    for yy in range(80):
        for xx in range(120):
            pix.set_pixel(xx, yy, (int(255 * xx / 120), int(255 * yy / 80), 90 + ((xx // 8 + yy // 8) % 2) * 100))
    p.insert_image(fitz.Rect(50, y + 20, 250, y + 150), pixmap=pix)
    p.insert_text((50, y + 170), "Die Grafik zeigt einen stetigen Anstieg der Flaeche.", fontsize=10)
    p2 = d.new_page(width=595, height=842)
    p2.insert_text((50, 70), "2. Ausblick", fontsize=14)
    p2.insert_text((50, 100), "Absatz des Ausblicks mit der Planung fuer das kommende Jahr.", fontsize=10)
    out = d.tobytes()
    d.close()
    return out


def warte_dokument(pid):
    for _i in range(45):
        time.sleep(2)
        d = js(req("GET", f"/api/projects/{pid}")[1])
        docs = d.get("documents") or []
        if docs and (d.get("project") or {}).get("status") not in ("extracting",):
            return docs[0]
    return None


s, b = req("POST", "/api/login", {"email": MAIL, "password": PW})
check(s == 200, f"Login {s}")
me = js(req("GET", "/api/me")[1])
verbraucht_vorher = (me.get("abo") or {}).get("verbraucht")

s, b = req("POST", "/api/projects", {"name": "E2E Pruefung " + time.strftime("%d.%m. %H:%M"), "tool": "pdf"})
pid = js(b).get("id") or js(b).get("project_id")
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("bericht_roh.pdf", testpdf(), "application/pdf")})
check(s == 200, f"Upload {s}")
doc = warte_dokument(pid)
check(doc is not None, "Dokument nach Upload")
did = doc["id"]

# Vor dem Tagging: 400
s, b = req("POST", f"/api/projects/{pid}/documents/{did}/pruefung")
check(s == 400, f"Pruefung vor dem Tagging = 400: {s} {b[:120]}")
st = js(req("GET", f"/api/projects/{pid}/documents/{did}/pruefung")[1])
check(st.get("status") in ("", None) and st.get("seiten") == 2 and st.get("preis") == 4, f"Stand vor dem Lauf: {st}")

# Tagging
s, b = req("POST", f"/api/projects/{pid}/documents/{did}/tagging", {"sprache": ""})
check(s == 200, f"Tagging gestartet {s} {b[:120]}")
for _ in range(90):
    time.sleep(2)
    tg = js(req("GET", f"/api/projects/{pid}/documents/{did}/tagging")[1])
    if tg.get("status") in ("fertig", "fehler"):
        break
check(tg.get("status") == "fertig", f"Tagging fertig: {tg.get('status')} {tg.get('bericht', {}).get('fehler')}")
check((tg.get("pruefung") or {}).get("preis") == 4 and (tg.get("pruefung") or {}).get("modell"), f"Stand im Tagging-Stand: {tg.get('pruefung')}")

# Pruefung
t0 = time.time()
s, b = req("POST", f"/api/projects/{pid}/documents/{did}/pruefung")
check(s == 200 and js(b).get("status") == "laeuft" and js(b).get("preis") == 4, f"Pruefung gestartet: {s} {b[:160]}")
s2, b2 = req("POST", f"/api/projects/{pid}/documents/{did}/pruefung")
check(s2 == 409, f"zweiter Start waehrend des Laufs = 409: {s2}")
pr = {}
for _ in range(120):
    time.sleep(2)
    pr = js(req("GET", f"/api/projects/{pid}/documents/{did}/pruefung")[1])
    if pr.get("status") in ("fertig", "fehler"):
        break
print(f"Pruefung: {pr.get('status')} in {time.time() - t0:.0f}s")
check(pr.get("status") == "fertig", f"Pruefung fertig: {pr.get('status')} {pr.get('bericht', {}).get('fehler')}")
ber = pr.get("bericht") or {}
check(ber.get("seiten") == 2 and ber.get("seiten_geprueft") == 2, f"2 Seiten geprueft: {ber.get('seiten')}/{ber.get('seiten_geprueft')}")
check(isinstance(ber.get("befunde"), list) and isinstance(ber.get("anzahl"), dict) and len(ber.get("je_seite") or []) == 2, "Berichtsstruktur")
print("Modell:", ber.get("modell"), "| Dauer:", ber.get("dauer_s"), "s | Befunde:", ber.get("anzahl"))
for f in ber.get("befunde") or []:
    print(f"  Seite {f['seite']} {f['element']} {f['typ']} „{f['text'][:40]}“ [{f['art']}/{f['sicherheit']}]: {f['befund']} -> {f['vorschlag']} | {f['beleg'][:80]}")
for j in ber.get("je_seite") or []:
    print(f"  Seite {j['seite']}: {j['zusammenfassung']}")
for f in ber.get("befunde") or []:
    check(f["seite"] in (1, 2) and f["sicherheit"] in ("hoch", "mittel", "niedrig") and f["befund"], f"Befund vollstaendig: {f}")
    if f["element"]:
        check(not f["hinweis"], f"Kennung bekannt: {f['element']} {f['hinweis']}")
# Erwartung aus dem Vormittag: „1. Ausgangslage“ als LI — ein Modell mit Seitenbild sollte das sehen
treffer = [f for f in ber.get("befunde") or [] if "Ausgangslage" in (f.get("text") or "") and f["art"] in ("rolle", "ebene")]
print("Befund „1. Ausgangslage“ als Ueberschrift erkannt:", bool(treffer), "(Modellurteil, kein harter Test)")

me2 = js(req("GET", "/api/me")[1])
verbraucht = (me2.get("abo") or {}).get("verbraucht")
if verbraucht_vorher is not None and verbraucht is not None:
    check(verbraucht - verbraucht_vorher >= 4 + 2, f"Credits verbucht (Tagging 2 + Pruefung 4): {verbraucht_vorher} -> {verbraucht}")
s, b = req("GET", f"/api/projects/999999/documents/{did}/pruefung")
check(s == 404, f"fremdes Projekt 404: {s}")

if not BEHALTEN:
    s, _b = req("DELETE", f"/api/projects/{pid}")
    print("Projekt geloescht:", s)
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})")
sys.exit(1 if fehlt else 0)
