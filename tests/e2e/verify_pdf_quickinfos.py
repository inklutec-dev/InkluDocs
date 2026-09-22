#!/usr/bin/env python3
"""Station „Quickinfos“ im PDF-Projekt (22.09.2026): ein PDF-Formular in einem PDF-Projekt hochladen ->
Bilder UND Felder werden gelesen; /api/projects/{id} meldet hat_felder; /felder liefert die Felder;
dokument-ansicht zaehlt sie je Dokument; Ansicht „quickinfos“ ist speicherbar; der Quickinfo-Export
laeuft; ein PDF ohne Felder bekommt keine Station. Werkzeugliste nach Dateiart.
Aufruf: verify_pdf_quickinfos.py <URL> <mail> <pw> <testformular.pdf> [--behalten]"""
import http.cookiejar, io, json, sys, time, urllib.error, urllib.request, uuid
B, MAIL, PW, FORM = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4]
BEHALTEN = "--behalten" in sys.argv
ok_n, fehlt = 0, []
def check(bed, text):
    global ok_n
    if bed: ok_n += 1
    else: fehlt.append(text); print("FEHLT:", text)
cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
def req(method, pfad, body=None, files=None):
    headers, data = {}, None
    if files:
        bnd = uuid.uuid4().hex; buf = io.BytesIO()
        for k, v in (body or {}).items(): buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fn, inhalt, mt) in files.items():
            buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: {mt}\r\n\r\n".encode()); buf.write(inhalt); buf.write(b"\r\n")
        buf.write(f"--{bnd}--\r\n".encode()); data = buf.getvalue(); headers["Content-Type"] = f"multipart/form-data; boundary={bnd}"
    elif body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    r = urllib.request.Request(B + pfad, data=data, method=method, headers=headers)
    try:
        with op.open(r, timeout=180) as resp: return resp.status, resp.read()
    except urllib.error.HTTPError as e: return e.code, e.read()
def js(raw):
    try: return json.loads(raw.decode())
    except Exception: return {}
s, b = req("POST", "/api/login", {"email": MAIL, "password": PW}); check(s == 200, "Login")
tools = js(req("GET", "/api/tools")[1]).get("tools", [])
namen = {t["key"]: t for t in tools}
check(namen.get("pdf", {}).get("name") == "PDF-Dokumente", f"Werkzeug pdf heisst PDF-Dokumente: {namen.get('pdf', {}).get('name')}")
check(namen.get("web", {}).get("name") == "Webseiten" and namen.get("grafik", {}).get("name") == "Grafiken", "Webseiten / Grafiken umbenannt")
check(namen.get("formular", {}).get("sichtbar") is False, "formular nicht mehr im Anlege-Menue (Kennung bleibt)")
check("pdf-a11y" not in namen, "Platzhalter pdf-a11y entfernt")
s, b = req("POST", "/api/projects", {"name": "E2E PDF-Quickinfos " + time.strftime("%H:%M"), "tool": "pdf"})
pid = js(b).get("id") or js(b).get("project_id"); check(pid, f"PDF-Projekt angelegt {s}")
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("testformular.pdf", open(FORM, "rb").read(), "application/pdf")}); check(s == 200, f"Upload {s} {b[:120]}")
for _ in range(60):
    time.sleep(2); d = js(req("GET", f"/api/projects/{pid}")[1])
    if (d.get("documents") or []) and (d.get("project") or {}).get("status") != "extracting": break
p = d.get("project") or {}; docs = d.get("documents") or []
check(p.get("status") == "extracted" and docs, f"Projekt extracted mit Dokument: {p.get('status')}")
check(docs and docs[0].get("extraction_method") in ("pdfix", "fitz"), f"Bild-Extraktionsweg bleibt: {docs and docs[0].get('extraction_method')}")
check(p.get("hat_felder") == 12, f"hat_felder = 12: {p.get('hat_felder')}")
f = js(req("GET", f"/api/projects/{pid}/felder")[1])
check(len(f.get("felder") or []) == 12, f"/felder liefert 12 Felder: {len(f.get('felder') or [])}")
check((f.get("project") or {}).get("hat_felder") == 12 and (f.get("project") or {}).get("tool") == "pdf", "felder: Projekt mit hat_felder und tool pdf")
da = js(req("GET", f"/api/projects/{pid}/dokument-ansicht")[1])
check((da.get("documents") or [{}])[0].get("felder") == 12 and (da.get("project") or {}).get("hat_felder") == 12, "dokument-ansicht zaehlt 12 Felder")
s, b = req("POST", f"/api/projects/{pid}/ansicht", {"ansicht": "quickinfos"}); check(s == 200, f"Ansicht quickinfos speicherbar: {s}")
check(js(req("GET", f"/api/projects/{pid}")[1]).get("project", {}).get("letzte_ansicht") == "quickinfos", "letzte_ansicht = quickinfos")
s, b = req("POST", f"/api/projects/{pid}/stammdaten-anwenden", {"nur_offene": True}); check(s == 200, f"Formular-Endpunkt fuer PDF-Projekt erlaubt: {s} {b[:100]}")
s, b = req("POST", f"/api/projects/{pid}/export/formular", {}); check(s in (200, 402), f"Quickinfo-Export aus PDF-Projekt: {s} {b[:100] if s != 200 else b'ok'}")
s, b = req("POST", f"/api/projects/{pid}/export/formular_csv", {}); check(s in (200, 402), f"Feldliste CSV aus PDF-Projekt: {s}")
# Dokument loeschen raeumt Felder mit auf
did = docs[0]["id"]
s, b = req("DELETE", f"/api/projects/{pid}/documents/{did}"); check(s == 200, f"Dokument loeschen {s}")
check(js(req("GET", f"/api/projects/{pid}")[1]).get("project", {}).get("hat_felder") == 0, "nach dem Loeschen keine Felder mehr")
# PDF ohne Felder: keine Station
import fitz
dd = fitz.open(); pg = dd.new_page(); pg.insert_text((50, 70), "Ohne Felder", fontsize=20); roh = dd.tobytes(); dd.close()
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("ohne_felder.pdf", roh, "application/pdf")}); check(s == 200, "Upload ohne Felder")
for _ in range(40):
    time.sleep(2); d = js(req("GET", f"/api/projects/{pid}")[1])
    if (d.get("project") or {}).get("status") != "extracting": break
check((d.get("project") or {}).get("hat_felder") == 0, "PDF ohne Felder: hat_felder 0")
s, b = req("POST", f"/api/projects/{pid}/ansicht", {"ansicht": "quickinfos"}); check(s == 200, "Ansicht quickinfos bleibt speicherbar (Wache ist die Oberflaeche)")
if not BEHALTEN: print("geloescht:", req("DELETE", f"/api/projects/{pid}")[0])
else: print("Projekt bleibt stehen:", pid)
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})"); sys.exit(1 if fehlt else 0)
