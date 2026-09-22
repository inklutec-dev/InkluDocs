#!/usr/bin/env python3
"""Fertige PDF (22.09.2026): PDF-Projekt mit dem Testformular -> Tagging -> eine Quickinfo von Hand ->
Export EINES Dokuments liefert eine PDF mit Struktur UND der Quickinfo; Ablage bekommt einen Eintrag (art pdf)
mit Datei, Bericht und Vorschau; ZIP-Export legt je Dokument einen Eintrag an.
Aufruf: verify_export_komplett.py <URL> <mail> <pw> <testformular.pdf> [--behalten]"""
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
        with op.open(r, timeout=180) as resp: return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e: return e.code, e.read(), dict(e.headers)
def js(raw):
    try: return json.loads(raw.decode())
    except Exception: return {}
def warten(pid):
    for _ in range(60):
        time.sleep(2); d = js(req("GET", f"/api/projects/{pid}")[1])
        if (d.get("project") or {}).get("status") != "extracting": return d
    return {}
s, b, _ = req("POST", "/api/login", {"email": MAIL, "password": PW}); check(s == 200, "Login")
s, b, _ = req("POST", "/api/projects", {"name": "E2E Fertige PDF " + time.strftime("%H:%M"), "tool": "pdf"}); pid = js(b).get("id") or js(b).get("project_id"); check(pid, "Projekt")
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("export_formular.pdf", open(FORM, "rb").read(), "application/pdf")}); check(s == 200, f"Upload {s}")
d = warten(pid); doc_id = (d.get("documents") or [{}])[0].get("id"); check(doc_id, "Dokument")
# Tagging
s, b, _ = req("POST", f"/api/projects/{pid}/documents/{doc_id}/tagging", {}); check(s == 200, f"Tagging gestartet {s} {b[:100]}")
for _ in range(90):
    time.sleep(2); st = js(req("GET", f"/api/projects/{pid}/documents/{doc_id}/tagging")[1])
    if st.get("status") in ("fertig", "fehler"): break
check(st.get("status") == "fertig", f"Tagging fertig: {st.get('status')}")
# Eine Quickinfo von Hand
f = js(req("GET", f"/api/projects/{pid}/felder")[1]); felder = [x for x in (f.get("felder") or []) if not str(x.get("anker", "")).startswith("#")]
check(len(felder) >= 10, f"Felder gelesen: {len(felder)}")
feld = felder[0]; s, b, _ = req("PATCH", f"/api/felder/{feld['id']}", {"quickinfo": "Testquickinfo aus dem E2E-Lauf"}); check(s == 200, f"Quickinfo gesetzt {s} {b[:100]}")
# Export eines Dokuments
s, b, h = req("POST", f"/api/projects/{pid}/export", {"document_id": doc_id})
check(s == 200 and b[:5] == b"%PDF-", f"Export liefert PDF: {s} {b[:120] if s != 200 else b''}")
check(h.get("X-Ausgabe-Id") or h.get("x-ausgabe-id"), "Antwort nennt den Ablage-Eintrag (X-Ausgabe-Id)")
try:
    import fitz
    dd = fitz.open(stream=b, filetype="pdf")
    tus = {w.field_name: (w.field_label or "") for pg in dd for w in pg.widgets()}
    check(any("Testquickinfo aus dem E2E-Lauf" in v for v in tus.values()), f"Quickinfo steht in der Export-PDF: {[v for v in tus.values() if v][:3]}")
    check("pdfuaid" in dd.get_xml_metadata() or dd.xref_get_key(dd.pdf_catalog(), "StructTreeRoot")[0] != "null", "Export-PDF traegt Struktur")
    check((dd.xref_get_key(dd.pdf_catalog(), "Lang")[1] or "").startswith("de"), f"Sprache in der Export-PDF: {dd.xref_get_key(dd.pdf_catalog(), 'Lang')}")
except ImportError:
    print("(fitz fehlt lokal)")
# Ablage
a = js(req("GET", f"/api/ausgaben?projekt={pid}")[1]); eintraege = a.get("ausgaben") or []
check(len(eintraege) == 1 and eintraege[0].get("art") == "pdf" and eintraege[0].get("datei_verfuegbar") and eintraege[0].get("vorschau"), f"Ablage: 1 Eintrag art pdf mit Datei und Vorschau: {eintraege[:1]}")
if eintraege:
    e1 = js(req("GET", f"/api/ausgaben/{eintraege[0]['id']}")[1]); ber = (e1.get("bericht") or e1.get("ausgabe", {}).get("bericht") or [])
    print("Ablage-Zusammenfassung:", (eintraege[0].get("zusammenfassung") or "")[:120])
    check(isinstance(ber, list) and ber and "pruefung" in ber[0], "Ablage-Bericht mit PDF/UA-Pruefung")
    s, b, _ = req("GET", f"/api/ausgaben/{eintraege[0]['id']}/datei"); check(s == 200 and b[:5] == b"%PDF-", "Ablage-Datei ladbar")
    s, b, _ = req("GET", f"/api/ausgaben/{eintraege[0]['id']}/vorschau"); check(s == 200 and b[:4] == b"\x89PNG", "Ablage-Vorschau ladbar")
# ZIP-Export (alle Dokumente) -> weiterer Eintrag
s, b, h = req("POST", f"/api/projects/{pid}/export", {}); check(s == 200, f"ZIP/Einzel-Export ohne document_id: {s}")
a = js(req("GET", f"/api/ausgaben?projekt={pid}")[1]); check(len(a.get("ausgaben") or []) == 2, f"Ablage hat 2 Eintraege: {len(a.get('ausgaben') or [])}")
if not BEHALTEN: print("geloescht:", req("DELETE", f"/api/projects/{pid}")[0])
else: print("Projekt bleibt stehen:", pid)
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})"); sys.exit(1 if fehlt else 0)
