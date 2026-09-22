#!/usr/bin/env python3
"""E2E Alt-Text-Uebernahme beim Tagging (22.09.2026): ungetaggtes PDF hochladen (Bild ueber fitz
extrahiert), Alt-Text generieren (echtes Modell, 5 Credits), dann taggen -> das neu ueber PDFix
extrahierte Bild muss den Alt-Text tragen (uebernommen = 1). Danach Neu-Taggen -> weiter uebernommen.
Aufruf: verify_tagging_uebernahme.py <URL> <mail> <pw> [--behalten]"""
import http.cookiejar, io, json, sys, time, urllib.error, urllib.request, uuid

B, MAIL, PW = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
BEHALTEN = "--behalten" in sys.argv
ok_n, fehlt = 0, []


def check(bed, text):
    global ok_n
    if bed:
        ok_n += 1
    else:
        fehlt.append(text); print("FEHLT:", text)


cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def req(method, pfad, body=None, files=None):
    headers, data = {}, None
    if files:
        bnd = uuid.uuid4().hex; buf = io.BytesIO()
        for k, v in (body or {}).items():
            buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fn, inhalt, mt) in files.items():
            buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: {mt}\r\n\r\n".encode()); buf.write(inhalt); buf.write(b"\r\n")
        buf.write(f"--{bnd}--\r\n".encode()); data = buf.getvalue(); headers["Content-Type"] = f"multipart/form-data; boundary={bnd}"
    elif body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
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
    d = fitz.open(); p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Uebernahme-Test", fontsize=20)
    text = ("Der Bericht fasst die Massnahmen des Landes zusammen und richtet sich an die Oeffentlichkeit. "
            "Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen. ") * 4
    y = 110
    for zeile in [text[i:i + 95] for i in range(0, len(text), 95)]:
        p.insert_text((50, y), zeile, fontsize=10); y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0)
    for yy in range(80):
        for xx in range(120):
            pix.set_pixel(xx, yy, (int(255 * xx / 120), int(255 * yy / 80), 90 + ((xx // 8 + yy // 8) % 2) * 100))
    p.insert_image(fitz.Rect(50, y + 20, 250, y + 150), pixmap=pix)
    p.insert_text((50, y + 170), "Abbildung 1: Farbverlauf mit Schachbrettmuster als Testbild.", fontsize=10)
    out = d.tobytes(); d.close(); return out


s, b = req("POST", "/api/login", {"email": MAIL, "password": PW}); check(s == 200, "Login")
s, b = req("POST", "/api/projects", {"name": "E2E Tagging Uebernahme " + time.strftime("%H:%M"), "tool": "pdf"})
pid = js(b).get("id") or js(b).get("project_id"); check(pid, "Projekt")
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("uebernahme_roh.pdf", testpdf(), "application/pdf")}); check(s == 200, f"Upload {s}")
for _ in range(40):
    time.sleep(2); d = js(req("GET", f"/api/projects/{pid}")[1])
    if (d.get("documents") or []) and (d.get("project") or {}).get("status") != "extracting":
        break
doc_id = d["documents"][0]["id"]; imgs = d.get("images") or []
check(len(imgs) == 1 and d["documents"][0].get("extraction_method") == "fitz", f"1 Bild ueber fitz vor dem Tagging: {len(imgs)} {d['documents'][0].get('extraction_method')}")
alt_bbox = (imgs[0].get("bbox_x0"), imgs[0].get("bbox_y0"), imgs[0].get("bbox_x1"), imgs[0].get("bbox_y1")) if imgs else None
print("bbox fitz:", alt_bbox)
# Alt-Text generieren (echtes Modell)
s, b = req("POST", f"/api/projects/{pid}/generate", {}); check(s == 200, f"Generierung gestartet {s} {b[:150]}")
alt = ""
for _ in range(60):
    time.sleep(3); d = js(req("GET", f"/api/projects/{pid}")[1])
    st = (d.get("project") or {}).get("status"); im = (d.get("images") or [{}])[0]
    if im.get("status") in ("done", "error") and st != "processing":
        alt = im.get("alt_text") or ""; break
check(bool(alt), f"Alt-Text generiert: {alt[:80]!r}")
# Handtext dazu, damit auch alt_text_edited wandert
s, b = req("POST", f"/api/images/{im.get('id')}/alt-text", {"alt_text": alt + " [bearbeitet]"})
print("Handtext-Endpunkt:", s)
# Taggen
s, b = req("POST", f"/api/projects/{pid}/documents/{doc_id}/tagging", {}); check(s == 200, f"Tagging gestartet {s} {b[:120]}")
for _ in range(90):
    time.sleep(2); stt = js(req("GET", f"/api/projects/{pid}/documents/{doc_id}/tagging")[1])
    if stt.get("status") in ("fertig", "fehler"):
        break
check(stt.get("status") == "fertig", f"Tagging fertig: {json.dumps(stt.get('bericht'))[:200]}")
bl = (stt.get("bericht") or {}).get("bilder") or {}
print("Bilder:", bl)
check(bl.get("methode") == "pdfix" and bl.get("nachher") == 1, "nach dem Tagging 1 Bild ueber PDFix")
check(bl.get("uebernommen") == 1, f"Alt-Text uebernommen: {bl.get('uebernommen')}")
d = js(req("GET", f"/api/projects/{pid}")[1]); im2 = (d.get("images") or [{}])[0]
print("bbox pdfix:", (im2.get("bbox_x0"), im2.get("bbox_y0"), im2.get("bbox_x1"), im2.get("bbox_y1")), "| status:", im2.get("status"))
check((im2.get("alt_text") or "") == alt, "Alt-Text am neuen Bild identisch")
check(im2.get("status") == "done", f"Status done uebernommen: {im2.get('status')}")
if s == 200:
    check((im2.get("alt_text_edited") or "").endswith("[bearbeitet]"), f"Handtext uebernommen: {im2.get('alt_text_edited')!r}")
check(stt.get("hat_alt_texte") == 1, f"Stand: hat_alt_texte = {stt.get('hat_alt_texte')}")
# Neu-Taggen: Uebernahme pdfix -> pdfix
s, b = req("POST", f"/api/projects/{pid}/documents/{doc_id}/tagging", {}); check(s == 200, "Neu-Taggen gestartet")
for _ in range(90):
    time.sleep(2); stt = js(req("GET", f"/api/projects/{pid}/documents/{doc_id}/tagging")[1])
    if stt.get("status") in ("fertig", "fehler"):
        break
bl = (stt.get("bericht") or {}).get("bilder") or {}
check(stt.get("status") == "fertig" and bl.get("uebernommen") == 1, f"Neu-Taggen: Alt-Text erneut uebernommen {bl}")
d = js(req("GET", f"/api/projects/{pid}")[1]); im3 = (d.get("images") or [{}])[0]
check((im3.get("alt_text") or "") == alt, "Alt-Text nach Neu-Taggen identisch")
# Export der getaggten PDF mit Alt-Text muss jetzt erlaubt sein (vorher: ungetaggt = 422)
s, b = req("POST", f"/api/projects/{pid}/export", {"format": "pdf", "document_id": doc_id})
print("Export:", s, b[:160] if s != 200 else "PDF-Export ok")
check(s in (200, 402), f"PDF-Export nach dem Tagging erlaubt (200) oder nur Credits (402): {s}")
if not BEHALTEN:
    print("geloescht:", req("DELETE", f"/api/projects/{pid}")[0])
else:
    print("Projekt bleibt stehen:", pid)
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})")
sys.exit(1 if fehlt else 0)
