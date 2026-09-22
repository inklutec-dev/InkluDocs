#!/usr/bin/env python3
"""Kette „Komplett barrierefrei machen“ (22.09.2026): PDF-Projekt mit einer ungetaggten PDF (1 Bild) und dem
Testformular (getaggt, 12 Felder) -> Vorschau (Umfang/Preise), Start, 409 waehrend des Laufs, Stand bis fertig,
danach: Dokument getaggt, Alt-Text erzeugt, Quickinfos erzeugt, Zusammenfassung. ECHTE Modellaufrufe (Credits).
Aufruf: verify_kette.py <URL> <mail> <pw> <testformular.pdf> [--behalten]"""
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
def bildpdf():
    import fitz
    d = fitz.open(); p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Kettentest", fontsize=20)
    text = ("Der Bericht fasst die Massnahmen des Landes zusammen und richtet sich an die Oeffentlichkeit. Die Flaeche der Schutzgebiete ist gewachsen. ") * 4
    y = 110
    for z in [text[i:i + 95] for i in range(0, len(text), 95)]: p.insert_text((50, y), z, fontsize=10); y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0)
    for yy in range(80):
        for xx in range(120): pix.set_pixel(xx, yy, (int(255 * xx / 120), int(255 * yy / 80), 90 + ((xx // 8 + yy // 8) % 2) * 100))
    p.insert_image(fitz.Rect(50, y + 20, 250, y + 150), pixmap=pix)
    p.insert_text((50, y + 170), "Abbildung 1: Farbverlauf mit Muster.", fontsize=10)
    out = d.tobytes(); d.close(); return out
def warten(pid):
    for _ in range(60):
        time.sleep(2); d = js(req("GET", f"/api/projects/{pid}")[1])
        if (d.get("project") or {}).get("status") not in ("extracting",): return d
    return {}
s, b = req("POST", "/api/login", {"email": MAIL, "password": PW}); check(s == 200, "Login")
verbraucht0 = (js(req("GET", "/api/me")[1]).get("abo") or {}).get("verbraucht")
s, b = req("POST", "/api/projects", {"name": "E2E Kette " + time.strftime("%H:%M"), "tool": "pdf"}); pid = js(b).get("id") or js(b).get("project_id"); check(pid, "Projekt")
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("kette_bild.pdf", bildpdf(), "application/pdf")}); check(s == 200, f"Upload Bild-PDF {s}"); warten(pid)
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("kette_formular.pdf", open(FORM, "rb").read(), "application/pdf")}); check(s == 200, f"Upload Formular {s}"); d = warten(pid)
docs = d.get("documents") or []
check(len(docs) == 2, f"2 Dokumente: {len(docs)}")
v = js(req("GET", f"/api/projects/{pid}/kette")[1])
print("Vorschau:", json.dumps({k: v.get(k) for k in ("tagging", "alttexte", "quickinfos", "gesamt", "erlaubt", "nichts_zu_tun", "laeuft")}, ensure_ascii=False)[:400])
# Beide Testdateien sind ungetaggt (auch das Testformular): 2 Dokumente, 3 Seiten.
check(v.get("tagging", {}).get("dokumente") == 2 and v["tagging"]["seiten"] == 3, f"Tagging: 2 Dokumente, 3 Seiten: {v.get('tagging')}")
check(v.get("alttexte", {}).get("bilder") == 1, f"Alt-Texte: 1 Bild: {v.get('alttexte')}")
check(v.get("quickinfos", {}).get("felder") == 12, f"Quickinfos: 12 Felder: {v.get('quickinfos')}")
check(v.get("gesamt") == v["tagging"]["preis"] + v["alttexte"]["preis"] + v["quickinfos"]["preis"] and v["gesamt"] == 3 + 5 + 12, f"Gesamtpreis 20: {v.get('gesamt')}")
check(v.get("erlaubt") is True and not v.get("laeuft"), "erlaubt, laeuft nicht")
s, b = req("GET", "/api/projects/999999/kette"); check(s == 404, f"fremdes Projekt 404: {s}")
s, b = req("POST", f"/api/projects/{pid}/kette", {}); st = js(b); check(s == 200 and st.get("gestartet"), f"Kette gestartet {s} {b[:150]}")
s2, b2 = req("POST", f"/api/projects/{pid}/kette", {}); check(s2 == 409, f"zweiter Start 409: {s2}")
fertig = None; schritte_gesehen = set()
for _ in range(150):
    time.sleep(3); k = js(req("GET", f"/api/projects/{pid}/kette")[1])
    stand = k.get("stand") or {}
    if stand.get("schritt"): schritte_gesehen.add(stand["schritt"])
    if stand and not stand.get("laeuft") and not k.get("laeuft"): fertig = stand; break
check(fertig is not None, "Kette beendet")
print("Stand:", json.dumps(fertig, ensure_ascii=False)[:600]); print("Schritte gesehen:", schritte_gesehen)
if fertig:
    sch = fertig.get("schritte") or {}
    check(fertig.get("schritt") == "fertig", f"Endstand fertig: {fertig.get('schritt')}")
    check(sch.get("tagging", {}).get("status") == "fertig" and sch["tagging"]["fertig"] == 2, f"Tagging fertig 2/2: {sch.get('tagging')}")
    check(sch.get("alttexte", {}).get("status") in ("fertig", "teilweise") and sch["alttexte"]["fertig"] >= 1, f"Alt-Texte erzeugt: {sch.get('alttexte')}")
    check(sch.get("quickinfos", {}).get("status") in ("fertig", "teilweise") and sch["quickinfos"]["fertig"] >= 10, f"Quickinfos erzeugt: {sch.get('quickinfos')}")
    check("Tagging: 2 von 2" in (fertig.get("zusammenfassung") or ""), f"Zusammenfassung: {fertig.get('zusammenfassung')}")
    d = js(req("GET", f"/api/projects/{pid}")[1]); docs = d.get("documents") or []; imgs = d.get("images") or []
    check(all(x.get("getaggt") in (1, True) for x in docs), f"alle Dokumente getaggt: {[x.get('getaggt') for x in docs]}")
    check(all(i.get("status") == "done" and (i.get("alt_text") or "") for i in imgs) and imgs, f"alle Bilder mit Alt-Text: {[(i.get('status'), bool(i.get('alt_text'))) for i in imgs]}")
    check((d.get("project") or {}).get("status") in ("extracted", "done"), f"Projektstatus nach der Kette: {(d.get('project') or {}).get('status')}")
    f = js(req("GET", f"/api/projects/{pid}/felder")[1]); mit = [x for x in (f.get("felder") or []) if (x.get("quickinfo") or "").strip()]
    check(len(mit) >= 10, f"Felder mit Quickinfo: {len(mit)}")
    da = js(req("GET", f"/api/projects/{pid}/dokument-ansicht")[1]); check((da.get("project") or {}).get("kette", {}).get("schritt") == "fertig", "dokument-ansicht liefert den Kettenstand")
    v1 = (js(req("GET", "/api/me")[1]).get("abo") or {}).get("verbraucht")
    if verbraucht0 is not None and v1 is not None:
        print("Credits verbraucht:", v1 - verbraucht0, "(Vorschau:", v["gesamt"], ")")
        check(v1 - verbraucht0 >= v["gesamt"] - 2 and v1 - verbraucht0 <= v["gesamt"] + 2, f"Verbrauch nahe am Vorschau-Preis: {v1 - verbraucht0} vs {v['gesamt']}")
if not BEHALTEN: print("geloescht:", req("DELETE", f"/api/projects/{pid}")[0])
else: print("Projekt bleibt stehen:", pid)
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})"); sys.exit(1 if fehlt else 0)
