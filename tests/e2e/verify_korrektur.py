#!/usr/bin/env python3
"""End-to-End Korrektur (Stufe 2, 22.09.2026) gegen Staging: rohes PDF (Ueberschrift 14 pt allein, Liste, Bild)
-> Tagging -> Pruefung (Messwerte, Doppelbeleg) -> Korrektur (kostenlos, Sicherung) -> Struktur veraendert,
Pruefbericht „von vor der Korrektur“, 409 bei zweiter Korrektur -> Rueckgaengig -> Struktur wie vorher.
Aufruf: verify_korrektur.py <URL> <mail> <pw> [--behalten]   (echter Modelllauf, kostet Credits)"""
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


cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def req(method, pfad, body=None, files=None):
    headers = {}; data = None
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
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read() or b"{}")
        except Exception: return e.code, {}


def testpdf() -> bytes:
    import fitz
    d = fitz.open(); p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Jahresbericht Naturschutz 2025", fontsize=20, fontname="hebo")
    y = 110
    text = ("Der Bericht fasst die Massnahmen des Landes im Jahr 2025 zusammen und richtet sich an die "
            "Oeffentlichkeit. Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen. ") * 3
    for zeile in [text[i:i + 95] for i in range(0, len(text), 95)]:
        p.insert_text((50, y), zeile, fontsize=10); y += 14
    p.insert_text((50, y + 28), "1. Ausgangslage", fontsize=14, fontname="hebo"); y += 60
    for zeile in ["Ankauf von 120 Hektar Wald im Schwarzwald", "Renaturierung von zwei Bachlaeufen"]:
        p.insert_text((60, y), "• " + zeile, fontsize=10); y += 14
    p.insert_text((50, y + 28), "2. Ausblick", fontsize=14, fontname="hebo"); y += 60
    p.insert_text((50, y), "Absatz des Ausblicks mit der Planung fuer das kommende Jahr.", fontsize=10)
    out = d.tobytes(); d.close(); return out


def warte(pid, did, pfad, n=120):
    for _ in range(n):
        time.sleep(2)
        s, b = req("GET", f"/api/projects/{pid}/documents/{did}/{pfad}")
        if b.get("status") in ("fertig", "fehler"):
            return b
    return b


def struktur_kurz(pid, did):
    s, st = req("GET", f"/api/projects/{pid}/documents/{did}/struktur")
    return [z for z in st.get("hoerprobe") or [] if z.startswith(("Überschrift", "Absatz", "Listenpunkt", "Liste"))]


s, b = req("POST", "/api/login", {"email": MAIL, "password": PW}); check(s == 200, f"Login {s}")
s, b = req("POST", "/api/projects", {"name": "E2E Korrektur " + time.strftime("%d.%m. %H:%M"), "tool": "pdf"})
pid = b.get("id") or b.get("project_id")
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("bericht_roh.pdf", testpdf(), "application/pdf")}); check(s == 200, f"Upload {s}")
did = None
for _ in range(45):
    time.sleep(2); s, d = req("GET", f"/api/projects/{pid}")
    docs = d.get("documents") or []
    if docs and (d.get("project") or {}).get("status") != "extracting":
        did = docs[0]["id"]; break
check(did is not None, "Dokument vorhanden")

# Korrektur ohne Pruefung -> 400
s, b = req("POST", f"/api/projects/{pid}/documents/{did}/korrektur", {"erneut_pruefen": False})
check(s == 400, f"Korrektur ohne Pruefung = 400: {s} {b}")
s, b = req("POST", f"/api/projects/{pid}/documents/{did}/tagging", {"sprache": ""}); check(s == 200, f"Tagging {s}")
tg = warte(pid, did, "tagging"); check(tg.get("status") == "fertig", f"Tagging fertig: {tg.get('status')}")
s, b = req("POST", f"/api/projects/{pid}/documents/{did}/pruefung"); check(s == 200, f"Pruefung gestartet {s}")
pr = warte(pid, did, "pruefung"); check(pr.get("status") == "fertig", f"Pruefung fertig: {pr.get('status')}")
pb = pr.get("bericht") or {}
befunde = pb.get("befunde") or []
print("Befunde:", pb.get("anzahl"))
for f in befunde:
    print(f"  {'AUTO' if f.get('auto') else 'HINW'} S{f['seite']} {f['typ']} „{(f['text'] or '')[:40]}“ -> {f['vorschlag']} [{f['sicherheit']}] {f.get('messung') or ''} | {f.get('doppelbeleg') or ''}")
check(all("messung" in f and "auto" in f and "doppelbeleg" in f for f in befunde), "Befunde tragen messung/auto/doppelbeleg")
check(all(f.get("obj") for f in befunde if f.get("element")), "Befunde mit Element tragen obj (Strukturlesung v2)")
ko = pr.get("korrektur") or {}
check(ko.get("verfuegbar") is True and "auto_befunde" in ko and ko.get("sicherung") is False, f"Korrektur-Stand: {ko}")
auto = [f for f in befunde if f.get("auto")]
if not auto:
    print("Hinweis: kein Befund mit Doppelbeleg in diesem Lauf (Modellurteil) — Korrektur-Teil uebersprungen")
    s, b = req("POST", f"/api/projects/{pid}/documents/{did}/korrektur", {"erneut_pruefen": False})
    check(s == 400, f"Korrektur ohne Doppelbeleg = 400: {s}")
else:
    vorher = struktur_kurz(pid, did)
    s, b = req("POST", f"/api/projects/{pid}/documents/{did}/korrektur", {"erneut_pruefen": False})
    check(s == 200 and b.get("gestartet") and b.get("preis_nachpruefung") == 0, f"Korrektur gestartet, kostenlos: {s} {b}")
    for _ in range(40):
        time.sleep(2); s, pr2 = req("GET", f"/api/projects/{pid}/documents/{did}/pruefung")
        ko2 = pr2.get("korrektur") or {}
        if not ko2.get("laeuft") and ko2.get("bericht"):
            break
    kb = ko2.get("bericht") or {}
    print("Korrektur:", kb.get("anzahl"), "Aenderungen |", [(a["typ_vorher"], a["typ_nachher"], (a["text"] or "")[:25], a["status"]) for a in kb.get("angewendet") or []])
    check(not kb.get("fehler") and kb.get("anzahl") == len(auto), f"alle Doppelbeleg-Befunde angewendet: {kb.get('anzahl')} von {len(auto)} {kb.get('fehler')}")
    check(all(a["status"] == "angewendet" for a in kb.get("angewendet") or []), "jede Aenderung angewendet (Objektnummer gefunden)")
    check(bool(ko2.get("korrigiert_am")) and ko2.get("sicherung") is True, f"Pruefbericht markiert + Sicherung: {ko2.get('korrigiert_am')} {ko2.get('sicherung')}")
    check(kb.get("verapdf") is not None, "veraPDF nach der Korrektur")
    nachher = struktur_kurz(pid, did)
    check(nachher != vorher, "Hoerprobe nach Korrektur veraendert")
    print("  vorher :", vorher[:5]); print("  nachher:", nachher[:5])
    s, b = req("POST", f"/api/projects/{pid}/documents/{did}/korrektur", {"erneut_pruefen": False})
    check(s == 409, f"zweite Korrektur auf altem Bericht = 409: {s}")
    s, b = req("POST", f"/api/projects/{pid}/documents/{did}/korrektur/rueckgaengig")
    check(s == 200, f"Rueckgaengig {s} {b}")
    zurueck = struktur_kurz(pid, did)
    check(zurueck == vorher, "Hoerprobe nach Rueckgaengig wie vorher")
    s, pr3 = req("GET", f"/api/projects/{pid}/documents/{did}/pruefung")
    ko3 = pr3.get("korrektur") or {}
    check(not ko3.get("korrigiert_am") and ko3.get("sicherung") is False and not ko3.get("bericht"), f"Stand nach Rueckgaengig: {ko3}")
    s, b = req("POST", f"/api/projects/{pid}/documents/{did}/korrektur/rueckgaengig")
    check(s == 400, f"zweites Rueckgaengig = 400: {s}")
s, b = req("POST", f"/api/projects/999999/documents/{did}/korrektur", {"erneut_pruefen": False}); check(s == 404, f"fremdes Projekt 404: {s}")
if not BEHALTEN:
    req("DELETE", f"/api/projects/{pid}")
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})")
sys.exit(1 if fehlt else 0)
