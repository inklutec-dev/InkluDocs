#!/usr/bin/env python3
"""End-to-End Strukturlesung/Hoerprobe/Strukturansicht gegen Staging (22.09.2026): getaggtes Formular
hochladen, Hoerprobe lesen (Felder mit Namen, Tabelle, Seiten, Sprache), Seite /struktur pruefen
(H1, Tabelle, Formularfelder, Escaping), Rechte (fremdes Projekt 404, ohne Login -> /login),
ungetaggte PDF -> verfuegbar=false mit Grund.
Aufruf: verify_struktur.py <URL> <mail> <pw> <formular_getaggt.pdf> [--behalten]
"""
import http.cookiejar
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

B, MAIL, PW, PDF = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4]
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


def req(method, pfad, body=None, files=None, opener=None):
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
        with (opener or op).open(r, timeout=180) as resp:
            return resp.status, resp.read(), resp.headers, resp.geturl()
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers, e.geturl()


def js(raw):
    try:
        return json.loads(raw.decode())
    except Exception:
        return {}


def rohpdf() -> bytes:
    import fitz
    d = fitz.open()
    p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Ungetaggtes Blatt", fontsize=18)
    p.insert_text((50, 100), "Nur ein Absatz ohne Tags.", fontsize=10)
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


s, b, _h, _u = req("POST", "/api/login", {"email": MAIL, "password": PW})
check(s == 200, f"Login {s}")

s, b, _h, _u = req("POST", "/api/projects", {"name": "E2E Struktur " + time.strftime("%d.%m. %H:%M"), "tool": "pdf"})
check(s == 200, f"Projekt anlegen {s} {b[:120]}")
pid = js(b).get("id") or js(b).get("project_id")
with open(PDF, "rb") as f:
    pdf = f.read()
s, b, _h, _u = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("formular_getaggt.pdf", pdf, "application/pdf")})
check(s == 200, f"Upload {s} {b[:160]}")
doc = warte_dokument(pid)
check(doc is not None, "Dokument nach Upload vorhanden")
doc_id = doc["id"]
check(doc.get("getaggt") in (1, True), f"Dokument ist getaggt: {doc.get('getaggt')}")

# Hoerprobe
t0 = time.time()
s, b, _h, _u = req("GET", f"/api/projects/{pid}/documents/{doc_id}/struktur")
st = js(b)
print(f"Strukturlesung: {s} in {time.time() - t0:.1f}s, {len(st.get('hoerprobe') or [])} Zeilen")
check(s == 200 and st.get("verfuegbar") is True, f"Struktur verfuegbar {s} {b[:200]}")
z = st.get("hoerprobe") or []
check(z and z[0].startswith("Sprache:"), f"Erste Zeile Sprache: {z[:1]}")
check(len(z) > 1 and z[1].startswith("Seiten: ") and (st.get("info") or {}).get("seiten", 0) >= 1, f"Seitenzeile: {z[1:2]} info={st.get('info')}")
check(len(z) > 2 and z[2].startswith("Zusammenfassung:") and st.get("zusammenfassung") == z[2], f"Zusammenfassung: {z[2:3]}")
check(any(x.startswith("Formularfeld vorname") for x in z), "Formularfeld vorname in der Hoerprobe")
check(any(x.startswith("Formularfeld email: E-Mail-Adresse") for x in z), "Quickinfo aus der Datei (email /TU) in der Hoerprobe")
check(any(x.startswith("Tabelle mit ") for x in z), "Tabelle in der Hoerprobe")
check(any(x.startswith("Kopfzeile: ") for x in z), "Kopfzeile (TH) in der Hoerprobe")
check(any(x.startswith("Absatz: Musterbank") for x in z), "Absatz (P) in der Hoerprobe")
check(any(x.startswith("Formularfeld zahlungsweise") for x in z), "Optionsfeld: Name vom Elternfeld (zahlungsweise)")
check(sum(1 for x in z if x.startswith("Formularfeld ")) >= 12, f"alle Felder in der Hoerprobe: {sum(1 for x in z if x.startswith('Formularfeld '))}")
check(any(x.startswith("Tabelle mit 17 Zeilen und 6 Spalten") for x in z), "Tabellenmasse nachgezaehlt (17 x 6)")
check("— Seite 1 —" in z, "Seitenmarke Seite 1")
check(st.get("seite_url") == f"/struktur/{pid}/{doc_id}", f"seite_url: {st.get('seite_url')}")
# Quickinfo aus der Datenbank ergaenzt die Datei
s2, b2, _h, _u = req("GET", f"/api/projects/{pid}/felder")
_j = js(b2)
felder = (_j.get("felder") if isinstance(_j, dict) else _j) or []
feld = next((f for f in felder if f.get("feld_name") == "nachname"), None)
if feld:
    s3, b3, _h, _u = req("PATCH", f"/api/felder/{feld['id']}", {"quickinfo": "Nachname wie im Ausweis"})
    if s3 in (200, 204):
        st2 = js(req("GET", f"/api/projects/{pid}/documents/{doc_id}/struktur")[1])
        check(any(x == "Formularfeld nachname: Nachname wie im Ausweis" for x in (st2.get("hoerprobe") or [])), "Quickinfo aus der Datenbank in der Hoerprobe")
    else:
        print("Hinweis: Quickinfo-PATCH nicht moeglich:", s3, b3[:100])
else:
    print("Hinweis: Feld nachname nicht in", [f.get("feld_name") or f.get("name") for f in felder][:8], "->", s2)
# Cache: zweiter Aufruf schnell, erneuern=1 liefert dasselbe
t1 = time.time()
st3 = js(req("GET", f"/api/projects/{pid}/documents/{doc_id}/struktur")[1])
check(time.time() - t1 < 3 and len(st3.get("hoerprobe") or []) == len(st2.get("hoerprobe") or z) if feld else True, "Zweiter Aufruf aus dem Cache")
st4 = js(req("GET", f"/api/projects/{pid}/documents/{doc_id}/struktur?erneuern=1")[1])
check((st4.get("info") or {}).get("elemente") == (st.get("info") or {}).get("elemente"), "erneuern=1 liefert dieselbe Elementzahl")

# Seite
s, b, _h, url = req("GET", f"/struktur/{pid}/{doc_id}")
html = b.decode("utf-8", "replace")
check(s == 200 and 'id="strukturTitel"' in html, f"Seite /struktur {s}")
check("Strukturansicht: formular_getaggt.pdf" in html, "H1 mit Dokumentname")
check("<table" in html and '<th scope="col">' in html, "Tabelle mit Kopfzellen auf der Seite")
check("Formularfeld vorname" in html, "Formularfeld auf der Seite")
check('id="strukturInhalt"' in html and 'id="strukturHoerprobe"' in html and "<h2" in html, "Inhalt und Hoerprobe-Abschnitt")
check(f'href="/app?projekt={pid}&amp;ansicht=dokument"' in html or f'href="/app?projekt={pid}&ansicht=dokument"' in html, "Zurueck-Link zum Projekt")
check(html.count("<h1") == 1, "genau eine H1")
check("<script>alert" not in html, "kein ungefilterter Skript-Text")

# Rechte
s, b, _h, _u = req("GET", f"/api/projects/999999/documents/{doc_id}/struktur")
check(s == 404, f"fremdes Projekt 404: {s}")
s, b, _h, _u = req("GET", f"/struktur/999999/{doc_id}")
check(s == 404, f"Seite fremdes Projekt 404: {s}")
op2 = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
s, b, _h, url = req("GET", f"/struktur/{pid}/{doc_id}", opener=op2)
check("/login" in url or s in (302, 303, 401), f"ohne Login -> Login: {s} {url}")
s, b, _h, _u = req("GET", f"/api/projects/{pid}/documents/{doc_id}/struktur", opener=op2)
check(s == 401, f"API ohne Login 401: {s}")

# Ungetaggte PDF
s, b, _h, _u = req("POST", "/api/projects", {"name": "E2E Struktur roh " + time.strftime("%H:%M"), "tool": "pdf"})
pid2 = js(b).get("id") or js(b).get("project_id")
s, b, _h, _u = req("POST", "/api/upload", {"project_id": pid2}, files={"file": ("roh.pdf", rohpdf(), "application/pdf")})
doc2 = warte_dokument(pid2)
if doc2:
    st5 = js(req("GET", f"/api/projects/{pid2}/documents/{doc2['id']}/struktur")[1])
    check(st5.get("verfuegbar") is False and "Tags" in (st5.get("grund") or ""), f"ungetaggt: verfuegbar=false mit Grund: {st5}")
    s, b, _h, _u = req("GET", f"/struktur/{pid2}/{doc2['id']}")
    check(s == 200 and 'id="strukturGrund"' in b.decode("utf-8", "replace"), f"Seite ungetaggt zeigt den Grund: {s}")
else:
    check(False, "rohes Dokument nach Upload vorhanden")

if not BEHALTEN:
    for p in (pid, pid2):
        s, _b, _h, _u = req("DELETE", f"/api/projects/{p}")
        print("Projekt geloescht:", p, s)
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})")
sys.exit(1 if fehlt else 0)
