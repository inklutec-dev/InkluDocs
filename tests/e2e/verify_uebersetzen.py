"""End-to-End-Test Uebersetzen-Werkzeug gegen Staging (18.09.2026).
Aufruf: python3 verify_uebersetzen.py [<basis-url> <email> <passwort> <fixture.docx>] [--behalten]
Zugangsdaten alternativ aus INKLUDOCS_E2E_URL/_MAIL/_PW. Nur Standardbibliothek + zipfile.
Legt ein Projekt an, uebersetzt mit dem ECHTEN Modell (kostet wenige Credits), prueft den
Export und loescht das Projekt (ausser --behalten). Alle Daten fiktiv. Muster: verify_formular.py.
"""
import http.cookiejar
import io
import json
import os
import re
import sys
import time
import urllib.request
import uuid
import zipfile
from xml.etree import ElementTree as ET

if len(sys.argv) >= 5 and not sys.argv[1].startswith("--"):
    BASE, MAIL, PW, FIX = sys.argv[1:5]
else:
    BASE = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
    MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
    FIX = os.environ.get("INKLUDOCS_E2E_FIXTURE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "testvortrag_inkludocs.docx"))
    if not MAIL or not PW:
        sys.exit("Zugangsdaten fehlen: INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW setzen (oder 4 Argumente uebergeben)")
BEHALTEN = "--behalten" in sys.argv
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
ok = fehler = 0
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, info)


def req(method, path, data=None, files=None, raw=False):
    url = BASE + path
    headers, body = {}, None
    if files:
        boundary = "----inkludocs" + uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in (data or {}).items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fn, content, ct) in files.items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: {ct}\r\n\r\n".encode())
            buf.write(content); buf.write(b"\r\n")
        buf.write(f"--{boundary}--\r\n".encode())
        body = buf.getvalue(); headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif data is not None:
        body = json.dumps(data).encode(); headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        resp = opener.open(r, timeout=300)
        content = resp.read()
        return resp.status, (content if raw else (json.loads(content) if content else {})), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        content = e.read()
        try:
            return e.code, json.loads(content), {k.lower(): v for k, v in e.headers.items()}
        except Exception:
            return e.code, {"raw": content[:200]}, {k.lower(): v for k, v in e.headers.items()}


def warte(pid, weg_von, max_s=600):
    t0 = time.time()
    while time.time() - t0 < max_s:
        s, b, _ = req("GET", f"/api/projects/{pid}/uebersetzung")
        if s == 200 and b["project"]["status"] != weg_von:
            return b
        time.sleep(3)
    return None


def texte(docx_bytes, part="word/document.xml"):
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        root = ET.fromstring(zf.read(part))
    out = []
    for p in root.iter(W + "p"):
        t = "".join((x.text or "") for x in p.iter(W + "t"))
        if t.strip():
            out.append(t)
    return out


# A Login
s, b, _ = req("POST", "/api/login", {"email": MAIL, "password": PW})
check("Login", s == 200, b)
s, b, _ = req("GET", "/api/tools")
werkzeug = next((t for t in b.get("tools", []) if t["key"] == "uebersetzen"), None)
check("Werkzeug „uebersetzen“ in der Liste, anlegbar", werkzeug and werkzeug["is_available"], werkzeug)

# B Projekt + Negativfaelle
s, b, _ = req("POST", "/api/projects", {"name": "E2E Übersetzen (fiktiv)", "tool": "uebersetzen"})
check("Projekt angelegt (project_type docx-uebersetzung)", s == 200 and b.get("project_type") == "docx-uebersetzung", b)
pid = b["project_id"]
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, {"file": ("x.pdf", b"%PDF-1.4 nicht wirklich", "application/pdf")})
check("PDF in Übersetzungsprojekt -> 400", s == 400, b)
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, {"file": ("kaputt.docx", b"kein zip", "application/octet-stream")})
check("Kaputte docx -> 400 mit Meldung", s == 400 and "Word" in json.dumps(b), b)
s, b, _ = req("POST", f"/api/projects/{pid}/uebersetzung/starten", {"zielsprache": "xx"})
check("Unbekannte Zielsprache -> 400", s == 400, b)
s, b, _ = req("POST", f"/api/projects/{pid}/uebersetzung/starten", {"zielsprache": "en"})
check("Start ohne Dokument: nichts zu tun", s == 200 and b.get("gestartet") is False, b)

# C Upload + Segmentierung
fix = open(FIX, "rb").read()
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, {"file": (os.path.basename(FIX), fix, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
check("Upload angenommen (extracting)", s == 200 and b.get("status") == "extracting", b)
doc_id = b.get("document_id")
data = warte(pid, "extracting", 120)
check("Segmentierung fertig (extracted)", data and data["project"]["status"] == "extracted", data and data["project"]["status"])
segs = data["segmente"] if data else []
ue = [x for x in segs if x["uebersetzbar"]]
check("Absätze gelesen (>= 12 übersetzbare)", len(ue) >= 12, len(ue))
check("Alt-Text/Bildtitel/Dokumenttitel als Segmente", any(x["art"] == "dokumenttitel" for x in segs), [x["art"] for x in segs])
check("Kopfzeile als Ort erkannt", any(x["ort"] == "Kopfzeile" for x in segs))
check("Keine Serverpfade/Marken-Innereien nach außen", all("stuecke" not in x and "original_path" not in x for x in segs) and "original_path" not in data["project"])
check("Zielsprachen-Liste mitgeliefert (>= 10)", len(data.get("zielsprachen", [])) >= 10)
check("Hinweise je Dokument als Objekt", isinstance(data["documents"][0].get("hinweise"), dict))

# D Vorschau + Lauf
s, v, _ = req("POST", f"/api/projects/{pid}/uebersetzung/vorschau", {})
check("Vorschau: Anzahl, Wörter, Preis", s == 200 and v["anzahl"] == len(ue) and v["woerter"] > 0 and v["preis"] >= 1, v)
check("Vorschau: Preis = ceil(Wörter/100)", v["preis"] == -(-v["woerter"] // v["woerter_je_credit"]), v)
s, b, _ = req("POST", f"/api/projects/{pid}/uebersetzung/starten", {"zielsprache": "en", "alt_texte": True, "sprache_setzen": True})
check("Lauf gestartet", s == 200 and b.get("gestartet") is True, b)
s, b2, _ = req("POST", f"/api/projects/{pid}/uebersetzung/starten", {"zielsprache": "en"})
check("Zweiter Start während des Laufs -> 409", s == 409, b2)
s, b3, _ = req("POST", f"/api/projects/{pid}/export/uebersetzung", {})
check("Export während des Laufs -> 409", s == 409, b3)
data = warte(pid, "processing", 600)
check("Lauf beendet", data is not None)
segs = data["segmente"]; ue = [x for x in segs if x["uebersetzbar"]]
fertig = [x for x in ue if x["status"] in ("fertig", "zusammengelegt")]
check("Alle übersetzbaren Segmente übersetzt", len(fertig) == len(ue), f"{len(fertig)}/{len(ue)} " + json.dumps([(x['anker'], x['status'], x['hinweis'][:60]) for x in ue if x['status'] not in ('fertig','zusammengelegt')]))
check("Übersetzungen englisch (Stichprobe)", any("Accessible" in x["uebersetzung"] or "accessible" in x["uebersetzung"] for x in fertig), [x["uebersetzung"][:40] for x in fertig[:3]])
check("Formatierung nicht zusammengelegt (keine Ersatzwege)", all(x["status"] == "fertig" for x in fertig), [x["anker"] for x in fertig if x["status"] != "fertig"])
lauf = data.get("lauf") or {}
check("Lauf-Status: Credits verbucht (>= 1)", lauf.get("credits", 0) >= 1, lauf)
check("Kopf-Info: Einstellungen des Laufs gespeichert", data["project"].get("einstellungen", {}).get("zielsprache") == "en", data["project"].get("einstellungen"))

# E Handkorrektur
ziel = next(x for x in ue if x["art"] == "absatz" and x["ueberschrift_ebene"] is None and x["ort"] == "Text")
s, b, _ = req("PATCH", f"/api/uebersetzung/segmente/{ziel['id']}", {"uebersetzung": "Manually corrected sentence (fictional)."})
check("Handkorrektur gespeichert (status hand)", s == 200 and b.get("status") == "hand", b)
s, b, _ = req("PATCH", f"/api/uebersetzung/segmente/{ziel['id']}", {"falsch": 1})
check("PATCH ohne Feld -> 400", s == 400, b)
s, b, _ = req("PATCH", f"/api/uebersetzung/segmente/999999999", {"uebersetzung": "x"})
check("Fremdes/unbekanntes Segment -> 404", s == 404, b)

# F Export + Ruecklesen
s, raw, h = req("POST", f"/api/projects/{pid}/export/uebersetzung", {"filename": "Vortrag EN (fiktiv)"}, raw=True)
check("Export 200 als docx", s == 200 and raw[:2] == b"PK", (s, h.get("content-type")))
check("Content-Disposition mit Wunschname", "Vortrag" in h.get("content-disposition", ""), h.get("content-disposition"))
en_texte = texte(raw)
orig_texte = texte(fix)
check("Gleiche Zahl an Absätzen wie das Original", len(en_texte) == len(orig_texte), (len(en_texte), len(orig_texte)))
check("Handkorrektur in der Datei", any("Manually corrected" in t for t in en_texte))
check("Original-Text nicht mehr in der Datei", not any("Barrierefreie Dokumente im Alltag" == t for t in en_texte))
with zipfile.ZipFile(io.BytesIO(raw)) as zf:
    st = zf.read("word/styles.xml").decode("utf-8")
    core = zf.read("docProps/core.xml").decode("utf-8")
    kopf = texte(raw, "word/header1.xml")
check("Sprachkennung en-US in styles.xml", 'w:val="en-US"' in st)
check("dc:language en-US in core.xml", "en-US" in core)
check("Dokumenttitel übersetzt", "Accessible" in core or "accessible" in core, re.findall(r"<dc:title>(.*?)</dc:title>", core))
check("Kopfzeile übersetzt", kopf and "Lecture" in kopf[0] or "lecture" in (kopf[0] if kopf else ""), kopf)
# Struktur: Original und Export haben dieselben Zip-Mitglieder, Bilder byteidentisch
with zipfile.ZipFile(io.BytesIO(fix)) as a, zipfile.ZipFile(io.BytesIO(raw)) as b_:
    check("Zip-Mitglieder identisch", a.namelist() == b_.namelist())
    check("Nicht-XML-Teile byteidentisch", all(a.read(n) == b_.read(n) for n in a.namelist() if not n.endswith(".xml")))
    orig_doc = a.read("word/document.xml").decode(); neu_doc = b_.read("word/document.xml").decode()
    check("Gleiche Zahl Läufe mit Fettung (w:b) wie im Original", orig_doc.count("<w:b/>") == neu_doc.count("<w:b/>"), (orig_doc.count("<w:b/>"), neu_doc.count("<w:b/>")))
    check("Hyperlink erhalten", orig_doc.count("<w:hyperlink") == neu_doc.count("<w:hyperlink") == 1)
    check("Tabelle erhalten", orig_doc.count("<w:tbl>") == neu_doc.count("<w:tbl>") == 1)

# G Fremdzugriff: zweiter Nutzer sieht nichts (falls Zweitkonto gesetzt)
MAIL2, PW2 = os.environ.get("INKLUDOCS_E2E_MAIL2", ""), os.environ.get("INKLUDOCS_E2E_PW2", "")
if MAIL2 and PW2:
    cj.clear(); s, b, _ = req("POST", "/api/login", {"email": MAIL2, "password": PW2})
    s, b, _ = req("GET", f"/api/projects/{pid}/uebersetzung")
    check("Fremdzugriff auf Projekt -> 404", s == 404, s)
    s, b, _ = req("PATCH", f"/api/uebersetzung/segmente/{ziel['id']}", {"uebersetzung": "x"})
    check("Fremdzugriff auf Segment -> 404", s == 404, s)
    s, b, _ = req("POST", f"/api/projects/{pid}/export/uebersetzung", {})
    check("Fremdexport -> 404", s == 404, s)
    cj.clear(); req("POST", "/api/login", {"email": MAIL, "password": PW})
else:
    print("INFO  Fremdzugriff: kein Zweitkonto (INKLUDOCS_E2E_MAIL2) — übersprungen")

# H Abgemeldet
cj.clear()
s, b, _ = req("GET", f"/api/projects/{pid}/uebersetzung")
check("Ohne Login -> 401", s == 401, s)
req("POST", "/api/login", {"email": MAIL, "password": PW})

# I Aufraeumen
if BEHALTEN:
    print(f"INFO  Projekt {pid} bleibt (--behalten): {BASE}/app?projekt={pid}")
else:
    s, b, _ = req("DELETE", f"/api/projects/{pid}")
    check("Projekt gelöscht", s == 200, b)
    s, b, _ = req("GET", f"/api/projects/{pid}/uebersetzung")
    check("Nach Löschen -> 404", s == 404, s)

print(f"\n{ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
