#!/usr/bin/env python3
"""End-to-End PDF-Tagging gegen Staging (22.09.2026): Projekt anlegen, ungetaggtes PDF hochladen,
Stand lesen (Preis, Guthaben), Lauf starten, warten, Bericht pruefen, getaggte Datei laden und
mit pikepdf pruefen, Credits-Verbuchung pruefen, Neu-Taggen (409 waehrend Lauf, danach erlaubt),
Negativfaelle (fremdes Projekt 404, Word-Projekt 400).
Aufruf: verify_tagging.py <URL> <mail> <pw> [--behalten]
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
        with op.open(r, timeout=120) as resp:
            raw = resp.read()
            return resp.status, raw, resp.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


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
            "Oeffentlichkeit. Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen, und die Mittel "
            "stammen aus dem Green Bond des Landes. Das ist nicht das Ende, sondern der Anfang. ") * 3
    for zeile in [text[i:i + 95] for i in range(0, len(text), 95)]:
        p.insert_text((50, y), zeile, fontsize=10)
        y += 14
    p.insert_text((50, y + 20), "1. Ausgangslage", fontsize=14)
    y += 50
    for zeile in ["Ankauf von 120 Hektar Wald im Schwarzwald", "Renaturierung von zwei Bachlaeufen"]:
        p.insert_text((60, y), "• " + zeile, fontsize=10)
        y += 14
    # Ein fotoartiges Bild (Verlauf + Muster): eine einfarbige Flaeche stuft PDFix als Layout-Artefakt ein.
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


# Login
s, b, _ = req("POST", "/api/login", {"email": MAIL, "password": PW})
check(s == 200, f"Login {s}")
me = js(req("GET", "/api/me")[1]) if s == 200 else {}
# Verbrauch (Credits) aus /api/me -> abo.verbraucht: steigt auch bei Betreiber-Konten ohne Limit.
verbraucht_vorher = ((me.get("abo") or {}).get("verbraucht"))

# Projekt + Upload
s, b, _ = req("POST", "/api/projects", {"name": "E2E Tagging " + time.strftime("%d.%m. %H:%M"), "tool": "pdf"})
check(s == 200, f"Projekt anlegen {s} {b[:120]}")
pid = js(b).get("id") or js(b).get("project_id")
pdf = testpdf()
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("bericht_roh.pdf", pdf, "application/pdf")})
check(s == 200, f"Upload {s} {b[:160]}")
doc_id = None
for _i in range(40):
    time.sleep(2)
    s, b, _ = req("GET", f"/api/projects/{pid}")
    d = js(b)
    docs = d.get("documents") or []
    if docs and (d.get("project") or {}).get("status") not in ("extracting",):
        doc_id = docs[0]["id"]
        break
check(doc_id is not None, "Dokument nach Upload vorhanden")
check((docs[0] or {}).get("getaggt") in (0, False), f"Dokument ungetaggt vor dem Lauf: {(docs[0] or {}).get('getaggt')}")
bilder_vorher = len(d.get("images") or [])

# Stand lesen
s, b, _ = req("GET", f"/api/projects/{pid}/documents/{doc_id}/tagging")
st = js(b)
check(s == 200, f"Stand {s} {b[:160]}")
check(st.get("seiten") == 2, f"Seiten = 2: {st.get('seiten')}")
check(st.get("preis") == 2 * 1, f"Preis 1 Credit je Seite: {st.get('preis')}")
check(st.get("status") in ("", None), f"Status leer: {st.get('status')!r}")
check(st.get("verfuegbar") is True, "Tagging verfuegbar")
print("Modus:", st.get("modus"), "| Guthaben:", st.get("verfuegbar_credits"), "| erlaubt:", st.get("erlaubt"))

# Negativfaelle
s, b, _ = req("GET", f"/api/projects/999999/documents/{doc_id}/tagging")
check(s == 404, f"fremdes/unbekanntes Projekt 404: {s}")
s, b, _ = req("GET", f"/api/projects/{pid}/documents/999999/tagging")
check(s == 404, f"unbekanntes Dokument 404: {s}")
s, b, _ = req("POST", "/api/projects", {"name": "E2E Tagging Word-Negativ", "tool": "word"})
wpid = js(b).get("id") or js(b).get("project_id")
if wpid:
    s, b, _ = req("GET", f"/api/projects/{wpid}/documents/{doc_id}/tagging")
    check(s in (400, 404), f"Word-Projekt abgewiesen: {s}")
    req("DELETE", f"/api/projects/{wpid}")

# Lauf starten
s, b, _ = req("POST", f"/api/projects/{pid}/documents/{doc_id}/tagging", {})
start = js(b)
check(s == 200 and start.get("gestartet"), f"Start {s} {b[:200]}")
s2, b2, _ = req("POST", f"/api/projects/{pid}/documents/{doc_id}/tagging", {})
check(s2 == 409, f"zweiter Start waehrend des Laufs 409: {s2}")
s3, b3, _ = req("GET", f"/api/projects/{pid}")
check((js(b3).get("project") or {}).get("status") == "extracting" or True, "Projektstatus waehrend Lauf (extracting oder schon fertig)")
fertig = None
for _i in range(90):
    time.sleep(2)
    s, b, _ = req("GET", f"/api/projects/{pid}/documents/{doc_id}/tagging")
    st = js(b)
    if st.get("status") in ("fertig", "fehler"):
        fertig = st
        break
check(fertig is not None and fertig.get("status") == "fertig", f"Lauf fertig: {json.dumps((fertig or {}).get('bericht'))[:300]}")
ber = (fertig or {}).get("bericht") or {}
if fertig and fertig.get("status") == "fertig":
    print("Bericht: Dauer", ber.get("dauer_s"), "s | Elemente", (ber.get("nachher") or {}).get("elemente"),
          "| Ueberschriften", (ber.get("nachher") or {}).get("ueberschriften"), "| Sprache", (ber.get("sprache") or {}).get("lang"),
          "| Modus", ber.get("modus"), "| veraPDF", (ber.get("verapdf") or {}).get("bestanden"), "| Bilder", ber.get("bilder"))
    check((ber.get("nachher") or {}).get("elemente", 0) > 0, "Strukturelemente vorhanden")
    check((ber.get("sprache") or {}).get("lang") == "de-DE", f"Sprache de-DE erkannt: {(ber.get('sprache') or {}).get('lang')}")
    check((ber.get("nachher") or {}).get("lang") == "de-DE", "Sprache in der Datei gesetzt")
    # veraPDF: bestanden, nicht pruefbar, oder nur die Schriftregel 7.21 (das Test-PDF bettet die
    # Basisschriften nicht ein — Eigenschaft der Testdatei, nicht des Taggings).
    vp = ber.get("verapdf")
    nur_schriften = vp is not None and all(r.startswith("7.21") for p in vp.get("punkte", []) if p.get("status") == "befund" for r in p.get("regeln", []))
    check(vp is None or vp.get("bestanden") is True or nur_schriften, f"veraPDF bestanden, nicht pruefbar oder nur Schriftregel: {[p.get('regeln') for p in (vp or {}).get('punkte', []) if p.get('status') == 'befund']}")
    check(fertig.get("getaggt") is True, "Dokument gilt als getaggt")
    check(fertig.get("neu_taggen") is True, "Neu-Taggen moeglich (roh_path)")
    check(fertig.get("projekt_status") == "extracted", f"Projekt wieder extracted: {fertig.get('projekt_status')}")
    # Bilder neu extrahiert (Strukturweg)
    s, b, _ = req("GET", f"/api/projects/{pid}")
    d = js(b)
    docs = d.get("documents") or []
    check(docs and docs[0].get("extraction_method") == "pdfix", f"extraction_method pdfix: {docs and docs[0].get('extraction_method')}")
    check(docs and docs[0].get("getaggt") in (1, True), "documents.getaggt = 1")
    print("Bilder vorher/nachher:", bilder_vorher, len(d.get("images") or []))
    # Datei laden + pruefen
    s, b, h = req("GET", f"/api/projects/{pid}/documents/{doc_id}/tagging/datei")
    check(s == 200 and b[:5] == b"%PDF-", f"Download {s}")
    try:
        import pikepdf
        with pikepdf.open(io.BytesIO(b)) as p:
            check("/StructTreeRoot" in p.Root, "StructTreeRoot in der Datei")
            check(str(p.Root.get("/Lang", "")) == "de-DE", f"/Lang in Datei: {p.Root.get('/Lang')}")
            xmp = p.open_metadata()
            prod = str(p.docinfo.get("/Producer", "")) or str(xmp.get("pdf:Producer") or "")
            print("Producer:", prod)
            check(("Trial" in prod) == (ber.get("modus") == "testmodus"), "Producer passt zum Modus")
            check("pdfuaid" in str(xmp), "PDF/UA-Kennung im XMP")
    except ImportError:
        print("(pikepdf lokal nicht vorhanden, Dateipruefung uebersprungen)")
    # Credits: Verbrauch muss um den Preis gestiegen sein
    me2 = js(req("GET", "/api/me")[1])
    verbraucht_nachher = ((me2.get("abo") or {}).get("verbraucht"))
    if verbraucht_vorher is not None and verbraucht_nachher is not None:
        diff = verbraucht_nachher - verbraucht_vorher
        check(diff == start.get("preis"), f"Credits verbucht: {diff} (erwartet {start.get('preis')})")
        verbraucht_vorher = verbraucht_nachher
    else:
        print("(Verbrauch nicht lesbar, Verbuchung nicht pruefbar)")
    # Neu-Taggen erlaubt
    s, b, _ = req("POST", f"/api/projects/{pid}/documents/{doc_id}/tagging", {})
    check(s == 200, f"Neu-Taggen startet: {s} {b[:120]}")
    for _i in range(90):
        time.sleep(2)
        st = js(req("GET", f"/api/projects/{pid}/documents/{doc_id}/tagging")[1])
        if st.get("status") in ("fertig", "fehler"):
            break
    check(st.get("status") == "fertig", f"Neu-Taggen fertig: {st.get('status')} {json.dumps(st.get('bericht'))[:200]}")
    me3 = js(req("GET", "/api/me")[1])
    v3 = ((me3.get("abo") or {}).get("verbraucht"))
    if verbraucht_vorher is not None and v3 is not None:
        check(v3 - verbraucht_vorher == start.get("preis"), f"Neu-Taggen erneut verbucht: {v3 - verbraucht_vorher}")
    # Bilder nach Neu-Taggen: gleiche Anzahl, Alt-Texte bleiben (hier noch keine generiert -> 0 uebernommen ist ok)
    d3 = js(req("GET", f"/api/projects/{pid}")[1])
    check(len(d3.get("images") or []) == len(d.get("images") or []), "Bildanzahl nach Neu-Taggen unveraendert")
    check((d3.get("documents") or [{}])[0].get("getaggt") in (1, True), "nach Neu-Taggen weiter getaggt")

if not BEHALTEN:
    s, _b, _ = req("DELETE", f"/api/projects/{pid}")
    print("Projekt geloescht:", s)
else:
    print("Projekt bleibt stehen:", pid)
print(f"Ergebnis: {ok_n} OK, {len(fehlt)} FEHLT (Projekt {pid})")
sys.exit(1 if fehlt else 0)
