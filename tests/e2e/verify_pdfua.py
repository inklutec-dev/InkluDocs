"""Barrierefreie PDF aus Word — End-to-End gegen Staging (29.08.2026).
Aufruf: python3 verify_pdfua.py <URL> <mail> <pw> <projekt_id eines Word-Projekts>
(z. B. das Projekt, das verify_docx.py mit --behalten stehen laesst)."""
import http.cookiejar, json, sys, urllib.request, zipfile, io

URL, MAIL, PW, PID = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", str(info)[:300])


def req(method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    try:
        with op.open(r, timeout=400) as resp:
            b = resp.read()
            return resp.status, (b if raw else json.loads(b or b"{}")), dict(resp.headers)
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b), dict(e.headers)
        except Exception:
            return e.code, b, dict(e.headers)


s, b, _ = req("POST", "/api/login", {"email": MAIL, "password": PW})
check("Login", s == 200, b)
s, b, _ = req("POST", f"/api/projects/{PID}/export/pdfua", {})
check("Umwandlung antwortet 200 mit Token", s == 200 and b.get("ok") and len(b.get("token", "")) == 24, (s, b))
if s == 200:
    print("     Zusammenfassung:", b.get("zusammenfassung"))
    doks = b.get("dokumente") or [{}]
    for dok in doks:
        print("     Dokument:", dok.get("dokument"), "| Bilder", dok.get("bilder"), "| Alt-Texte", dok.get("alt_texte"),
              "| bestanden:", dok.get("pruefung", {}).get("bestanden"))
        for p in dok.get("pruefung", {}).get("punkte", []):
            if p["status"] != "ok" or len(doks) == 1:
                print("       -", p["bereich"], "|", p["status"], "|", p["text"][:160])
    bilder = sum(int(d.get("bilder", 0)) for d in doks)
    check("Preis = 25 + 5 je angefangene 10 Bilder (alle Dokumente zusammen)", b.get("preis") == 25 + 5 * (-(-bilder // 10)), (b.get("preis"), bilder))
    check("Klartext-Punkte vorhanden (mind. 5 Kernbereiche)", all(len(d.get("pruefung", {}).get("punkte", [])) >= 5 for d in doks))
    check("veraPDF: ALLE Dokumente bestehen PDF/UA-1 (Stufe 2: Alt-Texte nachgetragen)", all(d.get("pruefung", {}).get("bestanden") is True for d in doks),
          [(d.get("dokument"), d.get("pruefung", {}).get("regeln_fehlgeschlagen"), d.get("nachbearbeitung")) for d in doks])
    check("Hoerprobe + Pruefbericht im Ergebnis", all(d.get("hoerprobe") and isinstance(d.get("pruefbericht"), list) for d in doks),
          [(len(d.get("hoerprobe") or []), len(d.get("pruefbericht") or [])) for d in doks])
    for d in doks:
        print("     Hoerprobe", d.get("dokument"), "->", " / ".join(d.get("hoerprobe", [])[:4])[:200])
        for bp in d.get("pruefbericht", []):
            print("       Pruefbericht:", bp["status"], "|", bp["text"][:120])
    sv, bv, _ = req("POST", f"/api/projects/{PID}/export/pdfua/vorschau", {})
    check("Vorschau-Endpunkt (kostenlos): Hoerprobe je Dokument", sv == 200 and len(bv.get("dokumente", [])) == len(doks)
          and all(x.get("hoerprobe") for x in bv["dokumente"]), (sv, str(bv)[:200]))
    s2, datei, h = req("GET", f"/api/projects/{PID}/export/pdfua/{b['token']}", raw=True)
    if len(doks) == 1:
        check("Download liefert PDF", s2 == 200 and datei[:5] == b"%PDF-", (s2, datei[:10]))
        check("Dateiname endet auf .pdf", str(b.get("dateiname", "")).endswith(".pdf"), b.get("dateiname"))
    else:
        namen = zipfile.ZipFile(io.BytesIO(datei)).namelist() if s2 == 200 and datei[:2] == b"PK" else []
        check("Download liefert ZIP mit einer PDF je Dokument", s2 == 200 and len(namen) == len(doks) and all(n.endswith(".pdf") for n in namen), (s2, namen))
        check("Dateiname endet auf _alle_pdfua.zip", str(b.get("dateiname", "")).endswith("_alle_pdfua.zip"), b.get("dateiname"))
    s3, _, _ = req("GET", f"/api/projects/{PID}/export/pdfua/deadbeefdeadbeefdeadbeef", raw=True)
    check("Fremder/unbekannter Token -> 404", s3 == 404, s3)
    s4, _, _ = req("GET", f"/api/projects/{PID}/export/pdfua/../../etc/passwd", raw=True)
    check("Pfad-Spielerei -> 404", s4 == 404, s4)
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
