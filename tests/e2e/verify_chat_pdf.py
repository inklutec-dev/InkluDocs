"""Chatbot bedient den PDF-Weg (Werkzeugsatz nach Dateiart, 22.09.2026) — End-to-End gegen Staging.
Aufruf: python3 verify_chat_pdf.py <URL> <mail> <pw> [--behalten]
LLM-gesteuert (INKLUAGENT_PROVIDER); geprueft werden die Werkzeugaufrufe, die serverseitige Rueckfrage
(kein Lauf ohne bestaetigt), Hintergrundlaeufe (Stand ueber die API), Anhang und Ablage. Kostet Credits."""
import http.cookiejar, io, json, sys, time, urllib.error, urllib.request, uuid

URL, MAIL, PW = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
BEHALTEN = "--behalten" in sys.argv
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", str(info)[:400])


def req(method, path, body=None, files=None, timeout=400):
    headers = {}
    data = None
    if files:
        bnd = uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in (body or {}).items():
            buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fn, inhalt, mt) in files.items():
            buf.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: {mt}\r\n\r\n".encode())
            buf.write(inhalt); buf.write(b"\r\n")
        buf.write(f"--{bnd}--\r\n".encode())
        data = buf.getvalue(); headers["Content-Type"] = f"multipart/form-data; boundary={bnd}"
    elif body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    r = urllib.request.Request(URL + path, data=data, method=method, headers=headers)
    try:
        with op.open(r, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw or b"{}")
            except Exception:
                return resp.status, {}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}


def chat(pid, text):
    t0 = time.time()
    s, b = req("POST", f"/api/projects/{pid}/chat", {"message": text})
    print(f"\n>> {text}\n<< ({s}, {time.time() - t0:.0f} s, Werkzeuge {b.get('werkzeuge')})\n{(b.get('reply') or '')[:600]}")
    return s, b


def testpdf() -> bytes:
    import fitz
    d = fitz.open()
    p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Jahresbericht Naturschutz 2025", fontsize=20)
    y = 110
    text = ("Der Bericht fasst die Massnahmen des Landes im Jahr 2025 zusammen und richtet sich an die "
            "Oeffentlichkeit. Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen. ") * 3
    for zeile in [text[i:i + 95] for i in range(0, len(text), 95)]:
        p.insert_text((50, y), zeile, fontsize=10); y += 14
    p.insert_text((50, y + 24), "1. Ausgangslage", fontsize=14); y += 54
    for zeile in ["Ankauf von 120 Hektar Wald im Schwarzwald", "Renaturierung von zwei Bachlaeufen"]:
        p.insert_text((60, y), "• " + zeile, fontsize=10); y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0)
    for yy in range(80):
        for xx in range(120):
            pix.set_pixel(xx, yy, (int(255 * xx / 120), int(255 * yy / 80), 90 + ((xx // 8 + yy // 8) % 2) * 100))
    p.insert_image(fitz.Rect(50, y + 20, 250, y + 150), pixmap=pix)
    p2 = d.new_page(width=595, height=842)
    p2.insert_text((50, 70), "2. Ausblick", fontsize=14)
    p2.insert_text((50, 100), "Absatz des Ausblicks mit der Planung fuer das kommende Jahr.", fontsize=10)
    out = d.tobytes(); d.close(); return out


def warte(pid, did, pfad, schluessel, ziel=("fertig", "fehler"), n=90):
    for _ in range(n):
        time.sleep(2)
        s, b = req("GET", f"/api/projects/{pid}/documents/{did}/{pfad}")
        if b.get(schluessel) in ziel:
            return b
    return b


s, b = req("POST", "/api/login", {"email": MAIL, "password": PW})
check("Login", s == 200, b)
s, b = req("POST", "/api/projects", {"name": "E2E Chat PDF " + time.strftime("%d.%m. %H:%M"), "tool": "pdf"})
pid = b.get("id") or b.get("project_id")
s, b = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("bericht_roh.pdf", testpdf(), "application/pdf")})
check("Upload", s == 200, b)
did = None
for _ in range(45):
    time.sleep(2)
    s, d = req("GET", f"/api/projects/{pid}")
    docs = d.get("documents") or []
    if docs and (d.get("project") or {}).get("status") != "extracting":
        did = docs[0]["id"]; break
check("Dokument vorhanden", did is not None)

# 1. Stand (kostenlos)
s, b = chat(pid, "Wie ist der Stand des Dokuments? Ist es schon getaggt?")
check("1 Stand: Werkzeug dokument_stand", s == 200 and "dokument_stand" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("1 Stand: Antwort nennt ungetaggt/Tagging/Seiten", any(w in (b.get("reply") or "").lower() for w in ("ungetaggt", "nicht getaggt", "tagging", "seiten", "getaggt")), (b.get("reply") or "")[:200])

# 2. Tagging: Rueckfrage, dann Ja
s, b = chat(pid, "Bitte mach das Dokument barrierefrei (Tagging).")
check("2 Tagging: Werkzeug barrierefrei_machen (Rueckfrage)", "barrierefrei_machen" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("2 Tagging: Antwort nennt Credits", "credit" in (b.get("reply") or "").lower(), (b.get("reply") or "")[:200])
s, tg = req("GET", f"/api/projects/{pid}/documents/{did}/tagging")
check("2 Tagging: Server hat NICHT gestartet (kein Lauf ohne Ja)", not tg.get("laeuft") and tg.get("status") in ("", None), tg.get("status"))
s, b = chat(pid, "Ja, bitte starten.")
check("3 Ja: barrierefrei_machen erneut aufgerufen", "barrierefrei_machen" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("3 Ja: Antwort sagt läuft/Hintergrund, nicht fertig", any(w in (b.get("reply") or "").lower() for w in ("läuft", "hintergrund", "gestartet", "dauert")), (b.get("reply") or "")[:200])
tg = warte(pid, did, "tagging", "status")
check("3 Tagging im Hintergrund fertig", tg.get("status") == "fertig", tg.get("status"))

# 4. Stand nach dem Lauf
s, b = chat(pid, "Ist das Tagging fertig? Wie sieht die Struktur aus?")
check("4 Stand: dokument_stand", "dokument_stand" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("4 Stand: Antwort nennt getaggt/Struktur/Überschrift", any(w in (b.get("reply") or "").lower() for w in ("getaggt", "struktur", "überschrift", "absätze", "elemente")), (b.get("reply") or "")[:200])

# 5. Hoerprobe
s, b = chat(pid, "Lies mir bitte die Hörprobe vor, wie ein Screenreader das Dokument liest.")
check("5 Hoerprobe: hoerprobe_lesen", "hoerprobe_lesen" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("5 Hoerprobe: Antwort enthaelt Zeilen (Absatz/Überschrift/Seite)", any(w in (b.get("reply") or "") for w in ("Absatz", "Überschrift", "Seite", "Grafik")), (b.get("reply") or "")[:200])

# 6. Pruefung: Rueckfrage, Ja, Bericht
s, b = chat(pid, "Starte bitte die automatische Prüfung.")
check("6 Pruefung: pruefung_starten (Rueckfrage)", "pruefung_starten" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
s, pr = req("GET", f"/api/projects/{pid}/documents/{did}/pruefung")
check("6 Pruefung: Server hat NICHT gestartet", not pr.get("laeuft") and pr.get("status") in ("", None), pr.get("status"))
s, b = chat(pid, "Ja.")
check("7 Ja: pruefung_starten erneut", "pruefung_starten" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
pr = warte(pid, did, "pruefung", "status")
check("7 Pruefung im Hintergrund fertig", pr.get("status") == "fertig", pr.get("status"))
s, b = chat(pid, "Was hat die Prüfung ergeben?")
check("8 Bericht: pruefbericht_lesen", "pruefbericht_lesen" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("8 Bericht: Antwort nennt Befund/Seite oder keine Befunde", any(w in (b.get("reply") or "").lower() for w in ("befund", "seite", "keine")), (b.get("reply") or "")[:200])

# 9. Fertige PDF: Rueckfrage, Ja -> Anhang + Ablage
s, l0 = req("GET", f"/api/ausgaben?projekt={pid}")
vorher = len(l0.get("ausgaben") or [])
s, b = chat(pid, "Exportiere bitte die fertige PDF.")
check("9 Export: exportiere_fertige_pdf (Rueckfrage)", "exportiere_fertige_pdf" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("9 Export: kein Anhang ohne Ja", not b.get("anhang"), b.get("anhang"))
s, b = chat(pid, "Ja, exportieren.")
check("10 Ja: exportiere_fertige_pdf erneut", "exportiere_fertige_pdf" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
anh = b.get("anhang") or []
check("10 Ja: Anhang mit download_url", anh and anh[0].get("download_url", "").startswith("/api/ausgaben/"), anh)
s, l1 = req("GET", f"/api/ausgaben?projekt={pid}")
check("10 Ja: Ablage hat einen Eintrag mehr (art pdf)", len(l1.get("ausgaben") or []) == vorher + 1 and (l1.get("ausgaben") or [{}])[0].get("art") == "pdf", l1.get("ausgaben", [])[:1])
if anh:
    r = urllib.request.Request(URL + anh[0]["download_url"])
    with op.open(r, timeout=60) as resp:
        check("10 Download beginnt mit %PDF", resp.read(5) == b"%PDF-")

# 11. Feld-Werkzeuge im PDF-Projekt (kein Feld -> saubere Antwort statt Fehler)
s, b = chat(pid, "Welche Formularfelder hat das Dokument?")
check("11 Felder: list_form_fields oder dokument_stand", any(w in (b.get("werkzeuge") or []) for w in ("list_form_fields", "dokument_stand")), b.get("werkzeuge"))
check("11 Felder: Antwort sagt keine Felder", any(w in (b.get("reply") or "").lower() for w in ("keine", "kein", "0 ")), (b.get("reply") or "")[:200])

if not BEHALTEN:
    req("DELETE", f"/api/projects/{pid}")
print(f"\nErgebnis: {ok} OK, {fehler} FEHLT (Projekt {pid})")
sys.exit(1 if fehler else 0)
