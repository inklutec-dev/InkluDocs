"""Chatbot bedient die Word-Werkzeuge (Meine Ausgaben Schritt 2, 11.09.2026) — End-to-End gegen Staging.
Aufruf: python3 verify_chat_ausgaben.py <URL> <mail> <pw> <projekt_id eines Word-Projekts>
Der Ablauf ist LLM-gesteuert (Gemini Flash); geprueft werden die Werkzeugaufrufe (werkzeuge), die
serverseitige Rueckfrage (kein Eintrag ohne bestaetigt), der Anhang unter der Antwort und das Regal.
Antworten werden ausgegeben, damit man den Ton mitlesen kann."""
import http.cookiejar, json, sys, time, urllib.request

URL, MAIL, PW, PID = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", str(info)[:400])


def req(method, path, body=None, timeout=400):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    try:
        with op.open(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}


def chat(text):
    t0 = time.time()
    s, b = req("POST", f"/api/projects/{PID}/chat", {"message": text})
    dauer = time.time() - t0
    print(f"\n>> {text}\n<< ({s}, {dauer:.0f} s, Werkzeuge {b.get('werkzeuge')})\n{(b.get('reply') or '')[:700]}")
    return s, b


s, b = req("POST", "/api/login", {"email": MAIL, "password": PW})
check("Login", s == 200, b)
# Alt-Texte vollstaendig (wie verify_pdfua), damit der Bot nicht erst Bilder beschriften will.
s, proj, = req("GET", f"/api/projects/{PID}")[:2]
imgs = proj.get("images") or []
for i in [i for i in imgs if not (i.get("alt_text_edited") or i.get("alt_text")) and i.get("original_alt") != "dekorativ"]:
    req("POST", f"/api/images/{i['id']}/alt-text", {"alt_text": f"Testtext für Bild {i['id']} (fiktiv, E2E)"})
check("Word-Projekt vorbereitet", proj.get("project", {}).get("project_type") == "docx", proj.get("project", {}).get("project_type"))
s, l0 = req("GET", f"/api/ausgaben?projekt={PID}")
vorher = len(l0.get("ausgaben") or [])
req("DELETE", f"/api/projects/{PID}/chat")

# 1. Pruefen (kostenlos)
s, b = chat("Prüfe bitte das Word-Dokument: Was würde ein Screenreader lesen und gibt es Hinweise?")
check("1 Pruefen: Werkzeug pruefe_word_dokument", s == 200 and "pruefe_word_dokument" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("1 Pruefen: Antwort nennt Titel/Überschrift/Bild oder Hinweis", any(w in (b.get("reply") or "") for w in ("Überschrift", "Titel", "Bild", "Hinweis", "Prüfbericht")), (b.get("reply") or "")[:200])
check("1 Pruefen: kein Anhang, keine Umwandlung", not b.get("anhang") and "konvertiere_zu_pdfua" not in (b.get("werkzeuge") or []))

# 2. Umwandeln verlangen -> serverseitige Rueckfrage (kein Eintrag ohne bestaetigt)
s, b = chat("Wandle das Dokument jetzt in eine barrierefreie PDF um.")
check("2 Umwandeln: Werkzeug konvertiere_zu_pdfua aufgerufen", s == 200 and "konvertiere_zu_pdfua" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("2 Umwandeln: Rueckfrage mit Preis (Credits) statt Anhang", "Credit" in (b.get("reply") or "") and not b.get("anhang"), (b.get("reply") or "")[:200])
s, l1 = req("GET", f"/api/ausgaben?projekt={PID}")
check("2 Umwandeln: KEIN neuer Eintrag ohne Bestaetigung (Server erzwingt die Rueckfrage)", len(l1.get("ausgaben") or []) == vorher, (vorher, len(l1.get("ausgaben") or [])))

# 3. Ja -> Umwandlung mit Anhang
s, b = chat("Ja, bitte umwandeln.")
anh = b.get("anhang") or []
check("3 Ja: Werkzeug konvertiere_zu_pdfua (bestaetigt)", s == 200 and "konvertiere_zu_pdfua" in (b.get("werkzeuge") or []), b.get("werkzeuge"))
check("3 Ja: Anhang mit Download-URL, Ausgaben-URL, ausgabe_id", len(anh) == 1 and anh[0].get("download_url", "").startswith("/api/ausgaben/") and "#ausgabe-" in anh[0].get("ausgaben_url", "") and anh[0].get("art") == "pdfua", anh)
check("3 Ja: Antwort nennt Ergebnis (bestanden/Hinweis) und den Ort (Ausgaben/herunterladen)", any(w in (b.get("reply") or "") for w in ("bestanden", "Hinweis")) and any(w in (b.get("reply") or "") for w in ("Ausgaben", "herunterladen", "Herunterladen")), (b.get("reply") or "")[:300])
aid = anh[0].get("ausgabe_id") if anh else None
s, l2 = req("GET", f"/api/ausgaben?projekt={PID}")
neu = [a for a in (l2.get("ausgaben") or []) if a.get("id") == aid]
check("3 Ja: Eintrag im Regal mit ausloeser=bot", neu and neu[0].get("ausloeser") == "bot" and neu[0].get("art") == "pdfua", neu[:1])
if aid:
    r = urllib.request.Request(URL + f"/api/ausgaben/{aid}/datei")
    with op.open(r, timeout=120) as resp:
        datei = resp.read()
    check("3 Ja: Datei aus dem Anhang ladbar (PDF oder ZIP)", datei[:5] == b"%PDF-" or datei[:2] == b"PK", datei[:8])
# Verlauf traegt den Anhang (nach Neuladen sichtbar)
s, h = req("GET", f"/api/projects/{PID}/chat/history")
letzte = [m for m in (h.get("messages") or []) if m.get("role") == "assistant"]
check("3 Ja: Verlauf speichert den Anhang", letzte and (letzte[-1].get("anhang") or [{}])[0].get("ausgabe_id") == aid, letzte[-1].get("anhang") if letzte else None)

# 4. Hoerprobe lesen
s, b = chat("Lies mir bitte den Anfang der Hörprobe dieser PDF vor.")
check("4 Hoerprobe: Werkzeug lies_ausgabe oder liste_ausgaben", s == 200 and any(w in (b.get("werkzeuge") or []) for w in ("lies_ausgabe", "liste_ausgaben")), b.get("werkzeuge"))
check("4 Hoerprobe: Antwort enthaelt Hoerprobe-Zeilen (Titel/Überschrift/Absatz)", any(w in (b.get("reply") or "") for w in ("Überschrift", "Titel", "Absatz", "Bild")), (b.get("reply") or "")[:200])

# 5. Word-Export mit Rueckfrage
s, b = chat("Gib mir das Word-Dokument mit den Alt-Texten.")
check("5 Word: Werkzeug exportiere_word (Rueckfrage)", s == 200 and "exportiere_word" in (b.get("werkzeuge") or []) and not b.get("anhang"), (b.get("werkzeuge"), b.get("anhang")))
s, b = chat("Ja.")
anh2 = b.get("anhang") or []
check("5 Word: nach Ja Anhang art=docx", len(anh2) == 1 and anh2[0].get("art") == "docx", anh2)
s, l3 = req("GET", f"/api/ausgaben?projekt={PID}")
w = [a for a in (l3.get("ausgaben") or []) if a.get("art") == "docx"]
check("5 Word: Eintrag art=docx im Regal, Dateiname .docx oder .zip", w and (w[0].get("dateiname", "").endswith(".docx") or w[0].get("dateiname", "").endswith(".zip")), w[:1])

# 6. Aufraeumen: Testeintraege loeschen
for a in (l3.get("ausgaben") or []):
    if a.get("id") not in [x.get("id") for x in (l0.get("ausgaben") or [])]:
        req("DELETE", f"/api/ausgaben/{a['id']}")
s, l4 = req("GET", f"/api/ausgaben?projekt={PID}")
check("6 Aufraeumen: Regal wieder auf Ausgangsstand", len(l4.get("ausgaben") or []) == vorher, (vorher, len(l4.get("ausgaben") or [])))
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
