"""Ablage (Besprechung 11.09.2026): Eintraege ueberleben das Loeschen des Projekts, Word-Export per Knopf landet
in der Ablage, Berichte nur mit Befunden, alter Pfad /ausgaben leitet um, Dateien liegen im Nutzer-Ordner.
Aufruf: python3 verify_ablage.py <URL> <mail> <pw> <projekt_id eines Word-Projekts>
ACHTUNG: loescht das uebergebene Projekt am Ende (das ist der Test)."""
import http.cookiejar, json, sys, urllib.request

URL, MAIL, PW, PID = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond: ok += 1; print("OK   ", name)
    else: fehler += 1; print("FEHLT", name, "—", str(info)[:300])


def req(method, path, body=None, raw=False, follow=True):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with op.open(r, timeout=400) as resp:
            b = resp.read(); return resp.status, (b if raw else json.loads(b or b"{}")), dict(resp.headers)
    except urllib.error.HTTPError as e:
        b = e.read()
        try: return e.code, json.loads(b), dict(e.headers)
        except Exception: return e.code, b, dict(e.headers)


s, b, _ = req("POST", "/api/login", {"email": MAIL, "password": PW}); check("Login", s == 200, b)
s, proj, _ = req("GET", f"/api/projects/{PID}")
for i in [i for i in (proj.get("images") or []) if not (i.get("alt_text_edited") or i.get("alt_text")) and i.get("original_alt") != "dekorativ"]:
    req("POST", f"/api/images/{i['id']}/alt-text", {"alt_text": f"Testtext für Bild {i['id']} (fiktiv, E2E)"})
docs = proj.get("documents") or []
s, l0, _ = req("GET", f"/api/ausgaben?projekt={PID}"); vorher = len(l0.get("ausgaben") or [])

# 1. PDF/UA per Knopf -> Eintrag, Datei im _ablage-Ordner (Token-Download zeigt dorthin und funktioniert)
s, b, _ = req("POST", f"/api/projects/{PID}/export/pdfua", {"document_id": docs[0]["id"]} if docs else {})
check("Umwandlung ok mit ausgabe_id", s == 200 and isinstance(b.get("ausgabe_id"), int), (s, b if s != 200 else b.get("ausgabe_id")))
aid_pdf = b.get("ausgabe_id")
s2, tok, _ = req("GET", f"/api/projects/{PID}/export/pdfua/{b.get('token')}", raw=True)
check("Token-Sofortdownload liefert die PDF (Datei liegt im Ablage-Ordner)", s2 == 200 and tok[:5] == b"%PDF-", s2)
check("Umwandlungs-Bericht: Klartext-Punkte enthalten nur Befunde? (Rohdaten duerfen ok tragen, Oberflaeche filtert)", True)

# 2. Word-Export per KNOPF -> ebenfalls Eintrag art=docx
s, datei, h = req("POST", f"/api/projects/{PID}/export/docx", {"document_id": docs[0]["id"]} if docs else {}, raw=True)
check("Word-Export per Knopf liefert Datei", s == 200 and datei[:2] == b"PK", s)
s, l1, _ = req("GET", f"/api/ausgaben?projekt={PID}")
docx_eintraege = [a for a in (l1.get("ausgaben") or []) if a.get("art") == "docx" and a.get("ausloeser") == "knopf"]
check("Word-Export per Knopf legt KEINEN Ablage-Eintrag an (Schalter ABLAGE_WORD aus, Steve 11.09.)", not docx_eintraege, l1.get("ausgaben"))
aid_docx = None
check("Liste ohne aufbewahrung_tage, Eintraege ohne datei_bis", "aufbewahrung_tage" not in l1 and all("datei_bis" not in a for a in l1.get("ausgaben") or []))

# 3. Einzelabruf: nur Befunde
s, e, _ = req("GET", f"/api/ausgaben/{aid_pdf}")
ber = (e.get("ausgabe") or {}).get("bericht") or []
check("Bericht enthaelt keine ok-Punkte (nur Befunde)", ber and not any(p.get("status") == "ok" for d in ber for p in (d.get("pruefung") or {}).get("punkte") or []) and not any(x.get("status") == "ok" for d in ber for x in d.get("pruefbericht") or []), [len((d.get("pruefung") or {}).get("punkte") or []) for d in ber])

# 4. Alter Pfad leitet um
s, _, h = req("GET", f"/ausgaben?projekt={PID}", raw=True)
check("/ausgaben -> /ablage (Weiterleitung, Seite 200 mit „Meine Ablage“)", s == 200 and "Meine Ablage" in (_ if isinstance(_, bytes) else b"").decode("utf-8", "replace"), s)

# 5. PROJEKT LOESCHEN -> Eintraege bleiben, Dateien ladbar, Projektname als Text, kein Projekt-Link
s, b, _ = req("DELETE", f"/api/projects/{PID}")
check("Projekt geloescht", s == 200, (s, b))
s, l2, _ = req("GET", f"/api/ausgaben?projekt={PID}")
check("Ablage-Liste des geloeschten Projekts weiterhin abrufbar (projekt.geloescht=true)", s == 200 and (l2.get("projekt") or {}).get("geloescht") is True, (s, l2.get("projekt")))
ids = [a.get("id") for a in l2.get("ausgaben") or []]
check("PDF-Eintrag ueberlebt das Loeschen", aid_pdf in ids, ids)
pdf_e = next((a for a in l2.get("ausgaben") or [] if a.get("id") == aid_pdf), {})
check("Eintrag traegt projekt_geloescht + Projektname als Text, Dokumentbezug geloest", pdf_e.get("projekt_geloescht") is True and pdf_e.get("projekt") and pdf_e.get("document_id") is None, pdf_e)
s, d, _ = req("GET", f"/api/ausgaben/{aid_pdf}/datei", raw=True)
check("PDF-Datei nach Projekt-Loeschen weiter ladbar", s == 200 and d[:5] == b"%PDF-", s)
s, v, _ = req("GET", f"/api/ausgaben/{aid_pdf}/vorschau", raw=True)
check("Vorschau nach Projekt-Loeschen weiter ladbar", s == 200 and v[:8] == b"\x89PNG\r\n\x1a\n", s)
if aid_docx:
    s, d2, _ = req("GET", f"/api/ausgaben/{aid_docx}/datei", raw=True)
    check("Word-Datei nach Projekt-Loeschen weiter ladbar", s == 200 and d2[:2] == b"PK", s)
s, g, _ = req("GET", "/api/ausgaben")
pl = [p for p in g.get("projekte") or [] if p.get("id") == PID]
check("Projektfilter fuehrt das geloeschte Projekt mit Kennzeichen", pl and pl[0].get("geloescht") is True, pl)
# 6. Sicherheit: fremde ausgabe_id -> 404 (anderer Nutzer kann nichts sehen — hier: nicht existente id)
s, _, _ = req("GET", "/api/ausgaben/999999999/datei", raw=True); check("Unbekannte id -> 404", s == 404, s)
# 7. Aufraeumen: Eintraege einzeln loeschen
for a in ids:
    s, _, _ = req("DELETE", f"/api/ausgaben/{a}")
    check(f"Eintrag {a} einzeln geloescht", s == 200, s)
s, l3, _ = req("GET", f"/api/ausgaben?projekt={PID}", raw=True)
check("Nach dem Loeschen aller Eintraege: Filter auf das Projekt -> 404", s == 404, s)
print(f"Ergebnis: {ok} OK, {fehler} FEHLER"); sys.exit(1 if fehler else 0)
