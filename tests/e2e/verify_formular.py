"""End-to-End-Test Quickinfo-Werkzeug gegen Staging (27.08.2026).
Aufruf: python3 verify_formular.py <basis-url> <email> <passwort> <fixture.pdf> [--behalten]
Nur Standardbibliothek. Alle Daten fiktiv. Legt ein Projekt an und loescht es
am Ende (ausser --behalten: dann bleibt es fuer den Klicktest / Hoertest).
Muster: verify_docx.py.
"""
import http.cookiejar
import io
import json
import sys
import time
import urllib.request
import uuid

import os
# Zugangsdaten: Argumente ODER Umgebung (INKLUDOCS_E2E_URL/_MAIL/_PW, Fixture aus dem Repo) — keine Geheimnisse im Repo.
if len(sys.argv) >= 5:
    BASE, MAIL, PW, FIX = sys.argv[1:5]
else:
    BASE = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
    MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
    FIX = os.environ.get("INKLUDOCS_E2E_FIXTURE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "testformular_inkludocs.pdf"))
    if not MAIL or not PW:
        sys.exit("Zugangsdaten fehlen: INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW setzen (oder 4 Argumente uebergeben)")
BEHALTEN = "--behalten" in sys.argv
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1
        print("OK   ", name)
    else:
        fehler += 1
        print("FEHLT", name, info)


def req(method, path, data=None, files=None, raw=False):
    url = BASE + path
    headers = {}
    body = None
    if files:
        boundary = "----inkludocs" + uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in (data or {}).items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fn, content, ct) in files.items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: {ct}\r\n\r\n".encode())
            buf.write(content)
            buf.write(b"\r\n")
        buf.write(f"--{boundary}--\r\n".encode())
        body = buf.getvalue()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        resp = opener.open(r, timeout=120)
        content = resp.read()
        return resp.status, (content if raw else (json.loads(content) if content else {})), dict(resp.headers)
    except urllib.error.HTTPError as e:
        content = e.read()
        try:
            return e.code, json.loads(content), dict(e.headers)
        except Exception:
            return e.code, {"raw": content[:200]}, dict(e.headers)


def stammdaten_aufraeumen():
    """Nur die Testeintraege dieses Skripts entfernen (Beschriftungen/Feldnamen des Fixtures)."""
    s, b, _ = req("GET", "/api/stammdaten")
    for e in b.get("stammdaten", []):
        if e["beschriftung"] in ("Vorname", "Nachname", "Geburtsdatum (TT.MM.JJJJ) *") or e["feld_name"] in ("anrede",):
            req("DELETE", f"/api/stammdaten/{e['id']}")


# A. Login + Werkzeug
s, b, _ = req("POST", "/api/login", {"email": MAIL, "password": PW})
check("Login", s == 200, b)
stammdaten_aufraeumen()   # Reste frueherer Laeufe wuerden sonst schon beim Upload angewendet
s, b, _ = req("GET", "/api/tools")
werkzeug = next((t for t in b.get("tools", []) if t["key"] == "formular"), None)
check("Werkzeug formular verfuegbar (Beta)", werkzeug and werkzeug["is_available"], werkzeug)
s, b, _ = req("POST", "/api/projects", {"name": "E2E Quickinfo-Test (fiktiv)", "tool": "formular"})
check("Projekt anlegen", s == 200 and b.get("project_type") == "pdfform", b)
pid = b.get("project_id")

# B. Negativfaelle
leer = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\ntrailer<</Root 1 0 R>>"
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("ohne_felder.pdf", leer, "application/pdf")})
check("PDF ohne Felder -> 400 mit Meldung", s == 400 and "Formularfeld" in str(b.get("detail", "")), (s, b))
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("bild.png", b"\x89PNG\r\n\x1a\n" + b"0" * 100, "image/png")})
check("Bild in Formular-Projekt -> abgelehnt", s in (400, 422), (s, b))
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("doc.docx", b"PK\x03\x04" + b"0" * 100, "application/octet-stream")})
check("Word in Formular-Projekt -> 400 mit Meldung", s == 400 and "PDF-Formulare" in str(b.get("detail", "")), (s, b))
s, b, _ = req("POST", f"/api/projects/{pid}/export/formular", {})
check("Export ohne Dokument -> 400 (keine Abrechnung)", s == 400, s)

# C. Upload + Extraktion
fix = open(FIX, "rb").read()
s, b, _ = req("POST", "/api/upload", {"project_id": pid}, files={"file": ("testformular_inkludocs.pdf", fix, "application/pdf")})
check("Upload angenommen (extracting)", s == 200 and b.get("status") == "extracting" and b.get("project_type") == "pdfform", b)
doc_id = b.get("document_id")
for _ in range(60):
    time.sleep(1)
    s, p, _ = req("GET", f"/api/projects/{pid}")
    if p.get("project", {}).get("status") != "extracting":
        break
check("Extraktion beendet", p.get("project", {}).get("status") == "extracted", p.get("project", {}).get("status"))
s, d, _ = req("GET", f"/api/projects/{pid}/felder")
felder = d.get("felder", [])
by = {f["anker"]: f for f in felder}
check("12 Felder gelesen", len(felder) == 12, len(felder))
check("Dokument mit Zaehlern", d["documents"] and d["documents"][0]["felder_gesamt"] == 12 and d["documents"][0]["felder_offen"] == 11, d.get("documents"))
check("vorhandene Quickinfo uebernommen (email)", by.get("email", {}).get("quickinfo") == "E-Mail-Adresse für Kontoauszüge" and by["email"]["quelle"] == "pdf")
check("Beschriftung/Abschnitt", by["vorname"]["beschriftung"] == "Vorname" and by["vorname"]["gruppe"] == "Angaben zum Kontoinhaber")
check("Feldwert nirgends", all("K-0000" not in json.dumps(f, ensure_ascii=False) for f in felder))
check("Pfade nicht nach aussen", all("path" not in k for f in felder for k in f))
check("Seitentext nur einmal je Seite", sum(1 for f in felder if f["page_text"]) == 2)
s, bild, h = req("GET", f"/api/felder/{by['vorname']['id']}/ausschnitt", raw=True)
check("Ausschnitt-PNG", s == 200 and bild[:4] == b"\x89PNG", s)
s, bild, h = req("GET", f"/api/felder/{by['vorname']['id']}/page-view", raw=True)
check("Seitenansicht-PNG", s == 200 and bild[:4] == b"\x89PNG", s)

# D. Quickinfo speichern / Original / Stammdaten
fid = by["vorname"]["id"]
s, b, _ = req("PATCH", f"/api/felder/{fid}", {"quickinfo": "Vorname des Kontoinhabers\n<script>x</script>"})
check("Speichern (Steuerzeichen raus, Text bleibt)", s == 200 and b["status"] == "beschrieben" and b["quelle"] == "hand" and "\n" not in b["quickinfo"], b)
s, b, _ = req("PATCH", f"/api/felder/{fid}", {"quickinfo": ""})
check("Leeren -> offen", s == 200 and b["status"] == "offen", b)
s, b, _ = req("PATCH", f"/api/felder/{fid}", {"quickinfo": "Vorname des Kontoinhabers"})
s, b, _ = req("POST", f"/api/felder/{by['email']['id']}/original")
check("Zurueck auf Original", s == 200 and b["quickinfo"] == "E-Mail-Adresse für Kontoauszüge" and b["quelle"] == "pdf", b)
s, b, _ = req("POST", f"/api/felder/{by['nachname']['id']}/stammdaten")
check("Offenes Feld -> nicht in Stammdaten (400)", s == 400, s)
s, b, _ = req("POST", f"/api/felder/{fid}/stammdaten")
check("In Stammdaten uebernehmen", s == 200 and b.get("stammdaten_id"), b)
sid = b.get("stammdaten_id")
s, b, _ = req("GET", "/api/stammdaten")
eintrag = next((e for e in b.get("stammdaten", []) if e["id"] == sid), None)
check("Stammdaten-Eintrag (Beschriftung Vorname, text)", eintrag and eintrag["beschriftung"] == "Vorname" and eintrag["feld_art"] == "text" and eintrag["herkunft"] == "feld", eintrag)
# Treffer fuer vorname_2 (gleiche Beschriftung, anderer Name)
s, d, _ = req("GET", f"/api/projects/{pid}/felder")
tr = d.get("stammdaten_treffer", {}).get(str(by["vorname_2"]["id"])) or d.get("stammdaten_treffer", {}).get(by["vorname_2"]["id"])
check("Treffer ueber Beschriftung fuer vorname_2", tr and tr[0]["treffer_art"] == "beschriftung", d.get("stammdaten_treffer"))
s, b, _ = req("POST", f"/api/felder/{by['vorname_2']['id']}/stammdaten-uebernehmen", {"stammdaten_id": sid})
check("Aus Stammdaten uebernehmen", s == 200 and b["quelle"] == "stammdaten" and b["quickinfo"] == "Vorname des Kontoinhabers", b)
s, b, _ = req("POST", f"/api/felder/{by['vorname_2']['id']}/stammdaten-uebernehmen", {"stammdaten_id": 999999999})
check("Fremder/unbekannter Stammdaten-Eintrag -> 404", s == 404, s)
# Stammdaten CRUD + Import/Export
s, b, _ = req("POST", "/api/stammdaten", {"beschriftung": "Nachname", "feld_art": "text", "quickinfo": "Nachname des Kontoinhabers", "sprache": "de"})
check("Stammdaten anlegen", s == 200 and b.get("id"), b)
sid2 = b.get("id")
s, b, _ = req("POST", "/api/stammdaten", {"beschriftung": "Nachname", "feld_art": "text", "quickinfo": "Familienname des Kontoinhabers"})
check("Gleicher Schluessel -> aktualisiert, keine Dublette", s == 200 and b.get("id") == sid2, b)
s, b, _ = req("POST", "/api/stammdaten", {"beschriftung": "", "feld_art": "text", "quickinfo": "x"})
check("Ohne Beschriftung und Feldname -> 400", s == 400, s)
s, b, _ = req("POST", "/api/stammdaten", {"beschriftung": "A", "feld_art": "kaputt", "quickinfo": "x"})
check("Unbekannte Feldart -> 400", s == 400, s)
s, b, _ = req("POST", f"/api/projects/{pid}/stammdaten-anwenden", {"nur_offene": True})
check("Stammdaten anwenden (nachname + nachname_2)", s == 200 and b.get("uebernommen") == 2, b)
s, b, _ = req("POST", f"/api/projects/{pid}/stammdaten-anwenden", {"nur_offene": False})
s2, d2, _ = req("GET", f"/api/projects/{pid}/felder")
by2 = {f["anker"]: f for f in d2.get("felder", [])}
check("nur_offene=false ersetzt keine Hand-Texte und keine PDF-Originale", by2["vorname"]["quickinfo"] == "Vorname des Kontoinhabers" and by2["vorname"]["quelle"] == "hand" and by2["email"]["quelle"] == "pdf", (by2["vorname"]["quelle"], by2["email"]["quelle"]))
check("Projekt ohne Serverpfade nach aussen", "original_path" not in d2.get("project", {}), list(d2.get("project", {}).keys()))
s, b, _ = req("PATCH", f"/api/felder/{fid}", None)
check("Ungueltiger JSON-Koerper -> 400", s == 400, s)
s, csvb, h = req("GET", "/api/stammdaten/export.csv", raw=True)
check("Stammdaten-CSV", s == 200 and b"Beschriftung;Feldart;Feldname;Quickinfo;Sprache" in csvb and "Familienname".encode() in csvb, s)
imp = "Beschriftung;Feldart;Feldname;Quickinfo;Sprache\nGeburtsdatum (TT.MM.JJJJ) *;text;;Geburtsdatum, Format Tag Punkt Monat Punkt Jahr;de\n;dropdown;anrede;Anrede auswählen;de\n"
s, b, _ = req("POST", "/api/stammdaten/import", files={"file": ("stammdaten.csv", imp.encode("utf-8"), "text/csv")})
check("Stammdaten-Import (2 Zeilen)", s == 200 and b.get("uebernommen") == 2, b)
s, b, _ = req("POST", f"/api/projects/{pid}/stammdaten-anwenden", {"nur_offene": True})
check("Anwenden nach Import (geburtsdatum + anrede)", s == 200 and b.get("uebernommen") == 2, b)
s, b, _ = req("PATCH", f"/api/stammdaten/{sid2}", {"quickinfo": "Nachname, wie im Ausweis"})
check("Stammdaten aendern", s == 200, b)
s, b, _ = req("PATCH", f"/api/stammdaten/{sid2}", {"beschriftung": "Vorname"})
check("Dublette per PATCH -> 409", s == 409, s)
s, csvb, h = req("GET", "/api/stammdaten/export.csv", raw=True)
cd_header = {k.lower(): v for k, v in h.items()}.get("content-disposition", "")
check("CSV-Export: Dateiname nach RFC 6266", s == 200 and "filename*=UTF-8" in cd_header, cd_header)
s, b, _ = req("DELETE", f"/api/stammdaten/{sid2}")
check("Stammdaten loeschen", s == 200, b)
s, b, _ = req("DELETE", f"/api/stammdaten/{sid2}")
check("Nochmal loeschen -> 404", s == 404, s)

# E. Export
s, pdf, h = req("POST", f"/api/projects/{pid}/export/formular", {"document_id": doc_id}, raw=True)
hl = {k.lower(): v for k, v in h.items()}
check("PDF-Export", s == 200 and pdf[:5] == b"%PDF-" and hl.get("x-export-method") == "formular", (s, hl))
check("Export-Staffel: 12 Felder = 5 + 2 = 7 Credits (Header)", hl.get("x-export-credits") == "7", hl.get("x-export-credits"))
s, b, _ = req("POST", f"/api/projects/{pid}/export/preis", {})
check("Preis-Endpunkt: anzahl 12, preis 7, Einheit felder", s == 200 and b.get("anzahl") == 12 and b.get("preis") == 7 and b.get("einheit") == "felder" and "erlaubt" in b, b)
# 7 = vorname (Hand), vorname_2 (Stammdaten), email (Original), nachname + nachname_2 (anwenden), geburtsdatum + anrede (Import)
check("Export: 7 Quickinfos geschrieben, 12 Felder, 5 offen", hl.get("x-export-tagged") == "7" and hl.get("x-export-total") == "12" and hl.get("x-export-open") == "5", hl)
if hl.get("x-export-writer") == "pymupdf":
    check("Export: Original ist Praefix (PyMuPDF-Weg)", pdf[:len(fix)] == fix)
else:
    check("Export ueber PDFix (Lizenz): kein Trial-Vermerk", hl.get("x-export-writer") == "pdfix" and b"Trial version" not in pdf, hl.get("x-export-writer"))
s, csvb, h = req("POST", f"/api/projects/{pid}/export/formular_csv", {}, raw=True)
check("CSV-Export (ohne Werte, mit Quickinfos)", s == 200 and b"Vorname des Kontoinhabers" in csvb and b"K-0000" not in csvb, s)
s, b, _ = req("POST", f"/api/projects/{pid}/export/docx", {})
check("Word-Export auf Formular-Projekt -> 400", s == 400, s)

# G. Stufe 2: KI-Vorschlaege (echte Bedrock-Aufrufe, ~2 Seiten = 2 Credits)
# KI-Fach (28.08.2026): nachname traegt einen Stammdaten-Text -> Generieren ersetzt ihn NICHT, Vorschlag geht ins Fach
s, b, _ = req("POST", f"/api/felder/{by['nachname']['id']}/generieren")
check("Einzelfeld generieren (Bedrock) bei Fremdtext: Text bleibt, KI-Vorschlag im Fach", s == 200 and b.get("uebernommen") is False and b.get("quelle") == "stammdaten" and len(b.get("ki_vorschlag", "")) > 3 and b.get("sicherheit") in ("hoch", "mittel", "niedrig"), b)
print("      nachname (Fach) ->", repr(b.get("ki_vorschlag")), "|", b.get("sicherheit"), "| Beleg:", repr(b.get("beleg")))
check("Einzelfeld: Beleg ist Text der Seite", "nachname" in (b.get("beleg") or "").lower() or b.get("sicherheit") != "hoch", b.get("beleg"))
s, d1, _ = req("GET", f"/api/projects/{pid}/felder")
n1 = next(f for f in d1["felder"] if f["anker"] == "nachname")
check("KI-Fach in /felder sichtbar, Feldtext unveraendert", n1.get("quickinfo_ki") == b.get("ki_vorschlag") and n1["quelle"] == "stammdaten", (n1.get("quickinfo_ki"), n1["quelle"]))
s, b, _ = req("POST", f"/api/felder/{by['nachname']['id']}/ki-vorschlag")
check("KI-Vorschlag uebernehmen -> quelle ki", s == 200 and b.get("quelle") == "ki" and b.get("quickinfo") == n1.get("quickinfo_ki"), b)
s, b, _ = req("POST", f"/api/felder/{by['nachname']['id']}/generieren")   # jetzt quelle ki -> wird ersetzt
check("Einzelfeld generieren bei KI-Text: ersetzt (uebernommen)", s == 200 and b.get("uebernommen") is True and b.get("quelle") == "ki", b)
s, b, _ = req("POST", f"/api/felder/{by['anschrift']['id']}/ki-vorschlag")
check("KI-Vorschlag uebernehmen ohne Fach -> 400", s == 400, s)
# Alle offenen generieren: vorher nachname_2 leeren, vorname bleibt Hand (darf nicht ueberschrieben werden)
req("PATCH", f"/api/felder/{by['nachname_2']['id']}", {"quickinfo": ""})
req("PATCH", f"/api/felder/{by['geburtsdatum']['id']}", {"quickinfo": ""})
s, b, _ = req("POST", f"/api/projects/{pid}/quickinfos/generieren", {})
check("Alle generieren gestartet", s == 200 and b.get("gestartet") is True and b.get("offen") >= 2, b)
s, b, _ = req("POST", f"/api/projects/{pid}/quickinfos/generieren", {})
check("Zweiter Start waehrend Lauf -> 409", s == 409, s)
for _ in range(90):
    time.sleep(2)
    s, d, _ = req("GET", f"/api/projects/{pid}/felder")
    g = d.get("generierung") or {}
    if d.get("project", {}).get("status") != "processing" and not g.get("laeuft", False):
        break
by3 = {f["anker"]: f for f in d.get("felder", [])}
check("Generierung beendet, Status extracted", d.get("project", {}).get("status") == "extracted", (d.get("project", {}).get("status"), g))
check("Offene Felder jetzt KI (nachname_2, geburtsdatum)", by3["nachname_2"]["quelle"] == "ki" and by3["geburtsdatum"]["quelle"] == "ki" and by3["nachname_2"]["quickinfo"], (by3["nachname_2"]["quelle"], by3["geburtsdatum"]["quelle"]))
check("Hand-Text nicht ueberschrieben (vorname)", by3["vorname"]["quelle"] == "hand" and by3["vorname"]["quickinfo"] == "Vorname des Kontoinhabers")
check("PDF-Original nicht ueberschrieben (email)", by3["email"]["quelle"] == "pdf")
check("Sicherheit/Beleg gespeichert", by3["nachname_2"]["sicherheit"] in ("hoch", "mittel", "niedrig") and isinstance(by3["nachname_2"]["ki_hinweise"], list), by3["nachname_2"].get("sicherheit"))
print("      geburtsdatum ->", repr(by3["geburtsdatum"]["quickinfo"]), "|", by3["geburtsdatum"]["sicherheit"], "| Beleg:", repr(by3["geburtsdatum"]["beleg"]))
print("      Lauf:", g)
check("Generierungs-Fehlerliste leer", not (g.get("fehler") or []), g.get("fehler"))
check("Keine Feldwerte in KI-Texten", all("K-0000" not in (f.get("quickinfo") or "") + (f.get("beleg") or "") for f in d.get("felder", [])))

# Alle neu generieren (modus ki_neu, 28.08.2026): nur KI-Felder, Hand/PDF bleiben
ki_vorher = [f for f in d.get("felder", []) if f["quelle"] == "ki"]
s, b, _ = req("POST", f"/api/projects/{pid}/quickinfos/generieren", {"modus": "ki_neu"})
check("Alle neu generieren: startet fuer genau die KI-Felder", s == 200 and b.get("gestartet") is True and b.get("modus") == "ki_neu" and b.get("offen") == len(ki_vorher), (b, len(ki_vorher)))
for _ in range(90):
    time.sleep(2)
    s, d, _ = req("GET", f"/api/projects/{pid}/felder")
    g = d.get("generierung") or {}
    if d.get("project", {}).get("status") != "processing" and not g.get("laeuft", False):
        break
by5 = {f["anker"]: f for f in d.get("felder", [])}
check("Alle neu generieren: Hand- und PDF-Texte unberuehrt", by5["vorname"]["quelle"] == "hand" and by5["email"]["quelle"] == "pdf", (by5["vorname"]["quelle"], by5["email"]["quelle"]))
check("Alle neu generieren: KI-Felder neu (felder_neu = Anzahl KI-Felder)", g.get("felder_neu") == len(ki_vorher), (g, len(ki_vorher)))

# Fuer den Klicktest (--behalten) wieder zwei Felder oeffnen (Alle-generieren-Knopf, Filter, Hoerprobe)
req("PATCH", f"/api/felder/{by['nachname_2']['id']}", {"quickinfo": ""})
req("PATCH", f"/api/felder/{by['anschrift']['id']}", {"quickinfo": ""})

# F. Fremdzugriff: Feld-Endpunkte ohne Login
cj.clear()
s, b, _ = req("GET", f"/api/projects/{pid}/felder")
check("Ohne Login -> 401", s == 401, s)
s, b, _ = req("PATCH", f"/api/felder/{fid}", {"quickinfo": "hack"})
check("PATCH ohne Login -> 401", s == 401, s)
req("POST", "/api/login", {"email": MAIL, "password": PW})

# G2. InkluAgent im Formular-Projekt (28.08.2026): echte Bedrock-Turns (Kontext + ein Werkzeug)
s, b, _ = req("GET", f"/api/projects/{pid}/chat/history")
check("Chat-Verlauf abrufbar (leer)", s == 200 and b.get("messages") == [], b)
s, b, _ = req("POST", f"/api/projects/{pid}/chat", {"message": "Wie viele Felder hat dieses Formular und wie viele davon haben noch keine Quickinfo? Nur die Zahlen bitte."})
antwort = (b.get("reply") or "")
tools_genutzt = [a.get("tool") for a in (b.get("actions") or []) if a.get("tool")]
# Die Zahlen stehen schon in der Projekt-Zusammenfassung (agent_loop._formular_summary) — der Agent darf
# ohne Werkzeug antworten; wichtig ist nur: richtige Zahl, kein Bild-Werkzeug.
check("Chat: Antwort nennt 12 Felder", s == 200 and "12" in antwort, (s, tools_genutzt, antwort[:160]))
check("Chat: keine Bild-Werkzeuge im Formular-Projekt", not any(t.startswith(("list_project_images", "view_image", "generate_alt", "update_alt")) for t in tools_genutzt), tools_genutzt)
print("      Chat:", antwort[:200].replace("\n", " "))
s, b, _ = req("POST", f"/api/projects/{pid}/chat", {"message": "Setze bei Feld 2 die Quickinfo auf genau diesen Text: Nachname des Kontoinhabers. Ja, bitte direkt speichern, das ist meine Zustimmung."})
acts = b.get("actions") or []
refresh = [a for a in acts if a.get("type") == "refresh_feld"]
check("Chat: update_quickinfo gespeichert + refresh_feld", s == 200 and any(a.get("tool") == "update_quickinfo" and a.get("ok") for a in acts) and refresh and refresh[-1].get("quickinfo") == "Nachname des Kontoinhabers", (s, [(a.get("tool"), a.get("ok"), a.get("error", "")[:80]) for a in acts if a.get("tool")]))
print("      Chat:", (b.get("reply") or "")[:200].replace("\n", " "))
s, d3, _ = req("GET", f"/api/projects/{pid}/felder")
by4 = {f["anker"]: f for f in d3.get("felder", [])}
check("Chat: Werkzeugliste in der Antwort (update_quickinfo enthalten)", isinstance(b.get("werkzeuge"), list) and "update_quickinfo" in b.get("werkzeuge", []), b.get("werkzeuge"))
s, hb, _ = req("GET", f"/api/projects/{pid}/chat/history")
letzte = [m for m in hb.get("messages", []) if m["role"] == "assistant"]
check("Chat-Verlauf speichert Werkzeuge je Antwort", letzte and isinstance(letzte[-1].get("werkzeuge"), list) and "update_quickinfo" in letzte[-1]["werkzeuge"], letzte[-1].get("werkzeuge") if letzte else None)
check("Chat-Speichern in DB (quelle chat, Sicherheit gesetzt)", by4["nachname"]["quickinfo"] == "Nachname des Kontoinhabers" and by4["nachname"]["quelle"] == "chat" and by4["nachname"]["sicherheit"] in ("hoch", "mittel"), (by4["nachname"]["quelle"], by4["nachname"]["sicherheit"], by4["nachname"].get("beleg")))
req("DELETE", f"/api/projects/{pid}/chat")

# H. Gast-Ansicht (28.08.2026): Freigabe anlegen, E-Mail-Gate, Felder lesen, Urteil, Abschluss
GAST = "gast-formular@beispiel.invalid"
s, b, _ = req("POST", f"/api/projects/{pid}/share", {"guest_email": GAST, "guest_name": "Testgast", "notify": False, "role": "kunde"})
check("Freigabe fuer Formular-Projekt anlegen (ohne Mail)", s == 200 and b.get("token") and b.get("sent") is False, b)
token = b.get("token", "")
print("Gast-Token:", token)
owner_cj = http.cookiejar.CookieJar()
for c in cj: owner_cj.set_cookie(c)
cj.clear()   # ab hier: Gast ohne Login
s, html_, _ = req("GET", f"/freigabe/{token}", raw=True)
check("Gast-Seite liefert app.html mit GUEST_TOOL formular", s == 200 and b'window.GUEST_TOOL = "formular"' in html_, s)
s, b, _ = req("GET", f"/api/freigabe/{token}/felder")
check("Gast ohne Bestaetigung -> 401", s == 401, s)
s, b, _ = req("POST", f"/api/freigabe/{token}/confirm", {"email": "falsch@beispiel.invalid"})
check("Falsche E-Mail am Gate -> 403", s == 403, s)
s, b, _ = req("POST", f"/api/freigabe/{token}/confirm", {"email": GAST})
check("Richtige E-Mail am Gate -> 200", s == 200, b)
s, b, _ = req("GET", f"/api/freigabe/{token}")
check("Gast-Projektkopf: formular=True, keine Serverpfade", s == 200 and b.get("formular") is True and "original_path" not in b.get("project", {}) and b.get("images") == [], b)
s, gd, _ = req("GET", f"/api/freigabe/{token}/felder")
gfelder = gd.get("felder", [])
check("Gast: 12 Felder, role kunde, in_review", s == 200 and len(gfelder) == 12 and gd.get("role") == "kunde" and gd.get("in_review") is True and gd.get("guest") is True, (s, len(gfelder), gd.get("role")))
check("Gast: keine Serverpfade, keine Feldwerte", all("path" not in k for f in gfelder for k in f) and all("K-0000" not in json.dumps(f, ensure_ascii=False) for f in gfelder))
check("Gast: keine Stammdaten-Treffer im Paket", "stammdaten_treffer" not in gd and "stammdaten_anzahl" not in gd)
gby = {f["anker"]: f for f in gfelder}
gfid = gby["vorname"]["id"]
s, bild, _ = req("GET", f"/api/freigabe/{token}/felder/{gfid}/ausschnitt", raw=True)
check("Gast: Ausschnitt-PNG", s == 200 and bild[:4] == b"\x89PNG", s)
s, bild, _ = req("GET", f"/api/freigabe/{token}/felder/{gfid}/page-view", raw=True)
check("Gast: Seitenansicht-PNG", s == 200 and bild[:4] == b"\x89PNG", s)
s, b, _ = req("GET", f"/api/freigabe/{token}/felder/{gfid + 100000}/ausschnitt", raw=True)
check("Gast: fremde Feld-ID -> 404", s == 404, s)
s, b, _ = req("POST", f"/api/felder/{gfid}/generieren")
check("Gast: KI-Endpunkt ohne Login -> 401", s == 401, s)
s, b, _ = req("GET", "/api/stammdaten")
check("Gast: Stammdaten ohne Login -> 401", s == 401, s)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gfid}/quickinfo", {"quickinfo": "Vorname der Kontoinhaberin\noder des Kontoinhabers"})
check("Gast: Quickinfo von Hand -> quelle gast, auto_status in_bearbeitung, Umbruch raus", s == 200 and b.get("quelle") == "gast" and b.get("auto_status") == "in_bearbeitung" and "\n" not in b.get("quickinfo", ""), b)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gfid}/review", {"status": "unsinn"})
check("Gast: ungueltiger Status -> 400", s == 400, s)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gfid}/review", {"status": "ruecksprache"})
check("Gast (Herausgeber): Ruecksprache -> 403", s == 403, s)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gfid}/review", {"status": "freigegeben", "comment": "Bitte so lassen."})
check("Gast: Freigeben mit Anmerkung", s == 200 and b.get("review_status") == "freigegeben" and b.get("comment") == "Bitte so lassen.", b)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gby['nachname']['id']}/review", {"status": "zu_ueberarbeiten", "comment": ""})
check("Gast: Aenderung wuenschen ohne Anmerkung", s == 200 and b.get("review_status") == "zu_ueberarbeiten", b)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gfid}/quickinfo", {"quickinfo": "Vorname des Kontoinhabers"})
check("Gast: erneutes Bearbeiten ueberschreibt gesetztes Urteil NICHT", s == 200 and b.get("auto_status") is None, b)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gfid}/review", {"status": "freigegeben"})
s, gdx, _ = req("GET", f"/api/freigabe/{token}/felder")
gx = {f["anker"]: f for f in gdx.get("felder", [])}
check("Gast: Urteil ohne comment-Feld laesst Anmerkung stehen", s == 200 and gx["vorname"]["review_note"] == "Bitte so lassen.", gx["vorname"].get("review_note"))
s, gd2, _ = req("GET", f"/api/freigabe/{token}/felder")
g2 = {f["anker"]: f for f in gd2.get("felder", [])}
check("Gast: Urteil + Anmerkung im Paket", g2["vorname"]["reviews"].get("kunde", {}).get("status") == "freigegeben" and g2["vorname"]["review_note"] == "Bitte so lassen." and g2["vorname"]["review_status"] == "freigegeben", g2["vorname"].get("reviews"))
s, b, _ = req("POST", f"/api/freigabe/{token}/complete", {"message": "E2E: Pruefung abgeschlossen."})
check("Gast: Pruefung abschliessen (1 freigegeben, 1 zu ueberarbeiten, 10 offen)", s == 200 and b.get("freigegeben") == 1 and b.get("zu_ueberarbeiten") == 1 and b.get("offen") == 10, b)
s, b, _ = req("POST", f"/api/freigabe/{token}/felder/{gfid}/review", {"status": "freigegeben", "comment": "Bitte so lassen."})
check("Gast: Wieder-Einstieg nach Abschluss moeglich", s == 200, s)
# Besitzer sieht das Urteil
cj.clear()
for c in owner_cj: cj.set_cookie(c)
s, od, _ = req("GET", f"/api/projects/{pid}/felder")
o = {f["anker"]: f for f in od.get("felder", [])}
check("Besitzer: in_review + share_roles kunde", od.get("in_review") is True and od.get("share_roles") == ["kunde"], (od.get("in_review"), od.get("share_roles")))
check("Besitzer: Urteil, Anmerkung und quelle gast am Feld", o["vorname"]["review_status"] == "freigegeben" and o["vorname"]["review_note"] == "Bitte so lassen." and o["vorname"]["quelle"] == "gast" and o["vorname"]["reviews"]["kunde"]["status"] == "freigegeben", o["vorname"].get("reviews"))
s, b, _ = req("GET", "/api/review-overview")
mine = [p for p in b.get("projects", []) if p.get("id") == pid]
check("Geteilte Projekte: Formular-Projekt mit Zaehlern", len(mine) == 1 and mine[0].get("total") == 12 and mine[0].get("freigegeben") == 1 and mine[0].get("zu_ueberarbeiten") == 1 and "Testgast" in (mine[0].get("guests") or ""), mine)
s, b, _ = req("GET", f"/api/projects/{pid}/shares")
check("Besitzer: Freigabe gelistet (active nach Wieder-Einstieg)", s == 200 and len(b.get("shares", [])) == 1 and b["shares"][0]["status"] == "active", b)

print(f"\nErgebnis: {ok} OK, {fehler} FEHLT (Projekt {pid})")
if not BEHALTEN:
    s, b, _ = req("DELETE", f"/api/projects/{pid}")
    print("Testprojekt geloescht:", s)
    stammdaten_aufraeumen()
sys.exit(1 if fehler else 0)
