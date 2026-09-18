#!/usr/bin/env python3
"""E2E-Probe der Public API v1 Dokumente gegen Staging (17.09.2026). Nur ueber HTTPS wie ein Partner.
Aufruf: INKLUDOCS_API_KEY=... python3 verify_api_v1_dokumente.py [fixtures-ordner]
Kostet Credits des Testkontos (Betreiber-Konto: unbegrenzt). Raeumt alle angelegten Dokumente wieder ab."""
import json
import os
import sys
import time

import httpx

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de").rstrip("/")
KEY = os.environ["INKLUDOCS_API_KEY"]
FIX = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/fixtures"
H = {"X-API-Key": KEY}
ok = fehler = 0
angelegt = []


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", str(info)[:300])


def warte(c, pid, ziel, sekunden=240):
    t0 = time.time()
    while time.time() - t0 < sekunden:
        r = c.get(f"/api/v1/documents/{pid}", headers=H)
        st = r.json().get("status")
        if st in ziel or st == "error":
            return r.json()
        time.sleep(3)
    return c.get(f"/api/v1/documents/{pid}", headers=H).json()


with httpx.Client(base_url=B, timeout=180) as c:
    # 1) Fehlerformat
    r = c.get("/api/v1/documents")
    check("Ohne Schluessel: 401 + error.code unauthorized", r.status_code == 401 and r.json()["error"]["code"] == "unauthorized", r.text)
    r = c.get("/api/v1/documents/999999999", headers=H)
    check("Fremdes/unbekanntes Dokument: 404 not_found", r.status_code == 404 and r.json()["error"]["code"] == "not_found", r.text)
    r = c.post("/api/v1/documents", headers={**H, "Content-Type": "text/plain"}, content=b"x")
    check("Falscher Inhaltstyp: 415", r.status_code == 415, r.text)
    r = c.post("/api/v1/documents", headers=H, files={"file": ("boese.exe", b"MZ", "application/octet-stream")})
    check("Unbekannter Dateityp: 400", r.status_code == 400 and r.json()["error"]["code"] == "bad_request", r.text)
    r = c.post("/api/v1/documents", headers=H, files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}, data={"language": "klingonisch"})
    check("Unbekannte Sprache: 400", r.status_code == 400, r.text)

    # 2) PDF-Dokument: hochladen -> ready -> generieren -> done -> items -> korrigieren -> exportieren
    with open(os.path.join(FIX, "01_lo_testdokument.pdf"), "rb") as f:
        r = c.post("/api/v1/documents", headers=H, files={"file": ("Testheft (fiktiv).pdf", f, "application/pdf")},
                   data={"name": "API-Probe PDF (fiktiv)", "language": "de"})
    check("PDF hochladen: 202/201 + id", r.status_code in (201, 202) and r.json().get("id"), r.text)
    pid = r.json()["id"]; angelegt.append(pid)
    check("Kopfzeilen: X-API-Version + Rate-Limit", r.headers.get("X-API-Version") == "1" and "X-RateLimit-Remaining-Minute" in r.headers, dict(r.headers))
    d = warte(c, pid, ("ready", "done"))
    check("PDF extrahiert: status ready, Bilder > 0", d["status"] == "ready" and d["counts"]["items"] > 0, d)
    r = c.get("/api/v1/documents", headers=H)
    check("Liste enthaelt das Dokument", r.status_code == 200 and any(x["id"] == pid for x in r.json()["documents"]), r.text[:200])
    r = c.post(f"/api/v1/documents/{pid}/generate", headers=H)
    check("Generieren gestartet: 202 + started", r.status_code == 202 and r.json().get("started"), r.text)
    r2 = c.post(f"/api/v1/documents/{pid}/generate", headers=H)
    check("Zweiter Start waehrend des Laufs: 409 conflict", r2.status_code == 409 and r2.json()["error"]["code"] == "conflict", r2.text)
    d = warte(c, pid, ("done",), 600)
    check("Lauf fertig: status done, with_text > 0", d["status"] == "done" and d["counts"]["with_text"] > 0, d.get("counts"))
    # 2b) scope/Einstellungen (18.09.2026): open = nichts offen -> nicht gestartet, keine Kosten; all + language + prompt
    r = c.post(f"/api/v1/documents/{pid}/generate", headers=H, json={"scope": "open"})
    check("scope=open ohne offene Eintraege: 200, started false, Hinweis", r.status_code == 200 and r.json().get("started") is False and "scope=all" in (r.json().get("hint") or ""), r.text[:200])
    r = c.post(f"/api/v1/documents/{pid}/generate", headers=H, json={"scope": "hexerei"})
    check("scope unbekannt: 400", r.status_code == 400, r.text[:200])
    r = c.post(f"/api/v1/documents/{pid}/generate", headers=H, json={"scope": "all", "prompt_id": 999999999})
    check("prompt_id fremd: 404, kein Start", r.status_code == 404, r.text[:200])
    r = c.get(f"/api/v1/documents/{pid}/items", headers=H, params={"status": "failed"})
    check("items?status=failed: count 0, total = alle", r.status_code == 200 and r.json()["count"] == 0 and r.json()["total"] == d["counts"]["items"], r.text[:200])
    r = c.get(f"/api/v1/documents/{pid}/items", headers=H, params={"status": "kaputt"})
    check("items?status ungueltig: 400", r.status_code == 400, r.text[:200])
    r = c.get(f"/api/v1/documents/{pid}/items", headers=H)
    items = r.json()["items"]
    check("Item traegt Feld error (null bei Erfolg)", all("error" in i and i["error"] is None for i in items), [i.get("error") for i in items][:3])
    erstes = items[0]["id"]
    r = c.post(f"/api/v1/documents/{pid}/items/{erstes}/generate", headers=H, json={"language": "en", "prompt": "Beschreibe in höchstens acht Wörtern. (API-Test, fiktiv)"})
    check("Einzel-Generate mit language+prompt: 200, Text englisch/kurz", r.status_code == 200 and r.json().get("language") == "en" and r.json().get("alt_text"), r.text[:300])
    r = c.get(f"/api/v1/documents/{pid}", headers=H)
    check("Dokument nach Einzel-Generate: language en (Einstellung gilt ab jetzt)", r.json().get("language") == "en", r.json().get("language"))
    items = c.get(f"/api/v1/documents/{pid}/items", headers=H).json()["items"]   # frischer Stand nach dem Einzel-Generate
    check("Items: Bilder mit alt_text und file_url", r.status_code == 200 and items and all(i["type"] == "image" and "alt_text" in i and i["file_url"] for i in items), r.text[:300])
    it = next((i for i in items if i["text_status"] == "mit_text"), items[0])
    vorher = it["alt_text"]
    r = c.patch(f"/api/v1/documents/{pid}/items/{it['id']}", headers=H, json={"langbeschreibung": "Lange Beschreibung (fiktiv)"})
    check("PATCH nur Langbeschreibung laesst Alt-Text stehen", r.status_code == 200 and r.json()["alt_text"] == vorher and r.json()["langbeschreibung"] == "Lange Beschreibung (fiktiv)", r.text[:300])
    r = c.patch(f"/api/v1/documents/{pid}/items/{it['id']}", headers=H, json={"alt_text": "Handtext ueber die API (fiktiv)"})
    check("PATCH Alt-Text: sichtbar + mit_text", r.status_code == 200 and r.json()["alt_text"] == "Handtext ueber die API (fiktiv)" and r.json()["text_status"] == "mit_text", r.text[:300])
    r = c.patch(f"/api/v1/documents/{pid}/items/999999999", headers=H, json={"alt_text": "x"})
    check("PATCH fremdes Item: 404", r.status_code == 404, r.text)
    r = c.get(f"/api/v1/documents/{pid}/items/{it['id']}/file", headers=H)
    check("Bilddatei abrufbar", r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"), r.headers.get("content-type"))
    r = c.post(f"/api/v1/documents/{pid}/export/xlsx", headers=H)
    check("Export xlsx: Datei + X-Export-Credits", r.status_code == 200 and "spreadsheet" in r.headers.get("content-type", "") and r.headers.get("X-Export-Credits"), (r.status_code, r.headers.get("content-type")))
    r = c.post(f"/api/v1/documents/{pid}/export/pdf", headers=H, json={"filename": "probe"})
    check("Export pdf: Datei (200) oder Abnahme-Befund (422) als JSON", (r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf")) or (r.status_code == 422 and r.json().get("error")), (r.status_code, r.text[:200]))
    r = c.post(f"/api/v1/documents/{pid}/export/hexerei", headers=H)
    check("Export unbekanntes Format: 400", r.status_code == 400, r.text)
    r = c.post(f"/api/v1/documents/{pid}/review-link", headers=H, json={"guest_email": "gast-api@example.invalid", "notify": False, "role": "lektorat"})
    check("Freigabe-Link: 201 + url ohne Mail", r.status_code == 201 and "/freigabe/" in r.json()["url"] and r.json()["email_sent"] is False, r.text)
    token = r.json()["token"]
    r = c.get(f"/api/v1/documents/{pid}/review-links", headers=H)
    check("Freigaben-Liste enthaelt den Link", r.status_code == 200 and any(s["token"] == token for s in r.json()["review_links"]), r.text[:200])
    r = c.post(f"/api/v1/documents/{pid}/review-links/revoke", headers=H, json={"token": token})
    check("Freigabe zurueckgezogen", r.status_code == 200 and r.json()["revoked"], r.text)

    # 3) Word-Dokument: ready -> export docx -> pdfua (JSON + Download)
    with open(os.path.join(FIX, "word_einfach.docx"), "rb") as f:
        r = c.post("/api/v1/documents", headers=H, files={"file": ("Word (fiktiv).docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    check("Word hochladen", r.status_code in (201, 202), r.text)
    wid = r.json()["id"]; angelegt.append(wid)
    d = warte(c, wid, ("ready", "done"))
    check("Word extrahiert: kind docx, Bilder > 0", d["status"] == "ready" and d["kind"] == "docx" and d["counts"]["items"] > 0, d)
    r = c.post(f"/api/v1/documents/{wid}/export/docx", headers=H)
    check("Export docx: Word-Datei", r.status_code == 200 and "wordprocessingml" in r.headers.get("content-type", ""), (r.status_code, r.headers.get("content-type")))
    r = c.post(f"/api/v1/documents/{wid}/export/pdfua", headers=H)
    check("Export pdfua: JSON mit Bericht + download_url", r.status_code == 200 and r.json().get("download_url", "").startswith(B + "/api/v1/documents/") and "bestanden" in r.json(), r.text[:300])
    if r.status_code == 200:
        r2 = c.get(r.json()["download_url"].replace(B, ""), headers=H)
        check("PDF/UA-Download ueber v1", r2.status_code == 200 and r2.headers.get("content-type", "").startswith("application/pdf"), (r2.status_code, r2.headers.get("content-type")))
    r = c.post(f"/api/v1/documents/{wid}/export/pdf", headers=H)
    check("PDF-Export bei Word-Dokument: 400", r.status_code == 400, r.text)

    # 4) Formular: Felder als Items, Quickinfo setzen, CSV
    with open(os.path.join(FIX, "testformular_inkludocs.pdf"), "rb") as f:
        r = c.post("/api/v1/documents", headers=H, files={"file": ("Formular (fiktiv).pdf", f, "application/pdf")}, data={"tool": "formular"})
    check("Formular hochladen", r.status_code in (201, 202), r.text)
    fid = r.json()["id"]; angelegt.append(fid)
    d = warte(c, fid, ("ready", "done"))
    check("Formular extrahiert: kind pdfform, Felder > 0", d["status"] == "ready" and d["kind"] == "pdfform" and d["counts"]["items"] > 0, d)
    r = c.get(f"/api/v1/documents/{fid}/items", headers=H)
    felder = r.json()["items"]
    check("Items sind Felder mit name + quickinfo", felder and all(i["type"] == "field" and "quickinfo" in i for i in felder), r.text[:300])
    r = c.patch(f"/api/v1/documents/{fid}/items/{felder[0]['id']}", headers=H, json={"quickinfo": "Quickinfo per API (fiktiv)"})
    check("PATCH Quickinfo", r.status_code == 200 and r.json()["quickinfo"] == "Quickinfo per API (fiktiv)" and r.json()["text_status"] == "mit_text", r.text[:300])
    r = c.post(f"/api/v1/documents/{fid}/export/formular_csv", headers=H)
    check("Export formular_csv", r.status_code == 200 and "csv" in r.headers.get("content-type", ""), (r.status_code, r.headers.get("content-type")))
    r = c.post(f"/api/v1/documents/{fid}/export/xlsx", headers=H)
    check("xlsx bei Formular: 400", r.status_code == 400, r.text)
    r = c.post(f"/api/v1/documents/{fid}/generate", headers=H, json={})
    check("Formular ohne scope: 400 (scope=all erforderlich, Review M3)", r.status_code == 400 and "scope=all" in r.json()["error"]["message"], r.text[:200])
    r = c.post(f"/api/v1/documents/{fid}/generate", headers=H, json={"scope": "open"})
    check("Formular scope=open: 400", r.status_code == 400, r.text[:200])
    # Review N4: ungueltiger Prompt beim Anlegen darf kein Projekt zuruecklassen
    r0 = c.get("/api/v1/documents", headers=H); vorher = len(r0.json()["documents"])
    with open(os.path.join(FIX, "word_einfach.docx"), "rb") as f:
        r = c.post("/api/v1/documents", headers=H, files={"file": ("x.docx", f, "application/octet-stream")}, data={"prompt_id": "999999999"})
    check("Anlegen mit fremder prompt_id: 404", r.status_code == 404, r.text[:200])
    r1 = c.get("/api/v1/documents", headers=H)
    check("Kein verwaistes Projekt nach abgelehntem Anlegen", len(r1.json()["documents"]) == vorher, (vorher, len(r1.json()["documents"])))
    r = c.post("/api/v1/documents", headers=H, json={"url": "https://example.com/", "prompt": "a", "prompt_id": 1})
    check("prompt und prompt_id zugleich: 400", r.status_code == 400, r.text[:200])

    # 5) Aufraeumen (auch den ueber die API angelegten Prompt der Kategorie „API“)
    for pid_ in list(angelegt):
        r = c.delete(f"/api/v1/documents/{pid_}", headers=H)
        check(f"Loeschen {pid_}", r.status_code == 200 and r.json()["deleted"], r.text)
        r = c.get(f"/api/v1/documents/{pid_}", headers=H)
        check(f"Nach dem Loeschen 404 ({pid_})", r.status_code == 404, r.text)

    # Der Test-Prompt ist ein gespeicherter Prompt des Kontos; ueber die App-Session ist er unter „Meine Prompts“
    # (Kategorie API) sichtbar — bleibt bewusst stehen, damit ein Mensch ihn dort sehen kann (Michael 18.09.).

print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
