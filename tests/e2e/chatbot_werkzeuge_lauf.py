#!/usr/bin/env python3
"""Rahmen fuer chatbot_werkzeuge_probe.py (30.09.2026) plus die HTTP-Seite: legt ein PDF-Projekt (Actino getaggt, Antrag
Pflege mit Formularfeldern, eine ungetaggte PDF) und ein Word-Projekt an, faehrt die Probe im Container, prueft danach ueber
HTTP, dass ausgeblendete Funktionen auch als Endpunkt nicht da sind (404), dass die Seite die Schalter aus EINEM Ort bekommt,
und dass kaputte Uploads einen verstaendlichen Grund bekommen (Audit NIEDRIG 6). Loescht am Ende alles, was es angelegt hat.
Aufruf (Host): python chatbot_werkzeuge_lauf.py <korpus-ordner> <word.docx>
Zugang aus /home/claude/.e2e.env."""
import io
import os
import subprocess
import sys
import time

import requests

env = {}
for z in open("/home/claude/.e2e.env"):
    if "=" in z and not z.startswith("#"):
        k, v = z.strip().split("=", 1)
        env[k] = v.strip().strip('"').strip("'")
B = env["INKLUDOCS_E2E_URL"]
K, DOCX = sys.argv[1], sys.argv[2]
s = requests.Session()
s.post(B + "/api/login", json={"email": env["INKLUDOCS_E2E_MAIL"], "password": env["INKLUDOCS_E2E_PW"]}, timeout=30).raise_for_status()
ok = fehler = 0


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK ", n)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:400])


def projekt(name, tool):
    r = s.post(B + "/api/projects", json={"name": name, "tool": tool}, timeout=30).json()
    return r.get("id") or r.get("project_id")


def hoch(pid, name, daten, media="application/pdf"):
    r = s.post(B + "/api/upload", data={"project_id": str(pid)}, files={"file": (name, daten, media)}, timeout=300)
    for _ in range(150):
        st = s.get(B + f"/api/projects/{pid}", timeout=60).json().get("project") or {}
        if st.get("status") not in ("extracting", "uploading", "processing", None):
            break
        time.sleep(2)
    return r


me = s.get(B + "/api/me", timeout=30).json()
uid = me.get("id") or (me.get("user") or {}).get("id")
projekte = []
try:
    pid = projekt("Chatbot-Werkzeuge 30.09. (Test)", "pdf")
    projekte.append(pid)
    hoch(pid, "actino.pdf", open(os.path.join(K, "actino_master_word.pdf"), "rb").read())
    hoch(pid, "antrag.pdf", open(os.path.join(K, "antrag_pflege.pdf"), "rb").read())
    hoch(pid, "roh.pdf", open(os.path.join(K, "synth_roh.pdf"), "rb").read())
    docs = {d["original_filename"]: d["id"] for d in s.get(B + f"/api/projects/{pid}/dokument-ansicht", timeout=120).json()["documents"]}
    wpid = projekt("Chatbot-Werkzeuge Word 30.09. (Test)", "word")
    projekte.append(wpid)
    hoch(wpid, "wort.docx", open(DOCX, "rb").read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    wdoc = s.get(B + f"/api/projects/{wpid}/dokument-ansicht", timeout=120).json()["documents"][0]["id"]
    print("Projekte", pid, wpid, "Dokumente", docs, wdoc, flush=True)

    print("== HTTP: ausgeblendete Funktionen sind auch als Endpunkt nicht da ==")
    act = docs["actino.pdf"]
    bild = [b["id"] for b in (s.get(B + f"/api/projects/{pid}", timeout=60).json().get("images") or [])][:1] or [0]
    for methode, pfad in (("post", f"/api/projects/{pid}/documents/{act}/pruefung"),
                          ("get", f"/api/projects/{pid}/documents/{act}/pruefung/befunde.csv"),
                          ("post", f"/api/projects/{pid}/documents/{act}/korrektur"),
                          ("post", f"/api/projects/{pid}/documents/{act}/korrektur/rueckgaengig"),
                          ("get", f"/api/projects/{pid}/kette"), ("post", f"/api/projects/{pid}/kette"),
                          ("post", f"/api/images/{bild[0]}/alt-text/zurueck")):
        r = getattr(s, methode)(B + pfad, json={}, timeout=60) if methode == "post" else s.get(B + pfad, timeout=60)
        check(f"{methode.upper()} {pfad.split(str(pid))[-1] if str(pid) in pfad else pfad}: 404", r.status_code == 404, r.status_code)
    seite = s.get(B + f"/app?projekt={pid}", timeout=60).text
    import json
    try:
        roh = seite.split("window.FUNKTIONEN = ", 1)[1].split(";</script>", 1)[0]
        schalter = json.loads(roh)
    except (IndexError, ValueError):
        schalter = {}
    check("Seite bekommt die Schalter aus funktionen.py (window.FUNKTIONEN, alles aus)",
          set(schalter) == {"ki_pruefung", "korrektur", "eigene_pruefungen", "urteil", "kette", "text_zurueck", "strukturansicht"}
          and not any(schalter.values()), schalter)

    print("== Chatbot-Werkzeuge im Container ==")
    subprocess.run(["sudo", "-n", "bash", "-c", "cat /home/openclaw/.openclaw/workspace/InkluDocs/tests/e2e/chatbot_werkzeuge_probe.py > /tmp/cwp.py && "
                    "docker cp /tmp/cwp.py inkludocs-staging:/tmp/chatbot_werkzeuge_probe.py"], check=True)
    r = subprocess.run(["sudo", "-n", "docker", "exec", "-w", "/app", "inkludocs-staging", "python3", "/tmp/chatbot_werkzeuge_probe.py",
                        str(pid), str(uid), str(act), str(docs["antrag.pdf"]), str(docs["roh.pdf"]), str(wpid), str(wdoc)],
                       capture_output=True, text=True, timeout=1500)
    zeilen = [z for z in r.stdout.splitlines() if z.startswith(("  OK", "  FEHLT", "==", "Ergebnis"))]
    print("\n".join(zeilen))
    if r.returncode not in (0, 1):
        print(r.stderr[-3000:])
    for z in zeilen:
        if z.startswith("  OK"):
            ok += 1
        elif z.startswith("  FEHLT"):
            fehler += 1
    if not any(z.startswith("Ergebnis") for z in zeilen):
        fehler += 1
        print("  FEHLT Probe lief nicht zu Ende --", r.stderr[-1500:])

    print("== Upload: verständliche Gründe statt „Verarbeitung fehlgeschlagen“ ==")
    import pikepdf
    buf = io.BytesIO()
    with pikepdf.open(os.path.join(K, "synth_roh.pdf")) as p:
        p.save(buf, encryption=pikepdf.Encryption(user="geheim-test", owner="geheim-test"))
    ganz = open(os.path.join(K, "actino_master_word.pdf"), "rb").read()
    for name, daten, erwartet in (("keine.pdf", b"Das ist nur Text, keine PDF.\n" * 20, "keine PDF"),
                                  ("passwort.pdf", buf.getvalue(), "Passwort"),
                                  ("abgeschnitten.pdf", ganz[: len(ganz) // 2], "beschädigt")):
        r = s.post(B + "/api/upload", data={"project_id": str(pid)}, files={"file": (name, daten, "application/pdf")}, timeout=120)
        detail = (r.json() or {}).get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
        check(f"{name}: 400 mit Grund „{erwartet}“, nichts angelegt", r.status_code == 400 and erwartet in str(detail), (r.status_code, detail))
    n = len(s.get(B + f"/api/projects/{pid}/dokument-ansicht", timeout=120).json()["documents"])
    check("keine Dokument-Leichen", n == 3, n)
finally:
    for p in projekte:
        for e in (s.get(B + f"/api/ausgaben?projekt={p}", timeout=60).json().get("ausgaben") or []):
            s.delete(B + f"/api/ausgaben/{e['id']}", timeout=60)
        print("  Testprojekt geloescht:", p, s.delete(B + f"/api/projects/{p}", timeout=60).status_code)
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
