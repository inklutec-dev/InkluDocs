#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2E: Export ist, was der Kunde sieht (Michael Karbe/Steve 01.09.2026) — Leeren wirkt.

Michael Karbe, 31.08.2026: „Beim Herunterladen sagst Du, dass 2 Bilder eine
Beschreibung haben, obwohl ich das Feld geleert habe." Seit 01.09.2026 gilt:
  - Leeres Feld = kein Text in der Datei (Zusammenfassung: „ohne Text").
  - Der Sammellauf „Alt-Texte generieren" nimmt IMMER alle Bilder (auch geleerte
    und eigene) — die Rueckfrage nennt die Zahl der eigenen Texte (eigene).
  - Eigener Text gewinnt in Anzeige und Export.

Der Test stellt das ueber die echte Schnittstelle nach und raeumt hinterher auf:
  1. Ein Bild mit KI-Text ohne Hand-Text suchen.
  2. Feld LEEREN -> alt_text_edited == "", KI-Text bleibt im Fach, Zusammenfassung
     mit_text -1 / ohne_text +1, Vorschau-Anzahl unveraendert (alle), eigene 0.
  3. Eigenen Text setzen -> mit_text zurueck, Vorschau eigene +1.
  4. Ausgangszustand wiederherstellen.

Aufruf: python3 verify_leeren.py   (Zugangsdaten aus ~/.e2e.env)
"""
import json
import os
import subprocess
import sys
import urllib.request

BASIS = "https://staging.inkludocs.inklutec.de"

env = {}
with open(os.path.expanduser("~/.e2e.env"), encoding="utf-8") as f:
    for z in f:
        z = z.strip()
        if z and not z.startswith("#") and "=" in z:
            k, v = z.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

ok = fail = 0


def check(name, bedingung, zusatz=""):
    global ok, fail
    if bedingung:
        ok += 1
        print("  OK   %s" % name)
    else:
        fail += 1
        print("  FAIL %s %s" % (name, zusatz))


def hole(pfad, token, daten=None):
    req = urllib.request.Request(BASIS + pfad, headers={"Cookie": "token=" + token})
    if daten is not None:
        req.data = json.dumps(daten).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=90) as r:
        roh = r.read().decode("utf-8")
    return json.loads(roh) if roh.strip() else {}


req = urllib.request.Request(
    BASIS + "/api/login",
    data=json.dumps({"email": env["INKLUDOCS_E2E_MAIL"],
                     "password": env["INKLUDOCS_E2E_PW"]}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:
    token = r.headers.get("Set-Cookie", "").split("token=", 1)[1].split(";", 1)[0]
print("Angemeldet.\n")

projekte = hole("/api/projects", token)
projekte = projekte if isinstance(projekte, list) else projekte.get("projects", [])
ziel = bild = None
for p in projekte:
    if p.get("project_type") != "pdf":
        continue
    d = hole("/api/projects/%d" % p["id"], token)
    for img in d.get("images", []):
        if (img.get("alt_text") or "").strip() and img.get("alt_text_edited") is None and img.get("image_type") != "dekorativ":
            ziel, bild = p, img
            break
    if bild:
        break
if not bild:
    sys.exit("ABBRUCH: kein PDF-Bild mit KI-Text ohne Hand-Text gefunden")
print("Testbild: Projekt %d, Bild %d, KI-Text %r\n" % (ziel["id"], bild["id"], (bild["alt_text"] or "")[:48]))

vorher = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
v_vorher = hole("/api/projects/%d/generate/vorschau" % ziel["id"], token, {"modus": "alle"})
print("Zusammenfassung vorher: mit_text=%s ohne_text=%s von %s | Vorschau: anzahl=%s eigene=%s\n"
      % (vorher.get("mit_text"), vorher.get("ohne_text"), vorher.get("total"), v_vorher.get("anzahl"), v_vorher.get("eigene")))
check("Zusammenfassung liefert mit_text/ohne_text", "mit_text" in vorher and "ohne_text" in vorher, str(vorher)[:120])
check("Vorschau hat den einen Modus „alle“ und liefert eigene", v_vorher.get("modus") == "alle" and "eigene" in v_vorher, str(v_vorher)[:120])

print("== Feld leeren ==")
hole("/api/images/%d/alt-text" % bild["id"], token, {"alt_text": ""})
d = hole("/api/projects/%d" % ziel["id"], token)
neu = [i for i in d["images"] if i["id"] == bild["id"]][0]
check("Anzeige ist leer (alt_text_edited == '')", neu.get("alt_text_edited") == "", "alt_text_edited=%r" % neu.get("alt_text_edited"))
check("Der KI-Text steht noch in seinem eigenen Fach", (neu.get("alt_text") or "").strip() != "")
nachher = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
check("mit_text um genau 1 gesunken", nachher.get("mit_text") == vorher.get("mit_text") - 1, "%s -> %s" % (vorher.get("mit_text"), nachher.get("mit_text")))
check("ohne_text um genau 1 gestiegen", nachher.get("ohne_text") == vorher.get("ohne_text") + 1, "%s -> %s" % (vorher.get("ohne_text"), nachher.get("ohne_text")))
check("Gesamtzahl unveraendert", nachher.get("total") == vorher.get("total"))
v_leer = hole("/api/projects/%d/generate/vorschau" % ziel["id"], token, {"modus": "alle"})
check("Sammellauf nimmt weiterhin ALLE Bilder (geleertes bleibt Kandidat)", v_leer.get("anzahl") == v_vorher.get("anzahl"), "%s vs %s" % (v_vorher.get("anzahl"), v_leer.get("anzahl")))
check("Geleertes Feld zaehlt nicht als eigener Text", v_leer.get("eigene") == v_vorher.get("eigene"), "%s vs %s" % (v_vorher.get("eigene"), v_leer.get("eigene")))

print("\n== Eigenen Text setzen ==")
hole("/api/images/%d/alt-text" % bild["id"], token, {"alt_text": "Prüftext Export-ist-Browser"})
d = hole("/api/projects/%d" % ziel["id"], token)
neu = [i for i in d["images"] if i["id"] == bild["id"]][0]
check("Eigener Text steht", neu.get("alt_text_edited") == "Prüftext Export-ist-Browser", repr(neu.get("alt_text_edited")))
z = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
check("mit_text wieder auf dem Ausgangswert", z.get("mit_text") == vorher.get("mit_text"), "%s" % z.get("mit_text"))
v_eig = hole("/api/projects/%d/generate/vorschau" % ziel["id"], token, {"modus": "alle"})
check("Vorschau zaehlt den eigenen Text (eigene + 1)", v_eig.get("eigene") == v_vorher.get("eigene") + 1, "%s -> %s" % (v_vorher.get("eigene"), v_eig.get("eigene")))
check("Sammellauf nimmt das Bild trotzdem (Generieren ueberschreibt alles)", v_eig.get("anzahl") == v_vorher.get("anzahl"))

print("\n== Ausgangszustand wiederherstellen ==")
subprocess.run(["sudo", "docker", "exec", "inkludocs-staging", "python3", "-c",
                "import sqlite3;c=sqlite3.connect('/app/data/inkludocs.db');"
                "c.execute('UPDATE images SET alt_text_edited=NULL WHERE id=%d');c.commit()" % bild["id"]],
               check=True)
z = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
check("Zusammenfassung wieder wie zu Beginn", z.get("mit_text") == vorher.get("mit_text") and z.get("ohne_text") == vorher.get("ohne_text"), str(z)[:100])

print()
print("%d/%d Pruefungen bestanden" % (ok, ok + fail))
sys.exit(1 if fail else 0)
