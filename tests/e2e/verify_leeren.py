#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2E: Ein geleerter Alt-Text bleibt leer — Anzeige, Zaehler und Export.

Michael Karbe, 31.08.2026: „Beim Herunterladen sagst Du, dass 2 Bilder eine
Beschreibung haben, obwohl ich das Feld geleert habe."

Ursache war die or-Kette in _display_alt_text: Ein geleertes Feld wird als
leerer String gespeichert, leer ist falsy, also kam der alte KI-Text zurueck —
in der Anzeige, im Zaehler und in der exportierten PDF.

Der Test stellt genau das nach und raeumt hinterher auf:
  1. Ein Bild mit KI-Text suchen.
  2. Feld ueber die echte Schnittstelle LEEREN.
  3. Anzeige muss leer sein (nicht der KI-Text).
  4. Der Zaehler "beschrieben" in der Export-Zusammenfassung muss um 1 sinken.
  5. Das Bild darf im Export nicht mehr als beschrieben gelten.
  6. Danach einen eigenen Text setzen — der muss stehen.
  7. Ausgangszustand wiederherstellen (alt_text_edited zurueck auf NULL).

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

# Ein PDF-Projekt mit beschriebenen Bildern suchen.
projekte = hole("/api/projects", token)
projekte = projekte if isinstance(projekte, list) else projekte.get("projects", [])
ziel = bild = None
for p in projekte:
    if p.get("project_type") != "pdf":
        continue
    d = hole("/api/projects/%d" % p["id"], token)
    for img in d.get("images", []):
        if (img.get("alt_text") or "").strip() and not (img.get("alt_text_edited") or "").strip():
            ziel, bild = p, img
            break
    if bild:
        break
if not bild:
    sys.exit("ABBRUCH: kein PDF-Bild mit KI-Text ohne Hand-Text gefunden")
print("Testbild: Projekt %d, Bild %d, KI-Text %r\n" % (ziel["id"], bild["id"], (bild["alt_text"] or "")[:48]))

vorher = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
print("Export-Zusammenfassung vorher: beschrieben=%s von %s\n" % (vorher.get("beschrieben"), vorher.get("total")))
# 01.09.2026 (Pruefbefund): Ein bewusst geleertes Bild darf KEIN Kandidat fuer
# „Alt-Texte generieren" (Modus ki_neu) mehr sein — sonst wird es bezahlt und
# beschrieben, der Text bleibt aber unsichtbar (leeres Feld hat Vorrang).
kand_vorher = hole("/api/projects/%d/generate/vorschau" % ziel["id"], token, {"modus": "ki_neu"})
print("Kandidaten „neu erzeugen“ vorher: %s\n" % kand_vorher.get("anzahl"))

print("== Feld leeren ==")
hole("/api/images/%d/alt-text" % bild["id"], token, {"alt_text": ""})
kand_nachher = hole("/api/projects/%d/generate/vorschau" % ziel["id"], token, {"modus": "ki_neu"})
check("Geleertes Bild ist kein Kandidat mehr fuer „neu erzeugen“ (Anzahl um 1 gesunken)",
      kand_nachher.get("anzahl") == kand_vorher.get("anzahl") - 1,
      "%s -> %s" % (kand_vorher.get("anzahl"), kand_nachher.get("anzahl")))
erst = hole("/api/projects/%d/generate/vorschau" % ziel["id"], token, {"modus": "luecken"})
check("Vorschau Erstlauf liefert mit_quelltext (mitgebrachte Texte)", "mit_quelltext" in erst, str(erst)[:100])
d = hole("/api/projects/%d" % ziel["id"], token)
neu = [i for i in d["images"] if i["id"] == bild["id"]][0]
check("Anzeige ist leer (kein KI-Text zurueck)",
      not (neu.get("alt_text_edited") or "").strip() and neu.get("alt_text_edited") == "",
      "alt_text_edited=%r" % neu.get("alt_text_edited"))
check("Der KI-Text steht noch in seinem eigenen Feld",
      (neu.get("alt_text") or "").strip() != "", "(er wird nur nicht mehr angezeigt)")

nachher = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
print("Export-Zusammenfassung nachher: beschrieben=%s von %s" % (nachher.get("beschrieben"), nachher.get("total")))
check("Zaehler 'beschrieben' um genau 1 gesunken",
      nachher.get("beschrieben") == vorher.get("beschrieben") - 1,
      "%s -> %s" % (vorher.get("beschrieben"), nachher.get("beschrieben")))
check("Gesamtzahl unveraendert", nachher.get("total") == vorher.get("total"))
# 01.09.2026 (Steve: „wieso noch nicht generiert, der hat doch alle generiert“): ein
# bewusst geleertes Bild ist „bewusst ohne Beschreibung“ (geleert), NICHT „noch nicht generiert“.
check("Das Bild zaehlt jetzt als bewusst geleert (geleert + 1)",
      nachher.get("geleert", 0) == vorher.get("geleert", 0) + 1,
      "%s -> %s" % (vorher.get("geleert"), nachher.get("geleert")))
check("… und NICHT als noch nicht generiert (offen unveraendert)",
      nachher.get("offen", 0) == vorher.get("offen", 0),
      "%s -> %s" % (vorher.get("offen"), nachher.get("offen")))
check("uebersprungen = fehler + offen + geleert",
      nachher.get("uebersprungen") == nachher.get("fehler", 0) + nachher.get("offen", 0) + nachher.get("geleert", 0), str(nachher))

print("\n== Eigenen Text setzen ==")
hole("/api/images/%d/alt-text" % bild["id"], token, {"alt_text": "Prüftext geleert-Fix"})
d = hole("/api/projects/%d" % ziel["id"], token)
neu = [i for i in d["images"] if i["id"] == bild["id"]][0]
check("Eigener Text steht", neu.get("alt_text_edited") == "Prüftext geleert-Fix",
      repr(neu.get("alt_text_edited")))
z = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
check("Zaehler wieder auf dem Ausgangswert", z.get("beschrieben") == vorher.get("beschrieben"),
      "%s" % z.get("beschrieben"))

print("\n== Ausgangszustand wiederherstellen ==")
subprocess.run(["sudo", "docker", "exec", "inkludocs-staging", "python3", "-c",
                "import sqlite3;c=sqlite3.connect('/app/data/inkludocs.db');"
                "c.execute('UPDATE images SET alt_text_edited=NULL WHERE id=%d');c.commit()" % bild["id"]],
               check=True)
z = hole("/api/projects/%d/export/summary" % ziel["id"], token, {})
check("Zaehler wieder wie zu Beginn", z.get("beschrieben") == vorher.get("beschrieben"),
      "%s vs %s" % (z.get("beschrieben"), vorher.get("beschrieben")))
kand_ende = hole("/api/projects/%d/generate/vorschau" % ziel["id"], token, {"modus": "ki_neu"})
check("Kandidaten „neu erzeugen“ wieder wie zu Beginn", kand_ende.get("anzahl") == kand_vorher.get("anzahl"),
      "%s vs %s" % (kand_ende.get("anzahl"), kand_vorher.get("anzahl")))

print()
print("%d/%d Pruefungen bestanden" % (ok, ok + fail))
sys.exit(1 if fail else 0)
