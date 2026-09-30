#!/usr/bin/env python3
"""Nachpruefung 30.09.2026 (MITTEL), laeuft IM Staging-Container mit echtem Bau: Obergrenze der Ablage und Ersetzen
kostenloser Eintraege.
  1. Ablage voll (Grenze = aktuelle Anzahl), kein kostenloser Eintrag des Dokuments da -> kostenloser Download liefert die
     Datei, legt NICHTS ab, Hinweis „Deine Ablage ist voll …“ in X-Export-Warnings
  2. Ablage frei -> kostenloser Download legt einen ersetzbaren Eintrag an; Ablage wieder voll, Anzeigename geaendert (Titel
     aendert sich) -> der neue Bau ERSETZT den kostenlosen Eintrag (Anzahl gleich, alte Datei weg)
  3. Ablage voll, bezahlter Download (Alt-Text bearbeitet) -> wird trotzdem abgelegt, nicht ersetzbar
Aufruf (im Container): python3 /tmp/ablage_probe.py <projekt-id> <user-id> <dokument-id>
Das Dokument: getaggt, ohne eigenen Titel, mit Bild (z. B. synth_getaggt.pdf). Setzt fiktive Testtexte (Testprojekt)."""
import json
import os
import sys
from unittest import mock

sys.path.insert(0, "/app")
PID, UID, DID = map(int, sys.argv[1:4])
import main  # noqa: E402

ok = fehler = 0


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK ", n)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:400])


def sql(q, *a, commit=False):
    c = main.get_db()
    try:
        r = c.execute(q, a).fetchall()
        if commit:
            c.commit()
        return r
    finally:
        c.close()


def anzahl():
    return sql("SELECT COUNT(*) FROM ablage WHERE user_id = ?", UID)[0][0]


def eintraege_dok():
    return [dict(r) for r in sql("SELECT id, datei_pfad, preis, COALESCE(ersetzbar, 0) AS ersetzbar FROM ablage "
                                 "WHERE user_id = ? AND document_id = ? ORDER BY id", UID, DID)]


def projekt():
    return dict(sql("SELECT * FROM projects WHERE id = ?", PID)[0])


def laden():
    erg = main._pdf_export_sync(UID, projekt(), DID, None, "knopf")
    main._export_anfrage_weg(erg.get("anfrage_dir"))
    return erg


def warnungen(erg):
    return json.loads(erg["headers"].get("X-Export-Warnings") or "[]")


for e in eintraege_dok():
    main._ablage_eintrag_weg(UID, sql("SELECT * FROM ablage WHERE id = ?", e["id"])[0])

print("== 1. Ablage voll, kein kostenloser Eintrag da ==")
n0 = anzahl()
with mock.patch.object(main, "ABLAGE_MAX_EINTRAEGE", n0):
    erg = laden()
check("Datei geliefert, nichts abgelegt, Hinweis „Deine Ablage ist voll“",
      os.path.basename(erg["pfad"]).endswith(".pdf") and erg["ausgabe_ids"] == [] and anzahl() == n0
      and any("Deine Ablage ist voll" in w for w in warnungen(erg)), (erg["ausgabe_ids"], anzahl(), warnungen(erg)))

print("== 2. Kostenloser Eintrag wird ersetzt, auch bei voller Ablage ==")
erg = laden()
e1 = eintraege_dok()
check("Ablage frei: ein ersetzbarer Eintrag", len(e1) == 1 and e1[0]["ersetzbar"] == 1 and e1[0]["preis"] == 0, e1)
sql("UPDATE documents SET display_name = 'Ablage-Probe neuer Titel (Test)' WHERE id = ?", DID, commit=True)
n1 = anzahl()
with mock.patch.object(main, "ABLAGE_MAX_EINTRAEGE", n1):
    erg = laden()
e2 = eintraege_dok()
check("Neuer Stand (Titel aus dem Namen): Neubau ersetzt den Eintrag — Anzahl gleich, neue id, alte Datei weg",
      len(e2) == 1 and e2[0]["id"] != e1[0]["id"] and anzahl() == n1 and not os.path.exists(e1[0]["datei_pfad"])
      and not warnungen(erg), (e1, e2, anzahl(), warnungen(erg)))

print("== 3. Bezahlter Download bei voller Ablage ==")
bild = sql("SELECT id FROM images WHERE document_id = ? ORDER BY id LIMIT 1", DID)
if bild:
    sql("UPDATE images SET alt_text_edited = 'Fiktiver Alt-Text Ablage-Probe' WHERE id = ?", bild[0][0], commit=True)
    n2 = anzahl()
    with mock.patch.object(main, "ABLAGE_MAX_EINTRAEGE", n2):
        erg = laden()
    e3 = eintraege_dok()
    bezahlt = [e for e in e3 if e["preis"] > 0]
    check("Bezahlt: trotz voller Ablage abgelegt, nicht ersetzbar; der kostenlose Eintrag bleibt",
          erg["preis"] > 0 and anzahl() == n2 + 1 and len(bezahlt) == 1 and bezahlt[0]["ersetzbar"] == 0 and len(e3) == 2, (erg["preis"], e3))
    sql("UPDATE images SET alt_text_edited = NULL WHERE id = ?", bild[0][0], commit=True)
else:
    print("  (übersprungen: Dokument ohne Bild)")
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
