#!/usr/bin/env python3
"""Pruefung 30.09.2026 (M1), laeuft IM Staging-Container mit echtem Bau: eine Quickinfo-Aenderung, die GENAU zwischen Planung
und Schreiben ankommt (erzwungen: die Aenderung wird unmittelbar vor dem Schreiben in die Datenbank gesetzt), kommt nicht in
diese Datei — geschrieben und berechnet wird die Momentaufnahme aus der Planung.
  A  geplant ohne Bearbeitung (0 Credits), waehrend des Baus geaendert -> Datei ohne die neue Quickinfo, 0 gebucht;
     danach kostet die neue Quickinfo (sie ist noch nicht bezahlt)
  B  geplant mit Quickinfo „A“ (26), waehrend des Baus auf „B“ geaendert -> Datei mit „A“, 26 gebucht; danach kostet „B“ wieder 26
Aufruf (im Container): python3 /tmp/momentaufnahme_probe.py <projekt-id> <user-id> <dokument-id> <feld-id>
Das Dokument ist ein Formular (z. B. antrag_pflege.pdf); das Feld bekommt fiktive Testtexte (Testprojekt)."""
import sys

sys.path.insert(0, "/app")
PID, UID, DID, FID = map(int, sys.argv[1:5])
import fitz  # noqa: E402
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


def setze(text):
    sql("UPDATE formularfelder SET quickinfo = ?, quelle = ? WHERE id = ?", text, "hand" if text else "", FID, commit=True)


def gebucht_seit(eid):
    return sql("SELECT COALESCE(SUM(credits), 0) FROM usage_events WHERE user_id = ? AND id > ?", UID, eid)[0][0]


def plan_preis():
    return main._pdf_export_plan(UID, main._load_pdf_export_units(projekt, UID, DID))["preis"]


def in_datei(pfad, text):
    with fitz.open(pfad) as d:
        return any((w.field_label or "") == text for p in d for w in (p.widgets() or []))


projekt = dict(sql("SELECT * FROM projects WHERE id = ?", PID)[0])
_orig = main._quickinfos_in_export
_spaet = {"text": None}


def _dazwischen(*a, **k):
    if _spaet["text"] is not None:
        setze(_spaet["text"])      # die Aenderung kommt NACH der Planung und VOR dem Schreiben an
        _spaet["text"] = None
    return _orig(*a, **k)


main._quickinfos_in_export = _dazwischen

print("== A. geplant ohne Bearbeitung, während des Baus geändert ==")
setze("")
e0 = sql("SELECT COALESCE(MAX(id), 0) FROM usage_events")[0][0]
_spaet["text"] = "Fiktive Quickinfo mitten im Bau"
sql("UPDATE ablage SET bau_stand = '' WHERE project_id = ?", PID, commit=True)   # kein Ablage-Treffer: wirklich bauen
erg = main._pdf_export_sync(UID, projekt, DID, None, "knopf")
check("Geschrieben wurde die Momentaufnahme: neue Quickinfo NICHT in der Datei, 0 gebucht",
      not in_datei(erg["pfad"], "Fiktive Quickinfo mitten im Bau") and erg["preis"] == 0 and gebucht_seit(e0) == 0, (erg["preis"], gebucht_seit(e0)))
check("Danach kostet die neue Quickinfo (26), sie ist nicht gratis mitgegangen", plan_preis() == 26, plan_preis())

print("== B. geplant mit Quickinfo „A“, während des Baus auf „B“ geändert ==")
setze("Fiktive Quickinfo A")
e0 = sql("SELECT COALESCE(MAX(id), 0) FROM usage_events")[0][0]
_spaet["text"] = "Fiktive Quickinfo B"
erg = main._pdf_export_sync(UID, projekt, DID, None, "knopf")
check("Datei mit „A“ (nicht „B“), 26 gebucht", in_datei(erg["pfad"], "Fiktive Quickinfo A") and not in_datei(erg["pfad"], "Fiktive Quickinfo B")
      and erg["preis"] == 26 and gebucht_seit(e0) == 26, (erg["preis"], gebucht_seit(e0)))
check("Danach kostet „B“ wieder 26 (bezahlt ist „A“)", plan_preis() == 26, plan_preis())
setze("")
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
