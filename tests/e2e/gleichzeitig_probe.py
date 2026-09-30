#!/usr/bin/env python3
"""Pruefung 30.09.2026 (H1/M2), laeuft IM Staging-Container, in EINEM Prozess wie der Server (dort teilen Knopf und Chatbot
dieselbe Sperre im Speicher):
  1. Chatbot (exportiere_fertige_pdf) und Knopf (derselbe Weg wie POST /export: _export_belegen + _pdf_export_sync)
     gleichzeitig, in beiden Reihenfolgen: genau einer baut, der andere bekommt „wird gerade schon eine PDF erstellt“,
     einmal gebucht.
  2. Atomarer Anspruch OHNE Sperre (als liefen zwei Server-Prozesse): zwei Bauten desselben Stands gleichzeitig ->
     genau eine Buchung von 30, der andere 0.
  3. Derselbe Stand danach ueber den Chatbot: aus der Ablage, kein neuer Eintrag.
Aufruf (im Container): python3 /tmp/gleichzeitig_probe.py <projekt-id> <user-id> <dokument-id>
Das Dokument muss getaggt sein und Bilder haben (z. B. actino_master_word.pdf). Setzt Alt-Texte des ersten Bildes auf
fiktive Testtexte (Testprojekt)."""
import sys
import threading
import time

sys.path.insert(0, "/app")
PID, UID, DID = map(int, sys.argv[1:4])
import main  # noqa: E402
from inkluagent.tools import pdf as T  # noqa: E402

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


def letzte_buchung():
    return sql("SELECT COALESCE(MAX(id), 0) FROM usage_events")[0][0]


def gebucht_seit(eid):
    return sql("SELECT COALESCE(SUM(credits), 0) FROM usage_events WHERE user_id = ? AND id > ?", UID, eid)[0][0]


def ablage_anzahl():
    return sql("SELECT COUNT(*) FROM ablage WHERE user_id = ? AND project_id = ?", UID, PID)[0][0]


bild = sql("SELECT id FROM images WHERE document_id = ? ORDER BY page_number, image_index LIMIT 1", DID)[0][0]
projekt = dict(sql("SELECT * FROM projects WHERE id = ?", PID)[0])


def neuer_stand(text):
    sql("UPDATE images SET alt_text_edited = ? WHERE id = ?", text, bild, commit=True)


def knopf(out):
    grund = main._export_belegen(UID)
    if grund:
        out["knopf"] = ("belegt", main._export_belegt_text(grund))
        return
    try:
        erg = main._pdf_export_sync(UID, projekt, DID, None, "knopf")
        out["knopf"] = ("ok", erg["preis"])
    finally:
        main._export_freigeben(UID)


def bot(out):
    r = T.exportiere_fertige_pdf(PID, UID, DID, bestaetigt=True)
    out["bot"] = ("ok", (r.get("result") or {}).get("preis")) if r.get("ok") and (r.get("result") or {}).get("ausgabe_id") else ("belegt", r.get("error"))


print("== 1. Chatbot und Knopf gleichzeitig ==")
for runde, reihenfolge in enumerate((("bot", "knopf"), ("knopf", "bot")), 1):
    neuer_stand(f"Fiktiver Alt-Text Gleichzeitig-Probe Runde {runde}")
    angebot = T.exportiere_fertige_pdf(PID, UID, DID, bestaetigt=False)
    preis = ((angebot.get("result") or {}).get("vorschau") or angebot.get("result") or {}).get("preis")
    e0, a0 = letzte_buchung(), ablage_anzahl()
    out = {}
    faeden = {"bot": threading.Thread(target=bot, args=(out,)), "knopf": threading.Thread(target=knopf, args=(out,))}
    faeden[reihenfolge[0]].start()
    time.sleep(0.05)
    faeden[reihenfolge[1]].start()
    for f in faeden.values():
        f.join()
    arten = sorted(v[0] for v in out.values())
    belegt = [v[1] for v in out.values() if v[0] == "belegt"]
    check(f"Runde {runde} ({reihenfolge[0]} zuerst): einer baut, der andere „wird gerade schon eine PDF erstellt“",
          arten == ["belegt", "ok"] and "gerade schon eine PDF" in (belegt[0] or ""), out)
    check(f"Runde {runde}: genau einmal 30 gebucht (Angebot {preis}), ein Ablage-Eintrag",
          gebucht_seit(e0) == 30 and ablage_anzahl() == a0 + 1, (gebucht_seit(e0), a0, ablage_anzahl()))

print("== 2. Atomarer Anspruch ohne Sperre (zwei Bauten desselben Stands) ==")
neuer_stand("Fiktiver Alt-Text Gleichzeitig-Probe ohne Sperre")
e0 = letzte_buchung()
preise = []
bar = threading.Barrier(2)


def ohne_sperre():
    bar.wait()
    preise.append(main._pdf_export_sync(UID, projekt, DID, None, "knopf")["preis"])


fs = [threading.Thread(target=ohne_sperre) for _ in range(2)]
for f in fs:
    f.start()
for f in fs:
    f.join()
check("Zwei gleichzeitige Bauten: einer bucht 30, der andere 0; insgesamt 30", sorted(preise) == [0, 30] and gebucht_seit(e0) == 30, (preise, gebucht_seit(e0)))
bezahlt = sql("SELECT export_bezahlt FROM documents WHERE id = ?", DID)[0][0] or ""
check("Bezahlter Stand je Teil gemerkt (a=…;q=)", bezahlt.startswith("a=") and ";q=" in bezahlt, bezahlt[:30])

print("== 3. Derselbe Stand noch einmal über den Chatbot: aus der Ablage ==")
T.exportiere_fertige_pdf(PID, UID, DID, bestaetigt=False)
e0, a0 = letzte_buchung(), ablage_anzahl()
r = T.exportiere_fertige_pdf(PID, UID, DID, bestaetigt=True)
res = r.get("result") or {}
check("Chatbot: 0 Credits, Ausgabe = vorhandener Ablage-Eintrag, kein neuer Eintrag",
      r.get("ok") and res.get("preis") == 0 and res.get("ausgabe_id") and gebucht_seit(e0) == 0 and ablage_anzahl() == a0, (r, a0, ablage_anzahl()))
sql("UPDATE images SET alt_text_edited = NULL WHERE id = ?", bild, commit=True)
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
