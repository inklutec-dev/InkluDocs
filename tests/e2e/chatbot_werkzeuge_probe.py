#!/usr/bin/env python3
"""Chatbot = Oberflaeche (30.09.2026): jedes neue Werkzeug IM Staging-Container gegen echte Daten, so wie der InkluAgent es
aufruft (ToolExecutor, Angebot in einer Nachricht, Ja in der naechsten). Laeuft in einem eigenen Prozess; die Ereignisschleife
des Servers (fuer die Sammellaeufe, main.im_hauptloop) ist hier eine eigene Schleife in einem Hintergrund-Thread.
Aufruf (im Container): python3 /tmp/chatbot_werkzeuge_probe.py <pdf-projekt> <user> <actino-doc> <antrag-doc> <roh-doc>
                                                               <word-projekt> <word-doc>
Rahmen (Projekte anlegen, danach loeschen): chatbot_werkzeuge_lauf.py. Kostet Credits des Testkontos (Exporte, 1 Alt-Text,
Quickinfos eines Formulars) und wenige KI-Aufrufe."""
import asyncio
import datetime
import json
import re
import os
import sys
import threading
import time

sys.path.insert(0, "/app")
PID, UID, ACT, ANT, ROH, WPID, WDOC = map(int, sys.argv[1:8])
import main  # noqa: E402
from inkluagent.tools.definitions import ToolExecutor  # noqa: E402

schleife = asyncio.new_event_loop()
threading.Thread(target=schleife.run_forever, daemon=True).start()
main._HAUPT_LOOP = schleife

ok = fehler = 0


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK ", n)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:500])


def sql(q, *a):
    c = main.get_db()
    try:
        return c.execute(q, a).fetchall()
    finally:
        c.close()


def gebucht_seit(eid):
    return sql("SELECT COALESCE(SUM(credits), 0) FROM usage_events WHERE user_id = ? AND id > ?", UID, eid)[0][0]


def letzte():
    return sql("SELECT COALESCE(MAX(id), 0) FROM usage_events")[0][0]


def ex(pid, word=False):
    return ToolExecutor(project_id=pid, user_id=UID, word=word, pdf=not word)


def zwei(pid, name, args, word=False):
    """Angebot (Nachricht 1), Ja (Nachricht 2) — wie im Chat."""
    r1 = ex(pid, word).execute(name, dict(args, bestaetigt=False))
    r2 = ex(pid, word).execute(name, dict(args, bestaetigt=True))
    return r1, r2


def datei_zum_token(url):
    token = url.rsplit("/", 1)[-1]
    meta = os.path.join(main.RESULTS_DIR, str(UID), url.split("/")[3], "_export", f"pdfua_{token}.json")
    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    with open(m["pfad"], "rb") as f:
        return m, f.read()


print("== Tagging: testweise_taggen ==")
r = ex(PID).execute("testweise_taggen", {"document_id": ACT})
check("schon getaggte PDF: kein Testlauf, Grund genannt", not r["ok"] and "schon getaggt" in r.get("error", ""), r)
r = ex(PID).execute("testweise_taggen", {"document_id": ROH})
check("ungetaggte PDF: Testlauf gestartet, kostenlos", r["ok"] and r["result"].get("gestartet") and r["result"].get("preis") == 0, r)
tl = None
for _ in range(90):
    time.sleep(2)
    st = ex(PID).execute("dokument_stand", {"document_id": ROH})
    tl = st["result"]["dokumente"][0].get("testlauf") or {}
    if tl.get("zeit") and not tl.get("laeuft"):
        break
check("dokument_stand zeigt den fertigen Testlauf (Zeit, Struktur, PDF/UA)", bool(tl.get("zeit")) and tl.get("struktur"), tl)
st = ex(PID).execute("dokument_stand", {})
d_act = [d for d in st["result"]["dokumente"] if d["document_id"] == ACT][0]
check("dokument_stand ohne KI-Prüfung und ohne Kette (ausgeblendet)", "pruefung" not in d_act and st["result"].get("kette") is None, list(d_act))

print("== Barrierefreiheitsprüfung: pruefdatei_erstellen, pruefdatei_lesen ==")
r = ex(PID).execute("pruefdatei_erstellen", {"document_id": ACT})
res = r.get("result") or {}
check("Prüfdatei erstellt: Urteil von veraPDF und Problemstellen mit Seiten", r["ok"] and res.get("verapdf_moeglich") is True
      and isinstance(res.get("probleme"), list) and res.get("anzahl_probleme") == len(res.get("probleme") or []), r)
check("Problemstellen im neuen Klartext (keine „32-mal“ für 16 Links)", not any("32-mal" in (p.get("anzahl") or "") for p in res.get("probleme") or []),
      [(p["satz"], p["anzahl"]) for p in (res.get("probleme") or [])][:6])
r = ex(PID).execute("pruefdatei_lesen", {"document_id": ACT, "teil": "hoerprobe", "anzahl": 40})
z = (r.get("result") or {}).get("zeilen_daten") or []
check("Hörprobe der fertigen Datei vorlesbar (Seitenmarke, Zeilen)", r["ok"] and any("— Seite 1 —" in x for x in z) and len(z) > 5, z[:5])
r = ex(PID).execute("pruefdatei_lesen", {"document_id": ROH})
check("ungetaggt: keine Prüfdatei, Grund", r["ok"] and r["result"]["status"] == "keine_pruefdatei" and r["result"]["getaggt"] is False, r)

print("== Herunterladen: Alt-Texte, Quickinfos, PDF ==")
e0 = letzte()
r1, r2 = zwei(PID, "exportiere_alt_texte", {"format": "csv", "document_id": ACT})
check("Alt-Texte CSV: erst Preis und Rückfrage", r1["ok"] and r1["result"]["rueckfrage_noetig"] and r1["result"]["preis"] == 10, r1)
m, daten = datei_zum_token(r2["anhang"]["download_url"]) if r2.get("ok") else ({}, b"")
check("nach dem Ja: CSV als Download-Knopf, 10 Credits gebucht", r2["ok"] and r2["anhang"]["label"] == "csv" and daten[:1] and gebucht_seit(e0) == 10,
      (r2, gebucht_seit(e0)))
bis = (r2.get("anhang") or {}).get("gueltig_bis") or ""
check("Download ohne Ablage trägt gueltig_bis (etwa 24 Stunden), auch im Ergebnis fürs Modell (Prüfung 4)",
      bool(re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", bis)) and r2["result"].get("gueltig_bis") == bis
      and 23 * 3600 < (datetime.datetime.strptime(bis, "%Y-%m-%dT%H:%M:%SZ") - datetime.datetime.utcnow()).total_seconds() <= 24 * 3600, (bis, r2.get("result")))
e0 = letzte()
r1, r2 = zwei(PID, "exportiere_alt_texte", {"format": "xlsx"})
m, daten = datei_zum_token(r2["anhang"]["download_url"]) if r2.get("ok") else ({}, b"")
check("Alt-Texte Excel für alle Dokumente: ZIP, 10 Credits", r2["ok"] and r2["anhang"]["label"] == "zip" and daten[:2] == b"PK" and gebucht_seit(e0) == 10,
      (r2, gebucht_seit(e0)))
e0 = letzte()
r1, r2 = zwei(PID, "exportiere_quickinfos", {"document_id": ANT})
m, daten = datei_zum_token(r2["anhang"]["download_url"]) if r2.get("ok") else ({}, b"")
check("Quickinfos CSV: Rückfrage, dann Feldliste, 10 Credits", r1["result"]["rueckfrage_noetig"] and r2["ok"]
      and "Nummer;Name;Quickinfo" in daten.decode("utf-8", "replace") and gebucht_seit(e0) == 10, (r1, r2, gebucht_seit(e0)))
e0 = letzte()
# 0 Credits: keine Rueckfrage, gleich die Datei (Pruefung 3, N6)
r2 = ex(PID).execute("exportiere_fertige_pdf", {"document_id": ROH})
m, daten = datei_zum_token(r2["result"]["download_url"]) if r2.get("ok") and r2["result"].get("download_url") else ({}, b"")
check("PDF ohne Tags über den Chat: ohne Rückfrage, unverändert, 0 Credits, Download-Knopf (keine Ablage)", r2["ok"]
      and not r2["result"].get("rueckfrage_noetig") and r2["result"]["ausgabe_id"] is None
      and r2["result"]["unveraendert_ohne_tags"] == "1" and daten[:5] == b"%PDF-" and gebucht_seit(e0) == 0, r2)
r2 = ex(PID).execute("exportiere_fertige_pdf", {"alle": True})
m, daten = datei_zum_token(r2["result"]["download_url"]) if r2.get("ok") and r2["result"].get("download_url") else ({}, b"")
check("Alle Dokumente als ZIP über den Chat (0 Credits, ohne Rückfrage)", r2["ok"] and r2["anhang"]["label"] == "zip" and daten[:2] == b"PK", r2)
r2 = ex(PID).execute("exportiere_fertige_pdf", {"document_id": ACT})
aid = (r2.get("result") or {}).get("ausgabe_id")
check("getaggte PDF: Eintrag in der Ablage, Link ohne Frist (kein gueltig_bis)", r2["ok"] and aid and not (r2.get("anhang") or {}).get("gueltig_bis"), r2)

print("== Ablage: ausgabe_loeschen ==")
r1, r2 = zwei(PID, "ausgabe_loeschen", {"ausgabe_id": aid})
check("Löschen: erst Rückfrage mit Karte (Servertext), nach dem Ja weg", r1["result"].get("rueckfrage_noetig")
      and (r1.get("anhang") or {}).get("art") == "bestaetigung" and r2["ok"] and r2["result"]["geloescht"]
      and not sql("SELECT 1 FROM ablage WHERE id = ?", aid), (r1, r2))
r = ex(PID).execute("ausgabe_loeschen", {"ausgabe_id": 999999999, "bestaetigt": True})
check("fremder/unbekannter Eintrag: Fehler", not r["ok"], r)

print("== Einstellungen: ki_kontext_setzen, eigener_prompt, stammdaten_anwenden ==")
r = ex(PID).execute("ki_kontext_setzen", {"an": False})
check("KI-Kontext aus", r["ok"] and sql("SELECT use_context FROM projects WHERE id = ?", PID)[0][0] == 0, r)
r = ex(PID).execute("ki_kontext_setzen", {"an": True})
check("KI-Kontext an", r["ok"] and sql("SELECT use_context FROM projects WHERE id = ?", PID)[0][0] == 1, r)
r = ex(PID).execute("eigener_prompt", {"auflisten": True})
check("Prompts auflisten", r["ok"] and isinstance(r["result"]["prompts"], list), r)
r = ex(PID).execute("eigener_prompt", {"prompt_id": 0})
check("kein eigener Prompt", r["ok"] and r["result"]["prompt_id"] is None, r)
r = ex(PID).execute("stammdaten_anwenden", {})
check("Stammdaten anwenden (kostenlos)", r["ok"] and isinstance(r["result"]["uebernommen"], int), r)

print("== Sammelläufe: alt_texte_generieren, quickinfos_generieren ==")
bild = sql("SELECT id FROM images WHERE document_id = ? ORDER BY id LIMIT 1", ROH)
e0 = letzte()
r1, r2 = zwei(PID, "alt_texte_generieren", {"document_id": ROH})
check("Alt-Texte generieren: erst Anzahl und Preis", r1["ok"] and r1["result"].get("rueckfrage_noetig") and r1["result"].get("bilder") == len(bild), r1)
check("nach dem Ja: Lauf gestartet", r2["ok"] and r2["result"].get("gestartet"), r2)
z = []
for _ in range(120):
    time.sleep(2)
    z = sql("SELECT status, alt_text, image_type FROM images WHERE id = ?", bild[0][0]) if bild else []
    if z and z[0][0] in ("done", "error") and sql("SELECT status FROM projects WHERE id = ?", PID)[0][0] != "processing":
        break
st_bild = tuple(z[0]) if z else None
check("der Lauf hat das Bild beschrieben (fertig; Text oder als Schmuckbild erkannt), gebucht wie der Knopf (5)",
      st_bild and st_bild[0] == "done" and ((st_bild[1] or "").strip() or st_bild[2] == "dekorativ") and gebucht_seit(e0) == 5,
      (st_bild, gebucht_seit(e0)))
e0 = letzte()
r1, r2 = zwei(PID, "quickinfos_generieren", {"document_id": ANT})
check("Quickinfos generieren: erst Felder und Preis", r1["ok"] and r1["result"].get("rueckfrage_noetig") and r1["result"].get("felder", 0) > 0, r1)
check("nach dem Ja: Lauf gestartet", r2["ok"] and r2["result"].get("gestartet"), r2)
for _ in range(150):
    time.sleep(2)
    if sql("SELECT status FROM projects WHERE id = ?", PID)[0][0] != "processing":
        break
neu = sql("SELECT COUNT(*) FROM formularfelder WHERE document_id = ? AND COALESCE(quickinfo, '') <> ''", ANT)[0][0]
check("Quickinfos geschrieben und gebucht (1 je Feld)", neu > 0 and gebucht_seit(e0) > 0, (neu, gebucht_seit(e0)))

print("== Ausgeblendet: im Chatbot nicht erreichbar ==")
for w in ("pruefung_starten", "pruefbericht_lesen", "korrektur_anwenden", "korrektur_rueckgaengig", "komplett_barrierefrei_machen", "revert_alt_text"):
    r = ex(PID).execute(w, {"document_id": ACT, "image_id": (bild[0][0] if bild else 1), "bestaetigt": True})
    check(f"{w}: unbekanntes Werkzeug", not r["ok"] and "Unbekannt" in r.get("error", ""), r)

print("== Word-Projekt ==")
r = ex(WPID, True).execute("dokument_umbenennen", {"document_id": WDOC, "name": "Word-Probe (Test)"})
check("Word: umbenennen", r["ok"] and sql("SELECT display_name FROM documents WHERE id = ?", WDOC)[0][0] == "Word-Probe (Test)", r)
r = ex(WPID, True).execute("dokument_loeschen", {"document_id": WDOC})
check("Word: löschen fragt erst (nichts gelöscht)", r["ok"] and r["result"].get("rueckfrage_noetig") and sql("SELECT 1 FROM documents WHERE id = ?", WDOC), r)
r = ex(WPID, True).execute("alt_sprache_setzen", {"sprache": "en"})
r2 = ex(WPID, True).execute("alt_sprache_setzen", {"sprache": "de"})
check("Word: Sprache der Alt-Texte setzen", r["ok"] and r2["ok"] and r2["result"]["vorher"] == "en", (r, r2))
e0 = letzte()
r1, r2 = zwei(WPID, "exportiere_alt_texte", {"format": "json"}, word=True)
m, daten = datei_zum_token(r2["anhang"]["download_url"]) if r2.get("ok") else ({}, b"")
check("Word: Alt-Texte als JSON, 10 Credits", r2["ok"] and r2["anhang"]["label"] == "json" and daten[:1] in (b"{", b"[") and gebucht_seit(e0) == 10,
      (r2, gebucht_seit(e0)))
r = ex(WPID, True).execute("korrektur_anwenden", {"document_id": WDOC})
check("Word: ausgeblendetes Werkzeug nicht erreichbar", not r["ok"], r)

schleife.call_soon_threadsafe(schleife.stop)
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
