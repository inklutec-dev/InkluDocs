#!/usr/bin/env python3
"""Migration „Texte einmal je Projekt“ (05.10.2026, docs/ANSICHTEN_LEISTUNG.md, Logik in projekt_texte.py).

Bestandsdaten: KI-Kontext (images.context_text) und Seitentext (images.page_text) stehen je Bild kopiert in der
Datenbank. Die Migration legt jeden Text EINMAL je Projekt in projekt_texte ab, setzt die Verweise und leert danach die
alten Spalten — nur dort, wo der Verweis nachweislich byte-gleich denselben Text liefert.

Aufruf im Container (Beispiel Staging):
  docker exec -w /app inkludocs-staging python3 scripts/texte_migration.py            # Probe: zaehlt nur
  docker exec -w /app inkludocs-staging python3 scripts/texte_migration.py --alles    # Sicherung, A, Pruefung, B, Bericht
Einzelschritte: --sicherung, --phase-a, --pruefen, --phase-b, --vacuum, --zurueck (Rueckweg: alte Spalten wieder fuellen).
Optionen: --db PFAD (Standard $INKLUDOCS_DB bzw. /app/data/inkludocs.db), --ohne-sicherung (nur Tests).

Sicherheit:
  - Vor Phase A und vor --zurueck wird IMMER eine Sicherung angelegt (sqlite3-Backup-API, konsistent auch bei WAL und
    laufendem Betrieb) und mit PRAGMA integrity_check geprueft; ohne gueltige Sicherung bricht das Skript ab.
  - Phase A ist idempotent (nur Bilder ohne Verweis), je Projekt eine kurze Transaktion.
  - Phase B laeuft nur nach einer Pruefung ohne Befund und leert nur, was gleich gespeichert ist (Bedingung im UPDATE).
  - --alles vergleicht den Fingerabdruck (SHA-256 des wirksamen Textes je Bild) vor A und nach B: muss gleich sein.
Exit-Code 0 = in Ordnung, 1 = Abbruch/Befund.
"""
import argparse
import json
import os
import sqlite3
import sys
import time

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HIER))   # /app bzw. backend/
import projekt_texte  # noqa: E402


def verbinden(pfad: str):
    conn = sqlite3.connect(pfad, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def sicherung(pfad: str) -> str:
    ziel = f"{pfad}.bak-pre-texte-{time.strftime('%Y%m%d-%H%M%S')}"
    t0 = time.time()
    src = sqlite3.connect(pfad, timeout=30)
    dst = sqlite3.connect(ziel)
    src.backup(dst)
    dst.close()
    src.close()
    chk = sqlite3.connect(ziel)
    ergebnis = chk.execute("PRAGMA integrity_check").fetchone()[0]
    bilder = chk.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    chk.close()
    if ergebnis != "ok":
        raise SystemExit(f"ABBRUCH: Sicherung {ziel} ist nicht in Ordnung: {ergebnis}")
    print(f"Sicherung: {ziel} ({os.path.getsize(ziel) / 1e6:.1f} MB, {bilder} Bilder, integrity_check ok, {time.time() - t0:.1f} s)")
    return ziel


def bericht(conn, titel: str) -> dict:
    p = projekt_texte.probe(conn)
    print(f"--- {titel}")
    for spalte in ("context_text", "page_text"):
        e = p[spalte]
        print(f"  {spalte}: {e['bilder_ohne_verweis']} Bilder ohne Verweis ({e['zeichen']:,} Zeichen) -> "
              f"{e['verschiedene_texte']} Texte ({e['zeichen_nachher']:,} Zeichen); mit Verweis noch gefuellt: {e['mit_verweis_noch_gefuellt']}")
    print(f"  projekt_texte: {p['projekt_texte'][0]} Texte, {p['projekt_texte'][1]:,} Zeichen")
    return p


def pruefung_ok(conn) -> bool:
    """Befund = Verweis liefert anderen Text, zeigt ins Leere oder in ein fremdes Projekt. Bilder ohne Verweis (noch nicht
    migriert oder waehrend der Migration von altem Code angelegt) sind KEIN Befund: Phase B laesst sie unangetastet, sie
    bleiben ueber die alte Spalte lesbar (Befund 2 der Pruefung Entwicklung 05.10.2026)."""
    t0 = time.time()
    p = projekt_texte.pruefen(conn)
    ok = p["abweichend"] == 0 and p["ins_leere"] == 0 and p["fremdes_projekt"] == 0
    print(f"Pruefung ({time.time() - t0:.1f} s): {json.dumps(p)} -> {'OK' if ok else 'BEFUND'}")
    if ok and p["noch_ohne_verweis"]:
        print(f"  Hinweis: {p['noch_ohne_verweis']} Bild(er) ohne Verweis — kein Datenverlust, sie behalten ihre alte Spalte. "
              "Schreibt noch ein Container mit ALTEM Code auf diese Datenbank? Regel: erst neuer Code, dann Migration; "
              "ein weiterer Lauf von --alles nimmt sie mit.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("INKLUDOCS_DB", "/app/data/inkludocs.db"))
    ap.add_argument("--alles", action="store_true")
    ap.add_argument("--sicherung", action="store_true")
    ap.add_argument("--phase-a", action="store_true")
    ap.add_argument("--pruefen", action="store_true")
    ap.add_argument("--phase-b", action="store_true")
    ap.add_argument("--vacuum", action="store_true")
    ap.add_argument("--zurueck", action="store_true")
    ap.add_argument("--ohne-sicherung", action="store_true", help="nur fuer Tests auf Wegwerf-Datenbanken")
    a = ap.parse_args()
    if not os.path.isfile(a.db):
        raise SystemExit(f"Datenbank nicht gefunden: {a.db}")
    groesse_vorher = os.path.getsize(a.db)
    conn = verbinden(a.db)
    projekt_texte.schema_anlegen(conn)   # idempotent (falls der Container mit neuem Code noch nicht gestartet war)
    spalten = {r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
    if not {"kontext_id", "seitentext_id"} <= spalten:
        raise SystemExit("ABBRUCH: images.kontext_id/seitentext_id fehlen — erst den Container mit dem neuen Code starten (init_db).")
    conn.commit()
    nichts = not (a.alles or a.sicherung or a.phase_a or a.pruefen or a.phase_b or a.vacuum or a.zurueck)
    print(f"Datenbank: {a.db} ({groesse_vorher / 1e6:.1f} MB)")
    bericht(conn, "Stand vorher")
    if nichts:
        print("Nur Probe — nichts geaendert. Ausfuehren mit --alles (oder Einzelschritten).")
        return 0
    if a.zurueck:
        if not a.ohne_sicherung:
            sicherung(a.db)
        print("Rueckweg:", projekt_texte.zurueck(conn))
        bericht(conn, "Stand nach dem Rueckweg")
        return 0 if pruefung_ok(conn) else 1
    if (a.alles or a.sicherung or a.phase_a) and not a.ohne_sicherung:
        sicherung(a.db)
    fingerabdruck_vorher = projekt_texte.fingerabdruck(conn) if a.alles else None
    if a.alles or a.phase_a:
        t0 = time.time()
        erg = projekt_texte.phase_a(conn)
        print(f"Phase A ({time.time() - t0:.1f} s): {erg}")
    if a.alles or a.pruefen or a.phase_b:
        if not pruefung_ok(conn):
            print("ABBRUCH vor Phase B: Pruefung mit Befund. Alte Spalten bleiben gefuellt, nichts verloren.")
            return 1
    if a.alles or a.phase_b:
        t0 = time.time()
        erg = projekt_texte.phase_b(conn)
        print(f"Phase B ({time.time() - t0:.1f} s): geleert {erg}")
        if not pruefung_ok(conn):
            return 1
    if fingerabdruck_vorher is not None:
        nachher = projekt_texte.fingerabdruck(conn)
        # Nur Bilder, die vorher UND nachher existieren: im laufenden Betrieb geloeschte oder neu angelegte Bilder sind
        # keine Abweichung (Befund 2 der Pruefung Entwicklung 05.10.2026).
        v = projekt_texte.fingerabdruck_vergleich(fingerabdruck_vorher, nachher)
        abw, weg, neu = v["abweichend"], v["geloescht"], v["neu"]
        print(f"Fingerabdruck (SHA-256 des wirksamen Textes je Bild): {v['verglichen']} Bilder verglichen, {len(abw)} Abweichungen"
              + (f" — z. B. {abw[:5]}" if abw else "")
              + (f"; waehrend des Laufs geloescht: {weg}, neu: {neu}" if (weg or neu) else ""))
        if abw:
            return 1
    if a.vacuum:
        t0 = time.time()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        print(f"VACUUM ({time.time() - t0:.1f} s)")
    bericht(conn, "Stand nachher")
    conn.close()
    print(f"Datei: {groesse_vorher / 1e6:.1f} MB -> {os.path.getsize(a.db) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
