"""Fortsetzen eines abgebrochenen „n neu generieren" kostet nicht doppelt (Steve 31.08.2026).

Im Staging-Container:
    docker cp tests/e2e/verify_doppelkosten.py inkludocs-staging:/tmp/ && \
    docker exec inkludocs-staging python3 /tmp/verify_doppelkosten.py

Befund des Pruefers vom 30.08.2026: Bricht der Lauf ab, weil das Guthaben nicht reicht, und
startet der Nutzer nach dem Aufstocken erneut, liefen die bereits frisch generierten Bilder ein
zweites Mal durch die KI — und kosteten ein zweites Mal. Seit den Aktionspreisen (5 Credits je
Alt-Text) faellt das ins Gewicht. Seit dem 31.08. merkt sich das Projekt den Rest.

Geprueft wird ausserdem der Lauf-Hinweis: Endet ein Lauf vorzeitig, muss der Statusabruf sagen,
warum und wie viele Bilder offen blieben — vorher stand das nur im Server-Log und die Oberflaeche
meldete „Alle Alt-Texte wurden generiert."

Kein Bedrock-Aufruf: die Generierung wird durch einen Platzhalter ersetzt. Es wird ein eigenes
Testprojekt angelegt, kein echtes Projekt angefasst.

SICHERHEITSBREMSE: laeuft nur im Staging-Container — das Skript drueckt das Guthaben eines
echten Kontos voruebergehend auf 0 und darf nie gegen Produktion laufen.
"""
import asyncio, json, os, sqlite3, sys
from datetime import datetime, timedelta

MAIL = "steve.weidel@gmail.com"
DB = "/app/data/inkludocs.db"
ok = fehler = 0

if os.environ.get("SCAN_ERLAUBE_LOOPBACK") != "1":
    sys.exit("ABBRUCH: Das ist kein Staging-Container (SCAN_ERLAUBE_LOOPBACK != 1). "
             "Dieses Skript veraendert Guthaben und darf nur auf Staging laufen.")


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", str(info)[:300])


sys.path.insert(0, "/app")
import billing  # noqa: E402
import main  # noqa: E402

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
uid = con.execute("SELECT id FROM users WHERE email=?", (MAIL,)).fetchone()["id"]
check("Enforcement ist an (sonst sagt die Wache immer ja)", billing.ABO_ENFORCEMENT is True,
      billing.ABO_ENFORCEMENT)

AKTION = "test_doppelkosten"
PID = None
BILD_IDS = []
echt_process = main._process_project
echt_gen = main.generate_alt_text
PREIS = billing.aktion_pruefung(uid, "bild_generierung")["preis"]


def guthaben_auf(ziel):
    """Synthetische Verbrauchszeilen, bis genau `ziel` Credits uebrig sind. Raeumt NUR die
    eigenen Zeilen weg (Aktion `test_doppelkosten`) — nie fremde Abrechnungszeilen."""
    con.execute("DELETE FROM usage_events WHERE konto_user_id=? AND aktion=?", (uid, AKTION))
    con.commit()
    rest = billing.verfuegbare_credits(uid)
    if rest is not None and rest > ziel:
        con.execute("INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits) "
                    "VALUES (?,?,?,?,?)", (uid, uid, "einzeln", AKTION, rest - ziel))
        con.commit()
    return billing.verfuegbare_credits(uid)


class FakeRequest:
    def __init__(self, daten):
        self._daten = daten

    async def json(self):
        return self._daten


def texte():
    return {r["id"]: (r["alt_text"] or "") for r in
            con.execute("SELECT id, alt_text FROM images WHERE project_id=? ORDER BY id", (PID,))}


def status_von(feld):
    return con.execute(f"SELECT {feld} FROM projects WHERE id=?", (PID,)).fetchone()[feld]


def vermerk():
    roh = status_von("ki_neu_rest")
    return set(json.loads(roh)["ids"]) if roh else set()


def buchungen():
    """Verbrauchszeilen, die auf unsere Testbilder gebucht wurden."""
    marken = ",".join("?" * len(BILD_IDS))
    return con.execute(f"SELECT COUNT(*) FROM usage_events WHERE image_id IN ({marken})",
                       BILD_IDS).fetchone()[0]


def stub_lauf_zuruecksetzen():
    """Nach einem abgefangenen Start aufraeumen: Der Platzhalter hat den Lauf nie ausgefuehrt,
    also stehen die Bilder noch auf 'pending' und das Projekt auf 'processing' — der naechste
    Start liefe sonst in 409 „Verarbeitung laeuft bereits"."""
    main._ki_neu_zurueck(con, set(BILD_IDS))
    con.execute("UPDATE projects SET status='done' WHERE id=?", (PID,))
    con.commit()


def ki_platzhalter(*a, **k):
    """Ersetzt die echte Generierung — jeder Aufruf liefert denselben neuen Text."""
    return {"bildtyp": "foto", "alt_text": "NEU generierter Testtext", "konfidenz": "hoch",
            "langbeschreibung": "", "needs_review": False, "pipeline_steps": "",
            "validation_result": "", "from_cache": False}


try:
    nutzer = dict(con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())
    check("Tageslimit steht dem Test nicht im Weg", main.tageslimit_wache(nutzer) is None,
          main.tageslimit_wache(nutzer))

    # --- Testprojekt: 4 fertige Bilder mit KI-Text (alle Kandidaten fuer „neu generieren")
    cur = con.execute(
        "INSERT INTO projects (user_id, filename, original_path, status, total_images, processed_images, "
        "project_type, name, tool) VALUES (?,?,?,?,?,?,?,?,?)",
        (uid, "doppelkosten-test.pdf", "/tmp/doppelkosten-test.pdf", "done", 4, 4, "pdf",
         "Doppelkosten-Test", "pdf"))
    PID = cur.lastrowid
    for i in range(4):
        c2 = con.execute("INSERT INTO images (project_id, page_number, image_index, image_path, alt_text, "
                         "alt_text_edited, image_type, status) VALUES (?,?,?,?,?,?,?,?)",
                         (PID, 1, i, f"/tmp/dk-{i}.png", f"Alter Text {i}", "", "foto", "done"))
        BILD_IDS.append(c2.lastrowid)
    con.commit()
    check("Testprojekt mit 4 Kandidaten angelegt", len(BILD_IDS) == 4, BILD_IDS)

    # --- 1. Lauf mit Guthaben fuer genau ZWEI Bilder -> Abbruch nach zweien
    guthaben_auf(2 * PREIS)
    uebergeben = {}

    async def _nichts():
        return None

    def stub_task(project_id, user_id, force=False, document_id=None, ki_neu_ids=None):
        uebergeben.clear(); uebergeben.update({"ids": set(ki_neu_ids or ())})
        return _nichts()

    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    main._process_project = echt_process
    check("Erster Start nimmt alle vier Bilder", uebergeben["ids"] == set(BILD_IDS), uebergeben)

    main.generate_alt_text = ki_platzhalter
    asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=set(BILD_IDS)))
    main.generate_alt_text = echt_gen

    neu = [i for i, t in texte().items() if t.startswith("NEU")]
    offen = [i for i, t in texte().items() if not t.startswith("NEU")]
    check("Genau zwei Bilder wurden neu generiert (Guthaben reichte fuer zwei)", len(neu) == 2, texte())
    check("Zwei Bilder blieben unveraendert — alter Text steht noch", len(offen) == 2, texte())
    check("Alle vier stehen wieder auf 'done' (nichts haengt auf pending)",
          [r["status"] for r in con.execute("SELECT status FROM images WHERE project_id=?", (PID,))]
          == ["done"] * 4)
    check("REST-VERMERK enthaelt genau die zwei offen gebliebenen Bilder", vermerk() == set(offen),
          (vermerk(), offen))
    check("Erster Lauf hat zwei Buchungen erzeugt", buchungen() == 2, buchungen())

    # --- 2. Lauf-Hinweis: Grund und Zahlen stehen am Projekt und kommen im Statusabruf an
    hinweis = json.loads(status_von("lauf_hinweis") or "null")
    check("Lauf-Hinweis nennt den Grund 'credits'", (hinweis or {}).get("grund") == "credits", hinweis)
    check("Lauf-Hinweis: zwei erledigt, zwei offen",
          (hinweis or {}).get("erledigt") == 2 and (hinweis or {}).get("offen") == 2, hinweis)
    antwort = asyncio.run(main.get_project_status(PID, user=nutzer))
    check("Statusabruf liefert den Hinweis an die Oberflaeche",
          (antwort.get("lauf_hinweis") or {}).get("offen") == 2, antwort)

    # --- 3. Zweiter Start nimmt NUR den Rest (das ist der eigentliche Fix)
    guthaben_auf(500)
    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    main._process_project = echt_process
    check("Zweiter Start nimmt NUR die zwei offen gebliebenen Bilder",
          uebergeben["ids"] == set(offen), (uebergeben, offen))
    check("Alter Hinweis ist beim Neustart weggeraeumt", status_von("lauf_hinweis") is None,
          status_von("lauf_hinweis"))

    main.generate_alt_text = ki_platzhalter
    asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=set(offen)))
    main.generate_alt_text = echt_gen

    check("Jetzt tragen alle vier Bilder den neuen Text",
          all(t.startswith("NEU") for t in texte().values()), texte())
    check("KEINE DOPPELTEN KOSTEN: vier Bilder, vier Buchungen", buchungen() == 4, buchungen())
    check("Vollstaendiger Lauf loescht den Rest-Vermerk", status_von("ki_neu_rest") is None,
          status_von("ki_neu_rest"))
    check("Vollstaendiger Lauf setzt keinen Hinweis", status_von("lauf_hinweis") is None,
          status_von("lauf_hinweis"))

    # --- 4. Nach dem vollstaendigen Lauf nimmt der naechste Klick wieder alle
    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    main._process_project = echt_process
    check("Ohne Vermerk nimmt der naechste Klick wieder alle vier", uebergeben["ids"] == set(BILD_IDS),
          uebergeben)
    stub_lauf_zuruecksetzen()

    # --- 5. Randfaelle des Vermerks: Verfall, Unsinn, fremde IDs
    alt = (datetime.now() - timedelta(hours=main.KI_NEU_REST_STUNDEN + 1)).isoformat(timespec="seconds")
    con.execute("UPDATE projects SET ki_neu_rest=? WHERE id=?",
                (json.dumps({"ids": BILD_IDS[:1], "ts": alt}), PID))
    con.commit()
    check("Vermerk aelter als die Frist wird ignoriert", main._ki_neu_rest_lesen(con, PID) == set(),
          main._ki_neu_rest_lesen(con, PID))

    con.execute("UPDATE projects SET ki_neu_rest='{kaputt' WHERE id=?", (PID,))
    con.commit()
    check("Unlesbarer Vermerk fuehrt zum normalen Verhalten, nicht zum Absturz",
          main._ki_neu_rest_lesen(con, PID) == set())

    con.execute("UPDATE projects SET ki_neu_rest=? WHERE id=?",
                (json.dumps({"ids": [BILD_IDS[0], 999999999], "ts": datetime.now().isoformat(timespec="seconds")}), PID))
    con.commit()
    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    main._process_project = echt_process
    check("Fremde ID im Vermerk erreicht kein fremdes Bild (Schnittmenge greift)",
          uebergeben["ids"] == {BILD_IDS[0]}, uebergeben)
    stub_lauf_zuruecksetzen()

    # --- 5b. Mehrere Dokumente (Pruefbefund Fable 5, 31.08.2026): ein Lauf in "Dokument B"
    #        darf den Rest von "Dokument A" weder ueberschreiben noch loeschen
    a_rest, b_ids = set(BILD_IDS[:2]), set(BILD_IDS[2:])
    con.execute("UPDATE projects SET ki_neu_rest=? WHERE id=?",
                (json.dumps({"ids": sorted(a_rest), "ts": datetime.now().isoformat(timespec="seconds")}), PID))
    con.execute("UPDATE images SET status='pending' WHERE id IN (?,?)", tuple(b_ids))
    con.execute("UPDATE projects SET status='processing' WHERE id=?", (PID,))
    con.commit()
    guthaben_auf(0)                      # B bricht sofort ab -> beide B-Bilder offen
    main.generate_alt_text = ki_platzhalter
    asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=b_ids))
    main.generate_alt_text = echt_gen
    check("Abbruch in B: Vermerk enthaelt A UND B (fortgeschrieben, nicht ueberschrieben)",
          vermerk() == a_rest | b_ids, (vermerk(), a_rest, b_ids))
    guthaben_auf(500)                    # B laeuft jetzt vollstaendig durch
    con.execute("UPDATE images SET status='pending' WHERE id IN (?,?)", tuple(b_ids))
    con.execute("UPDATE projects SET status='processing' WHERE id=?", (PID,))
    con.commit()
    main.generate_alt_text = ki_platzhalter
    asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=b_ids))
    main.generate_alt_text = echt_gen
    check("B vollstaendig: nur B aus dem Vermerk genommen, A bleibt stehen", vermerk() == a_rest,
          (vermerk(), a_rest))
    con.execute("UPDATE projects SET ki_neu_rest=NULL WHERE id=?", (PID,)); con.commit()

    # --- 6. Luecken-Modus fuehrt keinen Vermerk (dort bleibt der Rest zu Recht pending)
    con.execute("UPDATE images SET status='pending' WHERE id=?", (BILD_IDS[0],))
    con.execute("UPDATE projects SET ki_neu_rest=NULL WHERE id=?", (PID,))
    con.commit()
    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({}), user=nutzer))
    main._process_project = echt_process
    check("Luecken-Modus uebergibt keine Rettungsmenge", uebergeben["ids"] == set(), uebergeben)
    guthaben_auf(0)
    main.generate_alt_text = ki_platzhalter
    asyncio.run(main._process_project(PID, uid, force=False, ki_neu_ids=set()))
    main.generate_alt_text = echt_gen
    check("Luecken-Modus: das offene Bild bleibt pending, kein Vermerk",
          con.execute("SELECT status FROM images WHERE id=?", (BILD_IDS[0],)).fetchone()["status"] == "pending"
          and status_von("ki_neu_rest") is None, status_von("ki_neu_rest"))
    check("Luecken-Modus: der Hinweis wird trotzdem gesetzt (der Nutzer soll den Grund hoeren)",
          json.loads(status_von("lauf_hinweis") or "null") is not None, status_von("lauf_hinweis"))

finally:
    main._process_project = echt_process
    main.generate_alt_text = echt_gen
    if BILD_IDS:
        marken = ",".join("?" * len(BILD_IDS))
        con.execute(f"DELETE FROM usage_events WHERE image_id IN ({marken})", BILD_IDS)
    if PID:
        con.execute("DELETE FROM images WHERE project_id=?", (PID,))
        con.execute("DELETE FROM projects WHERE id=?", (PID,))
    con.execute("DELETE FROM usage_events WHERE konto_user_id=? AND aktion=?", (uid, AKTION))
    con.commit()
    rest_pr = con.execute("SELECT COUNT(*) FROM projects WHERE id=?", (PID,)).fetchone()[0] if PID else 0
    rest_ev = con.execute("SELECT COUNT(*) FROM usage_events WHERE konto_user_id=? AND aktion=?",
                          (uid, AKTION)).fetchone()[0]
    check("Aufgeraeumt: Testprojekt und eigene Verbrauchszeilen sind weg", rest_pr == 0 and rest_ev == 0,
          (rest_pr, rest_ev))

print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
