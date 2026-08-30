"""„n neu generieren" bricht sauber ab (Steve 30.08.2026) — im Staging-Container:
    docker cp tests/e2e/verify_ki_neu_abbruch.py inkludocs-staging:/tmp/ && \
    docker exec inkludocs-staging python3 /tmp/verify_ki_neu_abbruch.py

Hintergrund: Der Start-Endpunkt setzt im Modus ki_neu fertige Bilder auf 'pending', damit der
vorhandene Sammellauf sie aufgreift. Reichte das Guthaben nicht, brach die Schleife am ersten
Bild ab — und die Bilder blieben 'pending', obwohl ihr Alt-Text unveraendert in der Datenbank
stand (Anzeige und Zaehler sagten „nicht generiert").

Legt ein eigenes Testprojekt mit synthetischen Bildern an, KEIN echtes Projekt wird angefasst,
kein Bedrock-Aufruf: der Lauf bricht vor jeder Generierung an der Guthaben-Wache ab.

SICHERHEITSBREMSE: Das Skript weigert sich ausserhalb von Staging zu laufen. Es drueckt das
Guthaben eines echten Kontos voruebergehend auf 0 und darf deshalb NIE gegen Produktion laufen —
der Datenbankpfad ist dort derselbe, nur der Containername unterscheidet sich."""
import asyncio, os, sqlite3, sys

MAIL = "steve.weidel@gmail.com"
DB = "/app/data/inkludocs.db"
ok = fehler = 0

# --- Bremse: SCAN_ERLAUBE_LOOPBACK ist laut docker-compose ausdruecklich NUR auf Staging gesetzt
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

AKTION = "test_ki_neu_abbruch"
PID = None
echt_process = main._process_project
echt_lauf = main._process_project_lauf


def guthaben_auf(ziel):
    """Synthetische Verbrauchszeilen, bis genau `ziel` Credits uebrig sind. Raeumt NUR die
    eigenen Zeilen weg (Aktion `test_ki_neu_abbruch`) — nie fremde Abrechnungszeilen."""
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


def zustand():
    return [(r["id"], r["status"], (r["alt_text"] or ""), r["needs_review"]) for r in
            con.execute("SELECT id, status, alt_text, needs_review FROM images "
                        "WHERE project_id=? ORDER BY id", (PID,))]


try:
    # --- Testprojekt: 2 Bilder mit Text, 1 DEKORATIVES ohne Text (auch 'done'!), 1 echte Luecke
    cur = con.execute(
        "INSERT INTO projects (user_id, filename, original_path, status, total_images, processed_images, "
        "project_type, name, tool) VALUES (?,?,?,?,?,?,?,?,?)",
        (uid, "kineu-test.pdf", "/tmp/kineu-test.pdf", "done", 4, 3, "pdf", "ki_neu-Abbruchtest", "pdf"))
    PID = cur.lastrowid
    for i in range(2):
        con.execute("INSERT INTO images (project_id, page_number, image_index, image_path, alt_text, "
                    "alt_text_edited, image_type, status) VALUES (?,?,?,?,?,?,?,?)",
                    (PID, 1, i, f"/tmp/kineu-{i}.png", f"Guter Alt-Text {i}", "", "foto", "done"))
    con.execute("INSERT INTO images (project_id, page_number, image_index, image_path, alt_text, "
                "alt_text_edited, image_type, status) VALUES (?,?,?,?,?,?,?,?)",
                (PID, 1, 5, "/tmp/kineu-deko.png", "", "", "dekorativ", "done"))
    con.execute("INSERT INTO images (project_id, page_number, image_index, image_path, alt_text, "
                "alt_text_edited, image_type, status) VALUES (?,?,?,?,?,?,?,?)",
                (PID, 1, 9, "/tmp/kineu-luecke.png", "", "", "foto", "pending"))
    con.commit()
    vorher = zustand()
    check("Testprojekt: 2x done mit Text, 1x dekorativ done OHNE Text, 1x echte Luecke",
          [s for _, s, _, _ in vorher] == ["done", "done", "done", "pending"], vorher)

    # --- 1. Ohne Guthaben: 402, und NICHTS wird zurueckgesetzt
    guthaben_auf(0)
    from fastapi import HTTPException
    nutzer = dict(con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())
    code = None
    try:
        asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    except HTTPException as e:
        code = e.status_code
    check("Ohne Guthaben: Start-Endpunkt antwortet 402 (vorher lief er einfach los)", code == 402, code)
    check("Ohne Guthaben: KEIN Bild wurde auf pending gesetzt", zustand() == vorher, zustand())
    check("Ohne Guthaben: Projekt steht nicht auf processing",
          con.execute("SELECT status FROM projects WHERE id=?", (PID,)).fetchone()["status"] == "done")

    # --- 2. Mit Guthaben: welche Bilder merkt sich der Endpunkt? (Lauf wird abgefangen)
    guthaben_auf(500)
    aufruf = {}

    async def _nichts():
        return None

    def stub(project_id, user_id, force=False, document_id=None, ki_neu_ids=None):
        aufruf.update({"pid": project_id, "force": force, "ids": set(ki_neu_ids or ())})
        return _nichts()

    main._process_project = stub
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    main._process_project = echt_process
    kandidaten = {i for i, _, _, _ in vorher[:3]}          # zwei mit Text + das dekorative
    check("Endpunkt uebergibt die Bilder namentlich an den Lauf", aufruf.get("ids") == kandidaten,
          (aufruf.get("ids"), kandidaten))
    check("Das DEKORATIVE Bild ohne Text ist dabei (Befund des Pruefers)",
          vorher[2][0] in aufruf.get("ids", set()), aufruf.get("ids"))
    check("Die echte Luecke ist NICHT dabei", vorher[3][0] not in aufruf.get("ids", set()))
    check("Genau diese drei stehen jetzt auf pending",
          [s for _, s, _, _ in zustand()] == ["pending"] * 4, zustand())

    # --- 3. Abbruch mitten im Lauf: Guthaben weg -> alles muss zurueckgestellt werden
    guthaben_auf(0)
    asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=kandidaten))
    nachher = zustand()
    check("Nach Abbruch: alle drei Kandidaten stehen wieder auf done",
          [s for _, s, _, _ in nachher[:3]] == ["done"] * 3, nachher)
    check("Nach Abbruch: auch das dekorative Bild ohne Text ist gerettet", nachher[2][1] == "done", nachher[2])
    check("Nach Abbruch: die echte Luecke bleibt zu Recht pending", nachher[3][1] == "pending", nachher[3])
    check("Nach Abbruch: kein Alt-Text veraendert",
          [t for _, _, t, _ in nachher] == [t for _, _, t, _ in vorher], (vorher, nachher))
    p = con.execute("SELECT status, processed_images FROM projects WHERE id=?", (PID,)).fetchone()
    check("Nach Abbruch: Projektstatus done (kein Dauer-409)", p["status"] == "done", dict(p))
    check("Nach Abbruch: Zaehler frisch gezaehlt = 3", p["processed_images"] == 3, dict(p))

    # --- 4. Abbruch VON AUSSEN (Container-Neustart): Notaufraeumen muss greifen
    con.execute("UPDATE images SET status='pending' WHERE id IN (%s)" % ",".join("?" * len(kandidaten)),
                list(kandidaten))
    con.execute("UPDATE projects SET status='processing' WHERE id=?", (PID,))
    con.commit()

    async def abgebrochen(*a, **k):
        raise asyncio.CancelledError()

    main._process_project_lauf = abgebrochen
    geworfen = None
    try:
        asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=kandidaten))
    except BaseException as e:
        geworfen = type(e).__name__
    main._process_project_lauf = echt_lauf
    check("Abbruch von aussen wird weitergereicht, nicht verschluckt", geworfen == "CancelledError", geworfen)
    check("Abbruch von aussen: Bilder trotzdem gerettet",
          [s for _, s, _, _ in zustand()[:3]] == ["done"] * 3, zustand())
    check("Abbruch von aussen: Projekt haengt nicht auf processing",
          con.execute("SELECT status FROM projects WHERE id=?", (PID,)).fetchone()["status"] == "done")

    # --- 5. Hilfsfunktion: nur die uebergebenen, nur pending, wiederholbar
    stand = zustand()
    main._ki_neu_zurueck(con, set())
    check("Hilfsfunktion mit leerer Menge aendert nichts", zustand() == stand)
    main._ki_neu_zurueck(con, kandidaten)
    check("Hilfsfunktion laesst fertige Bilder in Ruhe (wiederholbar)", zustand() == stand)
    check("Hilfsfunktion fasst die echte Luecke nicht an",
          con.execute("SELECT status FROM images WHERE id=?", (vorher[3][0],)).fetchone()["status"] == "pending")

    # --- 5b. Reihenfolge der Wachen: Tageslimit muss VOR dem Guthaben greifen
    alt_limit = con.execute("SELECT api_tageslimit FROM users WHERE id=?", (uid,)).fetchone()["api_tageslimit"]
    ereignis_id = None
    try:
        con.execute("UPDATE users SET api_tageslimit=1 WHERE id=?", (uid,))
        cur2 = con.execute("INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits) "
                           "VALUES (?,?,?,?,?)", (uid, uid, "einzeln", "bild_generierung", 5))
        ereignis_id = cur2.lastrowid
        con.commit()
        guthaben_auf(0)                      # BEIDE Wachen wuerden zuschlagen
        nutzer2 = dict(con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())
        code2 = None
        try:
            asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer2))
        except HTTPException as e:
            code2 = e.status_code
        check("Tageslimit schlaegt vor dem Guthaben zu (429, nicht 402)", code2 == 429, code2)
        check("Bei 429 wird ebenfalls nichts auf pending gesetzt",
              [s for _, s, _, _ in zustand()[:3]] == ["done"] * 3, zustand())
    finally:
        con.execute("UPDATE users SET api_tageslimit=? WHERE id=?", (alt_limit, uid))
        if ereignis_id:
            con.execute("DELETE FROM usage_events WHERE id=?", (ereignis_id,))
        con.commit()

    # --- 6. Luecken-Modus bleibt unberuehrt
    guthaben_auf(500)
    aufruf.clear()
    main._process_project = stub
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({}), user=nutzer))
    main._process_project = echt_process
    check("Luecken-Modus: force ist aus und die Rettungsmenge leer",
          aufruf.get("force") is False and aufruf.get("ids") == set(), aufruf)

finally:
    main._process_project = echt_process
    main._process_project_lauf = echt_lauf
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
