"""„Alt-Texte generieren" nimmt IMMER alle Bilder (Michael Karbe, 02.09.2026).

Im Staging-Container:
    docker cp tests/e2e/verify_doppelkosten.py inkludocs-staging:/tmp/ && \
    docker exec inkludocs-staging python3 /tmp/verify_doppelkosten.py

Geschichte: Vom 31.08. bis 02.09.2026 nahm der naechste Start nach einem Abbruch
nur die offen gebliebenen Bilder (Rest-Vermerk, gegen doppelte Kosten). Michael
Karbe hat das am 01.09. verworfen: „Die Funktion Alt-Texte generieren erzeugt
immer neue Alt-Texte fuer alle Bilder." Dass ein erneuter Sammellauf bereits
generierte Bilder noch einmal kostet, ist seither die GEWOLLTE Folge — die
Rueckfrage nennt Anzahl und Preis vorher; fuer einzelne Bilder gibt es den
Knopf „Neu generieren" am Bild.

Geprueft wird weiterhin der Lauf-Hinweis: Endet ein Lauf vorzeitig (Guthaben,
Tageslimit, Abbruch), muss der Statusabruf Grund und Zahlen liefern — daraus
baut die Oberflaeche die sichtbare Statusmeldung.

Kein Bedrock-Aufruf: die Generierung wird durch einen Platzhalter ersetzt. Es
wird ein eigenes Testprojekt angelegt, kein echtes Projekt angefasst.

SICHERHEITSBREMSE: laeuft nur im Staging-Container — das Skript drueckt das
Guthaben eines echten Kontos voruebergehend auf 0 und darf nie gegen
Produktion laufen.
"""
import asyncio, json, os, sqlite3, sys

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

    # --- Testprojekt: 4 fertige Bilder mit KI-Text (alle Kandidaten fuer den Sammellauf)
    cur = con.execute(
        "INSERT INTO projects (user_id, filename, original_path, status, total_images, processed_images, "
        "project_type, name, tool) VALUES (?,?,?,?,?,?,?,?,?)",
        (uid, "doppelkosten-test.pdf", "/tmp/doppelkosten-test.pdf", "done", 4, 4, "pdf",
         "Doppelkosten-Test", "pdf"))
    PID = cur.lastrowid
    # alt_text_edited = None (nie von Hand angefasst) — die Fixture bildet echte Daten nach.
    for i in range(4):
        c2 = con.execute("INSERT INTO images (project_id, page_number, image_index, image_path, alt_text, "
                         "alt_text_edited, image_type, status) VALUES (?,?,?,?,?,?,?,?)",
                         (PID, 1, i, f"/tmp/dk-{i}.png", f"Alter Text {i}", None, "foto", "done"))
        BILD_IDS.append(c2.lastrowid)
    con.commit()
    check("Testprojekt mit 4 Kandidaten angelegt", len(BILD_IDS) == 4, BILD_IDS)

    # --- 1. Lauf mit Guthaben fuer genau ZWEI Bilder -> endet vorzeitig nach zweien
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
    check("KEIN Rest-Vermerk mehr (abgeloest 02.09.2026)", status_von("ki_neu_rest") is None,
          status_von("ki_neu_rest"))
    check("Erster Lauf hat zwei Buchungen erzeugt", buchungen() == 2, buchungen())

    # --- 2. Lauf-Hinweis: Grund und Zahlen stehen am Projekt und kommen im Statusabruf an
    hinweis = json.loads(status_von("lauf_hinweis") or "null")
    check("Lauf-Hinweis nennt den Grund 'credits'", (hinweis or {}).get("grund") == "credits", hinweis)
    check("Lauf-Hinweis: zwei erledigt, zwei offen",
          (hinweis or {}).get("erledigt") == 2 and (hinweis or {}).get("offen") == 2, hinweis)
    antwort = asyncio.run(main.get_project_status(PID, user=nutzer))
    check("Statusabruf liefert den Hinweis an die Oberflaeche",
          (antwort.get("lauf_hinweis") or {}).get("offen") == 2, antwort)

    # --- 3. Zweiter Start nimmt WIEDER ALLE VIER (Michaels Regel — das ist der Kern)
    guthaben_auf(500)
    v = asyncio.run(main.generate_vorschau(PID, FakeRequest({"modus": "alle"}), user=nutzer))
    check("Vorschau nach vorzeitigem Ende nennt wieder ALLE vier Bilder", v.get("anzahl") == 4, v)
    check("Vorschau kennt keinen Rest mehr (Schluessel 'rest'/'gesamt' weg)",
          "rest" not in v and "gesamt" not in v, sorted(v.keys()))
    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    main._process_project = echt_process
    check("Zweiter Start nimmt ALLE vier Bilder (nicht nur den Rest)",
          uebergeben["ids"] == set(BILD_IDS), (uebergeben, offen))
    check("Alter Hinweis ist beim Neustart weggeraeumt", status_von("lauf_hinweis") is None,
          status_von("lauf_hinweis"))

    main.generate_alt_text = ki_platzhalter
    asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=set(BILD_IDS)))
    main.generate_alt_text = echt_gen

    check("Jetzt tragen alle vier Bilder den neuen Text",
          all(t.startswith("NEU") for t in texte().values()), texte())
    check("GEWOLLTE Doppelkosten: 2 + 4 = sechs Buchungen (zwei Bilder zweimal bezahlt)",
          buchungen() == 6, buchungen())
    check("Vollstaendiger Lauf setzt keinen Hinweis", status_von("lauf_hinweis") is None,
          status_von("lauf_hinweis"))

    # --- 4. Auch direkt nach einem vollen Lauf: naechster Klick nimmt wieder alle
    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({"modus": "ki_neu"}), user=nutzer))
    main._process_project = echt_process
    check("Naechster Klick nimmt wieder alle vier", uebergeben["ids"] == set(BILD_IDS), uebergeben)
    stub_lauf_zuruecksetzen()

    # --- 5. Ein Modus: auch ein nie generiertes (pending) Bild ist Kandidat
    con.execute("UPDATE images SET status='pending' WHERE id=?", (BILD_IDS[0],))
    con.commit()
    main._process_project = stub_task
    asyncio.run(main.generate_alt_texts(PID, FakeRequest({}), user=nutzer))
    main._process_project = echt_process
    check("Ohne modus: Rettungsmenge = alle vier (ein Modus seit 01.09.)", uebergeben["ids"] == set(BILD_IDS), uebergeben)
    # Direkter Aufruf des Laufs OHNE Rettungsmenge (wie ein Lauf vor dem Umbau): das offene Bild bleibt pending.
    guthaben_auf(0)
    main.generate_alt_text = ki_platzhalter
    asyncio.run(main._process_project(PID, uid, force=False, ki_neu_ids=set()))
    main.generate_alt_text = echt_gen
    check("Lauf ohne Rettungsmenge: das offene Bild bleibt pending, kein Vermerk",
          con.execute("SELECT status FROM images WHERE id=?", (BILD_IDS[0],)).fetchone()["status"] == "pending"
          and status_von("ki_neu_rest") is None, status_von("ki_neu_rest"))
    check("Lauf ohne Rettungsmenge: der Hinweis wird trotzdem gesetzt (der Nutzer soll den Grund hoeren)",
          json.loads(status_von("lauf_hinweis") or "null") is not None, status_von("lauf_hinweis"))

    # --- 6. Abbruch durch den Nutzer (Michael Karbe 01.09.2026): Signal vor jedem Bild, geordnetes Ende
    con.execute("UPDATE images SET status='pending' WHERE project_id=?", (PID,))
    con.execute("UPDATE projects SET status='processing', ki_neu_rest=NULL, lauf_hinweis=NULL WHERE id=?", (PID,))
    con.commit()
    guthaben_auf(500)
    vor_buchungen = buchungen()

    def abbruch_platzhalter(*a, **k):
        main._abbruch_gewuenscht.add(PID)        # nach dem ERSTEN Bild abbrechen
        return ki_platzhalter(*a, **k)

    main.generate_alt_text = abbruch_platzhalter
    asyncio.run(main._process_project(PID, uid, force=True, ki_neu_ids=set(BILD_IDS)))
    main.generate_alt_text = echt_gen
    hinweis = json.loads(status_von("lauf_hinweis") or "null") or {}
    check("Abbruch: Hinweis nennt den Grund 'abbruch'", hinweis.get("grund") == "abbruch", hinweis)
    check("Abbruch: ein Bild erledigt, drei offen", hinweis.get("erledigt") == 1 and hinweis.get("offen") == 3, hinweis)
    check("Abbruch: genau eine Buchung (nur das bearbeitete Bild kostet)", buchungen() - vor_buchungen == 1, buchungen() - vor_buchungen)
    check("Abbruch: alle vier Bilder wieder auf done (nichts haengt)",
          [r["status"] for r in con.execute("SELECT status FROM images WHERE project_id=? ORDER BY id", (PID,))] == ["done"] * 4)
    check("Abbruch: KEIN Rest-Vermerk geschrieben (abgeloest 02.09.2026)", status_von("ki_neu_rest") is None,
          status_von("ki_neu_rest"))
    v_nach = asyncio.run(main.generate_vorschau(PID, FakeRequest({"modus": "alle"}), user=nutzer))
    check("Vorschau nach Abbruch nennt wieder ALLE vier (Michaels Regel)", v_nach.get("anzahl") == 4, v_nach)
    check("Abbruch: Projekt wieder auf done, Signal geloescht",
          status_von("status") == "done" and PID not in main._abbruch_gewuenscht, (status_von("status"), PID in main._abbruch_gewuenscht))
    con.execute("UPDATE projects SET lauf_hinweis=NULL WHERE id=?", (PID,)); con.commit()

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
