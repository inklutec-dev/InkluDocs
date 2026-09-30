#!/usr/bin/env python3
"""Pruefung 3 (Entwicklung N1), IM Staging-Container: Bestaetigungs-Karte und Endpunkt POST /api/projects/{id}/chat/bestaetigen
im selben Prozess wie die Angebote (FastAPI-TestClient gegen main.app, Angebote ueber den ToolExecutor wie im Chat).
  - zwei Ablage-Eintraege, zwei Loesch-Angebote (Karte je Angebot mit dem Text des Servers)
  - „Ja“ zum aelteren Angebot wird abgelehnt (nur das letzte gilt)
  - Knopf der Karte zum aelteren Angebot loescht GENAU diesen Eintrag, der andere bleibt; Verlauf speichert beides
  - dieselbe Karte zweimal: 409 „Schon bestätigt.“; Kennung in einem anderen Projekt: 404
  - Pruefung 4: Karten-Zustand im gespeicherten Verlauf („Erledigt“ / offen), Antwort „Erledigt: …“ ohne Angebotstext,
    Aktionen (Karte, Ansicht nachziehen); Doppelklick gleichzeitig: genau einmal ausgefuehrt, nichts Abgelehntes im Verlauf
Aufruf (im Container): python3 /tmp/bestaetigung_probe.py <projekt> <user> <dok1> <dok2> <anderes-projekt-des-nutzers>"""
import sys
import threading

sys.path.insert(0, "/app")
PID, UID, D1, D2, ANDERES = map(int, sys.argv[1:6])
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from inkluagent import storage  # noqa: E402
from inkluagent.tools import ausgaben  # noqa: E402
from inkluagent.tools.definitions import ToolExecutor  # noqa: E402

ok = fehler = 0


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK ", n)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:500])


def ex():
    return ToolExecutor(project_id=PID, user_id=UID, pdf=True)


def eintrag_da(aid):
    c = main.get_db()
    try:
        return c.execute("SELECT 1 FROM ablage WHERE id = ?", (aid,)).fetchone() is not None
    finally:
        c.close()


u = main.get_user_by_id(UID)
client = TestClient(main.app)
client.cookies.set("token", main.create_token(UID, u["email"], u.get("is_admin") or 0))

r1 = ex().execute("exportiere_fertige_pdf", {"document_id": D1})
r2 = ex().execute("exportiere_fertige_pdf", {"document_id": D2})
e1, e2 = (r1.get("result") or {}).get("ausgabe_id"), (r2.get("result") or {}).get("ausgabe_id")
check("0-Credit-Download ohne Rückfrage, zwei Ablage-Einträge", r1.get("ok") and r2.get("ok") and e1 and e2 and e1 != e2
      and not (r1.get("result") or {}).get("rueckfrage_noetig"), (r1, r2))
if not (e1 and e2 and e1 != e2):
    print(f"Ergebnis: {ok} OK, {fehler} FEHLER (abgebrochen: zwei verschiedene Ablage-Einträge nötig)")
    sys.exit(1)

k1 = ex().execute("ausgabe_loeschen", {"ausgabe_id": e1}).get("anhang") or {}
k2 = ex().execute("ausgabe_loeschen", {"ausgabe_id": e2}).get("anhang") or {}
check("Karte je Angebot: Text und Knopf vom Server, Ziel im Knopf", k1.get("art") == "bestaetigung" and "nicht rückgängig" in k1.get("text", "")
      and k1.get("knopf", "").endswith("bestätigen") and "„" in k1.get("knopf", "") and k1.get("angebot_id") != k2.get("angebot_id"), (k1, k2))

r = ex().execute("ausgabe_loeschen", {"ausgabe_id": e1, "bestaetigt": True})
check("getipptes Ja zum älteren Angebot: abgelehnt, nichts gelöscht",
      (r.get("result") or {}).get("rueckfrage_noetig") and eintrag_da(e1) and eintrag_da(e2), r)

# die Karten stehen wie im echten Chat unter einer gespeicherten Antwort (fiktiv)
storage.append_message(PID, "assistant", "Soll ich die Einträge löschen? (Probe)", werkzeuge=["ausgabe_loeschen"], anhang=[k1, k2])
vorher = len(storage.get_history(PID))
ergebnis = {}


def klick(i):
    r = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": k1["angebot_id"]})
    ergebnis[i] = (r.status_code, r.json())


th = [threading.Thread(target=klick, args=(i,)) for i in range(2)]
[t.start() for t in th]
[t.join() for t in th]
codes = sorted(v[0] for v in ergebnis.values())
j = next((v[1] for v in ergebnis.values() if v[0] == 200), {})
nein = next((v[1] for v in ergebnis.values() if v[0] != 200), {})
check("Doppelklick gleichzeitig: einmal ausgeführt (200), einmal 409 „Schon bestätigt.“",
      codes == [200, 409] and nein.get("detail") == "Schon bestätigt.", ergebnis)
check("Knopf der Karte zu Eintrag 1: genau dieser gelöscht, Eintrag 2 bleibt", j.get("ok") and not eintrag_da(e1) and eintrag_da(e2),
      (j, eintrag_da(e1), eintrag_da(e2)))
check("Antwort „Erledigt: …“ ohne Angebotstext, Karte „Erledigt“, Aktionen Karte + Ansicht nachziehen",
      j.get("reply", "").startswith("Erledigt:") and "rückgängig" not in j.get("reply", "")
      and (j.get("karte") or {}).get("zustand") == "erledigt"
      and {a.get("type") for a in j.get("actions") or []} == {"karte", "ansicht_aktualisieren"}, j)
verlauf = storage.get_history(PID)[vorher:]
check("Verlauf: genau Bestätigung und Ergebnis (der abgelehnte Doppelklick steht nicht drin)",
      len(verlauf) == 2 and verlauf[0]["role"] == "user" and "Knopf" in verlauf[0]["content"]
      and verlauf[1]["werkzeuge"] == ["ausgabe_loeschen"], verlauf)
h = client.get(f"/api/projects/{PID}/chat/history").json()["messages"]
karten = {a["angebot_id"]: a for m in h for a in (m.get("anhang") or []) if a.get("art") == "bestaetigung"}
check("gespeicherter Verlauf: Karte 1 „Erledigt“, Karte 2 noch offen",
      (karten.get(k1["angebot_id"]) or {}).get("zustand") == "erledigt" and (karten.get(k1["angebot_id"]) or {}).get("titel") == "Erledigt"
      and (karten.get(k2["angebot_id"]) or {}).get("zustand") == "offen", karten)
a = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": k1["angebot_id"]})
check("dieselbe Karte später noch einmal: 409 „Schon bestätigt.“ mit Karte „Erledigt“",
      a.status_code == 409 and a.json().get("detail") == "Schon bestätigt." and (a.json().get("karte") or {}).get("zustand") == "erledigt",
      (a.status_code, a.text))
check("… und nichts Neues im Verlauf", len(storage.get_history(PID)) == vorher + 2, len(storage.get_history(PID)) - vorher)
a = client.post(f"/api/projects/{ANDERES}/chat/bestaetigen", json={"angebot_id": k2["angebot_id"]})
check("Angebot aus Projekt A über Projekt B: 404, nichts gelöscht", a.status_code == 404 and eintrag_da(e2), (a.status_code, a.text))
a = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": "0" * 32})
check("erfundene Kennung: 404", a.status_code == 404, a.status_code)
a = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": k2["angebot_id"]})
check("Karte zu Eintrag 2: gelöscht", a.status_code == 200 and not eintrag_da(e2), (a.status_code, a.text))
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
