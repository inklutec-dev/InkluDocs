#!/usr/bin/env python3
"""Pruefung 3 (Entwicklung N1), IM Staging-Container: Bestaetigungs-Karte und Endpunkt POST /api/projects/{id}/chat/bestaetigen
im selben Prozess wie die Angebote (FastAPI-TestClient gegen main.app, Angebote ueber den ToolExecutor wie im Chat).
  - zwei Ablage-Eintraege, zwei Loesch-Angebote (Karte je Angebot mit dem Text des Servers)
  - „Ja“ zum aelteren Angebot wird abgelehnt (nur das letzte gilt)
  - Knopf der Karte zum aelteren Angebot loescht GENAU diesen Eintrag, der andere bleibt; Verlauf speichert beides
  - dieselbe Karte zweimal: 404; Kennung in einem anderen Projekt: 404
Aufruf (im Container): python3 /tmp/bestaetigung_probe.py <projekt> <user> <dok1> <dok2> <anderes-projekt-des-nutzers>"""
import sys

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
check("Karte je Angebot: Text und Knopf vom Server", k1.get("art") == "bestaetigung" and "nicht rückgängig" in k1.get("text", "")
      and k1.get("knopf", "").endswith("bestätigen") and k1.get("angebot_id") != k2.get("angebot_id"), (k1, k2))

r = ex().execute("ausgabe_loeschen", {"ausgabe_id": e1, "bestaetigt": True})
check("getipptes Ja zum älteren Angebot: abgelehnt, nichts gelöscht",
      (r.get("result") or {}).get("rueckfrage_noetig") and eintrag_da(e1) and eintrag_da(e2), r)

vorher = len(storage.get_history(PID))
a = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": k1["angebot_id"]})
j = a.json() if a.headers.get("content-type", "").startswith("application/json") else {}
check("Knopf der Karte zu Eintrag 1: genau dieser gelöscht, Eintrag 2 bleibt",
      a.status_code == 200 and j.get("ok") and not eintrag_da(e1) and eintrag_da(e2) and "Bestätigt und ausgeführt" in j.get("reply", ""),
      (a.status_code, j, eintrag_da(e1), eintrag_da(e2)))
verlauf = storage.get_history(PID)[vorher:]
check("Verlauf: Bestätigung und Ergebnis gespeichert", len(verlauf) == 2 and verlauf[0]["role"] == "user" and "Knopf" in verlauf[0]["content"]
      and verlauf[1]["werkzeuge"] == ["ausgabe_loeschen"], verlauf)
a = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": k1["angebot_id"]})
check("dieselbe Karte ein zweites Mal: 404 mit Text", a.status_code == 404 and "gilt nicht mehr" in a.json().get("detail", ""), (a.status_code, a.text))
a = client.post(f"/api/projects/{ANDERES}/chat/bestaetigen", json={"angebot_id": k2["angebot_id"]})
check("Angebot aus Projekt A über Projekt B: 404, nichts gelöscht", a.status_code == 404 and eintrag_da(e2), (a.status_code, a.text))
a = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": "0" * 32})
check("erfundene Kennung: 404", a.status_code == 404, a.status_code)
a = client.post(f"/api/projects/{PID}/chat/bestaetigen", json={"angebot_id": k2["angebot_id"]})
check("Karte zu Eintrag 2: gelöscht", a.status_code == 200 and not eintrag_da(e2), (a.status_code, a.text))
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
