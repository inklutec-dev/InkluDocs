#!/usr/bin/env python3
"""Setup/Abbau fuer ui_filter_stand.py (im Container ausfuehren, 17.09.2026).

Aufruf:  python3 /tmp/setup_ui_filter_stand.py <projekt-id> auf | ab
'auf' stellt das erste fertige Bild des Projekts auf status 'error' und merkt sich
den alten Stand in /tmp/ui_filter_stand.json; 'ab' stellt ihn wieder her.
Nur Staging, nur das eigene Testprojekt."""
import json
import os
import sys

sys.path.insert(0, "/app")
os.chdir("/app")
from database import get_db  # noqa: E402

MERK = "/tmp/ui_filter_stand.json"
pid = int(sys.argv[1])
was = sys.argv[2]
conn = get_db()
try:
    if was == "auf":
        row = conn.execute("SELECT id, status FROM images WHERE project_id = ? AND status = 'done' ORDER BY id LIMIT 1",
                           (pid,)).fetchone()
        if not row:
            print("kein fertiges Bild im Projekt", pid); sys.exit(2)
        json.dump({"id": row["id"], "status": row["status"]}, open(MERK, "w"))
        conn.execute("UPDATE images SET status = 'error' WHERE id = ?", (row["id"],))
        conn.commit()
        print(row["id"])
    else:
        if not os.path.exists(MERK):
            print("nichts zu restaurieren"); sys.exit(0)
        m = json.load(open(MERK))
        conn.execute("UPDATE images SET status = ? WHERE id = ?", (m["status"], m["id"]))
        conn.commit()
        os.remove(MERK)
        print("restauriert", m["id"])
finally:
    conn.close()
