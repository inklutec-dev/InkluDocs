#!/usr/bin/env python3
"""Setup/Abbau eines API-Schluessels fuer die E2E-Probe (im Container). Aufruf:
   python3 /tmp/setup_api_v1_key.py <mail> auf   -> gibt den rohen Schluessel aus
   python3 /tmp/setup_api_v1_key.py <mail> ab    -> loescht alle Schluessel namens e2e-api-v1"""
import os
import sys
sys.path.insert(0, "/app"); os.chdir("/app")
from database import create_api_key, delete_api_key, get_db, get_user_by_email, list_api_keys  # noqa: E402
NAME = "e2e-api-v1 (fiktiv)"
u = get_user_by_email(sys.argv[1])
if not u:
    print("Konto fehlt", file=sys.stderr); sys.exit(2)
if sys.argv[2] == "auf":
    kid, roh = create_api_key(u["id"], NAME)
    print(roh)
else:
    n = 0
    for k in list_api_keys(u["id"]):
        if k.get("name") == NAME:
            delete_api_key(u["id"], k["id"]); n += 1
    print("geloescht:", n)
