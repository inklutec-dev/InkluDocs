#!/bin/bash
# Quickinfo-Werkzeug: Unit-Tests im Container + End-to-End + Klicktest (28.08.2026).
# Aufruf aus dem Repo-Wurzelverzeichnis:  bash tests/e2e/formular_tests.sh
# Braucht in der Umgebung: INKLUDOCS_E2E_MAIL, INKLUDOCS_E2E_PW (Testkonto auf Staging),
# optional INKLUDOCS_E2E_URL (Standard Staging), INKLUDOCS_E2E_CONTAINER (Standard inkludocs-staging),
# INKLUDOCS_E2E_PYTHON_PW (Python mit Playwright, Standard python3). Keine Geheimnisse im Repo.
set -u
D="$(cd "$(dirname "$0")/../.." && pwd)"
C="${INKLUDOCS_E2E_CONTAINER:-inkludocs-staging}"
URL="${INKLUDOCS_E2E_URL:-https://staging.inkludocs.inklutec.de}"
PYPW="${INKLUDOCS_E2E_PYTHON_PW:-python3}"
: "${INKLUDOCS_E2E_MAIL:?INKLUDOCS_E2E_MAIL fehlt}"; : "${INKLUDOCS_E2E_PW:?INKLUDOCS_E2E_PW fehlt}"
echo "=== unit (Container $C)"
sudo docker exec "$C" mkdir -p /app/tests/fixtures
sudo docker cp "$D/tests/test_formular_roundtrip.py" "$C":/app/tests/ >/dev/null; sudo docker cp "$D/tests/test_formular_ki.py" "$C":/app/tests/ >/dev/null; sudo docker cp "$D/tests/test_billing_export.py" "$C":/app/tests/ >/dev/null
sudo docker cp "$D/tests/fixtures/testformular_inkludocs.pdf" "$C":/app/tests/fixtures/ >/dev/null
sudo docker exec -w /app "$C" python3 -m unittest /app/tests/test_formular_roundtrip.py /app/tests/test_formular_ki.py /app/tests/test_billing_export.py 2>&1 | grep -E "^Ran|^OK|FAILED|Error"
echo "=== verify_formular (E2E, Projekt bleibt fuer Klicktest)"
OUT=$(python3 "$D/tests/e2e/verify_formular.py" "$URL" "$INKLUDOCS_E2E_MAIL" "$INKLUDOCS_E2E_PW" "$D/tests/fixtures/testformular_inkludocs.pdf" --behalten 2>&1)
echo "$OUT" | grep -E "FEHLT|Ergebnis|Traceback|Error" | head -20
PID=$(echo "$OUT" | grep -oE "Projekt [0-9]+\)" | grep -oE "[0-9]+" | tail -1)
TOKEN=$(echo "$OUT" | grep -oE "^Gast-Token: .*" | awk '{print $2}')
if [ -n "$PID" ]; then
  echo "=== ui_formular (Klicktest, Projekt $PID)"
  "$PYPW" "$D/tests/e2e/ui_formular.py" "$PID" "$TOKEN" 2>&1 | grep -E "FEHLT|Ergebnis"
  python3 - "$PID" "$URL" "$INKLUDOCS_E2E_MAIL" "$INKLUDOCS_E2E_PW" <<'PY2'
import http.cookiejar, json, sys, urllib.request
pid, B, mail, pw = sys.argv[1:5]; cj=http.cookiejar.CookieJar(); op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
r=urllib.request.Request(B+"/api/login", data=json.dumps({"email":mail,"password":pw}).encode(), headers={"Content-Type":"application/json"}); op.open(r).read()
print("  Testprojekt geloescht:", op.open(urllib.request.Request(f"{B}/api/projects/{pid}", method="DELETE")).status)
for e in json.loads(op.open(B+"/api/stammdaten").read())["stammdaten"]:
    if e["beschriftung"] in ("Vorname","Nachname","Geburtsdatum (TT.MM.JJJJ) *") or e["feld_name"]=="anrede":
        op.open(urllib.request.Request(f"{B}/api/stammdaten/{e['id']}", method="DELETE"))
PY2
fi
