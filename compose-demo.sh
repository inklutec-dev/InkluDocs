#!/bin/bash
# InkluDocs Demo-Compose-Wrapper
# Fährt die isolierte Demo-Instanz (öffentliche Kostprobe ohne Anmeldung).
# Benutzt: ./compose-demo.sh up -d --build   ./compose-demo.sh logs -f   ./compose-demo.sh down
#
# Der Wrapper lädt automatisch .env.demo (eigene Geheimnis-Datei, chmod 600,
# nicht in Git). Aufbau analog zu compose-prod.sh / compose-staging.sh.
# Hintergrund + Architektur: siehe DEMO.md
set -e
cd "$(dirname "$0")"
if [ ! -f .env.demo ]; then
    echo "FEHLER: .env.demo nicht gefunden in $(pwd)" >&2
    exit 1
fi
exec docker compose -f docker-compose.demo.yml --env-file .env.demo "$@"
