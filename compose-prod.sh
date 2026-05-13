#!/bin/bash
# InkluDocs Production-Compose-Wrapper
# Benutzt: ./compose-prod.sh up -d   ./compose-prod.sh logs -f   ./compose-prod.sh down
set -e
cd "$(dirname "$0")"
if [ ! -f .env.prod ]; then
    echo "FEHLER: .env.prod nicht gefunden in $(pwd)" >&2
    exit 1
fi
exec docker compose -f docker-compose.yml --env-file .env.prod "$@"
