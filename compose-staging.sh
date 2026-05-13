#!/bin/bash
# InkluDocs Staging-Compose-Wrapper
# Benutzt: ./compose-staging.sh up -d   ./compose-staging.sh logs -f   ./compose-staging.sh down
set -e
cd "$(dirname "$0")"
if [ ! -f .env.staging ]; then
    echo "FEHLER: .env.staging nicht gefunden in $(pwd)" >&2
    exit 1
fi
exec docker compose -f docker-compose.staging.yml --env-file .env.staging "$@"
