# InkluDocs Deployment-Übersicht

Stand: 12.05.2026 (Option-C-Refactor)

## Zwei separate Container

| Container | URL | Image | ENV-File | Compose-File |
|-----------|-----|-------|----------|--------------|
| inkludocs (Production) | https://inkludocs.inklutec.de | inkludocs:v4-20260512 (fest) | `.env.prod` | `docker-compose.yml` |
| inkludocs-staging | https://staging.inkludocs.inklutec.de | build aus Workspace | `.env.staging` | `docker-compose.staging.yml` |

## Bedienung (Wrapper-Skripte)

```bash
# Production
./compose-prod.sh up -d              # starten
./compose-prod.sh logs -f             # Logs verfolgen
./compose-prod.sh restart             # neu starten
./compose-prod.sh down                # stoppen + entfernen

# Staging
./compose-staging.sh up -d --build    # starten + neu bauen
./compose-staging.sh logs -f
./compose-staging.sh restart
./compose-staging.sh down
```

Die Wrapper sorgen automatisch dafür, dass die richtige `.env`-Datei geladen wird.

## ENV-Files

- `.env.prod` — wirkt nur auf Production. chmod 600.
- `.env.staging` — wirkt nur auf Staging. chmod 600. **Hier darf experimentiert werden** ohne Production zu beeinflussen.
- Bei Änderungen: jeweiligen Container restart.

## Image-Updates (Staging → Production)

```bash
# 1. Neues Staging-Image bauen + testen
./compose-staging.sh up -d --build

# 2. Wenn auf Staging stabil: Image-Tag setzen
sudo docker tag $(sudo docker inspect inkludocs-staging --format {{.Image}} | sed s/sha256://) inkludocs:v4-NEUDATUM

# 3. In docker-compose.yml (Production) den image:-Wert auf den neuen Tag setzen

# 4. Production restart
./compose-prod.sh up -d
```

## Rollback

Alter Production-Image-Tag: `inkludocs:rollback-pre-v4-20260512-090603`

Im Notfall:
```bash
# docker-compose.yml image: zurueck auf rollback-Tag setzen
sed -i s/inkludocs:v4-20260512/inkludocs:rollback-pre-v4-20260512-090603/ docker-compose.yml
./compose-prod.sh up -d
```

## Backups

- DB-Snapshots: `/opt/inkludocs-backups/db/` (täglich 03:00)
- Pre-Migration: `/opt/inkludocs-backups/pre-v4-migration/`
- ENV-Backups: `.env.bak-*` neben den aktiven Files
