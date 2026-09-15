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

## Dritte Instanz: Demo (seit 13.06.2026)

Neben Production und Staging laeuft eine Demo-Instanz als oeffentliche Kostprobe
ohne Anmeldung. Sie ist vollstaendig isoliert: eigener Container `inkludocs-demo`
auf `127.0.0.1:8003`, eigene Wegwerf-Datenbank (Volume `inkludocs_demo_data`),
eigene Geheimnis-Datei `.env.demo` (chmod 600). Gesteuert ueber `DEMO_MODE=on`.

- Compose-Datei: `docker-compose.demo.yml`
- Env-File: `.env.demo` (generiert aus der laufenden Production-Instanz; nicht in Git)
- Wrapper: `./compose-demo.sh up -d --build`
- Oeffentliche Adresse nach Freischaltung: `https://demo.inkludocs.de` (noch nicht verbunden)

Architektur, der DEMO_MODE-Schalter und die Limits sind in `DEMO.md` dokumentiert.

Hinweis (13.06.2026): Im Repo lag bei Einrichtung KEINE `.env.prod` mehr — die
laufende Production-Instanz traegt ihre Konfiguration im Container. `compose-prod.sh up`
wuerde daher mangels `.env.prod` fehlschlagen, bis die Datei wiederhergestellt ist.
(Separat zu pruefen, nicht Teil der Demo.)

## Logs und Zeitzone (15.09.2026)

Alle drei App-Dienste (`inkludocs`, `inkludocs-demo`, `inkludocs-staging`) loggen ins System-Journal
(`logging: driver: journald`, Tag = Containername) und laufen mit `TZ=Europe/Berlin`.

- Verlauf über Container-Neubauten hinweg: `sudo journalctl CONTAINER_NAME=inkludocs --since "2026-09-14 15:00" -o short-iso`
- Laufender Container wie bisher: `sudo docker logs --since 1h inkludocs`
- Die Datenbank speichert weiter UTC (SQLite `datetime(now)`). Log- und Mail-Zeiten sind Berliner Zeit.
  Beim Abgleich Datenbank gegen Log: Sommer +2 h, Winter +1 h.
- Anlass: Am 14.09. gingen mit sechs Rollouts die Logs des Tages verloren (21× HTTP 429 nicht mehr prüfbar),
  und Uhrzeiten wurden im Bericht um zwei Stunden falsch notiert.
