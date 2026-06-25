# InkluDocs — Demo-Instanz

Stand: 13.06.2026 (Schritt 1 — Instanz aufgesetzt)

## Was ist das?

Die Demo-Instanz ist eine öffentliche **Kostprobe von InkluDocs ohne Anmeldung**.
Ein Besucher lädt eine einzelne Grafik hoch und bekommt in Sekunden einen
KI-Alt-Text plus Langbeschreibung — in derselben Qualität wie das echte Werkzeug.
Den Text kann er ansehen, von Hand bearbeiten, per InkluAgent verfeinern und
kopieren. Der eigentliche Produktwert — der fertig getaggte, exportierbare
PDF/Excel-Output — bleibt registrierten Nutzern vorbehalten. So überzeugt die
Demo von der Qualität, ohne das Produkt zu verschenken.

Öffentliche Adresse (nach Freischaltung, Schritt 6): `https://demo.inkludocs.de`

## Warum eine eigene Instanz?

Sie läuft als **dritter, vollständig isolierter Container** neben Production und
Staging — gleicher Code, aber strikt getrennt:

- **Eigene Wegwerf-Datenbank** (Volume `inkludocs_demo_data`). Anonymer Traffic und
  temporäre Demo-Projekte berühren nie die echten Nutzerdaten der Produktion.
- **Eigener Port** `127.0.0.1:8003` (Production 8001, Staging 8002, Preview 8004).
- **Eigene Geheimnis-Datei** `.env.demo` (chmod 600, nicht in Git).

Geteilt wird bewusst nur die **Intelligenz**: dieselbe Generierungs-Pipeline und
derselbe InkluAgent (Code aus dem Workspace, `build: .`). Jede künftige
Pipeline-Verbesserung wirkt damit automatisch auch in der Demo. Getrennt sind
dagegen Daten, Limits, der öffentliche Endpunkt und die Demo-Bot-Leitplanke.

## Der Schalter `DEMO_MODE`

`DEMO_MODE` ist **pro Instanz** fest gesetzt, nicht pro Anfrage:

- Demo-Container: `DEMO_MODE=on`
- Production / Staging: `DEMO_MODE=off` (bzw. nicht gesetzt)

Welche Instanz ein Besucher erreicht, entscheidet der Nginx-Proxy-Manager am
Hostnamen (`demo.inkludocs.de` → Demo-Container). Der Demo-Pfad kann deshalb
niemals in die Produktion „durchsickern": Demo-spezifischer Code ist an
`DEMO_MODE` gebunden, und die Daten liegen in getrennten Datenbanken.

Ab Schritt 2/3 schaltet `DEMO_MODE`:
- den öffentlichen, anmeldefreien Demo-Endpunkt frei (Wegwerf-Projekt pro Sitzung),
- die server-seitigen Limits (siehe unten),
- einen kleinen, nur angehängten Leitplanken-Absatz im System-Prompt des Agenten
  (hält ihn beim Thema Alt-Text/Barrierefreiheit; verhindert Missbrauch als
  allgemeiner Web-Assistent). Der Standard-Prompt des Agenten bleibt unverändert.

## Limits (server-seitig, NICHT im Prompt)

Die Grenzen werden im Code des Demo-Endpunkts durchgesetzt, **bevor** ein
Modell-Aufruf passiert — niemals vom Bot selbst. Konfigurierbar über `.env.demo`:

- `DEMO_DAILY_IMAGE_LIMIT` (Standard 3) — Generierungen pro Tag pro Besucher.
- `DEMO_DAILY_CHAT_LIMIT` (Standard 12) — Chat-Nachrichten pro Tag pro Besucher.
- `DEMO_GLOBAL_DAILY_LIMIT` (Standard 300) — globaler Tagesdeckel als Kostenschutz.

Zähler liegen in der Wegwerf-DB der Demo (pro Besucher über IP/Browser-Token,
plus ein globaler Tageszähler) und werden täglich zurückgesetzt.

## Bedienung

```bash
cd /home/coder/projects/InkluDocs
./compose-demo.sh up -d --build     # starten / neu bauen
./compose-demo.sh logs -f           # Logs verfolgen
./compose-demo.sh restart           # neu starten
./compose-demo.sh down              # stoppen + entfernen
```

Der Wrapper lädt automatisch `.env.demo`.

## Status der Schritte

1. **Instanz aufsetzen** — erledigt (dieser Stand). Container `inkludocs-demo` auf
   `127.0.0.1:8003`, eigene leere DB, läuft „dunkel" (keine DNS-/NPM-Anbindung).
2. Öffentlicher Demo-Endpunkt ohne Login (Wegwerf-Projekt pro Sitzung, Bild-Löschung nach Analyse).
3. Limit-Schicht + thematische Bot-Leitplanke.
4. Demo-Seite (Frontend, Firmenfarben, landet direkt in der Maske).
5. Dunkel verifizieren (Limits, DSGVO-Löschung, Barrierefreiheit, Kostendeckel).
6. Freischalten (DNS `demo.inkludocs.de`, NPM-Host + Zertifikat).
