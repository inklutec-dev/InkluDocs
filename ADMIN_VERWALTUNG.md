# Administrator-Verwaltung (Stand 14.06.2026)

Selbstverwaltung von Administratoren direkt in der InkluDocs-Oberfläche, mit
zwei Rechte-Stufen. Ersetzt das frühere manuelle Setzen von `is_admin` in der
Datenbank.

## Rechte-Stufen (Spalte `users.admin_level`)

- **`full` — Voll-Admin:** alle Rechte wie der Gründer-Account — Nutzer sehen,
  sperren/entsperren, Daten löschen (DSGVO), Passwörter zurücksetzen, neue
  Konten anlegen **und andere zu Admins machen / Stufen ändern / Rechte entziehen**.
- **`view` — Nur-Einsicht:** sieht die Nutzerliste und wer sich angemeldet hat,
  **darf aber nichts verändern** (kein Sperren, Löschen, Zurücksetzen, kein
  Admin-Hinzufügen). Gedacht z. B. für Vertriebspartner, die nur den
  Anmeldestand verfolgen sollen.

Bestehende Admins werden bei der Migration auf `full` gesetzt (Default), behalten
also ihre vollen Rechte.

## Oberfläche: `/benutzer` (Benutzerverwaltung)

Sichtbar nur für Admins. Neu auf der Seite:

- **Abschnitt „Administratoren"** — Liste aller Admins mit Name, E-Mail und Stufe.
  Pro Eintrag (nur für Voll-Admins, nicht beim eigenen Konto): Knopf
  „Auf Nur-Einsicht setzen" bzw. „Auf Voll-Admin setzen" und „Admin-Rechte entziehen".
- **Abschnitt „Admin hinzufügen"** — E-Mail eines **bestehenden** Kontos eingeben,
  Stufe per `fieldset`/`radio` wählen (Voll-Admin / Nur-Einsicht), „Zum Admin machen".
  Hat die E-Mail noch kein Konto, weist die Meldung darauf hin, zuerst unter
  „Neuen Benutzer anlegen" ein Konto zu erstellen.

Für **Nur-Einsicht-Admins** werden „Neuen Benutzer anlegen", „Admin hinzufügen"
und alle verändernden Knöpfe ausgeblendet (und serverseitig zusätzlich blockiert).

Barrierefreiheit: natives HTML (echte `form`/`label`/`fieldset`/`legend`/`button`),
ARIA nur wo nötig (`role="status"` für Meldungen). axe: keine neuen Verstöße.

## API-Endpunkte

- `GET  /api/admin/admins` — Admin-Liste (jeder Admin, auch Nur-Einsicht).
- `POST /api/admin/admins` — Konto zum Admin machen `{email, level}` (nur Voll-Admin).
- `PUT  /api/admin/admins/{id}` — Stufe ändern `{level}` (nur Voll-Admin).
- `DELETE /api/admin/admins/{id}` — Admin-Rechte entziehen (nur Voll-Admin).

Außerdem brauchen jetzt **alle verändernden** Admin-Aktionen (Nutzer sperren,
Passwort zurücksetzen, Konto anlegen, Konto löschen) die Stufe `full`.

## Sicherungen

- Niemand kann sich **selbst** die Admin-Rechte entziehen.
- Der **letzte Voll-Admin** kann weder entfernt noch auf „Nur-Einsicht" gesetzt
  werden (Schutz gegen Aussperren).
- `require_full_admin` liest die Stufe **frisch aus der Datenbank** — ein
  entzogenes/herabgestuftes Recht greift sofort, nicht erst nach Token-Ablauf.

## Technik

- DB-Migration: `ALTER TABLE users ADD COLUMN admin_level TEXT DEFAULT 'full'`
  (in `database.py` → `_migrate_columns`, läuft automatisch beim Start).
- `database.py`: `list_admins()`, `count_full_admins()`, `set_user_admin()`.
- `main.py`: `require_full_admin()`, `admin_level` in `/api/me`, die vier
  Verwaltungs-Endpunkte, Stufen-Schutz auf den verändernden Endpunkten.
- `frontend/benutzer.html`: Abschnitte + JS.

## Hinweis Login-Token

Wer neu zum Admin gemacht wird, muss sich **einmal ab- und wieder anmelden**,
damit das Admin-Recht im Login-Token landet (lesender Admin-Zugriff hängt am Token).

## Stand

Auf **Staging** gebaut und end-to-end getestet (Login, Liste, Hinzufügen,
Stufenwechsel, Entziehen, Validierung, Sicherungen). **Production-Promote
ausstehend** — wartet auf Abnahme.
