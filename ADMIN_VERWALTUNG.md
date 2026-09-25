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


# VERWALTUNG NEU: Kunden, Umsatz, API, Einstellungen (Stand 25.09.2026)

Die frühere eine lange Seite `/benutzer` ist aufgeteilt (Steve 25.09.2026: „zu voll“, „was, wenn es
1.000 Kunden sind?“). Alte Adressen leiten weiter (`/benutzer` → `/verwaltung/kunden`,
`/benutzer/report/<id>` → `/verwaltung/kunden/<id>`). Seitenleiste: „Verwaltung“; oben auf jeder
Seite vier Links (`_verwaltung_nav.html`, aktuelle Seite mit `aria-current="page"`).

- **Kunden** `/verwaltung/kunden` — Suchfeld (Name, E-Mail, Teamname), Filter als native
  Auswahlliste (Alle, Mit Abo, Abo auf Rechnung, Abo über Stripe, Mit Käufen, Neu in diesem Monat,
  Gesperrt, Administratoren), 25 je Seite, zuletzt aktive zuerst. Je Kunde EINE Zeile mit Link —
  keine Überschrift je Kunde, keine Tabelle. Suche/Filter/Seite stehen in der Adresse; „Zurück zur
  Kundenliste“ führt genau dorthin. „Neuen Kunden anlegen“ als Klappe (nur Voll-Admins).
- **Kundenseite** `/verwaltung/kunden/<id>` (ersetzt den Report) — Überblick, Käufe und
  Gutschriften (mit „Berichtigen“), API (Schlüssel, Tageslimit, „Limit ändern“), Nutzung (Details
  aufklappbar), ganz unten „Konto verwalten“ (Sperren, Passwort, Löschen). „Credits gutschreiben“
  und „Abo zuweisen oder ändern“ sind Dialoge mit dem Kunden schon eingetragen.
- **Umsatz** `/verwaltung/umsatz` — heute, laufender Monat, Jahr, gesamt; Zeitraum (Jahr, Monat,
  Alle/Nur Verkäufe/Nur Bonus); Monate des Jahres; Buchungen einzeln; Download Excel/CSV.
- **API** `/verwaltung/api` — nur Konten mit API-Schlüssel, neuester zuerst, heute genutzt, Limit.
- **Einstellungen** `/verwaltung/einstellungen` — Administratoren, Admin hinzufügen (wie bisher).

## Gutschrift und Abo: Art ist Pflicht

Anlass: Das alte Formular bot noch die Paketgrößen von vor dem 28.08. (100/500/1000). 2.500 Credits
mussten ins Kulanz-Feld — Jens' Kauf (25.09.) landete als Geschenk, der Bonus als Rechnung.

- `POST /api/admin/users/{id}/pakete` `{groesse, art: verkauf|bonus, betrag, rechnungsnummer, notiz,
  bestaetigt_gross}` — Verkauf: Betrag Pflicht (vorbelegt: Listenpreis, freie Mengen 4 Cent/Credit),
  Paket verfällt nie. Bonus: Grund Pflicht, 12 Monate, über `umsatz.BONUS_GRENZE` (500) nur mit
  Häkchen. Paket und Buchung in EINER Transaktion; „eingetragen von“ = Name des Admins.
- `POST /api/admin/users/{id}/plan` — für Bezahl-Pläne zusätzlich `art: verkauf|ohne` mit `betrag`
  (vorbelegt Monatspreis × Laufzeit) bzw. Grund. Free bucht nichts.
- Vor dem Speichern ein Bestätigungssatz im Dialog („2.500 Credits für … als Verkauf auf Rechnung
  über 87,50 € gutschreiben?“).
- `POST /api/admin/buchungen/{id}/korrektur` — Hand-Buchungen berichtigen (Verkauf ↔ Bonus,
  Betrag, Nummer) mit Pflicht-Grund; Protokollzeile an der Buchung, Paket-Quelle/-Verfall ziehen
  mit, Credits bleiben. Stripe-Buchungen gesperrt.

## Umsatz-Buchungen (`backend/umsatz.py`, Tabelle `buchungen`)

- Stripe-Paket: Webhook `checkout.session.completed` (amount_total), Status „ausstehend“ bei
  SEPA, „ok“ bei `async_payment_succeeded`, „rueckgelaufen“ bei `async_payment_failed`.
- Stripe-Abo: jede bezahlte Abo-Rechnung (`invoice.paid`, amount_paid); `stripe_ref` eindeutig →
  wiederholte Webhooks buchen nichts doppelt.
- Auto-Verlängerung eines Rechnungs-Abos (Tageslauf): Buchung zum Listenpreis, in derselben
  Transaktion wie die Verlängerung; war das Abo ohne Berechnung, bleibt die Verlängerung kostenlos.
- Umsatz = alles außer Bonus und Rücklastschrift (SEPA unterwegs zählt, wird gesondert genannt).
  Grenzen für Tag/Monat/Jahr in deutscher Zeit (Europe/Berlin), gespeichert wird UTC.
- Konto löschen: Buchungen BLEIBEN (Aufbewahrungspflicht), nur der Kontoverweis wird gelöst.
- Export: Excel mit Summenzeile, CSV mit Semikolon/Dezimalkomma; Formel-Einschleusung
  (`=`, `+`, `-`, `@` am Zellanfang) wird entschärft.
- Kein Rechnungssystem: Rechnungen schreibt Actino; die Rechnungsnummer verknüpft beides.
  Späteres Anbinden eines Rechnungsprogramms: Export-Spalten sind dafür ausgelegt.

## Nachtragen (einmalig beim Ausrollen)

`python3 /app/scripts/umsatz_nachtragen.py` (Vorschau) bzw. `--ausfuehren`; Berichtigungen mit
`--als-verkauf PAKET_ID:BETRAG` und `--als-bonus PAKET_ID`. Idempotent.

## Tests

`tests/test_umsatz.py` (18, Wegwerf-Datenbank, inkl. Rechte und Endpunkte),
`tests/e2e/ui_verwaltung.py` (Klicktest mit axe, nur Staging), `tests/e2e/ui_smoke.py` (Seiten).
