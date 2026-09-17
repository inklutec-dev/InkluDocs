# Public API v1 — Aufbau, Regeln, Tests

Stand 17.09.2026. Für Partner ist die Anleitung die Seite `/api/v1/docs` (Template
`backend/templates/api_docs.html`, sechs Sprachen). Diese Datei ist die Innensicht.

## Zwei Schichten, eine Pipeline

- **Einzelbilder** (`/api/v1/alt-text`, seit Juni 2026) stehen in `main.py`: Bild rein,
  Alt-Text raus, Ergebnis mit `result_id` in `api_results`.
- **Dokumente** (`/api/v1/documents…`, 17.09.2026) stehen in `backend/api_dokumente_v1.py`
  nach dem Muster von `formular_api.py`: `build_router(Deps)`, in `main.py` eingehängt.
  Das Modul ist eine dünne Schicht (Schlüssel-Auth, Rate-Limit, Fehlerformat, Antwortform)
  über den **bestehenden App-Routen**. Es schlägt die Zielroute per Methode und Pfad in
  `app.routes` nach (`_route`) und ruft ihren Endpoint direkt auf. Besitzprüfung, Credits,
  Tageslimit, Export-Abnahme, Dateiprüfung bleiben damit an einer Stelle. Ein Body für die
  delegierte Route wird als `_Body(daten, request)` übergeben: `json()` liefert genau die
  geprüften Felder, alles andere (Header, Cookies für `resolve_ui_language`) kommt vom echten Request.

Ein „Dokument“ der API ist ein Projekt der App mit einer Datei oder Adresse
(`projects.api_key_id` merkt sich den anlegenden Schlüssel). Werkzeug nach Dateityp:
`.pdf` → pdf (oder formular mit `tool=formular`), `.docx` → word, Bild → grafik, JSON `{url}` → web.

## Zustände und Zähler

`projects.status` → API: neu=uploaded, extracting, extracted=ready, processing=generating,
done, error. `images.status` → pending, processing=generating, done/completed=done, error=failed.
`text_status` (offen | mit_text | dekorativ) spiegelt `main._exportable_alt_text`, also die
Zählung des Herunterladen-Dialogs (Michael Karbe 17.09.2026: „Text da oder nicht, egal woher“).

## Fehlerformat

Immer `{"error": {"code", "message", "status", …}, "detail": "…"}`. Codes stabil und englisch
(`_CODES`); ein dict-`detail` der App (z. B. `credits_fehlen_detail`) liefert Code und Zahlen
(preis, verfuegbar, fehlend). Unerwartete Ausnahmen werden protokolliert und als 500
`internal_error` beantwortet — nie mit Traceback nach außen.

## Limits und Kosten

Schreibende Aufrufe (POST/PATCH/DELETE) laufen durch `check_api_rate_limit` (60/min,
1.000/Tag je Schlüssel); lesende (Status, Items, Datei) nicht, damit Polling nicht zählt.
Tageslimit je Konto und Credits prüfen die delegierten App-Routen. Jeder Aufruf schreibt eine
Zeile in `api_usage` (`model_used = "v1.documents.<op>"`), Grundlage von
`database.get_api_key_stats` (Seite „API-Schlüssel“, Dashboard-Kachel, Endpunkt
`GET /api/api-keys/stats`). Exporte lassen sich keinem Schlüssel zuordnen; die Kachel nennt
deshalb nur Bild-Credits der über den Schlüssel angelegten Dokumente.

## Sicherheit

Keine Pfade von außen; Item-IDs werden gegen das Dokument geprüft; Werkzeug und Sprache sind
Whitelists; Dateigröße/-typ prüft `/api/upload`; PDF/UA-Download-Token prüft die App-Route
(Hex, Besitz, Pfad unter Export-/Ablage-Ordner). Swagger/ReDoc/openapi.json sind seit
17.09.2026 abgeschaltet (zeigten alle internen Routen). `delete_api_key` löscht Verbrauchs-
protokoll und Einzelbild-Ergebnisse des Schlüssels mit (vorher IntegrityError).

## Tests

- `tests/test_api_v1_dokumente.py` — Verdrahtung (alle Zielrouten existieren), Fehlerformat,
  Werkzeugwahl, text_status, Schlüssel löschen, Verbrauchszähler (unittest im Container).
- `tests/test_api_docs_0917.py` — Swagger aus, Doku in sechs Sprachen mit echten Zahlen.
- `tests/e2e/verify_api_v1_dokumente.py` + `setup_api_v1_key.py` — 41 Prüfungen über HTTPS
  wie ein Partner (PDF, Word mit PDF/UA, Formular; Korrektur, Exporte, Freigabe, Löschen).
  Ablauf: Schlüssel mit `setup_api_v1_key.py <mail> auf` im Container anlegen, Skript mit
  `INKLUDOCS_API_KEY` laufen lassen, danach `… ab`.

## Ausbau (Stand 17.09.2026)

Offen: eigener API-Tarif mit Verbrauchsabrechnung, Webhook statt Polling, MCP-Server über
derselben API, Export-Credits je Schlüssel, englische Fehlermeldungen (Codes sind es schon).
