# Abrechnung / Abo-Credit-System

Stand: Etappe 1 (31.07.2026) — ZAEHLEN OHNE SPERREN. Modell-Doku fuer Entwickler.
Beschlussgrundlage: Steve + Michael, Meeting 31.07.2026 (Mail "Abo-Modell: finaler
Vorschlag mit Kalkulation auf Juli-Basis").

## Grundidee

EINE Waehrung (Credits), EIN zentrales Modul (`billing.py`), EIN Ereignis-Protokoll
(`usage_events`). Jede kostenpflichtige Aktion — egal welches Werkzeug — laeuft
ueber genau zwei Funktionen:

    billing.pruefe_kontingent(user_id)  -> darf die Aktion stattfinden?
    billing.verbuche(user_id, quelle, aktion=..., image_id=...)  -> Ereignis speichern

Neue Werkzeuge (PDF-Umwandlung, Punktschrift, ...) brauchen NUR eine neue Zeile in
`AKTIONS_PREISE` und rufen dieselben zwei Funktionen. Kein Umbau, nie.

## Spielregeln (fachlich, von Steve festgelegt 31.07.2026)

- 1 Credit = 1 Bild-Generierung, egal ueber welchen Weg.
- NUR Erfolge kosten: Fehllaeufe werden nicht verbucht.
- Cache-Treffer kosten nichts (`result["from_cache"]`, gesetzt in pdf_processor).
- Chatbot: Reden ist frei — sobald er einen Alt-Text ERZEUGT oder AENDERT,
  kostet es 1 Credit (Aktion `alt_text_aenderung_chatbot` fuer den Aendern-Fall).
- Einzel-Neu-Generieren kostet immer (echte neue KI-Anfrage, nie aus dem Cache).
- Der interne Verify-/Redakteurs-Pass kostet den Kunden nie extra.
- Admins werden GEZAEHLT (Datenlage), aber nie gesperrt.
- Uebertrag: max. 1 Monatskontingent in den Folgemonat, nur Bezahl-Plaene (Etappe 2).
- API zieht aus DEMSELBEN Topf wie die App; Schluessel erst ab erster Zahlung (Etappe 2/3).

## Andockpunkte (Etappe 1, alle live)

1. Sammellauf `POST /api/projects/{id}/generate` (main.py): Pruefung VOR JEDEM Bild
   in der Schleife (Abbruch laesst Rest auf 'pending'), Verbuchung je Erfolg,
   Quelle `sammellauf`. Deckt auch Web-Scan-Projekte ab (scan-url generiert nicht selbst).
2. Einzel-Neu-Generieren `POST /api/projects/{pid}/regenerate/{iid}` (main.py):
   Pruefung am Anfang (429), Verbuchung nach Erfolg, Quelle `einzeln`.
3. Public API `POST /api/v1/alt-text` (main.py): Pruefung nach dem Rate-Limit (429),
   Verbuchung nach Erfolg, Quelle `api`. Bestehende Minuten-/Tages-Rate-Limits
   bleiben als Burst-Schutz.
4. Chatbot Generierung: `inkluagent/adapters/inkludocs.py::run_pipeline_for_image`
   = gemeinsamer Trichter beider Chatbot-Wege (chat_engine + Tool-Schicht), Quelle `chatbot`.
5. Chatbot Text-Aenderung: `inkluagent/tools/altext.py::update_alt_text`,
   Aktion `alt_text_aenderung_chatbot`. (`revert_alt_text` kostet nichts.)
   Hinweis: `adapters/inkludocs.py::update_alt_text` wird aktuell nirgends aufgerufen
   (toter Import in chat_engine) — falls je aktiviert, dort ebenfalls verbuchen!

Demo-Instanz (DEMO_MODE): eigene Wegwerf-DB, bewusst OHNE Abo-Zaehlung.

## Schema

    usage_events(id, user_id, konto_user_id, quelle, aktion, credits, image_id, created_at)
    users.plan  TEXT DEFAULT 'free'   (free | pro | team | enterprise)

`konto_user_id` = wessen Kontingent belastet wird. Etappe 1: immer = user_id.
Etappe 2 (Team-Toepfe): Aufloesung ueber users.abo_owner_id in `billing._konto_fuer`.

## Schalter

    ABO_ENFORCEMENT=off   nur zaehlen (Etappe 1, aktueller Zustand)
    ABO_ENFORCEMENT=on    Kontingente werden durchgesetzt (429)

Gesetzt in .env.staging / .env.prod, durchgereicht in den Compose-Dateien.
Kontingente + Aktionspreise: NUR in billing.py (PLAN_KONTINGENTE, AKTIONS_PREISE).

## Ausblick (nicht Teil von Etappe 1)

Etappe 2: Dashboard-Verbrauchsanzeige (/api/me), Seite "Abo & Verbrauch",
Admin-Stufenverwaltung + Rechnungsweg, Team-Toepfe (abo_owner_id), Uebertrag-Deckel,
Kontingent-Checks in der Chatbot-Tool-Schicht mit freundlicher Meldung.
Etappe 3: Stripe (Checkout, Webhook, payments-Tabelle, Kundenportal), quota_pakete.
Etappe 4: maschinenlesbare API-Fehlercodes, api_results-Retention, PayPal-Rueckbau
(dashboard.js:171, demo-shell.js:96, nutzungsbedingungen.html), Preisseite.
