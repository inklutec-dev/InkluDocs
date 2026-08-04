# Abrechnung / Abo-Credit-System

Stand: Etappe 2 (31.07.2026) — Team-Toepfe, Uebertrag, Zusatz-Pakete,
Abo-Auskunft + Verwaltungs-Endpunkte. Enforcement weiter per Schalter.
Nachbesserungen aus dem Review vom 31.07.2026 eingearbeitet (Team-Beitritt
nur mit Zustimmung, transaktionale Paket-Abbuchung, Geister-Team-Schutz).
Modell-Doku fuer Entwickler.
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
   Kontingent-Sperre liefert den Marker `{"kontingent_erschoepft": True}` (NICHT None —
   None heisst "Bild nicht gefunden"); chat_engine und tools/altext.py sagen die Sperre
   mit der Meldung `KONTINGENT_MELDUNG` an, statt Bilder stumm zu ueberspringen.
5. Chatbot Text-Aenderung: `inkluagent/tools/altext.py::update_alt_text`,
   Aktion `alt_text_aenderung_chatbot`. (`revert_alt_text` kostet nichts.)
   Hinweis: `adapters/inkludocs.py::update_alt_text` wird aktuell nirgends aufgerufen
   (toter Import in chat_engine) — falls je aktiviert, dort ebenfalls verbuchen!

Demo-Instanz (DEMO_MODE): eigene Wegwerf-DB, bewusst OHNE Abo-Zaehlung.

## Schema

    usage_events(id, user_id, konto_user_id, quelle, aktion, credits, image_id, created_at)
    quota_pakete(id, user_id, groesse, verbleibend, quelle, notiz, erstellt_am, verfaellt_am)
    paket_abbuchungen(id, paket_id, konto_user_id, betrag, created_at)
    team_einladungen(id, token, inhaber_id, email, erstellt_am, gueltig_bis, eingeloest_am)
    users.plan             TEXT DEFAULT 'free'   (free | pro | team | enterprise)
    users.abo_owner_id     INTEGER  (NULL = eigenes Konto; sonst Team-Inhaber)
    users.plan_gueltig_bis TEXT     (NULL = unbefristet; Rechnungskunden/Gruender-Tarif)

`konto_user_id` = wessen Kontingent belastet wird (bei Team-Mitgliedern der
Inhaber). `quota_pakete.user_id` = ebenfalls das Abrechnungs-KONTO, nie das
Mitglied. Verfallene Pakete bleiben als Beleg stehen.
`paket_abbuchungen` ist das append-only Protokoll der Paket-Abbuchungen
(fuer die Differenz-Rechnung in `_pakete_abbuchen`, siehe unten).
`team_einladungen` traegt die offenen/eingeloesten Team-Einladungen (Token,
7 Tage gueltig; Muster wie password_resets).

## Etappe-2-Regeln (31.07.2026)

Team-Aufloesung: `billing._konto_fuer` loest users.abo_owner_id genau EIN
Level auf (keine Ketten); Fehlerfall -> eigenes Konto. Team-Verwaltung nur
durch den Inhaber (plan='team' UND selbst kein abo_owner_id). Sitze:
TEAM_SITZE_INKLUSIVE=5 in billing.py (inkl. Inhaber); mehr erst mit
Online-Zahlung (Etappe 3), bis dahin 400er.

Team-Beitritt NUR MIT ZUSTIMMUNG (Review-Befund 1, 31.07.2026):
`POST /api/team/einladen` haengt bestehende Konten NIE mehr direkt um,
sondern legt eine Einladung in `team_einladungen` an (Token, +7 Tage) und
mailt dem EINGELADENEN einen Annahme-Link. `GET /team-einladung/{token}`
verlangt Login mit der eingeladenen Adresse, prueft alle Regeln erneut
(Inhaber noch Team, Ziel nicht anderweitig gebunden, Sitz-Deckel — atomar
per BEGIN IMMEDIATE) und setzt erst dann abo_owner_id + eingeloest_am.
Nur bei UNBEKANNTEN Adressen wird das Konto direkt mit abo_owner_id
angelegt (es entsteht erst durch die Einladung). Die Endpunkt-Antwort ist
bewusst NEUTRAL ("Einladung versandt", kein Name/Flag) gegen
E-Mail-Enumeration; Konflikte werden erst beim Einloesen gemeldet.

Geister-Team-Schutz (Review-Befunde 4/5): `delete_user_data` loest beim
Loeschen eines Inhabers alle Mitglieder (abo_owner_id -> NULL) und entsorgt
seine offenen Einladungen; `POST /api/admin/users/{id}/plan` loest beim
Wechsel WEG von 'team' ebenfalls alle Mitglieder und meldet die Anzahl.
Faellt trotzdem einmal eine Konto-Zeile weg (baumelndes abo_owner_id),
prueft `pruefe_kontingent` NICHT fail-open, sondern das Free-Kontingent
gegen den EIGENEN Verbrauch des Nutzers.

Uebertrag ("max. ein Monatskontingent in den Folgemonat, nur Bezahl-Plaene,
Free nie"): ZUSTANDSLOS aus usage_events berechnet (`billing._uebertrag`).
Rekurrenz ueber alle Kalendermonate seit dem ersten Ereignis-Monat des
Kontos, laufender Monat ausgenommen:

    u_0    = 0
    u_next = clamp(K + u - verbrauch_monat, 0, K)

Bewusste Naeherung: aktuelles Plan-K fuer die ganze Historie (Planwechsel
werden nicht historisiert).

Verbrauchsreihenfolge: erst Monats-Budget (Kontingent + Uebertrag), dann
Zusatz-Pakete — Paket mit fruehestem verfaellt_am zuerst.
Gesperrt wird erst, wenn Monats-Budget aufgebraucht UND pakete_rest 0 ist
(nur bei ABO_ENFORCEMENT=on, nie fuer Admins, Enterprise/None nie).

Paket-Abbuchung (Review-Befund 2, 31.07.2026): `verbuche` schreibt das
Ereignis und bucht Pakete in EINER Transaktion (BEGIN IMMEDIATE, eine
Verbindung) — parallele Verbuchungen am Budget-Rand koennen sich weder
doppelt abbuchen noch Ueberhang verschlucken. Die Soll-Abbuchung ist eine
DIFFERENZ, kein Einzel-Ereignis-Anteil:

    soll    = max(0, verbraucht_monat - (kontingent + uebertrag))
    bereits = SUM(paket_abbuchungen.betrag) im laufenden Monat
    abzug   = max(0, soll - bereits)

Dadurch idempotent und selbstheilend: frueher ungedeckter Ueberhang wird
nachgebucht, sobald wieder Pakete da sind. Fehler in der Abbuchung rollen
die GANZE Transaktion zurueck und landen im Log (Nie-Crashen-Garantie
unveraendert in `verbuche`).

Pakete: `billing.schenke_credits(konto_id, menge, notiz, quelle)` legt ein
Paket mit 12 Monaten Verfall an (quelle 'admin' | 'stripe' | 'rechnung').
Noch ohne Oberflaeche/Endpunkt, bewusst — Steve 31.07.

Endpunkte Etappe 2:
- GET /api/me: zusaetzlicher Block "abo" (aufgeloestes Konto; daily_limit
  bleibt vorerst als deprecated erhalten). Seite: GET /abo (abo.html).
- GET /api/team, POST /api/team/einladen, DELETE /api/team/mitglied/{id}
  (nur Team-Inhaber; Einladung = Zustimmungs-Fluss, siehe oben; unbekannte
  Adressen bekommen Konto + Passwort-Link-Mail mit Team-Hinweis).
- GET /team-einladung/{token}: Annahme-Seite (Login-Pflicht, Bestaetigung
  im Stil der anderen _auth_notice_page-Seiten).
- POST /api/admin/users/{id}/plan (Voll-Admin): plan + gueltig_bis setzen;
  Wechsel weg von 'team' loest alle Mitglieder (Antwort nennt die Anzahl).

Mail-Hygiene (Review-Befund 7): display_name wird an allen Setz-Stellen
auf eine Zeile normiert und auf 100 Zeichen begrenzt
(`_normiere_display_name` in main.py); in den Einladungs-Mails werden
Name/E-Mail zusaetzlich mit html.escape() ausgegeben, Betreffzeilen der
Einladungs-Mails enthalten keine Nutzereingabe.

## Schalter

    ABO_ENFORCEMENT=off   nur zaehlen (Etappe 1, aktueller Zustand)
    ABO_ENFORCEMENT=on    Kontingente werden durchgesetzt (429)

Gesetzt in .env.staging / .env.prod, durchgereicht in ALLEN Compose-Dateien
(docker-compose.yml, .staging, .demo — Review-Befund 3; in der Demo hart
`off`, weil die Demo bewusst ohne Abo-Zaehlung laeuft).
Kontingente + Aktionspreise: NUR in billing.py (PLAN_KONTINGENTE, AKTIONS_PREISE).

## Ausblick (nicht Teil von Etappe 2)

Etappe 3: Stripe (Checkout, Webhook, payments-Tabelle, Kundenportal),
Sitz-Zukauf, Paket-Kauf (quota_pakete quelle='stripe'), Auswertung
plan_gueltig_bis, Credits-Geschenk-Oberflaeche.
Etappe 4: maschinenlesbare API-Fehlercodes, api_results-Retention, PayPal-Rueckbau
(dashboard.js:171, demo-shell.js:96, nutzungsbedingungen.html), Preisseite.

## Lizenzschluessel-Modell (04.08.2026 — Michaels Modell, Mail 03.08. bestaetigt)

Loest Pro/Team/Enterprise als VERKAUFTES Modell ab (die Plaene bleiben
technisch fuer Bestandsdaten und eine spaetere Team-Stufe bestehen):
EIN Schluessel pro Unternehmen, 9,95 EUR/Monat inkl. 50 Credits
(PLAN_KONTINGENTE["lizenz"]), 6 Monate Mindestlaufzeit, beliebig viele Nutzer.

Ablauf:
- Voll-Admin erzeugt Schluessel: POST /api/admin/lizenzen
  {domain?, laufzeit_monate=6, anzahl=1, notiz?} -> IDOC-XXXX-XXXX-XXXX
  (Zeichenvorrat ohne 0/O/1/I/L — telefonier- und vorlesbar). Das ist
  zugleich der RECHNUNGSWEG (Kauf auf Rechnung ueber Actino): Rechnung
  bezahlt -> Admin erzeugt Schluessel -> Kunde aktiviert.
- Aktivierung: POST /api/abo/lizenz {schluessel} auf der /abo-Seite.
  Der ERSTE Aktivierer wird Topf-Inhaber (plan="lizenz",
  plan_gueltig_bis = jetzt + laufzeit); jeder WEITERE Aktivierer haengt
  sich per users.abo_owner_id in den Firmen-Topf (Team-Topf-Technik).
- Domain-Bindung (Steves Nachschaerfung 3): Schluessel ist an die
  E-Mail-Domain der Firma gebunden — entweder vom Admin vorgegeben oder
  bei der Erst-Aktivierung an die Domain des Aktivierers gebunden.
  Freemail-Domains (billing.FREEMAIL_DOMAINS) werden abgelehnt.
  Aktivieren koennen nur Konten mit passender Domain (403 sonst;
  ungueltig und gesperrt antworten identisch 404 — kein Schluessel-Orakel).
- Auto-Rueckfall (Punkt 8): billing.effektiver_plan wertet
  plan_gueltig_bis LAZY aus — abgelaufener Bezahl-Plan zaehlt ueberall
  als "free", ohne Cronjob und ohne UPDATE. users.plan bleibt als Beleg
  stehen; Verlaengern = plan_gueltig_bis neu setzen.
- Topf-Verwaltung: /api/team + Mitglied-Entfernen gelten jetzt fuer
  Inhaber von "team" ODER "lizenz" (lizenz OHNE Sitz-Deckel);
  /api/team/einladen bleibt team-only — Lizenz-Beitritt laeuft NUR ueber
  den Schluessel, sonst waere die Domain-Bindung umgehbar.
- /api/me liefert abo.lizenz {domain, gueltig_bis, schluessel} — den
  Schluessel selbst sieht nur der Inhaber.

Tabelle lizenzschluessel: schluessel UNIQUE, domain, inhaber_user_id,
monats_credits (dokumentarisch, Kontingent kommt aus PLAN_KONTINGENTE),
laufzeit_monate, gueltig_bis, status (neu/aktiv/gesperrt), notiz,
erstellt_von, aktiviert_am. Migration additiv+idempotent.

Verifikation 04.08.2026: verify_lizenz.py (32 Checks im Container, API +
billing + Ablauf/Rueckfall), ui_lizenz.py (Playwright-Klicktest Aktivierung),
axe_abo.py (0 Verstoesse), verify_abo2.py Regression 26/26.

Offen (Folge-Runden laut Umbau-Liste 04.08.): Free 20->10 +
Domain-Buendelung + Wegwerf-/XFF-Schutz, Wasserzeichen im Free-PDF-Export
(PDFix-Lizenz-Frage!), Paketpreise (100=20 EUR, 500=87,50, 1000=150,
verfallen erst bei Kuendigung statt +12 Monate), Stripe-Halbjahresrechnung
(SEPA+Karte+Apple Pay+PayPal), Tarif-/Preisseite, Admin-Oberflaeche fuer
Schluessel (bisher nur API), Gruender-Regel (Basisschluessel 3 Monate frei).
