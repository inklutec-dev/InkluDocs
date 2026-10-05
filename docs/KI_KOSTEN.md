# KI-Kosten (Stand 05.10.2026, mit Korrekturrunde)

Steves Auftrag (05.10.2026): „Kann man das verbinden, dass wir sehen, welcher User wie viel mit KI generiert hat,
welche KI-Kosten angelaufen sind und die Gesamt-KI-Kosten, zum Beispiel für den Monat — so wie beim Umsatz?“ Gezählt
wird überall, wo KI verbraucht wird (heute vor allem Alt-Texte, außerdem Chatbot, Quickinfos, Übersetzen); spätere
KI-Funktionen zählen automatisch mit.

Bis dahin kannte InkluDocs nur eine Pauschale je Credit (`billing.KOSTEN_PRO_CREDIT_EUR`, 1,2 Cent), die die
Kundenseite als „grobe Schätzung“ zeigte. Die Token-Zahlen, die Google mit jeder Antwort liefert, wurden nur bei
`DEBUG_GEN_RAW=true` ins Log gedruckt und gingen sonst verloren.

## Was erfasst wird

Jeder KI-Aufruf eine Zeile in der Tabelle `ki_aufrufe` (`database.init_db`):

- Zeitpunkt (UTC), Umgebung (`prod`, `staging`, `demo`, `test`)
- Kunde (`user_id`), zahlender Topf (`konto_user_id`, bei Teams der Inhaber — `billing._konto_fuer` zum Zeitpunkt des
  Aufrufs), Projekt, Dokument, Bild
- Zweck (`alttext`, `chatbot`, `quickinfo`, `uebersetzung`, `tagging_ki`, `ki_pruefung`, `unbekannt`) und Schritt
  (Schema bzw. `chat`/`agent`)
- Anbieter, Modell, Tokens getrennt nach Eingabe, Eingabe aus dem Zwischenspeicher, Cache-Schreiben, Ausgabe, Denken
- Kosten in USD und in Euro-Cent mit dem beim Aufruf gültigen Preis und Kurs (`kosten_usd`, `kosten_eur_cent`,
  `kurs_usd_eur`) — spätere Preisänderungen schreiben die Vergangenheit nicht um
- `erfolg` 0 bei Antworten, die ankamen, aber unbrauchbar waren (sie kosten trotzdem), `fehler` als Kurztext

Kein Preis für das Modell bekannt: Kosten `NULL` („ohne bekannten Preis“), nie still 0.
Nicht erfasst werden Aufrufe, die gar nicht beim Anbieter ankamen (Netzfehler, HTTP 429/5xx) — die kosten nichts —
und Bilder aus dem Zwischenspeicher (kein Modellaufruf).

## Wo erfasst wird (Kern: `backend/ki_kosten.py`)

An den Stellen, an denen InkluDocs überhaupt mit einem KI-Anbieter spricht:

- `pipelines/v4/gemini_client._invoke_gemini` — jede angekommene Antwort (auch die wiederholten)
- `pipelines/v4/bedrock_client` (Converse und invoke_model), `pipelines/v4/openai_client`
- `inkluagent/providers/gemini._aufruf`, `inkluagent/providers/bedrock` (Chatbot)

Wer und wofür kommt über einen `contextvars`-Kontext:

- Einstiegspunkte setzen Kunde/Projekt/Bild: Sammellauf je Bild (`main._process_project_lauf`), Neu-Generieren,
  API-Einzelbild, Chat (`chat_engine.process_message`), Chatbot-Werkzeuge (`ToolExecutor.execute`), Quickinfos
  (`formular_api`), Übersetzen (`uebersetzung_api`), Tagging/Prüfung (`@ki_kosten.mit_kunde` an den Arbeitsfunktionen
  in `tagging_api`).
- Die fachliche Funktion setzt den Zweck (`@ki_kosten.fuer_zweck("alttext")` an `pdf_processor.generate_alt_text`,
  ebenso `formular_ki.generiere_seite`, `uebersetzung._modell_aufruf_standard`, `pdf_pruefung.pruefe_dokument`,
  `pdf_struktur_tagging.zuordnung_je_seite`). Ein Alt-Text, den der Chatbot erzeugen lässt, zählt so als Alt-Text.
- Threads: Der Standard-Executor des Servers ist ein `ki_kosten.KontextExecutor` (gesetzt in `main.lifespan`) —
  `loop.run_in_executor(None, …)` nimmt den Kontext sonst nicht mit. Eigene ThreadPools nutzen `ki_kosten.mit_kontext`.
- Fehlt der Kontext trotzdem, zählt der Aufruf „ohne Zuordnung“ — er wird nie verworfen.

Nie-Stören-Garantie: `ki_kosten.erfasse` wirft nie; ein Fehler beim Mitschreiben wird geloggt, der KI-Lauf geht weiter.
Ohne Datenbankdatei (Eval-Läufe auf dem Host) wird nichts geschrieben und keine Datei angelegt.

**Neue KI-Funktion?** Läuft sie über die vorhandenen Clients, zählt sie von selbst. Für eine saubere Zuordnung:
Zweck in `ki_kosten.ZWECKE` eintragen, die Fachfunktion mit `@ki_kosten.fuer_zweck("…")` versehen und am
Einstiegspunkt Kunde/Projekt setzen (`ki_kosten.kontext(...)` bzw. `setze(...)` in asyncio-Aufgaben).

## Preise

`ki_kosten.PREISE_STANDARD`: USD je 1 Mio. Tokens je Modell als Preisstufen mit „gültig ab“, Wechselkurs mit Quelle.
Abgerufen am 05.10.2026:

- `gemini-3.1-pro-preview`: Eingabe 2,00, Ausgabe (inkl. Denken) 12,00, Zwischenspeicher 0,20; über 200.000 Tokens
  Eingabe 4,00 / 18,00 / 0,40 (Quelle: ai.google.dev/gemini-api/docs/pricing; Vertex AI führt dieselben Listenpreise)
- `gemini-3.8-flash`: Einführungspreis bis 31.12.2026 Eingabe 0,75, Ausgabe 3,75, Zwischenspeicher 0,075; ab
  01.01.2027 1,50 / 7,50 / 0,15 (steht schon als eigene Stufe drin)
- `eu.anthropic.claude-sonnet-4-6` (Bedrock, EU-Regionsinferenz +10 %): 3,30 / 16,50, Cache-Lesen 0,33,
  Cache-Schreiben 4,125; `claude-sonnet-4-6` global 3,00 / 15,00 / 0,30 / 3,75
- Wechselkurs: EZB-Referenzkurs 02.10.2026, 1 EUR = 1,1225 USD → 1 USD = 0,8909 EUR

Die Verwaltung (Voll-Admins) kann Preise je Modell ändern oder neue Modelle eintragen (immer mit Quelle) und den
Kurs ändern; gespeichert in `system_kv` unter `ki_preise`, 60 Sekunden zwischengespeichert. Eine **neue Preisstufe**
(anderes „gültig ab“) übernimmt Staffel und Cache-Preise (`grenze`, `ein_lang`, `aus_lang`, `cache_lang`,
`cache_schreiben`, `cache` — Letzteres seit der Nachprüfung N5, wenn das Feld leer bleibt) aus der jüngsten Stufe davor (`ki_kosten_api.staffel_vorlage`) — sonst würden lange Prompts bei
Gemini Pro nach einer Preisänderung zum Grundpreis gerechnet (Prüfung Entwicklung 05.10.2026, Befund 12). Der Dialog
sagt, welche Staffel übernommen wird; die Meldung nach dem Speichern auch. Fehler nennen das Feld (Text beginnt mit der
Beschriftung, Fokus dorthin), ein kaputter JSON-Körper gibt 400. Modellkennungen werden
erst exakt gesucht, dann ohne Versionsendung, ohne Regionsvorsatz und ohne `anthropic.`.

## Verwaltung „KI-Kosten“ (`/verwaltung/ki-kosten`)

Rechte wie beim Umsatz: lesen jeder Admin, Preise ändern nur Voll-Admins. Wie die anderen Verwaltungsseiten ohne
Tabellen (Überschriften, „Begriff: Wert“, Listen), Kunden und Projekte zum Aufklappen, Einzelheiten werden erst beim
Öffnen geladen.

- Monat wählbar (deutsche Monatsgrenzen wie beim Umsatz): „KI-Kosten (Listenpreise netto)“, „Umsatz (Endpreise, ohne
  Umsatzsteuer nach § 19 UStG)“, was bleibt, Aufrufe, verbrauchte Credits, **Kosten je Credit im Schnitt** neben der
  bisherigen Pauschale, erfasst seit.
- **Kosten je Credit** (Befund 5): Zähler und Nenner für DIESELBE Arbeit im SELBEN Zeitraum. Nenner = Credits aus
  KI-Aktionen (`ki_kosten.KI_AKTIONEN`: Alt-Texte `bild_generierung`, `alt_text_aenderung_chatbot`, Quickinfos
  `quickinfo_generierung`, `quickinfo_aenderung_chatbot`, `uebersetzung`), Zähler = KI-Kosten der Kunden für die Zwecke
  `alttext`, `chatbot`, `quickinfo`, `uebersetzung` — beides erst **ab Messbeginn** (sonst wäre die Zahl im
  Einführungsmonat um ein Vielfaches zu niedrig). Nicht dabei: Tagging und Prüfung (überwiegend PDFix bzw. veraPDF,
  je Seite berechnet), Exporte, Express (Handarbeit). Die Seite schreibt darunter, was gezählt wird.
- **Umsatz und KI-Kosten vergleichbar** (Befund 13): InkluTec berechnet als Kleinunternehmer nach § 19 UStG keine
  Umsatzsteuer (Preisseite: „Alle Preise sind Endpreise — gemäß § 19 UStG wird keine Umsatzsteuer berechnet“); der
  Umsatz in `buchungen` ist also netto gleich brutto. Die KI-Kosten sind Listenpreise netto. **Offen für den
  Steuerberater:** Google (und AWS) rechnen als ausländische Unternehmer ab; für deren Leistungen schuldet InkluTec die
  Umsatzsteuer nach § 13b UStG und kann sie als Kleinunternehmer nicht als Vorsteuer abziehen — die echten Kosten wären
  dann rund 19 % höher als hier angezeigt. Bitte klären; danach ggf. einen Aufschlag in die Rechnung nehmen.
- Nach Zweck, nach Kunde (teuerste zuerst; je Kunde Credits und Umsatz im Monat; Projekte → Bilder; gelöschte Projekte
  mit Nummer „Gelöschtes Projekt 812“), nach Modell. Keine Ansage beim Laden; nach „Anzeigen“ Fokus auf die Überschrift
  mit dem Monat, beim Blättern auf „Seite X von Y“.
- Preisliste mit Quelle, künftigen Stufen und den Dialogen „Preis ändern“ / „Wechselkurs ändern“

Kundenseite (`/verwaltung/kunden/<id>`): „KI-Kosten dieses Kontos“ gemessen seit Messbeginn plus — nur für die
Credits davor — die Pauschale als Schätzung; je Monat „gemessen“, „teils gemessen, teils geschätzt“ oder „geschätzt“.

Endpunkte (`backend/ki_kosten_api.py`):

- `GET /api/admin/ki-kosten?jahr&monat[&umgebung]` — Monatsbericht
- `GET /api/admin/ki-kosten/kunde?konto=<id|ohne>&jahr&monat` — Projekte eines Kontos
- `GET /api/admin/ki-kosten/projekt?projekt=<id|ohne>&konto=<id|ohne>&jahr&monat` — Bilder eines Projekts
- `GET /api/admin/ki-preise`, `POST /api/admin/ki-preise/modell`, `POST /api/admin/ki-preise/kurs`

## Datenschutz

Die Zeilen enthalten keine Inhalte (keine Prompts, keine Bilder, keine Texte), nur Kennungen und Zahlen. Beim Löschen
eines Kontos (`database.delete_user_data`) werden Kunde, Topf, Projekt, Dokument und Bild geleert: die Kosten bleiben in
den Monatssummen (sie sind angefallen), zählen danach „ohne Zuordnung“.

## Grenzen

- Die Summe ist unsere Rechnung (Tokens × Listenpreis × Kurs). Googles Rechnung kann leicht abweichen (Rundung,
  Tageskurs, Rabatte, Gutschriften). **Noch offen:** Abgleich mit der echten Google-Abrechnung (Abrechnungsexport nach
  BigQuery, daraus Monatssumme und Restguthaben in der Verwaltung) — braucht eine Einrichtung im Google-Konto, nur
  auf Steves Wort.
- Staging und Produktion laufen über dasselbe Google-Projekt; jede Instanz sieht nur ihre eigene Datenbank. Die
  Google-Summe enthält also beides.
- Messbeginn ist der erste Aufruf nach dem Einspielen; davor gibt es nur die Pauschale.

## Tests

- `tests/test_ki_kosten.py` (Unit, eigene Wegwerf-Datenbank, 24): Preisrechnung inkl. Staffel und Stufen, unbekannte
  Preise, Kontext durch Executor/Threads/Dekoratoren, Gemini-Client-Haken, Auswertung, Kontolöschung; Kosten je Credit
  nur aus KI-Credits ab Messbeginn, Staffel und Cache-Preis bleiben bei neuer Stufe, Fehler mit Feld, kaputter
  JSON-Körper 400
- `tests/e2e/verify_ki_kosten.py` (im Staging-Container, echter Alt-Text-Lauf): Erfassung mit Kunde/Bild/Zweck,
  Neu-Generieren, Bericht und Drill-down, Kundenseite, Rechte (Kunde/Nur-Einsicht/Voll-Admin), Eingabeprüfungen
- `tests/e2e/ui_ki_kosten.py` (Playwright + axe, nur Staging): Seite, Aufklappen, beide Dialoge, Fokus, Englisch
- `tests/e2e/ui_smoke.py` kennt die neue Seite
