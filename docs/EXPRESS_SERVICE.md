# Express-Service Stufe 1 (Stand 05.10.2026)

Wunsch von Michael Karbe, Skizze mit Steve am 05.10.2026: Kunden, die ihre Dokumente lieber von Profis barrierefrei
aufbereiten oder prüfen lassen, bestellen das in InkluDocs. Lieferung innerhalb von 48 Stunden — dem Kunden wird bewusst
**keine Uhrzeit** genannt (Steve: „lässt sich schwierig bewerkstelligen, kann man später einbauen“). Bezahlt wird mit
Credits. Stufe 1 = der Kernablauf: bestellen, Posteingang in der Verwaltung, liefern, Mails, Credits.

**Nur auf Staging an.** Schalter `funktionen.EXPRESS` über die Umgebungsvariable `EXPRESS_SERVICE=an`
(`docker-compose.staging.yml`). Prod und Demo haben die Variable nicht → aus; dann antworten alle Express-Endpunkte und
-Seiten mit 404, Menüpunkte und Links fehlen. In der Demo ist er auch mit Variable aus.

## Ablauf für den Kunden

1. **Wege hinein:** Seitenleiste „Express-Service“ (`/express`), Link „Vom Express-Service bearbeiten lassen“ in der
   Projektansicht „Dokument“ (öffnet `/express?projekt=<id>` mit allen Dokumenten des Projekts angehakt), Abschnitt
   „Meine Express-Aufträge“ auf der Startseite.
2. **Dokumente wählen:** Projekt in einer Auswahlliste, dann Kontrollkästchen je Dokument (Name, Seitenzahl) und
   „Alle Dokumente dieses Projekts“. Nur PDFs (andere Dateien und passwortgeschützte PDFs sind gesperrt, mit Grund).
   Oder eine PDF **ohne Projekt** hochladen: InkluDocs legt dafür ein Projekt „Express-Auftrag <Nr>“ an (weitere
   Uploads in denselben Auftrag landen im selben Projekt). Bewusst **kein** „alle Dokumente aus allen Projekten“.
3. **Deine Auswahl** (= Warenkorb, serverseitig gespeichert, bleibt beim Verlassen der Seite): je Dokument die Leistung
   „Barrierefrei aufbereiten (mit Prüfung)“ oder „Nur prüfen (Prüfbericht)“, Entfernen-Knopf; jede Änderung sagt Anzahl,
   Seiten und Credits an.
4. **Angaben:** Ansprechpartner (vorbelegt mit dem Kontonamen), Telefon (freiwillig), Hinweise (freiwillig).
5. **Prüfen und bestellen:** Aufstellung je Dokument, Summe, verfügbares Guthaben, „Lieferung innerhalb von 48 Stunden“,
   bei zu wenig Guthaben Hinweis mit Link „Credits kaufen“. Zwei Pflicht-Kontrollkästchen, nicht vorab angehakt:
   Bedingungen (Seite `/express/bedingungen`, als ENTWURF gekennzeichnet) und Einverständnis zur Bearbeitung durch
   Mitarbeiter von InkluTec und Actino. Knopf „Zahlungspflichtig bestellen“ (§ 312j BGB).
6. **Auftragsübersicht** `/express/auftrag/<id>` (Nachweis, **keine Rechnung**): Stand, Credits (vorgemerkt / abgebucht /
   wieder frei), Dokumente mit Downloads nach der Lieferung, Angaben, Einverständnis mit Fassung und Zeitpunkt, Verlauf.
   Rückfragen beantwortet der Kunde direkt dort. **Drucken** (Druck-CSS blendet Seitenleiste und Knöpfe aus) und
   **Als PDF herunterladen** — die PDF wird nur ausgeliefert, wenn sie die PDF/UA-Prüfung (veraPDF) besteht, sonst
   bleibt die Druckansicht (Antwort 503 mit Hinweis).

## Credits: vormerken, abbuchen, freigeben

- **Bestellen merkt vor.** Die Summe offener Aufträge (Status neu, in Arbeit, Rückfrage) eines Topfs ist
  `billing.vorgemerkt(konto)`; sie mindert `verfuegbare_credits` und damit jede Guthaben-Prüfung (Alt-Texte, Tagging,
  Export …) und das Guthaben auf der Express-Seite. Kein eigener Kontostand, der driften könnte.
- **Liefern bucht ab** — in EINER Transaktion mit dem Statuswechsel: je Leistung ein Verbrauchs-Ereignis
  (`usage_events`, Quelle `express`, Aktionen `express_aufbereiten` / `express_pruefen`) auf den Topf der Bestellung,
  danach `billing._pakete_abbuchen` für einen Überhang. Ein zweites Liefern findet den Auftrag nicht mehr offen
  (Vergleichen-und-Tauschen im UPDATE) und bucht nichts.
- **Storno gibt frei** (nur vor der Lieferung): die Vormerkung fällt mit dem Status weg, abgebucht wurde nichts.
- Preis und Seiten werden beim Bestellen frisch gezählt und festgeschrieben; spätere Preisänderungen betreffen nur neue
  Aufträge. Team-Konten: es zahlt der Topf, aus dem das Konto beim Bestellen arbeitet (`billing._konto_fuer`).
- Startseite und Abo-Seite nennen „Davon für Express-Aufträge vorgemerkt: N Credits“.
- Grenze: Das Guthaben wird beim Bestellen geprüft. Eine im selben Augenblick parallel laufende Generierung kann es
  noch verbrauchen; dann bucht die Lieferung trotzdem ab (Überhang wie bei jeder Aktion), nie doppelt.

## Verwaltung „Express-Aufträge“

- `/verwaltung/express`: Posteingang nach Dringlichkeit — Überfällig, Neu, In Arbeit, Rückfrage beim Kunden; Geliefert
  und Storniert (je die letzten 100) zum Aufklappen. Je Auftrag Kunde, Dokumente, Seiten, „fällig in N Stunden“ bzw.
  „überfällig seit N Stunden“, Bearbeiter.
- `/verwaltung/express/<id>`: Überblick mit Frist, Knöpfe **Übernehmen / Mir zuweisen**, **Rückfrage stellen** (Dialog),
  **Liefern**, **Stornieren** (Dialog, Grund Pflicht). Je Dokument: Original herunterladen, alle Originale als ZIP,
  **Ergebnis (PDF) hochladen** (nur bei „aufbereiten“) und **Prüfbericht (PDF) hochladen**. Beim Ergebnis läuft veraPDF
  automatisch mit; meldet es Abweichungen, fragt „Liefern“ einmal nach („Trotzdem liefern?“). Fehlt ein Ergebnis bzw.
  bei „Nur prüfen“ der Bericht, nennt die Seite das, und Liefern lehnt ab. Interne Notiz (nie für den Kunden), Verlauf
  mit internen Schritten.
- **Frist:** Standard 48 Stunden ab Bestellung (Einstellung). Während einer Rückfrage ruht sie und verlängert sich bei
  der Antwort um die Wartezeit (so steht es in den Bedingungen). Intern gibt es Datum und Uhrzeit, dem Kunden nicht.
- **Einstellungen** (nur Voll-Admins): Credits je Seite für beide Leistungen (**Platzhalter 50 und 25**, Hinweis bis
  „Preise sind festgelegt“ angekreuzt ist), Frist in Stunden, höchstens Seiten und Dokumente je Auftrag (500 / 50),
  Adresse für Team-Benachrichtigungen (leer = Support-Postfach). Gespeichert in `system_kv` `express_einstellungen`.
- **Recht „Express-Bearbeiter“** (Spalte `users.express_bearbeiter`, vergibt nur ein Voll-Admin unter „Einstellungen des
  Express-Service“): für z. B. einen Partner, der die Dokumente aufbereitet. Sieht in der Verwaltung **nur** die
  Express-Aufträge (Navigation und Seitenleiste nur dieser Punkt), kann sie bearbeiten, sieht aber keine Kunden-,
  Umsatz-, KI-Kosten- oder API-Daten und keine Konto-Kennungen; Einstellungen und Bearbeiter-Recht kann er nicht
  ändern. Nur-Einsicht-Admins lesen, ändern nichts. Voll-Admins dürfen alles.

## E-Mails

Über den vorhandenen Systemmail-Weg (`main.send_email`), immer **ohne Anhang**, mit Link; Adressen auf `.invalid`
werden nie versandt (Tests). Kunde: Bestellbestätigung, Rückfrage, Lieferung, Storno. Team (Einstellungsadresse bzw.
Support-Postfach plus alle Express-Bearbeiter): neuer Auftrag, Antwort des Kunden, **Erinnerung 12 Stunden vor der
Frist** und **Überfällig** — je genau einmal (die Meldung wird vor dem Versand atomar in der Datenbank beansprucht).
Die Erinnerungen prüft eine Schleife alle 10 Minuten (gestartet in `main.lifespan`, nur wenn der Schalter an ist).

## Datenmodell (`database.init_db`)

- `express_auftraege`: ein Auftrag; Status `entwurf` (= Warenkorb, höchstens einer je Konto, eindeutiger Index) |
  `neu` | `in_arbeit` | `rueckfrage` | `geliefert` | `storniert`. Kunde, zahlender Topf, Angaben, Seiten, Credits, Frist,
  Fälligkeit, Zustimmung (Wortlaut beider Häkchen, Fassung `express.ZUSTIMMUNG_FASSUNG`, Sprache, Zeitpunkt,
  Absender-Kennung), Idempotenz-Schlüssel (eindeutig je Konto), Bearbeiter, Liefer- und Storno-Angaben, interne Notiz,
  Erinnerungs-Vermerke, Rückfrage-Beginn.
- `express_positionen`: je Dokument Quelle (Projekt, Dokument, Name als Momentaufnahme), Seiten, Leistung, Credits,
  Pfade für Original, Ergebnis, Prüfbericht, veraPDF-Ergebnis.
- `express_verlauf`: Schritte mit Zeit, Person, Text; `fuer_kunde = 0` für interne Schritte.
- Dateien: `results/<konto>/_express/<auftrag>/pos<id>_(original|ergebnis|bericht).pdf` — Pfade nur aus Zahlen, nie
  aus Dateinamen. Die Originale werden beim Bestellen kopiert (unveränderte Kundendatei `roh_path`, sonst die
  Arbeitsdatei): Was die Profis bekommen, ändert sich nicht mehr, auch wenn der Kunde das Projekt weiter bearbeitet oder
  löscht.
- Konto löschen: Aufträge, Positionen, Verlauf und der Ordner gehen mit; war das Konto Bearbeiter, bleibt nur der Name.

## Sicherheit

- Jede Kundenfunktion prüft den Besitz **im SQL** (Auftrag/Position/Dokument + `user_id`), Fremdes antwortet 404 —
  geprüft für Projekt-Dokumente, Auswahl, Positionen, Aufträge, Antwort, Downloads und Nachweis.
- Rechte frisch aus der Datenbank je Anfrage (`get_current_user` liest `is_admin`, die Bearbeiter-Prüfung liest
  `express_bearbeiter` und `is_active`).
- Uploads: Dateiendung und Magic Bytes (`%PDF-`), Größe (Kunde 50 MB, Ergebnis 100 MB), lesbar und ohne
  Öffnungspasswort (PyMuPDF), Vorprüfung wie beim normalen PDF-Upload. Dateinamen werden nur angezeigt (Pfadteile und
  Steuerzeichen entfernt), Downloads mit `Content-Disposition: attachment` (RFC 6266/5987) und `nosniff`.
- Doppelbestellung: Knopf sperrt im Browser, Server über Idempotenz-Schlüssel (eindeutiger Index) und eine Sperre um
  Prüfen-und-Vormerken; Doppel-Lieferung über Vergleichen-und-Tauschen in einer Transaktion.
- Bremsen je Konto: Bestellen 10/h, Upload 30/h, Antwort 20/h, Nachweis-PDF 30/h (429).
- CSRF wie im Bestand: Sitzungs-Cookie `SameSite=Lax`, JSON-Anfragen; Multipart-Uploads gehen ohne Cookie nicht durch.
- Fehlertexte ohne Interna (Pfade, SQL); Details nur im Server-Log.

## Bewusste Entscheidungen

- **Keine Uhrzeit** für den Kunden (Steve); intern Fälligkeit mit Uhrzeit für Erinnerung und „überfällig“.
- **Nachweis statt Rechnung:** Die Rechnung entsteht beim Credit-Kauf; eine zweite Rechnung würde die Umsatzsteuer
  doppelt ausweisen. Die Übersicht sagt das ausdrücklich. (Bitte von Michael/Steuerberater bestätigen lassen.)
- **Kein Bestellen über den Chatbot** (Abweichung vom Grundsatz „Chatbot = Oberfläche“): Eine zahlungspflichtige
  Bestellung braucht den gesetzlich beschrifteten Knopf und zwei bewusst gesetzte Häkchen. Der InkluAgent kennt den
  Express-Service in Stufe 1 noch nicht; ein Lese-Werkzeug („Stand meiner Aufträge“) wäre unkritisch und kann folgen.
- **Lieferung in die Auftragsübersicht, nicht als neue Fassung im Projekt** (Abweichung von der Skizze): InkluDocs
  kennt je Dokument genau eine Arbeitsdatei, an der Tags, Alt-Texte und Quickinfos des Kunden hängen. Eine „neue
  Fassung“ gibt es nicht; das Ergebnis darüber zu legen, würde den Arbeitsstand des Kunden still ersetzen. Darum bleibt
  das Projekt unberührt, und die fertigen Dateien stehen in der Auftragsübersicht (Link in der Liefer-Mail, auf der
  Startseite und unter „Meine Aufträge“). Eine echte Versionierung am Dokument ist ein eigener Schritt.
- **Nur PDF** in Stufe 1 (Word/PowerPoint später).
- **Bedingungen nur auf Deutsch** wie die übrigen Rechtstexte (I18N.md).
- Express-Credits zählen in der Verwaltung „KI-Kosten“ nicht zur Kennzahl „Kosten je Credit“ (Handarbeit, keine KI).

## Offen vor Prod (nur auf Steves Wort)

1. **Wer bearbeitet?** Datenschutzerklärung und AVV müssen die menschliche Bearbeitung durch InkluTec und Actino
   abdecken (Kategorien, Zweck, Speicherdauer), Vertraulichkeitsvereinbarung mit dem Partner.
2. **Echte Preise** (Platzhalter 50/25 Credits je Seite; zum Vergleich: automatisches Tagging 20 je Seite) und
   **Frist** (48 Stunden oder 2 Werktage — Wochenenden).
3. **Bedingungstext** abstimmen und rechtlich prüfen, dann `ZUSTIMMUNG_FASSUNG` hochzählen und „ENTWURF“ entfernen.
4. **Aufbewahrung:** Wie lange bleiben Originale und Ergebnisse liegen (heute bis zur Kontolöschung)?
5. **Verrechnung InkluTec ↔ Actino** für Express-Aufträge.
6. Prod: `EXPRESS_SERVICE=an` in `docker-compose.yml` setzen, Rollout wie üblich.

Spätere Stufen: Erinnerung/Rückfragen ausbauen, Warenkorb über mehrere Projekte komfortabler, Word/PowerPoint,
Ergebnis als neue Fassung am Dokument im Projekt (braucht Versionierung), Lese-Werkzeug im InkluAgent.

## Dateien

- Kern: `backend/express.py`; Endpunkte und Mails: `backend/express_api.py`; Guthaben: `backend/billing.py`
  (`vorgemerkt`, `EXPRESS_OFFEN`); Schalter: `backend/funktionen.py` (`EXPRESS`)
- Seiten: `templates/express.html`, `express_auftrag.html`, `express_bedingungen.html`, `verwaltung_express.html`,
  `verwaltung_express_auftrag.html`; Link im Projekt `frontend/dokument.js`; Startseite `templates/dashboard.html`;
  Seitenleiste `frontend/dashboard.js`; Druck-CSS `frontend/dashboard.css`

## Tests

- `tests/test_express.py` (Unit, eigene Wegwerf-Datenbank, 26): Warenkorb, Fremd-Zugriffe, Grenzen, Bestellen mit
  Vormerkung und Idempotenz (auch 4 gleichzeitige Klicks), Vormerkung sperrt andere Ausgaben, Liefern bucht genau einmal
  (auch 4 gleichzeitig), Storno, Dateinamen nie Pfad, Upload-Prüfung, Downloads erst nach Lieferung, Frist ruht bei
  Rückfrage, Erinnerung/Überfällig je einmal, Nachweis-OOXML, Kontolöschung, Einstellungen, Bearbeiter, Schalter
- `tests/e2e/verify_express.py` (im Staging-Container über HTTP, 90): ganzer Ablauf inkl. Nachweis-PDF und ZIP,
  IDOR-Fälle, Rechte (Kunde, Nur-Einsicht, Bearbeiter, Voll-Admin), Uploads, Doppel-Bestellung/-Lieferung, Storno
- `tests/e2e/ui_express.py` (Playwright + axe, nur Staging): Link im Projekt, Auswahl, Leistung, Entfernen mit Fokus,
  Pflichtfelder mit Fokus, Bestellen, Danke-Meldung, Startseite, Verwaltung (Liste, Auftrag, Dialoge, Upload, Liefern),
  Download beim Kunden — axe 0 Verstöße auf jeder Seite und in jedem Dialog
- `tests/e2e/ui_smoke.py` kennt `/express`, `/express/bedingungen`, `/verwaltung/express`
