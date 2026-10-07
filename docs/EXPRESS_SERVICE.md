# Express-Service Stufe 1 (Stand 06.10.2026, Runde 7)

Wunsch von Michael Karbe, Skizze mit Steve am 05.10.2026: Kunden, die ihre Dokumente lieber von Profis barrierefrei
aufbereiten oder prüfen lassen, bestellen das in InkluDocs. Lieferung innerhalb von 48 Stunden — dem Kunden wird bewusst
**keine Uhrzeit** genannt (Steve: „lässt sich schwierig bewerkstelligen, kann man später einbauen“). Bezahlt wird mit
Credits. Stufe 1 = der Kernablauf: bestellen, Posteingang in der Verwaltung, liefern, Mails, Credits.

**Nur auf Staging an.** Schalter `funktionen.EXPRESS` über die Umgebungsvariable `EXPRESS_SERVICE=an`
(`docker-compose.staging.yml`). Prod und Demo haben die Variable nicht → aus. In der Demo ist er auch mit Variable aus.

**Was der Schalter abschaltet** (Prüfung Entwicklung 05.10.2026, Befund 9 — die sichere Variante): nur das
**Neu-Bestellen** (Express-Warenkorb, Projektauswahl, Bestellen; Link und Knopf im Projekt). Sobald es **einen bestellten Auftrag**
gibt (`express.gibt_bestellte()`), bleiben Auftragsübersichten, Downloads, Antwort auf Rückfragen und die ganze Verwaltung
(Liefern, Stornieren, Hochladen) erreichbar, und die Erinnerungsschleife läuft weiter. So bleiben vorgemerkte Credits nach
dem Abschalten nie hängen, Kunden kommen an ihre Ergebnisse, und offene Aufträge lassen sich zu Ende führen. Die Seite
`/express` zeigt dann nur „Meine Aufträge“ mit dem Hinweis „Neue Express-Aufträge sind zurzeit nicht möglich“, die
Verwaltung einen entsprechenden Hinweis. Gab es nie einen Auftrag (Prod, Demo), antwortet alles mit 404 wie bisher.
Menüpunkt „Meine Aufträge“ (bis Runde 7 „Express-Service“; `/api/me` `user.express`): Schalter an oder eigene
Aufträge vorhanden.

## Ablauf für den Kunden

**Zwei Seiten** (Runde 7, Michael Karbe 06.10.2026): `/express` = **„Meine Aufträge“** (Navigation, wie „Meine
Projekte“ und „Meine Ablage“) — ganz oben die Aufträge als Karten (Punkt 6), darunter nur der Weg zum Express-Warenkorb.
`/express/warenkorb` = **Express-Warenkorb**, der neue Auftrag in vier Schritten (Punkte 2 bis 5). Auf `/express` steht
kein Bestellformular mehr.

1. **Wege hinein:** Seitenleiste „Meine Aufträge“ (`/express`) und „Express-Warenkorb“ (`/express/warenkorb`, Modus
   siehe unten), Knopf „In den Express-Warenkorb“ am Dokument, Link „Vom Express-Service bearbeiten lassen“ in der
   Projektansicht „Dokument“ (`/express?projekt=<id>` leitet auf `/express/warenkorb?projekt=<id>` um: alle Dokumente
   des Projekts angehakt, Fokus auf Schritt 1), Abschnitt „Meine Aufträge“ auf der Startseite (je Auftrag ein Sprung auf
   die aufgeklappte Karte, `/express?auftrag=<id>#exa_karte_<id>`, dazu „Alle Aufträge“).
2. **Dokumente wählen:** Projekt in einer Auswahlliste (nur Projekte eines angebotenen Dateityps — heute PDF — mit
   mindestens einem Dokument; gleichnamige mit Anlagedatum unterschieden), dann Kontrollkästchen je Dokument (Name,
   Seitenzahl) und „Alle Dokumente dieses Projekts“. Andere Dateitypen und passwortgeschützte PDFs sind gesperrt, mit
   Grund. **Kein Hochladen bei der Auswahl** (Runde 7, Michael Karbe 06.10.2026, Punkt 2): eine neue Datei kommt
   zuerst in ein Projekt (Hinweis mit Link „Neues Projekt anlegen“); `POST /api/express/warenkorb/hochladen` antwortet
   410 mit diesem Hinweis. Die erweiterbare Liste der Dateitypen bleibt. Bewusst **kein** „alle Dokumente aus allen
   Projekten“.
3. **Deine Auswahl** (= Warenkorb, serverseitig gespeichert, bleibt beim Verlassen der Seite): eine **reine
   Dokumentliste** — Name, Seiten, Credits, Entfernen-Knopf (Michael Karbe 05.10.2026, Punkt 3: „Wir bieten nur die
   Aufbereitung an.“). Eine Leistungswahl je Dokument erscheint nur, wenn es für den Dateityp mehr als eine eingeschaltete Leistung
   gibt (dann speichert sie erst, wenn die Auswahl einen Moment steht). Hinzufügen sagt Anzahl, Seiten, Credits und — wenn
   es nicht reicht — das fehlende Guthaben an; Entfernen zeigt eine sichtbare Meldung und setzt den Fokus darauf.
4. **Angaben:** Ansprechpartner (vorbelegt mit dem Kontonamen), Telefon (freiwillig), Hinweise (freiwillig).
5. **Prüfen und bestellen:** nur Summen — Dokumente, Seiten, die Zusammensetzung des Preises (Runde 7: „Seiten: 3 × 50
   Credits = 150 Credits. Grundpreis: 2 × 100 Credits = 200 Credits. Summe: 350 Credits.“), verfügbares Guthaben,
   „Lieferung innerhalb von 48 Stunden“ (die Dokumente stehen schon unter „Deine Auswahl“, Punkt 4); bei zu wenig
   Guthaben Hinweis mit Link „Credits kaufen“. **Ein** Pflicht-Kontrollkästchen „Ich akzeptiere die Bedingungen für den Express-Service.“ (`required`,
   Legende „Zustimmung (Pflicht)“, Fehler am Kästchen, der Fehler verschwindet beim Ankreuzen), nicht vorab angehakt. Die
   Bearbeitung durch Mitarbeiter von InkluTec und Actino steht ausdrücklich in den Bedingungen (Seite
   `/express/bedingungen`, als ENTWURF gekennzeichnet, Fassung `2026-10-05-entwurf-3`; Punkt 5) — beim Auftrag
   gespeichert werden wie bisher Wortlaut, Fassung, Sprache, Zeitpunkt und gekürztes Netz. Aufträge bis Fassung -2
   behalten ihr zweites Häkchen im Nachweis. Knopf „Zahlungspflichtig bestellen“ (§ 312j BGB). Die Seite schickt
   Korb-Nummer, angezeigte Summe und Korb-Fassung mit; hat sich etwas geändert (Preis, zweiter Tab, Seite aus dem Zurück-Speicher), bestellt der Server
   nicht, sondern antwortet 409, und die Seite zeigt die neue Aufstellung. Nach dem Bestellen landet der Kunde in
   „Meine Aufträge“ (`/express?neu=<id>`): Danke-Meldung oben (sichtbar, Fokus, keine zweite Ansage), der neue Auftrag
   aufgeklappt; die Adresse verliert danach `?neu`.
6. **Meine Aufträge** (Michael Karbe 05.10.2026, Punkte 1 und 2; seit Runde 7 eigene Seite mit H1 „Meine Aufträge“,
   `id="h-auftraege"`, damit alte Links `/express#h-auftraege` weiter treffen): Karten wie die Dokumente der
   Projektansicht „Dokument“ (`section.card.dok-karte` > `details.dok-klappe`, H3 mit Stand-Abzeichen im `summary`, Infos
   als Liste, Linie, Knöpfe darunter; Überschriften ohne Sprung: H1 „Meine Aufträge“, Karte H2, „Dokumente dieses
   Auftrags“ H3, „Neuer Express-Auftrag“ H2 — Runde 8). **Standardmäßig zu — auch bei nur einem Auftrag** (Michael sah im Konto mit einem
   Auftrag keine aufklappbare Liste); offen sind der Auftrag nach dem Bestellen (`?neu=<id>`) und der, auf den gesprungen
   wird (`?auftrag=<id>` oder `#exa_karte_<id>`, Fokus auf seine Überschrift). Darin aufklappbar „Dokumente dieses Auftrags (N)“ mit Name, Seiten, Stand und Downloads. Knöpfe:
   „Auftragsübersicht öffnen“, **„Umbenennen“** und — nur bei gelieferten oder stornierten Aufträgen — **„Löschen“**.
   - **Umbenennen:** eigener Name des Kunden (Spalte `auftrag_name`, höchstens 120 Zeichen, leer = „Auftrag <Nr>“). Er
     steht in der Liste, auf der Startseite, als H1 der Auftragsübersicht („Express-Auftrag 11: Jahresberichte“) und im
     Nachweis-PDF („Name des Auftrags“); die Nummer bleibt immer daneben. Die Verwaltung sieht ihn als „Name des Kunden
     für den Auftrag“.
   - **Löschen** (Bestätigungsdialog wie beim Dokument-Löschen, Fokus auf „Abbrechen“): laufende Aufträge nie (kein Knopf,
     Server 409). Gelöscht werden für den Kunden der Auftrag, alle Dateien (Originale, Ergebnisse, Prüfberichte), die
     Dokumentliste, Ansprechpartner, Telefon, Hinweise, interne Notiz, Storno-Grund, Name, Verlauf und die Wortlaute der
     Zustimmung (`express.kunde_loeschen`). Intern bleibt ein **knapper Buchungsnachweis**: Nummer, Kunde und zahlender
     Topf, Bestell- und Liefer- bzw. Stornodatum, Seiten, Credits, Stand, Fassung und Zeitpunkt der Zustimmung, dazu
     `kunde_geloescht_am`. Die Verwaltung zeigt „vom Kunden gelöscht“ (Liste) bzw. „Vom Kunden gelöscht am …“ mit
     diesem Nachweis und einem Verlaufseintrag; die Credits-Buchung bleibt unberührt. Danach antworten Auftrag,
     Downloads und Nachweis für den Kunden mit 404.
   Rückmeldungen nach Umbenennen und Löschen: sichtbare Meldung über der Liste, Fokus darauf, keine Live-Ansage.
7. **Auftragsübersicht** `/express/auftrag/<id>` (Nachweis, **keine Rechnung**): Stand, Credits (vorgemerkt / abgebucht /
   wieder frei), Dokumente — je Dokument aufklappbar mit Seiten, Leistung, Credits samt Zusammensetzung („200 (2 × 50
   Credits je Seite plus 100 Credits je Dokument)“), Stand, Prüfung und Downloads —, dazu **aufklappbar** (Runde 7,
   standardmäßig zu, H2 im `summary`): Angaben, Einverständnis mit Fassung und Zeitpunkt, Verlauf. Beim Drucken
   (`beforeprint`, auch über das Browser-Menü) ist alles aufgeklappt, danach wie vorher. Alte Links mit `?neu=1` zeigen
   weiter die Danke-Meldung.
8. **Anmeldung über Links** (Runde 7, Michaels „verlorenes Projekt“: Er öffnete den Link aus der Bestätigungsmail ohne
   Sitzung, meldete sich mit seinem zweiten Konto an und sah den Auftrag nicht):
   - Wer ohne Sitzung eine geschützte Seite aufruft — auch Projekt-Links, Lieferung, Rückfrage, Team-Einladung —, landet
     auf `/login?weiter=<Pfad>` (Hinweis „Nach der Anmeldung geht es weiter zur aufgerufenen Seite.“) und nach der
     Anmeldung wieder dort. Erlaubt sind nur interne, relative Pfade (`backend/weiterleitung.py`, `sicheres_ziel`:
     kein `//host`, kein Schema, keine Backslashes oder Steuerzeichen, nie `/login` selbst; der Pfad wird vorher
     normalisiert, „/..//host“ wird zu „/host“); das Anmeldeformular prüft noch einmal. Der Hinweis nennt das Ziel („…
     zu deinem Express-Auftrag“ bzw. „… zur aufgerufenen Seite“) und hängt per `aria-describedby` am E-Mail-Feld, das
     den Fokus bekommt (Runde 8). Kein offener Redirect. Läuft die Sitzung auf einer App-Seite ab, führt `zurAnmeldung()`
     (`frontend/dashboard.js`) ebenso mit Rücksprung zur Anmeldung.
   - Ist man mit einem anderen Konto angemeldet als dem, mit dem bestellt wurde, zeigt die Auftragsübersicht „Dieser
     Auftrag gehört nicht zu deinem Konto (<eigene Adresse>). Melde dich mit dem Konto an, mit dem du bestellt hast.“
     und den Knopf „Abmelden und anders anmelden“ (meldet ab und führt mit Rücksprung zur Anmeldung). Der Fokus geht
     auf den Meldungssatz (`tabindex="-1"`, wie die übrigen Meldungen), ohne zusätzliche Live-Ansage (Runde 8). Über den Auftrag
     wird nichts verraten: die API bleibt 404 — gleich für fremde, gelöschte und nicht vorhandene Aufträge. Aufträge
     gehören dem **bestellenden Konto**, auch wenn aus einem Team-Topf bezahlt wurde; der Topf-Inhaber sieht fremde
     Aufträge seiner Mitglieder nicht (nur die Vormerkung in `/api/team`).
   - Jede Kunden-Mail (Bestätigung, Rückfrage, Lieferung, Storno) nennt „Bestellt mit dem Konto <Adresse>. Melde dich
     mit diesem Konto an, um den Auftrag zu sehen.“

App-weit (Nachprüfung Barrierefreiheit N1): Eingabefelder, Auswahllisten und Textfelder zeigen bei Tastaturfokus einen
3-px-Fokusring (`input:focus-visible, select:focus-visible, textarea:focus-visible` in `frontend/style.css`), auch auf
Anmelden, Registrieren und Passwort vergessen.
   Rückfragen beantwortet der Kunde direkt dort. **Drucken** (Druck-CSS blendet Seitenleiste, Knöpfe und Meldungen aus)
   und **Als PDF herunterladen** — die PDF wird nur ausgeliefert, wenn sie die PDF/UA-Prüfung (veraPDF) besteht. Der
   Download läuft per `fetch`: ein Fehler (Umwandler aus, Bremse) steht als Satz neben dem Link (Fokus dorthin), nicht
   als rohe JSON-Seite. Datum im PDF ausgeschrieben („5. Oktober 2026, 12:04“), „1 Seite“ in der Einzahl. Das Ergebnis
   der automatischen Prüfung sieht der Kunde nur als „bestanden“ bzw. „mit Hinweisen, die unser Team geprüft hat“.

## Express-Warenkorb am Dokument und in der Navigation (Zusatz 05.10.2026, Steve)

Zwei Abkürzungen zum Warenkorb, beide **ohne Codeänderung umschaltbar** in der Verwaltung unter „Einstellungen des
Express-Service“ (`express_einstellungen`, nur Voll-Admins) — und nur wirksam, solange Neu-Bestellen an ist
(`funktionen.EXPRESS`):

- **Knopf „In den Express-Warenkorb“** (`korb_knopf`, Standard an): an jedem PDF-Dokument der Projektansicht „Dokument“,
  dort wo „Umbenennen“ und „Löschen“ stehen (`frontend/dokument.js`, `Dokument.inExpressKorb`; nie im Gastzugang). Er
  legt genau dieses Dokument in den Entwurfs-Korb — über denselben Endpunkt wie die Express-Seite
  (`POST /api/express/warenkorb/dokumente`), also nur eigene Dokumente (Besitz im SQL). Bei Team-Konten ist der Korb
  der des eigenen Kontos; gezahlt wird beim Bestellen aus dem Topf, aus dem das Konto dann arbeitet. Liegt das Dokument
  schon im Korb, sagt die Meldung „… liegt schon im Express-Warenkorb“. Bestätigung: sichtbarer Satz unter den Knöpfen
  mit Link „Zum Warenkorb“, der Fokus geht darauf — keine zusätzliche Live-Ansage. An die Oberfläche kommt der Schalter
  als `window.FUNKTIONEN.express_korb_knopf` (`express_api.fuer_oberflaeche()`).
- **Eintrag „Express-Warenkorb“ in der Hauptnavigation** (`korb_navigation`): „immer“ (Standard jetzt), „nur wenn etwas
  im Warenkorb liegt“ (`mit_inhalt`) oder „aus“ — welcher Modus bleibt, bespricht Steve mit Michael. Mit Inhalt heißt
  er „Express-Warenkorb: N Dokumente“ (Zahl als Text, Einzahl „1 Dokument“). Ändert sich die Zahl (Knopf am Dokument,
  Hinzufügen/Entfernen auf der Express-Seite), wechselt der Text still (`window.expressKorbAnzeigen` in
  `frontend/dashboard.js`) — die Bestätigung kommt vom auslösenden Knopf. Ziel ist `/express/warenkorb`: dieselbe Seite
  wie `/express`, geöffnet bei „2. Deine Auswahl“ (Fokus dorthin), Titel „Express-Warenkorb“; `aria-current` trägt dort
  nur dieser Eintrag, auf `/express` nur „Express-Service“. Daten: `/api/me` → `user.express_warenkorb` =
  `{"modus", "dokumente"}` oder `null` (aus), gezählt mit `express.korb_kurz()` (ohne Dateien zu öffnen).
- Tests: `tests/test_express.py` Klasse `WarenkorbZusatz`; `tests/e2e/verify_express.py` (Abschnitt A);
  Klicktest mit axe `tests/e2e/ui_express_korb.py` (Knopf, Bestätigung, Fokus, stille Zahl, aria-current, alle drei
  Modi, Knopf aus).

## Credits: vormerken, abbuchen, freigeben

- **Bestellen merkt vor.** Die Summe offener Aufträge (Status neu, in Arbeit, Rückfrage) eines Topfs ist
  `billing.vorgemerkt(konto)`; sie mindert `verfuegbare_credits` und damit jede Guthaben-Prüfung (Alt-Texte, Tagging,
  Export …) und das Guthaben auf der Express-Seite. Kein eigener Kontostand, der driften könnte.
- **Liefern bucht ab** — in EINER Transaktion mit dem Statuswechsel: je Leistung ein Verbrauchs-Ereignis
  (`usage_events`, Quelle `express`, Aktionen `express_aufbereiten` / `express_pruefen`) auf den Topf der Bestellung,
  danach `billing._pakete_abbuchen` für einen Überhang. Ein zweites Liefern findet den Auftrag nicht mehr offen
  (Vergleichen-und-Tauschen im UPDATE) und bucht nichts.
- **Storno gibt frei** (nur vor der Lieferung): die Vormerkung fällt mit dem Status weg, abgebucht wurde nichts. Bei
  Free-Domain-Konten bekommt zurück, wer im Bestellmonat danach aus Paketen gezahlt hat, was ohne den Auftrag gratis
  gewesen wäre (Storno-Ausgleich, Runde 6; `backend/ABRECHNUNG.md`).
- **Monatswechsel** (Befund 1): Die Abbuchung zählt zum Monat der **Bestellung** — dort war das Guthaben vorgemerkt.
  Liegt die Bestellung in einem früheren Kalendermonat, tragen die Verbrauchs-Ereignisse den Bestellzeitpunkt, und ein
  Überhang wird für jenen Monat von den Paketen abgebucht (`billing.pakete_abbuchen_fuer_monat`). Sonst verfiele beim
  Monatswechsel Übertrag, den der Kunde im Bestellmonat nicht nutzen durfte.
- **Eine Rechnung für „verfügbar“** (Nachkontrolle Runde 3, R1/R2): `billing.guthaben(konto, plan, kontingent,
  domain)` ist die einzige Stelle, die das Guthaben rechnet — in EINER Lese-Transaktion auf EINER Verbindung (Runde 5,
  R4-1: ein fester Stand, auch wenn gleichzeitig eine Lieferung committet). `pruefe_kontingent` (damit
  `verfuegbare_credits`, jede Werkzeug-Prüfung, die Sperre `erlaubt`, `/api/me` und die Startseite) und `/api/team`
  lesen nur daraus. Es gilt immer `verfuegbar = max(0, rest + Zusatz-Credits − vorgemerkt_laufend)`.
  - `rest` ist das Monatsbudget, in dem offene Vormonats-Bestellungen wie Verbrauch ihres Bestellmonats zählen
    (`_uebertrag(…, mit_vormerkung=True)`, N2) — so sieht der Übertrag nach der Lieferung aus.
  - `vorgemerkt` = alle offenen Express-Credits des Topfs (bzw. der Free-Domain); `vorgemerkt_laufend` = was davon das
    Guthaben dieses Monats bindet (`billing._express_bindung`): Bestellungen dieses Monats ganz (bei Free-Domains die
    der ganzen Domain), dazu je Monat seit der ältesten offenen Vormonats-Bestellung der Paket-Überhang, den die
    Lieferung dort noch nachbucht. Das ist im Bestellmonat der Teil der Bestellung, den sein Budget nicht trägt, und in
    den Monaten danach — auch im laufenden — der Verbrauch über dem kleineren Budget mit Vormerkung, den die
    Buchungswege nach dem echten Budget noch nicht von den Paketen abgebucht haben (R1: sonst ließ sich dieser Teil
    ungedeckt verbrauchen).
  - Buchungswege (Paket-Abbuchungen) rechnen weiter nur mit echten Ereignissen — ein späterer Storno kostet so keine
    Paket-Credits. Beim Liefern eines Vormonats-Auftrags werden Bestellmonat, die Monate dazwischen (Auftrag über mehr als
    einen Monatswechsel offen) und der laufende Monat abgeglichen (`pakete_abbuchen_fuer_monat`, `_pakete_abbuchen`).
  - Ergebnis: Das Guthaben ist vor und nach der Lieferung gleich, und wer immer wieder verbraucht, was angezeigt wird,
    kommt genau auf das, was Monatsbudgets und Pakete hergeben (Tests `Runde4`).
  - Free-Domains (Runde 5, Steves Regel; Einzelheiten `backend/ABRECHNUNG.md`): Das Gratis-Volumen ist gemeinsam,
    Pakete gehören dem kaufenden Konto. Jeder Auftrag hat an seinem Bestellzeitpunkt einen Paket-Teil (was das
    gemeinsame Volumen dort nicht mehr trägt); den bindet nur das bestellende Konto, und genau der geht bei der
    Lieferung von dessen Paketen ab. Offene Aufträge dieses Monats belegen für alle Konten der Domain das gemeinsame
    Volumen; wer danach bucht, zahlt den Überhang aus seinen Paketen. Die Verbrauchs-Ereignisse einer Lieferung tragen
    bei Free-Domain-Konten den Bestellzeitpunkt.
- **Bestellen in Schritten** (Befunde 2–4, 18): Der Korb wird zuerst eingefroren (Zwischenstand `bestellung` — Änderungen
  aus einem zweiten Tab landen in einem neuen Korb), Seiten frisch gezählt, Summe und Fassung mit dem verglichen, was
  der Kunde gesehen hat (sonst 409), das Guthaben **streng** geprüft (ein Datenbankfehler sperrt mit 503, statt alles zu
  erlauben), die Originale kopiert und in EINER Transaktion die Positionen neu gelesen, verglichen und vorgemerkt.
  Scheitert etwas, wird der Korb wieder freigegeben. Hängt ein Korb nach einem Absturz länger als 10 Minuten im
  Zwischenstand, wird er wieder zum Korb — gibt es inzwischen einen neuen, wandern seine Dokumente dorthin (N3). Der
  Idempotenz-Schlüssel gilt nur für den Korb, zu dem er gehört. Ist der Korb einer Seite inzwischen bestellt (zweiter
  Tab), antwortet der Server 409 „veraltet“ mit dem aktuellen (leeren) Korb, und die Seite lädt ihren Stand neu (N4).
- **Preis** (Runde 7, Michaels Richtpreis 06.10.2026): je Dokument Seiten × Preis je Seite + **Grundpreis je Dokument**
  — Standard 50 Credits je Seite plus 100 Credits je Dokument (`Leistung.preis_standard`, `grundpreis_standard`;
  `express.preis`, `preis_teile`). Beide Werte pflegt die Verwaltung je Leistung. Der Warenkorb liefert je Position
  `preis_seite`, `grundpreis`, `credits` und als Summen `credits_seiten`, `credits_grund`; die Korb-Fassung enthält
  die Grundpreise (Änderung → 409).
- Preis und Seiten werden beim Bestellen festgeschrieben (`credits`; seit Runde 7 auch `preis_seite`, `grundpreis` je
  Position für Auftragsübersicht und Nachweis); spätere Preisänderungen betreffen nur neue Aufträge, offene Aufträge
  behalten ihren Preis. Vormerkung, Abbuchung, Storno und die Free-Domain-Regel rechnen mit diesem gespeicherten Betrag.
  Team-Konten: es zahlt der Topf, aus dem das Konto beim Bestellen arbeitet (`billing._konto_fuer`); die Topf-Übersicht
  des Inhabers (`/api/team`) nennt `vorgemerkt` (bindend) und `verfuegbar_nach_vormerkung` — aus `billing.guthaben`. Free-Konten einer Firmen-Domain teilen
  sich das Volumen — und damit auch die Vormerkung (`billing.vorgemerkt_domain`, Befund 7).
- Startseite und Abo-Seite nennen „Davon für Express-Aufträge vorgemerkt: N Credits“ — N ist `vorgemerkt_laufend`, damit
  Monatsrest + Zusatz-Credits − N genau das verfügbare Guthaben ergibt (R2); reicht das Guthaben für eine andere Aktion
  nicht, nennt die Meldung dieselbe Zahl.
- Grenze: Das Guthaben wird beim Bestellen geprüft. Eine im selben Augenblick parallel laufende Generierung kann es
  noch verbrauchen; dann bucht die Lieferung trotzdem ab (Überhang wie bei jeder Aktion), nie doppelt.

## Verwaltung „Express-Aufträge“

- `/verwaltung/express`: Posteingang nach Dringlichkeit — Überfällig, Neu, In Arbeit, Rückfrage beim Kunden (nur Gruppen
  mit Aufträgen; sind alle leer, eine Zeile „Keine offenen Aufträge“); Geliefert und Storniert (je die letzten 100) zum
  Aufklappen, mit Überschrift im `summary`. Je Auftrag Kunde, Dokumente, Seiten, „fällig in N Stunden“ bzw. „überfällig
  seit N Stunden“ — bei einer Rückfrage „Frist ruht bis zur Antwort des Kunden“ —, Bearbeiter.
- `/verwaltung/express/<id>`: Überblick mit Frist, Knöpfe **Übernehmen / Mir zuweisen**, **Rückfrage stellen** (Dialog),
  **Liefern**, **Stornieren** (Dialog, Grund Pflicht). Je Dokument: Original herunterladen, alle Originale als ZIP
  (ungepackt gespeichert, Bremse 20/h, Platzprüfung, Reste abgebrochener Downloads werden nach einer Stunde gelöscht),
  **Ergebnis hochladen** (wenn die Leistung ein Ergebnis verlangt) und **Prüfbericht hochladen** — beide mit der
  Hochlade-Komponente der Projekte; Dateityp und Beschriftung kommen aus der Leistung. Beim Ergebnis läuft die
  automatische Prüfung seines Dateityps mit (PDF: veraPDF), bevor die Antwort kommt; solange sie läuft, wartet „Liefern“
  (409). Meldet sie Abweichungen, fragt „Liefern“ einmal nach („Trotzdem liefern?“). Hochladen und Liefern laufen unter
  derselben Schreibsperre; nach der Lieferung wird nichts mehr ersetzt (Befund 11). Fehlt ein Ergebnis bzw. bei „Nur
  prüfen“ der Bericht, nennt die Seite das, und Liefern lehnt ab. Interne Notiz (nie für den Kunden), Verlauf mit internen
  Schritten. Fehler in Dialogen und Formularen stehen am Feld (`aria-invalid`, Beschreibung), der Fokus geht dorthin.
  Die Hochladeknöpfe haben je Dokument einen eindeutigen Namen (versteckter Zusatz „: Ergebnis für „Jahresbericht.pdf““,
  Option `zusatz` in `hochladefeld.js`), ohne eigene Landmarke je Fläche (Nachprüfung Barrierefreiheit N2). Das
  Prüfergebnis nennt die Zahl der nicht erfüllten Regeln („3 Regeln nicht erfüllt“, N4).
- **Vom Kunden gelöschte Aufträge** (Runde 3): In der Liste mit dem Zusatz „vom Kunden gelöscht“; die Detailseite zeigt
  nur den Buchungsnachweis (Nummer, Kunde, Topf, Daten, Seiten, Credits, Stand, Zustimmung) mit „Vom Kunden gelöscht am
  …“ und den Verlaufseintrag — keine Dateien, keine Knöpfe.
- **Frist:** Standard 48 Stunden ab Bestellung (Einstellung). Während einer Rückfrage ruht sie und verlängert sich bei
  der Antwort um die Wartezeit (so steht es in den Bedingungen). Intern gibt es Datum und Uhrzeit, dem Kunden nicht.
- **Einstellungen** (nur Voll-Admins): je Leistung **Credits je Seite** und **Grundpreis je Dokument** (je ein Feld je
  Eintrag der Liste, Grundpreis 0 bis 100.000; das Platzhalter-Kennzeichen „Preise sind festgelegt“ ist seit Michaels
  Richtpreis entfallen; Zahlen nur ganz, Punkt oder Leerzeichen nur als Tausendertrennung wie „1.000“ — „1.5“ oder
  „100.00“ ergeben einen Fehler am Feld statt still 15 bzw. 10.000, Runde 8), Frist in Stunden, höchstens Seiten und
  Dokumente je
  Auftrag (500 / 50), **Dateien löschen nach Tagen** (0 = nie, Standard bis Steve entscheidet), Adresse für
  Team-Benachrichtigungen (leer = Support-Postfach). Gespeichert in `system_kv` `express_einstellungen`; Preise unter
  `preise`, Grundpreise unter `grundpreise` (fehlt er, gilt der Standard; die alten Schlüssel
  `preis_aufbereiten`/`preis_pruefen` werden beim Lesen übernommen). Fehler nennen das Feld (`grundpreis_<leistung>`).
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
Ging sie an **keinen** Empfänger raus (SMTP gestört), wird sie freigegeben und im nächsten Durchlauf erneut versucht,
höchstens sechsmal (Zähler `meldung_fehlversuche` in der Datenbank, übersteht Neustarts); ging sie an mindestens einen,
bleibt es dabei (keine Doppelmails, Befund 10). Die Schleife läuft alle
10 Minuten (gestartet in `main.lifespan`, immer — auch bei ausgeschaltetem Schalter) und räumt dabei auch Dateien nach der
Aufbewahrungsfrist und ZIP-Reste weg. Wird das Konto eines Team-Inhabers gelöscht, werden offene Aufträge seiner
Mitglieder aus seinem Topf storniert (Kunde und Team bekommen die Storno-Mail); wird ein Kundenkonto mit offenem Auftrag
gelöscht, bekommt das Team die Mail „Entfallen“.

**Testaufträge benachrichtigen nie Bearbeiter** (Runde 8, Vorfall 07.10.2026: 15 „[STAGING]“-Team-Mails aus
Testläufen an Michael, der auf Staging Express-Bearbeiter ist): Gehört ein Auftrag einem **Testkonto** — Mail-Domain auf
`.invalid` oder Adresse in der Umgebungsvariablen `EXPRESS_TESTKONTEN` (kommagetrennt; auf Staging die E2E-Konten aus
`~/.e2e.env`, eingetragen in `.env.staging`) —, gehen alle Team-Mails dazu (neuer Auftrag, Antwort, Erinnerung,
überfällig, Storno, entfallen) **nur an `support@inklutec.de`** (`EXPRESS_TEST_MAIL`), nie an die Bearbeiter
(`express.ist_testkonto`, `team_empfaenger(standard, auftrag)`). Steht in den Einstellungen eine Team-Adresse auf
`.invalid` (Testreihen), geht die Mail dorthin, also an niemanden. Fehlt die Adresse des Kunden, gilt der Auftrag
vorsichtshalber als Testauftrag. Echte Kunden (auch Michaels eigene Aufträge) benachrichtigen weiter alle wie oben. Die
Testreihen prüfen am Ende im Log, dass keine Mail an einen Bearbeiter ging.

## Datenmodell (`database.init_db`)

- `express_auftraege`: ein Auftrag (Runde 3: `auftrag_name`, `kunde_geloescht_am`, `meldung_fehlversuche`); Status
  `entwurf` (= Warenkorb, höchstens einer je Konto, eindeutiger Index) |
  `neu` | `in_arbeit` | `rueckfrage` | `geliefert` | `storniert`. Kunde, zahlender Topf, Angaben, Seiten, Credits, Frist,
  Fälligkeit, Zustimmung (Wortlaut des Häkchens, Fassung `express.ZUSTIMMUNG_FASSUNG`, Sprache, Zeitpunkt,
  Absender-Kennung; Spalte `zustimmung_bearbeitung` ist ab Fassung -3 leer und bleibt für alte Aufträge),
  Idempotenz-Schlüssel (eindeutig je Konto), Bearbeiter, Liefer- und Storno-Angaben, interne Notiz,
  Erinnerungs-Vermerke mit Zähler der Fehlversuche (`meldung_fehlversuche`), Rückfrage-Beginn, Name des Kunden für den
  Auftrag (`auftrag_name`, leer = „Auftrag <Nr>“), `kunde_geloescht_am` (Kunde hat gelöscht; die Zeile bleibt als
  Buchungsnachweis, Kundensicht filtert sie aus).
  Status `bestellung` = Korb während des Bestellens (nicht vorgemerkt, nicht sichtbar).
- `express_positionen`: je Dokument Quelle (Projekt, Dokument, Name als Momentaufnahme), **Dateityp** (`dateityp`,
  Schlüssel aus `express.DATEITYPEN`, Migration 05.10.2026, Standard `pdf`), Seiten, Leistung, Credits, Pfade für
  Original, Ergebnis, Prüfbericht, Ergebnis der automatischen Prüfung (Spalte `verapdf`: leer, `{"laeuft": Kennung,
  "seit"}`, `{"nicht_geprueft": true}` oder `{"bestanden", "zusammenfassung", "regeln_fehlgeschlagen"}`).
- `express_verlauf`: Schritte mit Zeit, Person, Text; `fuer_kunde = 0` für interne Schritte.
- Dateien: `results/<konto>/_express/<auftrag>/pos<id>_(original|ergebnis|bericht)<endung>` — Pfade nur aus Zahlen und der
  Endung des Dateityps, nie aus Dateinamen. Die Originale werden beim Bestellen kopiert (unveränderte Kundendatei `roh_path`, sonst die
  Arbeitsdatei): Was die Profis bekommen, ändert sich nicht mehr, auch wenn der Kunde das Projekt weiter bearbeitet oder
  löscht.
- Konto löschen: Aufträge, Positionen, Verlauf und der Ordner gehen mit; war das Konto Bearbeiter, bleibt nur der Name;
  war es zahlender Topf, werden offene Aufträge anderer Konten storniert (Befund 6), abgeschlossene verlieren den
  Topf-Bezug.
- **Aufbewahrung** (Befund 14): Einstellung „Dateien löschen nach Tagen“ (`aufbewahrung_tage`, Standard 0 = nichts
  löschen, bis Steve entscheidet). Ist sie gesetzt, löscht die Schleife Originale, Ergebnisse und Prüfberichte so viele
  Tage nach Lieferung bzw. Storno; Auftrag, Positionen und Verlauf bleiben als Nachweis (Verlauf „Dateien gelöscht“).
- **Absender der Zustimmung** (Befund 14): gespeichert wird nur das gekürzte Netz (IPv4 /24, IPv6 /48; IPv4 in
  IPv6-Form wie IPv4, N7; `express.netz_kurz`) — genug als Indiz, ohne den einzelnen Anschluss festzuhalten. Es
  bleibt mit dem Auftrag als Nachweis der Zustimmung, auch wenn der Kunde den Auftrag löscht.

## Sicherheit

- Jede Kundenfunktion prüft den Besitz **im SQL** (Auftrag/Position/Dokument + `user_id`), Fremdes antwortet 404 —
  geprüft für Projekt-Dokumente, Auswahl, Positionen, Aufträge, Antwort, Downloads und Nachweis, Umbenennen
  (`POST /api/express/auftraege/<id>/name`) und Löschen (`DELETE /api/express/auftraege/<id>`, zusätzlich nur bei
  geliefert/storniert, sonst 409). Vom Kunden gelöschte Aufträge gibt es für ihn nicht mehr (404 überall).
- Rechte frisch aus der Datenbank je Anfrage (`get_current_user` liest `is_admin`, die Bearbeiter-Prüfung liest
  `express_bearbeiter` und `is_active`).
- Uploads: Dateityp am Inhalt erkannt (`Dateityp.erkennen`, bei PDF `%PDF-`; beim Kunden zusätzlich die Endung), Größe
  (Kunde 50 MB, Ergebnis 100 MB), lesbar und ohne Öffnungspasswort (`Dateityp.seiten`, bei PDF PyMuPDF), Vorprüfung wie
  beim normalen PDF-Upload. Kaputter oder falsch geformter JSON-Körper: 400 (Befund 8). Dateinamen werden nur angezeigt (Pfadteile und
  Steuerzeichen entfernt), Downloads mit `Content-Disposition: attachment` (RFC 6266/5987) und `nosniff`.
- Doppelbestellung: Knopf sperrt im Browser, Server über Idempotenz-Schlüssel (eindeutiger Index) und eine Sperre um
  Prüfen-und-Vormerken; Doppel-Lieferung über Vergleichen-und-Tauschen in einer Transaktion.
- Bremsen je Konto: Bestellen 10/h, Upload 30/h, Antwort 20/h, Nachweis-PDF 30/h, Originale-ZIP 20/h (429). Der
  Nachweis läuft in einem eigenen kleinen Thread-Pool mit 60 Sekunden Zeitlimit zum Umwandler (Befund 16); die
  Dokumentgrenze wird vor dem Öffnen der PDFs geprüft.
- **Kein Einrahmen** (Nachprüfung N6, ganze App): jede Antwort trägt `X-Frame-Options: DENY` und
  `Content-Security-Policy: frame-ancestors 'none'` (ASGI-Schicht `_RahmenSchutz` in `main.py`). Die App bettet sich
  nirgends selbst ein; vor einem Prod- oder Demo-Rollout prüfen, ob eine andere Seite die Demo einrahmt.
- Seiten mit Bestell-Zustand senden `Cache-Control: no-store`; kommt eine Seite doch aus dem Zurück-Speicher
  (`pageshow`), erneuert sie Stand und Idempotenz-Schlüssel.
- CSRF wie im Bestand: Sitzungs-Cookie `SameSite=Lax`, JSON-Anfragen; Multipart-Uploads gehen ohne Cookie nicht durch.
- Fehlertexte ohne Interna (Pfade, SQL); Details nur im Server-Log.

## Bewusste Entscheidungen

- **Keine Uhrzeit** für den Kunden (Steve); intern Fälligkeit mit Uhrzeit für Erinnerung und „überfällig“.
- **Nachweis statt Rechnung:** Die Rechnung entsteht beim Credit-Kauf; eine zweite Rechnung würde die Umsatzsteuer
  doppelt ausweisen. Die Übersicht sagt das ausdrücklich. (Bitte von Michael/Steuerberater bestätigen lassen.)
- **Kein Bestellen über den Chatbot** (Abweichung vom Grundsatz „Chatbot = Oberfläche“): Eine zahlungspflichtige
  Bestellung braucht den gesetzlich beschrifteten Knopf und das bewusst gesetzte Häkchen. Der InkluAgent kennt den
  Express-Service in Stufe 1 noch nicht; ein Lese-Werkzeug („Stand meiner Aufträge“) wäre unkritisch und kann folgen.
- **Lieferung in die Auftragsübersicht, nicht als neue Fassung im Projekt** (Abweichung von der Skizze): InkluDocs
  kennt je Dokument genau eine Arbeitsdatei, an der Tags, Alt-Texte und Quickinfos des Kunden hängen. Eine „neue
  Fassung“ gibt es nicht; das Ergebnis darüber zu legen, würde den Arbeitsstand des Kunden still ersetzen. Darum bleibt
  das Projekt unberührt, und die fertigen Dateien stehen in der Auftragsübersicht (Link in der Liefer-Mail, auf der
  Startseite und unter „Meine Aufträge“). Eine echte Versionierung am Dokument ist ein eigener Schritt.
- **Nur PDF** in Stufe 1 (Steve 05.10.2026: vorerst nur PDF, aber jederzeit erweiterbar) — siehe „Erweitern um neue
  Dateitypen und Leistungen“. Die Oberfläche sagt nirgends „derzeit nur PDF“, sondern positiv, was geht („PDF-Datei
  auswählen“, Fehler „Bitte wähle eine PDF-Datei aus.“; Michael Karbe 05.10.2026, Punkt 6). Gleiches gilt für den
  Upload in ein Projekt: die Meldung passt zum Projekt (PDF-, Word- oder Bild-Projekt).
- **„Nur prüfen (Prüfbericht)“ ist abgeschaltet** (Michael Karbe 05.10.2026, Punkt 3), nicht gelöscht: In
  `express.LEISTUNGEN` steht `aktiv=False`. Abgeschaltete Leistungen sind nicht wählbar, fehlen in Seite, Stand und
  Einstellungen (ihr Preis bleibt in `preise` gespeichert), und eine solche Position in einem alten Warenkorb wechselt
  beim Laden auf die Standard-Leistung ihres Dateityps. Bestehende Aufträge mit „Nur prüfen“ laufen unverändert weiter.
  Wiedereinschalten: `aktiv=True` setzen — Leistungswahl je Dokument und Preisfeld erscheinen dann von selbst.
- **Bedingungen nur auf Deutsch** wie die übrigen Rechtstexte (I18N.md).
- Express-Credits zählen in der Verwaltung „KI-Kosten“ nicht zur Kennzahl „Kosten je Credit“ (Handarbeit, keine KI).

## Erweitern um neue Dateitypen und Leistungen

Alles Typ- und Produkt-Spezifische steht in **einer** Liste in `backend/express.py`; Warenkorb, Preise, Einstellungen,
Uploads, Lieferung, Abbuchung, Downloads, Mails und Oberfläche lesen daraus. Nichts davon ist im Ablauf fest verdrahtet.

- **`Dateityp`** (`DATEITYPEN`): `schluessel` (gespeichert je Position), `name` (Anzeige, mit `N_()` für die Kataloge),
  `endung` (gespeicherte Dateien und Downloads), `endungen` (Vorfilter beim Hochladen), `mime`, `accept` (Dateifeld),
  `projekt_typen` (welche `projects.project_type` in der Projektauswahl erscheinen), `erkennen(kopf)` (am Inhalt, nie nur
  am Namen), `seiten(pfad)` (Preisgrundlage; wirft `ExpressFehler` bei unlesbarer oder geschützter Datei), optional
  `pruefen(pfad)` (automatische Prüfung eines Ergebnisses, Rückgabe `{"bestanden", "zusammenfassung",
  "regeln_fehlgeschlagen"}` oder `None`) und `pruef_name` (z. B. „veraPDF“).
- **`Leistung`** (`LEISTUNGEN`): `schluessel`, `name`, `aktiv` (`False` = abgeschaltet, bleibt in der Liste, siehe
  „Bewusste Entscheidungen“), `preis_standard` (Credits je Seite, bis zur Einstellung), `grundpreis_standard` (Credits
  je Dokument, Standard 0),
  `dateitypen` (erlaubte Originale), `aktion` (Name in `usage_events` beim Abbuchen), `ergebnis_pflicht`,
  `bericht_pflicht`, `ergebnis_typen` (leer = wie das Original), `bericht_typen` (Standard PDF), `ergebnis_zusatz`
  (Zusatz im Download-Namen, Standard „ (barrierefrei)“).

**Beispiel Word** (heute auskommentiert in `DATEITYPEN`):
1. `Dateityp(schluessel="docx", name=N_("Word"), endung=".docx", endungen=(".docx",), mime=…, accept=".docx",
   projekt_typen=("docx",), erkennen=<ZIP mit word/document.xml>, seiten=<Seiten zählen>)` eintragen. Für die Seiten
   gibt es zwei Wege: aus `docProps/app.xml` (schnell, aber nur so genau wie die letzte Speicherung in Word) oder per
   Umwandlung mit dem Konverter (`pdfua_export`) — Steve/Michael entscheiden, was als Preisgrundlage gilt.
2. Eine Leistung erlaubt `"docx"` — eine bestehende (`dateitypen=("pdf", "docx")`) oder eine neue, z. B.
   „Word barrierefrei aufbereiten“ mit `ergebnis_typen=("docx",)` oder `("pdf",)`, wenn als PDF/UA geliefert wird.
3. Hochladen: geschieht im Projekt (`/api/upload` kann Word schon); der Express-Service liest die Projekte über
   `Dateityp.projekt_typen`. Ein Hochladen ohne Projekt gibt es seit Runde 7 nicht mehr.
4. Übersetzungen der neuen Namen in alle sechs Kataloge (`backend/locales/*`), `scripts/check_i18n.py`.
5. Preise in der Verwaltung festlegen (das Feld je Leistung erscheint von selbst), Bedingungen anpassen
   (`ZUSTIMMUNG_FASSUNG` hochzählen).
6. Tests: Vorlage ist `tests/test_express.py`, Klasse `Erweiterbar` — sie hängt im Test einen zweiten Dateityp mit eigener
   Leistung ein und prüft Korb, Leistungswahl je Typ, Preise, Bestellung, Upload, Liefern, Abbuchung und Download.

**Weitere Produkte** (z. B. „Übersetzen lassen“, „Formular aufbereiten“) sind neue `Leistung`-Einträge; Preis, Pflicht-
Dateien und Abbuchungs-Aktion kommen aus dem Eintrag. Eine neue Leistung mit anderer Preislogik als „je Seite“ bräuchte
zusätzlich eine eigene Preisfunktion in `express.preis`.

## Offene Rechtspunkte

1. **Widerruf (§ 356 Abs. 4 BGB):** Für Verbraucher beginnt die menschliche Dienstleistung sofort. Ist der
   Express-Service ein eigener Dienstleistungsvertrag, fehlt ein ausdrückliches Verlangen auf Beginn vor Ablauf der
   Widerrufsfrist mit Bestätigung der Kenntnis vom Erlöschen des Widerrufsrechts (Prüfung Entwicklung, Befund 17). Mit
   dem Bedingungstext klären; ggf. ein zweites Häkchen nur für Verbraucher.
2. **AVV und Partner:** Datenschutzerklärung und AVV müssen die menschliche Bearbeitung durch InkluTec und Actino
   abdecken — Kategorien (Dokumente **und** Kontaktdaten: Ansprechpartner, Telefon, Hinweise, E-Mail-Adresse, die der
   Express-Bearbeiter sieht und per Team-Mail bekommt), Zweck, Speicherdauer (Einstellung „Dateien löschen nach Tagen“),
   Vertraulichkeitsvereinbarung mit dem Partner. Die Bedingungen (Fassung `2026-10-05-entwurf-3`) nennen die
   Kontaktdaten und sagen ausdrücklich, dass Mitarbeiter von InkluTec und Actino die Dokumente sehen und bearbeiten;
   seit Runde 3 gibt es dafür kein eigenes Häkchen mehr (Michael Karbe 05.10.2026, Punkt 5) — die Zustimmung läuft über
   das eine Häkchen „Ich akzeptiere die Bedingungen“. Ob das für die Einwilligung genügt, mit dem Rechtstext klären.
3. **Nachweis statt Rechnung** und Umsatzsteuer: siehe „Bewusste Entscheidungen“; von Michael/Steuerberater bestätigen
   lassen. Zur Kennzahl „Bleibt nach KI-Kosten“ und § 13b UStG siehe `docs/KI_KOSTEN.md`.
4. **Bedingungstext** abstimmen und rechtlich prüfen, dann `ZUSTIMMUNG_FASSUNG` hochzählen und „ENTWURF“ entfernen.

## Offen vor Prod (nur auf Steves Wort)

1. **Wer bearbeitet?** — siehe „Offene Rechtspunkte“ 2.
2. **Preise:** Michaels Richtpreis 50 Credits je Seite plus 100 je Dokument ist eingestellt (06.10.2026); **Frist**
   (48 Stunden oder 2 Werktage — Wochenenden) offen.
3. **Aufbewahrung:** Frist für Originale und Ergebnisse festlegen (Einstellung vorhanden, heute 0 = nie löschen).
4. **Verrechnung InkluTec ↔ Actino** für Express-Aufträge.
5. Prod: `EXPRESS_SERVICE=an` in `docker-compose.yml` setzen, Rollout wie üblich. Vorher die Migrationen
   `express_positionen.dateityp`, `preis_seite`, `grundpreis`, `express_auftraege.auftrag_name`, `kunde_geloescht_am`,
   `meldung_fehlversuche` (laufen beim Start von selbst, idempotent). Der Rücksprung nach der Anmeldung
   (`weiterleitung.py`) gilt app-weit und kommt mit dem Rollout auch auf Prod.
6. **Einrahmen-Schutz** (`X-Frame-Options: DENY`, `frame-ancestors 'none'`) gilt app-weit: vor Prod und Demo prüfen,
   ob irgendeine Seite (z. B. inklutec.de) die Demo oder die App in einem iframe zeigt.

Spätere Stufen: Erinnerung/Rückfragen ausbauen, Warenkorb über mehrere Projekte komfortabler, Word/PowerPoint (siehe
„Erweitern“), Ergebnis als neue Fassung am Dokument im Projekt (braucht Versionierung), Lese-Werkzeug im InkluAgent.

## Dateien

- Kern mit `DATEITYPEN`/`LEISTUNGEN`: `backend/express.py`; Endpunkte, Mails, Erinnerungsschleife, Kontolöschung:
  `backend/express_api.py`; Guthaben: `backend/billing.py` (`guthaben`, `_express_bindung`, `vorgemerkt`, `vorgemerkt_domain`,
  `pakete_abbuchen_fuer_monat`, `EXPRESS_OFFEN`); Schalter: `backend/funktionen.py` (`EXPRESS`)
- Rücksprung nach der Anmeldung: `backend/weiterleitung.py` (`sicheres_ziel`, `login_adresse`), genutzt in
  `main._login_umleitung` und im Anmeldeformular `templates/index.html`; `zurAnmeldung()` in `frontend/dashboard.js`
- Seiten: `templates/express.html` (zwei Ansichten: „Meine Aufträge“ und Express-Warenkorb), `express_auftrag.html`,
  `express_bedingungen.html`, `verwaltung_express.html`,
  `verwaltung_express_auftrag.html`; Hochlade-Komponente `frontend/hochladefeld.js` (Vorbild `app.html`
  `uploadBlockHtml`/`setupProjectDropzone`); Fehler am Feld in der Verwaltung `frontend/verwaltung.js` (`feldFehler`,
  `serverFeldFehler`); Link im Projekt `frontend/dokument.js`; Startseite `templates/dashboard.html`; Seitenleiste
  `frontend/dashboard.js`; Druck-CSS `frontend/dashboard.css`

## Tests

- `tests/test_express.py` (Unit, eigene Wegwerf-Datenbank, 117): Warenkorb, Fremd-Zugriffe, Grenzen, Bestellen mit
  Vormerkung und Idempotenz (auch 4 gleichzeitige Klicks), Vormerkung sperrt andere Ausgaben, Liefern bucht genau einmal
  (auch 4 gleichzeitig), Storno, Dateinamen nie Pfad, Upload-Prüfung, Downloads erst nach Lieferung, Frist ruht bei
  Rückfrage, Erinnerung/Überfällig je einmal, Nachweis-OOXML, Kontolöschung, Einstellungen, Bearbeiter, Schalter.
  Klasse `Korrektur`: je Befund der Prüfungen vom 05.10.2026 ein Test (Monatswechsel mit Übertrag und Paket-Überhang,
  eingefrorener Korb mit zweitem Tab, Preis-/Auswahländerung 409, alter Schlüssel 409, Domain-Topf, fail-closed,
  Topf-Inhaber gelöscht, Erinnerung erneut ohne Doppelmails, Upload nach Lieferung, laufende/hängende Prüfung,
  Aufbewahrung, IP-Netz, Frist ruht in der Anzeige, Vormerkung im Guthaben-Text, Fehler am Feld, alte Preis-Schlüssel,
  Einzahl/Datum/Tausenderpunkt, Kundentext der Prüfung). Klasse `Erweiterbar`: zweiter Dateityp mit eigener Leistung.
  Klasse `Runde3`: Umbenennen (Name in Liste, Übersicht, Nachweis, Grenzen), Löschen nur geliefert/storniert mit
  Buchungsnachweis und 404 danach, Dokumente in der Liste, „Nur prüfen“ abgeschaltet (nicht wählbar, alte Position
  wechselt), ein Häkchen, positive Meldungen, N1 (laufender Monat beim Liefern abgeglichen), N2 (Vormonats-Vormerkung,
  Single vor und nach der Lieferung gleich), N3, N4, N7, Prüftext für Bearbeiter. Hilfsfunktion `_pruefen_an()` schaltet
  „Nur prüfen“ für die alten Tests im Test wieder ein. Klasse `Runde4` (Nachkontrolle Runde 3): der Kunde verbraucht
  immer wieder, was angezeigt wird, bis 0 — Summe genau wie Monatsbudget + Pakete abzüglich Bestellung (Single, auch in
  kleinen Schritten, Free-Einzelkonto, Free-Domain, Team-Topf), Storno nach dem Monatswechsel kostet nichts, Lieferung
  ändert das Guthaben nicht, Auftrag über zwei Monatswechsel, Sperre und Meldung nennen dieselbe Zahl (R2), ohne
  Vormerkung alles wie bisher. Klasse `Runde5` (Free-Domain mit Paketen und ein Stand) und `Runde6` (Storno-Ausgleich): siehe
  `backend/ABRECHNUNG.md`. Klasse `Runde7`: Standardpreis 50 + 100, Zusammensetzung gespeichert und gebucht, geänderter
  Grundpreis 409, offene Aufträge behalten ihren Preis, Grundpreis-Prüfung und 0, Free-Domain mit Grundpreis, Mails
  nennen das Konto, Nachweis mit Zusammensetzung. Die älteren Klassen rechnen mit Grundpreis 0 (in `Basis` gesetzt).
  Klasse `Runde8`: Testkonto erkennen (.invalid, Umgebung), Testauftrag nie an Bearbeiter, echter Kunde weiter an alle,
  Zahlenfelder nur mit Tausenderpunkt.
- `tests/test_weiterleitung.py` (4, mit Normalisierung): Rücksprung nur auf interne Pfade (kein `//host`, Schema, Backslash, `/login`).
- `tests/e2e/verify_express.py` (im Staging-Container über HTTP, 144): ganzer Ablauf inkl. Nachweis-PDF und ZIP,
  IDOR-Fälle, Rechte (Kunde, Nur-Einsicht, Bearbeiter, Voll-Admin), Uploads, Doppel-Bestellung/-Lieferung, Storno,
  dazu Preisänderung und alter Schlüssel (409), kaputter JSON-Körper (400), Feldfehler der Einstellungen, no-store,
  Projektliste, ZIP_STORED, Prüfergebnis für Kunden; Abschnitt G2 Umbenennen/Löschen (fremd 404, laufend 409, danach
  404, Verwaltung sieht Nachweis), G3 zweiter Tab 409 „veraltet“ und Einrahmen-Kopfzeilen auf `/express`,
  `/express/warenkorb`, `/app`, `/api/express/stand`; G4 Vormonats-Bestellung: `/api/me` nennt die bindende
  Vormerkung, Rest + Zusatz-Credits − vorgemerkt = verfügbar = Sperre; Runde 7: Hochladen ohne Projekt 410, Dokumente
  über `/api/upload` ins Projekt, B2 Preis 50 + 100 mit Teilsummen und Feldfehler des Grundpreises, R7 „Meine Aufträge“
  ohne Bestellformular, Warenkorb ohne Hochladen, `/express?projekt=` → Warenkorb, Anmeldung mit Rücksprung (auch mit
  Abfrage), nur interne Ziele, fremdes Konto 404, gespeicherte Preis-Zusammensetzung
- `tests/e2e/ui_express.py` (Playwright + axe, nur Staging, 111; Runde 8: axe heading-order, Fokus „fremdes Konto“, Hinweis am E-Mail-Feld): Link im Projekt, Auswahl, Hochlade-Komponente
  (Etikett-Knopf, Fokusring, Dateiname in der Statuszeile, Fehler am Feld) beim Kunden und in der Verwaltung,
  Leistung entprellt, Entfernen mit Meldung und Fokus, Pflichtfelder und Häkchen mit Fehler am Feld, Fokusring an
  „Zahlungspflichtig bestellen“, Rahmen der Eingabefelder, keine Ansage beim Laden, Bestellen, Danke-Meldung (nicht im
  Druck), Startseite, Verwaltung (Liste mit Überschrift im summary, Feldfehler der Einstellungen, Dialoge mit
  Beschreibung und Pflichtfeld, Upload, Liefern), Download, PDF-Fehler neben dem Link; Runde 3: keine Leistungswahl,
  nur Summen, kein „nur PDF“, ein Häkchen (Fehler verschwindet beim Ankreuzen), Fokusring per Tab, eindeutige
  Hochlade-Namen ohne Landmarke, Auftragskarten mit Umbenennen-/Lösch-Dialog, Verwaltung sieht „vom Kunden gelöscht“,
  Fokusring an Feldern und Kästchen auf Anmelden, Registrieren und Passwort vergessen; Runde 4: stornierter Auftrag
  daneben, Abzeichen „Storniert“ mit #475569 (6,9:1); Runde 7: Navigation „Meine Aufträge“, Link aus dem Projekt in den
  Warenkorb (Fokus Schritt 1), kein Hochladefeld, Preis und Zusammensetzung, nach dem Bestellen Liste mit offenem Auftrag
  und Danke-Meldung, ein Auftrag allein ist zu, Sprung auf eine Karte, alter Link `#h-auftraege` und `?neu=1`, Angaben/
  Einverständnis/Verlauf aufklappbar und beim Drucken offen, Startseite „Meine Aufträge“, Verwaltung mit Grundpreis-Feld,
  Anmeldung mit Rücksprung, fremdes Konto mit „Abmelden und anders anmelden“, kein offener Redirect — axe 0 Verstöße
- `tests/e2e/ui_express_korb.py` (25): Knopf am Dokument, Navigationseintrag in drei Modi, H1 „Express-Warenkorb“ auf
  `/express/warenkorb` passend zum Seitentitel (N5)
- `tests/e2e/ui_bildschirmfotos.py`: Fotos vorher/nachher der globalen CSS-Änderungen (Projekt-Hochladefläche, Knöpfe,
  Formulare) mit berechneten Stilen
- `verify_abo3.py` (Server, Team-Reihe) prüft die Vormerkung im Team-Topf (`/api/team`), nach dem Monatswechsel
  dieselbe Zahl wie das Guthaben des Mitglieds
- `tests/e2e/ui_smoke.py` kennt `/express`, `/express/bedingungen`, `/verwaltung/express`
