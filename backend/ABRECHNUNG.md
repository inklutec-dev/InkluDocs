
## PLANWECHSEL (07.08.2026, Steve nach dem Michael-Meeting)

Fachregeln — gelten fuer beide Buchungswege, technisch umgesetzt NUR fuer
Stripe-Abos (Rechnungskunden stellt der Admin um, siehe unten):

### Upgrade (teurerer Plan ODER laengere Laufzeit beim gleichen Plan)
- Wirkt SOFORT. Es startet eine NEUE Laufzeit ab heute
  (billing_cycle_anchor='now'), der neue Plan wird also voll berechnet.
- Bereits bezahlte Restzeit verfaellt NICHT: Stripe rechnet sie als
  Gutschrift an (proration_behavior='create_prorations'). Der Kunde zahlt
  den vollen neuen Preis abzueglich dieser Gutschrift.
- Der Kunde bekommt SOFORT das volle neue Monatskontingent: users.
  kontingent_reset_am wird auf jetzt gesetzt, billing.monats_verbrauch
  zaehlt ab da frisch (der Wert wirkt nur im Monat der Umstellung —
  spaetestens am Monatsersten gewinnt wieder der Monatsanfang).
- ZAHLUNG WIRD GEPRUEFT: Bleibt die Differenz-Rechnung offen (Karte
  platzt), wirft stripe_zahlung.ZahlungOffen; der Endpunkt antwortet 402
  und schaltet NICHTS frei. Der invoice.paid-Webhook holt den Plan nach,
  sobald bezahlt wurde.

### Downgrade (guenstigerer Plan ODER kuerzere Laufzeit)
- Wird nur VORGEMERKT (users.geplanter_plan/geplante_laufzeit/geplant_ab)
  und wirkt zum Ende der bezahlten Laufzeit. Bis dahin aendert sich nichts.
- Bei Stripe als Subscription-Schedule (Phase 1 unveraendert, Phase 2 neuer
  Preis). VOLLZUG bei Stripe-Abos ueber den invoice.paid-Webhook (dort ist
  die Zahlung belegt); der Tageslauf vollzieht nur Rechnungsweg-Konten.
- geplant_ab ist ein FESTER Stichtag: plan_gueltig_bis kann sich durch
  Webhooks verschieben, der Wechsel muss trotzdem am vorgemerkten Datum
  greifen.
- Widerruf moeglich, solange der Stichtag nicht erreicht ist (danach 409).

### Downgrade auf Free
- Gibt es NICHT als Plan-Wechsel — das ist die Kuendigung
  (/api/abo/kuendigen), Ergebnis identisch: am Laufzeitende Free.

### Team-Folgen
- Verliert das Konto durch den Wechsel die Sitze (Team/Enterprise ->
  Single), wird das Team am Stichtag aufgeloest. Die Mitglieder erfahren
  das SOFORT beim Vormerken: Mail + Zeile „Team endet am ..." in ihrem
  Abo-Bereich (/api/me liefert endet_am je Team-Kontext).

### Rechnungsweg (Actino/INKLUTEC)
- Konten ohne Stripe-Vertrag koennen NICHT selbst wechseln (sonst waere
  „Enterprise gratis" ein API-Aufruf) — der Endpunkt lehnt mit 400 ab, die
  Oberflaeche zeigt den Bereich gar nicht. Umgestellt wird ueber
  /benutzer -> „Abo zuweisen".
- Auf der Abo-Seite steht der Weg zum Partner: sales@actino.de
  (mailto-Link mit Betreff „InkluDocs-Anfrage").

### Doppelklick / Nebenlaeufigkeit
- Knopf wird waehrend des Requests gesperrt; der Stripe-Aufruf traegt
  einen Idempotency-Key aus Subscription + Plan + Laufzeit + Minute.
- Der Tageslauf-Vollzug raeumt die Vormerkung per bedingtem UPDATE ab —
  nur ein Lauf gewinnt.

### Stripe-Ereignisse koennen sich ueberholen
- customer.subscription.updated wird NICHT dem Payload geglaubt: der
  Zustand wird frisch bei Stripe abgefragt. Sonst konnte ein verspaetetes
  „Schedule geloest"-Ereignis eine Kuendigung stillschweigend aufheben.

## TAEGLICHER ABO-LAUF: ANSTOSS VON AUSSEN (10.08.2026)

### Warum es das gibt
Der Abo-Tageslauf `_abo_tageslauf()` erledigt vier Dinge: die Erinnerung
14 Tage vor einer Verlaengerung (an Kunde UND Betreiber, weil daran die
Rechnung haengt), die Auto-Verlaengerung am Stichtag, den Vollzug vorgemerkter
Plan-Wechsel bei Konten auf Rechnungsweg, und den Rueckfall auf Free bei
abgelaufenen Abos.

Bis zum 10.08.2026 wurde er an GENAU EINER Stelle angestossen: in `/api/me`,
also wenn ein angemeldeter Nutzer die App oeffnete. Belegt per Suche — die
uebrigen Fundstellen im Quelltext sind Kommentare. Solange nichts kostete,
war das folgenlos. Mit echtem Geld nicht mehr: Meldet sich einen Tag lang
niemand an, verschieben sich Erinnerungen und Verlaengerungen um einen Tag.
Die 14-Tage-Erinnerung ist aber eine Zusage aus den Nutzungsbedingungen.

NICHT betroffen ist das monatliche Guthaben. Das wird beim Zugriff
ausgerechnet (`kontingent_reset_am` gegen den Monatsanfang, siehe
`_abrechnungsbeginn_sql`), nicht vom Tageslauf gesetzt. Dort kann also auch
bei tagelangem Stillstand nichts verlorengehen.

### Was gebaut wurde
`POST /api/intern/tageslauf`
- Berechtigung ueber `TAGESLAUF_TOKEN` aus der Umgebung, mitgegeben im Kopf
  `X-Tageslauf-Token`, verglichen mit `secrets.compare_digest`.
- Ohne eingerichteten Token antwortet der Endpunkt 503 statt offen zu stehen.
- Startet den vorhandenen Lauf in einem Thread und wartet hoechstens 20
  Sekunden auf ihn: So enthaelt die Antwort im Normalfall schon den frischen
  Abschluss, ohne dass ein Waechter in seinen Zeitablauf laeuft. Dauert es
  laenger, laeuft der Lauf zu Ende und der naechste Aufruf sieht das Ergebnis.
- Antwort: `angestossen` (hat DIESER Aufruf den Lauf gestartet?), `lauf_datum`
  (wann war der Lauf zuletzt wirklich dran?), `letzter_abschluss`,
  `letzter_fehler`.

Neue Merker in `system_kv`, geschrieben von `_kv_setzen`, gelesen von
`_kv_lesen`:
- `abo_tageslauf_fertig` — Zeitpunkt des letzten vollstaendigen Durchlaufs.
- `abo_tageslauf_fehler` — Zeitpunkt des letzten Fehlschlags.

### Zwei Dinge, die erst beim Bauen sichtbar wurden
1. `lauf_datum` gehoert in die Antwort. Ohne dieses Feld kann ein Waechter
   „heute schon erledigt" nicht von „seit Tagen laeuft nichts" unterscheiden —
   beide Faelle liefern nur `angestossen: false`. Genau das ist beim ersten
   Test passiert und haette als Dauer-Fehlalarm geendet.
2. Nach einem Fehlschlag wird der Tag WIEDER FREIGEGEBEN (Prozessmerker auf
   None, `abo_tageslauf` auf "fehlgeschlagen"). Vorher galt ein abgebrochener
   Lauf als erledigt: Weder Cron noch `/api/me` haetten es an dem Tag nochmal
   versucht, die Erinnerungen waeren still ausgefallen. Ein zweiter Anlauf am
   selben Tag ist gefahrlos, weil jede einzelne Mail ueber `_kv_einmal` gegen
   Doppelversand gesichert ist.

### Selbstschutz, der unveraendert bleibt
Ein Lauf je Tag, atomar ueber den `system_kv`-Eintrag `abo_tageslauf`; dazu
ein prozesslokaler Merker, damit `/api/me` nicht bei jedem Aufruf die
Datenbank anfasst. Mehrfache Anstoesse koennen nichts doppelt tun. Der Weg
ueber `/api/me` bleibt als zweites Standbein bestehen.

Zu beachten: Der prozesslokale Merker bedeutet, dass sich nach dem ersten
gueltigen Aufruf an diesem Tag kein weiterer Lauf mehr ausloesen laesst —
auch nicht zum Testen. Fuer einen erzwungenen Testlauf den Container neu
starten und den `system_kv`-Eintrag zuruecksetzen; genau das macht
`verify_tageslauf.py`.

### Betrieb (Staging)
- `/home/claude/abo_tageslauf_anstoss.sh`, Rechte 700; Token in
  `/home/claude/.tageslauf-token`, Rechte 600 — bewusst NICHT in der Crontab.
- Cron: `15 3 * * *`, Protokoll unter `/home/claude/logs/abo_tageslauf.log`.
- Uptime Kuma als zweiter Waechter ist vorgesehen, aber noch nicht
  eingerichtet (braucht einmal Zugang zur Oberflaeche).
- Fuer Produktion sind beim Promote zu setzen: `TAGESLAUF_TOKEN` in der
  Prod-Umgebungsdatei, ein Cron-Eintrag gegen Port 8001, und die Antwort
  einmal von Hand pruefen.

### Mailversand auf Staging (Steve, 10.08.2026)
In der Staging-Datenbank stehen Adressen bei actino.de (`sales@`, `karbe@`,
`info@`). Das sind BEWUSST angelegte Testkonten — sie existieren genau dafuer,
den Mailweg echt zu pruefen. Ein Lauf, der dorthin schreibt, ist also kein
Unfall, sondern der Zweck.

`verify_tageslauf.py` haelt sich trotzdem zurueck: Es zaehlt vorher die
faelligen Konten und ueberspringt den erzwungenen Durchlauf, wenn welche da
sind. Das ist absichtlich konservativ, damit ein beilaeufiger Testlauf nicht
ungefragt Post ausloest. Wer den Mailweg wirklich pruefen will, setzt ein
Konto faellig und stoesst den Lauf von Hand an. Am 10.08.2026 war keines
faellig (naechster Ablauf 07.11.2026).

### Geprueft
`verify_tageslauf.py`: 12 Pruefungen, 0 Fehler — Zugangsschutz (ohne Token,
falscher Token, Token mit Anhaengsel: je 401), Antwortform, echter Durchlauf
mit Abschluss von heute, zweiter Aufruf loest nichts erneut aus.
Danach der volle Satz: verify_wechsel 63, verify_abo3 68, verify_stripe 32,
verify_sicherheit2 17, verify_loeschen 43, verify_recht 54, verify_admin 54 —
alle 0 Fehler; i18n 820 Strings in 6 Katalogen.

## MONATSABO (19.08.2026, Steves Vorgabe)

Jede Bezahlstufe gibt es zusaetzlich als Monatsabo (laufzeit_monate = 1)
mit rund 20 % Aufschlag: Single 11,95 / Team 23,95 / Enterprise 59,95 EUR
(billing.PLAN_PREISE_MONATLICH_EUR, Helfer billing.preis_pro_monat).
Regeln:
- NUR online ueber Stripe buchbar. Der Admin-/Rechnungsweg validiert gegen
  PLAN_LAUFZEITEN_RECHNUNG = (3, 6, 12) — Michael stellt keine
  Monatsrechnungen.
- Stripe-Preis: lookup_key idoc_abo_<plan>_1, recurring interval=month,
  interval_count=1; sichere_produkte() legt ihn idempotent an.
- Verlaengerung/Abbuchung macht Stripe monatlich (invoice.paid verschiebt
  plan_gueltig_bis um 1 Monat); Kuendigung wie gehabt ueber
  cancel_at_period_end = wirksam zum Ende des laufenden Monats. Damit ist
  das Monatsabo von sich aus konform mit 309 Nr. 9 BGB.
- KEINE 14-Tage-Erinnerungsmail beim Monatsabo (_abo_konto_pruefen steigt
  bei laufzeit 1 vor der Erinnerung aus) — waere monatlicher Spam; AGB
  Ziffer 7 sagt das ausdruecklich.
- Kontingent/Rollover/ Pakete verhalten sich wie bei allen Bezahlplaenen.
- Plan-Wechsel: laengere Laufzeit = sofort (neue Laufzeit, voller Preis,
  Anrechnung), Wechsel AUF das Monatsabo = kuerzere Laufzeit = zum
  Laufzeitende vorgemerkt. Unveraendert.
Rechtstext: Nutzungsbedingungen Ziffer 6 (Preise) + Ziffer 7 (monatliche
Verlaengerung, keine Erinnerung).

