
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



## TREUE-MONATSABO: ANSCHLUSS NACH FESTER LAUFZEIT (24.08.2026)

AGB Ziffer 7: Ein ONLINE gebuchter Vertrag mit fester Laufzeit laeuft nach
dem Laufzeitende automatisch als MONATSABO zur Treuekondition weiter —
zum monatlichen Effektivpreis der gebuchten Laufzeit (billing.
treue_preis_eur: 9,95/19,95/49,95), nicht zum regulaeren Monatsabo-Preis
(11,95/23,95/59,95). Jederzeit zum Monatsende kuendbar. Der Rechnungsweg
(Michael/Actino) verlaengert unveraendert um dieselbe Laufzeit.

Umsetzung:
- Stripe-Preise idoc_<plan>_treue_1m (sichere_produkte, jetzt 18 Preise).
  BEIM LIVE-GANG: sichere_produkte() im LIVE-Konto laufen lassen!
- plan_aus_lookup liefert (plan, monate, treue) — Aufrufer unterscheiden
  Treue (9,95, kein Kontingent-Neustart) von regulaerem Monatsabo (11,95).
- stripe_zahlung.plane_treue_anschluss: Zweiphasen-Schedule (Phase 1 =
  bezahlte Laufzeit, Phase 2 = 1 Monat Treuepreis, end_behavior release
  -> danach laeuft das Abo von selbst monatlich zum Treuepreis).
- Angelegt beim checkout.session.completed (Laufzeit > 1); SELBSTHEILUNG
  im invoice.paid-Webhook (nur_wenn_frei=True — vorgemerkte Downgrades
  werden nie ueberschrieben).
- Die drei Zerstoerungs-Wege sind abgedeckt: UPGRADE (wechsle_sofort setzt
  den Zeitplan fuer den neuen Plan neu auf), KUENDIGUNGS-WIDERRUF
  (_stripe_kuendigung_sync stellt ihn wieder her), DOWNGRADE
  (plane_wechsel_zum_periodenende plant Phase 3 = Treue des neuen Plans;
  der Downgrade-Widerruf stellt den Anschluss des laufenden Plans her).
- users.plan_treue markiert laufende Treue-Abos (Anzeige /abo, /api/me
  eigen.treue); Erinnerungsmail, Bestaetigungsmail und /abo-Texte nennen
  fuer Stripe-Laufzeiten den Treue-Anschluss statt einer Verlaengerung.
- Bestaetigungsmail dokumentiert seit 24.08. auch die protokollierte
  Zustimmung zum sofortigen Leistungsbeginn (Paragraf 312f BGB).

WIDERRUFSFUNKTION Paragraf 356a BGB (24.08.2026): /widerrufen —
zweistufig (Vertrag widerrufen -> Widerruf bestaetigen), ohne Anmeldung,
in allen Fusszeilen verlinkt, nutzt /api/kuendigung mit art=widerruf.
Belehrungs-Fassung 2026-08-24. Rueckabwicklung bleibt vorerst Handarbeit
(Steves Entscheidung 23.08.2026).

Tests: verify_treue.py (61 Checks, echte Stripe-Test-API, in
alle_tests.sh); verify_stripe-Pin auf 18 Preise.

## AKTIONSPREISE (Michael Karbe, bestaetigt 29.08.2026 — gebaut 29.08.2026)

Eine Preisquelle: `billing.AKTIONS_PREISE` (je Vorgang), `billing.EXPORT_ARTEN` +
`EXPORT_SCHRITT` (Export-Staffel), `billing.TABELLEN_EXPORTE` (feste Preise).
Werte: Alt-Text 5 (alle Wege inkl. API/Chat), Quickinfo 1 je FELD, Chat-Aenderung
Alt-Text 5 / Quickinfo 1, Datei-Export 25 + 5 je angefangene 10 Bilder (PDF/Word)
bzw. + 1 je angefangene 10 Felder (Formular), CSV/JSON/Formular-CSV 10.
Stammdaten-CSV bleibt frei (eigene Bibliothek). Reden mit dem InkluAgent frei.

Wache: `aktion_pruefung(user, aktion, menge)` / `export_pruefung(user, anzahl, art)`
— erlaubt nur, wenn `verfuegbare_credits` (Monatsrest + Pakete; None = unbegrenzt
bei Enterprise/Admin/Enforcement aus) den vollen Preis deckt. Antworten: 402 mit
`credits_fehlen_detail` (Export, Neu generieren, Formular-Generierung), 429 mit
Zahlen (Public API), Chat-Text `credits_fehlen_text` (InkluAgent). Sammellaeufe
pruefen je Bild bzw. je Formularseite (Preis = offene Felder) und lassen den Rest
offen. Verbucht wird IMMER erst nach erfolgreicher Aktion (Feld-Pass: je
geschriebenes Feld; Exporte: nach fertiger Antwort, Header `X-Export-Credits`).

Oberflaeche: `window.CREDIT_PREISE` (aus `preise_fuer_frontend()`, app.html) fuer
die Knoepfe „n … neu generieren, c Credits“; Preisseite und Abo-Seite ziehen die
Zahlen aus billing.py (Kontext `preis_*`, `plan_credits`). AGB Ziffer 6 nennt die
Preise als Text — bei Preisaenderung dort UND Rundmail-Text nachziehen.
Tests: tests/test_billing_export.py (Unit), tests/e2e/verify_formular.py (Header 27
bei 12 Feldern, CSV 10), /home/claude/verify_*.py (Pins mal fuenf).

## SAMMELLAUF-START (30.08.2026)
Der Start `POST /api/projects/{id}/generate` hatte KEINE Credit-Wache, nur das
Tageslimit. Jetzt: `billing.aktion_pruefung(user_id, "bild_generierung")` -> 402
`credits_fehlen_detail`. Reihenfolge im Endpunkt: Tageslimit (429) -> Besitz/
Zustand (404/409) -> Anzahl ermitteln -> Guthaben (402) -> erst DANN im Modus
ki_neu auf pending umschalten. Grund: vorher lief der Lauf ohne Wache los,
schaltete fertige Bilder auf pending und brach am ersten Bild ab; die Bilder
blieben als "nicht generiert" stehen. Admins und ABO_ENFORCEMENT=off sind
unberuehrt (verfuegbare_credits liefert None -> erlaubt). Details zum Abbruch:
docs/GENERIERUNG.md. Test: tests/e2e/verify_ki_neu_abbruch.py.

## TAGESLIMIT (29.08.2026 nachts — drei Luecken geschlossen)
`main.tageslimit_wache(user_row)` ist DIE Wache (None = darf, sonst {limit, genutzt});
Admins in der Oberflaeche ausgenommen (Public API weiterhin ohne Ausnahme, eigener
Zaehler api_usage). Genutzt = max(heute verarbeitete Bilder [alter Zaehler],
`billing.tagesverbrauch_ki` = heutige usage_events bild_generierung +
quickinfo_generierung). Eingesetzt: Start UND je Bild im Sammellauf (Luecke 1),
Einzel-Neu-Generieren (Luecke 2, vorher gar nicht), Formular-Start und je Seite im
Feld-Pass (Luecke 3, via Deps). Limit: users.api_tageslimit, sonst DAILY_IMAGE_LIMIT
(Steve 29.08.: Prod = 500 wie Staging). Test: tests/e2e/verify_tageslimit_luecken.py.
