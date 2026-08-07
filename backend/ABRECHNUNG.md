
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
