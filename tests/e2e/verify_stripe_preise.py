#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stimmen die Preise in Stripe mit billing.py ueberein? (Steve 31.08.2026)

Nur lesend. Laeuft im Container gegen den Schluessel in STRIPE_SECRET_KEY — mit dem
Test-Schluessel prueft es den Test-Modus (Staging), mit dem Live-Schluessel den Live-Modus.
    docker exec -e STRIPE_SECRET_KEY=... inkludocs-staging python3 /tmp/verify_stripe_preise.py
Prueft: jeder erwartete lookup_key existiert genau einmal aktiv, der Betrag stimmt auf den
Cent, und es gibt keine aktiven idoc_-Preise, die billing nicht kennt (Altlasten)."""
import os, sys
sys.path.insert(0, "/app")
import stripe, billing, stripe_zahlung as sz  # noqa: E402

stripe.api_key = os.environ["STRIPE_SECRET_KEY"].strip()
modus = "LIVE" if stripe.api_key.startswith("sk_live") else "TEST"
ok = fehler = 0
def check(name, cond, info=""):
    global ok, fehler
    if cond: ok += 1; print("OK   ", name)
    else: fehler += 1; print("FEHLT", name, "—", str(info)[:200])

erwartet = {}
for plan in billing.PLAN_PREISE_EUR:
    for monate in billing.PLAN_LAUFZEITEN:
        erwartet[sz._plan_lookup(plan, monate)] = round(billing.preis_pro_monat(plan, monate) * monate * 100)
    erwartet[sz._plan_lookup_treue(plan)] = round(billing.treue_preis_eur(plan) * 100)
for groesse, preis in billing.PAKET_PREISE.items():
    erwartet[sz._paket_lookup(groesse)] = round(preis * 100)

print(f"Modus: {modus} — {len(erwartet)} Preise erwartet")
aktiv = {}
for p in stripe.Price.list(limit=100, active=True).auto_paging_iter():
    lk = p.get("lookup_key") or ""
    if lk.startswith("idoc_"):
        aktiv.setdefault(lk, []).append(p)

for lk, cents in sorted(erwartet.items()):
    ps = aktiv.get(lk, [])
    if len(ps) != 1:
        check(f"{lk}: genau ein aktiver Preis", False, f"{len(ps)} gefunden"); continue
    p = ps[0]
    check(f"{lk}: {cents/100:.2f} EUR", p["unit_amount"] == cents and p["currency"] == "eur",
          f"Stripe hat {p['unit_amount']/100:.2f} {p['currency']}")
fremd = sorted(set(aktiv) - set(erwartet))
check("keine aktiven idoc_-Preise, die billing nicht kennt", not fremd, fremd)
print(f"\nErgebnis ({modus}): {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
