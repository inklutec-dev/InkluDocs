"""Stripe-Anbindung fuer das Abo-Modell (06.08.2026, zuerst NUR Testmodus).

Arbeitsteilung:
- billing.py bleibt die einzige Preis-/Kontingent-Quelle. Dieses Modul
  spiegelt die Konfiguration nach Stripe (Produkte/Preise, idempotent ueber
  lookup_keys) und liefert Checkout-/Portal-Links.
- Die eigentliche Freischaltung passiert im Webhook-Handler (main.py) ueber
  DIESELBE Mechanik wie der Admin-/Rechnungsweg (_setze_plan_kern) — Stripe
  ist nur ein zweiter Buchungsweg, kein zweites Abrechnungssystem.
- users.plan_quelle = 'stripe' kennzeichnet Stripe-verwaltete Abos: deren
  Verlaengerung kommt ueber invoice.paid, NICHT ueber den Tageslauf
  (sonst wuerde doppelt verlaengert).

Schluessel kommen aus der Umgebung (STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET,
sk_test_* im Testmodus). Ohne Schluessel ist das Modul inert (AKTIV=False) —
die Oberflaeche zeigt dann weiter den E-Mail-Hinweis statt Buchen-Knoepfen.
"""
import logging
import os

import billing

try:
    import stripe
except ImportError:  # Bau ohne Abhaengigkeit moeglich (AKTIV bleibt False)
    stripe = None

log = logging.getLogger("stripe_zahlung")

SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
AKTIV = bool(stripe and SECRET_KEY)
TESTMODUS = SECRET_KEY.startswith("sk_test_")
if AKTIV:
    stripe.api_key = SECRET_KEY

PLAN_NAMEN = {"single": "InkluDocs Single", "team": "InkluDocs Team",
              "enterprise": "InkluDocs Enterprise"}


def _plan_lookup(plan: str, monate: int) -> str:
    return f"idoc_{plan}_{monate}m"


def _paket_lookup(groesse: int) -> str:
    return f"idoc_paket_{groesse}"


def sichere_produkte() -> dict:
    """Legt Produkte und Preise in Stripe an, falls sie fehlen (idempotent).

    Preise werden ueber lookup_keys wiedergefunden — ein zweiter Lauf legt
    NICHTS doppelt an. Preisaenderungen in billing.py erzeugen bewusst KEINE
    automatische Aenderung in Stripe (Bestands-Abos!), sondern muessen dort
    als neue Preise gepflegt werden. Rueckgabe: lookup_key -> price_id.
    """
    if not AKTIV:
        raise RuntimeError("Stripe ist nicht konfiguriert")
    gewuenscht = {}
    for plan, preis_monat in billing.PLAN_PREISE_EUR.items():
        for monate in billing.PLAN_LAUFZEITEN:
            gewuenscht[_plan_lookup(plan, monate)] = ("abo", plan, monate,
                                                      round(preis_monat * monate * 100))
    for groesse, preis in billing.PAKET_PREISE.items():
        gewuenscht[_paket_lookup(groesse)] = ("paket", groesse, None, round(preis * 100))

    vorhanden = {}
    schluessel = list(gewuenscht)
    for i in range(0, len(schluessel), 10):   # lookup_keys: max. 10 pro Abfrage
        for p in stripe.Price.list(lookup_keys=schluessel[i:i + 10], limit=100).data:
            vorhanden[p.lookup_key] = p.id

    produkte = {}

    def _produkt(name: str, merkmal: str) -> str:
        if merkmal in produkte:
            return produkte[merkmal]
        for pr in stripe.Product.search(query=f"metadata['idoc']:'{merkmal}'").data:
            produkte[merkmal] = pr.id
            return pr.id
        pr = stripe.Product.create(name=name, metadata={"idoc": merkmal})
        produkte[merkmal] = pr.id
        return pr.id

    for lk, (art, a, monate, cents) in gewuenscht.items():
        if lk in vorhanden:
            continue
        if art == "abo":
            produkt_id = _produkt(PLAN_NAMEN[a], f"plan_{a}")
            preis = stripe.Price.create(
                product=produkt_id, currency="eur", unit_amount=cents,
                lookup_key=lk, nickname=f"{PLAN_NAMEN[a]} — {monate} Monate",
                recurring={"interval": "month", "interval_count": monate},
            )
        else:
            produkt_id = _produkt(f"InkluDocs Credit-Paket {a}", f"paket_{a}")
            preis = stripe.Price.create(
                product=produkt_id, currency="eur", unit_amount=cents,
                lookup_key=lk, nickname=f"{a} Credits",
            )
        vorhanden[lk] = preis.id
        log.info("Stripe-Preis angelegt: %s -> %s", lk, preis.id)
    return vorhanden


def _preis_id(lookup_key: str) -> str:
    preise = stripe.Price.list(lookup_keys=[lookup_key], limit=1).data
    if not preise:
        raise RuntimeError(f"Stripe-Preis fehlt: {lookup_key} (sichere_produkte() ausfuehren)")
    return preise[0].id


def kunde_fuer(db_user: dict) -> str:
    """Stripe-Customer zum Konto (anlegen, wenn noch keiner hinterlegt)."""
    kunde = (db_user.get("stripe_customer_id") or "").strip()
    if kunde:
        return kunde
    c = stripe.Customer.create(
        email=db_user["email"], name=db_user.get("display_name") or "",
        metadata={"idoc_user_id": str(db_user["id"])},
    )
    return c.id


def checkout_abo(db_user: dict, plan: str, monate: int, base_url: str,
                 kunde: str) -> str:
    """Checkout-Link fuer eine Abo-Buchung (fester Zeitraum, auto-verlaengernd)."""
    s = stripe.checkout.Session.create(
        customer=kunde,
        mode="subscription",
        line_items=[{"price": _preis_id(_plan_lookup(plan, monate)), "quantity": 1}],
        success_url=f"{base_url}/abo?zahlung=erfolg",
        cancel_url=f"{base_url}/abo?zahlung=abbruch",
        metadata={"idoc_user_id": str(db_user["id"]), "idoc_plan": plan,
                  "idoc_laufzeit": str(monate)},
        subscription_data={"metadata": {"idoc_user_id": str(db_user["id"]),
                                        "idoc_plan": plan,
                                        "idoc_laufzeit": str(monate)}},
    )
    return s.url


def checkout_paket(db_user: dict, groesse: int, base_url: str, kunde: str) -> str:
    """Checkout-Link fuer ein Zusatz-Credit-Paket (Einmalzahlung)."""
    s = stripe.checkout.Session.create(
        customer=kunde,
        mode="payment",
        line_items=[{"price": _preis_id(_paket_lookup(groesse)), "quantity": 1}],
        success_url=f"{base_url}/abo?zahlung=paket-erfolg",
        cancel_url=f"{base_url}/abo?zahlung=abbruch",
        invoice_creation={"enabled": True},
        metadata={"idoc_user_id": str(db_user["id"]), "idoc_paket": str(groesse)},
    )
    return s.url


_portal_conf_id = None


def portal_url(kunde: str, base_url: str) -> str:
    """Link ins Stripe-Kundenportal (Rechnungen, Zahlungsart, Kuendigung)."""
    global _portal_conf_id
    if _portal_conf_id is None:
        confs = stripe.billing_portal.Configuration.list(limit=10).data
        eigene = [c for c in confs if c.metadata.get("idoc") == "portal"]
        if eigene:
            _portal_conf_id = eigene[0].id
        else:
            conf = stripe.billing_portal.Configuration.create(
                business_profile={"headline": "InkluDocs — Rechnungen und Zahlungen"},
                features={
                    "invoice_history": {"enabled": True},
                    "payment_method_update": {"enabled": True},
                    "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
                },
                default_return_url=f"{base_url}/abo",
                metadata={"idoc": "portal"},
            )
            _portal_conf_id = conf.id
    s = stripe.billing_portal.Session.create(
        customer=kunde, configuration=_portal_conf_id, return_url=f"{base_url}/abo")
    return s.url


def sichere_webhook_endpunkt(base_url: str):
    """Webhook-Endpunkt in Stripe anlegen, falls er fehlt.

    Rueckgabe: (url, secret|None). Das Signier-Geheimnis liefert Stripe NUR
    bei der Neuanlage — es muss dann in die .env (STRIPE_WEBHOOK_SECRET).
    Existiert der Endpunkt schon, kommt (url, None) zurueck.
    """
    ziel = f"{base_url}/api/stripe/webhook"
    for ep in stripe.WebhookEndpoint.list(limit=20).data:
        if ep.url == ziel:
            return ziel, None
    ep = stripe.WebhookEndpoint.create(
        url=ziel,
        enabled_events=["checkout.session.completed", "invoice.paid",
                        "customer.subscription.updated",
                        "customer.subscription.deleted"],
        description="InkluDocs Abo-Freischaltung",
    )
    return ziel, ep.secret


def pruefe_webhook(payload: bytes, signatur: str):
    """Signatur pruefen und Event liefern (wirft bei ungueltiger Signatur)."""
    if not WEBHOOK_SECRET:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET fehlt")
    return stripe.Webhook.construct_event(payload, signatur, WEBHOOK_SECRET)


def kuendige_subscription(subscription_id: str, zum_periodenende: bool) -> None:
    """Kuendigt (oder reaktiviert) die Stripe-Subscription zum Periodenende."""
    stripe.Subscription.modify(subscription_id,
                               cancel_at_period_end=bool(zum_periodenende))
