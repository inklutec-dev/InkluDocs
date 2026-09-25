#!/usr/bin/env python3
"""Bisherige Kaeufe und Gutschriften als Umsatz-Buchungen nachtragen (25.09.2026).

Die Tabelle `buchungen` gibt es erst seit dem Umsatz-Umbau. Damit „Gesamt seit Start“ ab
dem ersten Tag stimmt, traegt dieses Skript alles nach, was vorher geschah:

- jedes Credit-Paket (quota_pakete) ohne Buchung:
    stripe   -> Betrag aus Stripe (Checkout-Session), ersatzweise Listenpreis
    rechnung -> Verkauf auf Rechnung zum Listenpreis (freie Mengen 4 Cent je Credit)
    admin    -> Bonus (kostenlos)
- jede bezahlte Stripe-Abo-Rechnung (invoice.paid) mit dem eingezogenen Betrag

Berichtigungen fuer falsch eingeordnete Pakete gibt man mit:
    --als-verkauf PAKET_ID:BETRAG   (z. B. 53:87,50)
    --als-bonus PAKET_ID
Sie aendern auch Quelle und Verfall des Pakets (Verkauf verfaellt nie, Bonus nach 12 Monaten).

Idempotent: Pakete mit Buchung und bereits gebuchte Stripe-Rechnungen werden uebersprungen.
Ohne --ausfuehren wird nur angezeigt, was passieren wuerde.

Aufruf im Container:  python3 /app/scripts/umsatz_nachtragen.py [--ausfuehren] [...]
"""
import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent))

import billing  # noqa: E402
import stripe_zahlung  # noqa: E402
import umsatz  # noqa: E402
from database import get_db  # noqa: E402

VON = "nachgetragen am 25.09.2026"


def stripe_betrag(session_id: str):
    if not (stripe_zahlung.AKTIV and session_id):
        return None
    try:
        s = stripe_zahlung.stripe.checkout.Session.retrieve(session_id)
        return int(s.get("amount_total") or 0)
    except Exception as e:  # noqa: BLE001
        print(f"  ! Stripe-Session {session_id} nicht lesbar: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ausfuehren", action="store_true")
    ap.add_argument("--als-verkauf", action="append", default=[], metavar="ID:BETRAG")
    ap.add_argument("--als-bonus", action="append", default=[], metavar="ID")
    ap.add_argument("--ohne-stripe-abos", action="store_true")
    a = ap.parse_args()

    als_verkauf = {}
    for eintrag in a.als_verkauf:
        pid, _, betrag = eintrag.partition(":")
        als_verkauf[int(pid)] = umsatz.euro_zu_cent(betrag)
    als_bonus = {int(x) for x in a.als_bonus}

    conn = get_db()
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'buchungen'").fetchone():
        sys.exit("Tabelle buchungen fehlt — zuerst die neue Version starten (legt sie beim Start an).")
    konten = {r["id"]: dict(r) for r in conn.execute("SELECT id, display_name, email FROM users")}
    pakete = [dict(r) for r in conn.execute(
        "SELECT * FROM quota_pakete WHERE id NOT IN "
        "(SELECT paket_id FROM buchungen WHERE paket_id IS NOT NULL) ORDER BY id")]
    print(f"{len(pakete)} Pakete ohne Buchung")
    summe = 0
    for p in pakete:
        konto = konten.get(p["user_id"]) or {"id": p["user_id"]}
        notiz = p.get("notiz") or ""
        quelle = p["quelle"]
        status, ref, weg = "ok", None, None
        if p["id"] in als_verkauf:
            weg, betrag = "rechnung", als_verkauf[p["id"]]
            quelle_neu, verfall_neu = "rechnung", "NULL"
        elif p["id"] in als_bonus:
            weg, betrag = "bonus", 0
            quelle_neu, verfall_neu = "admin", "COALESCE(verfaellt_am, datetime(erstellt_am, '+12 months'))"
        else:
            quelle_neu = verfall_neu = None
            if quelle == "stripe":
                weg = "stripe"
                m = re.search(r"cs_(?:live|test)_[A-Za-z0-9]+", notiz)
                ref = m.group(0) if m else None
                betrag = stripe_betrag(ref)
                if betrag is None:
                    betrag = umsatz.vorschlag_paket_cent(p["groesse"])
                if "Rücklastschrift" in notiz:
                    status = "rueckgelaufen"
                elif "(Lastschrift ausstehend)" in notiz:
                    status = "ausstehend"
            elif quelle == "rechnung":
                weg, betrag = "rechnung", umsatz.vorschlag_paket_cent(p["groesse"])
            else:
                weg, betrag = "bonus", 0
        if weg != "bonus" and status != "rueckgelaufen":
            summe += betrag
        print(f"  Paket {p['id']:>4} {konto.get('email', '?'):<34} {p['groesse']:>6} Credits "
              f"{quelle:<8} -> {umsatz.WEG_TEXT[weg]:<18} {umsatz.cent_text(betrag):>12} {status}"
              + (f"  (Quelle -> {quelle_neu})" if quelle_neu else ""))
        if a.ausfuehren:
            umsatz.buche(conn, konto=konto, art="paket", weg=weg, credits=p["groesse"],
                         betrag_cent=betrag, notiz=notiz, status=status, paket_id=p["id"],
                         stripe_ref=ref, von_name="Stripe (automatisch)" if weg == "stripe" else VON,
                         gebucht_am=p["erstellt_am"])
            if quelle_neu:
                conn.execute(f"UPDATE quota_pakete SET quelle = ?, verfaellt_am = {verfall_neu} WHERE id = ?",
                             (quelle_neu, p["id"]))

    if not a.ohne_stripe_abos and stripe_zahlung.AKTIV:
        print("Stripe-Abo-Rechnungen:")
        kunden = {r["stripe_customer_id"]: r["id"] for r in conn.execute(
            "SELECT id, stripe_customer_id FROM users WHERE stripe_customer_id IS NOT NULL")}
        gebucht = {r[0] for r in conn.execute("SELECT stripe_ref FROM buchungen WHERE stripe_ref IS NOT NULL")}
        for inv in stripe_zahlung.stripe.Invoice.list(status="paid", limit=100).auto_paging_iter():
            if not inv.get("subscription") or int(inv.get("amount_paid") or 0) <= 0 or inv["id"] in gebucht:
                continue
            plan = laufzeit = None
            for z in (inv.get("lines") or {}).get("data") or []:
                lk = ((z.get("price") or {}).get("lookup_key")
                      or ((z.get("pricing") or {}).get("price_details") or {}).get("lookup_key"))
                p_, m_, _t = stripe_zahlung.plan_aus_lookup(lk)
                if p_:
                    plan, laufzeit = p_, m_
            uid = kunden.get(inv.get("customer"))
            konto = konten.get(uid) or {"display_name": inv.get("customer_name") or "",
                                        "email": inv.get("customer_email") or ""}
            bezahlt = ((inv.get("status_transitions") or {}).get("paid_at") or inv.get("created"))
            wann = datetime.fromtimestamp(int(bezahlt), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            betrag = int(inv["amount_paid"])
            summe += betrag
            print(f"  {inv['id']} {konto.get('email', '?'):<34} {plan or '?':<10} {umsatz.cent_text(betrag):>12} {wann}")
            if a.ausfuehren:
                umsatz.buche(conn, konto=konto, art="abo", weg="stripe", plan=plan, laufzeit=laufzeit,
                             betrag_cent=betrag, stripe_ref=inv["id"], von_name="Stripe (automatisch)",
                             gebucht_am=wann)
    elif not stripe_zahlung.AKTIV:
        print("Stripe nicht eingerichtet — Abo-Rechnungen uebersprungen.")

    offen = conn.execute("SELECT email, plan, plan_laufzeit_monate FROM users WHERE plan_quelle = 'rechnung' "
                         "AND plan != 'free'").fetchall()
    for r in offen:
        print(f"  HINWEIS: Rechnungs-Abo ohne Betrag, bitte von Hand buchen: {r['email']} {r['plan']} "
              f"{r['plan_laufzeit_monate']} Monate")
    print(f"Umsatz der nachgetragenen Buchungen: {umsatz.cent_text(summe)}")
    if a.ausfuehren:
        conn.commit()
        print("GESCHRIEBEN.")
    else:
        conn.rollback()
        print("Nur Vorschau — mit --ausfuehren schreiben.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
