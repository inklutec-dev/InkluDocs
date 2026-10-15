"""Kaufen im Gespraech, Baustein 3 (Steve 14.10.2026: „man kann auch mit dem Bot Pakete und Credits buchen — aber man muss das
ausdrücklich bestätigen“) — Schalter funktionen.AGENT_KAUFEN, inkluagent/karten.py (Karten „Zusatz-Credits kaufen“, „Abo buchen“).
Wegwerf-Datenbank (/tmp), Stripe ist eine Attrappe (kein Netz, keine Zahlung).
    docker exec -w /app <container> python3 -m unittest /app/tests/test_kaufen.py

Prueft: Werkzeuge und Prompt nur mit Schalter, Karten und Online-Zahlung, nie in der Demo; die Karten mit denselben Saetzen, Zahlen,
Pflicht-Haken und Links wie „Abo & Verbrauch“ (Paket mit Preis in der Auswahl, Abo mit Plan, Gesamtpreis, Leistung und
Verlaengerung fest auf der Karte, Haken nicht vorbelegt); Abschicken nur mit gueltigem Paket bzw. Haken, Plan und Laufzeit aus der
Karte des Servers, derselbe Kern wie die Endpunkte (Zustimmung protokolliert), genau einmal; die Seite leitet nur zu Stripe-Checkout;
die Endpunkte der Abo-Seite verhalten sich wie vorher; Hilfe, Namen, Daten; der Auftrag schlaegt bei zu wenig Guthaben das passende
Paket vor.
"""
import asyncio
import itertools
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="kaufen_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KAUFEN", "INKLUAGENT_KARTEN", "INKLUAGENT_KONTO", "INKLUAGENT_RUECKMELDUNG", "INKLUAGENT_AUFTRAEGE"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import billing  # noqa: E402
import funktionen  # noqa: E402
import stripe_zahlung  # noqa: E402
from inkluagent import agent_loop, begleiter, karten, storage  # noqa: E402

try:
    import main  # noqa: E402
except Exception:  # noqa: BLE001
    main = None

_NR = itertools.count(1)
STRIPE_URL = "https://checkout.stripe.com/c/pay/cs_test_fiktiv"
_ = (lambda s: s)


class an:
    """Kaufen an: Schalter AGENT_KAUFEN, Karten und Online-Zahlung (Attrappe)."""

    def __init__(self, kaufen=True, stripe=True):
        self.patches = [mock.patch.object(funktionen, "AGENT_KARTEN", True), mock.patch.object(funktionen, "AGENT_KAUFEN", kaufen),
                        mock.patch.object(stripe_zahlung, "AKTIV", stripe)]

    def __enter__(self):
        for p in self.patches:
            p.start()

    def __exit__(self, *a):
        for p in reversed(self.patches):
            p.stop()


def nutzer(plan: str = "free") -> dict:
    mail = f"kaufen{next(_NR)}-{time.time_ns()}@beispiel.invalid"
    uid = database.create_user(mail, "geheim-12345", "Erika Mustermann (fiktiv)", email_verified=1)
    if plan != "free":
        c = database.get_db()
        c.execute("UPDATE users SET plan = ?, plan_gueltig_bis = date('now', '+30 days') WHERE id = ?", (plan, uid))
        c.commit()
        c.close()
    return {"id": uid, "email": mail}


def vorbereiten(name, args, uid, project_id=None):
    return karten.ausfuehren(name, args, uid, project_id, _)


def absenden(kid, user, werte, absender="schluessel-fiktiv"):
    return asyncio.run(karten.absenden(kid, user, werte, "de", absender))


class Basis(unittest.TestCase):
    def setUp(self):
        karten._KARTEN.clear()
        self.u = nutzer()
        self.schalter = an()
        self.schalter.__enter__()

    def tearDown(self):
        self.schalter.__exit__()
        mock.patch.stopall()


# ─── 1. Schalter ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Schalter(unittest.TestCase):
    NAMEN = {"credits_kaufen_vorbereiten", "abo_buchen_vorbereiten"}

    def test_nur_mit_schalter_karten_und_online_zahlung(self):
        for kaufen, stripe, erwartet in ((True, True, True), (False, True, False), (True, False, False)):
            with self.subTest(kaufen=kaufen, stripe=stripe), an(kaufen, stripe):
                namen = {d["name"] for d in karten.werkzeuge(None)}
                self.assertEqual(self.NAMEN <= namen, erwartet)
                from inkluagent.prompts import schubfaecher
                lage = schubfaecher.Lage("pdf", 1, karten=True)
                faecher = [f.name for f in schubfaecher.faecher(lage)]
                self.assertEqual("kaufen" in faecher, erwartet)
                self.assertIn("karten", faecher)
                self.assertEqual(karten.PROMPT_KAUFEN in schubfaecher.zusammensetzen(lage), erwartet)
        with mock.patch.object(funktionen, "AGENT_KARTEN", False), mock.patch.object(funktionen, "AGENT_KAUFEN", True), \
                mock.patch.object(stripe_zahlung, "AKTIV", True):
            self.assertFalse(karten.kaufen_an(), "ohne Karten kein Kaufen")
        with an():
            self.assertEqual(karten.ausfuehren("credits_kaufen_vorbereiten", {}, 1, None, _)["ok"], True)
        self.assertFalse(karten.ausfuehren("credits_kaufen_vorbereiten", {}, 1, None, _)["ok"], "Schalter aus: unbekannt")

    def test_nie_in_der_demo_und_im_prompt_des_gespraechs(self):
        gesehen = {}

        class Agent:
            kennt_frist = kann_streamen = False

            def invoke_with_tools(self, anthropic_messages, tools, system, **kw):
                gesehen["tools"], gesehen["system"] = {t["name"] for t in tools}, system
                return {"content": [{"type": "text", "text": "Gern (fiktiv)."}], "stop_reason": "end_turn"}
        u = nutzer()
        with an(), mock.patch.object(funktionen, "AGENT_KONTO", True):
            gid = storage.aktuelles_gespraech(u["id"])
            agent_loop.run_agent(None, u["id"], "Ich will Credits kaufen", None, Agent(), konto=True,
                                 begleiter_lauf=begleiter.Lauf(ui_sprache="de", nutzer_text="Ich will Credits kaufen"), gespraech_id=gid)
            self.assertTrue(self.NAMEN <= gesehen["tools"])
            self.assertIn(karten.PROMPT_KAUFEN, gesehen["system"])
            agent_loop.run_agent(None, u["id"], "Ich will Credits kaufen", None, Agent(), konto=True)
            self.assertFalse(self.NAMEN & gesehen["tools"], "ohne Lauf des Kontos (Demo) keine Karten")


# ─── 2. Karte „Zusatz-Credits kaufen“ ───────────────────────────────────────────────────────────────────────────────────────
class CreditsKarte(Basis):
    def test_paket_mit_preis_in_der_auswahl_wie_auf_der_abo_seite(self):
        r = vorbereiten("credits_kaufen_vorbereiten", {"paket": 2500}, self.u["id"])
        k = r["anhang"]
        self.assertEqual((k["karte"], k["titel"], k["knopf"]), ("credits_kaufen", "Zusatz-Credits kaufen", "Kostenpflichtig kaufen"))
        feld = k["felder"][0]
        self.assertEqual((feld["name"], feld["art"], feld["label"], feld["pflicht"], feld["wert"]),
                         ("paket", "auswahl", "Paketgröße (Credits)", True, "2500"))
        self.assertEqual([(o["wert"], o["label"]) for o in feld["auswahl"]],
                         [(str(g), f"{t} — {b}") for g, t, b in billing.pakete_fuer_anzeige()])
        self.assertIn("Bezahlt wird auf der sicheren Stripe-Seite.", k["text"])
        self.assertEqual(k["zeilen"], [karten.T_ENDPREISE])
        self.assertIn("bezahlt wird nur dort", r["result"]["hinweis"])

    def test_vorschlag_ohne_oder_falsch_ist_das_kleinste(self):
        for args in ({}, {"paket": 7}, {"paket": "viel"}):
            self.assertEqual(vorbereiten("credits_kaufen_vorbereiten", args, self.u["id"])["anhang"]["felder"][0]["wert"],
                             str(min(billing.PAKET_PREISE)))

    def test_guthaben_auf_der_karte_wenn_begrenzt(self):
        with mock.patch.object(billing, "verfuegbare_credits", return_value=1375):
            k = vorbereiten("credits_kaufen_vorbereiten", {}, self.u["id"])["anhang"]
        self.assertTrue(k["text"].startswith("Dein Guthaben: 1.375 Credits."), k["text"])


# ─── 3. Karte „Abo buchen“ ───────────────────────────────────────────────────────────────────────────────────────────────────
class AboKarte(Basis):
    def test_bestellung_fest_auf_der_karte_wie_auf_der_abo_seite(self):
        k = vorbereiten("abo_buchen_vorbereiten", {"plan": "single", "laufzeit_monate": 6}, self.u["id"])["anhang"]
        self.assertEqual((k["karte"], k["titel"], k["knopf"]), ("abo_buchen", "Abo buchen", "Kostenpflichtig buchen"))
        self.assertEqual(k["zeilen"], [
            "Single — 250 Credits im Monat, ein Konto",
            "Gesamtpreis: 59,70 € für 6 Monate — 9,95 € pro Monat.",
            "Leistung: 250 Credits pro Monat. Ein Alt-Text kostet 5 Credits, eine Quickinfo 1 Credit.",
            "Laufzeit 6 Monate, im Voraus zu zahlen. Wenn du nicht zum Laufzeitende kündigst, läuft das Abo danach als Monatsabo zur "
            "Treuekondition weiter: 9,95 € pro Monat, jederzeit monatlich kündbar.",
            karten.T_ENDPREISE])
        self.assertEqual(k["text"], karten.T_STRIPE)
        haken = k["felder"][0]
        self.assertEqual((haken["name"], haken["art"], haken["pflicht"], haken["wert"]), ("widerruf", "haken", True, False),
                         "Pflicht und NICHT vorbelegt")
        self.assertEqual(haken["label"], karten.ZUSTIMMUNG_LEISTUNGSBEGINN)
        self.assertEqual(haken["link"], {"url": "/widerruf", "text": "Widerrufsbelehrung und Muster-Widerrufsformular"})
        self.assertEqual(haken["links"], [{"url": "/nutzungsbedingungen", "text": "Nutzungsbedingungen"}])
        self.assertEqual(len(k["felder"]), 1, "Plan und Laufzeit stehen fest — der Browser schickt sie nicht")

    def test_monatsabo_team(self):
        z = vorbereiten("abo_buchen_vorbereiten", {"plan": "team", "laufzeit_monate": 1}, self.u["id"])["anhang"]["zeilen"]
        self.assertEqual(z[0], "Team — 500 Credits im Monat, bis 5 Personen")
        self.assertEqual(z[1], "Gesamtpreis: 23,95 € pro Monat, monatlich kündbar.")
        self.assertTrue(z[3].startswith("Monatsabo: 23,95 € werden jeweils für einen Monat im Voraus abgebucht."))

    def test_ohne_plan_oder_mit_abo_keine_karte(self):
        self.assertFalse(vorbereiten("abo_buchen_vorbereiten", {"plan": "gold", "laufzeit_monate": 6}, self.u["id"])["ok"])
        self.assertFalse(vorbereiten("abo_buchen_vorbereiten", {"plan": "single", "laufzeit_monate": 2}, self.u["id"])["ok"])
        mit_abo = nutzer("single")
        r = vorbereiten("abo_buchen_vorbereiten", {"plan": "team", "laufzeit_monate": 6}, mit_abo["id"])
        self.assertIsNone(r["result"]["karte"])
        self.assertNotIn("anhang", r)


# ─── 4. Abschicken ───────────────────────────────────────────────────────────────────────────────────────────────────────────
@unittest.skipIf(main is None, "main nicht ladbar")
class Abschicken(Basis):
    def test_credits_weiter_zu_stripe_genau_einmal(self):
        kid = vorbereiten("credits_kaufen_vorbereiten", {}, self.u["id"])["anhang"]["karten_id"]
        with mock.patch.object(main, "_paket_checkout_url", return_value=STRIPE_URL) as kern:
            status, a = absenden(kid, self.u, {"paket": "5000"})
            self.assertEqual(status, 200, a)
            self.assertEqual((a["weiter"], a["karte"]["zustand"], a["text"]), (STRIPE_URL, "gesendet", karten.T_WEITER))
            self.assertEqual(kern.call_args.args[0]["id"], self.u["id"])
            self.assertEqual(kern.call_args.args[1], 5000)
            status, _a = absenden(kid, self.u, {"paket": "5000"})
            self.assertEqual(status, 409, "genau einmal")
        self.assertEqual(kern.call_count, 1)

    def test_credits_nur_gueltige_pakete(self):
        kid = vorbereiten("credits_kaufen_vorbereiten", {}, self.u["id"])["anhang"]["karten_id"]
        with mock.patch.object(main, "_paket_checkout_url", return_value=STRIPE_URL) as kern:
            status, a = absenden(kid, self.u, {"paket": "999999"})
        self.assertEqual(status, 400)
        self.assertIn("paket", a["fehler"])
        kern.assert_not_called()

    def test_abo_nur_mit_haken_und_plan_aus_der_karte(self):
        kid = vorbereiten("abo_buchen_vorbereiten", {"plan": "single", "laufzeit_monate": 6}, self.u["id"])["anhang"]["karten_id"]
        with mock.patch.object(main, "_abo_checkout_url", return_value=STRIPE_URL) as kern:
            status, a = absenden(kid, self.u, {"widerruf": False})
            self.assertEqual(status, 400)
            self.assertEqual(a["fehler"], {"widerruf": "Bitte bestätige das."})
            kern.assert_not_called()
            status, a = absenden(kid, self.u, {"widerruf": True, "plan": "enterprise", "monate": 12})
        self.assertEqual(status, 200, a)
        self.assertEqual(kern.call_args.args[1:], ("single", 6, "schluessel-fiktiv"), "Plan und Laufzeit aus der Karte des Servers")
        self.assertEqual(a["weiter"], STRIPE_URL)

    def test_fehler_des_kerns_kommt_an_und_die_karte_bleibt_offen(self):
        from fastapi import HTTPException
        kid = vorbereiten("abo_buchen_vorbereiten", {"plan": "single", "laufzeit_monate": 6}, self.u["id"])["anhang"]["karten_id"]
        with mock.patch.object(main, "_abo_checkout_url", side_effect=HTTPException(status_code=502, detail="Die Zahlungsseite "
                                                                                    "konnte nicht erstellt werden")):
            status, a = absenden(kid, self.u, {"widerruf": True})
        self.assertEqual((status, a["detail"]), (502, "Die Zahlungsseite konnte nicht erstellt werden"))
        self.assertTrue(karten.gilt(kid), "noch einmal versuchen")

    def test_schalter_aus_die_karte_gilt_nicht_mehr(self):
        kid = vorbereiten("credits_kaufen_vorbereiten", {}, self.u["id"])["anhang"]["karten_id"]
        with mock.patch.object(funktionen, "AGENT_KAUFEN", False), mock.patch.object(main, "_paket_checkout_url") as kern:
            status, a = absenden(kid, self.u, {"paket": "500"})
        self.assertEqual((status, a["karte"]["zustand"]), (404, "abgelaufen"))
        kern.assert_not_called()

    def test_fremde_karte(self):
        kid = vorbereiten("credits_kaufen_vorbereiten", {}, self.u["id"])["anhang"]["karten_id"]
        with mock.patch.object(main, "_paket_checkout_url", return_value=STRIPE_URL) as kern:
            status, _a = absenden(kid, nutzer(), {"paket": "500"})
        self.assertEqual(status, 404)
        kern.assert_not_called()


# ─── 5. Die Endpunkte der Abo-Seite (derselbe Kern) ─────────────────────────────────────────────────────────────────────────
@unittest.skipIf(main is None, "main nicht ladbar")
class Endpunkte(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.u = nutzer()
        mock.patch.object(stripe_zahlung, "AKTIV", True).start()
        mock.patch.object(stripe_zahlung, "kunde_fuer", return_value="cus_fiktiv").start()
        self.abo = mock.patch.object(stripe_zahlung, "checkout_abo", return_value=STRIPE_URL).start()
        self.paket = mock.patch.object(stripe_zahlung, "checkout_paket", return_value=STRIPE_URL).start()
        self.c = TestClient(main.app)
        self.c.cookies.set("token", main.create_token(self.u["id"], self.u["email"], 0))

    def tearDown(self):
        mock.patch.stopall()

    def test_abo_checkout_mit_protokollierter_zustimmung(self):
        r = self.c.post("/api/abo/checkout", json={"plan": "team", "laufzeit_monate": 3, "widerruf_zustimmung": True})
        self.assertEqual(r.json(), {"ok": True, "url": STRIPE_URL})
        c = database.get_db()
        z = dict(c.execute("SELECT * FROM widerruf_zustimmungen WHERE user_id = ?", (self.u["id"],)).fetchone())
        c.close()
        self.assertEqual((z["plan"], z["laufzeit_monate"], z["summe_cent"]), ("team", 3, 5985))
        self.assertEqual(self.abo.call_args.kwargs["zustimmung_id"], z["id"])
        c = database.get_db()
        self.assertEqual(c.execute("SELECT stripe_customer_id FROM users WHERE id = ?", (self.u["id"],)).fetchone()[0], "cus_fiktiv")
        c.close()

    def test_abo_checkout_ohne_zustimmung_oder_mit_abo(self):
        self.assertEqual(self.c.post("/api/abo/checkout", json={"plan": "team", "laufzeit_monate": 3}).status_code, 400)
        c = database.get_db()
        c.execute("UPDATE users SET plan = 'single', plan_gueltig_bis = date('now', '+30 days') WHERE id = ?", (self.u["id"],))
        c.commit()
        c.close()
        r = self.c.post("/api/abo/checkout", json={"plan": "team", "laufzeit_monate": 3, "widerruf_zustimmung": True})
        self.assertEqual(r.status_code, 400)
        self.assertIn("bereits ein aktives Abo", r.json()["detail"])
        self.abo.assert_not_called()

    def test_paket_checkout(self):
        r = self.c.post("/api/abo/paket-checkout", json={"groesse": 500})
        self.assertEqual(r.json(), {"ok": True, "url": STRIPE_URL})
        self.assertEqual(self.paket.call_args.args[1], 500)
        self.assertEqual(self.c.post("/api/abo/paket-checkout", json={"groesse": 7}).status_code, 400)
        self.paket.side_effect = RuntimeError("Stripe weg (fiktiv)")
        self.assertEqual(self.c.post("/api/abo/paket-checkout", json={"groesse": 500}).status_code, 502)


# ─── 6. Oberflaeche, Hilfe, Namen, Auftrag ──────────────────────────────────────────────────────────────────────────────────
class Oberflaeche(unittest.TestCase):
    def setUp(self):
        backend = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
        frontend = next(k for k in (os.path.normpath(os.path.join(backend, "..", "frontend")), os.path.join(backend, "frontend"))
                        if os.path.isdir(k))
        self.js = open(os.path.join(frontend, "inkluagent.js"), encoding="utf-8").read()

    def test_nur_zu_stripe_checkout_und_als_ganze_seite(self):
        self.assertIn("return typeof url === 'string' && /^https:\\/\\/checkout\\.stripe\\.com\\//.test(url);", self.js)
        absenden = self.js[self.js.index("async function inkluagentFormularAbsenden("):self.js.index("function inkluagentZurBezahlung(")]
        self.assertIn("if (res.ok && inkluagentZurBezahlung(data.weiter))", absenden)
        self.assertIn("(window.top || window).location.assign(data.weiter)", absenden)

    def test_mehrere_links_am_feld(self):
        self.assertIn("[f.link].concat(Array.isArray(f.links) ? f.links : [])", self.js)
        feld = self.js[self.js.index("function inkluagentFormularFeld("):self.js.index("function inkluagentFormularFehlerZeigen(")]
        self.assertIn("if (!link || !eigenerPfad(link.url)) return;", feld, "nur Pfade dieser Anwendung")


class Hilfe(unittest.TestCase):
    def test_was_ausserhalb_bleibt(self):
        from inkluagent import hilfe
        with an():
            aus = hilfe.seite()["ausserhalb"]
        self.assertIn("Abo wechseln oder kündigen, Rechnungen und Zahlungsart", aus)
        self.assertNotIn("Abo und Zahlung", aus)
        self.assertFalse({"Zusatz-Credits kaufen", "Abo buchen"} & set(hilfe.formulare()), "ohne Schalter keine Kaufkarten")
        with an():
            self.assertTrue({"Zusatz-Credits kaufen", "Abo buchen"} <= set(hilfe.formulare()))
        self.assertIn("Abo und Zahlung", hilfe.seite()["ausserhalb"], "ohne Schalter wie bisher")

    def test_namen_und_daten(self):
        from inkluagent import daten
        from inkluagent.tools import namen
        for n in ("credits_kaufen_vorbereiten", "abo_buchen_vorbereiten"):
            self.assertIn(n, namen.WERKZEUG_NAMEN)
            self.assertEqual(daten.WERKZEUG_FREMDTEXT[n], ())


class Auftrag(unittest.TestCase):
    def test_passendes_paket(self):
        from inkluagent import auftraege
        self.assertEqual([auftraege._passendes_paket(n) for n in (1, 500, 501, 4000, 99999)], [500, 500, 2500, 5000, 5000])


if __name__ == "__main__":
    unittest.main()
