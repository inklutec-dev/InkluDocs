"""Paket 1 (Steve 13.10.2026): Formular-Karten des InkluAgent — Schalter funktionen.AGENT_KARTEN, inkluagent/karten.py,
POST /api/agent/karte. Wegwerf-Datenbank (/tmp), keine KI, keine Mails (send_email ist eine Attrappe).
    docker exec -w /app <container> python3 -m unittest /app/tests/test_paket1_karten.py

Prueft: Schema und Pruefung der Felder; Werkzeuge nur mit Schalter und nur im Chat des Kontos (nie in der Demo); Vorschlaege des
Modells werden wie Eingaben geprueft; nur eigene Projekte; „Zur Prüfung einladen“ ueber dieselbe Funktion wie der Dialog im
Projekt, genau einmal, nur vom eigenen Konto, Karte im Verlauf danach „abgeschickt“; „Zahlungspflichtig bestellen“ mit Preis,
Warenkorb, Fassung und Idempotenz vom SERVER, Fehler am Feld, veralteter Korb; Hilfe-Seite nennt die Formulare.
"""
import asyncio
import itertools
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="paket1_karten_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KARTEN", "EXPRESS"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, begleiter, karten, storage  # noqa: E402

_NR = itertools.count(1)
AN = mock.patch.object(funktionen, "AGENT_KARTEN", True)


def nutzer() -> dict:
    mail = f"karten{next(_NR)}-{time.time_ns()}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Erika Mustermann (fiktiv)"), "email": mail}


def projekt(uid: int, name: str = "Vereinsbrief (fiktiv)") -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, 'pdf', 'pdf', 'extracted')", (uid, name, name)).lastrowid
    c.commit()
    c.close()
    return pid


def leeren():
    karten._KARTEN.clear()


class Schema(unittest.TestCase):
    def test_pruefung_der_felder(self):
        werte, fehler = karten.werte_pruefen(karten.EINLADEN, {"email": " keine-adresse ", "rolle": "chef", "name": "x" * 121,
                                                                "fremd": "weg"}, lambda s: s)
        self.assertEqual(set(fehler), {"email", "rolle", "name"})
        self.assertNotIn("fremd", werte)
        werte, fehler = karten.werte_pruefen(karten.EINLADEN, {"email": "lektorin@beispiel.invalid", "rolle": "lektorat"},
                                             lambda s: s)
        self.assertEqual(fehler, {})
        self.assertEqual((werte["email"], werte["rolle"], werte["nachricht"]), ("lektorin@beispiel.invalid", "lektorat", ""))

    def test_haken_nur_mit_true(self):
        for wert in ("true", 1, "ja", None):
            _w, fehler = karten.werte_pruefen(karten.EXPRESS_BESTELLEN, {"ansprechpartner": "Erika (fiktiv)", "bedingungen": wert},
                                              lambda s: s)
            self.assertIn("bedingungen", fehler, wert)
        _w, fehler = karten.werte_pruefen(karten.EXPRESS_BESTELLEN, {"ansprechpartner": "Erika (fiktiv)", "bedingungen": True},
                                          lambda s: s)
        self.assertEqual(fehler, {})

    def test_karte_fuer_die_oberflaeche(self):
        k = karten.karte(karten.EINLADEN, "k1", {"email": "a@beispiel.invalid"}, lambda s: s, text="Projekt „X“")
        self.assertEqual((k["art"], k["karte"], k["karten_id"], k["zustand"]), ("formular", "einladen", "k1", "offen"))
        self.assertEqual(k["knopf"], "Einladung per E-Mail senden")
        felder = {f["name"]: f for f in k["felder"]}
        self.assertEqual(felder["email"]["wert"], "a@beispiel.invalid")
        self.assertTrue(felder["email"]["pflicht"])
        self.assertEqual([a["wert"] for a in felder["rolle"]["auswahl"]], ["kunde", "lektorat"])
        b = {f["name"]: f for f in karten.karte(karten.EXPRESS_BESTELLEN, "k2", {}, lambda s: s)["felder"]}["bedingungen"]
        self.assertEqual((b["art"], b["wert"], b["link"]["url"]), ("haken", False, "/express/bedingungen"))

    def test_jedes_werkzeug_hat_eine_karte(self):
        with mock.patch.object(funktionen, "EXPRESS", True):
            self.assertEqual({d["name"] for d in karten.werkzeuge(None)}, set(karten.NAMEN))
        self.assertEqual({d["name"] for d in karten.werkzeuge(None)}, {"einladen_vorbereiten"}, "Express nur mit Schalter")


class ImGespraech(unittest.TestCase):
    def setUp(self):
        leeren()
        self.u = nutzer()
        self.pid = projekt(self.u["id"])

    def _lauf(self, aufruf, mit_lauf=True, schalter=True):
        gesehen = {}

        class Modell:
            n = 0

            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3):
                gesehen.setdefault("tools", [t["name"] for t in tools])
                Modell.n += 1
                if Modell.n == 1 and aufruf["name"] in gesehen["tools"]:
                    return {"content": [{"type": "tool_use", "id": "t1", **aufruf}], "stop_reason": "tool_use"}
                return {"content": [{"type": "text", "text": "Die Karte steht unten (fiktiv)."}], "stop_reason": "end_turn"}
        lauf = begleiter.Lauf(ui_sprache="de", nutzer_text="Lad Lea ein") if mit_lauf else None
        with mock.patch.object(funktionen, "AGENT_KARTEN", schalter), mock.patch.object(funktionen, "AGENT_KONTO", True):
            r = agent_loop.run_agent(None, self.u["id"], "Lad Lea ein", None, Modell(), begleiter_lauf=lauf)
        return r, gesehen["tools"]

    def test_einladen_vorbereiten_legt_die_karte_ins_gespraech(self):
        r, tools = self._lauf({"name": "einladen_vorbereiten", "input": {
            "project_id": self.pid, "email": "lea@beispiel.invalid", "name": "Lea (fiktiv)", "rolle": "lektorat"}})
        self.assertIn("einladen_vorbereiten", tools)
        k = r["anhang"][0]
        self.assertEqual((k["art"], k["karte"]), ("formular", "einladen"))
        self.assertIn("Vereinsbrief (fiktiv)", k["text"])
        felder = {f["name"]: f["wert"] for f in k["felder"]}
        self.assertEqual((felder["email"], felder["name"], felder["rolle"]), ("lea@beispiel.invalid", "Lea (fiktiv)", "lektorat"))
        self.assertEqual(karten._KARTEN[k["karten_id"]]["kontext"], {"project_id": self.pid}, "das Projekt haelt der Server")

    def test_vorschlaege_werden_geprueft_und_nur_eigene_projekte(self):
        r, _t = self._lauf({"name": "einladen_vorbereiten", "input": {"project_id": self.pid, "email": "keine adresse",
                                                                       "rolle": "admin"}})
        felder = {f["name"]: f["wert"] for f in r["anhang"][0]["felder"]}
        self.assertEqual((felder["email"], felder["rolle"]), ("", "kunde"), "erfundene oder falsche Werte bleiben nicht stehen")
        fremd = projekt(nutzer()["id"], "Fremdes Projekt (fiktiv)")
        r, _t = self._lauf({"name": "einladen_vorbereiten", "input": {"project_id": fremd}})
        self.assertEqual(r["anhang"], [])
        self.assertFalse(r["actions"][0]["ok"])

    def test_nur_mit_schalter_und_nie_in_der_demo(self):
        _r, tools = self._lauf({"name": "einladen_vorbereiten", "input": {}}, schalter=False)
        self.assertNotIn("einladen_vorbereiten", tools)
        _r, tools = self._lauf({"name": "einladen_vorbereiten", "input": {}}, mit_lauf=False)
        self.assertNotIn("einladen_vorbereiten", tools, "ohne Lauf des Kontos (Demo) keine Karten")


class MitServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.TestClient = main, TestClient

    def setUp(self):
        leeren()
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        self.mails = []
        mock.patch.object(self.main, "send_email", side_effect=lambda an, betreff, inhalt, **kw: self.mails.append(an) or True).start()
        self.u = nutzer()
        self.pid = projekt(self.u["id"])

    def tearDown(self):
        mock.patch.stopall()

    def client(self, u=None):
        u = u or self.u
        c = self.TestClient(self.main.app)
        c.cookies.set("token", self.main.create_token(u["id"], u["email"], 0))
        return c

    def _einladen(self, **werte):
        k = karten._einladen_vorbereiten({"project_id": self.pid, **werte}, self.u["id"], None, lambda s: s)["anhang"]
        storage.konto_anhaengen(self.u["id"], None, "assistant", "Hier ist das Formular (fiktiv).", anhang=[k])
        return k

    def test_ohne_schalter_404(self):
        self.assertEqual(self.client().post("/api/agent/karte", json={"karten_id": "x", "werte": {}}).status_code, 404)
        self.assertFalse(self.main._inkluagent_schalter()["karten"])

    def test_einladen_genau_einmal_ueber_dieselbe_funktion(self):
        import sharing
        k = self._einladen(email="lea@beispiel.invalid")
        with AN, mock.patch.object(funktionen, "AGENT_KONTO", True):
            self.assertTrue(self.main._inkluagent_schalter()["karten"])
            r = self.client().post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": {"email": "", "rolle": "kunde"}})
            self.assertEqual(r.status_code, 400, r.text)
            self.assertIn("email", r.json()["fehler"])
            self.assertEqual(self.mails, [])
            werte = {"email": "lea@beispiel.invalid", "name": "Lea (fiktiv)", "rolle": "lektorat", "nachricht": "Bitte prüfen (fiktiv)"}
            r = self.client().post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": werte})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["karte"]["zustand"], "gesendet")
            self.assertIn("lea@beispiel.invalid", r.json()["text"])
            self.assertEqual(self.mails, ["lea@beispiel.invalid"])
            freigaben = sharing.list_shares_for_project(self.pid, self.u["id"])
            self.assertEqual([(f["guest_email"], f["role"]) for f in freigaben], [("lea@beispiel.invalid", "lektorat")])
            zweimal = self.client().post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": werte})
            self.assertEqual(zweimal.status_code, 409)
            self.assertEqual(len(self.mails), 1, "keine zweite Einladung")
        gespeichert = storage.konto_verlauf(self.u["id"])[-1]["anhang"][0]
        self.assertEqual(gespeichert["zustand"], "gesendet", "nach dem Neuladen keine offene Karte mehr")
        self.assertEqual(gespeichert["werte"]["email"], "lea@beispiel.invalid")

    def test_unbekannte_karte_steht_im_verlauf_nicht_mehr_offen(self):
        """14.10.2026 (Rauchtest Staging): nach einem Neustart kennt der Server die Karte nicht mehr — vorher sah sie im Verlauf
        noch offen aus, und erst das Abschicken sagte 404. Jetzt zeigt der Verlauf sie gleich als „nicht mehr gültig“."""
        self._einladen(email="lea@beispiel.invalid")
        with AN, mock.patch.object(funktionen, "AGENT_KONTO", True):
            def karte():
                return self.client().get("/api/agent/verlauf").json()["messages"][-1]["anhang"][0]
            self.assertEqual(karte()["zustand"], "offen")
            karten._KARTEN.clear()   # wie nach einem Neustart
            k = karte()
        self.assertEqual((k["zustand"], k["ergebnis"]), ("abgelaufen", "Diese Karte gilt nicht mehr. Bitte frag im Chat neu."))
        self.assertEqual(storage.konto_verlauf(self.u["id"])[-1]["anhang"][0]["zustand"], "offen", "nur die Anzeige, gespeichert bleibt")

    def test_nur_vom_eigenen_konto(self):
        k = self._einladen(email="lea@beispiel.invalid")
        with AN, mock.patch.object(funktionen, "AGENT_KONTO", True):
            r = self.client(nutzer()).post("/api/agent/karte", json={"karten_id": k["karten_id"],
                                                                     "werte": {"email": "x@beispiel.invalid", "rolle": "kunde"}})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.mails, [])
        self.assertEqual(karten._KARTEN[k["karten_id"]]["zustand"], "offen", "die Karte bleibt fuer ihren Besitzer")

    def test_abgelaufene_karte(self):
        k = self._einladen(email="lea@beispiel.invalid")
        karten._KARTEN[k["karten_id"]]["zeit"] -= karten.GUELTIG_S + 1
        with AN:
            r = self.client().post("/api/agent/karte", json={"karten_id": k["karten_id"],
                                                             "werte": {"email": "lea@beispiel.invalid", "rolle": "kunde"}})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["karte"]["zustand"], "abgelaufen")

    def _express_karte(self, guthaben=500):
        import billing
        import express
        korb = {"id": 41, "dokumente": 1, "seiten": 12, "credits": 240, "fassung": "f-123",
                "positionen": [{"dokument_name": "Satzung (fiktiv)", "seiten": 12, "leistung": "pdf_ua", "credits": 240}]}
        with mock.patch.object(funktionen, "EXPRESS", True), mock.patch.object(express, "warenkorb", return_value=korb), \
                mock.patch.object(express, "einstellungen", return_value={"frist_stunden": 48}), \
                mock.patch.object(express, "leistungen_liste", return_value=[{"schluessel": "pdf_ua", "name": "PDF/UA"}]), \
                mock.patch.object(billing, "verfuegbare_credits", return_value=guthaben):
            return karten._express_vorbereiten({"telefon": "040 123 (fiktiv)"}, self.u["id"], lambda s: s)

    def test_bestellen_mit_preis_und_korb_vom_server(self):
        import express_api
        r = self._express_karte()
        k = r["anhang"]
        self.assertEqual(k["karte"], "express_bestellen")
        self.assertIn("240 Credits", k["text"])
        self.assertIn("„Satzung (fiktiv)“: 12 Seiten, PDF/UA, 240 Credits", k["zeilen"])
        felder = {f["name"]: f["wert"] for f in k["felder"]}
        self.assertEqual(felder["ansprechpartner"], "Erika Mustermann (fiktiv)", "vorbelegt mit dem Anzeigenamen")
        self.assertFalse(felder["bedingungen"], "den Haken setzt nur der Nutzer")
        gesendet = {}

        async def bestellung(user_id, daten, sprache, absender):
            gesendet.update(daten, user_id=user_id)
            return {"auftrag_id": 77, "neu": True}
        werte = {"ansprechpartner": "Erika Mustermann (fiktiv)", "telefon": "", "hinweise": "", "bedingungen": True,
                 "korb_id": 1, "erwartete_credits": 1}   # vom Browser untergeschoben: zaehlt nicht
        with AN, mock.patch.object(funktionen, "EXPRESS", True), mock.patch.object(express_api, "bestellung", side_effect=bestellung):
            a = self.client().post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": werte})
        self.assertEqual(a.status_code, 200, a.text)
        self.assertEqual((gesendet["korb_id"], gesendet["erwartete_credits"], gesendet["fassung"]), (41, 240, "f-123"))
        self.assertTrue(gesendet["idempotenz"].startswith("karte-"))
        self.assertEqual(gesendet["user_id"], self.u["id"])
        self.assertEqual(a.json()["anhang"]["url"], "/express/auftrag/77")
        self.assertIn("77", a.json()["text"])

    def test_bestellen_fehler_am_feld_und_veralteter_korb(self):
        import express
        import express_api
        k = self._express_karte()["anhang"]
        werte = {"ansprechpartner": "Erika (fiktiv)", "telefon": "abc", "bedingungen": True}
        with AN, mock.patch.object(funktionen, "EXPRESS", True), mock.patch.object(
                express_api, "bestellung", side_effect=express.ExpressFehler("Telefon: bitte nur Ziffern.", feld="telefon")):
            a = self.client().post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": werte})
        self.assertEqual(a.status_code, 400)
        self.assertEqual(a.json()["fehler"], {"telefon": "Telefon: bitte nur Ziffern."})
        self.assertEqual(karten._KARTEN[k["karten_id"]]["zustand"], "offen", "korrigieren und erneut abschicken")
        with AN, mock.patch.object(funktionen, "EXPRESS", True), mock.patch.object(
                express_api, "bestellung", side_effect=express.ExpressFehler("Deine Auswahl hat sich geändert.", 409, veraltet=True)):
            a = self.client().post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": werte})
        self.assertEqual(a.status_code, 409)
        self.assertEqual(a.json()["karte"]["zustand"], "abgelaufen")

    def test_keine_karte_ohne_guthaben_oder_mit_leerem_korb(self):
        r = self._express_karte(guthaben=100)
        self.assertNotIn("anhang", r)
        self.assertTrue(r["result"]["guthaben_reicht_nicht"])
        import express
        with mock.patch.object(express, "warenkorb", return_value={"id": None, "positionen": []}):
            r = karten._express_vorbereiten({}, self.u["id"], lambda s: s)
        self.assertTrue(r["result"]["warenkorb_leer"])
        self.assertNotIn("anhang", r)


class Oberflaeche(unittest.TestCase):
    """Quelltext-Pruefungen der Karte im Browser; das Verhalten pruefen die Klickproben (lokal/probe_p1_karten.py)."""

    def setUp(self):
        backend = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
        frontend = next(k for k in (os.path.normpath(os.path.join(backend, "..", "frontend")), os.path.join(backend, "frontend"))
                        if os.path.isdir(k))
        self.js = open(os.path.join(frontend, "inkluagent.js"), encoding="utf-8").read()
        self.css = open(os.path.join(frontend, "style.css"), encoding="utf-8").read()
        self.app = open(os.path.join(backend, "templates", "app.html"), encoding="utf-8").read()

    def test_formular_aus_echten_elementen(self):
        karte = self.js[self.js.index("function inkluagentFormularKarte("):self.js.index("function inkluagentKarteEinfliegen(")]
        self.assertIn("document.createElement('form')", karte)
        self.assertIn("form.setAttribute('aria-labelledby', h.id)", karte, "benanntes Formular")
        self.assertIn("label.htmlFor = id;", karte, "sichtbares label je Feld")
        self.assertIn("if (f.pflicht) el.required = true;", karte)
        self.assertIn("el.setAttribute('aria-describedby', fehler.id);", karte)
        self.assertIn("el.setAttribute('aria-invalid', 'true')", karte)
        self.assertIn("fetch('/api/agent/karte'", karte)
        self.assertIn("fokusAuf(ergebnis);", karte, "nach dem Abschicken liest der Screenreader das Ergebnis")
        self.assertNotIn("innerHTML", karte, "Texte vom Server nur als Text")
        self.assertIn("window.INKLUAGENT && window.INKLUAGENT.karten", karte, "ohne Schalter nur zum Lesen")

    def test_karten_fliegen_nur_bei_ankunft_ein(self):
        melden = self.js[self.js.index("function inkluagentAntwortMelden("):self.js.index("function inkluagentKarteSetzen(")]
        self.assertIn("karten.forEach(inkluagentKarteEinfliegen);", melden)
        self.assertIn("inkluagentKlang(karten.length ? 'karte' : 'fertig');", melden)
        anhang = self.js[self.js.index("function inkluagentAnhangEl("):]
        self.assertNotIn("inkluagentKarteEinfliegen", anhang[:anhang.index("\n}\n")], "nicht beim Laden des Verlaufs")

    def test_bewegung_reduzieren_und_gemeinsame_gestaltung(self):
        self.assertIn("@keyframes inkluagent-karte-kommt", self.css)
        # „Bewegung reduzieren“: die gemeinsame Regel schaltet jede Animation ab — die Karte steht sofort da
        allgemein = self.css[self.css.index("/* Reduced Motion */"):]
        self.assertIn("animation: none !important;", allgemein[:allgemein.index("}\n}")])
        self.assertIn(".inkluagent-karte {", self.css, "Karten-Gestaltung fuer alle Seiten")
        self.assertNotIn(".inkluagent-bestaetigung {", self.app, "nicht mehr nur im Projekt")


class Hilfe(unittest.TestCase):
    def test_hilfe_nennt_die_formulare(self):
        from inkluagent import hilfe
        self.assertEqual(hilfe.seite()["formulare"], [])
        self.assertIn("Team und Gäste einladen", hilfe.seite()["ausserhalb"])
        with AN, mock.patch.object(funktionen, "EXPRESS", True):
            s = hilfe.seite()
        self.assertEqual(s["formulare"], ["Zur Prüfung einladen", "Express-Auftrag bestellen"])
        self.assertIn("Team einladen", s["ausserhalb"])
        self.assertNotIn("Team und Gäste einladen", s["ausserhalb"])


if __name__ == "__main__":
    unittest.main()
