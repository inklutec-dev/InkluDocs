"""Express-Service im Gespraech (Steve 16.10.2026): Der InkluAgent legt Dokumente selbst in den Express-Warenkorb und bringt
gleich die Karte „Zahlungspflichtig bestellen“; Ansprechpartner und Telefon kommen vom letzten Auftrag.

Anlass: Michael Karbe auf Staging (Konto 5, 16.10.2026): „Ich möchte gerne dies Dokument an den Express Service übergeben“ —
„Mach Du dies“ — der Agent antwortete „Das kann ich nicht für dich übernehmen“, weil express_bestellen_vorbereiten nur einen
schon gefuellten Warenkorb kannte; der Hinweis „Seite 1 prüfen“ ging verloren.

Prueft: document_ids legt ueber express.dokumente_hinzufuegen hinein (dieselbe Funktion wie der Knopf am Dokument) und die
Karte zeigt den Warenkorb mit vorbelegten Hinweisen; nur Dokumente des eigenen Kontos; doppelte und Nicht-PDF werden gezaehlt,
die Gruende stehen nur auf der Karte (Dokumentnamen nie im Werkzeug-Ergebnis fuers Modell); unsinnige Kennungen; Bestellen
bleibt der Knopf; Vorbelegung vom letzten bestellten Auftrag (nur eigene, keine geloeschten, Eingabe des Nutzers geht vor),
auch auf der Seite /express (GET /api/express/stand); Werkzeug-Schema und Prompt.
Wegwerf-Datenbank (/tmp), keine KI, keine Mails.
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_express_agent.py -v
"""
import itertools
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="express_agent_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_KARTEN", "EXPRESS"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.DB_PATH = os.environ["INKLUDOCS_DB"]
database.init_db()

import billing  # noqa: E402
import express  # noqa: E402
import funktionen  # noqa: E402
from inkluagent import karten  # noqa: E402

_NR = itertools.count(1)
EXPRESS_AN = mock.patch.object(funktionen, "EXPRESS", True)
GUTHABEN = mock.patch.object(billing, "verfuegbare_credits", return_value=100000)


def nutzer() -> dict:
    mail = f"express-agent{next(_NR)}-{time.time_ns()}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Erika Mustermann (fiktiv)"), "email": mail}


def projekt(uid: int) -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, 'Vereinsbrief (fiktiv)', '', 'Vereinsbrief (fiktiv)', 'pdf', 'pdf', 'extracted')", (uid,)).lastrowid
    c.commit()
    c.close()
    return pid


def dokument(pid: int, name: str, seiten: int = 2, pdf: bool = True) -> int:
    pfad = os.path.join(TMP, f"{pid}-{next(_NR)}-{name}")
    if pdf:
        import fitz
        d = fitz.open()
        for i in range(seiten):
            d.new_page().insert_text((72, 72), f"Seite {i + 1} (fiktiv)")
        d.save(pfad)
        d.close()
    else:
        with open(pfad, "w", encoding="utf-8") as f:
            f.write("keine PDF (fiktiv)")
    c = database.get_db()
    did = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, 1, ?, ?)",
                    (pid, name, pfad)).lastrowid
    c.commit()
    c.close()
    return did


def bestellter_auftrag(uid: int, name: str, telefon: str, bestellt_am: str = "2026-10-01 10:00:00") -> int:
    c = database.get_db()
    aid = c.execute("INSERT INTO express_auftraege (user_id, status, ansprechpartner, telefon, bestellt_am) "
                    "VALUES (?, 'geliefert', ?, ?, ?)", (uid, name, telefon, bestellt_am)).lastrowid
    c.commit()
    c.close()
    return aid


def vorbereiten(args: dict, uid: int) -> dict:
    with EXPRESS_AN, GUTHABEN:
        return karten.ausfuehren("express_bestellen_vorbereiten", args, uid, None, lambda s: s)


def felder(k: dict) -> dict:
    return {f["name"]: f["wert"] for f in k["felder"]}


class HineinlegenUndKarte(unittest.TestCase):
    def setUp(self):
        karten._KARTEN.clear()
        self.u = nutzer()
        self.pid = projekt(self.u["id"])

    def test_michaels_fall_dokument_hinein_und_karte_mit_hinweis(self):
        d1 = dokument(self.pid, "jubilaeum (fiktiv).pdf", seiten=3)
        r = vorbereiten({"document_ids": [d1], "hinweise": "Seite 1 prüfen"}, self.u["id"])
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["karte"], "express_bestellen")
        self.assertEqual((r["result"]["hineingelegt"], r["result"]["schon_im_warenkorb"], r["result"]["nicht_moeglich"]), (1, 0, 0))
        k = r["anhang"]
        self.assertEqual(k["knopf"], "Zahlungspflichtig bestellen", "bestellen bleibt der Knopf des Nutzers")
        self.assertEqual(felder(k)["hinweise"], "Seite 1 prüfen")
        self.assertFalse(felder(k)["bedingungen"], "den Haken setzt nur der Nutzer")
        self.assertEqual(len(k["zeilen"]), 1)
        self.assertIn("jubilaeum (fiktiv).pdf", k["zeilen"][0])
        korb = express.warenkorb(self.u["id"])
        self.assertEqual([p["document_id"] for p in korb["positionen"]], [d1])
        self.assertEqual(karten._KARTEN[k["karten_id"]]["kontext"]["korb_id"], korb["id"], "Karte gehoert zu genau diesem Korb")
        self.assertEqual(karten._KARTEN[k["karten_id"]]["kontext"]["erwartete_credits"], korb["credits"])
        self.assertNotIn("hinweise", k, "ohne Uebersprungene keine Hinweise auf der Karte")

    def test_nur_eigene_dokumente(self):
        fremd = nutzer()
        fremd_dok = dokument(projekt(fremd["id"]), "fremd (fiktiv).pdf")
        r = vorbereiten({"document_ids": [fremd_dok]}, self.u["id"])
        self.assertFalse(r["ok"])
        self.assertNotIn("anhang", r)
        self.assertEqual(express.warenkorb(self.u["id"])["positionen"], [])
        self.assertEqual(express.warenkorb(fremd["id"])["positionen"], [], "auch beim anderen Konto nichts")
        # gemischt: ein eigenes und ein fremdes — gar nichts hinein (alles oder nichts, wie am Knopf)
        eigen = dokument(self.pid, "eigen (fiktiv).pdf")
        r = vorbereiten({"document_ids": [eigen, fremd_dok]}, self.u["id"])
        self.assertFalse(r["ok"])
        self.assertEqual(express.warenkorb(self.u["id"])["positionen"], [])

    def test_doppelt_und_nicht_pdf_gezaehlt_namen_nur_auf_der_karte(self):
        d1 = dokument(self.pid, "satzung (fiktiv).pdf")
        txt = dokument(self.pid, "notiz-geheimname (fiktiv).txt", pdf=False)
        self.assertTrue(vorbereiten({"document_ids": [d1]}, self.u["id"])["ok"])
        r = vorbereiten({"document_ids": [d1, txt]}, self.u["id"])
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["result"]["hineingelegt"], r["result"]["schon_im_warenkorb"], r["result"]["nicht_moeglich"]), (0, 1, 1))
        self.assertIn("Nicht jedes Dokument", r["result"]["hinweis"])
        modell = json.dumps(r["result"], ensure_ascii=False)
        self.assertNotIn("geheimname", modell, "Dokumentnamen gehen nie ins Werkzeug-Ergebnis fuers Modell")
        self.assertNotIn("satzung", modell)
        hinweise = " ".join(r["anhang"].get("hinweise") or [])
        self.assertIn("notiz-geheimname (fiktiv).txt", hinweise)
        self.assertIn("schon in deiner Auswahl", hinweise)
        self.assertEqual(len(express.warenkorb(self.u["id"])["positionen"]), 1)

    def test_nur_nicht_pdf_ergibt_keine_karte(self):
        txt = dokument(self.pid, "notiz (fiktiv).txt", pdf=False)
        r = vorbereiten({"document_ids": [txt]}, self.u["id"])
        self.assertTrue(r["ok"])
        self.assertTrue(r["result"]["warenkorb_leer"])
        self.assertEqual(r["result"]["nicht_moeglich"], 1)
        self.assertIn("Keines der Dokumente", r["result"]["hinweis"])
        self.assertNotIn("anhang", r)

    def test_leerer_korb_ohne_kennungen_fragt_nach_dokumenten(self):
        r = vorbereiten({}, self.u["id"])
        self.assertTrue(r["result"]["warenkorb_leer"])
        self.assertIn("document_ids", r["result"]["hinweis"])
        self.assertNotIn("In den Express-Warenkorb", r["result"]["hinweis"], "nicht mehr zum Knopf am Dokument schicken")

    def test_unsinnige_kennungen(self):
        for falsch in ([True], [{"id": 1}], [1.5], ["abc"], [None]):
            r = vorbereiten({"document_ids": falsch}, self.u["id"])
            self.assertFalse(r["ok"], falsch)
            self.assertNotIn("anhang", r)
        self.assertEqual(express.warenkorb(self.u["id"])["positionen"], [])

    def test_einzelne_zahl_und_text_zahl_gehen(self):
        d1 = dokument(self.pid, "a (fiktiv).pdf")
        d2 = dokument(self.pid, "b (fiktiv).pdf")
        self.assertTrue(vorbereiten({"document_ids": d1}, self.u["id"])["ok"])
        self.assertTrue(vorbereiten({"document_ids": [str(d2)]}, self.u["id"])["ok"])
        self.assertEqual(sorted(p["document_id"] for p in express.warenkorb(self.u["id"])["positionen"]), sorted([d1, d2]))

    def test_guthaben_reicht_nicht_dokument_liegt_trotzdem_drin(self):
        d1 = dokument(self.pid, "lang (fiktiv).pdf", seiten=5)
        with EXPRESS_AN, mock.patch.object(billing, "verfuegbare_credits", return_value=0):
            r = karten.ausfuehren("express_bestellen_vorbereiten", {"document_ids": [d1]}, self.u["id"], None, lambda s: s)
        self.assertTrue(r["result"]["guthaben_reicht_nicht"])
        self.assertEqual(r["result"]["hineingelegt"], 1)
        self.assertNotIn("anhang", r)
        self.assertEqual(len(express.warenkorb(self.u["id"])["positionen"]), 1)

    def test_ohne_express_kein_hineinlegen(self):
        d1 = dokument(self.pid, "c (fiktiv).pdf")
        with mock.patch.object(funktionen, "EXPRESS", False), GUTHABEN:
            r = karten.ausfuehren("express_bestellen_vorbereiten", {"document_ids": [d1]}, self.u["id"], None, lambda s: s)
        self.assertFalse(r["ok"])
        self.assertEqual(express.warenkorb(self.u["id"])["positionen"], [])


class Vorbelegung(unittest.TestCase):
    def setUp(self):
        karten._KARTEN.clear()
        self.u = nutzer()
        self.d = dokument(projekt(self.u["id"]), "d (fiktiv).pdf")

    def test_vom_letzten_auftrag(self):
        bestellter_auftrag(self.u["id"], "Alt Ansprech (fiktiv)", "040 111 (fiktiv)", "2026-09-01 10:00:00")
        bestellter_auftrag(self.u["id"], "Neu Ansprech (fiktiv)", "040 222", "2026-10-01 10:00:00")
        r = vorbereiten({"document_ids": [self.d]}, self.u["id"])
        f = felder(r["anhang"])
        self.assertEqual((f["ansprechpartner"], f["telefon"]), ("Neu Ansprech (fiktiv)", "040 222"))
        self.assertEqual(sorted(r["result"]["vom_letzten_auftrag"]), ["ansprechpartner", "telefon"])

    def test_eingabe_des_nutzers_geht_vor(self):
        bestellter_auftrag(self.u["id"], "Alt (fiktiv)", "040 222")
        r = vorbereiten({"document_ids": [self.d], "ansprechpartner": "Max (fiktiv)", "telefon": "040 999"}, self.u["id"])
        f = felder(r["anhang"])
        self.assertEqual((f["ansprechpartner"], f["telefon"]), ("Max (fiktiv)", "040 999"))
        self.assertEqual(r["result"]["vom_letzten_auftrag"], [])

    def test_ohne_auftrag_anzeigename_und_kein_telefon(self):
        r = vorbereiten({"document_ids": [self.d]}, self.u["id"])
        f = felder(r["anhang"])
        self.assertEqual((f["ansprechpartner"], f["telefon"]), ("Erika Mustermann (fiktiv)", ""))

    def test_nur_eigene_und_keine_geloeschten_oder_entwuerfe(self):
        fremd = nutzer()
        bestellter_auftrag(fremd["id"], "Fremd (fiktiv)", "040 333")
        bestellter_auftrag(self.u["id"], "", "", "2026-10-02 10:00:00")    # vom Kunden geloescht: Felder geleert
        c = database.get_db()
        c.execute("INSERT INTO express_auftraege (user_id, status, ansprechpartner, telefon) VALUES (?, 'storniert', 'Nie (fiktiv)', '1')",
                  (self.u["id"],))   # nie bestellt
        c.commit()
        c.close()
        self.assertEqual(express.letzte_kontaktdaten(self.u["id"]), {"ansprechpartner": "", "telefon": ""})


class SeiteExpress(unittest.TestCase):
    """Die Seite /express belegt genauso vor (GET /api/express/stand, Formular in templates/express.html)."""

    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.TestClient = main, TestClient

    def client(self, u):
        c = self.TestClient(self.main.app)
        c.cookies.set("token", self.main.create_token(u["id"], u["email"], 0))
        return c

    def test_stand_mit_letzten_kontaktdaten(self):
        u = nutzer()
        with EXPRESS_AN:
            r = self.client(u).get("/api/express/stand")
            if r.status_code == 404:
                self.skipTest("Express-Router in diesem Abbild ohne Schalter nicht eingehaengt")
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual((r.json()["ansprechpartner"], r.json()["telefon"]), ("Erika Mustermann (fiktiv)", ""))
            bestellter_auftrag(u["id"], "Neu (fiktiv)", "040 222")
            r = self.client(u).get("/api/express/stand")
        self.assertEqual((r.json()["ansprechpartner"], r.json()["telefon"]), ("Neu (fiktiv)", "040 222"))

    def test_formular_uebernimmt_das_telefon(self):
        vorlage = os.path.join(HIER, "..", "backend", "templates", "express.html")
        if not os.path.exists(vorlage):
            vorlage = "/app/templates/express.html"
        with open(vorlage, encoding="utf-8") as f:
            html = f.read()
        self.assertIn("if (!byId('exTelefon').value) byId('exTelefon').value = stand.telefon || '';", html)


class WerkzeugUndPrompt(unittest.TestCase):
    def test_schema_mit_document_ids_ohne_enum(self):
        with EXPRESS_AN:
            defs = {d["name"]: d for d in karten.werkzeuge(None)}
        p = defs["express_bestellen_vorbereiten"]["input_schema"]["properties"]["document_ids"]
        self.assertEqual((p["type"], p["items"]), ("array", {"type": "integer"}), "Gemini: Zahlen ohne enum (Staging 14.10.2026)")
        self.assertIn("document_ids", defs["express_bestellen_vorbereiten"]["description"])
        self.assertNotIn("am Dokument „In den Express-Warenkorb“)", defs["express_bestellen_vorbereiten"]["description"])

    def test_prompt_mit_und_ohne_express(self):
        self.assertIn("Dokumente an den Express-Service geben", karten.PROMPT)
        self.assertIn("(document_ids)", karten.PROMPT)
        self.assertNotIn("Express", karten.PROMPT_OHNE_EXPRESS)
        self.assertIn("(einladen_vorbereiten), statt ihn auf eine Seite zu schicken. Fülle vor", karten.PROMPT_OHNE_EXPRESS)


if __name__ == "__main__":
    unittest.main()
