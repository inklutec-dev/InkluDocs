"""KI-Kosten (05.10.2026): Mitschreiben je KI-Aufruf, Preisrechnung, Kontext durch Threads, Auswertung.
Laeuft mit einer EIGENEN Wegwerf-Datenbank (INKLUDOCS_DB), nie gegen die echte:
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_ki_kosten.py -v
"""
import asyncio
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="ki_kosten_test_")
_DB = os.path.join(_TMP, "test.db")
os.environ["INKLUDOCS_DB"] = _DB

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402



def _eigene_db():
    """Vor JEDEM Test auf die eigene Wegwerf-Datenbank zeigen — andere Testdateien setzen beim gemeinsamen Lauf
    (unittest discover) ihre eigene; ki_kosten liest den Pfad bei jedem Aufruf, database beim Import."""
    os.environ["INKLUDOCS_DB"] = _DB
    database.DB_PATH = _DB


_eigene_db()
database.init_db()

import ki_kosten  # noqa: E402


def _zeilen(sql="SELECT * FROM ki_aufrufe ORDER BY id", werte=()):
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, werte)]
    finally:
        conn.close()


def _leeren():
    _eigene_db()
    conn = sqlite3.connect(_DB)
    conn.execute("DELETE FROM ki_aufrufe")
    conn.execute("DELETE FROM system_kv WHERE key = 'ki_preise'")
    conn.execute("DELETE FROM usage_events")
    conn.execute("DELETE FROM buchungen")
    conn.commit()
    conn.close()
    ki_kosten._preis_cache["wert"] = None


def _gemini_antwort(text='{"ok": true}', prompt=1000, aus=100, denk=50, cache=0):
    return {"usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": aus, "thoughtsTokenCount": denk,
                              "cachedContentTokenCount": cache},
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}


class Preisrechnung(unittest.TestCase):
    def setUp(self):
        _leeren()

    def test_gemini_pro_normal_und_lange_eingabe(self):
        n = ki_kosten.nutzung_gemini(_gemini_antwort(prompt=1_000_000, aus=500_000, denk=500_000))
        # 1 Mio Eingabe x 2 $ + 1 Mio Ausgabe (inkl. Denken) x 12 $ — unter der Grenze? Nein: 1 Mio > 200k -> lange Preise.
        self.assertAlmostEqual(ki_kosten.kosten_usd("gemini-3.1-pro-preview", n), 4.0 + 18.0)
        n = ki_kosten.nutzung_gemini(_gemini_antwort(prompt=100_000, aus=100_000, denk=0))
        self.assertAlmostEqual(ki_kosten.kosten_usd("gemini-3.1-pro-preview", n), 0.2 + 1.2)

    def test_gecachte_eingabe_guenstiger(self):
        n = ki_kosten.nutzung_gemini(_gemini_antwort(prompt=1_000_000, aus=0, denk=0, cache=100_000))
        self.assertEqual((n["ein"], n["cache"]), (900_000, 100_000))

    def test_flash_einfuehrungspreis_und_ab_2027(self):
        n = {"ein": 1_000_000, "aus": 1_000_000, "denk": 0, "cache": 0}
        self.assertAlmostEqual(ki_kosten.kosten_usd("gemini-3.8-flash", n, heute="2026-12-31"), 0.75 + 3.75)
        self.assertAlmostEqual(ki_kosten.kosten_usd("gemini-3.8-flash", n, heute="2027-01-01"), 1.50 + 7.50)

    def test_bedrock_eu_kennung_mit_version(self):
        n = ki_kosten.nutzung_anthropic({"usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                                                   "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 0}})
        self.assertAlmostEqual(ki_kosten.kosten_usd("eu.anthropic.claude-sonnet-4-6-v1:0", n), 3.30 + 16.50 + 0.33)
        self.assertAlmostEqual(ki_kosten.kosten_usd("us.anthropic.claude-sonnet-4-6", n), 3.00 + 15.00 + 0.30)

    def test_unbekanntes_modell_ist_unbekannt_nicht_null(self):
        self.assertIsNone(ki_kosten.kosten_usd("gpt-9-irgendwas", {"ein": 10, "aus": 10}))
        ki_kosten.erfasse("openai", "gpt-9-irgendwas", {"ein": 10, "aus": 10})
        z = _zeilen()[-1]
        self.assertIsNone(z["kosten_eur_cent"])
        self.assertIsNone(z["kosten_usd"])

    def test_preisliste_pruefung(self):
        liste = ki_kosten.preise()
        liste["usd_eur"] = 50
        with self.assertRaises(ValueError):
            ki_kosten.pruefe_preisliste(liste)
        liste = ki_kosten.preise()
        liste["modelle"]["Böses Modell"] = {"stufen": [{"ab": "2026-01-01", "ein": 1, "aus": 1}]}
        with self.assertRaises(ValueError):
            ki_kosten.pruefe_preisliste(liste)
        liste = ki_kosten.preise()
        liste["modelle"]["gemini-3.8-flash"]["stufen"][0]["ein"] = "viel"
        with self.assertRaises(ValueError):
            ki_kosten.pruefe_preisliste(liste)

    def test_gespeicherter_preis_gilt_fuer_neue_aufrufe_alte_bleiben(self):
        n = {"ein": 1_000_000, "aus": 0, "denk": 0, "cache": 0}
        ki_kosten.erfasse("gemini", "gemini-3.8-flash", n)
        liste = ki_kosten.preise()
        liste["modelle"]["gemini-3.8-flash"]["stufen"] = [{"ab": "2026-01-01", "ein": 10.0, "aus": 10.0}]
        ki_kosten.speichere_preise(liste)
        ki_kosten.erfasse("gemini", "gemini-3.8-flash", n)
        alt, neu = _zeilen()[-2:]
        self.assertAlmostEqual(alt["kosten_usd"], 0.75)
        self.assertAlmostEqual(neu["kosten_usd"], 10.0)
        self.assertAlmostEqual(neu["kosten_eur_cent"], 10.0 * liste["usd_eur"] * 100)


class Kontext(unittest.TestCase):
    def setUp(self):
        _leeren()

    def test_ohne_kontext_ohne_zuordnung(self):
        ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})
        z = _zeilen()[-1]
        self.assertIsNone(z["user_id"])
        self.assertEqual(z["zweck"], "unbekannt")

    def test_zweck_der_fachfunktion_behaelt_kunde(self):
        @ki_kosten.fuer_zweck("alttext")
        def pipeline():
            ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})
        with ki_kosten.kontext(user_id=7, project_id=3, zweck="chatbot"):
            pipeline()
            ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})
        a, b = _zeilen()[-2:]
        self.assertEqual((a["user_id"], a["project_id"], a["zweck"]), (7, 3, "alttext"))
        self.assertEqual(b["zweck"], "chatbot")

    def test_kontext_kommt_ueber_run_in_executor_an(self):
        async def lauf():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(ki_kosten.KontextExecutor())
            ki_kosten.setze(user_id=11, project_id=12, image_id=13)
            await loop.run_in_executor(None, ki_kosten.erfasse, "gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})
        asyncio.run(lauf())
        z = _zeilen()[-1]
        self.assertEqual((z["user_id"], z["project_id"], z["image_id"]), (11, 12, 13))

    def test_mit_kunde_aus_argumenten_und_kein_leck(self):
        @ki_kosten.mit_kunde
        def arbeit(project_id, document_id, user_id, preis):
            ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})
        t = threading.Thread(target=arbeit, args=(5, 6, 4, 0))
        t.start(); t.join()
        z = _zeilen()[-1]
        self.assertEqual((z["user_id"], z["project_id"], z["document_id"]), (4, 5, 6))
        self.assertEqual(ki_kosten.aktuell(), {})

    def test_mit_kontext_fuer_eigene_threadpools(self):
        from concurrent.futures import ThreadPoolExecutor
        with ki_kosten.kontext(user_id=21, project_id=22):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(ki_kosten.mit_kontext(lambda _: ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})), range(3)))
        zeilen = _zeilen()[-3:]
        self.assertTrue(all(z["user_id"] == 21 and z["project_id"] == 22 for z in zeilen))

    def test_unbekanntes_kontextfeld(self):
        with self.assertRaises(ValueError):
            ki_kosten.setze(passwort="x")

    def test_wirft_nie_ohne_datenbank(self):
        with mock.patch.dict(os.environ, {"INKLUDOCS_DB": os.path.join(_TMP, "gibt-es-nicht.db")}):
            ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})
            self.assertFalse(os.path.exists(os.path.join(_TMP, "gibt-es-nicht.db")))
        with mock.patch.object(ki_kosten, "preise", side_effect=RuntimeError("kaputt")):
            ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1, "aus": 1})


class GeminiClient(unittest.TestCase):
    """Der Haken im echten Client: jede angekommene Antwort zaehlt, auch eine unbrauchbare."""

    def setUp(self):
        _leeren()
        from pipelines.v4 import gemini_client
        self.gc = gemini_client

        class _Profil:
            bild_zuerst = False
            temperatur = None
            bildaufloesung = None
        for p in (mock.patch("pipelines.v4.gemini_auth.kopfzeilen", return_value={}),
                  mock.patch("pipelines.v4.gemini_auth.endpunkt", return_value="https://gemini.invalid/x"),
                  mock.patch("pipelines.v4.anbieter_profil.profil", return_value=_Profil()),
                  mock.patch("pipelines.v4.gemini_client.time.sleep")):
            p.start()
            self.addCleanup(p.stop)

    def test_unbrauchbar_und_dann_gut(self):
        antworten = [_gemini_antwort(text='{"kaputt'), _gemini_antwort(text='{"ok": 1}')]

        def urlopen(req, timeout=None):
            return io.BytesIO(json.dumps(antworten.pop(0)).encode())
        with ki_kosten.kontext(user_id=31, image_id=32):
            with mock.patch("pipelines.v4.gemini_client.urllib.request.urlopen", urlopen):
                self.gc._invoke_gemini("gemini-3.1-pro-preview", "P", None, "AltTextOutput", {"type": "object", "properties": {}}, 100, 0.0, None)
        a, b = _zeilen()[-2:]
        self.assertEqual((a["erfolg"], b["erfolg"]), (0, 1))
        self.assertEqual((b["user_id"], b["image_id"], b["schritt"], b["anbieter"]), (31, 32, "AltTextOutput", "gemini"))
        self.assertEqual((b["tokens_ein"], b["tokens_aus"], b["tokens_denk"]), (1000, 100, 50))
        self.assertGreater(b["kosten_eur_cent"], 0)


class Auswertung(unittest.TestCase):
    def setUp(self):
        _leeren()
        conn = sqlite3.connect(_DB)
        conn.execute("DELETE FROM users")
        conn.execute("INSERT INTO users (id, email, password_hash, display_name) VALUES (101, 'a@beispiel.invalid', 'x', 'Kundin A')")
        conn.execute("INSERT INTO users (id, email, password_hash, display_name) VALUES (102, 'b@beispiel.invalid', 'x', 'Kunde B')")
        conn.execute("INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits) VALUES (101, 101, 'sammellauf', 'bild_generierung', 10)")
        conn.execute("INSERT INTO buchungen (konto_user_id, kunde_name, kunde_email, art, weg, credits, betrag_cent, status) "
                     "VALUES (101, 'Kundin A', 'a@beispiel.invalid', 'paket', 'stripe', 500, 2000, 'ok')")
        conn.commit()
        conn.close()
        for uid, menge in ((101, 1_000_000), (101, 1_000_000), (102, 100_000), (None, 10)):
            with ki_kosten.kontext(user_id=uid, project_id=None if uid is None else 900 + uid, image_id=None if uid is None else 5000 + uid, zweck="alttext"):
                ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": menge, "aus": 0})

    def test_monatsbericht(self):
        import umsatz
        jetzt = umsatz.jetzt_lokal()
        b = ki_kosten.monatsbericht(jetzt.year, jetzt.month)
        self.assertEqual(b["gesamt"]["aufrufe"], 4)
        self.assertEqual(b["kunden"][0]["konto_user_id"], 101)          # teuerste zuerst
        self.assertEqual(b["kunden"][0]["umsatz_cent"], 2000)
        self.assertEqual(b["kunden"][0]["credits"], 10)
        self.assertIn(None, [k["konto_user_id"] for k in b["kunden"]])  # ohne Zuordnung erscheint
        self.assertEqual(b["umsatz_cent"], 2000)
        self.assertIsNotNone(b["kosten_je_credit_cent"])
        k = ki_kosten.kunde_bericht(101, jetzt.year, jetzt.month)
        self.assertEqual(k["projekte"][0]["project_id"], 1001)
        p = ki_kosten.projekt_bericht(1001, 101, jetzt.year, jetzt.month)
        self.assertEqual(p["bilder"][0]["image_id"], 5101)
        self.assertEqual(p["bilder"][0]["aufrufe"], 2)

    def test_konto_loeschen_entfernt_personenbezug_nicht_die_kosten(self):
        vorher = sum(z["kosten_eur_cent"] or 0 for z in _zeilen())
        database.delete_user_data(101)
        zeilen = _zeilen()
        self.assertFalse(any(z["user_id"] == 101 or z["konto_user_id"] == 101 for z in zeilen))
        self.assertAlmostEqual(sum(z["kosten_eur_cent"] or 0 for z in zeilen), vorher)


class KostenJeCredit(unittest.TestCase):
    """Pruefung Entwicklung 05.10.2026, Befund 5: Kosten je Credit nur aus Credits fuer KI-Arbeit und deren Kosten, beides
    ab Messbeginn — Credits aus Tagging, Exporten und Express und Credits VOR dem ersten erfassten Aufruf zaehlen nicht."""

    def setUp(self):
        _leeren()
        conn = sqlite3.connect(_DB)
        conn.execute("DELETE FROM users")
        conn.execute("INSERT INTO users (id, email, password_hash, display_name) VALUES (201, 'c@beispiel.invalid', 'x', 'Kundin C')")
        conn.commit()
        conn.close()

    def _credits(self, aktion, credits, wann="datetime('now')", quelle="sammellauf"):
        conn = sqlite3.connect(_DB)
        conn.execute(f"INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits, created_at) "
                     f"VALUES (201, 201, ?, ?, ?, {wann})", (quelle, aktion, credits))
        conn.commit()
        conn.close()

    def test_nur_ki_credits_ab_messbeginn(self):
        import umsatz
        jetzt = umsatz.jetzt_lokal()
        von, _ = umsatz.zeitraum(jetzt.year, jetzt.month)
        # VOR dem Messbeginn (aber im Monat): zaehlt nicht. Danach: KI-Arbeit zaehlt, Tagging/Export/Express nicht.
        self._credits("bild_generierung", 1000, wann=f"'{von}'")
        with ki_kosten.kontext(user_id=201, zweck="alttext"):
            ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1_000_000, "aus": 0})
        conn = sqlite3.connect(_DB)
        conn.execute("UPDATE ki_aufrufe SET created_at = datetime('now', '-1 minutes')")
        conn.commit()
        conn.close()
        self._credits("bild_generierung", 10)
        self._credits("quickinfo_generierung", 5)
        self._credits("pdf_tagging", 400)
        self._credits("pdf_export", 25)
        self._credits("express_aufbereiten", 500, quelle="express")
        with ki_kosten.kontext(user_id=201, zweck="tagging_ki"):           # KI-Kosten ohne passende Credits
            ki_kosten.erfasse("gemini", "gemini-3.8-flash", {"ein": 1_000_000, "aus": 0})
        b = ki_kosten.monatsbericht(jetzt.year, jetzt.month)
        self.assertEqual(b["ki_credits"], 15)
        alttext = [z for z in _zeilen() if z["zweck"] == "alttext"][0]["kosten_eur_cent"]
        self.assertAlmostEqual(b["ki_kosten_kunden_cent"], alttext)
        self.assertAlmostEqual(b["kosten_je_credit_cent"], alttext / 15)
        self.assertGreater(b["kosten_je_credit_ab"], von)
        self.assertEqual(b["credits"], 1000 + 10 + 5 + 400 + 25)          # die Kundenliste zeigt weiter alles ausser Express

    def test_ohne_messung_keine_zahl(self):
        import umsatz
        jetzt = umsatz.jetzt_lokal()
        self._credits("bild_generierung", 10)
        b = ki_kosten.monatsbericht(jetzt.year, jetzt.month)
        self.assertIsNone(b["kosten_je_credit_cent"])
        self.assertEqual(b["ki_credits"], 0)


class PreisApi(unittest.TestCase):
    """Befunde 4 und 8 (Fehler am Feld, kaputter JSON-Koerper) und 12 (Staffel bleibt bei neuer Preisstufe)."""

    def setUp(self):
        _leeren()
        import ki_kosten_api
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        nutzer = {"id": 1, "email": "admin@beispiel.invalid", "is_admin": True}
        app.include_router(ki_kosten_api.build_router(ki_kosten_api.Deps(
            require_admin=lambda request: nutzer, require_full_admin=lambda request: nutzer,
            pauschale_eur_je_credit=lambda: 0.012)))
        self.api = ki_kosten_api
        self.client = TestClient(app)

    def test_staffel_wird_uebernommen(self):
        r = self.client.post("/api/admin/ki-preise/modell", json={"modell": "gemini-3.1-pro-preview", "ein": "2,50", "aus": "15",
                                                                  "ab": "2027-01-01", "quelle": "Test"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("Staffel", r.json()["message"])
        stufe = next(s for s in ki_kosten.preise()["modelle"]["gemini-3.1-pro-preview"]["stufen"] if s["ab"] == "2027-01-01")
        self.assertEqual((stufe["ein"], stufe["grenze"], stufe["ein_lang"], stufe["aus_lang"]), (2.5, 200000, 4.0, 18.0))
        n = ki_kosten.nutzung_gemini(_gemini_antwort(prompt=1_000_000, aus=0, denk=0))
        self.assertAlmostEqual(ki_kosten.kosten_usd("gemini-3.1-pro-preview", n, heute="2027-02-01"), 4.0)

    def test_cache_preis_bleibt_wenn_feld_leer(self):
        """Nachpruefung Entwicklung 05.10.2026, N5: leeres Feld „Zwischenspeicher“ -> bisheriger Cache-Preis."""
        r = self.client.post("/api/admin/ki-preise/modell", json={"modell": "gemini-3.1-pro-preview", "ein": "2,50", "aus": "15",
                                                                  "cache": "", "ab": "2027-03-01", "quelle": "Test"})
        self.assertEqual(r.status_code, 200, r.text)
        stufe = next(s for s in ki_kosten.preise()["modelle"]["gemini-3.1-pro-preview"]["stufen"] if s["ab"] == "2027-03-01")
        self.assertEqual(stufe["cache"], 0.2)
        r = self.client.post("/api/admin/ki-preise/modell", json={"modell": "gemini-3.1-pro-preview", "ein": "2,50", "aus": "15",
                                                                  "cache": "0,25", "ab": "2027-04-01", "quelle": "Test"})
        stufe = next(s for s in ki_kosten.preise()["modelle"]["gemini-3.1-pro-preview"]["stufen"] if s["ab"] == "2027-04-01")
        self.assertEqual(stufe["cache"], 0.25)                           # eingetragener Wert gewinnt

    def test_vorlage_ist_die_juengste_davor(self):
        stufen = [{"ab": "2026-01-01", "grenze": 1}, {"ab": "2026-06-01", "grenze": 2}, {"ab": "2027-06-01", "grenze": 3}]
        self.assertEqual(self.api.staffel_vorlage(stufen, "2027-01-01")["grenze"], 2)
        self.assertEqual(self.api.staffel_vorlage(stufen, "2025-01-01")["grenze"], 1)
        self.assertEqual(self.api.staffel_vorlage([], "2027-01-01"), {})

    def test_fehler_nennen_das_feld(self):
        basis = {"modell": "testmodell-1", "ein": "1", "aus": "2", "ab": "2027-01-01", "quelle": "Test"}
        for aenderung, feld, anfang in (({"ein": "viel"}, "ein", "Eingabe:"), ({"aus": ""}, "aus", "Ausgabe (mit Denk-Tokens):"),
                                        ({"ab": "morgen"}, "ab", "Gültig ab:"), ({"quelle": ""}, "quelle", "Quelle:"),
                                        ({"modell": "Kein Modell!"}, "modell", "Modellkennung:")):
            r = self.client.post("/api/admin/ki-preise/modell", json=dict(basis, **aenderung))
            self.assertEqual(r.status_code, 400)
            self.assertEqual(r.json()["detail"]["feld"], feld)
            self.assertTrue(r.json()["detail"]["text"].startswith(anfang), r.json()["detail"]["text"])
        r = self.client.post("/api/admin/ki-preise/kurs", json={"usd_eur": "9", "quelle": "EZB"})
        self.assertEqual(r.json()["detail"]["feld"], "usd_eur")

    def test_kaputter_json_ist_400(self):
        for pfad in ("/api/admin/ki-preise/modell", "/api/admin/ki-preise/kurs"):
            for koerper in ("{kaputt", "[1]"):
                r = self.client.post(pfad, content=koerper, headers={"Content-Type": "application/json"})
                self.assertEqual(r.status_code, 400, (pfad, koerper))

if __name__ == "__main__":
    unittest.main()
