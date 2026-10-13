"""Pruefung 13.10.2026, Runde 2 — Server-Befunde (Berichte PRUEFUNG2-SICHERHEIT/-KORREKTHEIT, Doku docs/INKLUAGENT.md,
„Prüfung 13.10.2026, Runde 2: Server“). Wegwerf-Datenbank (/tmp), keine KI, kein Netz, keine Mails.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_pruefung2_server.py

Prueft:
- Frist der Nachricht auch innerhalb einer Runde (K-N3/S-N5): kein Werkzeug nach der Frist, der Anbieter haelt sie ein
  (providers.base.Uhr, Gemini-Wiederholungen und Rueckfall ohne Stream), FristErreicht endet mit dem Hinweis auf die Frist.
- „Ein Lauf je Konto“ haengt am Lauf, nicht an der Uhr; das Netz gegen haengengebliebene Laeufe (_AGENT_LAEUFT_NETZ_S).
- Laeufe in Faeden des Agenten-Bereichs, Karten-Knoepfe in einem eigenen kleinen Bereich (K-N5), Demo-Chat mit Sperre (S-N11).
- Adressfilter fuer IPv6 (S-N7), Entpacken mit Grenze, alle gepruefte Adressen der Reihe nach (S-N12).
- Keine Ausnahmetexte in HTTPException-Antworten (S-N8).
- Einrahmen nur fuer die Seiten der Positivliste (S-N6).
"""
import asyncio
import gzip
import itertools
import os
import re
import sys
import tempfile
import threading
import time
import types
import unittest
import zlib
from unittest import mock

TMP = tempfile.mkdtemp(prefix="pruefung2_server_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_SICHERHEIT", "INKLUAGENT_FRIST_S", "INKLUAGENT_PARALLEL", "SCAN_ERLAUBE_LOOPBACK"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop  # noqa: E402
from inkluagent.providers.base import FristErreicht, Uhr  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
_NR = itertools.count(1)
FRIST_TEXT = "Das dauert länger, als eine Nachricht dauern darf."


def nutzer() -> dict:
    mail = f"pruefung2-{next(_NR)}-{time.time_ns()}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, name: str = "Projekt (fiktiv)") -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, 'grafik', 'images', 'extracted')", (uid, name, name)).lastrowid
    c.commit()
    c.close()
    return pid


class Uhrwerk:
    """Steuerbare Zeit fuer agent_loop (time.monotonic)."""

    def __init__(self):
        self.jetzt = 1000.0

    def monotonic(self):
        return self.jetzt

    def time(self):
        return time.time()


# ─── Frist ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class UhrDesAnbieters(unittest.TestCase):
    def test_ohne_frist_jeder_aufruf_die_hoechstzeit(self):
        u = Uhr(None, 180)
        self.assertEqual([u.naechster(), u.naechster(), u.naechster()], [180, 180, 180])
        self.assertTrue(u.weiterer_erlaubt())
        u.pruefen()   # ohne Frist keine Gesamtgrenze

    def test_erster_aufruf_mindestens_eine_minute_weitere_nur_mit_zeit(self):
        u = Uhr(time.monotonic() + 5, 180)
        self.assertAlmostEqual(u.naechster(), Uhr.MIN_AUFRUF_S, delta=0.5, msg="ein begonnener Schritt bekommt seine Antwort")
        self.assertFalse(u.weiterer_erlaubt())
        with self.assertRaises(FristErreicht):
            u.naechster()

    def test_viel_zeit(self):
        u = Uhr(time.monotonic() + 1000, 180)
        self.assertEqual(u.naechster(), 180)
        self.assertEqual(u.naechster(), 180)
        u = Uhr(time.monotonic() + 100, 180)
        u.naechster()
        self.assertAlmostEqual(u.naechster(), 100, delta=0.5, msg="ein weiterer Aufruf endet spaetestens mit der Frist")

    def test_stream_endet_mit_der_zeit_seines_aufrufs(self):
        u = Uhr(time.monotonic() - 1000, 180)   # Frist laengst um: der erste Aufruf hat noch 60 s
        u.naechster()
        u.pruefen()
        u._bis = time.monotonic() - 1
        with self.assertRaises(FristErreicht):
            u.pruefen()


class FristInDerSchleife(unittest.TestCase):
    def setUp(self):
        self.u = nutzer()

    def test_nach_der_frist_kein_werkzeug_mehr_auch_nicht_in_derselben_runde(self):
        uhr = Uhrwerk()
        aufrufe = []

        class Modell:
            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3):
                aufrufe.append(1)
                return {"content": [{"type": "tool_use", "id": "a", "name": "projekte_auflisten", "input": {}},
                                    {"type": "tool_use", "id": "b", "name": "projekte_auflisten", "input": {}}],
                        "stop_reason": "tool_use"}

        def vor_dem_werkzeug(name, args):
            uhr.jetzt += 500   # das erste Werkzeug dauert ueber die Frist hinaus

        with mock.patch.object(agent_loop, "time", uhr), mock.patch.object(agent_loop, "_frist_s", return_value=300.0), \
                mock.patch.object(funktionen, "AGENT_KONTO", True):
            r = agent_loop.run_agent(None, self.u["id"], "Zeig meine Projekte", None, Modell(), on_tool=vor_dem_werkzeug)
        self.assertEqual(r["werkzeuge"], ["projekte_auflisten"], "das zweite Werkzeug beginnt nach der Frist nicht")
        self.assertEqual(len(aufrufe), 1, "keine weitere Runde")
        self.assertTrue(r["reply"].startswith(FRIST_TEXT), r["reply"])

    def test_anbieter_bekommt_die_frist_und_meldet_ihr_ende(self):
        gesehen = {}

        class Modell:
            kennt_frist = True

            def invoke_with_tools(self, anthropic_messages, tools, system=None, max_tokens=4096, temperature=0.3, frist=None):
                gesehen["frist"] = frist
                raise FristErreicht()

        with mock.patch.object(funktionen, "AGENT_KONTO", True):
            vorher = time.monotonic()
            r = agent_loop.run_agent(None, self.u["id"], "Hallo", None, Modell())
        self.assertGreater(gesehen["frist"], vorher + 29, "Ende der Nachricht als time.monotonic")
        self.assertTrue(r["reply"].startswith(FRIST_TEXT), r["reply"])
        self.assertEqual(r["intent"], "agentic", "kein Verbindungsfehler: die Antwort wird gespeichert")


class GeminiHaeltDieFrist(unittest.TestCase):
    def setUp(self):
        try:
            from inkluagent.providers import gemini
            from pipelines.v4 import gemini_auth
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"Gemini-Anschluss nicht ladbar: {e}")
        self.gemini = gemini
        for p in (mock.patch.object(gemini_auth, "kopfzeilen", return_value={}),
                  mock.patch.object(gemini, "_endpunkt", return_value="https://beispiel.invalid/m:generateContent"),
                  mock.patch.object(gemini.time, "sleep")):
            p.start()
        self.zeiten = []

        def weg(req, timeout=None):
            self.zeiten.append(timeout)
            raise gemini.urllib.error.URLError("nicht erreichbar (fiktiv)")
        mock.patch.object(gemini.urllib.request, "urlopen", side_effect=weg).start()
        self.p = gemini.GeminiProvider()

    def tearDown(self):
        mock.patch.stopall()

    def _aufruf(self, frist, on_text=None):
        return self.p.invoke_with_tools([{"role": "user", "content": "Hallo"}], [], frist=frist, on_text=on_text)

    def test_ohne_frist_drei_versuche(self):
        with self.assertRaises(self.gemini.GeminiProviderError):
            self._aufruf(None)
        self.assertEqual(self.zeiten, [180, 180, 180])

    def test_kurz_vor_der_frist_keine_wiederholung(self):
        with self.assertRaises(self.gemini.GeminiProviderError):
            self._aufruf(time.monotonic() + 5)
        self.assertEqual(len(self.zeiten), 1)
        self.assertAlmostEqual(self.zeiten[0], Uhr.MIN_AUFRUF_S, delta=0.5)

    def test_kein_rueckfall_ohne_stream_nach_der_frist(self):
        texte = []
        with self.assertRaises(FristErreicht):
            self._aufruf(time.monotonic() + 5, on_text=texte.append)
        self.assertEqual(len(self.zeiten), 1, "nur der Stream, kein zweiter Aufruf")
        self.assertIn(None, texte, "die Seite verwirft den Entwurf")

    def test_der_anbieter_sagt_dass_er_die_frist_kennt(self):
        self.assertTrue(self.gemini.GeminiProvider.kennt_frist)


# ─── Laeufe, Sperre, Arbeitsbereiche ─────────────────────────────────────────────────────────────────────────────────────────
class MitServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            from fastapi.testclient import TestClient
            from inkluagent import chat_engine
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main, cls.TestClient, cls.chat_engine = main, TestClient, chat_engine

    def setUp(self):
        mock.patch.object(self.main, "INKLUAGENT_ENABLED", True).start()
        self.u = nutzer()

    def tearDown(self):
        mock.patch.stopall()
        self.main._AGENT_LAEUFT.pop(self.u["id"], None)

    def client(self):
        c = self.TestClient(self.main.app)
        c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))
        return c

    def test_sperre_haengt_am_lauf_nicht_an_der_uhr(self):
        m, uid = self.main, self.u["id"]
        m._AGENT_LAEUFT[uid] = time.time() - 700   # vorher galt ein Lauf nach 600 s als haengengeblieben
        self.assertTrue(m._agent_laeuft(uid))
        self.assertIsNone(m._lauf_anmelden(uid), "kein zweiter Lauf, solange der erste arbeitet")
        m._AGENT_LAEUFT[uid] = time.time() - m._AGENT_LAEUFT_NETZ_S - 1
        with self.assertLogs("inkludocs", level="ERROR"):
            self.assertFalse(m._agent_laeuft(uid), "nur das Netz gegen einen Faden, der nie zurueckkehrt")
        alt = m._AGENT_LAEUFT[uid]
        neu = m._lauf_anmelden(uid)
        self.assertIsNotNone(neu)
        m._lauf_abmelden(uid, alt)   # der alte Lauf endet doch noch: er traegt den neuen nicht aus
        self.assertTrue(m._agent_laeuft(uid))
        m._lauf_abmelden(uid, neu)
        self.assertFalse(m._agent_laeuft(uid))

    def test_netz_ueber_dem_laengsten_lauf(self):
        """Frist (hoechstens 540 s) + ein Modellaufruf ueber der Frist + ein Werkzeug (kern.warten 180 s) < Netz."""
        self.assertGreater(self.main._AGENT_LAEUFT_NETZ_S, 540 + Uhr.MIN_AUFRUF_S + 180)

    def test_lauf_im_arbeitsbereich_des_agenten(self):
        faden = {}

        def antwort(user_id, message, **kw):
            faden["name"] = threading.current_thread().name
            return {"reply": "Gern (fiktiv).", "intent": "agentic", "werkzeuge": []}
        with mock.patch.object(funktionen, "AGENT_KONTO", True), mock.patch.object(funktionen, "AGENT_UEBERALL", True), \
                mock.patch.object(self.chat_engine, "process_konto_message", side_effect=antwort):
            r = self.client().post("/api/agent/nachricht", json={"message": "Hallo (fiktiv)"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(faden["name"].startswith("inkluagent_") or faden["name"].startswith("inkluagent-"), faden)
        self.assertFalse(faden["name"].startswith("inkluagent-knopf"))

    def test_knoepfe_im_eigenen_bereich(self):
        self.assertTrue(self.main._AGENT_KNOPF_EXECUTOR._thread_name_prefix.startswith("inkluagent-knopf"))
        quelle = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read()
        knopf = quelle[quelle.index("async def _karte_bestaetigen("):quelle.index("# Laufende Antworten je Konto")]
        self.assertIn("run_in_executor(_AGENT_KNOPF_EXECUTOR, _ausfuehren)", knopf)
        self.assertLess(knopf.index("_lauf_anmelden(user[\"id\"])"), knopf.index("run_in_executor("))

    def test_knopf_laeuft_noch_heisst_nicht_erledigt(self):
        """K-N6: dauert Datei ablegen ueber den Knopf laenger als 150 s, sagen Antwort und Karte „Läuft noch“, nicht „Erledigt“."""
        from inkluagent.tools import ausgaben
        angebot = {"werkzeug": "datei_ablegen", "args": {}, "text": "„Brief.pdf“ in „Vereinspost“ hochladen (fiktiv)"}
        laeuft = {"ok": True, "result": {"laeuft_noch": True}, "karten": [{"angebot_id": "a1", "zustand": "laeuft", "titel": "Läuft noch"}]}
        with mock.patch.object(funktionen, "AGENT_KONTO", True), mock.patch.object(funktionen, "AGENT_UEBERALL", True), \
                mock.patch.object(funktionen, "AGENT_KONTO_WERKZEUGE", True), mock.patch.object(funktionen, "AGENT_HOCHLADEN", True), \
                mock.patch.object(ausgaben, "angebot_reservieren", return_value=(((self.u["id"], 0, "datei_ablegen", None), angebot), "")), \
                mock.patch.object(ausgaben, "angebot_loslassen"), mock.patch.object(ausgaben, "karte_ausfuehren", return_value=laeuft):
            r = self.client().post("/api/agent/bestaetigen", json={"angebot_id": "a1"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["reply"], "Läuft noch: „Brief.pdf“ in „Vereinspost“ hochladen (fiktiv)")
        self.assertEqual(r.json()["karte"]["zustand"], "laeuft")

    def test_ohne_ueberall_im_projekt_409(self):
        """Prod-Stand (ohne AGENT_UEBERALL): auch der Chat im Projekt nimmt keine zweite Nachricht an, solange einer laeuft."""
        pid = projekt(self.u["id"])
        start = self.main._lauf_anmelden(self.u["id"])
        try:
            with mock.patch.object(funktionen, "AGENT_UEBERALL", False), mock.patch.object(funktionen, "AGENT_KONTO", False), \
                    mock.patch.object(self.chat_engine, "process_message", side_effect=AssertionError("darf nicht laufen")):
                r = self.client().post(f"/api/projects/{pid}/chat", json={"message": "Noch eine Frage (fiktiv)"})
        finally:
            self.main._lauf_abmelden(self.u["id"], start)
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("arbeitet noch", r.json()["detail"])

    def test_demo_chat_mit_sperre_und_arbeitsbereich(self):
        m = self.main
        uid = self.u["id"]
        projekt(uid, "Demo-Bild (fiktiv)")
        faden = {}

        def antwort(project_id, message, user_id, **kw):
            faden["name"] = threading.current_thread().name
            return {"reply": "Gern (fiktiv).", "intent": "agentic"}
        for p in (mock.patch.object(m.demo_mod, "demo_enabled", return_value=True),
                  mock.patch.object(m.demo_mod, "resolve_session_user_id", return_value=uid),
                  mock.patch.object(m.demo_mod, "check_chat", return_value={"allowed": True}),
                  mock.patch.object(m.demo_mod, "touch_session"), mock.patch.object(m.demo_mod, "record_chat"),
                  mock.patch.object(m.demo_mod, "remaining_for", return_value={})):
            p.start()
        c = self.TestClient(m.app)
        start = m._lauf_anmelden(uid)
        try:
            with mock.patch.object(self.chat_engine, "process_message", side_effect=AssertionError("darf nicht laufen")):
                besetzt = c.post("/api/demo/chat", json={"message": "Hallo (fiktiv)"})
        finally:
            m._lauf_abmelden(uid, start)
        self.assertEqual(besetzt.status_code, 409, besetzt.text)
        with mock.patch.object(self.chat_engine, "process_message", side_effect=antwort):
            r = c.post("/api/demo/chat", json={"message": "Hallo (fiktiv)"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(faden["name"].startswith("inkluagent"), faden)
        self.assertFalse(m._agent_laeuft(uid), "nach der Antwort ausgetragen")

    def test_einrahmen_nur_fuer_die_positivliste(self):
        c = self.TestClient(self.main.app)
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True):
            for pfad, erwartet in (("/einstellungen", "SAMEORIGIN"), ("/ablage?x=1", "SAMEORIGIN"), ("/login", "SAMEORIGIN"),
                                   ("/forgot", "DENY"), ("/reset", "DENY"), ("/dashboard", "DENY"), ("/einstellungenx", "DENY"),
                                   ("/api/me", "DENY")):
                r = c.get(pfad, follow_redirects=False)
                self.assertEqual(r.headers.get("x-frame-options"), erwartet, pfad)
                self.assertEqual(r.headers.get("content-security-policy"),
                                 "frame-ancestors 'self'" if erwartet == "SAMEORIGIN" else "frame-ancestors 'none'", pfad)
        with mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", False):
            self.assertEqual(c.get("/einstellungen", follow_redirects=False).headers.get("x-frame-options"), "DENY",
                             "ohne Schalter rahmt niemand ein")


class Positivliste(unittest.TestCase):
    def test_genau_oder_unterpfad(self):
        from inkluagent import fenster
        self.assertTrue(fenster.einrahmbar("/einstellungen"))
        self.assertTrue(fenster.einrahmbar("/projekte/12"))
        self.assertTrue(fenster.einrahmbar("/hilfe?thema=x#anker"))
        self.assertTrue(fenster.einrahmbar("/login"))
        self.assertFalse(fenster.im_fenster("/login"), "die Anmeldung wird eingerahmt, aber nicht im Fenster geoeffnet")
        for pfad in ("/forgot", "/reset", "/dashboard", "/einladung/abc", "/freigabe/abc", "/projektex", "", "einstellungen"):
            self.assertFalse(fenster.einrahmbar(pfad), pfad)


# ─── Fremde Adressen ─────────────────────────────────────────────────────────────────────────────────────────────────────────
class FremdeAdressen(unittest.TestCase):
    def setUp(self):
        try:
            import netz_sicher
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"netz_sicher nicht ladbar: {e}")
        self.ns = netz_sicher

    def test_ipv6_nur_globaler_unicast_und_ipv4_formen_wie_ipv4(self):
        for ip in ("::7f00:1", "::a9fe:a9fe", "::ffff:0:7f00:1", "fec0::1", "fc00::1", "fe80::1", "ff02::1", "::", "::1",
                   "2001:db8::1", "::ffff:10.0.0.1"):
            self.assertFalse(self.ns.adresse_erlaubt(ip), ip)
        for ip in ("2a00:1450:4001:80b::200e", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808"):
            self.assertTrue(self.ns.adresse_erlaubt(ip), ip)

    def _antwort(self, kodierung, stuecke):
        import httpx

        class Antwort:
            headers = httpx.Headers({"content-encoding": kodierung} if kodierung else {})

            async def aiter_raw(self):
                for s in stuecke:
                    yield s
        return Antwort()

    @staticmethod
    def _stuecke(daten, groesse=65536):
        return [daten[i:i + groesse] for i in range(0, len(daten), groesse)]

    def test_entpackt_nie_ueber_die_grenze(self):
        from fastapi import HTTPException
        bombe = gzip.compress(b"\0" * (64 * 1024 * 1024))   # 64 MB entpackt, wenige KB gepackt
        gesehen = []
        echt = self.ns._Entpacker.stueck

        def stueck(selbst, daten, hoechstens):
            out = echt(selbst, daten, hoechstens)
            gesehen.append(len(out))
            return out
        with mock.patch.object(self.ns._Entpacker, "stueck", stueck), self.assertRaises(HTTPException) as fehler:
            asyncio.run(self.ns.lesen_bis(self._antwort("gzip", self._stuecke(bombe)), 1024 * 1024))
        self.assertEqual(fehler.exception.status_code, 413)
        self.assertLessEqual(max(gesehen), 1024 * 1024 + 1, "je Stueck hoechstens ein Byte ueber die Grenze")

    def test_kodierungen(self):
        import httpx
        text = ("Grüße aus der Musterstraße (fiktiv). " * 2000).encode()
        roh_deflate = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        faelle = {"gzip": gzip.compress(text), "deflate": zlib.compress(text),
                  "x-gzip": gzip.compress(text)}
        for art, gepackt in faelle.items():
            self.assertEqual(asyncio.run(self.ns.lesen_bis(self._antwort(art, self._stuecke(gepackt, 1000)), None)), text, art)
        roh = roh_deflate.compress(text) + roh_deflate.flush()
        self.assertEqual(asyncio.run(self.ns.lesen_bis(self._antwort("deflate", self._stuecke(roh, 1000)), None)), text,
                         "deflate ohne zlib-Rahmen wie bei httpx")
        self.assertEqual(asyncio.run(self.ns.lesen_bis(self._antwort("", [text]), None)), text)
        self.assertEqual(asyncio.run(self.ns.lesen_bis(self._antwort("br", [b"unveraendert"]), None)), b"unveraendert",
                         "Unbekanntes bleibt wie bei httpx unveraendert")
        with self.assertRaises(httpx.DecodingError):
            asyncio.run(self.ns.lesen_bis(self._antwort("gzip, gzip", [gzip.compress(gzip.compress(text))]), None))
        with self.assertRaises(httpx.DecodingError):
            asyncio.run(self.ns.lesen_bis(self._antwort("gzip", [b"kein gzip"]), None))

    def test_alle_geprueften_adressen_der_reihe_nach(self):
        import httpcore

        async def aufloesen(host, port, **kw):
            return [(10, 1, 6, "", ("2606:4700:4700::1111", port, 0, 0)), (2, 1, 6, "", ("93.184.216.34", port))]

        async def los():
            netz = self.ns._PruefendesNetz()
            netz._netz = mock.Mock()
            netz._netz.connect_tcp = mock.AsyncMock(side_effect=[httpcore.ConnectError("IPv6 nicht erreichbar"), "verbunden"])
            loop = asyncio.get_running_loop()
            with mock.patch.object(loop, "getaddrinfo", side_effect=aufloesen):
                ergebnis = await netz.connect_tcp("beispiel.invalid", 443, timeout=5)
            return ergebnis, [a.args[0] for a in netz._netz.connect_tcp.call_args_list]
        ergebnis, versucht = asyncio.run(los())
        self.assertEqual(ergebnis, "verbunden")
        self.assertEqual(versucht, ["2606:4700:4700::1111", "93.184.216.34"])

    def test_scanner_zaehlt_groessere_fassungen_und_hat_zeitgrenzen(self):
        quelle = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read()
        scan = quelle[quelle.index("async def scan_url("):quelle.index("async def scan_url(") + 30000]
        self.assertIn("geladen_bytes += len(larger_response.content)", scan)
        self.assertIn("gesamt_s=_SCAN_ABRUF_BILD_S", scan)
        self.assertIn("gesamt_s=_SCAN_ABRUF_SEITE_S", scan)
        self.assertIn("time.monotonic() > bilder_bis", scan)


class GesamtzeitJeAbruf(unittest.TestCase):
    def test_troepfelnder_server(self):
        try:
            import httpx
            import main
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"main nicht ladbar: {e}")

        class Troepfeln(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    await asyncio.sleep(0.05)
                    yield b"x"

        def antwort(anfrage):
            return httpx.Response(200, stream=Troepfeln())

        async def los():
            async with httpx.AsyncClient(transport=httpx.MockTransport(antwort)) as c:
                return await main._safe_fetch(c, "http://127.0.0.1/langsam", max_bytes=10_000, gesamt_s=0.3)
        with mock.patch.dict(os.environ, {"SCAN_ERLAUBE_LOOPBACK": "1"}), self.assertRaises(httpx.TimeoutException):
            asyncio.run(los())


# ─── Keine Ausnahmetexte in Antworten ────────────────────────────────────────────────────────────────────────────────────────
class KeineAusnahmetexte(unittest.TestCase):
    def test_main_schreibt_keine_ausnahmetexte_in_httpexception(self):
        """S-N8: Texte von Bibliotheken (Pfade, Interna) gehoeren ins Log. Erlaubt ist str(e) nur fuer eigene Fehlerklassen, deren
        Texte fuer Menschen geschrieben sind (DocxFehler, FormularFehler, umsatz: ValueError/LookupError)."""
        zeilen = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read().splitlines()
        formatiert = re.compile(r"detail=f[\"'][^\"']*\{(str\()?(e|exc|err|ex|fehler)\)?[}!:]")   # {e}, {str(e)}, {e!r}
        self.assertEqual([i + 1 for i, z in enumerate(zeilen) if formatiert.search(z)], [])
        eigene = re.compile(r"except (DocxFehler|FormularFehler|ValueError|LookupError) as e:")
        for i, z in enumerate(zeilen):
            if re.search(r"detail=str\((e|exc|err|ex)\)", z):
                davor = "\n".join(zeilen[max(0, i - 8):i])
                self.assertTrue(eigene.search(davor), f"main.py:{i + 1}: str(e) ohne eigene Fehlerklasse")


if __name__ == "__main__":
    unittest.main()
