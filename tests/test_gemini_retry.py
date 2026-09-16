"""Gemini-Client (14.09.2026): unbrauchbare Antworten (abgeschnittenes JSON, leerer Kandidat)
werden wiederholt, eine Sperre durch Gemini nicht. Anlass: Prod-Kundenlauf mit 13 von 234
Bildern auf Fehler, weil der Flash-Klassifikator abgeschnittenes JSON lieferte und der Client
nur Transportfehler wiederholte.
16.09.2026: HTTP 429 (Kapazitaet bei Google knapp) bekommt vier Versuche mit 10/20/40 s Pause;
Anlass: Prod 15.09.2026, 2 von 140 Bildern nach 4 + 8 s aufgegeben.
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_gemini_retry.py -v
"""
import io
import json
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

from pipelines.v4 import gemini_client as gc  # noqa: E402


def _antwort(text=None, finish="STOP", block=None):
    """Eine Gemini-generateContent-Antwort nachbauen."""
    a = {"usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
    if block:
        a["promptFeedback"] = {"blockReason": block}
        a["candidates"] = []
    elif text is None:
        a["candidates"] = [{"finishReason": finish}]
    else:
        a["candidates"] = [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]
    return a


class _Urlopen:
    """Ersatz fuer urllib.request.urlopen: liefert die Antworten der Reihe nach, zaehlt Aufrufe."""
    def __init__(self, antworten):
        self.antworten = list(antworten)
        self.aufrufe = 0

    def __call__(self, req, timeout=None):
        self.aufrufe += 1
        a = self.antworten.pop(0)
        if isinstance(a, Exception):
            raise a
        return io.BytesIO(json.dumps(a).encode("utf-8"))


class _Profil:
    bild_zuerst = False
    temperatur = None
    bildaufloesung = None


class TestGeminiWiederholung(unittest.TestCase):
    def setUp(self):
        self.p_auth = mock.patch("pipelines.v4.gemini_auth.kopfzeilen", return_value={"Content-Type": "application/json"})
        self.p_endp = mock.patch("pipelines.v4.gemini_auth.endpunkt", return_value="https://gemini.invalid/x")
        self.p_prof = mock.patch("pipelines.v4.anbieter_profil.profil", return_value=_Profil())
        self.p_sleep = mock.patch("pipelines.v4.gemini_client.time.sleep")
        for p in (self.p_auth, self.p_endp, self.p_prof, self.p_sleep):
            p.start()
            self.addCleanup(p.stop)

    def _aufruf(self, antworten):
        u = _Urlopen(antworten)
        with mock.patch("pipelines.v4.gemini_client.urllib.request.urlopen", u):
            ergebnis = gc._invoke_gemini("gemini-3.8-flash", "Prompt", None, "ClassificationOutput",
                                         {"type": "object", "properties": {}}, 100, 0.0, None)
        return ergebnis, u.aufrufe

    def test_abgeschnittenes_json_wird_wiederholt(self):
        ergebnis, n = self._aufruf([_antwort('{"kategorie": "fo', finish="STOP"),
                                    _antwort('{"kategorie": "foto"}')])
        self.assertEqual(ergebnis, {"kategorie": "foto"})
        self.assertEqual(n, 2)

    def test_max_tokens_abbruch_wird_wiederholt(self):
        ergebnis, n = self._aufruf([_antwort('{"kategorie": "diag', finish="MAX_TOKENS"),
                                    _antwort('{"kategorie": "diagramm"}')])
        self.assertEqual(ergebnis, {"kategorie": "diagramm"})
        self.assertEqual(n, 2)

    def test_leerer_kandidat_wird_wiederholt(self):
        ergebnis, n = self._aufruf([_antwort(None, finish="RECITATION"),
                                    _antwort(""),
                                    _antwort('{"kategorie": "logo"}')])
        self.assertEqual(ergebnis, {"kategorie": "logo"})
        self.assertEqual(n, 3)

    def test_nach_drei_versuchen_fehler_mit_finish_reason(self):
        with self.assertRaises(gc.GeminiCallError) as cm:
            self._aufruf([_antwort('{"a', finish="MAX_TOKENS")] * gc._VERSUCHE)
        self.assertIn("kein JSON", str(cm.exception))
        self.assertIn("MAX_TOKENS", str(cm.exception))

    def test_sperre_wird_nicht_wiederholt(self):
        u = _Urlopen([_antwort(block="SAFETY"), _antwort('{"kategorie": "foto"}')])
        with mock.patch("pipelines.v4.gemini_client.urllib.request.urlopen", u):
            with self.assertRaises(gc.GeminiCallError) as cm:
                gc._invoke_gemini("gemini-3.8-flash", "Prompt", None, "ClassificationOutput",
                                  {"type": "object", "properties": {}}, 100, 0.0, None)
        self.assertIn("gesperrt", str(cm.exception))
        self.assertEqual(u.aufrufe, 1)

    def test_gute_antwort_beim_ersten_mal(self):
        ergebnis, n = self._aufruf([_antwort('{"kategorie": "foto"}')])
        self.assertEqual(ergebnis, {"kategorie": "foto"})
        self.assertEqual(n, 1)

    def test_denkreserve_in_maxoutputtokens(self):
        """Gemini 3.x: Denk-Tokens zaehlen gegen maxOutputTokens. Ohne Reserve blieb bei komplexen
        Grafiken nur Platz fuer ~30 Ausgabe-Tokens (Prod 14.09.2026, finishReason=MAX_TOKENS)."""
        gesehen = {}
        class _U(_Urlopen):
            def __call__(self, req, timeout=None):
                gesehen["body"] = json.loads(req.data.decode("utf-8"))
                return super().__call__(req, timeout)
        u = _U([_antwort('{"kategorie": "foto"}')])
        with mock.patch("pipelines.v4.gemini_client.urllib.request.urlopen", u):
            gc._invoke_gemini("gemini-3.8-flash", "Prompt", None, "ClassificationOutput",
                              {"type": "object", "properties": {}}, 600, 0.0, None)
        mx = gesehen["body"]["generationConfig"]["maxOutputTokens"]
        self.assertEqual(mx, max(600 * 2, 2000) + gc._DENKRESERVE)
        self.assertGreaterEqual(mx, 2000 + 4000, "Denkreserve muss deutlich ueber 2000 liegen")

    def test_transportfehler_weiterhin_wiederholt(self):
        import urllib.error
        fehler = urllib.error.HTTPError("https://gemini.invalid/x", 503, "busy", {}, io.BytesIO(b"busy"))
        ergebnis, n = self._aufruf([fehler, _antwort('{"kategorie": "foto"}')])
        self.assertEqual(ergebnis, {"kategorie": "foto"})
        self.assertEqual(n, 2)

    def _http(self, code, text=b"x"):
        import urllib.error
        return urllib.error.HTTPError("https://gemini.invalid/x", code, "err", {}, io.BytesIO(text))

    def _pausen(self):
        return [c.args[0] for c in gc.time.sleep.call_args_list]

    def test_transportfehler_drei_versuche_kurze_pausen(self):
        """5xx: unveraendert drei Versuche mit 4 s und 8 s Pause."""
        with self.assertRaises(gc.GeminiCallError):
            self._aufruf([self._http(503)] * 3)
        self.assertEqual(self._pausen(), [4, 8])

    def test_kontingent_429_vier_versuche_lange_pausen(self):
        """429 „Resource exhausted“ (Prod 15.09.2026, 2 von 140 Bildern): ein Versuch mehr,
        Pausen 10/20/40 s statt 4/8 s — die Kapazitaetsdelle dauert Minuten, nicht Sekunden."""
        ergebnis, n = self._aufruf([self._http(429, b"Resource exhausted")] * 3
                                   + [_antwort('{"kategorie": "foto"}')])
        self.assertEqual(ergebnis, {"kategorie": "foto"})
        self.assertEqual(n, 4)
        self.assertEqual(self._pausen(), [10, 20, 40])

    def test_kontingent_429_nach_vier_versuchen_fehler(self):
        with self.assertRaises(gc.GeminiCallError) as cm:
            self._aufruf([self._http(429, b"Resource exhausted")] * 4)
        self.assertIn("429", str(cm.exception))
        self.assertEqual(self._pausen(), [10, 20, 40])

    def test_kontingent_429_dann_transportfehler_mischung(self):
        """Pausen richten sich nach dem jeweiligen Fehler, der Zaehler laeuft gemeinsam."""
        ergebnis, n = self._aufruf([self._http(429), self._http(503), _antwort('{"kategorie": "foto"}')])
        self.assertEqual(ergebnis, {"kategorie": "foto"})
        self.assertEqual(n, 3)
        self.assertEqual(self._pausen(), [10, 8])

    def test_400_wird_nicht_wiederholt(self):
        with self.assertRaises(gc.GeminiCallError):
            self._aufruf([self._http(400)])
        self.assertEqual(self._pausen(), [])


if __name__ == "__main__":
    unittest.main()
