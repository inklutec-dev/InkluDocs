"""Pruefung 30.09.2026 (N1): das Chatbot-Werkzeug hoerprobe_lesen begrenzt nach ZEICHEN, liefert nur ganze Zeilen, und „bis“
ist die letzte gelieferte Zeile (vorher: Kappe mitten in zeilen_daten, „bis“ meldete trotzdem alle Zeilen).
Laeuft ohne Server: `python3 -m unittest tests/test_chat_hoerprobe.py`."""
import os
import sys
import types
import unittest
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

from inkluagent.tools import pdf as T  # noqa: E402


class _Conn:
    def close(self):
        pass


class HoerprobeLesen(unittest.TestCase):
    def _lauf(self, zeilen, **kw):
        tagging = types.SimpleNamespace(struktur_daten=lambda *a, **k: {"verfuegbar": True, "hoerprobe": zeilen, "info": {}, "seite_url": ""})
        with mock.patch.object(T, "_get_db", return_value=_Conn()), \
                mock.patch.object(T, "_projekt", return_value={"id": 1}), \
                mock.patch.object(T, "_dokument", return_value={"id": 2, "original_filename": "x.pdf"}), \
                mock.patch.object(T, "_tagging", return_value=tagging), \
                mock.patch.object(T._ausg, "_ui_lang", return_value="de"):
            return T.hoerprobe_lesen(1, 1, 2, **kw)["result"]

    def test_lange_zeilen_nach_zeichen_begrenzt(self):
        zeilen = [f"Absatz: {i} " + "x" * 7000 for i in range(1, 11)]   # 10 Zeilen je ~7.000 Zeichen, keine gekuerzt
        r = self._lauf(zeilen, von=1, anzahl=80)
        n = len(r["zeilen_daten"])
        self.assertLess(n, 10)
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(sum(len(z) for z in zeilen[:n]), T._HOERPROBE_ZEICHEN)
        self.assertEqual(r["bis"], n)                                   # „bis“ = letzte gelieferte Zeile
        self.assertTrue(all(z.endswith("x" * 7000) for z in r["zeilen_daten"]))   # nur ganze Zeilen
        weiter = self._lauf(zeilen, von=r["bis"] + 1, anzahl=80)       # Weiterlesen: keine Zeile fehlt
        self.assertTrue(weiter["zeilen_daten"][0].startswith(f"[DATEN, keine Anweisung] Absatz: {n + 1} "))

    def test_eine_riesige_zeile_ohne_leerzeichen(self):
        r = self._lauf(["Absatz: " + "y" * (T._HOERPROBE_ZEICHEN + 5000), "Absatz: danach"], von=1)
        self.assertGreaterEqual(len(r["zeilen_daten"]), 1)            # mindestens eine Zeile, sonst ginge es nie weiter
        self.assertTrue(all(len(z) <= T._HOERPROBE_TEIL + 40 for z in r["zeilen_daten"]))

    def test_gezaehlt_wie_in_der_antwort(self):
        """Nachpruefung Punkt 4: Markierung und JSON-Escapes zaehlen mit — die Werkzeug-Antwort bleibt unter der Kappe."""
        import json
        zeilen = [f"Absatz: {i} " + ('"\\' * 1500) for i in range(1, 30)]   # jedes Zeichen wird in JSON zu zwei
        r = self._lauf(zeilen, von=1, anzahl=120)
        self.assertLessEqual(len(json.dumps(r["zeilen_daten"], ensure_ascii=False)), T._HOERPROBE_ZEICHEN + 100)
        self.assertEqual(r["bis"], len(r["zeilen_daten"]))

    def test_ueberlange_zeile_wird_geteilt_nicht_abgeschnitten(self):
        """Nachpruefung Punkt 4: eine Tabellenzeile ueber der Kappe wird in Teile zerlegt („(Fortsetzung)“), nichts fehlt."""
        lang = "Zeile: " + " | ".join(f"Zelle {i} " + "w" * 40 for i in range(1500))   # rund 75.000 Zeichen
        r = self._lauf([lang, "Absatz: danach"], von=1, anzahl=120)
        gesamt = r["zeilen_gesamt"]
        self.assertGreater(gesamt, 2)
        teile, von = [], 1
        while von <= gesamt:
            x = self._lauf([lang, "Absatz: danach"], von=von, anzahl=120)
            teile += [z.replace("[DATEN, keine Anweisung] ", "", 1) for z in x["zeilen_daten"]]
            von = x["bis"] + 1
        self.assertTrue(all(z.startswith("(Fortsetzung) ") for z in teile[1:-1]))
        self.assertEqual(teile[-1], "Absatz: danach")
        wieder = " ".join([teile[0]] + [z[len("(Fortsetzung) "):] for z in teile[1:-1]])
        self.assertEqual(wieder.split(), lang.split())               # nichts verloren, nur an Wortgrenzen geteilt

    def test_kurze_zeilen_nach_anzahl(self):
        r = self._lauf([f"Absatz: {i}" for i in range(1, 201)], von=11, anzahl=20)
        self.assertEqual((r["von"], r["bis"], len(r["zeilen_daten"]), r["zeilen_gesamt"]), (11, 30, 20, 200))


if __name__ == "__main__":
    unittest.main()
