"""Korrekturwache im Pruefpass (03.09.2026): Pruefer-Korrekturen werden nie
abgeschnitten — kurze uebernommen, lange einmal gekuerzt, sonst verworfen.
Laeuft ohne Modell (Kuerzen wird ersetzt)."""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(HERE), "backend"), "/app"):
    sys.path.insert(0, _p)

from pipelines.v4 import orchestrator as o  # noqa: E402


def _verify(korr):
    return o.VerifyOutput(alt_text_belegt=False, strittige_aussagen=["x"], korrigierter_alt_text=korr)


class KorrekturwacheTest(unittest.TestCase):
    def test_schema_kappt_nicht_mehr_bei_400(self):
        lang = "Wort " * 100  # 500 Zeichen
        v = _verify(lang)
        self.assertGreater(len(v.korrigierter_alt_text), 400)
        self.assertFalse(v.korrigierter_alt_text.endswith("Wo"))  # kein Wortrest

    def test_kurze_korrektur_wird_uebernommen(self):
        korr, schritt = o._korrektur_absichern("/nix.png", _verify("Vier Menschen am See schauen auf Handy und Tablet."))
        self.assertEqual(schritt, "uebernommen")
        self.assertTrue(korr.startswith("Vier Menschen"))

    def test_lange_korrektur_wird_gekuerzt(self):
        lang = "Sehr lange Korrektur mit vielen Details. " * 12  # > 250
        with mock.patch.object(o, "_kuerze_korrektur", return_value="Kurze Fassung mit den Kernfakten des Bildes."):
            korr, schritt = o._korrektur_absichern("/nix.png", _verify(lang))
        self.assertEqual(schritt, "gekuerzt")
        self.assertEqual(korr, "Kurze Fassung mit den Kernfakten des Bildes.")

    def test_kuerzen_scheitert_dann_verworfen(self):
        lang = "Sehr lange Korrektur mit vielen Details. " * 12
        with mock.patch.object(o, "_kuerze_korrektur", return_value=None):
            korr, schritt = o._korrektur_absichern("/nix.png", _verify(lang))
        self.assertIsNone(korr)
        self.assertEqual(schritt, "verworfen")

    def test_kuerzen_liefert_immer_noch_zu_lang_dann_verworfen(self):
        lang = "Sehr lange Korrektur mit vielen Details. " * 12
        with mock.patch.object(o, "_kuerze_korrektur", return_value="x" * 401):
            korr, schritt = o._korrektur_absichern("/nix.png", _verify(lang))
        self.assertIsNone(korr)
        self.assertEqual(schritt, "verworfen")

    def test_keine_korrektur(self):
        korr, schritt = o._korrektur_absichern("/nix.png", _verify(None))
        self.assertIsNone(korr)
        self.assertEqual(schritt, "")

    def test_prompt_enthaelt_minimaleingriff_und_stilregeln(self):
        p = o._build_verify_prompt("Ein Alt-Text zur Pruefung, lang genug.")
        self.assertIn("MINIMALEINGRIFF", p)
        self.assertIn("STILREGELN", p)


if __name__ == "__main__":
    unittest.main()
