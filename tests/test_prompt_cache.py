"""Prompt-Caching (03.09.2026): Bilddaten ans Ende, fester Anfang cachefaehig.

Prueft ohne Modellaufruf:
  1. Schalter aus  -> Prompts enthalten die Werte inline, kein Marker (wie vorher).
  2. Schalter an   -> fester Teil ist fuer zwei verschiedene Bilder desselben Typs
                      byteidentisch, die Werte stehen nur im BILDDATEN-Block.
  3. Client-Split  -> [fester Text mit cache_control] -> Bild -> [variabler Text];
                      ohne Marker unveraendert Bild -> Text.
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(HERE), "backend"), "/app"):
    sys.path.insert(0, _p)

from prompts.builders import helpers as H  # noqa: E402
from prompts.builders.classification import build_classification_prompt  # noqa: E402
from prompts.builders.combo import build_combined_inventar_beschreibung_prompt  # noqa: E402
from prompts.builders.beschreibung import build_beschreibung_prompt_mini  # noqa: E402
from prompts.components.schemas import ClassificationOutput  # noqa: E402
from pipelines.v4 import orchestrator as o  # noqa: E402
from pipelines.v4 import bedrock_client as bc  # noqa: E402


def _combo(width, height, ctx, hint=None):
    return build_combined_inventar_beschreibung_prompt(
        bildtyp_top="foto", bildtyp_effective="foto_personen",
        enriched_context=ctx, width=width, height=height, user_hint=hint,
    )


class SchalterAusTest(unittest.TestCase):
    def test_werte_inline_kein_marker(self):
        p = _combo(640, 400, "Seite 3: Team beim Workshop")
        self.assertIn("BILDGROESSE: 640x400 Pixel", p)
        self.assertIn("Team beim Workshop", p)
        self.assertNotIn(H.BILDDATEN_MARKER, p)
        self.assertNotIn("siehe Block BILDDATEN", p)
        self.assertFalse(H.bilddaten_verlagert())

    def test_klassifikation_inline(self):
        p = build_classification_prompt("Kontext X", 100, 50, original_alt="Logo Y")
        self.assertIn("- Bildgroesse: 100x50 Pixel", p)
        self.assertIn("Original-Alt vom Autor: Logo Y", p)


class SchalterAnTest(unittest.TestCase):
    def test_fester_teil_identisch_fuer_verschiedene_bilder(self):
        with H.bilddaten_am_ende(True):
            a = _combo(640, 400, "Kontext-A-Marker-9911 mit Namen Zephyrin-9912", hint="Hinweis-9913 bitte kurz")
            b = _combo(1254, 1254, "Ganz anderer Kontext-B-Marker-9914")
        self.assertEqual(a, b, "fester Teil muss unabhaengig von Bilddaten sein")
        self.assertNotIn("640x400", a)
        self.assertNotIn("9911", a); self.assertNotIn("9912", a); self.assertNotIn("9913", a)
        self.assertIn("siehe Block BILDDATEN am Ende dieses Prompts", a)

    def test_mini_und_klassifikation_ohne_werte(self):
        cls = ClassificationOutput(bildtyp="logo", konfidenz="hoch", ist_dekorativ=False,
                                   original_alt_brauchbar=False, klassifikations_begruendung="Test-Begruendung fuer Logo")
        with H.bilddaten_am_ende(True):
            m = build_beschreibung_prompt_mini("logo", cls, "LINK-ZIEL: https://x.de", 120, 40, original_alt="Alt Z")
            k = build_classification_prompt("Kontext K", 100, 50, original_alt="Alt K")
        for p in (m, k):
            self.assertNotIn("Alt Z", p); self.assertNotIn("Alt K", p)
            self.assertNotIn("100x50", p); self.assertNotIn("120x40", p)
            self.assertIn("BILDDATEN", p)

    def test_orchestrator_haengt_block_an(self):
        with mock.patch.dict(os.environ, {"V4_PROMPT_CACHE": "on"}):
            with H.bilddaten_am_ende(o._prompt_cache_an()):
                fest = _combo(640, 400, "Kontext A")
            voll = o._mit_bilddaten(fest, width=640, height=400, enriched_context="Kontext A", user_hint="Hinweis H")
        self.assertTrue(voll.startswith(fest))
        self.assertIn(H.BILDDATEN_MARKER, voll)
        variabel = voll.split(H.BILDDATEN_MARKER, 1)[1]
        self.assertIn("BILDGROESSE: 640x400 Pixel", variabel)
        self.assertIn("Kontext A", variabel)
        self.assertIn("Hinweis H", variabel)

    def test_orchestrator_ohne_schalter_unveraendert(self):
        with mock.patch.dict(os.environ, {"V4_PROMPT_CACHE": "off"}):
            p = o._mit_bilddaten("PROMPT", width=1, height=1, enriched_context="x")
        self.assertEqual(p, "PROMPT")

    def test_verify_prompt_marker_nur_mit_schalter(self):
        with mock.patch.dict(os.environ, {"V4_PROMPT_CACHE": "on"}):
            p_an = o._build_verify_prompt("Ein Alt-Text zur Pruefung, lang genug.", enriched_context="Quelle Q")
        with mock.patch.dict(os.environ, {"V4_PROMPT_CACHE": "off"}):
            p_aus = o._build_verify_prompt("Ein Alt-Text zur Pruefung, lang genug.", enriched_context="Quelle Q")
        self.assertIn(H.BILDDATEN_MARKER, p_an); self.assertNotIn(H.BILDDATEN_MARKER, p_aus)
        self.assertEqual(p_an.replace(H.BILDDATEN_MARKER, ""), p_aus)
        self.assertNotIn("Quelle Q", p_an.split(H.BILDDATEN_MARKER, 1)[0])


class ClientSplitTest(unittest.TestCase):
    def test_bloecke_mit_marker(self):
        bl = bc._content_bloecke("FEST" + H.BILDDATEN_MARKER + "VARIABEL", "aGVsbG8=", "/x.png")
        self.assertEqual([b["type"] for b in bl], ["text", "image", "text"])
        self.assertEqual(bl[0]["text"], "FEST"); self.assertEqual(bl[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(bl[2]["text"], "VARIABEL"); self.assertNotIn("cache_control", bl[2])

    def test_bloecke_ohne_marker_wie_vorher(self):
        bl = bc._content_bloecke("NUR TEXT", "aGVsbG8=", "/x.png")
        self.assertEqual([b["type"] for b in bl], ["image", "text"])
        self.assertNotIn("cache_control", bl[1])

    def test_teilen(self):
        self.assertEqual(bc._prompt_teilen("a" + H.BILDDATEN_MARKER + "b"), ("a", "b"))
        self.assertEqual(bc._prompt_teilen("nur"), ("", "nur"))


if __name__ == "__main__":
    unittest.main()
