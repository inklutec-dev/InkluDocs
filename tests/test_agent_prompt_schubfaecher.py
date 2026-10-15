"""Schubfaecher des InkluAgent-Systemprompts (inkluagent/prompts/schubfaecher.py, Umbau 15.10.2026).

Prueft den Aufbau, nicht den Wortlaut (den halten tests/snapshots/agent_prompts/ fest): Projektart-Weiche wie die Werkzeuge,
feste Reihenfolge, jedes Fach nur in seinen Projektarten, in der Demo nie ein Fach mit demo=False, jedes genannte Werkzeug gibt
es wirklich, ein Zusammensetzer fuer Projekt, Konto und Gespraech.
"""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="schubfaecher_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, auftraege, begleiter, karten, konto, rueckmeldung  # noqa: E402
from inkluagent.prompts import schubfaecher as sf  # noqa: E402

try:
    import stripe_zahlung  # noqa: E402
except Exception:  # noqa: BLE001
    stripe_zahlung = None

ALLE_AN = ("AGENT_HOCHLADEN", "AGENT_GESPRAECHE", "AGENT_BEGLEITER", "AGENT_UEBERALL", "AGENT_KONTO", "AGENT_KARTEN",
           "AGENT_KAUFEN", "AGENT_RUECKMELDUNG", "AGENT_AUFTRAEGE", "AGENT_BILD_WERKZEUGE", "AGENT_KONTO_WERKZEUGE",
           "EXPRESS")


class alle_an:
    def __enter__(self):
        self.p = [mock.patch.object(funktionen, n, True) for n in ALLE_AN]
        if stripe_zahlung is not None:
            self.p.append(mock.patch.object(stripe_zahlung, "AKTIV", True))
        for p in self.p:
            p.start()

    def __exit__(self, *a):
        for p in reversed(self.p):
            p.stop()


def voll(art: str, **mehr) -> sf.Lage:
    return sf.Lage(art, 1, **dict(dict(begleiter=True, karten=True, rueckmeldung=True, auftraege=True), **mehr))


class Projektart(unittest.TestCase):
    def test_weiche(self):
        faelle = (({"project_type": "pdf", "tool": "formular"}, True, "formular"),
                  ({"project_type": "docx", "tool": "word"}, True, "word"),
                  ({"project_type": "pdf", "tool": "pdf"}, True, "pdf"),
                  ({"project_type": "images", "tool": "grafik"}, True, "grafik"),
                  ({"project_type": "url", "tool": "web"}, True, "web"),
                  ({"project_type": "images", "tool": "grafik"}, False, "alt_text"),
                  ({"project_type": "url", "tool": "web"}, False, "alt_text"),
                  ({"project_type": "pdf", "tool": "alttext"}, True, "alt_text"),
                  ({}, True, "alt_text"))
        for projekt, bild, erwartet in faelle:
            with self.subTest(projekt=projekt, bild=bild), mock.patch.object(funktionen, "AGENT_BILD_WERKZEUGE", bild):
                self.assertEqual(sf.projektart(projekt), erwartet)
        self.assertEqual(sf.projektart(None), "konto")

    def test_werkzeuge_folgen_derselben_weiche(self):
        """Text und Werkzeuge waehlen ueber dieselbe Funktion — ein Projekt bekommt nie den Text einer anderen Art."""
        merkmal = {"pdf": "barrierefrei_machen", "word": "pruefe_word_dokument", "formular": "generate_quickinfo",
                   "grafik": "alt_texte_generieren", "web": "dokument_umbenennen"}
        with alle_an():
            for art, (pt, tool) in {"pdf": ("pdf", "pdf"), "word": ("docx", "word"), "formular": ("pdf", "formular"),
                                    "grafik": ("images", "grafik"), "web": ("url", "web")}.items():
                with self.subTest(art=art):
                    defs, _ex, system = agent_loop._werkzeugsatz({"project_type": pt, "tool": tool}, 1, 1)
                    namen = {d["name"] for d in defs}
                    self.assertIn(merkmal[art], namen, sorted(namen))
                    self.assertEqual(system, sf.zusammensetzen(sf.Lage(art, 1)))

    def test_unbekannte_art(self):
        with self.assertRaises(ValueError):
            sf.zusammensetzen(sf.Lage("tabelle", 1))


class Aufbau(unittest.TestCase):
    def test_feste_reihenfolge(self):
        reihenfolge = [f.name for f in sf.FAECHER]
        self.assertEqual(reihenfolge, ["kosten", "hochladen", "hochladen_hier", "gespraeche", "begleiter", "karten", "kaufen",
                                       "rueckmeldung", "auftraege"])
        with alle_an():
            for art in sf.PROJEKTARTEN:
                with self.subTest(art=art):
                    namen = [n for n, _t in sf.teile(voll(art, demo_leitplanke=None))]
                    self.assertEqual(namen[0], "projektart:" + art, "die Projektart steht immer vorn")
                    faecher = namen[1:]
                    self.assertEqual(faecher, sorted(faecher, key=reihenfolge.index), "Faecher in der festen Reihenfolge")

    def test_nur_in_seinen_projektarten(self):
        with alle_an():
            konto_ = [n for n, _t in sf.teile(voll("konto"))]
            pdf = [n for n, _t in sf.teile(voll("pdf"))]
        self.assertNotIn("kosten", konto_, "ohne Projekt nichts Bezahltes")
        self.assertIn("gespraeche", konto_)
        self.assertNotIn("gespraeche", pdf, "Fruehere Gespraeche nur ohne Projekt (wie bisher)")
        self.assertIn("kosten", pdf)

    def test_ein_teil_je_leerzeile_und_nichts_leeres(self):
        with alle_an():
            lage = voll("word")
            teile = sf.teile(lage)
            self.assertTrue(all(t for _n, t in teile))
            self.assertEqual(sf.zusammensetzen(lage), "\n\n".join(t for _n, t in teile))

    def test_demo_nie_mit_faechern_des_gespraechs(self):
        """Auch wenn ein Aufrufer die Gespraechs-Schalter faelschlich setzt: in der Demo fehlen Begleiter, Karten, Kaufen,
        Rueckmeldung und Auftraege immer (dort liessen sich sonst Einladungen und Kaeufe vorbereiten)."""
        with alle_an():
            teile = sf.teile(voll("pdf", demo_leitplanke="LEITPLANKE (fiktiv)"))
        namen = [n for n, _t in teile]
        for f in sf.FAECHER:
            if not f.demo:
                self.assertNotIn(f.name, namen)
        self.assertEqual(namen[-1], "demo_leitplanke", "die Leitplanke steht ganz am Ende")
        self.assertEqual(teile[-1][1], "LEITPLANKE (fiktiv)")
        self.assertNotIn("demo_leitplanke", [n for n, _t in sf.teile(voll("pdf"))])

    def test_auftraege_entscheiden_den_pdf_schritt(self):
        from inkluagent.prompts import system_pdf
        with alle_an():
            mit = sf.zusammensetzen(voll("pdf"))
            ohne = sf.zusammensetzen(voll("pdf", auftraege=False))
        self.assertIn(system_pdf._SCHRITT2_AUFTRAG, mit)
        self.assertNotIn(system_pdf._SCHRITT2_AUFTRAG, ohne)
        self.assertNotIn(auftraege.PROMPT, ohne)

    def test_konto_werkzeugsatz_nimmt_den_zusammensetzer(self):
        uid = database.create_user(f"schubfach-{time.time_ns()}@beispiel.invalid", "geheim-12345", "Erika (fiktiv)",
                                   email_verified=1)
        with alle_an():
            _defs, _ex, system = konto.werkzeugsatz(uid)
            self.assertEqual(system, sf.zusammensetzen(sf.Lage("konto", uid)))


class Werkzeuge(unittest.TestCase):
    def test_jedes_genannte_werkzeug_gibt_es(self):
        """Die Werkzeuge, die ein Fach erklaert, stehen im Werkzeugsatz seines Moduls (Waechter fuer kuenftige Umbenennungen)."""
        with alle_an():
            vorhanden = ({d["name"] for d in begleiter.werkzeuge(1)} | {d["name"] for d in karten.werkzeuge(None)}
                         | {rueckmeldung.werkzeug_melden(None)["name"]} | set(auftraege.NAMEN)
                         | {konto.TOOL_DEFINITION_GESPRAECHE["name"]})
        for f in sf.FAECHER:
            for w in f.werkzeuge:
                with self.subTest(fach=f.name, werkzeug=w):
                    self.assertIn(w, vorhanden)
                    self.assertIn(w, f.text(sf.Lage("konto" if f.name == "gespraeche" else "pdf", 1)),
                                  "das Fach nennt sein Werkzeug auch im Text")


if __name__ == "__main__":
    unittest.main()
