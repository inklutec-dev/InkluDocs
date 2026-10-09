"""Sprache beim PDF-Tagging (Skriptpruefung 09.10.2026, Michael Karbe: „wir hinterlegen per Default Englisch, es sollte Deutsch
bzw. die ausgewaehlte Sprache sein“). PDFix' Voreinstellung traegt fest en-US ein; InkluDocs ersetzt sie (erkannt -> /Lang des
Dokuments -> Projektsprache). Geprueft hier: ungueltiges /Lang bricht nicht mehr ab, und der Chatbot gibt die Projektsprache weiter.
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_tagging_sprache.py -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import pdf_tagging  # noqa: E402

KURZ = "Sommerfest im Musterverein. Samstag ab 14 Uhr. Anmeldung beim Vorstand."      # zu kurz fuer eine sichere Erkennung
LANG_DE = ("Der fiktive Verein fasst die Arbeit des Jahres zusammen und richtet sich an die Mitglieder. Die Zahl der "
           "Mitglieder ist gestiegen, und die Kasse wurde von zwei Mitgliedern geprueft, die keine Fehler gefunden haben. ") * 5


def _pdf(pfad: str, text: str, lang: str = "") -> None:
    import fitz
    d = fitz.open()
    p = d.new_page(width=595, height=842)
    y = 60
    for zeile in [text[i:i + 90] for i in range(0, len(text), 90)][:40]:
        p.insert_text((50, y), zeile, fontsize=9)
        y += 14
    d.save(pfad)
    d.close()
    if lang:
        import pikepdf
        with pikepdf.open(pfad, allow_overwriting_input=True) as pdf:
            pdf.Root.Lang = pikepdf.String(lang)
            pdf.save(pfad)


class UngueltigeDokumentsprache(unittest.TestCase):
    def test_kurzer_text_mit_unterstrich_nimmt_projektsprache_und_ersetzt(self):
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "a.pdf")
            _pdf(pfad, KURZ, lang="de_DE")
            s = pdf_tagging.sprache_bestimmen(pfad, "fr")
            self.assertEqual(s["lang"], "fr-FR")
            self.assertEqual(s["quelle"], "vorgabe")
            self.assertTrue(s["overwrite"], "sonst laesst PDFix den ungueltigen Wert stehen")
            self.assertIn("de_DE", s["hinweis"])
            # die Konfiguration laesst sich damit bauen (vorher: TaggingFehler „Ungueltige Sprachangabe“)
            k = pdf_tagging.konfig_erzeugen(s["lang"], s["overwrite"], os.path.join(t, "k.json"))
            self.assertEqual(k["sprache"], "fr-FR")

    def test_ungueltige_angaben(self):
        for wert in ("de_DE", "Deutsch", "x-default"):
            with tempfile.TemporaryDirectory() as t:
                pfad = os.path.join(t, "a.pdf")
                _pdf(pfad, KURZ, lang=wert)
                s = pdf_tagging.sprache_bestimmen(pfad, "de")
                self.assertEqual((s["lang"], s["overwrite"]), ("de-DE", True), wert)

    def test_gueltige_angabe_bleibt_bei_kurzem_text(self):
        """Unveraendert: gueltiges /Lang bleibt, wenn der Text nicht sicher erkennbar ist (auch de-CH)."""
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "a.pdf")
            _pdf(pfad, KURZ, lang="de-CH")
            s = pdf_tagging.sprache_bestimmen(pfad, "fr")
            self.assertEqual((s["lang"], s["quelle"], s["overwrite"]), ("de-CH", "dokument", False))

    def test_langer_text_mit_unterstrich(self):
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "a.pdf")
            _pdf(pfad, LANG_DE, lang="de_DE")
            s = pdf_tagging.sprache_bestimmen(pfad, "en")
            self.assertEqual((s["lang"], s["quelle"], s["overwrite"]), ("de-DE", "erkannt", True))

    def test_lauf_mit_ungueltiger_sprache_testmodus(self):
        try:
            import pdfixsdk  # noqa: F401
        except Exception:
            self.skipTest("kein PDFix-SDK")
        with tempfile.TemporaryDirectory() as t:
            quelle, ziel = os.path.join(t, "roh.pdf"), os.path.join(t, "getaggt.pdf")
            _pdf(quelle, KURZ, lang="de_DE")
            with mock.patch.dict(os.environ, {"PDFIX_TAGGING_LIZENZ": "off"}):
                bericht = pdf_tagging.taggen(quelle, ziel, "de")
            self.assertEqual(bericht["nachher"]["lang"], "de-DE")


class ChatbotGibtProjektspracheWeiter(unittest.TestCase):
    """inkluagent/tools/pdf.barrierefrei_machen startet tagging_api.lauf_synchron — mit der Projektsprache wie Knopf und Kette."""

    def _lauf(self, projekt_sprache):
        from inkluagent.tools import pdf as bot

        class Conn:
            def close(self):
                pass
        t = mock.Mock()
        t.stand.return_value = {"verfuegbar": True, "laeuft": False, "seiten": 1, "preis": 20, "erlaubt": True,
                                "verfuegbar_credits": None, "fehlend": 0, "quelle_getaggt": False}
        t.lesbar_grund.return_value = ""
        gestartet = {}

        class Faden:
            def __init__(self, target=None, args=(), **kw):
                gestartet["args"] = args

            def start(self):
                pass
        with mock.patch.object(bot, "_tagging", return_value=t), mock.patch.object(bot, "_get_db", return_value=Conn()), \
                mock.patch.object(bot, "_projekt", return_value={"id": 7, "alt_language": projekt_sprache, "project_type": "pdf"}), \
                mock.patch.object(bot, "_dokument", return_value={"id": 9, "getaggt": 0}), \
                mock.patch.object(bot, "_name", return_value="Testdokument"), \
                mock.patch.object(bot, "_freigabe", return_value=None), \
                mock.patch.object(bot._ausg, "_ui_lang", return_value="en"), \
                mock.patch.object(bot.threading, "Thread", Faden), mock.patch.object(bot.time, "sleep"):
            r = bot.barrierefrei_machen(7, 1, 9, bestaetigt=True)
        self.assertTrue(r.get("ok"), r)
        return gestartet["args"]

    def test_projektsprache(self):
        self.assertEqual(self._lauf("fr"), (7, 9, 1, "fr", "en"))

    def test_ohne_projektsprache_kontosprache(self):
        self.assertEqual(self._lauf(None)[3], "en")


if __name__ == "__main__":
    unittest.main()
