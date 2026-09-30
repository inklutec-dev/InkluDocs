"""Chatbot = Oberflaeche (Steve 30.09.2026): der InkluAgent kann alles, was man in der Oberflaeche von Hand macht, und nichts,
was die Oberflaeche nicht anbietet. EIN Schalter je Funktion (backend/funktionen.py) steuert Oberflaeche, Chatbot-Werkzeuge,
Systemprompt und Endpunkte. Laeuft ohne Server: `python3 -m unittest tests/test_chatbot_oberflaeche.py`."""
import os
import sys
import unittest
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import funktionen  # noqa: E402
from inkluagent import agent_loop  # noqa: E402

AUSGEBLENDET = ("pruefung_starten", "pruefbericht_lesen", "korrektur_anwenden", "korrektur_rueckgaengig",
                "komplett_barrierefrei_machen", "revert_alt_text")
PDF = {"project_type": "pdf", "tool": "pdf"}
WORD = {"project_type": "docx", "tool": "word"}
FORMULAR = {"project_type": "pdfform", "tool": "formular"}


def satz(projekt):
    defs, executor, system = agent_loop._werkzeugsatz(projekt, 1, 1)
    return {d["name"] for d in defs}, executor, system, defs


class Schalter(unittest.TestCase):
    def test_voreinstellung_wie_die_oberflaeche(self):
        """Heute ausgeblendet: KI-Pruefung, Korrektur, Kette, „Text zurückholen“, Urteil, Strukturansicht, eigene Pruefungen."""
        self.assertEqual(funktionen.fuer_oberflaeche(), {"ki_pruefung": False, "korrektur": False, "eigene_pruefungen": False,
                                                         "urteil": False, "kette": False, "text_zurueck": False,
                                                         "strukturansicht": False})
        import abschluss
        self.assertIs(abschluss.EIGENE_PRUEFUNGEN, funktionen.EIGENE_PRUEFUNGEN)

    def test_ausgeblendete_werkzeuge_fehlen_ueberall(self):
        for projekt in (PDF, WORD, FORMULAR):
            namen, executor, system, _defs = satz(projekt)
            for w in AUSGEBLENDET:
                with self.subTest(projekt=projekt["tool"], werkzeug=w):
                    self.assertNotIn(w, namen)
                    self.assertNotIn(w, system, "der Systemprompt nennt ein ausgeblendetes Werkzeug")
                    r = executor.execute(w, {"document_id": 1, "image_id": 1})
                    self.assertFalse(r.get("ok"))
                    self.assertIn("Unbekannt", r.get("error", ""))
        _n, _e, system, _d = satz(PDF)
        for wort in ("KI-basierte Prüfung", "Doppelbeleg", "Strukturansicht öffnen"):
            self.assertNotIn(wort, system)

    def test_ein_schalter_steuert_alles(self):
        """Schalter an = Werkzeug, Prompt-Absatz und Handler da; derselbe Ort wie die Oberflaeche."""
        with mock.patch.object(funktionen, "KORREKTUR", True), mock.patch.object(funktionen, "KI_PRUEFUNG", True), \
                mock.patch.object(funktionen, "KETTE", True), mock.patch.object(funktionen, "TEXT_ZURUECK", True):
            namen, executor, system, _d = satz(PDF)
            for w in AUSGEBLENDET:
                self.assertIn(w, namen)
                self.assertIn(w, system)
                self.assertIn(w, executor._handlers())
            self.assertTrue(funktionen.fuer_oberflaeche()["korrektur"])

    def test_endpunkte_hinter_dem_schalter(self):
        from fastapi import HTTPException
        for schalter in ("KI_PRUEFUNG", "KORREKTUR", "KETTE", "TEXT_ZURUECK"):
            with self.assertRaises(HTTPException) as cm:
                funktionen.endpunkt_frei(schalter)
            self.assertEqual(cm.exception.status_code, 404)
        # die Endpunkte rufen den Schalter wirklich auf (im Quelltext belegt)
        import inspect
        import kette_api
        import tagging_api
        quelle = inspect.getsource(tagging_api) + inspect.getsource(kette_api)
        self.assertEqual(quelle.count('funktionen.endpunkt_frei("KI_PRUEFUNG")'), 2)   # POST pruefung, GET befunde.csv
        self.assertEqual(quelle.count('funktionen.endpunkt_frei("KORREKTUR")'), 2)     # korrektur, rueckgaengig
        self.assertEqual(quelle.count('funktionen.endpunkt_frei("KETTE")'), 2)         # GET/POST kette
        with open(os.path.join(os.path.dirname(inspect.getfile(funktionen)), "main.py"), encoding="utf-8") as f:
            main_quelle = f.read()
        self.assertIn('funktionen.endpunkt_frei("TEXT_ZURUECK")', main_quelle)


class Werkzeuge(unittest.TestCase):
    """Die Funktionen der Oberflaeche sind als Werkzeuge da (Bestandsaufnahme 30.09.2026)."""

    NEU_PDF = ("testweise_taggen", "pruefdatei_erstellen", "pruefdatei_lesen", "exportiere_alt_texte", "exportiere_quickinfos",
               "alt_texte_generieren", "quickinfos_generieren", "stammdaten_anwenden", "ki_kontext_setzen", "eigener_prompt",
               "ausgabe_loeschen")
    GRUND_PDF = ("dokument_stand", "barrierefrei_machen", "hoerprobe_lesen", "exportiere_fertige_pdf", "dokument_umbenennen",
                 "dokument_loeschen", "alt_sprache_setzen", "liste_ausgaben", "lies_ausgabe", "generate_alt_text",
                 "update_alt_text", "generate_quickinfo", "update_quickinfo", "revert_quickinfo")
    NEU_WORD = ("dokument_umbenennen", "dokument_loeschen", "alt_sprache_setzen", "exportiere_alt_texte", "alt_texte_generieren",
                "ki_kontext_setzen", "eigener_prompt", "ausgabe_loeschen")
    GRUND_WORD = ("pruefe_word_dokument", "konvertiere_zu_pdfua", "exportiere_word", "uebersetze_dokument", "exportiere_uebersetzung",
                  "liste_ausgaben", "lies_ausgabe")
    NEU_FORMULAR = ("exportiere_quickinfos", "quickinfos_generieren", "stammdaten_anwenden", "eigener_prompt")

    def pruefe(self, projekt, erwartet):
        namen, executor, _system, defs = satz(projekt)
        handler = executor._handlers()
        for w in erwartet:
            with self.subTest(werkzeug=w):
                self.assertIn(w, namen)
                self.assertIn(w, handler, "Werkzeug beschrieben, aber ohne Ausfuehrung")
        self.assertEqual(len(namen), len(defs), "Werkzeug doppelt im Satz")
        for d in defs:   # jede Beschreibung ist fuer das Modell brauchbar
            self.assertTrue(d.get("description") and d.get("input_schema", {}).get("type") == "object", d["name"])
            self.assertIn(d["name"], handler, f"{d['name']}: keine Ausfuehrung")

    def test_pdf(self):
        self.pruefe(PDF, self.NEU_PDF + self.GRUND_PDF)

    def test_word(self):
        self.pruefe(WORD, self.NEU_WORD + self.GRUND_WORD)

    def test_formular(self):
        self.pruefe(FORMULAR, self.NEU_FORMULAR)

    def test_kostenpflichtige_verlangen_bestaetigung(self):
        """Kostenpflichtige und unumkehrbare neue Werkzeuge haben den Schalter bestaetigt (Angebot -> Ja -> Ausfuehrung)."""
        _n, _e, _s, defs = satz(PDF)
        je = {d["name"]: d for d in defs}
        for w in ("exportiere_alt_texte", "exportiere_quickinfos", "alt_texte_generieren", "quickinfos_generieren",
                  "ausgabe_loeschen", "exportiere_fertige_pdf"):
            self.assertIn("bestaetigt", je[w]["input_schema"]["properties"], w)
        for w in ("testweise_taggen", "pruefdatei_erstellen", "pruefdatei_lesen", "stammdaten_anwenden", "ki_kontext_setzen"):
            self.assertNotIn("bestaetigt", je[w]["input_schema"]["properties"], w)


class Freigabe(unittest.TestCase):
    """Die Rueckfrage der neuen Werkzeuge ist dieselbe wie bei den bestehenden (pdf._freigabe): ohne Angebot keine
    Ausfuehrung, Zustimmung nie in derselben Nachricht."""

    def test_ohne_angebot_keine_ausfuehrung(self):
        from inkluagent.tools import ausgaben, oberflaeche
        ausgaben._ANGEBOTE.clear()

        class Turn:
            turn_id = "t1"
            kostenpflichtig = 0
        with mock.patch.object(oberflaeche, "_doc_id", return_value=None), \
                mock.patch.object(oberflaeche, "_main") as m:
            m.return_value.billing.aktion_pruefung.return_value = {"preis": 10, "erlaubt": True, "verfuegbar": None, "fehlend": 0}
            m.return_value.billing.TABELLEN_EXPORTE = {"csv": "csv_export"}
            r = oberflaeche.exportiere_alt_texte(1, 1, "csv", None, bestaetigt=True, turn=Turn())
            self.assertTrue(r["result"]["rueckfrage_noetig"])
            m.return_value._tabellen_export_bauen.assert_not_called()
            r = oberflaeche.exportiere_alt_texte(1, 1, "csv", None, bestaetigt=False, turn=Turn())       # Angebot
            self.assertTrue(r["result"]["rueckfrage_noetig"])
            r = oberflaeche.exportiere_alt_texte(1, 1, "csv", None, bestaetigt=True, turn=Turn())        # selbe Nachricht
            self.assertIn("eigenen", r["result"]["hinweis"])
            m.return_value._tabellen_export_bauen.assert_not_called()
            t2 = Turn()
            t2.turn_id = "t2"
            m.return_value._tabellen_export_bauen.return_value = {"daten": b"x", "dateiname": "a.csv", "media": "text/csv",
                                                                  "preis": 10, "aktion": "csv_export"}
            m.return_value.sofort_download_ablegen.return_value = "/api/projects/1/export/pdfua/" + "a" * 24
            oberflaeche.exportiere_alt_texte(1, 1, "csv", None, bestaetigt=False, turn=Turn())
            r = oberflaeche.exportiere_alt_texte(1, 1, "csv", None, bestaetigt=True, turn=t2)            # neue Nachricht
            self.assertTrue(r["ok"])
            self.assertEqual(r["anhang"]["label"], "csv")
            m.return_value.billing.verbuche.assert_called_once()


if __name__ == "__main__":
    unittest.main()
