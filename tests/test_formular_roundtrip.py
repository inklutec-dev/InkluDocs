"""Unit-Tests Quickinfo-Werkzeug: Lesen und Zurueckschreiben von Formularfeldern.

Aufruf im Container:
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_formular_roundtrip.py -v
Lokal (mit PyMuPDF, ohne PDFix -> Rueckfallpfade):
    cd backend && python3 -m unittest ../tests/test_formular_roundtrip.py -v

Fixture: tests/fixtures/testformular_inkludocs.pdf (FIKTIV, erzeugt von
make_testformular.py; 12 Felder: Text, Pflicht, vorhandene Quickinfo,
ausgefuellt, Checkbox, Radio-Gruppe, Dropdown, Unterschrift, zwei Seiten).
Alle Namen und Daten sind erfunden.

Abgedeckt:
  - Vorpruefung (kein Formular, Grenzen)
  - Feldliste: Anzahl, Feldarten, Seiten, Beschriftungen (links/oben/rechts),
    Abschnitt, Pflicht, Optionen, vorhandene Quickinfo, ausgefuellt
  - DATENSCHUTZ: eingetragener Wert taucht nirgends auf
  - Zurueckschreiben: Quickinfos gesetzt (auch Radio-Gruppe = Elternfeld),
    leere bleiben unberuehrt, unbekannte Namen gemeldet, Original = Praefix
    der Ausgabe (inkrementell), sichtbarer Text unveraendert, Idempotenz
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import fitz  # noqa: E402
import formular_processor as fp  # noqa: E402
import formular_export as fe  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "testformular_inkludocs.pdf")
WERT = "K-0000-TEST"   # Wert im Feld "kundennummer" — darf nirgends auftauchen


class TestVorpruefung(unittest.TestCase):
    def test_fixture_ist_formular(self):
        self.assertEqual(fp.validiere_formular(FIXTURE), 13)   # 13 Erscheinungen (Radio = 2)

    def test_kein_formular(self):
        with tempfile.TemporaryDirectory() as d:
            pfad = os.path.join(d, "leer.pdf")
            doc = fitz.open()
            doc.new_page().insert_text((50, 50), "Nur Text, keine Felder")
            doc.save(pfad)
            doc.close()
            with self.assertRaises(fp.FormularFehler):
                fp.validiere_formular(pfad)

    def test_keine_pdf(self):
        with self.assertRaises(fp.FormularFehler):
            fp.validiere_formular(__file__)


class TestLesen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.felder, cls.hinweise = fp.extract_formular(FIXTURE, cls.tmp.name, 0)
        cls.by = {f["anker"]: f for f in cls.felder}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_anzahl_und_arten(self):
        self.assertEqual(len(self.felder), 12)
        self.assertEqual(self.by["vorname"]["feld_art"], "text")
        self.assertEqual(self.by["newsletter"]["feld_art"], "checkbox")
        self.assertEqual(self.by["zahlungsweise"]["feld_art"], "radio")
        self.assertEqual(self.by["anrede"]["feld_art"], "dropdown")
        self.assertEqual(self.by["unterschrift"]["feld_art"], "signatur")

    def test_seiten(self):
        self.assertEqual(self.by["vorname"]["page_number"], 1)
        self.assertEqual(self.by["vorname_2"]["page_number"], 2)
        self.assertEqual(self.by["unterschrift"]["seiten"], [2])

    def test_beschriftungen(self):
        self.assertEqual((self.by["vorname"]["beschriftung"], self.by["vorname"]["beschriftung_lage"]), ("Vorname", "links"))
        self.assertEqual((self.by["anschrift"]["beschriftung"], self.by["anschrift"]["beschriftung_lage"]), ("Straße und Hausnummer", "oben"))
        self.assertEqual(self.by["newsletter"]["beschriftung_lage"], "rechts")
        self.assertIn("Newsletter", self.by["newsletter"]["beschriftung"])
        self.assertEqual(self.by["zahlungsweise"]["beschriftung"], "monatlich")

    def test_abschnitt(self):
        self.assertEqual(self.by["vorname"]["gruppe"], "Angaben zum Kontoinhaber")
        self.assertEqual(self.by["vorname_2"]["gruppe"], "Zweiter Kontoinhaber")

    def test_pflicht_optionen_quickinfo_ausgefuellt(self):
        self.assertTrue(self.by["geburtsdatum"]["pflicht"])
        self.assertFalse(self.by["vorname"]["pflicht"])
        self.assertEqual(self.by["anrede"]["optionen"], ["Frau", "Herr", "keine Angabe"])
        self.assertEqual(sorted(self.by["zahlungsweise"]["optionen"]), ["jaehrlich", "monatlich"])
        self.assertEqual(self.by["email"]["quickinfo_original"], "E-Mail-Adresse für Kontoauszüge")
        self.assertTrue(self.by["kundennummer"]["ausgefuellt"])
        self.assertFalse(self.by["vorname"]["ausgefuellt"])

    def test_datenschutz_wert_nirgends(self):
        for f in self.felder:
            for k, v in f.items():
                if isinstance(v, str):
                    self.assertNotIn(WERT, v, f"Feldwert in {f['anker']}.{k}")

    def test_bilder(self):
        for f in self.felder:
            self.assertTrue(os.path.isfile(f["ausschnitt_path"]), f["anker"])
            self.assertTrue(os.path.isfile(f["page_view_path"]), f["anker"])
        self.assertTrue(self.by["vorname"]["page_text"].startswith("Musterbank"))

    def test_hinweise_leer(self):
        self.assertEqual(self.hinweise["uebersprungen"], [])


class TestSchreiben(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "export.pdf")

    def tearDown(self):
        self.tmp.cleanup()

    def _tus(self, pfad):
        felder, _ = fp.extract_formular(pfad, None, 0)
        return {f["anker"]: f["quickinfo_original"] for f in felder}

    def test_roundtrip(self):
        erg = fe.write_quickinfos_to_pdf(FIXTURE, self.out, {
            "vorname": "Vorname des Kontoinhabers",
            "zahlungsweise": "Zahlungsweise: monatlich oder jährlich",
            "anrede": "Anrede auswählen",
            "email": "",              # leer -> Original bleibt
            "gibtsnicht": "x",        # unbekannt -> gemeldet
            "#99": "y",               # namenlos -> Warnung
        })
        self.assertEqual(erg.geschrieben, 3)
        self.assertEqual(erg.nicht_gefunden, ["gibtsnicht"])
        self.assertTrue(any("keinen Feldnamen" in w for w in erg.warnungen))
        tus = self._tus(self.out)
        self.assertEqual(tus["vorname"], "Vorname des Kontoinhabers")
        self.assertEqual(tus["zahlungsweise"], "Zahlungsweise: monatlich oder jährlich")
        self.assertEqual(tus["anrede"], "Anrede auswählen")
        self.assertEqual(tus["email"], "E-Mail-Adresse für Kontoauszüge")
        self.assertEqual(tus["nachname"], "")

    def test_original_ist_praefix(self):
        fe.write_quickinfos_to_pdf(FIXTURE, self.out, {"vorname": "Vorname"})
        with open(FIXTURE, "rb") as a, open(self.out, "rb") as b:
            orig = a.read()
            self.assertEqual(b.read(len(orig)), orig)

    def test_text_und_felder_unveraendert(self):
        fe.write_quickinfos_to_pdf(FIXTURE, self.out, {"vorname": "Vorname", "nachname": "Nachname"})
        a, b = fitz.open(FIXTURE), fitz.open(self.out)
        self.assertEqual(a.page_count, b.page_count)
        self.assertEqual(fe._widget_liste(a), fe._widget_liste(b))
        self.assertEqual(fe._seitentexte_ohne_widgets(a), fe._seitentexte_ohne_widgets(b))

    def test_nichts_zu_schreiben_kopiert(self):
        erg = fe.write_quickinfos_to_pdf(FIXTURE, self.out, {"vorname": "", "nachname": None})
        self.assertEqual(erg.geschrieben, 0)
        with open(FIXTURE, "rb") as a, open(self.out, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_idempotent(self):
        fe.write_quickinfos_to_pdf(FIXTURE, self.out, {"vorname": "Vorname"})
        out2 = os.path.join(self.tmp.name, "export2.pdf")
        fe.write_quickinfos_to_pdf(self.out, out2, {"vorname": "Vorname", "nachname": "Nachname"})
        tus = self._tus(out2)
        self.assertEqual((tus["vorname"], tus["nachname"]), ("Vorname", "Nachname"))

    def test_steuerzeichen_und_laenge(self):
        fe.write_quickinfos_to_pdf(FIXTURE, self.out, {"vorname": "Zeile 1\nZeile 2\t" + "x" * 2000})
        tus = self._tus(self.out)
        self.assertNotIn("\n", tus["vorname"])
        self.assertLessEqual(len(tus["vorname"]), 1000)


if __name__ == "__main__":
    unittest.main()
