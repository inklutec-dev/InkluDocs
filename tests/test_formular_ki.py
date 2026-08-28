"""Unit-Tests Feld-Pass-Nachpruefung (Quickinfo-Werkzeug Stufe 2, 27.08.2026).

Deterministisch, OHNE Modellaufruf: prueft nachpruefung(), konsistenz(),
seiten_zeilen() (widgetfreie Kopie, keine Feldwerte) und den Prompt-Builder
(Datenblock, Regeln, Sprache, Variation).
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_formular_ki.py -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import formular_ki as fk  # noqa: E402
import formular_processor as fp  # noqa: E402
from prompts.builders.quickinfo import build_quickinfo_prompt  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "testformular_inkludocs.pdf")


def setUpModule():
    if not os.path.isfile(FIXTURE):
        sys.path.insert(0, os.path.join(HERE, "fixtures"))
        import make_testformular  # noqa: E402
        make_testformular.erzeuge(FIXTURE)


class TestSeitenZeilen(unittest.TestCase):
    def test_zeilen_ohne_feldwerte(self):
        zeilen, text = fk.seiten_zeilen(FIXTURE, 1)
        self.assertTrue(any("Vorname" in z["text"] for z in zeilen))
        self.assertNotIn("K-0000-TEST", text)          # Feldwert nie im Kontext
        self.assertTrue(all(len(z["rect"]) == 4 for z in zeilen))

    def test_seite_nicht_vorhanden(self):
        with self.assertRaises(fk.FeldPassFehler):
            fk.seiten_zeilen(FIXTURE, 99)


class TestNachpruefung(unittest.TestCase):
    def setUp(self):
        self.zeilen, self.text = fk.seiten_zeilen(FIXTURE, 1)
        felder, _ = fp.extract_formular(FIXTURE, None, 0)
        self.by = {f["anker"]: dict(f, id=i + 1) for i, f in enumerate(felder)}

    def _v(self, anker, quickinfo, beleg, sicherheit="hoch"):
        f = self.by[anker]
        return fk.nachpruefung(fk.FeldVorschlag(feld_id=f["id"], quickinfo=quickinfo, beleg=beleg, sicherheit=sicherheit),
                               f, self.zeilen, self.text), f

    def test_beleg_neben_feld_bleibt_hoch(self):
        v, _ = self._v("vorname", "Vorname des Kontoinhabers", "Vorname")
        self.assertEqual(v.sicherheit, "hoch")
        self.assertEqual(v.hinweise, [])

    def test_beleg_nicht_im_text_wird_niedrig(self):
        v, _ = self._v("vorname", "Vorname des Kontoinhabers", "Steuernummer")
        self.assertEqual(v.sicherheit, "niedrig")
        self.assertTrue(any("nicht im Seitentext" in h for h in v.hinweise))

    def test_kein_beleg_wird_niedrig(self):
        v, _ = self._v("vorname", "Irgendwas", "")
        self.assertEqual(v.sicherheit, "niedrig")

    def test_beleg_weit_weg_wird_mittel(self):
        # Beleg "Anrede" steht weit unter dem Feld Vorname (nicht in Feldnaehe)
        v, _ = self._v("vorname", "Anrede auswählen", "Anrede")
        self.assertEqual(v.sicherheit, "mittel")
        self.assertTrue(any("Nähe" in h for h in v.hinweise))

    def test_regeln_floskel_feldart_format_pflicht(self):
        v, _ = self._v("vorname", "Bitte hier eingeben: Vorname, Textfeld", "Vorname")
        self.assertEqual(v.sicherheit, "mittel")
        self.assertTrue(any("Anleitungsfloskel" in h for h in v.hinweise))
        self.assertTrue(any("Feldart" in h for h in v.hinweise))
        v, _ = self._v("vorname", "Vorname, Pflichtfeld", "Vorname")
        self.assertTrue(any("Pflichtfeld" in h for h in v.hinweise))
        # Geburtsdatum ist Pflicht (Flag) und Format steht auf der Seite -> keine Hinweise
        v, _ = self._v("geburtsdatum", "Geburtsdatum, Format Tag Punkt Monat Punkt Jahr, Pflichtfeld", "Geburtsdatum (TT.MM.JJJJ) *")
        self.assertEqual(v.sicherheit, "hoch")
        self.assertEqual(v.hinweise, [])

    def test_laenge_wird_gekuerzt(self):
        v, _ = self._v("vorname", "x" * 500, "Vorname")
        self.assertLessEqual(len(v.quickinfo), fk.MAX_QUICKINFO_LAENGE)

    def test_konsistenz(self):
        a, fa = self._v("vorname", "Vorname des Kontoinhabers", "Vorname")
        b, fb = self._v("vorname_2", "Vorname der Person", "Vorname")
        fb = dict(fb, gruppe=fa["gruppe"])   # gleiche Gruppe erzwingen
        out = fk.konsistenz([a, b], {fa["id"]: fa, fb["id"]: fb})
        self.assertEqual(out[1].quickinfo, "Vorname des Kontoinhabers")
        self.assertTrue(any("angeglichen" in h for h in out[1].hinweise))


class TestBuilder(unittest.TestCase):
    def test_prompt_enthaelt_daten_regeln_sprache(self):
        zeilen, _ = fk.seiten_zeilen(FIXTURE, 1)
        felder, _ = fp.extract_formular(FIXTURE, None, 0)
        felder = [dict(f, id=i + 1) for i, f in enumerate(felder) if f["page_number"] == 1]
        system, prompt = build_quickinfo_prompt(zeilen, felder, formular_titel="Test", seite=1, seiten_gesamt=2, sprache="en",
                                                bestaetigte=[("Nachname", "Nachname des Kontoinhabers")], user_prompt="Kurz halten.", variation=True)
        self.assertIn("DATEN", system)
        self.assertIn("=== SEITENTEXT", prompt)
        self.assertIn("Vorname", prompt)
        self.assertIn("F1:", prompt)
        self.assertIn("Englisch", prompt)
        self.assertIn("NEU GENERIEREN", prompt)
        self.assertIn("Kurz halten.", prompt)
        self.assertIn("Nachname des Kontoinhabers", prompt)
        self.assertIn("feld_index", prompt)
        self.assertNotIn("K-0000-TEST", prompt)

class TestKonsistenzNummern(unittest.TestCase):
    """Befund Bankformular 28.08.2026: drei gleiche Bloecke [1]/[2]/[3] — die Angleichung
    darf Texte mit anderer Nummer/Ordnung NICHT ueberschreiben."""

    def _felder(self):
        return {1: {"beschriftung": "Nationality", "feld_art": "text", "gruppe": "activity"},
                2: {"beschriftung": "Nationality", "feld_art": "text", "gruppe": "activity"},
                3: {"beschriftung": "Nationality", "feld_art": "text", "gruppe": "activity"}}

    def test_andere_nummer_bleibt(self):
        vs = [fk.FeldVorschlag(1, "Staatsangehörigkeit des Berechtigten [1]."),
              fk.FeldVorschlag(2, "Staatsangehörigkeit des Berechtigten [2]."),
              fk.FeldVorschlag(3, "Staatsangehörigkeit des Berechtigten [3].")]
        out = fk.konsistenz(vs, self._felder())
        self.assertEqual([v.quickinfo for v in out], ["Staatsangehörigkeit des Berechtigten [1].",
                                                       "Staatsangehörigkeit des Berechtigten [2].",
                                                       "Staatsangehörigkeit des Berechtigten [3]."])
        self.assertTrue(all(not v.hinweise for v in out))

    def test_ordnungswort_bleibt(self):
        vs = [fk.FeldVorschlag(1, "Unterschrift des ersten Vertreters."),
              fk.FeldVorschlag(2, "Unterschrift des zweiten Vertreters.")]
        out = fk.konsistenz(vs, {1: {"beschriftung": "Vertreter", "feld_art": "text", "gruppe": ""},
                                 2: {"beschriftung": "Vertreter", "feld_art": "text", "gruppe": ""}})
        self.assertEqual(out[1].quickinfo, "Unterschrift des zweiten Vertreters.")

    def test_gleiche_nummer_wird_angeglichen(self):
        vs = [fk.FeldVorschlag(1, "Wohnadresse des Berechtigten [1]."),
              fk.FeldVorschlag(2, "Adresse des Berechtigten [1] (Wohnort).")]
        out = fk.konsistenz(vs, {1: {"beschriftung": "Home address", "feld_art": "text", "gruppe": "x"},
                                 2: {"beschriftung": "Home address", "feld_art": "text", "gruppe": "x"}})
        self.assertEqual(out[1].quickinfo, "Wohnadresse des Berechtigten [1].")
        self.assertIn("Wortlaut an gleiche Beschriftung angeglichen.", out[1].hinweise)

class TestSeitenbildAusnahme(unittest.TestCase):
    """Seitenbild-Ausnahme 28.08.2026: Prompt-Block nur mit Flag; Nachpruefung nennt die Quelle der Zuordnung."""

    def test_prompt_block_nur_mit_flag(self):
        zeilen, _ = fk.seiten_zeilen(FIXTURE, 1)
        felder, _ = fp.extract_formular(FIXTURE, None, 0)
        felder = [dict(f, id=i + 1) for i, f in enumerate(felder) if f["page_number"] == 1]
        _, ohne = build_quickinfo_prompt(zeilen, felder, seite=1)
        _, mit = build_quickinfo_prompt(zeilen, felder, seite=1, mit_seitenbild=True)
        self.assertNotIn("SEITENBILD", ohne)
        self.assertIn("SEITENBILD", mit)
        self.assertIn("WÖRTLICHE Textstelle", mit)

    def test_nachpruefung_hinweis_mit_seitenbild(self):
        zeilen, text = fk.seiten_zeilen(FIXTURE, 1)
        felder, _ = fp.extract_formular(FIXTURE, None, 0)
        f = next(x for x in felder if x["beschriftung"] == "Vorname")
        fern = [z for z in zeilen if not fk._in_feldnaehe(z, f["rect"], "text")]
        if not fern:
            self.skipTest("Fixture hat keine Zeile ausserhalb der Feldnaehe")
        weit = max(fern, key=lambda z: abs(z["rect"][1] - f["rect"][1]))
        v = fk.nachpruefung(fk.FeldVorschlag(1, "Test.", beleg=weit["text"], sicherheit="hoch"), f, zeilen, text, mit_seitenbild=True)
        self.assertEqual(v.sicherheit, "mittel")
        self.assertTrue(any("Seitenbild" in h for h in v.hinweise), v.hinweise)


if __name__ == "__main__":
    unittest.main()
