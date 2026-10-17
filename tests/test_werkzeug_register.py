"""Werkzeug-Register des InkluAgent (17.10.2026; Steve: „richtig anschließen und logisch bauen, damit wir anbauen können, jederzeit,
ohne das ganze Programm umbauen zu müssen“). Was ein Werkzeug ist und kann, steht an mehreren Stellen — jede mit gutem Grund, aber
vergisst man eine, funktioniert ein neues Werkzeug nur halb. Dieser Test haelt sie zusammen (erster Schritt zum Steckbrief je Werkzeug):

- das Werkzeug selbst: die Beschreibungen fuer das Modell (tools/definitions*.py, konto.py, begleiter.py, karten.py, rueckmeldung.py,
  auftraege.py) — hier ALLE, mit allen Schaltern an
- tools/namen.py WERKZEUG_NAMEN: Anzeigename — fuer GENAU diese Werkzeuge (keins ohne Namen, kein Name ohne Werkzeug)
- funktionen.WERKZEUG_SCHALTER: haengt an einem Schalter (der Schalter muss es geben)
- sicherheit.EINZEL_BEZAHLT: Preis der Einzelaktion (die Aktion muss in billing.AKTIONS_PREISE stehen)
- tools/ausgaben.UNUMKEHRBAR (nur per Knopf) und AENDERT_ANSICHT (die Ansicht wird danach nachgezogen)
- inkluagent/arbeitsfenster.ANSICHT_JE_WERKZEUG: in welcher Ansicht man es im Live-Fenster sieht
- rueckmeldung.START_WERKZEUGE: startet einen Lauf im Hintergrund (der Agent meldet sich, wenn er fertig ist)
- auftraege.KATALOG: darf Schritt eines Auftrags sein (mit einer Art im Laufbuch)
- daten.WERKZEUG_FREMDTEXT: welche Felder des Ergebnisses Text aus Dateien tragen (als Daten gekennzeichnet; tests/
  test_daten_keine_anweisung.py prueft, dass jedes Werkzeug eingeordnet ist)
Regeln: jede Liste nennt nur echte Werkzeuge; wer im Hintergrund laeuft oder Schritt eines Auftrags ist, ist im Live-Fenster zu
sehen; wer an einer Stelle arbeitet (Bild, Feld, Dokument), hat eine Ansicht — ausser Loeschen (die Stelle verschwindet).
Ein neues Werkzeug: Beschreibung + Anzeigename; die uebrigen Listen nur, wenn es das braucht — dieser Test sagt, was fehlt.
Wegwerf-Datenbank, keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_werkzeug_register.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="werkzeug_register_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
import laufbuch  # noqa: E402
from inkluagent import arbeitsfenster, auftraege, begleiter, daten, karten, konto, rueckmeldung, sicherheit  # noqa: E402
from inkluagent.tools import ausgaben, namen  # noqa: E402
from inkluagent.tools import definitions, definitions_formular, definitions_oberflaeche, definitions_pdf  # noqa: E402

# Loeschen arbeitet an einer Stelle, die danach nicht mehr da ist: die Art des Ziels (Bild, Dokument) reicht dem Fenster
OHNE_ANSICHT = frozenset({"bild_loeschen", "dokument_loeschen"})


def projekt_werkzeuge() -> dict:
    """Werkzeug -> Beschreibung aus allen Saetzen der Projekte (PDF, Word, Grafik, Web, Formular)."""
    out = {}
    for modul in (definitions, definitions_formular, definitions_oberflaeche, definitions_pdf):
        for name in dir(modul):
            liste = getattr(modul, name)
            if name.startswith("TOOL_DEFINITIONS") and isinstance(liste, list):
                for d in liste:
                    out.setdefault(d["name"], d)
    return out


def alle_werkzeuge() -> dict:
    """Werkzeug -> Beschreibung: Projekte und Gespraech (Konto, Begleiter, Karten, Rueckmeldung, Auftraege), alle Schalter an."""
    out = projekt_werkzeuge()
    uid = database.create_user("register@beispiel.invalid", "geheim-12345", "Test (fiktiv)")
    gespraech = konto.TOOL_DEFINITIONS_KONTO + konto.TOOL_DEFINITIONS_KONTO_AENDERN + [konto.TOOL_DEFINITION_GESPRAECHE]
    with mock.patch.object(funktionen, "EXPRESS", True), mock.patch.object(karten, "kaufen_an", return_value=True):
        gespraech += karten.werkzeuge(1)
    gespraech += begleiter.werkzeuge(uid) + [rueckmeldung.werkzeug_melden(1)] + auftraege.werkzeuge(1)
    for d in gespraech:
        out.setdefault(d["name"], d)
    return out


class Register(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alle = alle_werkzeuge()
        cls.namen = set(cls.alle)

    def test_es_gibt_sie_alle(self):
        self.assertGreaterEqual(len(self.namen), 60, "Sammlung unvollstaendig?")
        for w in ("generate_alt_text", "konvertiere_zu_pdfua", "auftrag_vorbereiten", "hinfuehren", "melden_wenn_fertig",
                  "express_bestellen_vorbereiten", "projekte_auflisten"):
            self.assertIn(w, self.namen)

    def test_anzeigename_fuer_genau_diese_werkzeuge(self):
        self.assertEqual(sorted(self.namen - set(namen.WERKZEUG_NAMEN)), [], "Werkzeug ohne Anzeigenamen (tools/namen.py)")
        self.assertEqual(sorted(set(namen.WERKZEUG_NAMEN) - self.namen), [], "Anzeigename eines Werkzeugs, das es nicht (mehr) gibt")

    def test_jede_liste_nennt_nur_echte_werkzeuge(self):
        listen = {
            "funktionen.WERKZEUG_SCHALTER": funktionen.WERKZEUG_SCHALTER,
            "sicherheit.EINZEL_BEZAHLT": sicherheit.EINZEL_BEZAHLT,
            "tools/ausgaben.UNUMKEHRBAR": ausgaben.UNUMKEHRBAR,
            "tools/ausgaben.AENDERT_ANSICHT": ausgaben.AENDERT_ANSICHT,
            "arbeitsfenster.ANSICHT_JE_WERKZEUG": arbeitsfenster.ANSICHT_JE_WERKZEUG,
            "rueckmeldung.START_WERKZEUGE": rueckmeldung.START_WERKZEUGE,
            "auftraege.KATALOG": [s.werkzeug for s in auftraege.KATALOG],
            "daten.WERKZEUG_FREMDTEXT": daten.WERKZEUG_FREMDTEXT,
            "OHNE_ANSICHT (hier)": OHNE_ANSICHT,
        }
        for titel, liste in listen.items():
            with self.subTest(liste=titel):
                self.assertEqual(sorted(set(liste) - self.namen), [], "unbekanntes Werkzeug (umbenannt oder entfernt?)")

    def test_schalter_und_preise_gibt_es(self):
        for w, schalter in funktionen.WERKZEUG_SCHALTER.items():
            self.assertTrue(hasattr(funktionen, schalter), (w, schalter))
        import billing
        for w, aktion in sicherheit.EINZEL_BEZAHLT.items():
            self.assertIn(aktion, billing.AKTIONS_PREISE, (w, aktion))

    def test_was_im_hintergrund_laeuft_sieht_man_im_live_fenster(self):
        ohne = sorted(set(rueckmeldung.START_WERKZEUGE) - set(arbeitsfenster.ANSICHT_JE_WERKZEUG))
        self.assertEqual(ohne, [], "startet einen Lauf, hat aber keine Ansicht (arbeitsfenster.ANSICHT_JE_WERKZEUG)")

    def test_jeder_auftragsschritt_ist_zu_sehen_und_im_laufbuch(self):
        for s in auftraege.KATALOG:
            with self.subTest(schritt=s.art):
                self.assertIn(s.werkzeug, arbeitsfenster.ANSICHT_JE_WERKZEUG, "keine Ansicht")
                self.assertIn(s.art, auftraege._LAUFBUCH_ART, "keine Art im Laufbuch (auftraege._LAUFBUCH_ART)")
                self.assertIn(auftraege._LAUFBUCH_ART[s.art], laufbuch.ARTEN)
                self.assertIn(auftraege._LAUFBUCH_ART[s.art], arbeitsfenster.ANSICHT_JE_LAUF)

    def test_wer_an_einer_stelle_arbeitet_hat_eine_ansicht(self):
        for w, d in sorted(projekt_werkzeuge().items()):
            felder = set(((d.get("input_schema") or {}).get("properties") or {}))
            if felder & {"image_id", "feld_id", "document_id"} and w not in OHNE_ANSICHT:
                with self.subTest(werkzeug=w):
                    self.assertIn(w, arbeitsfenster.ANSICHT_JE_WERKZEUG, "arbeitet an einer Stelle, hat aber keine Ansicht")
        self.assertLessEqual(OHNE_ANSICHT, set(ausgaben.UNUMKEHRBAR), "Ausnahmen nur fuers Loeschen")


if __name__ == "__main__":
    unittest.main()
