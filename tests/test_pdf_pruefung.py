"""Unit-Tests der automatischen Pruefung (pdf_pruefung.py, 22.09.2026) — Modell als Attrappe, kein Netz.
Aufruf im Container: python3 -m unittest /app/tests/test_pdf_pruefung.py"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, "/app")
import pdf_pruefung  # noqa: E402
from prompts.components.schemas.pdf_pruefung import PruefBefund, PruefSeiteOutput  # noqa: E402
from prompts.builders.pdf_pruefung import build_pruefung_prompt  # noqa: E402


def _struktur():
    return {
        "info": {"seiten": 2, "elemente": 6, "lang": "de-DE"},
        "elemente": [
            {"id": "0", "typ": "Document", "tiefe": 0, "seite": 1, "text": ""},
            {"id": "0.0", "typ": "H1", "tiefe": 1, "seite": 1, "text": "Jahresbericht"},
            {"id": "0.1", "typ": "L", "tiefe": 1, "seite": 1, "text": ""},
            {"id": "0.1.0", "typ": "LI", "tiefe": 2, "seite": 1, "text": "1. Ausgangslage"},
            {"id": "0.2", "typ": "Figure", "tiefe": 1, "seite": 1, "text": "", "alt": "Decorative"},
            {"id": "0.3", "typ": "P", "tiefe": 1, "seite": 2, "text": "Ausblick " * 60},
        ],
    }


def _pdf(pfad):
    import fitz
    d = fitz.open()
    for i in range(2):
        p = d.new_page(width=595, height=842)
        p.insert_text((50, 70), f"Seite {i + 1}", fontsize=18)
    d.save(pfad)
    d.close()


class StrukturlisteTest(unittest.TestCase):
    def test_zeilen(self):
        zeilen, kenn = pdf_pruefung.zeilen_fuer_seite(_struktur(), 1)
        self.assertEqual(zeilen[0], "E0.0 H1: Jahresbericht")
        self.assertIn("E0.1.0 LI: 1. Ausgangslage", zeilen)
        self.assertIn("E0.2 Figure: Alt-Text: Decorative", zeilen)
        self.assertNotIn("E0", kenn)                 # Document ist Container, keine Kennung
        self.assertIn("E0.1", kenn)                  # Liste selbst bleibt adressierbar
        self.assertTrue(all("Seite 2" not in z for z in zeilen))
        z2, _k = pdf_pruefung.zeilen_fuer_seite(_struktur(), 2)
        self.assertEqual(len(z2), 1)
        self.assertTrue(z2[0].endswith("…"))         # Text auf 200 Zeichen gekuerzt

    def test_prompt(self):
        system, prompt = build_pruefung_prompt(["E0.0 H1: X"], seite=1, seiten_gesamt=2, sprache_dokument="de-DE", sprache_ausgabe="en")
        self.assertIn("DATEN", system)
        self.assertIn("E0.0 H1: X", prompt)
        self.assertIn("Englisch", prompt)
        self.assertIn("Seite 1 von 2", prompt)


class NachpruefungTest(unittest.TestCase):
    def test_kennung_und_doppelte(self):
        _z, kenn = pdf_pruefung.zeilen_fuer_seite(_struktur(), 1)
        befunde = [
            PruefBefund(element="E0.1.0", art="rolle", befund="Überschrift als Listenpunkt getaggt", vorschlag="H2", beleg="fett, allein stehend", sicherheit="hoch"),
            PruefBefund(element="E0.1.0", art="rolle", befund="Überschrift als Listenpunkt getaggt", vorschlag="H2", beleg="x", sicherheit="hoch"),
            PruefBefund(element="E9.9", art="grafik", befund="Alt-Text passt nicht", vorschlag="", beleg="", sicherheit="hoch"),
            PruefBefund(element="", art="fehlt", befund="Sichtbarer Text ohne Tag: Impressum", vorschlag="", beleg="unten links", sicherheit="mittel"),
        ]
        out = pdf_pruefung.nachpruefung(befunde, kenn, 1)
        self.assertEqual(len(out), 3)                             # Doppelmeldung weg
        self.assertEqual(out[0]["typ"], "LI")
        self.assertEqual(out[0]["text"], "1. Ausgangslage")
        self.assertEqual(out[0]["sicherheit"], "hoch")
        self.assertEqual(out[1]["sicherheit"], "niedrig")         # unbekannte Kennung
        self.assertIn("nicht in der Strukturliste", out[1]["hinweis"])
        self.assertEqual(out[2]["element"], "")
        self.assertEqual(out[2]["seite"], 1)


class DokumentTest(unittest.TestCase):
    def test_bericht_mit_attrappe(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = os.path.join(d, "t.pdf")
            _pdf(pdf)
            aufrufe = []

            def attrappe(model, prompt, image_path, schema, max_tokens=0, temperature=0.0, system=None):
                aufrufe.append((model, image_path))
                self.assertTrue(os.path.isfile(image_path))
                if "Seite 1 von 2" in prompt:
                    return PruefSeiteOutput(befunde=[PruefBefund(element="E0.1.0", art="rolle", befund="Überschrift als Listenpunkt", vorschlag="H2", beleg="fett", sicherheit="hoch")], zusammenfassung="Eine Abweichung.")
                raise pdf_pruefung.llm_client.LLMCallError("Netz weg")

            with mock.patch.object(pdf_pruefung.llm_client, "call_with_schema", attrappe):
                b = pdf_pruefung.pruefe_dokument(pdf, _struktur(), d, sprache_ausgabe="de", dokument_name="t.pdf")
            self.assertEqual(len(aufrufe), 2)
            self.assertEqual(b["seiten"], 2)
            self.assertEqual(b["seiten_geprueft"], 1)
            self.assertEqual(len(b["befunde"]), 1)
            self.assertEqual(b["anzahl"], {"hoch": 1, "mittel": 0, "niedrig": 0, "auto": 0})
            self.assertEqual(b["hinweise"], ["Seite 2: KI-Anfrage fehlgeschlagen"])
            self.assertEqual(b["je_seite"][0]["zusammenfassung"], "Eine Abweichung.")
            self.assertTrue(os.path.isfile(os.path.join(d, "t.pdf.pruef_p1.png")))   # Seitenbild gecacht

    def test_alle_seiten_fehlgeschlagen(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = os.path.join(d, "t.pdf")
            _pdf(pdf)

            def kaputt(**kw):
                raise pdf_pruefung.llm_client.LLMCallError("x")

            with mock.patch.object(pdf_pruefung.llm_client, "call_with_schema", kaputt):
                with self.assertRaises(pdf_pruefung.PruefFehler):
                    pdf_pruefung.pruefe_dokument(pdf, _struktur(), d)

    def test_max_seiten(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = os.path.join(d, "t.pdf")
            _pdf(pdf)
            leer = lambda **kw: PruefSeiteOutput()  # noqa: E731
            with mock.patch.object(pdf_pruefung.llm_client, "call_with_schema", leer), mock.patch.object(pdf_pruefung, "MAX_SEITEN", 1):
                b = pdf_pruefung.pruefe_dokument(pdf, _struktur(), d)
            self.assertEqual(b["seiten_geprueft"], 1)
            self.assertIn("Nur die ersten 1 von 2 Seiten geprüft", b["hinweise"])


if __name__ == "__main__":
    unittest.main()
