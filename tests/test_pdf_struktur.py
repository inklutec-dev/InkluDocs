"""Unit-Tests Strukturlesung/Hoerprobe/Strukturansicht (pdf_struktur.py, 22.09.2026) — ohne PDFix.
Aufruf im Container: python3 -m unittest /app/tests/test_pdf_struktur.py"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
sys.path.insert(0, "/app")
import pdf_struktur  # noqa: E402


def _fixture():
    """Kleiner Tag-Baum: Document > H1, P, L(LI, LI), Table(TR(TH,TH), TR(TD,TD)), Figure ohne Alt,
    Figure mit Alt, Form (vorname, ohne /TU), P mit Skript-Text auf Seite 2."""
    return {
        "info": {"seiten": 2, "elemente": 16, "lang": "de-DE"},
        "elemente": [
            {"id": "0", "typ": "Document", "tiefe": 0, "seite": 1, "text": "", "kinder": 8},
            {"id": "0.0", "typ": "H1", "tiefe": 1, "seite": 1, "text": "Vertrag", "kinder": 0},
            {"id": "0.1", "typ": "P", "tiefe": 1, "seite": 1, "text": "Dieser Vertrag wird geschlossen.", "kinder": 0},
            {"id": "0.2", "typ": "L", "tiefe": 1, "seite": 1, "text": "", "kinder": 2},
            {"id": "0.2.0", "typ": "LI", "tiefe": 2, "seite": 1, "text": "1. Ausgangslage", "kinder": 2},
            {"id": "0.2.1", "typ": "LI", "tiefe": 2, "seite": 1, "text": "2. Ausblick", "kinder": 2},
            {"id": "0.3", "typ": "Table", "tiefe": 1, "seite": 1, "text": "", "zeilen": 2, "spalten": 2, "kinder": 2},
            {"id": "0.3.0", "typ": "TR", "tiefe": 2, "seite": 1, "text": "", "kinder": 2},
            {"id": "0.3.0.0", "typ": "TH", "tiefe": 3, "seite": 1, "text": "Posten", "kinder": 0},
            {"id": "0.3.0.1", "typ": "TH", "tiefe": 3, "seite": 1, "text": "Betrag", "kinder": 0},
            {"id": "0.3.1", "typ": "TR", "tiefe": 2, "seite": 1, "text": "", "kinder": 2},
            {"id": "0.3.1.0", "typ": "TD", "tiefe": 3, "seite": 1, "text": "Miete", "kinder": 0},
            {"id": "0.3.1.1", "typ": "TD", "tiefe": 3, "seite": 1, "text": "500 <b>EUR</b>", "kinder": 1},
            {"id": "0.3.1.1.0", "typ": "Form", "tiefe": 4, "seite": 1, "text": "", "feldname": "betrag", "quickinfo": "Betrag in Euro", "kinder": 0},
            {"id": "0.4", "typ": "Figure", "tiefe": 1, "seite": 1, "text": "", "kinder": 0},
            {"id": "0.5", "typ": "Figure", "tiefe": 1, "seite": 2, "text": "", "alt": "Logo der Firma", "kinder": 0},
            {"id": "0.6", "typ": "Form", "tiefe": 1, "seite": 2, "text": "", "feldname": "vorname", "kinder": 0},
            {"id": "0.7", "typ": "P", "tiefe": 1, "seite": 2, "text": "<script>alert(1)</script> Ende", "kinder": 0},
        ],
    }


class HoerprobeTest(unittest.TestCase):
    def test_zeilen(self):
        z = pdf_struktur.hoerprobe(_fixture(), felder_quickinfos={"vorname": "Vorname eingeben"})
        self.assertEqual(z[0], "Sprache: de-DE")
        self.assertEqual(z[1], "Seiten: 2")
        self.assertTrue(z[2].startswith("Zusammenfassung: 1 Überschriften, 1 Listen, 1 Tabellen, 2 Grafiken (1 ohne Alt-Text), 2 Formularfelder."), z[2])
        # Feld in einer Zelle: eigene Zeile direkt nach der Zeile der Tabelle (Lesereihenfolge)
        self.assertEqual(z[z.index("Zeile: Miete | 500 <b>EUR</b>") + 1], "Formularfeld betrag: Betrag in Euro")
        self.assertIn("— Seite 1 —", z)
        self.assertIn("— Seite 2 —", z)
        self.assertIn("Überschrift Ebene 1: Vertrag", z)
        self.assertIn("Absatz: Dieser Vertrag wird geschlossen.", z)
        self.assertIn("Liste mit 2 Einträgen", z)
        self.assertIn("Listenpunkt: 1. Ausgangslage", z)
        self.assertIn("Tabelle mit 2 Zeilen und 2 Spalten", z)
        self.assertIn("Kopfzeile: Posten | Betrag", z)
        self.assertIn("Zeile: Miete | 500 <b>EUR</b>", z)
        self.assertIn("Grafik ohne Alt-Text", z)
        self.assertIn("Grafik: Logo der Firma", z)
        self.assertIn("Formularfeld vorname: Vorname eingeben", z)
        # Container (Document, TR, TH, TD) erzeugen keine eigenen Zeilen
        self.assertFalse(any(x.startswith(("Document", "TR", "TH:", "TD:")) for x in z), z)
        # Seite 2 wird erst vor dem ersten Element der Seite 2 angesagt, und nur einmal
        self.assertEqual(z.count("— Seite 2 —"), 1)
        self.assertLess(z.index("Grafik ohne Alt-Text"), z.index("— Seite 2 —"))

    def test_ohne_quickinfo_und_sprache(self):
        f = _fixture()
        f["info"].pop("lang")
        z = pdf_struktur.hoerprobe(f)
        self.assertEqual(z[0], "Sprache: nicht gesetzt")
        self.assertIn("Formularfeld vorname ohne Quickinfo", z)

    def test_uebersetzung(self):
        z = pdf_struktur.hoerprobe(_fixture(), lambda s: s.replace("Seiten", "Pages"))
        self.assertEqual(z[1], "Pages: 2")

    def test_kuerzung(self):
        f = _fixture()
        f["elemente"][2]["text"] = "x" * 1000
        z = pdf_struktur.hoerprobe(f)
        lang = [x for x in z if x.startswith("Absatz: xxx")][0]
        self.assertLessEqual(len(lang), pdf_struktur.MAX_ZEILE + 12)
        self.assertTrue(lang.endswith("…"))


class HtmlTest(unittest.TestCase):
    def test_aufbau_und_escaping(self):
        h = pdf_struktur.html_ansicht(_fixture(), felder_quickinfos={"vorname": "Vorname eingeben"}, ebene_versatz=1)
        self.assertIn("<h2>Vertrag</h2>", h)                 # H1 der PDF wird h2 (Seite behaelt ihre H1)
        self.assertIn("<p>Dieser Vertrag wird geschlossen.</p>", h)
        self.assertIn("<ul>", h)
        self.assertIn("<li>1. Ausgangslage</li>", h)
        self.assertEqual(h.count("<ul>"), h.count("</ul>"))
        self.assertIn('<th scope="col">Posten</th>', h)
        # Feld in der Zelle steht IN der Zelle (kein <p> in der Tabelle) und nicht noch einmal danach
        self.assertIn('<td>500 &lt;b&gt;EUR&lt;/b&gt; <span class="struktur-rolle">Formularfeld betrag:</span> Betrag in Euro</td>', h)
        self.assertEqual(h.count("Formularfeld betrag"), 1)
        self.assertEqual(h.count("<table"), h.count("</table>"))
        self.assertIn("<figure>", h)
        self.assertIn("Logo der Firma", h)
        self.assertIn('<span class="struktur-rolle">Formularfeld vorname:</span> Vorname eingeben', h)
        self.assertNotIn("<script", h)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; Ende", h)
        # Liste wird vor der Tabelle geschlossen, Tabelle vor der Grafik
        self.assertLess(h.index("</ul>"), h.index("<table"))
        self.assertLess(h.index("</table>"), h.index("<figure>"))
        # Seitenmarken
        self.assertIn("— Seite 2 —", h)

    def test_ebenen_deckel(self):
        f = {"info": {}, "elemente": [{"id": "0", "typ": "H6", "tiefe": 0, "seite": 1, "text": "Tief", "kinder": 0}]}
        self.assertIn("<h6>Tief</h6>", pdf_struktur.html_ansicht(f, ebene_versatz=1))


class LesenTest(unittest.TestCase):
    def test_fehlende_datei(self):
        with self.assertRaises(pdf_struktur.StrukturFehler):
            pdf_struktur.lesen("/nirgends/x.pdf", "/tmp")

    def test_cache_wird_benutzt(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = os.path.join(d, "a.pdf")
            with open(pdf, "wb") as f:
                f.write(b"%PDF-1.4 fake")
            cache = os.path.join(d, "a.pdf.struktur.json")
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({"info": {"seiten": 1}, "elemente": []}, f)
            os.utime(cache, (time.time() + 5, time.time() + 5))
            self.assertEqual(pdf_struktur.lesen(pdf, d)["info"]["seiten"], 1)

    def test_kein_cache_bei_aelterer_datei(self):
        """Ist die PDF neuer als der Cache (Neu-Taggen), wird neu gelesen — hier: das Skript fehlt oder
        die PDF ist keine, also ein sauberer StrukturFehler statt alter Daten."""
        with tempfile.TemporaryDirectory() as d:
            pdf = os.path.join(d, "a.pdf")
            cache = os.path.join(d, "a.pdf.struktur.json")
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({"info": {"seiten": 99}, "elemente": []}, f)
            os.utime(cache, (time.time() - 100, time.time() - 100))
            with open(pdf, "wb") as f:
                f.write(b"%PDF-1.4 fake")
            try:
                r = pdf_struktur.lesen(pdf, d)
                self.assertNotEqual(r["info"].get("seiten"), 99)
            except pdf_struktur.StrukturFehler:
                pass


if __name__ == "__main__":
    unittest.main()
