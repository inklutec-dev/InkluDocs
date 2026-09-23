"""Tests fuer den Tagging-Weg „Struktur zuerst“ (pdf_struktur_tagging, 23.09.2026).

Deterministische Teile ohne Modell und ohne PDFix: Struktur-HTML aus einer PDF, Nachpruefung,
Stilprofil mit Klammer-Pass, Plan. Der Schreibweg (PDFix) laeuft nur, wenn das SDK da ist
(Container), mit einem Modell-Ersatz (mock) statt Gemini.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HIER), "backend"))
sys.path.insert(0, os.path.dirname(HIER))

import pdf_struktur_tagging as st  # noqa: E402
import pdf_tagging  # noqa: E402

FIXTURE = os.path.join(HIER, "fixtures", "testformular_inkludocs.pdf")


def _seiten_fixtur():
    """Zwei Seiten mit Stilen: 54 pt Titel (nur Titelseite), 15.6 pt Kapitel, 11.7 pt Zwischen."""
    def z(pno, n, size, bold, top, text):
        return {"id": f"s{pno}z{n}", "block": n, "text": text, "size": size, "bold": bold,
                "bbox_pdf": [50, 800 - top, 300, 812 - top], "top": top, "left": 50}
    return [
        {"seite": 1, "breite": 595, "hoehe": 842, "fliesstext": 12, "zeilen": [
            z(1, 1, 54, True, 100, "Titel"), z(1, 2, 24, True, 160, "Untertitel"), z(1, 3, 12, False, 300, "Text")],
         "bilder": [{"id": "s1b1", "bbox_pdf": [300, 100, 500, 300], "breite": 200, "hoehe": 200, "top": 542, "left": 300}]},
        {"seite": 2, "breite": 595, "hoehe": 842, "fliesstext": 12, "zeilen": [
            z(2, 1, 12, False, 30, "Kolumnentitel"), z(2, 2, 15.6, True, 80, "Kapitel"), z(2, 3, 11.7, True, 120, "Zwischen A"),
            z(2, 4, 12, False, 140, "Absatz"), z(2, 5, 11.7, True, 200, "Zwischen B"), z(2, 6, 15.6, True, 300, "Kapitel 2"),
            z(2, 7, 11.7, True, 340, "Zwischen C"), z(2, 8, 9, False, 800, "3")],
         "bilder": []},
    ]


class StrukturHtmlTest(unittest.TestCase):
    def test_html_aus_pdf(self):
        seiten = st.struktur_html(FIXTURE)
        self.assertGreaterEqual(len(seiten), 1)
        s = seiten[0]
        self.assertTrue(s["zeilen"], "Fixtur hat Textzeilen")
        self.assertEqual(s["zeilen"][0]["id"], "s1z1")
        self.assertIn('<section data-seite="1"', s["html"])
        self.assertIn('id="s1z1"', s["html"])
        for z in s["zeilen"]:
            l, b, r, t = z["bbox_pdf"]
            self.assertLess(l, r); self.assertLess(b, t)      # PDF-Koordinaten, Ursprung unten links
        self.assertGreater(s["fliesstext"], 0)


class NachpruefungUndStilprofilTest(unittest.TestCase):
    def test_nachpruefung_verwirft_unbekannte_und_ergaenzt_bilder(self):
        seiten = _seiten_fixtur()
        zu = {1: {"zeilen": [{"id": "s1z1", "rolle": "H1"}, {"id": "s1z9", "rolle": "H2"}, {"id": "s2z2", "rolle": "H1"}],
                  "bilder": [], "hat_tabelle": False},
              2: {"zeilen": [{"id": "s2z1", "rolle": "Artefakt"}, {"id": "s2z2", "rolle": "H1"}], "bilder": [], "hat_tabelle": True}}
        n = st.nachpruefung(seiten, zu)
        self.assertEqual(n["rollen"], {"s1z1": "H1", "s2z1": "Artefakt", "s2z2": "H1"})
        self.assertEqual(len(n["verworfen"]), 2)                 # s1z9 gibt es nicht, s2z2 gehoert nicht zu Seite 1
        self.assertEqual(n["bilder"]["s1b1"], {"inhaltlich": True, "alt": ""})   # vergessenes Bild -> inhaltlich
        self.assertEqual(n["vergessene_bilder"], 1)
        self.assertEqual(n["tabellen"], {1: False, 2: True})

    def test_stilprofil_ebenen_aus_groesse_ohne_sprung(self):
        seiten = _seiten_fixtur()
        # Modell: seitenlokale Ebenen (auf Seite 2 ist 15.6 pt „H1“), Titelseite 54 pt H1, Untertitel H2
        rollen = {"s1z1": "H1", "s1z2": "H2", "s2z1": "Artefakt", "s2z2": "H1", "s2z3": "H2", "s2z5": "H2", "s2z6": "H1", "s2z7": "H2", "s2z8": "Artefakt"}
        sp = st.stilprofil(seiten, rollen)
        r = sp["rollen"]
        self.assertEqual(r["s1z1"], "H1")                        # groesster Stil = H1
        self.assertEqual(r["s1z2"], "H2")                        # Untertitel nur auf Titelseite: Ebene des naechsten wiederkehrenden Stils
        self.assertEqual(r["s2z2"], "H2")                        # 15.6 pt Kapitel = H2 (schiebt nicht nach unten)
        self.assertEqual(r["s2z3"], "H3"); self.assertEqual(r["s2z5"], "H3")   # 11.7 pt = H3, beide gleich
        self.assertEqual(r["s2z6"], "H2"); self.assertEqual(r["s2z7"], "H3")
        self.assertEqual(r["s2z1"], "Artefakt"); self.assertEqual(r["s2z8"], "Artefakt")
        self.assertIn("54 pt fett = H1", sp["profil"])
        folge = [int(r[z][1]) for z in ("s1z1", "s1z2", "s2z2", "s2z3", "s2z5", "s2z6", "s2z7")]
        self.assertTrue(all(b <= a + 1 for a, b in zip(folge, folge[1:])), folge)   # nie ein Sprung

    def test_klammer_pass_hebt_zu_tiefe_ebene_an(self):
        seiten = _seiten_fixtur()
        # Nur 11.7 pt direkt nach dem Titel (kein 15.6 davor): H3 waere ein Sprung -> H2
        rollen = {"s1z1": "H1", "s2z2": "H1", "s2z3": "H2"}
        seiten[1]["zeilen"][1]["size"] = 11.7   # s2z2 auch 11.7 pt
        sp = st.stilprofil(seiten, rollen)
        self.assertEqual(sp["rollen"]["s2z2"], "H2")
        self.assertEqual(sp["rollen"]["s2z3"], "H2")
        self.assertGreaterEqual(sp["geklammert"], 0)

    def test_plan(self):
        seiten = _seiten_fixtur()
        rollen = {"s1z1": "H1", "s2z1": "Artefakt", "s2z2": "H2"}
        bilder = {"s1b1": {"inhaltlich": False, "alt": ""}}
        plan = st.plan_erzeugen(seiten, rollen, bilder, {1: False, 2: True}, "de-DE")
        self.assertEqual(plan["sprache"], "de-DE")
        s1, s2 = plan["seiten"]
        self.assertEqual(s1["rollen"], [{"bbox": list(seiten[0]["zeilen"][0]["bbox_pdf"]), "tag": "H1", "zeilen": 1}])
        self.assertEqual(s1["bilder"][0]["artefakt"], True)
        self.assertFalse(s1["tabellen"]); self.assertTrue(s2["tabellen"])
        self.assertEqual(len(s2["artefakte"]), 1)                # Kolumnentitel
        self.assertEqual(len(s2["zeilen"]), len(seiten[1]["zeilen"]) - 1)   # Artefakt-Zeile nicht in der Vollstaendigkeitsliste


class ListenTest(unittest.TestCase):
    def test_listen_mit_umbruch(self):
        def z(n, top, left, text):
            return {"id": f"s1z{n}", "block": n, "text": text, "size": 10, "bold": False,
                    "bbox_pdf": [left, 800 - top, 300, 812 - top], "top": top, "left": left}
        s = {"seite": 1, "zeilen": [
            z(1, 100, 50, "Material"),
            z(2, 120, 52, "■ Seile"), z(3, 134, 52, "■ kleine Kästen"),
            z(4, 148, 52, "■ Die Hindernisse sind:"), z(5, 162, 62, "Mauersprung und Hochtiefsprung"), z(6, 176, 62, "hintereinander."),
            z(7, 210, 50, "Ein normaler Absatz danach."),
            z(8, 240, 52, "1. Erster Schritt"), z(9, 254, 52, "2. Zweiter Schritt")]}
        listen = st.listen_erkennen(s)
        self.assertEqual(len(listen), 2)
        # Datum ist kein Listenpunkt; „(1) …“-Absaetze ohne Einzug: Folgezeile ohne Satzende gehoert dazu, naechste Nummer bleibt in derselben Liste
        s2 = {"seite": 1, "zeilen": [z(1, 100, 50, "Datum: 23.09.2026"), z(2, 114, 50, "23.09.2026 Lieferung"),
                                    z(3, 140, 50, "(1) Der Auftragnehmer verarbeitet Daten nur"), z(4, 154, 50, "im Auftrag des Verantwortlichen."),
                                    z(5, 168, 50, "(2) Weisungen erfolgen schriftlich.")]}
        l2 = st.listen_erkennen(s2)
        self.assertEqual(len(l2), 1)
        self.assertEqual([pkt["ids"] for pkt in l2[0]], [["s1z3", "s1z4"], ["s1z5"]])
        # Eine Ueberschrift (Rolle) oder eine Zeile in anderem Stil direkt unter einem Punkt ist keine Fortsetzung
        s3 = {"seite": 1, "zeilen": [z(1, 100, 52, "■ Station 6: Kegeln (kegeln)"),
                                    dict(z(2, 114, 50, "Materialliste für alle Stationen"), size=15.6, bold=True),
                                    z(3, 140, 52, "■ Seile")]}
        l3 = st.listen_erkennen(s3, {"s1z2": "H2"})
        self.assertEqual([[pkt["ids"] for pkt in l] for l in l3], [[["s1z1"]], [["s1z3"]]])
        l3b = st.listen_erkennen(s3)   # ohne Rolle: anderer Stil reicht
        self.assertEqual([[pkt["ids"] for pkt in l] for l in l3b], [[["s1z1"]], [["s1z3"]]])
        self.assertEqual([len(pkt["ids"]) for pkt in listen[0]], [1, 1, 3])      # dritter Punkt mit zwei Fortsetzungszeilen
        self.assertEqual(listen[0][2]["ids"], ["s1z4", "s1z5", "s1z6"])
        self.assertEqual([pkt["ids"] for pkt in listen[1]], [["s1z8"], ["s1z9"]])
        plan = st.plan_erzeugen([dict(s, breite=595, hoehe=842, fliesstext=10, bilder=[])], {"s1z1": "H2"}, {}, {}, "de-DE")
        self.assertEqual(len(plan["seiten"][0]["listen"]), 2)
        self.assertEqual(len(plan["seiten"][0]["listen"][0][2]["bboxes"]), 3)


class FormularUndVektorTest(unittest.TestCase):
    """23.09.2026 (Mannheimer-Antrag): Steuerzeichen, mehrzeilige Ueberschriften, Vektorgruppen."""

    def test_druckbar(self):
        self.assertFalse(st.druckbar("\x08"))
        self.assertFalse(st.druckbar("  \x08 "))
        self.assertTrue(st.druckbar("Euro"))

    def _z(self, n, top, left, text, size=12.3, bold=False, rechts=None):
        return {"id": f"s1z{n}", "block": n, "text": text, "size": size, "bold": bold,
                "bbox_pdf": [left, 800 - top, rechts or left + 230, 812 - top], "top": top, "left": left}

    def test_mehrzeiliger_titel_mit_nachbarspalte(self):
        z = self._z
        s = {"seite": 1, "zeilen": [z(1, 100, 48, "Antrag auf Haus- und Grundbesitzerhaftpflicht-"), z(2, 101, 382, "GS-Nr.:", 6.1, rechts=400),
                                    z(3, 115, 48, "versicherung ausschließlich oder überwiegend"), z(4, 116, 382, "VS-Nr.:", 6.1, rechts=400),
                                    z(5, 130, 48, "gewerblich genutzter Gebäude."), z(6, 160, 48, "Nicht versicherbar sind …", 8.5)]}
        g = st.rollen_gruppen(s, {"s1z1": "H1", "s1z3": "H1", "s1z5": "H1"})
        self.assertEqual(len(g), 1)
        self.assertEqual((g[0]["tag"], g[0]["zeilen"]), ("H1", 3))

    def test_absatz_dazwischen_trennt(self):
        z = self._z
        s = {"seite": 1, "zeilen": [z(1, 100, 48, "Kapitel A"), z(2, 113, 48, "ein Absatz dazwischen", 10), z(3, 126, 48, "Kapitel B")]}
        g = st.rollen_gruppen(s, {"s1z1": "H2", "s1z3": "H2"})
        self.assertEqual([x["zeilen"] for x in g], [1, 1])

    def test_vektorgruppen_nur_komplexe_zeichnungen(self):
        import fitz
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "v.pdf")
            d = fitz.open(); p = d.new_page(width=595, height=842)
            p.insert_text((50, 60), "Diagramm und Kästchen", fontsize=12)
            p.draw_rect(fitz.Rect(50, 100, 62, 112))                     # Ankreuzkästchen: einfach -> kein Bild
            sh = p.new_shape()
            for i in range(12):                                           # Kurvendiagramm: komplex -> Bildkandidat
                sh.draw_bezier((100 + i * 10, 400), (105 + i * 10, 300), (110 + i * 10, 350), (120 + i * 10, 380))
            sh.finish(); sh.commit(); d.save(pfad); d.close()
            seiten = st.struktur_html(pfad)
        vek = [b for b in seiten[0]["bilder"] if b.get("vektor")]
        self.assertEqual(len(vek), 1)
        self.assertIn('data-art="vektorzeichnung"', seiten[0]["html"])


class KonfigStrukturTest(unittest.TestCase):
    def test_konfig_ohne_strukturerkennung(self):
        with tempfile.TemporaryDirectory() as t:
            ziel = os.path.join(t, "k.json")
            pdf_tagging.konfig_erzeugen("de-DE", False, ziel, struktur_vorgegeben=True)
            k = json.load(open(ziel, encoding="utf-8"))
        namen = [a["name"] for a in k["actions"]]
        self.assertNotIn("add_tags", namen)
        self.assertNotIn("fix_headings", namen)                  # fuellt Spruenge mit LEEREN Tags — nicht bei vorgegebener Struktur
        self.assertIn("tag_annot", namen); self.assertIn("set_language", namen); self.assertIn("set_pdf_ua_standard", namen)
        self.assertLess(namen.index("create_web_links"), namen.index("tag_annot"))


@unittest.skipUnless(st.verfuegbar(), "PDFix-Skripte nicht eingerichtet")
class SchreibwegTest(unittest.TestCase):
    """Ganzer Weg auf der Fixtur mit Modell-Ersatz: erste Zeile wird H1, alles andere Absatz."""

    def test_taggen_mit_modell_ersatz(self):
        seiten = st.struktur_html(FIXTURE)
        erste = seiten[0]["zeilen"][0]["id"]

        class _Out:
            def __init__(self, d):
                self._d = d

            def model_dump(self):
                return self._d

        def fake_call(model, prompt, image_path, schema, max_tokens, temperature, system):
            import re
            m = re.search(r"Seite (\d+) von \d+", prompt)
            pno = int(m.group(1)) if m else 1
            zeilen = [{"id": erste, "rolle": "H1", "beleg": "Test"}] if pno == 1 else []
            return _Out({"zeilen": zeilen, "bilder": [], "hat_tabelle": False})

        with tempfile.TemporaryDirectory() as t, mock.patch.object(st.llm_client, "call_with_schema", side_effect=fake_call):
            out = os.path.join(t, "fertig.pdf")
            b = st.taggen(FIXTURE, out, "de", arbeitsordner=t, fortschritt=lambda a, b_: None)
            self.assertTrue(os.path.isfile(out))
            self.assertEqual(b["weg"], "struktur")
            self.assertGreaterEqual(b["nachher"]["ueberschriften"], 1)
            self.assertEqual(b["struktur"]["ueberschriften"], 1)
            self.assertEqual(b["struktur"]["geschrieben"]["rollen"], 1)
            self.assertNotIn("fix_headings", [a for a in b["konfig"].get("entfernt", [])])   # nur Info: Bericht traegt konfig
            from pdf_export import pdf_hat_tags
            self.assertTrue(pdf_hat_tags(out))


if __name__ == "__main__":
    unittest.main()
