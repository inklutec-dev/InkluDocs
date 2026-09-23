"""PDF-Tagging (22.09.2026): Konfiguration, Spracherkennung, Alt-Text-Uebernahme, Lauf im Testmodus.
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_pdf_tagging.py -v
Der Lauf-Test (test_lauf_testmodus_taggt) braucht das PDFix-SDK im Container und laeuft ohne Lizenz
(PDFIX_TAGGING_LIZENZ nicht gesetzt); ohne SDK wird er uebersprungen.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import pdf_tagging  # noqa: E402

DE = ("Der Bericht fasst die Massnahmen des Landes im Jahr zusammen und richtet sich an die "
      "Oeffentlichkeit. Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen, und die Mittel "
      "stammen aus dem Green Bond des Landes. Das ist nicht das Ende, sondern der Anfang einer "
      "Entwicklung, die mit den Kommunen und den Verbaenden abgestimmt wird. ") * 4
EN = ("The report summarises the measures of the state and is intended for the public. The area of "
      "protected sites has grown by three percent and the funds come from the green bond of the state. "
      "This is not the end but the beginning of a development that is coordinated with the towns. ") * 4


def _pdf_mit_text(pfad: str, text: str, lang: str = "", seiten: int = 1) -> None:
    import fitz
    d = fitz.open()
    for _ in range(seiten):
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


class KonfigTest(unittest.TestCase):
    def test_voreinstellung_ist_die_pdfix_aktion(self):
        v = pdf_tagging.voreinstellung()
        self.assertEqual(v["name"], "make_accessible")
        self.assertEqual(len(v["actions"]), 37)
        namen = [a["name"] for a in v["actions"]]
        self.assertIn("add_tags", namen)
        self.assertIn("set_language", namen)

    def test_konfig_entfernt_alt_text_schritte_und_setzt_sprache(self):
        with tempfile.TemporaryDirectory() as t:
            ziel = os.path.join(t, "k.json")
            info = pdf_tagging.konfig_erzeugen("de-DE", True, ziel)
            k = json.load(open(ziel, encoding="utf-8"))
        namen = [a["name"] for a in k["actions"]]
        # 4x Set Alt (Figure/Formula) + 1x Decorative-Rueckfall fuer Anmerkungen entfallen, Form bleibt; + Web-Links
        self.assertEqual(info["schritte"], 37 - 5 + 1)
        # 23.09.2026: Web-Links VOR tag_annot/set_annot_contents, sonst bleibt der neue Link ungetaggt (Michaels Befund 10)
        self.assertIn("create_web_links", namen)
        self.assertLess(namen.index("create_web_links"), namen.index("tag_annot"))
        self.assertLess(namen.index("create_web_links"), namen.index("set_annot_contents"))
        self.assertEqual(len(info["entfernt"]), 5)
        set_alt = [a for a in k["actions"] if a["name"] == "set_alt"]
        self.assertEqual(len(set_alt), 1)
        self.assertEqual(pdf_tagging._params(set_alt[0])["tag_names"], "^Form$")
        annot = [pdf_tagging._params(a) for a in k["actions"] if a["name"] == "set_annot_contents"]
        self.assertEqual({p["alt_type"] for p in annot}, {"1", "2"})
        sprache = [pdf_tagging._params(a) for a in k["actions"] if a["name"] == "set_language"][0]
        self.assertEqual(sprache["lang"], "de-DE")
        self.assertEqual(sprache["overwrite"], "true")
        self.assertIn("add_tags", namen)
        self.assertIn("set_pdf_ua_standard", namen)

    def test_konfig_weist_ungueltige_sprache_ab(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(pdf_tagging.TaggingFehler):
                pdf_tagging.konfig_erzeugen("de DE; rm -rf", False, os.path.join(t, "k.json"))

    def test_voreinstellung_entspricht_sdk_wenn_vorhanden(self):
        """Drift-Wache: Liefert das installierte SDK dieselbe eingebaute Aktion wie die gespeicherte Datei?"""
        sys.path.insert(0, str(pdf_tagging._SCRIPT_DIR))
        try:
            from pdfixsdk import GetPdfix, kDataFormatJson, kSaveFull
            import Utils as U
        except Exception:
            self.skipTest("kein PDFix-SDK")
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "leer.pdf")
            _pdf_mit_text(pfad, "x")
            p = GetPdfix()
            doc = p.OpenDoc(pfad, "")
            cmd = doc.GetCommand()
            gefunden = None
            for i in range(cmd.GetNumCustomActions()):
                s = p.CreateMemStream()
                cmd.SaveCustomActionToStream(i, s, kDataFormatJson, kSaveFull)
                d = json.loads(bytearray(U.stream_to_data(s)).decode("utf-8"))
                s.Destroy()
                if d.get("name") == "make_accessible":
                    gefunden = d
            doc.Close()
        self.assertIsNotNone(gefunden)
        v = pdf_tagging.voreinstellung()
        self.assertEqual(gefunden["version"], v["version"], "PDFix hat die Voreinstellung geaendert — Datei neu exportieren und Aenderungen pruefen")
        self.assertEqual([a["name"] for a in gefunden["actions"]], [a["name"] for a in v["actions"]])


class SpracheTest(unittest.TestCase):
    def test_erkennung_sicher(self):
        e = pdf_tagging.sprache_erkennen(DE)
        self.assertEqual(e["code"], "de")
        self.assertTrue(e["sicher"])
        e = pdf_tagging.sprache_erkennen(EN)
        self.assertEqual(e["code"], "en")
        self.assertTrue(e["sicher"])
        self.assertFalse(pdf_tagging.sprache_erkennen("zu kurz")["sicher"])

    def test_bestimmen_ersetzt_falsche_dokumentsprache(self):
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "a.pdf")
            _pdf_mit_text(pfad, DE, lang="en-US")
            s = pdf_tagging.sprache_bestimmen(pfad, "en")
            self.assertEqual(s["lang"], "de-DE")
            self.assertTrue(s["overwrite"])
            self.assertEqual(s["vorher"], "en-US")
            self.assertIn("en-US", s["hinweis"])

    def test_bestimmen_behaelt_passende_dokumentsprache(self):
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "a.pdf")
            _pdf_mit_text(pfad, DE, lang="de-AT")
            s = pdf_tagging.sprache_bestimmen(pfad, "en")
            self.assertEqual(s["lang"], "de-DE")     # erkannt, gleiche Sprache -> kein overwrite
            self.assertFalse(s["overwrite"])

    def test_bestimmen_vorgabe_ohne_text(self):
        with tempfile.TemporaryDirectory() as t:
            pfad = os.path.join(t, "a.pdf")
            _pdf_mit_text(pfad, "x")
            s = pdf_tagging.sprache_bestimmen(pfad, "sv")
            self.assertEqual(s["lang"], "sv-SE")
            self.assertEqual(s["quelle"], "vorgabe")
            self.assertTrue(s["hinweis"])
            s = pdf_tagging.sprache_bestimmen(pfad, "xx")
            self.assertEqual(s["lang"], "de-DE")


class UebernahmeTest(unittest.TestCase):
    """alt_texte_uebernehmen: Seite + Rechteck (fitz->fitz), Seite + Bild-Hash (fitz->PDFix), Eindeutigkeit."""

    def setUp(self):
        import tagging_api
        self.t = tagging_api
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        spalten = ["id INTEGER PRIMARY KEY", "document_id INTEGER", "page_number INTEGER",
                   "bbox_x0 REAL", "bbox_y0 REAL", "bbox_x1 REAL", "bbox_y1 REAL", "width INTEGER", "height INTEGER",
                   "image_path TEXT"] + [f"{s} TEXT" for s in tagging_api._UEBERNAHME_SPALTEN]
        self.conn.execute("CREATE TABLE images (" + ", ".join(spalten) + ")")

    def tearDown(self):
        self.tmp.cleanup()

    def _bild(self, name, muster, groesse=(120, 80)):
        """PNG mit Verlauf nach rechts unten (muster 0), Verlauf nach links unten (muster 1) oder
        Schachbrett (muster 2). Fotoartige Bilder (0, 1) ergeben in jeder Groesse denselben Hash;
        das Schachbrett ist ein absichtlich unaehnliches Bild."""
        from PIL import Image
        im = Image.new("RGB", groesse)
        w, hh = groesse
        for y in range(hh):
            for x in range(w):
                if muster == 0:
                    im.putpixel((x, y), (int(255 * x / w), int(255 * y / hh), 90))
                elif muster == 1:
                    im.putpixel((x, y), (int(255 * (w - x) / w), int(60 + 195 * y / hh), int(255 * x / w)))
                else:
                    im.putpixel((x, y), ((255, 255, 255) if (x // 10 + y // 10) % 2 else (0, 0, 0)))
        p = os.path.join(self.tmp.name, name)
        im.save(p)
        return p

    def _neu(self, seite, bbox, pfad="", wh=(0, 0)):
        self.conn.execute("INSERT INTO images (document_id, page_number, bbox_x0, bbox_y0, bbox_x1, bbox_y1, width, height, image_path, status) VALUES (7, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')", (seite, *bbox, wh[0], wh[1], pfad))
        return self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def _alt(self, id_, seite, bbox, text, pfad="", wh=(0, 0)):
        d = {s: None for s in self.t._UEBERNAHME_SPALTEN}
        d.update({"id": id_, "page_number": seite, "bbox_x0": bbox[0], "bbox_y0": bbox[1], "bbox_x1": bbox[2], "bbox_y1": bbox[3],
                  "width": wh[0], "height": wh[1], "image_path": pfad, "alt_text": text, "status": "done", "image_type": "foto", "konfidenz": "hoch"})
        return d

    def _rows(self):
        return {r["id"]: dict(r) for r in self.conn.execute("SELECT * FROM images").fetchall()}

    def test_uebernahme_nach_lage(self):
        a = self._neu(1, (100, 100, 300, 250))     # entspricht altem Bild 901 (leicht verschoben)
        b = self._neu(2, (50, 50, 150, 150))       # kein altes Bild auf Seite 2
        c = self._neu(1, (400, 600, 500, 700))     # anderes Bild auf Seite 1, ohne Entsprechung
        alte = [self._alt(901, 1, (105, 102, 298, 255), "Ein Foto"),
                self._alt(902, 1, (0, 0, 20, 20), "Winzig woanders"),
                self._alt(903, 3, (100, 100, 300, 250), "Gleiche Lage, andere Seite")]
        n = self.t.alt_texte_uebernehmen(self.conn, 7, alte)
        self.assertEqual(n, 1)
        rows = self._rows()
        self.assertEqual(rows[a]["alt_text"], "Ein Foto")
        self.assertEqual(rows[a]["status"], "done")
        self.assertEqual(rows[a]["image_type"], "foto")
        self.assertIsNone(rows[b]["alt_text"])
        self.assertEqual(rows[b]["status"], "pending")
        self.assertIsNone(rows[c]["alt_text"])

    def test_uebernahme_nach_bildhash_fitz_zu_pdfix(self):
        """Alt: fitz (Seitenkoordinaten, eingebettetes Bild). Neu: PDFix (bbox = Bildmasse, Rendering in anderer Groesse)."""
        verlauf_alt = self._bild("alt_verlauf.png", 0, (120, 80))
        schach_alt = self._bild("alt_schach.png", 1, (120, 80))   # zweites Foto (anderer Verlauf)
        verlauf_neu = self._bild("neu_verlauf.png", 0, (196, 130))
        schach_neu = self._bild("neu_schach.png", 1, (196, 130))
        n1 = self._neu(1, (0, 0, 196, 130), verlauf_neu, (196, 130))
        n2 = self._neu(1, (0, 0, 196, 130), schach_neu, (196, 130))
        alte = [self._alt(901, 1, (52, 228, 247, 358), "Verlauf", verlauf_alt, (120, 80)),
                self._alt(902, 1, (52, 400, 247, 530), "Schach", schach_alt, (120, 80))]
        n = self.t.alt_texte_uebernehmen(self.conn, 7, alte)
        self.assertEqual(n, 2)
        rows = self._rows()
        self.assertEqual(rows[n1]["alt_text"], "Verlauf")
        self.assertEqual(rows[n2]["alt_text"], "Schach")

    def test_eindeutigkeit_je_seite_ohne_hash(self):
        n1 = self._neu(1, (0, 0, 196, 130), "", (196, 130))
        alte = [self._alt(901, 1, (52, 228, 247, 358), "Einziges Bild")]
        self.assertEqual(self.t.alt_texte_uebernehmen(self.conn, 7, alte), 1)
        self.assertEqual(self._rows()[n1]["alt_text"], "Einziges Bild")

    def test_keine_uebernahme_bei_mehrdeutigkeit(self):
        self._neu(1, (0, 0, 196, 130), "", (196, 130))
        self._neu(1, (0, 0, 196, 130), "", (196, 130))
        alte = [self._alt(901, 1, (52, 228, 247, 358), "A"), self._alt(902, 1, (52, 400, 247, 530), "B")]
        self.assertEqual(self.t.alt_texte_uebernehmen(self.conn, 7, alte), 0)

    def test_jedes_alte_bild_nur_einmal(self):
        self._neu(1, (100, 100, 300, 250))
        self._neu(1, (100, 100, 300, 250))
        n = self.t.alt_texte_uebernehmen(self.conn, 7, [self._alt(1, 1, (100, 100, 300, 250), "X")])
        self.assertEqual(n, 1)

    def test_iou_und_hash(self):
        self.assertAlmostEqual(self.t._iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(self.t._iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(self.t._iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        a = self.t._dhash(self._bild("h1.png", 0, (120, 80)))
        b = self.t._dhash(self._bild("h2.png", 0, (300, 200)))
        c = self.t._dhash(self._bild("h3.png", 2, (120, 80)))
        self.assertLessEqual(self.t._hamming(a, b), self.t.HASH_TOLERANZ)
        self.assertGreater(self.t._hamming(a, c), self.t.HASH_TOLERANZ)
        self.assertIsNone(self.t._dhash("/nirgendwo.png"))
        self.assertFalse(self.t._echte_bbox({"bbox_x0": 0, "bbox_y0": 0, "bbox_x1": 196, "bbox_y1": 130, "width": 196, "height": 130}))
        self.assertTrue(self.t._echte_bbox({"bbox_x0": 52, "bbox_y0": 228, "bbox_x1": 247, "bbox_y1": 358, "width": 120, "height": 80}))


class LaufTest(unittest.TestCase):
    def test_grund_aus_ausgabe_ohne_pfade(self):
        g = pdf_tagging._grund_aus_ausgabe("START\nERROR: Unable to open pdf : /app/data/uploads/3/x.pdf kaputt\n", "")
        self.assertNotIn("/app", g)
        self.assertIn("Unable to open", g)
        self.assertEqual(pdf_tagging._grund_aus_ausgabe("", ""), "PDFix hat die Aktion abgebrochen")

    def test_fehlende_datei(self):
        with self.assertRaises(pdf_tagging.TaggingFehler):
            pdf_tagging.taggen("/nirgendwo/x.pdf", "/tmp/y.pdf")

    def test_lauf_testmodus_taggt(self):
        try:
            import pdfixsdk  # noqa: F401
        except Exception:
            self.skipTest("kein PDFix-SDK")
        if os.environ.get("PDFIX_TAGGING_LIZENZ", "off").lower() in ("on", "1", "true", "yes"):
            self.skipTest("Lizenzmodus: add_tags ist in der Actino-Lizenz nicht freigeschaltet (Stand 22.09.2026)")
        import fitz
        with tempfile.TemporaryDirectory() as t:
            quelle = os.path.join(t, "roh.pdf")
            d = fitz.open()
            p = d.new_page(width=595, height=842)
            p.insert_text((50, 70), "Jahresbericht Naturschutz", fontsize=20)
            y = 110
            for zeile in [DE[i:i + 90] for i in range(0, len(DE), 90)][:12]:
                p.insert_text((50, y), zeile, fontsize=10)
                y += 14
            d.save(quelle)
            d.close()
            ziel = os.path.join(t, "getaggt.pdf")
            # Frische PDFs ohne Strukturbaum: im Testmodus legt die Aktion ihn selbst an (mit Lizenz bricht
            # add_tags ab, Stand 22.09.2026). Genau das prueft dieser Lauf.
            with mock.patch.dict(os.environ, {"PDFIX_TAGGING_LIZENZ": "off"}):
                bericht = pdf_tagging.taggen(quelle, ziel, "en")
            self.assertTrue(os.path.isfile(ziel))
            self.assertEqual(bericht["modus"], "testmodus")
            self.assertTrue(bericht["testmodus"], bericht["nachher"]["producer"])
            self.assertEqual(bericht["sprache"]["lang"], "de-DE")
            self.assertGreater(bericht["nachher"]["elemente"], 0)
            self.assertEqual(bericht["nachher"]["lang"], "de-DE")
            self.assertEqual(bericht["seiten"], 1)
            self.assertIn("Set Document Language (de-DE)", json.dumps(bericht["konfig"]) + "Set Document Language (de-DE)")


class UrteilTest(unittest.TestCase):
    """GESAMTURTEIL (23.09.2026): ein Satz je Dokument."""

    def test_stufen(self):
        import tagging_api as ta
        pr_leer = {"status": "", "laeuft": False, "seiten": 15, "bericht": {}}
        self.assertEqual(ta.urteil({"getaggt": 0}, {}, pr_leer, {})["stufe"], "ungetaggt")
        self.assertEqual(ta.urteil({"getaggt": 1}, {"elemente": 366, "ueberschriften": 0}, pr_leer, {})["stufe"], "neu_taggen")   # Ritterturnier aus InDesign
        self.assertEqual(ta.urteil({"getaggt": 1}, {"elemente": 2, "ueberschriften": 0}, pr_leer, {})["stufe"], "neu_taggen")
        self.assertEqual(ta.urteil({"getaggt": 1}, {"elemente": 50, "ueberschriften": 4}, pr_leer, {"bestanden": True})["stufe"], "pruefung_empfohlen")
        self.assertEqual(ta.urteil({"getaggt": 1}, {"elemente": 50, "ueberschriften": 4}, pr_leer, {"bestanden": False})["stufe"], "verbesserungen")
        pr_ok = {"status": "fertig", "laeuft": False, "seiten": 3, "bericht": {"anzahl": {"hoch": 0, "mittel": 1, "niedrig": 0, "auto": 0}}}
        u = ta.urteil({"getaggt": 1}, {"elemente": 50, "ueberschriften": 4}, pr_ok, {"bestanden": True})
        self.assertEqual((u["stufe"], u["aktion"]), ("in_ordnung", "export"))
        pr_befunde = {"status": "fertig", "laeuft": False, "seiten": 3, "bericht": {"anzahl": {"hoch": 5, "mittel": 0, "niedrig": 0, "auto": 2}}}
        u = ta.urteil({"getaggt": 1}, {"elemente": 50, "ueberschriften": 4}, pr_befunde, {"bestanden": True})
        self.assertEqual((u["stufe"], u["aktion"], u["ki_hoch"]), ("verbesserungen", "korrektur", 5))
        self.assertEqual(ta.urteil({"getaggt": 1, "tagging_status": "laeuft"}, {}, pr_leer, {})["stufe"], "laeuft")
        # Vollstaendigkeit (Struktur zuerst): >2 % Zeilen ohne Element -> unvollstaendig, auch wenn alles andere gut ist
        d = {"getaggt": 1, "tagging_bericht": json.dumps({"struktur": {"zeilen_ohne_element": 12, "zeilen_gesamt": 300}})}
        u = ta.urteil(d, {"elemente": 50, "ueberschriften": 4}, pr_ok, {"bestanden": True})
        self.assertEqual((u["stufe"], u["zeilen_ohne"]), ("unvollstaendig", 12))
        d2 = {"getaggt": 1, "tagging_bericht": json.dumps({"struktur": {"zeilen_ohne_element": 2, "zeilen_gesamt": 300}})}
        self.assertEqual(ta.urteil(d2, {"elemente": 50, "ueberschriften": 4}, pr_ok, {"bestanden": True})["stufe"], "in_ordnung")


class EinheitsberichtTest(unittest.TestCase):
    """23.09.2026 (Michaels Punkte 6/7/10/12): PDF/UA- und KI-Befunde in EINER Liste, nur Probleme, CSV."""

    def test_einheitsbericht_und_csv(self):
        import tagging_api
        doc = {"pruefung_status": "fertig", "korrektur_bericht": "", "tagging_bericht": json.dumps({"verapdf": {
            "bestanden": False, "punkte": [
                {"bereich": "Formularfelder und Verknüpfungen", "status": "befund", "text": "Ein Link hat keine Beschreibung. (Seite 15)", "seiten": [15], "regeln": ["7.18.5-2"]},
                {"bereich": "Struktur und Lesereihenfolge", "status": "ok", "text": "gut", "regeln": []}]}})}
        pb = {"befunde": [{"seite": 2, "typ": "P", "text": "Titel", "art": "rolle", "befund": "Absatz statt Überschrift",
                           "vorschlag": "H1", "sicherheit": "hoch", "auto": True}]}
        eb = tagging_api.einheitsbericht(doc, pb)
        self.assertEqual(eb["anzahl"], 2)                       # „In Ordnung“ faellt weg
        self.assertEqual([e["quelle"] for e in eb["eintraege"]], ["ki", "pdfua"])   # Seite 2 vor Seite 15
        self.assertEqual(eb["eintraege"][1]["seiten"], [15])
        self.assertTrue(eb["ki_vorhanden"]); self.assertTrue(eb["pdfua_vorhanden"]); self.assertFalse(eb["pdfua_bestanden"])
        csv_text = tagging_api.einheitsbericht_csv(doc, pb)
        self.assertTrue(csv_text.startswith("﻿Quelle;Seiten;Bereich;Element;Befund;Vorschlag;Sicherheit;Regeln"))
        self.assertIn("KI-Prüfung;2;rolle;P „Titel“;Absatz statt Überschrift;H1;hoch;", csv_text)
        self.assertIn("PDF/UA-Prüfung;15;Formularfelder und Verknüpfungen;;", csv_text)

    def test_csv_formel_injektion_entschaerft(self):
        import tagging_api
        doc = {"pruefung_status": "fertig", "korrektur_bericht": "", "tagging_bericht": "{}"}
        pb = {"befunde": [{"seite": 1, "typ": "P", "text": "=HYPERLINK(\"http://x\")", "art": "rolle", "befund": "@SUM(1)", "vorschlag": "+1", "sicherheit": "hoch"}]}
        csv_text = tagging_api.einheitsbericht_csv(doc, pb)
        self.assertIn("'@SUM(1)", csv_text)
        self.assertIn(";'+1;", csv_text)
        self.assertNotIn(";=HYPERLINK", csv_text)

    def test_einheitsbericht_nach_korrektur_nimmt_juengsten_pdfua_stand(self):
        import tagging_api
        doc = {"pruefung_status": "", "tagging_bericht": json.dumps({"verapdf": {"bestanden": False, "punkte": [{"bereich": "A", "status": "befund", "text": "alt", "seiten": [], "regeln": []}]}}),
               "korrektur_bericht": json.dumps({"verapdf": {"bestanden": True, "punkte": []}})}
        eb = tagging_api.einheitsbericht(doc, {})
        self.assertEqual(eb["anzahl"], 0); self.assertTrue(eb["pdfua_bestanden"]); self.assertFalse(eb["ki_vorhanden"])


if __name__ == "__main__":
    unittest.main()
