"""Alt-Texte beim Tagging: „Feld = Datei“ (Steve 09.10.2026).
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_alttext_feld_datei.py -v

In die getaggte Datei kommt genau das an Alt-Texten, was in den Feldern von InkluDocs steht — beim „Testweise taggen“ wie
beim bezahlten Tagging, und das Feld behaelt nach dem Neu-Taggen seinen Stand. Fall-Matrix je Bild:
  B1 mitgebracht, nie angefasst | B2 von Hand geaendert | B3 per KI generiert | B4 bewusst geleert |
  B5 vom Autor als dekorativ markiert (Word: original_alt „dekorativ“; PDF: Artefakt, kein Feld) |
  B6 ohne Alt-Text, nie bearbeitet | B7 in InkluDocs als dekorativ markiert | NEU: von PDFix neu erkanntes Bild ohne Feld.
Dazu: die Abrechnung beim Herunterladen zaehlt vor und nach dem Tagging dieselben Bilder als bearbeitet
(main._alt_text_bearbeitet), mitgebrachte unveraenderte Texte nie.
Der Lauf-Test (LaufTest) braucht das PDFix-SDK und laeuft im Testmodus (ohne Tagging-Lizenz); ohne SDK uebersprungen.
Testdatei: tests/fixtures/gartenfest_feld_datei.pdf (fiktiv, make_gartenfest.py + LibreOffice-Umwandler).
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="feld-datei-")
os.environ["INKLUDOCS_DB"] = os.path.join(_TMP, "test.db")

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import tagging_api  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "gartenfest_feld_datei.pdf")

MITGEBRACHT = {"B1": "Gelbe Sonnenblumen vor blauem Himmel am Gartenzaun des Vereinsgartens.",
               "B2": "Kuchenbuffet mit zwölf Blechkuchen auf einem langen Tisch.",
               "B3": "Grüne Wimpelketten über der Festwiese.",
               "B4": "Plakat zum Gartenfest mit der Telefonnummer der Vorsitzenden.",
               "B6": "",
               "B7": "Logo des Nachbarschaftsvereins Musterstadt."}
HAND = "Von Hand: Zwölf Blechkuchen, darunter Apfel und Streusel, auf einem Biertisch."
KI = "Grüne und weiße diagonale Streifen."
# Was nach dem Tagging im Feld (Anzeige) und in der Datei (/Alt) stehen muss; None = kein /Alt-Eintrag, "" = /Alt ""
ERWARTET_FELD = {"B1": MITGEBRACHT["B1"], "B2": HAND, "B3": KI, "B4": "", "B6": "", "B7": "dekorativ", "NEU": ""}
ERWARTET_DATEI = {"B1": MITGEBRACHT["B1"], "B2": HAND, "B3": KI, "B4": None, "B6": None, "B7": "", "NEU": None}


def _main():
    try:
        import main
    except Exception as e:  # noqa: BLE001 — ausserhalb des Containers
        raise unittest.SkipTest(f"main nicht ladbar: {e}")
    return main


def _alte_zeile(i, seite, bild, pfad=""):
    """Bildzeile vor dem Tagging (wie nach dem Upload: PDFix-Weg, bbox = Bildmasse), Fall nach Bildname."""
    z = dict(tagging_api._NEUE_ZEILE)
    z.update({"id": 100 + i, "image_index": i, "page_number": seite, "image_path": pfad, "width": 0, "height": 0,
              "bbox_x0": 0, "bbox_y0": 0, "bbox_x1": 0, "bbox_y1": 0, "xref": -i, "is_vector": 1,
              "original_alt": MITGEBRACHT.get(bild, ""), "_bild": bild})
    if bild == "B2":
        z.update({"alt_text_edited": HAND, "status": "done"})
    elif bild == "B3":
        z.update({"alt_text": KI, "status": "done", "image_type": "foto", "konfidenz": "hoch"})
    elif bild == "B4":
        z.update({"alt_text_edited": ""})                         # bewusst geleert, nie generiert (status pending)
    elif bild == "B7":
        z.update({"alt_text_edited": "dekorativ"})
    return z


def _feld(z):
    """Anzeige im Feld — dieselbe Regel wie main._display_alt_text / app.html anzeigeAltText."""
    if z.get("alt_text_edited") is not None:
        return z["alt_text_edited"]
    return z.get("alt_text") or z.get("original_alt") or ""


class Zuordnung(unittest.TestCase):
    """felder_zuordnen ohne PDFix: neue Zeilen tragen die PDFix-Bildunterschrift als original_alt (wie nach dem Lauf)."""

    def _neue(self, seiten_bilder):
        bilder = [{"page_number": s, "image_path": "", "bbox": (0, 0, 0, 0), "width": 0, "height": 0,
                   "original_alt": f"Abbildung {i}: Unterschrift mit Sternc*en", "source": "pdfix", "xref": -i}
                  for i, (s, _b) in enumerate(seiten_bilder, start=1)]
        return tagging_api._als_zeilen(bilder)

    def test_matrix_je_seite_eindeutig(self):
        # je Seite genau ein altes und ein neues Bild -> Zuordnung ueber Rueckfall 3, ohne Bild-Hash
        reihen = [(1, "B1"), (2, "B2"), (3, "B3"), (4, "B4"), (5, "B6"), (6, "B7")]
        alte = [_alte_zeile(i, s, b) for i, (s, b) in enumerate(reihen, start=1)]
        neue = self._neue(reihen + [(7, "NEU")])
        n = tagging_api.felder_zuordnen(neue, alte, baum_neu=True)
        self.assertEqual(n, 6)
        for z, (_s, b) in zip(neue, reihen + [(7, "NEU")]):
            self.assertEqual(_feld(z), ERWARTET_FELD[b], b)
            self.assertNotIn("*", _feld(z), b)
        # original_alt bleibt der mitgebrachte Text (nicht die Unterschrift von PDFix), das neue Bild hat keinen
        self.assertEqual([z["original_alt"] for z in neue], [MITGEBRACHT[b] for _s, b in reihen] + [""])

    def test_ohne_neuen_baum_bleibt_pdfix_text_der_kundendatei(self):
        """PDFix hat vorhandene Tags behalten: dann stammt das /Alt eines nicht zugeordneten Bildes aus der Kundendatei."""
        neue = self._neue([(1, "X")])
        tagging_api.felder_zuordnen(neue, [], baum_neu=False)
        self.assertEqual(neue[0]["original_alt"], "Abbildung 1: Unterschrift mit Sternc*en")
        neue = self._neue([(1, "X")])
        tagging_api.felder_zuordnen(neue, [], baum_neu=True)
        self.assertEqual(neue[0]["original_alt"], "")

    def test_alle_alten_bilder_sind_kandidaten(self):
        """Vorher nur Bilder mit Text oder status done — ein bewusst geleertes, nie generiertes Bild ging verloren."""
        alte = [_alte_zeile(1, 1, "B4")]
        neue = self._neue([(1, "B4")])
        self.assertEqual(tagging_api.felder_zuordnen(neue, alte), 1)
        self.assertEqual(neue[0]["alt_text_edited"], "")
        self.assertEqual(_feld(neue[0]), "")

    def test_rest_je_seite_logo_des_testmodus(self):
        """Das Logo des Testmodus verdeckt einen Teil eines Bildes: sein Rendering liegt ueber HASH_TOLERANZ, aber unter
        HASH_TOLERANZ_REST — es bekommt trotzdem sein Feld. Ein fremdes Bild (neu von PDFix erkannt) bekommt keins."""
        from PIL import Image, ImageDraw
        t = tempfile.mkdtemp(prefix="rest-", dir=_TMP)

        def bild(name, art, logo=False):
            im = Image.new("RGB", (313, 195), "white")
            d = ImageDraw.Draw(im)
            if art == "verlauf":
                for x in range(313):
                    d.line((x, 0, x, 195), fill=(int(255 * x / 313), 120, 255 - int(255 * x / 313)))
                for i, x in enumerate(range(30, 313, 60)):
                    d.ellipse((x - 20, 100 - 15 * (i % 2), x + 20, 140 - 15 * (i % 2)), fill=(250, 200, 20))
            elif art == "raster":
                for r in range(3):
                    for c in range(4):
                        d.rectangle((10 + c * 75, 10 + r * 60, 70 + c * 75, 60 + r * 60), fill=(150, 80, 30))
            else:   # Schachbrett: ein fremdes Bild, das PDFix neu als Figure erkannt hat
                for r in range(0, 195, 20):
                    for c in range(0, 313, 20):
                        if (r // 20 + c // 20) % 2:
                            d.rectangle((c, r, c + 19, r + 19), fill=(0, 0, 0))
            if logo:
                d.rectangle((0, 60, 313, 140), fill=(40, 40, 40))   # wie das Logo des Testmodus quer ueber dem Bild
            p = os.path.join(t, name)
            im.save(p)
            return p

        alt1, alt2 = bild("a1.png", "verlauf"), bild("a2.png", "raster")
        neu1, neu2, neu3 = bild("n1.png", "verlauf", logo=True), bild("n2.png", "raster"), bild("n3.png", "schach")
        h = tagging_api._dhash
        ab = tagging_api._hamming(h(neu1), h(alt1))
        self.assertTrue(tagging_api.HASH_TOLERANZ < ab <= tagging_api.HASH_TOLERANZ_REST, ab)
        for fremd in (neu3,):
            self.assertGreater(tagging_api._hamming(h(fremd), h(alt1)), tagging_api.HASH_TOLERANZ_REST)
            self.assertGreater(tagging_api._hamming(h(fremd), h(alt2)), tagging_api.HASH_TOLERANZ_REST)
        alte = [_alte_zeile(1, 1, "B1", alt1), _alte_zeile(2, 1, "B2", alt2)]
        neue = tagging_api._als_zeilen([{"page_number": 1, "image_path": p, "bbox": (0, 0, 313, 195), "width": 313,
                                         "height": 195, "original_alt": "Abbildung", "source": "pdfix"} for p in (neu1, neu3, neu2)])
        self.assertEqual(tagging_api.felder_zuordnen(neue, alte), 2)
        self.assertEqual([_feld(z) for z in neue], [MITGEBRACHT["B1"], "", HAND])

    def test_baum_von_pdfix(self):
        b = tagging_api.baum_von_pdfix
        self.assertTrue(b({"vorher": {"elemente": 0}}))                          # ungetaggte Quelle
        self.assertTrue(b({"vorher": {"elemente": 40}, "tags_ersetzt": True}))   # Neu taggen mit Ersetzen
        self.assertTrue(b({"vorher": {"elemente": 40}, "weg": "struktur"}))
        self.assertFalse(b({"vorher": {"elemente": 40}, "tags_ersetzt": False}))  # „Preserve Existing Tags“

    def test_db_variante_gleich(self):
        """alt_texte_uebernehmen (Datenbank) fuehrt dieselbe Zuordnung aus."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        spalten = ["id INTEGER PRIMARY KEY", "document_id INTEGER", "page_number INTEGER", "image_index INTEGER",
                   "bbox_x0 REAL", "bbox_y0 REAL", "bbox_x1 REAL", "bbox_y1 REAL", "width INTEGER", "height INTEGER",
                   "image_path TEXT"] + [f"{s} TEXT" for s in tagging_api._UEBERNAHME_SPALTEN]
        conn.execute("CREATE TABLE images (" + ", ".join(spalten) + ")")
        for i in (1, 2):
            conn.execute("INSERT INTO images (document_id, page_number, image_index, bbox_x0, bbox_y0, bbox_x1, bbox_y1, width, "
                         "height, image_path, original_alt, status) VALUES (7, ?, ?, 0, 0, 0, 0, 0, 0, '', 'Unterschrift', 'pending')", (i, i))
        n = tagging_api.alt_texte_uebernehmen(conn, 7, [_alte_zeile(1, 1, "B1"), _alte_zeile(2, 2, "B4")], baum_neu=True)
        self.assertEqual(n, 2)
        rows = [dict(r) for r in conn.execute("SELECT * FROM images ORDER BY image_index")]
        self.assertEqual(rows[0]["original_alt"], MITGEBRACHT["B1"])
        self.assertEqual((rows[1]["alt_text_edited"], rows[1]["original_alt"]), ("", MITGEBRACHT["B4"]))


class TextUndAbrechnung(unittest.TestCase):
    """main._tagging_alt_text (was in die Figure kommt) und main._alt_text_bearbeitet (was das Herunterladen berechnet)."""

    @classmethod
    def setUpClass(cls):
        cls.m = _main()

    def test_text_je_fall(self):
        reihen = [(1, "B1"), (2, "B2"), (3, "B3"), (4, "B4"), (5, "B6"), (6, "B7")]
        for i, (s, b) in enumerate(reihen, start=1):
            z = _alte_zeile(i, s, b)
            erwartet = {"B1": MITGEBRACHT["B1"], "B2": HAND, "B3": KI, "B4": "", "B6": "", "B7": "dekorativ"}[b]
            self.assertEqual(self.m._tagging_alt_text(z), erwartet, b)
        # Word: vom Autor als dekorativ gekennzeichnet -> wie im Export
        z = dict(tagging_api._NEUE_ZEILE, original_alt="dekorativ", image_type="dekorativ", status="done")
        self.assertEqual(self.m._tagging_alt_text(z), "dekorativ")
        # technischer Fehlertext: der Export laesst die Figure unangetastet -> nach PDFix der Text der Kundendatei
        z = dict(tagging_api._NEUE_ZEILE, alt_text="Fehler bei der Analyse: Zeitueberschreitung", original_alt="Logo")
        self.assertIsNone(self.m._exportable_alt_text(z))
        self.assertEqual(self.m._tagging_alt_text(z), "Logo")
        z["original_alt"] = ""
        self.assertEqual(self.m._tagging_alt_text(z), "")

    def test_abrechnung_vor_und_nach_dem_tagging_gleich(self):
        """Steve: Der Preis beim Herunterladen zaehlt nur, was in InkluDocs bearbeitet wurde. Ein mitgebrachter,
        unveraenderter Alt-Text ist nicht unsere Arbeit — vor UND nach dem Neu-Taggen."""
        reihen = [(1, "B1"), (2, "B2"), (3, "B3"), (4, "B4"), (5, "B6"), (6, "B7")]
        alte = [_alte_zeile(i, s, b) for i, (s, b) in enumerate(reihen, start=1)]
        bilder = [{"page_number": s, "image_path": "", "bbox": (0, 0, 0, 0), "width": 0, "height": 0,
                   "original_alt": "Abbildung: Unterschrift", "source": "pdfix", "xref": -i}
                  for i, (s, _b) in enumerate(reihen + [(7, "NEU")], start=1)]
        neue = tagging_api._als_zeilen(bilder)
        tagging_api.felder_zuordnen(neue, alte, baum_neu=True)
        b = self.m._alt_text_bearbeitet
        vorher = {z["_bild"]: b(z) for z in alte}
        nachher = {bild: b(z) for z, (_s, bild) in zip(neue, reihen + [(7, "NEU")])}
        self.assertEqual(vorher, {"B1": False, "B2": True, "B3": True, "B4": True, "B6": False, "B7": True})
        self.assertEqual({k: v for k, v in nachher.items() if k != "NEU"}, vorher)
        self.assertFalse(nachher["NEU"])
        self.assertFalse(b(dict(tagging_api._NEUE_ZEILE, original_alt="dekorativ", image_type="dekorativ", status="done")))


def _figures_alt(pfad):
    """Figure-Elemente in Reihenfolge des Strukturbaums mit Seite (wie Heines Zaehlung): /Alt oder None."""
    import pikepdf
    out = []
    with pikepdf.open(pfad) as pdf:
        rm = pdf.Root.StructTreeRoot.get("/RoleMap") or {}
        rolle = {str(k): str(v) for k, v in rm.items()} if rm else {}
        gesehen = set()

        def typ(el):
            t = str(el.get("/S", ""))
            return rolle.get(t, t)

        def hat_seite(el, pg):
            if pg is not None or el.get("/Pg") is not None:
                return True
            k = el.get("/K")
            for x in (list(k) if isinstance(k, pikepdf.Array) else [k]):
                if isinstance(x, pikepdf.Dictionary) and x.get("/Pg") is not None:
                    return True
            return False

        def lauf(el, pg):
            if not isinstance(el, pikepdf.Dictionary):
                return
            if el.objgen != (0, 0):
                if el.objgen in gesehen:
                    return
                gesehen.add(el.objgen)
            pg = el.get("/Pg") or pg
            if typ(el) == "/Figure" and hat_seite(el, pg):
                out.append(str(el["/Alt"]) if "/Alt" in el else None)
            k = el.get("/K")
            for x in (list(k) if isinstance(k, pikepdf.Array) else [k]):
                lauf(x, pg)

        lauf(pdf.Root.StructTreeRoot, None)
    return out


class LaufTest(unittest.TestCase):
    """Echter PDFix-Lauf im Testmodus auf der Gartenfest-PDF (LibreOffice, 6 Figures mit Alt-Texten bzw. ohne, ein
    Schmuckbild als Artefakt) mit „Neu taggen“ (Tags ersetzen) — danach traegt jede Figure genau ihren Feldtext, ohne
    Sternchen, und die Felder behalten ihren Stand."""

    @classmethod
    def setUpClass(cls):
        try:
            import pdfixsdk  # noqa: F401
        except Exception:  # noqa: BLE001
            raise unittest.SkipTest("kein PDFix-SDK")
        if not os.path.isfile(FIXTURE):
            raise unittest.SkipTest("Testdatei fehlt")
        cls.m = _main()
        import pdf_tagging
        import pdf_processor
        cls.pt = pdf_tagging
        cls.tmp = tempfile.mkdtemp(prefix="feld-datei-lauf-")
        cls.alte_bilder = os.path.join(cls.tmp, "alt")
        os.makedirs(cls.alte_bilder)
        with mock.patch.dict(os.environ, {"PDFIX_ENABLED": "true"}):
            alt = pdf_processor.extract_images_from_pdf(FIXTURE, cls.alte_bilder, 1)
        # Reihenfolge im LibreOffice-Baum: B1, B2, B3, B4, B6, B7 (B5 ist ein Artefakt)
        cls.namen = ["B1", "B2", "B3", "B4", "B6", "B7"]
        assert len(alt) == 6, len(alt)
        cls.alte = []
        for i, (img, name) in enumerate(zip(alt, cls.namen), start=1):
            z = _alte_zeile(i, img["page_number"], name, img["image_path"])
            assert (z["original_alt"] or "") == (img.get("original_alt") or ""), (name, img.get("original_alt"))
            cls.alte.append(z)
        cls.getaggt = os.path.join(cls.tmp, "getaggt.pdf")
        with mock.patch.dict(os.environ, {"PDFIX_TAGGING_LIZENZ": "off"}):
            cls.roh = pdf_tagging.taggen(FIXTURE, cls.getaggt, "de", arbeitsordner=cls.tmp, testmodus=True, tags_ersetzen=True)
        cls.d_alt = tagging_api._d
        tagging_api._d = mock.Mock(alt_texte_einsetzen=cls.m._alt_texte_einsetzen, tagging_alt_text=cls.m._tagging_alt_text)

    @classmethod
    def tearDownClass(cls):
        tagging_api._d = cls.d_alt
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_feld_gleich_datei(self):
        import pdf_processor
        vorher = _figures_alt(self.getaggt)
        self.assertTrue(any(a and a.startswith("Abbildung") for a in vorher), vorher)   # PDFix: Bildunterschrift als /Alt
        neue_bilder = os.path.join(self.tmp, "neu")
        os.makedirs(neue_bilder)
        with mock.patch.dict(os.environ, {"PDFIX_ENABLED": "true"}):
            images = pdf_processor.extract_images_from_pdf(self.getaggt, neue_bilder, 1)
        aus = os.path.join(self.tmp, "feld.pdf")
        zeilen, info = tagging_api.feld_gleich_datei(self.getaggt, aus, images, self.alte, tagging_api.baum_von_pdfix(self.roh))
        self.assertEqual(info["methode"], "pdfix")
        self.assertEqual(info["zugeordnet"], 6, info)
        # Felder: jedes alte Bild genau einmal mit seinem Feldtext, das von PDFix neu erkannte Schmuckbild leer
        felder = sorted(_feld(z) for z in zeilen)
        erwartet = sorted([ERWARTET_FELD[b] for b in self.namen] + [""] * (len(zeilen) - 6))
        self.assertEqual(felder, erwartet)
        # Datei = Feld, Figure fuer Figure (laufende Nummer wie im Export)
        datei = _figures_alt(aus)
        self.assertEqual(len(datei), len(zeilen))
        for z, alt in zip(zeilen, datei):
            text = self.m._tagging_alt_text(z)
            soll = None if text == "" else ("" if text == "dekorativ" else text)
            self.assertEqual(alt, soll, (_feld(z), alt))
        self.assertFalse([a for a in datei if a and "*" in a], datei)
        # Testmodus-Vermerk bleibt (das Einsetzen taggt nicht)
        self.assertTrue(self.pt.tag_statistik(aus)["testmodus"])
        # Struktur unveraendert bis auf /Alt
        a, b = self.pt.tag_statistik(self.getaggt), self.pt.tag_statistik(aus)
        self.assertEqual(a["je_typ"], b["je_typ"])


if __name__ == "__main__":
    unittest.main()
