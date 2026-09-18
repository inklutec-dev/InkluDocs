"""Unit-Tests Uebersetzen-Werkzeug, Kern (18.09.2026): Segmentierer, Marken, Rueckschreiber,
Strukturvergleich, Ersatzweg — OHNE Modell (Modellaufruf wird nachgestellt).

Aufruf im Container:
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_uebersetzung_kern.py -v

Fixtures: tests/fixtures/testdokument_inkludocs.docx (fiktiv) und die von Word erzeugten
Dateien des LibreOffice-Korpus (word_einfach, word_vml_bild, word_textfeld_bild, ...).
"""
import os
import re
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

from lxml import etree  # noqa: E402

import uebersetzung as ue  # noqa: E402
from docx_processor import NS, _safe_parser  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
ALLE = sorted(f for f in os.listdir(FIX) if f.endswith(".docx")) if os.path.isdir(FIX) else []
HAUPT = os.path.join(FIX, "testdokument_inkludocs.docx")
W = NS["w"]


def _texte(docx_path: str, part: str = "word/document.xml") -> list[str]:
    with zipfile.ZipFile(docx_path) as zf:
        root = etree.fromstring(zf.read(part), _safe_parser)
    out = []
    for p in root.iter(f"{{{W}}}p"):
        t = "".join((x.text or "") for x in p.iter(f"{{{W}}}t") if ue._naechster_p(x) is p)
        if t.strip():
            out.append(t)
    return out


def _fake_modell(vorne: str = "EN:", drop_marke: bool = False):
    """Nachgestellter Modellaufruf: jedes markierte Stueck bekommt ein Praefix, Marken bleiben.
    drop_marke=True: laesst im ersten Segment die letzte Marke weg (Fehlerfall)."""
    def aufruf(prompt: str, max_tokens: int):
        segs = []
        block = prompt.split("===DOKUMENT===")[1]
        for m in re.finditer(r"^S(\d+) \([^\n]*\):\n(.*?)(?=\n\nS\d+ \(|\n\n?$)", block, re.S | re.M):
            nr, text = int(m.group(1)), m.group(2)
            neu = re.sub(r"\[\[(\d+)\]\](.*?)\[\[/\1\]\]",
                         lambda mm: f"[[{mm.group(1)}]]{vorne}{mm.group(2)}[[/{mm.group(1)}]]", text, flags=re.S)
            if drop_marke and not segs:
                neu = re.sub(r"\[\[(\d+)\]\](.*?)\[\[/\1\]\]\s*$", r"\2", neu, count=1, flags=re.S)
            segs.append(ue.SegmentUebersetzung(id=nr, text=neu))
        return ue.UebersetzungBatchOutput(segmente=segs)
    return aufruf


class TestSegmentierer(unittest.TestCase):
    def test_hauptfixture_liest_absaetze(self):
        s = ue.segmentiere_docx(HAUPT)
        self.assertGreater(s.absaetze, 5)
        self.assertGreater(s.woerter, 20)
        arten = {seg.art for seg in s.segmente}
        self.assertIn("absatz", arten)
        # jeder uebersetzbare Absatz hat mindestens eine Marke, Zahlen-Absaetze keine
        for seg in s.segmente:
            if seg.uebersetzbar:
                self.assertTrue(seg.marken, seg.anker)
                self.assertTrue(ue._BUCHSTABE_RE.search(seg.text))

    def test_ueberschriftenpfad_und_abschnitt(self):
        s = ue.segmentiere_docx(HAUPT)
        mit_kontext = [seg for seg in s.segmente if seg.art == "absatz" and seg.kontext]
        self.assertTrue(mit_kontext)
        self.assertTrue(all(seg.abschnitt >= 1 for seg in s.segmente))

    def test_marken_text_roundtrip(self):
        s = ue.segmentiere_docx(HAUPT)
        for seg in s.segmente:
            if seg.art != "absatz":
                continue
            markiert = ue.text_mit_marken(seg)
            zerlegt = ue.marken_zerlegen(markiert, seg)
            self.assertIsNotNone(zerlegt, seg.anker)
            for i in seg.marken:
                self.assertEqual(zerlegt[i], seg.stuecke[i])

    def test_marken_fehlt_oder_doppelt(self):
        seg = ue.Segment(anker="x|p1", part="x", art="absatz", stuecke=["Hallo ", "Welt", "!"], marken=[0, 1, 2])
        self.assertIsNone(ue.marken_zerlegen("[[1]]Hi [[/1]][[2]]world[[/2]]", seg))          # 3 fehlt
        self.assertIsNone(ue.marken_zerlegen("[[1]]Hi [[/1]][[1]]x[[/1]][[2]]w[[/2]][[3]]![[/3]]", seg))   # 1 doppelt
        self.assertIsNone(ue.marken_zerlegen("Extra [[1]]Hi [[/1]][[2]]w[[/2]][[3]]![[/3]]", seg))   # Text ausserhalb
        ok = ue.marken_zerlegen("[[2]]world[[/2]][[1]]Hi [[/1]][[3]]![[/3]]", seg)              # Reihenfolge egal
        self.assertEqual(ok, {0: "Hi ", 1: "world", 2: "!"})

    def test_whitespace_angleichen(self):
        seg = ue.Segment(anker="x|p1", part="x", art="absatz", stuecke=["Bitte ", "beachten", " Sie."], marken=[0, 1, 2])
        out = ue._whitespace_angleichen(seg, {0: "Please", 1: "note", 2: "this."})
        self.assertEqual(out[0], "Please ")
        self.assertEqual(out[2], " this.")

    def test_ersatzweg(self):
        seg = ue.Segment(anker="x|p1", part="x", art="absatz", stuecke=["Bitte ", "beachten", " Sie."], marken=[0, 1, 2])
        out = ue.ersatz_zusammenlegen(seg, "[[1]]Please note this.")
        self.assertEqual(out, {0: "Please note this. ", 1: "", 2: ""})

    def test_credits(self):
        self.assertEqual(ue.credits_fuer(0), 0)
        self.assertEqual(ue.credits_fuer(1), 1)
        self.assertEqual(ue.credits_fuer(100), 1)
        self.assertEqual(ue.credits_fuer(101), 2)
        self.assertEqual(ue.credits_fuer(3000), 30)

    def test_korpus_liest_alles(self):
        for f in ALLE:
            s = ue.segmentiere_docx(os.path.join(FIX, f))
            self.assertIsInstance(s.segmente, list, f)


class TestRueckschreiber(unittest.TestCase):
    def _uebersetze_datei(self, quelle: str, ziel_dir: str, modell=None) -> tuple[str, ue.Segmentierung, list]:
        s = ue.segmentiere_docx(quelle)
        nummeriert = [(i + 1, seg) for i, seg in enumerate(s.segmente) if seg.uebersetzbar]
        ergebnisse = []
        for batch in ue._batches(nummeriert):
            ergebnisse.extend(ue.uebersetze_batch(batch, "en", s.quellsprache, s.titel, modell_aufruf=modell or _fake_modell()))
        ziele = {}
        by_nr = {nr: seg for nr, seg in nummeriert}
        for e in ergebnisse:
            if e.ziel_stuecke:
                ziele[by_nr[e.nr].anker] = e.ziel_stuecke
        out = os.path.join(ziel_dir, "out.docx")
        erg = ue.schreibe_uebersetzung(quelle, out, ziele, sprache_ziel="en")
        return out, s, ergebnisse, erg

    def test_roundtrip_struktur_identisch(self):
        with tempfile.TemporaryDirectory() as d:
            out, s, ergebnisse, erg = self._uebersetze_datei(HAUPT, d)
            self.assertFalse(erg.nicht_gefunden, erg.nicht_gefunden)
            self.assertEqual(ue.strukturvergleich(HAUPT, out), [])
            # jeder Text traegt das Praefix, alle Absaetze uebersetzt
            texte = _texte(out)
            self.assertTrue(texte)
            for t in texte:
                if ue._BUCHSTABE_RE.search(t):
                    self.assertIn("EN:", t, t)
            self.assertTrue(all(e.status == "fertig" for e in ergebnisse), [e for e in ergebnisse if e.status != "fertig"])

    def test_sprache_gesetzt(self):
        with tempfile.TemporaryDirectory() as d:
            out, *_ = self._uebersetze_datei(HAUPT, d)
            with zipfile.ZipFile(out) as zf:
                st = etree.fromstring(zf.read("word/styles.xml"), _safe_parser)
                langs = {l.get(f"{{{W}}}val") for l in st.iter(f"{{{W}}}lang")}
                self.assertEqual(langs, {"en-US"})
                core = etree.fromstring(zf.read("docProps/core.xml"), _safe_parser)
                self.assertEqual(core.find("dc:language", NS).text, "en-US")

    def test_unberuehrte_teile_byteidentisch(self):
        with tempfile.TemporaryDirectory() as d:
            out, s, _e, erg = self._uebersetze_datei(HAUPT, d)
            with zipfile.ZipFile(HAUPT) as a, zipfile.ZipFile(out) as b:
                for n in a.namelist():
                    if n in erg.geaenderte_teile:
                        continue
                    self.assertEqual(a.read(n), b.read(n), n)

    def test_alt_texte_mituebersetzt(self):
        with tempfile.TemporaryDirectory() as d:
            out, s, _e, _erg = self._uebersetze_datei(HAUPT, d)
            alts = [seg for seg in s.segmente if seg.art == "alt"]
            if not alts:
                self.skipTest("Fixture ohne Alt-Text")
            with zipfile.ZipFile(out) as zf:
                root = etree.fromstring(zf.read("word/document.xml"), _safe_parser)
            descrs = [d_.get("descr") for d_ in root.iter(f"{{{NS['wp']}}}docPr") if d_.get("descr")]
            self.assertTrue(any(x.startswith("EN:") for x in descrs), descrs)

    def test_ersatzweg_wird_gemeldet(self):
        with tempfile.TemporaryDirectory() as d:
            out, s, ergebnisse, erg = self._uebersetze_datei(HAUPT, d, modell=_fake_modell(drop_marke=True))
            self.assertEqual(ue.strukturvergleich(HAUPT, out), [])   # Struktur bleibt auch im Ersatzweg
            stati = {e.status for e in ergebnisse}
            self.assertTrue(stati <= {"fertig", "zusammengelegt"}, stati)

    def test_korpus_roundtrip(self):
        for f in ALLE:
            with tempfile.TemporaryDirectory() as d:
                quelle = os.path.join(FIX, f)
                out, s, _e, erg = self._uebersetze_datei(quelle, d)
                self.assertEqual(ue.strukturvergleich(quelle, out), [], f)
                self.assertFalse(erg.nicht_gefunden, (f, erg.nicht_gefunden))
                with zipfile.ZipFile(out) as zf:
                    for n in zf.namelist():
                        if n.endswith(".xml"):
                            etree.fromstring(zf.read(n), _safe_parser)   # wohlgeformt

    def test_idempotent(self):
        """Zweimal schreiben (gleiche Ziele) = gleiches Ergebnis."""
        with tempfile.TemporaryDirectory() as d:
            out1, *_ = self._uebersetze_datei(HAUPT, d)
            out2 = os.path.join(d, "out2.docx")
            s = ue.segmentiere_docx(out1)
            ue.schreibe_uebersetzung(out1, out2, {}, sprache_ziel="en")
            self.assertEqual(ue.strukturvergleich(out1, out2), [])
            self.assertEqual(_texte(out1), _texte(out2))


class TestAbwehr(unittest.TestCase):
    def test_kein_zip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.docx")
            open(p, "wb").write(b"kein zip")
            with self.assertRaises(Exception):
                ue.segmentiere_docx(p)

    def test_xxe_bleibt_unaufgeloest(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "xxe.docx")
            with zipfile.ZipFile(HAUPT) as a, zipfile.ZipFile(p, "w") as b:
                for n in a.namelist():
                    daten = a.read(n)
                    if n == "word/document.xml":
                        daten = daten.replace(b"<w:document", b"<!DOCTYPE x [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><w:document", 1)
                        daten = daten.replace(b"</w:body>", b"<w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body>", 1)
                    b.writestr(n, daten)
            s = ue.segmentiere_docx(p)
            self.assertFalse(any("root:" in seg.text for seg in s.segmente))


if __name__ == "__main__":
    unittest.main()
