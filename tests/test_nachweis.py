"""„Fertig mit Nachweis“ (Steve 17.10.2026, backend/inkluagent/nachweis.py): am Ende zeigt der InkluAgent, dass das Ergebnis geprüft
ist — veraPDF, Alt-Texte, Anfang der Hörprobe mit „Anhören“.

Prüft: Nachweis aus der Ablage (Umwandeln: Urteil, Befunde, Alt-Texte, Hörprobe ohne die Sprachzeile; mehrere Dokumente; PDF
herunterladen ohne Hörprobe; „nicht geprüft“ ist nie „bestanden“; nur eigene Einträge; nie eine Ausnahme), aus der Prüfdatei,
Zeilen gekürzt und begrenzt; herausnehmen (nur bei Erfolg, nur Nachweise); Aufträge hängen ihn hinter die Datei des Schritts und geben
dem Modell nur das Urteil; das Gespräch nimmt ihn heraus, bevor das Modell das Ergebnis sieht; die drei Werkzeuge liefern ihn; die
Karte in der Seite (nur textContent, Sprache des Dokuments, eigene Pfade, „Anhören“, Urteil als Text).
Wegwerf-Datenbank, keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_nachweis.py
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="nachweis_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

from inkluagent import nachweis  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
_N = [0]

HOER = ["Dokumenttitel: Vereinsbrief (fiktiv)", "Sprache: Deutsch", "Überschrift Ebene 1: Einladung (fiktiv)",
        "Absatz: Liebe Mitglieder, wir laden euch ein (fiktiv).", "Grafik: Ein Haus mit rotem Dach (fiktiv)."] + \
       [f"Absatz: Zeile {i} (fiktiv)." for i in range(10)]


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def konto() -> int:
    _N[0] += 1
    return database.create_user(f"nachweis{_N[0]}@beispiel.invalid", "geheim-12345", "Test (fiktiv)")


def eintrag(uid: int, bericht, art="pdfua", bestanden=1) -> int:
    c = database.get_db()
    aid = c.execute("INSERT INTO ablage (user_id, project_id, projekt_name, dokument_name, art, dateiname, bericht, bestanden) "
                    "VALUES (?, 1, 'Vereinsbrief (fiktiv)', 'Einladung (fiktiv)', ?, 'brief.pdf', ?, ?)",
                    (uid, art, bericht if isinstance(bericht, str) else json.dumps(bericht), bestanden)).lastrowid
    c.commit()
    c.close()
    return aid


def dok(bestanden=True, befunde=0, bilder=3, alt=3, hoer=HOER, name="Einladung (fiktiv)"):
    punkte = [{"bereich": "Alt-Texte", "status": "ok", "text": "ok"}] + \
             [{"bereich": f"Bereich {i}", "status": "befund", "text": "Befund (fiktiv)"} for i in range(befunde)]
    return {"dokument": name, "bilder": bilder, "alt_texte": alt, "hoerprobe": hoer,
            "pruefung": {"bestanden": bestanden, "profil": "PDF/UA-1", "punkte": punkte}}


class AusDerAblage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")

    def setUp(self):
        self.uid = konto()

    def test_umwandeln_bestanden_mit_hoerprobe_ohne_sprachzeile(self):
        aid = eintrag(self.uid, [dok()])
        n = nachweis.aus_ausgabe(self.uid, aid, "de-DE")
        self.assertEqual((n["art"], n["quelle"], n["ausgabe_id"], n["bestanden"], n["befunde"], n["bilder"], n["alt_texte"]),
                         ("nachweis", "ablage", aid, True, 0, 3, 3))
        self.assertEqual(n["hoerprobe"][:3], [HOER[0], HOER[2], HOER[3]], "Titel bleibt, die Sprachzeile steht als lang an der Karte")
        self.assertEqual(len(n["hoerprobe"]), nachweis.HOERPROBE_ZEILEN, "nur der Anfang")
        self.assertEqual((n["lang"], n["ablage_url"], n["dokument"], n["hoerprobe_dokument"]),
                         ("de", f"/ablage#ausgabe-{aid}", "Einladung (fiktiv)", None))

    def test_mehrere_dokumente_und_befunde(self):
        aid = eintrag(self.uid, [dok(name="A (fiktiv)", hoer=[]), dok(False, 2, bilder=2, alt=1, name="B (fiktiv)")])
        n = nachweis.aus_ausgabe(self.uid, aid)
        self.assertEqual((n["bestanden"], n["befunde"], n["bilder"], n["alt_texte"], n["dokumente"]), (False, 2, 5, 4, 2))
        self.assertEqual(n["hoerprobe_dokument"], "B (fiktiv)", "die Hörprobe nennt ihr Dokument")

    def test_pdf_herunterladen_und_nicht_geprueft(self):
        aid = eintrag(self.uid, [{"dokument": "X (fiktiv)", "hoerprobe": [], "pruefung": {"bestanden": False, "punkte": [],
                                                                                       "nicht_geprueft": True}}], art="pdf")
        n = nachweis.aus_ausgabe(self.uid, aid)
        self.assertIsNone(n["bestanden"], "nicht geprüft ist nie „bestanden“ und nie „nicht bestanden“")
        self.assertEqual((n["hoerprobe"], n["bilder"], n["alt_texte"]), ([], None, None))
        aid = eintrag(self.uid, [{"dokument": "X (fiktiv)", "pruefung": {}}], art="pdf")
        self.assertIsNone(nachweis.aus_ausgabe(self.uid, aid)["bestanden"])

    def test_nur_eigene_und_nie_eine_ausnahme(self):
        aid = eintrag(self.uid, [dok()])
        self.assertIsNone(nachweis.aus_ausgabe(konto(), aid), "fremder Eintrag")
        self.assertIsNone(nachweis.aus_ausgabe(self.uid, eintrag(self.uid, "kaputt{")))
        self.assertIsNone(nachweis.aus_ausgabe(self.uid, eintrag(self.uid, [])), "nichts Geprüftes")
        self.assertIsNone(nachweis.aus_ausgabe(self.uid, "abc"))

    def test_zeilen_gekuerzt_und_bereinigt(self):
        lang = "Absatz: " + "sehr lang " * 80
        aid = eintrag(self.uid, [dok(hoer=["Titel", "Sprache", "  zwei\n Zeilen  ", "", lang])])
        z = nachweis.aus_ausgabe(self.uid, aid)["hoerprobe"]
        self.assertEqual(z[1], "zwei Zeilen")
        self.assertEqual(len(z[2]), nachweis.ZEILE_MAX)
        self.assertTrue(z[2].endswith("…"))


class AusDerPruefdatei(unittest.TestCase):
    def daten(self, **pd):
        return {"bilder": 4, "sprache": "en-GB", "pruefdatei": dict({"aktuell": True, "verapdf_moeglich": True, "bestanden": True,
                                                                     "anzahl_probleme": 0}, **pd),
                "hoerprobe": {"kopf": ["Sprache: Englisch", "Seiten: 2"], "seiten": [{"seite": 1, "zeilen": ["Heading level 1: Hi"]},
                                                                                       {"seite": 2, "zeilen": ["Paragraph: Text"]}]}}

    def test_bestanden_hoerprobe_ohne_kopf(self):
        n = nachweis.aus_pruefdatei(7, 9, "Brief (fiktiv)", self.daten(), 3)
        self.assertEqual((n["quelle"], n["bestanden"], n["befunde"], n["bilder"], n["alt_texte"], n["lang"], n["aktuell"]),
                         ("pruefdatei", True, 0, 4, 3, "en", True))
        self.assertEqual(n["hoerprobe"], ["Heading level 1: Hi", "Paragraph: Text"])
        self.assertEqual(n["pruefung_url"], "/app?projekt=7&ansicht=abschluss")

    def test_ohne_verapdf_nicht_geprueft_und_ohne_pruefdatei_nichts(self):
        self.assertIsNone(nachweis.aus_pruefdatei(7, 9, "", self.daten(verapdf_moeglich=False, bestanden=False))["bestanden"])
        self.assertFalse(nachweis.aus_pruefdatei(7, 9, "", self.daten(bestanden=False, anzahl_probleme=3))["bestanden"])
        self.assertIsNone(nachweis.aus_pruefdatei(7, 9, "", {"pruefdatei": None}))
        self.assertIsNone(nachweis.aus_pruefdatei("x", 9, "", self.daten()), "nie eine Ausnahme")


class Herausnehmen(unittest.TestCase):
    def test_nur_bei_erfolg_und_nur_nachweise(self):
        n = {"art": "nachweis", "hoerprobe": ["Zeile aus dem Dokument (fiktiv)"]}
        r = {"ok": True, "result": {}, "nachweis": [n, {"art": "link"}, None]}
        self.assertEqual(nachweis.herausnehmen(r), [n])
        self.assertNotIn("nachweis", r, "das Modell sieht ihn nicht")
        r = {"ok": False, "nachweis": n}
        self.assertEqual(nachweis.herausnehmen(r), [])
        self.assertNotIn("nachweis", r)
        self.assertEqual(nachweis.herausnehmen({"ok": True, "nachweis": n}), [n])
        self.assertEqual(nachweis.herausnehmen("kein dict"), [])


class Auftraege(unittest.TestCase):
    def test_hinter_der_datei_und_nur_das_urteil_fuers_modell(self):
        from inkluagent import auftraege
        uid = konto()
        n = {"art": "nachweis", "bestanden": False, "befunde": 2, "hoerprobe": ["Absatz: Geheimtext aus dem Dokument (fiktiv)"]}
        schritte = [{"art": "pdfua", "titel": "In barrierefreie PDF umwandeln", "zustand": "fertig",
                     "anhang": {"art": "pdfua", "dateiname": "brief.pdf", "download_url": "/api/ausgaben/1/datei"}, "nachweis": [n]},
                    {"art": "alt_texte", "titel": "Alt-Texte generieren", "zustand": "fertig"}]
        c = database.get_db()
        aid = c.execute("INSERT INTO agent_auftraege (user_id, project_id, schritte, zustand) VALUES (?, 1, ?, 'fertig')",
                        (uid, json.dumps(schritte))).lastrowid
        c.commit()
        c.close()
        a = auftraege.anhaenge(aid)
        self.assertEqual([x["art"] for x in a], ["pdfua", "nachweis"], "der Nachweis gleich hinter seiner Datei")
        f = json.dumps(auftraege.fakten(aid), ensure_ascii=False)
        self.assertIn('"nachweis": [{"pdf_ua": "nicht bestanden", "befunde": 2}]', f)
        self.assertNotIn("Geheimtext", f, "Zeilen der Hörprobe nie ans Modell")

    def test_schritt_nimmt_ihn_mit(self):
        q = lies(BACKEND, "inkluagent", "auftraege.py")
        self.assertIn("nachweise = nachweis.herausnehmen(r)", q)
        self.assertIn("return dict(felder, zustand=\"fertig\", anhang=anhang, nachweis=nachweise, beendet_am=_jetzt())", q)


class Anschluss(unittest.TestCase):
    def test_gespraech_nimmt_ihn_heraus_bevor_das_modell_es_sieht(self):
        q = lies(BACKEND, "inkluagent", "agent_loop.py")
        self.assertLess(q.index("for a in nachweis.herausnehmen(result):"), q.index("content_text = json.dumps(result"))

    def test_die_drei_werkzeuge_liefern_ihn(self):
        self.assertIn('"nachweis": _nachweis.aus_ausgabe(user_id, r["ausgabe_id"], project.get("alt_language"))',
                      lies(BACKEND, "inkluagent", "tools", "ausgaben.py"))
        self.assertIn('return {"ok": True, "result": result, "anhang": anhang, "nachweis": nachweise}',
                      lies(BACKEND, "inkluagent", "tools", "pdf.py"))
        self.assertIn('r["nachweis"] = nachweis.aus_pruefdatei(', lies(BACKEND, "inkluagent", "tools", "oberflaeche.py"))

    def test_pruefdatei_erstellen_haengt_ihn_an(self):
        from inkluagent.tools import oberflaeche
        daten = {"bilder": 1, "pruefdatei": {"verapdf_moeglich": True, "bestanden": True, "anzahl_probleme": 0, "aktuell": True},
                 "hoerprobe": {"seiten": [{"seite": 1, "zeilen": ["Absatz: Hallo (fiktiv)"]}]}}
        m = mock.MagicMock()
        m._abschluss_erstellen_sync.return_value = True
        with mock.patch.object(oberflaeche, "_main", return_value=m), \
                mock.patch.object(oberflaeche._pdf, "_get_db"), \
                mock.patch.object(oberflaeche._pdf, "_dokument", return_value={"id": 9}), \
                mock.patch.object(oberflaeche._pdf, "_name", return_value="Brief (fiktiv)"), \
                mock.patch.object(oberflaeche, "pruefdatei_lesen", return_value={"ok": True, "result": {}}), \
                mock.patch.object(oberflaeche, "_pruef_daten", return_value=({}, {"id": 9}, daten)), \
                mock.patch.object(nachweis, "alt_texte_im_dokument", return_value=1):
            r = oberflaeche.pruefdatei_erstellen(7, 1, 9)
        self.assertEqual((r["nachweis"]["bestanden"], r["nachweis"]["hoerprobe"], r["nachweis"]["dokument"]),
                         (True, ["Absatz: Hallo (fiktiv)"], "Brief (fiktiv)"))


class Karte(unittest.TestCase):
    def setUp(self):
        js = lies(FRONTEND, "inkluagent-nachrichten.js")
        k = js[js.index("function inkluagentNachweisKarte(a) {"):]
        self.k = k[:k.index("\n}\n")]
        self.js = js

    def test_eingebunden(self):
        self.assertIn("if (a && a.art === 'nachweis') {   // „Fertig mit Nachweis“ (17.10.2026)\n            box.appendChild(inkluagentNachweisKarte(a));",
                      self.js)

    def test_nur_text_eigene_pfade_sprache_anhoeren(self):
        self.assertNotIn("innerHTML", self.k)
        self.assertIn("zeilen.forEach(z => { const p = document.createElement('p'); p.textContent = z; hp.appendChild(p); });", self.k)
        self.assertIn("if (/^[a-z]{2}$/.test(a.lang || '')) hp.setAttribute('lang', a.lang);", self.k)
        self.assertIn("if (!url || !eigenerPfad(url)) return;", self.k)
        self.assertIn("inkluagentAnhoerenKnopf(titel, hp);", self.k)

    def test_urteil_als_text(self):
        for satz in ("PDF/UA-Prüfung (veraPDF): bestanden", "PDF/UA-Prüfung (veraPDF): nicht bestanden, {n} Befunde",
                     "PDF/UA-Prüfung (veraPDF): nicht bestanden, 1 Befund", "PDF/UA-Prüfung (veraPDF): nicht bestanden",
                     "PDF/UA-Prüfung (veraPDF): nicht geprüft", "Alt-Texte: {n} von {m} Bildern"):
            self.assertIn(f"t('{satz}'", self.k, satz)
        self.assertIn("if (a.bestanden === true)", self.k, "nur ein echtes true ist „bestanden“")


if __name__ == "__main__":
    unittest.main()
