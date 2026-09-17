"""Drift-Pruefung (17.09.2026, Steve): Joerg Heines PDFix-Skript ist die Vorlage, InkluDocs traegt
nur markierte Zeilen auf. Ohne diese Zeilen muss die Betriebsfassung BYTEIDENTISCH mit dem Original
unter original_heine/ sein — sonst hat jemand in Heines Logik geaendert.
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_pdfix_skript_drift.py
"""
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KANDIDATEN = ("/app/pdfix_scripts", os.path.join(os.path.dirname(HERE), "backend", "pdfix_scripts"))
SKRIPTE = os.path.join(next(k for k in KANDIDATEN if os.path.isdir(k)))
KOPF_ENDE = "# === InkluDocs-Kopf Ende ==="
ORIGINAL_MARKER = "# InkluDocs-Original: "
PAARE = [("Formular_Export_Quickinfo.py", "original_heine/Formulare_Export_08.py")]


def rekonstruiere(betrieb: str) -> str:
    """Kopfblock weg, '# InkluDocs-Original: <Zeile>' -> <Zeile>, Zeilen mit '# InkluDocs' weg."""
    out, nach_kopf = [], False
    for z in betrieb.split("\n"):
        if not nach_kopf:
            if z == KOPF_ENDE:
                nach_kopf = True
            continue
        if z.startswith(ORIGINAL_MARKER):
            out.append(z[len(ORIGINAL_MARKER):])
            continue
        if "# InkluDocs" in z:
            continue
        out.append(z)
    return "\n".join(out)


class DriftTest(unittest.TestCase):
    def test_betriebsfassung_ist_original_plus_markierte_zeilen(self):
        for betrieb, original in PAARE:
            with open(os.path.join(SKRIPTE, betrieb), encoding="utf-8") as f:
                b = f.read()
            with open(os.path.join(SKRIPTE, original), encoding="utf-8") as f:
                o = f.read()
            self.assertIn(KOPF_ENDE, b, betrieb)
            self.assertEqual(rekonstruiere(b), o, f"{betrieb} weicht von {original} ab (Heines Logik geaendert?)")

    def test_markierte_zeilen_sind_genau_die_betriebsanpassungen(self):
        with open(os.path.join(SKRIPTE, "Formular_Export_Quickinfo.py"), encoding="utf-8") as f:
            zeilen = f.read().split("\n")
        nach_kopf = zeilen[zeilen.index(KOPF_ENDE) + 1:]
        markiert = [z for z in nach_kopf if "# InkluDocs" in z]
        # Betriebslogik liegt im Helfer, nicht im Skript: jede Ergaenzung ist EINE Zeile.
        self.assertTrue(all("\n" not in z for z in markiert))
        for pflicht in ("import inkludocs_betrieb as betrieb", "'-c', '--csv'", "betrieb.feldwert_maskieren(feldwert)",
                        "betrieb.seiten_absichern(daten)", "FIELDS_FOUND", "pfadcsv = args.csv", "doc.Close()"):
            self.assertTrue(any(pflicht in z for z in markiert), pflicht)
        self.assertFalse(any('input("Drücke ENTER' in z and not z.startswith(ORIGINAL_MARKER) for z in nach_kopf))

    def test_betriebshelfer(self):
        import sys
        sys.path.insert(0, SKRIPTE)
        import inkludocs_betrieb as b
        self.assertEqual(b.feldwert_maskieren(""), "kein Wert")
        self.assertEqual(b.feldwert_maskieren("Off"), "kein Wert")
        self.assertEqual(b.feldwert_maskieren("Max Mustermann (fiktiv)"), "Wert vorhanden")
        self.assertEqual(b.sauber("a\ud800b"), "a?b")
        daten = [[1, "a", "", 4, "Text field", "kein Wert", "", "", "", "", "", 1], [2, "b", "", 4, "Text field", "kein Wert", "2", 10, 20, 30, 40, 1]]
        d = b.seiten_absichern(daten)
        self.assertEqual((d[0][6], d[0][7], d[0][10]), (0, 0, 0))
        self.assertEqual((d[1][6], d[1][7], d[1][10]), (2, 10, 40))


if __name__ == "__main__":
    unittest.main()
