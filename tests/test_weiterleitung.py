"""Ruecksprung nach der Anmeldung (Express Runde 7, 06.10.2026): nur interne Pfade — kein offener Redirect."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import weiterleitung as w  # noqa: E402


class SicheresZiel(unittest.TestCase):
    def test_interne_pfade(self):
        for ziel in ("/express/auftrag/12", "/app?projekt=957", "/express?auftrag=3#exa_karte_3", "/dashboard"):
            self.assertEqual(w.sicheres_ziel(ziel), ziel)

    def test_abgelehnt(self):
        for ziel in ("", None, "https://boese.example/", "//boese.example", "///boese.example", "/\\boese.example",
                     "\\\\boese", "javascript:alert(1)", "express/auftrag/1", " /app", "/app\n", "/a b", "/login",
                     "/login?weiter=/app", "/logout", "/" + "x" * 1000, "/%0d%0aSet-Cookie:x", "http:/boese"):
            self.assertEqual(w.sicheres_ziel(ziel), "" if ziel != "/%0d%0aSet-Cookie:x" else ziel, repr(ziel))

    def test_pfad_normalisiert(self):
        """Nachkontrolle Runde 7 (Hinweis): „..“ und „.“ fallen weg, nie entsteht //host."""
        self.assertEqual(w.sicheres_ziel("/..//boese.example"), "/boese.example")
        self.assertEqual(w.sicheres_ziel("/express/../app?projekt=1"), "/app?projekt=1")
        self.assertEqual(w.sicheres_ziel("/a/./b/"), "/a/b/")
        self.assertEqual(w.sicheres_ziel("/x/../login"), "")
        self.assertEqual(w.sicheres_ziel("/express/auftrag/5#oben"), "/express/auftrag/5#oben")

    def test_login_adresse(self):
        self.assertEqual(w.login_adresse("/express/auftrag/12"), "/login?weiter=%2Fexpress%2Fauftrag%2F12")
        self.assertEqual(w.login_adresse("/app", "projekt=957"), "/login?weiter=%2Fapp%3Fprojekt%3D957")
        self.assertEqual(w.login_adresse("/app"), "/login")
        self.assertEqual(w.login_adresse("/login"), "/login")
        self.assertEqual(w.login_adresse("//boese.example"), "/login")


if __name__ == "__main__":
    unittest.main()
