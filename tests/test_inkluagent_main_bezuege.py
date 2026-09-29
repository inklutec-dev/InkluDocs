"""Jede Funktion, die die InkluAgent-Werkzeuge aus main aufrufen (m = _main(); m.<name>), muss es in main geben.

Anlass (Review 29.09.2026): main._ungetaggte_pruefen wurde entfernt, backend/inkluagent/tools/pdf.py rief es weiter
auf — der PDF-Export im Chat antwortete nur noch mit „Tool-Ausführung crashte“. Kein anderer Test rief den Weg auf.
"""
import glob
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)
import main  # noqa: E402

WERKZEUGE = os.path.join(os.path.dirname(main.__file__), "inkluagent", "tools")


class MainBezuege(unittest.TestCase):
    def test_alle_m_namen_existieren(self):
        dateien = glob.glob(os.path.join(WERKZEUGE, "*.py"))
        self.assertTrue(dateien, WERKZEUGE)
        fehlend = []
        for pfad in dateien:
            text = open(pfad, encoding="utf-8").read()
            if "_main()" not in text:
                continue
            for name in sorted(set(re.findall(r"\bm\.([A-Za-z_]\w*)", text))):
                if not hasattr(main, name):
                    fehlend.append(f"{os.path.basename(pfad)}: m.{name}")
        self.assertEqual(fehlend, [], "Werkzeuge rufen Funktionen auf, die es in main nicht (mehr) gibt")


if __name__ == "__main__":
    unittest.main()
