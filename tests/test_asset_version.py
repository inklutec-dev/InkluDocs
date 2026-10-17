"""Cache-Busting (?v= an CSS/JS, i18n._compute_asset_version): jede Datei, die eine Vorlage mit ?v={{ asset_version }} laedt, zaehlt
in die Version. Sonst laeuft nach einem Neubau im Browser noch das alte Skript — passiert mit formular.js (09.09.2026), dokument.js
und abschluss.js (25.09.2026); am 17.10.2026 fehlten noch hochladefeld.js, klang.js und das Skript des InkluAgent, das seit dem Tag
aus Teilen besteht (inkluagent/skripte.py; ein alter Kern mit neuen Teilen waere gemischt). Nur Quelltext, ohne Server:
    python3 -m unittest tests/test_asset_version.py
"""
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = Path(next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                     if os.path.isdir(k)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import i18n  # noqa: E402
from inkluagent import skripte  # noqa: E402


class AssetVersion(unittest.TestCase):
    def setUp(self):
        mock.patch.object(i18n, "FRONTEND_DIR", FRONTEND).start()   # im Container ist das schon so; im Repo liegt frontend/ daneben
        self.addCleanup(mock.patch.stopall)

    def test_jede_versioniert_geladene_datei_zaehlt(self):
        geladen = set()
        vorlagen = Path(BACKEND) / "templates"
        for tpl in vorlagen.glob("*.html"):
            geladen |= set(re.findall(r"/static/([A-Za-z0-9_.-]+)\?v=\{\{ ?asset_version ?\}\}", tpl.read_text(encoding="utf-8")))
        self.assertIn('/static/{{ s }}?v={{ asset_version }}', (vorlagen / "base_app.html").read_text(encoding="utf-8"))
        geladen |= set(skripte.TEILE)   # die Teile des InkluAgent laedt eine Schleife ueber diese Liste
        gezaehlt = {f.name for f in i18n._asset_dateien()}
        self.assertEqual(sorted(geladen - gezaehlt), [], "mit ?v= geladen, aber nicht in i18n._ASSET_FILES")
        for f in i18n._asset_dateien():
            if f.name in geladen:
                self.assertTrue(f.is_file(), f)

    def test_ein_geaenderter_teil_aendert_die_version(self):
        ziel = Path(tempfile.mkdtemp(prefix="assets_"))
        for f in i18n._asset_dateien():
            if f.is_file():
                shutil.copy(f, ziel / f.name)
        with mock.patch.object(i18n, "FRONTEND_DIR", ziel):
            vorher = i18n._compute_asset_version()
            with open(ziel / "inkluagent-live.js", "a", encoding="utf-8") as f:
                f.write("// geaendert (Test)\n")
            self.assertNotEqual(i18n._compute_asset_version(), vorher, "nur ein Teil des InkluAgent geaendert")


if __name__ == "__main__":
    unittest.main()
