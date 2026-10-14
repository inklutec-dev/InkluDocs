"""Werkzeug-Schemata, die jeder KI-Anbieter annimmt (14.10.2026, Rauchtest Staging): Gemini lehnte die ganze Anfrage mit HTTP 400 ab,
weil ein Werkzeug eine Auswahlliste („enum“) aus Zahlen hatte — der InkluAgent antwortete dann gar nicht mehr, auch nicht ohne das
Werkzeug. Geprueft wird JEDES Werkzeug, das der Agent bekommen kann (alle Schalter an, alle Projektarten, Konto, Karten, Kaufen,
Rueckmeldung, Auftraege, Begleiter): Auswahllisten nur aus Texten und nur an Feldern vom Typ „string“.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_werkzeug_schema.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="werkzeug_schema_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
import stripe_zahlung  # noqa: E402
from inkluagent import agent_loop, auftraege, begleiter, karten, konto, rueckmeldung  # noqa: E402

PROJEKTARTEN = ({"project_type": "pdf", "tool": "pdf"}, {"project_type": "docx", "tool": "word"},
                {"project_type": "pdfform", "tool": "formular"}, {"project_type": "images", "tool": "grafik"},
                {"project_type": "url", "tool": "web"})


def _auswahllisten(schema, pfad=""):
    """(pfad, typ, enum) fuer jede Auswahlliste im Schema, auch in verschachtelten properties und items."""
    if not isinstance(schema, dict):
        return []
    out = [(pfad, schema.get("type"), schema["enum"])] if "enum" in schema else []
    for name, teil in (schema.get("properties") or {}).items():
        out += _auswahllisten(teil, f"{pfad}.{name}")
    if "items" in schema:
        out += _auswahllisten(schema["items"], pfad + "[]")
    return out


class Schemata(unittest.TestCase):
    def alle_werkzeuge(self):
        schalter = [n for n in dir(funktionen) if n.isupper() and isinstance(getattr(funktionen, n), bool)]
        with mock.patch.multiple(funktionen, **{n: True for n in schalter}), mock.patch.object(stripe_zahlung, "AKTIV", True):
            defs = []
            for projekt in PROJEKTARTEN:
                defs += agent_loop._werkzeugsatz(projekt, 1, 1)[0]
            defs += konto.werkzeugsatz(1)[0]
            defs += karten.werkzeuge(None) + karten.werkzeuge(1)
            defs += auftraege.werkzeuge(None) + auftraege.werkzeuge(1)
            defs += [rueckmeldung.werkzeug_melden(None), rueckmeldung.werkzeug_melden(1)]
            defs += begleiter.werkzeuge(1)
        return defs

    def test_auswahllisten_nur_aus_texten(self):
        defs = self.alle_werkzeuge()
        self.assertGreater(len({d["name"] for d in defs}), 60, "der Satz ist vollstaendig")
        for d in defs:
            for pfad, typ, werte in _auswahllisten(d.get("input_schema") or {}):
                with self.subTest(werkzeug=d["name"], feld=pfad):
                    self.assertEqual(typ, "string", "eine Auswahlliste nur an Feldern vom Typ string")
                    self.assertTrue(all(isinstance(w, str) for w in werte), werte)


if __name__ == "__main__":
    unittest.main()
