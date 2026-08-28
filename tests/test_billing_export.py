"""Export-Staffel (28.08.2026): 5 Credits + 1 je angefangene 10 Bilder/Felder.
    docker exec inkludocs-staging python3 -m unittest /app/tests/test_billing_export.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import billing  # noqa: E402


class TestExportPreis(unittest.TestCase):
    def test_staffel(self):
        self.assertEqual(billing.export_preis(0), 5)
        self.assertEqual(billing.export_preis(1), 6)
        self.assertEqual(billing.export_preis(10), 6)
        self.assertEqual(billing.export_preis(11), 7)
        self.assertEqual(billing.export_preis(26), 8)
        self.assertEqual(billing.export_preis(50), 10)
        self.assertEqual(billing.export_preis(100), 15)

    def test_konstanten(self):
        self.assertEqual(billing.EXPORT_GRUNDPREIS, 5)
        self.assertEqual(billing.EXPORT_STAFFEL, 10)
