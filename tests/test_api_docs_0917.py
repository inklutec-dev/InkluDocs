"""17.09.2026: Swagger/OpenAPI aus, API-Doku nennt echte Limits und Credits (im Container: python3 -m unittest /app/tests/test_api_docs_0917.py)."""
import os
import sys
import unittest

sys.path.insert(0, "/app")
os.chdir("/app")
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402


class ApiDokuTest(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(main.app)

    def test_swagger_und_openapi_sind_aus(self):
        for pfad in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(self.c.get(pfad).status_code, 404, pfad)

    def test_doku_nennt_echte_limits_und_credits(self):
        r = self.c.get("/api/v1/docs")
        self.assertEqual(r.status_code, 200)
        t = r.text
        self.assertNotIn("100 Bilder pro Tag", t)
        self.assertIn("500 generierte Alt-Texte pro Tag", t)
        self.assertIn("1.000 Anfragen pro Tag", t)
        self.assertIn('id="kosten"', t)
        self.assertIn("5 Credits", t)
        self.assertIn('href="#kosten"', t)
        self.assertNotIn("%%BASE_URL%%", t)


if __name__ == "__main__":
    unittest.main()
