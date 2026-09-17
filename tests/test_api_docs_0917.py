"""17.09.2026: Swagger/OpenAPI aus, API-Doku als mehrsprachiges Template mit echten Limits und Credits
(im Container: python3 -m unittest /app/tests/test_api_docs_0917.py)."""
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

    def test_doku_deutsch_nennt_zahlen_und_dokumente(self):
        r = self.c.get("/api/v1/docs", headers={"Accept-Language": "de"})
        self.assertEqual(r.status_code, 200)
        t = r.text
        self.assertIn('lang="de"', t)
        self.assertNotIn("100 Bilder pro Tag", t)
        self.assertIn(f"{main.DAILY_IMAGE_LIMIT} generierte Texte pro Tag", t)
        self.assertIn(f"{main.API_RATE_LIMIT_DAY} pro Tag", t)
        self.assertIn(f"{main.billing.AKTIONS_PREISE['bild_generierung']} Credits", t)
        for anker in ("h-auth", "h-grund", "h-bild", "h-dok", "h-wege", "h-fehler", "h-limits", "h-kosten", "h-beispiele"):
            self.assertIn(f'id="{anker}"', t, anker)
        self.assertIn("/api/v1/documents/{id}/review-link", t)
        self.assertNotIn("%%BASE_URL%%", t)
        self.assertIn(main.BASE_URL.rstrip("/") + "/api/v1/documents", t)

    def test_doku_in_allen_sechs_sprachen(self):
        erwartet = {"en": ("Documents", "What this is about"), "fr": ("Documents", "De quoi il s’agit"),
                    "es": ("Documentos", "De qué se trata"), "da": ("Dokumenter", "Hvad det handler om"),
                    "sv": ("Dokument", "Vad det handlar om"), "de": ("Dokumente", "Worum es geht")}
        for lang, (dok, intro) in erwartet.items():
            r = self.c.get("/api/v1/docs", headers={"Accept-Language": lang})
            self.assertEqual(r.status_code, 200, lang)
            self.assertIn(f'lang="{lang}"', r.text, lang)
            self.assertIn(f'<h2 id="h-dok">{dok}</h2>', r.text, lang)
            # Ueberschrift der Einleitung in der Zielsprache (die deutschen msgids stehen ohnehin in window.I18N)
            self.assertIn(f'<h2 id="h-intro">{intro}</h2>', r.text, lang)


if __name__ == "__main__":
    unittest.main()
