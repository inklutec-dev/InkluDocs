"""Michael Karbe, Feedback 20261012 - 1 (Mail 12.10.2026, Steves Go): InkluAgent-Aufbau (Eingabe oben, kein grauer Kasten, Feld
„Dateien mitschicken“ per Schalter aus, Beschreibung unter einer Linie, Platzhalter), Startseite (je drei: Projekte, Auftraege,
Ablage; keine „Alle …“-Knoepfe; „Geteilte Projekte“ neben „Neues Projekt anlegen“), lange Listen (fuenf + „Alle anzeigen“),
Alt-Texte/Quickinfos nur Dokumente mit Bildern bzw. Feldern. Am Quelltext und am Schalter, ohne Browser.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_michael_feedback_1012.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="michael_1012_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_HOCHLADEN", "INKLUAGENT_HOCHLADEFELD"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import funktionen  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


class Agent(unittest.TestCase):
    def setUp(self):
        self.js = lies(FRONTEND, "inkluagent.js")
        f = self.js[self.js.index("function inkluagentInhaltHtml("):]
        self.html = f[:f.index("\n}\n")]

    def test_eingabe_direkt_unter_dem_knopf_beschreibung_unter_einer_linie(self):
        h = self.html
        self.assertLess(h.index("inkluagentToggle"), h.index("inkluagentPanel"))
        self.assertLess(h.index("'<form id=\"inkluagentForm\""), h.index("id=\"inkluagentLog\""), "Punkt 1: Eingabe vor dem Verlauf")
        self.assertLess(h.index("id=\"inkluagentLog\""), h.index("inkluagent-trenner"))
        self.assertLess(h.index("inkluagent-trenner"), h.index("inkluagent-beschreibung"), "Punkt 4: Beschreibung unter der Linie")
        self.assertLess(h.index("inkluagent-beschreibung"), h.index("inkluagentKurzhilfeHtml(variante)"))
        self.assertLess(h.index("'<div id=\"inkluagentPanel\""), h.index("inkluagent-beschreibung"), "zugeklappt nur die Ueberschrift")

    def test_kein_grauer_kasten_und_platzhalter(self):
        css = lies(FRONTEND, "style.css")
        self.assertNotIn("Noch keine Nachrichten in diesem Projekt", css)
        self.assertIn(".inkluagent-log:empty { display: none; }", css)
        f = self.js[self.js.index("function inkluagentPlatzhalter("):]
        self.assertIn("t('Was soll der InkluAgent für dich erledigen?')", f[:f.index("\n}\n")])

    def test_dateifeld_nur_mit_eigenem_schalter(self):
        self.assertFalse(funktionen.AGENT_HOCHLADEFELD)
        try:
            import main
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        with mock.patch.object(funktionen, "AGENT_HOCHLADEN", True), mock.patch.object(main, "INKLUAGENT_ENABLED", True):
            self.assertFalse(main._inkluagent_schalter()["hochladen"], "Punkt 3: Feld aus, Server-Teil bleibt")
            with mock.patch.object(funktionen, "AGENT_HOCHLADEFELD", True):
                self.assertTrue(main._inkluagent_schalter()["hochladen"])

    def test_agent_bietet_ohne_feld_kein_hochladen_hier_an(self):
        """Staging-Stichprobe 12.10.2026: der Agent sagte „oder Dateien direkt hochladen“, obwohl das Feld aus war."""
        from inkluagent import konto
        beschreibung = next(d["description"] for d in konto.TOOL_DEFINITIONS_KONTO_AENDERN if d["name"] == "projekt_anlegen")
        self.assertNotIn("Hochladefeld", beschreibung)
        with mock.patch.object(funktionen, "AGENT_HOCHLADEN", True):
            system = konto.werkzeugsatz(1)[2]
            self.assertIn("gibt es kein Feld zum Hochladen", system)
            self.assertNotIn("Dateien mitschicken", system)
            with mock.patch.object(funktionen, "AGENT_HOCHLADEFELD", True):
                system = konto.werkzeugsatz(1)[2]
                self.assertIn("„Dateien mitschicken“", system)
                self.assertNotIn("kein Feld zum Hochladen", system)


class Startseite(unittest.TestCase):
    def setUp(self):
        self.html = lies(BACKEND, "templates", "dashboard.html")

    def test_je_drei_ohne_alle_knoepfe(self):
        self.assertIn("const STARTSEITE_ANZAHL = 3;", self.html)
        self.assertEqual(self.html.count(".slice(0, STARTSEITE_ANZAHL)"), 3)
        for weg in ("t('Alle Projekte anzeigen')", "t('Alle Aufträge')", "t('Ablage öffnen')"):
            self.assertNotIn(weg, self.html)
        self.assertIn("updated_at", self.html, "zuletzt BEARBEITET")

    def test_reihenfolge_projekte_auftraege_ablage(self):
        h = self.html
        self.assertLess(h.index('id="recent-h"'), h.index('id="express-h"'))
        self.assertLess(h.index('id="express-h"'), h.index('id="ausgaben-h"'))

    def test_geteilte_projekte_neben_neues_projekt(self):
        h = self.html
        start = h.index('<p class="dash-neu-projekt">')
        absatz = h[start:h.index("</p>", start)]
        self.assertIn("Neues Projekt anlegen", absatz)
        self.assertIn('id="sharedBtn"', absatz)


class Listen(unittest.TestCase):
    def test_fuenf_und_alle_anzeigen(self):
        dash = lies(FRONTEND, "dashboard.js")
        self.assertIn("function listeKuerzen(container, opts)", dash)
        self.assertIn("const anzahl = opts.anzahl || 5;", dash)
        self.assertIn("t('Alle anzeigen ({n})'", dash)
        self.assertIn("listeKuerzen(ul, { schluessel: 'projekte' })", lies(BACKEND, "templates", "projekte.html"))
        self.assertIn("listeKuerzen(ul, { schluessel: 'ablage' })", lies(BACKEND, "templates", "ablage.html"))
        self.assertIn("listeKuerzen(ziel, { schluessel: 'auftraege'", lies(BACKEND, "templates", "express.html"))
        self.assertIn("schluessel: 'dokumente-' + projectId, vonHinten: true", lies(FRONTEND, "dokument.js"))


class Ansichten(unittest.TestCase):
    def test_nur_dokumente_mit_bildern_bzw_feldern(self):
        app = lies(BACKEND, "templates", "app.html")
        self.assertIn("if (nurMitBildern && !pageMap.size) return '';", app)
        self.assertIn("(imPdfProjekt && !gast() && !(felderJeDoc.get(d.id) || []).length) ? ''", lies(FRONTEND, "formular.js"))


if __name__ == "__main__":
    unittest.main()
