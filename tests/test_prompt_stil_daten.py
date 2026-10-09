"""Stilrunde Datengrafiken (Oktober 2026): Einstieg ohne Ansage, Länge mit Zielmarke.

Anlass: Karten-Texte enthielten in rund der Hälfte eine Ansage ("Sie zeigt …", "Die Karte
zeigt …"), obwohl Stilregel 3 das verbietet. Ursache: Die Schreibweise-Regel sagte nur, das
Gattungswort stehe "als normales Wort im Satz". Das Modell schloss den Kopf ("Karte zum
Mobilitätskonzept 2030 der Beispielstadt.") meist als eigenen Satz ab und machte im nächsten
Satz die Karte zum Subjekt; das Kartenbeispiel begann die Langbeschreibung selbst mit
"Die Karte ist nach Norden ausgerichtet." Jetzt regelt Stilregel 4 der sachlichen Fassung
an einer Stelle für alle Datengrafiken: Gattungswort und Thema, Doppelpunkt, Aussage; danach
ist der Inhalt Subjekt, auch in der Langbeschreibung. Die Längenregel nennt die Kernfakten
einer Karte und eine Zielmarke (meist um 250 Zeichen, dicht höchstens 350).

Lauf im Container: docker exec -w /app inkludocs-staging python3 -m unittest tests/test_prompt_stil_daten.py
"""
from __future__ import annotations

import os
import re
import sys
import unittest

BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from prompts.builders.combo import build_combined_inventar_beschreibung_prompt  # noqa: E402
from prompts.builders.helpers import load_examples  # noqa: E402
from prompts.components import stilregeln  # noqa: E402

DATEN = {'diagramm': 'diagramm', 'tabelle': 'tabelle', 'karte': 'karte', 'infografik': 'infografik',
         'screenshot': 'screenshot', 'strukturformel': 'strukturformel'}
EINSTIEG = 'Gattungswort und Thema eröffnen den Alt-Text, ein Doppelpunkt führt direkt zur Aussage'
SUBJEKT = 'Danach ist der Inhalt Subjekt, nicht mehr die Grafik oder ein Teil von ihr'
GRAFIK_ALS_SUBJEKT = re.compile(r'^(Die|Der|Das|Beide|Zwei|Diese)\s+(\w+\s+){0,2}?(Karte|Karten|Grafik|Abbildung|'
                                r'Darstellung|Tabelle|Infografik|\w*[Dd]iagramm|Screenshot|Strukturformel)\b|^(Sie|Er|Es)\s+zeig')


def _flach(text: str) -> str:
    return ' '.join(text.split())


def _combo(effektiv: str, top: str) -> str:
    return _flach(build_combined_inventar_beschreibung_prompt(
        bildtyp_top=top, bildtyp_effective=effektiv, enriched_context='Abbildung 3: Beispiel', width=800, height=600))


class StilDatengrafikTest(unittest.TestCase):
    def test_einstieg_einmal_je_datengrafik(self):
        for effektiv, top in DATEN.items():
            p = _combo(effektiv, top)
            self.assertEqual(p.count(EINSTIEG), 1, f'{effektiv}: Einstiegsregel fehlt oder doppelt')
            self.assertEqual(p.count(SUBJEKT), 1, f'{effektiv}: Subjekt-Regel fehlt oder doppelt')
            self.assertNotIn('als normales Wort im Satz', p, f'{effektiv}: alter Wortlaut der Schreibweise')
            self.assertNotIn('Auch sie beginnt nicht mit einer Ansage', p, f'{effektiv}: doppelte Ansage-Regel')

    def test_laenge_mit_zielmarke_und_karte(self):
        p = _flach(stilregeln.STILREGELN_SACHLICH)
        self.assertIn('zwei bis drei kurze Sätze, meist um 250 Zeichen, bei dichten Grafiken höchstens 350.', p)
        self.assertIn('bei Karten die Orte, Gebiete oder Wege, um die es geht', p)
        self.assertEqual(p.count('Zeichen'), 2, 'Längenangaben nur an einer Stelle (Alt-Text 250/350, Langbeschreibung 2000)')

    def test_fotos_und_chatbot_unveraendert(self):
        """Die Foto-Fassung (auch im Chatbot und im Prüfpass) bekommt die Datengrafik-Regel nicht."""
        for text in (stilregeln.STILREGELN, stilregeln.STILREGELN_KERN):
            self.assertNotIn(EINSTIEG, _flach(text))
        from inkluagent.prompts.system_agent import SYSTEM_AGENT
        self.assertNotIn(EINSTIEG, _flach(SYSTEM_AGENT))
        self.assertIn(_flach(stilregeln.STILREGELN), _flach(SYSTEM_AGENT))

    def test_beispiele_halten_die_regel(self):
        """Gute Beispiele: Alt-Text mit Gattungswort und Doppelpunkt im ersten Satz, Langbeschreibung ohne
        die Grafik als Subjekt am Anfang (das Kartenbeispiel begann vorher mit "Die Karte ist …")."""
        for typ in DATEN:
            for ex in load_examples(typ).good_examples:
                antwort = ex.get('antwort') or {}
                alt, lang = antwort.get('alt_text', ''), antwort.get('langbeschreibung', '')
                self.assertIn(':', alt[:160], f'{typ}: Beispiel-Alt-Text ohne Doppelpunkt nach dem Kopf: {alt[:60]}')
                self.assertIsNone(GRAFIK_ALS_SUBJEKT.match(lang), f'{typ}: Langbeschreibung beginnt mit der Grafik: {lang[:60]}')
                self.assertNotIn('im Satz', ex.get('prinzip', ''), f'{typ}: Merksatz mit altem Wortlaut')


if __name__ == '__main__':
    unittest.main()
