"""Legende vor Alltagswissen (Oktober 2026): sichert die Regel im gerenderten Prompt ab.

Anlass: Eine Karte mit Legende wies eine hellblaue Fläche einem Eigentümer zu; das
Modell gab die Legende richtig wieder und nannte die Fläche trotzdem Gewässer.
Die allgemeine Regel steht genau einmal im gemeinsamen Block LESBARER TEXT UND
LEGENDE der Datengrafiken. Der Karten-Prompt wendet sie im AUFTRAG auf Flächen und
Bänder an, und das innere Inventar der Karte ordnet jede Fläche ihrem
Legendeneintrag zu. Fotos bekommen den Block nicht. Der Prüfpass (Chatbot-Speicherweg)
beanstandet Deutungen gegen die Legende, und beim Neu-Generieren ist der bisherige Text
Formulierungsvorlage, kein Beleg.

Lauf im Container: docker exec -w /app inkludocs-staging python3 -m unittest tests/test_prompt_legende.py
"""
from __future__ import annotations

import os
import sys
import unittest

BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from prompts.builders.combo import build_combined_inventar_beschreibung_prompt  # noqa: E402

REGEL = 'Eine Legende legt fest, was Farben, Muster, Linien und Symbole in dieser Grafik'
VORRANG = 'sie geht dem Alltagswissen vor'


def _combo(effektiv: str, top: str) -> str:
    return build_combined_inventar_beschreibung_prompt(
        bildtyp_top=top, bildtyp_effective=effektiv,
        enriched_context='Abbildung 3: Beispiel', width=800, height=600,
    )


def _abschnitt(text: str, titel: str, naechster: str) -> str:
    return text.split(f'\n{titel}\n', 1)[1].split(f'\n{naechster}', 1)[0]


class LegendeRegelTest(unittest.TestCase):
    def test_karte_regel_auftrag_und_inventar(self):
        p = _combo('karte', 'karte')
        self.assertEqual(p.count('LESBARER TEXT UND LEGENDE'), 1)
        self.assertIn(REGEL, p)
        self.assertIn(VORRANG, p)
        auftrag = _abschnitt(p, 'AUFTRAG', 'DEIN INNERES INVENTAR')
        self.assertIn('Hat die Karte eine Legende, gilt sie auch für die Flächen', auftrag)
        self.assertIn('ein Gewässer also nur, wenn', auftrag)
        inventar = _abschnitt(p, 'DEIN INNERES INVENTAR (Schritt 1)', 'ALT-TEXT')
        self.assertIn('die zu einem Eintrag passt, mit diesem Eintrag', inventar)

    def test_alle_datengrafiken_mit_block(self):
        for typ in ('diagramm', 'tabelle', 'karte', 'infografik', 'screenshot'):
            p = _combo(typ, typ)
            self.assertEqual(p.count(REGEL), 1, f'{typ}: Legenden-Regel fehlt oder steht doppelt')
            self.assertEqual(p.count(VORRANG), 1, f'{typ}: Vorrang der Legende fehlt oder steht doppelt')

    def test_fotos_ohne_block(self):
        for typ in ('foto_landschaft', 'foto_objekte', 'foto_event'):
            p = _combo(typ, 'foto')
            self.assertNotIn('LESBARER TEXT UND LEGENDE', p, f'{typ}: Datengrafik-Block im Foto-Prompt')
            self.assertNotIn(REGEL, p)

    def test_pruefpass_kriterium_legende(self):
        from pipelines.v4 import orchestrator as orch
        p = orch._build_verify_prompt('Ein Alt-Text', enriched_context='Kontext', bildtyp='karte')
        self.assertIn('- Legende: Farben, Muster, Linien und Symbole bedeuten, was die Legende ihnen zuweist.', p)

    def test_neu_generieren_vorlage_ist_kein_beleg(self):
        from pipelines.v4 import orchestrator as orch
        s = orch._variation_suffix('Karte zum Landabtausch: Ein Fluss teilt das Gebiet.')
        self.assertIn('Formulierungsvorlage, kein Beleg', s)
        self.assertIn('Fotomontage-/Collage-Kennzeichnung', s)  # belegte Kernfakten bleiben geschuetzt
        self.assertEqual(orch._variation_suffix(''), '')

    def test_pruefpass_gezielt_fuer_karten(self):
        from unittest import mock
        from pipelines.v4 import orchestrator as orch
        with mock.patch.dict(os.environ, {'V4_VERIFY_MODE': 'karte'}):
            self.assertTrue(orch._verify_scope_matches('karte'))
            self.assertFalse(orch._verify_scope_matches('foto_event'))
        with mock.patch.dict(os.environ, {'V4_VERIFY_MODE': 'karte, diagramm'}):
            self.assertTrue(orch._verify_scope_matches('diagramm'))
        with mock.patch.dict(os.environ, {'V4_VERIFY_MODE': 'off'}):
            self.assertFalse(orch._verify_scope_matches('karte'))
        with mock.patch.dict(os.environ, {'V4_VERIFY_MODE': 'kritisch'}):
            self.assertTrue(orch._verify_scope_matches('foto_event'))

    def test_faktenblatt_gezielt_fuer_karten(self):
        from unittest import mock
        from pipelines.v4 import orchestrator as orch
        with mock.patch.dict(os.environ, {'V4_FAKTENBLATT': 'karte'}):
            self.assertTrue(orch._faktenblatt_an('karte'))
            self.assertFalse(orch._faktenblatt_an('tabelle'))
        with mock.patch.dict(os.environ, {'V4_FAKTENBLATT': 'on'}):
            self.assertTrue(orch._faktenblatt_an('tabelle'))
        with mock.patch.dict(os.environ, {'V4_FAKTENBLATT': 'off'}):
            self.assertFalse(orch._faktenblatt_an('karte'))

    def test_bisherige_schalterwerte_unveraendert(self):
        """Die Typ-Liste ist ein Zusatz: jeder bisherige Wert wirkt wie vor dem Oktober 2026."""
        from unittest import mock
        from pipelines.v4 import orchestrator as orch
        typen = ('karte', 'tabelle', 'infografik', 'diagramm', 'screenshot', 'foto_event', 'foto_objekte')
        for wert in ('on', 'ON', ' on ', 'off', 'OFF', '', 'aus', 'an', 'true', '1', 'alle'):
            with mock.patch.dict(os.environ, {'V4_FAKTENBLATT': wert}):
                for typ in typen:
                    self.assertEqual(orch._faktenblatt_an(typ), wert.strip().lower() == 'on',
                                     f'V4_FAKTENBLATT={wert!r}, {typ}')
        for wert in ('off', 'OFF', '', 'aus', 'on', 'true', '1', 'alle', ' Alle ', 'kritisch', 'KRITISCH'):
            w = wert.strip().lower()
            with mock.patch.dict(os.environ, {'V4_VERIFY_MODE': wert}):
                for typ in typen:
                    erwartet = w == 'alle' or (w == 'kritisch' and typ in orch._VERIFY_KRITISCHE_TYPEN)
                    self.assertEqual(orch._verify_scope_matches(typ), erwartet, f'V4_VERIFY_MODE={wert!r}, {typ}')
        ohne = {k: v for k, v in os.environ.items() if k not in ('V4_FAKTENBLATT', 'V4_VERIFY_MODE')}
        with mock.patch.dict(os.environ, ohne, clear=True):
            self.assertTrue(orch._faktenblatt_an('karte'))          # Vorgabe on
            self.assertFalse(orch._verify_scope_matches('karte'))   # Vorgabe off


if __name__ == '__main__':
    unittest.main()
