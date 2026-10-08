"""Legende vor Alltagswissen (Oktober 2026): sichert die Regel im gerenderten Prompt ab.

Anlass: Eine Karte mit Legende wies eine hellblaue Fläche einem Eigentümer zu; das
Modell gab die Legende richtig wieder und nannte die Fläche trotzdem Gewässer.
Die allgemeine Regel steht genau einmal im gemeinsamen Block LESBARER TEXT UND
LEGENDE der Datengrafiken. Der Karten-Prompt wendet sie im AUFTRAG auf Flächen und
Bänder an, und das innere Inventar der Karte ordnet jede Fläche ihrem
Legendeneintrag zu. Fotos bekommen den Block nicht. Der Prüfpass (Chatbot-Speicherweg)
beanstandet Deutungen gegen die Legende, und beim Neu-Generieren ist der bisherige Text
der KI Abgrenzungs-Vorlage, kein Beleg.

Nacharbeit nach der Prüfung (08.10.2026): Der Karten-Wortlaut bleibt wie gemessen (eine
entwürfelte Fassung hielt die Legende schlechter). Was der Nutzer selbst geschrieben oder ergänzt hat, bleibt beim
Neu-Generieren Beleg (herkunft_vorlage). Unbekannte Einträge der Typ-Listen in
V4_FAKTENBLATT und V4_VERIFY_MODE stehen einmal als Warnung im Log. Im Faktenblatt
kommt die Kategorie eines Ortes nur noch aus der Legende, nicht aus seinem Aussehen.

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

    def test_karten_wortlaut_bleibt_wie_gemessen(self):
        """Nacharbeit 08.10.2026: Eine entwürfelte Fassung (Regel nur im Block, AUFTRAG als allgemeiner
        Halbsatz) hielt die Legende mit Faktenblatt schlechter (Karten-Falle 10 von 22 falsch gegenüber 1 von 12).
        Der gemessene Wortlaut bleibt; ändern nur mit neuer Messung (prompts/ARCHITEKTUR.md)."""
        p = ' '.join(_combo('karte', 'karte').split())
        self.assertIn('Die Bedeutung aus der Legende ist die ganze Aussage über dieses Element; eine zweite '
                      'Deutung nach dem Aussehen kommt nicht hinzu', p)
        self.assertIn('eine Farbe nach Alltagsbedeutung statt nach Legende gelesen', p)

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
        self.assertIn('auch wenn der Legendeneintrag nur einen Eigentümer oder einen Planungsstand nennt', p)

    def test_neu_generieren_vorlage_ist_kein_beleg(self):
        from pipelines.v4 import orchestrator as orch
        ki = 'Karte zum Landabtausch: Ein Fluss teilt das Gebiet.'
        for s in (orch._variation_suffix(ki), orch._variation_suffix(ki, ki)):
            flach = ' '.join(s.split())
            self.assertIn('Den bisherigen Text hat die KI geschrieben. Er zeigt dir nur, wovon sich die neue '
                          'Fassung abheben soll, und ist kein Beleg', flach)
            # eindeutig, wem widersprochen wird, und Unbelegtes faellt ausdruecklich weg
            self.assertIn('Was Bild, Legende oder Kontext widerspricht, ersetzt du durch die belegte Angabe, '
                          'und was sich dort nicht belegen lässt, fällt weg.', flach)
            self.assertNotIn('Formulierungsvorlage', flach)   # widersprach der Bitte um andere Formulierung
            self.assertNotIn('Faktenlage bleibt identisch', flach)
            self.assertIn('Fotomontage-/Collage-Kennzeichnung', flach)  # belegte Kernfakten bleiben geschuetzt
            self.assertNotIn('Nutzer selbst geschrieben', flach)
        self.assertEqual(orch._variation_suffix(''), '')

    def test_herkunft_vorlage(self):
        from pipelines.v4.orchestrator import herkunft_vorlage
        ki = 'Karte zum Landabtausch im Areal Beispielfeld: Ein Fluss teilt das Gebiet von Nord nach Süd.'
        self.assertEqual(herkunft_vorlage(ki, ki), ('ki', []))
        self.assertEqual(herkunft_vorlage(ki, None), ('ki', []))             # Aufrufer kennt den KI-Text nicht
        self.assertEqual(herkunft_vorlage('  ' + ki.replace(' ', '  ') + '\n', ki), ('ki', []))   # nur Leerzeichen
        self.assertEqual(herkunft_vorlage(ki.replace(' von Nord nach Süd', ''), ki), ('ki', []))  # nur gestrichen
        self.assertEqual(herkunft_vorlage(ki.replace('Beispielfeld:', 'Beispielfeld,').replace('Süd.', 'Süd'), ki),
                         ('ki', []))                                                            # nur Satzzeichen
        self.assertEqual(herkunft_vorlage('Mein eigener Text zur Karte.', ''), ('nutzer', []))   # nie generiert
        self.assertEqual(herkunft_vorlage('Plan des Planungsbüros Muster für die Gemeindeversammlung am 14. '
                                          'November mit neuer Parzellierung.', ki)[0], 'nutzer')  # fast alles neu
        bearbeitet = ki.replace('Beispielfeld:', 'Beispielfeld, Vorlage für die Gemeindeversammlung am 14. November:')
        self.assertEqual(herkunft_vorlage(bearbeitet, ki),
                         ('bearbeitet', ['Vorlage für die Gemeindeversammlung am 14. November']))
        self.assertEqual(herkunft_vorlage(ki.replace('Fluss', 'Gemeindeweg'), ki), ('bearbeitet', ['Gemeindeweg']))
        self.assertEqual(herkunft_vorlage('', ki), ('ki', []))

    def test_neu_generieren_nutzerangaben_bleiben_beleg(self):
        from pipelines.v4 import orchestrator as orch
        ki = 'Karte zum Landabtausch im Areal Beispielfeld: Ein Fluss teilt das Gebiet von Nord nach Süd.'
        bearbeitet = ki.replace('Beispielfeld:', 'Beispielfeld, Vorlage für die Gemeindeversammlung am 14. November:')
        flach = ' '.join(orch._variation_suffix(bearbeitet, ki).split())
        self.assertIn('Den bisherigen Text hat die KI geschrieben und der Nutzer danach bearbeitet. Ergänzt oder '
                      'geändert hat er: „Vorlage für die Gemeindeversammlung am 14. November“.', flach)
        # Steve 08.10.: von Hand Ergaenztes bleibt inhaltlich, auch wenn das Bild es nicht zeigt
        self.assertIn('Diese Angaben übernimmst du inhaltlich in BEIDE Felder der neuen Fassung, auch wenn das '
                      'Bild sie nicht zeigt (etwa einen Namen oder einen Anlass); nur die Formulierung darf sich '
                      'ändern. Sie entfallen nur, wo Bild oder Legende ihnen widersprechen', flach)
        self.assertIn('Der übrige Text zeigt dir nur, wovon sich die neue Fassung abheben soll, und ist kein Beleg',
                      flach)
        self.assertNotIn('wie Angaben im Kontext', flach)   # sonst griffe die Namensregel fuer Kontext
        eigen = ' '.join(orch._variation_suffix('Unser Lageplan für die Versammlung.', '').split())
        self.assertIn('Den bisherigen Text hat der Nutzer selbst geschrieben. Seine Angaben übernimmst du '
                      'inhaltlich in BEIDE Felder', eigen)
        self.assertNotIn('kein Beleg', eigen)

    def test_nutzerstellen_begrenzt(self):
        from pipelines.v4 import orchestrator as orch
        ki = ' '.join(f'Wort{i}' for i in range(400))
        bearbeitet = ki.replace('Wort10 ', 'Ergänzung ' * 150 + 'Wort10 ').replace('Wort300 ', 'Zusatz ' * 200)
        s = orch._variation_suffix(bearbeitet, ki)
        self.assertIn('…', s)
        self.assertLess(len(s), 3200)

    def test_pipeline_steps_nennen_herkunft(self):
        from unittest import mock
        from pipelines.v4 import orchestrator as orch
        from prompts.components.schemas import BeschreibungOutput
        aus = BeschreibungOutput(alt_text='Karte des Areals Beispielfeld mit zwei Plänen und Legende (fiktiv).',
                                 langbeschreibung='Langbeschreibung (fiktiv).', verwendete_inventar_items=[],
                                 nicht_verwendete_inventar_items=[], nicht_im_inventar=[])
        ki = 'Karte zum Landabtausch im Areal Beispielfeld: Ein Fluss teilt das Gebiet von Nord nach Süd.'
        with mock.patch.dict(os.environ, {'V4_FAKTENBLATT': 'off', 'V4_VERIFY_MODE': 'off'}), \
                mock.patch.object(orch, 'call_with_schema', return_value=aus) as aufruf:
            r = orch.generate_alt_text_v4('/tmp/fehlt.png', image_type_override='karte', previous_alt=ki + ' Ergänzt.',
                                          previous_alt_ki=ki, temperature=0.5)
        self.assertIn(',vorlage:bearbeitet', r['pipeline_steps'])
        self.assertIn('„Ergänzt.“', aufruf.call_args.kwargs['prompt'])
        with mock.patch.dict(os.environ, {'V4_FAKTENBLATT': 'off', 'V4_VERIFY_MODE': 'off'}), \
                mock.patch.object(orch, 'call_with_schema', return_value=aus):
            r = orch.generate_alt_text_v4('/tmp/fehlt.png', image_type_override='karte')
        self.assertNotIn('vorlage:', r['pipeline_steps'])

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

    def test_unbekannte_typen_einmal_gewarnt(self):
        """Tippfehler, Oberbegriffe und Mischwerte wirken wie bisher nicht, stehen aber einmal im Log."""
        from unittest import mock
        from pipelines.v4 import orchestrator as orch
        orch._SCHALTER_GEWARNT.clear()
        logname = 'pipelines.v4.orchestrator'
        for schalter, wert, pruefe, typ, wirkt, unbekannt in (
                ('V4_FAKTENBLATT', 'karten', orch._faktenblatt_an, 'karte', False, 'karten'),
                ('V4_FAKTENBLATT', 'karte,diagramm', orch._faktenblatt_an, 'karte', True, 'diagramm'),
                ('V4_VERIFY_MODE', 'foto', orch._verify_scope_matches, 'foto_event', False, 'foto'),
                ('V4_VERIFY_MODE', 'kritisch,karte', orch._verify_scope_matches, 'karte', True, 'kritisch'),
                ('V4_VERIFY_MODE', 'kritisch,karte', orch._verify_scope_matches, 'foto_event', False, 'kritisch'),
                ('V4_VERIFY_MODE', 'on,karte', orch._verify_scope_matches, 'karte', True, 'on'),
                ('V4_VERIFY_MODE', 'logo', orch._verify_scope_matches, 'logo', True, 'logo')):
            with mock.patch.dict(os.environ, {schalter: wert}):
                with self.assertLogs(logname, 'WARNING') as protokoll:
                    orch._SCHALTER_GEWARNT.discard((schalter, wert))
                    self.assertEqual(pruefe(typ), wirkt, f'{schalter}={wert}, {typ}')
                self.assertEqual(len(protokoll.records), 1)
                self.assertIn(unbekannt, protokoll.output[0])
                with mock.patch.object(orch.log, 'warning') as warnung:   # zweiter Aufruf: keine zweite Warnung
                    pruefe(typ)
                warnung.assert_not_called()
        # gueltige Listen bleiben still
        for schalter, wert, pruefe, typ in (('V4_FAKTENBLATT', 'karte', orch._faktenblatt_an, 'karte'),
                                            ('V4_FAKTENBLATT', 'karte, tabelle', orch._faktenblatt_an, 'tabelle'),
                                            ('V4_VERIFY_MODE', 'karte,foto_event', orch._verify_scope_matches, 'foto_event')):
            with mock.patch.dict(os.environ, {schalter: wert}), mock.patch.object(orch.log, 'warning') as warnung:
                self.assertTrue(pruefe(typ))
            warnung.assert_not_called()
        self.assertEqual(orch._VERIFY_TYPEN & orch._MINI_TYPES, frozenset())
        self.assertNotIn('foto', orch._VERIFY_TYPEN)
        self.assertLessEqual(orch._VERIFY_KRITISCHE_TYPEN, orch._VERIFY_TYPEN)

    def test_faktenblatt_karte_kategorie_nur_aus_legende(self):
        from pipelines.v4 import orchestrator as orch
        f = orch.KarteFakten(gebiet='Musterhausen', legende=['Hellblau: Gemeinde (Verwaltungsvermögen)'],
                             orte=[orch.KarteOrt(name='2101', kategorie='Gemeinde (Verwaltungsvermögen)', lage='Mitte'),
                                   orch.KarteOrt(name='Beispielweg', lage='am Südrand')],
                             lesbarkeit='gut')
        block = orch._faktenblatt_block(f, 'karte')
        flach = ' '.join(block.split())
        self.assertIn('Grundlage für jede Zahl, jede Bezeichnung und jede Zuordnung in Alt-Text und Langbeschreibung.',
                      flach)
        self.assertNotIn('jede Reihenfolge', flach)
        self.assertIn('  - 2101 (Legende: Gemeinde (Verwaltungsvermögen)), Mitte', block)
        self.assertIn('  - Beispielweg, am Südrand', block)
        felder = orch.KarteOrt.model_fields
        self.assertNotIn('Fluss', felder['lage'].description)
        self.assertNotIn('Symbol', felder['kategorie'].description)
        self.assertIn('leer, wenn kein Legendeneintrag passt', felder['kategorie'].description)


if __name__ == '__main__':
    unittest.main()
