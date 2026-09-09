"""Prompt-Hygiene: rendert alle Prompts und meldet Verstöße gegen docs/PROMPT-STANDARD.md.

Geprüft werden die Texte, die das Modell sieht: Combo-Prompts aller 13 Inventar-
Bildtypen, die drei Mini-Builder, der Klassifikator, der Prüfpass sowie die
Werte- und Zählblöcke. Wer eine Regel bricht, sieht hier sofort, wo.
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from collections import Counter

BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from prompts.builders.classification import build_classification_prompt  # noqa: E402
from prompts.builders.combo import build_combined_inventar_beschreibung_prompt  # noqa: E402
from prompts.builders.beschreibung import build_beschreibung_prompt_mini  # noqa: E402
from prompts.components.roles import SYSTEM_BESCHREIBUNG  # noqa: E402
from prompts.components.schemas import ClassificationOutput  # noqa: E402
from pipelines.v4 import orchestrator as orch  # noqa: E402


INVENTAR_TYPEN = {
    'foto_event': 'foto', 'foto_personen': 'foto', 'foto_objekte': 'foto', 'foto_essen': 'foto',
    'foto_landschaft': 'foto', 'foto_architektur': 'foto', 'illustration': 'illustration',
    'diagramm': 'diagramm', 'tabelle': 'tabelle', 'karte': 'karte', 'infografik': 'infografik',
    'screenshot': 'screenshot', 'strukturformel': 'strukturformel',
}
MINI_TYPEN = ('logo', 'icon', 'funktional')

# ASCII-Ersatzschreibungen, die im Modelltext nicht vorkommen dürfen (Standard: echte Umlaute).
ASCII_UMLAUTE = re.compile(r'\b(fuer|ueber|koennen|muessen|Groesse|moeglich|waehrend|zaehl\w*|Saetze|natuerlich|'
                           r'gehoert|hoechstens|praezis\w*|Beruehmt\w*|Rueck\w*|Hoehe|laesst|erlaeutert|'
                           r'Ausfuehr\w*|fuellen|Laenge|Buehne|Menue|oeffnen|Waehle|zusaetzlich|Wuerde|'
                           r'Muendung|Faelle|schaetz\w*|Woerter|Fliesstext|ausschliesslich|Massstab)\b')
ALTLASTEN = [
    r'\bSteve\b', r'\bMichael\b', r'\bKarbe\b', r'\bChatGPT\b', r'\bMistral\b', r'\bPixtral\b',
    r'\d{2}\.\d{2}\.20\d\d', r'Stand 20\d\d', r'seit 20\d\d',   # Datumsreste, keine Jahreszahlen in Beispielen
    r'FINAL CHECK', r'AUSGABE-SCHEMA', r'inventar\.lesbare_texte', r'\bOCR\b',
    r'Validator-Pass', r'Multi-Pass', r'Pass[- ][23]\b', r'Premium-Pipeline', r'Web-Scraper',
    r'LIZENZ_LOGOS_REGELN', r'STILREGELN Punkt', r'\bNIEMALS\b', r'Todsünde', r'Schritt 0',
    r'\bAcer\b', r'\bbudni\b',
]
FIKTIVE_MARKEN_AUSNAHMEN = ('Musterwerk', 'Beispiel AG', 'Beispiel Air')
# combo: 2400 seit 09.09.2026 (Stilregel Länge mit Telefon-Maßstab, Doppelpunkt-Klarstellung,
# zweites Diagramm-Beispiel mit drei Reihen; vorher lag foto_personen bei 1995, diagramm jetzt bei 2287).
LAENGEN_MAX_WOERTER = {'combo': 2400, 'mini': 700, 'klassifikator': 800, 'verify': 1200}


def _combo(effektiv: str, kontext: str = 'Abbildung 3: Beispiel') -> str:
    return build_combined_inventar_beschreibung_prompt(
        bildtyp_top=INVENTAR_TYPEN[effektiv], bildtyp_effective=effektiv,
        enriched_context=kontext, width=800, height=600,
    )


def _mini(typ: str) -> str:
    cls = ClassificationOutput(bildtyp=typ, konfidenz='hoch', ist_dekorativ=False,
                               original_alt_brauchbar=False, klassifikations_begruendung='Begründung für den Test')
    return build_beschreibung_prompt_mini(typ, cls, 'LINK-ZIEL: https://www.beispiel.de', 120, 40, original_alt='')


def _alle_prompts() -> dict[str, str]:
    p = {f'combo:{t}': _combo(t) for t in INVENTAR_TYPEN}
    p.update({f'mini:{t}': _mini(t) for t in MINI_TYPEN})
    p['klassifikator'] = build_classification_prompt('Kontext', 100, 50, original_alt='Logo Y')
    p['verify'] = orch._build_verify_prompt('Ein Alt-Text', enriched_context='Kontext', langbeschreibung='Lang', bildtyp='diagramm')
    p['system'] = SYSTEM_BESCHREIBUNG
    w = orch.WerteOutput(titel='Umsatz', diagrammtyp='Balken', achsen='0 bis 6', lesbarkeit='gut',
                         reihen=[orch.WerteReihe(name='A', punkte=[orch.WertePunkt(kategorie='2021', wert='2,5'),
                                                                     orch.WertePunkt(kategorie='2022', wert='4,4')])])
    p['werte_block'] = orch._werte_block(w)
    z = orch.ZaehlOutput(personen=[orch.ZaehlPerson(position='links', merkmal='blauer Blazer', sichtbarkeit='ganz')],
                         gruppen=[orch.ZaehlGruppe(bezeichnung='Schalen', anzahl=26, zaehlweise='exakt')])
    p['zaehl_block'] = orch._zaehl_block(z)
    for typ, text in orch._FAKTENBLATT_PROMPTS.items():
        p[f'faktenblatt_aufruf:{typ}'] = text
    f = orch.TabelleFakten(titel='Nährwerte', spaltenkoepfe=['Nährstoff', 'je 100 g'],
                           zeilen=[orch.TabelleZeile(bezeichnung='Energie', werte=['52 kcal'])], lesbarkeit='gut')
    p['faktenblatt_block'] = orch._faktenblatt_block(f, 'tabelle')
    return p


class PromptHygieneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = _alle_prompts()

    def _verstoesse(self, muster, text):
        return sorted({m.group(0) for m in re.finditer(muster, text)})

    def test_keine_ascii_umlaute(self):
        for name, text in self.prompts.items():
            treffer = self._verstoesse(ASCII_UMLAUTE, text)
            self.assertFalse(treffer, f'{name}: ASCII-Umlaute im Modelltext: {treffer[:12]}')

    def test_keine_altlasten(self):
        for name, text in self.prompts.items():
            for muster in ALTLASTEN:
                treffer = self._verstoesse(muster, text)
                self.assertFalse(treffer, f'{name}: Altlast {muster!r}: {treffer[:5]}')

    def test_abschnittstitel_nur_einmal(self):
        """Ein Abschnittstitel (Zeile nur aus Großbuchstaben) darf je Prompt nur einmal vorkommen."""
        titel_muster = re.compile(r'^(?:[A-ZÄÖÜ][A-ZÄÖÜ\- ]{3,}[A-ZÄÖÜ])(?: \([^)]*\))?\s*$', re.M)
        for name, text in self.prompts.items():
            zaehler = Counter(m.group(0).strip() for m in titel_muster.finditer(text))
            doppelt = [t for t, n in zaehler.items() if n > 1 and t not in ('BEISPIELE',)]
            self.assertFalse(doppelt, f'{name}: doppelte Abschnitte {doppelt}')

    def test_laengen(self):
        for name, text in self.prompts.items():
            art = name.split(':')[0]
            grenze = LAENGEN_MAX_WOERTER.get(art)
            if grenze:
                self.assertLessEqual(len(text.split()), grenze, f'{name}: {len(text.split())} Wörter, erlaubt {grenze}')

    def test_keine_grossschreib_kaskaden(self):
        """Höchstens zehn Wörter ganz in Großbuchstaben (ab fünf Zeichen) außerhalb der Abschnittstitel."""
        titel_muster = re.compile(r'^(?:[A-ZÄÖÜ][A-ZÄÖÜ\- ]{3,}[A-ZÄÖÜ])(?: \([^)]*\))?\s*$', re.M)
        for name, text in self.prompts.items():
            ohne_titel = titel_muster.sub('', text)
            grossworte = [w for w in re.findall(r'\b[A-ZÄÖÜ]{5,}\b', ohne_titel)
                          if w not in ('WCAG', 'UNESCO', 'BILDDATEN', 'JSON', 'PFLICHT', 'OPTIONAL', 'BILDTYP', 'BILDGRÖSSE', 'BILDGROESSE', 'ORIGINAL', 'LINKZIEL', 'ABGELESENE', 'WERTE', 'AUFGEZÄHLT', 'FAKTENBLATT')]
            self.assertLessEqual(len(grossworte), 10, f'{name}: Großschreib-Kaskade {grossworte[:15]}')

    def test_pflichtabschnitte_combo(self):
        for t in INVENTAR_TYPEN:
            text = self.prompts[f'combo:{t}']
            for abschnitt in ('BELEGREGELN', 'ARBEITSWEISE', 'BILDTYP:', 'AUFTRAG', 'ALT-TEXT', 'LANGBESCHREIBUNG', 'STILREGELN', 'BEISPIELE', 'KONTEXT'):
                self.assertIn(abschnitt, text, f'combo:{t}: Abschnitt {abschnitt} fehlt')


if __name__ == '__main__':
    unittest.main()
