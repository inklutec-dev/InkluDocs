"""Wiederverwendete Hilfsfunktionen für die Prompt-Builder der v4-Pipeline.

Drei Funktionen:
- user_hint_block: Optionalen Nutzer-Hinweis-Block formatieren
  (Variante 3 aus dem Alt-Text-Workflow-Konzept).
- extract_link_target_from_context: Link-Ziel aus enriched_context ziehen
  (für icon/logo wenn das Bild verlinkt ist).
- load_examples: Few-Shot-Beispiele aus prompts/components/examples/<bildtyp>/
  laden und für den Prompt formatieren.

Examples-Verzeichnis-Struktur:
  prompts/components/examples/<bildtyp>/good_01.json
  prompts/components/examples/<bildtyp>/good_02.json
  prompts/components/examples/<bildtyp>/bad_01.json
  ...

Jede JSON-Datei enthält ein Beispiel-Objekt mit Feldern wie alt_text,
langbeschreibung, begruendung etc. (Format ist bildtyp-spezifisch — siehe
§7.x der Architektur-Doku.)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Wurzel der Examples-Bibliothek — relativ zum builders-Modul.
# prompts/builders/helpers.py → ../components/examples/
_EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / 'components' / 'examples'


def user_hint_block(user_hint: Optional[str]) -> str:
    """Formatiert den optionalen Nutzer-Hinweis-Block für den Prompt.

    Wird in jedem Beschreibungs-Builder verwendet, damit der Nutzer
    (Workflow-Variante 3 aus den Alt-Text-Eingabe-Modi) dem Modell
    Hinweise geben kann, die HOHE PRIORITÄT haben.
    """
    if not user_hint:
        return ''
    return (
        '\n\nNUTZER-HINWEIS (HOHE PRIORITÄT — Nutzer sagt was über das Bild):\n'
        + user_hint
        + '\nWenn der Nutzer-Hinweis deiner Wahrnehmung widerspricht: Hinweis hat Vorrang. '
        + 'Erkläre kurz im Output warum.'
    )


_LINK_TARGET_PATTERNS = [
    # Häufige Patterns wie sie aus dem PDF-/Web-Kontext kommen.
    re.compile(r'(?:LINK[-_ ]?ZIEL|Verlinkt auf|Link zu|URL|Href)[: ]\s*(\S+)', re.IGNORECASE),
    re.compile(r'<a\s+href="([^"]+)"', re.IGNORECASE),
]


def extract_link_target_from_context(enriched_context: str) -> Optional[str]:
    """Extrahiert ein Link-Ziel aus dem enriched_context falls vorhanden.

    Wird von logo- und icon-Buildern genutzt, um verlinkte Bilder mit
    'Link zu …' im Alt-Text zu kennzeichnen. Gibt None zurück wenn
    kein Link-Ziel gefunden — der Builder fügt dann keine Link-Zeile ein.
    """
    if not enriched_context:
        return None
    for pattern in _LINK_TARGET_PATTERNS:
        m = pattern.search(enriched_context)
        if m:
            return m.group(1).strip()
    return None


@dataclass(frozen=True)
class Examples:
    """Sammlung von Few-Shot-Beispielen für einen Bildtyp.

    good_examples: positive Beispiele (Inventar → guter Output).
    bad_examples:  Anti-Pattern-Beispiele (typische Fehler + Begründung).
    """

    bildtyp: str
    good_examples: list[dict] = field(default_factory=list)
    bad_examples: list[dict] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Formatiert die Beispiele für die Aufnahme in den Prompt.

        Pro Beispiel kommt ein Header (POSITIVES BEISPIEL / NEGATIV-BEISPIEL)
        plus das vollständige JSON. Falls keine Beispiele vorhanden sind:
        Hinweis-Text 'noch keine Beispiele kuratiert' (zwingt den Eval-
        Workflow nicht, sondern signalisiert das dem Modell ehrlich).
        """
        if not self.good_examples and not self.bad_examples:
            return f'(Noch keine Few-Shot-Beispiele für Bildtyp "{self.bildtyp}" kuratiert.)'

        lines: list[str] = []
        for i, ex in enumerate(self.good_examples, start=1):
            lines.append(f'POSITIVES BEISPIEL {i}:')
            lines.append(json.dumps(ex, ensure_ascii=False, indent=2))
            lines.append('')
        for i, ex in enumerate(self.bad_examples, start=1):
            lines.append(f'ANTI-PATTERN-BEISPIEL {i} (NICHT so machen):')
            lines.append(json.dumps(ex, ensure_ascii=False, indent=2))
            lines.append('')
        return '\n'.join(lines).rstrip()


def load_examples(bildtyp: str) -> Examples:
    """Lädt Few-Shot-Beispiele für einen Bildtyp aus dem Examples-Verzeichnis.

    Konvention:
    - good_*.json → good_examples
    - bad_*.json  → bad_examples
    - andere Dateinamen werden ignoriert

    Verzeichnis fehlt → leere Examples zurück (kein Crash, weil Beispiele
    während der Kurations-Phase fehlen können — siehe C1-Status).
    Defekte JSON → wird übersprungen mit Warnung im stdout (nicht crashen,
    weil eine fehlerhafte Beispiel-Datei nicht die ganze Pipeline kippen darf).
    """
    bildtyp_dir = _EXAMPLES_ROOT / bildtyp
    if not bildtyp_dir.is_dir():
        return Examples(bildtyp=bildtyp)

    good: list[dict] = []
    bad: list[dict] = []
    for path in sorted(bildtyp_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            print(f'WARN load_examples: defekte JSON {path}: {e}')
            continue
        if path.name.startswith('good'):
            good.append(data)
        elif path.name.startswith('bad'):
            bad.append(data)
    return Examples(bildtyp=bildtyp, good_examples=good, bad_examples=bad)


# V4_PROMPT_MODE-Schalter (08.05.2026):
# 'full' = volle Mistral-Erziehung (Hedge-Wort-Verbote, harte Final-Checks).
# 'lean' = schlank fuer Sonnet (Sonnet halluziniert kaum, braucht keine Drillregeln).
# Default: 'lean' wenn LLM_PROVIDER=bedrock, sonst 'full'.
# So bleibt Mistral-Pipeline ohne Aenderung voll erzieherisch, Bedrock laeuft lean.
def resolve_prompt_mode() -> str:
    explicit = os.environ.get('V4_PROMPT_MODE', '').strip().lower()
    if explicit in ('full', 'lean'):
        return explicit
    provider = os.environ.get('LLM_PROVIDER', 'mistral').lower().strip()
    return 'lean' if provider == 'bedrock' else 'full'
