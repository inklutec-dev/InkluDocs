"""Eval-Runner für die v4-Pipeline.

Liest ein manifest.yaml mit Test-Bildern + Erwartungen, jagt jedes Bild
durch generate_alt_text_v4(), prüft den Output gegen die deklarierten
Checks und gibt einen Pass/Fail-Report aus.

Ausführung im Container:
  docker exec -w /app inkludocs-staging python3 -m eval.runner

Optional mit eigenem Manifest:
  docker exec -w /app inkludocs-staging python3 -m eval.runner /app/eval/korpus/eigener.json

Exit-Code:
  0  — alle Tests bestanden
  1  — mindestens ein Test fehlgeschlagen
  2  — Pipeline-Fehler (kein Test-Result)

Kosten-Hinweis:
  Jeder Test = 4 Mistral-API-Calls (Klassifikation + Inventar + Beschreibung
  + Validierung). Bei 10 Bildern entstehen ~40 Calls, ~0.50-1.00 EUR.
  Bei systematischen Eval-Runs nach jedem Constraint-Update einplanen.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json as _json_loader

from pipelines.v4.mistral_client import MistralCallError
from pipelines.v4.orchestrator import generate_alt_text_v4

DEFAULT_MANIFEST = Path(__file__).resolve().parent / 'korpus' / 'manifest.json'


@dataclass
class CheckResult:
    """Ergebnis einer einzelnen Check-Regel (must_contain etc.)."""
    name: str
    passed: bool
    detail: str = ''


@dataclass
class TestResult:
    """Ergebnis eines kompletten Test-Bilds."""
    test_id: str
    bildtyp_actual: str = ''
    sub_typ_actual: str | None = None
    pipeline_steps: str = ''
    duration_seconds: float = 0.0
    pipeline_error: str = ''
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.pipeline_error and all(c.passed for c in self.checks)

    @property
    def failure_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


def _check_must_contain(text: str, needles: list[str]) -> list[CheckResult]:
    results = []
    text_lower = text.lower()
    for needle in needles:
        present = needle.lower() in text_lower
        results.append(CheckResult(
            name=f'must_contain:"{needle}"',
            passed=present,
            detail='gefunden' if present else 'FEHLT im Output',
        ))
    return results


def _check_must_not_contain(text: str, needles: list[str]) -> list[CheckResult]:
    results = []
    text_lower = text.lower()
    for needle in needles:
        absent = needle.lower() not in text_lower
        results.append(CheckResult(
            name=f'must_not_contain:"{needle}"',
            passed=absent,
            detail='nicht gefunden (gut)' if absent else 'TROTZDEM IM OUTPUT (BUG)',
        ))
    return results


def _check_length(text: str, min_len: int | None, max_len: int | None) -> list[CheckResult]:
    results = []
    if min_len is not None:
        ok = len(text) >= min_len
        results.append(CheckResult(
            name=f'min_alt_length:{min_len}',
            passed=ok,
            detail=f'actual={len(text)}',
        ))
    if max_len is not None:
        ok = len(text) <= max_len
        results.append(CheckResult(
            name=f'max_alt_length:{max_len}',
            passed=ok,
            detail=f'actual={len(text)}',
        ))
    return results


def _check_bildtyp(actual: str, expected: str) -> CheckResult:
    return CheckResult(
        name=f'bildtyp_erwartung:{expected}',
        passed=actual == expected,
        detail=f'actual={actual}',
    )


def _check_sub_typ(actual: str | None, expected: str) -> CheckResult:
    return CheckResult(
        name=f'sub_typ_erwartung:{expected}',
        passed=actual == expected,
        detail=f'actual={actual}',
    )


def run_single_test(entry: dict) -> TestResult:
    """Führt einen Test aus dem manifest.yaml aus."""
    test_id = entry['id']
    result = TestResult(test_id=test_id)

    print(f'\n--- Test: {test_id} ---')
    t0 = time.time()
    try:
        pipeline_result = generate_alt_text_v4(
            image_path=entry['image_path'],
            enriched_context=entry.get('enriched_context', ''),
            width=entry.get('width', 0),
            height=entry.get('height', 0),
            original_alt=entry.get('original_alt', ''),
        )
    except MistralCallError as e:
        result.pipeline_error = f'MistralCallError: {e}'
        result.duration_seconds = time.time() - t0
        print(f'  PIPELINE-FEHLER: {e}')
        return result
    except Exception as e:
        result.pipeline_error = f'{type(e).__name__}: {e}'
        result.duration_seconds = time.time() - t0
        print(f'  PIPELINE-FEHLER: {e}')
        return result

    result.duration_seconds = time.time() - t0
    result.bildtyp_actual = pipeline_result['bildtyp']
    result.pipeline_steps = pipeline_result['pipeline_steps']

    # Sub-Typ extrahieren falls vorhanden
    if pipeline_result.get('inventar_json'):
        try:
            inv = json.loads(pipeline_result['inventar_json'])
            result.sub_typ_actual = inv.get('foto_subtyp')
        except (json.JSONDecodeError, KeyError):
            pass

    # Output-Text für Checks zusammensetzen
    full_text = pipeline_result.get('alt_text', '') + ' ' + pipeline_result.get('langbeschreibung', '')

    checks_cfg = entry.get('checks', {})
    if 'must_contain' in checks_cfg:
        result.checks.extend(_check_must_contain(full_text, checks_cfg['must_contain']))
    if 'must_not_contain' in checks_cfg:
        result.checks.extend(_check_must_not_contain(full_text, checks_cfg['must_not_contain']))
    result.checks.extend(_check_length(
        pipeline_result.get('alt_text', ''),
        checks_cfg.get('min_alt_length'),
        checks_cfg.get('max_alt_length'),
    ))

    # Bildtyp-/Sub-Typ-Checks (strict ist optional, default false)
    if checks_cfg.get('bildtyp_erwartung_strict') and 'bildtyp_erwartung' in entry:
        result.checks.append(_check_bildtyp(result.bildtyp_actual, entry['bildtyp_erwartung']))
    if checks_cfg.get('sub_typ_erwartung_strict') and 'sub_typ_erwartung' in entry:
        result.checks.append(_check_sub_typ(result.sub_typ_actual, entry['sub_typ_erwartung']))

    # Compact print
    print(f'  bildtyp={result.bildtyp_actual} sub={result.sub_typ_actual} dauer={result.duration_seconds:.1f}s')
    print(f'  alt_text: {pipeline_result.get("alt_text", "")[:120]}...')
    for c in result.checks:
        marker = 'OK' if c.passed else 'FAIL'
        print(f'  [{marker}] {c.name} — {c.detail}')

    return result


def run_manifest(manifest_path: Path) -> list[TestResult]:
    """Liest ein manifest.yaml und führt alle Tests aus."""
    if not manifest_path.exists():
        print(f'FEHLER: Manifest nicht gefunden: {manifest_path}')
        sys.exit(2)
    with manifest_path.open('r', encoding='utf-8') as fh:
        manifest = _json_loader.load(fh)
    entries = manifest.get('eval_korpus', [])
    if not entries:
        print('WARN: Manifest enthält keine Einträge in eval_korpus.')
        return []
    print(f'Eval-Run: {len(entries)} Tests aus {manifest_path}')
    return [run_single_test(e) for e in entries]


def print_summary(results: list[TestResult]) -> int:
    """Druckt zusammenfassenden Report. Gibt Exit-Code zurück."""
    print('\n' + '=' * 70)
    print('EVAL-SUMMARY')
    print('=' * 70)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pipeline_errors = sum(1 for r in results if r.pipeline_error)
    print(f'Tests gesamt:       {total}')
    print(f'Bestanden:          {passed}')
    print(f'Fehlgeschlagen:     {total - passed}')
    print(f'Pipeline-Fehler:    {pipeline_errors}')
    print()
    for r in results:
        if r.passed:
            print(f'  PASS {r.test_id} ({r.duration_seconds:.1f}s)')
        elif r.pipeline_error:
            print(f'  ERROR {r.test_id}: {r.pipeline_error}')
        else:
            print(f'  FAIL {r.test_id} — {r.failure_count} von {len(r.checks)} Checks fehlgeschlagen')
            for c in r.checks:
                if not c.passed:
                    print(f'         - {c.name} — {c.detail}')

    if pipeline_errors > 0:
        return 2
    if passed < total:
        return 1
    return 0


def main(argv: list[str]) -> int:
    manifest_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_MANIFEST
    if not os.environ.get('MISTRAL_API_KEY'):
        print('FEHLER: MISTRAL_API_KEY nicht gesetzt — Eval-Run nicht möglich.')
        return 2
    results = run_manifest(manifest_path)
    return print_summary(results)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
