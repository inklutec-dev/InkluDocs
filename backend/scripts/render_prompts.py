"""Render alle Prompt-Builder mit Demo-Werten als Markdown-Snapshots.

Aufruf (im Container):
    docker exec -w /app inkludocs-staging python3 -m scripts.render_prompts

Ergebnis:
    /app/prompts/snapshots/*.md — pro Builder + Modus eine Datei.

Zweck:
    - Audit-Tauglichkeit: Co-Entwickler, Karbe, andere Kunden koennen die
      kompletten Prompts lesen ohne Python-Code-Verstaendnis.
    - Diff-Lesbarkeit: bei Builder-Aenderungen ist der Snapshot-Diff direkt
      der Prompt-Diff (statt nur Code-Diff).
    - Verkaufstauglichkeit: zeigt dass die Prompt-Architektur durchdacht ist.

Demo-Werte sind realistisch (Workshop-Foto-Setting) aber generisch genug
fuer alle Bildtypen. Wer praezisere Snapshots braucht, kopiert das Skript
und passt die Demo-Konstanten an.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

# Pfad-Hack: Skript liegt unter /app/scripts/, muss /app im sys.path haben.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from prompts.components.schemas import (  # noqa: E402
    BeschreibungOutput,
    ClassificationOutput,
    InventarOutput,
    ObjektInBild,
    PersonInBild,
    TextInBild,
)

# ============================================================================
# DEMO-WERTE: je Bildtyp ein passender Kontext, damit jeder Snapshot den Prompt so
# zeigt, wie er im Betrieb für ein Bild dieser Art aussieht. Erfundene Namen.
# ============================================================================

DEMO_DATE = datetime.date.today().isoformat()

DEMO_WIDTH = 1280
DEMO_HEIGHT = 720

DEMO_CONTEXT_MINIMAL = ''
DEMO_CONTEXT_RICH = (
    'Workshop-Bericht: Inklusion in der digitalen Arbeitswelt. Am 5. Mai fand bei der '
    'Musterwerk GmbH ein eintägiger Workshop zur barrierefreien Software-Entwicklung statt. '
    'Teilnehmende waren Entwicklerinnen und Entwickler aus drei Partnerunternehmen.'
)

DEMO_KONTEXT_JE_TYP = {
    'foto_event': DEMO_CONTEXT_RICH,
    'foto_personen': 'Bildunterschrift: Anna Reimers, Gründerin der Musterwerk GmbH, in ihrem Büro in Bonn.',
    'foto_objekte': 'Produktkatalog Musterwerk, Seite 12: Handgefertigte Schalen aus der Serie Nordlicht.',
    'foto_essen': 'Speisekarte Beispiel-Bistro: Pizza Margherita, Tomaten, Mozzarella, Basilikum.',
    'foto_landschaft': 'Reisebericht: Wanderung im Berner Oberland, dritter Tag.',
    'foto_architektur': 'Jahresbericht Beispiel AG: Der neue Verwaltungsbau in Hannover wurde im Mai bezogen.',
    'illustration': 'Ratgeber Homeoffice, Kapitel 2: Den Arbeitsplatz einrichten.',
    'diagramm': 'Abbildung 3: Umsatzentwicklung 2021 bis 2023 nach Sparten, Angaben in Millionen Euro.',
    'tabelle': 'Tabelle 2: Nährwerte je 100 Gramm.',
    'karte': 'Abbildung 5: Beratungsstellen in Nordrhein-Westfalen, Stand Januar.',
    'infografik': 'Schaubild: So läuft die Antragstellung in vier Schritten.',
    'screenshot': 'Anleitung Musterwerk Projektverwaltung, Schritt 3: Projekt anlegen.',
    'strukturformel': 'Lehrbuch Organische Chemie, Kapitel 7: Acetylsalicylsäure.',
}

DEMO_ORIGINAL_ALT_LEER = ''
DEMO_ORIGINAL_ALT_BRAUCHBAR = 'Workshop-Foto Inklusion'
DEMO_ORIGINAL_ALT_UNBRAUCHBAR = 'IMG_2345.jpg'

DEMO_USER_HINT_NONE = None
DEMO_USER_HINT_SET = (
    'Das ist unser Workshop am 5. Mai mit der Beispiel AG als Kooperationspartner. '
    'Bitte den Workshop-Charakter betonen.'
)

# Inventar nur noch für die Inventar-Snapshots (Analyse-Schritt); der Produktionsweg ist
# der Combo-Aufruf, der kein Inventar-JSON mehr rendert.
DEMO_INVENTAR_FOTO_EVENT = InventarOutput(
    foto_subtyp='foto_event',
    personen=[
        PersonInBild(position='vorn links', haltung='stehend', blickrichtung='zur Präsentation', kleidungs_charakter='geschäftlich leger'),
        PersonInBild(position='Mitte', haltung='stehend', blickrichtung='zur Kamera', kleidungs_charakter='geschäftlich leger'),
        PersonInBild(position='hinten rechts', haltung='sitzend', blickrichtung='zur Präsentation', kleidungs_charakter='leger'),
        PersonInBild(position='Mitte rechts', haltung='stehend', kleidungs_charakter='geschäftlich leger'),
    ],
    objekte=[
        ObjektInBild(beschreibung='Projektionsfläche mit hellem Lichtkegel', position='hinten Mitte', sicherheit='hoch', moegliche_identifikationen=['Beamer-Projektion']),
        ObjektInBild(beschreibung='rechteckige weiße Karten an Personen befestigt', position='auf Brusthöhe der Personen', sicherheit='hoch', moegliche_identifikationen=['Namensschilder']),
        ObjektInBild(beschreibung='Tisch mit Getränkeflaschen und Gläsern', position='rechter Bildrand', sicherheit='hoch', moegliche_identifikationen=['Catering-Tisch']),
    ],
    lesbare_texte=[
        TextInBild(inhalt='Workshop Inklusion', typ='überschrift', vollstaendigkeit='vollständig'),
    ],
    setting={'raum_charakter': 'Seminarraum', 'beleuchtung': 'gedämpft, Projektionslicht', 'dominante_farben': 'blau, weiß, grau', 'ungefaehre_szene': 'Vortragssituation mit Publikum'},
    handlung='Präsentation vor stehendem und sitzendem Publikum',
    halluzinations_warnung=['Namensschilder nicht lesbar, keine Identifikationen ableiten.'],
    inventar_konfidenz_gesamt='hoch',
)

DEMO_CLASSIFICATION_LOGO = ClassificationOutput(bildtyp='logo', konfidenz='hoch', ist_dekorativ=False, original_alt_brauchbar=True,
                                                 klassifikations_begruendung='Erkennbares Markenlogo ohne weiteren Bildinhalt.')
DEMO_CLASSIFICATION_ICON = ClassificationOutput(bildtyp='icon', konfidenz='hoch', ist_dekorativ=False, original_alt_brauchbar=False,
                                                 klassifikations_begruendung='Kleines funktionales Symbol (Lupe) ohne Beschriftung.')
DEMO_CLASSIFICATION_FUNKTIONAL = ClassificationOutput(bildtyp='funktional', konfidenz='hoch', ist_dekorativ=False, original_alt_brauchbar=False,
                                                       klassifikations_begruendung='Pfeil-Element mit Zustand nächste Seite, Navigation.')


# ============================================================================
# UTILITIES
# ============================================================================

def _builder_source_link(builder_func) -> str:
    """Liefert 'path/to/file.py:LINE' fuer den Builder-Header."""
    try:
        import inspect
        src_file = Path(inspect.getsourcefile(builder_func)).resolve()
        # Make path relative to /app, wenn moeglich
        try:
            rel = src_file.relative_to(_BACKEND_ROOT)
            src_file = rel
        except ValueError:
            pass
        _, line = inspect.getsourcelines(builder_func)
        return f'{src_file}:{line}'
    except Exception:
        return '(source unknown)'


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    """Setzt ENV-Vars und liefert die alten Werte zurueck (fuer Cleanup)."""
    previous: dict[str, str | None] = {}
    for k, v in updates.items():
        previous[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for k, v in previous.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _write_snapshot(
    out_dir: Path,
    filename: str,
    title: str,
    builder_ref: str,
    mode_info: dict[str, str],
    demo_values: dict[str, str],
    prompt_text: str,
) -> Path:
    """Schreibt einen einzelnen Snapshot in Markdown-Form."""
    path = out_dir / filename

    header = [
        f'# {title}',
        '',
        f'- **Builder:** `{builder_ref}`',
        f'- **Generiert:** {DEMO_DATE}',
    ]
    if mode_info:
        header.append('- **ENV / Modus:**')
        for k, v in mode_info.items():
            header.append(f'  - `{k}` = `{v}`')
    if demo_values:
        header.append('- **Demo-Werte:**')
        for k, v in demo_values.items():
            header.append(f'  - {k}: {v}')
    header.extend([
        '',
        '---',
        '',
        '```text',
        prompt_text,
        '```',
        '',
    ])

    path.write_text('\n'.join(header), encoding='utf-8')
    return path


# ============================================================================
# RENDER-AUFRUFE
# ============================================================================

def render_all(out_dir: Path) -> list[Path]:
    """Rendert alle Prompts, die das Modell im Betrieb sieht. Liefert die geschriebenen Pfade."""
    out_dir.mkdir(parents=True, exist_ok=True)

    from prompts.builders import (
        build_beschreibung_prompt_mini,
        build_classification_prompt,
        build_combined_inventar_beschreibung_prompt,
        build_inventar_prompt,
    )
    from prompts.components.roles import SYSTEM_BESCHREIBUNG
    from pipelines.v4 import orchestrator as orch

    written: list[Path] = []
    common_demo = {'width × height': f'{DEMO_WIDTH} × {DEMO_HEIGHT}'}
    env = {'V4_PASS_MODE': 'lean', 'V4_PROMPT_MODE': '', 'LLM_PROVIDER': 'bedrock', 'V4_PROMPT_CACHE': 'off'}

    # 0) System-Prompt (gilt für alle Beschreibungs- und Prüfaufrufe)
    written.append(_write_snapshot(out_dir, filename='00_system.md', title='System-Prompt der Bildbeschreibung',
                                   builder_ref='prompts/components/roles.py', mode_info={}, demo_values={},
                                   prompt_text=SYSTEM_BESCHREIBUNG))

    # 1) Klassifikator in drei Varianten
    for filename, title, ctx, hint, note in (
        ('01_classification.lean.md', 'Klassifikator', DEMO_CONTEXT_RICH, DEMO_USER_HINT_NONE, 'Dokumentkontext (Workshop-Bericht)'),
        ('01_classification.lean.frontend-upload.md', 'Klassifikator, Einzelbild ohne Kontext', DEMO_CONTEXT_MINIMAL, DEMO_USER_HINT_NONE, '(leer)'),
        ('01_classification.lean.mit-nutzerhinweis.md', 'Klassifikator, mit Hinweis des Nutzers', DEMO_CONTEXT_RICH, DEMO_USER_HINT_SET, 'Dokumentkontext plus Nutzerhinweis'),
    ):
        prev = _set_env(env)
        try:
            text = build_classification_prompt(enriched_context=ctx, width=DEMO_WIDTH, height=DEMO_HEIGHT,
                                               original_alt=DEMO_ORIGINAL_ALT_LEER, user_hint=hint)
        finally:
            _restore_env(prev)
        written.append(_write_snapshot(out_dir, filename=filename, title=title,
                                       builder_ref=_builder_source_link(build_classification_prompt),
                                       mode_info={'V4_PASS_MODE': 'lean'},
                                       demo_values={**common_demo, 'Kontext': note}, prompt_text=text))

    # 2) Inventar-Schritt (nur im Analyse-Modus; im Betrieb steckt er im Combo-Aufruf)
    for bildtyp in ('foto', 'diagramm'):
        prev = _set_env(env)
        try:
            text = build_inventar_prompt(bildtyp=bildtyp, enriched_context=DEMO_KONTEXT_JE_TYP['foto_event' if bildtyp == 'foto' else 'diagramm'],
                                         width=DEMO_WIDTH, height=DEMO_HEIGHT)
        finally:
            _restore_env(prev)
        written.append(_write_snapshot(out_dir, filename=f'02_inventar.{bildtyp}.md', title=f'Inventar-Schritt, Bildtyp {bildtyp}',
                                       builder_ref=_builder_source_link(build_inventar_prompt), mode_info={},
                                       demo_values={**common_demo, 'Bildtyp': bildtyp}, prompt_text=text))

    # 3) Produktionsprompt (Combo) für alle 13 Bildtypen mit Inventar
    top_von = {'foto_event': 'foto', 'foto_personen': 'foto', 'foto_objekte': 'foto', 'foto_essen': 'foto',
               'foto_landschaft': 'foto', 'foto_architektur': 'foto'}
    for i, eff in enumerate(('foto_event', 'foto_personen', 'foto_objekte', 'foto_essen', 'foto_landschaft', 'foto_architektur',
                             'illustration', 'diagramm', 'tabelle', 'karte', 'infografik', 'screenshot', 'strukturformel'), start=1):
        prev = _set_env(env)
        try:
            text = build_combined_inventar_beschreibung_prompt(bildtyp_top=top_von.get(eff, eff), bildtyp_effective=eff,
                                                               enriched_context=DEMO_KONTEXT_JE_TYP[eff],
                                                               width=DEMO_WIDTH, height=DEMO_HEIGHT)
        finally:
            _restore_env(prev)
        written.append(_write_snapshot(out_dir, filename=f'03_beschreibung.{i:02d}_{eff}.md', title=f'Beschreibung, Bildtyp {eff}',
                                       builder_ref=_builder_source_link(build_combined_inventar_beschreibung_prompt),
                                       mode_info={'V4_PASS_MODE': 'lean'},
                                       demo_values={**common_demo, 'Kontext': DEMO_KONTEXT_JE_TYP[eff]}, prompt_text=text))

    # 4) Mini-Familie
    for eff, cls, ctx, alt in (('logo', DEMO_CLASSIFICATION_LOGO, 'LINK-ZIEL: https://www.musterwerk.example', DEMO_ORIGINAL_ALT_BRAUCHBAR),
                               ('icon', DEMO_CLASSIFICATION_ICON, '', DEMO_ORIGINAL_ALT_LEER),
                               ('funktional', DEMO_CLASSIFICATION_FUNKTIONAL, 'Seite 3 von 12', DEMO_ORIGINAL_ALT_UNBRAUCHBAR)):
        prev = _set_env(env)
        try:
            text = build_beschreibung_prompt_mini(eff, cls, ctx, 64, 64, original_alt=alt)
        finally:
            _restore_env(prev)
        written.append(_write_snapshot(out_dir, filename=f'04_mini_{eff}.md', title=f'Beschreibung, Bildtyp {eff}',
                                       builder_ref=_builder_source_link(build_beschreibung_prompt_mini), mode_info={},
                                       demo_values={'width × height': '64 × 64', 'Kontext': ctx or '(leer)', 'Original-Alt': alt or '(leer)'},
                                       prompt_text=text))

    # 5) Zusatzschritte: Werte-Ablesung, Aufzählung, Prüfpass
    prev = _set_env(env)
    try:
        w = orch.WerteOutput(titel='Umsatzentwicklung', diagrammtyp='gruppierte Balken', achsen='0 bis 6, Millionen Euro', lesbarkeit='gut',
                             reihen=[orch.WerteReihe(name='Hardware', punkte=[orch.WertePunkt(kategorie='2021', wert='2,5'), orch.WertePunkt(kategorie='2022', wert='4,4'), orch.WertePunkt(kategorie='2023', wert='2,0')]),
                                     orch.WerteReihe(name='Mobile', punkte=[orch.WertePunkt(kategorie='2021', wert='4,5'), orch.WertePunkt(kategorie='2022', wert='2,8'), orch.WertePunkt(kategorie='2023', wert='5,0')])])
        z = orch.ZaehlOutput(personen=[orch.ZaehlPerson(position='links', merkmal='blauer Blazer, Namensschild', sichtbarkeit='ganz'),
                                       orch.ZaehlPerson(position='rechts', merkmal='graues Hemd, Rücken zur Kamera', sichtbarkeit='teilweise verdeckt')],
                             gruppen=[orch.ZaehlGruppe(bezeichnung='Abstimmkarten', anzahl=6, zaehlweise='exakt')], lesbare_texte=['Workshop Inklusion'])
        stuecke = [
            ('05_werte_ablesung.md', 'Werte-Ablesung (Diagramm, eigener Aufruf)', orch._lies_diagramm_werte.__doc__ or '', _werte_prompt_text()),
            ('05_werte_block.md', 'Block ABGELESENE WERTE (wird an den Beschreibungs-Prompt gehängt)', '', orch._werte_block(w).strip()),
            ('05_zaehl_aufruf.md', 'Aufzähl-Schritt (Foto, eigener Aufruf)', '', _zaehl_prompt_text()),
            ('05_zaehl_block.md', 'Block AUFGEZÄHLT (wird an den Beschreibungs-Prompt gehängt)', '', orch._zaehl_block(z).strip()),
            ('05_faktenblatt_aufruf_tabelle.md', 'Faktenblatt Tabelle (eigener Aufruf)', '', orch._FAKTENBLATT_PROMPTS['tabelle']),
            ('05_faktenblatt_aufruf_karte.md', 'Faktenblatt Karte (eigener Aufruf)', '', orch._FAKTENBLATT_PROMPTS['karte']),
            ('05_faktenblatt_aufruf_infografik.md', 'Faktenblatt Infografik (eigener Aufruf)', '', orch._FAKTENBLATT_PROMPTS['infografik']),
            ('05_faktenblatt_block.md', 'Block FAKTENBLATT (wird an den Beschreibungs-Prompt gehängt)', '', orch._faktenblatt_block(
                orch.TabelleFakten(titel='Nährwerte je 100 Gramm', spaltenkoepfe=['Nährstoff', 'Menge'],
                                   zeilen=[orch.TabelleZeile(bezeichnung='Energie', werte=['52 kcal']),
                                           orch.TabelleZeile(bezeichnung='Kohlenhydrate', werte=['12 g']),
                                           orch.TabelleZeile(bezeichnung='Fett', werte=['0 g'])],
                                   fussnoten=['Quelle: Beispiel AG'], lesbarkeit='gut'), 'tabelle').strip()),
            ('06_pruefpass.md', 'Prüfpass', '', orch._build_verify_prompt(
                'Balkendiagramm zur Umsatzentwicklung 2021 bis 2023: Nur Mobile liegt am Ende über dem Ausgangswert und erreicht 5,0.',
                enriched_context=DEMO_KONTEXT_JE_TYP['diagramm'], langbeschreibung='(Langbeschreibung des Erzeugers)',
                bildtyp='diagramm', fakten_block=orch._werte_block(w))),
        ]
    finally:
        _restore_env(prev)
    for filename, title, _doc, text in stuecke:
        written.append(_write_snapshot(out_dir, filename=filename, title=title, builder_ref='pipelines/v4/orchestrator.py',
                                       mode_info={}, demo_values={}, prompt_text=text))

    return written


def _werte_prompt_text() -> str:
    """Der Prompt der Werte-Ablesung, ohne Modellaufruf (gleicher Wortlaut wie in orchestrator._lies_diagramm_werte)."""
    import inspect
    from pipelines.v4 import orchestrator as orch
    return _prompt_literal_aus_quelle(inspect.getsource(orch._lies_diagramm_werte))


def _zaehl_prompt_text() -> str:
    import inspect
    from pipelines.v4 import orchestrator as orch
    return _prompt_literal_aus_quelle(inspect.getsource(orch._zaehle_bild))


def _prompt_literal_aus_quelle(quelle: str) -> str:
    """Liest das String-Literal `prompt = (...)` aus dem Funktionsquelltext (die Aufrufe
    selbst brauchen ein Bild und ein Modell)."""
    import ast
    baum = ast.parse(quelle.lstrip() if not quelle.startswith('def') else quelle)
    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.Assign) and any(getattr(t, 'id', '') == 'prompt' for t in knoten.targets):
            try:
                return ast.literal_eval(knoten.value)
            except Exception:
                pass
    return '(Prompt nicht als Literal auffindbar)'


def write_index(out_dir: Path, files: list[Path]) -> Path:
    """Schreibt README.md im snapshots-Ordner mit Liste aller Dateien."""
    lines = [
        '# Prompt-Snapshots — InkluDocs v4',
        '',
        f'Automatisch generiert am {DEMO_DATE} via `python3 -m scripts.render_prompts`.',
        '',
        'Jede Datei zeigt einen Builder mit Demo-Werten — der **gerenderte Prompt-Text**, ',
        'den das Modell tatsaechlich bekommt. Aenderungen an den Builder-Python-Dateien ',
        'sollten hier per `git diff` sichtbar werden.',
        '',
        '**Source of Truth bleibt der Python-Code.** Diese Snapshots sind Lese-Artefakte, ',
        'keine Edit-Quelle.',
        '',
        '## Builder',
        '',
    ]
    for path in sorted(files):
        rel = path.name
        lines.append(f'- [`{rel}`](./{rel})')
    lines.append('')

    readme = out_dir / 'README.md'
    readme.write_text('\n'.join(lines), encoding='utf-8')
    return readme


def main() -> int:
    out_dir = _BACKEND_ROOT / 'prompts' / 'snapshots'
    print(f'Render-Ziel: {out_dir}')

    files = render_all(out_dir)
    index = write_index(out_dir, files)

    print(f'{len(files)} Snapshots geschrieben + {index.name}.')
    for f in files:
        print(f'  - {f.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
