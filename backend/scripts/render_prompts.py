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
# DEMO-WERTE — bewusst plausibel + bildtyp-agnostisch
# ============================================================================

DEMO_DATE = datetime.date.today().isoformat()

DEMO_WIDTH = 1280
DEMO_HEIGHT = 720

# Zwei Kontext-Varianten — minimal (Frontend-Upload ohne PDF) und rich (PDF-Seite)
DEMO_CONTEXT_MINIMAL = ''
DEMO_CONTEXT_RICH = (
    'Workshop-Bericht: Inklusion in der digitalen Arbeitswelt. '
    'Am 5. Mai 2026 fand bei INKLUTEC ein eintaegiger Workshop zur '
    'barrierefreien Software-Entwicklung statt. Teilnehmende waren '
    'Entwickler:innen aus drei Partnerunternehmen.'
)

DEMO_ORIGINAL_ALT_LEER = ''
DEMO_ORIGINAL_ALT_BRAUCHBAR = 'Workshop-Foto Inklusion 2026'
DEMO_ORIGINAL_ALT_UNBRAUCHBAR = 'IMG_2345.jpg'

DEMO_USER_HINT_NONE = None
DEMO_USER_HINT_SET = (
    'Das ist unser Workshop am 5. Mai 2026 mit der Firma Acer Deutschland '
    'als Kooperationspartner. Bitte Workshop-Charakter betonen.'
)

# Demo-Inventar fuer Foto-Builder (Workshop-Setting, 4 Personen, Beamer, Catering)
DEMO_INVENTAR_FOTO_EVENT = InventarOutput(
    foto_subtyp='foto_event',
    personen=[
        PersonInBild(
            position='vorn links',
            haltung='stehend',
            blickrichtung='zur Praesentation',
            kleidungs_charakter='Business-casual',
        ),
        PersonInBild(
            position='Mitte',
            haltung='stehend',
            blickrichtung='zur Kamera',
            kleidungs_charakter='Business-casual',
        ),
        PersonInBild(
            position='hinten rechts',
            haltung='sitzend',
            blickrichtung='zur Praesentation',
            kleidungs_charakter='legere Kleidung',
        ),
        PersonInBild(
            position='Mitte rechts',
            haltung='stehend',
            kleidungs_charakter='Business-casual',
        ),
    ],
    objekte=[
        ObjektInBild(
            beschreibung='Projektionsflaeche mit hellem Lichtkegel',
            position='Hintergrund Mitte',
            sicherheit='hoch',
            moegliche_identifikationen=['Beamer-Projektion'],
        ),
        ObjektInBild(
            beschreibung='rechteckige weisse Karten an Personen befestigt',
            position='auf Brusthoehe der Personen',
            sicherheit='hoch',
            moegliche_identifikationen=['Namensschilder'],
        ),
        ObjektInBild(
            beschreibung='Tisch mit Getraenkeflaschen und Glaesern',
            position='rechter Bildrand',
            sicherheit='hoch',
            moegliche_identifikationen=['Catering-Tisch'],
        ),
    ],
    lesbare_texte=[
        TextInBild(
            inhalt='acer',
            typ='logo',
            vollstaendigkeit='vollständig',
        ),
        TextInBild(
            inhalt='Workshop Inklusion 2026',
            typ='überschrift',
            vollstaendigkeit='vollständig',
        ),
    ],
    setting={
        'raum_charakter': 'Seminarraum',
        'beleuchtung': 'gedaempft, Projektionslicht',
        'dominante_farben': 'blau, weiss, grau',
        'ungefaehre_szene': 'Vortragssituation mit Publikum',
    },
    handlung='Praesentation vor stehendem und sitzendem Publikum',
    halluzinations_warnung=[
        'Namensschilder nicht lesbar — keine Identifikationen ableiten.',
        'Karten an Personen nicht als Stimmkarten/Flyer interpretieren.',
    ],
    inventar_konfidenz_gesamt='hoch',
)

# Generisches Inventar fuer Nicht-Foto-Builder (Diagramm-/Daten-Bildtyp)
DEMO_INVENTAR_GENERISCH = InventarOutput(
    foto_subtyp=None,
    personen=[],
    objekte=[
        ObjektInBild(
            beschreibung='blauer Balken mit Beschriftung 2024',
            position='links',
            sicherheit='hoch',
        ),
        ObjektInBild(
            beschreibung='oranger Balken mit Beschriftung 2025',
            position='Mitte',
            sicherheit='hoch',
        ),
        ObjektInBild(
            beschreibung='gruener Balken mit Beschriftung 2026',
            position='rechts',
            sicherheit='hoch',
        ),
    ],
    lesbare_texte=[
        TextInBild(inhalt='Umsatzentwicklung 2024-2026', typ='überschrift', vollstaendigkeit='vollständig'),
        TextInBild(inhalt='Mio. EUR', typ='beschriftung', vollstaendigkeit='vollständig'),
        TextInBild(inhalt='12.4', typ='zahl', vollstaendigkeit='vollständig'),
        TextInBild(inhalt='15.7', typ='zahl', vollstaendigkeit='vollständig'),
        TextInBild(inhalt='18.2', typ='zahl', vollstaendigkeit='vollständig'),
    ],
    setting={'raum_charakter': 'kein Raum (Diagramm)'},
    handlung=None,
    halluzinations_warnung=[],
    inventar_konfidenz_gesamt='hoch',
)

# Demo-Classification fuer Mini-Builder (logo/icon/funktional)
DEMO_CLASSIFICATION_LOGO = ClassificationOutput(
    bildtyp='logo',
    konfidenz='hoch',
    ist_dekorativ=False,
    original_alt_brauchbar=True,
    klassifikations_begruendung='Erkennbares Marken-Logo der Firma Acer ohne weiteren Bildinhalt.',
)

DEMO_CLASSIFICATION_ICON = ClassificationOutput(
    bildtyp='icon',
    konfidenz='hoch',
    ist_dekorativ=False,
    original_alt_brauchbar=False,
    klassifikations_begruendung='Kleines funktionales Symbol (Lupe) ohne Beschriftung im Bild.',
)

DEMO_CLASSIFICATION_FUNKTIONAL = ClassificationOutput(
    bildtyp='funktional',
    konfidenz='hoch',
    ist_dekorativ=False,
    original_alt_brauchbar=False,
    klassifikations_begruendung='Pfeil-Element mit Zustand "naechste Seite" — Navigation, kein reines Icon.',
)

# Demo-Beschreibung fuer Validierungs-Builder
DEMO_BESCHREIBUNG = BeschreibungOutput(
    alt_text='Vier Workshop-Teilnehmende in einem Seminarraum vor einer Projektionsflaeche, im Vordergrund das Acer-Logo.',
    langbeschreibung=(
        'Das Bild zeigt eine Workshop-Situation in einem gedaempft beleuchteten Seminarraum. '
        'Vier Personen in Business-casual und legerer Kleidung stehen und sitzen vor einer hellen '
        'Projektionsflaeche im Hintergrund. An ihren Bruesten haengen weisse Namensschilder. '
        'Am rechten Bildrand befindet sich ein Catering-Tisch mit Getraenkeflaschen und Glaesern. '
        'Ueber dem Bild der Schriftzug "Workshop Inklusion 2026" und das Acer-Logo.'
    ),
    verwendete_inventar_items=['personen', 'objekte', 'lesbare_texte', 'setting'],
    nicht_verwendete_inventar_items=[],
    nicht_im_inventar=[],
    atmosphaere_belege=[],
)


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
    """Rendert alle Builder. Liefert Liste geschriebener Pfade."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-Import, damit ENV-Patches beim Module-Load nicht greifen.
    from prompts.builders import (
        build_beschreibung_prompt_mini,
        build_beschreibung_prompt_with_inventar,
        build_classification_prompt,
        build_combined_inventar_beschreibung_prompt,
        build_inventar_prompt,
        build_validierung_prompt,
    )
    from prompts.builders.beschreibung_daten import (
        build_beschreibung_prompt_diagramm,
        build_beschreibung_prompt_illustration,
        build_beschreibung_prompt_infografik,
        build_beschreibung_prompt_karte,
        build_beschreibung_prompt_screenshot,
        build_beschreibung_prompt_strukturformel,
        build_beschreibung_prompt_tabelle,
    )
    from prompts.builders.beschreibung_foto import (
        build_beschreibung_prompt_foto_architektur,
        build_beschreibung_prompt_foto_essen,
        build_beschreibung_prompt_foto_event,
        build_beschreibung_prompt_foto_landschaft,
        build_beschreibung_prompt_foto_objekte,
        build_beschreibung_prompt_foto_personen,
    )
    from prompts.builders.beschreibung_mini import (
        build_beschreibung_prompt_funktional,
        build_beschreibung_prompt_icon,
        build_beschreibung_prompt_logo,
    )

    written: list[Path] = []

    common_demo = {
        'width × height': f'{DEMO_WIDTH} × {DEMO_HEIGHT}',
    }

    # ------------------------------------------------------------------
    # 1) Klassifikator — lean + full
    # ------------------------------------------------------------------
    for mode in ('lean', 'full'):
        prev = _set_env({'V4_PASS_MODE': mode, 'V4_PROMPT_MODE': '', 'LLM_PROVIDER': 'bedrock'})
        try:
            text = build_classification_prompt(
                enriched_context=DEMO_CONTEXT_RICH,
                width=DEMO_WIDTH,
                height=DEMO_HEIGHT,
                original_alt=DEMO_ORIGINAL_ALT_LEER,
                user_hint=DEMO_USER_HINT_NONE,
            )
        finally:
            _restore_env(prev)

        path = _write_snapshot(
            out_dir,
            filename=f'01_classification.{mode}.md',
            title=f'Klassifikator — Modus: {mode}',
            builder_ref=_builder_source_link(build_classification_prompt),
            mode_info={'V4_PASS_MODE': mode},
            demo_values={
                **common_demo,
                'enriched_context': 'rich (Workshop-PDF-Auszug)',
                'original_alt': '(leer)',
                'user_hint': '(keiner)',
            },
            prompt_text=text,
        )
        written.append(path)

    # Klassifikator — Frontend-Upload-Variante (leerer Kontext, lean)
    prev = _set_env({'V4_PASS_MODE': 'lean', 'V4_PROMPT_MODE': '', 'LLM_PROVIDER': 'bedrock'})
    try:
        text = build_classification_prompt(
            enriched_context=DEMO_CONTEXT_MINIMAL,
            width=DEMO_WIDTH,
            height=DEMO_HEIGHT,
            original_alt=DEMO_ORIGINAL_ALT_LEER,
        )
    finally:
        _restore_env(prev)

    written.append(_write_snapshot(
        out_dir,
        filename='01_classification.lean.frontend-upload.md',
        title='Klassifikator — Modus: lean, Frontend-Einzelbild-Upload (leerer Kontext)',
        builder_ref=_builder_source_link(build_classification_prompt),
        mode_info={'V4_PASS_MODE': 'lean'},
        demo_values={
            **common_demo,
            'enriched_context': '(leer — Frontend-Einzelbild-Upload ohne PDF)',
            'original_alt': '(leer)',
        },
        prompt_text=text,
    ))

    # Klassifikator — Variante MIT Nutzer-Hinweis (zeigt die Vorrang-Ordnung
    # des user_hint_block aus helpers.py im gerenderten Prompt)
    prev = _set_env({'V4_PASS_MODE': 'lean', 'V4_PROMPT_MODE': '', 'LLM_PROVIDER': 'bedrock'})
    try:
        text = build_classification_prompt(
            enriched_context=DEMO_CONTEXT_RICH,
            width=DEMO_WIDTH,
            height=DEMO_HEIGHT,
            original_alt=DEMO_ORIGINAL_ALT_LEER,
            user_hint=DEMO_USER_HINT_SET,
        )
    finally:
        _restore_env(prev)

    written.append(_write_snapshot(
        out_dir,
        filename='01_classification.lean.mit-nutzerhinweis.md',
        title='Klassifikator — Modus: lean, MIT Nutzer-Hinweis (Vorrang-Ordnung)',
        builder_ref=_builder_source_link(build_classification_prompt),
        mode_info={'V4_PASS_MODE': 'lean'},
        demo_values={
            **common_demo,
            'enriched_context': 'rich (Workshop-PDF-Auszug)',
            'original_alt': '(leer)',
            'user_hint': 'gesetzt (Workshop Acer Deutschland)',
        },
        prompt_text=text,
    ))

    # ------------------------------------------------------------------
    # 2) Inventar — fuer foto + diagramm (kein Lean/Full-Schalter im Builder)
    # ------------------------------------------------------------------
    for bildtyp in ('foto', 'diagramm'):
        prev = _set_env({'V4_PASS_MODE': 'full', 'V4_PROMPT_MODE': '', 'LLM_PROVIDER': 'bedrock'})
        try:
            text = build_inventar_prompt(
                bildtyp=bildtyp,
                enriched_context=DEMO_CONTEXT_RICH,
                width=DEMO_WIDTH,
                height=DEMO_HEIGHT,
            )
        finally:
            _restore_env(prev)

        written.append(_write_snapshot(
            out_dir,
            filename=f'02_inventar.{bildtyp}.md',
            title=f'Inventar (Pass 2) — Bildtyp: {bildtyp}',
            builder_ref=_builder_source_link(build_inventar_prompt),
            mode_info={'V4_PASS_MODE': 'full'},
            demo_values={**common_demo, 'bildtyp': bildtyp, 'enriched_context': 'rich'},
            prompt_text=text,
        ))

    # ------------------------------------------------------------------
    # 3) Combo (Lean-Mode) — fuer foto_event + diagramm
    # ------------------------------------------------------------------
    combo_cases = [
        ('foto', 'foto_event'),
        ('foto', 'foto_objekte'),
        ('diagramm', 'diagramm'),
    ]
    for bildtyp_top, bildtyp_eff in combo_cases:
        prev = _set_env({'V4_PASS_MODE': 'lean', 'V4_PROMPT_MODE': 'lean', 'LLM_PROVIDER': 'bedrock'})
        try:
            text = build_combined_inventar_beschreibung_prompt(
                bildtyp_top=bildtyp_top,
                bildtyp_effective=bildtyp_eff,
                enriched_context=DEMO_CONTEXT_RICH,
                width=DEMO_WIDTH,
                height=DEMO_HEIGHT,
                original_alt=DEMO_ORIGINAL_ALT_LEER,
            )
        finally:
            _restore_env(prev)

        written.append(_write_snapshot(
            out_dir,
            filename=f'03_combo.{bildtyp_eff}.md',
            title=f'Combo (Lean-Mode Pass 2+3) — Bildtyp: {bildtyp_eff}',
            builder_ref=_builder_source_link(build_combined_inventar_beschreibung_prompt),
            mode_info={'V4_PASS_MODE': 'lean', 'V4_PROMPT_MODE': 'lean'},
            demo_values={**common_demo, 'bildtyp_top': bildtyp_top, 'bildtyp_effective': bildtyp_eff},
            prompt_text=text,
        ))

    # ------------------------------------------------------------------
    # 4) Premium-Foto-Builder (event/personen/objekte) — lean + full Modi
    # ------------------------------------------------------------------
    premium_foto_builders = [
        ('foto_event', build_beschreibung_prompt_foto_event, DEMO_INVENTAR_FOTO_EVENT),
        ('foto_personen', build_beschreibung_prompt_foto_personen, DEMO_INVENTAR_FOTO_EVENT),
        ('foto_objekte', build_beschreibung_prompt_foto_objekte, DEMO_INVENTAR_FOTO_EVENT),
    ]
    for subtyp, builder, inventar in premium_foto_builders:
        for prompt_mode in ('lean', 'full'):
            prev = _set_env({
                'V4_PASS_MODE': 'full',
                'V4_PROMPT_MODE': prompt_mode,
                'LLM_PROVIDER': 'bedrock' if prompt_mode == 'lean' else 'mistral',
            })
            try:
                text = builder(
                    inventar=inventar,
                    enriched_context=DEMO_CONTEXT_RICH,
                    width=DEMO_WIDTH,
                    height=DEMO_HEIGHT,
                )
            finally:
                _restore_env(prev)

            written.append(_write_snapshot(
                out_dir,
                filename=f'04_premium_{subtyp}.{prompt_mode}.md',
                title=f'Premium-Builder {subtyp} — Prompt-Modus: {prompt_mode}',
                builder_ref=_builder_source_link(builder),
                mode_info={'V4_PROMPT_MODE': prompt_mode, 'LLM_PROVIDER': 'bedrock' if prompt_mode == 'lean' else 'mistral'},
                demo_values={**common_demo, 'inventar': 'Workshop-Setting (4 Personen, Beamer, Catering)'},
                prompt_text=text,
            ))

    # ------------------------------------------------------------------
    # 5) Standard-Foto-Builder (essen/landschaft/architektur) — nur Default-Modus
    # ------------------------------------------------------------------
    standard_foto_builders = [
        ('foto_essen', build_beschreibung_prompt_foto_essen),
        ('foto_landschaft', build_beschreibung_prompt_foto_landschaft),
        ('foto_architektur', build_beschreibung_prompt_foto_architektur),
    ]
    for subtyp, builder in standard_foto_builders:
        prev = _set_env({'V4_PROMPT_MODE': 'lean', 'LLM_PROVIDER': 'bedrock'})
        try:
            text = builder(
                inventar=DEMO_INVENTAR_FOTO_EVENT,
                enriched_context=DEMO_CONTEXT_RICH,
                width=DEMO_WIDTH,
                height=DEMO_HEIGHT,
            )
        finally:
            _restore_env(prev)

        written.append(_write_snapshot(
            out_dir,
            filename=f'05_standard_{subtyp}.md',
            title=f'Standard-Builder {subtyp}',
            builder_ref=_builder_source_link(builder),
            mode_info={'V4_PROMPT_MODE': 'lean'},
            demo_values={**common_demo, 'inventar': 'Workshop-Setting (generisch)'},
            prompt_text=text,
        ))

    # ------------------------------------------------------------------
    # 6) Daten-Builder (diagramm/tabelle/karte/infografik/screenshot/strukturformel/illustration)
    # ------------------------------------------------------------------
    daten_builders = [
        ('illustration', build_beschreibung_prompt_illustration),
        ('diagramm', build_beschreibung_prompt_diagramm),
        ('tabelle', build_beschreibung_prompt_tabelle),
        ('karte', build_beschreibung_prompt_karte),
        ('infografik', build_beschreibung_prompt_infografik),
        ('screenshot', build_beschreibung_prompt_screenshot),
        ('strukturformel', build_beschreibung_prompt_strukturformel),
    ]
    for bildtyp, builder in daten_builders:
        prev = _set_env({'V4_PROMPT_MODE': 'lean', 'LLM_PROVIDER': 'bedrock'})
        try:
            text = builder(
                inventar=DEMO_INVENTAR_GENERISCH,
                enriched_context=DEMO_CONTEXT_RICH,
                width=DEMO_WIDTH,
                height=DEMO_HEIGHT,
            )
        finally:
            _restore_env(prev)

        written.append(_write_snapshot(
            out_dir,
            filename=f'06_daten_{bildtyp}.md',
            title=f'Daten-Builder {bildtyp}',
            builder_ref=_builder_source_link(builder),
            mode_info={'V4_PROMPT_MODE': 'lean'},
            demo_values={**common_demo, 'inventar': 'Diagramm-Setting (3 Balken)'},
            prompt_text=text,
        ))

    # ------------------------------------------------------------------
    # 7) Mini-Builder (logo/icon/funktional) — brauchen ClassificationOutput
    # ------------------------------------------------------------------
    mini_builders = [
        ('logo', build_beschreibung_prompt_logo, DEMO_CLASSIFICATION_LOGO, DEMO_ORIGINAL_ALT_BRAUCHBAR),
        ('icon', build_beschreibung_prompt_icon, DEMO_CLASSIFICATION_ICON, DEMO_ORIGINAL_ALT_LEER),
        ('funktional', build_beschreibung_prompt_funktional, DEMO_CLASSIFICATION_FUNKTIONAL, DEMO_ORIGINAL_ALT_LEER),
    ]
    for bildtyp, builder, classification, orig_alt in mini_builders:
        prev = _set_env({'V4_PROMPT_MODE': 'lean', 'LLM_PROVIDER': 'bedrock'})
        try:
            text = builder(
                classification=classification,
                enriched_context=DEMO_CONTEXT_RICH,
                width=DEMO_WIDTH,
                height=DEMO_HEIGHT,
                original_alt=orig_alt,
            )
        finally:
            _restore_env(prev)

        written.append(_write_snapshot(
            out_dir,
            filename=f'07_mini_{bildtyp}.md',
            title=f'Mini-Builder {bildtyp}',
            builder_ref=_builder_source_link(builder),
            mode_info={'V4_PROMPT_MODE': 'lean'},
            demo_values={**common_demo, 'classification.bildtyp': classification.bildtyp, 'original_alt': orig_alt or '(leer)'},
            prompt_text=text,
        ))

    # ------------------------------------------------------------------
    # 8) Validierung — fuer foto_event + diagramm
    # ------------------------------------------------------------------
    validierung_cases = [
        ('foto_event', DEMO_INVENTAR_FOTO_EVENT),
        ('diagramm', DEMO_INVENTAR_GENERISCH),
    ]
    for bildtyp, inventar in validierung_cases:
        prev = _set_env({'V4_PROMPT_MODE': 'lean', 'LLM_PROVIDER': 'bedrock'})
        try:
            text = build_validierung_prompt(
                bildtyp=bildtyp,
                inventar=inventar,
                beschreibung=DEMO_BESCHREIBUNG,
                enriched_context=DEMO_CONTEXT_RICH,
            )
        finally:
            _restore_env(prev)

        written.append(_write_snapshot(
            out_dir,
            filename=f'08_validierung.{bildtyp}.md',
            title=f'Validierung (Pass 4) — Bildtyp: {bildtyp}',
            builder_ref=_builder_source_link(build_validierung_prompt),
            mode_info={'V4_PROMPT_MODE': 'lean'},
            demo_values={**common_demo, 'bildtyp': bildtyp},
            prompt_text=text,
        ))

    return written


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
