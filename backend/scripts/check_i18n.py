#!/usr/bin/env python3
"""Bau-Zeit-Vollstaendigkeits-Check fuer i18n.

Extrahiert alle _('...')-Strings aus den Jinja2-Templates und alle
t('...')-Strings aus dem JavaScript (Templates + JS-Dateien mit
I18N-Anbindung) und prueft sie gegen alle vier .po-Kataloge.

Bricht mit Exit-Code 1 ab, wenn:
  - ein msgid in einem Katalog fehlt (-> stiller Deutsch-Fallback), oder
  - eine Uebersetzung in en/fr/es leer ist (de darf leer sein, msgid IST Deutsch).

Aufruf ohne Argumente; Pfade werden relativ zum Skript aufgeloest und
funktionieren sowohl im Repo (backend/scripts/) als auch im Container (/app).
"""
import re
import sys
from pathlib import Path

from babel.messages.pofile import read_po

LANGS = ["de", "en", "fr", "es", "da"]
LANGS_REQUIRE_TRANSLATION = ["en", "fr", "es", "da"]

HERE = Path(__file__).resolve().parent
# Im Container liegt das Skript unter /app/scripts, im Repo unter backend/scripts.
BACKEND = HERE.parent
TEMPLATES = BACKEND / "templates"
LOCALES = BACKEND / "locales"
# frontend/ liegt im Container unter /app/frontend, im Repo neben backend/.
FRONTEND = BACKEND / "frontend" if (BACKEND / "frontend").is_dir() else BACKEND.parent / "frontend"

# _('...') / _("...") in Templates — mit Support fuer escapte Quotes
RE_GETTEXT = re.compile(r"""\b_\(\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")\s*[,)]""")
# t('...') / t("...") in JS — nur Literal als erstes Argument
RE_T = re.compile(r"""\bt\(\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")\s*[,)]""")


def unescape(s: str) -> str:
    return s.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def extract(pattern, text):
    for m in pattern.finditer(text):
        yield unescape(m.group(1) if m.group(1) is not None else m.group(2))


def collect_sources():
    """Liefert dict msgid -> Liste der Fundstellen."""
    found = {}

    def add(msgid, where):
        if not msgid:
            return
        found.setdefault(msgid, []).append(where)

    for tpl in sorted(TEMPLATES.glob("*.html")):
        if ".bak" in tpl.name:
            continue
        text = tpl.read_text(encoding="utf-8")
        for msgid in extract(RE_GETTEXT, text):
            add(msgid, f"templates/{tpl.name}")
        # Seiten-JS in Templates nutzt denselben t()-Helfer
        for msgid in extract(RE_T, text):
            add(msgid, f"templates/{tpl.name} (JS)")

    for js in sorted(FRONTEND.glob("*.js")):
        if ".bak" in js.name or "archive" in js.name:
            continue
        text = js.read_text(encoding="utf-8")
        # Nur JS-Dateien, die an die Uebersetzungs-Schicht angebunden sind
        if "I18N" not in text:
            continue
        for msgid in extract(RE_T, text):
            add(msgid, f"frontend/{js.name}")

    return found


def main():
    sources = collect_sources()
    if not sources:
        print("i18n-Check: FEHLER — keine Strings gefunden (Pfade falsch?)", file=sys.stderr)
        return 1

    errors = []
    for lang in LANGS:
        po_path = LOCALES / lang / "LC_MESSAGES" / "messages.po"
        if not po_path.is_file():
            errors.append(f"[{lang}] Katalog fehlt: {po_path}")
            continue
        with open(po_path, "rb") as f:
            catalog = read_po(f)
        ids = {m.id: (m.string or "") for m in catalog if m.id}
        for msgid, wo in sorted(sources.items()):
            if msgid not in ids:
                errors.append(f"[{lang}] msgid FEHLT: \"{msgid}\"  (aus {', '.join(sorted(set(wo)))})")
            elif lang in LANGS_REQUIRE_TRANSLATION and not ids[msgid].strip():
                errors.append(f"[{lang}] Uebersetzung LEER: \"{msgid}\"  (aus {', '.join(sorted(set(wo)))})")

    if errors:
        print(f"i18n-Check FEHLGESCHLAGEN — {len(errors)} Problem(e):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        print("Hinweis: msgid muss EXAKT dem Template-/JS-Text entsprechen (echte Umlaute!).", file=sys.stderr)
        return 1

    print(f"i18n-Check OK: {len(sources)} Strings in allen {len(LANGS)} Katalogen vollstaendig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
