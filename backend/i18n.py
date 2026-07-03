"""
InkluDocs i18n-Modul

Zentrale Sprach-Erkennung und Babel-basierte Uebersetzungen fuer Jinja2-Templates.

Vier-stufige Sprach-Aufloesung:
  1. User-Praeferenz aus DB (falls eingeloggt und gesetzt)
  2. Session-Cookie 'lang' (aus Footer-Switcher)
  3. HTTP Accept-Language Header vom Browser
  4. Fallback: Deutsch

Verwendung in FastAPI-Routen:
    from i18n import detect_language, template_context, get_templates
    templates = get_templates()

    @app.get("/")
    async def index(request: Request):
        lang = detect_language(request)
        return templates.TemplateResponse(
            "index.html",
            template_context(request, lang, extra_key=value)
        )

In Templates: {{ _('Text') }}
"""
import hashlib
import json
from pathlib import Path
from typing import Optional, Callable
from babel.support import Translations
from fastapi import Request

BASE_DIR = Path(__file__).parent
LOCALES_DIR = BASE_DIR / "locales"
TEMPLATES_DIR = BASE_DIR / "templates"
FRONTEND_DIR = BASE_DIR / "frontend"

# Gemeinsame statische Assets der Template-Seiten. Ihr Inhalts-Hash haengt
# als ?v=... an den CSS/JS-URLs (Cache-Busting): nach einem Deploy mit
# geaenderten Dateien laden Browser garantiert die neue Version statt der
# alten aus dem Cache (sonst z.B. gemischte Sprachen durch alte dashboard.js).
_ASSET_FILES = ("style.css", "dashboard.css", "dashboard.js", "pwtoggle.js")


def _compute_asset_version() -> str:
    """Kurzer, deterministischer Inhalts-Hash der statischen Assets.
    Aendert sich genau dann, wenn sich eine der Dateien aendert."""
    h = hashlib.md5()
    for name in _ASSET_FILES:
        f = FRONTEND_DIR / name
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:8]


ASSET_VERSION = _compute_asset_version()

SUPPORTED_LANGUAGES = ["de", "en", "fr", "es", "da"]
DEFAULT_LANGUAGE = "de"

# Sprachnamen in der jeweiligen Landessprache (damit User eigene Sprache findet)
LANGUAGE_LABELS = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "da": "Dansk",
}

_translations_cache: dict = {}


def _load_translations(lang: str) -> Translations:
    """Lade und cache Babel-Translations fuer eine Sprache."""
    if lang not in _translations_cache:
        _translations_cache[lang] = Translations.load(
            str(LOCALES_DIR), [lang], domain="messages"
        )
    return _translations_cache[lang]


def get_gettext(lang: str) -> Callable[[str], str]:
    """Gibt gettext-Funktion fuer eine Sprache zurueck."""
    return _load_translations(lang).gettext


def parse_accept_language(header: str) -> Optional[str]:
    """Parse HTTP Accept-Language header, gib erste unterstuetzte Sprache zurueck."""
    if not header:
        return None
    for part in header.split(","):
        lang = part.split(";")[0].split("-")[0].strip().lower()
        if lang in SUPPORTED_LANGUAGES:
            return lang
    return None


def detect_language(request: Request, user_language: Optional[str] = None) -> str:
    """
    4-stufiges Fallback fuer Sprachwahl.
    user_language: optional, wird hier noch nicht aus DB geholt (kommt in naechster Session).
    """
    if user_language and user_language in SUPPORTED_LANGUAGES:
        return user_language

    cookie_lang = request.cookies.get("lang")
    if cookie_lang and cookie_lang in SUPPORTED_LANGUAGES:
        return cookie_lang

    header_lang = parse_accept_language(request.headers.get("accept-language", ""))
    if header_lang:
        return header_lang

    return DEFAULT_LANGUAGE


def _catalog_json(lang: str) -> str:
    """Uebersetzungstabelle der Sprache als JSON fuer die Client-Seite
    (window.I18N). Nur echte msgid->Uebersetzung-Paare (Header/Plurale raus).
    '<' wird zu \\u003c kodiert, damit kein </script> das Skript beendet."""
    trans = _load_translations(lang)
    catalog = getattr(trans, "_catalog", {}) or {}
    data = {k: v for k, v in catalog.items() if isinstance(k, str) and k and v}
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


def template_context(request: Request, lang: str, **extra) -> dict:
    """Standard-Kontext fuer Jinja2-Templates (inkl. gettext und Sprach-Metadaten)."""
    return {
        "request": request,
        "_": get_gettext(lang),
        "current_lang": lang,
        "language_labels": LANGUAGE_LABELS,
        "supported_languages": SUPPORTED_LANGUAGES,
        "i18n_json": _catalog_json(lang),
        "asset_version": ASSET_VERSION,
        **extra,
    }


def get_templates():
    """Erzeuge Jinja2Templates-Instanz fuer FastAPI."""
    from fastapi.templating import Jinja2Templates
    return Jinja2Templates(directory=str(TEMPLATES_DIR))
