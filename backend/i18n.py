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
from pathlib import Path
from typing import Optional, Callable
from babel.support import Translations
from fastapi import Request

BASE_DIR = Path(__file__).parent
LOCALES_DIR = BASE_DIR / "locales"
TEMPLATES_DIR = BASE_DIR / "templates"

SUPPORTED_LANGUAGES = ["de", "en", "fr", "es"]
DEFAULT_LANGUAGE = "de"

# Sprachnamen in der jeweiligen Landessprache (damit User eigene Sprache findet)
LANGUAGE_LABELS = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Français",
    "es": "Español",
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


def template_context(request: Request, lang: str, **extra) -> dict:
    """Standard-Kontext fuer Jinja2-Templates (inkl. gettext und Sprach-Metadaten)."""
    return {
        "request": request,
        "_": get_gettext(lang),
        "current_lang": lang,
        "language_labels": LANGUAGE_LABELS,
        "supported_languages": SUPPORTED_LANGUAGES,
        **extra,
    }


def get_templates():
    """Erzeuge Jinja2Templates-Instanz fuer FastAPI."""
    from fastapi.templating import Jinja2Templates
    return Jinja2Templates(directory=str(TEMPLATES_DIR))
