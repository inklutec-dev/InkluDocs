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
# formular.js seit 09.09.2026 dabei: Michael Karbe sah nach einem Staging-Neubau noch das alte
# Formular-Skript aus dem Browser-Cache, weil sich der Hash ohne diese Datei nicht aenderte.
# 25.09.2026: dokument.js und abschluss.js fehlten — dieselbe Falle wie bei formular.js
# (nach einem Neubau lief im Browser noch das alte Skript); verwaltung.js ist neu.
# 17.10.2026: hochladefeld.js, klang.js und das Skript des InkluAgent fehlten — dieselbe Falle. Das InkluAgent-Skript besteht seit
# heute aus Teilen (inkluagent/skripte.py): alle frontend/inkluagent*.js zaehlen (ein alter Kern mit neuen Teilen waere gemischt).
# tests/test_asset_version.py prueft, dass jede mit ?v= geladene Datei hier mitzaehlt.
_ASSET_FILES = ("style.css", "dashboard.css", "dashboard.js", "pwtoggle.js", "start.css",
                "demo.css", "demo.js", "demo-shell.js", "formular.js", "uebersetzen.js",
                "dokument.js", "abschluss.js", "verwaltung.js", "hochladefeld.js", "klang.js")


def _asset_dateien() -> list:
    """Die Dateien, deren Inhalt die Version bestimmt (feste Liste + alle Teile des InkluAgent)."""
    return [FRONTEND_DIR / name for name in _ASSET_FILES] + sorted(FRONTEND_DIR.glob("inkluagent*.js"))


def _compute_asset_version() -> str:
    """Kurzer, deterministischer Inhalts-Hash der statischen Assets.
    Aendert sich genau dann, wenn sich eine der Dateien aendert."""
    h = hashlib.md5()
    for f in _asset_dateien():
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:8]


ASSET_VERSION = _compute_asset_version()

SUPPORTED_LANGUAGES = ["de", "en", "fr", "es", "da", "sv"]
DEFAULT_LANGUAGE = "de"

# Sprachnamen in der jeweiligen Landessprache (damit User eigene Sprache findet)
LANGUAGE_LABELS = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "da": "Dansk",
    "sv": "Svenska",
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


def _sprache_von_gettext(_) -> str:
    """Oberflaechensprache einer gettext-Funktion aus get_gettext (gebundene Methode der gecachten Translations);
    sonst die Vorgabe (z. B. die Identitaet in Unit-Tests)."""
    obj = getattr(_, "__self__", None)
    if obj is not None:
        for k, v in _translations_cache.items():
            if v is obj:
                return k
    return DEFAULT_LANGUAGE


def sprache_anzeige(code, _=None, ui_lang: Optional[str] = None) -> str:
    """Dokumentsprache als „Name (Kürzel)“ in der Oberflaechensprache, z. B. „Englisch (en-US)“, „anglais (en-US)“
    (Steve 30.09.2026: nicht nur das Kuerzel — bei Word und PDF). Namen aus CLDR (Babel), dieselbe Quelle wie
    Intl.DisplayNames im Browser (app.html spracheAnzeige). Unbekanntes Kuerzel: nur das Kuerzel."""
    code = str(code or "").strip()
    if not code:
        return ""
    lang = ui_lang or (_sprache_von_gettext(_) if _ is not None else DEFAULT_LANGUAGE)
    basis = code.replace("_", "-").split("-")[0].lower()
    try:
        from babel import Locale
        name = Locale.parse(lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE).languages.get(basis)
    except Exception:  # noqa: BLE001
        name = None
    return f"{name} ({code})" if name else code


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


def eingebettet(request: Request) -> bool:
    """Laeuft die Seite im Arbeitsfenster der Agentenansicht (Steve 12./13.10.2026, Schalter AGENT_ARBEITSFENSTER)? Erkannt am
    Kopf „Sec-Fetch-Dest: iframe“, den der Browser bei jeder Navigation IM Rahmen schickt (so bleiben auch Links darin
    eingebettet), oder am Parameter ?eingebettet=1. base_app.html laesst dann Seitenleiste, Fusszeile und den InkluAgent weg.

    Pruefung 13.10.2026 (K7): der Parameter gilt NICHT bei „Sec-Fetch-Dest: document“ — dann laedt das Hauptfenster. Er wandert
    sonst ueber Weiterleitungen mit (Anmeldung mit ?weiter=/express?eingebettet=1), und der Nutzer stand nach der Anmeldung auf
    einer Seite ohne Seitenleiste und Agent. Der Parameter bleibt nur fuer Browser ohne Fetch-Metadaten."""
    try:
        import funktionen
        if not funktionen.an("AGENT_ARBEITSFENSTER"):
            return False
        ziel = request.headers.get("sec-fetch-dest")
        if ziel == "iframe":
            return True
        if ziel == "document":
            return False
        return request.query_params.get("eingebettet") == "1"
    except Exception:  # noqa: BLE001
        return False


def template_context(request: Request, lang: str, **extra) -> dict:
    """Standard-Kontext fuer Jinja2-Templates (inkl. gettext und Sprach-Metadaten)."""
    extra.setdefault("eingebettet", eingebettet(request))
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
