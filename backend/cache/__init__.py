"""Persistent content-hash Cache fuer Alt-Text-Pipeline.

Cache-Schicht VOR der Pipeline-Auswahl in pdf_processor.generate_alt_text.
Greift cross-projekt: dasselbe Bild in verschiedenen Projekten teilt EINEN Cache-Eintrag.
"""
from .content_cache import (
    build_cache_key,
    get_cached,
    set_cached,
    evict_by_content_hash,
    init_schema,
)

__all__ = [
    "build_cache_key",
    "get_cached",
    "set_cached",
    "evict_by_content_hash",
    "init_schema",
]
