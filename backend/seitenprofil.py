"""Seitenprofil und komplexe Bildtypen — die zwei Funktionen, die aus dem
frueheren context_engine.py (Legacy-Pipeline v3.7, abgebaut 07.09.2026) im
v4-Weg weiterleben.

- extract_page_profile: kompaktes Seitenprofil (Titel + Meta-Beschreibung)
  fuer den Web-Scan in main.py.
- is_complex_type: Bildtypen, die im Werkzeug eine Langbeschreibung bekommen
  (zweiter Pass in main.py).
"""
from __future__ import annotations

# Complex types that should include langbeschreibung
COMPLEX_TYPES = {"diagramm", "karte", "tabelle", "infografik", "strukturformel", "screenshot"}


def is_complex_type(image_type: str) -> bool:
    """Check if an image type should include a langbeschreibung field."""
    return image_type in COMPLEX_TYPES


def extract_page_profile(soup) -> str:
    """Extract a compact page profile from BeautifulSoup HTML.

    v2.2.3: Only title + meta description, no headings (max 300 chars).
    Headings come via image-specific context (nearest heading to each image).
    This prevents the page profile from consuming the context budget."""
    parts = []
    if soup.title and soup.title.string:
        parts.append(f"[Seitentitel] {soup.title.string.strip()[:100]}")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        parts.append(f"[Meta-Beschreibung] {meta_desc['content'][:150]}")
    profile = "\n".join(parts) if parts else ""
    return profile[:300]
