"""
Context Engine for InkluDocs Alt-Text Generation.
Version 3.1 – Anti-Halluzination Update (21.03.2026)

Pipeline:
  Stufe 1 (Qwen, lokal): Klassifikation (Bildtyp + dekorativ + konfidenz)
  Stufe 2 (Mistral, API): Alt-Text-Generierung mit Insight-First-Prompt
  Fallback: Qwen generiert wenn Mistral fehlschlaegt

Modi (PIPELINE_MODE Umgebungsvariable):
  mistral_primary: Qwen klassifiziert, Mistral generiert (Default, beste Qualitaet)
  hybrid: Qwen macht einfache Bilder, Mistral die komplexen (spart Kosten)
  qwen_only: Alles ueber Qwen (Fallback, niedrigste Kosten)
"""

import os
import re

PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "mistral_primary")

# ─── Klassifikations-Prompt (Stufe 1, Qwen) ──────────────────
CLASSIFICATION_PROMPT = """/no_think
Klassifiziere dieses Bild. Antworte NUR mit diesem JSON:
{{"bildtyp": "foto|diagramm|tabelle|strukturformel|logo|icon|karte|screenshot|infografik|dekorativ", "ist_dekorativ": true|false, "konfidenz": "hoch|mittel|niedrig"}}

Regeln:
- dekorativ = rein abstrakte Formen, Farbverlaeufe, Trennlinien, winzige Icons
- Wenn Text, Personen, Daten oder Inhalte sichtbar sind: NICHT dekorativ
- konfidenz: Wie sicher bist du bei der Klassifikation?

Kontext: {context}"""

# ─── Generierungs-Prompt (Stufe 2, Mistral oder Qwen-Fallback) ─
GENERATION_PROMPT = """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Das Bild wurde als Typ "{bildtyp}" klassifiziert.

INSIGHT-FIRST METHODE:
- Liefere sofort die ERKENNTNIS: Was vermittelt dieses Bild? Was ist die Kernaussage?
- Nenne ALLE lesbaren Zahlen, Texte, Beschriftungen
- Bei Diagrammen: Trend und Muster zuerst, dann Details
- Bei Tabellen: Alle Spalten, Zeilen und Werte strukturiert auflisten
- Bei Formeln: Vollstaendige Notation und Name der Verbindung
- Bei Fotos: Kontext und Bedeutung beschreiben
- Bei Logos: NUR den Namen, KEINE visuelle Beschreibung
- Antwort MUSS auf Deutsch sein

ANTI-HALLUZINATION (WICHTIG – bei Verstoessen ist der Alt-Text FALSCH):
- Marken, Hersteller, Modellnamen NUR nennen wenn Logo oder Schriftzug im Bild KLAR LESBAR ist
- Kennzeichen, Nummernschilder NUR transkribieren wenn Text VOLLSTAENDIG UND EINDEUTIG lesbar ist
- Personen NUR namentlich benennen wenn der Kontext den Namen enthaelt oder ein Namensschild lesbar ist
- Gebaeude NUR benennen wenn ein Schild lesbar ist oder der Kontext es eindeutig sagt
- Bei JEDER Unsicherheit: allgemein beschreiben statt raten ("ein Auto" statt "ein VW Golf")
- Erfinde NICHTS was du nicht klar erkennen oder aus dem Kontext belegen kannst
- Lieber zu wenig Details als FALSCHE Details

ZUSATZREGELN:
- KEIN ALTER: Nenne Name und Funktion, aber KEINE biografischen Daten wie Alter
- VERLINKTE BILDER: Wenn der Kontext ein [Link-Ziel] enthaelt, beschreibe die FUNKTION des Links
- ANTI-REDUNDANZ: Wiederhole NICHTS was im Kontext steht
- FARBEN: Nur informationstragende Farben (Warnschilder, Diagramme). Keine optischen Farben
- WISSENSTRANSFER: Nutze den Kontext um das Gezeigte zu IDENTIFIZIEREN – aber halluziniere KEINE Fakten

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

alt_text: Max 250 Zeichen, Insight-First. Bei Logos max 80 Zeichen.
langbeschreibung: Alle Details, max 1000 Zeichen. Leer wenn nicht noetig.

Kontext (Umgebender Text und Seiteninfo): {context}"""

# ─── Legacy General-Prompt (fuer qwen_only Modus und Fallback) ─
GENERAL_PROMPT = """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
ZIEL: Blinde Nutzer erhalten die GLEICHE INFORMATION wie Sehende.

Antworte NUR mit diesem exakten JSON-Format:
{{"bildtyp": "foto|diagramm|tabelle|screenshot|icon|logo|karte|dekorativ|strukturformel", "alt_text": "...", "ist_dekorativ": true/false, "konfidenz": "hoch|mittel|niedrig"}}

FORMAT-REGELN:
- Bei Diagrammen, Karten, Tabellen, Infografiken und Strukturformeln: Beginne mit dem Typ als Praefix.
- Bei Fotos, Screenshots und Logos: KEIN Praefix, starte direkt mit der Beschreibung.
- LAENGE: 1-3 Saetze (100-250 Zeichen). Nur die Kernaussage.

HARTE REGELN:
1. Antwort MUSS auf Deutsch sein.
2. Beschreibe was im Bild ist, nutze den Kontext um es zu IDENTIFIZIEREN. Erfinde NICHTS.
3. ANTI-HALLUZINATION: Marken/Modelle NUR nennen wenn Logo/Schriftzug KLAR LESBAR. Kennzeichen NUR wenn VOLLSTAENDIG lesbar. Bei Unsicherheit allgemein beschreiben.
4. Wenn Text im Bild LESBAR ist, ist es NIEMALS dekorativ.
5. Nur informationstragende Farben. Keine optischen Farben.
6. Wiederhole NICHTS was im Kontext steht.
7. Identitaet aus Kontext nennen. Kein Alter nennen.
8. Bei [Link-Ziel] die Link-Funktion beschreiben.

Kontext: {context}"""

# ─── Types that Mistral should always handle in hybrid mode ───
MISTRAL_TYPES = {"diagramm", "tabelle", "karte", "strukturformel", "infografik"}

# Complex types that should include langbeschreibung
COMPLEX_TYPES = {"diagramm", "karte", "tabelle", "infografik", "strukturformel"}

# Patterns for detecting image types from surrounding text
_TYPE_PATTERNS = {
    "diagramm": [
        r"(?i)\b(?:Diagramm|Balkendiagramm|Kreisdiagramm|Liniendiagramm|Saeulend|Tortendiagramm|Chart|Graph)\b",
        r"(?i)\b(?:Abb(?:ildung)?\.?\s*\d+\s*:\s*.*(?:Diagramm|Verteilung|Entwicklung|Verlauf|Vergleich))",
    ],
    "karte": [
        r"(?i)\b(?:Karte|Landkarte|Stadtplan|Lageplan|Map|Uebersichtskarte|Standortkarte)\b",
    ],
    "tabelle": [
        r"(?i)\b(?:Tabelle|Tab\.?\s*\d|Uebersicht\s*\d)\b",
    ],
    "infografik": [
        r"(?i)\b(?:Infografik|Schaubild|Uebersichtsgrafik)\b",
    ],
    "foto": [
        r"(?i)\b(?:Foto|Bild|Aufnahme|Portrait|Gruppenbild)\b",
    ],
    "logo": [
        r"(?i)\b(?:Logo|Wortmarke|Bildmarke|Firmenzeichen|Markenzeichen)\b",
    ],
    "screenshot": [
        r"(?i)\b(?:Screenshot|Bildschirmfoto|Bildschirmaufnahme|Screencapture)\b",
    ],
    "dekorativ": [
        r"(?i)\b(?:Schmuckbild|Dekoration|Zierbild|Hintergrundbild|Titelbild)\b",
    ],
    "strukturformel": [
        r"(?i)\b(?:Strukturformel|Molekuel|Summenformel|Reaktionsgleichung|Bindung|Atom|IUPAC)\b",
        r"(?i)\b(?:Methanol|Ethanol|Iodmethan|Kohlenwasserstoff|Wasserstoff|Sauerstoff|Stickstoff)\b",
    ],
}


def get_classification_prompt(context_text: str = "") -> str:
    """Return the classification prompt for Qwen (Stufe 1)."""
    context = context_text[:400] if context_text else "Kein Kontext."
    return CLASSIFICATION_PROMPT.format(context=context)


def get_generation_prompt(bildtyp: str, context_text: str = "") -> str:
    """Return the Insight-First generation prompt for Mistral (Stufe 2)."""
    context = context_text[:1000] if context_text else "Kein Kontext."
    return GENERATION_PROMPT.format(bildtyp=bildtyp, context=context)


def get_prompt(image_type: str = None, context_text: str = "") -> str:
    """Legacy: Return prompt for Qwen-only/fallback mode."""
    context = context_text[:800] if context_text else "Kein Kontext."
    return GENERAL_PROMPT.format(context=context)


def should_use_mistral(bildtyp: str, konfidenz: str) -> bool:
    """Decide if Mistral should generate the alt-text based on pipeline mode."""
    if PIPELINE_MODE == "qwen_only":
        return False
    if PIPELINE_MODE == "mistral_primary":
        return True
    # hybrid mode
    if bildtyp in MISTRAL_TYPES:
        return True
    if konfidenz in ("niedrig", "mittel"):
        return True
    return False


def detect_type_from_context(context_text: str) -> str | None:
    """Analyze surrounding text for clues about the image type."""
    if not context_text:
        return None
    scores = {}
    for img_type, patterns in _TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, context_text)
            score += len(matches)
        if score > 0:
            scores[img_type] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def is_complex_type(image_type: str) -> bool:
    """Check if an image type should include a langbeschreibung field."""
    return image_type in COMPLEX_TYPES


def extract_page_profile(soup) -> str:
    """Extract a page profile from BeautifulSoup HTML for better context."""
    parts = []
    if soup.title and soup.title.string:
        parts.append(f"[Seitentitel] {soup.title.string.strip()}")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        parts.append(f"[Meta-Beschreibung] {meta_desc['content'][:200]}")
    headings = []
    for tag in ["h1", "h2", "h3"]:
        for h in soup.find_all(tag):
            text = h.get_text(strip=True)
            if text:
                headings.append(f"{tag.upper()}: {text}")
    if headings:
        parts.append(f"[Ueberschriften] {' | '.join(headings[:10])}")
    return "\n".join(parts) if parts else ""
