"""InkluDocs-spezifischer Adapter: stellt Bruecke zwischen chat_engine
und InkluDocs-Datenmodell (projects/images, Pipeline) her."""
import re
from typing import Optional

from database import get_db


MAX_IMAGES_PER_REQUEST = 10

# Wie viele Bilder gehen maximal als Uebersicht in den Smalltalk-Kontext?
# Bei mehr Bildern wird abgeschnitten + Hinweis "...und N weitere".
MAX_IMAGES_IN_SMALLTALK_CONTEXT = 30
# Maximale Laenge des Alt-Text-Auszugs pro Bild im Kontext (Tokens sparen).
MAX_ALT_LEN_IN_CONTEXT = 250


def build_project_summary(project: dict) -> str:
    """Kompakte Bilder-Uebersicht als Kontext fuer den Smalltalk-Bot.

    Damit kann er Fragen wie 'was steht bei Bild 3' oder 'welche Bilder
    haben noch keinen Alt-Text' direkt beantworten, ohne das Bild selbst
    sehen zu muessen.
    """
    if not project or not project.get("images"):
        return f"Aktuelles Projekt: '{project.get('filename', '?')}' (noch keine Bilder)."

    images = project["images"]
    lines = [f"Aktuelles Projekt: '{project['filename']}'"]
    if project.get("source_url"):
        lines.append(f"Quelle: {project['source_url']}")
    lines.append(f"Anzahl Bilder im Projekt: {len(images)}")
    lines.append("")
    lines.append("Bilder-Uebersicht (Bild-Nr, Bildtyp, aktueller Alt-Text):")

    for img in images[:MAX_IMAGES_IN_SMALLTALK_CONTEXT]:
        alt = (img.get("alt_effective") or "").strip()
        if len(alt) > MAX_ALT_LEN_IN_CONTEXT:
            alt = alt[:MAX_ALT_LEN_IN_CONTEXT].rstrip() + "..."
        if not alt:
            alt = "(noch kein Alt-Text)"
        bildtyp = img.get("image_type") or "unbekannt"
        lines.append(f"Bild {img['nr']} ({bildtyp}): {alt}")

    if len(images) > MAX_IMAGES_IN_SMALLTALK_CONTEXT:
        rest = len(images) - MAX_IMAGES_IN_SMALLTALK_CONTEXT
        lines.append(f"... und {rest} weitere Bilder (im Kontext gekuerzt, in der DB vollstaendig)")

    return "\n".join(lines)


def get_project_context(project_id: int, user_id: int) -> Optional[dict]:
    """Lade Projekt-Daten mit Berechtigungspruefung.

    Returns None wenn Projekt nicht existiert oder nicht dem User gehoert.
    images werden positional 1..N nummeriert (nach ID-ASC).
    """
    conn = get_db()
    try:
        proj = conn.execute(
            "SELECT id, filename, project_type, source_url "
            "FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        if not proj:
            return None
        rows = conn.execute(
            "SELECT id, image_type, alt_text, alt_text_edited, langbeschreibung, "
            "context_text, width, height, image_path "
            "FROM images WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()

    images = []
    for pos, img in enumerate(rows, start=1):
        d = dict(img)
        d["nr"] = pos
        d["alt_effective"] = d.get("alt_text_edited") or d.get("alt_text") or ""
        images.append(d)
    return {
        "id": proj["id"],
        "filename": proj["filename"],
        "project_type": proj["project_type"],
        "source_url": proj["source_url"],
        "images": images,
    }


_NUMBER_WORDS_DE = {
    "ein": 1, "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fuenf": 5, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
    "elf": 11, "zwoelf": 12, "zwölf": 12, "dreizehn": 13, "vierzehn": 14,
    "fuenfzehn": 15, "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17,
    "achtzehn": 18, "neunzehn": 19, "zwanzig": 20,
}


def _replace_number_words(msg: str) -> str:
    """Ersetze ausgeschriebene deutsche Zahlwoerter durch Ziffern.

    'bild eins' -> 'bild 1', 'bilder zwei und drei' -> 'bilder 2 und 3'.
    Wirkt nur auf gueltige Wort-Grenzen.
    """
    def _sub(match: "re.Match[str]") -> str:
        word = match.group(0).lower()
        return str(_NUMBER_WORDS_DE.get(word, word))

    pattern = r"\b(" + "|".join(re.escape(w) for w in _NUMBER_WORDS_DE) + r")\b"
    return re.sub(pattern, _sub, msg, flags=re.IGNORECASE)


def resolve_image_refs(user_message: str, images: list[dict]) -> tuple[list[dict], Optional[str]]:
    """Extrahiere referenzierte Bilder aus der User-Nachricht.

    Returns: (list_of_referenced_images, error_message_or_None)
    Wenn error_message gesetzt ist, soll der Bot diese als Antwort
    zurueckgeben und KEINE weitere Aktion ausfuehren.

    Erkannte Muster:
    - "Bild 3", "bild 3", "Image 3"
    - "Bild eins", "Bild zwei" (deutsche Zahlwoerter bis 20)
    - "Bild 1, 5, 7"
    - "Bilder 5-10", "Bilder 5 bis 10"
    - "alle Bilder", "alle"
    """
    msg = _replace_number_words(user_message.lower())
    total = len(images)

    refs: set[int] = set()

    # "alle Bilder" / "alle"
    if re.search(r"\b(alle\s+bilder|alle|jedes\s+bild|saemtliche)\b", msg):
        refs.update(range(1, total + 1))

    # "Bilder 5-10" oder "Bilder 5 bis 10"
    for m in re.finditer(r"\b(?:bild(?:er)?|image[s]?)\s+(\d+)\s*(?:-|bis|–)\s*(\d+)", msg):
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        refs.update(range(a, b + 1))

    # "Bild 3" oder "Bilder 1, 5, 7" oder "Bild 3 und 7"
    for m in re.finditer(r"\b(?:bild(?:er)?|image[s]?)\s+([\d,\s\sund]+)", msg):
        nums_str = m.group(1)
        for n in re.findall(r"\d+", nums_str):
            refs.add(int(n))

    # Filter: nur gueltige Positionen
    valid_refs = sorted(r for r in refs if 1 <= r <= total)

    if not valid_refs:
        return [], None  # keine Referenz erkannt — Bot soll im Stub-Modus antworten

    if len(valid_refs) > MAX_IMAGES_PER_REQUEST:
        return [], (
            f"Du hast {len(valid_refs)} Bilder genannt. Pro Anfrage kann ich "
            f"maximal {MAX_IMAGES_PER_REQUEST} verarbeiten. Soll ich die ersten "
            f"{MAX_IMAGES_PER_REQUEST} nehmen, oder nenne mir bitte eine kleinere "
            f"Auswahl (z.B. 'Bilder 1-{MAX_IMAGES_PER_REQUEST}')."
        )

    referenced = [img for img in images if img["nr"] in valid_refs]
    return referenced, None


def update_alt_text(project_id: int, image_id: int, new_alt: str) -> bool:
    """Schreibe neuen Alt-Text in images.alt_text_edited (User-Edit-Spalte).

    Returns True bei Erfolg.
    """
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE images SET alt_text_edited = ? WHERE id = ? AND project_id = ?",
            (new_alt, image_id, project_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def run_pipeline_for_image(image_id: int, project_id: int, user_id: int) -> Optional[dict]:
    """Rufe die Standard-Pipeline (v3.7) fuer ein einzelnes Bild auf.

    Wiederverwendung der bestehenden generate_alt_text-Funktion aus
    pdf_processor.py. Schreibt das Ergebnis in die DB (alt_text,
    langbeschreibung, etc.) und gibt das Result zurueck.
    """
    conn = get_db()
    try:
        img = conn.execute(
            "SELECT i.* FROM images i JOIN projects p ON i.project_id = p.id "
            "WHERE i.id = ? AND i.project_id = ? AND p.user_id = ?",
            (image_id, project_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if not img:
        return None

    from pdf_processor import generate_alt_text

    result = generate_alt_text(
        img["image_path"],
        img["context_text"] or "",
        img["image_type"] if img["image_type"] != "unknown" else None,
        img["width"] or 0,
        img["height"] or 0,
        img["original_alt"] or "",
        True,  # force_regenerate
    )

    conn = get_db()
    try:
        conn.execute(
            "UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?, "
            "langbeschreibung = ?, alt_text_edited = NULL, needs_review = ?, "
            "pipeline_steps = ?, validation_result = ?, status = 'done' "
            "WHERE id = ?",
            (
                result["alt_text"],
                result["bildtyp"],
                result.get("konfidenz", "mittel"),
                result.get("langbeschreibung", ""),
                1 if result.get("needs_review") else 0,
                result.get("pipeline_steps", ""),
                result.get("validation_result", ""),
                image_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return result
