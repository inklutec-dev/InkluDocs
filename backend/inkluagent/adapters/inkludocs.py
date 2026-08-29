"""InkluDocs-spezifischer Adapter: stellt Bruecke zwischen chat_engine
und InkluDocs-Datenmodell (projects/images, Pipeline) her."""
import logging
import re
from typing import Optional

import billing  # Abo-/Credit-System Etappe 1
from database import get_db

log = logging.getLogger(__name__)


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

    Multi-Datei (08.06.2026): bei Projekten mit mehreren Dokumenten wird
    pro Dokument eine Untersektion gerendert, und jede Bild-Zeile nutzt
    das ui_label ('Dokument X, Bild N'). So bleibt jede Referenz fuer den
    Bot eindeutig — er kann dieselbe Schreibweise zurueckgeben.
    """
    if not project or not project.get("images"):
        return f"Aktuelles Projekt: '{project.get('filename', '?')}' (noch keine Bilder)."

    images = project["images"]
    documents = project.get("documents") or []
    multi_doc = project.get("multi_doc") or False
    lines = [f"Aktuelles Projekt: '{project['filename']}'"]
    if project.get("source_url"):
        lines.append(f"Quelle: {project['source_url']}")
    lines.append(f"Anzahl Bilder im Projekt: {len(images)}")

    if multi_doc:
        lines.append(f"Anzahl Dokumente: {len(documents)}")
        lines.append("")
        lines.append(
            "Bilder werden im UI pro Dokument bei 1 nummeriert. Verwende in deinen "
            "Antworten die Schreibweise 'Dokument X, Bild N', damit Referenzen eindeutig "
            "bleiben. Wenn der User nur 'Bild N' sagt und das mehrdeutig ist, frage kurz "
            "nach, welches Dokument gemeint ist."
        )
        lines.append("")
        # Pro Dokument eine Untersektion ausspielen
        budget_left = MAX_IMAGES_IN_SMALLTALK_CONTEXT
        for doc in documents:
            doc_imgs = [i for i in images if (i.get("doc_index") or 0) == doc["doc_index"]]
            doc_label = (doc.get("display_name") or "").strip() or (doc.get("original_filename") or "").strip()
            lines.append(f"Dokument {doc['doc_index']}: '{doc_label}' ({len(doc_imgs)} Bilder)")
            shown = 0
            for img in doc_imgs:
                if budget_left <= 0:
                    break
                alt = (img.get("alt_effective") or "").strip()
                if len(alt) > MAX_ALT_LEN_IN_CONTEXT:
                    alt = alt[:MAX_ALT_LEN_IN_CONTEXT].rstrip() + "..."
                if not alt:
                    alt = "(noch kein Alt-Text)"
                bildtyp = img.get("image_type") or "unbekannt"
                lines.append(f"  {img['ui_label']} ({bildtyp}): {alt}")
                budget_left -= 1
                shown += 1
            if shown < len(doc_imgs):
                lines.append(f"  ... und {len(doc_imgs) - shown} weitere Bilder in diesem Dokument (gekuerzt)")
            lines.append("")
        if budget_left <= 0:
            lines.append("(weitere Dokumente/Bilder im Kontext gekuerzt, in der DB vollstaendig)")
    else:
        lines.append("")
        lines.append("Bilder-Uebersicht (Bild-Nr, Bildtyp, aktueller Alt-Text):")
        for img in images[:MAX_IMAGES_IN_SMALLTALK_CONTEXT]:
            alt = (img.get("alt_effective") or "").strip()
            if len(alt) > MAX_ALT_LEN_IN_CONTEXT:
                alt = alt[:MAX_ALT_LEN_IN_CONTEXT].rstrip() + "..."
            if not alt:
                alt = "(noch kein Alt-Text)"
            bildtyp = img.get("image_type") or "unbekannt"
            lines.append(f"{img['ui_label']} ({bildtyp}): {alt}")
        if len(images) > MAX_IMAGES_IN_SMALLTALK_CONTEXT:
            rest = len(images) - MAX_IMAGES_IN_SMALLTALK_CONTEXT
            lines.append(f"... und {rest} weitere Bilder (im Kontext gekuerzt, in der DB vollstaendig)")

    return "\n".join(lines)


def get_project_context(project_id: int, user_id: int) -> Optional[dict]:
    """Lade Projekt-Daten mit Berechtigungspruefung.

    Returns None wenn Projekt nicht existiert oder nicht dem User gehoert.

    Multi-Datei (08.06.2026, Steve): Bilder werden ZUSAETZLICH zur globalen
    Reihenfolge (nr, 1..N) auch pro Dokument indiziert (doc_position, 1..M).
    'ui_label' liefert den Anzeige-String, den der Chatbot in Replies nutzt —
    bei Single-Doc bleibt es schlicht 'Bild N', bei Multi-Doc wird daraus
    'Dokument X, Bild N', damit Referenzen eindeutig bleiben. Die Reihenfolge
    pro Dokument ist (page_number, image_index) — identisch zur Frontend-
    Anzeige im Doc-Block.
    """
    conn = get_db()
    try:
        proj = conn.execute(
            "SELECT id, filename, project_type, source_url, tool "
            "FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        if not proj:
            return None
        docs = conn.execute(
            "SELECT id, doc_index, original_filename, display_name "
            "FROM documents WHERE project_id = ? ORDER BY doc_index",
            (project_id,),
        ).fetchall()
        # Reihenfolge im Projekt-Kontext: nach Dokument-Index, dann Seite, dann
        # image_index — exakt wie das Frontend gruppiert. id_ASC ist hier
        # ungeeignet, weil ein angehaengtes zweites Dokument hoehere ids hat
        # und damit der globale nr-Index falsch nach hinten waeren.
        rows = conn.execute(
            "SELECT i.id, i.image_type, i.alt_text, i.alt_text_edited, i.langbeschreibung, "
            "i.context_text, i.width, i.height, i.image_path, i.page_number, i.image_index, "
            "i.document_id, d.doc_index AS doc_index "
            "FROM images i "
            "LEFT JOIN documents d ON d.id = i.document_id "
            "WHERE i.project_id = ? "
            "ORDER BY COALESCE(d.doc_index, 0), i.page_number, i.image_index, i.id ASC",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()

    documents = [dict(d) for d in docs]
    multi_doc = len(documents) > 1

    # Doc-Position pro Bild errechnen: pro doc_index bei 1 anfangen.
    images = []
    per_doc_counter: dict[int, int] = {}
    for pos, img in enumerate(rows, start=1):
        d = dict(img)
        d["nr"] = pos
        d["alt_effective"] = d.get("alt_text_edited") or d.get("alt_text") or ""
        di = d.get("doc_index") or 0
        per_doc_counter[di] = per_doc_counter.get(di, 0) + 1
        d["doc_position"] = per_doc_counter[di]
        # ui_label: bei Multi-Doc-Projekten DIE eindeutige Referenz, sonst
        # weiterhin schlicht "Bild N" wie bisher.
        if multi_doc and di:
            d["ui_label"] = f"Dokument {di}, Bild {d['doc_position']}"
        else:
            d["ui_label"] = f"Bild {d['doc_position']}"
        images.append(d)

    return {
        "id": proj["id"],
        "filename": proj["filename"],
        "project_type": proj["project_type"],
        "tool": proj["tool"],
        "source_url": proj["source_url"],
        "documents": documents,
        "multi_doc": multi_doc,
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

    Multi-Datei (08.06.2026): zusaetzliche eindeutige Doc-Referenzen:
    - "Dokument 2, Bild 1" / "Doc 2 Bild 1" / "D2B1" / "Bild 1 in Dokument 2"
    - "alle Bilder in Dokument 2"

    Backwards-Compat: ein blankes "Bild N" wird weiterhin als PROJEKTWEITE
    Position N interpretiert (== Position in der Liste 'images', sortiert
    nach (doc_index, page, image_index)). Bei Single-Doc-Projekten ist das
    identisch zur per-Doc-Position. Bei Multi-Doc-Projekten loest 'Bild N'
    eindeutig auf das N-te Bild im Projekt — der LLM-Prompt weist ihn an,
    in Replies das eindeutige ui_label zu nutzen, sodass der User auch
    weiss, in welchem Dokument er gelandet ist.
    """
    msg = _replace_number_words(user_message.lower())
    total = len(images)

    # ── 1) Eindeutige Doc-Refs zuerst (binden Dokument + Bild) ─────────
    # Wir sammeln (doc_index, doc_position)-Tupel; aufgeloest am Schluss.
    doc_bild_refs: set[tuple[int, int]] = set()
    doc_only_refs: set[int] = set()  # "alle Bilder in Dokument 3"

    # "Dokument 2, Bild 5" / "Dokument 2 Bild 5" / "Doc 2 Bild 5"
    # / "doc2 bild5" / "Bild 5 in Dokument 2" / "Bild 5 (Dokument 2)"
    DOC = r"(?:dokument|doc|dok)\s*(\d+)"
    BLD = r"(?:bild|image)\s*(\d+)"
    for m in re.finditer(rf"{DOC}\s*[,:\-]?\s*{BLD}", msg):
        doc_bild_refs.add((int(m.group(1)), int(m.group(2))))
    for m in re.finditer(rf"{BLD}\s*(?:in|aus|im|von)?\s*(?:[(\[])?\s*{DOC}\s*(?:[)\]])?", msg):
        doc_bild_refs.add((int(m.group(2)), int(m.group(1))))
    # Kurzform "D2B1" oder "D2 B1"
    for m in re.finditer(r"\bd\s*(\d+)\s*[-_:]?\s*b\s*(\d+)\b", msg):
        doc_bild_refs.add((int(m.group(1)), int(m.group(2))))

    # "alle Bilder in Dokument 2" / "alle in Dokument 2"
    for m in re.finditer(rf"(?:alle\s+bilder|alles|alle)\s+(?:in|aus|im|von)\s+{DOC}", msg):
        doc_only_refs.add(int(m.group(1)))
    # "Dokument 2 — alle"
    for m in re.finditer(rf"{DOC}\s*[—,–-]?\s*alle(?:\s+bilder)?\b", msg):
        doc_only_refs.add(int(m.group(1)))

    # ── 2) Klassische projektweite Refs (Position 1..total) ─────────────
    project_wide: set[int] = set()
    # "alle Bilder" / "alle"
    if re.search(r"\b(alle\s+bilder|alle|jedes\s+bild|saemtliche)\b", msg):
        # Nur projektweit "alle", wenn nicht in Verbindung mit Dokument
        # (das wird oben ueber doc_only_refs eingefangen). Doppelte
        # Eintraege sind unschaedlich (set).
        if not doc_only_refs:
            project_wide.update(range(1, total + 1))

    # "Bilder 5-10" oder "Bilder 5 bis 10" — wenn vor "Bilder" kein
    # "Dokument N," steht, projektweit.
    for m in re.finditer(r"\b(?:bild(?:er)?|image[s]?)\s+(\d+)\s*(?:-|bis|–)\s*(\d+)", msg):
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        project_wide.update(range(a, b + 1))

    # "Bild 3" oder "Bilder 1, 5, 7" oder "Bild 3 und 7"
    for m in re.finditer(r"\b(?:bild(?:er)?|image[s]?)\s+([\d,\s\sund]+)", msg):
        nums_str = m.group(1)
        for n in re.findall(r"\d+", nums_str):
            project_wide.add(int(n))

    # Die Doc-Refs koennten "Bild 5" doppelt als projektweite Position 5
    # geliefert haben — wir filtern projektweite Positionen, die schon
    # ueber einen Doc-Ref abgedeckt sind, nicht zusaetzlich raus, weil
    # das in seltenen Faellen tatsaechlich beides referenziert. Aber: wenn
    # doc_bild_refs vorhanden ist UND keine eigenstaendigen 'Bild N'-
    # Mentions ohne Dokument-Kontext im Text — dann sollen die Bild-
    # Position-Refs nicht doppelt ausgewertet werden. Heuristik: wenn
    # mindestens ein doc_bild_ref existiert UND die projektweite Position
    # exakt eine Bild-Nummer im Doc-Ref ist, lassen wir sie weg, um nicht
    # versehentlich ein "Bild 5" aus einem anderen Dokument zu binden.
    if doc_bild_refs:
        bild_nums_from_doc = {b for (_d, b) in doc_bild_refs}
        project_wide = project_wide - bild_nums_from_doc

    # ── 3) Resolve auf images-Liste ─────────────────────────────────────
    referenced_ids: set[int] = set()

    # Doc-only: alle Bilder im Dokument
    if doc_only_refs:
        for img in images:
            di = img.get("doc_index") or 0
            if di in doc_only_refs:
                referenced_ids.add(img["id"])

    # Doc + Bild-Position innerhalb des Dokuments
    for (di, pos) in doc_bild_refs:
        for img in images:
            if (img.get("doc_index") or 0) == di and img.get("doc_position") == pos:
                referenced_ids.add(img["id"])
                break

    # Projektweit per nr (Position in der globalen Liste 1..total)
    valid_pw = {n for n in project_wide if 1 <= n <= total}
    if valid_pw:
        for img in images:
            if img["nr"] in valid_pw:
                referenced_ids.add(img["id"])

    if not referenced_ids:
        return [], None  # keine Referenz erkannt

    referenced = [img for img in images if img["id"] in referenced_ids]
    # Reihenfolge wie in der Quell-Liste beibehalten

    if len(referenced) > MAX_IMAGES_PER_REQUEST:
        return [], (
            f"Du hast {len(referenced)} Bilder genannt. Pro Anfrage kann ich "
            f"maximal {MAX_IMAGES_PER_REQUEST} verarbeiten. Soll ich die ersten "
            f"{MAX_IMAGES_PER_REQUEST} nehmen, oder nenne mir bitte eine kleinere "
            f"Auswahl (z.B. 'Bilder 1-{MAX_IMAGES_PER_REQUEST}')."
        )

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


# Review-Befund 6 (31.07.2026): EINE Formulierung fuer die Kontingent-Sperre
# im Chatbot — gleiche freundliche Meldung wie in tools/altext.py.
KONTINGENT_MELDUNG = ("Das Monatskontingent dieses Kontos ist aufgebraucht. "
                      "Unter Einstellungen → Abo & Verbrauch gibt es Zusatz-Credits.")


def run_pipeline_for_image(image_id: int, project_id: int, user_id: int) -> Optional[dict]:
    """Rufe die Standard-Pipeline (v3.7) fuer ein einzelnes Bild auf.

    Wiederverwendung der bestehenden generate_alt_text-Funktion aus
    pdf_processor.py. Schreibt das Ergebnis in die DB (alt_text,
    langbeschreibung, etc.) und gibt das Result zurueck.
    """
    # Abo-Etappe 2: Kontingent-Wache VOR dem Pipeline-Start (dieser Adapter
    # ist der gemeinsame Trichter beider Chatbot-Wege). Greift nur bei
    # ABO_ENFORCEMENT=on, nie fuer Admins. Rueckgabe ist ein MARKER-dict
    # statt None (Review-Befund 6, 31.07.2026): None bedeutet weiterhin
    # "Bild nicht gefunden" — die Kontingent-Sperre muessen die Aufrufer
    # aber ANSAGEN koennen, statt das Bild stumm zu ueberspringen.
    _wache = billing.aktion_pruefung(user_id, "bild_generierung")
    if not _wache["erlaubt"]:
        log.info("run_pipeline_for_image: Guthaben reicht nicht (user=%s, image=%s, preis=%s, verfuegbar=%s)",
                 user_id, image_id, _wache["preis"], _wache["verfuegbar"])
        return {"kontingent_erschoepft": True}
    conn = get_db()
    try:
        img = conn.execute(
            "SELECT i.*, p.alt_language AS alt_language, p.prompt_id AS prompt_id FROM images i "
            "JOIN projects p ON i.project_id = p.id "
            "WHERE i.id = ? AND i.project_id = ? AND p.user_id = ?",
            (image_id, project_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if not img:
        return None

    # Eigener Prompt (06.07.2026): aktive Projekt-Einstellung gilt auch fuer
    # Generierungen ueber den InkluAgent (gleicher Pipeline-Pfad).
    user_prompt = ""
    if img["prompt_id"]:
        conn = get_db()
        try:
            _up = conn.execute(
                "SELECT prompt_text FROM user_prompts WHERE id = ? AND user_id = ?",
                (img["prompt_id"], user_id),
            ).fetchone()
        finally:
            conn.close()
        if _up and _up["prompt_text"]:
            user_prompt = _up["prompt_text"]

    from pdf_processor import generate_alt_text

    result = generate_alt_text(
        img["image_path"],
        img["context_text"] or "",
        img["image_type"] if img["image_type"] != "unknown" else None,
        img["width"] or 0,
        img["height"] or 0,
        img["original_alt"] or "",
        True,  # force_regenerate
        language=(img["alt_language"] or "de"),  # Projekt-Ausgabesprache (03.07.2026)
        user_prompt=user_prompt,
    )

    conn = get_db()
    try:
        conn.execute(
            "UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?, "
            "langbeschreibung = ?, alt_text_edited = NULL, needs_review = ?, "
            "pipeline_steps = ?, validation_result = ?, gen_language = ?, status = 'done' "
            "WHERE id = ?",
            (
                result["alt_text"],
                result["bildtyp"],
                result.get("konfidenz", "mittel"),
                result.get("langbeschreibung", ""),
                1 if result.get("needs_review") else 0,
                result.get("pipeline_steps", ""),
                result.get("validation_result", ""),
                (img["alt_language"] or "de"),
                image_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    # Abo-Etappe-1: Chatbot-Generierung kostet den Alt-Text-Preis (AKTIONS_PREISE
    # bild_generierung, seit 29.08.2026 = 5). Dieser Adapter ist der
    # gemeinsame Trichter beider Chatbot-Wege (chat_engine + Tool-Schicht);
    # laeuft mit force_regenerate, also nie aus dem Cache.
    billing.verbuche(user_id, "chatbot", image_id=image_id)
    return result
