"""PDF-Tagging — Endpunkte und Hintergrundlauf (22.09.2026, Steve + Michael Karbe + Joerg Heine).

„Barrierefrei machen“ fuer ein PDF-Dokument eines PDF- oder Formular-Projekts: PDFix taggt die
Datei (pdf_tagging.py, Heines Make_Accessible), danach werden die Bilder des Dokuments NEU
extrahiert (jetzt ueber den Strukturbaum, extraction_method pdfix) und vorhandene Alt-Texte
lagegenau uebernommen. Das getaggte PDF wird die Arbeitsdatei des Dokuments (documents.
original_path); die unveraenderte Kundendatei bleibt als documents.roh_path liegen — Neu-Taggen
setzt immer wieder darauf auf.

Ablauf und Regeln (Michael 21.09.: „nicht mit dem Upload, sondern auf Knopf; kann dauern;
Neu-Taggen erlaubt“):
  GET  /api/projects/{id}/documents/{doc}/tagging        Stand: Status, Bericht, Preis, Guthaben
  POST /api/projects/{id}/documents/{doc}/tagging        Lauf starten (Credits je Seite, Wache vorher)
  GET  /api/projects/{id}/documents/{doc}/tagging/datei  getaggtes PDF herunterladen
  - nur Besitzer, nur Projekte mit project_type pdf (Werkzeuge pdf + formular), nie im Gastweg
  - waehrend des Laufs steht das Projekt auf status 'extracting' (Generierung und Export warten,
    wie beim Upload); danach 'extracted'
  - Credits (billing AKTION_TAGGING, je Seite) werden NUR nach Erfolg verbucht
  - Ausfall von veraPDF ist kein Fehler (Bericht ohne Pruefung)
  - Nach Neustart des Servers gelten haengende Laeufe als abgebrochen (Status 'fehler', Projekt zurueck)
Kein Eintrag in der Demo (dort gibt es keine Projekte mit Konto).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

import pdf_tagging

log = logging.getLogger(__name__)

AKTION = "pdf_tagging"      # billing.AKTIONS_PREISE["pdf_tagging"] = Credits je Seite
QUELLE = "tagging"          # usage_events.quelle
STATUS_LAEUFT, STATUS_FERTIG, STATUS_FEHLER = "laeuft", "fertig", "fehler"
TAGGING_PROJEKTE = ("pdf",)                 # project_type
TAGGING_WERKZEUGE = ("pdf", "formular")     # tool
BBOX_UEBERLAPPUNG = 0.5                     # IoU, ab der ein neues Bild als dasselbe gilt (Alt-Text-Uebernahme)


@dataclass
class Deps:
    get_current_user: Callable
    get_db: Callable
    results_dir: str
    billing: object
    extract_images_from_pdf: Callable          # (pdf_path, img_dir, project_id) -> list
    bilder_uebernehmen: Callable               # (conn, project_id, document_id, images, art, file_path) -> extraction_method
    doc_label: Callable                        # (doc) -> Dateinamensbestandteil
    get_gettext: Callable = None               # (lang) -> _
    resolve_ui_language: Callable = None       # (request) -> lang
    ausgaben_anzahl: Callable = None           # (project_id) -> int (Zaehler „Ablage (n)“ im Projektkopf)


_d: Optional[Deps] = None
_laeuft: dict[int, dict] = {}   # document_id -> {"seit": ts, "project_id": id}


def _user():
    return _d.get_current_user


def _projekt_und_dokument(conn, project_id: int, document_id: int, user_id: int) -> tuple[dict, dict]:
    """Projekt des Besitzers + Dokument darin (404 fremd/unbekannt, 400 kein PDF-Projekt)."""
    project = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
    if not project:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    project = dict(project)
    if project.get("project_type") not in TAGGING_PROJEKTE or project.get("tool") not in TAGGING_WERKZEUGE:
        raise HTTPException(status_code=400, detail="Das Tagging gibt es nur für PDF-Dokumente")
    doc = conn.execute("SELECT * FROM documents WHERE id = ? AND project_id = ?", (document_id, project_id)).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    doc = dict(doc)
    if not (doc.get("original_path") or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Das Tagging gibt es nur für PDF-Dokumente")
    return project, doc


def _bericht(doc: dict) -> dict:
    try:
        b = json.loads(doc.get("tagging_bericht") or "{}")
        return b if isinstance(b, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _seiten(doc: dict) -> int:
    quelle = doc.get("roh_path") or doc.get("original_path") or ""
    try:
        return pdf_tagging.seitenzahl(quelle)
    except Exception:  # noqa: BLE001
        return 0


def stand(conn, project: dict, doc: dict, user_id: int) -> dict:
    """Alles, was die Ansicht „Dokument“ zum Tagging braucht."""
    seiten = _seiten(doc)
    pruefung = _d.billing.aktion_pruefung(user_id, AKTION, seiten) if seiten else None
    alt = conn.execute(
        "SELECT COUNT(*) FROM images WHERE document_id = ? AND ((alt_text IS NOT NULL AND alt_text <> '') OR (alt_text_edited IS NOT NULL AND alt_text_edited <> ''))",
        (doc["id"],)).fetchone()[0]
    status = doc.get("tagging_status") or ""
    laeuft = doc["id"] in _laeuft or status == STATUS_LAEUFT
    return {
        "document_id": doc["id"],
        "verfuegbar": pdf_tagging.verfuegbar(),
        "modus": pdf_tagging.lizenz_modus(),
        "getaggt": (None if doc.get("getaggt") is None else bool(doc.get("getaggt"))),
        "status": (STATUS_LAEUFT if laeuft else status),
        "laeuft": laeuft,
        "seiten": seiten,
        "preis": (pruefung or {}).get("preis", 0),
        "verfuegbar_credits": (pruefung or {}).get("verfuegbar"),
        "erlaubt": bool((pruefung or {}).get("erlaubt")) if pruefung else False,
        "fehlend": (pruefung or {}).get("fehlend", 0),
        "hat_alt_texte": int(alt or 0),
        "neu_taggen": bool(doc.get("roh_path")),
        "bericht": _bericht(doc),
        "projekt_status": project.get("status"),
    }


def _struktur(doc: dict) -> dict:
    """Kennzahlen der Arbeitsdatei fuer die Ansicht „Dokument“ (pikepdf, ohne PDFix): Sprache, Titel, Elemente."""
    try:
        s = pdf_tagging.tag_statistik(doc.get("original_path") or "")
        return {k: s.get(k) for k in ("lang", "titel", "elemente", "ueberschriften", "listen", "tabellen", "bilder", "absaetze", "lesezeichen", "testmodus")}
    except Exception:  # noqa: BLE001
        return {}


_PROJEKT_FELDER = ("id", "name", "filename", "status", "tool", "project_type", "total_images", "processed_images",
                   "alt_language", "use_context", "prompt_id", "created_at", "updated_at", "lauf_hinweis", "letzte_ansicht")


def dokument_ansicht(conn, project: dict, user_id: int) -> dict:
    """Alles fuer die Ansicht „Dokument“ (frontend/dokument.js) in einem Aufruf: Projekt (ohne Serverpfade),
    je Dokument Anzeige-Felder, Seiten, Struktur und der Tagging-Stand."""
    docs = [dict(r) for r in conn.execute(
        "SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index", (project["id"],)).fetchall()]
    aussen = []
    for d in docs:
        eintrag = {k: d.get(k) for k in ("id", "doc_index", "original_filename", "display_name", "total_images",
                                          "extraction_method", "created_at", "hinweise")}
        eintrag["getaggt"] = (None if d.get("getaggt") is None else bool(d.get("getaggt")))
        eintrag["felder"] = int(conn.execute("SELECT COUNT(*) FROM formularfelder WHERE document_id = ?", (d["id"],)).fetchone()[0] or 0)
        eintrag["seiten"] = _seiten(d)
        eintrag["struktur"] = _struktur(d)
        eintrag["tagging"] = stand(conn, project, d, user_id)
        aussen.append(eintrag)
    projekt_aussen = {k: project.get(k) for k in _PROJEKT_FELDER}
    projekt_aussen["hat_felder"] = sum(e["felder"] for e in aussen)
    return {
        "project": projekt_aussen,
        "documents": aussen,
        "ausgaben_anzahl": (_d.ausgaben_anzahl(project["id"]) if _d.ausgaben_anzahl else 0),
    }


def _vorschau_pfad(doc: dict, user_id: int, project_id: int) -> Optional[str]:
    """PNG der ersten Seite: die Seitenansicht aus der Extraktion, sonst ein eigenes Rendering (gecacht)."""
    img_dir = os.path.join(_d.results_dir, str(user_id), str(project_id), f"doc{doc.get('doc_index') or 1}")
    vorhanden = os.path.join(img_dir, "p1_seitenansicht.png")
    if os.path.isfile(vorhanden):
        return vorhanden
    quelle = doc.get("original_path") or ""
    if not os.path.isfile(quelle):
        return None
    ziel = os.path.join(img_dir, "_vorschau_p1.png")
    try:
        if os.path.isfile(ziel) and os.path.getmtime(ziel) >= os.path.getmtime(quelle):
            return ziel
        import fitz
        os.makedirs(img_dir, exist_ok=True)
        with fitz.open(quelle) as pdf:
            if len(pdf) == 0:
                return None
            pdf[0].get_pixmap(dpi=96).save(ziel)
        return ziel
    except Exception as e:  # noqa: BLE001
        log.warning("[tagging] Vorschau fuer Dokument %s nicht moeglich: %r", doc.get("id"), e)
        return None


# ---------------------------------------------------------------------------
# Hintergrundlauf
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    fa = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    fb = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = fa + fb - inter
    return inter / union if union > 0 else 0.0


_UEBERNAHME_SPALTEN = ("alt_text", "alt_text_edited", "langbeschreibung", "image_type", "status", "konfidenz",
                       "feedback", "needs_review", "gen_language", "context_mode", "alt_text_vorher", "display_name")
HASH_TOLERANZ = 12   # Hamming-Abstand der Bild-Hashes (64 Bit), bis zu dem zwei Renderings als dasselbe Bild gelten


def _dhash(pfad: str, groesse: int = 8) -> Optional[int]:
    """Wahrnehmungs-Hash (dHash, 64 Bit) einer Bilddatei: robust gegen Skalierung und leichte
    Renderunterschiede (fitz liefert das eingebettete Bild, PDFix ein Rendering des Figure-Rechtecks)."""
    try:
        from PIL import Image
        with Image.open(pfad) as im:
            im = im.convert("L").resize((groesse + 1, groesse), Image.LANCZOS)
            px = list(im.getdata())
    except Exception:  # noqa: BLE001
        return None
    bits = 0
    for y in range(groesse):
        for x in range(groesse):
            links, rechts = px[y * (groesse + 1) + x], px[y * (groesse + 1) + x + 1]
            bits = (bits << 1) | (1 if links > rechts else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _echte_bbox(row: dict) -> bool:
    """Seitenkoordinaten (fitz-Weg) oder nur Bildmasse (PDFix-Weg speichert bbox = (0, 0, Breite, Hoehe))?"""
    werte = (row.get("bbox_x0"), row.get("bbox_y0"), row.get("bbox_x1"), row.get("bbox_y1"))
    if any(v is None for v in werte):
        return False
    if werte[0] == 0 and werte[1] == 0 and werte[2] == (row.get("width") or -1) and werte[3] == (row.get("height") or -1):
        return False
    return werte[2] > werte[0] and werte[3] > werte[1]


def alt_texte_uebernehmen(conn, document_id: int, alte_bilder: list[dict]) -> int:
    """Alt-Texte der alten Bildzeilen auf die neu extrahierten Bilder desselben Dokuments uebertragen.
    Ein altes und ein neues Bild gelten als dasselbe, wenn sie auf derselben Seite liegen und
      1. beide Seitenkoordinaten haben und sich zu >= BBOX_UEBERLAPPUNG ueberlappen (fitz -> fitz), oder
      2. ihre Bild-Hashes hoechstens HASH_TOLERANZ auseinanderliegen (fitz -> PDFix, PDFix -> PDFix), oder
      3. es auf der Seite genau ein altes Bild mit Text und genau ein neues Bild gibt.
    Jedes alte Bild wird hoechstens einmal verwendet. Rueckgabe: Anzahl uebernommener Bilder."""
    neue = [dict(r) for r in conn.execute(
        "SELECT id, page_number, bbox_x0, bbox_y0, bbox_x1, bbox_y1, width, height, image_path FROM images WHERE document_id = ?",
        (document_id,)).fetchall()]
    kandidaten = [a for a in alte_bilder if (a.get("alt_text") or a.get("alt_text_edited") or a.get("status") == "done")]
    if not neue or not kandidaten:
        return 0
    hashes: dict = {}

    def h(row):
        k = ("n" if row in neue else "a", row["id"])
        if k not in hashes:
            hashes[k] = _dhash(row.get("image_path") or "") if row.get("image_path") else None
        return hashes[k]

    paare = []   # (score, neu_id, alt_id) — kleiner ist besser
    for neu in neue:
        for alt in kandidaten:
            if alt.get("page_number") != neu.get("page_number"):
                continue
            if _echte_bbox(neu) and _echte_bbox(alt):
                iou = _iou((neu["bbox_x0"], neu["bbox_y0"], neu["bbox_x1"], neu["bbox_y1"]),
                           (alt["bbox_x0"], alt["bbox_y0"], alt["bbox_x1"], alt["bbox_y1"]))
                if iou >= BBOX_UEBERLAPPUNG:
                    paare.append((1.0 - iou, neu["id"], alt["id"]))
                    continue
            hn, ha = h(neu), h(alt)
            if hn is not None and ha is not None:
                abstand = _hamming(hn, ha)
                if abstand <= HASH_TOLERANZ:
                    paare.append((1.0 + abstand / 64.0, neu["id"], alt["id"]))
    # Rueckfall 3: eindeutige Paare je Seite
    je_seite_neu: dict = {}
    je_seite_alt: dict = {}
    for n in neue:
        je_seite_neu.setdefault(n.get("page_number"), []).append(n)
    for a in kandidaten:
        je_seite_alt.setdefault(a.get("page_number"), []).append(a)
    for seite, ns in je_seite_neu.items():
        alts = je_seite_alt.get(seite) or []
        if len(ns) == 1 and len(alts) == 1:
            paare.append((3.0, ns[0]["id"], alts[0]["id"]))
    paare.sort(key=lambda p: p[0])
    benutzt_neu: set = set()
    benutzt_alt: set = set()
    alt_je_id = {a["id"]: a for a in kandidaten}
    n = 0
    for _score, neu_id, alt_id in paare:
        if neu_id in benutzt_neu or alt_id in benutzt_alt:
            continue
        best = alt_je_id[alt_id]
        werte = [best.get(s) for s in _UEBERNAHME_SPALTEN]
        conn.execute("UPDATE images SET " + ", ".join(f"{s} = ?" for s in _UEBERNAHME_SPALTEN) + " WHERE id = ?",
                     (*werte, neu_id))
        benutzt_neu.add(neu_id)
        benutzt_alt.add(alt_id)
        n += 1
    return n


def _ziel_pfad(doc: dict) -> str:
    """Das getaggte PDF liegt neben der Kundendatei im Upload-Ordner (der Export liest nur von dort)."""
    quelle = doc.get("roh_path") or doc["original_path"]
    ordner = os.path.dirname(quelle)
    stamm = os.path.splitext(os.path.basename(quelle))[0]
    if stamm.endswith("_getaggt"):
        stamm = stamm[:-len("_getaggt")]
    return os.path.join(ordner, f"{stamm}_getaggt.pdf")


def _lauf_sync(project_id: int, document_id: int, user_id: int, preis: int, sprache_vorgabe: str,
               status_vorher: str, ui_lang: str) -> None:
    """Der eigentliche Lauf (im Executor): taggen -> pruefen -> Bilder neu extrahieren -> Alt-Texte
    uebernehmen -> Dokument umhaengen -> Credits. Jeder Fehler landet als Grund im Bericht."""
    conn = _d.get_db()
    try:
        doc = dict(conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone())
    finally:
        conn.close()
    quelle = doc.get("roh_path") or doc["original_path"]
    ziel = _ziel_pfad(doc)
    ziel_tmp = ziel + f".{int(time.time())}.tmp.pdf"
    uebers = _d.get_gettext(ui_lang) if (_d.get_gettext and ui_lang) else None
    bericht: dict = {}
    try:
        bericht = pdf_tagging.taggen(quelle, ziel_tmp, sprache_vorgabe, arbeitsordner=os.path.dirname(ziel))
        bericht["verapdf"] = pdf_tagging.verapdf(ziel_tmp, uebers)
        # Bilder des Dokuments neu extrahieren — jetzt ueber den Strukturbaum.
        img_dir = os.path.join(_d.results_dir, str(user_id), str(project_id), f"doc{doc.get('doc_index') or 1}")
        os.makedirs(img_dir, exist_ok=True)
        images = _d.extract_images_from_pdf(ziel_tmp, img_dir, project_id)
        os.replace(ziel_tmp, ziel)
        conn = _d.get_db()
        try:
            alte = [dict(r) for r in conn.execute("SELECT * FROM images WHERE document_id = ?", (document_id,)).fetchall()]
            # Eine Transaktion fuer Loeschen, Neu-Eintragen, Uebernahme und Umhaengen (Muster billing.verbuche):
            # isolation_level=None, sonst setzt das sqlite3-Modul selbst ein BEGIN ab.
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM images WHERE document_id = ? AND project_id = ?", (document_id, project_id))
            methode = _d.bilder_uebernehmen(conn, project_id, document_id, images, "pdf", ziel)
            uebernommen = alt_texte_uebernehmen(conn, document_id, alte)
            bericht["bilder"] = {"vorher": len(alte), "nachher": len(images), "uebernommen": uebernommen, "methode": methode}
            conn.execute(
                "UPDATE documents SET original_path = ?, roh_path = COALESCE(NULLIF(roh_path, ''), ?), getaggt = 1, "
                "tagging_status = ?, tagging_bericht = ? WHERE id = ?",
                (ziel, quelle, STATUS_FERTIG, json.dumps(bericht, ensure_ascii=False), document_id))
            gesamt = conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ?", (project_id,)).fetchone()[0]
            fertig = conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ? AND status = 'done'", (project_id,)).fetchone()[0]
            conn.execute("UPDATE projects SET status = 'extracted', total_images = ?, processed_images = ?, "
                         "extraction_method = COALESCE((SELECT extraction_method FROM documents WHERE project_id = projects.id ORDER BY doc_index LIMIT 1), extraction_method) WHERE id = ?",
                         (gesamt, fertig, project_id))
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            conn.close()
        # Alte Bilddateien, die kein neuer Eintrag mehr nutzt, aufraeumen (nur unterhalb des Bildordners).
        neue_pfade = {os.path.realpath(i.get("image_path") or "") for i in images}
        for a in alte:
            p = a.get("image_path") or ""
            rp = os.path.realpath(p) if p else ""
            if rp and rp not in neue_pfade and rp.startswith(os.path.realpath(img_dir) + os.sep) and os.path.isfile(rp):
                try:
                    os.remove(rp)
                except OSError:
                    pass
        _d.billing.verbuche(user_id, QUELLE, AKTION, credits=preis)
        log.info("[tagging] Dokument %s fertig: %s Seiten, %s Elemente, %s Bilder (%s uebernommen), %s Credits",
                 document_id, bericht.get("seiten"), (bericht.get("nachher") or {}).get("elemente"),
                 len(images), uebernommen, preis)
    except Exception as e:  # noqa: BLE001
        grund = str(e) if isinstance(e, pdf_tagging.TaggingFehler) else "Unerwarteter Fehler beim Tagging"
        if not isinstance(e, pdf_tagging.TaggingFehler):
            log.exception("[tagging] Dokument %s fehlgeschlagen", document_id)
        else:
            log.warning("[tagging] Dokument %s fehlgeschlagen: %s", document_id, grund)
        for p in (ziel_tmp,):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        conn = _d.get_db()
        try:
            conn.execute("UPDATE documents SET tagging_status = ?, tagging_bericht = ? WHERE id = ?",
                         (STATUS_FEHLER, json.dumps({"fehler": grund, "zeit": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "modus": pdf_tagging.lizenz_modus()}, ensure_ascii=False), document_id))
            conn.execute("UPDATE projects SET status = ? WHERE id = ? AND status = 'extracting'", (status_vorher or "extracted", project_id))
            conn.commit()
        finally:
            conn.close()
    finally:
        _laeuft.pop(document_id, None)


def haengende_laeufe_zuruecksetzen() -> None:
    """Nach einem Neustart: Laeufe, die noch 'laeuft' tragen, sind abgebrochen."""
    conn = _d.get_db()
    try:
        rows = conn.execute("SELECT id, project_id FROM documents WHERE tagging_status = ?", (STATUS_LAEUFT,)).fetchall()
        for r in rows:
            conn.execute("UPDATE documents SET tagging_status = ?, tagging_bericht = ? WHERE id = ?",
                         (STATUS_FEHLER, json.dumps({"fehler": "Der Server wurde während des Taggings neu gestartet", "zeit": time.strftime("%Y-%m-%d %H:%M:%S")}), r["id"]))
            conn.execute("UPDATE projects SET status = 'extracted' WHERE id = ? AND status = 'extracting'", (r["project_id"],))
        if rows:
            conn.commit()
            log.warning("[tagging] %d haengende Laeufe zurueckgesetzt", len(rows))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def build_router(deps: Deps) -> APIRouter:
    global _d
    _d = deps
    router = APIRouter()

    @router.on_event("startup")
    async def _startup():
        try:
            haengende_laeufe_zuruecksetzen()
        except Exception as e:  # noqa: BLE001
            log.warning("[tagging] Start-Reparatur uebersprungen: %r", e)

    @router.get("/api/projects/{project_id}/documents/{document_id}/tagging")
    async def lesen(project_id: int, document_id: int, user: dict = Depends(_user())):
        conn = _d.get_db()
        try:
            project, doc = _projekt_und_dokument(conn, project_id, document_id, user["id"])
            return stand(conn, project, doc, user["id"])
        finally:
            conn.close()

    @router.post("/api/projects/{project_id}/documents/{document_id}/tagging")
    async def starten(project_id: int, document_id: int, request: Request, user: dict = Depends(_user())):
        if not pdf_tagging.verfuegbar():
            raise HTTPException(status_code=503, detail="PDF-Tagging ist auf diesem Server nicht eingerichtet")
        conn = _d.get_db()
        try:
            project, doc = _projekt_und_dokument(conn, project_id, document_id, user["id"])
            if document_id in _laeuft or doc.get("tagging_status") == STATUS_LAEUFT:
                raise HTTPException(status_code=409, detail="Das Tagging läuft bereits")
            if project.get("status") in ("extracting", "processing"):
                raise HTTPException(status_code=409, detail="Das Projekt wird gerade verarbeitet. Bitte warte, bis der Lauf fertig ist.")
            seiten = _seiten(doc)
            if seiten <= 0:
                raise HTTPException(status_code=400, detail="Die PDF konnte nicht gelesen werden")
            if seiten > pdf_tagging.MAX_SEITEN:
                raise HTTPException(status_code=400, detail=f"Das Tagging ist auf {pdf_tagging.MAX_SEITEN} Seiten begrenzt")
            pruefung = _d.billing.aktion_pruefung(user["id"], AKTION, seiten)
            if not pruefung["erlaubt"]:
                raise HTTPException(status_code=402, detail=_d.billing.credits_fehlen_detail(pruefung, "Das Tagging"))
            status_vorher = project.get("status") or "extracted"
            _laeuft[document_id] = {"seit": time.time(), "project_id": project_id}
            conn.execute("UPDATE documents SET tagging_status = ?, tagging_bericht = ? WHERE id = ?",
                         (STATUS_LAEUFT, json.dumps({"gestartet": time.strftime("%Y-%m-%d %H:%M:%S")}), document_id))
            conn.execute("UPDATE projects SET status = 'extracting' WHERE id = ?", (project_id,))
            conn.commit()
        finally:
            conn.close()
        sprache = (project.get("alt_language") or user.get("language") or "de")
        ui_lang = _d.resolve_ui_language(request) if _d.resolve_ui_language else ""
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _lauf_sync, project_id, document_id, user["id"], int(pruefung["preis"]), sprache, status_vorher, ui_lang)
        return {"gestartet": True, "document_id": document_id, "seiten": seiten, "preis": int(pruefung["preis"]),
                "modus": pdf_tagging.lizenz_modus()}

    @router.get("/api/projects/{project_id}/dokument-ansicht")
    async def ansicht(project_id: int, user: dict = Depends(_user())):
        """Datenquelle der Ansicht „Dokument“ (nur Besitzer, nur PDF-Projekte)."""
        conn = _d.get_db()
        try:
            project = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])).fetchone()
            if not project:
                raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
            project = dict(project)
            if project.get("project_type") not in TAGGING_PROJEKTE or project.get("tool") not in TAGGING_WERKZEUGE:
                raise HTTPException(status_code=400, detail="Die Ansicht Dokument gibt es nur für PDF-Projekte")
            return dokument_ansicht(conn, project, user["id"])
        finally:
            conn.close()

    @router.get("/api/projects/{project_id}/documents/{document_id}/vorschau")
    async def vorschau(project_id: int, document_id: int, user: dict = Depends(_user())):
        """PNG der ersten Seite fuer die Karte in der Ansicht „Dokument“."""
        conn = _d.get_db()
        try:
            _project, doc = _projekt_und_dokument(conn, project_id, document_id, user["id"])
        finally:
            conn.close()
        loop = asyncio.get_running_loop()
        pfad = await loop.run_in_executor(None, _vorschau_pfad, doc, user["id"], project_id)
        if not pfad:
            raise HTTPException(status_code=404, detail="Vorschau nicht gefunden")
        return FileResponse(pfad, media_type="image/png")

    @router.get("/api/projects/{project_id}/documents/{document_id}/tagging/datei")
    async def datei(project_id: int, document_id: int, user: dict = Depends(_user())):
        conn = _d.get_db()
        try:
            _project, doc = _projekt_und_dokument(conn, project_id, document_id, user["id"])
        finally:
            conn.close()
        if doc.get("tagging_status") != STATUS_FERTIG or not doc.get("roh_path"):
            raise HTTPException(status_code=404, detail="Für dieses Dokument gibt es noch keine getaggte Fassung")
        pfad = doc.get("original_path") or ""
        erlaubt = os.path.realpath(os.path.dirname(doc["roh_path"]))
        if not os.path.realpath(pfad).startswith(erlaubt + os.sep) or not os.path.isfile(pfad):
            raise HTTPException(status_code=404, detail="Datei nicht gefunden")
        return FileResponse(pfad, filename=f"{_d.doc_label(doc)}_getaggt.pdf", media_type="application/pdf")

    return router
