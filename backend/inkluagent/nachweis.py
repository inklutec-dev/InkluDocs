"""„Fertig mit Nachweis“ (Steve 17.10.2026): am Ende sagt der InkluAgent nicht nur „hier ist deine Datei“ — er zeigt, dass sie
geprüft ist: PDF/UA nach veraPDF, Alt-Texte, und den Anfang der Hörprobe, so wie ein Screenreader das Dokument vorliest, mit
„Anhören“ (Stimmen auf dem Gerät). Für Sehende der Moment, in dem sie hören, was blinde Nutzer hören.

Nichts wird neu geprüft. Quellen:
- ein Eintrag in Meine Ablage: „In PDF umwandeln“ (veraPDF der fertigen PDF + Hörprobe aus dem Word-Dokument), „PDF herunterladen“
  (veraPDF, keine Hörprobe) — aus_ausgabe
- die Prüfdatei eines Dokuments im PDF-Projekt (veraPDF + Hörprobe der fertigen Datei) — aus_pruefdatei
Ergebnis: ein Anhang {"art": "nachweis", …} unter der Antwort bzw. der Rückmeldung (frontend/inkluagent-nachrichten.js
inkluagentNachweisKarte). Werkzeuge geben ihn als eigenes Feld „nachweis“ neben „anhang“ zurück; agent_loop und auftraege nehmen
ihn heraus, bevor das Modell das Ergebnis sieht — Zeilen aus dem Dokument gehen nie an das Modell und stehen in der Seite nur als
textContent. Nie eine Ausnahme nach außen (ein fehlender Nachweis hält kein Werkzeug auf).
"""
from __future__ import annotations

import importlib
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)

HOERPROBE_ZEILEN = 8   # der Anfang — die ganze Hörprobe steht in der Ablage bzw. der Barrierefreiheitsprüfung
ZEILE_MAX = 300


def _main():
    return importlib.import_module("main")


def _zeilen(zeilen) -> list:
    out = []
    for z in zeilen or []:
        z = " ".join(str(z or "").split())
        if not z:
            continue
        out.append(z if len(z) <= ZEILE_MAX else z[:ZEILE_MAX - 1].rstrip() + "…")
        if len(out) >= HOERPROBE_ZEILEN:
            break
    return out


def _urteil(pruefung) -> Optional[bool]:
    """veraPDF: True bestanden, False nicht bestanden, None nicht geprüft (Dienst weg)."""
    if not isinstance(pruefung, dict) or pruefung.get("nicht_geprueft") or "bestanden" not in pruefung:
        return None
    return bool(pruefung.get("bestanden"))


def _sprache(lang) -> Optional[str]:
    lang = str(lang or "").strip()[:2].lower()
    return lang if lang.isalpha() and len(lang) == 2 else None


def aus_ausgabe(user_id: int, ausgabe_id: int, lang: Optional[str] = None) -> Optional[dict]:
    """Nachweis zu einem Eintrag in Meine Ablage — None, wenn es ihn für dieses Konto nicht gibt oder er nichts Geprüftes enthält."""
    try:
        m = _main()
        row = m._ausgabe_row(int(user_id), int(ausgabe_id))
        if not row:
            return None
        a = m._ausgabe_dict(row)
        bericht = [d for d in (json.loads(row["bericht"] or "[]") or []) if isinstance(d, dict)]
        if not bericht:
            return None
        urteile = [_urteil(d.get("pruefung")) for d in bericht]
        bestanden = None if None in urteile else all(urteile)
        befunde = sum(1 for d in bericht for p in ((d.get("pruefung") or {}).get("punkte") or []) if p.get("status") != "ok")
        mit_bildern = [d for d in bericht if d.get("bilder") is not None]
        erst = next((d for d in bericht if d.get("hoerprobe")), None)
        hp = list((erst or {}).get("hoerprobe") or [])
        if row["art"] == "pdfua" and len(hp) > 1:
            hp = hp[:1] + hp[2:]   # Word-Hörprobe: Zeile 0 Titel, Zeile 1 Sprache (steht als lang an der Karte)
        return {"art": "nachweis", "quelle": "ablage", "ausgabe_id": int(row["id"]),
                "dokument": str(a.get("dokument") or a.get("projekt") or "")[:200], "dokumente": len(bericht),
                "bestanden": bestanden, "befunde": befunde,
                "bilder": sum(int(d.get("bilder") or 0) for d in mit_bildern) if mit_bildern else None,
                "alt_texte": sum(int(d.get("alt_texte") or 0) for d in mit_bildern) if mit_bildern else None,
                "hoerprobe": _zeilen(hp),
                "hoerprobe_dokument": (str(erst.get("dokument") or "")[:200] if erst and len(bericht) > 1 else None),
                "lang": _sprache(lang), "ablage_url": f"/ablage#ausgabe-{int(row['id'])}"}
    except Exception:  # noqa: BLE001
        log.exception("Nachweis zur Ausgabe %s nicht gebaut (ignoriert)", ausgabe_id)
        return None


def aus_pruefdatei(project_id: int, document_id: int, dokument: str, daten: dict, bilder_mit_alt: Optional[int] = None) -> Optional[dict]:
    """Nachweis aus der Prüfdatei eines Dokuments (main._abschluss_dokument mit voll=True) — None ohne Prüfdatei."""
    try:
        pd = (daten or {}).get("pruefdatei") or {}
        if not pd:
            return None
        zeilen = [z for s in ((daten.get("hoerprobe") or {}).get("seiten") or []) for z in (s.get("zeilen") or [])]
        return {"art": "nachweis", "quelle": "pruefdatei", "project_id": int(project_id), "document_id": int(document_id),
                "dokument": str(dokument or "")[:200], "dokumente": 1,
                "bestanden": bool(pd.get("bestanden")) if pd.get("verapdf_moeglich") else None,
                "befunde": pd.get("anzahl_probleme"), "aktuell": bool(pd.get("aktuell")),
                "bilder": daten.get("bilder"), "alt_texte": bilder_mit_alt,
                "hoerprobe": _zeilen(zeilen), "hoerprobe_dokument": None, "lang": _sprache(daten.get("sprache")),
                "pruefung_url": f"/app?projekt={int(project_id)}&ansicht=abschluss"}
    except Exception:  # noqa: BLE001
        log.exception("Nachweis zur Prüfdatei %s nicht gebaut (ignoriert)", document_id)
        return None


def alt_texte_im_dokument(project_id: int, document_id: int) -> Optional[int]:
    """Wie viele Bilder des Dokuments einen Alt-Text haben (eigener Text, KI-Text oder aus der Datei)."""
    try:
        from database import get_db
        conn = get_db()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ? AND document_id = ? AND TRIM(COALESCE("
                                    "alt_text_edited, alt_text, original_alt, '')) != ''", (project_id, document_id)).fetchone()[0])
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        log.exception("Nachweis: Alt-Texte nicht gezaehlt (ignoriert)")
        return None


def herausnehmen(ergebnis) -> list:
    """Den Nachweis aus einem Werkzeug-Ergebnis nehmen (vor dem Modell): Liste von Anhängen, leer ohne."""
    if not isinstance(ergebnis, dict):
        return []
    n = ergebnis.pop("nachweis", None)
    liste = n if isinstance(n, list) else [n]
    return [x for x in liste if isinstance(x, dict) and x.get("art") == "nachweis"] if ergebnis.get("ok") else []
