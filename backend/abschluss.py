"""Station „Abschlussprüfung“ eines PDF-Projekts (24.09.2026, Steve; Name vorläufig).

Letzter Schritt nach Dokument (hochladen, taggen), Alt-Texte und Quickinfos (bearbeiten): anschauen, anhören,
Probleme finden, herunterladen. Geprüft wird immer die FERTIGE Datei — genau die PDF, die der Kunde beim
Herunterladen bekommt (Struktur + Alt-Texte + Quickinfos, derselbe Bau wie der Export in main.py
_build_pdf_for_document). Die Prüfdatei kostet nichts; Credits kostet wie bisher erst das Herunterladen.

Ablage je Dokument: RESULTS_DIR/<user>/<projekt>/_abschluss/doc<id>.pdf + doc<id>.json (Stand, Fingerabdruck,
veraPDF-Bericht, fehlende Zeilen). Der FINGERABDRUCK über Arbeitsdatei, Alt-Texte und Quickinfos zeigt, ob die
Prüfdatei noch zum Stand in InkluDocs passt („nicht mehr aktuell“ nach einer Änderung).

Probleme (Quelle je Eintrag, immer mit Seite, wenn bekannt):
  - technisch: veraPDF (PDF/UA-1) der fertigen Datei, in Klartext (pdfua_export.klartext);
  - Struktur: leere Überschriften, Ebenensprünge, Grafiken ohne Alt-Text, Tabellen ohne Kopfzellen,
    Formularfelder ohne Quickinfo — gelesen aus dem Tag-Baum der fertigen Datei (pdf_struktur);
  - Vollständigkeit: sichtbare Textzeilen (PyMuPDF), die im Tag-Baum nicht vorkommen (tolerant: 60 % der Wörter;
    Kopf-/Fußzeilen, die sich auf mehreren Seiten wiederholen, und reine Seitenzahlen gelten als Artefakt);
  - KI-basierte Prüfung: Befunde der letzten Prüfung des Dokuments (Stand der getaggten Datei).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import time
from typing import Callable, Optional

# Buchstaben (jedes Alphabet) und Ziffern getrennt: im Baum zusammengeklebte Woerter („Unterrichtsideen1./2. Klasse“)
# sollen die sichtbare Zeile „Unterrichtsideen“ trotzdem finden (Ritterturnier 24.09.2026); kyrillische, griechische
# und alle anderen Buchstaben zaehlen mit (Pruefbericht 24.09.2026).
_WORT = re.compile(r"[^\W\d_]{3,}|\d{3,}")
# Version des Pruefdatei-Baus: steigt, wenn sich der Bau aendert — alte Pruefdateien gelten dann als nicht aktuell.
ABSCHLUSS_VERSION = 3   # 3: Strukturlesung mit Leerzeichen-Korrektur (28.09.2026) — alte Pruefdateien gelten als nicht aktuell
_NUR_ZAHL = re.compile(r"^[\s\d\W]{1,8}$")
VOLLSTAENDIG_ANTEIL = 0.6
# NUR veraPDF (Michael Karbe, Feedback 20260928 - 2, Punkte 8 und 9): „für den Beginn auf die Prüffunktionen von veraPDF
# beschränken. Der Anwender bekommt dann das gleiche Resultat wie bei einer lokalen Prüfung … jede falsche Meldung
# vermeiden“; KI-Prüfung ausblenden. Die EIGENEN Prüfungen (Struktur, Vollständigkeit, KI) bleiben im Code, werden aber
# nicht gezeigt — Anlass war der Fehlalarm vom 28.09. (Vollständigkeit). Wieder einschalten nur, wenn sie verlässlich sind
# (Steve 29.09.: „wichtig ist, dass das Ergebnis am Ende wirklich korrekt ist“; evtl. ganz neu, Vorlage: veraPDF-
# „PDF4WCAG Human Checks“ von Dual Lab, Michaels Mail 29.09.).
import funktionen as _funktionen   # seit 30.09.2026 EIN Ort fuer Oberflaeche, Chatbot und Endpunkte
EIGENE_PRUEFUNGEN = _funktionen.EIGENE_PRUEFUNGEN
MAX_FEHLEND_JE_SEITE = 12


def _identitaet(s: str) -> str:
    return s


def pfade(results_dir: str, user_id: int, project_id: int, doc_id: int) -> tuple[str, str, str]:
    ordner = os.path.join(results_dir, str(int(user_id)), str(int(project_id)), "_abschluss")
    return ordner, os.path.join(ordner, f"doc{int(doc_id)}.pdf"), os.path.join(ordner, f"doc{int(doc_id)}.json")


def fingerabdruck(doc: dict, alt_texte: list, quickinfos: list, ersteller: str = "") -> str:
    """Stand, aus dem die Prüfdatei gebaut ist — alles, was in die fertige Datei eingeht: Arbeitsdatei (Pfad, Größe,
    Zeit in ns), Anzeigename (wird Dokumenttitel), Ersteller des Kontos, Alt-Texte je Bild, Quickinfos je Feld-Anker,
    Version des Baus. quickinfos: [(anker, text)] — je Feld, nicht je Name (Optionsgruppen teilen sich Namen)."""
    pfad = doc.get("original_path") or ""
    try:
        st = os.stat(pfad)
        datei = f"{pfad}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        datei = pfad
    h = hashlib.sha256()
    h.update(f"v{ABSCHLUSS_VERSION}\x1e{datei}\x1e{doc.get('display_name') or ''}\x1e{ersteller or ''}".encode("utf-8", "replace"))
    for img_id, text in sorted(alt_texte, key=lambda x: x[0]):
        h.update(f"\x1e{img_id}\x1f{text if text is not None else '<leer>'}".encode("utf-8", "replace"))
    for anker, text in sorted(quickinfos, key=lambda x: str(x[0])):
        h.update(f"\x1d{anker}\x1f{text or ''}".encode("utf-8", "replace"))
    return h.hexdigest()


def meta_lesen(meta_pfad: str) -> Optional[dict]:
    try:
        with open(meta_pfad, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def meta_schreiben(meta_pfad: str, meta: dict) -> None:
    tmp = meta_pfad + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp, meta_pfad)


# ---------------------------------------------------------------------------
# Vollständigkeit
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join((s or "").split())


def fehlende_zeilen(pdf_pfad: str, struktur: dict) -> list[dict]:
    """Sichtbare Textzeilen, deren Wörter im Tag-Baum derselben Seite fehlen. [{seite, text}]"""
    import fitz
    worte_je_seite: dict = collections.defaultdict(set)
    for e in struktur.get("elemente") or []:
        for feld in ("text", "alt", "actual"):
            for w in _WORT.findall((e.get(feld) or "").lower()):
                worte_je_seite[int(e.get("seite") or 0)].add(w)
    alle = set().union(*worte_je_seite.values()) if worte_je_seite else set()
    zeilen_je_seite: dict = {}
    vorkommen: collections.Counter = collections.Counter()
    with fitz.open(pdf_pfad) as d:
        for pno, page in enumerate(d, 1):
            liste = []
            for b in page.get_text("dict").get("blocks", []):
                for ln in b.get("lines", []):
                    t = _norm("".join(s.get("text", "") for s in ln.get("spans", [])))
                    if t:
                        liste.append(t)
            zeilen_je_seite[pno] = liste
            for t in set(liste):
                vorkommen[t.lower()] += 1
    out = []
    for pno, liste in zeilen_je_seite.items():
        n = 0
        da = worte_je_seite.get(pno, set()) | worte_je_seite.get(0, set())
        gemeldet = set()
        for t in liste:
            if t.lower() in gemeldet:
                continue   # dieselbe Zeile doppelt gesetzt (Schatten-/Konturtext) nur einmal melden
            ws = _WORT.findall(t.lower())
            if not ws or _NUR_ZAHL.match(t):
                continue
            if vorkommen[t.lower()] >= 2 and len(zeilen_je_seite) >= 2:
                continue   # wiederholter Kopf-/Fußtext = Artefakt, darf fehlen
            if sum(1 for w in ws if w in da) / len(ws) >= VOLLSTAENDIG_ANTEIL:
                continue
            if sum(1 for w in ws if w in alle) / len(ws) >= VOLLSTAENDIG_ANTEIL:
                continue   # steht im Baum, nur auf anderer Seite zugeordnet
            out.append({"seite": pno, "text": t[:160]})
            gemeldet.add(t.lower())
            n += 1
            if n >= MAX_FEHLEND_JE_SEITE:
                break
    return out


# ---------------------------------------------------------------------------
# Probleme
# ---------------------------------------------------------------------------

def _kurz(s: str, n: int = 120) -> str:
    s = _norm(s)
    return s if len(s) <= n else s[:n].rstrip() + " …"


def struktur_probleme(struktur: dict, _: Callable[[str], str] = _identitaet, quickinfos: Optional[dict] = None) -> list[dict]:
    elemente = struktur.get("elemente") or []
    out = []
    vorher = 0
    ohne_alt: collections.Counter = collections.Counter()   # Grafiken ohne Alt-Text: eine Zeile je Seite
    for idx, e in enumerate(elemente):
        typ = e.get("typ") or ""
        seite = int(e.get("seite") or 0)
        if typ.startswith("H") and typ[1:].isdigit():
            ebene = int(typ[1:])
            text = _norm(e.get("text") or "")
            if not text:
                out.append({"seite": seite, "art": "struktur", "text": _("Leere Überschrift (Ebene {n}): Ein Screenreader liest „Überschrift, leer“.").format(n=ebene)})
            elif vorher and ebene > vorher + 1:
                out.append({"seite": seite, "art": "struktur", "text": _("Überschriften-Ebene springt von {a} auf {b}: „{t}“.").format(a=vorher, b=ebene, t=_kurz(text, 80))})
            vorher = ebene
        elif typ == "Figure" and not (e.get("alt") or e.get("actual")):
            ohne_alt[seite] += 1
        elif typ == "Table":
            praefix = e["id"] + "."
            kopf = any(f.get("typ") == "TH" for f in elemente[idx + 1:] if f.get("id", "").startswith(praefix))
            if not kopf:
                out.append({"seite": seite, "art": "struktur", "text": _("Tabelle ohne Kopfzellen: Ein Screenreader kann die Spalten nicht benennen.")})
        elif typ == "Form":
            name = e.get("feldname") or ""
            if not (e.get("quickinfo") or ((quickinfos or {}).get(name) if name else "")):
                out.append({"seite": seite, "art": "formular", "text": _("Formularfeld „{n}“ ohne Quickinfo.").format(n=name or _("ohne Namen"))})
    for seite, n in sorted(ohne_alt.items()):
        out.append({"seite": seite, "art": "alttext",
                    "text": _("Grafik ohne Alt-Text.") if n == 1 else _("{n} Grafiken ohne Alt-Text.").format(n=n)})
    return out


def _struktur_seiten_je_regel(struktur: Optional[dict]) -> dict:
    """Regel -> Seiten aus dem Strukturbaum, fuer Regeln, deren veraPDF-Kontext keine Seite nennt."""
    ohne = {"Figure": set(), "Formula": set()}
    for e in ((struktur or {}).get("elemente") or []):
        seite = int(e.get("seite") or 0)
        typ = e.get("typ") or ""
        if seite and typ in ohne and not (e.get("alt") or e.get("actual")):
            ohne[typ].add(seite)
    # 7.3-1 Figure ohne Alt-Text, 7.7-1 Formula ohne Alt-Text
    return {"7.3-1": sorted(ohne["Figure"]), "7.7-1": sorted(ohne["Formula"])}


def probleme_zusammenstellen(meta: dict, struktur: Optional[dict], ki_befunde: list,
                             _: Callable[[str], str] = _identitaet, quickinfos: Optional[dict] = None) -> list[dict]:
    """Eine Liste, nach Seite sortiert: {seite, seiten, art, quelle, text}."""
    out = []
    if EIGENE_PRUEFUNGEN and meta.get("vollstaendigkeit_geprueft") is False:
        # ehrlich sagen, dass ein Teil der Pruefung nicht lief — sonst sieht „keine fehlenden Zeilen“ wie ein Ergebnis aus
        out.append({"seite": 0, "seiten": [], "art": "hinweis", "quelle": _("Vollständigkeit"),
                    "text": _("Die Vollständigkeit konnte nicht geprüft werden. Bitte die Hörprobe selbst durchgehen.")})
    # Seiten fuer veraPDF-Befunde ohne Seitenangabe (Michael Karbe, Feedback 20261001 - 1, Punkt 10: „Die Anzeige der
    # Problemstellen wird nicht mehr angezeigt“): veraPDF nennt Seiten nur, wenn der Kontextpfad eine Seite enthaelt — bei
    # Strukturelementen (7.3-1 „Figure ohne Alt“) nie. Seit „nur veraPDF“ (28.09.) fehlten damit Seitenbild und Problemseite fuer
    # die haeufigsten Befunde. Die Seiten stehen im Strukturbaum derselben Pruefdatei.
    struktur_seiten = _struktur_seiten_je_regel(struktur)
    for p in ((meta.get("verapdf") or {}).get("punkte") or []):
        if p.get("status") != "befund":
            continue
        # je verletztem Pruefpunkt eine Zeile (pdfua_export._einzeln, Michael Karbe 24.09.2026, Punkt 12)
        for e in (p.get("einzeln") or [{"text": p.get("text") or "", "seiten": p.get("seiten") or []}]):
            seiten = [int(x) for x in (e.get("seiten") or []) if str(x).isdigit()]
            if not seiten:
                seiten = sorted({s for r in (e.get("regeln") or []) for s in struktur_seiten.get(r, [])})
            # Regelnummer von veraPDF dazu (Michael 28.09.2026: vergleichbar mit einer lokalen Pruefung). Seit 30.09.2026 ohne
            # das Wort „veraPDF“ in der Zeile (Michael Karbe, Feedback 202609230 - 1, Punkt 8: „oben weisen wir bereits auf
            # veraPDF hin“); die Regelnummer bleibt, die Ansicht nennt oben, dass sie von veraPDF stammt.
            regeln = e.get("regeln") or []
            ref = ((" " + (_("(Regel {r})") if len(regeln) == 1 else _("(Regeln {r})")).format(r=", ".join(regeln)))
                   if regeln else "")
            # teile: dieselbe Zeile in Stuecken, damit die Ansicht die Seiten nicht doppelt nennt (vorne „Seiten 1, 9, 10 –“) und
            # einen nicht uebersetzten englischen Satz mit lang="en" auszeichnet (Pruefung Barrierefreiheit 30.09.2026, Punkt 7)
            teile = ({"bereich": p.get("bereich", ""), "satz": e.get("satz") or "", "lang": e.get("lang") or "",
                      "mal": e.get("mal") or "", "ref": ref.strip()} if "satz" in e else None)
            out.append({"seite": (seiten[0] if seiten else 0), "seiten": seiten, "art": "technisch", "regeln": regeln,
                        "quelle": _("veraPDF (PDF/UA-1)"), "text": (f"{p.get('bereich', '')}: {e.get('text', '')}".strip(": ") + ref),
                        "teile": teile})
    if not EIGENE_PRUEFUNGEN:
        out.sort(key=lambda p: (p["seite"] or 10 ** 6, p["art"]))
        for i, p in enumerate(out, 1):
            p["nr"] = i
        return out
    if struktur:
        for p in struktur_probleme(struktur, _, quickinfos):
            p.update({"seiten": [p["seite"]] if p["seite"] else [], "quelle": _("Struktur")})
            out.append(p)
    for z in meta.get("fehlend") or []:
        out.append({"seite": z["seite"], "seiten": [z["seite"]], "art": "vollstaendigkeit", "quelle": _("Vollständigkeit"),
                    "text": _("Sichtbarer Text fehlt vermutlich in der Vorlesung: „{t}“.").format(t=_kurz(z.get("text") or "", 100))})
    for b in ki_befunde or []:
        if b.get("sicherheit") not in ("hoch", "mittel"):
            continue
        seite = int(b.get("seite") or 0)
        out.append({"seite": seite, "seiten": [seite] if seite else [], "art": "ki", "quelle": _("KI-basierte Prüfung"),
                    "text": _kurz(b.get("befund") or "", 200) + ((" " + _("(Sicherheit: {s})").format(s=_(b["sicherheit"]))) if b.get("sicherheit") else "")})
    out.sort(key=lambda p: (p["seite"] or 10 ** 6, p["art"]))
    for i, p in enumerate(out, 1):
        p["nr"] = i
    return out


def hoerprobe_seiten(zeilen: list[str], _: Callable[[str], str] = _identitaet, seiten_gesamt: int = 0) -> dict:
    """Die Hörprobe (pdf_struktur.hoerprobe) nach Seiten gegliedert: {kopf: [Sprache, Seiten, Zusammenfassung],
    seiten: [{seite, zeilen}]}. Zeilen vor der ersten Seitenmarke gehören zur Seite 1. Mit seiten_gesamt stehen ALLE
    Seiten 1..n in der Liste, auch die, auf denen ein Screenreader nichts vorliest (reine Bildseiten ohne Alt-Text —
    gerade die gehoeren in die Pruefung; Pruefbericht 24.09.2026)."""
    kopf = zeilen[:3]
    seiten: list = []
    aktuell = {"seite": 1, "zeilen": []}
    marke = re.compile(r"^—\s*\D*?(\d+)\s*—$")
    for z in zeilen[3:]:
        m = marke.match(z.strip())
        if m:
            if aktuell["zeilen"] or seiten:
                seiten.append(aktuell)
            aktuell = {"seite": int(m.group(1)), "zeilen": []}
            continue
        aktuell["zeilen"].append(z)
    if aktuell["zeilen"] or not seiten:
        seiten.append(aktuell)
    # gleiche Seitennummer zusammenlegen (Elemente, die wieder auf eine fruehere Seite springen)
    zusammen: dict = collections.OrderedDict()
    for s in seiten:
        zusammen.setdefault(s["seite"], []).extend(s["zeilen"])
    if seiten_gesamt:
        for n in range(1, int(seiten_gesamt) + 1):
            zusammen.setdefault(n, [])
        return {"kopf": kopf, "seiten": [{"seite": k, "zeilen": zusammen[k]} for k in sorted(zusammen)]}
    return {"kopf": kopf, "seiten": [{"seite": k, "zeilen": v} for k, v in zusammen.items() if v]}


def jetzt() -> str:
    return time.strftime("%d.%m.%Y %H:%M")
