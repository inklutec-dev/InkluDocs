"""Strukturlesung, Hoerprobe und Strukturansicht einer getaggten PDF (22.09.2026, Steve).

Grundlage ist das eigene PDFix-Skript pdfix_scripts/Struktur_Export.py: es laeuft den Tag-Baum in
Lesereihenfolge ab und liefert je Element Rolle, Tiefe, Seite, Text, Alt-Text, Sprache, bei Tabellen
Zeilen/Spalten, bei Formularfeldern Feldname und Quickinfo. Daraus entstehen:

  - die HOERPROBE: eine Liste in Lesereihenfolge, je Zeile das, was ein Screenreader aus den Tags
    bekommt („Überschrift Ebene 1: …“, „Liste mit 3 Einträgen“, „Tabelle, 4 Zeilen, 3 Spalten“,
    „Grafik: Alt-Text …“, „Formularfeld Vorname: Quickinfo …“) — dieselbe Form wie beim Word-Weg;
  - die STRUKTURANSICHT: dieselben Tags als semantisches HTML (h1–h6, p, ul/ol/li, table/th/td,
    figure mit Alt-Text, Felder als Beschreibungsliste), damit ein Screenreader-Nutzer die PDF im
    Tool wie eine Webseite navigiert (Ueberschriftensprünge) und Reihenfolge, Luecken, Rollen prueft;
  - die Eingabe fuer die automatische Pruefung (pdf_pruefung.py).

Verlaesslichkeit (ehrlich, Steve 22.09.): Das ist exakt der Inhalt der Tags — was fehlt oder falsch
ist, ist auch in Acrobat falsch. Eigenheiten einzelner Reader (Tabellen-Navigation, Artefakte) bildet
erst der echte Screenreader-Lauf ab (NVDA-Server, spaeter).

Cache: <arbeitsordner>/<pdfname>.struktur.json (im Upload-Ordner neben der Datei); gilt, solange die PDF nicht neuer ist.
"""
from __future__ import annotations

import html
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent / "pdfix_scripts"
_SCRIPT = _SCRIPT_DIR / "Struktur_Export.py"
_TIMEOUT_SECONDS = int(os.environ.get("PDFIX_STRUKTUR_TIMEOUT", "180"))
MAX_ZEILE = 400
STRUKTUR_VERSION = 2   # Cache-Version: 2 = mit Objektnummern (obj) je Element; aeltere Caches werden neu gelesen


class StrukturFehler(Exception):
    """Nutzertauglicher Grund."""


def _identitaet(s: str) -> str:
    return s


def verfuegbar() -> bool:
    if not _SCRIPT.is_file():
        return False
    try:
        import pdfixsdk  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# Lesen (Subprocess + Cache)
# ---------------------------------------------------------------------------

def lesen(pdf_pfad: str, arbeitsordner: str, erneuern: bool = False) -> dict:
    """Strukturlesung als dict {"info": {...}, "elemente": [...]}; gecacht in <arbeitsordner>/<pdfname>.struktur.json."""
    if not os.path.isfile(pdf_pfad):
        raise StrukturFehler("Die Datei fehlt")
    os.makedirs(arbeitsordner, exist_ok=True)
    cache = os.path.join(arbeitsordner, os.path.basename(pdf_pfad) + ".struktur.json")
    if not erneuern and os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(pdf_pfad):
        try:
            with open(cache, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and "elemente" in d and (d.get("info") or {}).get("version") == STRUKTUR_VERSION:
                return d
        except Exception:  # noqa: BLE001
            pass
    if not verfuegbar():
        raise StrukturFehler("Die Strukturlesung ist auf diesem Server nicht eingerichtet")
    # eindeutige Temp-Datei je Lauf (Pruefbericht 24.09.2026): zwei gleichzeitige Lesungen derselben PDF schrieben
    # sonst in dieselbe .tmp und ersetzten sich gegenseitig
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(pdf_pfad) + ".", suffix=".struktur.tmp", dir=arbeitsordner)
    os.close(fd)
    cmd = [sys.executable, str(_SCRIPT), "-i", pdf_pfad, "-o", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=str(_SCRIPT_DIR))
    except subprocess.TimeoutExpired:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise StrukturFehler("Die Strukturlesung hat zu lange gedauert")
    if r.returncode != 0 or not os.path.getsize(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    if r.returncode == 3:
        raise StrukturFehler("Die PDF hat keinen Strukturbaum (keine Tags)")
    if r.returncode != 0 or not os.path.isfile(tmp):
        log.warning("Strukturlesung fehlgeschlagen rc=%s stderr=%s", r.returncode, (r.stderr or "")[-300:])
        raise StrukturFehler("Die Struktur konnte nicht gelesen werden")
    os.replace(tmp, cache)
    with open(cache, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Hoerprobe
# ---------------------------------------------------------------------------

_ROLLEN_STILL = {"Document", "Part", "Art", "Sect", "Div", "NonStruct", "Private", "DocumentFragment", "Aside",
                 "TBody", "THead", "TFoot", "LBody", "Lbl", "Span", "Annot", "Ruby", "Warichu", "Reference", "BibEntry"}


def _kurz(text: str, n: int = MAX_ZEILE) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + " …"


def _kinder(elemente: list, idx: int) -> list:
    """Direkte Kinder des Elements an Position idx."""
    e = elemente[idx]
    praefix = e["id"] + "."
    tiefe = e["id"].count(".") + 1
    out = []
    for f in elemente[idx + 1:]:          # Nachfahren stehen in Tiefensuche direkt dahinter
        if not f["id"].startswith(praefix):
            break
        if f["id"].count(".") == tiefe:
            out.append(f)
    return out


def hoerprobe(struktur: dict, _: Callable[[str], str] = _identitaet, felder_quickinfos: Optional[dict] = None) -> list[str]:
    """Zeilen in Lesereihenfolge. felder_quickinfos: {Feldname: Quickinfo} aus der Datenbank (ergaenzt
    die /TU-Werte der Datei, falls die Datei noch ohne Quickinfos ist)."""
    elemente = struktur.get("elemente") or []
    info = struktur.get("info") or {}
    zeilen: list[str] = []
    zeilen.append(_("Sprache: {s}").format(s=info.get("lang")) if info.get("lang") else _("Sprache: nicht gesetzt"))
    zeilen.append(_("Seiten: {n}").format(n=info.get("seiten", 0)))
    letzte_seite = 0
    n_ueberschriften = n_bilder = n_bilder_ohne = n_tabellen = n_listen = n_felder = 0
    for idx, e in enumerate(elemente):
        typ = e.get("typ") or ""
        text = e.get("text") or ""
        seite = e.get("seite") or 0
        if seite and seite != letzte_seite and typ not in _ROLLEN_STILL:
            zeilen.append(_("— Seite {n} —").format(n=seite))
            letzte_seite = seite
        if typ in _ROLLEN_STILL:
            continue
        if typ.startswith("H") and typ[1:].isdigit():
            n_ueberschriften += 1
            zeilen.append(_("Überschrift Ebene {n}: {t}").format(n=typ[1:], t=_kurz(text) or _("(leer)")))
        elif typ == "H":
            n_ueberschriften += 1
            zeilen.append(_("Überschrift: {t}").format(t=_kurz(text) or _("(leer)")))
        elif typ == "P":
            zeilen.append(_("Absatz: {t}").format(t=_kurz(text)) if text else _("Absatz ohne Text"))
        elif typ == "L":
            n_listen += 1
            n = len([k for k in _kinder(elemente, idx) if k.get("typ") == "LI"])
            zeilen.append(_("Liste mit {n} Einträgen").format(n=n))
        elif typ == "LI":
            zeilen.append(_("Listenpunkt: {t}").format(t=_kurz(text)))
        elif typ == "Table":
            n_tabellen += 1
            zeilen.append(_("Tabelle mit {r} Zeilen und {c} Spalten").format(r=e.get("zeilen", "?"), c=e.get("spalten", "?")))
        elif typ == "TR":
            zellen = _kinder(elemente, idx)
            kopf = [k for k in zellen if k.get("typ") == "TH"]
            zeilen.append(_("Zeile: {t}").format(t=" | ".join(_kurz(k.get("text") or "", 80) for k in zellen)) if zellen else _("Leere Zeile"))
            if kopf and len(kopf) == len(zellen):
                zeilen[-1] = _("Kopfzeile: {t}").format(t=" | ".join(_kurz(k.get("text") or "", 80) for k in zellen))
        elif typ in ("TH", "TD"):
            continue   # in der Zeile enthalten
        elif typ in ("Figure", "Formula"):
            n_bilder += 1
            alt = e.get("alt") or ""
            if alt:
                zeilen.append(_("Grafik: {t}").format(t=_kurz(alt)) if typ == "Figure" else _("Formel: {t}").format(t=_kurz(alt)))
            elif e.get("actual"):
                zeilen.append(_("Grafik mit Ersatztext: {t}").format(t=_kurz(e["actual"])))
            else:
                n_bilder_ohne += 1
                zeilen.append(_("Grafik ohne Alt-Text") if typ == "Figure" else _("Formel ohne Alt-Text"))
        elif typ == "Form":
            n_felder += 1
            name = e.get("feldname") or ""
            qi = e.get("quickinfo") or ((felder_quickinfos or {}).get(name) if name else "") or ""
            if qi:
                zeilen.append(_("Formularfeld {name}: {t}").format(name=name or _("ohne Namen"), t=_kurz(qi)))
            else:
                zeilen.append(_("Formularfeld {name} ohne Quickinfo").format(name=name or _("ohne Namen")))
        elif typ == "Link":
            zeilen.append(_("Link: {t}").format(t=_kurz(text) or _("(ohne Text)")))
        elif typ == "Caption":
            zeilen.append(_("Bildunterschrift: {t}").format(t=_kurz(text)))
        elif typ == "Note":
            zeilen.append(_("Fußnote: {t}").format(t=_kurz(text)))
        elif typ in ("TOC", "TOCI"):
            zeilen.append(_("Inhaltsverzeichnis") if typ == "TOC" else _("Eintrag: {t}").format(t=_kurz(text)))
        elif typ == "BlockQuote":
            zeilen.append(_("Zitat: {t}").format(t=_kurz(text)))
        elif typ == "Code":
            zeilen.append(_("Code: {t}").format(t=_kurz(text)))
        else:
            zeilen.append(_("{typ}: {t}").format(typ=typ, t=_kurz(text)) if text else _("{typ} (ohne Text)").format(typ=typ))
    zusammen = _("Zusammenfassung: {u} Überschriften, {l} Listen, {t} Tabellen, {b} Grafiken ({o} ohne Alt-Text), {f} Formularfelder.").format(
        u=n_ueberschriften, l=n_listen, t=n_tabellen, b=n_bilder, o=n_bilder_ohne, f=n_felder)
    zeilen.insert(2, zusammen)
    return zeilen


# ---------------------------------------------------------------------------
# Strukturansicht (semantisches HTML, alles escaped)
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _nachfahren(elemente: list, idx: int) -> list:
    praefix = elemente[idx]["id"] + "."
    out = []
    for f in elemente[idx + 1:]:
        if not f["id"].startswith(praefix):
            break
        out.append(f)
    return out


def _objekt_html(e: dict, _: Callable[[str], str], felder_quickinfos: Optional[dict]) -> str:
    """Kurzform fuer ein Feld / eine Grafik innerhalb einer Tabellenzelle."""
    typ = e.get("typ") or ""
    if typ == "Form":
        name = e.get("feldname") or ""
        qi = e.get("quickinfo") or ((felder_quickinfos or {}).get(name) if name else "") or ""
        return f'<span class="struktur-rolle">{_esc(_("Formularfeld"))} {_esc(name) if name else _esc(_("ohne Namen"))}:</span> {_esc(qi) if qi else _esc(_("ohne Quickinfo"))}'
    if typ in ("Figure", "Formula"):
        alt = e.get("alt") or e.get("actual") or ""
        return f'<span class="struktur-rolle">{_esc(_("Grafik"))}:</span> {_esc(alt) if alt else _esc(_("ohne Alt-Text"))}'
    return ""


def html_ansicht(struktur: dict, _: Callable[[str], str] = _identitaet, felder_quickinfos: Optional[dict] = None,
                 ebene_versatz: int = 1) -> str:
    """Tag-Baum als semantisches HTML. ebene_versatz: H1 der PDF wird h(1+versatz), damit die Seite selbst
    eine H1 behaelt (WCAG: eine H1 je Seite). Container werden durchsichtig, Tabellen werden echte Tabellen."""
    elemente = struktur.get("elemente") or []
    teile: list[str] = []
    offen: list[tuple[str, str]] = []   # Stapel (id, schliess-tag)
    verbraucht: set = set()             # Felder/Grafiken, die schon in einer Zelle stehen

    def schliesse_bis(tiefe_id: str):
        while offen and not tiefe_id.startswith(offen[-1][0] + "."):
            teile.append(offen.pop()[1])

    letzte_seite = 0
    for idx, e in enumerate(elemente):
        typ = e.get("typ") or ""
        text = e.get("text") or ""
        eid = e.get("id") or ""
        if eid in verbraucht:
            continue
        schliesse_bis(eid)
        seite = e.get("seite") or 0
        if seite and seite != letzte_seite and typ not in _ROLLEN_STILL and typ not in ("TR", "TH", "TD", "LI"):
            teile.append(f'<p class="struktur-seite" aria-label="{_esc(_("Seite {n}").format(n=seite))}">{_esc(_("— Seite {n} —").format(n=seite))}</p>')
            letzte_seite = seite
        if typ.startswith("H") and typ[1:].isdigit():
            n = min(6, max(1, int(typ[1:]) + ebene_versatz))
            teile.append(f"<h{n}>{_esc(text) or _esc(_('(leer)'))}</h{n}>")
        elif typ == "H":
            teile.append(f"<h{min(6, 1 + ebene_versatz)}>{_esc(text)}</h{min(6, 1 + ebene_versatz)}>")
        elif typ == "P":
            teile.append(f"<p>{_esc(text)}</p>" if text else f'<p class="struktur-leer">{_esc(_("Absatz ohne Text"))}</p>')
        elif typ == "L":
            teile.append("<ul>")
            offen.append((eid, "</ul>"))
        elif typ == "LI":
            teile.append(f"<li>{_esc(text)}</li>")
        elif typ == "Table":
            teile.append('<table class="struktur-tabelle">')
            offen.append((eid, "</table>"))
        elif typ == "TR":
            zellen = _kinder(elemente, idx)
            zellen_html = []
            for k in zellen:
                kidx = elemente.index(k)
                inhalt = [_esc(k.get("text") or "")]
                for o in _nachfahren(elemente, kidx):
                    if o.get("typ") in ("Form", "Figure", "Formula"):
                        inhalt.append(_objekt_html(o, _, felder_quickinfos))
                        verbraucht.add(o.get("id") or "")
                innen = " ".join(x for x in inhalt if x)
                zellen_html.append(f'<th scope="col">{innen}</th>' if k.get("typ") == "TH" else f"<td>{innen}</td>")
            teile.append("<tr>" + "".join(zellen_html) + "</tr>")
        elif typ in ("TH", "TD"):
            continue
        elif typ in ("Figure", "Formula"):
            alt = e.get("alt") or e.get("actual") or ""
            teile.append(f'<figure><p><span class="struktur-rolle">{_esc(_("Grafik"))}:</span> {_esc(alt) if alt else _esc(_("ohne Alt-Text"))}</p></figure>')
        elif typ == "Form":
            name = e.get("feldname") or ""
            qi = e.get("quickinfo") or ((felder_quickinfos or {}).get(name) if name else "") or ""
            teile.append(f'<p><span class="struktur-rolle">{_esc(_("Formularfeld"))} {_esc(name) if name else _esc(_("ohne Namen"))}:</span> {_esc(qi) if qi else _esc(_("ohne Quickinfo"))}</p>')
        elif typ == "Link":
            teile.append(f'<p><span class="struktur-rolle">{_esc(_("Link"))}:</span> {_esc(text)}</p>')
        elif typ == "Caption":
            teile.append(f"<p><em>{_esc(text)}</em></p>")
        elif typ == "Note":
            teile.append(f'<p><span class="struktur-rolle">{_esc(_("Fußnote"))}:</span> {_esc(text)}</p>')
        elif typ == "BlockQuote":
            teile.append(f"<blockquote><p>{_esc(text)}</p></blockquote>")
        elif typ == "Code":
            teile.append(f"<pre>{_esc(text)}</pre>")
        elif typ in _ROLLEN_STILL or typ in ("TOC", "TOCI"):
            if typ == "TOCI" and text:
                teile.append(f"<p>{_esc(text)}</p>")
            continue
        else:
            teile.append(f'<p><span class="struktur-rolle">{_esc(typ)}:</span> {_esc(text)}</p>' if text else "")
    while offen:
        teile.append(offen.pop()[1])
    return "\n".join(t for t in teile if t)
