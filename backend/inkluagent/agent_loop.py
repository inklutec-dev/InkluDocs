"""Agentic Tool-Loop für InkluAgent (Sonnet 4.6 via Bedrock, 12.05.2026).

Aktiviert wenn INKLUAGENT_PROVIDER=bedrock und INKLUAGENT_AGENTIC nicht 'false'.

Der Loop:
  1. Baut messages aus storage-History + neuer user-Nachricht
  2. Ruft provider.invoke_with_tools(messages, tools, system)
  3. Inspiziert response.content auf tool_use-Blocks
     - Wenn nur text: fertig, return
     - Wenn tool_use: ToolExecutor ausführen, tool_result anfügen, goto 2
  4. Max _MAX_ITERATIONS Tool-Roundtrips pro User-Turn

Spezialfall view_image: Tool-Result kommt zurück mit zusätzlichem
`image_bytes`-Key. Wir hängen das Bild als image-content-Block direkt
an die nächste user-Nachricht an, damit Claude es in seinem nächsten
Schritt sieht.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Callable, Optional

from . import storage
from .providers.base import FristErreicht
from .providers.bedrock import BedrockProvider, BedrockProviderError
from .tools.definitions import TOOL_DEFINITIONS, ToolExecutor
from .tools.definitions_formular import TOOL_DEFINITIONS_FORMULAR, ToolExecutorFormular
from .tools.definitions_pdf import TOOL_DEFINITIONS_PDF   # PDF-Projekte (22.09.2026)
from .tools.definitions_oberflaeche import (TOOL_DEFINITIONS_OBERFLAECHE_PDF, TOOL_DEFINITIONS_OBERFLAECHE_WORD,
                                            TOOL_DEFINITIONS_OBERFLAECHE_FORMULAR,
                                            TOOL_DEFINITIONS_OBERFLAECHE_GRAFIK)   # alles, was die Oberflaeche kann (30.09.2026)
from .prompts import schubfaecher   # Systemprompt aus Schubfaechern (15.10.2026): Projektart, Faehigkeiten, Demo
import funktionen   # Funktionsschalter: ausgeblendet in der Oberflaeche = ausgeblendet im Chatbot (30.09.2026)
from .tools.definitions import TOOL_DEFINITIONS_WORD
from .adapters.inkludocs import build_project_summary
from .sanitize import sanitize_markdown
from .daten import DATEN_MARKE, KONTEXT_KOPF, ohne_marke   # Fremdtext ist keine Anweisung (09.10.2026)
from . import sicherheit   # Sicherheitsfundament (Ausbau Runde 1, Schritt 2; Deckel und Ja-Pruefung immer, Grenzen am Schalter)

log = logging.getLogger(__name__)

_MAX_ITERATIONS = 6
_HISTORY_TURNS = 20  # weniger als 30, weil Tool-Calls + Images Context-Hungry sind
_DEFAULT_MAX_TOKENS = 4096
_MAX_TOOL_RESULT_CHARS = 40000  # Werkzeug-Ergebnis im tool_result (28.08.2026, vorher 8000)


def _frist_s() -> float:
    """Gesamtfrist je Nachricht ueber alle Runden (Pruefung 13.10.2026, S4): ohne sie konnten sechs Runden mit Werkzeugen und
    Wiederholungen eine halbe Stunde dauern. Sie gilt auch innerhalb einer Runde (Runde 2, K-N3/S-N5): nach ihr beginnt kein
    Werkzeug mehr, und der Anbieter haelt sie ein, wenn er sie kennt (providers.base.Uhr: ein Schritt ueberzieht hoechstens um
    eine Minute). Ein Lauf endet damit spaetestens nach Frist + einem Modellaufruf + einem Werkzeug (kern.warten: 180 s) —
    weit unter dem Netz gegen haengengebliebene Laeufe in main (_AGENT_LAEUFT_NETZ_S)."""
    try:
        return max(30.0, min(float(os.environ.get("INKLUAGENT_FRIST_S") or 300), 540.0))
    except ValueError:
        return 300.0


def _args_kurz(args: Any) -> dict:
    """Werkzeug-Argumente fuer das Log: Kennungen und Zahlen ja, Texte (Alt-Texte, Namen, Suchbegriffe) nur mit ihrer Laenge
    (Pruefung 13.10.2026, S10 — personenbezogene Inhalte gehoeren nicht ins INFO-Log)."""
    if not isinstance(args, dict):
        return {}
    return {k: (v if isinstance(v, (bool, int, float)) or v is None else f"<{type(v).__name__} {len(str(v))}>")
            for k, v in args.items()}


def _detect_media_type(img_bytes: bytes) -> str:
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


# ─── Werkzeugsatz je Projektart (28.08.2026) ───
# Ein Formular-Projekt (Quickinfo-Werkzeug) hat Felder statt Bilder: eigener
# System-Prompt, eigener Werkzeugsatz, eigene Projekt-Zusammenfassung. Die
# Bild-Werkzeuge existieren fuer den Formular-Agenten nicht (und umgekehrt),
# damit sich die Werkzeuge nie vermischen. Neue Werkzeuge = neuer Eintrag hier.
def _ist_formular(project: dict) -> bool:
    return (project or {}).get("tool") == "formular"


def _werkzeugsatz(project: dict, project_id: int, user_id: int, auftraege: bool = False):
    """(tool_definitions, executor, system_prompt) fuer dieses Projekt. Seit 30.09.2026 (Steve: der Chatbot kann alles, was
    die Oberflaeche kann, und nichts, was sie nicht anbietet): Werkzeuge hinter einem ausgeschalteten Schalter
    (funktionen.py) fehlen hier, der Executor fuehrt sie nicht aus, und die Systemprompts nennen sie nicht.
    Der Text kommt aus den Schubfaechern (prompts/schubfaecher.py) — hier der Teil des Projekts ohne die Faecher des Gespraechs
    (Begleiter, Karten, Rueckmeldung, Auftraege; die setzt run_agent dazu). auftraege: gibt es in diesem Gespraech Auftraege,
    sagt der PDF-Prompt „mehrere Schritte = ein Auftrag“."""
    defs, executor = _werkzeugsatz_roh(project, project_id, user_id)
    namen = set()
    eindeutig = []
    for d in funktionen.werkzeuge_filtern(defs):
        if d["name"] not in namen:     # ein Werkzeug nur einmal (Word + Oberflaeche teilen sich Namen)
            namen.add(d["name"])
            eindeutig.append(d)
    # Schritt 2 (seit Runde 2 der Pruefung 13.10.2026 immer): Einzelaktionen nehmen bestaetigt an, der Prompt nennt die Server-Regel
    eindeutig = sicherheit.werkzeuge_anpassen(eindeutig)
    lage = schubfaecher.Lage(schubfaecher.projektart(project), int(user_id or 0), auftraege=auftraege)
    return eindeutig, executor, schubfaecher.zusammensetzen(lage)


def _werkzeugsatz_roh(project: dict, project_id: int, user_id: int):
    """(tool_definitions, executor) nach Projektart — dieselbe Weiche wie der Text (schubfaecher.projektart)."""
    art = schubfaecher.projektart(project)
    if art == "formular":   # alte Formular-Projekte (tool formular) behalten ihren Satz
        return (TOOL_DEFINITIONS_FORMULAR + TOOL_DEFINITIONS_OBERFLAECHE_FORMULAR,
                ToolExecutorFormular(project_id=project_id, user_id=user_id))
    # Word-Projekte (11.09.2026, Meine Ausgaben Schritt 2): Bild-Werkzeuge + Pruefen/Umwandeln/Word-Export/Regal.
    if art == "word":
        doku = [d for d in TOOL_DEFINITIONS_PDF if d["name"] in ("dokument_umbenennen", "dokument_loeschen", "alt_sprache_setzen")]
        return (TOOL_DEFINITIONS + TOOL_DEFINITIONS_WORD + doku + TOOL_DEFINITIONS_OBERFLAECHE_WORD,
                ToolExecutor(project_id=project_id, user_id=user_id, word=True))
    # PDF-Projekte (Werkzeugsatz nach Dateiart, 22.09.2026, Michael 18.09. + Steve): EIN Satz fuer alle Stationen —
    # Bild-Werkzeuge + Feld-Werkzeuge (Quickinfos) + PDF-Werkzeuge + Werkzeuge der Oberflaeche.
    if art == "pdf":
        bekannt = {d["name"] for d in TOOL_DEFINITIONS}
        felder = [d for d in TOOL_DEFINITIONS_FORMULAR if d["name"] not in bekannt]
        ablage = [d for d in TOOL_DEFINITIONS_WORD if d["name"] in ("liste_ausgaben", "lies_ausgabe")]
        return (TOOL_DEFINITIONS + felder + TOOL_DEFINITIONS_PDF + ablage + TOOL_DEFINITIONS_OBERFLAECHE_PDF,
                ToolExecutor(project_id=project_id, user_id=user_id, pdf=True))
    # Grafik- und Webseiten-Projekte (InkluAgent-Ausbau Runde 1, Schritt 4, Schalter AGENT_BILD_WERKZEUGE): alles, was die
    # Oberflaeche dort anbietet. Schalter aus = Projektart alt_text, die sechs Bild-Werkzeuge wie bisher.
    if art in ("grafik", "web"):
        doku = [d for d in TOOL_DEFINITIONS_PDF if d["name"] == "alt_sprache_setzen"
                or (art == "web" and d["name"] in ("dokument_umbenennen", "dokument_loeschen"))]
        return (TOOL_DEFINITIONS + TOOL_DEFINITIONS_OBERFLAECHE_GRAFIK + doku,
                ToolExecutor(project_id=project_id, user_id=user_id, bild=art))
    return TOOL_DEFINITIONS, ToolExecutor(project_id=project_id, user_id=user_id)


def _formular_summary(project_id: int, user_id: int) -> str:
    """Kompakter Projekt-Kontext fuer Formular-Projekte (Gegenstueck zu build_project_summary)."""
    try:
        from .tools.formular import list_form_fields
        r = list_form_fields(project_id, user_id)
    except Exception:
        return "Formular-Projekt (Quickinfo-Werkzeug). Hol den Stand mit list_form_fields."
    if not r.get("ok"):
        return "Formular-Projekt (Quickinfo-Werkzeug). Hol den Stand mit list_form_fields."
    res = r["result"]
    p = res["project"]
    return (f"Formular-Projekt „{p['name']}“ (Quickinfo-Werkzeug), Status {p['status']}, Sprache der Quickinfos "
            f"{p['sprache_der_quickinfos']}: {res['feld_count']} Felder in {len(res['documents'])} Dokument(en), "
            f"{res['offen']} ohne Quickinfo, {res['beschrieben']} beschrieben; {res['stammdaten_eintraege']} Stammdaten-Eintraege "
            "im Konto. Details je Feld: list_form_fields / get_field_details.")


def projekt_marke(name: str, geloescht: bool = False) -> str:
    """Marke im Gedaechtnis des Agenten, wenn das Gespraech des Kontos das Projekt wechselt (Ausbau Runde 2, Schritt 2). Der
    Name stammt vom Nutzer oder aus einer Datei: als Daten gekennzeichnet."""
    return f"[Projektwechsel: ab hier geht es um Projekt {DATEN_MARKE}„{name}“{' (inzwischen gelöscht)' if geloescht else ''}]"


# Nachrichten von Seiten ohne Projekt (Ausbau Runde 2, Schritt 3, inkluagent/konto.py)
OHNE_PROJEKT_MARKE = "[Ab hier: Seite ohne Projekt]"


def _mit_marken(history: list[dict]) -> list[dict]:
    """Gespraech des Kontos: vor die erste Nachricht jedes Projektabschnitts die Marke setzen, und vor die neue Nachricht,
    wenn das aktuelle Projekt ein anderes ist als zuletzt. Rueckgabe ohne Marke fuer die neue Nachricht, die haengt der Aufrufer an."""
    out: list[dict] = []
    vorher = object()
    for h in history:
        pid = h.get("project_id")
        content = h.get("content", "")
        if pid != vorher:
            marke = projekt_marke(h.get("projekt_name") or "", h.get("projekt_geloescht")) if pid is not None else OHNE_PROJEKT_MARKE
            content = marke + "\n" + content
            vorher = pid
        out.append(dict(h, content=content))
    return out


def _normalisieren(messages: list[dict]) -> list[dict]:
    """Abwechselnd Nutzer und Agent, beginnend mit dem Nutzer: aufeinanderfolgende Nachrichten derselben Rolle zusammenfassen
    (eine Antwort, die nicht gespeichert wurde, hinterlaesst zwei Nutzer-Nachrichten hintereinander)."""
    out: list[dict] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"] and isinstance(out[-1]["content"], str) and isinstance(m["content"], str):
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


def _build_initial_messages(
    project_id: int,
    user_message: str,
    project: dict,
    user_id: int = 0,
    konto: bool = False,
    ohne_nachricht: Optional[int] = None,
    gespraech_id: Optional[int] = None,
) -> list[dict]:
    """Baut die Anthropic-Format messages-Liste aus storage-History + user_message.

    History-Format aus storage: [{"role": "user|assistant", "content": "..."}].
    Tool-Calls werden NICHT in History gespeichert (vereinfacht den Reload).
    Die JUENGSTEN _HISTORY_TURNS Nachrichten (bis 12.10.2026 lieferte get_history mit LIMIT die aeltesten — in langen
    Gespraechen sah der Agent den Anfang statt des aktuellen Stands).
    konto (Ausbau Runde 2, Schritt 2, Schalter AGENT_KONTO): das Gespraech des Kontos ueber alle Projekte, mit Projektmarken.
    ohne_nachricht: Kennung der Nachricht dieser Runde, die der Endpunkt schon gespeichert hat — sie kommt als user_message
    (fuer das Modell, z. B. mit der Notiz des Hochladefelds) und nicht noch einmal aus dem Verlauf (bis 12.10.2026 doppelt).
    """
    # gespraech_id (Pruefung 13.10.2026, Runde 2, K-N4): das Gedaechtnis aus dem Gespraech der Frage — wie die Antwort (K3)
    history = (storage.konto_verlauf(user_id, limit=_HISTORY_TURNS + 1, gespraech_id=gespraech_id) if konto
               else storage.get_recent(project_id, limit=_HISTORY_TURNS + 1)) or []
    history = _gedaechtnis([h for h in history if not (ohne_nachricht and h.get("id") == ohne_nachricht)][-_HISTORY_TURNS:])
    # Das Gedaechtnis beginnt mit einer Nutzer-Nachricht (abgeschnittener Anfang)
    while history and history[0].get("role") != "user":
        history = history[1:]
    if konto:
        letztes = history[-1].get("project_id") if history else None
        history = _mit_marken(history)
        if letztes != project_id:
            name = (project or {}).get("name") or (project or {}).get("filename") or ""
            user_message = projekt_marke(name) + "\n" + user_message
    messages: list[dict] = []

    # Projekt-Summary als erste user-Nachricht (Context-Vorgabe). Sie steht in der Rolle des Nutzers, enthaelt aber Namen und
    # Alt-Texte aus den Dateien: die Kopfzeile sagt, dass das Daten sind, keine Anweisungen (09.10.2026, daten.py).
    summary = _formular_summary(project_id, user_id) if _ist_formular(project) else build_project_summary(project)
    messages.append({
        "role": "user",
        "content": f"{KONTEXT_KOPF}\n{summary}",
    })
    messages.append({
        "role": "assistant",
        "content": "Verstanden, Projekt-Kontext angenommen. Was kann ich für dich tun?",
    })

    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return _normalisieren(messages)


def _gedaechtnis(history: list[dict]) -> list[dict]:
    """Gespeicherte Nachrichten so, wie das Modell sie liest: getippter Text, Zeilen des Servers als Daten (storage.fuer_modell,
    Pruefung 13.10.2026, S3)."""
    return [dict(h, content=storage.fuer_modell(h)) for h in history]


def _append_tool_results(
    messages: list[dict],
    assistant_content_blocks: list[dict],
    tool_results: list[dict],
    pending_image_bytes: list[bytes],
) -> None:
    """Anthropic-Pattern: assistant-Turn mit tool_use Blocks UND
    nächster user-Turn mit tool_result Blocks (+ optional image-Blocks).
    """
    messages.append({"role": "assistant", "content": assistant_content_blocks})

    next_user_content: list[dict] = list(tool_results)  # tool_result blocks zuerst
    for img_bytes in pending_image_bytes:
        next_user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _detect_media_type(img_bytes),
                "data": base64.b64encode(img_bytes).decode("ascii"),
            },
        })
    if not next_user_content:
        return
    # Ein leerer text-block am Ende ist optional, hilft aber bei manchen Klienten
    messages.append({"role": "user", "content": next_user_content})


def run_agent(
    project_id: int,
    user_id: int,
    user_message: str,
    project: dict,
    provider: BedrockProvider,
    system_suffix: str = None,
    on_tool=None,
    nur_mit_karte: bool = False,
    konto: bool = False,
    ohne_nachricht: Optional[int] = None,
    uebersetzen: Optional[Callable[[str], str]] = None,
    nutzer_text: Optional[str] = None,
    begleiter_lauf=None,
    gespraech_id: Optional[int] = None,
) -> dict[str, Any]:
    """Führt den agentic Loop aus. Returns:
        {"reply": str, "intent": "agentic", "image_refs": [...] or None,
         "actions": [{"tool": "...", "args": {...}, "ok": bool}]}.

    Hinweis zur Persistenz: User-Message + Bot-Reply werden NICHT hier
    gespeichert — das übernimmt main.py:_chat_endpoint vor/nach diesem
    Aufruf zentral.

    system_suffix: optionaler Text, der NUR im Demo-Modus an den System-Prompt
    angehängt wird (thematische Leitplanke). Default None = normaler Pfad,
    System-Prompt unverändert.
    on_tool (28.08.2026, Werkzeug-Transparenz): Callback (name, args) VOR jedem
    Werkzeugaufruf — der Chat-Endpunkt streamt daraus die Live-Zeile „Ruft gerade
    auf: …“ an die Oberflaeche. Ergebnis enthaelt zusaetzlich "werkzeuge": Namen
    in Aufrufreihenfolge (leer = ohne Werkzeug aus dem Verlauf geantwortet).
    Livestream (Arbeitsfenster, 12.10.2026): on_tool darf zusaetzlich Attribute tragen — .fertig(name, args, result,
    neue_aktionen) nach jedem Werkzeug, .text(stueck) fuer die Antwort Wort fuer Wort (None = bisherigen Entwurf verwerfen; nur
    wenn der Provider kann_streamen), .zwischen() wenn gestreamter Text nur ein Zwischenstand vor Werkzeugen war. Fehlen sie,
    bleibt alles wie bisher.
    nutzer_text (Pruefung 13.10.2026, K1/S13): was der Nutzer getippt hat — die Ja-Pruefung liest DAS, nicht user_message mit den
    Notizen des Servers (Seite, Hochladen, Webadressen). None = user_message (Demo, Tests).
    begleiter_lauf (K9): der Lauf des Chat-Endpunkts im Konto (begleiter.Lauf). Nur damit gibt es Begleiter- und Karten-Werkzeuge
    (je nach ihrem Schalter) — nie in der Demo. uebersetzen: Uebersetzung der festen Saetze an den Nutzer (Sprache der Oberflaeche).
    """
    _t = uebersetzen or (lambda text: text)
    # Welche Zusaetze gelten, steht VOR dem Werkzeugsatz fest: mit Auftraegen sagt der PDF-Prompt „mehrere Schritte = ein
    # Auftrag“ statt „Schritt fuer Schritt“ (Michael 15.10.2026: sonst bot der Agent nur das Tagging an und blieb danach stehen)
    from . import auftraege as _auftraege, begleiter as _begleiter, karten as _karten, rueckmeldung as _rueckmeldung
    begleiter_an = begleiter_lauf is not None and _begleiter.an()
    karten_an = begleiter_lauf is not None and _karten.an()
    rueckmeldung_an = begleiter_lauf is not None and gespraech_id is not None and _rueckmeldung.an()
    auftraege_an = rueckmeldung_an and karten_an and _auftraege.an()
    # Den Systemprompt setzt EIN Zusammensetzer aus den Schubfaechern zusammen (prompts/schubfaecher.py, 15.10.2026); hier
    # kommen nur noch die Werkzeuge der Faecher dazu. Den Text, den die Werkzeugsaetze mitliefern, braucht es hier nicht.
    lage = schubfaecher.Lage(schubfaecher.projektart(project), int(user_id or 0), begleiter=begleiter_an, karten=karten_an,
                             rueckmeldung=rueckmeldung_an, auftraege=auftraege_an, demo_leitplanke=system_suffix or None)
    if project is None:
        # Runde 2, Schritt 3: Seite ohne Projekt — Konto-Werkzeuge, Konto-Kontext, Gespraech des Kontos (inkluagent/konto.py)
        from . import konto as _konto
        tool_defs, executor, _projekt_text = _konto.werkzeugsatz(user_id, uebersetzen)
    else:
        tool_defs, executor, _projekt_text = _werkzeugsatz(project, project_id, user_id, auftraege=auftraege_an)
    # Begleiter durch InkluDocs (Steve 12.10.2026, Schalter AGENT_BEGLEITER): Hinfuehren und harmlose Einstellungen — in jedem
    # Gespraech, mit und ohne Projekt; ausgefuehrt von inkluagent/begleiter.py, nicht vom Executor des Projekts
    if begleiter_an:
        tool_defs = list(tool_defs) + _begleiter.werkzeuge(user_id)
    # Formular-Karten (Paket 1, Schalter AGENT_KARTEN): wie der Begleiter nur aus dem Chat-Endpunkt des Kontos (begleiter_lauf
    # gesetzt), nie in der Demo — von dort liessen sich sonst Einladungen an beliebige Adressen vorbereiten
    if karten_an:
        tool_defs = list(tool_defs) + _karten.werkzeuge(project_id)   # mit dem Kaufen nur, wenn es das gibt (Baustein 3)
    # Rueckmeldung (Steve 14.10.2026, Schalter AGENT_RUECKMELDUNG): ebenfalls nur aus dem Chat-Endpunkt des Kontos und nur mit
    # Gespraech — startet ein Werkzeug einen Lauf im Hintergrund, meldet sich der Agent in diesem Gespraech, wenn er zu Ende ist
    rueckmeldung_kontext: dict = {}
    if rueckmeldung_an:
        tool_defs = _rueckmeldung.werkzeuge_anpassen(list(tool_defs), project_id)
        rueckmeldung_kontext = {"user_id": user_id, "project_id": project_id, "gespraech_id": gespraech_id,
                                "sprache": begleiter_lauf.ui_sprache, "auftrag": begleiter_lauf.nutzer_text}
    # Auftraege, Baustein 2 (14.10.2026, Schalter AGENT_AUFTRAEGE): mehrere Schritte als EIN Auftrag mit Gesamtpreis — die Karte
    # bringen die Formular-Karten, das Melden am Ende die Rueckmeldung; also nur, wenn beide hier gelten (auftraege_an oben)
    if auftraege_an:
        tool_defs = list(tool_defs) + _auftraege.werkzeuge(project_id)
    # Ja-Pruefung auf dem Server (Schritt 2): die Freigabe liest die Nachricht des Nutzers dieser Runde
    sicherheit.nachricht_merken(executor.turn_id, user_message if nutzer_text is None else nutzer_text, user_id)
    # Runde 2, Schritt 1: nach dem Hochladen in dieser Nachricht nichts Kostenpflichtiges ohne Karte
    executor.nur_mit_karte = bool(nur_mit_karte)
    if project is None:
        messages = _konto.nachrichten(user_id, user_message, ohne_nachricht=ohne_nachricht, verlauf_laenge=_HISTORY_TURNS,
                                      gespraech_id=gespraech_id)
    else:
        messages = _build_initial_messages(project_id, user_message, project, user_id, konto=konto, ohne_nachricht=ohne_nachricht,
                                           gespraech_id=gespraech_id)
    system_text = schubfaecher.zusammensetzen(lage)

    actions_log: list[dict] = []
    werkzeuge: list[str] = []
    anhaenge: list[dict] = []   # Download-Knoepfe unter der Antwort (Meine Ausgaben, 11.09.2026)
    image_refs_seen: set[int] = set()
    last_reply_text = ""
    melder_fertig = getattr(on_tool, "fertig", None) if on_tool else None
    melder_text = getattr(on_tool, "text", None) if on_tool else None
    strom = {"on_text": melder_text} if melder_text and getattr(provider, "kann_streamen", False) else {}

    ende = time.monotonic() + _frist_s()
    frist = {"frist": ende} if getattr(provider, "kennt_frist", False) else {}
    frist_um = False
    for iteration in range(_MAX_ITERATIONS):
        if iteration and time.monotonic() > ende:
            frist_um = True
            break
        try:
            payload = provider.invoke_with_tools(
                anthropic_messages=messages,
                tools=tool_defs,
                system=system_text,
                max_tokens=_DEFAULT_MAX_TOKENS,
                temperature=0.3,
                **strom,
                **frist,
            )
        except FristErreicht:
            frist_um = True
            break
        except BedrockProviderError as e:
            # Einzelheiten (Ressourcenpfad, Antwort des Anbieters) nur ins Log, nicht in den Chat (S10). Eine vom Anbieter gesperrte
            # Anfrage ist keine Stoerung: dann ehrlich sagen, dass die KI darauf nicht antwortet (Runde 2, K-N7)
            log.exception("KI-Aufruf im agentic Loop fehlgeschlagen")
            return {
                "reply": (_t("Auf diese Anfrage antwortet die KI nicht. Formuliere sie bitte anders.") if getattr(e, "gesperrt", False)
                          else _t("Entschuldigung, die Verbindung zur KI ist gerade unterbrochen. Bitte versuch es gleich noch einmal.")),
                "intent": "error",
                "image_refs": None,
                "actions": actions_log,
            }

        content_blocks = payload.get("content", []) or []
        stop_reason = payload.get("stop_reason", "")

        # Text-Anteile sammeln
        text_chunks = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        last_reply_text = "".join(text_chunks).strip()

        # Tool-Use-Blocks sammeln
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_use_blocks:
            # Reines Text-Response, Loop fertig
            log.info("Agent fertig nach %d Iterationen, stop=%s", iteration, stop_reason)
            break

        # Gestreamter Text vor Werkzeugen war nur ein Zwischenstand (die Oberflaeche nimmt ihn aus der Antwort)
        if strom and getattr(on_tool, "zwischen", None):
            try:
                on_tool.zwischen()
            except Exception:
                log.exception("zwischen-Callback fehlgeschlagen (ignoriert)")

        # Tool-Calls ausführen
        tool_results: list[dict] = []
        pending_images: list[bytes] = []
        for tu in tool_use_blocks:
            if time.monotonic() > ende:   # nach der Frist beginnt kein Werkzeug mehr, auch nicht in dieser Runde
                frist_um = True
                break
            name = tu.get("name", "")
            args = tu.get("input", {}) or {}
            tu_id = tu.get("id", "")

            log.info("Agent ruft Tool %s mit args=%s", name, _args_kurz(args))
            werkzeuge.append(name)
            if on_tool:
                try:
                    on_tool(name, args)
                except Exception:
                    log.exception("on_tool-Callback fehlgeschlagen (ignoriert)")
            if begleiter_an and name in _begleiter.NAMEN:
                result = _begleiter.ausfuehren(name, args, user_id, begleiter_lauf)
            elif karten_an and name in _karten.NAMEN:
                result = _karten.ausfuehren(name, args, user_id, project_id, _t)
            elif rueckmeldung_an and name in _rueckmeldung.NAMEN:
                result = _rueckmeldung.ausfuehren(name, args, **rueckmeldung_kontext)
            elif auftraege_an and name in _auftraege.NAMEN:
                result = _auftraege.ausfuehren(name, args, _t=_t, **rueckmeldung_kontext)
            else:
                result = executor.execute(name, args)
                if rueckmeldung_an:   # Lauf im Hintergrund gestartet: vormerken, das Ergebnis sagt dem Modell, dass er sich meldet
                    _rueckmeldung.nach_werkzeug(name, args, result, **rueckmeldung_kontext)
            vorher = len(actions_log)
            actions_log.append({
                "tool": name,
                "args": args,
                "ok": bool(result.get("ok")),
                "error": result.get("error", "") if not result.get("ok") else "",
            })

            # Refresh-Marker fuer Frontend: nach erfolgreichem DB-Update den
            # frischen Anzeige-Text mitliefern, damit das Textarea live updaten
            # kann ohne Liste-Reload. Fallback-Kette gleich wie main._display_alt_text.
            if name in ("update_alt_text", "revert_alt_text") and result.get("ok"):
                img_id_raw = args.get("image_id") if isinstance(args, dict) else None
                if img_id_raw is not None:
                    try:
                        img_id_int = int(img_id_raw)
                        from database import get_db   # dieselbe Datenbank wie ueberall (vorher fest /app/data/…)
                        _conn = get_db()
                        try:
                            _row = _conn.execute(
                                "SELECT alt_text_edited, alt_text, original_alt, langbeschreibung "
                                "FROM images WHERE id = ?",
                                (img_id_int,),
                            ).fetchone()
                        finally:
                            _conn.close()
                        if _row is not None:
                            actions_log.append({
                                "type": "refresh_image",
                                "image_id": img_id_int,
                                "alt_text": _row["alt_text_edited"] or _row["alt_text"] or _row["original_alt"] or "",
                                "langbeschreibung": _row["langbeschreibung"] or "",
                            })
                    except Exception:
                        log.exception("refresh_image-Action nicht angefuegt (img=%s)", img_id_raw)

            # Formular (28.08.2026): nach Aenderung einer Quickinfo den frischen Stand
            # mitliefern, damit formular.js Textfeld + Badge live setzt (refresh_feld).
            if name in ("update_quickinfo", "revert_quickinfo", "generate_quickinfo") and result.get("ok"):
                r = result.get("result") or {}
                actions_log.append({
                    "type": "refresh_feld", "feld_id": r.get("feld_id"), "quickinfo": r.get("quickinfo", ""),
                    "quelle": r.get("quelle", ""), "sicherheit": r.get("sicherheit", ""), "beleg": r.get("beleg", ""),
                    "ki_hinweise": r.get("hinweise", []),
                    "status": "beschrieben" if (r.get("quickinfo") or "").strip() else "offen",
                    "quickinfo_ki": r.get("ki_vorschlag") or r.get("quickinfo", ""),
                    "uebernommen": r.get("uebernommen", True),
                })

            # Livestream: Ergebnis dieses Werkzeugs (ok, Fehler, neuer Alt-Text/Quickinfo) an die Oberflaeche
            if melder_fertig:
                try:
                    melder_fertig(name, args, result, actions_log[vorher:])
                except Exception:
                    log.exception("fertig-Callback fehlgeschlagen (ignoriert)")

            # Pruefung 4 (M2/M3): Zustand verbrauchter Karten und „Ansicht nachziehen“ gehen an die Oberflaeche, nicht ans Modell
            if isinstance(result, dict):
                for k in (result.pop("karten", None) or []):
                    actions_log.append(dict({"type": "karte"}, **k))
                if result.pop("aktualisieren", False):
                    actions_log.append({"type": "ansicht_aktualisieren"})
                # Konto-Werkzeuge (12.10.2026): weitere Aktionen fuer die Oberflaeche, z. B. {"type": "seite_wechseln"}
                weitere = result.pop("aktionen", None) or []
                if result.get("ok"):
                    actions_log.extend(a for a in weitere if isinstance(a, dict))

            # anhang-Sonderfall (konvertiere_zu_pdfua / exportiere_word, 11.09.2026): Download-Knopf fuer die
            # Oberflaeche — aus dem tool_result nehmen (das Modell braucht ihn nicht), an die Antwort haengen.
            anhang = result.pop("anhang", None) if isinstance(result, dict) else None
            if anhang and result.get("ok"):
                # mehrere Links aus einem Werkzeug (fruehere_gespraeche, 12.10.2026) kommen als Liste
                for a in (anhang if isinstance(anhang, list) else [anhang]):
                    anhaenge.append(a)
                    actions_log.append(dict({"type": "anhang"}, **a))

            # image_bytes-Sonderfall (view_image / view_field)
            img_bytes = result.pop("image_bytes", None) if isinstance(result, dict) else None
            if img_bytes:
                pending_images.append(img_bytes)
                if isinstance(args, dict) and "image_id" in args:
                    image_refs_seen.add(int(args["image_id"]))

            # Andere referenzierte image_ids tracken
            if isinstance(args, dict) and "image_id" in args:
                try:
                    image_refs_seen.add(int(args["image_id"]))
                except (TypeError, ValueError):
                    pass

            # Serialisiere result als kompaktes JSON für tool_result.
            # 28.08.2026: Kappe 8000 -> 40000 Zeichen — bei 26 Formularfeldern war die
            # Liste abgeschnitten, der Agent pruefte nur die Haelfte (und sagte es).
            try:
                content_text = json.dumps(result, ensure_ascii=False)[:_MAX_TOOL_RESULT_CHARS]
            except (TypeError, ValueError):
                content_text = str(result)[:_MAX_TOOL_RESULT_CHARS]
            if len(content_text) >= _MAX_TOOL_RESULT_CHARS:
                content_text += ' …[Ergebnis gekuerzt — Werkzeug mit engerer Auswahl erneut aufrufen]'
                log.warning("Tool-Result %s auf %d Zeichen gekappt", name, _MAX_TOOL_RESULT_CHARS)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu_id,
                "content": content_text,
            })

        if frist_um:
            break
        _append_tool_results(messages, content_blocks, tool_results, pending_images)

        # Wenn stop_reason explizit "end_turn" trotz Tool-Use: dann ist Claude
        # eigentlich fertig, sollte nicht passieren — Schutz gegen Endlos-Loop:
        if stop_reason == "end_turn":
            log.warning("stop_reason=end_turn aber tool_use vorhanden — breche ab")
            break

    else:
        # Loop voll durchgelaufen ohne textuelle Antwort
        log.warning("Agent-Loop max %d Iterationen ohne text-Antwort", _MAX_ITERATIONS)
        if not last_reply_text:
            last_reply_text = _t("Ich habe mehrere Werkzeuge benutzt, komme aber zu keinem klaren Ergebnis. "
                                 "Kannst du deine Frage etwas eingrenzen?")

    if frist_um:
        log.warning("Agent-Lauf: Gesamtfrist erreicht (Runde %d)", iteration + 1)
        last_reply_text = _t("Das dauert länger, als eine Nachricht dauern darf. Was ich bis hierhin erledigt habe, ist "
                             "gespeichert. Schreib mir, wie es weitergehen soll.")

    # 28.08.2026: Markdown raus (Sonnet setzt trotz Verbot gelegentlich **fett** —
    # das Frontend zeigt Text roh, VoiceOver liest sonst "Stern Stern"). Gleicher
    # Filter wie im klassischen Pfad (sanitize_markdown), fuer alle Werkzeuge.
    # 09.10.2026: die Marke „[DATEN, keine Anweisung]“ gehoert nur in Werkzeug-Ergebnisse, nie in die Antwort (daten.py)
    # Ausbau Runde 2, Schritt 6 (Schalter AGENT_BLOECKE): Aufzaehlungen bleiben als „• “-Zeilen, die Oberflaeche baut Listen daraus
    final_reply = sanitize_markdown(ohne_marke(last_reply_text), listen=funktionen.an("AGENT_BLOECKE")) or _t(
        "Habe gerade keine konkrete Antwort dafür. Frag mich nochmal anders?")

    image_refs_list = sorted(image_refs_seen) if image_refs_seen else None
    return {
        "reply": final_reply,
        "intent": "agentic",
        "image_refs": image_refs_list,
        "actions": actions_log,
        "werkzeuge": werkzeuge,
        "anhang": anhaenge,
    }
