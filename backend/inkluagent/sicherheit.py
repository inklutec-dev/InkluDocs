"""Sicherheitsfundament des InkluAgent (Ausbau Runde 1, Schritt 2, 09.10.2026; Konzept InkluAgent 3.3 und 3.4).

Vier Sperren auf dem SERVER, nicht nur im Prompt. 1 und 2 gelten seit der Pruefung 13.10.2026 (Runde 2, S2) immer: sie sind
Teil der Zustimmung selbst, ohne sie genuegte irgendeine spaetere Nachricht und es liefen beliebig viele bezahlte Einzelaktionen
ohne Karte. 3 und 4 aendern Grenzen und haengen weiter am Schalter funktionen.AGENT_SICHERHEIT (INKLUAGENT_SICHERHEIT=an):

1. Deckel fuer bezahlte Einzelaktionen ohne Karte (Steve: „höchstens eine je Nachricht, alles darüber mit Karte“):
   Alt-Text generieren oder aendern, Quickinfo generieren oder aendern (EINZEL_BEZAHLT). Je Nutzer-Nachricht laeuft hoechstens
   EINE davon ohne Rueckfrage; jede weitere legt der Server als Angebot mit Karte vor (dieselbe Zwei-Schritt-Freigabe wie die
   grossen Aktionen, tools/ausgaben.py). Vorher konnten sechs Werkzeug-Runden mit parallelen Aufrufen viele Einzelbuchungen ausloesen.
2. Ja-Pruefung: bestaetigt=true vom MODELL gilt nur, wenn die Nachricht, die der Nutzer in dieser Runde GETIPPT hat, ein kurzes,
   eindeutiges Ja ist (ist_klares_ja, sechs Oberflaechensprachen, ohne Verneinung, ohne Frage) und direkt auf das Angebot folgt.
   Sonst bleibt nur der Knopf der Karte. Eingeschleuster Text kann keine Nutzer-Nachricht schreiben. Seit der Pruefung
   13.10.2026 (S2) unabhaengig vom Schalter: ohne sie genuegte irgendeine spaetere Nachricht, und das Modell bestimmte mit seinem
   Antworttext, worauf der Nutzer „Ja“ sagt. Unumkehrbares (Loeschen) geht nur ueber den Knopf (tools/ausgaben.py).
3. Robuste Tagesgrenze: eigener Zaehler je Konto und Tag (Tabelle agent_tageszaehler). Vorher zaehlte die Grenze ueber
   chat_messages: Projekt loeschen oder Verlauf leeren senkte den Zaehler.
4. Tages-Kostendeckel je Konto aus ki_aufrufe (Zweck „chatbot“, also die Kosten des Gespraechs selbst; Alt-Texte und
   Quickinfos bezahlt der Kunde mit Credits): INKLUAGENT_KOSTEN_DECKEL_EUR, Vorgabe 10 Euro je Konto und Tag (UTC).
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Optional

import funktionen

# Werkzeug -> Aktion in billing.AKTIONS_PREISE (Preis der Einzelaktion)
EINZEL_BEZAHLT = {
    "generate_alt_text": "bild_generierung",
    "update_alt_text": "alt_text_aenderung_chatbot",
    "generate_quickinfo": "quickinfo_generierung",
    "update_quickinfo": "quickinfo_aenderung_chatbot",
}
EINZEL_OHNE_KARTE_JE_NACHRICHT = 1


def an() -> bool:
    return funktionen.an("AGENT_SICHERHEIT")


def _kosten_deckel_cent() -> float:
    try:
        return max(0.0, float((os.environ.get("INKLUAGENT_KOSTEN_DECKEL_EUR") or "10").replace(",", "."))) * 100.0
    except ValueError:
        return 1000.0


# ─── 2. Ja-Pruefung ─────────────────────────────────────────────────────────────────────────────────────────────────────
# Ein klares Ja besteht NUR aus diesen Woertern (hoechstens sechs, hoechstens 60 Zeichen) und enthaelt mindestens eines aus
# JA_KERN. Alles andere (Zahlen, Bildnummern, „aber“, Fragen, Verneinungen) ist kein klares Ja — dann zaehlt nur die Karte.
# Sprachen der Oberflaeche: de, en, fr, es, da, sv.
JA_KERN = frozenset("""
ja jawohl jep jo ok okay einverstanden passt genau klar gerne los bestätigt bestätige mach machs starte
yes yeah yep sure confirmed confirm agreed go proceed
oui ouais daccord confirmé confirme vasy
sí si vale claro confirmado confirmo adelante hazlo
jep okay bekræftet bekræfter gør
japp visst bekräftat bekräftar kör
""".split())
JA_ERLAUBT = JA_KERN | frozenset("""
bitte danke mach machen das es weiter fortfahren starte starten speichern übernehmen löschen umwandeln herunterladen
taggen übersetzen generieren exportieren ausführen bitteschön
please thanks thank you go ahead do it proceed start save delete convert download translate generate export run
merci sil vous plaît vasy fais le continue continuer lance lancer enregistrer supprimer convertir télécharger traduire générer
por favor gracias hazlo haz lo sigue continúa empieza guarda guardar borra borrar convierte descarga traduce genera
tak gør det fortsæt start gem slet konverter download oversæt generer
tack gör det fortsätt starta spara radera konvertera ladda ner översätt generera
""".split())
VERNEINUNG = frozenset("""
nein nicht kein keine keinen nö nee stopp stop abbrechen warte warten später doch aber
no not dont don't never cancel wait later but
non pas ne jamais annuler attends attendre plus tard mais
nunca cancelar espera esperar luego pero
nej ikke aldrig annuller vent senere men
inte aldrig avbryt vänta senare
""".split())
_ZEICHEN = re.compile(r"[^\wäöüßàâçéèêëîïôûùüÿñæøåáíóú' ]+", re.IGNORECASE)


def ist_klares_ja(text: Optional[str]) -> bool:
    """Kurz und eindeutig zugestimmt? „Ja“, „Ja, mach das.“, „OK, bitte speichern“, „Yes please“, „Oui, vas-y“, „Sí, hazlo“."""
    if not text:
        return False
    roh = " ".join(str(text).split()).strip()
    if not roh or len(roh) > 60 or "?" in roh:
        return False
    flach = _ZEICHEN.sub(" ", roh.lower().replace("’", "'").replace("d'accord", "daccord").replace("vas-y", "vasy")
                         .replace("s'il", "sil").replace("don't", "dont"))
    woerter = [w for w in flach.split() if w]
    if not woerter or len(woerter) > 6:
        return False
    if any(w in VERNEINUNG for w in woerter):
        return False
    return all(w in JA_ERLAUBT for w in woerter) and any(w in JA_KERN for w in woerter)


# Die getippte Nachricht des Nutzers je Werkzeug-Runde (turn_id des ToolExecutors) und die Runde davor desselben Kontos —
# agent_loop merkt sie sich, die Freigabe in tools/ausgaben._angebot_einloesen liest sie. Im Prozess gehalten; eine Zustimmung
# muss ohnehin binnen Minuten kommen.
_NACHRICHT: dict[str, tuple[str, float, Optional[str]]] = {}   # turn_id -> (Text, Zeit, vorige turn_id des Kontos)
_LETZTE_RUNDE: dict[int, str] = {}                             # user_id -> juengste turn_id
_NACHRICHT_SPERRE = threading.Lock()
_NACHRICHT_BEHALTEN_S = 30 * 60


def nachricht_merken(turn_id: str, text: str, user_id: Optional[int] = None) -> None:
    jetzt = time.time()
    with _NACHRICHT_SPERRE:
        for k in [k for k, e in _NACHRICHT.items() if jetzt - e[1] > _NACHRICHT_BEHALTEN_S]:
            _NACHRICHT.pop(k, None)
        vorige = _LETZTE_RUNDE.get(int(user_id)) if user_id is not None else None
        _NACHRICHT[turn_id] = (text or "", jetzt, vorige)
        if user_id is not None:
            _LETZTE_RUNDE[int(user_id)] = turn_id


def nachricht(turn_id: Optional[str]) -> Optional[str]:
    with _NACHRICHT_SPERRE:
        e = _NACHRICHT.get(turn_id or "")
    return e[0] if e else None


def vorige_runde(turn_id: Optional[str]) -> Optional[str]:
    with _NACHRICHT_SPERRE:
        e = _NACHRICHT.get(turn_id or "")
    return e[2] if e else None


G_JA = ("Die Nachricht des Nutzers ist kein eindeutiges Ja. Als Zustimmung zählt nur eine kurze, eindeutige Antwort wie „Ja“, "
        "„Ja, mach das“ oder „OK“, ohne Verneinung und ohne Rückfrage. Führe nichts aus: frag noch einmal mit Preis und Ziel, "
        "oder sag dem Nutzer, dass er die Karte unter deiner Antwort bestätigen kann.")
G_DIREKT = ("Ein getipptes Ja gilt nur in der Nachricht direkt nach dem Angebot. Führe nichts aus: frag für diese Aktion neu "
            "(ohne bestaetigt), oder sag dem Nutzer, dass er die Karte unter deiner Antwort bestätigen kann.")


def ja_grund(turn_id: Optional[str], angebot_turn: Optional[str] = None) -> Optional[str]:
    """None = das getippte Ja zaehlt; sonst der Grund fuer das Modell. angebot_turn: die Runde, in der das Angebot entstand —
    das Ja muss in der Runde direkt danach stehen (S2)."""
    if not ist_klares_ja(nachricht(turn_id)):
        return G_JA
    if angebot_turn is not None and vorige_runde(turn_id) != angebot_turn:
        return G_DIREKT
    return None


# ─── 1. Deckel fuer bezahlte Einzelaktionen ────────────────────────────────────────────────────────────────────────────────
G_EINZEL = ("In dieser Nachricht lief schon eine bezahlte Einzelaktion ohne Rückfrage — mehr lässt der Server nicht zu. "
            "Für diese weitere liegt jetzt ein Angebot vor; unter deiner Antwort steht eine Karte mit Ziel und Preis. Nenne "
            "dem Nutzer, was du tun willst, Ziel und Preis. Ein klares Ja in seiner nächsten Nachricht (dann erneut mit "
            "bestaetigt=true) oder der Knopf der Karte führt es aus.")


G_HOCHGELADEN = ("Nach dem Hochladen startet der Server in derselben Nachricht nichts, was Credits kostet. Für diese Aktion "
                 "liegt jetzt ein Angebot vor; unter deiner Antwort steht eine Karte mit Ziel und Preis. Nenne beides und "
                 "warte auf den Knopf oder ein klares Ja in der nächsten Nachricht.")


def _ziel(executor, name: str, args: dict) -> tuple[Optional[int], str]:
    """(Kennung, Anzeige wie in der Oberflaeche) des Bildes bzw. Feldes, um das es geht."""
    try:
        if name in ("generate_alt_text", "update_alt_text"):
            iid = int(args.get("image_id"))
            return iid, bild_label(executor.project_id, iid)
        fid = int(args.get("feld_id"))
        from database import get_db
        from .tools import formular as _f
        conn = get_db()
        try:
            return fid, _f._ui_labels(conn, executor.project_id).get(fid, f"Feld {fid}")
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None, ""


def bild_label(project_id: int, image_id: int) -> str:
    """„Bild N“ bzw. „Dokument D, Bild N“ — dieselbe Zaehlung wie list_project_images und die Oberflaeche."""
    from database import get_db
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT i.id, COALESCE(d.doc_index, 0) AS di FROM images i LEFT JOIN documents d ON d.id = i.document_id "
            "WHERE i.project_id = ? ORDER BY COALESCE(d.doc_index, 0), i.page_number, i.image_index, i.id",
            (project_id,)).fetchall()
        mehrere = conn.execute("SELECT COUNT(*) FROM documents WHERE project_id = ?", (project_id,)).fetchone()[0] > 1
    finally:
        conn.close()
    je: dict[int, int] = {}
    for r in rows:
        je[r["di"]] = je.get(r["di"], 0) + 1
        if r["id"] == image_id:
            return f"Dokument {r['di']}, Bild {je[r['di']]}" if (mehrere and r["di"]) else f"Bild {je[r['di']]}"
    return f"Bild {image_id}"


def einzel_vorab(executor, name: str, args: dict) -> Optional[dict]:
    """VOR einer bezahlten Einzelaktion (ToolExecutor.execute). None = ausfuehren. Sonst ein Werkzeug-Ergebnis: Rueckfrage
    mit neuem Angebot (die Karte haengt ausgaben.nachbereiten an) oder der Grund, warum die Zustimmung nicht gilt."""
    if name not in EINZEL_BEZAHLT:
        return None
    # Hochgeladene Dateien (Runde 2, Schritt 1): in dieser Nachricht nichts Kostenpflichtiges ohne Karte
    nur_mit_karte = bool(getattr(executor, "nur_mit_karte", False))
    from .tools import ausgaben
    import billing
    aktion = EINZEL_BEZAHLT[name]
    ziel_id, ziel = _ziel(executor, name, args)
    schluessel = (int(executor.user_id), int(executor.project_id), "einzel:" + name, ziel_id)
    preis = int(billing.AKTIONS_PREISE.get(aktion) or 0)
    tid = getattr(executor, "turn_id", None) or ""
    if args.get("bestaetigt"):
        grund = ausgaben._angebot_einloesen(schluessel, preis, tid)
        if grund:
            return {"ok": True, "result": {"rueckfrage_noetig": True, "dokument": ziel, "preis": preis, "hinweis": grund,
                                           "grund": ausgaben._GRUND_KENNWORT.get(grund, "angebot")}}
        executor._einzel_bestaetigt = True      # zaehlt nicht zum Deckel „ohne Karte“
        return None
    if not nur_mit_karte and getattr(executor, "einzel_ohne_karte", 0) < EINZEL_OHNE_KARTE_JE_NACHRICHT:
        return None
    wache = billing.aktion_pruefung(executor.user_id, aktion)
    vorschau = {"rueckfrage_noetig": True, "dokument": ziel, "preis": preis, "verfuegbar": wache.get("verfuegbar"),
                "erlaubt": bool(wache.get("erlaubt")), "grund": "einzel_je_nachricht"}
    if not wache.get("erlaubt"):
        vorschau["hinweis"] = "Das Guthaben reicht nicht. Sag dem Nutzer Preis und Guthaben und verweise auf Abo & Verbrauch."
        return {"ok": True, "result": vorschau}
    ausgaben._angebot_merken(schluessel, preis, tid)
    vorschau["hinweis"] = (G_HOCHGELADEN if nur_mit_karte else G_EINZEL)
    return {"ok": True, "result": vorschau}


def einzel_nachher(executor, name: str, ergebnis: Any) -> None:
    """NACH der Ausfuehrung: eine erfolgreiche Einzelaktion ohne Karte zaehlt zum Deckel dieser Nachricht."""
    if name not in EINZEL_BEZAHLT:
        return
    if getattr(executor, "_einzel_bestaetigt", False):
        executor._einzel_bestaetigt = False
        return
    if isinstance(ergebnis, dict) and ergebnis.get("ok") and not (ergebnis.get("result") or {}).get("rueckfrage_noetig"):
        executor.einzel_ohne_karte = getattr(executor, "einzel_ohne_karte", 0) + 1


def werkzeuge_anpassen(defs: list) -> list:
    """Die vier Einzelaktionen nehmen bestaetigt an (nach Karte oder klarem Ja) — immer (Runde 2, S2)."""
    out = []
    for d in defs:
        if d.get("name") in EINZEL_BEZAHLT and "bestaetigt" not in ((d.get("input_schema") or {}).get("properties") or {}):
            d = dict(d)
            schema = dict(d.get("input_schema") or {})
            props = dict(schema.get("properties") or {})
            props["bestaetigt"] = {"type": "boolean", "description": (
                "Nur nach einer Rückfrage des Servers (rueckfrage_noetig) und einem klaren Ja des Nutzers in der Nachricht "
                "direkt danach: true führt genau dieses Angebot aus. Sonst weglassen.")}
            schema["properties"] = props
            d["input_schema"] = schema
        out.append(d)
    return out


PROMPT_ZUSATZ = """Bezahlte Einzelaktionen und Zustimmung (Server-Regel)

Alt-Text generieren oder ändern und Quickinfo generieren oder ändern kosten je Vorgang Credits. Je Nachricht des Nutzers führt der Server höchstens EINE davon ohne Rückfrage aus; für jede weitere liefert das Werkzeug eine Rückfrage (rueckfrage_noetig) mit Ziel und Preis, und unter deiner Antwort steht eine Karte. Nenne dann Ziel und Preis und warte. Ein getipptes Ja zählt nur, wenn die Nachricht des Nutzers direkt nach dem Angebot kommt und eine kurze, eindeutige Zustimmung ist („Ja“, „Ja, mach das“, „OK“) — ohne Verneinung, ohne Rückfrage, ohne weitere Wünsche. Sonst führt der Server nichts aus; frag noch einmal oder verweise auf den Knopf der Karte. Löschen geht nur über den Knopf der Karte."""


# ─── 3. Tagesgrenze und 4. Kostendeckel ────────────────────────────────────────────────────────────────────────────────────

def tageszaehler(user_id: int) -> int:
    from database import get_db
    conn = get_db()
    try:
        r = conn.execute("SELECT nachrichten FROM agent_tageszaehler WHERE user_id = ? AND tag = date('now')", (user_id,)).fetchone()
        return int(r[0]) if r else 0
    finally:
        conn.close()


def tageszaehler_erhoehen(user_id: int) -> None:
    """Eine angenommene Nachricht zaehlen. Loeschen von Projekten oder Verlauf aendert den Zaehler nicht."""
    from database import get_db
    conn = get_db()
    try:
        conn.execute("INSERT INTO agent_tageszaehler (user_id, tag, nachrichten) VALUES (?, date('now'), 1) "
                     "ON CONFLICT(user_id, tag) DO UPDATE SET nachrichten = nachrichten + 1", (user_id,))
        conn.commit()
    finally:
        conn.close()


def kosten_heute_cent(user_id: int) -> float:
    """KI-Kosten des Gespraechs (Zweck chatbot) heute (UTC), die DIESE Person verursacht hat. Je Person wie die Nachrichtengrenze
    (Pruefung 13.10.2026, S18: vorher je Team-Topf — ein Mitglied konnte den InkluAgent fuer das ganze Team bis Mitternacht
    sperren). Websuchen (Tavily) haben keine Token-Kosten in ki_aufrufe; ihre Zahl begrenzt die Nachrichtengrenze."""
    from database import get_db
    conn = get_db()
    try:
        r = conn.execute("SELECT COALESCE(SUM(kosten_eur_cent), 0) FROM ki_aufrufe WHERE user_id = ? AND zweck = 'chatbot' "
                         "AND created_at >= date('now')", (int(user_id),)).fetchone()
        return float(r[0] or 0)
    finally:
        conn.close()


def kosten_deckel_erreicht(user_id: int) -> bool:
    """Hat diese Person heute den Tages-Kostendeckel des InkluAgent erreicht? (Nachricht und Rueckmeldung, rueckmeldung.py)"""
    deckel = _kosten_deckel_cent()
    return bool(deckel and kosten_heute_cent(user_id) >= deckel)


def chat_sperre(user_id: int, limit: int, _=None) -> Optional[str]:
    """Vor einer Nachricht (Admins ausgenommen, das prueft der Endpunkt): Text fuer 429 oder None."""
    _ = _ or (lambda s: s)
    if tageszaehler(user_id) >= limit:
        return _('Du hast die {n} Chat-Nachrichten für heute genutzt. Morgen geht es weiter.').format(n=limit)
    if kosten_deckel_erreicht(user_id):
        return _('Der InkluAgent hat für dein Konto heute das Tageslimit erreicht. Morgen geht es weiter.')   # je Person (S18)
    return None
