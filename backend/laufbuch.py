"""Laufbuch: was in diesem Prozess gerade im Hintergrund arbeitet (Steve 16./17.10.2026, Live-Fenster „zum Anklemmen“).

Jeder Lauf, der länger dauert oder mehrere Stücke abarbeitet — Alt-Texte, Quickinfos, Tagging, Prüfung, Umwandeln, Übersetzen,
Export, ein Auftrag mit mehreren Schritten —, meldet sich hier an, sagt, an welchem Stück er gerade ist, und meldet sich ab. Das
Arbeitsfenster der Agentenansicht liest daraus (GET /api/agent/live, inkluagent/arbeitsfenster.live): was läuft, wo, wie weit, und
fährt mit der Kamera dorthin. Ein neues Werkzeug schließt sich mit drei Aufrufen an:

    with laufbuch.lauf("tagging", user_id=u, project_id=p, document_id=d, gesamt=1) as l:
        l.stueck(document_id=d)          # woran er gerade ist (Bild, Feld, Dokument, Seite, Ausgabe)
        ...                              # arbeiten
        l.weiter()                       # ein Stück fertig

Ohne `with`: `l = laufbuch.beginnen(...)`, am Ende `l.beenden(ok)` — im finally, damit auch ein Abbruch austrägt.

Grundsätze:
- Nur, was hier wirklich läuft. Ein Status in der Datenbank kann nach einem Absturz hängen bleiben; das Laufbuch lebt im Prozess und
  ist nach einem Neustart leer. Darum zeigt das Fenster nie einen hängenden Stand.
- Nur, was der InkluAgent macht (Steve 16.10.2026: „natürlich soll nur das gestreamt werden, was der Agent auch an Aufgaben macht“):
  `quelle` ist "agent", wenn der Lauf aus einem Werkzeug des Agenten heraus beginnt (agent_kontext, ein ContextVar — er geht an
  asyncio-Aufgaben weiter; an Threads gibt der Aufrufer ihn über `quelle=` mit), sonst "oberflaeche". Das Fenster zeigt nur
  Agenten-Läufe.
- Kein Fremdtext: das Laufbuch speichert nur Kennungen und Zahlen. Namen holt arbeitsfenster.live aus der Datenbank (mit
  Besitzprüfung) — so kann kein Lauf etwas in die Seite schreiben, das nicht aus den eigenen Daten des Kontos kommt.
- Nie eine Ausnahme nach außen: Melden darf keinen Lauf stören.
- Beendete Läufe bleiben kurz (NACHLAUF_S) stehen, damit die Seite „fertig“ und das letzte Stück noch zeigen kann.
"""
from __future__ import annotations

import contextlib
import contextvars
import itertools
import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

NACHLAUF_S = 90          # so lange bleibt ein beendeter Lauf sichtbar
HOECHSTENS = 200         # Schutz gegen ein volllaufendes Laufbuch (es wird ohnehin aufgeräumt)

# Die Arten — eine Liste, damit die Seite sie kennt und Tests sie prüfen können. Neue Art = ein Eintrag hier und einer in
# arbeitsfenster.ANSICHT_JE_LAUF (welche Ansicht das Fenster dafür zeigt).
ARTEN = ("alt_texte", "quickinfos", "tagging", "korrektur", "pruefung", "umwandeln", "uebersetzen", "export", "auftrag")

# Lief der Lauf aus einem Werkzeug des Agenten heraus an? (agent_kontext setzt ihn für die Dauer eines Werkzeugs)
_AGENT: contextvars.ContextVar[bool] = contextvars.ContextVar("laufbuch_agent", default=False)

_SPERRE = threading.Lock()
_LAEUFE: dict[int, "Lauf"] = {}
_NUMMER = itertools.count(1)


@contextlib.contextmanager
def als(quelle: str):
    """Für die Dauer des Blocks gelten neue Läufe als Läufe dieser Quelle ("agent" oder "oberflaeche")."""
    token = _AGENT.set(quelle == "agent")
    try:
        yield
    finally:
        _AGENT.reset(token)


def agent_kontext():
    """Für die Dauer eines Agenten-Werkzeugs (ToolExecutor.execute): Läufe, die darin beginnen, gelten als Läufe des Agenten — auch
    in asyncio-Aufgaben, die es startet (die erben den Kontext). Threads und Aufgaben auf ANDEREN Schleifen erben ihn nicht: dafür
    mitnehmen() und mitnehmen_coro()."""
    return als("agent")


def durch_agent() -> bool:
    """Läuft der aktuelle Code innerhalb eines Agenten-Werkzeugs?"""
    return _AGENT.get()


def quelle() -> str:
    return "agent" if _AGENT.get() else "oberflaeche"


def mitnehmen(fn):
    """Für Threads und Pools (threading.Thread, executor.submit, run_in_executor): fn läuft dort mit der Quelle von JETZT —
    ein Lauf, den der Agent anstößt, bleibt ein Lauf des Agenten."""
    q = quelle()

    def mit_quelle(*args, **kwargs):
        with als(q):
            return fn(*args, **kwargs)
    return mit_quelle


def mitnehmen_coro(coro):
    """Für Koroutinen, die auf einer anderen Schleife laufen (run_coroutine_threadsafe, main.im_hauptloop): dieselbe Quelle."""
    q = quelle()

    async def mit_quelle():
        with als(q):
            return await coro
    return mit_quelle()


def _zahl(x) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


class Lauf:
    """Ein Eintrag im Laufbuch. Alle Methoden sind sicher aus jedem Thread und werfen nie."""

    def __init__(self, art: str, user_id: int, project_id: Optional[int], document_id: Optional[int], gesamt: int, quelle: str):
        self.nr = next(_NUMMER)
        self.art = art
        self.user_id = int(user_id)
        self.project_id = project_id
        self.document_id = document_id
        self.gesamt = max(0, int(gesamt or 0))
        self.fertig = 0
        self.stueck_daten: Optional[dict] = None   # {"art": "bild"|"feld"|"dokument"|"seite"|"ausgabe", Kennungen …}
        self.vorher_daten: Optional[dict] = None   # das Stück davor (fertig), für „Gerade fertig“
        self.station: Optional[str] = None         # bei Aufträgen: die Art des Schritts, der gerade läuft (z. B. "tagging")
        self.ergebnis_daten: Optional[dict] = None # wo das Ergebnis liegt, z. B. {"art": "ausgabe", "ausgabe_id": 12} (Ablage)
        self.quelle = quelle
        self.seit = time.time()
        self.ende: Optional[float] = None
        self.ok: Optional[bool] = None

    # ── melden ──
    def stueck(self, *, image_id=None, feld_id=None, document_id=None, seite=None, ausgabe_id=None, project_id=None) -> None:
        """Woran der Lauf gerade ist. Das vorige Stück wird „vorher“ (Gerade fertig)."""
        try:
            neu: dict = {}
            for name, wert in (("image_id", image_id), ("feld_id", feld_id), ("document_id", document_id), ("seite", seite),
                               ("ausgabe_id", ausgabe_id), ("project_id", project_id)):
                z = _zahl(wert)
                if z is not None:
                    neu[name] = z
            if "image_id" in neu:
                neu["art"] = "bild"
            elif "feld_id" in neu:
                neu["art"] = "feld"
            elif "ausgabe_id" in neu:
                neu["art"] = "ausgabe"
            elif "seite" in neu:
                neu["art"] = "seite"
            elif "document_id" in neu:
                neu["art"] = "dokument"
            else:
                return
            with _SPERRE:
                if self.stueck_daten and self.stueck_daten != neu:
                    self.vorher_daten = self.stueck_daten
                self.stueck_daten = neu
        except Exception:  # noqa: BLE001
            log.exception("Laufbuch: Stück nicht gemeldet (ignoriert)")

    def weiter(self, n: int = 1) -> None:
        """n Stücke fertig."""
        try:
            with _SPERRE:
                self.fertig = max(0, self.fertig + int(n))
                if self.gesamt and self.fertig > self.gesamt:
                    self.gesamt = self.fertig
        except Exception:  # noqa: BLE001
            log.exception("Laufbuch: Fortschritt nicht gemeldet (ignoriert)")

    def stand(self, fertig: int, gesamt: Optional[int] = None) -> None:
        """Fortschritt absolut setzen (wenn ein Lauf ihn selbst zaehlt, z. B. „Seite 3 von 12“)."""
        try:
            with _SPERRE:
                self.fertig = max(0, int(fertig))
                if gesamt is not None:
                    self.gesamt = max(self.fertig, int(gesamt))
        except Exception:  # noqa: BLE001
            log.exception("Laufbuch: Stand nicht gesetzt (ignoriert)")

    def gesamt_setzen(self, gesamt: int) -> None:
        try:
            with _SPERRE:
                self.gesamt = max(self.fertig, int(gesamt or 0))
        except Exception:  # noqa: BLE001
            log.exception("Laufbuch: Gesamt nicht gesetzt (ignoriert)")

    def ergebnis(self, *, ausgabe_id=None) -> None:
        """Wo das Ergebnis liegt (heute: ein Eintrag in „Meine Ablage“) — das Fenster zeigt es am Ende."""
        try:
            z = _zahl(ausgabe_id)
            if z is not None:
                with _SPERRE:
                    self.ergebnis_daten = {"art": "ausgabe", "ausgabe_id": z}
        except Exception:  # noqa: BLE001
            log.exception("Laufbuch: Ergebnis nicht gemeldet (ignoriert)")

    def station_setzen(self, art: Optional[str]) -> None:
        """Bei Aufträgen: welcher Schritt gerade läuft (eine der ARTEN) — das Fenster zeigt dessen Ansicht."""
        try:
            with _SPERRE:
                self.station = art if art in ARTEN else None
        except Exception:  # noqa: BLE001
            log.exception("Laufbuch: Station nicht gesetzt (ignoriert)")

    def beenden(self, ok: bool = True) -> None:
        try:
            with _SPERRE:
                if self.ende is None:
                    self.ende = time.time()
                    self.ok = bool(ok)
                    if self.stueck_daten and ok:
                        self.vorher_daten = self.stueck_daten   # das letzte Stück ist jetzt fertig
                        self.stueck_daten = None
        except Exception:  # noqa: BLE001
            log.exception("Laufbuch: Ende nicht gemeldet (ignoriert)")

    @property
    def laeuft(self) -> bool:
        return self.ende is None

    def als_dict(self) -> dict:
        with _SPERRE:
            return {"nr": self.nr, "art": self.art, "user_id": self.user_id, "project_id": self.project_id,
                    "document_id": self.document_id, "gesamt": self.gesamt, "fertig": self.fertig,
                    "stueck": dict(self.stueck_daten) if self.stueck_daten else None,
                    "vorher": dict(self.vorher_daten) if self.vorher_daten else None,
                    "station": self.station, "quelle": self.quelle, "seit": self.seit, "laeuft": self.ende is None,
                    "ok": self.ok, "ergebnis": dict(self.ergebnis_daten) if self.ergebnis_daten else None}


def _aufraeumen(jetzt: float) -> None:
    for nr, l in list(_LAEUFE.items()):
        if l.ende is not None and jetzt - l.ende > NACHLAUF_S:
            _LAEUFE.pop(nr, None)
    if len(_LAEUFE) > HOECHSTENS:
        for nr in sorted(_LAEUFE)[:len(_LAEUFE) - HOECHSTENS]:
            _LAEUFE.pop(nr, None)


def beginnen(art: str, *, user_id, project_id=None, document_id=None, gesamt: int = 0, quelle: Optional[str] = None) -> Lauf:
    """Einen Lauf anmelden. Gibt immer einen Lauf zurück (auch bei unbekannter Art — dann wird er nur nicht gezeigt)."""
    q = quelle if quelle in ("agent", "oberflaeche") else ("agent" if _AGENT.get() else "oberflaeche")
    try:
        l = Lauf(art, int(user_id), _zahl(project_id), _zahl(document_id), gesamt, q)
    except Exception:  # noqa: BLE001 — z. B. user_id fehlt: ein Lauf, der nirgends steht
        log.exception("Laufbuch: Lauf nicht angemeldet (ignoriert)")
        return Lauf(art, 0, None, None, 0, "oberflaeche")
    if art not in ARTEN:
        log.warning("Laufbuch: unbekannte Art %r — Lauf wird nicht gezeigt", art)
        return l
    with _SPERRE:
        _aufraeumen(time.time())
        _LAEUFE[l.nr] = l
    return l


@contextlib.contextmanager
def lauf(art: str, **kw):
    """Anmelden, am Ende abmelden — auch bei Ausnahme oder Abbruch (CancelledError). ok=False, wenn eine Ausnahme kam."""
    l = beginnen(art, **kw)
    try:
        yield l
    except BaseException:
        l.beenden(False)
        raise
    else:
        l.beenden(True)


def fuer_konto(user_id: int, nur_agent: bool = True) -> list[dict]:
    """Die Läufe dieses Kontos, jüngste zuerst: laufende und kürzlich beendete. nur_agent: nur Läufe des Agenten."""
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return []
    with _SPERRE:
        _aufraeumen(time.time())
        liste = [l for l in _LAEUFE.values() if l.user_id == uid and (not nur_agent or l.quelle == "agent")]
    liste.sort(key=lambda l: (l.ende is None, l.nr), reverse=True)   # laufende zuerst, darin die jüngsten
    return [l.als_dict() for l in liste]


def laeuft_im_projekt(project_id: int, art: Optional[str] = None) -> bool:
    """Arbeitet in diesem Prozess gerade ein Lauf (dieser Art) in diesem Projekt? Für Rückmeldung und Überblick."""
    with _SPERRE:
        return any(l.ende is None and l.project_id == project_id and (art is None or l.art == art) for l in _LAEUFE.values())


def _leeren() -> None:
    """Nur für Tests."""
    with _SPERRE:
        _LAEUFE.clear()
