"""Schubfaecher des InkluAgent-Systemprompts — EIN Zusammensetzer statt Anhaengen an vielen Stellen (Steve 14.10.2026:
„professionell aufgebaut … jederzeit stolz vom Code und vom Aufbau jemandem zeigen“; Umbau 15.10.2026 ohne Wortaenderung, Beweis:
tests/snapshots/agent_prompts/ Byte fuer Byte gleich).

Der Prompt hat drei Ebenen, immer in dieser Reihenfolge:
1. PROJEKTART (genau eine, projektart()): pdf, word, grafik, web, formular, alt_text (Bild-Projekt ohne AGENT_BILD_WERKZEUGE
   und Altbestand), konto (Seite ohne Projekt). Bringt Rolle, Werkzeuge und Ablauf dieser Projektart.
2. FAEHIGKEITEN (FAECHER, feste Reihenfolge): je ein Schubfach mit Schalter, Projektarten, Werkzeugen und der Angabe, ob es in
   der Demo vorkommen darf. Ein Fach steht im Prompt, wenn es in der Lage gilt.
3. DEMO-LEITPLANKE (nur in der Demo, ganz am Ende).

Teile werden mit einer Leerzeile verbunden („\\n\\n“). Was gilt, entscheidet die Lage (Schalter, Gespraech des Kontos), nicht
ein „ausser wenn“ im Text: Beispiel Auftraege — mit ihnen bekommt die Projektart PDF den Schritt „mehrere Schritte = ein Auftrag“
(system_pdf(auftraege=True)), ohne sie „Schritt fuer Schritt“.

Die Werkzeuge selbst (Definitionen und Ausfuehrung) stellt weiter agent_loop zusammen; hier steht nur der Text. Die Module der
Faecher werden erst beim Zusammensetzen geladen (sie laden ihrerseits Teile dieses Pakets).
Doku: inkluagent/prompts/ARCHITEKTUR.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import funktionen

PROJEKTARTEN = ("pdf", "word", "grafik", "web", "formular", "alt_text", "konto")
_IM_PROJEKT = frozenset(PROJEKTARTEN) - {"konto"}


@dataclass(frozen=True)
class Lage:
    """Wofuer der Prompt gebaut wird. Die vier Gespraechs-Schalter setzt agent_loop.run_agent (nur mit Lauf und Gespraech des
    Kontos, also nie in der Demo); ohne sie entsteht der Prompt der Projektart mit Kosten und Hochladen."""
    projektart: str
    user_id: int = 0
    begleiter: bool = False
    karten: bool = False
    rueckmeldung: bool = False
    auftraege: bool = False
    demo_leitplanke: Optional[str] = None   # gesetzt = Demo: Faecher mit demo=False fehlen immer

    @property
    def demo(self) -> bool:
        return bool(self.demo_leitplanke)


def projektart(project: Optional[dict]) -> str:
    """Die Projektart eines Projekts (None = Seite ohne Projekt). Dieselbe Weiche waehlt in agent_loop die Werkzeuge."""
    if project is None:
        return "konto"
    p = project or {}
    if p.get("tool") == "formular":                                   # alte Formular-Projekte behalten ihren Satz
        return "formular"
    if p.get("project_type") == "docx":
        return "word"
    if p.get("project_type") == "pdf" and p.get("tool") == "pdf":
        return "pdf"
    bild = ("grafik" if (p.get("project_type") == "images" or p.get("tool") == "grafik") else
            "web" if (p.get("project_type") == "url" or p.get("tool") == "web") else "")
    if bild and funktionen.an("AGENT_BILD_WERKZEUGE"):
        return bild
    return "alt_text"


# ─── 1. Projektart ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def text_projektart(lage: Lage) -> str:
    art = lage.projektart
    if art == "formular":
        from .system_formular import SYSTEM_FORMULAR
        return SYSTEM_FORMULAR
    if art == "konto":
        from .. import konto
        return konto.system_konto(konto.aendern_an())
    from .system_agent import system_agent
    basis = system_agent()                                            # Alt-Text-Grundlage aller Projekte mit Bildern
    if art == "word":
        from .system_ausgaben import SYSTEM_AUSGABEN
        return basis + "\n\n" + SYSTEM_AUSGABEN
    if art == "pdf":
        from .system_pdf import system_pdf
        return basis + "\n\n" + system_pdf(auftraege=lage.auftraege)
    if art in ("grafik", "web"):
        from .system_bild import system_bild
        return basis + "\n\n" + system_bild(art)
    if art == "alt_text":
        return basis
    raise ValueError(f"Unbekannte Projektart: {art}")


# ─── 2. Faehigkeiten ─────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Fach:
    name: str
    schalter: str                  # wer es einschaltet (fuer Doku und Pruefung; die Bedingung selbst steht in gilt)
    projektarten: frozenset        # wo es vorkommen kann
    werkzeuge: tuple               # Werkzeuge, die es erklaert (Namen wie im Werkzeugsatz)
    demo: bool                     # darf in der Demo vorkommen
    gilt: Callable[[Lage], bool]
    text: Callable[[Lage], str]


def _kosten(lage: Lage) -> str:
    from .. import sicherheit
    return sicherheit.PROMPT_ZUSATZ


def _hochladen(lage: Lage) -> str:
    from .. import hochladen
    return hochladen.PROMPT_ZUSATZ


def _hochladen_hier(lage: Lage) -> str:
    """Feld, Knopf „Dateien anhängen“ oder (ohne Projekt) der Hinweis auf die Oberflaeche — im Projekt ohne Weg im Agenten leer."""
    from .. import hochladen
    if lage.projektart == "konto":
        return hochladen.prompt_hochladen_hier(lage.user_id)
    return hochladen.prompt_hochladen_hier(lage.user_id, im_projekt=True)


def _gespraeche(lage: Lage) -> str:
    from .. import konto
    return konto._GESPRAECHE


def _begleiter(lage: Lage) -> str:
    """Mit der Kauf-Karte ohne das pauschale „Abo … Zahlung machst du nie selbst“ (Widerspruch 2)."""
    from .. import begleiter
    return begleiter.PROMPT_MIT_KAUFEN if _kaufen_gilt(lage) else begleiter.PROMPT


def _karten(lage: Lage) -> str:
    """Ohne Express-Service ohne express_bestellen_vorbereiten (das Werkzeug fehlt dann)."""
    from .. import karten
    return karten.PROMPT if funktionen.an("EXPRESS") else karten.PROMPT_OHNE_EXPRESS


def _kaufen_gilt(lage: Lage) -> bool:
    from .. import karten
    return lage.karten and karten.kaufen_an()


def _kaufen(lage: Lage) -> str:
    from .. import karten
    return karten.PROMPT_KAUFEN


def _rueckmeldung(lage: Lage) -> str:
    """Nennt nur die Arbeiten im Hintergrund, die es gibt (Widerspruch 3)."""
    from .. import rueckmeldung
    return rueckmeldung.prompt()


def _auftraege(lage: Lage) -> str:
    """Mit den PDF-Saetzen (Michaels Beispiel, schon getaggt) nur im PDF-Projekt und ohne Projekt."""
    from .. import auftraege
    return auftraege.prompt(lage.projektart)


FAECHER: tuple = (
    Fach("kosten", "immer im Projekt (Server-Regel seit Pruefung 13.10.2026, Runde 2)", _IM_PROJEKT, (), True,
         lambda l: True, _kosten),
    Fach("hochladen", "AGENT_HOCHLADEN", frozenset(PROJEKTARTEN), (), True,
         lambda l: funktionen.an("AGENT_HOCHLADEN"), _hochladen),
    # Wie die Datei hierher kommt: im Projekt nur, wenn es im Agenten einen Weg gibt (sonst leer und weg), ohne Projekt immer
    Fach("hochladen_hier", "AGENT_HOCHLADEN + AGENT_HOCHLADEFELD bzw. Agentenansicht (hochladen.prompt_hochladen_hier)",
         frozenset(PROJEKTARTEN), (), True, lambda l: True, _hochladen_hier),
    Fach("gespraeche", "AGENT_GESPRAECHE", frozenset({"konto"}), ("fruehere_gespraeche",), True,
         lambda l: l.projektart == "konto" and funktionen.an("AGENT_GESPRAECHE"), _gespraeche),
    Fach("begleiter", "AGENT_BEGLEITER + AGENT_UEBERALL + AGENT_KONTO, Gespraech des Kontos", frozenset(PROJEKTARTEN),
         ("hinfuehren", "einstellung_aendern"), False, lambda l: l.begleiter, _begleiter),
    Fach("karten", "AGENT_KARTEN, Gespraech des Kontos", frozenset(PROJEKTARTEN),
         ("einladen_vorbereiten", "express_bestellen_vorbereiten"), False, lambda l: l.karten, _karten),
    Fach("kaufen", "AGENT_KAUFEN + Karten + Stripe eingerichtet (karten.kaufen_an)", frozenset(PROJEKTARTEN),
         ("credits_kaufen_vorbereiten", "abo_buchen_vorbereiten"), False, _kaufen_gilt, _kaufen),
    Fach("rueckmeldung", "AGENT_RUECKMELDUNG + AGENT_KONTO, Gespraech des Kontos", frozenset(PROJEKTARTEN),
         ("melden_wenn_fertig",), False, lambda l: l.rueckmeldung, _rueckmeldung),
    Fach("auftraege", "AGENT_AUFTRAEGE + Karten + Rueckmeldung", frozenset(PROJEKTARTEN),
         ("auftrag_vorbereiten", "auftrag_stand", "auftrag_abbrechen"), False, lambda l: l.auftraege, _auftraege),
)
NACH_NAME = {f.name: f for f in FAECHER}


def faecher(lage: Lage) -> list[Fach]:
    """Die Faecher, die in dieser Lage im Prompt stehen (Reihenfolge von FAECHER). In der Demo nie ein Fach mit demo=False."""
    return [f for f in FAECHER if lage.projektart in f.projektarten and not (lage.demo and not f.demo) and f.gilt(lage)]


# ─── Zusammensetzen ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def teile(lage: Lage) -> list[tuple[str, str]]:
    """(Name, Text) in der Reihenfolge des Prompts — fuer Tests und die Doku. Leere Faecher fehlen."""
    if lage.projektart not in PROJEKTARTEN:
        raise ValueError(f"Unbekannte Projektart: {lage.projektart}")
    out = [("projektart:" + lage.projektart, text_projektart(lage))]
    for f in faecher(lage):
        text = f.text(lage)
        if text:
            out.append((f.name, text))
    if lage.demo:
        out.append(("demo_leitplanke", lage.demo_leitplanke))
    return out


def zusammensetzen(lage: Lage) -> str:
    """Der Systemprompt fuer diese Lage."""
    return "\n\n".join(text for _name, text in teile(lage))
