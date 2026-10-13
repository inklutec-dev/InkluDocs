"""LLM-Provider-Schnittstelle fuer InkluAgent.

Ein Provider kapselt einen LLM-Anbieter (Bedrock/Claude, spaeter ggf. weitere). Der
chat_engine ruft nur diese Schnittstelle — kennt keine Provider-Details.
"""
import time
from abc import ABC, abstractmethod
from typing import Optional


class FristErreicht(Exception):
    """Die Frist der Nachricht (agent_loop.run_agent) ist um, bevor die KI geantwortet hat. Kein Fehler der Verbindung: der Lauf
    endet mit dem Hinweis auf die Frist (Pruefung 13.10.2026, Runde 2, K-N3/S-N5)."""


class Uhr:
    """Zeit fuer die Aufrufe EINES Schritts an die KI unter der Frist der Nachricht (time.monotonic, aus agent_loop.run_agent).

    Vorher galt die Frist nur zwischen den Runden: ein Schritt mit langsamem Anbieter (Stream bis 180 s, dann drei Versuche ohne
    Stream zu je 180 s, dazu ein zweiter Versuch nach einer entgleisten Antwort) konnte sie um zwanzig Minuten ueberziehen. Jetzt
    bekommt der erste Aufruf eines Schritts mindestens MIN_AUFRUF_S (ein begonnener Schritt darf seine Antwort noch bekommen),
    jeder weitere (Wiederholung, Rueckfall ohne Stream) beginnt nur, solange vor der Frist noch MIN_AUFRUF_S bleiben, und endet
    spaetestens mit ihr. Ein Schritt ueberzieht die Frist so hoechstens um MIN_AUFRUF_S. Ohne Frist: jeder Aufruf `hoechstens`."""

    MIN_AUFRUF_S = 60.0

    def __init__(self, frist: Optional[float], hoechstens: float) -> None:
        self.frist = frist
        self.hoechstens = float(hoechstens)
        self._erster = True
        self._bis: Optional[float] = None   # Ende des laufenden Aufrufs (nur mit Frist)

    def _rest(self) -> float:
        return self.frist - time.monotonic() if self.frist is not None else float("inf")

    def weiterer_erlaubt(self) -> bool:
        """Darf nach dem ersten Aufruf noch einer beginnen?"""
        return self._rest() >= self.MIN_AUFRUF_S

    def naechster(self) -> float:
        """Zeit (Sekunden) fuer den naechsten Aufruf; FristErreicht, wenn keiner mehr beginnen darf."""
        if self.frist is None:
            zeit = self.hoechstens
        elif self._erster:
            zeit = min(self.hoechstens, max(self._rest(), self.MIN_AUFRUF_S))
        elif not self.weiterer_erlaubt():
            raise FristErreicht()
        else:
            zeit = min(self.hoechstens, self._rest())
        self._erster = False
        if self.frist is not None:
            self._bis = time.monotonic() + zeit
        return zeit

    def pruefen(self) -> None:
        """Waehrend eines Aufrufs, der Stueck fuer Stueck liest (Stream): FristErreicht, wenn seine Zeit um ist. Der Timeout der
        Verbindung gilt je Lesevorgang und begrenzt einen tröpfelnden Stream nicht."""
        if self._bis is not None and time.monotonic() > self._bis:
            raise FristErreicht()


class LLMProvider(ABC):
    """Abstrakte Schnittstelle fuer Chat-LLMs mit optionaler Bild-Eingabe."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        images: Optional[list[bytes]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        """Sende eine Konversation, erhalte die Assistenten-Antwort.

        messages: [{"role": "system|user|assistant", "content": "..."}]
        images:   optionale Liste roher Bytes (jpeg/png). Werden, falls
                  unterstuetzt, der letzten user-Nachricht beigefuegt.
        model:    Modell-Override (sonst Provider-Default).
        Returns:  Antworttext des Assistenten.
        """
        ...
