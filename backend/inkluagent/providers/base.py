"""LLM-Provider-Schnittstelle fuer InkluAgent.

Ein Provider kapselt einen LLM-Anbieter (Mistral, Gemini, ...). Der
chat_engine ruft nur diese Schnittstelle — kennt keine Provider-Details.
"""
from abc import ABC, abstractmethod
from typing import Optional


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
