"""InkluAgent-Tools für agentic Sonnet-Architektur (12.05.2026).

Jedes Modul hier kapselt einen Werkzeugkasten:
- project.py:  Projekt + Bilder lesen, Bild anschauen
- altext.py:   Alt-Text generieren + speichern
- search.py:   Web-Suche (Tavily)
- definitions.py:  Anthropic-Tool-Schemas + Dispatcher

Convention: jedes Tool-Funktion gibt ein dict zurück
({"ok": True, "result": ...} oder {"ok": False, "error": "..."}).
Der Tool-Loop in chat_engine.py serialisiert das als JSON-String fuer
Claude als tool_result-Block.
"""
