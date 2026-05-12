"""Markdown-Filter fuer InkluAgent-Antworten.

Mistral-Medium produziert auch bei explizitem Verbot weiter Markdown.
Dieser Filter raeumt auf, BEVOR die Antwort im Frontend gezeigt wird —
das Frontend rendert kein Markdown, sonst stehen rohe Sternchen im Text.

Was entfernt wird:
- **fett** und __fett__ → fett
- *kursiv* (innerhalb von Worten erhalten, am Wortrand entfernt)
- Bullet-Praefixe '- ', '* ', '+ ' am Zeilenanfang
- Heading-Praefixe '# ', '## ', '### ' am Zeilenanfang
- Trennlinien '---' oder '***' als ganze Zeilen
- Inline-Code `xxx` → xxx
- Code-Block-Fences ``` (Backticks)

Was NICHT entfernt wird:
- Emojis (sind seit 05.05.2026 sparsam erlaubt)
- Einzelne Sterne in Texten wie "3*5"
- Bindestriche in Wortzusammensetzungen
- Numerische Listen "1." (lassen wir, sind oft semantisch sinnvoll)
"""
import re


_RE_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_RE_BOLD_UNDER = re.compile(r"__(.+?)__", re.DOTALL)
# Italic mit Stern: ein Stern, dann Inhalt ohne Stern oder Newline, dann Stern.
# Nur am Wort-/Satzrand, sonst koennten Multiplikationen erwischt werden.
_RE_ITALIC_STAR = re.compile(r"(?<![\w*])\*([^\s*][^*\n]*?[^\s*]|\S)\*(?![\w*])")
_RE_ITALIC_UNDER = re.compile(r"(?<![\w_])_([^\s_][^_\n]*?[^\s_]|\S)_(?![\w_])")
_RE_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
_RE_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_RE_HRULE = re.compile(r"^[ \t]*([-*_])[ \t]*\1[ \t]*\1[\1\s]*$", re.MULTILINE)
_RE_CODE_FENCE = re.compile(r"```[a-zA-Z0-9]*\n?")
_RE_INLINE_CODE = re.compile(r"`([^`\n]+)`")


def sanitize_markdown(text: str) -> str:
    """Entfernt Markdown-Markup aus einem Text. Idempotent."""
    if not text:
        return text
    s = text
    s = _RE_CODE_FENCE.sub("", s)
    s = _RE_INLINE_CODE.sub(r"\1", s)
    s = _RE_BOLD_STAR.sub(r"\1", s)
    s = _RE_BOLD_UNDER.sub(r"\1", s)
    s = _RE_ITALIC_STAR.sub(r"\1", s)
    s = _RE_ITALIC_UNDER.sub(r"\1", s)
    s = _RE_HEADING.sub("", s)
    s = _RE_HRULE.sub("", s)
    s = _RE_BULLET.sub("", s)
    # Mehrere leere Zeilen auf max. eine reduzieren
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
