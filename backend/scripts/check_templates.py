#!/usr/bin/env python3
"""Bau-Zeit-Pruefung der Jinja2-Templates.

Anlass (31.08.2026): In `abo.html` standen zwei Jinja-Ausdruecke INNERHALB des
`{% raw %}`-Blocks. Jinja ersetzt dort nichts — das `{{ ... }}` landete woertlich
im JavaScript, das komplette Seitenskript brach mit "Unexpected token '{'" ab und
die Abo-Seite zeigte nur noch "Wird geladen ...". Eingeschleppt am 29.08. mit den
Aktionspreisen, gefunden erst zwei Tage spaeter — weil kein Test prueft, ob eine
Seite ueberhaupt etwas anzeigt. Derselbe Fehler hatte am 25.08. schon einmal
`app.html` lahmgelegt (Jinja-Kommentar im raw-Block).

Geprueft wird deshalb je Template:
  1. `{% raw %}` und `{% endraw %}` kommen gleich oft vor und sind sauber
     verschachtelt (kein endraw ohne raw, kein offener raw-Block am Dateiende).
  2. Innerhalb eines raw-Blocks steht KEIN `{{ ... }}` und KEIN `{# ... #}`.
     Wer dort einen Wert braucht, schliesst den raw-Block kurz:
         {% endraw %}
         const X = {{ wert|tojson }};
         {% raw %}

Aufruf ohne Argumente. Exit-Code 1, wenn etwas gefunden wird.
Pfade werden relativ zum Skript aufgeloest — laeuft im Repo (backend/scripts/)
wie im Container (/app/scripts/).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "templates"

RE_RAW = re.compile(r"\{%-?\s*raw\s*-?%\}")
RE_ENDRAW = re.compile(r"\{%-?\s*endraw\s*-?%\}")
# Ausdruck {{ ... }} oder Kommentar {# ... #} — beides wird im raw-Block NICHT
# ersetzt und ist damit in einem <script> ein Syntaxfehler.
RE_JINJA = re.compile(r"\{\{|\{#")


def pruefe(pfad: Path):
    """Liefert eine Liste von Befunden (Zeilennummer, Text) fuer EIN Template."""
    befunde = []
    zeilen = pfad.read_text(encoding="utf-8").split("\n")
    tiefe = 0
    offen_seit = None

    for nr, zeile in enumerate(zeilen, 1):
        # Erst schliessen, dann oeffnen: eine Zeile kann beides enthalten.
        if RE_ENDRAW.search(zeile):
            if tiefe == 0:
                befunde.append((nr, "{% endraw %} ohne offenen {% raw %}-Block"))
            else:
                tiefe -= 1
                offen_seit = None
            continue
        if RE_RAW.search(zeile):
            tiefe += 1
            offen_seit = nr
            continue
        if tiefe > 0 and RE_JINJA.search(zeile):
            befunde.append((nr, "Jinja im raw-Block (ab Zeile %d): %s"
                            % (offen_seit, zeile.strip()[:110])))

    if tiefe > 0:
        befunde.append((offen_seit or len(zeilen),
                        "{% raw %} wird nie geschlossen"))
    return befunde


def main():
    if not TEMPLATES.is_dir():
        print("Template-Verzeichnis nicht gefunden: %s" % TEMPLATES)
        return 1

    dateien = [p for p in sorted(TEMPLATES.glob("*.html")) if ".bak" not in p.name]
    treffer = 0
    for pfad in dateien:
        for nr, text in pruefe(pfad):
            print("FEHLER %s:%d — %s" % (pfad.name, nr, text))
            treffer += 1

    if treffer:
        print()
        print("%d Befund(e). Werte gehoeren AUSSERHALB des raw-Blocks:" % treffer)
        print("  {% endraw %}")
        print("  const X = {{ wert|tojson }};")
        print("  {% raw %}")
        return 1

    print("Template-Check OK: %d Templates, raw-Bloecke sauber, kein Jinja darin."
          % len(dateien))
    return 0


if __name__ == "__main__":
    sys.exit(main())
