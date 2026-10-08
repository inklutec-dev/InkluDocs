"""Lesefassung der Prompts: je Prompt eine TXT-Datei plus eine Gesamtdatei, aus den Snapshots.

Die Snapshots (prompts/snapshots, erzeugt mit `python3 -m scripts.render_prompts`) sind Markdown mit
Kopfzeilen; die Lesefassung ist reiner Text für Menschen mit Screenreader: Titelzeile, Leerzeile,
Prompt-Text. Keine Tabellen, keine Trennlinien aus Gleichheitszeichen.

Aufruf (auf dem Host oder im Container):
    python3 backend/scripts/lesefassung.py <snapshot-ordner> <PROMPT-STANDARD.md> <ziel-ordner> \
        "<Datum>" "<Schalterstand je Umgebung>"
Ergebnis: <ziel-ordner>/*.txt (ältere .txt dort werden ersetzt) und <ziel-ordner>/../<Gesamtname>.txt.

Seit 08.10.2026 im Repo (vorher nur im Arbeitsordner der Messung). Der Schalterstand wird beim
Erzeugen übergeben, weil ein Container nur seine eigene Umgebung kennt.
"""
import os
import re
import sys

ABSCHNITTE = [
    ('00_Prompt-Standard', 'die Regeln für alle Prompts.'),
    ('00_system', 'der System-Prompt, gilt für jede Beschreibung und jede Prüfung.'),
    ('01_classification', 'der Klassifikator in drei Varianten (mit Kontext, ohne Kontext, mit Nutzerhinweis).'),
    ('02_inventar', 'der Inventar-Schritt (nur im Analysemodus).'),
    ('03_beschreibung_01 bis 13', 'der Produktionsprompt je Bildtyp. Der Kopf (Rolle, Belegregeln, Arbeitsweise) ist bei '
                                  'allen gleich, der Kategorie-Teil ab BILDTYP ist das Besondere. Die Datengrafiken '
                                  '(Diagramm, Tabelle, Karte, Infografik, Screenshot) teilen den Block LESBARER TEXT UND LEGENDE.'),
    ('04_mini', 'Logo, Symbol, Bedienelement.'),
    ('05_faktenblatt, 05_werte und 05_zaehl', 'die Zusatzschritte, je Aufruf und angehängter Block.'),
    ('06_pruefpass', 'der unabhängige Prüfer.'),
    ('07_anbieter_zusatz', 'der kurze Zusatz für Gemini am Ende des Produktionsprompts.'),
    ('08_zusatz', 'was an jeden Beschreibungs-Prompt angehängt werden kann: Neu generieren (bisheriger Text von der KI, '
                  'vom Nutzer bearbeitet, vom Nutzer selbst geschrieben), Ausgabesprache, eigene Vorgaben des Nutzers.'),
]


def main(snap, standard, ziel, datum, schalter, neu=''):
    os.makedirs(ziel, exist_ok=True)
    for alt in os.listdir(ziel):
        if alt.endswith('.txt'):
            os.remove(os.path.join(ziel, alt))
    stuecke = []
    std = open(standard, encoding='utf-8').read()
    std = re.sub(r'^#+\s*', '', std, flags=re.M).replace('`', '')
    stuecke.append(('00_Prompt-Standard', 'Prompt-Standard', std.strip() + '\n'))
    for name in sorted(os.listdir(snap)):
        if not name.endswith('.md') or name == 'README.md':
            continue
        text = open(os.path.join(snap, name), encoding='utf-8').read()
        titel = text.splitlines()[0].lstrip('# ').strip()
        m = re.search(r'```text\n(.*?)```', text, re.S)
        stuecke.append((name[:-3], titel, (m.group(1) if m else text).rstrip() + '\n'))
    for datei, titel, inhalt in stuecke:
        with open(os.path.join(ziel, datei + '.txt'), 'w', encoding='utf-8') as f:
            f.write(f'{titel}\n\n{inhalt}')
    reihenfolge = '\n'.join(f'{a}: {b}' for a, b in ABSCHNITTE)
    lies = (f'InkluDocs Prompts, Lesefassung vom {datum}\n\n'
            'Dieser Ordner enthält jeden Prompt genau so, wie das Sprachmodell ihn im Betrieb bekommt, gerendert mit '
            'erfundenen Beispieldaten. Dazu der Prompt-Standard, nach dem alle Prompts geschrieben sind.\n\n'
            f'Reihenfolge:\n{reihenfolge}\n\n'
            f'Welche Zusatzschritte wo eingeschaltet sind:\n{schalter}\n\n'
            + (f'{neu}\n\n' if neu else '')
            + 'Die Dateien werden nach jeder Änderung neu erzeugt. Handschriftliche Änderungen hier gehen verloren; '
              'Änderungswünsche bitte an Claude.\n')
    open(os.path.join(ziel, 'LIES-MICH.txt'), 'w', encoding='utf-8').write(lies)
    gesamt = os.path.join(os.path.dirname(os.path.abspath(ziel)), os.path.basename(os.path.abspath(ziel)) + '.txt')
    with open(gesamt, 'w', encoding='utf-8') as f:
        f.write(lies + '\n\n')
        for datei, titel, inhalt in stuecke:
            f.write(f'Abschnitt {datei}: {titel}\n\n{inhalt}\n\n')
    print('Lesefassung:', len(stuecke) + 1, 'Dateien in', ziel, '- Gesamtdatei', gesamt)


if __name__ == '__main__':
    main(*sys.argv[1:7])
