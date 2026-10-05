#!/usr/bin/env python3
"""tests/fixtures/make_messpdfs.py — erzeugt 7 klar fiktive, getaggte Mess-PDFs (Chromium, tagged=True), die Michaels Projekt 957
in Bildzahl, Seitenzahl und Kontextlaenge nachbilden (Messreihe Ansichtswechsel 05.10.2026).

Docs 4-7 haben KEINE Ueberschriften -> PDFix-Kapitelkontext = ganzes Dokument (wie bei 957).
Docs 1-3 haben je Bild eine Ueberschrift -> kurzer Kontext.
Alle Namen, Orte, Zahlen sind erfunden (Musterstadt, Beispielhausen, Musterbach)."""
import base64
import io
import os
import random
import sys

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/messpdfs"
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(20261005)

ORTE = ["Musterstadt", "Beispielhausen", "Musterdorf", "Probehagen", "Testingen", "Fiktivau"]
GEWAESSER = ["Musterbach", "Beispielfluss", "Probesee", "Testgraben"]
THEMEN = ["Bodenschutz", "Abfallwirtschaft", "Hitzevorsorge", "Strahlenschutz", "Gewässergüte", "Luftreinhaltung",
          "Flächenverbrauch", "Grundwasser", "Lärmminderung", "Stadtgrün"]
SAETZE = [
    "Im Berichtsjahr hat die Stadtverwaltung von {ort} insgesamt {n} Messstellen ausgewertet und die Ergebnisse mit dem Vorjahr verglichen.",
    "Der Anteil versiegelter Flächen stieg in {ort} um {p} Prozent, vor allem durch neue Gewerbegebiete am Stadtrand.",
    "Die Proben aus dem {gew} zeigen eine leichte Verbesserung, die Nitratwerte lagen im Mittel bei {n} Milligramm je Liter.",
    "Für das Thema {thema} wurden im Haushalt {n} Tausend Euro bereitgestellt, davon floss etwa ein Drittel in Beratung.",
    "Bürgerinnen und Bürger konnten sich an {n} Informationsabenden beteiligen und eigene Vorschläge einbringen.",
    "Die Auswertung zeigt, dass Hitzetage in den letzten zehn Jahren deutlich häufiger geworden sind; im Schnitt waren es {n} Tage im Jahr.",
    "Besonders betroffen sind dicht bebaute Quartiere ohne Bäume, in denen sich die Luft nachts kaum abkühlt.",
    "Das Programm „{thema} vor Ort“ unterstützt Vereine, Schulen und Kitas mit Material und kurzen Schulungen.",
    "Im Vergleich der Ortsteile schneidet {ort} am besten ab, während in {ort2} noch Nachholbedarf besteht.",
    "Die Abbildung zeigt die Entwicklung der Messwerte seit dem Jahr {jahr}; die gestrichelte Linie markiert den Zielwert.",
    "Zur Einordnung: Ein Wert von {n} Einheiten entspricht ungefähr dem Durchschnitt vergleichbarer Mittelstädte.",
    "Die Sammelquote für Bioabfall lag bei {p} Prozent und damit erstmals über dem Ziel des Abfallwirtschaftsplans.",
    "Altglas, Papier und Leichtverpackungen werden weiterhin getrennt erfasst; die Restmüllmenge sank auf {n} Kilogramm je Kopf.",
    "Bei der Sanierung von Altlasten wurden {n} Verdachtsflächen untersucht, von denen {n2} saniert werden müssen.",
    "Die Messwerte der Strahlenschutz-Station in {ort} lagen durchgehend im Bereich der natürlichen Hintergrundstrahlung.",
    "Im Rückblick auf das Jahr {jahr} zeigt sich, wie wichtig verlässliche Daten für gute Entscheidungen sind.",
    "Die Stadt plant, bis zum Jahr {jahr2} weitere {n} Bäume zu pflanzen und Dachbegrünung stärker zu fördern.",
    "Ein Teil der Maßnahmen ist bereits umgesetzt, andere befinden sich noch in der Abstimmung mit den Fachämtern.",
    "Die Tabelle im Anhang listet alle Messstellen mit Lage, Messbeginn und den wichtigsten Kennwerten auf.",
    "Für Rückfragen steht das Umweltamt der Stadt {ort} zur Verfügung; die Kontaktdaten finden Sie auf der letzten Seite.",
    "Hinweis: Alle Angaben in diesem Bericht sind fiktiv und dienen ausschließlich als Testmaterial.",
    "Nach Einschätzung des Fachbeirats reicht das bisherige Tempo nicht aus, um die Ziele für {thema} zu erreichen.",
    "Deshalb schlägt der Beirat vor, die Mittel ab dem Jahr {jahr2} schrittweise um {p} Prozent zu erhöhen.",
    "Die Grundwasserstände im Umfeld von {ort2} sanken in trockenen Sommern um bis zu {n} Zentimeter.",
]


def satz():
    s = rnd.choice(SAETZE)
    return s.format(ort=rnd.choice(ORTE), ort2=rnd.choice(ORTE), gew=rnd.choice(GEWAESSER), thema=rnd.choice(THEMEN),
                    n=rnd.randint(3, 480), n2=rnd.randint(1, 40), p=rnd.randint(2, 68), jahr=rnd.randint(2010, 2024),
                    jahr2=rnd.randint(2027, 2035))


def absatz(ziel_zeichen):
    out = []
    laenge = 0
    while laenge < ziel_zeichen:
        s = satz()
        out.append(s)
        laenge += len(s) + 1
    return " ".join(out)


def bild_png(nr, w=520, h=300):
    im = Image.new("RGB", (w, h), (250, 250, 247))
    d = ImageDraw.Draw(im)
    farben = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
    n = rnd.randint(4, 9)
    breite = (w - 60) // n
    for i in range(n):
        hh = rnd.randint(30, h - 60)
        d.rectangle([40 + i * breite + 6, h - 30 - hh, 40 + (i + 1) * breite - 6, h - 30], fill=farben[i % len(farben)])
    d.line([40, h - 30, w - 10, h - 30], fill=(60, 60, 60), width=2)
    d.line([40, 10, 40, h - 30], fill=(60, 60, 60), width=2)
    d.text((50, 12), f"Abb. {nr} (fiktive Messwerte)", fill=(20, 20, 20))
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# (Dateiname, Titel, Bilder, Text gesamt (Zeichen), mit Ueberschriften je Bild)
DOKS = [
    ("fiktiv_01_merkblatt_musterstadt.pdf", "Merkblatt Musterstadt (fiktiv)", 1, 600, True),
    ("fiktiv_02_jubilaeum_beispielhausen.pdf", "Jubiläumsheft Beispielhausen (fiktiv)", 3, 3000, True),
    ("fiktiv_03_aktionsplan_musterdorf.pdf", "Aktionsplan Musterdorf (fiktiv)", 5, 2500, True),
    ("fiktiv_04_hitzeratgeber_probehagen.pdf", "Hitzeratgeber Probehagen (fiktiv)", 34, 15000, False),
    ("fiktiv_05_bodenbericht_testingen.pdf", "Bodenbericht Testingen (fiktiv)", 115, 276000, False),
    ("fiktiv_06_abfallbilanz_fiktivau.pdf", "Abfallbilanz Fiktivau (fiktiv)", 63, 40000, False),
    ("fiktiv_07_strahlenschutz_musterbach.pdf", "Strahlenschutzbericht Musterbach (fiktiv)", 48, 78000, False),
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8.6pt; line-height: 1.32; color: #111; }
p { margin: 0 0 5pt 0; text-align: justify; }
img { width: 62mm; height: auto; display: block; margin: 4pt 0 6pt 0; }
h2 { font-size: 13pt; margin: 8pt 0 4pt 0; }
"""


def html_fuer(titel, bilder, zeichen, mit_ue):
    teile = [f"<!doctype html><html lang='de'><head><meta charset='utf-8'><title>{titel}</title><style>{CSS}</style></head><body>"]
    if mit_ue:
        teile.append(f"<p><strong>{titel}</strong></p>")
        je = max(80, zeichen // max(bilder, 1))
        for i in range(bilder):
            teile.append(f"<h2>Abschnitt {i + 1}</h2>")
            teile.append(f"<p>{absatz(je)}</p>")
            teile.append(f"<img src='data:image/png;base64,{bild_png(i + 1)}' alt='Säulendiagramm mit fiktiven Messwerten, Abbildung {i + 1}'>")
        teile.append("</body></html>")
        return "".join(teile)
    # Ohne Ueberschriften: Titel nur als fetter Absatz, Text gleichmaessig zwischen den Bildern
    teile.append(f"<p><strong>{titel}</strong></p>")
    je_block = zeichen // (bilder + 1)
    for i in range(bilder):
        rest = je_block
        while rest > 0:
            stueck = min(rest, 900)
            teile.append(f"<p>{absatz(stueck)}</p>")
            rest -= stueck
        teile.append(f"<img src='data:image/png;base64,{bild_png(i + 1)}' alt='Säulendiagramm mit fiktiven Messwerten, Abbildung {i + 1}'>")
    teile.append(f"<p>{absatz(je_block)}</p></body></html>")
    return "".join(teile)


with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page()
    for name, titel, bilder, zeichen, mit_ue in DOKS:
        pg.set_content(html_fuer(titel, bilder, zeichen, mit_ue), wait_until="load")
        ziel = os.path.join(OUT, name)
        pg.pdf(path=ziel, format="A4", tagged=True, print_background=True)
        print(name, os.path.getsize(ziel))
    br.close()
