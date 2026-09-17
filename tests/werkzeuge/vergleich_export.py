# Alt vs. Neu: Heines Export auf denselben Formularen, Reihenfolge und Zaehler vergleichen (im Container).
import csv, os, subprocess, sys
ALT, NEU = "/app/pdfix_scripts", "/tmp/pdfix_neu"
def lauf(skriptdir, pdf, out):
    r = subprocess.run([sys.executable, os.path.join(skriptdir, "Formular_Export_Quickinfo.py"), "-i", pdf, "-c", out],
                       capture_output=True, text=True, cwd=skriptdir, timeout=120)
    if r.returncode != 0 or not os.path.exists(out):
        print("FEHLER", skriptdir, r.returncode, r.stderr[-400:]); return None
    zeilen = [z for z in csv.reader(open(out, encoding="utf-8"), delimiter=";") if z and z[0] != "Nummer"]
    return zeilen, r.stdout
for pdf in sys.argv[1:]:
    a = lauf(ALT, pdf, "/tmp/alt.csv"); n = lauf(NEU, pdf, "/tmp/neu.csv")
    if not a or not n: continue
    za, zn = a[0], n[0]
    print(f"\n== {os.path.basename(pdf)}: alt {len(za)} Felder, neu {len(zn)} Felder, Spalten alt {len(za[0])} / neu {len(zn[0])}")
    print("   FIELDS_FOUND neu:", [l for l in n[1].splitlines() if l.startswith("FIELDS_FOUND")], "| Werte maskiert:", set(z[5] for z in zn))
    print("   Nummern fortlaufend:", [int(z[0]) for z in zn] == list(range(1, len(zn)+1)))
    print("   alte Reihenfolge:", [z[1][:14] for z in za][:12])
    print("   neue Reihenfolge:", [z[1][:14] for z in zn][:12])
    print("   neu (Seite, top, left):", [(z[6], z[10], z[7]) for z in zn][:12])
    print("   gleiche Namensmenge:", sorted(z[1] for z in za) == sorted(z[1] for z in zn))
