# =============================================================================
#  Korrektur_Anwenden.py — InkluDocs: Tag-Rollen in einer getaggten PDF aendern (22.09.2026)
# =============================================================================
#  EIGENES Skript von InkluDocs (kein Skript von Joerg Heine). Fuehrt Befunde der automatischen
#  Pruefung aus, die den Doppelbeleg tragen (Modell + Messung): je Element eine neue Rolle
#  (z. B. P -> H2, H3 -> P, TH -> TD). Elemente werden ueber die Objektnummer ihres
#  Struktur-Woerterbuchs gefunden (Struktur_Export.py schreibt sie als "obj"); mehr als
#  SetType passiert nicht — kein Verschieben, kein Loeschen, keine Inhalte.
#
#  Aufruf: python3 Korrektur_Anwenden.py -i <pdf> -o <pdf_out> -k <korrekturen.json>
#     korrekturen.json: [{"obj": 123, "typ": "H2"}, ...]
#  Ausgabe (stdout, letzte Zeile): JSON {"angewendet": n, "nicht_gefunden": [obj...], "unveraendert": [obj...]}
#  Exit 0 = geschrieben; 2 = PDF nicht lesbar; 3 = kein Strukturbaum; 4 = Speichern fehlgeschlagen.
# =============================================================================
import argparse
import json
import sys

from pdfixsdk import *  # noqa: F401,F403

import inkludocs_betrieb as betrieb

_ERLAUBT = {"P", "H", "H1", "H2", "H3", "H4", "H5", "H6", "TH", "TD", "L", "LI", "Figure", "Caption", "Artifact"}


def _sammeln(tree, elem, out):
    try:
        out[elem.GetObject().GetId()] = elem
    except Exception:  # noqa: BLE001
        pass
    for i in range(elem.GetNumChildren()):
        if elem.GetChildType(i) == kPdsStructChildElement:
            c = tree.GetStructElementFromObject(elem.GetChildObject(i))
            if c is not None:
                _sammeln(tree, c, out)


def main():
    ap = argparse.ArgumentParser(description="Tag-Rollen aendern (InkluDocs)")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-k", "--korrekturen", required=True)
    args = ap.parse_args()
    with open(args.korrekturen, encoding="utf-8") as f:
        korrekturen = json.load(f)
    pdfix = GetPdfix()
    if pdfix is None:
        raise SystemExit("Pdfix Initialization fail")
    betrieb.lizenz_aktivieren(pdfix)
    doc = pdfix.OpenDoc(args.input, "")
    if doc is None:
        sys.exit(betrieb.pdf_nicht_geoeffnet(pdfix))
    try:
        tree = doc.GetStructTree()
        if tree is None or tree.GetNumChildren() == 0:
            print("Kein Strukturbaum", file=sys.stderr)
            sys.exit(3)
        elemente: dict = {}
        for i in range(tree.GetNumChildren()):
            e = tree.GetStructElementFromObject(tree.GetChildObject(i))
            if e is not None:
                _sammeln(tree, e, elemente)
        angewendet, fehlt, unveraendert = 0, [], []
        for k in korrekturen:
            obj = int(k.get("obj") or 0)
            typ = str(k.get("typ") or "")
            if typ not in _ERLAUBT:
                unveraendert.append(obj)
                continue
            e = elemente.get(obj)
            if e is None:
                fehlt.append(obj)
                continue
            if e.GetType(True) == typ:
                unveraendert.append(obj)
                continue
            if e.SetType(typ):
                angewendet += 1
            else:
                unveraendert.append(obj)
        if not doc.Save(args.output, kSaveFull):
            print("Speichern fehlgeschlagen", file=sys.stderr)
            sys.exit(4)
        print(json.dumps({"angewendet": angewendet, "nicht_gefunden": fehlt, "unveraendert": unveraendert}))
    finally:
        doc.Close()


if __name__ == "__main__":
    main()
