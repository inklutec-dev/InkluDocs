"""Konformitaetstest beider Wege (15.09.2026): Jedes Korpus-Dokument laeuft einmal ueber PDFix und einmal ueber den
Ersatzweg (fitz) — Extraktion, feste Testtexte, Export, Export-Abnahme — und die Messwerte werden gegen die
Messlatte (messlatte.json) verglichen. Kein Kundendokument, keine KI, keine Credits.

    docker exec -w /app inkludocs-staging python3 tests/korpus/konformitaet.py                 # pruefen
    docker exec -w /app inkludocs-staging python3 tests/korpus/konformitaet.py --messlatte     # Messlatte neu schreiben

Exit 1, wenn ein Weg die Abnahme nicht besteht oder ein Messwert von der Messlatte abweicht.
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(os.path.dirname(HERE)), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import pdf_export  # noqa: E402
import pdf_processor  # noqa: E402
import pdfix_roundtrip  # noqa: E402
from export_abnahme import abnahme_pdf  # noqa: E402

DOKUMENTE = os.path.join(HERE, "dokumente")
MESSLATTE = os.path.join(HERE, "messlatte.json")
VERGLEICH = ("bilder", "abnahme_ok", "figures_mit_alt", "texte_gefunden", "waisen", "unbalanciert",
             "ruecksprung", "verapdf_neu", "verapdf_schlechter", "warnungen")


def _texte(images):
    return {i: f"Testtext Bild {n} (fiktiv, Korpus)" for n, i in enumerate(range(len(images)), start=1)}


def weg_fitz(pdf, work):
    os.environ["PDFIX_ENABLED"] = "false"
    os.makedirs(os.path.join(work, "bilder"), exist_ok=True)
    images = pdf_processor.extract_images_from_pdf(pdf, os.path.join(work, "bilder"), 0)
    texte = _texte(images)
    def _bb(im):
        b = im.get("bbox") or (None, None, None, None)
        return {"bbox_x0": b[0], "bbox_y0": b[1], "bbox_x1": b[2], "bbox_y1": b[3]}
    rows = [{"xref": im["xref"], "page_number": im["page_number"], "is_vector": im.get("is_vector"), **_bb(im)}
            for im in images]
    layout = pdf_export.layout_vektorbilder(pdf, rows)
    alt_texts, meta = {}, []
    for n, im in enumerate(images):
        if im["xref"] in layout:
            continue
        alt_texts[im["xref"]] = texte[n]
        meta.append({"xref": im["xref"], "page_number": im["page_number"], "is_vector": bool(im.get("is_vector")),
                     "bbox": im.get("bbox"), "alt_text": texte[n], "image_path": im["image_path"]})
    out = os.path.join(work, "export_fitz.pdf")
    r = pdf_export.write_alt_texts_to_pdf(pdf, out, alt_texts, meta)
    pdf_export.finalize_export_pdf(out, title="Korpus", verfahren="fitz", schonen=set(r.get("figure_xrefs") or []))
    a = abnahme_pdf(out, pdf, list(alt_texts.values()), erwartet_getaggt=r.get("tagged_count"))
    return _mess(len(images), a, len(r.get("warnings") or []), layout=len(layout), tagged=r.get("tagged_count"))


def weg_pdfix(pdf, work):
    if not (pdfix_roundtrip.is_pdfix_available() and pdfix_roundtrip.is_tagged_pdf(pdf)):
        return None
    os.environ["PDFIX_ENABLED"] = "true"
    os.makedirs(os.path.join(work, "pdfix"), exist_ok=True)
    images = pdf_processor._extract_via_pdfix(pdf, os.path.join(work, "pdfix"))
    texte = _texte(images)
    by_lfnr = {im["image_index"]: texte[n] for n, im in enumerate(images)}
    out = os.path.join(work, "export_pdfix.pdf")
    count = pdfix_roundtrip.import_alt_texts_pdfix(pdf, out, by_lfnr, work_dir=work)
    pdf_export.finalize_export_pdf(out, title="Korpus", verfahren="pdfix")
    a = abnahme_pdf(out, pdf, list(by_lfnr.values()), erwartet_getaggt=count)
    return _mess(len(images), a, 0, tagged=count)


def _mess(bilder, a, warnungen, layout=0, tagged=None):
    kz = a["kennzahlen"]
    return {"bilder": bilder, "layoutbereiche": layout, "getaggt": tagged, "abnahme_ok": a["ok"], "befunde": a["befunde"],
            "figures_mit_alt": kz.get("figures_mit_alt"), "texte_gefunden": kz.get("texte_gefunden"),
            "waisen": kz.get("waisen"), "unbalanciert": kz.get("seiten_unbalanciert"),
            "ruecksprung": kz.get("ruecksprünge_gross"), "verapdf": kz.get("verapdf"),
            "verapdf_neu": 0, "verapdf_schlechter": 0, "warnungen": warnungen}


def _verapdf_zaehlen(m):
    if not m:
        return
    for b in m["befunde"]:
        if b.startswith("veraPDF: neue"):
            m["verapdf_neu"] = b.count("(")
        if b.startswith("veraPDF: Regel"):
            m["verapdf_schlechter"] = b.count("->")


def main():
    schreiben = "--messlatte" in sys.argv
    latte = json.load(open(MESSLATTE)) if os.path.exists(MESSLATTE) else {}
    ergebnis = {}
    fehler = []
    for name in sorted(os.listdir(DOKUMENTE)):
        if not name.endswith(".pdf"):
            continue
        pdf = os.path.join(DOKUMENTE, name)
        work = tempfile.mkdtemp(prefix="korpus_")
        try:
            ef = weg_fitz(pdf, work)
            ep = weg_pdfix(pdf, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        _verapdf_zaehlen(ef); _verapdf_zaehlen(ep)
        ergebnis[name] = {"fitz": ef, "pdfix": ep}
        for weg, m in (("fitz", ef), ("pdfix", ep)):
            if m is None:
                print(f"{name:26s} {weg:6s} —  (nicht anwendbar)")
                continue
            print(f"{name:26s} {weg:6s} bilder={m['bilder']} layout={m['layoutbereiche']} getaggt={m['getaggt']} "
                  f"alt={m['figures_mit_alt']} texte={m['texte_gefunden']} waisen={m['waisen']} unbal={m['unbalanciert']} "
                  f"rueck={m['ruecksprung']} verapdf={m['verapdf']} warn={m['warnungen']} abnahme={'ok' if m['abnahme_ok'] else 'FEHLT: ' + '; '.join(m['befunde'])}")
            if not m["abnahme_ok"]:
                fehler.append(f"{name} {weg}: Abnahme nicht bestanden: {'; '.join(m['befunde'])}")
            soll = (latte.get(name) or {}).get(weg)
            if soll and not schreiben:
                for k in VERGLEICH:
                    if soll.get(k) != m.get(k):
                        fehler.append(f"{name} {weg}: {k} = {m.get(k)}, Messlatte {soll.get(k)}")
        # Beide Wege, gleicher Anspruch (Steve 15.09.2026): der Ersatzweg muss so viele Texte in die Datei bringen
        # wie PDFix und darf nicht mehr „Bilder“ anbieten (= berechnen) als PDFix Figures findet (+2 Toleranz fuer
        # ungetaggte Bilder, die PDFix bewusst nicht sieht).
        if ef and ep:
            if (ef["texte_gefunden"] or 0) < (ep["texte_gefunden"] or 0):
                fehler.append(f"{name}: Ersatzweg bringt {ef['texte_gefunden']} Texte in die Datei, PDFix {ep['texte_gefunden']}")
            if ef["bilder"] > ep["bilder"] + 2:
                fehler.append(f"{name}: Ersatzweg bietet {ef['bilder']} Bilder an (= Berechnung), PDFix findet {ep['bilder']}")
    if schreiben:
        json.dump({n: {w: ({k: m.get(k) for k in VERGLEICH} if m else None) for w, m in e.items()} for n, e in ergebnis.items()},
                  open(MESSLATTE, "w"), indent=1, ensure_ascii=False)
        print("Messlatte geschrieben:", MESSLATTE)
    print()
    if fehler:
        print("FEHLER:"); [print(" -", f) for f in fehler]; sys.exit(1)
    print("Konformitaet: alle Dokumente auf beiden Wegen sauber")


if __name__ == "__main__":
    main()
