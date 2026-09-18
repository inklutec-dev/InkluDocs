"""Fixture (FIKTIV): Vortrag mit gemischter Formatierung fuer das Uebersetzen-Werkzeug.
Fettung/Kursiv mitten im Satz, Hyperlink, nummerierte Liste, Tabelle mit Kopfzeile,
Fussnote-aehnlicher Verweis, Zahlenabsatz, Kopfzeile. Alle Namen erfunden."""
import sys
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def hyperlink(paragraph, url, text):
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), r_id)
    r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u); r.append(rpr)
    t = OxmlElement("w:t"); t.text = text; r.append(t); h.append(r); paragraph._p.append(h)

def erzeuge(pfad):
    d = Document()
    d.core_properties.title = "Barrierefreie Dokumente im Alltag (fiktiver Vortrag)"
    d.sections[0].header.paragraphs[0].text = "Musterverein Beispielstadt (fiktiv) – Vortragsreihe 2026"
    d.add_heading("Barrierefreie Dokumente im Alltag", 0)
    d.add_heading("1 Warum das Thema alle betrifft", 1)
    p = d.add_paragraph("Rund ")
    p.add_run("zehn Prozent").bold = True
    p.add_run(" der Bevölkerung sind auf barrierefreie Dokumente angewiesen. Das ist ")
    p.add_run("keine Randgruppe").italic = True
    p.add_run(", sondern jeder zehnte Leser.")
    p = d.add_paragraph("Weitere Informationen stehen unter ")
    hyperlink(p, "https://www.beispiel-verein.example/barrierefrei", "www.beispiel-verein.example")
    p.add_run(" bereit.")
    d.add_heading("1.1 Drei typische Fehler", 2)
    for s in ("Überschriften sind nur fett formatiert, nicht als Überschrift ausgezeichnet.",
              "Bilder haben keinen Alternativtext.",
              "Tabellen haben keine Kopfzeile."):
        d.add_paragraph(s, style="List Number")
    d.add_heading("2 Zahlen aus dem Musterverein", 1)
    t = d.add_table(rows=3, cols=3); t.style = "Table Grid"
    for i, h in enumerate(("Jahr", "Geprüfte Dokumente", "Davon barrierefrei")): t.cell(0, i).text = h
    for r, row in enumerate((("2024", "120", "38 %"), ("2025", "310", "71 %")), start=1):
        for c, v in enumerate(row): t.cell(r, c).text = v
    d.add_paragraph("2026-09-18")
    p = d.add_paragraph("Fazit: Wer ")
    p.add_run("heute").bold = True
    p.add_run(" anfängt, hat ")
    r = p.add_run("morgen"); r.bold = True; r.italic = True
    p.add_run(" weniger Arbeit. Kontakt: vortrag@beispiel-verein.example, Telefon 0000 123456 (fiktiv).")
    d.save(pfad)

if __name__ == "__main__":
    erzeuge(sys.argv[1] if len(sys.argv) > 1 else "testvortrag_inkludocs.docx")
