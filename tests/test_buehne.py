"""Buehnen-Vertrag (Live-Fenster „zum Anklemmen“, 17.10.2026; Steve: „ohne das ganze Programm umbauen zu müssen“).

Die Seite draussen (inkluagent.js) spricht mit der Seite im Arbeitsfenster NUR ueber window.InkluBuehne (Kern: dashboard.js
buehneEinrichten). Jede Seite sagt selbst, wie sie ihre Stellen findet (app.html buehneStelleFinden, ablage.html). Dieser Test haelt
die Teile zusammen, damit eine geaenderte Karte die Kamerafahrt nicht still bricht:
- jede Kennung, die ein Finder sucht, erzeugt die Ansicht, zu der sie gehoert, wirklich — mit demselben Schluessel (Bild-, Feld-,
  Dokument-Kennung; Seite = Dokument_Seite), und die Klappen (details), die der Finder oeffnet, gibt es;
- die Arten der Stellen passen zu denen, die das Laufbuch meldet; jede Ansicht der Server-Listen gibt es in der Projektseite;
- der Kern bewegt nur in der Seite im Fenster (scrollTo, kein scrollIntoView), achtet auf weniger Bewegung, nimmt keinen Fokus, sagt
  nichts an, schreibt kein HTML; ein Beobachter stoert die Seite nie; die Seite draussen ruft nur, was der Kern anbietet.
Nur Quelltext, keine Datenbank.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_buehne.py
"""
import os
import re
import sys
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import laufbuch  # noqa: E402
from inkluagent import arbeitsfenster  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
FRONTEND = next(k for k in (os.path.normpath(os.path.join(BACKEND, "..", "frontend")), os.path.join(BACKEND, "frontend"))
                if os.path.isdir(k))
TEMPLATES = os.path.join(BACKEND, "templates")

# Kennung im Finder -> die Quellen der Ansicht(en), die sie erzeugen. Eine neue Kennung im Finder braucht hier einen Eintrag.
ERZEUGER = {
    "img_heading_": ("app.html",), "imgcard_": ("app.html",), "alttext_": ("app.html",), "page_heading_": ("app.html",),
    "doc_heading_": ("app.html", "formular.js", "uebersetzen.js"),
    "feld_heading_": ("formular.js",), "feldcard_": ("formular.js",), "quickinfo_": ("formular.js",), "feld_page_": ("formular.js",),
    "dok_heading_": ("dokument.js",), "dok_karte_": ("dokument.js",), "dok_badge_": ("dokument.js",),
    "ab_heading_": ("abschluss.js",), "ab_karte_": ("abschluss.js",), "ab_badge_": ("abschluss.js",),
}


def lies(*teile):
    with open(os.path.join(*teile), encoding="utf-8") as f:
        return f.read()


def quelle(datei):
    return lies(TEMPLATES, datei) if datei.endswith(".html") else lies(FRONTEND, datei)


def funktion(text, kopf, ende="\n}\n"):
    t = text[text.index(kopf):]
    return t[:t.index(ende)]


class Vertrag(unittest.TestCase):
    def setUp(self):
        self.app = lies(TEMPLATES, "app.html")
        self.finder = funktion(self.app, "function buehneStelleFinden(", "\n    }\n")

    def test_jede_kennung_gibt_es_in_ihrer_ansicht(self):
        gesucht = set(re.findall(r"el\('([a-z_]+_)'", self.finder))
        self.assertEqual(gesucht, set(ERZEUGER), "neue oder entfernte Kennung im Finder: ERZEUGER nachziehen")
        for kennung, dateien in ERZEUGER.items():
            for datei in dateien:
                self.assertRegex(quelle(datei), r"id\s*=\s*\\?[\"']" + re.escape(kennung), f"{kennung} fehlt in {datei}")

    def test_dieselben_schluessel(self):
        app = self.app
        # Bilder: Kennung des Bildes; Seiten: Dokument_Seite (ohne Dokument 0); Dokumente: documents.id
        self.assertIn('id="img_heading_${img.id}"', app)
        self.assertIn('id="imgcard_${img.id}"', app)
        self.assertIn('id="alttext_${img.id}"', app)
        self.assertIn("const pageId = `${docKey}_${pageNum}`;", app)
        self.assertIn('id="page_heading_${pageId}"', app)
        self.assertIn("el('page_heading_' + d + '_' + s.seite)", self.finder)
        f = quelle("formular.js")
        self.assertIn("const key = docKey + '_' + pageNum;", f)
        self.assertIn("id=\"feld_page_' + key", f)
        self.assertIn("el('feld_page_' + d + '_' + s.seite)", self.finder)
        for datei in ("formular.js", "uebersetzen.js"):
            self.assertIn("const docKey = doc.id;", quelle(datei), datei)
        for kennung in ("feld_heading_", "feldcard_", "quickinfo_"):
            self.assertIn(f"id=\"{kennung}' + f.id", f, kennung)
        for datei, vorne in (("dokument.js", "dok_"), ("abschluss.js", "ab_")):
            q = quelle(datei)
            for teil in ("heading_", "karte_", "badge_"):
                self.assertIn(f"id=\"{vorne}{teil}' + d.id", q, datei + teil)
        self.assertIn("li.id = 'ausgabe-' + a.id;", lies(TEMPLATES, "ablage.html"))

    def test_klappen_die_der_finder_oeffnet(self):
        klassen = set(re.findall(r"closest\('details\.([a-z-]+)'\)", self.finder))
        self.assertEqual(klassen, {"img-section", "page-section", "doc-section"})
        alle = "".join(quelle(d) for d in ("app.html", "formular.js", "uebersetzen.js"))
        for k in klassen:
            self.assertRegex(alle, r"<details class=\\?\"[^\"']*\b" + re.escape(k) + r"\b", k)

    def test_arten_wie_das_laufbuch(self):
        arten_finder = set(re.findall(r"s\.art === '([a-z]+)'", self.finder))
        self.assertEqual(arten_finder, {"bild", "feld", "dokument", "seite", "ansicht"})
        lb = lies(BACKEND, "laufbuch.py")
        for art in arten_finder - {"ansicht"} | {"ausgabe"}:
            self.assertIn(f'neu["art"] = "{art}"', lb, art)
        # „ansicht“ (eine Station ohne einzelne Stelle) setzt der Server selbst; die Stelle ist die Ueberschrift 1 jeder Projektansicht
        self.assertIn('{"art": "ansicht", "project_id": pid, "ansicht": ansicht}', lies(BACKEND, "inkluagent", "arbeitsfenster.py"))
        self.assertIn("const kopf = el('projectName');", self.finder)
        for datei in ("app.html", "formular.js", "uebersetzen.js"):
            self.assertIn('<h1 id="projectName"', quelle(datei), datei)
        self.assertIn("s.art !== 'ausgabe'", lies(TEMPLATES, "ablage.html"), "die Ablage findet Ergebnisse")

    def test_jede_ansicht_gibt_es_in_der_projektseite(self):
        listen = funktion(self.app, "function ansichtenFuer(", "\n    }\n")
        ansichten = set(re.findall(r"'([a-z]+)'", "".join(re.findall(r"return \[([^\]]+)\];", listen))))
        self.assertEqual(ansichten, {"dokument", "tagging", "alttexte", "quickinfos", "uebersetzung", "abschluss"})
        server = set(arbeitsfenster.ANSICHT_JE_WERKZEUG.values()) | set(arbeitsfenster.ANSICHT_JE_LAUF.values())
        self.assertLessEqual(server, ansichten, server - ansichten)
        self.assertLessEqual(set(re.findall(r"ansicht: '([a-z]+)'", self.finder)), ansichten)
        self.assertEqual(set(arbeitsfenster.ANSICHT_JE_LAUF), set(laufbuch.ARTEN))

    def test_finder_nur_dieses_projekt_und_ganze_zahlen(self):
        self.assertIn("if (s.project_id != null && String(s.project_id) !== String(hier)) return null;", self.finder)
        self.assertIn(r"const zahl = (x) => (/^\d+$/.test(String(x == null ? '' : x)) ? String(x) : null);", self.finder)
        for art, feld in (("bild", "image_id"), ("feld", "feld_id")):
            self.assertIn(f"if (s.art === '{art}' && zahl(s.{feld}))", self.finder)
        self.assertIn("&& zahl(s.document_id))", self.finder)
        self.assertNotIn("innerHTML", self.finder)
        self.assertNotIn("querySelector", self.finder, "nur Kennungen, keine freien Selektoren")


class Kern(unittest.TestCase):
    def setUp(self):
        self.dash = lies(FRONTEND, "dashboard.js")
        self.kern = funktion(self.dash, "function buehneEinrichten(")

    def test_nur_im_rahmen_und_feste_schnittstelle(self):
        self.assertIn("if (!imRahmen() || !seite || typeof seite.finden !== 'function') return null;", self.kern)
        self.assertIn("return { version: 1, zeigen, neuZeichnen, lesen, titel, beobachten,", self.kern)
        self.assertIn("window.buehneEinrichten = buehneEinrichten;", self.dash)

    def test_bewegt_nur_hier_und_ruhig(self):
        self.assertNotIn("scrollIntoView", self.kern, "scrollIntoView rollt auch die Seite draussen")
        self.assertIn("window.scrollTo({ top: oben, behavior: weich ? 'smooth' : 'auto' })", self.kern)
        self.assertIn("window.matchMedia('(prefers-reduced-motion: reduce)').matches", self.kern)
        self.assertIn("rollen(f.karte, !ruhig() && opts.weich !== false);", self.kern)
        self.assertIn("if (Date.now() < zustand.halten && !document.hasFocus()) {", self.kern, "wer in der Seite arbeitet, wird nicht gerollt")
        self.assertIn("const lauf = ++zustand.lauf;", self.kern, "nur die juengste Fahrt zaehlt")

    def test_kein_fokus_keine_ansage_kein_html(self):
        for verboten in ("announce(", ".focus(", "innerHTML", "insertAdjacentHTML"):
            self.assertNotIn(verboten, self.kern, verboten)
        a = funktion(self.dash, "function announce(")
        erste = a.split("{", 1)[1].strip().splitlines()
        code = [z for z in erste if not z.strip().startswith("//")][0]
        self.assertIn("if (typeof imRahmen === 'function' && imRahmen() && typeof document !== 'undefined' && !document.hasFocus()) return;",
                      code, "die Seite im Fenster sagt nur an, wenn jemand in ihr arbeitet")

    def test_ein_beobachter_stoert_die_seite_nie(self):
        self.assertIn("zustand.hoerer.forEach(fn => { try { fn(daten); } catch (e)", self.kern)
        self.assertIn("}).observe(main, { childList: true, subtree: true, characterData: true });", self.kern)

    def test_seiten_melden_sich_an(self):
        app = lies(TEMPLATES, "app.html")
        self.assertIn("if (window.EINGEBETTET === true && typeof buehneEinrichten === 'function') {\n        window.InkluBuehne = buehneEinrichten({\n"
                      "            finden: buehneStelleFinden,", app)
        w = funktion(app, "async ansichtWechseln(ziel) {", "\n            },\n")
        self.assertIn("if (!p || !verfuegbareAnsichten(p).includes(ziel)) return false;", w, "eine Ansicht, die das Projekt nicht kennt")
        self.assertIn("if (await alleAusstehendenSpeichern()) return false;", w, "keine Eingabe geht verloren")
        self.assertIn("history.replaceState(history.state, '', u.toString());", w)
        self.assertIn("await showProject(p.id, true);", w)
        abl = lies(TEMPLATES, "ablage.html")
        self.assertIn("window.InkluBuehne = buehneEinrichten({", abl)
        self.assertIn("neuZeichnen: () => laden(),", abl)

    def test_draussen_nur_was_der_kern_anbietet(self):
        from inkluagent import skripte   # alle Teile des Agenten-Skripts (17.10.2026)
        js = skripte.quelltext(FRONTEND)
        angeboten = set(re.findall(r"\b(\w+)\b", self.kern[self.kern.rindex("return { version: 1,"):]))
        for methode in ("zeigen", "neuZeichnen", "beobachten"):
            self.assertIn(methode, angeboten)
        benutzt = set(re.findall(r"\bb\.(\w+)\(", funktion(js, "function inkluagentFensterFolgen(")
                                 + funktion(js, "function inkluagentRahmenNachziehen(") + funktion(js, "function inkluagentBuehneGeladen(")))
        self.assertTrue(benutzt, "die Seite draussen ruft die Buehne")
        self.assertLessEqual(benutzt, angeboten, benutzt - angeboten)


if __name__ == "__main__":
    unittest.main()
