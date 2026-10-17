"""Aus der Ablage in ein Projekt (Steve 17.10.2026, backend/ablage_projekt.py): Werkzeug ausgabe_in_projekt des InkluAgent —
kreuz und quer, in Projekten und auf Seiten ohne Projekt; einen Knopf in der Ablage gibt es bewusst nicht (Steve: „brauche ich
eigentlich gar nicht“; von Hand: herunterladen und im Zielprojekt „Dokumente hinzufügen“).

Prüft: Datei des Eintrags (PDF, Word, ZIP ausgepackt; nur PDF/Word; Größe; Anzahl), neues Projekt mit Namen aus der Herkunft oder
weiteres Dokument in ein eigenes Projekt derselben Art (fremdes Projekt 404, andere Art 400), fremder Eintrag und fehlende Datei 404,
Tagesgrenze, Laufbuch nur, wenn der Agent es tut, Werkzeug (Einträge jedes eigenen Projekts, nie fremde; Seitenwechsel und Link bei
anderem Projekt, Fokus des Gesprächs; pruefen), die ganze Ablage auf Seiten ohne Projekt (lesen, laden), kein Knopf in der Ablage.
Wegwerf-Datenbank, main.upload_file durch eine Attrappe ersetzt (kein Auslesen), keine KI.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_ablage_projekt.py
"""
import asyncio
import io
import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

TMP = tempfile.mkdtemp(prefix="ablage_projekt_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_MAX_DATEIEN", "INKLUAGENT_MAX_NEUE_PROJEKTE_TAG"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

from fastapi import HTTPException  # noqa: E402

import ablage_projekt  # noqa: E402
import laufbuch  # noqa: E402
from inkluagent import hochladen  # noqa: E402

BACKEND = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


def docx() -> bytes:
    puffer = io.BytesIO()
    with zipfile.ZipFile(puffer, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "<w:document/>")
    return puffer.getvalue()


def zip_mit(**dateien) -> bytes:
    puffer = io.BytesIO()
    with zipfile.ZipFile(puffer, "w") as z:
        for name, inhalt in dateien.items():
            z.writestr(name.replace("__", "/"), inhalt)
    return puffer.getvalue()


_N = [0]


def konto() -> dict:
    _N[0] += 1
    mail = f"bruecke{_N[0]}@beispiel.invalid"
    return {"id": database.create_user(mail, "geheim-12345", "Test (fiktiv)"), "email": mail}


def projekt(uid: int, tool: str = "word", typ: str = "docx", name: str = "Vereinsbrief (fiktiv)") -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, ?, ?, 'extracted')", (uid, name, name, tool, typ)).lastrowid
    c.commit()
    c.close()
    return pid


class Basis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"main nicht ladbar: {e}")
        cls.main = main

    def setUp(self):
        laufbuch._leeren()
        self.u = konto()
        self.quelle = projekt(self.u["id"])
        self.hochgeladen = []
        mock.patch.object(self.main, "upload_file", side_effect=self._upload).start()
        self.addCleanup(mock.patch.stopall)

    async def _upload(self, file, project_id, user, request=None):
        """Wie der Knopf: ohne Projekt ein neues (nur Dateiname, ohne Namen), sonst ein Dokument dazu."""
        await file.read()
        self.hochgeladen.append((file.filename, project_id))
        c = database.get_db()
        try:
            if project_id is None:
                pdf = file.filename.lower().endswith(".pdf")
                project_id = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                                       "VALUES (?, ?, '', '', ?, ?, 'extracted')",
                                       (user["id"], file.filename, "pdf" if pdf else "word", "pdf" if pdf else "docx")).lastrowid
            n = c.execute("SELECT COUNT(*) FROM documents WHERE project_id = ?", (project_id,)).fetchone()[0]
            did = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, ?, ?, '')",
                            (project_id, n + 1, file.filename)).lastrowid
            c.commit()
        finally:
            c.close()
        return {"ok": True, "project_id": project_id, "document_id": did, "status": "extracting"}

    def eintrag(self, inhalt=PDF, dateiname="inkludocs_vereinsbrief.pdf", media="application/pdf", art="pdfua", uid=None,
                pid=None, dokument="Vereinsbrief (fiktiv)") -> int:
        uid = uid or self.u["id"]
        pfad = os.path.join(self.main._ablage_dir(uid), f"probe_{_N[0]}_{len(self.hochgeladen)}_{os.urandom(4).hex()}")
        with open(pfad, "wb") as f:
            f.write(inhalt)
        c = database.get_db()
        aid = c.execute("INSERT INTO ablage (user_id, project_id, projekt_name, dokument_name, art, dateiname, datei_pfad, media) "
                        "VALUES (?, ?, 'Vereinsbrief (fiktiv)', ?, ?, ?, ?, ?)",
                        (uid, pid or self.quelle, dokument, art, dateiname, pfad, media)).lastrowid
        c.commit()
        c.close()
        return aid

    def uebernehmen(self, aid, project_id=None, user=None):
        return asyncio.run(ablage_projekt.uebernehmen(user or self.u, aid, project_id, None))

    def projekt_zeile(self, pid):
        c = database.get_db()
        try:
            return dict(c.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone())
        finally:
            c.close()


class Dateien(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="bruecke_dateien_")

    def schreiben(self, inhalt):
        p = os.path.join(self.d, os.urandom(4).hex())
        with open(p, "wb") as f:
            f.write(inhalt)
        return p

    def test_einzelne_datei_mit_passender_endung(self):
        (n, d, a), = ablage_projekt.dateien(self.schreiben(PDF), "brief.pdf", "application/pdf", 1000, 5)
        self.assertEqual((n, d, a), ("brief.pdf", PDF, "pdf"))
        (n, _d, a), = ablage_projekt.dateien(self.schreiben(docx()), "brief", "application/vnd…", 1000, 5)
        self.assertEqual((n, a), ("brief.docx", "docx"))

    def test_zip_ausgepackt_nur_pdf_und_word_flach(self):
        z = zip_mit(**{"01_a.pdf": PDF, "ordner__02_b.pdf": PDF, "liesmich.txt": b"fiktiv", "falsch.pdf": b"kein pdf"})
        liste = ablage_projekt.dateien(self.schreiben(z), "alle.zip", "application/zip", 1000, 5)
        self.assertEqual([(n, a) for n, _d, a in liste], [("01_a.pdf", "pdf"), ("02_b.pdf", "pdf")])

    def test_grenzen(self):
        with self.assertRaises(HTTPException) as e:
            ablage_projekt.dateien(self.schreiben(PDF + b"x" * 100), "brief.pdf", "application/pdf", 50, 5)
        self.assertEqual(e.exception.status_code, 413)
        with self.assertRaises(HTTPException) as e:
            ablage_projekt.dateien(self.schreiben(zip_mit(**{f"{i}.pdf": PDF for i in range(4)})), "alle.zip", "application/zip", 1000, 3)
        self.assertEqual(e.exception.status_code, 400)
        for inhalt in (b"Nur Text (fiktiv)", zip_mit(liesmich_txt=b"x")):
            with self.assertRaises(HTTPException) as e:
                ablage_projekt.dateien(self.schreiben(inhalt), "x.zip", "application/zip", 1000, 5)
            self.assertEqual(e.exception.status_code, 400)


class Uebernehmen(Basis):
    def test_neues_projekt_mit_namen_aus_der_herkunft(self):
        aid = self.eintrag()
        r = self.uebernehmen(aid)
        self.assertEqual((r["neues_projekt"], r["art"], len(r["dokumente"]), r["url"]), (True, "pdf", 1, f"/app?projekt={r['project_id']}"))
        self.assertEqual(self.hochgeladen, [("inkludocs_vereinsbrief.pdf", None)])
        p = self.projekt_zeile(r["project_id"])
        self.assertEqual((p["user_id"], p["tool"], p["name"]), (self.u["id"], "pdf", "Vereinsbrief (fiktiv) (barrierefreie PDF)"))
        self.assertEqual(r["projekt"], "Vereinsbrief (fiktiv) (barrierefreie PDF)")
        c = database.get_db()
        name = c.execute("SELECT display_name FROM documents WHERE id = ?", (r["dokumente"][0],)).fetchone()[0]
        c.close()
        self.assertEqual(name, "Vereinsbrief (fiktiv)", "das Dokument heisst wie in der Herkunft, nicht wie die Datei")

    def test_zip_wird_ein_projekt_mit_mehreren_dokumenten(self):
        aid = self.eintrag(zip_mit(**{"01_a.pdf": PDF, "02_b.pdf": PDF}), "alle_pdfua.zip", "application/zip", dokument="")
        r = self.uebernehmen(aid)
        self.assertEqual(len(r["dokumente"]), 2)
        self.assertEqual([pid for _n, pid in self.hochgeladen], [None, r["project_id"]], "die zweite Datei kommt ins selbe Projekt")
        self.assertEqual(self.projekt_zeile(r["project_id"])["name"], "Vereinsbrief (fiktiv) (barrierefreie PDF)", "Projekt der Herkunft")

    def test_in_ein_vorhandenes_eigenes_projekt_derselben_art(self):
        ziel = projekt(self.u["id"], "pdf", "pdf", "Jahresbericht (fiktiv)")
        r = self.uebernehmen(self.eintrag(), ziel)
        self.assertEqual((r["project_id"], r["neues_projekt"]), (ziel, False))
        self.assertEqual(self.hochgeladen, [("inkludocs_vereinsbrief.pdf", ziel)])
        self.assertEqual(self.projekt_zeile(ziel)["name"], "Jahresbericht (fiktiv)", "der Name bleibt")
        for anderes, code in ((projekt(konto()["id"], "pdf", "pdf"), 404), (projekt(self.u["id"], "word", "docx"), 400)):
            with self.assertRaises(HTTPException) as e:
                self.uebernehmen(self.eintrag(), anderes)
            self.assertEqual(e.exception.status_code, code)

    def test_fremder_eintrag_und_fehlende_datei(self):
        fremd = konto()
        aid = self.eintrag(uid=fremd["id"], pid=projekt(fremd["id"]))
        with self.assertRaises(HTTPException) as e:
            self.uebernehmen(aid)
        self.assertEqual(e.exception.status_code, 404)
        aid = self.eintrag()
        c = database.get_db()
        os.remove(c.execute("SELECT datei_pfad FROM ablage WHERE id = ?", (aid,)).fetchone()[0])
        c.close()
        with self.assertRaises(HTTPException) as e:
            self.uebernehmen(aid)
        self.assertEqual(e.exception.status_code, 404)
        self.assertEqual(self.hochgeladen, [], "nichts hochgeladen")

    def test_pdf_und_word_gemischt(self):
        aid = self.eintrag(zip_mit(**{"a.pdf": PDF, "b.docx": docx()}), "alle.zip", "application/zip")
        with self.assertRaises(HTTPException) as e:
            self.uebernehmen(aid)
        self.assertEqual(e.exception.status_code, 400)

    def test_tagesgrenze_wie_beim_hochladen_im_gespraech(self):
        with mock.patch.object(hochladen, "max_neue_projekte_tag", return_value=1):
            self.uebernehmen(self.eintrag())
            with self.assertRaises(HTTPException) as e:
                self.uebernehmen(self.eintrag())
            self.assertEqual(e.exception.status_code, 429)
            ziel = projekt(self.u["id"], "pdf", "pdf")
            self.assertFalse(self.uebernehmen(self.eintrag(), ziel)["neues_projekt"], "in ein vorhandenes: immer")

    def test_laufbuch_nur_wenn_der_agent_es_tut(self):
        r = self.uebernehmen(self.eintrag())
        self.assertEqual(laufbuch.fuer_konto(self.u["id"]), [], "ohne den Agenten: nicht gestreamt")
        d = laufbuch.fuer_konto(self.u["id"], nur_agent=False)[0]
        self.assertEqual((d["art"], d["project_id"], d["laeuft"], d["ok"]), ("uebernehmen", r["project_id"], False, True))
        with laufbuch.agent_kontext():
            r2 = self.uebernehmen(self.eintrag())
        d = laufbuch.fuer_konto(self.u["id"])[0]
        self.assertEqual((d["art"], d["project_id"], d["vorher"]), ("uebernehmen", r2["project_id"],
                                                                     {"art": "dokument", "document_id": r2["dokumente"][0]}))


class Werkzeug(Basis):
    def test_kreuz_und_quer_aber_nie_fremde_eintraege(self):
        from inkluagent.tools import ausgaben
        anderes = projekt(self.u["id"])
        r = ausgaben.ausgabe_in_projekt(self.quelle, self.u["id"], self.eintrag(pid=anderes))
        self.assertTrue(r["ok"], "ein Eintrag aus einem ANDEREN eigenen Projekt")
        lies = ausgaben.lies_ausgabe(self.quelle, self.u["id"], self.eintrag(pid=anderes))
        self.assertTrue(lies["ok"], lies)
        fremd = konto()
        aid = self.eintrag(uid=fremd["id"], pid=projekt(fremd["id"]))
        for r in (ausgaben.ausgabe_in_projekt(self.quelle, self.u["id"], aid), ausgaben.lies_ausgabe(self.quelle, self.u["id"], aid)):
            self.assertFalse(r["ok"])
            self.assertIn("liste_ausgaben", r["error"])

    def test_liste_dieses_projekts_oder_der_ganzen_ablage(self):
        from inkluagent.tools import ausgaben
        hier, dort = self.eintrag(), self.eintrag(pid=projekt(self.u["id"], name="Jahresbericht (fiktiv)"))
        nur = ausgaben.liste_ausgaben(self.quelle, self.u["id"])["result"]
        self.assertEqual({a["ausgabe_id"] for a in nur["ausgaben"]} & {hier, dort}, {hier})
        alle = ausgaben.liste_ausgaben(self.quelle, self.u["id"], alle=True)["result"]
        self.assertTrue({hier, dort} <= {a["ausgabe_id"] for a in alle["ausgaben"]})
        self.assertTrue(all("projekt_daten" in a and "project_id" in a for a in alle["ausgaben"]), "aus welchem Projekt, als Daten")
        self.assertEqual(alle["ausgaben_url"], "/ablage")

    def test_anderes_projekt_mit_seitenwechsel_link_und_fokus(self):
        from inkluagent import storage
        from inkluagent.tools import ausgaben
        with mock.patch.object(storage, "fokus_setzen") as fokus:
            r = ausgaben.ausgabe_in_projekt(self.quelle, self.u["id"], self.eintrag())
        self.assertTrue(r["ok"], r)
        pid = r["result"]["project_id"]
        self.assertEqual(r["aktionen"], [{"type": "seite_wechseln", "url": f"/app?projekt={pid}", "project_id": pid}])
        self.assertEqual((r["anhang"]["art"], r["anhang"]["url"]), ("link", f"/app?projekt={pid}"))
        fokus.assert_called_once_with(self.u["id"], pid)
        self.assertIn("Nach deiner Antwort wechselt die Seite", r["result"]["hinweis"])
        self.assertIn("projekt_daten", r["result"], "Name als Daten gekennzeichnet (Fremdtext ist keine Anweisung)")
        self.assertNotIn("projekt", r["result"])

    def test_in_dieses_projekt_ohne_seitenwechsel(self):
        from inkluagent.tools import ausgaben
        hier = projekt(self.u["id"], "pdf", "pdf")
        r = ausgaben.ausgabe_in_projekt(hier, self.u["id"], self.eintrag(pid=hier), hier)
        self.assertTrue(r["ok"], r)
        self.assertNotIn("aktionen", r)
        self.assertEqual(r["result"]["project_id"], hier)

    def test_pruefen_nachweis_aus_der_echten_pdf_sonst_aus_der_ablage(self):
        from inkluagent import nachweis
        from inkluagent.tools import ausgaben, oberflaeche
        aid = self.eintrag()
        n_pdf = {"art": "nachweis", "quelle": "pruefdatei", "bestanden": True, "befunde": 0}
        with mock.patch.object(oberflaeche, "pruefdatei_erstellen", return_value={"ok": True, "result": {}, "nachweis": n_pdf}) as pd:
            r = ausgaben.ausgabe_in_projekt(self.quelle, self.u["id"], aid, pruefen=True)
        self.assertTrue(r["ok"], r)
        pd.assert_called_once()
        self.assertEqual(pd.call_args[0][0], r["result"]["project_id"], "geprueft wird im NEUEN Projekt")
        self.assertEqual(r["nachweis"], [n_pdf])
        self.assertEqual(r["result"]["pruefung"], [{"pdf_ua": "bestanden", "befunde": 0}], "das Modell bekommt nur das Urteil")
        n_ablage = {"art": "nachweis", "quelle": "ablage", "bestanden": None, "befunde": 0}
        with mock.patch.object(oberflaeche, "pruefdatei_erstellen", return_value={"ok": False, "error": "keine Tags (fiktiv)"}), \
                mock.patch.object(nachweis, "aus_ausgabe", return_value=n_ablage) as aus:
            r = ausgaben.ausgabe_in_projekt(self.quelle, self.u["id"], self.eintrag(), pruefen=True)
        self.assertEqual((r["nachweis"], r["result"]["pruefung"][0]["pdf_ua"]), ([n_ablage], "nicht geprüft"))
        aus.assert_called_once()
        with mock.patch.object(oberflaeche, "pruefdatei_erstellen") as pd:
            r = ausgaben.ausgabe_in_projekt(self.quelle, self.u["id"], self.eintrag())
        pd.assert_not_called()
        self.assertNotIn("nachweis", r, "ohne pruefen keine Pruefung")

    def test_im_werkzeugsatz_von_word_und_pdf(self):
        from inkluagent import agent_loop
        for tool, typ in (("word", "docx"), ("pdf", "pdf")):
            p = {"id": 1, "tool": tool, "project_type": typ}
            defs, executor, _prompt = agent_loop._werkzeugsatz(p, 1, self.u["id"])
            self.assertIn("ausgabe_in_projekt", {d["name"] for d in defs}, tool)
            self.assertIn("ausgabe_in_projekt", executor._handlers(), tool)


class OhneProjekt(Basis):
    """Auf Seiten ohne Projekt (Startseite, Meine Ablage …) sieht der Agent die ganze Ablage, liest jeden eigenen Eintrag und laedt
    ihn in ein Projekt — als Arbeit des Agenten (Laufbuch)."""

    def test_ganze_ablage_lesen_und_laden(self):
        from inkluagent import konto
        aid = self.eintrag()
        k = konto.KontoExecutor(self.u["id"])
        liste = k.execute("liste_ausgaben", {})
        self.assertTrue(liste["ok"], liste)
        self.assertIn(aid, [a["ausgabe_id"] for a in liste["result"]["ausgaben"]])
        self.assertTrue(k.execute("lies_ausgabe", {"ausgabe_id": aid})["ok"])
        with mock.patch.object(konto, "aendern_an", return_value=False):
            self.assertFalse(k.execute("ausgabe_in_projekt", {"ausgabe_id": aid})["ok"], "Schalter aus: aendert nichts")
        with mock.patch.object(konto, "aendern_an", return_value=True):
            r = k.execute("ausgabe_in_projekt", {"ausgabe_id": aid})
        self.assertTrue(r["ok"], r)
        pid = r["result"]["project_id"]
        self.assertEqual(r["aktionen"], [{"type": "seite_wechseln", "url": f"/app?projekt={pid}", "project_id": pid}])
        d = laufbuch.fuer_konto(self.u["id"])[0]
        self.assertEqual((d["art"], d["quelle"]), ("uebernehmen", "agent"), "Konto-Werkzeuge laufen als Agent (Live-Fenster)")

    def test_im_werkzeugsatz_ohne_projekt(self):
        from inkluagent import konto
        with mock.patch.object(konto, "aendern_an", return_value=True):
            defs, _e, _s = konto.werkzeugsatz(self.u["id"])
        namen = {d["name"] for d in defs}
        self.assertTrue({"liste_ausgaben", "lies_ausgabe", "ausgabe_in_projekt"} <= namen, namen)
        with mock.patch.object(konto, "aendern_an", return_value=False):
            defs, _e, _s = konto.werkzeugsatz(self.u["id"])
        self.assertNotIn("ausgabe_in_projekt", {d["name"] for d in defs}, "aendernd: nur mit dem Schalter")


class KeinKnopf(unittest.TestCase):
    def test_kein_knopf_und_kein_endpunkt(self):
        """Steve 17.10.2026: „eigentlich brauche ich den Button in der Ablage gar nicht … es geht nur darum, dass der Agent auf
        alles Zugriff hat“ — von Hand: im Zielprojekt „Dokumente hinzufügen“."""
        with open(os.path.join(BACKEND, "templates", "ablage.html"), encoding="utf-8") as f:
            self.assertNotIn("in-projekt", f.read())
        with open(os.path.join(BACKEND, "main.py"), encoding="utf-8") as f:
            self.assertNotIn('"/api/ausgaben/{ausgabe_id}/in-projekt"', f.read())


if __name__ == "__main__":
    unittest.main()
