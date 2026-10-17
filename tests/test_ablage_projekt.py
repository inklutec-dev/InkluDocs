"""Aus der Ablage in ein Projekt (Steve 17.10.2026, backend/ablage_projekt.py): Knopf „In ein Projekt laden“ in Meine Ablage
(POST /api/ausgaben/{id}/in-projekt) und Werkzeug ausgabe_in_projekt des InkluAgent — EINE Funktion für beide Wege.

Prüft: Datei des Eintrags (PDF, Word, ZIP ausgepackt; nur PDF/Word; Größe; Anzahl), neues Projekt mit Namen aus der Herkunft oder
weiteres Dokument in ein eigenes Projekt derselben Art (fremdes Projekt 404, andere Art 400), fremder Eintrag und fehlende Datei 404,
Tagesgrenze nur über den InkluAgent, Laufbuch nur, wenn der Agent es tut, Endpunkt (422 bei unsinniger project_id), Werkzeug
(nur Einträge dieses Projekts, Seitenwechsel und Link bei anderem Projekt, Fokus des Gesprächs), Knopf in ablage.html am Quelltext.
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

    def uebernehmen(self, aid, project_id=None, durch_agent=False, user=None):
        return asyncio.run(ablage_projekt.uebernehmen(user or self.u, aid, project_id, None, None, durch_agent=durch_agent))

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

    def test_tagesgrenze_nur_ueber_den_agenten(self):
        with mock.patch.object(hochladen, "max_neue_projekte_tag", return_value=1):
            self.uebernehmen(self.eintrag(), durch_agent=True)
            with self.assertRaises(HTTPException) as e:
                self.uebernehmen(self.eintrag(), durch_agent=True)
            self.assertEqual(e.exception.status_code, 429)
            self.assertTrue(self.uebernehmen(self.eintrag())["neues_projekt"], "der Knopf zaehlt nicht zur Grenze des Agenten")
            ziel = projekt(self.u["id"], "pdf", "pdf")
            self.assertFalse(self.uebernehmen(self.eintrag(), ziel, durch_agent=True)["neues_projekt"], "in ein vorhandenes: immer")

    def test_laufbuch_nur_wenn_der_agent_es_tut(self):
        r = self.uebernehmen(self.eintrag())
        self.assertEqual(laufbuch.fuer_konto(self.u["id"]), [], "Knopf: nicht gestreamt")
        d = laufbuch.fuer_konto(self.u["id"], nur_agent=False)[0]
        self.assertEqual((d["art"], d["project_id"], d["laeuft"], d["ok"]), ("uebernehmen", r["project_id"], False, True))
        with laufbuch.agent_kontext():
            r2 = self.uebernehmen(self.eintrag())
        d = laufbuch.fuer_konto(self.u["id"])[0]
        self.assertEqual((d["art"], d["project_id"], d["vorher"]), ("uebernehmen", r2["project_id"],
                                                                     {"art": "dokument", "document_id": r2["dokumente"][0]}))


class Endpunkt(Basis):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        self.c = TestClient(self.main.app)
        self.c.cookies.set("token", self.main.create_token(self.u["id"], self.u["email"], 0))

    def test_neues_und_vorhandenes_projekt(self):
        r = self.c.post(f"/api/ausgaben/{self.eintrag()}/in-projekt", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["url"].startswith("/app?projekt="))
        ziel = projekt(self.u["id"], "pdf", "pdf")
        r = self.c.post(f"/api/ausgaben/{self.eintrag()}/in-projekt", json={"project_id": ziel})
        self.assertEqual((r.status_code, r.json()["project_id"]), (200, ziel))
        self.assertEqual(self.c.post(f"/api/ausgaben/{self.eintrag()}/in-projekt").status_code, 200, "ohne Inhalt: neues Projekt")

    def test_unsinn_und_fremdes(self):
        aid = self.eintrag()
        for unsinn in ("abc", True, -1, 1.5):
            self.assertEqual(self.c.post(f"/api/ausgaben/{aid}/in-projekt", json={"project_id": unsinn}).status_code, 422, unsinn)
        fremd = konto()
        aid = self.eintrag(uid=fremd["id"], pid=projekt(fremd["id"]))
        self.assertEqual(self.c.post(f"/api/ausgaben/{aid}/in-projekt", json={}).status_code, 404)
        self.assertEqual(self.hochgeladen, [])


class Werkzeug(Basis):
    def test_nur_eintraege_dieses_projekts(self):
        from inkluagent.tools import ausgaben
        anderes = projekt(self.u["id"])
        aid = self.eintrag(pid=anderes)
        r = ausgaben.ausgabe_in_projekt(self.quelle, self.u["id"], aid)
        self.assertFalse(r["ok"])
        self.assertIn("liste_ausgaben", r["error"])

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


class Oberflaeche(unittest.TestCase):
    def test_knopf_in_der_ablage(self):
        with open(os.path.join(BACKEND, "templates", "ablage.html"), encoding="utf-8") as f:
            t = f.read()
        k = t[t.index("function projektKnopf(a) {"):]
        k = k[:k.index("\n  }\n")]
        self.assertIn("if (a.datei_verfuegbar && (a.art === 'pdfua' || a.art === 'pdf' || a.art === 'docx')) aktionen.appendChild(projektKnopf(a));", t)
        self.assertIn("fetch('/api/ausgaben/' + a.id + '/in-projekt'", k)
        self.assertIn(r"/^\/app\?projekt=\d+$/.test(d.url)", k, "nur eine Projektadresse dieser Seite")
        self.assertIn("ja.setAttribute('aria-disabled', 'true');", k, "Fokus bleibt auf dem Knopf (nicht disabled)")
        self.assertNotIn(".disabled = true", k)
        self.assertIn("lab.htmlFor = 'ziel_' + a.id;", k, "Auswahl beschriftet")
        self.assertIn("announce(meldung.textContent);", k)
        self.assertNotIn("innerHTML = '<", k)
        self.assertIn("(p.tool === 'pdf' || p.tool === 'formular')", k, "nur Projekte derselben Art (wie hochladen._passt)")


if __name__ == "__main__":
    unittest.main()
