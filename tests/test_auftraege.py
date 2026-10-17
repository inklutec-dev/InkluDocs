"""Auftraege des InkluAgent, Baustein 2 (Steve 14.10.2026) — Schalter funktionen.AGENT_AUFTRAEGE, inkluagent/auftraege.py.
Wegwerf-Datenbank (/tmp), Ersatzmodell, Ersatzwerkzeuge (holen die Zustimmung wie die echten), keine Mails.
    docker exec -w /app <container> python3 -m unittest /app/tests/test_auftraege.py

Prueft: Plan aus dem Katalog (feste Reihenfolge, doppelte einmal, nur was es im Projekt gibt, Tagging gesperrt, Pruefung nur
getaggt, Obergrenze fuer das Herunterladen nach dem Bearbeiten, Guthaben); Karte nur ohne Fehler, mit Guthaben und ohne laufenden
Auftrag, Erteilen nur ueber den Knopf (kein „bestaetigt“), Summe neu gerechnet, Auftrag und Rueckmeldung in EINER Transaktion,
ein Auftrag je Projekt, JE_KONTO. Freigabe nur im Faden des Schritts, nur seine Art, nur einmal, hoechstens die Rest-Summe, nie
Unumkehrbares. Abarbeiten: nacheinander, Anlaufzeit, teurer als freigegeben haelt an, Scheitern haelt an, teilweise nicht, im
Projekt belegt wartet, Abbrechen, in der Oberflaeche abgebrochen, Projekt geloescht, Neustart, unterbrochener Start, unerwartete
Fehler, zu lang. Melden: wartet bis zum Ende, Dateien unter der Nachricht, nie zusammengelegt, keine zwoelf Stunden, melden_wenn_
fertig, Kontoloeschung, Konto-Werkzeuge, Agent-Schleife (nie in der Demo), Oberflaeche, Hilfe.
"""
import asyncio
import itertools
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="auftraege_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
for _name in ("INKLUAGENT_AUFTRAEGE", "INKLUAGENT_RUECKMELDUNG", "INKLUAGENT_KONTO", "INKLUAGENT_KARTEN"):
    os.environ.pop(_name, None)
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, auftraege as af, begleiter, chat_engine, karten, rueckmeldung as rm, storage  # noqa: E402
from inkluagent.daten import DATEN_MARKE  # noqa: E402
from inkluagent.tools import ausgaben, oberflaeche  # noqa: E402

try:
    import main  # noqa: E402 — die Vorschau rechnet mit denselben Kernfunktionen wie die Werkzeuge
except Exception as _e:  # noqa: BLE001
    main = None
    _MAIN_FEHLT = str(_e)

_NR = itertools.count(1)
WORD = ("docx", "word")
PDF = ("pdf", "pdf")


class an:
    """Alle Schalter, die Auftraege brauchen (Kontextmanager)."""

    def __init__(self):
        self.patches = [mock.patch.object(funktionen, n, True) for n in
                        ("AGENT_AUFTRAEGE", "AGENT_KARTEN", "AGENT_RUECKMELDUNG", "AGENT_KONTO")]

    def __enter__(self):
        for p in self.patches:
            p.start()

    def __exit__(self, *a):
        for p in reversed(self.patches):
            p.stop()


def nutzer() -> dict:
    mail = f"auftrag{next(_NR)}-{time.time_ns()}@beispiel.invalid"
    uid = database.create_user(mail, "geheim-12345", "Erika Mustermann (fiktiv)", email_verified=1)
    return {"id": uid, "email": mail}


def projekt(uid: int, art=WORD, name="Vereinsbrief (fiktiv)", bilder=3, status="extracted", getaggt=0) -> int:
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, ?, '', ?, ?, ?, ?)", (uid, name, name, art[1], art[0], status)).lastrowid
    endung = ".pdf" if art == PDF else ".docx"
    did = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path, getaggt) VALUES (?, 1, ?, ?, ?)",
                    (pid, "brief" + endung, "/tmp/fiktiv/brief" + endung, getaggt)).lastrowid
    for i in range(bilder):
        c.execute("INSERT INTO images (project_id, document_id, page_number, image_index, image_path, status) "
                  "VALUES (?, ?, 1, ?, '', 'done')", (pid, did, i))
    c.commit()
    c.close()
    return pid


def sql(befehl: str, *werte):
    c = database.get_db()
    try:
        r = c.execute(befehl, werte)
        rows = [dict(x) for x in r.fetchall()] if befehl.lstrip().upper().startswith("SELECT") else r.rowcount
        c.commit()
        return rows
    finally:
        c.close()


def leeren() -> None:
    """Jeder Test fuer sich: Rundgaenge sehen die Eintraege ALLER Konten."""
    sql("DELETE FROM agent_auftraege")
    sql("DELETE FROM agent_rueckmeldungen")


class Werkzeug:
    """Ersatz fuer ein Werkzeug (alt_texte_generieren, exportiere_word …): holt die Zustimmung wie das echte ueber
    ausgaben._angebot_einloesen (Art und Preis), merkt sich die Freigabe, die es im Faden sah, und startet dann — im Hintergrund
    (setzt den Projektstatus wie der Sammellauf) oder sofort (mit Datei)."""

    def __init__(self, art: str, preis: int, hintergrund: bool = True, erlaubt: bool = True, ergebnis=None):
        self.art, self.preis, self.hintergrund, self.erlaubt, self.ergebnis = art, preis, hintergrund, erlaubt, ergebnis
        self.aufrufe, self.freigaben = [], []
        self.nebenher = None   # Funktion, die waehrend des Aufrufs in einem ANDEREN Faden laeuft (etwa das Gespraech)

    def __call__(self, project_id, user_id, document_id=None, *a, bestaetigt=False, turn=None, **kw):
        self.aufrufe.append({"document_id": document_id, "bestaetigt": bestaetigt, **kw})
        if self.nebenher:
            t = threading.Thread(target=self.nebenher)
            t.start()
            t.join()
        f = getattr(ausgaben._AUFTRAG, "f", None)
        self.freigaben.append(dict(f) if f else None)
        if not self.erlaubt:
            return {"ok": True, "result": {"rueckfrage_noetig": True, "erlaubt": False, "hinweis": "Das Guthaben reicht nicht."}}
        grund = ausgaben._angebot_einloesen((user_id, project_id, self.art, document_id), self.preis, "runde")
        if grund:
            return {"ok": True, "result": {"rueckfrage_noetig": True, "erlaubt": True, "hinweis": grund}}
        if self.ergebnis is not None:
            return self.ergebnis
        if self.hintergrund:
            sql("UPDATE projects SET status = 'processing' WHERE id = ?", project_id)
            return {"ok": True, "result": {"gestartet": True, "anzahl": 3}}
        return {"ok": True, "result": {"dateiname": "brief-fiktiv.docx", "preis": self.preis},
                "anhang": {"art": "docx", "dateiname": "brief-fiktiv.docx", "download_url": "/api/sofort/fiktiv", "label": "docx"}}


class Modell:
    def __init__(self, text="Fertig: Alt-Texte für 3 Bilder und die Word-Datei (fiktiv)."):
        self.text, self.aufrufe = text, []

    def chat(self, messages, images=None, model=None, max_tokens=1024, temperature=0.4):
        self.aufrufe.append(messages)
        return self.text


@unittest.skipIf(main is None, "main nicht ladbar")
class Basis(unittest.TestCase):
    def setUp(self):
        leeren()
        karten._KARTEN.clear()
        self.schalter = an()
        self.schalter.__enter__()
        self.u = nutzer()
        self.pid = projekt(self.u["id"])
        self.gid = storage.aktuelles_gespraech(self.u["id"])
        self.modell = Modell()
        mock.patch.object(chat_engine, "_provider", self.modell).start()
        mock.patch.object(rm, "_agent_arbeitet", return_value=False).start()
        mock.patch.object(main, "send_email", return_value=True).start()
        # Word-Export: dieselbe Rechnung wie die Oberflaeche braucht Dateien — hier fest 30 Credits fuer 3 Bilder
        mock.patch.object(ausgaben, "_kosten_vorschau", return_value={"preis": 30, "bilder": 3}).start()

    def tearDown(self):
        mock.patch.stopall()
        self.schalter.__exit__()

    def wunsch(self, *schritte, **mehr):
        return af.wunsch_aus(dict({"schritte": list(schritte)}, **mehr))

    def kontext(self, *schritte, pid=None, **mehr):
        p = af._eigenes_projekt(self.u["id"], pid or self.pid)
        plan = af.planen(self.u["id"], p, self.wunsch(*schritte, **mehr))
        return {"project_id": p["id"], "wunsch": self.wunsch(*schritte, **mehr), "summe": plan["summe"],
                "fingerabdruck": af.fingerabdruck(plan), "gespraech_id": self.gid, "sprache": "de",
                "auftrag": "Mach die Alt-Texte und dann die Word-Datei (fiktiv)"}

    def erteilen(self, *schritte, per_mail=False, **mehr):
        status, antwort = af.erteilen(self.kontext(*schritte, **mehr), self.u, {"per_mail": per_mail}, lambda s: s)
        self.assertEqual(status, 200, antwort)
        return antwort["auftrag_id"]

    def auftrag(self, aid: int) -> dict:
        return af._laden(aid)

    def schritt(self, aid: int, i: int) -> dict:
        return self.auftrag(aid)["schritte"][i]

    def bilder_status(self, status: str, n: int = 1):
        sql("UPDATE images SET status = ? WHERE id IN (SELECT id FROM images WHERE project_id = ? LIMIT ?)", status, self.pid, n)


# ─── 1. Plan ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Plan(Basis):
    def projekt_dict(self, pid=None):
        return af._eigenes_projekt(self.u["id"], pid or self.pid)

    def test_feste_reihenfolge_doppelte_einmal_und_summe(self):
        plan = af.planen(self.u["id"], self.projekt_dict(), self.wunsch("word", "alt_texte", "ALT_TEXTE"))
        self.assertEqual(plan["fehler"], [])
        self.assertEqual([e["art"] for e in plan["eintraege"]], ["alt_texte", "word"], "Reihenfolge des Katalogs, nicht des Modells")
        self.assertEqual([e["preis_geplant"] for e in plan["eintraege"]], [15, 30], "3 Bilder je 5 Credits; Word fest")
        self.assertEqual(plan["summe"], 45)
        self.assertEqual([e["freigabe"] for e in plan["eintraege"]], ["alt_texte", "docx"], "die Art des Angebots-Schluessels")
        self.assertFalse(plan["hoechstens"])

    def test_unbekannt_oder_nicht_im_projekt(self):
        plan = af.planen(self.u["id"], self.projekt_dict(), self.wunsch("alt_texte", "fliegen", "pruefung"))
        self.assertEqual(plan["eintraege"], [], "mit einem Fehler gibt es keinen halben Plan")
        self.assertIn("Einen Schritt „fliegen“ gibt es nicht.", plan["fehler"])
        self.assertTrue(any("KI-basierte Prüfung" in f for f in plan["fehler"]), plan["fehler"])

    def test_wunsch_wird_begrenzt(self):
        w = af.wunsch_aus({"schritte": ["alt_texte"] * 50 + ["x" * 500], "zielsprache": "EN-GB" + "x" * 50, "format": "XLSX"})
        self.assertEqual(w["schritte"], ["alt_texte"])
        self.assertEqual(w["zielsprache"], "en-gbxxxxx")
        self.assertEqual(w["format"], "xlsx")
        self.assertEqual(af.wunsch_aus({"schritte": "word"})["schritte"], ["word"])

    def test_tagging_gesperrt(self):
        pid = projekt(self.u["id"], PDF)
        with mock.patch.object(funktionen, "TAGGING_PROFESSIONELL", False):
            plan = af.planen(self.u["id"], self.projekt_dict(pid), self.wunsch("tagging", "alt_texte"))
        self.assertEqual(plan["eintraege"], [])
        self.assertIn(funktionen.TAGGING_PROFESSIONELL_GESPERRT, plan["fehler"])

    def test_pruefung_nur_fuer_getaggte_oder_im_auftrag_getaggte(self):
        import pdf_tagging
        import tagging_api
        pid = projekt(self.u["id"], PDF)
        mock.patch.object(tagging_api, "_seiten", return_value=4).start()
        mock.patch.object(tagging_api, "lesbar_grund", return_value=None).start()
        mock.patch.object(pdf_tagging, "verfuegbar", return_value=True).start()
        with mock.patch.object(funktionen, "KI_PRUEFUNG", True), mock.patch.object(funktionen, "TAGGING_PROFESSIONELL", True):
            ohne = af.planen(self.u["id"], self.projekt_dict(pid), self.wunsch("pruefung"))
            mit = af.planen(self.u["id"], self.projekt_dict(pid), self.wunsch("pruefung", "tagging"))
        self.assertEqual(ohne["eintraege"], [])
        self.assertTrue(any("nur für getaggte Dokumente" in f for f in ohne["fehler"]), ohne["fehler"])
        self.assertEqual([(e["art"], e["preis_geplant"]) for e in mit["eintraege"]], [("tagging", 80), ("pruefung", 8)],
                         "4 Seiten: Tagging 20 je Seite, Pruefung 2 je Seite")
        self.assertEqual(mit["eintraege"][0]["args"], {"document_id": mit["eintraege"][0]["document_id"]})

    def test_herunterladen_nach_dem_bearbeiten_als_obergrenze(self):
        import billing
        pid = projekt(self.u["id"], PDF, bilder=2, getaggt=1)
        with mock.patch.object(main, "_pdf_export_plan", return_value={"preis": 0}) as genau:
            allein = af.planen(self.u["id"], self.projekt_dict(pid), self.wunsch("pdf"))
            danach = af.planen(self.u["id"], self.projekt_dict(pid), self.wunsch("alt_texte", "pdf"))
        self.assertEqual((allein["eintraege"][0]["preis_geplant"], allein["eintraege"][0]["hoechstens"]), (0, False))
        genau.assert_called_once()
        pdf = danach["eintraege"][-1]
        self.assertEqual((pdf["preis_geplant"], pdf["hoechstens"]), (billing.pdf_download_preis(2, 0)["preis"], True))
        self.assertTrue(danach["hoechstens"])

    def test_guthaben(self):
        import billing
        with mock.patch.object(billing, "verfuegbare_credits", return_value=40):
            plan = af.planen(self.u["id"], self.projekt_dict(), self.wunsch("alt_texte", "word"))
        self.assertEqual((plan["summe"], plan["verfuegbar"], plan["erlaubt"]), (45, 40, False))

    def test_einzahl(self):
        """„ein Bild“, nicht „1 Bilder“ (Klickprobe 14.10.2026)."""
        pid = projekt(self.u["id"], WORD, bilder=1)
        plan = af.planen(self.u["id"], self.projekt_dict(pid), self.wunsch("alt_texte"))
        self.assertEqual(plan["eintraege"][0]["umfang"], "ein Bild")
        self.assertEqual(af._menge(lambda s: s, 2, "ein Feld", "{n} Felder"), "2 Felder")

    def test_nichts_zu_tun_steht_als_hinweis(self):
        pid = projekt(self.u["id"], WORD, bilder=0)
        plan = af.planen(self.u["id"], self.projekt_dict(pid), self.wunsch("alt_texte", "word"))
        self.assertEqual([e["art"] for e in plan["eintraege"]], ["word"])
        self.assertIn("Alt-Texte generieren entfällt: Es gibt keine Bilder dafür.", plan["hinweise"])


# ─── 2. Werkzeug auftrag_vorbereiten ────────────────────────────────────────────────────────────────────────────────────────
class Vorbereiten(Basis):
    def vorbereiten(self, args, project_id="eigenes", gespraech_id="eigenes"):
        return af.ausfuehren(af.WERKZEUG_VORBEREITEN, args, user_id=self.u["id"],
                             project_id=self.pid if project_id == "eigenes" else project_id,
                             gespraech_id=self.gid if gespraech_id == "eigenes" else gespraech_id, sprache="de",
                             auftrag="Mach alles fertig (fiktiv)")

    def test_karte_mit_schritten_preisen_summe_und_haken(self):
        r = self.vorbereiten({"schritte": ["word", "alt_texte"], "per_mail": True})
        self.assertTrue(r["ok"], r)
        k = r["anhang"]
        self.assertEqual((k["art"], k["karte"], k["titel"], k["knopf"], k["zustand"]),
                         ("formular", "auftrag", "Dein Auftrag", "Auftrag erteilen (45 Credits)", "offen"),
                         "der Knopf nennt die Summe (Pruefung N7)")
        self.assertEqual(k["zeilen"], ["Alt-Texte generieren (3 Bilder): 15 Credits", "Als Word herunterladen (3 Bilder): 30 Credits"],
                         "nur die Schritte, ohne Nummer im Text")
        self.assertTrue(k["nummeriert"], "die Oberflaeche nummeriert sie (ol)")
        self.assertNotIn("hinweise", k, "keine Hinweise: kein leerer Absatz")
        self.assertIn("Projekt „Vereinsbrief (fiktiv)“. Zusammen 45 Credits.", k["text"])
        self.assertIn("du kannst die Seite danach schließen", k["text"])
        self.assertEqual([(f["name"], f["art"], f["wert"]) for f in k["felder"]], [("per_mail", "haken", True)])
        e = karten._KARTEN[k["karten_id"]]
        self.assertEqual((e["art"], e["user_id"], e["kontext"]["summe"], e["kontext"]["wunsch"]["schritte"]),
                         ("auftrag", self.u["id"], 45, ["word", "alt_texte"]))
        self.assertEqual(e["kontext"]["gespraech_id"], self.gid)
        res = r["result"]
        self.assertTrue(res["projekt_daten"].startswith(DATEN_MARKE))
        self.assertTrue(all(s["umfang_daten"].startswith(DATEN_MARKE) for s in res["schritte"]), "Namen aus Dateien als Daten")
        self.assertIn("erteilen kann nur der Nutzer", res["hinweis"])
        self.assertEqual(sql("SELECT * FROM agent_auftraege"), [], "die Karte legt noch keinen Auftrag an")

    def test_ohne_mailwunsch_ist_der_haken_leer(self):
        k = self.vorbereiten({"schritte": ["alt_texte"], "per_mail": "nein"})["anhang"]
        self.assertFalse(k["felder"][0]["wert"])

    def test_kein_ja_kann_erteilen(self):
        defs = af.werkzeuge(self.pid)
        self.assertNotIn("bestaetigt", defs[0]["input_schema"]["properties"], "erteilen kann nur der Knopf")
        r = self.vorbereiten({"schritte": ["alt_texte"], "bestaetigt": True})
        self.assertEqual(r["anhang"]["karte"], "auftrag")
        self.assertEqual(sql("SELECT * FROM agent_auftraege"), [])

    def test_keine_karte_bei_fehler_guthaben_oder_laufendem_auftrag(self):
        import billing
        r = self.vorbereiten({"schritte": ["alt_texte", "tagging"]})
        self.assertIsNone(r["result"]["karte"])
        self.assertNotIn("anhang", r)
        self.assertIn("alt_texte", r["result"]["moegliche_schritte"])
        with mock.patch.object(billing, "verfuegbare_credits", return_value=10):
            r = self.vorbereiten({"schritte": ["alt_texte", "word"]})
        self.assertTrue(r["result"]["guthaben_reicht_nicht"])
        self.assertNotIn("anhang", r)
        self.erteilen("alt_texte")
        r = self.vorbereiten({"schritte": ["word"]})
        self.assertIsNone(r["result"]["karte"])
        self.assertIn("läuft schon ein Auftrag", r["result"]["hinweis"])

    def test_nur_eigene_projekte_und_nur_im_gespraech(self):
        fremd = projekt(nutzer()["id"])
        self.assertFalse(self.vorbereiten({"schritte": ["alt_texte"]}, project_id=fremd)["ok"])
        self.assertFalse(self.vorbereiten({"schritte": ["alt_texte"]}, gespraech_id=None)["ok"])
        konto = self.vorbereiten({"schritte": ["alt_texte"], "project_id": self.pid}, project_id=None)
        self.assertEqual(konto["anhang"]["karte"], "auftrag", "auf Seiten ohne Projekt mit project_id")
        self.assertIsNone(karten._KARTEN[konto["anhang"]["karten_id"]]["project_id"], "Verlauf des Kontos")
        self.assertIn("project_id", af.werkzeuge(None)[0]["input_schema"]["required"])


# ─── 3. Erteilen ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Erteilen(Basis):
    def test_auftrag_und_rueckmeldung_in_einem_schritt(self):
        aid = self.erteilen("word", "alt_texte", per_mail=True)
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["freigegeben"], a["verbraucht"], a["gespraech_id"]), ("laeuft", 45, 0, self.gid))
        self.assertEqual([(s["art"], s["zustand"], s["preis_geplant"]) for s in a["schritte"]],
                         [("alt_texte", "offen", 15), ("word", "offen", 30)])
        z = sql("SELECT * FROM agent_rueckmeldungen")
        self.assertEqual([(x["art"], x["auftrag_id"], x["per_mail"], x["zustand"], x["auftrag"]) for x in z],
                         [("auftrag", aid, 1, "wartet", "Mach die Alt-Texte und dann die Word-Datei (fiktiv)")])

    def test_antwort_sagt_hintergrund_und_rueckmeldung(self):
        status, antwort = af.erteilen(self.kontext("alt_texte"), self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual(antwort["karte"]["zustand"], "gesendet")
        self.assertEqual(antwort["text"], "Auftrag erteilt. Die Schritte laufen jetzt nacheinander im Hintergrund; du kannst die "
                                          "Seite schließen. " + rm.T_MELDE_MICH)

    def test_andere_summe_karte_gilt_nicht_mehr(self):
        k = self.kontext("alt_texte", "word")
        projekt_bild = sql("SELECT id FROM images WHERE project_id = ? LIMIT 1", self.pid)[0]["id"]
        sql("DELETE FROM images WHERE id = ?", projekt_bild)   # jetzt nur noch 2 Bilder: 40 statt 45 Credits
        status, antwort = af.erteilen(k, self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual((status, antwort["karte"]["zustand"]), (409, "abgelaufen"))
        self.assertEqual(sql("SELECT * FROM agent_auftraege"), [])

    def test_guthaben_laesst_die_karte_offen(self):
        import billing
        k = self.kontext("alt_texte", "word")
        with mock.patch.object(billing, "verfuegbare_credits", return_value=40):
            status, antwort = af.erteilen(k, self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual(status, 402)
        self.assertNotIn("karte", antwort, "Guthaben aufladen und noch einmal klicken")
        self.assertIn("5 Credits fehlen", antwort["detail"])

    def test_ein_auftrag_je_projekt_und_hoechstens_je_konto(self):
        self.erteilen("alt_texte")
        status, antwort = af.erteilen(self.kontext("word"), self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual(status, 409)
        self.assertNotIn("karte", antwort)
        for _i in range(af.JE_KONTO - 1):
            self.pid = projekt(self.u["id"])
            self.erteilen("alt_texte")
        self.pid = projekt(self.u["id"])
        status, antwort = af.erteilen(self.kontext("alt_texte"), self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual(status, 409)
        self.assertIn(f"Es laufen schon {af.JE_KONTO} Aufträge", antwort["detail"])

    def test_zu_viele_rueckmeldungen_nichts_angelegt(self):
        for _i in range(rm.WARTEND_JE_KONTO):
            rm.anmelden(self.u["id"], self.gid, projekt(self.u["id"]), "alt_texte")
        status, antwort = af.erteilen(self.kontext("alt_texte"), self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual(status, 409)
        self.assertEqual(sql("SELECT * FROM agent_auftraege"), [], "zurueckgerollt: kein Auftrag ohne Rueckmeldung")

    def test_fremdes_oder_geloeschtes_projekt(self):
        k = self.kontext("alt_texte")
        status, antwort = af.erteilen(k, nutzer(), {"per_mail": False}, lambda s: s)
        self.assertEqual((status, antwort["karte"]["zustand"]), (404, "abgelaufen"))
        with mock.patch.object(funktionen, "AGENT_AUFTRAEGE", False):
            status, _a = af.erteilen(k, self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual(status, 404, "Schalter aus: die Karte gilt nicht mehr")


# ─── 4. Freigabe im Faden des Schritts ──────────────────────────────────────────────────────────────────────────────────────
class Freigabe(unittest.TestCase):
    S = (1, 2, "alt_texte", None)

    def test_nur_art_nur_einmal_hoechstens_rest(self):
        with ausgaben.auftrag_schritt({"alt_texte"}, 20) as f:
            self.assertEqual(ausgaben._angebot_einloesen((1, 2, "docx", None), 5, "t"), ausgaben._G_KEIN, "andere Art")
            self.assertEqual(ausgaben._angebot_einloesen(self.S, 25, "t"), ausgaben.G_AUFTRAG_SUMME, "mehr als der Rest")
            self.assertEqual((f["grund"], f["preis_noetig"]), ("summe", 25))
            self.assertIsNone(ausgaben._angebot_einloesen(self.S, 15, "t"))
            self.assertEqual(f["preis"], 15)
            self.assertEqual(ausgaben._angebot_einloesen(self.S, 1, "t"), ausgaben._G_KEIN, "nur einmal je Schritt")
        self.assertEqual(ausgaben._angebot_einloesen(self.S, 15, "t"), ausgaben._G_KEIN, "ausserhalb gilt sie nicht")

    def test_nie_unumkehrbares(self):
        with ausgaben.auftrag_schritt({"loeschen", "projekt_loeschen"}, 1000):
            self.assertEqual(ausgaben._angebot_einloesen((1, 2, "loeschen", 3), 0, "t"), ausgaben._G_KEIN)

    def test_ein_anderer_faden_sieht_sie_nicht(self):
        gesehen = []
        with ausgaben.auftrag_schritt({"alt_texte"}, 100):
            t = threading.Thread(target=lambda: gesehen.append(ausgaben._angebot_einloesen(self.S, 5, "t")))
            t.start()
            t.join()
        self.assertEqual(gesehen, [ausgaben._G_KEIN])

    def test_ein_angebot_aus_dem_gespraech_gilt_im_schritt_nicht(self):
        schluessel = (7, 8, "alt_texte", None)   # eigener Schluessel: das Angebot bleibt im Prozess liegen
        ausgaben._angebot_merken(schluessel, 5, "frueher")
        with ausgaben.auftrag_schritt({"docx"}, 100) as f:
            self.assertEqual(ausgaben._angebot_einloesen(schluessel, 5, "jetzt"), ausgaben._G_KEIN)
        self.assertIsNone(f["preis"])


@unittest.skipIf(main is None, "main nicht ladbar")
class FreigabeEchtesWerkzeug(Basis):
    def test_tabelle_nur_mit_freigabe(self):
        """Das echte Werkzeug exportiere_alt_texte: mit bestaetigt=true ohne Angebot fragt es nach — im Faden eines Auftrags
        laeuft es, und abgebucht wird genau ein Mal."""
        p = af._eigenes_projekt(self.u["id"], self.pid)
        namen, executor = af._werkzeugsatz(p, self.u["id"])
        self.assertIn("exportiere_alt_texte", namen)
        bau = {"dateiname": "alt-texte.csv", "media": "text/csv", "daten": b"a;b", "aktion": "csv_export", "preis": 10}
        mock.patch.object(main, "_tabellen_export_bauen", return_value=bau).start()
        mock.patch.object(main, "sofort_download_ablegen", return_value="/api/sofort/fiktiv").start()
        buchen = mock.patch.object(main.billing, "verbuche").start()
        ohne = executor.execute("exportiere_alt_texte", {"format": "csv", "bestaetigt": True})
        self.assertTrue(ohne["result"]["rueckfrage_noetig"])
        buchen.assert_not_called()
        namen, executor = af._werkzeugsatz(p, self.u["id"])
        with ausgaben.auftrag_schritt({"tabelle_csv"}, 10) as f:
            mit = executor.execute("exportiere_alt_texte", {"format": "csv", "bestaetigt": True})
        self.assertEqual(mit["anhang"]["download_url"], "/api/sofort/fiktiv")
        self.assertEqual(f["preis"], 10)
        buchen.assert_called_once()


# ─── 5. Abarbeiten ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class MitWerkzeugen(Basis):
    """Alt-Texte (im Hintergrund, 15 Credits) und Word-Datei (sofort, 30 Credits) als Ersatzwerkzeuge."""

    def setUp(self):
        super().setUp()
        self.alt = Werkzeug("alt_texte", 15)
        self.word = Werkzeug("docx", 30, hintergrund=False)
        mock.patch.object(oberflaeche, "alt_texte_generieren", self.alt).start()
        mock.patch.object(ausgaben, "exportiere_word", self.word).start()

    def fertig_machen(self):
        """Der Sammellauf ist zu Ende (so sieht ihn die Oberflaeche)."""
        sql("UPDATE projects SET status = 'extracted' WHERE id = ?", self.pid)


class Abarbeiten(MitWerkzeugen):
    def test_nacheinander_mit_rest_summe_bis_fertig(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["schritte"][0]["zustand"], a["schritte"][0]["preis"], a["verbraucht"]), ("laeuft", 15, 15))
        self.assertEqual(self.alt.aufrufe[0]["bestaetigt"], True)
        self.assertEqual(self.alt.freigaben[0]["hoechstens"], 45)
        self.assertEqual(self.word.aufrufe, [], "der zweite wartet auf den ersten")
        af.durchlauf()
        self.assertEqual(self.schritt(aid, 0)["zustand"], "laeuft", "laeuft noch")
        self.assertTrue(self.schritt(aid, 0)["gesehen"])
        self.fertig_machen()
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual([s["zustand"] for s in a["schritte"]], ["fertig", "fertig"])
        self.assertEqual(self.word.freigaben[0]["hoechstens"], 30, "Rest-Summe: 45 - 15")
        self.assertEqual(a["schritte"][1]["anhang"]["dateiname"], "brief-fiktiv.docx")
        self.assertEqual((a["zustand"], a["verbraucht"]), ("fertig", 45))
        self.assertTrue(a["beendet_am"])
        self.assertEqual(len(self.alt.aufrufe) + len(self.word.aufrufe), 2, "jeder Schritt genau einmal")

    def test_anlaufzeit(self):
        """Ein Lauf, der noch nicht als laufend sichtbar ist, gilt in der Anlaufzeit nicht als fertig."""
        self.alt.hintergrund = False
        self.alt.ergebnis = {"ok": True, "result": {"gestartet": True}}   # startet, setzt den Status aber (noch) nicht
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        af.durchlauf()
        self.assertEqual(self.schritt(aid, 0)["zustand"], "laeuft")
        a = self.auftrag(aid)
        a["schritte"][0]["gestartet_am"] -= af.ANLAUF_S + 1
        af._speichern(a)
        af.durchlauf()
        self.assertEqual(self.schritt(aid, 0)["zustand"], "fertig")

    def test_teurer_als_freigegeben_haelt_an(self):
        self.word.preis = 31
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.fertig_machen()
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][1]["zustand"], a["verbraucht"]), ("gescheitert", "fehler", 15))
        self.assertEqual(a["schritte"][1]["grund"], af.T_SUMME.format(n=31, rest=30))
        self.assertEqual(a["schritte"][1]["preis"], 0)

    def test_guthaben_weg_haelt_an(self):
        self.word.erlaubt = False
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.fertig_machen()
        af.durchlauf()
        self.assertEqual(self.schritt(aid, 1)["grund"], af.T_GUTHABEN)

    def test_scheitern_haelt_an_teilweise_nicht(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.bilder_status("error")
        self.fertig_machen()
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual([s["zustand"] for s in a["schritte"]], ["teilweise", "fertig"], "ein Bild ohne Alt-Text haelt nicht an")
        leeren()
        self.alt.aufrufe.clear()
        self.word.aufrufe.clear()
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.bilder_status("error", n=3)
        self.fertig_machen()
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual([s["zustand"] for s in a["schritte"]], ["fehler", "ausgelassen"])
        self.assertEqual(a["zustand"], "gescheitert")
        self.assertEqual(self.word.aufrufe, [], "was danach kaeme, startet nicht und kostet nichts")

    def test_startfehler_des_werkzeugs(self):
        self.alt.ergebnis = {"ok": False, "error": "Verarbeitung laeuft bereits"}
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"], a["verbraucht"]), ("gescheitert", "Verarbeitung laeuft bereits", 0))

    def test_nichts_mehr_zu_tun(self):
        self.alt.ergebnis = {"ok": True, "result": {"gestartet": False, "anzahl": 0, "hinweis": "Nichts zu generieren."}}
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual([s["zustand"] for s in a["schritte"]], ["uebersprungen", "fertig"])
        self.assertEqual(a["schritte"][0]["grund"], af.T_NICHTS)

    def test_belegt_wartet_und_gibt_dann_auf(self):
        aid = self.erteilen("alt_texte", "word")
        sql("UPDATE projects SET status = 'processing' WHERE id = ?", self.pid)   # aus der Oberflaeche gestartet
        af.durchlauf()
        self.assertEqual(self.alt.aufrufe, [])
        s = self.schritt(aid, 0)
        self.assertEqual(s["zustand"], "offen")
        a = self.auftrag(aid)
        a["schritte"][0]["wartet_seit"] -= af.WARTEN_HOECHSTENS_S + 1
        af._speichern(a)
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("gescheitert", af.T_BELEGT))
        self.assertEqual(self.alt.aufrufe, [])

    def test_abbrechen_nach_dem_laufenden_schritt(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.assertTrue(af.abbrechen(self.u["id"], self.pid))
        self.assertFalse(af.abbrechen(nutzer()["id"], self.pid), "nur eigene")
        af.durchlauf()
        self.assertEqual(self.auftrag(aid)["zustand"], "laeuft", "der laufende Schritt laeuft zu Ende")
        self.fertig_machen()
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], [s["zustand"] for s in a["schritte"]]), ("abgebrochen", ["fertig", "ausgelassen"]))
        self.assertEqual(self.word.aufrufe, [])

    def test_in_der_oberflaeche_abgebrochen(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        sql("UPDATE projects SET lauf_hinweis = '{\"grund\": \"abbruch\"}' WHERE id = ?", self.pid)
        self.fertig_machen()
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("abgebrochen", af.T_VOM_NUTZER))
        self.assertEqual(self.word.aufrufe, [])

    def test_projekt_geloescht(self):
        aid = self.erteilen("alt_texte", "word")
        sql("DELETE FROM projects WHERE id = ?", self.pid)
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("gescheitert", af.T_PROJEKT_WEG))

    def test_neustart(self):
        laeuft = self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.pid = projekt(self.u["id"])
        dazwischen = self.erteilen("alt_texte", "word")
        af.nach_neustart()
        a = self.auftrag(laeuft)
        self.assertEqual((a["zustand"], [s["zustand"] for s in a["schritte"]]), ("gescheitert", ["fehler", "ausgelassen"]))
        self.assertEqual(a["schritte"][0]["grund"], af.T_NEUSTART)
        self.assertEqual(self.auftrag(dazwischen)["zustand"], "laeuft", "zwischen zwei Schritten geht es weiter")

    def test_unterbrochener_start_wird_nicht_wiederholt(self):
        aid = self.erteilen("alt_texte", "word")
        a = self.auftrag(aid)
        a["schritte"][0].update(zustand="laeuft", gestartet=False, gestartet_am=time.time())
        af._speichern(a)
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("gescheitert", af.T_UNTERBROCHEN))
        self.assertEqual(self.alt.aufrufe, [])

    def test_unerwartete_fehler(self):
        aid = self.erteilen("alt_texte", "word")
        with mock.patch.object(af, "_belegt", side_effect=RuntimeError("Datenbank gesperrt (fiktiv)")):
            for _i in range(af.FEHLVERSUCHE):
                af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("gescheitert", af.T_INTERN))

    def test_start_wirft(self):
        aid = self.erteilen("alt_texte", "word")
        with mock.patch.object(af, "_werkzeugsatz", side_effect=RuntimeError("kaputt (fiktiv)")):
            af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("gescheitert", af.T_INTERN), "einmal, nicht wiederholt")

    def test_zu_lang(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        a = self.auftrag(aid)
        a["schritte"][0]["gestartet_am"] -= af.SCHRITT_HOECHSTENS_S + 1
        af._speichern(a)
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("gescheitert", af.T_ZU_LANG))

    def test_stand_und_laufende_arbeit(self):
        from inkluagent import konto
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        st = af.ausfuehren(af.WERKZEUG_STAND, {}, user_id=self.u["id"], project_id=self.pid, gespraech_id=self.gid)["result"]
        self.assertEqual(st["auftraege"][0]["auftrag_id"], aid)
        self.assertEqual([(s["schritt"], s["zustand"]) for s in st["auftraege"][0]["schritte"]],
                         [("Alt-Texte generieren", "läuft"), ("Als Word herunterladen", "wartet")])
        self.assertTrue(st["auftraege"][0]["projekt_daten"].startswith(DATEN_MARKE))
        self.assertIn({"project_id": self.pid, "projekt": "Vereinsbrief (fiktiv)",
                       "was": "Auftrag, Schritt 1 von 2: Alt-Texte generieren",
                       "auftrag": {"schritt": 1, "von": 2, "titel": "Alt-Texte generieren"}}, konto.laufende(self.u["id"]))

    def test_lauf_abbrechen_im_konto_haelt_den_auftrag_an(self):
        from inkluagent import konto
        aid = self.erteilen("alt_texte", "word")
        ex = konto.KontoExecutor(self.u["id"])
        with mock.patch.object(konto, "aendern_an", return_value=True), \
                mock.patch.object(konto.KontoExecutor, "_endpunkt", return_value={"angefordert": True}):
            r = ex.execute("lauf_abbrechen", {"project_id": self.pid})
        self.assertIn("Auftrag", r["result"]["abgebrochen"])
        self.assertEqual(self.auftrag(aid)["abbrechen"], 1)


# ─── 6. Melden ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Melden(MitWerkzeugen):
    def bis_zum_ende(self, per_mail=False):
        aid = self.erteilen("alt_texte", "word", per_mail=per_mail)
        af.durchlauf()
        self.fertig_machen()
        af.durchlauf()
        self.assertEqual(self.auftrag(aid)["zustand"], "fertig")
        return aid

    def test_wartet_bis_zum_ende_meldet_mit_dateien(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        sql("UPDATE agent_rueckmeldungen SET angelegt_am = datetime('now', '-60 seconds')")
        self.assertEqual(rm.durchlauf(), 0, "der Auftrag laeuft noch")
        self.fertig_machen()
        af.durchlauf()
        self.assertEqual(rm.durchlauf(), 1)
        n = storage.konto_verlauf(self.u["id"], gespraech_id=self.gid)[-1]
        self.assertEqual(n["text"], self.modell.text)
        self.assertEqual([a.get("dateiname") or a.get("text") for a in n["anhang"]],
                         ["brief-fiktiv.docx", "Projekt „Vereinsbrief (fiktiv)“ öffnen"], "Dateien zuerst, dann das Projekt")
        eingabe = self.modell.aufrufe[0][1]["content"]
        self.assertIn("Was lief: Dein Auftrag", eingabe)
        self.assertIn('"schritt": "Als Word herunterladen"', eingabe)
        self.assertIn('"datei": "brief-fiktiv.docx"', eingabe)
        self.assertIn('"alt_texte": {"bilder": 3', eingabe, "dieselben Zahlen wie bei der Alt-Text-Generierung")
        self.assertNotIn(rm.NEUSTART_HINWEIS, eingabe)
        self.assertEqual(sql("SELECT zustand, auftrag_id FROM agent_rueckmeldungen"), [{"zustand": "gemeldet", "auftrag_id": aid}])

    def test_gescheiterter_auftrag_meldet_sich_ehrlich(self):
        self.word.preis = 99
        self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.fertig_machen()
        af.durchlauf()
        sql("UPDATE agent_rueckmeldungen SET angelegt_am = datetime('now', '-60 seconds')")
        self.assertEqual(rm.durchlauf(), 1)
        eingabe = self.modell.aufrufe[0][1]["content"]
        self.assertIn("angehalten, weil ein Schritt nicht ging", eingabe)
        self.assertIn("würde jetzt 99 Credits kosten", eingabe)

    def test_mail_auf_wunsch(self):
        self.bis_zum_ende(per_mail=True)
        sql("UPDATE agent_rueckmeldungen SET angelegt_am = datetime('now', '-60 seconds')")
        rm.durchlauf()
        main.send_email.assert_called_once()
        self.assertEqual(main.send_email.call_args.args[0], self.u["email"], "nur an die Adresse des eigenen Kontos")
        self.assertEqual(storage.konto_verlauf(self.u["id"], gespraech_id=self.gid)[-1]["zusatz"], rm.T_MAIL_GING)

    def test_nie_zusammengelegt_und_ohne_auftrag_keine_auftragsmeldung(self):
        c = database.get_db()
        try:
            self.assertTrue(rm.anmelden_in(c, self.u["id"], self.gid, self.pid, "auftrag", auftrag_id=901))
            self.assertTrue(rm.anmelden_in(c, self.u["id"], self.gid, self.pid, "auftrag", auftrag_id=902))
            self.assertFalse(rm.anmelden_in(c, self.u["id"], self.gid, self.pid, "auftrag"), "Art auftrag nur mit Auftrag")
            self.assertFalse(rm.anmelden_in(c, self.u["id"], self.gid, self.pid, "alt_texte", auftrag_id=903))
            c.commit()
        finally:
            c.close()
        rm.anmelden(self.u["id"], self.gid, self.pid, "alt_texte")
        self.assertEqual(sorted((x["art"], x["auftrag_id"]) for x in sql("SELECT * FROM agent_rueckmeldungen")),
                         [("alt_texte", None), ("auftrag", 901), ("auftrag", 902)])

    def test_keine_zwoelf_stunden_fuer_auftraege(self):
        self.erteilen("alt_texte", "word")
        af.durchlauf()
        sql("UPDATE agent_rueckmeldungen SET angelegt_am = datetime('now', '-13 hours')")
        self.assertEqual(rm.durchlauf(), 0)
        self.assertEqual(sql("SELECT zustand FROM agent_rueckmeldungen"), [{"zustand": "wartet"}])

    def test_melden_wenn_fertig_bei_laufendem_auftrag(self):
        self.erteilen("alt_texte", "word")
        af.durchlauf()
        r = rm.ausfuehren(rm.WERKZEUG_MELDEN, {"per_mail": True}, user_id=self.u["id"], project_id=self.pid,
                          gespraech_id=self.gid, sprache="de")
        self.assertEqual(r["result"]["angemeldet"], ["Dein Auftrag"])
        z = sql("SELECT art, per_mail FROM agent_rueckmeldungen")
        self.assertEqual(z, [{"art": "auftrag", "per_mail": 1}], "keine zweite Rueckmeldung fuer den Schritt darin")

    def test_kontoloeschung_nimmt_auftraege_mit(self):
        self.erteilen("alt_texte", "word")
        database.delete_user_data(self.u["id"])
        self.assertEqual(sql("SELECT * FROM agent_auftraege WHERE user_id = ?", self.u["id"]), [])
        self.assertEqual(af.durchlauf(), 0)


# ─── 7. Agent-Schleife, Karte ueber den Server, Oberflaeche, Hilfe ──────────────────────────────────────────────────────────
class Agent:
    kennt_frist = False
    kann_streamen = False

    def __init__(self, gesehen):
        self.gesehen = gesehen

    def invoke_with_tools(self, anthropic_messages, tools, system, **kw):
        self.gesehen["tools"] = {t["name"] for t in tools}
        self.gesehen["system"] = system
        return {"content": [{"type": "text", "text": "Gern (fiktiv)."}], "stop_reason": "end_turn"}


@unittest.skipIf(main is None, "main nicht ladbar")
class Schleife(Basis):
    def lauf(self, mit_lauf=True):
        g = {}
        p = af._eigenes_projekt(self.u["id"], self.pid)
        lauf = begleiter.Lauf(ui_sprache="de", nutzer_text="Mach alles fertig") if mit_lauf else None
        agent_loop.run_agent(self.pid, self.u["id"], "Mach alles fertig", p, Agent(g), konto=True, begleiter_lauf=lauf,
                             gespraech_id=self.gid if mit_lauf else None)
        return g

    def test_werkzeuge_mit_allen_schaltern_nie_in_der_demo(self):
        g = self.lauf()
        self.assertTrue(af.NAMEN <= g["tools"])
        self.assertIn(af.PROMPT, g["system"])
        self.assertFalse(af.NAMEN & self.lauf(mit_lauf=False)["tools"], "nie in der Demo")
        with mock.patch.object(funktionen, "AGENT_KARTEN", False):
            self.assertFalse(af.NAMEN & self.lauf()["tools"], "ohne Karten kein Auftrag")
        with mock.patch.object(funktionen, "AGENT_AUFTRAEGE", False):
            self.assertFalse(af.NAMEN & self.lauf()["tools"])

    def test_ueber_den_knopf_der_karte(self):
        from fastapi.testclient import TestClient
        mock.patch.object(main, "INKLUAGENT_ENABLED", True).start()
        k = af.ausfuehren(af.WERKZEUG_VORBEREITEN, {"schritte": ["alt_texte"]}, user_id=self.u["id"], project_id=self.pid,
                          gespraech_id=self.gid, sprache="de")["anhang"]
        storage.konto_anhaengen(self.u["id"], self.pid, "assistant", "Hier ist dein Auftrag (fiktiv).", anhang=[k],
                                gespraech_id=self.gid)
        c = TestClient(main.app)
        c.cookies.set("token", main.create_token(self.u["id"], self.u["email"], 0))
        r = c.post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": {"per_mail": True}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["karte"]["zustand"], "gesendet")
        self.assertEqual(sql("SELECT per_mail FROM agent_rueckmeldungen"), [{"per_mail": 1}], "der Haken der Karte zaehlt")
        zweiter = c.post("/api/agent/karte", json={"karten_id": k["karten_id"], "werte": {}})
        self.assertEqual(zweiter.status_code, 409, "genau ein Auftrag je Karte")
        self.assertEqual(len(sql("SELECT * FROM agent_auftraege")), 1)
        verlauf = storage.konto_verlauf(self.u["id"], gespraech_id=self.gid)
        self.assertEqual(verlauf[-1]["anhang"][0]["zustand"], "gesendet", "nach dem Neuladen nicht wieder offen")


# ─── 8. Unabhaengige Pruefung 14.10.2026: je Befund ein Test ────────────────────────────────────────────────────────────────
class Befunde(MitWerkzeugen):
    def test_m1_schritt_fasst_angebote_und_karten_des_gespraechs_nicht_an(self):
        uid, pid = self.u["id"], self.pid
        gespraech = {}

        def im_gespraech():   # waehrend der Schritt laeuft: ein neues Angebot mit Karte und ein verbrauchtes (anderer Faden)
            gespraech["neu"] = ausgaben._angebot_merken((uid, pid, "uebersetzung:en", None), 7, "runde-gespraech")
            ausgaben.angebot_nach_id(gespraech["neu"])[1].update(werkzeug="uebersetze_dokument", args={"zielsprache": "en"})
            with ausgaben._SPERRE:
                ausgaben._VERBRAUCHT.setdefault((uid, pid), []).append(("frueher-fiktiv", {"id": "frueher-fiktiv"}, True))
        self.word.nebenher = im_gespraech
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.fertig_machen()
        try:
            af.durchlauf()
            self.assertEqual(self.auftrag(aid)["zustand"], "fertig")
            neu = ausgaben.angebot_nach_id(gespraech["neu"])[1]
            self.assertEqual((neu["werkzeug"], neu["args"]), ("uebersetze_dokument", {"zielsprache": "en"}),
                             "die Karte des Gespraechs fuehrt weiter genau sein Angebot aus")
            self.assertIn("frueher-fiktiv", [x[0] for x in ausgaben._VERBRAUCHT.get((uid, pid), [])],
                          "verbrauchte Angebote des Gespraechs bleiben dem Gespraech")
            with ausgaben.auftrag_schritt({"docx"}, 10):
                self.assertEqual(ausgaben._angebot_merken((uid, pid, "docx", None), 5, "t"), "", "im Schritt entsteht kein Angebot")
        finally:
            with ausgaben._SPERRE:
                ausgaben._VERBRAUCHT.pop((uid, pid), None)

    def test_m2_uebersetzung_bucht_nie_mehr_als_die_vorschau(self):
        """Zehn Pakete zu je 150 Woertern: Vorschau 15 Credits — vorher wurden je Paket 2, also 20 gebucht."""
        import uebersetzung_api
        from uebersetzung import SegmentErgebnis
        c = database.get_db()
        did = c.execute("SELECT id FROM documents WHERE project_id = ?", (self.pid,)).fetchone()[0]
        for i in range(300):
            c.execute("INSERT INTO uebersetzung_segmente (project_id, document_id, position, anker, woerter, stuecke, original) "
                      "VALUES (?, ?, ?, ?, 5, ?, ?)", (self.pid, did, i, f"w/p{i}|0", json.dumps(["eins zwei drei vier fünf"]),
                                                       "eins zwei drei vier fünf"))
        c.commit()
        v = uebersetzung_api._vorschau(c, self.u["id"], self.pid, None, "en", True)
        c.close()
        self.assertEqual((v["woerter"], v["preis"]), (1500, 15))
        gebucht = []

        def paket(batch, *a, **k):
            return [SegmentErgebnis(nr=nr, ziel_stuecke={0: "one two three four five"}, status="fertig") for nr, _s in batch]
        uebersetzung_api._lauf.pop(self.pid, None)
        try:
            with mock.patch.object(uebersetzung_api.ue, "uebersetze_batch", side_effect=paket), \
                    mock.patch.object(uebersetzung_api._d.billing, "verbuche", side_effect=lambda *a, **k: gebucht.append(k["credits"])):
                asyncio.run(uebersetzung_api._uebersetze_projekt(self.pid, self.u["id"],
                                                                 {"zielsprache": "en", "alt_texte": True, "sprache_setzen": True}, None))
            st = uebersetzung_api._lauf[self.pid]
            self.assertEqual((st["pakete_gesamt"], st["segmente_fertig"]), (10, 300))
            self.assertEqual((sum(gebucht), st["credits"]), (15, 15), gebucht)
        finally:
            uebersetzung_api._lauf.pop(self.pid, None)

    def test_m3_alt_texte_ohne_neuen_text_halten_an(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        # Rettung im Sammellauf: KI fuer alle Bilder ausgefallen, der alte Text bleibt (Status done), Fehlergrund gesetzt
        sql("UPDATE images SET fehler_grund = 'KI nicht erreichbar (fiktiv)', needs_review = 1 WHERE project_id = ?", self.pid)
        self.fertig_machen()
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual([s["zustand"] for s in a["schritte"]], ["fehler", "ausgelassen"])
        self.assertEqual(a["schritte"][0]["grund"], af.T_KEIN_ALT_TEXT)
        self.assertEqual(self.word.aufrufe, [], "kein bezahltes Herunterladen danach")

    def test_m3_auswertung_der_anderen_schritte(self):
        import formular_api
        import tagging_api
        import uebersetzung_api
        a = {"user_id": self.u["id"], "project_id": self.pid, "id": 0}
        for st, erwartet in (({"felder_neu": 0, "fehler": ["Seite 1: Guthaben (fiktiv)"]}, "fehler"),
                             ({"felder_neu": 3, "fehler": ["Seite 4 (fiktiv)"]}, "teilweise"),
                             ({"felder_neu": 0, "fehler": []}, "fehler"), ({"felder_neu": 4, "fehler": []}, "fertig")):
            with self.subTest(st=st), mock.patch.dict(formular_api._generierung, {self.pid: st}):
                self.assertEqual(af._auswerten(a, {"art": "quickinfos"})["zustand"], erwartet)
        did = sql("SELECT id FROM documents WHERE project_id = ?", self.pid)[0]["id"]
        sql("UPDATE documents SET tagging_status = ?, getaggt = 1 WHERE id = ?", tagging_api.STATUS_FERTIG, did)
        self.assertEqual(af._auswerten(a, {"art": "tagging", "document_id": did}), {"zustand": "fertig"})
        sql("UPDATE documents SET tagging_status = ?, getaggt = 0, tagging_bericht = ? WHERE id = ?", tagging_api.STATUS_FEHLER,
            json.dumps({"fehler": "PDFix stieg aus (fiktiv)"}), did)
        self.assertEqual(af._auswerten(a, {"art": "tagging", "document_id": did}), {"zustand": "fehler", "grund": "PDFix stieg aus (fiktiv)"})
        sql("UPDATE documents SET pruefung_status = ? WHERE id = ?", tagging_api.STATUS_FERTIG, did)
        self.assertEqual(af._auswerten(a, {"art": "pruefung", "document_id": did}), {"zustand": "fertig"})
        gesehen = []

        def export(p, u):   # die uebersetzte Datei: kostenlos, im Faden des Schritts (ohne Freigabe, ohne Karten des Gespraechs)
            gesehen.append(getattr(ausgaben._AUFTRAG, "f", None))
            return {"ok": True, "result": {"dateiname": "brief-en.docx"},
                    "anhang": {"art": "docx", "dateiname": "brief-en.docx", "download_url": "/api/sofort/fiktiv"}}
        with mock.patch.dict(uebersetzung_api._lauf, {self.pid: {"laeuft": False, "segmente_fertig": 9, "fehler": []}}), \
                mock.patch.object(ausgaben, "exportiere_uebersetzung", side_effect=export):
            e = af._auswerten(a, {"art": "uebersetzung"})
        self.assertEqual((e["zustand"], e["anhang"]["dateiname"]), ("fertig", "brief-en.docx"))
        self.assertEqual(gesehen[0]["arten"], frozenset(), "keine Freigabe fuer irgendetwas")
        with mock.patch.dict(uebersetzung_api._lauf, {self.pid: {"laeuft": False, "segmente_fertig": 0, "fehler": ["Paket weg (fiktiv)"]}}):
            self.assertEqual(af._auswerten(a, {"art": "uebersetzung"}), {"zustand": "fehler", "grund": "Paket weg (fiktiv)"})

    def test_m4_nach_zwei_tagen_aufgegeben_und_gemeldet(self):
        aid = self.erteilen("alt_texte", "word")
        af.durchlauf()
        alt = self.auftrag(aid)   # der Stand eines Abarbeiters, der in einem Werkzeug haengt
        sql("UPDATE agent_rueckmeldungen SET angelegt_am = datetime('now', '-49 hours')")
        self.assertEqual(rm.durchlauf(), 1)
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["zustand"], a["schritte"][0]["grund"]), ("gescheitert", "fehler", af.T_AUFGEGEBEN))
        self.assertEqual(sql("SELECT zustand FROM agent_rueckmeldungen"), [{"zustand": "gemeldet"}])
        self.assertFalse(af._speichern(alt), "kehrt er doch noch zurueck, schreibt er den Stand nicht mehr um")
        self.assertEqual(self.auftrag(aid)["schritte"][0]["zustand"], "fehler")

    def test_m4_schalter_aus_und_aufraeumen_scheitert(self):
        aid = self.erteilen("alt_texte", "word")
        self.assertEqual(af.ausgeschaltet(), 1)
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["grund"]), ("gescheitert", af.T_AUS))

        async def start():
            with mock.patch.object(af, "nach_neustart", side_effect=RuntimeError("Datenbank gesperrt (fiktiv)")):
                af.starten()
            laeuft = af._AUFGABE is not None and not af._AUFGABE.done()
            af._AUFGABE.cancel()
            return laeuft
        try:
            self.assertTrue(asyncio.run(start()), "die Schleife startet trotzdem")
        finally:
            af._AUFGABE, af._LOOP, af._WECKER = None, None, None

    def test_n1_zwei_abarbeiter_starten_nie_denselben_schritt(self):
        aid = self.erteilen("alt_texte", "word")
        erster, zweiter = self.auftrag(aid), self.auftrag(aid)
        for a in (erster, zweiter):
            a["schritte"][0].update(zustand="laeuft", gestartet=False)
        self.assertTrue(af._beanspruchen(erster))
        self.assertFalse(af._beanspruchen(zweiter), "der zweite las einen alten Stand")
        with mock.patch.object(af, "_laden", side_effect=lambda i: zweiter if i == aid else None):
            af.weiter(aid)
        self.assertEqual(self.alt.aufrufe, [], "und startet deshalb nichts")

    def test_n2_schon_getaggt_und_seitengrenze(self):
        import pdf_tagging
        import tagging_api
        from inkluagent.tools import pdf as pdf_tools
        self.pid = projekt(self.u["id"], PDF)
        mock.patch.object(tagging_api, "_seiten", return_value=2).start()
        mock.patch.object(tagging_api, "lesbar_grund", return_value=None).start()
        mock.patch.object(pdf_tagging, "verfuegbar", return_value=True).start()
        mock.patch.object(funktionen, "TAGGING_PROFESSIONELL", True).start()
        tagging = Werkzeug("tagging", 40)
        mock.patch.object(pdf_tools, "barrierefrei_machen", tagging).start()
        aid = self.erteilen("tagging", "alt_texte")
        sql("UPDATE documents SET getaggt = 1 WHERE project_id = ?", self.pid)   # inzwischen per Knopf getaggt
        af.durchlauf()
        s = self.schritt(aid, 0)
        self.assertEqual((s["zustand"], s["grund"], s["preis"]), ("uebersprungen", af.T_SCHON_GETAGGT, 0))
        self.assertEqual(tagging.aufrufe, [], "nicht noch einmal taggen und bezahlen")
        neu = projekt(self.u["id"], PDF)
        with mock.patch.object(pdf_tagging, "MAX_SEITEN", 1):
            plan = af.planen(self.u["id"], af._eigenes_projekt(self.u["id"], neu), self.wunsch("tagging"))
        self.assertEqual(plan["eintraege"], [])
        self.assertTrue(any("geht bis 1 Seiten" in f for f in plan["fehler"]), plan["fehler"])

    def test_n3_herunterladen_belegt_oder_gedrosselt_wartet(self):
        aid = self.erteilen("word")
        main._export_nutzer.add(self.u["id"])   # fuer das Konto wird gerade eine PDF gebaut
        try:
            af.durchlauf()
            self.assertEqual((self.schritt(aid, 0)["zustand"], self.word.aufrufe), ("offen", []))
        finally:
            main._export_nutzer.discard(self.u["id"])
        self.word.ergebnis = {"ok": False, "error": main._export_belegt_text("drossel")}
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual((a["zustand"], a["schritte"][0]["zustand"], a["verbraucht"], len(self.word.aufrufe)), ("laeuft", "offen", 0, 1))
        af.durchlauf()
        self.assertEqual(len(self.word.aufrufe), 1, "Pause: nicht gleich noch einmal")
        self.word.ergebnis = None
        a = self.auftrag(aid)
        a["schritte"][0]["pause_bis"] = 0
        af._speichern(a)
        af.durchlauf()
        self.assertEqual(self.auftrag(aid)["zustand"], "fertig")

    def test_n4_die_karte_zeigte_ein_anderes_dokument(self):
        k = self.kontext("alt_texte", "word")
        k["fingerabdruck"][0][1] = 999999
        status, antwort = af.erteilen(k, self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual((status, antwort["karte"]["zustand"]), (409, "abgelaufen"))
        self.assertEqual(sql("SELECT * FROM agent_auftraege"), [])

    def test_n5_schlichte_meldung_mit_dem_ergebnis_je_schritt(self):
        self.word.preis = 99
        self.erteilen("alt_texte", "word")
        af.durchlauf()
        self.fertig_machen()
        af.durchlauf()
        sql("UPDATE agent_rueckmeldungen SET angelegt_am = datetime('now', '-60 seconds')")
        with mock.patch.object(rm, "text_vom_agenten", return_value=None):
            self.assertEqual(rm.durchlauf(), 1)
        self.assertEqual(storage.konto_verlauf(self.u["id"], gespraech_id=self.gid)[-1]["text"],
                         "Dein Auftrag im Projekt „Vereinsbrief (fiktiv)“ ist zu Ende. Das Ergebnis siehst du im Projekt. "
                         "Alt-Texte generieren: fertig. Als Word herunterladen: gescheitert.")

    def test_n6_erst_die_wartende_rueckmeldung_aus_baustein_1(self):
        aid = self.erteilen("alt_texte", "word")
        sql("INSERT INTO agent_rueckmeldungen (user_id, gespraech_id, project_id, art, sprache) VALUES (?, ?, ?, 'verarbeitung', 'de')",
            self.u["id"], self.gid, self.pid)
        af.durchlauf()
        self.assertEqual((self.schritt(aid, 0)["zustand"], self.alt.aufrufe), ("offen", []))
        sql("UPDATE agent_rueckmeldungen SET zustand = 'gemeldet' WHERE art = 'verarbeitung'")
        af.durchlauf()
        self.assertEqual(self.schritt(aid, 0)["zustand"], "laeuft")

    def test_n7_hinweise_als_eigener_absatz(self):
        sql("UPDATE images SET alt_text_edited = 'Eigener Text (fiktiv)' WHERE project_id = ?", self.pid)
        k = af.ausfuehren(af.WERKZEUG_VORBEREITEN, {"schritte": ["alt_texte"]}, user_id=self.u["id"], project_id=self.pid,
                          gespraech_id=self.gid, sprache="de")["anhang"]
        self.assertEqual(k["zeilen"], ["Alt-Texte generieren (3 Bilder): 15 Credits"])
        self.assertEqual(k["hinweise"], ["Alt-Texte generieren überschreibt 3 eigene Texte."])

    def test_n8_ueberblick_in_der_sprache_der_oberflaeche(self):
        from fastapi.testclient import TestClient
        mock.patch.object(main, "INKLUAGENT_ENABLED", True).start()
        mock.patch.object(funktionen, "AGENT_ARBEITSFENSTER", True).start()
        self.erteilen("alt_texte", "word")
        sql("UPDATE users SET language = 'en' WHERE id = ?", self.u["id"])
        c = TestClient(main.app)
        c.cookies.set("token", main.create_token(self.u["id"], self.u["email"], 0))
        laeuft = c.get("/api/agent/ueberblick").json()["laeuft"]
        self.assertTrue(any(x["was"].startswith("Task, step 1 of 2: ") for x in laeuft), laeuft)

    def test_n9_projekt_loeschen_und_aufbewahrung(self):
        from fastapi.testclient import TestClient
        self.erteilen("alt_texte", "word")
        c = TestClient(main.app)
        c.cookies.set("token", main.create_token(self.u["id"], self.u["email"], 0))
        self.assertEqual(c.delete(f"/api/projects/{self.pid}").status_code, 200)
        self.assertEqual(sql("SELECT * FROM agent_auftraege WHERE project_id = ?", self.pid), [])
        self.pid = projekt(self.u["id"])
        alt = self.erteilen("alt_texte")
        af.ausgeschaltet()
        sql("UPDATE agent_auftraege SET beendet_am = datetime('now', '-31 days') WHERE id = ?", alt)
        af._AUFGERAEUMT_AM = 0
        af._aufraeumen()
        self.assertTrue(self.auftrag(alt), "seine Rueckmeldung wartet noch: er bleibt")
        sql("UPDATE agent_rueckmeldungen SET zustand = 'gemeldet' WHERE auftrag_id = ?", alt)
        af._AUFGERAEUMT_AM = 0
        af._aufraeumen()
        self.assertIsNone(self.auftrag(alt), "nach 30 Tagen geloescht")

    def test_n10_frist_der_dateien_in_den_ergebnissen(self):
        aid = self.erteilen("word")
        af.durchlauf()
        a = self.auftrag(aid)
        self.assertEqual(a["zustand"], "fertig")
        a["schritte"][0]["anhang"]["gueltig_bis"] = "2026-10-15T14:00:00Z"
        sql("UPDATE agent_auftraege SET schritte = ? WHERE id = ?", json.dumps(a["schritte"]), aid)
        self.assertEqual(af.fakten(aid)["auftrag"]["schritte"][0]["datei_herunterladbar_bis"], "2026-10-15T14:00:00Z")

    def test_gleichzeitige_klicks_legen_genau_einen_an(self):
        k = self.kontext("alt_texte", "word")
        start, ergebnisse = threading.Barrier(4), []

        def klick():
            start.wait()
            ergebnisse.append(af.erteilen(dict(k), self.u, {"per_mail": False}, lambda s: s)[0])
        faeden = [threading.Thread(target=klick) for _i in range(4)]
        for f in faeden:
            f.start()
        for f in faeden:
            f.join()
        self.assertEqual(sorted(ergebnisse), [200, 409, 409, 409])
        self.assertEqual(len(sql("SELECT * FROM agent_auftraege")), 1)

    def test_mehrere_dokumente_taggen_und_pruefen(self):
        import pdf_tagging
        import tagging_api
        self.pid = projekt(self.u["id"], PDF)
        sql("INSERT INTO documents (project_id, doc_index, original_filename, original_path, getaggt) VALUES (?, 2, 'anlage.pdf', "
            "'/tmp/fiktiv/anlage.pdf', 0)", self.pid)
        mock.patch.object(tagging_api, "_seiten", return_value=3).start()
        mock.patch.object(tagging_api, "lesbar_grund", return_value=None).start()
        mock.patch.object(pdf_tagging, "verfuegbar", return_value=True).start()
        with mock.patch.object(funktionen, "KI_PRUEFUNG", True), mock.patch.object(funktionen, "TAGGING_PROFESSIONELL", True):
            plan = af.planen(self.u["id"], af._eigenes_projekt(self.u["id"], self.pid), self.wunsch("pruefung", "tagging"))
        self.assertEqual([(e["art"], e["preis_geplant"]) for e in plan["eintraege"]],
                         [("tagging", 60), ("tagging", 60), ("pruefung", 6), ("pruefung", 6)])
        self.assertEqual(len({e["document_id"] for e in plan["eintraege"]}), 2)


# ─── 9. Michael 15.10.2026: „Mach das Dokument 2 barrierefrei, erzeug die Alt-Texte und gib mir dann die fertige PDF“ ───────
# Der Agent bot nur das Tagging an (der PDF-Prompt sagte „Schritt fuer Schritt“), und die Rueckmeldung danach sagte „sieh dir
# die Struktur an“ statt weiterzumachen. Jetzt: mehrere Schritte = ein Auftrag (Prompt nach Schalter), auch fuer EIN Dokument.
@unittest.skipIf(main is None, "main nicht ladbar")
class EinDokument(Basis):
    def setUp(self):
        super().setUp()
        import pdf_tagging
        import tagging_api
        self.pid = projekt(self.u["id"], PDF, bilder=3)
        self.erstes = sql("SELECT id FROM documents WHERE project_id = ?", self.pid)[0]["id"]
        c = database.get_db()
        self.zweites = c.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path, getaggt) "
                                 "VALUES (?, 2, 'anlage-fiktiv.pdf', '/tmp/fiktiv/anlage.pdf', 0)", (self.pid,)).lastrowid
        for i in range(2):
            c.execute("INSERT INTO images (project_id, document_id, page_number, image_index, image_path, status) "
                      "VALUES (?, ?, 1, ?, '', 'done')", (self.pid, self.zweites, i))
        c.commit()
        c.close()
        mock.patch.object(tagging_api, "_seiten", return_value=5).start()
        mock.patch.object(tagging_api, "lesbar_grund", return_value=None).start()
        mock.patch.object(pdf_tagging, "verfuegbar", return_value=True).start()
        mock.patch.object(funktionen, "TAGGING_PROFESSIONELL", True).start()

    def p(self):
        return af._eigenes_projekt(self.u["id"], self.pid)

    def test_wunsch_liest_die_dokument_kennung(self):
        for wert, erwartet in ((None, None), ("", None), (7, 7), ("7", 7), (7.0, 7), (" 12 ", 12), (True, -1), (False, -1),
                               ("Dokument 2", -1), (0, -1), (-3, -1), (2.5, -1), ([2], -1)):
            with self.subTest(wert=wert):
                self.assertEqual(af.wunsch_aus({"schritte": ["pdf"], "document_id": wert})["document_id"], erwartet)
        self.assertIsNone(af.wunsch_aus({"schritte": ["pdf"]})["document_id"], "ohne: alle Dokumente wie bisher")

    def test_nur_dieses_dokument_taggen_beschreiben_herunterladen(self):
        plan = af.planen(self.u["id"], self.p(), self.wunsch("pdf", "alt_texte", "tagging", document_id=self.zweites))
        self.assertEqual(plan["fehler"], [])
        self.assertEqual([(e["art"], e["document_id"], e["args"]) for e in plan["eintraege"]],
                         [("tagging", self.zweites, {"document_id": self.zweites}),
                          ("alt_texte", self.zweites, {"document_id": self.zweites}),
                          ("pdf", self.zweites, {"document_id": self.zweites})])
        self.assertEqual(plan["eintraege"][1]["umfang"], "2 Bilder", "nur die Bilder dieses Dokuments")
        self.assertEqual(plan["eintraege"][2]["umfang"], "ein Dokument", "kein ZIP aller Dokumente")
        self.assertEqual(plan["dokument"], "anlage-fiktiv.pdf")

    def test_ohne_dokument_wie_bisher_alle(self):
        plan = af.planen(self.u["id"], self.p(), self.wunsch("tagging", "alt_texte", "pdf"))
        self.assertEqual(plan["fehler"], [])
        self.assertEqual([e["art"] for e in plan["eintraege"]], ["tagging", "tagging", "alt_texte", "pdf"])
        self.assertEqual(plan["eintraege"][2]["args"], {})
        self.assertIsNone(plan["eintraege"][2]["document_id"])
        self.assertEqual(plan["eintraege"][2]["umfang"], "5 Bilder")
        self.assertEqual(plan["eintraege"][3]["args"], {"alle": True})
        self.assertIsNone(plan["dokument"])

    def test_schon_getaggt_faellt_weg_mit_namen(self):
        sql("UPDATE documents SET getaggt = 1 WHERE id = ?", self.zweites)
        plan = af.planen(self.u["id"], self.p(), self.wunsch("tagging", "alt_texte", "pdf", document_id=self.zweites))
        self.assertEqual(plan["fehler"], [])
        self.assertEqual([e["art"] for e in plan["eintraege"]], ["alt_texte", "pdf"])
        self.assertIn("Barrierefrei machen entfällt: „anlage-fiktiv.pdf“ ist schon getaggt.", plan["hinweise"])

    def test_fremdes_oder_unbekanntes_dokument(self):
        anderer = nutzer()
        fremd = sql("SELECT id FROM documents WHERE project_id = ?", projekt(anderer["id"], PDF))[0]["id"]
        for dok in (fremd, 999999, "Dokument 2"):
            with self.subTest(dok=dok):
                plan = af.planen(self.u["id"], self.p(), self.wunsch("alt_texte", "pdf", document_id=dok))
                self.assertEqual(plan["eintraege"], [], "kein halber Plan und nie das ganze Projekt")
                self.assertTrue(any("Dieses Dokument gibt es in diesem Projekt nicht" in f for f in plan["fehler"]), plan["fehler"])

    def test_schritte_nur_fuers_ganze_projekt(self):
        with mock.patch.object(af, "moeglich", return_value=["alt_texte", "alt_texte_tabelle", "pdf"]):
            plan = af.planen(self.u["id"], self.p(), self.wunsch("alt_texte", "alt_texte_tabelle", document_id=self.zweites))
        self.assertEqual(plan["eintraege"], [])
        self.assertIn("„Alt-Texte herunterladen“ geht im Auftrag nur für das ganze Projekt, nicht für ein einzelnes Dokument.",
                      plan["fehler"])
        # Hat das Projekt nur dieses eine Dokument, ist es dasselbe wie das ganze Projekt
        einzeln = projekt(self.u["id"], WORD)
        dok = sql("SELECT id FROM documents WHERE project_id = ?", einzeln)[0]["id"]
        plan = af.planen(self.u["id"], af._eigenes_projekt(self.u["id"], einzeln), self.wunsch("alt_texte", "word", document_id=dok))
        self.assertEqual(plan["fehler"], [])
        self.assertEqual([e["art"] for e in plan["eintraege"]], ["alt_texte", "word"])

    def test_karte_nennt_das_dokument_und_der_knopf_rechnet_gleich(self):
        r = af.ausfuehren(af.WERKZEUG_VORBEREITEN, {"schritte": ["alt_texte"], "document_id": self.zweites}, user_id=self.u["id"],
                          project_id=self.pid, gespraech_id=self.gid, sprache="de", auftrag="Mach Dokument 2 fertig (fiktiv)")
        self.assertTrue(r["ok"], r)
        k = r["anhang"]
        self.assertIn("Projekt „Vereinsbrief (fiktiv)“, Dokument „anlage-fiktiv.pdf“. Zusammen 10 Credits.", k["text"])
        self.assertEqual(k["zeilen"], ["Alt-Texte generieren (2 Bilder): 10 Credits"])
        kontext = karten._KARTEN[k["karten_id"]]["kontext"]
        self.assertEqual(kontext["wunsch"]["document_id"], self.zweites, "der Knopf rechnet mit demselben Dokument")
        status, antwort = af.erteilen(kontext, self.u, {"per_mail": False}, lambda s: s)
        self.assertEqual(status, 200, antwort)
        s = af._laden(antwort["auftrag_id"])["schritte"][0]
        self.assertEqual((s["art"], s["document_id"], s["args"]), ("alt_texte", self.zweites, {"document_id": self.zweites}))

    def test_auswertung_zaehlt_nur_die_bilder_des_dokuments(self):
        sql("UPDATE images SET status = 'pending' WHERE document_id = ?", self.erstes)
        a = {"user_id": self.u["id"], "project_id": self.pid, "id": 0}
        self.assertEqual(af._auswerten(a, {"art": "alt_texte", "document_id": self.zweites}), {"zustand": "fertig"})
        self.assertEqual(af._auswerten(a, {"art": "alt_texte", "document_id": None})["zustand"], "teilweise",
                         "ohne Dokument zaehlt das ganze Projekt wie bisher")

    def test_werkzeug_nimmt_document_id(self):
        props = next(d for d in af.werkzeuge(self.pid) if d["name"] == af.WERKZEUG_VORBEREITEN)["input_schema"]["properties"]
        self.assertEqual(props["document_id"]["type"], "integer")
        self.assertNotIn("enum", props["document_id"], "keine Zahlen-Auswahlliste (Gemini lehnte sonst jede Anfrage ab)")
        self.assertIn("ist ein Auftrag mit den Schritten tagging, alt_texte und pdf", af.prompt("pdf"))
        self.assertNotIn("barrierefrei_machen", af.prompt("word"), "das Werkzeug gibt es nur im PDF-Projekt")
        self.assertNotIn("barrierefrei_machen", af.prompt("konto"))


@unittest.skipIf(main is None, "main nicht ladbar")
class PromptNachSchalter(Basis):
    """Der PDF-Prompt sagt mit Auftraegen „mehrere Schritte = ein Auftrag“, ohne sie wie bisher „Schritt fuer Schritt“ — der
    Schalter entscheidet im Code, kein „ausser wenn“ im Text."""

    def lauf(self, mit_lauf=True):
        g = {}
        pid = projekt(self.u["id"], PDF)
        p = af._eigenes_projekt(self.u["id"], pid)
        lauf = begleiter.Lauf(ui_sprache="de", nutzer_text="Mach die PDF barrierefrei, erzeug die Alt-Texte und gib mir die PDF") \
            if mit_lauf else None
        agent_loop.run_agent(pid, self.u["id"], "Mach die PDF barrierefrei, erzeug die Alt-Texte und gib mir die PDF", p, Agent(g),
                             konto=True, begleiter_lauf=lauf, gespraech_id=self.gid if mit_lauf else None)
        return g

    def test_mit_auftraegen_ein_auftrag_statt_schritt_fuer_schritt(self):
        from inkluagent.prompts import system_pdf
        system = self.lauf()["system"]
        self.assertIn(system_pdf._SCHRITT2_AUFTRAG, system)
        self.assertIn(system_pdf._SCHLUSS_AUFTRAG, system)
        self.assertNotIn("Schritt für Schritt", system)
        self.assertNotIn("frage für die nächste neu", system)
        self.assertIn("auftrag_vorbereiten", self.lauf()["tools"])

    def test_ohne_auftraege_wie_bisher(self):
        from inkluagent.prompts import system_pdf
        demo = self.lauf(mit_lauf=False)["system"]   # nie in der Demo
        with mock.patch.object(funktionen, "AGENT_AUFTRAEGE", False):
            aus = self.lauf()["system"]
        for name, system in (("Demo", demo), ("Schalter aus", aus)):
            with self.subTest(name):
                self.assertTrue(system_pdf._SCHRITT2 in system or system_pdf._SCHRITT2_KETTE in system)
                self.assertIn(system_pdf._SCHLUSS_EINZELN, system)
                self.assertNotIn(system_pdf._SCHRITT2_AUFTRAG, system)
                self.assertNotIn("auftrag_vorbereiten", system)
        self.assertEqual(system_pdf.system_pdf(), system_pdf.system_pdf(auftraege=False), "Standard unveraendert")

    def test_rueckmeldung_fragt_ob_sie_weitermachen_soll(self):
        self.assertIn("schließ mit genau einer Frage, ob du das jetzt erledigen sollst", rm.SYSTEM_MELDUNG)
        self.assertIn("keine Bedingung zum Weitermachen", rm.SYSTEM_MELDUNG)


class Oberflaeche(unittest.TestCase):
    def test_nach_dem_absenden_horcht_die_seite_auf_die_rueckmeldung(self):
        backend = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
        frontend = next(k for k in (os.path.normpath(os.path.join(backend, "..", "frontend")), os.path.join(backend, "frontend"))
                        if os.path.isdir(k))
        from inkluagent import skripte   # alle Teile des Agenten-Skripts (17.10.2026)
        js = skripte.quelltext(frontend)
        absenden = js[js.index("async function inkluagentFormularAbsenden("):js.index("function inkluagentKarteEinfliegen(")]
        self.assertIn("if (res.ok) inkluagentRueckmeldungenBeobachten();", absenden)
        karte = js[js.index("function inkluagentFormularKarte("):js.index("function inkluagentFormularFeld(")]
        self.assertIn("document.createElement(a.nummeriert ? 'ol' : 'ul')", karte, "Schritte als nummerierte Liste (N7)")
        self.assertIn("p.className = 'inkluagent-karte-hinweis';", karte, "Hinweise als eigene Absaetze (N7)")

    def test_namen_und_daten(self):
        from inkluagent import daten
        from inkluagent.tools import namen
        for n in af.NAMEN:
            self.assertIn(n, namen.WERKZEUG_NAMEN)
            self.assertIn(n, daten.WERKZEUG_FREMDTEXT)
        self.assertEqual(karten.KARTENTYPEN["auftrag"], karten.AUFTRAG)


class Hilfe(unittest.TestCase):
    def test_abschnitt_nur_mit_schalter(self):
        from inkluagent import hilfe
        self.assertFalse(hilfe.seite()["auftraege"])
        with an():
            self.assertTrue(hilfe.seite()["auftraege"])
        backend = next(k for k in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app") if os.path.isdir(k))
        vorlage = open(os.path.join(backend, "templates", "hilfe_inkluagent.html"), encoding="utf-8").read()
        self.assertIn("{% if hilfe.auftraege %}", vorlage)
        self.assertIn("Ein Auftrag kostet nie mehr als die Summe auf der Karte.", vorlage)


if __name__ == "__main__":
    unittest.main()
