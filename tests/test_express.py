"""EXPRESS-SERVICE Stufe 1 (05.10.2026): Kern express.py mit eigener Wegwerf-Datenbank — Warenkorb, Bestellung mit
Vormerkung, Lieferung bucht genau einmal ab, Storno gibt frei, Besitzpruefungen, Zustandswechsel, Upload-Pruefung,
Frist ruht bei Rueckfrage, Erinnerungen genau einmal, Kontoloeschung.
Korrekturrunde (Pruefungen Entwicklung und Barrierefreiheit 05.10.2026): je Befund ein Test — Klasse Korrektur (Monats-
wechsel, eingefrorener Korb, Fassung/Summe, Idempotenz, Domain-Topf, fail-closed, Topf-Loeschung, Erinnerung erneut,
Upload nach Lieferung, laufende Pruefung, Aufbewahrung, IP-Netz, Felder der Einstellungen, Texte) und Klasse Erweiterbar
(ein zweiter Dateityp mit eigener Leistung, nur im Test eingehaengt).
    docker exec -w /app inkludocs-staging python3 -m unittest /app/tests/test_express.py -v
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
from datetime import timedelta
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="express_test_")
_DB = os.path.join(_TMP, "test.db")
os.environ["INKLUDOCS_DB"] = _DB

HERE = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.join(os.path.dirname(HERE), "backend")):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402


def _eigene_db():
    os.environ["INKLUDOCS_DB"] = _DB
    database.DB_PATH = _DB


_eigene_db()
database.init_db()

import billing  # noqa: E402
import express  # noqa: E402

express.RESULTS_DIR = os.path.join(_TMP, "results")


def _pdf(pfad, seiten=2):
    import fitz
    d = fitz.open()
    for i in range(seiten):
        d.new_page().insert_text((72, 72), f"Seite {i + 1} (fiktiv)")
    d.save(pfad)
    d.close()
    return pfad


def _sql(q, *p):
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    try:
        r = [dict(x) for x in conn.execute(q, p).fetchall()]
        conn.commit()
        return r
    finally:
        conn.close()


PERSON = {"id": 900, "name": "Bearbeiterin Test"}


class Basis(unittest.TestCase):
    def setUp(self):
        _eigene_db()
        billing.ABO_ENFORCEMENT = True
        for t in ("express_verlauf", "express_positionen", "express_auftraege", "usage_events", "paket_abbuchungen",
                  "quota_pakete", "documents", "projects", "users", "system_kv"):
            _sql(f"DELETE FROM {t}")
        _sql("INSERT INTO users (id, email, password_hash, display_name, plan) VALUES (1, 'kundin@beispiel.invalid', 'x', 'Kundin', 'free')")
        _sql("INSERT INTO users (id, email, password_hash, display_name, plan) VALUES (2, 'fremd@beispiel.invalid', 'x', 'Fremd', 'free')")
        self.pid = self._projekt(1, "Projekt A")
        self.d1 = self._dokument(self.pid, "Bericht.pdf", 2)
        self.d2 = self._dokument(self.pid, "Flyer.pdf", 1)
        self.fremd_pid = self._projekt(2, "Fremdes Projekt")
        self.fremd_doc = self._dokument(self.fremd_pid, "Geheim.pdf", 1)

    def _projekt(self, uid, name):
        conn = sqlite3.connect(_DB)
        cur = conn.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type) VALUES (?, ?, '', ?, 'pdf', 'pdf')",
                           (uid, name, name))
        conn.commit()
        conn.close()
        return cur.lastrowid

    def _dokument(self, pid, name, seiten):
        pfad = _pdf(os.path.join(_TMP, f"{pid}_{name}"), seiten)
        conn = sqlite3.connect(_DB)
        cur = conn.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, 1, ?, ?)",
                           (pid, name, pfad))
        conn.commit()
        conn.close()
        return cur.lastrowid

    def _guthaben(self, credits):
        _sql("INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) VALUES (1, ?, ?, 'admin', 'Test', NULL)",
             credits, credits)

    def _bestellen(self, idem="abcdefgh-1", korb=None, uid=1):
        """Wie die Seite: mit dem Korb, der Summe und der Fassung, die der Kunde gesehen hat."""
        w = korb or express.warenkorb(uid)
        return express.bestellen(uid, ansprechpartner="Kundin", telefon="+49 40 123", hinweise="bitte Seite 2",
                                 bedingungen=True, bearbeitung=True, idempotenz=idem, korb_id=w["id"],
                                 erwartete_credits=w["credits"], fassung=w["fassung"])

    def _auftrag(self, dokumente=None):
        express.dokumente_hinzufuegen(1, dokumente or [self.d1, self.d2])
        self._guthaben(1000)
        return self._bestellen()["auftrag_id"]

    def _ergebnis(self, aid, bestanden=True):
        """Ergebnis hochladen und die automatische Pruefung (im Router: veraPDF) mit festem Ergebnis abschliessen."""
        for p in express.auftrag_fuer_verwaltung(aid)["positionen"]:
            with open(_pdf(os.path.join(_TMP, f"erg_{p['id']}.pdf"), 1), "rb") as f:
                inhalt = f.read()
            erg = express.datei_speichern(aid, p["id"], "ergebnis", inhalt, "../../etc/passwd.pdf", PERSON)
            self.assertTrue(erg["pruef_kennung"])
            express.pruefung_merken(p["id"], {"bestanden": bestanden, "zusammenfassung": "Test"}, erg["pruef_kennung"])


class Warenkorb(Basis):
    def test_hinzufuegen_preis_und_doppelte(self):
        r = express.dokumente_hinzufuegen(1, [self.d1, self.d2])
        self.assertEqual(r["hinzugefuegt"], 2)
        w = r["warenkorb"]
        self.assertEqual((w["dokumente"], w["seiten"], w["credits"]), (2, 3, 150))   # 3 Seiten x 50 (Platzhalter)
        r = express.dokumente_hinzufuegen(1, [self.d1])
        self.assertEqual(r["hinzugefuegt"], 0)
        self.assertTrue(r["hinweise"])

    def test_fremdes_dokument_wird_abgewiesen(self):
        with self.assertRaises(express.ExpressFehler) as e:
            express.dokumente_hinzufuegen(1, [self.d1, self.fremd_doc])
        self.assertEqual(e.exception.status, 404)
        self.assertEqual(express.warenkorb(1)["dokumente"], 0)

    def test_fremde_position_aendern_oder_entfernen_geht_nicht(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        pos = express.warenkorb(1)["positionen"][0]["id"]
        with self.assertRaises(express.NichtGefunden):
            express.leistung_setzen(2, pos, "pruefen")
        with self.assertRaises(express.NichtGefunden):
            express.position_entfernen(2, pos)
        w = express.leistung_setzen(1, pos, "pruefen")
        self.assertEqual(w["credits"], 50)   # 2 Seiten x 25
        with self.assertRaises(express.ExpressFehler):
            express.leistung_setzen(1, pos, "alles")

    def test_keine_pdf_wird_nicht_aufgenommen(self):
        conn = sqlite3.connect(_DB)
        text = os.path.join(_TMP, "kein.pdf")
        with open(text, "w") as f:
            f.write("kein pdf")
        cur = conn.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, 3, 'x.docx', ?)",
                           (self.pid, text))
        conn.commit()
        conn.close()
        r = express.dokumente_hinzufuegen(1, [cur.lastrowid])
        self.assertEqual(r["hinzugefuegt"], 0)

    def test_grenzen(self):
        # Alte Feldnamen (preis_<leistung>) werden weiter angenommen.
        express.speichere_einstellungen({"preis_aufbereiten": "50", "preis_pruefen": "25", "frist_stunden": "48",
                                         "max_seiten_auftrag": "2", "max_dokumente_auftrag": "50"})
        with self.assertRaises(express.ExpressFehler):
            express.dokumente_hinzufuegen(1, [self.d1, self.d2])

    def test_dokumentgrenze_vor_dem_oeffnen(self):
        """Befund 16: zu viele Dokumente -> abgewiesen, ohne eine einzige Datei zum Seitenzaehlen zu oeffnen."""
        import dataclasses
        express.speichere_einstellungen({"preise": {"aufbereiten": "50", "pruefen": "25"}, "frist_stunden": "48",
                                         "max_seiten_auftrag": "500", "max_dokumente_auftrag": "1"})
        geoeffnet = mock.Mock(return_value=1)
        with mock.patch.dict(express.DATEITYPEN, {"pdf": dataclasses.replace(express.DATEITYPEN["pdf"], seiten=geoeffnet)}):
            with self.assertRaises(express.ExpressFehler):
                express.dokumente_hinzufuegen(1, [self.d1, self.d2])
        self.assertEqual(geoeffnet.call_count, 0)


class Bestellen(Basis):
    def test_ohne_haekchen_kein_auftrag(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        with self.assertRaises(express.ExpressFehler):
            express.bestellen(1, ansprechpartner="K", telefon="", hinweise="", bedingungen=True, bearbeitung=False,
                              idempotenz="abcdefgh-2")
        with self.assertRaises(express.ExpressFehler):
            express.bestellen(1, ansprechpartner="K", telefon="", hinweise="", bedingungen="true", bearbeitung=True,
                              idempotenz="abcdefgh-2")

    def test_zu_wenig_guthaben(self):
        express.dokumente_hinzufuegen(1, [self.d1])   # 100 Credits
        # Free hat 50 im Monat, kein Paket
        with self.assertRaises(express.KeinGuthaben) as e:
            self._bestellen()
        self.assertEqual(e.exception.status, 402)
        self.assertEqual(e.exception.extra["preis"], 100)

    def test_bestellen_merkt_vor_und_ist_idempotent(self):
        express.dokumente_hinzufuegen(1, [self.d1, self.d2])
        self._guthaben(1000)
        vorher = billing.verfuegbare_credits(1)
        korb = express.warenkorb(1)
        r = self._bestellen(korb=korb)
        self.assertTrue(r["neu"])
        self.assertEqual(billing.vorgemerkt(1), 150)
        self.assertEqual(billing.verfuegbare_credits(1), vorher - 150)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM usage_events")[0]["n"], 0)   # noch nichts abgebucht
        nochmal = self._bestellen(korb=korb)        # Doppelklick: gleicher Schluessel, gleicher Korb
        self.assertFalse(nochmal["neu"])
        self.assertEqual(nochmal["auftrag_id"], r["auftrag_id"])
        self.assertEqual(billing.vorgemerkt(1), 150)
        a = express.auftrag_fuer_kunde(1, r["auftrag_id"])
        self.assertEqual(a["status"], "neu")
        self.assertEqual(a["zustimmung"]["fassung"], express.ZUSTIMMUNG_FASSUNG)
        self.assertTrue(a["zustimmung"]["bearbeitung"].startswith("Ich bin einverstanden"))
        for p in _sql("SELECT original_pfad FROM express_positionen"):
            self.assertTrue(os.path.isfile(p["original_pfad"]))
            self.assertTrue(p["original_pfad"].startswith(express.RESULTS_DIR))
        self.assertEqual(express.warenkorb(1)["dokumente"], 0)   # neuer, leerer Warenkorb

    def test_gleichzeitig_nur_ein_auftrag(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        ergebnisse = []
        korb = express.warenkorb(1)

        def los():
            try:
                ergebnisse.append(self._bestellen("gleich-123456", korb=korb))
            except Exception as e:  # noqa: BLE001
                ergebnisse.append(e)
        threads = [threading.Thread(target=los) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        ids = {r["auftrag_id"] for r in ergebnisse if isinstance(r, dict)}
        self.assertEqual(len(ids), 1)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_auftraege WHERE status = 'neu'")[0]["n"], 1)

    def test_vormerkung_sperrt_andere_ausgaben(self):
        express.dokumente_hinzufuegen(1, [self.d1])   # 100
        self._guthaben(60)                            # 50 Monat + 60 Paket = 110
        self._bestellen()
        self.assertEqual(billing.verfuegbare_credits(1), 10)
        self.assertFalse(billing.aktion_pruefung(1, "pdf_tagging")["erlaubt"])   # 20 > 10
        self.assertTrue(billing.aktion_pruefung(1, "bild_generierung")["erlaubt"])  # 5 <= 10

    def test_telefon_pruefung(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        with self.assertRaises(express.ExpressFehler):
            express.bestellen(1, ansprechpartner="K", telefon="<script>", hinweise="", bedingungen=True,
                              bearbeitung=True, idempotenz="abcdefgh-3")


class Ablauf(Basis):
    def test_kunde_sieht_nur_eigene(self):
        aid = self._auftrag()
        with self.assertRaises(express.NichtGefunden):
            express.auftrag_fuer_kunde(2, aid)
        self.assertEqual(express.auftraege_des_kunden(2), [])
        with self.assertRaises(express.NichtGefunden):
            express.antwort_kunde(2, aid, "hallo")

    def test_liefern_bucht_genau_einmal_ab(self):
        aid = self._auftrag()
        with self.assertRaises(express.ExpressFehler) as e:
            express.liefern(aid, PERSON)            # Ergebnisse fehlen
        self.assertIn("fehlt", e.exception.text)
        self._ergebnis(aid)
        express.liefern(aid, PERSON)
        self.assertEqual(billing.vorgemerkt(1), 0)
        ev = _sql("SELECT quelle, aktion, credits FROM usage_events")
        self.assertEqual(sum(x["credits"] for x in ev), 150)
        self.assertTrue(all(x["quelle"] == "express" for x in ev))
        with self.assertRaises(express.ExpressFehler) as e:
            express.liefern(aid, PERSON)
        self.assertEqual(e.exception.status, 409)
        self.assertEqual(sum(x["credits"] for x in _sql("SELECT credits FROM usage_events")), 150)
        with self.assertRaises(express.ExpressFehler):
            express.stornieren(aid, PERSON, "zu spaet")   # nach der Lieferung kein Storno

    def test_gleichzeitig_liefern_bucht_einmal(self):
        aid = self._auftrag()
        self._ergebnis(aid)
        fehler = []

        def los():
            try:
                express.liefern(aid, PERSON)
            except express.ExpressFehler as e:
                fehler.append(e.status)
        threads = [threading.Thread(target=los) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sum(x["credits"] for x in _sql("SELECT credits FROM usage_events")), 150)
        self.assertEqual(len(fehler), 3)

    def test_storno_gibt_frei(self):
        aid = self._auftrag()
        with self.assertRaises(express.ExpressFehler):
            express.stornieren(aid, PERSON, "")      # Grund ist Pflicht
        express.stornieren(aid, PERSON, "Kunde wollte nicht mehr")
        self.assertEqual(billing.vorgemerkt(1), 0)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM usage_events")[0]["n"], 0)
        a = express.auftrag_fuer_kunde(1, aid)
        self.assertEqual((a["status"], a["credits_stand"]), ("storniert", "frei"))

    def test_dateiname_wird_nie_pfad(self):
        aid = self._auftrag([self.d1])
        self._ergebnis(aid)
        p = express.auftrag_fuer_verwaltung(aid)["positionen"][0]
        pfad = _sql("SELECT ergebnis_pfad FROM express_positionen")[0]["ergebnis_pfad"]
        self.assertTrue(pfad.startswith(express.ordner(1, aid)))
        self.assertNotIn("..", pfad)
        self.assertEqual(p["ergebnis_name"], "passwd.pdf")

    def test_upload_pruefung(self):
        aid = self._auftrag([self.d1])
        pos = express.auftrag_fuer_verwaltung(aid)["positionen"][0]["id"]
        for inhalt in (b"", b"MZ\x90\x00 kein pdf", b"%PDF-1.7 kaputt"):
            with self.assertRaises(express.ExpressFehler):
                express.datei_speichern(aid, pos, "ergebnis", inhalt, "x.pdf", PERSON)
        with self.assertRaises(express.ExpressFehler):
            express.datei_speichern(aid, pos, "skript", b"%PDF-", "x.pdf", PERSON)
        with self.assertRaises(express.NichtGefunden):
            express.datei_speichern(aid + 1000, pos, "ergebnis", b"%PDF-", "x.pdf", PERSON)
        with mock.patch.object(express, "MAX_ERGEBNIS_BYTES", 10):
            with self.assertRaises(express.ExpressFehler) as e:
                express.datei_speichern(aid, pos, "ergebnis", b"%PDF-" + b"x" * 20, "x.pdf", PERSON)
            self.assertEqual(e.exception.status, 413)

    def test_kunde_laedt_erst_nach_lieferung(self):
        aid = self._auftrag([self.d1])
        self._ergebnis(aid)
        pos = express.auftrag_fuer_verwaltung(aid)["positionen"][0]["id"]
        with self.assertRaises(express.NichtGefunden):
            express.datei_fuer_kunde(1, aid, pos, "ergebnis")
        express.liefern(aid, PERSON)
        pfad, name = express.datei_fuer_kunde(1, aid, pos, "ergebnis")
        self.assertTrue(os.path.isfile(pfad))
        self.assertTrue(name.endswith("(barrierefrei).pdf"))
        with self.assertRaises(express.NichtGefunden):
            express.datei_fuer_kunde(2, aid, pos, "ergebnis")
        with self.assertRaises(express.NichtGefunden):
            express.datei_fuer_kunde(1, aid, pos, "original_pfad")

    def test_rueckfrage_frist_ruht(self):
        aid = self._auftrag()
        express.uebernehmen(aid, PERSON)
        vorher = _sql("SELECT faellig_am FROM express_auftraege WHERE id = ?", aid)[0]["faellig_am"]
        express.rueckfrage(aid, PERSON, "Welche Sprache hat Seite 2?")
        _sql("UPDATE express_auftraege SET rueckfrage_seit = datetime('now', '-3 hours') WHERE id = ?", aid)
        a = express.antwort_kunde(1, aid, "Englisch")
        self.assertEqual(a["status"], "in_arbeit")
        nachher = _sql("SELECT faellig_am FROM express_auftraege WHERE id = ?", aid)[0]["faellig_am"]
        delta = (express._als_dt(nachher) - express._als_dt(vorher)).total_seconds()
        self.assertAlmostEqual(delta, 3 * 3600, delta=120)
        with self.assertRaises(express.ExpressFehler) as e:
            express.antwort_kunde(1, aid, "nochmal")
        self.assertEqual(e.exception.status, 409)

    def test_erinnerung_und_ueberfaellig_je_einmal(self):
        aid = self._auftrag()
        self.assertEqual(express.faellige_meldungen(), [])
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?",
             express._utc(express._jetzt() + timedelta(hours=5)), aid)
        self.assertEqual(express.faellige_meldungen(), [("erinnerung", aid)])
        self.assertEqual(express.faellige_meldungen(), [])
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?",
             express._utc(express._jetzt() - timedelta(hours=1)), aid)
        self.assertEqual(express.faellige_meldungen(), [("ueberfaellig", aid)])
        self.assertEqual(express.faellige_meldungen(), [])

    def test_liste_nach_dringlichkeit(self):
        aid = self._auftrag([self.d1])
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?",
             express._utc(express._jetzt() - timedelta(hours=1)), aid)
        g = express.liste_fuer_verwaltung()
        self.assertEqual([a["id"] for a in g["ueberfaellig"]], [aid])
        self.assertEqual(g["neu"], [])

    def test_nachweis_docx(self):
        aid = self._auftrag()
        a = express.auftrag_fuer_kunde(1, aid)
        ziel = os.path.join(_TMP, "nachweis.docx")
        titel = express.nachweis_docx(a, ziel, lambda t: t)
        import zipfile
        with zipfile.ZipFile(ziel) as z:
            doc = z.read("word/document.xml").decode()
            self.assertIn('w:val="Heading1"', doc)
            self.assertIn("keine Rechnung", doc)
            self.assertIn("<dc:language>de-DE</dc:language>", z.read("docProps/core.xml").decode())
        self.assertIn(str(aid), titel)

    def test_kontoloeschung(self):
        aid = self._auftrag()
        database.delete_user_data(1)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_auftraege")[0]["n"], 0)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_positionen WHERE auftrag_id = ?", aid)[0]["n"], 0)


class Einstellungen(Basis):
    def test_pruefung(self):
        for falsch in ({"preis_aufbereiten": "0"}, {"preis_aufbereiten": "viel"}, {"frist_stunden": "-5"},
                       {"team_mail": "kein mail"}):
            daten = {"preis_aufbereiten": "50", "preis_pruefen": "25", "frist_stunden": "48",
                     "max_seiten_auftrag": "500", "max_dokumente_auftrag": "50"}
            daten.update(falsch)
            with self.assertRaises(express.ExpressFehler):
                express.speichere_einstellungen(daten)
        e = express.speichere_einstellungen({"preise": {"aufbereiten": "60", "pruefen": "30"}, "frist_stunden": "72",
                                             "max_seiten_auftrag": "300", "max_dokumente_auftrag": "20",
                                             "team_mail": "team@beispiel.invalid", "preise_festgelegt": True})
        self.assertEqual((e["preise"]["aufbereiten"], e["frist_stunden"], e["preise_festgelegt"]), (60, 72, True))
        self.assertEqual(express.einstellungen()["preise"], {"aufbereiten": 60, "pruefen": 30})
        self.assertEqual(express.team_empfaenger("support@beispiel.invalid"), ["team@beispiel.invalid"])

    def test_bearbeiter(self):
        self.assertFalse(express.ist_bearbeiter(2))
        express.bearbeiter_setzen(email="FREMD@beispiel.invalid", an=True)
        self.assertTrue(express.ist_bearbeiter(2))
        self.assertIn("fremd@beispiel.invalid", express.team_empfaenger("support@beispiel.invalid"))
        express.bearbeiter_setzen(user_id=2, an=False)
        self.assertFalse(express.ist_bearbeiter(2))
        with self.assertRaises(express.ExpressFehler):
            express.bearbeiter_setzen(email="niemand@beispiel.invalid")


def _monat_davor(n=1):
    """(jahr, monat) n Monate vor dem laufenden (UTC)."""
    jetzt = express._jetzt()
    j, m = jetzt.year, jetzt.month - n
    while m < 1:
        m, j = m + 12, j - 1
    return j, m


class Korrektur(Basis):
    """Je Befund der Pruefungen vom 05.10.2026 ein Test (E = Entwicklung, A = Barrierefreiheit)."""

    # ── E1: Monatswechsel ──
    def test_e1_abbuchung_zaehlt_zum_bestellmonat(self):
        """Szenario des Befunds: Single (250), 249 Uebertrag in den Bestellmonat, Auftrag 400, geliefert im Folgemonat.
        Richtig: wie Verbrauch im Bestellmonat -> im laufenden Monat 99 Uebertrag + 250 = 349 frei (vorher nur 100)."""
        _sql("UPDATE users SET plan = 'single', plan_gueltig_bis = '2099-12-31' WHERE id = 1")
        j2, m2 = _monat_davor(2)
        _sql("INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits, created_at) "
             "VALUES (1, 1, 'web', 'bild_generierung', 1, ?)", f"{j2:04d}-{m2:02d}-15 10:00:00")
        d = self._dokument(self.pid, "Gross.pdf", 8)                  # 8 Seiten x 50 = 400
        aid = self._auftrag([d])
        j1, m1 = _monat_davor(1)
        bestellt = f"{j1:04d}-{m1:02d}-28 10:00:00"
        _sql("UPDATE express_auftraege SET bestellt_am = ? WHERE id = ?", bestellt, aid)
        self._ergebnis(aid)
        express.liefern(aid, PERSON)
        ev = _sql("SELECT created_at, credits FROM usage_events WHERE quelle = 'express'")
        self.assertEqual([(e["created_at"], e["credits"]) for e in ev], [(bestellt, 400)])
        z = billing.pruefe_kontingent(1)
        self.assertEqual((z["uebertrag"], z["verbraucht"], z["rest"]), (99, 0, 349))

    def test_e1_paket_ueberhang_im_bestellmonat(self):
        """Free (50 im Monat), Paket 1000, Auftrag 100 im Vormonat bestellt: der Ueberhang 50 geht im VORMONAT vom
        Paket ab; der laufende Monat bleibt unberuehrt."""
        aid = self._auftrag([self.d1])                               # 100 Credits, _auftrag schenkt 1000 Paket
        j1, m1 = _monat_davor(1)
        bestellt = f"{j1:04d}-{m1:02d}-20 09:00:00"
        _sql("UPDATE express_auftraege SET bestellt_am = ? WHERE id = ?", bestellt, aid)
        self._ergebnis(aid)
        express.liefern(aid, PERSON)
        ab = _sql("SELECT betrag, created_at FROM paket_abbuchungen")
        self.assertEqual([(a["betrag"], a["created_at"]) for a in ab], [(50, bestellt)])
        self.assertEqual(billing.pakete_rest(1), 950)
        self.assertEqual(billing.pruefe_kontingent(1)["verbraucht"], 0)

    # ── E2: Korb waehrend des Bestellens ──
    def test_e2_korb_ist_waehrend_der_bestellung_eingefroren(self):
        express.dokumente_hinzufuegen(1, [self.d1])                   # 100 Credits
        self._guthaben(1000)
        korb = express.warenkorb(1)
        pos = korb["positionen"][0]["id"]
        echt = shutil.copyfile
        zweiter_tab = {}

        def kopieren(quelle, ziel):
            # Waehrend die Originale kopiert werden, arbeitet ein zweiter Tab am Korb.
            if not zweiter_tab:
                with self.assertRaises(express.NichtGefunden):
                    express.position_entfernen(1, pos)
                with self.assertRaises(express.NichtGefunden):
                    express.leistung_setzen(1, pos, "pruefen")
                zweiter_tab["neu"] = express.dokumente_hinzufuegen(1, [self.d2])["warenkorb"]
            return echt(quelle, ziel)
        with mock.patch.object(express.shutil, "copyfile", side_effect=kopieren):
            r = self._bestellen(korb=korb)
        a = express.auftrag_fuer_verwaltung(r["auftrag_id"])
        self.assertEqual([p["document_id"] for p in a["positionen"]], [self.d1])
        self.assertEqual((a["credits_gesamt"], a["seiten_gesamt"]), (100, 2))
        neu = express.warenkorb(1)
        self.assertNotEqual(neu["id"], r["auftrag_id"])
        self.assertEqual([p["document_id"] for p in neu["positionen"]], [self.d2])   # landet im NEUEN Korb

    def test_e2_fehlschlag_gibt_den_korb_frei(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        korb = express.warenkorb(1)                                    # kein Guthaben: Free 50 < 100
        with self.assertRaises(express.KeinGuthaben):
            self._bestellen(korb=korb)
        self.assertEqual(_sql("SELECT status FROM express_auftraege WHERE id = ?", korb["id"])[0]["status"], "entwurf")
        self.assertEqual(express.warenkorb(1)["id"], korb["id"])

    def test_e2_haengender_korb_wird_wieder_frei(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        wid = express.warenkorb(1)["id"]
        _sql("UPDATE express_auftraege SET status = 'bestellung', updated_at = datetime('now', '-30 minutes') WHERE id = ?", wid)
        self.assertEqual(express.warenkorb(1)["id"], wid)

    # ── E3: Preis/Fassung ──
    def test_e3_anderer_preis_als_angezeigt_wird_nicht_bestellt(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        gesehen = express.warenkorb(1)                                 # 100 Credits
        express.speichere_einstellungen({"preise": {"aufbereiten": "80", "pruefen": "25"}, "frist_stunden": "48",
                                         "max_seiten_auftrag": "500", "max_dokumente_auftrag": "50"})
        with self.assertRaises(express.ExpressFehler) as e:
            self._bestellen(korb=gesehen)
        self.assertEqual(e.exception.status, 409)
        self.assertTrue(e.exception.extra.get("veraltet"))
        self.assertEqual(e.exception.extra.get("neu_credits"), 160)
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_auftraege WHERE status = 'neu'")[0]["n"], 0)
        r = self._bestellen("abcdefgh-neu")                           # mit dem neuen Stand geht es
        self.assertEqual(express.auftrag_fuer_kunde(1, r["auftrag_id"])["credits"], 160)

    def test_e3_geaenderte_auswahl_wird_nicht_bestellt(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        gesehen = express.warenkorb(1)
        express.dokumente_hinzufuegen(1, [self.d2])                   # zweiter Tab
        with self.assertRaises(express.ExpressFehler) as e:
            self._bestellen(korb=gesehen)
        self.assertEqual(e.exception.status, 409)

    # ── E4: Idempotenz an den Korb gebunden ──
    def test_e4_alter_schluessel_bestellt_nicht_still_den_alten_auftrag(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        erster = self._bestellen("schluessel-1")
        express.dokumente_hinzufuegen(1, [self.d2])                   # neuer Korb
        with self.assertRaises(express.ExpressFehler) as e:
            self._bestellen("schluessel-1")                            # Seite aus dem Zurueck-Speicher
        self.assertEqual(e.exception.status, 409)
        self.assertTrue(e.exception.extra.get("veraltet"))
        self.assertEqual(express.warenkorb(1)["dokumente"], 1)        # neuer Korb unbestellt, aber sichtbar
        zweiter = self._bestellen("schluessel-2")
        self.assertNotEqual(zweiter["auftrag_id"], erster["auftrag_id"])

    # ── E7: Free-Domain-Topf ──
    def test_e7_domain_topf_wird_nicht_doppelt_vorgemerkt(self):
        _sql("UPDATE users SET email = 'a@firma-test-beispiel.de' WHERE id = 1")
        _sql("UPDATE users SET email = 'b@firma-test-beispiel.de' WHERE id = 2")
        d_a = self._dokument(self.pid, "Eins.pdf", 1)                 # 50 Credits = das ganze Domain-Volumen
        d_b = self._dokument(self.fremd_pid, "Zwei.pdf", 1)
        express.dokumente_hinzufuegen(1, [d_a])
        self._bestellen()
        express.dokumente_hinzufuegen(2, [d_b])
        with self.assertRaises(express.KeinGuthaben):
            self._bestellen("abcdefgh-b", uid=2)
        self.assertEqual(billing.vorgemerkt_domain("firma-test-beispiel.de"), 50)

    # ── E18: fail-closed ──
    def test_e18_datenbankfehler_sperrt_die_bestellung(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(1000)
        korb = express.warenkorb(1)
        with mock.patch.object(billing, "monats_verbrauch", side_effect=sqlite3.OperationalError("database is locked")):
            with self.assertRaises(express.ExpressFehler) as e:
                self._bestellen(korb=korb)
        self.assertEqual(e.exception.status, 503)
        self.assertEqual(_sql("SELECT status FROM express_auftraege WHERE id = ?", korb["id"])[0]["status"], "entwurf")
        self.assertEqual(billing.vorgemerkt(1), 0)

    # ── E6: Topf-Inhaber geloescht ──
    def test_e6_topf_inhaber_geloescht_storniert_statt_umhaengen(self):
        _sql("INSERT INTO users (id, email, password_hash, display_name, plan) VALUES (3, 'inhaber@beispiel.invalid', 'x', 'Inhaber', 'team')")
        aid = self._auftrag([self.d1])
        _sql("UPDATE express_auftraege SET konto_user_id = 3 WHERE id = ?", aid)
        info = express.vor_kontoloeschung(3)
        self.assertEqual((info["topf"], info["eigene"]), ([aid], []))
        database.delete_user_data(3)
        a = _sql("SELECT status, konto_user_id, storno_grund FROM express_auftraege WHERE id = ?", aid)[0]
        self.assertEqual((a["status"], a["konto_user_id"]), ("storniert", None))
        self.assertIn("gelöscht", a["storno_grund"])
        self.assertEqual(billing.vorgemerkt(1), 0)                    # das Mitglied ist nicht belastet
        self.assertTrue(billing.aktion_pruefung(1, "bild_generierung")["erlaubt"])
        self.assertEqual(_sql("SELECT COUNT(*) AS n FROM express_verlauf WHERE auftrag_id = ? AND art = 'storniert'", aid)[0]["n"], 1)

    def test_e6_eigene_offene_auftraege_werden_gemeldet(self):
        aid = self._auftrag([self.d1])
        info = express.vor_kontoloeschung(1)
        self.assertEqual([a["id"] for a in info["eigene"]], [aid])
        betreff, inhalt = express.mail_team("entfallen", info["eigene"][0], "https://beispiel.invalid")
        self.assertIn(str(aid), betreff)
        self.assertIn("gelöscht", inhalt)

    # ── E9: Schalter aus, Auftraege bleiben erreichbar ──
    def test_e9_bestellte_auftraege_erkennen(self):
        self.assertFalse(express.gibt_bestellte())
        express.dokumente_hinzufuegen(1, [self.d1])
        self.assertFalse(express.gibt_bestellte())                    # ein Warenkorb zaehlt nicht
        self._auftrag([self.d2])
        self.assertTrue(express.gibt_bestellte())
        self.assertTrue(express.hat_bestellte(1))
        self.assertFalse(express.hat_bestellte(2))
        self.assertEqual(express.offene_anzahl(), 1)

    # ── E10: Erinnerung nach Fehlversand erneut ──
    def test_e10_freigegebene_erinnerung_kommt_wieder(self):
        aid = self._auftrag()
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?", express._utc(express._jetzt() + timedelta(hours=5)), aid)
        self.assertEqual(express.faellige_meldungen(), [("erinnerung", aid)])
        express.meldung_freigeben("erinnerung", aid)                  # Versand ging an niemanden
        self.assertEqual(express.faellige_meldungen(), [("erinnerung", aid)])
        self.assertEqual(express.faellige_meldungen(), [])

    def test_e10_wiederholung_nur_ohne_jeden_erfolg(self):
        import express_api
        aid = self._auftrag()
        with mock.patch.object(express_api, "_mails_nach", return_value=0), \
                mock.patch.object(express, "meldung_freigeben") as frei:
            express_api._meldung_versuche.clear()
            for _ in range(express_api.MELDUNG_VERSUCHE):
                express_api._meldung_senden("ueberfaellig", aid)
            self.assertEqual(frei.call_count, express_api.MELDUNG_VERSUCHE - 1)   # danach aufgegeben
        with mock.patch.object(express_api, "_mails_nach", return_value=1), \
                mock.patch.object(express, "meldung_freigeben") as frei:
            express_api._meldung_senden("ueberfaellig", aid)
            frei.assert_not_called()                                   # an einen ging es raus: keine Doppelmails

    # ── E11: nach der Lieferung nichts mehr ersetzen; laufende Pruefung ──
    def test_e11_upload_nach_lieferung_abgewiesen(self):
        aid = self._auftrag([self.d1])
        self._ergebnis(aid)
        express.liefern(aid, PERSON)
        pos = express.auftrag_fuer_verwaltung(aid)["positionen"][0]["id"]
        vorher = _sql("SELECT ergebnis_pfad, ergebnis_am, verapdf FROM express_positionen WHERE id = ?", pos)[0]
        with open(_pdf(os.path.join(_TMP, "spaet.pdf"), 1), "rb") as f:
            with self.assertRaises(express.ExpressFehler) as e:
                express.datei_speichern(aid, pos, "ergebnis", f.read(), "spaet.pdf", PERSON)
        self.assertEqual(e.exception.status, 409)
        self.assertEqual(_sql("SELECT ergebnis_pfad, ergebnis_am, verapdf FROM express_positionen WHERE id = ?", pos)[0], vorher)

    def test_e11_liefern_wartet_auf_laufende_pruefung(self):
        aid = self._auftrag([self.d1])
        pos = express.auftrag_fuer_verwaltung(aid)["positionen"][0]["id"]
        with open(_pdf(os.path.join(_TMP, "e1.pdf"), 1), "rb") as f:
            erst = express.datei_speichern(aid, pos, "ergebnis", f.read(), "e1.pdf", PERSON)
        with self.assertRaises(express.ExpressFehler) as e:
            express.liefern(aid, PERSON)
        self.assertEqual(e.exception.status, 409)
        self.assertTrue(e.exception.extra.get("pruefung_laeuft"))
        with open(_pdf(os.path.join(_TMP, "e2.pdf"), 1), "rb") as f:
            zweit = express.datei_speichern(aid, pos, "ergebnis", f.read(), "e2.pdf", PERSON)
        # Das Ergebnis der ERSTEN Pruefung gehoert zu einer ersetzten Datei und wird verworfen.
        self.assertIsNone(express.pruefung_merken(pos, {"bestanden": True}, erst["pruef_kennung"]))
        self.assertTrue(express.lieferbar(express.auftrag_fuer_verwaltung(aid))["pruefung_laeuft"])
        self.assertIsNotNone(express.pruefung_merken(pos, {"bestanden": False, "zusammenfassung": "2 Regeln"}, zweit["pruef_kennung"]))
        with self.assertRaises(express.ExpressFehler) as e:
            express.liefern(aid, PERSON)                              # Befund: einmal nachfragen
        self.assertTrue(e.exception.extra.get("nachfrage"))
        express.liefern(aid, PERSON, trotz_befunden=True)

    def test_e11_haengende_pruefung_blockiert_nicht_ewig(self):
        aid = self._auftrag([self.d1])
        pos = express.auftrag_fuer_verwaltung(aid)["positionen"][0]["id"]
        with open(_pdf(os.path.join(_TMP, "h.pdf"), 1), "rb") as f:
            express.datei_speichern(aid, pos, "ergebnis", f.read(), "h.pdf", PERSON)
        alt = express._utc(express._jetzt() - timedelta(minutes=express.PRUEFUNG_HAENGT_MIN + 1))
        _sql("UPDATE express_positionen SET verapdf = ? WHERE id = ?", '{"laeuft": "x", "seit": "%s"}' % alt, pos)
        express.liefern(aid, PERSON)
        p = express.auftrag_fuer_kunde(1, aid)["positionen"][0]
        self.assertIsNone(p["verapdf"])                               # Kunde sieht kein Pruefergebnis

    # ── E14: Aufbewahrung und IP-Netz ──
    def test_e14_aufbewahrung(self):
        aid = self._auftrag([self.d1])
        self._ergebnis(aid)
        express.liefern(aid, PERSON)
        _sql("UPDATE express_auftraege SET geliefert_am = datetime('now', '-3 days') WHERE id = ?", aid)
        self.assertEqual(express.aufraeumen(), 0)                     # Standard 0 = nichts loeschen
        self.assertTrue(os.path.isdir(express.ordner(1, aid)))
        express.speichere_einstellungen({"preise": {"aufbereiten": "50", "pruefen": "25"}, "frist_stunden": "48",
                                         "max_seiten_auftrag": "500", "max_dokumente_auftrag": "50", "aufbewahrung_tage": "2"})
        self.assertEqual(express.aufraeumen(), 1)
        self.assertFalse(os.path.isdir(express.ordner(1, aid)))
        p = _sql("SELECT original_pfad, ergebnis_pfad FROM express_positionen WHERE auftrag_id = ?", aid)[0]
        self.assertEqual((p["original_pfad"], p["ergebnis_pfad"]), ("", ""))
        a = express.auftrag_fuer_kunde(1, aid)
        self.assertEqual(a["status"], "geliefert")                    # Auftrag bleibt als Nachweis
        self.assertIn("dateien_geloescht", [v["art"] for v in a["verlauf"]])
        self.assertEqual(express.aufraeumen(), 0)                     # nur einmal

    def test_e14_ip_netz_gekuerzt(self):
        self.assertEqual(express.netz_kurz("203.0.113.77"), "203.0.113.0/24")
        self.assertEqual(express.netz_kurz("2001:db8:abcd:12::1"), "2001:db8:abcd::/48")
        self.assertEqual(express.netz_kurz("kein netz"), "")
        aid = self._auftrag([self.d1])
        self.assertEqual(_sql("SELECT zustimmung_absender FROM express_auftraege WHERE id = ?", aid)[0]["zustimmung_absender"], "")

    # ── E15: Frist ruht bei Rueckfrage ──
    def test_e15_frist_ruht_in_der_anzeige(self):
        aid = self._auftrag([self.d1])
        _sql("UPDATE express_auftraege SET faellig_am = ? WHERE id = ?", express._utc(express._jetzt() - timedelta(hours=1)), aid)
        express.rueckfrage(aid, PERSON, "Frage")
        a = express.auftrag_fuer_verwaltung(aid)
        self.assertEqual((a["frist_ruht"], a["ueberfaellig"], a["faellig_in_stunden"]), (True, False, None))
        g = express.liste_fuer_verwaltung()
        self.assertEqual([x["id"] for x in g["rueckfrage"]], [aid])
        self.assertEqual(g["ueberfaellig"], [])

    def test_e15_kein_guthaben_text_nennt_vormerkung(self):
        express.dokumente_hinzufuegen(1, [self.d1])
        self._guthaben(60)
        self._bestellen()                                              # 100 vorgemerkt, 10 frei
        p = billing.aktion_pruefung(1, "pdf_tagging")                  # 20 > 10
        self.assertFalse(p["erlaubt"])
        self.assertIn("vorgemerkt", billing.credits_fehlen_text(p))
        self.assertIn("vorgemerkt", billing.credits_fehlen_detail(p)["text"])

    # ── Einstellungen: Fehler am Feld (A4) und alte Preis-Schluessel ──
    def test_a4_einstellungen_nennen_das_feld(self):
        basis = {"preise": {"aufbereiten": "50", "pruefen": "25"}, "frist_stunden": "48", "max_seiten_auftrag": "500",
                 "max_dokumente_auftrag": "50"}
        for aenderung, feld, anfang in (({"frist_stunden": "x"}, "frist_stunden", "Lieferfrist in Stunden"),
                                        ({"preise": {"aufbereiten": "0", "pruefen": "25"}}, "preis_aufbereiten", "Barrierefrei aufbereiten"),
                                        ({"aufbewahrung_tage": "-1"}, "aufbewahrung_tage", "Dateien löschen nach Tagen"),
                                        ({"team_mail": "kein mail"}, "team_mail", "Benachrichtigung an")):
            daten = dict(basis, **aenderung)
            with self.assertRaises(express.ExpressFehler) as e:
                express.speichere_einstellungen(daten)
            self.assertEqual(e.exception.extra.get("feld"), feld)
            self.assertTrue(e.exception.text.startswith(anfang), e.exception.text)

    def test_alte_preis_schluessel_werden_uebernommen(self):
        _sql("INSERT INTO system_kv (key, value) VALUES ('express_einstellungen', ?)",
             '{"preis_aufbereiten": 70, "preis_pruefen": 30, "frist_stunden": 24}')
        e = express.einstellungen()
        self.assertEqual((e["preise"], e["frist_stunden"], e["aufbewahrung_tage"]), ({"aufbereiten": 70, "pruefen": 30}, 24, 0))

    # ── Texte: Einzahl, deutsches Datum, Tausenderpunkt (A11, A22) ──
    def test_a11_texte(self):
        self.assertEqual(express.datum_deutsch("2026-10-05 12:04"), "5. Oktober 2026, 12:04")
        self.assertEqual(express.datum_deutsch("2026-03-01"), "1. März 2026")
        self.assertEqual(express._seiten(1), "1 Seite")
        self.assertEqual(express._seiten(1200), "1.200 Seiten")
        aid = self._auftrag([self.d2])                                 # 1 Seite
        a = express.auftrag_fuer_kunde(1, aid)
        ziel = os.path.join(_TMP, "nachweis2.docx")
        express.nachweis_docx(a, ziel)
        import zipfile
        with zipfile.ZipFile(ziel) as z:
            doc = z.read("word/document.xml").decode()
        self.assertIn("1 Seite,", doc)
        self.assertNotIn("1 Seiten", doc)
        self.assertNotRegex(doc, r"Bestellt am: \d{4}-\d{2}-\d{2}")
        betreff, inhalt = express.mail_kunde("bestellt", express.auftrag_fuer_verwaltung(aid), "https://beispiel.invalid")
        self.assertIn("1 Dokument, 1 Seite", inhalt)

    def test_a12_kunde_sieht_keine_technik_zusammenfassung(self):
        aid = self._auftrag([self.d1])
        self._ergebnis(aid, bestanden=False)
        express.liefern(aid, PERSON, trotz_befunden=True)
        p = express.auftrag_fuer_kunde(1, aid)["positionen"][0]
        self.assertEqual(p["verapdf"], {"bestanden": False})
        self.assertEqual((p["pruef_name"], p["ergebnis_typ"]), ("veraPDF", "pdf"))


class Erweiterbar(Basis):
    """Steve 05.10.2026: vorerst nur PDF, aber jederzeit erweiterbar. Hier wird — NUR im Test — ein zweiter Dateityp
    mit eigener Leistung eingehaengt; Warenkorb, Preise, Bestellung, Upload, Liefern und Download muessen ohne
    Code-Aenderung damit umgehen."""

    def setUp(self):
        super().setUp()
        self.txt = express.Dateityp(schluessel="txt", name="Text", endung=".txt", endungen=(".txt",), mime="text/plain",
                                    accept=".txt", projekt_typen=("pdf",), erkennen=lambda kopf: kopf.startswith(b"TXT:"),
                                    seiten=lambda pfad: max(1, os.path.getsize(pfad) // 100))
        self.lesen = express.Leistung(schluessel="vorlesen", name="Vorlesen lassen", preis_standard=7, dateitypen=("txt",),
                                      aktion="express_vorlesen", ergebnis_pflicht=True, bericht_pflicht=False,
                                      bericht_typen=("pdf",), ergebnis_zusatz=" (gelesen)")
        self._p1 = mock.patch.dict(express.DATEITYPEN, {"txt": self.txt})
        self._p2 = mock.patch.dict(express.LEISTUNGEN, {"vorlesen": self.lesen})
        self._p1.start()
        self._p2.start()
        pfad = os.path.join(_TMP, "notiz.txt")
        with open(pfad, "wb") as f:
            f.write(b"TXT:" + b"x" * 296)                              # 300 Bytes = 3 „Seiten“
        conn = sqlite3.connect(_DB)
        cur = conn.execute("INSERT INTO documents (project_id, doc_index, original_filename, original_path) VALUES (?, 5, 'notiz.txt', ?)",
                           (self.pid, pfad))
        conn.commit()
        conn.close()
        self.dtxt = cur.lastrowid

    def tearDown(self):
        self._p2.stop()
        self._p1.stop()

    def test_neuer_typ_durch_den_ganzen_ablauf(self):
        self.assertEqual(express.typen_text(), "PDF oder Text")
        r = express.dokumente_hinzufuegen(1, [self.d1, self.dtxt])
        pos = {p["dateityp"]: p for p in r["warenkorb"]["positionen"]}
        self.assertEqual((pos["txt"]["leistung"], pos["txt"]["seiten"], pos["txt"]["credits"]), ("vorlesen", 3, 21))
        self.assertEqual(pos["txt"]["leistungen"], ["vorlesen"])
        self.assertEqual(pos["pdf"]["leistungen"], ["aufbereiten", "pruefen"])
        with self.assertRaises(express.ExpressFehler):
            express.leistung_setzen(1, pos["pdf"]["id"], "vorlesen")      # nicht fuer PDF
        with self.assertRaises(express.ExpressFehler):
            express.leistung_setzen(1, pos["txt"]["id"], "aufbereiten")   # nicht fuer Text
        e = express.speichere_einstellungen({"preise": {"aufbereiten": "50", "pruefen": "25", "vorlesen": "9"},
                                             "frist_stunden": "48", "max_seiten_auftrag": "500", "max_dokumente_auftrag": "50"})
        self.assertEqual(e["preise"]["vorlesen"], 9)
        self._guthaben(1000)
        aid = self._bestellen()["auftrag_id"]
        a = express.auftrag_fuer_verwaltung(aid)
        p_txt = next(p for p in a["positionen"] if p["dateityp"] == "txt")
        self.assertEqual(p_txt["credits"], 27)
        self.assertEqual([t["schluessel"] for t in p_txt["ergebnis_typen"]], ["txt"])
        orig = _sql("SELECT original_pfad FROM express_positionen WHERE id = ?", p_txt["id"])[0]["original_pfad"]
        self.assertTrue(orig.endswith("_original.txt"))
        with self.assertRaises(express.ExpressFehler) as fe:
            express.datei_speichern(aid, p_txt["id"], "ergebnis", b"%PDF-1.7", "x.pdf", PERSON)
        self.assertIn("Text", fe.exception.text)
        erg = express.datei_speichern(aid, p_txt["id"], "ergebnis", b"TXT: fertig", "fertig.txt", PERSON)
        self.assertEqual(erg["pruef_kennung"], "")                      # Typ ohne automatische Pruefung
        p_pdf = next(p for p in a["positionen"] if p["dateityp"] == "pdf")
        with open(_pdf(os.path.join(_TMP, "erg_x.pdf"), 1), "rb") as f:
            k = express.datei_speichern(aid, p_pdf["id"], "ergebnis", f.read(), "e.pdf", PERSON)["pruef_kennung"]
        express.pruefung_merken(p_pdf["id"], {"bestanden": True}, k)
        express.liefern(aid, PERSON)
        ev = {x["aktion"]: x["credits"] for x in _sql("SELECT aktion, credits FROM usage_events WHERE quelle = 'express'")}
        self.assertEqual(ev, {"express_vorlesen": 27, "express_aufbereiten": 100})
        pfad, name = express.datei_fuer_kunde(1, aid, p_txt["id"], "ergebnis")
        self.assertEqual((name, express.mime_der_datei(pfad)), ("notiz (gelesen).txt", "text/plain"))

    def test_upload_ohne_projekt_erkennt_den_typ(self):
        self.assertEqual(express.dateityp_fuer_upload("a.txt", b"TXT:abc").schluessel, "txt")
        self.assertEqual(express.dateityp_fuer_upload("a.PDF", b"%PDF-1.4").schluessel, "pdf")
        with self.assertRaises(express.ExpressFehler):
            express.dateityp_fuer_upload("a.txt", b"%PDF-1.4")         # Endung und Inhalt passen nicht
        with self.assertRaises(express.ExpressFehler):
            express.dateityp_fuer_upload("a.docx", b"PK\x03\x04")


class WarenkorbZusatz(Basis):
    """Zusatz 05.10.2026 (Steve): Knopf „In den Express-Warenkorb“ am Dokument und Eintrag „Express-Warenkorb“ in der
    Navigation — beides ohne Codeaenderung in den Express-Einstellungen umschaltbar, nur bei eingeschaltetem Express."""

    BASIS = {"preise": {"aufbereiten": "50", "pruefen": "25"}, "frist_stunden": "48", "max_seiten_auftrag": "500",
             "max_dokumente_auftrag": "50"}

    def test_standard_und_umschalten(self):
        e = express.einstellungen()
        self.assertEqual((e["korb_knopf"], e["korb_navigation"]), (True, "immer"))
        e = express.speichere_einstellungen(dict(self.BASIS, korb_knopf=False, korb_navigation="mit_inhalt"))
        self.assertEqual((e["korb_knopf"], e["korb_navigation"]), (False, "mit_inhalt"))
        e = express.speichere_einstellungen(dict(self.BASIS))                 # ohne Angabe: bleibt, wie es war
        self.assertEqual((e["korb_knopf"], e["korb_navigation"]), (False, "mit_inhalt"))
        with self.assertRaises(express.ExpressFehler) as fe:
            express.speichere_einstellungen(dict(self.BASIS, korb_navigation="manchmal"))
        self.assertEqual(fe.exception.extra.get("feld"), "korb_navigation")

    def test_korb_kurz_zaehlt_nur_den_eigenen_entwurf(self):
        self.assertEqual(express.korb_kurz(1), {"dokumente": 0, "seiten": 0})
        express.dokumente_hinzufuegen(1, [self.d1, self.d2])
        self.assertEqual(express.korb_kurz(1), {"dokumente": 2, "seiten": 3})
        self.assertEqual(express.korb_kurz(2), {"dokumente": 0, "seiten": 0})
        self._guthaben(1000)
        self._bestellen()                                                      # bestellt = kein Warenkorb mehr
        self.assertEqual(express.korb_kurz(1)["dokumente"], 0)

    def test_knopf_legt_nur_eigene_dokumente_und_meldet_doppelte(self):
        # Der Knopf am Dokument nutzt denselben Weg wie die Express-Seite (dokumente_hinzufuegen): Besitz im SQL.
        r = express.dokumente_hinzufuegen(1, [self.d1])
        self.assertEqual(r["hinzugefuegt"], 1)
        r = express.dokumente_hinzufuegen(1, [self.d1])
        self.assertEqual(r["hinzugefuegt"], 0)
        self.assertIn(self.d1, [p["document_id"] for p in r["warenkorb"]["positionen"]])
        with self.assertRaises(express.ExpressFehler) as fe:
            express.dokumente_hinzufuegen(2, [self.d1])                         # fremdes Dokument
        self.assertEqual(fe.exception.status, 404)

    def test_schalter_fuer_oberflaeche_und_navigation(self):
        import express_api
        import funktionen
        express.dokumente_hinzufuegen(1, [self.d1])
        with mock.patch.object(funktionen, "EXPRESS", True):
            self.assertEqual(express_api.fuer_oberflaeche(), {"express_korb_knopf": True})
            self.assertEqual(express_api.korb_fuer_me(1), {"modus": "immer", "dokumente": 1})
            express.speichere_einstellungen(dict(self.BASIS, korb_knopf=False, korb_navigation="aus"))
            self.assertEqual(express_api.fuer_oberflaeche(), {"express_korb_knopf": False})
            self.assertIsNone(express_api.korb_fuer_me(1))
            express.speichere_einstellungen(dict(self.BASIS, korb_knopf=True, korb_navigation="mit_inhalt"))
            self.assertEqual(express_api.korb_fuer_me(2), {"modus": "mit_inhalt", "dokumente": 0})
        with mock.patch.object(funktionen, "EXPRESS", False):                  # Schalter aus: beides weg
            self.assertEqual(express_api.fuer_oberflaeche(), {"express_korb_knopf": False})
            self.assertIsNone(express_api.korb_fuer_me(1))


class NurPdf(unittest.TestCase):
    def test_heute_nur_pdf(self):
        self.assertEqual([t.schluessel for t in express.angebotene_dateitypen()], ["pdf"])
        self.assertEqual(express.typen_text(), "PDF")
        self.assertEqual(list(express.LEISTUNGEN), ["aufbereiten", "pruefen"])
        self.assertEqual(express.AKTION_JE_LEISTUNG, {"aufbereiten": "express_aufbereiten", "pruefen": "express_pruefen"})


class Schalter(unittest.TestCase):
    def test_umgebung(self):
        import importlib
        import funktionen
        try:
            for umg, erwartet in (({"EXPRESS_SERVICE": "an", "DEMO_MODE": "off"}, True),
                                  ({"EXPRESS_SERVICE": "", "DEMO_MODE": "off"}, False),
                                  ({"EXPRESS_SERVICE": "an", "DEMO_MODE": "on"}, False)):
                with mock.patch.dict(os.environ, umg):
                    self.assertIs(importlib.reload(funktionen).EXPRESS, erwartet, umg)
        finally:
            importlib.reload(funktionen)


if __name__ == "__main__":
    unittest.main()
