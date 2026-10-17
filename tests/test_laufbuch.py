"""Laufbuch (backend/laufbuch.py, 17.10.2026): was in diesem Prozess gerade im Hintergrund arbeitet — die eine Stelle, an die sich
jeder Lauf anschließt (Alt-Texte, Quickinfos, Tagging, Prüfung, Umwandeln, Übersetzen, Export, Aufträge).

Prüft: anmelden, Stück, Fortschritt, abmelden — auch bei Ausnahme und Abbruch; das vorige Stück wird „vorher“; nur eigene Läufe;
nur Läufe des Agenten (agent_kontext, auch über asyncio-Aufgaben hinweg; Threads bekommen quelle mitgegeben); laufende zuerst;
beendete verschwinden nach NACHLAUF_S; unbekannte Arten werden nicht gezeigt; nur Kennungen, keine Texte; nie eine Ausnahme;
sicher aus vielen Threads.
"""
import asyncio
import os
import sys
import threading
import unittest
from unittest import mock

HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in ("/app", os.path.normpath(os.path.join(HIER, "..", "backend"))):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import laufbuch  # noqa: E402


class Laufbuch(unittest.TestCase):
    def setUp(self):
        laufbuch._leeren()

    def test_anmelden_stueck_fortschritt_abmelden(self):
        with laufbuch.agent_kontext():
            with laufbuch.lauf("alt_texte", user_id=7, project_id=3, gesamt=2) as l:
                l.stueck(image_id=11)
                d = laufbuch.fuer_konto(7)[0]
                self.assertEqual((d["art"], d["project_id"], d["gesamt"], d["fertig"], d["laeuft"]), ("alt_texte", 3, 2, 0, True))
                self.assertEqual(d["stueck"], {"art": "bild", "image_id": 11})
                l.weiter()
                l.stueck(image_id=12)
                d = laufbuch.fuer_konto(7)[0]
                self.assertEqual((d["fertig"], d["stueck"]["image_id"], d["vorher"]["image_id"]), (1, 12, 11))
                l.weiter()
        d = laufbuch.fuer_konto(7)[0]
        self.assertEqual((d["laeuft"], d["ok"], d["stueck"], d["vorher"]["image_id"]), (False, True, None, 12),
                         "am Ende ist das letzte Stück fertig")

    def test_abbruch_und_ausnahme_tragen_aus(self):
        with laufbuch.agent_kontext():
            with self.assertRaises(RuntimeError):
                with laufbuch.lauf("tagging", user_id=7, project_id=3, document_id=5) as l:
                    l.stueck(document_id=5)
                    raise RuntimeError("PDFix weg (fiktiv)")
        d = laufbuch.fuer_konto(7)[0]
        self.assertEqual((d["laeuft"], d["ok"]), (False, False))
        self.assertEqual(d["stueck"], {"art": "dokument", "document_id": 5}, "bei Fehler bleibt sichtbar, wo es war")

        async def abgebrochen():
            with laufbuch.lauf("alt_texte", user_id=8, project_id=4, quelle="agent"):
                raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(abgebrochen())
        self.assertFalse(laufbuch.fuer_konto(8)[0]["laeuft"])
        self.assertFalse(laufbuch.laeuft_im_projekt(4))

    def test_nur_eigene_und_nur_agent(self):
        laufbuch.beginnen("export", user_id=7, project_id=3)                      # Oberfläche: nicht gestreamt
        with laufbuch.agent_kontext():
            laufbuch.beginnen("export", user_id=9, project_id=4)                  # fremdes Konto
            a = laufbuch.beginnen("export", user_id=7, project_id=3)
        self.assertEqual([d["nr"] for d in laufbuch.fuer_konto(7)], [a.nr])
        self.assertEqual(len(laufbuch.fuer_konto(7, nur_agent=False)), 2)
        self.assertEqual(laufbuch.fuer_konto("x"), [])

    def test_agent_kontext_geht_an_asyncio_aufgaben_threads_bekommen_quelle(self):
        async def im_werkzeug():
            async def hintergrund():
                return laufbuch.beginnen("alt_texte", user_id=7, project_id=3)
            return await asyncio.create_task(hintergrund())
        with laufbuch.agent_kontext():
            l = asyncio.run(im_werkzeug())
        self.assertEqual(l.quelle, "agent")
        ergebnis = {}
        with laufbuch.agent_kontext():
            q = "agent" if laufbuch.durch_agent() else "oberflaeche"
            t = threading.Thread(target=lambda: ergebnis.update(l=laufbuch.beginnen("tagging", user_id=7, quelle=q)))
            t.start()
            t.join()
        self.assertEqual(ergebnis["l"].quelle, "agent")
        self.assertFalse(laufbuch.durch_agent(), "außerhalb des Werkzeugs nicht mehr")

    def test_laufende_zuerst_und_nachlauf(self):
        with laufbuch.agent_kontext():
            alt = laufbuch.beginnen("export", user_id=7, project_id=3)
            alt.beenden()
            neu = laufbuch.beginnen("tagging", user_id=7, project_id=3)
        self.assertEqual([d["nr"] for d in laufbuch.fuer_konto(7)], [neu.nr, alt.nr])
        with mock.patch.object(laufbuch.time, "time", return_value=alt.ende + laufbuch.NACHLAUF_S + 1):
            self.assertEqual([d["nr"] for d in laufbuch.fuer_konto(7)], [neu.nr], "beendet und alt: weg")

    def test_unbekannte_art_und_kaputte_angaben(self):
        with laufbuch.agent_kontext():
            l = laufbuch.beginnen("raketenstart", user_id=7)
            l.stueck(image_id=1)
            l.beenden()
            laufbuch.beginnen("export", user_id=None)
        self.assertEqual(laufbuch.fuer_konto(7), [])
        with laufbuch.agent_kontext():
            l = laufbuch.beginnen("export", user_id=7, project_id="abc")
        l.stueck(image_id="x", feld_id=True)          # nichts Brauchbares: kein Stück
        l.weiter("x")
        d = laufbuch.fuer_konto(7)[0]
        self.assertEqual((d["project_id"], d["stueck"]), (None, None))

    def test_nur_kennungen_keine_texte(self):
        with laufbuch.agent_kontext():
            l = laufbuch.beginnen("umwandeln", user_id=7, project_id=3)
        l.stueck(document_id=5, seite=2)
        self.assertEqual(l.als_dict()["stueck"], {"art": "seite", "document_id": 5, "seite": 2})
        self.assertNotIn("name", str(l.als_dict()))

    def test_mitnehmen_in_threads_und_fremde_schleifen(self):
        ergebnis = {}
        with laufbuch.agent_kontext():
            ziel = laufbuch.mitnehmen(lambda: ergebnis.update(t=laufbuch.beginnen("tagging", user_id=7)))
        t = threading.Thread(target=ziel)
        t.start()
        t.join()
        self.assertEqual(ergebnis["t"].quelle, "agent", "Thread: Quelle mitgenommen")
        t2 = threading.Thread(target=lambda: ergebnis.update(o=laufbuch.beginnen("tagging", user_id=7)))
        with laufbuch.agent_kontext():
            t2.start()
            t2.join()
        self.assertEqual(ergebnis["o"].quelle, "oberflaeche", "ohne mitnehmen erbt ein Thread nichts")

        async def lauf():
            return laufbuch.beginnen("alt_texte", user_id=7)
        schleife = asyncio.new_event_loop()
        th = threading.Thread(target=schleife.run_forever)
        th.start()
        try:
            with laufbuch.agent_kontext():
                fut = asyncio.run_coroutine_threadsafe(laufbuch.mitnehmen_coro(lauf()), schleife)
            self.assertEqual(fut.result(5).quelle, "agent", "andere Schleife: Quelle mitgenommen")
        finally:
            schleife.call_soon_threadsafe(schleife.stop)
            th.join()
            schleife.close()

    def test_ergebnis_in_der_ablage(self):
        with laufbuch.agent_kontext():
            l = laufbuch.beginnen("umwandeln", user_id=7, project_id=3)
        l.ergebnis(ausgabe_id=12)
        l.ergebnis(ausgabe_id="x")
        l.beenden()
        self.assertEqual(laufbuch.fuer_konto(7)[0]["ergebnis"], {"art": "ausgabe", "ausgabe_id": 12})

    def test_station_bei_auftraegen(self):
        with laufbuch.agent_kontext():
            l = laufbuch.beginnen("auftrag", user_id=7, project_id=3, gesamt=3)
        l.station_setzen("tagging")
        self.assertEqual(laufbuch.fuer_konto(7)[0]["station"], "tagging")
        l.station_setzen("unbekannt")
        self.assertIsNone(laufbuch.fuer_konto(7)[0]["station"])

    def test_viele_threads(self):
        with laufbuch.agent_kontext():
            l = laufbuch.beginnen("alt_texte", user_id=7, project_id=3, gesamt=400)
        def arbeit(i):
            for k in range(100):
                l.stueck(image_id=i * 1000 + k)
                l.weiter()
        ts = [threading.Thread(target=arbeit, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(laufbuch.fuer_konto(7)[0]["fertig"], 400)


if __name__ == "__main__":
    unittest.main()
