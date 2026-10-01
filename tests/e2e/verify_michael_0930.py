#!/usr/bin/env python3
"""End-to-End gegen Staging (30.09.2026): Abrechnung nach Michaels Mail „Feedback 202609230 - 1“ (Punkte 11, 12) und die
Befunde aus dem Messlauf (Paket 2 a–c). Nur ueber die echte App-API mit dem Testkonto (Betreiberkonto, Guthaben unbegrenzt;
gezaehlt wird der Verbrauch in /api/me). Legt eigene Projekte an und loescht sie am Ende (ausser --behalten).

  A  Tagging: Preis 20 je Seite, genau einmal gebucht; Herunterladen ohne Bearbeitung 0 Credits (Tagging nie beim Download)
  B  Alt-Text von Hand bearbeitet: Dialog-Preis = Abrechnung = 30; derselbe Stand noch einmal: 0 (keine Doppelabbuchung);
     geaendert: wieder 30
  C  Hoerprobe: echte Bindestriche bleiben, nichts wird zusammengeklebt, langer Absatz ungekuerzt, Sprache als „Name (Kürzel)“
     (H: dasselbe am echten Dokument aus dem Messlauf, der AVV)
  D  Schon getaggte PDF: erkannt, Tagging und Testlauf 409 „schon getaggt“, nichts gebucht
  E  Feldbeschriftungen (Lbl) in der Hoerprobe (Antrag Pflege, schon getaggt, 0 Credits)
  F  Ungetaggtes Formular: nur BEARBEITETE Quickinfos kosten (26), gleicher Stand 0; ZIP mit schon bezahlten Staenden 0
  G  Lange Tabellenzellen (Rechnung INKL-002) ungekuerzt
  I  Pruefung 30.09. (H1/M2): vier gleichzeitige Downloads -> einer baut, einmal gebucht, GET /api/me bleibt schnell;
     derselbe Stand noch einmal kommt aus der Ablage (kein Neubau, kein neuer Eintrag)
  J  Nachpruefung (HOCH): zwei Dokumente „gleich.pdf“ -> ZIP, Ablage und Einzel-Downloads liefern je die eigene Datei
  K  Nachpruefung (MITTEL): Umbenennen bei eigenem Titel = kein Neubau; ohne Titel ersetzt der Neubau den kostenlosen Eintrag

Aufruf: /home/claude/.venv-pw/bin/python verify_michael_0930.py <ordner-mit-korpus> [--behalten]
  Korpus: actino_master_word.pdf, testformular_inkludocs.pdf, antrag_pflege.pdf, synth_getaggt.pdf, rechnung_inkl_002.pdf,
  probe_avv.pdf (die letzten zwei optional)
Zugang aus /home/claude/.e2e.env (INKLUDOCS_E2E_URL/MAIL/PW)."""
import io
import json
import os
import sys
import time

import requests

env = dict(os.environ)
if os.path.isfile("/home/claude/.e2e.env"):
    for z in open("/home/claude/.e2e.env"):
        if "=" in z and not z.startswith("#"):
            k, v = z.strip().split("=", 1)
            env.setdefault(k, v.strip().strip('"').strip("'"))
B = env.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = env["INKLUDOCS_E2E_MAIL"], env["INKLUDOCS_E2E_PW"]
KORPUS = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/home/claude/michael-0930"
BEHALTEN = "--behalten" in sys.argv
ok = fehler = 0
projekte = []


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK ", n)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:400])


s = requests.Session()
s.post(B + "/api/login", json={"email": MAIL, "password": PW}, timeout=30).raise_for_status()


def verbraucht():
    return (s.get(B + "/api/me", timeout=30).json().get("abo") or {}).get("verbraucht")


def projekt(name):
    r = s.post(B + "/api/projects", json={"name": name, "tool": "pdf"}, timeout=30)
    r.raise_for_status()
    pid = r.json().get("id") or r.json().get("project_id")
    projekte.append(pid)
    return pid


def hochladen(pid, name, daten):
    r = s.post(B + "/api/upload", data={"project_id": str(pid)}, files={"file": (name, daten, "application/pdf")}, timeout=300)
    r.raise_for_status()
    for _ in range(150):
        st = s.get(B + f"/api/projects/{pid}", timeout=60).json().get("project") or {}
        if st.get("status") not in ("extracting", "uploading", "processing", None):
            break
        time.sleep(2)
    docs = s.get(B + f"/api/projects/{pid}/dokument-ansicht", timeout=120).json()["documents"]
    return [d for d in docs if (d.get("original_filename") or "") == name][-1]


def taggen(pid, did):
    r = s.post(B + f"/api/projects/{pid}/documents/{did}/tagging", timeout=60)
    if not r.ok:
        return r
    for _ in range(600):
        time.sleep(1)
        st = s.get(B + f"/api/projects/{pid}/documents/{did}/tagging", timeout=60).json()
        if not st.get("laeuft") and st.get("status") in ("fertig", "fehler"):
            return st
    return {}


def hoerprobe(pid, did):
    r = s.get(B + f"/api/projects/{pid}/documents/{did}/struktur?erneuern=1", timeout=600)
    return (r.json().get("hoerprobe") or []) if r.ok else []


def export(pid, did=None):
    body = {"document_id": did} if did else {}
    return s.post(B + f"/api/projects/{pid}/export", json=body, timeout=600)


def summary(pid, did=None):
    body = {"document_id": did} if did else {}
    return s.post(B + f"/api/projects/{pid}/export/summary", json=body, timeout=120).json()


def testpdf() -> bytes:
    """2 Seiten, deutsch: echte Bindestriche mitten in der Zeile und am Zeilenende, eine Silbentrennung am Zeilenende,
    ein langer Absatz (> 600 Zeichen) und ein Bild."""
    import fitz
    d = fitz.open()
    p = d.new_page(width=595, height=842)
    p.insert_text((50, 70), "Prüfbericht zur Datenverarbeitung", fontsize=20)
    zeilen = [
        "Die KI-gestützte Prüfung der Unterlagen läuft über unsere Internet-",
        "Services und wird per E-Mail bestätigt. Für die Auftrags-",
        "verarbeitung gelten die EU-Standardvertragsklauseln und die Regeln",
        "für die Datenverarbeitung durch den Auftragnehmer im Rahmen der",
        "vereinbarten Leistungen, die in diesem Bericht beschrieben sind.",
    ]
    y = 110
    for z in zeilen:
        p.insert_text((50, y), z, fontsize=10)
        y += 14
    y += 14
    lang = ("Der Bericht fasst die Massnahmen des Landes zusammen und richtet sich an die Oeffentlichkeit und an alle, "
            "die mit der Pruefung der Unterlagen befasst sind. Die Flaeche der Schutzgebiete ist um drei Prozent gewachsen, "
            "die Mittel stammen aus dem Green Bond und werden jaehrlich neu vergeben. Wir beschreiben die Verfahren, die "
            "Zustaendigkeiten und die Fristen, damit jede Stelle ihre Aufgaben kennt und die Ergebnisse vergleichbar bleiben. "
            "Ausserdem nennen wir die Ansprechpersonen und die Wege, auf denen Rueckfragen beantwortet werden, sowie die Orte, "
            "an denen die Unterlagen eingesehen werden koennen. Am Ende steht das Schlusswort Endpunktkontrolle.")
    for i in range(0, len(lang), 95):
        p.insert_text((50, y), lang[i:i + 95], fontsize=10)
        y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0)
    for yy in range(80):
        for xx in range(120):
            pix.set_pixel(xx, yy, (int(255 * xx / 120), int(255 * yy / 80), 90))
    p.insert_image(fitz.Rect(50, y + 20, 250, y + 150), pixmap=pix)
    p2 = d.new_page(width=595, height=842)
    p2.insert_text((50, 70), "Ausblick", fontsize=14, fontname="hebo")
    p2.insert_text((50, 100), "Absatz des Ausblicks mit der Planung fuer das kommende Jahr.", fontsize=10)
    out = d.tobytes()
    d.close()
    return out


try:
    print("== A. Tagging 20 Credits je Seite, Herunterladen ohne Bearbeitung kostenlos ==")
    pid = projekt("Abrechnung 30.09. (Test) " + time.strftime("%H:%M"))
    doc = hochladen(pid, "abrechnung_test.pdf", testpdf())
    did = doc["id"]
    st = s.get(B + f"/api/projects/{pid}/documents/{did}/tagging", timeout=60).json()
    check("Stand: 2 Seiten, Preis 40, 20 je Seite, Quelle nicht getaggt", st.get("seiten") == 2 and st.get("preis") == 40 and st.get("preis_je_seite") == 20 and st.get("quelle_getaggt") is False, {k: st.get(k) for k in ("seiten", "preis", "preis_je_seite", "quelle_getaggt")})
    v0 = verbraucht()
    st = taggen(pid, did)
    v1 = verbraucht()
    check("Tagging fertig", isinstance(st, dict) and st.get("status") == "fertig", st if isinstance(st, dict) else st.status_code)
    check("Tagging hat genau 40 Credits gebucht", isinstance(v0, int) and v1 - v0 == 40, (v0, v1))
    sm = summary(pid, did)
    check("Dialog vorher: getaggt, nichts bearbeitet -> Preis 0", sm.get("preis") == 0 and sm.get("alt_bearbeitet") == 0 and sm.get("qi_bearbeitet") == 0 and sm.get("dokumente_getaggt") == 1, {k: sm.get(k) for k in ("preis", "alt_bearbeitet", "qi_bearbeitet", "dokumente_getaggt", "schon_bezahlt")})
    r = export(pid, did)
    v2 = verbraucht()
    check("Herunterladen ohne Bearbeitung: PDF, X-Export-Credits 0, nichts gebucht (Tagging nie beim Download)", r.ok and r.content[:5] == b"%PDF-" and r.headers.get("x-export-credits") == "0" and v2 == v1, (r.status_code, r.headers.get("x-export-credits"), v1, v2))

    print("== B. Alt-Text von Hand: 30 Credits, derselbe Stand kein zweites Mal ==")
    bilder = s.get(B + f"/api/projects/{pid}", timeout=60).json().get("images") or []
    check("Ein Bild nach dem Tagging", len(bilder) == 1, len(bilder))
    s.post(B + f"/api/images/{bilder[0]['id']}/alt-text", json={"alt_text": "Farbverlauf von Blau nach Rot, fiktives Testbild"}, timeout=30)
    sm = summary(pid, did)
    check("Dialog vorher: 1 bearbeiteter Alt-Text -> 30 Credits", sm.get("preis") == 30 and sm.get("alt_bearbeitet") == 1, {k: sm.get(k) for k in ("preis", "alt_bearbeitet", "qi_bearbeitet")})
    r = export(pid, did)
    v3 = verbraucht()
    check("Herunterladen: Header 30 = gebucht 30 (Anzeige = Abrechnung)", r.ok and r.headers.get("x-export-credits") == "30" and v3 - v2 == 30, (r.headers.get("x-export-credits"), v2, v3))
    sm = summary(pid, did)
    check("Dialog danach: derselbe Stand schon bezahlt -> 0", sm.get("preis") == 0 and sm.get("schon_bezahlt") == 1, {k: sm.get(k) for k in ("preis", "schon_bezahlt")})
    r = export(pid, did)
    v4 = verbraucht()
    check("Zweites Herunterladen desselben Stands: 0 Credits, keine Doppelabbuchung", r.ok and r.headers.get("x-export-credits") == "0" and r.headers.get("x-export-schon-bezahlt") == "1" and v4 == v3, (r.headers.get("x-export-credits"), r.headers.get("x-export-schon-bezahlt"), v3, v4))
    s.post(B + f"/api/images/{bilder[0]['id']}/alt-text", json={"alt_text": "Farbverlauf von Blau nach Rot, fiktives Testbild, nachgebessert"}, timeout=30)
    check("Nach einer Änderung kostet es wieder 30", summary(pid, did).get("preis") == 30)

    print("== C. Hörprobe: Bindestriche, Silbentrennung, keine Kürzung, Sprache ==")
    hp = hoerprobe(pid, did)
    text = "\n".join(hp)
    # Mitten in der Zeile bleibt der Bindestrich. Am Zeilenende macht PDFix den Strich zum Artefakt (belegt im
    # Seiteninhalt, 30.09.2026): im Tag steht dann „Internet“ + „Services“ — die Hoerprobe liest, was im Tag steht, und
    # klebt die Woerter nicht mehr zusammen (vorher „InternetServices“).
    check("Echte Bindestriche mitten in der Zeile bleiben: KI-gestützte, E-Mail, EU-Standardvertragsklauseln", all(w in text for w in ("KI-gestützte", "E-Mail", "EU-Standardvertragsklauseln")), text[:900])
    check("Nichts zusammengeklebt: kein KIgestützte, InternetServices, EMail, EUStandard…", not any(w in text for w in ("KIgestützte", "InternetServices", "EMail", "EUStandard")), text[:900])
    check("Langer Absatz ungekürzt (Schlusswort am Ende da, kein „…“)", "Endpunktkontrolle" in text and not any(z.rstrip().endswith("…") for z in hp), [z[-60:] for z in hp if len(z) > 400])
    check("Sprache als Name (Kürzel): „Sprache: Deutsch (de-DE)“", hp[:1] == ["Sprache: Deutsch (de-DE)"], hp[:1])

    print("== D. Schon getaggte PDF: „Neu taggen“ ersetzt die Tags (Michael Karbe, Feedback 20261001 - 1, Punkt 1) ==")
    act = os.path.join(KORPUS, "actino_master_word.pdf")
    doc2 = hochladen(pid, "actino_master_word.pdf", open(act, "rb").read())
    did2 = doc2["id"]
    st2 = s.get(B + f"/api/projects/{pid}/documents/{did2}/tagging", timeout=60).json()
    check("Stand: getaggt, Quelle schon getaggt, „Neu taggen“ möglich, Preis wie Tagging (20 je Seite)",
          st2.get("getaggt") is True and st2.get("quelle_getaggt") is True and st2.get("neu_taggen") is True
          and st2.get("preis") == 20 * (st2.get("seiten") or 0) and st2.get("preis") > 0, {k: st2.get(k) for k in ("getaggt", "quelle_getaggt", "neu_taggen", "preis", "seiten")})
    v5 = verbraucht()
    r = s.post(B + f"/api/projects/{pid}/documents/{did2}/tagging/test", timeout=60)
    check("„Testweise taggen“ auf der schon getaggten PDF startet (kostenlos)", r.status_code == 200, (r.status_code, r.text[:200]))
    tst = {}
    for _ in range(90):
        time.sleep(2)
        tst = (s.get(B + f"/api/projects/{pid}/documents/{did2}/tagging", timeout=60).json().get("test") or {})
        if not tst.get("laeuft") and (tst.get("struktur") or tst.get("fehler")):
            break
    check("Testlauf hat die Tags wirklich ersetzt (PDFix „Replace Existing Tags“: andere Struktur als vorher, 175 Elemente)",
          tst.get("tags_ersetzt") is True and tst.get("vorher_elemente") == 175 and (tst.get("struktur") or {}).get("elemente") not in (None, 175),
          {k: tst.get(k) for k in ("tags_ersetzt", "vorher_elemente", "struktur", "fehler")})
    check("Testlauf kostet nichts", verbraucht() == v5, (v5, verbraucht()))
    antrag_d = os.path.join(KORPUS, "antrag_pflege.pdf")
    doc3 = hochladen(pid, "antrag_neu_taggen.pdf", open(antrag_d, "rb").read())
    st3 = s.get(B + f"/api/projects/{pid}/documents/{doc3['id']}/tagging", timeout=60).json()
    v6 = verbraucht()
    r = s.post(B + f"/api/projects/{pid}/documents/{doc3['id']}/tagging", timeout=60)
    check("„Neu taggen“ einer schon getaggten PDF (1 Seite) startet", r.status_code == 200 and st3.get("quelle_getaggt") is True, (r.status_code, r.text[:200]))
    for _ in range(90):
        time.sleep(2)
        st3 = s.get(B + f"/api/projects/{pid}/documents/{doc3['id']}/tagging", timeout=60).json()
        if not st3.get("laeuft"):
            break
    bericht3 = st3.get("bericht") or {}
    check("Lauf fertig, Tags ersetzt, Preis wie Tagging gebucht (20 Credits)",
          st3.get("status") == "fertig" and bericht3.get("tags_ersetzt") is True and verbraucht() - v6 == 20,
          (st3.get("status"), {k: bericht3.get(k) for k in ("tags_ersetzt", "vorher", "nachher", "fehler")}, verbraucht() - v6))

    print("== D2. Problemstellen mit Seite auch für „Bild ohne Alt-Text“ (Feedback 20261001 - 1, Punkt 10) ==")
    doc4 = hochladen(pid, "actino_pruefung.pdf", open(act, "rb").read())
    r = s.post(B + f"/api/projects/{pid}/documents/{doc4['id']}/abschluss", timeout=600)
    det4 = {}
    for _ in range(60):
        det4 = s.get(B + f"/api/projects/{pid}/documents/{doc4['id']}/abschluss", timeout=300).json()
        if det4.get("probleme") is not None and not det4.get("laeuft"):
            break
        time.sleep(3)
    bild = [p for p in (det4.get("probleme") or []) if "7.3-1" in (p.get("regeln") or [])]
    hp_seiten = [x["seite"] for x in ((det4.get("hoerprobe") or {}).get("seiten") or [])]
    check("Prüfung erstellt; „Ein Bild hat keinen Alternativtext“ (7.3-1) hat jetzt eine Seite — Seitenansicht möglich",
          r.status_code == 200 and bild and bild[0].get("seiten") and bild[0].get("seite") in hp_seiten,
          (r.status_code, [(p.get("regeln"), p.get("seiten")) for p in (det4.get("probleme") or [])]))

    print("== E. Feldbeschriftungen (Lbl) in der Hörprobe ==")
    antrag = os.path.join(KORPUS, "antrag_pflege.pdf")
    if os.path.isfile(antrag):
        pid_a = projekt("Lbl-Hörprobe 30.09. (Test) " + time.strftime("%H:%M"))
        doc_a = hochladen(pid_a, "antrag_pflege.pdf", open(antrag, "rb").read())
        hp_a = hoerprobe(pid_a, doc_a["id"])
        besch = [z for z in hp_a if z.startswith("Beschriftung: ")]
        check("Antrag Pflege: Feldbeschriftungen werden vorgelesen (mindestens 15 „Beschriftung: …“)", len(besch) >= 15, (len(besch), besch[:3]))
    else:
        print("  (übersprungen: antrag_pflege.pdf fehlt)")

    print("== F. Ungetaggtes Formular: nur bearbeitete Quickinfos kosten; ZIP ohne Doppelabbuchung ==")
    form = os.path.join(KORPUS, "testformular_inkludocs.pdf")
    doc3 = hochladen(pid, "testformular_inkludocs.pdf", open(form, "rb").read())
    did3 = doc3["id"]
    sm = summary(pid, did3)
    check("Formular ungetaggt, keine Quickinfo bearbeitet: 0 Credits, unverändert", sm.get("preis") == 0 and sm.get("dokumente_unveraendert") == 1, {k: sm.get(k) for k in ("preis", "dokumente_unveraendert", "dokumente_quickinfos")})
    felder = [f for f in (s.get(B + f"/api/projects/{pid}/felder", timeout=60).json().get("felder") or []) if f.get("document_id") == did3]
    s.patch(B + f"/api/felder/{felder[0]['id']}", json={"quickinfo": "Fiktive Quickinfo für den Abrechnungstest"}, timeout=30)
    sm = summary(pid, did3)
    check("1 bearbeitete Quickinfo: 26 Credits (25 + 1)", sm.get("preis") == 26 and sm.get("qi_bearbeitet") == 1 and sm.get("dokumente_quickinfos") == 1, {k: sm.get(k) for k in ("preis", "qi_bearbeitet", "dokumente_quickinfos")})
    v6 = verbraucht()
    r = export(pid, did3)
    v7 = verbraucht()
    check("Herunterladen: PDF mit Quickinfo, 26 gebucht", r.ok and r.headers.get("x-export-unveraendert") == "0" and r.headers.get("x-export-credits") == "26" and v7 - v6 == 26, (r.headers.get("x-export-credits"), v6, v7))
    # Alt-Text von B wieder auf den bezahlten Stand zuruecksetzen geht nicht (anderer Text) -> erst einzeln bezahlen, dann ZIP
    r = export(pid, did)
    v8 = verbraucht()
    check("Einzeln: geänderter Alt-Text 30", r.headers.get("x-export-credits") == "30" and v8 - v7 == 30, (r.headers.get("x-export-credits"), v7, v8))
    r = export(pid)
    v9 = verbraucht()
    check("ZIP aller Dokumente, alle Stände schon bezahlt: 0 Credits, keine Doppelabbuchung", r.ok and r.content[:2] == b"PK" and r.headers.get("x-export-credits") == "0" and v9 == v8, (r.status_code, r.headers.get("x-export-credits"), r.headers.get("x-export-schon-bezahlt"), v8, v9))

    print("== I. Gleichzeitige Downloads (Prüfung H1/M2): einer baut, einmal gebucht, App bleibt flüssig; Ablage statt Neubau ==")
    s.post(B + f"/api/images/{bilder[0]['id']}/alt-text", json={"alt_text": "Farbverlauf von Blau nach Rot, fiktives Testbild, dritte Fassung"}, timeout=30)
    check("Neuer Stand: 30 Credits im Dialog", summary(pid, did).get("preis") == 30)
    anzahl_ablage = lambda: len(s.get(B + f"/api/ausgaben?projekt={pid}", timeout=60).json().get("ausgaben") or [])
    a0 = anzahl_ablage()
    v10 = verbraucht()
    import threading
    ergebnisse = []

    def _laden():
        ergebnisse.append(export(pid, did))
    faeden = [threading.Thread(target=_laden) for _ in range(4)]
    for f in faeden:
        f.start()
    time.sleep(0.4)
    t0 = time.time()
    me = s.get(B + "/api/me", timeout=30)
    dauer = time.time() - t0
    for f in faeden:
        f.join()
    v11 = verbraucht()
    codes = sorted(r.status_code for r in ergebnisse)
    ok200 = [r for r in ergebnisse if r.status_code == 200]
    texte = [((r.json() or {}).get("detail") or "") for r in ergebnisse if r.status_code == 429]
    check("4 gleichzeitige Klicks: genau einer baut (200), die anderen 429 mit Text „wird gerade schon eine PDF erstellt“",
          codes.count(200) == 1 and codes.count(429) == 3 and all("gerade schon eine PDF" in t for t in texte), (codes, texte[:1]))
    check("Genau einmal 30 Credits gebucht (Header = Verbrauch)", len(ok200) == 1 and ok200[0].headers.get("x-export-credits") == "30" and v11 - v10 == 30, ([r.headers.get("x-export-credits") for r in ok200], v10, v11))
    check("Während des Baus antwortet die App sofort (GET /api/me unter 1 s, vorher 2,77 s)", me.ok and dauer < 1.0, round(dauer, 2))
    a1 = anzahl_ablage()
    check("Ein neuer Ablage-Eintrag für den bezahlten Stand", a1 == a0 + 1, (a0, a1))
    r = export(pid, did)
    v12 = verbraucht()
    check("Noch einmal derselbe Stand: aus der Ablage (X-Export-Aus-Ablage 1), 0 Credits, kein neuer Eintrag",
          r.ok and r.content[:5] == b"%PDF-" and r.headers.get("x-export-aus-ablage") == "1" and r.headers.get("x-export-credits") == "0" and v12 == v11 and anzahl_ablage() == a1,
          (r.status_code, r.headers.get("x-export-aus-ablage"), r.headers.get("x-export-credits"), v11, v12, anzahl_ablage()))
    check("Die Datei aus der Ablage ist dieselbe wie die bezahlte", r.content == ok200[0].content if ok200 else False)

    print("== J. Gleichnamige Dokumente (Nachprüfung, HOCH): jede Datei gehört zu ihrem Dokument, einzeln und im ZIP ==")
    import fitz
    import zipfile

    def seiten(daten: bytes) -> int:
        with fitz.open(stream=daten, filetype="pdf") as d:
            return d.page_count

    def titel(daten: bytes) -> str:
        with fitz.open(stream=daten, filetype="pdf") as d:
            return (d.metadata or {}).get("title") or ""

    def ablage_von(p_id):
        return s.get(B + f"/api/ausgaben?projekt={p_id}", timeout=60).json().get("ausgaben") or []
    pid_g = projekt("Gleicher Name 30.09. (Test) " + time.strftime("%H:%M"))
    for datei in ("actino_master_word.pdf", "antrag_pflege.pdf"):
        hochladen(pid_g, "gleich.pdf", open(os.path.join(KORPUS, datei), "rb").read())
    docs_g = sorted(s.get(B + f"/api/projects/{pid_g}/dokument-ansicht", timeout=120).json()["documents"], key=lambda d: d["id"])
    erwartet = {docs_g[0]["id"]: 10, docs_g[1]["id"]: 1}
    check("Zwei Dokumente „gleich.pdf“ (10 und 1 Seite)", len(docs_g) == 2 and all(d["original_filename"] == "gleich.pdf" for d in docs_g))
    r = export(pid_g)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        im_zip = [seiten(zf.read(n)) for n in sorted(zf.namelist())]
    check("ZIP: 10 und 1 Seite", r.ok and im_zip == [10, 1], im_zip)
    eintraege = ablage_von(pid_g)
    passend = [(e["document_id"], seiten(s.get(B + f"/api/ausgaben/{e['id']}/datei", timeout=60).content)) for e in eintraege]
    check("Ablage nach dem ZIP: jeder Eintrag hat die Datei SEINES Dokuments (vorher beide 1 Seite)",
          len(passend) == 2 and all(erwartet.get(d) == n for d, n in passend), passend)
    for d in docs_g:
        r = export(pid_g, d["id"])
        check(f"Einzeln danach Dokument {d['id']}: aus der Ablage, {erwartet[d['id']]} Seiten",
              r.ok and r.headers.get("x-export-aus-ablage") == "1" and seiten(r.content) == erwartet[d["id"]],
              (r.status_code, r.headers.get("x-export-aus-ablage"), seiten(r.content) if r.ok else None))
    r = export(pid_g)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        im_zip = [seiten(zf.read(n)) for n in sorted(zf.namelist())]
    check("Zweites ZIP: wieder 10 und 1 Seite, kein neuer Ablage-Eintrag", im_zip == [10, 1] and len(ablage_von(pid_g)) == 2, (im_zip, len(ablage_von(pid_g))))

    print("== K. Umbenennen und kostenlose Neubauten (Nachprüfung, MITTEL) ==")
    pid_k = projekt("Umbenennen 30.09. (Test) " + time.strftime("%H:%M"))
    d_titel = hochladen(pid_k, "mit_titel.pdf", open(os.path.join(KORPUS, "actino_master_word.pdf"), "rb").read())
    d_ohne = hochladen(pid_k, "ohne_titel.pdf", open(os.path.join(KORPUS, "synth_getaggt.pdf"), "rb").read())
    export(pid_k, d_titel["id"])
    n0 = len(ablage_von(pid_k))
    s.patch(B + f"/api/projects/{pid_k}/documents/{d_titel['id']}", json={"display_name": "Neuer Name (Test)"}, timeout=30)
    r = export(pid_k, d_titel["id"])
    check("Datei mit eigenem Titel: Umbenennen erzwingt keinen Neubau (aus der Ablage, kein neuer Eintrag, Titel bleibt)",
          r.ok and r.headers.get("x-export-aus-ablage") == "1" and len(ablage_von(pid_k)) == n0 and titel(r.content) == "Actino Software Testdokument",
          (r.headers.get("x-export-aus-ablage"), n0, len(ablage_von(pid_k)), titel(r.content) if r.ok else None))
    export(pid_k, d_ohne["id"])
    n1 = len(ablage_von(pid_k))
    alt_ids = {e["id"] for e in ablage_von(pid_k) if e["document_id"] == d_ohne["id"]}
    s.patch(B + f"/api/projects/{pid_k}/documents/{d_ohne['id']}", json={"display_name": "Titel aus dem Namen (Test)"}, timeout=30)
    r = export(pid_k, d_ohne["id"])
    neu_ids = {e["id"] for e in ablage_von(pid_k) if e["document_id"] == d_ohne["id"]}
    check("Datei ohne Titel: der Name wird Titel, Neubau ERSETZT den kostenlosen Eintrag (Anzahl gleich, neue Datei)",
          r.ok and r.headers.get("x-export-aus-ablage") is None and titel(r.content) == "Titel aus dem Namen (Test)"
          and len(ablage_von(pid_k)) == n1 and len(neu_ids) == 1 and not (neu_ids & alt_ids),
          (r.headers.get("x-export-aus-ablage"), titel(r.content) if r.ok else None, n1, len(ablage_von(pid_k)), alt_ids, neu_ids))
    r = export(pid_k, d_ohne["id"])
    check("Noch einmal: aus der Ablage (der ersetzende Eintrag trägt den Stand)", r.headers.get("x-export-aus-ablage") == "1" and len(ablage_von(pid_k)) == n1)

    print("== H. Bindestriche im echten Dokument (AVV, Messlauf: „KIgestützte“, „EUStandardvertragsklauseln“, „EMail“) ==")
    avv = os.path.join(KORPUS, "probe_avv.pdf")
    if os.path.isfile(avv):
        pid_v = projekt("Bindestriche AVV 30.09. (Test) " + time.strftime("%H:%M"))
        doc_v = hochladen(pid_v, "probe_avv.pdf", open(avv, "rb").read())
        st_v = taggen(pid_v, doc_v["id"])
        tv = "\n".join(hoerprobe(pid_v, doc_v["id"])) if isinstance(st_v, dict) and st_v.get("status") == "fertig" else ""
        check("AVV: KI-gestützte, EU-Standardvertragsklauseln, E-Mail mit Bindestrich, nichts zusammengeklebt",
              all(w in tv for w in ("KI-gestützte", "EU-Standardvertragsklauseln", "E-Mail")) and not any(w in tv for w in ("KIgestützte", "EUStandardvertragsklauseln", "EMail")), tv[:600])
    else:
        print("  (übersprungen: probe_avv.pdf fehlt)")

    print("== G. Lange Tabellenzellen ungekürzt (Rechnung INKL-002) ==")
    inkl = os.path.join(KORPUS, "rechnung_inkl_002.pdf")
    if os.path.isfile(inkl):
        pid_r = projekt("Tabellenzellen 30.09. (Test) " + time.strftime("%H:%M"))
        doc_r = hochladen(pid_r, "rechnung_inkl_002.pdf", open(inkl, "rb").read())
        st_r = taggen(pid_r, doc_r["id"])
        hp_r = hoerprobe(pid_r, doc_r["id"]) if isinstance(st_r, dict) and st_r.get("status") == "fertig" else []
        zellen = [z for z in hp_r if z.startswith(("Zeile: ", "Kopfzeile: "))]
        laengste = max((len(x) for z in zellen for x in z.split(" | ")), default=0)
        check("Tabellenzeilen ohne Kürzung: keine Zelle endet auf „…“, längste Zelle über 80 Zeichen", zellen and laengste > 80 and not any(x.rstrip().endswith("…") for z in zellen for x in z.split(" | ")), (laengste, zellen[:4]))
    else:
        print("  (übersprungen: rechnung_inkl_002.pdf fehlt)")
finally:
    if not BEHALTEN:
        for p in projekte:
            print("  Testprojekt geloescht:", p, s.delete(B + f"/api/projects/{p}", timeout=60).status_code)
    else:
        print("  Projekte bleiben stehen:", projekte)

print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
