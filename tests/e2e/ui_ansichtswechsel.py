#!/usr/bin/env python3
"""Klicktest Ansichtswechsel (05.10.2026, docs/ANSICHTEN_LEISTUNG.md), erweitert um alle Befunde der Pruefungen
Barrierefreiheit und Entwicklung (Korrekturrunde 05.10.2026):
  A. Tippen und SOFORT wechseln (Alt-Text, Browser-Vor, Quickinfo) — keine Eingabe geht verloren; keine feste Wartezeit.
  B. Ansicht Alt-Texte: Seitenansicht/Seitentext beim Aufklappen, Platzhalter mit aria-busy, Fokus bleibt; Seitenbild
     mit Namen und Hinweis (Befund 9); Projektantwort ohne KI-Kontext und Serverpfade.
  C. Gastansicht: Seitentext ueber den Gastweg, Seitentitel mit Projektname (Befund 10).
  D. Schneller Doppelwechsel (40 und 250 ms): Adresse, Bildschirm, Fokus und EINE Ansage gehoeren zum zweiten Klick
     (Befunde 1, 2).
  E. Browser-Zurueck: Fokus auf der H1 und genau eine Ansage (Befund 8).
  F. Speicherfehler beim Wechsel (Alt-Text, Quickinfo): Ansicht bleibt, Meldung sichtbar und angesagt, Eingabe bleibt im
     Feld, nie „Gespeichert“ bei Fehler; zweiter Klick wechselt (Befund 4).
  G. Seite verlassen nach dem Tippen: Seitenleiste, Neuladen, Tab schliessen, Abmelden — gespeichert (Befund 7 /
     Entwicklung 1).
  H. Seitentext langsam (1,5 s) und mit Fehler: „Seitentext geladen.“ bzw. Fehlersatz angesagt (Befunde 3, 5).
  I. Uebersetzung oeffnen: genau eine Ansage (Befund 6; Word-Projekt „E2E Übersetzen (fiktiv)“ des Testkontos, ID als
     Argument --uebersetzung=<id>, Standard 849).
Legt sein Projekt selbst an (klar fiktive Inhalte) und loescht es wieder (ausser --behalten).
Aufruf: /home/claude/.venv-pw/bin/python ui_ansichtswechsel.py [<testformular.pdf>] [--uebersetzung=849] [--behalten]
Braucht INKLUDOCS_E2E_URL/MAIL/PW."""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de").rstrip("/")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
HIER = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FORM = args[0] if args else os.path.join(HIER, "..", "fixtures", "testformular_inkludocs.pdf")
BEHALTEN = "--behalten" in sys.argv
UEB = int(([a.split("=", 1)[1] for a in sys.argv if a.startswith("--uebersetzung=")] or ["849"])[0])
NAMEN = {"dokument": "Dokument", "tagging": "Tagging", "alttexte": "Alt-Texte", "abschluss": "Barrierefreiheitsprüfung",
         "quickinfos": "Quickinfos", "uebersetzung": "Übersetzung"}
ok = fehler = 0
# Live-Regionen mitschreiben: jede Textaenderung INNERHALB einer Live-Region (aria-live != off, role status/alert,
# output) — so, wie ein Screenreader sie hoeren wuerde.
INIT = """
window.__ansagen = [];
window.__h1 = [];
(function () {
  const live = (el) => { for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const l = e.getAttribute('aria-live'); if (l === 'off') return null;
      if (l || ['status', 'alert'].includes(e.getAttribute('role')) || e.tagName === 'OUTPUT') return e; } return null; };
  new MutationObserver(ms => ms.forEach(m => {
      const el = m.target.nodeType === 1 ? m.target : m.target.parentElement;
      const lr = live(el);
      if (lr && lr.textContent.trim()) {
        const text = lr.textContent.trim().slice(0, 200);
        const letzte = window.__ansagen[window.__ansagen.length - 1];
        if (!letzte || letzte.text !== text || performance.now() - letzte.t > 300) window.__ansagen.push({ t: Math.round(performance.now()), text });
      }
      const h = document.getElementById('projectName');
      if (h) { const t = h.textContent.trim().slice(0, 60); if (window.__h1[window.__h1.length - 1] !== t) window.__h1.push(t); }
  })).observe(document, { subtree: true, childList: true, characterData: true });
})();
"""


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK   ", n, flush=True)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:300], flush=True)


def anmelden(ctx):
    r = ctx.request.post(B + "/api/login", data={"email": MAIL, "password": PW})
    tok = None
    for h in r.headers_array:
        if h["name"].lower() == "set-cookie" and h["value"].startswith("token="):
            tok = h["value"].split(";")[0].split("=", 1)[1]
    if B.startswith("http://") and tok:
        ctx.add_cookies([{"name": "token", "value": tok, "url": B}])
    return r.ok


def testpdf() -> bytes:
    """Zwei Seiten mit je einem Bild und Fliesstext (fiktiv)."""
    import fitz
    d = fitz.open()
    for nr in (1, 2):
        p = d.new_page(width=595, height=842)
        p.insert_text((50, 70), f"Musterstadt Umweltbericht (fiktiv), Seite {nr}", fontsize=16)
        y = 100
        for z in range(12):
            p.insert_text((50, y), f"Zeile {z + 1}: Die Messstellen in Beispielhausen zeigen fiktive Werte ({nr}).", fontsize=10)
            y += 14
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), 0)
        for yy in range(80):
            for xx in range(120):
                pix.set_pixel(xx, yy, (int(255 * xx / 120), int(255 * yy / 80), 90 + nr * 40))
        p.insert_image(fitz.Rect(50, y + 20, 250, y + 150), pixmap=pix)
    out = d.tobytes()
    d.close()
    return out


def warte_ansicht(pg, name, timeout=60000):
    pg.wait_for_function("(n) => { const h = document.getElementById('projectName'); return h && h.textContent.startsWith(n); }",
                         arg=name, timeout=timeout)


def warte_lesen(ctx, pid):
    for _ in range(200):
        time.sleep(1.0)
        if ctx.request.get(B + f"/api/projects/{pid}/status").json().get("status") != "extracting":
            return


def ansagen(pg):
    return [a["text"] for a in pg.evaluate("() => window.__ansagen || []")]


def ansagen_leeren(pg):
    pg.wait_for_timeout(400)   # verspaetete Ansage des vorigen Schritts (announce setzt den Text nach 100 ms) abwarten
    pg.evaluate("() => { window.__ansagen = []; window.__h1 = []; }")


def gezeichnet(pg):
    return pg.evaluate("() => window.__h1 || []")


def aufklappen(pg, seite=0):
    """Erstes Dokument und die gewuenschte Seite offen (setzen statt klicken: der Auf/Zu-Zustand wird gemerkt)."""
    pg.evaluate("(i) => { const d = document.querySelector('details.doc-section'); d.open = true; const s = d.querySelectorAll('details.page-section')[i]; s.open = true; }", seite)


def bild(ctx, pid, bid):
    return [b for b in ctx.request.get(B + f"/api/projects/{pid}").json()["images"] if b["id"] == bid][0]


def fokus_auf_h1(pg):
    return pg.evaluate("document.activeElement && document.activeElement.id === 'projectName'")


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
    ctx.add_init_script(INIT)
    check("Anmeldung", anmelden(ctx))
    pg = ctx.new_page()
    fehler_js = []
    pg.on("pageerror", lambda e: fehler_js.append(str(e)))
    r = ctx.request.post(B + "/api/projects", data={"name": "Klicktest Ansichtswechsel (fiktiv) " + time.strftime("%H:%M"), "tool": "pdf"})
    pid = r.json().get("id") or r.json().get("project_id")
    print("Projekt", pid, flush=True)
    try:
        r = ctx.request.post(B + "/api/upload", multipart={"file": {"name": "musterstadt_fiktiv.pdf", "mimeType": "application/pdf", "buffer": testpdf()},
                                                           "project_id": str(pid)}, timeout=180000)
        check("Upload Test-PDF", r.ok, r.status)
        warte_lesen(ctx, pid)
        with open(FORM, "rb") as fh:
            r = ctx.request.post(B + "/api/upload", multipart={"file": {"name": "testformular_fiktiv.pdf", "mimeType": "application/pdf", "buffer": fh.read()},
                                                               "project_id": str(pid)}, timeout=180000)
        check("Upload Testformular", r.ok, r.status)
        warte_lesen(ctx, pid)

        # --- B. Alt-Texte ------------------------------------------------------------------------------------------
        print("== B. Ansicht Alt-Texte", flush=True)
        voll = ctx.request.get(B + f"/api/projects/{pid}").json()
        bilder = voll.get("images") or []
        schwer = {"context_text", "page_text", "pipeline_steps", "validation_result", "image_path", "page_view_path"}
        check("Projektantwort: Bilder da, ohne KI-Kontext/Seitentext/Serverpfade", bilder and not any(schwer & set(b) for b in bilder),
              [sorted(schwer & set(b)) for b in bilder][:2])
        kopf = ctx.request.get(B + f"/api/projects/{pid}/kopf").json()
        check("/kopf: ohne Bildliste, mit Zaehlern und hat_felder", "images" not in kopf and kopf.get("bilder_gesamt", 0) >= 1 and kopf["project"].get("hat_felder", 0) > 0)
        seitenbilder = []
        pg.on("request", lambda rq: seitenbilder.append(rq.url) if "/page-view" in rq.url else None)
        pg.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="domcontentloaded")
        warte_ansicht(pg, "Alt-Texte")
        pg.wait_for_timeout(1200)
        check("Erstes Oeffnen: keine Seitenansicht geladen", not seitenbilder, seitenbilder[:3])
        aufklappen(pg, 0)
        seite = pg.locator("details.page-section").first
        st = seite.locator(":scope > details.page-text-details")
        st.locator(":scope > summary").focus()
        pg.keyboard.press("Enter")
        pg.wait_for_function("(el) => { const z = el.querySelector('.page-text-content'); return z && !z.hasAttribute('aria-busy') && z.textContent.length > 20; }",
                             arg=st.element_handle(), timeout=10000)
        z = st.locator(".page-text-content")
        check("Seitentext im benannten Bereich (role=region, Name Seitentext)", "Beispielhausen" in z.inner_text() and z.get_attribute("role") == "region" and z.get_attribute("aria-label") == "Seitentext")
        check("Fokus bleibt auf „Seitentext anzeigen“", pg.evaluate("document.activeElement && document.activeElement.tagName === 'SUMMARY' && document.activeElement.textContent.includes('Seitentext')"))
        sa = seite.locator(":scope > details.page-view-details")
        alt = sa.locator("img.page-view-image").get_attribute("alt") or ""
        check("Befund 9: Seitenbild hat einen Namen („Seitenansicht: Seite 1 als Bild“)", alt == "Seitenansicht: Seite 1 als Bild", alt)
        check("Befund 9: Hinweis auf „Seitentext anzeigen“ in der Seitenansicht", "Den Text dieser Seite gibt es unter „Seitentext anzeigen“." in (sa.text_content() or ""))
        vorher = len(seitenbilder)
        sa.locator(":scope > summary").click()
        pg.wait_for_function("(el) => { const i = el.querySelector('img.page-view-image'); return i && i.complete && i.naturalWidth > 0; }", arg=sa.element_handle(), timeout=10000)
        check("Seitenansicht laedt beim Aufklappen (genau eine Anfrage)", len(seitenbilder) == vorher + 1, seitenbilder)

        # --- A. Tippen und sofort wechseln ------------------------------------------------------------------------
        print("== A. Tippen und sofort wechseln", flush=True)
        feld = seite.locator("textarea.alt-text-field:not(.langtext-field)").first
        bild_id = int(feld.get_attribute("data-image-id"))
        text1 = "Fiktiver Testtext eins " + time.strftime("%H%M%S")
        feld.fill(text1)
        t0 = time.time()
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
        warte_ansicht(pg, "Dokument")
        dauer = (time.time() - t0) * 1000
        check("Alt-Text getippt, sofort zu „Dokument“: gespeichert", bild(ctx, pid, bild_id).get("alt_text_edited") == text1)
        check(f"Wechsel mit offener Eingabe ohne feste Wartezeit ({dauer:.0f} ms < 900 ms)", dauer < 900, dauer)
        pg.go_back()
        warte_ansicht(pg, "Alt-Texte")
        aufklappen(pg, 0)
        feld = pg.locator(f"#alttext_{bild_id}")
        check("Nach Zurueck: Feld zeigt den gespeicherten Text", feld.input_value() == text1, feld.input_value())
        text2 = "Fiktiver Testtext zwei " + time.strftime("%H%M%S")
        feld.fill(text2)
        pg.go_forward()
        warte_ansicht(pg, "Dokument")
        pg.wait_for_timeout(300)
        check("Alt-Text getippt, sofort Browser-Vor: gespeichert", bild(ctx, pid, bild_id).get("alt_text_edited") == text2)
        t0 = time.time()
        pg.click(".ansicht-knoepfe a[data-ansicht=tagging]")
        warte_ansicht(pg, "Tagging")
        dauer = (time.time() - t0) * 1000
        check(f"Wechsel ohne Eingabe: {dauer:.0f} ms (unter 900 ms)", dauer < 900, dauer)
        pg.click(".ansicht-knoepfe a[data-ansicht=quickinfos]")
        warte_ansicht(pg, "Quickinfos")
        pg.wait_for_timeout(400)
        pg.evaluate("() => document.querySelectorAll('details').forEach(d => d.open = true)")
        qf = pg.locator("textarea.quickinfo-field").first
        feld_id = int(qf.get_attribute("data-feld-id"))
        text3 = "Fiktive Quickinfo " + time.strftime("%H%M%S")
        qf.fill(text3)
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
        warte_ansicht(pg, "Dokument")
        f = [x for x in (ctx.request.get(B + f"/api/projects/{pid}/felder").json().get("felder") or []) if x["id"] == feld_id]
        check("Quickinfo getippt, sofort zu „Dokument“: gespeichert", f and f[0].get("quickinfo") == text3)

        # --- D. Schneller Doppelwechsel ---------------------------------------------------------------------------
        print("== D. Schneller Doppelwechsel (Befunde 1, 2)", flush=True)
        for erst, zweit, ms in (("tagging", "alttexte", 40), ("dokument", "quickinfos", 40), ("abschluss", "dokument", 40),
                                ("alttexte", "tagging", 40), ("quickinfos", "abschluss", 250), ("dokument", "alttexte", 40)):
            jetzt = pg.evaluate("new URLSearchParams(location.search).get('ansicht')")
            if erst == jetzt:
                continue
            ansagen_leeren(pg)
            pg.evaluate("""([a, b, ms]) => { document.querySelector('.ansicht-knoepfe a[data-ansicht=' + a + ']').click();
                setTimeout(() => { const l = document.querySelector('.ansicht-knoepfe a[data-ansicht=' + b + ']'); if (l) l.click(); }, ms); }""",
                        [erst, zweit, ms])
            pg.wait_for_timeout(2500)
            adresse = pg.evaluate("new URLSearchParams(location.search).get('ansicht')")
            h1 = pg.locator("#projectName").inner_text()
            an = [a for a in ansagen(pg) if a.startswith("Ansicht ")]
            h1s = gezeichnet(pg)
            # Jede Ansage gehoert zu einer Ansicht, die wirklich gezeichnet wurde (war der erste Wechsel schon fertig, ist
            # seine Ansage richtig); die letzte Ansage, Adresse, H1 und Fokus gehoeren zum zweiten Klick.
            verwaist = [a for a in an if not any(h.startswith(a[len("Ansicht "):-len(" geöffnet.")]) for h in h1s)]
            check(f"{erst} -> {zweit} ({ms} ms): Adresse, Bildschirm, Fokus und letzte Ansage gehoeren zu {zweit}, keine verwaiste Ansage",
                  adresse == zweit and h1.startswith(NAMEN[zweit]) and fokus_auf_h1(pg) and an and an[-1] == f"Ansicht {NAMEN[zweit]} geöffnet."
                  and not verwaist and h1s and h1s[-1].startswith(NAMEN[zweit]),
                  (adresse, h1[:30], fokus_auf_h1(pg), an, h1s, verwaist))

        # Langsame Antworten (300 ms je Abruf): der erste Wechsel ist sicher noch unterwegs, wenn der zweite Klick kommt —
        # dann darf die erste Ansicht weder gezeichnet noch angesagt werden.
        langsam = re.compile(r".*/api/projects/\d+(/kopf|/dokument-ansicht|/abschluss|/felder)?(\?.*)?$")
        pg.route(langsam, lambda rt: (time.sleep(0.3), rt.continue_()) if rt.request.method == "GET" else rt.continue_())
        for erst, zweit in (("tagging", "alttexte"), ("abschluss", "dokument"), ("alttexte", "quickinfos")):
            jetzt = pg.evaluate("new URLSearchParams(location.search).get('ansicht')")
            if erst == jetzt or zweit == jetzt:
                continue
            ansagen_leeren(pg)
            pg.evaluate("""([a, b]) => { document.querySelector('.ansicht-knoepfe a[data-ansicht=' + a + ']').click();
                setTimeout(() => document.querySelector('.ansicht-knoepfe a[data-ansicht=' + b + ']').click(), 40); }""", [erst, zweit])
            pg.wait_for_timeout(4000)
            an = [a for a in ansagen(pg) if a.startswith("Ansicht ")]
            h1s = gezeichnet(pg)
            check(f"Langsame Leitung, {erst} -> {zweit} (40 ms): {NAMEN[erst]} nie gezeichnet, genau eine Ansage fuer {zweit}",
                  an == [f"Ansicht {NAMEN[zweit]} geöffnet."] and not any(h.startswith(NAMEN[erst]) for h in h1s)
                  and pg.evaluate("new URLSearchParams(location.search).get('ansicht')") == zweit and fokus_auf_h1(pg),
                  (an, h1s))
        pg.unroute(langsam)

        # --- E. Browser-Zurueck ----------------------------------------------------------------------------------
        print("== E. Browser-Zurueck (Befund 8)", flush=True)
        ansagen_leeren(pg)
        pg.go_back()
        pg.wait_for_timeout(1500)
        adresse = pg.evaluate("new URLSearchParams(location.search).get('ansicht')")
        an = [a for a in ansagen(pg) if a.startswith("Ansicht ")]
        check("Zurueck: Fokus auf der H1 und genau eine Ansage passend zur Adresse",
              fokus_auf_h1(pg) and pg.locator("#projectName").inner_text().startswith(NAMEN.get(adresse, "?")) and an == [f"Ansicht {NAMEN.get(adresse)} geöffnet."],
              (adresse, fokus_auf_h1(pg), an))

        # --- F. Speicherfehler beim Wechsel -------------------------------------------------------------------------
        print("== F. Speicherfehler beim Wechsel (Befund 4)", flush=True)
        pg.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="domcontentloaded")
        warte_ansicht(pg, "Alt-Texte")
        aufklappen(pg, 0)
        pg.route("**/api/images/*/alt-text", lambda rt: rt.fulfill(status=500, content_type="application/json", body='{"detail":"Testfehler"}')
                 if rt.request.method == "POST" else rt.continue_())
        feld = pg.locator(f"#alttext_{bild_id}")
        text4 = "Fiktiver Text, der nicht gespeichert wird " + time.strftime("%H%M%S")
        feld.fill(text4)
        ansagen_leeren(pg)
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
        pg.wait_for_timeout(1500)
        warnung = pg.locator("#speicherWarnung")
        check("Ansicht bleibt offen (H1 und Adresse Alt-Texte)", pg.locator("#projectName").inner_text().startswith("Alt-Texte")
              and pg.evaluate("new URLSearchParams(location.search).get('ansicht')") == "alttexte")
        check("Meldung sichtbar unter dem Projektkopf", warnung.count() == 1 and warnung.is_visible() and "konnte nicht gespeichert werden" in warnung.inner_text(),
              warnung.inner_text() if warnung.count() else None)
        check("Meldung angesagt", any("Achtung: 1 Eingabe konnte nicht gespeichert werden." in a for a in ansagen(pg)), ansagen(pg))
        check("Eingabe steht noch im Feld", feld.input_value() == text4)
        check("Am Feld „Nicht gespeichert“ statt „Gespeichert“", pg.locator(f"#saved_{bild_id}").inner_text() == "Nicht gespeichert")
        pg.unroute("**/api/images/*/alt-text")
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
        warte_ansicht(pg, "Dokument")
        check("Zweiter Klick wechselt", pg.locator("#projectName").inner_text().startswith("Dokument"))
        # Quickinfo mit Fehler
        pg.click(".ansicht-knoepfe a[data-ansicht=quickinfos]")
        warte_ansicht(pg, "Quickinfos")
        pg.wait_for_timeout(300)
        pg.evaluate("() => document.querySelectorAll('details').forEach(d => d.open = true)")
        pg.route("**/api/felder/*", lambda rt: rt.fulfill(status=500, content_type="application/json", body='{"detail":"Testfehler"}')
                 if rt.request.method == "PATCH" else rt.continue_())
        pg.locator(f"textarea.quickinfo-field[data-feld-id='{feld_id}']").fill("Fiktive Quickinfo mit Fehler")
        ansagen_leeren(pg)
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
        pg.wait_for_timeout(1500)
        check("Quickinfo-Fehler: Ansicht bleibt, Meldung angesagt", pg.locator("#projectName").inner_text().startswith("Quickinfos")
              and any("konnte nicht gespeichert werden" in a for a in ansagen(pg)), ansagen(pg))
        pg.unroute("**/api/felder/*")

        # --- G. Seite verlassen ------------------------------------------------------------------------------------
        print("== G. Seite verlassen nach dem Tippen (Befund 7 / Entwicklung 1)", flush=True)
        for art in ("Seitenleiste", "Neuladen", "Tab schliessen", "Abmelden"):
            seite_g = pg if art != "Tab schliessen" else ctx.new_page()
            seite_g.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="domcontentloaded")
            warte_ansicht(seite_g, "Alt-Texte")
            aufklappen(seite_g, 0)
            text = f"Fiktiver Text vor {art} " + time.strftime("%H%M%S")
            seite_g.locator(f"#alttext_{bild_id}").fill(text)
            if art == "Seitenleiste":
                seite_g.locator("a[href='/projekte']").first.click()
                seite_g.wait_for_url("**/projekte*", timeout=20000)
            elif art == "Neuladen":
                seite_g.reload(wait_until="domcontentloaded")
            elif art == "Tab schliessen":
                seite_g.close()
            else:
                # Runde 10: Abmelden steht unter „Konto“ — ohne Fokuswechsel aufklappen, damit der Klick auf
                # „Abmelden“ wie bisher der erste Schritt weg vom Feld ist
                seite_g.evaluate("() => { document.getElementById('navKonto').open = true; }")
                seite_g.click("#logoutBtn")
                seite_g.wait_for_url(B + "/", timeout=20000)
                anmelden(ctx)
            time.sleep(1.0)
            check(f"{art} direkt nach dem Tippen: gespeichert", bild(ctx, pid, bild_id).get("alt_text_edited") == text,
                  bild(ctx, pid, bild_id).get("alt_text_edited"))

        # --- H. Seitentext langsam und mit Fehler ------------------------------------------------------------------
        print("== H. Seitentext langsam und mit Fehler (Befunde 3, 5)", flush=True)
        pg.route("**/seitentext", lambda rt: (time.sleep(1.5), rt.continue_()))
        pg.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="domcontentloaded")
        warte_ansicht(pg, "Alt-Texte")
        ansagen_leeren(pg)
        pg.evaluate("""() => { const d = document.querySelector('details.doc-section'); d.open = true;
            const s = d.querySelectorAll('details.page-section')[0]; s.open = true;
            setTimeout(() => { s.querySelector(':scope > details.page-text-details').open = true; }, 50); }""")
        pg.wait_for_timeout(3000)
        check("Langsam, Klappe gleich mit der Seite geoeffnet: „Seitentext geladen.“ angesagt", "Seitentext geladen." in ansagen(pg), ansagen(pg))
        ansagen_leeren(pg)
        pg.evaluate("""() => { const d = document.querySelector('details.doc-section');
            const s = d.querySelectorAll('details.page-section')[1]; s.open = true;
            setTimeout(() => { s.querySelector(':scope > details.page-text-details').open = true; }, 300); }""")
        pg.wait_for_timeout(3000)
        check("Langsam, Klappe waehrend des Vorladens geoeffnet: „Seitentext geladen.“ angesagt", "Seitentext geladen." in ansagen(pg), ansagen(pg))
        pg.unroute("**/seitentext")
        pg.route("**/seitentext", lambda rt: rt.fulfill(status=500, content_type="application/json", body='{"detail":"Testfehler"}'))
        pg.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="domcontentloaded")
        warte_ansicht(pg, "Alt-Texte")
        ansagen_leeren(pg)
        pg.evaluate("""() => { const d = document.querySelector('details.doc-section'); d.open = true;
            const s = d.querySelectorAll('details.page-section')[0]; s.open = true;
            s.querySelector(':scope > details.page-text-details').open = true; }""")
        pg.wait_for_timeout(1500)
        check("Fehler: Satz angesagt", any("konnte nicht geladen werden" in a for a in ansagen(pg)), ansagen(pg))
        pg.unroute("**/seitentext")
        pg.evaluate("""() => { const t = document.querySelector('details.page-section details.page-text-details'); t.open = false; }""")
        pg.wait_for_timeout(200)
        pg.evaluate("""() => { const t = document.querySelector('details.page-section details.page-text-details'); t.open = true; }""")
        pg.wait_for_timeout(1500)
        check("Fehler: erneutes Aufklappen laedt den Text", "Beispielhausen" in pg.locator("details.page-section .page-text-content").first.inner_text())

        # --- I. Uebersetzung oeffnen: eine Ansage ------------------------------------------------------------------
        print("== I. Uebersetzung oeffnen (Befund 6)", flush=True)
        pg.goto(B + f"/app?projekt={UEB}&ansicht=dokument", wait_until="domcontentloaded")
        warte_ansicht(pg, "Dokument")
        pg.wait_for_timeout(800)
        ansagen_leeren(pg)
        pg.click(".ansicht-knoepfe a[data-ansicht=uebersetzung]")
        warte_ansicht(pg, "Übersetzung")
        pg.wait_for_timeout(1500)
        an = ansagen(pg)
        check("Uebersetzung: genau eine Ansage („Ansicht Übersetzung geöffnet.“)", an == ["Ansicht Übersetzung geöffnet."], an)

        # --- C. Gastansicht ----------------------------------------------------------------------------------------
        print("== C. Gastansicht (Befund 10)", flush=True)
        r = ctx.request.post(B + f"/api/projects/{pid}/share", data={"guest_email": "gast@example.invalid", "notify": False, "role": "kunde"})
        token = r.json().get("token") if r.ok else None
        check("Freigabe angelegt (ohne Mail)", bool(token), r.status)
        if token:
            gctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
            g = gctx.new_page()
            g.on("pageerror", lambda e: fehler_js.append("Gast: " + str(e)))
            rc = gctx.request.post(f"{B}/api/freigabe/{token}/confirm", data={"email": "gast@example.invalid"})
            check("Gast: E-Mail bestaetigt", rc.ok, rc.status)
            if B.startswith("http://"):
                for h in rc.headers_array:
                    if h["name"].lower() == "set-cookie" and h["value"].startswith("guest_token="):
                        gctx.add_cookies([{"name": "guest_token", "value": h["value"].split(";")[0].split("=", 1)[1], "url": B}])
            g.goto(f"{B}/freigabe/{token}", wait_until="domcontentloaded")
            g.wait_for_selector("details.doc-section", timeout=20000)
            check("Gast: Seitentitel nennt das Projekt", "Projekt: Klicktest Ansichtswechsel" in g.title(), g.title())
            gs = gctx.request.get(f"{B}/api/freigabe/{token}")
            if gs.ok:
                check("Gast-Projektantwort ohne KI-Kontext/Seitentext", not any(schwer & set(b) for b in gs.json().get("images", [])))
            g.evaluate("() => { const d = document.querySelector('details.doc-section'); d.open = true; d.querySelector('details.page-section').open = true; }")
            gst = g.locator("details.page-section").first.locator(":scope > details.page-text-details")
            gst.locator(":scope > summary").click()
            g.wait_for_function("(el) => { const z = el.querySelector('.page-text-content'); return z && !z.hasAttribute('aria-busy') && z.textContent.length > 20; }",
                                arg=gst.element_handle(), timeout=10000)
            check("Gast: Seitentext beim Aufklappen geladen", "Beispielhausen" in gst.locator(".page-text-content").inner_text())
            gctx.close()
        check("Keine Skriptfehler", not [x for x in fehler_js if "401" not in x], fehler_js[:3])
    finally:
        if not BEHALTEN:
            r = ctx.request.delete(B + f"/api/projects/{pid}")
            print("Projekt", pid, "geloescht" if r.ok else f"NICHT geloescht ({r.status})", flush=True)
        br.close()
print(f"\nErgebnis: {ok} OK, {fehler} FEHLT")
sys.exit(1 if fehler else 0)
