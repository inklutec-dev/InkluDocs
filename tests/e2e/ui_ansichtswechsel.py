#!/usr/bin/env python3
"""Klicktest Ansichtswechsel (05.10.2026, docs/ANSICHTEN_LEISTUNG.md):
  A. Tippen und SOFORT wechseln — keine Eingabe geht verloren (Alt-Text -> Dokument, Alt-Text -> Browser-Zurueck,
     Quickinfo -> Dokument); die feste Wartezeit von 900 ms ist weg (Wechsel ohne Eingabe schnell).
  B. Ansicht Alt-Texte: Seitenansicht und Seitentext erst beim Aufklappen, Platzhalter mit aria-busy, Fokus bleibt
     auf dem Schalter, Text danach im benannten Bereich; Projektantwort ohne KI-Kontext und Serverpfade.
  C. Gastansicht (Freigabe): Seitentext beim Aufklappen ueber den Gast-Abruf.
Legt sein Projekt selbst an (klar fiktive Inhalte) und loescht es wieder (ausser --behalten).
Aufruf: /home/claude/.venv-pw/bin/python ui_ansichtswechsel.py [<testformular.pdf>] [--behalten]
(Standard-Formular: tests/fixtures/testformular_inkludocs.pdf). Braucht INKLUDOCS_E2E_URL/MAIL/PW."""
import os
import sys
import time

from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de").rstrip("/")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
HIER = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FORM = args[0] if args else os.path.join(HIER, "..", "fixtures", "testformular_inkludocs.pdf")
BEHALTEN = "--behalten" in sys.argv
ok = fehler = 0


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


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
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

        # --- B. Alt-Texte: Projektantwort schlank, Seitenansicht/Seitentext beim Aufklappen ---------------------
        print("== B. Ansicht Alt-Texte", flush=True)
        voll = ctx.request.get(B + f"/api/projects/{pid}").json()
        bilder = voll.get("images") or []
        schwer = {"context_text", "page_text", "pipeline_steps", "validation_result", "image_path", "page_view_path"}
        check("Projektantwort: Bilder da, ohne KI-Kontext/Seitentext/Serverpfade", bilder and not any(schwer & set(b) for b in bilder),
              [sorted(schwer & set(b)) for b in bilder][:2])
        check("Projektantwort: Merker hat_seitenansicht/hat_seitentext", any(b.get("hat_seitentext") for b in bilder) and any(b.get("hat_seitenansicht") for b in bilder))
        kopf = ctx.request.get(B + f"/api/projects/{pid}/kopf").json()
        check("/kopf: ohne Bildliste, mit Zaehlern und hat_felder", "images" not in kopf and kopf.get("bilder_gesamt", 0) >= 1 and kopf["project"].get("hat_felder", 0) > 0, {k: kopf.get(k) for k in ("bilder_gesamt", "bilder_status")})
        seitenbilder = []
        pg.on("request", lambda rq: seitenbilder.append(rq.url) if "/page-view" in rq.url else None)
        pg.goto(B + f"/app?projekt={pid}&ansicht=alttexte", wait_until="domcontentloaded")
        warte_ansicht(pg, "Alt-Texte")
        pg.wait_for_timeout(1500)
        check("Erstes Oeffnen: keine Seitenansicht geladen (alles zugeklappt)", not seitenbilder, seitenbilder[:3])
        pg.locator("details.doc-section > summary").first.click()
        seite = pg.locator("details.page-section").first
        seite.locator(":scope > summary").click()
        pg.wait_for_timeout(200)
        st = seite.locator(":scope > details.page-text-details")
        check("Seitentext-Klappe vorhanden (mit Bild-Verweis)", st.count() == 1 and st.get_attribute("data-seitentext-bild"))
        st.locator(":scope > summary").focus()
        pg.keyboard.press("Enter")   # wie mit der Tastatur/VoiceOver
        pg.wait_for_function("(el) => { const z = el.querySelector('.page-text-content'); return z && !z.hasAttribute('aria-busy') && z.textContent.length > 20; }",
                             arg=st.element_handle(), timeout=10000)
        z = st.locator(".page-text-content")
        check("Seitentext da, im benannten Bereich (role=region, Name Seitentext)", "Beispielhausen" in z.inner_text() and z.get_attribute("role") == "region" and z.get_attribute("aria-label") == "Seitentext", z.inner_text()[:80])
        check("Fokus bleibt auf dem Schalter „Seitentext anzeigen“", pg.evaluate("document.activeElement && document.activeElement.tagName === 'SUMMARY' && document.activeElement.textContent.includes('Seitentext')"))
        # Platzhalter-Weg (ohne Vorlauf): neue Seite aufklappen und Klappe im selben Augenblick oeffnen
        weg = pg.evaluate("""async () => {
            const seiten = document.querySelectorAll('details.page-section');
            const s = seiten[seiten.length - 1];
            s.open = true;
            const d = s.querySelector(':scope > details.page-text-details');
            if (!d) return { fehlt: true };
            const z = d.querySelector('.page-text-content');
            d.open = true;
            const vorher = { busy: z.getAttribute('aria-busy'), text: z.textContent.trim() };
            const t0 = performance.now();
            while (z.hasAttribute('aria-busy') && performance.now() - t0 < 10000) await new Promise(r => setTimeout(r, 10));
            return { vorher, ms: Math.round(performance.now() - t0), nachher: z.textContent.trim().slice(0, 40) };
        }""")
        nachher = weg.get("nachher", "")
        check("Ohne Vorlauf: Platzhalter „Seitentext wird geladen …“ mit aria-busy, danach Text",
              weg.get("vorher", {}).get("busy") == "true" and "wird geladen" in weg.get("vorher", {}).get("text", "")
              and len(nachher) > 10 and "wird geladen" not in nachher, weg)
        sa = seite.locator(":scope > details.page-view-details")
        if sa.count():
            vorher = len(seitenbilder)
            sa.locator(":scope > summary").click()
            pg.wait_for_function("(el) => { const i = el.querySelector('img.page-view-image'); return i && i.complete && i.naturalWidth > 0; }", arg=sa.element_handle(), timeout=10000)
            check("Seitenansicht laedt beim Aufklappen (genau eine Anfrage)", len(seitenbilder) == vorher + 1, seitenbilder)

        # --- A. Tippen und sofort wechseln -------------------------------------------------------------------
        print("== A. Tippen und sofort wechseln", flush=True)
        feld = seite.locator("textarea.alt-text-field:not(.langtext-field)").first
        bild_id = int(feld.get_attribute("data-image-id"))
        text1 = "Fiktiver Testtext eins " + time.strftime("%H%M%S")
        feld.fill(text1)                                   # loest input aus — Speichern erst nach 800 ms faellig
        t0 = time.time()
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")   # SOFORT wechseln
        warte_ansicht(pg, "Dokument")
        dauer = (time.time() - t0) * 1000
        gespeichert = [b for b in ctx.request.get(B + f"/api/projects/{pid}").json()["images"] if b["id"] == bild_id][0]
        check("Alt-Text getippt, sofort zu „Dokument“: Text gespeichert", gespeichert.get("alt_text_edited") == text1, gespeichert.get("alt_text_edited"))
        check(f"Wechsel mit offener Eingabe ohne feste Wartezeit ({dauer:.0f} ms < 900 ms)", dauer < 900, dauer)
        # Browser-Zurueck nach dem Tippen
        pg.go_back()
        warte_ansicht(pg, "Alt-Texte")
        # Dokument und Seite offen (der Auf/Zu-Stand wird ueber das Neu-Zeichnen gemerkt — darum setzen statt klicken)
        pg.evaluate("() => { const d = document.querySelector('details.doc-section'); d.open = true; d.querySelector('details.page-section').open = true; }")
        feld = pg.locator(f"#alttext_{bild_id}")
        check("Nach Zurueck: Feld zeigt den gespeicherten Text", feld.input_value() == text1, feld.input_value())
        text2 = "Fiktiver Testtext zwei " + time.strftime("%H%M%S")
        feld.fill(text2)
        pg.go_forward()
        warte_ansicht(pg, "Dokument")
        pg.wait_for_timeout(300)
        gespeichert = [b for b in ctx.request.get(B + f"/api/projects/{pid}").json()["images"] if b["id"] == bild_id][0]
        check("Alt-Text getippt, sofort Browser-Vor: Text gespeichert", gespeichert.get("alt_text_edited") == text2, gespeichert.get("alt_text_edited"))
        # Wechsel ohne Eingabe: schnell
        t0 = time.time()
        pg.click(".ansicht-knoepfe a[data-ansicht=tagging]")
        warte_ansicht(pg, "Tagging")
        dauer = (time.time() - t0) * 1000
        check(f"Wechsel ohne Eingabe: {dauer:.0f} ms (Ziel unter 900 ms, ohne feste Wartezeit)", dauer < 900, dauer)
        # Quickinfo
        quick = pg.locator(".ansicht-knoepfe a[data-ansicht=quickinfos]")
        if quick.count():
            quick.click()
            warte_ansicht(pg, "Quickinfos")
            pg.wait_for_timeout(500)
            pg.evaluate("() => document.querySelectorAll('details').forEach(d => d.open = true)")
            qf = pg.locator("textarea.quickinfo-field").first
            feld_id = int(qf.get_attribute("data-feld-id"))
            text3 = "Fiktive Quickinfo " + time.strftime("%H%M%S")
            qf.fill(text3)
            pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
            warte_ansicht(pg, "Dokument")
            felder = ctx.request.get(B + f"/api/projects/{pid}/felder").json().get("felder") or []
            f = [x for x in felder if x["id"] == feld_id]
            check("Quickinfo getippt, sofort zu „Dokument“: gespeichert", f and f[0].get("quickinfo") == text3, f[0].get("quickinfo") if f else None)
        else:
            check("Quickinfo-Ansicht verfuegbar (Testformular hat Felder)", False, "Knopf fehlt")

        # --- C. Gastansicht -----------------------------------------------------------------------------------
        print("== C. Gastansicht", flush=True)
        r = ctx.request.post(B + f"/api/projects/{pid}/share", data={"guest_email": "gast@example.invalid", "notify": False, "role": "kunde"})
        token = r.json().get("token") if r.ok else None
        check("Freigabe angelegt (ohne Mail)", bool(token), r.status)
        if token:
            gctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
            g = gctx.new_page()
            g.on("pageerror", lambda e: fehler_js.append("Gast: " + str(e)))
            # E-Mail bestaetigen ueber denselben Abruf wie das Formular; ueber http (Wegwerf-Container) das Secure-Cookie von Hand
            rc = gctx.request.post(f"{B}/api/freigabe/{token}/confirm", data={"email": "gast@example.invalid"})
            check("Gast: E-Mail bestaetigt", rc.ok, rc.status)
            if B.startswith("http://"):
                for h in rc.headers_array:
                    if h["name"].lower() == "set-cookie" and h["value"].startswith("guest_token="):
                        gctx.add_cookies([{"name": "guest_token", "value": h["value"].split(";")[0].split("=", 1)[1], "url": B}])
            g.goto(f"{B}/freigabe/{token}", wait_until="domcontentloaded")
            g.wait_for_selector("details.doc-section", timeout=20000)
            gs = gctx.request.get(f"{B}/api/freigabe/{token}")
            if gs.ok:
                check("Gast-Projektantwort ohne KI-Kontext/Seitentext", not any(schwer & set(b) for b in gs.json().get("images", [])))
            g.locator("details.doc-section > summary").first.click()
            gseite = g.locator("details.page-section").first
            gseite.locator(":scope > summary").click()
            gst = gseite.locator(":scope > details.page-text-details")
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
