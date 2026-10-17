#!/usr/bin/env python3
"""Klickprobe Staging: das Projekt selbst im Arbeitsfenster, Kamerafahrt, Mitsehen und Mithoeren (16.10.2026), seit 17.10.2026 ueber
die feste Schnittstelle window.InkluBuehne: dazu Station ohne Stelle (Ansicht), ein ANDERES eigenes Projekt, Meine Ablage und der
Weg von der Startseite. Ohne KI: zwei fiktive PDFs, Agentenansicht, Kamerafahrt per inkluagentFensterFolgen. Foto /tmp/buehne.png.
Projekte danach geloescht, Ansicht des Kontos wie vorher. Aufruf (vor jedem Einspielen auf Staging, Playwright): python3 tests/e2e/live_buehne_probe.py <mail> <passwort>"""
import base64
import io
import os
import sys
import time

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_BASIS", "https://staging.inkludocs.inklutec.de")   # nie gegen Production
ok = fehler = 0


def check(n, c, info=""):
    global ok, fehler
    ok, fehler = (ok + 1, fehler) if c else (ok, fehler + 1)
    print(("  OK   " if c else "  FEHLT ") + n + ("" if c else " -- " + str(info)[:500]))


def bild(farbe):
    b = Image.new("RGB", (480, 300), (255, 255, 255))
    ImageDraw.Draw(b).ellipse((120, 50, 360, 250), fill=farbe)
    puffer = io.BytesIO()
    b.save(puffer, "PNG")
    return base64.b64encode(puffer.getvalue()).decode()


HTML = "<!doctype html><html lang='de'><head><meta charset='utf-8'><title>Probe (fiktiv)</title></head><body style='font-family:Arial;margin:40px'>" \
       "<h1>Probeheft (fiktiv)</h1>" + "".join(
           f"<p>Abschnitt {i} mit etwas Text, damit die Seite lang wird.</p><img src='data:image/png;base64,{bild(c)}' width='320' height='200'>"
           "<p style='margin-bottom:300px'>Weiterer Text (fiktiv).</p>" for i, c in enumerate([(200, 60, 60), (60, 160, 60), (60, 60, 200)], 1)) + "</body></html>"

with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(locale="de-DE", viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    fehlerliste = []
    pg.on("pageerror", lambda e: fehlerliste.append("seite: " + str(e)))
    pg.on("console", lambda m: fehlerliste.append("konsole: " + m.text) if m.type == "error" and "favicon" not in m.text else None)
    pg.set_content(HTML)
    pdf = pg.pdf(format="A4")
    check("Anmeldung", pg.request.post(B + "/api/login", data={"email": sys.argv[1], "password": sys.argv[2]}).ok)
    vorher = pg.request.get(B + "/api/me").json()["user"]["oberflaeche"]
    r = pg.request.post(B + "/api/upload", multipart={"file": {"name": "probeheft-fiktiv.pdf", "mimeType": "application/pdf", "buffer": pdf}})
    pid = (r.json().get("project_id") or r.json().get("id")) if r.ok else None
    check("Fiktive PDF hochgeladen", bool(pid), r.text()[:200])
    pg.set_content(HTML.replace("Probeheft (fiktiv)", "Zweitheft (fiktiv)"))
    r2 = pg.request.post(B + "/api/upload", multipart={"file": {"name": "zweitheft-fiktiv.pdf", "mimeType": "application/pdf", "buffer": pg.pdf(format="A4")}})
    pid2 = (r2.json().get("project_id") or r2.json().get("id")) if r2.ok else None
    check("Zweite fiktive PDF hochgeladen", bool(pid2), r2.text()[:200])
    try:
        for _i in range(60):
            st = pg.request.get(B + f"/api/projects/{pid}/status").json()
            if st.get("status") not in ("extracting", "processing", None):
                break
            time.sleep(3)
        daten = pg.request.get(B + f"/api/projects/{pid}").json()
        bilder = [x["id"] for x in (daten.get("images") or [])]
        check("Drei Bilder", len(bilder) == 3, daten.keys())
        for _i in range(60):
            if pg.request.get(B + f"/api/projects/{pid2}/status").json().get("status") not in ("extracting", "processing", None):
                break
            time.sleep(3)
        bilder2 = [x["id"] for x in (pg.request.get(B + f"/api/projects/{pid2}").json().get("images") or [])]
        live = pg.request.get(B + "/api/agent/live")
        check("Live-Stand aus dem Laufbuch: 200, Liste der Laeufe", live.ok and isinstance(live.json().get("laeufe"), list), live.text()[:200])
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": "agent"})
        pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
        pg.wait_for_timeout(4000)
        rahmen = pg.frame_locator("#arbeitsSeiteRahmen")
        s = pg.evaluate("""() => { const r = document.getElementById('arbeitsSeiteRahmen'); const d = r.contentDocument; const w = r.contentWindow;
            return { offen: !document.getElementById('arbeitsSeite').hidden, url: d.location.pathname + d.location.search,
                     eingebettet: w.EINGEBETTET, h1: (d.querySelector('#main h1') || {}).textContent || '',
                     agent: !!(d.getElementById('inkluagentBereich') || d.querySelector('.inkluagent-section') || d.querySelector('[data-agentenansicht]')),
                     leiste: !!d.getElementById('appSidebar'),
                     titel: document.getElementById('arbeitsSeiteTitel').textContent,
                     folgen: [!document.getElementById('arbeitsFolgen').hidden, document.getElementById('arbeitsFolgen').getAttribute('aria-pressed')],
                     zu: document.getElementById('arbeitsSeiteZu').hidden && getComputedStyle(document.getElementById('arbeitsSeiteZu')).display === 'none',
                     buehne: !!(w.InkluBuehne && w.InkluBuehne.version === 1 && typeof w.InkluBuehne.zeigen === 'function') }; }""")
        print("  Bühne: " + str(s))
        check("Projekt steht im Fenster (eingebettet, dieses Projekt)", s["offen"] and s["url"].startswith(f"/app?projekt={pid}") and s["eingebettet"] is True, s)
        check("Echte Oberfläche: Überschrift da, kein zweiter Agent, keine Seitenleiste", bool(s["h1"]) and not s["agent"] and not s["leiste"], s)
        check("Die Seite im Fenster bietet die Bühne an (InkluBuehne, Version 1)", s["buehne"], s)
        check("Titel = Überschrift der Projektseite, „Mitverfolgen“ an, kein „Seite schließen“", "Projekt: probeheft-fiktiv.pdf" in s["titel"] and s["folgen"] == [True, "true"] and s["zu"], s)
        y0 = pg.evaluate("() => window.scrollY")
        b3 = bilder[2]
        pg.evaluate(f"() => inkluagentFensterFolgen({{ art: 'bild', image_id: {b3} }})")
        try:
            pg.wait_for_function(f"""() => {{ const d = document.getElementById('arbeitsSeiteRahmen').contentDocument; const el = d && d.getElementById('img_heading_{b3}');
                const k = el && (el.closest('section.image-review, details, .card') || el); return !!(k && k.classList.contains('inkluagent-hin-ziel')); }}""", timeout=15000)
            da = True
        except Exception:  # noqa: BLE001
            da = False
        z = pg.evaluate("""() => { const r = document.getElementById('arbeitsSeiteRahmen');
            return { url: r.contentDocument.location.search, fy: r.contentWindow.scrollY, y: window.scrollY,
                     hoeren: !document.getElementById('arbeitsHoeren').hidden, text: document.getElementById('arbeitsHoerenText').textContent,
                     knopf: !!document.querySelector('#arbeitsHoeren .inkluagent-anhoeren') }; }""")
        print("  Fahrt: " + str(z))
        check("Kamerafahrt: Ansicht Alt-Texte geladen, Karte von Bild 3 markiert", da and "ansicht=alttexte" in z["url"], z)
        sicht = pg.evaluate(f"() => {{ const el = document.getElementById('arbeitsSeiteRahmen').contentDocument.getElementById('imgcard_{b3}'); return el ? el.getBoundingClientRect().height : -1; }}")
        check("Die Karte ist im Fenster wirklich sichtbar (Höhe > 0)", sicht > 50, sicht)
        pg.wait_for_timeout(3500)   # die Seite im Fenster fragt nach und zeichnet neu — Lage und Marke muessen das ueberstehen
        check("Nach dem Neuzeichnen: Titel zeigt die Ansicht Alt-Texte", pg.evaluate("() => document.getElementById('arbeitsSeiteTitel').textContent").startswith("Alt-Texte"))
        z2 = pg.evaluate("() => ({ fy: document.getElementById('arbeitsSeiteRahmen').contentWindow.scrollY, y: window.scrollY })")
        check("Nur das Fenster rollt, die Seite draußen nicht", z2["y"] == y0 and z2["fy"] > 0, (y0, z2))
        check("„So klingt diese Stelle“ mit Hinweis ohne Alt-Text und „Anhören“", z["hoeren"] and "noch keinen Alternativtext" in z["text"] and z["knopf"], z)
        pg.locator("#arbeitsfenster").screenshot(path="/tmp/buehne.png")
        # Fokus im Fenster: nichts wird weggezogen
        pg.evaluate("() => document.getElementById('arbeitsSeiteRahmen').focus()")
        fy = pg.evaluate("() => document.getElementById('arbeitsSeiteRahmen').contentWindow.scrollY")
        pg.evaluate(f"() => inkluagentFensterFolgen({{ art: 'bild', image_id: {bilder[0]} }})")
        pg.wait_for_timeout(1500)
        check("Mit Fokus im Fenster fährt es nicht weg", pg.evaluate("() => document.getElementById('arbeitsSeiteRahmen').contentWindow.scrollY") == fy)
        pg.evaluate("() => document.getElementById('inkluagentInput').focus()")
        # selbst im Fenster klicken: „Mitverfolgen“ geht aus
        box = pg.locator("#arbeitsSeiteRahmen").bounding_box()
        pg.mouse.click(box["x"] + box["width"] - 20, box["y"] + box["height"] / 2)
        pg.wait_for_timeout(300)
        pg.evaluate("() => document.getElementById('inkluagentInput').focus()")
        check("Selbst im Fenster geklickt: „Mitverfolgen“ aus", pg.evaluate("() => document.getElementById('arbeitsFolgen').getAttribute('aria-pressed')") == "false")
        fy = pg.evaluate("() => document.getElementById('arbeitsSeiteRahmen').contentWindow.scrollY")
        pg.evaluate(f"() => inkluagentFensterFolgen({{ art: 'bild', image_id: {bilder[0]} }})")
        pg.wait_for_timeout(1500)
        check("Aus: es fährt nicht", pg.evaluate("() => document.getElementById('arbeitsSeiteRahmen').contentWindow.scrollY") == fy)
        pg.click("#arbeitsFolgen")
        pg.wait_for_timeout(1800)
        z = pg.evaluate(f"""() => {{ const d = document.getElementById('arbeitsSeiteRahmen').contentDocument; const el = d.getElementById('img_heading_{bilder[0]}');
            const k = el && (el.closest('section.image-review, details, .card') || el);
            return {{ an: document.getElementById('arbeitsFolgen').getAttribute('aria-pressed'), markiert: !!(k && k.classList.contains('inkluagent-hin-ziel')) }}; }}""")
        check("Wieder an: fährt zur zuletzt gewünschten Stelle (Bild 1)", z["an"] == "true" and z["markiert"], z)
        # andere Seite im Fenster, dann zurück zum Projekt
        pg.evaluate("() => { const a = [...document.querySelectorAll('#appSidebar a')].find(x => x.getAttribute('href') === '/einstellungen'); a && a.click(); }")
        pg.wait_for_timeout(3500)
        t1 = pg.evaluate("() => [document.getElementById('arbeitsSeiteTitel').textContent, document.getElementById('arbeitsSeiteZu').hidden, document.getElementById('arbeitsFolgen').hidden]")
        check("Einstellungen im Fenster: „Seite: …“, „Seite schließen“ da, „Mitverfolgen“ weg", t1[0].startswith("Seite: ") and t1[1] is False and t1[2] is True, t1)
        pg.click("#arbeitsSeiteZu")
        pg.wait_for_timeout(3500)
        t2 = pg.evaluate("() => document.getElementById('arbeitsSeiteTitel').textContent")
        check("„Seite schließen“ bringt das Projekt zurück", "Projekt: probeheft-fiktiv.pdf" in t2, t2)
        pg.wait_for_timeout(1500)
        # Station ohne einzelne Stelle: die Ansicht selbst (Prüfung), oben an der Überschrift
        pg.evaluate(f"() => inkluagentFensterFolgen({{ art: 'ansicht', project_id: {pid}, ansicht: 'abschluss' }})")
        try:
            pg.wait_for_function("""() => { const d = document.getElementById('arbeitsSeiteRahmen').contentDocument; const h = d && d.getElementById('projectName');
                return !!(h && h.classList.contains('inkluagent-hin-ziel') && d.location.search.indexOf('ansicht=abschluss') >= 0); }""", timeout=15000)
            st = True
        except Exception:  # noqa: BLE001
            st = False
        check("Station ohne Stelle: Fenster wechselt in „Barrierefreiheitsprüfung“ und zeigt die Überschrift", st,
              pg.evaluate("() => document.getElementById('arbeitsSeiteRahmen').contentDocument.location.search"))
        # ein ANDERES eigenes Projekt: das Fenster folgt, „Seite schließen“ bringt das eigene zurück
        if bilder2:
            pg.evaluate(f"() => inkluagentFensterFolgen({{ art: 'bild', image_id: {bilder2[0]}, project_id: {pid2}, ansicht: 'alttexte' }})")
            try:
                pg.wait_for_function(f"""() => {{ const d = document.getElementById('arbeitsSeiteRahmen').contentDocument; const el = d && d.getElementById('img_heading_{bilder2[0]}');
                    const k = el && (el.closest('section.image-review, details, .card') || el); return !!(k && k.classList.contains('inkluagent-hin-ziel')); }}""", timeout=20000)
                anders = True
            except Exception:  # noqa: BLE001
                anders = False
            a = pg.evaluate("""() => ({ url: document.getElementById('arbeitsSeiteRahmen').contentDocument.location.search, titel: document.getElementById('arbeitsSeiteTitel').textContent,
                                       zu: document.getElementById('arbeitsSeiteZu').hidden, folgen: document.getElementById('arbeitsFolgen').hidden, haupt: location.search })""")
            print("  Anderes Projekt: " + str(a))
            check("Anderes eigenes Projekt: Fenster folgt dorthin, Karte markiert, Seite draußen bleibt",
                  anders and f"projekt={pid2}" in a["url"] and f"projekt={pid}" in a["haupt"], a)
            check("… mit „Seite schließen“ und „Mitverfolgen“, Titel = Überschrift des anderen Projekts",
                  a["zu"] is False and a["folgen"] is False and "zweitheft-fiktiv.pdf" in a["titel"], a)
            pg.click("#arbeitsSeiteZu")
            pg.wait_for_timeout(3500)
            t3 = pg.evaluate("() => [document.getElementById('arbeitsSeiteTitel').textContent, document.getElementById('arbeitsSeiteRahmen').contentDocument.location.search]")
            check("„Seite schließen“ bringt das eigene Projekt zurück", "probeheft-fiktiv.pdf" in t3[0] and f"projekt={pid}" in t3[1], t3)
        # Meine Ablage: ein vorhandener Eintrag des Kontos als Ergebnis
        ausg = (pg.request.get(B + "/api/ausgaben").json().get("ausgaben") or [])
        if ausg:
            aid = ausg[0]["id"]
            pg.wait_for_timeout(1000)
            pg.evaluate(f"() => inkluagentFensterFolgen({{ art: 'ausgabe', ausgabe_id: {aid} }})")
            try:
                pg.wait_for_function(f"""() => {{ const d = document.getElementById('arbeitsSeiteRahmen').contentDocument; const el = d && d.getElementById('ausgabe-{aid}');
                    return !!(el && el.classList.contains('inkluagent-hin-ziel') && d.location.pathname === '/ablage'); }}""", timeout=20000)
                abl = True
            except Exception:  # noqa: BLE001
                abl = False
            check("Ergebnis: Fenster fährt in „Meine Ablage“ zum Eintrag", abl,
                  pg.evaluate("() => [document.getElementById('arbeitsSeiteRahmen').contentDocument.location.pathname, document.getElementById('arbeitsSeiteTitel').textContent]"))
            pg.click("#arbeitsSeiteZu")
            pg.wait_for_timeout(4000)
            t4 = pg.evaluate("() => [document.getElementById('arbeitsSeiteTitel').textContent, document.getElementById('arbeitsSeiteRahmen').contentDocument.location.pathname + document.getElementById('arbeitsSeiteRahmen').contentDocument.location.search]")
            check("„Seite schließen“ in der Ablage bringt das eigene Projekt zurück (und bleibt dort)", "probeheft-fiktiv.pdf" in t4[0] and f"projekt={pid}" in t4[1], t4)
        else:
            print("  (keine Ablage-Eintraege im Konto: Ablage-Fall uebersprungen)")
        # Von der Startseite aus: das Fenster folgt dem Agenten in ein Projekt
        pg.goto(B + "/dashboard", wait_until="networkidle")
        pg.wait_for_timeout(3000)
        da_fenster = pg.evaluate("() => !!document.getElementById('arbeitsSeiteRahmen')")
        if da_fenster:
            pg.evaluate(f"() => inkluagentFensterFolgen({{ art: 'bild', image_id: {bilder[1]}, project_id: {pid}, ansicht: 'alttexte' }})")
            try:
                pg.wait_for_function(f"""() => {{ const d = document.getElementById('arbeitsSeiteRahmen').contentDocument; const el = d && d.getElementById('img_heading_{bilder[1]}');
                    const k = el && (el.closest('section.image-review, details, .card') || el); return !!(k && k.classList.contains('inkluagent-hin-ziel')); }}""", timeout=20000)
                start = True
            except Exception:  # noqa: BLE001
                start = False
            check("Startseite: Fenster folgt dem Agenten ins Projekt, Seite draußen bleibt die Startseite",
                  start and pg.evaluate("() => location.pathname") == "/dashboard",
                  pg.evaluate("() => [location.pathname, (document.getElementById('arbeitsSeiteRahmen').contentDocument || {location: {}}).location.href]"))
        else:
            check("Startseite hat ein Arbeitsfenster", False, "kein #arbeitsSeiteRahmen")
        check("Keine Fehler in Seite oder Fenster", not fehlerliste, fehlerliste)
    finally:
        for x in (pid, pid2):
            if x:
                pg.request.delete(B + f"/api/projects/{x}")
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": vorher})
        print("  Ansicht zurueck auf: " + pg.request.get(B + "/api/me").json()["user"]["oberflaeche"])
    br.close()
print(f"ERGEBNIS: {ok} OK, {fehler} FEHLT")
