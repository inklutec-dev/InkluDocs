#!/usr/bin/env python3
"""Rauchtest Staging, Live-Fenster (16.10.2026) mit dem ECHTEN Modell und einem echten Alt-Text-Lauf im Hintergrund:
fiktive PDF mit drei Bildern, Agentenansicht, „Erzeuge bitte die Alt-Texte für alle Bilder.“, Knopf der Bestätigungskarte; dann
muss das Arbeitsfenster live mitlaufen (Bild in Arbeit, Fortschritt, „Gerade fertig“ mit Alt-Text) und am Ende „fertig“ zeigen.
Fotos nach /tmp/live_*.png. Kostet wenige Cent KI. Projekt danach geloescht, Ansicht des Kontos wie vorher.
Aufruf (vor jedem Einspielen auf Staging, Playwright): python3 tests/e2e/live_alttexte_rauch.py <mail> <passwort>"""
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


def bild(farbe, form):
    b = Image.new("RGB", (480, 300), (255, 255, 255))
    d = ImageDraw.Draw(b)
    if form == "haus":
        d.rectangle((150, 160, 330, 250), fill=farbe)
        d.polygon([(240, 80), (120, 160), (360, 160)], fill=(160, 40, 40))
    elif form == "baum":
        d.rectangle((225, 180, 255, 260), fill=(110, 70, 40))
        d.ellipse((150, 60, 330, 200), fill=farbe)
    else:
        d.rectangle((60, 220, 120, 260), fill=farbe)
        d.rectangle((160, 160, 220, 260), fill=farbe)
        d.rectangle((260, 100, 320, 260), fill=farbe)
        d.rectangle((360, 60, 420, 260), fill=farbe)
    puffer = io.BytesIO()
    b.save(puffer, "PNG")
    return base64.b64encode(puffer.getvalue()).decode()


HTML = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Vereinsheft (fiktiv)</title></head>
<body style="font-family: Arial, sans-serif; margin: 40px;"><h1>Vereinsheft Musterstadt (fiktiv)</h1>
<p>Unser neues Vereinsheim ist fertig.</p><img src="data:image/png;base64,{bild((240, 220, 170), 'haus')}" width="320" height="200">
<p>Im Garten steht jetzt ein Apfelbaum.</p><img src="data:image/png;base64,{bild((60, 150, 70), 'baum')}" width="320" height="200">
<p>Die Mitgliederzahl ist in vier Jahren gewachsen.</p><img src="data:image/png;base64,{bild((40, 90, 170), 'balken')}" width="320" height="200">
</body></html>"""

with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(locale="de-DE", viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    js = []
    pg.on("pageerror", lambda e: js.append(str(e)))
    pg.set_content(HTML)
    pdf = pg.pdf(format="A4")
    check("Anmeldung", pg.request.post(B + "/api/login", data={"email": sys.argv[1], "password": sys.argv[2]}).ok)
    vorher = pg.request.get(B + "/api/me").json()["user"]["oberflaeche"]
    r = pg.request.post(B + "/api/upload", multipart={"file": {"name": "vereinsheft-fiktiv.pdf", "mimeType": "application/pdf", "buffer": pdf}})
    pid = (r.json().get("project_id") or r.json().get("id")) if r.ok else None
    check("Fiktive PDF mit drei Bildern hochgeladen", bool(pid), (r.status, r.text()[:200]))
    try:
        for _i in range(60):
            st = pg.request.get(B + f"/api/projects/{pid}/status").json()
            if st.get("status") not in ("extracting", "processing", None):
                break
            time.sleep(3)
        check("Drei Bilder erkannt", (st.get("total_images") or 0) >= 3, st)
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": "agent"})
        pg.goto(B + f"/app?projekt={pid}", wait_until="networkidle")
        pg.wait_for_timeout(2500)
        pg.locator("#arbeitsfenster").screenshot(path="/tmp/live_ruhe.png")
        check("Live-Block ohne Lauf verborgen", pg.evaluate("() => document.getElementById('arbeitsLive').hidden"))
        pg.evaluate("() => document.querySelectorAll('#inkluagentLog .inkluagent-message').forEach(m => m._alt = true)")
        pg.fill("#inkluagentInput", "Erzeuge bitte die Alt-Texte für alle Bilder.")
        pg.click("#inkluagentSendBtn")
        KARTE = "[...document.querySelectorAll('#inkluagentLog .inkluagent-bestaetigung button')].filter(b => !b.closest('.inkluagent-message')._alt && !b.disabled).pop()"
        try:
            pg.wait_for_function(f"() => !!({KARTE})", timeout=150000)
            da = True
        except Exception:  # noqa: BLE001
            da = False
        antwort = pg.evaluate("([...document.querySelectorAll('#inkluagentLog .inkluagent-message.assistant')].pop() || {}).textContent || ''")
        print("  Antwort: " + antwort[:300].replace("\n", " "))
        check("Bestätigungskarte für die Alt-Texte", da, antwort)
        if da:
            pg.evaluate(f"() => ({KARTE}).click()")
            t0 = time.time()
            try:
                pg.wait_for_function("() => !document.getElementById('arbeitsLive').hidden && document.getElementById('arbeitsLive').dataset.zustand === 'laeuft'", timeout=40000)
                live = True
            except Exception:  # noqa: BLE001
                live = False
            check("Live-Block erscheint, solange der Lauf arbeitet", live, round(time.time() - t0, 1))
            if live:
                pg.wait_for_timeout(1200)
                s = pg.evaluate("""() => { const b = document.getElementById('arbeitsLive');
                    return { titel: b.querySelector('h3').textContent, kopf: b.classList.contains('arbeitsfenster-live-kopf'),
                             bildVerborgen: b.querySelector('figure').hidden, stand: document.getElementById('arbeitsStand').textContent,
                             max: b.querySelector('progress').max }; }""")
                print("  Live: " + str(s))
                check("Kopfzeile nennt Projekt und Fortschritt; das Bild zeigt die echte Oberfläche, nicht der Kopf",
                      s["titel"].startswith("Im Hintergrund: Alt-Texte für „") and "von 3 fertig" in s["titel"] and s["kopf"] and s["bildVerborgen"], s)
                FAHRT = """() => { const r = document.getElementById('arbeitsSeiteRahmen'); const d = r.contentDocument; const w = r.contentWindow;
                    const z = window._arbeitsFolgeZiel; const l = window._arbeitsLive && window._arbeitsLive.lauf;
                    if (!z || !l || !l.stelle || z.image_id !== l.stelle.image_id) return null;
                    const el = d.getElementById('imgcard_' + z.image_id); if (!el) return null;
                    const rr = el.getBoundingClientRect();
                    return { url: d.location.search, id: z.image_id, h: Math.round(rr.height), oben: Math.round(rr.top), innen: w.innerHeight }; }"""
                try:
                    pg.wait_for_function(f"() => {{ const f = ({FAHRT})(); return !!(f && f.url.indexOf('ansicht=alttexte') >= 0 && f.h > 50 && f.oben < f.innen && f.oben + f.h > 0); }}", timeout=40000)
                    fahrt = pg.evaluate(FAHRT)
                except Exception:  # noqa: BLE001
                    fahrt = pg.evaluate(FAHRT)
                print("  Fahrt: " + str(fahrt))
                check("Kamerafahrt: das Fenster steht in „Alt-Texte“ bei der Karte des Bildes in Arbeit (sichtbar)",
                      bool(fahrt) and "ansicht=alttexte" in fahrt["url"] and fahrt["h"] > 50 and fahrt["oben"] < fahrt["innen"], fahrt)
                aussen = pg.evaluate("() => document.activeElement && (document.activeElement.id || document.activeElement.tagName)")
                check("Der Fokus blieb draußen, nicht im Fenster (Laden und Fahrt ziehen ihn nicht hinein)", aussen != "arbeitsSeiteRahmen", aussen)
                pg.locator("#arbeitsfenster").screenshot(path="/tmp/live_laeuft.png")
                try:
                    pg.wait_for_function("() => (document.getElementById('arbeitsHoerenText').textContent || '').startsWith('Bild, ')", timeout=180000)
                    hoeren = pg.evaluate("() => document.getElementById('arbeitsHoerenText').textContent")
                except Exception:  # noqa: BLE001
                    hoeren = pg.evaluate("() => document.getElementById('arbeitsHoerenText').textContent")
                print("  So klingt diese Stelle: " + hoeren[:200])
                check("„So klingt diese Stelle“: „Bild, <neuer Alt-Text>“", hoeren.startswith("Bild, ") and len(hoeren) > 30, hoeren)
                pg.locator("#arbeitsfenster").screenshot(path="/tmp/live_fertig1.png")
                try:
                    pg.wait_for_function("() => document.getElementById('arbeitsLive').dataset.zustand === 'fertig'", timeout=300000)
                    ende = True
                except Exception:  # noqa: BLE001
                    ende = False
                check("Am Ende „Im Hintergrund: Alt-Texte fertig“", ende and pg.evaluate("() => document.getElementById('arbeitsLiveTitel').textContent") == "Im Hintergrund: Alt-Texte fertig",
                      pg.evaluate("() => document.getElementById('arbeitsLiveTitel').textContent"))
                pg.wait_for_timeout(4000)
                karten = pg.evaluate("""() => { const d = document.getElementById('arbeitsSeiteRahmen').contentDocument;
                    return [...d.querySelectorAll('textarea.alt-text-field:not(.langtext-field)')].map(t => t.value.trim().length); }""")
                check("In der echten Oberfläche im Fenster stehen jetzt alle drei Alt-Texte", len(karten) == 3 and all(n > 10 for n in karten), karten)
                pg.locator("#arbeitsfenster").screenshot(path="/tmp/live_ende.png")
                stand = pg.evaluate("() => document.getElementById('arbeitsStand').textContent")
                check("Stand nach dem Knopf nicht mehr „Wartet auf dich“", not stand.startswith("Wartet auf dich"), stand)
                pg.click("#arbeitsVorlesen")
                pg.wait_for_timeout(800)
                ansage = pg.evaluate("() => [...document.querySelectorAll('[aria-live]')].map(e => e.textContent).join(' | ')")
                check("„Stand vorlesen“ nennt den gerade fertigen Alt-Text", "Gerade fertig: Bild" in ansage, ansage)
            alle = pg.request.get(B + f"/api/projects/{pid}/status").json()
            check("Alle drei Bilder haben jetzt Alt-Texte", alle.get("processed_images") == alle.get("total_images") == 3, alle)
        check("Keine JavaScript-Fehler", not js, js)
    finally:
        if pid:
            pg.request.delete(B + f"/api/projects/{pid}")
        pg.request.put(B + "/api/me/oberflaeche", data={"oberflaeche": vorher})
        print("  Ansicht zurueck auf: " + pg.request.get(B + "/api/me").json()["user"]["oberflaeche"])
    br.close()
print(f"ERGEBNIS: {ok} OK, {fehler} FEHLT")
