#!/usr/bin/env python3
"""Messreihe Ansichtswechsel (05.10.2026, docs/ANSICHTEN_LEISTUNG.md) — echter Browser (Playwright/Chromium).

Misst in einem Projekt mit 7 klar fiktiven PDF (269 Bilder, rund 41 Mio. Zeichen KI-Kontext; Generator
tests/fixtures/make_messpdfs.py) und prueft die Zielwerte:
  - Ansichtswechsel (Klick bis die H1 der neuen Ansicht den Fokus hat), Median je Ansicht:
      lokal <= 300 ms, 50 Mbit/s (20 ms) <= 500 ms, 16 Mbit/s (30 ms) <= 1000 ms
  - Dokument, Tagging und Pruefung laden die Bildliste nicht (kein GET /api/projects/{id})
  - Projektantwort (Bildliste) <= 400 KB
  - erstes Oeffnen der Alt-Texte: keine Seitenansicht-Anfrage, bevor eine Seite aufgeklappt wird
  - Aufklappen bis Inhalt da: Seite (Bildvorschauen geladen), Seitentext, Seitenansicht — lokal und 50 Mbit/s <= 300 ms;
    dazu der Weg fuer Screenreader: Fokus bleibt auf dem Schalter, Platzhalter mit aria-busy, danach Text.

Aufruf (Playwright: /home/claude/.venv-pw/bin/python):
  mess_ansichtswechsel.py alles [profil ...]     legt das Projekt an, misst, loescht es wieder (Standard)
  mess_ansichtswechsel.py anlegen                 nur anlegen, gibt die Projekt-ID aus
  mess_ansichtswechsel.py messen <id> [profil ...]
  mess_ansichtswechsel.py loeschen <id>
Profile: lokal, 50mbit, 16mbit (Standard: alle drei). Ergebnis als JSON nach $MESS_AUSGABE (Standard /tmp).
Braucht INKLUDOCS_E2E_URL (Standard Staging), INKLUDOCS_E2E_MAIL, INKLUDOCS_E2E_PW. Funktioniert auch gegen einen
Wegwerf-Container ueber http (Anmeldung ueber die API, Cookie von Hand gesetzt).
Exit-Code 1, wenn ein Zielwert verfehlt wird."""
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de").rstrip("/")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
AUSGABE = os.environ.get("MESS_AUSGABE", "/tmp")
HIER = os.path.dirname(os.path.abspath(__file__))
FOLGE = ["tagging", "alttexte", "abschluss", "dokument", "alttexte", "abschluss", "tagging", "dokument"]
NAMEN = {"dokument": "Dokument", "tagging": "Tagging", "alttexte": "Alt-Texte", "abschluss": "Barrierefreiheitsprüfung"}
PROFILE = {
    "lokal": None,
    "50mbit": {"offline": False, "latency": 20, "downloadThroughput": 50e6 / 8, "uploadThroughput": 10e6 / 8},
    "16mbit": {"offline": False, "latency": 30, "downloadThroughput": 16e6 / 8, "uploadThroughput": 2e6 / 8},
}
ZIEL_WECHSEL = {"lokal": 300, "50mbit": 500, "16mbit": 1000}
ZIEL_AUFKLAPPEN = {"lokal": 300, "50mbit": 300}          # 16 Mbit/s: nur berichtet
ZIEL_PROJEKTANTWORT = 400 * 1024
INIT = """
window.__long = [];
try { new PerformanceObserver(l => l.getEntries().forEach(e => window.__long.push(Math.round(e.duration))))
      .observe({ type: 'longtask', buffered: true }); } catch (e) {}
"""
befunde = []


def ziel(name, ok, info=""):
    print(("  OK    " if ok else "  FEHLT ") + name + (f" -- {info}" if info else ""), flush=True)
    if not ok:
        befunde.append(f"{name}: {info}")


def anmelden(ctx, pg):
    """Anmeldung ueber die API; ueber http (Wegwerf-Container) das Secure-Cookie von Hand setzen."""
    r = ctx.request.post(B + "/api/login", data={"email": MAIL, "password": PW})
    if not r.ok:
        sys.exit(f"Anmeldung fehlgeschlagen: {r.status}")
    tok = None
    for h in r.headers_array:
        if h["name"].lower() == "set-cookie" and h["value"].startswith("token="):
            tok = h["value"].split(";")[0].split("=", 1)[1]
    if B.startswith("http://") and tok:
        ctx.add_cookies([{"name": "token", "value": tok, "url": B}])


def pdfs_erzeugen() -> list:
    ziel_ordner = os.path.join(AUSGABE, "messpdfs")
    vorhanden = sorted(glob.glob(os.path.join(ziel_ordner, "fiktiv_*.pdf")))
    if len(vorhanden) == 7:
        return vorhanden
    subprocess.run([sys.executable, os.path.join(HIER, "..", "fixtures", "make_messpdfs.py"), ziel_ordner], check=True)
    return sorted(glob.glob(os.path.join(ziel_ordner, "fiktiv_*.pdf")))


def anlegen(ctx) -> int:
    r = ctx.request.post(B + "/api/projects", data={"name": "Messprojekt Ansichtswechsel (fiktiv) " + time.strftime("%d.%m. %H:%M"), "tool": "pdf"})
    pid = r.json().get("id") or r.json().get("project_id")
    print("Messprojekt", pid, flush=True)
    for f in pdfs_erzeugen():
        t0 = time.time()
        with open(f, "rb") as fh:
            buf = fh.read()
        r = ctx.request.post(B + "/api/upload", multipart={
            "file": {"name": os.path.basename(f), "mimeType": "application/pdf", "buffer": buf}, "project_id": str(pid)},
            timeout=300000)
        if not r.ok:
            sys.exit(f"Upload {os.path.basename(f)} fehlgeschlagen: {r.status} {r.text()[:200]}")
        for _ in range(400):
            time.sleep(1.5)
            if ctx.request.get(B + f"/api/projects/{pid}/status").json().get("status") != "extracting":
                break
        print(f"  {os.path.basename(f)} gelesen ({time.time() - t0:.1f} s)", flush=True)
    return pid


def loeschen(ctx, pid):
    r = ctx.request.delete(B + f"/api/projects/{pid}")
    print("Messprojekt", pid, "geloescht" if r.ok else f"NICHT geloescht ({r.status})", flush=True)


def kurz(url):
    u = urlparse(url)
    return u.path + (("?" + u.query) if u.query else "")


def messen(ctx, pid, profil) -> dict:
    pg = ctx.new_page()
    fehler_js = []
    pg.on("pageerror", lambda e: fehler_js.append(str(e)))
    ctx.request.post(B + f"/api/projects/{pid}/ansicht", data={"ansicht": "dokument"})
    if PROFILE[profil]:
        cdp = ctx.new_cdp_session(pg)
        cdp.send("Network.enable")
        cdp.send("Network.emulateNetworkConditions", PROFILE[profil])
    reqs, offen, letzte = [], set(), [time.time()]

    def an(req):
        offen.add(id(req)); letzte[0] = time.time()

    def ab(req):
        offen.discard(id(req)); letzte[0] = time.time()

    def fertig(req):
        try:
            s = req.sizes()
            t = req.timing
            reqs.append({"m": req.method, "url": kurz(req.url), "bytes": s.get("responseBodySize", 0),
                         "start": t["startTime"], "ttfb": round(t["responseStart"], 1), "ende": round(t["responseEnd"], 1)})
        except Exception as e:  # noqa: BLE001
            reqs.append({"url": kurz(req.url), "fehler": str(e)})
    pg.on("request", an)
    pg.on("requestfinished", ab)
    pg.on("requestfailed", ab)
    pg.on("requestfinished", fertig)

    def netzruhe(ruhe=0.6, maxs=240):
        ende = time.time() + maxs
        while time.time() < ende:
            pg.wait_for_timeout(100)
            if not offen and time.time() - letzte[0] >= ruhe:
                return
    erg = {"projekt": pid, "profil": profil, "wechsel": [], "aufklappen": {}}
    t0 = time.time()
    pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="domcontentloaded")
    pg.wait_for_function("() => { const h = document.getElementById('projectName'); return h && h.textContent.startsWith('Dokument'); }", timeout=180000)
    erg["erstaufruf_ms"] = round((time.time() - t0) * 1000)
    netzruhe()
    erste_alttexte = True
    for ziel_ansicht in FOLGE:
        pg.wait_for_timeout(300)
        reqs.clear()
        pg.evaluate("() => { window.__long = []; }")
        t0 = time.time() * 1000
        pg.click(f".ansicht-knoepfe a[data-ansicht={ziel_ansicht}]")
        pg.wait_for_function(
            "(n) => { const h = document.getElementById('projectName'); return h && document.activeElement === h && h.textContent.startsWith(n); }",
            arg=NAMEN[ziel_ansicht], timeout=240000, polling=25)
        t1 = time.time() * 1000
        netzruhe()
        rq = [dict(r, ab_klick=round(r["start"] - t0)) for r in reqs if "start" in r]
        w = {"ziel": ziel_ansicht, "bedienbar_ms": round(t1 - t0), "requests": rq, "bytes": sum(r.get("bytes") or 0 for r in rq),
             "longtasks_ms": sum(pg.evaluate("() => window.__long"))}
        liste = [r for r in rq if r["m"] == "GET" and r["url"] == f"/api/projects/{pid}"]
        if ziel_ansicht == "alttexte":
            w["projektantwort_bytes"] = max((r.get("bytes") or 0) for r in liste) if liste else None
            seitenbilder = [r for r in rq if "/page-view" in r["url"]]
            w["seitenansicht_anfragen"] = len(seitenbilder)
            if erste_alttexte:
                erg["seitenansicht_beim_ersten_oeffnen"] = len(seitenbilder)
                erste_alttexte = False
        else:
            w["bildliste_geladen"] = bool(liste)
        erg["wechsel"].append(w)
        print(f"  [{profil}] -> {ziel_ansicht:10s} {w['bedienbar_ms']:5d} ms | {len(rq):3d} Anfragen {w['bytes'] / 1024:8.1f} KB | Longtasks {w['longtasks_ms']} ms", flush=True)
    erg["median_ms"] = {z: statistics.median([w["bedienbar_ms"] for w in erg["wechsel"] if w["ziel"] == z]) for z in NAMEN}
    erg["aufklappen"] = aufklappen(pg, pid)
    erg["js_fehler"] = fehler_js
    pg.close()
    return erg


AUFKLAPP_JS = """
async ([sel, art]) => {
  const det = document.querySelector(sel);
  if (!det) return { fehler: 'nicht gefunden: ' + sel };
  const sum = det.querySelector(':scope > summary');
  sum.scrollIntoView({ block: 'start' });
  sum.focus();
  await new Promise(r => requestAnimationFrame(r));
  const t0 = performance.now();
  sum.click();
  const fertig = () => {
    if (!det.open) return false;
    if (art === 'seite') {
      // sichtbare Bildvorschauen (lazy: unterhalb des Fensters laedt der Browser erst beim Scrollen)
      const imgs = [...det.querySelectorAll('img.image-preview')].filter(i => { const r = i.getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0; });
      return imgs.length > 0 && imgs.every(i => i.complete && i.naturalWidth > 0);
    }
    if (art === 'text') {
      const z = det.querySelector('.page-text-content');
      return z && !z.hasAttribute('aria-busy') && z.textContent.trim().length > 0;
    }
    if (art === 'ansicht') {
      const i = det.querySelector('img.page-view-image');
      return i && i.complete && i.naturalWidth > 0;
    }
    return true;
  };
  let platzhalter = null;
  if (art === 'text') { const z = det.querySelector('.page-text-content'); platzhalter = z ? { busy: z.getAttribute('aria-busy'), text: z.textContent.trim() } : null; }
  const ende = t0 + 20000;
  while (!fertig() && performance.now() < ende) await new Promise(r => requestAnimationFrame(r));
  return { ms: Math.round(performance.now() - t0), fertig: fertig(), fokus_auf_schalter: document.activeElement === sum, platzhalter };
}
"""


def aufklappen(pg, pid) -> dict:
    """In der Ansicht Alt-Texte: Dokument 5 (115 Bilder) aufklappen, dann drei Seiten:
    a) Seite -> Bildvorschauen geladen; sofort danach Seitentext (schlimmster Fall, Vorladen laeuft noch);
    b) Seite, 1 s warten, Seitentext (Normalfall: vorgeladen); c) Seitenansicht."""
    if not pg.locator("#projectName").inner_text().startswith("Alt-Texte"):
        pg.click(".ansicht-knoepfe a[data-ansicht=alttexte]")
        pg.wait_for_function("() => { const h = document.getElementById('projectName'); return h && h.textContent.startsWith('Alt-Texte'); }", timeout=120000)
    docs = pg.evaluate("() => [...document.querySelectorAll('details.doc-section')].map(d => [d.dataset.doc, d.querySelectorAll('details.page-section').length])")
    doc = max(docs, key=lambda d: d[1])[0]
    pg.evaluate(f"() => {{ const d = document.querySelector('details.doc-section[data-doc=\"{doc}\"]'); if (!d.open) d.querySelector(':scope > summary').click(); }}")
    pg.wait_for_timeout(300)
    seiten = pg.evaluate(f"() => [...document.querySelectorAll('details.doc-section[data-doc=\"{doc}\"] details.page-section')].map(d => d.dataset.page)")
    erg = {}
    s1, s2, s3 = seiten[2], seiten[5], seiten[8]
    erg["seite"] = pg.evaluate(AUFKLAPP_JS, [f'details.page-section[data-page="{s1}"]', "seite"])
    erg["seitentext_sofort"] = pg.evaluate(AUFKLAPP_JS, [f'details.page-section[data-page="{s1}"] > details.page-text-details', "text"])
    pg.evaluate(AUFKLAPP_JS, [f'details.page-section[data-page="{s2}"]', "keins"])
    pg.wait_for_timeout(1000)
    erg["seitentext_vorgeladen"] = pg.evaluate(AUFKLAPP_JS, [f'details.page-section[data-page="{s2}"] > details.page-text-details', "text"])
    pg.evaluate(AUFKLAPP_JS, [f'details.page-section[data-page="{s3}"]', "keins"])
    erg["seitenansicht"] = pg.evaluate(AUFKLAPP_JS, [f'details.page-section[data-page="{s3}"] > details.page-view-details', "ansicht"])
    # Ohne Vorlauf: Seite und Seitentext-Klappe im selben Augenblick oeffnen (Platzhalter-Weg, schlimmster Fall)
    s4 = seiten[11] if len(seiten) > 11 else seiten[-1]
    erg["seitentext_ohne_vorlauf"] = pg.evaluate("""async (seite) => {
        const s = document.querySelector('details.page-section[data-page="' + seite + '"]');
        const d = s && s.querySelector(':scope > details.page-text-details');
        if (!d) return { fehler: 'keine Seitentext-Klappe' };
        const z = d.querySelector('.page-text-content');
        const t0 = performance.now();
        s.open = true; d.open = true;
        const platzhalter = { busy: z.getAttribute('aria-busy'), text: z.textContent.trim().slice(0, 40) };
        while (z.hasAttribute('aria-busy') && performance.now() - t0 < 20000) await new Promise(r => setTimeout(r, 5));
        return { ms: Math.round(performance.now() - t0), fertig: !z.hasAttribute('aria-busy') && z.textContent.trim().length > 0, platzhalter };
    }""", s4)
    erg["region"] = pg.evaluate(f"""() => {{ const z = document.querySelector('details.page-section[data-page="{s2}"] .page-text-content');
        return {{ rolle: z.getAttribute('role'), name: z.getAttribute('aria-label'), busy: z.getAttribute('aria-busy'), zeichen: z.textContent.length }}; }}""")
    for k, v in erg.items():
        print(f"  Aufklappen {k}: {v}", flush=True)
    return erg


def pruefen(erg):
    p = erg["profil"]
    print(f"== Zielwerte {p}", flush=True)
    for z, ms in erg["median_ms"].items():
        ziel(f"{p}: Wechsel zu {NAMEN[z]} Median {ms:.0f} ms <= {ZIEL_WECHSEL[p]} ms", ms <= ZIEL_WECHSEL[p], f"{ms:.0f} ms")
    ohne_liste = [w for w in erg["wechsel"] if w["ziel"] != "alttexte"]
    ziel(f"{p}: Dokument/Tagging/Pruefung laden keine Bildliste", not any(w.get("bildliste_geladen") for w in ohne_liste),
         [w["ziel"] for w in ohne_liste if w.get("bildliste_geladen")])
    gr = [w.get("projektantwort_bytes") for w in erg["wechsel"] if w["ziel"] == "alttexte" and w.get("projektantwort_bytes")]
    if p == "lokal":
        ziel(f"{p}: Projektantwort <= 400 KB", bool(gr) and max(gr) <= ZIEL_PROJEKTANTWORT, f"{max(gr) / 1024:.0f} KB" if gr else "nicht gemessen")
        ziel(f"{p}: erstes Oeffnen der Alt-Texte ohne Seitenansicht-Anfragen", erg.get("seitenansicht_beim_ersten_oeffnen") == 0,
             erg.get("seitenansicht_beim_ersten_oeffnen"))
    a = erg["aufklappen"]
    for k in ("seite", "seitentext_sofort", "seitentext_vorgeladen", "seitentext_ohne_vorlauf", "seitenansicht"):
        v = a.get(k) or {}
        if p in ZIEL_AUFKLAPPEN:
            ziel(f"{p}: Aufklappen {k} bis Inhalt da <= {ZIEL_AUFKLAPPEN[p]} ms", v.get("fertig") and v.get("ms", 99999) <= ZIEL_AUFKLAPPEN[p], v)
        else:
            print(f"  INFO  {p}: Aufklappen {k}: {v.get('ms')} ms, fertig={v.get('fertig')}", flush=True)
    st = a.get("seitentext_sofort") or {}
    ziel(f"{p}: Seitentext — Fokus bleibt auf dem Schalter", st.get("fokus_auf_schalter") is True, st)
    ov = (a.get("seitentext_ohne_vorlauf") or {}).get("platzhalter") or {}
    ziel(f"{p}: Seitentext ohne Vorlauf — Platzhalter mit aria-busy", ov.get("busy") == "true" and "wird geladen" in (ov.get("text") or ""), ov)
    reg = a.get("region") or {}
    ziel(f"{p}: Seitentext-Bereich ist eine benannte Region ohne aria-busy, mit Text", reg.get("rolle") == "region" and reg.get("name") and not reg.get("busy") and reg.get("zeichen", 0) > 0, reg)
    ziel(f"{p}: keine Skriptfehler", not erg["js_fehler"], erg["js_fehler"][:3])


def main():
    if not MAIL or not PW:
        sys.exit("Zugangsdaten fehlen: INKLUDOCS_E2E_MAIL / INKLUDOCS_E2E_PW")
    befehl = sys.argv[1] if len(sys.argv) > 1 else "alles"
    with sync_playwright() as p:
        br = p.chromium.launch()
        ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
        ctx.add_init_script(INIT)
        anmelden(ctx, None)
        if befehl == "anlegen":
            print(anlegen(ctx))
            return
        if befehl == "loeschen":
            loeschen(ctx, int(sys.argv[2]))
            return
        if befehl == "messen":
            pid, profile = int(sys.argv[2]), (sys.argv[3:] or list(PROFILE))
        else:
            pid, profile = anlegen(ctx), (sys.argv[2:] or list(PROFILE))
        try:
            alle = []
            for prof in profile:
                print(f"== Messung {prof}", flush=True)
                # je Profil frischer Kontext (leerer Cache), damit Profile sich nicht gegenseitig helfen
                c2 = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
                c2.add_init_script(INIT)
                anmelden(c2, None)
                erg = messen(c2, pid, prof)
                c2.close()
                alle.append(erg)
                pruefen(erg)
                with open(os.path.join(AUSGABE, f"mess_ansichtswechsel_{prof}.json"), "w") as fh:
                    json.dump(erg, fh, ensure_ascii=False, indent=1)
        finally:
            if befehl == "alles":
                loeschen(ctx, pid)
        br.close()
    print("\nZusammenfassung:")
    for e in alle:
        print(f"  {e['profil']}: Median {', '.join(f'{NAMEN[k]} {v:.0f} ms' for k, v in e['median_ms'].items())}")
    print(f"Ergebnis: {len(befunde)} Zielwert(e) verfehlt" + ("" if not befunde else ": " + " | ".join(befunde)))
    sys.exit(1 if befunde else 0)


if __name__ == "__main__":
    main()
