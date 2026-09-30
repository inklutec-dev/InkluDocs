#!/usr/bin/env python3
"""Klicktest Pruefung 3 Barrierefreiheit (30.09.2026): Chat und Upload mit Mitschnitt aller Live-Regionen.
  H1  Verlauf wird beim Neuzeichnen (Ansichtswechsel, Chat zu/auf, Neuladen) NICHT angesagt; Log ist aria-live="off"
  M1  „Genutzt: …“ mit Anzeigenamen vom Server, keine rohen Werkzeugnamen
  N1  eigene Nachricht und Antwort kommen nicht ueber eine Live-Region (die Antwort bekommt den Fokus)
  N6  Download-Knopf traegt den Dateinamen; Bestaetigungs-Karte (Ueberschrift, Servertext, Knopf „… bestätigen“, Status)
  M2  kaputter Upload: der Fehler wird EINMAL angesagt, kein „… wird hochgeladen“ danach
Der Verlauf wird im Container mit einer fiktiven Unterhaltung gefuellt (kein KI-Aufruf). Legt ein Projekt an und loescht es.
Aufruf: /home/claude/.venv-pw/bin/python ui_chat_barrierefrei.py <ordner mit antrag_pflege.pdf>"""
import json
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ["INKLUDOCS_E2E_MAIL"], os.environ["INKLUDOCS_E2E_PW"]
K = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/michael-0930"
AXE = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"
ok = fehler = 0

MITSCHNITT = """
window.__live = [];
function __istLive(el) {
  for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
    const al = n.getAttribute('aria-live');
    if (al === 'off') return null;
    if (al === 'polite' || al === 'assertive') return n;
    const r = n.getAttribute('role');
    if (r === 'status' || r === 'alert' || r === 'log') return n;
  }
  return null;
}
new MutationObserver(ms => { for (const m of ms) {
  const ziel = m.target.nodeType === 1 ? m.target : m.target.parentElement;
  const live = ziel ? __istLive(ziel) : null;
  if (!live) continue;
  let text = '';
  if (m.type === 'characterData') text = m.target.textContent;
  m.addedNodes.forEach(x => { text += ' ' + (x.textContent || ''); });
  text = text.trim();
  if (text) window.__live.push({ id: live.id || live.getAttribute('role') || live.tagName, text: text.slice(0, 300) });
}}).observe(document, { childList: true, subtree: true, characterData: true });
"""


def check(n, c, i=""):
    global ok, fehler
    if c:
        ok += 1
        print("  OK ", n)
    else:
        fehler += 1
        print("  FEHLT", n, "--", str(i)[:400])


def im_container(code):
    r = subprocess.run(["sudo", "-n", "docker", "exec", "-w", "/app", "inkludocs-staging", "python3", "-c", code],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip() + r.stderr.strip()[-300:]


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1280, "height": 900}, locale="de-DE")
    ctx.add_init_script(MITSCHNITT)
    pg = ctx.new_page()
    js = []
    pg.on("pageerror", lambda e: js.append(str(e)))
    pg.goto(B + "/login")
    pg.fill("#email", MAIL)
    pg.fill("#password", PW)
    pg.keyboard.press("Enter")
    pg.wait_for_timeout(2500)
    r = pg.request.post(B + "/api/projects", data={"name": "Chat barrierefrei 30.09. (Test)", "tool": "pdf"})
    pid = r.json().get("id") or r.json().get("project_id")
    try:
        with open(os.path.join(K, "antrag_pflege.pdf"), "rb") as f:
            pg.request.post(B + "/api/upload", multipart={"file": {"name": "antrag.pdf", "mimeType": "application/pdf", "buffer": f.read()},
                                                          "project_id": str(pid)})
        for _ in range(60):
            time.sleep(2)
            if (pg.request.get(B + f"/api/projects/{pid}").json().get("project") or {}).get("status") not in ("extracting", "processing"):
                break
        anhang = [{"art": "pdf", "dateiname": "inkludocs_antrag.pdf", "download_url": f"/api/projects/{pid}/export/pdfua/{'a' * 24}", "label": "pdf"},
                  {"art": "bestaetigung", "angebot_id": "f" * 32, "titel": "Bestätigung nötig",
                   "text": "Ablage-Eintrag löschen: „fiktiv.pdf“. Das lässt sich nicht rückgängig machen.", "knopf": "Ablage-Eintrag löschen bestätigen"}]
        print(im_container("from inkluagent import storage; "
                           f"storage.append_message({pid}, 'user', 'Wie ist der Stand? (fiktiver Verlauf)'); "
                           f"storage.append_message({pid}, 'assistant', 'Das Dokument ist schon getaggt. (fiktive Antwort)', "
                           f"werkzeuge=['dokument_stand', 'pruefdatei_erstellen', 'exportiere_fertige_pdf'], anhang=__import__('json').loads({json.dumps(anhang)!r}))"))
        pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.evaluate(f"localStorage.setItem('inkluagent.panel.{pid}', 'open')")
        pg.goto(B + f"/app?projekt={pid}&ansicht=dokument", wait_until="networkidle")
        pg.wait_for_timeout(2000)
        check("Chat offen mit Verlauf", pg.locator("#inkluagentLog .inkluagent-message").count() == 2)
        check("H1/N1: Verlauf nicht live (aria-live=off am Log)", pg.locator("#inkluagentLog").get_attribute("aria-live") == "off")

        def verlauf_angesagt():
            return [x for x in pg.evaluate("window.__live") if "fiktive" in x["text"] or "fiktiver" in x["text"]]
        check("Neuladen mit offenem Chat: Verlauf nicht angesagt", not verlauf_angesagt(), verlauf_angesagt())
        pg.evaluate("window.__live = []")
        pg.click(".ansicht-knoepfe a[data-ansicht=tagging]")
        pg.wait_for_timeout(2500)
        check("Ansichtswechsel: Verlauf nicht angesagt", not verlauf_angesagt(), verlauf_angesagt())
        pg.evaluate("window.__live = []")
        pg.click("#inkluagentToggle")
        pg.wait_for_timeout(400)
        pg.click("#inkluagentToggle")
        pg.wait_for_timeout(1500)
        check("Chat zu und auf: Verlauf nicht angesagt", not verlauf_angesagt(), verlauf_angesagt())
        # Gegenprobe: hört der Mitschnitt überhaupt mit? Ohne aria-live="off" (wie vor der Behebung) muss er den Verlauf fangen.
        pg.evaluate("window.__live = []; document.getElementById('inkluagentLog').removeAttribute('aria-live')")
        pg.evaluate("(() => { const l = document.getElementById('inkluagentLog'); const k = l.firstElementChild.cloneNode(true); l.appendChild(k); })()")
        pg.wait_for_timeout(300)
        check("Gegenprobe: ohne aria-live=off fängt der Mitschnitt den Verlauf", bool(verlauf_angesagt()), pg.evaluate("window.__live"))
        pg.evaluate("(() => { const l = document.getElementById('inkluagentLog'); l.lastElementChild.remove(); l.setAttribute('aria-live', 'off'); })()")
        pg.evaluate("window.__live = []")

        tools = pg.locator("#inkluagentLog .inkluagent-message-tools").first.inner_text()
        check("M1: „Genutzt: Dokumentstand, Prüfdatei erstellen, PDF herunterladen“ — keine rohen Namen",
              tools == "Genutzt: Dokumentstand, Prüfdatei erstellen, PDF herunterladen" and "_" not in tools, tools)
        dl = pg.locator("#inkluagentLog .inkluagent-message-anhang a").first
        check("N6: Download-Knopf mit Dateinamen im Namen", "inkludocs_antrag.pdf" in dl.inner_text() and dl.inner_text().startswith("PDF herunterladen"), dl.inner_text())
        karte = pg.locator("#inkluagentLog .inkluagent-bestaetigung")
        check("Karte: Überschrift, Servertext, Knopf „… bestätigen“", karte.count() == 1 and karte.locator("h4").inner_text() == "Bestätigung nötig"
              and "fiktiv.pdf" in karte.inner_text() and karte.locator("button").inner_text() == "Ablage-Eintrag löschen bestätigen", karte.inner_text() if karte.count() else "")
        pg.evaluate("window.__live = []")
        karte.locator("button").click()
        pg.wait_for_timeout(1500)
        st = karte.locator("[role=status]").inner_text()
        check("abgelaufenes Angebot: sichtbare Meldung in der Karte, Knopf wieder bedienbar, nichts gelöscht",
              "gilt nicht mehr" in st and karte.locator("button").is_enabled(), st)
        live = pg.evaluate("window.__live")
        check("… und genau einmal angesagt", sum("gilt nicht mehr" in x["text"] for x in live) == 1, live)
        pg.add_script_tag(url=AXE)
        pg.wait_for_timeout(400)
        v = pg.evaluate("async () => { const r = await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}); return r.violations.map(v => ({id: v.id, impact: v.impact})); }")
        check("axe mit Chat, Karte und Download-Knopf: keine ernsten Verstöße", not [x for x in v if x["impact"] in ("serious", "critical")], v)

        print("== M2: Upload-Fehler ==")
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
        pg.wait_for_timeout(1500)
        pg.evaluate("window.__live = []")
        pg.set_input_files("#projUpload", {"name": "keine.pdf", "mimeType": "application/pdf", "buffer": b"Nur Text, keine PDF.\n" * 30})
        pg.wait_for_timeout(3000)
        live = pg.evaluate("window.__live")
        fehlertexte = [x for x in live if "keine PDF" in x["text"]]
        nach = live[live.index(fehlertexte[0]) + 1:] if fehlertexte else []
        check("Fehler genau einmal angesagt, danach kein „… wird hochgeladen“",
              len(fehlertexte) == 1 and not any("hochgeladen" in x["text"] for x in nach), live)
        check("keine Skriptfehler", not js, js[:3])
    finally:
        print("  Testprojekt geloescht:", pg.request.delete(B + f"/api/projects/{pid}").status)
        br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
