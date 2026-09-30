#!/usr/bin/env python3
"""Klicktest Chat und Upload mit Mitschnitt aller Live-Regionen (Pruefung 3 und 4, 30.09.2026).
  H1  Verlauf wird beim Neuzeichnen (Ansichtswechsel, Chat zu/auf, Neuladen) NICHT angesagt; Log ist aria-live="off"
  M1  „Genutzt: …“ mit Anzeigenamen vom Server, keine rohen Werkzeugnamen
  N6  Download-Knopf traegt den Dateinamen
  P4-M1  Fokus nur dann auf die neue Antwort, wenn er noch im Chat liegt; sonst „Antwort vom InkluAgent ist da.“ (einmal),
         bei zugeklapptem Chat „Chatbot, neue Antwort“ am Knopf, beim Oeffnen Fokus auf die Antwort; wer im Chat-Feld
         weiterschreibt, behaelt Fokus und Text (Antworten hier als Attrappe im Browser, ohne KI)
  P4-M3  veraltete Karte: aus dem Verlauf ohne Knopf („Nicht mehr gültig“); live geklickt: Knopf aria-disabled, Fokus bleibt
  P4-M2  echte Karte aus dem Chat (KI): Dokument per Karte loeschen -> Dokumentkarte sofort weg, Zaehler stimmt, Karte
         „Erledigt“, Fokus auf der Ergebnis-Antwort, keine Doppelansage
  M2  kaputter Upload: der Fehler wird EINMAL angesagt, kein „… wird hochgeladen“ danach
Legt ein Projekt an und loescht es (samt Ablage). Aufruf: /home/claude/.venv-pw/bin/python ui_chat_barrierefrei.py <korpus>"""
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

# Chat-Antwort als Attrappe (nur wenn window.__chatAttrappe gesetzt): kommt nach ms Millisekunden als NDJSON wie vom Server
ATTRAPPE = """
(() => {
  if (window.__fetchEcht) return;
  window.__fetchEcht = window.fetch;
  window.fetch = async (url, opt) => {
    const a = window.__chatAttrappe;
    if (a && typeof url === 'string' && /\\/api\\/projects\\/\\d+\\/chat$/.test(url) && opt && opt.method === 'POST') {
      await new Promise(r => setTimeout(r, a.ms));
      const zeilen = [{ type: 'tool', name: 'dokument_stand' },
                      { type: 'reply', reply: a.antwort, werkzeuge: ['dokument_stand'], actions: [], anhang: [] }];
      return new Response(zeilen.map(z => JSON.stringify(z)).join('\\n') + '\\n', { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } });
    }
    return window.__fetchEcht(url, opt);
  };
})();
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


def hoch(pg, pid, pfad, name):
    with open(pfad, "rb") as f:
        pg.request.post(B + "/api/upload", multipart={"file": {"name": name, "mimeType": "application/pdf", "buffer": f.read()},
                                                      "project_id": str(pid)})
    for _ in range(60):
        time.sleep(2)
        if (pg.request.get(B + f"/api/projects/{pid}").json().get("project") or {}).get("status") not in ("extracting", "processing"):
            break


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
        hoch(pg, pid, os.path.join(K, "antrag_pflege.pdf"), "antrag.pdf")
        hoch(pg, pid, os.path.join(K, "synth_roh.pdf"), "roh.pdf")
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

        def live_mit(text):
            return [x for x in pg.evaluate("window.__live") if text in x["text"]]
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

        print("== Prüfung 4, M3: veraltete Karten ==")
        karte = pg.locator("#inkluagentLog .inkluagent-bestaetigung").first
        check("Karte aus dem Verlauf, deren Angebot nicht mehr gilt: „Nicht mehr gültig“, ohne Knopf",
              karte.locator("h4").inner_text() == "Nicht mehr gültig" and karte.locator("button").count() == 0 and "fiktiv.pdf" in karte.inner_text(),
              karte.inner_text())
        pg.evaluate("""inkluagentAppendMessage('assistant', 'Soll ich das löschen? (fiktive Karte)', ['ausgabe_loeschen'],
            [{art: 'bestaetigung', angebot_id: 'e'.repeat(32), titel: 'Bestätigung nötig', text: 'Ablage-Eintrag löschen: „fiktiv2.pdf“.',
              knopf: 'Ablage-Eintrag löschen („fiktiv2.pdf“) bestätigen', zustand: 'offen'}])""")
        lk = pg.locator(".inkluagent-bestaetigung[data-angebot-id='" + "e" * 32 + "']")
        pg.evaluate("window.__live = []")
        lk.locator("button").click()
        pg.wait_for_timeout(1500)
        st = lk.locator("[role=status]").inner_text()
        fokus_auf_knopf = pg.evaluate("document.activeElement && document.activeElement.closest('.inkluagent-bestaetigung') !== null && document.activeElement.tagName === 'BUTTON'")
        check("live geklickt, Angebot gilt nicht mehr: Meldung, Überschrift „Nicht mehr gültig“, Knopf aria-disabled, Fokus bleibt auf dem Knopf",
              "gilt nicht mehr" in st and lk.locator("h4").inner_text() == "Nicht mehr gültig"
              and lk.locator("button").get_attribute("aria-disabled") == "true" and fokus_auf_knopf, (st, lk.inner_text(), fokus_auf_knopf))
        check("… genau einmal angesagt", len(live_mit("gilt nicht mehr")) == 1, pg.evaluate("window.__live"))
        pg.add_script_tag(url=AXE)
        pg.wait_for_timeout(400)
        v = pg.evaluate("async () => { const r = await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}); return r.violations.map(v => ({id: v.id, impact: v.impact})); }")
        check("axe mit Chat, Karten und Download-Knopf: keine ernsten Verstöße", not [x for x in v if x["impact"] in ("serious", "critical")], v)

        print("== Prüfung 4, M1: Fokus nur im Chat (Antwort als Attrappe) ==")
        pg.evaluate(ATTRAPPE)

        def senden(frage, antwort, ms=1500):
            pg.evaluate(f"window.__chatAttrappe = {{ ms: {ms}, antwort: {json.dumps(antwort)} }}; window.__live = []")
            pg.focus("#inkluagentInput")
            pg.fill("#inkluagentInput", frage)
            pg.keyboard.press("Enter")

        def aktiv():
            return pg.evaluate("""(() => { const a = document.activeElement; return { id: a.id, tag: a.tagName, text: (a.innerText || '').slice(0, 200),
                                   wert: a.value || '', antwort: !!(a.classList && a.classList.contains('assistant')) }; })()""")
        senden("Frage A (Test)", "Fiktive Antwort A")
        pg.wait_for_timeout(2500)
        a = aktiv()
        check("A: Fokus blieb im Chat -> neue Antwort hat den Fokus, keine zusätzliche Ansage",
              a["antwort"] and "Fiktive Antwort A" in a["text"] and not live_mit("Antwort vom InkluAgent"), (a, pg.evaluate("window.__live")))
        senden("Frage B (Test)", "Fiktive Antwort B")
        pg.wait_for_timeout(200)
        pg.focus("#projectName")
        pg.wait_for_timeout(2500)
        a = aktiv()
        check("B: Fokus woanders (H1) -> bleibt dort, „Antwort vom InkluAgent ist da.“ genau einmal",
              a["id"] == "projectName" and len(live_mit("Antwort vom InkluAgent ist da.")) == 1, (a, pg.evaluate("window.__live")))
        check("B: Antwort steht trotzdem im Verlauf", "Fiktive Antwort B" in pg.locator("#inkluagentLog").inner_text())
        senden("Frage D (Test)", "Fiktive Antwort D")
        pg.wait_for_timeout(200)
        pg.keyboard.type("weiter")
        pg.wait_for_timeout(2500)
        a = aktiv()
        check("D: wer im Chat-Feld weiterschreibt, behält Fokus und Text; Ansage einmal",
              a["id"] == "inkluagentInput" and a["wert"] == "weiter" and len(live_mit("Antwort vom InkluAgent ist da.")) == 1,
              (a, pg.evaluate("window.__live")))
        pg.fill("#inkluagentInput", "")
        print(im_container("from inkluagent import storage; "
                           f"storage.append_message({pid}, 'assistant', 'Fiktive Antwort C (gespeichert)', werkzeuge=['dokument_stand'])"))
        senden("Frage C (Test)", "Fiktive Antwort C")
        pg.wait_for_timeout(200)
        pg.click("#inkluagentToggle")      # Chat zu, waehrend die Antwort unterwegs ist
        pg.wait_for_timeout(2500)
        a = aktiv()
        knopf = pg.locator("#inkluagentToggle").inner_text().strip()
        check("C: Chat zugeklappt -> Fokus bleibt am Knopf, Knopf „Chatbot, neue Antwort“, Ansage einmal",
              a["id"] == "inkluagentToggle" and knopf == "Chatbot, neue Antwort" and len(live_mit("Antwort vom InkluAgent ist da.")) == 1,
              (a, knopf, pg.evaluate("window.__live")))
        pg.click("#inkluagentToggle")
        pg.wait_for_timeout(1500)
        a = aktiv()
        knopf = pg.locator("#inkluagentToggle").inner_text().strip()
        check("C: beim Öffnen Fokus auf die neue Antwort, Hinweis am Knopf weg",
              a["antwort"] and "Fiktive Antwort C" in a["text"] and knopf == "Chatbot", (a, knopf))
        pg.evaluate("window.__chatAttrappe = null")

        print("== Prüfung 4, M2/M3: Dokument per Karte löschen (echte Karte aus dem Chat) ==")
        pg.click(".ansicht-knoepfe a[data-ansicht=dokument]")
        pg.wait_for_timeout(2000)
        vorher = pg.locator("#dokumenteHeading").inner_text()
        karte = None
        for versuch in range(2):
            pg.evaluate("window.__live = []")
            n_vorher = pg.locator("#inkluagentLog .inkluagent-bestaetigung[data-zustand=offen]").count()
            pg.focus("#inkluagentInput")
            pg.fill("#inkluagentInput", "Lösche bitte das Dokument „roh.pdf“ aus diesem Projekt. Frag mich vorher.")
            pg.keyboard.press("Enter")
            for _ in range(90):
                pg.wait_for_timeout(2000)
                if not pg.locator("#inkluagentSendBtn").is_disabled():
                    break
            offene = pg.locator("#inkluagentLog .inkluagent-bestaetigung[data-zustand=offen]")
            if offene.count() > n_vorher:
                karte = offene.last
                break
        if not karte:
            check("Chat legt eine Lösch-Karte an", False, pg.locator("#inkluagentLog .inkluagent-message.assistant").last.inner_text())
        else:
            aid = karte.get_attribute("data-angebot-id")
            knopf = karte.locator("button").inner_text()
            check("Karte mit Ziel im Knopf („Dokument löschen („roh.pdf“) bestätigen“)", "roh.pdf" in knopf and knopf.endswith("bestätigen"), knopf)
            pg.evaluate("window.__live = []")
            karte.locator("button").click()
            for _ in range(30):
                pg.wait_for_timeout(1000)
                if pg.locator("#dokumenteHeading").inner_text() != vorher:
                    break
            pg.wait_for_timeout(1500)
            liste = pg.locator("#dokListe").inner_text()
            check("Dokumentkarte sofort weg, Zähler stimmt („Dokumente (1)“)",
                  pg.locator("#dokumenteHeading").inner_text() == "Dokumente (1)" and "roh.pdf" not in liste,
                  (vorher, pg.locator("#dokumenteHeading").inner_text(), liste[:200]))
            a = aktiv()
            check("Fokus auf der Ergebnis-Antwort („Erledigt: „roh.pdf“ ist gelöscht.“)", a["antwort"] and "Erledigt" in a["text"] and "gelöscht" in a["text"]
                  and "rückgängig" not in a["text"], a)
            k2 = pg.locator(".inkluagent-bestaetigung[data-angebot-id='" + aid + "']")
            check("Karte „Erledigt“, ohne Knopf", k2.count() == 1 and k2.locator("h4").inner_text() == "Erledigt" and k2.locator("button").count() == 0,
                  k2.inner_text() if k2.count() else "")
            live = pg.evaluate("window.__live")
            check("keine Doppelansage (keine „Antwort … ist da“, keine Ansichts-Ansage, kein Verlauf)",
                  not [x for x in live if "Antwort vom InkluAgent" in x["text"] or "geöffnet" in x["text"] or "fiktiv" in x["text"]], live)
            check("Chat bleibt offen", pg.locator("#inkluagentPanel").is_visible())

        print("== M2: Upload-Fehler ==")
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
        for e in (pg.request.get(B + f"/api/ausgaben?projekt={pid}").json().get("ausgaben") or []):
            pg.request.delete(B + f"/api/ausgaben/{e['id']}")
        print("  Testprojekt geloescht:", pg.request.delete(B + f"/api/projects/{pid}").status)
        br.close()
print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
