// InkluAgent: Chat-Bereich im Projekt (Aufbau, Verlauf, Senden, Bestaetigungs-Karten, Fokus- und Ansageregeln).
//
// Ausgelagert aus backend/templates/app.html (09.10.2026, Konzept InkluAgent, Schritt 1) — der Code ist unveraendert,
// nur 4 Leerzeichen weniger eingerueckt (Beleg im Commit: Pruefsumme des Blocks vor und nach dem Umzug gleich).
// Geladen von app.html VOR dem Seitenskript; diese Datei definiert nur Funktionen (und einen Merker) und fuehrt beim
// Laden nichts aus. Sie nutzt die globalen Helfer der Seite: t() (Texte aus window.I18N, 6 Sprachen), announce(),
// showProject(), window.WERKZEUG_NAMEN, window.Formular. Aufrufer: app.html (inkluagentSectionHtml/inkluagentInit),
// dokument.js, formular.js, uebersetzen.js.

// ─── InkluAgent (KI-Agent pro Projekt) ────────────────────
// Karbe-Feintuning (06.06.2026):
//  - Button-Beschriftung auf ein Wort (öffnen/schließen wird per
//    aria-expanded an Screenreader gemeldet; visuell reicht das Wort).
//    Seit 09.10.2026 „InkluAgent“ statt „Chatbot“ (Steve, Michael einverstanden).
//  - Früherer H2 "Chatbot zur Alt-Text-Hilfe" entfällt, weil der Button
//    nun selbst die Sektion betitelt (aria-labelledby auf den Button).
//  - Der erklärende Infotext steht jetzt UNTER dem Button — Karbes Wunsch,
//    damit der Schalter sofort ins Auge fällt, der Text dient als Hinweis.
// Variante (28.08.2026): 'formular' = Quickinfo-Werkzeug — gleicher Kasten, gleiche
// Bedienung, nur Einleitung und Platzhalter sprechen von Feldern statt Bildern.
function inkluagentSectionHtml(projectId, variante) {
    const formular = variante === 'formular';
    return ''
        + '<section class="inkluagent-section" aria-labelledby="inkluagentToggle" data-project-id="' + projectId + '" data-variante="' + (formular ? 'formular' : 'bilder') + '">'
        // Akkordeon-Muster (WAI-ARIA, Steve 28.08.2026): der Auf/Zu-Knopf steht IN der Ueberschrift
        // Ebene 2 — per Ueberschriften-Navigation anspringen und sofort druecken; die Nachrichten
        // sind Ebene 3 darunter. Ein Wort fuer beides: „InkluAgent“ (09.10.2026, vorher „Chatbot“) — derselbe
        // Name wie der Absender der Nachrichten (Ueberschrift 3) und in der Einleitung.
        +   '<h2 id="inkluagentHeading" class="section-title" style="margin:0 0 0.4rem 0;">'
        +     '<button id="inkluagentToggle" type="button" aria-expanded="false" aria-controls="inkluagentPanel" class="inkluagent-toggle">'
        +       t('InkluAgent')
        +     '</button>'
        +   '</h2>'
        // InkluAgent-Ausbau Runde 1, Schritt 3 (Schalter funktionen.AGENT_HILFE, window.FUNKTIONEN.agent_hilfe): drei kurze
        // Stichpunkte statt der langen Einleitung, der erste ist der KI-Hinweis nach KI-VO Art. 50; darunter der Weg zur
        // Hilfe-Seite. Echte Liste (VoiceOver: „Liste, 3 Objekte“). Schalter aus = die Einleitung wie bisher.
        + (inkluagentKurzhilfe() ? inkluagentKurzhilfeHtml() : (''
        +   '<p class="inkluagent-intro">'
        // KI-VO Art. 50 Abs. 1 (gilt seit 02.08.2026): Menschen muessen
        // erkennen, dass sie mit einem KI-System sprechen. Der Name
        // „InkluAgent“ sagt das allein nicht — der ausdrueckliche Satz
        // „… ist ein KI-Assistent“ direkt unter dem Knopf sagt es.
        // (25.08.2026: stand als Jinja-Kommentar INNERHALB des raw-Blocks,
        // blieb woertlich im JavaScript stehen und brach das ganze
        // App-Skript mit einem SyntaxError ab. Im raw-Block nur
        // JS-Kommentare verwenden.)
        +   (formular
            ? t('Der InkluAgent ist ein KI-Assistent. Bitte ihn um Hilfe — z.B. eine Quickinfo für ein Feld generieren, einen Text kürzer oder einheitlich formulieren, in den Stammdaten nachsehen oder die Anmerkungen des Gastes zusammenfassen. Felder per Nummer benennen (z.B. <em>Feld 3</em>, <em>Felder 1-5</em>).')
            : t('Der InkluAgent ist ein KI-Assistent. Bitte ihn um Hilfe — z.B. einen Alt-Text für ein Bild generieren, einen vorhandenen Text in leichter Sprache umformulieren oder in eine andere Sprache übersetzen. Bilder per Nummer benennen (z.B. <em>Bild 3</em>, <em>Bilder 1-5</em>). Maximal 10 Bilder pro Anfrage.'))
        +   '</p>'))
        +   '<div id="inkluagentPanel" class="inkluagent-panel" hidden>'
        // tabindex=0 (11.09.2026, axe scrollable-region-focusable): der Verlauf wird bei vielen
        // Nachrichten scrollbar und muss dann per Tastatur erreichbar sein (WCAG 2.1.1).
        // aria-live="off" (Pruefung 3 Barrierefreiheit, H1/N1): role=log ist sonst von selbst „polite“ — beim Neuzeichnen
        // (Ansichtswechsel, Neuladen, Auf/Zu) wurde der ganze Verlauf erneut angesagt, jede neue Antwort doppelt (Live-Region
        // UND Fokus) und die eigene Nachricht als Echo. Jetzt sagt nichts im Verlauf von selbst etwas an: die neue Antwort
        // bekommt den Fokus (der Screenreader liest sie vollständig), Warten und Fehler meldet die Statuszeile.
        +     '<div id="inkluagentLog" class="inkluagent-log" role="log" aria-live="off" aria-label="' + t('Chat-Verlauf') + '" tabindex="0"></div>'
        +     '<form id="inkluagentForm" class="inkluagent-form">'
        +       '<label for="inkluagentInput" class="visually-hidden">' + t('Nachricht an den InkluAgent') + '</label>'
        +       '<textarea id="inkluagentInput" rows="2" maxlength="5000" '
        +         'placeholder="' + (formular ? t("z.B. 'Feld 3 kürzer fassen' oder 'Generiere eine Quickinfo für Feld 5'") : t("z.B. 'Bild 3 in leichter Sprache' oder 'Generiere Alt-Text für Bild 5'")) + '" '
        +         'aria-describedby="inkluagentHint"></textarea>'
        +       '<div id="inkluagentHint" class="inkluagent-hint">' + t('Enter sendet, Shift+Enter macht einen Zeilenumbruch. Maximal 5000 Zeichen.') + '</div>'
        +       '<div class="inkluagent-controls">'
        +         '<button type="submit" id="inkluagentSendBtn" class="btn btn-primary">' + t('Senden') + '</button>'
        +         '<span id="inkluagentStatus" class="inkluagent-status" role="status" aria-live="polite"></span>'
        +       '</div>'
        +     '</form>'
        +   '</div>'
        + '</section>';
}

// Schritt 3 (InkluAgent-Ausbau Runde 1): Kurzhilfe vor dem Chat — nur mit Schalter (window.FUNKTIONEN aus funktionen.py)
function inkluagentKurzhilfe() {
    return !!(window.FUNKTIONEN && window.FUNKTIONEN.agent_hilfe);
}
function inkluagentKurzhilfeHtml() {
    return ''
        + '<ul class="inkluagent-intro inkluagent-kurzhilfe">'
        +   '<li>' + t('Der InkluAgent ist eine KI und arbeitet mit deinen Projekten und Dateien.') + '</li>'
        +   '<li>' + t('Sag ihm in eigenen Worten, was du brauchst.') + '</li>'
        +   '<li>' + t('Bevor etwas Credits kostet oder sich nicht rückgängig machen lässt, fragt er dich.') + '</li>'
        + '</ul>'
        + '<p class="inkluagent-hilfe-link"><a href="/hilfe/inkluagent">' + t('Alles, was der InkluAgent kann') + '</a></p>';
}

// Schritt 5 (InkluAgent-Ausbau Runde 1): Ansicht je Konto. In der Agentenansicht (<body data-oberflaeche="agent">, vom
// Server gesetzt) zeigt eine Projektseite nur die Hauptueberschrift und den InkluAgent, gross und geoeffnet; alles andere
// im Hauptbereich bekommt hidden (auch fuer Screenreader weg). Die Seitenleiste bleibt. Laeuft nach jedem Neuzeichnen
// (inkluagentInit), weil die Projektansichten den Hauptbereich neu bauen. Manuelle Ansicht: nichts aendert sich.
function inkluagentAgentenansicht() {
    if (!document.body || document.body.getAttribute('data-oberflaeche') !== 'agent') return false;
    if (window._agentenansichtHierAus) return false;   // „Dieses Projekt in der manuellen Ansicht zeigen“ (nur diese Seite)
    const main = document.getElementById('main');
    const sec = main && main.querySelector('.inkluagent-section');
    if (!sec) return false;
    const h1 = main.querySelector('h1');
    const behalten = [sec, h1].filter(Boolean);
    (function lauf(el) {
        Array.from(el.children).forEach(k => {
            if (behalten.indexOf(k) >= 0) return;
            if (behalten.some(b => k.contains(b))) { lauf(k); return; }
            if (k.hidden || /^(SCRIPT|STYLE|TEMPLATE|DIALOG)$/.test(k.tagName)) return;
            k.hidden = true;
            k.setAttribute('data-agentenansicht', 'aus');
        });
    })(main);
    sec.classList.add('inkluagent-gross');
    // Hochladen und die Knoepfe des Projekts gibt es in Runde 1 nur in der manuellen Ansicht (das Hochladefeld im Agenten
    // kommt in Runde 2): ein Knopf zeigt dieses Projekt fuer diese Seite manuell, ohne die gespeicherte Einstellung zu aendern.
    if (!sec.querySelector('.inkluagent-agentenansicht-hinweis')) {
        const p = document.createElement('p');
        p.className = 'inkluagent-agentenansicht-hinweis';
        p.appendChild(document.createTextNode(t('Hochladen und die Knöpfe des Projekts findest du in der manuellen Ansicht.') + ' '));
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn-secondary';
        b.textContent = t('Dieses Projekt in der manuellen Ansicht zeigen');
        b.addEventListener('click', inkluagentAgentenansichtAufheben);
        p.appendChild(b);
        const kopf = sec.querySelector('h2');
        sec.insertBefore(p, kopf ? kopf.nextSibling : sec.firstChild);
    }
    return true;
}

// Nur fuer diese Seite zurueck zur manuellen Ansicht: alles wieder sichtbar, Fokus auf die Hauptueberschrift
function inkluagentAgentenansichtAufheben() {
    window._agentenansichtHierAus = true;
    document.querySelectorAll('[data-agentenansicht="aus"]').forEach(el => { el.hidden = false; el.removeAttribute('data-agentenansicht'); });
    document.querySelectorAll('.inkluagent-gross').forEach(el => el.classList.remove('inkluagent-gross'));
    document.querySelectorAll('.inkluagent-agentenansicht-hinweis').forEach(el => el.remove());
    const h = document.querySelector('#main h1');
    if (h) { if (!h.hasAttribute('tabindex')) h.setAttribute('tabindex', '-1'); h.focus(); }
}

function inkluagentInit(projectId) {
    const p = inkluagentInitLaden(projectId);
    window._inkluagentBereit = p;
    return p;
}
async function inkluagentInitLaden(projectId) {
    window._inkluagentProjectId = projectId;
    const toggle = document.getElementById('inkluagentToggle');
    const input = document.getElementById('inkluagentInput');
    const form = document.getElementById('inkluagentForm');
    if (!toggle) return;
    toggle.addEventListener('click', () => {
        const isOpen = toggle.getAttribute('aria-expanded') === 'true';
        if (isOpen) inkluagentClose(); else inkluagentOpen(projectId, true);
    });
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            inkluagentSend();
        });
    }
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                inkluagentSend();
            }
        });
    }
    let wasOpen = false;
    try { wasOpen = localStorage.getItem('inkluagent.panel.' + projectId) === 'open'; } catch (e) {}
    // Agentenansicht (Schritt 5): nur der InkluAgent gross und geoeffnet; der Fokus bleibt, wo er ist. Geoeffnet, ohne es sich
    // fuer die manuelle Ansicht zu merken — dort steht er weiter so, wie man ihn zuletzt gelassen hat.
    if (inkluagentAgentenansicht()) { await inkluagentOpen(projectId, false, true); return; }
    if (wasOpen) await inkluagentOpen(projectId, false);
    else if (window._inkluagentNeu != null && String(window._inkluagentNeu) === String(projectId)) inkluagentNeuMarkieren(true);
}

async function inkluagentOpen(projectId, focusInput, nichtMerken) {
    const toggle = document.getElementById('inkluagentToggle');
    const panel = document.getElementById('inkluagentPanel');
    const input = document.getElementById('inkluagentInput');
    if (!toggle || !panel) return;
    toggle.setAttribute('aria-expanded', 'true');
    // Beschriftung bleibt „InkluAgent“ — Auf/Zu-Status meldet aria-expanded.
    toggle.textContent = t('InkluAgent');
    panel.hidden = false;
    if (!nichtMerken) { try { localStorage.setItem('inkluagent.panel.' + projectId, 'open'); } catch (e) {} }
    const neu = window._inkluagentNeu != null && String(window._inkluagentNeu) === String(projectId);
    inkluagentNeuMarkieren(false);
    await inkluagentLoadHistory(projectId);
    const letzte = neu ? inkluagentLetzteAntwort() : null;
    if (focusInput && letzte) letzte.focus();
    else if (focusInput && input) input.focus();
}

function inkluagentClose() {
    const toggle = document.getElementById('inkluagentToggle');
    const panel = document.getElementById('inkluagentPanel');
    if (!toggle || !panel) return;
    toggle.setAttribute('aria-expanded', 'false');
    // Beschriftung bleibt „InkluAgent“ — Auf/Zu-Status meldet aria-expanded.
    toggle.textContent = t('InkluAgent');
    panel.hidden = true;
    const projectId = window._inkluagentProjectId;
    if (projectId) {
        try { localStorage.removeItem('inkluagent.panel.' + projectId); } catch (e) {}
    }
}

async function inkluagentLoadHistory(projectId) {
    const log = document.getElementById('inkluagentLog');
    if (!log) return;
    log.textContent = '';
    try {
        const res = await fetch('/api/projects/' + projectId + '/chat/history');
        if (res.status === 401) { window.location.href = '/login'; return; }
        if (!res.ok) {
            inkluagentSetStatus(t('Verlauf konnte nicht geladen werden.'), true);
            return;
        }
        const data = await res.json();
        for (const m of data.messages || []) {
            inkluagentAppendMessage(m.role, m.content, m.werkzeuge, m.anhang);
        }
        log.scrollTop = log.scrollHeight;
    } catch (e) {
        inkluagentSetStatus(t('Fehler beim Laden des Verlaufs.'), true);
    }
}

// Pruefung 4 (Barrierefreiheit M1): eine neue Antwort holt den Fokus nur, wenn er noch im Chat liegt (oder nirgends) —
// wer inzwischen woanders liest oder tippt, bleibt dort und hoert nur „Antwort vom InkluAgent ist da.“ (allgemeine
// Ansage-Region); bei zugeklapptem Chat traegt der Knopf „InkluAgent, neue Antwort“, bis man ihn oeffnet (dann Fokus auf die
// Antwort). Wer im Eingabefeld des Chats schon weiterschreibt, behaelt es.
function inkluagentIstEingabe(el) {
    if (!el) return false;
    if (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
    return el.tagName === 'INPUT' && ['button', 'submit', 'reset', 'checkbox', 'radio', 'file', 'image'].indexOf((el.type || '').toLowerCase()) < 0;
}
function inkluagentFokusImChat() {
    const panel = document.getElementById('inkluagentPanel');
    if (!panel || panel.hidden) return false;
    const ae = document.activeElement;
    if (!ae || ae === document.body || ae === document.documentElement) return true;
    const sec = panel.closest('.inkluagent-section') || panel;
    if (!sec.contains(ae)) return false;
    return !(ae.id === 'inkluagentInput' && (ae.value || '').trim());
}
function inkluagentLetzteAntwort() {
    const alle = document.querySelectorAll('#inkluagentLog .inkluagent-message.assistant');
    return alle.length ? alle[alle.length - 1] : null;
}
function inkluagentNeuMarkieren(an) {
    const toggle = document.getElementById('inkluagentToggle');
    if (!an) window._inkluagentNeu = null;
    if (!toggle) return;
    let m = toggle.querySelector('.inkluagent-neu');
    if (an && !m) {
        m = document.createElement('span');
        m.className = 'inkluagent-neu';
        m.textContent = t(', neue Antwort');
        toggle.appendChild(m);
    } else if (!an && m) {
        m.remove();
    }
}
// fokussieren: vorher (VOR dem Nachziehen) mit inkluagentFokusImChat() bestimmt
function inkluagentAntwortMelden(el, fokussieren) {
    if (fokussieren && el) { el.focus(); return true; }
    announce(t('Antwort vom InkluAgent ist da.'));
    const panel = document.getElementById('inkluagentPanel');
    if (!panel || panel.hidden) { window._inkluagentNeu = window._inkluagentProjectId; inkluagentNeuMarkieren(true); }
    return false;
}
// Karte auf ihren neuen Zustand stellen (Pruefung 4, M3): „Erledigt“ bzw. „Nicht mehr gültig“. Hat ihr Knopf gerade den
// Fokus, bleibt er stehen (aria-disabled), sonst faellt er weg.
function inkluagentKarteSetzen(angebotId, felder) {
    if (!angebotId || !felder || !felder.zustand) return;
    document.querySelectorAll('.inkluagent-bestaetigung[data-angebot-id="' + CSS.escape(angebotId) + '"]').forEach(k => {
        k.dataset.zustand = felder.zustand;
        const h = k.querySelector('h4'); if (h && felder.titel) h.textContent = felder.titel;
        const p = k.querySelector('.inkluagent-bestaetigung-text'); if (p && felder.text) p.textContent = felder.text;
        const b = k.querySelector('button');
        if (b) {
            if (document.activeElement === b) { b.setAttribute('aria-disabled', 'true'); b.dataset.gesperrt = '1'; }
            else b.remove();
        }
    });
}
// Pruefung 4 (M2): nach Chat-Aktionen, die den Projektzustand aendern, die offene Ansicht still nachziehen (Dokument weg,
// Zaehler, Tagging-Stand …). Der Chat wird dabei aus dem gespeicherten Verlauf neu gezeichnet (Karten mit Zustand vom Server),
// ein angefangener Chat-Text bleibt. Liegt der Fokus in der Ansicht, kommt er auf dasselbe Element zurueck (gleiche id); tippt
// jemand gerade in einem Feld der Ansicht, wartet das Nachziehen, bis er es verlaesst — sonst ginge Getipptes verloren.
let _inkluagentNachziehenWartet = false;
async function inkluagentAnsichtNachziehen(projectId) {
    if (typeof showProject !== 'function' || !projectId) return false;
    const ae = document.activeElement;
    const sec = document.querySelector('.inkluagent-section');
    const imChat = !!(sec && ae && sec.contains(ae));
    if (ae && !imChat && inkluagentIstEingabe(ae)) {
        if (!_inkluagentNachziehenWartet) {
            _inkluagentNachziehenWartet = true;
            ae.addEventListener('focusout', () => setTimeout(() => { _inkluagentNachziehenWartet = false; inkluagentAnsichtNachziehen(projectId); }, 900), { once: true });
        }
        return false;
    }
    const warWo = (ae && ae !== document.body && !imChat) ? inkluagentFokusMerken(ae) : null;
    const entwurfEl = document.getElementById('inkluagentInput');
    const entwurf = entwurfEl ? entwurfEl.value : '';
    await showProject(projectId, true);
    try { await window._inkluagentBereit; } catch (e) {}
    const neuEl = document.getElementById('inkluagentInput');
    if (neuEl && entwurf && !neuEl.value) neuEl.value = entwurf;
    if (warWo) {
        const el = inkluagentFokusFinden(warWo) || document.getElementById('projectName');
        if (el) el.focus();
    }
    return true;
}
// Stelle des Fokus ueber ein Neuzeichnen hinweg: id, sonst Elementart + Text + Position unter gleichen
function inkluagentFokusMerken(el) {
    if (el.id) return { id: el.id };
    const text = (el.textContent || '').trim();
    const gleiche = Array.from(document.querySelectorAll(el.tagName)).filter(x => (x.textContent || '').trim() === text);
    return { tag: el.tagName, text: text, i: gleiche.indexOf(el) };
}
function inkluagentFokusFinden(m) {
    if (m.id) return document.getElementById(m.id);
    const gleiche = Array.from(document.querySelectorAll(m.tag)).filter(x => (x.textContent || '').trim() === m.text);
    return gleiche[Math.max(0, m.i)] || null;
}
// Aktionen einer Antwort (Chat oder Karte): Karten-Zustaende setzen; true = Ansicht nachziehen
function inkluagentAktionenKarten(actions) {
    let nachziehen = false;
    (actions || []).forEach(a => {
        if (!a) return;
        if (a.type === 'karte') inkluagentKarteSetzen(a.angebot_id, a);
        if (a.type === 'ansicht_aktualisieren') nachziehen = true;
    });
    return nachziehen;
}

// Werkzeug-Transparenz (Steve 28.08.2026): Namen der Agenten-Werkzeuge fuer die Live-Zeile „Ruft gerade auf: …“ und die
// Zeile „Genutzt: …“ unter jeder Antwort. Seit Pruefung 3 (30.09.2026, M1) kommen ALLE Namen vom Server
// (window.WERKZEUG_NAMEN, inkluagent/tools/namen.py, 6 Sprachen) — nie ein roher Name wie „dokument_stand“.
function inkluagentWerkzeugLabel(name) {
    const map = window.WERKZEUG_NAMEN || {};
    return map[name] || t('Werkzeug');
}
function inkluagentWerkzeugText(werkzeuge) {
    if (!Array.isArray(werkzeuge)) return '';
    if (!werkzeuge.length) return t('Ohne Werkzeug (aus dem Gesprächsverlauf)');
    const namen = []; werkzeuge.forEach(w => { const l = inkluagentWerkzeugLabel(w); if (namen.indexOf(l) < 0) namen.push(l); });
    return t('Genutzt: {w}', { w: namen.join(', ') });
}

// Bestaetigungs-Karte (Pruefung 3, Entwicklung N1): Text und Knopf kommen vom SERVER; der Knopf fuehrt genau das gespeicherte
// Angebot aus (POST /chat/bestaetigen mit seiner Kennung). Pruefung 4 (M3): waehrend der Anfrage aria-disabled statt disabled
// (der Fokus bleibt auf dem Knopf), danach traegt die Karte ihren Zustand („Erledigt“ / „Nicht mehr gültig“); Fehler stehen
// in ihrer Statuszeile. Das Ergebnis kommt als neue Antwort — Fokus nach der Regel oben (M1).
async function inkluagentBestaetigen(angebotId, btn, statusEl) {
    const projectId = window._inkluagentProjectId;
    if (!projectId || !btn || btn.getAttribute('aria-disabled') === 'true') return;
    btn.setAttribute('aria-disabled', 'true');
    if (statusEl) statusEl.textContent = t('Wird ausgeführt …');
    try {
        const res = await fetch('/api/projects/' + projectId + '/chat/bestaetigen', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ angebot_id: angebotId }) });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            if (statusEl) statusEl.textContent = (data && typeof data.detail === 'string') ? data.detail : t('Fehler. Bitte erneut versuchen.');
            if (data && data.karte) inkluagentKarteSetzen(angebotId, data.karte);
            else btn.removeAttribute('aria-disabled');
            if (data && inkluagentAktionenKarten(data.actions)) await inkluagentAnsichtNachziehen(projectId);
            return;
        }
        if (statusEl) statusEl.textContent = '';
        const fokussieren = inkluagentFokusImChat();
        const el = inkluagentAppendMessage('assistant', data.reply || '', data.werkzeuge || [], data.anhang || []);
        if (data.karte) inkluagentKarteSetzen(angebotId, data.karte);
        const nachziehen = inkluagentAktionenKarten(data.actions);
        const log = document.getElementById('inkluagentLog');
        if (log) log.scrollTop = log.scrollHeight;
        if (nachziehen && await inkluagentAnsichtNachziehen(projectId)) {
            inkluagentAntwortMelden(inkluagentLetzteAntwort(), fokussieren);
        } else {
            inkluagentAntwortMelden(el, fokussieren);
        }
        if (btn.isConnected && document.activeElement !== btn) btn.remove();
    } catch (e) {
        btn.removeAttribute('aria-disabled');
        if (statusEl) statusEl.textContent = t('Verbindungsfehler. Bitte erneut versuchen.');
    }
}

// Meine Ausgaben (11.09.2026): Download-Knopf + Link ins Regal unter einer Bot-Antwort, wenn der Bot
// umgewandelt oder exportiert hat (anhang aus dem Werkzeug-Ergebnis, im Verlauf mitgespeichert).
function inkluagentAnhangEl(anhang) {
    const box = document.createElement('div');
    box.className = 'inkluagent-message-anhang';
    box.style.cssText = 'display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;margin-top:0.5rem;';
    (anhang || []).forEach(a => {
        if (a && a.art === 'bestaetigung' && a.angebot_id) {
            const offen = !a.zustand || a.zustand === 'offen';
            const karte = document.createElement('div');
            karte.className = 'inkluagent-bestaetigung';
            karte.dataset.angebotId = a.angebot_id;
            karte.dataset.zustand = offen ? 'offen' : a.zustand;
            const h = document.createElement('h4');
            h.textContent = a.titel || t('Bestätigung nötig');
            const p = document.createElement('p');
            p.className = 'inkluagent-bestaetigung-text';
            p.textContent = a.text || '';
            karte.appendChild(h); karte.appendChild(p);
            // Erledigte und veraltete Karten ohne Knopf: bei der Ueberschriften-Navigation keine offene Frage mehr
            if (offen) {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'btn btn-primary';
                b.textContent = a.knopf || t('Bestätigen');
                const st = document.createElement('p');
                st.className = 'inkluagent-bestaetigung-status';
                st.setAttribute('role', 'status');
                b.addEventListener('click', () => inkluagentBestaetigen(a.angebot_id, b, st));
                karte.appendChild(b); karte.appendChild(st);
            }
            box.appendChild(karte);
            return;
        }
        if (!a || !a.download_url) return;
        if (a.gueltig_bis && Date.parse(a.gueltig_bis) < Date.now()) {
            const weg = document.createElement('p');
            weg.className = 'feld-hinweis';
            weg.textContent = t('Download von „{name}“ abgelaufen (Links gelten 24 Stunden). Bitte im Chat neu anfordern.', { name: a.dateiname || '' });
            box.appendChild(weg);
            return;
        }
        const dl = document.createElement('a');
        dl.className = 'btn btn-primary';
        dl.href = a.download_url;
        dl.setAttribute('download', a.dateiname || '');
        // Seit 30.09.2026 auch Tabellen aus dem Chatbot (Alt-Texte / Quickinfos herunterladen); ein ZIP aus Tabellen sagt, was drin
        // ist, und jeder Knopf traegt den Dateinamen (Pruefung 3, N6 — im Verlauf standen sonst gleichnamige Knoepfe)
        const formate = { csv: 'CSV', xlsx: 'Excel', json: 'JSON' };
        const beschriftung = { zip: (a.format && formate[a.format]) ? t('ZIP mit {f}-Dateien herunterladen', { f: formate[a.format] }) : t('ZIP herunterladen'),
                               csv: t('CSV herunterladen'), xlsx: t('Excel herunterladen'), json: t('JSON herunterladen'),
                               // Testfassung aus „Testweise taggen“ (09.10.2026): derselbe Name wie der Knopf in der Karte
                               testfassung: t('Testfassung herunterladen (mit Wasserzeichen)') };
        dl.textContent = beschriftung[a.label] || (a.art === 'docx' ? t('Word herunterladen') : t('PDF herunterladen'));
        if (a.dateiname) {
            const name = document.createElement('span');
            name.className = 'visually-hidden';
            name.textContent = ' – ' + a.dateiname;
            dl.appendChild(name);
        }
        box.appendChild(dl);
        if (a.ausgaben_url) {
            const zu = document.createElement('a');
            zu.className = 'btn btn-secondary';
            zu.href = a.ausgaben_url;
            zu.textContent = t('Zur Ablage');
            box.appendChild(zu);
        }
        const tab = document.getElementById('ausgabenTab');
        if (tab && typeof a.ausgaben_anzahl === 'number') tab.textContent = t('Ablage ({n})', { n: a.ausgaben_anzahl });
    });
    return box.children.length ? box : null;
}

function inkluagentAppendMessage(role, content, werkzeuge, anhang) {
    const log = document.getElementById('inkluagentLog');
    if (!log) return;
    // VoiceOver (Steve 28.08.2026): jede Nachricht ist fokussierbar (tabindex -1) und
    // traegt ihren Absender als Ueberschrift 3 — so laesst sich der Verlauf per
    // Ueberschriften-Navigation durchgehen und eine Antwort in Ruhe nachlesen; die
    // Live-Ansage allein war bei langen Antworten abgeschnitten.
    const wrapper = document.createElement('div');
    wrapper.className = 'inkluagent-message ' + (role === 'user' ? 'user' : 'assistant');
    wrapper.tabIndex = -1;
    const roleEl = document.createElement('h3');
    roleEl.className = 'inkluagent-message-role';
    roleEl.style.margin = '0 0 0.3rem 0';
    roleEl.textContent = (role === 'user' ? t('Du') : 'InkluAgent');
    const contentEl = document.createElement('div');
    contentEl.className = 'inkluagent-message-content';
    contentEl.textContent = content || '';
    wrapper.appendChild(roleEl);
    wrapper.appendChild(contentEl);
    if (role === 'assistant' && Array.isArray(anhang) && anhang.length) {
        const aEl = inkluagentAnhangEl(anhang);
        if (aEl) wrapper.appendChild(aEl);
    }
    const wtext = role === 'assistant' ? inkluagentWerkzeugText(werkzeuge) : '';
    if (wtext) {
        const wEl = document.createElement('div');
        wEl.className = 'inkluagent-message-tools';
        wEl.style.cssText = 'font-size:0.875rem;color:var(--text-muted);margin-top:0.35rem;';
        wEl.textContent = wtext;
        wrapper.appendChild(wEl);
    }
    log.appendChild(wrapper);
    return wrapper;
}

function inkluagentSetStatus(text, isError) {
    const status = document.getElementById('inkluagentStatus');
    if (!status) return;
    status.textContent = text || '';
    if (isError) status.setAttribute('data-state', 'error');
    else status.removeAttribute('data-state');
    // Pruefung 4 (M1): die Statuszeile liegt im Chat — ist er zu oder der Fokus woanders, hoert man sie nicht
    if (isError && text && !inkluagentFokusImChat()) announce(text);
}

async function inkluagentSend() {
    const projectId = window._inkluagentProjectId;
    const input = document.getElementById('inkluagentInput');
    const sendBtn = document.getElementById('inkluagentSendBtn');
    const log = document.getElementById('inkluagentLog');
    if (!projectId || !input || !sendBtn || !log) return;
    const message = input.value.trim();
    if (!message) {
        inkluagentSetStatus(t('Bitte eine Nachricht eingeben.'), true);
        input.focus();
        return;
    }
    if (message.length > 5000) {
        inkluagentSetStatus(t('Nachricht ist zu lang (max. 5000 Zeichen).'), true);
        return;
    }
    inkluagentAppendMessage('user', message);
    log.scrollTop = log.scrollHeight;
    input.value = '';
    sendBtn.disabled = true;
    inkluagentSetStatus(t('InkluAgent denkt nach...'), false);
    try {
        // Stream (NDJSON, 28.08.2026): je Werkzeugaufruf eine Zeile -> Live-Zeile im Status
        // (aria-live, VoiceOver sagt sie an), am Ende die Antwort mit Werkzeugliste.
        const res = await fetch('/api/projects/' + projectId + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/x-ndjson' },
            body: JSON.stringify({ message })
        });
        if (res.status === 401) { window.location.href = '/login'; return; }
        if (!res.ok) {
            let detail = 'HTTP ' + res.status;
            try { const e = await res.json(); detail = e.detail || detail; } catch (_) {}
            inkluagentSetStatus(detail, true);
            return;
        }
        let data = null;
        const genutzt = [];
        if (res.body && res.body.getReader && (res.headers.get('Content-Type') || '').indexOf('x-ndjson') >= 0) {
            const reader = res.body.getReader(); const dec = new TextDecoder(); let rest = '';
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                rest += dec.decode(value, { stream: true });
                let nl;
                while ((nl = rest.indexOf('\n')) >= 0) {
                    const zeile = rest.slice(0, nl).trim(); rest = rest.slice(nl + 1);
                    if (!zeile) continue;
                    let ev; try { ev = JSON.parse(zeile); } catch (_) { continue; }
                    if (ev.type === 'tool') {
                        genutzt.push(ev.name);
                        inkluagentSetStatus(t('Ruft gerade auf: {w}', { w: inkluagentWerkzeugLabel(ev.name) }), false);
                    } else if (ev.type === 'reply') { data = ev; }
                }
            }
            if (!data && rest.trim()) { try { data = JSON.parse(rest.trim()); } catch (_) {} }
        } else {
            data = await res.json();
        }
        if (!data) { inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true); return; }
        // Fokusregel (Pruefung 4, M1) VOR allem anderen bestimmen: liegt der Fokus noch im Chat?
        const fokussieren = inkluagentFokusImChat();
        const antwortEl = inkluagentAppendMessage('assistant', data.reply || t('(leere Antwort)'), Array.isArray(data.werkzeuge) ? data.werkzeuge : genutzt, data.anhang);
        log.scrollTop = log.scrollHeight;
        inkluagentSetStatus('', false);
        let nachziehen = inkluagentAktionenKarten(data.actions);
        // Fokus auf die Antwort statt zurueck ins Eingabefeld: der Screenreader liest sie
        // vollstaendig und von vorn; Tab fuehrt danach zum Eingabefeld.
        window._inkluagentFokusAntwort = true;
        // Refresh-Actions vom Chatbot: pro Bild Textarea-Wert live setzen.
        // Wenn Bild nicht im DOM (z.B. noch nicht gerendert), Fallback auf Full-Reload.
        const refreshActions = (data.actions || []).filter(a => a && a.type === 'refresh_image');
        let needsFullReload = false;
        let updatedCount = 0;
        for (const ra of refreshActions) {
            const ta = document.getElementById('alttext_' + ra.image_id);
            if (ta && ra.alt_text != null) {
                ta.value = ra.alt_text;
                updatedCount++;
                const ind = document.getElementById('saved_' + ra.image_id);
                if (ind) { ind.classList.add('visible'); setTimeout(() => ind.classList.remove('visible'), 2000); }
            } else if (!ta) {
                needsFullReload = true;
            }
            const lta = document.getElementById('langtext_' + ra.image_id);
            if (lta && ra.langbeschreibung != null) {
                lta.value = ra.langbeschreibung;
            } else if (ra.langbeschreibung) {
                // Neue Langbeschreibung gesetzt, aber Feld noch nicht im DOM -> Full-Reload
                needsFullReload = true;
            }
        }
        if (updatedCount > 0) {
            announce(updatedCount === 1 ? t('InkluAgent hat 1 Alt-Text aktualisiert.') : t('InkluAgent hat {n} Alt-Texte aktualisiert.', { n: updatedCount }));
        }
        // Formular-Projekte (28.08.2026): refresh_feld-Aktionen setzt formular.js selbst um.
        if (window.Formular && typeof Formular.chatAktionen === 'function') Formular.chatAktionen(data.actions || []);
        // Ansicht nachziehen (Pruefung 4, M2) — auch, wenn ein Bild noch nicht im DOM war (frueher: showProject nach 800 ms,
        // Fokus ging dabei verloren). Danach bekommt die neue Antwort den Fokus, wenn er vorher im Chat lag.
        if ((nachziehen || needsFullReload) && await inkluagentAnsichtNachziehen(projectId)) {
            inkluagentAntwortMelden(inkluagentLetzteAntwort(), fokussieren);
        } else {
            inkluagentAntwortMelden(antwortEl, fokussieren);
        }
    } catch (e) {
        inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true);
    } finally {
        sendBtn.disabled = false;
        // Zurueck ins Eingabefeld nur, wenn der Fokus im Chat war (Fehlerfall) — nie aus einer anderen Stelle heraus
        if (window._inkluagentFokusAntwort) { window._inkluagentFokusAntwort = false; }
        else if (inkluagentFokusImChat() && document.activeElement !== input) { input.focus(); }
    }
}
