// InkluAgent — Gespräch des Kontos: Verlauf, „Neues Gespräch“, „Frühere Gespräche“.
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

// Runde 2, Schritt 2 (Schalter funktionen.AGENT_KONTO, window.INKLUAGENT.konto): EIN Gespraech je Konto ueber alle Projekte.
// Wo das Projekt wechselt, steht eine Marke („Ab hier: Projekt …“); das fruehere Gespraech dieses Projekts (vor dem Umstieg)
// bleibt lesbar — als natives Aufklappelement ueber dem Verlauf, geladen erst beim Aufklappen. Offene Karten anderer Projekte
// zeigt der Server ohne Knopf („Zu bestätigen im Projekt …“).
function inkluagentKonto() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.konto);
}
// Gespraeche (Steve 12.10.2026, Schalter AGENT_GESPRAECHE): im Agenten nur zwei Dinge — „Neues Gespräch“ und „Frühere Gespräche“ auf
// eine eigene Seite (die Liste steht bewusst nicht im Agenten). Seit Michael Karbe, Feedback 20261012 - 2, neben „Senden“; „Frühere
// Gespräche“ sieht aus wie ein Knopf, bleibt aber ein Link (es fuehrt auf eine Seite — HTML vor ARIA; in der Agentenansicht oeffnet der
// zentrale Link-Handler sie im Arbeitsfenster).
function inkluagentGespraecheAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.gespraeche);
}
function inkluagentGespraechKnoepfeHtml() {
    return '<span class="inkluagent-gespraech-knoepfe">'
        + '<button type="button" id="inkluagentNeuesGespraech" class="btn btn-secondary">' + t('Neues Gespräch') + '</button>'
        + '<a href="/gespraeche" class="btn btn-secondary">' + t('Frühere Gespräche') + '</a>'
        + '</span>';
}
async function inkluagentNeuesGespraech() {
    const btn = document.getElementById('inkluagentNeuesGespraech');
    if (!btn || btn.dataset.laeuft || !inkluagentWechselErlaubt()) return;
    btn.dataset.laeuft = '1';
    inkluagentSperreZeigen();
    try {
        const res = await fetch('/api/agent/gespraech/neu', { method: 'POST' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const log = document.getElementById('inkluagentLog');
        if (log) log.textContent = '';
        window._inkluagentLetztesProjekt = undefined;
        inkluagentSetStatus(t('Neues Gespräch begonnen. Das bisherige findest du unter „Frühere Gespräche“.'), false);
        if (inkluagentChatstilAktiv()) inkluagentArbeitsZuruecksetzen();
        if (typeof window.seitenleisteGespraecheLaden === 'function') window.seitenleisteGespraecheLaden();
        const input = document.getElementById('inkluagentInput');
        if (input) input.focus();
    } catch (e) {
        inkluagentSetStatus(t('Das ging nicht. Bitte erneut versuchen.'), true);
    } finally {
        delete btn.dataset.laeuft;
        inkluagentSperreZeigen();
    }
}
function inkluagentMarkeAnhaengen(log, projektId, name, geloescht) {
    const p = document.createElement('p');
    p.className = 'inkluagent-marke';
    p.textContent = projektId == null ? t('Ab hier: ohne Projekt')
        : (geloescht ? t('Ab hier: Projekt „{name}“ (gelöscht)', { name: name || '' }) : t('Ab hier: Projekt „{name}“', { name: name || '' }));
    log.appendChild(p);
    window._inkluagentLetztesProjekt = projektId == null ? null : String(projektId);
}
async function inkluagentKontoVerlaufLaden(projectId, log) {
    try {
        const res = await fetch('/api/agent/verlauf' + (projectId == null ? '' : '?projekt=' + encodeURIComponent(projectId)));
        if (res.status === 401) { inkluagentZurAnmeldung(); return; }
        if (!res.ok) { inkluagentSetStatus(t('Verlauf konnte nicht geladen werden.'), true); return; }
        const data = await res.json();
        window._inkluagentProjektName = data.projekt_name || '';
        window._inkluagentLetztesProjekt = undefined;
        let vorher;
        for (const m of data.messages || []) {
            if (vorher === undefined || m.project_id !== vorher) {
                inkluagentMarkeAnhaengen(log, m.project_id, m.projekt_name, m.projekt_geloescht);
                vorher = m.project_id;
            }
            const el = inkluagentAppendMessage(m.role, m.content, m.werkzeuge, m.anhang, m.sprache);
            if (el && m.id != null) el.dataset.nachrichtId = String(m.id);   // Rueckmeldung: nicht doppelt anhaengen
        }
        inkluagentFrueherZeigen(projectId, data.frueher || 0);
        log.scrollTop = log.scrollHeight;
    } catch (e) {
        inkluagentSetStatus(t('Fehler beim Laden des Verlaufs.'), true);
    }
}
// Vor der eigenen Nachricht eine Marke, wenn das Gespraech zuletzt in einem anderen Projekt war
function inkluagentMarkeVorNachricht(projectId) {
    const log = document.getElementById('inkluagentLog');
    if (!inkluagentKonto() || !log) return;
    const ziel = projectId == null ? null : String(projectId);
    if (window._inkluagentLetztesProjekt !== ziel) inkluagentMarkeAnhaengen(log, projectId, window._inkluagentProjektName || '', false);
}
function inkluagentFrueherZeigen(projectId, anzahl) {
    const panel = document.getElementById('inkluagentPanel');
    if (!panel) return;
    panel.querySelectorAll('.inkluagent-frueher').forEach(el => el.remove());
    if (!anzahl) return;
    const d = document.createElement('details');
    d.className = 'inkluagent-frueher';
    const s = document.createElement('summary');
    s.textContent = anzahl === 1 ? t('Früheres Gespräch in diesem Projekt (1 Nachricht)')
        : t('Früheres Gespräch in diesem Projekt ({n} Nachrichten)', { n: anzahl });
    const box = document.createElement('div');
    box.className = 'inkluagent-frueher-liste';
    d.appendChild(s);
    d.appendChild(box);
    d.addEventListener('toggle', async () => {
        if (!d.open || box.dataset.geladen) return;
        box.dataset.geladen = '1';
        try {
            const res = await fetch('/api/projects/' + projectId + '/chat/history');
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            // Nur lesen: Text und Absender, keine Karten oder Download-Knoepfe (die gelten nicht mehr)
            (data.messages || []).forEach(m => {
                if (m.role !== 'user' && m.role !== 'assistant') return;
                const w = document.createElement('div');
                w.className = 'inkluagent-message ' + (m.role === 'user' ? 'user' : 'assistant');
                const h = document.createElement('h3');
                h.className = 'inkluagent-message-role';
                h.textContent = m.role === 'user' ? t('Du') : 'InkluAgent';
                const c = document.createElement('div');
                c.className = 'inkluagent-message-content';
                inkluagentSpracheSetzen(c, m.sprache);   // Runde 2 (B11, Punkt 4)
                c.textContent = m.content || '';
                w.appendChild(h);
                if (m.role === 'assistant') inkluagentAnhoerenKnopf(h, c);   // auch hier anhören (16.10.2026)
                w.appendChild(c);
                box.appendChild(w);
            });
        } catch (e) {
            box.textContent = t('Das frühere Gespräch konnte nicht geladen werden.');
            delete box.dataset.geladen;
        }
    });
    const trenner = panel.querySelector('.inkluagent-trenner');
    const log = document.getElementById('inkluagentLog');
    // Chatstil (Agentenansicht wie ChatGPT): das aeltere Gespraech steht ueber dem Verlauf, in der Reihenfolge der Zeit
    if (inkluagentChatstilAktiv() && log) { log.parentNode.insertBefore(d, log); return; }
    panel.insertBefore(d, (trenner && trenner.parentNode === panel) ? trenner : null);   // unter Verlauf und Leiste, über der Linie
}

async function inkluagentLoadHistory(projectId) {
    const log = document.getElementById('inkluagentLog');
    if (!log) return;
    log.textContent = '';
    if (inkluagentKonto()) return inkluagentKontoVerlaufLaden(projectId, log);
    try {
        const res = await fetch('/api/projects/' + projectId + '/chat/history');
        if (res.status === 401) { inkluagentZurAnmeldung(); return; }
        if (!res.ok) {
            inkluagentSetStatus(t('Verlauf konnte nicht geladen werden.'), true);
            return;
        }
        const data = await res.json();
        for (const m of data.messages || []) {
            inkluagentAppendMessage(m.role, m.content, m.werkzeuge, m.anhang, m.sprache);
        }
        log.scrollTop = log.scrollHeight;
    } catch (e) {
        inkluagentSetStatus(t('Fehler beim Laden des Verlaufs.'), true);
    }
}
