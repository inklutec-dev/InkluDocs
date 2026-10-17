// InkluAgent — Rückmeldung: der Agent meldet sich, wenn ein Lauf oder Auftrag fertig ist.
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

// ─── Rückmeldung (Steve 14.10.2026; Schalter AGENT_RUECKMELDUNG, window.INKLUAGENT.rueckmeldung, inkluagent/rueckmeldung.py) ───
// Startet der Agent einen Lauf im Hintergrund, meldet er sich, wenn er zu Ende ist: der Server schreibt die Nachricht ins Gespraech
// (auch bei geschlossener Seite). Diese Seite fragt GET /api/agent/status nur, SOLANGE eine wartet (alle 15 s), und meldet eine neue
// wie eine Antwort — Klang „fertig“, Ansage, Ansicht nachziehen —, aber ohne den Fokus zu nehmen: der Nutzer tut gerade etwas
// anderes. Die Nachricht wird angehaengt, nicht der Verlauf neu gebaut (wer im Gespraech liest, bleibt an seiner Stelle). Steht sie
// in einem anderen als dem laufenden Gespraech, sagt die Ansage, in welchem.
// Zustand wie die anderen Merker der Seite im window (auf oberster Ebene nur Funktionen, test_inkluagent_js):
// _inkluagentRmBekannt = juengste Rueckmeldung, die diese Seite kennt (beim ersten Blick: die schon vorhandene), _inkluagentRmBeobachtet.
function inkluagentRueckmeldungAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.rueckmeldung) && inkluagentUeberall();
}
// {wartet, letzte, neu: [{id, im_laufenden, titel, project_id}]}; {aus: true} = nicht angemeldet oder ohne Schalter; null = gerade
// nicht erreichbar
async function inkluagentRueckmeldungStand() {
    try {
        const res = await fetch('/api/agent/status');
        if (res.status === 401 || res.status === 404) return { aus: true };
        if (!res.ok) return null;
        return ((await res.json()) || {}).rueckmeldung || { aus: true };
    } catch (e) {
        return null;
    }
}
async function inkluagentRueckmeldungenBeobachten() {
    const ABSTAND_MS = 15000;
    const FEHLER_MAX = 5;   // so viele Fehlversuche nacheinander (Netz weg), dann erst beim naechsten Anlass wieder
    inkluagentLiveBeobachten();   // Live-Fenster (16.10.2026): dieselben Anlaesse koennen einen Lauf im Hintergrund gestartet haben
    if (!inkluagentRueckmeldungAn() || window._inkluagentRmBeobachtet) return;
    window._inkluagentRmBeobachtet = true;
    try {
        let rm = await inkluagentRueckmeldungStand();
        let fehler = 0;
        while (true) {
            if (rm === null) {
                if (++fehler >= FEHLER_MAX) break;
            } else {
                fehler = 0;
                if (rm.aus) break;
                if (window._inkluagentRmBekannt == null) window._inkluagentRmBekannt = rm.letzte || 0;
                if ((rm.letzte || 0) > window._inkluagentRmBekannt && !inkluagentBeschaeftigt()) {
                    // alle, die neuer sind als die zuletzt gesehene — kommen zwei zwischen zwei Abfragen, geht keine verloren
                    const neue = (rm.neu || []).filter(r => r.id > window._inkluagentRmBekannt);
                    window._inkluagentRmBekannt = rm.letzte;
                    if (neue.length) await inkluagentRueckmeldungZeigen(neue);
                }
                // weiter fragen, solange eine wartet oder eine neue noch nicht gezeigt ist (die Seite sendete gerade selbst)
                if (!(rm.wartet > 0) && !((rm.letzte || 0) > window._inkluagentRmBekannt)) break;
            }
            await new Promise(r => setTimeout(r, ABSTAND_MS));
            rm = await inkluagentRueckmeldungStand();
        }
    } finally {
        window._inkluagentRmBeobachtet = false;
    }
}
// neue: Rueckmeldungen, aelteste zuerst. Die im laufenden Gespraech werden angehaengt (wenn der Agent offen ist), EIN Klang und EINE
// Ansage fuer alle zusammen; die in anderen Gespraechen nennt die Ansage.
async function inkluagentRueckmeldungZeigen(neue) {
    if (typeof window.seitenleisteGespraecheLaden === 'function') window.seitenleisteGespraecheLaden();
    // Arbeitete ein Lauf in dem Projekt, das gerade offen ist: die Ansicht zeigt seine Ergebnisse ohne Neuladen
    const hier = window._inkluagentProjectId == null ? null : String(window._inkluagentProjectId);
    if (neue.some(r => r.project_id != null && String(r.project_id) === hier)) await inkluagentAnsichtNachziehen(window._inkluagentProjectId);
    const imGespraech = neue.filter(r => r.im_laufenden);
    const anderswo = neue.filter(r => !r.im_laufenden);
    const panel = document.getElementById('inkluagentPanel');
    const elemente = (panel && !panel.hidden && imGespraech.length) ? await inkluagentRueckmeldungAnhaengen(imGespraech.map(r => r.id)) : [];
    const el = elemente.length ? elemente[elemente.length - 1] : null;
    const teile = [];
    if (imGespraech.length === 1) {
        teile.push(el && inkluagentUeberlagerungOffen() ? t('Rückmeldung vom InkluAgent ist da. „Zurück zum Gespräch“ führt zu ihr.')
                                                         : t('Rückmeldung vom InkluAgent ist da.'));
    } else if (imGespraech.length > 1) {
        teile.push(t('{n} Rückmeldungen vom InkluAgent sind da.', { n: imGespraech.length }));
    }
    if (anderswo.length === 1) {
        teile.push(t('Der InkluAgent hat sich im Gespräch „{titel}“ gemeldet. Du findest es unter „Frühere Gespräche“.',
                     { titel: anderswo[0].titel || t('Gespräch ohne Titel') }));
    } else if (anderswo.length > 1) {
        teile.push(t('{n} Rückmeldungen stehen in anderen Gesprächen. Du findest sie unter „Frühere Gespräche“.', { n: anderswo.length }));
    }
    if (imGespraech.length) {
        inkluagentAntwortMelden(el, false, teile.join(' '));
    } else {
        inkluagentKlang('fertig');
        announce(teile.join(' '));
    }
}
// Die Rueckmeldungen (Kennungen ids) aus dem Verlauf holen und anhaengen — mit Projektmarke, wenn das Gespraech zuletzt woanders
// war. Rueckgabe: die neuen Elemente (schon vorhandene — zweite Seite, neu geladen — werden nicht noch einmal angehaengt).
async function inkluagentRueckmeldungAnhaengen(ids) {
    const neu = [];
    try {
        const pid = window._inkluagentProjectId;
        const res = await fetch('/api/agent/verlauf' + (pid == null ? '' : '?projekt=' + encodeURIComponent(pid)));
        const log = document.getElementById('inkluagentLog');
        if (!res.ok || !log) return neu;
        const nachrichten = (await res.json()).messages || [];
        // Nur mitscrollen, wer schon unten war — wer weiter oben liest, behaelt seine Stelle
        const unten = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
        for (const nid of ids) {
            const m = nachrichten.find(x => x.id === nid);
            if (!m || log.querySelector('[data-nachricht-id="' + nid + '"]')) continue;
            if (window._inkluagentLetztesProjekt !== (m.project_id == null ? null : String(m.project_id))) {
                inkluagentMarkeAnhaengen(log, m.project_id, m.projekt_name, m.projekt_geloescht);
            }
            const el = inkluagentAppendMessage('assistant', m.content, m.werkzeuge, m.anhang, m.sprache);
            if (el) { el.dataset.nachrichtId = String(nid); neu.push(el); }
        }
        if (unten) log.scrollTop = log.scrollHeight;
    } catch (e) {
        // ohne Verlauf bleibt es bei Klang und Ansage; die Nachricht steht beim naechsten Laden da
    }
    return neu;
}
