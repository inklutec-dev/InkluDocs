// InkluAgent — Seitenwechsel durch den Agenten (Projekt öffnen, Seite wechseln, Projekt weg).
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

// ─── Seitenwechsel durch den Agenten (Steve 12.10.2026; Pruefung 13.10.2026, B10, B11, B15) ───
// Nach einer Antwort kann der Server die Seite wechseln lassen: seite_wechseln (genau ein Projekt: oeffnen, anlegen, Datei oder Webseite
// als Projekt; dort mit agent=offen, Fokus auf dieser Antwort), und der Begleiter: hinfuehren (Zielseite, Fokus aufs Ziel),
// sprache_wechseln (ueber /set-language: Konto und Cookie, zurueck auf diese Seite), neu_laden (Ansicht gewechselt). EIN Weg fuer alle:
// 1. Liegt der Fokus nicht im Gespraech, steht Text im Eingabefeld oder ist eine Datei angehaengt, wechselt nichts von selbst: unter der
//    Antwort steht der Link bzw. Knopf dafuer, und die Ansage sagt es (vorher riss der Wechsel mitten aus dem Tippen, B10).
// 2. Sonst erst offene Eingaben speichern (alleAusstehendenSpeichern, wie Seitenleiste und Abmelden), dann wechseln.
// 3. Ein angefangener Chat-Text geht beim Verlassen jeder Seite mit (inkluagentEingabeMerken).
// Sprache und Ansicht wechseln auf ausdrueckliche Bitte direkt, ohne Karte (Steve); nach dem Neuladen steht der Fokus auf der Antwort
// mit dem Rueckweg, nicht auf der Ueberschrift 1 (B11). Hinfuehren in der Agentenansicht oeffnet das Ziel im Arbeitsfenster und setzt
// den Fokus dort — nicht erst auf die Antwort und gleich danach in den Rahmen (B15). Nur eigene Pfade (S14).
function inkluagentWechselZiel(actions) {
    for (const a of (actions || [])) {
        if (!a) continue;
        if (a.type === 'seite_wechseln' && typeof a.url === 'string' && /^\/app\?projekt=\d+$/.test(a.url)) {
            if (window._inkluagentProjectId != null && String(window._inkluagentProjectId) === String(a.project_id)
                && /^\/app\/?$/.test(window.location.pathname)) continue;   // schon dort
            return { url: a.url + '&agent=offen', link: a.url, text: t('Projekt öffnen') };
        }
        if (a.type === 'hinfuehren' && eigenerPfad(a.url) && /^\/[A-Za-z0-9\/_-]*\?hin=[A-Za-z][\w-]{0,40}$/.test(a.url)) {
            const name = String(a.name || '');
            return { url: a.url, link: a.url, name: name, text: t('Hinführen: {name}', { name: name }),
                     fenster: inkluagentSeiteImFensterMoeglich(a.url) };
        }
        if (a.type === 'sprache_wechseln' && /^[a-z]{2}$/.test(a.sprache || '')) {
            return { url: '/set-language/' + a.sprache, link: '/set-language/' + a.sprache, sprache: a.sprache, antwortFokus: true };
        }
        if (a.type === 'neu_laden') return { neuLaden: true, antwortFokus: true };
    }
    return null;
}
// Ergebnis: true = erledigt — die Seite wechselt, das Ziel ist im Arbeitsfenster offen, oder der Weg steht unter der Antwort und ist
// angesagt. fokussieren: vorher mit inkluagentFokusImChat() bestimmt.
async function inkluagentWechseln(ziel, antwortEl, fokussieren) {
    const input = document.getElementById('inkluagentInput');
    // Angefangenes zaehlt: Text im Feld und angehaengte Dateien (die gingen beim Seitenwechsel verloren — Dateien lassen sich nicht merken)
    const frei = !!fokussieren && !(input && input.value.trim()) && !(window._inkluagentAnhaenge || []).length;
    if (frei && ziel.fenster) { inkluagentSeiteImFenster(ziel.url, ziel.name); return true; }
    if (frei) {
        if (typeof window.alleAusstehendenSpeichern === 'function') {
            const fehler = await window.alleAusstehendenSpeichern();
            if (fehler) {
                if (typeof window.speicherFehlerMelden === 'function') window.speicherFehlerMelden(fehler);
                inkluagentWechselAnbieten(antwortEl, ziel);
                return true;
            }
        }
        inkluagentWechselAusfuehren(ziel);
        return true;
    }
    inkluagentWechselAnbieten(antwortEl, ziel);
    if (antwortEl) window._inkluagentUngelesen = antwortEl;
    announce(t('Antwort vom InkluAgent ist da. Weil du gerade woanders arbeitest, wechselt die Seite nicht von selbst — der Weg dorthin steht unter der Antwort.'));
    return true;
}
function inkluagentWechselAusfuehren(ziel) {
    if (ziel.antwortFokus) inkluagentFokusAntwortMerken();
    if (ziel.neuLaden) window.location.reload();
    else window.location.href = ziel.url;
}
// Merker fuer die naechste Seite: Fokus auf die letzte Antwort statt auf die Ueberschrift 1 (B11; inkluagentBinden liest ihn). EINE Stelle
// fuer den Wechsel selbst, den angebotenen Link „Sprache wechseln: …“ und den Rueckweg „Zurück zu …“ (Runde 2, HN3).
function inkluagentFokusAntwortMerken() {
    try { sessionStorage.setItem('inkluagent.fokusAntwort', '1'); } catch (e) { /* ohne Speicher: kein Fokus-Sprung */ }
}
// Der Weg unter der Antwort, wenn nicht von selbst gewechselt wird — nur, wenn er nicht schon dasteht („Projekt … öffnen“, „Hinführen: …“)
function inkluagentWechselAnbieten(antwortEl, ziel) {
    if (!antwortEl) return;
    if (ziel.link && Array.from(antwortEl.querySelectorAll('a[href]')).some(x => x.getAttribute('href') === ziel.link)) return;
    let box = antwortEl.querySelector('.inkluagent-message-anhang');
    if (!box) {
        box = document.createElement('div');
        box.className = 'inkluagent-message-anhang';
        antwortEl.insertBefore(box, antwortEl.querySelector('.inkluagent-message-tools'));
    }
    if (ziel.neuLaden) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn-secondary';
        b.textContent = t('Ansicht jetzt wechseln');
        b.addEventListener('click', async () => {
            if (typeof window.alleAusstehendenSpeichern === 'function') {
                const fehler = await window.alleAusstehendenSpeichern();
                if (fehler && typeof window.speicherFehlerMelden === 'function') { window.speicherFehlerMelden(fehler); return; }
            }
            inkluagentWechselAusfuehren(ziel);
        });
        box.appendChild(b);
        return;
    }
    const l = document.createElement('a');
    l.className = 'btn btn-secondary';
    l.href = ziel.link || ziel.url;
    if (ziel.sprache) {
        // Der Name der Sprache in ihr selbst („English“, „français“), ausgezeichnet mit ihrer Sprache (WCAG 3.1.2)
        let name = ziel.sprache;
        try { name = new Intl.DisplayNames([ziel.sprache], { type: 'language' }).of(ziel.sprache) || name; } catch (e) { /* nur das Kuerzel */ }
        l.appendChild(document.createTextNode(t('Sprache wechseln:') + ' '));
        const sp = document.createElement('span');
        sp.lang = ziel.sprache;
        sp.textContent = name;
        l.appendChild(sp);
        l.addEventListener('click', inkluagentFokusAntwortMerken);
    } else {
        l.textContent = ziel.text || l.getAttribute('href');
    }
    box.appendChild(l);
}

// Schritt 4: der Agent hat das Projekt geloescht, auf dessen Seite man ist — der Hauptbereich sagt das und verlinkt die Uebersicht.
// Kein automatischer Seitenwechsel (der Fokus bleibt auf dem Knopf der Karte, die Antwort ist noch zu hoeren); der Agent bleibt.
// Pruefung 13.10.2026 (B1): der Satz mit dem Link bleibt auch in der Agentenansicht sichtbar (data-agentenansicht-bleibt) — vorher
// blendete der Beobachter ihn sofort wieder aus.
function inkluagentProjektWeg(projectId) {
    if (window._inkluagentProjectId == null || String(projectId) !== String(window._inkluagentProjectId)) return;
    const main = document.getElementById('main');
    if (!main) return;
    main.textContent = '';
    const h = document.createElement('h1');
    h.textContent = t('Projekt gelöscht');
    const p = document.createElement('p');
    p.textContent = t('Das Projekt ist gelöscht.') + ' ';
    p.setAttribute('data-agentenansicht-bleibt', '');
    const a = document.createElement('a');
    a.href = '/dashboard';
    a.textContent = t('Zur Übersicht');
    p.appendChild(a);
    main.appendChild(h);
    main.appendChild(p);
    window._inkluagentProjectId = null;
    announce(t('Das Projekt ist gelöscht.'));
}
