// InkluAgent — Nachrichten bauen: Anhänge (Karten, Knöpfe, Links), Blöcke, „Anhören“.
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

// Meine Ausgaben (11.09.2026): Download-Knopf + Link ins Regal unter einer Bot-Antwort, wenn der Bot
// umgewandelt oder exportiert hat (anhang aus dem Werkzeug-Ergebnis, im Verlauf mitgespeichert).
function inkluagentAnhangEl(anhang) {
    const box = document.createElement('div');
    box.className = 'inkluagent-message-anhang';   // Anordnung in style.css (seit 13.10.2026; vorher als style-Attribut)
    (anhang || []).forEach(a => {
        if (a && a.art === 'nachweis') {   // „Fertig mit Nachweis“ (17.10.2026)
            box.appendChild(inkluagentNachweisKarte(a));
            return;
        }
        if (a && a.art === 'formular' && a.karten_id) {   // Formular-Karte (Paket 1)
            box.appendChild(inkluagentFormularKarte(a));
            return;
        }
        if (a && a.art === 'bestaetigung' && a.angebot_id) {
            const offen = !a.zustand || a.zustand === 'offen';
            const karte = document.createElement('div');
            karte.className = 'inkluagent-karte inkluagent-bestaetigung';
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
                b.addEventListener('click', () => inkluagentBestaetigen(a.angebot_id, b, st, !!a.konto));
                karte.appendChild(b); karte.appendChild(st);
            }
            box.appendChild(karte);
            return;
        }
        // Link unter der Antwort (Schritt 3, projekt_oeffnen): nur Pfade dieser Anwendung (eigenerPfad, Pruefung 13.10.2026, S14 —
        // vorher liess die Pruefung „/\host“ durch). Ins Arbeitsfenster oeffnet ihn, wo passend, der Klick-Handler der Seite (B1).
        // lang (B11): ein Link in einer anderen Sprache als die Seite („Zurück zu Deutsch“ nach einem Sprachwechsel) traegt sie.
        // Runde 2 (HN3): ein Sprachwechsel-Link („Zurück zu …“) setzt den Merker wie „Sprache wechseln: …“ — nach dem Neuladen steht
        // der Fokus wieder auf der Antwort, nicht auf der Ueberschrift 1
        if (a && a.art === 'link' && eigenerPfad(a.url)) {
            const l = document.createElement('a');
            l.className = 'btn btn-secondary';
            l.href = a.url;
            l.textContent = a.text || a.url;
            inkluagentSpracheSetzen(l, a.lang);
            if (/^\/set-language\//.test(a.url)) l.addEventListener('click', inkluagentFokusAntwortMerken);
            box.appendChild(l);
            return;
        }
        if (!a || !eigenerPfad(a.download_url)) return;
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
        if (eigenerPfad(a.ausgaben_url)) {
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

// Ausbau Runde 2, Schritt 6 (Schalter AGENT_BLOECKE): die Antwort als Bloecke — Leerzeile = neuer Absatz (p), Zeilen mit „• “ =
// Liste (ul), mindestens zwei Zeilen „1. “, „2. “ … = nummerierte Liste (ol). Nur createElement und textContent, kein innerHTML.
// Aeltere Antworten (Aufzaehlungszeichen damals entfernt) werden zu Absaetzen.
function inkluagentBloeckeBauen(el, text) {
    String(text).split(/\n[ \t]*\n/).forEach(absatz => {
        const gruppen = [];   // [{art: 'p'|'ul'|'ol', zeilen: [...], start}]
        absatz.split('\n').forEach(z => {
            const punkt = /^\s*•\s+(.*)$/.exec(z);
            const nr = /^\s*(\d+)[.)]\s+(.*)$/.exec(z);
            const art = punkt ? 'ul' : (nr ? 'ol' : 'p');
            const inhalt = punkt ? punkt[1] : (nr ? nr[2] : z);
            if (!inhalt.trim()) return;
            const letzte = gruppen[gruppen.length - 1];
            if (letzte && letzte.art === art) letzte.zeilen.push(inhalt);
            else gruppen.push({ art: art, zeilen: [inhalt], start: nr ? parseInt(nr[1], 10) : 1, roh: [] });
            gruppen[gruppen.length - 1].roh.push(z.trim());
        });
        gruppen.forEach(g => {
            // Eine einzelne „1. …“-Zeile ist meist ein Datum oder ein Satzanfang, keine Liste
            if (g.art === 'ol' && g.zeilen.length < 2) { g.art = 'p'; g.zeilen = g.roh; }
            if (g.art === 'p') {
                const p = document.createElement('p');
                p.textContent = g.zeilen.join('\n');
                el.appendChild(p);
                return;
            }
            const liste = document.createElement(g.art);
            if (g.art === 'ol' && g.start !== 1) liste.start = g.start;
            g.zeilen.forEach(z => { const li = document.createElement('li'); li.textContent = z; liste.appendChild(li); });
            el.appendChild(liste);
        });
    });
}

// Pruefung 13.10.2026, Runde 2 (B11, Punkt 4; WCAG 3.1.2): die Sprache, in der eine Nachricht geschrieben wurde (der Server liefert sie
// je Verlaufseintrag als „sprache“), steht als lang am Nachrichtentext. Sonst las VoiceOver nach einem Wechsel auf Englisch die deutsche
// Antwort, auf der der Fokus steht, mit englischer Stimme. Nur am Text: Absender-Ueberschrift („Du“), „Genutzt: …“ und die Knoepfe
// darunter baut die Seite in ihrer jetzigen Sprache. Ohne gueltige Angabe gilt die Sprache der Seite (neue Nachrichten).
function inkluagentSpracheSetzen(el, sprache) {
    if (el && typeof sprache === 'string' && /^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$/.test(sprache)) el.lang = sprache;
}
function inkluagentAppendMessage(role, content, werkzeuge, anhang, sprache) {
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
    inkluagentSpracheSetzen(contentEl, sprache);
    if (role === 'assistant' && window.INKLUAGENT && window.INKLUAGENT.bloecke) inkluagentBloeckeBauen(contentEl, content || '');
    else contentEl.textContent = content || '';
    wrapper.appendChild(roleEl);
    if (role === 'assistant') inkluagentAnhoerenKnopf(roleEl, contentEl);
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

// Anhören (16.10.2026, Steve: „wie bei InkluEdit auch im Chat einen Button zum Antwortvorlesen“): unter der Überschrift jeder
// Antwort ein Knopf „Anhören“, wie in InkluEdit seit 15.10. Start und Stopp über denselben Knopf (aria-pressed), Escape hält an,
// immer nur eine Stimme — auch zusammen mit „Anhören“ beim Alt-Text und den Hörproben (gemeinsame Klasse tts-btn, app.html
// vorlesenStopp setzt auch diesen Knopf zurück). Wie dort nur Stimmen, die AUF DEM GERÄT laufen (voice.localService): Chrome bietet
// auch Google-Stimmen an, die den Text übers Netz schicken, die nehmen wir aus Datenschutzgründen nie. Gelesen wird nur der Text der
// Antwort (nicht Karte, Download oder „Genutzt: …“), in ihrer Sprache (lang am Inhalt), satzweise (Chrome bricht lange Äußerungen
// ab). Ohne Sprachausgabe im Browser gibt es keinen Knopf.
// window._inkluagentAnhoerenLauf: jeder Start zaehlt hoch; spaete onend/onerror alter Aeusserungen laufen dann ins Leere (auf
// oberster Ebene stehen nur Funktionen, test_inkluagent_js)
function inkluagentAnhoerenStopp() {
    window._inkluagentAnhoerenLauf = (window._inkluagentAnhoerenLauf || 0) + 1;
    try { if (window.speechSynthesis) window.speechSynthesis.cancel(); } catch (e) { /* nichts zu stoppen */ }
    document.querySelectorAll('.tts-btn[aria-pressed="true"]').forEach(b => {
        b.setAttribute('aria-pressed', 'false');
        b.textContent = b.dataset.label || t('Anhören');
    });
}
function inkluagentLokaleStimmen() {
    return new Promise(fertig => {
        const synth = window.speechSynthesis;
        const jetzt = (synth && synth.getVoices()) || [];
        if (jetzt.length || !synth) { fertig(jetzt); return; }
        let erledigt = false;
        const ende = () => { if (!erledigt) { erledigt = true; fertig(synth.getVoices() || []); } };
        synth.addEventListener('voiceschanged', ende, { once: true });
        setTimeout(ende, 1200);
    });
}
// „Fertig mit Nachweis“ (Steve 17.10.2026, inkluagent/nachweis.py): am Ende zeigt der Agent, dass das Ergebnis geprueft ist —
// PDF/UA nach veraPDF, Alt-Texte und der Anfang der Hoerprobe, so wie ein Screenreader das Dokument vorliest, mit „Anhören“ (Stimmen
// auf dem Geraet, inkluagentAnhoerenKnopf). Das Urteil steht immer im Text, nie nur in einer Farbe. Zeilen der Hoerprobe stammen aus
// dem Dokument: nur textContent; die Sprache des Dokuments (lang) waehlt die Stimme. Links nur auf eigene Pfade.
function inkluagentNachweisKarte(a) {
    const karte = document.createElement('div');
    karte.className = 'inkluagent-karte inkluagent-nachweis';
    const h = document.createElement('h4');
    h.textContent = a.dokument ? t('Nachweis: {dokument}', { dokument: String(a.dokument) }) : t('Nachweis');
    karte.appendChild(h);
    const punkte = [];
    const n = Number(a.befunde) || 0;
    if (a.bestanden === true) punkte.push(t('PDF/UA-Prüfung (veraPDF): bestanden'));
    else if (a.bestanden === false) punkte.push(n > 1 ? t('PDF/UA-Prüfung (veraPDF): nicht bestanden, {n} Befunde', { n: n })
        : (n === 1 ? t('PDF/UA-Prüfung (veraPDF): nicht bestanden, 1 Befund') : t('PDF/UA-Prüfung (veraPDF): nicht bestanden')));
    else punkte.push(t('PDF/UA-Prüfung (veraPDF): nicht geprüft'));
    if (Number(a.bilder) > 0 && a.alt_texte != null) punkte.push(t('Alt-Texte: {n} von {m} Bildern', { n: Number(a.alt_texte) || 0, m: Number(a.bilder) }));
    if (a.aktuell === false) punkte.push(t('Die Prüfdatei ist älter als der jetzige Stand.'));
    const ul = document.createElement('ul');
    ul.className = 'inkluagent-karte-zeilen';
    punkte.forEach(p => { const li = document.createElement('li'); li.textContent = p; ul.appendChild(li); });
    karte.appendChild(ul);
    const zeilen = Array.isArray(a.hoerprobe) ? a.hoerprobe.filter(z => typeof z === 'string' && z.trim()) : [];
    if (zeilen.length) {
        const titel = document.createElement('p');
        titel.className = 'inkluagent-nachweis-titel';
        titel.textContent = a.hoerprobe_dokument ? t('So liest ein Screenreader den Anfang von „{dokument}“ vor:', { dokument: String(a.hoerprobe_dokument) })
            : t('So liest ein Screenreader den Anfang vor:');
        const hp = document.createElement('div');
        hp.className = 'inkluagent-nachweis-hoerprobe';
        if (/^[a-z]{2}$/.test(a.lang || '')) hp.setAttribute('lang', a.lang);
        zeilen.forEach(z => { const p = document.createElement('p'); p.textContent = z; hp.appendChild(p); });
        karte.appendChild(titel);
        karte.appendChild(hp);
        inkluagentAnhoerenKnopf(titel, hp);
    }
    const links = [[a.ablage_url, t('Prüfbericht in der Ablage')], [a.pruefung_url, t('Zur Barrierefreiheitsprüfung')]];
    links.forEach(([url, text]) => {
        if (!url || !eigenerPfad(url)) return;
        const l = document.createElement('a');
        l.className = 'btn btn-secondary';
        l.href = url;
        l.textContent = text;
        karte.appendChild(l);
    });
    return karte;
}
function inkluagentAnhoerenKnopf(roleEl, contentEl) {
    if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) return;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn btn-secondary btn-small tts-btn inkluagent-anhoeren';
    b.textContent = t('Anhören');
    b.dataset.label = t('Anhören');
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => inkluagentAnhoeren(b, contentEl));
    roleEl.insertAdjacentElement('afterend', b);
    if (!window._inkluagentAnhoerenEsc) {
        window._inkluagentAnhoerenEsc = true;
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && document.querySelector('.inkluagent-anhoeren[aria-pressed="true"]')) inkluagentAnhoerenStopp();
        });
    }
}
async function inkluagentAnhoeren(b, contentEl) {
    if (b.getAttribute('aria-pressed') === 'true') { inkluagentAnhoerenStopp(); return; }
    inkluagentAnhoerenStopp();
    const lauf = window._inkluagentAnhoerenLauf;
    // Saetze trennen ohne Lookbehind (Safari < 16.4 kennt ihn nicht, wie in app.html)
    const teile = String(contentEl.innerText || contentEl.textContent || '').replace(/([.!?:])\s+/g, '$1\n').split(/\n+/)
        .map(x => x.trim()).filter(Boolean);
    if (!teile.length) { announce(t('Kein Text zum Vorlesen vorhanden.')); return; }
    const zurueck = () => {
        if (lauf !== window._inkluagentAnhoerenLauf) return;
        b.setAttribute('aria-pressed', 'false');
        b.textContent = t('Anhören');
    };
    // Zustand VOR dem Warten auf die Stimmen: ein zweiter Druck in der Wartezeit stoppt, statt doppelt zu starten
    b.setAttribute('aria-pressed', 'true');
    b.textContent = t('Stopp');
    const traeger = contentEl.closest('[lang]');
    const kennung = String((traeger && traeger.getAttribute('lang')) || window.LANG || 'de').slice(0, 2).toLowerCase();
    const stimme = (await inkluagentLokaleStimmen()).find(v => v.localService && v.lang && v.lang.toLowerCase().indexOf(kennung) === 0);
    if (lauf !== window._inkluagentAnhoerenLauf) return;   // inzwischen gestoppt oder neu gestartet
    if (!stimme) {
        zurueck();
        announce(t('Auf diesem Gerät ist keine Stimme für diese Sprache installiert. Aus Datenschutzgründen lesen wir nicht über einen Sprachdienst im Internet vor.'));
        return;
    }
    teile.forEach((teil, i) => {
        const u = new SpeechSynthesisUtterance(teil);
        u.voice = stimme;
        u.lang = stimme.lang;
        if (i === teile.length - 1) u.onend = zurueck;
        u.onerror = zurueck;
        window.speechSynthesis.speak(u);
    });
}
