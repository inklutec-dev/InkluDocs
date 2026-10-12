// InkluAgent: Chat-Bereich im Projekt (Aufbau, Verlauf, Senden, Bestaetigungs-Karten, Fokus- und Ansageregeln).
//
// Ausgelagert aus backend/templates/app.html (09.10.2026, Konzept InkluAgent, Schritt 1) — der Code ist unveraendert,
// nur 4 Leerzeichen weniger eingerueckt (Beleg im Commit: Pruefsumme des Blocks vor und nach dem Umzug gleich).
// Geladen von app.html VOR dem Seitenskript; diese Datei definiert nur Funktionen (und Merker) und fuehrt beim Laden nichts
// aus — ausser inkluagentBereichAufbauen() am Dateiende, das ohne festen Bereich sofort zurueckkehrt (Schritt 3). Sie nutzt die globalen Helfer der Seite: t() (Texte aus window.I18N, 6 Sprachen), announce(),
// showProject(), window.WERKZEUG_NAMEN, window.Formular. Aufrufer: app.html (inkluagentSectionHtml/inkluagentInit),
// dokument.js, formular.js, uebersetzen.js, abschluss.js.
// Ausbau Runde 2, Schritt 3 (Schalter AGENT_UEBERALL, window.INKLUAGENT.ueberall): base_app.html laedt die Datei auf JEDER
// Seite und hat einen festen Bereich <aside id="inkluagentBereich"> nach dem Hauptinhalt. Dann baut inkluagentBereichAufbauen()
// den Agenten einmal dort auf; die Ansichten eines Projekts betten nichts mehr ein (inkluagentSectionHtml liefert '') und binden
// den Bereich mit inkluagentInit an ihr Projekt. Seiten ohne Projekt nutzen das Gespraech des Kontos mit Konto-Werkzeugen.

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
    // Schritt 3: der Agent steht im festen Bereich — die Ansicht bettet nichts ein, merkt sich nur ihre Variante fuer inkluagentInit
    if (inkluagentUeberall()) { window._inkluagentVariante = variante || 'bilder'; return ''; }
    return inkluagentInhaltHtml(projectId, variante, false);
}
function inkluagentUeberall() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.ueberall && document.getElementById('inkluagentBereich'));
}
// Einleitung und Platzhalter je Variante: 'formular' (Felder), 'konto' (Seite ohne Projekt, Schritt 3), sonst Bilder
function inkluagentEinleitung(variante) {
    if (variante === 'formular') return t('Der InkluAgent ist ein KI-Assistent. Bitte ihn um Hilfe — z.B. eine Quickinfo für ein Feld generieren, einen Text kürzer oder einheitlich formulieren, in den Stammdaten nachsehen oder die Anmerkungen des Gastes zusammenfassen. Felder per Nummer benennen (z.B. <em>Feld 3</em>, <em>Felder 1-5</em>).');
    if (variante === 'konto') return t('Der InkluAgent ist ein KI-Assistent. Frag ihn nach deinen Projekten, deinem Guthaben oder dem, was gerade läuft. In einem Projekt hilft er bei Alt-Texten, Tagging, Prüfung und mehr.');
    return t('Der InkluAgent ist ein KI-Assistent. Bitte ihn um Hilfe — z.B. einen Alt-Text für ein Bild generieren, einen vorhandenen Text in leichter Sprache umformulieren oder in eine andere Sprache übersetzen. Bilder per Nummer benennen (z.B. <em>Bild 3</em>, <em>Bilder 1-5</em>). Maximal 10 Bilder pro Anfrage.');
}
function inkluagentPlatzhalter(variante) {   // Michael Karbe, Feedback 20261012 - 1, Punkt 5: ueberall derselbe Satz
    return t('Was soll der InkluAgent für dich erledigen?');
}
// alsBereich (Schritt 3): Inhalt des festen Bereichs — ein div statt section, denn das aside ist schon der Bereich (Landmark)
function inkluagentInhaltHtml(projectId, variante, alsBereich) {
    const formular = variante === 'formular';
    // Aufbau seit Michael Karbe, Feedback 20261012 - 1: aufgeklappt steht direkt unter dem Knopf das Eingabefeld (Punkt 1), darunter
    // der Verlauf, dann „Neues Gespräch“/„Frühere Gespräche“, eine Linie und die Beschreibung (Punkt 4). Zugeklappt sieht man nur
    // die Ueberschrift mit dem Knopf. Der KI-Hinweis (KI-VO Art. 50) steht in der Beschreibung, also immer, wenn man ihn benutzt.
    return ''
        + (alsBereich
            ? '<div class="inkluagent-section" data-variante="' + (variante || 'bilder') + '">'
            : '<section class="inkluagent-section" aria-labelledby="inkluagentToggle" data-project-id="' + projectId + '" data-variante="' + (formular ? 'formular' : 'bilder') + '">')
        // Akkordeon-Muster (WAI-ARIA, Steve 28.08.2026): der Auf/Zu-Knopf steht IN der Ueberschrift
        // Ebene 2 — per Ueberschriften-Navigation anspringen und sofort druecken; die Nachrichten
        // sind Ebene 3 darunter. Ein Wort fuer beides: „InkluAgent“ (09.10.2026, vorher „Chatbot“) — derselbe
        // Name wie der Absender der Nachrichten (Ueberschrift 3) und in der Einleitung.
        +   '<h2 id="inkluagentHeading" class="section-title" style="margin:0 0 0.4rem 0;">'
        +     '<button id="inkluagentToggle" type="button" aria-expanded="false" aria-controls="inkluagentPanel" class="inkluagent-toggle">'
        +       t('InkluAgent')
        +     '</button>'
        +   '</h2>'
        +   '<div id="inkluagentPanel" class="inkluagent-panel" hidden>'
        // novalidate (Runde 2, Schritt 1): Meldungen kommen aus inkluagentSend in die Statuszeile — nie die Blase des Browsers, die
        // ohne Ansage verschwindet und das Senden still verhindert
        +     '<form id="inkluagentForm" class="inkluagent-form" novalidate>'
        +       '<label for="inkluagentInput" class="visually-hidden">' + t('Nachricht an den InkluAgent') + '</label>'
        +       '<textarea id="inkluagentInput" rows="2" maxlength="5000" '
        +         'placeholder="' + inkluagentPlatzhalter(variante) + '" '
        +         'aria-describedby="inkluagentHint"></textarea>'
        +       '<div id="inkluagentHint" class="inkluagent-hint">' + t('Enter sendet, Shift+Enter macht einen Zeilenumbruch. Maximal 5000 Zeichen.') + '</div>'
        +       (inkluagentHochladenAn() ? inkluagentHochladeHtml() : '')
        +       '<div class="inkluagent-controls">'
        +         '<button type="submit" id="inkluagentSendBtn" class="btn btn-primary">' + t('Senden') + '</button>'
        +         '<span id="inkluagentStatus" class="inkluagent-status" role="status" aria-live="polite"></span>'
        +       '</div>'
        +     '</form>'
        // tabindex=0 (11.09.2026, axe scrollable-region-focusable): der Verlauf wird bei vielen
        // Nachrichten scrollbar und muss dann per Tastatur erreichbar sein (WCAG 2.1.1).
        // aria-live="off" (Pruefung 3 Barrierefreiheit, H1/N1): role=log ist sonst von selbst „polite“ — beim Neuzeichnen
        // (Ansichtswechsel, Neuladen, Auf/Zu) wurde der ganze Verlauf erneut angesagt, jede neue Antwort doppelt (Live-Region
        // UND Fokus) und die eigene Nachricht als Echo. Jetzt sagt nichts im Verlauf von selbst etwas an: die neue Antwort
        // bekommt den Fokus (der Screenreader liest sie vollständig), Warten und Fehler meldet die Statuszeile.
        +     '<div id="inkluagentLog" class="inkluagent-log" role="log" aria-live="off" aria-label="' + t('Chat-Verlauf') + '" tabindex="0"></div>'
        +     (inkluagentGespraecheAn() ? inkluagentGespraechLeisteHtml() : '')
        +     '<hr class="inkluagent-trenner">'
        +     '<div class="inkluagent-beschreibung">'
        // InkluAgent-Ausbau Runde 1, Schritt 3 (Schalter funktionen.AGENT_HILFE): KI-Hinweis, Beispiele und Weg zur Hilfe-Seite;
        // Schalter aus = die Einleitung wie bisher. KI-VO Art. 50 Abs. 1 (gilt seit 02.08.2026): der ausdrueckliche Satz „… ist
        // ein KI-Assistent“ bzw. „… ist eine KI“ steht in beiden. (25.08.2026: Jinja-Kommentare im raw-Block brachen das ganze
        // App-Skript — hier nur JS-Kommentare.)
        +       (inkluagentKurzhilfe() ? inkluagentKurzhilfeHtml(variante) : ('<p class="inkluagent-intro">' + inkluagentEinleitung(variante) + '</p>'))
        +     '</div>'
        +   '</div>'
        + (alsBereich ? '</div>' : '</section>');
}

// Runde 2, Schritt 1 (Schalter funktionen.AGENT_HOCHLADEN, window.INKLUAGENT aus base_app.html): Dateien gehen mit der
// Nachricht an den Agenten (Webadressen stehen seit 12.10.2026 in der Nachricht selbst). Natives Feld mit sichtbarem Label (Datei-Auswahl); die Zuordnung zu
// Projekten macht der Server (/api/agent/hochladen), das Ergebnis steht als Liste unter dem Feld.
function inkluagentHochladenAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.hochladen);
}
function inkluagentHochladeHtml() {
    const k = window.INKLUAGENT || {};
    return ''
        + '<div class="inkluagent-hochladen">'
        +   '<label for="inkluagentDateien">' + t('Dateien mitschicken') + '</label>'
        +   '<input type="file" id="inkluagentDateien" multiple aria-describedby="inkluagentHochladenHint" '
        +     'accept=".pdf,.docx,.jpg,.jpeg,.png,.gif,.webp,.heic,.heif,.bmp,.tif,.tiff,.svg">'
        // Kein eigenes Feld fuer Webadressen (Steve 12.10.2026): eine Adresse steht einfach in der Nachricht, der Server erkennt sie
        // und der Agent fragt mit Knoepfen, was damit passieren soll (hochladen.webadressen_karten)
        +   '<p id="inkluagentHochladenHint" class="inkluagent-hint">'
        +     t('PDF, Word (.docx) und Bilder, höchstens {n} Dateien, je bis {mb} MB. Webadressen schreibst du einfach in deine Nachricht. Der InkluAgent legt alles ins passende Projekt und nennt dir die Preise, bevor etwas startet.',
                { n: k.max_dateien || 10, mb: k.max_mb || 50 })
        +   '</p>'
        +   '<ul id="inkluagentHochgeladen" class="inkluagent-hochgeladen" aria-label="' + t('Ergebnis des Hochladens') + '" hidden></ul>'
        + '</div>';
}
// Ergebnisliste unter dem Feld (nur Text, kein innerHTML). Bleibt ueber ein Neuzeichnen der Ansicht hinweg stehen
// (inkluagentInitLaden zeichnet sie fuer dasselbe Projekt wieder), bis zum naechsten Hochladen.
function inkluagentHochgeladenZeigen(projectId, ergebnisse) {
    window._inkluagentHochgeladen = { projectId: String(projectId), ergebnisse: ergebnisse || [] };
    const ul = document.getElementById('inkluagentHochgeladen');
    if (!ul) return;
    ul.textContent = '';
    (ergebnisse || []).forEach(e => {
        const li = document.createElement('li');
        li.dataset.ok = e.ok ? '1' : '0';
        const name = e.name || '';
        if (!e.ok) {
            li.textContent = t('{name}: nicht hochgeladen. {grund}', { name: name, grund: e.fehler || '' });
        } else if (e.wartet) {
            // Zuordnung (Steve 12.10.2026): es gibt schon ein aehnliches Projekt — die Datei wartet auf den Knopf unter der Antwort
            li.textContent = t('{name}: wartet auf deine Entscheidung. Die Knöpfe stehen unter der Antwort des InkluAgent.', { name: name });
        } else if (!e.neues_projekt) {
            li.textContent = t('{name}: hochgeladen in dieses Projekt.', { name: name });
        } else {
            li.textContent = (e.eigenes_projekt_weil ? t('{name}: passt nicht in dieses Projekt, darum hochgeladen in das neue Projekt „{projekt}“.', { name: name, projekt: e.projekt || '' })
                                                     : t('{name}: hochgeladen in das neue Projekt „{projekt}“.', { name: name, projekt: e.projekt || '' })) + ' ';
            const a = document.createElement('a');
            a.href = '/app?projekt=' + encodeURIComponent(e.project_id);
            a.textContent = t('Projekt „{projekt}“ öffnen', { projekt: e.projekt || '' });
            li.appendChild(a);
        }
        ul.appendChild(li);
    });
    ul.hidden = !ul.children.length;
}
// Dateien und Webadresse hochladen. Ergebnis: das Server-Ergebnis oder null (Fehler steht dann in der Statuszeile).
async function inkluagentHochladen(projectId, dateien, webadresse) {
    const k = window.INKLUAGENT || {};
    const max = k.max_dateien || 10;
    if (dateien.length > max) {
        inkluagentSetStatus(t('Höchstens {n} Dateien auf einmal. Bitte teile die Auswahl auf.', { n: max }), true);
        return null;
    }
    const fd = new FormData();
    dateien.forEach(f => fd.append('dateien', f, f.name));
    fd.append('webadresse', webadresse || '');
    if (projectId != null) fd.append('project_id', String(projectId));
    inkluagentSetStatus(dateien.length ? t('Wird hochgeladen …') : t('Webadresse wird geladen …'), false);
    let res;
    try {
        res = await fetch('/api/agent/hochladen', { method: 'POST', body: fd });
    } catch (e) {
        inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true);
        return null;
    }
    if (res.status === 401) { window.location.href = '/login'; return null; }
    const erg = await res.json().catch(() => ({}));
    if (!res.ok) {
        inkluagentSetStatus((erg && typeof erg.detail === 'string') ? erg.detail : t('Hochladen nicht möglich.'), true);
        return null;
    }
    return erg;
}
// Eine gebuendelte Ansage nach dem Hochladen (Zahlen; die Gruende stehen in der Liste)
function inkluagentHochladenSatz(ergebnisse) {
    const alle = (ergebnisse || []).length;
    const gut = (ergebnisse || []).filter(e => e.ok && !e.wartet).length;
    const wartend = (ergebnisse || []).filter(e => e.wartet).length;
    const schlecht = alle - gut - wartend;
    return t('{gut} von {alle} hochgeladen.', { gut: gut, alle: alle })
        + (wartend ? ' ' + (wartend === 1 ? t('1 wartet auf deine Entscheidung.') : t('{n} warten auf deine Entscheidung.', { n: wartend })) : '')
        + (schlecht ? ' ' + t('Die Gründe stehen unter dem Feld.') : '');
}

// ─── Dateien anhängen in der Agentenansicht (Steve 12.10.2026) ───────────────────────────────────────────────────────────────
// Wie die Büroklammer bei ChatGPT: Knopf „Dateien anhängen“ vor „Senden“ (öffnet die normale Dateiauswahl); Dateien lassen sich
// auch ins Gespräch ziehen oder ins Eingabefeld einfügen. Die Auswahl steht als Liste mit „Entfernen“ unter den Knöpfen und geht
// mit der nächsten Nachricht denselben Weg wie das Feld „Dateien mitschicken“ (POST /api/agent/hochladen: Zuordnung, Bremsen,
// Preise). Nur im Chatstil und nur, wenn das Feld selbst aus ist (Michael Karbe, Punkt 3: in der manuellen Ansicht kein Hochladen
// im Agenten). Meldungen: sichtbar unter der Liste und einmal angesagt — nie über die Statuszeile, die gehört dem laufenden Auftrag.
function inkluagentAnhaengenAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.anhaengen) && inkluagentArbeitsfensterAn() && !inkluagentHochladenAn();
}
function inkluagentAnhangAktiv() {
    return !!document.getElementById('inkluagentAnhaengenBtn');
}
function inkluagentAnhangBauen(form, gespraech) {
    if (!inkluagentAnhaengenAn() || !form || inkluagentAnhangAktiv()) return;
    const k = window.INKLUAGENT || {};
    if (!Array.isArray(window._inkluagentAnhaenge)) window._inkluagentAnhaenge = [];
    const datei = document.createElement('input');
    datei.type = 'file';
    datei.id = 'inkluagentAnhangDatei';
    datei.multiple = true;
    datei.hidden = true;
    datei.tabIndex = -1;
    datei.accept = '.pdf,.docx,.jpg,.jpeg,.png,.gif,.webp,.heic,.heif,.bmp,.tif,.tiff,.svg';
    datei.addEventListener('change', () => { inkluagentAnhaengeHinzu(Array.from(datei.files || [])); datei.value = ''; });
    const knopf = document.createElement('button');
    knopf.type = 'button';
    knopf.id = 'inkluagentAnhaengenBtn';
    knopf.className = 'btn btn-secondary';
    knopf.textContent = t('Dateien anhängen');
    knopf.setAttribute('aria-describedby', 'inkluagentAnhangHinweis');
    knopf.addEventListener('click', () => datei.click());
    const hinweis = document.createElement('span');
    hinweis.id = 'inkluagentAnhangHinweis';
    hinweis.className = 'visually-hidden';
    hinweis.textContent = t('PDF, Word (.docx) und Bilder, höchstens {n} Dateien, je bis {mb} MB. Du kannst Dateien auch ins Gespräch ziehen oder einfügen.',
                            { n: k.max_dateien || 10, mb: k.max_mb || 50 });
    const liste = document.createElement('ul');
    liste.id = 'inkluagentAnhaenge';
    liste.className = 'inkluagent-anhaenge';
    liste.setAttribute('aria-label', t('Angehängte Dateien'));
    liste.hidden = true;
    const meldung = document.createElement('p');
    meldung.id = 'inkluagentAnhangMeldung';
    meldung.className = 'inkluagent-hint inkluagent-anhang-meldung';
    meldung.hidden = true;
    // Ergebnis des Hochladens wie beim Feld (inkluagentHochgeladenZeigen fuellt es)
    const ergebnis = document.createElement('ul');
    ergebnis.id = 'inkluagentHochgeladen';
    ergebnis.className = 'inkluagent-hochgeladen';
    ergebnis.setAttribute('aria-label', t('Ergebnis des Hochladens'));
    ergebnis.hidden = true;
    const send = document.getElementById('inkluagentSendBtn');
    if (send && send.parentNode) send.parentNode.insertBefore(knopf, send);   // erst „Dateien anhängen“, dann „Senden“
    else form.appendChild(knopf);
    [datei, hinweis, liste, meldung, ergebnis].forEach(el => form.appendChild(el));
    // Hineinziehen (Maus): das ganze Gespraech nimmt Dateien an
    if (gespraech && !gespraech.dataset.ablage) {
        gespraech.dataset.ablage = '1';
        const mitDateien = e => Array.from((e.dataTransfer && e.dataTransfer.types) || []).indexOf('Files') >= 0;
        gespraech.addEventListener('dragover', e => {
            if (!inkluagentAnhangAktiv() || !mitDateien(e)) return;
            e.preventDefault();
            gespraech.classList.add('inkluagent-ablage-aktiv');
        });
        gespraech.addEventListener('dragleave', e => {
            if (!gespraech.contains(e.relatedTarget)) gespraech.classList.remove('inkluagent-ablage-aktiv');
        });
        gespraech.addEventListener('drop', e => {
            gespraech.classList.remove('inkluagent-ablage-aktiv');
            if (!inkluagentAnhangAktiv() || !mitDateien(e)) return;
            e.preventDefault();
            inkluagentAnhaengeHinzu(Array.from(e.dataTransfer.files || []));
        });
    }
    // Einfuegen (Befehl-V) im Eingabefeld: Dateien aus der Zwischenablage anhaengen, Text bleibt Text
    const input = document.getElementById('inkluagentInput');
    if (input && !input.dataset.einfuegen) {
        input.dataset.einfuegen = '1';
        input.addEventListener('paste', e => {
            if (!inkluagentAnhangAktiv()) return;
            const f = Array.from((e.clipboardData && e.clipboardData.files) || []);
            if (!f.length) return;
            e.preventDefault();
            inkluagentAnhaengeHinzu(f);
        });
    }
    inkluagentAnhaengeZeigen();
}
function inkluagentAnhangEntfernenAlles() {
    ['inkluagentAnhaengenBtn', 'inkluagentAnhangDatei', 'inkluagentAnhangHinweis', 'inkluagentAnhaenge', 'inkluagentAnhangMeldung']
        .forEach(id => { const el = document.getElementById(id); if (el) el.remove(); });
    const erg = document.getElementById('inkluagentHochgeladen');
    if (erg && !erg.closest('.inkluagent-hochladen')) erg.remove();
}
function inkluagentGroesse(bytes) {
    const lang = document.documentElement.lang || 'de';
    if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)).toLocaleString(lang) + ' KB';
    return (bytes / (1024 * 1024)).toLocaleString(lang, { maximumFractionDigits: 1 }) + ' MB';
}
function inkluagentAnhangZahl(n) {
    return n === 0 ? t('Keine Datei angehängt.') : (n === 1 ? t('1 Datei angehängt.') : t('{n} Dateien angehängt.', { n: n }));
}
function inkluagentAnhangMelden(text, fehler) {
    const p = document.getElementById('inkluagentAnhangMeldung');
    if (p) {
        p.textContent = text || '';
        p.hidden = !text;
        if (fehler) p.setAttribute('data-state', 'error'); else p.removeAttribute('data-state');
    }
    if (text) announce(text);
}
function inkluagentAnhaengeHinzu(dateien) {
    if (!dateien || !dateien.length || !inkluagentAnhangAktiv()) return;
    const k = window.INKLUAGENT || {};
    const max = k.max_dateien || 10;
    const maxMb = k.max_mb || 50;
    const liste = window._inkluagentAnhaenge;
    const zuGross = [];
    let zuViele = 0;
    dateien.forEach(f => {
        if (f.size > maxMb * 1024 * 1024) { zuGross.push(f.name); return; }
        if (liste.some(x => x.name === f.name && x.size === f.size && x.lastModified === f.lastModified)) return;   // schon dabei
        if (liste.length >= max) { zuViele++; return; }
        liste.push(f);
    });
    inkluagentAnhaengeZeigen();
    const teile = [];
    if (zuGross.length) teile.push(t('Zu groß, höchstens {mb} MB je Datei: {namen}.', { mb: maxMb, namen: zuGross.join(', ') }));
    if (zuViele) teile.push(t('Höchstens {n} Dateien auf einmal. Bitte teile die Auswahl auf.', { n: max }));
    inkluagentAnhangMelden(teile.concat([inkluagentAnhangZahl(liste.length)]).join(' '), teile.length > 0);
}
function inkluagentAnhaengeZeigen() {
    const ul = document.getElementById('inkluagentAnhaenge');
    if (!ul) return;
    const liste = window._inkluagentAnhaenge || [];
    ul.textContent = '';
    liste.forEach((f, i) => {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.className = 'inkluagent-anhang-name';
        name.textContent = f.name + ' (' + inkluagentGroesse(f.size) + ')';
        const weg = document.createElement('button');
        weg.type = 'button';
        weg.className = 'btn btn-secondary inkluagent-anhang-weg';
        weg.textContent = t('Entfernen');
        weg.setAttribute('aria-label', t('„{name}“ entfernen', { name: f.name }));
        weg.addEventListener('click', () => inkluagentAnhangEntfernen(i));
        li.appendChild(name);
        li.appendChild(weg);
        ul.appendChild(li);
    });
    ul.hidden = !liste.length;
}
function inkluagentAnhangEntfernen(i) {
    const liste = window._inkluagentAnhaenge || [];
    const weg = liste.splice(i, 1)[0];
    inkluagentAnhaengeZeigen();
    // Fokus auf das naechste „Entfernen“, sonst zurueck auf „Dateien anhängen“
    const knoepfe = document.querySelectorAll('#inkluagentAnhaenge .inkluagent-anhang-weg');
    const ziel = knoepfe[Math.min(i, knoepfe.length - 1)] || document.getElementById('inkluagentAnhaengenBtn');
    if (ziel) ziel.focus();
    inkluagentAnhangMelden(t('„{name}“ entfernt.', { name: weg ? weg.name : '' }) + ' ' + inkluagentAnhangZahl(liste.length), false);
}
function inkluagentAnhaengeLeeren() {
    window._inkluagentAnhaenge = [];
    inkluagentAnhaengeZeigen();
    inkluagentAnhangMelden('', false);
}

// Schritt 3 (InkluAgent-Ausbau Runde 1): Kurzhilfe vor dem Chat — nur mit Schalter (window.FUNKTIONEN aus funktionen.py)
function inkluagentKurzhilfe() {
    return !!((window.FUNKTIONEN && window.FUNKTIONEN.agent_hilfe) || (window.INKLUAGENT && window.INKLUAGENT.hilfe));
}
// Steve 09.10.2026: die Hilfe gehoert IN den Bereich des InkluAgent, nicht in die Seitenleiste — der KI-Hinweis (KI-VO Art. 50)
// mit dem Versprechen „fragt dich“, zwei, drei Anwendungsbeispiele passend zur Seite und der Link zur Hilfe-Seite.
function inkluagentBeispiele(variante) {
    if (variante === 'formular') return [t('Schreib eine Quickinfo für Feld 5.'), t('Welche Felder haben noch keine Quickinfo?'),
                                         t('Fasse die Quickinfos kürzer.')];
    if (variante === 'konto') return [t('Welche Projekte habe ich?'), t('Öffne mein neuestes Projekt.'), t('Wie viele Credits habe ich noch?')];
    return [t('Beschreibe Bild 3 in leichter Sprache.'), t('Erstelle Alt-Texte für alle Bilder.'),
            t('Was fehlt noch, damit das Dokument barrierefrei ist?')];
}
function inkluagentBeispieleHtml(variante) {
    return inkluagentBeispiele(variante).map(b => '<li>' + escHtmlAgent(b) + '</li>').join('');
}
function escHtmlAgent(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function inkluagentKurzhilfeHtml(variante) {
    return ''
        + '<div class="inkluagent-intro inkluagent-kurzhilfe">'
        +   '<p>' + t('Der InkluAgent ist eine KI und arbeitet mit deinen Projekten und Dateien. Bevor etwas Credits kostet oder sich nicht rückgängig machen lässt, fragt er dich.') + '</p>'
        +   '<p id="inkluagentBeispieleTitel" class="inkluagent-beispiele-titel">' + t('Sag ihm in eigenen Worten, was du brauchst, zum Beispiel:') + '</p>'
        +   '<ul class="inkluagent-beispiele" aria-labelledby="inkluagentBeispieleTitel">' + inkluagentBeispieleHtml(variante) + '</ul>'
        + '</div>'
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
    // Schritt 3: im festen Bereich steht der Agent nach main (dann bleibt in main nur die Hauptueberschrift)
    const sec = document.querySelector('.inkluagent-section');
    if (!main || !sec) return false;
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
    // Wie ChatGPT/Claude mit Arbeitsfenster (Michael Karbe, Feedback 20261012 - 1, Punkt 6; Schalter AGENT_ARBEITSFENSTER)
    if (inkluagentArbeitsfensterAn()) { inkluagentChatstil(true); inkluagentArbeitsZeigen(); }
    // Die Knoepfe des Projekts gibt es nur in der manuellen Ansicht (Hochladen ohne Hochladefeld ebenso): ein Knopf zeigt dieses
    // Projekt fuer diese Seite manuell, ohne die gespeicherte Einstellung zu aendern.
    if (!sec.querySelector('.inkluagent-agentenansicht-hinweis')) {
        const p = document.createElement('p');
        p.className = 'inkluagent-agentenansicht-hinweis';
        // Mit dem Hochladefeld (Runde 2, Schritt 1) geht Hochladen auch hier — dann nennt der Hinweis nur noch die Knoepfe
        const ohneProjekt = window._inkluagentProjectId == null;
        p.appendChild(document.createTextNode((ohneProjekt ? t('Den Inhalt dieser Seite findest du in der manuellen Ansicht.')
            : ((inkluagentHochladenAn() || inkluagentAnhaengenAn()) ? t('Die Knöpfe des Projekts findest du in der manuellen Ansicht.')
                                       : t('Hochladen und die Knöpfe des Projekts findest du in der manuellen Ansicht.'))) + ' '));
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn-secondary';
        b.textContent = ohneProjekt ? t('Diese Seite in der manuellen Ansicht zeigen') : t('Dieses Projekt in der manuellen Ansicht zeigen');
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
    inkluagentChatstil(false);
    document.querySelectorAll('[data-agentenansicht="aus"]').forEach(el => { el.hidden = false; el.removeAttribute('data-agentenansicht'); });
    document.querySelectorAll('.inkluagent-gross').forEach(el => el.classList.remove('inkluagent-gross'));
    document.querySelectorAll('.inkluagent-agentenansicht-hinweis').forEach(el => el.remove());
    const h = document.querySelector('#main h1');
    if (h) { if (!h.hasAttribute('tabindex')) h.setAttribute('tabindex', '-1'); h.focus(); }
}

// ─── Agentenansicht wie ChatGPT/Claude mit Arbeitsfenster ─────────────────────────────────────────────────────────────────────
// Michael Karbe, Feedback 20261012 - 1, Punkt 6, und Steves Vision vom 12.10.2026 (Schalter funktionen.AGENT_ARBEITSFENSTER,
// window.INKLUAGENT.arbeitsfenster; nur im festen Bereich und nur in der Agentenansicht). Mitte: das Gespraech — Verlauf oben, die
// Eingabe darunter (angeheftet am unteren Rand), die Beschreibung klein darunter. Rechts: das Arbeitsfenster — Stand zum
// Nachlesen, Knopf „Stand vorlesen“, Ort/Dokument/Seite, Vorschau des Bildes, an dem der Agent arbeitet, und die Schritte des
// letzten Auftrags. Die Gespraeche stehen in der Seitenleiste (dashboard.js). Ansagen nur bei Start, Ende (Fokus auf die
// Antwort), Rueckfrage und Fehler — die einzelnen Schritte landen still im Arbeitsfenster.
function inkluagentArbeitsfensterAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.arbeitsfenster) && inkluagentUeberall();
}
function inkluagentChatstilAktiv() {
    return !!document.querySelector('#inkluagentBereich .inkluagent-chatstil');
}
// Aufbau umstellen. Die Teile werden VERSCHOBEN, nicht neu gebaut — Verlauf, Entwurf und Ereignisse bleiben. an = false: zurueck
// zur Reihenfolge der manuellen Ansicht (Eingabe zuerst, Michaels Punkt 1).
function inkluagentChatstil(an) {
    const sec = document.querySelector('#inkluagentBereich .inkluagent-section');
    const panel = document.getElementById('inkluagentPanel');
    if (!sec || !panel || an === sec.classList.contains('inkluagent-chatstil')) return;
    const form = document.getElementById('inkluagentForm');
    const log = document.getElementById('inkluagentLog');
    const leiste = panel.querySelector('.inkluagent-gespraech-leiste');
    const frueher = panel.querySelector('.inkluagent-frueher');
    const trenner = panel.querySelector('.inkluagent-trenner');
    const beschreibung = panel.querySelector('.inkluagent-beschreibung');
    sec.classList.toggle('inkluagent-chatstil', an);
    document.documentElement.classList.toggle('agentenansicht-chatstil', an);
    if (an) {
        const spalten = document.createElement('div');
        spalten.className = 'inkluagent-spalten';
        const gespraech = document.createElement('div');
        gespraech.className = 'inkluagent-gespraech';
        [frueher, log, form, leiste, trenner, beschreibung].forEach(el => { if (el) gespraech.appendChild(el); });
        spalten.appendChild(gespraech);
        spalten.appendChild(inkluagentArbeitsfensterBauen());
        panel.appendChild(spalten);
        inkluagentAnhangBauen(form, gespraech);   // „Dateien anhängen“ (nur ohne das Feld „Dateien mitschicken“)
    } else {
        inkluagentAnhangEntfernenAlles();
        [form, log, leiste, frueher, trenner, beschreibung].forEach(el => { if (el) panel.appendChild(el); });
        const spalten = panel.querySelector('.inkluagent-spalten');
        if (spalten) spalten.remove();
    }
}
// Ort nachziehen, wenn die Seite ihre Ueberschrift oder Navigation spaeter setzt (Startseite: „Hallo, …“ erst nach dem Laden)
function inkluagentArbeitsOrtBeobachten() {
    if (window._arbeitsOrtBeobachter || typeof MutationObserver === 'undefined') return;
    let zuletzt = inkluagentArbeitsOrt();
    let wartet = false;
    const beobachter = window._arbeitsOrtBeobachter = new MutationObserver(() => {
        if (wartet) return;
        wartet = true;
        setTimeout(() => {
            wartet = false;
            const jetzt = inkluagentArbeitsOrt();
            if (jetzt !== zuletzt && document.getElementById('arbeitsfenster')) { zuletzt = jetzt; inkluagentArbeitsZeigen(); }
        }, 300);
    });
    ['main', 'appSidebar'].forEach(id => {
        const el = document.getElementById(id);
        if (el) beobachter.observe(el, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['aria-current'] });
    });
}
function inkluagentArbeitsfensterBauen() {
    inkluagentArbeitsOrtBeobachten();
    const s = document.createElement('section');
    s.id = 'arbeitsfenster';
    s.className = 'arbeitsfenster';
    s.setAttribute('aria-labelledby', 'arbeitsfensterTitel');
    // Texte aus dem eigenen Katalog; alles aus Daten (Namen, Schritte) setzt inkluagentArbeitsZeigen mit textContent
    s.innerHTML = ''
        + '<h2 id="arbeitsfensterTitel">' + t('Arbeitsfenster') + '</h2>'
        + '<p id="arbeitsStand" class="arbeitsfenster-stand"></p>'
        + '<progress id="arbeitsFortschritt" class="arbeitsfenster-fortschritt" aria-labelledby="arbeitsStand" hidden></progress>'
        + '<p id="arbeitsNotiz" class="arbeitsfenster-notiz" hidden></p>'
        + '<button type="button" id="arbeitsVorlesen" class="btn btn-secondary">' + t('Stand vorlesen') + '</button>'
        // Livestream (Steve 12.10.2026): jeden Schritt ansagen — von Anfang an an, die Wahl bleibt in diesem Browser
        + '<p class="arbeitsfenster-ansagen"><input type="checkbox" id="arbeitsAnsagen"> <label for="arbeitsAnsagen">'
        +   t('Jeden Schritt ansagen') + '</label></p>'
        + '<dl id="arbeitsAngaben" class="arbeitsfenster-angaben" hidden></dl>'
        + '<figure id="arbeitsBild" class="arbeitsfenster-bild" hidden><img alt=""><figcaption></figcaption></figure>'
        + '<div id="arbeitsErgebnis" class="arbeitsfenster-ergebnis" hidden><h3>' + t('Zuletzt fertig') + '</h3><p></p></div>'
        + '<h3 id="arbeitsSchritteTitel" hidden>' + t('Schritte') + '</h3>'
        + '<ol id="arbeitsSchritte" class="arbeitsfenster-schritte" aria-labelledby="arbeitsSchritteTitel" hidden></ol>';
    s.querySelector('#arbeitsVorlesen').addEventListener('click', inkluagentStandVorlesen);
    const ansagen = s.querySelector('#arbeitsAnsagen');
    ansagen.checked = inkluagentSchritteAnsagen();
    ansagen.addEventListener('change', () => {
        try { localStorage.setItem('inkluagent.schritte_ansagen', ansagen.checked ? 'an' : 'aus'); } catch (e) { /* nur diese Seite */ }
    });
    return s;
}
// Zustand je Tab (sessionStorage „inkluagent.arbeitsfenster“), damit er einen Seitenwechsel ins Projekt uebersteht; nach 30 Minuten
// vergessen. Auf oberster Ebene nur Funktionen (tests/test_inkluagent_js.py) — der Zustand liegt an window._arbeits.
function inkluagentArbeitsZustand() {
    if (!window._arbeits) {
        let z = null;
        try { z = JSON.parse(sessionStorage.getItem('inkluagent.arbeitsfenster') || 'null'); } catch (e) { z = null; }
        if (!z || typeof z !== 'object' || Date.now() - (Number(z.zeit) || 0) > 30 * 60 * 1000) z = {};
        z.schritte = Array.isArray(z.schritte) ? z.schritte : [];
        // Lief beim Verlassen der vorigen Seite noch etwas: ob es weiterlaeuft, sagt inkluagentLaufVerfolgen (Server)
        if (z.laeuft) { z.schritte.forEach(x => { x.fertig = true; }); z.stand = t('Fertig. Die Antwort steht im Gespräch.'); }
        z.laeuft = false;
        window._arbeits = z;
    }
    return window._arbeits;
}
function inkluagentArbeitsMerken() {
    const z = inkluagentArbeitsZustand();
    z.zeit = Date.now();
    try { sessionStorage.setItem('inkluagent.arbeitsfenster', JSON.stringify(z)); } catch (e) { /* ohne Speicher: nur diese Seite */ }
    inkluagentArbeitsZeigen();
}
function inkluagentZielText(ziel) {
    if (!ziel) return '';
    if (ziel.art === 'bild') return ziel.nummer ? t('Bild {n}', { n: ziel.nummer }) : t('Bild');
    if (ziel.art === 'feld') return ziel.nummer ? t('Feld {n}', { n: ziel.nummer }) : t('Feld');
    if (ziel.art === 'dokument') return String(ziel.dokument || '');
    return '';
}
// Ort: im Projekt die Hauptueberschrift (Projektname), sonst der aktuelle Eintrag der Navigation („Startseite“, „Meine Ablage“ …).
// Dazu Dokument und Seite — das Ziel nur, wenn es zum Projekt dieser Seite gehoert.
function inkluagentArbeitsOrt() {
    const h1 = document.querySelector('#main h1');
    const titel = h1 ? h1.textContent.trim() : '';
    if (window._inkluagentProjectId != null && titel) return titel;
    const nav = document.querySelector('#appSidebar nav a[aria-current="page"]');
    return nav ? nav.textContent.trim() : titel;
}
function inkluagentArbeitsAngaben(z) {
    const out = [];
    const ort = inkluagentArbeitsOrt();
    if (ort) out.push([t('Ort'), ort]);
    const hier = window._inkluagentProjectId == null ? null : String(window._inkluagentProjectId);
    const ziel = (z.ziel && z.projekt === hier) ? z.ziel : null;
    if (ziel && ziel.dokument) {
        out.push([t('Dokument'), ziel.dokument_nr ? t('Dokument {n}: {name}', { n: ziel.dokument_nr, name: String(ziel.dokument) }) : String(ziel.dokument)]);
    }
    if (ziel && ziel.seite) out.push([t('Seite'), String(ziel.seite)]);
    return out;
}
function inkluagentArbeitsZeigen() {
    const f = document.getElementById('arbeitsfenster');
    if (!f) return;
    const z = inkluagentArbeitsZustand();
    f.querySelector('#arbeitsStand').textContent = z.stand || t('Bereit. Schreib dem InkluAgent, was er erledigen soll.');
    f.querySelector('#arbeitsFortschritt').hidden = !z.laeuft;
    const notiz = f.querySelector('#arbeitsNotiz');
    notiz.textContent = z.notiz ? t('Zwischenstand: {text}', { text: String(z.notiz) }) : '';
    notiz.hidden = !z.notiz;
    const erg = f.querySelector('#arbeitsErgebnis');
    const p = erg.querySelector('p');
    p.textContent = '';
    if (z.ergebnis && z.ergebnis.text) {
        const b = document.createElement('strong');
        b.textContent = String(z.ergebnis.schritt || '') + ': ';
        p.appendChild(b);
        p.appendChild(document.createTextNode(String(z.ergebnis.text)));
    }
    erg.hidden = !(z.ergebnis && z.ergebnis.text);
    const dl = f.querySelector('#arbeitsAngaben');
    dl.textContent = '';
    const angaben = inkluagentArbeitsAngaben(z);
    angaben.forEach(([k, v]) => {
        const dt = document.createElement('dt'); dt.textContent = k;
        const dd = document.createElement('dd'); dd.textContent = v;
        dl.appendChild(dt); dl.appendChild(dd);
    });
    dl.hidden = !angaben.length;
    // Vorschau: die ganze Seite (Seitenansicht), sonst das Bild selbst — nur Adressen der eigenen Bild-Endpunkte
    const fig = f.querySelector('#arbeitsBild');
    const img = fig.querySelector('img');
    const hier = window._inkluagentProjectId == null ? null : String(window._inkluagentProjectId);
    const ziel = (z.ziel && z.projekt === hier && z.ziel.art === 'bild') ? z.ziel : null;
    const gut = u => typeof u === 'string' && /^\/api\/images\/\d+\/(file|page-view)$/.test(u);
    const url = ziel ? (gut(ziel.seite_url) ? ziel.seite_url : (gut(ziel.bild_url) ? ziel.bild_url : '')) : '';
    if (url) {
        img.onerror = () => {
            if (img.getAttribute('src') !== ziel.bild_url && gut(ziel.bild_url)) img.src = ziel.bild_url;
            else fig.hidden = true;
        };
        if (img.getAttribute('src') !== url) img.src = url;
        // Die Bildunterschrift sagt, was zu sehen ist; das Bild selbst ist dann Schmuck (alt="")
        fig.querySelector('figcaption').textContent = [inkluagentZielText(ziel), ziel.seite ? t('Seite {n}', { n: ziel.seite }) : '']
            .filter(Boolean).join(' · ');
        fig.hidden = false;
    } else {
        fig.hidden = true;
    }
    const ol = f.querySelector('#arbeitsSchritte');
    ol.textContent = '';
    z.schritte.forEach(x => {
        const li = document.createElement('li');
        li.dataset.zustand = x.fehler ? 'fehler' : (x.fertig ? 'erledigt' : 'laeuft');
        li.appendChild(document.createTextNode(String(x.text || '')));
        const sp = document.createElement('span');
        sp.className = 'schritt-zustand';
        sp.textContent = ' – ' + (x.fehler ? t('hat nicht geklappt') : (x.fertig ? t('erledigt') : t('läuft')));
        li.appendChild(sp);
        ol.appendChild(li);
    });
    ol.hidden = !z.schritte.length;
    f.querySelector('#arbeitsSchritteTitel').hidden = ol.hidden;
}
// „Jeden Schritt ansagen“ (Vorgabe an; ohne Speicher auch an)
function inkluagentSchritteAnsagen() {
    const k = document.getElementById('arbeitsAnsagen');
    if (k) return k.checked;
    try { return localStorage.getItem('inkluagent.schritte_ansagen') !== 'aus'; } catch (e) { return true; }
}
// Livestream (Steve 12.10.2026): die Antwort waechst sichtbar als Entwurf (Wort fuer Wort aus dem Stream); die fertige, bereinigte
// Antwort ersetzt ihn. Der Verlauf sagt nichts von selbst an (aria-live off) — VoiceOver bekommt die fertige Antwort mit dem Fokus.
function inkluagentEntwurfText(stueck) {
    const log = document.getElementById('inkluagentLog');
    if (!log || typeof stueck !== 'string') return;
    let el = document.getElementById('inkluagentEntwurf');
    if (!el) {
        el = document.createElement('div');
        el.id = 'inkluagentEntwurf';
        el.className = 'inkluagent-message assistant inkluagent-entwurf';
        el.setAttribute('aria-busy', 'true');
        const h = document.createElement('h3');
        h.className = 'inkluagent-message-role';
        h.textContent = 'InkluAgent';
        const c = document.createElement('div');
        c.className = 'inkluagent-message-content';
        el.appendChild(h);
        el.appendChild(c);
        log.appendChild(el);
        window._inkluagentEntwurf = '';
        // Arbeitsfenster: jetzt schreibt er (ohne Ansage — die fertige Antwort bekommt den Fokus)
        if (inkluagentChatstilAktiv()) {
            const z = inkluagentArbeitsZustand();
            if (z.laeuft) { z.stand = t('Schreibt die Antwort …'); inkluagentArbeitsMerken(); }
        }
    }
    window._inkluagentEntwurf = (window._inkluagentEntwurf || '') + stueck;
    if (window._inkluagentEntwurfPlan) return;
    window._inkluagentEntwurfPlan = requestAnimationFrame(() => {
        window._inkluagentEntwurfPlan = 0;
        const e = document.getElementById('inkluagentEntwurf');
        if (!e) return;
        const unten = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 200;
        // Sterne und Rauten wie die Bereinigung auf dem Server (sanitize_markdown) — nur fuer den Entwurf
        e.querySelector('.inkluagent-message-content').textContent = window._inkluagentEntwurf.replace(/\*\*|__/g, '').replace(/^#{1,6}\s*/gm, '');
        if (unten) window.scrollTo(0, document.documentElement.scrollHeight);   // mitlaufen, wer unten liest
    });
}
function inkluagentEntwurfVerwerfen() {
    const el = document.getElementById('inkluagentEntwurf');
    if (el) el.remove();
    window._inkluagentEntwurf = '';
}
// Gestreamter Text vor Werkzeugen ist ein Zwischenstand: raus aus dem Gespraech, ins Arbeitsfenster
function inkluagentEntwurfZwischen() {
    const text = (window._inkluagentEntwurf || '').replace(/\*\*|__/g, '').trim();
    inkluagentEntwurfVerwerfen();
    if (!text || !inkluagentChatstilAktiv()) return;
    const z = inkluagentArbeitsZustand();
    z.notiz = text.slice(0, 400);
    inkluagentArbeitsMerken();
}
// Ergebnis eines Werkzeugs: Schritt erledigt oder „hat nicht geklappt“, neuer Alt-Text/Quickinfo unter „Zuletzt fertig“
function inkluagentArbeitsFertig(ev) {
    if (!inkluagentChatstilAktiv() || !ev) return;
    const z = inkluagentArbeitsZustand();
    const schritt = z.schritte.slice().reverse().find(x => !x.fertig);
    if (schritt) {
        schritt.fertig = true;
        if (!ev.ok) schritt.fehler = true;
    }
    if (ev.ok && ev.text) z.ergebnis = { schritt: schritt ? schritt.text : inkluagentWerkzeugLabel(ev.name), text: String(ev.text) };
    inkluagentArbeitsMerken();
    if (!ev.ok && schritt && inkluagentSchritteAnsagen()) announce(schritt.text + ' – ' + t('hat nicht geklappt'));
}
// Lauf: Start (Schritte neu, ausser man verfolgt den Lauf einer vorigen Seite), je Werkzeug ein Schritt, Ende, Fehler
function inkluagentArbeitsStart(text, behalten) {
    if (!inkluagentChatstilAktiv()) return;
    const z = inkluagentArbeitsZustand();
    z.laeuft = true;
    if (!behalten) { z.schritte = []; z.notiz = ''; z.ergebnis = null; }
    z.stand = text || t('Arbeitet: liest deine Nachricht und plant die Schritte.');
    inkluagentArbeitsMerken();
}
function inkluagentArbeitsSchritt(name, ziel) {
    if (!inkluagentChatstilAktiv()) return;
    const z = inkluagentArbeitsZustand();
    z.schritte.forEach(x => { x.fertig = true; });
    const zt = inkluagentZielText(ziel);
    const text = inkluagentWerkzeugLabel(name) + (zt ? ' – ' + zt : '');
    z.schritte.push({ text, fertig: false });
    if (z.schritte.length > 30) z.schritte = z.schritte.slice(-30);
    if (ziel && typeof ziel === 'object') {
        z.ziel = ziel;
        z.projekt = window._inkluagentProjectId == null ? null : String(window._inkluagentProjectId);
    }
    z.stand = t('Arbeitet an: {schritt}', { schritt: text });
    inkluagentArbeitsMerken();
    if (inkluagentSchritteAnsagen()) announce(text);   // „Jeden Schritt ansagen“ (Steve 12.10.2026)
}
function inkluagentArbeitsEnde(wartet) {
    if (!inkluagentChatstilAktiv()) return;
    const z = inkluagentArbeitsZustand();
    z.laeuft = false;
    z.schritte.forEach(x => { x.fertig = true; });
    z.stand = wartet ? t('Wartet auf dich: Bitte entscheide unter der Antwort.') : t('Fertig. Die Antwort steht im Gespräch.');
    inkluagentArbeitsMerken();
}
function inkluagentArbeitsFehler(text) {
    if (!inkluagentChatstilAktiv()) return;
    const z = inkluagentArbeitsZustand();
    if (!z.laeuft) return;
    z.laeuft = false;
    z.stand = t('Fehler: {text}', { text: String(text || '') });
    inkluagentArbeitsMerken();
}
function inkluagentArbeitsZuruecksetzen() {
    window._arbeits = { schritte: [] };
    inkluagentArbeitsMerken();
}
// „Stand vorlesen“: Stand und Angaben einmal ansagen; der Fokus bleibt auf dem Knopf
function inkluagentStandVorlesen() {
    const z = inkluagentArbeitsZustand();
    const teile = [z.stand || t('Bereit. Schreib dem InkluAgent, was er erledigen soll.')]
        .concat(inkluagentArbeitsAngaben(z).map(([k, v]) => k + ': ' + v));
    if (z.ergebnis && z.ergebnis.text) teile.push(t('Zuletzt fertig') + ': ' + z.ergebnis.schritt + ': ' + z.ergebnis.text);
    announce(teile.map(x => x.replace(/[.\s]+$/, '')).join('. ') + '.');
}
// Wartet eine Antwort auf eine Entscheidung (offene Karte)?
function inkluagentWartetAufDich(anhang) {
    return Array.isArray(anhang) && anhang.some(a => a && a.art === 'bestaetigung' && a.angebot_id && (!a.zustand || a.zustand === 'offen'));
}

function inkluagentInit(projectId) {
    const p = inkluagentUeberall() ? inkluagentBinden(projectId) : inkluagentInitLaden(projectId);
    window._inkluagentBereit = p;
    return p;
}
// Bedienung (Auf/Zu, Senden, Enter) — im festen Bereich EINMAL, in eingebetteten Ansichten nach jedem Neuzeichnen
function inkluagentBedienungAnbinden() {
    const toggle = document.getElementById('inkluagentToggle');
    const input = document.getElementById('inkluagentInput');
    const form = document.getElementById('inkluagentForm');
    if (!toggle || toggle.dataset.angebunden) return !!toggle;
    toggle.dataset.angebunden = '1';
    toggle.addEventListener('click', () => {
        const isOpen = toggle.getAttribute('aria-expanded') === 'true';
        if (isOpen) inkluagentClose(); else inkluagentOpen(window._inkluagentProjectId, true);
    });
    if (form) form.addEventListener('submit', (e) => { e.preventDefault(); inkluagentSend(); });
    const neu = document.getElementById('inkluagentNeuesGespraech');
    if (neu) neu.addEventListener('click', inkluagentNeuesGespraech);
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); inkluagentSend(); }
        });
    }
    return true;
}
// Wo „offen“ gemerkt wird: je Projekt (eingebettet) oder einmal fuer alle Seiten (fester Bereich)
function inkluagentMerkSchluessel(projectId) {
    return inkluagentUeberall() ? 'inkluagent.panel.ueberall' : 'inkluagent.panel.' + projectId;
}

// ─── Schritt 3: fester Bereich auf jeder Seite ───────────────────────────────────────────────────────────────────────────────
// Einmal je Seite: Inhalt in <aside id="inkluagentBereich"> bauen. Projektseiten (/app) binden ihn, sobald ihre Ansicht gezeichnet
// ist (inkluagentInit); alle anderen Seiten sofort ohne Projekt.
function inkluagentBereichAufbauen() {
    const bereich = document.getElementById('inkluagentBereich');
    if (!bereich || !inkluagentUeberall() || bereich.dataset.gebaut) return;
    bereich.dataset.gebaut = '1';
    const projektseite = /^\/app\/?$/.test(window.location.pathname);
    bereich.innerHTML = inkluagentInhaltHtml(null, projektseite ? 'bilder' : 'konto', true);
    inkluagentBedienungAnbinden();
    // Seiten ohne Projekt: binden, wenn die Seite steht (ihre Skripte bauen main oft erst noch auf — wichtig fuer die Agentenansicht)
    if (!projektseite) {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => inkluagentInit(null), { once: true });
        else inkluagentInit(null);
    }
}
function inkluagentVarianteSetzen(variante) {
    const sec = document.querySelector('#inkluagentBereich .inkluagent-section');
    if (!sec || sec.dataset.variante === variante) return;
    sec.dataset.variante = variante;
    const intro = sec.querySelector('p.inkluagent-intro');
    if (intro) intro.innerHTML = inkluagentEinleitung(variante);   // Texte aus dem eigenen Katalog (mit <em>)
    const beispiele = sec.querySelector('ul.inkluagent-beispiele');
    if (beispiele) beispiele.innerHTML = inkluagentBeispieleHtml(variante);   // maskiert (escHtmlAgent)
    const input = document.getElementById('inkluagentInput');
    if (input) input.placeholder = inkluagentPlatzhalter(variante);
}
// Den festen Bereich an ein Projekt binden (oder an keins: null). Gleiches Projekt nach einem Neuzeichnen der Ansicht: nur die
// Agentenansicht neu anwenden — Verlauf, Entwurf und Auf/Zu bleiben, weil der Bereich ausserhalb von main steht.
async function inkluagentBinden(projectId) {
    inkluagentBereichAufbauen();
    const pid = projectId == null ? null : projectId;
    const neu = !window._inkluagentGebunden || String(window._inkluagentProjectId) !== String(pid);
    window._inkluagentGebunden = true;
    window._inkluagentProjectId = pid;
    inkluagentVarianteSetzen(pid == null ? 'konto' : (window._inkluagentVariante || 'bilder'));
    window._inkluagentVariante = null;
    const hg = window._inkluagentHochgeladen;
    if (hg && hg.projectId === String(pid)) inkluagentHochgeladenZeigen(pid, hg.ergebnisse);
    const gross = inkluagentAgentenansicht();
    if (!neu) return;
    let wasOpen = false;
    try { wasOpen = localStorage.getItem(inkluagentMerkSchluessel(pid)) === 'open'; } catch (e) {}
    // „Hier weitermachen“ (Seite „Frühere Gespräche“) fuehrt mit ?agent=offen hierher: Agent auf, Fokus auf die letzte Nachricht
    const weiter = new URLSearchParams(window.location.search).get('agent') === 'offen';
    if (weiter) {
        try { const u = new URL(window.location.href); u.searchParams.delete('agent'); history.replaceState(history.state, '', u); } catch (e) {}
        await inkluagentOpen(pid, false);
        const alle = document.querySelectorAll('#inkluagentLog .inkluagent-message');
        if (alle.length) alle[alle.length - 1].focus();
        else { const i = document.getElementById('inkluagentInput'); if (i) i.focus(); }
    } else if (gross) await inkluagentOpen(pid, false, true);
    else if (wasOpen) await inkluagentOpen(pid, false);
    inkluagentLaufVerfolgen();
}
// Laufende Antwort (Schritt 3): hat der Server fuer dieses Konto noch eine Antwort in Arbeit (z. B. weil man die Seite gewechselt
// hat), sagt die Statuszeile es und die Seite fragt nach, bis sie da ist — dann neu laden und melden wie eine neue Antwort.
let _inkluagentVerfolgt = false;
async function inkluagentLaufVerfolgen() {
    if (!inkluagentUeberall() || _inkluagentVerfolgt) return;
    const sendBtn = document.getElementById('inkluagentSendBtn');
    if (sendBtn && sendBtn.disabled) return;   // diese Seite sendet gerade selbst
    let laeuft = false;
    try { laeuft = !!(await (await fetch('/api/agent/status')).json()).laeuft; } catch (e) { return; }
    if (!laeuft) return;
    _inkluagentVerfolgt = true;
    inkluagentSetStatus(t('Der InkluAgent arbeitet noch an seiner letzten Antwort …'), false);
    inkluagentArbeitsStart(t('Arbeitet noch an der letzten Antwort …'), true);
    const ende = Date.now() + 10 * 60 * 1000;
    while (Date.now() < ende) {
        await new Promise(r => setTimeout(r, 2500));
        try { laeuft = !!(await (await fetch('/api/agent/status')).json()).laeuft; } catch (e) { laeuft = true; }
        if (!laeuft) break;
    }
    _inkluagentVerfolgt = false;
    inkluagentSetStatus('', false);
    inkluagentArbeitsEnde(false);
    if (typeof window.seitenleisteGespraecheLaden === 'function') window.seitenleisteGespraecheLaden();
    const panel = document.getElementById('inkluagentPanel');
    const fokussieren = inkluagentFokusImChat();
    if (panel && !panel.hidden) {
        await inkluagentLoadHistory(window._inkluagentProjectId);
        inkluagentAntwortMelden(inkluagentLetzteAntwort(), fokussieren);
    } else {
        inkluagentAntwortMelden(null, false);
    }
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
    const neuesGespraech = document.getElementById('inkluagentNeuesGespraech');
    if (neuesGespraech) neuesGespraech.addEventListener('click', inkluagentNeuesGespraech);
    // Ergebnis des letzten Hochladens bleibt nach dem Neuzeichnen der Ansicht stehen (gleiches Projekt)
    const hg = window._inkluagentHochgeladen;
    if (hg && hg.projectId === String(projectId)) inkluagentHochgeladenZeigen(projectId, hg.ergebnisse);
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
    if (!nichtMerken) { try { localStorage.setItem(inkluagentMerkSchluessel(projectId), 'open'); } catch (e) {} }
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
    if (projectId || inkluagentUeberall()) {
        try { localStorage.removeItem(inkluagentMerkSchluessel(projectId)); } catch (e) {}
    }
}

// Runde 2, Schritt 2 (Schalter funktionen.AGENT_KONTO, window.INKLUAGENT.konto): EIN Gespraech je Konto ueber alle Projekte.
// Wo das Projekt wechselt, steht eine Marke („Ab hier: Projekt …“); das fruehere Gespraech dieses Projekts (vor dem Umstieg)
// bleibt lesbar — als natives Aufklappelement ueber dem Verlauf, geladen erst beim Aufklappen. Offene Karten anderer Projekte
// zeigt der Server ohne Knopf („Zu bestätigen im Projekt …“).
function inkluagentKonto() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.konto);
}
// Gespraeche (Steve 12.10.2026, Schalter AGENT_GESPRAECHE): oben im Agenten nur zwei Dinge — „Neues Gespräch“ und der Link
// „Frühere Gespräche“ auf eine eigene Seite (die Liste steht bewusst nicht im Agenten).
function inkluagentGespraecheAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.gespraeche);
}
function inkluagentGespraechLeisteHtml() {
    return '<div class="inkluagent-gespraech-leiste">'
        + '<button type="button" id="inkluagentNeuesGespraech" class="btn btn-secondary">' + t('Neues Gespräch') + '</button>'
        + '<a href="/gespraeche">' + t('Frühere Gespräche') + '</a>'
        + '</div>';
}
async function inkluagentNeuesGespraech() {
    const btn = document.getElementById('inkluagentNeuesGespraech');
    if (!btn || btn.getAttribute('aria-disabled') === 'true') return;
    btn.setAttribute('aria-disabled', 'true');
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
        btn.removeAttribute('aria-disabled');
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
        if (res.status === 401) { window.location.href = '/login'; return; }
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
            inkluagentAppendMessage(m.role, m.content, m.werkzeuge, m.anhang);
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
                c.textContent = m.content || '';
                w.appendChild(h);
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
        if (a.type === 'projekt_geloescht') inkluagentProjektWeg(a.project_id);
    });
    return nachziehen;
}

// Seitenwechsel (Steve 12.10.2026): geht es nach einer Antwort um GENAU ein Projekt (oeffnen, anlegen, Datei oder Webseite als
// Projekt), schickt der Server {"type": "seite_wechseln", "url": "/app?projekt=…"}. Die Seite wechselt dorthin; mit agent=offen ist
// der InkluAgent dort offen und der Fokus steht auf dieser Antwort (inkluagentBinden). Nur Projektseiten dieser Anwendung.
function inkluagentSeiteWechseln(actions) {
    const a = (actions || []).find(x => x && x.type === 'seite_wechseln' && typeof x.url === 'string' && /^\/app\?projekt=\d+$/.test(x.url));
    if (!a) return false;
    if (window._inkluagentProjectId != null && String(window._inkluagentProjectId) === String(a.project_id)
        && /^\/app\/?$/.test(window.location.pathname)) return false;   // schon dort
    window.location.href = a.url + '&agent=offen';
    return true;
}

// Schritt 4: der Agent hat das Projekt geloescht, auf dessen Seite man ist — der Hauptbereich sagt das und verlinkt die Uebersicht.
// Kein automatischer Seitenwechsel (der Fokus bleibt auf dem Knopf der Karte, die Antwort ist noch zu hoeren); der Agent bleibt.
function inkluagentProjektWeg(projectId) {
    if (window._inkluagentProjectId == null || String(projectId) !== String(window._inkluagentProjectId)) return;
    const main = document.getElementById('main');
    if (!main) return;
    main.textContent = '';
    const h = document.createElement('h1');
    h.textContent = t('Projekt gelöscht');
    const p = document.createElement('p');
    p.textContent = t('Das Projekt ist gelöscht.') + ' ';
    const a = document.createElement('a');
    a.href = '/dashboard';
    a.textContent = t('Zur Übersicht');
    p.appendChild(a);
    main.appendChild(h);
    main.appendChild(p);
    window._inkluagentProjectId = null;
    announce(t('Das Projekt ist gelöscht.'));
}

// Werkzeug-Transparenz (Steve 28.08.2026): Namen der Agenten-Werkzeuge fuer die Live-Zeile „Ruft gerade auf: …“ und die
// Zeile „Genutzt: …“ unter jeder Antwort. Seit Pruefung 3 (30.09.2026, M1) kommen ALLE Namen vom Server
// (window.WERKZEUG_NAMEN, inkluagent/tools/namen.py, 6 Sprachen) — nie ein roher Name wie „dokument_stand“.
function inkluagentWerkzeugLabel(name) {
    // Seiten ohne app.html (Schritt 3): die Namen kommen mit window.INKLUAGENT
    const map = window.WERKZEUG_NAMEN || (window.INKLUAGENT && window.INKLUAGENT.werkzeug_namen) || {};
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
async function inkluagentBestaetigen(angebotId, btn, statusEl, konto) {
    const projectId = window._inkluagentProjectId;
    // konto (Ausbau Runde 2, Schritt 4): Karten aus Seiten ohne Projekt (Projekt löschen) gelten auf jeder Seite
    if ((!konto && !projectId) || !btn || btn.getAttribute('aria-disabled') === 'true') return;
    btn.setAttribute('aria-disabled', 'true');
    if (statusEl) statusEl.textContent = t('Wird ausgeführt …');
    try {
        const res = await fetch(konto ? '/api/agent/bestaetigen' : '/api/projects/' + projectId + '/chat/bestaetigen', {
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
        if (inkluagentSeiteWechseln(data.actions)) return;
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
                b.addEventListener('click', () => inkluagentBestaetigen(a.angebot_id, b, st, !!a.konto));
                karte.appendChild(b); karte.appendChild(st);
            }
            box.appendChild(karte);
            return;
        }
        // Link unter der Antwort (Schritt 3, projekt_oeffnen): nur Adressen dieser Seite (beginnen mit „/“)
        if (a && a.art === 'link' && typeof a.url === 'string' && /^\/(?!\/)/.test(a.url)) {
            const l = document.createElement('a');
            l.className = 'btn btn-secondary';
            l.href = a.url;
            l.textContent = a.text || a.url;
            box.appendChild(l);
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
    if (role === 'assistant' && window.INKLUAGENT && window.INKLUAGENT.bloecke) inkluagentBloeckeBauen(contentEl, content || '');
    else contentEl.textContent = content || '';
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
    if (isError && text && typeof inkluagentArbeitsFehler === 'function') inkluagentArbeitsFehler(text);
}

async function inkluagentSend() {
    const projectId = window._inkluagentProjectId;
    const input = document.getElementById('inkluagentInput');
    const sendBtn = document.getElementById('inkluagentSendBtn');
    const log = document.getElementById('inkluagentLog');
    // Ohne Projekt nur im festen Bereich (Schritt 3): dann geht die Nachricht an /api/agent/nachricht
    const ohneProjekt = projectId == null;
    if ((ohneProjekt && !inkluagentUeberall()) || (!ohneProjekt && !projectId) || !input || !sendBtn || !log) return;
    if (sendBtn.disabled) return;   // Enter waehrend des Hochladens oder Antwortens: nichts doppelt senden
    const message = input.value.trim();
    // Hochladefeld (Runde 2, Schritt 1): Dateien duerfen ohne eigenen Text gesendet werden (Webadressen stehen in der Nachricht)
    const dateienEl = document.getElementById('inkluagentDateien');
    const dateien = inkluagentAnhangAktiv() ? (window._inkluagentAnhaenge || []).slice()   // „Dateien anhängen“ (Agentenansicht)
        : ((inkluagentHochladenAn() && dateienEl && dateienEl.files) ? Array.from(dateienEl.files) : []);
    const webadresse = '';
    if (!message && !dateien.length) {
        inkluagentSetStatus((inkluagentHochladenAn() || inkluagentAnhangAktiv()) ? t('Bitte eine Nachricht eingeben oder eine Datei auswählen.') : t('Bitte eine Nachricht eingeben.'), true);
        input.focus();
        return;
    }
    if (message.length > 5000) {
        inkluagentSetStatus(t('Nachricht ist zu lang (max. 5000 Zeichen).'), true);
        return;
    }
    sendBtn.disabled = true;
    let hochgeladen = null;
    let verlaufText = message;
    let nachziehenHochladen = false;
    let nachgezogen = false;
    let satz = '';
    if (dateien.length || webadresse) {
        if (inkluagentChatstilAktiv()) {
            inkluagentArbeitsStart(dateien.length === 1 ? t('Lädt 1 Datei hoch …') : t('Lädt {n} Dateien hoch …', { n: dateien.length }));
        }
        const erg = await inkluagentHochladen(projectId, dateien, webadresse);
        if (!erg) { sendBtn.disabled = false; return; }
        const ergebnisse = erg.ergebnisse || [];
        inkluagentHochgeladenZeigen(projectId, ergebnisse);
        if (dateienEl) dateienEl.value = '';
        if (inkluagentAnhangAktiv()) inkluagentAnhaengeLeeren();
        satz = inkluagentHochladenSatz(ergebnisse);
        nachziehenHochladen = !ohneProjekt && ergebnisse.some(e => e.ok && String(e.project_id) === String(projectId));
        // Nichts angekommen und kein eigener Text: keine Nachricht an den Agenten (die Gruende stehen schon am Feld)
        if (!message && !ergebnisse.some(e => e.ok)) {
            inkluagentSetStatus(satz, true);
            sendBtn.disabled = false;
            return;
        }
        hochgeladen = erg.notiz_id || null;
        // Dieselbe Zeile, die der Server in den Verlauf schreibt (hochladen.nachricht_fuer_verlauf)
        verlaufText = (message && erg.kurz) ? message + '\n' + erg.kurz : (message || erg.kurz || '');
    }
    inkluagentMarkeVorNachricht(projectId);
    inkluagentAppendMessage('user', verlaufText);
    log.scrollTop = log.scrollHeight;
    input.value = '';
    // Eine gebuendelte Ansage: Ergebnis des Hochladens und dass der Agent arbeitet
    // Chatstil: EINE Ansage zum Start, die einzelnen Schritte stehen still im Arbeitsfenster (Steve 12.10.2026)
    const chatstil = inkluagentChatstilAktiv();
    inkluagentSetStatus((satz ? satz + ' ' : '') + (chatstil ? t('InkluAgent arbeitet. Was er gerade tut, steht im Arbeitsfenster.')
                                                              : t('InkluAgent denkt nach...')), false);
    inkluagentArbeitsStart();
    try {
        // Stream (NDJSON, 28.08.2026): je Werkzeugaufruf eine Zeile -> Live-Zeile im Status
        // (aria-live, VoiceOver sagt sie an), am Ende die Antwort mit Werkzeugliste.
        const res = await fetch(ohneProjekt ? '/api/agent/nachricht' : '/api/projects/' + projectId + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/x-ndjson' },
            body: JSON.stringify(hochgeladen ? { message, hochgeladen } : { message })
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
                        if (chatstil) inkluagentArbeitsSchritt(ev.name, ev.ziel);
                        else inkluagentSetStatus(t('Ruft gerade auf: {w}', { w: inkluagentWerkzeugLabel(ev.name) }), false);
                    } else if (ev.type === 'werkzeug_fertig') {   // Livestream (Arbeitsfenster): Ergebnis je Werkzeug
                        if (chatstil) inkluagentArbeitsFertig(ev);
                    } else if (ev.type === 'text') {   // Antwort Wort fuer Wort (nur im Chatstil sichtbar)
                        if (chatstil) inkluagentEntwurfText(ev.text);
                    } else if (ev.type === 'text_neu') {
                        inkluagentEntwurfVerwerfen();
                    } else if (ev.type === 'zwischen') {
                        if (chatstil) inkluagentEntwurfZwischen(); else inkluagentEntwurfVerwerfen();
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
        inkluagentEntwurfVerwerfen();   // die fertige Antwort ersetzt den Entwurf aus dem Stream
        const antwortEl = inkluagentAppendMessage('assistant', data.reply || t('(leere Antwort)'), Array.isArray(data.werkzeuge) ? data.werkzeuge : genutzt, data.anhang);
        log.scrollTop = log.scrollHeight;
        inkluagentSetStatus('', false);
        if (data.intent === 'error') inkluagentArbeitsFehler(data.reply || '');
        else inkluagentArbeitsEnde(inkluagentWartetAufDich(data.anhang));
        if (chatstil) {
            if (typeof window.seitenleisteGespraecheLaden === 'function') window.seitenleisteGespraecheLaden();
            // Die neue Antwort sichtbar machen (die Seite scrollt, nicht der Verlauf)
            if (antwortEl && antwortEl.scrollIntoView) antwortEl.scrollIntoView({ block: 'nearest' });
        }
        if (inkluagentSeiteWechseln(data.actions)) { window._inkluagentFokusAntwort = true; return; }
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
        nachgezogen = nachziehen || needsFullReload || nachziehenHochladen;
        if (nachgezogen && await inkluagentAnsichtNachziehen(projectId)) {
            inkluagentAntwortMelden(inkluagentLetzteAntwort(), fokussieren);
        } else {
            inkluagentAntwortMelden(antwortEl, fokussieren);
        }
    } catch (e) {
        inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true);
    } finally {
        sendBtn.disabled = false;
        inkluagentEntwurfVerwerfen();   // abgebrochen: kein halber Entwurf bleibt stehen
        // Sicherheitsnetz: kein Weg darf das Arbeitsfenster auf „arbeitet“ stehen lassen
        if (window._arbeits && window._arbeits.laeuft) inkluagentArbeitsEnde(false);
        // In dieses Projekt hochgeladen, aber die Antwort kam nicht: die Ansicht trotzdem nachziehen (neue Dokumente, Bilder)
        if (nachziehenHochladen && !nachgezogen) inkluagentAnsichtNachziehen(projectId);
        // Zurueck ins Eingabefeld nur, wenn der Fokus im Chat war (Fehlerfall) — nie aus einer anderen Stelle heraus
        if (window._inkluagentFokusAntwort) { window._inkluagentFokusAntwort = false; }
        else if (inkluagentFokusImChat() && document.activeElement !== input) { input.focus(); }
    }
}

// Schritt 3: den festen Bereich gleich beim Laden bauen (das aside steht im Seitenrahmen vor diesem Skript); ohne Schalter oder
// ohne Bereich (app.html bettet ein) passiert nichts.
inkluagentBereichAufbauen();
