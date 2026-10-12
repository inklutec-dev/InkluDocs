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
function inkluagentPlatzhalter(variante) {
    if (variante === 'formular') return t("z.B. 'Feld 3 kürzer fassen' oder 'Generiere eine Quickinfo für Feld 5'");
    if (variante === 'konto') return t("z.B. 'Welche Projekte habe ich?' oder 'Wie viele Credits habe ich noch?'");
    return t("z.B. 'Bild 3 in leichter Sprache' oder 'Generiere Alt-Text für Bild 5'");
}
// alsBereich (Schritt 3): Inhalt des festen Bereichs — ein div statt section, denn das aside ist schon der Bereich (Landmark)
function inkluagentInhaltHtml(projectId, variante, alsBereich) {
    const formular = variante === 'formular';
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
        // InkluAgent-Ausbau Runde 1, Schritt 3 (Schalter funktionen.AGENT_HILFE, window.FUNKTIONEN.agent_hilfe): drei kurze
        // Stichpunkte statt der langen Einleitung, der erste ist der KI-Hinweis nach KI-VO Art. 50; darunter der Weg zur
        // Hilfe-Seite. Echte Liste (VoiceOver: „Liste, 3 Objekte“). Schalter aus = die Einleitung wie bisher.
        + (inkluagentKurzhilfe() ? inkluagentKurzhilfeHtml(variante) : (''
        +   '<p class="inkluagent-intro">'
        // KI-VO Art. 50 Abs. 1 (gilt seit 02.08.2026): Menschen muessen
        // erkennen, dass sie mit einem KI-System sprechen. Der Name
        // „InkluAgent“ sagt das allein nicht — der ausdrueckliche Satz
        // „… ist ein KI-Assistent“ direkt unter dem Knopf sagt es.
        // (25.08.2026: stand als Jinja-Kommentar INNERHALB des raw-Blocks,
        // blieb woertlich im JavaScript stehen und brach das ganze
        // App-Skript mit einem SyntaxError ab. Im raw-Block nur
        // JS-Kommentare verwenden.)
        +   inkluagentEinleitung(variante)
        +   '</p>'))
        +   '<div id="inkluagentPanel" class="inkluagent-panel" hidden>'
        // tabindex=0 (11.09.2026, axe scrollable-region-focusable): der Verlauf wird bei vielen
        // Nachrichten scrollbar und muss dann per Tastatur erreichbar sein (WCAG 2.1.1).
        // aria-live="off" (Pruefung 3 Barrierefreiheit, H1/N1): role=log ist sonst von selbst „polite“ — beim Neuzeichnen
        // (Ansichtswechsel, Neuladen, Auf/Zu) wurde der ganze Verlauf erneut angesagt, jede neue Antwort doppelt (Live-Region
        // UND Fokus) und die eigene Nachricht als Echo. Jetzt sagt nichts im Verlauf von selbst etwas an: die neue Antwort
        // bekommt den Fokus (der Screenreader liest sie vollständig), Warten und Fehler meldet die Statuszeile.
        +     '<div id="inkluagentLog" class="inkluagent-log" role="log" aria-live="off" aria-label="' + t('Chat-Verlauf') + '" tabindex="0"></div>'
        // novalidate (Runde 2, Schritt 1): eine unvollstaendige Webadresse meldet inkluagentSend in der Statuszeile und setzt den
        // Fokus ins Feld — statt der Blase des Browsers, die ohne Ansage verschwindet und das Senden still verhindert
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
        +   '</div>'
        + (alsBereich ? '</div>' : '</section>');
}

// Runde 2, Schritt 1 (Schalter funktionen.AGENT_HOCHLADEN, window.INKLUAGENT aus base_app.html): Dateien und eine Webadresse
// gehen mit der Nachricht an den Agenten. Native Felder mit sichtbarem Label (Datei-Auswahl, type=url); die Zuordnung zu
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
        +   '<label for="inkluagentWebadresse">' + t('Webadresse (freiwillig)') + '</label>'
        +   '<input type="url" id="inkluagentWebadresse" autocomplete="url" placeholder="https://" aria-describedby="inkluagentHochladenHint">'
        +   '<p id="inkluagentHochladenHint" class="inkluagent-hint">'
        +     t('PDF, Word (.docx) und Bilder, höchstens {n} Dateien, je bis {mb} MB. Der InkluAgent legt sie ins passende Projekt und nennt dir die Preise, bevor etwas startet.',
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
    const gut = (ergebnisse || []).filter(e => e.ok).length;
    return t('{gut} von {alle} hochgeladen.', { gut: gut, alle: alle }) + (gut < alle ? ' ' + t('Die Gründe stehen unter dem Feld.') : '');
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
    // Die Knoepfe des Projekts gibt es nur in der manuellen Ansicht (Hochladen ohne Hochladefeld ebenso): ein Knopf zeigt dieses
    // Projekt fuer diese Seite manuell, ohne die gespeicherte Einstellung zu aendern.
    if (!sec.querySelector('.inkluagent-agentenansicht-hinweis')) {
        const p = document.createElement('p');
        p.className = 'inkluagent-agentenansicht-hinweis';
        // Mit dem Hochladefeld (Runde 2, Schritt 1) geht Hochladen auch hier — dann nennt der Hinweis nur noch die Knoepfe
        const ohneProjekt = window._inkluagentProjectId == null;
        p.appendChild(document.createTextNode((ohneProjekt ? t('Den Inhalt dieser Seite findest du in der manuellen Ansicht.')
            : (inkluagentHochladenAn() ? t('Die Knöpfe des Projekts findest du in der manuellen Ansicht.')
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
    document.querySelectorAll('[data-agentenansicht="aus"]').forEach(el => { el.hidden = false; el.removeAttribute('data-agentenansicht'); });
    document.querySelectorAll('.inkluagent-gross').forEach(el => el.classList.remove('inkluagent-gross'));
    document.querySelectorAll('.inkluagent-agentenansicht-hinweis').forEach(el => el.remove());
    const h = document.querySelector('#main h1');
    if (h) { if (!h.hasAttribute('tabindex')) h.setAttribute('tabindex', '-1'); h.focus(); }
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
    if (gross) await inkluagentOpen(pid, false, true);
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
    const ende = Date.now() + 10 * 60 * 1000;
    while (Date.now() < ende) {
        await new Promise(r => setTimeout(r, 2500));
        try { laeuft = !!(await (await fetch('/api/agent/status')).json()).laeuft; } catch (e) { laeuft = true; }
        if (!laeuft) break;
    }
    _inkluagentVerfolgt = false;
    inkluagentSetStatus('', false);
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
    panel.insertBefore(d, panel.firstChild);
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
    // Hochladefeld (Runde 2, Schritt 1): Dateien oder Webadresse duerfen ohne eigenen Text gesendet werden
    const dateienEl = document.getElementById('inkluagentDateien');
    const webEl = document.getElementById('inkluagentWebadresse');
    const dateien = (inkluagentHochladenAn() && dateienEl && dateienEl.files) ? Array.from(dateienEl.files) : [];
    const webadresse = (inkluagentHochladenAn() && webEl) ? webEl.value.trim() : '';
    if (!message && !dateien.length && !webadresse) {
        inkluagentSetStatus(inkluagentHochladenAn() ? t('Bitte eine Nachricht eingeben oder eine Datei auswählen.') : t('Bitte eine Nachricht eingeben.'), true);
        input.focus();
        return;
    }
    if (message.length > 5000) {
        inkluagentSetStatus(t('Nachricht ist zu lang (max. 5000 Zeichen).'), true);
        return;
    }
    if (webEl && webadresse && !webEl.checkValidity()) {
        inkluagentSetStatus(t('Bitte eine vollständige Webadresse eingeben, zum Beispiel https://www.beispiel.de'), true);
        webEl.focus();
        return;
    }
    sendBtn.disabled = true;
    let hochgeladen = null;
    let verlaufText = message;
    let nachziehenHochladen = false;
    let nachgezogen = false;
    let satz = '';
    if (dateien.length || webadresse) {
        const erg = await inkluagentHochladen(projectId, dateien, webadresse);
        if (!erg) { sendBtn.disabled = false; return; }
        const ergebnisse = erg.ergebnisse || [];
        inkluagentHochgeladenZeigen(projectId, ergebnisse);
        if (dateienEl) dateienEl.value = '';
        if (webEl) webEl.value = '';
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
    inkluagentSetStatus((satz ? satz + ' ' : '') + t('InkluAgent denkt nach...'), false);
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
