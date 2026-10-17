// InkluAgent, Kern: Chat-Bereich (Aufbau, Binden an die Seite, Senden, Status, Bestaetigungs-Karten, Fokus- und Ansageregeln).
//
// Seit 09.10.2026 eine eigene Datei (vorher in backend/templates/app.html, Konzept InkluAgent, Schritt 1); seit 17.10.2026 in Teile
// je Thema aufgeteilt (Steve: „so strukturieren, dass er nicht verbastelt wirkt und wir ihn immer weiter erweitern koennen“) —
// reines Verschieben, kein geaendertes Verhalten. Welche Teile es gibt und in welcher Reihenfolge sie laden: backend/inkluagent/
// skripte.py (die Vorlagen binden genau diese Liste ein, die Tests lesen sie zusammen). Der Kern laedt ZULETZT: an seinem Ende
// steht die einzige Ausfuehrung beim Laden, inkluagentBereichAufbauen(), und die braucht alle Teile.
//
// Geladen VOR dem Seitenskript, nach dashboard.js; auf oberster Ebene stehen nur Funktionen und zwei Merker — einzige Ausfuehrung
// beim Laden ist inkluagentBereichAufbauen() am Dateiende, das ohne festen Bereich sofort zurueckkehrt (Schritt 3). Globale Helfer
// aus dashboard.js: t() (Texte aus window.I18N, 6 Sprachen), announce(), fokusAuf(), eigenerPfad(), gehoertGanzeSeite(),
// zurAnmeldung(); aus den Projektansichten showProject(), window.WERKZEUG_NAMEN, window.Formular. Aufrufer: app.html
// (inkluagentSectionHtml/inkluagentInit), dokument.js, formular.js, uebersetzen.js, abschluss.js, base_app.html (fester Bereich).
// Ausbau Runde 2, Schritt 3 (Schalter AGENT_UEBERALL, window.INKLUAGENT.ueberall): base_app.html laedt die Datei auf JEDER
// Seite und hat einen festen Bereich #inkluagentBereich nach dem Hauptinhalt (manuelle Ansicht: aside nach main; Agentenansicht seit
// der Pruefung 13.10.2026, B17: section im Hauptbereich, neben div#main). Dann baut inkluagentBereichAufbauen()
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
    // Im Arbeitsfenster (eingebettet, Live-Fenster 16.10.2026) gibt es keinen Agenten: ohne den festen Bereich fiel die Ansicht sonst auf
    // den alten eingebetteten Agenten zurueck — und dessen Agentenansicht versteckte die Bildliste im Fenster (Klickprobe 16.10.2026)
    if (window.EINGEBETTET === true) return '';
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
            : '<section class="inkluagent-section" aria-labelledby="inkluagentToggle" data-project-id="' + escHtmlAgent(projectId) + '" data-variante="' + (formular ? 'formular' : 'bilder') + '">')
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
        // Pruefung 13.10.2026 (B21, WCAG 2.5.3): sichtbar steht nur der Platzhalter — derselbe Satz ist jetzt der Name des Feldes
        // (vorher „Nachricht an den InkluAgent“: Sprachsteuerung fand das Feld unter dem sichtbaren Text nicht)
        +       '<label for="inkluagentInput" class="visually-hidden">' + inkluagentPlatzhalter(variante) + '</label>'
        +       '<textarea id="inkluagentInput" rows="2" maxlength="5000" '
        +         'placeholder="' + inkluagentPlatzhalter(variante) + '" '
        +         'aria-describedby="inkluagentHint"></textarea>'
        +       '<div id="inkluagentHint" class="inkluagent-hint">' + t('Enter sendet, Shift+Enter macht einen Zeilenumbruch. Maximal 5000 Zeichen.') + '</div>'
        +       (inkluagentHochladenAn() ? inkluagentHochladeHtml() : '')
        +       '<div class="inkluagent-controls">'
        +         '<button type="submit" id="inkluagentSendBtn" class="btn btn-primary">' + t('Senden') + '</button>'
        // Michael Karbe, Feedback 20261012 - 2, Punkte 3 und 4: „Neues Gespräch“ und „Frühere Gespräche“ in derselben Reihe wie
        // „Senden“ (vorher eine eigene Leiste unter dem Verlauf)
        +         (inkluagentGespraecheAn() ? inkluagentGespraechKnoepfeHtml() : '')
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
// Abgelaufene Sitzung: zur Anmeldung mit Ruecksprung auf diese Seite (dashboard.js, K7); ohne die gemeinsame Datei wie bisher
function inkluagentZurAnmeldung() {
    if (typeof zurAnmeldung === 'function') zurAnmeldung(); else window.location.href = '/login';
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

function inkluagentInit(projectId) {
    // Im Arbeitsfenster (eingebettet) ist die Projektseite nur Buehne: nur das Projekt merken, kein Agent (siehe inkluagentSectionHtml)
    if (window.EINGEBETTET === true) {
        window._inkluagentProjectId = projectId == null ? null : projectId;
        window._inkluagentBereit = Promise.resolve();
        return window._inkluagentBereit;
    }
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
    // Adresse beim Laden, noch mit ?hin= und ?agent= (beide entfernen andere Teile gleich danach) — fuer das Arbeitsfenster (B1);
    // seit Runde 2 (A-N9) mit Sprungmarke, die wie ?hin= zaehlt (nur in der Form einer id — andere Anker bleiben wie bisher weg)
    const anker = /^#[A-Za-z0-9_-]+$/.test(window.location.hash) ? window.location.hash : '';
    window._inkluagentStartAdresse = window.location.pathname + window.location.search + anker;
    const projektseite = /^\/app\/?$/.test(window.location.pathname);
    bereich.innerHTML = inkluagentInhaltHtml(null, projektseite ? 'bilder' : 'konto', true);
    inkluagentBedienungAnbinden();
    // Angefangener Chat-Text ueberlebt Seitenwechsel und Neuladen (Pruefung 13.10.2026, B10)
    inkluagentEingabeHolen();
    window.addEventListener('pagehide', inkluagentEingabeMerken);
    // Seiten ohne Projekt: binden, wenn die Seite steht (ihre Skripte bauen main oft erst noch auf — wichtig fuer die Agentenansicht)
    if (!projektseite) {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => inkluagentInit(null), { once: true });
        else inkluagentInit(null);
    }
}
// Beim Verlassen der Seite (Link, Seitenwechsel durch den Agenten, Neuladen) den angefangenen Chat-Text merken — 30 Minuten, nur in
// diesem Tab; die naechste Seite setzt ihn wieder ins Feld (B10, Schritt 3 der Regel fuer Seitenwechsel)
function inkluagentEingabeMerken() {
    const input = document.getElementById('inkluagentInput');
    try {
        if (input && input.value.trim()) sessionStorage.setItem('inkluagent.eingabe', JSON.stringify({ text: input.value, zeit: Date.now() }));
        else sessionStorage.removeItem('inkluagent.eingabe');
    } catch (e) { /* ohne Speicher: nur diese Seite */ }
}
function inkluagentEingabeHolen() {
    const input = document.getElementById('inkluagentInput');
    let w = null;
    try { w = JSON.parse(sessionStorage.getItem('inkluagent.eingabe') || 'null'); sessionStorage.removeItem('inkluagent.eingabe'); } catch (e) { w = null; }
    if (input && !input.value && w && typeof w.text === 'string' && Date.now() - (Number(w.zeit) || 0) < 30 * 60 * 1000) {
        input.value = w.text.slice(0, 5000);
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
    // Im Arbeitsfenster (eingebettet, Live-Fenster 16.10.2026) ist die Projektseite nur Buehne: der Agent sitzt draussen. Kein Bereich,
    // kein Verfolgen, keine Rueckmeldung — die meldet die Seite draussen, sonst kaeme jede Ansage doppelt.
    if (window.EINGEBETTET === true) { window._inkluagentProjectId = projectId == null ? null : projectId; return; }
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
    // „Hier weitermachen“ (Seite „Frühere Gespräche“) fuehrt mit ?agent=offen hierher: Agent auf, Fokus auf die letzte Nachricht.
    // Pruefung 13.10.2026 (B11): ebenso nach einem Neuladen, das der Agent ausgeloest hat (Sprache, Ansicht) — Fokus auf seine Antwort
    // mit dem Rueckweg, nicht auf die Ueberschrift 1. K17: in der Agentenansicht oeffnen, ohne „offen“ fuer die manuelle Ansicht zu merken.
    const weiter = new URLSearchParams(window.location.search).get('agent') === 'offen';
    let antwortFokus = false;
    try { antwortFokus = sessionStorage.getItem('inkluagent.fokusAntwort') === '1'; sessionStorage.removeItem('inkluagent.fokusAntwort'); } catch (e) {}
    if (weiter || antwortFokus) {
        if (weiter) { try { const u = new URL(window.location.href); u.searchParams.delete('agent'); history.replaceState(history.state, '', u); } catch (e) {} }
        await inkluagentOpen(pid, false, gross);
        const alle = document.querySelectorAll('#inkluagentLog .inkluagent-message');
        const ziel = antwortFokus ? inkluagentLetzteAntwort() : alle[alle.length - 1];
        if (ziel) ziel.focus();
        else { const i = document.getElementById('inkluagentInput'); if (i) i.focus(); }
    } else if (gross) await inkluagentOpen(pid, false, true);
    else if (wasOpen) await inkluagentOpen(pid, false);
    inkluagentLaufVerfolgen();
    inkluagentRueckmeldungenBeobachten();   // wartet noch eine Rueckmeldung (auch von einer anderen Seite angestossen)?
}
// Laufende Antwort (Schritt 3): hat der Server fuer dieses Konto noch eine Antwort in Arbeit (z. B. weil man die Seite gewechselt
// hat), sagt die Statuszeile es und die Seite fragt nach, bis sie da ist — dann neu laden und melden wie eine neue Antwort.
// Pruefung 13.10.2026: solange verfolgt wird, ist Senden gesperrt (K2 — vorher liefen zwei Laeufe, und Poller und eigener Stream
// schrieben gegeneinander). nachAbbruch (K5): der eigene Stream riss ab, der Server arbeitet aber weiter und speichert — dann wird
// verfolgt statt „Verbindungsfehler“ gesagt; ist der Lauf schon fertig, kommt seine Antwort gleich aus dem Verlauf.
let _inkluagentVerfolgt = false;
async function inkluagentLaufVerfolgen(nachAbbruch) {
    if (!inkluagentUeberall() || _inkluagentVerfolgt || window._inkluagentSendet) return;   // diese Seite sendet gerade selbst
    let st = null;
    try { st = await (await fetch('/api/agent/status')).json(); } catch (e) { st = null; }
    if (!nachAbbruch && !(st && st.laeuft)) return;
    _inkluagentVerfolgt = true;
    inkluagentSperreZeigen();
    // Fortschritt nach einem Seitenwechsel (Steve 12.10.2026, Schalter AGENT_ARBEITSFENSTER): der Server nennt Projekt und Schritte
    let projekt = (st && st.projekt != null) ? String(st.projekt) : null;
    try {
        if (st && st.laeuft) {
            inkluagentSetStatus(t('Der InkluAgent arbeitet noch an seiner letzten Antwort …'), false);
            inkluagentArbeitsStart(t('Arbeitet noch an der letzten Antwort …'), true);
            let gesehen = inkluagentFortschrittZeigen(st, 0, true);
            const ende = Date.now() + 10 * 60 * 1000;
            while (Date.now() < ende) {
                await new Promise(r => setTimeout(r, 2000));
                try { st = await (await fetch('/api/agent/status')).json(); } catch (e) { st = { laeuft: true }; }
                if (!st.laeuft) break;
                if (st.projekt != null) projekt = String(st.projekt);
                gesehen = inkluagentFortschrittZeigen(st, gesehen, false);
            }
        }
    } finally {
        _inkluagentVerfolgt = false;
        inkluagentSperreZeigen();
    }
    inkluagentSetStatus('', false);
    if (typeof window.seitenleisteGespraecheLaden === 'function') window.seitenleisteGespraecheLaden();
    const panel = document.getElementById('inkluagentPanel');
    const fokussieren = inkluagentFokusImChat();
    // Hat der Lauf in DIESEM Projekt gearbeitet, zeigt die Ansicht seine Aenderungen (z. B. neue Alt-Texte) ohne Neuladen
    const hier = window._inkluagentProjectId == null ? null : String(window._inkluagentProjectId);
    if (projekt && hier === projekt) await inkluagentAnsichtNachziehen(window._inkluagentProjectId);
    if (panel && !panel.hidden) {
        await inkluagentLoadHistory(window._inkluagentProjectId);
        const alle = document.querySelectorAll('#inkluagentLog .inkluagent-message');
        const beantwortet = alle.length > 0 && alle[alle.length - 1].classList.contains('assistant');
        // Fehlerantworten (intent „error“) speichert der Server nicht (seit Runde 2 auch nach einem Absturz im Lauf, Korrektheit N7) —
        // steht keine Antwort im Verlauf, kam wirklich nichts
        if (nachAbbruch && !beantwortet) {
            inkluagentSetStatus(t('Die Verbindung ist abgebrochen, und eine Antwort kam nicht an. Bitte frag noch einmal.'), true);
            return;
        }
        inkluagentArbeitsEnde(false);
        inkluagentAntwortMelden(inkluagentLetzteAntwort(), fokussieren);
    } else {
        inkluagentArbeitsEnde(false);
        inkluagentAntwortMelden(null, false);
    }
    inkluagentRueckmeldungenBeobachten();   // der verfolgte Lauf kann einen Lauf im Hintergrund gestartet haben
}
// ─── Gesperrt, solange diese Seite sendet oder einen Lauf verfolgt (Pruefung 13.10.2026, B12, K2, K3) ───
// „Senden“, „Neues Gespräch“ (im Agenten und in der Seitenleiste) tragen dann aria-disabled statt disabled: der Fokus bleibt auf dem
// Knopf (ein disabled-Knopf verliert ihn — gemessen, B12), ein Druck darauf sagt, warum es gerade nicht geht. Die Antwort gehoert zum
// Gespraech der Frage; ein Gespraechswechsel waehrend des Laufs liess sie im falschen landen (K3).
function inkluagentBeschaeftigt() {
    return !!(window._inkluagentSendet || _inkluagentVerfolgt);
}
function inkluagentSperreZeigen() {
    const an = inkluagentBeschaeftigt();
    ['inkluagentSendBtn', 'inkluagentNeuesGespraech', 'navNeuesGespraech'].forEach(id => {
        const b = document.getElementById(id);
        if (!b) return;
        if (an || b.dataset.laeuft) b.setAttribute('aria-disabled', 'true'); else b.removeAttribute('aria-disabled');
    });
}
// Fuer Gespraechswechsel (Neues Gespräch, Fortsetzen in der Seitenleiste, „Hier weitermachen“): true = erlaubt, sonst sagt es warum
function inkluagentWechselErlaubt() {
    if (!inkluagentBeschaeftigt()) return true;
    announce(t('Der InkluAgent arbeitet noch an deiner letzten Nachricht. Ein anderes Gespräch geht, sobald seine Antwort da ist.'));
    return false;
}
// Schritte des laufenden Auftrags vom Server zeigen (nach einem Seitenwechsel): im Chatstil im Arbeitsfenster (neue Schritte mit
// „Jeden Schritt ansagen“), sonst in der Statuszeile wie „Ruft gerade auf: …“. Ergebnis: die Zahl der schon gezeigten Schritte.
function inkluagentFortschrittZeigen(st, gesehen, erstesMal) {
    const schritte = Array.isArray(st && st.schritte) ? st.schritte : [];
    const text = x => { const zt = inkluagentZielText(x.ziel); return inkluagentWerkzeugLabel(x.name) + (zt ? ' – ' + zt : ''); };
    if (inkluagentChatstilAktiv()) {
        if (!schritte.length && !st.schreibt) return gesehen;
        const z = inkluagentArbeitsZustand();
        z.laeuft = true;
        z.schritte = schritte.map(x => ({ text: text(x), fertig: !!x.fertig, fehler: !!x.fertig && x.ok === false }));
        const ziel = schritte.map(x => x.ziel).filter(x => x && typeof x === 'object').pop();
        if (ziel) { z.ziel = ziel; z.projekt = st.projekt != null ? String(st.projekt) : null; }
        const laufend = schritte.filter(x => !x.fertig).pop();
        z.stand = st.schreibt ? t('Schreibt die Antwort …')
            : (laufend ? t('Arbeitet an: {schritt}', { schritt: text(laufend) }) : t('Arbeitet noch an der letzten Antwort …'));
        inkluagentArbeitsMerken();
        const neu = schritte.slice(gesehen).map(text);
        if (!erstesMal) neu.forEach(inkluagentSchrittAnsagen);
    } else if (schritte.length && (erstesMal || schritte.length > gesehen)) {
        inkluagentSetStatus(t('Der InkluAgent arbeitet noch an seiner letzten Antwort …') + ' '
            + t('Ruft gerade auf: {w}', { w: text(schritte[schritte.length - 1]) }), false);
    }
    return schritte.length;
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

// Pruefung 4 (Barrierefreiheit M1): eine neue Antwort holt den Fokus nur, wenn er noch im Chat liegt (oder nirgends) —
// wer inzwischen woanders liest oder tippt, bleibt dort und hoert nur „Antwort vom InkluAgent ist da.“ (allgemeine
// Ansage-Region); bei zugeklapptem Chat traegt der Knopf „InkluAgent, neue Antwort“, bis man ihn oeffnet (dann Fokus auf die
// Antwort). Wer im Eingabefeld des Chats schon weiterschreibt, behaelt es.
// Pruefung 13.10.2026 (B4): „im Chat“ heisst im GESPRAECH — das Arbeitsfenster samt eingebetteter Seite zaehlt nicht dazu (vorher holte
// die Antwort den Fokus mitten aus einem Feld der Seite im Rahmen). Eingaben im Rahmen erkennt inkluagentIstEingabe mit.
function inkluagentIstEingabe(el) {
    if (!el) return false;
    if (el.tagName === 'IFRAME') {   // B4: im Rahmen zaehlt das Element dort (nur eigene Herkunft lesbar)
        try { return inkluagentIstEingabe(el.contentDocument && el.contentDocument.activeElement); } catch (e) { return false; }
    }
    if (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
    return el.tagName === 'INPUT' && ['button', 'submit', 'reset', 'checkbox', 'radio', 'file', 'image'].indexOf((el.type || '').toLowerCase()) < 0;
}
function inkluagentFokusImChat() {
    const panel = document.getElementById('inkluagentPanel');
    if (!panel || panel.hidden) return false;
    // B3: liegt das Gespraech unter der Ueberlagerung (inert), kann die Antwort den Fokus nicht nehmen
    const gespraech = panel.querySelector('.inkluagent-gespraech');
    if (gespraech && gespraech.closest('[inert]')) return false;
    const ae = document.activeElement;
    if (!ae || ae === document.body || ae === document.documentElement) return true;
    const fenster = document.getElementById('arbeitsfenster');
    if (fenster && fenster.contains(ae)) return false;   // B4: Arbeitsfenster und Seite im Rahmen
    const sec = panel.closest('.inkluagent-section') || panel;
    if (!sec.contains(ae) && ae !== document.getElementById('inkluagentBereich')) return false;
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
// fokussieren: vorher (VOR dem Nachziehen) mit inkluagentFokusImChat() bestimmt. Pruefung 13.10.2026 (B3): ob der Fokus wirklich
// angekommen ist, wird geprueft (in einem inerten Bereich tut focus() nichts) — sonst kommt die Ansage. Nimmt die Antwort den Fokus
// nicht, merkt sich die Seite sie: „Zurück zum Gespräch“ fuehrt dann zu ihr; bei offener Ueberlagerung sagt die Ansage das.
// Klangspur (Paket 1, frontend/klang.js, Schalter AGENT_KLANG): nur, wenn die Seite sie geladen hat; ob es klingt, entscheidet das Konto
function inkluagentKlang(name) {
    if (window.Klang && typeof window.Klang.spielen === 'function') window.Klang.spielen(name);
}
// ansage (Rueckmeldung, 14.10.2026): die Nachricht kam von selbst (der Agent meldet sich nach einem Lauf im Hintergrund) — dann
// sagt die Seite diesen Text, und ohne neues Element (Agent zu, Verlauf nicht erreichbar) gilt NICHT die letzte alte Antwort: keine
// alte Karte fliegt noch einmal ein, „Zurück zum Gespräch“ fuehrt nicht zu ihr.
function inkluagentAntwortMelden(el, fokussieren, ansage) {
    const rueckmeldung = typeof ansage === 'string';
    // Wartet in der Antwort eine Karte auf dich, fliegt sie ins Gespraech und rauscht hoerbar herein (von rechts in die Mitte), sonst
    // der Klang „fertig“
    const antwort = el || (rueckmeldung ? null : inkluagentLetzteAntwort());
    const karten = antwort ? antwort.querySelectorAll('.inkluagent-karte[data-zustand="offen"]') : [];
    karten.forEach(inkluagentKarteEinfliegen);
    inkluagentKlang(karten.length ? 'karte' : 'fertig');
    if (fokussieren && el) {
        el.focus();
        if (document.activeElement === el) return true;
    }
    if (antwort || !rueckmeldung) window._inkluagentUngelesen = antwort;   // ohne Rueckmeldung wie bisher
    if (rueckmeldung) {
        announce(ansage);
    } else {
        announce(inkluagentUeberlagerungOffen() ? t('Antwort vom InkluAgent ist da. „Zurück zum Gespräch“ führt zu ihr.')
                                                : t('Antwort vom InkluAgent ist da.'));
    }
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
    const main = document.getElementById('main');
    const imChat = !!(sec && ae && sec.contains(ae));
    // Pruefung 13.10.2026 (B4): nur, was das Neuzeichnen ersetzt (der Hauptbereich), braucht Schutz — eine Eingabe im Arbeitsfenster
    // bleibt ohnehin stehen (vorher galt der Rahmen als „im Chat“)
    const imHauptbereich = !!(main && ae && main.contains(ae)) && !imChat;
    if (imHauptbereich && inkluagentIstEingabe(ae)) {
        if (!_inkluagentNachziehenWartet) {
            _inkluagentNachziehenWartet = true;
            ae.addEventListener('focusout', () => setTimeout(() => { _inkluagentNachziehenWartet = false; inkluagentAnsichtNachziehen(projectId); }, 900), { once: true });
        }
        return false;
    }
    const warWo = imHauptbereich ? inkluagentFokusMerken(ae) : null;
    const entwurfEl = document.getElementById('inkluagentInput');
    const entwurf = entwurfEl ? entwurfEl.value : '';
    await showProject(projectId, true);
    inkluagentRahmenNachziehen();   // Live-Fenster: die Oberflaeche im Fenster zeigt dieselben Aenderungen
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
        const el = inkluagentAppendMessage('assistant', data.reply || '', data.werkzeuge || [], data.anhang || [], data.sprache);
        inkluagentEntschieden(data.anhang);
        const wechsel = inkluagentWechselZiel(data.actions);   // B10: derselbe Weg wie nach einer Nachricht
        if (wechsel && await inkluagentWechseln(wechsel, el, fokussieren)) return;
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
        inkluagentRueckmeldungenBeobachten();   // der Knopf kann einen Lauf im Hintergrund gestartet haben
    } catch (e) {
        btn.removeAttribute('aria-disabled');
        if (statusEl) statusEl.textContent = t('Verbindungsfehler. Bitte erneut versuchen.');
    }
}

function inkluagentSetStatus(text, isError) {
    const status = document.getElementById('inkluagentStatus');
    if (!status) return;
    status.textContent = text || '';
    if (isError) status.setAttribute('data-state', 'error');
    else status.removeAttribute('data-state');
    // Pruefung 4 (M1): die Statuszeile liegt im Chat. Pruefung 13.10.2026 (B9, B3): eine Live-Region spricht unabhaengig vom Fokus —
    // global angesagt wird ein Fehler nur, wenn man die Statuszeile NICHT hoert: Chat zugeklappt (hidden) oder unter der Ueberlagerung
    // (inert). Vorher kam er bei offenem Chat und Fokus woanders doppelt, unter der Ueberlagerung gar nicht.
    if (isError && text && status.closest('[hidden], [inert]')) announce(text);
    if (isError && text) inkluagentKlang('fehler');
    if (isError && text && typeof inkluagentArbeitsFehler === 'function') inkluagentArbeitsFehler(text);
}

// Was mit der Nachricht an den Server geht; mit Begleiter auch die Seite, auf der der Nutzer gerade ist (Pfad, Ueberschrift 1)
function inkluagentNachrichtDaten(message, hochgeladen) {
    const d = hochgeladen ? { message, hochgeladen } : { message };
    if (window.INKLUAGENT && window.INKLUAGENT.begleiter) {
        const h1 = document.querySelector('#main h1');
        d.seite = { pfad: window.location.pathname, titel: h1 ? h1.textContent.trim().slice(0, 120) : '' };
    }
    return d;
}

// Pruefung 13.10.2026 (B16): eine Nachricht, die der Server nicht angenommen hat (Tagesgrenze, Kostendeckel, laufender Auftrag,
// Verbindung), steht nicht mehr da, als waere sie angekommen — die Blase sagt es in Worten (nicht nur mit Farbe), und der Text steht
// noch im Eingabefeld (er wird erst geleert, wenn der Server die Nachricht angenommen hat).
// Runde 2 (A-N4): auch die Statuszeile (die Ansage) beginnt mit „Nicht gesendet:“, nennt den Grund und endet mit „Dein Text steht noch im
// Eingabefeld.“ — vorher stand das nur sichtbar in der Blase, die Ansage nannte nur den Grund. Der Fokus bleibt, wo er ist (inkluagentSend).
function inkluagentNichtGesendet(el, mitText, grund) {
    if (el && !el.querySelector('.inkluagent-nicht-gesendet-hinweis')) {
        el.classList.add('inkluagent-nicht-gesendet');
        const p = document.createElement('p');
        p.className = 'inkluagent-nicht-gesendet-hinweis';
        p.textContent = mitText ? t('Nicht gesendet. Dein Text steht noch im Eingabefeld.') : t('Nicht gesendet.');
        el.appendChild(p);
    }
    let satz = String(grund || '').trim();
    if (satz && !/[.!?…]$/.test(satz)) satz += '.';
    inkluagentSetStatus(t('Nicht gesendet: {grund}', { grund: satz }) + (mitText ? ' ' + t('Dein Text steht noch im Eingabefeld.') : ''), true);
}
async function inkluagentSend() {
    const projectId = window._inkluagentProjectId;
    const input = document.getElementById('inkluagentInput');
    const sendBtn = document.getElementById('inkluagentSendBtn');
    const log = document.getElementById('inkluagentLog');
    // Ohne Projekt nur im festen Bereich (Schritt 3): dann geht die Nachricht an /api/agent/nachricht
    const ohneProjekt = projectId == null;
    if ((ohneProjekt && !inkluagentUeberall()) || (!ohneProjekt && !projectId) || !input || !sendBtn || !log) return;
    // Pruefung 13.10.2026 (B12, K2): gesperrt, solange diese Seite sendet oder einen Lauf verfolgt — der Text bleibt im Feld, die Ansage
    // sagt, warum (vorher still ignoriert; waehrend des Verfolgens nach einem Seitenwechsel startete sogar ein zweiter Lauf)
    if (inkluagentBeschaeftigt()) {
        announce(t('Der InkluAgent arbeitet noch an deiner letzten Nachricht. Deine neue Nachricht bleibt im Feld, bis er fertig ist.'));
        return;
    }
    const message = input.value.trim();
    // Hochladefeld (Runde 2, Schritt 1): Dateien duerfen ohne eigenen Text gesendet werden (Webadressen stehen in der Nachricht)
    const dateienEl = document.getElementById('inkluagentDateien');
    const dateien = inkluagentAnhangAktiv() ? (window._inkluagentAnhaenge || []).slice()   // „Dateien anhängen“ (Agentenansicht)
        : ((inkluagentHochladenAn() && dateienEl && dateienEl.files) ? Array.from(dateienEl.files) : []);
    if (!message && !dateien.length && !window._inkluagentNotiz) {
        inkluagentSetStatus((inkluagentHochladenAn() || inkluagentAnhangAktiv()) ? t('Bitte eine Nachricht eingeben oder eine Datei auswählen.') : t('Bitte eine Nachricht eingeben.'), true);
        input.focus();
        return;
    }
    if (message.length > 5000) {
        inkluagentSetStatus(t('Nachricht ist zu lang (max. 5000 Zeichen).'), true);
        return;
    }
    // aria-disabled statt disabled (B12): der Fokus bleibt auf „Senden“
    window._inkluagentSendet = true;
    inkluagentSperreZeigen();
    const chatstil = inkluagentChatstilAktiv();
    let verlaufText = message;
    let nachziehenHochladen = false;
    let nachgezogen = false;
    let satz = '';
    let nutzerEl = null;
    let angenommen = false;
    let verfolgen = '';   // 'lauf' = einen laufenden Auftrag verfolgen (409), 'abbruch' = eigener Stream riss ab (K5)
    try {
        if (dateien.length) {
            if (chatstil) inkluagentArbeitsStart(dateien.length === 1 ? t('Lädt 1 Datei hoch …') : t('Lädt {n} Dateien hoch …', { n: dateien.length }));
            // K12 (Runde 2): Kennung des Stapels — dieselben Dateien nach einem Abbruch noch einmal = dieselbe Kennung
            const erg = await inkluagentHochladen(projectId, dateien, inkluagentHochladenKennung(dateien));
            if (!erg) return;   // der Grund steht in der Statuszeile, die Anhaenge (und ihre Kennung) bleiben
            const ergebnisse = erg.ergebnisse || [];
            inkluagentHochgeladenZeigen(projectId, ergebnisse);
            if (dateienEl) dateienEl.value = '';
            if (inkluagentAnhangAktiv()) inkluagentAnhaengeLeeren(dateien);
            satz = inkluagentHochladenSatz(ergebnisse);
            nachziehenHochladen = !ohneProjekt && ergebnisse.some(e => e.ok && String(e.project_id) === String(projectId));
            // Nichts angekommen und kein eigener Text: keine Nachricht an den Agenten (die Gruende stehen schon am Feld)
            if (!message && !ergebnisse.some(e => e.ok)) {
                inkluagentSetStatus(satz, true);
                return;
            }
            // K4: die Notiz zum Hochladen gilt, bis eine Nachricht angenommen ist — scheitert diese (Tagesgrenze, Verbindung), geht sie
            // mit dem naechsten Senden noch mit; sonst fehlten die Karten zu den wartenden Dateien
            window._inkluagentNotiz = erg.notiz_id ? { id: erg.notiz_id, kurz: erg.kurz || '' } : null;
        }
        const notiz = window._inkluagentNotiz || null;
        // So, wie der gespeicherte Verlauf die Nachricht zeigt: der getippte Text, darunter die Zeile des Servers (notiz.kurz, im Speicher
        // die Spalte zusatz) — inkluagent/storage.anzeige (Runde 2, K-N9: der Verweis nannte die entfernte hochladen.nachricht_fuer_verlauf)
        if (notiz) verlaufText = (message && notiz.kurz) ? message + '\n' + notiz.kurz : (message || notiz.kurz || '');
        inkluagentMarkeVorNachricht(projectId);
        nutzerEl = inkluagentAppendMessage('user', verlaufText);
        log.scrollTop = log.scrollHeight;
        // Eine gebuendelte Ansage: Ergebnis des Hochladens und dass der Agent arbeitet
        // Chatstil: EINE Ansage zum Start, die einzelnen Schritte stehen still im Arbeitsfenster (Steve 12.10.2026)
        inkluagentSetStatus((satz ? satz + ' ' : '') + (chatstil ? t('InkluAgent arbeitet. Was er gerade tut, steht im Arbeitsfenster.')
                                                                  : t('InkluAgent denkt nach...')), false);
        inkluagentArbeitsStart();
        // Stream (NDJSON, 28.08.2026): je Werkzeugaufruf eine Zeile -> Live-Zeile im Status
        // (aria-live, VoiceOver sagt sie an), am Ende die Antwort mit Werkzeugliste.
        const res = await fetch(ohneProjekt ? '/api/agent/nachricht' : '/api/projects/' + projectId + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/x-ndjson' },
            body: JSON.stringify(inkluagentNachrichtDaten(message, notiz ? notiz.id : null))
        });
        if (res.status === 401) { inkluagentZurAnmeldung(); return; }
        if (!res.ok) {
            // Satz des Servers (Tagesgrenze, Kostendeckel, 409 …); ohne ihn ein Satz mit der Fehlernummer statt nur „HTTP 500“
            let detail = t('Der Server hat die Nachricht nicht angenommen (Fehler {n}).', { n: res.status });
            try { const e = await res.json(); if (e && typeof e.detail === 'string') detail = e.detail; } catch (_) {}
            inkluagentNichtGesendet(nutzerEl, !!message, detail);
            // K2: 409 = fuer dieses Konto arbeitet schon ein Auftrag (anderer Tab, vorige Seite) — der Satz kommt vom Server; dessen
            // Antwort verfolgt die Seite dann wie nach einem Seitenwechsel
            if (res.status === 409) verfolgen = 'lauf';
            return;
        }
        // Angenommen (B16): erst jetzt das Feld leeren — und nur, wenn noch derselbe Text drinsteht; die Notiz ist verbraucht (K4), die
        // Kennung des Stapels ebenso (K12)
        angenommen = true;
        window._inkluagentNotiz = null;
        window._inkluagentStapel = null;
        if (input.value.trim() === message) input.value = '';
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
                    // {"type": "ping"} alle 15 s bei Stille (Server, K5) haelt nur die Verbindung — wie jede unbekannte Zeile ignoriert
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
        // K5: der Stream endete ohne Antwort (Proxy-Grenze, Netz) — der Server arbeitet weiter und speichert sie; die Seite verfolgt den
        // Lauf, statt „erneut versuchen“ zu sagen (ein zweites Senden haette einen zweiten Lauf gestartet). Ohne den Agenten auf jeder
        // Seite (kein Status-Endpunkt) bleibt die Meldung.
        if (!data) {
            if (inkluagentUeberall()) verfolgen = 'abbruch';
            else inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true);
            return;
        }
        // Fokusregel (Pruefung 4, M1) VOR allem anderen bestimmen: liegt der Fokus noch im Chat?
        const fokussieren = inkluagentFokusImChat();
        inkluagentEntwurfVerwerfen();   // die fertige Antwort ersetzt den Entwurf aus dem Stream
        const antwortEl = inkluagentAppendMessage('assistant', data.reply || t('(leere Antwort)'), Array.isArray(data.werkzeuge) ? data.werkzeuge : genutzt,
                                                  data.anhang, data.sprache);
        log.scrollTop = log.scrollHeight;
        inkluagentSetStatus('', false);
        if (data.intent === 'error') inkluagentArbeitsFehler(data.reply || '');
        else inkluagentArbeitsEnde(inkluagentWartetAufDich(data.anhang));
        if (chatstil) {
            if (typeof window.seitenleisteGespraecheLaden === 'function') window.seitenleisteGespraecheLaden();
            // Die neue Antwort sichtbar machen (die Seite scrollt, nicht der Verlauf)
            if (antwortEl && antwortEl.scrollIntoView) antwortEl.scrollIntoView({ block: 'nearest' });
        }
        // Fokus auf die Antwort statt zurueck ins Eingabefeld: der Screenreader liest sie
        // vollstaendig und von vorn; Tab fuehrt danach zum Eingabefeld.
        window._inkluagentFokusAntwort = true;
        // Seitenwechsel durch den Agenten: ein Weg mit Fokus- und Eingaberegel (B10, B11, B15)
        const wechsel = inkluagentWechselZiel(data.actions);
        if (wechsel && await inkluagentWechseln(wechsel, antwortEl, fokussieren)) return;
        let nachziehen = inkluagentAktionenKarten(data.actions);
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
        // K5: riss die Verbindung NACH dem Annehmen ab, arbeitet der Server weiter — verfolgen; davor kam die Nachricht nicht an (B16)
        if (angenommen && inkluagentUeberall()) {
            verfolgen = 'abbruch';
        } else if (!angenommen) {
            inkluagentNichtGesendet(nutzerEl, !!message, t('Verbindungsfehler. Bitte erneut versuchen.'));
        } else {
            inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true);
        }
    } finally {
        window._inkluagentSendet = false;
        inkluagentSperreZeigen();
        inkluagentEntwurfVerwerfen();   // abgebrochen: kein halber Entwurf bleibt stehen
        // Sicherheitsnetz: kein Weg darf das Arbeitsfenster auf „arbeitet“ stehen lassen — ausser, der Lauf wird gleich verfolgt
        if (!verfolgen && window._arbeits && window._arbeits.laeuft) inkluagentArbeitsEnde(false);
        // In dieses Projekt hochgeladen, aber die Antwort kam nicht: die Ansicht trotzdem nachziehen (neue Dokumente, Bilder)
        if (nachziehenHochladen && !nachgezogen) inkluagentAnsichtNachziehen(projectId);
        // Runde 2 (A-N4): zurueck ins Eingabefeld nur, wenn der Fokus verloren ging (auf body) und das Gespraech offen ist — sonst bleibt
        // er, wo er ist: auf „Senden“ (aria-disabled haelt ihn, B12) bzw. im Feld (Enter). Vorher sprang er nach „Nicht gesendet“ vom
        // Knopf ins Feld, und VoiceOver las den ganzen Nachrichtentext statt der Statuszeile (die Regel stammte aus der Zeit von disabled).
        const ae = document.activeElement;
        if (window._inkluagentFokusAntwort) { window._inkluagentFokusAntwort = false; }
        else if ((!ae || ae === document.body || ae === document.documentElement) && inkluagentFokusImChat()) { input.focus(); }
        if (verfolgen) inkluagentLaufVerfolgen(verfolgen === 'abbruch');
        else inkluagentRueckmeldungenBeobachten();   // die Antwort kann einen Lauf im Hintergrund gestartet haben
    }
}

// Schritt 3: den festen Bereich gleich beim Laden bauen (das aside steht im Seitenrahmen vor diesem Skript); ohne Schalter oder
// ohne Bereich (app.html bettet ein) passiert nichts.
inkluagentBereichAufbauen();
