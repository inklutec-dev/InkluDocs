// InkluAgent: Chat-Bereich (Aufbau, Verlauf, Senden, Bestaetigungs-Karten, Fokus- und Ansageregeln) und — in der Agentenansicht —
// Chatstil mit Arbeitsfenster (Stand, Schritte, Seiten im Rahmen, Vergroessern).
//
// Seit 09.10.2026 eine eigene Datei (vorher in backend/templates/app.html, Konzept InkluAgent, Schritt 1); seitdem stark ausgebaut
// (Ausbau Runde 1 und 2, Stufe A, Pruefung 13.10.2026 — die Abschnitte nennen jeweils Datum und Grund).
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
// Dateien hochladen. Ergebnis: das Server-Ergebnis oder null (Fehler steht dann in der Statuszeile). Webadressen stehen seit 12.10.2026
// in der Nachricht selbst — das Feld dafuer und sein Parameter sind weg (Pruefung 13.10.2026, HK2).
// kennung (Runde 2, K12): hochladen_id des Stapels (inkluagentHochladenKennung) — derselbe Stapel noch einmal liefert beim Server binnen
// 15 Minuten das fruehere Ergebnis, statt doppelt hochzuladen. Ein Server ohne dieses Feld uebergeht es.
async function inkluagentHochladen(projectId, dateien, kennung) {
    const k = window.INKLUAGENT || {};
    const max = k.max_dateien || 10;
    if (dateien.length > max) {
        inkluagentSetStatus(t('Höchstens {n} Dateien auf einmal. Bitte teile die Auswahl auf.', { n: max }), true);
        return null;
    }
    const fd = new FormData();
    dateien.forEach(f => fd.append('dateien', f, f.name));
    if (projectId != null) fd.append('project_id', String(projectId));
    if (kennung) fd.append('hochladen_id', kennung);
    inkluagentSetStatus(t('Wird hochgeladen …'), false);
    let res;
    try {
        res = await fetch('/api/agent/hochladen', { method: 'POST', body: fd });
    } catch (e) {
        inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true);
        return null;
    }
    if (res.status === 401) { inkluagentZurAnmeldung(); return null; }
    const erg = await res.json().catch(() => ({}));
    if (!res.ok) {   // 429 mit fertigem Satz: Tagesgrenze schon VOR dem Hochladen geprueft (Server, K4)
        inkluagentSetStatus((erg && typeof erg.detail === 'string') ? erg.detail : t('Hochladen nicht möglich.'), true);
        return null;
    }
    return erg;
}
// Pruefung 13.10.2026, Runde 2 (K12): brach die Verbindung beim Hochladen ab (Proxy-Grenze bei grossen Dateien), sagte die Seite
// „Hochladen nicht möglich“, die Anhaenge blieben — und erneutes Senden lud alles noch einmal hoch, obwohl der Server den ersten
// Versuch oft schon fertig hatte. Jetzt bekommt jeder Stapel eine Kennung; dieselben Dateien (dieselben File-Objekte in derselben
// Reihenfolge) behalten sie, bis die Nachricht angenommen ist (inkluagentSend). Ein anderer Stapel bekommt eine neue.
function inkluagentHochladenKennung(dateien) {
    const s = window._inkluagentStapel;
    if (s && s.dateien.length === dateien.length && s.dateien.every((f, i) => f === dateien[i])) return s.id;
    window._inkluagentStapel = { id: inkluagentNeueKennung(), dateien: dateien.slice() };
    return window._inkluagentStapel.id;
}
// 32 Hex-Zeichen aus dem Zufall des Browsers (passt zu [A-Za-z0-9_-], hoechstens 64 Zeichen); getRandomValues gibt es auch ohne HTTPS
function inkluagentNeueKennung() {
    const b = new Uint8Array(16);
    if (window.crypto && window.crypto.getRandomValues) window.crypto.getRandomValues(b);
    else for (let i = 0; i < b.length; i++) b[i] = Math.floor(Math.random() * 256);
    return Array.from(b, x => x.toString(16).padStart(2, '0')).join('');
}
// Abgelaufene Sitzung: zur Anmeldung mit Ruecksprung auf diese Seite (dashboard.js, K7); ohne die gemeinsame Datei wie bisher
function inkluagentZurAnmeldung() {
    if (typeof zurAnmeldung === 'function') zurAnmeldung(); else window.location.href = '/login';
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
function inkluagentAnhangBauen(form, meldungen) {
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
    // Pruefung 13.10.2026 (B21): Grenzen und „ziehen oder einfügen“ stehen sichtbar da — vorher nur fuer Screenreader, Sehende erfuhren
    // sie erst aus einer Fehlermeldung. Runde 2 (HN1): nicht mehr im angehefteten Teil unter den Knoepfen (er wurde dadurch dauerhaft
    // 2 rem hoeher, bei 390 px Breite 27 % des Bildschirms), sondern im Fluss oben in der Beschreibung unter der Eingabe; der Knopf
    // „Dateien anhängen“ verweist weiter per aria-describedby darauf.
    const hinweis = document.createElement('p');
    hinweis.id = 'inkluagentAnhangHinweis';
    hinweis.className = 'inkluagent-hint inkluagent-anhang-hinweis';
    hinweis.textContent = t('PDF, Word (.docx) und Bilder, höchstens {n} Dateien, je bis {mb} MB. Du kannst Dateien auch ins Gespräch ziehen oder einfügen.',
                            { n: k.max_dateien || 10, mb: k.max_mb || 50 });
    const liste = document.createElement('ul');
    liste.id = 'inkluagentAnhaenge';
    liste.className = 'inkluagent-anhaenge';
    liste.setAttribute('aria-label', t('Angehängte Dateien'));
    liste.hidden = true;
    const send = document.getElementById('inkluagentSendBtn');
    if (send && send.parentNode) send.parentNode.insertBefore(knopf, send);   // erst „Dateien anhängen“, dann „Senden“
    else form.appendChild(knopf);
    [datei, liste].forEach(el => form.appendChild(el));
    const beschreibung = form.parentNode ? form.parentNode.querySelector(':scope > .inkluagent-beschreibung') : null;
    if (beschreibung) beschreibung.insertBefore(hinweis, beschreibung.firstChild); else form.appendChild(hinweis);
    // Meldung und Ergebnis des Hochladens stehen im Fluss ueber der Eingabe, nicht im angehefteten Teil (Pruefung 13.10.2026, B6): dort
    // wuchsen sie auf bis zu 60 % des Bildschirms und verdeckten den Fokus
    const meldung = document.createElement('p');
    meldung.id = 'inkluagentAnhangMeldung';
    meldung.className = 'inkluagent-hint inkluagent-anhang-meldung';
    meldung.hidden = true;
    const ziel = meldungen || form;
    const status = document.getElementById('inkluagentStatus');
    const davor = (status && status.parentNode === ziel) ? status : null;
    if (!document.getElementById('inkluagentHochgeladen')) {   // Ergebnis des Hochladens wie beim Feld (inkluagentHochgeladenZeigen)
        const ergebnis = document.createElement('ul');
        ergebnis.id = 'inkluagentHochgeladen';
        ergebnis.className = 'inkluagent-hochgeladen';
        ergebnis.setAttribute('aria-label', t('Ergebnis des Hochladens'));
        ergebnis.hidden = true;
        ziel.insertBefore(ergebnis, davor);
    }
    ziel.insertBefore(meldung, davor);
    inkluagentAblageEinrichten();
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
// Hineinziehen (Maus): Dateien gehen an JEDER Stelle der Seite in die Anhaenge — vorher nur ueber dem Gespraech; daneben oeffnete der
// Browser die Datei als Seite, und die Agentenansicht war weg (Pruefung 13.10.2026, HK7). Die eingebettete Seite im Arbeitsfenster gibt
// Dateien hierher weiter (dashboard.js). Einmal je Seite; wirkt nur, solange es „Dateien anhängen“ gibt. Ablagen der Seite selbst
// (die das Ereignis schon behandeln) gehen vor. Waehrend des Ziehens ist das Gespraech deutlich umrandet.
function inkluagentAblageEinrichten() {
    if (window._inkluagentAblage) return;
    window._inkluagentAblage = true;
    const mitDateien = e => Array.from((e.dataTransfer && e.dataTransfer.types) || []).indexOf('Files') >= 0;
    const markieren = an => {
        const g = document.querySelector('.inkluagent-gespraech');
        if (g) g.classList.toggle('inkluagent-ablage-aktiv', an);
    };
    document.addEventListener('dragover', e => {
        if (e.defaultPrevented || !inkluagentAnhangAktiv() || !mitDateien(e)) return;
        e.preventDefault();
        markieren(true);
    });
    document.addEventListener('dragleave', e => { if (!e.relatedTarget) markieren(false); });
    document.addEventListener('dragend', () => markieren(false));
    document.addEventListener('drop', e => {
        markieren(false);
        if (e.defaultPrevented || !inkluagentAnhangAktiv() || !mitDateien(e)) return;
        e.preventDefault();
        inkluagentAnhaengeHinzu(Array.from(e.dataTransfer.files || []));
    });
}
function inkluagentGroesse(bytes) {
    const lang = document.documentElement.lang || 'de';
    if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)).toLocaleString(lang) + ' KB';
    return (bytes / (1024 * 1024)).toLocaleString(lang, { maximumFractionDigits: 1 }) + ' MB';
}
function inkluagentAnhangZahl(n) {
    return n === 0 ? t('Keine Datei angehängt.') : (n === 1 ? t('1 Datei angehängt.') : t('{n} Dateien angehängt.', { n: n }));
}
function inkluagentAnhangMelden(text, fehler, still) {
    const p = document.getElementById('inkluagentAnhangMeldung');
    if (p) {
        p.textContent = text || '';
        p.hidden = !text;
        if (fehler) p.setAttribute('data-state', 'error'); else p.removeAttribute('data-state');
    }
    if (text && !still) announce(text);
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
// Nach dem Hochladen: nur die Dateien weg, die gerade mitgingen (Pruefung 13.10.2026, B12) — was waehrend des Hochladens neu
// angehaengt wurde, bleibt fuer die naechste Nachricht (vorher loeschte das Leeren es stumm mit). Ohne Liste: alle.
function inkluagentAnhaengeLeeren(gesendet) {
    const weg = Array.isArray(gesendet) ? gesendet : null;
    window._inkluagentAnhaenge = weg ? (window._inkluagentAnhaenge || []).filter(f => weg.indexOf(f) < 0) : [];
    inkluagentAnhaengeZeigen();
    const rest = window._inkluagentAnhaenge.length;
    inkluagentAnhangMelden(rest ? inkluagentAnhangZahl(rest) : '', false, true);
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
    inkluagentAgentenansichtAusblenden();
    // Was die Seite spaeter neu in den Hauptbereich setzt, kommt ebenso weg (einmal je Seite beobachten, gebuendelt). Der Beobachter
    // blendet NUR aus (Pruefung 13.10.2026, K19): vorher baute er bei jeder Aenderung im verborgenen Hauptbereich — Projektansichten
    // fragen alle 2,5 s nach — auch das ganze Arbeitsfenster neu, und ein Fokus auf einem Link dort fiel auf body.
    if (!window._agentenansichtBeobachter && typeof MutationObserver !== 'undefined') {
        let wartet = false;
        window._agentenansichtBeobachter = new MutationObserver(() => {
            if (wartet || window._agentenansichtHierAus) return;
            wartet = true;
            setTimeout(() => { wartet = false; if (!window._agentenansichtHierAus) inkluagentAgentenansichtAusblenden(); }, 50);
        });
        window._agentenansichtBeobachter.observe(main, { childList: true, subtree: true });
    }
    sec.classList.add('inkluagent-gross');
    // Wie ChatGPT/Claude mit Arbeitsfenster (Michael Karbe, Feedback 20261012 - 1, Punkt 6; Schalter AGENT_ARBEITSFENSTER)
    if (inkluagentArbeitsfensterAn()) { inkluagentChatstil(); inkluagentArbeitsZeigen(); }
    // Die Knoepfe des Projekts gibt es nur in der manuellen Ansicht (Hochladen ohne Hochladefeld ebenso): ein Knopf zeigt dieses
    // Projekt fuer diese Seite manuell, ohne die gespeicherte Einstellung zu aendern.
    // Im Chatstil (Arbeitsfenster) kein „nur diese Seite manuell“: Seiten oeffnen sich im Arbeitsfenster, die Ansicht wechselt nur
    // ueber den Umschalter der Seitenleiste (Steve 12.10.2026: „sonst vermischen wir die Ansichten“)
    if (!inkluagentArbeitsfensterAn() && !sec.querySelector('.inkluagent-agentenansicht-hinweis')) {
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
// Im Hauptbereich bleiben nur die Ueberschrift 1, der Agent (wo er eingebettet ist) und was ausdruecklich bleiben soll
// (data-agentenansicht-bleibt, z. B. „Projekt gelöscht … Zur Übersicht“); alles andere bekommt hidden (auch fuer Screenreader weg).
function inkluagentAgentenansichtAusblenden() {
    const main = document.getElementById('main');
    if (!main) return;
    const h1 = main.querySelector('h1');
    const behalten = [document.querySelector('.inkluagent-section'), h1].filter(Boolean);
    (function lauf(el) {
        Array.from(el.children).forEach(k => {
            if (behalten.indexOf(k) >= 0 || k.hasAttribute('data-agentenansicht-bleibt')) return;
            if (behalten.some(b => k.contains(b))) { lauf(k); return; }
            if (/^(SCRIPT|STYLE|TEMPLATE|DIALOG)$/.test(k.tagName) || k.hasAttribute('data-agentenansicht')) return;
            // Steve 12.10.2026 (Staging): Teile, die beim Laden noch versteckt waren und die die Seite erst danach fuellt und
            // einblendet (letzte Projekte auf der Startseite), rutschten durch. Sie werden jetzt auch markiert — die Markierung
            // haelt sie per CSS versteckt, auch wenn die Seite hidden spaeter aufhebt.
            if (k.hidden) { k.setAttribute('data-agentenansicht', 'war-versteckt'); return; }
            k.hidden = true;
            k.setAttribute('data-agentenansicht', 'aus');
        });
    })(main);
}

// Nur fuer diese Seite zurueck zur manuellen Ansicht: alles wieder sichtbar, Fokus auf die Hauptueberschrift
function inkluagentAgentenansichtAufheben() {
    window._agentenansichtHierAus = true;
    document.querySelectorAll('[data-agentenansicht]').forEach(el => {
        if (el.getAttribute('data-agentenansicht') === 'aus') el.hidden = false;   // „war-versteckt“: so, wie die Seite es jetzt will
        el.removeAttribute('data-agentenansicht');
    });
    document.querySelectorAll('.inkluagent-gross').forEach(el => el.classList.remove('inkluagent-gross'));
    document.querySelectorAll('.inkluagent-agentenansicht-hinweis').forEach(el => el.remove());
    const h = document.querySelector('#main h1');
    if (h) fokusAuf(h);   // ohne dauerhaftes tabindex (Pruefung 13.10.2026, B19)
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
// Aufbau umstellen. Die Teile werden VERSCHOBEN, nicht neu gebaut — Verlauf, Entwurf und Ereignisse bleiben. Einmal je Seite: einen Weg
// zurueck gibt es nicht (die manuelle Ansicht kommt ueber den Umschalter der Seitenleiste, mit Neuladen); der fruehere Zweig „zurueck“
// war nie erreichbar und ist weg (Pruefung 13.10.2026, HK2).
function inkluagentChatstil() {
    const sec = document.querySelector('#inkluagentBereich .inkluagent-section');
    const panel = document.getElementById('inkluagentPanel');
    if (!sec || !panel || sec.classList.contains('inkluagent-chatstil')) return;
    const form = document.getElementById('inkluagentForm');
    const log = document.getElementById('inkluagentLog');
    const frueher = panel.querySelector('.inkluagent-frueher');
    const trenner = panel.querySelector('.inkluagent-trenner');
    const beschreibung = panel.querySelector('.inkluagent-beschreibung');
    sec.classList.add('inkluagent-chatstil');
    document.documentElement.classList.add('agentenansicht-chatstil');
    // Chat-Start (leeres Gespraech mittig) nur dort, wo der Agent die Seite ersetzt — nicht unter dem Inhalt anderer Seiten
    document.documentElement.classList.add('agentenansicht-chatstart');
    // Steve 12.10.2026: in der Agentenansicht laesst sich der Agent nicht auf- und zuklappen — die Ueberschrift 2 „InkluAgent“
    // ist dann schlicht (der Knopf ist weg, der Bereich immer offen)
    const toggle = document.getElementById('inkluagentToggle');
    const kopf = document.getElementById('inkluagentHeading');
    if (toggle && kopf) {
        if (!kopf.querySelector('.inkluagent-titel')) {
            const titel = document.createElement('span');
            titel.className = 'inkluagent-titel';
            titel.textContent = t('InkluAgent');
            kopf.appendChild(titel);
        }
        toggle.hidden = true;
        panel.hidden = false;
    }
    const spalten = document.createElement('div');
    spalten.className = 'inkluagent-spalten';
    const gespraech = document.createElement('div');
    gespraech.className = 'inkluagent-gespraech';
    // Meldungen (Ergebnis des Hochladens, Anhang-Meldung, Statuszeile) im Fluss zwischen Verlauf und Eingabe; angeheftet bleiben nur
    // Feld und Knoepfe (Pruefung 13.10.2026, B6)
    const meldungen = document.createElement('div');
    meldungen.className = 'inkluagent-meldungen';
    ['inkluagentHochgeladen', 'inkluagentStatus'].forEach(id => { const el = document.getElementById(id); if (el) meldungen.appendChild(el); });
    [frueher, log, meldungen, form, trenner, beschreibung].forEach(el => { if (el) gespraech.appendChild(el); });
    // Hier scrollt die Seite, nicht der Verlauf: kein leerer Tab-Stopp (B22; tabindex="0" braucht nur der scrollbare Kasten der
    // manuellen Ansicht, axe scrollable-region-focusable)
    if (log) log.removeAttribute('tabindex');
    spalten.appendChild(gespraech);
    spalten.appendChild(inkluagentArbeitsfensterBauen());
    panel.appendChild(spalten);
    inkluagentAnhangBauen(form, meldungen);   // „Dateien anhängen“ (nur ohne das Feld „Dateien mitschicken“)
    inkluagentEingabeHoeheBeobachten(form);
    document.addEventListener('click', inkluagentLinkImFenster);   // B1: interne Links oeffnen im Arbeitsfenster
}
// Wie hoch die angeheftete Eingabe gerade ist, steht in --agent-eingabe-hoehe; daraus folgt der Platz, den der Browser unter dem Fokus
// freihaelt (scroll-padding-bottom, style.css). Vorher eine feste Zahl fuer eine wachsende Hoehe (Pruefung 13.10.2026, B6, WCAG 2.4.11).
function inkluagentEingabeHoeheBeobachten(form) {
    if (!form || typeof ResizeObserver === 'undefined') return;
    new ResizeObserver(() => {
        document.documentElement.style.setProperty('--agent-eingabe-hoehe', Math.ceil(form.getBoundingClientRect().height) + 'px');
    }).observe(form);
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
        // Vergroessern (13.10.2026): Umschaltknopf mit aria-pressed, der Fokus bleibt; schmal liegt das Fenster ueber dem Gespraech
        + '<div class="arbeitsfenster-werkzeuge">'
        +   '<button type="button" id="arbeitsGross" class="btn btn-secondary" aria-pressed="false">' + t('Arbeitsfenster vergrößern') + '</button>'
        +   '<button type="button" id="arbeitsZurueck" class="btn btn-secondary">' + t('Zurück zum Gespräch') + '</button>'
        + '</div>'
        + '<p id="arbeitsStand" class="arbeitsfenster-stand"></p>'
        + '<progress id="arbeitsFortschritt" class="arbeitsfenster-fortschritt" aria-labelledby="arbeitsStand" hidden></progress>'
        // Seiten im Arbeitsfenster (13.10.2026, Steve: wie die Vorschau in InkluEdit): die echte Seite, eingebettet und bedienbar
        + '<div id="arbeitsSeite" class="arbeitsfenster-seite" hidden>'
        +   '<div class="arbeitsfenster-seite-kopf"><h3 id="arbeitsSeiteTitel"></h3>'
        +     '<button type="button" id="arbeitsSeiteZu" class="btn btn-secondary">' + t('Seite schließen') + '</button></div>'
        +   '<iframe id="arbeitsSeiteRahmen" src="about:blank"></iframe>'
        + '</div>'
        // Runde 2 (K-N1): kam eine Seite nicht (Fehlerantwort), steht hier, dass sie sich gerade nicht oeffnen laesst
        + '<p id="arbeitsSeiteMeldung" class="arbeitsfenster-seite-meldung" hidden></p>'
        + '<p id="arbeitsNotiz" class="arbeitsfenster-notiz" hidden></p>'
        + '<button type="button" id="arbeitsVorlesen" class="btn btn-secondary">' + t('Stand vorlesen') + '</button>'
        // Livestream (Steve 12.10.2026): jeden Schritt ansagen — von Anfang an an, die Wahl bleibt in diesem Browser
        + '<p class="arbeitsfenster-ansagen"><input type="checkbox" id="arbeitsAnsagen"> <label for="arbeitsAnsagen">'
        +   t('Jeden Schritt ansagen') + '</label></p>'
        + '<dl id="arbeitsAngaben" class="arbeitsfenster-angaben" hidden></dl>'
        // Ueberblick, solange er nichts tut (Steve 12.10.2026): Guthaben, was laeuft, Express, zuletzt bearbeitet
        + '<div id="arbeitsUeberblick" class="arbeitsfenster-ueberblick" hidden><h3>' + t('Überblick') + '</h3><dl></dl></div>'
        // alt: der vorhandene Alt-Text des Bildes, wenn der Server ihn mitschickt (HB6), sonst leer — die Bildunterschrift nennt das Bild
        + '<figure id="arbeitsBild" class="arbeitsfenster-bild" hidden><img alt=""><figcaption></figcaption></figure>'
        + '<div id="arbeitsErgebnis" class="arbeitsfenster-ergebnis" hidden><h3>' + t('Zuletzt fertig') + '</h3><p></p></div>'
        + '<h3 id="arbeitsSchritteTitel" hidden>' + t('Schritte') + '</h3>'
        + '<ol id="arbeitsSchritte" class="arbeitsfenster-schritte" aria-labelledby="arbeitsSchritteTitel" hidden></ol>';
    s.querySelector('#arbeitsVorlesen').addEventListener('click', inkluagentStandVorlesen);
    setTimeout(inkluagentUeberblickLaden, 0);
    s.querySelector('#arbeitsGross').addEventListener('click', () => inkluagentFensterGross());
    s.querySelector('#arbeitsZurueck').addEventListener('click', inkluagentZumGespraech);
    s.querySelector('#arbeitsSeiteZu').addEventListener('click', inkluagentSeiteSchliessen);
    // Jedes Dokument, das der Rahmen laedt, meldet sich (Pruefung 13.10.2026, K6/K7/B14) — dauerhaft, nicht nur beim Oeffnen
    const rahmen = s.querySelector('#arbeitsSeiteRahmen');
    rahmen.addEventListener('load', () => inkluagentRahmenGeladen(rahmen));
    inkluagentUeberlagerungEinrichten();
    setTimeout(inkluagentFensterWiederherstellen, 0);
    const ansagen = s.querySelector('#arbeitsAnsagen');
    ansagen.checked = inkluagentSchritteAnsagen();
    ansagen.addEventListener('change', () => {
        try { localStorage.setItem('inkluagent.schritte_ansagen', ansagen.checked ? 'an' : 'aus'); } catch (e) { /* nur diese Seite */ }
    });
    return s;
}
// ─── Seiten im Arbeitsfenster (13.10.2026) ───
// Die Agentenansicht zeigt nur Gespraech + Arbeitsfenster. Wohin der Nutzer will (Navigation, Links, Hinfuehren), oeffnet sich als echte
// Seite im Rahmen des Arbeitsfensters (eingebettet: ohne Seitenleiste, Fusszeile, Agent — i18n.eingebettet, base_app.html und seit
// Runde 2 auch base_oeffentlich.html) und ist dort bedienbar. Welche Seiten das sind, sagt EINE Liste des Servers
// (window.INKLUAGENT.fensterSeiten, dashboard.js fensterSeite); alles andere — Projekte und Startseite (dort ist der Agent selbst),
// Sprachwechsel, Anmeldung, Rechtstexte — oeffnet als ganze Seite.
// Pruefung 13.10.2026: EIN Zustand window._arbeitsSeite ({offen, pfad, name, fokus, hin}); das geladene Dokument ist die Wahrheit
// (K6/B14) — Titel, Name des Rahmens und gemerkte Seite folgen ihm bei jeder Navigation im Rahmen.
function inkluagentSeiteImFensterMoeglich(pfad) {
    if (!inkluagentChatstilAktiv() || typeof pfad !== 'string' || !eigenerPfad(pfad)) return false;
    if (!/^\/[A-Za-z0-9\/_-]*(\?[A-Za-z0-9=&_%-]*)?(#[A-Za-z0-9_-]*)?$/.test(pfad)) return false;   // mit Anker („Zur Ablage“)
    return fensterSeite(pfad);   // Runde 2 (S-N6): Liste des Servers; ohne sie die bisherige Regel (dashboard.js)
}
// Pfad ohne die genannten Parameter (eingebettet, hin, agent) — so wird gemerkt und verglichen
function inkluagentPfadOhne(pfad, namen) {
    const s = String(pfad);
    const h = s.indexOf('#');
    const anker = h >= 0 ? s.slice(h) : '';
    const ohneAnker = h >= 0 ? s.slice(0, h) : s;
    const i = ohneAnker.indexOf('?');
    if (i < 0) return s;
    const p = new URLSearchParams(ohneAnker.slice(i + 1));
    namen.forEach(n => p.delete(n));
    const q = p.toString();
    return ohneAnker.slice(0, i) + (q ? '?' + q : '') + anker;
}
// Pruefung 13.10.2026 (B1): EINE Stelle entscheidet, wo eine interne Seite in der Agentenansicht erscheint — jeder Link dieser Seite
// (Navigation, „Alles, was der InkluAgent kann“, „Zur Ablage“, „Gespräch … lesen“, Hinfuehren, Fusszeile …), der ins Arbeitsfenster
// passt, oeffnet dort. Vorher taten das nur die Eintraege der Seitenleiste und Hinfuehren-Links; alle anderen luden die ganze Seite, deren
// Inhalt die Agentenansicht dann ausblendete. Eigene Klick-Handler (Gespraech fortsetzen) und Strg/Cmd-Klick gehen vor.
function inkluagentLinkImFenster(e) {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = (e.target && e.target.closest) ? e.target.closest('a[href]') : null;
    if (!a || a.hasAttribute('download') || (a.target && a.target !== '_self')) return;
    const href = a.getAttribute('href') || '';
    if (!inkluagentSeiteImFensterMoeglich(href)) return;
    e.preventDefault();
    inkluagentSeiteImFenster(href, a.textContent.trim());
}
// Seite im Fenster oeffnen. still: ohne Fokus (Neuladen, Seite selbst); sonst geht der Fokus in die Seite, sobald sie geladen ist — auf
// ihre Ueberschrift 1 bzw. das Ziel von ?hin= oder einer Sprungmarke. Keine eigene Ansage mehr (B15): Name des Rahmens und Ueberschrift
// sagen es. Runde 2 (A-N9): eine Sprungmarke („Zur Ablage“ → /ablage?projekt=1#ausgabe-3) zaehlt wie ?hin= — das Ziel bestimmt die
// Seite, nicht vorher noch die Ueberschrift 1 (vorher zwei Fokusspruenge: erst „Meine Ablage“, gleich danach der Eintrag).
function inkluagentSeiteImFenster(pfad, name, still) {
    const f = document.getElementById('arbeitsfenster');
    if (!f || !inkluagentSeiteImFensterMoeglich(pfad)) { if (eigenerPfad(pfad)) window.location.href = pfad; return; }
    // laedt: bis das Dokument da ist — ein spaetes load des leeren Startrahmens (about:blank) schliesst die Seite dann nicht
    window._arbeitsSeite = { offen: true, pfad: inkluagentPfadOhne(pfad, ['hin', 'eingebettet']),
                             name: String(name || '').trim() || pfad.split(/[?#]/)[0], fokus: !still,
                             hin: /[?&]hin=/.test(pfad) || /#./.test(pfad), laedt: true };
    f.querySelector('#arbeitsSeiteMeldung').hidden = true;   // K-N1: eine fruehere „lässt sich nicht öffnen“-Meldung gilt nicht mehr
    inkluagentSeiteZeigen();
    const h = pfad.indexOf('#');
    const ohneAnker = h >= 0 ? pfad.slice(0, h) : pfad;
    inkluagentRahmenLaden(f.querySelector('#arbeitsSeiteRahmen'),
                          ohneAnker + (ohneAnker.indexOf('?') >= 0 ? '&' : '?') + 'eingebettet=1' + (h >= 0 ? pfad.slice(h) : ''));
}
// K6: laden OHNE Eintrag im Verlauf des Tabs — vorher legte jedes src= einen an (und „Seite schließen“ noch einen); die Zurueck-Taste
// bewegte dann erst den oft verborgenen Rahmen. ?eingebettet=1 nur fuer Browser ohne Fetch-Metadaten (i18n.eingebettet).
function inkluagentRahmenLaden(rahmen, url) {
    if (!rahmen) return;
    try { rahmen.contentWindow.location.replace(url); } catch (e) { rahmen.src = url; }
}
// Sichtbarer Stand aus window._arbeitsSeite: Ueberschrift und Name des Rahmens
function inkluagentSeiteZeigen() {
    const f = document.getElementById('arbeitsfenster');
    const s = window._arbeitsSeite;
    if (!f || !s) return;
    const titel = f.querySelector('#arbeitsSeiteTitel');
    const neu = t('Seite: {name}', { name: s.name });
    if (titel.textContent !== neu) titel.textContent = neu;
    f.querySelector('#arbeitsSeiteRahmen').title = t('Arbeitsfenster: {name}', { name: s.name });
    f.querySelector('#arbeitsSeite').hidden = false;
    inkluagentArbeitsZeigen();
}
// Merker fuers Neuladen (mit der Seite, auf der man war: B1). Runde 2 (K-N1): erst, wenn die Seite eingebettet GELADEN ist
// (inkluagentRahmenGeladen) — vorher wurde sie schon beim Oeffnen gemerkt; kam dann eine Fehlerseite, oeffnete jede weitere Seite
// sie wieder (und jeder Versuch erneuerte die Zeit, die 2-Stunden-Grenze lief nie ab).
function inkluagentSeiteMerken() {
    const s = window._arbeitsSeite;
    if (!s) return;
    try {
        sessionStorage.setItem('inkluagent.fenster', JSON.stringify({ pfad: s.pfad, name: s.name, zeit: Date.now(),
            seite: inkluagentPfadOhne(window.location.pathname + window.location.search, ['agent', 'hin']) }));
    } catch (e) { /* nur diese Seite */ }
}
// Jedes Dokument, das der Rahmen laedt (geoeffnet, Link darin, Zurueck-Taste, Weiterleitung), kommt hier an (Pruefung 13.10.2026).
// Runde 2 (K-N1): vier Faelle, sauber getrennt — vorher warf jede Fehlerantwort (404/403 als JSON, 500 als Text) im Rahmen den Nutzer
// aus der Agentenansicht auf die rohe Fehlerseite, weil sie wie „Seite ohne eingebettete Fassung“ behandelt wurde.
// 1. Erfolg: die eingebettete Fassung einer Fensterseite (window.EINGEBETTET === true). K6/B14: Pfad und Ueberschrift 1 (sonst der
//    Seitentitel) werden Ueberschrift, Name des Rahmens und gemerkte Seite. Der Fokus geht in die Seite, wenn sie mit Fokus geoeffnet
//    wurde oder er schon im Rahmen stand (ein Link darin).
// 2. Anmeldung (Sitzung abgelaufen, K7): ganze Seite mit Ruecksprung auf DIESE Seite; die Seite im Fenster bleibt gemerkt.
// 3. Ganze Seite: eine Seite der App, die nicht ins Fenster gehoert (nicht in der Liste: Projekt, Startseite, Rechtstexte) oder sich
//    ausdruecklich nicht einbettet (window.EINGEBETTET === false aus base_app.html/base_oeffentlich.html). Merker weg, dann wechseln.
// 4. Sonst — Fehlerantwort, kein HTML, vom Server nicht eingerahmt, fremde Herkunft: NICHT wegnavigieren. Die Seite im Fenster
//    schliessen, den Merker loeschen und es im Fenster sagen (inkluagentSeiteNichtGeladen).
// Leerer Rahmen (about:blank) = keine Seite offen.
function inkluagentRahmenGeladen(rahmen) {
    const f = document.getElementById('arbeitsfenster');
    if (!f) return;
    let win = null;
    let doc = null;
    try { win = rahmen.contentWindow; doc = rahmen.contentDocument; } catch (e) { doc = null; }
    const loc = doc && doc.location;
    if (!doc || !loc) { inkluagentSeiteNichtGeladen(rahmen); return; }   // fremd oder vom Server nicht eingerahmt (DENY)
    if (loc.href === 'about:blank') { if (window._arbeitsSeite && !window._arbeitsSeite.laedt) inkluagentSeiteZustandZu(); return; }
    const pfad = inkluagentPfadOhne(loc.pathname + loc.search, ['eingebettet', 'hin']);
    const eigene = loc.origin === window.location.origin;
    const eingebettet = win ? win.EINGEBETTET : undefined;   // true, false (Seite der App ohne Einbettung) oder fehlt (keine Seite der App)
    if (eigene && eingebettet === true && fensterSeite(loc.pathname)) {                    // 1. Erfolg
        const h1 = doc.querySelector('#main h1');
        const titel = (h1 && h1.textContent.trim()) || String(doc.title || '').split(' · ')[0].trim();
        const vorher = window._arbeitsSeite || {};
        const fokus = !!vorher.fokus || document.activeElement === rahmen;
        window._arbeitsSeite = { offen: true, pfad: pfad, name: titel || vorher.name || loc.pathname, fokus: false, hin: false };
        inkluagentSeiteZeigen();
        inkluagentSeiteMerken();
        if (fokus) inkluagentRahmenFokus(rahmen, !!(vorher.fokus && vorher.hin), loc.hash);
        return;
    }
    if (eigene && /^\/login(\/|$)/.test(loc.pathname)) { inkluagentZurAnmeldung(); return; }   // 2. Anmeldung
    const hier = inkluagentPfadOhne(window.location.pathname + window.location.search, ['agent', 'hin']);
    if (eigene && eigenerPfad(pfad) && pfad !== hier && (!fensterSeite(loc.pathname) || eingebettet === false)) {   // 3. ganze Seite
        inkluagentSeiteZustandZu();
        window.location.href = pfad + loc.hash;
        return;
    }
    // 4. (auch: die ganze Seite waere die, auf der man schon ist — kein Neuladen im Kreis)
    inkluagentSeiteNichtGeladen(rahmen);
}
// Runde 2 (K-N1): die Seite im Fenster kam nicht. Sie wird geschlossen und vergessen; im Arbeitsfenster steht, dass sie sich gerade nicht
// oeffnen laesst — mit ihrem Namen, wenn wir sie selbst geoeffnet haben (bei einem Link in der Seite kennen wir das Ziel nicht).
// Hoeren: stand der Fokus im Rahmen (Link darin), kommt er auf den Satz (er wird vorgelesen, nichts faellt auf body); wurde die Seite
// gerade mit Fokus geoeffnet (Navigation, Link im Gespraech), eine Ansage — der Fokus bleibt, wo er ist; beim stillen Wiederherstellen
// nach dem Neuladen nur der sichtbare Satz.
function inkluagentSeiteNichtGeladen(rahmen) {
    const f = document.getElementById('arbeitsfenster');
    if (!f) return;
    const vorher = window._arbeitsSeite || {};
    const imRahmen = document.activeElement === rahmen;
    inkluagentSeiteZustandZu();
    inkluagentRahmenLaden(rahmen, 'about:blank');
    const text = (vorher.laedt && vorher.name) ? t('„{name}“ lässt sich gerade nicht öffnen. Bitte versuch es später noch einmal.', { name: vorher.name })
                                               : t('Diese Seite lässt sich gerade nicht öffnen. Bitte versuch es später noch einmal.');
    const m = f.querySelector('#arbeitsSeiteMeldung');
    m.textContent = text;
    m.hidden = false;
    if (imRahmen) fokusAuf(m);
    else if (vorher.fokus) announce(text);
}
// Fokus in die Seite im Fenster: Ueberschrift 1 (ohne dauerhaftes tabindex, B19). selbst: die Seite setzt ihr Ziel selbst (?hin= ueber
// dashboard.js, Sprungmarke). Runde 2 (A-N9): bei einer Sprungmarke bekommt ihr Ziel den Fokus, wenn es beim Laden schon da ist (ein
// Sprung); baut die Seite es erst (Ablage: #ausgabe-… nach dem Laden der Liste), setzt sie ihn selbst (fokusAuf). Nur ohne Ueberschrift
// eine Ansage.
function inkluagentRahmenFokus(rahmen, selbst, anker) {
    let doc = null;
    let win = null;
    try { win = rahmen.contentWindow; win.focus(); doc = rahmen.contentDocument; } catch (e) { return; }
    if (!doc) return;
    const auf = typeof win.fokusAuf === 'function' ? win.fokusAuf : fokusAuf;
    if (anker && anker.length > 1) {
        let ziel = null;
        try { ziel = doc.getElementById(decodeURIComponent(anker.slice(1))); } catch (e) { ziel = null; }
        if (ziel) auf(ziel);
        return;
    }
    if (selbst) return;
    const h1 = doc.querySelector('#main h1');
    if (h1) { auf(h1); return; }
    announce(t('Im Arbeitsfenster geöffnet: {name}', { name: (window._arbeitsSeite || {}).name || '' }));
}
// Keine Seite mehr offen (nur der Zustand — laden tut der Aufrufer)
function inkluagentSeiteZustandZu() {
    const f = document.getElementById('arbeitsfenster');
    window._arbeitsSeite = null;
    if (!f) return;
    f.querySelector('#arbeitsSeite').hidden = true;
    f.querySelector('#arbeitsSeiteRahmen').removeAttribute('title');
    f.querySelector('#arbeitsSeiteTitel').textContent = '';
    try { sessionStorage.removeItem('inkluagent.fenster'); } catch (e) { /* egal */ }
    inkluagentArbeitsZeigen();
}
function inkluagentSeiteSchliessen() {
    const f = document.getElementById('arbeitsfenster');
    if (!f) return;
    inkluagentSeiteZustandZu();
    inkluagentRahmenLaden(f.querySelector('#arbeitsSeiteRahmen'), 'about:blank');
    if (document.documentElement.classList.contains('arbeitsfenster-gross')) inkluagentFensterGross(false);
    announce(t('Seite im Arbeitsfenster geschlossen.'));
    const input = document.getElementById('inkluagentInput');
    if (input) input.focus();
}
// Nach einem Neuladen (Sprache, Ansicht, Seitenwechsel) wieder dieselbe Seite und Groesse — still, ohne Fokus und Ansage.
// Pruefung 13.10.2026 (B1): wurde eine Seite, deren Inhalt die Agentenansicht ausblendet (Einstellungen, Hilfe, ein Gespraech …), direkt
// geladen (Lesezeichen, Link — nicht dieselbe Seite wie beim Merken), oeffnet sie selbst im Arbeitsfenster: Ueberschrift 1 und Inhalt
// passen dann zusammen. Mit ?hin= (Hinfuehren in einem neuen Tab) — seit Runde 2 (A-N9) auch mit einer Sprungmarke — geht der Fokus
// wie beim Hinfuehren aufs Ziel. Laesst sich die gemerkte Seite nicht mehr laden, vergisst das Fenster sie (K-N1).
function inkluagentFensterWiederherstellen() {
    let w = null;
    try { w = JSON.parse(sessionStorage.getItem('inkluagent.fenster') || 'null'); } catch (e) { w = null; }
    const frisch = !!(w && typeof w.pfad === 'string' && Date.now() - (Number(w.zeit) || 0) < 2 * 60 * 60 * 1000);
    const hier = inkluagentPfadOhne(window._inkluagentStartAdresse || (window.location.pathname + window.location.search), ['agent']);
    if (inkluagentSeiteImFensterMoeglich(hier) && !(frisch && w.seite === inkluagentPfadOhne(hier.split('#')[0], ['hin']))) {
        const h1 = document.querySelector('#main h1');
        inkluagentSeiteImFenster(hier, h1 ? h1.textContent.trim() : '', !(/[?&]hin=/.test(hier) || /#./.test(hier)));
    } else if (frisch) {
        inkluagentSeiteImFenster(w.pfad, w.name, true);
    }
    let gross = '';
    try { gross = sessionStorage.getItem('inkluagent.fenster.gross') || ''; } catch (e) { gross = ''; }
    if (gross === '1' && !inkluagentFensterSchmal()) inkluagentFensterGross(true);
}

// ─── „Arbeitsfenster vergrößern“ und die Ueberlagerung (13.10.2026; Pruefung 13.10.2026, B2/K8, B3, B7) ───
// EIN Zustand: die Klasse arbeitsfenster-gross am html (Knopf mit aria-pressed). Breit heisst das: Fenster gross, Gespraech schmal. Schmal
// (unter 1100 px, dieselbe Grenze wie style.css) liegt das Fenster ueber allem — dann ist der Rest inert, das Fenster traegt die Rolle
// Dialog, und Escape schliesst. Alles davon wird aus dem Zustand ABGELEITET (inkluagentUeberlagerungAnwenden), beim Umschalten und bei
// jeder Breitenaenderung (Zoom, iPad drehen). Vorher wurde inert nur im Moment des Klicks gesetzt und lief der Darstellung davon.
// Warum kein natives <dialog> (Steve mag es, HTML vor ARIA): im Browser gemessen (lokal/probe_pr13_dialog_versuch.py, Chromium) laedt ein
// Rahmen neu, wenn man ihn in einen Dialog verschiebt (die Seite im Fenster verloere ihre Eingaben); Escape im Rahmen erreicht den
// Dialog nicht; und die Ansage-Region ausserhalb faellt bei showModal() aus dem Barrierefreiheitsbaum. Ein dauerhafter <dialog> waere
// breit falsch benannt. Darum: der Abschnitt bleibt ein Abschnitt und bekommt nur als Ueberlagerung role="dialog" (benannt ueber seine
// Ueberschrift „Arbeitsfenster“). Die Modalitaet macht inert (HTML).
// Pruefung 13.10.2026, Runde 2 (A-N1, gemessen): KEIN aria-modal mehr. Es tat zusaetzlich zu inert dasselbe, aber ohne Ausnahme — sobald
// sich der Fokus im Dialog bewegte, stand die Ansage-Region als „ignoriert, activeAriaModalDialog“ im Barrierefreiheitsbaum: fertige
// Antwort, Fehler und Schritte blieben bei offener Ueberlagerung stumm (auf dem iPhone, wo das Fenster immer schmal ist, der Normalfall).
// Ohne aria-modal bleibt die Region im Baum; Rolle, Name und inert bleiben (lokal/probe_r2_a11y_baum.py misst es).
function inkluagentFensterSchmal() {
    return !!(window.matchMedia && window.matchMedia('(max-width: 1099px)').matches);
}
function inkluagentUeberlagerungOffen() {
    return document.documentElement.classList.contains('arbeitsfenster-gross') && inkluagentFensterSchmal();
}
function inkluagentFensterGross(an) {
    const knopf = document.getElementById('arbeitsGross');
    if (!knopf) return;
    if (typeof an !== 'boolean') an = knopf.getAttribute('aria-pressed') !== 'true';
    knopf.setAttribute('aria-pressed', an ? 'true' : 'false');
    document.documentElement.classList.toggle('arbeitsfenster-gross', an);
    try { sessionStorage.setItem('inkluagent.fenster.gross', an ? '1' : ''); } catch (e) { /* egal */ }
    inkluagentUeberlagerungAnwenden();
}
// Leitet aus dem Zustand ab: inert auf allem ausser dem Fenster und der Ansage-Region (sonst hoerte man bei offener Ueberlagerung keine
// Ansage, B3), Rolle Dialog (ohne aria-modal, A-N1). Inert wird auf den Geschwistern entlang des Wegs vom Fenster bis body gesetzt — so
// sind auch die Sprunglinks und die Fusszeile erfasst (B7). Was schon vorher inert war, bleibt unberuehrt.
// Runde 2 (A-N2): beim Oeffnen eine kurze Ansage, was passiert ist und wie es zurueckgeht. Der Fokus bleibt auf dem Umschaltknopf — er
// liegt schon im Dialog, darum sagten Screenreader „Dialog“ nicht von selbst (das tun sie nur, wenn der Fokus von aussen hineinkommt).
function inkluagentUeberlagerungAnwenden() {
    const f = document.getElementById('arbeitsfenster');
    (window._arbeitsInert || []).forEach(el => { el.inert = false; });
    window._arbeitsInert = [];
    if (!f) return;
    const ueber = inkluagentUeberlagerungOffen();
    const war = !!window._arbeitsUeberlagerungOffen;
    window._arbeitsUeberlagerungOffen = ueber;
    if (ueber) f.setAttribute('role', 'dialog'); else f.removeAttribute('role');
    if (!ueber) return;
    const ansage = document.getElementById('liveRegion');
    for (let el = f; el && el.parentElement && el !== document.body; el = el.parentElement) {
        Array.from(el.parentElement.children).forEach(k => {
            if (k === el || k === ansage || k.inert || /^(SCRIPT|STYLE|TEMPLATE)$/.test(k.tagName)) return;
            k.inert = true;
            window._arbeitsInert.push(k);
        });
    }
    // Lag der Fokus jetzt hinter der Ueberlagerung (Breite geaendert), kommt er auf den Knopf — nie unsichtbar dahinter (2.4.11)
    const ae = document.activeElement;
    if (!ae || ae === document.body || !f.contains(ae)) { const k = f.querySelector('#arbeitsGross'); if (k) k.focus(); }
    if (!war) announce(t('Das Arbeitsfenster liegt jetzt über dem Gespräch. Escape oder „Zurück zum Gespräch“ führt zurück.'));
}
function inkluagentUeberlagerungEinrichten() {
    if (window._arbeitsUeberlagerung) return;
    window._arbeitsUeberlagerung = true;
    const mq = window.matchMedia ? window.matchMedia('(max-width: 1099px)') : null;
    if (mq && mq.addEventListener) mq.addEventListener('change', inkluagentUeberlagerungAnwenden);
    else if (mq && mq.addListener) mq.addListener(inkluagentUeberlagerungAnwenden);   // aeltere Safari
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && !e.defaultPrevented && inkluagentEscape()) e.preventDefault();
    });
}
// Escape schliesst die Ueberlagerung — aus dieser Seite und aus der Seite im Rahmen (dashboard.js gibt die Taste weiter, B7). Der Fokus
// kommt auf den Knopf zurueck, der sie geoeffnet hat. true = geschlossen.
function inkluagentEscape() {
    if (!inkluagentUeberlagerungOffen()) return false;
    inkluagentFensterGross(false);
    const k = document.getElementById('arbeitsGross');
    if (k) k.focus();
    return true;
}
// „Zurück zum Gespräch“ (Knopf der Ueberlagerung, Sprunglink der Seite im Rahmen): kam inzwischen eine Antwort, die den Fokus nicht
// nehmen durfte (B3/B4), steht er auf ihr, sonst im Eingabefeld
function inkluagentZumGespraech() {
    if (inkluagentUeberlagerungOffen()) inkluagentFensterGross(false);
    const neu = window._inkluagentUngelesen;
    window._inkluagentUngelesen = null;
    if (neu && neu.isConnected && neu === inkluagentLetzteAntwort()) { neu.focus(); return; }   // nur die neueste, nie eine alte
    const input = document.getElementById('inkluagentInput');
    if (input) input.focus();
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
    // Pruefung 13.10.2026 (B23): die Seite im Rahmen heisst „Im Arbeitsfenster“, die Seitenzahl im PDF „Seite im Dokument“ — vorher
    // stand zweimal „Seite“ fuer zwei verschiedene Dinge („Seite: 3. Seite: Meine Ablage“)
    if (window._arbeitsSeite && window._arbeitsSeite.offen) out.push([t('Im Arbeitsfenster'), window._arbeitsSeite.name]);
    const hier = window._inkluagentProjectId == null ? null : String(window._inkluagentProjectId);
    const ziel = (z.ziel && z.projekt === hier) ? z.ziel : null;
    if (ziel && ziel.dokument) {
        out.push([t('Dokument'), ziel.dokument_nr ? t('Dokument {n}: {name}', { n: ziel.dokument_nr, name: String(ziel.dokument) }) : String(ziel.dokument)]);
    }
    if (ziel && ziel.seite) out.push([t('Seite im Dokument'), String(ziel.seite)]);
    return out;
}
// Ueberblick laden (ohne KI, nur Datenbank) — beim Aufbau des Arbeitsfensters und nach jedem Auftrag
function inkluagentUeberblickLaden() {
    if (window._arbeitsUeberblickLaedt) return;
    window._arbeitsUeberblickLaedt = true;
    fetch('/api/agent/ueberblick').then(r => r.ok ? r.json() : null).then(d => {
        window._arbeitsUeberblick = d || null;
        inkluagentArbeitsZeigen();
    }).catch(() => {}).finally(() => { window._arbeitsUeberblickLaedt = false; });
}
// Zeilen des Ueberblicks: [Bezeichnung, Text oder Liste von Links]
function inkluagentUeberblickZeilen() {
    const d = window._arbeitsUeberblick;
    if (!d) return [];
    const out = [];
    out.push([t('Guthaben'), d.guthaben == null ? t('unbegrenzt') : t('{n} Credits', { n: d.guthaben })]);
    const laeuft = Array.isArray(d.laeuft) ? d.laeuft : [];
    out.push([t('Läuft gerade'), laeuft.length ? laeuft.map(x => String(x.projekt || '') + ': ' + t(String(x.was || ''))).join('; ') : t('nichts')]);
    if (d.express) out.push([t('Express'), t('{offen} offen, {korb} im Warenkorb', { offen: d.express.offen || 0, korb: d.express.warenkorb || 0 })]);
    const projekte = Array.isArray(d.projekte) ? d.projekte : [];
    if (projekte.length) out.push([t('Zuletzt bearbeitet'), projekte]);
    return out;
}
// Text nur setzen, wenn er sich aendert (ein gleicher Text ersetzt sonst den Textknoten, und der VoiceOver-Cursor verliert die Stelle)
function inkluagentTextSetzen(el, text) {
    if (el && el.textContent !== text) el.textContent = text;
}
// Einen Teil des Arbeitsfensters nur neu bauen, wenn sich seine Daten geaendert haben (Pruefung 13.10.2026, B8/K19); stand der Fokus
// darin, kommt er auf denselben Link zurueck (sonst auf die Ueberschrift des Fensters). Ergebnis: true = neu gebaut.
function inkluagentTeilBauen(el, daten, bauen) {
    const stand = JSON.stringify(daten);
    if (!el || el._stand === stand) return false;
    el._stand = stand;
    const ae = document.activeElement;
    const fokus = (ae && el.contains(ae)) ? (ae.getAttribute('href') || '') : null;
    el.textContent = '';
    if (daten) bauen(el, daten);
    if (fokus !== null) {
        const wieder = Array.from(el.querySelectorAll('a[href]')).find(x => x.getAttribute('href') === fokus);
        if (wieder) wieder.focus(); else fokusAuf(document.getElementById('arbeitsfensterTitel'));
    }
    return true;
}
// Schritte: neue anhaengen, beim fertigen nur den Zustand umschreiben — nie die ganze Liste unter dem VoiceOver-Cursor ersetzen (B8)
function inkluagentSchritteZeigen(ol, schritte) {
    schritte.forEach((x, i) => {
        const text = String(x.text || '');
        const zustand = x.fehler ? 'fehler' : (x.fertig ? 'erledigt' : 'laeuft');
        let li = ol.children[i];
        if (!li || li._text !== text) {
            const neu = document.createElement('li');
            neu._text = text;
            neu.appendChild(document.createTextNode(text));
            const sp = document.createElement('span');
            sp.className = 'schritt-zustand';
            neu.appendChild(sp);
            if (li) ol.replaceChild(neu, li); else ol.appendChild(neu);
            li = neu;
        }
        if (li.dataset.zustand !== zustand) {
            li.dataset.zustand = zustand;
            li.lastChild.textContent = ' – ' + (x.fehler ? t('hat nicht geklappt') : (x.fertig ? t('erledigt') : t('läuft')));
        }
    });
    while (ol.children.length > schritte.length) ol.lastChild.remove();
}
function inkluagentArbeitsZeigen() {
    const f = document.getElementById('arbeitsfenster');
    if (!f) return;
    const z = inkluagentArbeitsZustand();
    inkluagentTextSetzen(f.querySelector('#arbeitsStand'), z.stand || t('Bereit. Schreib dem InkluAgent, was er erledigen soll.'));
    f.querySelector('#arbeitsFortschritt').hidden = !z.laeuft;
    const notiz = f.querySelector('#arbeitsNotiz');
    inkluagentTextSetzen(notiz, z.notiz ? t('Zwischenstand: {text}', { text: String(z.notiz) }) : '');
    notiz.hidden = !z.notiz;
    const erg = f.querySelector('#arbeitsErgebnis');
    const fertig = (z.ergebnis && z.ergebnis.text) ? [String(z.ergebnis.schritt || ''), String(z.ergebnis.text)] : null;
    inkluagentTeilBauen(erg.querySelector('p'), fertig, (p, d) => {
        const b = document.createElement('strong');
        b.textContent = d[0] + ': ';
        p.appendChild(b);
        p.appendChild(document.createTextNode(d[1]));
    });
    erg.hidden = !fertig;
    const dl = f.querySelector('#arbeitsAngaben');
    const angaben = inkluagentArbeitsAngaben(z);
    inkluagentTeilBauen(dl, angaben, (el, d) => d.forEach(([k, v]) => {
        const dt = document.createElement('dt'); dt.textContent = k;
        const dd = document.createElement('dd'); dd.textContent = v;
        el.appendChild(dt); el.appendChild(dd);
    }));
    dl.hidden = !angaben.length;
    // Ueberblick nur, solange kein Auftrag laeuft und keine Seite im Fenster offen ist
    const ub = f.querySelector('#arbeitsUeberblick');
    const seiteOffen = !!(window._arbeitsSeite && window._arbeitsSeite.offen);
    const zeilen = (z.laeuft || seiteOffen) ? [] : inkluagentUeberblickZeilen();
    inkluagentTeilBauen(ub.querySelector('dl'), zeilen, (ubdl, d) => d.forEach(([k, v]) => {
        const dt = document.createElement('dt'); dt.textContent = k;
        const dd = document.createElement('dd');
        if (Array.isArray(v)) {
            const ul = document.createElement('ul');
            v.forEach(pr => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = '/app?projekt=' + encodeURIComponent(pr.project_id);
                a.textContent = String(pr.name || '');
                li.appendChild(a);
                ul.appendChild(li);
            });
            dd.appendChild(ul);
        } else {
            dd.textContent = v;
        }
        ubdl.appendChild(dt); ubdl.appendChild(dd);
    }));
    ub.hidden = !zeilen.length;
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
        // HB6: der vorhandene Alt-Text des Bildes, wenn der Server ihn mitschickt (ziel.alt); sonst ist das Bild Schmuck (alt="") und
        // die Bildunterschrift sagt, was zu sehen ist
        const alt = (typeof ziel.alt === 'string') ? ziel.alt.trim().slice(0, 500) : '';
        if (img.getAttribute('alt') !== alt) img.setAttribute('alt', alt);
        inkluagentTextSetzen(fig.querySelector('figcaption'), [inkluagentZielText(ziel), ziel.seite ? t('Seite {n}', { n: ziel.seite }) : '']
            .filter(Boolean).join(' · '));
        fig.hidden = false;
    } else {
        fig.hidden = true;
    }
    const ol = f.querySelector('#arbeitsSchritte');
    inkluagentSchritteZeigen(ol, z.schritte);
    ol.hidden = !z.schritte.length;
    f.querySelector('#arbeitsSchritteTitel').hidden = ol.hidden;
}
// „Jeden Schritt ansagen“ (Vorgabe an; ohne Speicher auch an)
function inkluagentSchritteAnsagen() {
    const k = document.getElementById('arbeitsAnsagen');
    if (k) return k.checked;
    try { return localStorage.getItem('inkluagent.schritte_ansagen') !== 'aus'; } catch (e) { return true; }
}
// Schritt ansagen (nur mit „Jeden Schritt ansagen“). Pruefung 13.10.2026 (B9): Schritte, die in wenigen Millisekunden nacheinander kamen,
// ersetzten sich in der Ansage-Region, bevor VoiceOver sie sprach. Jetzt gesammelt und hoechstens einmal je Sekunde gebuendelt gesagt;
// ist die Antwort da, faellt Uebriges weg (es wuerde die Antwort unterbrechen). Die Vorgabe bleibt „an“ (Steve, Runde 2: B9 bleibt so).
// Runde 2 (A-N8): an announce geht ein Schritt als verwerfbar — wartet er dort noch (Mindestabstand), streicht ihn das Ende ebenfalls;
// vorher konnte er bis zu 100 ms NACH dem Fokus auf der Antwort in die Region kommen und deren Vorlesen unterbrechen.
function inkluagentSchrittAnsagen(text, sofort) {
    if (!text || !inkluagentSchritteAnsagen()) return;
    const b = window._schrittAnsage || (window._schrittAnsage = { warte: [], zuletzt: 0, plan: 0 });
    b.warte.push(text);
    const sagen = (verwerfbar) => {
        b.plan = 0;
        b.zuletzt = Date.now();
        const satz = b.warte.join('. ');
        b.warte = [];
        announce(satz, { verwerfbar: verwerfbar });
    };
    // Ein Fehler (sofort) wartet nicht — er geht mit dem, was schon wartet, gleich hinaus und faellt beim Ende nicht weg
    if (sofort) { clearTimeout(b.plan); sagen(false); return; }
    if (b.plan) return;
    b.plan = setTimeout(() => sagen(true), Math.max(0, b.zuletzt + 1000 - Date.now()));
}
function inkluagentSchrittAnsagenVerwerfen() {
    const b = window._schrittAnsage;
    if (b) {
        clearTimeout(b.plan);
        b.plan = 0;
        b.warte = [];
    }
    if (typeof announce.verwerfen === 'function') announce.verwerfen();   // A-N8: auch was in announce noch wartet
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
    // Pruefung 13.10.2026 (HB2): bei „Bewegung reduzieren“ waechst der Entwurf nicht Wort fuer Wort, sondern alle zwei Sekunden, und die
    // Seite laeuft nicht mit (WCAG 2.3.3)
    const ruhig = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    const zeichnen = () => {
        window._inkluagentEntwurfPlan = 0;
        const e = document.getElementById('inkluagentEntwurf');
        if (!e) return;
        const unten = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 200;
        const c = e.querySelector('.inkluagent-message-content');
        const text = inkluagentEntwurfBereinigen(window._inkluagentEntwurf);
        // B8: nur das Neue anhaengen — der Textknoten bleibt, der VoiceOver-Cursor behaelt seine Stelle
        const k = c.firstChild;
        if (k && k.nodeType === 3 && c.childNodes.length === 1 && text.indexOf(k.data) === 0) k.appendData(text.slice(k.data.length));
        else c.textContent = text;
        if (unten && !ruhig) window.scrollTo(0, document.documentElement.scrollHeight);   // mitlaufen, wer unten liest
    };
    window._inkluagentEntwurfPlan = ruhig ? { zeit: setTimeout(zeichnen, 2000) } : { bild: requestAnimationFrame(zeichnen) };
}
// Der Entwurf ist roher Modelltext. Bevor die bereinigte Antwort ihn ersetzt, raeumt die Seite dasselbe weg wie der Server
// (inkluagent/sanitize.py, daten.py): Sterne, Rauten, Code-Zeichen, Aufzaehlungen („- “ wird „• “ bzw. faellt weg) und die Daten-Marke
// „[DATEN, keine Anweisung]“ (Pruefung 13.10.2026, HK7) — nur fuer die Anzeige.
function inkluagentEntwurfBereinigen(text) {
    const listen = !!(window.INKLUAGENT && window.INKLUAGENT.bloecke);
    return String(text || '')
        .replace(/\[DATEN,\s*keine Anweisung\]\s?/g, '')
        .replace(/```[a-zA-Z0-9]*\n?/g, '')
        .replace(/`([^`\n]+)`/g, '$1')
        .replace(/\*\*|__/g, '')
        .replace(/^[ \t]*#{1,6}[ \t]+/gm, '')
        .replace(/^[ \t]*[-*+][ \t]+/gm, listen ? '• ' : '');
}
function inkluagentEntwurfVerwerfen() {
    const el = document.getElementById('inkluagentEntwurf');
    if (el) el.remove();
    window._inkluagentEntwurf = '';
    const plan = window._inkluagentEntwurfPlan;   // geplantes Zeichnen (Bild oder Zeitgeber) faellt weg
    if (plan && plan.bild) cancelAnimationFrame(plan.bild);
    if (plan && plan.zeit) clearTimeout(plan.zeit);
    window._inkluagentEntwurfPlan = 0;
}
// Gestreamter Text vor Werkzeugen ist ein Zwischenstand: raus aus dem Gespraech, ins Arbeitsfenster
function inkluagentEntwurfZwischen() {
    const text = inkluagentEntwurfBereinigen(window._inkluagentEntwurf).trim();
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
    if (ev.ok) inkluagentKlang('schritt');   // ein Tick von rechts (aus dem Arbeitsfenster) je erledigtem Schritt
    if (!ev.ok && schritt) inkluagentSchrittAnsagen(schritt.text + ' – ' + t('hat nicht geklappt'), true);
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
    inkluagentSchrittAnsagen(text);   // „Jeden Schritt ansagen“ (Steve 12.10.2026), gebuendelt (B9)
}
function inkluagentArbeitsEnde(wartet) {
    inkluagentSchrittAnsagenVerwerfen();
    if (!inkluagentChatstilAktiv()) return;
    const z = inkluagentArbeitsZustand();
    z.laeuft = false;
    z.schritte.forEach(x => { x.fertig = true; });
    z.stand = wartet ? t('Wartet auf dich: Bitte entscheide unter der Antwort.') : t('Fertig. Die Antwort steht im Gespräch.');
    inkluagentArbeitsMerken();
    inkluagentUeberblickLaden();   // Guthaben, Laeufe und Projekte koennen sich geaendert haben
}
function inkluagentArbeitsFehler(text) {
    inkluagentSchrittAnsagenVerwerfen();
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
        .concat(inkluagentArbeitsAngaben(z).map(([k, v]) => k + ': ' + v));   // mit „Im Arbeitsfenster: …“ (B23)
    if (z.ergebnis && z.ergebnis.text) teile.push(t('Zuletzt fertig') + ': ' + z.ergebnis.schritt + ': ' + z.ergebnis.text);
    const seiteOffen = !!(window._arbeitsSeite && window._arbeitsSeite.offen);
    if (!seiteOffen && !z.laeuft) inkluagentUeberblickZeilen().forEach(([k, v]) => teile.push(k + ': ' + (Array.isArray(v) ? v.map(x => x.name).join(', ') : v)));
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
            inkluagentAppendMessage(m.role, m.content, m.werkzeuge, m.anhang, m.sprache);
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
function inkluagentAntwortMelden(el, fokussieren) {
    // Wartet in der Antwort eine Karte auf dich, rauscht sie hoerbar herein (von rechts in die Mitte), sonst der Klang „fertig“
    const antwort = el || inkluagentLetzteAntwort();
    inkluagentKlang(antwort && antwort.querySelector('.inkluagent-bestaetigung[data-zustand="offen"]') ? 'karte' : 'fertig');
    if (fokussieren && el) {
        el.focus();
        if (document.activeElement === el) return true;
    }
    window._inkluagentUngelesen = el || inkluagentLetzteAntwort();
    announce(inkluagentUeberlagerungOffen() ? t('Antwort vom InkluAgent ist da. „Zurück zum Gespräch“ führt zu ihr.')
                                            : t('Antwort vom InkluAgent ist da.'));
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
    } catch (e) {
        btn.removeAttribute('aria-disabled');
        if (statusEl) statusEl.textContent = t('Verbindungsfehler. Bitte erneut versuchen.');
    }
}

// Meine Ausgaben (11.09.2026): Download-Knopf + Link ins Regal unter einer Bot-Antwort, wenn der Bot
// umgewandelt oder exportiert hat (anhang aus dem Werkzeug-Ergebnis, im Verlauf mitgespeichert).
function inkluagentAnhangEl(anhang) {
    const box = document.createElement('div');
    box.className = 'inkluagent-message-anhang';   // Anordnung in style.css (seit 13.10.2026; vorher als style-Attribut)
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
    }
}

// Schritt 3: den festen Bereich gleich beim Laden bauen (das aside steht im Seitenrahmen vor diesem Skript); ohne Schalter oder
// ohne Bereich (app.html bettet ein) passiert nichts.
inkluagentBereichAufbauen();
