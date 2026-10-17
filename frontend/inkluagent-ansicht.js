// InkluAgent — Agentenansicht und Chatstil: Ansicht je Konto, Gespräch + Arbeitsfenster, Aufbau des Arbeitsfensters.
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

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
        // Live-Fenster (16.10.2026): das Fenster folgt dem Agenten durch die Oberflaeche des Projekts (Kamerafahrt) — aus, sobald man
        // dort selbst etwas tut; ein Druck folgt wieder. Nur sichtbar, wenn das Projekt im Fenster steht.
        +   '<button type="button" id="arbeitsFolgen" class="btn btn-secondary" aria-pressed="true" hidden>' + t('Mitverfolgen') + '</button>'
        + '</div>'
        + '<p id="arbeitsStand" class="arbeitsfenster-stand"></p>'
        + '<progress id="arbeitsFortschritt" class="arbeitsfenster-fortschritt" aria-labelledby="arbeitsStand" hidden></progress>'
        // Live-Fenster (Steve 16.10.2026): was ein Lauf im Hintergrund gerade tut — das Bild, an dem er schreibt, darunter der Alt-Text
        // des Bildes davor, sobald er fertig ist. Texte aus Daten setzt inkluagentLiveZeigen mit textContent.
        + '<div id="arbeitsLive" class="arbeitsfenster-live" hidden>'
        +   '<h3 id="arbeitsLiveTitel"></h3>'
        +   '<progress id="arbeitsLiveFortschritt" class="arbeitsfenster-fortschritt" aria-labelledby="arbeitsLiveTitel"></progress>'
        +   '<figure class="arbeitsfenster-live-bild"><img alt=""><figcaption></figcaption></figure>'
        +   '<p id="arbeitsLiveFertig" class="arbeitsfenster-live-fertig" hidden></p>'
        + '</div>'
        // Seiten im Arbeitsfenster (13.10.2026, Steve: wie die Vorschau in InkluEdit): die echte Seite, eingebettet und bedienbar
        + '<div id="arbeitsSeite" class="arbeitsfenster-seite" hidden>'
        +   '<div class="arbeitsfenster-seite-kopf"><h3 id="arbeitsSeiteTitel"></h3>'
        +     '<button type="button" id="arbeitsSeiteZu" class="btn btn-secondary">' + t('Seite schließen') + '</button></div>'
        +   '<iframe id="arbeitsSeiteRahmen" src="about:blank"></iframe>'
        + '</div>'
        // Runde 2 (K-N1): kam eine Seite nicht (Fehlerantwort), steht hier, dass sie sich gerade nicht oeffnen laesst
        + '<p id="arbeitsSeiteMeldung" class="arbeitsfenster-seite-meldung" hidden></p>'
        // Mitsehen und Mithoeren (16.10.2026): wie ein Screenreader die Stelle vorliest, an der der Agent gerade ist
        + '<div id="arbeitsHoeren" class="arbeitsfenster-hoeren" hidden>'
        +   '<h3 id="arbeitsHoerenTitel">' + t('So klingt diese Stelle') + '</h3>'
        +   '<p id="arbeitsHoerenText" class="arbeitsfenster-hoeren-text"></p>'
        + '</div>'
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
    s.querySelector('#arbeitsFolgen').addEventListener('click', () => inkluagentFolgenSetzen(!inkluagentFolgenAn(), true));
    inkluagentAnhoerenKnopf(s.querySelector('#arbeitsHoerenTitel'), s.querySelector('#arbeitsHoerenText'));
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
