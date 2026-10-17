// InkluAgent — Seiten im Arbeitsfenster und das Projekt auf der Bühne (Kamerafahrt, Mitsehen und Mithören — nur über window.InkluBuehne).
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

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
    const neu = s.projekt ? s.name : t('Seite: {name}', { name: s.name });   // Projekt: seine Ueberschrift („Alt-Texte – Projekt: …“)
    // Das eigene Projekt ist der Grundzustand des Fensters auf seiner Seite: kein „Seite schließen“; ein anderes Projekt oder die Ablage,
    // in die der Agent gefahren ist, schliesst man wieder (zurueck zum eigenen). „Mitverfolgen“, wo die Kamera fahren kann.
    const eigenes = !!(s.projekt && String(s.projektId) === String(window._inkluagentProjectId));
    f.querySelector('#arbeitsSeiteZu').hidden = eigenes;
    f.querySelector('#arbeitsFolgen').hidden = !(s.projekt || s.vomAgenten);
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
    // das Projekt oeffnet das Fenster auf seiner Seite ohnehin, und was der Agent geoeffnet hat, gehoert zu seinem Lauf (Live-Fenster)
    if (!s || s.projekt || s.vomAgenten) return;
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
    const projekt = eigene && eingebettet === true && inkluagentIstProjektImRahmen(loc);   // Live-Fenster: DIESES Projekt
    if (eigene && eingebettet === true && (fensterSeite(loc.pathname) || projekt)) {        // 1. Erfolg
        const h1 = doc.querySelector('#main h1');
        const titel = (h1 && h1.textContent.trim()) || String(doc.title || '').split(' · ')[0].trim();
        const vorher = window._arbeitsSeite || {};
        const fokus = !!vorher.fokus || document.activeElement === rahmen;
        const name = projekt ? ((h1 && h1.textContent.trim()) || vorher.name || titel) : (titel || vorher.name || loc.pathname);
        // Live-Fenster (17.10.2026): ob der Agent die Seite geoeffnet hat (fremdes Projekt, Ablage fuer ein Ergebnis) — nur dann folgt
        // die Kamera darin weiter; geht der Nutzer im Rahmen auf eine andere Seite, ist es seine
        const gleich = vorher.pfad && inkluagentPfadOhne(vorher.pfad, ['ansicht']).split('#')[0] === inkluagentPfadOhne(pfad, ['ansicht']).split('#')[0];
        window._arbeitsSeite = { offen: true, pfad: pfad, name: name, fokus: false, hin: false, projekt: projekt,
                                 projektId: projekt ? new URLSearchParams(loc.search).get('projekt') : null,
                                 ablage: /^\/ablage\/?$/.test(loc.pathname), vomAgenten: !!(vorher.vomAgenten && gleich) };
        inkluagentSeiteZeigen();
        inkluagentSeiteMerken();
        if (fokus) inkluagentRahmenFokus(rahmen, !!(vorher.fokus && vorher.hin), loc.hash);
        inkluagentBuehneGeladen(win, doc);   // Seiten mit Bühne (Projekt, Ablage): Kamerafahrt, Titel, Mithoeren
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
    // Wer schliesst, will dort nicht wieder hin: die letzte Stelle des Agenten vergessen (sonst holte das Laden des eigenen Projekts
    // das andere Projekt bzw. die Ablage gleich zurueck); der naechste Schritt des Agenten zeigt wieder, wo er ist
    window._arbeitsFolgeZiel = null;
    if (inkluagentProjektFensterAn()) inkluagentProjektImFenster();   // Live-Fenster: wieder das Projekt
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
    } else if (frisch && (!inkluagentProjektFensterAn() || w.seite === inkluagentPfadOhne(hier.split('#')[0], ['hin']))) {
        inkluagentSeiteImFenster(w.pfad, w.name, true);
    } else if (inkluagentProjektFensterAn()) {
        // Live-Fenster (16.10.2026): auf einer Projektseite die echte Oberflaeche dieses Projekts — eine gemerkte andere Seite nur,
        // wenn sie auf GENAU dieser Seite offen war (Neuladen), nicht von einer frueheren Seite her
        inkluagentProjektImFenster();
    }
    let gross = '';
    try { gross = sessionStorage.getItem('inkluagent.fenster.gross') || ''; } catch (e) { gross = ''; }
    if (gross === '1' && !inkluagentFensterSchmal()) inkluagentFensterGross(true);
}

// ─── Das Projekt selbst im Arbeitsfenster: Live-Fenster mit Kamerafahrt, Mitsehen und Mithoeren (Steve 16./17.10.2026) ───
// „Dass man wirklich einen Livestream sieht, wo er gerade ist — den Live-Ausschnitt von der Oberflaeche.“ Auf einer Projektseite zeigt das
// Arbeitsfenster ein Projekt als echte, bedienbare Oberflaeche (eingebettet: ohne Seitenleiste und ohne eigenen Agenten, siehe
// inkluagentInit). Der Agent steuert das Fenster: bei jedem Schritt mit Ziel und bei jeder Stelle eines Laufs im Hintergrund faehrt es dorthin
// — in sein Projekt (auch ein anderes eigenes Projekt, in dem er gerade arbeitet) oder in die Ablage (das fertige Ergebnis).
// FESTE SCHNITTSTELLE (17.10.2026): diese Seite spricht mit der Seite im Fenster NUR ueber window.InkluBuehne (dashboard.js
// buehneEinrichten; jede Seite sagt selbst, wie sie ihre Stellen findet: app.html, ablage.html): zeigen, neuZeichnen, beobachten. Sie greift
// nicht in Kennungen oder Funktionen der Seite — Ansichten duerfen sich aendern, ohne die Kamerafahrt zu brechen.
// Nie, solange der Fokus im Fenster steht, und nicht mehr, sobald man dort selbst etwas tut („Mitverfolgen“ springt dann auf aus; ein Druck
// folgt wieder). Darunter „So klingt diese Stelle“: wie ein Screenreader die Stelle ungefaehr vorliest, mit „Anhören“.
// Zustand im window (_arbeitsSeite: offen, projekt, projektId, ablage, vomAgenten …; _arbeitsFolgeZiel).
function inkluagentProjektFensterAn() {
    return inkluagentChatstilAktiv() && window._inkluagentProjectId != null && !window.GUEST_MODE;
}
function inkluagentZahl(x) {
    return /^\d+$/.test(String(x == null ? '' : x)) ? String(x) : null;
}
function inkluagentProjektPfad(projektId, ansicht) {
    return '/app?projekt=' + encodeURIComponent(projektId) + (ansicht ? '&ansicht=' + encodeURIComponent(ansicht) : '');
}
// Ein Projekt gilt im Rahmen als Erfolg, wenn es das Projekt dieser Seite ist oder das, das diese Seite gerade dort geoeffnet hat
// (der Agent arbeitet darin). Ein anderes Projekt, das jemand im Rahmen per Link oeffnet, wechselt wie bisher die ganze Seite.
function inkluagentIstProjektImRahmen(loc) {
    if (!loc || !/^\/app\/?$/.test(loc.pathname)) return false;
    const p = new URLSearchParams(loc.search).get('projekt');
    if (!p) return false;
    const s = window._arbeitsSeite;
    return (window._inkluagentProjectId != null && p === String(window._inkluagentProjectId)) || !!(s && s.projekt && String(s.projektId) === p);
}
function inkluagentProjektImFenster(projektId, ansicht) {
    const f = document.getElementById('arbeitsfenster');
    const pid = inkluagentZahl(projektId == null ? window._inkluagentProjectId : projektId);
    // Das eigene Projekt nur auf einer Projektseite; ein Projekt, in dem der Agent arbeitet, auf jeder Seite mit Agent (auch der Startseite)
    const darf = projektId == null ? inkluagentProjektFensterAn() : (inkluagentChatstilAktiv() && !window.GUEST_MODE);
    if (!f || !pid || !darf) return;
    const eigenes = String(window._inkluagentProjectId) === pid;
    const h1 = document.querySelector('#main h1');
    window._arbeitsSeite = { offen: true, projekt: true, projektId: pid, pfad: inkluagentProjektPfad(pid, ansicht),
                             name: eigenes && h1 ? h1.textContent.trim() : t('Projekt'), fokus: false, hin: false, laedt: true,
                             vomAgenten: !eigenes };
    f.querySelector('#arbeitsSeiteMeldung').hidden = true;
    inkluagentSeiteZeigen();
    inkluagentRahmenLaden(f.querySelector('#arbeitsSeiteRahmen'), inkluagentProjektPfad(pid, ansicht) + '&eingebettet=1');
}
// Die Ablage im Fenster, weil ein Lauf dort sein Ergebnis abgelegt hat (Umwandeln, Herunterladen)
function inkluagentAblageImFenster() {
    if (!inkluagentChatstilAktiv()) return;
    inkluagentSeiteImFenster('/ablage', t('Meine Ablage'), true);
    if (window._arbeitsSeite) Object.assign(window._arbeitsSeite, { ablage: true, vomAgenten: true });
}
// Eine Seite mit Bühne ist im Fenster geladen: wer dort selbst etwas tut, wird nicht mehr weggezogen; Titel und „So klingt diese Stelle“
// kommen von der Bühne (beobachten); eine Stelle, die waehrend des Ladens kam, wird jetzt gezeigt
function inkluagentBuehneGeladen(win, doc) {
    if (!doc || doc._inkluagentSelbst) return;
    doc._inkluagentSelbst = true;
    ['pointerdown', 'wheel', 'keydown', 'touchstart'].forEach(art => doc.addEventListener(art, e => {
        if (e.isTrusted) inkluagentFolgenSetzen(false, false);
    }, { passive: true }));
    const b = win && win.InkluBuehne;
    if (b && typeof b.beobachten === 'function') b.beobachten(d => inkluagentBuehneGemeldet(doc, d));
    if (window._arbeitsFolgeZiel) inkluagentFensterFolgen(window._arbeitsFolgeZiel);
}
// Die Bühne meldet: Titel der Seite (beim Projekt seine Ueberschrift mit Ansicht, „Alt-Texte – Projekt: …“) und was an der Stelle steht
function inkluagentBuehneGemeldet(doc, d) {
    const s = window._arbeitsSeite;
    const rahmen = document.getElementById('arbeitsSeiteRahmen');
    if (!s || !d || !rahmen) return;
    try { if (rahmen.contentDocument !== doc) return; } catch (e) { return; }   // inzwischen eine andere Seite im Fenster
    if (s.projekt && d.titel && d.titel !== s.name) { s.name = d.titel; inkluagentSeiteZeigen(); }
    const g = d.gelesen;
    if (g && g.art === 'bild') inkluagentMithoerenSetzen({ art: 'bild', alt: g.text });
    else if (g && g.art === 'feld') inkluagentMithoerenSetzen({ art: 'feld', info: g.text });
}
function inkluagentFolgenAn() {
    const k = document.getElementById('arbeitsFolgen');
    return !k || k.getAttribute('aria-pressed') !== 'false';
}
// selbst: per Knopf (dann gleich wieder zur letzten Stelle). Ohne Ansage: der Knopf sagt seinen Zustand selbst.
function inkluagentFolgenSetzen(an, selbst) {
    const k = document.getElementById('arbeitsFolgen');
    if (!k || (k.getAttribute('aria-pressed') === 'true') === !!an) return;
    k.setAttribute('aria-pressed', an ? 'true' : 'false');
    if (an && selbst && window._arbeitsFolgeZiel) inkluagentFensterFolgen(window._arbeitsFolgeZiel);
}
// Kamerafahrt zu einer Stelle (vom Server: art bild|feld|dokument|seite|ausgabe, Kennungen, project_id, ansicht). Erst die richtige
// Seite ins Fenster (sein Projekt, die Ablage), dann zeigt die Bühne die Stelle. Eine Seite, die der Nutzer selbst geoeffnet hat
// (Einstellungen …), bleibt stehen.
function inkluagentFensterFolgen(stelle) {
    if (!stelle || typeof stelle !== 'object' || !inkluagentChatstilAktiv()) return;
    window._arbeitsFolgeZiel = stelle;
    if (!inkluagentFolgenAn()) return;
    const rahmen = document.getElementById('arbeitsSeiteRahmen');
    if (!rahmen || document.activeElement === rahmen) return;            // wer im Fenster arbeitet, wird nicht weggezogen
    const s = window._arbeitsSeite;
    if (s && s.offen && !s.projekt && !s.vomAgenten) return;             // eine selbst geoeffnete Seite bleibt
    if (stelle.art === 'ausgabe') {
        if (!inkluagentZahl(stelle.ausgabe_id)) return;
        if (!(s && s.ablage)) { inkluagentAblageImFenster(); return; }   // nach dem Laden zeigt inkluagentBuehneGeladen die Stelle
    } else {
        const pid = inkluagentZahl(stelle.project_id != null ? stelle.project_id : window._inkluagentProjectId);
        if (!pid) return;
        if (!(s && s.projekt && String(s.projektId) === pid)) { inkluagentProjektImFenster(pid, stelle.ansicht); return; }
    }
    if (s && s.laedt) return;
    let b = null;
    try { b = rahmen.contentWindow && rahmen.contentWindow.InkluBuehne; } catch (e) { return; }
    if (!b || typeof b.zeigen !== 'function') return;
    Promise.resolve(b.zeigen(stelle, { weich: true })).catch(() => {});
}
// Nach Aenderungen des Agenten: die Seite im Fenster neu zeichnen — nicht, solange man dort arbeitet. Die Bühne haelt Stelle und Marke.
function inkluagentRahmenNachziehen() {
    const s = window._arbeitsSeite;
    const rahmen = document.getElementById('arbeitsSeiteRahmen');
    if (!s || !s.offen || s.laedt || !rahmen || document.activeElement === rahmen) return;
    let b = null;
    try { b = rahmen.contentWindow && rahmen.contentWindow.InkluBuehne; } catch (e) { return; }
    if (b && typeof b.neuZeichnen === 'function') Promise.resolve(b.neuZeichnen()).catch(() => {});
}
// „So klingt diese Stelle“: vereinfacht, wie VoiceOver oder NVDA es sagen — Bild mit Alternativtext, Formularfeld mit Quickinfo
function inkluagentMithoerenSetzen(stelle) {
    const box = document.getElementById('arbeitsHoeren');
    if (!box || !stelle) return;
    let text = '';
    if (stelle.art === 'bild') {
        const alt = String(stelle.alt || '').trim();
        if (stelle.status === 'error') text = t('Hier gibt es noch keinen Alternativtext. Ein Screenreader sagt nur „Bild“ oder überspringt es.');
        else text = alt ? t('Bild, {alt}', { alt: alt }) : t('Hier gibt es noch keinen Alternativtext. Ein Screenreader sagt nur „Bild“ oder überspringt es.');
    } else if (stelle.art === 'feld') {
        const info = String(stelle.info || '').trim();
        text = info ? t('Formularfeld, {info}', { info: info }) : t('Dieses Feld hat noch keine Quickinfo. Ein Screenreader nennt dann oft nur „Textfeld“.');
    }
    if (!text) return;
    inkluagentTextSetzen(box.querySelector('#arbeitsHoerenText'), text);
    box.hidden = false;
}
