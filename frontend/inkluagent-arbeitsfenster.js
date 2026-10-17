// InkluAgent — Arbeitsfenster: Stand, Schritte, Entwurf, Vergrößern und Überlagerung, „Stand vorlesen“.
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

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
    // Live-Fenster (16.10.2026): tut der Agent selbst nichts, aber ein Lauf arbeitet im Hintergrund, sagt der Stand das
    const ll = window._arbeitsLive && window._arbeitsLive.lauf;
    const lv = ll && ll.laeuft ? ll : null;
    const hinten = (!z.laeuft && lv) ? (lv.gesamt > 1
        ? t('Arbeitet im Hintergrund: {was}, {n} von {m} fertig.', { was: inkluagentLiveWas(lv.art === 'auftrag' && lv.station ? lv.station : lv.art), n: lv.fertig, m: lv.gesamt })
        : t('Arbeitet im Hintergrund: {was}.', { was: inkluagentLiveWas(lv.art === 'auftrag' && lv.station ? lv.station : lv.art) })) : '';
    inkluagentTextSetzen(f.querySelector('#arbeitsStand'), hinten || z.stand || t('Bereit. Schreib dem InkluAgent, was er erledigen soll.'));
    f.dataset.zustand = z.laeuft ? 'laeuft' : (lv ? 'hinten' : 'ruhe');   // nur fuer den Punkt vor dem Stand (style.css)
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
    if (!behalten && !(window._arbeitsLive && window._arbeitsLive.lauf && window._arbeitsLive.lauf.laeuft)) {
        window._arbeitsLive = null;
        const lb = document.getElementById('arbeitsLive');
        if (lb) lb.hidden = true;
    }
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
    if (ziel && typeof ziel === 'object') inkluagentFensterFolgen(ziel);   // Live-Fenster: Kamerafahrt zur Stelle
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
    inkluagentLiveTexte().forEach(x => teile.push(x));   // Live-Fenster: was im Hintergrund laeuft und was gerade fertig wurde
    announce(teile.map(x => x.replace(/[.\s]+$/, '')).join('. ') + '.');
}
// Nach dem Knopf einer Karte (Bestaetigung, Formular) ist die Entscheidung gefallen: „Wartet auf dich“ im Arbeitsfenster stimmt nicht
// mehr (16.10.2026, gefunden im Rauchtest des Live-Fensters — der Stand blieb bis zur naechsten Nachricht stehen). Wartet die neue
// Antwort selbst wieder auf eine Entscheidung, bleibt es dabei.
function inkluagentEntschieden(anhang) {
    if (!inkluagentChatstilAktiv()) return;
    const z = inkluagentArbeitsZustand();
    if (z.laeuft) return;
    z.stand = inkluagentWartetAufDich(anhang) ? t('Wartet auf dich: Bitte entscheide unter der Antwort.') : t('Fertig. Die Antwort steht im Gespräch.');
    inkluagentArbeitsMerken();
}
// Wartet eine Antwort auf eine Entscheidung (offene Karte)?
function inkluagentWartetAufDich(anhang) {
    return Array.isArray(anhang) && anhang.some(a => a && (!a.zustand || a.zustand === 'offen')
        && ((a.art === 'bestaetigung' && a.angebot_id) || (a.art === 'formular' && a.karten_id)));
}
