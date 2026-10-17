// InkluAgent — Live-Fenster: was der Agent im Hintergrund tut (GET /api/agent/live, Laufbuch).
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

// ─── Live-Fenster (Steve 16./17.10.2026: „wie einen Livestream im Arbeitsfenster“; „nur das, was der Agent auch an Aufgaben macht“) ───
// Arbeitet der Agent im Hintergrund (nach „Auftrag erteilen“, dem Knopf einer Karte oder einer Antwort), fragt die Seite GET /api/agent/live
// alle 2,5 Sekunden, solange etwas laeuft und die Seite sichtbar ist. Der Server liest das Laufbuch (laufbuch.py — jeder Lauf meldet sich
// dort an: Alt-Texte, Quickinfos, Tagging, Korrektur, Pruefung, Umwandeln, Uebersetzen, Herunterladen, Auftraege mit Stationen; nur
// Laeufe des Agenten). Gezeigt wird der juengste: Kopfzeile mit Art, Projekt und Fortschritt; die Kamera faehrt zu seiner Stelle (ein Bild,
// ein Feld, ein Dokument, eine Seite) und am Ende zum Ergebnis (der Eintrag in der Ablage). Je fertigem Stueck ein Tick und — bei Bildern
// und Feldern — „So klingt diese Stelle“ mit dem neuen Text. Keine Ansagen: VoiceOver bekommt alles mit „Stand vorlesen“ und am Ende die
// Rueckmeldung des Agenten. Ein neuer Lauf braucht hier nichts — nur eine Art im Laufbuch und eine Ansicht (arbeitsfenster.ANSICHT_JE_LAUF).
// Zustand im window (_arbeitsLive); auf oberster Ebene nur Funktionen (test_inkluagent_js).
function inkluagentLiveBeobachten() {
    if (!inkluagentArbeitsfensterAn()) return;
    // Ein Lauf, den die Antwort oder der Knopf gerade erst angestossen hat, steht vielleicht noch nicht beim Server: zweimal nachsehen
    window._arbeitsLiveAnlauf = 2;
    if (window._arbeitsLiveBeobachtet) return;
    window._arbeitsLiveBeobachtet = true;
    const runde = async () => {
        let d = null;
        try {
            const res = await fetch('/api/agent/live');
            d = res.ok ? await res.json() : null;
        } catch (e) {
            d = null;
        }
        if (d) inkluagentLiveZeigen(d);
        const laeuft = !!(d && Array.isArray(d.laeufe) && d.laeufe.some(x => x && x.laeuft));
        if (laeuft) {
            // unsichtbarer Tab: erst wieder fragen, wenn er sichtbar wird
            const weiter = () => setTimeout(runde, 2500);
            if (document.visibilityState === 'hidden') document.addEventListener('visibilitychange', weiter, { once: true });
            else weiter();
        } else if (d && !laeuft && window._arbeitsLiveAnlauf > 0 && !(Array.isArray(d.laeufe) && d.laeufe.length)) {
            window._arbeitsLiveAnlauf--;
            setTimeout(runde, 3000);
        } else {
            window._arbeitsLiveBeobachtet = false;
        }
    };
    runde();
}
// Was ein Lauf tut, in Worten der Oberflaeche (Art des Laufbuchs; bei Auftraegen die Station)
function inkluagentLiveWas(art) {
    const namen = { alt_texte: t('Alt-Texte'), quickinfos: t('Quickinfos'), tagging: t('Tagging'), korrektur: t('Korrektur'),
                    pruefung: t('Prüfung'), umwandeln: t('In PDF umwandeln'), uebersetzen: t('Übersetzung'), export: t('Herunterladen'),
                    auftrag: t('Auftrag') };
    return namen[art] || t('Arbeit');
}
// Kopfzeile eines Laufs: „Im Hintergrund: Alt-Texte für „Projekt“, 2 von 3 fertig“ (ohne Zahlen, wenn er nur ein Stueck hat)
function inkluagentLiveTitel(l) {
    const was = inkluagentLiveWas(l.art === 'auftrag' && l.station ? l.station : l.art);
    const projekt = String(l.projekt || '');
    if (!l.laeuft) return l.ok === false ? t('Im Hintergrund: {was} – hat nicht geklappt', { was: was }) : t('Im Hintergrund: {was} fertig', { was: was });
    if (l.art === 'auftrag') return t('Im Hintergrund: Auftrag für „{projekt}“, Schritt {n} von {m}: {was}', { projekt: projekt, n: Math.min(l.gesamt, l.fertig + 1), m: l.gesamt, was: was });
    if (l.gesamt > 1) return t('Im Hintergrund: {was} für „{projekt}“, {n} von {m} fertig', { was: was, projekt: projekt, n: l.fertig, m: l.gesamt });
    return t('Im Hintergrund: {was} für „{projekt}“', { was: was, projekt: projekt });
}
function inkluagentLiveSchluessel(x) {
    return x ? JSON.stringify([x.art, x.project_id, x.image_id, x.feld_id, x.document_id, x.seite, x.ausgabe_id, x.ansicht]) : '';
}
function inkluagentLiveZeigen(d) {
    const box = document.getElementById('arbeitsLive');
    if (!box) return;
    const alt = window._arbeitsLive || {};
    const l = (d && Array.isArray(d.laeufe) && d.laeufe[0]) || null;
    const neu = { lauf: l, nr: l ? l.nr : null, stelle: l ? inkluagentLiveSchluessel(l.stelle) : '', vorher: l ? inkluagentLiveSchluessel(l.vorher) : '' };
    window._arbeitsLive = neu;
    if (!l) { box.hidden = true; inkluagentArbeitsZeigen(); return; }
    const derselbe = alt.lauf && alt.nr === l.nr;
    const erster = !('lauf' in alt);   // erster Blick nach dem Laden der Seite
    // Ein Stueck ist fertig (das „vorher“ wechselt): Tick, die echte Oberflaeche im Fenster neu zeichnen, „So klingt diese Stelle“
    if (derselbe && neu.vorher && neu.vorher !== alt.vorher) { inkluagentKlang('schritt'); inkluagentRahmenNachziehen(); }
    if (l.vorher && l.vorher.art === 'bild') inkluagentMithoerenSetzen({ art: 'bild', alt: l.vorher.text, status: l.vorher.status });
    else if (l.vorher && l.vorher.art === 'feld') inkluagentMithoerenSetzen({ art: 'feld', info: l.vorher.text });
    // Zu Ende: neu zeichnen und zum Ergebnis fahren (die fertige Datei in der Ablage). Auch ein Lauf, der zwischen zwei Nachfragen
    // begann und endete (Umwandeln, Herunterladen: der Agent wartet im Werkzeug darauf) — nicht beim ersten Blick nach dem Laden
    // der Seite (das waere ein alter Lauf), jedes Ende nur einmal. Ohne Datei in der Ablage (Word herunterladen ohne ABLAGE_WORD)
    // zeigt ein so kurzer Lauf wenigstens seine Station: die Ansicht, in der man ihn sieht
    if ((derselbe && alt.lauf.laeuft && !l.laeuft) || (!derselbe && !erster && !l.laeuft)) {
        inkluagentRahmenNachziehen();
        if (window._arbeitsLiveGefolgt !== l.nr) {
            window._arbeitsLiveGefolgt = l.nr;
            if (l.ergebnis) inkluagentFensterFolgen(l.ergebnis);
            else if (!derselbe && l.ansicht && l.project_id != null) inkluagentFensterFolgen({ art: 'ansicht', project_id: l.project_id, ansicht: l.ansicht });
        }
    }
    // Kamerafahrt zur Stelle, wenn sie sich geaendert hat (oder ein neuer Lauf beginnt)
    if (l.laeuft && l.stelle && (!derselbe || neu.stelle !== alt.stelle)) inkluagentFensterFolgen(l.stelle);
    // Steht ein Projekt im Fenster, zeigt dessen echte Oberflaeche Bild und Text; der Block oben bleibt nur Kopfzeile. Ohne Buehne
    // (etwa auf der Startseite) zeigt er das Bild, an dem der Lauf gerade ist.
    const buehne = !!(window._arbeitsSeite && window._arbeitsSeite.offen && (window._arbeitsSeite.projekt || window._arbeitsSeite.ablage));
    box.classList.toggle('arbeitsfenster-live-kopf', buehne);
    box.hidden = false;
    box.dataset.zustand = l.laeuft ? 'laeuft' : 'fertig';
    inkluagentTextSetzen(box.querySelector('#arbeitsLiveTitel'), inkluagentLiveTitel(l));
    const p = box.querySelector('#arbeitsLiveFortschritt');
    p.hidden = !(l.laeuft && l.gesamt > 1);
    if (!p.hidden) { p.max = Math.max(1, Number(l.gesamt) || 1); p.value = Math.min(p.max, Number(l.fertig) || 0); }
    const fig = box.querySelector('figure');
    const img = fig.querySelector('img');
    const gut = u => typeof u === 'string' && /^\/api\/images\/\d+\/(file|page-view)$/.test(u);
    const b = l.laeuft && l.stelle && l.stelle.art === 'bild' ? l.stelle : null;
    if (!buehne && b && (gut(b.bild_url) || gut(b.seite_url))) {
        const url = gut(b.bild_url) ? b.bild_url : b.seite_url;
        img.onerror = () => { if (gut(b.seite_url) && img.getAttribute('src') !== b.seite_url) img.src = b.seite_url; else fig.hidden = true; };
        const wechsel = img.getAttribute('src') !== url;
        if (wechsel) img.src = url;
        inkluagentTextSetzen(fig.querySelector('figcaption'), t('In Arbeit: {bild}', { bild: inkluagentLiveBildText(b) }));
        fig.hidden = false;
        // Die Hand zeigt auf das erste Bild eines Laufs (mit dem Wusch der Klangspur — darum nur einmal, nicht je Bild)
        if (wechsel && !derselbe && typeof zeigeHand === 'function') zeigeHand(fig);
    } else {
        fig.hidden = true;
    }
    const fertig = box.querySelector('#arbeitsLiveFertig');
    const v = l.vorher;
    const fertigText = (!buehne && v && v.art === 'bild') ? (v.status === 'error'
        ? t('Gerade fertig: {bild} – hat nicht geklappt', { bild: inkluagentLiveBildText(v) })
        : t('Gerade fertig: {bild}', { bild: inkluagentLiveBildText(v) })) : '';
    if (fertig._stand !== JSON.stringify([fertigText, v && v.text])) {
        fertig._stand = JSON.stringify([fertigText, v && v.text]);
        fertig.textContent = '';
        if (fertigText) {
            const st = document.createElement('strong');
            st.textContent = fertigText;
            fertig.appendChild(st);
            if (v.status !== 'error' && v.text) {
                const q = document.createElement('span');
                q.className = 'arbeitsfenster-live-alt';
                q.textContent = String(v.text);
                fertig.appendChild(q);
            }
        }
    }
    fertig.hidden = !fertigText;
    inkluagentArbeitsZeigen();   // Stand in der Kopfzeile nachziehen
}
// „Bild 3 · Seite 2 · Dokument 1: Name“ — wie die Bildunterschrift der Vorschau
function inkluagentLiveBildText(b) {
    if (!b) return '';
    const dok = b.dokument ? (b.dokument_nr ? t('Dokument {n}: {name}', { n: b.dokument_nr, name: String(b.dokument) }) : String(b.dokument)) : '';
    return [inkluagentZielText({ art: 'bild', nummer: b.nummer }), b.seite ? t('Seite {n}', { n: b.seite }) : '', dok].filter(Boolean).join(' · ');
}
// Eine Stelle in Worten (fuer „Stand vorlesen“): Bild, Feld, Dokument, Seite
function inkluagentLiveStelleText(x) {
    if (!x) return '';
    if (x.art === 'bild') return inkluagentLiveBildText(x);
    const dok = x.dokument ? (x.dokument_nr ? t('Dokument {n}: {name}', { n: x.dokument_nr, name: String(x.dokument) }) : String(x.dokument)) : '';
    if (x.art === 'feld') return [inkluagentZielText({ art: 'feld', nummer: x.nummer }), x.seite ? t('Seite {n}', { n: x.seite }) : '', dok].filter(Boolean).join(' · ');
    if (x.art === 'seite') return [t('Seite {n}', { n: x.seite }), dok].filter(Boolean).join(' · ');
    return dok;
}
// Fuer „Stand vorlesen“ und die Kopfzeile: was im Hintergrund laeuft, wo, und was gerade fertig wurde
function inkluagentLiveTexte() {
    const lv = window._arbeitsLive;
    const l = lv && lv.lauf;
    if (!l) return [];
    const out = [inkluagentLiveTitel(l)];
    const wo = l.laeuft ? inkluagentLiveStelleText(l.stelle) : '';
    if (wo) out.push(t('In Arbeit: {bild}', { bild: wo }));
    const v = l.vorher;
    if (v && v.art === 'bild') out.push(v.status === 'error' ? t('Gerade fertig: {bild} – hat nicht geklappt', { bild: inkluagentLiveBildText(v) })
                                                            : t('Gerade fertig: {bild}', { bild: inkluagentLiveBildText(v) }) + (v.text ? ': ' + String(v.text) : ''));
    else if (v && v.art === 'feld' && v.text) out.push(t('Gerade fertig: {bild}', { bild: inkluagentLiveStelleText(v) }) + ': ' + String(v.text));
    return out;
}
