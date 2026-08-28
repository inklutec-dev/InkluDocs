/* =============================================================================
 * formular.js — Quickinfo-Werkzeug (PDF-Formulare), Projektansicht
 * =============================================================================
 * 27.08.2026, Steve + Fable 5. Eigene Ansicht fuer Formular-Projekte
 * (tool "formular", project_type "pdfform"), bewusst getrennt von der
 * Bild-/Alt-Text-Ansicht in app.html — ein Formularfeld ist kein Bild.
 * Die FORM ist absichtlich dieselbe wie bei den Alt-Texten (Steve: "sich von
 * der Standardansicht gar nicht so abheben"): H1 Projekt, H2 Dokument (klappbar),
 * H3 Seite (klappbar) mit Seitenansicht und Seitentext, H4 Feld; ein Eingabefeld
 * je Feld mit Auto-Speichern (800 ms), Live-Ansagen ueber announce().
 *
 * Gemeinsame Helfer aus app.html/dashboard.js (globale Funktionen): t(),
 * announce(), escHtml(), uploadBlockHtml(), setupProjectDropzone(),
 * docDisplayName(), openDocRename(), openDocDelete(), downloadBlob().
 *
 * Datenquelle: GET /api/projects/{id}/felder (siehe backend/formular_api.py).
 * Stufe 2 (27.08.2026): KI-Vorschlaege wie bei den Alt-Texten — "Alle generieren"
 * fuellt nur Luecken, "Generieren"/"Neu generieren" am Feld ueberschreibt bewusst,
 * "Zurueck auf Original" bleibt. Jeder KI-Text traegt Sicherheit (hoch/mittel/
 * niedrig, nach Nachpruefung) und den Beleg-Satz; Filter "Nur unsichere".
 * Sicherheit: alle Texte aus dem Server laufen durch escHtml(); Eingaben
 * gehen als JSON an PATCH /api/felder/{id}; keine innerHTML-Zuweisung mit
 * unescapten Nutzerdaten.
 * GAST-ANSICHT (28.08.2026): dieselbe Ansicht im Gast-Modus (window.GUEST_MODE,
 * Freigabe-Link /freigabe/{token}). Daten kommen dann von /api/freigabe/{token}/
 * felder; der Gast (Herausgeber/Lektorat) liest, aendert Quickinfos von Hand und
 * setzt je Feld ein Urteil (Freigeben / Aenderung wuenschen) mit optionaler
 * Anmerkung — KEINE KI, keine Stammdaten, kein Export, kein Upload. Der Besitzer
 * laedt ueber „Zur Pruefung freigeben" ein (Dialog aus app.html: shareDialogHtml)
 * und sieht danach je Feld das juengste Urteil (Badge wie bei Bildern:
 * unifiedKeyFor/unifiedStatusLabel/unifiedStatusColor aus app.html) + Anmerkung.
 * ========================================================================== */
(function () {
    'use strict';

    // Gast-Modus: Token statt Projekt-ID, eigene Endpunkte, keine Schreibrechte
    // ausser Quickinfo-Text und Urteil.
    function gast() { return !!window.GUEST_MODE; }
    function gastBasis() { return '/api/freigabe/' + encodeURIComponent(window.SHARE_TOKEN || ''); }
    function felderUrl(key) { return gast() ? gastBasis() + '/felder' : '/api/projects/' + key + '/felder'; }
    function ausschnittUrl(fid) { return gast() ? gastBasis() + '/felder/' + fid + '/ausschnitt' : '/api/felder/' + fid + '/ausschnitt'; }
    function seitenansichtUrl(fid) { return gast() ? gastBasis() + '/felder/' + fid + '/page-view' : '/api/felder/' + fid + '/page-view'; }
    function rolle() { return window.SHARE_ROLE || 'kunde'; }
    let inReview = false;   // Besitzer: Projekt ist freigegeben -> Pruef-Badges + Anmerkungen zeigen

    const FELDART = {
        text: () => t('Textfeld'), checkbox: () => t('Kontrollkästchen'), radio: () => t('Auswahlknopf'),
        dropdown: () => t('Auswahlliste'), liste: () => t('Listenfeld'), button: () => t('Schaltfläche'),
        signatur: () => t('Unterschriftsfeld'), unbekannt: () => t('Feld'),
    };
    const LAGE = {
        links: () => t('links vom Feld'), rechts: () => t('rechts vom Feld'),
        oben: () => t('über dem Feld'), innen: () => t('im Feld'),
    };
    const QUELLE = {
        pdf: () => t('vorhanden (aus der PDF)'), hand: () => t('von Hand'),
        stammdaten: () => t('aus Stammdaten'), ki: () => t('KI-Vorschlag'), gast: () => t('vom Gast bearbeitet'),
        chat: () => t('im Chat bestätigt'),
    };

    // Auf/Zu-Zustand ueber Neu-Rendern hinweg (wie openDocs/openPages in app.html).
    let offeneDocs = new Set();
    let offeneSeiten = new Set();
    let zustandProjekt = null;
    let projektStatus = '';
    let exportZielDoc = null;     // Dokument-ID fuer den Export vom Knopf am Dokument, null = ganzes Projekt
    let aktuelleDocs = [];
    let nurOffene = false;
    let nurUnsichere = false;

    function feldartText(art) { return (FELDART[art] || FELDART.unbekannt)(); }

    // Michael 28.08.2026 (Punkt 1): Seite raus (steht in der Klappe darueber), Feldname rein.
    // Kurze Namen („1“, „E“ — Bankformular) werden als „Feldname 1“ vorgelesen, damit es nicht
    // wie eine Wiederholung der Feldnummer klingt; sprechende Namen stehen fuer sich.
    function feldUeberschrift(f) {
        const art = feldartText(f.feld_art);
        let name = istNamenlos(f) ? t('ohne Feldnamen') : (f.feld_name || '');
        if (name && !istNamenlos(f) && name.length <= 2) name = t('Feldname {name}', { name: name });
        return name ? t('Feld {n}, {name}, {art}', { n: f.feld_index, name: name, art: art })
                    : t('Feld {n}, {art}', { n: f.feld_index, art: art });
    }

    // Kontextabsatz: exakt das, was auch die KI (Stufe 2) sehen wird — der
    // Screenreader-Nutzer hoert dieselbe Grundlage wie die Maschine.
    function kontextHtml(f) {
        const teile = [];
        if (f.beschriftung) {
            const lage = LAGE[f.beschriftung_lage] ? ' (' + LAGE[f.beschriftung_lage]() + ')' : '';
            teile.push(t('Beschriftung im Formular: {b}', { b: escHtml(f.beschriftung) }) + lage + '.');
        } else {
            teile.push(t('Keine Beschriftung in der Nähe des Feldes erkannt.'));
        }
        if (f.gruppe) teile.push(t('Abschnitt: {g}', { g: escHtml(f.gruppe) }) + '.');
        if (f.optionen && f.optionen.length) teile.push(t('Optionen: {o}', { o: escHtml(f.optionen.join(', ')) }) + '.');
        if (f.seiten && f.seiten.length > 1) teile.push(t('Erscheint auf den Seiten {s}', { s: escHtml(f.seiten.join(', ')) }) + '.');
        if (istNamenlos(f)) teile.push(t('Dieses Feld hat keinen Feldnamen. Eine Quickinfo kann dafür nicht in die PDF geschrieben werden.'));
        return '<p class="feld-kontext" id="feld_kontext_' + f.id + '">' + teile.join(' ') + '</p>';
    }

    function istNamenlos(f) { return typeof f.anker === 'string' && f.anker.charAt(0) === '#'; }

    const SICHERHEIT = { hoch: () => t('sicher'), mittel: () => t('mittel'), niedrig: () => t('unsicher') };

    function statusText(f) {
        if (!f.quickinfo || !f.quickinfo.trim()) return t('Quickinfo fehlt');
        if (f.quelle === 'ki' && f.sicherheit) return t('KI-Vorschlag, {s}', { s: SICHERHEIT[f.sicherheit] ? SICHERHEIT[f.sicherheit]() : f.sicherheit });
        return (QUELLE[f.quelle] || QUELLE.hand)();
    }

    // Beleg-Satz + Hinweise der Nachpruefung: der Nutzer hoert, WARUM die KI so
    // formuliert hat — dieselbe Grundlage, die die Maschine geprueft hat.
    // Michael 28.08.2026 (Punkt 3): Beleg und Hinweise nicht mehr offen unter dem Feld (bis zu 100 Felder je
    // Seite), sondern als zugeklappte Klappe — fuer Screenreader-Nutzer bleibt die Grundlage erreichbar.
    function belegHtml(f) {
        const teile = [];
        if (f.quelle === 'ki') {
            if (f.beleg) teile.push(t('Beleg: „{b}“', { b: escHtml(f.beleg) }));
            else teile.push(t('Kein Beleg auf der Seite gefunden.'));
            (f.ki_hinweise || []).forEach(h => teile.push(escHtml(h)));
        }
        return '<details class="feld-beleg-details" id="feld_beleg_details_' + f.id + '" style="margin:0.3rem 0 0;"' + (f.quelle === 'ki' ? '' : ' hidden') + '>'
            + '<summary style="font-size:0.9rem;color:var(--text-muted);cursor:pointer;">' + t('Beleg und Hinweise') + '</summary>'
            + '<p class="feld-beleg" id="feld_beleg_' + f.id + '" style="font-size:0.9rem;color:var(--text-muted);margin:0.3rem 0 0;">' + teile.join(' ') + '</p></details>';
    }

    function feldCardHtml(f, treffer) {
        const offen = !(f.quickinfo && f.quickinfo.trim());
        const badges = [];
        const unsicher = f.quelle === 'ki' && f.sicherheit === 'niedrig';
        badges.push('<span class="badge ' + (offen || unsicher ? 'badge-pending' : 'badge-done') + '" id="feld_status_' + f.id + '">' + escHtml(statusText(f)) + '</span>');
        if (f.pflicht) badges.push('<span class="badge" style="background:#a15c00;color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;">' + t('Pflichtfeld') + '</span>');
        if (f.ausgefuellt) badges.push('<span class="badge" style="background:#4b5563;color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;">' + t('bereits ausgefüllt') + '</span>');
        // Pruef-Badge (Gast immer, Besitzer nur bei freigegebenem Projekt) — nur fuer
        // Felder, die einen Text haben; Felder ohne Quickinfo tragen kein Urteil.
        if ((gast() || inReview) && !offen && typeof unifiedKeyFor === 'function') {
            const lr = latestReview(f);
            const key = lr ? unifiedKeyFor(lr.role, lr.status) : 'neu';
            badges.push('<span class="badge" id="feld_unibadge_' + f.id + '" style="background:' + unifiedStatusColor(key)
                + ';color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;">' + unifiedStatusLabel(key)
                + (!gast() && f.review_note ? ' ' + t('— mit Anmerkung') : '') + '</span>');
        }
        const bild = f.hat_ausschnitt
            ? '<img src="' + ausschnittUrl(f.id) + '" alt="" class="image-preview feld-ausschnitt" loading="lazy">'
            : '';
        const vorschlag = (gast() || !(treffer && treffer.length)) ? '' : ''
            + '<div class="feld-stammdaten-treffer" id="feld_treffer_' + f.id + '" style="margin-top:0.5rem;">'
                + '<span id="feld_treffer_text_' + f.id + '">' + t('Vorschlag aus deinen Stammdaten: {q}', { q: escHtml(treffer[0].quickinfo) }) + '</span> '
                + '<button type="button" class="btn btn-secondary btn-small" onclick="Formular.ausStammdaten(' + f.id + ', ' + treffer[0].id + ')">' + t('Aus Stammdaten übernehmen') + '</button>'
              + '</div>';
        return ''
            + '<section class="image-review feld-review" id="feldcard_' + f.id + '" aria-labelledby="feld_heading_' + f.id + '" data-status="' + (offen ? 'offen' : 'beschrieben') + '" data-unsicher="' + (unsicher ? '1' : '0') + '">'
            // Michael 28.08.2026 (Punkt 2): Status rechts oben neben der Ueberschrift — spart eine Zeile je Feld.
            + '<div class="image-review-header feld-kopf" style="align-items:flex-start;margin-bottom:0.4rem;">'
            +   '<h4 id="feld_heading_' + f.id + '" class="image-heading" style="margin:0;">' + escHtml(feldUeberschrift(f)) + '</h4>'
            +   '<span class="feld-badges" style="display:flex;gap:0.35rem;flex-wrap:wrap;justify-content:flex-end;">' + badges.join(' ') + '</span>'
            + '</div>'
            + bild
            + kontextHtml(f)
            + '<label for="quickinfo_' + f.id + '" style="display:block;font-weight:600;margin-bottom:0.3rem;">' + t('Quickinfo')
            +   ' <span class="save-indicator" id="feld_saved_' + f.id + '">' + t('Gespeichert') + '</span></label>'
            + '<textarea class="alt-text-field quickinfo-field" id="quickinfo_' + f.id + '" data-feld-id="' + f.id + '" aria-describedby="feld_kontext_' + f.id + '" maxlength="1000"'
            +   (istNamenlos(f) ? ' disabled' : '')
            +   ' placeholder="' + t('Noch keine Quickinfo – hier eingeben oder aus Stammdaten übernehmen') + '">' + escHtml(f.quickinfo || '') + '</textarea>'
            + belegHtml(f)
            + vorschlag
            + (gast() ? gastUrteilHtml(f, offen) : ''
                + (inReview && f.review_note && typeof reviewNoteDetails === 'function' ? reviewNoteDetails(f.review_note, t('Anmerkung des Gastes anzeigen')) : '')
                + '<div style="margin-top:0.5rem;display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">'
                +   (istNamenlos(f) ? '' : '<button type="button" class="btn btn-secondary btn-small" id="feld_gen_' + f.id + '" onclick="Formular.generieren(' + f.id + ')">' + (offen ? t('Generieren') : t('Neu generieren')) + '</button>')
                +   (f.quickinfo_original ? '<button type="button" class="btn btn-secondary btn-small" id="feld_orig_' + f.id + '" onclick="Formular.original(' + f.id + ')">' + t('Zurück auf Original') + '</button>' : '')
                // KI-Fach (28.08.2026): liegt ein anderer Text ueber dem KI-Vorschlag, holt der Knopf ihn zurueck.
                +   '<button type="button" class="btn btn-secondary btn-small" id="feld_ki_' + f.id + '" onclick="Formular.kiVorschlag(' + f.id + ')"' + (f.quickinfo_ki && f.quickinfo_ki !== f.quickinfo ? '' : ' hidden') + '>' + t('KI-Vorschlag übernehmen') + '</button>'
                +   '<button type="button" class="btn btn-secondary btn-small" id="feld_sd_' + f.id + '" onclick="Formular.inStammdaten(' + f.id + ')"' + (offen ? ' disabled' : '') + '>' + t('In Stammdaten übernehmen') + '</button>'
                +   '<span id="feld_msg_' + f.id + '" role="status" aria-live="polite" style="font-size:0.85rem;"></span>'
                + '</div>')
            + '</section>';
    }

    // ─── Gast: Urteil je Feld (Muster: Gast-Pruefblock der Bildkarte in app.html) ───
    // Ein Klick = Status sofort gesetzt (aria-pressed am aktiven Knopf, Ansage per
    // announce()); die Anmerkung ist vom Status entkoppelt (natives <details>).
    // Der Block wird auch fuer leere Felder gerendert, nur versteckt — fuellt der
    // Gast das Feld, blendet speichern() ihn ein.
    function eigenerStatus(f) { const r = (f.reviews || {})[rolle()]; return (r && r.status) || 'offen'; }
    function statusLabel(status) { return (typeof unifiedStatusLabel === 'function') ? unifiedStatusLabel(unifiedKeyFor(rolle(), status)) : status; }
    function gastUrteilHtml(f, offen) {
        const st = eigenerStatus(f);
        const lr = latestReview(f);
        const anzeige = lr ? unifiedStatusLabel(unifiedKeyFor(lr.role, lr.status)) : t('Neu');
        return '<div class="guest-review feld-urteil" id="feld_urteil_' + f.id + '" data-status="' + escHtml(st) + '"' + (offen ? ' hidden' : '') + ' style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid var(--border);">'
            + '<p style="margin:0 0 0.4rem 0;font-weight:600;">' + t('Status:') + ' <span id="feld_rev_status_' + f.id + '">' + anzeige + '</span> <output id="feld_rev_saved_' + f.id + '" class="rev-saved"></output></p>'
            + '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">'
            +   '<button type="button" class="btn btn-secondary btn-small rev-frei" id="feld_btn_frei_' + f.id + '" aria-pressed="' + (st === 'freigegeben') + '" onclick="Formular.urteil(' + f.id + ', \'freigegeben\')">' + t('Freigeben') + '</button>'
            +   '<button type="button" class="btn btn-secondary btn-small" id="feld_btn_aend_' + f.id + '" aria-pressed="' + (st === 'zu_ueberarbeiten') + '" onclick="Formular.urteil(' + f.id + ', \'zu_ueberarbeiten\')">' + t('Änderung wünschen') + '</button>'
            + '</div>'
            + '<details class="guest-note-edit" id="feld_note_details_' + f.id + '" style="margin-top:0.5rem;">'
            +   '<summary id="feld_note_summary_' + f.id + '">' + (f.review_note ? t('Anmerkung bearbeiten') : t('Anmerkung hinzufügen')) + '</summary>'
            +   '<div style="margin-top:0.4rem;">'
            +     '<label for="feld_note_' + f.id + '" style="display:block;font-weight:600;margin-bottom:0.3rem;">' + t('Anmerkung (optional)') + '</label>'
            +     '<textarea id="feld_note_' + f.id + '" maxlength="2000" style="width:100%;min-height:60px;padding:0.5rem;border:1px solid var(--border);border-radius:4px;font-family:inherit;font-size:0.95rem;">' + escHtml(f.review_note || '') + '</textarea>'
            +     '<div style="margin-top:0.4rem;display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">'
            +       '<button type="button" class="btn btn-primary btn-small" onclick="Formular.anmerkungSpeichern(' + f.id + ')">' + t('Anmerkung speichern') + '</button>'
            +       '<button type="button" class="btn btn-delete btn-small" onclick="Formular.anmerkungLoeschen(' + f.id + ')">' + t('Anmerkung löschen') + '</button>'
            +       '<output id="feld_note_msg_' + f.id + '" style="font-size:0.85rem;"></output>'
            +     '</div>'
            +   '</div>'
            + '</details>'
            + '</div>';
    }

    function urteilAnzeigen(feldId, status) {
        const cont = document.getElementById('feld_urteil_' + feldId);
        if (cont) { cont.dataset.status = status; cont.hidden = false; }
        const st = document.getElementById('feld_rev_status_' + feldId);
        if (st) st.textContent = statusLabel(status);
        const ub = document.getElementById('feld_unibadge_' + feldId);
        if (ub && typeof unifiedKeyFor === 'function') {
            const key = unifiedKeyFor(rolle(), status);
            ub.textContent = unifiedStatusLabel(key); ub.style.background = unifiedStatusColor(key);
        }
        const bF = document.getElementById('feld_btn_frei_' + feldId);
        const bA = document.getElementById('feld_btn_aend_' + feldId);
        if (bF) bF.setAttribute('aria-pressed', status === 'freigegeben');
        if (bA) bA.setAttribute('aria-pressed', status === 'zu_ueberarbeiten');
    }

    async function urteilSenden(feldId, status, anmerkung) {
        try {
            const r = await fetch(gastBasis() + '/felder/' + feldId + '/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin', body: JSON.stringify({ status: status, comment: anmerkung }) });
            return r.ok;
        } catch (e) { return false; }
    }

    async function urteil(feldId, status) {
        const field = document.getElementById('feld_note_' + feldId);
        const anmerkung = field ? field.value.trim() : '';
        if (!(await urteilSenden(feldId, status, anmerkung))) { announce(t('Fehler beim Speichern.')); return; }
        urteilAnzeigen(feldId, status);
        const saved = document.getElementById('feld_rev_saved_' + feldId);
        if (saved) { saved.textContent = t('Gespeichert.'); setTimeout(() => { saved.textContent = ''; }, 4000); }
        announce(t('Status gesetzt: {s}', { s: statusLabel(status) }));
    }

    async function anmerkungSpeichern(feldId) {
        const cont = document.getElementById('feld_urteil_' + feldId);
        const status = (cont && cont.dataset.status) || 'offen';
        const field = document.getElementById('feld_note_' + feldId);
        const note = field ? field.value.trim() : '';
        const msg = document.getElementById('feld_note_msg_' + feldId);
        if (!(await urteilSenden(feldId, status, note))) { if (msg) msg.textContent = t('Fehler beim Speichern.'); announce(t('Fehler beim Speichern.')); return; }
        if (msg) msg.textContent = note ? t('Anmerkung gespeichert.') : t('Anmerkung entfernt.');
        const sum = document.getElementById('feld_note_summary_' + feldId);
        if (sum) sum.textContent = note ? t('Anmerkung bearbeiten') : t('Anmerkung hinzufügen');
        const det = document.getElementById('feld_note_details_' + feldId);
        if (det) det.open = false;
        if (sum) sum.focus();
        announce(note ? t('Anmerkung gespeichert.') : t('Anmerkung entfernt.'));
    }

    async function anmerkungLoeschen(feldId) {
        const field = document.getElementById('feld_note_' + feldId);
        if (!field || !field.value.trim()) { announce(t('Keine Anmerkung vorhanden.')); return; }
        field.value = '';
        await anmerkungSpeichern(feldId);
    }

    // ─── Gast: Pruefung abschliessen (Endpunkt /api/freigabe/{token}/complete aus main.py) ───
    function abschlussHtml() {
        return '<button class="btn btn-primary" id="fGuestCompleteBtn" onclick="Formular.abschlussOeffnen()">' + t('Prüfung abschließen') + '</button>'
            + '<button type="button" class="btn btn-secondary" id="guestExitBtn" onclick="guestExit()">' + t('Beenden') + '</button>'
            + '<dialog id="fGuestCompleteDialog" aria-labelledby="fGuestCompleteHeading" class="invite-dialog">'
            +   '<h2 id="fGuestCompleteHeading" style="margin-top:0;">' + t('Prüfung abschließen') + '</h2>'
            +   '<label for="fGuestCompleteMsg" style="display:block;font-weight:600;margin-bottom:0.3rem;">' + t('Anmerkung') + '</label>'
            +   '<textarea id="fGuestCompleteMsg" maxlength="2000" style="width:100%;min-height:70px;padding:0.5rem;border:1px solid var(--border);border-radius:4px;"></textarea>'
            +   '<div style="margin-top:0.8rem;display:flex;gap:0.5rem;flex-wrap:wrap;">'
            +     '<button class="btn btn-primary" id="fGuestCompleteSend" onclick="Formular.abschliessen()">' + t('Abschließen &amp; senden') + '</button>'
            +     '<button class="btn btn-secondary" onclick="Formular.abschlussSchliessen()">' + t('Abbrechen') + '</button>'
            +   '</div>'
            +   '<output id="fGuestCompleteStatus" style="display:block;margin-top:0.5rem;"></output>'
            + '</dialog>';
    }
    function abschlussOeffnen() {
        const dlg = document.getElementById('fGuestCompleteDialog');
        if (!dlg) return;
        if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
        const m = document.getElementById('fGuestCompleteMsg'); if (m) m.focus();
    }
    function abschlussSchliessen() {
        const dlg = document.getElementById('fGuestCompleteDialog');
        if (dlg && typeof dlg.close === 'function') dlg.close(); else if (dlg) dlg.removeAttribute('open');
    }
    async function abschliessen() {
        const main = document.getElementById('main');
        const msgEl = document.getElementById('fGuestCompleteMsg');
        const statusEl = document.getElementById('fGuestCompleteStatus');
        const sendBtn = document.getElementById('fGuestCompleteSend');
        if (sendBtn) sendBtn.disabled = true;
        if (statusEl) statusEl.textContent = t('Wird gesendet…');
        announce(t('Prüfung wird gesendet …'));
        try {
            const r = await fetch(gastBasis() + '/complete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin', body: JSON.stringify({ message: msgEl ? msgEl.value : '' }) });
            if (!r.ok) { if (statusEl) statusEl.textContent = t('Fehler beim Abschließen. Bitte erneut versuchen.'); if (sendBtn) sendBtn.disabled = false; return; }
            abschlussSchliessen();
            main.innerHTML = '<section class="card" style="max-width:32rem;margin:1.5rem 0;">'
                + '<h1 class="section-title" id="guestThanksHeading" tabindex="-1">' + t('Vielen Dank!') + '</h1>'
                + '<p>' + t('Ihre Prüfung wurde an den Ersteller gesendet. Sie können dieses Fenster schließen — oder über Ihren Link jederzeit zurückkommen und Änderungen vornehmen.') + '</p>'
                + '<div style="margin-top:1rem;"><button type="button" class="btn btn-primary" onclick="Formular.showProject(window.SHARE_TOKEN)">' + t('Zurück zum Projekt') + '</button></div></section>';
            const h = document.getElementById('guestThanksHeading'); if (h) h.focus();
            announce(t('Prüfung abgeschlossen und gesendet.'));
        } catch (e) {
            if (statusEl) statusEl.textContent = t('Netzwerkfehler. Bitte erneut versuchen.');
            if (sendBtn) sendBtn.disabled = false;
        }
    }

    // Hoerprobe: so klingt das Formular beim Durchgehen mit einem Screenreader —
    // Feld fuer Feld, in der Reihenfolge der Felder. Billig zu erzeugen, erklaert
    // jedem Sehenden in zehn Sekunden, warum Quickinfos wichtig sind.
    function hoerprobeHtml(felder) {
        const items = felder.map(f => {
            const art = feldartText(f.feld_art);
            const text = (f.quickinfo && f.quickinfo.trim()) ? escHtml(f.quickinfo) : '<em>' + t('ohne Bezeichnung') + '</em>';
            return '<li>' + text + ', ' + escHtml(art) + '</li>';
        }).join('');
        const offen = felder.filter(f => !(f.quickinfo && f.quickinfo.trim())).length;
        const satz = offen === 0
            ? t('Jedes Feld hat eine Quickinfo. So hört sich das Formular an:')
            : t('{n} Felder werden nur als „ohne Bezeichnung“ vorgelesen. So hört sich das Formular gerade an:', { n: offen });
        return '<details class="page-text-details doc-hoerprobe"><summary>' + t('Hörprobe: so liest ein Screenreader das Formular') + '</summary>'
            + '<div class="page-text-content" role="region" aria-label="' + t('Hörprobe') + '" tabindex="0"><p>' + satz + '</p><ol>' + items + '</ol></div></details>';
    }

    function hinweiseHtml(doc) {
        const h = doc.hinweise;
        if (!h) return '';
        const items = [];
        (h.uebersprungen || []).forEach(u => {
            if (u.art === 'ohne_name') items.push('<li>' + t('Feld Nr. {n} ({art}) hat keinen Feldnamen und kann nicht beschrieben werden.', { n: u.nummer, art: escHtml(feldartText(u.feld_art)) }) + '</li>');
            else if (u.art === 'ohne_darstellung') items.push('<li>' + t('Feld „{name}“ ({art}) hat keine sichtbare Darstellung auf einer Seite.', { name: escHtml(u.name || ''), art: escHtml(feldartText(u.feld_art)) }) + '</li>');
        });
        (h.warnungen || []).forEach(w => items.push('<li>' + escHtml(w) + '</li>'));
        if (!items.length) return '';
        return '<details class="page-text-details doc-hinweise"><summary>' + t('{n} Hinweise zu diesem Formular', { n: items.length }) + '</summary>'
            + '<div class="page-text-content" role="region" aria-label="' + t('Hinweise') + '" tabindex="0"><ul>' + items.join('') + '</ul></div></details>';
    }

    function seiteHtml(docKey, pageNum, felder, treffer) {
        const first = felder[0];
        const key = docKey + '_' + pageNum;
        const offen = felder.filter(f => !(f.quickinfo && f.quickinfo.trim())).length;
        const count = t('{n} Felder, {o} offen', { n: felder.length, o: offen });
        const seitenansicht = first.hat_seitenansicht
            ? '<details class="page-view-details"><summary>' + t('Seitenansicht anzeigen') + '</summary>'
              + '<img src="' + seitenansichtUrl(first.id) + '" alt="" class="page-view-image"></details>' : '';
        const seitentext = first.page_text
            ? '<details class="page-text-details"><summary>' + t('Seitentext anzeigen') + '</summary>'
              + '<div class="page-text-content" role="region" aria-label="' + t('Seitentext') + '" tabindex="0">' + escHtml(first.page_text) + '</div></details>' : '';
        const kopf = pageNum > 0 ? t('Seite {n}', { n: pageNum }) : t('Ohne Seite');
        return '<details class="page-section" data-page="' + key + '"' + (offeneSeiten.has(key) ? ' open' : '') + '>'
            + '<summary class="page-summary"><h3 class="page-heading" id="feld_page_' + key + '">' + kopf + ' <span class="page-count">(' + count + ')</span></h3></summary>'
            + seitenansicht + seitentext
            + felder.map(f => feldCardHtml(f, treffer[f.id])).join('')
            + '</details>';
    }

    function dokumentHtml(doc, pos, felder, treffer) {
        const docKey = doc.id;
        const name = escHtml(docDisplayName(doc));
        const seiten = new Map();
        felder.forEach(f => { const p = f.page_number || 0; if (!seiten.has(p)) seiten.set(p, []); seiten.get(p).push(f); });
        const inner = hinweiseHtml(doc) + hoerprobeHtml(felder)
            + Array.from(seiten.entries()).sort((a, b) => a[0] - b[0]).map(([p, fs]) => seiteHtml(docKey, p, fs, treffer)).join('');
        const offen = felder.filter(f => !(f.quickinfo && f.quickinfo.trim())).length;
        const meta = '(' + t('{n} Felder, {o} offen', { n: felder.length, o: offen }) + ')';
        const vh = t('– Formular „{name}“', { name: name });
        const docOffen = felder.filter(f => !(f.quickinfo && f.quickinfo.trim()) && !istNamenlos(f)).length;
        const docKi = felder.filter(f => f.quelle === 'ki' && !istNamenlos(f));
        const docKiSeiten = new Set(docKi.filter(f => f.page_number > 0).map(f => f.page_number)).size;
        const docBusy = projektStatus === 'processing' || projektStatus === 'extracting';
        return '<div class="doc-block">'
            + '<details class="doc-section" data-doc="' + docKey + '"' + (offeneDocs.has(docKey) ? ' open' : '') + '>'
            +   '<summary class="doc-summary"><h2 class="doc-heading" id="doc_heading_' + docKey + '">' + t('Dokument {n}: {name}', { n: pos, name: name }) + ' <span class="page-count">' + meta + '</span></h2></summary>'
            +   inner
            + '</details>'
            // Knoepfe je Dokument (Michael/Steve 28.08.2026): Alle generieren / n neu generieren + Exportieren nur fuer dieses Dokument.
            + (gast() ? '' : '<span class="doc-actions">'
            +   (docOffen && !docBusy ? '<button type="button" class="doc-action-btn" onclick="Formular.alleGenerieren(' + zustandProjekt + ', ' + docKey + ')">' + t('Alle generieren') + '<span class="visually-hidden"> ' + vh + '</span></button>'
                : (docKi.length && !docBusy ? '<button type="button" class="doc-action-btn" onclick="Formular.alleNeuGenerieren(' + zustandProjekt + ', ' + docKey + ')">' + t('{n} Quickinfos neu generieren, {p} Credits', { n: docKi.length, p: docKiSeiten }) + '<span class="visually-hidden"> ' + vh + '</span></button>' : ''))
            +   (felder.length ? '<button type="button" class="doc-action-btn" onclick="Formular.exportOeffnen(' + docKey + ')">' + t('Exportieren') + '<span class="visually-hidden"> ' + vh + '</span></button>' : '')
            +   '<button type="button" class="doc-action-btn" data-kind="formdoc" data-doc-id="' + docKey + '" data-doc-name="' + name + '" onclick="openDocRename(event)">' + t('Umbenennen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            +   '<button type="button" class="doc-action-btn doc-action-danger" data-kind="formdoc" data-doc-id="' + docKey + '" data-doc-name="' + name + '" data-doc-count="' + felder.length + '" onclick="openDocDelete(event)">' + t('Löschen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            + '</span>') + '</div>';
    }


    function kopfHtml(project, data) {
        const felder = data.felder, docs = data.documents;
        const offen = felder.filter(f => !(f.quickinfo && f.quickinfo.trim())).length;
        const title = (project.name && project.name.trim()) ? project.name : project.filename;
        let badge, badgeCls;
        if (project.status === 'extracting') { badge = t('Wird gelesen'); badgeCls = 'badge-processing'; }
        else if (project.status === 'error') { badge = t('Fehler'); badgeCls = 'badge-error'; }
        else if (!felder.length) { badge = t('Neu'); badgeCls = 'badge-pending'; }
        else if (offen === 0) { badge = t('Vollständig'); badgeCls = 'badge-done'; }
        else { badge = t('In Arbeit'); badgeCls = 'badge-pending'; }
        const unsicher = felder.filter(f => f.quelle === 'ki' && f.sicherheit === 'niedrig').length;
        const kiFelder = felder.filter(f => f.quelle === 'ki' && !istNamenlos(f));
        const kiSeiten = new Set(kiFelder.filter(f => f.page_number > 0).map(f => f.document_id + '_' + f.page_number)).size;
        const gen = data.generierung;
        let info = felder.length
            ? (gast()
                ? t('{n} Felder in {d} Dokumenten, {o} ohne Quickinfo.', { n: felder.length, d: docs.length, o: offen })
                : t('{n} Felder in {d} Dokumenten, {o} ohne Quickinfo. Stammdaten: {s} Einträge.', { n: felder.length, d: docs.length, o: offen, s: data.stammdaten_anzahl || 0 }))
            : t('Noch kein Formular hochgeladen.');
        if (unsicher) info += ' ' + t('{u} KI-Vorschläge unsicher.', { u: unsicher });
        if (project.status === 'processing' && gen) { badge = t('KI generiert'); badgeCls = 'badge-processing'; info += ' ' + t('Seite {i} von {n} wird bearbeitet.', { i: Math.min(gen.seiten_fertig + 1, gen.seiten_gesamt || 1), n: gen.seiten_gesamt || 1 }); }
        return '<div class="card">'
            + '<div class="card-header"><h1 id="projectName" class="card-name" tabindex="-1">' + escHtml(title) + '</h1>'
            + '<span class="badge ' + badgeCls + '" id="projectStatusBadge">' + badge + '</span></div>'
            + '<div class="card-info" id="projectHeadInfo"' + (gast() ? '' : ' data-docs="' + docs.length + '" data-stammdaten="' + (data.stammdaten_anzahl || 0) + '"') + '>' + info + '</div>'
            // Gast: nur Abschluss/Beenden + Filter „Nur offene Felder" — keine KI, keine
            // Stammdaten, kein Export, keine Sprach-/Prompt-Einstellungen.
            + (gast() && felder.length ? ''
                + '<div class="card-actions">' + abschlussHtml()
                +   '<div style="flex-basis:100%;margin-top:0.6rem;display:flex;gap:1.2rem;flex-wrap:wrap;"><label for="fNurOffene" class="context-toggle" style="display:inline-flex;align-items:center;gap:0.5rem;cursor:pointer;">'
                +     '<span style="font-weight:600;">' + t('Nur offene Felder anzeigen') + '</span>'
                +     '<input type="checkbox" id="fNurOffene"' + (nurOffene ? ' checked' : '') + ' onchange="Formular.filter(this.checked)" style="width:1.2rem;height:1.2rem;"></label></div>'
                + '</div>' : '')
            + (!gast() && felder.length ? ''
                + '<div class="card-actions">'
                // EIN Knopf (Steve 28.08.2026): „Alle generieren“, solange Felder offen sind (fuellt nur Luecken);
                // sind alle gefuellt, wird er zu „Alle neu generieren“ (ueberschreibt nur KI-Vorschlaege — Hand, PDF,
                // Stammdaten, Gast bleiben). Keine Rueckfrage: der Knopf nennt fuer den Screenreader Anzahl und Credits.
                +   (offen && project.status !== 'processing' ? '<button class="btn btn-primary" id="fGenAllBtn" onclick="Formular.alleGenerieren(' + project.id + ')">' + t('Alle generieren') + '</button>'
                    : (kiFelder.length && project.status !== 'processing' ? '<button class="btn btn-secondary" id="fGenAllBtn" data-modus="ki_neu" onclick="Formular.alleNeuGenerieren(' + project.id + ')">'
                        + t('{n} Quickinfos neu generieren, {p} Credits', { n: kiFelder.length, p: kiSeiten }) + '</button>' : ''))
                +   '<button class="btn btn-primary" id="fExportOpenBtn" onclick="Formular.exportOeffnen()">' + (docs.length > 1 ? t('Ganzes Projekt exportieren') : t('Exportieren')) + '</button>'
                +   '<button class="btn btn-secondary" id="fStammdatenBtn" onclick="Formular.stammdatenAnwenden(' + project.id + ')">' + t('Stammdaten auf alle Felder anwenden') + '</button>'
                +   '<a class="btn btn-secondary" href="/stammdaten">' + t('Meine Stammdaten öffnen') + '</a>'
                // Gast-Ansicht (28.08.2026): Einladung wie bei Bild-Projekten — Knopf + Dialog aus app.html.
                +   (typeof shareDialogHtml === 'function' ? shareDialogHtml(project) : '')
                +   '<dialog id="fExportPanel" class="invite-dialog" aria-labelledby="fExportHeading">'
                +     '<h2 id="fExportHeading" style="margin:0 0 0.6rem 0;">' + t('Export-Optionen') + '</h2>'
                +     '<p id="fExportSummary" role="status" style="margin:0 0 0.8rem 0;">' + t('{b} von {n} Feldern haben eine Quickinfo. Felder ohne Quickinfo bleiben in der PDF unverändert.', { b: felder.length - offen, n: felder.length }) + '</p>'
                +     '<div class="form-group" style="margin-bottom:0.8rem;"><label for="fExportFilename" style="display:block;font-weight:600;margin-bottom:0.3rem;">' + t('Dateiname (optional)') + '</label>'
                +       '<input type="text" id="fExportFilename" style="width:100%;padding:0.5rem;border:1px solid var(--border);border-radius:4px;font-size:0.95rem;"></div>'
                +     '<div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">'
                +       '<button class="btn btn-primary" onclick="Formular.exportieren(' + project.id + ', \'formular\')">' + t('Als PDF mit Quickinfos') + '</button>'
                +       '<button class="btn btn-secondary" onclick="Formular.exportieren(' + project.id + ', \'formular_csv\')">' + t('Als CSV (Feldliste)') + '</button>'
                +       '<button class="btn btn-secondary" onclick="Formular.exportSchliessen()">' + t('Abbrechen') + '</button>'
                +     '</div><span id="fExportStatus" role="status" aria-live="polite" style="display:block;margin-top:0.5rem;"></span>'
                +   '</dialog>'
                +   '<div style="flex-basis:100%;margin-top:0.6rem;display:flex;gap:1.2rem;flex-wrap:wrap;"><label for="fNurOffene" class="context-toggle" style="display:inline-flex;align-items:center;gap:0.5rem;cursor:pointer;">'
                +     '<span style="font-weight:600;">' + t('Nur offene Felder anzeigen') + '</span>'
                +     '<input type="checkbox" id="fNurOffene"' + (nurOffene ? ' checked' : '') + ' onchange="Formular.filter(this.checked)" style="width:1.2rem;height:1.2rem;"></label>'
                +   '<label for="fNurUnsichere" class="context-toggle" style="display:inline-flex;align-items:center;gap:0.5rem;cursor:pointer;">'
                +     '<span style="font-weight:600;">' + t('Nur unsichere KI-Vorschläge anzeigen') + '</span>'
                +     '<input type="checkbox" id="fNurUnsichere"' + (nurUnsichere ? ' checked' : '') + ' onchange="Formular.filterUnsicher(this.checked)" style="width:1.2rem;height:1.2rem;"></label></div>'
                // Sprache der Quickinfos + gespeicherte Prompts: dieselben Endpunkte und
                // Funktionen wie bei den Alt-Texten (app.html: setAltLanguage, populatePromptSelect, setPromptSetting).
                +   '<div style="flex-basis:100%;margin-top:0.6rem;display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">'
                +     '<label for="altLangSelect" style="font-weight:600;">' + t('Sprache der Quickinfos') + '</label>'
                +     '<select id="altLangSelect" onchange="setAltLanguage(' + project.id + ', this.value)" data-confirmed="' + escHtml(project.alt_language || 'de') + '" style="padding:0.4rem;border:1px solid var(--border,#ccc);border-radius:4px;font-size:0.9rem;">'
                +       ['de','en','da','fr','es','sv'].map(code => '<option value="' + code + '"' + ((project.alt_language || 'de') === code ? ' selected' : '') + '>' + ({de:'Deutsch',en:'English',da:'Dansk',fr:'Français',es:'Español',sv:'Svenska'})[code] + '</option>').join('')
                +     '</select><span id="altLangStatus" style="font-size:0.9rem;"></span></div>'
                +   '<div style="flex-basis:100%;margin-top:0.6rem;display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">'
                +     '<label for="ownPromptSelect" style="font-weight:600;">' + t('Gespeicherte Prompts') + '</label>'
                +     '<select id="ownPromptSelect" onchange="setPromptSetting(' + project.id + ', this.value)" style="padding:0.4rem;border:1px solid var(--border,#ccc);border-radius:4px;font-size:0.9rem;"><option value="">' + t('Kein eigener Prompt') + '</option></select></div>'
                + '</div>' : '')
            + '</div>';
    }

    function bindAutosave() {
        document.querySelectorAll('.quickinfo-field').forEach(ta => {
            let timer;
            ta.addEventListener('input', () => {
                clearTimeout(timer);
                timer = setTimeout(() => speichern(Number(ta.dataset.feldId), ta.value), 800);
            });
        });
        document.querySelectorAll('details.page-section').forEach(d => d.addEventListener('toggle', () => {
            if (d.open) offeneSeiten.add(d.dataset.page); else offeneSeiten.delete(d.dataset.page);
        }));
        document.querySelectorAll('details.doc-section').forEach(d => d.addEventListener('toggle', () => {
            const k = Number(d.dataset.doc);
            if (d.open) offeneDocs.add(k); else offeneDocs.delete(k);
        }));
    }

    // Michael 28.08.2026 (Punkt 5): Zaehler „n Felder, o offen“ in Seiten-/Dokument-Klappen und der
    // Kopfzeile liefen nach dem Tippen nicht mit — jetzt bei jeder Statusaenderung nachgezogen.
    function zaehlerAktualisieren() {
        const zaehle = (wurzel) => {
            const karten = wurzel.querySelectorAll('section.feld-review');
            return { n: karten.length, o: Array.from(karten).filter(c => c.dataset.status === 'offen').length };
        };
        document.querySelectorAll('details.page-section').forEach(s => {
            const z = zaehle(s); const el = s.querySelector('.page-count');
            if (el) el.textContent = '(' + t('{n} Felder, {o} offen', { n: z.n, o: z.o }) + ')';
        });
        document.querySelectorAll('details.doc-section').forEach(d => {
            const z = zaehle(d); const el = d.querySelector('.doc-heading .page-count');
            if (el) el.textContent = '(' + t('{n} Felder, {o} offen', { n: z.n, o: z.o }) + ')';
        });
        const info = document.getElementById('projectHeadInfo');
        if (info && info.dataset.docs) {
            const z = zaehle(document.getElementById('feldListe') || document);
            info.textContent = t('{n} Felder in {d} Dokumenten, {o} ohne Quickinfo. Stammdaten: {s} Einträge.', { n: z.n, d: info.dataset.docs, o: z.o, s: info.dataset.stammdaten || 0 });
            const badge = document.getElementById('projectStatusBadge');
            if (badge && !badge.classList.contains('badge-processing') && !badge.classList.contains('badge-error') && z.n) {
                badge.textContent = z.o === 0 ? t('Vollständig') : t('In Arbeit');
                badge.className = 'badge ' + (z.o === 0 ? 'badge-done' : 'badge-pending');
            }
        }
    }

    function statusSetzen(feldId, antwort) {
        const card = document.getElementById('feldcard_' + feldId);
        const badge = document.getElementById('feld_status_' + feldId);
        const sd = document.getElementById('feld_sd_' + feldId);
        const offen = antwort.status === 'offen';
        if (card) card.dataset.status = offen ? 'offen' : 'beschrieben';
        if (badge) {
            badge.textContent = offen ? t('Quickinfo fehlt') : (QUELLE[antwort.quelle] || QUELLE.hand)();
            badge.className = 'badge ' + (offen ? 'badge-pending' : 'badge-done');
        }
        if (sd) sd.disabled = offen;
        const gen = document.getElementById('feld_gen_' + feldId);
        if (gen) gen.textContent = offen ? t('Generieren') : t('Neu generieren');
        const unsicher = antwort.quelle === 'ki' && antwort.sicherheit === 'niedrig';
        if (card) card.dataset.unsicher = unsicher ? '1' : '0';
        if (badge && antwort.quelle === 'ki' && antwort.sicherheit) {
            badge.textContent = t('KI-Vorschlag, {s}', { s: SICHERHEIT[antwort.sicherheit] ? SICHERHEIT[antwort.sicherheit]() : antwort.sicherheit });
            badge.className = 'badge ' + (unsicher ? 'badge-pending' : 'badge-done');
        }
        const beleg = document.getElementById('feld_beleg_' + feldId);
        const belegDet = document.getElementById('feld_beleg_details_' + feldId);
        if (beleg) {
            if (antwort.quelle === 'ki') {
                const teile = [antwort.beleg ? t('Beleg: „{b}“', { b: escHtml(antwort.beleg) }) : t('Kein Beleg auf der Seite gefunden.')];
                (antwort.ki_hinweise || []).forEach(h => teile.push(escHtml(h)));
                beleg.innerHTML = teile.join(' '); if (belegDet) belegDet.hidden = false;
            } else { beleg.innerHTML = ''; if (belegDet) belegDet.hidden = true; }
        }
        zaehlerAktualisieren();
        // Beim Bearbeiten bewusst NICHT ausblenden, auch wenn der Filter „nur offene" aktiv ist (Fokus bleibt im Feld).
    }

    async function generieren(feldId) {
        const btn = document.getElementById('feld_gen_' + feldId);
        const msg = document.getElementById('feld_msg_' + feldId);
        if (btn) btn.disabled = true;
        if (msg) msg.textContent = t('Quickinfo wird generiert …');
        announce(t('Quickinfo wird generiert …'));
        try {
            const res = await fetch('/api/felder/' + feldId + '/generieren', { method: 'POST' });
            const d = await res.json().catch(() => ({}));
            if (!res.ok) { const m = d.detail || t('Generieren fehlgeschlagen.'); if (msg) msg.textContent = m; announce(m); return; }
            const ta = document.getElementById('quickinfo_' + feldId);
            if (d.uebernommen === false) {
                // KI-Fach: der eigene Text bleibt, der Vorschlag wartet hinter dem Knopf.
                kiKnopf(feldId, true);
                const m2 = t('KI-Vorschlag erzeugt, {s}: „{q}“ – dein Text bleibt. Übernehmen über den Knopf „KI-Vorschlag übernehmen“.', { s: SICHERHEIT[d.sicherheit] ? SICHERHEIT[d.sicherheit]() : '', q: d.ki_vorschlag || '' });
                if (msg) msg.textContent = m2;
                announce(m2);
                const kb = document.getElementById('feld_ki_' + feldId); if (kb) kb.focus();
                return;
            }
            if (ta) ta.value = d.quickinfo || '';
            statusSetzen(feldId, d);
            kiKnopf(feldId, false);
            if (msg) msg.textContent = '';
            announce(t('Quickinfo generiert, {s}: {q}', { s: SICHERHEIT[d.sicherheit] ? SICHERHEIT[d.sicherheit]() : '', q: d.quickinfo }));
            if (ta) ta.focus();
        } catch (e) {
            if (msg) msg.textContent = t('Verbindungsfehler.');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    async function alleGenerieren(projectId, docId) {
        const btn = document.getElementById('fGenAllBtn');
        if (btn) btn.disabled = true;
        try {
            const res = await fetch('/api/projects/' + projectId + '/quickinfos/generieren', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(docId ? { document_id: docId } : {}) });
            const d = await res.json().catch(() => ({}));
            if (!res.ok) { announce(d.detail || t('Generieren fehlgeschlagen.')); if (btn) btn.disabled = false; return; }
            if (!d.gestartet) { announce(t('Keine offenen Felder – nichts zu generieren.')); if (btn) btn.disabled = false; return; }
            announce(t('Generierung gestartet für {n} offene Felder. Vorhandene Quickinfos bleiben unverändert.', { n: d.offen }));
            await showProject(projectId);
        } catch (e) { announce(t('Verbindungsfehler.')); if (btn) btn.disabled = false; }
    }

    async function alleNeuGenerieren(projectId, docId) {
        const btn = document.getElementById('fGenAllBtn');
        if (btn) btn.disabled = true;
        try {
            const body = { modus: 'ki_neu' };
            if (docId) body.document_id = docId;
            const res = await fetch('/api/projects/' + projectId + '/quickinfos/generieren', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const d = await res.json().catch(() => ({}));
            if (!res.ok) { announce(d.detail || t('Generieren fehlgeschlagen.')); if (btn) btn.disabled = false; return; }
            if (!d.gestartet) { announce(t('Keine KI-Vorschläge vorhanden – nichts zu generieren.')); if (btn) btn.disabled = false; return; }
            announce(t('Neu-Generierung gestartet für {n} Felder. Texte von Hand, aus der PDF, aus Stammdaten und vom Gast bleiben unverändert.', { n: d.offen }));
            await showProject(projectId);
        } catch (e) { announce(t('Verbindungsfehler.')); if (btn) btn.disabled = false; }
    }

    async function speichern(feldId, text) {
        try {
            const res = gast()
                ? await fetch(gastBasis() + '/felder/' + feldId + '/quickinfo', { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin', body: JSON.stringify({ quickinfo: text }) })
                : await fetch('/api/felder/' + feldId, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ quickinfo: text }) });
            if (!res.ok) { announce(t('Speichern fehlgeschlagen.')); return; }
            const d = await res.json();
            statusSetzen(feldId, d);
            if (!gast()) kiKnopf(feldId, !!(d.quickinfo_ki && d.quickinfo_ki !== d.quickinfo));
            if (gast()) {
                // Urteil-Block bei erstmals gefuelltem Feld einblenden; Auto-Status
                // 'in_bearbeitung' (Server) sichtbar machen, gesetztes Urteil bleibt.
                const cont = document.getElementById('feld_urteil_' + feldId);
                if (cont) cont.hidden = d.status === 'offen';
                if (d.auto_status) urteilAnzeigen(feldId, d.auto_status);
            }
            const ind = document.getElementById('feld_saved_' + feldId);
            if (ind) { ind.classList.add('visible'); setTimeout(() => ind.classList.remove('visible'), 2000); }
        } catch (e) { announce(t('Verbindungsfehler beim Speichern.')); }
    }

    function kiKnopf(feldId, sichtbar) {
        const kb = document.getElementById('feld_ki_' + feldId);
        if (kb) kb.hidden = !sichtbar;
    }

    async function kiVorschlag(feldId) {
        const res = await fetch('/api/felder/' + feldId + '/ki-vorschlag', { method: 'POST' });
        const d = await res.json().catch(() => ({}));
        if (!res.ok) { announce(d.detail || t('Übernahme fehlgeschlagen.')); return; }
        const ta = document.getElementById('quickinfo_' + feldId);
        if (ta) ta.value = d.quickinfo || '';
        statusSetzen(feldId, d);
        kiKnopf(feldId, false);
        announce(t('KI-Vorschlag übernommen: {q}', { q: d.quickinfo }));
        if (ta) ta.focus();
    }

    async function original(feldId) {
        const res = await fetch('/api/felder/' + feldId + '/original', { method: 'POST' });
        if (!res.ok) { announce(t('Zurücksetzen fehlgeschlagen.')); return; }
        const d = await res.json();
        const ta = document.getElementById('quickinfo_' + feldId);
        if (ta) ta.value = d.quickinfo || '';
        statusSetzen(feldId, d);
        announce(t('Quickinfo auf das Original zurückgesetzt.'));
    }

    async function inStammdaten(feldId) {
        const msg = document.getElementById('feld_msg_' + feldId);
        const res = await fetch('/api/felder/' + feldId + '/stammdaten', { method: 'POST' });
        if (!res.ok) { const e = await res.json().catch(() => ({})); const m = e.detail || t('Übernahme fehlgeschlagen.'); if (msg) msg.textContent = m; announce(m); return; }
        const m = t('In deine Stammdaten übernommen.');
        if (msg) msg.textContent = m;
        announce(m);
    }

    async function ausStammdaten(feldId, stammdatenId) {
        const res = await fetch('/api/felder/' + feldId + '/stammdaten-uebernehmen', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ stammdaten_id: stammdatenId }) });
        if (!res.ok) { announce(t('Übernahme fehlgeschlagen.')); return; }
        const d = await res.json();
        const ta = document.getElementById('quickinfo_' + feldId);
        if (ta) ta.value = d.quickinfo || '';
        statusSetzen(feldId, d);
        const tr = document.getElementById('feld_treffer_' + feldId);
        if (tr) tr.hidden = true;
        announce(t('Quickinfo aus Stammdaten übernommen: {q}', { q: d.quickinfo }));
        if (ta) ta.focus();
    }

    async function stammdatenAnwenden(projectId) {
        // Michael 28.08.2026 (Punkt 4): auf ALLE Felder — ersetzt auch PDF-Originale und KI-Texte, nie Hand/Gast/Chat.
        const res = await fetch('/api/projects/' + projectId + '/stammdaten-anwenden', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ nur_offene: false }) });
        if (!res.ok) { announce(t('Stammdaten konnten nicht angewendet werden.')); return; }
        const d = await res.json();
        announce((d.uebernommen === 1 ? t('1 Quickinfo aus Stammdaten übernommen.') : t('{n} Quickinfos aus Stammdaten übernommen.', { n: d.uebernommen })) + ' ' + t('Texte von Hand, vom Gast und aus dem Chat bleiben unverändert.'));
        await showProject(projectId);
    }

    function filterUnsicher(nur) {
        nurUnsichere = nur;
        filter(nurOffene, true);
        announce(nur ? t('Nur unsichere KI-Vorschläge werden angezeigt.') : t('Alle Felder werden angezeigt.'));
    }

    function filter(nur, still) {
        nurOffene = nur;
        document.querySelectorAll('.feld-review').forEach(c => {
            c.hidden = (nur && c.dataset.status !== 'offen') || (nurUnsichere && c.dataset.unsicher !== '1');
        });
        nur = nur || nurUnsichere;   // leere Klappen auch beim Unsicher-Filter ausblenden
        // Leer gewordene Seiten- und Dokument-Klappen mit ausblenden (sonst hoert der
        // Screenreader eine Ueberschrift ohne Inhalt).
        document.querySelectorAll('details.page-section').forEach(s => {
            s.hidden = nur && s.querySelectorAll('.feld-review:not([hidden])').length === 0;
        });
        document.querySelectorAll('.doc-block').forEach(b => {
            b.hidden = nur && b.querySelectorAll('.feld-review:not([hidden])').length === 0;
        });
        if (!still) announce(nurOffene ? t('Nur offene Felder werden angezeigt.') : t('Alle Felder werden angezeigt.'));
    }

    function exportOeffnen(docId) {
        const panel = document.getElementById('fExportPanel');
        if (!panel) return;
        const input = document.getElementById('fExportFilename');
        if (input) input.value = '';
        // Export-Ziel (Steve 28.08.2026): keine Auswahlliste — Knopf am Dokument = nur dieses Dokument,
        // Hauptknopf = ganzes Projekt (ZIP bei mehreren Dokumenten).
        exportZielDoc = docId || null;
        const head = document.getElementById('fExportHeading');
        if (head) {
            const doc = docId ? (aktuelleDocs || []).find(d => d.id === docId) : null;
            head.textContent = doc ? t('Dokument „{name}“ exportieren', { name: docDisplayName(doc) })
                : ((aktuelleDocs || []).length > 1 ? t('Ganzes Projekt exportieren ({n} Dokumente)', { n: aktuelleDocs.length }) : t('Exportieren'));
        }
        if (typeof panel.showModal === 'function') panel.showModal(); else panel.setAttribute('open', '');
        announce(t('Export-Optionen geöffnet.'));
        exportPreisLaden();
    }

    // Export-Staffel (28.08.2026): Preis und Guthaben fuer den gewaehlten Umfang in die Zusammenfassung.
    async function exportPreisLaden() {
        const el = document.getElementById('fExportSummary');
        if (!el || !zustandProjekt) return;
        const body = {};
        if (exportZielDoc) body.document_id = exportZielDoc;
        try {
            const res = await fetch('/api/projects/' + zustandProjekt + '/export/preis', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            if (!res.ok) return;
            const d = await res.json();
            const basis = el.dataset.basis || el.textContent; el.dataset.basis = basis;
            el.textContent = basis + ' ' + (typeof exportPreisText === 'function' ? exportPreisText(d.preis, d.verfuegbar) : '');
        } catch (e) { /* Preis ist Komfort, kein Blocker */ }
    }

    function exportSchliessen(silent) {
        const panel = document.getElementById('fExportPanel');
        if (panel && panel.open) { if (typeof panel.close === 'function') panel.close(); else panel.removeAttribute('open'); }
        if (!silent) announce(t('Export abgebrochen.'));
    }

    let exportLaeuft = false;
    async function exportieren(projectId, format) {
        if (exportLaeuft) return;   // Doppelklick-Sperre: ein Export je Dialog
        exportLaeuft = true;
        const knoepfe = Array.from(document.querySelectorAll('#fExportPanel button'));
        knoepfe.forEach(b => { b.disabled = true; });
        try {
            await _exportieren(projectId, format);
        } finally {
            exportLaeuft = false;
            knoepfe.forEach(b => { b.disabled = false; });
        }
    }

    async function _exportieren(projectId, format) {
        const statusEl = document.getElementById('fExportStatus');
        const body = {};
        if (exportZielDoc) body.document_id = exportZielDoc;
        const name = (document.getElementById('fExportFilename') || {}).value || '';
        if (name.trim()) body.filename = name.trim();
        if (statusEl) statusEl.textContent = t('Wird exportiert...');
        announce(t('Export läuft …'));
        try {
            const res = await fetch('/api/projects/' + projectId + '/export/' + format, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            if (typeof exportCreditsAbgefangen === 'function' && await exportCreditsAbgefangen(res)) { if (statusEl) statusEl.textContent = ''; return; }
            if (!res.ok) {
                const e = await res.json().catch(() => ({}));
                const m = e.detail || t('Fehler beim Export.');
                if (statusEl) statusEl.textContent = m;
                announce(m);
                return;
            }
            const blob = await res.blob();
            const cd = res.headers.get('Content-Disposition') || '';
            const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
            let serverName = null;
            if (m) { try { serverName = decodeURIComponent(m[1]); } catch (e) { serverName = m[1]; } }
            const fallback = (body.filename || 'formular') + (format === 'formular_csv' ? '.csv' : '.pdf');
            downloadBlob(blob, serverName || fallback);
            const warn = res.headers.get('X-Export-Warnings');
            let ansage = t('{name} wurde heruntergeladen.', { name: serverName || fallback });
            if (warn) { try { const w = JSON.parse(warn); if (w.length) ansage += ' ' + t('{n} Hinweise: {w}', { n: w.length, w: w.join(' ') }); } catch (e) { /* nur Anzeige */ } }
            announce(ansage);
            exportSchliessen(true);
        } catch (e) {
            if (statusEl) statusEl.textContent = t('Verbindungsfehler.');
        }
    }

    async function showProject(projectId, erneut) {
        // Im Gast-Modus ist projectId der Freigabe-Token (app.html reicht ihn durch).
        const main = document.getElementById('main');
        const res = await fetch(felderUrl(projectId), { credentials: 'same-origin' });
        if (res.status === 401) { if (gast() && typeof initGuest === 'function') { return initGuest(); } window.location.href = '/'; return; }
        if (!res.ok) { main.innerHTML = '<div class="card"><p>' + t('Projekt konnte nicht geladen werden.') + '</p></div>'; return; }
        const data = await res.json();
        const project = data.project;
        projektStatus = project.status || '';
        aktuelleDocs = data.documents || [];
        inReview = !gast() && !!data.in_review;
        if (gast() && data.role) window.SHARE_ROLE = data.role;
        if (zustandProjekt !== projectId) { offeneDocs = new Set(); offeneSeiten = new Set(); zustandProjekt = projectId; }
        // currentDocProjectId (Umbenennen/Loeschen-Dialoge) setzt app.html.showProject vor der Weiche.
        const felderJeDoc = new Map();
        data.felder.forEach(f => { if (!felderJeDoc.has(f.document_id)) felderJeDoc.set(f.document_id, []); felderJeDoc.get(f.document_id).push(f); });
        const docsHtml = data.documents.map((d, i) => dokumentHtml(d, i + 1, felderJeDoc.get(d.id) || [], data.stammdaten_treffer || {})).join('');
        main.innerHTML = kopfHtml(project, data)
            + (gast() ? '' : uploadBlockHtml(project))
            + '<div id="feldListe">' + (data.felder.length ? docsHtml : (gast() ? '<div class="card"><p>' + t('Dieses Formular enthält noch keine Felder.') + '</p></div>' : '')) + '</div>'
            // InkluAgent (28.08.2026): derselbe Chat-Kasten wie bei den Alt-Texten (app.html),
            // Variante formular — nur fuer den Besitzer, Gaeste bekommen keinen Chatbot.
            + (!gast() && data.felder.length && typeof inkluagentSectionHtml === 'function' ? inkluagentSectionHtml(project.id, 'formular') : '');
        bindAutosave();
        if (!gast() && data.felder.length && typeof inkluagentInit === 'function') inkluagentInit(project.id);
        if (nurOffene || nurUnsichere) filter(nurOffene, true);
        if (!gast()) setupProjectDropzone(projectId);
        if (!gast() && data.felder.length && typeof populatePromptSelect === 'function') populatePromptSelect(project.id, project.prompt_id);
        const h1 = document.getElementById('projectName');
        if (h1 && !erneut) h1.focus();
        // Laeuft die Extraktion oder die KI-Generierung noch: weiter abfragen, ohne den
        // Fokus zu bewegen; am Ende einmal ansagen. (Gast: kein Polling — er sieht den Stand beim Laden.)
        if (!gast() && (project.status === 'extracting' || project.status === 'processing')) {
            setTimeout(async () => {
                try {
                    const r = await fetch('/api/projects/' + projectId);
                    if (!r.ok) return;
                    const d = await r.json();
                    if (d.project && d.project.status !== project.status) {
                        if (project.status === 'extracting') announce(t('Formular gelesen.'));
                        else {
                            const g = (await (await fetch('/api/projects/' + projectId + '/felder')).json()).generierung || {};
                            const f = (g.fehler || []).length ? ' ' + t('Hinweise: {w}', { w: g.fehler.join(' ') }) : '';
                            announce(t('Generierung abgeschlossen: {n} Quickinfos neu.', { n: g.felder_neu || 0 }) + f);
                        }
                        showProject(projectId);
                    } else {
                        showProject(projectId, true);
                    }
                } catch (e) { /* naechster Versuch beim naechsten Aufruf */ }
            }, 2500);
        }
    }

    // Aktionen des InkluAgent (refresh_feld aus agent_loop.py): Textfeld + Badge + Beleg
    // live setzen, ohne Neu-Rendern (ungespeicherte Eingaben anderer Felder bleiben).
    function chatAktionen(actions) {
        let n = 0;
        (actions || []).forEach(a => {
            if (!a || a.type !== 'refresh_feld' || !a.feld_id) return;
            const ta = document.getElementById('quickinfo_' + a.feld_id);
            if (!ta) return;
            if (a.uebernommen === false) { kiKnopf(a.feld_id, true); n += 1; return; }
            ta.value = a.quickinfo || '';
            statusSetzen(a.feld_id, a);
            kiKnopf(a.feld_id, !!(a.quickinfo_ki && a.quickinfo_ki !== a.quickinfo));
            const ind = document.getElementById('feld_saved_' + a.feld_id);
            if (ind) { ind.classList.add('visible'); setTimeout(() => ind.classList.remove('visible'), 2000); }
            n += 1;
        });
        if (n > 0) announce(n === 1 ? t('InkluAgent hat 1 Quickinfo aktualisiert.') : t('InkluAgent hat {n} Quickinfos aktualisiert.', { n: n }));
    }

    window.Formular = { showProject, original, kiVorschlag, inStammdaten, ausStammdaten, stammdatenAnwenden, filter, filterUnsicher, chatAktionen,
                        generieren, alleGenerieren, alleNeuGenerieren, exportOeffnen, exportSchliessen, exportieren,
                        urteil, anmerkungSpeichern, anmerkungLoeschen, abschlussOeffnen, abschlussSchliessen, abschliessen };
})();
