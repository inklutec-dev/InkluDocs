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
 * Sicherheit: alle Texte aus dem Server laufen durch escHtml(); Eingaben
 * gehen als JSON an PATCH /api/felder/{id}; keine innerHTML-Zuweisung mit
 * unescapten Nutzerdaten.
 * ========================================================================== */
(function () {
    'use strict';

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
        stammdaten: () => t('aus Stammdaten'), ki: () => t('KI-Vorschlag'),
    };

    // Auf/Zu-Zustand ueber Neu-Rendern hinweg (wie openDocs/openPages in app.html).
    let offeneDocs = new Set();
    let offeneSeiten = new Set();
    let zustandProjekt = null;
    let nurOffene = false;

    function feldartText(art) { return (FELDART[art] || FELDART.unbekannt)(); }

    function feldUeberschrift(f) {
        const art = feldartText(f.feld_art);
        return f.page_number > 0
            ? t('Feld {n}, {art}, Seite {p}', { n: f.feld_index, art: art, p: f.page_number })
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
        if (f.feld_name) teile.push(t('Technischer Feldname: {n}', { n: escHtml(f.feld_name) }) + '.');
        return '<p class="feld-kontext" id="feld_kontext_' + f.id + '">' + teile.join(' ') + '</p>';
    }

    function statusText(f) {
        if (!f.quickinfo || !f.quickinfo.trim()) return t('Quickinfo fehlt');
        return (QUELLE[f.quelle] || QUELLE.hand)();
    }

    function feldCardHtml(f, treffer) {
        const offen = !(f.quickinfo && f.quickinfo.trim());
        const badges = [];
        badges.push('<span class="badge ' + (offen ? 'badge-pending' : 'badge-done') + '" id="feld_status_' + f.id + '">' + escHtml(statusText(f)) + '</span>');
        if (f.pflicht) badges.push('<span class="badge" style="background:#a15c00;color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;">' + t('Pflichtfeld') + '</span>');
        if (f.ausgefuellt) badges.push('<span class="badge" style="background:#4b5563;color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;">' + t('bereits ausgefüllt') + '</span>');
        const bild = f.hat_ausschnitt
            ? '<img src="/api/felder/' + f.id + '/ausschnitt" alt="" class="image-preview feld-ausschnitt" loading="lazy">'
            : '';
        const vorschlag = (treffer && treffer.length)
            ? '<div class="feld-stammdaten-treffer" id="feld_treffer_' + f.id + '" style="margin-top:0.5rem;">'
                + '<span id="feld_treffer_text_' + f.id + '">' + t('Vorschlag aus deinen Stammdaten: {q}', { q: escHtml(treffer[0].quickinfo) }) + '</span> '
                + '<button type="button" class="btn btn-secondary btn-small" onclick="Formular.ausStammdaten(' + f.id + ', ' + treffer[0].id + ')">' + t('Aus Stammdaten übernehmen') + '</button>'
              + '</div>'
            : '';
        return ''
            + '<section class="image-review feld-review" id="feldcard_' + f.id + '" aria-labelledby="feld_heading_' + f.id + '" data-status="' + (offen ? 'offen' : 'beschrieben') + '">'
            + '<h4 id="feld_heading_' + f.id + '" class="image-heading">' + escHtml(feldUeberschrift(f)) + '</h4>'
            + '<div class="image-review-header">' + badges.join(' ') + '</div>'
            + bild
            + kontextHtml(f)
            + '<label for="quickinfo_' + f.id + '" style="display:block;font-weight:600;margin-bottom:0.3rem;">' + t('Quickinfo')
            +   ' <span class="save-indicator" id="feld_saved_' + f.id + '">' + t('Gespeichert') + '</span></label>'
            + '<textarea class="alt-text-field quickinfo-field" id="quickinfo_' + f.id + '" data-feld-id="' + f.id + '" aria-describedby="feld_kontext_' + f.id + '"'
            +   ' placeholder="' + t('Noch keine Quickinfo – hier eingeben oder aus Stammdaten übernehmen') + '">' + escHtml(f.quickinfo || '') + '</textarea>'
            + vorschlag
            + '<div style="margin-top:0.5rem;display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">'
            +   (f.quickinfo_original ? '<button type="button" class="btn btn-secondary btn-small" id="feld_orig_' + f.id + '" onclick="Formular.original(' + f.id + ')">' + t('Zurück auf Original') + '</button>' : '')
            +   '<button type="button" class="btn btn-secondary btn-small" id="feld_sd_' + f.id + '" onclick="Formular.inStammdaten(' + f.id + ')"' + (offen ? ' disabled' : '') + '>' + t('In Stammdaten übernehmen') + '</button>'
            +   '<span id="feld_msg_' + f.id + '" role="status" aria-live="polite" style="font-size:0.85rem;"></span>'
            + '</div>'
            + '</section>';
    }

    // Hoerprobe: so klingt das Formular beim Durchgehen mit einem Screenreader —
    // Feld fuer Feld, in der Reihenfolge der Felder. Billig zu erzeugen, erklaert
    // jedem Sehenden in zehn Sekunden, warum Quickinfos wichtig sind.
    function hoerprobeHtml(docId, felder) {
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
              + '<img src="/api/felder/' + first.id + '/page-view" alt="" class="page-view-image"></details>' : '';
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
        const inner = hinweiseHtml(doc) + hoerprobeHtml(docKey, felder)
            + Array.from(seiten.entries()).sort((a, b) => a[0] - b[0]).map(([p, fs]) => seiteHtml(docKey, p, fs, treffer)).join('');
        const offen = felder.filter(f => !(f.quickinfo && f.quickinfo.trim())).length;
        const meta = '(' + t('{n} Felder, {o} offen', { n: felder.length, o: offen }) + ')';
        const vh = t('– Dokument „{name}"', { name: name });
        return '<div class="doc-block">'
            + '<details class="doc-section" data-doc="' + docKey + '"' + (offeneDocs.has(docKey) ? ' open' : '') + '>'
            +   '<summary class="doc-summary"><h2 class="doc-heading" id="doc_heading_' + docKey + '">' + t('Dokument {n}: {name}', { n: pos, name: name }) + ' <span class="page-count">' + meta + '</span></h2></summary>'
            +   inner
            + '</details>'
            + '<span class="doc-actions">'
            +   '<button type="button" class="doc-action-btn" data-kind="doc" data-doc-id="' + docKey + '" data-doc-name="' + name + '" onclick="openDocRename(event)">' + t('Umbenennen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            +   '<button type="button" class="doc-action-btn doc-action-danger" data-kind="doc" data-doc-id="' + docKey + '" data-doc-name="' + name + '" data-doc-count="' + felder.length + '" onclick="openDocDelete(event)">' + t('Löschen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            + '</span></div>';
    }

    function exportScopeHtml(documents) {
        if (!documents || documents.length <= 1) return '';
        const items = documents.map((d, i) => '<li class="export-scope-item"><label class="export-scope-label">'
            + '<input class="export-scope-radio" type="radio" name="fExportScope" value="doc:' + d.id + '"><span>'
            + t('Dokument {n}: {name}', { n: i + 1, name: escHtml(docDisplayName(d)) }) + '</span></label></li>').join('');
        return '<fieldset class="export-scope-fieldset"><legend class="export-scope-legend">' + t('Was exportieren?') + '</legend><ul class="export-scope-list">'
            + '<li class="export-scope-item"><label class="export-scope-label"><input class="export-scope-radio" type="radio" name="fExportScope" value="all" checked><span>' + t('Alle Dokumente (eine ZIP-Datei)') + '</span></label></li>'
            + items + '</ul></fieldset>';
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
        const info = felder.length
            ? t('{n} Felder in {d} Dokumenten, {o} ohne Quickinfo. Stammdaten: {s} Einträge.', { n: felder.length, d: docs.length, o: offen, s: data.stammdaten_anzahl || 0 })
            : t('Noch kein Formular hochgeladen.');
        return '<div class="card">'
            + '<div class="card-header"><h1 id="projectName" class="card-name" tabindex="-1">' + escHtml(title) + '</h1>'
            + '<span class="badge ' + badgeCls + '" id="projectStatusBadge">' + badge + '</span></div>'
            + '<div class="card-info" id="projectHeadInfo">' + info + '</div>'
            + (felder.length ? ''
                + '<div class="card-actions">'
                +   '<button class="btn btn-primary" id="fExportOpenBtn" onclick="Formular.exportOeffnen()">' + t('Exportieren') + '</button>'
                +   '<button class="btn btn-secondary" id="fStammdatenBtn" onclick="Formular.stammdatenAnwenden(' + project.id + ')">' + t('Stammdaten auf offene Felder anwenden') + '</button>'
                +   '<a class="btn btn-secondary" href="/stammdaten">' + t('Meine Stammdaten öffnen') + '</a>'
                +   '<dialog id="fExportPanel" class="invite-dialog" aria-labelledby="fExportHeading">'
                +     '<h2 id="fExportHeading" style="margin:0 0 0.6rem 0;">' + t('Export-Optionen') + '</h2>'
                +     '<p id="fExportSummary" role="status" style="margin:0 0 0.8rem 0;">' + t('{b} von {n} Feldern haben eine Quickinfo. Felder ohne Quickinfo bleiben in der PDF unverändert.', { b: felder.length - offen, n: felder.length }) + '</p>'
                +     exportScopeHtml(docs)
                +     '<div class="form-group" style="margin-bottom:0.8rem;"><label for="fExportFilename" style="display:block;font-weight:600;margin-bottom:0.3rem;">' + t('Dateiname (optional)') + '</label>'
                +       '<input type="text" id="fExportFilename" style="width:100%;padding:0.5rem;border:1px solid var(--border);border-radius:4px;font-size:0.95rem;"></div>'
                +     '<div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">'
                +       '<button class="btn btn-primary" onclick="Formular.exportieren(' + project.id + ', \'formular\')">' + t('Als PDF mit Quickinfos') + '</button>'
                +       '<button class="btn btn-secondary" onclick="Formular.exportieren(' + project.id + ', \'formular_csv\')">' + t('Als CSV (Feldliste)') + '</button>'
                +       '<button class="btn btn-secondary" onclick="Formular.exportSchliessen()">' + t('Abbrechen') + '</button>'
                +     '</div><span id="fExportStatus" role="status" aria-live="polite" style="display:block;margin-top:0.5rem;"></span>'
                +   '</dialog>'
                +   '<div style="flex-basis:100%;margin-top:0.6rem;"><label for="fNurOffene" class="context-toggle" style="display:inline-flex;align-items:center;gap:0.5rem;cursor:pointer;">'
                +     '<span style="font-weight:600;">' + t('Nur offene Felder anzeigen') + '</span>'
                +     '<input type="checkbox" id="fNurOffene"' + (nurOffene ? ' checked' : '') + ' onchange="Formular.filter(this.checked)" style="width:1.2rem;height:1.2rem;"></label></div>'
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
        if (nurOffene && card) card.hidden = !offen && false; // beim Bearbeiten nicht ausblenden (Fokus bleibt)
    }

    async function speichern(feldId, text) {
        try {
            const res = await fetch('/api/felder/' + feldId, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ quickinfo: text }) });
            if (!res.ok) { announce(t('Speichern fehlgeschlagen.')); return; }
            const d = await res.json();
            statusSetzen(feldId, d);
            const ind = document.getElementById('feld_saved_' + feldId);
            if (ind) { ind.classList.add('visible'); setTimeout(() => ind.classList.remove('visible'), 2000); }
        } catch (e) { announce(t('Verbindungsfehler beim Speichern.')); }
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
        const res = await fetch('/api/projects/' + projectId + '/stammdaten-anwenden', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ nur_offene: true }) });
        if (!res.ok) { announce(t('Stammdaten konnten nicht angewendet werden.')); return; }
        const d = await res.json();
        announce(d.uebernommen === 1 ? t('1 Quickinfo aus Stammdaten übernommen.') : t('{n} Quickinfos aus Stammdaten übernommen.', { n: d.uebernommen }));
        await showProject(projectId);
    }

    function filter(nur) {
        nurOffene = nur;
        document.querySelectorAll('.feld-review').forEach(c => { c.hidden = nur && c.dataset.status !== 'offen'; });
        announce(nur ? t('Nur offene Felder werden angezeigt.') : t('Alle Felder werden angezeigt.'));
    }

    function exportOeffnen() {
        const panel = document.getElementById('fExportPanel');
        if (!panel) return;
        const input = document.getElementById('fExportFilename');
        if (input) input.value = '';
        if (typeof panel.showModal === 'function') panel.showModal(); else panel.setAttribute('open', '');
        announce(t('Export-Optionen geöffnet.'));
    }

    function exportSchliessen(silent) {
        const panel = document.getElementById('fExportPanel');
        if (panel && panel.open) { if (typeof panel.close === 'function') panel.close(); else panel.removeAttribute('open'); }
        if (!silent) announce(t('Export abgebrochen.'));
    }

    async function exportieren(projectId, format) {
        const statusEl = document.getElementById('fExportStatus');
        const chosen = document.querySelector('input[name="fExportScope"]:checked');
        const body = {};
        if (chosen && /^doc:(\d+)$/.test(chosen.value)) body.document_id = Number(chosen.value.slice(4));
        const name = (document.getElementById('fExportFilename') || {}).value || '';
        if (name.trim()) body.filename = name.trim();
        if (statusEl) statusEl.textContent = t('Wird exportiert...');
        announce(t('Export läuft …'));
        try {
            const res = await fetch('/api/projects/' + projectId + '/export/' + format, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
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

    async function showProject(projectId) {
        const main = document.getElementById('main');
        const res = await fetch('/api/projects/' + projectId + '/felder');
        if (res.status === 401) { window.location.href = '/'; return; }
        if (!res.ok) { main.innerHTML = '<div class="card"><p>' + t('Projekt konnte nicht geladen werden.') + '</p></div>'; return; }
        const data = await res.json();
        const project = data.project;
        if (zustandProjekt !== projectId) { offeneDocs = new Set(); offeneSeiten = new Set(); zustandProjekt = projectId; }
        // currentDocProjectId (Umbenennen/Loeschen-Dialoge) setzt app.html.showProject vor der Weiche.
        const felderJeDoc = new Map();
        data.felder.forEach(f => { if (!felderJeDoc.has(f.document_id)) felderJeDoc.set(f.document_id, []); felderJeDoc.get(f.document_id).push(f); });
        const docsHtml = data.documents.map((d, i) => dokumentHtml(d, i + 1, felderJeDoc.get(d.id) || [], data.stammdaten_treffer || {})).join('');
        main.innerHTML = kopfHtml(project, data)
            + uploadBlockHtml(project)
            + '<div id="feldListe">' + (data.felder.length ? docsHtml : '') + '</div>';
        bindAutosave();
        if (nurOffene) filter(true);
        setupProjectDropzone(projectId);
        const h1 = document.getElementById('projectName');
        if (h1) h1.focus();
    }

    window.Formular = { showProject, original, inStammdaten, ausStammdaten, stammdatenAnwenden, filter, exportOeffnen, exportSchliessen, exportieren };
})();
