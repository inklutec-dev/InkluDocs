/* =============================================================================
 * uebersetzen.js — Übersetzen-Werkzeug (Word-Dokumente), Projektansicht
 * =============================================================================
 * 18.09.2026, Steve + Fable 5 (Anlass: Mark Hounschild). Eigene Ansicht fuer
 * Uebersetzungsprojekte (tool "uebersetzen", project_type "docx-uebersetzung"),
 * bewusst getrennt von der Bild-/Alt-Text-Ansicht in app.html — ein Absatz ist kein
 * Bild, und Steve wollte KEINE Doppel-Listen in einem Werkzeug („überladen“):
 * EIN Werkzeug, EINE Aufgabe, EINE Ansicht.
 *
 * Die FORM ist dieselbe wie bei den Alt-Texten und Quickinfos: H1 Projekt, H2 Dokument
 * (klappbar), H3 Abschnitt (klappbar, nach Ueberschrift 1), H4 Absatz; je Absatz das
 * Original als Text und ein Eingabefeld „Übersetzung“ mit Auto-Speichern (800 ms),
 * Live-Ansagen ueber announce(). Filter „Alle / Nur mit Hinweis / Nur offen“ als
 * Radiogruppe in einem fieldset (wie die Bildfilter seit 18.09.2026).
 *
 * Gemeinsame Helfer aus app.html/dashboard.js (globale Funktionen): t(), announce(),
 * escHtml(), uploadBlockHtml(), setupProjectDropzone(), docDisplayName(),
 * openDocRename(), openDocDelete(), downloadBlob(), exportCreditsAbgefangen(),
 * exportDocIconHtml(), icon(). Diese Datei ist an window.I18N angebunden (t()).
 *
 * Datenquelle: GET /api/projects/{id}/uebersetzung (backend/uebersetzung_api.py).
 * Sicherheit: alle Texte aus dem Server laufen durch escHtml(); Eingaben gehen als JSON
 * an PATCH /api/uebersetzung/segmente/{id}; keine innerHTML-Zuweisung mit unescapten
 * Nutzerdaten; kein Gast-Modus (Uebersetzungen werden nicht freigegeben).
 * ========================================================================== */
(function () {
    'use strict';

    let offeneDocs = new Set();
    let offeneAbschnitte = new Set();
    let zustandProjekt = null;
    let projektStatus = '';
    let aktuelleDocs = [];
    let aktuelleSegmente = [];
    let zielsprachen = [];
    let einstellungen = {};
    let filterModus = 'alle';          // alle | hinweis | offen
    let exportZielDoc = null;
    let laufZielDoc = null;
    let exportFertig = false;
    let exportLaeuft = false;

    function ico(name) { return (typeof icon === 'function') ? icon(name) : ''; }
    function sprachName(code) { const z = zielsprachen.find(s => s.code === code); return z ? z.name : (code || ''); }

    const STATUS = {
        offen: () => t('Noch nicht übersetzt'),
        fertig: () => t('Übersetzt'),
        zusammengelegt: () => t('Übersetzt, Formatierung zusammengelegt'),
        hand: () => t('Von Hand korrigiert'),
        fehler: () => t('Keine Übersetzung erhalten'),
    };
    const ART = {
        absatz: () => t('Absatz'), alt: () => t('Alternativtext'), titel: () => t('Bildtitel'), dokumenttitel: () => t('Dokumenttitel'),
    };

    function istFertig(s) { return s.status === 'fertig' || s.status === 'zusammengelegt' || s.status === 'hand'; }

    function anfang(text, n) {
        const s = (text || '').replace(/\s+/g, ' ').trim();
        return s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s;
    }

    function segUeberschrift(s) {
        const art = s.art === 'absatz'
            ? (s.ueberschrift_ebene === 0 ? t('Titel') : (s.ueberschrift_ebene ? t('Überschrift') : t('Absatz')))
            : (ART[s.art] || ART.absatz)();
        return t('{art} {n}: {anfang}', { art: art, n: s.position, anfang: anfang(s.original, 60) });
    }

    function segCardHtml(s) {
        const offen = !istFertig(s);
        const badges = ['<span class="badge ' + (offen ? 'badge-pending' : 'badge-done') + '" id="seg_status_' + s.id + '">' + escHtml((STATUS[s.status] || STATUS.offen)()) + '</span>'];
        if (s.ort && s.ort !== 'Text') badges.push('<span class="badge" style="background:#4b5563;color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;">' + escHtml(ortText(s.ort)) + '</span>');
        const hinweis = s.hinweis
            ? '<p class="seg-hinweis" id="seg_hinweis_' + s.id + '" style="font-size:0.9rem;color:var(--text-muted);margin:0.3rem 0 0;">' + escHtml(s.hinweis) + '</p>'
            : '<p class="seg-hinweis" id="seg_hinweis_' + s.id + '" style="font-size:0.9rem;color:var(--text-muted);margin:0.3rem 0 0;" hidden></p>';
        return ''
            + '<section class="image-review seg-review" id="segcard_' + s.id + '" aria-labelledby="seg_heading_' + s.id + '" data-status="' + (offen ? 'offen' : 'fertig') + '" data-hinweis="' + (s.hinweis ? '1' : '0') + '">'
            + '<div class="image-review-header" style="align-items:flex-start;margin-bottom:0.4rem;">'
            +   '<h4 id="seg_heading_' + s.id + '" class="image-heading" style="margin:0;">' + escHtml(segUeberschrift(s)) + '</h4>'
            +   '<span style="display:flex;gap:0.35rem;flex-wrap:wrap;justify-content:flex-end;">' + badges.join(' ') + '</span>'
            + '</div>'
            + '<p class="seg-original" id="seg_original_' + s.id + '"><span style="font-weight:600;">' + t('Original:') + '</span> ' + escHtml(s.original) + '</p>'
            + (s.uebersetzbar ? ''
                + '<label for="seg_ziel_' + s.id + '" style="display:block;font-weight:600;margin-bottom:0.3rem;">' + t('Übersetzung')
                +   ' <span class="save-indicator" id="seg_saved_' + s.id + '">' + t('Gespeichert') + '</span></label>'
                + '<textarea class="alt-text-field seg-ziel" id="seg_ziel_' + s.id + '" data-seg-id="' + s.id + '" aria-describedby="seg_original_' + s.id + ' seg_hinweis_' + s.id + '" maxlength="20000"'
                +   ' placeholder="' + t('Noch keine Übersetzung – wird beim Lauf gefüllt oder hier von Hand eingeben') + '">' + escHtml(s.uebersetzung || '') + '</textarea>'
                : '<p style="font-size:0.9rem;color:var(--text-muted);margin:0;">' + t('Dieser Absatz enthält keinen übersetzbaren Text (Zahlen oder Zeichen) und bleibt unverändert.') + '</p>')
            + hinweis
            + '</section>';
    }

    function ortText(ort) {
        return ({ 'Kopfzeile': t('Kopfzeile'), 'Fußzeile': t('Fußzeile'), 'Fußnote': t('Fußnote'), 'Endnote': t('Endnote'),
                  'Tabelle': t('Tabelle'), 'Textfeld': t('Textfeld'), 'Dokumenteigenschaften': t('Dokumenteigenschaften') })[ort] || ort;
    }

    // Gruppen je Dokument: Abschnitte des Hauptteils (nach Ueberschrift 1), danach Kopf-/Fusszeilen,
    // Fussnoten, Alternativtexte, Dokumenttitel — jede Gruppe eine H3-Klappe.
    function gruppen(segmente) {
        const haupt = new Map();
        const sonst = { 'Kopfzeile': [], 'Fußzeile': [], 'Fußnote': [], 'Endnote': [], bilder: [], titel: [] };
        segmente.forEach(s => {
            if (s.art === 'dokumenttitel') sonst.titel.push(s);
            else if (s.art === 'alt' || s.art === 'titel') sonst.bilder.push(s);
            else if (s.ort === 'Kopfzeile' || s.ort === 'Fußzeile' || s.ort === 'Fußnote' || s.ort === 'Endnote') sonst[s.ort].push(s);
            else { const k = s.abschnitt || 1; if (!haupt.has(k)) haupt.set(k, { titel: s.abschnitt_titel, segs: [] }); haupt.get(k).segs.push(s); }
        });
        const out = [];
        Array.from(haupt.entries()).sort((a, b) => a[0] - b[0]).forEach(([k, g]) => out.push({
            key: 'a' + k, kopf: g.titel ? t('Abschnitt {n}: {titel}', { n: k, titel: anfang(g.titel, 60) }) : t('Abschnitt {n}', { n: k }), segs: g.segs }));
        [['Kopfzeile', t('Kopfzeile')], ['Fußzeile', t('Fußzeile')], ['Fußnote', t('Fußnoten')], ['Endnote', t('Endnoten')]].forEach(([k, label]) => {
            if (sonst[k].length) out.push({ key: k, kopf: label, segs: sonst[k] });
        });
        if (sonst.bilder.length) out.push({ key: 'bilder', kopf: t('Alternativtexte und Titel der Bilder'), segs: sonst.bilder });
        if (sonst.titel.length) out.push({ key: 'titel', kopf: t('Dokumenttitel'), segs: sonst.titel });
        return out;
    }

    function zaehlText(segs) {
        const ue = segs.filter(s => s.uebersetzbar);
        const fertig = ue.filter(istFertig).length;
        return t('{n} Absätze, {f} übersetzt', { n: ue.length, f: fertig });
    }

    function gruppeHtml(docKey, g) {
        const key = docKey + '_' + g.key;
        return '<details class="page-section" data-page="' + key + '"' + (offeneAbschnitte.has(key) ? ' open' : '') + '>'
            + '<summary class="page-summary"><h3 class="page-heading" id="seg_group_' + key + '">' + escHtml(g.kopf) + ' <span class="page-count">(' + zaehlText(g.segs) + ')</span></h3></summary>'
            + g.segs.map(segCardHtml).join('')
            + '</details>';
    }

    function hinweiseHtml(doc) {
        const h = doc.hinweise || {};
        const items = (h.hinweise || []).map(x => '<li>' + escHtml(x) + '</li>');
        if (!items.length) return '';
        return '<details class="page-text-details doc-hinweise"><summary>' + t('{n} Hinweise zu diesem Dokument', { n: items.length }) + '</summary>'
            + '<div class="page-text-content" role="region" aria-label="' + t('Hinweise') + '" tabindex="0"><ul>' + items.join('') + '</ul></div></details>';
    }

    function dokumentHtml(doc, pos, segs) {
        const docKey = doc.id;
        const name = escHtml(docDisplayName(doc));
        const ue = segs.filter(s => s.uebersetzbar);
        const woerter = ue.reduce((a, s) => a + (s.woerter || 0), 0);
        const meta = '(' + t('{n} Absätze, {w} Wörter, {f} übersetzt', { n: ue.length, w: woerter, f: ue.filter(istFertig).length }) + ')';
        const vh = t('– Dokument „{name}“', { name: name });
        const busy = projektStatus === 'processing' || projektStatus === 'extracting';
        return '<div class="doc-block">'
            + '<details class="doc-section" data-doc="' + docKey + '"' + (offeneDocs.has(docKey) ? ' open' : '') + '>'
            +   '<summary class="doc-summary"><h2 class="doc-heading" id="doc_heading_' + docKey + '">' + t('Dokument {n}: {name}', { n: pos, name: name }) + ' <span class="page-count">' + meta + '</span></h2></summary>'
            +   hinweiseHtml(doc)
            +   gruppen(segs).map(g => gruppeHtml(docKey, g)).join('')
            + '</details>'
            + '<span class="doc-actions">'
            +   (ue.length && !busy ? '<button type="button" class="doc-action-btn" onclick="Uebersetzen.laufOeffnen(' + docKey + ')">' + ico('sparkle') + t('Übersetzen') + '<span class="visually-hidden"> ' + vh + '</span></button>' : '')
            +   (ue.some(istFertig) && !busy ? '<button type="button" class="doc-action-btn" onclick="Uebersetzen.exportOeffnen(' + docKey + ')">' + ico('download') + t('Herunterladen') + '<span class="visually-hidden"> ' + vh + '</span></button>' : '')
            +   '<button type="button" class="doc-action-btn" data-kind="uebdoc" data-doc-id="' + docKey + '" data-doc-name="' + name + '" onclick="openDocRename(event)">' + ico('pencil') + t('Umbenennen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            +   '<button type="button" class="doc-action-btn doc-action-danger" data-kind="uebdoc" data-doc-id="' + docKey + '" data-doc-name="' + name + '" data-doc-count="' + ue.length + '" onclick="openDocDelete(event)">' + ico('trash') + t('Löschen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            + '</span></div>';
    }

    // Fortschritt als eigene Karte unter dem Upload-Feld (wie Alt-Texte und Quickinfos).
    function fortschrittKarteHtml(project, data) {
        const l = data.lauf;
        if (project.status !== 'processing' || !l) return '';
        const gesamt = Number(l.pakete_gesamt) || 0, fertig = Number(l.pakete_fertig) || 0;
        const prozent = gesamt > 0 ? Math.round((fertig + (fertig < gesamt ? 0.5 : 0)) / gesamt * 100) : 0;
        return '<section class="card" id="progressCard" aria-labelledby="progressHeading">'
            + '<h2 id="progressHeading" class="section-title">' + t('Übersetzung läuft') + '</h2>'
            + '<div class="progress-bar" role="progressbar" aria-valuenow="' + prozent + '" aria-valuemin="0" aria-valuemax="100" aria-label="' + t('Fortschritt: {p} Prozent', { p: prozent }) + '">'
            +   '<div class="progress-fill" style="width:' + prozent + '%"></div></div>'
            + '<p id="processingInfo" aria-live="polite">' + t('{f} von {n} Absätzen übersetzt.', { f: Number(l.segmente_fertig) || 0, n: Number(l.segmente_gesamt) || 0 }) + '</p>'
            + '<button type="button" class="btn btn-secondary" id="uAbortBtn" onclick="Uebersetzen.abbrechen(' + project.id + ')">' + t('Übersetzung abbrechen') + '</button>'
            + '</section>';
    }

    function laufMeldungHtml() {
        return '<div id="uLaufMeldung" class="lauf-meldung" tabindex="-1" hidden><output id="uLaufMeldungText"></output>'
            + '<button type="button" class="btn btn-secondary" onclick="Uebersetzen.meldungSchliessen()">' + t('Schließen') + '</button></div>';
    }

    function zeigeMeldung(text) {
        const box = document.getElementById('uLaufMeldung');
        const out = document.getElementById('uLaufMeldungText');
        if (!box || !out) { announce(text); return; }
        out.textContent = text;
        box.hidden = false;
        if (document.activeElement === document.body) box.focus();
    }
    function meldungSchliessen() {
        const box = document.getElementById('uLaufMeldung');
        const out = document.getElementById('uLaufMeldungText');
        if (out) out.textContent = '';
        if (box) box.hidden = true;
    }

    function filterFieldsetHtml() {
        const chips = [['alle', t('Alle')], ['hinweis', t('Nur mit Hinweis')], ['offen', t('Nur noch nicht übersetzt')]];
        return '<fieldset class="filter-fieldset" id="segFilterFieldset" style="flex-basis:100%;margin-top:0.6rem;border:1px solid var(--border);border-radius:6px;padding:0.5rem 0.8rem;">'
            + '<legend style="font-weight:600;padding:0 0.3rem;">' + t('Absätze filtern') + '</legend>'
            + '<div style="display:flex;gap:1rem;flex-wrap:wrap;">'
            + chips.map(([k, label]) => '<label style="display:inline-flex;align-items:center;gap:0.4rem;cursor:pointer;">'
                + '<input type="radio" name="segFilter" value="' + k + '"' + (filterModus === k ? ' checked' : '') + ' onchange="Uebersetzen.filter(this.value)">' + label + '</label>').join('')
            + '</div><p id="segFilterStatus" role="status" aria-live="polite" style="margin:0.3rem 0 0;font-size:0.9rem;"></p></fieldset>';
    }

    function kopfHtml(project, data) {
        const segs = data.segmente, docs = data.documents;
        const ue = segs.filter(s => s.uebersetzbar);
        const fertig = ue.filter(istFertig).length;
        const hinweise = segs.filter(s => s.hinweis).length;
        const woerter = ue.reduce((a, s) => a + (s.woerter || 0), 0);
        const title = (project.name && project.name.trim()) ? project.name : project.filename;
        let badge, badgeCls;
        if (project.status === 'extracting') { badge = t('Wird gelesen'); badgeCls = 'badge-processing'; }
        else if (project.status === 'processing') { badge = t('Wird übersetzt'); badgeCls = 'badge-processing'; }
        else if (project.status === 'error') { badge = t('Fehler'); badgeCls = 'badge-error'; }
        else if (!ue.length) { badge = t('Neu'); badgeCls = 'badge-pending'; }
        else if (fertig >= ue.length) { badge = t('Vollständig'); badgeCls = 'badge-done'; }
        else if (fertig) { badge = t('In Arbeit'); badgeCls = 'badge-pending'; }
        else { badge = t('Bereit'); badgeCls = 'badge-pending'; }
        const ziel = einstellungen.zielsprache ? sprachName(einstellungen.zielsprache) : '';
        let info = ue.length
            ? t('{n} Absätze mit {w} Wörtern in {d} Dokumenten.', { n: ue.length, w: woerter, d: docs.length })
            : t('Noch kein Dokument hochgeladen.');
        if (ue.length && ziel) info += ' ' + t('Übersetzung {sprache}: {f} von {n} Absätzen fertig.', { sprache: ziel, f: fertig, n: ue.length });
        if (hinweise) info += ' ' + (hinweise === 1 ? t('1 Absatz mit Hinweis.') : t('{h} Absätze mit Hinweis.', { h: hinweise }));
        const busy = project.status === 'processing' || project.status === 'extracting';
        return '<div class="card">'
            + '<div class="card-header"><h1 id="projectName" class="card-name" tabindex="-1">' + t('Projekt: {name}', { name: escHtml(title) }) + '</h1>'
            + '<span class="badge ' + badgeCls + '" id="projectStatusBadge">' + badge + '</span></div>'
            + '<div class="card-info" id="projectHeadInfo">' + info + '</div>'
            + (ue.length ? ''
                + '<div class="card-actions">'
                +   (!busy ? '<button class="btn btn-primary" id="uStartBtn" onclick="Uebersetzen.laufOeffnen()">' + ico('sparkle') + t('Übersetzen') + '<span class="visually-hidden"> ' + t('– ganzes Projekt') + '</span></button>' : '')
                +   (fertig && !busy ? '<button class="btn btn-primary" id="uExportOpenBtn" onclick="Uebersetzen.exportOeffnen()">' + ico('download') + (docs.length > 1 ? t('Ganzes Projekt herunterladen') : t('Herunterladen')) + '</button>' : '')
                +   laufDialogHtml(project)
                +   exportDialogHtml(project)
                +   filterFieldsetHtml()
                + '</div>' : '')
            + '</div>';
    }

    // ─── Dialog „Übersetzen“: Zielsprache, zwei Schalter, Rueckfrage mit Umfang und Preis ───
    function laufDialogHtml(project) {
        const vorgabe = einstellungen.zielsprache || 'en';
        return '<dialog id="uLaufDialog" class="invite-dialog" aria-labelledby="uLaufHeading">'
            + '<div class="export-kopf"><h2 id="uLaufHeading" style="margin:0 0 0.6rem 0;">' + t('Übersetzen') + '</h2>'
            +   (typeof exportDocIconHtml === 'function' ? exportDocIconHtml('word') : '') + '</div>'
            + '<p id="uLaufUmfang" style="margin:0 0 0.6rem 0;font-weight:600;"></p>'
            + '<div class="form-group" style="margin-bottom:0.8rem;"><label for="uZielsprache" style="display:block;font-weight:600;margin-bottom:calc(0.3rem + 3pt);">' + t('Zielsprache') + '</label>'
            +   '<select id="uZielsprache" style="padding:0.4rem;border:1px solid var(--border,#ccc);border-radius:4px;font-size:0.95rem;min-width:14rem;">'
            +     zielsprachen.map(z => '<option value="' + escHtml(z.code) + '"' + (z.code === vorgabe ? ' selected' : '') + '>' + escHtml(z.name) + '</option>').join('')
            +   '</select></div>'
            + '<div style="display:flex;flex-direction:column;gap:0.5rem;margin-bottom:0.8rem;">'
            +   '<label style="display:inline-flex;align-items:center;gap:0.5rem;cursor:pointer;"><input type="checkbox" id="uAltTexte"' + (einstellungen.alt_texte === false ? '' : ' checked') + ' style="width:1.2rem;height:1.2rem;"><span>' + t('Alternativtexte der Bilder mitübersetzen') + '</span></label>'
            +   '<label style="display:inline-flex;align-items:center;gap:0.5rem;cursor:pointer;"><input type="checkbox" id="uSpracheSetzen"' + (einstellungen.sprache_setzen === false ? '' : ' checked') + ' style="width:1.2rem;height:1.2rem;"><span>' + t('Dokumentsprache in der Datei auf die Zielsprache setzen (damit Screenreader richtig vorlesen)') + '</span></label>'
            + '</div>'
            + '<div id="uLaufSummary" role="status" style="margin:0 0 0.8rem 0;padding:0.6rem 0.8rem;border-radius:6px;background:var(--bg-muted,#f3f4f6);border:1px solid var(--border);font-size:0.95rem;"></div>'
            + '<p style="margin:0 0 0.8rem 0;font-size:0.9rem;color:var(--text-muted);">' + t('Die Formatierung bleibt vollständig erhalten: Nur der Text wird ausgetauscht. Vorhandene Übersetzungen werden ersetzt, von Hand korrigierte Absätze bleiben.') + '</p>'
            + '<div id="uLaufFooter" style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;justify-content:flex-end;">'
            +   '<button class="btn btn-primary" id="uLaufOk" onclick="Uebersetzen.laufStarten(' + project.id + ')">' + t('Übersetzung starten') + '</button>'
            +   '<button class="btn btn-secondary" id="uLaufCancel" onclick="Uebersetzen.laufSchliessen()">' + t('Abbrechen') + '</button>'
            + '</div><output id="uLaufStatus" style="display:block;margin-top:0.5rem;"></output>'
            + '</dialog>';
    }

    async function laufOeffnen(docId) {
        const dlg = document.getElementById('uLaufDialog');
        if (!dlg) return;
        laufZielDoc = docId || null;
        const umfang = document.getElementById('uLaufUmfang');
        if (umfang) umfang.textContent = docId ? t('Nur dieses Dokument') : (aktuelleDocs.length > 1 ? t('Ganzes Projekt, {n} Dokumente', { n: aktuelleDocs.length }) : t('Ganzes Projekt'));
        const sum = document.getElementById('uLaufSummary');
        const ok = document.getElementById('uLaufOk');
        if (sum) sum.textContent = t('Umfang wird ermittelt …');
        if (ok) ok.disabled = true;
        const st = document.getElementById('uLaufStatus'); if (st) st.textContent = '';
        if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
        announce(t('Übersetzen: Einstellungen geöffnet.'));
        const body = {}; if (docId) body.document_id = docId;
        try {
            const res = await fetch('/api/projects/' + zustandProjekt + '/uebersetzung/vorschau', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const v = await res.json().catch(() => ({}));
            if (!res.ok) { if (sum) sum.textContent = v.detail || t('Vorschau fehlgeschlagen.'); return; }
            if (!v.anzahl) { if (sum) sum.textContent = t('Hier gibt es nichts zu übersetzen.'); return; }
            let satz = v.anzahl === 1
                ? t('Es wird 1 Absatz mit {w} Wörtern übersetzt. Das benötigt {c} Credits (1 Credit je angefangene {j} Wörter).', { w: v.woerter, c: v.preis, j: v.woerter_je_credit })
                : t('Es werden {n} Absätze mit {w} Wörtern übersetzt. Das benötigt {c} Credits (1 Credit je angefangene {j} Wörter).', { n: v.anzahl, w: v.woerter, c: v.preis, j: v.woerter_je_credit });
            if (v.verfuegbar === null || v.verfuegbar === undefined) satz += ' ' + t('Dein Konto hat unbegrenztes Guthaben.');
            else if (v.verfuegbar >= v.preis) satz += ' ' + t('Dein Konto verfügt derzeit über ein Guthaben von {v} Credits.', { v: v.verfuegbar });
            else if (v.machbar >= 1) satz += ' ' + t('Du hast {v} Credits — es reicht für etwa {m} von {n} Absätzen, danach hört der Lauf auf.', { v: v.verfuegbar, m: v.machbar, n: v.anzahl });
            else satz += ' ' + t('Dein Guthaben reicht nicht: Der Lauf braucht mindestens 1 Credit, du hast {v}.', { v: v.verfuegbar });
            satz += ' ' + t('Der Lauf kann bei Bedarf auch nach dem Start abgebrochen werden.');
            if (sum) sum.textContent = satz;
            if (ok) ok.disabled = !v.erlaubt;
            const sel = document.getElementById('uZielsprache'); if (sel) sel.focus();
        } catch (e) { if (sum) sum.textContent = t('Verbindungsfehler.'); }
    }

    function laufSchliessen() {
        const dlg = document.getElementById('uLaufDialog');
        if (dlg && dlg.open) { if (typeof dlg.close === 'function') dlg.close(); else dlg.removeAttribute('open'); }
    }

    async function laufStarten(projectId) {
        const ok = document.getElementById('uLaufOk');
        const st = document.getElementById('uLaufStatus');
        const body = {
            zielsprache: (document.getElementById('uZielsprache') || {}).value || 'en',
            alt_texte: !!(document.getElementById('uAltTexte') || {}).checked,
            sprache_setzen: !!(document.getElementById('uSpracheSetzen') || {}).checked,
        };
        if (laufZielDoc) body.document_id = laufZielDoc;
        if (ok) ok.disabled = true;
        if (st) st.textContent = t('Wird gestartet …');
        try {
            const res = await fetch('/api/projects/' + projectId + '/uebersetzung/starten', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            if (typeof exportCreditsAbgefangen === 'function' && await exportCreditsAbgefangen(res)) { if (ok) ok.disabled = false; if (st) st.textContent = ''; return; }
            const d = await res.json().catch(() => ({}));
            if (!res.ok) { const m = d.detail || t('Start fehlgeschlagen.'); if (st) st.textContent = m; announce(m); if (ok) ok.disabled = false; return; }
            laufSchliessen();
            if (!d.gestartet) { announce(t('Hier gibt es nichts zu übersetzen.')); return; }
            announce(t('Übersetzung gestartet: {n} Absätze nach {sprache}.', { n: d.anzahl, sprache: sprachName(body.zielsprache) }));
            await showProject(projectId);
        } catch (e) { if (st) st.textContent = t('Verbindungsfehler.'); if (ok) ok.disabled = false; }
    }

    async function abbrechen(projectId) {
        const btn = document.getElementById('uAbortBtn');
        if (btn) { btn.disabled = true; btn.textContent = t('Abbruch angefordert …'); }
        try {
            const res = await fetch('/api/projects/' + projectId + '/uebersetzung/abbrechen', { method: 'POST' });
            const d = await res.json().catch(() => ({}));
            if (!res.ok) { announce(d.detail || t('Abbruch fehlgeschlagen.')); if (btn) { btn.disabled = false; btn.textContent = t('Übersetzung abbrechen'); } return; }
            announce(d.angefordert ? t('Abbruch angefordert – der Lauf endet nach dem laufenden Paket.') : t('Es läuft gerade keine Übersetzung.'));
        } catch (e) { announce(t('Verbindungsfehler.')); if (btn) { btn.disabled = false; btn.textContent = t('Übersetzung abbrechen'); } }
    }

    // ─── Export-Dialog: eine Datei je Dokument (ZIP bei mehreren), kostenlos ───
    function exportDialogHtml(project) {
        return '<dialog id="uExportPanel" class="invite-dialog" aria-labelledby="uExportHeading">'
            + '<div class="export-kopf"><h2 id="uExportHeading" style="margin:0 0 0.6rem 0;">' + t('Übersetzung herunterladen') + '</h2>'
            +   (typeof exportDocIconHtml === 'function' ? exportDocIconHtml('word') : '') + '</div>'
            + '<div id="uExportSummary" role="status" style="margin:0 0 0.8rem 0;padding:0.6rem 0.8rem;border-radius:6px;background:var(--bg-muted,#f3f4f6);border:1px solid var(--border);font-size:0.95rem;"></div>'
            + '<div class="form-group" style="margin-bottom:0.8rem;"><label for="uExportFilename" style="display:block;font-weight:600;margin-bottom:calc(0.3rem + 3pt);">' + t('Dateiname (optional)') + '</label>'
            +   '<input type="text" id="uExportFilename" autocomplete="off" aria-describedby="uExportFilenameHint" style="width:100%;padding:0.5rem;border:1px solid var(--border);border-radius:4px;font-size:0.95rem;">'
            +   '<p id="uExportFilenameHint" style="margin:0.3rem 0 0 0;color:var(--text-muted);font-size:0.85rem;">' + t('Leer lassen, um den Vorgabe-Namen zu übernehmen. Die Dateiendung wird automatisch angehängt.') + '</p></div>'
            + '<div id="uExportFooter" style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;justify-content:flex-end;">'
            +   '<button class="btn btn-primary" id="uExportBtn" onclick="Uebersetzen.exportieren(' + project.id + ')">' + t('Als Word herunterladen') + '</button>'
            +   '<button class="btn btn-secondary" id="uExportCancelBtn" onclick="Uebersetzen.exportSchliessen()">' + t('Abbrechen') + '</button>'
            + '</div><output id="uExportStatus" style="display:block;margin-top:0.5rem;" tabindex="-1"></output>'
            + '</dialog>';
    }

    function exportOeffnen(docId) {
        const panel = document.getElementById('uExportPanel');
        if (!panel) return;
        exportZielDoc = docId || null;
        exportFertig = false;
        const input = document.getElementById('uExportFilename'); if (input) input.value = '';
        const s0 = document.getElementById('uExportStatus'); if (s0) s0.textContent = '';
        const c0 = document.getElementById('uExportCancelBtn'); if (c0) c0.textContent = t('Abbrechen');
        const segs = aktuelleSegmente.filter(s => s.uebersetzbar && (!docId || s.document_id === docId));
        const fertig = segs.filter(istFertig).length;
        const ziel = einstellungen.zielsprache ? sprachName(einstellungen.zielsprache) : '';
        const sum = document.getElementById('uExportSummary');
        if (sum) {
            let text = (!docId && aktuelleDocs.length > 1)
                ? t('Ganzes Projekt, {n} Dokumente als ZIP.', { n: aktuelleDocs.length })
                : t('Nur dieses Dokument.');
            text += ' ' + t('{f} von {n} Absätzen sind übersetzt{sprache}. Nicht übersetzte Absätze bleiben in der Ausgangssprache. Struktur und Formatierung der Datei bleiben unverändert. Das Herunterladen kostet keine Credits.', { f: fertig, n: segs.length, sprache: ziel ? ' (' + ziel + ')' : '' });
            sum.textContent = text;
        }
        if (typeof panel.showModal === 'function') panel.showModal(); else panel.setAttribute('open', '');
        announce(t('Export-Optionen geöffnet.'));
    }

    function exportSchliessen(silent) {
        const panel = document.getElementById('uExportPanel');
        if (panel && panel.open) { if (typeof panel.close === 'function') panel.close(); else panel.removeAttribute('open'); }
        if (!silent && !exportFertig) announce(t('Export abgebrochen.'));
        exportFertig = false;
    }

    async function exportieren(projectId) {
        if (exportLaeuft) { announce(t('Der Export läuft bereits.')); return; }
        exportLaeuft = true;
        const knoepfe = Array.from(document.querySelectorAll('#uExportPanel button'));
        const aktiv = document.getElementById('uExportBtn');
        const aktivInhalt = aktiv ? aktiv.innerHTML : '';
        knoepfe.forEach(b => { b.disabled = true; });
        if (aktiv) aktiv.textContent = t('Wird exportiert...');
        const statusEl = document.getElementById('uExportStatus');
        try {
            const body = {};
            if (exportZielDoc) body.document_id = exportZielDoc;
            const name = (document.getElementById('uExportFilename') || {}).value || '';
            if (name.trim()) body.filename = name.trim();
            if (statusEl) statusEl.textContent = t('Wird exportiert...');
            announce(t('Export läuft …'));
            const res = await fetch('/api/projects/' + projectId + '/export/uebersetzung', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            if (!res.ok) {
                const e = await res.json().catch(() => ({}));
                const m = e.detail || t('Fehler beim Export.');
                if (statusEl) statusEl.textContent = m; announce(m); return;
            }
            const blob = await res.blob();
            const cd = res.headers.get('Content-Disposition') || '';
            const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
            let serverName = null;
            if (m) { try { serverName = decodeURIComponent(m[1]); } catch (e) { serverName = m[1]; } }
            const fallback = (body.filename || 'uebersetzung') + '.docx';
            downloadBlob(blob, serverName || fallback);
            let ansage = t('Heruntergeladen: „{name}“.', { name: serverName || fallback }) + ' ' + t('Du findest die Datei bei deinen Downloads.');
            const warn = res.headers.get('X-Export-Warnings');
            if (warn) { try { const w = JSON.parse(warn); if (w.length) ansage += ' ' + t('{n} Hinweise: {w}', { n: w.length, w: w.join(' ') }); } catch (e) { /* nur Anzeige */ } }
            if (statusEl) statusEl.textContent = ansage;
            announce(ansage);
            exportFertig = true;
            const cancel = document.getElementById('uExportCancelBtn');
            if (cancel) cancel.textContent = t('Zurück zum Projekt');
        } catch (e) {
            if (statusEl) statusEl.textContent = t('Verbindungsfehler.');
        } finally {
            exportLaeuft = false;
            knoepfe.forEach(b => { b.disabled = false; });
            if (aktiv) aktiv.innerHTML = aktivInhalt;
            if (exportFertig && statusEl) statusEl.focus();
        }
    }

    // ─── Speichern (Handkorrektur), Filter ───
    async function speichern(segId, text) {
        try {
            const res = await fetch('/api/uebersetzung/segmente/' + segId, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ uebersetzung: text }) });
            if (!res.ok) { announce(t('Speichern fehlgeschlagen.')); return; }
            const d = await res.json();
            statusSetzen(segId, d);
            const ind = document.getElementById('seg_saved_' + segId);
            if (ind) { ind.classList.add('visible'); setTimeout(() => ind.classList.remove('visible'), 2000); }
        } catch (e) { announce(t('Verbindungsfehler beim Speichern.')); }
    }

    function statusSetzen(segId, d) {
        const card = document.getElementById('segcard_' + segId);
        const badge = document.getElementById('seg_status_' + segId);
        const hinweis = document.getElementById('seg_hinweis_' + segId);
        const offen = !(d.status === 'fertig' || d.status === 'zusammengelegt' || d.status === 'hand');
        if (card) { card.dataset.status = offen ? 'offen' : 'fertig'; card.dataset.hinweis = d.hinweis ? '1' : '0'; }
        if (badge) { badge.textContent = (STATUS[d.status] || STATUS.offen)(); badge.className = 'badge ' + (offen ? 'badge-pending' : 'badge-done'); }
        if (hinweis) { hinweis.textContent = d.hinweis || ''; hinweis.hidden = !d.hinweis; }
        const s = aktuelleSegmente.find(x => x.id === segId);
        if (s) { s.status = d.status; s.hinweis = d.hinweis || ''; s.uebersetzung = d.uebersetzung || ''; }
        zaehlerAktualisieren();
    }

    function zaehlerAktualisieren() {
        document.querySelectorAll('details.page-section').forEach(sec => {
            const karten = Array.from(sec.querySelectorAll('section.seg-review')).filter(c => c.querySelector('.seg-ziel'));
            const el = sec.querySelector('.page-count');
            if (el) el.textContent = '(' + t('{n} Absätze, {f} übersetzt', { n: karten.length, f: karten.filter(c => c.dataset.status === 'fertig').length }) + ')';
        });
    }

    function filter(modus, still) {
        filterModus = modus || 'alle';
        let sichtbar = 0;
        document.querySelectorAll('.seg-review').forEach(c => {
            const hat = c.querySelector('.seg-ziel');
            c.hidden = (filterModus === 'hinweis' && c.dataset.hinweis !== '1') || (filterModus === 'offen' && !(hat && c.dataset.status === 'offen'));
            if (!c.hidden) sichtbar++;
        });
        const nur = filterModus !== 'alle';
        document.querySelectorAll('details.page-section').forEach(s => { s.hidden = nur && s.querySelectorAll('.seg-review:not([hidden])').length === 0; });
        document.querySelectorAll('.doc-block').forEach(b => { b.hidden = nur && b.querySelectorAll('.seg-review:not([hidden])').length === 0; });
        const st = document.getElementById('segFilterStatus');
        const text = filterModus === 'alle' ? t('Alle Absätze werden angezeigt.')
            : (filterModus === 'hinweis' ? t('{n} Absätze mit Hinweis werden angezeigt.', { n: sichtbar }) : t('{n} noch nicht übersetzte Absätze werden angezeigt.', { n: sichtbar }));
        if (st) st.textContent = text;
        if (!still) announce(text);
    }

    function bindAutosave() {
        document.querySelectorAll('.seg-ziel').forEach(ta => {
            let timer;
            ta.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => speichern(Number(ta.dataset.segId), ta.value), 800); });
        });
        document.querySelectorAll('details.page-section').forEach(d => d.addEventListener('toggle', () => {
            if (d.open) offeneAbschnitte.add(d.dataset.page); else offeneAbschnitte.delete(d.dataset.page);
        }));
        document.querySelectorAll('details.doc-section').forEach(d => d.addEventListener('toggle', () => {
            const k = Number(d.dataset.doc);
            if (d.open) offeneDocs.add(k); else offeneDocs.delete(k);
        }));
    }

    async function showProject(projectId, erneut) {
        const main = document.getElementById('main');
        const res = await fetch('/api/projects/' + projectId + '/uebersetzung', { credentials: 'same-origin' });
        if (res.status === 401) { window.location.href = '/login'; return; }
        if (!res.ok) { main.innerHTML = '<div class="card"><p>' + t('Projekt konnte nicht geladen werden.') + '</p></div>'; return; }
        const data = await res.json();
        const project = data.project;
        projektStatus = project.status || '';
        aktuelleDocs = data.documents || [];
        aktuelleSegmente = data.segmente || [];
        zielsprachen = data.zielsprachen || [];
        einstellungen = project.einstellungen || {};
        if (zustandProjekt !== projectId) { offeneDocs = new Set(); offeneAbschnitte = new Set(); zustandProjekt = projectId; }
        const segsJeDoc = new Map();
        aktuelleSegmente.forEach(s => { if (!segsJeDoc.has(s.document_id)) segsJeDoc.set(s.document_id, []); segsJeDoc.get(s.document_id).push(s); });
        const docsHtml = aktuelleDocs.map((d, i) => dokumentHtml(d, i + 1, segsJeDoc.get(d.id) || [])).join('');
        main.innerHTML = kopfHtml(project, data)
            + uploadBlockHtml(project)
            + fortschrittKarteHtml(project, data)
            + laufMeldungHtml()
            + '<div id="segListe">' + docsHtml + '</div>';
        bindAutosave();
        if (filterModus !== 'alle') filter(filterModus, true);
        setupProjectDropzone(projectId);
        const h1 = document.getElementById('projectName');
        if (h1 && !erneut) h1.focus();
        if (project.status === 'extracting' || project.status === 'processing') {
            setTimeout(async () => {
                try {
                    const r = await fetch('/api/projects/' + projectId + '/uebersetzung');
                    if (!r.ok) return;
                    const d = await r.json();
                    if (d.project && d.project.status !== project.status) {
                        let meldung = '';
                        if (project.status === 'extracting') announce(t('Dokument gelesen.'));
                        else {
                            const l = d.lauf || {};
                            const f = (l.fehler || []).length ? ' ' + t('Hinweise: {w}', { w: l.fehler.join(' ') }) : '';
                            meldung = (l.abbruch
                                ? t('Die Übersetzung wurde abgebrochen: {n} Absätze wurden übersetzt.', { n: l.segmente_fertig || 0 })
                                : t('Übersetzung abgeschlossen: {n} Absätze übersetzt, {c} Credits verbraucht.', { n: l.segmente_fertig || 0, c: l.credits || 0 })) + f;
                        }
                        await showProject(projectId);
                        if (meldung) zeigeMeldung(meldung);
                    } else {
                        showProject(projectId, true);
                    }
                } catch (e) { /* naechster Versuch beim naechsten Aufruf */ }
            }, 2500);
        }
    }

    window.Uebersetzen = { showProject, laufOeffnen, laufSchliessen, laufStarten, abbrechen, meldungSchliessen,
                           exportOeffnen, exportSchliessen, exportieren, filter };
})();
