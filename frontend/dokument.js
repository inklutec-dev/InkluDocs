/* =============================================================================
 * dokument.js — Ansicht „Dokument“ eines PDF-Projekts (PDF-Tagging, 22.09.2026)
 * =============================================================================
 * Steve + Michael Karbe + Joerg Heine, Fable 5.1. Seit dem Testumbau (18.09.2026) ist ein
 * Projekt ein Dateityp mit Ansichten. Ein PDF-Projekt hat die Ansichten „Dokument“ (diese
 * Datei) und „Alt-Texte“ (app.html). Gewechselt wird ueber die Zeile „Ansicht“ im Projekt-
 * kopf (app.html: ansichtWahlHtml/wechsleAnsicht, ?ansicht=dokument). Nur EINE Ansicht ist
 * sichtbar, nie im Gastmodus.
 *
 * „Dokument“ ist die Drehscheibe fuer alles, was die ganze Datei betrifft (Michael 21.09.:
 * „eine Ansicht vergleichbar der Ablage“): je Datei eine Karte mit Vorschau der ersten Seite,
 * Stand (ungetaggt / getaggt / PDF/UA geprueft), Sprache, Seiten, Struktur, und den Knoepfen
 * „Barrierefrei machen“ (PDFix-Tagging, tagging_api.py), „Alt-Texte bearbeiten“ (Ansicht
 * wechseln), „Getaggte PDF herunterladen“, „Umbenennen“, „Loeschen“. Der Bericht des letzten
 * Laufs liegt als Klappe unter der Karte.
 *
 * FORM fuer Screenreader: H1 Projekt, H2 „PDF hinzufuegen“ (Upload), H2 „Dokumente (n)“,
 * je Datei eine H3; darunter eine Beschreibungsliste (dl), native Knoepfe, ein <output> je
 * Datei fuer den Laufstatus (role status), der Bericht als <details>. Rueckfrage vor dem Lauf
 * als natives <dialog> wie #genConfirmDialog (Fokusfang, Escape, Abbrechen links / Start rechts).
 * Ansagen nur bei Zustandswechseln (Start, fertig, Fehler), nicht bei jedem Tick.
 *
 * Gemeinsame Helfer aus app.html/dashboard.js: t(), announce(), escHtml(), uploadBlockHtml(),
 * setupProjectDropzone(), docDisplayName(), openDocRename(), openDocDelete(), icon(),
 * ansichtWahlHtml(), inkluagentSectionHtml(), inkluagentInit(), zeigeCreditsMeldung().
 * Datenquelle: GET /api/projects/{id}/dokument-ansicht (tagging_api.py).
 * Sicherheit: alle Servertexte laufen durch escHtml(); keine innerHTML mit rohen Nutzerdaten.
 * ========================================================================== */
(function () {
    'use strict';

    let zustandProjekt = null;
    let aktuelleDaten = null;
    let pollTimer = null;
    let laufZielDoc = null;
    let laufAktiv = false;
    let offeneBerichte = new Set();

    function ico(name) { return (typeof icon === 'function') ? icon(name) : ''; }
    function esc(s) { return (typeof escHtml === 'function') ? escHtml(s == null ? '' : String(s)) : String(s == null ? '' : s); }

    // ─── Texte ───
    function standText(d) {
        const tg = d.tagging || {};
        if (tg.laeuft) return t('Wird barrierefrei gemacht …');
        if (tg.status === 'fehler') return t('Letzter Lauf fehlgeschlagen');
        const v = tg.bericht && tg.bericht.verapdf;
        if (tg.status === 'fertig' && v && v.bestanden) return t('Getaggt, PDF/UA-Prüfung bestanden');
        if (tg.status === 'fertig' && v && v.bestanden === false) return t('Getaggt, PDF/UA-Prüfung mit Hinweisen');
        if (tg.status === 'fertig') return t('Getaggt');
        if (d.getaggt === true) return t('Getaggt (vom Ersteller)');
        if (d.getaggt === false) return t('Ungetaggt');
        return t('Unbekannt');
    }
    function standKlasse(d) {
        const tg = d.tagging || {};
        if (tg.laeuft) return 'badge-processing';
        if (tg.status === 'fehler') return 'badge-error';
        if (tg.status === 'fertig' || d.getaggt === true) return 'badge-done';
        return 'badge-ready';
    }
    function strukturText(s) {
        if (!s || !s.elemente) return t('keine Struktur');
        const teile = [];
        teile.push(t('{n} Überschriften', { n: s.ueberschriften || 0 }));
        teile.push(t('{n} Listen', { n: s.listen || 0 }));
        teile.push(t('{n} Tabellen', { n: s.tabellen || 0 }));
        teile.push(t('{n} Bilder', { n: s.bilder || 0 }));
        return t('{n} Elemente', { n: s.elemente }) + ' (' + teile.join(', ') + ')';
    }

    // ─── Karte je Dokument ───
    function berichtHtml(d) {
        const tg = d.tagging || {};
        const b = tg.bericht || {};
        if (!tg.status || tg.status === 'laeuft') return '';
        const zeilen = [];
        if (tg.status === 'fehler') {
            zeilen.push('<li>' + t('Fehler: {grund}', { grund: esc(b.fehler || t('unbekannt')) }) + (b.zeit ? ' (' + esc(b.zeit) + ')' : '') + '</li>');
        } else {
            zeilen.push('<li>' + t('Getaggt am {zeit} in {s} Sekunden, {n} Seiten.', { zeit: esc(b.zeit || ''), s: esc(b.dauer_s != null ? b.dauer_s : '?'), n: esc(b.seiten || d.seiten || '?') }) + '</li>');
            if (b.sprache) {
                const q = b.sprache.quelle === 'erkannt' ? t('aus dem Text erkannt') : (b.sprache.quelle === 'dokument' ? t('aus dem Dokument übernommen') : t('Projektsprache angenommen'));
                zeilen.push('<li>' + t('Dokumentsprache {lang} ({quelle}).', { lang: esc(b.sprache.lang), quelle: q }) + '</li>');
            }
            if (b.nachher) zeilen.push('<li>' + t('Struktur: {s}', { s: esc(strukturText(b.nachher)) }) + (b.vorher && b.vorher.elemente ? ' ' + t('Vorher: {s}', { s: esc(strukturText(b.vorher)) }) : '') + '</li>');
            if (b.nachher && b.nachher.titel) zeilen.push('<li>' + t('Dokumenttitel: {titel}', { titel: esc(b.nachher.titel) }) + '</li>');
            if (b.bilder) zeilen.push('<li>' + t('{n} Bilder über die Struktur gefunden, {u} Alt-Texte aus dem vorherigen Stand übernommen.', { n: esc(b.bilder.nachher), u: esc(b.bilder.uebernommen) }) + '</li>');
            (b.hinweise || []).forEach(h => zeilen.push('<li>' + esc(h) + '</li>'));
            if (b.testmodus) zeilen.push('<li>' + t('Testmodus: Die Datei trägt „Trial version of PDFix SDK“ als Hersteller, solange das Tagging nicht in der Lizenz freigeschaltet ist.') + '</li>');
        }
        let pruef = '';
        const v = b.verapdf;
        if (v) {
            pruef = '<h4>' + t('PDF/UA-Prüfung') + '</h4><p>' + esc(v.zusammenfassung || (v.bestanden ? t('Bestanden.') : t('Mit Hinweisen.'))) + '</p><ul>'
                + (v.punkte || []).map(p => '<li>' + (p.status === 'befund' ? t('Hinweis') : t('In Ordnung')) + ' – ' + esc(p.bereich) + ': ' + esc(p.text) + '</li>').join('')
                + '</ul>';
        } else if (tg.status === 'fertig') {
            pruef = '<p>' + t('Die PDF/UA-Prüfung war nicht möglich (Prüfdienst nicht erreichbar).') + '</p>';
        }
        return '<details class="page-text-details dok-bericht" data-doc="' + d.id + '"' + (offeneBerichte.has(d.id) ? ' open' : '') + '>'
            + '<summary>' + t('Bericht lesen') + '</summary>'
            + '<div class="page-text-content" role="region" aria-label="' + t('Bericht zum Tagging') + '" tabindex="0"><ul>' + zeilen.join('') + '</ul>' + pruef + '</div></details>';
    }

    function karteHtml(project, d, pos) {
        const name = esc(docDisplayName(d));
        const tg = d.tagging || {};
        const busy = project.status === 'processing' || project.status === 'extracting' || tg.laeuft || !!(project.kette && project.kette.laeuft);
        const vh = t('– Dokument „{name}“', { name: name });
        const preis = tg.preis || 0;
        const seiten = d.seiten || tg.seiten || 0;
        const knopfText = tg.status === 'fertig' ? t('Neu taggen') : t('Barrierefrei machen');
        const bilderZeile = (d.total_images || 0)
            ? t('{n} Bilder, {m} mit Alt-Text', { n: d.total_images, m: tg.hat_alt_texte || 0 })
            : t('keine Bilder gefunden');
        return '<section class="card dok-karte" id="dok_karte_' + d.id + '" aria-labelledby="dok_heading_' + d.id + '">'
            + '<div class="ausgabe-karte">'
            + (seiten ? '<img class="ausgabe-vorschau" src="/api/projects/' + project.id + '/documents/' + d.id + '/vorschau" alt="' + t('Vorschau der ersten Seite von {name}', { name: name }) + '" loading="lazy">' : '')
            + '<div class="ausgabe-text">'
            + '<h3 id="dok_heading_' + d.id + '" class="doc-heading">' + t('Dokument {n}: {name}', { n: pos, name: name }) + ' <span class="badge ' + standKlasse(d) + '" id="dok_badge_' + d.id + '">' + standText(d) + '</span></h3>'
            + '<dl class="dok-meta">'
            +   '<dt>' + t('Stand') + '</dt><dd id="dok_stand_' + d.id + '">' + standText(d) + '</dd>'
            +   '<dt>' + t('Seiten') + '</dt><dd>' + esc(seiten || '?') + '</dd>'
            +   '<dt>' + t('Sprache') + '</dt><dd>' + esc((d.struktur && d.struktur.lang) || t('nicht gesetzt')) + '</dd>'
            +   '<dt>' + t('Struktur') + '</dt><dd>' + esc(strukturText(d.struktur)) + '</dd>'
            +   '<dt>' + t('Bilder') + '</dt><dd>' + bilderZeile + '</dd>'
            +   ((d.felder || 0) > 0 ? '<dt>' + t('Formularfelder') + '</dt><dd>' + t('{n} Felder', { n: d.felder }) + '</dd>' : '')
            + '</dl>'
            + (tg.modus === 'testmodus' && !tg.laeuft ? '<p class="feld-hinweis">' + t('Das Tagging läuft im Testmodus von PDFix, bis die Freischaltung in der Lizenz vorliegt.') + '</p>' : '')
            + '<div class="ausgabe-aktionen">'
            +   (!busy && tg.verfuegbar && seiten ? '<button type="button" class="btn btn-primary" id="dok_tag_' + d.id + '" onclick="Dokument.laufOeffnen(' + d.id + ')">' + ico('sparkle') + knopfText + '<span class="visually-hidden"> ' + vh + ', ' + t('{n} Seiten, {c} Credits', { n: seiten, c: preis }) + '</span></button>' : '')
            +   ((d.total_images || 0) > 0 ? '<button type="button" class="btn btn-secondary" onclick="Dokument.zurAnsicht(' + project.id + ', \'alttexte\')">' + t('Alt-Texte bearbeiten') + '<span class="visually-hidden"> ' + vh + '</span></button>' : '')
            +   ((d.felder || 0) > 0 ? '<button type="button" class="btn btn-secondary" onclick="Dokument.zurAnsicht(' + project.id + ', \'quickinfos\')">' + t('Quickinfos bearbeiten') + '<span class="visually-hidden"> ' + vh + '</span></button>' : '')
            +   (d.getaggt === true && !busy ? '<button type="button" class="btn btn-secondary" id="dok_export_' + d.id + '" onclick="Dokument.exportieren(' + project.id + ', ' + d.id + ')">' + ico('download') + t('Fertige PDF herunterladen') + '<span class="visually-hidden"> ' + vh + ', ' + t('mit Alt-Texten und Quickinfos, kommt in die Ablage') + '</span></button>' : '')
            +   '<button type="button" class="doc-action-btn" data-kind="doc" data-doc-id="' + d.id + '" data-doc-name="' + name + '" onclick="openDocRename(event)">' + ico('pencil') + t('Umbenennen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            +   '<button type="button" class="doc-action-btn doc-action-danger" data-kind="doc" data-doc-id="' + d.id + '" data-doc-name="' + name + '" data-doc-count="' + (d.total_images || 0) + '" onclick="openDocDelete(event)">' + ico('trash') + t('Löschen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            + '</div>'
            + '<output id="dok_status_' + d.id + '" class="dok-status" style="display:block;margin-top:0.5rem;">' + (tg.laeuft ? t('Wird barrierefrei gemacht … Das kann bei großen Dateien einige Minuten dauern.') : '') + '</output>'
            + berichtHtml(d)
            + '</div></div></section>';
    }

    // ─── Kopf ───
    function kopfHtml(project, data) {
        const title = (project.name && project.name.trim()) ? project.name : project.filename;
        const docs = data.documents || [];
        const kette = project.kette || {};
        const laeuft = docs.some(d => d.tagging && d.tagging.laeuft);
        const busy = laeuft || !!kette.laeuft || project.status === 'extracting' || project.status === 'processing';
        let badge, cls;
        if (kette.laeuft) { badge = t('Komplett barrierefrei machen läuft'); cls = 'badge-processing'; }
        else if (laeuft) { badge = t('Wird barrierefrei gemacht …'); cls = 'badge-processing'; }
        else if (project.status === 'extracting') { badge = t('Wird gelesen'); cls = 'badge-processing'; }
        else if (project.status === 'processing') { badge = t('Alt-Texte werden generiert...'); cls = 'badge-processing'; }
        else if (project.status === 'error') { badge = t('Fehler'); cls = 'badge-error'; }
        else if (!docs.length) { badge = t('Neu'); cls = 'badge-ready'; }
        else if (docs.every(d => d.getaggt === true)) { badge = t('Alle Dokumente getaggt'); cls = 'badge-done'; }
        else { badge = t('Bereit'); cls = 'badge-ready'; }
        return '<div class="card">'
            + '<div class="card-header"><h1 id="projectName" class="card-name" tabindex="-1">' + t('Projekt: {name}', { name: esc(title) }) + '</h1>'
            + '<span class="badge ' + cls + '" id="projectStatusBadge">' + badge + '</span></div>'
            + '<div class="card-info" id="projectHeadInfo" hidden></div>'
            + '<div class="card-actions">'
            // Kette (22.09.2026, Steve + Michael): ein Knopf fuer alle Stationen — Tagging, Alt-Texte, Quickinfos.
            +   (docs.length && !busy ? '<button class="btn btn-primary" id="dkKetteBtn" onclick="Dokument.ketteOeffnen(' + project.id + ')">' + ico('sparkle') + t('Komplett barrierefrei machen') + '<span class="visually-hidden"> ' + t('– ganzes Projekt') + '</span></button>' : '')
            +   ((data.ausgaben_anzahl || 0) > 0 ? '<a class="btn btn-secondary" id="ausgabenTab" href="/ablage?projekt=' + project.id + '">' + t('Ablage ({n})', { n: data.ausgaben_anzahl || 0 }) + '</a>' : '')
            +   laufDialogHtml(project)
            +   ketteDialogHtml(project)
            + '</div>'
            + (typeof ansichtWahlHtml === 'function' ? '<div class="card-actions">' + ansichtWahlHtml(project, 'dokument') + '</div>' : '')
            + '</div>';
    }

    // ─── Rueckfrage vor dem Lauf ───
    function laufDialogHtml(project) {
        return '<dialog id="dkLaufDialog" class="app-dialog" aria-labelledby="dkLaufHeading" aria-describedby="dkLaufUmfang dkLaufSummary">'
            + '<h2 id="dkLaufHeading">' + t('Barrierefrei machen') + '</h2>'
            + '<p id="dkLaufUmfang" class="dialog-hint"></p>'
            + '<p id="dkLaufSummary" role="status"></p>'
            + '<p class="dialog-hint" style="margin:0 0 0.8rem 0;">' + t('Die PDF bekommt eine Struktur (Überschriften, Absätze, Listen, Tabellen, Bilder), Titel, Sprache, Lesezeichen und die PDF/UA-Kennung. Der sichtbare Inhalt bleibt unverändert. Danach werden die Bilder neu über die Struktur gefunden; vorhandene Alt-Texte bleiben erhalten.') + '</p>'
            + '<div class="dialog-actions">'
            +   '<button type="button" class="btn btn-secondary" id="dkLaufCancel" onclick="Dokument.laufSchliessen()">' + t('Abbrechen') + '</button>'
            +   '<button type="button" class="btn btn-primary" id="dkLaufOk" onclick="Dokument.laufStarten(' + project.id + ')">' + t('Tagging starten') + '</button>'
            + '</div><output id="dkLaufStatus" style="display:block;margin-top:0.5rem;"></output>'
            + '</dialog>';
    }

    function laufOeffnen(docId) {
        const dlg = document.getElementById('dkLaufDialog');
        const d = (aktuelleDaten && aktuelleDaten.documents || []).find(x => x.id === docId);
        if (!dlg || !d) return;
        laufZielDoc = docId;
        const tg = d.tagging || {};
        const name = docDisplayName(d);
        const umfang = document.getElementById('dkLaufUmfang');
        const summary = document.getElementById('dkLaufSummary');
        const status = document.getElementById('dkLaufStatus');
        const ok = document.getElementById('dkLaufOk');
        if (umfang) umfang.textContent = t('Dokument „{name}“: {n} Seiten.', { name: name, n: tg.seiten || d.seiten || 0 });
        let satz = tg.verfuegbar_credits == null
            ? t('Preis: {c} Credits (1 Credit je Seite).', { c: tg.preis || 0 })
            : t('Preis: {c} Credits (1 Credit je Seite). Verfügbar: {v} Credits.', { c: tg.preis || 0, v: tg.verfuegbar_credits });
        if (tg.status === 'fertig') satz += ' ' + t('Das Dokument wird aus der ursprünglichen Datei neu getaggt; Alt-Texte bleiben erhalten.');
        if (tg.modus === 'testmodus') satz += ' ' + t('Testmodus: Die Datei trägt „Trial version of PDFix SDK“ als Hersteller.');
        if (summary) summary.textContent = satz;
        if (status) status.textContent = '';
        if (ok) { ok.disabled = !tg.erlaubt; ok.textContent = tg.status === 'fertig' ? t('Neu taggen') : t('Tagging starten'); }
        if (!tg.erlaubt && status) status.textContent = t('Dafür reicht das Guthaben nicht: {c} Credits nötig, {v} vorhanden.', { c: tg.preis || 0, v: tg.verfuegbar_credits == null ? 0 : tg.verfuegbar_credits });
        dlg.showModal();
        const cancel = document.getElementById('dkLaufCancel');
        if (cancel) cancel.focus();
    }
    function laufSchliessen() {
        const dlg = document.getElementById('dkLaufDialog');
        if (dlg && dlg.open) dlg.close();
        const btn = laufZielDoc ? document.getElementById('dok_tag_' + laufZielDoc) : null;
        if (btn) btn.focus();
    }
    async function laufStarten(projectId) {
        if (laufAktiv || !laufZielDoc) return;
        const docId = laufZielDoc;
        const status = document.getElementById('dkLaufStatus');
        const ok = document.getElementById('dkLaufOk');
        laufAktiv = true;
        if (ok) ok.disabled = true;
        if (status) status.textContent = t('Wird gestartet …');
        try {
            const res = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/tagging', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            const data = await res.json().catch(() => ({}));
            if (res.status === 402) {
                laufSchliessen();
                if (typeof zeigeCreditsMeldung === 'function') zeigeCreditsMeldung(data.detail); else announce((data.detail && data.detail.text) || t('Dafür reicht das Guthaben nicht.'));
                return;
            }
            if (!res.ok) {
                const m = (data.detail && (data.detail.text || data.detail)) || t('Das Tagging konnte nicht gestartet werden.');
                if (status) status.textContent = typeof m === 'string' ? m : t('Das Tagging konnte nicht gestartet werden.');
                announce(status ? status.textContent : '');
                if (ok) ok.disabled = false;
                return;
            }
            laufSchliessen();
            announce(t('Das Tagging läuft. Du wirst benachrichtigt, sobald es fertig ist.'));
            await showProject(projectId, true);
            const out = document.getElementById('dok_status_' + docId);
            if (out) out.focus && out.setAttribute('tabindex', '-1');
        } catch (e) {
            if (status) status.textContent = t('Verbindungsfehler.');
            if (ok) ok.disabled = false;
        } finally {
            laufAktiv = false;
        }
    }

    // ─── Kette „Komplett barrierefrei machen“ (22.09.2026) ───
    const SCHRITT_NAMEN = { tagging: () => t('Barrierefrei machen (Tagging)'), alttexte: () => t('Alt-Texte'), quickinfos: () => t('Quickinfos') };
    const SCHRITT_STATUS = {
        offen: () => t('wartet'), laeuft: () => t('wird ausgeführt'), fertig: () => t('fertig'), teilweise: () => t('mit Hinweisen'),
        fehler: () => t('fehlgeschlagen'), uebersprungen: () => t('übersprungen (nichts zu tun)'),
    };
    function ketteDialogHtml(project) {
        return '<dialog id="dkKetteDialog" class="app-dialog" aria-labelledby="dkKetteHeading" aria-describedby="dkKettePlan dkKetteSummary">'
            + '<h2 id="dkKetteHeading">' + t('Komplett barrierefrei machen') + '</h2>'
            + '<ol id="dkKettePlan" class="dialog-hint" style="padding-left:1.4rem;"></ol>'
            + '<p id="dkKetteSummary" role="status"></p>'
            + '<p class="dialog-hint" style="margin:0 0 0.8rem 0;">' + t('Die Stationen laufen nacheinander: erst das Tagging, dann Alt-Texte für alle Bilder, dann Quickinfos für alle Felder. Vorhandene Texte werden dabei neu erzeugt, wie bei „Alt-Texte generieren“. Die Zahl der Bilder kann sich nach dem Tagging ändern; jede Station bucht ihre Credits selbst.') + '</p>'
            + '<div class="dialog-actions">'
            +   '<button type="button" class="btn btn-secondary" id="dkKetteCancel" onclick="Dokument.ketteSchliessen()">' + t('Abbrechen') + '</button>'
            +   '<button type="button" class="btn btn-primary" id="dkKetteOk" onclick="Dokument.ketteStarten(' + project.id + ')">' + t('Alles starten') + '</button>'
            + '</div><output id="dkKetteStatus" style="display:block;margin-top:0.5rem;"></output>'
            + '</dialog>';
    }
    async function ketteOeffnen(projectId) {
        const dlg = document.getElementById('dkKetteDialog');
        if (!dlg) return;
        const plan = document.getElementById('dkKettePlan');
        const summary = document.getElementById('dkKetteSummary');
        const status = document.getElementById('dkKetteStatus');
        const ok = document.getElementById('dkKetteOk');
        if (plan) plan.innerHTML = '';
        if (summary) summary.textContent = t('Umfang wird ermittelt …');
        if (status) status.textContent = '';
        if (ok) ok.disabled = true;
        dlg.showModal();
        const cancel = document.getElementById('dkKetteCancel');
        if (cancel) cancel.focus();
        try {
            const res = await fetch('/api/projects/' + projectId + '/kette');
            const v = await res.json();
            if (!res.ok) { if (summary) summary.textContent = (v.detail && (v.detail.text || v.detail)) || t('Umfang konnte nicht ermittelt werden.'); return; }
            const zeilen = [];
            zeilen.push(t('Tagging: {n} Dokumente, {s} Seiten, {c} Credits', { n: v.tagging.dokumente, s: v.tagging.seiten, c: v.tagging.preis })
                + (v.tagging.schon_getaggt ? ' ' + t('({n} Dokumente sind schon getaggt)', { n: v.tagging.schon_getaggt }) : ''));
            zeilen.push(t('Alt-Texte: {n} Bilder, {c} Credits', { n: v.alttexte.bilder, c: v.alttexte.preis }));
            zeilen.push(t('Quickinfos: {n} Felder, {c} Credits', { n: v.quickinfos.felder, c: v.quickinfos.preis }));
            if (plan) plan.innerHTML = zeilen.map(z => '<li>' + esc(z) + '</li>').join('');
            let satz = v.verfuegbar == null ? t('Gesamt: {c} Credits.', { c: v.gesamt }) : t('Gesamt: {c} Credits. Verfügbar: {v} Credits.', { c: v.gesamt, v: v.verfuegbar });
            if (v.nichts_zu_tun) satz = t('Nichts zu tun: alle Dokumente sind getaggt, alle Bilder und Felder beschrieben.');
            else if (!v.erlaubt) satz += ' ' + t('Dafür reicht das Guthaben nicht.');
            if (summary) summary.textContent = satz;
            if (ok) ok.disabled = !v.erlaubt || v.nichts_zu_tun || v.laeuft;
        } catch (e) {
            if (summary) summary.textContent = t('Verbindungsfehler.');
        }
    }
    function ketteSchliessen() {
        const dlg = document.getElementById('dkKetteDialog');
        if (dlg && dlg.open) dlg.close();
        const btn = document.getElementById('dkKetteBtn');
        if (btn) btn.focus();
    }
    async function ketteStarten(projectId) {
        const status = document.getElementById('dkKetteStatus');
        const ok = document.getElementById('dkKetteOk');
        if (ok) ok.disabled = true;
        if (status) status.textContent = t('Wird gestartet …');
        try {
            const res = await fetch('/api/projects/' + projectId + '/kette', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            const data = await res.json().catch(() => ({}));
            if (res.status === 402) {
                ketteSchliessen();
                if (typeof zeigeCreditsMeldung === 'function') zeigeCreditsMeldung(data.detail); else announce((data.detail && data.detail.text) || t('Dafür reicht das Guthaben nicht.'));
                return;
            }
            if (!res.ok || !data.gestartet) {
                const m = (data.detail && (data.detail.text || data.detail)) || t('Die Kette konnte nicht gestartet werden.');
                if (status) status.textContent = typeof m === 'string' ? m : t('Die Kette konnte nicht gestartet werden.');
                announce(status ? status.textContent : '');
                if (ok) ok.disabled = false;
                return;
            }
            ketteSchliessen();
            announce(t('Die Kette läuft. Du wirst benachrichtigt, sobald alles fertig ist.'));
            await showProject(projectId, true);
            const card = document.getElementById('ketteCard');
            if (card) { card.setAttribute('tabindex', '-1'); card.focus(); }
        } catch (e) {
            if (status) status.textContent = t('Verbindungsfehler.');
            if (ok) ok.disabled = false;
        }
    }
    function ketteSchrittText(k) {
        const reihe = ['tagging', 'alttexte', 'quickinfos'];
        const i = Math.max(0, reihe.indexOf(k.schritt));
        const s = (k.schritte || {})[k.schritt] || {};
        let text = t('Schritt {i} von {n}: {name}', { i: i + 1, n: reihe.length, name: SCHRITT_NAMEN[k.schritt] ? SCHRITT_NAMEN[k.schritt]() : k.schritt });
        if (s.geplant) text += ' (' + t('{f} von {n}', { f: s.fertig || 0, n: s.geplant }) + ')';
        return text;
    }
    function ketteKarteHtml(project) {
        const k = project.kette || {};
        if (!k.laeuft) return '';
        const reihe = ['tagging', 'alttexte', 'quickinfos'];
        return '<section class="card" id="ketteCard" aria-labelledby="ketteHeading">'
            + '<h2 id="ketteHeading" class="section-title">' + t('Komplett barrierefrei machen läuft') + '</h2>'
            + '<p id="ketteSchritt" aria-live="polite">' + esc(ketteSchrittText(k)) + '</p>'
            + '<ol id="ketteListe" style="padding-left:1.4rem;">' + reihe.map(r => { const st = (k.schritte || {})[r] || {}; return '<li>' + esc(SCHRITT_NAMEN[r]()) + ': ' + esc((SCHRITT_STATUS[st.status] || SCHRITT_STATUS.offen)()) + '</li>'; }).join('') + '</ol>'
            + '<p class="feld-hinweis">' + t('Das kann einige Minuten dauern. Du kannst die Seite offen lassen; am Ende erscheint eine Meldung.') + '</p>'
            + '</section>';
    }
    function ketteAktualisieren(k) {
        const p = document.getElementById('ketteSchritt');
        if (p) { const neu = ketteSchrittText(k); if (p.textContent !== neu) p.textContent = neu; }
        const ol = document.getElementById('ketteListe');
        if (ol) { const reihe = ['tagging', 'alttexte', 'quickinfos']; ol.innerHTML = reihe.map(r => { const st = (k.schritte || {})[r] || {}; return '<li>' + esc(SCHRITT_NAMEN[r]()) + ': ' + esc((SCHRITT_STATUS[st.status] || SCHRITT_STATUS.offen)()) + '</li>'; }).join(''); }
    }

    // ─── Fertige PDF herunterladen (22.09.2026): derselbe Export wie „Als PDF“ in der Alt-Text-Ansicht
    // (Struktur + Alt-Texte + Quickinfos), die Datei landet zusaetzlich in der Ablage. ───
    async function exportieren(projectId, docId) {
        const btn = document.getElementById('dok_export_' + docId);
        const out = document.getElementById('dok_status_' + docId);
        if (btn) btn.disabled = true;
        if (out) out.textContent = t('Wird exportiert...');
        announce(t('Export läuft …'));
        try {
            const res = await fetch('/api/projects/' + projectId + '/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ document_id: docId }) });
            if (res.status === 402) {
                const e = await res.json().catch(() => ({}));
                if (out) out.textContent = '';
                if (typeof zeigeCreditsMeldung === 'function') zeigeCreditsMeldung(e.detail); else announce((e.detail && e.detail.text) || t('Dafür reicht das Guthaben nicht.'));
                return;
            }
            if (!res.ok) {
                const e = await res.json().catch(() => ({}));
                const m = (e.detail && (e.detail.text || e.detail)) || t('Fehler beim Export.');
                if (out) out.textContent = typeof m === 'string' ? m : t('Fehler beim Export.');
                announce(out ? out.textContent : '');
                return;
            }
            const blob = await res.blob();
            const cd = res.headers.get('Content-Disposition') || '';
            const mStar = /filename\*=UTF-8''([^;]+)/i.exec(cd);
            const m = mStar || /filename="?([^";]+)"?/i.exec(cd);
            let name = null;
            if (m) { try { name = decodeURIComponent(m[1]); } catch (e) { name = m[1]; } }
            name = name || 'inkludocs.pdf';
            downloadBlob(blob, name);
            const credits = res.headers.get('X-Export-Credits');
            let ansage = t('Heruntergeladen: „{name}“.', { name: name }) + ' ' + t('Die Datei liegt auch in deiner Ablage.');
            if (credits) ansage += ' ' + t('{c} Credits verbraucht.', { c: credits });
            const warn = res.headers.get('X-Export-Warnings');
            if (warn) { try { const w = JSON.parse(warn); if (w.length) ansage += ' ' + t('{n} Hinweise: {w}', { n: w.length, w: w.join(' ') }); } catch (e) { /* nur Anzeige */ } }
            announce(ansage);
            await showProject(projectId, true);   // Ablage-Zaehler im Kopf
            const o2 = document.getElementById('dok_status_' + docId);
            if (o2) { o2.textContent = ansage; o2.setAttribute('tabindex', '-1'); o2.focus(); }
        } catch (e) {
            if (out) out.textContent = t('Verbindungsfehler.');
        } finally {
            const b2 = document.getElementById('dok_export_' + docId);
            if (b2) b2.disabled = false;
        }
    }

    // ─── Ansicht wechseln (Knopf „Alt-Texte bearbeiten“) ───
    function zurAnsicht(projectId, ziel) {
        const sel = document.getElementById('ansichtSelect');
        if (sel) sel.value = ziel;
        if (typeof wechsleAnsicht === 'function') wechsleAnsicht(projectId);
    }

    // ─── Laufmeldung (wie Alt-Texte/Uebersetzung) ───
    function laufMeldungHtml() {
        return '<div id="dkLaufMeldung" class="lauf-meldung" tabindex="-1" hidden><output id="dkLaufMeldungText"></output>'
            + '<button type="button" class="btn btn-secondary" onclick="Dokument.meldungSchliessen()">' + t('Schließen') + '</button></div>';
    }
    function zeigeMeldung(text) {
        const box = document.getElementById('dkLaufMeldung');
        const out = document.getElementById('dkLaufMeldungText');
        if (!box || !out) { announce(text); return; }
        out.textContent = text;
        box.hidden = false;
        box.focus();
    }
    function meldungSchliessen() {
        const box = document.getElementById('dkLaufMeldung');
        const out = document.getElementById('dkLaufMeldungText');
        if (out) out.textContent = '';
        if (box) box.hidden = true;
    }

    function pollStoppen() { if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; } }

    function abschlussText(d) {
        const tg = d.tagging || {};
        const b = tg.bericht || {};
        const name = docDisplayName(d);
        if (tg.status === 'fehler') return t('Das Tagging von „{name}“ ist fehlgeschlagen: {grund}', { name: name, grund: b.fehler || t('unbekannter Fehler') });
        const v = b.verapdf;
        let s = t('„{name}“ ist getaggt: {struktur}.', { name: name, struktur: strukturText(b.nachher) });
        if (v) s += ' ' + (v.zusammenfassung || '');
        if (b.bilder && b.bilder.nachher) s += ' ' + t('{n} Bilder gefunden, {u} Alt-Texte übernommen.', { n: b.bilder.nachher, u: b.bilder.uebernommen || 0 });
        return s;
    }

    async function showProject(projectId, erneut) {
        pollStoppen();
        const main = document.getElementById('main');
        const res = await fetch('/api/projects/' + projectId + '/dokument-ansicht', { credentials: 'same-origin' });
        if (res.status === 401) { window.location.href = '/login'; return; }
        if (!res.ok) { main.innerHTML = '<div class="card"><p>' + t('Projekt konnte nicht geladen werden.') + '</p></div>'; return; }
        const data = await res.json();
        const project = data.project;
        aktuelleDaten = data;
        if (zustandProjekt !== projectId) { offeneBerichte = new Set(); zustandProjekt = projectId; }
        const docs = data.documents || [];
        main.innerHTML = kopfHtml(project, data)
            + uploadBlockHtml(project)
            + ketteKarteHtml(project)
            + laufMeldungHtml()
            + '<h2 class="section-title" id="dokumenteHeading" tabindex="-1" style="margin-top:1.5rem">' + t('Dokumente ({n})', { n: docs.length }) + '</h2>'
            + (docs.length ? '' : '<p class="feld-hinweis">' + t('Noch kein Dokument hochgeladen.') + '</p>')
            + '<div id="dokListe">' + docs.map((d, i) => karteHtml(project, d, i + 1)).join('') + '</div>'
            + (typeof inkluagentSectionHtml === 'function' ? inkluagentSectionHtml(projectId) : '');
        document.querySelectorAll('details.dok-bericht').forEach(el => el.addEventListener('toggle', () => {
            const k = Number(el.dataset.doc);
            if (el.open) offeneBerichte.add(k); else offeneBerichte.delete(k);
        }));
        if (typeof inkluagentInit === 'function') inkluagentInit(projectId);
        setupProjectDropzone(projectId);
        const h1 = document.getElementById('projectName');
        if (h1 && !erneut) h1.focus();
        const laufende = docs.filter(d => d.tagging && d.tagging.laeuft).map(d => d.id);
        const ketteLief = !!(project.kette && project.kette.laeuft);
        if (laufende.length || ketteLief || project.status === 'extracting' || project.status === 'processing') {
            const tick = async () => {
                if (zustandProjekt !== projectId) return;
                if (!document.getElementById('dokListe')) { pollStoppen(); return; }   // Ansicht gewechselt
                try {
                    const r = await fetch('/api/projects/' + projectId + '/dokument-ansicht');
                    if (!r.ok) { pollTimer = setTimeout(tick, 2500); return; }
                    const d2 = await r.json();
                    if (zustandProjekt !== projectId) return;
                    // Kette: waehrend des Laufs nur die Statuskarte fortschreiben (kein Neuaufbau, Fokus bleibt);
                    // am Ende einmal neu zeichnen und die Zusammenfassung melden.
                    if (ketteLief) {
                        const k2 = (d2.project && d2.project.kette) || {};
                        if (k2.laeuft) { ketteAktualisieren(k2); pollTimer = setTimeout(tick, 2500); return; }
                        await showProject(projectId, true);
                        zeigeMeldung(t('Komplett barrierefrei machen ist fertig.') + ' ' + (k2.zusammenfassung || ''));
                        return;
                    }
                    const jetzt = (d2.documents || []).filter(x => x.tagging && x.tagging.laeuft).map(x => x.id);
                    const fertigGeworden = (d2.documents || []).filter(x => laufende.includes(x.id) && !(x.tagging && x.tagging.laeuft));
                    const statusWechsel = d2.project.status !== project.status;
                    if (fertigGeworden.length || (statusWechsel && !jetzt.length)) {
                        await showProject(projectId, true);
                        if (fertigGeworden.length) {
                            zeigeMeldung(fertigGeworden.map(abschlussText).join(' '));
                        } else if (project.status === 'extracting' && d2.project.status !== 'extracting') {
                            announce(t('Dokument gelesen.'));
                        }
                        return;
                    }
                    pollTimer = setTimeout(tick, 2500);
                } catch (e) { pollTimer = setTimeout(tick, 2500); }
            };
            pollTimer = setTimeout(tick, 2500);
        }
    }

    window.Dokument = { showProject, laufOeffnen, laufSchliessen, laufStarten, zurAnsicht, meldungSchliessen, pollStoppen,
                        ketteOeffnen, ketteSchliessen, ketteStarten, exportieren };
})();
