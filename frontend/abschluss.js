/*
 * I18N: alle sichtbaren Texte über t() (window.I18N), Kataloge backend/locales, geprüft von scripts/check_i18n.py.
 * Station „Abschlussprüfung“ eines PDF-Projekts (24.09.2026, Steve; Name vorläufig).
 *
 * Letzter Schritt nach Dokument (hochladen, taggen), Alt-Texte und Quickinfos (bearbeiten): die FERTIGE Datei
 * anschauen, anhören, Probleme finden, herunterladen. Geprüft wird immer genau die PDF, die der Kunde beim
 * Herunterladen bekommt (Backend: main.py „STATION ABSCHLUSSPRUEFUNG“, abschluss.py). Die Prüfdatei kostet nichts;
 * Credits kostet wie bisher erst „PDF herunterladen“ (derselbe Export wie bisher, mit Ablage-Eintrag).
 *
 * Aufbau wie die Ansicht „Dokument“: Projektkopf (app.html projektKopfHtml) + Feld „Funktionen und Einstellungen“,
 * je Datei eine aufklappbare Karte (<details>, H3 im summary). In der offenen Karte: Stand der Prüfdatei, Knöpfe,
 * Filter „Ganzes Dokument / Nur Problemstellen“, Problemliste mit Seitenangabe und die Seitenansicht (Seitenbild +
 * Hörprobe dieser Seite, seitenweise für große Dokumente). Vorlesen nur auf Knopfdruck und nur mit Stimmen, die auf
 * dem Gerät laufen (app.html vorlesen) — der Dokumenttext geht an keinen Sprachdienst im Netz.
 *
 * Gemeinsame Helfer aus app.html/dashboard.js: t(), announce(), escHtml(), icon(), docDisplayName(), downloadBlob(),
 * projektKopfHtml(), funktionenKarteHtml(), zeigeCreditsMeldung(), vorlesen(), vorlesenStopp().
 */
(function () {
    'use strict';

    let zustandProjekt = null;
    let offeneDokumente = new Set();
    let geschlosseneDokumente = new Set();
    const details = {};     // docId -> volle Daten (Probleme, Hörprobe)
    const ansicht = {};     // docId -> { filter: 'alle' | 'probleme', seite: n }

    function ico(name) { return (typeof icon === 'function') ? icon(name) : ''; }
    function esc(s) { return (typeof escHtml === 'function') ? escHtml(s == null ? '' : String(s)) : String(s == null ? '' : s); }
    function name(d) { return (typeof docDisplayName === 'function') ? docDisplayName(d) : (d.display_name || d.original_filename || ''); }

    // ─── Stand-Texte ───
    function standText(d) {
        if (!d.getaggt) return t('Noch nicht getaggt');
        const p = d.pruefdatei;
        if (!p) return t('Noch keine Prüfdatei');
        if (!p.aktuell) return t('Prüfdatei nicht mehr aktuell');
        const n = p.anzahl_probleme || 0;
        return n ? t('{n} Problemstellen', { n: n }) : t('Keine Problemstellen gefunden');
    }
    function standKlasse(d) {
        if (!d.getaggt || !d.pruefdatei || !d.pruefdatei.aktuell) return 'badge-ready';
        return (d.pruefdatei.anzahl_probleme || 0) ? 'badge-processing' : 'badge-done';
    }
    function metaZeile(bez, wert) { return '<li>' + bez + ': <span>' + wert + '</span></li>'; }

    // ─── Karte je Dokument ───
    function karteOffen(d, anzahl) {
        if (geschlosseneDokumente.has(d.id)) return false;
        return anzahl <= 1 || offeneDokumente.has(d.id);
    }
    function karteHtml(project, d, pos, anzahl) {
        const nm = esc(name(d));
        const vh = t('– Dokument „{name}“', { name: nm });
        const p = d.pruefdatei;
        let meta = '';
        if (!d.getaggt) {
            meta = '<p class="feld-hinweis">' + t('Dieses Dokument ist noch nicht getaggt. Mache es zuerst in der Ansicht „Dokument“ barrierefrei.') + '</p>';
        } else {
            meta = '<ul class="dok-meta">'
                + metaZeile(t('Seiten'), esc(d.seiten || '?'))
                + metaZeile(t('Prüfdatei'), p ? t('erstellt am {zeit}', { zeit: esc(p.erstellt_am || '') }) : t('noch nicht erstellt'))
                + (p ? metaZeile(t('Stand'), p.aktuell ? t('aktuell') : t('nicht mehr aktuell — seitdem wurden Alt-Texte, Quickinfos oder die Datei geändert')) : '')
                + (p ? metaZeile(t('PDF/UA-Prüfung'), p.verapdf_moeglich ? (p.bestanden ? t('bestanden') : t('mit Hinweisen')) : t('nicht möglich (Prüfdienst nicht erreichbar)')) : '')
                + (p ? metaZeile(t('Problemstellen'), esc(p.anzahl_probleme || 0)) : '')
                + '</ul>';
        }
        const erstellenText = !p ? t('Prüfdatei erstellen') : t('Prüfdatei neu erstellen');
        const erstellenPrimaer = !p || !p.aktuell;
        const aktionen = d.getaggt
            ? '<div class="ausgabe-aktionen">'
              + '<button type="button" class="btn ' + (erstellenPrimaer ? 'btn-primary' : 'btn-secondary') + '" id="ab_erstellen_' + d.id + '" onclick="Abschluss.erstellen(' + project.id + ', ' + d.id + ')"' + (d.laeuft ? ' disabled' : '') + '>' + ico('sparkle') + erstellenText + '<span class="visually-hidden"> ' + vh + ', ' + t('kostenlos') + '</span></button>'
              + (p ? '<button type="button" class="btn ' + (erstellenPrimaer ? 'btn-secondary' : 'btn-primary') + '" id="ab_export_' + d.id + '" onclick="Abschluss.herunterladen(' + project.id + ', ' + d.id + ')">' + ico('download') + t('PDF herunterladen') + '<span class="visually-hidden"> ' + vh + ', ' + t('mit Alt-Texten und Quickinfos, kommt in die Ablage') + '</span></button>' : '')
              + (p ? '<a class="btn btn-secondary" id="ab_struktur_' + d.id + '" href="/struktur/' + project.id + '/' + d.id + '?quelle=abschluss">' + t('Mit eigenem Screenreader prüfen') + '<span class="visually-hidden"> ' + vh + '</span></a>' : '')
              + '</div>'
            : '';
        return '<section class="card dok-karte ab-karte" id="ab_karte_' + d.id + '">'
            + '<details class="dok-klappe ab-klappe" data-doc="' + d.id + '"' + (karteOffen(d, anzahl) ? ' open' : '') + '>'
            + '<summary><h3 id="ab_heading_' + d.id + '" class="doc-heading">' + t('Dokument {n}: {name}', { n: pos, name: nm }) + ' <span class="badge ' + standKlasse(d) + '" id="ab_badge_' + d.id + '">' + esc(standText(d)) + '</span></h3></summary>'
            + '<div class="ab-inhalt">'
            + meta
            + (p ? '<p class="feld-hinweis">' + t('Geprüft wird die fertige Datei, genau die PDF, die du herunterlädst. Das Erstellen der Prüfdatei ist kostenlos; Credits kostet erst das Herunterladen.') + '</p>'
                 : (d.getaggt ? '<p class="feld-hinweis">' + t('Erstelle die Prüfdatei: Sie ist genau die PDF, die du herunterlädst, mit Struktur, Alt-Texten und Quickinfos. Das ist kostenlos.') + '</p>' : ''))
            + aktionen
            + '<output id="ab_status_' + d.id + '" class="dok-status" style="display:block;margin-top:0.5rem;" tabindex="-1">' + (d.laeuft ? t('Prüfdatei wird erstellt …') : '') + '</output>'
            + '<div class="ab-detail" id="ab_detail_' + d.id + '">' + (p && details[d.id] ? detailHtml(project, details[d.id]) : (p ? '<p>' + t('Wird geladen …') + '</p>' : '')) + '</div>'
            + '</div></details></section>';
    }

    // ─── Inhalt einer offenen Karte: Filter, Problemliste, Seitenansicht ───
    function zustand(docId) {
        if (!ansicht[docId]) ansicht[docId] = { filter: 'alle', seite: 0 };
        return ansicht[docId];
    }
    function problemeDerSeite(dd, seite) { return (dd.probleme || []).filter(p => (p.seiten && p.seiten.length ? p.seiten.includes(seite) : p.seite === seite)); }
    function seitenListe(dd, z) {
        const alle = ((dd.hoerprobe && dd.hoerprobe.seiten) || []).map(s => s.seite);
        if (z.filter !== 'probleme') return alle;
        const mit = new Set();
        (dd.probleme || []).forEach(p => (p.seiten && p.seiten.length ? p.seiten : [p.seite]).forEach(s => { if (s) mit.add(s); }));
        return alle.filter(s => mit.has(s));
    }
    function problemText(p) {
        return (p.seiten && p.seiten.length > 1 ? t('Seiten {n}', { n: p.seiten.join(', ') }) : (p.seite ? t('Seite {n}', { n: p.seite }) : t('Dokument')))
            + ' – ' + esc(p.quelle) + ': ' + esc(p.text);
    }
    function detailHtml(project, dd) {
        const d = dd;
        const z = zustand(d.id);
        const probleme = d.probleme || [];
        const hp = d.hoerprobe || { kopf: [], seiten: [] };
        const seiten = seitenListe(d, z);
        if (!seiten.includes(z.seite)) z.seite = seiten.length ? seiten[0] : 0;
        const fid = 'ab_filter_' + d.id;
        let s = '<fieldset class="filter-fieldset ab-filter" id="' + fid + '" style="border:1px solid var(--border);border-radius:6px;padding:0.5rem 0.8rem;margin:0.8rem 0;">'
            + '<legend>' + t('Anzeigen') + '</legend>'
            + '<label class="filter-chip"><input type="radio" name="' + fid + '" value="alle"' + (z.filter === 'alle' ? ' checked' : '') + ' onchange="Abschluss.filter(' + project.id + ', ' + d.id + ', this.value)"> ' + t('Ganzes Dokument ({n} Seiten)', { n: hp.seiten.length }) + '</label> '
            + '<label class="filter-chip"><input type="radio" name="' + fid + '" value="probleme"' + (z.filter === 'probleme' ? ' checked' : '') + ' onchange="Abschluss.filter(' + project.id + ', ' + d.id + ', this.value)"> ' + t('Nur Problemstellen ({n})', { n: probleme.length }) + '</label>'
            + '</fieldset>';
        // Problemliste (immer sichtbar, kurz): nummeriert, mit Sprung zur Seite
        // Lange Listen (mehr als 10) zugeklappt, damit die Seitenansicht erreichbar bleibt; Zustand bleibt beim Blaettern.
        if (!probleme.length) {
            s += '<h4 id="ab_probleme_' + d.id + '">' + t('Problemstellen ({n})', { n: 0 }) + '</h4>'
                + '<p>' + t('Keine Problemstellen gefunden: PDF/UA-Prüfung, Struktur und Vollständigkeit sind ohne Befund.') + '</p>';
        } else {
            if (z.listeOffen === undefined) z.listeOffen = probleme.length <= 10;
            s += '<details class="ab-problemklappe" data-doc="' + d.id + '"' + (z.listeOffen ? ' open' : '') + ' ontoggle="Abschluss.listeGeklappt(' + d.id + ', this.open)">'
                + '<summary><h4 id="ab_probleme_' + d.id + '" class="ab-inline">' + t('Problemstellen ({n})', { n: probleme.length }) + '</h4></summary>'
                + '<ol class="ab-problemliste">' + probleme.map(p => '<li class="ab-problem ab-art-' + esc(p.art) + '"><span class="ab-marke" aria-hidden="true">!</span> ' + problemText(p)
                + (p.seite ? ' <button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.zurSeite(' + project.id + ', ' + d.id + ', ' + p.seite + ')">' + t('Zur Seite {n}', { n: p.seite }) + '</button>' : '') + '</li>').join('') + '</ol></details>';
        }
        // Kopf der Hörprobe (Sprache, Seiten, Zusammenfassung)
        if (hp.kopf && hp.kopf.length) s += '<ul class="dok-meta ab-kopf">' + hp.kopf.map(k => '<li>' + esc(k) + '</li>').join('') + '</ul>';
        // Seitenansicht
        if (!seiten.length) {
            s += '<p>' + (z.filter === 'probleme' ? t('Keine Seite mit Problemstellen.') : t('Die Hörprobe ist leer.')) + '</p>';
            return s;
        }
        const idx = seiten.indexOf(z.seite);
        const seiteDaten = hp.seiten.find(x => x.seite === z.seite) || { zeilen: [] };
        const pSeite = problemeDerSeite(d, z.seite);
        s += '<section class="ab-seite" id="ab_seite_' + d.id + '">'
            + '<h4 id="ab_seite_heading_' + d.id + '" tabindex="-1">' + t('Seite {n} von {m}', { n: z.seite, m: hp.seiten.length }) + (pSeite.length ? ' – ' + t('{n} Problemstellen', { n: pSeite.length }) : '') + '</h4>'
            + '<div class="ab-seitennav">'
            + '<button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.blaettern(' + project.id + ', ' + d.id + ', -1)"' + (idx <= 0 ? ' disabled' : '') + '>' + t('Vorherige Seite') + '</button>'
            + '<label for="ab_seitenwahl_' + d.id + '">' + t('Gehe zu Seite') + '</label>'
            + '<select id="ab_seitenwahl_' + d.id + '">' + seiten.map(n => { const k = problemeDerSeite(d, n).length; return '<option value="' + n + '"' + (n === z.seite ? ' selected' : '') + '>' + t('Seite {n}', { n: n }) + (k ? ' (' + t('{n} Problemstellen', { n: k }) + ')' : '') + '</option>'; }).join('') + '</select>'
            + '<button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.zurSeite(' + project.id + ', ' + d.id + ', Number(document.getElementById(\'ab_seitenwahl_' + d.id + '\').value))">' + t('Öffnen') + '</button>'
            + '<button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.blaettern(' + project.id + ', ' + d.id + ', 1)"' + (idx >= seiten.length - 1 ? ' disabled' : '') + '>' + t('Nächste Seite') + '</button>'
            + '</div>'
            + '<div class="ab-seite-inhalt">'
            + '<img class="ab-seitenbild" src="/api/projects/' + project.id + '/documents/' + d.id + '/abschluss/seite/' + z.seite + '?v=' + encodeURIComponent((d.pruefdatei && d.pruefdatei.erstellt_am) || '') + '" alt="' + t('Seitenbild von Seite {n}', { n: z.seite }) + '" loading="lazy">'
            + '<div class="ab-seite-text">'
            + (pSeite.length ? '<div class="ab-seite-probleme"><p><strong>' + t('Problemstellen auf dieser Seite') + '</strong></p><ul>' + pSeite.map(p => '<li><span class="ab-marke" aria-hidden="true">!</span> ' + t('Problem {n}', { n: p.nr }) + ': ' + esc(p.quelle) + ': ' + esc(p.text) + '</li>').join('') + '</ul></div>' : '')
            + '<p><button type="button" class="btn btn-secondary btn-small tts-btn" id="ab_vorlesen_' + d.id + '" aria-pressed="false" onclick="Abschluss.vorlesenSeite(' + d.id + ', this)">' + t('Seite vorlesen') + '</button></p>'
            + '<h5 class="ab-hoerprobe-titel">' + t('Hörprobe: so bekommt ein Screenreader diese Seite') + '</h5>'
            + '<div class="ausgabe-hoerprobe ab-hoerprobe" role="region" aria-label="' + t('Hörprobe von Seite {n}', { n: z.seite }) + '" tabindex="0">'
            + (seiteDaten.zeilen.length ? seiteDaten.zeilen.map(zl => '<p>' + esc(zl) + '</p>').join('') : '<p>' + t('Auf dieser Seite liest ein Screenreader nichts vor.') + '</p>')
            + '</div></div></div></section>';
        return s;
    }

    function detailNeuZeichnen(projectId, docId, fokus) {
        const box = document.getElementById('ab_detail_' + docId);
        if (!box || !details[docId]) return;
        box.innerHTML = detailHtml({ id: projectId }, details[docId]);
        if (fokus) { const h = document.getElementById(fokus); if (h) h.focus(); }
    }
    async function detailLaden(projectId, docId) {
        try {
            const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/abschluss', { credentials: 'same-origin' });
            if (!r.ok) throw new Error(String(r.status));
            details[docId] = await r.json();
            detailNeuZeichnen(projectId, docId);
        } catch (e) {
            const box = document.getElementById('ab_detail_' + docId);
            if (box) box.innerHTML = '<p>' + t('Die Prüfung konnte nicht geladen werden.') + '</p>';
        }
    }

    // ─── Bedienung ───
    function filter(projectId, docId, wert) {
        const z = zustand(docId);
        z.filter = wert === 'probleme' ? 'probleme' : 'alle';
        if (typeof vorlesenStopp === 'function') vorlesenStopp();
        detailNeuZeichnen(projectId, docId);
        const dd = details[docId] || {};
        announce(z.filter === 'probleme'
            ? t('Nur Problemstellen: {n} Seiten.', { n: seitenListe(dd, z).length })
            : t('Ganzes Dokument: {n} Seiten.', { n: ((dd.hoerprobe && dd.hoerprobe.seiten) || []).length }));
        const radio = document.querySelector('#ab_filter_' + docId + ' input[value="' + z.filter + '"]');
        if (radio) radio.focus();
    }
    function zurSeite(projectId, docId, seite) {
        const z = zustand(docId);
        const dd = details[docId];
        if (!dd) return;
        if (!seitenListe(dd, z).includes(seite)) z.filter = 'alle';
        z.seite = seite;
        if (typeof vorlesenStopp === 'function') vorlesenStopp();
        detailNeuZeichnen(projectId, docId, 'ab_seite_heading_' + docId);
    }
    function blaettern(projectId, docId, schritt) {
        const z = zustand(docId);
        const dd = details[docId];
        if (!dd) return;
        const liste = seitenListe(dd, z);
        const i = liste.indexOf(z.seite) + schritt;
        if (i < 0 || i >= liste.length) return;
        zurSeite(projectId, docId, liste[i]);
    }
    function vorlesenSeite(docId, btn) {
        const dd = details[docId];
        if (!dd || typeof vorlesen !== 'function') return;
        const z = zustand(docId);
        const seite = ((dd.hoerprobe && dd.hoerprobe.seiten) || []).find(x => x.seite === z.seite);
        const text = seite ? seite.zeilen.join('. ') : '';
        vorlesen(text, (dd.sprache || 'de').slice(0, 2).toLowerCase(), btn, t('Seite vorlesen'));
    }

    async function erstellen(projectId, docId) {
        const btn = document.getElementById('ab_erstellen_' + docId);
        const out = document.getElementById('ab_status_' + docId);
        if (btn) btn.disabled = true;
        if (out) out.textContent = t('Prüfdatei wird erstellt … Das kann bei großen Dateien eine Minute dauern.');
        announce(t('Prüfdatei wird erstellt.'));
        try {
            const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/abschluss', { method: 'POST', credentials: 'same-origin' });
            const j = await r.json().catch(() => ({}));
            if (!r.ok) {
                const grund = (j.detail && (j.detail.text || j.detail)) || t('unbekannter Fehler');
                if (out) { out.textContent = t('Die Prüfdatei konnte nicht erstellt werden: {grund}', { grund: typeof grund === 'string' ? grund : t('unbekannter Fehler') }); out.focus(); }
                if (btn) btn.disabled = false;
                return;
            }
            details[docId] = j;
            offeneDokumente.add(docId); geschlosseneDokumente.delete(docId);
            await showProject(projectId, true);
            const o2 = document.getElementById('ab_status_' + docId);
            const n = (j.probleme || []).length;
            const text = t('Prüfdatei erstellt.') + ' ' + (n ? t('{n} Problemstellen gefunden.', { n: n }) : t('Keine Problemstellen gefunden.'));
            if (o2) { o2.textContent = text; o2.focus(); }
            announce(text);
        } catch (e) {
            if (out) out.textContent = t('Verbindungsfehler.');
            if (btn) btn.disabled = false;
        }
    }

    async function herunterladenAntwort(res, out) {
        if (res.status === 402) {
            const e = await res.json().catch(() => ({}));
            if (out) out.textContent = '';
            if (typeof zeigeCreditsMeldung === 'function') zeigeCreditsMeldung(e.detail); else announce((e.detail && e.detail.text) || t('Dafür reicht das Guthaben nicht.'));
            return null;
        }
        if (!res.ok) {
            const e = await res.json().catch(() => ({}));
            const m = (e.detail && (e.detail.text || e.detail)) || t('Fehler beim Export.');
            if (out) { out.textContent = typeof m === 'string' ? m : t('Fehler beim Export.'); out.focus(); }
            return null;
        }
        const blob = await res.blob();
        const cd = res.headers.get('Content-Disposition') || '';
        const mStar = /filename\*=UTF-8''([^;]+)/i.exec(cd);
        const m = mStar || /filename="?([^";]+)"?/i.exec(cd);
        let nm = null;
        if (m) { try { nm = decodeURIComponent(m[1]); } catch (e) { nm = m[1]; } }
        nm = nm || 'inkludocs.pdf';
        if (typeof downloadBlob === 'function') downloadBlob(blob, nm);
        let ansage = t('Heruntergeladen: „{name}“.', { name: nm }) + ' ' + t('Die Datei liegt auch in deiner Ablage.');
        const credits = res.headers.get('X-Export-Credits');
        if (credits) ansage += ' ' + t('{c} Credits verbraucht.', { c: credits });
        return ansage;
    }
    async function herunterladen(projectId, docId) {
        const btn = document.getElementById('ab_export_' + docId);
        const out = document.getElementById('ab_status_' + docId);
        if (btn) btn.disabled = true;
        if (out) out.textContent = t('Wird exportiert...');
        announce(t('Export läuft …'));
        try {
            const res = await fetch('/api/projects/' + projectId + '/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ document_id: docId }) });
            const ansage = await herunterladenAntwort(res, out);
            if (ansage) {
                await showProject(projectId, true);
                const o2 = document.getElementById('ab_status_' + docId);
                if (o2) { o2.textContent = ansage; o2.focus(); }
                announce(ansage);
            }
        } catch (e) {
            if (out) out.textContent = t('Verbindungsfehler.');
        } finally {
            const b2 = document.getElementById('ab_export_' + docId);
            if (b2) b2.disabled = false;
        }
    }
    async function alleHerunterladen(projectId) {
        const btn = document.getElementById('abAlleBtn');
        const out = document.getElementById('abAlleStatus');
        if (btn) btn.disabled = true;
        if (out) out.textContent = t('Wird exportiert...');
        announce(t('Export läuft …'));
        try {
            const res = await fetch('/api/projects/' + projectId + '/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            const ansage = await herunterladenAntwort(res, out);
            if (ansage) {
                await showProject(projectId, true);
                const o2 = document.getElementById('abAlleStatus');
                if (o2) { o2.textContent = ansage; o2.focus(); }
                announce(ansage);
            }
        } catch (e) {
            if (out) out.textContent = t('Verbindungsfehler.');
        } finally {
            const b2 = document.getElementById('abAlleBtn');
            if (b2) b2.disabled = false;
        }
    }

    // ─── Ansicht ───
    async function showProject(projectId, erneut) {
        const main = document.getElementById('main');
        const res = await fetch('/api/projects/' + projectId + '/abschluss', { credentials: 'same-origin' });
        if (res.status === 401) { window.location.href = '/login'; return; }
        if (!res.ok) { main.innerHTML = '<div class="card"><p>' + t('Projekt konnte nicht geladen werden.') + '</p></div>'; return; }
        const data = await res.json();
        const project = data.project;
        if (zustandProjekt !== projectId) {
            offeneDokumente = new Set(); geschlosseneDokumente = new Set(); zustandProjekt = projectId;
            Object.keys(details).forEach(k => delete details[k]);
            Object.keys(ansicht).forEach(k => delete ansicht[k]);
        }
        const docs = data.documents || [];
        const title = (project.name && project.name.trim()) ? project.name : project.filename;
        const alleGetaggt = docs.length > 1 && docs.every(d => d.getaggt);
        const aktionen = (alleGetaggt ? '<button class="btn btn-primary" id="abAlleBtn" onclick="Abschluss.alleHerunterladen(' + project.id + ')">' + ico('download') + t('Alle Dokumente herunterladen') + '<span class="visually-hidden"> ' + t('als ZIP, mit Alt-Texten und Quickinfos') + '</span></button>' : '')
            + ((data.ausgaben_anzahl || 0) > 0 ? '<a class="btn btn-secondary" id="ausgabenTab" href="/ablage?projekt=' + project.id + '">' + t('Ablage ({n})', { n: data.ausgaben_anzahl || 0 }) + '</a>' : '')
            + (alleGetaggt ? '<output id="abAlleStatus" class="dok-status" tabindex="-1" style="flex-basis:100%;"></output>' : '');
        main.innerHTML = projektKopfHtml(project, 'abschluss', title, '<div class="card-info" id="projectHeadInfo" hidden></div>')
            + funktionenKarteHtml(aktionen)
            + '<h2 class="section-title" id="dokumenteHeading" tabindex="-1" style="margin-top:1.5rem">' + t('Dokumente ({n})', { n: docs.length }) + '</h2>'
            + (docs.length ? '<p class="feld-hinweis">' + t('Hier prüfst du das Ergebnis, bevor du es herunterlädst: Problemstellen, Seitenbild und Hörprobe der fertigen Datei.') + '</p>'
                           : '<p class="feld-hinweis">' + t('Noch kein Dokument hochgeladen. Das geht in der Ansicht „Dokument“.') + '</p>')
            + '<div id="abListe">' + docs.map((d, i) => karteHtml(project, d, i + 1, docs.length)).join('') + '</div>';
        document.querySelectorAll('details.ab-klappe').forEach(el => {
            const gezeichnetOffen = el.open;
            let erstesEreignis = true;
            el.addEventListener('toggle', () => {
                const echt = !(erstesEreignis && el.open === gezeichnetOffen);
                erstesEreignis = false;
                const k = Number(el.dataset.doc);
                if (el.open && !details[k]) {
                    const d = docs.find(x => x.id === k);
                    if (d && d.pruefdatei) detailLaden(project.id, k);
                }
                if (!echt) return;
                if (el.open) { offeneDokumente.add(k); geschlosseneDokumente.delete(k); }
                else { offeneDokumente.delete(k); geschlosseneDokumente.add(k); if (typeof vorlesenStopp === 'function') vorlesenStopp(); }
            });
            // offen gezeichnete Karte mit Prüfdatei: Inhalt gleich laden (toggle kommt nicht in jedem Browser)
            const k = Number(el.dataset.doc);
            const d = docs.find(x => x.id === k);
            if (el.open && d && d.pruefdatei && !details[k]) detailLaden(project.id, k);
        });
        const h1 = document.getElementById('projectName');
        if (h1 && !erneut) h1.focus();
    }

    function listeGeklappt(docId, offen) { zustand(docId).listeOffen = !!offen; }

    window.Abschluss = { showProject, erstellen, herunterladen, alleHerunterladen, filter, zurSeite, blaettern, vorlesenSeite, listeGeklappt };
})();
