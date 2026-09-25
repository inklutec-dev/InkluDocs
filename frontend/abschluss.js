/*
 * I18N: alle sichtbaren Texte über t() (window.I18N), Kataloge backend/locales, geprüft von scripts/check_i18n.py.
 * Station „Prüfung“ eines PDF-Projekts (24.09.2026 als „Abschlussprüfung“; umbenannt und umgebaut am 25.09.2026 nach
 * Michael Karbe, Feedback 24.09.2026 - 3). Intern heißt die Ansicht weiter 'abschluss'.
 *
 * Die FERTIGE Datei prüfen: Problemstellen finden, Problemseiten ansehen und anhören. Geprüft wird immer genau die
 * PDF, die der Kunde herunterlädt (Backend: main.py „STATION ABSCHLUSSPRUEFUNG“, abschluss.py). Die Prüfdatei kostet
 * nichts. HERUNTERGELADEN wird in der Ansicht „Dokument“ (Punkte 4 und 6), nicht hier.
 *
 * Aufbau: Projektkopf (app.html projektKopfHtml), je Datei eine aufklappbare Karte (<details>, H3 im summary). In der
 * offenen Karte: Dokumentinfos samt Sprache und Zusammenfassung (Punkt 10), Knöpfe, die KI-basierte Prüfung als Knopf
 * (Punkt 13, Block aus dokument.js), die Problemliste ohne „!“ (Punkt 7), darunter nach einer Linie (Punkt 11) die
 * Problemseiten — nur die (Punkt 12, kein Filter mehr) — mit Seitenbild und Hörprobe; „Vorherige/Nächste Seite“
 * nebeneinander (Punkt 8). Vorlesen: Ansage in der Kontosprache, Inhalt in der Dokumentsprache (Punkt 9, app.html
 * vorlesenTeile), nur mit Stimmen auf dem Gerät — der Dokumenttext geht an keinen Sprachdienst im Netz.
 *
 * Gemeinsame Helfer aus app.html/dashboard.js/dokument.js: t(), announce(), escHtml(), icon(), docDisplayName(),
 * projektKopfHtml(), vorlesenTeile(), vorlesenStopp(), Dokument.kiBlockHtml() und Dokument.setNeuLaden().
 */
(function () {
    'use strict';

    let zustandProjekt = null;
    let offeneDokumente = new Set();
    let geschlosseneDokumente = new Set();
    const details = {};     // docId -> volle Daten (Probleme, Hörprobe)
    const laedt = new Set();   // docIds, deren Details gerade geladen werden (kein doppelter Abruf)
    let pollTimer = null;      // solange eine Prüfdatei gebaut wird (auch aus einem anderen Tab)
    const ansicht = {};     // docId -> { seite: n, listeOffen }
    let dokDaten = {};      // docId -> Dokument aus /dokument-ansicht (Stand der KI-Pruefung fuer den KI-Block)
    let dokProjekt = null;

    function ico(name) { return (typeof icon === 'function') ? icon(name) : ''; }
    function esc(s) { return (typeof escHtml === 'function') ? escHtml(s == null ? '' : String(s)) : String(s == null ? '' : s); }
    function name(d) { return (typeof docDisplayName === 'function') ? docDisplayName(d) : (d.display_name || d.original_filename || ''); }

    // ─── Stand-Texte ───
    function standText(d) {
        if (!d.getaggt) return t('Noch nicht getaggt');
        const p = d.pruefdatei;
        if (!p) return t('Noch keine Prüfdatei');
        if (!p.aktuell) return t('Prüfdatei nicht mehr aktuell');
        if (p.anzahl_probleme == null) return t('Prüfdatei wird erstellt …');
        const n = p.anzahl_probleme || 0;
        return n ? anzahlProbleme(n) : t('Keine Problemstellen gefunden');
    }
    function standKlasse(d) {
        if (!d.getaggt || !d.pruefdatei || !d.pruefdatei.aktuell) return 'badge-ready';
        return (d.pruefdatei.anzahl_probleme || 0) ? 'badge-processing' : 'badge-done';
    }
    function metaZeile(bez, wert) { return '<li>' + bez + ': <span>' + wert + '</span></li>'; }
    function anzahlProbleme(n) { return n === 1 ? t('1 Problemstelle') : t('{n} Problemstellen', { n: n }); }

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
                + (p && p.anzahl_probleme != null ? metaZeile(t('Problemstellen'), esc(p.anzahl_probleme)) : '')
                + (p && p.vollstaendigkeit_geprueft === false ? metaZeile(t('Vollständigkeit'), t('nicht geprüft')) : '')
                + '</ul>';
        }
        const erstellenText = !p ? t('Prüfdatei erstellen') : t('Prüfdatei neu erstellen');
        const erstellenPrimaer = !p || !p.aktuell;
        // Kein „PDF herunterladen“ mehr hier (Feedback 24.09.2026 - 3, Punkt 6) — das macht die Ansicht „Dokument“.
        const aktionen = d.getaggt
            ? '<div class="ausgabe-aktionen">'
              + '<button type="button" class="btn ' + (erstellenPrimaer ? 'btn-primary' : 'btn-secondary') + '" id="ab_erstellen_' + d.id + '" onclick="Abschluss.erstellen(' + project.id + ', ' + d.id + ')"' + (d.laeuft ? ' disabled' : '') + '>' + ico('sparkle') + erstellenText + '<span class="visually-hidden"> ' + vh + ', ' + t('kostenlos') + '</span></button>'
              + (p ? '<a class="btn btn-secondary" id="ab_struktur_' + d.id + '" href="/struktur/' + project.id + '/' + d.id + '?quelle=abschluss">' + t('Mit eigenem Screenreader prüfen') + '<span class="visually-hidden"> ' + vh + '</span></a>' : '')
              + '</div>'
            : '';
        const dk = dokDaten[d.id];
        const ki = (dk && dokProjekt && window.Dokument && typeof Dokument.kiBlockHtml === 'function') ? Dokument.kiBlockHtml(dokProjekt, dk) : '';
        return '<section class="card dok-karte ab-karte" id="ab_karte_' + d.id + '">'
            + '<details class="dok-klappe ab-klappe" data-doc="' + d.id + '"' + (karteOffen(d, anzahl) ? ' open' : '') + '>'
            + '<summary><h3 id="ab_heading_' + d.id + '" class="doc-heading dok-kopfzeile"><span>' + t('Dokument {n}: {name}', { n: pos, name: nm }) + '</span> <span class="badge ' + standKlasse(d) + '" id="ab_badge_' + d.id + '">' + esc(standText(d)) + '</span></h3></summary>'
            + '<div class="ab-inhalt">'
            + meta
            // Sprache und Zusammenfassung oben bei den Infos (Punkt 10); gefuellt, sobald die Details geladen sind
            + '<ul class="dok-meta ab-kopf" id="ab_kopf_' + d.id + '">' + (details[d.id] ? kopfZeilenHtml(details[d.id]) : '') + '</ul>'
            + (p ? '<p class="feld-hinweis">' + t('Geprüft wird die fertige Datei, genau die PDF, die du in der Ansicht „Dokument“ herunterlädst. Das Erstellen der Prüfdatei ist kostenlos.') + '</p>'
                 : (d.getaggt ? '<p class="feld-hinweis">' + t('Erstelle die Prüfdatei: Sie ist genau die PDF, die du herunterlädst, mit Struktur, Alt-Texten und Quickinfos. Das ist kostenlos.') + '</p>' : ''))
            + (p && !p.aktuell ? '<p class="feld-hinweis"><strong>' + t('Die Prüfdatei ist nicht mehr aktuell.') + '</strong> ' + t('Erstelle sie neu, damit du genau die Datei prüfst, die du herunterlädst.') + '</p>' : '')
            + aktionen
            + '<output id="ab_status_' + d.id + '" class="dok-status" style="display:block;margin-top:0.5rem;" tabindex="-1">' + (d.laeuft ? t('Prüfdatei wird erstellt …') : '') + '</output>'
            + ki
            + '<div class="ab-detail" id="ab_detail_' + d.id + '">' + (d.laeuft ? '' : (p && details[d.id] ? detailHtml(project, details[d.id]) : (p ? '<p>' + t('Wird geladen …') + '</p>' : ''))) + '</div>'
            + '</div></details></section>';
    }

    // Kopf der Hoerprobe: Sprache und Zusammenfassung (die Seitenzahl steht schon in den Infos)
    function kopfZeilenHtml(dd) {
        const kopf = ((dd.hoerprobe && dd.hoerprobe.kopf) || []).filter((k, i) => i !== 1);
        return kopf.map(k => '<li>' + esc(k) + '</li>').join('');
    }

    // ─── Inhalt einer offenen Karte: Problemliste, Problemseiten ───
    function zustand(docId) {
        if (!ansicht[docId]) ansicht[docId] = { seite: 0 };
        return ansicht[docId];
    }
    function problemeDerSeite(dd, seite) { return (dd.probleme || []).filter(p => (p.seiten && p.seiten.length ? p.seiten.includes(seite) : p.seite === seite)); }
    // Nur Seiten mit Problemstellen (Michael Karbe, Feedback 24.09.2026 - 3, Punkt 12: Filter gestrichen)
    function seitenListe(dd) {
        const alle = ((dd.hoerprobe && dd.hoerprobe.seiten) || []).map(s => s.seite);
        const mit = new Set();
        (dd.probleme || []).forEach(p => (p.seiten && p.seiten.length ? p.seiten : [p.seite]).forEach(s => { if (s) mit.add(s); }));
        return alle.filter(s => mit.has(s));
    }
    function problemText(p) {
        return (p.seiten && p.seiten.length > 1 ? t('Seiten {n}', { n: p.seiten.join(', ') }) : (p.seite ? t('Seite {n}', { n: p.seite }) : t('Dokument')))
            + ' – ' + esc(p.quelle) + ': ' + esc(p.text);
    }
    // Sprache des Dokuments nur als saubere Sprachkennung (de, de-DE, en-GB …) — geht in lang="" und an die Stimme
    function dokSprache(dd) {
        const s = String(dd.sprache || '').trim();
        return /^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*$/.test(s) ? s : '';
    }
    // Eine Hoerproben-Zeile „Ansage: Inhalt“ in ihre Teile: die Ansage stammt von uns (Oberflaechensprache), der Inhalt
    // aus dem Dokument. Zeilen ohne „: “ sind reine Ansagen („Liste mit 3 Einträgen“, „Grafik ohne Alt-Text“).
    function zeilenTeile(zl) {
        const i = String(zl).indexOf(': ');
        return i > 0 ? { ansage: zl.slice(0, i), inhalt: zl.slice(i + 2) } : { ansage: zl, inhalt: '' };
    }
    function zeileHtml(zl, lang) {
        const z = zeilenTeile(zl);
        if (!z.inhalt) return '<p>' + esc(z.ansage) + '</p>';
        // lang am Inhalt (Punkt 9): VoiceOver/NVDA wechseln dort selbst die Stimme, wie im echten Dokument
        return '<p>' + esc(z.ansage) + ': <span' + (lang ? ' lang="' + esc(lang) + '"' : '') + '>' + esc(z.inhalt) + '</span></p>';
    }
    function detailHtml(project, dd) {
        const d = dd;
        const z = zustand(d.id);
        const probleme = d.probleme || [];
        const hp = d.hoerprobe || { kopf: [], seiten: [] };
        const seiten = seitenListe(d);
        if (!seiten.includes(z.seite)) z.seite = seiten.length ? seiten[0] : 0;
        let s = '';
        // Problemliste: nummeriert, mit Sprung zur Seite, ohne „!“ (Punkt 7). Lange Listen (mehr als 10) zugeklappt,
        // damit die Seitenansicht erreichbar bleibt; Zustand bleibt beim Blaettern.
        if (!probleme.length) {
            s += '<h4 id="ab_probleme_' + d.id + '">' + t('Problemstellen ({n})', { n: 0 }) + '</h4>'
                + '<p>' + t('Keine Problemstellen gefunden: PDF/UA-Prüfung, Struktur und Vollständigkeit sind ohne Befund.') + '</p>';
            return s;
        }
        if (z.listeOffen === undefined) z.listeOffen = probleme.length <= 10;
        s += '<details class="ab-problemklappe" data-doc="' + d.id + '"' + (z.listeOffen ? ' open' : '') + ' ontoggle="Abschluss.listeGeklappt(' + d.id + ', this.open)">'
            + '<summary><h4 id="ab_probleme_' + d.id + '" class="ab-inline">' + t('Problemstellen ({n})', { n: probleme.length }) + '</h4></summary>'
            + '<ol class="ab-problemliste">' + probleme.map(p => '<li class="ab-problem">' + problemText(p)
            + (p.seite ? ' <button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.zurSeite(' + project.id + ', ' + d.id + ', ' + p.seite + ')">' + t('Zur Seite {n}', { n: p.seite }) + '</button>' : '') + '</li>').join('') + '</ol></details>';
        if (!seiten.length) {
            // Probleme ohne Seitenangabe (z. B. das ganze Dokument betreffend): keine Seitenansicht
            s += '<p>' + t('Keine der Problemstellen gehört zu einer bestimmten Seite.') + '</p>';
            return s;
        }
        const idx = seiten.indexOf(z.seite);
        const seiteDaten = hp.seiten.find(x => x.seite === z.seite) || { zeilen: [] };
        const pSeite = problemeDerSeite(d, z.seite);
        const lang = dokSprache(d);
        s += '<section class="ab-seite" id="ab_seite_' + d.id + '" aria-labelledby="ab_seite_heading_' + d.id + '">'
            + '<h4 id="ab_seite_heading_' + d.id + '" tabindex="-1">' + t('Problemseite {i} von {n}: Seite {s}', { i: idx + 1, n: seiten.length, s: z.seite }) + ' – ' + anzahlProbleme(pSeite.length) + '</h4>'
            // „Vorherige Seite“ und „Nächste Seite“ direkt nebeneinander, danach die Seitenwahl (Punkt 8)
            + '<div class="ab-seitennav">'
            + '<button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.blaettern(' + project.id + ', ' + d.id + ', -1)"' + (idx <= 0 ? ' disabled' : '') + '>' + t('Vorherige Seite') + '</button>'
            + '<button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.blaettern(' + project.id + ', ' + d.id + ', 1)"' + (idx >= seiten.length - 1 ? ' disabled' : '') + '>' + t('Nächste Seite') + '</button>'
            + '<span class="ab-seitenwahl"><label for="ab_seitenwahl_' + d.id + '">' + t('Gehe zu Seite') + '</label> '
            + '<select id="ab_seitenwahl_' + d.id + '">' + seiten.map(n => { const k = problemeDerSeite(d, n).length; return '<option value="' + n + '"' + (n === z.seite ? ' selected' : '') + '>' + t('Seite {n}', { n: n }) + ' (' + anzahlProbleme(k) + ')</option>'; }).join('') + '</select> '
            + '<button type="button" class="btn btn-secondary btn-small" onclick="Abschluss.zurSeite(' + project.id + ', ' + d.id + ', Number(document.getElementById(\'ab_seitenwahl_' + d.id + '\').value))">' + t('Öffnen') + '</button></span>'
            + '</div>'
            + '<div class="ab-seite-inhalt">'
            + '<img class="ab-seitenbild" src="/api/projects/' + project.id + '/documents/' + d.id + '/abschluss/seite/' + z.seite + '?v=' + encodeURIComponent((d.pruefdatei && d.pruefdatei.erstellt_am) || '') + '" alt="' + t('Seitenbild von Seite {n}', { n: z.seite }) + '" loading="lazy">'
            + '<div class="ab-seite-text">'
            + '<div class="ab-seite-probleme"><p><strong>' + t('Problemstellen auf dieser Seite') + '</strong></p><ul>' + pSeite.map(p => '<li>' + t('Problem {n}', { n: p.nr }) + ': ' + esc(p.quelle) + ': ' + esc(p.text) + '</li>').join('') + '</ul></div>'
            + '<p><button type="button" class="btn btn-secondary btn-small tts-btn" id="ab_vorlesen_' + d.id + '" aria-pressed="false" onclick="Abschluss.vorlesenSeite(' + d.id + ', this)">' + t('Seite vorlesen') + '</button></p>'
            + '<h5 class="ab-hoerprobe-titel">' + t('Hörprobe: so bekommt ein Screenreader diese Seite') + '</h5>'
            + '<div class="ausgabe-hoerprobe ab-hoerprobe" role="region" aria-label="' + t('Hörprobe von Seite {n}', { n: z.seite }) + '" tabindex="0">'
            + (seiteDaten.zeilen.length ? seiteDaten.zeilen.map(zl => zeileHtml(zl, lang)).join('') : '<p>' + t('Auf dieser Seite liest ein Screenreader nichts vor.') + '</p>')
            + '</div></div></div></section>';
        return s;
    }

    function detailNeuZeichnen(projectId, docId, fokus) {
        const box = document.getElementById('ab_detail_' + docId);
        if (!box || !details[docId]) return;
        box.innerHTML = detailHtml({ id: projectId }, details[docId]);
        const kopf = document.getElementById('ab_kopf_' + docId);
        if (kopf) kopf.innerHTML = kopfZeilenHtml(details[docId]);
        if (fokus) { const h = document.getElementById(fokus); if (h) h.focus(); }
    }
    async function detailLaden(projectId, docId) {
        if (laedt.has(docId)) return;
        laedt.add(docId);
        try {
            const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/abschluss', { credentials: 'same-origin' });
            if (!r.ok) throw new Error(String(r.status));
            details[docId] = await r.json();
            detailNeuZeichnen(projectId, docId);
        } catch (e) {
            const box = document.getElementById('ab_detail_' + docId);
            if (box) box.innerHTML = '<p>' + t('Die Prüfung konnte nicht geladen werden.') + '</p>';
        } finally {
            laedt.delete(docId);
        }
    }

    // ─── Bedienung ───
    function zurSeite(projectId, docId, seite) {
        const z = zustand(docId);
        const dd = details[docId];
        if (!dd || !seitenListe(dd).includes(seite)) return;
        z.seite = seite;
        if (typeof vorlesenStopp === 'function') vorlesenStopp();
        detailNeuZeichnen(projectId, docId, 'ab_seite_heading_' + docId);
    }
    function blaettern(projectId, docId, schritt) {
        const z = zustand(docId);
        const dd = details[docId];
        if (!dd) return;
        const liste = seitenListe(dd);
        const i = liste.indexOf(z.seite) + schritt;
        if (i < 0 || i >= liste.length) return;
        zurSeite(projectId, docId, liste[i]);
    }
    // Vorlesen (Punkt 9): Ansage in der Kontosprache, Inhalt in der Dokumentsprache — vorlesenTeile waehlt je Teil die
    // Stimme und faellt ohne lokale Stimme der Dokumentsprache auf die Kontosprache zurueck.
    function vorlesenSeite(docId, btn) {
        const dd = details[docId];
        if (!dd || typeof vorlesenTeile !== 'function') return;
        const z = zustand(docId);
        const seite = ((dd.hoerprobe && dd.hoerprobe.seiten) || []).find(x => x.seite === z.seite);
        const lang = dokSprache(dd);
        const teile = [];
        (seite ? seite.zeilen : []).forEach(zl => {
            const zt = zeilenTeile(zl);
            teile.push({ text: zt.ansage + (zt.inhalt ? ':' : '.'), lang: '' });
            if (zt.inhalt) teile.push({ text: zt.inhalt + '.', lang: lang });
        });
        vorlesenTeile(teile, btn, t('Seite vorlesen'));
    }

    async function erstellen(projectId, docId) {
        const btn = document.getElementById('ab_erstellen_' + docId);
        const out = document.getElementById('ab_status_' + docId);
        if (btn) btn.disabled = true;
        if (out) { out.textContent = t('Prüfdatei wird erstellt … Das kann bei großen Dateien eine Minute dauern.'); out.focus(); }
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
            const text = (j.neu_gebaut === false ? t('Die Prüfdatei ist schon aktuell.') : t('Prüfdatei erstellt.')) + ' '
                + (n ? (n === 1 ? t('1 Problemstelle gefunden.') : t('{n} Problemstellen gefunden.', { n: n })) : t('Keine Problemstellen gefunden.'));
            // Die Statuszeile ist ein <output> (Live-Region) und bekommt den Fokus — keine zusaetzliche announce()
            if (o2) { o2.textContent = text; o2.focus(); } else { announce(text); }
        } catch (e) {
            if (out) out.textContent = t('Verbindungsfehler.');
            if (btn) btn.disabled = false;
        }
    }

    // ─── Ansicht ───
    // KI-basierte Pruefung laeuft (aus dieser Ansicht gestartet): nur die Statuszeile fortschreiben, kein Neuaufbau
    // (Fokus bleibt); am Ende einmal neu zeichnen, die Problemliste frisch laden und das Ergebnis in die Statuszeile.
    let kiTimer = null;
    function kiPollStoppen() { if (kiTimer) { clearTimeout(kiTimer); kiTimer = null; } }
    function kiLaeuft(dk) {
        const pr = dk && dk.tagging && dk.tagging.pruefung;
        return !!(pr && (pr.laeuft || (pr.korrektur && pr.korrektur.laeuft)));
    }
    function kiPollStarten(projectId) {
        kiPollStoppen();
        const laufend = Object.keys(dokDaten).map(Number).filter(k => kiLaeuft(dokDaten[k]));
        if (!laufend.length) return;
        const tick = async () => {
            kiTimer = null;
            if (zustandProjekt !== projectId || !document.getElementById('abListe')) return;   // Ansicht gewechselt
            try {
                const r = await fetch('/api/projects/' + projectId + '/dokument-ansicht', { credentials: 'same-origin' });
                if (!r.ok) { kiTimer = setTimeout(tick, 2500); return; }
                const d2 = await r.json();
                if (zustandProjekt !== projectId || !document.getElementById('abListe')) return;
                const fertig = (d2.documents || []).filter(x => laufend.includes(x.id) && !kiLaeuft(x));
                if (fertig.length) {
                    fertig.forEach(x => { delete details[x.id]; offeneDokumente.add(x.id); geschlosseneDokumente.delete(x.id); });
                    await showProject(projectId, true);
                    const x = fertig[0];
                    const korr = x.tagging && x.tagging.pruefung && x.tagging.pruefung.korrektur && x.tagging.pruefung.korrektur.bericht && x.tagging.pruefung.korrektur.bericht.zeit;
                    const text = fertig.map(y => Dokument.pruefAbschlussText(y)).join(' ');
                    const out = document.getElementById('dok_pruef_status_' + x.id);
                    if (out) { out.textContent = korr ? Dokument.korrAbschlussText(x) : text; out.focus(); } else { announce(text); }
                    return;
                }
                (d2.documents || []).forEach(x => {
                    const pr = x.tagging && x.tagging.pruefung;
                    const out = document.getElementById('dok_pruef_status_' + x.id);
                    if (pr && pr.laeuft && out) {
                        const txt = t('Prüfung läuft … Seite {a} von {b}.', { a: pr.seite || 0, b: pr.seiten || 0 });
                        if (out.textContent !== txt) out.textContent = txt;
                    }
                });
                kiTimer = setTimeout(tick, 2500);
            } catch (e) { kiTimer = setTimeout(tick, 2500); }
        };
        kiTimer = setTimeout(tick, 2500);
    }

    async function showProject(projectId, erneut) {
        projectId = Number(projectId);   // Adresse liefert Text, Knoepfe eine Zahl — ohne das ging der Zustand verloren
        if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
        kiPollStoppen();
        if (typeof vorlesenStopp === 'function') vorlesenStopp();
        const main = document.getElementById('main');
        // Stand der Pruefdatei + Stand der KI-Pruefung (derselbe Abruf wie in „Dokument“) parallel
        const [res, resDok] = await Promise.all([
            fetch('/api/projects/' + projectId + '/abschluss', { credentials: 'same-origin' }),
            fetch('/api/projects/' + projectId + '/dokument-ansicht', { credentials: 'same-origin' }).catch(() => null),
        ]);
        if (res.status === 401) { window.location.href = '/login'; return; }
        if (!res.ok) { main.innerHTML = '<div class="card"><p>' + t('Projekt konnte nicht geladen werden.') + '</p></div>'; return; }
        const data = await res.json();
        const dokJson = (resDok && resDok.ok) ? await resDok.json().catch(() => null) : null;
        dokDaten = {};
        dokProjekt = dokJson ? dokJson.project : null;
        ((dokJson && dokJson.documents) || []).forEach(x => { dokDaten[x.id] = x; });
        const project = data.project;
        if (zustandProjekt !== projectId) {
            offeneDokumente = new Set(); geschlosseneDokumente = new Set(); zustandProjekt = projectId;
            Object.keys(ansicht).forEach(k => delete ansicht[k]);
        }
        // Details beim Oeffnen der Ansicht frisch laden; beim Neuzeichnen nach einer Aktion bleiben sie.
        if (!erneut) Object.keys(details).forEach(k => delete details[k]);
        // Nach einer KI-Pruefung/Korrektur aus dieser Ansicht zeichnet DIESE Ansicht neu (nicht „Dokument“)
        if (window.Dokument && typeof Dokument.setNeuLaden === 'function') Dokument.setNeuLaden(pid => showProject(pid, true));
        const docs = data.documents || [];
        const title = (project.name && project.name.trim()) ? project.name : project.filename;
        main.innerHTML = projektKopfHtml(project, 'abschluss', title, '<div class="card-info" id="projectHeadInfo" hidden></div>')
            + '<h2 class="section-title" id="dokumenteHeading" tabindex="-1" style="margin-top:1.5rem">' + t('Dokumente ({n})', { n: docs.length }) + '</h2>'
            + (docs.length ? '<p class="feld-hinweis">' + t('Hier prüfst du das Ergebnis: Problemstellen, Seitenbild und Hörprobe der fertigen Datei. Heruntergeladen wird in der Ansicht „Dokument“.') + '</p>'
                           : '<p class="feld-hinweis">' + t('Noch kein Dokument hochgeladen. Das geht in der Ansicht „Dokument“.') + '</p>')
            + '<div id="abListe">' + docs.map((d, i) => karteHtml(project, d, i + 1, docs.length)).join('') + '</div>';
        if (window.Dokument && typeof Dokument.kiKlappenBinden === 'function') Dokument.kiKlappenBinden();
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
        // Laeuft ein Bau (auch aus einem anderen Tab), die Ansicht nachziehen, bis er fertig ist
        if (docs.some(d => d.laeuft)) {
            pollTimer = setTimeout(() => {
                pollTimer = null;
                if (document.getElementById('abListe')) showProject(projectId, true);
            }, 3000);
        } else {
            kiPollStarten(projectId);
        }
    }

    function listeGeklappt(docId, offen) { zustand(docId).listeOffen = !!offen; }

    window.Abschluss = { showProject, erstellen, zurSeite, blaettern, vorlesenSeite, listeGeklappt };
})();
