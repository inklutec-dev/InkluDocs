/* =============================================================================
 * dokument.js — Ansicht „Dokument“ eines PDF-Projekts (PDF-Tagging, 22.09.2026)
 * I18N: alle sichtbaren Texte über t() (window.I18N), Kataloge backend/locales, geprüft von scripts/check_i18n.py
 *       (seit 24.09.2026 — vorher erfasste der Check diese Datei nicht, die Texte blieben in allen Sprachen deutsch).
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
 * „Barrierefrei machen“ (PDFix-Tagging, tagging_api.py), „PDF herunterladen“, „Umbenennen“,
 * „Loeschen“. Der Bericht des letzten Laufs liegt als Klappe unter der Karte. Seit 25.09.2026
 * (Michael Karbe, Feedback 24.09.2026 - 2 und - 3): Knoepfe unter einer Linie ueber die volle
 * Breite, Ergebnis des Laufs farbig IN der Karte, KI-basierte Pruefung in der Station „Prüfung“.
 *
 * FORM fuer Screenreader: H1 Projekt, H2 „PDF hinzufuegen“ (Upload), H2 „Dokumente (n)“,
 * je Datei eine H3; darunter eine Beschreibungsliste (dl), native Knoepfe, ein <output> je
 * Datei fuer den Laufstatus (role status), der Bericht als <details>. Rueckfrage vor dem Lauf
 * als natives <dialog> wie #genConfirmDialog (Fokusfang, Escape, Abbrechen links / Start rechts).
 * Ansagen nur bei Zustandswechseln (Start, fertig, Fehler), nicht bei jedem Tick.
 *
 * WORD-PROJEKTE (30.09.2026, Steve: „Word soll die gleiche Ansicht wie PDF bekommen, auch mit der Dokumentenverwaltung“):
 * dieselbe Datei zeichnet auch die Ansicht „Dokument“ eines Word-Projekts — gleicher Kopf, gleiche Karte (<details>, H3 im
 * summary), gleiche Knopfleiste unter der Linie, gleiche Dialoge (Hörprobe, Herunterladen, Umbenennen, Löschen). Nur die
 * Dokumentinfos kommen aus der Word-Datei (Titel, Anwendung, Seiten falls bekannt, Sprache, Überschriften, Tabellen,
 * Bilder; backend/docx_ansicht.py, ohne KI), und „Herunterladen“ öffnet den Dialog im Modus 'word' (Word-Datei,
 * barrierefreie PDF, Übersetzung). Die Hörprobe liest die Word-Datei mit den Alt-Texten aus InkluDocs
 * (POST …/export/pdfua/vorschau, kostenlos). Datenquelle bleibt GET /api/projects/{id}/dokument-ansicht (Weiche nach
 * Dateityp im Backend). Tagging gibt es bei Word nicht.
 *
 * Gemeinsame Helfer aus app.html/dashboard.js: t(), announce(), escHtml(), uploadBlockHtml(),
 * setupProjectDropzone(), docDisplayName(), openDocRename(), openDocDelete(), icon(),
 * ansichtWahlHtml(), inkluagentSectionHtml(), inkluagentInit(), zeigeCreditsMeldung().
 * Datenquelle: GET /api/projects/{id}/dokument-ansicht (tagging_api.py).
 * Sicherheit: alle Servertexte laufen durch escHtml(); keine innerHTML mit rohen Nutzerdaten.
 * ========================================================================== */
(function () {
    'use strict';

    // AUSGEBLENDET (Steve 24.09.2026, nicht geloescht): Strukturansicht-Link und KI-basierte Pruefung in der Dokument-
    // Karte. Die Strukturansicht der fertigen Datei gibt es in der Abschlusspruefung („Mit eigenem Screenreader pruefen“);
    // die KI-Pruefung soll spaeter im Hintergrund laufen und im Tagging-Preis stecken statt per Knopf. Zum Wieder-
    // einblenden den Schalter auf true setzen.
    // Seit 30.09.2026 aus EINEM Ort: backend/funktionen.py (window.FUNKTIONEN) — derselbe Schalter blendet die Funktion
    // auch im Chatbot und in den Endpunkten aus.
    const F = window.FUNKTIONEN || {};
    const ZEIGE_STRUKTURANSICHT = !!F.strukturansicht;
    // KI-basierte Pruefung: seit 25.09.2026 NICHT mehr in „Dokument“, sondern als Knopf in der Station „Prüfung“
    // (Michael Karbe, Feedback 24.09.2026 - 3, Punkt 13). Der Block kommt weiter von hier (kiBlockHtml), die Prüfung
    // zeichnet ihn kompakt. Das Urteil bleibt aus (Feedback 24.09.2026, Punkt 6: Anwender bilden sich ihr Urteil).
    const ZEIGE_KI_PRUEFUNG = !!F.ki_pruefung;
    const ZEIGE_KORREKTUR = !!F.korrektur;
    const ZEIGE_URTEIL = !!F.urteil;
    // Kette „Komplett barrierefrei machen“ und Ablage-Knopf im Kopf (Feedback 24.09.2026, Punkt 1: vorerst aus)
    const ZEIGE_PROJEKT_KNOEPFE = !!F.kette;

    let zustandProjekt = null;
    let aktuelleDaten = null;
    // BETRIEBSART (Michael Karbe, Feedback 20260928 - 2, Punkte 1, 6, 7): dieselbe Datei zeichnet zwei Ansichten eines
    // PDF-Projekts — 'dokument' = reine Dateiverwaltung (Metadaten; Hörprobe, Herunterladen, Umbenennen, Löschen) und
    // 'tagging' = Barrierefrei machen, Testweise taggen, Hörprobe (Struktur und Bilder in der Karte, Bericht, Testlauf).
    // Beide teilen Datenquelle, Rückfrage, Fortschritt und Abschlussmeldung, damit nichts doppelt gepflegt wird.
    let modus = 'dokument';
    let istWord = false;   // Word-Projekt (project_type 'docx'): Ansicht „Dokument“ mit Word-Karten (30.09.2026)
    let pollTimer = null;
    let laufZielDoc = null;
    let laufAktiv = false;
    let offeneBerichte = new Set();
    let offenePruefungen = new Set();   // KI-basierte Pruefung: Klappen, die offen bleiben sollen
    let offeneDokumente = new Set();    // Dokument-Klappen (Michael Karbe, PS 24.09.2026), bleiben beim Neuzeichnen offen
    let geschlosseneDokumente = new Set();   // ... bzw. zu, wenn der Nutzer sie zugeklappt hat
    // Ergebnis des letzten Laufs je Dokument, IN der Karte unter dem Dokument (Michael Karbe, Feedback 24.09.2026 - 2,
    // Punkt 2): farbig, mit Fokus, bleibt bis „Meldung schließen“ oder zum nächsten Lauf. docId -> {text, fehler}
    const ergebnisMeldung = {};
    // Wer nach einer KI-Pruefung/Korrektur neu zeichnet: die Ansicht, die den Block gerade zeigt (seit 25.09.2026 die
    // Station „Prüfung“, abschluss.js setzt das ueber Dokument.setNeuLaden).
    let neuLaden = (pid) => showProject(pid, true);

    function ico(name) { return (typeof icon === 'function') ? icon(name) : ''; }
    function esc(s) { return (typeof escHtml === 'function') ? escHtml(s == null ? '' : String(s)) : String(s == null ? '' : s); }
    function sprachText(code) { return (typeof spracheAnzeige === 'function') ? spracheAnzeige(code) : String(code || ''); }

    // ─── Texte ───
    function standText(d) {
        const tg = d.tagging || {};
        if (tg.laeuft) return t('Wird barrierefrei gemacht …');
        if (tg.status === 'fehler') return t('Letzter Lauf fehlgeschlagen');
        // Nur „Getaggt“ / „Nicht getaggt“ (Michael Karbe, Feedback 24.09.2026, Punkt 4) — das PDF/UA-Ergebnis steht im Bericht
        // Beim Hochladen schon getaggt (Quelle hat Tags, InkluDocs taggt sie nicht neu): unterscheidbar von „Getaggt“, das man
        // neu taggen kann (Prüfung Barrierefreiheit 30.09.2026, Punkt 5)
        // „Getaggt (beim Hochladen)“ entfällt (Michael Karbe, Feedback 20261001 - 2, Punkt 2): nur „Getaggt“ / „Nicht getaggt“
        if (tg.status === 'fertig' || d.getaggt === true) return t('Getaggt');
        if (d.getaggt === false) return t('Nicht getaggt');
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
        // Einzahl bei 1 (Prüfung Barrierefreiheit 30.09.2026, Punkt 11: „1 Bilder“)
        teile.push((s.ueberschriften || 0) === 1 ? t('1 Überschrift') : t('{n} Überschriften', { n: s.ueberschriften || 0 }));
        teile.push((s.listen || 0) === 1 ? t('1 Liste') : t('{n} Listen', { n: s.listen || 0 }));
        teile.push((s.tabellen || 0) === 1 ? t('1 Tabelle') : t('{n} Tabellen', { n: s.tabellen || 0 }));
        teile.push((s.bilder || 0) === 1 ? t('1 Bild') : t('{n} Bilder', { n: s.bilder || 0 }));
        return t('{n} Elemente', { n: s.elemente }) + ' (' + teile.join(', ') + ')';
    }

    // ─── Karte je Dokument ───
    function befundZeile(e) {
        if (!e.satz) return esc(e.text || '');
        const seiten = e.seiten || [];
        const seitenText = seiten.length === 1 ? ' ' + t('(Seite {s})', { s: seiten[0] }) : (seiten.length ? ' ' + t('(Seiten {s})', { s: seiten.join(', ') }) : '');
        const regeln = e.regeln || [];
        const ref = regeln.length ? ' ' + (regeln.length === 1 ? t('(Regel {r})', { r: regeln[0] }) : t('(Regeln {r})', { r: regeln.join(', ') })) : '';
        return (e.lang ? '<span lang="' + esc(e.lang) + '">' + esc(e.satz) + '</span>' : esc(e.satz)) + (e.mal ? ' ' + esc(e.mal) : '') + esc(seitenText) + esc(ref);
    }
    function berichtHtml(d, project) {
        const tg = d.tagging || {};
        const b = tg.bericht || {};
        if (!tg.status || tg.status === 'laeuft') return '';
        const zeilen = [];
        if (tg.status === 'fehler') {
            zeilen.push('<li>' + t('Fehler: {grund}', { grund: esc(b.fehler || t('unbekannt')) }) + (b.zeit ? ' (' + esc(b.zeit) + ')' : '') + '</li>');
        } else {
            // Ohne Dokumentinfos (Sprache, Struktur, Titel stehen schon oben an der Karte — Michael Karbe, Feedback
            // 24.09.2026, Punkt 9) und ohne Testmodus-Hinweis (Punkt 7).
            // Dezimaltrennzeichen der Oberflaechensprache (Pruefung 3, N4: „0.2 Sekunden“ las VoiceOver als „null Punkt zwei“)
            const dauer = b.dauer_s != null ? new Intl.NumberFormat(window.LANG || 'de', { maximumFractionDigits: 1 }).format(Number(b.dauer_s)) : '?';
            zeilen.push('<li>' + (Number(b.dauer_s) === 1 ? t('Getaggt am {zeit} in 1 Sekunde.', { zeit: esc(b.zeit || '') })
                : t('Getaggt am {zeit} in {s} Sekunden.', { zeit: esc(b.zeit || ''), s: esc(dauer) })) + '</li>');
            if (b.bilder && b.bilder.uebernommen) zeilen.push('<li>' + t('{u} Alt-Texte aus dem vorherigen Stand übernommen.', { u: esc(b.bilder.uebernommen) }) + '</li>');
            (b.hinweise || []).forEach(h => zeilen.push('<li>' + esc(h) + '</li>'));
        }
        let pruef = '';
        const v = b.verapdf;
        if (v) {
            // Nur Fehler, keine „In Ordnung“-Zeilen (Mail 22.09.2026, Punkt 6), ohne das Wort „Hinweis“ und je verletztem
            // Pruefpunkt eine Zeile (Feedback 24.09.2026, Punkte 10-12; aeltere Berichte ohne "einzeln": ein Absatz je Bereich)
            const befunde = [];
            // wie die Problemzeilen der Prüfung (Pruefung 3, N3): englische veraPDF-Sätze mit lang="en", Seiten und Regelnummer
            (v.punkte || []).filter(p => p.status === 'befund').forEach(p => (p.einzeln || [{ text: p.text }]).forEach(e => befunde.push(esc(p.bereich) + ': ' + befundZeile(e))));
            // Stand direkt nach dem Taggen (Audit 30.09.2026, MITTEL 4): nicht die fertige Datei — die prüft die
            // Barrierefreiheitsprüfung (mit Alt-Texten, Quickinfos und Titel). Vorher hieß beides „PDF/UA-Prüfung“.
            pruef = '<h4>' + t('PDF/UA-Prüfung direkt nach dem Taggen') + '</h4>'
                + '<p>' + t('Das ist der Stand direkt nach dem Taggen, vor Alt-Texten und Quickinfos. Die fertige Datei prüfst du in der {pruefung}.', { pruefung: project ? ansichtLink(project, 'abschluss') : t('Barrierefreiheitsprüfung') }) + '</p>'
                + (befunde.length ? '<ul>' + befunde.map(x => '<li>' + x + '</li>').join('') + '</ul>' : '<p>' + t('Bestanden.') + '</p>');
        } else if (tg.status === 'fertig') {
            pruef = '<p>' + t('Die PDF/UA-Prüfung war nicht möglich (Prüfdienst nicht erreichbar).') + '</p>';
        }
        return '<details class="page-text-details dok-bericht" data-doc="' + d.id + '"' + (offeneBerichte.has(d.id) ? ' open' : '') + '>'
            + '<summary>' + t('Bericht lesen') + '<span class="visually-hidden"> ' + t('– Dokument „{name}“', { name: esc(docDisplayName(d)) }) + '</span></summary>'
            + '<div class="page-text-content" role="region" aria-label="' + t('Bericht zum Tagging') + '" tabindex="0"><ul>' + zeilen.join('') + '</ul>' + pruef + '</div></details>';
    }

    // ─── Hoerprobe (22.09.2026): Zeilen in Lesereihenfolge aus den Tags, erst beim Aufklappen geladen
    // (eigenes PDFix-Skript pdfix_scripts/Struktur_Export.py, Modul pdf_struktur.py). Die Strukturansicht
    // ist eine eigene Seite (/struktur/<projekt>/<dokument>), damit Ueberschriftensprünge durch die PDF gehen.
    // Hörprobe: seit 29.09.2026 als Dialog über den Knopf „Hörprobe“ (hoerprobeOeffnen), nicht mehr als Klappe in der Karte.

    // ─── Automatische Pruefung (Schritt 5, erste Fassung, 22.09.2026): ein KI-Modell vergleicht je Seite
    // Seitenbild und Tags und meldet nur Befunde mit Beleg und Sicherheit. Aendert nichts an der Datei.
    function sicherheitText(s) {
        return s === 'hoch' ? t('Sicherheit hoch') : (s === 'mittel' ? t('Sicherheit mittel') : t('Sicherheit niedrig'));
    }
    function artText(a) {
        const m = { rolle: t('Rolle'), ebene: t('Ebene'), reihenfolge: t('Reihenfolge'), tabelle: t('Tabelle'), grafik: t('Grafik'), fehlt: t('Fehlt'), sprache: t('Sprache'), sonstiges: t('Sonstiges') };
        return m[a] || a;
    }
    // ─── Korrektur (Stufe 2, 22.09.2026): nur Befunde mit Doppelbeleg (Modell + Messung), kostenlos, mit Rückweg;
    // die Nachprüfung ist ein eigener, bezahlter Knopf (Steve: der Kunde wählt).
    function korrekturHtml(project, d, pr) {
        const ko = pr.korrektur || {};
        const kb = ko.bericht || {};
        const busy = (d.tagging && d.tagging.laeuft) || pr.laeuft || ko.laeuft || !!(project.kette && project.kette.laeuft);
        const vh = t('– Dokument „{name}“', { name: esc(docDisplayName(d)) });
        let s = '';
        if (ko.laeuft) s += '<p><output class="dok-status">' + t('Korrektur läuft …') + '</output></p>';
        if (pr.status === 'fertig' && ko.verfuegbar && ko.auto_befunde > 0 && !ko.korrigiert_am && !busy) {
            s += '<p>' + t('{n} Befunde tragen den Doppelbeleg: Modell und Messung zeigen dieselbe Richtung. Nur diese werden automatisch korrigiert, alle anderen bleiben Hinweise. Vor der Korrektur wird eine Sicherung angelegt.', { n: ko.auto_befunde }) + '</p>'
                + '<p><button type="button" class="btn btn-primary" id="dok_korr_' + d.id + '" onclick="Dokument.korrekturStarten(' + project.id + ', ' + d.id + ', false)">' + ico('sparkle') + t('{n} Befunde korrigieren', { n: ko.auto_befunde }) + '<span class="visually-hidden"> ' + vh + ', ' + t('kostenlos') + '</span></button> '
                + '<button type="button" class="btn btn-secondary" id="dok_korr2_' + d.id + '" onclick="Dokument.korrekturStarten(' + project.id + ', ' + d.id + ', true)">' + t('Korrigieren und erneut prüfen') + '<span class="visually-hidden"> ' + vh + ', ' + t('{c} Credits', { c: pr.preis || 0 }) + '</span></button></p>';
        }
        if (ko.korrigiert_am) s += '<p class="feld-hinweis">' + t('Dieser Prüfbericht stammt von vor der Korrektur ({zeit}). „Erneut prüfen“ zeigt den neuen Stand.', { zeit: esc(ko.korrigiert_am) }) + '</p>';
        if (kb.fehler) s += '<p class="feld-hinweis">' + t('Korrektur fehlgeschlagen: {grund}', { grund: esc(kb.fehler) }) + (kb.zeit ? ' (' + esc(kb.zeit) + ')' : '') + '</p>';
        if (kb.zeit && !kb.fehler) {
            s += '<h5>' + t('Korrektur vom {zeit}: {n} Änderungen', { zeit: esc(kb.zeit), n: kb.anzahl || 0 }) + '</h5><ul class="dok-befunde">'
                + (kb.angewendet || []).map(a => '<li>' + t('Seite {n}', { n: a.seite }) + ': ' + esc(a.typ_vorher || '?') + ' → ' + esc(a.typ_nachher || '?') + (a.text ? ' „' + esc(a.text) + '“' : '')
                    + (a.status !== 'angewendet' ? ' <em>' + t('nicht gefunden') + '</em>' : '') + (a.begruendung ? ' <span class="dok-messung">' + esc(a.begruendung) + '</span>' : '') + '</li>').join('') + '</ul>';
            if (kb.verapdf) s += '<p>' + t('PDF/UA-Prüfung nach der Korrektur: {s}', { s: esc(kb.verapdf.zusammenfassung || '') }) + '</p>';
            if (ko.sicherung && !busy) s += '<p><button type="button" class="btn btn-secondary" id="dok_korr_undo_' + d.id + '" onclick="Dokument.korrekturRueckgaengig(' + project.id + ', ' + d.id + ')">' + t('Korrektur rückgängig machen') + '<span class="visually-hidden"> ' + vh + '</span></button></p>';
        }
        return s;
    }
    async function korrekturStarten(projectId, docId, erneut) {
        const k1 = document.getElementById('dok_korr_' + docId);
        const k2 = document.getElementById('dok_korr2_' + docId);
        if (k1) k1.disabled = true;
        if (k2) k2.disabled = true;
        try {
            const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/korrektur', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ erneut_pruefen: !!erneut }) });
            const j = await r.json().catch(() => ({}));
            if (r.status === 402 && j.detail && typeof zeigeCreditsMeldung === 'function') { zeigeCreditsMeldung(j.detail); if (k1) k1.disabled = false; if (k2) k2.disabled = false; return; }
            if (!r.ok) {
                announce(t('Die Korrektur konnte nicht gestartet werden: {grund}', { grund: (j.detail && (j.detail.text || j.detail)) || t('unbekannter Fehler') }));
                if (k1) k1.disabled = false; if (k2) k2.disabled = false;
                return;
            }
            offenePruefungen.add(docId);
            await neuLaden(projectId);
            announce(erneut ? t('Korrektur gestartet, danach folgt die Prüfung.') : t('Korrektur gestartet.'));
        } catch (e) {
            announce(t('Die Korrektur konnte nicht gestartet werden: {grund}', { grund: String(e) }));
            if (k1) k1.disabled = false; if (k2) k2.disabled = false;
        }
    }
    async function korrekturRueckgaengig(projectId, docId) {
        const k = document.getElementById('dok_korr_undo_' + docId);
        if (k) k.disabled = true;
        try {
            const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/korrektur/rueckgaengig', { method: 'POST', credentials: 'same-origin' });
            const j = await r.json().catch(() => ({}));
            if (!r.ok) { announce(t('Rückgängig nicht möglich: {grund}', { grund: (j.detail && (j.detail.text || j.detail)) || t('unbekannter Fehler') })); if (k) k.disabled = false; return; }
            offenePruefungen.add(docId);
            await neuLaden(projectId);
            // Meldung in der Statuszeile des KI-Blocks — die gibt es in jeder Ansicht, die den Block zeigt
            const out = document.getElementById('dok_pruef_status_' + docId);
            const text = t('Die Korrektur wurde rückgängig gemacht. Der Prüfbericht gilt wieder.');
            if (out) { out.textContent = text; out.focus(); } else { zeigeMeldung(text); }
        } catch (e) {
            announce(t('Rückgängig nicht möglich: {grund}', { grund: String(e) }));
            if (k) k.disabled = false;
        }
    }
    function korrAbschlussText(d) {
        const pr = (d.tagging && d.tagging.pruefung) || {};
        const kb = (pr.korrektur && pr.korrektur.bericht) || {};
        const name = docDisplayName(d);
        if (kb.fehler) return t('Die Korrektur von „{name}“ ist fehlgeschlagen: {grund}', { name: name, grund: kb.fehler });
        return t('Korrektur von „{name}“ fertig: {n} Änderungen.', { name: name, n: kb.anzahl || 0 });
    }

    function pruefBerichtHtml(pr) {
        const b = pr.bericht || {};
        if (pr.status === 'fehler') return '<p class="feld-hinweis">' + t('Fehler: {grund}', { grund: esc(b.fehler || t('unbekannt')) }) + '</p>';
        if (pr.status !== 'fertig') return '';
        const befunde = b.befunde || [];
        const anz = b.anzahl || {};
        let s = '<p>' + (befunde.length
            ? t('{n} Befunde ({h} hoch, {m} mittel, {l} niedrig), {s} Seiten geprüft am {zeit}.', { n: befunde.length, h: anz.hoch || 0, m: anz.mittel || 0, l: anz.niedrig || 0, s: b.seiten_geprueft || 0, zeit: esc(b.zeit || '') })
            : t('Keine Befunde: Tags und Seitenbild passen zusammen ({s} Seiten geprüft am {zeit}).', { s: b.seiten_geprueft || 0, zeit: esc(b.zeit || '') })) + '</p>';
        const eb = pr.einheitsbericht;
        if (eb && eb.anzahl) {
            // EINHEITSBERICHT (23.09.2026): PDF/UA-Befunde (mit Seiten) und KI-Befunde in EINER Liste, nur Probleme
            s += '<p>' + t('Alle Befunde in einer Liste, technische Prüfung (PDF/UA) und KI-Prüfung, nach Seiten sortiert: {n}.', { n: eb.anzahl }) + '</p>';
            s += '<ol class="dok-befunde">' + eb.eintraege.map(e => '<li>'
                + '<strong>' + (e.seiten.length ? (e.seiten.length > 1 ? t('Seiten {n}', { n: e.seiten.join(', ') }) : t('Seite {n}', { n: e.seiten[0] })) : t('Dokument'))
                + ' – ' + (e.quelle === 'pdfua' ? t('PDF/UA-Prüfung') : t('KI-Prüfung')) + (e.element ? ', ' + esc(e.element) : '') + (e.quelle === 'pdfua' && e.bereich ? ', ' + esc(e.bereich) : '') + ':</strong> '
                + esc(e.text)
                + (e.vorschlag ? ' ' + t('Vorschlag: {v}.', { v: esc(e.vorschlag) }) : '')
                + (e.sicherheit ? ' <span class="badge ' + (e.sicherheit === 'hoch' ? 'badge-ok' : (e.sicherheit === 'mittel' ? 'badge-warn' : 'badge-muted')) + '">' + sicherheitText(e.sicherheit) + '</span>' : '')
                + (e.auto ? ' <span class="badge badge-ok">' + t('Automatisch korrigierbar') + '</span>' : '')
                + '</li>').join('') + '</ol>';
            s += '<p><a class="btn btn-secondary" href="/api/projects/' + pr.projectId + '/documents/' + pr.docId + '/pruefung/befunde.csv" download>' + t('Befunde als CSV herunterladen') + '</a></p>';
        } else if (befunde.length) {
            s += '<ol class="dok-befunde">' + befunde.map(f => '<li>'
                + '<strong>' + t('Seite {n}', { n: f.seite }) + (f.typ ? ', ' + esc(f.typ) : '') + (f.text ? ' „' + esc(f.text) + '“' : '') + ':</strong> '
                + esc(f.befund)
                + (f.vorschlag ? ' ' + t('Vorschlag: {v}.', { v: esc(f.vorschlag) }) : '')
                + (f.beleg ? ' ' + t('Beleg: {b}', { b: esc(f.beleg) }) : '')
                + ' <span class="badge ' + (f.sicherheit === 'hoch' ? 'badge-ok' : (f.sicherheit === 'mittel' ? 'badge-warn' : 'badge-muted')) + '">' + sicherheitText(f.sicherheit) + '</span>'
                + (f.auto ? ' <span class="badge badge-ok">' + t('Automatisch korrigierbar') + '</span>' : '')
                + ' <span class="visually-hidden">' + artText(f.art) + '</span>'
                + (f.messung ? ' <span class="dok-messung">' + t('Messung: {m}', { m: esc(f.messung) }) + '</span>' : '')
                + (f.auto && f.doppelbeleg ? ' <span class="dok-messung">' + t('Doppelbeleg: {b}', { b: esc(f.doppelbeleg) }) + '</span>' : '')
                + (f.hinweis ? ' <em>' + esc(f.hinweis) + '</em>' : '')
                + '</li>').join('') + '</ol>';
        }
        if ((b.hinweise || []).length) s += '<ul>' + b.hinweise.map(h => '<li>' + esc(h) + '</li>').join('') + '</ul>';
        s += '<p class="feld-hinweis">' + t('Die Prüfung ändert nichts an der Datei. Sie ersetzt keinen Test mit einem echten Screenreader.') + '</p>';
        return s;
    }
    // kompakt (Station „Prüfung“): ohne Befundliste — die Befunde stehen dort schon in der Liste der Problemstellen.
    function pruefungHtml(project, d, kompakt) {
        if ((!ZEIGE_KI_PRUEFUNG && !kompakt) || d.getaggt !== true) return '';
        const tg = d.tagging || {};
        const pr = tg.pruefung || {};
        const busy = tg.laeuft || pr.laeuft || !!(project.kette && project.kette.laeuft);
        const vh = t('– Dokument „{name}“', { name: esc(docDisplayName(d)) });
        const knopf = pr.status === 'fertig' ? t('KI-Prüfung erneut starten') : t('KI-Prüfung starten');
        const offen = offenePruefungen.has(d.id) || pr.laeuft;
        // Station „Prüfung“ (kompakt, Michael Karbe, Feedback 24.09.2026 - 3, Punkt 13): ein Abschnitt mit Knopf, keine Klappe
        const auf = kompakt
            ? '<section class="ab-ki" aria-labelledby="ab_ki_heading_' + d.id + '"><h4 id="ab_ki_heading_' + d.id + '">' + t('KI-basierte Prüfung (experimentell)') + '</h4><div>'
            : '<details class="page-text-details dok-pruefung" data-doc="' + d.id + '"' + (offen ? ' open' : '') + '>'
              // Name „KI-basierte Prüfung“ (Michael Karbe, Mail 22.09.2026, Punkt 11)
              + '<summary>' + t('KI-basierte Prüfung (experimentell)') + (pr.status === 'fertig' && pr.bericht && pr.bericht.befunde ? ' (' + t('{n} Befunde', { n: pr.bericht.befunde.length }) + ')' : '') + '</summary>'
              + '<div class="page-text-content" role="region" aria-label="' + t('KI-basierte Prüfung') + '" tabindex="0">';
        const zu = kompakt ? '</div></section>' : '</div></details>';
        return auf
            + '<p><strong>' + t('Experimentell:') + '</strong> ' + t('Wir arbeiten noch an dieser Prüfung. Die Befunde können unvollständig oder falsch sein.') + '</p>'
            + '<p>' + t('Ein KI-Modell vergleicht je Seite das Seitenbild mit den Tags und meldet nur, was es sicher belegen kann: Überschriften als Listenpunkte, falsche Ebenen, Tabellen ohne Kopfzeile, Alt-Texte, die nicht zum Bild passen, sichtbarer Text ohne Tag.') + '</p>'
            + (!busy && pr.seiten ? '<p><button type="button" class="btn btn-secondary" id="dok_pruef_' + d.id + '" onclick="Dokument.pruefungStarten(' + project.id + ', ' + d.id + ')">' + ico('sparkle') + knopf + '<span class="visually-hidden"> ' + vh + ', ' + t('{n} Seiten, {c} Credits', { n: pr.seiten, c: pr.preis || 0 }) + '</span></button></p>' : '')
            + '<output id="dok_pruef_status_' + d.id + '" class="dok-status" style="display:block;" tabindex="-1">' + (pr.laeuft ? t('Prüfung läuft … Seite {a} von {b}.', { a: pr.seite || 0, b: pr.seiten || 0 }) : '') + '</output>'
            + (kompakt ? pruefKurzHtml(pr) : pruefBerichtHtml(Object.assign({ projectId: project.id, docId: d.id }, pr)))
            + (ZEIGE_KORREKTUR ? korrekturHtml(project, d, pr) : '')   // eigener Schalter funktionen.KORREKTUR (30.09.2026)
            + zu;
    }
    function pruefKurzHtml(pr) {
        const b = pr.bericht || {};
        if (pr.status === 'fehler') return '<p class="feld-hinweis">' + t('Fehler: {grund}', { grund: esc(b.fehler || t('unbekannt')) }) + '</p>';
        if (pr.status !== 'fertig') return '';
        const n = (b.befunde || []).length;
        return '<p>' + (n
            ? t('Letzte Prüfung am {zeit}: {n} Befunde. Sie erscheinen in der Liste der Problemstellen der Prüfdatei.', { zeit: esc(b.zeit || ''), n: n })
            : t('Letzte Prüfung am {zeit}: keine Befunde.', { zeit: esc(b.zeit || '') })) + '</p>';
    }
    async function pruefungStarten(projectId, docId) {
        const knopf = document.getElementById('dok_pruef_' + docId);
        if (knopf) knopf.disabled = true;
        try {
            const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/pruefung', { method: 'POST', credentials: 'same-origin' });
            const j = await r.json().catch(() => ({}));
            if (r.status === 402 && j.detail && typeof zeigeCreditsMeldung === 'function') { zeigeCreditsMeldung(j.detail); if (knopf) knopf.disabled = false; return; }
            if (!r.ok) {
                const grund = (j.detail && (j.detail.text || j.detail)) || t('unbekannter Fehler');
                announce(t('Die Prüfung konnte nicht gestartet werden: {grund}', { grund: grund }));
                if (knopf) knopf.disabled = false;
                return;
            }
            offenePruefungen.add(docId);
            await neuLaden(projectId);
            const out = document.getElementById('dok_pruef_status_' + docId);
            if (out) out.focus();
            announce(t('Prüfung gestartet, {n} Seiten.', { n: j.seiten || 0 }));
        } catch (e) {
            announce(t('Die Prüfung konnte nicht gestartet werden: {grund}', { grund: String(e) }));
            if (knopf) knopf.disabled = false;
        }
    }
    function pruefAbschlussText(d) {
        const pr = (d.tagging && d.tagging.pruefung) || {};
        const b = pr.bericht || {};
        const name = docDisplayName(d);
        if (pr.status === 'fehler') return t('Die Prüfung von „{name}“ ist fehlgeschlagen: {grund}', { name: name, grund: b.fehler || t('unbekannter Fehler') });
        const n = (b.befunde || []).length;
        return n ? t('Prüfung von „{name}“ fertig: {n} Befunde.', { name: name, n: n }) : t('Prüfung von „{name}“ fertig: keine Befunde.', { name: name });
    }

    // GESAMTURTEIL (23.09.2026, Steve): ein Satz je Dokument, was zu tun ist — statt veraPDF und KI-Befunde selbst zu deuten.
    function urteilHtml(project, d, tg, busy) {
        const u = tg.urteil;
        if (!ZEIGE_URTEIL || !u || u.stufe === 'laeuft') return '';
        const tech = u.technisch === true ? t('technisch in Ordnung (PDF/UA)') : (u.technisch === false ? t('technisch mit Befunden (PDF/UA)') : t('technische Prüfung fehlt'));
        let text = '', cls = 'badge-muted';
        if (u.stufe === 'ungetaggt') { text = t('Keine Struktur: Die PDF muss barrierefrei gemacht werden.'); cls = 'badge-warn'; }
        else if (u.stufe === 'neu_taggen') { text = t('Struktur unbrauchbar (kaum Elemente oder keine Überschriften): Neu taggen empfohlen.'); cls = 'badge-warn'; }
        else if (u.stufe === 'unvollstaendig') { text = t('Struktur unvollständig: {n} von {g} Textzeilen haben kein Element. Bitte die Hörprobe prüfen.', { n: u.zeilen_ohne, g: u.zeilen_gesamt }); cls = 'badge-warn'; }
        else if (u.stufe === 'in_ordnung') { text = t('In Ordnung: Struktur geprüft, nichts zu tun. Export möglich.'); cls = 'badge-ok'; }
        else if (!ZEIGE_KI_PRUEFUNG && (u.stufe === 'verbesserungen' || u.stufe === 'pruefung_empfohlen')) {
            // KI-Pruefung ausgeblendet: nur das technische Ergebnis, keine Aufforderung zu einem Knopf, den es nicht gibt
            text = tech.charAt(0).toUpperCase() + tech.slice(1) + '.'; cls = u.technisch === false ? 'badge-warn' : 'badge-muted';
        }
        else if (u.stufe === 'verbesserungen') { text = (u.ki_hoch != null ? t('Verbesserungen möglich: {n} sichere Befunde, {tech}.', { n: u.ki_hoch, tech: tech }) : t('Verbesserungen möglich: {tech}. KI-Prüfung starten.', { tech: tech })); cls = 'badge-warn'; }
        else if (u.stufe === 'pruefung_empfohlen') { text = t('{tech}. Die KI-Prüfung fehlt noch, sie zeigt, ob die Struktur zum Seitenbild passt.', { tech: tech.charAt(0).toUpperCase() + tech.slice(1) }); cls = 'badge-muted'; }
        return '<p class="dok-urteil" id="dok_urteil_' + d.id + '"><span class="badge ' + cls + '">' + t('Urteil') + '</span> ' + text + '</p>';
    }

    // Dokumentinfo je Zeile „Bezeichnung: Wert“ (Michael Karbe, Mail 22.09.2026, Punkte 2 und 4): eine Liste
    // ohne Aufzaehlungszeichen in derselben Schrift und Groesse wie der Bericht darunter.
    // Link auf eine andere Ansicht des Projekts (wechselt ohne Neuladen wie die Ansichts-Knöpfe; Strg/Cmd öffnet einen Tab)
    function ansichtLink(project, ziel) {
        const namen = (typeof ansichtNamen === 'function') ? ansichtNamen() : {};
        return '<a href="/app?projekt=' + encodeURIComponent(project.id) + '&amp;ansicht=' + ziel + '" onclick="return ansichtKlick(event, ' + project.id + ', \'' + ziel + '\')">' + esc(namen[ziel] || ziel) + '</a>';
    }
    function metaZeile(bez, wert, id) {
        return '<li>' + bez + ': <span' + (id ? ' id="' + id + '"' : '') + '>' + wert + '</span></li>';
    }
    // Jede Datei ist eine aufklappbare Karte (Michael Karbe, PS 24.09.2026: „Wenn man mehrere PDF hat, dann muss
    // man immer scrollen“). Natives <details>, die Ueberschrift ist der Schalter (wie die Dokument-Klappen der
    // Alt-Text-Ansicht). Ein einzelnes Dokument ist offen; bei mehreren sind alle zu, ausser der Nutzer hat eine
    // geoeffnet oder dort laeuft gerade etwas.
    function karteOffen(d, anzahl) {
        const tg = d.tagging || {};
        if (tg.laeuft || (tg.pruefung && tg.pruefung.laeuft)) return true;
        if (geschlosseneDokumente.has(d.id)) return false;
        return anzahl <= 1 || offeneDokumente.has(d.id);
    }
    // ─── Karte eines WORD-Dokuments (30.09.2026): Aufbau wie die PDF-Karte in „Dokument“ — Überschrift als Schalter, darunter
    // die Dokumentinfos je Zeile „Bezeichnung: Wert“, eine Linie, darunter Hörprobe, Herunterladen, Umbenennen, Löschen und
    // sonst nichts (Michael Karbe, Feedback 20260928 - 2, Punkte 1 und 2). Kein Stand-Abzeichen: Word kennt kein Tagging.
    // Kein Vorschaubild: für Word gibt es ohne Umwandlung keine Seitenansicht.
    function wordKarteHtml(project, d, pos, anzahl) {
        const name = esc(docDisplayName(d));
        const vh = t('– Dokument „{name}“', { name: name });
        const busy = project.status === 'processing' || project.status === 'extracting';
        const info = d.info || {};
        const b = d.bilder || {};
        const bilderText = (b.gesamt || 0)
            ? (b.gesamt === 1 ? t('1 Bild, {m} mit Alt-Text', { m: b.mit_text || 0 }) : t('{n} Bilder, {m} mit Alt-Text', { n: b.gesamt, m: b.mit_text || 0 }))
              + (b.dekorativ ? t(', {n} als dekorativ gekennzeichnet', { n: b.dekorativ }) : '')
            : t('keine Bilder gefunden');
        const meta = info.lesbar === false
            ? '<li>' + t('Die Dokumentinfos konnten nicht aus der Word-Datei gelesen werden.') + '</li>' + metaZeile(t('Bilder'), esc(bilderText))
            // Reihenfolge wie bei PDF (Titel, Anwendung, Seiten, Sprache), danach der Aufbau; Seiten nur, wenn Word sie belegt
            : metaZeile(t('Titel'), esc(info.titel || t('kein Titel')))
              + metaZeile(t('Anwendung'), esc(info.anwendung || t('nicht angegeben')))
              + (info.seiten ? metaZeile(t('Seiten'), esc(info.seiten)) : '')
              // „Englisch (en-US)“ statt nur des Kürzels, in der Oberflächensprache (Steve 30.09.2026; app.html spracheAnzeige)
              + metaZeile(t('Sprache'), esc(sprachText(info.sprache) || t('nicht gesetzt')))
              + metaZeile(t('Überschriften'), esc(info.ueberschriften || 0))
              + metaZeile(t('Tabellen'), esc(info.tabellen || 0))
              + metaZeile(t('Bilder'), esc(bilderText));
        const knoepfe = '<button type="button" class="btn btn-secondary" id="dok_hp_' + d.id + '" onclick="Dokument.hoerprobeOeffnen(' + project.id + ', ' + d.id + ')">' + t('Hörprobe') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            // Herunterladen mit derselben Rückfrage wie bei PDF (app.html openExportPanel), Modus 'word': Word-Datei oder barrierefreie PDF
            + (!busy ? '<button type="button" class="btn btn-secondary" id="dok_export_' + d.id + '" onclick="openExportPanel(' + project.id + ', ' + d.id + ', \'word\')">' + ico('download') + t('Herunterladen') + '<span class="visually-hidden"> ' + vh + ', ' + t('als Word-Datei oder barrierefreie PDF') + '</span></button>' : '')
            + '<button type="button" class="doc-action-btn" data-kind="worddoc" data-doc-id="' + d.id + '" data-doc-name="' + name + '" onclick="openDocRename(event)">' + ico('pencil') + t('Umbenennen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            // Art 'worddoc': der Lösch-Dialog sagt, dass auch Alt-Texte und Übersetzungen verloren gehen (A11y-Prüfung 30.09.2026)
            + '<button type="button" class="doc-action-btn doc-action-danger" data-kind="worddoc" data-doc-id="' + d.id + '" data-doc-name="' + name + '" onclick="openDocDelete(event)">' + ico('trash') + t('Löschen') + '<span class="visually-hidden"> ' + vh + '</span></button>';
        return '<section class="card dok-karte" id="dok_karte_' + d.id + '">'
            + '<details class="dok-klappe" data-doc="' + d.id + '"' + (karteOffen(d, anzahl) ? ' open' : '') + '>'
            + '<summary><h3 id="dok_heading_' + d.id + '" class="doc-heading dok-kopfzeile"><span>' + t('Dokument {n}: {name}', { n: pos, name: name }) + '</span></h3></summary>'
            + '<div class="ausgabe-karte"><div class="ausgabe-text"><ul class="dok-meta">' + meta + '</ul></div></div>'
            + '<div class="dok-werkbank"><div class="ausgabe-aktionen">' + knoepfe + '</div></div>'
            + '</details></section>';
    }

    function karteHtml(project, d, pos, anzahl) {
        if (istWord) return wordKarteHtml(project, d, pos, anzahl);
        const name = esc(docDisplayName(d));
        const tg = d.tagging || {};
        const busy = project.status === 'processing' || project.status === 'extracting' || tg.laeuft || !!(project.kette && project.kette.laeuft);
        const vh = t('– Dokument „{name}“', { name: name });
        const seiten = d.seiten || tg.seiten || 0;
        const imTagging = modus === 'tagging';
        // Kopfzeile, Vorschau und Linie wie bisher (Michael Karbe, Feedback 24.09.2026 - 2 und - 3); die Infos darunter
        // je Ansicht: „Dokument“ = Metadaten, „Tagging“ = Struktur und Bilder (Feedback 20260928 - 2, Punkt 7).
        const meta = imTagging
            ? metaZeile(t('Struktur'), esc(strukturText(d.struktur)))
              // „mit Alt-Text“ zählt seit 30.09.2026 wie Dialog und Hörprobe auch Alt-Texte aus der Datei (Punkt 11), Einzahl bei 1
              + metaZeile(t('Bilder'), (d.total_images || 0) ? (d.total_images === 1 ? t('1 Bild, {m} mit Alt-Text', { m: tg.hat_alt_texte || 0 }) : t('{n} Bilder, {m} mit Alt-Text', { n: d.total_images, m: tg.hat_alt_texte || 0 })) : t('keine Bilder gefunden'))
            // Reihenfolge nach Michael Karbe (Feedback 24.09.2026, Punkte 2, 3, 5): Titel, Anwendung, Erstellt mit, Stand,
            // PDF-Standard, dann Seiten und Sprache
            : metaZeile(t('Titel'), esc((d.struktur && d.struktur.titel) || (d.meta && d.meta.titel) || t('kein Titel')))
              + metaZeile(t('Anwendung'), esc((d.meta && d.meta.anwendung) || t('nicht angegeben')))
              + metaZeile(t('Erstellt mit'), esc((d.meta && d.meta.erstellt_mit) || t('nicht angegeben')))
              + metaZeile(t('Stand'), standText(d), 'dok_stand_' + d.id)
              + metaZeile(t('PDF-Standard'), esc(((d.meta && d.meta.standard) || []).join(', ') || t('keiner')))
              + metaZeile(t('Seiten'), esc(seiten || '?'))
              + metaZeile(t('Sprache'), esc(sprachText(d.struktur && d.struktur.lang) || t('nicht gesetzt')))
              + ((d.felder || 0) > 0 ? metaZeile(t('Formularfelder'), t('{n} Felder', { n: d.felder })) : '');
        const hoerprobeKnopf = d.getaggt === true && !busy
            ? '<button type="button" class="btn btn-secondary" id="dok_hp_' + d.id + '" onclick="Dokument.hoerprobeOeffnen(' + project.id + ', ' + d.id + ')">' + t('Hörprobe') + '<span class="visually-hidden"> ' + vh + '</span></button>'
            : '';
        let knoepfe;
        let vorKnoepfen = '';
        if (imTagging) {
            const preis = tg.preis || 0;
            // PROFESSIONELLES TAGGING bis zur Tagging-Lizenz sperrbar (09.10.2026, Steve nach Absprache mit Michael Karbe;
            // backend/funktionen.py TAGGING_PROFESSIONELL): statt des Knopfs ein ehrlicher Hinweis, kein ausgegrauter Knopf.
            const profiFrei = (tg.professionell !== undefined) ? !!tg.professionell : !!F.tagging_professionell;
            if (!profiFrei && tg.verfuegbar && seiten) {
                vorKnoepfen = '<p class="feld-hinweis dok-profi-hinweis" id="dok_profi_hinweis_' + d.id + '">'
                    + t('Das professionelle Tagging schalten wir in Kürze frei. Bis dahin kannst du dein Dokument kostenlos testweise taggen und die Testfassung herunterladen.') + '</p>';
            }
            // Schon getaggte PDF (beim Hochladen): „Neu taggen“ ersetzt die vorhandenen Tags (Michael Karbe, Feedback 20261001 - 1,
            // Punkt 1); vorher gab es hier gar keinen Knopf
            const knopfText = (tg.status === 'fertig' || tg.quelle_getaggt === true) ? t('Neu taggen') : t('Barrierefrei machen');
            knoepfe = (!busy && profiFrei && tg.verfuegbar && seiten ? '<button type="button" class="btn btn-primary" id="dok_tag_' + d.id + '" onclick="Dokument.laufOeffnen(' + d.id + ')">' + ico('sparkle') + knopfText + '<span class="visually-hidden"> ' + vh + ', ' + t('{n} Seiten, {c} Credits', { n: seiten, c: preis }) + '</span></button>' : '')
                // TESTWEISE TAGGEN (Michael Karbe, Feedback 24.09.2026 - 2, Punkt 3): kostenlos, Testmodus, das Original bleibt
                + (!busy && tg.verfuegbar && tg.test_moeglich !== false && seiten && !(tg.test && tg.test.laeuft) ? '<button type="button" class="btn btn-secondary" id="dok_test_' + d.id + '" onclick="Dokument.testOeffnen(' + d.id + ')">' + t('Testweise taggen') + '<span class="visually-hidden"> ' + vh + ', ' + t('kostenlos, im Testmodus') + '</span></button>' : '')
                + hoerprobeKnopf
                + (ZEIGE_STRUKTURANSICHT && d.getaggt === true ? '<a class="btn btn-secondary" id="dok_struktur_' + d.id + '" href="/struktur/' + project.id + '/' + d.id + '">' + t('Strukturansicht öffnen') + '<span class="visually-hidden"> ' + vh + '</span></a>' : '');
        } else {
            // „Dokument“ = Dateiverwaltung (Feedback 20260928 - 2, Punkt 1): Hörprobe, Herunterladen, Umbenennen, Löschen —
            // Herunterladen mit derselben Rückfrage wie in „Alt-Texte“ (app.html openExportPanel, Modus 'pdf')
            knoepfe = hoerprobeKnopf
                // auch OHNE Tags (Feedback 20260928 - 2, Punkt 5): dann unverändert bzw. nur mit Quickinfos, der Dialog sagt es vorher
                + (!busy && seiten ? '<button type="button" class="btn btn-secondary" id="dok_export_' + d.id + '" onclick="openExportPanel(' + project.id + ', ' + d.id + ', \'pdf\')">' + ico('download') + t('PDF herunterladen') + '<span class="visually-hidden"> ' + vh + ', ' + (d.getaggt === true ? t('mit Alt-Texten und Quickinfos, kommt in die Ablage') : ((d.quickinfos_bearbeitet || 0) > 0 ? t('ohne Tags: ohne Alt-Texte, mit bearbeiteten Quickinfos') : t('ohne Tags: unverändert und kostenlos'))) + '</span></button>' : '')
                // EXPRESS-WARENKORB (Zusatz 05.10.2026, Steve): genau dieses Dokument in den Warenkorb des Express-Service —
                // eigener Schalter in der Verwaltung (window.FUNKTIONEN.express_korb_knopf), nie im Gastzugang
                + (F.express_korb_knopf && !window.GUEST_MODE && seiten ? '<button type="button" class="doc-action-btn" id="dok_express_' + d.id + '" onclick="Dokument.inExpressKorb(' + d.id + ')">' + t('In den Express-Warenkorb') + '<span class="visually-hidden"> ' + vh + '</span></button>' : '')
                + '<button type="button" class="doc-action-btn" data-kind="doc" data-doc-id="' + d.id + '" data-doc-name="' + name + '" onclick="openDocRename(event)">' + ico('pencil') + t('Umbenennen') + '<span class="visually-hidden"> ' + vh + '</span></button>'
                + '<button type="button" class="doc-action-btn doc-action-danger" data-kind="doc" data-doc-id="' + d.id + '" data-doc-name="' + name + '" data-doc-count="' + (d.total_images || 0) + '" onclick="openDocDelete(event)">' + ico('trash') + t('Löschen') + '<span class="visually-hidden"> ' + vh + '</span></button>';
        }
        return '<section class="card dok-karte" id="dok_karte_' + d.id + '">'
            + '<details class="dok-klappe" data-doc="' + d.id + '"' + (karteOffen(d, anzahl) ? ' open' : '') + '>'
            // „, Stand:“ nur für Screenreader: sonst klang das Abzeichen wie ein Teil des Dateinamens (A11y-Review 29.09.2026)
            + '<summary><h3 id="dok_heading_' + d.id + '" class="doc-heading dok-kopfzeile"><span>' + t('Dokument {n}: {name}', { n: pos, name: name }) + '<span class="visually-hidden">, ' + t('Stand') + ':</span></span> <span class="badge ' + standKlasse(d) + '" id="dok_badge_' + d.id + '">' + standText(d) + '</span></h3></summary>'
            + '<div class="ausgabe-karte">'
            + (seiten ? '<img class="ausgabe-vorschau" src="/api/projects/' + project.id + '/documents/' + d.id + '/vorschau" alt="' + t('Vorschau der ersten Seite von {name}', { name: name }) + '" loading="lazy">' : '')
            + '<div class="ausgabe-text"><ul class="dok-meta">' + meta + '</ul>'
            + (imTagging ? urteilHtml(project, d, tg, busy) : '')
            + '</div></div>'
            + '<div class="dok-werkbank">'
            // Satz in normaler Textgröße, ohne Verweis auf einen Knopf, den es hier nicht gibt, und mit dem Weg weiter
            // (Prüfung Barrierefreiheit 30.09.2026, Punkt 5): Links zu „Alt-Texte“ und „Barrierefreiheitsprüfung“
            // Info „beim Hochladen schon getaggt …“ entfällt (Feedback 20261001 - 2, Punkt 3)
            + vorKnoepfen
            + '<div class="ausgabe-aktionen">' + knoepfe + '</div>'
            // Unter den Knöpfen in „Dokument“ nichts weiter (Feedback 20260928 - 2, Punkt 2); Ergebnis, Laufstatus, Testlauf
            // und Bericht gehören zum Tagging.
            + (imTagging
                ? ergebnisHtml(d)
                  + '<output id="dok_status_' + d.id + '" class="dok-status" style="display:block;margin-top:0.5rem;" tabindex="-1">' + (tg.laeuft ? (tg.fortschritt && tg.fortschritt.seiten ? t('Wird barrierefrei gemacht … Seite {a} von {b} zugeordnet.', { a: tg.fortschritt.seite || 0, b: tg.fortschritt.seiten }) : t('Wird barrierefrei gemacht …')) : '') + '</output>'
                  + testHtml(project, d)
                  + berichtHtml(d, project)
                  + pruefungHtml(project, d)
                // Bestätigung „In den Express-Warenkorb“: sichtbar, mit Link, Fokus darauf (keine zusätzliche Ansage)
                : (F.express_korb_knopf && !window.GUEST_MODE ? '<p class="verwaltung-bestaetigung dok-express-meldung" id="dok_express_meldung_' + d.id + '" tabindex="-1" hidden></p>' : ''))
            + '</div></details></section>';
    }

    // ─── Hörprobe als Dialog (Feedback 20260928 - 2, Punkte 1 und 6: Knopf „Hörprobe“ in „Dokument“ und „Tagging“; unter den
    // Knöpfen steht nichts mehr). Natives <dialog> (Fokusfang, Escape), Inhalt = was ein Screenreader aus den Tags bekommt.
    function hoerprobeDialogHtml() {
        // Name = Überschrift, Beschreibung = Hinweis; die Region heißt anders als der Dialog (sonst dreimal dieselbe Ansage)
        // und der Ladestand kommt als Statuszeile im Dialog (A11y-Review 29.09.2026).
        return '<dialog id="dkHoerprobeDialog" class="app-dialog" aria-labelledby="dkHpHeading" aria-describedby="dkHpHinweis">'
            + '<h2 id="dkHpHeading">' + t('Hörprobe') + '</h2>'
            // Word hat keine Tags: gelesen wird die Word-Datei mit den Alt-Texten aus InkluDocs (30.09.2026)
            + '<p class="dialog-hint" id="dkHpHinweis">' + (istWord
                ? t('In dieser Reihenfolge liest ein Screenreader dieses Word-Dokument mit den Alt-Texten aus InkluDocs vor.')
                : t('In dieser Reihenfolge liest ein Screenreader den getaggten Inhalt des Dokumentes vor.')) + '</p>'
            + '<p id="dkHpStatus" role="status" class="visually-hidden"></p>'
            + '<div class="ausgabe-hoerprobe" id="dkHpInhalt" role="region" aria-label="' + t('Vorgelesener Text') + '" tabindex="0" style="max-height:24rem;overflow:auto;"></div>'
            // „Hörprobe vorlesen“ wie bei Word (Prüfung Barrierefreiheit 30.09.2026, Punkt 1; Steve: der Name bleibt, also muss es
            // etwas zu hören geben): gemeinsame Funktion vorlesenTeile, Ansage in der Kontosprache, Inhalt in der Dokumentsprache,
            // nur Stimmen auf dem Gerät. Ohne Stimme steht die Meldung SICHTBAR in der Statuszeile darunter (einzige Ansage).
            + '<p class="ab-vorlese-status" id="dkHpVorleseStatus" role="status"></p>'
            + '<div class="dialog-actions">'
            + '<button type="button" class="btn btn-secondary tts-btn" id="dkHpVorlesen" aria-pressed="false" onclick="Dokument.hoerprobeVorlesen(this)">' + t('Hörprobe vorlesen') + '</button>'
            + '<button type="button" class="btn btn-secondary" id="dkHpZu" onclick="Dokument.hoerprobeSchliessen()">' + t('Schließen') + '</button></div>'
            + '</dialog>';
    }
    let hoerprobeDoc = null;
    let hoerprobeKnopf = null;   // Knopf, der den Dialog geöffnet hat (Fokus zurück beim Schließen)
    let hoerprobeDaten = { zeilen: [], eigene: new Set(), lang: '' };   // geladene Hörprobe für „Hörprobe vorlesen“
    // quelle 'test' (09.10.2026, Steve): Hörprobe der TESTFASSUNG aus „Testweise taggen“ (GET …/tagging/test/hoerprobe) — der
    // Knopf „Hörprobe“ in der Karte liest weiter das Dokument selbst (Original bzw. getaggte Fassung).
    async function hoerprobeOeffnen(projectId, docId, quelle) {
        const dlg = document.getElementById('dkHoerprobeDialog');
        const box = document.getElementById('dkHpInhalt');
        const kopf = document.getElementById('dkHpHeading');
        const hinweis = document.getElementById('dkHpHinweis');
        const d = ((aktuelleDaten && aktuelleDaten.documents) || []).find(x => x.id === docId);
        if (!dlg || !box) return;
        const testfassung = quelle === 'test' && !istWord;
        hoerprobeDoc = docId;
        hoerprobeKnopf = document.getElementById((testfassung ? 'dok_testhp_' : 'dok_hp_') + docId);
        const status = document.getElementById('dkHpStatus');
        if (kopf) kopf.textContent = testfassung ? t('Hörprobe der Testfassung: {name}', { name: d ? docDisplayName(d) : '' })
                                                 : t('Hörprobe: {name}', { name: d ? docDisplayName(d) : '' });
        if (hinweis && !istWord) hinweis.textContent = testfassung
            ? t('In dieser Reihenfolge liest ein Screenreader die Testfassung vor. Text unter dem Wasserzeichen kann fehlen.')
            : t('In dieser Reihenfolge liest ein Screenreader den getaggten Inhalt des Dokumentes vor.');
        if (status) status.textContent = '';
        box.innerHTML = '<p>' + t('Hörprobe wird geladen …') + '</p>';
        dlg.showModal();
        // Inhalt in der Dokumentsprache (lang am Inhalt, wie in der Barrierefreiheitsprüfung): VoiceOver liest ihn dann mit der
        // Stimme, die auch ein echter Screenreader nähme — die Ansage („Überschrift Ebene 1“) bleibt in der Oberflächensprache.
        const teStruktur = (d && d.tagging && d.tagging.test && d.tagging.test.struktur) || {};
        const roh = String((testfassung ? teStruktur.lang : '') || (d && ((d.struktur && d.struktur.lang) || (d.info && d.info.sprache))) || '').trim();
        const lang = /^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*$/.test(roh) ? roh : '';
        let eigene = new Set();
        // eigen = Zeile von InkluDocs (Sprache, Seiten, Zusammenfassung): ohne lang der Dokumentsprache (Prüfung 30.09.2026, Punkt 2)
        const zeile = (z, idx) => {
            const i = String(z).indexOf(': ');
            if (i <= 0 || eigene.has(idx)) return '<p>' + esc(z) + '</p>';
            return '<p>' + esc(z.slice(0, i)) + ': <span' + (lang ? ' lang="' + esc(lang) + '"' : '') + '>' + esc(z.slice(i + 2)) + '</span></p>';
        };
        hoerprobeDaten = { zeilen: [], eigene: new Set(), lang: lang };
        const vst = document.getElementById('dkHpVorleseStatus');
        if (vst) vst.textContent = '';
        let meldung;
        try {
            let j;
            if (istWord) {
                // Word (30.09.2026): Hörprobe der Word-Datei mit den Alt-Texten aus InkluDocs — derselbe kostenlose Weg wie
                // früher „Hörprobe und Prüfbericht“ im Herunterladen-Dialog (main._pdfua_vorschau_sync, docx_hoerprobe)
                const r = await fetch('/api/projects/' + projectId + '/export/pdfua/vorschau', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ document_id: docId }) });
                const w = await r.json().catch(() => null);
                const dok = r.ok && w ? (w.dokumente || [])[0] : null;
                j = dok ? { verfuegbar: true, hoerprobe: dok.hoerprobe || [], hoerprobe_eigene: dok.hoerprobe_eigene || [] }
                        : { verfuegbar: false, grund: (w && typeof w.detail === 'string' && w.detail) || '' };
            } else {
                const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + (testfassung ? '/tagging/test/hoerprobe' : '/struktur'), { credentials: 'same-origin' });
                j = r.ok ? await r.json() : null;
            }
            if (hoerprobeDoc !== docId || !dlg.open) return;
            if (!j || !j.verfuegbar) {
                meldung = (j && j.grund) || t('Die Hörprobe konnte nicht geladen werden.');
                box.innerHTML = '<p>' + esc(meldung) + '</p>';
            } else {
                const zeilen = j.hoerprobe || [];
                eigene = new Set(j.hoerprobe_eigene || []);
                hoerprobeDaten = { zeilen: zeilen, eigene: eigene, lang: lang };
                box.innerHTML = zeilen.map(zeile).join('') || '<p>' + t('Kein Text zum Vorlesen vorhanden.') + '</p>';
                meldung = zeilen.length === 1 ? t('Hörprobe geladen, 1 Zeile.') : t('Hörprobe geladen, {n} Zeilen.', { n: zeilen.length });
            }
        } catch (e) {
            meldung = t('Die Hörprobe konnte nicht geladen werden.');
            box.innerHTML = '<p>' + esc(meldung) + '</p>';
        }
        if (!dlg.open) return;
        if (status) status.textContent = meldung;
        if (document.activeElement !== box) box.focus();
    }
    // „Hörprobe vorlesen“ im Dialog: Ansage (Kontosprache) und Inhalt (Dokumentsprache) getrennt, Zeilen von InkluDocs ganz in
    // der Kontosprache — wie „Hörprobe vorlesen“ in der Barrierefreiheitsprüfung (abschluss.js).
    function hoerprobeVorlesen(btn) {
        if (typeof vorlesenTeile !== 'function') return;
        const teile = [];
        hoerprobeDaten.zeilen.forEach((zl, i) => {
            const k = String(zl).indexOf(': ');
            if (k <= 0 || hoerprobeDaten.eigene.has(i)) { teile.push({ text: String(zl), lang: '' }); return; }
            teile.push({ text: zl.slice(0, k) + ':', lang: '' });
            teile.push({ text: zl.slice(k + 2) + '.', lang: hoerprobeDaten.lang });
        });
        vorlesenTeile(teile, btn, t('Hörprobe vorlesen'), document.getElementById('dkHpVorleseStatus'));
    }
    function hoerprobeSchliessen() {
        if (typeof vorlesenStopp === 'function') vorlesenStopp();
        const dlg = document.getElementById('dkHoerprobeDialog');
        if (dlg && dlg.open) dlg.close();
        const btn = (hoerprobeKnopf && document.contains(hoerprobeKnopf)) ? hoerprobeKnopf
                  : (hoerprobeDoc ? document.getElementById('dok_hp_' + hoerprobeDoc) : null);
        if (btn) btn.focus();
    }

    // ─── Testweise taggen (25.09.2026): Ergebnis des letzten Testlaufs als Klappe, das Dokument bleibt unverändert
    // (tagging_api._test_sync). Seit 09.10.2026 (Steve nach Absprache mit Michael Karbe; vorher „nicht herunterladbar“,
    // Steve 25.09.) ist die Testfassung mit Wasserzeichen kostenlos herunterladbar (GET …/tagging/test/datei), und die Klappe
    // hat eine eigene „Hörprobe der Testfassung“.
    function testText(te) {
        if (te.fehler) return t('Der Testlauf ist fehlgeschlagen: {grund}', { grund: te.fehler });
        let s = t('Testlauf vom {zeit}: {struktur}.', { zeit: te.zeit || '', struktur: strukturText(te.struktur) });
        // nicht „Deine PDF ist fertig …“ (Audit 30.09.2026): die Testfassung ist ein Zwischenstand
        if (te.verapdf) s += ' ' + (te.verapdf.bestanden ? t('PDF/UA-Prüfung der Testfassung: bestanden.') : t('PDF/UA-Prüfung der Testfassung: nicht bestanden.'));
        return s;
    }
    // Meldung nach dem Testlauf (in der Karte, mit Fokus): Ergebnis und, wenn es die Datei gibt, wo der Download steht
    function testMeldung(te) {
        return testText(te) + (te.datei_verfuegbar && !te.fehler ? ' ' + t('Die Testfassung kannst du unter „Ergebnis des Testlaufs“ kostenlos herunterladen.') : '');
    }
    function testHtml(project, d) {
        const te = (d.tagging && d.tagging.test) || {};
        if (te.laeuft) return '<p class="feld-hinweis" id="dok_test_laeuft_' + d.id + '">' + t('Testlauf läuft … Das Original bleibt unverändert.') + '</p>';
        if (!te.zeit) return '';
        const vh = t('– Dokument „{name}“', { name: esc(docDisplayName(d)) });
        const datei = te.datei_verfuegbar && !te.fehler;
        return '<details class="page-text-details dok-test" data-doc="' + d.id + '" data-projekt="' + project.id + '">'
            + '<summary>' + t('Ergebnis des Testlaufs') + '</summary>'
            + '<div class="page-text-content" role="region" aria-label="' + t('Ergebnis des Testlaufs') + '" tabindex="0">'
            + '<p>' + esc(testText(te)) + '</p>'
            + '<p class="feld-hinweis">' + t('Der Testlauf zeigt, wie das Tagging mit PDFix ausfallen würde. Er kostet nichts und ändert das Dokument nicht.') + '</p>'
            // Download der Testfassung (09.10.2026): echter Link mit download, Hinweis per aria-describedby am Link; die eigene
            // Hörprobe liest die Testfassung (der Knopf „Hörprobe“ in der Karte liest das Dokument selbst)
            + (datei
                ? '<p class="feld-hinweis" id="dok_testdl_hinweis_' + d.id + '">' + t('Die Testfassung ist kostenlos und trägt ein Wasserzeichen von PDFix. Unter dem Wasserzeichen kann Text fehlen. Die fertige Fassung beim Barrierefrei-Machen hat beides nicht.')
                  + ' ' + t('Die Testfassung bleibt {n} Tage lang abrufbar.', { n: te.aufbewahrung_tage || 30 }) + '</p>'
                  + '<div class="ausgabe-aktionen">'
                  + '<a class="btn btn-secondary" id="dok_testdl_' + d.id + '" href="/api/projects/' + project.id + '/documents/' + d.id + '/tagging/test/datei" download aria-describedby="dok_testdl_hinweis_' + d.id + '">'
                  + ico('download') + t('Testfassung herunterladen (mit Wasserzeichen)') + '<span class="visually-hidden"> ' + vh + '</span></a>'
                  + '<button type="button" class="btn btn-secondary" id="dok_testhp_' + d.id + '" onclick="Dokument.hoerprobeOeffnen(' + project.id + ', ' + d.id + ', \'test\')">'
                  + t('Hörprobe der Testfassung') + '<span class="visually-hidden"> ' + vh + '</span></button>'
                  + '</div>'
                : '')
            + '</div></details>';
    }
    async function testHoerprobeLaden(el) {
        if (el.dataset.geladen) return;
        const box = el.querySelector('.dok-test-hoerprobe');
        if (!box) return;
        el.dataset.geladen = '1';
        try {
            const r = await fetch('/api/projects/' + el.dataset.projekt + '/documents/' + el.dataset.doc + '/tagging/test/hoerprobe', { credentials: 'same-origin' });
            const j = r.ok ? await r.json() : null;
            if (!j || !j.verfuegbar) { box.innerHTML = '<p>' + esc((j && j.grund) || t('Die Hörprobe konnte nicht geladen werden.')) + '</p>'; delete el.dataset.geladen; return; }
            box.innerHTML = (j.hoerprobe || []).map(z => '<p>' + esc(z) + '</p>').join('');
        } catch (e) {
            box.innerHTML = '<p>' + t('Die Hörprobe konnte nicht geladen werden.') + '</p>';
            delete el.dataset.geladen;
        }
    }
    let testStartLaeuft = false;
    async function testStarten(projectId, docId) {
        if (testStartLaeuft) return;
        testStartLaeuft = true;
        const knopf = document.getElementById('dok_test_' + docId);
        if (knopf) knopf.disabled = true;
        try {
            const r = await fetch('/api/projects/' + projectId + '/documents/' + docId + '/tagging/test', { method: 'POST', credentials: 'same-origin' });
            const j = await r.json().catch(() => ({}));
            if (!r.ok) {
                const out = document.getElementById('dok_status_' + docId);
                const grund = (j.detail && (j.detail.text || j.detail)) || t('Der Testlauf konnte nicht gestartet werden.');
                if (out) { out.textContent = typeof grund === 'string' ? grund : t('Der Testlauf konnte nicht gestartet werden.'); out.focus(); }
                if (knopf) knopf.disabled = false;
                return;
            }
            delete ergebnisMeldung[docId];
            offeneDokumente.add(docId); geschlosseneDokumente.delete(docId);
            await showProject(projectId, true);
            // Schon fertig (schneller Fehlschlag, z. B. Quelldatei fehlt)? Dann gleich das Ergebnis statt „gestartet“
            const dd = ((aktuelleDaten && aktuelleDaten.documents) || []).find(x => x.id === docId);
            const te = (dd && dd.tagging && dd.tagging.test) || {};
            if (!te.laeuft && te.zeit) {
                ergebnisMeldung[docId] = { text: testMeldung(te), fehler: !!te.fehler };
                await showProject(projectId, true);
                const ziel = document.getElementById('dok_ergebnis_text_' + docId);
                if (ziel) ziel.focus();
                return;
            }
            const out = document.getElementById('dok_status_' + docId);
            if (out) { out.textContent = t('Testlauf gestartet. Er kostet nichts; das Original bleibt unverändert.'); out.focus(); }
        } catch (e) {
            if (knopf) knopf.disabled = false;
            announce(t('Verbindungsfehler.'));
        } finally {
            testStartLaeuft = false;
        }
    }

    // Ergebnis des Laufs in der Karte (grün bei Erfolg, rot bei Fehler, immer mit Text — nicht nur Farbe).
    function ergebnisHtml(d) {
        const m = ergebnisMeldung[d.id];
        if (!m) return '';
        return '<div class="dok-ergebnis' + (m.fehler ? ' dok-ergebnis-fehler' : '') + '" id="dok_ergebnis_' + d.id + '">'
            + '<p id="dok_ergebnis_text_' + d.id + '" tabindex="-1">' + esc(m.text) + '</p>'
            + '<button type="button" class="btn btn-secondary btn-small" onclick="Dokument.ergebnisSchliessen(' + d.id + ')">' + t('Meldung schließen') + '</button></div>';
    }
    function ergebnisSchliessen(docId) {
        delete ergebnisMeldung[docId];
        const box = document.getElementById('dok_ergebnis_' + docId);
        if (box) box.remove();
        const h = document.querySelector('#dok_karte_' + docId + ' summary');
        if (h) h.focus();
    }

    // ─── Herunterladen: seit 28.09.2026 ueber den Dialog aus app.html (openExportPanel, Modus 'pdf') statt direkt
    // (Michael Karbe, Feedback 28.09.2026 - 1, Punkt 4). Meldung, Fokus und Doppelklick-Schutz macht der Dialog (doExport).

    // ─── Kopf ───
    // Projektkopf wie in allen Ansichten (app.html projektKopfHtml): Name + Dateityp-Symbol + Ansichts-Knoepfe; keine
    // Laufstatus-Anzeige oben rechts (Michael Karbe, Mail 22.09.2026, Punkt 9) — der Stand steht an jedem Dokument.
    // Das Feld „Funktionen und Einstellungen“ (Kette „Komplett barrierefrei machen“, Ablage) ist aus (ZEIGE_PROJEKT_KNOEPFE,
    // Feedback 24.09.2026, Punkt 1). Die Dialoge liegen ausserhalb der Felder (showModal).
    function kopfHtml(project, data) {
        const title = (project.name && project.name.trim()) ? project.name : project.filename;
        const docs = data.documents || [];
        const kette = project.kette || {};
        const busy = docs.some(d => d.tagging && d.tagging.laeuft) || !!kette.laeuft || project.status === 'extracting' || project.status === 'processing';
        const aktionen = !ZEIGE_PROJEKT_KNOEPFE ? '' : (docs.length && !busy ? '<button class="btn btn-primary" id="dkKetteBtn" onclick="Dokument.ketteOeffnen(' + project.id + ')">' + ico('sparkle') + t('Komplett barrierefrei machen') + '<span class="visually-hidden"> ' + t('– ganzes Projekt') + '</span></button>' : '')
            + ((data.ausgaben_anzahl || 0) > 0 ? '<a class="btn btn-secondary" id="ausgabenTab" href="/ablage?projekt=' + project.id + '">' + t('Ablage ({n})', { n: data.ausgaben_anzahl || 0 }) + '</a>' : '');
        // EXPRESS-SERVICE (05.10.2026): Weg zu den Profis — öffnet einen Auftrag mit den Dokumenten dieses Projekts.
        const express = (window.FUNKTIONEN && window.FUNKTIONEN.express && modus === 'dokument' && docs.length && !window.GUEST_MODE)
            ? '<p class="dok-express"><a href="/express?projekt=' + encodeURIComponent(project.id) + '">' + t('Vom Express-Service bearbeiten lassen') + '</a></p>'
            : '';
        return projektKopfHtml(project, modus, title, '<div class="card-info" id="projectHeadInfo" hidden></div>')
            + express
            + funktionenKarteHtml(modus === 'dokument' ? aktionen : '')
            + (modus === 'tagging' ? laufDialogHtml(project) + testDialogHtml(project) : '')
            + hoerprobeDialogHtml()
            // Herunterladen-Dialog (Feedback 28.09.2026 - 1, Punkt 4): derselbe wie in „Alt-Texte“, hier nur mit der PDF.
            + (modus === 'dokument' && docs.length && typeof exportDialogHtml === 'function' ? exportDialogHtml(project) : '')
            + (ZEIGE_PROJEKT_KNOEPFE ? ketteDialogHtml(project) : '');
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

    // „Testweise taggen“ mit Rückfrage wie „Barrierefrei machen“ (Michael Karbe, Feedback 20261001 - 2, Punkt 4). Text ist ein
    // ENTWURF, mit Michael abzustimmen. Wichtig: das Original wird nicht ersetzt, danach kann man richtig taggen.
    let testZielDoc = null;
    function testDialogHtml(project) {
        return '<dialog id="dkTestDialog" class="app-dialog" aria-labelledby="dkTestHeading" aria-describedby="dkTestText">'
            + '<h2 id="dkTestHeading">' + t('Testweise taggen') + '</h2>'
            + '<p id="dkTestText">' + t('Testweise taggen zeigt dir kostenlos, wie das Tagging-Ergebnis aussehen würde. Die Testfassung trägt ein Wasserzeichen von PDFix und lässt sich kostenlos herunterladen. Deine Original-PDF bleibt unverändert, du kannst sie danach richtig taggen.') + '</p>'
            + '<div class="dialog-actions">'
            +   '<button type="button" class="btn btn-secondary" id="dkTestCancel" onclick="Dokument.testSchliessen()">' + t('Abbrechen') + '</button>'
            +   '<button type="button" class="btn btn-primary" id="dkTestOk" onclick="Dokument.testBestaetigt(' + project.id + ')">' + t('Testweise taggen') + '</button>'
            + '</div></dialog>';
    }
    function testOeffnen(docId) {
        const dlg = document.getElementById('dkTestDialog');
        if (!dlg) return;
        testZielDoc = docId;
        dlg.showModal();
        const c = document.getElementById('dkTestCancel');
        if (c) c.focus();
    }
    function testSchliessen() {
        const dlg = document.getElementById('dkTestDialog');
        if (dlg && dlg.open) dlg.close();
        const btn = testZielDoc ? document.getElementById('dok_test_' + testZielDoc) : null;
        if (btn) btn.focus();
    }
    function testBestaetigt(projectId) {
        const dlg = document.getElementById('dkTestDialog');
        if (dlg && dlg.open) dlg.close();
        if (testZielDoc) testStarten(projectId, testZielDoc);
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
        // Dialog gekürzt auf den Preissatz (Michael Karbe, Feedback 20261001 - 1, Punkt 4); welches Dokument, steht am Knopf
        if (umfang) { umfang.textContent = ''; umfang.hidden = true; }
        // Preis je Seite aus dem Server (billing.AKTIONS_PREISE, seit 30.09.2026 20 Credits je Seite — Michael Karbe, Feedback
        // 202609230 - 1, Punkt 11); bezahlt wird nur dieser Lauf, nie das Herunterladen (Punkt 12)
        const jeSeite = tg.preis_je_seite != null ? tg.preis_je_seite : ((window.CREDIT_PREISE || {}).pdf_tagging || 0);
        const satz = t('Preis: {c} Credits ({p} Credits je Seite).', { c: tg.preis || 0, p: jeSeite })
            + ' ' + t('Das Tagging bezahlst du nur in diesem Moment; beim Herunterladen wird es nicht noch einmal berechnet.');
        if (summary) summary.textContent = satz;
        if (status) status.textContent = '';
        const neu = tg.status === 'fertig' || tg.quelle_getaggt === true;
        if (ok) { ok.disabled = !tg.erlaubt; ok.textContent = neu ? t('Neu taggen') : t('Tagging starten'); }
        const kopf = document.getElementById('dkLaufHeading');
        if (kopf) kopf.textContent = neu ? t('Neu taggen') : t('Barrierefrei machen');   // wie der Knopf (A11y-Review 29.09.2026)
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
        delete ergebnisMeldung[docId];   // die Meldung des letzten Laufs gilt nicht mehr
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
            if (out) out.focus();   // war: out.focus && … — rief focus() nie auf (Pruefbericht 25.09.2026)
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

    // Fokus über einen Neuaufbau retten (A11y-Review 29.09.2026): Element mit id, sonst das summary einer Karte.
    function fokusMerken() {
        const a = document.activeElement;
        if (!a || a === document.body) return null;
        if (a.id) return { id: a.id };
        const det = a.tagName === 'SUMMARY' ? a.closest('details[data-doc]') : null;
        return det ? { karte: det.dataset.doc, klasse: det.classList.contains('dok-klappe') ? 'dok-klappe' : '' } : null;
    }
    function fokusWiederherstellen(f) {
        if (!f) return;
        let el = f.id ? document.getElementById(f.id) : null;
        if (!el && f.karte) el = document.querySelector('details' + (f.klasse ? '.' + f.klasse : '') + '[data-doc="' + CSS.escape(String(f.karte)) + '"] > summary');
        if (el && document.activeElement !== el) el.focus();   // summary ist nativ fokussierbar — kein tabindex (sonst nicht mehr per Tab erreichbar)
    }

    // Ergebnis von veraPDF nach dem Taggen in einem Satz — ausdrücklich als Zwischenstand (Audit 30.09.2026, MITTEL 4; vorher
    // „Deine PDF ist fertig und hat die Prüfung auf PDF/UA bestanden“, obwohl Alt-Texte und Quickinfos noch fehlen).
    function taggingPruefSatz(v) {
        const n = (v.punkte || []).filter(p => p.status === 'befund').reduce((a, p) => a + ((p.einzeln || [p]).length), 0);
        // Michael Karbe, Feedback 20261001 - 2, Punkt 5 (wörtlich); ohne Abweichung der Gegensatz im selben Wortlaut
        const satz = v.bestanden ? t('Die automatische PDF/UA-Prüfung hat keine Abweichungen vom Standard gefunden.')
            : t('Die automatische PDF/UA-Prüfung hat Abweichungen vom Standard identifiziert.');
        return satz + ' ' + t('Die fertige Datei prüfst du in der Barrierefreiheitsprüfung.');
    }
    function abschlussText(d) {
        const tg = d.tagging || {};
        const b = tg.bericht || {};
        const name = docDisplayName(d);
        if (tg.status === 'fehler') return t('Das Tagging von „{name}“ ist fehlgeschlagen: {grund}', { name: name, grund: b.fehler || t('unbekannter Fehler') });
        const v = b.verapdf;
        let s = t('„{name}“ ist getaggt: {struktur}.', { name: name, struktur: strukturText(b.nachher) });
        if (v) s += ' ' + taggingPruefSatz(v);
        if (b.bilder && b.bilder.nachher) {
            // Einzahl (Pruefung 3, N4: „1 Bilder gefunden“)
            const n = b.bilder.nachher, u = b.bilder.uebernommen || 0;
            s += ' ' + (n === 1 ? t('1 Bild gefunden') : t('{n} Bilder gefunden', { n: n })) + ', '
                + (u === 1 ? t('1 Alt-Text übernommen') : t('{u} Alt-Texte übernommen', { u: u })) + '.';
        }
        return s;
    }

    async function showProject(projectId, erneut, neuerModus) {
        const lauf = typeof ansichtLaufMerken === 'function' ? ansichtLaufMerken() : 0;   // Befund 1 (05.10.2026): nur der juengste Ansichtswechsel zeichnet
        const veraltet = () => typeof ansichtNochAktuell === 'function' && !ansichtNochAktuell(lauf);
        if (neuerModus === 'dokument' || neuerModus === 'tagging') modus = neuerModus;
        projectId = Number(projectId);   // Adresse liefert Text, Knoepfe eine Zahl — ohne das ging der Klapp-Zustand verloren
        pollStoppen();
        const main = document.getElementById('main');
        const res = await fetch('/api/projects/' + projectId + '/dokument-ansicht', { credentials: 'same-origin' });
        if (veraltet()) return;
        if (res.status === 401) { window.location.href = '/login'; return; }
        if (!res.ok) { main.innerHTML = '<div class="card"><p>' + t('Projekt konnte nicht geladen werden.') + '</p></div>'; return; }
        const data = await res.json();
        if (veraltet()) return;
        const project = data.project;
        aktuelleDaten = data;
        istWord = project.project_type === 'docx';
        if (istWord) modus = 'dokument';   // Word kennt nur die Dateiverwaltung, kein Tagging
        if (zustandProjekt !== projectId) { offeneBerichte = new Set(); offenePruefungen = new Set(); offeneDokumente = new Set(); geschlosseneDokumente = new Set(); zustandProjekt = projectId; }
        const docs = data.documents || [];
        neuLaden = (pid) => showProject(pid, true);   // diese Ansicht zeichnet nach Aktionen selbst neu
        if (typeof exportKontextSetzen === 'function') exportKontextSetzen(docs, project.project_type);
        // Mehrere Dokumente, alle getaggt: alles auf einmal als ZIP (bis 25.09.2026 in der Abschlusspruefung)
        const alleGetaggt = modus === 'dokument' && docs.length > 1 && docs.every(d => !(d.tagging && d.tagging.laeuft));   // seit 29.09. auch ungetaggte
        const wordBusy = istWord && (project.status === 'processing' || project.status === 'extracting');
        const alleKnopf = (alleGetaggt && !wordBusy)
            ? '<p class="ausgabe-aktionen"><button type="button" class="btn btn-secondary" id="dkAlleBtn" onclick="openExportPanel(' + project.id + ', 0, \'' + (istWord ? 'word' : 'pdf') + '\')">' + ico('download') + t('Alle Dokumente herunterladen') + '<span class="visually-hidden"> '
              + (istWord ? t('als ZIP, als Word-Dateien oder barrierefreie PDF')
                         : (docs.every(d => d.getaggt === true) ? t('als ZIP, mit Alt-Texten und Quickinfos') : t('als ZIP; Dokumente ohne Tags bekommen keine Alt-Texte'))) + '</span></button>'
              + '</p>'
            : '';
        main.innerHTML = kopfHtml(project, data)
            + uploadBlockHtml(project)
            + ketteKarteHtml(project)
            + laufMeldungHtml()
            + '<h2 class="section-title" id="dokumenteHeading" tabindex="-1" style="margin-top:1.5rem">' + t('Dokumente ({n})', { n: docs.length }) + '</h2>'
            + (docs.length ? '' : '<p class="feld-hinweis">' + t('Noch kein Dokument hochgeladen.') + '</p>')
            + alleKnopf
            + '<div id="dokListe">' + docs.map((d, i) => karteHtml(project, d, i + 1, docs.length)).join('') + '</div>'
            + (typeof inkluagentSectionHtml === 'function' ? inkluagentSectionHtml(projectId) : '');
        // Nur echte Bedienung merken: Chrome feuert fuer jede offen gezeichnete Klappe einmal „toggle“ ohne
        // Zustandswechsel — das ist keine Nutzerentscheidung und darf eine Karte nicht dauerhaft offen halten.
        document.querySelectorAll('details.dok-klappe').forEach(el => {
            const gezeichnetOffen = el.open;
            let erstesEreignis = true;
            el.addEventListener('toggle', () => {
                const echt = !(erstesEreignis && el.open === gezeichnetOffen);
                erstesEreignis = false;
                if (!echt) return;
                const k = Number(el.dataset.doc);
                if (el.open) { offeneDokumente.add(k); geschlosseneDokumente.delete(k); }
                else { offeneDokumente.delete(k); geschlosseneDokumente.add(k); }
            });
        });
        document.querySelectorAll('details.dok-bericht').forEach(el => el.addEventListener('toggle', () => {
            const k = Number(el.dataset.doc);
            if (el.open) offeneBerichte.add(k); else offeneBerichte.delete(k);
        }));
        document.querySelectorAll('details.dok-pruefung').forEach(el => el.addEventListener('toggle', () => {
            const k = Number(el.dataset.doc);
            if (el.open) offenePruefungen.add(k); else offenePruefungen.delete(k);
        }));
        if (typeof inkluagentInit === 'function') inkluagentInit(projectId);
        setupProjectDropzone(projectId);
        const h1 = document.getElementById('projectName');
        if (h1 && !erneut) h1.focus();
        const laufende = docs.filter(d => d.tagging && d.tagging.laeuft).map(d => d.id);
        const testende = docs.filter(d => d.tagging && d.tagging.test && d.tagging.test.laeuft).map(d => d.id);
        const pruefende = docs.filter(d => d.tagging && d.tagging.pruefung && d.tagging.pruefung.laeuft).map(d => d.id);
        const korrigierende = docs.filter(d => d.tagging && d.tagging.pruefung && d.tagging.pruefung.korrektur && d.tagging.pruefung.korrektur.laeuft).map(d => d.id);
        const ketteLief = !!(project.kette && project.kette.laeuft);
        document.querySelectorAll('details.dok-test').forEach(el => el.addEventListener('toggle', () => { if (el.open) testHoerprobeLaden(el); }));
        if (laufende.length || testende.length || pruefende.length || korrigierende.length || ketteLief || project.status === 'extracting' || project.status === 'processing') {
            const tick = async () => {
                if (zustandProjekt !== projectId) return;
                if (!document.getElementById('dokListe')) { pollStoppen(); return; }   // Ansicht gewechselt
                // Offener Herunterladen-Dialog: nicht neu zeichnen (das raeumte den Dialog mitten im Export weg), spaeter weiter.
                const dlg = document.getElementById('exportPanel');
                const hp = document.getElementById('dkHoerprobeDialog');
                if ((dlg && dlg.open) || (hp && hp.open)) { pollTimer = setTimeout(tick, 2500); return; }
                try {
                    const r = await fetch('/api/projects/' + projectId + '/dokument-ansicht');
                    if (!r.ok) { pollTimer = setTimeout(tick, 2500); return; }
                    const d2 = await r.json();
                    // nach dem Warten erneut: Ansicht gewechselt? Dann nichts zeichnen (sonst holte der Poll „Dokument“ zurueck)
                    if (zustandProjekt !== projectId || !document.getElementById('dokListe')) { pollStoppen(); return; }
                    // … oder inzwischen ein Dialog geöffnet? Dann nicht wegzeichnen (Review 29.09.2026: Prüfung vor UND nach dem Abruf)
                    const dlg2 = document.getElementById('exportPanel'), hp2 = document.getElementById('dkHoerprobeDialog');
                    if ((dlg2 && dlg2.open) || (hp2 && hp2.open)) { pollTimer = setTimeout(tick, 2500); return; }
                    // Kette: waehrend des Laufs nur die Statuskarte fortschreiben (kein Neuaufbau, Fokus bleibt);
                    // am Ende einmal neu zeichnen und die Zusammenfassung melden.
                    if (ketteLief) {
                        const k2 = (d2.project && d2.project.kette) || {};
                        if (k2.laeuft) { ketteAktualisieren(k2); pollTimer = setTimeout(tick, 2500); return; }
                        await showProject(projectId, true);
                        zeigeMeldung(t('Komplett barrierefrei machen ist fertig.') + ' ' + (k2.zusammenfassung || ''));
                        return;
                    }
                    // Alles sammeln, was in diesem Takt fertig wurde — dann EINMAL neu zeichnen (Pruefbericht 25.09.2026:
                    // vorher meldete nur der erste Zweig, die anderen Ergebnisse gingen verloren).
                    const docs2 = d2.documents || [];
                    const korrFertig = docs2.filter(x => korrigierende.includes(x.id) && !(x.tagging && x.tagging.pruefung && x.tagging.pruefung.korrektur && x.tagging.pruefung.korrektur.laeuft));
                    const pruefFertig = docs2.filter(x => pruefende.includes(x.id) && !(x.tagging && x.tagging.pruefung && x.tagging.pruefung.laeuft));
                    const testFertig = docs2.filter(x => testende.includes(x.id) && !(x.tagging && x.tagging.test && x.tagging.test.laeuft));
                    const fertigGeworden = docs2.filter(x => laufende.includes(x.id) && !(x.tagging && x.tagging.laeuft));
                    const jetzt = docs2.filter(x => x.tagging && x.tagging.laeuft).map(x => x.id);
                    const statusWechsel = d2.project.status !== project.status;
                    if (korrFertig.length || pruefFertig.length || testFertig.length || fertigGeworden.length || (statusWechsel && !jetzt.length)) {
                        testFertig.forEach(x => {
                            const te = (x.tagging && x.tagging.test) || {};
                            ergebnisMeldung[x.id] = te.zeit ? { text: testMeldung(te), fehler: !!te.fehler }
                                                            : { text: t('Der Testlauf wurde abgebrochen. Bitte starte ihn erneut.'), fehler: true };
                            offeneDokumente.add(x.id); geschlosseneDokumente.delete(x.id);
                        });
                        // Tagging gewinnt vor dem Testlauf desselben Dokuments (neuerer, wichtigerer Stand)
                        fertigGeworden.forEach(x => {
                            ergebnisMeldung[x.id] = { text: abschlussText(x), fehler: !!(x.tagging && x.tagging.status === 'fehler') };
                            offeneDokumente.add(x.id); geschlosseneDokumente.delete(x.id);
                        });
                        const texte = korrFertig.map(korrAbschlussText).concat(pruefFertig.map(pruefAbschlussText));
                        const fokus = fokusMerken();
                        await showProject(projectId, true);
                        const erster = fertigGeworden.concat(testFertig)[0];
                        const ziel = erster ? document.getElementById('dok_ergebnis_text_' + erster.id) : null;
                        if (ziel) ziel.focus();
                        else if (erster && ergebnisMeldung[erster.id]) zeigeMeldung(ergebnisMeldung[erster.id].text);   // Ansicht „Dokument“
                        if (texte.length) { if (ziel) announce(texte.join(' ')); else zeigeMeldung(texte.join(' ')); }
                        else if (!erster && project.status === 'extracting' && d2.project.status !== 'extracting') announce(t('Dokument gelesen.'));
                        if (!erster) fokusWiederherstellen(fokus);   // Neuaufbau ohne eigenes Fokusziel: Position behalten (A11y-Review)
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
                    pollTimer = setTimeout(tick, 2500);
                } catch (e) { pollTimer = setTimeout(tick, 2500); }
            };
            pollTimer = setTimeout(tick, 2500);
        }
    }

    // Herunterladen ueber den Dialog aus app.html (openExportPanel); der Ansichtswechsel laeuft ueber die Ansichts-Knoepfe
    // (app.html ansichtWahlHtml).
    // Fuer die Station „Prüfung“ (abschluss.js): der KI-Block kompakt, wer danach neu zeichnet, die Abschlusstexte.
    function kiBlockHtml(project, d) { return pruefungHtml(project, d, true); }
    function setNeuLaden(fn) { neuLaden = fn; }
    function kiKlappenBinden() {
        document.querySelectorAll('details.dok-pruefung').forEach(el => el.addEventListener('toggle', () => {
            const k = Number(el.dataset.doc);
            if (el.open) offenePruefungen.add(k); else offenePruefungen.delete(k);
        }));
    }
    // ─── Express-Warenkorb (Zusatz 05.10.2026) ───
    // Legt genau dieses Dokument in den Entwurfs-Korb (derselbe Endpunkt wie die Express-Seite: nur eigene Dokumente,
    // Besitz im SQL). Ist es schon drin, sagt die Meldung das. Bestätigung: sichtbarer Satz mit Link „Zum Warenkorb“, Fokus
    // darauf; die Zahl im Navigations-Eintrag wechselt still (expressKorbAnzeigen in dashboard.js).
    async function inExpressKorb(docId) {
        const meldung = document.getElementById('dok_express_meldung_' + docId);
        if (!meldung || meldung.dataset.laeuft) return;
        meldung.dataset.laeuft = '1';
        let r = null, d = {};
        try {
            r = await fetch('/api/express/warenkorb/dokumente', { method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                body: JSON.stringify({ document_ids: [docId] }) });
            d = await r.json().catch(() => ({}));
        } catch (e) { r = null; }
        delete meldung.dataset.laeuft;
        const zahl = (n) => { try { return new Intl.NumberFormat(window.LANG || 'de').format(Number(n) || 0); } catch (e) { return String(n); } };
        let text;
        if (r && r.ok) {
            const w = d.warenkorb || {};
            const pos = (w.positionen || []).find((p) => p.document_id === docId);
            const dok = Number(w.dokumente) === 1 ? t('1 Dokument') : t('{n} Dokumente', { n: zahl(w.dokumente) });
            if (d.hinzugefuegt && pos) {
                text = t('„{name}“ liegt jetzt im Express-Warenkorb.', { name: pos.dokument_name }) + ' '
                    + t('Im Warenkorb: {dokumente}, {c} Credits.', { dokumente: dok, c: zahl(w.credits) });
            } else if (pos) {
                text = t('„{name}“ liegt schon im Express-Warenkorb.', { name: pos.dokument_name });
            } else {
                text = (d.hinweise || []).join(' ') || t('Das Dokument konnte nicht in den Express-Warenkorb gelegt werden.');
            }
            if (window.expressKorbAnzeigen) window.expressKorbAnzeigen(w.dokumente);
        } else {
            const x = d && d.detail;
            text = (typeof x === 'string' ? x : (x && x.text)) || t('Das Dokument konnte nicht in den Express-Warenkorb gelegt werden.');
        }
        meldung.textContent = text + ' ';
        const link = document.createElement('a');
        link.href = '/express/warenkorb';
        link.textContent = t('Zum Warenkorb');
        meldung.appendChild(link);
        meldung.hidden = false;
        meldung.focus();
    }

    window.Dokument = { showProject, inExpressKorb, laufOeffnen, laufSchliessen, laufStarten, testOeffnen, testSchliessen, testBestaetigt, meldungSchliessen, pollStoppen,
                        ketteOeffnen, ketteSchliessen, ketteStarten, pruefungStarten, korrekturStarten, korrekturRueckgaengig,
                        ergebnisSchliessen, hoerprobeOeffnen, hoerprobeSchliessen, kiBlockHtml, setNeuLaden, kiKlappenBinden, testStarten,
                        pruefAbschlussText, korrAbschlussText, hoerprobeVorlesen };
})();
