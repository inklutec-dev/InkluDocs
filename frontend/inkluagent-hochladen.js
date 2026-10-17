// InkluAgent — Dateien hochladen und anhängen: Feld „Dateien mitschicken“ und Büroklammer der Agentenansicht.
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

// Runde 2, Schritt 1 (Schalter funktionen.AGENT_HOCHLADEN, window.INKLUAGENT aus base_app.html): Dateien gehen mit der
// Nachricht an den Agenten (Webadressen stehen seit 12.10.2026 in der Nachricht selbst). Natives Feld mit sichtbarem Label (Datei-Auswahl); die Zuordnung zu
// Projekten macht der Server (/api/agent/hochladen), das Ergebnis steht als Liste unter dem Feld.
function inkluagentHochladenAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.hochladen);
}
function inkluagentHochladeHtml() {
    const k = window.INKLUAGENT || {};
    return ''
        + '<div class="inkluagent-hochladen">'
        +   '<label for="inkluagentDateien">' + t('Dateien mitschicken') + '</label>'
        +   '<input type="file" id="inkluagentDateien" multiple aria-describedby="inkluagentHochladenHint" '
        +     'accept=".pdf,.docx,.jpg,.jpeg,.png,.gif,.webp,.heic,.heif,.bmp,.tif,.tiff,.svg">'
        // Kein eigenes Feld fuer Webadressen (Steve 12.10.2026): eine Adresse steht einfach in der Nachricht, der Server erkennt sie
        // und der Agent fragt mit Knoepfen, was damit passieren soll (hochladen.webadressen_karten)
        +   '<p id="inkluagentHochladenHint" class="inkluagent-hint">'
        +     t('PDF, Word (.docx) und Bilder, höchstens {n} Dateien, je bis {mb} MB. Webadressen schreibst du einfach in deine Nachricht. Der InkluAgent legt alles ins passende Projekt und nennt dir die Preise, bevor etwas startet.',
                { n: k.max_dateien || 10, mb: k.max_mb || 50 })
        +   '</p>'
        +   '<ul id="inkluagentHochgeladen" class="inkluagent-hochgeladen" aria-label="' + t('Ergebnis des Hochladens') + '" hidden></ul>'
        + '</div>';
}
// Ergebnisliste unter dem Feld (nur Text, kein innerHTML). Bleibt ueber ein Neuzeichnen der Ansicht hinweg stehen
// (inkluagentInitLaden zeichnet sie fuer dasselbe Projekt wieder), bis zum naechsten Hochladen.
function inkluagentHochgeladenZeigen(projectId, ergebnisse) {
    window._inkluagentHochgeladen = { projectId: String(projectId), ergebnisse: ergebnisse || [] };
    const ul = document.getElementById('inkluagentHochgeladen');
    if (!ul) return;
    ul.textContent = '';
    (ergebnisse || []).forEach(e => {
        const li = document.createElement('li');
        li.dataset.ok = e.ok ? '1' : '0';
        const name = e.name || '';
        if (!e.ok) {
            li.textContent = t('{name}: nicht hochgeladen. {grund}', { name: name, grund: e.fehler || '' });
        } else if (e.wartet) {
            // Zuordnung (Steve 12.10.2026): es gibt schon ein aehnliches Projekt — die Datei wartet auf den Knopf unter der Antwort
            li.textContent = t('{name}: wartet auf deine Entscheidung. Die Knöpfe stehen unter der Antwort des InkluAgent.', { name: name });
        } else if (!e.neues_projekt) {
            li.textContent = t('{name}: hochgeladen in dieses Projekt.', { name: name });
        } else {
            li.textContent = (e.eigenes_projekt_weil ? t('{name}: passt nicht in dieses Projekt, darum hochgeladen in das neue Projekt „{projekt}“.', { name: name, projekt: e.projekt || '' })
                                                     : t('{name}: hochgeladen in das neue Projekt „{projekt}“.', { name: name, projekt: e.projekt || '' })) + ' ';
            const a = document.createElement('a');
            a.href = '/app?projekt=' + encodeURIComponent(e.project_id);
            a.textContent = t('Projekt „{projekt}“ öffnen', { projekt: e.projekt || '' });
            li.appendChild(a);
        }
        ul.appendChild(li);
    });
    ul.hidden = !ul.children.length;
}
// Dateien hochladen. Ergebnis: das Server-Ergebnis oder null (Fehler steht dann in der Statuszeile). Webadressen stehen seit 12.10.2026
// in der Nachricht selbst — das Feld dafuer und sein Parameter sind weg (Pruefung 13.10.2026, HK2).
// kennung (Runde 2, K12): hochladen_id des Stapels (inkluagentHochladenKennung) — derselbe Stapel noch einmal liefert beim Server binnen
// 15 Minuten das fruehere Ergebnis, statt doppelt hochzuladen. Ein Server ohne dieses Feld uebergeht es.
async function inkluagentHochladen(projectId, dateien, kennung) {
    const k = window.INKLUAGENT || {};
    const max = k.max_dateien || 10;
    if (dateien.length > max) {
        inkluagentSetStatus(t('Höchstens {n} Dateien auf einmal. Bitte teile die Auswahl auf.', { n: max }), true);
        return null;
    }
    const fd = new FormData();
    dateien.forEach(f => fd.append('dateien', f, f.name));
    if (projectId != null) fd.append('project_id', String(projectId));
    if (kennung) fd.append('hochladen_id', kennung);
    inkluagentSetStatus(t('Wird hochgeladen …'), false);
    let res;
    try {
        res = await fetch('/api/agent/hochladen', { method: 'POST', body: fd });
    } catch (e) {
        inkluagentSetStatus(t('Verbindungsfehler. Bitte erneut versuchen.'), true);
        return null;
    }
    if (res.status === 401) { inkluagentZurAnmeldung(); return null; }
    const erg = await res.json().catch(() => ({}));
    if (!res.ok) {   // 429 mit fertigem Satz: Tagesgrenze schon VOR dem Hochladen geprueft (Server, K4)
        inkluagentSetStatus((erg && typeof erg.detail === 'string') ? erg.detail : t('Hochladen nicht möglich.'), true);
        return null;
    }
    return erg;
}
// Pruefung 13.10.2026, Runde 2 (K12): brach die Verbindung beim Hochladen ab (Proxy-Grenze bei grossen Dateien), sagte die Seite
// „Hochladen nicht möglich“, die Anhaenge blieben — und erneutes Senden lud alles noch einmal hoch, obwohl der Server den ersten
// Versuch oft schon fertig hatte. Jetzt bekommt jeder Stapel eine Kennung; dieselben Dateien (dieselben File-Objekte in derselben
// Reihenfolge) behalten sie, bis die Nachricht angenommen ist (inkluagentSend). Ein anderer Stapel bekommt eine neue.
function inkluagentHochladenKennung(dateien) {
    const s = window._inkluagentStapel;
    if (s && s.dateien.length === dateien.length && s.dateien.every((f, i) => f === dateien[i])) return s.id;
    window._inkluagentStapel = { id: inkluagentNeueKennung(), dateien: dateien.slice() };
    return window._inkluagentStapel.id;
}
// 32 Hex-Zeichen aus dem Zufall des Browsers (passt zu [A-Za-z0-9_-], hoechstens 64 Zeichen); getRandomValues gibt es auch ohne HTTPS
function inkluagentNeueKennung() {
    const b = new Uint8Array(16);
    if (window.crypto && window.crypto.getRandomValues) window.crypto.getRandomValues(b);
    else for (let i = 0; i < b.length; i++) b[i] = Math.floor(Math.random() * 256);
    return Array.from(b, x => x.toString(16).padStart(2, '0')).join('');
}
// Eine gebuendelte Ansage nach dem Hochladen (Zahlen; die Gruende stehen in der Liste)
function inkluagentHochladenSatz(ergebnisse) {
    const alle = (ergebnisse || []).length;
    const gut = (ergebnisse || []).filter(e => e.ok && !e.wartet).length;
    const wartend = (ergebnisse || []).filter(e => e.wartet).length;
    const schlecht = alle - gut - wartend;
    return t('{gut} von {alle} hochgeladen.', { gut: gut, alle: alle })
        + (wartend ? ' ' + (wartend === 1 ? t('1 wartet auf deine Entscheidung.') : t('{n} warten auf deine Entscheidung.', { n: wartend })) : '')
        + (schlecht ? ' ' + t('Die Gründe stehen unter dem Feld.') : '');
}

// ─── Dateien anhängen in der Agentenansicht (Steve 12.10.2026) ───────────────────────────────────────────────────────────────
// Wie die Büroklammer bei ChatGPT: Knopf „Dateien anhängen“ vor „Senden“ (öffnet die normale Dateiauswahl); Dateien lassen sich
// auch ins Gespräch ziehen oder ins Eingabefeld einfügen. Die Auswahl steht als Liste mit „Entfernen“ unter den Knöpfen und geht
// mit der nächsten Nachricht denselben Weg wie das Feld „Dateien mitschicken“ (POST /api/agent/hochladen: Zuordnung, Bremsen,
// Preise). Nur im Chatstil und nur, wenn das Feld selbst aus ist (Michael Karbe, Punkt 3: in der manuellen Ansicht kein Hochladen
// im Agenten). Meldungen: sichtbar unter der Liste und einmal angesagt — nie über die Statuszeile, die gehört dem laufenden Auftrag.
function inkluagentAnhaengenAn() {
    return !!(window.INKLUAGENT && window.INKLUAGENT.anhaengen) && inkluagentArbeitsfensterAn() && !inkluagentHochladenAn();
}
function inkluagentAnhangAktiv() {
    return !!document.getElementById('inkluagentAnhaengenBtn');
}
function inkluagentAnhangBauen(form, meldungen) {
    if (!inkluagentAnhaengenAn() || !form || inkluagentAnhangAktiv()) return;
    const k = window.INKLUAGENT || {};
    if (!Array.isArray(window._inkluagentAnhaenge)) window._inkluagentAnhaenge = [];
    const datei = document.createElement('input');
    datei.type = 'file';
    datei.id = 'inkluagentAnhangDatei';
    datei.multiple = true;
    datei.hidden = true;
    datei.tabIndex = -1;
    datei.accept = '.pdf,.docx,.jpg,.jpeg,.png,.gif,.webp,.heic,.heif,.bmp,.tif,.tiff,.svg';
    datei.addEventListener('change', () => { inkluagentAnhaengeHinzu(Array.from(datei.files || [])); datei.value = ''; });
    const knopf = document.createElement('button');
    knopf.type = 'button';
    knopf.id = 'inkluagentAnhaengenBtn';
    knopf.className = 'btn btn-secondary';
    knopf.textContent = t('Dateien anhängen');
    knopf.setAttribute('aria-describedby', 'inkluagentAnhangHinweis');
    knopf.addEventListener('click', () => datei.click());
    // Pruefung 13.10.2026 (B21): Grenzen und „ziehen oder einfügen“ stehen sichtbar da — vorher nur fuer Screenreader, Sehende erfuhren
    // sie erst aus einer Fehlermeldung. Runde 2 (HN1): nicht mehr im angehefteten Teil unter den Knoepfen (er wurde dadurch dauerhaft
    // 2 rem hoeher, bei 390 px Breite 27 % des Bildschirms), sondern im Fluss oben in der Beschreibung unter der Eingabe; der Knopf
    // „Dateien anhängen“ verweist weiter per aria-describedby darauf.
    const hinweis = document.createElement('p');
    hinweis.id = 'inkluagentAnhangHinweis';
    hinweis.className = 'inkluagent-hint inkluagent-anhang-hinweis';
    hinweis.textContent = t('PDF, Word (.docx) und Bilder, höchstens {n} Dateien, je bis {mb} MB. Du kannst Dateien auch ins Gespräch ziehen oder einfügen.',
                            { n: k.max_dateien || 10, mb: k.max_mb || 50 });
    const liste = document.createElement('ul');
    liste.id = 'inkluagentAnhaenge';
    liste.className = 'inkluagent-anhaenge';
    liste.setAttribute('aria-label', t('Angehängte Dateien'));
    liste.hidden = true;
    const send = document.getElementById('inkluagentSendBtn');
    if (send && send.parentNode) send.parentNode.insertBefore(knopf, send);   // erst „Dateien anhängen“, dann „Senden“
    else form.appendChild(knopf);
    [datei, liste].forEach(el => form.appendChild(el));
    const beschreibung = form.parentNode ? form.parentNode.querySelector(':scope > .inkluagent-beschreibung') : null;
    if (beschreibung) beschreibung.insertBefore(hinweis, beschreibung.firstChild); else form.appendChild(hinweis);
    // Meldung und Ergebnis des Hochladens stehen im Fluss ueber der Eingabe, nicht im angehefteten Teil (Pruefung 13.10.2026, B6): dort
    // wuchsen sie auf bis zu 60 % des Bildschirms und verdeckten den Fokus
    const meldung = document.createElement('p');
    meldung.id = 'inkluagentAnhangMeldung';
    meldung.className = 'inkluagent-hint inkluagent-anhang-meldung';
    meldung.hidden = true;
    const ziel = meldungen || form;
    const status = document.getElementById('inkluagentStatus');
    const davor = (status && status.parentNode === ziel) ? status : null;
    if (!document.getElementById('inkluagentHochgeladen')) {   // Ergebnis des Hochladens wie beim Feld (inkluagentHochgeladenZeigen)
        const ergebnis = document.createElement('ul');
        ergebnis.id = 'inkluagentHochgeladen';
        ergebnis.className = 'inkluagent-hochgeladen';
        ergebnis.setAttribute('aria-label', t('Ergebnis des Hochladens'));
        ergebnis.hidden = true;
        ziel.insertBefore(ergebnis, davor);
    }
    ziel.insertBefore(meldung, davor);
    inkluagentAblageEinrichten();
    // Einfuegen (Befehl-V) im Eingabefeld: Dateien aus der Zwischenablage anhaengen, Text bleibt Text
    const input = document.getElementById('inkluagentInput');
    if (input && !input.dataset.einfuegen) {
        input.dataset.einfuegen = '1';
        input.addEventListener('paste', e => {
            if (!inkluagentAnhangAktiv()) return;
            const f = Array.from((e.clipboardData && e.clipboardData.files) || []);
            if (!f.length) return;
            e.preventDefault();
            inkluagentAnhaengeHinzu(f);
        });
    }
    inkluagentAnhaengeZeigen();
}
// Hineinziehen (Maus): Dateien gehen an JEDER Stelle der Seite in die Anhaenge — vorher nur ueber dem Gespraech; daneben oeffnete der
// Browser die Datei als Seite, und die Agentenansicht war weg (Pruefung 13.10.2026, HK7). Die eingebettete Seite im Arbeitsfenster gibt
// Dateien hierher weiter (dashboard.js). Einmal je Seite; wirkt nur, solange es „Dateien anhängen“ gibt. Ablagen der Seite selbst
// (die das Ereignis schon behandeln) gehen vor. Waehrend des Ziehens ist das Gespraech deutlich umrandet.
function inkluagentAblageEinrichten() {
    if (window._inkluagentAblage) return;
    window._inkluagentAblage = true;
    const mitDateien = e => Array.from((e.dataTransfer && e.dataTransfer.types) || []).indexOf('Files') >= 0;
    const markieren = an => {
        const g = document.querySelector('.inkluagent-gespraech');
        if (g) g.classList.toggle('inkluagent-ablage-aktiv', an);
    };
    document.addEventListener('dragover', e => {
        if (e.defaultPrevented || !inkluagentAnhangAktiv() || !mitDateien(e)) return;
        e.preventDefault();
        markieren(true);
    });
    document.addEventListener('dragleave', e => { if (!e.relatedTarget) markieren(false); });
    document.addEventListener('dragend', () => markieren(false));
    document.addEventListener('drop', e => {
        markieren(false);
        if (e.defaultPrevented || !inkluagentAnhangAktiv() || !mitDateien(e)) return;
        e.preventDefault();
        inkluagentAnhaengeHinzu(Array.from(e.dataTransfer.files || []));
    });
}
function inkluagentGroesse(bytes) {
    const lang = document.documentElement.lang || 'de';
    if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)).toLocaleString(lang) + ' KB';
    return (bytes / (1024 * 1024)).toLocaleString(lang, { maximumFractionDigits: 1 }) + ' MB';
}
function inkluagentAnhangZahl(n) {
    return n === 0 ? t('Keine Datei angehängt.') : (n === 1 ? t('1 Datei angehängt.') : t('{n} Dateien angehängt.', { n: n }));
}
function inkluagentAnhangMelden(text, fehler, still) {
    const p = document.getElementById('inkluagentAnhangMeldung');
    if (p) {
        p.textContent = text || '';
        p.hidden = !text;
        if (fehler) p.setAttribute('data-state', 'error'); else p.removeAttribute('data-state');
    }
    if (text && !still) announce(text);
}
function inkluagentAnhaengeHinzu(dateien) {
    if (!dateien || !dateien.length || !inkluagentAnhangAktiv()) return;
    const k = window.INKLUAGENT || {};
    const max = k.max_dateien || 10;
    const maxMb = k.max_mb || 50;
    const liste = window._inkluagentAnhaenge;
    const zuGross = [];
    let zuViele = 0;
    dateien.forEach(f => {
        if (f.size > maxMb * 1024 * 1024) { zuGross.push(f.name); return; }
        if (liste.some(x => x.name === f.name && x.size === f.size && x.lastModified === f.lastModified)) return;   // schon dabei
        if (liste.length >= max) { zuViele++; return; }
        liste.push(f);
    });
    inkluagentAnhaengeZeigen();
    const teile = [];
    if (zuGross.length) teile.push(t('Zu groß, höchstens {mb} MB je Datei: {namen}.', { mb: maxMb, namen: zuGross.join(', ') }));
    if (zuViele) teile.push(t('Höchstens {n} Dateien auf einmal. Bitte teile die Auswahl auf.', { n: max }));
    inkluagentAnhangMelden(teile.concat([inkluagentAnhangZahl(liste.length)]).join(' '), teile.length > 0);
}
function inkluagentAnhaengeZeigen() {
    const ul = document.getElementById('inkluagentAnhaenge');
    if (!ul) return;
    const liste = window._inkluagentAnhaenge || [];
    ul.textContent = '';
    liste.forEach((f, i) => {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.className = 'inkluagent-anhang-name';
        name.textContent = f.name + ' (' + inkluagentGroesse(f.size) + ')';
        const weg = document.createElement('button');
        weg.type = 'button';
        weg.className = 'btn btn-secondary inkluagent-anhang-weg';
        weg.textContent = t('Entfernen');
        weg.setAttribute('aria-label', t('„{name}“ entfernen', { name: f.name }));
        weg.addEventListener('click', () => inkluagentAnhangEntfernen(i));
        li.appendChild(name);
        li.appendChild(weg);
        ul.appendChild(li);
    });
    ul.hidden = !liste.length;
}
function inkluagentAnhangEntfernen(i) {
    const liste = window._inkluagentAnhaenge || [];
    const weg = liste.splice(i, 1)[0];
    inkluagentAnhaengeZeigen();
    // Fokus auf das naechste „Entfernen“, sonst zurueck auf „Dateien anhängen“
    const knoepfe = document.querySelectorAll('#inkluagentAnhaenge .inkluagent-anhang-weg');
    const ziel = knoepfe[Math.min(i, knoepfe.length - 1)] || document.getElementById('inkluagentAnhaengenBtn');
    if (ziel) ziel.focus();
    inkluagentAnhangMelden(t('„{name}“ entfernt.', { name: weg ? weg.name : '' }) + ' ' + inkluagentAnhangZahl(liste.length), false);
}
// Nach dem Hochladen: nur die Dateien weg, die gerade mitgingen (Pruefung 13.10.2026, B12) — was waehrend des Hochladens neu
// angehaengt wurde, bleibt fuer die naechste Nachricht (vorher loeschte das Leeren es stumm mit). Ohne Liste: alle.
function inkluagentAnhaengeLeeren(gesendet) {
    const weg = Array.isArray(gesendet) ? gesendet : null;
    window._inkluagentAnhaenge = weg ? (window._inkluagentAnhaenge || []).filter(f => weg.indexOf(f) < 0) : [];
    inkluagentAnhaengeZeigen();
    const rest = window._inkluagentAnhaenge.length;
    inkluagentAnhangMelden(rest ? inkluagentAnhangZahl(rest) : '', false, true);
}
