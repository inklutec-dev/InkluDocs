// InkluAgent — Formular-Karten: der Agent bringt Felder und Knöpfe (Bestellen, Bezahlen, Aufträge).
// Teil der Agenten-Skripte (Liste und Ladereihenfolge: backend/inkluagent/skripte.py; Kern: inkluagent.js). Auf oberster
// Ebene nur Funktionen — beim Laden laeuft nichts. Texte ueber t() aus window.I18N (check_i18n.py prueft sie).

// ─── Formular-Karten (Paket 1, Steve 13.10.2026; Schalter AGENT_KARTEN, inkluagent/karten.py, docs/INKLUAGENT.md) ───
// „Der Agent bringt die Knöpfe zu einem“: statt auf eine Seite zu schicken, legt er das Formular ins Gespraech — vorbelegt mit dem, was
// der Nutzer gesagt hat. Abschicken kann nur der Knopf der Karte (POST /api/agent/karte, dieselbe Funktion wie die Seite). Gebaut aus
// echten Formularelementen (HTML vor ARIA): sichtbares label je Feld, required, ein benanntes <form> (im Rotor unter „Formulare“),
// Fehler am Feld (aria-invalid und aria-describedby, die Saetze kommen vom Server in der Sprache der Oberflaeche). Nach dem Abschicken,
// abgelaufen oder ohne Schalter (window.INKLUAGENT.karten) steht die Karte nur noch zum Lesen da, mit ihrem Ergebnis.
function inkluagentFormularKarte(a) {
    const schalter = !!(window.INKLUAGENT && window.INKLUAGENT.karten);
    const offen = (!a.zustand || a.zustand === 'offen') && schalter;
    const kennung = 'kf-' + String(a.karten_id).replace(/[^A-Za-z0-9_-]/g, '');
    const karte = document.createElement('div');
    karte.className = 'inkluagent-karte inkluagent-formular';
    karte.dataset.kartenId = a.karten_id;
    karte.dataset.zustand = offen ? 'offen' : (a.zustand && a.zustand !== 'offen' ? a.zustand : 'abgelaufen');
    const h = document.createElement('h4');
    h.id = kennung + '-titel';
    h.textContent = a.titel || t('Formular');
    karte.appendChild(h);
    if (a.text) {
        const p = document.createElement('p');
        p.className = 'inkluagent-karte-text';
        p.textContent = a.text;
        karte.appendChild(p);
    }
    if (Array.isArray(a.zeilen) && a.zeilen.length) {
        // Schritte eines Auftrags als nummerierte Liste (der Screenreader sagt „Liste, 3 Einträge“ und die Nummer), sonst Aufzaehlung
        const ul = document.createElement(a.nummeriert ? 'ol' : 'ul');
        ul.className = 'inkluagent-karte-zeilen';
        a.zeilen.forEach(z => { const li = document.createElement('li'); li.textContent = String(z); ul.appendChild(li); });
        karte.appendChild(ul);
    }
    // Hinweise zur Karte als eigene Absaetze, nicht als weitere Punkte der Liste (Pruefung Auftraege, N7)
    (Array.isArray(a.hinweise) ? a.hinweise : []).forEach(h => {
        const p = document.createElement('p');
        p.className = 'inkluagent-karte-hinweis';
        p.textContent = String(h);
        karte.appendChild(p);
    });
    if (!offen) {
        inkluagentFormularErgebnis(karte, a.ergebnis || (karte.dataset.zustand === 'gesendet' ? t('Abgeschickt.') : t('Nicht mehr gültig.')));
        return karte;
    }
    const form = document.createElement('form');
    form.className = 'inkluagent-formular-felder';
    form.noValidate = true;   // geprueft wird auf dem Server; er sagt es am Feld, in der Sprache der Oberflaeche
    form.setAttribute('aria-labelledby', h.id);
    (a.felder || []).forEach(f => form.appendChild(inkluagentFormularFeld(kennung, f)));
    const knopf = document.createElement('button');
    knopf.type = 'submit';
    knopf.className = 'btn btn-primary';
    knopf.textContent = a.knopf || t('Absenden');
    const status = document.createElement('p');
    status.className = 'inkluagent-karte-status';
    status.setAttribute('role', 'status');
    form.appendChild(knopf);
    form.appendChild(status);
    form.addEventListener('submit', (e) => { e.preventDefault(); inkluagentFormularAbsenden(karte, a, form, knopf, status); });
    karte.appendChild(form);
    return karte;
}
// Ein Feld nach dem Schema des Servers (art: text, email, tel, textarea, auswahl, haken), vorbelegt mit wert
function inkluagentFormularFeld(kennung, f) {
    const wrap = document.createElement('div');
    wrap.className = 'inkluagent-formular-feld' + (f.art === 'haken' ? ' inkluagent-formular-haken' : '');
    const id = kennung + '-' + String(f.name || '').replace(/[^A-Za-z0-9_-]/g, '');
    let el;
    if (f.art === 'textarea') {
        el = document.createElement('textarea');
        el.rows = 3;
        el.value = typeof f.wert === 'string' ? f.wert : '';
    } else if (f.art === 'auswahl') {
        el = document.createElement('select');
        (f.auswahl || []).forEach(o => {
            const opt = document.createElement('option');
            opt.value = o.wert;
            opt.textContent = o.label;
            if (o.wert === f.wert) opt.selected = true;
            el.appendChild(opt);
        });
    } else {
        el = document.createElement('input');
        el.type = { haken: 'checkbox', email: 'email', tel: 'tel' }[f.art] || 'text';
        if (f.art === 'haken') el.checked = f.wert === true;
        else el.value = typeof f.wert === 'string' ? f.wert : '';
    }
    el.id = id;
    el.name = f.name;
    if (f.pflicht) el.required = true;
    if (f.max && (el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && el.type !== 'checkbox'))) el.maxLength = f.max;
    if (f.autocomplete) el.setAttribute('autocomplete', f.autocomplete);
    const label = document.createElement('label');
    label.htmlFor = id;
    label.textContent = f.label || f.name;
    const fehler = document.createElement('p');
    fehler.id = id + '-fehler';
    fehler.className = 'inkluagent-formular-fehler';
    el.setAttribute('aria-describedby', fehler.id);
    // Wie im Express-Warenkorb: der Link zu den Bedingungen steht VOR dem Haken; „Abo buchen“ (Baustein 3) hat zwei
    // (Widerrufsbelehrung, Nutzungsbedingungen) — je ein Absatz, nur Pfade dieser Anwendung
    [f.link].concat(Array.isArray(f.links) ? f.links : []).forEach(link => {
        if (!link || !eigenerPfad(link.url)) return;
        const lp = document.createElement('p');
        const l = document.createElement('a');
        l.href = link.url;
        l.target = '_blank';
        l.rel = 'noopener';
        l.textContent = link.text || link.url;
        lp.appendChild(l);
        wrap.appendChild(lp);
    });
    if (f.art === 'haken') { wrap.appendChild(el); wrap.appendChild(label); }
    else { wrap.appendChild(label); wrap.appendChild(el); }
    wrap.appendChild(fehler);
    return wrap;
}
// Fehler des Servers an die Felder (leer = alle Fehler weg). Rueckgabe: das erste fehlerhafte Feld.
function inkluagentFormularFehlerZeigen(form, fehler) {
    let erstes = null;
    form.querySelectorAll('input, select, textarea').forEach(el => {
        const text = fehler && fehler[el.name];
        const p = document.getElementById(el.id + '-fehler');
        if (p) p.textContent = text ? String(text) : '';
        if (text) { el.setAttribute('aria-invalid', 'true'); erstes = erstes || el; }
        else el.removeAttribute('aria-invalid');
    });
    return erstes;
}
// Ergebnis unter der Karte (abgeschickt, abgelaufen), dazu ein Link der Antwort (z. B. „Auftrag ansehen“)
function inkluagentFormularErgebnis(karte, text, anhang) {
    const p = document.createElement('p');
    p.className = 'inkluagent-karte-ergebnis';
    p.textContent = text;
    karte.appendChild(p);
    if (anhang && anhang.art === 'link' && eigenerPfad(anhang.url)) {
        const l = document.createElement('a');
        l.className = 'btn btn-secondary';
        l.href = anhang.url;
        l.textContent = anhang.text || anhang.url;
        karte.appendChild(l);
    }
    return p;
}
// Abschicken: nur der Knopf (aria-disabled haelt den Fokus, solange es laeuft). Erfolg: das Formular weicht dem Ergebnis, der Fokus
// geht darauf (der Screenreader liest „Die Einladung an … ist verschickt.“); Fehler am Feld: Fokus aufs erste fehlerhafte Feld, der
// Grund steht in der Statuszeile; abgelaufen oder schon abgeschickt: nur noch das Ergebnis.
async function inkluagentFormularAbsenden(karte, a, form, knopf, status) {
    if (knopf.getAttribute('aria-disabled') === 'true') return;
    knopf.setAttribute('aria-disabled', 'true');
    status.textContent = t('Wird gesendet …');
    const werte = {};
    (a.felder || []).forEach(f => {
        const el = form.elements.namedItem(f.name);
        if (el) werte[f.name] = f.art === 'haken' ? !!el.checked : el.value;
    });
    inkluagentFormularFehlerZeigen(form, {});
    let res = null;
    let data = {};
    try {
        res = await fetch('/api/agent/karte', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({ karten_id: a.karten_id, werte: werte }) });
        if (res.status === 401) { inkluagentZurAnmeldung(); return; }
        data = await res.json().catch(() => ({}));
    } catch (e) {
        knopf.removeAttribute('aria-disabled');
        status.textContent = t('Verbindungsfehler. Bitte erneut versuchen.');
        inkluagentKlang('fehler');
        return;
    }
    const zustand = data && data.karte && data.karte.zustand;
    if (res.ok || (zustand && zustand !== 'offen')) {
        const text = (data.karte && data.karte.ergebnis) || data.text || (typeof data.detail === 'string' ? data.detail : t('Abgeschickt.'));
        karte.dataset.zustand = zustand || 'gesendet';
        form.remove();
        const ergebnis = inkluagentFormularErgebnis(karte, text, res.ok ? data.anhang : null);
        inkluagentKlang(res.ok ? 'fertig' : 'fehler');
        fokusAuf(ergebnis);
        // Ein erteilter Auftrag (Baustein 2) meldet sich, wenn er zu Ende ist: wie nach dem Knopf einer Bestaetigungs-Karte
        // nachsehen, solange eine Rueckmeldung wartet (fragt nur, solange eine wartet)
        if (res.ok) inkluagentEntschieden(data.anhang);
        if (res.ok) inkluagentRueckmeldungenBeobachten();
        // Kaufen im Gespraech (Baustein 3): weiter zur Bezahlseite von Stripe — nur dorthin, als ganze Seite (Zahlung nie im
        // Rahmen, wie auf „Abo & Verbrauch“); kurz warten, damit das Ergebnis der Karte noch angesagt wird
        if (res.ok && inkluagentZurBezahlung(data.weiter)) {
            setTimeout(() => { (window.top || window).location.assign(data.weiter); }, 1500);
        }
        return;
    }
    knopf.removeAttribute('aria-disabled');
    status.textContent = typeof data.detail === 'string' ? data.detail : t('Fehler. Bitte erneut versuchen.');
    inkluagentKlang('fehler');
    const erstes = inkluagentFormularFehlerZeigen(form, data.fehler || {});
    if (erstes) erstes.focus();
}
// Adresse der Bezahlseite (Baustein 3): nur Stripe-Checkout ueber https — eine andere Adresse aus einer Antwort des Servers wird
// nie angesteuert
function inkluagentZurBezahlung(url) {
    return typeof url === 'string' && /^https:\/\/checkout\.stripe\.com\//.test(url);
}
// Paket 1 (Steve 13.10.2026: „wenn der Knopf dann so angerauscht kommt“): eine neue Karte fliegt von rechts — von dort, wo das
// Arbeitsfenster steht — ins Gespraech, zugleich mit dem Wusch der Klangspur (von rechts in die Mitte); danach laeuft einmal ein Glanz
// ueber ihren Knopf. Nur fuer Karten, die gerade ankommen (inkluagentAntwortMelden), nicht beim Laden des Verlaufs. Mit „Bewegung
// reduzieren“ steht sie sofort da (style.css, „Reduced Motion“). Rein optisch: Fokus und Vorlesen bleiben, wie sie sind.
function inkluagentKarteEinfliegen(karte) {
    karte.classList.remove('inkluagent-karte-kommt');
    void karte.offsetWidth;   // eine laufende Animation neu beginnen
    karte.classList.add('inkluagent-karte-kommt');
    clearTimeout(karte._kommtPlan);
    karte._kommtPlan = setTimeout(() => karte.classList.remove('inkluagent-karte-kommt'), 2000);   // nach Flug und Glanz
}
