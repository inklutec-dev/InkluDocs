/* VERWALTUNG (25.09.2026, Steve): gemeinsame Helfer der vier Verwaltungs-Seiten
 * (Kunden, Kunde, Umsatz, API, Einstellungen) — Datums- und Betragstexte, Laden mit
 * Rechte-Behandlung, die Dialoge „API-Limit ändern“ und „Buchung berichtigen“.
 * I18N: alle sichtbaren Texte über t() (window.I18N), Kataloge backend/locales, geprüft von scripts/check_i18n.py.
 * Barrierefreiheit: natives HTML (Links, Knöpfe, <select>, <dialog>), keine Tabellen
 * (Steve hört die Seiten mit VoiceOver), Meldungen über announce() (Live-Region der App-Hülle).
 */
(function () {
  'use strict';

  const PLAN_NAMEN = { free: 'Free', single: 'Single', team: 'Team', enterprise: 'Enterprise' };

  function datumLang(iso) {
    // '2026-09-25' oder '2026-09-25 09:59' -> '25. September 2026' (Sprache der Oberfläche).
    if (!iso) return '';
    const p = String(iso).slice(0, 10).split('-');
    if (p.length !== 3) return String(iso);
    const d = new Date(Date.UTC(+p[0], +p[1] - 1, +p[2]));
    try {
      return new Intl.DateTimeFormat(window.LANG || 'de',
        { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }).format(d);
    } catch (e) { return p[2] + '.' + p[1] + '.' + p[0]; }
  }

  function datumZeit(iso) {
    // '2026-09-25 09:59' -> '25. September 2026, 09:59'
    if (!iso) return '';
    const uhr = String(iso).slice(11, 16);
    return datumLang(iso) + (uhr ? ', ' + uhr : '');
  }

  function datumKurz(iso) {
    // '2026-09-25 09:59' -> '25. September, 09:59' (Jahr steht schon in der Überschrift).
    if (!iso) return '';
    const p = String(iso).slice(0, 10).split('-');
    const uhr = String(iso).slice(11, 16);
    let tag;
    try {
      tag = new Intl.DateTimeFormat(window.LANG || 'de', { day: 'numeric', month: 'long', timeZone: 'UTC' })
        .format(new Date(Date.UTC(+p[0], +p[1] - 1, +p[2])));
    } catch (e) { tag = p[2] + '.' + p[1] + '.'; }
    return tag + (uhr ? ', ' + uhr : '');
  }

  function monatLang(ym) {
    const p = String(ym).split('-');
    if (p.length < 2) return String(ym);
    const d = new Date(Date.UTC(+p[0], +p[1] - 1, 1));
    try {
      return new Intl.DateTimeFormat(window.LANG || 'de',
        { month: 'long', year: 'numeric', timeZone: 'UTC' }).format(d);
    } catch (e) { return String(ym); }
  }

  function euro(cent) {
    const wert = (Number(cent) || 0) / 100;
    try {
      return new Intl.NumberFormat(window.LANG || 'de', { style: 'currency', currency: 'EUR' }).format(wert);
    } catch (e) { return wert.toFixed(2).replace('.', ',') + ' €'; }
  }

  function euroFeld(cent) {
    // Vorbelegung eines Betragsfeldes: '87,50' (ohne Währungszeichen, Dezimalkomma).
    return ((Number(cent) || 0) / 100).toFixed(2).replace('.', ',');
  }

  function zahl(n) {
    try { return new Intl.NumberFormat(window.LANG || 'de').format(Number(n) || 0); }
    catch (e) { return String(n); }
  }

  function el(tag, klasse, text) {
    const e = document.createElement(tag);
    if (klasse) e.className = klasse;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  function zeile(ziel, begriff, wert) {
    // Schlichte Absätze „Begriff: Wert“ statt Tabelle oder <dl> (Hörtest 11.08.2026).
    if (wert === null || wert === undefined || wert === '') return null;
    const p = el('p', 'dash-report-zeile', begriff + ': ' + String(wert));
    ziel.appendChild(p);
    return p;
  }

  function leer(ziel, text) {
    ziel.innerHTML = '';
    ziel.appendChild(el('p', 'dash-empty', text));
    ziel.setAttribute('aria-busy', 'false');
  }

  // GET mit einheitlicher Behandlung: 401 -> Anmeldung, 403 -> Kein-Zugriff-Text im Ziel.
  async function ladeJson(url, ziel) {
    let res;
    try { res = await fetch(url, { headers: { 'Accept': 'application/json' } }); }
    catch (e) { if (ziel) leer(ziel, t('Die Daten konnten nicht geladen werden. Bitte die Seite neu laden.')); return null; }
    if (res.status === 401) { window.location.href = '/login'; return null; }
    if (res.status === 403) { if (ziel) leer(ziel, t('Kein Zugriff – diese Seite ist nur für Administratoren.')); return null; }
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      if (ziel) leer(ziel, d.detail || t('Die Daten konnten nicht geladen werden. Bitte die Seite neu laden.'));
      return null;
    }
    return res.json();
  }

  async function sendeJson(url, methode, daten) {
    let res;
    try {
      res = await fetch(url, {
        method: methode, headers: { 'Content-Type': 'application/json' },
        body: daten === undefined ? undefined : JSON.stringify(daten),
      });
    } catch (e) { return { ok: false, daten: { detail: t('Keine Verbindung zum Server. Bitte erneut versuchen.') } }; }
    if (res.status === 401) { window.location.href = '/login'; return { ok: false, daten: {} }; }
    const d = await res.json().catch(() => ({}));
    return { ok: res.ok, daten: d };
  }

  function istVollAdmin() {
    // currentUser ist ein globales let aus dashboard.js (nicht window.currentUser).
    return typeof currentUser !== 'undefined' && !!currentUser && currentUser.admin_level === 'full';
  }

  function planText(plan, quelle) {
    const name = PLAN_NAMEN[plan] || plan || 'Free';
    if (!plan || plan === 'free') return t('Free');
    if (quelle === 'stripe') return t('{plan} über Stripe', { plan: name });
    if (quelle === 'rechnung') return t('{plan} auf Rechnung', { plan: name });
    return name;
  }

  // Eine Buchung als Teile in fester Reihenfolge: Datum, (Kunde), was, Betrag und Weg, Zusätze.
  // kurz = Datum ohne Jahr (Umsatz-Seite: das Jahr steht in der Überschrift).
  function buchungTeile(b, mitKunde, kurz) {
    const teile = [];
    teile.push(kurz ? datumKurz(b.gebucht_am) : datumZeit(b.gebucht_am));
    if (mitKunde) teile.push(b.kunde_name || b.kunde_email || t('gelöschtes Konto'));
    let was;
    if (b.art === 'abo') {
      was = t('Abo {plan}', { plan: PLAN_NAMEN[b.plan] || b.plan || '' }).trim();
      if (b.laufzeit_monate) was += ', ' + (b.laufzeit_monate === 1 ? t('1 Monat') : t('{monate} Monate', { monate: b.laufzeit_monate }));
    } else {
      was = t('{menge} Credits', { menge: zahl(b.credits) });
    }
    teile.push(was);
    if (b.weg === 'bonus') teile.push(b.art === 'abo' ? t('ohne Berechnung') : t('Bonus (kostenlos)'));
    else if (b.weg === 'stripe') teile.push(t('über Stripe, {betrag}', { betrag: euro(b.betrag_cent) }));
    else teile.push(t('Verkauf auf Rechnung, {betrag}', { betrag: euro(b.betrag_cent) }));
    if (b.status === 'ausstehend') teile.push(t('Lastschrift noch ausstehend'));
    if (b.status === 'rueckgelaufen') teile.push(t('Lastschrift zurückgegangen, zählt nicht zum Umsatz'));
    if (b.status === 'storniert') teile.push(t('storniert, zählt nicht zum Umsatz'));
    if (b.rechnungsnummer) teile.push(t('Rechnung {nummer}', { nummer: b.rechnungsnummer }));
    if (b.notiz) teile.push(b.notiz);
    // Bei Stripe sagt „über Stripe“ schon alles; „eingetragen von“ nur bei Hand-Buchungen.
    if (b.gebucht_von && b.weg !== 'stripe') teile.push(t('eingetragen von {name}', { name: b.gebucht_von }));
    if (b.korrigiert && b.status !== 'storniert') teile.push(t('berichtigt'));
    return teile;
  }

  function buchungText(b, mitKunde, kurz) {
    return buchungTeile(b, mitKunde, kurz).join(' · ');
  }

  // Nach einer Aktion: ERST neu laden, DANN ansagen und den Fokus auf die Abschnitts-
  // Überschrift setzen. Umgekehrt überschrieb das Neuladen die Ansage, und der Fokus fiel
  // auf <body>, weil der auslösende Knopf beim Neuaufbau verschwand (Prüfbericht 25.09.2026).
  async function abschliessen(danach, meldung, fokusId) {
    if (danach) await danach();
    const h = fokusId && byId(fokusId);
    if (h) { h.setAttribute('tabindex', '-1'); h.focus(); }
    announce(meldung);
  }

  // Bonus-Schwelle (wie beim Gutschreiben) — steht am Dialog, damit jede Seite sie kennt.
  function bonusGrenze() {
    const dlg = byId('korrekturDialog');
    return dlg ? parseInt(dlg.getAttribute('data-bonus-grenze') || '500', 10) : 500;
  }

  // ── Dialog „Buchung berichtigen“ (Kundenseite + Umsatz-Seite) ──────────
  let _korrektur = null;
  function korrekturOeffnen(b, danach, fokusId) {
    const dlg = byId('korrekturDialog');
    if (!dlg) return;
    _korrektur = { b: b, danach: danach, fokus: fokusId };
    byId('korrekturFuer').textContent = buchungText(b, true);
    byId('korrekturGross').checked = false;
    byId('korrekturArt').value = b.weg === 'bonus' ? 'bonus' : 'verkauf';
    byId('korrekturBetrag').value = b.weg === 'bonus' ? '' : euroFeld(b.betrag_cent);
    byId('korrekturNummer').value = b.rechnungsnummer || '';
    byId('korrekturGrund').value = '';
    byId('korrekturFehler').textContent = '';
    korrekturArtWechsel();
    dlg.showModal();
    byId('korrekturArt').focus();
  }
  function korrekturArtWechsel() {
    const verkauf = byId('korrekturArt').value === 'verkauf';
    byId('korrekturVerkauf').hidden = !verkauf;
    const gross = !verkauf && _korrektur && _korrektur.b.weg !== 'bonus' && _korrektur.b.credits > bonusGrenze();
    byId('korrekturGrossFeld').hidden = !gross;
    byId('korrekturGrossText').textContent = t('Ja, ich möchte bewusst mehr als {grenze} Credits verschenken.', { grenze: zahl(bonusGrenze()) });
  }
  function korrekturEinrichten() {
    const dlg = byId('korrekturDialog');
    if (!dlg) return;
    byId('korrekturArt').addEventListener('change', korrekturArtWechsel);
    byId('korrekturAbbrechen').addEventListener('click', () => dlg.close());
    let laeuft = false;
    byId('korrekturForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      if (laeuft) return;
      const fehler = byId('korrekturFehler');
      fehler.textContent = '';
      const art = byId('korrekturArt').value;
      const grund = byId('korrekturGrund').value.trim();
      if (art === 'verkauf' && !byId('korrekturBetrag').value.trim()) {
        fehler.textContent = t('Bitte den Betrag eintragen.'); byId('korrekturBetrag').focus(); return;
      }
      if (grund.length < 3) {
        fehler.textContent = t('Bitte kurz den Grund der Korrektur angeben.'); byId('korrekturGrund').focus(); return;
      }
      if (!byId('korrekturGrossFeld').hidden && !byId('korrekturGross').checked) {
        fehler.textContent = t('Bitte das Häkchen setzen, um den großen Bonus zu bestätigen.'); byId('korrekturGross').focus(); return;
      }
      laeuft = true;
      const r = await sendeJson('/api/admin/buchungen/' + _korrektur.b.id + '/korrektur', 'POST', {
        art: art, betrag: byId('korrekturBetrag').value.trim(),
        rechnungsnummer: byId('korrekturNummer').value.trim(), grund: grund,
        bestaetigt_gross: byId('korrekturGross').checked,
      });
      laeuft = false;
      if (!r.ok) { fehler.textContent = r.daten.detail || t('Die Korrektur konnte nicht gespeichert werden.'); return; }
      dlg.close();
      await abschliessen(_korrektur.danach, r.daten.message || t('Buchung berichtigt.'), _korrektur.fokus);
    });
  }

  // ── Dialog „Gutschrift stornieren“ (Kundenseite + Umsatz-Seite) ────────
  let _storno = null;
  function kannStornieren(b) {
    return b.status !== 'storniert';
  }
  function kannBerichtigen(b) {
    return b.weg !== 'stripe' && b.status !== 'storniert';
  }
  function stornoOeffnen(b, danach, fokusId) {
    const dlg = byId('stornoDialog');
    if (!dlg) return;
    _storno = { b: b, danach: danach, fokus: fokusId };
    byId('stornoFuer').textContent = buchungText(b, true);
    const rest = (b.paket_rest === null || b.paket_rest === undefined) ? 0 : b.paket_rest;
    let folgen = b.art === 'abo'
      ? t('Die Buchung bleibt sichtbar, zählt aber nicht mehr zum Umsatz. Der Plan des Kunden bleibt unverändert — ihn bei Bedarf über „Abo zuweisen oder ändern“ anpassen.')
      : t('Zurückgenommen werden die noch nicht verbrauchten {rest} von {menge} Credits. Bereits verbrauchte Credits bleiben verbraucht. Die Buchung bleibt sichtbar, zählt aber nicht mehr zum Umsatz.',
        { rest: zahl(rest), menge: zahl(b.credits) });
    if (b.weg === 'stripe') folgen += ' ' + t('Das Geld erstattest du im Stripe-Dashboard — hier wird nur die Buchung berichtigt.');
    byId('stornoFolgen').textContent = folgen;
    byId('stornoGrund').value = '';
    byId('stornoFehler').textContent = '';
    dlg.showModal();
    byId('stornoGrund').focus();
  }
  function stornoEinrichten() {
    const dlg = byId('stornoDialog');
    if (!dlg) return;
    byId('stornoAbbrechen').addEventListener('click', () => dlg.close());
    let laeuft = false;
    byId('stornoForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      if (laeuft) return;
      const fehler = byId('stornoFehler');
      fehler.textContent = '';
      const grund = byId('stornoGrund').value.trim();
      if (grund.length < 3) { fehler.textContent = t('Bitte kurz den Grund für das Stornieren angeben.'); byId('stornoGrund').focus(); return; }
      laeuft = true;
      const r = await sendeJson('/api/admin/buchungen/' + _storno.b.id + '/storno', 'POST', { grund: grund });
      laeuft = false;
      if (!r.ok) { fehler.textContent = r.daten.detail || t('Das Stornieren hat nicht geklappt.'); return; }
      dlg.close();
      await abschliessen(_storno.danach, r.daten.message || t('Gutschrift storniert.'), _storno.fokus);
    });
  }

  // Knöpfe „Berichtigen“ und „Stornieren“ an eine Buchungszeile hängen (nur Voll-Admins).
  function buchungKnoepfe(li, b, danach, fokusId) {
    if (!istVollAdmin()) return;
    const datum = datumZeit(b.gebucht_am);
    if (kannBerichtigen(b)) {
      const k = el('button', 'btn btn-secondary btn-small', t('Berichtigen'));
      k.type = 'button';
      k.setAttribute('aria-label', t('Buchung vom {datum} berichtigen', { datum: datum }));
      k.addEventListener('click', () => korrekturOeffnen(b, danach, fokusId));
      li.appendChild(document.createTextNode(' '));
      li.appendChild(k);
    }
    if (kannStornieren(b)) {
      const k = el('button', 'btn btn-delete btn-small', t('Stornieren'));
      k.type = 'button';
      k.setAttribute('aria-label', t('Buchung vom {datum} stornieren', { datum: datum }));
      k.addEventListener('click', () => stornoOeffnen(b, danach, fokusId));
      li.appendChild(document.createTextNode(' '));
      li.appendChild(k);
    }
  }

  // ── Dialog „API-Tageslimit ändern“ (Kundenseite + API-Seite) ───────────
  let _limit = null;
  function limitOeffnen(konto, danach, fokusId) {
    const dlg = byId('limitDialog');
    if (!dlg) return;
    _limit = { konto: konto, danach: danach, fokus: fokusId };
    byId('limitFuer').textContent = t('Für: {name} ({email})', { name: konto.name, email: konto.email });
    byId('limitWert').value = String(konto.limit);
    byId('limitFehler').textContent = '';
    dlg.showModal();
    byId('limitWert').focus();
  }
  function limitEinrichten() {
    const dlg = byId('limitDialog');
    if (!dlg) return;
    byId('limitAbbrechen').addEventListener('click', () => dlg.close());
    let laeuft = false;
    byId('limitForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      if (laeuft) return;
      const fehler = byId('limitFehler');
      fehler.textContent = '';
      const roh = byId('limitWert').value.trim();
      let limit = null;
      if (roh !== '') {
        limit = parseInt(roh, 10);
        if (!Number.isInteger(limit) || String(limit) !== roh || limit < 0 || limit > 1000000) {
          fehler.textContent = t('Das Tageslimit muss eine ganze Zahl von 0 bis 1000000 sein.');
          byId('limitWert').focus();
          return;
        }
        // Genau der Standard = Standard, kein eigener Eintrag (Steve 11.08.2026).
        if (limit === window.API_LIMIT_STANDARD) limit = null;
      }
      laeuft = true;
      const r = await sendeJson('/api/admin/users/' + _limit.konto.id + '/api-limit', 'POST', { limit: limit });
      laeuft = false;
      if (!r.ok) { fehler.textContent = r.daten.detail || t('Das Limit konnte nicht gespeichert werden.'); return; }
      dlg.close();
      await abschliessen(_limit.danach, r.daten.message || t('Gespeichert.'), _limit.fokus);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    korrekturEinrichten();
    stornoEinrichten();
    limitEinrichten();
  });

  window.Verwaltung = {
    PLAN_NAMEN, datumLang, datumZeit, monatLang, euro, euroFeld, zahl, el, zeile, leer,
    ladeJson, sendeJson, istVollAdmin, planText, buchungText, buchungTeile, datumKurz, korrekturOeffnen, limitOeffnen,
    buchungKnoepfe, zaehltZumUmsatz: (b) => b.weg !== 'bonus' && (b.status === 'ok' || b.status === 'ausstehend'),
  };
})();
