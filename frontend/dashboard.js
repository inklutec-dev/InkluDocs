// Gemeinsame Logik für alle Dashboard-Seiten (App-Shell): Helfer, Login-Status,
// Navigationsleiste (mit aria-current), Credit-/Tageslimit-Anzeige,
// Fußzeile, Abmelden.
//
// Drei Betriebsarten der Hülle (Schalter setzt das jeweilige Template):
//   (keiner)           eingeloggte App-Seite (base_app.html): /api/me ist
//                      Pflicht, 401 leitet zur Anmeldung um.
//   window.GUEST_MODE  Gast-Review (Upload-Zone/Gastzugang): kein /api/me,
//                      nur das Logo in der Seitenleiste.
//   window.OEFFENTLICH öffentliche Inhaltsseite (base_oeffentlich.html,
//                      25.08.2026): /api/me wird versucht — eingeloggte
//                      Nutzer bekommen ihre normale Navigation, alle anderen
//                      die öffentliche (Preise, Kontakt, Über uns, Anmelden).
//                      Nie eine Weiterleitung: Impressum, Kündigungsknopf,
//                      Widerruf und Kontakt müssen ohne Login erreichbar sein.

const byId = (id) => document.getElementById(id);

// i18n-Helfer: schlaegt Text in der vom Server gelieferten Uebersetzungstabelle
// (window.I18N) nach. Ohne Tabelle (z.B. noch nicht migrierte Seite) faellt er
// auf den deutschen Ausgangstext zurueck. Optionale {platzhalter} werden ersetzt.
function t(s, params) {
  var out = (window.I18N && window.I18N[s]) || s;
  if (params) {
    Object.keys(params).forEach(function (k) {
      // Ersetzung ueber eine Funktion: sonst deuten $&, $1 … in Namen/Texten als Ersetzungsmuster (Pruefbericht 24.09.2026)
      out = out.replace('{' + k + '}', function () { return String(params[k]); });
    });
  }
  return out;
}

// Lange Listen kuerzen (Michael Karbe, Feedback 20261012 - 1, Punkt 7): Meine Projekte, Meine Auftraege, Meine Ablage und die
// Dokumente im Projekt zeigen je die fuenf neuesten; der Rest ist verborgen (hidden, auch fuer Screenreader weg). Darunter der Knopf
// „Alle anzeigen (n)“: zeigt alle, setzt den Fokus auf den ersten bisher verborgenen Eintrag und merkt sich das fuer diese Liste,
// solange die Seite offen ist (Neuzeichnen kuerzt nicht wieder). vonHinten: die LETZTEN fuenf bleiben sichtbar (Dokumente stehen
// aelteste zuerst). behalten(el): Eintraege, die sichtbar bleiben muessen (offen, Sprungziel, laeuft gerade).
window._listenAlle = window._listenAlle || {};
function listeKuerzen(container, opts) {
  opts = opts || {};
  const anzahl = opts.anzahl || 5;
  if (!container || (opts.schluessel && window._listenAlle[opts.schluessel])) return;
  const alt = container.nextElementSibling;
  if (alt && alt.classList.contains('liste-alle')) alt.remove();
  const eintraege = Array.from(container.children).filter((e) => !/^(SCRIPT|TEMPLATE)$/.test(e.tagName));
  if (eintraege.length <= anzahl) return;
  const sichtbar = opts.vonHinten ? eintraege.slice(-anzahl) : eintraege.slice(0, anzahl);
  const versteckt = eintraege.filter((e) => sichtbar.indexOf(e) < 0 && !(opts.behalten && opts.behalten(e)));
  if (!versteckt.length) return;
  versteckt.forEach((e) => { e.hidden = true; });
  const p = document.createElement('p');
  p.className = 'liste-alle';
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'btn btn-secondary';
  b.textContent = t('Alle anzeigen ({n})', { n: eintraege.length });
  b.addEventListener('click', () => {
    if (opts.schluessel) window._listenAlle[opts.schluessel] = true;
    versteckt.forEach((e) => { e.hidden = false; });
    p.remove();
    const erstes = versteckt[0];
    const ziel = erstes.querySelector('a[href], button, summary, [tabindex]');
    if (ziel) ziel.focus();
    else { erstes.setAttribute('tabindex', '-1'); erstes.focus(); }
  });
  p.appendChild(b);
  container.insertAdjacentElement('afterend', p);
}
window.listeKuerzen = listeKuerzen;

// Die eine Ansage-Region aller Seiten (#liveRegion). Pruefung 13.10.2026, B9: erst leeren, dann nach 100 ms setzen — sonst spricht
// VoiceOver denselben Text beim zweiten Mal oft nicht („Stand vorlesen“ zweimal). Was bis zum Setzen dazukommt, wird an die wartende
// Ansage angehaengt statt sie zu ersetzen. app.html hat seit diesem Tag keine eigene Kopie.
// Pruefung 13.10.2026, Runde 2 (A-N8): Mindestabstand. Vorher leerte JEDER Aufruf die Region sofort — eine Ansage, die erst 30 ms
// dastand, verschwand ungesprochen (gemessen: Schritt kurz vor „Antwort vom InkluAgent ist da.“, „1 Alt-Text aktualisiert.“ kurz
// davor). Jetzt wird die Region fruehestens ABSTAND_MS nach der vorigen Ansage geleert und neu gesetzt; was bis dahin kommt, wartet
// und wird angehaengt. opts.verwerfbar: ein Zwischenstand (ein Schritt des Agenten), den announce.verwerfen() streicht, solange er
// noch wartet — ist die Antwort da, soll er ihr Vorlesen nicht unterbrechen (inkluagentSchrittAnsagenVerwerfen).
function announce(msg, opts) {
  // Im Arbeitsfenster spricht die eingebettete Seite nur, wenn jemand in ihr arbeitet (Live-Fenster, 17.10.2026): sonst sagt die Seite
  // draussen an — vorher hoerte man Laden, Nachfragen und Ergebnisse der Seite im Fenster doppelt oder mitten im Lesen
  if (typeof imRahmen === 'function' && imRahmen() && typeof document !== 'undefined' && !document.hasFocus()) return;
  const el = byId('liveRegion');
  if (!el || !msg) return;
  const z = announce._z || (announce._z = { warte: [], plan: 0, zuletzt: 0 });
  const text = String(msg);
  if (!z.warte.some((x) => x.text === text)) z.warte.push({ text: text, verwerfbar: !!(opts && opts.verwerfbar) });
  if (z.plan) return;   // schon geplant: kommt mit
  const setzen = () => {
    z.plan = 0;
    if (!z.warte.length) return;   // inzwischen verworfen
    el.textContent = z.warte.map((x) => x.text).join(' ');
    z.warte = [];
    z.zuletzt = Date.now();
  };
  const leeren = () => { el.textContent = ''; z.plan = setTimeout(setzen, 100); };
  const warten = z.zuletzt + announce.ABSTAND_MS - 100 - Date.now();
  if (warten > 0) z.plan = setTimeout(leeren, warten); else leeren();
}
announce.ABSTAND_MS = 1000;   // etwa so lange braucht ein Screenreader, um eine kurze Ansage aufzunehmen
// Wartende Zwischenstaende streichen (nur die mit verwerfbar; Fehler und alles andere bleiben)
announce.verwerfen = function () {
  const z = announce._z;
  if (z) z.warte = z.warte.filter((x) => !x.verwerfbar);
};

// Fokus auf ein Element, das sonst nicht fokussierbar ist (Ueberschrift, Meldung, Bereich). Pruefung 13.10.2026, B19 — Muster aus
// express.html (09.10.2026): ein DAUERHAFTES tabindex="-1" laesst VoiceOver „klickbar“ sagen. Das Element traegt tabindex nur, solange
// es den Fokus hat. Natuerlich fokussierbare Elemente und vorhandenes tabindex bleiben unangetastet.
function fokusAuf(el, opts) {
  if (!el) return;
  if (!el.hasAttribute('tabindex') && !el.matches('a[href], button, input, select, textarea, summary, iframe, [contenteditable="true"]')) {
    el.setAttribute('tabindex', '-1');
    el.addEventListener('blur', () => el.removeAttribute('tabindex'), { once: true });
  }
  el.focus(opts);
}
window.fokusAuf = fokusAuf;

// Ein Pfad DIESER Anwendung (Pruefung 13.10.2026, S14): beginnt mit genau einem „/“, ohne Backslash (Browser machen aus „/\host“
// „//host“), und bleibt nach dem Aufloesen auf der eigenen Herkunft. Eine Stelle fuer alle Links, Downloads, Rahmen-Ziele und
// „weiter“ im Frontend — heute kommen diese Werte vom eigenen Server; die Pruefung haelt, falls je Fremdes hineingeraet.
function eigenerPfad(u) {
  if (typeof u !== 'string' || !/^\/(?![\/\\])/.test(u) || u.indexOf('\\') >= 0) return false;
  try { return new URL(u, window.location.origin).origin === window.location.origin; } catch (e) { return false; }
}
window.eigenerPfad = eigenerPfad;

// Datum robust formatieren (kein Date()-Parsing des Rohstrings wegen
// Safari-Eigenheiten bei SQLite-Zeitstempeln "YYYY-MM-DD HH:MM:SS").
// Seit 11.08.2026 (Steve): in der Sprache der Oberflaeche AUSGESCHRIEBEN —
// "7. November 2026" statt "07.11.2026"; VoiceOver liest das richtig herum.
// Der Browser kennt alle Sprachen selbst (Intl), Fallback bleibt das alte
// Punkt-Format.
function formatDate(s) {
  if (!s) return '';
  const p = String(s).substring(0, 10).split('-');
  if (p.length !== 3) return String(s);
  try {
    const d = new Date(Date.UTC(+p[0], +p[1] - 1, +p[2]));
    return new Intl.DateTimeFormat(window.LANG || 'de',
      { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }).format(d);
  } catch (e) { return `${p[2]}.${p[1]}.${p[0]}`; }
}

// Credit-Zeilen aus dem abo-Block von /api/me bauen (Abo-Modell Etappe 2,
// 31.07.2026). EINE Quelle der Wahrheit fuer die Wortwahl: Startseite und
// /abo-Seite rufen dieselbe Funktion auf. Liefert fertige Textzeilen
// (Zeile 1 = Monats-Credits, optional Zeile 2 = Zusatz-Credits aus Paketen).
// Defensive Fallbacks, falls der Server verfuegbar_monat/rest nicht liefert.
function buildCreditLines(abo, kurz) {
  const lines = [];
  if (abo.ist_betreiber && kurz) {
    // Abo-Seite: die Betreiber-Aussage steht schon in der Plan-Zeile darueber —
    // hier nur noch der reine Verbrauch, damit Screenreader-Nutzer sie nicht
    // zweimal hintereinander hoeren (Steve-Hoertest 31.07.).
    lines.push(t('Diesen Monat verbraucht: {verbraucht} Credits.', { verbraucht: abo.verbraucht }));
  } else if (abo.ist_betreiber) {
    // Betreiber-Konto: kein Kontingent, aber der Verbrauch bleibt sichtbar —
    // so sieht der Betreiber seine eigenen KI-Kosten, ohne eine Grenze
    // vorgegaukelt zu bekommen, die fuer ihn nicht gilt.
    lines.push(t('Betreiber-Konto: unbegrenzte Credits. Diesen Monat verbraucht: {verbraucht}.', { verbraucht: abo.verbraucht }));
  } else if (abo.kontingent === null || abo.kontingent === undefined) {
    // Enterprise/unbegrenzt: kein Kontingent, nur den Verbrauch nennen.
    lines.push(t('Credits diesen Monat: {verbraucht} verbraucht (unbegrenzter Plan).', { verbraucht: abo.verbraucht }));
  } else {
    const verfuegbar = (abo.verfuegbar_monat === null || abo.verfuegbar_monat === undefined)
      ? abo.kontingent + (abo.uebertrag || 0)
      : abo.verfuegbar_monat;
    const rest = (abo.rest === null || abo.rest === undefined)
      ? Math.max(0, verfuegbar - abo.verbraucht)
      : abo.rest;
    let zeile = t('Credits diesen Monat: {verbraucht} von {verfuegbar} verbraucht — {rest} verfügbar.', { verbraucht: abo.verbraucht, verfuegbar: verfuegbar, rest: rest });
    if (abo.uebertrag > 0) {
      zeile += ' ' + t('Davon sind {uebertrag} Credits aus dem Vormonat übernommen.', { uebertrag: abo.uebertrag });
    }
    lines.push(zeile);
  }
  // Tausenderpunkt wie überall sonst („2.000“, Sichtprüfung Runde 8)
  const tausender = (n) => { try { return new Intl.NumberFormat(window.LANG || 'de').format(Number(n) || 0); } catch (e) { return String(n); } };
  if (abo.pakete_rest > 0) {
    lines.push(t('Zusatz-Credits: {anzahl} verfügbar.', { anzahl: tausender(abo.pakete_rest) }));
  }
  if (abo.vorgemerkt > 0) {
    // Express-Service (05.10.2026): vorgemerkt fuer offene Auftraege (abgebucht erst bei der Lieferung) — diese Credits
    // stehen fuer anderes nicht mehr zur Verfuegung.
    lines.push(t('Davon für Express-Aufträge vorgemerkt: {anzahl} Credits.', { anzahl: tausender(abo.vorgemerkt) }));
  }
  return lines;
}

let currentUser = null;

// Zur Anmeldung MIT Ruecksprung auf die aktuelle Seite (Express Runde 7): nach der Anmeldung geht es hierher zurueck.
// Der Server prueft das Ziel (nur interne Pfade); /login selbst ist nie Ziel.
// ─── Arbeitsfenster der Agentenansicht (13.10.2026) ───
// Laeuft die Seite eingebettet im Arbeitsfenster (window.EINGEBETTET aus base_app.html), oeffnen Ziele, die nicht in den Rahmen
// gehoeren, die ganze Seite: Projekte und Startseite (dort ist der Agent selbst), Sprachwechsel (die ganze Oberflaeche),
// Anmeldung, Zahlung. Alles andere bleibt im Rahmen — der Browser schickt dann „Sec-Fetch-Dest: iframe“, die Seite bleibt eingebettet.
function imRahmen() {
  try { return !!(window.EINGEBETTET && window.top !== window); } catch (e) { return false; }
}
// Im Arbeitsfenster nimmt eine Seite den Fokus nur, wenn er schon in ihr steht (Live-Fenster 16.10.2026): die Projektseite im Fenster
// fokussierte beim ersten Zeichnen ihre Ueberschrift 1 — der Fokus sprang aus dem Gespraech in den Rahmen, VoiceOver landete dort, und das
// Fenster hielt den Nutzer fuer jemanden, der darin arbeitet. Als ganze Seite gilt wie bisher: Fokus erlaubt.
function fokusHierErlaubt() {
  try { return !imRahmen() || document.hasFocus(); } catch (e) { return true; }
}
window.fokusHierErlaubt = fokusHierErlaubt;

// ─── Bühne für das Arbeitsfenster (Live-Fenster „zum Anklemmen“, 17.10.2026) ───
// Eine Seite, die im Arbeitsfenster der Agentenansicht steht (eingebettet), bietet der Seite draussen EINE feste Schnittstelle:
//   window.InkluBuehne = { version, zeigen(stelle, opts), neuZeichnen(), lesen(stelle), titel(), ansicht(), beobachten(fn) }
// Die Seite draussen (inkluagent.js) greift sonst nicht in diese Seite: Ansichten duerfen sich aendern, solange die Seite ihre Stellen
// kennt. Jede Seite meldet nur, wie sie eine Stelle findet — buehneEinrichten({ finden, ansicht, ansichtWechseln, neuZeichnen,
// aufdecken }), finden(stelle) -> { karte, kopf, text, ansicht } | null. Den Rest macht dieser Kern: passende Ansicht (still),
// Karte aufdecken (zugeklappt, gefiltert, gekuerzt), nur diese Seite rollen, markieren, Hand, nach jedem Neuzeichnen Marke und Lage
// halten (nicht, solange jemand in dieser Seite arbeitet), und Beobachter benachrichtigen (Titel und Text der Stelle fuer „So klingt
// diese Stelle“). Stellen kommen vom Server (inkluagent/arbeitsfenster.py): art bild|feld|dokument|seite|ausgabe|ansicht, Kennungen, ansicht.
function buehneEinrichten(seite) {
  if (!imRahmen() || !seite || typeof seite.finden !== 'function') return null;
  const zustand = { stelle: null, bis: 0, halten: 0, hoerer: [], lauf: 0 };
  const ruhig = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const main = document.getElementById('main');
  const warten = async (fn, ms) => {
    const ende = Date.now() + ms;
    let x = fn();
    while (!x && Date.now() < ende) { await new Promise(r => setTimeout(r, 200)); x = fn(); }
    return x;
  };
  function aufdecken(el) {
    for (let d = el.closest('details'); d; d = d.parentElement ? d.parentElement.closest('details') : null) d.open = true;
    if (typeof seite.aufdecken === 'function') seite.aufdecken(el);   // eigene Filter der Seite
    for (let x = el; x && x !== main && x !== document.body; x = x.parentElement) if (x.hidden) x.hidden = false;   // gekuerzte Listen
  }
  function rollen(karte, weich) {
    const r = karte.getBoundingClientRect();
    const oben = Math.max(0, window.scrollY + r.top - Math.max(12, (window.innerHeight - Math.min(r.height, window.innerHeight)) / 2));
    try { window.scrollTo({ top: oben, behavior: weich ? 'smooth' : 'auto' }); } catch (e) { window.scrollTo(0, oben); }
  }
  function markieren(karte) {
    karte.classList.add('inkluagent-hin-ziel');
    clearTimeout(karte._buehneMarke);
    karte._buehneMarke = setTimeout(() => karte.classList.remove('inkluagent-hin-ziel'), Math.max(0, zustand.bis - Date.now()));
  }
  function textVon(el) {
    if (!el) return '';
    return String(('value' in el && typeof el.value === 'string') ? el.value : el.textContent || '').trim();
  }
  function lesen(stelle) {
    const f = stelle ? seite.finden(stelle) : null;
    return f ? { art: stelle.art, text: textVon(f.text) } : null;
  }
  function titel() {
    const h = document.querySelector('#main h1');
    return h ? h.textContent.trim() : String(document.title || '').split(' · ')[0].trim();
  }
  function melden() {
    const daten = { titel: titel(), stelle: zustand.stelle, gelesen: zustand.stelle ? lesen(zustand.stelle) : null };
    zustand.hoerer.forEach(fn => { try { fn(daten); } catch (e) { /* ein Beobachter stoert die Seite nie */ } });
  }
  async function zeigen(stelle, opts) {
    opts = opts || {};
    if (!stelle || typeof stelle !== 'object') return false;
    const lauf = ++zustand.lauf;   // nur die juengste Fahrt zaehlt
    const vorab = seite.finden(stelle);
    const soll = stelle.ansicht || (vorab && vorab.ansicht) || null;
    if (soll && typeof seite.ansicht === 'function' && seite.ansicht() !== soll && typeof seite.ansichtWechseln === 'function') {
      const ok = await seite.ansichtWechseln(soll);   // still: ohne Fokus und Ansage; geht die Ansicht nicht, bleibt die Seite
      if (lauf !== zustand.lauf) return false;
      if (ok === false && !(vorab && vorab.karte)) return false;
    }
    const f = await warten(() => { const g = seite.finden(stelle); return g && g.karte ? g : null; }, opts.warten || 8000);
    if (!f || lauf !== zustand.lauf) return false;
    zustand.stelle = stelle;
    zustand.bis = Date.now() + 3500;
    zustand.halten = Date.now() + 30000;
    aufdecken(f.kopf || f.karte);
    rollen(f.karte, !ruhig() && opts.weich !== false);
    markieren(f.karte);
    if (opts.hand !== false && typeof zeigeHand === 'function') {
      setTimeout(() => { try { zeigeHand(f.kopf || f.karte); } catch (e) { /* nur Schmuck */ } }, ruhig() ? 0 : 500);
    }
    melden();
    return true;
  }
  async function neuZeichnen() {
    if (typeof seite.neuZeichnen === 'function') await seite.neuZeichnen();
    melden();
    return true;
  }
  function beobachten(fn) {
    if (typeof fn === 'function') zustand.hoerer.push(fn);
    melden();
  }
  // Nach jedem Neuzeichnen der Seite (Abruf, Nachfragen waehrend eines Laufs, Ansichtswechsel): Marke und Lage halten, Beobachter
  // benachrichtigen — gebuendelt. Die Lage nur, solange niemand in dieser Seite arbeitet (document.hasFocus).
  if (main && typeof MutationObserver !== 'undefined') {
    let wartet = false;
    new MutationObserver(() => {
      if (wartet) return;
      wartet = true;
      setTimeout(() => {
        wartet = false;
        const st = zustand.stelle;
        const f = st ? seite.finden(st) : null;
        if (f && f.karte) {
          if (Date.now() < zustand.bis && !f.karte.classList.contains('inkluagent-hin-ziel')) { aufdecken(f.kopf || f.karte); markieren(f.karte); }
          if (Date.now() < zustand.halten && !document.hasFocus()) {
            const r = f.karte.getBoundingClientRect();
            if (r.bottom < 0 || r.top > window.innerHeight) rollen(f.karte, false);
          }
        }
        melden();
      }, 150);
    }).observe(main, { childList: true, subtree: true, characterData: true });
  }
  return { version: 1, zeigen, neuZeichnen, lesen, titel, beobachten,
           ansicht: () => (typeof seite.ansicht === 'function' ? seite.ansicht() : null) };
}
window.buehneEinrichten = buehneEinrichten;
function ganzeSeite(url) {
  (imRahmen() ? window.top : window).location.href = url;
}
window.ganzeSeite = ganzeSeite;
// ─── Welche Seiten im Arbeitsfenster erscheinen (Pruefung 13.10.2026, Runde 2, S-N6 und A-N6) ───
// EINE Liste des Servers: window.INKLUAGENT.fensterSeiten, Pfad-Praefixe. Der Server rahmt genau diese Seiten ein (SAMEORIGIN), alles
// andere bekommt DENY — Oberflaeche und Server koennen nicht mehr auseinanderlaufen (vorher zwei gegenlaeufige Listen: eine Sperrliste
// im Server, eine eigene hier). Ein Pfad passt, wenn er gleich einem Praefix ist oder mit Praefix + „/“ beginnt; Abfrage und Anker
// zaehlen nicht. Die eingebettete Seite nimmt die Liste ihres Hauptfensters, wenn sie selbst keine hat (oeffentliche Seiten aus
// base_oeffentlich.html bekommen kein window.INKLUAGENT). Fuer die Elternseite (inkluagentSeiteImFensterMoeglich, der load-Hoerer des
// Rahmens) und die eingebettete Seite (Klicks unten) dieselbe Funktion.
function fensterSeitenListe() {
  const liste = (w) => {
    try { const k = w && w.INKLUAGENT; return (k && Array.isArray(k.fensterSeiten)) ? k.fensterSeiten : null; } catch (e) { return null; }
  };
  return liste(window) || (imRahmen() ? liste(window.parent) : null);
}
function fensterSeite(pfad) {
  let p;
  try { p = new URL(String(pfad), window.location.origin).pathname; } catch (e) { return false; }
  const liste = fensterSeitenListe();
  if (!liste) return !gehoertGanzeSeite(p);   // Server ohne Liste: die bisherige Regel
  return liste.some((x) => typeof x === 'string' && x.charAt(0) === '/' && (p === x || p.indexOf(x + '/') === 0));
}
window.fensterSeite = fensterSeite;
// Rueckfall, solange der Server keine Liste mitgibt: Pfade, die nie im Rahmen erscheinen. Projekte und Startseite (dort ist der Agent
// selbst), Sprachwechsel, Anmeldung, die statischen Seiten aus frontend/ (Datensicherheit, Impressum, Widerruf, Nutzungsbedingungen)
// und die Rechtstexte. Runde 2 (A-N6): Ueber uns und Kontakt, Preise, Kuendigen und Widerrufen sind nicht mehr dabei —
// base_oeffentlich.html hat jetzt eine eingebettete Fassung wie base_app.html.
function gehoertGanzeSeite(pfad) {
  return pfad === '/' || /^\/(app|dashboard|set-language|login|logout|register|forgot|reset|datensicherheit|impressum-app|widerruf-app|nutzungsbedingungen-app|widerruf|impressum|datenschutz|nutzungsbedingungen|avv|api)(\/|\?|#|$)/.test(pfad);
}
window.gehoertGanzeSeite = gehoertGanzeSeite;
// Ein Link auf das Projekt, das diese eingebettete Seite gerade zeigt (/app?projekt=<dasselbe>)
function gleichesProjekt(u) {
  try {
    const hier = new URLSearchParams(window.location.search).get('projekt');
    return /^\/app\/?$/.test(window.location.pathname) && /^\/app\/?$/.test(u.pathname) && !!hier && u.searchParams.get('projekt') === hier;
  } catch (e) { return false; }
}
function eingebettetEinrichten() {
  if (!imRahmen()) return;
  const zurueck = byId('zumGespraech');
  if (zurueck) zurueck.addEventListener('click', (e) => {
    e.preventDefault();
    try { window.parent.inkluagentZumGespraech(); } catch (err) { /* ohne Agent: nichts */ }
  });
  // Escape schliesst die Ueberlagerung des Arbeitsfensters auch, wenn der Fokus in dieser Seite steht (Pruefung 13.10.2026, B7): eine
  // Taste im Rahmen erreicht die Elternseite nicht — auch nicht ein natives <dialog> dort (im Browser gemessen). Nicht, wenn die Seite
  // Escape selbst braucht (eigener offener Dialog oder schon behandelt).
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || e.defaultPrevented || document.querySelector('dialog[open]')) return;
    try { if (typeof window.parent.inkluagentEscape === 'function' && window.parent.inkluagentEscape()) e.preventDefault(); } catch (err) { /* ohne Agent */ }
  });
  // Dateien, die diese Seite nicht selbst annimmt, gehen an „Dateien anhängen“ des InkluAgent — sonst oeffnete der Browser sie im Rahmen
  // als Seite (Pruefung 13.10.2026, HK7). Eigene Ablagen der Seite (Hochladen) behandeln das Ereignis vorher und gehen vor.
  const anhaengen = () => {
    try { return typeof window.parent.inkluagentAnhangAktiv === 'function' && window.parent.inkluagentAnhangAktiv(); } catch (err) { return false; }
  };
  const mitDateien = (e) => Array.from((e.dataTransfer && e.dataTransfer.types) || []).indexOf('Files') >= 0;
  document.addEventListener('dragover', (e) => { if (!e.defaultPrevented && mitDateien(e) && anhaengen()) e.preventDefault(); });
  document.addEventListener('drop', (e) => {
    if (e.defaultPrevented || !mitDateien(e) || !anhaengen()) return;
    e.preventDefault();
    window.parent.inkluagentAnhaengeHinzu(Array.from(e.dataTransfer.files || []));
  });
  document.addEventListener('click', (e) => {
    const a = e.target.closest && e.target.closest('a[href]');
    if (!a || a.hasAttribute('download') || a.target === '_blank' || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    let u;
    try { u = new URL(a.href, window.location.href); } catch (err) { return; }
    // Runde 2: was nicht in der Liste der Fensterseiten steht, oeffnet die ganze Seite (der Server rahmt es nicht ein). Live-Fenster
    // (16.10.2026): Links auf DASSELBE Projekt (Ansichten „Alt-Texte“, „Dokument“ …) bleiben im Rahmen — die Projektseite faengt sie selbst ab
    if (u.origin !== window.location.origin || fensterSeite(u.pathname) || gleichesProjekt(u)) return;
    e.preventDefault();
    ganzeSeite(u.pathname + u.search + u.hash);
  }, true);
}
// Ziel eines Hinfuehrens (Begleiter, 12.10.2026): ?hin=<id> bzw. ueberschrift — suchen (die Seite baut oft noch), aufklappen,
// Fokus, sichtbar markieren; der Parameter verschwindet aus der Adresse. Auf jeder Seite, auch im Arbeitsfenster.
// Pruefung 13.10.2026: nur ids aus der Liste des Servers (S14: window.INKLUAGENT.hinZiele, aus inkluagent/begleiter.py). Die Markierung
// bleibt, bis der Fokus das Ziel verlaesst (HB1; vorher 6 Sekunden — mit Lupe oft zu kurz). Fokus ueber fokusAuf (B19).
// Pruefung 13.10.2026, Runde 2 (S-N9): fail-closed. Fehlt die Liste oder ist sie kein Array, wird ?hin= gar nicht ausgewertet (vorher galt
// dann jede id in der Form einer id — ein fremder Link wie „/konto?hin=<id>“ setzte den Fokus auf ein beliebiges Element, z. B. einen
// Knopf); sonst zaehlen nur ids aus der Liste (auch „ueberschrift“ steht darin). Der Server gibt die Liste seit Runde 2 auf jeder Seite
// mit window.INKLUAGENT mit, unabhaengig vom Begleiter (main._inkluagent_schalter). Oeffentliche Seiten (base_oeffentlich.html) haben kein
// window.INKLUAGENT: eingebettet nehmen sie die Liste ihres Hauptfensters, als ganze Seite dieselbe Liste als window.HIN_ZIELE (der
// Begleiter fuehrt auch zu Preisen, Über uns, Kündigen und Widerrufen — ohne Liste fiele der Fokus dort ins Leere).
function hinZieleListe() {
  const liste = (w) => {
    try { const k = w && w.INKLUAGENT; return (k && Array.isArray(k.hinZiele)) ? k.hinZiele : null; } catch (e) { return null; }
  };
  return liste(window) || (imRahmen() ? liste(window.parent) : null) || (Array.isArray(window.HIN_ZIELE) ? window.HIN_ZIELE : null);
}
function hinfuehrenBeimLaden() {
  let ziel = null;
  try { ziel = new URLSearchParams(window.location.search).get('hin'); } catch (e) { return; }
  if (!ziel) return;
  try {
    const u = new URL(window.location.href);
    u.searchParams.delete('hin');
    history.replaceState(history.state, '', u);
  } catch (e) { /* Adresse bleibt */ }
  if (!/^[A-Za-z][\w-]{0,40}$/.test(ziel)) return;
  const liste = hinZieleListe();
  if (!liste || liste.indexOf(ziel) < 0) return;
  let n = 0;
  const suchen = () => {
    const el = ziel === 'ueberschrift' ? document.querySelector('#main h1') : document.getElementById(ziel);
    const det = el && el.closest('details');
    if (det && !det.open) det.open = true;
    if (!el || !(el.offsetWidth || el.offsetHeight || el.getClientRects().length)) {
      if (++n < 40) setTimeout(suchen, 100);
      return;
    }
    el.classList.add('inkluagent-hin-ziel');
    el.addEventListener('blur', () => el.classList.remove('inkluagent-hin-ziel'), { once: true });
    el.scrollIntoView({ block: 'center' });
    fokusAuf(el, { preventScroll: true });
    zeigeHand(el);
  };
  suchen();
}

// Die zeigende Hand (16.10.2026, Steve: „dass der Agent das mit der Hand zeigt“ — wie in InkluEdit seit 15.10.): beim Hinfuehren gleitet
// eine Hand vom rechten Rand zur Stelle, an der Fingerspitze breitet sich einmal ein Ring aus, dann verschwindet sie. Die Markierung
// und der Fokus oben bleiben, wie sie waren — die Hand ist nur fuers Auge (aria-hidden, nimmt keine Klicks). Wer im System
// „Bewegung reduzieren“ eingestellt hat, bekommt keine Hand, nur die Markierung; den Klang hoert er trotzdem (Klang ist keine
// Bewegung). Klang: 'karte' (Wusch von rechts) aus klang.js, im Arbeitsfenster der des Hauptfensters; ob er klingt, entscheidet
// das Konto (Töne).
// Form nach dem Symbol „pointer“ der freien Symbolsammlung Lucide (lucide.dev), ISC-Lizenz — Copyright (c) Lucide Contributors 2022,
// Teile Copyright (c) Cole Bemis 2013-2022 (Feather, MIT). Die Lizenz erlaubt Nutzung, Aenderung und Weitergabe, auch kommerziell,
// solange dieser Hinweis erhalten bleibt.
const HAND_PFADE = ['M22 14a8 8 0 0 1-8 8', 'M18 11v-1a2 2 0 0 0-2-2a2 2 0 0 0-2 2',
  'M14 10V9a2 2 0 0 0-2-2a2 2 0 0 0-2 2v1', 'M10 9.5V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v10',
  'M18 11a2 2 0 1 1 4 0v3a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15'];
function zeigeKlang() {
  const k = (w) => { try { return w && w.Klang && typeof w.Klang.spielen === 'function' ? w.Klang : null; } catch (e) { return null; } };
  const klang = k(window) || (imRahmen() ? k(window.parent) : null);
  if (klang) { try { klang.spielen('karte'); } catch (e) { /* ohne Ton */ } }
}
function zeigeHand(el) {
  zeigeKlang();
  try {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    document.querySelectorAll('.inkluagent-hand, .inkluagent-hand-ring').forEach(x => x.remove());   // nie zwei Haende
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '64');
    svg.setAttribute('height', '64');
    svg.setAttribute('focusable', 'false');
    [['#ffffff', '4.5'], ['#1b2a4a', '1.9']].forEach(([farbe, breite]) => HAND_PFADE.forEach(d => {
      const p = document.createElementNS(ns, 'path');
      p.setAttribute('d', d);
      p.setAttribute('fill', 'none');
      p.setAttribute('stroke', farbe);
      p.setAttribute('stroke-width', breite);
      p.setAttribute('stroke-linecap', 'round');
      p.setAttribute('stroke-linejoin', 'round');
      svg.appendChild(p);
    }));
    const hand = document.createElement('div');
    hand.className = 'inkluagent-hand';
    hand.setAttribute('aria-hidden', 'true');
    hand.appendChild(svg);
    const W = window.innerWidth, H = window.innerHeight;
    hand.style.transform = 'translate(' + (W + 24) + 'px,' + Math.round(H * 0.65) + 'px)';
    document.body.appendChild(hand);
    hand.getBoundingClientRect();   // Startlage festhalten, sonst springt die Hand ohne Gleiten ans Ziel
    // Ziel erst nach dem Rollen messen: Fingerspitze an der linken unteren Ecke des Ziels (bei breiten Zielen etwas eingerueckt)
    requestAnimationFrame(() => {
      const r = el.getBoundingClientRect();
      const zx = Math.round(Math.max(4, Math.min(W - 68, r.left + Math.min(r.width * 0.3, 140) - 21)));
      const zy = Math.round(Math.max(4, Math.min(H - 68, r.bottom - 6)));
      hand.style.transform = 'translate(' + zx + 'px,' + zy + 'px)';
      setTimeout(() => {
        const ring = document.createElement('div');
        ring.className = 'inkluagent-hand-ring';
        ring.setAttribute('aria-hidden', 'true');
        ring.style.left = (zx + 1) + 'px';
        ring.style.top = (zy - 15) + 'px';
        document.body.appendChild(ring);
        ring.getBoundingClientRect();
        ring.classList.add('inkluagent-hand-ring-weit');
        setTimeout(() => ring.remove(), 900);
      }, 900);
    });
    setTimeout(() => { hand.style.opacity = '0'; }, 2600);
    setTimeout(() => hand.remove(), 3100);
  } catch (e) { /* ohne Hand: Markierung und Fokus bleiben */ }
}

// Zur Anmeldung mit Ruecksprung. Im Rahmen des Arbeitsfensters (Pruefung 13.10.2026, K7) ist das Ziel die Seite des HAUPTFENSTERS —
// sonst ging es nach der Anmeldung auf „/express?eingebettet=1“: eine eingebettete Seite ohne Seitenleiste und Agent.
function zurAnmeldung(ziel) {
  let hier = ziel;
  if (!hier) {
    let w = window;
    try { if (imRahmen()) w = window.top; hier = w.location.pathname + w.location.search + w.location.hash; }
    catch (e) { hier = ''; }
  }
  ganzeSeite((!hier || hier.indexOf('/login') === 0 || !eigenerPfad(hier)) ? '/login' : '/login?weiter=' + encodeURIComponent(hier));
}
window.zurAnmeldung = zurAnmeldung;

async function loadCurrentUser() {
  if (window.GUEST_MODE) { currentUser = null; return null; }  // Gast: kein /api/me, keine Weiterleitung
  let res;
  try {
    res = await fetch('/api/me');
  } catch (e) {
    // Netzfehler: eine oeffentliche Seite bleibt trotzdem benutzbar (anonyme
    // Huelle); App-Seiten verhalten sich wie bisher.
    if (window.OEFFENTLICH) { currentUser = null; return null; }
    throw e;
  }
  if (res.status === 401) {
    // Oeffentliche Seite: kein Login noetig, keine Weiterleitung — die Huelle
    // rendert gleich die oeffentliche Navigation (renderSidebar).
    if (window.OEFFENTLICH) { currentUser = null; return null; }
    zurAnmeldung();
    return null;
  }
  const data = await res.json();
  currentUser = data.user || null;
  const greeting = byId('greeting');
  if (greeting && currentUser) {
    greeting.textContent = currentUser.display_name ? (t('Hallo') + ', ' + currentUser.display_name) : t('Willkommen');
  }
  // Credit-Anzeige auf der Startseite (Abo-Modell Etappe 2, 31.07.2026):
  // ersetzt die alte Tageslimit-Zeile, sobald /api/me den abo-Block liefert
  // (Steve: "erst mal nur die Credits", kein zusaetzlicher Bilder-Zaehler).
  // Fehlt der Block (Backend noch nicht umgestellt), bleibt die bisherige
  // Tageslimit-Anzeige als Fallback unveraendert stehen. Bewusst KEINE
  // Live-Region: ruhige Textzeilen im Dokumentfluss.
  const limitInfo = byId('dailyLimitInfo');
  const dl = data.daily_limit;
  if (limitInfo && data.abo) {
    // Idempotent wie renderLegalNote (Review-Befund 12): bei einem erneuten
    // Aufruf zuerst die frueher eingefuegten Zusatzzeilen entfernen, sonst
    // stapeln sich Credit-Zeilen und Abo-Link mit jedem Durchlauf.
    limitInfo.parentNode.querySelectorAll('.dash-limit-zusatz').forEach((el) => el.remove());
    const lines = buildCreditLines(data.abo);
    // Die Credit-Zeile selbst ist der Link zu /abo (Steve 11.08.2026) —
    // vorher stand dahinter ein eigener "Abo & Verbrauch öffnen"-Link.
    // Ein unsichtbarer Zusatz nennt das Linkziel (WCAG 2.4.4), Sehende
    // erkennen den Link an der Unterstreichung.
    limitInfo.textContent = '';
    const creditLink = document.createElement('a');
    creditLink.href = '/abo';
    creditLink.textContent = lines[0] || '';
    const srZiel = document.createElement('span');
    srZiel.className = 'sr-only';
    srZiel.textContent = ' — ' + t('öffnet Abo & Verbrauch');
    creditLink.appendChild(srZiel);
    limitInfo.appendChild(creditLink);
    let anker = limitInfo;
    lines.slice(1).forEach((zeile) => {
      const p = document.createElement('p');
      p.className = 'dash-limit dash-limit-zusatz';
      p.textContent = zeile;
      anker.insertAdjacentElement('afterend', p);
      anker = p;
    });
  } else if (limitInfo && dl) {
    limitInfo.textContent = t('Heute {used} von {limit} Bildern genutzt – noch {remaining} übrig.', { used: dl.used, limit: dl.limit, remaining: dl.remaining });
  }
  return currentUser;
}

// Navigationspunkte. Ein neuer Eintrag hier erscheint auf allen Seiten.
// Karbe-Wunsch (06.06.2026): Eigener Menüpunkt für den Datenschutz-Text.
// Update 08.06.2026 (Karbe + Steve): Eintrag heißt jetzt "Datensicherheit"
// und zeigt auf die In-App-Sicht /datensicherheit — damit die Sidebar beim
// Klick sichtbar bleibt (wie bei Projekten / Einstellungen). Der Inhalt der
// Seite wird im Frontend per fetch aus /datenschutz geladen (Single Source).
// Die rechtlichen Footer-Links Impressum | Datenschutz | Nutzungsbedingungen
// (juristische Bezeichnung) bleiben unverändert.
//
// AUFGERÄUMT (Runde 10, 08.10.2026 — Michael Karbe 06./07.10., Steves Go): 7 statt 13 Einträge.
//   1 Startseite · 2 Meine Projekte · 3 Meine Ablage · 4 Meine Aufträge (+ Express-Warenkorb nach Einstellung)
//   · 5 Meine Vorlagen (Prompts + Stammdaten) · 6 Über uns und Kontakt · 7 Konto (aufklappbar: KONTO_ITEMS + Abmelden)
// „Neues Projekt anlegen“ ist ein Knopf oben auf „Meine Projekte“ und der Startseite. `aktiv`: weitere Pfade, auf
// denen der Eintrag als aktuelle Seite gilt (Unterseiten); `bereich`: alles unterhalb eines Pfades.
const NAV_ITEMS = [
  { href: '/dashboard', label: t('Startseite') },
  { href: '/projekte', label: t('Meine Projekte'), aktiv: ['/projekt-neu'] },
  { href: '/ablage', label: t('Meine Ablage') },   // Sicherung der barrierefreien Office-Dokumente inkl. Pruefbericht (11.09.2026)
  // EXPRESS-SERVICE (05.10.2026): Profis bereiten Dokumente auf — nur, wenn die Funktion an ist (/api/me user.express).
  // Runde 7 (Michael Karbe 06.10.2026): heißt „Meine Aufträge“ wie „Meine Projekte“ / „Meine Ablage“.
  { href: '/express', label: t('Meine Aufträge'), express: true, bereich: '/express', ausser: ['/express/warenkorb'] },
  // Express-Warenkorb (Zusatz 05.10.2026, Steve): fester Eintrag mit Anzahl als Text; Anzeige-Modus in der Verwaltung
  // (immer / nur mit Inhalt / aus, /api/me user.express_warenkorb). Text siehe expressKorbText().
  { href: '/express/warenkorb', label: '', korb: true },
  // Runde 10: EIN Eintrag für „Meine Prompts“ (25.08.2026) und „Meine Stammdaten“ (Quickinfo-Werkzeug 27.08.2026);
  // beide Seiten tragen die Ansichtswahl „Prompts | Stammdaten“ und gelten als „Meine Vorlagen“.
  { href: '/vorlagen', label: t('Meine Vorlagen'), aktiv: ['/prompts', '/stammdaten'] },
  // Runde 10 (Michael: „Über uns sollte in dem Dashboard bleiben, da dies ein Wettbewerbsvorteil ist“): eine
  // gemeinsame Seite, Kontakt als Abschnitt darauf (/kontakt leitet auf /ueber-uns#kontakt). Seit 25.08.2026 in der
  // Navigation, für Besucher ohne Login in OEFFENTLICH_NAV.
  { href: '/ueber-uns', label: t('Über uns und Kontakt'), aktiv: ['/kontakt'] },
  // „Hilfe“ stand hier seit Runde 1, Schritt 3 — Steve 09.10.2026: die Hilfe gehoert in den Bereich des InkluAgent (Beispiele und
  // Link „Alles, was der InkluAgent kann“), nicht in die Seitenleiste. Deshalb kein Eintrag mehr.
  // 25.09.2026 (Steve): „Verwaltung“ mit den Unterseiten Kunden, Umsatz, API, Einstellungen —
  // der Eintrag bleibt auf allen /verwaltung/…-Seiten als aktuelle Seite markiert.
  { href: '/verwaltung/kunden', label: t('Verwaltung'), admin: true, bereich: '/verwaltung' },
  // Express-Bearbeiter ohne Admin-Recht (05.10.2026): nur der Weg zu den Express-Auftraegen.
  { href: '/verwaltung/express', label: t('Express-Aufträge'), bearbeiter: true, bereich: '/verwaltung/express' },
];

// Runde 10: Seltenes unter „Konto“ (natives <details>, standardmäßig zu, offen auf diesen Seiten). Die Unterseiten der
// Einstellungen (E-Mail & Passwort, Abo & Verbrauch, Team, API-Schlüssel) gelten als „Einstellungen“. Abmelden folgt.
const KONTO_ITEMS = [
  { href: '/einstellungen', label: t('Einstellungen'), aktiv: ['/konto', '/abo', '/team', '/api-schluessel'] },
  { href: '/datensicherheit', label: t('Datensicherheit') },
];

// Navigation der oeffentlichen Seiten fuer Besucher OHNE Anmeldung
// (25.08.2026). Dieselben Ziele wie die Kopfzeile der Startseite
// (base_start.html) — Runde 10: „Preise“, „Über uns und Kontakt“. Ein Eintrag
// mehr hier erscheint auf allen oeffentlichen Seiten.
const OEFFENTLICH_NAV = [
  { href: '/preise', label: t('Preise') },
  { href: '/ueber-uns', label: t('Über uns und Kontakt') },
];

// Ist der Eintrag die aktuelle Seite? Pfad selbst, weitere Pfade (aktiv) oder ein ganzer Bereich (bereich).
function navAktuell(it, path) {
  if (it.ausser && it.ausser.includes(path)) return false;
  return path === it.href || (it.aktiv && it.aktiv.includes(path))
    || !!(it.bereich && (path === it.bereich || path.startsWith(it.bereich + '/')));
}

// Besucher ohne Konto-Kontext: Gast-Review oder oeffentliche Seite ohne Login.
function istAnonym() {
  return !!window.GUEST_MODE || (!!window.OEFFENTLICH && !currentUser);
}

// ─── Express-Warenkorb in der Navigation (Zusatz 05.10.2026) ───
function expressKorbSichtbar() {
  const k = currentUser && currentUser.express_warenkorb;
  if (!k) return false;
  return k.modus === 'immer' || (k.modus === 'mit_inhalt' && Number(k.dokumente) > 0);
}

function expressKorbText() {
  const n = Number((currentUser && currentUser.express_warenkorb && currentUser.express_warenkorb.dokumente) || 0);
  if (!n) return t('Express-Warenkorb');
  // Zahl als Text, nicht nur als Symbol; Einzahl wie auf der Express-Seite.
  const dok = n === 1 ? t('1 Dokument') : t('{n} Dokumente', { n: new Intl.NumberFormat(window.LANG || 'de').format(n) });
  return t('Express-Warenkorb: {dokumente}', { dokumente: dok });
}

// Neue Anzahl nach Hinzufügen/Entfernen: Text still austauschen (keine eigene Ansage — die Bestätigung kommt vom
// auslösenden Knopf). Muss der Eintrag erscheinen oder verschwinden (Modus „nur mit Inhalt“), wird die Leiste neu gebaut.
window.expressKorbAnzeigen = function (n) {
  if (!currentUser || !currentUser.express_warenkorb) return;
  const vorher = expressKorbSichtbar();
  currentUser.express_warenkorb.dokumente = Number(n) || 0;
  const a = document.querySelector('a[data-express-korb]');
  if (a && expressKorbSichtbar()) { a.textContent = expressKorbText(); return; }
  if (vorher !== expressKorbSichtbar()) renderSidebar();
};

function renderSidebar() {
  const host = byId('appSidebar');
  if (!host) return;
  const path = window.location.pathname;
  host.innerHTML = '';

  const brand = document.createElement('a');
  brand.className = 'app-brand';
  brand.href = istAnonym() ? '/' : '/dashboard';
  brand.innerHTML = '<span class="brand">Inklu</span>Docs';
  // a11y-Fix 13.07.2026 (axe "region", Memory todo_inkludocs_app_a11y): der
  // Marken-Link lag als einziges Element ausserhalb jeder Landmarke. Das
  // <header>-Element (role banner) macht ihn zu Landmarken-Inhalt — fuer
  // Gaeste (nur Logo) und Eingeloggte (Logo + nav) gleichermassen. Das
  // Styling haengt nur an .app-brand, der Wrapper ist layoutneutral.
  const brandHeader = document.createElement('header');
  brandHeader.appendChild(brand);
  host.appendChild(brandHeader);

  if (window.GUEST_MODE) {
    // Gast-Modus: nur das Logo in der Huelle — keine Navigation, keine
    // "Anmelden/Registrieren"-Liste (konsistent mit Demo/Production, die
    // keinen solchen Block haben). Das Brand-Logo wurde oben bereits eingefuegt.
    return;
  }

  const nav = document.createElement('nav');
  nav.setAttribute('aria-label', t('Hauptnavigation'));
  const ul = document.createElement('ul');
  ul.className = 'app-nav';

  if (window.OEFFENTLICH && !currentUser) {
    // Oeffentliche Seite ohne Login: Preise / Ueber uns und Kontakt, und an der
    // Stelle, an der sonst „Abmelden“ steht, der Weg zur Anmeldung (gleiches
    // Muster wie die Demo-Huelle demo-shell.js).
    OEFFENTLICH_NAV.forEach((it) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = it.href;
      a.textContent = it.label;
      if (path === it.href) a.setAttribute('aria-current', 'page');
      li.appendChild(a);
      ul.appendChild(li);
    });
    const liIn = document.createElement('li');
    liIn.className = 'app-nav-logout';
    const aIn = document.createElement('a');
    aIn.href = '/login';
    aIn.textContent = t('Anmelden oder registrieren');
    liIn.appendChild(aIn);
    ul.appendChild(liIn);
    nav.appendChild(ul);
    host.appendChild(nav);
    return;
  }

  NAV_ITEMS.forEach((it) => {
    if (it.admin && !(currentUser && currentUser.is_admin)) return;
    if (it.express && !(currentUser && currentUser.express)) return;
    if (it.bearbeiter && !(currentUser && currentUser.express_bearbeiter && !currentUser.is_admin)) return;
    if (it.korb && !expressKorbSichtbar()) return;
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = it.href;
    a.textContent = it.korb ? expressKorbText() : it.label;
    if (it.korb) a.setAttribute('data-express-korb', '');
    if (navAktuell(it, path)) a.setAttribute('aria-current', 'page');
    li.appendChild(a);
    ul.appendChild(li);
  });

  // „Konto“ als letzter Eintrag (Runde 10): natives <details>/<summary> — Tastatur, Zustand „aufgeklappt/zugeklappt“
  // und Fokus liefert der Browser, keine eigene Ansage. Zu, außer man steht gerade auf einer seiner Seiten.
  const liKonto = document.createElement('li');
  liKonto.className = 'app-nav-konto';
  const konto = document.createElement('details');
  konto.id = 'navKonto';
  const sum = document.createElement('summary');
  sum.textContent = t('Konto');
  konto.appendChild(sum);
  const unter = document.createElement('ul');
  unter.className = 'app-nav-unter';
  KONTO_ITEMS.forEach((it) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = it.href;
    a.textContent = it.label;
    if (navAktuell(it, path)) { a.setAttribute('aria-current', 'page'); konto.open = true; }
    li.appendChild(a);
    unter.appendChild(li);
  });

  // Abmelden als letzter Eintrag unter „Konto“.
  const liOut = document.createElement('li');
  const btn = document.createElement('button');
  btn.id = 'logoutBtn';
  btn.type = 'button';
  btn.textContent = t('Abmelden');
  btn.addEventListener('click', async () => {
    // Offene Eingaben (Alt-Text, Quickinfo, Uebersetzung) erst speichern — nach dem Abmelden ginge das nicht mehr
    // (Befund 7 der Pruefung 05.10.2026). Schlaegt es fehl, bleibt man angemeldet und hoert, warum.
    if (typeof window.alleAusstehendenSpeichern === 'function') {
      const fehler = await window.alleAusstehendenSpeichern();
      if (fehler && typeof window.speicherFehlerMelden === 'function') {
        window.speicherFehlerMelden(fehler, t('Du bist noch angemeldet, damit nichts verloren geht. Ein zweiter Klick meldet ohne Speichern ab.'));
        return;
      }
    }
    await fetch('/api/logout', { method: 'POST' });
    window.location.href = '/';
  });
  liOut.appendChild(btn);
  unter.appendChild(liOut);
  konto.appendChild(unter);
  liKonto.appendChild(konto);
  // Agentenansicht wie ChatGPT (Michael Karbe, Feedback 20261012 - 1, Punkt 6): „Gespräche“ in DIESER Seitenleiste, keine zweite
  const liGespraeche = seitenleisteGespraecheBauen(path);
  if (liGespraeche) ul.appendChild(liGespraeche);
  ul.appendChild(liKonto);

  nav.appendChild(ul);
  host.appendChild(nav);
  ansichtUmschalter(host);
}

// ─── Gespräche in der Seitenleiste (Agentenansicht wie ChatGPT/Claude, Schalter AGENT_ARBEITSFENSTER) ───
// Nur in der Agentenansicht: Überschrift „Gespräche“, Knopf „Neues Gespräch“, die acht neuesten Gespräche (das laufende mit
// aria-current) und „Alle Gespräche“ (Seite /gespraeche). Ein Gespräch anklicken setzt es fort — wie „Hier weitermachen“: weiter
// im letzten Projekt des Gesprächs, der InkluAgent offen, Fokus auf der letzten Nachricht. Ohne Skript oder mit Strg/Cmd führt der
// Link auf die Lese-Seite. In der manuellen Ansicht bleibt alles beim InkluAgent unten auf der Seite.
function seitenleisteGespraecheAn() {
  return !!(currentUser && currentUser.oberflaeche === 'agent' && window.INKLUAGENT && window.INKLUAGENT.arbeitsfenster
            && window.INKLUAGENT.gespraeche);
}
function seitenleisteGespraecheBauen(path) {
  if (!seitenleisteGespraecheAn()) return null;
  document.documentElement.classList.add('mit-seitenleiste-gespraeche');
  const li = document.createElement('li');
  li.className = 'app-nav-gespraeche';
  // Benannter Bereich statt Ueberschrift (Steve 12.10.2026): eine Ueberschrift 2 in der Seitenleiste stand vor der Ueberschrift 1
  // der Seite — wer per Ueberschriften navigiert, landete zuerst hier. Erreichbar ueber die Landmarken („Gespräche“).
  const bereich = document.createElement('section');
  bereich.setAttribute('aria-labelledby', 'navGespraecheTitel');
  const h = document.createElement('p');
  h.id = 'navGespraecheTitel';
  h.className = 'app-nav-titel';
  h.textContent = t('Gespräche');
  const neu = document.createElement('button');
  neu.type = 'button';
  neu.id = 'navNeuesGespraech';
  neu.textContent = t('Neues Gespräch');
  neu.addEventListener('click', seitenleisteNeuesGespraech);
  const liste = document.createElement('ul');
  liste.id = 'navGespraecheListe';
  liste.className = 'app-nav-gespraeche-liste';
  liste.setAttribute('aria-labelledby', 'navGespraecheTitel');
  // Pruefung 13.10.2026 (B20): der Link zeigt auf die Lese-Seite, ein Klick setzt das Gespraech aber fort — das sagt jeder Eintrag
  // als Beschreibung („Link, Wie mache ich …, setzt das Gespräch fort“); mit Strg/Cmd bleibt es die Lese-Seite.
  // Runde 2 (A-N7): hidden statt visually-hidden — aria-describedby liest auch verborgene Elemente, aber mit den VoiceOver-Pfeilen
  // stand der Satz sonst noch einmal lose nach der Liste (gemessen: im Barrierefreiheitsbaum, nicht ignoriert)
  const hinweis = document.createElement('span');
  hinweis.id = 'navGespraecheHinweis';
  hinweis.hidden = true;
  hinweis.textContent = t('Setzt das Gespräch fort.');
  const alle = document.createElement('a');
  alle.href = '/gespraeche';
  alle.className = 'app-nav-alle';
  alle.textContent = t('Alle Gespräche');
  if (path === '/gespraeche') alle.setAttribute('aria-current', 'page');
  bereich.appendChild(h);
  bereich.appendChild(neu);
  bereich.appendChild(liste);
  bereich.appendChild(hinweis);
  bereich.appendChild(alle);
  li.appendChild(bereich);
  seitenleisteGespraecheLaden();
  return li;
}
async function seitenleisteGespraecheLaden() {
  if (!seitenleisteGespraecheAn()) return;
  let daten;
  try {
    const r = await fetch('/api/agent/gespraeche?anzahl=8');
    if (!r.ok) return;
    daten = await r.json();
  } catch (e) { return; }
  const liste = byId('navGespraecheListe');
  if (!liste) return;
  const eintraege = (daten.gespraeche || []).map((g) => ({ id: Number(g.id), aktuell: !!g.aktuell,
    titel: g.titel || t('Gespräch vom {datum}', { datum: g.datum || '' }) })).filter((g) => g.id);
  // Pruefung 13.10.2026 (B8): nur neu bauen, wenn sich die Liste geaendert hat — sie laedt nach jeder Antwort, und ein Neubau unter
  // dem VoiceOver-Cursor oder dem Tastaturfokus liess beide an den Anfang springen. Wird doch neu gebaut, kommt der Fokus auf
  // dasselbe Gespraech zurueck.
  const stand = JSON.stringify(eintraege);
  if (liste._stand === stand) return;
  liste._stand = stand;
  const ae = document.activeElement;
  const fokusId = (ae && liste.contains(ae) && ae.dataset) ? ae.dataset.gespraech : null;
  liste.textContent = '';
  eintraege.forEach((g) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '/gespraeche/' + g.id;
    a.dataset.gespraech = String(g.id);
    // Gekuerzt wird der Text, nicht der Link (16.10.2026): mit der Kuerzung am Link schien die dritte Zeile durch den Innenabstand und
    // lag ueber dem naechsten Eintrag. Der volle Titel bleibt der Name des Links (VoiceOver liest ihn ganz).
    const text = document.createElement('span');
    text.className = 'app-nav-gespraech-text';
    text.textContent = g.titel;
    a.appendChild(text);
    a.setAttribute('aria-describedby', 'navGespraecheHinweis');
    if (g.aktuell) a.setAttribute('aria-current', 'true');
    a.addEventListener('click', (e) => seitenleisteGespraechFortsetzen(e, g.id, a));
    li.appendChild(a);
    liste.appendChild(li);
  });
  if (fokusId) {
    const wieder = liste.querySelector('a[data-gespraech="' + fokusId + '"]') || byId('navNeuesGespraech');
    if (wieder) wieder.focus();
  }
}
window.seitenleisteGespraecheLaden = seitenleisteGespraecheLaden;
// Waehrend der InkluAgent an einer Antwort arbeitet, wechselt kein Gespraech (Pruefung 13.10.2026, K3): die Antwort gehoert zum
// Gespraech der Frage. inkluagentWechselErlaubt (inkluagent.js) sagt sonst, warum nicht.
function gespraechWechselErlaubt() {
  return typeof window.inkluagentWechselErlaubt !== 'function' || window.inkluagentWechselErlaubt();
}
async function seitenleisteGespraechFortsetzen(e, id, a) {
  if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  if (!gespraechWechselErlaubt()) return;
  if (typeof window.alleAusstehendenSpeichern === 'function') {
    const fehler = await window.alleAusstehendenSpeichern();
    if (fehler && typeof window.speicherFehlerMelden === 'function') { window.speicherFehlerMelden(fehler); return; }
  }
  try {
    const r = await fetch('/api/agent/gespraeche/' + id + '/fortsetzen', { method: 'POST' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    const ziel = String(d.weiter || '');
    window.location.href = (eigenerPfad(ziel) && /^\/(app|dashboard)\?/.test(ziel)) ? ziel : a.href;
  } catch (err) {
    window.location.href = a.href;
  }
}
async function seitenleisteNeuesGespraech() {
  if (!gespraechWechselErlaubt()) return;
  // Mit dem InkluAgent auf der Seite: dessen „Neues Gespräch“ (leert den Verlauf, Fokus ins Eingabefeld)
  const imAgenten = byId('inkluagentNeuesGespraech');
  if (imAgenten && typeof window.inkluagentNeuesGespraech === 'function') {
    const panel = byId('inkluagentPanel');
    if (panel && panel.hidden && typeof window.inkluagentOpen === 'function') await window.inkluagentOpen(window._inkluagentProjectId, false, true);
    await window.inkluagentNeuesGespraech();
    return;
  }
  try {
    const r = await fetch('/api/agent/gespraech/neu', { method: 'POST' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    window.location.href = '/dashboard?agent=offen';
  } catch (e) {
    announce(t('Das ging nicht. Bitte erneut versuchen.'));
  }
}

// ─── Ansicht je Konto: schneller Umschalter (InkluAgent-Ausbau Runde 1, Schritt 5) ───
// Unter der Navigation, nur mit Schalter funktionen.AGENT_ANSICHT (/api/me user.oberflaeche_waehlbar). Der Knopf sagt, was
// passiert („Zur Agentenansicht wechseln“ / „Zur manuellen Ansicht wechseln“). Danach laedt die Seite neu, der Fokus steht
// auf der Hauptueberschrift. Hauptort der Einstellung bleibt die Seite „Einstellungen“.
function ansichtUmschalter(host) {
  if (!currentUser || !currentUser.oberflaeche_waehlbar) return;
  const agent = currentUser.oberflaeche === 'agent';
  const p = document.createElement('p');
  p.className = 'app-ansicht-wechsel';
  const b = document.createElement('button');
  b.type = 'button';
  b.id = 'ansichtWechselBtn';
  b.className = 'btn btn-secondary';
  b.textContent = agent ? t('Zur manuellen Ansicht wechseln') : t('Zur Agentenansicht wechseln');
  b.addEventListener('click', async () => {
    // Offene Eingaben erst speichern (wie beim Abmelden) — das Neuladen ginge sonst darueber hinweg
    if (typeof window.alleAusstehendenSpeichern === 'function') {
      const fehler = await window.alleAusstehendenSpeichern();
      if (fehler && typeof window.speicherFehlerMelden === 'function') { window.speicherFehlerMelden(fehler); return; }
    }
    const r = await fetch('/api/me/oberflaeche', { method: 'PUT', headers: { 'Content-Type': 'application/json' },
                                                   body: JSON.stringify({ oberflaeche: agent ? 'klassisch' : 'agent' }) });
    if (!r.ok) { if (typeof announce === 'function') announce(t('Die Ansicht konnte nicht gespeichert werden. Bitte versuch es noch einmal.')); return; }
    try { sessionStorage.setItem('ansichtGewechselt', '1'); } catch (e) { /* ohne Speicher: kein Fokus-Sprung */ }
    window.location.reload();
  });
  p.appendChild(b);
  host.appendChild(p);
}

// Nach dem Wechsel: Fokus auf die Hauptueberschrift (die Projektseite baut sie erst nach dem Laden der Daten). Nur im Hauptfenster
// (Pruefung 13.10.2026, HK8/B19): die eingebettete Seite im Arbeitsfenster teilt sich sessionStorage mit ihm — lud sie schneller, nahm
// sie den Merker und zog den Fokus in den Rahmen. Fokus ueber fokusAuf (kein dauerhaftes tabindex an der Ueberschrift).
function ansichtFokusNachWechsel() {
  if (imRahmen()) return;
  let gemerkt = false;
  try { gemerkt = sessionStorage.getItem('ansichtGewechselt') === '1'; sessionStorage.removeItem('ansichtGewechselt'); } catch (e) { /* egal */ }
  if (!gemerkt) return;
  let n = 0;
  const t0 = setInterval(() => {
    const h = document.querySelector('#main h1');
    if (h || ++n > 50) {
      clearInterval(t0);
      if (h) fokusAuf(h);
    }
  }, 100);
}

// Sprunglinks oben auf der Seite (Pruefung 13.10.2026, B19): ein Ziel ohne eigenes tabindex (der Bereich des InkluAgent) bekommt den
// Fokus ueber fokusAuf — vorher trug es dafuer dauerhaft tabindex="-1". Ziele mit eigenem tabindex (#main) behandelt der Browser wie
// bisher.
function sprunglinksEinrichten() {
  // „Zurück zum Gespräch“ im Rahmen (#zumGespraech) fuehrt ins Elternfenster — eingebettetEinrichten
  document.querySelectorAll('a.dash-skip[href^="#"]:not(#zumGespraech)').forEach((a) => {
    a.addEventListener('click', (e) => {
      if (e.defaultPrevented) return;
      const ziel = document.getElementById(a.getAttribute('href').slice(1));
      if (!ziel || ziel.hasAttribute('tabindex')) return;
      e.preventDefault();
      fokusAuf(ziel);
    });
  });
}

// Rechtliche Footer-Links (Impressum · Datenschutz · Nutzungsbedingungen) zentral
// in jede Dashboard-Fußzeile rendern (10.06.2026, Michael-Wunsch „Fußzeile überall
// identisch"). Vorher standen die Links statisch und uneinheitlich in jeder Seite
// (mal zwei, mal drei) und /app hatte sogar eine ganz andere Fußzeile. Jetzt eine
// einzige Quelle -> alle eingeloggten Seiten (inkl. /app) zeigen garantiert
// dieselbe Fußzeile, und sie kann nicht mehr auseinanderdriften.
// Die Links werden als ERSTES Element der Fußzeile eingesetzt (vor DSGVO-Hinweis
// und Unterstützungsblock), damit die Reihenfolge der bereits abgenommenen
// Übersichtsseiten erhalten bleibt: Links -> DSGVO-Hinweis -> Unterstützung.
// 25.08.2026: Die -app-Sichten setzen einen Login voraus (sonst Umleitung
// zur Anmeldung). Fuer Besucher ohne Login (Gast-Review) nennt `oeffentlich`
// das frei erreichbare Ziel. Beschriftungen und Reihenfolge = _fusszeile.html
// (ui_geruest.py vergleicht beide Listen).
const LEGAL_LINKS = [
  { href: '/impressum-app', oeffentlich: '/impressum', label: t('Impressum') },
  { href: '/datensicherheit', oeffentlich: '/datenschutz', label: t('Datenschutz') },
  { href: '/nutzungsbedingungen-app', oeffentlich: '/nutzungsbedingungen', label: t('Nutzungsbedingungen') },
  // Preise: kam am 31.08.2026 mit dem Rollout in die oeffentliche Fusszeile
  // (_fusszeile.html). Beide Listen muessen synchron bleiben
  // (docs/SEITENGERUEST.md, geprueft von ui_geruest.py) — darum steht der
  // Link auch hier, an derselben Stelle. Keine -app-Variante noetig.
  { href: '/preise', label: t('Preise') },
  { href: '/widerruf-app', oeffentlich: '/widerruf', label: t('Widerrufsbelehrung') },
  // § 312k BGB verlangt, dass der Kuendigungsknopf staendig verfuegbar und
  // leicht erreichbar ist — deshalb in JEDER Fusszeile, nicht nur im Abo-Bereich.
  { href: '/kuendigen', label: t('Vertrag kündigen') },
  // Widerrufsfunktion nach § 356a BGB: waehrend der Widerrufsfrist dauerhaft
  // gut sichtbar und leicht zugaenglich — deshalb ebenfalls in jeder Fusszeile.
  { href: '/widerrufen', label: t('Vertrag widerrufen') },
  // Kontakt und Über uns: seit 25.08.2026 in der Seitenleiste (NAV_ITEMS /
  // OEFFENTLICH_NAV, seit Runde 10 ein Eintrag „Über uns und Kontakt“), deshalb hier nicht mehr doppelt.
];
function renderLegalLinks() {
  document.querySelectorAll('.dash-footer').forEach((footer) => {
    // idempotent — und: base_oeffentlich.html liefert die Links bereits
    // serverseitig (ohne JavaScript sichtbar), dann bleibt hier alles so.
    if (footer.querySelector('.dash-legal-links')) return;
    const wrap = document.createElement('div');
    wrap.className = 'dash-legal-links';
    LEGAL_LINKS.forEach((it, i) => {
      // '\u00a0· ': Punkt bleibt am vorigen Link, danach darf umbrochen werden (25.09.2026 — mit
      // geschütztem Leerzeichen auf beiden Seiten ragte die Zeile auf dem Handy über den Rand).
      if (i > 0) wrap.appendChild(document.createTextNode('\u00a0· '));
      const a = document.createElement('a');
      a.href = (istAnonym() && it.oeffentlich) ? it.oeffentlich : it.href;
      a.textContent = it.label;
      wrap.appendChild(a);
    });
    footer.insertBefore(wrap, footer.firstChild);
  });
}

// Spenden-Hinweis: am 08.08.2026 aus der App ENTFERNT (Steve).
// Seit dem Abomodell zahlt die Kundschaft hier fuer den Dienst — ein
// Spendenaufruf daneben passt nicht und wirkt wie eine zweite Kasse.
// Auf der oeffentlichen Demo (demo-shell.js) bleibt er, dort ist das
// Angebot tatsaechlich kostenlos.

// DSGVO-Hinweis in jede Dashboard-Fußzeile (zentral gepflegt, gleicher Text wie auf /app).
// Idempotent. /app hat .legal-footer (eigener Hinweis) und wird hier nicht getroffen.
// 11.09.2026: Chat-Assistent laeuft seit 10.09. ebenfalls auf Gemini (Vertex) — der Hinweis nannte noch Claude/Bedrock.
function renderLegalNote() {
  document.querySelectorAll('.dash-footer').forEach((footer) => {
    if (footer.querySelector('.dsgvo-note')) return;
    const p = document.createElement('p');
    p.className = 'dsgvo-note';
    const strong = document.createElement('strong');
    strong.textContent = t('DSGVO-konform – Hosting und Datenhaltung in Deutschland.');
    p.appendChild(strong);
    p.appendChild(document.createTextNode(t(' Hosting bei Hetzner Online (Falkenstein, Deutschland). Bildbeschreibungen, Quickinfos und der Chat-Assistent nutzen das Modell Gemini von Google über Google Vertex AI. Google ist vertraglich als Auftragsverarbeiter gebunden und nutzt deine Inhalte nicht zum Training. Einzelheiten in unserer Datenschutzerklärung.')));
    footer.appendChild(p);
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  sprunglinksEinrichten();
  eingebettetEinrichten();   // Arbeitsfenster der Agentenansicht (13.10.2026)
  hinfuehrenBeimLaden();     // Begleiter: Ziel auf dieser Seite
  await loadCurrentUser();   // setzt currentUser, Begrüßung, Tageslimit
  renderSidebar();
  ansichtFokusNachWechsel();
  renderLegalLinks();
  renderLegalNote();
});
