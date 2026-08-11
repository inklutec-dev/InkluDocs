// Gemeinsame Logik für alle Dashboard-Seiten (App-Shell): Helfer, Login-Status,
// Navigationsleiste (mit aria-current), Credit-/Tageslimit-Anzeige,
// Spenden-Footer, Abmelden.

const byId = (id) => document.getElementById(id);

// i18n-Helfer: schlaegt Text in der vom Server gelieferten Uebersetzungstabelle
// (window.I18N) nach. Ohne Tabelle (z.B. noch nicht migrierte Seite) faellt er
// auf den deutschen Ausgangstext zurueck. Optionale {platzhalter} werden ersetzt.
function t(s, params) {
  var out = (window.I18N && window.I18N[s]) || s;
  if (params) {
    Object.keys(params).forEach(function (k) {
      out = out.replace('{' + k + '}', params[k]);
    });
  }
  return out;
}

function announce(msg) {
  const el = byId('liveRegion');
  if (el) el.textContent = msg;
}

// Datum robust formatieren (kein Date()-Parsing wegen Safari-Eigenheiten bei
// SQLite-Zeitstempeln "YYYY-MM-DD HH:MM:SS").
function formatDate(s) {
  if (!s) return '';
  const p = String(s).substring(0, 10).split('-');
  return p.length === 3 ? `${p[2]}.${p[1]}.${p[0]}` : String(s);
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
  if (abo.pakete_rest > 0) {
    lines.push(t('Zusatz-Credits: {anzahl} verfügbar.', { anzahl: abo.pakete_rest }));
  }
  return lines;
}

let currentUser = null;

async function loadCurrentUser() {
  if (window.GUEST_MODE) { currentUser = null; return null; }  // Gast: kein /api/me, keine Weiterleitung
  const res = await fetch('/api/me');
  if (res.status === 401) { window.location.href = '/'; return null; }
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
const NAV_ITEMS = [
  { href: '/dashboard', label: t('Startseite') },
  { href: '/projekt-neu', label: t('Neues Projekt anlegen') },
  { href: '/projekte', label: t('Meine Projekte') },
  { href: '/prompts', label: t('Eigene Prompts') },
  { href: '/einstellungen', label: t('Einstellungen') },
  { href: '/datensicherheit', label: t('Datensicherheit') },
  { href: '/benutzer', label: t('Benutzerverwaltung'), admin: true },
];

function renderSidebar() {
  const host = byId('appSidebar');
  if (!host) return;
  const path = window.location.pathname;
  host.innerHTML = '';

  const brand = document.createElement('a');
  brand.className = 'app-brand';
  brand.href = window.GUEST_MODE ? '/' : '/dashboard';
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
  NAV_ITEMS.forEach((it) => {
    if (it.admin && !(currentUser && currentUser.is_admin)) return;
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = it.href;
    a.textContent = it.label;
    if (path === it.href) a.setAttribute('aria-current', 'page');
    li.appendChild(a);
    ul.appendChild(li);
  });

  // Abmelden als letzter Eintrag der Navigationsliste (kein separater Abschnitt).
  const liOut = document.createElement('li');
  liOut.className = 'app-nav-logout';
  const btn = document.createElement('button');
  btn.id = 'logoutBtn';
  btn.type = 'button';
  btn.textContent = t('Abmelden');
  btn.addEventListener('click', async () => {
    await fetch('/api/logout', { method: 'POST' });
    window.location.href = '/';
  });
  liOut.appendChild(btn);
  ul.appendChild(liOut);

  nav.appendChild(ul);
  host.appendChild(nav);
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
const LEGAL_LINKS = [
  { href: '/impressum-app', label: t('Impressum') },
  { href: '/datensicherheit', label: t('Datenschutz') },
  { href: '/nutzungsbedingungen-app', label: t('Nutzungsbedingungen') },
  { href: '/widerruf-app', label: t('Widerrufsbelehrung') },
  // § 312k BGB verlangt, dass der Kuendigungsknopf staendig verfuegbar und
  // leicht erreichbar ist — deshalb in JEDER Fusszeile, nicht nur im Abo-Bereich.
  { href: '/kuendigen', label: t('Vertrag kündigen') },
];
function renderLegalLinks() {
  document.querySelectorAll('.dash-footer').forEach((footer) => {
    if (footer.querySelector('.dash-legal-links')) return;   // idempotent
    const wrap = document.createElement('div');
    wrap.className = 'dash-legal-links';
    LEGAL_LINKS.forEach((it, i) => {
      if (i > 0) wrap.appendChild(document.createTextNode(' · '));
      const a = document.createElement('a');
      a.href = it.href;
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
function renderLegalNote() {
  document.querySelectorAll('.dash-footer').forEach((footer) => {
    if (footer.querySelector('.dsgvo-note')) return;
    const p = document.createElement('p');
    p.className = 'dsgvo-note';
    const strong = document.createElement('strong');
    strong.textContent = t('DSGVO-konform – Verarbeitung in der EU.');
    p.appendChild(strong);
    p.appendChild(document.createTextNode(t(' Hosting bei Hetzner Online (Falkenstein, Deutschland). Die KI-Verarbeitung erfolgt über Amazon Bedrock (Modell Claude von Anthropic) in Rechenzentren innerhalb der EU; Amazon Bedrock gibt keine Inhalte an den Modellanbieter weiter und nutzt sie nicht zum Training. Einzelheiten in unserer Datenschutzerklärung.')));
    footer.appendChild(p);
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  await loadCurrentUser();   // setzt currentUser, Begrüßung, Tageslimit
  renderSidebar();
  renderLegalLinks();
  renderLegalNote();

});
