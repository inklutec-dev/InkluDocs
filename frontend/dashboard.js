// Gemeinsame Logik für alle Dashboard-Seiten (App-Shell): Helfer, Login-Status,
// Navigationsleiste (mit aria-current), Tageslimit, Spenden-Footer, Abmelden.

const byId = (id) => document.getElementById(id);

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

let currentUser = null;

async function loadCurrentUser() {
  const res = await fetch('/api/me');
  if (res.status === 401) { window.location.href = '/'; return null; }
  const data = await res.json();
  currentUser = data.user || null;
  const greeting = byId('greeting');
  if (greeting && currentUser) {
    greeting.textContent = currentUser.display_name ? `Hallo, ${currentUser.display_name}` : 'Willkommen';
  }
  // Tageslimit auf der Startseite anzeigen (falls das Element vorhanden ist)
  const limitInfo = byId('dailyLimitInfo');
  const dl = data.daily_limit;
  if (limitInfo && dl) {
    limitInfo.textContent = `Heute ${dl.used} von ${dl.limit} Alt-Texten generiert – noch ${dl.remaining} übrig.`;
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
  { href: '/dashboard', label: 'Startseite' },
  { href: '/projekt-neu', label: 'Neues Projekt anlegen' },
  { href: '/projekte', label: 'Meine Projekte' },
  { href: '/einstellungen', label: 'Einstellungen' },
  { href: '/datensicherheit', label: 'Datensicherheit' },
  { href: '/benutzer', label: 'Benutzerverwaltung', admin: true },
];

function renderSidebar() {
  const host = byId('appSidebar');
  if (!host) return;
  const path = window.location.pathname;
  host.innerHTML = '';

  const brand = document.createElement('a');
  brand.className = 'app-brand';
  brand.href = '/dashboard';
  brand.innerHTML = '<span class="brand">Inklu</span>Docs';
  host.appendChild(brand);

  const nav = document.createElement('nav');
  nav.setAttribute('aria-label', 'Hauptnavigation');
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
  btn.textContent = 'Abmelden';
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
  { href: '/impressum-app', label: 'Impressum' },
  { href: '/datensicherheit', label: 'Datenschutz' },
  { href: '/nutzungsbedingungen-app', label: 'Nutzungsbedingungen' },
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

// Spenden-Hinweis dezent im Fussbereich (auf jeder Dashboard-Seite).
function renderSupportFooter() {
  document.querySelectorAll('.dash-footer').forEach((footer) => {
    if (footer.querySelector('.dash-support')) return;
    const sec = document.createElement('div');
    sec.className = 'dash-support';
    const p = document.createElement('p');
    p.className = 'dash-support-text';
    p.textContent = 'InkluDocs ist kostenlos und wird laufend weiterentwickelt. Wenn du das Projekt unterstützen möchtest, freuen wir uns über einen freiwilligen Beitrag.';
    const a = document.createElement('a');
    a.className = 'dash-support-link';
    a.href = 'https://www.paypal.com/donate?business=steve.weidel%40gmail.com&item_name=InkluDocs+-+Freiwilliger+Beitrag&currency_code=EUR';
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = 'InkluDocs per PayPal unterstützen';
    a.setAttribute('aria-label', 'InkluDocs per PayPal unterstützen, öffnet in neuem Tab');
    const note = document.createElement('p');
    note.className = 'dash-support-note';
    note.textContent = 'Ihr Beitrag hilft, Barrierefreiheit im Web voranzubringen.';
    sec.appendChild(p);
    sec.appendChild(a);
    sec.appendChild(note);
    footer.appendChild(sec);
  });
}

// DSGVO-Hinweis in jede Dashboard-Fußzeile (zentral gepflegt, gleicher Text wie auf /app).
// Idempotent. /app hat .legal-footer (eigener Hinweis) und wird hier nicht getroffen.
function renderLegalNote() {
  document.querySelectorAll('.dash-footer').forEach((footer) => {
    if (footer.querySelector('.dsgvo-note')) return;
    const p = document.createElement('p');
    p.className = 'dsgvo-note';
    const strong = document.createElement('strong');
    strong.textContent = 'DSGVO-konform – Verarbeitung in der EU.';
    p.appendChild(strong);
    p.appendChild(document.createTextNode(' Hosting bei Hetzner Online (Falkenstein, Deutschland). Die KI-Verarbeitung erfolgt über Amazon Bedrock (Modell Claude von Anthropic) in Rechenzentren innerhalb der EU; Amazon Bedrock gibt keine Inhalte an den Modellanbieter weiter und nutzt sie nicht zum Training. Einzelheiten in unserer Datenschutzerklärung.'));
    footer.appendChild(p);
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  await loadCurrentUser();   // setzt currentUser, Begrüßung, Tageslimit
  renderSidebar();
  renderLegalLinks();
  renderLegalNote();
  renderSupportFooter();
});
