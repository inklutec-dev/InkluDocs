/* ─────────────────────────────────────────────────────────────────────────
   demo-shell.js — gemeinsame Hülle der Demo-Seiten (Sidebar + Fußzeile) plus
   ein Lader für die Rechtstexte. EINE Quelle für Startseite UND Rechtsseiten,
   damit Navigation und Footer nicht auseinanderdriften (Single Source).

   Vorbild ist dashboard.js (renderSidebar/renderFooter) aus dem eingeloggten
   Werkzeug — hier bewusst STATISCH/anonym für die Demo: keine Projekte/
   Einstellungen/Abmelden, dafür die Conversion „Anmelden oder registrieren".
   Die Rechtsseiten der Demo nutzen die öffentlichen Routen (/impressum, …),
   die ohne Login erreichbar sind; die -app-Varianten des Werkzeugs setzen
   einen Login voraus und sind in der anonymen Demo nicht nutzbar. 15.06.2026.

   04.07.2026: fünfsprachig über window.I18N (liefert base_demo.html) und den
   t()-Helfer unten — dieselben Fußzeilen-msgids wie dashboard.js, damit die
   Übersetzungen nicht doppelt gepflegt werden.
   ───────────────────────────────────────────────────────────────────────── */

// Übersetzungs-Helfer — identisch zu dashboard.js (das die Demo-Seiten nicht
// laden). Fallback ist der deutsche Ausgangstext, Platzhalter im Muster {name}.
function t(s, params) {
  var out = (window.I18N && window.I18N[s]) || s;
  if (params) {
    Object.keys(params).forEach(function (k) {
      out = out.replace('{' + k + '}', params[k]);
    });
  }
  return out;
}

(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  // Sidebar-Navigation der Demo. „Datensicherheit" führt auf die In-Rahmen-Sicht
  // /demo-datenschutz (zeigt den öffentlichen Datenschutz-Text mit Sidebar).
  // 25.08.2026 (Steve): Die Demo ist das Schaufenster — Preise, Kontakt und
  // Über uns fuehren auf die oeffentlichen Seiten des Werkzeugs (eine Quelle,
  // keine Kopien in der Demo), wie „Anmelden oder registrieren“ unten.
  const NAV = [
    { href: "/", label: t("Demo testen") },
    { href: "/demo-datenschutz", label: t("Datensicherheit") },
    { href: "https://inkludocs.de/preise", label: t("Preise") },
    { href: "https://inkludocs.de/kontakt", label: t("Kontakt") },
    { href: "https://inkludocs.de/ueber-uns", label: t("Über uns") },
  ];

  function renderSidebar() {
    const host = $("appSidebar");
    if (!host) return;
    const path = window.location.pathname;
    host.innerHTML = "";

    const brand = document.createElement("a");
    brand.className = "app-brand";
    brand.href = "/";
    brand.innerHTML = '<span class="brand">Inklu</span>Docs';
    host.appendChild(brand);

    const nav = document.createElement("nav");
    nav.setAttribute("aria-label", t("Hauptnavigation"));
    const ul = document.createElement("ul");
    ul.className = "app-nav";
    NAV.forEach((it) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = it.href;
      a.textContent = it.label;
      if (path === it.href) a.setAttribute("aria-current", "page");
      li.appendChild(a);
      ul.appendChild(li);
    });
    // Conversion an der Stelle, an der im Dashboard „Abmelden" steht.
    const liOut = document.createElement("li");
    liOut.className = "app-nav-logout";
    const aOut = document.createElement("a");
    aOut.href = "https://inkludocs.de/login";  // 02.09.2026: / ist die Startseite, Anmeldung unter /login
    aOut.textContent = t("Anmelden oder registrieren");
    liOut.appendChild(aOut);
    ul.appendChild(liOut);

    nav.appendChild(ul);
    host.appendChild(nav);
  }

  // Fußzeile identisch zum Werkzeug (rechtliche Links, DSGVO-Hinweis, Spenden),
  // nur mit den öffentlichen Demo-Routen verlinkt. Idempotent. Die Texte nutzen
  // dieselben msgids wie dashboard.js (eine Übersetzung, zwei Renderer).
  function renderFooter() {
    document.querySelectorAll(".dash-footer").forEach((footer) => {
      if (footer.querySelector(".dash-legal-links")) return;
      footer.innerHTML =
        '<div class="dash-legal-links">' +
          '<a href="/demo-impressum">' + t("Impressum") + '</a> · ' +
          '<a href="/demo-datenschutz">' + t("Datenschutz") + '</a> · ' +
          '<a href="/demo-nutzungsbedingungen">' + t("Nutzungsbedingungen") + '</a>' +
        '</div>' +
        '<p class="dsgvo-note"><strong>' + t("DSGVO-konform – Verarbeitung in der EU.") + '</strong>' +
          t(" Hosting bei Hetzner Online (Falkenstein, Deutschland). Die KI-Verarbeitung erfolgt über Amazon Bedrock (Modell Claude von Anthropic) in Rechenzentren innerhalb der EU; Amazon Bedrock gibt keine Inhalte an den Modellanbieter weiter und nutzt sie nicht zum Training. Einzelheiten in unserer Datenschutzerklärung.") +
        '</p>' +
        '<div class="dash-support">' +
          '<p class="dash-support-text">' + t("Diese Demo ist kostenlos und wird laufend weiterentwickelt. Wenn du das Projekt unterstützen möchtest, freuen wir uns über einen freiwilligen Beitrag.") + '</p>' +
          '<a class="dash-support-link" href="https://www.paypal.com/donate?business=steve.weidel%40gmail.com&item_name=InkluDocs+-+Freiwilliger+Beitrag&currency_code=EUR" ' +
            'target="_blank" rel="noopener" aria-label="' + t("InkluDocs per PayPal unterstützen, öffnet in neuem Tab") + '">' + t("InkluDocs per PayPal unterstützen") + '</a>' +
          '<p class="dash-support-note">' + t("Ihr Beitrag hilft, Barrierefreiheit im Web voranzubringen.") + '</p>' +
        '</div>';
    });
  }

  // Lädt den öffentlichen Rechtstext (Single Source) und bettet seinen
  // .legal-container in den Demo-Rahmen ein — exakt das Muster der -app-Seiten
  // im Werkzeug. So bleibt die Sidebar sichtbar und der Text wird nicht doppelt
  // gepflegt.
  async function loadLegalInline(sourceUrl, label) {
    const target = $("legalContent");
    if (!target) return;
    try {
      const res = await fetch(sourceUrl, { credentials: "same-origin" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const doc = new DOMParser().parseFromString(await res.text(), "text/html");
      const container = doc.querySelector(".legal-container");
      if (!container) throw new Error("Inhalt nicht gefunden (.legal-container fehlt)");
      // Doppelte H1 und der eigene Auth-Subfooter der öffentlichen Seite raus —
      // die H1 steht schon im Rahmen, die Fußzeile liefert die Demo-Hülle.
      container.querySelectorAll("h1").forEach((h) => h.remove());
      container.querySelectorAll("nav.auth-links").forEach((n) => n.remove());
      target.innerHTML = "";
      target.appendChild(container);
      target.setAttribute("aria-busy", "false");
    } catch (err) {
      target.innerHTML = '<p style="color:var(--error,#c33);">' +
        t("{was} konnte nicht geladen werden ({fehler}). Bitte die öffentliche Seite aufrufen:",
          { was: label, fehler: (err && err.message ? err.message : t("Fehler")) }) +
        ' <a href="' + sourceUrl + '">' + sourceUrl + '</a></p>';
      target.setAttribute("aria-busy", "false");
    }
  }

  // Beim Laden Hülle aufbauen. Den jeweiligen Rechtstext lädt die einzelne Seite.
  renderSidebar();
  renderFooter();
  window.DemoShell = { loadLegalInline: loadLegalInline };
})();
