# Seitengerüste in InkluDocs

Stand 25.08.2026. Diese Datei beschreibt, aus welchen Gerüsten (Jinja2-Basis-
Templates) die Seiten der App gebaut werden, welches Gerüst für welche Seite
gilt und wie man eine neue Seite anlegt, ohne das Gefüge zu verletzen.

## Die drei Gerüste

| Gerüst | Für wen | Hülle rendert | Login |
|---|---|---|---|
| `base_app.html` | eingeloggte App-Seiten (Dashboard, Projekte, Abo, Einstellungen …) | `dashboard.js` | Pflicht — `/api/me` 401 leitet zur Anmeldung |
| `base_oeffentlich.html` | öffentliche Inhaltsseiten (Preise, Kontakt, Über uns, AVV, Impressum, Datenschutz, Nutzungsbedingungen, Widerrufsbelehrung, Kündigen, Widerrufen) | `dashboard.js` im Modus `window.OEFFENTLICH` | optional — eingeloggt: App-Navigation, sonst öffentliche Navigation, **nie** eine Weiterleitung |
| `base_demo.html` | Demo-Instanz (demo.inkludocs.de) | `demo-shell.js` | keiner |
| `base_start.html` | Startseite (`/`) und die vier Login-Karten (`/login`, `/register`, `/forgot`, `/reset`) | nichts — Kopfzeile serverseitig | Startseite: Eingeloggte werden ins Dashboard geleitet |

Dazu `base.html` als nacktes Wurzel-Template (Kopf, Stylesheet, Titel mit
Staging-Zusatz) — es wird von `base_oeffentlich.html` und `base_start.html`
erweitert.

Die ersten drei Hüllen sehen gleich aus: Skip-Link, Seitenleiste links
(`#appSidebar`), Hauptfläche `main#main.dash-main` mit der H1, Fußzeile
`.dash-footer`. Stylesheets: `style.css` + `dashboard.css`.

`base_start.html` (seit 02.09.2026) ist die Ausnahme: **keine Seitenleiste**,
sondern eine schlanke, serverseitig gerenderte Kopfzeile (Marke, Preise,
Kontakt, Über uns, Anmelden, Knopf „Kostenlos starten“), Hauptfläche in voller
Breite, dieselbe Fußzeile. Die vier Login-Karten (`index.html` = Anmeldung,
`register.html`, `forgot.html`, `reset.html`) erweitern dieses Gerüst und
behalten ihr zentriertes Karten-Layout (`.auth-container`); ihre H1 nennt das
Thema (Anmeldung, Registrierung …), die Marke steht in der Kopfzeile.
Stylesheet: `style.css` + `start.css`. Details: `docs/STARTSEITE.md`.

## Regeln für die H1

Die H1 nennt das **Thema der Seite** („Kontakt“, „Impressum“, „Preise“),
nie den Markennamen (WCAG 2.4.6 — wer per Überschriften-Navigation ankommt,
muss hören, wo er ist). Der Markenname steht in der Seitenleiste. Der
Staging-Hinweis „(Testumgebung)“ steht ausschließlich im Fenstertitel
(`base.html`), den Screenreader beim Laden ansagen — nicht in der H1.

Unter der H1 folgt direkt der Inhalt. Ein Zurück-Link gehört nur auf
Unterseiten, die eine Ebene tiefer liegen und keinen Seitenleisten-Eintrag
haben (z. B. „E-Mail & Passwort“ unter Einstellungen) — Muster siehe
`konto.html`. Öffentliche Seiten sind keine Unterseiten; die Seitenleiste
führt immer zur Startseite bzw. ins Dashboard.

## Die Fußzeile — zwei Listen, eine Regel

Die rechtlichen Links stehen auf **jeder** Seite in derselben Reihenfolge:

Impressum · Datenschutz · Nutzungsbedingungen · Widerrufsbelehrung ·
Vertrag kündigen · Vertrag widerrufen

**Kontakt** und **Über uns** sind seit 25.08.2026 (Michael) Einträge der
Seitenleiste — für Eingeloggte in `NAV_ITEMS`, für Besucher ohne Login in
`OEFFENTLICH_NAV` (beide in `dashboard.js`). Im Startgerüst stehen sie in der
Kopfzeile; die Fußzeile ist dort dieselbe wie auf den Gerüst-Seiten
(`rechtslinks(' · ')`). Der Makro-Parameter `mit_kontakt=True` wird seit
02.09.2026 von keiner Seite mehr benutzt.

Es gibt zwei Quellen, weil die Ziele sich unterscheiden:

- `backend/templates/_fusszeile.html` — Jinja-Makro `rechtslinks(trenner)`,
  **öffentliche Ziele** (`/impressum`, `/datenschutz`, …). Wird
  serverseitig gerendert, steht also auch ohne JavaScript im HTML
  (§ 5 DDG: „leicht erkennbar, unmittelbar erreichbar“). Nutzen
  `base_oeffentlich.html` (Trenner „·“) und die vier Login-Karten
  (Trenner „|“).
- `frontend/dashboard.js`, Konstante `LEGAL_LINKS` — dieselben
  Beschriftungen, aber **App-Ziele** (`/impressum-app`, `/datensicherheit`,
  `/nutzungsbedingungen-app`, `/widerruf-app`), damit Eingeloggte in der
  App bleiben. Jeder Eintrag trägt zusätzlich `oeffentlich:` mit dem freien
  Ziel; das nimmt die Hülle, wenn kein Login vorliegt (Gast-Review).
  `renderLegalLinks()` ist idempotent: Findet sie schon `.dash-legal-links`
  in der Fußzeile (serverseitig gerendert), lässt sie die Links in Ruhe und
  ergänzt nur den DSGVO-Hinweis.

**Wer einen Link ergänzt, umbenennt oder umsortiert, tut das an beiden
Stellen.** Der Klicktest `ui_geruest.py` vergleicht die Beschriftungen der
öffentlichen Fußzeile mit denen der App-Fußzeile und schlägt sonst fehl.

Pflichtlinks und ihre Grundlage: Impressum (§ 5 DDG), Datenschutz
(Art. 13 DSGVO), Nutzungsbedingungen (§ 312d BGB), Widerrufsbelehrung
(Art. 246a EGBGB), Vertrag kündigen (§ 312k BGB — Kündigungsknopf, ohne
Anmeldung erreichbar), Vertrag widerrufen (§ 356a BGB — Widerrufsfunktion,
ohne Anmeldung), Kontakt (§ 5 DDG, zweiter Kommunikationsweg).

## Die Seitenleiste im öffentlichen Modus

Ohne Login rendert `dashboard.js` (`OEFFENTLICH_NAV`) die Einträge Preise,
Kontakt, Über uns und unten — an der Stelle von „Abmelden“ — „Anmelden oder
registrieren“ (→ `/login`). Der Marken-Link zeigt auf `/` (Startseite). Mit Login erscheint
die normale `NAV_ITEMS`-Navigation (Startseite, Neues Projekt anlegen, Meine
Projekte, Meine Prompts, Einstellungen, Datensicherheit, Kontakt, Über uns,
für Admins Benutzerverwaltung, Abmelden), der Marken-Link zeigt auf
`/dashboard`.

## Rechtstexte: eine Quelle, drei Sichten

Impressum, Datenschutz, Nutzungsbedingungen und Widerrufsbelehrung sind
seit 25.08.2026 Templates (`backend/templates/impressum.html` usw.) auf
`base_oeffentlich.html`. Der Text steht in einem
`<div id="legalContent"><div class="legal-container">…</div></div>`.

Dieser `.legal-container` ist die **einzige Quelle**. Zwei weitere Sichten
holen ihn per `fetch` + `DOMParser` aus der öffentlichen Seite und betten
ihn in ihren Rahmen ein:

- die In-App-Sichten `frontend/impressum-app.html`, `datensicherheit.html`,
  `nutzungsbedingungen-app.html`, `widerruf-app.html` (Route
  `_serve_protected_page`, Login nötig),
- die Demo-Seiten `demo-impressum.html`, `demo-datenschutz.html`,
  `demo-nutzungsbedingungen.html` (`demo-shell.js`, `loadLegalInline`).

Deshalb: Klasse `.legal-container` und die Struktur darin nicht umbenennen.
Die Rechtstexte bleiben bewusst deutsch (juristischer Text; siehe
`backend/I18N.md`), nur das Gerüst drumherum ist sechssprachig.

Die Fassung der Widerrufsbelehrung, der Kunden beim Buchen zustimmen, steht
in `backend/main.py` (`WIDERRUFSBELEHRUNG_FASSUNG`) — bei jeder inhaltlichen
Änderung von `templates/widerruf.html` hochzählen.

## Neue öffentliche Seite anlegen

1. Template `backend/templates/<name>.html`:

   ```jinja
   {% extends "base_oeffentlich.html" %}
   {% block title %}InkluDocs - {{ _('Seitentitel') }}{% endblock %}
   {% block seitentitel %}{{ _('Seitentitel') }}{% endblock %}
   {% block main %}
   <section class="dash-card" aria-labelledby="abschnitt-h">
     <h2 id="abschnitt-h">{{ _('Abschnitt') }}</h2>
     …
   </section>
   {% endblock %}
   {% block page_script %}<script>(() => { … })();</script>{% endblock %}
   ```

2. Route in `backend/main.py`:

   ```python
   @app.get("/<name>", response_class=HTMLResponse)
   async def name_page(request: Request):
       lang = detect_language(request)
       return templates.TemplateResponse(
           "<name>.html",
           template_context(request, lang, is_staging="staging" in BASE_URL),
       )
   ```

3. Soll die Seite in der öffentlichen Seitenleiste erscheinen: Eintrag in
   `OEFFENTLICH_NAV` (`dashboard.js`). Soll sie in die Fußzeile: beide Listen
   (siehe oben).

4. Neue Texte in allen sechs Sprachkatalogen nachtragen
   (`backend/scripts/check_i18n.py` meldet fehlende).

5. `ui_geruest.py` um die Seite ergänzen (Liste `SEITEN`).

### Stolperstein: Seitenskripte

`dashboard.js` ist auf jeder Gerüst-Seite geladen und definiert die
globalen Helfer `byId()`, `t()`, `announce()`, `formatDate()`. Ein
Seitenskript darf auf oberster Ebene **keine** eigenen `const byId`, `const T`
o. ä. anlegen — der Browser bricht dann mit „Identifier has already been
declared“ ab und das ganze Skript läuft nicht. Seitenskripte deshalb immer
in eine IIFE `(() => { … })();` packen und die globalen Helfer nutzen.

## Prüfen

- `tests/e2e/ui_start.py` (Playwright + axe): Startseite und Login-Karten
  im Startgerüst — Kopfzeile, H1, Abschnitte, Fußzeile, Meta/JSON-LD,
  Weiterleitungen (`/app` → `/login`, eingeloggt `/` → `/dashboard`),
  robots.txt, sitemap.xml, schmale Bildschirme, axe 0.
- `ui_geruest.py` (Playwright + axe, ohne Setup): Skip-Link, genau eine H1
  mit dem Seitenthema, Seitenleiste anonym und eingeloggt, Fußzeile
  vollständig und identisch zur App-Fußzeile, kein Rest der alten
  Login-Karte, axe 0 Verstöße auf allen öffentlichen Seiten.
- `ui_widerrufen.py`, `ui_recht.py`: die Formularabläufe von Widerruf und
  Kündigung im neuen Gerüst.
- `verify_recht.py`: Pflichtlinks im rohen HTML (ohne JavaScript).
- `backend/scripts/check_i18n.py`: Vollständigkeit der Übersetzungen.
