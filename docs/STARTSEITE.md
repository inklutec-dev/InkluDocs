# Startseite und Anmeldeseite

Stand 02.09.2026 (Steve Weidel + Michael Karbe, gebaut mit Claude Code).
Diese Datei beschreibt die öffentliche Startseite `/`, die Anmeldeseite
`/login`, das gemeinsame Gerüst `base_start.html`, die Suchmaschinen-
Angaben und die Prüfungen. Übersicht aller Gerüste: `docs/SEITENGERUEST.md`.

## Warum

Bis zum 02.09.2026 war die Wurzel `inkludocs.de` die Anmeldemaske. Wer
InkluDocs nicht kannte, erfuhr dort nicht, was das Werkzeug tut, für wen
es ist und was es kostet; Suchmaschinen sahen nur den Titel „Anmeldung“.
Michael hatte auf Steves eigene Website (inklutec.de) als Vorbild verwiesen:
Versprechen, Leistungen, Produkt, Preis, Kontakt auf einer Seite.

## Was wo liegt

| Pfad | Template | Gerüst | Route (`backend/main.py`) |
|---|---|---|---|
| `/` | `start.html` | `base_start.html` | `start_page` — Eingeloggte → `/dashboard`; Demo-Instanz → `demo.html` |
| `/login` | `index.html` | `base_start.html` | `login_page` — `?geloescht=1` zeigt die Bestätigung nach Konto-Löschung |
| `/register`, `/forgot`, `/reset` | gleichnamig | `base_start.html` | unverändert, nur `noindex` und `nav_aktiv` |
| `/robots.txt` | — | — | `robots_txt` |
| `/sitemap.xml` | — | — | `sitemap_xml` |

Stylesheet: `frontend/start.css` (zusätzlich zu `style.css`; im Cache-Busting
`_ASSET_FILES` in `backend/i18n.py` eingetragen).

## Das Gerüst `base_start.html`

Bewusst **ohne** die App-Seitenleiste. Stattdessen:

- Skip-Link `.dash-skip` auf `#main` als erstes Element im Body.
- `header.start-header` (Navy): Marken-Link auf `/`, `nav` mit
  `aria-label="Hauptnavigation"` und den Einträgen Preise, Kontakt, Über uns,
  Anmelden, Knopf „Kostenlos starten“ (→ `/register`, nur wenn die
  Registrierung offen ist). Der aktive Eintrag trägt `aria-current="page"`
  (Template-Variable `nav_aktiv`).
- `main#main.start-main` mit `tabindex="-1"` (Ziel des Skip-Links). Die
  Login-Karten geben über den Block `main_class` die Klasse
  `start-main-auth` mit (zentrierte Karte).
- `footer.start-footer` mit der gemeinsamen Fußzeile `_fusszeile.html`
  (`rechtslinks(' · ')`, sieben Links wie auf allen öffentlichen Seiten).
- Kopf- und Fußzeile sind **serverseitig** gerendert; `dashboard.js` wird
  nicht geladen. Die Seite funktioniert ohne JavaScript vollständig.
- Block `seo` für Meta-/Link-Elemente im `<head>`; auf Staging oder bei
  `noindex=True` setzt das Gerüst `<meta name="robots" content="noindex, nofollow">`.

Die Login-Karten (`index.html`, `register.html`, `forgot.html`, `reset.html`)
behalten ihr Karten-Layout (`.auth-container`, `.auth-links`). Ihre H1 nennt
das Thema (Anmeldung, Registrierung, Passwort zurücksetzen, Neues Passwort
setzen), die Marke steht in der Kopfzeile. Ihre frühere, von Hand getippte
`.legal-footer` ist weg; die Fußzeile kommt vom Gerüst.

## Aufbau der Startseite (`start.html`)

Jeder Abschnitt ist eine `<section aria-labelledby="…">` mit eigener H2;
Karten sind `<li>` in einer `<ul>` mit H3 (kein `<article>` — VoiceOver führt
article als Orientierungspunkt, Steve 03.09.2026). Die Überschriften-Navigation
per Screenreader ergibt damit
das Inhaltsverzeichnis der Seite.

1. **Kopfbereich** (Navy): H1 aus zwei Zeilen — Markensatz „Ein Inhalt.
   Viele Menschen. Gleiche Chancen.“ (Michael, 03.09.2026, als `<span
   class="start-claim">` in der H1, damit es EINE Überschrift bleibt) und
   darunter das Versprechen mit Suchbegriffen „Alt-Texte per KI für
   barrierefreie PDF, Word und Formulare“. Der Absatz beginnt mit Michaels
   Satz „Wir machen Informationen barrierefrei nutzbar, unabhängig von
   Fähigkeiten, Geräten und Situationen.“ Drei Knöpfe: „Ohne Anmeldung
   selbst erleben“ (→ demo.inkludocs.de), „Kostenlos starten“
   (→ `/register`) und „Anmelden“ (→ `/login`, Michael 03.09.: Kunden
   sollen nicht in die Kopfzeile suchen müssen). Rechts daneben Michaels
   KI-Bild (03.09.2026, `frontend/startseite-hero-2026-09.webp` + `.jpg`
   840 px, `<picture>`, width/height gegen Layoutsprünge; unter 900 px
   einspaltig unter den Knöpfen). Alt-Text von InkluDocs selbst erzeugt
   (Staging-Projekt 660), Bildnachweis im Impressum („teilweise
   KI-generiert mit ChatGPT/OpenAI“, Personen nicht real). Jeder Knopf hat seine
   eigene Unterzeile direkt darunter („Die Demo läuft ganz ohne Konto.“,
   „{n} Credits im Monat, ohne Zahlungsdaten.“, „Du hast schon ein Konto.“)
   — Steve 03.09.: bewusst OHNE aria-describedby, der Satz steht im HTML
   hinter seinem Link und wird als nächste Zeile gelesen; ARIA würde ihn
   beim Zeilenlesen doppelt ansagen.
2. **Drei Werkzeuge**: Alt-Texte für PDF und Bilder, barrierefreie
   Word-Dokumente, Quickinfos für Formulare.
3. **So funktioniert es**: Hochladen, die KI beschreibt, prüfen und
   exportieren (`<ol>`, Nummern per CSS-Counter).
4. **Vorher / nachher**: Dateiname gegen einen Alt-Text, ausdrücklich als
   vereinfachtes Beispiel mit erfundenen Zahlen gekennzeichnet.
5. **Für wen**: Behörden und Kommunen, Verlage und Redaktionen, Agenturen
   und Dienstleister, Hochschulen und Bildung als H3 je Gruppe (bis
   03.09. fetter Text — Codys Befund: ohne Trennzeichen las der
   Screenreader Titel und Erklärung in einem Rutsch) — und der Satz für alle
   anderen (Selbstständige, kleine Unternehmen, Vereine, Privatpersonen).
   Steve 02.09.2026: offiziell für Behörden, nutzen darf es jeder; die
   Nutzungsbedingungen sehen Verbraucher ausdrücklich vor (Widerruf,
   Kündigungsknopf).
6. **Preise**: Sätze mit Zahlen aus `billing.py` (Free-Kontingent,
   Alt-Text-/Quickinfo-Preis, günstigstes Paket, Single-Preis), Link auf
   `/preise`, § 19 UStG.
7. **Wer dahintersteht**: InkluTec (Steve, blind, Barrierefreiheit + KI),
   Actino Software (PDF-Fachwissen), „Deine Dateien bleiben deine Dateien“
   (EU, kein Training), Links Über uns und Datenschutzerklärung.
8. **Häufige Fragen**: sieben native `<details>/<summary>`, kein JavaScript.
9. **Schluss**: Demo-Knopf und „Kostenlos starten“ plus Kontakt (ohne
   „Anmelden“ — der steht nur im Kopfbereich und in der Kopfzeile).

Texte sind auf Suchbegriffe geschrieben (Alt-Texte KI, barrierefreie PDF,
PDF/UA, barrierefreies Word, Quickinfos Formular, BFSG, BITV 2.0, WCAG),
in Du-Form wie das Werkzeug; Französisch siezt. Alle Strings laufen über
`_()` und liegen in den sechs Katalogen (`backend/locales/*/messages.po`,
Block „Startseite + Startgerüst“ am Ende).

## Suchmaschinen

- `<title>` und Meta-Beschreibung sechssprachig, `canonical` auf die
  Hauptdomain ohne www (`_oeffentliche_basis()`), Open Graph (Titel,
  Beschreibung, URL, `og:locale` je Sprache), `twitter:card`.
- JSON-LD `SoftwareApplication` (Anbieter InkluTec, Angebot ab 0 Euro,
  Link auf `/preise`).
- `/robots.txt`: auf **Staging** `Disallow: /` (die Testumgebung darf nie in
  den Index); sonst App, API, Login-Karten gesperrt, Verweis auf die Sitemap.
- `/sitemap.xml`: die öffentlichen Seiten (`_SITEMAP_SEITEN`), Startseite mit
  Priorität 1.0.
- Login-Karten tragen `noindex`.
- **Nach dem Prod-Rollout (Steves Handgriff):** Google Search Console für
  inkludocs.de anmelden und `https://inkludocs.de/sitemap.xml` einreichen.
  Bing Webmaster Tools analog.

## Verhalten rund um die Anmeldung

- Alle Weiterleitungen für Nicht-Eingeloggte zeigen auf `/login`
  (`_serve_protected_page`, `_render_protected_template`, Team-Einladung,
  `dashboard.js` bei 401, Seitenleiste „Anmelden oder registrieren“, Links
  in Registrierung/Passwort-Seiten und den Inline-Bestätigungsseiten).
- Abmelden führt auf `/` (die Startseite).
- Eingeloggte, die `/` aufrufen, landen im Dashboard.
- Demo-Instanz (`DEMO_MODE=on`): `/` bleibt die Demo-Seite, `/login`
  leitet auf `/`.

## Barrierefreiheit (BITV 2.0 / WCAG 2.2 AA)

Skip-Link, Landmarken header/nav/main/footer (nur diese vier — Abschnitte tragen
id, aber KEIN aria-labelledby; einziges ARIA ist aria-current="page", Steve
03.09.2026: semantisches HTML, ARIA nur wo nötig), genau eine H1 je Seite,
H2/H3-Hierarchie, `aria-current`, sichtbarer Fokus (3 px Orange auf hellem
Grund, 3 px Weiß auf Navy), Zielgrößen ≥ 44 px, Kontraste (Weiß auf Navy
14,2:1, Weiß auf Orange 4,6:1, Fließtext 15,4:1, gedämpft 8,9:1), keine
Bewegung, keine Bilder, Lesereihenfolge = DOM-Reihenfolge, native
`details/summary` statt eigener Akkordeons. `prefers-reduced-motion` wird
respektiert.

## Prüfen

- `tests/e2e/ui_start.py` (Playwright + axe): siehe Docstring — Startseite,
  Login-Karten, Weiterleitungen, robots/sitemap, schmaler Bildschirm.
  Aufruf auf dem Server:
  `/home/claude/.venv-pw/bin/python tests/e2e/ui_start.py http://localhost:8002`
- `ui_geruest.py` (außerhalb des Repos, `/home/claude/`): öffentliche
  Gerüst-Seiten; seit 02.09.2026 erwartet es `/login` als Ziel von „Anmelden
  oder registrieren“ und prüft die Login-Karten unter `/login`.
- `backend/scripts/check_i18n.py` läuft im Docker-Build (fehlende msgids
  brechen den Build).

## Sicherungen und Rückweg

Angefasste Dateien liegen als `*.bak-pre-startseite-20260902` daneben
(`main.py`, `index.html`, `register.html`, `forgot.html`, `reset.html`,
`dashboard.js`, `i18n.py`, `SEITENGERUEST.md`, die sechs `.po`). Neue Dateien:
`base_start.html`, `start.html`, `start.css`, `tests/e2e/ui_start.py`,
diese Doku. Rückweg = Sicherungen zurückkopieren, neue Dateien entfernen,
Staging neu bauen.
