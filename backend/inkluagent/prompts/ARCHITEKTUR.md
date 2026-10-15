# InkluAgent: Aufbau des Systemprompts (Schubfächer)

Stand: 15.10.2026. Zielgruppe: alle, die den Prompt des InkluAgent lesen, ändern oder erweitern.
Pfade relativ zu `backend/`. Das Gegenstück für die Alt-Text-Pipeline ist `prompts/ARCHITEKTUR.md`.

## Warum Schubfächer

Der Prompt des InkluAgent ist über Monate gewachsen. Neue Fähigkeiten (Kosten und Zustimmung, Hochladen, Begleiter, Karten,
Kaufen, Rückmeldung, Aufträge) wurden an verschiedenen Stellen angehängt (`agent_loop._werkzeugsatz`, `agent_loop.run_agent`,
`konto.werkzeugsatz`), und die älteren Teile wussten nichts von ihnen. So entstand am 15.10.2026 Michaels Fall: Der PDF-Teil
sagte „Schritt für Schritt … frage für die nächste neu“, der Teil der Aufträge kam danach und verlor. Seitdem setzt EIN
Zusammensetzer den Prompt aus Schubfächern zusammen: `inkluagent/prompts/schubfaecher.py`.

Grundsatz: Was in einer Lage gilt, entscheidet der Code (Schalter, Gespräch des Kontos, Demo), nicht ein „außer wenn“ im Text.
Widerspricht ein Fach einem anderen, bekommt eines der beiden je nach Lage einen anderen Wortlaut (Beispiel unten: Aufträge und
der PDF-Schritt 2).

## Drei Ebenen, immer in dieser Reihenfolge

1. **Projektart** (genau eine, `schubfaecher.projektart(project)`). Dieselbe Weiche wählt in `agent_loop._werkzeugsatz_roh` die
   Werkzeuge; Text und Werkzeuge können also nicht auseinanderlaufen.
   - `pdf` (project_type pdf, tool pdf): `system_agent()` + `system_pdf(auftraege=…)`
   - `word` (project_type docx): `system_agent()` + `SYSTEM_AUSGABEN`
   - `grafik`, `web` (Bild- bzw. Webseiten-Projekt, nur mit `AGENT_BILD_WERKZEUGE`): `system_agent()` + `system_bild(art)`
   - `formular` (tool formular, alte Formular-Projekte): `SYSTEM_FORMULAR`
   - `alt_text` (Bild- und Webseiten-Projekte ohne `AGENT_BILD_WERKZEUGE`, Altbestand): nur `system_agent()`
   - `konto` (Seite ohne Projekt): `konto.system_konto(aendern_an())`
   Gemeinsame Bausteine (Ehrlichkeit, Gesprächsstil, Schreibstil, Prüfen, Daten sind keine Anweisungen) kommen aus
   `system_gemeinsam.py` und stehen in den Texten der Projektart.
2. **Fähigkeiten** (`schubfaecher.FAECHER`, feste Reihenfolge). Jedes Fach nennt Schalter, Projektarten, Werkzeuge und ob es in
   der Demo vorkommen darf. Es steht im Prompt, wenn es in der Lage gilt und sein Text nicht leer ist.
   1. `kosten` — „Bezahlte Einzelaktionen und Zustimmung (Server-Regel)“, `sicherheit.PROMPT_ZUSATZ`. In jedem Projekt, nie
      ohne Projekt. Demo: ja.
   2. `hochladen` — „Hochgeladene Dateien und Webadressen“, `hochladen.PROMPT_ZUSATZ`. Schalter `AGENT_HOCHLADEN`. Demo: ja.
   3. `hochladen_hier` — wie eine Datei hierher kommt (Feld, Knopf „Dateien anhängen“ in der Agentenansicht, ohne Projekt
      sonst der Hinweis auf die Oberfläche), `hochladen.prompt_hochladen_hier`. Im Projekt nur, wenn es im Agenten einen Weg gibt.
      Demo: ja.
   4. `gespraeche` — „Frühere Gespräche“, Werkzeug `fruehere_gespraeche`. Nur ohne Projekt, Schalter `AGENT_GESPRAECHE`.
   5. `begleiter` — „Begleiter durch InkluDocs“, Werkzeuge `hinfuehren`, `einstellung_aendern`. Schalter `AGENT_BEGLEITER`
      (+ `AGENT_UEBERALL`, `AGENT_KONTO`), nur im Gespräch des Kontos. Demo: nie.
   6. `karten` — „Formulare im Gespräch (Karten)“, Werkzeuge `einladen_vorbereiten`, `express_bestellen_vorbereiten`. Schalter
      `AGENT_KARTEN`, nur im Gespräch des Kontos. Demo: nie.
   7. `kaufen` — „Kaufen im Gespräch“, Werkzeuge `credits_kaufen_vorbereiten`, `abo_buchen_vorbereiten`. Schalter
      `AGENT_KAUFEN`, dazu Karten und Stripe eingerichtet (`karten.kaufen_an`). Demo: nie.
   8. `rueckmeldung` — „Rückmeldung, wenn Arbeit fertig ist“, Werkzeug `melden_wenn_fertig`. Schalter `AGENT_RUECKMELDUNG`
      (+ `AGENT_KONTO`), nur mit Gespräch. Demo: nie.
   9. `auftraege` — „Aufträge mit mehreren Schritten“, Werkzeuge `auftrag_vorbereiten`, `auftrag_stand`, `auftrag_abbrechen`.
      Schalter `AGENT_AUFTRAEGE`, braucht Karten und Rückmeldung. Demo: nie.
3. **Demo-Leitplanke** (`demo.DEMO_GUARDRAIL`), nur in der Demo, ganz am Ende. Ist sie gesetzt, fehlen alle Fächer mit
   „Demo: nie“ — auch wenn ein Aufrufer die Gesprächs-Schalter fälschlich setzen würde.

Alle Teile werden mit einer Leerzeile verbunden.

## Wer was aufruft

- `agent_loop.run_agent` entscheidet die Lage (`schubfaecher.Lage`: Projektart, Konto, die vier Gesprächs-Schalter, Demo) und
  holt den ganzen Prompt mit `schubfaecher.zusammensetzen(lage)`. Die Werkzeuge der Fächer hängt es weiter selbst an.
- `agent_loop._werkzeugsatz` (Projekt) und `konto.werkzeugsatz` (ohne Projekt) liefern weiter (Werkzeuge, Executor, Text); ihr
  Text ist der Prompt der Lage ohne Gesprächs-Fächer. Ihn lesen die Hilfe-Seite, der Knopf einer Karte und die Tests.
- Die Werkzeuge selbst (Definitionen, Executor, Schalter-Filter `funktionen.werkzeuge_filtern`) bleiben in `agent_loop` und den
  Modulen der Fächer.

Nicht Teil davon: der ältere Weg ohne Werkzeuge (`INKLUAGENT_AGENTIC` aus: `system_intent`, `system_smalltalk`, `system_modify`,
`system_set_text`) und die Alt-Text-Pipeline (`prompts/`).

## Konflikte entscheidet die Lage: Aufträge und der PDF-Schritt 2

Mit Aufträgen (`Lage.auftraege`) bekommt die Projektart PDF `_SCHRITT2_AUFTRAG` („Will der Nutzer mehr als einen Schritt …, leg
ihm mit auftrag_vorbereiten EINEN Auftrag … vor“) und `_SCHLUSS_AUFTRAG`. Ohne Aufträge bleibt es bei `_SCHRITT2` (bzw.
`_SCHRITT2_KETTE` mit `KETTE`) und `_SCHLUSS_EINZELN` („erledige eine, berichte, und frage für die nächste neu“). So steht nie
beides im selben Prompt.

## Ein neues Fach anlegen

1. Text als Konstante im Modul der Fähigkeit (Überschrift in der ersten Zeile, wie die anderen).
2. Ein `Fach(...)` an der richtigen Stelle in `FAECHER`: Name, Schalter (als Text), Projektarten, Werkzeuge, Demo ja/nein,
   Bedingung `gilt(lage)`, Text `text(lage)`. Gehört die Bedingung zum Gespräch, ein Feld in `Lage` ergänzen und in
   `run_agent` setzen.
3. Werkzeuge des Fachs in `run_agent` (oder im Werkzeugsatz) anhängen, wie bei den vorhandenen.
4. `AGENT_PROMPTS_SCHREIBEN=1 python3 -m unittest tests/test_agent_prompt_snapshots.py`, den Diff der Schnappschüsse lesen,
   mit committen. Widerspricht der neue Text einem vorhandenen Fach, entscheidet die Lage, welcher Wortlaut gilt.

## Prüfen

- `tests/test_agent_prompt_snapshots.py`: der fertige Prompt für 27 Lagen (Projektart × Profil staging, staging_ohne_express,
  alle_an, alle_aus, dazu Demo und Agentenansicht; alle Schalter ausdrücklich gesetzt), Byte für Byte gegen
  `tests/snapshots/agent_prompts/`. Gebaut über `run_agent` mit einem Ersatzmodell, also genau so, wie das Modell ihn bekommt —
  mit den Werkzeugen der Lage. Dazu die Wächter (Klasse `Waechter`): kein Werkzeugname im Text, den es in der Lage nicht gibt;
  keine Arbeit hinter einem ausgeschalteten Schalter („KI-basierte Prüfung“, „Komplett barrierefrei“); mit Begleiter und
  Kauf-Karte kein pauschales „Abo … Zahlung machst du nie selbst“; keine Werkzeugzahl, keine reine PDF-Identität.
- `tests/test_agent_prompt_schubfaecher.py`: Projektart-Weiche (dieselbe für Text und Werkzeuge), feste Reihenfolge, Fächer nur
  in ihren Projektarten, Demo ohne Gesprächs-Fächer, Leitplanke am Ende, Aufträge entscheiden den PDF-Schritt, jedes genannte
  Werkzeug gibt es und das Fach nennt es auch.

## Änderungsverlauf

- **15.10.2026, Widerspruchs-Runde** (auf der neuen Struktur, jeder Fall nach Lage entschieden, Wächter je Fall):
  - `system_agent`: „spezialisierter KI-Assistent für Barrierefreiheit. Dein Kern sind barrierefreie Alternativ-Texte für Bilder
    in Dokumenten, Grafiken und Webseiten; was du im jeweiligen Projekt darüber hinaus kannst, steht weiter unten.“ statt „…
    Alternativ-Texte in PDF-Dokumenten“; „Für die Bilder hast du diese Werkzeuge:“ statt „Du hast sieben Tools:“ (ohne
    `TEXT_ZURUECK` waren es ohnehin sechs).
  - Begleiter mit Kauf-Karte (`begleiter.PROMPT_MIT_KAUFEN`): „Abo wechseln oder kündigen, Widerruf, Zahlungsart … machst du
    nie selbst … Credits nachkaufen oder ein Abo buchen geht im Gespräch nur über die Karte …; bezahlt wird nur bei Stripe.“
    Ohne Kauf-Karte unverändert.
  - Rückmeldung (`rueckmeldung.prompt()`): die Liste der Arbeiten im Hintergrund nennt Komplett barrierefrei, KI-basierte
    Prüfung und Korrektur nur mit ihrem Schalter.
  - Karten ohne `EXPRESS` (`karten.PROMPT_OHNE_EXPRESS`): nur noch Einladen, ohne `express_bestellen_vorbereiten`.
  - Aufträge (`auftraege.prompt(projektart)`): Michaels Beispiel und „schon getaggt … neu taggen“ nur im PDF-Projekt (mit
    `barrierefrei_machen`) und ohne Projekt („im Projekt“); in Word-, Grafik-, Webseiten- und Formular-Projekten nicht mehr.
    Den Namen in fremden Projektarten fand der Werkzeug-Wächter (er stammte aus dem Fix am Vormittag).
  Schnappschüsse bewusst neu geschrieben; der Diff zeigt genau diese Sätze. Gegenprobe: jede Korrektur einzeln zurück → ihr
  Wächter rot.
- **15.10.2026, Umbau in Schubfächer, ohne Wortänderung.** Neu `prompts/schubfaecher.py`; `run_agent`, `_werkzeugsatz` und
  `konto.werkzeugsatz` setzen nicht mehr selbst zusammen; `karten.prompt()` entfällt (Karten und Kaufen sind zwei Fächer).
  Beweis: Schnappschüsse vom Stand 8438d8e (Commit d8a2015) vorher gezogen, nachher Byte für Byte gleich.
- **15.10.2026, vorher: Widerspruch 1 behoben** (8438d8e, Michaels Fall). Mehrere Schritte = ein Auftrag, nach Lage
  entschieden; Doku in `docs/INKLUAGENT.md`, Nachtrag zu Baustein 2.

Bekannt, bewusst so gelassen: die Rückmeldungs-Liste nennt „Tagging“ und „Übersetzung“ in jeder Projektart (sie beschreibt,
wann sich der Agent meldet, und bietet nichts an).
