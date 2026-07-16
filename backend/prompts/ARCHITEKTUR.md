# InkluDocs Prompt-Architektur (v4)

Stand: 16.07.2026 (nach Paket 1 der Prompt-Generalinspektion).
Zielgruppe: kuenftige Leser und Bearbeiter des Prompt-Systems.
Alle Pfade relativ zur Backend-Wurzel.

## Die sechs Ebenen des Prompt-Geruests

Jeder Prompt der v4-Pipeline wird aus sechs Ebenen zusammengesetzt, von allgemein nach speziell:

### Ebene 1: Rollen

Datei: `prompts/components/roles.py`.

Vier Rollen, eine pro Pass: ROLE_KLASSIFIKATOR, ROLE_INVENTARISIERER, ROLE_BESCHREIBER, ROLE_VALIDATOR, dazu SYSTEM_BESCHREIBUNG als System-Prompt der Beschreibungs-Aufrufe. Die Rollen definieren ZUSTAENDIGKEIT, nicht Einmaligkeit: Kernverbote werden bewusst mehrfach verstaerkt (Rolle plus ANTI_HALLUZINATION plus Bildtyp-Sektion plus Final Check).

### Ebene 2: Geteilte Constraints

Verzeichnis: `prompts/components/constraints/`.

Seit Paket 1 nur noch sechs lebende Module:

- `halluzination.py` — ANTI_HALLUZINATION_REGELN, die einzige Schicht, die wirklich in ALLEN Buildern vorangestellt wird (Zwei-Wege-Logik: benennen oder neutral beschreiben, nie hedgen).
- `atmosphere_evidenz.py` — ATMOSPHAERE_REGEL (Wertungen nur mit sichtbarem Beleg im selben Satz).
- `eigennamen.py` — EIGENNAMEN_REGELN (im Bild lesbarer Eigenname schlaegt Kontext; aktiv im karte-Builder und Validator).
- `evidenz_stufen.py` — EVIDENZ_STUFEN_REGELN, seit Paket 1 eingegrenzt auf Marken-/Produkt-/Text-Identifikationen; aktiv nur im logo-Builder.
- `kontaktdaten.py` — KONTAKTDATEN_PFLICHT (lesbare Kontaktdaten wortgetreu; aktiv in diagramm-full, tabelle, screenshot).
- `lizenz_logos.py` — LIZENZ_LOGOS_REGELN (CC-Symbole, Zertifikate; aktiv in logo-Builder und Validator).

### Ebene 3: Die 17 Kategorie-Schubladen

Verzeichnis: `prompts/builders/`.

Jedes Bild wird vom Klassifikator in genau eine Schublade geroutet; jede Schublade hat ihren eigenen Beschreibungs-Builder:

- Foto-Familie (6, in `beschreibung_foto.py`): foto_personen, foto_event, foto_objekte, foto_essen, foto_landschaft, foto_architektur. Geteilte Helper-Bloecke (Personenregeln, Kontextregeln, Unterschriften, Atmosphaere, Zweck, Kompaktheit) sichern Konsistenz.
- Daten-Familie (7, in `beschreibung_daten.py`): illustration, diagramm, tabelle, karte, infografik, screenshot, strukturformel.
- Mini-Familie (3, in `beschreibung_mini.py`): logo, icon, funktional — kurze funktionale Alt-Texte ohne Langbeschreibung.
- Sonderfall dekorativ (`dekorativ.py`): kein Prompt, reine Code-Heuristik, Ergebnis ist der WCAG-konforme leere Alt-Text.

Dazu Quermodule: `classification.py` (Pass 1), `inventar.py` (Pass 2), `validierung.py` (Pass 4), `combo.py` (Lean-Mode-Wrapper), `helpers.py` (user_hint_block, load_examples, resolve_prompt_mode).

### Ebene 4: Schemas

Verzeichnis: `prompts/components/schemas/`, gerendert ueber `schema_helpers.py`.

Pydantic-Modelle erzwingen die Output-Struktur: BeschreibungOutput (alt_text 20 bis 400 Zeichen, langbeschreibung maximal 2000), IconBeschreibungOutput (3 bis 80 Zeichen), ClassificationOutput, InventarOutput. Die Schema-Grenzen sind seit Paket 1 die EINZIGEN harten Zeichen-Caps; alles darunter sind Richtwerte.

### Ebene 5: Few-Shots

Verzeichnis: `prompts/components/examples/<bildtyp>/` mit `good_*.json` und `bad_*.json`, geladen ueber `helpers.load_examples`.

Aktuell kuratiert: foto_architektur, foto_essen, foto_event, foto_landschaft, foto_objekte (12 Dateien). Fehlende Ordner erzeugen einen ehrlichen Platzhalter-Hinweis im Prompt.

### Ebene 6: Pruef-Ebene

- Verify-Pass (`pipelines/v4/orchestrator.py`): optionaler Gegencheck per ENV V4_VERIFY_MODE.
- Validator-Pass (`prompts/builders/validierung.py`): Pass 4 im Full-Modus, Verhalten per VALIDATOR_MODE (z.B. flag = needs_review setzen).

Die ENV-Schalter des Gesamtsystems:

- PIPELINE_VERSION — `v4` (Builder-Welt) oder wörtlich `v3_7` (Legacy `context_engine.py`; jeder andere Wert bricht den Container-Start ab).
- LLM_PROVIDER — bedrock (Claude Sonnet) oder mistral; steuert auch den Prompt-Modus-Default.
- V4_PASS_MODE — full (4 Paesse) oder lean (2 Aufrufe).
- V4_PROMPT_MODE — full (Mistral-Drill-Bloecke) oder lean (schlank fuer Sonnet); Default lean bei bedrock.
- V4_VERIFY_MODE — Verify-Pass an/aus.
- VALIDATOR_MODE — Verhalten des Validator-Passes.

## Aufruf-Fluss

### Lean (2 Aufrufe, empfohlen mit LLM_PROVIDER=bedrock)

1. Klassifikation (inklusive foto_subtyp).
2. Combo-Aufruf: Inventar wird intern "im Kopf" erstellt, Output ist direkt das BeschreibungOutput-Schema. Kein Validator-Pass.

### Full (4 Paesse, Default; richtige Wahl fuer Mistral)

1. Klassifikation.
2. Inventar (forensische Bestandsaufnahme).
3. Beschreibung (Kategorie-Builder).
4. Validator (Beleg-Pruefung gegen das Inventar).

## Temperaturen

Definiert in `main.py`: GENERATION_TEMPERATURE = 0.3 fuer den Normalbetrieb (Bulk-Verarbeitung), REGENERATE_TEMPERATURE = 0.5 fuer das Einzel-Neu-Generieren (bewusste Variation). Die Client-Defaults in `pipelines/v4/` stehen auf 0.0 und werden von main.py ueberstimmt.

## Prompts rendern (lesbar machen)

Das komplette Geruest laesst sich ohne Python-Kenntnisse lesbar machen:

    docker exec -w /app <container> python3 -m scripts.render_prompts

Ergebnis: `prompts/snapshots/*.md` — pro Builder und Modus eine Markdown-Datei. Bei Builder-Aenderungen ist der Snapshot-Diff direkt der Prompt-Diff.

## Entscheidungen Paket 1 (16.07.2026)

Aufraeum-Runde nach der Regel-Inventur vom 16.07.2026. Jede Aenderung mit Begruendung:

1. Tote Constraint-Module geloescht (personen_regeln.py, wcag.py, kontext_nutzung.py, verbotene_formulierungen.py): kein Builder band sie mehr ein, sie taeuschten eine Single Source of Truth vor und trugen teils Regeln, die den aktiven Prompts direkt widersprachen (Gesichtserkennungs-Bann vs. gewollte Promi-Benennung).
2. Tote Importe entfernt (PERSONEN_REGELN, EVIDENZ_STUFEN_REGELN, KONTAKTDATEN_PFLICHT in beschreibung_foto.py; detect_type_from_context in pdf_processor.py): ungenutzte Importe suggerierten Regelgeltung, die nicht bestand.
3. Anti-Redundanz und budni-Korrektur in den Zweck-Block aller sechs Foto-Builder portiert: die E4-Korrektur (keine erfundene Handlung "beim Einkaufen") lebte vorher NUR im toten Modul und erreichte den aktiven Code nie.
4. Illustration-Builder: vorgeschriebene Hedge-Formulierung "vermutlich Hypothese 1 oder 2" durch die Zwei-Wege-konforme Form ersetzt ("als Katze oder Fuchs deutbar"): der Builder widersprach seiner eigenen ANTI_HALLUZINATION-Schicht im selben Prompt.
5. Full-Personenzweig an die 16.06.-Lockerung des Lean-Zweigs angeglichen (grobe Alters-/Erscheinungskategorien erlaubt, nur praezise Alterszahlen verboten): dasselbe Bild bekam je nach V4_PROMPT_MODE gegensaetzliche Regeln; Ethnie/Religion/Gesundheit-Verbote blieben unangetastet.
6. EVIDENZ_STUFEN_REGELN auf Marken-/Produkt-/Text-Identifikationen eingegrenzt: die Stufe-3-Wand galt woertlich auch fuer Personen, Orte und Gebaeude und kollidierte mit der Wahrzeichen- und Promi-Erlaubnis der Foto-Builder; aktiv ist das Modul ohnehin nur im logo-Builder.
7. Laengenregime der Daten-Familie vereinheitlicht (tabelle/karte/screenshot/strukturformel/infografik): die alten Inline-MUSS-Caps (250/350/400) sind jetzt Richtwerte nach dem Muster des Foto-Kompaktheitsblocks; harte Obergrenze ist allein das Schema (400/2000). Langbeschreibungs-Caps (1500/1000/800) analog zu Richtwerten mit Schema-Obergrenze 2000 umformuliert.
8. InkluAgent-Laengen an die Pipeline gezogen: maximal 400 statt 500 Zeichen (hartes Schema-Limit), typische Laenge 80 bis 300 mit explizit genannter 400er-Obergrenze — der Chat versprach vorher Werte, die die Pipeline gar nicht speichern kann.
9. foto_event-Beispiel "rund zehn Personen" zu "zehn Personen" korrigiert: das Beispiel verstiess gegen die eigene Zaehl-Regel desselben Prompts (schaetzen nur bei Verdeckung).
10. Screenshot-Praefix explizit erwuenscht gemacht: der v4-Builder hatte das v3.7-Praefix-Verbot beim Portieren verloren, sein Positiv-Beispiel begann aber schon mit "Screenshot der …" — jetzt ist dokumentiert, dass das eine bewusste Entscheidung ist (konsistent mit den Praefix-Pflichten der uebrigen Daten-Familie).
11. Nutzer-Hinweis-Prioritaet praezisiert (helpers.py): explizite Ordnung — Hinweis gewinnt bei Fokus, Zweck und Wortwahl; sichtbare Fakten kommen immer aus dem Bild; Seitenkontext dient nur der Gewichtung. Loest den unaufgeloesten Konflikt mit "BILD GEWINNT GEGEN KONTEXT" auf.
12. Validator-Duplikat entfernt: der Absatz "Plus halluzinations_warnung-Respekt" stand wortgleich zweimal im foto_objekte-Spezialblock (Copy-Paste-Fehler).
13. Dekorativ-Schwelle im Klassifikations-Prompt von 80x80 auf 100x100 angeglichen: die Code-Heuristik (_DEKORATIV_MAX_SIDE in dekorativ.py) arbeitet mit 100 px; zwei Zahlen fuer denselben Zweck waren Drift. Sync-Pflicht ist jetzt im Code kommentiert.
14. roles.py-Docstring ehrlich gemacht: die Behauptung, Verbote wuerden nicht in den Prompts wiederholt, stimmte nicht mehr — Kernverbote werden bewusst mehrfach verstaerkt; die Rollen definieren Zustaendigkeit, nicht Einmaligkeit.
15. context_engine.py deutlich als LEGACY v3.7 gekennzeichnet und das budni-Beispiel dort korrigiert ("Kundin im Drogeriemarkt budni" ohne erfundene Handlung): solange der v3.7-Pfad produktiv erreichbar ist, soll niemand dort versehentlich neue Regeln pflegen — und die E4-Korrektur erreicht endlich auch den lebenden Legacy-Prompt.
16. Diese Datei (ARCHITEKTUR.md) neu angelegt: das Geruest war bisher nur ueber Code-Kommentare und die Regel-Inventur rekonstruierbar.

Bewusste NICHT-Aenderungen in Paket 1:

- Die harte Behaelter-Wortliste im Validator (foto_objekte-Spezialblock) bleibt unveraendert, bis der A/B-Test klaert, ob Sonnet sie noch braucht.
- Die Legacy-Engine (context_engine.py) bleibt funktional bestehen, bis der Rueckbau des v3.7-Pfads separat entschieden ist — sie ist jetzt nur klar gekennzeichnet.
- Zwei Legacy-Divergenzen bleiben bewusst bestehen, bis der v3.7-Rueckbau entschieden ist: das Farben-Verbot der Legacy-Foto-Prompts (v4 nennt Farben, wo sie tragen) und die Legacy-Erlaubnis von Atmosphaere bei Screenshots (v4 verbietet sie dort). Beide gelten nur im v3.7-Pfad.

### Review-Feinschliff (16.07.2026, nach Zwei-Rollen-Review)

Nach der adversarischen Review (Code-Korrektheit + Prompt-Konsistenz) ergaenzt: ANTI_HALLUZINATION Regel 2 erlaubt jetzt ausdruecklich die gleichwertige Nennung zweier naheliegender Deutungen („als Katze oder Fuchs deutbar") als praezise Mehrdeutigkeits-Beschreibung — damit ist die Illustration-Alternativen-Form systemweit gedeckt. Die Nutzer-Hinweis-Vorrang-Ordnung stellt klar, dass Hinweise Wissen liefern duerfen, das dem Bild nicht anzusehen ist (Bildtyp, Identitaet), und dass der Seitenkontext belegte Namen und Funktionen liefert. Meta-Kommentare und Code-Verweise wurden aus den Modell-Texten in Python-Kommentare verschoben (Full-Personen-Block, Evidenz-Stufen, Screenshot-Praefix). Neuer Snapshot „01_classification.lean.mit-nutzerhinweis" macht die Vorrang-Ordnung sichtbar.
- Das Benennungs-SOLL (einheitliche Charta fuer Personen/Orte/Marken ueber alle Generationen) und die neue Fotomontage-/Collage-Regel kommen als Paket 2 — das sind inhaltliche Regelaenderungen, kein Aufraeumen.
