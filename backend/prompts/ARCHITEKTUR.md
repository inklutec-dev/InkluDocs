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
- `kontaktdaten.py` — KONTAKTDATEN_PFLICHT (lesbare Kontaktdaten wortgetreu; aktiv in tabelle und screenshot — diagramm traegt seit Paket 4 eine eigene LESBARE-TEXTE-Sektion, infografik eine eigene Kontaktdaten-/URL-Passage).
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
- V4_VERIFY_MODE — Verify-Pass an/aus (off/kritisch/alle).
- V4_VERIFY_KORREKTUR — off (Default) = eine mitgelieferte Redakteurs-Korrektur wird ignoriert (Flag-Verhalten wie bisher); on = Korrektur wird uebernommen (siehe Paket 3 unten). Nur im Lean-Pfad wirksam.
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

## Paket 2+3 (16.07.2026 abends)

Paket 2 — Qualitaets-Charta (inhaltliche Regelaenderungen):

1. Wahrzeichen-Benennung auf SOLL-Niveau vereinheitlicht: foto_landschaft hebt eindeutig erkennbare ikonische Motive (Eiffelturm, Brandenburger Tor, Golden Gate Bridge, Koelner Dom) vom frueheren "duerfen" auf das SOLL-Niveau von foto_architektur — vage Umschreibung trotz eindeutiger Erkennbarkeit ist ein Qualitaetsfehler, bei echter Unsicherheit weiter neutral beschreiben, nie raten.
2. SYSTEM_BESCHREIBUNG (roles.py) erweitert den Personen-Benennungs-Auftrag um zweifelsfrei erkennbare Wahrzeichen und beruehmte Bauwerke — gleiche Logik: eindeutig erkennbar heisst benennen ist Teil des Auftrags.
3. Neue systemweite Regel 5 "FOTOMONTAGEN UND COLLAGEN" in ANTI_HALLUZINATION_REGELN (halluzination.py): erkennbar nicht zusammenpassende Bildelemente werden ausdruecklich als Fotomontage/Collage benannt und getrennt beschrieben; da die Schicht in allen Buildern vorangestellt ist, gilt die Regel ueberall automatisch. Der Inventar-Pass bekommt zusaetzlich einen Montage-Check-Pruefauftrag (Indikatoren als halluzinations_warnung-Eintrag, kein Schema-Feld geaendert).
4. Icon-Formel praezisiert (Schwingshandl R3): Funktion zuerst, optional kurze Formbeschreibung in runden Klammern ("Suche (Lupe)", "Menue oeffnen (drei Striche)"); Farben- und Praefix-Verbote unveraendert.
5. Eineindeutigkeits-Prueffrage im Validator (Schwingshandl R6): Koennte ein Grafiker aus Kontext plus Alt-Text ein Bild erstellen, das dieselbe Funktion erfuellt — fehlende zentrale Funktions-Information wird als fehlend markiert.
6. Zaehl-Disziplin als geteilter Baustein (_render_zaehl_block) in allen 6 Foto-Buildern: bis etwa 15 exakt zaehlen, circa/rund/etwa NUR bei echter Verdeckung sichtbarer Teile und dann mit Grund; die fruehere Inline-Version in foto_event (PERSONENZAHL) ist durch den Baustein ersetzt.

Paket 3 — Redakteurs-Pass (pipelines/v4/orchestrator.py):

7. VerifyOutput um zwei optionale Felder erweitert: korrigierter_alt_text (20-400 Zeichen) und korrektur_begruendung (kurz); tolerant normalisiert, damit der Verify nie an Formalien scheitert.
8. Der Verify-Prompt arbeitet jetzt als Redakteur statt als reiner Widerleger (Vorbild: InkluAgent-Modify-Muster): binaerer Punkt-fuer-Punkt-Abgleich jeder Aussage ("weitgehend korrekt" verboten), exaktes Nachzaehlen von Personen/Objekten (Hedge-Woerter nur mit sichtbarem Verdeckungs-, Anschnitt- oder Unschaerfe-Grund — gleiche Ausnahmen wie der Zaehl-Baustein), Vollstaendigkeits-Check (fehlende zentrale Elemente wie lesbarer Text, Wahrzeichen, praegende Objekte) und Montage-Check; bei Beanstandung liefert er eine korrigierte Fassung. Die Zielsprache wird dem Redakteur explizit benannt (gleiche Quelle wie _language_suffix), das Laengen-Regime ist an das Original gekoppelt (einfache Motive unter 150, komplexe bis etwa 250, harte Obergrenze 400). Achtung: der Prompt selbst ist damit strenger als der alte Refuter — bei V4_VERIFY_MODE=kritisch/alle kann sich die needs_review-Rate auch mit KORREKTUR=off aendern (gleicher Mechanismus, strengerer Pruefer).
9. Neue ENV V4_VERIFY_KORREKTUR: off (Default) = gleicher Mechanismus wie bisher — needs_review wird nur bei alt_text_belegt=false gesetzt, ein mitgelieferter korrigierter Text wird ignoriert (liefert der Redakteur eine Korrektur bei alt_text_belegt=true, z.B. wegen Unvollstaendigkeit, passiert bei off nichts; sie steht nur im validation_result-JSON); on = der korrigierte Alt-Text wird uebernommen (auch bei alt_text_belegt=true), needs_review gesetzt, Original und Begruendung stehen im Log und die Anwendung im pipeline_steps-Audit-Trail. Die Langbeschreibung bleibt in beiden Faellen unangetastet; Verify laeuft weiterhin nur nach V4_VERIFY_MODE (off/kritisch/alle) — und wie bisher ausschliesslich im Lean-Pfad: die Full-Pipeline (V4_PASS_MODE=full) hat keinen Verify/Redakteur, dort prueft der Validator-Pass.

Nachbesserung (16.07.2026, nach Steve-Review): Die Wahrzeichen-Regel ist von einer Beispiel-Liste auf eine Faehigkeits-Regel umgestellt (SYSTEM_BESCHREIBUNG, foto_landschaft, foto_architektur) — benannt wird jedes Motiv, das ein durchschnittlicher Sehender auf einen Blick erkennt, die verbliebenen 2-3 Beispiele sind ausdruecklich als nicht abschliessend markiert und ein Schwellen-Gegenbeispiel ergaenzt, weil Modelle sonst an den genannten Beispielen ankern und NUR diese benennen. Ausserdem ist nach dem Workshop-Foto-Befund (8 statt 7 Personen) eine Gruppen-Zaehlregel ergaenzt: Zaehl-Baustein und Redakteurs-Prompt verlangen das Gesamtbild ("acht Personen in einer Reihe, dahinter weitere Personen"), leicht versetzte oder angeschnittene Personen werden nicht weggezaehlt; die drei Grundsaetze (Zaehlen, Montage, Wahrzeichen) stehen jetzt auch im InkluAgent-System-Prompt.

Kosten: der Verify-Pass kostet wie bisher rund einen zusaetzlichen Sonnet-Aufruf pro geprueftem Bild (jetzt mit etwas mehr Output-Tokens fuer die Korrektur); V4_VERIFY_KORREKTUR=on erzeugt KEINEN weiteren Aufruf. Das Default-Verhalten der Pipeline bleibt bis zur bewussten Aktivierung beider ENV-Schalter unveraendert.

## Paket 4 (16./17.07.2026): Daten-Familie auf Premium

Alle 7 Builder in `prompts/builders/beschreibung_daten.py` auf das Premium-Muster von beschreibung_foto.py (foto_objekte als Vorlage) gehoben: ROLE_BESCHREIBER + ANTI_HALLUZINATION_REGELN voran, BILDTYP/BILDGROESSE, ZIEL-Absatz, geteilte Bloecke (Inventar, Kontext mit Bild-gewinnt-Regel, Zweck, Kompaktheit — jetzt als Modul-Helper statt "Helper-Kandidat"-Pseudokommentare im Modell-Text), kompakte kategorie-spezifische Regeln in ruhiger Formulierung, Few-Shot-Sektion via load_examples (Ordner fehlen noch, Platzhalter-Fallback greift), AUSGABE-SCHEMA inline statt schema_doc-Anhaengsel, FINAL CHECK als nummerierte Pruefliste. Die Paket-1-Richtwerte (250/350 Alt-Text, 1500/1000/800 Langbeschreibung) sind unveraendert; harte Obergrenze bleibt allein das Schema (400/2000). Pro Kategorie:

1. diagramm — Zwitter aufgeloest: EIN Builder fuer beide Modi auf Basis des Premium-Lean-Zweigs vom 15.05.; der Mistral-Full-Zweig (14.04.) ist ersetzt, seine Pflichten (Kontaktdaten/lesbare Texte, Kernaussage mit konkreten Werten, Vollstaendigkeit, Konsistenz alt_text/langbeschreibung, keine Markdown-Tabellen) leben in Sektionen und Final Check weiter; erhalten: 6 Diagramm-Subtypen + Trend-Vokabular; entdrillt: INSIGHT-FIRST-PFLICHT zu "Kernaussage zuerst", ROLE_BESCHREIBER ergaenzt (fehlte im Lean-Zweig).
2. illustration — erhalten: Spezialwarnung fuer stilisierte Darstellungen, Spezies-/Charakter-Regel mit der Alternativen-Form ("als Katze oder Fuchs deutbar"), Interaktions-Regel mit Hund/Mikroskop-Beispiel, Nebenelemente-Vollstaendigkeit; entdrillt: VERBOTEN/MUSS-Duktus zu FALSCH/RICHTIG-Beispielen, neu ZIEL/Zweck/Kompaktheit/Langbeschreibungs-Struktur/Final Check.
3. tabelle — erhalten: Bilanz-Regel (Zwischensummen vs. Bilanzsumme samt FALSCH/RICHTIG-Beispiel), die 5 Lese-Schritte der Spalten-Zuordnung, Einheiten-Treue, OCR als Primaerquelle, KONTAKTDATEN_PFLICHT, 4x4-Regel; entdrillt: die (KRITISCH)-Etiketten und der MUSS-Drill weichen der ruhigen Formulierung, Bilanz und Spaltenzuordnung sind jetzt auch Final-Check-Punkte.
4. karte — erhalten: "Karte —"-Eroeffnung mit Gebiet/Thema/raeumlicher Kernaussage, Ortsnamen wortgetreu in Originalsprache (Bordeaux/Istanbul/Koeln), EIGENNAMEN_REGELN (TURKU) jetzt in der Ortsnamen-Sektion statt pauschal voran, Legenden-/Himmelsrichtungs-/Hintergrund-Ortsnamen-/Symbolik-Regeln, keine erfundenen Orte/Routen; entdrillt: MUSS-Listen zu Beschreibungs-Reihenfolge plus Final Check.
5. infografik — erhalten: "Infografik —"-Eroeffnung mit Kernaussage und Datenpunkten, Stationen-Logik mit Beziehungen, Zahlen-Vollstaendigkeit, Layout-Beschreibungs-Verbot mit inhaltlichen Gegenbeispielen, OCR-Pflichtquelle, Kontaktdaten/URL-Passage; ATMOSPHAERE_REGEL ist mit ihrer Kampagnen-Einschraenkung sauber in die ATMOSPHAERE-Sektion integriert statt pauschal vorangestellt (die alte Einschraenkung stand als Pseudo-Kommentar im Modell-Text).
6. screenshot — erhalten: Praefix-Regel "Screenshot der/des …" (Paket-1-Entscheidung), Anwendungs-Identifikation nur mit Beleg (jetzt explizit als Zwei-Wege-Logik formuliert), funktionale UI-Hierarchie, wortgetreue UI-Texte, KONTAKTDATEN_PFLICHT, Dark-/Light-Mode-Hinweis, keine emotionalen Wertungen; entdrillt: MUSS-Duktus, Atmosphaere-Verbot als eigene Sektion statt Pseudo-Kommentar.
7. strukturformel — erhalten: Praefix Strukturformel/Reaktionsgleichung mit beiden RICHTIG-Beispielen, fachliche Beschreibungs-Listen fuer Strukturen und Reaktionen, Screenreader-Notation (CH3 statt CH-tiefgestellt-3, Ladungen explizit, Pfeile als "reagiert zu"), Chemie-Genauigkeit (keine erfundenen Atome, Stoffnamen nur aus Kontext/Beschriftung); entdrillt: "ANTI-HALLUZINATION CHEMIE" heisst jetzt "CHEMISCHE GENAUIGKEIT" (die Schicht-Ueberschrift kommt nur noch einmal pro Prompt vor).

Querschnitt: ANTI_HALLUZINATION_REGELN stehen jetzt auch im Combo-Modus genau EINMAL pro gerendertem Prompt — im Lean-Pass-Modus (V4_PASS_MODE=lean) traegt der Inventar-Teil des Combo-Prompts die Schicht, die Daten-Builder lassen sie dann weg (Helper `_basis_schichten`; combo.py, Dispatcher und alle Signaturen unveraendert). Im Full-Pass-Modus (Default, eigenstaendiger Pass 3) tragen die Builder die Schicht selbst. Die Foto-Familie dupliziert im Combo-Modus weiterhin (bekannter Restpunkt). Snapshots via scripts/render_prompts unveraendert nutzbar.


### Nachtrag Paket-4-Review (17.07.2026)

Der geteilte Inventar-Block der Daten-Familie uebernimmt BEWUSST die gelockerte Foto-Premium-Formulierung (sichtbare Bildinformationen duerfen ergaenzt werden, duerfen dem Inventar aber nicht widersprechen) statt des alten AUSSCHLIESSLICH-Wortlauts. Diagramm und Illustration nutzen seit Paket 4 NEU das 150/250-Richtwert-Regime der Foto-Familie. Die V4_PASS_MODE-Normalisierung ist jetzt an allen drei Lesestellen identisch (strip und lower — vorher konnte ein Leerzeichen im ENV-Wert die Anti-Halluzinations-Schicht im Standalone-Pass still entfernen). render_prompts pinnt V4_PASS_MODE=full fuer alle Standalone-Snapshots. Die Diagramm-Sektion LESBARE TEXTE nennt die Trennzeichen-Treue jetzt ausdruecklich.
