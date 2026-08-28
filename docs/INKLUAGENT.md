# InkluAgent: ein Assistent, mehrere Werkzeuge

Stand 28.08.2026. Der InkluAgent ist der Chatbot in jedem InkluDocs-Projekt
(Kasten „Chatbot“ unter der Bild- bzw. Feldliste). Er läuft als Tool-Use-Loop
auf Sonnet über Bedrock (`backend/inkluagent/agent_loop.py`) und bekommt je
Werkzeug des Projekts einen eigenen Fachteil, aber denselben Charakter.

## Gerüst

```
backend/inkluagent/
├── chat_engine.py            Einstieg process_message(); agentic Pfad -> agent_loop.run_agent
├── agent_loop.py             Tool-Use-Loop; WEICHE nach project.tool (_werkzeugsatz)
├── prompts/
│   ├── system_gemeinsam.py   Ehrlichkeit, Gesprächsstil, Schreibstil — EINE Quelle für alle Werkzeuge
│   ├── system_agent.py       Fachteil Alt-Texte (Bild-Projekte: PDF, Word, Web, Grafik)
│   └── system_formular.py    Fachteil Quickinfos (Formular-Projekte)
├── tools/
│   ├── definitions.py        Werkzeugsatz Bilder (list_project_images, view_image, generate/update/revert_alt_text, tavily_search)
│   ├── definitions_formular.py  Werkzeugsatz Formulare (list_form_fields, get_field_details, view_field, generate/update/revert_quickinfo, search/save master data, tavily_search)
│   ├── project.py, altext.py Bild-Werkzeuge
│   ├── formular.py           Formular-Werkzeuge
│   └── search.py             Tavily (gemeinsam)
├── adapters/inkludocs.py     Projekt-Kontext (liefert project.tool)
└── storage.py                Chat-Verlauf je Projekt (chat_messages)
```

Die Weiche liegt an EINER Stelle: `agent_loop._werkzeugsatz(project, …)` gibt
`(tool_definitions, executor, system_prompt)` zurück. `project.tool == "formular"`
→ Formular-Satz + `SYSTEM_FORMULAR`; sonst Bild-Satz + `SYSTEM_AGENT`. Der
Bild-Agent kennt keine Feld-Werkzeuge, der Formular-Agent keine Bild-Werkzeuge —
sie können sich nicht vermischen. Der Chat-Verlauf ist ohnehin je Projekt getrennt.

## Was für alle Werkzeuge gilt (einmal ändern, überall wirksam)

`prompts/system_gemeinsam.py` hält die drei Abschnitte, die den Charakter des
Agenten ausmachen: Ehrlichkeit gegenüber der eigenen History, Gesprächsstil,
Schreibstil (keine Tabellen, keine Markdown-Optik, Vorschläge in Anführungszeichen).
Beide Fach-Prompts setzen sie ein; beim Umbau war der Alt-Text-Prompt
byte-gleich zur Fassung vor dem 28.08. (belegt), seither kommt in beiden der
Block „Prüfen heißt aufrufen“ (`PRUEFEN`) dazu. Wer den Ton des Agenten
ändern will, ändert ihn dort.

Grenze: Der Werkzeug-Modus braucht `INKLUAGENT_PROVIDER=bedrock` und
`INKLUAGENT_AGENTIC=true`. Der klassische Vier-Pfad-Dispatcher in
`chat_engine.py` (Mistral-Zeit) kennt nur Bilder; Formular-Projekte bekommen
ohne Werkzeug-Modus oder bei einem Absturz des Loops eine klare Fehlermeldung
statt des Bild-Dispatchers.

Sicherheit gegen Prompt-Injection: Fremdtexte (Seitentext, Umfeld, Anmerkung
des Gastes) kommen als `…_daten`-Felder mit Kennzeichnung ins Tool-Result,
und der Formular-Prompt erklärt, dass Werkzeug-Inhalte Daten und nie
Anweisungen sind. Im Chat abgenommene Texte tragen `quelle = chat` und
werden von „Alle neu generieren“ nicht angefasst.

Gemeinsam sind außerdem: der Loop selbst (höchstens 6 Werkzeug-Runden je Turn,
Werkzeug-Ergebnisse auf 40.000 Zeichen gekappt mit Hinweis an das Modell,
Bilder als image-Block im nächsten Turn, `refresh_*`-Aktionen fürs Frontend),
die Websuche (`tavily_search`), die Kontingent-Wache (`billing.pruefe_kontingent`),
die Abrechnungsregel „Reden ist frei, Erzeugen oder Ändern kostet 1 Credit“ und
die Sicherheitsregel, dass `project_id`/`user_id` nie aus den Modell-Argumenten
kommen, sondern aus der Sitzung (ToolExecutor).

## Fachteil Formulare (Quickinfos)

Prompt `system_formular.py`: Rolle (Quickinfo = zugänglicher Name, Nutzer sieht
die Beschriftung nicht), UI-Nummern „Feld n“ ↔ echte `feld_id`, Multi-Dokument,
Werkzeuge, Feld-Pass erklärt, Speichern nur nach Bestätigung, Beleg-Pflicht,
Stammdaten zuerst, Gast-Prüfung lesen, Proaktivität, Standards (bedienbar ≠
PDF/UA-konform). Die Stilregeln kommen WÖRTLICH aus dem Feld-Pass
(`prompts/builders/quickinfo.STILBLOCK`) — Chatbot und Pipeline schreiben nach
denselben Regeln.

Werkzeuge (`tools/formular.py`):

- `list_form_fields` — Übersicht mit `ui_label`, Status, Quelle, Sicherheit, Prüfstatus.
- `get_field_details` — Beschriftung mit Lage, Abschnitt, Umfeld, Optionen,
  Original, Beleg/Hinweise, Anmerkung des Gastes, Seitentext (Kontext).
- `view_field` — Ausschnitt oder ganze Seite (widgetfreie Kopie, nie Feldwerte).
- `generate_quickinfo` — Feld-Pass für ein Feld (`formular_ki.generiere_seite`,
  Variation), speichert sofort, quelle `ki`, 1 Credit (`quickinfo_generierung`).
- `update_quickinfo` — speichert nach Zustimmung; vorher DIESELBE Nachprüfung wie
  der Feld-Pass (`formular_ki.nachpruefung`: Beleg im Seitentext, Lage in
  Feldnähe, Regeln). Ergebnis „niedrig“ wird nicht gespeichert (`force=true` nur
  nach ausdrücklichem Beharren). quelle `ki` mit Sicherheit — das Badge in der
  Oberfläche zeigt „KI-Vorschlag, sicher/mittel“. 1 Credit (`quickinfo_aenderung_chatbot`).
- `revert_quickinfo` — Original aus der PDF, kostenlos.
- `search_master_data` / `save_to_master_data` — Stammdaten des Kontos.
- `tavily_search` — wie bei den Bildern.

Frontend: `app.html` `inkluagentSectionHtml(projectId, 'formular')` liefert
denselben Kasten (Knopf „Chatbot“, Verlauf, Eingabefeld, Enter sendet) mit
Formular-Einleitung; `formular.js` hängt ihn unter die Feldliste (nur Besitzer,
Gäste bekommen keinen Chatbot) und setzt `refresh_feld`-Aktionen live um
(Textfeld, Badge, Beleg — ohne Neu-Rendern, `Formular.chatAktionen`).

## Werkzeug-Transparenz (28.08.2026)

Der Chat-Endpunkt `POST /api/projects/{id}/chat` streamt mit `Accept:
application/x-ndjson` je Werkzeugaufruf eine Zeile `{"type":"tool","name":…}`
(Callback `on_tool` im Agent-Loop) und am Ende `{"type":"reply", …,
"werkzeuge":[…]}`; ohne den Header bleibt die JSON-Antwort. Die Oberfläche
zeigt während des Laufs „Ruft gerade auf: Feld-Details“ (Status-Zeile,
aria-live) und unter jeder Antwort „Geprüft mit: Feldliste, Feld-Details“
bzw. „Ohne Werkzeug (aus dem Gesprächsverlauf)“; die Liste wird je Antwort in
`chat_messages.werkzeuge` gespeichert und im Verlauf wieder angezeigt.
Passend dazu die gemeinsame Prompt-Regel „Prüfen heißt aufrufen“
(`system_gemeinsam.PRUEFEN`): Prüf-/Bewertungsfragen lösen im selben Turn
einen Werkzeugaufruf aus; ohne Aufruf sagt der Agent, dass er aus dem
Verlauf antwortet.

## Chat-Bremse (28.08.2026)

Reden mit dem Agenten kostet keine Credits, aber Bedrock-Token. Deshalb gilt
je Konto eine Tagesgrenze von `DAILY_CHAT_LIMIT` Nutzer-Nachrichten (Standard
100, Umgebung; 0 sperrt den Chat), gezählt über alle Projekte des Kontos in
`chat_messages` (`database.get_daily_chat_count`, nur `role = user`, UTC-Tag).
Admins sind ausgenommen. Die Prüfung läuft vor dem Speichern der Nachricht;
darüber antwortet der Endpunkt mit 429 und „Du hast die 100 Chat-Nachrichten
für heute genutzt. Morgen geht es weiter.“ (6 Sprachen), die Oberfläche
zeigt den Text in der Statuszeile. Die Demo hat ihre eigene Grenze je
Besucher (`DEMO_DAILY_CHAT_LIMIT`, 12).

## Ein neues Werkzeug anschließen (Kochrezept)

1. `tools/<werkzeug>.py`: Funktionen `(…, project_id, user_id) -> {"ok", "result"|"error"}`,
   Zugriff immer über `projects.user_id`, falsche ids mit Liste der echten ids beantworten.
2. `tools/definitions_<werkzeug>.py`: Anthropic-Schemas + Executor (Argumente
   nur fachlich; `tavily_search` aus `definitions.py` übernehmen).
3. `prompts/system_<werkzeug>.py`: Fachteil + `system_gemeinsam`-Blöcke.
4. `agent_loop._werkzeugsatz`: Zweig für `project.tool`; passende
   `refresh_*`-Aktion im Loop; Projekt-Zusammenfassung.
5. Frontend: `inkluagentSectionHtml(projectId, '<variante>')` einbinden, Aktionen umsetzen.
6. Tests: E2E-Chat-Turn in `tests/e2e/verify_<werkzeug>.py`, Klicktest Kasten vorhanden/abwesend beim Gast.
