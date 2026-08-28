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
Beide Fach-Prompts setzen sie ein; der Alt-Text-Prompt ist dadurch byte-gleich
zur Fassung vor dem 28.08. (belegt beim Umbau). Wer den Ton des Agenten ändern
will, ändert ihn dort.

Gemeinsam sind außerdem: der Loop selbst (höchstens 6 Werkzeug-Runden je Turn,
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

## Ein neues Werkzeug anschließen (Kochrezept)

1. `tools/<werkzeug>.py`: Funktionen `(…, project_id, user_id) -> {"ok", "result"|"error"}`,
   Zugriff immer über `projects.user_id`, falsche ids mit Liste der echten ids beantworten.
2. `tools/definitions_<werkzeug>.py`: Anthropic-Schemas + Executor (Argumente
   nur fachlich; `tavily_search` aus `definitions.py` übernehmen).
3. `prompts/system_<werkzeug>.py`: Fachteil + `system_gemeinsam`-Blöcke.
4. `agent_loop._werkzeugsatz`: Zweig für `project.tool`; passende
   `refresh_*`-Aktion im Loop; Projekt-Zusammenfassung.
5. Frontend: `inkluagentSectionHtml(projectId, '<variante>')` einbinden, Aktionen umsetzen.
6. Tests: E2E-Chat-Turn in `verify_<werkzeug>.py`, Klicktest Kasten vorhanden/abwesend beim Gast.
