# InkluDocs: Gerüst der Fähigkeiten (Landkarte)

Stand 22.09.2026. Zweck (Steve): Jede Fähigkeit bleibt einzeln sichtbar und anbaubar, nichts vermischt sich.
Jede Fähigkeit hat **sechs Bausteine**. Wer eine neue Fähigkeit baut (z. B. PowerPoint), füllt dieselben
sechs Felder und trägt sie hier ein. Pfade relativ zur Repo-Wurzel; `backend/` ist die Backend-Wurzel.

Die sechs Bausteine:

1. **Kern** — das Backend-Modul mit der Logik (ohne HTTP).
2. **Endpunkte** — ein Router oder ein klar abgegrenzter Abschnitt; Knopf und Chatbot rufen dieselben Kernfunktionen.
3. **Oberfläche** — eine Datei je Ansicht (Station), Texte in den sechs Katalogen (`backend/locales`).
4. **Prompt und Schema** — je Modellaufruf ein Baustein in `backend/prompts/builders/` und ein Schema in `backend/prompts/components/schemas/`.
5. **Chatbot** — ein Werkzeugmodul in `backend/inkluagent/tools/`, eine Definitionsdatei, ein Prompt-Zusatz in `backend/inkluagent/prompts/`; die Weiche nach Dateiart steht in `agent_loop._werkzeugsatz`.
6. **Tests und Doku** — Unit-Test, End-to-End-Test, Klicktest, Abschnitt in `docs/`.

## Alt-Texte (Bilder in PDF, Word, Web, Grafik)

- Kern: `backend/pipelines/v4/` (Orchestrator, `llm_client.py`, Anbieter-Clients), `backend/pdf_processor.py`
- Endpunkte: `backend/main.py` (Upload, Extraktion, Sammellauf, Einzelbild, Export) — der Sammelplatz, siehe „Bekannte Schulden“
- Oberfläche: `backend/templates/app.html` (Ansicht „Alt-Texte“)
- Prompt und Schema: `backend/prompts/` (Rollen, Regeln, 17 Schubladen, Beispiele; `ARCHITEKTUR.md`)
- Chatbot: `tools/project.py`, `tools/altext.py`, `tools/search.py`, `tools/definitions.py` (TOOL_DEFINITIONS), `prompts/system_agent.py`
- Tests und Doku: `tests/` (Pipeline), `tests/e2e/ui_smoke.py`, `docs/`

## Quickinfos (Formularfelder)

- Kern: `backend/formular_processor.py`, `backend/formular_ki.py`, `backend/formular_export.py`
- Endpunkte: `backend/formular_api.py`
- Oberfläche: `frontend/formular.js` (Ansicht „Quickinfos“)
- Prompt und Schema: `prompts/builders/quickinfo.py`, `schemas/quickinfo.py`
- Chatbot: `tools/formular.py`, `tools/definitions_formular.py`, `prompts/system_formular.py` (Formular-Projekte); im PDF-Projekt Teil des PDF-Satzes
- Tests und Doku: `tests/e2e/ui_formular.py`, `verify_formular*.py`, `docs/QUICKINFOS.md`

## Word: Hörprobe, Prüfbericht, Umwandlung in PDF/UA, Word-Export

- Kern: `backend/docx_hoerprobe.py`, `backend/docx_export.py`, `backend/pdfua_export.py`, `backend/docx_struktur.py`
- Endpunkte: `backend/main.py` Abschnitt Word-Export / PDF/UA (`_pdfua_umwandeln_sync`, `_word_export_ausgabe_sync`)
- Oberfläche: `backend/templates/app.html` (Word-Ansicht „Alt-Texte“ mit Export-Bereich), `backend/templates/ablage.html`
- Prompt und Schema: keine Modellaufrufe (Regeln und Konverter)
- Chatbot: `tools/ausgaben.py`, `tools/definitions_word.py`, `prompts/system_ausgaben.py`
- Tests und Doku: `tests/e2e/verify_pdfua.py`, `verify_ablage.py`, `ui_ausgaben.py`, `verify_chat_ausgaben.py`, `docs/ABLAGE.md`, `docs/WORD.md`

## Übersetzen (Word)

- Kern: `backend/uebersetzung.py`
- Endpunkte: `backend/uebersetzung_api.py`
- Oberfläche: `frontend/uebersetzen.js` (Ansicht „Übersetzung“)
- Prompt und Schema: `prompts/builders/uebersetzung.py`, `schemas/uebersetzung.py`
- Chatbot: in `tools/ausgaben.py` (uebersetze_dokument, uebersetzung_stand, exportiere_uebersetzung), Definitionen in `tools/definitions_word.py`
- Tests und Doku: `tests/e2e/verify_uebersetzung.py`, `ui_uebersetzen.py`, `docs/UEBERSETZEN.md`

## PDF-Tagging (PDFix) und Kette

- Kern: `backend/pdf_tagging.py`, `backend/pdfix_scripts/` (Heines Originale unter `original_heine/`, Betriebsfassung, eigene Skripte)
- Endpunkte: `backend/tagging_api.py` (Tagging, Ansicht Dokument, Vorschau), `backend/kette_api.py` (Komplett barrierefrei machen)
- Oberfläche: `frontend/dokument.js` (Ansicht „Dokument“)
- Prompt und Schema: keine (Regeln von PDFix)
- Chatbot: `tools/pdf.py` (barrierefrei_machen, komplett_barrierefrei_machen, dokument_stand, exportiere_fertige_pdf …), `tools/definitions_pdf.py`, `prompts/system_pdf.py`
- Tests und Doku: `tests/test_pdf_tagging.py`, `test_pdfix_skript_drift.py`, `tests/e2e/verify_tagging.py`, `verify_kette.py`, `verify_export_komplett.py`, `ui_dokument.py`, `verify_chat_pdf.py`, `docs/TAGGING.md`

## Strukturlesung und Hörprobe (PDF)

- Kern: `backend/pdf_struktur.py`, `backend/pdfix_scripts/Struktur_Export.py`
- Endpunkte: in `tagging_api.py` (`…/struktur`), Seite `/struktur/{projekt}/{dokument}` in `main.py`
- Oberfläche: `frontend/dokument.js` (Klappe „Hörprobe lesen“, Link „Strukturansicht öffnen“), `backend/templates/struktur.html`
- Prompt und Schema: keine
- Chatbot: `tools/pdf.py` (hoerprobe_lesen)
- Tests und Doku: `tests/test_pdf_struktur.py`, `tests/e2e/verify_struktur.py`, `docs/TAGGING.md`

## Automatische Prüfung (PDF, Schritt 5)

- Kern: `backend/pdf_pruefung.py` (Modell an EINER Stelle: `MODELL`)
- Endpunkte: in `tagging_api.py` (`…/pruefung`) — bekommt einen eigenen Router, sobald die Korrektur dazukommt
- Oberfläche: `frontend/dokument.js` (Klappe „Automatische Prüfung“)
- Prompt und Schema: `prompts/builders/pdf_pruefung.py`, `schemas/pdf_pruefung.py`
- Chatbot: `tools/pdf.py` (pruefung_starten, pruefbericht_lesen)
- Tests und Doku: `tests/test_pdf_pruefung.py`, `tests/e2e/verify_pruefung.py`, `docs/TAGGING.md`

## Querschnitt

- Abrechnung: `backend/billing.py` (AKTIONS_PREISE: eine Zeile je kostenpflichtiger Aktion, GUELTIGE_QUELLEN)
- Datenbank und Migrationen: `backend/database.py`
- Ablage („Meine Ablage“): `main.py` (`_ausgabe_anlegen`, `_pdf_in_ablage`), `templates/ablage.html`
- Chatbot-Kern: `backend/inkluagent/agent_loop.py` (Weiche `_werkzeugsatz`), `chat_engine.py`, `providers/`
- Übersetzungen der Oberfläche: `backend/locales/<lang>/LC_MESSAGES/messages.po`, Prüfung `backend/scripts/check_i18n.py`

## Regeln für neue Fähigkeiten

1. Erst der Kern ohne HTTP, dann ein eigener Router (nicht in `main.py`), dann die Oberfläche.
2. Ein Modellaufruf = ein Prompt-Baustein + ein Schema; nichts aus anderen Bausteinen einbinden, was dort nicht hingehört.
3. Der Chatbot bekommt ein eigenes Werkzeugmodul und eine Definitionsdatei; kostenpflichtige und unumkehrbare Werkzeuge nutzen die Zwei-Schritt-Freigabe (`tools/ausgaben._angebot_merken/_angebot_einloesen`).
4. Knopf und Chatbot rufen dieselbe Kernfunktion.
5. Preis = eine Zeile in `billing.AKTIONS_PREISE`.
6. Tests: Unit für den Kern, End-to-End gegen Staging, Klicktest mit axe; Doku-Abschnitt.

## Bekannte Schulden (Stand 22.09.2026)

- `backend/main.py` (> 10.000 Zeilen) trägt die Alt-Text-Pipeline, PDF-Export, Word-Umwandlung, Ablage und viele Endpunkte. Neue Fähigkeiten liegen seit Wochen in eigenen Routern; das Alte ist noch drin. Aufräumen = eigener Tag.
- `tagging_api.py` trägt Tagging, Strukturlesung und Prüfung; die Prüfung bekommt mit der Korrektur einen eigenen Router.
