# Export-Abnahme

Stand: 15.09.2026. Modul `backend/export_abnahme.py`, Tests `tests/test_export_abnahme.py`,
Einbau in `backend/main.py` (`_build_pdf_for_document`, `export_pdf`).

## Zweck

Jede exportierte PDF wird **nach** dem Schreiben mit einem zweiten, unabhängigen Werkzeug
(pikepdf) nachgemessen. Der Schreibweg (PDFix-Roundtrip oder Ersatzweg mit PyMuPDF) prüft
sich damit nicht selbst. Maßstab: **Was der Kunde in InkluDocs sieht, steht in der Datei.**

Anlass: Am 14.09.2026 hätte der Ersatzweg ein Kundendokument mit 234 Alt-Texten ausgeliefert,
die im Strukturbaum unerreichbar waren und vom Abschluss-Schritt entfernt wurden. Die
Oberfläche hätte „234 getaggt“ gemeldet.

## Regeln (jede Verletzung = Befund, Abnahme nicht bestanden)

1. Die Datei lässt sich öffnen und hat so viele Seiten wie das Original.
2. Wurden Alt-Texte geschrieben, gibt es einen Strukturbaum (`/StructTreeRoot`).
3. Kein `Figure`-Element mit `/Alt` liegt außerhalb des Strukturbaums (keine Waisen).
4. Auf keiner Seite sind `BDC`/`BMC` und `EMC` unbalanciert (Inhaltsstrom-Parser, kein Regex).
5. Mindestens so viele der geschriebenen Texte stehen erreichbar in der Datei, wie der Export
   als „getaggt“ meldet. Vergleich über die ersten 60 Zeichen, Whitespace normalisiert.
   `""` (Alt entfernen) und `dekorativ` zählen nicht als Text.
6. Lesereihenfolge: Die Seiten der Bild-Elemente steigen in Baumreihenfolge an. Rücksprünge bis
   `LESEREIHENFOLGE_TOLERANZ_SEITEN` (8) sind erlaubt (Doppelseiten, InDesign-Reihenfolge),
   größere sind ein Befund.

Die Abnahme ändert die Datei nie und wirft keine Ausnahme nach außen; der Aufrufer fängt alles.

## Folgen eines Befunds

- Warnung im Export-Dialog (Kopfzeile `X-Export-Warnings`, Anzeige in `app.html` und
  `formular.js`).
- **Keine Credits**: `X-Export-Credits: 0`, `billing.verbuche` wird übersprungen. Beim ZIP
  reicht eine Datei mit Befund, dann ist der ganze Export kostenlos.
- Logzeile für Nexus:
  `EXPORT-ABNAHME ok|FEHLGESCHLAGEN projekt= dokument= verfahren= seiten=a/b figures_alt= waisen= unbalanciert= ruecksprung= texte=gefunden/erwartet [befunde=...] datei=`
  Läuft die Abnahme selbst nicht: `EXPORT-ABNAHME NICHT MOEGLICH projekt= dokument=: <Fehler>`.
  Der Nexus-Stundencheck greift `EXPORT-ABNAHME FEHLGESCHLAGEN` und `EXPORT-ABNAHME NICHT MOEGLICH`.

## Prüfen

    docker exec inkludocs-staging mkdir -p /app/tests
    docker cp tests/test_export_abnahme.py inkludocs-staging:/app/tests/
    docker exec -w /app inkludocs-staging python3 -m unittest tests.test_export_abnahme -v
    python3 /home/claude/export_probe.py 69 114 123     # echte API-Exporte, Kopfzeilen
    docker logs --since 10m inkludocs-staging 2>&1 | grep EXPORT-ABNAHME

Die Testdateien liegen nicht im Image (`/app/tests` fehlt), daher vorher hineinkopieren.

## Nicht abgedeckt (offen)

- veraPDF-Vergleich Original gegen Export (fand am 14.09. „tagged content inside Artifact“);
  braucht Java + veraPDF im Image.
- Word-Weg nach PDF/UA (`export_pdfua`) hat noch keine Abnahme.
- Inhaltlicher Abgleich je Bild (richtiger Text am richtigen Bild) — heute nur Seite und Anzahl.
