# Prüfkorpus und Konformitätstest beider Wege

Stand: 15.09.2026. Ordner `tests/korpus/`: `erzeuge_korpus.py` (Dokumente bauen), `dokumente/` (eigene
Testdokumente, keine Kundendateien), `konformitaet.py` (Lauf über beide Wege), `messlatte.json` (Sollwerte).

## Zweck

Jedes Dokument läuft einmal über PDFix und einmal über den Ersatzweg (PyMuPDF): Extraktion, feste
Testtexte ohne KI, Export, Export-Abnahme (`docs/EXPORT-ABNAHME.md`). Anspruch (Steve, 15.09.2026):
**Der Kunde bekommt auf beiden Wegen dieselbe Qualität.** Der Ersatzweg muss so viele Texte in die
Datei bringen wie PDFix und darf nicht mehr „Bilder“ anbieten (und berechnen) als PDFix Figures findet.

Anlass: Der Export-Fehler vom 14.09. lag seit Juni im Code, weil der Ersatzweg nie auf einem
Satzprogramm-Dokument gelaufen war.

## Dokumente

- `01_lo_testdokument.pdf`, `02_lo_diagramm.pdf`: Word-Vorlagen aus `tests/fixtures` über den Konverter
  (LibreOffice, getaggt).
- `03_synth_satz.pdf`: Satzprogramm-Stand-in (bis Michaels InDesign-Datei da ist): Figures mit MCIDs,
  `/K` als Referenz, RoleMap `PlacedGraphic`, Collage, ungetaggtes Bild, Doppelseiten-Reihenfolge,
  wiederholtes Logo (gleiches XObject auf zwei Seiten).
- `04_untagged_raster.pdf`, `05_scan.pdf`, `06_vektor_layout.pdf`: ohne Strukturbaum (nur Ersatzweg).
- `07_formular.pdf`: Formular ohne Bilder.

## Laufen lassen (Staging-Container, braucht Konverter + PDFix)

    docker exec inkludocs-staging mkdir -p /app/tests/korpus /app/tests/fixtures
    docker cp tests/fixtures/. inkludocs-staging:/app/tests/fixtures/
    docker cp tests/korpus/. inkludocs-staging:/app/tests/korpus/
    docker exec -w /app inkludocs-staging python3 tests/korpus/erzeuge_korpus.py tests/korpus/dokumente
    docker exec -w /app inkludocs-staging python3 tests/korpus/konformitaet.py           # Exit 1 bei Abweichung
    docker exec -w /app inkludocs-staging python3 tests/korpus/konformitaet.py --messlatte   # Sollwerte neu setzen

Die Messlatte wird nur bewusst neu gesetzt, wenn eine Änderung die Werte verbessert. Ein rot markierter
Wert in der Messlatte (`abnahme_ok: false`) ist ein bekannter, noch offener Mangel.

## Befunde des ersten Laufs (15.09.2026)

- Ersatzweg auf LibreOffice-PDFs: 19 „Bilder“ statt 6, weil jede Seite alle Bild-Ressourcen des
  Dokuments listet, auch nicht gezeichnete. Der Kunde würde 19 statt 6 Bilder berechnet bekommen.
  Dazu veraPDF 7.1-3 (Inhalt weder Artefakt noch getaggt) 5× neu.
- Ersatzweg: Alt-Texte sind nach XObject-Nummer (`xref`) verschlüsselt. Dasselbe Bild auf mehreren
  Seiten (Logo) bekommt überall den zuletzt geschriebenen Text.
- Ersatzweg auf Word-Diagramm: Vektorgrafik gefunden, aber nicht getaggt (0 von 1), PDFix 1 von 1.
- Abnahme: `PlacedGraphic` über RoleMap zählte nicht als Figure (behoben, Regel 7 ohne Metadaten-Regeln).
