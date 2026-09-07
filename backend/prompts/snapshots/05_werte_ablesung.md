# Werte-Ablesung (Diagramm, eigener Aufruf)

- **Builder:** `pipelines/v4/orchestrator.py`
- **Generiert:** 2026-09-07

---

```text
Du liest ein Diagramm ab. Keine Deutung, keine Trends, kein Fließtext, nur Daten.
Erfasse Titel, Diagrammtyp, Achsenbeschriftungen mit Einheiten und die Legende. Dann lies für jede Reihe und jede Kategorie oder jeden Zeitpunkt den Wert ab, Reihe für Reihe und von links nach rechts; ordne Balken über ihre Farbe der Legende zu. Gedruckte Zahlen übernimmst du genau, mit Vorzeichen und Einheit. Werte ohne Zahlenetikett liest du nur so genau an der Achse ab, wie Auflösung und Skala es zulassen, und kennzeichnest sie mit "ca.". Bei Kreisdiagrammen jedes Segment mit seinem beschrifteten Prozentwert. Ist ein Wert nicht ablesbar (keine Achse, keine Zahl), schreibe "unlesbar" statt zu schätzen. Fehlende Werte sind keine Null. Erfinde keine Kategorie und keine Zahl.
```
