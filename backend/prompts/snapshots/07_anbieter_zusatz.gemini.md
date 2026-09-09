# Anbieter-Zusatz gemini (wird an den Beschreibungs-Prompt gehängt)

- **Builder:** `pipelines/v4/anbieter_profil.py`
- **Generiert:** 2026-09-09
- **Demo-Werte:**
  - Temperatur: (Aufrufer)
  - Bild zuerst: False
  - Bildauflösung: (Vorgabe)

---

```text
BESONDERE SORGFALT

- Farben nennst du so, wie sie im Bild stehen: goldfarben statt gelb, türkis
  statt blau, dunkelblau statt blau-grau. Bei Datengrafiken nur dort, wo eine
  Farbe etwas erklärt.
- Material, Untergrund und Gewässerart nennst du nur, wenn das Bild sie zeigt:
  kein "hölzern" für einen grauen Kasten, kein "asphaltiert" für einen
  Schotterweg, kein "Seeufer", wenn es auch ein Fluss sein kann.
- Eine Deutung bleibt Beschreibung: rötliches Licht ist rötliches Licht, keine
  Dämmerung; ein Raum mit Flipcharts ist ein Raum mit Flipcharts, kein Seminar,
  solange der Kontext es nicht sagt.
- Orte, Bauwerke, Fachbegriffe (korinthische Säule, Balkendiagramm), Künstler
  und Jahr stehen im Alt-Text, nicht nur in der Langbeschreibung.
- Sind Objekte am Bildrand angeschnitten oder teils verdeckt, ist die Zahl
  keine exakte Zahl: "mindestens 23 Schalen, einige am Rand angeschnitten"
  (Belegregel 7).
```
