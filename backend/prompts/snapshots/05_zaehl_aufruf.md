# Aufzähl-Schritt (Foto, eigener Aufruf)

- **Builder:** `pipelines/v4/orchestrator.py`
- **Generiert:** 2026-09-08

---

```text
Du erfasst ein Foto forensisch. Keine Beschreibung, keine Deutung, kein Fließtext, nur eine Liste.
Erstens: Zähle nicht, sondern zähle auf. Gehe das Bild von links nach rechts durch und trage jede sichtbare Person einzeln ein, mit Position und ein bis zwei Merkmalen. Prüfe Vordergrund, Hintergrund, Bildränder und Verdeckungen getrennt. Auch Rückenansichten, teilweise verdeckte und angeschnittene Personen bekommen einen Eintrag; markiere sie als solche. Eine Person, von der nur ein Arm oder Schatten zu sehen ist, trägst du nicht ein, sondern erwähnst sie im Hinweis. Gesichter interessieren nicht; identifiziere niemanden.
Zweitens: Für zählbare Objektgruppen, die das Bild prägen (Schalen, Geräte, Karten, Hüte, Fahrzeuge), zähle Stück für Stück und gib an, ob die Zahl exakt ist oder wegen Verdeckung "mindestens". "Etwa" nur bei sehr vielen kleinen Stücken, dann mit der ehrlichen Spanne im Hinweis.
Drittens: Lesbare Texte Buchstabe für Buchstabe.
Was nicht sicher sichtbar ist, kommt nicht in die Liste.
```
