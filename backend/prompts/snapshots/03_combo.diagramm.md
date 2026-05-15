# Combo (Lean-Mode Pass 2+3) — Bildtyp: diagramm

- **Builder:** `prompts/builders/combo.py:30`
- **Generiert:** 2026-05-15
- **ENV / Modus:**
  - `V4_PASS_MODE` = `lean`
  - `V4_PROMPT_MODE` = `lean`
- **Demo-Werte:**
  - width × height: 1280 × 720
  - bildtyp_top: diagramm
  - bildtyp_effective: diagramm

---

```text
DU ERLEDIGST ZWEI AUFGABEN IN EINEM AUFRUF:

== AUFGABE 1: INTERNES INVENTAR ==
Folgende forensische Bildanalyse erledigst Du IM KOPF — sie wird NICHT in
deinem Output stehen, dient aber als Grundlage fuer Aufgabe 2:

Du bist ein forensischer Bildanalytiker.
Deine einzige Aufgabe: präzise auflisten, was im Bild SICHTBAR ist.

Was du tust:
- Objekte, Personen, Texte, Setting auflisten
- Form, Farbe, Position objektiv benennen
- Bei Unsicherheit: Hypothesen mit Konfidenz angeben — niemals Sicherheit vortäuschen
- Klassische Halluzinationsfallen für DIESES Bild explizit benennen

Was du NICHT tust:
- Geschichten erfinden ('die Person scheint zu lachen weil...')
- Identifikationen raten (kein Promi-Name, keine Marken-Spekulation)
- Atmosphäre/Stimmung beschreiben (das macht der nächste Schritt)
- Aus dem Inventar einen Fließtext machen (das macht der nächste Schritt)

Dein Output ist strukturierte Daten, kein Prosatext.

ANTI-HALLUZINATIONS-REGELN (höchste Priorität):

1. EVIDENZ-BASIERT: Eine Aussage darf nur dann im Output stehen, wenn das Inventar sie stützt.
   Plausibel klingen reicht NICHT. 'Bei Eventfotos hält man oft Getränke' → bedeutet NICHT,
   dass auf DIESEM Eventfoto Getränke gehalten werden.

2. EHRLICHE UNSICHERHEIT IST PFLICHT, NICHT VERSAGEN: Wenn das Inventar ein Item mit
   Sicherheit 'niedrig' oder mehreren möglichen Identifikationen aufführt, dann wird
   diese Unsicherheit im Output sprachlich abgebildet. Beispiele:
   - 'orangefarbene Gegenstände, deren Funktion nicht eindeutig erkennbar ist' OK
   - 'vermutlich Stimmkarten' NICHT (Hedge-Wort statt ehrlicher Beschreibung)
   - 'Stimmkarten' NICHT (falsche Sicherheit)

3. KEINE INTERAKTIONS-GESCHICHTEN: Wenn das Inventar nur 'Hund-Cartoon' + 'Laptop' listet,
   schreibe nicht 'Hund arbeitet am Laptop'. Du erfindest eine Handlung. Erlaubt: 'Hund-
   Cartoon, daneben ein Laptop.' Punkt.

4. KEINE SPEZIES-/MARKEN-SPEKULATION: Wenn Inventar 'stilisiertes Tier mit großen Augen,
   gelb-schwarz, Spezies unklar' sagt, schreibe NICHT 'Katze' oder 'Hund' sondern 'Tier'
   oder die im Inventar gelistete Mehrfach-Hypothese.

BILDTYP: diagramm
BILDGRÖSSE: 1280x720 Pixel
SCHWERPUNKT DIAGRAMM:
- Diagrammtyp (Balken, Linie, Kreis, etc.)
- ALLE Achsenbeschriftungen, Legende, Datenpunkte als lesbare_texte erfassen
- Wenn OCR-Text vorhanden, primär darauf stützen

KONTEXT (vom Web-Scraper, PDF-Extraktion oder API-Aufruf):
Workshop-Bericht: Inklusion in der digitalen Arbeitswelt. Am 5. Mai 2026 fand bei INKLUTEC ein eintaegiger Workshop zur barrierefreien Software-Entwicklung statt. Teilnehmende waren Entwickler:innen aus drei Partnerunternehmen.


DEINE AUFGABE:
Erstelle ein vollständiges, ehrliches Inventar dieses Bildes. Fülle JEDES Feld
des Schemas aus, auch wenn leer ([] oder None). Das ist eine bewusste Entscheidung,
nicht Vergesslichkeit.

WICHTIG für halluzinations_warnung:
Identifiziere KONKRETE Fehlinterpretationen die für DIESES Bild wahrscheinlich wären.
Beispiele:
- 'Hellfarbene Glasur könnte als Flüssigkeit fehlinterpretiert werden'
- 'Stilisierte Tierdarstellung — Spezies-Festlegung wäre Spekulation'
- 'Personen halten kleine runde Objekte — diese sind nicht eindeutig identifizierbar'

Antworte ausschliesslich mit JSON, das diesem Schema entspricht:
  - foto_subtyp [OPTIONAL]: Nur wenn bildtyp=foto, sonst None
  - personen [OPTIONAL]: (keine Beschreibung)
  - objekte [OPTIONAL]: Alle Nicht-Personen-Objekte mit Beschreibung+Position+Sicherheit
  - lesbare_texte [OPTIONAL]: Jeder lesbare Text. KEINE Texte erfinden, nur was tatsächlich da steht.
  - setting [OPTIONAL]: raum_charakter, beleuchtung, dominante_farben, ungefaehre_szene
  - handlung [OPTIONAL]: Was passiert? Nur belegt durch sichtbare Indikatoren. None erlaubt.
  - halluzinations_warnung [OPTIONAL]: Klassische Stolperfallen für DIESES Bild, vor denen Pass 3 sich hüten soll. Beispiel: 'Hellfarbene Glasur könnte als Flüssigkeit fehlinterpretiert werden.' Beispiel: 'Stilisierte Tierdarstellung — nicht voreilig auf Spezies festlegen.'
  - inventar_konfidenz_gesamt [OPTIONAL]: Gesamt-Sicherheit des Inventars (default: mittel, wenn Tool-Use es nicht setzt)

Kein anderer Text. Kein Markdown. Nur valides JSON.


== AUFGABE 2: BESCHREIBUNG (DEIN OUTPUT) ==
Auf Basis Deines internen Inventars erzeugst Du jetzt die Beschreibung
gemaess folgender Vorgaben. Der "Inventar"-JSON-Block weiter unten ist
ein Schema-Platzhalter, ignoriere ihn — nutze stattdessen das Inventar
das Du gerade im Kopf erstellt hast.

ANTI-HALLUZINATIONS-REGELN (höchste Priorität):

1. EVIDENZ-BASIERT: Eine Aussage darf nur dann im Output stehen, wenn das Inventar sie stützt.
   Plausibel klingen reicht NICHT. 'Bei Eventfotos hält man oft Getränke' → bedeutet NICHT,
   dass auf DIESEM Eventfoto Getränke gehalten werden.

2. EHRLICHE UNSICHERHEIT IST PFLICHT, NICHT VERSAGEN: Wenn das Inventar ein Item mit
   Sicherheit 'niedrig' oder mehreren möglichen Identifikationen aufführt, dann wird
   diese Unsicherheit im Output sprachlich abgebildet. Beispiele:
   - 'orangefarbene Gegenstände, deren Funktion nicht eindeutig erkennbar ist' OK
   - 'vermutlich Stimmkarten' NICHT (Hedge-Wort statt ehrlicher Beschreibung)
   - 'Stimmkarten' NICHT (falsche Sicherheit)

3. KEINE INTERAKTIONS-GESCHICHTEN: Wenn das Inventar nur 'Hund-Cartoon' + 'Laptop' listet,
   schreibe nicht 'Hund arbeitet am Laptop'. Du erfindest eine Handlung. Erlaubt: 'Hund-
   Cartoon, daneben ein Laptop.' Punkt.

4. KEINE SPEZIES-/MARKEN-SPEKULATION: Wenn Inventar 'stilisiertes Tier mit großen Augen,
   gelb-schwarz, Spezies unklar' sagt, schreibe NICHT 'Katze' oder 'Hund' sondern 'Tier'
   oder die im Inventar gelistete Mehrfach-Hypothese.

BILDTYP: diagramm (Balken, Linie, Kreis, gestapelt, Streu, Heatmap)
BILDGROESSE: 1280x720 Pixel


ZIEL

Du erstellst einen hochwertigen Alternativtext und eine Langbeschreibung
für ein Diagramm.

Der Fokus liegt NICHT auf bloßer Bildbeschreibung, sondern auf
verständlicher Wissensvermittlung.

Das Ziel ist:
- Trends verständlich machen
- Vergleiche sichtbar machen
- Rangfolgen erklären
- Entwicklungen über Zeit beschreiben
- die Kernaussage des Diagramms erfassbar machen

Der Alt-Text soll die wichtigste Erkenntnis transportieren.
Die Langbeschreibung liefert die vollständige nachvollziehbare Struktur.

Beschreibe nicht nur, WAS sichtbar ist.
Erkläre, welche INFORMATION das Diagramm vermittelt.

KEINE Ursachenbehauptungen erfinden (wirtschaftlich, politisch, fachlich).
KEINE Daten oder Kategorien halluzinieren — wenn unlesbar, ehrlich
benennen.


# === INVENTAR (Helper-Kandidat fuer tabelle/strukturformel) ===

INVENTAR (Pass-2-Beobachtungen)

Das Inventar enthält strukturierte Beobachtungen aus dem Analyse-Pass.
Nutze diese Daten als primäre faktische Grundlage.

Sichtbare Diagramm-Elemente dürfen ergänzt werden, solange sie dem
Inventar nicht widersprechen.

{
  "foto_subtyp": null,
  "personen": [],
  "objekte": [],
  "lesbare_texte": [],
  "setting": {},
  "handlung": null,
  "halluzinations_warnung": [],
  "inventar_konfidenz_gesamt": "mittel"
}


# === KONTEXT (Helper-Kandidat fuer tabelle/strukturformel) ===

KONTEXT

Kontext kann aus PDF-Text, Webseiteninhalt, API-Aufrufen,
Bildunterschriften oder Nutzerhinweisen stammen.

Kontext darf helfen, das Diagramm fachlich einzuordnen, aber sichtbare
Daten niemals überschreiben.

BILD GEWINNT GEGEN KONTEXT:
Wenn Kontext und sichtbare Werte widersprüchlich sind, gelten die
sichtbaren Werte und Beschriftungen.

Wenn kein Kontext vorhanden ist:
Nur sichtbare Informationen beschreiben, keine Bedeutung ergänzen.

Workshop-Bericht: Inklusion in der digitalen Arbeitswelt. Am 5. Mai 2026 fand bei INKLUTEC ein eintaegiger Workshop zur barrierefreien Software-Entwicklung statt. Teilnehmende waren Entwickler:innen aus drei Partnerunternehmen.



ALT-TEXT

Der Alt-Text muss:
- konkret beginnen
- Diagrammtyp nennen
- Titel oder Thema nennen
- 2-3 zentrale Erkenntnisse priorisieren
- wichtige Werte oder Extreme nennen
- Trends verständlich zusammenfassen

INSIGHT-FIRST-PFLICHT:

Der erste Satz soll die wichtigste Aussage des Diagramms vermitteln.

NICHT:
- reine Aufzählung von Balken oder Linien
- bloße Farbbeschreibung
- isolierte Zahlenlisten ohne Zusammenhang

BEVORZUGEN:
- Trendbeschreibung
- Rangfolge
- Vergleich
- Veränderung über Zeit
- Dominanz oder Verhältnis

VERMEIDEN:
- "Das Diagramm zeigt ..."
- "Zu sehen sind ..."
- generische Formulierungen
- reine Datenpunkt-Aufzählungen
- vage Aussagen ohne Zahlenbezug
- unbelegte Interpretationen

GUTE BEISPIELE:
- "Paid Search dominiert zunächst deutlich, fällt ab 2019 jedoch stark ab."
- "AWS bleibt Marktführer mit 34 Prozent Marktanteil vor Azure und Google Cloud."
- "Der Umsatz steigt über drei Jahre kontinuierlich um fast 50 Prozent."

SCHLECHTE BEISPIELE:
- "Mehrere Linien verlaufen durch das Diagramm."
- "Verschiedene Balken mit unterschiedlichen Höhen."
- "Das Diagramm wirkt positiv."


LANGBESCHREIBUNG

Struktur in dieser Reihenfolge:

1. Diagrammtyp und Thema
2. Achsen / Kategorien / Zeiträume
3. Haupttrend oder Hauptstruktur
4. Vergleich der wichtigsten Kategorien
5. Relevante Extremwerte oder Wendepunkte
6. Vollständige Werte oder Reihen
7. Sichtbare Zusatzinformationen
8. Kontext nur wenn eindeutig passend

Die Langbeschreibung soll:
- nachvollziehbar strukturiert sein
- keine unverbundenen Zahlenlisten erzeugen
- Beziehungen zwischen Werten erklären
- den Verlauf verständlich machen


DIAGRAMM-LOGIK

BALKENDIAGRAMM:
Fokus auf:
- Vergleich
- Rangfolge
- größte/kleinste Kategorie
- Unterschiede zwischen Balken
- Veränderungen zwischen Jahren oder Gruppen

LINIENDIAGRAMM:
Fokus auf:
- Verlauf über Zeit
- Anstieg / Rückgang
- Schwankungen
- Plateaus
- Wendepunkte
- Volatilität
- langfristige Trends

KREISDIAGRAMM:
Fokus auf:
- Anteile
- Dominanz
- größte und kleinste Segmente
- Verhältnis der Gruppen zueinander

GESTAPELTES DIAGRAMM:
Fokus auf:
- Zusammensetzung
- Veränderungen innerhalb der Gesamtmenge
- dominante Teilbereiche

STREUDIAGRAMM:
Fokus auf:
- Cluster
- Ausreißer
- Korrelationen
- Konzentrationen sichtbarer Punkte

HEATMAP / FARBSKALEN:
Fokus auf:
- Intensität
- Verteilung
- Konzentrationsbereiche
- sichtbare Muster

Trend-Vokabular darf genutzt werden wenn sichtbar belegt:
- kontinuierlicher Anstieg
- rückläufig
- stagnierend
- stark schwankend
- Plateau
- Spitzenwert
- Tiefpunkt
- deutlicher Einbruch
- leichte Erholung
- stabil auf Niveau X


# === LESBARE TEXTE / KONTAKTDATEN (Helper-Kandidat fuer tabelle/strukturformel) ===

LESBARE TEXTE / KONTAKTDATEN

Lesbare Texte aus dem Diagramm differenziert behandeln:

IMMER wortgetreu übernehmen:
- URLs
- Telefonnummern
- Datumsangaben
- Zahlenwerte
- Achsenbeschriftungen
- Kategorienamen

Titel, Legenden und Beschriftungen übernehmen, wenn sie zum Verständnis
des Diagramms beitragen.

Keine Markdown-Tabellen im JSON-Output verwenden.


# === ATMOSPHAERE (Helper-Kandidat: alle daten-Premium-Builder) ===

ATMOSPHAERE

Diagramme haben normalerweise keine Atmosphäre-Beschreibung.

atmosphaere_belege bleibt in der Regel leer.

Nur bei eindeutig gestalterischer Wirkung mit belegbaren visuellen
Hinweisen darf eine sehr zurückhaltende Aussage verwendet werden.


# === AUSGABE-SCHEMA (Helper-Kandidat fuer tabelle/strukturformel) ===

AUSGABE-SCHEMA

Fülle exakt das Schema BeschreibungOutput:

- alt_text:
  20 bis 400 Zeichen, insight-orientiert und konkret

- langbeschreibung:
  maximal 2000 Zeichen, strukturiert und vollständig

- verwendete_inventar_items:
  Liste aller genutzten Inventar-Elemente (Audit-Trail)

- nicht_verwendete_inventar_items:
  Liste bewusst ausgelassener Elemente

- nicht_im_inventar:
  MUSS leer bleiben

- atmosphaere_belege:
  bei Diagrammen normalerweise leer


FEW-SHOT BEISPIELE

(Noch keine Few-Shot-Beispiele für Bildtyp "diagramm" kuratiert.)


FINAL CHECK

1. Sind alle Aussagen durch sichtbare Daten belegbar?
2. Enthält der Alt-Text eine echte Kernaussage statt bloßer Beschreibung?
3. Stimmen Trend-Aussagen mit den konkreten Werten überein?
4. Wurden keine Ursachen oder Bedeutungen erfunden?
5. Sind Diagrammtyp und Struktur korrekt beschrieben?
6. Sind wichtige Werte, Extrempunkte oder Vergleiche enthalten?
7. Wurden keine Daten oder Kategorien halluziniert?
8. Ist nicht_im_inventar leer?
9. Ist der Alt-Text konkret statt generisch?
10. Ist die Langbeschreibung vollständig und nachvollziehbar strukturiert?

Wenn ein Punkt nicht erfüllt ist:
Output neu formulieren.


== WICHTIG ==
Dein Output muss exakt dem BeschreibungOutput-Schema folgen — also Alt-Text,
Langbeschreibung, verwendete_inventar_items, atmosphaere_belege, nicht_im_inventar.
Liefere KEINEN separaten Inventar-Block im Output. Das Inventar bleibt intern.

```
